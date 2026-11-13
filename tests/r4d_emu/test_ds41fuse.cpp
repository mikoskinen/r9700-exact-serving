/* tests/r4d_emu: ds41-fuse's levers on the host SIMT emulator (the GPU twin: r4d_selftest --case rfuse / mixfin):
 *
 *   FUSE-R  router_gemm_f32_scored_topk_scatter (form 2: the router's projection, selection and sort in one launch, the
 *           GEMM's last workgroup ranking) and router_scored_topk_scatter without a gate (form 1) against the stock chain
 *           gemm_nt_f32 -> router_scored_topk (both emulated) -> the stable counting sort written out here (moe_scatter's
 *           contract), byte for byte: logits, ids, weights, sorted_tok, expert_offset, expert_count, each with a guard
 *           tail; DeepSeek's shape (384 experts, top 6, sqrtsoftplus, a 96-expert slice at bases 0 / 96 / 288), bf16 and
 *           f32 projections, random / tied (duplicated weight rows) / NaN rows, at every launch form's height (M 1 / 3 /
 *           8: 8 waves; 9 / 16: 16 waves; 17 / 33 / 48: the 8x1 tile, rows wrapping). Form 2 launched twice (its arrival
 *           counter must come back to zero), and refused past its bounds (M 49, 385 experts, a non-sqrtsoftplus score,
 *           no instance).
 *   FUSE-M  mhc_mix under libr4d's fn4 ds41_mix_fin 1 vs 0 (the one-launch FIN form at <= 48 rows), byte for byte: the
 *           scratch's partials, pre, post and comb at M 1 / 8 / 9 / 48, and M 49 (above the band: the switch must not act);
 *           rad_fuse_stats's FIN counter must move exactly where it acts.
 *
 *   r4d_emu_ds41fuse [quick] */
#include "rad_abi.h"
#include "rad_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" int r4d_rad_gemm_nt_f32(const RadArgs*, RadStream);
extern "C" int r4d_rad_router_scored_topk(const RadArgs*, RadStream);
extern "C" int r4d_rad_router_scored_topk_scatter(const RadArgs*, RadStream);
extern "C" int r4d_rad_router_gemm_scored_scatter(const RadArgs*, RadStream);
extern "C" int r4d_rad_router_gemm_scored_scatter_init(const RadParam*, int, int, int, void**);
extern "C" void r4d_rad_router_gemm_scored_scatter_fini(void*);
extern "C" int r4d_rad_mhc_mix(const RadArgs*, RadStream);
extern "C" int rad_fuse_stats(long long* out, int n);
void r4d_mix_fin_set(int v);

namespace {

int g_cases = 0, g_fail = 0;
void check(bool ok, const std::string& what, const std::string& detail = "") {
    ++g_cases;
    if (!ok) ++g_fail;
    std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", what.c_str(), detail.empty() ? "" : " -- ", detail.c_str());
    std::fflush(stdout);
}
std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
    char b[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}
uint64_t g_rs = 0x2545F4914F6CDD1Dull;
uint64_t rnd() {
    uint64_t z = (g_rs += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
double uni() { return (double)(rnd() >> 11) * (1.0 / 9007199254740992.0); }
double gauss() {
    double u1 = uni(), u2 = uni();
    if (u1 < 1e-300) u1 = 1e-300;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}
uint16_t bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
RadTensor tens(const void* p, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t{};
    t.data = const_cast<void*>(p);
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int64_t st = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        t.shape[i] = shape[i];
        t.stride[i] = st;
        st *= shape[i];
    }
    return t;
}
RadParam pint(const char* k, long long v) { return { k, RAD_P_INT, v, 0, nullptr, 0.0 }; }
RadParam pf64(const char* k, double v) { return { k, RAD_P_F64, 0, 0, nullptr, v }; }
RadParam pstr(const char* k, const char* v) { return { k, RAD_P_STR, 0, 0, v, 0.0 }; }
RadArgs mkargs(std::vector<RadTensor>& t, std::vector<RadParam>& p, void* inst = nullptr) {
    RadArgs a{};
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = p.data();
    a.n_p = (int)p.size();
    a.world_size = 1;
    a.instance = inst;
    return a;
}
template <class V>
long long ndiff(const std::vector<V>& a, const std::vector<V>& b) {
    long long d = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) d += std::memcmp(&a[i], &b[i], sizeof(V)) != 0;
    return d + (long long)(a.size() > b.size() ? a.size() - b.size() : b.size() - a.size());
}

/* moe_scatter's contract (no protected experts): slot i = token * top_k + slot goes to its expert's run in ascending i; an
 * expert outside [base, base + nl) drops the slot; sorted rows past the live count are -1. */
void sort_model(const std::vector<int32_t>& ids, int T, int nl, int base, int32_t* sorted, int32_t* off, int32_t* cnt) {
    std::vector<int32_t> c((size_t)nl, 0), o((size_t)nl + 1, 0), seen((size_t)nl, 0);
    for (int i = 0; i < T; ++i) {
        const int32_t e = ids[(size_t)i] - base;
        if (e >= 0 && e < nl) ++c[(size_t)e];
    }
    for (int e = 0; e < nl; ++e) o[(size_t)e + 1] = o[(size_t)e] + c[(size_t)e];
    for (int i = 0; i < T; ++i) sorted[i] = -1;
    for (int i = 0; i < T; ++i) {
        const int32_t e = ids[(size_t)i] - base;
        if (e >= 0 && e < nl) sorted[o[(size_t)e] + seen[(size_t)e]++] = i;
    }
    for (int e = 0; e < nl; ++e) { off[e] = o[(size_t)e]; cnt[e] = c[(size_t)e]; }
    off[nl] = o[(size_t)nl];
}

constexpr int64_t NE = 384, TK = 6, NL = 96, TAIL = 16;
constexpr int32_t kPoison = -77;

void rf(void* inst, int64_t M, int64_t K, int wbf, int mode, int64_t base) {
    std::vector<float> wf((size_t)(NE * K)), bias((size_t)NE);
    for (auto& v : wf) v = (float)(gauss() * 0.02);
    for (auto& v : bias) v = (float)(gauss() * 0.05);
    if (mode == 1)
        for (int64_t e = 1; e < NE; e += 2) {
            std::memcpy(&wf[(size_t)(e * K)], &wf[(size_t)((e - 1) * K)], (size_t)K * 4);
            bias[(size_t)e] = bias[(size_t)e - 1];
        }
    std::vector<uint16_t> wb((size_t)(NE * K));
    for (size_t i = 0; i < wb.size(); ++i) {
        wb[i] = bf(wf[i]);
        if (wbf) {
            uint32_t u = (uint32_t)wb[i] << 16;
            std::memcpy(&wf[i], &u, 4);
        }
    }
    const void* wp = wbf ? (const void*)wb.data() : (const void*)wf.data();
    std::vector<uint16_t> x((size_t)(M * K));
    for (int64_t r = 0; r < M; ++r) {
        const double rs = std::ldexp(1.0, (int)(rnd() % 5) - 2);
        for (int64_t i = 0; i < K; ++i) x[(size_t)(r * K + i)] = bf((float)(gauss() * rs));
        if (mode == 2 && (r % 3) == 1) x[(size_t)(r * K + (int64_t)(rnd() % K))] = bf(NAN);
    }
    struct Out {
        std::vector<float> lg, w;
        std::vector<int32_t> ids, so, off, cnt;
        void init(int64_t m) {
            lg.assign((size_t)(m * NE + TAIL), -3.0f);
            w.assign((size_t)(m * TK + TAIL), -3.0f);
            ids.assign((size_t)(m * TK + TAIL), kPoison);
            so.assign((size_t)(m * TK + TAIL), kPoison);
            off.assign((size_t)(NL + 1 + TAIL), kPoison);
            cnt.assign((size_t)(NL + TAIL), kPoison);
        }
    };
    Out o[3];
    for (auto& q : o) q.init(M);
    const uint32_t WD = wbf ? RAD_BF16 : RAD_F32;
    /* the chain: gemm_nt_f32, router_scored_topk, the sort model */
    std::vector<RadParam> pg = { pint("M", M), pint("N", NE), pint("K", K), pstr("dtype", "bf16") };
    std::vector<RadParam> psel = { pint("M", M), pint("n_expert", NE), pint("top_k", TK), pint("norm", 1),
                                   pf64("route_scale", 1.5), pstr("score", "sqrtsoftplus"), pstr("dtype", "f32") };
    std::vector<RadTensor> tg0 = { tens(x.data(), RAD_BF16, { M, K }), tens(wp, WD, { NE, K }), tens(o[0].lg.data(), RAD_F32, { M, NE }) };
    std::vector<RadTensor> ts0 = { tens(o[0].lg.data(), RAD_F32, { M, NE }), tens(bias.data(), RAD_F32, { NE }),
                                   tens(o[0].ids.data(), RAD_I32, { M, TK }), tens(o[0].w.data(), RAD_F32, { M, TK }) };
    RadArgs ag0 = mkargs(tg0, pg), as0 = mkargs(ts0, psel);
    const int r0a = r4d_rad_gemm_nt_f32(&ag0, nullptr), r0b = r4d_rad_router_scored_topk(&as0, nullptr);
    sort_model(o[0].ids, (int)(M * TK), (int)NL, (int)base, o[0].so.data(), o[0].off.data(), o[0].cnt.data());
    /* form 1: gemm_nt_f32, router_scored_topk_scatter (no gate) */
    std::vector<RadParam> p1 = { pint("M", M), pint("n_expert", NE), pint("n_local", NL), pint("top_k", TK), pint("norm", 1),
                                 pf64("route_scale", 1.5), pstr("score", "sqrtsoftplus"), pstr("dtype", "f32"),
                                 pint("expert_base", base) };
    std::vector<RadTensor> tg1 = { tens(x.data(), RAD_BF16, { M, K }), tens(wp, WD, { NE, K }), tens(o[1].lg.data(), RAD_F32, { M, NE }) };
    std::vector<RadTensor> t1 = { tens(o[1].lg.data(), RAD_F32, { M, NE }), tens(bias.data(), RAD_F32, { NE }),
                                  tens(o[1].ids.data(), RAD_I32, { M, TK }), tens(o[1].w.data(), RAD_F32, { M, TK }),
                                  tens(o[1].so.data(), RAD_I32, { M * TK }), tens(o[1].off.data(), RAD_I32, { NL + 1 }),
                                  tens(o[1].cnt.data(), RAD_I32, { NL }) };
    RadArgs ag1 = mkargs(tg1, pg), a1 = mkargs(t1, p1);
    const int r1a = r4d_rad_gemm_nt_f32(&ag1, nullptr), r1b = r4d_rad_router_scored_topk_scatter(&a1, nullptr);
    /* form 2: one launch, twice */
    std::vector<RadParam> p2 = { pint("M", M), pint("K", K), pint("n_expert", NE), pint("n_local", NL), pint("top_k", TK),
                                 pint("norm", 1), pf64("route_scale", 1.5), pstr("score", "sqrtsoftplus"), pstr("dtype", "bf16"),
                                 pint("expert_base", base) };
    std::vector<RadTensor> t2 = { tens(x.data(), RAD_BF16, { M, K }), tens(wp, WD, { NE, K }), tens(bias.data(), RAD_F32, { NE }),
                                  tens(o[2].lg.data(), RAD_F32, { M, NE }), tens(o[2].ids.data(), RAD_I32, { M, TK }),
                                  tens(o[2].w.data(), RAD_F32, { M, TK }), tens(o[2].so.data(), RAD_I32, { M * TK }),
                                  tens(o[2].off.data(), RAD_I32, { NL + 1 }), tens(o[2].cnt.data(), RAD_I32, { NL }) };
    RadArgs a2 = mkargs(t2, p2, inst);
    const int r2a = r4d_rad_router_gemm_scored_scatter(&a2, nullptr);
    const int r2b = r4d_rad_router_gemm_scored_scatter(&a2, nullptr);
    long long d1 = 0, d2 = 0;
    d1 += ndiff(o[0].lg, o[1].lg) + ndiff(o[0].ids, o[1].ids) + ndiff(o[0].w, o[1].w) + ndiff(o[0].so, o[1].so) +
          ndiff(o[0].off, o[1].off) + ndiff(o[0].cnt, o[1].cnt);
    const long long dl = ndiff(o[0].lg, o[2].lg), di = ndiff(o[0].ids, o[2].ids), dw = ndiff(o[0].w, o[2].w),
                    dso = ndiff(o[0].so, o[2].so), doff = ndiff(o[0].off, o[2].off), dc = ndiff(o[0].cnt, o[2].cnt);
    d2 = dl + di + dw + dso + doff + dc;
    const bool rcs = r0a == RAD_OK && r0b == RAD_OK && r1a == RAD_OK && r1b == RAD_OK && r2a == RAD_OK && r2b == RAD_OK;
    check(rcs && d1 == 0 && d2 == 0,
          fmt("FUSE-R %s weights, %s, M=%lld K=%lld base %lld: form 1 and form 2 (twice) byte for byte vs the chain",
              wbf ? "bf16" : "f32", mode == 0 ? "random" : mode == 1 ? "tied" : "NaN rows", (long long)M, (long long)K,
              (long long)base),
          fmt("rc %d %d / %d %d / %d %d; form 1 differing words %lld; form 2 logits %lld ids %lld weights %lld sorted %lld "
              "offsets %lld counts %lld",
              r0a, r0b, r1a, r1b, r2a, r2b, d1, dl, di, dw, dso, doff, dc));
}

void rf_refusals(void* inst) {
    const int64_t K = 256;
    std::vector<uint16_t> x((size_t)(49 * K), 0), wb((size_t)(385 * K), 0);
    std::vector<float> bias(385, 0.0f), lg((size_t)(49 * 385), 0.0f), w((size_t)(49 * TK), 0.0f);
    std::vector<int32_t> ids((size_t)(49 * TK)), so((size_t)(49 * TK)), off(NL + 1), cnt(NL);
    auto run = [&](int64_t M, int64_t ne, const char* score, void* in) {
        std::vector<RadParam> p = { pint("M", M), pint("K", K), pint("n_expert", ne), pint("n_local", NL), pint("top_k", TK),
                                    pint("norm", 1), pf64("route_scale", 1.5), pstr("score", score), pstr("dtype", "bf16") };
        std::vector<RadTensor> t = { tens(x.data(), RAD_BF16, { M, K }), tens(wb.data(), RAD_BF16, { ne, K }),
                                     tens(bias.data(), RAD_F32, { ne }), tens(lg.data(), RAD_F32, { M, ne }),
                                     tens(ids.data(), RAD_I32, { M, TK }), tens(w.data(), RAD_F32, { M, TK }),
                                     tens(so.data(), RAD_I32, { M * TK }), tens(off.data(), RAD_I32, { NL + 1 }),
                                     tens(cnt.data(), RAD_I32, { NL }) };
        RadArgs a = mkargs(t, p, in);
        return r4d_rad_router_gemm_scored_scatter(&a, nullptr);
    };
    check(run(49, NE, "sqrtsoftplus", inst) != RAD_OK, "FUSE-R form 2 refuses M 49 (past the decode band)");
    check(run(8, 385, "sqrtsoftplus", inst) != RAD_OK, "FUSE-R form 2 refuses 385 experts");
    check(run(8, NE, "sigmoid", inst) != RAD_OK, "FUSE-R form 2 refuses a score other than sqrtsoftplus");
    check(run(8, NE, "sqrtsoftplus", nullptr) != RAD_OK, "FUSE-R form 2 refuses a launch without its instance");
}

void mf(int64_t M) {
    constexpr int64_t n = 5120, hc = 4, W = hc * n, MIX = hc * (hc + 2), S = MIX + 1, iters = 20;
    const int64_t chunks = (W + 1023) / 1024;
    std::vector<float> fn((size_t)(MIX * W)), base(MIX), scale(3);
    for (auto& x : fn) x = (float)(gauss() * 0.01);
    for (auto& x : base) x = (float)(gauss() * 0.5);
    for (auto& x : scale) x = (float)(gauss() * 0.1 + 0.5);
    std::vector<uint16_t> h((size_t)(M * W));
    for (int64_t r = 0; r < M; ++r) {
        const double rs = std::ldexp(1.0, (int)(rnd() % 5) - 2);
        for (int64_t i = 0; i < W; ++i) h[(size_t)(r * W + i)] = bf((float)(gauss() * rs));
    }
    struct Out {
        std::vector<float> pre, post, comb, scr;
    };
    std::vector<RadParam> ps = { pint("M", M), pint("n", n), pint("hc", hc), pint("iterations", iters), pf64("eps", 1e-6),
                                 pf64("hc_eps", 1e-6), pstr("dtype", "bf16") };
    auto go = [&](Out& o, int fin) {
        o.pre.assign((size_t)(M * hc), -7.0f);
        o.post.assign((size_t)(M * hc), -7.0f);
        o.comb.assign((size_t)(M * hc * hc), -7.0f);
        o.scr.assign((size_t)(M * S * chunks + 1024), -9.0f);
        std::vector<RadTensor> t = { tens(h.data(), RAD_BF16, { M, W }), tens(fn.data(), RAD_F32, { MIX, W }),
                                     tens(scale.data(), RAD_F32, { 3 }), tens(base.data(), RAD_F32, { MIX }),
                                     tens(o.pre.data(), RAD_F32, { M, hc }), tens(o.post.data(), RAD_F32, { M, hc }),
                                     tens(o.comb.data(), RAD_F32, { M, hc * hc }) };
        RadArgs a{};
        a.t = t.data();
        a.n_t = (int)t.size();
        a.p = ps.data();
        a.n_p = (int)ps.size();
        a.world_size = 1;
        a.scratch = o.scr.data();
        a.scratch_bytes = (int64_t)o.scr.size() * 4;
        r4d_mix_fin_set(fin);
        return r4d_rad_mhc_mix(&a, nullptr);
    };
    Out R, F;
    const int rc0 = go(R, 0);
    long long s0[4] = { 0, 0, 0, 0 }, s1[4] = { 0, 0, 0, 0 };
    rad_fuse_stats(s0, 4);
    const int rc1 = go(F, 1);
    rad_fuse_stats(s1, 4);
    r4d_mix_fin_set(0);
    const bool took = s1[0] > s0[0];
    const long long dp = ndiff(R.scr, F.scr), dpre = ndiff(R.pre, F.pre), dpo = ndiff(R.post, F.post), dcb = ndiff(R.comb, F.comb);
    check(rc0 == RAD_OK && rc1 == RAD_OK && dp + dpre + dpo + dcb == 0 && took == (M <= 48),
          fmt("FUSE-M mhc_mix ds41_mix_fin 1 vs 0, M=%lld (partials, pre, post, comb)", (long long)M),
          fmt("rc %d / %d; differing words: partials %lld pre %lld post %lld comb %lld; FIN launches +%lld (want %s)", rc0, rc1,
              dp, dpre, dpo, dcb, s1[0] - s0[0], M <= 48 ? "> 0" : "0"));
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && !std::strcmp(argv[1], "quick");
    void* inst = nullptr;
    const int irc = r4d_rad_router_gemm_scored_scatter_init(nullptr, 0, 0, 1, &inst);
    check(irc == RAD_OK && inst, "FUSE-R form 2 init takes an instance");
    if (inst) {
        for (int wbf : { 1, 0 })
            for (int mode : { 0, 1, 2 }) {
                const std::vector<int64_t> ms = quick ? std::vector<int64_t>{ 1, 9, 17 }
                                                      : std::vector<int64_t>{ 1, 3, 8, 9, 16, 17, 33, 48 };
                for (int64_t M : ms) {
                    const int64_t K = M <= 8 ? 5120 : 1056;   /* 1056: past whole groups of 16 steps (the tail loop) */
                    for (int64_t base : { (int64_t)0, (int64_t)96, (int64_t)288 }) {
                        if (quick && base == 96) continue;
                        rf(inst, M, K, wbf, mode, base);
                    }
                }
            }
        rf_refusals(inst);
        r4d_rad_router_gemm_scored_scatter_fini(inst);
    }
    for (int64_t M : quick ? std::vector<int64_t>{ 1, 49 } : std::vector<int64_t>{ 1, 8, 9, 48, 49 }) mf(M);
    std::printf("%d cases, %d failed\n", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
