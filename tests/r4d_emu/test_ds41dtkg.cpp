/* tests/r4d_emu: lever DTKG (DRAFTER track round 110, fn4 drafter_mx + 100 KE + 1000 G; DECODE review round 71 = tp8 loop QUEUE W40) on
 * the host SIMT emulator -- the host twin of r4d_selftest SPD_PART=dtk (which ran on a card, xeon GPU1): the DSpark drafter's router (128
 * experts, top 3) selecting its top KE IN PROCESS must give the bytes of a router DECLARED top-KE, slot for slot, with the dropped slots
 * written as id -1 / weight 0 and left out of the sort:
 *   (a) router_gemm_f32_scored_topk_scatter (the fused launch the DSpark router serves) at every ds41_f32dec form it reaches -- 0 (stock),
 *       1 .. 20, 21 (RGK: the keyed row selection, BEST STK's ds41_f32dec 9) and 22 (RGD: the WMMA logits at 128 experts under value 12,
 *       BEST STK12's), each launched twice: logits, ids / weights (slots 0 .. KE-1 = the declared call's, KE .. 2 = -1 / 0), expert_offset,
 *       expert_count, sorted_tok (the declared call's slot r KE + j renamed r 3 + j; -1 past the live rows) and the inverse (-1 for a dropped
 *       slot), with guard tails, every expert local and a 32-expert slice at base 32, KE 2 and 1, M 1 .. 64;
 *   (b) router_scored_topk (the three-launch chain's selection) and router_scored_topk_scatter (selection + sort, no GEMM) on given
 *       logits: the same rule;
 *   (c) the row gate: 1200 (<= 8 rows) / 3200 (<= 24 rows) are top 2 at or under the bound and today's top-3 bytes above it;
 *   (d) the trunk's router (384 experts, top 6) and a 128-expert call whose KE is not below k (top 2 under 200) keep today's bytes under
 *       the switch (r4d_dtk_pack's n_expert / KE gates);
 *   (e) the switch's own functions: r4d_dtk_known / _set / _get / _pack (rad_fn4_* is r4d_ds41_ggml.hip's, not in this build).
 *
 *   r4d_emu_ds41dtkg [quick] */
#include "rad_abi.h"
#include "rad_plugin.h"

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" int r4d_rad_router_gemm_scored_scatter(const RadArgs*, RadStream);
extern "C" int r4d_rad_router_gemm_scored_scatter_init(const RadParam*, int, int, int, void**);
extern "C" void r4d_rad_router_gemm_scored_scatter_fini(void*);
extern "C" int r4d_rad_router_scored_topk(const RadArgs*, RadStream);
extern "C" int r4d_rad_router_scored_topk_scatter(const RadArgs*, RadStream);
extern "C" void r4d_f32dec_pin(int rgs_form, int gemm_form);
int r4d_f32dec_set(int v);
int r4d_dtk_known(int code);
void r4d_dtk_set(int code);
int r4d_dtk_get();
int r4d_dtk_pack(int ne, int k, int m);

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
uint64_t g_rs = 0x9E3779B97F4A7C15ull;
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

constexpr int64_t TAIL = 16;

struct Out {
    std::vector<float> lg, w;
    std::vector<int32_t> ids, so, off, cnt, inv;
    void init(int64_t m, int64_t ne, int64_t k, int64_t nl) {
        lg.assign((size_t)(m * ne + TAIL), -3.0f);
        w.assign((size_t)(m * k + TAIL), -3.0f);
        ids.assign((size_t)(m * k + TAIL), -77);
        so.assign((size_t)(m * k + TAIL), -77);
        off.assign((size_t)(nl + 1 + TAIL), -77);
        cnt.assign((size_t)(nl + TAIL), -77);
        inv.assign((size_t)(m * k + TAIL), -77);
    }
};

/* The kernel inputs of one router: bf16 rows x, the projection (bf16 or f32), the bias; logits for the GEMM-less entries. */
struct Router {
    int64_t ne, M, K;
    std::vector<float> wf, bias, lgin;
    std::vector<uint16_t> wb, x;
    Router(int64_t ne_, int64_t M_, int64_t K_, int mode) : ne(ne_), M(M_), K(K_) {
        wf.resize((size_t)(ne * K));
        bias.resize((size_t)ne);
        for (auto& v : wf) v = (float)(gauss() * 0.02);
        for (auto& v : bias) v = (float)(gauss() * 0.05);
        if (mode == 1)   /* tied pairs: the rank key's expert-index tie break decides */
            for (int64_t e = 1; e < ne; e += 2) {
                std::memcpy(&wf[(size_t)(e * K)], &wf[(size_t)((e - 1) * K)], (size_t)K * 4);
                bias[(size_t)e] = bias[(size_t)e - 1];
            }
        wb.resize(wf.size());
        for (size_t i = 0; i < wb.size(); ++i) {
            wb[i] = bf(wf[i]);
            uint32_t u = (uint32_t)wb[i] << 16;
            std::memcpy(&wf[i], &u, 4);   /* the f32 projection carries the bf16 values: one logit function either dtype */
        }
        x.resize((size_t)(M * K));
        for (int64_t r = 0; r < M; ++r) {
            const double rs = std::ldexp(1.0, (int)(rnd() % 5) - 2);
            for (int64_t i = 0; i < K; ++i) x[(size_t)(r * K + i)] = bf((float)(gauss() * rs));
        }
        lgin.resize((size_t)(M * ne));
        for (auto& v : lgin) v = (float)(gauss() * 1.5);
        if (mode == 1)
            for (int64_t r = 0; r < M; ++r)
                for (int64_t e = 1; e < ne; e += 2) lgin[(size_t)(r * ne + e)] = lgin[(size_t)(r * ne + e - 1)];
    }
};

/* entry 0: router_gemm_f32_scored_topk_scatter (pinned form, twice); 1: router_scored_topk; 2: router_scored_topk_scatter */
int run(void* inst, const Router& R, int entry, int form, bool wbf, int64_t k, int64_t nl, int64_t base, int code, Out& o) {
    const int64_t M = R.M, ne = R.ne;
    o.init(M, ne, k, nl);
    r4d_dtk_set(code);
    int rc = RAD_OK;
    if (entry == 0) {
        std::vector<RadParam> p = { pint("M", M), pint("K", R.K), pint("n_expert", ne), pint("n_local", nl), pint("top_k", k),
                                    pint("norm", 1), pf64("route_scale", 1.5), pstr("score", "sqrtsoftplus"), pstr("dtype", "bf16"),
                                    pint("expert_base", base) };
        const void* wp = wbf ? (const void*)R.wb.data() : (const void*)R.wf.data();
        std::vector<RadTensor> t = { tens(R.x.data(), RAD_BF16, { M, R.K }), tens(wp, wbf ? RAD_BF16 : RAD_F32, { ne, R.K }),
                                     tens(R.bias.data(), RAD_F32, { ne }), tens(o.lg.data(), RAD_F32, { M, ne }),
                                     tens(o.ids.data(), RAD_I32, { M, k }), tens(o.w.data(), RAD_F32, { M, k }),
                                     tens(o.so.data(), RAD_I32, { M * k }), tens(o.off.data(), RAD_I32, { nl + 1 }),
                                     tens(o.cnt.data(), RAD_I32, { nl }), tens(o.inv.data(), RAD_I32, { M * k }) };
        RadArgs a = mkargs(t, p, inst);
        r4d_f32dec_pin(form, -1);
        const int ra = r4d_rad_router_gemm_scored_scatter(&a, nullptr);
        const int rb = r4d_rad_router_gemm_scored_scatter(&a, nullptr);
        r4d_f32dec_pin(-1, -1);
        rc = ra != RAD_OK ? ra : rb;
    } else if (entry == 1) {
        std::vector<RadParam> p = { pint("n_expert", ne), pint("top_k", k), pint("norm", 1), pf64("route_scale", 1.5),
                                    pstr("score", "sqrtsoftplus") };
        std::vector<RadTensor> t = { tens(R.lgin.data(), RAD_F32, { M, ne }), tens(R.bias.data(), RAD_F32, { ne }),
                                     tens(o.ids.data(), RAD_I32, { M, k }), tens(o.w.data(), RAD_F32, { M, k }) };
        RadArgs a = mkargs(t, p);
        rc = r4d_rad_router_scored_topk(&a, nullptr);
    } else {
        std::vector<RadParam> p = { pint("n_expert", ne), pint("n_local", nl), pint("top_k", k), pint("norm", 1),
                                    pf64("route_scale", 1.5), pstr("score", "sqrtsoftplus"), pint("expert_base", base) };
        RadTensor none{};
        std::vector<RadTensor> t = { tens(R.lgin.data(), RAD_F32, { M, ne }), tens(R.bias.data(), RAD_F32, { ne }),
                                     tens(o.ids.data(), RAD_I32, { M, k }), tens(o.w.data(), RAD_F32, { M, k }),
                                     tens(o.so.data(), RAD_I32, { M * k }), tens(o.off.data(), RAD_I32, { nl + 1 }),
                                     tens(o.cnt.data(), RAD_I32, { nl }), none, tens(o.inv.data(), RAD_I32, { M * k }) };
        RadArgs a = mkargs(t, p);
        rc = r4d_rad_router_scored_topk_scatter(&a, nullptr);
    }
    r4d_dtk_set(0);
    return rc;
}

/* d: the top-k call under the switch (KE kept); e: a call DECLARED top-KE, the switch off. Words that differ from the rule. */
long long rule_diff(const Out& d, const Out& e, int64_t M, int64_t ne, int64_t k, int64_t ke, int64_t nl, bool sorted, std::string* why) {
    long long n = 0;
    auto bad = [&](const char* what, long long i) {
        if (!n && why) *why = fmt("first: %s at %lld", what, i);
        ++n;
    };
    for (int64_t i = 0; i < M * ne + TAIL; ++i)
        if (std::memcmp(&d.lg[(size_t)i], &e.lg[(size_t)i], 4)) bad("logit", i);
    for (int64_t r = 0; r < M; ++r)
        for (int64_t j = 0; j < k; ++j) {
            const size_t di = (size_t)(r * k + j);
            if (j < ke) {
                const size_t ei = (size_t)(r * ke + j);
                if (d.ids[di] != e.ids[ei]) bad("id", (long long)di);
                if (std::memcmp(&d.w[di], &e.w[ei], 4)) bad("weight", (long long)di);
                if (sorted && (d.inv[di] < 0) != (e.inv[ei] < 0)) bad("inverse (live)", (long long)di);
                if (sorted && d.inv[di] != e.inv[ei]) bad("inverse", (long long)di);
            } else {
                uint32_t wu;
                std::memcpy(&wu, &d.w[di], 4);
                if (d.ids[di] != -1) bad("dropped id", (long long)di);
                if (wu != 0u) bad("dropped weight", (long long)di);
                if (sorted && d.inv[di] != -1) bad("dropped inverse", (long long)di);
            }
        }
    for (int64_t i = M * k; i < M * k + TAIL; ++i) {   /* guard tails untouched */
        if (d.ids[(size_t)i] != -77) bad("ids tail", i);
        if (d.w[(size_t)i] != -3.0f) bad("weights tail", i);
        if (sorted && d.so[(size_t)i] != -77) bad("sorted tail", i);
        if (sorted && d.inv[(size_t)i] != -77) bad("inverse tail", i);
    }
    if (sorted) {
        for (int64_t x = 0; x < nl + 1 + TAIL; ++x)
            if (d.off[(size_t)x] != e.off[(size_t)x]) bad("expert_offset", x);
        for (int64_t x = 0; x < nl + TAIL; ++x)
            if (d.cnt[(size_t)x] != e.cnt[(size_t)x]) bad("expert_count", x);
        for (int64_t p = 0; p < M * k; ++p) {
            int32_t want = -1;
            if (p < M * ke) {
                const int32_t v = e.so[(size_t)p];
                want = v < 0 ? v : (int32_t)((v / ke) * k + v % ke);   /* slot r KE + j of the declared call = slot r k + j */
            }
            if (d.so[(size_t)p] != want) bad("sorted_tok", p);
        }
    }
    return n;
}

void case_rule(void* inst, int entry, int64_t M, int64_t K, bool wbf, int mode, int64_t ke, int64_t nl, int64_t base, int code,
               const std::vector<int>& forms) {
    const int64_t NE = 128, TK = 3;
    const Router R(NE, M, K, mode);
    std::string bad;
    for (int f : forms) {
        Out d, e;
        const int rd = run(inst, R, entry, f, wbf, TK, nl, base, code, d);
        const int re = run(inst, R, entry, f, wbf, ke, nl, base, 0, e);
        std::string why;
        const long long n = rd == RAD_OK && re == RAD_OK ? rule_diff(d, e, M, NE, TK, ke, nl, entry != 1, &why) : -1;
        if (n) bad += fmt(" form %d: rc %d / %d, %lld words (%s);", f, rd, re, n, why.c_str());
    }
    static const char* en[] = { "router_gemm_f32_scored_topk_scatter", "router_scored_topk", "router_scored_topk_scatter" };
    check(bad.empty(),
          fmt("DTKG %s, 128 experts top 3, drafter_mx + %d = a DECLARED top-%lld call: %s weights, %s, M=%lld K=%lld, %lld local at %lld, "
              "%zu form(s)",
              en[entry], code, (long long)ke, wbf ? "bf16" : "f32", mode ? "tied" : "random", (long long)M, (long long)K, (long long)nl,
              (long long)base, forms.size()),
          bad);
}

/* the switch on but not applying (n_expert, KE >= k, rows above the gate): today's bytes, every output word and tail */
void case_off(void* inst, int entry, int64_t ne, int64_t k, int64_t M, int64_t K, int code, const std::vector<int>& forms,
              const char* why_off) {
    const Router R(ne, M, K, 0);
    std::string bad;
    for (int f : forms) {
        Out d, e;
        const int rd = run(inst, R, entry, f, true, k, ne, 0, code, d);
        const int re = run(inst, R, entry, f, true, k, ne, 0, 0, e);
        const long long n = ndiff(d.lg, e.lg) + ndiff(d.ids, e.ids) + ndiff(d.w, e.w) + ndiff(d.so, e.so) + ndiff(d.off, e.off) +
                            ndiff(d.cnt, e.cnt) + ndiff(d.inv, e.inv);
        if (rd != RAD_OK || re != RAD_OK || n) bad += fmt(" form %d: rc %d / %d, %lld words;", f, rd, re, n);
    }
    check(bad.empty(),
          fmt("DTKG off by its gate (%s): %lld experts top %lld, M=%lld, drafter_mx + %d = the switch off byte for byte, %zu form(s)",
              why_off, (long long)ne, (long long)k, (long long)M, code, forms.size()),
          bad);
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && !std::strcmp(argv[1], "quick");
    /* (e) the switch's functions */
    {
        std::string bad;
        for (int c : { 0, 100, 200, 1100, 1200, 3200, 4200, 8100, 8200 })
            if (r4d_dtk_known(c) != 1) bad += fmt(" %d refused;", c);
        for (int c : { 300, 900, 9200, 9100, 250, 201, 1000, -100, 10200 })
            if (r4d_dtk_known(c) != 0) bad += fmt(" %d taken;", c);
        r4d_dtk_set(3200);
        const int g = r4d_dtk_get();
        const int p24 = r4d_dtk_pack(128, 3, 24), p25 = r4d_dtk_pack(128, 3, 25), p1 = r4d_dtk_pack(128, 3, 1),
                  t384 = r4d_dtk_pack(384, 6, 8), k2 = r4d_dtk_pack(128, 2, 8);
        r4d_dtk_set(200);
        const int any = r4d_dtk_pack(128, 3, 4096);
        r4d_dtk_set(300);   /* unknown: stored as 0 (rad_fn4_accepts refuses it before) */
        const int g300 = r4d_dtk_get(), p300 = r4d_dtk_pack(128, 3, 8);
        r4d_dtk_set(0);
        if (g != 3200) bad += fmt(" get %d after 3200;", g);
        if (p24 != (3 | 2 << 16) || p1 != (3 | 2 << 16) || p25 != 3) bad += fmt(" 3200 packs 24 / 1 / 25 rows %#x / %#x / %#x;", p24, p1, p25);
        if (t384 != 6 || k2 != 2) bad += fmt(" 3200 at 384 / top 2: %d / %d;", t384, k2);
        if (any != (3 | 2 << 16)) bad += fmt(" 200 at 4096 rows %#x;", any);
        if (g300 != 0 || p300 != 3) bad += fmt(" 300 stored %d, packs %d;", g300, p300);
        check(bad.empty(), "DTKG switch: r4d_dtk_known takes 0 / 100 KE + 1000 G (KE 1..2, G 0..8), refuses the rest; pack gates on 128 experts, KE < k, "
                           "rows <= 8 G; an unknown code stores 0", bad);
    }
    void* inst = nullptr;
    const int irc = r4d_rad_router_gemm_scored_scatter_init(nullptr, 0, 0, 1, &inst);
    check(irc == RAD_OK && inst, "router_gemm_f32_scored_topk_scatter init takes an instance");
    if (!inst) {
        std::printf("%d cases, %d failed\n", g_cases, g_fail);
        return 1;
    }
    std::vector<int> all;
    for (int f = 0; f <= 21; ++f) all.push_back(f);   /* 0 = the stock form; 22 below (needs K 5120 + ds41_f32dec 12) */
    const std::vector<int> few = { 0, 3, 19, 21 };
    const std::vector<int64_t> ms = quick ? std::vector<int64_t>{ 8, 24 } : std::vector<int64_t>{ 1, 8, 9, 16, 24, 32, 48, 64 };
    /* (a) the fused launch, every form, every local and a slice; KE 2 (the arm's) and KE 1 */
    for (int64_t M : ms) {
        case_rule(inst, 0, M, 1280, true, 0, 2, 128, 0, 200, M == 24 || M == 64 ? all : few);
        case_rule(inst, 0, M, 1280, true, 0, 2, 32, 32, 200, few);
        if (!quick) {
            case_rule(inst, 0, M, 1280, false, 0, 2, 128, 0, 200, few);
            case_rule(inst, 0, M, 1280, true, 1, 2, 128, 0, 200, few);
            case_rule(inst, 0, M, 1280, true, 0, 1, 128, 0, 100, few);
        }
    }
    /* RGD: form 22 at 128 experts under ds41_f32dec 12 (the WMMA logits; BEST STK12's), the trunk router's K 5120, bf16 */
    r4d_f32dec_set(12);
    for (int64_t M : quick ? std::vector<int64_t>{ 24 } : std::vector<int64_t>{ 8, 24, 33 }) {
        case_rule(inst, 0, M, 5120, true, 0, 2, 128, 0, 200, { 22 });
        case_rule(inst, 0, M, 5120, true, 0, 2, 32, 32, 200, { 22 });
    }
    r4d_f32dec_set(0);
    /* (b) the chain's selection and the GEMM-less selection + sort */
    for (int64_t M : ms) {
        case_rule(inst, 1, M, 1280, true, 0, 2, 128, 0, 200, { 0 });
        case_rule(inst, 2, M, 1280, true, 0, 2, 128, 0, 200, { 0 });
        case_rule(inst, 2, M, 1280, true, 0, 2, 32, 96, 200, { 0 });
        if (!quick) {
            case_rule(inst, 1, M, 1280, true, 1, 1, 128, 0, 100, { 0 });
            case_rule(inst, 2, M, 1280, true, 1, 2, 128, 0, 200, { 0 });
        }
    }
    /* (c) the row gate: at or under the bound = the rule, above it = off */
    case_rule(inst, 0, 8, 1280, true, 0, 2, 128, 0, 1200, few);
    case_off(inst, 0, 128, 3, 9, 1280, 1200, few, "9 rows > 8");
    case_rule(inst, 0, 24, 1280, true, 0, 2, 128, 0, 3200, few);
    case_off(inst, 0, 128, 3, 25, 1280, 3200, few, "25 rows > 24");
    case_off(inst, 2, 128, 3, 33, 1280, 4200, { 0 }, "33 rows > 32");
    case_rule(inst, 2, 32, 1280, true, 0, 2, 128, 0, 4200, { 0 });
    case_rule(inst, 1, 64, 1280, true, 0, 2, 128, 0, 8200, { 0 });
    case_off(inst, 1, 128, 3, 65, 1280, 8200, { 0 }, "65 rows > 64");
    /* (d) the trunk router and a top-2 call keep today's bytes under the switch */
    case_off(inst, 0, 384, 6, 24, 1280, 200, few, "384 experts: the trunk");
    case_off(inst, 2, 384, 6, 9, 1280, 200, { 0 }, "384 experts: the trunk");
    case_off(inst, 0, 128, 2, 9, 1280, 200, few, "KE 2 not below top 2");
    r4d_f32dec_set(12);
    case_off(inst, 0, 384, 6, 24, 5120, 200, { 22 }, "384 experts at RGM");
    r4d_f32dec_set(0);
    r4d_rad_router_gemm_scored_scatter_fini(inst);
    std::printf("%d cases, %d failed\n", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
