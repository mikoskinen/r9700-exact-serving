/* tests/r4d_emu: lever SPD-F (ds41-etp-spd, 15.11.2026) on the host SIMT emulator (the GPU twin: r4d_selftest --case spdf32):
 * every decode-band form of the two f32 projections that libr4d's fn4 ds41_f32dec reaches, pinned by r4d_f32dec_pin, byte for
 * byte against the stock form at that height --
 *   router_gemm_f32_scored_topk_scatter (forms 1..14: R x C a wave, U steps, NW waves a workgroup; the tail's rows wrap over
 *       NW waves): logits, ids, weights, sorted_tok, expert_offset, expert_count with guard tails, 384 experts, top 6,
 *       sqrtsoftplus, every expert local (the expert split's layout) and a 96-expert slice at base 96, bf16 and f32 weights,
 *       random / tied / NaN rows, each form launched twice (the arrival counter comes back to zero);
 *   gemm_nt_f32 (forms 1..12 tile_wave, 20..31 the tiled forms): y with a guard tail, N 264 (no tile width divides it: the
 *       column clamp) and 1024, K 1280 (U 16's tail loop runs; whole chunks of 256 for the tiled forms) and 1056 (the tiled
 *       forms decline to the stock rule);
 * at M 1 / 9 / 17 / 33 / 48 (quick: 9 / 33), and the switch itself (r4d_f32dec_set / _get; rad_fn4_* is r4d_ds41_ggml.hip's, not in this build): 0 / 1 / 3 applied, 4 refused; under 0 no call takes a
 * non-stock form (rad_f32dec_stats).
 *
 *   r4d_emu_ds41spd [quick] */
#include "rad_abi.h"
#include "rad_plugin.h"

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" int r4d_rad_gemm_nt_f32(const RadArgs*, RadStream);
extern "C" int r4d_rad_router_gemm_scored_scatter(const RadArgs*, RadStream);
extern "C" int r4d_rad_router_gemm_scored_scatter_init(const RadParam*, int, int, int, void**);
extern "C" void r4d_rad_router_gemm_scored_scatter_fini(void*);
extern "C" void r4d_f32dec_pin(int rgs_form, int gemm_form);
extern "C" int rad_f32dec_stats(long long* out, int n);
int r4d_f32dec_set(int v);
int r4d_f32dec_get();

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

constexpr int64_t NE = 384, TK = 6, TAIL = 16;
constexpr int kRgsForms = 14;
const int kGemmForms[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 };

std::vector<uint16_t> rows_x(int64_t M, int64_t K, bool nan_rows) {
    std::vector<uint16_t> x((size_t)(M * K));
    for (int64_t r = 0; r < M; ++r) {
        const double rs = std::ldexp(1.0, (int)(rnd() % 5) - 2);
        for (int64_t i = 0; i < K; ++i) x[(size_t)(r * K + i)] = bf((float)(gauss() * rs));
        if (nan_rows && (r % 3) == 1) x[(size_t)(r * K + (int64_t)(rnd() % K))] = bf(NAN);
    }
    return x;
}

void rgs(void* inst, int64_t M, int64_t K, int wbf, int mode, int64_t nl, int64_t base) {
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
        if (wbf) { uint32_t u = (uint32_t)wb[i] << 16; std::memcpy(&wf[i], &u, 4); }
    }
    const void* wp = wbf ? (const void*)wb.data() : (const void*)wf.data();
    const std::vector<uint16_t> x = rows_x(M, K, mode == 2);
    struct Out {
        std::vector<float> lg, w;
        std::vector<int32_t> ids, so, off, cnt;
        void init(int64_t m, int64_t nl) {
            lg.assign((size_t)(m * NE + TAIL), -3.0f);
            w.assign((size_t)(m * TK + TAIL), -3.0f);
            ids.assign((size_t)(m * TK + TAIL), -77);
            so.assign((size_t)(m * TK + TAIL), -77);
            off.assign((size_t)(nl + 1 + TAIL), -77);
            cnt.assign((size_t)(nl + TAIL), -77);
        }
    };
    const uint32_t WD = wbf ? RAD_BF16 : RAD_F32;
    std::vector<RadParam> p2 = { pint("M", M), pint("K", K), pint("n_expert", NE), pint("n_local", nl), pint("top_k", TK),
                                 pint("norm", 1), pf64("route_scale", 1.5), pstr("score", "sqrtsoftplus"), pstr("dtype", "bf16"),
                                 pint("expert_base", base) };
    auto run = [&](Out& o, int form) {
        o.init(M, nl);
        std::vector<RadTensor> t = { tens(x.data(), RAD_BF16, { M, K }), tens(wp, WD, { NE, K }), tens(bias.data(), RAD_F32, { NE }),
                                     tens(o.lg.data(), RAD_F32, { M, NE }), tens(o.ids.data(), RAD_I32, { M, TK }),
                                     tens(o.w.data(), RAD_F32, { M, TK }), tens(o.so.data(), RAD_I32, { M * TK }),
                                     tens(o.off.data(), RAD_I32, { nl + 1 }), tens(o.cnt.data(), RAD_I32, { nl }) };
        RadArgs a = mkargs(t, p2, inst);
        r4d_f32dec_pin(form, -1);
        const int ra = r4d_rad_router_gemm_scored_scatter(&a, nullptr);
        const int rb = r4d_rad_router_gemm_scored_scatter(&a, nullptr);
        r4d_f32dec_pin(-1, -1);
        return ra == RAD_OK && rb == RAD_OK ? RAD_OK : (ra != RAD_OK ? ra : rb);
    };
    Out ref, o;
    const int r0 = run(ref, 0);
    std::string bad;
    for (int f = 1; f <= kRgsForms; ++f) {
        const int rc = run(o, f);
        const long long d = ndiff(ref.lg, o.lg) + ndiff(ref.ids, o.ids) + ndiff(ref.w, o.w) + ndiff(ref.so, o.so) +
                            ndiff(ref.off, o.off) + ndiff(ref.cnt, o.cnt);
        if (rc != RAD_OK || d) bad += fmt(" form %d: rc %d, %lld words;", f, rc, d);
    }
    check(r0 == RAD_OK && bad.empty(),
          fmt("SPD-F router %s weights, %s, M=%lld K=%lld, %lld local at %lld: forms 1..%d (each twice) byte for byte vs the stock form",
              wbf ? "bf16" : "f32", mode == 0 ? "random" : mode == 1 ? "tied" : "NaN rows", (long long)M, (long long)K,
              (long long)nl, (long long)base, kRgsForms),
          bad.empty() ? std::string() : fmt("stock rc %d;%s", r0, bad.c_str()));
}

void gemm(int64_t M, int64_t N, int64_t K, int wbf) {
    std::vector<float> wf((size_t)(N * K));
    for (auto& v : wf) v = (float)(gauss() * 0.02);
    std::vector<uint16_t> wb((size_t)(N * K));
    for (size_t i = 0; i < wb.size(); ++i) {
        wb[i] = bf(wf[i]);
        if (wbf) { uint32_t u = (uint32_t)wb[i] << 16; std::memcpy(&wf[i], &u, 4); }
    }
    const void* wp = wbf ? (const void*)wb.data() : (const void*)wf.data();
    const std::vector<uint16_t> x = rows_x(M, K, true);
    std::vector<RadParam> pg = { pint("M", M), pint("N", N), pint("K", K), pstr("dtype", "bf16") };
    auto run = [&](std::vector<float>& y, int form) {
        y.assign((size_t)(M * N + TAIL), -5.0f);
        std::vector<RadTensor> t = { tens(x.data(), RAD_BF16, { M, K }), tens(wp, wbf ? RAD_BF16 : RAD_F32, { N, K }),
                                     tens(y.data(), RAD_F32, { M, N }) };
        RadArgs a = mkargs(t, pg);
        r4d_f32dec_pin(-1, form);
        const int rc = r4d_rad_gemm_nt_f32(&a, nullptr);
        r4d_f32dec_pin(-1, -1);
        return rc;
    };
    std::vector<float> ref, y;
    const int r0 = run(ref, 0);
    std::string bad;
    for (int f : kGemmForms) {
        const int rc = run(y, f);
        const long long d = ndiff(ref, y);
        if (rc != RAD_OK || d) bad += fmt(" form %d: rc %d, %lld words;", f, rc, d);
    }
    check(r0 == RAD_OK && bad.empty(),
          fmt("SPD-F gemm_nt_f32 %s weights M=%lld N=%lld K=%lld: %d forms byte for byte vs the stock form", wbf ? "bf16" : "f32",
              (long long)M, (long long)N, (long long)K, (int)(sizeof(kGemmForms) / sizeof(kGemmForms[0]))),
          bad.empty() ? std::string() : fmt("stock rc %d;%s", r0, bad.c_str()));
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && !std::strcmp(argv[1], "quick");
    /* the switch: 0 .. 3 taken, 4 refused; under 0 (unpinned) no call takes a non-stock form */
    check(r4d_f32dec_set(1) == 1 && r4d_f32dec_get() == 1 && r4d_f32dec_set(3) == 1 && r4d_f32dec_get() == 3 &&
              r4d_f32dec_set(4) < 0 && r4d_f32dec_get() == 3 && r4d_f32dec_set(0) == 1 && r4d_f32dec_get() == 0,
          "SPD-F switch ds41_f32dec (libr4d's r4d_f32dec_set / _get): 0 / 1 / 3 applied and read back, 4 refused");
    void* inst = nullptr;
    const int irc = r4d_rad_router_gemm_scored_scatter_init(nullptr, 0, 0, 1, &inst);
    check(irc == RAD_OK && inst, "router_gemm_f32_scored_topk_scatter init takes an instance");
    const std::vector<int64_t> ms = quick ? std::vector<int64_t>{ 9, 33 } : std::vector<int64_t>{ 1, 9, 17, 33, 48 };
    if (inst) {
        for (int64_t M : ms) {
            rgs(inst, M, 1280, 1, 0, NE, 0);
            rgs(inst, M, 1280, 1, 0, 96, 96);
            if (!quick) {
                rgs(inst, M, 1280, 0, 0, NE, 0);
                rgs(inst, M, 1056, 1, 1, NE, 0);
                rgs(inst, M, 1280, 1, 2, NE, 0);
            }
        }
        r4d_rad_router_gemm_scored_scatter_fini(inst);
    }
    for (int64_t M : ms) {
        gemm(M, 264, 1280, 1);
        gemm(M, 1024, 1280, 1);
        if (!quick) {
            gemm(M, 264, 1280, 0);
            gemm(M, 264, 1056, 1);
        }
    }
    /* unpinned under ds41_f32dec 0: the stock rule everywhere (no call counted) */
    long long s0[2] = { 0, 0 }, s1[2] = { 0, 0 };
    rad_f32dec_stats(s0, 2);
    r4d_f32dec_set(0);
    {
        const int64_t M = 33, N = 264, K = 1280;
        std::vector<uint16_t> x = rows_x(M, K, false), w((size_t)(N * K), 0);
        std::vector<float> y((size_t)(M * N));
        std::vector<RadParam> pg = { pint("M", M), pint("N", N), pint("K", K), pstr("dtype", "bf16") };
        std::vector<RadTensor> t = { tens(x.data(), RAD_BF16, { M, K }), tens(w.data(), RAD_BF16, { N, K }), tens(y.data(), RAD_F32, { M, N }) };
        RadArgs a = mkargs(t, pg);
        r4d_rad_gemm_nt_f32(&a, nullptr);
    }
    rad_f32dec_stats(s1, 2);
    check(s1[0] == s0[0] && s1[1] == s0[1], "SPD-F ds41_f32dec 0, unpinned: no call takes a non-stock form",
          fmt("router %lld -> %lld, gemm %lld -> %lld", s0[0], s1[0], s0[1], s1[1]));
    std::printf("%d cases, %d failed\n", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
