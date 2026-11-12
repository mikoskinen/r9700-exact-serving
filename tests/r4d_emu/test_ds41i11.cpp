/* tests/r4d_emu: ds41-int11's lever 1U-MX on the host SIMT emulator (the GPU twin: r4d_selftest --case mix_wv):
 *
 *   MX   mhc_mix under libr4d's fn4 ds41_mix_wv 12 / 16 / 24 (mix_split_kernel with that many waves a chunk at the decode
 *        heights) against 0 (8 waves), byte for byte: the scratch's partials, pre, post and comb, at M 1 / 2 / 3 / 6 / 8, and
 *        M 9 / 24 (above the heights: the switch must leave the launch alone). The wide-form counter (rad_mx_stats) must
 *        move exactly where the switch acts.
 *
 *   r4d_emu_ds41i11 [quick] */
#include "rad_abi.h"
#include "rad_plugin.h"

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" int r4d_rad_mhc_mix(const RadArgs*, RadStream);
extern "C" int64_t r4d_rad_mhc_mix_scratch(const RadArgs*);
extern "C" int rad_mx_stats(long long* out, int n);
void r4d_mix_wv_set(int v);
int r4d_mix_wv_get();

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
template <class V>
long long ndiff(const std::vector<V>& a, const std::vector<V>& b) {
    long long d = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) d += std::memcmp(&a[i], &b[i], sizeof(V)) != 0;
    return d + (long long)(a.size() > b.size() ? a.size() - b.size() : b.size() - a.size());
}

void mx(int64_t M, const std::vector<int>& wvs) {
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
    auto go = [&](Out& o, int wv) {
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
        r4d_mix_wv_set(wv);
        return r4d_rad_mhc_mix(&a, nullptr);
    };
    Out R;
    const int rc0 = go(R, 0);
    for (int wv : wvs) {
        Out F;
        long long s0[2] = { 0, 0 }, s1[2] = { 0, 0 };
        rad_mx_stats(s0, 2);
        const int rc1 = go(F, wv);
        rad_mx_stats(s1, 2);
        const bool took = s1[0] > s0[0];
        const long long dp = ndiff(R.scr, F.scr), dpre = ndiff(R.pre, F.pre), dpo = ndiff(R.post, F.post),
                        dcb = ndiff(R.comb, F.comb);
        check(rc0 == RAD_OK && rc1 == RAD_OK && dp + dpre + dpo + dcb == 0 && took == (M <= 8),
              fmt("mhc_mix ds41_mix_wv %d vs 0, M=%lld (partials, pre, post, comb)", wv, (long long)M),
              fmt("rc %d / %d; differing words: partials %lld pre %lld post %lld comb %lld; wide launches +%lld (want %s)",
                  rc0, rc1, dp, dpre, dpo, dcb, s1[0] - s0[0], M <= 8 ? "> 0" : "0"));
    }
    r4d_mix_wv_set(0);
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && !std::strcmp(argv[1], "quick");
    r4d_mix_wv_set(13);
    check(r4d_mix_wv_get() == 0, "ds41_mix_wv: an unknown value (13) reads back 0");
    for (int64_t M : { 1, 2, 6 }) mx(M, { 12, 16, 24 });
    if (!quick) {
        mx(3, { 12, 24 });
        mx(8, { 12, 16, 24 });
    }
    mx(9, { 24 });
    if (!quick) mx(24, { 12 });
    std::printf("%s: %d cases, %d failures\n", g_fail ? "FAIL" : "PASS", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
