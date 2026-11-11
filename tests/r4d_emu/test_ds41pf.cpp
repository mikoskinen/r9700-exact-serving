/* tests/r4d_emu: lever GPF's prefill WMMA forms (r4d_ds41_ggml_pf.hip, gg_pf_kernel at every built (MT, NT, WN))
 * against the stock WMMA kernel (gg_wm_kernel, the same unit's copy of r4d_ds41_ggml.hip's source) on the host
 * SIMT emulator, byte for byte.
 *
 * q5_K / q6_K / q8_0 (and q6_K with the head's f32 output), K 256 / 768 / 1280, N 300 (a partial 16-column tile
 * and a partial block at every block width 128 / 256 / 512), M 49 / 64 / 97 / 111 (partial 16-row tiles and blocks
 * at every block height 32 / 48 / 64). Weights: random quant bytes, scales random finite f16 (normal, subnormal,
 * near the f16 top); activations: random bf16 with zeros of both signs, subnormals, huge values (products and sums
 * that overflow to inf, inf - inf NaNs) and infinities.
 *
 * Lever GPF-b (r4d_ds41_f32pf.hip): every tiled gemm_nt_f32 form against the stock gemm_nt_f32 (r4d_glm.hip's
 * launcher with the switch off, the form it picks at that M) at the real K 5120 and N 96 (router, a quarter of 384),
 * 384 and 1024 (compressor kv | gate), M 49 / 64 / 97 / 130, for f32 weights and for bf16 weights (the
 * RADIANCE_DS41_ROUTER_BF16 storage), pitches past the widths; tail columns of the output pitch must stay untouched.
 *
 *   r4d_emu_ds41pf [quick]      quick: one K, two M (and fewer f32 cases) */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_ds41_ggml_pf.h"
#include "r4d_plugin.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int r4d_gg_pf_emu_stock(const R4dGgPfCall& c);
extern "C" void r4d_gg_pf_pin(int code);
extern "C" const char* r4d_gg_pf_last();
extern "C" int r4d_gg_pf_forms(int* out, int cap);
extern "C" void r4d_f32pf_pin(int code);
extern "C" const char* r4d_f32pf_last();
extern "C" int r4d_f32pf_forms(int* out, int cap);

namespace {

enum { Q8_0 = 0, Q5_K = 1, Q6_K = 2 };   /* gg_gemm's GG_* codes */

uint64_t g_rs = 0x9E3779B97F4A7C15ull;
uint64_t rnd() { g_rs ^= g_rs << 13; g_rs ^= g_rs >> 7; g_rs ^= g_rs << 17; return g_rs; }

uint16_t f2bf(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
uint16_t rnd_f16() {
    const uint32_t k = (uint32_t)(rnd() % 100);
    const uint16_t sign = (uint16_t)((rnd() & 1) << 15);
    uint16_t e;
    if (k < 8) e = 0;
    else if (k < 12) e = (uint16_t)(28 + rnd() % 3);
    else e = (uint16_t)(5 + rnd() % 8);
    return (uint16_t)(sign | (e << 10) | (uint16_t)(rnd() & 0x3ff));
}
void fill_row(uint16_t* x, int K, int mode) {
    for (int k = 0; k < K; ++k) {
        const uint32_t c = (uint32_t)(rnd() % 1000);
        if (c < 5) x[k] = 0x0000;
        else if (c < 10) x[k] = 0x8000;
        else if (c < 15) x[k] = (uint16_t)((rnd() & 1 ? 0x8000 : 0) | (rnd() % 0x7f + 1));
        else x[k] = f2bf((float)((int)(rnd() % 20001) - 10000) / 2500.0f);
    }
    if (mode == 1) x[rnd() % K] = (uint16_t)((rnd() & 1 ? 0x8000 : 0) | 0x7f80);
    if (mode == 2)
        for (int e = 0; e < 3; ++e)
            x[rnd() % K] = f2bf((rnd() & 1 ? -1.0f : 1.0f) * std::ldexp(1.0f + (float)(rnd() % 1000) / 1000.0f, 110 + (int)(rnd() % 17)));
}

int g_fail = 0, g_cases = 0;
long long g_fin = 0, g_inf = 0, g_nan = 0;

struct Ty { int ty; const char* name; int qk, bb; };
const Ty kTypes[] = { { Q5_K, "q5_K", 256, 176 }, { Q6_K, "q6_K", 256, 210 }, { Q8_0, "q8_0", 32, 34 } };

std::vector<uint8_t> make_w(const Ty& t, int N, int K) {
    const long long rb = (long long)K / t.qk * t.bb;
    std::vector<uint8_t> w((size_t)(N * rb));
    for (auto& b : w) b = (uint8_t)rnd();
    for (int n = 0; n < N; ++n)
        for (int b = 0; b < K / t.qk; ++b) {
            uint8_t* blk = w.data() + n * rb + (long long)b * t.bb;
            const uint16_t d = rnd_f16(), dm = rnd_f16();
            if (t.ty == Q6_K) std::memcpy(blk + 208, &d, 2);
            else { std::memcpy(blk, &d, 2); if (t.ty == Q5_K) std::memcpy(blk + 2, &dm, 2); }
        }
    return w;
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && !std::strcmp(argv[1], "quick");
    const std::vector<int> Ks = quick ? std::vector<int>{ 768 } : std::vector<int>{ 256, 768, 1280 };
    const std::vector<int> Ms = quick ? std::vector<int>{ 49, 111 } : std::vector<int>{ 49, 64, 97, 111 };
    int fbuf[64];
    const int nf = r4d_gg_pf_forms(fbuf, 64);
    std::vector<int> forms;
    for (int i = 0; i < nf && i < 64; ++i)
        if (fbuf[i] < 1000) forms.push_back(fbuf[i]);   /* the ablations differ by design: card timing only */
    std::printf("lever GPF emulator check: %zu forms vs gg_wm_kernel\n", forms.size());
    const int N = 300;
    for (const Ty& t : kTypes) {
        for (int K : Ks) {
            if (K % t.qk) continue;
            const std::vector<uint8_t> w = make_w(t, N, K);
            const long long w_ld = (long long)K / t.qk * t.bb;
            for (int M : Ms) {
                std::vector<uint16_t> x((size_t)M * K);
                for (int m = 0; m < M; ++m) { const int r8 = (int)(rnd() % 8); fill_row(x.data() + (size_t)m * K, K, r8 == 0 ? 1 : r8 == 1 ? 2 : 0); }
                for (int f32 = 0; f32 < 2; ++f32) {
                    if (f32 && t.ty != Q6_K) continue;
                    const size_t ob = f32 ? 4 : 2;
                    std::vector<uint8_t> ref((size_t)M * N * ob, 0xA5), got((size_t)M * N * ob);
                    R4dGgPfCall c{ t.ty, f32 != 0, x.data(), (long long)K, w.data(), w_ld, ref.data(), (long long)N, M, N, K };
                    if (r4d_gg_pf_emu_stock(c) != 0) { std::printf("FAIL stock launch %s K=%d M=%d\n", t.name, K, M); ++g_fail; continue; }
                    for (size_t e = 0; e < ref.size(); e += ob) {
                        float v;
                        if (f32) std::memcpy(&v, &ref[e], 4);
                        else { const uint32_t u = (uint32_t)(ref[e] | (ref[e + 1] << 8)) << 16; std::memcpy(&v, &u, 4); }
                        ++(std::isnan(v) ? g_nan : std::isinf(v) ? g_inf : g_fin);
                    }
                    for (int code : forms) {
                        std::fill(got.begin(), got.end(), 0xC3);
                        c.y = got.data();
                        r4d_gg_pf_pin(code);
                        const int rc = r4d_gg_pf_gemm(nullptr, c);
                        r4d_gg_pf_pin(0);
                        ++g_cases;
                        if (rc != 0) { std::printf("FAIL %s K=%d M=%d form %d: rc %d\n", t.name, K, M, code, rc); ++g_fail; continue; }
                        if (got != ref) {
                            size_t bad = 0, first = 0;
                            for (size_t e = 0; e < got.size(); e += ob)
                                if (std::memcmp(&got[e], &ref[e], ob)) { if (!bad) first = e / ob; ++bad; }
                            std::printf("FAIL %s%s K=%d M=%d form %d (%s): %zu of %zu outputs differ, first m=%zu n=%zu\n",
                                        t.name, f32 ? " f32" : "", K, M, code, r4d_gg_pf_last(), bad, got.size() / ob,
                                        first / N, first % N);
                            ++g_fail;
                        }
                    }
                }
            }
            std::printf("%s K=%d: done (%d cases, %d failures so far)\n", t.name, K, g_cases, g_fail);
            std::fflush(stdout);
        }
    }
    std::printf("INFO stock reference outputs: %lld finite, %lld inf, %lld NaN\n", g_fin, g_inf, g_nan);
    /* ---- lever GPF-b: gemm_nt_f32's tiled form vs the stock launcher */
    {
        const int nfb = r4d_f32pf_forms(fbuf, 64);
        struct F32Shape { int N, K; };
        const std::vector<F32Shape> fsh = quick ? std::vector<F32Shape>{ { 96, 5120 } }
                                                : std::vector<F32Shape>{ { 96, 5120 }, { 384, 5120 }, { 1024, 5120 } };
        const std::vector<int> fM = quick ? std::vector<int>{ 49, 97 } : std::vector<int>{ 49, 64, 97, 130 };
        int fcases = 0, ffails = 0;
        for (const F32Shape& sh : fsh) {
            const int N = sh.N, K = sh.K;
            for (int M : fM) {
                if (N >= 1024 && M > 64) continue;   /* emulator time */
                const int xp = K + 8, yp = N + 3;
                std::vector<uint16_t> x((size_t)M * xp);
                for (int m = 0; m < M; ++m) { const int r8 = (int)(rnd() % 8); fill_row(x.data() + (size_t)m * xp, K, r8 == 0 ? 1 : r8 == 1 ? 2 : 0); }
                for (int bf = 0; bf < 2; ++bf) {
                    /* weights: random bf16-representable values (zeros, subnormals, a few huge), as f32 or as bf16 */
                    std::vector<uint16_t> wb((size_t)N * K);
                    for (int c = 0; c < N; ++c) fill_row(wb.data() + (size_t)c * K, K, (c % 13) == 0 ? 2 : 0);
                    /* f32 storage: full-precision values (products inexact, so the fma's single rounding shows), with the
                     * bf16 rows' zeros / subnormals / huge values kept; bf16 storage: the bf16 values themselves */
                    std::vector<float> wf((size_t)N * K);
                    for (size_t e = 0; e < wb.size(); ++e) {
                        uint32_t u = (uint32_t)wb[e] << 16;
                        if ((u & 0x7f800000u) != 0 && (u & 0x7f800000u) != 0x7f800000u) u |= (uint32_t)(rnd() & 0xffffu);
                        std::memcpy(&wf[e], &u, 4);
                    }
                    RadTensor t[3]{};
                    t[0].data = x.data(); t[0].dtype = RAD_BF16; t[0].rank = 2; t[0].shape[0] = M; t[0].shape[1] = K; t[0].stride[0] = xp; t[0].stride[1] = 1;
                    t[1].data = bf ? (void*)wb.data() : (void*)wf.data(); t[1].dtype = bf ? RAD_BF16 : RAD_F32; t[1].rank = 2;
                    t[1].shape[0] = N; t[1].shape[1] = K; t[1].stride[0] = K; t[1].stride[1] = 1;
                    t[2].dtype = RAD_F32; t[2].rank = 2; t[2].shape[0] = M; t[2].shape[1] = N; t[2].stride[0] = yp; t[2].stride[1] = 1;
                    RadParam p[4]{};
                    p[0] = { "M", RAD_P_INT, M };
                    p[1] = { "N", RAD_P_INT, N };
                    p[2] = { "K", RAD_P_INT, K };
                    p[3].key = "dtype"; p[3].kind = RAD_P_STR; p[3].sval = "bf16";
                    RadArgs a{};
                    a.t = t; a.n_t = 3; a.p = p; a.n_p = 4;
                    std::vector<float> ref((size_t)M * yp, -7.25f), got((size_t)M * yp);
                    t[2].data = ref.data();
                    r4d_f32pf_pin(0);
                    if (r4d_rad_gemm_nt_f32(&a, nullptr) != RAD_OK) { std::printf("FAIL f32 stock rc N=%d M=%d\n", N, M); ++ffails; continue; }
                    for (int i = 0; i < nfb; ++i) {
                        std::fill(got.begin(), got.end(), -7.25f);
                        t[2].data = got.data();
                        r4d_f32pf_pin(fbuf[i]);
                        const int rc = r4d_rad_gemm_nt_f32(&a, nullptr);
                        r4d_f32pf_pin(0);
                        ++fcases;
                        if (rc != RAD_OK || std::memcmp(got.data(), ref.data(), got.size() * 4)) {
                            size_t bad = 0, first = 0;
                            for (size_t e = 0; e < got.size(); ++e)
                                if (std::memcmp(&got[e], &ref[e], 4)) { if (!bad) first = e; ++bad; }
                            std::printf("FAIL f32 %s N=%d K=%d M=%d form %d (%s): rc %d, %zu words differ, first m=%zu n=%zu\n", bf ? "bf16-w" : "f32-w",
                                        N, K, M, fbuf[i], r4d_f32pf_last(), rc, bad, first / yp, first % yp);
                            ++ffails;
                        }
                    }
                }
            }
            std::printf("f32 N=%d K=%d: done (%d cases, %d failures so far)\n", N, K, fcases, ffails);
            std::fflush(stdout);
        }
        g_cases += fcases;
        g_fail += ffails;
    }
    std::printf("%s: %d cases, %d failures\n", g_fail ? "FAIL" : "PASS", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
