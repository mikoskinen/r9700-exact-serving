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
 *   r4d_emu_ds41pf [quick]      quick: one K, two M */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_ds41_ggml_pf.h"

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
    std::printf("%s: %d cases, %d failures\n", g_fail ? "FAIL" : "PASS", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
