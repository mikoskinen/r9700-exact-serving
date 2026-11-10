/* tests/r4d_emu: ds41-drk's drafter kernels against the stock kernels on the host SIMT emulator, byte for byte.
 *   drafter_mx: moe_gemm_ggml type=mxfp4's multi-row forms (r4d_ds41_mxfp4_mr.hip) vs the stock gemv band
 *     (mx_gemv_kernel, MOE) and tiled kernel (mx_tiled_kernel, MOE), the unit's copies of r4d_ds41_mxfp4.hip's source:
 *     8 experts a rank of 32, top 3, routings with several rows an expert, with and without a dropped tail,
 *     stacked experts and a pointer table, tokens 1..48, every form.
 *   drafter_bf: the skinny bf16 GEMM's prefetching form (r4d_gemm_bf16_nt_pf.hip) vs gemm_bf16_nt_m64's launch at
 *     MB = 1, splits 1 / 2 / 4, M 1..48, every form (the emulator's WMMA, the same for both).
 *   r4d_emu_ds41drk [quick] */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_ds41_mxfp4_mr.h"
#include "r4d_gemm_bf16_nt_pf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int r4d_mx_mr_emu_stock(const R4dMxMrCall& c, int bm);
extern "C" void r4d_mx_mr_pin(int on, int h, int r, int mt, int wn, int grid);
extern "C" const char* r4d_mx_mr_last();
extern "C" void r4d_gemm_bf16_nt_m64(long a, long w, long c, int M, int K, int N, int WV, int SK, int MB, long stream);
extern "C" void r4d_bf_pf_pin(int on, int ct, int ku, int wv);
extern "C" const char* r4d_bf_pf_last();

namespace {
uint64_t g_rs = 0x9E3779B97F4A7C15ull;
uint64_t rnd() { g_rs ^= g_rs << 13; g_rs ^= g_rs >> 7; g_rs ^= g_rs << 17; return g_rs; }
uint16_t f2bf(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
uint16_t rx() {
    const uint32_t c = (uint32_t)(rnd() % 1000);
    if (c < 5) return 0x0000;
    if (c < 10) return 0x8000;
    if (c < 15) return (uint16_t)((rnd() & 1 ? 0x8000 : 0) | (rnd() % 0x7f + 1));
    return f2bf((float)((int)(rnd() % 20001) - 10000) / 2500.0f);
}
int g_fail = 0, g_cases = 0;

/* ---- drafter_mx */
void mx_part(bool quick) {
    const int E = 8, EG = 32, top_k = 3;
    struct Sh { int N, K; bool srt; };
    const Sh shapes[] = { { 37, 320, false }, { 100, 1280, true }, { 72, 64, false } };
    std::vector<int> toks = quick ? std::vector<int>{ 1, 5, 16, 17, 40 } : std::vector<int>{ 1, 2, 3, 5, 7, 8, 10, 15, 16, 17, 20, 25, 33, 40, 48 };
    struct P { int on, h, r, mt, wn, grid; };
    const P gp[] = { { -1, 0, 0, 0, 0, 0 }, { 1, 4, 1, 2, 0, 0 }, { 1, 4, 1, 4, 1, 0 }, { 1, 4, 1, 8, 2, 3 }, { 1, 4, 1, 16, 0, 0 },
                     { 1, 4, 2, 4, 0, 0 }, { 1, 4, 2, 8, 1, 2 }, { 1, 2, 1, 4, 0, 0 }, { 1, 2, 1, 8, 1, 1 } };
    const P sp[] = { { -1, 0, 0, 0, 0, 0 }, { 1, 0, 1, 2, 0, 0 }, { 1, 0, 1, 4, 1, 0 }, { 1, 0, 1, 8, 0, 2 }, { 1, 0, 2, 2, 2, 0 },
                     { 1, 0, 2, 4, 0, 0 }, { 1, 0, 2, 8, 1, 1 } };
    for (const Sh& sh : shapes) {
        const long long rb = sh.K / 32 * 17, es = (long long)sh.N * rb;
        std::vector<uint8_t> w((size_t)(E * es));
        for (size_t b = 0; b < w.size(); b += 17) {
            w[b] = (uint8_t)(118 + rnd() % 14);
            for (int i = 1; i < 17 && b + i < w.size(); ++i) w[b + i] = (uint8_t)rnd();
        }
        std::vector<const uint8_t*> tab((size_t)E);
        for (int e = 0; e < E; ++e) tab[(size_t)e] = w.data() + e * es;
        for (int tk : toks)
            for (int variant = 0; variant < 4; ++variant) {
                const bool tail = variant & 1, table = variant & 2;
                std::vector<std::vector<int32_t>> by((size_t)EG);
                std::vector<int> pool(4);
                for (int t = 0; t < tk; ++t) {
                    if (t % 5 == 0) for (auto& p : pool) p = (int)(rnd() % EG);
                    int pick[3] = { -1, -1, -1 };
                    for (int j = 0; j < top_k; ++j) {
                        int e, guard = 0;
                        do { e = rnd() % 10 < 7 ? pool[rnd() % 4] : (int)(rnd() % EG); } while ((e == pick[0] || e == pick[1] || e == pick[2]) && ++guard < 64);
                        pick[j] = e;
                        by[(size_t)e].push_back(t * top_k + j);
                    }
                }
                std::vector<int32_t> off((size_t)(E + 1), 0), sorted;
                const int rank = tk % 4;
                for (int e = 0; e < E; ++e) {
                    const auto& v = by[(size_t)(rank * E + e)];
                    off[(size_t)(e + 1)] = off[(size_t)e] + (int32_t)v.size();
                    sorted.insert(sorted.end(), v.begin(), v.end());
                }
                if (tail) sorted.resize((size_t)(tk * top_k), -1);
                const int T = (int)sorted.size();
                if (T == 0) continue;
                const int AR = sh.srt ? T : tk;
                std::vector<uint16_t> x((size_t)AR * sh.K);
                for (auto& v : x) v = rx();
                std::vector<uint16_t> ref((size_t)T * sh.N), got((size_t)T * sh.N);
                for (int gv = 0; gv < 2; ++gv) {
                    const bool gemv = gv == 0 && tk <= 16;
                    if (gv == 0 && tk > 16) continue;
                    R4dMxMrCall c{ x.data(), (long long)sh.K, (AR & 1) ? 1 : 0, AR, table ? nullptr : w.data(),
                                   table ? tab.data() : nullptr, rb, es, ref.data(), (long long)sh.N, sorted.data(), off.data(),
                                   T, sh.N, sh.K, E, top_k, sh.srt ? 1 : 0, tk, gemv };
                    std::fill(ref.begin(), ref.end(), 0xA5A5);
                    if (r4d_mx_mr_emu_stock(c, (tk & 1) ? 32 : 128) != 0) { std::printf("FAIL mx stock launch\n"); ++g_fail; continue; }
                    const P* pl = gemv ? gp : sp;
                    const size_t np = gemv ? sizeof(gp) / sizeof(gp[0]) : sizeof(sp) / sizeof(sp[0]);
                    for (size_t pi = 0; pi < np; ++pi) {
                        const P& p = pl[pi];
                        std::fill(got.begin(), got.end(), 0x5A5A);
                        c.y = got.data();
                        r4d_mx_mr_pin(p.on, p.h, p.r, p.mt, p.wn, p.grid);
                        const int rc = r4d_mx_mr_moe(nullptr, c);
                        r4d_mx_mr_pin(0, 0, 0, 0, 0, 0);
                        c.y = ref.data();
                        ++g_cases;
                        if (rc != 0) { std::printf("FAIL mx %dx%d tokens %d %s form %d/%d/%d/%d: rc %d\n", sh.N, sh.K, tk, gemv ? "gemv" : "tiled", p.h, p.r, p.mt, p.wn, rc); ++g_fail; continue; }
                        if (got != ref) {
                            size_t bad = 0, first = 0;
                            for (size_t e = 0; e < got.size(); ++e) if (got[e] != ref[e]) { if (!bad) first = e; ++bad; }
                            std::printf("FAIL mx %dx%d tokens %d %s%s%s form %d/%d/%d/%d (%s): %zu of %zu differ, first row %zu col %zu\n", sh.N, sh.K, tk,
                                        gemv ? "gemv" : "tiled", tail ? " tail" : "", table ? " table" : "", p.h, p.r, p.mt, p.wn, r4d_mx_mr_last(), bad,
                                        got.size(), first / (size_t)sh.N, first % (size_t)sh.N);
                            ++g_fail;
                        }
                    }
                }
            }
        std::printf("mx %dx%d: done (%d cases, %d failures so far)\n", sh.N, sh.K, g_cases, g_fail);
        std::fflush(stdout);
    }
}

/* ---- drafter_bf */
void bf_part(bool quick) {
    struct Sh { int N, K; };
    const Sh shapes[] = { { 40, 256 }, { 100, 1280 }, { 72, 64 } };
    std::vector<int> Ms = quick ? std::vector<int>{ 1, 5, 17, 40 } : std::vector<int>{ 1, 2, 5, 6, 10, 16, 17, 20, 24, 33, 40, 48 };
    struct P { int on, ct, ku, wv; };
    const P pins[] = { { -1, 0, 0, 0 }, { 1, 1, 4, 0 }, { 1, 1, 8, 1 }, { 1, 2, 4, 0 }, { 1, 2, 8, 2 }, { 1, 4, 4, 1 } };
    for (const Sh& sh : shapes) {
        std::vector<uint16_t> w((size_t)sh.N * sh.K);
        for (auto& v : w) v = f2bf((float)((int)(rnd() % 2001) - 1000) / 8000.0f);
        for (int SK : { 1, 2, 4 }) {
            if (sh.K % (SK * 16)) continue;
            for (int M : Ms) {
                std::vector<uint16_t> x((size_t)M * sh.K), ref((size_t)M * sh.N, 0xA5A5), got((size_t)M * sh.N);
                for (auto& v : x) v = rx();
                r4d_gemm_bf16_nt_m64((long)(uintptr_t)x.data(), (long)(uintptr_t)w.data(), (long)(uintptr_t)ref.data(), M, sh.K, sh.N, 1, SK, 1, 0);
                for (const P& p : pins) {
                    std::fill(got.begin(), got.end(), 0x5A5A);
                    const R4dBfPfCall c{ x.data(), w.data(), got.data(), M, sh.K, sh.N, SK, sh.K, sh.N };
                    r4d_bf_pf_pin(p.on, p.ct, p.ku, p.wv);
                    const int rc = r4d_bf_pf_gemm(nullptr, c);
                    r4d_bf_pf_pin(0, 0, 0, 0);
                    ++g_cases;
                    if (rc != 0) { std::printf("FAIL bf %dx%d SK %d M %d form %d/%d/%d: rc %d\n", sh.N, sh.K, SK, M, p.ct, p.ku, p.wv, rc); ++g_fail; continue; }
                    if (got != ref) {
                        size_t bad = 0, first = 0;
                        for (size_t e = 0; e < got.size(); ++e) if (got[e] != ref[e]) { if (!bad) first = e; ++bad; }
                        std::printf("FAIL bf %dx%d SK %d M %d form %d/%d/%d (%s): %zu of %zu differ, first m %zu n %zu\n", sh.N, sh.K, SK, M, p.ct, p.ku,
                                    p.wv, r4d_bf_pf_last(), bad, got.size(), first / (size_t)sh.N, first % (size_t)sh.N);
                        ++g_fail;
                    }
                }
            }
        }
        std::printf("bf %dx%d: done (%d cases, %d failures so far)\n", sh.N, sh.K, g_cases, g_fail);
        std::fflush(stdout);
    }
}
}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && !std::strcmp(argv[1], "quick");
    mx_part(quick);
    bf_part(quick);
    std::printf("%s: %d cases, %d failures\n", g_fail ? "FAIL" : "PASS", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
