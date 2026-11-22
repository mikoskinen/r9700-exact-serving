/* tests/r4d_emu: lever GSE (DRAFTER track round 91) -- the DSpark drafter's routed gate_up with swiglu_clamped in its epilogue
 * (moe_gemm_mxil_glu, r4d_mx_mr_moe_il_glu: xm_wm_kernel GLU 1 on lever WCN's IL form) on the host SIMT emulator against the pair it
 * replaces -- the IL gate_up at drafter_mx 7 (the served one) into [T, N] followed by swiglu_clamped_kernel's expression (r4d_glm.hip:
 * g = fminf(gate, limit), u = fminf(fmaxf(up, -limit), limit), f32_to_bf16((g / (1 + expf(-g))) * u), the gate rows [0, N / 2) and the
 * up rows [N / 2, N)) -- byte for byte into the down's input [T, N / 2] (written by DECODE round 64 = QUEUE W30, the review of the
 * DRAFTER's patch-r92-gse-dsf.mbox). The GLU lanes read their rows' code bytes and scales from the rows' IL tiles (slot row % 16) and
 * take the up sums by DPP row_xmask 8; shapes / routings as test_ds41wcn (K 2048 / 2176 / 2080: the staging pipeline, PD-aligned, the
 * guarded loop; N 48 / 144 / 32 = 24 / 72 / 16 output columns: a wave's 8 columns across one or two IL tiles), dead rows (a dropped
 * tail), tokens past AR, stacked experts and a table, a_order token / sorted; limits 7 and 0.5 (the clamp bites).
 *   r4d_emu_ds41glu [quick] */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_ds41_mxfp4_mr.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" const char* r4d_mx_mr_last();
int r4d_mx_mr_moe_il_glu(RadStream s, const R4dMxMrCall& c, float limit);   /* r4d_ds41_mxfp4_mr.hip (lever GSE) */

namespace {
uint64_t g_rs = 0x5DEECE66DULL;
uint64_t rnd() { g_rs ^= g_rs << 13; g_rs ^= g_rs >> 7; g_rs ^= g_rs << 17; return g_rs; }
uint16_t f2bf_rne(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }   /* r4d_common.h f32_to_bf16 */
float bf2f(uint16_t h) { const uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }
uint16_t rx() {
    const uint32_t c = (uint32_t)(rnd() % 1000);
    if (c < 5) return 0x0000;
    if (c < 10) return 0x8000;
    if (c < 15) return (uint16_t)((rnd() & 1 ? 0x8000 : 0) | (rnd() % 0x7f + 1));
    const float v = (float)((int)(rnd() % 20001) - 10000) / 2500.0f;
    uint32_t u; std::memcpy(&u, &v, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
int g_fail = 0, g_cases = 0;

void to_il(const uint8_t* nat, uint8_t* il, int N, int nb) {   /* = test_ds41wcn's (r4d_selftest --mxilcheck's host interleave) */
    for (int g = 0; g < N / 16; ++g)
        for (int b = 0; b < nb; ++b)
            for (int r = 0; r < 16; ++r) {
                const uint8_t* q = nat + ((long long)(16 * g + r) * nb + b) * 17;
                uint8_t* t = il + ((long long)g * nb + b) * 272;
                for (int hh = 0; hh < 2; ++hh)
                    for (int i = 0; i < 4; ++i) { t[8 * (r + 16 * hh) + i] = q[1 + 4 * hh + i]; t[8 * (r + 16 * hh) + 4 + i] = q[9 + 4 * hh + i]; }
                t[256 + r] = q[0];
            }
}

void run(bool quick) {
    const int E = 8, EG = 32, top_k = 3;
    struct Sh { int N, K; bool srt; const char* path; };
    const Sh shapes[] = { { 48, 2048, false, "UK 64: pipeline / stage1" }, { 144, 2176, true, "UK 68: PD-aligned, stage1" },
                          { 32, 2080, false, "UK 65: guarded loop" } };
    const std::vector<int> toks = quick ? std::vector<int>{ 1, 6, 17, 40 } : std::vector<int>{ 1, 2, 5, 6, 8, 16, 17, 24, 33, 40, 48 };
    for (const Sh& sh : shapes) {
        const int nb = sh.K / 32, NH = sh.N / 2;
        const long long rb = (long long)nb * 17, es = (long long)sh.N * rb;
        std::vector<uint8_t> w((size_t)(E * es)), wi((size_t)(E * es));
        for (size_t b = 0; b < w.size(); b += 17) {
            w[b] = (uint8_t)(118 + rnd() % 14);
            for (int i = 1; i < 17 && b + i < w.size(); ++i) w[b + i] = (uint8_t)rnd();
        }
        for (int e = 0; e < E; ++e) to_il(w.data() + e * es, wi.data() + e * es, sh.N, nb);
        std::vector<const uint8_t*> tabi((size_t)E);
        for (int e = 0; e < E; ++e) tabi[(size_t)e] = wi.data() + e * es;
        int sc = 0, sf = 0;
        for (int tk : toks)
            for (int variant = 0; variant < 16; ++variant) {
                const bool tail = variant & 1, table = variant & 2, al = variant & 4, short_ar = variant & 8;
                if (quick && (variant & 8) && tk != 17) continue;
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
                int AR = sh.srt ? T : tk;
                if (short_ar) AR = AR > 1 ? AR - AR / 3 : AR;
                std::vector<uint16_t> x((size_t)(sh.srt ? T : tk) * sh.K);
                for (auto& v : x) v = rx();
                /* the pair: drafter_mx 7's IL gate_up (the served form) into [T, N], then swiglu_clamped on the host */
                std::vector<uint16_t> gu((size_t)T * sh.N, 0xA5A5);
                R4dMxMrCall c{ x.data(), (long long)sh.K, al ? 1 : 0, AR, table ? nullptr : wi.data(), table ? tabi.data() : nullptr, rb, es,
                               gu.data(), (long long)sh.N, sorted.data(), off.data(), T, sh.N, sh.K, E, top_k, sh.srt ? 1 : 0, tk, false };
                r4d_mx_mr_set(7);
                const int rr = r4d_mx_mr_moe_il(nullptr, c);
                r4d_mx_mr_set(0);
                if (rr != 0) { std::printf("FAIL %dx%d tokens %d: pair gate_up rc %d\n", sh.N, sh.K, tk, rr); ++g_fail; ++sf; continue; }
                for (float lim : { 7.0f, 0.5f }) {
                    std::vector<uint16_t> ref((size_t)T * NH), got((size_t)T * NH, 0x5A5A);
                    for (int r = 0; r < T; ++r)
                        for (int i = 0; i < NH; ++i) {
                            const float g = std::fmin(bf2f(gu[(size_t)r * sh.N + i]), lim);
                            const float u = std::fmin(std::fmax(bf2f(gu[(size_t)r * sh.N + NH + i]), -lim), lim);
                            ref[(size_t)r * NH + i] = f2bf_rne((g / (1.0f + std::exp(-g))) * u);
                        }
                    R4dMxMrCall cg = c;
                    cg.y = got.data();
                    cg.y_ld = NH;
                    const int rc = r4d_mx_mr_moe_il_glu(nullptr, cg, lim);
                    ++g_cases; ++sc;
                    if (rc != 0) { std::printf("FAIL %dx%d tokens %d limit %g: GLU rc %d\n", sh.N, sh.K, tk, lim, rc); ++g_fail; ++sf; continue; }
                    if (got != ref) {
                        size_t bad = 0, first = 0;
                        for (size_t e = 0; e < got.size(); ++e) if (got[e] != ref[e]) { if (!bad) first = e; ++bad; }
                        std::printf("FAIL %dx%d tokens %d%s%s%s%s limit %g (%s): %zu of %zu differ, first row %zu col %zu (%04x vs %04x)\n", sh.N, sh.K,
                                    tk, tail ? " tail" : "", table ? " table" : "", al ? " x_al" : "", short_ar ? " short AR" : "", lim, r4d_mx_mr_last(),
                                    bad, got.size(), first / (size_t)NH, first % (size_t)NH, got[first], ref[first]);
                        ++g_fail; ++sf;
                    }
                }
            }
        std::printf("GLU %dx%d (%s): %d cases, %d failures\n", sh.N, sh.K, sh.path, sc, sf);
        std::fflush(stdout);
    }
}
}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && !std::strcmp(argv[1], "quick");
    run(quick);
    std::printf("%s: %d cases, %d failures\n", g_fail ? "FAIL" : "PASS", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
