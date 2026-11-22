/* tests/r4d_emu: lever WIL's IL forms (moe_gemm_mxil, r4d_mx_mr_moe_il) on the host SIMT emulator against the native
 * WMMA form drafter_mx 2 serves (xm_wm WR 1 NT 1 PD 2, 8 waves, the 17-byte rows), byte for byte -- the host twin of
 * r4d_selftest --mxilcheck, written for the DRAFTER's lever WCN (round 87: the IL gate_up's loads kept in flight, a
 * PD-aligned K loop, the staging pipeline xw_ld16 / xw_st16, xw_stage1). Each expert's IL bytes are the native blocks
 * moved (the relayout's map, as --mxilcheck checks it on a card). Forms: drafter_mx 7 (PD 4 at 12 waves a SIMD: the
 * served gate_up), drafter_mx 8 (7 without WCN: xm_wm_kernel AB 80, the A arm of WCN's A / B), drafter_mx 4 / 6 (PD 9;
 * 6 at the resident grid), and the IL down (K < 2048: xm_wt PD 9 TS 4). Shapes pick the paths inside WCN:
 * K 2048 (UK 64: the staging pipeline with aligned rows, xw_stage1 without), K 2176 (UK 68: PD-aligned, xw_stage1), K 2080
 * (UK 65: the guarded loop), K 576 (the down). Routings: several rows an expert (items past the grid), a dropped tail
 * (dead rows: the pipeline reads row 0 and stores zeros), tokens past AR, stacked experts and a table, a_order token /
 * sorted, N below / across the 128-row group.
 *   r4d_emu_ds41wcn [quick] */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_ds41_mxfp4_mr.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" void r4d_mx_mr_pin(int on, int h, int r, int mt, int wn, int grid);
extern "C" const char* r4d_mx_mr_last();

namespace {
uint64_t g_rs = 0x5DEECE66DULL;
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
int g_fail = 0, g_cases = 0, g_launch = 0;

/* the IL layout: 16-row group g, block b -> 272 bytes (r4d_selftest --mxilcheck's host interleave of the native blocks) */
void to_il(const uint8_t* nat, uint8_t* il, int N, int nb) {
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
    const Sh shapes[] = { { 48, 2048, false, "UK 64: pipeline (x_al) / stage1" }, { 144, 2176, true, "UK 68: PD-aligned, stage1" },
                          { 32, 2080, false, "UK 65: guarded loop" }, { 64, 576, true, "K 576: IL down (xm_wt)" } };
    const std::vector<int> toks = quick ? std::vector<int>{ 1, 6, 17, 40 } : std::vector<int>{ 1, 2, 5, 6, 8, 16, 17, 24, 33, 40, 48 };
    for (const Sh& sh : shapes) {
        const int nb = sh.K / 32;
        const long long rb = (long long)nb * 17, es = (long long)sh.N * rb;
        std::vector<uint8_t> w((size_t)(E * es)), wi((size_t)(E * es));
        for (size_t b = 0; b < w.size(); b += 17) {
            w[b] = (uint8_t)(118 + rnd() % 14);
            for (int i = 1; i < 17 && b + i < w.size(); ++i) w[b + i] = (uint8_t)rnd();
        }
        for (int e = 0; e < E; ++e) to_il(w.data() + e * es, wi.data() + e * es, sh.N, nb);
        std::vector<const uint8_t*> tab((size_t)E), tabi((size_t)E);
        for (int e = 0; e < E; ++e) { tab[(size_t)e] = w.data() + e * es; tabi[(size_t)e] = wi.data() + e * es; }
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
                if (short_ar) AR = AR > 1 ? AR - AR / 3 : AR;   /* rows past AR: zeros */
                std::vector<uint16_t> x((size_t)(sh.srt ? T : tk) * sh.K);
                for (auto& v : x) v = rx();
                std::vector<uint16_t> ref((size_t)T * sh.N, 0xA5A5), got((size_t)T * sh.N);
                R4dMxMrCall c{ x.data(), (long long)sh.K, al ? 1 : 0, AR, table ? nullptr : w.data(), table ? tab.data() : nullptr, rb, es,
                               ref.data(), (long long)sh.N, sorted.data(), off.data(), T, sh.N, sh.K, E, top_k, sh.srt ? 1 : 0, tk, false };
                /* the reference: drafter_mx 2's WMMA form on the native rows (pin: WMMA, PD 2, WR 1, NT 1, 8 waves) */
                r4d_mx_mr_set(0);
                r4d_mx_mr_pin(1, 1024 + 2, 1, 1, 8, 0);
                const int rr = r4d_mx_mr_moe(nullptr, c);
                const std::string fr = r4d_mx_mr_last();
                r4d_mx_mr_pin(0, 0, 0, 0, 0, 0);
                if (rr != 0) { std::printf("FAIL %dx%d tokens %d: reference rc %d\n", sh.N, sh.K, tk, rr); ++g_fail; ++sf; continue; }
                for (int mx : { 7, 8, 4, 6 }) {
                    std::fill(got.begin(), got.end(), 0x5A5A);
                    R4dMxMrCall ci = c;
                    ci.w = table ? nullptr : wi.data();
                    ci.wtab = table ? tabi.data() : nullptr;
                    ci.y = got.data();
                    r4d_mx_mr_set(mx);
                    const int rc = r4d_mx_mr_moe_il(nullptr, ci);
                    r4d_mx_mr_set(0);
                    ++g_cases; ++sc; ++g_launch;
                    if (rc != 0) { std::printf("FAIL %dx%d tokens %d mx %d: IL rc %d\n", sh.N, sh.K, tk, mx, rc); ++g_fail; ++sf; continue; }
                    if (got != ref) {
                        size_t bad = 0, first = 0;
                        for (size_t e = 0; e < got.size(); ++e) if (got[e] != ref[e]) { if (!bad) first = e; ++bad; }
                        std::printf("FAIL %dx%d tokens %d%s%s%s%s mx %d (%s | ref %s): %zu of %zu differ, first row %zu col %zu\n", sh.N, sh.K, tk,
                                    tail ? " tail" : "", table ? " table" : "", al ? " x_al" : "", short_ar ? " short AR" : "", mx, r4d_mx_mr_last(),
                                    fr.c_str(), bad, got.size(), first / (size_t)sh.N, first % (size_t)sh.N);
                        ++g_fail; ++sf;
                    }
                    if (mx == 7 && sh.K < 2048) break;   /* the down's form does not depend on drafter_mx */
                }
            }
        std::printf("IL %dx%d (%s): %d cases, %d failures\n", sh.N, sh.K, sh.path, sc, sf);
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
