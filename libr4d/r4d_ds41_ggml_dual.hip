/* r4d_ds41_ggml_dual.hip -- lever BI-e: gemm_nt_ggml over a decode row group and a prefill row group of one weight in
 * ONE launch (gemm_nt_ggml_dual's kernel; r4d_ds41_ggml_dual.h says why).
 *
 * ---- the launch
 *
 *  A 1-D grid of nA + nB blocks of 256 threads. nA blocks run gg_mr_body (r4d_ds41_ggml_mr.hip, lever D2's multi-row
 *  form) over the decode group (x0, y0, M0 <= 48) at H 4, R 1, WN 8 (32 weight rows a block) and MT activation rows
 *  a block (4, 8, 16 or 24; M0 above MT splits over a second index, as the form's grid.y does); nB blocks run
 *  gg_pf_body (r4d_ds41_ggml_pf.hip, lever GPF) over the prefill group (x1, y1, M1 > 48) at the form gemm_nt_ggml
 *  would take there: the GPF rule's code when gg_pf is on (422 or 426 here), else 322, the stock WMMA kernel's
 *  geometry and order. A block's index in its group is (index mod the group's grid.x, index div it), the same
 *  (blockIdx.x, blockIdx.y) its own launch gives that block. The dynamic LDS is the larger of the two (the decode
 *  group's slab capped at the prefill form's, so the prefill blocks' occupancy is their own launch's).
 *
 * ---- exactness: every output is the bytes of its group's own gemm_nt_ggml launch
 *
 *  A block runs exactly one body, and the body is the code its own kernel runs (gg_mr_kernel and gg_pf_kernel are
 *  that body called with blockIdx.x / .y), with the block index its own launch would give it, the same arguments and
 *  256 threads (8 waves, as the forms' own launches here). No output is written by both groups (y0 and y1 are the
 *  two groups' rows), no block reads what another writes.
 *   - Decode rows: lever D2's argument (r4d_ds41_ggml_mr.hip) -- every (H, R, MT, WN, SS) form gives each output the
 *     stock decode kernel's operations (gg_gemv_kernel, lane = unit mod 32, the same butterfly); gg_gemm takes D2's
 *     form or that stock kernel at M0 <= 48, so these are its bytes whichever it takes. r4d_selftest --mrcheck checks
 *     the form, --dualcheck this launch.
 *   - Prefill rows: lever GPF's argument (r4d_ds41_ggml_pf.hip) -- every form gives each output the stock WMMA
 *     kernel's operations; 322 is the stock geometry itself. gg_gemm takes the GPF rule's form (gg_pf on) or the stock
 *     kernel at M1 > 48 when the operands allow the WMMA kernel (the same test here), so these are its bytes.
 *  The order of the two groups in the grid (decode blocks first or last) changes only when a block runs.
 */
#define R4D_GG_BLOCKS_ONLY 1
#define R4D_GG_WM_SRC 1
#include "r4d_ds41_ggml.hip"
#define R4D_GG_MR_KERNEL_ONLY 1
#include "r4d_ds41_ggml_mr.hip"
#define R4D_GG_PF_KERNEL_ONLY 1
#include "r4d_ds41_ggml_pf.hip"
#include "r4d_ds41_ggml_dual.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kDualThreads = 256;   /* both forms' blocks here: MR at WN 8, GPF at WN 8 */

/* `last`: 0 the decode group's blocks first, 1 last. */
template <int T, int MTD, int PMT, int PFI>
__global__ __launch_bounds__(kDualThreads) void gg_dual_kernel(GgMrArgs A, GgTiledArgs B, int nA, int gxA, int gxB,
                                                               int last) {
    const int b = (int)blockIdx.x, nB = (int)gridDim.x - nA;
    const int d = last ? b - nB : b;
    if (d >= 0 && d < nA) {
        gg_mr_body<T, 4, 1, MTD, uint16_t, 0>(A, d % gxA, d / gxA);
        return;
    }
    const int p = last ? b : b - nA;
    gg_pf_body<T, PMT, 2, 8, uint16_t, 0, PFI>(B, p % gxB, p / gxB);
}

template <int T, int MTD>
int dual_launch_m(RadStream s, int pcode, const GgMrArgs& A, const GgTiledArgs& B, int nA, int gxA, int gxB, int nB,
                  int last, unsigned lds) {
    const dim3 grid((unsigned)(nA + nB)), blk(kDualThreads);
    switch (pcode) {
    case 322: return R4D_EMIT(s, (gg_dual_kernel<T, MTD, 3, 0>), grid, blk, lds, A, B, nA, gxA, gxB, last);
    case 422: return R4D_EMIT(s, (gg_dual_kernel<T, MTD, 4, 0>), grid, blk, lds, A, B, nA, gxA, gxB, last);
    case 426: return R4D_EMIT(s, (gg_dual_kernel<T, MTD, 4, 1>), grid, blk, lds, A, B, nA, gxA, gxB, last);
    default: return R4D_GG_DUAL_DECLINE;
    }
}

template <int T>
int dual_launch_t(RadStream s, int mtd, int pcode, const GgMrArgs& A, const GgTiledArgs& B, int nA, int gxA, int gxB,
                  int nB, int last, unsigned lds) {
    switch (mtd) {
    case 4: return dual_launch_m<T, 4>(s, pcode, A, B, nA, gxA, gxB, nB, last, lds);
    case 8: return dual_launch_m<T, 8>(s, pcode, A, B, nA, gxA, gxB, nB, last, lds);
    case 16: return dual_launch_m<T, 16>(s, pcode, A, B, nA, gxA, gxB, nB, last, lds);
    case 24: return dual_launch_m<T, 24>(s, pcode, A, B, nA, gxA, gxB, nB, last, lds);
    default: return R4D_GG_DUAL_DECLINE;
    }
}

unsigned dual_pf_lds(int ty, int pmt) {
    switch (ty) {
    case GG_Q8_0: return pf_lds<GG_Q8_0>(pmt);
    case GG_Q5_K: return pf_lds<GG_Q5_K>(pmt);
    case GG_Q6_K: return pf_lds<GG_Q6_K>(pmt);
    default: return 0;
    }
}

/* The decode group's activation rows a block: the fewest padded rows (MT x tiles), ties to the larger MT. */
int dual_mtd(int M0, int* tiles) {
    int best = 0, bcost = 1 << 30, bt = 0;
    for (int mt : { 4, 8, 16, 24 }) {
        const int t = (M0 + mt - 1) / mt, cost = t * mt;
        if (cost < bcost || (cost == bcost && mt > best)) { best = mt; bcost = cost; bt = t; }
    }
    *tiles = bt;
    return best;
}

/* bench pin (r4d_selftest --dualcheck): 0 the rule, -1 decline (gemm_nt_ggml_dual issues its two launches), 1 decode
 * blocks first, 2 decode blocks last */
int g_dual_pin = 0;
/* RADIANCE_DS41_GG_DUAL_LAST at load: 1 the decode group's blocks last, else first */
int dual_last_env() {
    const char* v = std::getenv("RADIANCE_DS41_GG_DUAL_LAST");
    return v && v[0] == '1' && !v[1] ? 1 : 0;
}
const int g_dual_last = dual_last_env();
char g_dual_last_form[128];
/* gg_al16 (r4d_ds41_ggml.hip, past this unit's blocks): 16-byte rows, as the WMMA kernel stages them */
bool dual_al16(const void* p, long long ld) { return ((uintptr_t)p & 15u) == 0 && ld % 8 == 0; }

}  // namespace

int r4d_gg_dual_gemm(RadStream s, const R4dGgDualCall& c) {
    if (g_dual_pin == -1) return R4D_GG_DUAL_DECLINE;
    if (c.ty != GG_Q5_K && c.ty != GG_Q6_K && c.ty != GG_Q8_0) return R4D_GG_DUAL_DECLINE;
    if (c.M0 < 1 || c.M0 > kMrMaxM || c.M1 <= kMrMaxM || c.N < 1 || c.K < 1) return R4D_GG_DUAL_DECLINE;
    /* the WMMA kernel's operands, as gg_gemm_run tests them above the band */
    if (!dual_al16(c.x1, c.x1_ld) || c.K % (2 * kWmS) || (c.N + kWmBN - 1) / kWmBN > 65535) return R4D_GG_DUAL_DECLINE;
    if (c.ty == GG_Q5_K && ((uintptr_t)c.w & 3u)) return R4D_GG_DUAL_DECLINE;
    int pcode = r4d_gg_pf_code(c.ty, c.N, c.K);
    if (pcode == 0) pcode = 322;
    if (pcode != 322 && pcode != 422 && pcode != 426) return R4D_GG_DUAL_DECLINE;
    const int pmt = pcode / 100;

    /* the decode group: lever D2's form at H 4, R 1, WN 8 (r4d_gg_mr_gemm's arguments for that form) */
    int gy = 0;
    const int mtd = dual_mtd(c.M0, &gy);
    constexpr int H = 4, LPR = 32 / H, VPS = 32 * LPR, WN = 8;
    const int PD = mtd >= 24 ? 1 : (mtd >= 8 ? 2 : 4);   /* = mr_pd<MT, 1>() */
    const long long sb = (long long)mtd * VPS * 4;
    if (sb > kMrLdsMax) return R4D_GG_DUAL_DECLINE;
    const int Q = c.ty == GG_Q8_0 ? 32 : 256, UPB = c.ty == GG_Q8_0 ? 1 : 8;
    const int UK = c.K / Q * UPB, NS = (UK + LPR - 1) / LPR;
    const unsigned pl = dual_pf_lds(c.ty, pmt);
    long long SS = 4096LL * WN / sb;
    if (SS > PD) SS = SS / PD * PD;
    if (SS * sb > (long long)pl) SS = (long long)pl / sb;   /* the prefill blocks keep their own launch's LDS */
    if (SS < 1) SS = 1;
    if (SS >= NS) SS = NS;
    if (SS * sb > 65536) return R4D_GG_DUAL_DECLINE;
    const long long gxA = ((long long)c.N + WN * H - 1) / (WN * H);
    GgMrArgs A{};
    A.x = c.x0; A.x_ld = c.x0_ld; A.w = c.w; A.w_ld = c.w_ld; A.y = c.y0; A.y_ld = c.y0_ld;
    A.M = c.M0; A.N = c.N; A.K = c.K;
    A.UK = UK; A.NS = NS; A.SS = (int)SS;
    A.x_al = dual_al16(c.x0, c.x0_ld) ? 1 : 0;

    /* the prefill group: lever GPF's form (r4d_gg_pf_gemm's arguments) */
    GgTiledArgs B{};
    B.x = c.x1; B.x_ld = c.x1_ld;
    B.w = c.w; B.wtab = nullptr; B.w_ld = c.w_ld; B.w_es = 0;
    B.y = c.y1; B.y_ld = c.y1_ld;
    B.M = c.M1; B.N = c.N; B.K = c.K;
    B.x_al = 1; B.tab_bytes = 0;
    const long long gxB = ((long long)c.M1 + 16 * pmt - 1) / (16 * pmt), gyB = ((long long)c.N + 255) / 256;
    const long long nA = gxA * gy, nB = gxB * gyB;
    if (nA + nB > 0x7fffffffLL) return R4D_GG_DUAL_DECLINE;
    const unsigned lds = (unsigned)std::max<long long>(SS * sb, (long long)pl);
    const int last = g_dual_pin == 1 ? 0 : g_dual_pin == 2 ? 1 : g_dual_last;
    std::snprintf(g_dual_last_form, sizeof g_dual_last_form, "dual decode H4 R1 MT%d WN8 SS%lld %lldx%d + pf %d %lldx%lld lds %u %s",
                  mtd, SS, gxA, gy, pcode, gxB, gyB, lds, last ? "decode last" : "decode first");
    switch (c.ty) {
    case GG_Q8_0: return dual_launch_t<GG_Q8_0>(s, mtd, pcode, A, B, (int)nA, (int)gxA, (int)gxB, (int)nB, last, lds);
    case GG_Q5_K: return dual_launch_t<GG_Q5_K>(s, mtd, pcode, A, B, (int)nA, (int)gxA, (int)gxB, (int)nB, last, lds);
    case GG_Q6_K: return dual_launch_t<GG_Q6_K>(s, mtd, pcode, A, B, (int)nA, (int)gxA, (int)gxB, (int)nB, last, lds);
    default: return R4D_GG_DUAL_DECLINE;
    }
}

/* bench / test hooks (r4d_selftest --dualcheck) */
extern "C" void r4d_gg_dual_pin(int mode) { g_dual_pin = (mode >= -1 && mode <= 2) ? mode : 0; }
/* the last one-launch form ("dual decode H4 R1 MT8 ... + pf 426 ..."), for the bench's lines */
extern "C" const char* r4d_gg_dual_last() { return g_dual_last_form; }
