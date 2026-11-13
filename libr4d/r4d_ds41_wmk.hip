/* r4d_ds41_wmk.hip -- lever WMK (ds41-wmk2, 13.11.2026): gg_wk_kernel, the part-order WMMA form of gemm_nt_ggml /
 * logits_gemm_ggml / gemm_nt_ggml_pair / the K1 folds a / e, in a unit of its own (its own code object: nothing of it is
 * loaded on a card until a part-order launch is issued). Includes r4d_ds41_ggml.hip's block section and its WMMA
 * section (R4D_GG_BLOCKS_ONLY + R4D_GG_WM_SRC, as r4d_ds41_ggml_pf.hip does): the Wq decoders, the staging helpers and
 * gg_wmma are the stock kernel's own source. Entry r4d_gg_wk_run (r4d_ds41_wmk.h); the bench counters r4d_gg_wk_count /
 * r4d_gg_wk_last. */
#define R4D_GG_BLOCKS_ONLY 1
#define R4D_GG_WM_SRC 1
#define R4D_GG_WM_PARTFOLD 1   /* the stock WMMA kernel's template here carries the part fold (r4d_ds41_ggml.hip's unit: not) */
#include "r4d_ds41_ggml.hip"
/* Lever WMK's prefill rows (stage 3, G3): lever GPF's tile forms (gg_pf_kernel, the stock WMMA arithmetic in other
 * geometries) compiled here with the part fold -- the same part order as gg_wk_kernel's, so a row's bytes are the same in
 * a <= 48-row launch and in a prefill launch of any height. r4d_ds41_ggml_pf.hip's own unit keeps the stock order. */
#define R4D_GG_PF_KERNEL_ONLY 1
#define R4D_GG_PF_PARTFOLD 1
#include "r4d_ds41_ggml_pf.hip"
#include "r4d_ds41_ggml_pf.h"
#include "r4d_ds41_wmk.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {
/* ================================================================== gg_wk_kernel (ds41-wmk, BENCH prototype 13.11.2026)
 *
 * The WMMA form for every row count with the K range split across waves -- a NEW per-output order (the "part order"),
 * defined once and for every M, tile place and launch geometry:
 *   S = K / 128 slabs; P = gg_wk_parts(K) = the largest of 8, 4, 2 that divides S (1 otherwise) equal consecutive parts;
 *   part p's partial = gg_wm_kernel's arithmetic over the part's values alone (its WMMA chains a scale group, the group
 *   scale fma chain over the part's groups in ascending K from +0.0f), an f32;
 *   y = ((p0 + p1) + p2) + ... + p(P-1), f32 adds in ascending part order, rounded once to the output dtype.
 * P = 1 is gg_wm_kernel's order (the bytes of the stock kernel). Rows are independent in every operation, so a row's
 * bytes do not depend on M, on its place in a tile or on the geometry below.
 *
 * Geometry (a block = 8 waves, one 16-row tile of M): KP of the K range's P / KP-part runs side by side (KP = 1, 2, 4;
 * KP divides P), each on NS = 8 / KP waves of 32 weight rows (two 16-column tiles), so a block is 32 NS weight rows and
 * its waves walk S / KP slabs in step: the tile's 16 rows are staged at the KP runs' current slabs (the stock staging
 * with the M tiles replaced by the K runs: KP 16-byte chunks a thread a slab, the group sums by DPP), one barrier a
 * slab. A wave ends a part by folding it: run 0's waves keep the running total (they hold the prefix), the other
 * runs' waves store their parts' partials in LDS, and after the last slab run 0's waves add them in part order and
 * write the outputs. Two weights of one type and K (lever 1U-GQ's pairs) are one launch: the grid's first gy0 block
 * rows are the first weight's. */
struct GgWkArgs {
    const uint16_t* x[2];
    long long x_ld[2];
    const uint8_t* w[2];
    long long w_ld;
    void* y[2];
    long long y_ld[2];
    int N[2];
    int gy0;                             /* grid.y rows of weight 0 */
    int M, K, P;
    int S, SW, SP, PPW;                  /* slabs: of K, a wave's run, a part; parts a run */
    int pfm, pfl;                        /* weight prefetch: 0 the stock loop, 1 none, 2 unrolled (full waves); lines */
    long long psb;                       /* bytes a slab a weight row */
    /* Lever K1 folds a / e in the part-order form (PRO > 0, ds41-wmk2): the activation's producer computed in the
     * staging, from its own input in x[0] (PRO 1: rmsnorm's x [M, K]; PRO 2: swiglu_clamped's gate_up [M, 2K], the up
     * half at x + K), each value the producer's expression rounded to bf16 as the producer stores it (the bytes of
     * rmsnorm / swiglu_clamped then gemm_nt_ggml); grid column 0 also stores them to h, the producer's own output. */
    const float* g;                      /* PRO 1: the f32 gain [K] */
    float eps, wadd, mult;               /* PRO 1: rmsnorm's scalars as its launch reads them */
    int nw;                              /* PRO 1: the waves of rmsnorm's workgroup a row (r4d_act_threads(M, K) / 32) */
    float limit;                         /* PRO 2 */
    uint16_t* h;
    long long h_ld;
    int h_al;
};


constexpr int gg_wk_parts(long long K) {
    const long long S = K / kWmS;
    return S % 8 == 0 ? 8 : S % 4 == 0 ? 4 : S % 2 == 0 ? 2 : 1;
}

/* WL (lever WMK stage 2, G1): each wave stages ITS OWN run's 16 rows a slab into its own LDS region, single-buffered, so the
 * K loop has no block barrier (a wave-scope fence a slab); the staged values, the group sums (the same lanes, the same
 * DPP partners, the same order) and every operation after them are the block-staged form's, so the bytes are the same. */
template <int T, int KP, bool WL, typename OT, int PRO = 0>
__global__ __launch_bounds__(256) void gg_wk_kernel(GgWkArgs A) {
    using W = Wq<T>;
    using Pf = typename W::Pf;
    static_assert(W::TB == 0, "gg_wk_kernel: the table types are not built");
    constexpr int NS = 8 / KP;
    constexpr int BM = WL ? 16 : 16 * KP;      /* rows of a staged region (WL: a wave's own) */
    constexpr int NX = WL ? 8 : KP;            /* 16-byte chunks a thread stages a slab */
    constexpr int NB = WL ? 1 : 2;             /* staging buffers */
    constexpr int NR = WL ? 8 : 1;             /* staged regions (WL: one a wave) */
    constexpr int XE = BM * kWmXP;
    constexpr int XGE = W::XG > 0 ? kWmS / W::XG * BM : 0;
    extern __shared__ __attribute__((aligned(16))) uint8_t gg_smem[];
    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int kp = __builtin_amdgcn_readfirstlane(wave / NS), ns = __builtin_amdgcn_readfirstlane(wave % NS);
    const int mi = (int)blockIdx.y >= A.gy0 ? 1 : 0;
    const int n0 = ((int)blockIdx.y - (mi ? A.gy0 : 0)) * (32 * NS);
    const int N = A.N[mi];
    const int r0 = (int)blockIdx.x * 16, r1 = min(r0 + 16, A.M);
    const int wreg = WL ? __builtin_amdgcn_readfirstlane(wave) : 0;
    uint16_t* Xs = (uint16_t*)(void*)gg_smem + wreg * XE;
    float* Xg = (float*)(void*)(gg_smem + NR * NB * XE * 2) + wreg * XGE;
    float* Pp = (float*)(void*)(gg_smem + NR * NB * XE * 2 + NR * NB * XGE * 4);
    auto sync = [&]() {
        if constexpr (WL) {
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
        } else {
            lds_barrier();
        }
    };
    const int S = A.S, SW = A.SW, SP = A.SP, PPW = A.PPW;
    const int sb = kp * SW;                                         /* the wave's first slab */
    const uint8_t* wb = A.w[mi];
    const int h = lane >> 4, c16 = lane & 15;
    const uint8_t* wr[2];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const int n = n0 + ns * 32 + nt * 16 + c16;
        wr[nt] = wb + (long long)(n < N ? n : N - 1) * A.w_ld;
    }
    /* Activation staging: KP 16-byte chunks a thread a slab (the tile's row tid >> 4, values 8 (tid & 15), at run i's
     * slab); a row past M stages the tile's last row (rows are independent; its outputs are not written). WL: 8 chunks a
     * lane of the wave's own run (row (lane >> 4) + 2 i, values 8 (lane & 15)): a chunk's DPP partners (lane ^ 1, ^ 2) are
     * the same row's neighbouring chunks, as in the block staging. */
    const uint16_t* xsrc[NX];
    int xdo[NX], xgo[NX];
#pragma unroll
    for (int i = 0; i < NX; ++i) {
        const int row = WL ? (lane >> 4) + 2 * i : tid >> 4, kc = (tid & 15) * 8;
        const int run = WL ? kp : i, srow = WL ? row : i * 16 + row;
        xsrc[i] = A.x[mi] + (long long)min(r0 + row, r1 - 1) * A.x_ld[mi] + kc + (long long)run * SW * kWmS;
        xdo[i] = srow * kWmXP + kc;
        xgo[i] = ((tid & 15) * 8 / (W::XG > 0 ? W::XG : 1)) * BM + srow;
    }
    r4d_u32x4 xr[NX];
    /* PRO 1: the norm scale of each live row of the tile (mr_norm_scales' arithmetic = rmsnorm_body's: virtual wave vw's
     * lane sums columns 32 vw + lane, + 256, ... in order from 0, r4d_wave_sum, the eight sums in order, then nw - 8
     * zeros, rsqrtf(sum / K + eps)); the gain's 8 values of a chunk (the same columns in every WL chunk of a lane). */
    __shared__ float s_ws[PRO == 1 ? 16 : 1][8], s_sc[PRO == 1 ? 16 : 1];
    constexpr int NGQ = PRO == 1 ? (WL ? 1 : NX) : 1;
    float4 gq[NGQ][2];
    r4d_u32x4 xu[PRO == 2 ? NX : 1];   /* PRO 2: the up half's chunks */
    const bool hw = PRO > 0 && blockIdx.y == 0 && (!WL || ns == 0);   /* this thread's staged chunks go to h too */
    if constexpr (PRO == 1) {
        const int mc = r1 - r0;
        for (int pv = wave; pv < mc * 8; pv += 8) {
            const int m = pv >> 3, vw = pv & 7;
            const uint16_t* xr0 = A.x[0] + (long long)(r0 + m) * A.x_ld[0];
            float ss = 0.0f;
            for (int c = vw * 32 + lane; c < A.K; c += 256) {
                const float v = bf16_to_f32(xr0[c]);
                ss += v * v;
            }
            ss = r4d_wave_sum(ss);
            if (lane == 0) s_ws[m][vw] = ss;
        }
        lds_barrier();
        if (tid < mc) {
            float tt = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) tt += s_ws[tid][w];
            for (int w = 8; w < A.nw; ++w) tt += 0.0f;
            s_sc[tid] = rsqrtf(tt / (float)A.K + A.eps);
        }
        lds_barrier();
    }
    int xt = 0;   /* the run slab xr holds (the h column of a fold's chunk) */
    auto xload = [&](int t) {
        xt = t;
#pragma unroll
        for (int i = 0; i < NX; ++i) xr[i] = *(gg_gu128)(const void*)(xsrc[i] + t * kWmS);
        if constexpr (PRO == 2) {
#pragma unroll
            for (int i = 0; i < NX; ++i) xu[i] = *(gg_gu128)(const void*)(xsrc[i] + A.K + t * kWmS);
        }
        if constexpr (PRO == 1) {
#pragma unroll
            for (int i = 0; i < NGQ; ++i) {
                const int k = (WL ? kp : i) * SW * kWmS + t * kWmS + (tid & 15) * 8;
                gq[i][0] = *(const float4*)(const void*)(A.g + k);
                gq[i][1] = *(const float4*)(const void*)(A.g + k + 4);
            }
        }
    };
    /* the producer's bf16 octet of chunk i (row lr of the tile, columns k .. k + 7) in place of the raw chunk */
    auto produce = [&](int i, int lr) {
        if constexpr (PRO > 0) {
            uint16_t hb[8];
            if constexpr (PRO == 1) {
                const float sc = s_sc[lr];
                const float gv[8] = { gq[WL ? 0 : i][0].x, gq[WL ? 0 : i][0].y, gq[WL ? 0 : i][0].z, gq[WL ? 0 : i][0].w,
                                      gq[WL ? 0 : i][1].x, gq[WL ? 0 : i][1].y, gq[WL ? 0 : i][1].z, gq[WL ? 0 : i][1].w };
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const float x = (t & 1) ? bf16w_hi(xr[i][t >> 1]) : bf16w_lo(xr[i][t >> 1]);
                    hb[t] = f32_to_bf16(x * sc * (gv[t] + A.wadd) * A.mult);
                }
            } else {
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const float gg = fminf((t & 1) ? bf16w_hi(xr[i][t >> 1]) : bf16w_lo(xr[i][t >> 1]), A.limit);
                    const float uu = fminf(fmaxf((t & 1) ? bf16w_hi(xu[i][t >> 1]) : bf16w_lo(xu[i][t >> 1]), -A.limit), A.limit);
                    hb[t] = f32_to_bf16((gg / (1.0f + expf(-gg))) * uu);
                }
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) xr[i][q] = (uint32_t)hb[2 * q] | ((uint32_t)hb[2 * q + 1] << 16);
        }
    };
    auto xstore = [&](int b) {
        uint16_t* xs = Xs + b * XE;
#pragma unroll
        for (int i = 0; i < NX; ++i) {
            if constexpr (PRO > 0) {
                const int row = WL ? (lane >> 4) + 2 * i : tid >> 4;   /* the tile row this chunk stages */
                const int lr = min(row, r1 - r0 - 1);
                produce(i, lr);
                if (hw && row < r1 - r0) {   /* a live row's octet to h (a re-staged last slab writes the same bytes again) */
                    const int k = (WL ? kp : i) * SW * kWmS + xt * kWmS + (tid & 15) * 8;
                    uint16_t* d = A.h + (long long)(r0 + row) * A.h_ld + k;
                    if (A.h_al) {
                        *(r4d_u32x4*)(void*)d = xr[i];
                    } else {
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            d[2 * q] = (uint16_t)(xr[i][q] & 0xffffu);
                            d[2 * q + 1] = (uint16_t)(xr[i][q] >> 16);
                        }
                    }
                }
            }
            *(r4d_u32x4*)(void*)(xs + xdo[i]) = xr[i];
            if constexpr (W::XG > 0) {
                float sm = 0.0f;
#pragma unroll
                for (int k = 0; k < 4; ++k) sm += bf16w_lo(xr[i][k]) + bf16w_hi(xr[i][k]);
                sm += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(sm), 0xB1, 0xF, 0xF, false));
                if constexpr (W::XG == 32)
                    sm += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(sm), 0x4E, 0xF, 0xF, false));
                Xg[b * XGE + xgo[i]] = W::XGK * sm;
            }
        }
    };
    const uint8_t* prow = gg_sptr(wb + (long long)(n0 + ns * 32) * A.w_ld);
    const int prmax = __builtin_amdgcn_readfirstlane(N - 1 - (n0 + ns * 32));   /* rows past N: the last */
    const long long psb = A.psb;
    const int pfl = A.pfl, pfm = A.pfm;
    auto prefetch = [&](int s) {
        if (pfm == 1) return;
        const uint8_t* p = prow + (long long)min(s + 3, S - 1) * psb;
        if (pfm == 2 && prmax >= 31) {
#pragma unroll
            for (int i = 0; i < 32; ++i) __builtin_amdgcn_s_prefetch_data(p + (long long)i * A.w_ld, pfl);
            return;
        }
#pragma unroll 1
        for (int i = 0; i < 32; ++i) {
            __builtin_amdgcn_s_prefetch_data(p, pfl);
            if (i < prmax) p += A.w_ld;
        }
    };
    const uint16_t* xa = Xs + ((WL ? 0 : kp * 16) + c16) * kWmXP;
    const float* xgl = Xg + 8 * h + (WL ? 0 : kp * 16);
    gg_v8f acc[2], tot[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        acc[j] = (gg_v8f)(0.0f);
        tot[j] = (gg_v8f)(0.0f);
    }
    uint32_t rE[2][2][W::RAW], rR[2][W::RAW];
    xload(0);
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        uint32_t t[W::RAW];
        W::load(wr[nt], sb, h, t);
#pragma unroll
        for (int k = 0; k < W::RAW; ++k) {
            if ((W::EARLY >> k) & 1u) rE[0][nt][k] = t[k];
            else rR[nt][k] = t[k];
        }
    }
    xstore(0);
    xload(1 < SW ? 1 : SW - 1);
    r4d_u32x4 bf[2][2];
    Pf pf[2];
    sync();
    {
        uint32_t rc[2][W::RAW];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int k = 0; k < W::RAW; ++k) rc[nt][k] = ((W::EARLY >> k) & 1u) ? rE[0][nt][k] : rR[nt][k];
        if constexpr (W::PIPE) {
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) W::template pre<0>(rc[nt], h, gg_smem, pf[nt]);
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) bf[0][nt] = W::template bfrag<0>(rc[nt], h, pf[nt], sb);
        }
    }
    int pleft = SP, pgp = kp * PPW;   /* slabs left in the wave's current part, the part's index */
    /* Slab t of the wave's run (the K range's slab s = sb + t) from buffer / raw set PB = t & 1: gg_wm_kernel's slab. */
    auto slab = [&](auto pc, int t) {
        constexpr int PB = decltype(pc)::value;
        const int s = sb + t;
        uint32_t rc[2][W::RAW];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int k = 0; k < W::RAW; ++k) rc[nt][k] = ((W::EARLY >> k) & 1u) ? rE[PB][nt][k] : rR[nt][k];
        const int sn = t + 1 < SW ? s + 1 : s;
        if constexpr (PB == 0) prefetch(s);
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            uint32_t tw[W::RAW];
            W::load(wr[nt], sn, h, tw);
#pragma unroll
            for (int k = 0; k < W::RAW; ++k)
                if ((W::EARLY >> k) & 1u) rE[PB ^ 1][nt][k] = tw[k];
        }
        const uint16_t* xb = xa + (WL ? 0 : PB * XE);
        const float* xg = xgl + (WL ? 0 : PB * XGE);
        gg_v8f P[2];
        r4d_u32x4 af[2];
        gg_v8f ci;
        af[0] = *(const r4d_u32x4*)(const void*)(xb + W::xoff(0, h));
        if constexpr (W::CINIT && W::PIPE) ci = *(const gg_v8f*)(const void*)(xg);
        gg_for<8>([&](auto jc) {
            constexpr int J = decltype(jc)::value, C = J & 1;
            if constexpr (!W::PIPE) {
                if constexpr (J < 7) af[C ^ 1] = *(const r4d_u32x4*)(const void*)(xb + W::xoff(J + 1, h));
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) bf[C][nt] = W::template bfrag<J>(rc[nt], h, pf[nt], s);
            } else if constexpr (J < 7) {
                af[C ^ 1] = *(const r4d_u32x4*)(const void*)(xb + W::xoff(J + 1, h));
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) W::template pre<J + 1>(rc[nt], h, gg_smem, pf[nt]);
            } else {
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
                    for (int k = 0; k < W::RAW; ++k) rc[nt][k] = ((W::EARLY >> k) & 1u) ? rE[PB ^ 1][nt][k] : rR[nt][k];
                    W::template pre<0>(rc[nt], h, gg_smem, pf[nt]);
                }
            }
            gg_v8f g8;
            if constexpr (W::MINS && W::PIPE && J % W::SPG == W::SPG - 1)
                g8 = *(const gg_v8f*)(const void*)(xg + (J * 16 / W::XG) * BM);
            if constexpr (J % W::SPG == 0) {
                gg_v8f c0 = (gg_v8f)(0.0f);
                if constexpr (W::CINIT && W::PIPE) c0 = ci;
                else if constexpr (W::CINIT) c0 = *(const gg_v8f*)(const void*)(xg + (J * 16 / W::XG) * BM);
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) P[nt] = gg_wmma(af[C], bf[C][nt], c0);
            } else {
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) P[nt] = gg_wmma(af[C], bf[C][nt], P[nt]);
            }
            if constexpr (W::CINIT && W::PIPE && J < 7 && (J + 1) % W::SPG == 0)
                ci = *(const gg_v8f*)(const void*)(xg + ((J + 1) * 16 / W::XG) * BM);
            if constexpr (W::PIPE) {
#pragma unroll
                for (int nt = 0; nt < 2; ++nt)
                    bf[C ^ 1][nt] = W::template bfrag<(J + 1) & 7>(rc[nt], h, pf[nt], J < 7 ? s : sn);
            }
            if constexpr (J % W::SPG == W::SPG - 1) {
                float sc[2], mn[2];
                if constexpr (J < 7 || !W::PIPE) {
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        sc[nt] = W::scale(rc[nt], J / W::SPG, s);
                        if constexpr (W::MINS) mn[nt] = W::mins(rc[nt], J / W::SPG, s);
                    }
                } else {
#pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        uint32_t ro[W::RAW];
#pragma unroll
                        for (int k = 0; k < W::RAW; ++k) ro[k] = ((W::EARLY >> k) & 1u) ? rE[PB][nt][k] : rR[nt][k];
                        sc[nt] = W::scale(ro, J / W::SPG, s);
                        if constexpr (W::MINS) mn[nt] = W::mins(ro, J / W::SPG, s);
                    }
                }
                if constexpr (W::MINS && !W::PIPE) g8 = *(const gg_v8f*)(const void*)(xg + (J * 16 / W::XG) * BM);
#pragma unroll
                for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                    for (int q = 0; q < 8; ++q) {
                        acc[nt][q] = __builtin_fmaf(P[nt][q], sc[nt], acc[nt][q]);
                        if constexpr (W::MINS) acc[nt][q] = __builtin_fmaf(g8[q], mn[nt], acc[nt][q]);
                    }
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) gg_pin8(acc[nt]);
            }
            if constexpr (W::ROLL) {
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) W::template reload<J>(wr[nt], sn, h, rR[nt]);
            }
            if constexpr (J == 3 && !WL) {
                xstore(PB ^ 1);
                xload(t + 2 < SW ? t + 2 : SW - 1);
            }
            __builtin_amdgcn_sched_barrier(0);
        });
        if constexpr (WL) {   /* the one buffer: slab t + 1 in once slab t's reads are issued (a wave's LDS ops run in order) */
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
            xstore(0);
            sync();
            xload(t + 2 < SW ? t + 2 : SW - 1);
        } else {
            lds_barrier();
        }
        /* A part ends with this slab: fold it (run 0 holds the prefix; the others keep their partials for run 0). */
        if (--pleft == 0) {
            const int gp = pgp;
            pleft = SP;
            ++pgp;
            if (kp == 0) {
#pragma unroll
                for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                    for (int q = 0; q < 8; ++q) tot[nt][q] = gp == 0 ? acc[nt][q] : tot[nt][q] + acc[nt][q];
            } else if constexpr (KP > 1) {
                float* pp = Pp + ((gp - PPW) * NS * 32 + ns * 32 + lane) * 16;
                *(gg_v8f*)(void*)pp = acc[0];
                *(gg_v8f*)(void*)(pp + 8) = acc[1];
            }
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                acc[nt] = (gg_v8f)(0.0f);
                gg_pin8(tot[nt]);
            }
        }
    };
    int t = 0;
    for (; t + 1 < SW; t += 2) {
        slab(std::integral_constant<int, 0>{}, t);
        slab(std::integral_constant<int, 1>{}, t + 1);
    }
    if (t < SW) slab(std::integral_constant<int, 0>{}, t);
    if constexpr (KP > 1) {
        lds_barrier();
        if (kp != 0) return;
        for (int gp = PPW; gp < A.P; ++gp) {
            const float* pp = Pp + ((gp - PPW) * NS * 32 + ns * 32 + lane) * 16;
            const gg_v8f a0 = *(const gg_v8f*)(const void*)pp, a1 = *(const gg_v8f*)(const void*)(pp + 8);
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                tot[0][q] = tot[0][q] + a0[q];
                tot[1][q] = tot[1][q] + a1[q];
            }
        }
    }
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const int n = n0 + ns * 32 + nt * 16 + c16;
        if (n >= N) continue;
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const int row = r0 + 8 * h + q;
            if (row >= r1) continue;
            gg_put((OT*)A.y[mi] + (long long)row * A.y_ld[mi] + n, tot[nt][q]);
        }
    }
}

/* gg_wk_kernel's launch (ds41-wmk, bench). g_gg_wk, the bench pin r4d_gg_wk_pin (no engine path sets it): 0 off; else
 * bits 0-3 the KP cap (1, 2, 4), bit 4 P = 1 (the stock kernel's order: its bytes), bit 5 KP = the cap whenever the
 * parts allow it, bits 8-15 the block target of the rule (0: 32): the smallest KP whose grid has at least that many
 * blocks, else the largest allowed; bit 6 no weight prefetch, bit 17 the unrolled prefetch; bit 7 the launch asks for
 * 64 KiB of LDS (one block a WGP); bit 16 a grid of more than 64 KP = 1 blocks takes the cap (the head). Stage 2 (ds41-wmk2):
 * bit 18 WL -- each wave stages its own run's slab (no block barrier in the K loop), and a cap of 8 (bits 0-3 >= 8) allows
 * KP 8; bit 19 the rule counts the blocks of ONE 16-row tile (so M 17..48 keep M <= 16's KP instead of dropping to KP 1).
 * Every code computes the same part order: the bytes do not depend on the code. */
std::atomic<long long> g_gg_wk_count{0};
std::atomic<int> g_gg_wk_last{0};
template <int T>
constexpr unsigned gg_wk_lds(int kp, int P, bool wl = false) {
    if (wl)   /* a region a wave, single-buffered */
        return (unsigned)(8 * (16 * kWmXP * 2 + (Wq<T>::XG > 0 ? (kWmS / Wq<T>::XG) * 16 * 4 : 0)) +
                          (kp > 1 ? (P - P / kp) * (8 / kp) * 32 * 64 : 0));
    return (unsigned)(2 * 16 * kp * kWmXP * 2 + (Wq<T>::XG > 0 ? 2 * (kWmS / Wq<T>::XG) * 16 * kp * 4 : 0) +
                      (kp > 1 ? (P - P / kp) * (8 / kp) * 32 * 64 : 0));
}
template <int T>
int gg_launch_wk_t(RadStream s, int code, int kp, bool f32out, int pro, const GgWkArgs& A, dim3 grid) {
    const bool wl = (code & 262144) != 0;
    const unsigned lds = (code & 128) ? 65536u : gg_wk_lds<T>(kp, A.P, wl);
    if (pro) {   /* the folds (bf16 out): the (KP, WL) their K allows (q_b K 1280: P 2; shexp down 768 / 512: P 2 / 4) */
        if (f32out) return R4D_WK_DECLINE;
#define GG_WK_F(KPV, WLV, PV)                                                                                     \
        if (kp == KPV && wl == WLV && pro == PV) return R4D_EMIT(s, (gg_wk_kernel<T, KPV, WLV, uint16_t, PV>), grid, dim3(256), lds, A);
        GG_WK_F(1, false, 1) GG_WK_F(2, false, 1) GG_WK_F(1, true, 1) GG_WK_F(2, true, 1)
        GG_WK_F(1, false, 2) GG_WK_F(2, false, 2) GG_WK_F(4, false, 2) GG_WK_F(1, true, 2) GG_WK_F(2, true, 2) GG_WK_F(4, true, 2)
#undef GG_WK_F
        return R4D_WK_DECLINE;
    }
#define GG_WK_X(KPV, WLV)                                                                              \
    if (kp == KPV && wl == WLV) {                                                                      \
        if (f32out) return R4D_EMIT(s, (gg_wk_kernel<T, KPV, WLV, float>), grid, dim3(256), lds, A);   \
        return R4D_EMIT(s, (gg_wk_kernel<T, KPV, WLV, uint16_t>), grid, dim3(256), lds, A);            \
    }
    GG_WK_X(1, false)
    GG_WK_X(2, false)
    GG_WK_X(4, false)
    GG_WK_X(1, true)
    GG_WK_X(2, true)
    GG_WK_X(4, true)
    GG_WK_X(8, true)
#undef GG_WK_X
    return R4D_NO(RAD_E_SHAPE);
}
/* 1 or 2 weights (nm) of one type at one K over the same M rows under the launch code `code` (g_gg_wk's bits: the bench
 * pin's value, or kWmkCode for the engine's lever WMK); R4D_WK_DECLINE when the form does not take the call. fo (nm 1
 * only): a fold's producer in the staging (lever WMK's folds a / e); a fold it declines runs as its two launches, the
 * same bytes. */

}  // namespace

int r4d_gg_wk_run(RadStream s, int code, int ty, bool f32out, int nm, const uint16_t* const* xd, const long long* x_ld,
              const int* x_al, const uint8_t* const* wd, long long w_ld, void* const* yd, const long long* y_ld,
              const long long* Nv, long long M, long long K, const GgWkFold* fo) {
    if (!code || (ty != GG_Q5_K && ty != GG_Q6_K && ty != GG_Q8_0) || K % (2 * kWmS) || M > 0x7fffff)
        return R4D_WK_DECLINE;
    if (fo && (nm != 1 || f32out || (fo->pro == 1 && (!fo->g || ((uintptr_t)fo->g & 15u))) || !fo->h))
        return R4D_WK_DECLINE;
    for (int i = 0; i < nm; ++i)
        if (!x_al[i]) return R4D_WK_DECLINE;
    const bool wl = (code & 262144) != 0;   /* bit 18: G1's wave-local staging (KP up to 8) */
    const int cap = (code & 15) >= 8 && wl ? 8 : (code & 15) >= 4 ? 4 : (code & 15) >= 2 ? 2 : 1;
    const int P = (code & 16) ? 1 : gg_wk_parts(K);
    const long long target = ((code >> 8) & 255) ? ((code >> 8) & 255) : 32;
    const long long gx = (M + 15) / 16;
    const long long gxr = (code & 524288) ? 1 : gx;   /* bit 19: the rule counts a 16-row tile's blocks (KP as at M <= 16) */
    auto blocks = [&](int kp) {
        long long b = 0;
        for (int i = 0; i < nm; ++i) b += (Nv[i] + 256 / kp - 1) / (256 / kp);
        return b * gxr;
    };
    int kp = 1;
    if (code & 32) {
        while (kp * 2 <= cap && P % (kp * 2) == 0) kp *= 2;
    } else {
        while (blocks(kp) < target && kp * 2 <= cap && P % (kp * 2) == 0) kp *= 2;
    }
    if ((code & 65536) && blocks(1) > 64)
        while (kp * 2 <= cap && P % (kp * 2) == 0) kp *= 2;
    while (kp > 1 && gg_wk_lds<GG_Q6_K>(kp, P, wl) > 65536u) kp /= 2;
    GgWkArgs A{};
    long long gy = 0;
    for (int i = 0; i < 2; ++i) {
        const int j = i < nm ? i : 0;
        A.x[i] = xd[j]; A.x_ld[i] = x_ld[j]; A.w[i] = wd[j]; A.y[i] = yd[j]; A.y_ld[i] = y_ld[j]; A.N[i] = (int)Nv[j];
        if (i < nm) {
            if (i == 1) A.gy0 = (int)gy;
            gy += (Nv[i] + 256 / kp - 1) / (256 / kp);
        }
    }
    if (nm == 1) A.gy0 = (int)gy;
    A.w_ld = w_ld;
    A.M = (int)M; A.K = (int)K; A.P = P;
    A.S = (int)(K / kWmS); A.SW = A.S / kp; A.SP = A.S / P; A.PPW = P / kp;
    A.psb = w_ld / A.S;
    A.pfl = (int)((2 * A.psb + 127) / 128 < 31 ? (2 * A.psb + 127) / 128 : 31);
    A.pfm = (code & 64) ? 1 : (code & 131072) ? 2 : 0;
    if (fo) {
        A.g = fo->g; A.eps = fo->eps; A.wadd = fo->wadd; A.mult = fo->mult; A.nw = fo->nw; A.limit = fo->limit;
        A.h = fo->h; A.h_ld = fo->h_ld; A.h_al = fo->h_al;
    }
    if (gy > 65535 || gx > 65535) return R4D_WK_DECLINE;
    g_gg_wk_count.fetch_add(1, std::memory_order_relaxed);
    g_gg_wk_last.store(kp * 100 + P, std::memory_order_relaxed);
    const dim3 grid((unsigned)gx, (unsigned)gy);
    switch (ty) {
    case GG_Q8_0: return gg_launch_wk_t<GG_Q8_0>(s, code, kp, f32out, fo ? fo->pro : 0, A, grid);
    case GG_Q6_K: return gg_launch_wk_t<GG_Q6_K>(s, code, kp, f32out, fo ? fo->pro : 0, A, grid);
    default: return gg_launch_wk_t<GG_Q5_K>(s, code, kp, f32out, fo ? fo->pro : 0, A, grid);
    }
}

extern "C" long long r4d_gg_wk_count() { return g_gg_wk_count.load(std::memory_order_relaxed); }
extern "C" int r4d_gg_wk_last() { return g_gg_wk_last.load(std::memory_order_relaxed); }

/* ------------------------------------------------------------------ the prefill rows (G3)
 * r4d_gg_wkpf_run: y[M, N] = x[M, K] . w[N, K]^T above the decode band under ds41_wmk 1, in the part order, on lever GPF's form
 * `form` (the one gg_pf would pick, 0 = the stock geometry 322); a form not built here with the part fold runs 322. The
 * geometry moves time, never bytes (each output's operations are the part order's in every form). */
namespace {
#define WKPF_FORMS(X) X(3, 2, 8, 0) X(4, 2, 8, 0) X(4, 2, 8, 1) X(3, 2, 8, 1) X(3, 2, 4, 0) X(2, 2, 8, 0) \
    X(4, 1, 16, 0) X(4, 1, 16, 1) X(2, 1, 16, 0)   /* one column tile a wave, 16 waves: half the accumulators a lane (the fold's total fits) */
/* The running total in registers (measured 14.11., --perfwkpf, one card: the scratch-held total TS costs +30..110 % a launch,
 * more than the registers' spills). g_wkpf_ts (bench pin r4d_gg_wkpf_ts): 0 / 1 registers, 2 scratch.
 * The form: 322 (the stock geometry; the fastest part-order form at most dense shapes, 14.11.), whatever gg_pf picks; the bench
 * pin r4d_gg_wkpf_follow(1) makes it follow gg_pf's pick (the form timing). */
int g_wkpf_ts = 0;
int g_wkpf_follow = 0;
int g_wkpf_default = -1;   /* bench pin r4d_gg_wkpf_default: the form W's prefill rows run (0 = the stock kernel + fold; -1 the table) */
/* W's prefill form by (type, N, K) (the table below; every form computes the same bytes). Filled from --perfwkpf (one card, M 2048 /
 * 4096); a shape not listed runs DEFAULT_FORM. */
constexpr int kWkpfDefault = 322;
struct WkpfEnt { int ty, N, K, form; };
const WkpfEnt kWkpfTab[] = { { -1, 0, 0, 0 } };
int wkpf_form_tab(int ty, long long N, long long K) {
    for (const WkpfEnt& e : kWkpfTab)
        if (e.ty == ty && e.N == N && e.K == K) return e.form;
    return kWkpfDefault;
}
template <int T, int MT, int NT, int WN, int FI>
int wkpf_launch_f(RadStream s, bool f32out, const GgTiledArgs& A) {
    const dim3 grid((unsigned)((A.M + 16 * MT - 1) / (16 * MT)), (unsigned)((A.N + 16 * NT * WN - 1) / (16 * NT * WN)));
    if (grid.y > 65535u) return R4D_WK_DECLINE;
    const unsigned lds = pf_lds<T>(MT);
    const bool ts = g_wkpf_ts == 2;
    if (ts) {
        if (f32out) return R4D_EMIT(s, (gg_pf_kernel<T, MT, NT, WN, float, 8, FI>), grid, dim3(32 * WN), lds, A);
        return R4D_EMIT(s, (gg_pf_kernel<T, MT, NT, WN, uint16_t, 8, FI>), grid, dim3(32 * WN), lds, A);
    }
    if (f32out) return R4D_EMIT(s, (gg_pf_kernel<T, MT, NT, WN, float, 0, FI>), grid, dim3(32 * WN), lds, A);
    return R4D_EMIT(s, (gg_pf_kernel<T, MT, NT, WN, uint16_t, 0, FI>), grid, dim3(32 * WN), lds, A);
}
template <int T>
int wkpf_launch_t(RadStream s, int code, bool f32out, const GgTiledArgs& A) {
    if (code == 0) {   /* the stock WMMA kernel (gg_wm_kernel, MT 3) with the part fold */
        const dim3 grid((unsigned)((A.M + 47) / 48), (unsigned)((A.N + kWmBN - 1) / kWmBN));
        if (grid.y > 65535u) return R4D_WK_DECLINE;
        const unsigned lds = pf_lds<T>(3);
        if (f32out) return R4D_EMIT(s, (gg_wm_kernel<T, 3, false, float>), grid, dim3(256), lds, A);
        return R4D_EMIT(s, (gg_wm_kernel<T, 3, false, uint16_t>), grid, dim3(256), lds, A);
    }
#define WKPF_L(mt, nt, wn, fi) if (code == pf_code(mt, nt, wn, fi, 0)) return wkpf_launch_f<T, mt, nt, wn, fi>(s, f32out, A);
    WKPF_FORMS(WKPF_L)
#undef WKPF_L
    return wkpf_launch_f<T, 3, 2, 8, 0>(s, f32out, A);   /* 322, the stock geometry */
}
std::atomic<int> g_wkpf_last{0};
}  // namespace

int r4d_gg_wkpf_run(RadStream s, int ty, bool f32out, const uint16_t* x, long long x_ld, int x_al, const uint8_t* w,
                    long long w_ld, void* y, long long y_ld, long long M, long long N, long long K, int form) {
    if ((ty != GG_Q5_K && ty != GG_Q6_K && ty != GG_Q8_0) || !x_al || K % (2 * kWmS) || M < 1 || M > 0x7fffffff || N > 0x7fffffff)
        return R4D_WK_DECLINE;
    GgTiledArgs A{};
    A.x = x; A.x_ld = x_ld;
    A.w = w; A.wtab = nullptr; A.w_ld = w_ld; A.w_es = 0;
    A.y = y; A.y_ld = y_ld;
    A.M = (int)M; A.N = (int)N; A.K = (int)K;
    A.x_al = 1; A.tab_bytes = 0;
    if (!g_wkpf_follow) form = g_wkpf_default >= 0 ? g_wkpf_default : wkpf_form_tab(ty, N, K);
    g_wkpf_last.store(form, std::memory_order_relaxed);
    switch (ty) {
    case GG_Q8_0: return wkpf_launch_t<GG_Q8_0>(s, form, f32out, A);
    case GG_Q6_K: return wkpf_launch_t<GG_Q6_K>(s, form, f32out, A);
    default: return wkpf_launch_t<GG_Q5_K>(s, form, f32out, A);
    }
}
extern "C" int r4d_gg_wkpf_last() { return g_wkpf_last.load(std::memory_order_relaxed); }
extern "C" void r4d_gg_wkpf_ts(int v) { g_wkpf_ts = v == 1 || v == 2 ? v : 0; }
extern "C" void r4d_gg_wkpf_follow(int on) { g_wkpf_follow = on ? 1 : 0; }
extern "C" void r4d_gg_wkpf_default(int form) { g_wkpf_default = form; }
