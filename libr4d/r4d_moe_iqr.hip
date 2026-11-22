/* r4d_moe_iqr.hip -- the routed mixture of experts over IQR weights (r4d_iqr.h), decoded from
 * their codes where they are read. The grouped GEMV of the decode band: one block per (expert,
 * 16-row weight tile group); a wave a 16-row weight tile, its lanes across K, each lane's
 * decoded values of a row one sudot4 against a token's int8 activation, folded per (row, 128 K)
 * weight scale x activation scale. A run of `R` sorted rows is handed to a block in groups of
 * MT tokens; the K loop repeats the decode per group (the prefill tile decodes each unit once,
 * phase 2).
 *
 * Operands are moe_gemm_q's grouping: `sorted_tok`, `expert_offset`, and `a_order` "token"
 * (gather sorted_tok[i] / top_k, gate_up) or "sorted" (read i, the down projection). The
 * weights are WEIGHT TABLES of IQR planes (one entry an expert), scales the matching tables; a
 * stacked plane's stride carries the entry pitch in bytes. The activation is had_quant_act_i8g's
 * int8 codes (rotated fwht128) with an f32 scale per (row, 128).
 *
 * V1: one table, one geometry, parts = 1, decode band only; parity classes and the prefill tile
 * are phase 2/3 (iqr-moe/PLAN.md).
 */
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_fp8_wmma.h"
#include "r4d_iqr.h"
#include "r4d_iqr_dec.h"

#include <algorithm>
#include <cstring>

namespace {

using namespace r4d_fp8;
using namespace r4d_iqr;
using namespace r4d_iqr_dev;

constexpr uint32_t kWave = 32, kBlock = 128, kFK = 256;
constexpr uint32_t kGW = 8;    /* waves a block: 128 weight rows */

/* An entry of a weight-table operand (r4d_moe.hip, wtab_at): `tab` is the pointer array when
 * `stride0` is 0, else the base of a stacked plane. */
__device__ __forceinline__ const uint8_t* wtab_at(const void* const* tab, size_t stride0,
                                                  uint32_t e) {
    if (stride0 == 0) return (const uint8_t*)tab[e];
    return (const uint8_t*)tab + (size_t)e * stride0;
}

/* ---- the decode GEMV ----------------------------------------------------------------------- */
template <class F, uint32_t MT>
__global__ __launch_bounds__(kGW * kWave) R4D_WAVES(6)
void moe_iqr_gemv_kernel(Ctx c0, const int8_t* __restrict__ X, const float* __restrict__ XS,
                         const void* const* __restrict__ Wtab,
                         const void* const* __restrict__ Stab,
                         size_t w_s0, size_t s_s0,
                         const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
                         uint16_t* __restrict__ Y, uint32_t T, uint32_t N, uint32_t n_expert,
                         uint32_t top_k, uint32_t a_ld, uint32_t xs_ld, uint32_t y_ld,
                         uint32_t a_sorted) {
    using D = Dec<F, true>;
    __shared__ __attribute__((aligned(16))) int8_t s_tab[D::kTab];
    const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1);
    const uint32_t wave = __builtin_amdgcn_readfirstlane(tid / kWave);

    const uint32_t e = blockIdx.y;
    if (e >= n_expert) return;
    const uint32_t rs = off[e];
    const uint32_t re0 = off[e + 1];
    const uint32_t re = re0 < T ? re0 : T;

    Ctx c = c0;
    c.tab = s_tab;
    c.W = wtab_at(Wtab, w_s0, e);
    c.S = wtab_at(Stab, s_s0, e);
    D::stage(s_tab, c, tid, kGW * kWave);
    __syncthreads();
    if (rs >= re) return;                       /* an expert with no routed slot */

    const uint32_t rt = blockIdx.x * kGW + wave;   /* this wave's 16-row weight tile */
    const bool alive = rt * 16u < N;
    const uint32_t myrow = rt * 16u + (lane & 15u);

    /* A run of R sorted rows against this 16-row weight tile, in groups of MT tokens. The
     * weight stream is fetched a block ahead, as the dense GEMV. */
    for (uint32_t g0 = 0; g0 < re - rs; g0 += MT) {
        const uint32_t mt = min(MT, re - rs - g0);
        int32_t tok[MT];
#pragma unroll
        for (uint32_t m = 0; m < MT; ++m) {
            tok[m] = -1;
            if (m < mt) {
                const int32_t f = sorted[rs + g0 + m];
                tok[m] = f < 0 ? -1 : (int32_t)(a_sorted ? (rs + g0 + m) : (uint32_t)f / top_k);
            }
        }

        float acc[16][MT];
#pragma unroll
        for (uint32_t q = 0; q < 16; ++q)
#pragma unroll
            for (uint32_t m = 0; m < MT; ++m) acc[q][m] = 0.0f;

        typename D::Raw cur = D::fetch(c, rt, 0u, lane, alive), nxt;
        for (uint32_t kb = 0; kb < c.nkb; ++kb) {
            if (kb + 1u < c.nkb) nxt = D::fetch(c, rt, kb + 1u, lane, alive);
            uint32_t xv[MT];
            float as[MT];
#pragma unroll
            for (uint32_t m = 0; m < MT; ++m) {
                const bool live = m < mt && tok[m] >= 0;
                xv[m] = live ? *(const uint32_t*)(X + (size_t)tok[m] * a_ld + kb * kBlock + 4u * lane)
                             : 0u;
                as[m] = live ? XS[(size_t)tok[m] * xs_ld + kb] : 0.0f;
            }
            /* Row (lane & 15)'s fold scale and tcq_rb ratio, read by every lane through
             * readlane (lane q holds row q's). */
            const float fs = alive ? fold_f32<F>(c, myrow, kb) : 0.0f;
            const float ratio = alive ? rb_ratio<F>(c, myrow, kb) : 0.0f;
            uint32_t w[16];
            /* A computed trellis off the tcq_rb grid decodes each value as its byte sum, the
             * value plus 126: one sudot4 of the unsigned sums less 126 times the activation
             * sum (r4d_gemm_iqr.hip's GEMV). */
            constexpr bool UNS = D::MAD && F::LRB == 0;
            [[maybe_unused]] int xsum[MT];
            if constexpr (UNS) {
#pragma unroll
                for (uint32_t m = 0; m < MT; ++m)
                    xsum[m] = 126 * __builtin_amdgcn_sudot4(true, (int)xv[m], false, 0x01010101,
                                                            0, false);
            }
            D::template rows<UNS>(cur, c, ratio, w);
            D::unzero(c, rt, kb, w);
#pragma unroll
            for (uint32_t q = 0; q < 16; ++q) {
                const float wsq = __builtin_bit_cast(float,
                                                     __builtin_amdgcn_readlane(__float_as_int(fs), q));
#pragma unroll
                for (uint32_t m = 0; m < MT; ++m) {
                    if (m >= mt) continue;
                    int d = __builtin_amdgcn_sudot4(!UNS, (int)w[q], true, (int)xv[m], 0, false);
                    if constexpr (UNS) d -= xsum[m];
                    acc[q][m] = fmaf((float)d, wsq * as[m], acc[q][m]);
                }
            }
            cur = nxt;
        }
#pragma unroll
        for (uint32_t q = 0; q < 16; ++q)
#pragma unroll
            for (uint32_t m = 0; m < MT; ++m) acc[q][m] = r4d_wave_sum(acc[q][m]);
        if constexpr (F::SC == kRowF32) {
            const float rq = alive ? row_scale<F>(c, myrow) : 0.0f;
#pragma unroll
            for (uint32_t q = 0; q < 16; ++q) {
                const float rsq = __builtin_bit_cast(float,
                                                     __builtin_amdgcn_readlane(__float_as_int(rq), q));
#pragma unroll
                for (uint32_t m = 0; m < MT; ++m) acc[q][m] *= rsq;
            }
        }
        if (lane < 16u && alive) {
#pragma unroll
            for (uint32_t q = 0; q < 16; ++q)
                if (lane == q) {
#pragma unroll
                    for (uint32_t m = 0; m < MT; ++m) {
                        if (m >= mt) continue;
                        const size_t o = (size_t)(rs + g0 + m) * y_ld + rt * 16u + q;
                        Y[o] = tok[m] < 0 ? 0u : f32_to_bf16(acc[q][m]);
                    }
                }
        }
    }
}

/* ---- the prefill tile (phase 2): a run of sorted rows through the int8 narrow tile ----------
 *
 * r4d_gemm_iqr.hip's iqr_m64_kernel, over one expert's run of sorted rows: blockIdx.x the
 * 128-weight-row tile, blockIdx.y the 64 sorted rows, blockIdx.z the expert. A weight unit is
 * decoded once per tile (not once per token group), its values through the wave's LDS fragment
 * order, and the activation rows are gathered through sorted_tok: each row is its token's int8
 * codes, the scale its own. The output lands row-major in sorted order. A run shorter than the
 * tile leaves the tile's dead rows zeroed. No K split: a block is the whole K. */
template <class F, uint32_t NT>
__global__ __launch_bounds__(8u * kWave) R4D_WAVES(4)
void moe_iqr_tile_kernel(Ctx c0, const void* const* __restrict__ Wtab,
                         const void* const* __restrict__ Stab, size_t w_s0, size_t s_s0,
                         const int8_t* __restrict__ X, const float* __restrict__ XS,
                         const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
                         uint16_t* __restrict__ Y, uint32_t N, uint32_t T, uint32_t n_expert,
                         uint32_t top_k, uint32_t a_ld, uint32_t xs_ld, uint32_t y_ld,
                         uint32_t a_sorted) {
    using D = Dec<F>;
    constexpr uint32_t MW = 8, FK = kFK, FN = NT * 16, FM = MW * 16, kU = FK / kBlock;
    __shared__ uint8_t s_b[FN * lds_pitch<FK>()];
    __shared__ __attribute__((aligned(16))) int8_t s_tab[D::kTab];
    __shared__ __attribute__((aligned(16))) uint8_t s_tr[MW * kTrBytes];

    const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const uint32_t e = blockIdx.z;
    if (e >= n_expert) return;
    Ctx c = c0;
    c.W = wtab_at(Wtab, w_s0, e);
    c.S = wtab_at(Stab, s_s0, e);
    c.tab = s_tab;
    uint8_t* const tr = s_tr + wave * kTrBytes;
    D::stage(s_tab, c, tid, MW * kWave);

    const uint32_t cols = (uint32_t)c.nkb * kBlock;
    const uint32_t rs = (uint32_t)off[e], re0 = (uint32_t)off[e + 1];
    const uint32_t re = re0 < T ? re0 : T;
    const uint32_t m0 = blockIdx.x * FM, n0 = blockIdx.y * FN;
    if (rs + n0 >= re) return;                 /* this tile's rows beyond the run */
    const uint32_t wrow0 = m0 + wave * 16u;    /* the wave's 16 weight rows */
    const bool alive = wrow0 < N;

    f8v raw[NT], out[NT];
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) { raw[j] = acc_zero(); out[j] = acc_zero(); }

    /* The activation staging: kTPR threads a row; the token of a row is sorted_tok's. */
    constexpr uint32_t kSeg = 32, kTPR = FK / kSeg, kRPP = (MW * kWave) / kTPR;
    constexpr uint32_t kBP = (FN + kRPP - 1) / kRPP;
    const uint32_t sn = tid / kTPR, sk = (tid % kTPR) * kSeg;
    const uint32_t brow0 = n0 + frag_c(lane);
    const uint32_t arow = wrow0 / 16u;
    Stage8 pb[kBP];
#pragma unroll
    for (uint32_t p = 0; p < kBP; ++p) {
        const uint32_t r = sn + p * kRPP, g = rs + n0 + r;
        const bool live = sk + kSeg <= cols && g < re;
        const int32_t f = g < re ? sorted[g] : 0;
        const uint32_t tok = a_sorted ? g : (uint32_t)(f < 0 ? 0 : (int32_t)f / (int32_t)top_k);
        const uint8_t* bsrc = (const uint8_t*)X + (size_t)tok * a_ld + sk;
        pb[p] = stage8_load(bsrc, live);
    }
    typename D::Raw pf[kU], pfn[kU];
#pragma unroll
    for (uint32_t p = 0; p < kU; ++p) pf[p] = D::fetch(c, arow, p, lane, alive && p < c.nkb);

    for (uint32_t c0k = 0; c0k < cols; c0k += FK) {
#pragma unroll
        for (uint32_t p = 0; p < kBP; ++p) {
            const uint32_t r = sn + p * kRPP;
            if (r < FN) stage8_store<FK>(s_b, r, sk, pb[p]);
        }
        __syncthreads();
        const uint32_t c1 = c0k + FK;
        if (c1 < cols) {
#pragma unroll
            for (uint32_t p = 0; p < kBP; ++p) {
                const uint32_t r = sn + p * kRPP, g = rs + n0 + r;
                const bool live = c1 + sk + kSeg <= cols && g < re;
                const int32_t f = g < re ? sorted[g] : 0;
                const uint32_t tok = a_sorted ? g : (uint32_t)(f < 0 ? 0 : (int32_t)f / (int32_t)top_k);
                const uint8_t* bsrc = (const uint8_t*)X + (size_t)tok * a_ld + sk;
                pb[p] = stage8_load(bsrc + c1, live);
            }
#pragma unroll
            for (uint32_t p = 0; p < kU; ++p)
                pfn[p] = D::fetch(c, arow, c1 / kBlock + p, lane, alive && c1 / kBlock + p < c.nkb);
        }

#pragma unroll
        for (uint32_t p = 0; p < kU; ++p) {
            const uint32_t kbu = c0k / kBlock + p;
            uint32_t w[16];
            D::rows(pf[p], c, alive ? rb_ratio<F>(c, wrow0 + (lane & 15u), kbu) : 0.0f, w);
            D::unzero(c, arow, kbu, w);
            Stage8 a8[2];
            frag(w, tr, lane, a8);
#pragma unroll
            for (uint32_t kk = 0; kk < kBlock / kK; ++kk) {
                const uint32_t ks = p * kBlock + kk * kK;
                const i2v af = frag_of(a8[kk / 4u], kk % 4u);
                i2v bf[NT];
#pragma unroll
                for (uint32_t j = 0; j < NT; ++j) bf[j] = frag<FK>(s_b, j * 16, ks, lane);
#pragma unroll
                for (uint32_t j = 0; j < NT; ++j) raw[j] = mma8<true>(af, bf[j], raw[j]);
            }
            {
                const uint32_t kb = kbu;
                float wsr[8];
                if constexpr (F::SC == kRowF32) {
                    const float f = alive ? fold_f32<F>(c, wrow0, kb) : 0.0f;
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t) wsr[t] = f;
                } else {
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t)
                        wsr[t] = alive ? fold_f32<F>(c, wrow0 + acc_row(lane, t), kb) : 0.0f;
                }
#pragma unroll
                for (uint32_t j = 0; j < NT; ++j) {
                    const uint32_t gcol = rs + brow0 + j * 16;
                    const int32_t f = gcol < re ? sorted[gcol] : 0;
                    const uint32_t srow = a_sorted ? gcol : (uint32_t)(f < 0 ? 0 : (int32_t)f / (int32_t)top_k);
                    const float s = gcol < re ? XS[(size_t)srow * xs_ld + kb] : 0.0f;
                    if constexpr (F::SC == kRowF32) {
                        const float ws = wsr[0] * s;
#pragma unroll
                        for (uint32_t t = 0; t < 8; ++t) {
                            out[j][t] += acc_val<true>(raw[j][t]) * ws;
                            raw[j][t] = 0.0f;
                        }
                    } else {
#pragma unroll
                        for (uint32_t t = 0; t < 8; ++t) {
                            out[j][t] += acc_val<true>(raw[j][t]) * (wsr[t] * s);
                            raw[j][t] = 0.0f;
                        }
                    }
                }
            }
        }
#pragma unroll
        for (uint32_t p = 0; p < kU; ++p) pf[p] = pfn[p];
        __syncthreads();
    }

    /* The int8 narrow tile's store, row-major in sorted order: eight weight rows of one sorted
     * row a lane a tile, one 16-byte write. */
    const uint32_t cl = frag_c(lane);
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) {
        const uint32_t r0 = wrow0 + acc_row(lane, 0);
        const uint32_t gcol = rs + n0 + j * 16u + cl;
        if (r0 >= N || gcol >= re) continue;
        f8v ov = out[j];
        if constexpr (F::SC == kRowF32)
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) ov[t] *= row_scale<F>(c, r0 + t);
        const size_t o = (size_t)gcol * y_ld + r0;
        *(uint4*)(Y + o) = make_uint4(bf16x2_rne(ov[0], ov[1]), bf16x2_rne(ov[2], ov[3]),
                                      bf16x2_rne(ov[4], ov[5]), bf16x2_rne(ov[6], ov[7]));
    }
}

/* ---- the op's operands and launch ---------------------------------------------------------- */
struct Operands {
    const RadTensor *a, *as, *w, *ws, *sorted, *off, *y;
    const void* const* wtab;
    const void* const* stab;
    long long M, N, K, nkb, T, a_ld, xs_ld, y_ld;
    size_t w_s0, s_s0;
    uint32_t word;
};

/* A weight/scale table: rank 3, [n_expert, rows, cols] declared; a zero leading stride marks a
 * pointer array, else the entry pitch in bytes (V1's convention for the private dtype). */
int wtab_of(const RadTensor* t, uint32_t base, uint32_t word, long long n_expert, long long rows,
            long long cols, const void* const** tab, size_t* s0, long long* ld) {
    if (!t || !t->data || t->rank != 3) return R4D_NO(RAD_E_SHAPE);
    if ((t->dtype & ~kWordMask) != base) return R4D_NO(RAD_E_DTYPE);
    if ((t->dtype & kWordMask) != word) return R4D_NO(RAD_E_DTYPE);
    if (t->shape[0] < n_expert || t->shape[1] < rows || t->shape[2] < cols)
        return R4D_NO(RAD_E_SHAPE);
    if (t->stride[2] != 1) return R4D_NO(RAD_E_STRIDE);
    *tab = (const void* const*)t->data;
    *s0  = (size_t)t->stride[0];           /* bytes an entry of a stacked plane; 0 pointer table */
    *ld  = t->stride[1];
    return RAD_OK;
}

int operands(const RadArgs* a, Operands* g) {
    g->a = r4d_opt(a, 0);
    g->as = r4d_opt(a, 1);
    g->w = r4d_opt(a, 2);
    g->ws = r4d_opt(a, 3);
    g->sorted = r4d_opt(a, 4);
    g->off = r4d_opt(a, 5);
    g->y = r4d_opt(a, 6);
    if (!g->a || !g->as || !g->w || !g->ws || !g->sorted || !g->off || !g->y)
        return R4D_NO(RAD_E_INVAL);
    if (!is_weight_dtype(g->w->dtype) || !is_scale_dtype(g->ws->dtype))
        return R4D_NO(RAD_E_DTYPE);
    g->word = g->w->dtype & kWordMask;
    if ((g->ws->dtype & kWordMask) != g->word) return R4D_NO(RAD_E_DTYPE);
    if (g->a->dtype != RAD_I8 || g->as->dtype != RAD_F32 || g->y->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (g->sorted->dtype != RAD_I32 || g->off->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    if (g->w->rank != 3 || g->ws->rank != 3) return R4D_NO(RAD_E_SHAPE);
    if (g->y->rank < 2 || g->a->rank < 2 || g->as->rank != 2) return R4D_NO(RAD_E_SHAPE);
    g->M = r4d_geti_or(a, "M", 0);
    g->N = r4d_geti_or(a, "N", 0);
    g->K = r4d_geti_or(a, "K", 0);
    const long long ne = r4d_geti_or(a, "n_expert", 0);
    const long long tk = r4d_geti_or(a, "top_k", 0);
    if (g->M < 1 || g->N <= 0 || g->K <= 0 || ne <= 0 || tk <= 0) return R4D_NO(RAD_E_SHAPE);
    const long long parts = r4d_geti_or(a, "parts", 1);
    if (parts < 1 || parts > 2 || g->N % parts) return R4D_NO(RAD_E_UNSUPPORTED);
    if (r4d_geti_or(a, "group", kBlock) != (long long)kBlock) return R4D_NO(RAD_E_SHAPE);
    if (g->N % 16 || g->K % kBlock) return R4D_NO(RAD_E_SHAPE);
    g->nkb = g->K / kBlock;
    if (r4d_numel(g->y) % g->N) return R4D_NO(RAD_E_SHAPE);
    /* Row count from the sliced output, as moe_gemm_q does. `M` is the band cap at issue, not the
     * live token count, so demanding T == M * top_k refuses every real step. */
    g->T = r4d_numel(g->y) / g->N;
    if (g->T < 1 || g->T > g->M * tk) return R4D_NO(RAD_E_SHAPE);
    const char* ord = r4d_gets(a, "a_order");
    const bool a_sorted = ord && !std::strcmp(ord, "sorted");
    if (ord && !a_sorted && std::strcmp(ord, "token")) return R4D_NO(RAD_E_UNSUPPORTED);
    const long long AR = a_sorted ? g->T : (g->T + tk - 1) / tk;
    long long a_rows = 0, as_rows = 0;
    g->a_ld  = r4d_rows_pitch(g->a, g->K, &a_rows);
    g->xs_ld = r4d_rows_pitch(g->as, g->nkb, &as_rows);
    g->y_ld  = r4d_row_pitch(g->y);
    if (g->a_ld < 0 || g->xs_ld < 0 || g->y_ld != g->N) return R4D_NO(RAD_E_STRIDE);
    /* Same rule as moe_gemm_q: the live row count is the sliced operand, not the band cap `M`. */
    if (a_rows < AR || as_rows < a_rows) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(g->sorted) < g->T || r4d_numel(g->off) < ne + 1) return R4D_NO(RAD_E_SHAPE);
    const Form f = form(g->word);
    if (is_trellis(f.kind) && g->N % ((long long)1 << f.lrun)) return R4D_NO(RAD_E_SHAPE);
    const long long wt = weight_bytes(f, g->N, g->K);
    const long long st = scale_bytes(f, g->N, g->K);
    long long lw = 0, ls = 0;
    int rc = wtab_of(g->w, R4D_DT_IQR_W, g->word, ne, g->N, g->K, &g->wtab, &g->w_s0, &lw);
    if (rc != RAD_OK) return rc;
    rc = wtab_of(g->ws, R4D_DT_IQR_S, g->word, ne, g->N, g->nkb, &g->stab, &g->s_s0, &ls);
    if (rc != RAD_OK) return rc;
    if (g->w_s0 > 0xFFFFFFFFull || g->s_s0 > 0xFFFFFFFFull) return R4D_NO(RAD_E_STRIDE);
    if (g->w_s0 != 0 && g->w_s0 < (size_t)wt) return R4D_NO(RAD_E_STRIDE);
    if (g->s_s0 != 0 && g->s_s0 < (size_t)st) return R4D_NO(RAD_E_STRIDE);
    (void)a_sorted;
    return RAD_OK;
}

/* The forms built (r4d_iqr.h's list), one instantiation each. */
#define R4D_IQR_T(k, b, lv, l, lrb, sc) FormT<k, b, lv, l, lrb, sc>

int moe_iqr_launch(const RadArgs* a, const Operands& g, RadStream stream) {
    const uint32_t N = (uint32_t)g.N, ne = (uint32_t)r4d_geti_or(a, "n_expert", 0);
    const uint32_t tk = (uint32_t)r4d_geti_or(a, "top_k", 0);
    const char* ord = r4d_gets(a, "a_order");
    const uint32_t a_sorted = ord && !std::strcmp(ord, "sorted") ? 1u : 0u;
    const uint32_t tiles = (N + 16u * kGW - 1u) / (16u * kGW);
    const dim3 grid(tiles, ne, 1);
    Ctx c{};
    c.W = nullptr;
    c.S = nullptr;
    c.N = N;
    c.nkb = (uint32_t)g.nkb;
    const Form f = form(g.word);
    c.run_tiles = is_trellis(f.kind) ? (1u << f.lrun) / 16u : 1u;
    /* The prefill tile (phase 2) from 64 sorted rows: the weight units decoded once per tile,
     * its output row-major in sorted order; the decode GEMV below it. */
    if (g.T >= 64 && g.K % kFK == 0 && N % 16u == 0 && (N + 127u) / 128u >= 1u) {
        const dim3 grid((N + 127u) / 128u, ((uint32_t)g.T + 63u) / 64u, ne);
        int rt = RAD_E_SHAPE;
#define R4D_IQR_TILE(k, b, lv, l, lrb, sc)                                                         \
        if (kernel_word(g.word) == R4D_IQR_T(k, b, lv, l, lrb, sc)::word())                       \
            rt = R4D_EMIT(stream, (moe_iqr_tile_kernel<R4D_IQR_T(k, b, lv, l, lrb, sc), 4u>),     \
                          grid, dim3(8u * kWave), 0, c, g.wtab, g.stab, g.w_s0, g.s_s0,           \
                          (const int8_t*)g.a->data, (const float*)g.as->data,                     \
                          (const int32_t*)g.sorted->data, (const int32_t*)g.off->data,            \
                          (uint16_t*)g.y->data, N, (uint32_t)g.T, ne, tk, (uint32_t)g.a_ld,       \
                          (uint32_t)g.xs_ld, (uint32_t)g.y_ld, a_sorted);
        R4D_IQR_FORMS(R4D_IQR_TILE)
#undef R4D_IQR_TILE
        if (rt == RAD_OK && hipGetLastError() == hipSuccess) return RAD_OK;
    }
    int rc = RAD_E_SHAPE;
#define R4D_IQR_GO(...)                                                                          \
    if (kernel_word(g.word) == R4D_IQR_T(__VA_ARGS__)::word())                                  \
        rc = R4D_EMIT(stream, (moe_iqr_gemv_kernel<R4D_IQR_T(__VA_ARGS__), 8u>), grid,           \
                      dim3(kGW * kWave), 0, c, (const int8_t*)g.a->data, (const float*)g.as->data, \
                      g.wtab, g.stab, g.w_s0, g.s_s0,                                       \
                      (const int32_t*)g.sorted->data, (const int32_t*)g.off->data,            \
                      (uint16_t*)g.y->data, (uint32_t)g.T, N, ne, tk, (uint32_t)g.a_ld,      \
                      (uint32_t)g.xs_ld, (uint32_t)g.y_ld, a_sorted);
    R4D_IQR_FORMS(R4D_IQR_GO)
#undef R4D_IQR_GO
    return rc == RAD_OK && hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}

}  // namespace

extern "C" int64_t r4d_rad_moe_gemm_iqr_scratch(const RadArgs*) { return 0; }

extern "C" int r4d_rad_moe_gemm_iqr(const RadArgs* a, RadStream s) {
    Operands g;
    const int rc = operands(a, &g);
    if (rc != RAD_OK) return rc;
    if (!built(g.word)) return R4D_NO(RAD_E_UNSUPPORTED);
    return moe_iqr_launch(a, g, s);
}
