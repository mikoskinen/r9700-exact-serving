/* r4d_ds41_mxfp4_mr.hip -- drafter_mx: moe_gemm_ggml type=mxfp4 (the DSpark drafter's routed experts) in
 * multi-row forms that read and dequantise each weight block ONCE for all of an expert's rows of the call, with
 * every output produced by the same floating-point operations as the stock kernel that call would run, so the
 * bytes are the same. Off unless switched on (r4d_ds41_mxfp4_mr.h); mx_moe calls it ahead of the stock launch.
 *
 * ---- why the stock grouped GEMM is slow on the drafter's calls (r4d_ds41_mxfp4.hip, mx_moe)
 *
 *  The drafter runs 5 rows a sequence, top-3 of 128 experts, 32 experts a rank: at 1 user 15 slots, a rank's
 *  live experts are a handful with 1..5 rows each; gate_up is 4608 x 5120, down 5120 x 2304 (MXFP4, 17 bytes a
 *  block of 32).
 *  (a) The gemv band (<= 16 tokens) sizes its work as if min(n_expert, T) experts were live: rgpi row groups an
 *      item so that L x row groups / rgpi <= 256 items, where L = min(32, 15) = 15. With 3-5 experts live on the
 *      rank only 3-5 x ipe of the 255 launched blocks find work, and each walks rgpi = 17 row groups in turn.
 *  (b) An expert with 2+ rows at K = 5120 does not fit its rows whole in the 36 KiB f32 staging buffer (one row
 *      of 5120 does), so the pass runs "chunked": K in chunks of 1024 values restaged a row group, two block
 *      barriers around ONE block a lane, the chunk's weight loads issued only after its barrier.
 *  (c) Above 16 tokens (2+ users) the tiled kernel runs: 128-row weight tiles dequantised into LDS a 32-value
 *      slab at a time for 32 activation rows, an expert's 1-4 live rows padding a 32-row tile.
 *
 * ---- the forms
 *
 *  Work items: (live expert, weight row group, segment of at most MT of the expert's rows). Wave 0 of every block
 *  counts the live experts (mx_range over the offsets, as the stock kernels) and their items into LDS with a
 *  wave prefix sum; a block walks items blockIdx.x, + gridDim.x, ... so only live work is launched work. A
 *  segment whose rows are all dropped slots only writes its zeros.
 *  - xm_gemv_kernel (the stock path is the gemv band): lever D2's form (r4d_ds41_ggml_mr.hip) on MXFP4 blocks:
 *    H weight rows a wave side by side (32 / H lanes a row), R rows a lane, the segment's rows staged as f32 in
 *    LDS in slabs of whole stages, each block dequantised once and applied to all MT rows, PD stages of weight
 *    words in flight; H partials a lane, rotated a stage, closed by the stock butterfly's levels.
 *  - xm_seq_kernel (the stock path is the tiled kernel): a lane owns a weight row (R of them, 32 apart) and the
 *    segment's MT rows; it walks the row's blocks in order, dequantises each once and adds its 32 products to
 *    each row's one chain; the activations are LDS broadcasts (every lane reads the same values).
 *
 * ---- exactness
 *
 *  gemv band (mx_gemv_kernel, MOE): output (sorted row i, weight row n) is lane p's sum from 0.0f of blocks
 *  U = p, p + 32, ... in order (mx_row: lane = block mod 32, four in flight, accumulated in ascending V), each
 *  block's groups G = 0..3 and values t = 0..7 as acc += x * w (w = Mx::dec<G>, x the bf16 widened; separate
 *  rounded products, -ffp-contract=off), then r4d_wave_sum and one rounding (mx_put), 0.0f where the slot is
 *  dead. xm_gemv_kernel keeps partial p of each output in one lane slot as D2 does (the argument of
 *  r4d_ds41_ggml_mr.hip, word for word: a block is a unit, UPB = 1): the same blocks in the same order into the
 *  same partial, the same butterfly levels on the same operands, the same rounding and the same zeros.
 *  tiled (mx_tiled_kernel, MOE): output (row, n) is ONE chain from 0.0f over the slabs (blocks) in order and the
 *  slab's 32 values kk = 8 G + t in order: acc += x * w, then mx_put, 0.0f for a dead row. xm_seq_kernel's lane
 *  runs exactly that chain for each of its outputs. Neither form's bits depend on H, R, MT, the grid or the slab
 *  sizes; r4d_selftest --mxcheck byte-compares both against the stock kernels, tests/r4d_emu ds41mx on the host.
 */
#define R4D_MX_BLOCKS_ONLY 1
#include "r4d_ds41_mxfp4.hip"

#include <cstdio>

namespace {

constexpr int kXmThreads = 256;          /* WN <= 8 waves a block */
/* The forms serve the drafter's query passes: 5 rows a drafting sequence (DSpark's whole block), so up to ~100 sequences
 * (DRAFTER track 18.11.2026: was 48 tokens, which sent a 16-sequence lane's 80-token calls to the stock tiled kernel). */
constexpr int kXmMaxTokens = 512;

struct MxMrArgs {
    const uint16_t* x;
    long long x_ld;
    const uint8_t* w;
    const uint8_t* const* wtab;
    long long w_ld, w_es;
    uint16_t* y;
    long long y_ld;
    const int32_t* sorted;
    const int32_t* off;
    int T, N, K;
    int n_expert, top_k, a_sorted, AR, x_al;
    int UK;                              /* blocks a row */
    int NS;                              /* stages a row (gemv form; seq: blocks) */
    int SS;                              /* stages (blocks) a slab */
    int RB;                              /* weight rows a block */
    int nrg;                             /* row groups an expert */
    int live_cap;                        /* live-expert entries in LDS */
    int live_off, x_off;                 /* LDS byte offsets (the value table at 0) */
};

/* ------------------------------------------------------------------ work items */
struct XmLive {
    int* e;                              /* the live experts, ascending */
    int* s;                              /* their first sorted row */
    int* r;                              /* their rows */
    int* c;                              /* items before them */
};
__device__ __forceinline__ XmLive xm_live(uint8_t* smem, const MxMrArgs& A) {
    int* b = (int*)(smem + A.live_off);
    return { b, b + A.live_cap, b + 2 * A.live_cap, b + 3 * A.live_cap };
}

/* Wave 0: the live experts and their items (segments x row groups), counted in expert order. */
template <int MT>
__device__ __forceinline__ void xm_count(const MxMrArgs& A, XmLive L, int* s_n, int* s_items) {
    const int lane = (int)threadIdx.x & 31;
    if (((int)threadIdx.x >> 5) != 0) return;
    int cnt = 0;
    uint32_t base = 0;
    for (int e0 = 0; e0 < A.n_expert; e0 += 32) {
        const int e = e0 + lane;
        int s = 0, t = 0;
        if (e < A.n_expert) mx_range(A.off, e, A.n_expert, A.T, &s, &t);
        const int rows = e < A.n_expert ? t - s : 0;
        const bool live = rows > 0;
        const uint32_t items = live ? (uint32_t)(((rows + MT - 1) / MT) * A.nrg) : 0u;
        const uint32_t inc = r4d_wave_prefix_u32(items, lane);
        const uint32_t tot = (uint32_t)__builtin_amdgcn_readlane((int)inc, 31);
        const uint32_t bal = __builtin_amdgcn_ballot_w32(live);
        const int pos = cnt + __builtin_popcount(bal & ((1u << lane) - 1u));
        if (live && pos < A.live_cap) {
            L.e[pos] = e;
            L.s[pos] = s;
            L.r[pos] = rows;
            L.c[pos] = (int)(base + inc - items);
        }
        cnt += __builtin_popcount(bal);
        base += tot;
    }
    if (lane == 0) {
        *s_n = cnt < A.live_cap ? cnt : A.live_cap;
        *s_items = (int)base;
    }
}

/* Item `it` -> its expert, sorted rows [r0, r1) and row group: the last live expert whose items start at or
 * before `it`, then row group-major, segment-minor (an expert's segments of one row group run side by side). */
template <int MT>
__device__ __forceinline__ void xm_item(XmLive L, int n, int it, int* e, int* r0, int* r1, int* rg) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (L.c[mid] <= it) lo = mid;
        else hi = mid - 1;
    }
    const int local = it - L.c[lo], segs = (L.r[lo] + MT - 1) / MT;
    *rg = local / segs;
    const int seg = local - *rg * segs;
    *e = L.e[lo];
    *r0 = L.s[lo] + seg * MT;
    *r1 = min(*r0 + MT, L.s[lo] + L.r[lo]);
}

/* The segment's live rows (a dropped slot or a token past AR is dead: a row of zeros). */
template <int MT>
__device__ __forceinline__ uint32_t xm_livem(const MxMrArgs& A, int r0, int mc) {
    uint32_t m = 0;
#pragma unroll
    for (int i = 0; i < MT; ++i)
        if (i < mc && mx_moe_tok(A.sorted, r0 + i, A.top_k, A.a_sorted, A.AR) >= 0) m |= 1u << i;
    return m;
}

/* Rows [r0, r0 + mc) x the item's weight rows [n0, n0 + RB) written as zeros (a segment of dead slots only). */
__device__ __forceinline__ void xm_zero(const MxMrArgs& A, int r0, int mc, int n0) {
    for (int idx = (int)threadIdx.x; idx < mc * A.RB; idx += (int)blockDim.x) {
        const int m = idx / A.RB, n = n0 + (idx - m * A.RB);
        if (n < A.N) mx_put(A.y + (long long)(r0 + m) * A.y_ld + n, 0.0f);
    }
}

/* Stages of weight words in flight ahead of the one computed (the stage loop unrolled that many times, a buffer
 * a stage): enough to cover a DRAM round trip when MT x R makes a stage's arithmetic short. */
template <int MT, int R>
constexpr int xm_pd() { return MT * R >= 24 ? 1 : (MT * R >= 8 ? 2 : 4); }

/* ------------------------------------------------------------------ the gemv band's form (D2 on MXFP4) */
/* r4d_wave_sum's levels over the H partials of one output in this lane and the slot's 32 / H lanes. */
template <int H>
__device__ __forceinline__ float xm_reduce(const float (&a)[H]) {
    if constexpr (H == 1) {
        return r4d_wave_sum(a[0]);
    } else {
        constexpr int LPR = 32 / H;
        float v[H];
#pragma unroll
        for (int j = 0; j < H; ++j) v[j] = a[j];
#pragma unroll
        for (int X = 16; X >= LPR; X >>= 1) {
            const int D = X / LPR;
#pragma unroll
            for (int j = 0; j < D; ++j) v[j] = v[j] + v[j + D];
        }
        float r = v[0];
        if constexpr (LPR >= 16) r = r4d_dpp_xor_add<0x168>(r);
        if constexpr (LPR >= 8) r = r4d_dpp_xor_add<0x164>(r);
        if constexpr (LPR >= 4) r = r4d_dpp_xor_add<0x162>(r);
        if constexpr (LPR >= 2) r = r4d_dpp_xor_add<0x161>(r);
        return r;
    }
}

/* Stages [s0, s1) of the segment's MT rows into the slab, [stage][row][group][half][lane][4] f32 (zeros past mc,
 * for a dead slot and past K). A unit is a block of 32: group G of block U starts at 32 U + 8 G. */
template <int H, int MT>
__device__ __forceinline__ void xm_stage_g(const MxMrArgs& A, float* xs, int r0, int mc, int s0, int s1) {
    constexpr int LPR = 32 / H, VPS = 32 * LPR, PPS = VPS / 8;
    const int np = (s1 - s0) * MT * PPS;
    for (int idx = (int)threadIdx.x; idx < np; idx += (int)blockDim.x) {
        const int q = idx % PPS, rest = idx / PPS;
        const int m = rest % MT, ss = rest / MT;
        const int s = s0 + ss, k = s * VPS + 8 * q;
        const int tok = (m < mc && k < A.K) ? mx_moe_tok(A.sorted, r0 + m, A.top_k, A.a_sorted, A.AR) : -1;
        float v[8];
        if (tok >= 0) {
            mx_x8(A.x + (long long)tok * A.x_ld + k, A.x_al != 0, v);
        } else {
#pragma unroll
            for (int t = 0; t < 8; ++t) v[t] = 0.0f;
        }
        const int U = k >> 5, G = (k >> 3) & 3, i = U - s * LPR;
        float* d = xs + (ss * MT + m) * VPS;
        *(float4*)(d + ((2 * G) * LPR + i) * 4) = make_float4(v[0], v[1], v[2], v[3]);
        *(float4*)(d + ((2 * G + 1) * LPR + i) * 4) = make_float4(v[4], v[5], v[6], v[7]);
    }
}

template <int T, int H, int R, int MT>
__global__ __launch_bounds__(kXmThreads) void xm_gemv_kernel(MxMrArgs A) {
    using Q = Mx<T>;
    constexpr int LPR = 32 / H, VPS = 32 * LPR, PD = xm_pd<MT, R>();
    R4D_DYN_LDS(r4d_u32x4, xm_lds);
    uint8_t* const smem = (uint8_t*)xm_lds;
    __shared__ int s_n, s_items;
    const int lane = (int)threadIdx.x & 31, wave = (int)threadIdx.x >> 5;
    const int g = lane / LPR, i = lane - g * LPR;
    mx_load_table<T>(smem);
    const void* tab = smem;
    const XmLive L = xm_live(smem, A);
    xm_count<MT>(A, L, &s_n, &s_items);
    __syncthreads();
    const int nlive = s_n, nitems = s_items;
    float* const xs = (float*)(smem + A.x_off);
    for (int it = (int)blockIdx.x; it < nitems; it += (int)gridDim.x) {
        int e = 0, r0 = 0, r1 = 0, rg = 0;
        xm_item<MT>(L, nlive, it, &e, &r0, &r1, &rg);
        const int mc = r1 - r0, n0 = rg * A.RB;
        const uint32_t livem = xm_livem<MT>(A, r0, mc);
        if (!livem) {
            xm_zero(A, r0, mc, n0);
            continue;
        }
        const uint8_t* wb = A.wtab ? A.wtab[e] : A.w + (long long)e * A.w_es;
        const int nl = n0 + wave * (H * R) + g * R;
        const uint8_t* wr[R];
#pragma unroll
        for (int r = 0; r < R; ++r) wr[r] = wb + (long long)min(nl + r, A.N - 1) * A.w_ld;
        float acc[R][MT][H];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int m = 0; m < MT; ++m)
#pragma unroll
                for (int j = 0; j < H; ++j) acc[r][m][j] = 0.0f;
        uint32_t raw[PD][R][Q::NR];
        auto load = [&](int s, uint32_t (&rw)[R][Q::NR]) {
            const int U = s * LPR + i;
            if (s < A.NS && U < A.UK) {
#pragma unroll
                for (int r = 0; r < R; ++r) Q::load(wr[r] + (long long)U * Q::BB, rw[r]);
            }
        };
#pragma unroll
        for (int k = 0; k < PD; ++k) load(k, raw[k]);
        auto rotate = [&]() {
            if constexpr (H > 1) {
#pragma unroll
                for (int r = 0; r < R; ++r)
#pragma unroll
                    for (int m = 0; m < MT; ++m) {
                        const float t0 = acc[r][m][0];
#pragma unroll
                        for (int j = 0; j + 1 < H; ++j) acc[r][m][j] = acc[r][m][j + 1];
                        acc[r][m][H - 1] = t0;
                    }
            }
        };
        auto stage = [&](int s, const float* xb, uint32_t (&rw)[R][Q::NR]) {
            const int U = s * LPR + i;
            const bool act = U < A.UK;
            float w[R][32];
            if (act) {
#pragma unroll
                for (int r = 0; r < R; ++r)
                    mx_for<4>([&](auto gi) {
                        constexpr int G = decltype(gi)::value;
                        Q::template dec<G>(rw[r], tab, &w[r][8 * G]);
                    });
            }
            load(s + PD, rw);
            if (act) {
#pragma unroll
                for (int G = 0; G < 4; ++G) {
#pragma unroll
                    for (int m = 0; m < MT; ++m) {
                        const float* xp = xb + m * VPS;
                        const float4 a = *(const float4*)(xp + ((2 * G) * LPR + i) * 4);
                        const float4 c = *(const float4*)(xp + ((2 * G + 1) * LPR + i) * 4);
                        const float xv[8] = { a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w };
#pragma unroll
                        for (int r = 0; r < R; ++r)
#pragma unroll
                            for (int t = 0; t < 8; ++t) acc[r][m][0] += xv[t] * w[r][8 * G + t];
                    }
                }
            }
            rotate();
        };
        for (int s0 = 0; s0 < A.NS; s0 += A.SS) {
            const int s1 = min(s0 + A.SS, A.NS);
            lds_barrier();
            xm_stage_g<H, MT>(A, xs, r0, mc, s0, s1);
            lds_barrier();
#pragma unroll 1
            for (int sb = s0 - s0 % PD; sb < s1; sb += PD) {
#pragma unroll
                for (int k = 0; k < PD; ++k)
                    if (sb + k >= s0 && sb + k < s1) stage(sb + k, xs + (sb + k - s0) * MT * VPS, raw[k]);
            }
        }
        if constexpr (H > 1) {
            const int pad = (H - A.NS % H) % H;
#pragma unroll 1
            for (int e2 = 0; e2 < pad; ++e2) rotate();
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int n = nl + r;
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                const float v = xm_reduce<H>(acc[r][m]);
                if (i == 0 && m < mc && n < A.N)
                    mx_put(A.y + (long long)(r0 + m) * A.y_ld + n, ((livem >> m) & 1u) ? v : 0.0f);
            }
        }
    }
}

/* ------------------------------------------------------------------ the tiled kernel's order */
/* Blocks [b0, b1) of the segment's MT rows into the slab, [block][row][32] f32 in K order (zeros past mc and for a
 * dead slot): every lane reads the same values, LDS broadcasts. */
template <int MT, int EX = 0>
__device__ __forceinline__ void xm_stage_s(const MxMrArgs& A, float* xs, int r0, int mc, int b0, int b1) {
    const int np = (b1 - b0) * MT * 4;
    for (int idx = (int)threadIdx.x; idx < np; idx += (int)blockDim.x) {
        const int q = idx & 3, rest = idx >> 2;
        const int m = rest % MT, bb = rest / MT;
        if (EX && m >= mc) continue;   /* EX reads rows < mc only */
        const int k = (b0 + bb) * 32 + 8 * q;
        const int tok = m < mc ? mx_moe_tok(A.sorted, r0 + m, A.top_k, A.a_sorted, A.AR) : -1;
        float v[8];
        if (tok >= 0) {
            mx_x8(A.x + (long long)tok * A.x_ld + k, A.x_al != 0, v);
        } else {
#pragma unroll
            for (int t = 0; t < 8; ++t) v[t] = 0.0f;
        }
        float* d = xs + (bb * MT + m) * 32 + 8 * q;
        *(float4*)d = make_float4(v[0], v[1], v[2], v[3]);
        *(float4*)(d + 4) = make_float4(v[4], v[5], v[6], v[7]);
    }
}

template <int T, int R, int MT, int EX>
__global__ __launch_bounds__(kXmThreads) void xm_seq_kernel(MxMrArgs A) {
    using Q = Mx<T>;
    constexpr int PD = MT * R >= 16 ? 2 : 4;
    R4D_DYN_LDS(r4d_u32x4, xm_lds);
    uint8_t* const smem = (uint8_t*)xm_lds;
    __shared__ int s_n, s_items;
    const int lane = (int)threadIdx.x & 31, wave = (int)threadIdx.x >> 5;
    mx_load_table<T>(smem);
    const void* tab = smem;
    const XmLive L = xm_live(smem, A);
    xm_count<MT>(A, L, &s_n, &s_items);
    __syncthreads();
    const int nlive = s_n, nitems = s_items;
    float* const xs = (float*)(smem + A.x_off);
    for (int it = (int)blockIdx.x; it < nitems; it += (int)gridDim.x) {
        int e = 0, r0 = 0, r1 = 0, rg = 0;
        xm_item<MT>(L, nlive, it, &e, &r0, &r1, &rg);
        const int mc = r1 - r0, n0 = rg * A.RB;
        const uint32_t livem = xm_livem<MT>(A, r0, mc);
        if (!livem) {
            xm_zero(A, r0, mc, n0);
            continue;
        }
        /* EX (DRAFTER track 18.11.2026): the chains run over the segment's mc rows only (block-uniform, read into an
         * SGPR), not MT: rows past mc are never written, so the outputs are the same operations as before. */
        const int mcu = EX ? __builtin_amdgcn_readfirstlane(mc) : MT;
        const uint8_t* wb = A.wtab ? A.wtab[e] : A.w + (long long)e * A.w_es;
        const int nb = n0 + wave * 32 * R + lane;          /* rows nb + 32 r */
        const uint8_t* wr[R];
#pragma unroll
        for (int r = 0; r < R; ++r) wr[r] = wb + (long long)min(nb + 32 * r, A.N - 1) * A.w_ld;
        float acc[R][MT];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int m = 0; m < MT; ++m) acc[r][m] = 0.0f;
        uint32_t raw[PD][R][Q::NR];
        auto load = [&](int b, uint32_t (&rw)[R][Q::NR]) {
            if (b < A.UK) {
#pragma unroll
                for (int r = 0; r < R; ++r) Q::load(wr[r] + (long long)b * Q::BB, rw[r]);
            }
        };
#pragma unroll
        for (int k = 0; k < PD; ++k) load(k, raw[k]);
        /* one block: its 32 values once, then each row's chain over them in K order */
        auto block = [&](int b, const float* xb, uint32_t (&rw)[R][Q::NR]) {
            if constexpr ((EX & 8) != 0) {
                /* EX 9 (DRAFTER track): the block's raw words copied out and the next load issued first, then group by
                 * group (8 values decoded at a time) each row's chain over the group's values: a row's chain still runs
                 * the block's 32 values in K order, with 8 decoded weights a row live instead of 32.
                 * EX bit 16 (EX 25): each step as one fma. The product of a bf16 activation (8 significant bits) and an MXFP4
                 * value (kv * 2^(e - 128), kv of at most 2 significant bits) has at most 10 significant bits, so the stock
                 * x * w is exact and fma(x, w, acc) rounds the same sum once: the same bits, one VALU op a MAC instead of 2
                 * (unless the product falls below f32's normal range, ~2^-126, where the separate product could round). */
                uint32_t cur[R][Q::NR];
#pragma unroll
                for (int r = 0; r < R; ++r)
#pragma unroll
                    for (int i = 0; i < Q::NR; ++i) cur[r][i] = rw[r][i];
                load(b + PD, rw);
                mx_for<4>([&](auto gi) {
                    constexpr int G = decltype(gi)::value;
                    float w8[R][8];
#pragma unroll
                    for (int r = 0; r < R; ++r) Q::template dec<G>(cur[r], tab, w8[r]);
#pragma unroll
                    for (int m = 0; m < MT; ++m) {
                        if (m >= mcu) break;
                        const float* xp = xb + m * 32 + 8 * G;
                        const float4 a = *(const float4*)xp;
                        const float4 c = *(const float4*)(xp + 4);
                        const float xv[8] = { a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w };
#pragma unroll
                        for (int r = 0; r < R; ++r)
#pragma unroll
                            for (int t = 0; t < 8; ++t) {
                                if constexpr ((EX & 16) != 0)
                                    acc[r][m] = __builtin_fmaf(xv[t], w8[r][t], acc[r][m]);
                                else
                                    acc[r][m] += xv[t] * w8[r][t];
                            }
                    }
                });
                return;
            }
            float w[R][32];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if constexpr ((EX & 4) != 0) {   /* bench ablation: no decode (the raw words as values; output differs) */
#pragma unroll
                    for (int t = 0; t < 32; ++t) w[r][t] = __uint_as_float(rw[r][t % 5] ^ (uint32_t)t);
                } else {
                    mx_for<4>([&](auto gi) {
                        constexpr int G = decltype(gi)::value;
                        Q::template dec<G>(rw[r], tab, &w[r][8 * G]);
                    });
                }
            }
            load(b + PD, rw);
            if constexpr ((EX & 2) != 0) {   /* bench ablation: no products (one add a value; output differs) */
#pragma unroll
                for (int r = 0; r < R; ++r)
#pragma unroll
                    for (int t = 0; t < 32; ++t) acc[r][0] += w[r][t];
                return;
            }
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                if (EX && m >= mcu) break;
                const float* xp = xb + m * 32;
#pragma unroll
                for (int h = 0; h < 8; ++h) {
                    const float4 a = *(const float4*)(xp + 4 * h);
                    const float xv[4] = { a.x, a.y, a.z, a.w };
#pragma unroll
                    for (int r = 0; r < R; ++r)
#pragma unroll
                        for (int t = 0; t < 4; ++t) acc[r][m] += xv[t] * w[r][4 * h + t];
                }
            }
        };
        for (int b0 = 0; b0 < A.UK; b0 += A.SS) {
            const int b1 = min(b0 + A.SS, A.UK);
            lds_barrier();
            xm_stage_s<MT, EX>(A, xs, r0, mc, b0, b1);
            lds_barrier();
#pragma unroll 1
            for (int bb = b0 - b0 % PD; bb < b1; bb += PD) {
#pragma unroll
                for (int k = 0; k < PD; ++k)
                    if (bb + k >= b0 && bb + k < b1) block(bb + k, xs + (bb + k - b0) * MT * 32, raw[k]);
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int n = nb + 32 * r;
            if (n >= A.N) continue;
#pragma unroll
            for (int m = 0; m < MT; ++m)
                if (m < mc) mx_put(A.y + (long long)(r0 + m) * A.y_ld + n, ((livem >> m) & 1u) ? acc[r][m] : 0.0f);
        }
    }
}

/* ------------------------------------------------------------------ the WMMA form (rule B; DRAFTER track 18.11.2026)
 * xm_wm_kernel: the tiled band's calls on v_wmma_f32_16x16x16_bf16. Not the stock order: rule B (deterministic and
 * batch-invariant against its own reference), r4d_selftest / dbench compare it with itself across row placements.
 *  - A item is (live expert, RB = WN x WR x 16 weight rows, a segment of up to 16 NT of the expert's sorted rows): the weight
 *    blocks are read and decoded once for all of the segment's rows (an expert's rows are one segment up to 16 NT).
 *  - Operands: A = 16 activation rows (a token tile), B = 16 weight rows; lane L is row / column L % 16 and K slots 8 (L / 16)
 *    .. + 7. A block's 32 values are two WMMAs; slot pair (2 p, 2 p + 1) of lane half h in WMMA j holds the two nibbles of
 *    qs[8 j + 4 h + p] (values q and q + 16), so one LDS read of a 256-entry table turns a byte into its two bf16 values
 *    (kv unscaled: integers <= 12, exact) and the staged activations carry the same interleave.
 *  - The scale is factored out: P = the block's 32 products summed by the two WMMAs from 0 (the products are exact: bf16 x
 *    an integer <= 12), then acc = fma(2^(e - 128), P, acc) in the lane that owns the weight row (D lane L is column L % 16,
 *    rows 8 (L / 16) + i). Each output: blocks in K order, each block's partial by the hardware's WMMA sum, one fma a block,
 *    one rounding to bf16 at the end -- the same operations whatever the other rows of the tile are (a dead or absent row is
 *    a zero column of A and never mixes into another row's sum).
 *  - The activations are staged in LDS a slab of SS blocks at a time as [block][tile][j][h][row][8 bf16] (a lane's fragment
 *    is 16 contiguous bytes; a half-wave reads 256 contiguous bytes). */
typedef __bf16 xw_v8bf __attribute__((ext_vector_type(8)));
typedef float xw_v8f __attribute__((ext_vector_type(8)));

__device__ __forceinline__ xw_v8f xw_wmma(r4d_u32x4 a, r4d_u32x4 b, xw_v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(xw_v8bf, a), __builtin_bit_cast(xw_v8bf, b), c);
}
__device__ __forceinline__ uint32_t xw_bf(float v) { return __float_as_uint(v) >> 16; }   /* exact for kv */

constexpr int kXwTab = 0;                /* the byte -> bf16 pair table is static LDS (xw_tab), none in the dynamic part */

/* Blocks [b0, b1) of the segment's rows (nt tiles) into the slab. */
template <int NT>
__device__ __forceinline__ void xw_stage(const MxMrArgs& A, uint8_t* xs, int r0, int mc, int nt, int b0, int b1) {
    const int np = (b1 - b0) * nt * 64;
    for (int idx = (int)threadIdx.x; idx < np; idx += (int)blockDim.x) {
        const int n = idx & 15, jh = (idx >> 4) & 3, rest = idx >> 6;
        const int t = rest % nt, bb = rest / nt;
        const int m = 16 * t + n;
        const int tok = m < mc ? mx_moe_tok(A.sorted, r0 + m, A.top_k, A.a_sorted, A.AR) : -1;
        r4d_u32x4 o = { 0u, 0u, 0u, 0u };
        if (tok >= 0) {
            const int q = 8 * (jh >> 1) + 4 * (jh & 1);
            const uint16_t* p = A.x + (long long)tok * A.x_ld + (long long)(b0 + bb) * 32 + q;
            uint32_t d0, d1, e0, e1;
            if (A.x_al) {
                const uint2 a = *(const uint2*)(const void*)p;
                const uint2 c = *(const uint2*)(const void*)(p + 16);
                d0 = a.x; d1 = a.y; e0 = c.x; e1 = c.y;
            } else {
                d0 = ldg_u16(p) | (ldg_u16(p + 1) << 16); d1 = ldg_u16(p + 2) | (ldg_u16(p + 3) << 16);
                e0 = ldg_u16(p + 16) | (ldg_u16(p + 17) << 16); e1 = ldg_u16(p + 18) | (ldg_u16(p + 19) << 16);
            }
            o[0] = (d0 & 0xffffu) | (e0 << 16);
            o[1] = (d0 >> 16) | (e0 & 0xffff0000u);
            o[2] = (d1 & 0xffffu) | (e1 << 16);
            o[3] = (d1 >> 16) | (e1 & 0xffff0000u);
        }
        *(r4d_u32x4*)(xs + ((((bb * NT + t) * 2 + (jh >> 1)) * 2 + (jh & 1)) * 16 + n) * 16) = o;
    }
}

template <int WR, int NT, int PD, int AB = 0>   /* AB: bench ablations (outputs differ): 1 no WMMA / fma, 2 no decode, 4 no staging */
__global__ __launch_bounds__(kXmThreads) void xm_wm_kernel(MxMrArgs A) {
    using Q = Mx<MX_FP4>;
    constexpr int MT = 16 * NT;
    R4D_DYN_LDS(r4d_u32x4, xm_lds);
    uint8_t* const smem = (uint8_t*)xm_lds;
    __shared__ int s_n, s_items;
    const int lane = (int)threadIdx.x & 31, wave = (int)threadIdx.x >> 5;
    const int nl = lane & 15, h = lane >> 4;
    __shared__ uint32_t xw_tab[256];            /* static: its address folds into the ds_load offset */
    uint32_t* const tab = xw_tab;
    for (int b = (int)threadIdx.x; b < 256; b += (int)blockDim.x)
        tab[b] = xw_bf(kMxValues[b & 15]) | (xw_bf(kMxValues[b >> 4]) << 16);
    const XmLive L = xm_live(smem, A);
    xm_count<MT>(A, L, &s_n, &s_items);
    __syncthreads();
    const int nlive = s_n, nitems = s_items;
    uint8_t* const xs = smem + A.x_off;
    for (int it = (int)blockIdx.x; it < nitems; it += (int)gridDim.x) {
        int e = 0, r0 = 0, r1 = 0, rg = 0;
        xm_item<MT>(L, nlive, it, &e, &r0, &r1, &rg);
        const int mc = r1 - r0, n0 = rg * A.RB;
        /* rows past the segment's last live row (the dropped tail opened to T) take no tile: zeros only */
        int ml = 0;
        for (int m0 = 0; m0 < mc; m0 += 32) {
            const int m = m0 + lane;
            const bool lv = m < mc && mx_moe_tok(A.sorted, r0 + m, A.top_k, A.a_sorted, A.AR) >= 0;
            const uint32_t bal = __builtin_amdgcn_ballot_w32(lv);
            if (bal) ml = m0 + 32 - __builtin_clz(bal);
        }
        ml = __builtin_amdgcn_readfirstlane(ml);
        if (ml == 0) {
            xm_zero(A, r0, mc, n0);
            continue;
        }
        const int nt = (ml + 15) >> 4;
        const uint8_t* wb = A.wtab ? A.wtab[e] : A.w + (long long)e * A.w_es;
        const int nw = n0 + wave * 16 * WR + nl;          /* this lane's weight rows nw + 16 r */
        const uint8_t* wr[WR];
#pragma unroll
        for (int r = 0; r < WR; ++r) wr[r] = wb + (long long)min(nw + 16 * r, A.N - 1) * A.w_ld;
        xw_v8f acc[WR][NT];
#pragma unroll
        for (int r = 0; r < WR; ++r)
#pragma unroll
            for (int t = 0; t < NT; ++t)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[r][t][i] = 0.0f;
        /* In flight: each row's block as the 16 raw bytes of the dword window at (block & ~3) + 4 h -- lane half 0 holds e
         * and qs[0..3], qs[8..11], half 1 qs[4..7], qs[12..15] at the same window offsets -- decoded only when used. */
        int sb[WR];
#pragma unroll
        for (int r = 0; r < WR; ++r) sb[r] = (int)((uintptr_t)wr[r] & 3u);
        r4d_u32x4 raw[PD][WR];
        auto load = [&](int b, r4d_u32x4 (&rw)[WR]) {
            if (b < A.UK) {
#pragma unroll
                for (int r = 0; r < WR; ++r) {
                    const uintptr_t pa = ((uintptr_t)(wr[r] + (long long)b * Q::BB) & ~(uintptr_t)3) + 4 * h;
                    const mx_gu32 g = (mx_gu32)(const void*)pa;
                    rw[r][0] = g[0]; rw[r][1] = g[1]; rw[r][2] = g[2]; rw[r][3] = g[3];
                }
            }
        };
#pragma unroll
        for (int k = 0; k < PD; ++k) load(k, raw[k]);
        auto block = [&](int b, const uint8_t* xb, r4d_u32x4 (&rw)[WR]) {
            uint32_t q0[WR], q1[WR];
            float d[WR];
#pragma unroll
            for (int r = 0; r < WR; ++r) {
                const r4d_u32x4 w = rw[r];
                const int sA = (sb[r] + b) & 3;           /* the block's byte offset in its window (17 b = b mod 4) */
                const int sh = (sA + 1) & 3;
                const bool s3 = sA == 3;
                q0[r] = __builtin_amdgcn_alignbyte(w[1], s3 ? w[1] : w[0], sh);
                q1[r] = __builtin_amdgcn_alignbyte(w[3], s3 ? w[3] : w[2], sh);
                const uint32_t e0 = (w[0] >> (8 * sA)) & 0xffu;
                const uint32_t e1 = (uint32_t)__builtin_amdgcn_permlanex16((int)e0, (int)e0, 0x76543210, 0xfedcba98, false, false);
                d[r] = mx_e8m0_half(h ? e1 : e0);
            }
            load(b + PD, rw);
            r4d_u32x4 bw[WR][2];
#pragma unroll
            for (int r = 0; r < WR; ++r) {
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    if constexpr ((AB & 2) != 0) {
                        bw[r][0][p] = q0[r] ^ (uint32_t)p;
                        bw[r][1][p] = q1[r] ^ (uint32_t)p;
                    } else {
                        bw[r][0][p] = tab[(q0[r] >> (8 * p)) & 0xffu];
                        bw[r][1][p] = tab[(q1[r] >> (8 * p)) & 0xffu];
                    }
                }
            }
            if constexpr ((AB & 1) != 0) {
#pragma unroll
                for (int r = 0; r < WR; ++r)
#pragma unroll
                    for (int p = 0; p < 4; ++p)
                        acc[r][0][p] += __uint_as_float((bw[r][0][p] ^ bw[r][1][p]) & 0x3fffffffu) * d[r];
                return;
            }
            const xw_v8f z = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                if (t >= nt) break;
                const r4d_u32x4 a0 = *(const r4d_u32x4*)(xb + (((t * 2 + 0) * 2 + h) * 16 + nl) * 16);
                const r4d_u32x4 a1 = *(const r4d_u32x4*)(xb + (((t * 2 + 1) * 2 + h) * 16 + nl) * 16);
#pragma unroll
                for (int r = 0; r < WR; ++r) {
                    xw_v8f P = xw_wmma(a0, bw[r][0], z);
                    P = xw_wmma(a1, bw[r][1], P);
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc[r][t][i] = __builtin_fmaf(d[r], P[i], acc[r][t][i]);
                }
            }
        };
        for (int b0 = 0; b0 < A.UK; b0 += A.SS) {
            const int b1 = min(b0 + A.SS, A.UK);
            if constexpr ((AB & 4) == 0) {
                lds_barrier();
                xw_stage<NT>(A, xs, r0, mc, nt, b0, b1);
                lds_barrier();
            }
#pragma unroll 1
            for (int bb = b0 - b0 % PD; bb < b1; bb += PD) {
#pragma unroll
                for (int k = 0; k < PD; ++k)
                    if (bb + k >= b0 && bb + k < b1) block(bb + k, xs + (bb + k - b0) * NT * 1024, raw[k]);
            }
        }
#pragma unroll
        for (int r = 0; r < WR; ++r) {
            const int n = nw + 16 * r;
            if (n >= A.N) continue;
#pragma unroll
            for (int t = 0; t < NT; ++t) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int m = 16 * t + 8 * h + i;
                    if (m < mc) {
                        const bool live = mx_moe_tok(A.sorted, r0 + m, A.top_k, A.a_sorted, A.AR) >= 0;
                        mx_put(A.y + (long long)(r0 + m) * A.y_ld + n, live ? acc[r][t][i] : 0.0f);
                    }
                }
            }
        }
    }
}

/* xm_wq_kernel (DRAFTER track 18.11.2026; tools/dbench/pat.cpp): the WMMA form with the weight rows read in long runs. A lane
 * half reads its row's half of a 16-block super-step (blocks 8 h .. 8 h + 7: 136 contiguous bytes), so a row is visited 272
 * bytes at a time instead of a 16-byte window a block (pat on the drafter's gate_up rows, 12 live experts: 583 vs 140 GB/s;
 * linear 627), and the two halves of a row trade the pieces the other needs (v_permlanex16). One 16-row tile a wave, 8 waves
 * (128 rows) a block, the next super-step's 34 words loaded while the current one is computed. Per output: the blocks in the
 * order 0, 8, 1, 9, .., 7, 15 of each super-step, each block's partial by its two WMMAs from 0 and one fma -- a fixed order
 * (rule B), otherwise xm_wm_kernel's operations. Needs UK % 8 == 0 (the drafter's K 5120 / 2304: 160 / 72 blocks). */
constexpr int kWqWords = 34;             /* 136 bytes: a lane half of a super-step */

template <int NT>
__global__ __launch_bounds__(256) void xm_wq_kernel(MxMrArgs A) {
    constexpr int MT = 16 * NT;
    R4D_DYN_LDS(r4d_u32x4, xm_lds);
    uint8_t* const smem = (uint8_t*)xm_lds;
    __shared__ int s_n, s_items;
    const int lane = (int)threadIdx.x & 31, wave = (int)threadIdx.x >> 5;
    const int nl = lane & 15, h = lane >> 4;
    __shared__ uint32_t xw_tab[256];            /* static: its address folds into the ds_load offset */
    uint32_t* const tab = xw_tab;
    for (int b = (int)threadIdx.x; b < 256; b += (int)blockDim.x)
        tab[b] = xw_bf(kMxValues[b & 15]) | (xw_bf(kMxValues[b >> 4]) << 16);
    const XmLive L = xm_live(smem, A);
    xm_count<MT>(A, L, &s_n, &s_items);
    __syncthreads();
    const int nlive = s_n, nitems = s_items;
    uint8_t* const xs = smem + A.x_off;
    const int NSS = (A.UK + 15) >> 4;
    for (int it = (int)blockIdx.x; it < nitems; it += (int)gridDim.x) {
        int e = 0, r0 = 0, r1 = 0, rg = 0;
        xm_item<MT>(L, nlive, it, &e, &r0, &r1, &rg);
        const int mc = r1 - r0, n0 = rg * A.RB;
        int ml = 0;
        for (int m0 = 0; m0 < mc; m0 += 32) {
            const int m = m0 + lane;
            const bool lv = m < mc && mx_moe_tok(A.sorted, r0 + m, A.top_k, A.a_sorted, A.AR) >= 0;
            const uint32_t bal = __builtin_amdgcn_ballot_w32(lv);
            if (bal) ml = m0 + 32 - __builtin_clz(bal);
        }
        ml = __builtin_amdgcn_readfirstlane(ml);
        if (ml == 0) {
            xm_zero(A, r0, mc, n0);
            continue;
        }
        const int nt = (ml + 15) >> 4;
        const uint8_t* wb = A.wtab ? A.wtab[e] : A.w + (long long)e * A.w_es;
        const int nw = n0 + wave * 16 + nl;
        const uint8_t* row = wb + (long long)min(nw, A.N - 1) * A.w_ld;
        xw_v8f acc[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[t][i] = 0.0f;
        auto load = [&](int ss, uint32_t (&d)[kWqWords]) {
            if (ss < NSS) {
                const int hb = 16 * ss + 8 * h;
                const mx_gu32 g = (mx_gu32)(const void*)(row + (hb < A.UK ? (long long)hb * 17 : 0));
#pragma unroll
                for (int i = 0; i < kWqWords; ++i) d[i] = g[i];
            }
        };
        const xw_v8f z = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
        auto blk = [&](const uint8_t* xb, uint32_t q0, uint32_t q1, uint32_t ex) {
            r4d_u32x4 bw0, bw1;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                bw0[p] = tab[(q0 >> (8 * p)) & 0xffu];
                bw1[p] = tab[(q1 >> (8 * p)) & 0xffu];
            }
            const float d = mx_e8m0_half(ex);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                if (t >= nt) break;
                const r4d_u32x4 a0 = *(const r4d_u32x4*)(xb + (((t * 2 + 0) * 2 + h) * 16 + nl) * 16);
                const r4d_u32x4 a1 = *(const r4d_u32x4*)(xb + (((t * 2 + 1) * 2 + h) * 16 + nl) * 16);
                xw_v8f P = xw_wmma(a0, bw0, z);
                P = xw_wmma(a1, bw1, P);
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[t][i] = __builtin_fmaf(d, P[i], acc[t][i]);
            }
        };
        auto step = [&](int ss, uint32_t (&cur)[kWqWords], uint32_t (&nxt)[kWqWords]) {
            load(ss + 1, nxt);
            const int b0 = 16 * ss, b1 = min(b0 + 16, A.UK);
            lds_barrier();
            xw_stage<NT>(A, xs, r0, mc, nt, b0, b1);
            lds_barrier();
            const bool hi = b0 + 8 < A.UK;
            mx_for<8>([&](auto jj) {
                constexpr int j = decltype(jj)::value, o = 17 * j;
                auto dw = [&](auto oc) {
                    constexpr int q = decltype(oc)::value;
                    if constexpr ((q & 3) == 0) return cur[q >> 2];
                    else return (uint32_t)__builtin_amdgcn_alignbyte(cur[(q >> 2) + 1], cur[q >> 2], q & 3);
                };
                const uint32_t pA0 = dw(std::integral_constant<int, o + 1>{}), pB0 = dw(std::integral_constant<int, o + 5>{});
                const uint32_t pA1 = dw(std::integral_constant<int, o + 9>{}), pB1 = dw(std::integral_constant<int, o + 13>{});
                const uint32_t E = (cur[o >> 2] >> (8 * (o & 3))) & 0xffu;
                const uint32_t s0 = h ? pA0 : pB0, s1 = h ? pA1 : pB1;
                const uint32_t v0 = (uint32_t)__builtin_amdgcn_permlanex16((int)s0, (int)s0, 0x76543210, 0xfedcba98, false, false);
                const uint32_t v1 = (uint32_t)__builtin_amdgcn_permlanex16((int)s1, (int)s1, 0x76543210, 0xfedcba98, false, false);
                const uint32_t Ep = (uint32_t)__builtin_amdgcn_permlanex16((int)E, (int)E, 0x76543210, 0xfedcba98, false, false);
                /* block j (lane half 0's run) and block j + 8 (half 1's) */
                blk(xs + j * NT * 1024, h ? v0 : pA0, h ? v1 : pA1, h ? Ep : E);
                if (hi) blk(xs + (j + 8) * NT * 1024, h ? pB0 : v0, h ? pB1 : v1, h ? E : Ep);
            });
        };
        uint32_t bufA[kWqWords], bufB[kWqWords];
        load(0, bufA);
#pragma unroll 1
        for (int ss = 0; ss < NSS; ss += 2) {
            step(ss, bufA, bufB);
            if (ss + 1 < NSS) step(ss + 1, bufB, bufA);
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int m = 16 * t + 8 * h + i;
                if (m < mc && nw < A.N) {
                    const bool live = mx_moe_tok(A.sorted, r0 + m, A.top_k, A.a_sorted, A.AR) >= 0;
                    mx_put(A.y + (long long)(r0 + m) * A.y_ld + nw, live ? acc[t][i] : 0.0f);
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ forms */
/* gemv band: (H, R, MT); tiled order: (R, MT). */
#define XM_GFORMS(X) X(4, 1, 2) X(4, 1, 4) X(4, 1, 8) X(4, 1, 16) X(4, 2, 2) X(4, 2, 4) X(4, 2, 8) X(2, 1, 4) X(2, 1, 8)
#define XM_SFORMS(X) X(1, 2, 0) X(1, 4, 0) X(1, 8, 0) X(2, 2, 0) X(2, 4, 0) X(2, 8, 0) \
    X(1, 4, 1) X(1, 8, 1) X(1, 16, 1) X(2, 4, 1) X(2, 8, 1) X(2, 16, 1) X(1, 4, 3) X(1, 4, 5) X(1, 4, 7) \
    X(1, 4, 9) X(1, 8, 9) X(1, 16, 9) X(2, 8, 9) X(1, 4, 25) X(1, 8, 25) X(1, 16, 25) X(2, 8, 25)

int xm_launch_g(RadStream s, int H, int R, int MT, const MxMrArgs& A, unsigned grid, unsigned thr, unsigned lds) {
#define XM_G(h, r, mt) \
    if (H == h && R == r && MT == mt) return R4D_EMIT(s, (xm_gemv_kernel<MX_FP4, h, r, mt>), dim3(grid), dim3(thr), lds, A);
    XM_GFORMS(XM_G)
#undef XM_G
    return R4D_MX_MR_DECLINE;
}
int xm_launch_s(RadStream s, int R, int MT, int EX, const MxMrArgs& A, unsigned grid, unsigned thr, unsigned lds) {
#define XM_S(r, mt, ex) \
    if (R == r && MT == mt && EX == ex) return R4D_EMIT(s, (xm_seq_kernel<MX_FP4, r, mt, ex>), dim3(grid), dim3(thr), lds, A);
    XM_SFORMS(XM_S)
#undef XM_S
    return R4D_MX_MR_DECLINE;
}
/* WMMA form: (WR, NT, PD, AB). */
#define XM_WFORMS(X) X(1, 1, 2, 0) X(1, 1, 4, 0) X(1, 2, 4, 0) X(1, 4, 2, 0) X(2, 1, 2, 0) X(2, 1, 4, 0) X(2, 1, 6, 0) X(2, 2, 2, 0) \
    X(2, 2, 4, 0) X(2, 4, 2, 0) X(4, 1, 2, 0) X(2, 1, 2, 1) X(2, 1, 2, 2) X(2, 1, 2, 4) X(2, 1, 2, 3) X(2, 1, 2, 7) X(2, 1, 2, 5)
int xm_launch_w(RadStream s, int WR, int NT, int PD, int AB, const MxMrArgs& A, unsigned grid, unsigned thr, unsigned lds) {
#define XM_W(wr, nt, pd, ab) \
    if (WR == wr && NT == nt && PD == pd && AB == ab) return R4D_EMIT(s, (xm_wm_kernel<wr, nt, pd, ab>), dim3(grid), dim3(thr), lds, A);
    XM_WFORMS(XM_W)
#undef XM_W
    return R4D_MX_MR_DECLINE;
}
bool xm_has_w(int WR, int NT, int PD, int AB) {
#define XM_WH(wr, nt, pd, ab) if (WR == wr && NT == nt && PD == pd && AB == ab) return true;
    XM_WFORMS(XM_WH)
#undef XM_WH
    return false;
}
int xm_launch_q(RadStream s, int NT, const MxMrArgs& A, unsigned grid, unsigned lds) {
    if (NT == 1) return R4D_EMIT(s, (xm_wq_kernel<1>), dim3(grid), dim3(256), lds, A);
    if (NT == 2) return R4D_EMIT(s, (xm_wq_kernel<2>), dim3(grid), dim3(256), lds, A);
    return R4D_MX_MR_DECLINE;
}
bool xm_has_g(int H, int R, int MT) {
#define XM_GH(h, r, mt) if (H == h && R == r && MT == mt) return true;
    XM_GFORMS(XM_GH)
#undef XM_GH
    return false;
}
bool xm_has_s(int R, int MT, int EX) {
#define XM_SH(r, mt, ex) if (R == r && MT == mt && EX == ex) return true;
    XM_SFORMS(XM_SH)
#undef XM_SH
    return false;
}

int xm_cu_count() {
#ifdef R4D_EMU_HOST
    return 64;
#else
    static int cache[64];
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) return 64;
    if (cache[dev] <= 0) {
        int v = 0;
        if (hipDeviceGetAttribute(&v, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess || v <= 0) v = 64;
        cache[dev] = v;
    }
    return cache[dev];
#endif
}

/* ------------------------------------------------------------------ the switch, the pin, the rule */
int env_drafter_mx() {
    const char* v = std::getenv("RADIANCE_DS41_DRAFTER_MX");
    const int on = (v && (*v == '1' || *v == '2') && !v[1]) ? *v - '0' : 0;
    if (v && *v && !on && std::strcmp(v, "0"))
        std::fprintf(stderr, "radiance: libr4d: RADIANCE_DS41_DRAFTER_MX=%s is not 0 / 1 / 2: the stock MXFP4 grouped GEMM\n", v);
    if (on == 1)
        std::fprintf(stderr, "radiance: libr4d: moe_gemm_ggml type=mxfp4 takes the multi-row forms (RADIANCE_DS41_DRAFTER_MX=1, "
                             "the fn4 switch drafter_mx)\n");
    if (on == 2)
        std::fprintf(stderr, "radiance: libr4d: moe_gemm_ggml type=mxfp4 takes the WMMA form at every call (RADIANCE_DS41_DRAFTER_MX=2, "
                             "the fn4 switch drafter_mx 2: rule B, the drafter's own reference)\n");
    return on;
}
int g_xm_on = env_drafter_mx();
int g_xm_pin = 0;                         /* 0 none, -1 the rule forced, 1 a pinned form */
int g_xm_pin_h = 4, g_xm_pin_r = 1, g_xm_pin_mt = 4, g_xm_pin_wn = 0, g_xm_pin_grid = 0;
int g_xm_cap = kXmMaxTokens;              /* the forms' token bound (bench hook r4d_mx_mr_cap) */
char g_xm_last[160];
int g_xm_wm_slab = 0;                     /* WMMA form: LDS bytes of a slab (0: 16 KiB; bench hook) */

}  // namespace

int r4d_mx_mr_on() { return g_xm_pin != 0 ? 1 : g_xm_on; }
void r4d_mx_mr_set(int on) { g_xm_on = on == 2 ? 2 : (on ? 1 : 0); }
int r4d_mx_mr_get() { return g_xm_on; }

/* The WMMA form's launch (rule B): WR 16-row weight tiles a wave, segments of up to 16 NT rows, WN waves a block. */
static int xm_wm_moe(RadStream s, const R4dMxMrCall& c, int WR, int NT, int PD, int AB, int WN, int gridcap) {
    if (!xm_has_w(WR, NT, PD, AB) || (WN != 1 && WN != 2 && WN != 4 && WN != 8)) return R4D_MX_MR_DECLINE;
    MxMrArgs A{};
    A.x = c.x; A.x_ld = c.x_ld; A.w = c.w; A.wtab = c.wtab; A.w_ld = c.w_ld; A.w_es = c.w_es;
    A.y = c.y; A.y_ld = c.y_ld; A.sorted = c.sorted; A.off = c.off;
    A.T = c.T; A.N = c.N; A.K = c.K; A.n_expert = c.n_expert; A.top_k = c.top_k; A.a_sorted = c.a_sorted;
    A.AR = c.AR; A.x_al = c.x_al;
    A.UK = c.K / 32;
    A.NS = A.UK;
    A.RB = WN * 16 * WR;
    A.nrg = (c.N + A.RB - 1) / A.RB;
    const long long sb = (long long)NT * 1024;   /* bytes a block of the segment's tiles */
    long long SS = (g_xm_wm_slab > 0 ? g_xm_wm_slab : 16384) / sb;
    if (SS < 1) SS = 1;
    if (SS > A.NS) SS = A.NS;
    A.SS = (int)SS;
    A.live_cap = c.n_expert < c.T ? c.n_expert : c.T;
    A.live_off = kXwTab;
    A.x_off = (int)(kXwTab + (4LL * A.live_cap * 4 + 15) / 16 * 16);
    const long long lds = A.x_off + SS * sb;
    if (lds > 65536) return R4D_MX_MR_DECLINE;
    const long long est = (long long)A.live_cap * A.nrg;
    long long grid = gridcap > 0 ? gridcap : 32LL * xm_cu_count();
    if (grid > est) grid = est;
    if (grid < 1) grid = 1;
    std::snprintf(g_xm_last, sizeof g_xm_last, "wmma WR%d NT%d PD%d%s WN%d SS%d RB%d grid %lld lds %lld", WR, NT, PD, AB ? " ABL" : "", WN, A.SS,
                  A.RB, grid, lds);
    return xm_launch_w(s, WR, NT, PD, AB, A, (unsigned)grid, 32u * WN, (unsigned)lds);
}

/* The long-run WMMA form's launch (rule B): 8 waves of one 16-row tile, segments of up to 16 NT rows, super-steps of 16 blocks. */
static int xm_wq_moe(RadStream s, const R4dMxMrCall& c, int NT, int gridcap) {
    if ((NT != 1 && NT != 2) || (c.K / 32) % 8 != 0) return R4D_MX_MR_DECLINE;
    MxMrArgs A{};
    A.x = c.x; A.x_ld = c.x_ld; A.w = c.w; A.wtab = c.wtab; A.w_ld = c.w_ld; A.w_es = c.w_es;
    A.y = c.y; A.y_ld = c.y_ld; A.sorted = c.sorted; A.off = c.off;
    A.T = c.T; A.N = c.N; A.K = c.K; A.n_expert = c.n_expert; A.top_k = c.top_k; A.a_sorted = c.a_sorted;
    A.AR = c.AR; A.x_al = c.x_al;
    A.UK = c.K / 32;
    A.NS = A.UK;
    A.RB = 8 * 16;
    A.nrg = (c.N + A.RB - 1) / A.RB;
    A.SS = 16;
    A.live_cap = c.n_expert < c.T ? c.n_expert : c.T;
    A.live_off = kXwTab;
    A.x_off = (int)(kXwTab + (4LL * A.live_cap * 4 + 15) / 16 * 16);
    const long long lds = A.x_off + 16LL * NT * 1024;
    if (lds > 65536) return R4D_MX_MR_DECLINE;
    const long long est = (long long)A.live_cap * A.nrg;
    long long grid = gridcap > 0 ? gridcap : 32LL * xm_cu_count();
    if (grid > est) grid = est;
    if (grid < 1) grid = 1;
    std::snprintf(g_xm_last, sizeof g_xm_last, "wmma-run NT%d WN8 SS16 RB128 grid %lld lds %lld", NT, grid, lds);
    return xm_launch_q(s, NT, A, (unsigned)grid, (unsigned)lds);
}

/* The rule: the gemv band (<= 16 tokens): MT from the call's tokens (an expert's rows in a decode step are few: 4 to 8
 * tokens' worth), H 4 x R 1 in 8-wave blocks (32 weight rows a block), a grid of four blocks a WGP or fewer when the live
 * work is smaller. The tiled order (DRAFTER track 18.11.2026, tools/dbench on one R9700, the drafter's 3 stages x
 * (gate_up + down)): R 1 x MT 8 with EX 25 (each segment's chains over its own rows only, decoded a group of 8 values at a
 * time, each step one fma: the same bits, the product being exact) in 4-wave blocks (128 rows), slabs of 16 KiB, a grid of 32
 * blocks a WGP. drafter_mx 2 takes every call (both bands) to the WMMA form instead (xm_wm_kernel WR 1 NT 1 PD 2, 8 waves; rule B;
 * lq125 real routing, one card: 80 tokens 1.26-1.53 ms vs EX 25's 2.68-3.42, 40 tokens 0.85-1.18 vs 1.20-2.15). Synthetic routing (pool 6, 7 in 10): 80
 * tokens 4.18 ms a pass vs the stock tiled kernel's 18.62 (x4.46), 40 tokens 2.94 vs the previous rule's (MT 8, no EX, 4
 * blocks a WGP) 5.99; routing where a sequence's 5 rows share 3 experts (pool 3, all picks; its stock time 11.56 ms is
 * within 8 % of the served 12.48): 80 tokens 3.15 vs 11.56, 40 tokens 2.18 vs 6.91. The same bytes (the stock chain order). */
int r4d_mx_mr_moe(RadStream s, const R4dMxMrCall& c) {
    if (c.tokens < 1 || c.tokens > g_xm_cap || c.T < 1 || c.N < 1 || c.K < 32 || c.K % 32 || c.n_expert < 1)
        return R4D_MX_MR_DECLINE;
    const bool pinned = g_xm_pin > 0;
    if (pinned && g_xm_pin_h == 4096) return xm_wq_moe(s, c, g_xm_pin_mt, g_xm_pin_grid);   /* long-run WMMA pin */
    /* drafter_mx 2 (rule B): every call, either band, in the WMMA form (WR 1, NT 1, PD 2, 8 waves), so a sequence's drafter
     * bits do not depend on how many sequences share the call; what it declines takes the exact forms below. */
    if (!pinned && g_xm_pin == 0 && g_xm_on == 2) {
        const int wm = xm_wm_moe(s, c, 1, 1, 2, 0, 8, 0);
        if (wm != R4D_MX_MR_DECLINE) return wm;
    }
    if (pinned && g_xm_pin_h >= 1024)   /* WMMA pin (any band): h = 1024 + PD + 16 AB */
        return xm_wm_moe(s, c, g_xm_pin_r, g_xm_pin_mt, (g_xm_pin_h & 15) ? (g_xm_pin_h & 15) : 2, (g_xm_pin_h >> 4) & 15,
                         g_xm_pin_wn > 0 ? g_xm_pin_wn : 4, g_xm_pin_grid);
    int H = 4, R = 1, MT = c.tokens <= 2 ? 2 : (c.tokens <= 8 ? 4 : 8), WN = 0;
    if (pinned) { H = g_xm_pin_h; R = g_xm_pin_r; MT = g_xm_pin_mt; WN = g_xm_pin_wn; }
    /* a seq pin's h: 1 = EX (the segment's rows only) */
    int EX = pinned && !c.gemv ? (g_xm_pin_h & 1 ? g_xm_pin_h : 0) : 0;   /* 3 / 5 / 7: bench ablations */
    if (!pinned && !c.gemv) { MT = 8; EX = 25; }
    if (c.gemv ? !xm_has_g(H, R, MT) : !xm_has_s(R, MT, EX)) return R4D_MX_MR_DECLINE;
    if (WN <= 0) WN = c.gemv ? 8 : 4;
    if (WN != 1 && WN != 2 && WN != 4 && WN != 8) return R4D_MX_MR_DECLINE;
    MxMrArgs A{};
    A.x = c.x; A.x_ld = c.x_ld; A.w = c.w; A.wtab = c.wtab; A.w_ld = c.w_ld; A.w_es = c.w_es;
    A.y = c.y; A.y_ld = c.y_ld; A.sorted = c.sorted; A.off = c.off;
    A.T = c.T; A.N = c.N; A.K = c.K; A.n_expert = c.n_expert; A.top_k = c.top_k; A.a_sorted = c.a_sorted;
    A.AR = c.AR; A.x_al = c.x_al;
    A.UK = c.K / 32;
    long long sb;                              /* bytes a stage (block) of the segment's rows */
    if (c.gemv) {
        const int LPR = 32 / H;
        A.NS = (A.UK + LPR - 1) / LPR;
        A.RB = WN * H * R;
        sb = (long long)MT * 32 * LPR * 4;
    } else {
        A.NS = A.UK;
        A.RB = WN * 32 * R;
        sb = (long long)MT * 32 * 4;
    }
    A.nrg = (c.N + A.RB - 1) / A.RB;
    long long SS = 16384 / sb;
    if (SS < 1) SS = 1;
    if (SS > A.NS) SS = A.NS;
    A.SS = (int)SS;
    A.live_cap = c.n_expert < c.T ? c.n_expert : c.T;
    const long long tb = 64;                   /* Mx<MX_FP4>::TB, rounded to 16 */
    A.live_off = (int)tb;
    A.x_off = (int)(tb + (4LL * A.live_cap * 4 + 15) / 16 * 16);
    const long long lds = A.x_off + SS * sb;
    if (lds > 65536) return R4D_MX_MR_DECLINE;
    /* items: at most min(n_expert, T) experts live, each one segment or more */
    const long long est = (long long)A.live_cap * A.nrg;
    long long grid = g_xm_pin_grid > 0 && pinned ? g_xm_pin_grid : (c.gemv ? 4LL : 32LL) * xm_cu_count();
    if (grid > est) grid = est;
    if (grid < 1) grid = 1;
    std::snprintf(g_xm_last, sizeof g_xm_last, "%s H%d R%d MT%d%s WN%d SS%d RB%d grid %lld lds %lld", c.gemv ? "gemv" : "seq",
                  c.gemv ? H : 32, R, MT, EX ? " EX" : "", WN, A.SS, A.RB, grid, lds);
    return c.gemv ? xm_launch_g(s, H, R, MT, A, (unsigned)grid, 32u * WN, (unsigned)lds)
                  : xm_launch_s(s, R, MT, EX, A, (unsigned)grid, 32u * WN, (unsigned)lds);
}

/* r4d_mx_mr_pin (bench / test hook, r4d_selftest --mxcheck / --perfmx): on 0 clears the pin (the switch decides),
 * -1 forces the rule, 1 pins (H, R, MT, WN; the tiled order reads R and MT) and a grid cap (0: the rule's). */
extern "C" void r4d_mx_mr_pin(int on, int h, int r, int mt, int wn, int grid) {
    g_xm_pin = on == 1 || on == -1 ? on : 0;
    if (on == 1) { g_xm_pin_h = h; g_xm_pin_r = r; g_xm_pin_mt = mt; g_xm_pin_wn = wn; }
    g_xm_pin_grid = grid > 0 ? grid : 0;
}
extern "C" const char* r4d_mx_mr_last() { return g_xm_last; }
/* r4d_mx_mr_cap (bench hook, DRAFTER track): the forms' token bound (<= 0: the shipped kXmMaxTokens). */
extern "C" void r4d_mx_mr_cap(int tokens) { g_xm_cap = tokens > 0 ? tokens : kXmMaxTokens; }
/* r4d_mx_wm_slab (bench hook, DRAFTER track): the WMMA form's slab in LDS bytes (<= 0: 16 KiB). */
extern "C" void r4d_mx_wm_slab(int bytes) { g_xm_wm_slab = bytes > 0 ? bytes : 0; }

#ifdef R4D_EMU_HOST
/* tests/r4d_emu (ds41mx): the stock grouped GEMM's launches as mx_moe issues them (gemv band or tiled), from this
 * unit's copies of mx_gemv_kernel / mx_tiled_kernel -- the plugin's source. */
long long xm_emu_xbuf(int xm, long long K, int* xlm, int* xr, int* kc) {
    *xr = (int)(K + K / 8);
    const long long f = 9216 / (K + K / 8);
    *xlm = (int)(f < xm ? f : xm);
    long long cc = 9216 / ((long long)xm * 1152);
    if (cc < 1) cc = 1;
    const long long kr = (K + 1023) / 1024 * 1024;
    *kc = (int)(1024LL * cc < kr ? 1024LL * cc : kr);
    const long long whole = (long long)*xlm * *xr, chunk = (long long)xm * (*kc + *kc / 8);
    return *xlm >= xm ? whole : (whole > chunk ? whole : chunk);
}
int r4d_mx_mr_emu_stock(const R4dMxMrCall& c, int bm) {
    const long long tb = 64, T = c.T, ne = c.n_expert, L = ne < T ? ne : T;
    if (c.gemv) {
        MxGemvArgs A{};
        A.x = c.x; A.x_ld = c.x_ld; A.w = c.w; A.wtab = c.wtab; A.w_ld = c.w_ld; A.w_es = c.w_es;
        A.y = c.y; A.y_ld = c.y_ld; A.sorted = c.sorted; A.off = c.off;
        A.M = c.T; A.N = c.N; A.K = c.K;
        A.XM = c.tokens < kMxXM ? (c.tokens < 1 ? 1 : c.tokens) : kMxXM;
        const long long xbuf = xm_emu_xbuf(A.XM, c.K, &A.XLM, &A.XR, &A.KC);
        const int rpw = c.N >= 1024 ? 2 : 1;
        const long long nrg = (c.N + 8 * rpw - 1) / (8 * rpw);
        long long rgpi = (L * nrg + 255) / 256;
        if (rgpi < 1) rgpi = 1;
        const long long ipe = (nrg + rgpi - 1) / rgpi;
        A.rgpi = (int)rgpi; A.ipe = (int)ipe;
        A.n_expert = c.n_expert; A.top_k = c.top_k; A.a_sorted = c.a_sorted; A.AR = c.AR; A.x_al = c.x_al;
        A.live_off = (int)tb;
        A.x_off = (int)(tb + (ne * 4 + 15) / 16 * 16);
        const long long lds = A.x_off + xbuf * 4, items = L * ipe;
        const unsigned grid = (unsigned)(items < 256 ? items : 256);
        if (rpw == 2) return R4D_EMIT(nullptr, (mx_gemv_kernel<MX_FP4, 2, true>), dim3(grid), dim3(kMxThreads), (unsigned)lds, A);
        return R4D_EMIT(nullptr, (mx_gemv_kernel<MX_FP4, 1, true>), dim3(grid), dim3(kMxThreads), (unsigned)lds, A);
    }
    MxTiledArgs A{};
    A.x = c.x; A.x_ld = c.x_ld; A.w = c.w; A.wtab = c.wtab; A.w_ld = c.w_ld; A.w_es = c.w_es;
    A.y = c.y; A.y_ld = c.y_ld; A.sorted = c.sorted; A.off = c.off;
    A.M = c.T; A.N = c.N; A.K = c.K;
    A.n_expert = c.n_expert; A.top_k = c.top_k; A.a_sorted = c.a_sorted; A.AR = c.AR; A.x_al = c.x_al;
    A.tab_bytes = (int)tb;
    const long long gx = (T + bm - 1) / bm + L, gy = (c.N + kTN - 1) / kTN;
    const unsigned lds = (unsigned)(tb + 32 * (kTP + bm + 4) * 4);
    if (bm == 32) return R4D_EMIT(nullptr, (mx_tiled_kernel<MX_FP4, 32, true>), dim3((unsigned)gx, (unsigned)gy), dim3(kMxThreads), lds, A);
    return R4D_EMIT(nullptr, (mx_tiled_kernel<MX_FP4, 128, true>), dim3((unsigned)gx, (unsigned)gy), dim3(kMxThreads), lds, A);
}
#endif
