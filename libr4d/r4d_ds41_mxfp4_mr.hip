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

constexpr int kXmPre = 4;                 /* xm_count: offset chunks (32 experts) loaded up front (the drafter's 128 under DEP) */
/* Wave 0: the live experts and their items (segments x row groups), counted in expert order. */
template <int MT>
__device__ __forceinline__ void xm_count(const MxMrArgs& A, XmLive L, int* s_n, int* s_items) {
    const int lane = (int)threadIdx.x & 31;
    if (((int)threadIdx.x >> 5) != 0) return;
    int cnt = 0;
    uint32_t base = 0;
    /* round 10: the first kXmPre chunks' offsets loaded up front (one round trip instead of one a chunk; the same values) */
    int pa[kXmPre], pb[kXmPre];
#pragma unroll
    for (int k = 0; k < kXmPre; ++k) {
        const int e = 32 * k + lane;
        pa[k] = 0; pb[k] = 0;
        if (e < A.n_expert) {
            pa[k] = e == 0 ? 0 : A.off[e];
            pb[k] = e == A.n_expert - 1 ? A.T : A.off[e + 1];
        }
    }
    for (int e0 = 0; e0 < A.n_expert; e0 += 32) {
        const int e = e0 + lane;
        int s = 0, t = 0;
        if (e < A.n_expert) {
            int a = 0, b = 0;
            bool got = false;
#pragma unroll
            for (int k = 0; k < kXmPre; ++k)
                if (e0 == 32 * k) { a = pa[k]; b = pb[k]; got = true; }
            if (!got) { a = e == 0 ? 0 : A.off[e]; b = e == A.n_expert - 1 ? A.T : A.off[e + 1]; }
            a = a < 0 ? 0 : (a > A.T ? A.T : a);
            b = b < a ? a : (b > A.T ? A.T : b);
            s = a; t = b;
        }
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

/* xw_stage split in two (DRAFTER track, xm_wq AB 16): the slab's units into registers (global / L2 loads in flight while the
 * current slab is computed), then into LDS. UPT units a thread at 256 threads (16 blocks x NT tiles x 64 units). */
template <int NT>
__device__ __forceinline__ void xw_stage_ld(const MxMrArgs& A, int r0, int mc, int nt, int b0, int b1, r4d_u32x4 (&rg)[4 * NT]) {
    const int np = (b1 - b0) * nt * 64;
#pragma unroll
    for (int k = 0; k < 4 * NT; ++k) {
        const int idx = (int)threadIdx.x + 256 * k;
        r4d_u32x4 o = { 0u, 0u, 0u, 0u };
        if (idx < np) {
            const int n = idx & 15, jh = (idx >> 4) & 3, rest = idx >> 6;
            const int t = rest % nt, bb = rest / nt;
            const int m = 16 * t + n;
            const int tok = m < mc ? mx_moe_tok(A.sorted, r0 + m, A.top_k, A.a_sorted, A.AR) : -1;
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
        }
        rg[k] = o;
    }
}
template <int NT>
__device__ __forceinline__ void xw_stage_st(uint8_t* xs, int nt, int b0, int b1, const r4d_u32x4 (&rg)[4 * NT]) {
    const int np = (b1 - b0) * nt * 64;
#pragma unroll
    for (int k = 0; k < 4 * NT; ++k) {
        const int idx = (int)threadIdx.x + 256 * k;
        if (idx < np) {
            const int n = idx & 15, jh = (idx >> 4) & 3, rest = idx >> 6;
            const int t = rest % nt, bb = rest / nt;
            *(r4d_u32x4*)(xs + ((((bb * NT + t) * 2 + (jh >> 1)) * 2 + (jh & 1)) * 16 + n) * 16) = rg[k];
        }
    }
}

/* IL (DRAFTER track 19.11.2026 round 27, lever WIL; bench form): the weights in the TILE-INTERLEAVED layout xm_il_pack writes
 * (r4d_ds41_mxfp4_mr.h): an expert's rows in 16-row groups, a group's blocks in K order, each (group, block) 272 bytes = lane
 * L's 8 code bytes (qs[4 h .. 4 h + 3], qs[8 + 4 h .. 8 + 4 h + 3] of row L % 16, h = L / 16) at 8 L, then the 16 rows' E8M0
 * scales at 256 + row -- the same 17 bytes a block, so a wave's 32 lanes read 256 contiguous bytes a block (one 64-bit load a
 * lane, no window realignment, no cross-half scale shuffle). The operands each WMMA sees are the 17-byte form's (the same
 * nibbles in the same slots, the same scale): IL moves only where the bytes sit, not one operation (the same bytes). */
typedef __attribute__((address_space(1))) const uint64_t* mx_gu64;
typedef __attribute__((address_space(1))) const uint8_t* mx_gu8;
constexpr int kXmIlTile = 272;

template <int WR, int NT, int PD, int AB = 0, int IL = 0>   /* AB: bench ablations (outputs differ): 1 no WMMA / fma, 2 no decode, 4 no staging */
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
        for (int r = 0; r < WR; ++r)
            wr[r] = IL ? wb + (long long)(min(nw + 16 * r, A.N - 1) >> 4) * A.UK * kXmIlTile
                       : wb + (long long)min(nw + 16 * r, A.N - 1) * A.w_ld;
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
                    if constexpr (IL != 0) {
                        const uint8_t* tp = wr[r] + (long long)b * kXmIlTile;
                        const uint64_t v = *(mx_gu64)(const void*)(tp + 8 * lane);
                        rw[r][0] = (uint32_t)v; rw[r][1] = (uint32_t)(v >> 32);
                        rw[r][2] = *(mx_gu8)(const void*)(tp + 256 + nl);
                    } else {
                        const uintptr_t pa = ((uintptr_t)(wr[r] + (long long)b * Q::BB) & ~(uintptr_t)3) + 4 * h;
                        const mx_gu32 g = (mx_gu32)(const void*)pa;
                        rw[r][0] = g[0]; rw[r][1] = g[1]; rw[r][2] = g[2]; rw[r][3] = g[3];
                    }
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
                if constexpr (IL != 0) {
                    q0[r] = w[0]; q1[r] = w[1];
                    d[r] = mx_e8m0_half(w[2]);
                } else {
                const int sA = (sb[r] + b) & 3;           /* the block's byte offset in its window (17 b = b mod 4) */
                const int sh = (sA + 1) & 3;
                const bool s3 = sA == 3;
                q0[r] = __builtin_amdgcn_alignbyte(w[1], s3 ? w[1] : w[0], sh);
                q1[r] = __builtin_amdgcn_alignbyte(w[3], s3 ? w[3] : w[2], sh);
                const uint32_t e0 = (w[0] >> (8 * sA)) & 0xffu;
                const uint32_t e1 = (uint32_t)__builtin_amdgcn_permlanex16((int)e0, (int)e0, 0x76543210, 0xfedcba98, false, false);
                d[r] = mx_e8m0_half(h ? e1 : e0);
                }
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

/* xm_wt_kernel (DRAFTER track 19.11.2026 round 12, lever DTS; tools/dbench/patd.cpp): xm_wm_kernel WR 1 NT 1 with TS row tiles a
 * wave IN SEQUENCE, for short K (lever DEP's down: K 576 = 18 blocks of 306-byte rows). xm_wm at 18 blocks an item spends its
 * time in the item's set-up (the live count, the item search, wtab / sorted / activation round trips, two barriers) and its
 * pipeline fill: patd on the down shape (33 live experts, 51.5 MB) reads the window pattern alone at 540 GB/s (95.7 us) and a
 * coalesced range at 605 (85.4), while xm_wm's down takes 116-162 us a call (loads-only ablation 119-136). Here an item is
 * (expert, segment, TS x WN x 16 rows): the activations are staged ONCE an item (the whole K: UK <= SS), the segment's live
 * rows read once, and the weight loads run PD blocks ahead straight across the tiles' boundaries (one stream of TS x UK
 * blocks a wave). Tile t's rows: n0 + (t WN + wave) 16 + nl (the block's waves read one contiguous range a tile).
 * Exactness: each output is xm_wm_kernel's (WR 1 NT 1): blocks in K order, a block's two WMMAs from 0, one fma of its scale,
 * one rounding -- TS and WN only move which lane computes it and when its words are loaded (the same bytes as drafter_mx 2). */
template <int PD, int TS, int IL = 0>   /* IL: the tile-interleaved weights (xm_wm_kernel's IL note) */
__global__ __launch_bounds__(kXmThreads) void xm_wt_kernel(MxMrArgs A) {
    using Q = Mx<MX_FP4>;
    constexpr int MT = 16;
    R4D_DYN_LDS(r4d_u32x4, xm_lds);
    uint8_t* const smem = (uint8_t*)xm_lds;
    __shared__ int s_n, s_items;
    const int lane = (int)threadIdx.x & 31, wave = (int)threadIdx.x >> 5;
    const int nl = lane & 15, h = lane >> 4;
    const int WN = (int)blockDim.x >> 5;
    __shared__ uint32_t xw_tab[256];
    uint32_t* const tab = xw_tab;
    for (int b = (int)threadIdx.x; b < 256; b += (int)blockDim.x)
        tab[b] = xw_bf(kMxValues[b & 15]) | (xw_bf(kMxValues[b >> 4]) << 16);
    const XmLive L = xm_live(smem, A);
    xm_count<MT>(A, L, &s_n, &s_items);
    __syncthreads();
    const int nlive = s_n, nitems = s_items;
    uint8_t* const xs = smem + A.x_off;
    const int UK = A.UK, NB = TS * UK;
    for (int it = (int)blockIdx.x; it < nitems; it += (int)gridDim.x) {
        int e = 0, r0 = 0, r1 = 0, rg = 0;
        xm_item<MT>(L, nlive, it, &e, &r0, &r1, &rg);
        const int mc = r1 - r0, n0 = rg * A.RB;
        const uint8_t* wb = A.wtab ? A.wtab[e] : A.w + (long long)e * A.w_es;
        auto rowp = [&](int t) -> const uint8_t* {
            if constexpr (IL != 0) return wb + (long long)(min(n0 + (t * WN + wave) * 16 + nl, A.N - 1) >> 4) * UK * kXmIlTile;
            return wb + (long long)min(n0 + (t * WN + wave) * 16 + nl, A.N - 1) * A.w_ld;
        };
        /* the load cursor (tile tl, block bl) and the first PD blocks in flight before the set-up's round trips */
        int tl = 0, bl = 0;
        const uint8_t* lrow = rowp(0);
        r4d_u32x4 raw[PD];
        auto load = [&](r4d_u32x4& rw) {
            if (tl < TS) {
                if constexpr (IL != 0) {
                    const uint8_t* tp = lrow + (long long)bl * kXmIlTile;
                    const uint64_t v = *(mx_gu64)(const void*)(tp + 8 * lane);
                    rw[0] = (uint32_t)v; rw[1] = (uint32_t)(v >> 32);
                    rw[2] = *(mx_gu8)(const void*)(tp + 256 + nl);
                } else {
                const uintptr_t pa = ((uintptr_t)(lrow + (long long)bl * Q::BB) & ~(uintptr_t)3) + 4 * h;
                const mx_gu32 g = (mx_gu32)(const void*)pa;
                rw[0] = g[0]; rw[1] = g[1]; rw[2] = g[2]; rw[3] = g[3];
                }
                if (++bl == UK) { bl = 0; ++tl; lrow = rowp(tl); }
            }
        };
#pragma unroll
        for (int k = 0; k < PD; ++k) load(raw[k]);
        /* the segment's live rows (lanes 0..15 one each), this lane's 8 output rows m = 8 h + i as a mask */
        const bool lv = lane < mc && mx_moe_tok(A.sorted, r0 + lane, A.top_k, A.a_sorted, A.AR) >= 0;
        const uint32_t bal = __builtin_amdgcn_ballot_w32(lv);
        const uint32_t lm = (bal >> (8 * h)) & 0xffu;
        if (bal == 0u) {   /* a segment of dead slots only: zeros (its loads land in registers nobody reads) */
            xm_zero(A, r0, mc, n0);
            continue;
        }
        lds_barrier();
        xw_stage<1>(A, xs, r0, mc, 1, 0, UK);
        lds_barrier();
        xw_v8f acc;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
        int tc = 0, bc = 0;
        const uint8_t* crow = rowp(0);
        const xw_v8f z = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
#pragma unroll 1
        for (int j0 = 0; j0 < NB; j0 += PD) {
#pragma unroll
            for (int k = 0; k < PD; ++k) {
                if (j0 + k < NB) {
                    const r4d_u32x4 w = raw[k];
                    uint32_t q0, q1;
                    float d;
                    if constexpr (IL != 0) {
                        q0 = w[0]; q1 = w[1];
                        d = mx_e8m0_half(w[2]);
                    } else {
                    const int sA = (int)(((uintptr_t)crow + 17u * (uintptr_t)bc) & 3u);
                    const int sh = (sA + 1) & 3;
                    const bool s3 = sA == 3;
                    q0 = __builtin_amdgcn_alignbyte(w[1], s3 ? w[1] : w[0], sh);
                    q1 = __builtin_amdgcn_alignbyte(w[3], s3 ? w[3] : w[2], sh);
                    const uint32_t e0 = (w[0] >> (8 * sA)) & 0xffu;
                    const uint32_t e1 = (uint32_t)__builtin_amdgcn_permlanex16((int)e0, (int)e0, 0x76543210, 0xfedcba98, false, false);
                    d = mx_e8m0_half(h ? e1 : e0);
                    }
                    load(raw[k]);
                    r4d_u32x4 bw0, bw1;
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        bw0[p] = tab[(q0 >> (8 * p)) & 0xffu];
                        bw1[p] = tab[(q1 >> (8 * p)) & 0xffu];
                    }
                    const uint8_t* xb = xs + bc * 1024;
                    const r4d_u32x4 a0 = *(const r4d_u32x4*)(xb + ((0 * 2 + h) * 16 + nl) * 16);
                    const r4d_u32x4 a1 = *(const r4d_u32x4*)(xb + ((1 * 2 + h) * 16 + nl) * 16);
                    xw_v8f P = xw_wmma(a0, bw0, z);
                    P = xw_wmma(a1, bw1, P);
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc[i] = __builtin_fmaf(d, P[i], acc[i]);
                    if (++bc == UK) {
                        const int n = n0 + (tc * WN + wave) * 16 + nl;
                        if (n < A.N) {
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                const int m = 8 * h + i;
                                if (m < mc) mx_put(A.y + (long long)(r0 + m) * A.y_ld + n, ((lm >> i) & 1u) ? acc[i] : 0.0f);
                            }
                        }
#pragma unroll
                        for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
                        bc = 0;
                        ++tc;
                        crow = rowp(tc);
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

template <int NT, int AB = 0>   /* AB (bench ablations, outputs differ): 1 no decode / WMMA / fma (a xor of the words), 4 no staging */
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
            if constexpr ((AB & 1) != 0) {
                acc[0][0] += __uint_as_float((q0 ^ q1 ^ ex) & 0x3fffffffu);
                return;
            }
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
        constexpr bool DB = (AB & 16) != 0;   /* the activations double-buffered: slab ss + 1 loaded while ss is computed */
        r4d_u32x4 xr[4 * NT];
        if constexpr (DB) {
            lds_barrier();
            xw_stage_ld<NT>(A, r0, mc, nt, 0, min(16, A.UK), xr);
            xw_stage_st<NT>(xs, nt, 0, min(16, A.UK), xr);
            lds_barrier();
        }
        auto step = [&](int ss, uint32_t (&cur)[kWqWords], uint32_t (&nxt)[kWqWords]) {
            load(ss + 1, nxt);
            const int b0 = 16 * ss, b1 = min(b0 + 16, A.UK);
            uint8_t* xcur = xs;
            if constexpr (DB) {
                xcur = xs + (ss & 1) * NT * 16384;
                if (ss + 1 < NSS) xw_stage_ld<NT>(A, r0, mc, nt, b0 + 16, min(b0 + 32, A.UK), xr);
            } else if constexpr ((AB & 4) == 0) {
                lds_barrier();
                xw_stage<NT>(A, xs, r0, mc, nt, b0, b1);
                lds_barrier();
            }
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
                blk(xcur + j * NT * 1024, h ? v0 : pA0, h ? v1 : pA1, h ? Ep : E);
                if (hi) blk(xcur + (j + 8) * NT * 1024, h ? pB0 : v0, h ? pB1 : v1, h ? E : Ep);
            });
            if constexpr (DB) {
                if (ss + 1 < NSS) xw_stage_st<NT>(xs + ((ss + 1) & 1) * NT * 16384, nt, b0 + 16, min(b0 + 32, A.UK), xr);
                lds_barrier();
            }
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

/* xm_wl_kernel (DRAFTER track 19.11.2026 round 10): xm_wm_kernel WR 1's OPERATIONS -- per output the blocks in K order, each
 * block's partial P by its two v_wmma_f32_16x16x16_bf16 from 0 on the same fragments (A = the staged activations in
 * xw_stage's interleave, B = the byte -> bf16-pair table's values of qs[4 h ..] / qs[8 + 4 h ..]), acc = fma(2^(e - 128), P,
 * acc), one rounding (mx_put), 0 for a dead row: the SAME BYTES as drafter_mx 2's form -- with the weights read the way
 * xm_a8_kernel reads them: a block of 4 waves = 64 weight rows of ONE expert (grid ceil(N / 64) x n_expert, the experts'
 * row ranges as mx_range gives them), stages of 8 MXFP4 blocks, two threads a row each loading 4 blocks (68
 * contiguous bytes: 17 dwords, or 18 aligned dwords realigned by v_alignbyte when rows are not dword-aligned (K % 128)),
 * PD stages ahead in registers while the current stage is computed from LDS (qs reordered for the lane halves, the scale as
 * f32); the activations a stage at a time the same way. Only LDS barriers (no global_inv). AL: every row dword-aligned. */
constexpr int kWlBN = 64, kWlSK = 8, kWlRS = kWlSK * 4 + 2;   /* rows a block, blocks a stage, dwords a staged row (+2: banks) */
template <int NT, int AL, int PD>
__global__ __launch_bounds__(128) void xm_wl_kernel(MxMrArgs A) {
    constexpr int MT = 16 * NT;
    constexpr int NA = 4 * NT;                                   /* activation units (16 B) a thread a stage */
    __shared__ uint32_t xw_tab[256];
    __shared__ __align__(16) uint32_t braw[kWlBN * kWlRS];
    __shared__ float bd[kWlBN * (kWlSK + 1)];
    __shared__ __align__(16) uint8_t xs[kWlSK * NT * 1024];
    __shared__ int src[MT];
    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5, nl = lane & 15, h = lane >> 4;
    const int col0 = (int)blockIdx.x * kWlBN, e = (int)blockIdx.y;
    int lo = 0, hi = 0;
    mx_range(A.off, e, A.n_expert, A.T, &lo, &hi);   /* xm_count's ranges: expert 0 from row 0, the last one to T */
    if (lo >= hi) return;
    for (int b = tid; b < 256; b += 128) xw_tab[b] = xw_bf(kMxValues[b & 15]) | (xw_bf(kMxValues[b >> 4]) << 16);
    const uint8_t* wb = A.wtab ? A.wtab[e] : A.w + (long long)e * A.w_es;
    const int lr = tid >> 1, lh = tid & 1;                       /* loader: row lr, blocks 4 lh .. 4 lh + 3 of a stage */
    const uint8_t* wrow = wb + (long long)min(col0 + lr, A.N - 1) * A.w_ld;
    const int NSTG = (A.UK + kWlSK - 1) / kWlSK;
    const int myrow = wave * 16 + nl, n = col0 + myrow;          /* compute: this lane's weight row */
    for (int r0 = lo; r0 < hi; r0 += MT) {
        const int mc = min(MT, hi - r0);
        lds_barrier();
        if (tid < MT) src[tid] = tid < mc ? mx_moe_tok(A.sorted, r0 + tid, A.top_k, A.a_sorted, A.AR) : -1;
        lds_barrier();
        int ml = 0;
        for (int m = 0; m < mc; ++m)
            if (src[m] >= 0) ml = m + 1;
        if (ml == 0) {
            for (int idx = tid; idx < mc * kWlBN; idx += 128) {
                const int c = col0 + (idx % kWlBN);
                if (c < A.N) mx_put(A.y + (long long)(r0 + idx / kWlBN) * A.y_ld + c, 0.0f);
            }
            continue;
        }
        const int nt = (ml + 15) >> 4;
        uint32_t wreg[PD][18];
        r4d_u32x4 areg[PD][NA];
        auto load = [&](int stg, uint32_t (&wr)[18], r4d_u32x4 (&ar)[NA]) {
            if (stg >= NSTG) return;
            const int b0 = stg * kWlSK + 4 * lh;
            if (b0 < A.UK) {
                const uint8_t* p = wrow + (long long)b0 * 17;
                const int nb = min(4, A.UK - b0);
                if constexpr (AL) {
                    const mx_gu32 q = (mx_gu32)(const void*)p;
                    const int ndw = (17 * nb + 3) >> 2;
#pragma unroll
                    for (int i = 0; i < 17; ++i)
                        if (i < ndw) wr[i] = q[i];
                } else {
                    const mx_gu32 q = (mx_gu32)(const void*)((uintptr_t)p & ~(uintptr_t)3);
                    const int ndw = (17 * nb + (int)((uintptr_t)p & 3u) + 3) >> 2;
#pragma unroll
                    for (int i = 0; i < 18; ++i)
                        if (i < ndw) wr[i] = q[i];
                }
            }
            const int bs = stg * kWlSK, np = (min(bs + kWlSK, A.UK) - bs) * nt * 64;
#pragma unroll
            for (int k = 0; k < NA; ++k) {
                const int idx = tid + 128 * k;
                r4d_u32x4 o = { 0u, 0u, 0u, 0u };
                if (idx < np) {
                    const int ni = idx & 15, jh = (idx >> 4) & 3, rest = idx >> 6;
                    const int t = rest % nt, bb = rest / nt;
                    const int m = 16 * t + ni;
                    const int tok = m < mc ? src[m] : -1;
                    if (tok >= 0) {
                        const int q = 8 * (jh >> 1) + 4 * (jh & 1);
                        const uint16_t* px = A.x + (long long)tok * A.x_ld + (long long)(bs + bb) * 32 + q;
                        const uint2 a = *(const uint2*)(const void*)px;
                        const uint2 c = *(const uint2*)(const void*)(px + 16);
                        o[0] = (a.x & 0xffffu) | (c.x << 16);
                        o[1] = (a.x >> 16) | (c.x & 0xffff0000u);
                        o[2] = (a.y & 0xffffu) | (c.y << 16);
                        o[3] = (a.y >> 16) | (c.y & 0xffff0000u);
                    }
                }
                ar[k] = o;
            }
        };
        auto store = [&](int stg, uint32_t (&wr)[18], const r4d_u32x4 (&ar)[NA]) {
            const int b0 = stg * kWlSK + 4 * lh;
            if (b0 < A.UK) {
                if constexpr (!AL) {
                    const int sh = (int)((uintptr_t)(wrow + (long long)b0 * 17) & 3u);
#pragma unroll
                    for (int i = 0; i < 17; ++i) wr[i] = __builtin_amdgcn_alignbyte(wr[i + 1], wr[i], sh);
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (b0 + i >= A.UK) break;
                    const int o = 17 * i;
                    const uint32_t ex = (wr[o / 4] >> (8 * (o % 4))) & 0xffu;
                    uint32_t q[4];
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const int bo = o + 1 + 4 * k, di = bo / 4, s = bo % 4;
                        q[k] = s ? __builtin_amdgcn_alignbyte(wr[di + 1], wr[di], s) : wr[di];
                    }
                    *(r4d_u32x4*)&braw[lr * kWlRS + (4 * lh + i) * 4] = r4d_u32x4{ q[0], q[2], q[1], q[3] };
                    bd[lr * (kWlSK + 1) + 4 * lh + i] = mx_e8m0_half(ex);
                }
            }
            const int bs = stg * kWlSK, np = (min(bs + kWlSK, A.UK) - bs) * nt * 64;
#pragma unroll
            for (int k = 0; k < NA; ++k) {
                const int idx = tid + 128 * k;
                if (idx < np) {
                    const int ni = idx & 15, jh = (idx >> 4) & 3, rest = idx >> 6;
                    const int t = rest % nt, bb = rest / nt;
                    *(r4d_u32x4*)(xs + ((((bb * NT + t) * 2 + (jh >> 1)) * 2 + (jh & 1)) * 16 + ni) * 16) = ar[k];
                }
            }
        };
        xw_v8f acc[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[t][i] = 0.0f;
        const xw_v8f z = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
        auto compute = [&](int stg) {
            const int nb = min(kWlSK, A.UK - stg * kWlSK);
#pragma unroll
            for (int bb = 0; bb < kWlSK; ++bb) {
                if (bb >= nb) break;
                const uint2 qq = *(const uint2*)&braw[myrow * kWlRS + bb * 4 + 2 * h];
                const float d = bd[myrow * (kWlSK + 1) + bb];
                r4d_u32x4 b0, b1;
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    b0[p] = xw_tab[(qq.x >> (8 * p)) & 0xffu];
                    b1[p] = xw_tab[(qq.y >> (8 * p)) & 0xffu];
                }
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    if (t >= nt) break;
                    const uint8_t* xb = xs + (bb * NT + t) * 1024;
                    const r4d_u32x4 a0 = *(const r4d_u32x4*)(xb + ((0 * 2 + h) * 16 + nl) * 16);
                    const r4d_u32x4 a1 = *(const r4d_u32x4*)(xb + ((1 * 2 + h) * 16 + nl) * 16);
                    xw_v8f P = xw_wmma(a0, b0, z);
                    P = xw_wmma(a1, b1, P);
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc[t][i] = __builtin_fmaf(d, P[i], acc[t][i]);
                }
            }
        };
#pragma unroll
        for (int k = 0; k < PD; ++k) load(k, wreg[k], areg[k]);
#pragma unroll 1
        for (int s0 = 0; s0 < NSTG; s0 += PD) {
#pragma unroll
            for (int k = 0; k < PD; ++k) {
                const int stg = s0 + k;
                if (stg < NSTG) {
                    store(stg, wreg[k], areg[k]);
                    lds_barrier();
                    load(stg + PD, wreg[k], areg[k]);
                    compute(stg);
                    lds_barrier();
                }
            }
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int m = 16 * t + 8 * h + i;
                if (m < mc && n < A.N) mx_put(A.y + (long long)(r0 + m) * A.y_ld + n, src[m] >= 0 ? acc[t][i] : 0.0f);
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
#define XM_WFORMS(X) X(1, 1, 2, 0) X(1, 1, 4, 0) X(1, 1, 6, 0) X(1, 1, 8, 0) X(1, 1, 9, 0) X(1, 1, 12, 0) X(1, 1, 14, 0) X(1, 2, 4, 0) X(1, 4, 2, 0) X(2, 1, 2, 0) X(2, 1, 4, 0) X(2, 1, 6, 0) X(2, 2, 2, 0) \
    X(2, 2, 4, 0) X(2, 4, 2, 0) X(4, 1, 2, 0) X(2, 1, 2, 1) X(2, 1, 2, 2) X(2, 1, 2, 4) X(2, 1, 2, 3) X(2, 1, 2, 7) X(2, 1, 2, 5)
/* lever WIL (bench): the IL forms -- WR 1 NT 1 at PD 2 / 9 (drafter_mx 4 / 5's gate_up) and the sequential-tile PD 9 TS 4 */
int xm_launch_wil(RadStream s, int PD, const MxMrArgs& A, unsigned grid, unsigned thr, unsigned lds) {
#define XM_WIL(pd) if (PD == pd) return R4D_EMIT(s, (xm_wm_kernel<1, 1, pd, 0, 1>), dim3(grid), dim3(thr), lds, A);
    XM_WIL(2) XM_WIL(6) XM_WIL(9) XM_WIL(12) XM_WIL(16)
#undef XM_WIL
    return R4D_MX_MR_DECLINE;
}
int xm_launch_til(RadStream s, int PD, int TS, const MxMrArgs& A, unsigned grid, unsigned thr, unsigned lds) {
#define XM_TIL(pd, ts) if (PD == pd && TS == ts) return R4D_EMIT(s, (xm_wt_kernel<pd, ts, 1>), dim3(grid), dim3(thr), lds, A);
    XM_TIL(9, 4) XM_TIL(12, 4) XM_TIL(16, 4) XM_TIL(9, 2) XM_TIL(9, 8) XM_TIL(6, 4)
#undef XM_TIL
    return R4D_MX_MR_DECLINE;
}
int xm_launch_w(RadStream s, int WR, int NT, int PD, int AB, const MxMrArgs& A, unsigned grid, unsigned thr, unsigned lds) {
#define XM_W(wr, nt, pd, ab) \
    if (WR == wr && NT == nt && PD == pd && AB == ab) return R4D_EMIT(s, (xm_wm_kernel<wr, nt, pd, ab>), dim3(grid), dim3(thr), lds, A);
    XM_WFORMS(XM_W)
#undef XM_W
    return R4D_MX_MR_DECLINE;
}
/* sequential-tile form (lever DTS): (PD, TS) */
#define XM_TFORMS(X) X(2, 1) X(2, 2) X(2, 4) X(2, 8) X(4, 2) X(4, 4) X(4, 8) X(6, 4) X(6, 8) X(9, 2) X(9, 4) X(9, 8) X(12, 2) X(12, 4) X(12, 8) X(14, 4) X(14, 8)
int xm_launch_t(RadStream s, int PD, int TS, const MxMrArgs& A, unsigned grid, unsigned thr, unsigned lds) {
#define XM_T(pd, ts) \
    if (PD == pd && TS == ts) return R4D_EMIT(s, (xm_wt_kernel<pd, ts>), dim3(grid), dim3(thr), lds, A);
    XM_TFORMS(XM_T)
#undef XM_T
    return R4D_MX_MR_DECLINE;
}
bool xm_has_t(int PD, int TS) {
#define XM_TH(pd, ts) if (PD == pd && TS == ts) return true;
    XM_TFORMS(XM_TH)
#undef XM_TH
    return false;
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
    if (NT == 101) return R4D_EMIT(s, (xm_wq_kernel<1, 1>), dim3(grid), dim3(256), lds, A);   /* bench: q101 = no decode / WMMA */
    if (NT == 105) return R4D_EMIT(s, (xm_wq_kernel<1, 5>), dim3(grid), dim3(256), lds, A);   /* bench: q105 = loads only */
    if (NT == 104) return R4D_EMIT(s, (xm_wq_kernel<1, 4>), dim3(grid), dim3(256), lds, A);   /* bench: q104 = no staging */
    if (NT == 116) return R4D_EMIT(s, (xm_wq_kernel<1, 16>), dim3(grid), dim3(256), lds, A);  /* q116: activations double-buffered */
    if (NT == 216) return R4D_EMIT(s, (xm_wq_kernel<2, 16>), dim3(grid), dim3(256), lds, A);
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

/* ------------------------------------------------------------------ drafter_mx 3: fp8 x fp8 (DRAFTER track 18.11.2026 round 3)
 * xm_a8_kernel = moe_gemm_mxfp4a8's expert-major form (r4d_moe_mxfp4.hip moe_mxfp4_expert_kernel<4, 256>, the one its body() takes at
 * these shapes) on the drafter's own 17-byte ggml MXFP4 blocks, the activations quantised inside the kernel as r4d_quant_act_fp8 does
 * them (E4M3, an f32 scale amax / 448 a (row, 128 of K), clamped to +-448, RNE): the same values and, output by output, the same
 * operations as that pair of libr4d launches on the canonical planes of the same weights (tools/dbench a8 arm: byte-identical), with
 * no scratch. Each E2M1 code as its exact E4M3 byte (r4d_mxfp4.h's row-0 magnitude table), a 32-value group's two fp8 WMMAs from 0,
 * then total += raw x act_scale x 2^(e - 127). Rule B (a row's quantisation and sums are its own: deterministic, batch- and
 * placement-invariant); NOT the bf16-activation forms' values (rms 2.5 % relative on the drafter's experts, dbench b36): proposals
 * only, verify keeps the text. A block = 64 weight rows (4 waves x 16) x the expert's rows in 16-row tiles, stages of 256 K: the
 * weights two threads a row, 17 aligned dwords (4 blocks) each, decoded at the LDS store; the next stage in registers meanwhile. */
}  // namespace
#include "r4d_fp8_wmma.h"
#include "r4d_mxfp4.h"
namespace {
constexpr int kA8SK = 256, kA8P = kA8SK + 8, kA8BN = 64, kA8KB = kA8SK / 128;
__global__ __launch_bounds__(128) void xm_a8_kernel(MxMrArgs A) {
    __shared__ __align__(16) uint8_t aa[16 * kA8P], bb[kA8BN * kA8P];
    __shared__ uint32_t ss[kA8BN * kA8KB];
    __shared__ float sa[16 * kA8KB];
    __shared__ int src[16];
    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5, col0 = (int)blockIdx.x * kA8BN, e = (int)blockIdx.y;
    if (e == A.n_expert) {   /* the dropped slots past the last expert: zero rows */
        const int r0 = min(max(A.off[A.n_expert], 0), A.T);
        for (long long i = tid; i < (long long)(A.T - r0) * kA8BN; i += 128) {
            const int n = col0 + (int)(i % kA8BN);
            if (n < A.N) A.y[(long long)(r0 + i / kA8BN) * A.y_ld + n] = 0;
        }
        return;
    }
    const int lo = max(A.off[e], 0), hi = min(A.off[e + 1], A.T);
    if (lo >= hi) return;
    const uint8_t* wb = A.wtab ? A.wtab[e] : A.w + (long long)e * A.w_es;
    const int rowbase = (lane >> 4) * 8, mycol = wave * 16 + (lane & 15);
    const int lr = tid >> 1, lh = tid & 1;                   /* weight loader: row lr, blocks 4 lh .. 4 lh + 3 of a stage */
    const uint32_t* wsrc = (const uint32_t*)(wb + (long long)min(col0 + lr, A.N - 1) * A.w_ld) + lh * 17;
    const int ar = tid >> 3, ac = tid & 7;                   /* activation loader: tile row ar, values 32 ac .. 32 ac + 31 */
    for (int r0 = lo; r0 < hi; r0 += 16) {
        __syncthreads();
        if (tid < 16) {
            const int r = r0 + tid;
            src[tid] = r < hi ? mx_moe_tok(A.sorted, r, A.top_k, A.a_sorted, A.AR) : -1;
        }
        __syncthreads();
        bool rv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) rv[j] = src[rowbase + j] >= 0;
        const int arow = src[ar];
        uint32_t wreg[17];
        uint4 xreg[4];
        auto load = [&](int k0) {
            const uint32_t* p = wsrc + (k0 / 256) * 34;
#pragma unroll
            for (int i = 0; i < 17; ++i) wreg[i] = p[i];
#pragma unroll
            for (int i = 0; i < 4; ++i) xreg[i] = uint4{ 0u, 0u, 0u, 0u };
            if (arow >= 0) {
                const uint4* xp = (const uint4*)(const void*)(A.x + (long long)arow * A.x_ld + k0 + 32 * ac);
#pragma unroll
                for (int i = 0; i < 4; ++i) xreg[i] = xp[i];
            }
        };
        auto store = [&]() {
            uint32_t ex = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {                    /* block 4 lh + i at byte 17 i of my 68 */
                const int o = 17 * i;
                ex |= ((wreg[o / 4] >> (8 * (o % 4))) & 0xffu) << (8 * i);
#pragma unroll
                for (int q = 0; q < 4; ++q) {                /* qs[4 q .. 4 q + 3]: elements 4 q .. (low nibbles), 16 + 4 q .. (high) */
                    const int bo = o + 1 + 4 * q, di = bo / 4, sh = bo % 4;
                    const uint32_t w = sh ? __builtin_amdgcn_alignbyte(wreg[di + 1], wreg[di], sh) : wreg[di];
                    const uint32_t ev = w & 0x0F0F0F0Fu, od = (w >> 4) & 0x0F0F0F0Fu;
                    const uint32_t be = __builtin_amdgcn_perm(R4D_MXFP4_MAG0_HI, R4D_MXFP4_MAG0_LO, ev & 0x07070707u) | ((ev & 0x08080808u) << 4);
                    const uint32_t bo8 = __builtin_amdgcn_perm(R4D_MXFP4_MAG0_HI, R4D_MXFP4_MAG0_LO, od & 0x07070707u) | ((od & 0x08080808u) << 4);
                    const int kk = 32 * (4 * lh + i) + 4 * q;
                    *(uint32_t*)(bb + lr * kA8P + kk) = be;
                    *(uint32_t*)(bb + lr * kA8P + kk + 16) = bo8;
                }
            }
            ss[lr * kA8KB + lh] = ex;
            /* the activations: r4d_quant_act_fp8's arithmetic on my 32 values of a 128-group (4 threads a group) */
            float v[32];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t u[4] = { xreg[i].x, xreg[i].y, xreg[i].z, xreg[i].w };
#pragma unroll
                for (int t = 0; t < 4; ++t) { v[8 * i + 2 * t] = __uint_as_float(u[t] << 16); v[8 * i + 2 * t + 1] = __uint_as_float(u[t] & 0xffff0000u); }
            }
            float m = 0.0f;
#pragma unroll
            for (int j = 0; j < 32; ++j) m = fmaxf(m, fabsf(v[j]));
            m = fmaxf(m, __shfl_xor(m, 1));
            m = fmaxf(m, __shfl_xor(m, 2));
            const float sc = m > 0.0f ? m * (1.0f / 448.0f) : 1.0f;
            const float inv = 1.0f / sc;
            uint32_t pk[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                int p = 0;
                p = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(v[4 * j] * inv, -448.0f), 448.0f),
                                                    fminf(fmaxf(v[4 * j + 1] * inv, -448.0f), 448.0f), p, false);
                p = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(v[4 * j + 2] * inv, -448.0f), 448.0f),
                                                    fminf(fmaxf(v[4 * j + 3] * inv, -448.0f), 448.0f), p, true);
                pk[j] = (uint32_t)p;
            }
            uint4* dst = (uint4*)(aa + ar * kA8P + 32 * ac);
            dst[0] = uint4{ pk[0], pk[1], pk[2], pk[3] };
            dst[1] = uint4{ pk[4], pk[5], pk[6], pk[7] };
            if ((ac & 3) == 0) sa[ar * kA8KB + (ac >> 2)] = arow >= 0 ? sc : 0.0f;
        };
        auto total = r4d_fp8::acc_zero();
        load(0);
        for (int k0 = 0; k0 < A.K; k0 += kA8SK) {
            store();
            __syncthreads();
            if (k0 + kA8SK < A.K) load(k0 + kA8SK);
#pragma unroll
            for (int kb = 0; kb < kA8KB; ++kb) {
                const uint32_t exps = ss[mycol * kA8KB + kb];
#pragma unroll
                for (int group = 0; group < 4; ++group) {
                    auto raw = r4d_fp8::acc_zero();
#pragma unroll
                    for (int k = kb * 128 + group * 32; k < kb * 128 + (group + 1) * 32; k += 16)
                        raw = r4d_fp8::mma(r4d_fp8::frag<kA8SK>(aa, 0, k, lane), r4d_fp8::frag<kA8SK>(bb, wave * 16, k, lane), raw);
                    const uint32_t exp = (exps >> (8 * group)) & 0xffu;
                    const float scale = __uint_as_float(exp == 255 ? 0x7fc00000u : exp ? (exp << 23) : 0x00400000u);
#pragma unroll
                    for (int row = 0; row < 8; ++row)
                        if (rv[row]) total[row] += raw[row] * sa[(rowbase + row) * kA8KB + kb] * scale;
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int row = 0; row < 8; ++row) {
            const int r = r0 + rowbase + row, n = col0 + mycol;
            if (r < hi && n < A.N) A.y[(long long)r * A.y_ld + n] = f32_to_bf16(total[row]);
        }
    }
}
/* ------------------------------------------------------------------ the switch, the pin, the rule */
int env_drafter_mx() {
    const char* v = std::getenv("RADIANCE_DS41_DRAFTER_MX");
    const int on = (v && (*v == '1' || *v == '2' || *v == '3' || *v == '4' || *v == '5') && !v[1]) ? *v - '0' : 0;
    if (v && *v && !on && std::strcmp(v, "0"))
        std::fprintf(stderr, "radiance: libr4d: RADIANCE_DS41_DRAFTER_MX=%s is not 0 / 1 / 2 / 3 / 4 / 5: the stock MXFP4 grouped GEMM\n", v);
    if (on == 5)
        std::fprintf(stderr, "radiance: libr4d: moe_gemm_ggml type=mxfp4 takes the WMMA form at every call, 9 blocks in flight a lane at "
                             "K >= 2048 and 4 row tiles a wave in sequence below (RADIANCE_DS41_DRAFTER_MX=5, the fn4 switch drafter_mx 5: "
                             "drafter_mx 2's bytes)\n");
    if (on == 4)
        std::fprintf(stderr, "radiance: libr4d: moe_gemm_ggml type=mxfp4 takes the WMMA form at every call with 9 blocks in flight a lane at "
                             "K >= 2048 (RADIANCE_DS41_DRAFTER_MX=4, the fn4 switch drafter_mx 4: drafter_mx 2's bytes)\n");
    if (on == 3)
        std::fprintf(stderr, "radiance: libr4d: moe_gemm_ggml type=mxfp4 takes the fp8 x fp8 form at every call (RADIANCE_DS41_DRAFTER_MX=3, "
                             "the fn4 switch drafter_mx 3: activations quantised to E4M3 a (row, 128), rule B, the drafter's own reference)\n");
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
int g_xm_il = 0;                          /* lever WIL (bench hook r4d_mx_mr_il): the weights are in the IL layout */

}  // namespace

int r4d_mx_mr_on() { return g_xm_pin != 0 ? 1 : g_xm_on; }
void r4d_mx_mr_set(int on) { g_xm_on = on == 2 || on == 3 || on == 4 || on == 5 ? on : (on ? 1 : 0); }
int r4d_mx_mr_get() { return g_xm_on; }

/* The WMMA form's launch (rule B): WR 16-row weight tiles a wave, segments of up to 16 NT rows, WN waves a block. */
static int xm_wm_moe(RadStream s, const R4dMxMrCall& c, int WR, int NT, int PD, int AB, int WN, int gridcap, int il = g_xm_il) {
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
    if (il) {   /* lever WIL: 16-row groups, so N % 16 == 0; WR 1 NT 1 only */
        if (WR != 1 || NT != 1 || AB != 0 || c.N % 16) return R4D_MX_MR_DECLINE;
        std::snprintf(g_xm_last, sizeof g_xm_last, "wmma-IL WR1 NT1 PD%d WN%d SS%d RB%d grid %lld lds %lld", PD, WN, A.SS, A.RB, grid, lds);
        return xm_launch_wil(s, PD, A, (unsigned)grid, 32u * WN, (unsigned)lds);
    }
    return xm_launch_w(s, WR, NT, PD, AB, A, (unsigned)grid, 32u * WN, (unsigned)lds);
}

/* The sequential-tile form's launch (lever DTS; drafter_mx 2's bytes): WN waves, TS 16-row tiles a wave in sequence, segments
 * of up to 16 rows, the whole K's activations in one slab (declines K above ~60 blocks). */
static int xm_wt_moe(RadStream s, const R4dMxMrCall& c, int PD, int TS, int WN, int gridcap, int il = g_xm_il) {
    if (!xm_has_t(PD, TS) || (WN != 1 && WN != 2 && WN != 4 && WN != 8)) return R4D_MX_MR_DECLINE;
    MxMrArgs A{};
    A.x = c.x; A.x_ld = c.x_ld; A.w = c.w; A.wtab = c.wtab; A.w_ld = c.w_ld; A.w_es = c.w_es;
    A.y = c.y; A.y_ld = c.y_ld; A.sorted = c.sorted; A.off = c.off;
    A.T = c.T; A.N = c.N; A.K = c.K; A.n_expert = c.n_expert; A.top_k = c.top_k; A.a_sorted = c.a_sorted;
    A.AR = c.AR; A.x_al = c.x_al;
    A.UK = c.K / 32;
    A.NS = A.UK;
    A.SS = A.UK;
    A.RB = WN * 16 * TS;
    A.nrg = (c.N + A.RB - 1) / A.RB;
    A.live_cap = c.n_expert < c.T ? c.n_expert : c.T;
    A.live_off = kXwTab;
    A.x_off = (int)(kXwTab + (4LL * A.live_cap * 4 + 15) / 16 * 16);
    const long long lds = A.x_off + (long long)A.UK * 1024;
    if (lds > 65536) return R4D_MX_MR_DECLINE;
    const long long est = (long long)A.live_cap * A.nrg;
    long long grid = gridcap > 0 ? gridcap : 32LL * xm_cu_count();
    if (grid > est) grid = est;
    if (grid < 1) grid = 1;
    std::snprintf(g_xm_last, sizeof g_xm_last, "wmma-seq PD%d TS%d WN%d RB%d grid %lld lds %lld", PD, TS, WN, A.RB, grid, lds);
    if (il) {
        if (c.N % 16) return R4D_MX_MR_DECLINE;
        std::snprintf(g_xm_last, sizeof g_xm_last, "wmma-seq-IL PD%d TS%d WN%d RB%d grid %lld lds %lld", PD, TS, WN, A.RB, grid, lds);
        return xm_launch_til(s, PD, TS, A, (unsigned)grid, 32u * WN, (unsigned)lds);
    }
    return xm_launch_t(s, PD, TS, A, (unsigned)grid, 32u * WN, (unsigned)lds);
}

/* The long-run WMMA form's launch (rule B): 8 waves of one 16-row tile, segments of up to 16 NT rows, super-steps of 16 blocks. */
static int xm_wq_moe(RadStream s, const R4dMxMrCall& c, int NT, int gridcap) {
    const int NTL = NT > 200 ? 2 : (NT > 100 ? 1 : NT);   /* 101 / 104 / 105: bench ablations of NT 1; 116 / 216: double-buffered */
    const int DBF = NT == 116 || NT == 216 ? 2 : 1;
    if ((NTL != 1 && NTL != 2) || (c.K / 32) % 8 != 0) return R4D_MX_MR_DECLINE;
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
    const long long lds = A.x_off + 16LL * NTL * 1024 * DBF;
    if (lds > 65536) return R4D_MX_MR_DECLINE;
    const long long est = (long long)A.live_cap * A.nrg;
    long long grid = gridcap > 0 ? gridcap : 32LL * xm_cu_count();
    if (grid > est) grid = est;
    if (grid < 1) grid = 1;
    std::snprintf(g_xm_last, sizeof g_xm_last, "wmma-run NT%d WN8 SS16 RB128 grid %lld lds %lld", NT, grid, lds);
    return xm_launch_q(s, NT, A, (unsigned)grid, (unsigned)lds);
}

/* drafter_mx 3's launch (xm_a8_kernel above): 17-byte blocks whose rows are dword-aligned, K a multiple of 256, aligned activations. */
static int xm_a8_moe(RadStream s, const R4dMxMrCall& c) {
    if (c.K % kA8SK || !c.x_al || c.x_ld % 8 || c.N < 1 || c.n_expert < 1 || c.n_expert > 65534) return R4D_MX_MR_DECLINE;
    const long long rb = (long long)(c.K / 32) * 17;
    if (rb % 4 || (c.w_ld != 0 && c.w_ld % 4)) return R4D_MX_MR_DECLINE;
    if (!c.wtab && (((uintptr_t)c.w & 3u) || c.w_es % 4)) return R4D_MX_MR_DECLINE;
    MxMrArgs A{};
    A.x = c.x; A.x_ld = c.x_ld; A.w = c.w; A.wtab = c.wtab; A.w_ld = c.w_ld; A.w_es = c.w_es;
    A.y = c.y; A.y_ld = c.y_ld; A.sorted = c.sorted; A.off = c.off;
    A.T = c.T; A.N = c.N; A.K = c.K; A.n_expert = c.n_expert; A.top_k = c.top_k; A.a_sorted = c.a_sorted;
    A.AR = c.AR; A.x_al = c.x_al;
    std::snprintf(g_xm_last, sizeof g_xm_last, "a8 fp8 x fp8 BN64 SK256 grid %d x %d", (c.N + kA8BN - 1) / kA8BN, c.n_expert + 1);
    return R4D_EMIT(s, xm_a8_kernel, dim3((unsigned)((c.N + kA8BN - 1) / kA8BN), (unsigned)(c.n_expert + 1)), dim3(128), 0, A);
}

/* xm_wl_kernel's launch (the same bytes as xm_wm_moe's WR 1 forms): 16-byte-aligned activation rows, NT 1 / 2, PD 1 / 2; rows
 * dword-aligned -> AL 1 (17-dword loads), else the realigning loader (lever DEP's down: K 576 = 18 blocks, 306-byte rows). */
static int xm_wl_moe(RadStream s, const R4dMxMrCall& c, int NT, int PD) {
    if ((NT != 1 && NT != 2) || (PD != 1 && PD != 2) || !c.x_al || c.N < 1 || c.n_expert < 1 || c.n_expert > 65534)
        return R4D_MX_MR_DECLINE;
    const long long rb = (long long)(c.K / 32) * 17;
    const int al = rb % 4 == 0 && c.w_ld % 4 == 0 && (c.wtab || ((((uintptr_t)c.w & 3u) == 0) && c.w_es % 4 == 0)) ? 1 : 0;
    MxMrArgs A{};
    A.x = c.x; A.x_ld = c.x_ld; A.w = c.w; A.wtab = c.wtab; A.w_ld = c.w_ld; A.w_es = c.w_es;
    A.y = c.y; A.y_ld = c.y_ld; A.sorted = c.sorted; A.off = c.off;
    A.T = c.T; A.N = c.N; A.K = c.K; A.n_expert = c.n_expert; A.top_k = c.top_k; A.a_sorted = c.a_sorted;
    A.AR = c.AR; A.x_al = c.x_al;
    A.UK = c.K / 32;
    const dim3 grid((unsigned)((c.N + kWlBN - 1) / kWlBN), (unsigned)c.n_expert);
    std::snprintf(g_xm_last, sizeof g_xm_last, "wl NT%d PD%d AL%d BN64 SK8 grid %u x %u", NT, PD, al, grid.x, grid.y);
#define XM_WL(nt, a, pd) \
    if (NT == nt && al == a && PD == pd) return R4D_EMIT(s, (xm_wl_kernel<nt, a, pd>), grid, dim3(128), 0, A);
    XM_WL(1, 1, 1) XM_WL(1, 0, 1) XM_WL(1, 1, 2) XM_WL(1, 0, 2) XM_WL(2, 1, 1) XM_WL(2, 0, 1) XM_WL(2, 1, 2) XM_WL(2, 0, 2)
#undef XM_WL
    return R4D_MX_MR_DECLINE;
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
    if (pinned && g_xm_pin_h >= 16384 && g_xm_pin_h < 16400)   /* xm_wl pin: h = 16384 + PD, mt = NT */
        return xm_wl_moe(s, c, g_xm_pin_mt, g_xm_pin_h - 16384);
    /* drafter_mx 2 (rule B): every call, either band, in the WMMA form (WR 1, NT 1, PD 2, 8 waves), so a sequence's drafter
     * bits do not depend on how many sequences share the call; what it declines takes the exact forms below. */
    if (!pinned && g_xm_pin == 0 && g_xm_on == 3) {   /* drafter_mx 3: fp8 x fp8 at every call (rule B); a decline takes mx 2 */
        const int a8 = xm_a8_moe(s, c);
        if (a8 != R4D_MX_MR_DECLINE) return a8;
    }
    if (pinned && g_xm_pin_h == 8192) return xm_a8_moe(s, c);   /* bench pin of the fp8 form */
    if (!pinned && g_xm_pin == 0 && g_xm_on >= 2) {
        /* drafter_mx 4 (DRAFTER track 19.11. round 10): the same form and bytes as 2 (PD only moves when a block's words are
         * loaded), 9 blocks in flight a lane at K >= 2048 (gate_up K 5120; EP's down K 2304). tools/dbench --quad on lq125's
         * 40-token records, xeon, in-process: lever DEP's shapes gate_up 165-177 vs 193-210 us a call (-13..-15 %, pass -10 %),
         * its down (K 576) keeps PD 2 (PD 4-14 within +-5 % or slower); EP shapes: down sum -9..-11 %, gate_up -2..-6 %. */
        /* drafter_mx 5 (round 12, lever DTS): 4's forms, and below K 2048 (lever DEP's down, K 576 = 18 blocks) xm_wt_kernel PD 9
         * TS 4 WN 8: an item = (expert, segment, 512 rows), the activations staged once an item, the loads 9 blocks ahead across
         * the 4 tiles -- tools/dbench --quad on lq168's records, xeon GPU1, steady state (300 passes): DEP down 422-450 -> 329-343
         * us a pass (-22..-25 %), pass 1,065-1,129 -> 982-1,031 (-7.8..-9.2 %) at 32 and 40 tokens; bytes IDENTICAL to mx 2. */
        if (g_xm_on == 5 && c.K < 2048) {
            const int t = xm_wt_moe(s, c, 9, 4, 8, 0);
            if (t != R4D_MX_MR_DECLINE) return t;
        }
        const int wm = xm_wm_moe(s, c, 1, 1, (g_xm_on == 4 || g_xm_on == 5) && c.K >= 2048 ? 9 : 2, 0, 8, 0);
        if (wm != R4D_MX_MR_DECLINE) return wm;
    }
    if (pinned && g_xm_pin_h >= 2048 && g_xm_pin_h < 4096) {   /* sequential-tile pin (lever DTS): h = 2048 + PD + 16 TS */
        const int t = xm_wt_moe(s, c, g_xm_pin_h & 15, (g_xm_pin_h >> 4) & 15, g_xm_pin_wn > 0 ? g_xm_pin_wn : 8, g_xm_pin_grid);
        if (t != R4D_MX_MR_DECLINE) return t;
        return xm_wm_moe(s, c, 1, 1, c.K >= 2048 ? 9 : 2, 0, 8, 0);   /* long K (gate_up): drafter_mx 4's form */
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

/* Lever WIL's serving entry (moe_gemm_mxil, r4d_ds41_mxfp4.inc): the weights are in the IL layout its row's relayout
 * writes, so only the IL forms may read them -- drafter_mx 5's forms (and bytes): K below 2048 (lever DEP's down, K 576)
 * the sequential-tile form PD 9 TS 4, else the WMMA form WR 1 NT 1 PD 9, 8 waves. A decline is the caller's error (no
 * other kernel reads this layout). Independent of drafter_mx and of the bench pins. */
int r4d_mx_mr_moe_il(RadStream s, const R4dMxMrCall& c) {
    if (c.tokens < 1 || c.tokens > kXmMaxTokens || c.T < 1 || c.N < 16 || c.N % 16 || c.K < 32 || c.K % 32 || c.n_expert < 1)
        return R4D_MX_MR_DECLINE;
    if (c.K < 2048) {
        const int t = xm_wt_moe(s, c, 9, 4, 8, 0, 1);
        if (t != R4D_MX_MR_DECLINE) return t;
    }
    return xm_wm_moe(s, c, 1, 1, 9, 0, 8, 0, 1);
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
/* lever WIL (bench hook): 1 = the calls' weights are in the IL layout (xm_wm_kernel's IL note): drafter_mx 4 / 5 take the IL
 * kernels (WR 1 NT 1 / the sequential-tile PD 9 TS 4), any other form declines. 0 = the 17-byte rows (today). */
extern "C" void r4d_mx_mr_il(int on) { g_xm_il = on ? 1 : 0; }

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
