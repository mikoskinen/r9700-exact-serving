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
constexpr int kXmMaxTokens = 48;         /* the forms serve the decode band's row counts */

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
template <int MT>
__device__ __forceinline__ void xm_stage_s(const MxMrArgs& A, float* xs, int r0, int mc, int b0, int b1) {
    const int np = (b1 - b0) * MT * 4;
    for (int idx = (int)threadIdx.x; idx < np; idx += (int)blockDim.x) {
        const int q = idx & 3, rest = idx >> 2;
        const int m = rest % MT, bb = rest / MT;
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

template <int T, int R, int MT>
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
            float w[R][32];
#pragma unroll
            for (int r = 0; r < R; ++r)
                mx_for<4>([&](auto gi) {
                    constexpr int G = decltype(gi)::value;
                    Q::template dec<G>(rw[r], tab, &w[r][8 * G]);
                });
            load(b + PD, rw);
#pragma unroll
            for (int m = 0; m < MT; ++m) {
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
            xm_stage_s<MT>(A, xs, r0, mc, b0, b1);
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

/* ------------------------------------------------------------------ forms */
/* gemv band: (H, R, MT); tiled order: (R, MT). */
#define XM_GFORMS(X) X(4, 1, 2) X(4, 1, 4) X(4, 1, 8) X(4, 1, 16) X(4, 2, 2) X(4, 2, 4) X(4, 2, 8) X(2, 1, 4) X(2, 1, 8)
#define XM_SFORMS(X) X(1, 2) X(1, 4) X(1, 8) X(2, 2) X(2, 4) X(2, 8)

int xm_launch_g(RadStream s, int H, int R, int MT, const MxMrArgs& A, unsigned grid, unsigned thr, unsigned lds) {
#define XM_G(h, r, mt) \
    if (H == h && R == r && MT == mt) return R4D_EMIT(s, (xm_gemv_kernel<MX_FP4, h, r, mt>), dim3(grid), dim3(thr), lds, A);
    XM_GFORMS(XM_G)
#undef XM_G
    return R4D_MX_MR_DECLINE;
}
int xm_launch_s(RadStream s, int R, int MT, const MxMrArgs& A, unsigned grid, unsigned thr, unsigned lds) {
#define XM_S(r, mt) \
    if (R == r && MT == mt) return R4D_EMIT(s, (xm_seq_kernel<MX_FP4, r, mt>), dim3(grid), dim3(thr), lds, A);
    XM_SFORMS(XM_S)
#undef XM_S
    return R4D_MX_MR_DECLINE;
}
bool xm_has_g(int H, int R, int MT) {
#define XM_GH(h, r, mt) if (H == h && R == r && MT == mt) return true;
    XM_GFORMS(XM_GH)
#undef XM_GH
    return false;
}
bool xm_has_s(int R, int MT) {
#define XM_SH(r, mt) if (R == r && MT == mt) return true;
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
    const int on = (v && *v == '1' && !v[1]) ? 1 : 0;
    if (v && *v && !on && std::strcmp(v, "0"))
        std::fprintf(stderr, "radiance: libr4d: RADIANCE_DS41_DRAFTER_MX=%s is not 0 / 1: the stock MXFP4 grouped GEMM\n", v);
    if (on)
        std::fprintf(stderr, "radiance: libr4d: moe_gemm_ggml type=mxfp4 takes the multi-row forms (RADIANCE_DS41_DRAFTER_MX=1, "
                             "the fn4 switch drafter_mx)\n");
    return on;
}
int g_xm_on = env_drafter_mx();
int g_xm_pin = 0;                         /* 0 none, -1 the rule forced, 1 a pinned form */
int g_xm_pin_h = 4, g_xm_pin_r = 1, g_xm_pin_mt = 4, g_xm_pin_wn = 0, g_xm_pin_grid = 0;
char g_xm_last[160];

}  // namespace

int r4d_mx_mr_on() { return g_xm_pin != 0 ? 1 : g_xm_on; }
void r4d_mx_mr_set(int on) { g_xm_on = on ? 1 : 0; }
int r4d_mx_mr_get() { return g_xm_on; }

/* The rule: MT from the call's tokens (an expert's rows in a decode step are few: 4 to 8 tokens' worth), the gemv
 * band's form H 4 x R 1 in 8-wave blocks (32 weight rows a block), the tiled order's R 1 in 4-wave blocks (128
 * rows), slabs of 16 KiB; a grid of two blocks a CU or fewer when the live work is smaller. */
int r4d_mx_mr_moe(RadStream s, const R4dMxMrCall& c) {
    if (c.tokens < 1 || c.tokens > kXmMaxTokens || c.T < 1 || c.N < 1 || c.K < 32 || c.K % 32 || c.n_expert < 1)
        return R4D_MX_MR_DECLINE;
    const bool pinned = g_xm_pin > 0;
    int H = 4, R = 1, MT = c.tokens <= 2 ? 2 : (c.tokens <= 8 ? 4 : 8), WN = 0;
    if (pinned) { H = g_xm_pin_h; R = g_xm_pin_r; MT = g_xm_pin_mt; WN = g_xm_pin_wn; }
    if (c.gemv ? !xm_has_g(H, R, MT) : !xm_has_s(R, MT)) return R4D_MX_MR_DECLINE;
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
    long long grid = g_xm_pin_grid > 0 && pinned ? g_xm_pin_grid : 4LL * xm_cu_count();
    if (grid > est) grid = est;
    if (grid < 1) grid = 1;
    std::snprintf(g_xm_last, sizeof g_xm_last, "%s H%d R%d MT%d WN%d SS%d RB%d grid %lld lds %lld", c.gemv ? "gemv" : "seq",
                  c.gemv ? H : 32, R, MT, WN, A.SS, A.RB, grid, lds);
    return c.gemv ? xm_launch_g(s, H, R, MT, A, (unsigned)grid, 32u * WN, (unsigned)lds)
                  : xm_launch_s(s, R, MT, A, (unsigned)grid, 32u * WN, (unsigned)lds);
}

/* r4d_mx_mr_pin (bench / test hook, r4d_selftest --mxcheck / --perfmx): on 0 clears the pin (the switch decides),
 * -1 forces the rule, 1 pins (H, R, MT, WN; the tiled order reads R and MT) and a grid cap (0: the rule's). */
extern "C" void r4d_mx_mr_pin(int on, int h, int r, int mt, int wn, int grid) {
    g_xm_pin = on == 1 || on == -1 ? on : 0;
    if (on == 1) { g_xm_pin_h = h; g_xm_pin_r = r; g_xm_pin_mt = mt; g_xm_pin_wn = wn; }
    g_xm_pin_grid = grid > 0 ? grid : 0;
}
extern "C" const char* r4d_mx_mr_last() { return g_xm_last; }

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
