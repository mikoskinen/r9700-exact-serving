/* Canonical E4M3 weights with exact F32 block scales: native FP8 WMMA and F32 scale folding, no
 * checkpoint scale narrowing. The legacy dense form computes one 16x64 output tile a block; the
 * dense decode and prefill forms below it produce the same bytes faster, and the grouped form runs
 * expert-major over the same per-element sequence. */
#include <hip/hip_runtime.h>
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include "r4d_fp8_wmma.h"
#include "r4d_launch.h"
#include <atomic>
#include <climits>
#include <cstdlib>
#include <cstdint>
#include <cstring>

namespace {
struct Meta {
    long ar, asr, yr, wr, sr, wstep[2], sstep[2];
    int T, N, K, experts, topk, sorted_a;
};
__device__ const uint8_t* entry(const void* p, long step, int e) {
    return step ? (const uint8_t*)p + step * e : ((const uint8_t* const*)p)[e];
}
/* The legacy dense form: a block per 16 rows by 64 columns, the operands staged through LDS a
 * dword at a time, so it serves any row pitch and alignment. */
__global__ void fp8_f32scale(const uint8_t* a, const float* as, const uint8_t* w, const float* ws,
                             uint16_t* y, Meta m) {
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, n0 = blockIdx.x * 64;
    __shared__ uint8_t qa[16 * 136], wb[64 * 136];
    __shared__ int source[16];
    const int first = (int)blockIdx.y * 16, end = min(first + 16, m.T);
    if (tid < 16) {
        const int r = first + tid;
        source[tid] = r < end ? r : -1;
    }
    __syncthreads();
    auto acc = r4d_fp8::acc_zero();
    for (int k0 = 0; k0 < m.K; k0 += 128) {
        for (int j = tid; j < 16 * 32; j += 128) {
            const int r = j / 32, c = (j % 32) * 4, src = source[r];
            uint32_t v = 0;
            if (src >= 0) {
                const auto* p = a + (long)src * m.ar + k0 + c;
                v = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
                    ((uint32_t)p[3] << 24);
            }
            *(uint32_t*)(qa + r * 136 + c) = v;
        }
        for (int j = tid; j < 64 * 32; j += 128) {
            const int r = j / 32, c = (j % 32) * 4;
            const auto* p = w + (long)(n0 + r) * m.wr + k0 + c;
            *(uint32_t*)(wb + r * 136 + c) = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                                             ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
        }
        __syncthreads();
        auto raw = r4d_fp8::acc_zero();
#pragma unroll
        for (int k = 0; k < 128; k += 16)
            raw = r4d_fp8::mma(r4d_fp8::frag<128>(qa, 0, k, lane),
                               r4d_fp8::frag<128>(wb, wave * 16, k, lane), raw);
        const float weight_scale = ws[(long)(n0 / 128) * m.sr + k0 / 128];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int src = source[(lane >> 4) * 8 + j];
            const float scale = src >= 0 ? as[(long)src * m.asr + k0 / 128] * weight_scale : 0;
            acc[j] += raw[j] * scale;
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int r = first + (lane >> 4) * 8 + j, c = n0 + wave * 16 + (lane & 15);
        if (r < end) y[(long)r * m.yr + c] = f32_to_bf16(acc[j]);
    }
}
/* The grouped form expert-major: a block per (column span, expert) takes every row of its
 * expert in 16-row tiles, so an expert slice is read from memory once however its rows fall in
 * the sorted order (16-row tiles of the sorted order read it once per tile its run spans). Each
 * element keeps the sequence above: per 128 of K in order a fresh chain of eight WMMAs, folded as
 * raw * (activation scale * weight scale). A block walks ColTiles column tiles of 64, each over K
 * in stages of StageK, as one stage sequence with the next stage loaded into registers while the
 * current one computes from LDS. Block row `experts` zeroes the dropped rows past
 * offsets[experts]. */
typedef unsigned f8_u4 __attribute__((ext_vector_type(4)));
template <int StageK>
struct F8Stage {
    static constexpr int KB = StageK / 128;
    f8_u4 a[StageK / 128], w[StageK / 32];
    float wsc[KB], sa;
};
template <int StageK, int ColTiles>
__global__ __launch_bounds__(128) void fp8_f32scale_experts(const uint8_t* a, const float* as,
                                                            const void* w0, const void* s0,
                                                            const void* w1, const void* s1,
                                                            const int* sorted, const int* offsets,
                                                            uint16_t* y, Meta m) {
    using St = F8Stage<StageK>;
    constexpr int KB = St::KB, P = StageK + 8;
    __shared__ __align__(16) uint8_t qa[16 * P], wb[64 * P];
    __shared__ float sa[16 * KB];
    __shared__ int source[16];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, e = blockIdx.y,
              colb = blockIdx.x * 64 * ColTiles;
    if (e == m.experts) {
        const int r0 = offsets[m.experts];
        for (long i = tid; i < (long)(m.T - r0) * 64 * ColTiles; i += 128) {
            const int c = colb + (int)(i % (64 * ColTiles));
            if (c < m.N) y[(long)(r0 + i / (64 * ColTiles)) * m.yr + c] = 0;
        }
        return;
    }
    const int lo = offsets[e], hi = offsets[e + 1];
    if (lo >= hi) return;
    const int parity = e & 1;
    const auto* w = entry(parity ? w1 : w0, m.wstep[parity], e / 2);
    const auto* ws = (const float*)entry(parity ? s1 : s0, m.sstep[parity], e / 2);
    const int ksn = (m.K + StageK - 1) / StageK, tiles = min(ColTiles, (m.N - colb + 63) / 64),
              S = tiles * ksn;
    const int rowbase = (lane >> 4) * 8;
    for (int r0 = lo; r0 < hi; r0 += 16) {
        __syncthreads();
        if (tid < 16) {
            const int r = r0 + tid, flat = r < hi ? sorted[r] : -1;
            source[tid] = r < hi && flat >= 0 ? (m.sorted_a ? r : flat / m.topk) : -1;
        }
        __syncthreads();
        bool rv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) rv[j] = source[rowbase + j] >= 0;
        auto load = [&](int s, St& R) {
            const int col0 = colb + (s / ksn) * 64, k0 = (s % ksn) * StageK;
#pragma unroll
            for (int i = 0; i < StageK / 128; ++i) {
                const int j = tid + i * 128, r = j / (StageK / 16), c = (j % (StageK / 16)) * 16,
                          src = source[r];
                R.a[i] = f8_u4{ 0, 0, 0, 0 };
                if (src >= 0 && k0 + c < m.K)
                    __builtin_memcpy(&R.a[i], a + (long)src * m.ar + k0 + c, 16);
            }
#pragma unroll
            for (int i = 0; i < StageK / 32; ++i) {
                const int j = tid + i * 128, r = j / (StageK / 16), c = (j % (StageK / 16)) * 16;
                R.w[i] = f8_u4{ 0, 0, 0, 0 };
                if (k0 + c < m.K)
                    __builtin_memcpy(&R.w[i], w + (long)(col0 + r) * m.wr + k0 + c, 16);
            }
#pragma unroll
            for (int kb = 0; kb < KB; ++kb)
                R.wsc[kb] =
                    k0 + kb * 128 < m.K ? ws[(long)(col0 / 128) * m.sr + (k0 + kb * 128) / 128] : 0;
            if (tid < 16 * KB) {
                const int src = source[tid / KB], kb = k0 + (tid % KB) * 128;
                R.sa = src >= 0 && kb < m.K ? as[(long)src * m.asr + kb / 128] : 0;
            }
        };
        auto store = [&](const St& R) {
#pragma unroll
            for (int i = 0; i < StageK / 128; ++i) {
                const int j = tid + i * 128, r = j / (StageK / 16), c = (j % (StageK / 16)) * 16;
                *reinterpret_cast<f8_u4*>(qa + r * P + c) = R.a[i];
            }
#pragma unroll
            for (int i = 0; i < StageK / 32; ++i) {
                const int j = tid + i * 128, r = j / (StageK / 16), c = (j % (StageK / 16)) * 16;
                *reinterpret_cast<f8_u4*>(wb + r * P + c) = R.w[i];
            }
            if (tid < 16 * KB) sa[tid] = R.sa;
        };
        auto acc = r4d_fp8::acc_zero();
        St R;
        load(0, R);
        for (int s = 0; s < S; ++s) {
            store(R);
            float wsc[KB];
#pragma unroll
            for (int kb = 0; kb < KB; ++kb) wsc[kb] = R.wsc[kb];
            __syncthreads();
            if (s + 1 < S) load(s + 1, R);
            const int k0 = (s % ksn) * StageK;
#pragma unroll
            for (int kb = 0; kb < KB; ++kb) {
                if (k0 + kb * 128 >= m.K) break;
                auto raw = r4d_fp8::acc_zero();
#pragma unroll
                for (int k = kb * 128; k < (kb + 1) * 128; k += 16)
                    raw = r4d_fp8::mma(r4d_fp8::frag<StageK>(qa, 0, k, lane),
                                       r4d_fp8::frag<StageK>(wb, wave * 16, k, lane), raw);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float scale = rv[j] ? sa[(rowbase + j) * KB + kb] * wsc[kb] : 0;
                    acc[j] += raw[j] * scale;
                }
            }
            if (s % ksn == ksn - 1) {
                const int c = colb + (s / ksn) * 64 + wave * 16 + (lane & 15);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int r = r0 + rowbase + j;
                    if (r < hi) y[(long)r * m.yr + c] = f32_to_bf16(acc[j]);
                }
                acc = r4d_fp8::acc_zero();
            }
            __syncthreads();
        }
    }
}
/* The small-M grouped form at GLM's MTP gate_up slice, with fp8_f32scale_experts' per-element
 * sequence and so its bytes. A draft pass routes one token to eight experts, and the four column
 * tiles a block of the form above leave sixteen blocks, each a chain of 64 stages of 256 K. Here a
 * block takes one column tile of the y-th touched expert (the grid has min(experts, rows) + 1
 * rows, as at the MXFP4 small-M form), in stages of 512 K, with loads unconditional from clamped
 * addresses and masked at the store, barriers that order LDS only, and a 128 of K's eight fragment
 * pairs loaded before its WMMA chain. */
typedef unsigned f8_g16 __attribute__((ext_vector_type(4), aligned(1)));
__device__ __forceinline__ f8_u4 f8_ldg16(const uint8_t* p) {
    return *(const __attribute__((address_space(1))) f8_g16*)p;
}
__device__ __forceinline__ float f8_ldgf(const float* p) {
    return *(const __attribute__((address_space(1))) float*)p;
}
template <int StageK>
struct F8Small {
    static constexpr int Threads = 128, KB = StageK / 128, P = StageK + 8;
    static constexpr int AChunks = StageK / Threads, WChunks = 64 * StageK / 16 / Threads;
    f8_u4 a[AChunks], w[WChunks];
    float wsc[KB], sa;
    uint32_t ok;
};
template <int StageK>
__global__ __launch_bounds__(128) void fp8_f32scale_small(const uint8_t* a, const float* as,
                                                          const void* w0, const void* s0,
                                                          const void* w1, const void* s1,
                                                          const int* sorted, const int* offsets,
                                                          uint16_t* y, Meta m) {
    using St = F8Small<StageK>;
    constexpr int KB = St::KB, P = St::P, Threads = St::Threads;
    static_assert(St::AChunks <= 8 && St::WChunks <= 16 && KB <= 4 && 16 * KB <= Threads &&
                      64 * P >= 4 * 513,
                  "stage chunks must fit the load mask and the offsets the weight tile");
    __shared__ __align__(16) uint8_t qa[16 * P], wb[64 * P];
    __shared__ float sa[16 * KB];
    __shared__ int source[16];
    __shared__ int pick;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, col0 = blockIdx.x * 64;
    if ((int)blockIdx.y == (int)gridDim.y - 1) {
        const int r0 = offsets[m.experts];
        for (long i = tid; i < (long)(m.T - r0) * 64; i += Threads) {
            const int c = col0 + (int)(i % 64);
            if (c < m.N) y[(long)(r0 + i / 64) * m.yr + c] = 0;
        }
        return;
    }
    {
        // The offsets in the weight tile, whose bytes are free until the first stage.
        int* offs = reinterpret_cast<int*>(wb);
#pragma unroll
        for (int q = 0; q < 5; ++q) {
            const int x = tid + q * Threads;
            if (x <= m.experts) offs[x] = offsets[x];
        }
        lds_barrier();
        if (wave == 0) {
            int want = blockIdx.y, found = -1;
            for (int base = 0; base < m.experts; base += 32) {
                const int x = base + lane;
                const bool live = x < m.experts && offs[x + 1] > offs[x];
                const uint32_t mask = __builtin_amdgcn_ballot_w32(live);
                const int n = __builtin_popcount(mask);
                if (want < n) {
                    const uint32_t below = mask & ((1u << lane) - 1u);
                    found = base + __builtin_ctz(__builtin_amdgcn_ballot_w32(
                                       live && __builtin_popcount(below) == want));
                    break;
                }
                want -= n;
            }
            if (lane == 0) pick = found;
        }
        lds_barrier();
    }
    const int e = pick;
    if (e < 0) return;
    const int lo = offsets[e], hi = offsets[e + 1];
    const int parity = e & 1;
    const auto* w = entry(parity ? w1 : w0, m.wstep[parity], e / 2);
    const auto* ws = (const float*)entry(parity ? s1 : s0, m.sstep[parity], e / 2);
    const int S = (m.K + StageK - 1) / StageK, rowbase = (lane >> 4) * 8;
    for (int r0 = lo; r0 < hi; r0 += 16) {
        lds_barrier();
        if (tid < 16) {
            const int r = r0 + tid, flat = r < hi ? sorted[r] : -1;
            source[tid] = flat >= 0 ? (m.sorted_a ? r : flat / m.topk) : -1;
        }
        lds_barrier();
        bool rv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) rv[j] = source[rowbase + j] >= 0;
        auto load = [&](int s, St& L) {
            const int k0 = s * StageK;
            uint32_t ok = 0;
#pragma unroll
            for (int i = 0; i < St::AChunks; ++i) {
                const int j = tid + i * Threads, r = j / (StageK / 16), c = (j % (StageK / 16)) * 16,
                          src = source[r];
                const bool v = src >= 0 && k0 + c < m.K;
                __builtin_memcpy(&L.a[i], a + (v ? (long)src * m.ar + k0 + c : 0), 16);
                ok |= (uint32_t)v << i;
            }
#pragma unroll
            for (int i = 0; i < St::WChunks; ++i) {
                const int j = tid + i * Threads, r = j / (StageK / 16), c = (j % (StageK / 16)) * 16;
                const bool v = k0 + c < m.K;
                L.w[i] = f8_ldg16(w + (v ? (long)(col0 + r) * m.wr + k0 + c : 0));
                ok |= (uint32_t)v << (8 + i);
            }
#pragma unroll
            for (int kb = 0; kb < KB; ++kb) {
                const bool v = k0 + kb * 128 < m.K;
                L.wsc[kb] =
                    f8_ldgf(ws + (v ? (long)(col0 / 128) * m.sr + (k0 + kb * 128) / 128 : 0));
                ok |= (uint32_t)v << (24 + kb);
            }
            {
                const int src = source[(tid / KB) & 15], kb = k0 + (tid % KB) * 128;
                const bool v = tid < 16 * KB && src >= 0 && kb < m.K;
                L.sa = as[v ? (long)src * m.asr + kb / 128 : 0];
                ok |= (uint32_t)v << 28;
            }
            L.ok = ok;
        };
        auto store = [&](const St& L) {
#pragma unroll
            for (int i = 0; i < St::AChunks; ++i) {
                const int j = tid + i * Threads, r = j / (StageK / 16), c = (j % (StageK / 16)) * 16;
                *reinterpret_cast<f8_u4*>(qa + r * P + c) =
                    (L.ok >> i) & 1u ? L.a[i] : f8_u4{ 0, 0, 0, 0 };
            }
#pragma unroll
            for (int i = 0; i < St::WChunks; ++i) {
                const int j = tid + i * Threads, r = j / (StageK / 16), c = (j % (StageK / 16)) * 16;
                *reinterpret_cast<f8_u4*>(wb + r * P + c) =
                    (L.ok >> (8 + i)) & 1u ? L.w[i] : f8_u4{ 0, 0, 0, 0 };
            }
            if (tid < 16 * KB) sa[(tid % KB) * 16 + tid / KB] = (L.ok >> 28) & 1u ? L.sa : 0.f;
        };
        auto acc = r4d_fp8::acc_zero();
        St L;
        load(0, L);
        for (int s = 0; s < S; ++s) {
            store(L);
            float wsc[KB];
#pragma unroll
            for (int kb = 0; kb < KB; ++kb) wsc[kb] = (L.ok >> (24 + kb)) & 1u ? L.wsc[kb] : 0.f;
            lds_barrier();
            if (s + 1 < S) load(s + 1, L);
            const int k0 = s * StageK;
#pragma unroll
            for (int kb = 0; kb < KB; ++kb) {
                if (k0 + kb * 128 >= m.K) break;
                r4d_fp8::i2v fa[8], fb[8];
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    fa[t] = r4d_fp8::frag<StageK>(qa, 0, kb * 128 + 16 * t, lane);
                    fb[t] = r4d_fp8::frag<StageK>(wb, wave * 16, kb * 128 + 16 * t, lane);
                }
                auto raw = r4d_fp8::acc_zero();
#pragma unroll
                for (int t = 0; t < 8; ++t) raw = r4d_fp8::mma(fa[t], fb[t], raw);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float scale = rv[j] ? sa[kb * 16 + rowbase + j] * wsc[kb] : 0;
                    acc[j] += raw[j] * scale;
                }
            }
            lds_barrier();
        }
        const int c = col0 + wave * 16 + (lane & 15);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int r = r0 + rowbase + j;
            if (r < hi) y[(long)r * m.yr + c] = f32_to_bf16(acc[j]);
        }
    }
}
// Both forms below keep fp8_f32scale's arithmetic for every output: per 128-K block the same
// eight-mma chain from zero over the same fragments (the bytes at k + 8*(lane/16) of the block's
// row segment, which the legacy form copies through LDS unchanged), then acc += raw * (a_scale *
// w_scale) in ascending block order. A WMMA output depends only on its own A row, B column and
// C, so which rows and columns share a tile does not change it.

// Decode: one wave C adjacent 16-column tiles, S waves of a block on S consecutive K blocks at
// once, the fragments loaded straight from global memory; R row tiles share each weight fragment
// and C column tiles each activation fragment. Each wave scales its own partial products (the
// a_scale x w_scale product and the multiply are the legacy fold's own two roundings) into LDS;
// then every wave folds a share of the outputs, adding the round's products in block order to the
// output's running sum, which lives in LDS between rounds. The block's activation scales are
// staged in LDS when they fit (`stage`).
template <int R, int C>
__global__ void fp8_f32scale_decode(const uint8_t* a, const float* as, const uint8_t* w,
                                    const float* ws, uint16_t* y, Meta m, int stage) {
    constexpr int SL = R * C * 8;
    R4D_DYN_LDS(float, lds);
    const int lane = threadIdx.x & 31, s = threadIdx.x >> 5, S = blockDim.x >> 5, h = lane >> 4;
    const int c0 = blockIdx.x * 16 * C, first = (int)blockIdx.y * R * 16, KB = m.K / 128;
    float* part = lds;
    float* sums = part + S * SL * 32;
    float* scales = sums + SL * 32;
    if (stage)
        for (int i = threadIdx.x; i < R * 16 * KB; i += blockDim.x) {
            const int r = first + i / KB;
            scales[i] = r < m.T ? as[(long)r * m.asr + i % KB] : 0;
        }
    for (int i = threadIdx.x; i < SL * 32; i += blockDim.x) sums[i] = 0;
    const uint8_t* arow[R];
    bool alive[R];
#pragma unroll
    for (int t = 0; t < R; ++t) {
        const int r = first + t * 16 + (lane & 15);
        alive[t] = r < m.T;
        arow[t] = a + (long)(alive[t] ? r : 0) * m.ar + h * 8;
    }
    __syncthreads();
    for (int k0 = 0; k0 < KB; k0 += S) {
        const int kb = k0 + s;
        if (kb < KB) {
            r4d_fp8::i2v b[C][8];
            float weight_scale[C];
#pragma unroll
            for (int c = 0; c < C; ++c) {
                const uint8_t* wrow =
                    w + (long)(c0 + c * 16 + (lane & 15)) * m.wr + h * 8 + (long)kb * 128;
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    const uint2 v = *(const uint2*)(wrow + q * 16);
                    b[c][q][0] = (int)v.x;
                    b[c][q][1] = (int)v.y;
                }
                weight_scale[c] = ws[(long)((c0 + c * 16) / 128) * m.sr + kb];
            }
#pragma unroll
            for (int t = 0; t < R; ++t) {
                r4d_fp8::i2v f[8];
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    uint2 v = make_uint2(0, 0);
                    if (alive[t]) v = *(const uint2*)(arow[t] + (long)kb * 128 + q * 16);
                    f[q][0] = (int)v.x;
                    f[q][1] = (int)v.y;
                }
                float act[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int rr = t * 16 + h * 8 + j, r = first + rr;
                    act[j] =
                        stage ? scales[rr * KB + kb] : (r < m.T ? as[(long)r * m.asr + kb] : 0);
                }
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    auto raw = r4d_fp8::acc_zero();
#pragma unroll
                    for (int q = 0; q < 8; ++q) raw = r4d_fp8::mma(f[q], b[c][q], raw);
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int r = first + t * 16 + h * 8 + j;
                        const float scale = r < m.T ? act[j] * weight_scale[c] : 0;
                        part[((long)s * SL + (t * C + c) * 8 + j) * 32 + lane] = raw[j] * scale;
                    }
                }
            }
        }
        __syncthreads();
        for (int slot = s; slot < SL; slot += S) {
            float acc = sums[slot * 32 + lane];
            for (int q = 0; q < S && k0 + q < KB; ++q)
                acc += part[((long)q * SL + slot) * 32 + lane];
            sums[slot * 32 + lane] = acc;
        }
        __syncthreads();
    }
    for (int slot = s; slot < SL; slot += S) {
        const int t = slot / (C * 8), c = slot / 8 % C, j = slot % 8,
                  r = first + t * 16 + h * 8 + j;
        if (r < m.T)
            y[(long)r * m.yr + c0 + c * 16 + (lane & 15)] = f32_to_bf16(sums[slot * 32 + lane]);
    }
}

// Prefill: the legacy form's 64-column block and LDS layout over R row tiles, so a staged weight
// block serves R*16 rows instead of 16, staged with 16-byte loads and the next K block loaded
// into registers while the current one is multiplied.
template <int R>
__global__ void __launch_bounds__(128)
    fp8_f32scale_rows(const uint8_t* a, const float* as, const uint8_t* w, const float* ws,
                      uint16_t* y, Meta m) {
    constexpr int P = 136, AV = R * 16 * 8 / 128, WV = 64 * 8 / 128;
    __shared__ uint8_t qa[R * 16 * P], wb[64 * P];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, h = lane >> 4,
              n0 = blockIdx.x * 64;
    const int first = (int)blockIdx.y * R * 16, KB = m.K / 128;
    uint4 av[AV], wv[WV];
    auto load = [&](int kb) {
#pragma unroll
        for (int i = 0; i < AV; ++i) {
            const int j = tid + i * 128, r = j / 8, c = (j % 8) * 16;
            av[i] = first + r < m.T
                        ? *(const uint4*)(a + (long)(first + r) * m.ar + (long)kb * 128 + c)
                        : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int i = 0; i < WV; ++i) {
            const int j = tid + i * 128, r = j / 8, c = (j % 8) * 16;
            wv[i] = *(const uint4*)(w + (long)(n0 + r) * m.wr + (long)kb * 128 + c);
        }
    };
    r4d_fp8::f8v acc[R];
#pragma unroll
    for (int t = 0; t < R; ++t) acc[t] = r4d_fp8::acc_zero();
    load(0);
    for (int kb = 0; kb < KB; ++kb) {
        // The 136-byte pitch keeps 8-byte alignment only, so each 16 bytes go in as two halves.
        auto put = [](uint8_t* p, const uint4& v) {
            ((uint2*)p)[0] = make_uint2(v.x, v.y);
            ((uint2*)p)[1] = make_uint2(v.z, v.w);
        };
#pragma unroll
        for (int i = 0; i < AV; ++i) {
            const int j = tid + i * 128;
            put(qa + (j / 8) * P + (j % 8) * 16, av[i]);
        }
#pragma unroll
        for (int i = 0; i < WV; ++i) {
            const int j = tid + i * 128;
            put(wb + (j / 8) * P + (j % 8) * 16, wv[i]);
        }
        __syncthreads();
        if (kb + 1 < KB) load(kb + 1);
        const float weight_scale = ws[(long)(n0 / 128) * m.sr + kb];
#pragma unroll
        for (int t = 0; t < R; ++t) {
            auto raw = r4d_fp8::acc_zero();
#pragma unroll
            for (int k = 0; k < 128; k += 16)
                raw = r4d_fp8::mma(r4d_fp8::frag<128>(qa, t * 16, k, lane),
                                   r4d_fp8::frag<128>(wb, wave * 16, k, lane), raw);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int r = first + t * 16 + h * 8 + j;
                const float scale = r < m.T ? as[(long)r * m.asr + kb] * weight_scale : 0;
                acc[t][j] += raw[j] * scale;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int t = 0; t < R; ++t)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int r = first + t * 16 + h * 8 + j, c = n0 + wave * 16 + (lane & 15);
            if (r < m.T) y[(long)r * m.yr + c] = f32_to_bf16(acc[t][j]);
        }
}

// The dense form: a decode form (`decode`, C column tiles a wave, S waves a block) or the row
// tiles (R of 16 rows a block), by the rule at the end of dense_form. The forms are
// byte-identical; the choice is speed only.
struct Form {
    bool decode = false;
    int r = 1, s = 1, c = 1;
};
constexpr int kLdsCap = 64 * 1024;
long decode_lds(int R, int S, int C, long KB, bool stage) {
    return ((long)S * R * C * 8 * 32 + (long)R * C * 8 * 32 + (stage ? (long)R * 16 * KB : 0)) * 4;
}
Form dense_form(long N, long K, long T) {
    // Measured on gfx1201 (r4d_selftest --perfq, cold) at GLM's six shapes: decode1x8 up to 16
    // rows; from 17 to 64 two column tiles a wave (decode1x8x2) lead by 5 to 15 %, except at N 512
    // (too few column tiles, 1x8 stays) and N 4096 over K 1536, where the 16-row tile wins; from 65
    // to 128 rows decode1x8 holds at K 4096 and the row tiles win at K 2048 and below. Among the
    // row tiles (and past 128 rows) the 16-row tile beats the 64-row tile by a quarter; the 32-row
    // tile wins at N 4096.
    const long kb = K / 128;
    Form f;
    const bool rows =
        T > 128 || (T > 64 && K <= 2048) || (T > 16 && T <= 64 && K <= 1536 && N >= 4096);
    if (!rows) {
        f.decode = true;
        f.c = T > 16 && T <= 64 && N > 512 ? 2 : 1;
        f.s = 8;
        while (f.s > 1 && f.s > kb) f.s /= 2;
        return f;
    }
    f.r = T > 64 && N >= 4096 ? 2 : 1;
    return f;
}
template <int R>
int emit_rows(RadStream stream, const uint8_t* xa, const float* xsa, const uint8_t* wa,
              const float* wsa, uint16_t* ya, const Meta& m) {
    return R4D_EMIT(stream, fp8_f32scale_rows<R>,
                    dim3(m.N / 64, (unsigned)((m.T + R * 16 - 1) / (R * 16))), dim3(128), 0, xa,
                    xsa, wa, wsa, ya, m);
}
int dense_fast(RadStream stream, const RadTensor& x, const RadTensor& xs, const RadTensor& w,
               const RadTensor& ws, const RadTensor& y, const Meta& m, bool* done) {
    *done = false;
    const Form f = dense_form(m.N, m.K, m.T);
    if (m.wr % 16 || m.ar % 16 ||
        (reinterpret_cast<uintptr_t>(x.data) | reinterpret_cast<uintptr_t>(w.data)) % 16)
        return RAD_OK;
    const uint8_t *xa = (const uint8_t*)x.data, *wa = (const uint8_t*)w.data;
    const float *xsa = (const float*)xs.data, *wsa = (const float*)ws.data;
    uint16_t* ya = (uint16_t*)y.data;
    if (f.decode) {
        const long kb = m.K / 128;
        const int stage = decode_lds(1, f.s, f.c, kb, true) <= kLdsCap;
        const long lds = decode_lds(1, f.s, f.c, kb, stage);
        if (lds > kLdsCap || m.N % (16 * f.c)) return RAD_OK;
        *done = true;
        const dim3 grid(m.N / (16 * f.c), (unsigned)((m.T + 15) / 16)), block(32 * f.s);
        if (f.c == 2)
            return R4D_EMIT(stream, (fp8_f32scale_decode<1, 2>), grid, block, (uint32_t)lds, xa,
                            xsa, wa, wsa, ya, m, stage);
        return R4D_EMIT(stream, (fp8_f32scale_decode<1, 1>), grid, block, (uint32_t)lds, xa, xsa,
                        wa, wsa, ya, m, stage);
    }
    *done = true;
    if (f.r == 2) return emit_rows<2>(stream, xa, xsa, wa, wsa, ya, m);
    return emit_rows<1>(stream, xa, xsa, wa, wsa, ya, m);
}
int body(const RadArgs* a, RadStream stream, bool grouped) {
    if (!a || !a->t || a->n_t < (grouped ? 9 : 5)) return RAD_E_INVAL;
    const auto &x = a->t[0], &xs = a->t[1], &w = a->t[2], &ws = a->t[3], &y = a->t[grouped ? 8 : 4];
    const int64_t N = r4d_geti_or(a, "N", 0), K = r4d_geti_or(a, "K", 0);
    if (N < 128 || N % 128 || K < 128 || K % 128 || N > INT_MAX || K > INT_MAX ||
        r4d_geti_or(a, "group", 0) != 128 || r4d_geti_or(a, "pb1", 0) > 0)
        return RAD_E_SHAPE;
    if (x.dtype != RAD_F8E4M3 || xs.dtype != RAD_F32 || w.dtype != RAD_F8E4M3 ||
        ws.dtype != RAD_F32 || y.dtype != RAD_BF16)
        return RAD_E_DTYPE;
    if (!x.data || !xs.data || !w.data || !ws.data || !y.data || x.rank != 2 || xs.rank != 2 ||
        y.rank != 2 || x.shape[1] != K || xs.shape[0] != x.shape[0] || xs.shape[1] != K / 128 ||
        y.shape[1] != N || x.stride[1] != 1 || xs.stride[1] != 1 || y.stride[1] != 1 ||
        y.shape[0] < 1 || y.shape[0] > INT_MAX)
        return RAD_E_SHAPE;
    const uint32_t rank = grouped ? 3 : 2;
    if (w.rank != rank || ws.rank != rank || w.shape[rank - 2] != N || w.shape[rank - 1] != K ||
        ws.shape[rank - 2] != N / 128 || ws.shape[rank - 1] != K / 128 || w.stride[rank - 1] != 1 ||
        ws.stride[rank - 1] != 1)
        return RAD_E_SHAPE;
    Meta m{};
    m.ar = x.stride[0];
    m.asr = xs.stride[0];
    m.yr = y.stride[0];
    m.wr = w.stride[rank - 2];
    m.sr = ws.stride[rank - 2];
    m.T = (int)y.shape[0];
    m.N = N;
    m.K = K;
    if (!grouped) {
        if (x.shape[0] != m.T) return RAD_E_SHAPE;
        bool done = false;
        const int rc = dense_fast(stream, x, xs, w, ws, y, m, &done);
        if (done || rc < 0) return rc;
        return R4D_EMIT(stream, fp8_f32scale, dim3(N / 64, (m.T + 15) / 16), dim3(128), 0,
                        (const uint8_t*)x.data, (const float*)xs.data, (const uint8_t*)w.data,
                        (const float*)ws.data, (uint16_t*)y.data, m);
    }
    m.experts = r4d_geti_or(a, "n_expert", 0);
    m.topk = r4d_geti_or(a, "top_k", 0);
    const char* order = r4d_gets(a, "a_order");
    m.sorted_a = order && !std::strcmp(order, "sorted");
    const auto &wo = a->t[6], &so = a->t[7], &st = a->t[4], &off = a->t[5];
    if (m.experts < 2 || m.topk < 1 || r4d_geti_or(a, "N_odd", N) != N ||
        r4d_geti_or(a, "K_odd", K) != K || wo.dtype != RAD_F8E4M3 || so.dtype != RAD_F32 ||
        wo.rank != 3 || so.rank != 3 || wo.shape[1] != N || wo.shape[2] != K ||
        so.shape[1] != N / 128 || so.shape[2] != K / 128 || wo.stride[1] != m.wr ||
        so.stride[1] != m.sr || wo.stride[2] != 1 || so.stride[2] != 1 || st.dtype != RAD_I32 ||
        off.dtype != RAD_I32 || st.rank != 1 || off.rank != 1 || st.stride[0] != 1 ||
        off.stride[0] != 1 || st.shape[0] < m.T || off.shape[0] < m.experts + 1 || !wo.data ||
        !so.data || !st.data || !off.data)
        return RAD_E_SHAPE;
    m.wstep[0] = w.stride[0];
    m.wstep[1] = wo.stride[0];
    m.sstep[0] = ws.stride[0] * 4;
    m.sstep[1] = so.stride[0] * 4;
    if (w.shape[0] < (m.experts + 1) / 2 || wo.shape[0] < m.experts / 2) return RAD_E_SHAPE;
    /* The small-M form at the MTP layer's gate_up slice up to 384 routed rows: a draft pass routes
     * one token, a verify pass a few. On gfx1201 (perfmxf8, cold) it halves the expert-major time
     * at one token and gains 3 to 21 % from 2 to 16. Down's K of 256 gains nothing from it (within
     * a few percent, a quarter slower at three tokens). */
    if (m.experts == 288 && m.topk == 8 && N == 512 && K == 4096 && m.T <= 384)
        return R4D_EMIT(stream, fp8_f32scale_small<512>,
                        dim3(N / 64, (m.T < m.experts ? m.T : m.experts) + 1), dim3(128), 0,
                        (const uint8_t*)x.data, (const float*)xs.data, w.data, ws.data, wo.data,
                        so.data, (const int*)st.data, (const int*)off.data, (uint16_t*)y.data, m);
    /* The expert-major form at every routed-row count. Measured at GLM's TP8 slices: a long K
     * (gate_up's 4096) wants the 256 stage, 25 to 40 % ahead of 128, and under 128 routed rows
     * four tiles a block on top; at down's K of 256 every variant reads within a few percent of
     * the bandwidth, so it keeps 128 over four tiles. */
#define F8_EXPERTS(SK, C)                                                                          \
    R4D_EMIT(stream, (fp8_f32scale_experts<SK, C>),                                                \
             dim3((N + 64 * C - 1) / (64 * C), m.experts + 1), dim3(128), 0,                       \
             (const uint8_t*)x.data, (const float*)xs.data, w.data, ws.data, wo.data, so.data,     \
             (const int*)st.data, (const int*)off.data, (uint16_t*)y.data, m)
    if (K > 256) return m.T < 128 ? F8_EXPERTS(256, 4) : F8_EXPERTS(256, 1);
    return F8_EXPERTS(128, 4);
#undef F8_EXPERTS
}
int dense(const RadArgs* a, RadStream stream) { return body(a, stream, false); }
int grouped(const RadArgs* a, RadStream stream) { return body(a, stream, true); }

/* tp8 loop P30o r2 (WIP, first form): DeepSeek-V4's released block FP8 (gemm_nt_q dtype fp8a8e8) -- E4M3 [N, K] with an E8M0
 * exponent per 32 x 32 block against quant_act_fp8_ue8m0's pair (an f32 power-of-two scale per row per 32 of K). The legacy
 * 16 x 64 tile above (LDS staged a dword at a time, any activation pitch, C at its own row pitch) with the release's fold:
 * per 32 of K a fresh chain of two WMMAs, folded as raw * (activation scale * weight scale) and added (inference/kernel.py
 * fp8_gemm at block 32: C_accum += C_local * sa * sb; both scales powers of two, so the two products are exact and the
 * fold's one rounding is the add). A wave's 16 weight rows sit in one 32-row scale block. Correctness form: the narrow /
 * prefill speed forms are r2 (b). */
__device__ __forceinline__ float e8m0f(uint32_t e) { return __uint_as_float(e ? e << 23 : 0x00400000u); }
/* MT activation rows a block (PREFILL round 21, P30o r2 (b) step 1): 16 = the first form; 32 / 64 stage MT rows of the activation
 * beside the same 64 weight rows, so a block reads its weights once for MT rows instead of 16 (the first form re-reads every weight
 * T / 16 times: at M 1100 the dense is weight-read bound), and each wave reuses its weight fragments over MT / 16 accumulator
 * tiles. Every output's arithmetic is the first form's (the same two-WMMA chain a 32-K block over the same 16 K halves, the same
 * fold in the same block order), so the bytes do not depend on MT. */
/* V16 (PREFILL round 21): the operands' rows and pitches 16-byte aligned (the engine's pair, the sidecar's weights) -- each thread
 * stages 16 bytes a load (two 8-byte LDS stores: the 136-byte LDS row keeps 8-byte alignment) instead of four byte loads a dword;
 * the same bytes land in the same LDS places, so nothing else changes */
template <int MT, bool V16>
__global__ void fp8e8_b32(const uint8_t* a, const float* as, const uint8_t* w, const uint8_t* ws, uint16_t* y, Meta m) {
    constexpr int F = MT / 16;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, n0 = blockIdx.x * 64;
    __shared__ uint8_t qa[MT * 136], wb[64 * 136];
    __shared__ int source[MT];
    const int first = (int)blockIdx.y * MT, end = min(first + MT, m.T);
    if (tid < MT) {
        const int r = first + tid;
        source[tid] = r < end ? r : -1;
    }
    __syncthreads();
    decltype(r4d_fp8::acc_zero()) acc[F];
#pragma unroll
    for (int f = 0; f < F; ++f) acc[f] = r4d_fp8::acc_zero();
    const long wsrow = (long)((n0 + wave * 16) / 32) * m.sr;
    for (int k0 = 0; k0 < m.K; k0 += 128) {
        if constexpr (V16) {
            for (int j = tid; j < MT * 8; j += 128) {
                const int r = j / 8, c = (j % 8) * 16, src = source[r];
                uint4 v = make_uint4(0u, 0u, 0u, 0u);
                if (src >= 0) v = *(const uint4*)(a + (long)src * m.ar + k0 + c);
                *(uint2*)(qa + r * 136 + c) = make_uint2(v.x, v.y);
                *(uint2*)(qa + r * 136 + c + 8) = make_uint2(v.z, v.w);
            }
            for (int j = tid; j < 64 * 8; j += 128) {
                const int r = j / 8, c = (j % 8) * 16;
                const uint4 v = *(const uint4*)(w + (long)(n0 + r) * m.wr + k0 + c);
                *(uint2*)(wb + r * 136 + c) = make_uint2(v.x, v.y);
                *(uint2*)(wb + r * 136 + c + 8) = make_uint2(v.z, v.w);
            }
        } else {
        for (int j = tid; j < MT * 32; j += 128) {
            const int r = j / 32, c = (j % 32) * 4, src = source[r];
            uint32_t v = 0;
            if (src >= 0) {
                const auto* p = a + (long)src * m.ar + k0 + c;
                v = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
            }
            *(uint32_t*)(qa + r * 136 + c) = v;
        }
        for (int j = tid; j < 64 * 32; j += 128) {
            const int r = j / 32, c = (j % 32) * 4;
            const auto* p = w + (long)(n0 + r) * m.wr + k0 + c;
            *(uint32_t*)(wb + r * 136 + c) = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
                                             ((uint32_t)p[3] << 24);
        }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 128; kk += 32) {
            const auto wf0 = r4d_fp8::frag<128>(wb, wave * 16, kk, lane), wf1 = r4d_fp8::frag<128>(wb, wave * 16, kk + 16, lane);
            const int kb = (k0 + kk) / 32;
            const float wsc = e8m0f(ws[wsrow + kb]);
#pragma unroll
            for (int f = 0; f < F; ++f) {
                auto raw = r4d_fp8::acc_zero();
                raw = r4d_fp8::mma(r4d_fp8::frag<128>(qa, f * 16, kk, lane), wf0, raw);
                raw = r4d_fp8::mma(r4d_fp8::frag<128>(qa, f * 16, kk + 16, lane), wf1, raw);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int src = source[f * 16 + (lane >> 4) * 8 + j];
                    const float scale = src >= 0 ? as[(long)src * m.asr + kb] * wsc : 0;
                    acc[f][j] += raw[j] * scale;
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int f = 0; f < F; ++f)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int r = first + f * 16 + (lane >> 4) * 8 + j, c = n0 + wave * 16 + (lane & 15);
            if (r < end) y[(long)r * m.yr + c] = f32_to_bf16(acc[f][j]);
        }
}
/* the M tile: R4D_FP8E8_MT=16 / 32 / 64 at load (unset: 16, the first form), r4d_fp8e8_pin_mt() in process (selftest, emu);
 * a launch of at most 16 rows always takes 16 (one tile either way). The staging: R4D_FP8E8_LD=4 at load forces the first form's
 * byte loads, else 16-byte loads wherever the operands allow; r4d_fp8e8_pin_ld(4 / 16 / 0) in process */
std::atomic<int> g_fp8e8_mt{ -1 }, g_fp8e8_ld{ -1 };
int fp8e8_ld() {
    int v = g_fp8e8_ld.load(std::memory_order_relaxed);
    if (v < 0) {
        const char* e = std::getenv("R4D_FP8E8_LD");
        v = e && !std::strcmp(e, "4") ? 4 : 16;
        g_fp8e8_ld.store(v, std::memory_order_relaxed);
    }
    return v;
}
int fp8e8_mt() {
    int v = g_fp8e8_mt.load(std::memory_order_relaxed);
    if (v < 0) {
        const char* e = std::getenv("R4D_FP8E8_MT");
        v = e && (!std::strcmp(e, "32") || !std::strcmp(e, "64")) ? std::atoi(e) : 16;
        g_fp8e8_mt.store(v, std::memory_order_relaxed);
    }
    return v;
}
int dense_e8(const RadArgs* a, RadStream stream) {
    if (!a || !a->t || a->n_t < 5) return RAD_E_INVAL;
    const auto &x = a->t[0], &xs = a->t[1], &w = a->t[2], &ws = a->t[3], &y = a->t[4];
    const int64_t N = r4d_geti_or(a, "N", 0), K = r4d_geti_or(a, "K", 0);
    if (N < 64 || N % 64 || K < 128 || K % 128 || N > INT_MAX || K > INT_MAX || r4d_geti_or(a, "group", 0) != 32 ||
        r4d_geti_or(a, "pb1", 0) > 0)
        return R4D_NO(RAD_E_SHAPE);
    if (x.dtype != RAD_F8E4M3 || xs.dtype != RAD_F32 || w.dtype != RAD_F8E4M3 || ws.dtype != RAD_E8M0 ||
        y.dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (!x.data || !xs.data || !w.data || !ws.data || !y.data || x.rank != 2 || xs.rank != 2 || y.rank != 2 ||
        w.rank != 2 || ws.rank != 2 || x.shape[1] != K || xs.shape[0] != x.shape[0] || xs.shape[1] != K / 32 ||
        y.shape[1] != N || w.shape[0] != N || w.shape[1] != K || ws.shape[0] != N / 32 || ws.shape[1] != K / 32 ||
        x.stride[1] != 1 || xs.stride[1] != 1 || y.stride[1] != 1 || w.stride[1] != 1 || ws.stride[1] != 1 ||
        y.shape[0] < 1 || y.shape[0] > INT_MAX || x.shape[0] != y.shape[0])
        return R4D_NO(RAD_E_SHAPE);
    Meta m{};
    m.ar = x.stride[0];
    m.asr = xs.stride[0];
    m.yr = y.stride[0];
    m.wr = w.stride[0];
    m.sr = ws.stride[0];
    m.T = (int)y.shape[0];
    m.N = N;
    m.K = K;
    const int mt = m.T <= 16 ? 16 : fp8e8_mt();
    /* 16-byte staging: both operands' bases and row pitches (bytes: E4M3 is a byte a value) on 16 bytes */
    const bool v16 = fp8e8_ld() == 16 && !(((uintptr_t)x.data | (uintptr_t)w.data | (uintptr_t)m.ar | (uintptr_t)m.wr) & 15u);
#define FP8E8_GO(MT_, V_)                                                                                             \
    R4D_EMIT(stream, (fp8e8_b32<MT_, V_>), dim3(N / 64, (m.T + MT_ - 1) / MT_), dim3(128), 0, (const uint8_t*)x.data,    \
             (const float*)xs.data, (const uint8_t*)w.data, (const uint8_t*)ws.data, (uint16_t*)y.data, m)
    if (v16) {
        if (mt == 64) return FP8E8_GO(64, true);
        if (mt == 32) return FP8E8_GO(32, true);
        return FP8E8_GO(16, true);
    }
    if (mt == 64) return FP8E8_GO(64, false);
    if (mt == 32) return FP8E8_GO(32, false);
    return FP8E8_GO(16, false);
#undef FP8E8_GO
}
}
extern "C" int r4d_rad_fp8_f32scale(const RadArgs* a, RadStream stream) { return dense(a, stream); }
/* tp8 loop P30o r2 (WIP): gemm_nt_q at dtype fp8a8e8 (the release's block FP8), the first form */
extern "C" int r4d_rad_gemm_fp8a8e8(const RadArgs* a, RadStream stream) { return dense_e8(a, stream); }
/* PREFILL round 21: the fp8a8e8 form's M tile in process (16 / 32 / 64; anything else back to R4D_FP8E8_MT / 16) -- the same bytes */
extern "C" void r4d_fp8e8_pin_mt(int mt) { g_fp8e8_mt.store(mt == 16 || mt == 32 || mt == 64 ? mt : -1, std::memory_order_relaxed); }
/* ... and its staging (4 = the first form's byte loads, 16 = 16-byte loads where aligned; anything else back to R4D_FP8E8_LD) */
extern "C" void r4d_fp8e8_pin_ld(int ld) { g_fp8e8_ld.store(ld == 4 || ld == 16 ? ld : -1, std::memory_order_relaxed); }
/* Its planes as the release stores them (affine {codes fp8_e4m3, scale e8m0 [32x32]}): read without a relayout. */
extern "C" int r4d_layout_fp8a8e8(const RadParam*, int, int operand, const RadEncoding* e, const int* sel,
                                  const RadTensor*, int n, RadLayout*) {
    if (operand != 2 && operand != 3) return RAD_E_UNSUPPORTED;
    if (!e || !sel || n != 1 || !rad_enc_is(e, "affine") || e->n_planes != 2 || e->plane[0].dtype != RAD_F8E4M3 ||
        e->plane[1].dtype != RAD_E8M0 || e->plane[1].block[0] != 32 || e->plane[1].block[1] != 32 ||
        sel[0] != (operand == 3 ? 1 : 0))
        return RAD_E_DTYPE;
    return RAD_E_UNSUPPORTED;
}
extern "C" int r4d_rad_moe_fp8_f32scale(const RadArgs* a, RadStream stream) {
    return grouped(a, stream);
}
extern "C" int r4d_layout_fp8_f32scale(const RadParam*, int, int operand, const RadEncoding* e,
                                       const int* sel, const RadTensor*, int n, RadLayout*) {
    if (operand != 2 && operand != 3 && operand != 6 && operand != 7) return RAD_E_UNSUPPORTED;
    if (!e || !sel || n != 1 || !rad_enc_is(e, "affine") || e->n_planes != 2 ||
        e->plane[0].dtype != RAD_F8E4M3 || e->plane[1].dtype != RAD_F32 ||
        e->plane[1].block[0] != 128 || e->plane[1].block[1] != 128 ||
        sel[0] != (operand == 3 || operand == 7 ? 1 : 0))
        return RAD_E_DTYPE;
    return RAD_E_UNSUPPORTED; // Both canonical planes are read without a relayout.
}
