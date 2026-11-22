// r4d_ds41_pcn.h -- DeepSeek-V4.1's mhc_post -> mhc_collapse -> rmsnorm (mhc_post_collapse_rmsnorm) as the epilogue of a
// word-partitioned launch: the four-rank exact all-reduce's (r4d_ar_oneshot_Nrank_exact.hip, ar_mhc_post_collapse_rmsnorm)
// and its single-card test form (r4d_ds41_elem.hip, pcn_words_kernel). One definition, so both are the same bytes as
// r4d_ds41_elem.hip's post_collapse_norm_kernel, which forms a column and the norm the same way:
//
//   out[k][i] = f32_to_bf16(sum_j comb[j][k] h[j][i] (from zero, j ascending) + post[k] x[i])
//   xc[i]     = ds_bf16(sum_k pre[k] ds_bf(out[k][i]) (from zero, k ascending))
//   y[i]      = f32_to_bf16(bf(xc[i]) * rsqrtf(ss / n + eps) * (g[i] + wadd) * mult)
//
// ss is the norm kernel's: virtual thread v of THR = r4d_act_threads(M, n) adds bf(xc)^2 over columns v, v + THR, ..
// from zero; a wave of 32 virtual threads closes with r4d_wave_sum; the wave sums are added from zero in order.
// A launch's words are split over its blocks as the one-shot's are (ceil(words / nb) a block, rows not aligned): each
// block forms and stores its words' out and xc, adds its word count of each row it touches to that row's counter, and
// the block that completes a row (old + mine == n / 8) runs the norm over the stored xc and zeroes the counter again.
// No wait anywhere: the counters are atomics, never polled.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include "r4d_common.h"

namespace r4d_ds41pcn {
namespace {

// rad_f32_to_bf16: round to nearest even, a NaN quieted (r4d_ds41_elem.hip ds_bf16)
__device__ __forceinline__ uint16_t pcn_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float pcn_bf(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }

struct PcnArgs {
    const uint16_t* h;  long hp;          // [M, hc * n] the streams before
    const float* post;  long ptp;         // [M, hc]
    const float* comb;  long cbp;         // [M, hc * hc]
    uint16_t* out;      long op;          // [M, hc * n] the streams after (must not alias h)
    const float* pre;   long pp, pst;     // [M, hc] the next sublayer's collapse weights
    uint16_t* xc;       uint32_t x_ld;    // [M, n] the collapsed row
    const void* w;      int wf32;         // [n] the norm's gain, f32 or bf16
    uint16_t* y;        uint32_t y_ld;    // [M, n] the normed row
    uint32_t n, M;
    int hc, thr_v;                        // thr_v = r4d_act_threads(M, n)
    float eps, wadd, mult;
    unsigned* ctr;                        // [M] row counters, zero between launches
};

// out and xc for the eight columns i0 .. i0 + 7 of row m, x[0..7] the row's (reduced) input there.
template <int HCM>
__device__ __forceinline__ void pcn_word(const PcnArgs& A, uint32_t m, uint32_t i0, const uint16_t (&xw)[8]) {
    const int hc = A.hc;
    const float* pr = A.pre + (long)m * A.pp;
    float pj[HCM], pk[HCM], cb[HCM][HCM];
#pragma unroll
    for (int q = 0; q < HCM; ++q) {
        const int kk = q < hc ? q : hc - 1;
        pj[q] = pr[(long)kk * A.pst];
        pk[q] = A.post[(long)m * A.ptp + kk];
#pragma unroll
        for (int j = 0; j < HCM; ++j) cb[j][q] = A.comb[(long)m * A.cbp + (j < hc ? j : hc - 1) * hc + kk];
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const uint32_t i = i0 + (uint32_t)e;
        uint16_t hv[HCM];
#pragma unroll
        for (int j = 0; j < HCM; ++j) hv[j] = A.h[(long)m * A.hp + (long)(j < hc ? j : hc - 1) * A.n + i];
        float v = 0;
#pragma unroll
        for (int q = 0; q < HCM; ++q) {
            if (q >= hc) continue;
            float u = 0;
#pragma unroll
            for (int j = 0; j < HCM; ++j)
                if (j < hc) u += cb[j][q] * bf16_to_f32(hv[j]);
            u += pk[q] * bf16_to_f32(xw[e]);
            const uint16_t o = f32_to_bf16(u);
            A.out[(long)m * A.op + (long)q * A.n + i] = o;
            v += pj[q] * pcn_bf(o);
        }
        A.xc[(size_t)m * A.x_ld + i] = pcn_bf16(v);
    }
}

// The rows this block's words [w0, w1) touch: arrive at each, and finish (norm) the rows it completes. Every thread of
// the block calls it (barriers inside); `nt` the block's threads (a multiple of 32). LDS: vss[1024], wsum[32].
__device__ __forceinline__ void pcn_arrive(const PcnArgs& A, uint32_t w0, uint32_t w1, uint32_t nt, float* vss,
                                           float* wsum, unsigned* s_done) {
    const uint32_t wpr = A.n / 8;
    if (w0 >= w1) return;
    __threadfence();   // this block's out and xc, device-wide, before its arrivals
    __syncthreads();
    const uint32_t m0 = w0 / wpr, m1 = (w1 - 1) / wpr;
    for (uint32_t m = m0; m <= m1; ++m) {
        const uint32_t a0 = max(w0, m * wpr), a1 = min(w1, (m + 1) * wpr), mine = a1 - a0;
        if (threadIdx.x == 0) {
            const unsigned old = __hip_atomic_fetch_add(&A.ctr[m], mine, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
            *s_done = old + mine == wpr ? 1u : 0u;
            if (*s_done) __hip_atomic_store(&A.ctr[m], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        __syncthreads();
        if (*s_done) {
            __threadfence();   // the other blocks' xc words of this row
            const uint16_t* xr = A.xc + (size_t)m * A.x_ld;
            const int tv = A.thr_v;
            for (int v = (int)threadIdx.x; v < tv; v += (int)nt) {
                float ss = 0.0f;
                for (uint32_t i = (uint32_t)v; i < A.n; i += (uint32_t)tv) {
                    const float xv = bf16_to_f32(xr[i]);
                    ss += xv * xv;
                }
                vss[v] = ss;
            }
            __syncthreads();
            const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, nwv = tv / 32;
            for (int vw = wave; vw < nwv; vw += (int)(nt / 32)) {
                const float t = r4d_wave_sum(vss[vw * 32 + lane]);
                if (lane == 0) wsum[vw] = t;
            }
            __syncthreads();
            float sum = 0.0f;
            for (int q = 0; q < nwv; ++q) sum += wsum[q];
            const float sc = rsqrtf(sum / (float)A.n + A.eps);
            uint16_t* yr = A.y + (size_t)m * A.y_ld;
            for (uint32_t i = threadIdx.x; i < A.n; i += nt) {
                const float g = A.wf32 ? ((const float*)A.w)[i] : bf16_to_f32(((const uint16_t*)A.w)[i]);
                yr[i] = f32_to_bf16(bf16_to_f32(xr[i]) * sc * (g + A.wadd) * A.mult);
            }
        }
        __syncthreads();
    }
}

}  // namespace
}  // namespace r4d_ds41pcn
