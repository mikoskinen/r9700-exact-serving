// r4d_gemm_f32_tile.h -- gemm_nt_f32's tile form: one wave computes R rows by C columns of y = x w^T
// (x bf16, w f32 or bf16 widened, y f32). The body of r4d_glm.hip's gemm_nt_f32_tile_kernel, moved here
// unchanged so the DeepSeek-V4.1 router fusion (r4d_ds41_elem.hip router_gemm_scored_scatter_kernel, lever
// FUSE-R, ds41-fuse 13.11.2026) runs the stock GEMM's own code: one definition, so the fused launch and
// the stock launch give every logit the same bytes.
//
// Every output has the same arithmetic: lane l accumulates k = l, l+32, ... in ascending order with fmaf,
// then the xor butterfly over 16, 8, 4, 2, 1. A lane loads U steps of k before their FMAs. Tail rows and
// columns read a clamped valid address and are never stored. The arithmetic of an output depends on
// neither R, C, U nor which wave of which workgroup forms it, so every tile form and every grid gives the
// same bytes (r4d_glm.hip gemm_f32_body's note).
//
// Internal linkage, as r4d_moe_sort.h: each unit compiles its own copy.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

#include "r4d_common.h"

namespace r4d_gf32 {
namespace {

// WT is the weight's storage: float (f32), or uint16_t for bf16 weights (RADIANCE_DS41_ROUTER_BF16,
// a DeepSeek-V4.1 router or compressor stored as the BF16 its checkpoint holds). A bf16 weight is
// widened to the f32 value it stands for (exact) before the same fmaf in the same order, so both
// storages give the same bytes for the same values.
__device__ __forceinline__ float gemm_f32_wval(float v) { return v; }
__device__ __forceinline__ float gemm_f32_wval(uint16_t v) { return bf16_to_f32(v); }

// One wave's tile: rows r0 .. r0+R-1, columns c0 .. c0+C-1 (c0 < n; rows and columns past m / n are
// clamped for the loads and not stored). Lane 0 stores.
template <int R, int C, int U, typename WT>
__device__ __forceinline__ void tile_wave(const uint16_t* x, const WT* w, float* y, long xp, long wp,
                                          long yp, int m, int n, int k, int c0, int r0, int lane) {
    const uint16_t* xr[R];
    const WT* wc[C];
#pragma unroll
    for (int q = 0; q < R; ++q) xr[q] = x + (long)(r0 + q < m ? r0 + q : m - 1) * xp;
#pragma unroll
    for (int c = 0; c < C; ++c) wc[c] = w + (long)(c0 + c < n ? c0 + c : n - 1) * wp;
    float sum[R][C];
#pragma unroll
    for (int q = 0; q < R; ++q)
#pragma unroll
        for (int c = 0; c < C; ++c) sum[q][c] = 0;
    int j = lane;
    for (; j + 32 * (U - 1) < k; j += 32 * U) {
        float weight[U][C];
        uint16_t in[U][R];
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int c = 0; c < C; ++c) weight[u][c] = gemm_f32_wval(wc[c][j + 32 * u]);
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int q = 0; q < R; ++q) in[u][q] = xr[q][j + 32 * u];
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int q = 0; q < R; ++q) {
                const float v = bf16_to_f32(in[u][q]);
#pragma unroll
                for (int c = 0; c < C; ++c) sum[q][c] = fmaf(v, weight[u][c], sum[q][c]);
            }
    }
    for (; j < k; j += 32) {
#pragma unroll
        for (int q = 0; q < R; ++q) {
            const float v = bf16_to_f32(xr[q][j]);
#pragma unroll
            for (int c = 0; c < C; ++c) sum[q][c] = fmaf(v, gemm_f32_wval(wc[c][j]), sum[q][c]);
        }
    }
#pragma unroll
    for (int q = 0; q < R; ++q)
#pragma unroll
        for (int c = 0; c < C; ++c) {
#pragma unroll
            for (int mask = 16; mask; mask >>= 1) sum[q][c] += __shfl_xor(sum[q][c], mask, 32);
        }
    if (lane) return;
#pragma unroll
    for (int q = 0; q < R; ++q)
        if (r0 + q < m)
#pragma unroll
            for (int c = 0; c < C; ++c)
                if (c0 + c < n) y[(long)(r0 + q) * yp + c0 + c] = sum[q][c];
}

}  // namespace
}  // namespace r4d_gf32
