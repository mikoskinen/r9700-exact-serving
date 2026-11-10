// r4d_ds41_qrt.h -- DeepSeek-V4.1's scalar rules for the rope rows and the block-scaled quantise-dequantise
// (quant_roundtrip), one definition for every kernel that forms them:
//
//   r4d_ds41_elem.hip    rope_rows, quant_roundtrip and the fused chains (rmsnorm_rope_roundtrip,
//                        rope_rows_roundtrip, rmsnorm_rope_roundtrip_kv_store)
//   r4d_ds41_sparse.hip  attn_sparse_sink_rope_roundtrip's epilogue (the inverse rope and the round trip
//                        of the attention output in the decode band's kernel)
//
// Moved here from r4d_ds41_elem.hip unchanged (10.11.2026, lever K1), so a fused form calls the very
// functions the separate rows call: the same products and sums (unfused, -ffp-contract=off), the same
// roundings, the same bits. libref's rules: rad_f32_to_bf16 (RNE, a NaN quieted), rad_fp8e4m3_to_f32 /
// rad_f32_to_fp8e4m3, e8m0, std::max / std::clamp as libstdc++ evaluates them, ref_quant_roundtrip's
// block_scale and roundtrip (ref_sparse_kv.cpp).
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

namespace r4d_ds41q {
namespace {

// rad_f32_to_bf16 (abi/rad_plugin.h): round to nearest even, a NaN quieted rather than rounded.
__device__ __forceinline__ uint16_t ds_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float ds_bf(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }

// rad_fp8e4m3_to_f32.
__device__ __forceinline__ float ds_e4m3(uint32_t b) {
    const uint32_t s = (b >> 7) << 31;
    const uint32_t e = (b >> 3) & 0xfu;
    uint32_t m = b & 0x7u, u;
    if (e == 0u) {
        if (m == 0u) {
            u = s;
        } else {
            int sh = 0;
            while (!(m & 0x8u)) { m <<= 1; ++sh; }
            m &= 0x7u;
            u = s | ((uint32_t)(127 - 6 - sh) << 23) | (m << 20);
        }
    } else if (e == 0xfu && m == 0x7u) {
        u = s | 0x7fc00000u;
    } else {
        u = s | ((e - 7u + 127u) << 23) | (m << 20);
    }
    return __uint_as_float(u);
}

// rad_f32_to_fp8e4m3: RNE, saturating at 448, NaN to 0x7f.
__device__ __forceinline__ uint32_t ds_to_e4m3(float f) {
    const uint32_t u = __float_as_uint(f);
    const uint32_t s = (u >> 24) & 0x80u;
    const uint32_t a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return s | 0x7fu;
    if (a >= 0x43e00000u) return s | 0x7eu;
    const int e = (int)(a >> 23) - 127;
    uint32_t m = a & 0x7fffffu;
    if (e < -10) return s;
    if (e < -6) {
        const int sh = -6 - e + 20;
        m |= 0x800000u;
        const uint32_t lo = m & ((1u << sh) - 1u), half = 1u << (sh - 1);
        uint32_t q = m >> sh;
        if (lo > half || (lo == half && (q & 1u))) ++q;
        return s | q;
    }
    uint32_t q = m >> 20;
    const uint32_t lo = m & 0xfffffu;
    int ee = e + 7;
    if (lo > 0x80000u || (lo == 0x80000u && (q & 1u))) ++q;
    if (q == 0x8u) { q = 0; ++ee; }
    if (ee > 15 || (ee == 15 && q >= 7u)) return s | 0x7eu;
    return s | ((uint32_t)ee << 3) | q;
}

// ref e8m0_to_f32: 2^(e-127), 0xff NaN (e 0 is the subnormal 2^-127).
__device__ __forceinline__ float ds_e8m0(uint32_t e) {
    if (e == 0xffu) return __uint_as_float(0x7fc00000u);
    return __uint_as_float(e ? e << 23 : 0x00400000u);
}

// std::max(a, b) and std::clamp as libstdc++ evaluates them (a NaN `b` is dropped by max, kept by
// clamp), which fmaxf/fminf would not reproduce.
__device__ __forceinline__ float ds_max(float a, float b) { return (a < b) ? b : a; }
__device__ __forceinline__ float ds_clamp(float v, float lo, float hi) {
    return v < lo ? lo : (hi < v ? hi : v);
}

/* ==================================================================== quant_roundtrip's rules
 * A block's amax, its scale and the round trip of each value, as ref_quant_roundtrip's
 * block_scale and roundtrip. */

enum { kFp8 = 0, kFp4 = 1 };
enum { kUe8m0 = 0, kE4m3 = 1, kScaleF32 = 2 };

__device__ __forceinline__ int ds_ceil_log2(float v) {
    const uint32_t u = __float_as_uint(v);
    return (int)((u >> 23) & 0xffu) - 127 + ((u & 0x7fffffu) ? 1 : 0);
}
__device__ __forceinline__ float ds_pow2(int e) { return __uint_as_float((uint32_t)(e + 127) << 23); }

__device__ __forceinline__ float block_scale(float amax, int fmt, int sc) {
    if (fmt == kFp8) {
        amax = ds_max(amax, 1e-4f);
        const float inv = 1.0f / 448.0f;
        return sc == kUe8m0 ? ds_pow2(ds_ceil_log2(amax * inv)) : amax * inv;
    }
    if (sc == kE4m3) {
        amax = ds_max(amax, 6.0f * 0x1p-9f);
        return ds_e4m3(ds_to_e4m3(amax / 6.0f));
    }
    amax = ds_max(amax, 6.0f * 0x1p-126f);
    const float inv = 1.0f / 6.0f;
    return sc == kUe8m0 ? ds_pow2(ds_ceil_log2(amax * inv)) : amax * inv;
}

// ref e2m1_code then mxfp4_elem_to_f32: the nearest of the eight magnitudes, ties to the even
// code, the sign kept (so -0 for a small negative), NaN to +0.
__device__ __forceinline__ float ds_e2m1_rt(float v) {
    if (v != v) return 0.0f;
    const bool neg = (__float_as_uint(v) >> 31) != 0;
    const float a = fabsf(v);
    const float mag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    float bm = 0.0f;
#pragma unroll
    for (int c = 1; c < 8; ++c) {
        const float d = fabsf(mag[c] - a), db = fabsf(bm - a);
        if (d < db || (d == db && (c & 1) == 0)) bm = mag[c];
    }
    return neg ? -bm : bm;
}

__device__ __forceinline__ float roundtrip(float v, float scale, int fmt) {
    if (fmt == kFp8) return ds_e4m3(ds_to_e4m3(ds_clamp(v / scale, -448.0f, 448.0f))) * scale;
    return ds_e2m1_rt(ds_clamp(v / scale, -6.0f, 6.0f)) * scale;
}

// A value's magnitude as quant_group_kernel takes it into its block's amax: |v|, a NaN as 0 (std::max(amax,
// NaN) keeps amax, so a NaN never wins). Never negative and never NaN, so the block's fmaxf butterfly is the
// maximum whatever order or grouping forms it (fmaxf returns one of its operands; +0 is the only zero here).
__device__ __forceinline__ float ds_amax_term(float v) {
    const float a = fabsf(v);
    return a == a ? a : 0.0f;
}

// rope_rows' operands and geometry, as rope_rows_body passes them to rope_rows_kernel.
struct FzRope {
    const float* cs;
    long cp, cst;
    const int32_t* pos;
    long ps, tab_rows;
    int hd, half, off, pairs, gptj;
    float sign;
};

}  // namespace
}  // namespace r4d_ds41q
