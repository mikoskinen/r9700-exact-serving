#pragma once
// r4d_ar_coded.h: the coded wire's codec (ds41-int10: moved here VERBATIM from r4d_ar_twoshot_Nrank_exact.hip, where
// r4d_ar_twoshot_coded_body uses it, so tests/r4d_emu can hold the codec itself to the values it carries; nothing in it
// changed). Device code over one wave (32 lanes): r4d_arc_enc / r4d_arc_dec code a 16-byte chunk (8 bf16 values) on its
// group's base exponent, r4d_arc_put / r4d_arc_get move a wave's 32 chunks into / out of a region.
#include <hip/hip_runtime.h>
#include <cstddef>

// ---- the coded form: both pushes carry ~12.2 bits a bf16 value instead of 16, losslessly
//
// The prefill overlap's reductions are link-bound, and a residual stream's bf16 words have few
// distinct exponents: on the 31B (4x R9700, 64 groups of 512 x 5376 values from a 9.9k-token
// prefill) 2.5-3.2 bits of exponent entropy against 8 stored. A chunk is one int4 word (8
// values); 8 consecutive chunks (64 values) form a group that shares a base exponent, and each
// value keeps its sign, its 7 mantissa bits and a 4-bit exponent offset (0 = an exact zero, 1..15 =
// base .. base + 14), 8 values to 3 dwords. A group holding a subnormal, an inf or NaN, or an
// exponent range past 14 (0.02% of groups there) is stored raw in a shadow area, its base byte
// 0xFF. A region (one source's shard in an owner's scratch) is [3 dwords a chunk][a base byte a
// group][the shadow, 16 bytes a chunk]; A and B are two regions in a slot. A wave codes 32
// consecutive chunks: the group's exponents meet by lane shuffles, and the packed dwords are
// redistributed so every store a wave makes is contiguous, keeping the P2P writes coalesced.
//
// Decoding gives back the same bf16 values, so the reduce folds the same numbers in the same rank
// order: the bytes are the plain kernel's. Four-card microbenchmark (the real payload above, the
// two-shot of a 1024-row slice beside the w4a8 prefill GEMM): byte-identical; the reduction 1383
// us at the plain lane launch (2 x 256) against 996 us coded at 4 x 1024, with the GEMM beside it
// 746 -> 724 us. Below 1024 threads a launch codes too slowly (2 x 256: 2492 us).
__device__ __forceinline__ unsigned r4d_arc_shfl(unsigned v, int src) {
  return (unsigned)__builtin_amdgcn_ds_bpermute(src << 2, (int)v);
}
struct R4dArcEnc { unsigned w[3]; unsigned base; };
// Code this lane's chunk; every lane of the wave calls (a lane past the end with valid false).
__device__ __forceinline__ R4dArcEnc r4d_arc_enc(int4 v, bool valid) {
  const unsigned d[4] = {(unsigned)v.x, (unsigned)v.y, (unsigned)v.z, (unsigned)v.w};
  unsigned emin = 255u, emax = 0u, bad = 0u;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const unsigned u = (d[i >> 1] >> ((i & 1) * 16)) & 0xffffu;
    const unsigned e = (u >> 7) & 0xffu, m = u & 0x7fu;
    if (e) { emin = e < emin ? e : emin; emax = e > emax ? e : emax; }
    bad |= (unsigned)(e == 0u && m != 0u) | (unsigned)(e == 255u);
  }
  if (!valid) { emin = 255u; emax = 0u; bad = 0u; }
  const int lane = (int)(threadIdx.x & 31u);
#pragma unroll
  for (int s = 1; s < 8; s <<= 1) {
    const unsigned a = r4d_arc_shfl(emin, lane ^ s), b = r4d_arc_shfl(emax, lane ^ s);
    const unsigned c = r4d_arc_shfl(bad, lane ^ s);
    emin = a < emin ? a : emin; emax = b > emax ? b : emax; bad |= c;
  }
  R4dArcEnc r;
  const unsigned base = emin == 255u ? 1u : emin;
  r.base = (bad || emax > base + 14u) ? 0xFFu : base;
  unsigned c[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const unsigned u = (d[i >> 1] >> ((i & 1) * 16)) & 0xffffu;
    const unsigned e = (u >> 7) & 0xffu, m = u & 0x7fu, sg = u >> 15;
    c[i] = (sg << 11) | (m << 4) | (e ? (e - base + 1u) & 15u : 0u);
  }
  r.w[0] = c[0] | (c[1] << 12) | (c[2] << 24);
  r.w[1] = (c[2] >> 8) | (c[3] << 4) | (c[4] << 16) | (c[5] << 28);
  r.w[2] = (c[5] >> 4) | (c[6] << 8) | (c[7] << 20);
  return r;
}
__device__ __forceinline__ int4 r4d_arc_dec(const unsigned (&w)[3], unsigned base) {
  unsigned c[8];
  c[0] = w[0] & 0xfffu; c[1] = (w[0] >> 12) & 0xfffu; c[2] = (w[0] >> 24) | ((w[1] & 0xfu) << 8);
  c[3] = (w[1] >> 4) & 0xfffu; c[4] = (w[1] >> 16) & 0xfffu; c[5] = (w[1] >> 28) | ((w[2] & 0xffu) << 4);
  c[6] = (w[2] >> 8) & 0xfffu; c[7] = w[2] >> 20;
  unsigned u[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const unsigned off = c[i] & 15u;
    u[i] = ((c[i] >> 11) << 15) | ((off ? base + off - 1u : 0u) << 7) | ((c[i] >> 4) & 0x7fu);
  }
  int4 v;
  v.x = (int)(u[0] | (u[1] << 16)); v.y = (int)(u[2] | (u[3] << 16));
  v.z = (int)(u[4] | (u[5] << 16)); v.w = (int)(u[6] | (u[7] << 16));
  return v;
}
struct R4dArcRegion { unsigned* packed; unsigned* bases; int4* shadow; };
__host__ __device__ constexpr size_t r4d_arc_shadow_off(long SL) {
  return ((size_t)12 * SL + (size_t)SL / 8 + 15) & ~(size_t)15;
}
__host__ __device__ constexpr size_t r4d_arc_region_bytes(long SL) {
  return (r4d_arc_shadow_off(SL) + (size_t)16 * SL + 255) & ~(size_t)255;
}
__device__ __forceinline__ R4dArcRegion r4d_arc_region(const void* p, int SL) {
  char* b = (char*)p;
  return R4dArcRegion{(unsigned*)b, (unsigned*)(b + (size_t)12 * SL), (int4*)(b + r4d_arc_shadow_off(SL))};
}
// A wave's coded chunks [k0, k0 + 32) (k0 a multiple of 32) into ND regions, every store contiguous.
template <int ND>
__device__ __forceinline__ void r4d_arc_put(R4dArcRegion (&dst)[ND], int k0, const R4dArcEnc& e,
                                            int4 raw, bool valid) {
  const int lane = (int)(threadIdx.x & 31u);
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int p = 32 * j + lane, sl = p / 3, wi = p - sl * 3;
    const unsigned a0 = r4d_arc_shfl(e.w[0], sl), a1 = r4d_arc_shfl(e.w[1], sl);
    const unsigned a2 = r4d_arc_shfl(e.w[2], sl), vs = r4d_arc_shfl(valid ? 1u : 0u, sl);
    const unsigned val = wi == 0 ? a0 : wi == 1 ? a1 : a2;
    if (vs) {
#pragma unroll
      for (int q = 0; q < ND; ++q) dst[q].packed[(size_t)k0 * 3 + p] = val;
    }
  }
  const unsigned b0 = r4d_arc_shfl(e.base, 0), b1 = r4d_arc_shfl(e.base, 8);
  const unsigned b2 = r4d_arc_shfl(e.base, 16), b3 = r4d_arc_shfl(e.base, 24);
  if (lane == 0 && valid) {
#pragma unroll
    for (int q = 0; q < ND; ++q) dst[q].bases[k0 / 32] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
  }
  if (valid && e.base == 0xFFu) {
#pragma unroll
    for (int q = 0; q < ND; ++q) dst[q].shadow[k0 + lane] = raw;
  }
}
// A wave's chunks [k0, k0 + 32) back from a region; every lane calls.
__device__ __forceinline__ int4 r4d_arc_get(const R4dArcRegion& src, int k0, bool valid) {
  const int lane = (int)(threadIdx.x & 31u);
  unsigned ld[3];
#pragma unroll
  for (int j = 0; j < 3; ++j) ld[j] = valid ? src.packed[(size_t)k0 * 3 + 32 * j + lane] : 0u;
  unsigned w[3];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const int q = 3 * lane + i, sl = q & 31, j = q >> 5;
    const unsigned a0 = r4d_arc_shfl(ld[0], sl), a1 = r4d_arc_shfl(ld[1], sl);
    const unsigned a2 = r4d_arc_shfl(ld[2], sl);
    w[i] = j == 0 ? a0 : j == 1 ? a1 : a2;
  }
  const unsigned base = ((valid ? src.bases[k0 / 32] : 0u) >> ((lane >> 3) * 8)) & 0xffu;
  if (base == 0xFFu) return valid ? src.shadow[k0 + lane] : make_int4(0, 0, 0, 0);
  return r4d_arc_dec(w, base);
}
