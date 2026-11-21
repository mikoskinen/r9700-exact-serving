#pragma once
// r4d_ar_fp8.h: lever DAR b (DRAFTER track 21.11.2026 round 58) -- the wide two-shot all-reduce on an FP8 WIRE, for the
// DSpark drafter's reductions only (its proposals move; the trunk's verify keeps the served text). NOT exact: a value
// crosses the wire as E4M3 with a per-word E8M0 scale. Deterministic and BATCH-INVARIANT by construction (rule B):
//
//   * a codec group = one 16-byte word of the message = 8 consecutive bf16 values of ONE row (a 5120-wide row is 640
//     words), so a group never mixes rows and its content does not depend on the message size or the shard table;
//   * EVERY contribution is quantised, the owner's own too: which rank owns a word moves with the message size, so an
//     unquantised own part would make a row's sum depend on how many rows travel with it;
//   * the owner folds the 4 decoded contributions in fp32, ascending rank from 0.0f (the exact wire's order), then
//     quantises the sum once more; its own out is dequant(quant(sum)) like every peer's, so all ranks hold the same bits.
//
// Wire format of a word: 8 E4M3 bytes (OCP E4M3FN: bias 7, subnormals m x 2^-9, 0x7F / 0xFF NaN, largest finite 448)
// in an int2 + one E8M0 byte = 2^s (s + 127; 0xFF = the group held an inf / NaN -> 8 NaNs). s is the smallest power
// that keeps every |v| / 2^s <= 448 (from the group's largest magnitude: its f32 exponent - 8, + 1 when its mantissa is
// past 1.75), floored at -117 so every decoded non-zero value is a normal f32. A decoded value e4m3 x 2^s has <= 4
// significant bits: exactly a bf16, so the bf16 store rounds nothing. 9 bytes a word instead of 16 (56.25 %).
//
// Device codec = the hardware's own conversions (v_cvt_pk_fp8_f32 RNE, v_cvt_pk_f32_fp8, v_ldexp_f32); the host codec
// arf_sw_* is the same function written in integer arithmetic (the reference and the emulator's form): tools/dbench/
// arf.cpp checks the two byte for byte (every bf16 pattern at several scales, random and edge f32 groups, every code x
// scale), then the all-reduce on 4 ranks against the host fold.
//
// Region of one source in an owner's slot (bytes): A = data int2 [sh_max] at 0, scales u8 [sh_max] at 8 sh_max; B at
// arf_rb(sh_max) = round_up_16(9 sh_max), same layout. Both fit wherever the exact wire's 2 x 16 sh_max fit.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstring>
#include <cmath>

#define R4D_ARF_SMIN (-117)

// s from the largest magnitude's f32 bits (sign cleared); 128 = inf / NaN in the group
__host__ __device__ __forceinline__ int r4d_arf_scale_exp(unsigned amax_bits) {
  if (amax_bits >= 0x7F800000u) return 128;
  int s = (int)(amax_bits >> 23) - 135 + ((amax_bits & 0x7FFFFFu) > 0x600000u ? 1 : 0);
  return s < R4D_ARF_SMIN ? R4D_ARF_SMIN : s;
}
__host__ __device__ __forceinline__ long r4d_arf_rb(long sh_max) { return (9 * sh_max + 15) / 16 * 16; }
// bytes a slot needs for both regions of the longest shard
__host__ __device__ __forceinline__ long r4d_arf_slot_bytes(long sh_max) { return 2 * r4d_arf_rb(sh_max); }

// the gather form's host launchers (r4d_ar_twoshot_Nrank_exact.hip; the wide form's are in r4d.h)
int r4d_ar_gather_twoshot_fp8_ok(long M, long n, long top_k, long slot_stride16);
void r4d_ar_gather_twoshot_4rank_fp8(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                     long peer_flags0, long peer_flags1, long peer_flags2,
                                     long my_scratch, long my_flags, long seq_ctrs, long slot_stride16,
                                     long y, long M, long n, long ye, long ew, long sorted, long sh,
                                     long sg, long ye_ld, long ew_ld, long sh_ld, long sg_ld,
                                     long top_k, long sig, long rank, long stream, long nblocks,
                                     long nthreads, long drain, long acq, long pub, long gate,
                                     long gate_ctr);

// ---- host codec (the reference): the same bits as the device's
static inline unsigned char r4d_arf_sw_e4m3(float x) {   // RNE, |x| <= 448 or NaN
  unsigned u; std::memcpy(&u, &x, 4);
  const unsigned sg = (u >> 24) & 0x80u, a = u & 0x7FFFFFFFu;
  if (a > 0x7F800000u) return (unsigned char)(sg | 0x7Fu);
  const int eb = (int)(a >> 23);
  const unsigned mant = a & 0x7FFFFFu;
  if (eb - 127 >= -6) {
    unsigned r = mant >> 20, rem = mant & 0xFFFFFu;
    if (rem > 0x80000u || (rem == 0x80000u && (r & 1u))) ++r;
    unsigned E = (unsigned)(eb - 127 + 7);
    if (r == 8u) { r = 0u; ++E; }
    if (E > 15u || (E == 15u && r == 7u)) return (unsigned char)(sg | 0x7Fu);   // past 448: not produced here
    return (unsigned char)(sg | (E << 3) | r);
  }
  // E4M3 subnormal: m = RNE(x * 2^9); x = M * 2^(e - 23)
  const unsigned M = eb ? (mant | 0x800000u) : mant;
  const int e = eb ? eb - 127 : -126;
  const int sh = 14 - e;   // >= 21
  unsigned m = 0u;
  if (sh < 32) {
    m = M >> sh;
    const unsigned rem = M & ((1u << sh) - 1u), half = 1u << (sh - 1);
    if (rem > half || (rem == half && (m & 1u))) ++m;
  }
  return (unsigned char)(sg | m);   // m == 8 is 0x08 = 2^-6, the smallest normal: right
}
static inline float r4d_arf_sw_f32(unsigned char c) {
  const unsigned E = (c >> 3) & 15u, m = c & 7u;
  if (E == 15u && m == 7u) { const unsigned n = 0x7FC00000u | ((unsigned)(c & 0x80u) << 24); float f; std::memcpy(&f, &n, 4); return f; }
  const float v = E ? std::ldexp((float)(8u + m), (int)E - 10) : std::ldexp((float)m, -9);
  return (c & 0x80u) ? -v : v;
}
// one word: 8 values (f32; a bf16 word passes its exact f32 widenings) -> 8 codes + the scale byte
static inline void r4d_arf_sw_enc8(const float v[8], unsigned char code[8], unsigned char& sc) {
  unsigned amax = 0u;
  for (int i = 0; i < 8; ++i) { unsigned u; std::memcpy(&u, &v[i], 4); u &= 0x7FFFFFFFu; amax = u > amax ? u : amax; }
  const int s = r4d_arf_scale_exp(amax);
  if (s == 128) { sc = 0xFFu; for (int i = 0; i < 8; ++i) code[i] = 0u; return; }
  sc = (unsigned char)(s + 127);
  for (int i = 0; i < 8; ++i) code[i] = r4d_arf_sw_e4m3(std::ldexp(v[i], -s));
}
static inline void r4d_arf_sw_dec8(const unsigned char code[8], unsigned char sc, float v[8]) {
  if (sc == 0xFFu) { const unsigned n = 0x7FC00000u; float f; std::memcpy(&f, &n, 4); for (int i = 0; i < 8; ++i) v[i] = f; return; }
  const int s = (int)sc - 127;
  for (int i = 0; i < 8; ++i) v[i] = std::ldexp(r4d_arf_sw_f32(code[i]), s);
}

#if defined(__HIP__)
// ---- device codec
__device__ __forceinline__ void r4d_arf_enc8(const float v[8], int2& d, unsigned& sc) {
  unsigned amax = 0u;
#pragma unroll
  for (int i = 0; i < 8; ++i) { const unsigned u = __float_as_uint(v[i]) & 0x7FFFFFFFu; amax = u > amax ? u : amax; }
  const int s = r4d_arf_scale_exp(amax);
  if (s == 128) { sc = 0xFFu; d = make_int2(0, 0); return; }
  sc = (unsigned)(s + 127);
  float q[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) q[i] = __builtin_amdgcn_ldexpf(v[i], -s);
  int lo = __builtin_amdgcn_cvt_pk_fp8_f32(q[0], q[1], 0, false);
  lo = __builtin_amdgcn_cvt_pk_fp8_f32(q[2], q[3], lo, true);
  int hi = __builtin_amdgcn_cvt_pk_fp8_f32(q[4], q[5], 0, false);
  hi = __builtin_amdgcn_cvt_pk_fp8_f32(q[6], q[7], hi, true);
  d = make_int2(lo, hi);
}
__device__ __forceinline__ void r4d_arf_dec8(int2 d, unsigned sc, float v[8]) {
  if (sc == 0xFFu) {
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __uint_as_float(0x7FC00000u);
    return;
  }
  const int s = (int)sc - 127;
  const int w[2] = {d.x, d.y};
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const auto p0 = __builtin_amdgcn_cvt_pk_f32_fp8(w[h], false);
    const auto p1 = __builtin_amdgcn_cvt_pk_f32_fp8(w[h], true);
    v[4 * h + 0] = __builtin_amdgcn_ldexpf(p0[0], s);
    v[4 * h + 1] = __builtin_amdgcn_ldexpf(p0[1], s);
    v[4 * h + 2] = __builtin_amdgcn_ldexpf(p1[0], s);
    v[4 * h + 3] = __builtin_amdgcn_ldexpf(p1[1], s);
  }
}
__device__ __forceinline__ void r4d_arf_bf16x8(int4 w, float v[8]) {
  const unsigned u[4] = {(unsigned)w.x, (unsigned)w.y, (unsigned)w.z, (unsigned)w.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) { v[2 * i] = __uint_as_float(u[i] << 16); v[2 * i + 1] = __uint_as_float(u[i] & 0xFFFF0000u); }
}
// a decoded word to bf16 (exact: <= 4 significant bits; NaN stays NaN)
__device__ __forceinline__ int4 r4d_arf_to_bf16x8(const float v[8]) {
  unsigned u[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) u[i] = (__float_as_uint(v[2 * i]) >> 16) | (__float_as_uint(v[2 * i + 1]) & 0xFFFF0000u);
  return make_int4((int)u[0], (int)u[1], (int)u[2], (int)u[3]);
}

#ifdef R4D_ARF_BODY   // the body needs the wide family's transport (r4d_ar_wide.h) and R4DArShardOff
// The fp8 two-shot at WS ranks, lockstep, one piece (decode-sized messages); the exact wide body's structure
// (r4d_ar_twoshot_Nrank_exact.hip): push1 -> handshake 2s-1 -> reduce fused with the phase-2 push -> handshake 2s ->
// assemble. In place is safe for the same reason: word k is read and written by one thread in all three steps.
// HS false (tools/dbench/arf.cpp --phase only): the same loops without the sequence take and the handshakes, slot 0 --
// one rank's codec + data path timed alone; the wire's form is HS true.
template <int WS, bool HS = true>
__device__ __forceinline__ void r4d_ar_twoshot_fp8_body(
    const int4* __restrict__ in4, R4DArWidePeers<int4, WS - 1> peers, const int4* __restrict__ my_base,
    unsigned int* __restrict__ my_flags, int4* __restrict__ out4, unsigned int* __restrict__ seq_ctrs,
    R4DArShardOff<WS> sh, int slot_stride16, int rank, int drain, int acq, int pub, long long* gate,
    unsigned int* gate_ctr) {
  constexpr int NPEERS = WS - 1;
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  const long long gate_word = r4d_ar_link_word(gate, (unsigned int)tid);
  if (HS) r4d_ar_seq_take(seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot);
  const unsigned int v0 = HS ? 2u * s_seq - 1u : 0u;
  const int parity = HS ? (int)s_slot : 0;
  const int sh_max = sh.sh_max;
  const long RB = r4d_arf_rb(sh_max);
  const int w = (sh_max + nb - 1) / nb;
  const int w0 = b * w;
  int w1 = w0 + w; if (w1 > sh_max) w1 = sh_max;
  const size_t slot_b = (size_t)slot_stride16 * 16u;

  unsigned char* pa[NPEERS];   // region A of my slot in every peer's scratch
#pragma unroll
  for (int p = 0; p < NPEERS; ++p)
    pa[p] = reinterpret_cast<unsigned char*>(peers.scratch[p]) + ((size_t)parity * WS + rank) * slot_b;
  const unsigned char* mb = reinterpret_cast<const unsigned char*>(my_base) + (size_t)parity * WS * slot_b;

  // phase 1: my contribution to each peer-owned shard, coded, into the owner's region A
  for (int i = 0; i < NPEERS; ++i) {
    const int owner = peers.rank[i];
    const int4* s1 = in4 + (size_t)sh.off[owner];
    int2* dd = reinterpret_cast<int2*>(pa[i]);
    unsigned char* ds = pa[i] + (size_t)8 * sh_max;
    const int len = sh.off[owner + 1] - sh.off[owner];
    const int lim = w1 < len ? w1 : len;
    for (int k = w0 + tid; k < lim; k += nt) {
      float v[8]; r4d_arf_bf16x8(s1[k], v);
      int2 d; unsigned sc; r4d_arf_enc8(v, d, sc);
      dd[k] = d; ds[k] = (unsigned char)sc;
    }
  }
  if (HS) r4d_ar_handshake<WS>(peers, my_flags, b, rank, v0, drain, acq, pub);

  // reduce my shard: every contribution decoded (mine coded and decoded here), fp32 ascending rank, coded once more
  {
    const int4* mine = in4 + (size_t)sh.off[rank];
    int4* my_out = out4 + (size_t)sh.off[rank];
    const int my_len = sh.off[rank + 1] - sh.off[rank];
    const int my_lim = w1 < my_len ? w1 : my_len;
    for (int k = w0 + tid; k < my_lim; k += nt) {
      float acc[8];
#pragma unroll
      for (int l = 0; l < 8; ++l) acc[l] = 0.0f;
#pragma unroll
      for (int r = 0; r < WS; ++r) {
        int2 d; unsigned sc;
        if (r == rank) {
          float v[8]; r4d_arf_bf16x8(mine[k], v);
          r4d_arf_enc8(v, d, sc);
        } else {
          const unsigned char* ra = mb + (size_t)r * slot_b;
          d = reinterpret_cast<const int2*>(ra)[k];
          sc = ra[(size_t)8 * sh_max + k];
        }
        float v[8]; r4d_arf_dec8(d, sc, v);
#pragma unroll
        for (int l = 0; l < 8; ++l) acc[l] += v[l];
      }
      int2 d; unsigned sc; r4d_arf_enc8(acc, d, sc);
      float o[8]; r4d_arf_dec8(d, sc, o);
      my_out[k] = r4d_arf_to_bf16x8(o);
#pragma unroll
      for (int p = 0; p < NPEERS; ++p) {
        reinterpret_cast<int2*>(pa[p] + RB)[k] = d;
        pa[p][RB + (size_t)8 * sh_max + k] = (unsigned char)sc;
      }
    }
  }
  if (HS) r4d_ar_handshake<WS>(peers, my_flags, b, rank, v0 + 1u, drain, acq, pub);
  else __syncthreads();
  r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nb, (unsigned int)tid);

  // assemble: every peer-owned shard from my region B
  for (int i = 0; i < NPEERS; ++i) {
    const int owner = peers.rank[i];
    const unsigned char* rb = mb + (size_t)owner * slot_b + RB;
    int4* dst = out4 + (size_t)sh.off[owner];
    const int len = sh.off[owner + 1] - sh.off[owner];
    const int lim = w1 < len ? w1 : len;
    for (int k = w0 + tid; k < lim; k += nt) {
      float o[8]; r4d_arf_dec8(reinterpret_cast<const int2*>(rb)[k], rb[(size_t)8 * sh_max + k], o);
      dst[k] = r4d_arf_to_bf16x8(o);
    }
  }
}
#endif  // R4D_ARF_BODY
#endif  // device
