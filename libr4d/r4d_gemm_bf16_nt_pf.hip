// r4d_gemm_bf16_nt_pf.hip -- drafter_bf: the skinny bf16 GEMM's decode band (gemm_bf16_nt_m64 at MB = 1, M <= 64,
// the DSpark drafter's attention / main_proj / shared-expert linears at DeepSeek-V4.1's 2.25 bpw line) in a form
// with the weight fragments of the next KU k steps in flight while the current KU are multiplied, and CT column
// tiles a wave. Every output is the stock kernel's chain to the bit; off unless switched on (r4d_gemm_bf16_nt_pf.h).
//
// Why: the stock wave (r4d_gemm_bf16_nt_m64.hip) owns one 16-wide column tile over one k slice and walks it KU = 8
// steps at a time: the group's sixteen fragment loads, then its eight WMMAs. Nothing of the next group is in flight
// while a group's loads are outstanding, so a wave waits a memory round trip every 8 steps (4 KiB of weight), and
// at the drafter's shapes (N 512..8192, the untuned split giving ~128-512 one-wave blocks) the card sees a few
// waves a SIMD each with one group in flight. Profiled at 1 user: gemm_nt 29.3 us a call, ~1.4 ms a step.
//
// Exactness: an output element of row tile mt and column tile nt is, in the stock kernel, the sum over the SK
// k-splits s = 0..SK-1 (from 0.f, in that order, through LDS) of acc_s, where acc_s is the WMMA chain
// acc = mma(A_frag(mt, step), W_frag(nt, step), acc) from 0.f over the split's steps in ascending order, with the
// fragments loaded by fragU from rowGo's addresses (a row past M or N clamped to the last), then converted with the
// language's float-to-bf16 conversion. This form keeps SK (the caller's, from the same cfg_for), the split's k
// range, the fragments (the same fragU over the same rowGo offsets), each chain's steps in ascending order (a group
// of KU steps, the next group, then the remainder, as the stock KU loop and its tail), the LDS reduction over the
// splits in order from 0.f and the conversion. Only which wave owns which tile and when its loads are issued
// change. r4d_selftest --bfcheck byte-compares it with the stock row at the drafter's shapes, M 1..48.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_bf16_gemm_rows.h"
#include "r4d_gdn_wmma.h"
#include "r4d_emit.h"
#include "r4d_gemm_bf16_nt_pf.h"
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define PF_WAVE32 32

namespace {

typedef __attribute__((address_space(1))) const unsigned long long* pf_gu2;

// The stock kernel's fragment (r4d_gemm_bf16_nt_m64.hip fragU): k step u of the pointer, two 8-byte loads.
__device__ __forceinline__ v8bf pf_frag(pf_gu2 p, int u) {
  union { unsigned long long w[2]; v8bf f; } x;
  x.w[0] = p[4 * u];
  x.w[1] = p[4 * u + 2];
  return x.f;
}

template <int CT, int KU>
__global__ __launch_bounds__(256) void r4d_gemm_bf16_pf_kernel(
    const unsigned short* __restrict__ A, const unsigned short* __restrict__ W,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV, int SK, int LDA, int LDC) {
#ifdef R4D_EMU_HOST
  float* const red = (float*)emu_dyn_lds();
#else
  extern __shared__ float red[];               // WV*CT*SK*256 floats: one 16x16 tile per (column, split)
#endif
  const int tid = threadIdx.x, wave = tid / PF_WAVE32, lane = tid % PF_WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int ntb = (blockIdx.x * WV + col_local) * CT;
  const int mt0 = blockIdx.y;                  // MB = 1: one row tile a block
  const int Ks = K / SK, ksteps = Ks / 16, kbase = ksplit * Ks;

  v8f acc[CT];
#pragma unroll
  for (int c = 0; c < CT; ++c) acc[c] = (v8f)(0.f);

  bool any = false;
  pf_gu2 pw[CT];
#pragma unroll
  for (int c = 0; c < CT; ++c) {
    const int n0 = (ntb + c) * 16;
    any = any || n0 < N;
    const unsigned woff = rowGo((unsigned)K, n0 < N ? n0 : 0, N - 1, lane) + (unsigned)kbase;
    pw[c] = (pf_gu2)(const unsigned long long*)(W + woff);
  }
  const unsigned aoff = rowGo((unsigned)LDA, mt0 * 16, M - 1, lane) + (unsigned)kbase;
  pf_gu2 pa = (pf_gu2)(const unsigned long long*)(A + aoff);

  if (any) {
    // tiles past N keep their accumulator at 0 and are never stored, as the stock `if (n0 < N)` leaves them
    bool ok[CT];
#pragma unroll
    for (int c = 0; c < CT; ++c) ok[c] = (ntb + c) * 16 < N;
    const int ng = ksteps / KU;
    v8bf fa[2][KU], fw[2][KU][CT];
    auto load = [&](int g, v8bf (&a)[KU], v8bf (&w)[KU][CT]) {
      const pf_gu2 qa = pa + (long)g * KU * 4;
#pragma unroll
      for (int u = 0; u < KU; ++u) {
        a[u] = pf_frag(qa, u);
#pragma unroll
        for (int c = 0; c < CT; ++c) w[u][c] = pf_frag(pw[c] + (long)g * KU * 4, u);
      }
    };
    auto mult = [&](const v8bf (&a)[KU], const v8bf (&w)[KU][CT]) {
#pragma unroll
      for (int u = 0; u < KU; ++u)
#pragma unroll
        for (int c = 0; c < CT; ++c)
          if (ok[c]) acc[c] = mma(a[u], w[u][c], acc[c]);
    };
    if (ng > 0) load(0, fa[0], fw[0]);
    for (int g = 0; g < ng; g += 2) {
      if (g + 1 < ng) load(g + 1, fa[1], fw[1]);
      mult(fa[0], fw[0]);
      if (g + 1 < ng) {
        if (g + 2 < ng) load(g + 2, fa[0], fw[0]);
        mult(fa[1], fw[1]);
      }
    }
    for (int s = ng * KU; s < ksteps; ++s) {
      const v8bf a = pf_frag(pa + (long)s * 4, 0);
#pragma unroll
      for (int c = 0; c < CT; ++c)
        if (ok[c]) acc[c] = mma(a, pf_frag(pw[c] + (long)s * 4, 0), acc[c]);
    }
  }

  // The stock reduction: each (column, split) tile through LDS, the splits summed in order from 0.f.
  const int nthreads = WV * SK * PF_WAVE32;
  __syncthreads();
#pragma unroll
  for (int c = 0; c < CT; ++c) {
    const int slot = ((col_local * CT + c) * SK + ksplit) * 256;
#pragma unroll
    for (int e = 0; e < 8; e++) red[slot + e * PF_WAVE32 + lane] = acc[c][e];
  }
  __syncthreads();
  for (int idx = tid; idx < WV * CT * 256; idx += nthreads) {
    const int col = idx >> 8, off = idx & 255;
    float v = 0.f;
    for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 256 + off];
    const int l = off & 31;
    const int n = (blockIdx.x * WV * CT + col) * 16 + (l & 15);
    const int m = mt0 * 16 + 8 * (l >> 4) + (off >> 5);
    if (n < N && m < M) C[(size_t)m * LDC + n] = (__hip_bfloat16)v;
  }
}

#define BF_FORMS(X) X(1, 4) X(1, 8) X(2, 4) X(2, 8) X(4, 4)

bool bf_has(int CT, int KU) {
#define BF_H(ct, ku) if (CT == ct && KU == ku) return true;
  BF_FORMS(BF_H)
#undef BF_H
  return false;
}

int bf_env() {
  const char* v = std::getenv("RADIANCE_DS41_DRAFTER_BF");
  const int on = (v && *v == '1' && !v[1]) ? 1 : 0;
  if (v && *v && !on && std::strcmp(v, "0"))
    std::fprintf(stderr, "radiance: libr4d: RADIANCE_DS41_DRAFTER_BF=%s is not 0 / 1: the stock skinny bf16 GEMM\n", v);
  if (on)
    std::fprintf(stderr, "radiance: libr4d: gemm_nt bf16's decode band takes the prefetching form (RADIANCE_DS41_DRAFTER_BF=1, "
                         "the fn4 switch drafter_bf)\n");
  return on;
}
int g_bf_on = bf_env();
int g_bf_pin = 0, g_bf_pin_ct = 1, g_bf_pin_ku = 8, g_bf_pin_wv = 0;
char g_bf_last[128];

}  // namespace

int r4d_bf_pf_on() { return g_bf_pin != 0 ? 1 : g_bf_on; }
void r4d_bf_pf_set(int on) { g_bf_on = on ? 1 : 0; }
int r4d_bf_pf_get() { return g_bf_on; }

// The rule: two column tiles a wave where N gives each wave of the stock grid a partner tile (N >= 4096: the
// q_b / o_b / down shapes), else one; KU 8; WV so a block is at most 8 waves. SK is the caller's.
int r4d_bf_pf_gemm(RadStream s, const R4dBfPfCall& c) {
  if (c.M < 1 || c.M > 64 || c.K < 16 || c.K % 16 || c.N < 1 || c.SK < 1 || c.K % (c.SK * 16)) return R4D_BF_PF_DECLINE;
  if (c.lda < c.K || c.ldc < c.N || c.lda % 4 || (long long)(c.M - 1) * c.lda + c.K >= (1ll << 32) || c.ldc > INT32_MAX)
    return R4D_BF_PF_DECLINE;
  if (!r4d_bf16_gemm_chunk_rows(c.M, c.N, c.K, 16, 16)) return R4D_BF_PF_DECLINE;
  const bool pinned = g_bf_pin > 0;
  int CT = c.N >= 4096 ? 2 : 1, KU = 8, WV = 0;
  if (pinned) { CT = g_bf_pin_ct; KU = g_bf_pin_ku; WV = g_bf_pin_wv; }
  if (!bf_has(CT, KU)) return R4D_BF_PF_DECLINE;
  if (WV <= 0) WV = 8 / c.SK > 0 ? 8 / c.SK : 1;
  if (WV < 1 || (long long)WV * c.SK * PF_WAVE32 > 256) return R4D_BF_PF_DECLINE;   // the kernel's launch bound
  const int ntiles = (c.N + 15) / 16, mtiles = (c.M + 15) / 16;
  const long long gx = (ntiles + (long long)WV * CT - 1) / ((long long)WV * CT);
  if (gx > 0x7fffffffLL || mtiles > 65535) return R4D_BF_PF_DECLINE;
  const dim3 grid((unsigned)gx, (unsigned)mtiles), block((unsigned)(WV * c.SK * PF_WAVE32));
  const size_t smem = (size_t)WV * CT * c.SK * 256 * sizeof(float);
  if (smem > 65536) return R4D_BF_PF_DECLINE;
  std::snprintf(g_bf_last, sizeof g_bf_last, "CT%d KU%d WV%d SK%d grid %lldx%d", CT, KU, WV, c.SK, gx, mtiles);
#define BF_L(ct, ku)                                                                                        \
  if (CT == ct && KU == ku)                                                                                 \
    return R4D_EMIT(s, (r4d_gemm_bf16_pf_kernel<ct, ku>), grid, block, (unsigned)smem,                      \
                    (const unsigned short*)c.a, (const unsigned short*)c.w, (__hip_bfloat16*)c.y, c.M, c.K, c.N, \
                    WV, c.SK, (int)c.lda, (int)c.ldc);
  BF_FORMS(BF_L)
#undef BF_L
  return R4D_BF_PF_DECLINE;
}

// r4d_bf_pf_pin (bench / test hook, r4d_selftest --bfcheck / --perfbf): on 0 clears the pin, -1 forces the rule,
// 1 pins CT column tiles a wave, KU steps a group and WV column groups a block (0: the rule's).
extern "C" void r4d_bf_pf_pin(int on, int ct, int ku, int wv) {
  g_bf_pin = on == 1 || on == -1 ? on : 0;
  if (on == 1) { g_bf_pin_ct = ct; g_bf_pin_ku = ku; g_bf_pin_wv = wv; }
}
extern "C" const char* r4d_bf_pf_last() { return g_bf_last; }
