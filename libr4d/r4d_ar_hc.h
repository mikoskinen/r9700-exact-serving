#pragma once
// r4d_ar_hc.h: per-word arithmetic of the all-reduces fused with the gated residual write: the MoE
// gather before the message and the hc write after the reduction. Shared by the exact kernels
// (r4d_ar_oneshot_2rank_exact.hip) and the rotated 6-bit ones (r4d_ar_oneshot_2rank_wht6.hip), so
// both fused forms produce the same bytes as the unfused chain moe_gather, all-reduce, hc_write.
//
// A word is eight consecutive bf16 elements of one row. A row is `n` elements and n is a multiple
// of eight, so a word never straddles a row.
#include <hip/hip_runtime.h>
#include "r4d_gdn_wmma.h"   // f2bf / bf2f: the gated residual's own bf16 arithmetic

// LDS slot table bounds: rows per block, and slots per row.
#define R4D_ARG_ROWS  8
#define R4D_ARG_SLOTS 64

__device__ __forceinline__ unsigned short arg_bf16_rne(float f) {
  const unsigned u = __builtin_bit_cast(unsigned, f);
  return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

struct R4dArGather {
  const unsigned short* ye;   // [T, ye_ld] the routed experts' outputs, sorted-row order
  const unsigned short* ew;   // [M, ew_ld] each token's slot weights
  const int* sorted;          // [T] the (token * top_k + slot) each sorted row holds
  const unsigned short* sh;   // [M, sh_ld] the shared arm, or null
  const unsigned short* sg;   // [M, sg_ld] its gate, or null
  long ye_ld, ew_ld, sh_ld, sg_ld;
  int top_k, T, sig;
  int ew_f32;                 // ew holds f32 weights (moe_gather's F32 form), else bf16
};

// A slot weight as moe_gather reads it: f32 as stored, or bf16 widened.
__device__ __forceinline__ float r4d_ar_gather_w(const R4dArGather& G, long i) {
  return G.ew_f32 ? reinterpret_cast<const float*>(G.ew)[i] : bf2f(G.ew[i]);
}

// One word of the gather: y[row][i0 .. i0+8) = sum over the row's slots, ascending, of
// w[row][j] * y_expert[slot][i0 ..], rounded to bf16, with the shared arm folded in as moe_gather
// does it. Its narrows are round-to-nearest-even, unlike hc_write's.
//
// With INV, slots come from the inverse table inv[token * top_k + slot] (the sorted row holding
// it, or -1) and weights and gate from global memory. Without it they come from the block's LDS
// table, where `r` is the row's index.
template <bool INV>
__device__ __forceinline__ int4 r4d_ar_gather_word(const R4dArGather& G,
                                                   const int* __restrict__ inv, const int* s_i,
                                                   const float* s_w, const float* s_g, int row,
                                                   int r, int i0) {
  const int K = G.top_k;
  float acc[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
  /* Four slot rows in flight at once: all four loads issue before any add, so they do not
   * serialise behind the empty-slot test. An empty slot reads row 0, which always exists, and is
   * skipped in the ascending add. */
  constexpr int kU = 4;
  for (int j0 = 0; j0 < K; j0 += kU) {
    int4 v[kU];
    int ii[kU];
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      ii[u] = j0 + u < K ? (INV ? inv[(long)row * K + j0 + u] : s_i[r * K + j0 + u]) : -1;
      v[u] = *reinterpret_cast<const int4*>(G.ye + (long)(ii[u] < 0 ? 0 : ii[u]) * G.ye_ld + i0);
    }
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      if (ii[u] < 0) continue;
      const float wj = INV ? r4d_ar_gather_w(G, (long)row * G.ew_ld + j0 + u) : s_w[r * K + j0 + u];
      const unsigned short* ve = reinterpret_cast<const unsigned short*>(&v[u]);
      /* moe_gather's two forms: f32 weights fused (fmaf), bf16 weights multiplied then added. */
      if (G.ew_f32) {
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] = fmaf(wj, bf2f(ve[e]), acc[e]);
      } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] += wj * bf2f(ve[e]);
      }
    }
  }
  int4 out;
  unsigned short* oe = reinterpret_cast<unsigned short*>(&out);
  if (!G.sh) {
#pragma unroll
    for (int e = 0; e < 8; ++e) oe[e] = arg_bf16_rne(acc[e]);
  } else {
    float gate;
    if (INV) {
      const float v0 = bf2f(G.sg[(long)row * G.sg_ld]);
      gate = G.sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
    } else {
      gate = s_g[r];
    }
    const unsigned short* shp = G.sh + (long)row * G.sh_ld + i0;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const unsigned short g = arg_bf16_rne(acc[e]);
      const unsigned short pr = arg_bf16_rne(bf2f(shp[e]) * gate);
      oe[e] = arg_bf16_rne(bf2f(pr) + bf2f(g));
    }
  }
  return out;
}

// ---- The slot LIST: every row's live slots once, for a block that gathers words of ANY row (the
// four-rank two-shot gather, r4d_ar_twoshot_Nrank_exact.hip: a block's words lie in every rank's
// shard, so rows all over the message). Bounds of the block's LDS table: rows, and rows x top_k.
#define R4D_ARGL_ROWS 256
#define R4D_ARGL_TAB  2048

// Fills the block's table for the M = T / top_k rows of the call: s_n[r] is row r's live slot
// count, s_i[r * K + c] / s_w[r * K + c] (c < s_n[r]) its sorted rows and weights in ASCENDING slot
// order, s_g[r] its shared gate (with the shared arm). The same slots, weights and gate
// r4d_ar_gather_word reads from its table (a slot no sorted row claims skipped, the weight as
// r4d_ar_gather_w reads it, the gate's sigmoid the same expression); only the dead slots are left
// out, which that loop skips. Every thread of the block calls it; it ends on a barrier.
__device__ __forceinline__ void r4d_ar_gather_list(const R4dArGather& G, int* s_i, float* s_w,
                                                   int* s_n, float* s_g) {
  const int K = G.top_k, T = G.T, M = T / K;
  const int tid = threadIdx.x, nt = blockDim.x;
  for (int x = tid; x < T; x += nt) s_i[x] = -1;
  __syncthreads();
  for (int i = tid; i < T; i += nt) {
    const int f = G.sorted[i];
    if (f >= 0 && f < T) s_i[f] = i;
  }
  __syncthreads();
  /* A thread a row, compacting in place: entry c is written only after entry j >= c was read. */
  for (int r = tid; r < M; r += nt) {
    int c = 0;
    for (int j = 0; j < K; ++j) {
      const int si = s_i[r * K + j];
      if (si < 0) continue;
      s_i[r * K + c] = si;
      s_w[r * K + c] = r4d_ar_gather_w(G, (long)r * G.ew_ld + j);
      ++c;
    }
    s_n[r] = c;
    if (G.sh) {
      const float v0 = bf2f(G.sg[(long)r * G.sg_ld]);
      s_g[r] = G.sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
    }
  }
  __syncthreads();
}

// One word of the gather over row `row`'s list (li / lw: its s_i / s_w entries, cnt of them):
// r4d_ar_gather_word's arithmetic step for step -- acc from 0, each live slot in ascending slot
// order (fmaf with f32 weights, multiply then add with bf16 ones; libr4d compiles without FMA
// contraction), each narrow round-to-nearest-even, the shared arm's product rounded before its
// add -- with only the live slots loaded (that loop loads a dead slot's row 0 and skips it).
__device__ __forceinline__ int4 r4d_ar_gather_word_list(const R4dArGather& G, const int* li,
                                                        const float* lw, int cnt, float gate,
                                                        int row, int i0) {
  float acc[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
  constexpr int kU = 4;
  int4 zero;
  zero.x = zero.y = zero.z = zero.w = 0;
  for (int c0 = 0; c0 < cnt; c0 += kU) {
    int4 v[kU];
#pragma unroll
    for (int u = 0; u < kU; ++u)
      v[u] = c0 + u < cnt ? *reinterpret_cast<const int4*>(G.ye + (long)li[c0 + u] * G.ye_ld + i0)
                          : zero;
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      if (c0 + u >= cnt) continue;
      const float wj = lw[c0 + u];
      const unsigned short* ve = reinterpret_cast<const unsigned short*>(&v[u]);
      if (G.ew_f32) {
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] = fmaf(wj, bf2f(ve[e]), acc[e]);
      } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] += wj * bf2f(ve[e]);
      }
    }
  }
  int4 out;
  unsigned short* oe = reinterpret_cast<unsigned short*>(&out);
  if (!G.sh) {
#pragma unroll
    for (int e = 0; e < 8; ++e) oe[e] = arg_bf16_rne(acc[e]);
  } else {
    const unsigned short* shp = G.sh + (long)row * G.sh_ld + i0;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const unsigned short g = arg_bf16_rne(acc[e]);
      const unsigned short pr = arg_bf16_rne(bf2f(shp[e]) * gate);
      oe[e] = arg_bf16_rne(bf2f(pr) + bf2f(g));
    }
  }
  return out;
}

// One word of the write: h[row][c*n + i0 ..] += inj[row][c] * y[row][i0 ..] for every stream c, in
// hc_write's arithmetic (r4d_hc_bf16.hip). `yv` is the reduced word as stored bf16, which is what
// hc_write reads back from memory.
__device__ __forceinline__ void r4d_ar_hc_write_word(const unsigned short* __restrict__ inj,
                                                     unsigned short* __restrict__ h, int row,
                                                     int i0, int n, int hc, long irow, long hrow,
                                                     const unsigned short* yv) {
  float v[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = bf2f(yv[j]);
  for (int c = 0; c < hc; ++c) {
    const float g = bf2f(inj[(long)row * irow + c]);
    int4* hp = reinterpret_cast<int4*>(h + (long)row * hrow + (long)c * n + i0);
    int4 hv = *hp;
    unsigned short* he = reinterpret_cast<unsigned short*>(&hv);
#pragma unroll
    for (int j = 0; j < 8; ++j) he[j] = f2bf(bf2f(he[j]) + g * v[j]);
    *hp = hv;
  }
}

// ---- The SLOT-ORDER sum (moe_gather_all_reduce_slot, lever D3s; r4d_ar_slotsum.hip).
//
// moe_gather_all_reduce and every form above sum a token's slots ON ITS RANK, round that partial
// (with the rank's shared-arm product) to bf16 and fold the ranks' partials in ascending rank
// order: moving an expert to another rank moves its term into another partial, so the bytes
// depend on where each expert sits. The slot-order sum does not: for every element,
//
//   acc = 0
//   acc += H_r            for every rank r ascending   (H_r = bf16(shared_r * gate), the rank's
//                                                       width slice of the shared arm, moe_gather's
//                                                       narrow; absent without a shared arm)
//   acc = fmaf(w_j, S_j, acc)  for every slot j ascending (S_j = the slot's routed-expert row as the
//                                                       expert GEMM stored it; with bf16 weights
//                                                       acc + w_j * S_j, no contraction)
//   y = bf16(acc)         rounded once, nearest-even
//
// S_j is the same bytes whichever rank computed the slot (the expert's GEMM row depends on that
// token and that expert only), and the H_r come from fixed width slices, so the sum is the same
// whoever holds which expert: expert placement, rebalancing and replication cannot move a byte.
// Every slot must be held by exactly one rank (the placement is total; there are no dropped slots).
//
// Transport: every rank writes its own slots' rows (and its H) into PLANES of the reducing
// rank's scratch, plane j for slot j and plane K + r for rank r's H, each a run of the reduced
// words, so a slot's word lands at the same place whichever rank sends it and the reducer needs no
// placement to find it.

// The block's slot table over every row of the call: s_i[r * K + j] = the sorted row holding
// slot j of row r ON THIS RANK, -1 where another rank holds it; s_w[r * K + j] every slot's weight
// (the replicated router's, so the same on every rank); s_g[r] the shared gate (with the arm).
// M <= R4D_ARGL_ROWS, M * K <= R4D_ARGL_TAB. Every thread calls it; it ends on a barrier.
__device__ __forceinline__ void r4d_ar_slot_table(const R4dArGather& G, int* s_i, float* s_w,
                                                  float* s_g) {
  const int K = G.top_k, T = G.T, M = T / K;
  const int tid = threadIdx.x, nt = blockDim.x;
  for (int x = tid; x < T; x += nt) {
    s_i[x] = -1;
    s_w[x] = r4d_ar_gather_w(G, (long)(x / K) * G.ew_ld + x % K);
  }
  __syncthreads();
  for (int i = tid; i < T; i += nt) {
    const int f = G.sorted[i];
    if (f >= 0 && f < T) s_i[f] = i;
  }
  if (G.sh)
    for (int r = tid; r < M; r += nt) {
      const float v0 = bf2f(G.sg[(long)r * G.sg_ld]);
      s_g[r] = G.sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
    }
  __syncthreads();
}

// This rank's share of word (row, i0) into the planes at `P` (plane stride `ps` words, word `k`):
// its shared product into plane K + rank, each slot of the row it holds into that slot's plane.
// Returns nothing; the planes are the reducer's (a peer's scratch or this rank's own).
__device__ __forceinline__ void r4d_ar_slot_push(const R4dArGather& G, const int* s_i,
                                                 const float* s_g, int row, int i0, int rank,
                                                 int4* __restrict__ P, long ps, long k) {
  const int K = G.top_k;
  if (G.sh) {
    const int4 v = *reinterpret_cast<const int4*>(G.sh + (long)row * G.sh_ld + i0);
    const unsigned short* ve = reinterpret_cast<const unsigned short*>(&v);
    const float gate = s_g[row];
    int4 o;
    unsigned short* oe = reinterpret_cast<unsigned short*>(&o);
#pragma unroll
    for (int e = 0; e < 8; ++e) oe[e] = arg_bf16_rne(bf2f(ve[e]) * gate);
    P[(long)(K + rank) * ps + k] = o;
  }
  for (int j = 0; j < K; ++j) {
    const int si = s_i[row * K + j];
    if (si < 0) continue;
    P[(long)j * ps + k] = *reinterpret_cast<const int4*>(G.ye + (long)si * G.ye_ld + i0);
  }
}

// The reducer's word k: the WS shared products in ascending rank order (has_sh), then the K
// slots in ascending slot order with their weights `w` (the row's s_w entries), rounded once.
// Loads go out four at a time ahead of the adds, which stay in order.
template <int WS>
__device__ __forceinline__ int4 r4d_ar_slot_sum_word(const int4* __restrict__ P, long ps, long k,
                                                     int K, bool has_sh, const float* w,
                                                     int ew_f32) {
  float acc[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
  if (has_sh) {
    int4 v[WS];
#pragma unroll
    for (int r = 0; r < WS; ++r) v[r] = P[(long)(K + r) * ps + k];
#pragma unroll
    for (int r = 0; r < WS; ++r) {
      const unsigned short* ve = reinterpret_cast<const unsigned short*>(&v[r]);
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[e] += bf2f(ve[e]);
    }
  }
  constexpr int kU = 4;
  for (int j0 = 0; j0 < K; j0 += kU) {
    int4 v[kU];
#pragma unroll
    for (int u = 0; u < kU; ++u)
      if (j0 + u < K) v[u] = P[(long)(j0 + u) * ps + k];
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      if (j0 + u >= K) continue;
      const float wj = w[j0 + u];
      const unsigned short* ve = reinterpret_cast<const unsigned short*>(&v[u]);
      if (ew_f32) {
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] = fmaf(wj, bf2f(ve[e]), acc[e]);
      } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] += wj * bf2f(ve[e]);
      }
    }
  }
  int4 out;
  unsigned short* oe = reinterpret_cast<unsigned short*>(&out);
#pragma unroll
  for (int e = 0; e < 8; ++e) oe[e] = arg_bf16_rne(acc[e]);
  return out;
}

// Plane counts: the one-shot keeps K + WS planes of the whole message a parity, the two-shot
// K + WS planes of the longest shard plus WS of its reduced words (region B, one an owner).
#define R4D_ARS_ONESHOT_PLANES(K, WS) ((K) + (WS))
#define R4D_ARS_TWOSHOT_PLANES(K, WS) ((K) + 2 * (WS))
