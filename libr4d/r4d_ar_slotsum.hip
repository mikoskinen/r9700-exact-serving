// r4d_ar_slotsum.hip: moe_gather_all_reduce_slot (lever D3s) -- the MoE gather and the four-rank
// exact all-reduce as ONE slot-order sum, which no rank's share of the experts decides.
//
// Each token's output is acc = (the ranks' shared-arm products H_r, ascending rank) then (every
// slot's routed row S_j times its weight, ascending slot), rounded once (r4d_ar_hc.h, "The
// SLOT-ORDER sum"). A slot's row is the same bytes whichever rank computed it, so the result does
// not depend on expert placement: experts may be moved between ranks or replicated, and a token's
// slot may be served by any copy, without changing a byte. Unlike moe_gather_all_reduce it is NOT
// moe_gather + all_reduce: there is no per-rank partial and no rank-order fold.
//
// Two transports on the wide family's instance (flags, seq counters, parity, r4d_ar_wide.h):
//
//  * one-shot (below ts_min): every rank writes its H and its slots' rows of every word of the
//    block's range into planes of EVERY rank's scratch (its own included), one handshake, then each
//    rank sums every word of the block's range from its own planes. A parity's half of the scratch
//    holds K + WS planes of the whole message.
//  * two-shot (from ts_min): the message's words in WS shards (r4d_ar_shards.h, the even split);
//    phase 1 writes the H and slot rows of each shard's words into that shard's owner's planes, the
//    owner sums its shard after the first handshake and pushes the reduced words into every rank's
//    region B (one plane an owner), and after the second each rank assembles the peers' shards.
//    A parity's half holds K + 2 WS planes of the longest shard.
//
// A plane entry (j, k) is written by exactly one rank -- the one holding slot j of word k's row --
// and H_r's by rank r, so no two ranks write one word. The planes span the whole parity half (all
// WS slots of the stride), which is safe on a shared instance as the other wide kernels' (parity,
// source) slots are: a rank enters launch n + 2 (the same parity) only after launch n + 1's
// handshake, which every peer reaches only after its launch n has finished on its stream. Within a
// launch block b's words are written by every rank's block b and read after the handshake with
// them (the same grid on every rank: the same M, n and rule).
#include "r4d.h"
#include <cstring>
#include "r4d_ar_wide.h"
#include "r4d_ar_shards.h"
#include "r4d_ar_hc.h"
#include <stdexcept>
#include <string>

// The shard table as a kernel argument (the two-shot's own, r4d_ar_twoshot_Nrank_exact.hip's form).
template <int WS>
struct R4DArsShardOff {
  int off[WS + 1];
  int sh_max;
};

template <int WS>
__global__ void r4d_ar_slot_oneshot_kernel(
    int4* __restrict__ y4, R4DArWidePeers<int4, WS - 1> peers, int4* __restrict__ my_base,
    unsigned int* __restrict__ my_flags, unsigned int* __restrict__ seq_ctrs, int n16, int wpr,
    long half16, int rank, int drain, int acq, int pub, R4dArGather G, long long* gate,
    unsigned int* gate_ctr) {
  constexpr int NPEERS = WS - 1;
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  __shared__ int   s_i[R4D_ARGL_TAB];
  __shared__ float s_w[R4D_ARGL_TAB];
  __shared__ float s_g[R4D_ARGL_ROWS];
  const long long gate_word = r4d_ar_link_word(gate, (unsigned int)tid);
  r4d_ar_seq_take(seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot);  // device-resident, replay-safe
  const unsigned int s = s_seq;
  const int parity = (int)s_slot;

  r4d_ar_slot_table(G, s_i, s_w, s_g);
  const int K = G.top_k;

  const int w = (n16 + nb - 1) / nb;
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;
  const long ps = n16;

  int4* pp[NPEERS];
#pragma unroll
  for (int p = 0; p < NPEERS; ++p) pp[p] = peers.scratch[p] + (size_t)parity * (size_t)half16;
  int4* mp = my_base + (size_t)parity * (size_t)half16;

  // phase 1: my H and my slots' rows of every word of the range, into every rank's planes
  for (int k = start + tid; k < end; k += nt) {
    const int row = k / wpr, i0 = (k - row * wpr) * 8;
    r4d_ar_slot_push(G, s_i, s_g, row, i0, rank, mp, ps, k);
#pragma unroll
    for (int p = 0; p < NPEERS; ++p) r4d_ar_slot_push(G, s_i, s_g, row, i0, rank, pp[p], ps, k);
  }

  r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s, drain, acq, pub);
  // Every peer's write for this launch has landed and mine retired behind the flag: from here on
  // the sum is local, so the last block ends the engine's hold (r4d_ar_seq.h).
  r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nb, (unsigned int)tid);

  for (int k = start + tid; k < end; k += nt) {
    const int row = k / wpr;
    y4[k] = r4d_ar_slot_sum_word<WS>(mp, ps, k, K, G.sh != nullptr, s_w + row * K, G.ew_f32);
  }
}

template <int WS>
__global__ void r4d_ar_slot_twoshot_kernel(
    int4* __restrict__ y4, R4DArWidePeers<int4, WS - 1> peers, int4* __restrict__ my_base,
    unsigned int* __restrict__ my_flags, unsigned int* __restrict__ seq_ctrs, R4DArsShardOff<WS> sh,
    int wpr, long half16, int rank, int drain, int acq, int pub, R4dArGather G, long long* gate,
    unsigned int* gate_ctr) {
  constexpr int NPEERS = WS - 1;
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  __shared__ int   s_i[R4D_ARGL_TAB];
  __shared__ float s_w[R4D_ARGL_TAB];
  __shared__ float s_g[R4D_ARGL_ROWS];
  const long long gate_word = r4d_ar_link_word(gate, (unsigned int)tid);
  r4d_ar_seq_take(seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot);  // device-resident, replay-safe
  const unsigned int v0 = 2u * s_seq - 1u;
  const int parity = (int)s_slot;

  r4d_ar_slot_table(G, s_i, s_w, s_g);
  const int K = G.top_k;

  const int sh_max = sh.sh_max;
  const int w = (sh_max + nb - 1) / nb;    // block b owns words [w0,w1) of the SHARD space
  const int w0 = b * w;
  int w1 = w0 + w; if (w1 > sh_max) w1 = sh_max;
  const long ps = sh_max;
  const long breg = (long)(K + WS) * ps;   // region B: plane K + WS + owner

  int4* pp[NPEERS];
#pragma unroll
  for (int p = 0; p < NPEERS; ++p) pp[p] = peers.scratch[p] + (size_t)parity * (size_t)half16;
  int4* mp = my_base + (size_t)parity * (size_t)half16;

  // phase 1: my H and my slots' rows of each shard's words in my range, into the owner's planes
  for (int o = 0; o < WS; ++o) {
    const int off = sh.off[o], len = sh.off[o + 1] - off;
    const int lim = w1 < len ? w1 : len;
    int4* dst = o == rank ? mp : pp[o < rank ? o : o - 1];
    for (int k = w0 + tid; k < lim; k += nt) {
      const int gw = off + k, row = gw / wpr, i0 = (gw - row * wpr) * 8;
      r4d_ar_slot_push(G, s_i, s_g, row, i0, rank, dst, ps, k);
    }
  }
  r4d_ar_handshake<WS>(peers, my_flags, b, rank, v0, drain, acq, pub);

  // the sum of my shard's words, into my y and every peer's region B (my plane there)
  {
    const int my_off = sh.off[rank], my_len = sh.off[rank + 1] - my_off;
    const int my_lim = w1 < my_len ? w1 : my_len;
    for (int k = w0 + tid; k < my_lim; k += nt) {
      const int row = (my_off + k) / wpr;
      const int4 o = r4d_ar_slot_sum_word<WS>(mp, ps, k, K, G.sh != nullptr, s_w + row * K, G.ew_f32);
      y4[my_off + k] = o;
#pragma unroll
      for (int p = 0; p < NPEERS; ++p) pp[p][breg + (long)rank * ps + k] = o;
    }
  }
  r4d_ar_handshake<WS>(peers, my_flags, b, rank, v0 + 1u, drain, acq, pub);
  // As in the plain two-shot: every remote store of both phases has retired behind its flag and
  // every peer's push to this rank has been waited on; only the local assemble is left.
  r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nb, (unsigned int)tid);

  // assemble: every peer-owned shard from my region B
  for (int p = 0; p < NPEERS; ++p) {
    const int owner = peers.rank[p];
    const int4* srcB = mp + breg + (long)owner * ps;
    int4* dst = y4 + (size_t)sh.off[owner];
    const int len = sh.off[owner + 1] - sh.off[owner];
    const int lim = w1 < len ? w1 : len;
    for (int k = w0 + tid; k < lim; k += nt) dst[k] = srcB[k];
  }
}

/* --- host side --- */

/* The slot form fits: a slot table in the block's LDS (M <= 256 rows, M * top_k <= 2048 slots),
 * whole 16B rows, a word a rank, and the form's planes in a parity's half of the scratch (`form`
 * 0 the one-shot, 1 the two-shot). */
int r4d_ar_slot_ok(long M, long n, long top_k, long slot_stride16, long form) {
  if (M < 1 || n < 8 || n % 8 || top_k < 1 || top_k > R4D_ARG_SLOTS) return 0;
  if (M > R4D_ARGL_ROWS || M * top_k > R4D_ARGL_TAB) return 0;
  const long n16 = M * n / 8;
  if (n16 < 4 || M * n > R4D_AR_TWOSHOT_WIDE_MAX_ELEMS) return 0;
  const long half16 = 4L * slot_stride16;
  if (form == 0) return R4D_ARS_ONESHOT_PLANES(top_k, 4) * n16 <= half16 ? 1 : 0;
  long off[5];
  const long sh_max = r4d_ar_shard_table(n16, 4, nullptr, off);
  return R4D_ARS_TWOSHOT_PLANES(top_k, 4) * sh_max <= half16 ? 1 : 0;
}

/* The scratch an instance needs for the slot form at M rows of n and top_k slots: the one-shot's
 * planes, in elements of one slot (the instance's max_numel, the key r4d_ar_entry sizes it by). */
long r4d_ar_slot_numel(long M, long n, long top_k) {
  const long planes = R4D_ARS_ONESHOT_PLANES(top_k, 4);
  return (M * n * planes + 3) / 4 + 8;
}

void r4d_ar_slot_4rank(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                       long peer_flags0, long peer_flags1, long peer_flags2, long my_scratch,
                       long my_flags, long seq_ctrs, long slot_stride16, long y, long M, long n,
                       long ye, long ew, long sorted, long sh, long sg, long ye_ld, long ew_ld,
                       long sh_ld, long sg_ld, long top_k, long sig, long rank, long stream,
                       long form, long nblocks, long nthreads, long drain, long acq, long pub,
                       long gate, long gate_ctr) {
  constexpr int WS = 4, NPEERS = 3;
  if (rank < 0 || rank >= WS)
    throw std::runtime_error("moe_gather_ar_slot: rank " + std::to_string(rank) +
                             " out of range for four ranks");
  if (!r4d_ar_slot_ok(M, n, top_k, slot_stride16, form))
    throw std::runtime_error("moe_gather_ar_slot: M " + std::to_string(M) + ", n " +
                             std::to_string(n) + ", top_k " + std::to_string(top_k) + " outside the " +
                             (form ? "two" : "one") + "-shot form (slot table, word a rank, planes)");
  if (!ye || !ew || !sorted || ye_ld % 8 || (sh && (sh_ld < n || sh_ld % 8)) || (sh != 0) != (sg != 0))
    throw std::runtime_error("moe_gather_ar_slot: the gather's operands are not 16B rows");
  const long n16 = M * n / 8;
  const long half16 = 4L * slot_stride16;

  int nb = (int)nblocks;
  if (nb < 1) nb = 2;
  long cap = n16;
  long off[WS + 1];
  long sh_max = 0;
  if (form) {
    sh_max = r4d_ar_shard_table(n16, WS, nullptr, off);
    cap = sh_max;
  }
  if (nb > cap) nb = (int)cap;
  if (nb > R4D_AR_WIDE_MAX_BLOCKS) nb = R4D_AR_WIDE_MAX_BLOCKS;
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;

  R4DArWidePeers<int4, NPEERS> peers{};
  const long ps[3] = {peer_scratch0, peer_scratch1, peer_scratch2};
  const long pf[3] = {peer_flags0, peer_flags1, peer_flags2};
  for (int i = 0; i < NPEERS; ++i) {
    if (!ps[i] || !pf[i])
      throw std::runtime_error("moe_gather_ar_slot: null peer scratch/flag pointer at index " +
                               std::to_string(i));
    peers.scratch[i] = (int4*)ps[i];
    peers.flags[i] = (unsigned int*)pf[i];
    peers.rank[i] = (i < (int)rank) ? i : i + 1;
  }
  const long slot_bytes = 2L * WS * (long)slot_stride16 * 16;
  const long flag_bytes = (long)nb * WS * (long)sizeof(unsigned int);
  r4d_ar_check_extent(my_scratch, slot_bytes, "moe_gather_ar_slot", "my_scratch");
  r4d_ar_check_extent(my_flags, flag_bytes, "moe_gather_ar_slot", "my_flags");
  r4d_ar_check_extent(seq_ctrs, 2L * R4D_AR_WIDE_MAX_BLOCKS * (long)sizeof(unsigned int),
                      "moe_gather_ar_slot", "seq_ctrs");
  for (int i = 0; i < NPEERS; ++i) {
    r4d_ar_check_extent(ps[i], slot_bytes, "moe_gather_ar_slot", "peer scratch");
    r4d_ar_check_extent(pf[i], flag_bytes, "moe_gather_ar_slot", "peer flags");
  }

  /* Zeroed including padding: passed by value, so its tail padding is part of the kernel-argument
   * bytes a recorded pass's audit compares. `sig` bit 0: the shared gate's sigmoid; bit 1: f32
   * slot weights. */
  R4dArGather G;
  std::memset(&G, 0, sizeof G);
  G.ye = (const unsigned short*)ye;
  G.ew = (const unsigned short*)ew;
  G.sorted = (const int*)sorted;
  G.sh = (const unsigned short*)sh;
  G.sg = (const unsigned short*)sg;
  G.ye_ld = ye_ld; G.ew_ld = ew_ld; G.sh_ld = sh_ld; G.sg_ld = sg_ld;
  G.top_k = (int)top_k; G.T = (int)(M * top_k); G.sig = (int)(sig & 1); G.ew_f32 = (int)((sig >> 1) & 1);

  hipStream_t st = (hipStream_t)stream;
  if (form) {
    R4DArsShardOff<WS> shd{};
    for (int s = 0; s <= WS; ++s) shd.off[s] = (int)off[s];
    shd.sh_max = (int)sh_max;
    r4d_ar_slot_twoshot_kernel<WS><<<nb, nt, 0, st>>>(
        (int4*)y, peers, (int4*)my_scratch, (unsigned int*)my_flags, (unsigned int*)seq_ctrs, shd,
        (int)(n / 8), half16, (int)rank, (int)drain, (int)acq, (int)pub, G, (long long*)gate,
        (unsigned int*)gate_ctr);
  } else {
    r4d_ar_slot_oneshot_kernel<WS><<<nb, nt, 0, st>>>(
        (int4*)y, peers, (int4*)my_scratch, (unsigned int*)my_flags, (unsigned int*)seq_ctrs,
        (int)n16, (int)(n / 8), half16, (int)rank, (int)drain, (int)acq, (int)pub, G,
        (long long*)gate, (unsigned int*)gate_ctr);
  }
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess)
    throw std::runtime_error(std::string("moe_gather_ar_slot: launch failed: ") + hipGetErrorString(e));
}
