// r4d_ar_sp_rows.hip (fn4 SP-hc, 10.11.2026): the four-rank two-shot's two halves as their own launches, over WHOLE-ROW
// shards, for a sequence-parallel residual (the gated residual's write and read on a rank's quarter of the rows between
// them):
//
//   r4d_ar_rs_rows_4rank_ti8  reduce-scatter: every rank's partial [rows][row] in, the sum of rank r's shard rows out in
//                             place on rank r (its own contribution read exact, the peers' through the ti8 wire, fp32
//                             folded in ascending source rank, then rounded once to the payload type). Other rows of
//                             the buffer are left as they were.
//   r4d_ar_ag_rows_4rank      all-gather: rank r's shard rows of [rows][row] in place to every rank -- on the ti8 wire
//                             (`raw` 0: every rank, the owner too, stores the decode of the owner's words, so all hold
//                             the same bytes) or as raw payload words (`raw` 1: exact).
//
// THE SHARD IS WHOLE ROWS: q = ceil(rows / 4) rows a rank, rank s owns rows [min(s q, rows), min((s + 1) q, rows)) -- the
// gated residual's ops run per row, so a shard may not cut one. The wire is r4d_ar_twoshot_Nrank_ti8.hip's (codes | rec
// | scales a slot, 38 bytes a 32-element group), the transport and sequencing the same (r4d_ar_seq.h, r4d_ar_wide.h):
// ONE handshake a launch (the data-ready one), since a launch reads only what its peers pushed before it and the next
// launch on the instance takes the other slot (the parity counters). Each op is its own instance (its own declare).
#include "r4d.h"
#include "r4d_ar_ti8.h"
#include "r4d_ar_wide.h"
#include <stdexcept>
#include <string>

namespace {

// Rank s's shard: rows [lo, hi) of q a rank, as 32-element groups from group 0 of the buffer.
__device__ __forceinline__ void sp_shard(int s, int n_rows, int q, int rg, long& g0, int& ng) {
  const int lo = s * q < n_rows ? s * q : n_rows;
  const int hi = lo + q < n_rows ? lo + q : n_rows;
  g0 = (long)lo * rg;
  ng = (hi - lo) * rg;
}

}  // namespace

template <typename T, int WS>
__global__ void r4d_ar_rs_rows_ti8_kernel(T* __restrict__ buf, R4DArWidePeers<char, WS - 1> peers,
                                          char* __restrict__ my_base, unsigned int* __restrict__ my_flags,
                                          unsigned int* __restrict__ seq_ctrs, int n_rows, int rg,
                                          long slot_stride_bytes, int rank, int drain, int acq, int pub,
                                          long long* gate, unsigned int* gate_ctr) {
  constexpr int NPEERS = WS - 1;
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  const long long gate_word = r4d_ar_link_word(gate, (unsigned int)tid);
  r4d_ar_seq_take(seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot);
  const unsigned int s = s_seq;
  const int parity = (int)s_slot;
  const int q = (n_rows + WS - 1) / WS;
  const int sg = q * rg;                                   // the longest shard, the slot layout's size
  const int rec_off = TI8_REC_OFF * sg, sc_off = TI8_SC_OFF * sg;
  const int gpb = (sg + nb - 1) / nb;
  const int gA = b * gpb;
  const int gB = gA + gpb < sg ? gA + gpb : sg;

  char* pslot[NPEERS];
#pragma unroll
  for (int p = 0; p < NPEERS; ++p)
    pslot[p] = peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)slot_stride_bytes;

  // ---- my contribution to each peer-owned shard, encoded into its owner's slot for me
  for (int p = 0; p < NPEERS; ++p) {
    long g0; int ng;
    sp_shard(peers.rank[p], n_rows, q, rg, g0, ng);
    const T* src = buf + g0 * TI8_GROUP;
    const int gE = gB < ng ? gB : ng;
    for (int k = 4 * gA + tid; k < 4 * gE; k += nt) {
      const int lig = k & 3, g = k >> 2;
      uint2 codes;
      unsigned int rec;
      __half scale;
      ti8_encode_group<T>(src + (long)k * TI8_EPL, lig, &codes, &rec, &scale);
      ((uint2*)pslot[p])[k] = codes;
      if (lig == 0) {
        ((unsigned int*)(pslot[p] + rec_off))[g] = rec;
        ((__half*)(pslot[p] + sc_off))[g] = scale;
      }
    }
  }
  r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s, drain, acq, pub);
  r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nb, (unsigned int)tid);

  // ---- my shard: own contribution exact, peers decoded, fp32 fold in ascending source rank, one rounding
  long g0; int ng;
  sp_shard(rank, n_rows, q, rg, g0, ng);
  T* mine = buf + g0 * TI8_GROUP;
  const int gM = gB < ng ? gB : ng;
  for (int k = 4 * gA + tid; k < 4 * gM; k += nt) {
    const int lig = k & 3, g = k >> 2;
    float acc[TI8_EPL];
#pragma unroll
    for (int i = 0; i < TI8_EPL; ++i) acc[i] = 0.f;
    for (int r = 0; r < WS; ++r) {
      float v[TI8_EPL];
      if (r == rank) {
        ti8_load8(mine + (long)k * TI8_EPL, v);
      } else {
        const char* base = my_base + ((size_t)parity * WS + r) * (size_t)slot_stride_bytes;
        const uint2 c = ((const uint2*)base)[k];
        const unsigned int rc = ((const unsigned int*)(base + rec_off))[g];
        const float scl = __half2float(((const __half*)(base + sc_off))[g]);
        ti8_decode_group<T>(c, rc, scl, lig, v);
      }
#pragma unroll
      for (int i = 0; i < TI8_EPL; ++i) acc[i] += v[i];
    }
    ti8_store8(mine + (long)k * TI8_EPL, acc);
  }
}

// fn4 SP-hc on the exact line (12.11.2026): the reduce-scatter on RAW payload words -- every rank pushes its raw
// contribution to each peer-owned shard; the owner folds every rank's words (its own included) in fp32 in ascending
// source rank and rounds once, the four-rank exact all-reduce's per-element arithmetic (its two-shot's first half).
template <typename T, int WS, bool XP = false>
__global__ void r4d_ar_rs_rows_raw_kernel(T* __restrict__ buf, R4DArWidePeers<char, WS - 1> peers,
                                          char* __restrict__ my_base, unsigned int* __restrict__ my_flags,
                                          unsigned int* __restrict__ seq_ctrs, int n_rows, int rg,
                                          long slot_stride_bytes, int rank, int drain, int acq, int pub,
                                          long long* gate, unsigned int* gate_ctr) {
  constexpr int NPEERS = WS - 1;
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  const long long gate_word = r4d_ar_link_word(gate, (unsigned int)tid);
  r4d_ar_seq_take(seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot);
  const unsigned int s = s_seq;
  const int parity = (int)s_slot;
  const int q = (n_rows + WS - 1) / WS;
  const int sg = q * rg;
  const int gpb = (sg + nb - 1) / nb;
  const int gA = b * gpb;
  const int gB = gA + gpb < sg ? gA + gpb : sg;

  char* pslot[NPEERS];
#pragma unroll
  for (int p = 0; p < NPEERS; ++p)
    pslot[p] = peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)slot_stride_bytes;

  // ---- my contribution to each peer-owned shard, raw, into its owner's slot for me
  if constexpr (XP) {
    /* fn4 RSF: every peer's word in the same pass, so all links carry the pushes at once (the same stores) */
    const T* src[NPEERS];
    int kE[NPEERS], kmax = 4 * gA;
#pragma unroll
    for (int p = 0; p < NPEERS; ++p) {
      long g0; int ng;
      sp_shard(peers.rank[p], n_rows, q, rg, g0, ng);
      src[p] = buf + g0 * TI8_GROUP;
      kE[p] = 4 * (gB < ng ? gB : ng);
      kmax = kE[p] > kmax ? kE[p] : kmax;
    }
    for (int k = 4 * gA + tid; k < kmax; k += nt) {
      uint4 v[NPEERS];
#pragma unroll
      for (int p = 0; p < NPEERS; ++p) if (k < kE[p]) v[p] = *reinterpret_cast<const uint4*>(src[p] + (long)k * TI8_EPL);
#pragma unroll
      for (int p = 0; p < NPEERS; ++p) if (k < kE[p]) ((uint4*)pslot[p])[k] = v[p];
    }
  } else
  for (int p = 0; p < NPEERS; ++p) {
    long g0; int ng;
    sp_shard(peers.rank[p], n_rows, q, rg, g0, ng);
    const T* src = buf + g0 * TI8_GROUP;
    const int gE = gB < ng ? gB : ng;
    for (int k = 4 * gA + tid; k < 4 * gE; k += nt)
      ((uint4*)pslot[p])[k] = *reinterpret_cast<const uint4*>(src + (long)k * TI8_EPL);
  }
  r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s, drain, acq, pub);
  r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nb, (unsigned int)tid);

  // ---- my shard: every rank's words, fp32 fold in ascending source rank, one rounding
  long g0; int ng;
  sp_shard(rank, n_rows, q, rg, g0, ng);
  T* mine = buf + g0 * TI8_GROUP;
  const int gM = gB < ng ? gB : ng;
  for (int k = 4 * gA + tid; k < 4 * gM; k += nt) {
    float acc[TI8_EPL];
#pragma unroll
    for (int i = 0; i < TI8_EPL; ++i) acc[i] = 0.f;
    for (int r = 0; r < WS; ++r) {
      float v[TI8_EPL];
      if (r == rank) ti8_load8(mine + (long)k * TI8_EPL, v);
      else ti8_load8(reinterpret_cast<const T*>(my_base + ((size_t)parity * WS + r) * (size_t)slot_stride_bytes) + (long)k * TI8_EPL, v);
#pragma unroll
      for (int i = 0; i < TI8_EPL; ++i) acc[i] += v[i];
    }
    ti8_store8(mine + (long)k * TI8_EPL, acc);
  }
}

template <typename T, int WS, bool RAW>
__global__ void r4d_ar_ag_rows_kernel(T* __restrict__ buf, R4DArWidePeers<char, WS - 1> peers,
                                      char* __restrict__ my_base, unsigned int* __restrict__ my_flags,
                                      unsigned int* __restrict__ seq_ctrs, int n_rows, int rg,
                                      long slot_stride_bytes, int rank, int drain, int acq, int pub,
                                      long long* gate, unsigned int* gate_ctr) {
  constexpr int NPEERS = WS - 1;
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  const long long gate_word = r4d_ar_link_word(gate, (unsigned int)tid);
  r4d_ar_seq_take(seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot);
  const unsigned int s = s_seq;
  const int parity = (int)s_slot;
  const int q = (n_rows + WS - 1) / WS;
  const int sg = q * rg;
  const int rec_off = TI8_REC_OFF * sg, sc_off = TI8_SC_OFF * sg;
  const int gpb = (sg + nb - 1) / nb;
  const int gA = b * gpb;
  const int gB = gA + gpb < sg ? gA + gpb : sg;

  char* pslot[NPEERS];
#pragma unroll
  for (int p = 0; p < NPEERS; ++p)
    pslot[p] = peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)slot_stride_bytes;

  // ---- my shard to every peer; on the ti8 wire I keep the decode of my own words
  {
    long g0; int ng;
    sp_shard(rank, n_rows, q, rg, g0, ng);
    T* mine = buf + g0 * TI8_GROUP;
    const int gM = gB < ng ? gB : ng;
    for (int k = 4 * gA + tid; k < 4 * gM; k += nt) {
      if constexpr (RAW) {
        const uint4 w = *reinterpret_cast<const uint4*>(mine + (long)k * TI8_EPL);
#pragma unroll
        for (int p = 0; p < NPEERS; ++p) ((uint4*)pslot[p])[k] = w;
      } else {
        const int lig = k & 3, g = k >> 2;
        uint2 codes;
        unsigned int rec;
        __half scale;
        ti8_encode_group<T>(mine + (long)k * TI8_EPL, lig, &codes, &rec, &scale);
#pragma unroll
        for (int p = 0; p < NPEERS; ++p) {
          ((uint2*)pslot[p])[k] = codes;
          if (lig == 0) {
            ((unsigned int*)(pslot[p] + rec_off))[g] = rec;
            ((__half*)(pslot[p] + sc_off))[g] = scale;
          }
        }
        float v[TI8_EPL];
        ti8_decode_group<T>(codes, rec, __half2float(scale), lig, v);
        ti8_store8(mine + (long)k * TI8_EPL, v);
      }
    }
  }
  r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s, drain, acq, pub);
  r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nb, (unsigned int)tid);

  // ---- every peer's shard from the slot it pushed into mine
  for (int p = 0; p < NPEERS; ++p) {
    const int owner = peers.rank[p];
    const char* base = my_base + ((size_t)parity * WS + owner) * (size_t)slot_stride_bytes;
    long g0; int ng;
    sp_shard(owner, n_rows, q, rg, g0, ng);
    T* dst = buf + g0 * TI8_GROUP;
    const int gE = gB < ng ? gB : ng;
    for (int k = 4 * gA + tid; k < 4 * gE; k += nt) {
      if constexpr (RAW) {
        *reinterpret_cast<uint4*>(dst + (long)k * TI8_EPL) = ((const uint4*)base)[k];
      } else {
        const int lig = k & 3, g = k >> 2;
        const uint2 c = ((const uint2*)base)[k];
        const unsigned int rc = ((const unsigned int*)(base + rec_off))[g];
        const float scl = __half2float(((const __half*)(base + sc_off))[g]);
        float v[TI8_EPL];
        ti8_decode_group<T>(c, rc, scl, lig, v);
        ti8_store8(dst + (long)k * TI8_EPL, v);
      }
    }
  }
}

// ---------------------------------------------------------------------------------- host side
// `op` 0 the reduce-scatter (ti8), 1 the all-gather on the ti8 wire, 2 the all-gather raw, 3 the reduce-scatter raw.
template <int WS>
static void sp_rows_launch(int op, const long* peer_scratch, const long* peer_flags, long my_scratch, long my_flags,
                           long seq_ctrs, long slot_stride_bytes, long buf, long n_rows, long row, long dtype,
                           long rank, long stream_i, long nblocks, long nthreads, long drain, long acq, long pub,
                           long gate, long gate_ctr, bool xpush = false) {
  constexpr int NPEERS = WS - 1;
  const char* who = (op == 0 || op == 3) ? "sp rs" : "sp ag";
  if (rank < 0 || rank >= WS) throw std::runtime_error(std::string(who) + ": rank out of range");
  if (dtype != 0 && dtype != 1) throw std::runtime_error(std::string(who) + ": only bf16/fp16 payloads");
  if (row <= 0 || row % TI8_GROUP) throw std::runtime_error(std::string(who) + ": the row must be whole 32-element groups");
  if (n_rows < WS) throw std::runtime_error(std::string(who) + ": fewer rows than ranks");
  const long rg = row / TI8_GROUP, q = (n_rows + WS - 1) / WS, sg = q * rg;
  if (sg > 0x7fffffffL / 4) throw std::runtime_error(std::string(who) + ": shard too large");
  const long need = (op == 2 || op == 3) ? sg * TI8_GROUP * 2 : (long)TI8_GROUP_BYTES * sg;
  if (need > slot_stride_bytes)
    throw std::runtime_error(std::string(who) + ": slot stride " + std::to_string(slot_stride_bytes) +
                             " too small (needs " + std::to_string(need) + ")");
  int nb = (int)nblocks;
  if (nb < 1) nb = R4D_AR_NB_DESIGN;
  if (nb > sg) nb = (int)sg;
  if (nb > R4D_AR_WIDE_MAX_BLOCKS) nb = R4D_AR_WIDE_MAX_BLOCKS;
  int nt = (int)nthreads;
  if (nt < 64) nt = 64;
  if (nt > 1024) nt = 1024;
  if (nt % TI8_LPG) throw std::runtime_error(std::string(who) + ": nthreads must be a multiple of 4");
  R4DArWidePeers<char, NPEERS> peers{};
  for (int i = 0; i < NPEERS; ++i) {
    if (!peer_scratch[i] || !peer_flags[i]) throw std::runtime_error(std::string(who) + ": null peer pointer");
    peers.scratch[i] = (char*)peer_scratch[i];
    peers.flags[i] = (unsigned int*)peer_flags[i];
    peers.rank[i] = (i < (int)rank) ? i : i + 1;
  }
  const long slot_bytes = 2L * WS * slot_stride_bytes;
  const long flag_bytes = (long)nb * WS * (long)sizeof(unsigned int);
  r4d_ar_check_extent(my_scratch, slot_bytes, who, "my_scratch");
  r4d_ar_check_extent(my_flags, flag_bytes, who, "my_flags");
  r4d_ar_check_extent(seq_ctrs, 2L * R4D_AR_WIDE_MAX_BLOCKS * (long)sizeof(unsigned int), who, "seq_ctrs");
  for (int i = 0; i < NPEERS; ++i) {
    r4d_ar_check_extent(peer_scratch[i], slot_bytes, who, "peer scratch");
    r4d_ar_check_extent(peer_flags[i], flag_bytes, who, "peer flags");
  }
  hipStream_t st = (hipStream_t)stream_i;
#define R4D_SP_ARGS(T)                                                                                     \
  (T*)buf, peers, (char*)my_scratch, (unsigned int*)my_flags, (unsigned int*)seq_ctrs, (int)n_rows, (int)rg, \
      (long)slot_stride_bytes, (int)rank, (int)drain, (int)acq, (int)pub, (long long*)gate, (unsigned int*)gate_ctr
  if (dtype == 0) {
    if (op == 0) r4d_ar_rs_rows_ti8_kernel<__hip_bfloat16, WS><<<nb, nt, 0, st>>>(R4D_SP_ARGS(__hip_bfloat16));
    else if (op == 3 && xpush) r4d_ar_rs_rows_raw_kernel<__hip_bfloat16, WS, true><<<nb, nt, 0, st>>>(R4D_SP_ARGS(__hip_bfloat16));
    else if (op == 3) r4d_ar_rs_rows_raw_kernel<__hip_bfloat16, WS><<<nb, nt, 0, st>>>(R4D_SP_ARGS(__hip_bfloat16));
    else if (op == 1) r4d_ar_ag_rows_kernel<__hip_bfloat16, WS, false><<<nb, nt, 0, st>>>(R4D_SP_ARGS(__hip_bfloat16));
    else r4d_ar_ag_rows_kernel<__hip_bfloat16, WS, true><<<nb, nt, 0, st>>>(R4D_SP_ARGS(__hip_bfloat16));
  } else {
    if (op == 0) r4d_ar_rs_rows_ti8_kernel<__half, WS><<<nb, nt, 0, st>>>(R4D_SP_ARGS(__half));
    else if (op == 3 && xpush) r4d_ar_rs_rows_raw_kernel<__half, WS, true><<<nb, nt, 0, st>>>(R4D_SP_ARGS(__half));
    else if (op == 3) r4d_ar_rs_rows_raw_kernel<__half, WS><<<nb, nt, 0, st>>>(R4D_SP_ARGS(__half));
    else if (op == 1) r4d_ar_ag_rows_kernel<__half, WS, false><<<nb, nt, 0, st>>>(R4D_SP_ARGS(__half));
    else r4d_ar_ag_rows_kernel<__half, WS, true><<<nb, nt, 0, st>>>(R4D_SP_ARGS(__half));
  }
#undef R4D_SP_ARGS
}

void r4d_ar_sp_rows_4rank(int op, long peer_scratch0, long peer_scratch1, long peer_scratch2, long peer_flags0,
                          long peer_flags1, long peer_flags2, long my_scratch, long my_flags, long seq_ctrs,
                          long slot_stride_bytes, long buf, long n_rows, long row, long dtype, long rank, long stream,
                          long nblocks, long nthreads, long drain, long acq, long pub, long gate, long gate_ctr,
                          long xpush) {
  const long ps[3] = {peer_scratch0, peer_scratch1, peer_scratch2};
  const long pf[3] = {peer_flags0, peer_flags1, peer_flags2};
  sp_rows_launch<4>(op, ps, pf, my_scratch, my_flags, seq_ctrs, slot_stride_bytes, buf, n_rows, row, dtype, rank, stream,
                    nblocks, nthreads, drain, acq, pub, gate, gate_ctr, xpush != 0);
}
