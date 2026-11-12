#pragma once
// r4d_ar_xpush.h (ds41-int13 XPUSH, ported from fn4-x3-sp3 808866c2, FN RSF item 204): the exact two-shot's phase-1 word
// loop in the interleaved form (R4D_AR8_FORM_XPUSH, r4d_ar_shards.h). r4d_ar_twoshot_wide_exact_body's push1 runs it for
// that form; tests/r4d_emu ds41xpush holds it to the lockstep loop's stores. Nothing else includes it.
#include <hip/hip_runtime.h>
#include <cstddef>

// The words [p0, p1) of every peer-owned shard -- peer i's rank peer_rank[i] owns the shard at off[owner] of in4,
// off[owner + 1] - off[owner] words long -- go to pa[i][k], the owner's region A of this rank's slot. Lockstep stores
// them peer after peer (one link busy at a time); here a thread loads word k of every peer's shard, then stores each,
// so every link carries phase-1 bytes at once. The same word to the same place: the same bits.
template <int NPEERS>
__device__ __forceinline__ void r4d_ar_push1_xpush(const int4* __restrict__ in4, int4* const* pa,
                                                   const int* peer_rank, const int* off, int p0, int p1, int tid,
                                                   int nt) {
  const int4* s1[NPEERS];
  int lim[NPEERS], lmax = p0;
#pragma unroll
  for (int i = 0; i < NPEERS; ++i) {
    const int owner = peer_rank[i];
    const int len = off[owner + 1] - off[owner];
    s1[i] = in4 + (size_t)off[owner];
    lim[i] = p1 < len ? p1 : len;
    lmax = lim[i] > lmax ? lim[i] : lmax;
  }
  for (int k = p0 + tid; k < lmax; k += nt) {
    int4 v[NPEERS];
#pragma unroll
    for (int i = 0; i < NPEERS; ++i) if (k < lim[i]) v[i] = s1[i][k];
#pragma unroll
    for (int i = 0; i < NPEERS; ++i) if (k < lim[i]) pa[i][k] = v[i];
  }
}
