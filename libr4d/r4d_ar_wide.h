#pragma once
// r4d_ar_wide.h: shared transport of the wide (ws>2) all-reduce family. Every wide kernel
// (one-shot exact, two-shot exact, two-shot ti8) publishes in the 2*seq flag space (the two-shots
// add the 2s-1 half-step), advances seq_ctrs[b] once per AR (a two-shot in pieces by its piece
// count, reserving the extra flag values it publishes), and lays scratch out as 2*ws slots of
// one stride (parity x source rank). The parity is the launch's, from the parity counters in
// r4d_ar_seq.h, so seq_ctrs is 2 * R4D_AR_WIDE_MAX_BLOCKS words. The constants sizing the shared
// flag/seq/scratch set are defined only here so they cannot drift apart.
//
// The 2-rank pair is not here: it has its own flag layout (one word per block, no ws stride), an
// acquire-on-load spin and its own block ceiling.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "r4d_ar_seq.h"
#include "r4d_ar_shards.h"

#define R4D_AR_SPIN_MAX 4000000000ULL
#define R4D_AR_WIDE_MAX_BLOCKS 64
#define R4D_AR_NB_DESIGN 16
#define R4D_AR_WIDE_MAX_PEERS 7

// Ascending peer rank throughout; rank[] maps a spin lane to its global source rank. The
// scratch word type is the one per-family difference (int4 exact wire, char ti8 wire).
template <typename S, int NPEERS>
struct R4DArWidePeers {
  S* scratch[NPEERS];
  unsigned int* flags[NPEERS];
  int rank[NPEERS];
};

__device__ __forceinline__ void r4d_ar_store_sys_rel(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void r4d_ar_store_sys_rlx(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned int r4d_ar_load_sys_rlx(const unsigned int* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ float r4d_ar_to_f(const float& x) { return x; }
__device__ __forceinline__ float r4d_ar_to_f(const __half& x) { return __half2float(x); }
__device__ __forceinline__ float r4d_ar_to_f(const __hip_bfloat16& x) { return (float)x; }
__device__ __forceinline__ void r4d_ar_from_f(float v, float& o) { o = v; }
__device__ __forceinline__ void r4d_ar_from_f(float v, __half& o) { o = __float2half(v); }
__device__ __forceinline__ void r4d_ar_from_f(float v, __hip_bfloat16& o) { o = (__hip_bfloat16)v; }

// drain: 0 none, 1 all-thread threadfence_system, 2 single-thread (applied by the publisher
// inside the handshake), 3 s_wait_storecnt 0.
// fn4 PFF (16.11.2026, qwen/prefill; libr4d switch ar_fence, r4d_ar_entry.hip): 5 = every wave's release half only
// (global_wb scope:SCOPE_SYS + its store wait: drain 1 without the global_inv scope:SCOPE_SYS that drain 1 adds a wave);
// 7 = the push went out as system-scope stores (r4d_ar_push16 with drain 7: written past this card's caches, so no
// writeback is needed for them), so every wave only waits for its stores to complete (s_wait_storecnt 0). drain 7 is
// passed only to kernels whose pushes go through r4d_ar_push16 (the SP rows' raw reduce-scatter and all-gather).
__device__ __forceinline__ void r4d_ar_drain(int drain) {
  if (drain == 1) __threadfence_system();
  else if (drain == 3 || drain == 7) asm volatile("s_wait_storecnt 0x0" ::: "memory");
  else if (drain == 5) __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
}
// acq: 0 acquire-only fence (global_inv, no writeback half), 1 full threadfence_system,
// 2 none -- fine-grained scratch only.
// fn4 PFF: 3 = ONE system-scope invalidate a block (wave 0, waited for, then a barrier) and every wave's device-scope
// invalidate (global_inv scope:SCOPE_DEV: its CU's vector caches) -- the L2 is the card's, so one system invalidate covers
// every wave of the block; 4 = every wave's device-scope invalidate only (the slots are fine-grained: this rank reads them
// past L2, r4d_ar_entry.hip's allocation note, so only the vector caches can hold an older copy). Every thread calls it.
__device__ __forceinline__ void r4d_ar_acquire(int acq) {
  if (acq == 1) __threadfence_system();
  else if (acq == 3) {
    if (threadIdx.x < 32u) {
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
      asm volatile("s_wait_loadcnt 0x0" ::: "memory");
    }
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  } else if (acq == 4) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  else if (acq != 2) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
}
// fn4 PFF: one 16-byte push word into a peer's slot. drain 7: as two relaxed system-scope 64-bit stores (global_store_b64
// scope:SCOPE_SYS, written past this card's caches; r4d_ar_drain(7) then waits for them); otherwise the plain store.
__device__ __forceinline__ void r4d_ar_push16(void* dst, const uint4& v, int drain) {
  if (drain == 7) {
    unsigned long long* d = reinterpret_cast<unsigned long long*>(dst);
    __hip_atomic_store(d, ((unsigned long long)v.y << 32) | v.x, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    __hip_atomic_store(d + 1, ((unsigned long long)v.w << 32) | v.z, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  } else {
    *reinterpret_cast<uint4*>(dst) = v;
  }
}

// Publish `val` to every peer's flag word [b][rank], then wait until each of this block's own
// words has reached or passed it (peers may be a step ahead; counters only grow). One spin lane
// per source, wrap-safe monotonic compare, never equality. Every rank publishes before it waits,
// so there is no cycle; a handshake that cannot complete traps.
//
// `tl`, when not null, takes two wall-clock stamps: when the whole block has issued and drained
// its stores, and when every peer's flag has been seen. A diagnostic: the eight-rank two-shot
// passes it only for a launch asking for its timeline (r4d_ar_tl_stamp).
template <int WS, typename S, int NPEERS>
__device__ __forceinline__ void r4d_ar_handshake(const R4DArWidePeers<S, NPEERS>& peers,
                                                 unsigned int* __restrict__ my_flags, int b,
                                                 int rank, unsigned int val, int drain, int acq,
                                                 int pub, long long* tl = nullptr) {
  r4d_ar_drain(drain);
  __syncthreads();
  const int tid = threadIdx.x;
  if (tl && tid == 0) tl[0] = (long long)wall_clock64();
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    if (pub) {
      // One system writeback for the whole publish, then relaxed stores. Per-peer release stores
      // would add writebacks that order nothing, since the flags go to different peers.
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
#pragma unroll
      for (int p = 0; p < NPEERS; ++p)
        r4d_ar_store_sys_rlx(&peers.flags[p][(size_t)b * WS + rank], val);
    } else {
#pragma unroll
      for (int p = 0; p < NPEERS; ++p)
        r4d_ar_store_sys_rel(&peers.flags[p][(size_t)b * WS + rank], val);
    }
  }
  if (tid < NPEERS) {
    const unsigned int* f = &my_flags[(size_t)b * WS + peers.rank[tid]];
    unsigned long long z = 0;
    while ((int)(r4d_ar_load_sys_rlx(f) - val) < 0) {
      __builtin_amdgcn_s_sleep(2);
      if (++z > R4D_AR_SPIN_MAX) __builtin_trap();  // lockstep violation
    }
  }
  __syncthreads();
  if (tl && tid == 0) tl[1] = (long long)wall_clock64();
  r4d_ar_acquire(acq);
}

// The two halves of r4d_ar_handshake, for a kernel that tells its peers how far it has come before
// it needs to know how far they have (the eight-rank two-shot's split form). Publish: once this
// block's stores have drained, every peer's word [b][rank] takes `val`.
template <int WS, typename S, int NPEERS>
__device__ __forceinline__ void r4d_ar_publish(const R4DArWidePeers<S, NPEERS>& peers, int b,
                                               int rank, unsigned int val, int drain, int pub) {
  r4d_ar_drain(drain);
  __syncthreads();
  if (threadIdx.x == 0) {
    if (drain == 2) __threadfence_system();
    if (pub) {
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
#pragma unroll
      for (int p = 0; p < NPEERS; ++p)
        r4d_ar_store_sys_rlx(&peers.flags[p][(size_t)b * WS + rank], val);
    } else {
#pragma unroll
      for (int p = 0; p < NPEERS; ++p)
        r4d_ar_store_sys_rel(&peers.flags[p][(size_t)b * WS + rank], val);
    }
  }
}

// Await: wait until each of this block's words from every peer has reached `val`, then acquire
// what they published before it. `tl` as r4d_ar_handshake's, stamped on entry and on release.
template <int WS, typename S, int NPEERS>
__device__ __forceinline__ void r4d_ar_await(const R4DArWidePeers<S, NPEERS>& peers,
                                             unsigned int* __restrict__ my_flags, int b,
                                             unsigned int val, int acq, long long* tl = nullptr) {
  const int tid = threadIdx.x;
  if (tl && tid == 0) tl[0] = (long long)wall_clock64();
  if (tid < NPEERS) {
    const unsigned int* f = &my_flags[(size_t)b * WS + peers.rank[tid]];
    unsigned long long z = 0;
    while ((int)(r4d_ar_load_sys_rlx(f) - val) < 0) {
      __builtin_amdgcn_s_sleep(2);
      if (++z > R4D_AR_SPIN_MAX) __builtin_trap();  // lockstep violation
    }
  }
  __syncthreads();
  if (tl && tid == 0) tl[1] = (long long)wall_clock64();
  r4d_ar_acquire(acq);
}

// The two-shot timeline (layout in r4d_ar_shards.h), one stamp of the device wall clock.
static_assert(R4D_AR_TL_BLOCKS == R4D_AR_WIDE_MAX_BLOCKS, "r4d_ar_shards.h restates the ceiling");
__device__ __forceinline__ void r4d_ar_tl_stamp(long long* tl, int i) {
  if (tl && threadIdx.x == 0) tl[i] = (long long)wall_clock64();
}

// Host side: checks stride and grid against the real allocations. Every scratch, flag and seq
// pointer comes from hipMalloc / hipIpcOpenMemHandle, so the runtime can report its extent; a
// pointer it cannot size is refused.
inline void r4d_ar_check_extent(long p, long need, const char* prefix, const char* what) {
  hipDeviceptr_t base = nullptr;
  size_t sz = 0;
  hipError_t e = hipMemGetAddressRange(&base, &sz, (hipDeviceptr_t)(uintptr_t)p);
  if (e != hipSuccess)
    throw std::runtime_error(std::string(prefix) + ": " + what +
                             " is not an allocation the runtime can size (" +
                             hipGetErrorString(e) + ")");
  const long avail = (long)(((char*)base + sz) - (char*)(uintptr_t)p);
  if (avail < need)
    throw std::runtime_error(std::string(prefix) + ": " + what + " backs " +
                             std::to_string(avail) + " bytes from the given pointer, needs " +
                             std::to_string(need));
}
