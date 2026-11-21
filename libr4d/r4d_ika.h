// r4d_ika.h -- lever IKA (DRAFTER track round 50, 21.11.2026): a kernel that acquires and releases at agent scope ITSELF, so the
// engine may play its AQL packet without the packet processor's acquire and release (core/device/aql.cpp ika_kernel, RADIANCE_AQL_IKA=1).
//
// Why: between two dependent kernels on one queue the packet's agent-scope acquire (every CU's caches invalidated by the
// packet processor) and release cost 1.2-2.2 us on the R9700 (drafter/kit/mes/mesp.cpp, 200-packet chains, host-timed:
// acquire + release agent 3.86 / 4.84 us a tiny kernel (two modes), none / none 2.63 us; 64 x 256 threads 6.05 / 7.09 -> 4.75).
// LLVM's agent acquire fence (global_inv scope:SCOPE_DEV in each wave, ~0.04-0.26 us a kernel) and agent release fence
// (s_wait_storecnt 0 before the wave ends) do the same work for the CUs the kernel runs on. The scalar cache is not covered (LLVM's
// acquire does not invalidate it): the engine keeps the packet acquire where a line the kernel reads could be stale there
// (Ctx::haz_mark, lines read and then written since the last packet acquire). cohbench3 (drafter/kit/mes): 0 stale elements in
// 300 trials x 3 sizes with this entry and the packets' acquire + release none, vs 1.2 M - 1.9 G stale without it.
//
// R4D_IKA_ENTRY() is the first statement of a converted kernel's body (drafter/kit/ika_apply.py); the release runs at every exit
// (a local's destructor). Built in only with -DR4D_IKA=ON (CMake option; default OFF = the kernels as before, and an empty
// r4d_ika_kernels list, so the engine never treats a kernel as self-acquiring).
//
// Lever CIK (DRAFTER track round 62): with R4D_IKA_CHAIN (CMake option, needs R4D_IKA) the same entry is CHAIN-CAPABLE -- when the
// engine wrote a chain record into the argument slot's tail (r4d_chain.h; the engine does so only under RADIANCE_AQL_CIK=1 for a
// module exporting r4d_ika_chain() = 1), the block waits at entry for the queue's chain counter to reach the record's target, and
// every lane adds 1 to the counter at its exit (after the release fence; the atomic optimizer makes it one add of popcount(exec) a
// wave, so early returns stay legal and no block barrier sits at an exit). The record's unit is therefore LANES (aql.cpp lanes_of).
// No record (a HIP launch, the engine with chaining off): the IKA entry exactly, plus one scalar check at each end.
// R4D_IKA_CHAIN_ENTRY() is the chain-capable form under any build (a bench names it directly).
#pragma once
#ifndef R4D_IKA
#define R4D_IKA 0
#endif
#ifndef R4D_IKA_CHAIN
#define R4D_IKA_CHAIN 0
#endif
#if R4D_IKA && defined(__HIP__) && defined(__HIP_DEVICE_COMPILE__)
#include "r4d_chain.h"
struct R4dIkaRel {
    __device__ __forceinline__ ~R4dIkaRel() { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent"); }
};
struct R4dIkaChainRel {
    __device__ __forceinline__ ~R4dIkaChainRel() {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        const R4dChainRec* r = r4d_chain_rec();
        if (r) __hip_atomic_fetch_add(r->ctr, 1ull, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
};
/* r4d_chain_wait(): no record -> returns at once; else lane 0 polls, __syncthreads, agent acquire fence. The entry's own acquire
 * fence follows in both cases (IKA's). */
#define R4D_IKA_CHAIN_ENTRY() r4d_chain_wait(); __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent"); \
    const R4dIkaChainRel r4d_ika_rel_ {}; (void)r4d_ika_rel_
#if R4D_IKA_CHAIN
#define R4D_IKA_ENTRY() R4D_IKA_CHAIN_ENTRY()
#else
#define R4D_IKA_ENTRY() __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent"); const R4dIkaRel r4d_ika_rel_ {}; (void)r4d_ika_rel_
#endif
#else
#define R4D_IKA_ENTRY() do {} while (0)
#define R4D_IKA_CHAIN_ENTRY() do {} while (0)
#endif
