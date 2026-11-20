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
// (exl3r4d copy of radiance libr4d/r4d_ika.h, DRAFTER track round 50.) R4D_IKA_ENTRY() is the first statement of a converted kernel's body (drafter/kit/ika_apply.py); the release runs at every exit
// (a local's destructor). Built in only with -DR4D_IKA=ON (CMake option; default OFF = the kernels as before, and an empty
// r4d_ika_kernels list, so the engine never treats a kernel as self-acquiring).
#pragma once
#ifndef R4D_IKA
#define R4D_IKA 0
#endif
#if R4D_IKA && defined(__HIP__) && defined(__HIP_DEVICE_COMPILE__)
struct R4dIkaRel {
    __device__ __forceinline__ ~R4dIkaRel() { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent"); }
};
#define R4D_IKA_ENTRY() __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent"); const R4dIkaRel r4d_ika_rel_ {}; (void)r4d_ika_rel_
#else
#define R4D_IKA_ENTRY() do {} while (0)
#endif
