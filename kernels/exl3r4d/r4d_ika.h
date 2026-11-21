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
//
// Form B (Qwen loop round 8, 21.11.2026; FINDINGS F22 / COORD 05:46): the fences run only in a PLAYED dispatch. The engine tells the
// kernel through its hidden_global_offset_z argument (HIP code never reads the global offsets; the AQL backend writes them 0): it
// writes 1 there in the tape's copy of the arguments of a listed kernel (core/device/aql.cpp tape_note_dispatch), so a played
// dispatch -- the one that goes out without the packet's fences -- fences itself, and an issued one (every prefill pass, any
// dispatch outside a recorded pass: packet fences kept) runs as the OFF build's kernel plus one uniform scalar test. The library
// says it holds this form with r4d_ika_protocol() == 2; the engine treats a listed kernel as self-acquiring only when the kernel's
// metadata carries hidden_global_offset_z (else the kernel is not IKA: it keeps every packet fence).
#pragma once
#ifndef R4D_IKA
#define R4D_IKA 0
#endif
#define R4D_IKA_PROTOCOL 2
#if R4D_IKA && defined(__HIP__) && defined(__HIP_DEVICE_COMPILE__)
extern "C" __device__ unsigned long __ockl_get_global_offset(unsigned int dim);
struct R4dIkaRel {
    bool on;
    __device__ __forceinline__ explicit R4dIkaRel(bool v) : on(v) {}
    __device__ __forceinline__ ~R4dIkaRel() {
        if (on) __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    }
};
#define R4D_IKA_ENTRY()                                                                                      \
    const bool r4d_ika_on_ = __ockl_get_global_offset(2u) != 0ul;                                            \
    if (r4d_ika_on_) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");                                      \
    const R4dIkaRel r4d_ika_rel_(r4d_ika_on_);                                                               \
    (void)r4d_ika_rel_
#else
#define R4D_IKA_ENTRY() do {} while (0)
#endif
