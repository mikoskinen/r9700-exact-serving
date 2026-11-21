// r4d_chain.h -- a chained launch's wait and completion (RadKernelChainFn, abi/rad_abi.h).
//
// Between two dependent kernels on one queue the packet processor waits for the first to finish
// before it launches the second (the AQL barrier bit): on 4x R9700, ~1.3 ms of a 14 ms DFlash2 step
// on the 31B. A chained launch drops the bit. The engine writes a record into the
// kernel's argument slot, and the kernel waits for its queue's completion counter to reach the
// blocks issued before it, then adds its own blocks as they finish. A kernel's chained form
// (`CHAIN` true) brackets its body with r4d_chain_wait / r4d_chain_done; the plain form never reads
// the record, so a launch from outside the engine (r4d_selftest, rad-kbench) is untouched.
//
// Every block must reach r4d_chain_done exactly once, so a kernel whose blocks return early keeps
// its body in a __device__ function called from the chained wrapper.
#pragma once
#include <hip/hip_runtime.h>
#include "rad_abi.h"

#if !defined(__HIP__)
/* The host SIMT emulator (tests/r4d_emu) compiles some of these sources as C++: no engine record
 * ever reaches a launch there, so a chained form is its plain form. */
__device__ __forceinline__ void r4d_chain_wait() {}
__device__ __forceinline__ void r4d_chain_done() {}
#else

struct R4dChainRec {
    unsigned long long  magic, target;
    unsigned long long* ctr;
    unsigned long long  check;
};

// The record, or null where it is not the engine's. Read from the argument segment (scalar loads)
// at both ends of the kernel, so nothing is held across the body: a register-heavy body (the
// split-KV decode attention) would otherwise pay for the pointer.
__device__ __forceinline__ const R4dChainRec* r4d_chain_rec() {
    const R4dChainRec* r = (const R4dChainRec*)((const char*)__builtin_amdgcn_kernarg_segment_ptr() +
                                                RAD_CHAIN_KARG_OFF);
    const unsigned long long magic = r->magic, target = r->target, check = r->check;
    const unsigned long long* ctr = r->ctr;
    if (magic != RAD_CHAIN_MAGIC ||
        check != (magic ^ target ^ (unsigned long long)(uintptr_t)ctr ^ RAD_CHAIN_CHECK))
        return nullptr;
    return r;
}

// Before the first read of an operand: one lane waits for the counter to reach the target.
// Every resident block polls the same word, so a wide kernel launched behind a long one (a GEMM
// behind a junction's all-reduce) would put hundreds of loads into one L2 channel every few dozen
// clocks for as long as the wait lasts, slowing whatever else that channel serves. The first
// R4D_CHAIN_FAST polls are ~64 clocks apart, the rest R4D_CHAIN_SLOW x 64: a long wait ends at most
// that much late and puts 1/R4D_CHAIN_SLOW of the load on the channel.
#ifndef R4D_CHAIN_FAST
#define R4D_CHAIN_FAST 8
#endif
#ifndef R4D_CHAIN_SLOW
#define R4D_CHAIN_SLOW 16
#endif
// The acquire behind a chained WAIT (upstream 8f2e511f part 2, ported in tp8 loop R48): the vector caches by the fence, the scalar
// data cache by hand, which the fence leaves. A load whose address every lane shares is a scalar load served from the scalar data
// cache (a GEMM's row scales, a routed kernel's counts and offsets); the packet processor's acquire empties that cache when it
// LAUNCHES the kernel, which for a chained kernel waiting in its prologue is before the kernels in front have written. Upstream,
// R9700, packet acquire on: one launch in 1,800 read a stale shared value without it, none with it. Every wave, for the compute
// unit it runs on. A record with target 0 (lever CIK: launched WITH its barrier bit, behind everything before it) waits for
// nothing and keeps the fence alone: its packet's acquire, where kept, ran after the kernels in front completed; where lever
// CHI / CTA dropped it, lever IKA's scalar-cache rule (Ctx::haz_mark) decided, as for an IKA record.
__device__ __forceinline__ void r4d_chain_acquire() {
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    asm volatile("s_dcache_inv" ::: "memory");
}

__device__ __forceinline__ void r4d_chain_wait() {
    const R4dChainRec* r = r4d_chain_rec();
    if (!r) return;
    /* Target 0 = nothing to wait for (the queue's first chained kernel, or -- lever CIK, DRAFTER round 62 -- a chained kernel the
     * engine issued WITH its barrier bit, whose packet already started behind everything before it): no poll, no block barrier. */
    if (r->target != 0) {
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            const unsigned long long target = r->target;
            unsigned long long* ctr = r->ctr;
            int n = 0;
            while (__hip_atomic_load(ctr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < target) {
                if (n < R4D_CHAIN_FAST) {
                    __builtin_amdgcn_s_sleep(1);
                    ++n;
                } else {
                    __builtin_amdgcn_s_sleep(R4D_CHAIN_SLOW);
                }
            }
        }
        __syncthreads();
        r4d_chain_acquire();   /* tp8 loop R48: upstream 8f2e511f part 2 */
        return;
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}

// After the block's last write: release, then one add a block.
__device__ __forceinline__ void r4d_chain_done() {
    const R4dChainRec* r = r4d_chain_rec();
    if (!r) return;
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        __hip_atomic_fetch_add(r->ctr, 1ull, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}
#endif  /* __HIP__ */

/* The launchers given a chained form here take it only at steps of up to 32 token rows: above that
 * the waiting blocks hold room a wide predecessor needs. Flash-Next on 2x R9700 at MTP depth 5 with
 * the adaptive window, sampled: 32 against 64 rows 214/349/494/654/890 -> 218/355/494/662/893 tok/s
 * at 1/2/4/8/16 streams. */
constexpr long long kR4dChainMaxRows = 32;
inline bool r4d_chain_rows_ok(long long rows) { return rows <= kR4dChainMaxRows; }

/* A launcher that takes no RadArgs reads its row's `chain` here; the plugin
 * ABI entry sets it around the launch only, so a direct C call never chains. */
inline thread_local bool g_r4d_chain_arg = false;
struct R4dChainArg {
    explicit R4dChainArg(bool on) { g_r4d_chain_arg = on; }
    ~R4dChainArg() { g_r4d_chain_arg = false; }
};
