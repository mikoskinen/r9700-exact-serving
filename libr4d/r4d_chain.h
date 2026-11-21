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

#ifndef R4D_CHAIN_WAIT
#define R4D_CHAIN_WAIT 1
#endif
#if R4D_CHAIN_WAIT == 7
// Lever CW7 (DRAFTER track round 66; CMake R4D_CHAIN_WAIT7): the same wait and acquire in a form that does not change the body's
// code. LLVM's uniform-load analysis (amdgpu.noclobber) takes an atomic load, an s_sleep, a volatile asm and the s_dcache_inv
// builtin as memory clobbers (fences, s_barrier and sched_barrier not): behind form 1 every later load of a pointer read from
// memory stops being a scalar load (exl3r4d's routed-expert GEMM: waterfall loops round each weight load, +56..+68 % at 16-48 rows,
// DRAFTER round 65; this library: ds41_index_score's scalar forms 145-166 -> 256 VGPRs + spills, xm_seq / xm_gemv, post_collapse_norm
// 8-stream, every IKA kernel's entry +2 vector loads -- PREFILL NOTE 13:20). Here lane 0 polls by a raw buffer load at device
// scope (a read-only intrinsic), the loop kept by a sched_barrier (a side effect the analysis ignores); the load's own latency
// (~0.5-1 us at L2) is the back-off. The scalar data cache is then emptied by the same wave through a NON-volatile asm with an
// SGPR output (readnone to the IR: no MemoryDef), fed the last polled word (so it follows the loop) and kept by a branch on its
// output; s_wait_kmcnt inside it, so the block barrier releases the other waves only after the invalidate completed. Every wave of
// a block runs on the WGP whose scalar cache that wave emptied (cohbench3 r50d: wave 0 + barrier s_dcache_inv 0 stale; the chained
// case: drafter/kit/mes/coh4.hip).
__device__ __forceinline__ unsigned long long r4d_chain_peek(const unsigned long long* ctr, unsigned k) {
    const __amdgpu_buffer_rsrc_t rs = __builtin_amdgcn_make_buffer_rsrc((void*)ctr, (short)0, 8, 0x31004000);
    /* soffset (k >> 31): 0 for 2^31 polls but not provably so, so the load is not loop-invariant; past 8 bytes the bound reads 0 */
    const auto v = __builtin_amdgcn_raw_buffer_load_b64(rs, 0, (int)(k >> 31), 0x10 /* SCOPE_DEV */);
    return (unsigned long long)v[0] | ((unsigned long long)v[1] << 32);
}
__device__ __forceinline__ void r4d_chain_wait() {
    const R4dChainRec* r = r4d_chain_rec();
    if (!r) return;
    if (r->target != 0) {
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            const unsigned long long target = r->target;
            const unsigned long long* ctr = r->ctr;
            unsigned long long v;
            unsigned k = 0;
            while ((v = r4d_chain_peek(ctr, k)) < target) {
                ++k;
                __builtin_amdgcn_sched_barrier(0);
            }
            unsigned z;
            asm("s_dcache_inv\n\ts_wait_kmcnt 0x0\n\ts_mov_b32 %0, 0" : "=s"(z) : "s"((unsigned)v));
            if (z) __builtin_amdgcn_sched_barrier(0);
        }
        __syncthreads();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        return;
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}
#else
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
#endif  /* R4D_CHAIN_WAIT */

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
