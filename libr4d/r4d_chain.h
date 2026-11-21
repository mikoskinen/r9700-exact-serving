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
#ifndef R4D_CHAIN_WAIT
#define R4D_CHAIN_WAIT 1
#endif
#if R4D_CHAIN_WAIT == 7
// Lever CW7 (qfn-on-ds Qwen loop round 14, QUEUE Q36; ported from the DeepSeek DRAFTER track, ds41-drafter-cw7 628bc72f; CMake
// R4D_CHAIN_WAIT7, default OFF): the same wait in a form that (a) empties the WGP's scalar data cache behind the wait -- a chained
// kernel launches before its predecessor ends, so a line an earlier kernel left in that cache can be older than what the predecessor
// wrote (DRAFTER kit/mes/coh4.hip on an R9700: with the poll above and no invalidate EVERY scalar read of a rewritten buffer was
// stale, 0 with one) -- and (b) does not change the body's code: LLVM's uniform-load analysis takes the atomic poll and s_sleep as
// memory clobbers, so behind the form above uniform loads of values read from memory compile as vector loads; a raw buffer load, a
// sched_barrier and a non-volatile asm with an SGPR output are no clobber. Lane 0 polls the counter at device scope (the load's own
// latency is the back-off), then s_dcache_inv + s_wait_kmcnt in that wave (fed the last polled word, kept by a branch on its
// output) before the block barrier, then the agent acquire as before. OFF (R4D_CHAIN_WAIT 1) = the wait below, unchanged.
__device__ __forceinline__ unsigned long long r4d_chain_peek(const unsigned long long* ctr, unsigned k) {
    const __amdgpu_buffer_rsrc_t rs = __builtin_amdgcn_make_buffer_rsrc((void*)ctr, (short)0, 8, 0x31004000);
    /* soffset (k >> 31): 0 for 2^31 polls but not provably so, so the load is not loop-invariant; past 8 bytes the bound reads 0 */
    const auto v = __builtin_amdgcn_raw_buffer_load_b64(rs, 0, (int)(k >> 31), 0x10 /* SCOPE_DEV */);
    return (unsigned long long)v[0] | ((unsigned long long)v[1] << 32);
}
__device__ __forceinline__ void r4d_chain_wait() {
    const R4dChainRec* r = r4d_chain_rec();
    if (!r) return;
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
}
#else
__device__ __forceinline__ void r4d_chain_wait() {
    const R4dChainRec* r = r4d_chain_rec();
    if (!r) return;
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
