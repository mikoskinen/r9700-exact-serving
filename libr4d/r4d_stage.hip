/* r4d_stage.hip -- stage_send / stage_recv on the card: the two-node pipeline's crossings (tp8 S4;
 * abi/rad_pp.h, docs/OPS.md "Stage crossings"; libref/ref_stage.cpp is the reference).
 *
 * The ring is the core's, in pinned host memory every card reads and writes over its own link
 * (host-staged both ends: the NIC never touches card memory). One kernel a launch, a grid of copy
 * blocks:
 *
 *   stage_send  every block's thread 0 waits (system-scope acquire, s_sleep back-off) until the
 *               link has taken the last payload (`sent` >= `ready`) or the ring is aborted; the
 *               blocks copy the parts device -> host ring in 16-byte words; each block fences at
 *               system scope and counts itself in a device counter; the last block writes the
 *               header (rows, parts, bytes) and publishes `ready` + 1 with a system-scope release
 *               and resets the counter. Only rank `sender` launches; the others return at once.
 *   stage_recv  every block's thread 0 waits until `landed` > this rank's `taken` (acquire at
 *               system scope, which also invalidates what the block would read stale); thread 0
 *               of block 0 checks the header against the step (rows, parts, bytes) and marks a
 *               mismatch; the blocks copy host ring -> the parts; the last block publishes
 *               `taken[rank]` + 1 (release) and resets the counter.
 *
 * Failure. A wait that sees `abort`, or runs past the ring's deadline (wall_clock64 against the
 * card's wall clock rate), or a header that is not this step's, stores `abort` = 2 / 3 / 4 into the
 * ring and copies nothing; the core checks the abort words after the step's readback (the leader)
 * and before sending a payload (the link's proxy), so a step that crossed nothing is never read as
 * a result. Counters are read at run time, so a recorded pass replays.
 *
 * Status (17.11.2026): compiled for gfx1201, never launched (no card in this round). The host
 * memory's visibility rules this relies on (system-scope acquire on locked host pages; a later
 * kernel's plain loads after it) are the libr4d host-mailbox collectives' (r4d_ar_entry.hip), which
 * run with --p2p off; the first card run checks every word (s4 plan: the TP2 + TP2 loop on gpu03).
 */
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include "r4d_plugin.h"

#include "rad_pp.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <mutex>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 512;

struct Parts {
    uint64_t p[RAD_PP_MAX_PARTS];      /* device addresses of the parts */
    uint64_t bytes[RAD_PP_MAX_PARTS];  /* each a multiple of 16 (checked on the host) */
    uint64_t off[RAD_PP_MAX_PARTS];    /* each part's offset in the payload */
    uint64_t total;                    /* aligned payload bytes */
    int32_t  n;
    int32_t  rank;
    uint64_t rows;
    uint64_t ticks;                    /* the deadline in wall-clock ticks, 0 none */
};

__device__ inline uint64_t ld_sys(const volatile uint64_t* p) {
    return __hip_atomic_load((uint64_t*)p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ inline void st_sys(volatile uint64_t* p, uint64_t v) {
    __hip_atomic_store((uint64_t*)p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

/* Thread 0 of each block: wait until `cond` holds; false (and `abort` set) on abort or deadline. */
template <class Cond>
__device__ bool wait_ring(RadPpRing* r, uint64_t ticks, Cond cond) {
    const uint64_t t0 = wall_clock64();
    uint32_t polls = 0;
    for (;;) {
        if (cond()) return true;
        if (ld_sys(&r->abort)) return false;
        if (ticks && wall_clock64() - t0 > ticks) {
            st_sys(&r->abort, 3ull);
            return false;
        }
        /* s_sleep takes a constant: a short sleep for the first polls (a decode crossing lands within
         * microseconds), the longest after, so a long wait does not hammer the link. */
        if (++polls < 64) __builtin_amdgcn_s_sleep(2);
        else              __builtin_amdgcn_s_sleep(127);
    }
}

__global__ void __launch_bounds__(kThreads) stage_send_kernel(RadPpRing* r, Parts P, uint32_t* counter) {
    __shared__ int ok;
    if (threadIdx.x == 0) {
        const uint64_t rd = ld_sys(&r->ready);
        ok = wait_ring(r, P.ticks, [&] { return ld_sys(&r->sent) >= rd; }) ? 1 : 0;
    }
    __syncthreads();
    if (ok) {
        char* dst = reinterpret_cast<char*>(r) + RAD_PP_RING_HDR;
        const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
        for (int k = 0; k < P.n; ++k) {
            const uint4* s = reinterpret_cast<const uint4*>(P.p[k]);
            uint4* d = reinterpret_cast<uint4*>(dst + P.off[k]);
            const uint64_t words = P.bytes[k] / 16;
            for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < words; i += stride)
                d[i] = s[i];
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t done = atomicAdd(counter, 1u) + 1u;
        if (done == gridDim.x) {
            *counter = 0u;
            if (ok) {
                r->rows = P.rows;
                r->parts = (uint64_t)P.n;
                for (int k = 0; k < P.n; ++k) r->part_bytes[k] = P.bytes[k];
                r->bytes = P.total;
                __threadfence_system();
                st_sys(&r->ready, ld_sys(&r->ready) + 1);
            }
        }
    }
}

__global__ void __launch_bounds__(kThreads) stage_recv_kernel(RadPpRing* r, Parts P, uint32_t* counter) {
    __shared__ int ok;
    if (threadIdx.x == 0) {
        const uint64_t want = ld_sys(&r->taken[P.rank]) + 1;
        ok = wait_ring(r, P.ticks, [&] { return ld_sys(&r->landed) >= want; }) ? 1 : 0;
        /* This step's payload, or the nodes are out of step. */
        if (ok && (r->rows != P.rows || r->parts != (uint64_t)P.n || r->bytes != P.total)) {
            st_sys(&r->abort, 4ull);
            ok = 0;
        }
    }
    __syncthreads();
    if (ok) {
        const char* src = reinterpret_cast<const char*>(r) + RAD_PP_RING_HDR;
        const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
        for (int k = 0; k < P.n; ++k) {
            const uint4* s = reinterpret_cast<const uint4*>(src + P.off[k]);
            uint4* d = reinterpret_cast<uint4*>(P.p[k]);
            const uint64_t words = P.bytes[k] / 16;
            for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < words; i += stride)
                d[i] = s[i];
        }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t done = atomicAdd(counter, 1u) + 1u;
        if (done == gridDim.x) {
            *counter = 0u;
            if (ok) st_sys(&r->taken[P.rank], ld_sys(&r->taken[P.rank]) + 1);
        }
    }
}

/* One 4-byte counter a (device, op): the blocks of a launch count themselves in it and the last
 * resets it, so consecutive launches on one stream share it. Allocated once a card, never freed
 * (a few bytes for the process's life). */
uint32_t* counter_for(int dev, int which) {
    static std::mutex mu;
    static uint32_t* c[64][2] = {};
    if (dev < 0 || dev >= 64) return nullptr;
    std::lock_guard<std::mutex> lk(mu);
    if (!c[dev][which]) {
        void* p = nullptr;
        if (hipMalloc(&p, 64) != hipSuccess) return nullptr;
        if (hipMemset(p, 0, 64) != hipSuccess) return nullptr;
        c[dev][which] = static_cast<uint32_t*>(p);
    }
    return c[dev][which];
}

uint64_t ticks_per_ms(int dev) {
    static uint64_t t[64] = {};
    if (dev < 0 || dev >= 64) return 100000;
    if (!t[dev]) {
        int khz = 0;
        if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, dev) != hipSuccess || khz <= 0)
            khz = 100000;   /* 100 MHz, gfx12's */
        t[dev] = (uint64_t)khz;
    }
    return t[dev];
}

/* The parts after the ring, as the kernel takes them. */
int collect(const RadArgs* a, const RadPpRing* r, Parts* P) {
    P->n = 0;
    P->total = 0;
    int64_t rows = -1;
    for (int i = 1; i < a->n_t && i <= RAD_PP_MAX_PARTS; ++i) {
        const RadTensor* t = r4d_opt(a, i);
        if (!t) break;
        if (t->rank < 1 || !r4d_contig(t)) return RAD_E_STRIDE;
        if (rows < 0) rows = t->shape[0];
        else if (t->shape[0] != rows) return RAD_E_SHAPE;
        const uint64_t b = (uint64_t)rad_dtype_bytes(t->dtype, r4d_numel(t));
        if (b % 16 || (r4d_addr(t) & 15)) return RAD_E_ALIGN;
        P->p[P->n] = (uint64_t)r4d_addr(t);
        P->bytes[P->n] = b;
        P->off[P->n] = P->total;
        P->total += rad_pp_align(b);
        ++P->n;
    }
    if (P->n == 0 || rows < 0) return RAD_E_SHAPE;
    if (P->total > r->cap) return RAD_E_FULL;
    P->rows = (uint64_t)rows;
    return RAD_OK;
}

int launch(const RadArgs* a, RadStream stream, bool send) {
    const RadTensor* rt = r4d_opt(a, 0);
    RadPpRing* r = rt ? static_cast<RadPpRing*>(rt->data) : nullptr;
    if (!r || r->magic != RAD_PP_RING_MAGIC) return R4D_NO(RAD_E_STATE);
    if (send && a->rank != (int)r4d_geti_or(a, "sender", 0)) return RAD_OK;
    if (!send && (a->rank < 0 || a->rank >= RAD_PP_MAX_RANKS || (uint64_t)a->rank >= r->n_ranks))
        return R4D_NO(RAD_E_STATE);
    Parts P{};
    const int st = collect(a, r, &P);
    if (st < 0) return R4D_NO(st);
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return R4D_NO(RAD_E_DEVICE);
    P.rank = a->rank;
    P.ticks = r->deadline_ms ? r->deadline_ms * ticks_per_ms(dev) : 0;
    uint32_t* counter = counter_for(dev, send ? 0 : 1);
    if (!counter) return R4D_NO(RAD_E_NOMEM);
    const uint64_t words = P.total / 16;
    uint32_t blocks = (uint32_t)((words + kThreads * 4 - 1) / (kThreads * 4));
    if (blocks < 1) blocks = 1;
    if (blocks > kMaxBlocks) blocks = kMaxBlocks;
    return send ? R4D_EMIT(stream, stage_send_kernel, dim3(blocks), dim3(kThreads), 0, r, P, counter)
                : R4D_EMIT(stream, stage_recv_kernel, dim3(blocks), dim3(kThreads), 0, r, P, counter);
}

}  /* namespace */

extern "C" int r4d_rad_stage_send(const RadArgs* a, RadStream stream) { return launch(a, stream, true); }
extern "C" int r4d_rad_stage_recv(const RadArgs* a, RadStream stream) { return launch(a, stream, false); }
