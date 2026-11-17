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
 * card's wall clock rate), or a header that is not this step's, or a trailer that is not the payload
 * waited for (review F6), stores `abort` = 2 / 3 / 4 / 5 into the ring and copies nothing; the core checks the abort words after the step's readback (the leader)
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
#include <map>
#include <mutex>
#include <tuple>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 512;

struct Parts {
    uint64_t p[RAD_PP_MAX_PARTS];      /* device addresses of the parts */
    uint64_t bytes[RAD_PP_MAX_PARTS];  /* each a multiple of 4 (checked on the host) */
    uint32_t w16;                      /* bit k: part k is 16-byte words at a 16-byte address (else 4-byte words) */
    uint64_t layout;                   /* rad_pp_layout of the parts' bytes: the trailer's second word */
    uint64_t off[RAD_PP_MAX_PARTS];    /* each part's offset in the payload */
    uint64_t total;                    /* aligned payload bytes */
    int32_t  n;
    int32_t  rank;
    uint64_t rows;
    uint64_t ticks;                    /* the deadline in wall-clock ticks, 0 none */
    int32_t  loop;                     /* tp8 dev M4: 1 = a segment boundary on one node (the ring is this rank's buffer) */
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

/* One part, grid-strided: 16-byte words where the part is in them at a 16-byte address (the residual stream's bf16 /
 * f32 rows), else 4-byte words (tp8 S6: a compressor's i32 pages, the keep words, rows of any count). The ring side is
 * always 256-byte aligned. */
__device__ inline void copy_part(const char* src, char* dst, uint64_t bytes, uint32_t w16, uint64_t stride) {
    if (w16) {
        const uint4* s = reinterpret_cast<const uint4*>(src);
        uint4* d = reinterpret_cast<uint4*>(dst);
        const uint64_t words = bytes / 16;
        for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < words; i += stride) d[i] = s[i];
        return;
    }
    const uint32_t* s = reinterpret_cast<const uint32_t*>(src);
    uint32_t* d = reinterpret_cast<uint32_t*>(dst);
    const uint64_t words = bytes / 4;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < words; i += stride) d[i] = s[i];
}

__global__ void __launch_bounds__(kThreads) stage_send_kernel(RadPpRing* r, Parts P, uint32_t* counter) {
    __shared__ int ok;
    if (threadIdx.x == 0) {
        if (P.loop) {
            ok = 1;   /* tp8 dev M4: its own buffer, written behind this rank's last receive on the same stream */
        } else {
            const uint64_t rd = ld_sys(&r->ready);
            ok = wait_ring(r, P.ticks, [&] { return ld_sys(&r->sent) >= rd; }) ? 1 : 0;
        }
    }
    __syncthreads();
    if (ok) {
        char* dst = reinterpret_cast<char*>(r) + RAD_PP_RING_HDR;
        const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
        for (int k = 0; k < P.n; ++k) copy_part(reinterpret_cast<const char*>(P.p[k]), dst + P.off[k], P.bytes[k],
                                                (P.w16 >> k) & 1u, stride);
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t done = atomicAdd(counter, 1u) + 1u;
        if (done == gridDim.x) {
            *counter = 0u;
            if (ok) {
                /* review F6: the payload's number in its trailer, before `ready`; tp8 dev M4 loop: a fresh header, number 1,
                 * this rank's receive the only reader */
                if (P.loop) {
                    st_sys(&r->abort, 0ull);
                    st_sys(&r->taken[P.rank], 0ull);
                    st_sys(&r->ready, 0ull);
                    r->n_ranks = RAD_PP_MAX_RANKS;
                    r->deadline_ms = 0;
                }
                const uint64_t seq = ld_sys(&r->ready) + 1;
                st_sys(reinterpret_cast<volatile uint64_t*>(reinterpret_cast<char*>(r) + RAD_PP_RING_HDR + P.total + 8),
                       P.layout);   /* tp8 S6: the layout word, before the number */
                st_sys(reinterpret_cast<volatile uint64_t*>(reinterpret_cast<char*>(r) + RAD_PP_RING_HDR + P.total), seq);
                r->rows = P.rows;
                r->parts = (uint64_t)P.n;
                for (int k = 0; k < P.n; ++k) r->part_bytes[k] = P.bytes[k];
                r->bytes = P.total + RAD_PP_TRAILER;
                __threadfence_system();
                if (P.loop) {
                    st_sys(&r->sent, seq);
                    st_sys(&r->landed, seq);
                }
                st_sys(&r->ready, seq);
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
        if (ok && (r->rows != P.rows || r->parts != (uint64_t)P.n || r->bytes != P.total + RAD_PP_TRAILER)) {
            st_sys(&r->abort, 4ull);
            ok = 0;
        }

        /* review F6: the payload waited for, not a stale one of the same shape (its trailer, one
         * system-scope load after the acquire of `landed`) */
        if (ok && ld_sys(reinterpret_cast<const volatile uint64_t*>(reinterpret_cast<const char*>(r) + RAD_PP_RING_HDR +
                                                                     P.total)) != want) {
            st_sys(&r->abort, 5ull);
            ok = 0;
        }
        /* tp8 S6: parts carry their own rows, so the sender's layout word (each part's bytes) must be this node's */
        if (ok && ld_sys(reinterpret_cast<const volatile uint64_t*>(reinterpret_cast<const char*>(r) + RAD_PP_RING_HDR +
                                                                     P.total + 8)) != P.layout) {
            st_sys(&r->abort, 4ull);
            ok = 0;
        }
    }
    __syncthreads();
    if (ok) {
        const char* src = reinterpret_cast<const char*>(r) + RAD_PP_RING_HDR;
        const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
        for (int k = 0; k < P.n; ++k) copy_part(src + P.off[k], reinterpret_cast<char*>(P.p[k]), P.bytes[k],
                                                (P.w16 >> k) & 1u, stride);
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

/* One 4-byte counter a (device, ring, op): the blocks of a launch count themselves in it and the last
 * resets it, so consecutive launches on one ring share it. Keyed by the ring too (review robustness
 * 7): two rings' stage ops in flight on one card at once (S5's lanes, each with its own rings, on
 * their own streams) never count into one counter, so no launch can see another's blocks as its own.
 * Allocated once a (card, ring), never freed (a few bytes for the process's life). */
uint32_t* counter_for(int dev, const void* ring, int which) {
    static std::mutex mu;
    static std::map<std::tuple<int, const void*, int>, uint32_t*> c;
    if (dev < 0) return nullptr;
    std::lock_guard<std::mutex> lk(mu);
    uint32_t*& slot = c[std::make_tuple(dev, ring, which)];
    if (!slot) {
        void* p = nullptr;
        if (hipMalloc(&p, 64) != hipSuccess) return nullptr;
        if (hipMemset(p, 0, 64) != hipSuccess) return nullptr;
        slot = static_cast<uint32_t*>(p);
    }
    return slot;
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
int collect(const RadArgs* a, const RadPpRing* r, uint64_t ring_bytes, Parts* P) {
    P->n = 0;
    P->total = 0;
    P->w16 = 0;
    int64_t rows = -1;
    for (int i = 1; i < a->n_t && i <= RAD_PP_MAX_PARTS; ++i) {
        const RadTensor* t = r4d_opt(a, i);
        if (!t) break;
        if (t->rank < 1 || !r4d_contig(t)) return RAD_E_STRIDE;
        if (rows < 0) rows = t->shape[0];   /* part 0's: tp8 S6 parts carry their own rows */
        const uint64_t b = (uint64_t)rad_dtype_bytes(t->dtype, r4d_numel(t));
        if (b % 4 || (r4d_addr(t) & 3)) return RAD_E_ALIGN;
        if (b % 16 == 0 && (r4d_addr(t) & 15) == 0) P->w16 |= 1u << P->n;
        P->p[P->n] = (uint64_t)r4d_addr(t);
        P->bytes[P->n] = b;
        P->off[P->n] = P->total;
        P->total += rad_pp_align(b);
        ++P->n;
    }
    if (P->n == 0 || rows < 0) return RAD_E_SHAPE;
    P->layout = rad_pp_layout(P->bytes, P->n);
    /* tp8 dev M4 loop: the ring is this rank's buffer in card memory, its extent the capacity (never read here) */
    const uint64_t cap = P->loop ? ring_bytes - RAD_PP_RING_HDR : r->cap;
    if (P->total + RAD_PP_TRAILER > cap) return RAD_E_FULL;
    P->rows = (uint64_t)rows;
    return RAD_OK;
}

int launch(const RadArgs* a, RadStream stream, bool send) {
    const RadTensor* rt = r4d_opt(a, 0);
    RadPpRing* r = rt ? static_cast<RadPpRing*>(rt->data) : nullptr;
    /* tp8 dev M4: `loop` 1 -- the ring is a buffer of this rank in card memory (a segment boundary on one node): its header
     * is never read on the host, every rank sends into its own */
    const bool loop = r4d_geti_or(a, "loop", 0) == 1;
    const uint64_t ring_bytes = loop && rt ? (uint64_t)rad_dtype_bytes(rt->dtype, r4d_numel(rt)) : 0;
    if (!r || (!loop && r->magic != RAD_PP_RING_MAGIC)) return R4D_NO(RAD_E_STATE);
    if (loop && ring_bytes < RAD_PP_RING_HDR + RAD_PP_TRAILER) return R4D_NO(RAD_E_FULL);
    if (send && !loop && a->rank != (int)r4d_geti_or(a, "sender", 0)) return RAD_OK;
    if (!send && (a->rank < 0 || a->rank >= RAD_PP_MAX_RANKS || (!loop && (uint64_t)a->rank >= r->n_ranks)))
        return R4D_NO(RAD_E_STATE);
    Parts P{};
    P.loop = loop ? 1 : 0;
    const int st = collect(a, r, ring_bytes, &P);
    if (st < 0) return R4D_NO(st);
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return R4D_NO(RAD_E_DEVICE);
    P.rank = a->rank;
    P.ticks = loop ? 0 : r->deadline_ms ? r->deadline_ms * ticks_per_ms(dev) : 0;
    uint32_t* counter = counter_for(dev, r, send ? 0 : 1);
    if (!counter) return R4D_NO(RAD_E_NOMEM);
    const uint64_t words = P.total / (P.w16 == (1u << P.n) - 1u ? 16 : 4);
    uint32_t blocks = (uint32_t)((words + kThreads * 4 - 1) / (kThreads * 4));
    if (blocks < 1) blocks = 1;
    if (blocks > kMaxBlocks) blocks = kMaxBlocks;
    return send ? R4D_EMIT(stream, stage_send_kernel, dim3(blocks), dim3(kThreads), 0, r, P, counter)
                : R4D_EMIT(stream, stage_recv_kernel, dim3(blocks), dim3(kThreads), 0, r, P, counter);
}

}  /* namespace */

extern "C" int r4d_rad_stage_send(const RadArgs* a, RadStream stream) { return launch(a, stream, true); }
extern "C" int r4d_rad_stage_recv(const RadArgs* a, RadStream stream) { return launch(a, stream, false); }
