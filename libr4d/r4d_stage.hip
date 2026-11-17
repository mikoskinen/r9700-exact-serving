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
#include <cstdlib>
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
    /* tp8 loop P8 (RadPpRing::send_split, a send ring only): 0 rank `sender` copies the payload; 1 every rank its share
     * [lo, hi) and rank `sender` publishes; 2 rank `sender` copies it all, the others compare their share */
    int32_t  split;
    int32_t  sender;
    int32_t  nr;                       /* the node's ranks */
    uint64_t lo, hi;                   /* this rank's share of the payload in bytes, 256-byte aligned */
    uint64_t cmp;                      /* (2) 4-byte words of the parts inside [lo, hi) */
    int32_t  idle;                     /* tp8 dev D8: 1 = rank 0's receive adds its wait to the ring (RadPpRing::wait_on) */
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

/* tp8 loop P8: 4-byte words of `src` that differ from the ring's, grid-strided (RADIANCE_PP_SEND_SPLIT=2's check) */
__device__ inline uint32_t cmp_part(const char* src, const char* ring, uint64_t bytes, uint64_t stride) {
    const uint32_t* a = reinterpret_cast<const uint32_t*>(src);
    const volatile uint32_t* b = reinterpret_cast<const volatile uint32_t*>(ring);
    const uint64_t words = bytes / 4;
    uint32_t bad = 0;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < words; i += stride) bad += a[i] != b[i] ? 1u : 0u;
    return bad;
}

/* tp8 loop P8 (RADIANCE_PP_SEND_SPLIT=1 / 2): every rank of the sending node launches. Thread 0 of each block waits for
 * the ring (1, and the publisher of 2: the link has taken the payload before the one this rank works on, `sent` >=
 * its `done`) or for rank `sender`'s full copy (2's other ranks: `staged`); the blocks copy (1) or compare (2) the rank's
 * share [lo, hi) of every part, from the rank's own copy of the parts; the last block of a rank other than `sender`
 * counts the payload in its `done` (2: its differing words first); the last block of `sender` waits for every other
 * rank's `done` and then publishes as stage_send_kernel does (header, trailer, `ready`) and counts its own `done`. */
__global__ void __launch_bounds__(kThreads) stage_send_split_kernel(RadPpRing* r, Parts P, uint32_t* counter,
                                                                    uint32_t* diff) {
    __shared__ int ok;
    __shared__ uint64_t mine;
    const bool pub = P.rank == P.sender;
    if (threadIdx.x == 0) {
        const uint64_t d = ld_sys(&r->done[P.rank]);
        mine = d + 1;
        if (P.split == 1 || pub) ok = wait_ring(r, P.ticks, [&] { return ld_sys(&r->sent) >= d; }) ? 1 : 0;
        else                     ok = wait_ring(r, P.ticks, [&] { return ld_sys(&r->staged) >= d + 1; }) ? 1 : 0;
    }
    __syncthreads();
    if (ok) {
        char* ring = reinterpret_cast<char*>(r) + RAD_PP_RING_HDR;
        const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
        const bool all = P.split == 2 && pub;
        const uint64_t lo = all ? 0 : P.lo, hi = all ? P.total : P.hi;
        for (int k = 0; k < P.n; ++k) {
            const uint64_t a = lo > P.off[k] ? lo : P.off[k];
            const uint64_t e = hi < P.off[k] + P.bytes[k] ? hi : P.off[k] + P.bytes[k];
            if (a >= e) continue;
            const char* src = reinterpret_cast<const char*>(P.p[k]) + (a - P.off[k]);
            if (P.split == 2 && !pub) {
                const uint32_t bad = cmp_part(src, ring + a, e - a, stride);
                if (bad) atomicAdd(diff + k, bad);   /* one counter a part (counter_for's 64 bytes: 16 words) */
            } else {
                copy_part(src, ring + a, e - a, (P.w16 >> k) & 1u, stride);
            }
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t n = atomicAdd(counter, 1u) + 1u;
        if (n == gridDim.x) {
            *counter = 0u;
            if (ok && !pub) {
                if (P.split == 2) {
                    uint64_t dw = 0;
                    for (int k = 0; k < P.n; ++k) {
                        const uint32_t d = atomicExch(diff + k, 0u);
                        dw += d;
                        if (d) st_sys(&r->diff_part[P.rank][k], ld_sys(&r->diff_part[P.rank][k]) + d);
                    }
                    st_sys(&r->diff_words[P.rank], ld_sys(&r->diff_words[P.rank]) + dw);
                    st_sys(&r->cmp_words[P.rank], ld_sys(&r->cmp_words[P.rank]) + P.cmp);
                }
                __threadfence_system();
                st_sys(&r->done[P.rank], mine);
            } else if (ok) {
                if (P.split == 2) st_sys(&r->staged, mine);
                bool all = true;
                for (int q = 0; q < P.nr && all; ++q)
                    if (q != P.sender) all = wait_ring(r, P.ticks, [&] { return ld_sys(&r->done[q]) >= mine; });
                if (all) {
                    const uint64_t seq = mine;
                    st_sys(reinterpret_cast<volatile uint64_t*>(reinterpret_cast<char*>(r) + RAD_PP_RING_HDR + P.total + 8),
                           P.layout);
                    st_sys(reinterpret_cast<volatile uint64_t*>(reinterpret_cast<char*>(r) + RAD_PP_RING_HDR + P.total), seq);
                    r->rows = P.rows;
                    r->parts = (uint64_t)P.n;
                    for (int k = 0; k < P.n; ++k) r->part_bytes[k] = P.bytes[k];
                    r->bytes = P.total + RAD_PP_TRAILER;
                    __threadfence_system();
                    st_sys(&r->done[P.rank], seq);
                    st_sys(&r->ready, seq);
                }
            }
        }
    }
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

/* tp8 loop P10 (RadPpRing::async_rows): the parts packed into the node's card staging buffer at their payload offsets,
 * on the pass's stream; stage_send_kernel then copies the staging buffer's parts into the ring from a side stream. */
__global__ void __launch_bounds__(kThreads) stage_pack_kernel(Parts P, char* stage) {
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (int k = 0; k < P.n; ++k)
        copy_part(reinterpret_cast<const char*>(P.p[k]), stage + P.off[k], P.bytes[k], (P.w16 >> k) & 1u, stride);
}

/* tp8 loop P10b (RadPpRing::recv_rows): the payload's parts out of the rank's card staging buffer, on the pass's stream,
 * once the side stream's stage_recv_kernel copied the ring into it */
__global__ void __launch_bounds__(kThreads) stage_unpack_kernel(Parts P, const char* stage) {
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (int k = 0; k < P.n; ++k)
        copy_part(stage + P.off[k], reinterpret_cast<char*>(P.p[k]), P.bytes[k], (P.w16 >> k) & 1u, stride);
}

__global__ void __launch_bounds__(kThreads) stage_recv_kernel(RadPpRing* r, Parts P, uint32_t* counter) {
    __shared__ int ok;
    if (threadIdx.x == 0) {
        const uint64_t want = ld_sys(&r->taken[P.rank]) + 1;
        const uint64_t w0 = P.idle ? wall_clock64() : 0;
        ok = wait_ring(r, P.ticks, [&] { return ld_sys(&r->landed) >= want; }) ? 1 : 0;
        /* tp8 dev D8 (RADIANCE_PP_NODE_IDLE=1): the card's wait for the other node, block 0 of rank 0 (one writer, in stream
         * order); two host-memory round trips a receive, only when asked */
        if (P.idle && blockIdx.x == 0) {
            const uint64_t dt = wall_clock64() - w0;
            st_sys(&r->wait_ticks, ld_sys(&r->wait_ticks) + dt);
            st_sys(&r->waits, ld_sys(&r->waits) + 1);
        }
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

/* tp8 loop P10 (RadPpRing::async_rows; the core issues such passes live, so nothing here is recorded): a card's side
 * stream and its two events -- `packed` (the pack on the pass's stream is done: the side stream may read the staging
 * buffer) and `copied` (the side stream's copy into the ring is done: the next pack may write it). One staging buffer a
 * node, so consecutive async payloads of either lane take turns on it. */
struct AsyncCard {
    hipStream_t side = nullptr;
    hipEvent_t packed = nullptr, copied = nullptr;
    bool copied_once = false;
    /* tp8 loop P10b: `staged` (the side stream's ring copy is done: the pass's stream may unpack), `unpacked` (the
     * unpack is done: the next side-stream copy may write the staging buffer) */
    hipEvent_t staged = nullptr, unpacked = nullptr;
    bool unpacked_once = false;
    /* its own side stream: a receive queued there waits on the card for a payload to land, which must never hold up a
     * send behind it (the peer may be waiting for that send) */
    hipStream_t side_rx = nullptr;
};

AsyncCard* async_card(int dev) {
    static std::mutex mu;
    static std::map<int, AsyncCard> cards;
    std::lock_guard<std::mutex> lk(mu);
    AsyncCard& c = cards[dev];
    if (!c.side) {
        if (hipStreamCreateWithFlags(&c.side, hipStreamNonBlocking) != hipSuccess) { c.side = nullptr; return nullptr; }
        if (hipEventCreateWithFlags(&c.packed, hipEventDisableTiming) != hipSuccess ||
            hipEventCreateWithFlags(&c.copied, hipEventDisableTiming) != hipSuccess ||
            hipEventCreateWithFlags(&c.staged, hipEventDisableTiming) != hipSuccess ||
            hipEventCreateWithFlags(&c.unpacked, hipEventDisableTiming) != hipSuccess)
            return nullptr;
        if (hipStreamCreateWithFlags(&c.side_rx, hipStreamNonBlocking) != hipSuccess) { c.side_rx = nullptr; return nullptr; }
    }
    return &c;
}

/* Side-stream blocks: the copy is PCIe-bound (~27 GB/s a card), so a few blocks keep up with it and leave the CUs to the
 * next pass (RADIANCE_PP_SEND_ASYNC_BLOCKS, default 64). */
uint32_t async_blocks() {
    static const uint32_t b = [] {
        const char* v = std::getenv("RADIANCE_PP_SEND_ASYNC_BLOCKS");
        const long n = v && *v ? std::strtol(v, nullptr, 10) : 64;
        return (uint32_t)(n >= 1 && n <= kMaxBlocks ? n : 64);
    }();
    return b;
}

int send_async(RadPpRing* r, const Parts& P, int dev, RadStream stream) {
    AsyncCard* c = async_card(dev);
    if (!c) return R4D_NO(RAD_E_DEVICE);
    char* stage = reinterpret_cast<char*>((uintptr_t)r->stage_dev);
    uint32_t* counter = counter_for(dev, r, 3);   /* the side stream's own: never shared with a launch on the pass's */
    if (!counter) return R4D_NO(RAD_E_NOMEM);
    hipStream_t s = (hipStream_t)stream;
    /* the staging buffer is free once the last async copy on this card is done */
    if (c->copied_once && hipStreamWaitEvent(s, c->copied, 0) != hipSuccess) return R4D_NO(RAD_E_DEVICE);
    const uint64_t words = P.total / (P.w16 == (1u << P.n) - 1u ? 16 : 4);
    uint32_t blocks = (uint32_t)((words + kThreads * 4 - 1) / (kThreads * 4));
    if (blocks < 1) blocks = 1;
    if (blocks > kMaxBlocks) blocks = kMaxBlocks;
    int rc = R4D_EMIT(stream, stage_pack_kernel, dim3(blocks), dim3(kThreads), 0, P, stage);
    if (rc < 0) return rc;
    if (hipEventRecord(c->packed, s) != hipSuccess) return R4D_NO(RAD_E_DEVICE);
    if (hipStreamWaitEvent(c->side, c->packed, 0) != hipSuccess) return R4D_NO(RAD_E_DEVICE);
    /* the same send, its parts now the staging buffer's (256-byte aligned offsets: 16-byte words where the bytes are) */
    Parts Q = P;
    Q.w16 = 0;
    for (int k = 0; k < Q.n; ++k) {
        Q.p[k] = (uint64_t)(uintptr_t)(stage + P.off[k]);
        if (Q.bytes[k] % 16 == 0) Q.w16 |= 1u << k;
    }
    uint32_t sb = async_blocks();
    if (sb > blocks) sb = blocks;
    rc = R4D_EMIT((RadStream)c->side, stage_send_kernel, dim3(sb), dim3(kThreads), 0, r, Q, counter);
    if (rc < 0) return rc;
    if (hipEventRecord(c->copied, c->side) != hipSuccess) return R4D_NO(RAD_E_DEVICE);
    c->copied_once = true;
    r->async_sends = r->async_sends + 1;   /* host memory, this rank's thread only */
    return RAD_OK;
}

/* tp8 loop P10b: the receive through rank `rank`'s staging buffer: the ring copy (stage_recv_kernel with the staging
 * buffer as its parts: the same wait, checks and `taken` count) on the side stream, the unpack on the pass's stream */
int recv_async(RadPpRing* r, const Parts& P, int dev, RadStream stream, int rank) {
    AsyncCard* c = async_card(dev);
    if (!c) return R4D_NO(RAD_E_DEVICE);
    char* stage = reinterpret_cast<char*>((uintptr_t)r->recv_stage[rank]);
    uint32_t* counter = counter_for(dev, r, 4);
    if (!counter) return R4D_NO(RAD_E_NOMEM);
    hipStream_t s = (hipStream_t)stream;
    /* the staging buffer is free once this card's last unpack is done */
    if (c->unpacked_once && hipStreamWaitEvent(c->side_rx, c->unpacked, 0) != hipSuccess) return R4D_NO(RAD_E_DEVICE);
    Parts Q = P;
    Q.w16 = 0;
    for (int k = 0; k < Q.n; ++k) {
        Q.p[k] = (uint64_t)(uintptr_t)(stage + P.off[k]);
        if (Q.bytes[k] % 16 == 0) Q.w16 |= 1u << k;
    }
    const uint64_t words = P.total / (P.w16 == (1u << P.n) - 1u ? 16 : 4);
    uint32_t blocks = (uint32_t)((words + kThreads * 4 - 1) / (kThreads * 4));
    if (blocks < 1) blocks = 1;
    if (blocks > kMaxBlocks) blocks = kMaxBlocks;
    uint32_t sb = async_blocks();
    if (sb > blocks) sb = blocks;
    int rc = R4D_EMIT((RadStream)c->side_rx, stage_recv_kernel, dim3(sb), dim3(kThreads), 0, r, Q, counter);
    if (rc < 0) return rc;
    if (hipEventRecord(c->staged, c->side_rx) != hipSuccess) return R4D_NO(RAD_E_DEVICE);
    if (hipStreamWaitEvent(s, c->staged, 0) != hipSuccess) return R4D_NO(RAD_E_DEVICE);
    rc = R4D_EMIT(stream, stage_unpack_kernel, dim3(blocks), dim3(kThreads), 0, P, (const char*)stage);
    if (rc < 0) return rc;
    if (hipEventRecord(c->unpacked, s) != hipSuccess) return R4D_NO(RAD_E_DEVICE);
    c->unpacked_once = true;
    r->recv_async[rank] = r->recv_async[rank] + 1;   /* host memory, this rank's thread only */
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
    /* tp8 loop P8: RADIANCE_PP_SEND_SPLIT (the core sets the send ring's send_split / send_ranks at start; never in loop) */
    const int sender = (int)r4d_geti_or(a, "sender", 0);
    const int split = send && !loop && (r->send_split == 1 || r->send_split == 2) && r->send_ranks >= 2 &&
                              sender >= 0 && (uint64_t)sender < r->send_ranks
                          ? (int)r->send_split : 0;
    if (split && (a->rank < 0 || (uint64_t)a->rank >= r->send_ranks)) return R4D_NO(RAD_E_STATE);
    if (send && !loop && !split && a->rank != sender) return RAD_OK;
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
    /* tp8 dev D8: rank 0's receive times its wait when the core asked (RadPpRing::wait_on) */
    if (!send && !loop && a->rank == 0 && r->wait_on) {
        P.idle = 1;
        r->wait_tpms = ticks_per_ms(dev);
    }
    uint32_t* counter = counter_for(dev, r, send ? 0 : 1);
    if (!counter) return R4D_NO(RAD_E_NOMEM);
    if (split) {
        P.split = split;
        P.sender = sender;
        P.nr = (int32_t)r->send_ranks;
        const uint64_t U = P.total / RAD_PP_PART_ALIGN;   /* the payload's 256-byte units, split evenly over the ranks */
        P.lo = (uint64_t)a->rank * U / (uint64_t)P.nr * RAD_PP_PART_ALIGN;
        P.hi = ((uint64_t)a->rank + 1) * U / (uint64_t)P.nr * RAD_PP_PART_ALIGN;
        for (int k = 0; k < P.n; ++k) {
            const uint64_t lo = P.lo > P.off[k] ? P.lo : P.off[k];
            const uint64_t hi = P.hi < P.off[k] + P.bytes[k] ? P.hi : P.off[k] + P.bytes[k];
            if (lo < hi) P.cmp += (hi - lo) / 4;
        }
        uint32_t* diff = counter_for(dev, r, 2);
        if (!diff) return R4D_NO(RAD_E_NOMEM);
        const bool all = split == 2 && a->rank == sender;
        const uint64_t share = all ? P.total : P.hi - P.lo;
        uint32_t blocks = (uint32_t)((share / 16 + kThreads * 4 - 1) / (kThreads * 4));
        if (blocks < 1) blocks = 1;
        if (blocks > kMaxBlocks) blocks = kMaxBlocks;
        return R4D_EMIT(stream, stage_send_split_kernel, dim3(blocks), dim3(kThreads), 0, r, P, counter, diff);
    }
    /* tp8 loop P10b: a large payload lands through this rank's card staging buffer from a side stream */
    if (!send && !loop && r->recv_rows && P.rows >= r->recv_rows && a->rank >= 0 && a->rank < RAD_PP_MAX_RANKS &&
        r->recv_stage[a->rank] && P.total <= r->recv_cap)
        return recv_async(r, P, dev, stream, a->rank);
    /* tp8 loop P10: a large payload crosses from a side stream through the node's card staging buffer */
    if (send && !loop && r->async_rows && r->stage_dev && P.rows >= r->async_rows && P.total <= r->stage_cap)
        return send_async(r, P, dev, stream);
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
