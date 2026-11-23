// kv_resolve: turn a streamed KV group's selected block ids into the ring slots that hold them,
// copy a block in from the pinned host mirror when it is absent, and rewrite the selection in place
// (docs/OPS.md "kv_resolve", docs/KV-STREAMING.md §3.4). `sel` is the only place the sparse path
// names KV storage, so after this op the attention reads ring slots unchanged.
//
// libref/ref_kvstream.cpp is the contract and its seven steps are kept, so both place the same
// blocks in the same slots. Here:
//
//   1. the epoch is bumped;
//   2. one lookup per entry below nsel[q]. A hit stamps its slot with the epoch; the thread whose
//      exchange stamps it first bumps its heat counter and counts it into the union. A miss claims
//      its map entry once (atomicCAS -1 -> -2) and appends the block to the claim list, so the CAS
//      count is the distinct-claim count. The value read is kept for step 7 (the entry cache);
//   3. the claim list is sorted ascending (bitonic, in LDS), so atomic claim order never reaches
//      the placement;
//   4. one victim per miss from the hand: the block's own slot when free, else a free slot, else
//      CLOCK over the heat counter. Phase 3b counts the free slots once in parallel, so a full ring
//      goes straight to CLOCK;
//   5. both maps are re-pointed as each victim is taken;
//   6. each placed block is copied, one warp a miss, so every miss's host read is in flight at once;
//   7. every valid entry is rewritten as its slot.
//
// Overflow: unplaced claims revert to -1, ctl[3] latches, and their entries are rewritten as slot 0
// (the id would be an address past the ring, which attn_paged dereferences unchecked). The row is
// stateful, so it is not in r4d_rows.cpp's concurrent list; it is recordable, since the maps, ring
// and mirror never move.
//
// Cost. A decode step issues this once per QSA layer, so a launch's cost is its misses (tens a
// launch at 16K-160K of history), and nothing it does is sized by `blocks`: the claim list comes
// out of the lookup pass, the union is counted on first stamps, and the copies run in parallel.
// What remains is the host read, PCIe transactions on mirror pages the card has not touched. The
// sweep visits about one slot a miss, since each launch halves the counters it passes.
//
// The claim list. A call claims at most `M * sel_pitch` blocks (in fact at most `lookups`), so the
// list is `next_pow2(min(M * sel_pitch, LDS room))` ids of dynamic LDS (kv_claim_cap): 4096 for
// four sequences at `sel_pitch` 513. Only the first `slots` claims can be placed. A launch wider
// than the list (28 rows or more at pitch 513) gathers its claims from the selection into the row's
// scratch, padded to a power of two, and sorts them there: still `lookups` entries, never `blocks`.
// Only a call claiming more blocks than the ring holds, which has overflowed, walks the map to put
// the smallest `slots` claims in order. ctl[12] records the largest claim count.
//
// The entry cache. Steps 2 and 7 both read `res[b]` for every entry, scattered over the map. A
// hit's slot cannot move during the call, because the sweep never takes a slot stamped with this
// call's epoch, so step 2 keeps what it read (one int an entry, dynamic LDS after the claim list,
// when it fits) and step 7 reads the map again only for entries that were absent. Same entry index,
// same thread, so no barrier between them.
//
// Scratch: `next_pow2(slots)` block ids, then `slots` slots. State faults: libref refuses them
// before writing; a kernel cannot return a status, so here an out-of-range hand starts at 0, a map
// value outside its domain addresses nothing, an out-of-range id stays in `sel` and nsel is clamped
// to `sel_pitch`. A legal call is bit-identical to the reference.
//
// The shim takes the row count, row pitch and ring depth from the operands (`sel`, `slot_block`),
// with the ranged `M`, `sel_pitch` and `slots` as ceilings (arch/common/rad_arch.h, "step-sized
// operands"), and addresses the byte planes by row pitch, so one kernel serves every cache layout.
// It returns a RAD status rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_launch.h"

#include <hip/hip_runtime.h>
#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace {

constexpr int kThreads = 1024;
/* fn4 kvs_vic (qwen/KV-STREAM-EXACT.md): the free list is built when more slots than this are occupied (below it the
 * serial scan skips little). The switch (rad_fn4_switch("kvs_vic"), RADIANCE_FN4_KVS_VIC at start; 0 = today's serial
 * victim search) is the launch's `vic`. */
constexpr int kVicListMin = 256;

/* A 16-byte vector as a builtin type: `uint4` copies through a memcpy call, which keeps a load and
 * its store adjacent, and the miss copy needs its four loads issued first (step 6). */
typedef unsigned int kv_v16 __attribute__((ext_vector_type(4)));

/* The next power of two at or above `n`, clamped to what an i32 can index: the sort pads to one and
 * the row's scratch is sized from the same expression. */
static long long kv_pow2ceil(long long n) {
    long long p = 1;
    while (p < n && p <= (1LL << 30)) p <<= 1;
    return p;
}

/* The claim list's capacity in ids, from the launch's operands and the device's LDS room (header,
 * "The claim list"); 0 when a workgroup has no room, which the shim refuses. `*room_out` is the
 * dynamic LDS room in ints, for the entry cache. */
static int kv_claim_cap(long long M, long long pitch, int* room_out) {
    /* Static LDS: the two reduction arrays and the scalars. Everything else is dynamic. */
    const size_t stat = (size_t)kThreads * 2 * sizeof(int) + 64;
    static const int room = [] {
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess) return 0;
        hipDeviceProp_t pr{};
        if (hipGetDeviceProperties(&pr, dev) != hipSuccess) return 0;
        if (pr.sharedMemPerBlock <= stat) return 0;
        size_t r = (pr.sharedMemPerBlock - stat) / sizeof(int);
        size_t p = 1;
        while (p * 2 <= r) p *= 2;      /* a power of two: the sort pads to one, and must not pad
                                         * past the buffer */
        return (int)p;
    }();
    *room_out = room;
    if (room < 1) return 0;
    /* The largest power of two the room holds: the sort pads the list to one. */
    int top = 1;
    while (top <= room / 2) top *= 2;
    const long long bound = M * pitch;   /* the call's claim ceiling */
    int p = 1;
    while ((long long)p < bound && p < top) p <<= 1;
    return p;
}

/* The heat counter's two writes, as libref's heat_up/heat_down: a use bumps by one and saturates
 * at 3, a sweep pass halves, and a negative value (never written by a legal call) reads as cold. */
__device__ __forceinline__ int kv_heat_up(int h) { return h < 0 ? 1 : (h < 3 ? h + 1 : 3); }
__device__ __forceinline__ int kv_heat_down(int h) { return h > 0 ? (h >> 1) : 0; }

/* Sort `n` distinct claim ids ascending, padded to a power of two with INT32_MAX. The caller's
 * buffer: LDS for the claim list, the row's scratch for a launch wider than it. */
__device__ __forceinline__ void kv_sort_claims(int* list, int n, int tid) {
    int n2 = 1;
    while (n2 < n) n2 <<= 1;
    for (int i = n + tid; i < n2; i += kThreads) list[i] = INT32_MAX;
    __syncthreads();
    for (int k = 2; k <= n2; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < n2; i += kThreads) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const int a = list[i], b = list[ixj];
                    /* Ascending overall: a pair whose lower index sits in the first half of a
                     * k-block sorts up, the other half sorts down (bitonic). */
                    const bool up = (i & k) == 0;
                    if (up ? a > b : a < b) { list[i] = b; list[ixj] = a; }
                }
            }
            __syncthreads();
        }
    }
}


/* fn4 kvs_vic: an order-keeping compaction of the ring in the hand's rotated order -- slot q = (h0 + r) mod slots for
 * r = 0 .. slots-1 -- of the slots `pred` accepts, into out[0 .. cap); returns how many it accepts (all of them, also
 * past cap), the same in every thread. Thread t owns the rotated positions [t * ch, (t + 1) * ch), so the list keeps the
 * sweep's order. `pred` must read state nothing writes during the call. s_cnt / s_off: kThreads ints each. */
template <class P>
__device__ __forceinline__ int kv_compact_rot(int h0, int slots, int* out, int cap, int tid, int* s_cnt, int* s_off,
                                              P pred) {
    const int ch = (slots + kThreads - 1) / kThreads;
    const long long lo0 = (long long)tid * ch;
    const int lo = lo0 < slots ? (int)lo0 : slots;
    const int hi = lo + ch < slots ? lo + ch : slots;
    int c = 0;
    for (int r = lo; r < hi; ++r) {
        int q = h0 + r;
        if (q >= slots) q -= slots;
        if (pred(q)) ++c;
    }
    s_cnt[tid] = c;
    s_off[tid] = c;
    __syncthreads();
    for (int step = 1; step < kThreads; step <<= 1) {
        const int v = tid >= step ? s_off[tid - step] : 0;
        __syncthreads();
        s_off[tid] += v;
        __syncthreads();
    }
    int pos = s_off[tid] - c;
    const int total = s_off[kThreads - 1];
    if (pos < cap) {
        for (int r = lo; r < hi && pos < cap; ++r) {
            int q = h0 + r;
            if (q >= slots) q -= slots;
            if (pred(q)) out[pos++] = q;
        }
    }
    __syncthreads();
    return total;
}

/* fn4 kvs_flat (Qwen loop round 52, QUEUE Q101; rad_fn4_switch("kvs_flat"), RADIANCE_FN4_KVS_FLAT at start; 0 = today):
 * steps 2 and 7 over the flat entry space, kFlatU entries a thread in flight. The entry's block, or -1 when the entry is
 * past the launch, past its row's nsel, or names no block of the id space (the row loop's `continue`s). */
constexpr int kFlatU = 8;
__device__ __forceinline__ int kv_flat_block(const int32_t* __restrict__ sel, const int32_t* __restrict__ nsel,
                                             int e, int E, int pitch, int blocks) {
    if (e >= E) return -1;
    /* 32-bit: the shim takes this form only while M * pitch fits an int */
    const int q = (int)((unsigned)e / (unsigned)pitch), i = e - q * pitch;
    int n = nsel[q];
    if (n < 0) n = 0;
    if (n > pitch) n = pitch;
    if (i >= n) return -1;
    const int b = sel[e];
    return (unsigned)b < (unsigned)blocks ? b : -1;
}

/* Qwen loop round 53 (QUEUE Q105, bench only): Stamp = thread 0 writes wall_clock64 (100 MHz) at the phase ends into the
 * row scratch past the victim list (kv_st_base), and the call's counts after them; reachable only through r4d_kv_stamp_set
 * (r4d_selftest --perfkvres): the served instances (Stamp false) compile as before. A phase without a barrier at its end gets
 * one in the stamped instance only (KV_STB). Slots: 0 entry, 1 epoch, 2 lookups + hit count, 3 sort, 4 free count (3b),
 * 5 free list (4a), 6 serial step 4, 7 4b check pass, 8 4b rotations + batch re-point, 9 4b end, 10 revert, 11 copies,
 * 12 rewrite, 13 end; 16.. counts (claims, nmiss, placed, hits, free, kc, vser, lookups). */
#define KV_ST(k)  do { if constexpr (Stamp) { if (tid == 0) kst[(k)] = kv_now(); } } while (0)
#define KV_STB(k) do { if constexpr (Stamp) { __syncthreads(); if (tid == 0) kst[(k)] = kv_now(); } } while (0)
__device__ __forceinline__ unsigned long long kv_now() {
#ifdef R4D_EMU_HOST
    return 0ull;   /* the host SIMT emulator has no clock; the stamped instance is bench-only */
#else
    return (unsigned long long)wall_clock64();
#endif
}
__device__ __forceinline__ unsigned long long* kv_st_base(int32_t* miss_slot, int slots) {
    return (unsigned long long*)(((uintptr_t)(miss_slot + 2LL * slots) + 7u) & ~(uintptr_t)7u);
}

template <bool Flat, bool Stamp = false>
__global__ __launch_bounds__(kThreads) void kv_resolve_kernel(
    int32_t* __restrict__ sel, const int32_t* __restrict__ nsel,
    int32_t* __restrict__ res, int32_t* __restrict__ slot_block,
    int32_t* __restrict__ slot_state, int32_t* __restrict__ ctl,
    const uint8_t* __restrict__ host, uint8_t* __restrict__ cache,
    int32_t* __restrict__ miss_block, int32_t* __restrict__ miss_slot,
    int M, int blocks, int slots, int pitch, int bbytes, int sst_ld, int cap, int ecache, int vic) {

    __shared__ int s_epoch, s_lookups, s_nmiss, s_distinct, s_placed, s_overflow, s_claims, s_hits;
    __shared__ int s_free;
    /* fn4 kvs_vic: step 4's hand-off to 4b */
    __shared__ int s_kc, s_hand, s_vovf, s_vser, s_vavail, s_vheat;
    __shared__ int s_cnt[kThreads], s_off[kThreads];
    /* The claim list, `cap` ids of dynamic LDS (kv_claim_cap), in LDS because the sort reads every
     * element from every thread; the placed miss list stays in the row's scratch. The entry cache
     * follows it (header, "The entry cache"). */
    R4D_DYN_LDS(int, s_dyn);
    int* const s_list = s_dyn;
    int* const s_ent = s_dyn + cap;
    /* The claim gather's fill counter. */
    __shared__ int s_wc;

    const int tid = (int)threadIdx.x;
    unsigned long long* const kst = Stamp ? kv_st_base(miss_slot, slots) : nullptr;
    (void)kst;
    KV_ST(0);

    /* ---- 1. the epoch ---- */
    if (tid == 0) {
        s_epoch = ctl[0] + 1;
        s_lookups = 0;
        s_claims = 0;
        s_hits = 0;
        s_distinct = 0;
        s_nmiss = 0;
        s_placed = 0;
        s_overflow = 0;
        s_free = 0;
        /* This call's miss count, cleared here: a call that claims nothing returns before the sweep. */
        ctl[2] = 0;
    }
    __syncthreads();
    const int epoch = s_epoch;
    KV_ST(1);

    /* ---- 2. one lookup per entry; the map is its own set ---- */
    int lookups = 0, hits = 0;
    if constexpr (Flat) {
        /* fn4 kvs_flat (Qwen loop round 52): the same lookups over the flat entry space e = q * pitch + i, every thread
         * busy and kFlatU entries a thread in flight -- their selection loads, then their map loads, then their stamps /
         * claims, then the heat bumps and list appends. Today's loop runs row by row, one entry a thread a row, so at 64
         * rows x 513 only 513 of 1,024 threads work and each walks 64 dependent load -> atomic chains. Every outcome is
         * the row loop's: the map is the set (one CAS wins a block), a slot's first stamper alone bumps its heat (two
         * entries of one thread on one slot: its atomics to one address land in program order), and the claim list is
         * sorted in step 3, so the order of claims never reaches the placement. An entry cache value read before a
         * same-thread claim reads -1 where the row loop read -2; step 7 reads the map for both. */
        const int E = M * pitch;
        for (int base = tid; base < E; base += kThreads * kFlatU) {
            int bb[kFlatU], ss[kFlatU], ox[kFlatU], oc[kFlatU];
#pragma unroll
            for (int u = 0; u < kFlatU; ++u) bb[u] = kv_flat_block(sel, nsel, base + u * kThreads, E, pitch, blocks);
#pragma unroll
            for (int u = 0; u < kFlatU; ++u) ss[u] = bb[u] >= 0 ? res[bb[u]] : -3;
#pragma unroll
            for (int u = 0; u < kFlatU; ++u) {
                ox[u] = epoch;
                oc[u] = 0;
                if (bb[u] < 0) continue;
                const int s = ss[u];
                if (ecache > 0) s_ent[base + u * kThreads] = s;
                if (s >= 0 && s < slots) ox[u] = atomicExch(&slot_state[(size_t)s * sst_ld], epoch);
                else if (s == -1) oc[u] = atomicCAS(&res[bb[u]], -1, -2);
            }
#pragma unroll
            for (int u = 0; u < kFlatU; ++u) {
                if (bb[u] < 0) continue;
                const int s = ss[u];
                if (s >= 0 && s < slots) {
                    if (ox[u] != epoch) {
                        slot_state[(size_t)s * sst_ld + 1] = kv_heat_up(slot_state[(size_t)s * sst_ld + 1]);
                        ++hits;
                    }
                } else if (s == -1 && oc[u] == -1) {
                    const int k = atomicAdd(&s_claims, 1);
                    if (k < cap) s_list[k] = bb[u];
                }
            }
        }
        if (tid == 0)
            for (int q = 0; q < M; ++q) {
                int n = nsel[q];
                if (n < 0) n = 0;
                if (n > pitch) n = pitch;
                lookups += n;
            }
    } else
    for (int q = 0; q < M; ++q) {
        int n = nsel[q];
        if (n < 0) n = 0;
        if (n > pitch) n = pitch;
        lookups += n;
        for (int i = tid; i < n; i += kThreads) {
            const size_t e = (size_t)q * pitch + i;
            const int b = sel[e];
            if ((unsigned)b >= (unsigned)blocks) continue;
            const int s = res[b];
            /* The value this lookup saw, for step 7 (header, "The entry cache"). */
            if (ecache > 0) s_ent[e] = s;
            if (s >= 0 && s < slots) {
                /* The first thread to stamp the slot counts it into the union. */
                if (atomicExch(&slot_state[(size_t)s * sst_ld], epoch) != epoch) {
                    /* One heat bump a slot a call, by the first stamper. */
                    slot_state[(size_t)s * sst_ld + 1] =
                        kv_heat_up(slot_state[(size_t)s * sst_ld + 1]);
                    ++hits;
                }
            } else if (s == -1) {
                /* One CAS wins per block: the distinct-claim count and the list's fill, without a map walk. */
                if (atomicCAS(&res[b], -1, -2) == -1) {
                    const int k = atomicAdd(&s_claims, 1);
                    if (k < cap) s_list[k] = b;
                }
            }
        }
    }
    /* The launch's hit count: per-thread totals, then the warp-first reduction the union uses. */
    s_cnt[tid] = hits;
    __syncthreads();
    if (tid < 32) {
        int t = 0;
        for (int i = tid; i < kThreads; i += 32) t += s_cnt[i];
        s_off[tid] = t;
    }
    __syncthreads();
    if (tid == 0) {
        int used = 0;
        for (int t = 0; t < 32; ++t) used += s_off[t];
        s_hits = used;
        s_lookups = lookups;
        s_distinct = s_claims;
    }
    __syncthreads();
    KV_ST(2);

    /* ---- 3. the canonical miss order: ascending block id ---- */
    const int claims = s_distinct;
    const int want = claims < slots ? claims : slots;
    if (claims > 0) {
        if (claims <= cap) {
            kv_sort_claims(s_list, claims, tid);
            for (int i = tid; i < want; i += kThreads) miss_block[i] = s_list[i];
        } else {
            /* Above the list (header, "The claim list"): the claims are gathered from the selection
             * into the row's scratch and sorted there. A call with more claims than slots keeps
             * the map walk below, since the smallest `slots` claims need the whole claim set. */
            if (claims <= slots) {
                if (tid == 0) s_wc = 0;
                __syncthreads();
                for (int q = 0; q < M; ++q) {
                    int n = nsel[q];
                    if (n < 0) n = 0;
                    if (n > pitch) n = pitch;
                    for (int i = tid; i < n; i += kThreads) {
                        const int b = sel[(size_t)q * pitch + i];
                        if ((unsigned)b >= (unsigned)blocks) continue;
                        /* The map is the set: -3 marks a block already collected, restored below. */
                        if (atomicCAS(&res[b], -2, -3) == -2) {
                            const int k = atomicAdd(&s_wc, 1);
                            if (k < slots) miss_block[k] = b;
                        }
                    }
                }
                __syncthreads();
                for (int q = 0; q < M; ++q) {
                    int n = nsel[q];
                    if (n < 0) n = 0;
                    if (n > pitch) n = pitch;
                    for (int i = tid; i < n; i += kThreads) {
                        const int b = sel[(size_t)q * pitch + i];
                        if ((unsigned)b >= (unsigned)blocks) continue;
                        if (res[b] == -3) res[b] = -2;
                    }
                }
                __syncthreads();
                /* Padded to a power of two, as the row's scratch is sized (r4d_rad_kv_resolve_scratch). */
                kv_sort_claims(miss_block, claims, tid);
            } else {
                /* More claims than slots: the call has overflowed, the sweep stops at its first
                 * unplaceable miss and ctl[3] latches. The ascending order of the whole claim set
                 * is the contract and nothing can hold it, so the map is walked. A call the ring can
                 * hold never comes here. */
                const int chunk = (blocks + kThreads - 1) / kThreads;
                const int lo = (int)((long long)tid * chunk);
                const int hi = lo + chunk < blocks ? lo + chunk : blocks;
                const bool vec4 = (lo & 3) == 0 && (hi & 3) == 0 && ((uintptr_t)res & 15u) == 0;
                int c = 0;
                if (vec4) {
                    const int4* const r4 = (const int4*)res;
                    for (int i = lo / 4; i < hi / 4; ++i) {
                        const int4 v = r4[i];
                        c += (v.x == -2) + (v.y == -2) + (v.z == -2) + (v.w == -2);
                    }
                } else {
                    for (int b = lo; b < hi; ++b) c += (res[b] == -2);
                }
                s_cnt[tid] = c;
                s_off[tid] = c;
                __syncthreads();
                for (int step = 1; step < kThreads; step <<= 1) {
                    const int v = tid >= step ? s_off[tid - step] : 0;
                    __syncthreads();
                    s_off[tid] += v;
                    __syncthreads();
                }
                __syncthreads();
                int pos = s_off[tid] - s_cnt[tid];
                if (vec4) {
                    const int4* const r4 = (const int4*)res;
                    for (int i = lo / 4; i < hi / 4; ++i) {
                        const int4 v = r4[i];
                        if (v.x == -2 && pos < slots) miss_block[pos++] = 4 * i;
                        if (v.y == -2 && pos < slots) miss_block[pos++] = 4 * i + 1;
                        if (v.z == -2 && pos < slots) miss_block[pos++] = 4 * i + 2;
                        if (v.w == -2 && pos < slots) miss_block[pos++] = 4 * i + 3;
                    }
                } else {
                    for (int b = lo; b < hi; ++b)
                        if (res[b] == -2 && pos < slots) miss_block[pos++] = b;
                }
                __syncthreads();
            }
        }
        if (tid == 0) {
            s_nmiss = want;
            s_overflow = claims > slots;
            s_placed = 0;
        }
        __syncthreads();
    }

    KV_STB(3);
    /* ---- 3b. how many slots are free ---- */
    /* Free slots are taken before any resident block is evicted (step 4). One coalesced pass over
     * the slot maps counts them, so a full ring goes straight to CLOCK instead of scanning a turn
     * per miss. A slot this call stamped is not free: it is the call's own. */
    /* fn4 kvs_vic: every thread's copy of the hand, read before the barrier below (thread 0 rewrites ctl[1]) */
    int hand0 = ctl[1];
    if (hand0 < 0 || hand0 >= slots) hand0 = 0;
    if (claims > 0) {
        int nf = 0;
        for (int s = tid; s < slots; s += kThreads)
            if (slot_block[s] < 0 && slot_state[(size_t)s * sst_ld] != epoch) ++nf;
        if (nf != 0) atomicAdd(&s_free, nf);
    }
    __syncthreads();
    KV_ST(4);

    /* fn4 kvs_vic (16.11.2026, qwen/KV-STREAM-EXACT.md): the victims of step 4 without thread 0
     * walking the ring. Thread 0 alone scanned for a free slot (skipping every occupied one) and swept
     * CLOCK one slot at a time, a global load or two a slot: a ring that is kept warm (fn4 kvs_apart)
     * and fills paid hundreds of serial loads a miss. Here the same victims, in the same order, come
     * out of order-keeping compactions of the ring in the hand's rotated order:
     *   4a  the free slots after the hand (one list, built when more than kVicListMin slots are
     *       occupied; below that the serial scan skips little), taken in list order by thread 0, an
     *       entry stamped or filled meanwhile (an identity placement) skipped -- the scan's choice;
     *   4b  the misses left when no free slot remains go to CLOCK in at most three rotations, each one
     *       compaction of the slots a sweep would take on this pass (unstamped, heat 0 or no block);
     *       a full rotation halves every other unstamped slot's heat, the last one only the slots
     *       before its last victim -- what the serial sweep halves on its way.
     * The serial code below stays for what 4b does not model: a call with more claims than slots
     * (overflow), a heat above 3, an identity placement left at the CLOCK phase (unreachable with
     * consistent maps: it needs a free unstamped slot), or fewer unstamped slots than misses. */
    int* const vlist = miss_slot + slots;   /* the row scratch's third part: `slots` slot ids */
    const int nmiss0 = claims > 0 ? s_nmiss : 0;
    const bool vlist_on = vic && claims > 0 && !s_overflow && s_free > 0 && slots - s_free > kVicListMin;
    if (vlist_on) {
        const int nfree = kv_compact_rot(hand0, slots, vlist, slots, tid, s_cnt, s_off,
            [&](int q) { return slot_block[q] < 0 && slot_state[(size_t)q * sst_ld] != epoch; });
        (void)nfree;   /* = s_free: the same predicate over the same state */
    }
    KV_STB(5);

    /* ---- 4. one victim per miss: the block's own slot first, else a free slot, else CLOCK ---- */
    /* Identity first: a block whose id fits the ring and whose own slot is free goes there, so a
     * context shorter than the ring has res[b] = b and the attention's run of adjacent ids walks
     * one stripe in order. It never evicts and leaves the hand alone, so the fallback (free slot,
     * then CLOCK) is unchanged. The free scan changes nothing when it finds nothing, and
     * `free_left` (step 3b) only falls during the call, so a full ring pays the count once.
     * kvs_vic: thread 0 stops at the first miss that needs CLOCK (s_kc) when 4b can take the rest. */
    if (tid == 0 && claims > 0) {
        int hand = hand0;
        int placed = 0;
        bool overflow = s_overflow;
        int free_left = s_free;
        const int nmiss = s_nmiss;
        int fp = 0;
        int kc = nmiss;
        for (int k = 0; k < nmiss; ++k) {
            const int b = miss_block[k];
            int victim = -1;
            /* Identity, if the id fits and the slot is free or already holds it. */
            if (b < slots) {
                const int want = b;
                if (slot_state[(size_t)want * sst_ld] != epoch &&
                    (slot_block[want] < 0 || slot_block[want] == b))
                    victim = want;
            }
            /* Free slots first, in CLOCK's hand order; a slot stamped this call is never taken. */
            if (victim < 0 && free_left > 0) {
                if (vlist_on) {
                    /* kvs_vic 4a: the scan's first free unstamped slot from the hand is the list's
                     * next entry that is still free and unstamped */
                    while (fp < s_free) {
                        const int s = vlist[fp++];
                        if (slot_state[(size_t)s * sst_ld] == epoch) continue;
                        if (slot_block[s] >= 0) continue;
                        victim = s;
                        hand = s + 1 == slots ? 0 : s + 1;
                        break;
                    }
                } else {
                    int h = hand;
                    for (int scanned = 0; scanned < slots; ++scanned) {
                        const int s = h;
                        h = h + 1 == slots ? 0 : h + 1;
                        if (slot_state[(size_t)s * sst_ld] == epoch) continue;  /* used by this call */
                        if (slot_block[s] >= 0) continue;                       /* holds a block */
                        victim = s;
                        hand = h;
                        break;
                    }
                }
                if (victim < 0) free_left = 0;
            }
            if (victim < 0 && !overflow && vic) {
                /* kvs_vic 4b takes this miss and the rest (all threads, below) */
                kc = k;
                break;
            }
            if (victim < 0) {
                for (long long scanned = 0; scanned < 3LL * slots; ++scanned) {
                    const int s = hand;
                    hand = hand + 1 == slots ? 0 : hand + 1;
                    if (slot_state[(size_t)s * sst_ld] == epoch) continue;  /* used by this call */
                    const int old = slot_block[s];
                    if (old >= 0 && old < blocks &&
                        slot_state[(size_t)s * sst_ld + 1] > 0) {            /* second chance */
                        slot_state[(size_t)s * sst_ld + 1] =
                            kv_heat_down(slot_state[(size_t)s * sst_ld + 1]);
                        continue;
                    }
                    victim = s;
                    break;
                }
            }
            if (victim < 0) { overflow = true; break; }
            /* ---- 5. re-point both maps ---- */
            const int s = victim;
            const int old = slot_block[s];
            if (old < 0 && free_left > 0) --free_left;   /* a free slot the sweep just filled */
            if (old >= 0 && old < blocks) res[old] = -1;
            slot_block[s] = b;
            slot_state[(size_t)s * sst_ld] = epoch;
            slot_state[(size_t)s * sst_ld + 1] =
                kv_heat_up(slot_state[(size_t)s * sst_ld + 1]);
            res[b] = s;
            miss_slot[k] = s;
            ++placed;
        }
        s_kc = kc;
        s_hand = hand;
        s_placed = placed;
        s_vovf = overflow ? 1 : 0;
        if (kc == nmiss) {
            ctl[1] = hand;
            ctl[2] = placed;
            if (overflow) ctl[3] = 1;
        }
    }
    __syncthreads();
    KV_ST(6);

    /* ---- 4b. kvs_vic: CLOCK for misses kc .. nmiss-1, in rotations of the ring from the hand ---- */
    if (claims > 0 && s_kc < nmiss0) {
        const int kc = s_kc, nc = nmiss0 - kc, hc = s_hand;
        /* What the model needs: every remaining miss non-identity (consistent maps), heats in 0..3,
         * and at least nc unstamped slots. Else thread 0 runs the serial sweep (s_vser). */
        if (tid == 0) { s_vser = 0; s_vavail = 0; s_vheat = 0; }
        __syncthreads();
        {
            int av = 0, hm = 0;
            for (int q = tid; q < slots; q += kThreads) {
                if (slot_state[(size_t)q * sst_ld] == epoch) continue;
                ++av;
                const int hq = slot_state[(size_t)q * sst_ld + 1];
                if (hq > hm) hm = hq;
            }
            if (av) atomicAdd(&s_vavail, av);
            if (hm) atomicMax(&s_vheat, hm);
            for (int k = kc + tid; k < nmiss0; k += kThreads) {
                const int b = miss_block[k];
                if (b < slots && slot_state[(size_t)b * sst_ld] != epoch &&
                    (slot_block[b] < 0 || slot_block[b] == b))
                    atomicExch(&s_vser, 1);
            }
        }
        __syncthreads();
        if (s_vavail < nc || s_vheat > 3) { if (tid == 0) s_vser = 1; }
        __syncthreads();
        KV_ST(7);
        if (!s_vser) {
            int got = 0, hand = hc;
            for (int rot = 0; rot < 3 && got < nc; ++rot) {
                const int want_n = nc - got;
                int* const out = miss_slot + kc + got;
                const int n = kv_compact_rot(hc, slots, out, want_n, tid, s_cnt, s_off, [&](int q) {
                    if (slot_state[(size_t)q * sst_ld] == epoch) return false;
                    const int old = slot_block[q];
                    return !(old >= 0 && old < blocks && slot_state[(size_t)q * sst_ld + 1] > 0);
                });
                const int take = n < want_n ? n : want_n;
                /* the sweep's reach this rotation: all of it, or up to (not including) its last victim */
                const int lim = n >= want_n ? (out[take - 1] - hc + slots) % slots : slots;
                for (int r = tid; r < lim; r += kThreads) {
                    int q = hc + r;
                    if (q >= slots) q -= slots;
                    if (slot_state[(size_t)q * sst_ld] == epoch) continue;
                    const int old = slot_block[q];
                    const int hq = slot_state[(size_t)q * sst_ld + 1];
                    if (old >= 0 && old < blocks && hq > 0)
                        slot_state[(size_t)q * sst_ld + 1] = kv_heat_down(hq);
                }
                __syncthreads();
                /* this rotation's victims are the call's now: a later rotation skips them */
                for (int i = tid; i < take; i += kThreads) slot_state[(size_t)out[i] * sst_ld] = epoch;
                __syncthreads();
                if (n >= want_n) hand = out[take - 1] + 1 == slots ? 0 : out[take - 1] + 1;
                got += take;
            }
            /* ---- 5. re-point both maps, every CLOCK victim at once (distinct slots and blocks) ---- */
            for (int i = tid; i < nc; i += kThreads) {
                const int k = kc + i, q = miss_slot[k], b = miss_block[k];
                const int old = slot_block[q];
                if (old >= 0 && old < blocks) res[old] = -1;
                slot_block[q] = b;
                slot_state[(size_t)q * sst_ld] = epoch;
                slot_state[(size_t)q * sst_ld + 1] = kv_heat_up(slot_state[(size_t)q * sst_ld + 1]);
                res[b] = q;
            }
            __syncthreads();
            KV_ST(8);
            if (tid == 0) {
                const int placed = s_placed + nc;
                ctl[1] = hand;
                ctl[2] = placed;
                if (s_vovf) ctl[3] = 1;
                s_placed = placed;
            }
        } else if (tid == 0) {
            /* the serial sweep, as step 4 runs it, for misses kc .. nmiss-1 */
            int hand = hc, placed = s_placed;
            bool overflow = s_vovf != 0;
            for (int k = kc; k < nmiss0; ++k) {
                const int b = miss_block[k];
                int victim = -1;
                if (b < slots) {
                    if (slot_state[(size_t)b * sst_ld] != epoch &&
                        (slot_block[b] < 0 || slot_block[b] == b))
                        victim = b;
                }
                if (victim < 0) {
                    for (long long scanned = 0; scanned < 3LL * slots; ++scanned) {
                        const int q = hand;
                        hand = hand + 1 == slots ? 0 : hand + 1;
                        if (slot_state[(size_t)q * sst_ld] == epoch) continue;
                        const int old = slot_block[q];
                        if (old >= 0 && old < blocks && slot_state[(size_t)q * sst_ld + 1] > 0) {
                            slot_state[(size_t)q * sst_ld + 1] = kv_heat_down(slot_state[(size_t)q * sst_ld + 1]);
                            continue;
                        }
                        victim = q;
                        break;
                    }
                }
                if (victim < 0) { overflow = true; break; }
                const int q = victim;
                const int old = slot_block[q];
                if (old >= 0 && old < blocks) res[old] = -1;
                slot_block[q] = b;
                slot_state[(size_t)q * sst_ld] = epoch;
                slot_state[(size_t)q * sst_ld + 1] = kv_heat_up(slot_state[(size_t)q * sst_ld + 1]);
                res[b] = q;
                miss_slot[k] = q;
                ++placed;
            }
            ctl[1] = hand;
            ctl[2] = placed;
            if (overflow) ctl[3] = 1;
            s_placed = placed;
        }
    }
    __syncthreads();
    KV_ST(9);

    /* Unplaced claims (still -2) revert to -1 so a later call may claim them. Every claim is named
     * by a selection entry, so the selection reaches all of them; the list is the shorter pass when
     * it holds every claim. */
    if (claims > 0) {
        if (s_distinct <= slots) {
            for (int k = tid; k < s_distinct; k += kThreads)
                if (res[miss_block[k]] == -2) res[miss_block[k]] = -1;
        } else {
            for (int q = 0; q < M; ++q) {
                int n = nsel[q];
                if (n < 0) n = 0;
                if (n > pitch) n = pitch;
                for (int i = tid; i < n; i += kThreads) {
                    const int b = sel[(size_t)q * pitch + i];
                    if ((unsigned)b >= (unsigned)blocks) continue;
                    if (res[b] == -2) res[b] = -1;
                }
            }
        }
    }
    __syncthreads();
    KV_ST(10);

    /* ---- 6. the copies, one block each, in miss order, in this launch ---- */
    /* One warp a miss, so every miss's host read is in flight at once: a 2 KiB copy from the
     * mirror is PCIe transactions at about 1 us each, and a decode launch has tens of misses.
     *
     * Four loads per thread before any store. A wave issues in order and a store waits on its
     * load, so `dst[i] = src[i]` at a stride of 32 would pay the read latency once per iteration;
     * the trip count is runtime, so the compiler cannot unroll it. A 2 KiB block is one full turn
     * of the group (128 vectors), so the common path never takes the tail. Four more VGPRs
     * (32 -> 36, no spill); the kernel runs one workgroup. `kv_v16` rather than `uint4` keeps
     * the loads apart from their stores. */
    const int placed = s_placed;
    const int lane = tid & 31;
    if (bbytes >= 16 && bbytes % 16 == 0) {
        const int vec = bbytes / 16;
        for (int k = tid >> 5; k < placed; k += kThreads / 32) {
            const kv_v16* __restrict__ const src =
                (const kv_v16*)host + (size_t)miss_block[k] * vec;
            kv_v16* __restrict__ const dst = (kv_v16*)cache + (size_t)miss_slot[k] * vec;
            int i = lane;
            for (; i + 96 < vec; i += 128) {
                const kv_v16 a = src[i], b = src[i + 32], c = src[i + 64], d = src[i + 96];
                dst[i] = a;
                dst[i + 32] = b;
                dst[i + 64] = c;
                dst[i + 96] = d;
            }
            /* A block that is not a whole number of four-wide turns finishes one vector at a time. */
            for (; i < vec; i += 32) dst[i] = src[i];
        }
    } else {
        for (int k = tid >> 5; k < placed; k += kThreads / 32) {
            const uint8_t* src = host + (size_t)miss_block[k] * bbytes;
            uint8_t* const dst = cache + (size_t)miss_slot[k] * bbytes;
            for (int i = lane; i < bbytes; i += 32) dst[i] = src[i];
        }
    }

    KV_STB(11);
    /* ---- 7. the rewrite: every valid entry becomes the slot holding its block ---- */
    if constexpr (Flat) {
        /* fn4 kvs_flat: the same rewrite over the flat entry space, kFlatU entries a thread in flight (each entry is its own
         * thread's: sel[e] is read and written by one thread, as below). */
        const int E = M * pitch;
        for (int base = tid; base < E; base += kThreads * kFlatU) {
            int bb[kFlatU], sv[kFlatU];
#pragma unroll
            for (int u = 0; u < kFlatU; ++u) bb[u] = kv_flat_block(sel, nsel, base + u * kThreads, E, pitch, blocks);
#pragma unroll
            for (int u = 0; u < kFlatU; ++u) {
                sv[u] = -1;
                if (bb[u] < 0) continue;
                int s = ecache > 0 ? s_ent[base + u * kThreads] : -1;
                if (s < 0) s = res[bb[u]];
                sv[u] = s;
            }
#pragma unroll
            for (int u = 0; u < kFlatU; ++u)
                if (bb[u] >= 0) sel[base + u * kThreads] = sv[u] >= 0 ? sv[u] : 0;
        }
    } else {
    for (int q = 0; q < M; ++q) {
        int n = nsel[q];
        if (n < 0) n = 0;
        if (n > pitch) n = pitch;
        for (int i = tid; i < n; i += kThreads) {
            const size_t e = (size_t)q * pitch + i;
            const int b = sel[e];
            if ((unsigned)b >= (unsigned)blocks) continue;
            /* A hit's slot is the one step 2 read; for an entry absent then, only the map knows where the
             * sweep placed it. */
            int s = ecache > 0 ? s_ent[e] : -1;
            if (s < 0) s = res[b];
            sel[e] = s >= 0 ? s : 0;
        }
    }
    }

    KV_STB(12);
    /* The union: the distinct slots this call hit or filled, plus the claims it could not place. */
    if (tid == 0) {
        const int placed = s_placed;
        const int uni = s_hits + s_distinct;
        if (uni > ctl[10]) ctl[10] = uni;
        if (placed > ctl[11]) ctl[11] = placed;
        /* The depth the claim list had to hold. */
        if (s_distinct > ctl[12]) ctl[12] = s_distinct;
        uint64_t* const ctr = (uint64_t*)(ctl + 4);
        ctr[0] += (uint64_t)placed;
        ctr[1] += (uint64_t)s_lookups;
        ctr[2] += 1u;
        ctl[0] = epoch;
        if constexpr (Stamp) {
            kst[13] = kv_now();
            kst[16] = (unsigned long long)s_distinct; kst[17] = (unsigned long long)s_nmiss;
            kst[18] = (unsigned long long)s_placed;   kst[19] = (unsigned long long)s_hits;
            kst[20] = (unsigned long long)s_free;     kst[21] = (unsigned long long)s_kc;
            kst[22] = (unsigned long long)s_vser;     kst[23] = (unsigned long long)s_lookups;
        }
    }
}
#undef KV_ST
#undef KV_STB

}  // namespace

/* `sel`         [M][sel_pitch]     i32, block ids in, ring slots out
 * `nsel`        [M]                i32, valid entries of each row
 * `res`         [blocks]           i32, block -> slot, -1 absent, -2 claimed by this call
 * `slot_block`  [slots]            i32, slot -> block, -1 free
 * `slot_state`  [slots][sst_ld]    i32, [s][0] epoch of last use, [s][1] two-bit heat counter
 * `ctl`         [16]               i32, [0] epoch, [1] hand, [2] placed, [3] overflow, u64 from [4]
 * `host`        [blocks][bbytes]   u8, the layer's host mirror
 * `cache`       [slots][bbytes]    u8, the layer's device ring
 * `miss_block`, `miss_slot`        the row scratch, `next_pow2(slots)` ids then `slots` slots
 * `cap`                            the claim list's capacity, from kv_claim_cap
 * `ecache`                         the entry cache's capacity in ints: `M * sel_pitch`, or 0 when
 *                                  the claim list leaves no room for it (header, "the entry cache")
 *
 * The two planes are addressed by `bbytes` alone, so the shim has already checked that each
 * block's row pitch in bytes is exactly `block_bytes` and that its trailing extents fill it. */
/* Qwen loop round 53 (QUEUE Q105, bench only): r4d_kv_stamp_set(1) = the stamped instances (kv_resolve_kernel<.., true>),
 * which write 24 u64 past the row scratch's victim list (the caller's scratch must hold r4d_rad_kv_resolve_scratch + 256 B). */
static std::atomic<int> g_kv_stamp{0};
extern "C" int r4d_kv_stamp_set(int on) {
    if (on < 0 || on > 1) return -1;
    g_kv_stamp.store(on, std::memory_order_relaxed);
    return 1;
}

static int r4d_kv_resolve_emit(long sel, long nsel, long res, long slot_block, long slot_state,
                               long ctl, long host, long cache, long miss_block, long miss_slot,
                               int M, int blocks, int slots, int pitch, int bbytes, int sst_ld,
                               int vic, int flat, RadStream stream) {
    if (!sel || !nsel || !res || !slot_block || !slot_state || !ctl || !host || !cache ||
        !miss_block || !miss_slot)
        return R4D_NO(RAD_E_INVAL);
    if (M <= 0 || blocks <= 0 || slots <= 0 || pitch <= 0 || bbytes <= 0 || sst_ld < 2)
        return R4D_NO(RAD_E_SHAPE);
    /* A workgroup with no LDS room for one id cannot run the op: a shape the caller cannot satisfy,
     * so refuse. */
    int room = 0;
    const int cap = kv_claim_cap(M, pitch, &room);
    if (cap < 1) return R4D_NO(RAD_E_SHAPE);
    /* The entry cache, one int an entry, while it fits beside the claim list: a decode launch caches
     * its 513, a wide prefill does not and step 7 reads the map. */
    const long long entries = (long long)M * pitch;
    const int ecache = entries <= (long long)(room - cap) ? (int)entries : 0;
    /* fn4 kvs_flat: steps 2 and 7 over the flat entry space (the kernel's note at kv_flat_block), 32-bit entry indices:
     * a launch whose M * pitch passes them keeps the row loops (none does: max_tok x 513) */
    const bool stamp = g_kv_stamp.load(std::memory_order_relaxed) != 0;
    if (stamp) {
        if (flat && entries + (long long)kThreads * kFlatU <= (long long)INT32_MAX)
            return R4D_EMIT(stream, (kv_resolve_kernel<true, true>), dim3(1), dim3(kThreads),
                            (uint32_t)(cap + ecache) * sizeof(int),
                            (int32_t*)sel, (const int32_t*)nsel, (int32_t*)res, (int32_t*)slot_block,
                            (int32_t*)slot_state, (int32_t*)ctl, (const uint8_t*)host, (uint8_t*)cache,
                            (int32_t*)miss_block, (int32_t*)miss_slot, M, blocks, slots, pitch,
                            bbytes, sst_ld, cap, ecache, vic);
        return R4D_EMIT(stream, (kv_resolve_kernel<false, true>), dim3(1), dim3(kThreads),
                        (uint32_t)(cap + ecache) * sizeof(int),
                        (int32_t*)sel, (const int32_t*)nsel, (int32_t*)res, (int32_t*)slot_block,
                        (int32_t*)slot_state, (int32_t*)ctl, (const uint8_t*)host, (uint8_t*)cache,
                        (int32_t*)miss_block, (int32_t*)miss_slot, M, blocks, slots, pitch,
                        bbytes, sst_ld, cap, ecache, vic);
    }
    if (flat && entries + (long long)kThreads * kFlatU <= (long long)INT32_MAX)
        return R4D_EMIT(stream, (kv_resolve_kernel<true>), dim3(1), dim3(kThreads),
                        (uint32_t)(cap + ecache) * sizeof(int),
                        (int32_t*)sel, (const int32_t*)nsel, (int32_t*)res, (int32_t*)slot_block,
                        (int32_t*)slot_state, (int32_t*)ctl, (const uint8_t*)host, (uint8_t*)cache,
                        (int32_t*)miss_block, (int32_t*)miss_slot, M, blocks, slots, pitch,
                        bbytes, sst_ld, cap, ecache, vic);
    return R4D_EMIT(stream, (kv_resolve_kernel<false>), dim3(1), dim3(kThreads),
                    (uint32_t)(cap + ecache) * sizeof(int),
                    (int32_t*)sel, (const int32_t*)nsel, (int32_t*)res, (int32_t*)slot_block,
                    (int32_t*)slot_state, (int32_t*)ctl, (const uint8_t*)host, (uint8_t*)cache,
                    (int32_t*)miss_block, (int32_t*)miss_slot, M, blocks, slots, pitch,
                    bbytes, sst_ld, cap, ecache, vic);
}

/* ============================================================== the plugin ABI entry point */

enum { KR_SEL = 0, KR_NSEL = 1, KR_RES = 2, KR_SBLK = 3, KR_SST = 4, KR_CTL = 5, KR_HOST = 6,
       KR_CACHE = 7 };

/* Fan a byte extent into the shape's trailing elements: one block a row, dense, so the row pitch
 * in bytes and the product of the trailing extents' element bytes both equal `block_bytes`. */
static int r4d_kv_block_row_bytes(const RadTensor* t, long long* out) {
    long long acc = 1;
    for (unsigned d = 1; d < t->rank; ++d) acc *= t->shape[d];
    *out = acc * (long long)rad_dtype_bytes(t->dtype, 1);
    return t->stride[0] * (long long)rad_dtype_bytes(t->dtype, 1) == *out ? RAD_OK : RAD_E_STRIDE;
}

/* fn4 kvs_vic (16.11.2026, qwen/KV-STREAM-EXACT.md; RADIANCE_FN4_KVS_VIC, rad_fn4_switch("kvs_vic")): 1 = step 4's victims
 * by compaction (4a / 4b), 0 = thread 0's serial scan and sweep. The same victims either way (the placement is libref's).
 * A launch parameter `vic` overrides it (r4d_selftest). -1 until the environment is read. */
static std::atomic<int> g_kv_vic{-1};
static int kv_vic(const RadArgs* a) {
    int v = g_kv_vic.load(std::memory_order_relaxed);
    if (v < 0) {
        const char* e = std::getenv("RADIANCE_FN4_KVS_VIC");
        const int on = (e && *e && std::atoi(e) > 0) ? 1 : 0;
        int want = -1;
        g_kv_vic.compare_exchange_strong(want, on, std::memory_order_relaxed);
        v = g_kv_vic.load(std::memory_order_relaxed);
    }
    return (int)r4d_geti_or(a, "vic", v) ? 1 : 0;
}
extern "C" int r4d_kv_vic_set(int on) {
    if (on < 0 || on > 1) return -1;
    g_kv_vic.store(on, std::memory_order_relaxed);
    return 1;
}

/* fn4 kvs_flat (Qwen loop round 52, QUEUE Q101; RADIANCE_FN4_KVS_FLAT, rad_fn4_switch("kvs_flat")): 1 = steps 2 and 7 over
 * the flat entry space (kv_resolve_kernel<true>), 0 = today's row loops. The same placement, maps, counters and rewritten
 * selection either way (libref's). A launch parameter `flat` overrides it (the host emulator's tests). */
static std::atomic<int> g_kv_flat{-1};
static int kv_flat(const RadArgs* a) {
    int v = g_kv_flat.load(std::memory_order_relaxed);
    if (v < 0) {
        const char* e = std::getenv("RADIANCE_FN4_KVS_FLAT");
        const int on = (e && *e && std::atoi(e) > 0) ? 1 : 0;
        int want = -1;
        g_kv_flat.compare_exchange_strong(want, on, std::memory_order_relaxed);
        v = g_kv_flat.load(std::memory_order_relaxed);
    }
    return (int)r4d_geti_or(a, "flat", v) ? 1 : 0;
}
extern "C" int r4d_kv_flat_set(int on) {
    if (on < 0 || on > 1) return -1;
    g_kv_flat.store(on, std::memory_order_relaxed);
    return 1;
}

extern "C" int r4d_rad_kv_resolve(const RadArgs* a, RadStream stream) {
    const RadTensor* se = r4d_opt(a, KR_SEL);
    const RadTensor* ns = r4d_opt(a, KR_NSEL);
    const RadTensor* re = r4d_opt(a, KR_RES);
    const RadTensor* sb = r4d_opt(a, KR_SBLK);
    const RadTensor* st = r4d_opt(a, KR_SST);
    const RadTensor* ct = r4d_opt(a, KR_CTL);
    const RadTensor* ho = r4d_opt(a, KR_HOST);
    const RadTensor* ca = r4d_opt(a, KR_CACHE);
    if (!se || !ns || !re || !sb || !st || !ct || !ho || !ca)
        return R4D_NO(RAD_E_INVAL);

    /* `M` is ranged, so the geometry carries the band's ceiling (max_tok), not this launch's rows: the
     * row count is `sel`'s extent, and `M` the ceiling it may not pass. */
    const long long band  = r4d_geti_or(a, "M", 0);
    const long long M     = se->shape[0];
    const long long blocks = r4d_geti_or(a, "blocks", 0);
    /* The ring depth is `slot_block`'s: the pool is planned after declare, so a ring --kv-resident
     * sized smaller arrives shorter than the declared `slots`, its ceiling. */
    const long long slots_decl = r4d_geti_or(a, "slots", 0);
    const long long slots = sb->shape[0];
    /* The row pitch is `sel`'s too: one different from the geometry would rewrite a row into its
     * neighbour. */
    const long long declared_pitch = r4d_geti_or(a, "sel_pitch", 0);
    const long long pitch = r4d_row_pitch(se);
    const long long bbytes = r4d_geti_or(a, "block_bytes", 0);
    if (M <= 0 || blocks <= 0 || slots <= 0 || slots > slots_decl || pitch <= 0 || bbytes <= 0)
        return R4D_NO(RAD_E_SHAPE);
    if (M > INT32_MAX || blocks > INT32_MAX || slots > INT32_MAX || pitch > INT32_MAX ||
        bbytes > INT32_MAX)
        return R4D_NO(RAD_E_SHAPE);
    /* A launch above the band was resolved against constraints it does not satisfy. */
    if (band > 0 && M > band) return R4D_NO(RAD_E_SHAPE);
    if (declared_pitch > 0 && pitch != declared_pitch) return R4D_NO(RAD_E_STRIDE);

    /* The maps and counts are one format: i32, dense. */
    const RadTensor* state[] = { se, ns, re, sb, st, ct };
    for (const RadTensor* t : state)
        if (t->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    if (se->rank != 2 || st->rank != 2) return R4D_NO(RAD_E_SHAPE);
    if (se->stride[1] != 1) return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(ns) < M || ns->stride[0] != 1) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(re) < blocks || re->stride[0] != 1) return R4D_NO(RAD_E_SHAPE);
    if (sb->stride[0] != 1) return R4D_NO(RAD_E_SHAPE);
    if (st->shape[0] < slots || st->stride[1] != 1 || st->stride[0] < 2)
        return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(ct) < 16 || ct->stride[0] != 1) return R4D_NO(RAD_E_SHAPE);
    if (ho->rank < 2 || ca->rank < 2) return R4D_NO(RAD_E_SHAPE);
    if (ho->shape[0] < blocks || ca->shape[0] < slots) return R4D_NO(RAD_E_SHAPE);

    /* The mirror's row and the pool's slot are both `block_bytes`. The pool is a KV group's cache, so
     * its dtype is the cache layout's; only its byte pitch is fixed. */
    long long hdense = 0, pdense = 0;
    if (r4d_kv_block_row_bytes(ho, &hdense) != RAD_OK || hdense != bbytes)
        return R4D_NO(RAD_E_STRIDE);
    if (r4d_kv_block_row_bytes(ca, &pdense) != RAD_OK || pdense != bbytes)
        return R4D_NO(RAD_E_STRIDE);
    /* The counters are read as u64 in an i32 block. */
    if (((uintptr_t)ct->data & 7u) != 0) return R4D_NO(RAD_E_STRIDE);

    /* The row's scratch: `nlist` ids (the miss list, a power of two when a wide launch sorts there),
     * then `slots` slots. */
    const int64_t nlist = kv_pow2ceil(slots);
    /* + `slots` ids for fn4 kvs_vic's victim lists (the third part, after miss_slot) */
    const long long need = (nlist + 2LL * slots) * 4;
    if (!a->scratch || a->scratch_bytes < need) return R4D_NO(RAD_E_INVAL);
    int32_t* const miss_block = (int32_t*)a->scratch;
    int32_t* const miss_slot = (int32_t*)((char*)a->scratch + nlist * 4);

    return r4d_kv_resolve_emit(r4d_addr(se), r4d_addr(ns), r4d_addr(re), r4d_addr(sb),
                               r4d_addr(st), r4d_addr(ct), r4d_addr(ho), r4d_addr(ca),
                               (long)(uintptr_t)miss_block, (long)(uintptr_t)miss_slot,
                               (int)M, (int)blocks, (int)slots, (int)pitch, (int)bbytes,
                               (int)st->stride[0], kv_vic(a), kv_flat(a), stream);
}

/* `next_pow2(slots)` block ids (a wide launch sorts its claims here), then `slots` slots, then `slots` ids of victim
 * list (fn4 kvs_vic). */
extern "C" int64_t r4d_rad_kv_resolve_scratch(const RadArgs* a) {
    const long long slots = r4d_geti_or(a, "slots", 0);
    if (slots <= 0 || slots > INT32_MAX) return 0;
    return (kv_pow2ceil(slots) + 2LL * slots) * 4;   /* + fn4 kvs_vic's victim list */
}
