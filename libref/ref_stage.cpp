/* ref_stage.cpp -- the two-node pipeline's stage crossings on the host backend (tp8 S4,
 * abi/rad_pp.h, docs/OPS.md "Stage crossings").
 *
 *   stage_send  pack the live rows of up to eight buffers into this node's send ring and publish
 *               them (`ready`); the core's link carries them to the other node
 *   stage_recv  wait for the other node's payload to land in this node's receive ring, copy it
 *               into up to eight buffers, count this rank in `taken`
 *
 * The bytes move verbatim: a crossing changes no number, so the receiving layer computes what it
 * would have in one process. Every rank of the receiving node takes the payload into its own
 * buffers (the residual stream is replicated over the ranks, spec §9); one rank of the sending
 * node publishes it (`sender`, rank 0 unless given), the others' launch returns at once.
 *
 * Blocking. stage_recv waits on the rank's thread, which is what the host backend's stream is; it
 * gives up with RAD_E_IO when the core's link sets the ring's abort word (the peer is gone) and
 * with RAD_E_IO after the ring's deadline. stage_send waits only for the link to have taken the
 * last payload (`sent`), which one step a lane always has.
 */
#include "ref_common.h"
#include "ref_ops.h"

#include "rad_pp.h"
#include "rad_plugin.h"

#include <algorithm>
#include <vector>
#include <mutex>
#include <map>
#include <atomic>
#include <chrono>
#include <ctime>
#include <cstring>
#include <thread>

namespace {

uint64_t mono_ms() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

/* CLOCK_MONOTONIC, as the core's stamps read it (tp8 S5's stage busy stamps). */
uint64_t mono_ns() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* The parts after the ring: [1, 1 + parts). Bytes of each, from its own extent. `rows` is part 0's:
 * tp8 S6 lets each part carry its own rows (a kv source's entries over the compressor's work rows
 * beside the residual stream over the step's), and the receiver checks every part's bytes. */
int parts_of(const RadArgs* a, int* n, uint64_t bytes[RAD_PP_MAX_PARTS], int64_t* rows) {
    *n = 0;
    *rows = -1;
    for (int i = 1; i < a->n_t && i <= RAD_PP_MAX_PARTS; ++i) {
        const RadTensor& t = a->t[i];
        if (!t.data) break;
        if (t.rank < 1) return RAD_E_SHAPE;
        if (*rows < 0) *rows = t.shape[0];
        const int64_t el = rad_tensor_numel(&t);
        bytes[*n] = (uint64_t)rad_dtype_bytes(t.dtype, el);
        ++*n;
    }
    return *n > 0 ? RAD_OK : RAD_E_SHAPE;
}

/* Contiguity: a column slice (stride wider than the row) cannot be copied as one run. */
bool dense(const RadTensor& t) {
    int64_t want = 1;
    for (int d = (int)t.rank - 1; d >= 0; --d) {
        if (t.shape[d] > 1 && t.stride[d] != want) return false;
        want *= t.shape[d];
    }
    return true;
}

}  /* namespace */

/* tp8 dev M4: `loop` 1 -- a segment boundary on one node (abi/rad_pp.h RAD_PP_PART_SEG): the ring operand is a buffer of
 * this rank (the plugin's), every rank sends its own parts into its own and takes them back with the receive; the send
 * publishes the payload to itself (a fresh header: number 1, landed and sent with ready), so the buffer needs no setup. */
bool loop_of(const RadArgs* a) { return rad_param_geti(a->p, a->n_p, "loop", 0) == 1; }

/* tp8 loop P10 (RadPpRing::async_rows), as r4d_stage.hip's send_async on a card: the parts packed into the node's staging
 * buffer on the rank's thread (the pass's stream), then the ring copy + publish on a helper thread (the side stream), which
 * the next async payload of the node waits for before it packs (the `copied` event). The ring's bytes are the same. */
namespace {
struct AsyncStage {
    std::atomic<bool> busy{ false };
};
AsyncStage& async_stage(uint64_t stage) {
    static std::mutex mu;
    static std::map<uint64_t, AsyncStage> st;
    std::lock_guard<std::mutex> lk(mu);
    return st[stage];
}
}  /* namespace */

static void publish_from(RadPpRing* r, const char* stage, int n, const uint64_t* bytes, uint64_t total, int64_t rows) {
    const uint64_t t0 = mono_ms();
    while (__atomic_load_n(&r->sent, __ATOMIC_ACQUIRE) < r->ready) {
        if (__atomic_load_n(&r->abort, __ATOMIC_ACQUIRE)) return;
        if (r->deadline_ms && mono_ms() - t0 > r->deadline_ms) {   /* as the card's wait: its deadline into the ring */
            __atomic_store_n(&r->abort, (uint64_t)3, __ATOMIC_RELEASE);
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    if (__atomic_load_n(&r->abort, __ATOMIC_ACQUIRE)) return;
    char* dst = reinterpret_cast<char*>(r) + RAD_PP_RING_HDR;
    std::memcpy(dst, stage, (size_t)total);
    for (int i = 0; i < n; ++i) r->part_bytes[i] = bytes[i];
    const uint64_t seq = r->ready + 1;
    std::memset(dst + total, 0, RAD_PP_TRAILER);
    *reinterpret_cast<uint64_t*>(dst + total + 8) = rad_pp_layout(bytes, n);
    __atomic_store_n(reinterpret_cast<uint64_t*>(dst + total), seq, __ATOMIC_RELEASE);
    r->rows = (uint64_t)rows;
    r->parts = (uint64_t)n;
    r->bytes = total + RAD_PP_TRAILER;
    __atomic_store_n(&r->ready, seq, __ATOMIC_RELEASE);
    r->ready_ns = mono_ns();
}

static int send_async(const RadArgs* a, RadPpRing* r, int n, const uint64_t* bytes, uint64_t total, int64_t rows) {
    char* stage = reinterpret_cast<char*>((uintptr_t)r->stage_dev);
    AsyncStage& st = async_stage(r->stage_dev);
    /* the staging buffer is free once the node's last async copy is done */
    while (st.busy.load(std::memory_order_acquire)) {
        if (__atomic_load_n(&r->abort, __ATOMIC_ACQUIRE)) return RAD_E_IO;
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    uint64_t off = 0;
    for (int i = 0; i < n; ++i) {
        std::memcpy(stage + off, a->t[1 + i].data, (size_t)bytes[i]);
        if (rad_pp_align(bytes[i]) > bytes[i]) std::memset(stage + off + bytes[i], 0, (size_t)(rad_pp_align(bytes[i]) - bytes[i]));
        off += rad_pp_align(bytes[i]);
    }
    std::vector<uint64_t> b(bytes, bytes + n);
    st.busy.store(true, std::memory_order_release);
    r->async_sends = r->async_sends + 1;
    std::thread([r, stage, n, b, total, rows, &st] {
        publish_from(r, stage, n, b.data(), total, rows);
        st.busy.store(false, std::memory_order_release);
    }).detach();
    return RAD_OK;
}
uint64_t loop_cap(const RadTensor& t) { return (uint64_t)rad_dtype_bytes(t.dtype, rad_tensor_numel(&t)); }

/* tp8 loop P8 (RADIANCE_PP_SEND_SPLIT=1 / 2), as r4d_stage.hip's stage_send_split_kernel on a rank's thread: wait for the
 * ring (1, and 2's publisher) or for the publisher's full copy (2's other ranks); copy (1) or compare (2) this rank's share
 * [lo, hi) of the payload's 256-byte units of every part from its own buffers; a rank other than `sender` counts the
 * payload in its `done`; `sender` waits for every other rank's, then publishes. */
static int wait_for(RadPpRing* r, const volatile uint64_t* w, uint64_t at_least) {
    const uint64_t t0 = mono_ms();
    while (__atomic_load_n(w, __ATOMIC_ACQUIRE) < at_least) {
        if (__atomic_load_n(&r->abort, __ATOMIC_ACQUIRE)) return RAD_E_IO;
        if (r->deadline_ms && mono_ms() - t0 > r->deadline_ms) return RAD_E_IO;
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    return __atomic_load_n(&r->abort, __ATOMIC_ACQUIRE) ? RAD_E_IO : RAD_OK;
}

static int send_split(const RadArgs* a, RadPpRing* r, int split, int sender, int n, const uint64_t* bytes, uint64_t total,
                      int64_t rows) {
    const int rank = a->rank;
    const bool pub = rank == sender;
    const uint64_t d = __atomic_load_n(&r->done[rank], __ATOMIC_ACQUIRE), mine = d + 1;
    RAD_REF_TRY(split == 1 || pub ? wait_for(r, &r->sent, d) : wait_for(r, &r->staged, mine));
    const uint64_t nr = r->send_ranks, U = total / RAD_PP_PART_ALIGN;
    const bool all = split == 2 && pub;
    const uint64_t lo = all ? 0 : (uint64_t)rank * U / nr * RAD_PP_PART_ALIGN;
    const uint64_t hi = all ? total : ((uint64_t)rank + 1) * U / nr * RAD_PP_PART_ALIGN;
    char* ring = reinterpret_cast<char*>(r) + RAD_PP_RING_HDR;
    uint64_t off = 0, bad = 0, cmp = 0;
    for (int i = 0; i < n; ++i) {
        const uint64_t s = std::max(lo, off), e = std::min(hi, off + bytes[i]);
        if (s < e) {
            const char* src = static_cast<const char*>(a->t[1 + i].data) + (s - off);
            if (split == 2 && !pub) {
                uint64_t bk = 0;
                for (uint64_t w = 0; w < (e - s) / 4; ++w) {
                    uint32_t x, y;
                    std::memcpy(&x, src + 4 * w, 4);
                    std::memcpy(&y, ring + s + 4 * w, 4);
                    bk += x != y ? 1 : 0;
                }
                bad += bk;
                if (bk) __atomic_store_n(&r->diff_part[rank][i], __atomic_load_n(&r->diff_part[rank][i], __ATOMIC_RELAXED) + bk,
                                         __ATOMIC_RELAXED);
                cmp += (e - s) / 4;
            } else {
                std::memcpy(ring + s, src, (size_t)(e - s));
            }
        }
        off += rad_pp_align(bytes[i]);
    }
    if (!pub) {
        if (split == 2) {
            __atomic_store_n(&r->diff_words[rank], __atomic_load_n(&r->diff_words[rank], __ATOMIC_RELAXED) + bad,
                             __ATOMIC_RELAXED);
            __atomic_store_n(&r->cmp_words[rank], __atomic_load_n(&r->cmp_words[rank], __ATOMIC_RELAXED) + cmp,
                             __ATOMIC_RELAXED);
        }
        __atomic_store_n(&r->done[rank], mine, __ATOMIC_RELEASE);
        return RAD_OK;
    }
    if (split == 2) __atomic_store_n(&r->staged, mine, __ATOMIC_RELEASE);
    for (uint64_t q = 0; q < nr; ++q)
        if ((int)q != sender) RAD_REF_TRY(wait_for(r, &r->done[q], mine));
    for (int i = 0; i < n; ++i) r->part_bytes[i] = bytes[i];
    std::memset(ring + total, 0, RAD_PP_TRAILER);
    *reinterpret_cast<uint64_t*>(ring + total + 8) = rad_pp_layout(bytes, n);
    __atomic_store_n(reinterpret_cast<uint64_t*>(ring + total), mine, __ATOMIC_RELEASE);
    r->rows = (uint64_t)rows;
    r->parts = (uint64_t)n;
    r->bytes = total + RAD_PP_TRAILER;
    __atomic_store_n(&r->done[rank], mine, __ATOMIC_RELEASE);
    __atomic_store_n(&r->ready, mine, __ATOMIC_RELEASE);
    r->ready_ns = mono_ns();
    return RAD_OK;
}

int ref_stage_send(const RadArgs* a, RadStream) {
    if (!a || a->n_t < 2) return RAD_E_INVAL;
    RadPpRing* r = static_cast<RadPpRing*>(a->t[0].data);
    const bool loop = loop_of(a);
    if (!r || (!loop && r->magic != RAD_PP_RING_MAGIC)) return RAD_E_STATE;
    if (loop && loop_cap(a->t[0]) < RAD_PP_RING_HDR) return RAD_E_FULL;
    const int sender = (int)rad_param_geti(a->p, a->n_p, "sender", 0);
    /* tp8 loop P8 (RadPpRing::send_split): every rank of the node takes part (1 its share, 2 a check of it) */
    const int split = !loop && (r->send_split == 1 || r->send_split == 2) && r->send_ranks >= 2 && sender >= 0 &&
                              (uint64_t)sender < r->send_ranks
                          ? (int)r->send_split : 0;
    if (split && (a->rank < 0 || (uint64_t)a->rank >= r->send_ranks)) return RAD_E_STATE;
    if (!loop && !split && a->rank != sender) return RAD_OK;
    int n = 0;
    int64_t rows = 0;
    uint64_t bytes[RAD_PP_MAX_PARTS] = {};
    RAD_REF_TRY(parts_of(a, &n, bytes, &rows));
    uint64_t total = 0;
    for (int i = 0; i < n; ++i) {
        if (!dense(a->t[1 + i])) return RAD_E_STRIDE;
        total += rad_pp_align(bytes[i]);
    }
    if (total + RAD_PP_TRAILER > (loop ? loop_cap(a->t[0]) - RAD_PP_RING_HDR : r->cap)) return RAD_E_FULL;
    if (split) return send_split(a, r, split, sender, n, bytes, total, rows);
    /* tp8 loop P10: a large payload through the node's staging buffer, published from a helper (the card's side stream) */
    if (!loop && r->async_rows && r->stage_dev && (uint64_t)rows >= r->async_rows && total <= r->stage_cap) {
        if (__atomic_load_n(&r->abort, __ATOMIC_ACQUIRE)) return RAD_E_IO;
        return send_async(a, r, n, bytes, total, rows);
    }
    if (loop) {   /* tp8 dev M4: a fresh header; the receive on this rank takes payload 1 */
        r->abort = 0;
        r->ready = 0;
        r->sent = 0;
        r->landed = 0;
        r->taken[a->rank] = 0;
        r->n_ranks = RAD_PP_MAX_RANKS;
        r->deadline_ms = 0;
    }
    /* The last payload must be off the ring (the link sets `sent`). */
    const uint64_t t0 = mono_ms();
    while (__atomic_load_n(&r->sent, __ATOMIC_ACQUIRE) < r->ready) {
        if (__atomic_load_n(&r->abort, __ATOMIC_ACQUIRE)) return RAD_E_IO;
        if (r->deadline_ms && mono_ms() - t0 > r->deadline_ms) return RAD_E_IO;
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    if (__atomic_load_n(&r->abort, __ATOMIC_ACQUIRE)) return RAD_E_IO;
    char* dst = reinterpret_cast<char*>(r) + RAD_PP_RING_HDR;
    uint64_t off = 0;
    for (int i = 0; i < n; ++i) {
        std::memcpy(dst + off, a->t[1 + i].data, (size_t)bytes[i]);
        r->part_bytes[i] = bytes[i];
        off += rad_pp_align(bytes[i]);
    }
    /* review F6: the payload's number in its trailer */
    const uint64_t seq = r->ready + 1;
    std::memset(dst + total, 0, RAD_PP_TRAILER);
    *reinterpret_cast<uint64_t*>(dst + total + 8) = rad_pp_layout(bytes, n);   /* tp8 S6 */
    __atomic_store_n(reinterpret_cast<uint64_t*>(dst + total), seq, __ATOMIC_RELEASE);
    r->rows = (uint64_t)rows;
    r->parts = (uint64_t)n;
    r->bytes = total + RAD_PP_TRAILER;
    if (loop) {   /* tp8 dev M4: to itself: landed and off the ring at once */
        __atomic_store_n(&r->sent, seq, __ATOMIC_RELEASE);
        __atomic_store_n(&r->landed, seq, __ATOMIC_RELEASE);
    }
    __atomic_store_n(&r->ready, seq, __ATOMIC_RELEASE);
    if (!loop) r->ready_ns = mono_ns();   /* tp8 S5: the stage busy stamps (read after the pass) */
    return RAD_OK;
}

int ref_stage_recv(const RadArgs* a, RadStream) {
    if (!a || a->n_t < 2) return RAD_E_INVAL;
    RadPpRing* r = static_cast<RadPpRing*>(a->t[0].data);
    const bool loop = loop_of(a);
    if (!r || (!loop && r->magic != RAD_PP_RING_MAGIC)) return RAD_E_STATE;
    const int rank = a->rank;
    if (rank < 0 || rank >= RAD_PP_MAX_RANKS || (uint64_t)rank >= r->n_ranks) return RAD_E_STATE;
    int n = 0;
    int64_t rows = 0;
    uint64_t bytes[RAD_PP_MAX_PARTS] = {};
    RAD_REF_TRY(parts_of(a, &n, bytes, &rows));
    uint64_t total = 0;
    for (int i = 0; i < n; ++i) {
        if (!dense(a->t[1 + i])) return RAD_E_STRIDE;
        total += rad_pp_align(bytes[i]);
    }
    const uint64_t want = __atomic_load_n(&r->taken[rank], __ATOMIC_ACQUIRE) + 1;
    const uint64_t t0 = mono_ms();
    uint32_t spins = 0;
    while (__atomic_load_n(&r->landed, __ATOMIC_ACQUIRE) < want) {
        if (__atomic_load_n(&r->abort, __ATOMIC_ACQUIRE)) return RAD_E_IO;
        if (r->deadline_ms && mono_ms() - t0 > r->deadline_ms) return RAD_E_IO;
        if (++spins > 64) std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    /* The sender's rows and layout must be this step's: a mismatch is a schedule bug (the two
     * nodes are not running the same step), never something to copy past. */
    if (r->rows != (uint64_t)rows || r->bytes != total + RAD_PP_TRAILER || r->parts != (uint64_t)n) return RAD_E_SHAPE;

    const char* src = reinterpret_cast<const char*>(r) + RAD_PP_RING_HDR;
    /* review F6: the payload waited for, not a stale one of the same shape */
    if (__atomic_load_n(reinterpret_cast<const uint64_t*>(src + total), __ATOMIC_ACQUIRE) != want) return RAD_E_STATE;
    /* tp8 S6: every part this step's size (parts carry their own rows) */
    if (*reinterpret_cast<const uint64_t*>(src + total + 8) != rad_pp_layout(bytes, n)) return RAD_E_SHAPE;
    uint64_t off = 0;
    for (int i = 0; i < n; ++i) {
        std::memcpy(a->t[1 + i].data, src + off, (size_t)bytes[i]);
        off += rad_pp_align(bytes[i]);
    }
    __atomic_store_n(&r->taken[rank], want, __ATOMIC_RELEASE);
    if (rank == 0 && !loop) r->taken_ns = mono_ns();   /* tp8 S5: the stage busy stamps */
    return RAD_OK;
}

/* A crossing has no cold form: its ring is the core's, and a call reads or writes another node.
 * A tool that sizes operands to call ops cold (rad-kbench, the oracle) skips it as unsupported. */
REF_SHAPE_FN(ref_stage_send) {
    (void)p; (void)n_p; (void)operand; (void)out;
    return RAD_E_UNSUPPORTED;
}
REF_SHAPE_FN(ref_stage_recv) {
    (void)p; (void)n_p; (void)operand; (void)out;
    return RAD_E_UNSUPPORTED;
}
