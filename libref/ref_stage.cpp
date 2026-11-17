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

/* The parts after the ring: [1, 1 + parts). Bytes of each, from its own extent. */
int parts_of(const RadArgs* a, int* n, uint64_t bytes[RAD_PP_MAX_PARTS], int64_t* rows) {
    *n = 0;
    *rows = -1;
    for (int i = 1; i < a->n_t && i <= RAD_PP_MAX_PARTS; ++i) {
        const RadTensor& t = a->t[i];
        if (!t.data) break;
        if (t.rank < 1) return RAD_E_SHAPE;
        if (*rows < 0) *rows = t.shape[0];
        else if (t.shape[0] != *rows) return RAD_E_SHAPE;   /* every part carries the step's rows */
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

int ref_stage_send(const RadArgs* a, RadStream) {
    if (!a || a->n_t < 2) return RAD_E_INVAL;
    RadPpRing* r = static_cast<RadPpRing*>(a->t[0].data);
    if (!r || r->magic != RAD_PP_RING_MAGIC) return RAD_E_STATE;
    const int sender = (int)rad_param_geti(a->p, a->n_p, "sender", 0);
    if (a->rank != sender) return RAD_OK;
    int n = 0;
    int64_t rows = 0;
    uint64_t bytes[RAD_PP_MAX_PARTS] = {};
    RAD_REF_TRY(parts_of(a, &n, bytes, &rows));
    uint64_t total = 0;
    for (int i = 0; i < n; ++i) {
        if (!dense(a->t[1 + i])) return RAD_E_STRIDE;
        total += rad_pp_align(bytes[i]);
    }
    if (total + RAD_PP_TRAILER > r->cap) return RAD_E_FULL;
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
    __atomic_store_n(reinterpret_cast<uint64_t*>(dst + total), seq, __ATOMIC_RELEASE);
    r->rows = (uint64_t)rows;
    r->parts = (uint64_t)n;
    r->bytes = total + RAD_PP_TRAILER;
    __atomic_store_n(&r->ready, seq, __ATOMIC_RELEASE);
    r->ready_ns = mono_ns();   /* tp8 S5: the stage busy stamps (read after the pass) */
    return RAD_OK;
}

int ref_stage_recv(const RadArgs* a, RadStream) {
    if (!a || a->n_t < 2) return RAD_E_INVAL;
    RadPpRing* r = static_cast<RadPpRing*>(a->t[0].data);
    if (!r || r->magic != RAD_PP_RING_MAGIC) return RAD_E_STATE;
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
    uint64_t off = 0;
    for (int i = 0; i < n; ++i) {
        std::memcpy(a->t[1 + i].data, src + off, (size_t)bytes[i]);
        off += rad_pp_align(bytes[i]);
    }
    __atomic_store_n(&r->taken[rank], want, __ATOMIC_RELEASE);
    if (rank == 0) r->taken_ns = mono_ns();   /* tp8 S5: the stage busy stamps */
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
