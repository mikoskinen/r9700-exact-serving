/* lanes.h -- tp8 S5: the lanes of a two-node pipeline (abi/rad_pp.h, core/engine_node.cpp).
 *
 * One engine on two machines runs a step's layers on one machine, then the other. With one step in
 * flight (tp8 S4) the two machines work in turn. Lanes make them work at once: the leader runs up to
 * --pp-lanes independent step streams, each a Scheduler of its own, and while one lane's step is on
 * the follower's layers another lane's is on the leader's.
 *
 *   - A sequence is bound to one lane for life (the lane its request was routed to: the one with the
 *     fewest live requests, then the fewest drafted rows). Each lane has its own request table, step
 *     plan, draft controller (so draft depth and dcap follow the lane's own occupancy), batch builder
 *     (its own staging slab on every card, so a step of one lane never overwrites another's arrays),
 *     and its own pair of rings on each node. Every lane has at most one step in flight.
 *   - The lanes share one lock (lane 0's) and the KV pool, and number their steps in one sequence,
 *     so a block id or a step number means one thing on both nodes. The card indices (StepSlots) are
 *     split between them, so no two lanes' sequences share a slot.
 *   - Newcomer split: a prefill chunk a lane plans is cut into one piece a lane, in queue order: the
 *     first piece runs in its own step, the others in the next steps of the other lanes, each of
 *     which plans before this lane plans again (the engine tells the set the order), so on every node
 *     the pieces run in position order and each piece's keys are in place before the next reads them.
 *     The lane that owns the request commits every piece (lane_piece_done). A lone cold prompt so
 *     alternates its pieces between the lanes: a two-stage chunk pipeline.
 *
 * Exactness rests on the build's invariance to step composition (which rows share a step, how a
 * prompt is cut into pieces): the bytes of every row are the same in any step that carries it. Lanes
 * and pieces change only composition. The pieces keep to the quantum grid and to at least
 * RADIANCE_PP_SPLIT_MIN rows (default 128, the line's row count above which no kernel form depends on
 * a piece's rows); RADIANCE_PP_SPLIT=0 turns the split off (lanes stay). On one lane nothing here is
 * built: the engine's scheduler runs as before.
 *
 * Every method below that names "_locked" is called with the lanes' lock held (from inside a lane
 * scheduler's own locked method); the others take it.
 */
#pragma once
#include "batch.h"
#include "scheduler.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace rad {

class LaneSet {
public:
    /* `lanes[0]` is the engine's scheduler; every other a Scheduler built on it (its lock). */
    void init(std::vector<Scheduler*> lanes);
    int  n() const { return (int)lanes_.size(); }
    Scheduler* lane(int l) const { return l >= 0 && l < (int)lanes_.size() ? lanes_[(size_t)l] : nullptr; }

    /* ---- requests (HTTP threads) */
    /* The lane a new request goes to: the fewest live requests (running + waiting), then the fewest
     * drafted rows, then the most free cards, then the lowest index (RADIANCE_PP_PICK=rr: equally
     * loaded lanes in turn). */
    int  pick() const;
    void set_pick_rr(bool on) { pick_rr_ = on; }
    /* The lane holding request `id`, or null. */
    Scheduler* owner(uint64_t id) const;
    /* tp8 dev D7: whether lane `l` has prompt rows to plan (a waiting request, or a running one whose prompt is not all
     * computed or lent): its next step would carry prefill rows (MIXED / prefill), not decode rows alone. Takes the lock. */
    bool prefill_pending(int l) const;
    /* ... and how many pieces other lanes lent it wait in its inbox (it must plan them in the promised order). */
    size_t inbox_n(int l) const {
        std::lock_guard<std::mutex> lk(*mu_);
        return l >= 0 && l < (int)inbox_.size() ? inbox_[(size_t)l].size() : 0;
    }
    /* Every lane's published copy (Scheduler::observed_metrics), summed: counters, queues and rates
     * add (the lanes run at once); the KV pool is one, so its groups are lane 0's. */
    SchedMetrics observed_metrics() const;
    /* Every lane's live request table, one after another. */
    int observed_requests(Scheduler::ReqStat* out, int max) const;

    /* ---- the newcomer split (engine thread, the lanes' lock held inside a lane's step()) */
    /* The engine sets, before a lane plans, the lanes that will plan before that lane plans again,
     * in the order they will: lanes with no step in flight first, then those in flight, oldest
     * first. A lane lends its pieces to them in that order. */
    void set_planners(std::vector<int> order) {
        std::lock_guard<std::mutex> lk(*mu_);
        order_ = std::move(order);
    }
    bool split_on() const { return split_on_; }
    /* tp8 prefill P1 (RADIANCE_PP_SPAN=0): a lone prefill under lanes plans --prefill-span's span pieces and cuts each at
     * its interior checkpoint-interval boundaries that leave >= 128 rows a side (Scheduler::plan_chunk, lane_span_cut):
     * every boundary of the span plan stays, the pieces between are whole intervals, so the lanes keep two of its chunks
     * in flight where a span chunk (4,096 rows) went only to an idle lane and filled the window pool at three. (Before
     * FINDINGS F3 21:38 it dropped the span: 64k's tail became 512 + 114 rows, its text moved.) Default on (1): spans. */
    bool span_on() const { return span_on_; }
    void set_span(bool on) { span_on_ = on; }
    /* tp8 loop P7 (RADIANCE_PP_SPAN_TAIL=X rows, 0 = off): the same cut as RADIANCE_PP_SPAN=0, but only for a span piece
     * that starts within the last X rows of its prompt (Scheduler::lane_span_cut): the span plan's 4,096-row chunks for
     * the body (half the chunks of the interval plan: each chunk costs the leader a fixed ~13-23 ms, lq18 / lq19) and
     * whole intervals at the end, where the follower's lag behind the leader is the drain (C' lq19: drain 609 / 771 /
     * 834 ms vs the span plan's 765 / 928 / 1003). Every boundary is one of the span plan's or an interval boundary,
     * as RADIANCE_PP_SPAN=0's (cold texts identical at 16k / 64k / 128k, lq19). Default 0: today's plan. */
    int64_t span_tail() const { return span_tail_; }
    void set_span_tail(int64_t rows) { span_tail_ = rows > 0 ? rows : 0; }
    /* tp8 loop M2 (RADIANCE_PP_LEND_BUSY=0): a chunk is lent only to a lane with no request of its own running or
     * waiting. Lent into a lane that decodes, a whole chunk is planned first in that lane's step (lane_reserve) and its
     * decoders ride a MIXED step of up to 2,048 rows: the stall the 20:58 PROFILE measured, now in the lane that had
     * none; and the prompt's plan is no longer the one a lane of decoders plans (lever DST / option D cut it on its grid
     * beside decode rows, whatever their count). Default 1: lent within the borrower's room, as before. */
    bool lend_busy() const { return lend_busy_; }
    void set_lend_busy(bool on) { lend_busy_ = on; }
    /* tp8 dev M5 (RADIANCE_PP_LEND_MIXED=1): a newcomer that lever BI's option D (sched_bi_chunk 3) cuts beside decode rows
     * lends the pieces one process cuts after this step's piece (its grid, >= 128 rows a side, the room one process plans
     * them at) one a lane to the lanes that plan next, as a cold prompt's chunks go: piece k's follower pass then runs beside
     * piece k - 1's leader tail, and the borrower's decoders keep a decode room beside the piece. Default off. */
    bool lend_mixed() const { return lend_mixed_; }
    void set_lend_mixed(bool on) { lend_mixed_ = on; }
    int64_t split_min() const { return split_min_; }
    /* Overrides the environment's (tests). */
    void set_split(bool on, int64_t min_rows) { split_on_ = on; split_min_ = min_rows > 0 ? min_rows : split_min_; }
    /* tp8 S5 + S6: the leader's head part writes sequence state (a cut W(h, t) with h > 0: layers [0, h) are the
     * head's). Then a chunk that takes a snapshot is never followed by another chunk of its sequence before its commit
     * (the next chunk's head would rewrite the layers [0, h) state the snapshot copies behind the tail). */
    void set_head_state(bool on) { head_state_ = on; }
    bool head_state() const { return head_state_; }

    struct Piece {
        int       home = -1;
        StepEntry e;
    };
    const std::vector<int>& planners_locked() const { return order_; }
    void    lend_locked(int to, const Piece& p, bool count = true) { inbox_[(size_t)to].push_back(p); if (count) ++lent_; }
    int64_t inbox_rows_locked(int lane) const;
    size_t  inbox_n_locked(int lane) const { return inbox_[(size_t)lane].size(); }
    /* Moves lane `lane`'s pieces into `out`. */
    void    take_locked(int lane, std::vector<Piece>* out);
    /* Takes back lane `home`'s piece of request `seq` at `ctx_len` from lane `to`'s inbox (a lender's
     * step whose build failed: Scheduler::lane_unsplit_own); false when it is not there. */
    bool    unlend_locked(int to, int home, uint64_t seq, int32_t ctx_len);
    void    unnote_lent_locked(int64_t rows, bool tok) { rows_ -= rows; if (tok) --tok_; }

    /* tp8 prefill P0: why a newcomer's chunk was not lent, or its sequence waited (Counts::why, the leader's lanes
     * line and each prompt's own line at its first token, Scheduler::lane_prompt_note):
     *   kOffGrid   its home chunk is not the one an idle engine plans there (other rows cut it): not lent at all;
     *   kSnap      a chunk taking a snapshot at a cut W(h > 0, t) (the head writes state): not lent, or lent last;
     *   kWideBusy  a span-wide chunk and the next planner is busy (wide chunks go only to an idle lane);
     *   kRoom      the next planner's inbox / cards are full;
     *   kKv        its KV (the window group from the keep floor) was refused (lends never preempt);
     *   kNoLane    every lane that plans before the home plans again has a chunk of it already;
     *   kHold      a plan that left the sequence out, waiting for a lent snapshot's commit (lent_hold);
     *   kWinWait   a plan in which its next home chunk waited for its lent chunks (KV refused while they hold it);
     *   kBusy      the next planner runs requests of its own and RADIANCE_PP_LEND_BUSY=0 (lend_busy). */
    enum { kOffGrid, kSnap, kWideBusy, kRoom, kKv, kNoLane, kHold, kWinWait, kBusy, kWhyN };
    static const char* why_name(int k) {
        static const char* const n[kWhyN] = { "off the one-process grid", "a snapshot at h > 0", "wide vs a busy lane",
                                              "borrower room", "KV refused", "no lane left", "held for a snapshot's commit",
                                              "waited for its lent chunks' window", "a borrower with requests of its own" };
        return k >= 0 && k < kWhyN ? n[k] : "?";
    }
    void note_why_locked(int k) { if (k >= 0 && k < kWhyN) ++why_[k]; }

    /* Counters (any thread, under the lock): pieces lent, pieces committed, rows lent. */
    struct Counts { int64_t pieces = 0, done = 0, rows = 0, token_pieces = 0; int64_t why[kWhyN] = {}; int64_t dpieces = 0; };
    Counts counts() const {
        std::lock_guard<std::mutex> lk(*mu_);
        Counts c;
        c.pieces = lent_;
        c.done = done_;
        c.rows = rows_;
        c.token_pieces = tok_;
        c.dpieces = dpiece_;
        for (int k = 0; k < kWhyN; ++k) c.why[k] = why_[k];
        return c;
    }
    void note_lent_locked(int64_t rows, bool tok) { rows_ += rows; if (tok) ++tok_; }
    /* tp8 dev M5: a newcomer option D cut beside decode rows lent its next pieces this step */
    void note_dpiece_locked() { ++dpiece_; }
    /* A piece was taken out of position order (the planning order the engine promised was not
     * kept): the pipeline's answers would be wrong, so the engine stops on it. */
    void note_broken_locked(const std::string& why) { if (broken_.empty()) broken_ = why; }
    std::string broken() const {
        std::lock_guard<std::mutex> lk(*mu_);
        return broken_;
    }
    void note_done_locked() { ++done_; }

private:
    std::vector<Scheduler*> lanes_;
    std::mutex* mu_ = nullptr;
    std::vector<int> order_;
    std::vector<std::vector<Piece>> inbox_;
    bool    pick_rr_ = false;               /* RADIANCE_PP_PICK=rr (pick) */
    mutable std::atomic<uint64_t> rr_{ 0 };
    bool    split_on_ = true;
    bool    span_on_ = true;                /* RADIANCE_PP_SPAN (span_on) */
    int64_t span_tail_ = 0;                 /* RADIANCE_PP_SPAN_TAIL (span_tail) */
    bool    lend_busy_ = true;              /* RADIANCE_PP_LEND_BUSY (lend_busy) */
    bool    lend_mixed_ = false;            /* RADIANCE_PP_LEND_MIXED (lend_mixed, tp8 dev M5) */
    int64_t split_min_ = 128;
    bool    head_state_ = false;
    int64_t lent_ = 0, done_ = 0, rows_ = 0, tok_ = 0, dpiece_ = 0;
    int64_t why_[kWhyN] = {};
    std::string broken_;
};

}  /* namespace rad */
