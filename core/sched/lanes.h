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

    /* Counters (any thread, under the lock): pieces lent, pieces committed, rows lent. */
    struct Counts { int64_t pieces = 0, done = 0, rows = 0, token_pieces = 0; };
    Counts counts() const {
        std::lock_guard<std::mutex> lk(*mu_);
        Counts c;
        c.pieces = lent_;
        c.done = done_;
        c.rows = rows_;
        c.token_pieces = tok_;
        return c;
    }
    void note_lent_locked(int64_t rows, bool tok) { rows_ += rows; if (tok) ++tok_; }
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
    int64_t split_min_ = 128;
    bool    head_state_ = false;
    int64_t lent_ = 0, done_ = 0, rows_ = 0, tok_ = 0;
    std::string broken_;
};

}  /* namespace rad */
