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

#include <algorithm>
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
    /* tp8 loop P7 (RADIANCE_PP_SPAN_SHORT=N rows, 0 = off): RADIANCE_PP_SPAN=0's cut for every span piece of a prompt of at
     * most N rows. On the cards the span plan's step from a 2,048- to a 4,096-row chunk leaves the follower idle ~one
     * 2,048-row chunk's time once (lq18 / lq25 16k 1.97-1.98 s), which interval chunks avoid (lq19 / lq20 C' 1.84-1.87 s)
     * at a fixed ~13-23 ms a chunk more on the leader -- the better trade for a short prompt, the worse for a long one
     * (C' 64k / 128k +2.4 / +1.2 %). Every boundary is the span plan's or an interval boundary. Default 0: today's plan. */
    int64_t span_short() const { return span_short_; }
    void set_span_short(int64_t rows) { span_short_ = rows > 0 ? rows : 0; }
    /* tp8 loop P34 (RADIANCE_PP_SPAN_FIRST=F rows, 0 = off): a lone cold prompt the span cut plans in intervals (SPAN=0, or
     * SPAN_SHORT's prompts) and that holds more than two intervals runs its FIRST interval as two pieces, [0, F) and [F, I),
     * when both have >= kBiMinPiece rows and F is on the quantum: the follower's first pass waits for an F-row head instead
     * of an I-row one (the pipeline's fill; BEST lq202b 16k: ~111 ms of its 1.40 s, the follower the bound). Every other
     * boundary is the cut plan's; F is off the resume grid and takes no checkpoint (the interval's end takes its own, as
     * before). Exact as every piece of the cut is (option D: no kernel form depends on a piece's rows from kBiMinPiece on).
     * Default 0: today's plan. */
    int64_t span_first() const { return span_first_; }
    void set_span_first(int64_t rows) { span_first_ = rows > 0 ? rows : 0; }
    /* tp8 loop P14 (RADIANCE_PP_SPAN_INNER=1, with RADIANCE_PP_SPAN=0 / SPAN_TAIL / SPAN_SHORT): a span piece that carries a
     * snapshot inside it (lever TI's tail snapshot of the prompt's last chunk) is cut too, at its first interior interval
     * boundary that leaves >= kBiMinPiece rows a side AND the snapshot in the later piece; that piece is planned again with
     * the snapshot inside it (its interval boundary takes the periodic one, as every cut piece's end does). At --prefill-span
     * 2 a cold prompt's last chunk spans one interval + its tail (no such boundary: 16k / 64k / 128k unchanged); at span 4
     * it spans up to three intervals + the tail (lq41: 5,537 / 6,258 / 6,209 rows into the drain), which this cuts to
     * intervals + the tail, the span-2 / interval plan's end. Default off: today's plan. */
    bool span_inner() const { return span_inner_; }
    void set_span_inner(bool on) { span_inner_ = on; }
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
    bool lend_mixed() const { return lend_mixed_ == 1; }
    void set_lend_mixed(bool on) { lend_mixed_ = on ? 1 : 0; }
    /* tp8 dev M5b (RADIANCE_PP_LEND_MIXED=2): the published admission (the newcomer planned whole beside decode rows, as one
     * process plans it) and, at the split, the home's whole-prompt MIXED entry cut on the 512 grid near its middle (>= 128
     * rows a side; the grid lq21's loadid found answer-neutral), its rest lent to ONE lane with an empty inbox and no request
     * waiting -- else the entry stays whole. */
    bool lend_split() const { return lend_mixed_ == 2; }
    void set_lend_split(bool on) { lend_mixed_ = on ? 2 : 0; }
    /* tp8 dev M8 (RADIANCE_PP_BG_LANE=k, 1 <= k < --pp-lanes): the LAST k lanes are background prefill lanes. They hold no
     * sequence (pick() and Scheduler::init's card split skip them: the other lanes split --max-num-seqs), and a newcomer that a
     * lane plans whole beside its decoders (a MIXED step's prompt entry that produces its first token, no prefix restore) is
     * lent WHOLE to one of them (Scheduler::lane_split_bg: the first in the planning order with room): its chunk is the home's
     * own plan -- no new cut -- and the home's step goes out with its decode rows alone. With k = 2 two newcomers are in flight
     * at once (one's leader tail beside the next one's follower segments). Cold prompts' chunks keep to the other lanes. The
     * follower runs the BG lanes' passes in layer segments (RADIANCE_PP_SEG_LAYERS, without RADIANCE_PP_SEG_PREFILL) with the
     * decode lanes' passes between them; the leader's RADIANCE_PP_READY_FIRST=1 lets decode tails go before them. Default 0. */
    static int bg_env();   /* k from the environment (0: off) */
    /* qfn-on-ds lever BGL (Qwen loop round 24, QUEUE Q5b S1 stage 1; RADIANCE_BG_LANE_1M=1, one machine, no --pp-nodes): two
     * lanes in one process -- lane 0 holds every sequence, lane 1 is M8's background prefill lane (bg_env() reads 1 under it
     * when RADIANCE_PP_BG_LANE is unset). Stage 1 runs the BG lane's steps on the same queue between lane 0's steps
     * (Engine::step_loop), so it changes composition only (a newcomer's prompt in a lone step instead of a MIXED one);
     * stage 2 issues them on a second Ctx a rank beside lane 0's decode steps. Off by default. */
    static bool bg1m_env();
    /* RADIANCE_BG_LANE_1M=2: stage 2 -- as 1, plus a second Ctx a rank for lane 1's steps (Rank::bg_ctx; WIP: built and budgeted,
     * the loop does not issue on it yet) */
    static bool bg1m2_env();
    /* RADIANCE_BG_LANE_1M=3: stage 2b -- as 2, with lane 1's step issued whole on the second Ctx and finished by its event while lane 0
     * steps on (the overlap) */
    static bool bg1m3_env();
    /* tp8 loop D47 (RADIANCE_PP_BG_ROWS=N, 1 .. 256; with BG lanes, the same value on both nodes): a BG lane's step slab carves
     * its block tables for N rows instead of --max-num-seqs (each row --max-model-len wide: at 32 x 1 M that is ~512 MiB a card
     * a lane, the whole of a BG lane's slab), and lane_split_bg lends into a BG lane only while its inbox holds fewer than N
     * pieces (a newcomer then goes to the next BG lane with room, or runs at home, as when its rows do not fit). A BG lane's
     * step is its inbox (it holds no sequence of its own), so its plans keep to N sequences; build() refuses more by name.
     * Capacity / step composition only. 0 (unset): --max-num-seqs, today. bg_rows(max_seqs) = a BG lane's cap. */
    static int64_t bg_rows_env();
    int64_t bg_rows(int64_t max_seqs) const {
        return bg_lane() >= 0 && bg_rows_n_ > 0 && bg_rows_n_ < max_seqs ? bg_rows_n_ : max_seqs;
    }
    void note_bg_rows_kept_locked() { ++bg_rows_kept_; }
    /* tp8 loop D47b (RADIANCE_PP_LANE_ROWS=M, 1 .. 256; the same value on both nodes): a decode lane (not a BG lane, not lane 0 =
     * the MERGE target, which keeps --max-num-seqs rows) carves its
     * block tables for its own cards + M rows (its own sequences never exceed its cards: Scheduler::init's split), and
     * lane_split_own lends a piece into it only while its inbox holds fewer than M (else the piece stays home, as when its rows
     * do not fit). 0 (unset): --max-num-seqs, today. lane_rows(cards, max_seqs) = a decode lane's rows. */
    static int64_t lane_rows_env();
    int64_t lane_rows_m() const { return lane_rows_m_; }
    static int64_t lane_rows(int64_t cards, int64_t m, int64_t max_seqs) {
        return m > 0 && cards + m < max_seqs ? cards + m : max_seqs;
    }
    void note_lane_rows_kept_locked() { ++lane_rows_kept_; }
    int64_t lane_rows_kept() const { std::lock_guard<std::mutex> lk(*mu_); return lane_rows_kept_; }
    int64_t bg_rows_kept() const { std::lock_guard<std::mutex> lk(*mu_); return bg_rows_kept_; }
    /* the first BG lane, or -1 */
    int  bg_lane() const { return bg_ > 0 && (int)lanes_.size() > bg_ ? (int)lanes_.size() - bg_ : -1; }
    int  bg_n() const { return bg_lane() >= 0 ? bg_ : 0; }
    bool is_bg(int l) const { return bg_lane() >= 0 && l >= bg_lane() && l < (int)lanes_.size(); }
    void set_bg(int k) { bg_ = k > 0 ? k : 0; }
    /* M8's counts: newcomers lent whole to the BG lane, their rows, and why one stayed home (a prefix restore, the BG lane's
     * room, the BG lane not among this plan's planners) */
    enum { kBgRestore, kBgRoom, kBgOrder, kBgSplitRoom, kBgWhyN };
    void note_bg_locked(int64_t rows) { ++bg_lent_; bg_rows_ += rows; }
    /* RADIANCE_PP_BG_SPLIT=1 (with two BG lanes or more): a newcomer cut on the 512 grid near its middle, its two pieces to two
     * BG lanes in the planning order (Scheduler::lane_split_bg) */
    bool bg_split() const { return bg_split_; }
    void set_bg_split(bool on) { bg_split_ = on; }
    void note_bg_split_locked() { ++bg_split_n_; }
    /* tp8 loop F22 (with BG lanes): a lent piece's commit records the block
     * drafter's range in its home's request as the home's own Scheduler::block_end would have (the lane that ran the piece
     * ran the drafter's context pass over its rows: block_begin mirrors every entry, the drafter's KV is the sequence's in
     * the one pool). A newcomer lent WHOLE to a BG lane otherwise reached its home's first block_begin with no range, so
     * draft_floor() was its prompt's end (hit_holes_draft_) and it drafted nothing for its first window of tokens (lq46:
     * 1,582 undrafted decode rows in the c16 window, 13 newcomers x ~128). Always on where BG lanes are on (its opt-out
     * RADIANCE_PP_BG_DRAFT_FLOOR=0 was removed 18.11., tp8 loop REPORT rule 7). */
    bool bg_draft_floor() const { return bg_lane() >= 0; }
    void note_bg_floor_locked() { ++bg_floor_n_; }
    void note_bg_skip_locked(int k) { if (k >= 0 && k < kBgWhyN) ++bg_skip_[k]; }
    /* tp8 dev D11 (RADIANCE_PP_BG_HOLD_MS=N, 1 .. 60000; with BG lanes): a newcomer the BG lanes have no room for (kBgRoom) is
     * left out of its home's step -- the step goes out with its decode rows alone -- and planned again at the home's next step,
     * when a BG lane may take it whole; after N ms of holding it runs at home as today's MIXED step. A held newcomer has no
     * prefix restore planned (those stay home, kBgRestore). 0 (unset): off, today's MIXED step at once. */
    int64_t bg_hold_ms() const {
        if (bg_lane() < 0) return 0;
        /* tp8 MERGE (pp_merge bit 16 with the merge on): D11's hold at merge_hold_ms (RADIANCE_PP_MERGE_HOLD_MS) -- a newcomer the
         * BG lanes have no room for waits for them (the lanes merge for the BG work in flight) instead of a MIXED step */
        return bg_hold_ms_ > 0 ? bg_hold_ms_ : merge_hold_ms_;
    }
    void set_merge_hold_ms(int64_t ms) { merge_hold_ms_ = ms > 0 ? ms : 0; }
    void set_bg_hold_ms(int64_t ms) { bg_hold_ms_ = ms > 0 ? ms : 0; }
    /* tp8 loop P23 (RADIANCE_PP_BG_ROOM=N, 256 .. 65536 rows; with BG lanes): a BG lane's inbox -- and so its step -- may hold up to
     * N rows of lent newcomers (capped at --max-num-batched-tokens) instead of the shared budget (min(MNBT, ChunkGeometry::
     * shared_tok), 2112 at the DS geometry: one ~1.1k newcomer), so the newcomers that arrive while a BG step runs go to the BG
     * lane's next step TOGETHER (one pass, each prompt its own BI segment and PSK cone; under fn4 ds41_bi 7 one routed-MoE pass
     * a layer for all of them) instead of running at home as MIXED steps. A step of several sequences above the shared budget is
     * what lever NOA plans on one machine. 0 (unset): the shared budget, today. bg_room_cap(base) = the cap lane_split_bg uses. */
    int64_t bg_room() const { return bg_lane() >= 0 ? bg_room_ : 0; }
    void set_bg_room(int64_t rows) { bg_room_ = rows > 0 ? rows : 0; }
    int64_t bg_room_cap(int64_t base, int64_t max_tok) const {
        return bg_room() > 0 ? std::max(base, std::min(bg_room(), max_tok)) : base;
    }
    /* its counts (every BG lane step that took lent pieces): steps, newcomers (pieces with their first token), steps with >= 2 /
     * >= 3 newcomers, their rows, the widest step */
    void note_bg_step_locked(int64_t newc, int64_t rows) {
        ++bgst_n_;
        bgst_newc_ += newc;
        if (newc >= 2) { ++bgst_b2_; bgst_b2_newc_ += newc; bgst_b2_rows_ += rows; }
        if (newc >= 3) ++bgst_b3_;
        if (rows > bgst_max_rows_) bgst_max_rows_ = rows;
    }
    /* tp8 loop P23b (RADIANCE_PP_BG_JOIN_MS=N, 1 .. 60000; with RADIANCE_PP_BG_ROOM): a newcomer that the shared budget's room
     * would keep home (today's MIXED step at once) joins a BUSY BG lane's inbox past that room only when the BG lane's step in
     * flight is predicted to commit within N ms -- its rows x the BG steps' measured us a row (plan to commit, a running mean) --
     * else it runs at home as today, so a batch never makes a newcomer wait out a long BG step. No measurement yet, or no step in
     * flight: it joins (P23's rule). 0 (unset): P23 as it is. */
    int64_t bg_join_ms() const { return bg_room() > 0 ? bg_join_ms_ : 0; }
    void set_bg_join_ms(int64_t ms) { bg_join_ms_ = ms > 0 ? ms : 0; }
    /* a BG lane took its inbox (`rows`) into a step at `now_us` / committed that step */
    void note_bg_take_locked(int lane, int64_t rows, int64_t now_us) {
        if (lane < 0 || (size_t)lane >= bgt0_.size()) return;
        bgt0_[(size_t)lane] = now_us > 0 ? now_us : 1;
        bgrows_[(size_t)lane] = rows;
    }
    void note_bg_commit_locked(int lane, int64_t now_us) {
        if (lane < 0 || (size_t)lane >= bgt0_.size() || bgt0_[(size_t)lane] <= 0) return;
        const int64_t rows = bgrows_[(size_t)lane], us = now_us - bgt0_[(size_t)lane];
        bgt0_[(size_t)lane] = 0;
        if (rows <= 0 || us <= 0) return;
        const double r = (double)us / (double)rows;
        bg_us_row_ = bg_us_row_n_ == 0 ? r : 0.8 * bg_us_row_ + 0.2 * r;   /* a running mean that follows the load */
        ++bg_us_row_n_;
    }
    /* us until lane's BG step in flight is predicted to commit: 0 with none in flight, -1 with no measurement yet */
    int64_t bg_remaining_us_locked(int lane, int64_t now_us) const {
        if (lane < 0 || (size_t)lane >= bgt0_.size() || bgt0_[(size_t)lane] <= 0) return 0;
        if (bg_us_row_n_ == 0) return -1;
        return std::max<int64_t>(0, bgt0_[(size_t)lane] + (int64_t)(bg_us_row_ * (double)bgrows_[(size_t)lane]) - now_us);
    }
    void note_bg_join_locked(bool joined) { ++(joined ? bg_join_yes_ : bg_join_no_); }
    /* its counts: steps a newcomer was held out of, newcomers held, of them lent to a BG lane later / run at home after the
     * hold ran out, and the held time (first hold to its lend or home step) in all and the longest */
    void note_hold_locked(bool first) { ++hold_steps_; if (first) ++hold_reqs_; }
    void note_hold_end_locked(bool lent, int64_t us) {
        ++(lent ? hold_lent_ : hold_home_);
        hold_us_ += us;
        if (us > hold_max_us_) hold_max_us_ = us;
    }
    /* ---- tp8 MERGE (fn4 pp_merge; Engine::step_loop_lanes): lane membership in process. The decode lanes (all but the BG lanes)
     * run as ONE lane `to` while merged: the others are parked (never planned, never a planner), pick() names `to`, and their
     * requests and free cards move to `to`. A MOVE is a request's SchedReq leaving one lane's tables for another's (a new slot,
     * the same card, KV, drafter range, proposal and estimates), under the lock, between two IDLE lanes (no step open or issued
     * ahead in either). Movable: in running_ / waiting_, in no step (no lent piece in flight or in an inbox, no held snapshot),
     * not pending free or reap, no kept context rows (the depth-0 skip's rows live in their lane's ring region). Exact: a move
     * changes only which lane's steps carry the sequence -- composition, as lanes themselves do. */
    struct MoveCounts {
        int64_t moved = 0, cards = 0;   /* requests moved, free cards moved */
        int64_t lent = 0, ctx = 0, other = 0, busy = 0;   /* left where they were: a lent piece, kept rows, a step / free; a busy lane */
    };
    /* every movable request and free card of the decode lanes other than `to` into `to`; `to`'s drafter takes the merged rule */
    MoveCounts merge_into(int to);
    /* `from`'s movable requests over the decode lanes (live requests balanced, every other one in its running order first),
     * free cards so every decode lane holds max_seqs / decode lanes (held + free); `from`'s drafter back to its own rule */
    MoveCounts split_from(int from);
    int  merged_to() const { return merged_to_.load(std::memory_order_acquire); }
    bool parked(int l) const {
        const int m = merged_to();
        return m >= 0 && l != m && l >= 0 && l < (int)lanes_.size() && !is_bg(l);
    }
    /* requests a parked lane still holds (any state), read under the lock: 0 = nothing left to move */
    int64_t parked_live() const;
    /* lane l's live requests (running + waiting), under the lock */
    int64_t live(int l) const;
    /* tp8 MERGE bit 32 (the relay lane): while merged a PARKED decode lane with no request of its own takes whole newcomers as a
     * BG lane does (lane_split_bg names it after the idle BG lanes), so a third newcomer's head can start beside the other two's
     * parts instead of waiting for a BG lane's commit; never a cold prompt's chunk */
    void set_relay(bool on) { relay_ = on; }
    bool relay_bg(int l) const { return relay_ && parked(l); }
    /* newcomers held out of their decode lane's steps for the BG lanes now (D11's hold: SchedReq::bg_hold_us), read under the lock */
    int64_t held_now() const;
    int  decode_n() const { return (int)lanes_.size() - bg_n(); }
    int64_t split_min() const { return split_min_; }
    /* Overrides the environment's (tests). */
    void set_split(bool on, int64_t min_rows) { split_on_ = on; split_min_ = min_rows > 0 ? min_rows : split_min_; }
    /* tp8 S5 + S6: the leader's head part writes sequence state (a cut W(h, t) with h > 0: layers [0, h) are the
     * head's). Then a chunk that takes a snapshot is never followed by another chunk of its sequence before its commit
     * (the next chunk's head would rewrite the layers [0, h) state the snapshot copies behind the tail). */
    void set_head_state(bool on) { head_state_ = on; }
    bool head_state() const { return head_state_; }
    /* qfn-on-ds Qwen tp8 S7b (round 43; RADIANCE_PP_HEAD_SNAP=1 on the leader, Engine::head_stage): the leader stages a
     * chunk-END snapshot's head-layer rows inside the chunk's HEAD part, so such a chunk no longer holds its sequence; a
     * snapshot inside a chunk (lever TI / TIP) still does. */
    void set_head_snap(bool on) { head_snap_ = on; }
    bool head_snap() const { return head_snap_; }
    bool snap_holds(int64_t ckpt_tok, int64_t n_tokens) const {
        return head_state_ && (!head_snap_ || ckpt_tok < n_tokens - 1);
    }

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
    struct Counts {
        int64_t pieces = 0, done = 0, rows = 0, token_pieces = 0; int64_t why[kWhyN] = {}; int64_t dpieces = 0;
        int64_t bg = 0, bg_rows = 0, bg_skip[kBgWhyN] = {}, bg_split = 0;   /* tp8 dev M8 */
        int64_t bg_floor = 0;   /* tp8 loop F22: requests whose drafter range a lent piece's commit opened */
        int64_t hold_steps = 0, hold_reqs = 0, hold_lent = 0, hold_home = 0, hold_us = 0, hold_max_us = 0;   /* tp8 dev D11 */
        /* tp8 loop P23: BG lane steps with lent pieces, their newcomers, steps with >= 2 / >= 3 newcomers (+ the >= 2 ones'
         * newcomers and rows), the widest BG step */
        int64_t bgst_n = 0, bgst_newc = 0, bgst_b2 = 0, bgst_b3 = 0, bgst_b2_newc = 0, bgst_b2_rows = 0, bgst_max_rows = 0;
        /* tp8 loop P23b: newcomers past the shared room that joined a busy BG lane / were sent home by RADIANCE_PP_BG_JOIN_MS, the
         * BG steps' measured us a row (x 100) */
        int64_t bg_join_yes = 0, bg_join_no = 0, bg_us_row_x100 = 0;
    };
    Counts counts() const {
        std::lock_guard<std::mutex> lk(*mu_);
        Counts c;
        c.pieces = lent_;
        c.done = done_;
        c.rows = rows_;
        c.token_pieces = tok_;
        c.dpieces = dpiece_;
        for (int k = 0; k < kWhyN; ++k) c.why[k] = why_[k];
        c.bg = bg_lent_;
        c.bg_rows = bg_rows_;
        for (int k = 0; k < kBgWhyN; ++k) c.bg_skip[k] = bg_skip_[k];
        c.bg_split = bg_split_n_;
        c.bg_floor = bg_floor_n_;
        c.hold_steps = hold_steps_;
        c.hold_reqs = hold_reqs_;
        c.hold_lent = hold_lent_;
        c.hold_home = hold_home_;
        c.hold_us = hold_us_;
        c.hold_max_us = hold_max_us_;
        c.bgst_n = bgst_n_;
        c.bgst_newc = bgst_newc_;
        c.bgst_b2 = bgst_b2_;
        c.bgst_b3 = bgst_b3_;
        c.bgst_b2_newc = bgst_b2_newc_;
        c.bgst_b2_rows = bgst_b2_rows_;
        c.bgst_max_rows = bgst_max_rows_;
        c.bg_join_yes = bg_join_yes_;
        c.bg_join_no = bg_join_no_;
        c.bg_us_row_x100 = (int64_t)(bg_us_row_ * 100.0);
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
    int64_t span_short_ = 0;                /* RADIANCE_PP_SPAN_SHORT (span_short) */
    int64_t span_first_ = 0;                /* RADIANCE_PP_SPAN_FIRST (span_first, tp8 loop P34) */
    bool    span_inner_ = false;            /* RADIANCE_PP_SPAN_INNER (span_inner, tp8 loop P14) */
    bool    lend_busy_ = true;              /* RADIANCE_PP_LEND_BUSY (lend_busy) */
    int     lend_mixed_ = 0;                /* RADIANCE_PP_LEND_MIXED 1 (lend_mixed, tp8 dev M5) / 2 (lend_split, M5b) */
    int     bg_ = 0;                        /* RADIANCE_PP_BG_LANE=k (bg_lane, tp8 dev M8) */
    int64_t bg_lent_ = 0, bg_rows_ = 0, bg_skip_[kBgWhyN] = {}, bg_split_n_ = 0;
    bool    bg_split_ = false;              /* RADIANCE_PP_BG_SPLIT=1 (bg_split, tp8 dev M8) */
    int64_t bg_floor_n_ = 0;
    int64_t bg_hold_ms_ = 0;                /* RADIANCE_PP_BG_HOLD_MS (bg_hold_ms, tp8 dev D11) */
    int64_t bg_room_ = 0;                   /* RADIANCE_PP_BG_ROOM (bg_room, tp8 loop P23) */
    int64_t bg_rows_n_ = 0;                 /* RADIANCE_PP_BG_ROWS (bg_rows, tp8 loop D47) */
    int64_t bg_rows_kept_ = 0;              /* lends a BG lane's rows cap turned away (tp8 loop D47) */
    int64_t lane_rows_m_ = 0;               /* RADIANCE_PP_LANE_ROWS (lane_rows_m, tp8 loop D47b) */
    int64_t lane_rows_kept_ = 0;            /* lends a decode lane's rows cap turned away (tp8 loop D47b) */
    int64_t bg_join_ms_ = 0;                /* RADIANCE_PP_BG_JOIN_MS (bg_join_ms, tp8 loop P23b) */
    std::vector<int64_t> bgt0_, bgrows_;    /* a BG lane's step in flight: when it took its inbox, its rows */
    double  bg_us_row_ = 0.0;
    int64_t bg_us_row_n_ = 0, bg_join_yes_ = 0, bg_join_no_ = 0;
    int64_t bgst_n_ = 0, bgst_newc_ = 0, bgst_b2_ = 0, bgst_b3_ = 0, bgst_b2_newc_ = 0, bgst_b2_rows_ = 0, bgst_max_rows_ = 0;
    int64_t hold_steps_ = 0, hold_reqs_ = 0, hold_lent_ = 0, hold_home_ = 0, hold_us_ = 0, hold_max_us_ = 0;
    int64_t split_min_ = 128;
    bool    head_state_ = false;
    bool    head_snap_ = false;    /* Qwen tp8 S7b */
    int64_t lent_ = 0, done_ = 0, rows_ = 0, tok_ = 0, dpiece_ = 0;
    int64_t why_[kWhyN] = {};
    std::string broken_;
    std::atomic<int> merged_to_{ -1 };      /* tp8 MERGE: the merged decode lane, -1 split */
    int64_t merge_hold_ms_ = 0;             /* tp8 MERGE bit 16: D11's hold while merged (bg_hold_ms) */
    bool    relay_ = false;                 /* tp8 MERGE bit 32: parked lanes take whole newcomers (relay_bg) */
    /* tp8 MERGE: why `q` (in `s`) may not move now (MoveCounts' fields 2..4 as 0 / 1 / 2), -1 movable; and the move itself */
    int  unmovable_locked(const Scheduler& s, const SchedReq& q) const;
    void move_locked(Scheduler& src, int32_t slot, Scheduler& dst);
};

}  /* namespace rad */
