/* engine_lanes.cpp -- tp8 S5: the leader of a two-node pipeline with lanes (core/sched/lanes.h).
 *
 * One lane (tp8 S4, Engine::step_loop) runs a step's three stage passes in turn -- the leader's head
 * (the embedding, stage_send X1), the follower's layers, the leader's tail (stage_recv X2, its layers,
 * the head, the sampler) -- and only then builds the next step, so each machine waits on the other.
 * With N lanes each lane is a scheduler with one step in flight, and the leader issues the lanes'
 * passes in one queue order on its stream:
 *
 *   leader   | head A1 | tail B0 ......| head B1 | tail A1 ......| head A2 | tail B1 ......|
 *   follower |    | mid A1 ......| mid B1 ......| mid A2 ......| mid B2 ......|
 *
 * Every tail waits on the card for its lane's X2 (stage_recv), every follower pass for its X1, so
 * a pass issued before its input lands costs no host round trip. The loop:
 *
 *   1. plan every lane with no step in flight (its scheduler's step()), in turn from the lane after
 *      the last one planned, ship its build to the follower and issue its head pass;
 *   2. issue the tail of the oldest lane step in flight (heads go out first, so the follower always
 *      has the next lane's X1), its drafter's passes behind it, wait for its tokens, commit, finish
 *      its drafter; that lane is then free and step 1 plans it again.
 *
 * Serialised on purpose, with the reason:
 *   - one tail is outstanding at a time: the sampler's staging, the readback, the drafter's slots,
 *     the step-end event and StepOut are the step loop's single copies, so a second tail would need
 *     its own (the leader's host gap between two tails stays on its stream: section "what needs
 *     cards" of S5-LANES.md prices it);
 *   - no step is issued ahead (issue_ahead): its rows would be advanced on the leader's cards, which
 *     the follower does not have (positions, slots, acceptance; S6 can carry them in X1);
 *   - a switch change (fn4) lands at a head release, as at a step release on one machine; a lane step
 *     already in flight then runs its tail under the new setting (its head and the follower's pass ran
 *     under the old): exact levers only change speed; one that changes arithmetic is changed with
 *     the pipeline idle;
 *   - the prefix cache's host tier stays off (tp8 S6); a prefix hit's restores go before its head (on the
 *     follower before its pass, from the step frame: pp_ck_list), its saves behind its tail (on the follower
 *     behind its pass). A chunk with a checkpoint or a restore is never lent in pieces (lanes.cpp), and the
 *     sequence's next chunk is planned only after its commit, so no other lane's part writes the state a
 *     save copies between the head and the save; a snapshot is the prefix cache's only from its step's
 *     commit (commit_checkpoint), so a restore from it is planned behind the save on both nodes;
 *   - the KV-copy check (RADIANCE_PP_KV_CHECK) compares the copies with no lane step in flight: when one is
 *     due, the loop stops planning until every tail is in, then sends it (two lanes' copies would otherwise
 *     differ by the step whose head ran and whose tail has not);
 *   - the depth-0 context skip (draft_ctxskip) keeps the trunk's rows of consecutive steps in one
 *     ring on the card; lanes interleave their steps, so the skip is refused (fn4 and at start).
 * Every lane's step numbers come from one counter (lanes.h): a step number names one build on both
 * nodes. The tail's and the drafter's state belong to one lane at a time: cur_sched_ is that lane
 * from its tail release to its drafter's finish.
 */
#include "engine_priv.h"

#include "net/node_link.h"
#include "runtime/fn4_switch.h"
#include "util/stallwatch.h"
#include "device/device.h"   /* tp8 loop P7: card-local events (rad_event_create_local) */
#include "pp_nodeidle.h"     /* tp8 dev D8 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <thread>

namespace rad {

bool pp_x1_ids();   /* engine_node.cpp: the plugin declared X1's token-ids part (tp8 dev D10 round 2) */

/* tp8 loop P7 (RADIANCE_PP_PF_TIMELINE=1; FINDINGS F6 22:37): where a lone cold prompt's time goes on the leader. The
 * leader's chunk rhythm on the cards fits rows x 0.078-0.081 ms + a fixed 13-23 ms a chunk (lq18 / lq19), and a prefill
 * pass is unrecorded, so RADIANCE_AQL_STAMP does not see it. Here card-local events go onto rank 0's stream around every
 * part of a lone prefill's lane steps (A before a head's release, B after its ranks issued it, C before a tail's
 * release, D after its drafter): an event completes when the stream reaches it, so the time from one record to the next
 * is that part's card time (B - A, D - C: a tail waits on the card for its X2) or, from a part's end to the next part's
 * start, what the card did meanwhile (idle while the host had not issued the next part, or work outside the brackets).
 * The host's own time a chunk (plan, head release = prepare + restores + ship, the ranks' head issue, the tail's issue,
 * collect + commit + drafter finish) beside it; one line a prompt once its first token is in. Events are taken from a
 * pool (at most kCap) and read in stream order with rad_event_query, so nothing waits on them. */
struct PfTimeline {
    enum Kind : uint8_t { kA, kB, kC, kD };
    static constexpr size_t kCap = 2048;
    struct Rec { RadEvent ev; uint8_t kind; uint64_t seq; int ci; };
    struct Chunk { int64_t rows = 0, ctx = 0; float gap_a = -1, head = -1, gap_c = -1, tail = -1; };
    struct Prompt {
        std::vector<Chunk> ch;
        double plan = 0, rel = 0, issue = 0, tissue = 0, fin = 0;
        int64_t nh = 0, nt = 0;
    };
    RadStream s = nullptr;
    std::vector<RadEvent> pool;
    size_t made = 0;
    std::deque<Rec> recs;
    Rec last{ nullptr, 0, 0, -1 };
    std::map<uint64_t, Prompt> by_seq;
    std::map<int, std::pair<uint64_t, int>> by_step;   /* step -> (seq, chunk) between its head and its tail */
    double plan_ms = 0;                                /* the loop's last plan, for the head that follows it */
    int64_t dropped = 0, unread = 0;

    ~PfTimeline() {
        for (RadEvent e : pool) rad_event_destroy(e);
        for (const Rec& r : recs) rad_event_destroy(r.ev);
        if (last.ev) rad_event_destroy(last.ev);
    }
    RadEvent take() {
        if (!pool.empty()) { RadEvent e = pool.back(); pool.pop_back(); return e; }
        if (made >= kCap) return nullptr;
        RadEvent e = nullptr;
        if (rad_event_create_local(&e) < 0) return nullptr;
        ++made;
        return e;
    }
    void mark(Kind k, uint64_t seq, int ci) {
        RadEvent e = take();
        if (!e) { ++dropped; return; }
        if (rad_event_record(e, s) < 0) { pool.push_back(e); ++dropped; return; }
        recs.push_back({ e, (uint8_t)k, seq, ci });
    }
    /* The records whose events completed, in stream order: each one's time since its predecessor. */
    void drain() {
        while (!recs.empty() && rad_event_query(recs.front().ev) == 1) {
            const Rec r = recs.front();
            recs.pop_front();
            if (last.ev && last.seq == r.seq) {
                float ms = 0.0f;
                auto it = by_seq.find(r.seq);
                if (rad_event_elapsed_ms(last.ev, r.ev, &ms) < 0) ++unread;
                else if (it != by_seq.end() && r.ci >= 0 && (size_t)r.ci < it->second.ch.size()) {
                    Chunk& c = it->second.ch[(size_t)r.ci];
                    (r.kind == kA ? c.gap_a : r.kind == kB ? c.head : r.kind == kC ? c.gap_c : c.tail) = ms;
                }
            }
            if (last.ev) pool.push_back(last.ev);
            last = r;
        }
    }
};

/* tp8 loop P7: a timeline chunk's collect is in; at its prompt's first token, the prompt's line (every record up to its
 * last tail has completed: the collect waited for that step's end). */
static void pf_finish(PfTimeline* tl, const RadBatch* b, uint64_t t0, bool tok) {
    auto it = tl->by_step.find(b->step);
    if (it == tl->by_step.end()) { tl->drain(); return; }
    const uint64_t seq = it->second.first;
    tl->by_step.erase(it);
    auto pit = tl->by_seq.find(seq);
    if (pit != tl->by_seq.end()) pit->second.fin += (double)(rad_mono_ns() - t0) / 1e6;
    tl->drain();
    if (!tok || pit == tl->by_seq.end()) return;
    const PfTimeline::Prompt& P = pit->second;
    const size_t n = P.ch.size();
    int64_t rows = 0;
    double head = 0, tail = 0, ga = 0, gc = 0, gmax = -1;
    int nga = 0, ngc = 0, nhead = 0, ntail = 0;
    size_t gmax_at = 0;
    bool gmax_head = true;
    for (size_t k = 0; k < n; ++k) {
        const PfTimeline::Chunk& c = P.ch[k];
        rows += c.rows;
        if (c.head >= 0) { head += c.head; ++nhead; }
        if (c.tail >= 0 && k + 1 < n) { tail += c.tail; ++ntail; }
        if (c.gap_a >= 0) {
            ga += c.gap_a; ++nga;
            if (c.gap_a > gmax) { gmax = c.gap_a; gmax_at = k; gmax_head = true; }
        }
        if (c.gap_c >= 0) {
            gc += c.gap_c; ++ngc;
            if (c.gap_c > gmax) { gmax = c.gap_c; gmax_at = k; gmax_head = false; }
        }
    }
    const double nh = std::max<int64_t>(P.nh, 1), nt = std::max<int64_t>(P.nt, 1);
    const float last = n ? P.ch[n - 1].tail : -1.0f;
    RAD_INFO("prefill timeline (RADIANCE_PP_PF_TIMELINE=1, tp8 loop P7): request %llu, %zu chunks, %lld rows; the leader's "
             "rank-0 card: heads %.1f ms (%d; %.2f ms a 1k rows), tails %.1f ms (%d before the last; a tail waits on the card "
             "for its X2), the last tail %.1f ms (its X2 = the follower's last chunk + its cone part, then layers t.. + head + "
             "sampler + drafter), idle or other work between parts %.1f ms (before heads %.1f in %d, before tails %.1f in %d; "
             "the longest %.1f before chunk %zu's %s); host a chunk: plan %.2f + head release %.2f (prepare, restores, ship) + "
             "head issue %.2f (the ranks' issue) ms, tail issue %.2f, collect + commit + drafter finish %.2f ms; events "
             "dropped %lld, unread %lld", (unsigned long long)seq, n, (long long)rows, head, nhead,
             rows > 0 ? head * 1000.0 / (double)rows : 0.0, tail, ntail, (double)last, ga + gc, ga, nga, gc, ngc,
             gmax < 0 ? 0.0 : gmax, gmax_at, gmax_head ? "head" : "tail", P.plan / nh, P.rel / nh, P.issue / nh,
             P.tissue / nt, P.fin / nt, (long long)tl->dropped, (long long)tl->unread);
    /* every chunk, 16 a line: index rows@context: idle before | head ms, idle before | tail ms */
    for (size_t k0 = 0; k0 < n; k0 += 16) {
        std::string t;
        for (size_t k = k0; k < n && k < k0 + 16; ++k) {
            const PfTimeline::Chunk& c = P.ch[k];
            t += fmt("%s%zu:%lld@%lld %.1f|%.1f %.1f|%.1f", k > k0 ? "  " : "", k, (long long)c.rows, (long long)c.ctx,
                     (double)c.gap_a, (double)c.head, (double)c.gap_c, (double)c.tail);
        }
        RAD_INFO("prefill timeline request %llu chunks %zu-%zu (rows@context: idle before | head ms, idle before | tail ms; "
                 "-1 = not read): %s", (unsigned long long)seq, k0, std::min(n, k0 + 16) - 1, t.c_str());
    }
    tl->by_seq.erase(pit);
}

void Engine::lane_select(int l) {
    cur_sched_ = &sched_of(l);
    issue_sched_.store(cur_sched_, std::memory_order_release);
}

void Engine::pp_stamp(const char* what, int lane, int step, uint64_t t0, uint64_t t1) {
    if (!pp_stamps_) return;
    std::fprintf(pp_stamps_, "%s %d %d %llu %llu\n", what, lane, step, (unsigned long long)t0, (unsigned long long)t1);
    if ((++pp_stamp_n_ & 255) == 0) std::fflush(pp_stamps_);
}

int Engine::lanes_init(const std::vector<BatchBuilder::RankIO>& rio) {
    lane_sched_.clear();
    std::vector<Scheduler*> all{ &sched_ };
    for (int l = 1; l < n_lanes_; ++l) {
        auto s = std::make_unique<Scheduler>(sched_, l, n_lanes_, &lanes_);
        RAD_TRY(s->init(ranks_[0]->program, cfg_, kv_iface_.get(), pc_iface_.get(), rio, &geo_));
        s->set_eog(sched_.eog_ids());
        all.push_back(s.get());
        lane_sched_.push_back(std::move(s));
    }
    lanes_.init(all);
    /* tp8 M56: at W(h > 0, t) the leader's head part runs layers [0, h) and writes their state (lanes.h) */
    lanes_.set_head_state(cfg_.pp_cut_h > 0);
    /* tp8 S5 review: draft_ctxskip=1 refused by name under lanes (fn4), not accepted and left off: a run kit
     * that reads its switches back drops it instead of stopping on a value that differs */
    /* tp8 dev D4: under RADIANCE_PP_CTXSKIP_LANES=1 the ring is one region a lane (ctxskip_setup), so the skip is taken */
    if (!(ctxs_.ready && ctxs_.lanes > 1)) fn4::ctxskip_refuse_lanes();
    /* Every lane's batch is its own set of recordings (Ctx::pass_key reads its slab's pointers). */
    for (auto& rk : ranks_) rk->ctx.set_tape_lanes(n_lanes_);
    /* lane l holds cards [l * S / N, (l + 1) * S / N) (Scheduler::init) */
    const int64_t per = (cfg_.max_seqs + n_lanes_ - 1) / n_lanes_, least = cfg_.max_seqs / n_lanes_;
    RAD_INFO("two-node pipeline: %d lanes, each a scheduler with its own step slab on every card and at most %lld "
             "sequences (cards %lld-%lld a lane); the newcomer's chunks over the lanes %s (whole chunks of the "
             "one-process grid: the chunks an idle engine plans, one a lane in turn; RADIANCE_PP_SPLIT=0 off)%s", n_lanes_,
             (long long)per, (long long)least, (long long)per, lanes_.split_on() ? "on" : "off",
             cfg_.pp_cut_h > 0 ? "; a chunk that takes a snapshot is followed by its next chunk only after its commit "
                                 "(the head part writes layers [0, h)'s state)" : "");
    if (!lanes_.lend_busy())
        RAD_INFO("two-node pipeline: a newcomer's chunk is lent only to a lane with no request of its own "
                 "(RADIANCE_PP_LEND_BUSY=0): a decoding lane never carries another lane's chunk");
    if (lanes_.bg_lane() >= 0)
        RAD_INFO("two-node pipeline: lane(s) %d-%d are background prefill lanes (RADIANCE_PP_BG_LANE=%d, tp8 dev M8): they hold no "
                 "sequence (the other %d lane(s) split the %lld cards), and a newcomer planned whole beside decode rows is lent whole to "
                 "one, its first token there, so the home lane's step carries its decode rows alone; cold prompts' chunks keep to lanes "
                 "0-%d%s", lanes_.bg_lane(), n_lanes_ - 1, lanes_.bg_n(), n_lanes_ - lanes_.bg_n(), (long long)cfg_.max_seqs,
                 lanes_.bg_lane() - 1,
                 lanes_.lend_split() || lanes_.lend_mixed() ? " (before RADIANCE_PP_LEND_MIXED, which takes what it leaves)" : "");
    if (lanes_.bg_lane() >= 0)
        RAD_INFO("two-node pipeline: %s", lanes_.bg_draft_floor()
                     ? "a lent piece's commit opens its home request's drafter range as the home's own chunk would, so a newcomer "
                       "the BG lane ran whole drafts from its second token on (tp8 loop F22; RADIANCE_PP_BG_DRAFT_FLOOR=0 off)"
                     : "RADIANCE_PP_BG_DRAFT_FLOOR=0 -- a newcomer the BG lane ran whole drafts only once its home has run the "
                       "drafter's window of its tokens (lq46's behaviour, tp8 loop F22)");
    if (lanes_.bg_lane() >= 0 && lanes_.bg_split())
        RAD_INFO("two-node pipeline: RADIANCE_PP_BG_SPLIT=1 -- %s", lanes_.bg_n() >= 2
                     ? "a BG newcomer is cut at the 512-grid point nearest its middle (>= 128 rows a side) and its two pieces go to two "
                       "BG lanes in the planning order (tp8 dev M8)"
                     : "not taken: it needs RADIANCE_PP_BG_LANE=2 or more");
    if (lanes_.bg_lane() == 1)
        RAD_WARN("two-node pipeline: RADIANCE_PP_BG_LANE leaves ONE decode lane at --pp-lanes %d: decode steps no longer overlap on "
                 "the two nodes (use --pp-lanes %d)", n_lanes_, n_lanes_ + 1);
    if (const char* rf = std::getenv("RADIANCE_PP_READY_FIRST"); lanes_.bg_lane() >= 0 && !(rf && !std::strcmp(rf, "1")))
        RAD_WARN("two-node pipeline: RADIANCE_PP_BG_LANE without RADIANCE_PP_READY_FIRST=1: a decode lane's tail waits behind a BG "
                 "step's in the leader's order (tp8 dev M8)");
    if (lanes_.lend_split())
        RAD_INFO("two-node pipeline: a newcomer planned whole beside decode rows is cut at the 512-grid point nearest its middle and "
                 "its rest lent to one lane with an empty inbox and nothing waiting, its first token there (RADIANCE_PP_LEND_MIXED=2, "
                 "tp8 dev M5b)");
    if (lanes_.lend_mixed())
        RAD_INFO("two-node pipeline: a newcomer that lever BI's option D cuts beside decode rows lends the pieces one process "
                 "cuts next to the lanes that plan next, its first token in the last (RADIANCE_PP_LEND_MIXED=1, tp8 dev M5)%s",
                 fn4::sched_bi_chunk() == 3 ? "" : " -- inactive until fn4 sched_bi_chunk is 3");
    if (!lanes_.span_on())
        RAD_INFO("two-node pipeline: a lone prefill plans one checkpoint interval a chunk, not --prefill-span %d's span "
                 "(RADIANCE_PP_SPAN=0): the lanes run two of its chunks at once", cfg_.prefill_span);
    if (lanes_.span_on() && lanes_.span_short() > 0)
        RAD_INFO("two-node pipeline: a lone prefill of at most %lld rows plans one checkpoint interval a chunk "
                 "(RADIANCE_PP_SPAN_SHORT, tp8 loop P7); a longer one --prefill-span %d's span chunks%s",
                 (long long)lanes_.span_short(), cfg_.prefill_span, lanes_.span_tail() > 0 ? " (and RADIANCE_PP_SPAN_TAIL below)" : "");
    if (lanes_.span_on() && lanes_.span_tail() > 0)
        RAD_INFO("two-node pipeline: a lone prefill plans --prefill-span %d's span chunks, and one checkpoint interval a "
                 "chunk in its last %lld rows (RADIANCE_PP_SPAN_TAIL, tp8 loop P7): a shorter drain at the prompt's end",
                 cfg_.prefill_span, (long long)lanes_.span_tail());
    if (lanes_.span_inner())
        RAD_INFO("two-node pipeline: a chunk those cuts reach that carries a snapshot inside (lever TI's tail) is cut too, "
                 "before the snapshot, at checkpoint-interval boundaries (RADIANCE_PP_SPAN_INNER=1, tp8 loop P14)%s",
                 !lanes_.span_on() || lanes_.span_short() > 0 || lanes_.span_tail() > 0
                     ? "" : " -- inactive: RADIANCE_PP_SPAN=0, _SPAN_SHORT or _SPAN_TAIL names those cuts");
    return RAD_OK;
}

/* A lane step's first leader pass: its build shipped to the follower, then the leader's part up to
 * and including stage_send X1, with no sampler (kIssueModel). */
/* tp8 loop P7: the lane step's one prefill entry when it is a lone prefill's chunk (the timeline's steps), else null */
static const StepEntry* pf_entry(const RadBatch* b, const StepPlan& plan) {
    if (!b || b->phase != RAD_PHASE_PREFILL || plan.e.size() != 1 || !plan.e[0].is_prefill) return nullptr;
    return &plan.e[0];
}

int Engine::lane_head(int lane, const RadBatch* b) {
    lane_select(lane);
    /* tp8 loop P7 (RADIANCE_PP_PF_TIMELINE=1): this chunk's head, bracketed on rank 0's stream */
    const StepEntry* pfe = pf_tl_ ? pf_entry(b, lsched().plan()) : nullptr;
    const uint64_t pf0 = pfe ? rad_mono_ns() : 0;
    clear_fresh_states();
    if (const int ps = prepare_release(b); ps < 0) return ps;
    if (!pp_trunk_pending_[lane]) {
        RAD_ERR("two-node pipeline: lane %d's step %d was released without its build", lane, b->step);
        return RAD_E_STATE;
    }
    pp_trunk_pending_[lane] = false;
    /* tp8 S6: a prefix hit's snapshot restores, on this node's streams before the head (the follower runs them from
     * the frame before its pass); the tail reads the restored state on the same stream */
    RAD_TRY(restore_checkpoints());
    RAD_TRY(pp_ship(b, true, lane));
    pp_set_pass(lane, RAD_PP_PART_HEAD);
    issue_what_.store(kIssueModel, std::memory_order_release);
    step_batch_.store(b, std::memory_order_release);
    const uint64_t t0 = pp_stamps_ ? rad_mono_ns() : 0;
    PfTimeline::Prompt* pfp = nullptr;
    int pf_ci = -1;
    uint64_t pf1 = 0;
    if (pfe) {   /* the ranks wait at the barrier: rank 0's stream is the engine thread's to record on */
        if (pf_tl_->by_seq.size() > 64) pf_tl_->by_seq.erase(pf_tl_->by_seq.begin());   /* prompts never finished */
        pfp = &pf_tl_->by_seq[pfe->seq];
        pf_ci = (int)pfp->ch.size();
        PfTimeline::Chunk c;
        c.rows = pfe->n_tokens;
        c.ctx = pfe->ctx_len;
        pfp->ch.push_back(c);
        pf_tl_->by_step[b->step] = { pfe->seq, pf_ci };
        pfp->plan += pf_tl_->plan_ms;
        pf1 = rad_mono_ns();
        pfp->rel += (double)(pf1 - pf0) / 1e6;
        pf_tl_->mark(PfTimeline::kA, pfe->seq, pf_ci);
    }
    if (pp_ni_) pp_ni_->begin();      /* tp8 dev D8 */
    barrier_->arrive_and_wait();      /* release the ranks */
    barrier_->arrive_and_wait();      /* and wait for them */
    if (pp_ni_) pp_ni_->end();
    if (pfp) {
        pf_tl_->mark(PfTimeline::kB, pfe->seq, pf_ci);
        pfp->issue += (double)(rad_mono_ns() - pf1) / 1e6;
        ++pfp->nh;
    }
    /* busy from the release to X1's publish (the host backend notes it; a card leaves it 0) */
    if (pp_stamps_) {
        const RadPpRing* x1 = pp_send_[lane * RAD_PP_N_RINGS + RAD_PP_X1];
        pp_stamp("head", lane, b->step, t0, x1 && x1->ready_ns ? x1->ready_ns : rad_mono_ns());
    }
    pp_set_pass(lane, RAD_PP_PART_WHOLE);
    const int st = step_status_.load(std::memory_order_acquire);
    if (st < 0) RAD_ERR("lane %d step %d's head pass failed: %s -- the engine cannot continue", lane, b->step, rad_strerror(st));
    return st;
}

void Engine::lanes_note(int64_t lane_steps, int64_t rounds_two) {
    const LaneSet::Counts c = lanes_.counts();
    /* tp8 prefill P0: why chunks were not lent / sequences waited (LaneSet::kWhyN), summed over the run */
    std::string why;
    for (int k = 0; k < LaneSet::kWhyN; ++k)
        why += fmt("%s%s %lld", k ? ", " : "", LaneSet::why_name(k), (long long)c.why[k]);
    RAD_INFO("two-node pipeline: %lld lane steps, %lld of them with another lane's step in flight at their tail; "
             "newcomer split: %lld pieces lent (%lld rows, %lld carrying a first token), %lld committed; not lent / "
             "waited: %s%s", (long long)lane_steps, (long long)rounds_two, (long long)c.pieces, (long long)c.rows,
             (long long)c.token_pieces, (long long)c.done, why.c_str(),
             lanes_.lend_mixed() ? fmt("; %lld lends from newcomers option D cut beside decode rows (RADIANCE_PP_LEND_MIXED=1, tp8 "
                                       "dev M5)", (long long)c.dpieces).c_str()
             : lanes_.lend_split() ? fmt("; %lld newcomers split on the 512 grid beside decode rows (RADIANCE_PP_LEND_MIXED=2, tp8 "
                                         "dev M5b)", (long long)c.dpieces).c_str() : "");
    if (lanes_.bg_lane() >= 0)
        RAD_INFO("two-node pipeline: BG lane(s) from %d (RADIANCE_PP_BG_LANE, tp8 dev M8): %lld newcomers lent whole beside decode "
                 "rows (%lld rows; %lld of them in two pieces on the 512 grid, RADIANCE_PP_BG_SPLIT; %lld whole for want of two "
                 "lanes' room); kept home: a prefix restore %lld, the BG lanes' room %lld, not a planner %lld; drafter ranges opened "
                 "by a lent piece's commit %lld (%s, tp8 loop F22)", lanes_.bg_lane(),
                 (long long)c.bg, (long long)c.bg_rows, (long long)c.bg_split, (long long)c.bg_skip[LaneSet::kBgSplitRoom],
                 (long long)c.bg_skip[LaneSet::kBgRestore], (long long)c.bg_skip[LaneSet::kBgRoom],
                 (long long)c.bg_skip[LaneSet::kBgOrder], (long long)c.bg_floor,
                 lanes_.bg_draft_floor() ? "on" : "off: RADIANCE_PP_BG_DRAFT_FLOOR=0");
}

int Engine::step_loop_lanes() {
    if (rad_dev_set(ranks_[0]->device) < 0)
        RAD_ERR("step loop: could not bind rank 0's device");
    StepOut so;
    int idle = 0;
    std::vector<uint64_t> cancel;
    /* As step_loop for a pipeline node: no step issued ahead (see the head comment). */
    pipelining_ = false;
    early_ids_ = !cfg_.kld() && !embedding_;
    if (tree_) {
        RAD_ERR("speculation: drafter '%s' proposes trees, which are verified only on steps issued ahead; a "
                "two-node pipeline issues none", drafter_.name);
        return fail_loop(RAD_E_UNSUPPORTED);
    }
    if (ctxs_.ready && ctxs_.on && ctxs_.lanes <= 1) {
        RAD_WARN("two-node pipeline with %d lanes: the depth-0 context skip (draft_ctxskip) keeps consecutive "
                 "steps' rows in one ring; lanes interleave their steps, so it is off", n_lanes_);
        ctxs_.on = 0;
    }
    const bool t_on = (int)log_level() >= (int)Log::Debug;
    /* tp8 dev 1b: RADIANCE_PP_TAIL_AHEAD=1 (both nodes; the follower does nothing with it) -- the next lane's model part
     * goes out before this lane's collect, so the leader's cards run it while the host commits, plans and ships. Only
     * where the leader's head part issues no work the follower waits on (RADIANCE_PP_FOLLOWER_EMBED=1 at W(0, t)):
     * otherwise the next head would queue behind both tails and the follower would start it late. */
    tail_ahead_ = false;
    tail_ahead_n_ = 0;
    if (const char* ta = std::getenv("RADIANCE_PP_TAIL_AHEAD"); ta && !std::strcmp(ta, "1")) {
        const char* fe = std::getenv("RADIANCE_PP_FOLLOWER_EMBED");
        const bool fe_on = fe && !std::strcmp(fe, "1") && cfg_.pp_cut_h == 0;
        tail_ahead_ = fe_on && !embedding_;
        RAD_INFO("two-node pipeline: RADIANCE_PP_TAIL_AHEAD=1 %s", tail_ahead_
                     ? "-- a lane step's model part is issued while the lane before it is collected and committed; its "
                       "sampler and drafter after that commit (tp8 dev 1b)"
                     : "is not taken: it needs RADIANCE_PP_FOLLOWER_EMBED=1 at a cut W(0, t) (the leader's head part then "
                       "issues nothing the follower waits on)");
    }
    /* tp8 dev D7: RADIANCE_PP_DECODE_FIRST=1 (both nodes; the follower does nothing with it) -- in each planning round the
     * lanes with only decode rows to plan go before those with prompt rows (LaneSet::prefill_pending), round-robin kept
     * within each group: a decode lane's step then runs on the follower ahead of a newcomer's long MIXED pass and its tail on
     * the leader while the follower runs that pass, instead of queueing behind it on both nodes (PROFILE 20:58). Only the
     * order of the heads changes: step composition and step numbers, not a row's bytes; a lender still lends to the lanes
     * that plan right behind it. */
    bool decode_first = false;
    if (const char* df = std::getenv("RADIANCE_PP_DECODE_FIRST"); df && !std::strcmp(df, "1")) {
        decode_first = true;
        RAD_INFO("two-node pipeline: RADIANCE_PP_DECODE_FIRST=1 -- each round plans the lanes with decode rows alone before "
                 "those with prompt rows to plan (tp8 dev D7)");
    }
    int64_t reordered = 0, held = 0;
    /* tp8 dev M4: RADIANCE_PP_READY_FIRST=1 (both nodes; the follower does nothing with it; not with TAIL_AHEAD) -- when the
     * oldest lane step in flight is a long pass (>= RADIANCE_PP_SEG_ROWS rows, default 512) whose X2 has not landed, the
     * tail of a later lane step whose X2 has landed goes first (a decode step of its own sequences: its rows read only
     * committed KV, none the long pass writes); the host waits for the first X2 to land instead of queueing the long pass's
     * tail on the stream, where every other lane's tail would wait behind it. With the follower's segments (the follower
     * runs other lanes' passes between the long pass's segments) the other lanes' steps then complete during the long
     * pass. Only the order of the tails changes, never a step's composition. */
    bool ready_first = false;
    int64_t rf_rows = 512, rf_taken = 0, rf_waits = 0;
    if (const char* v = std::getenv("RADIANCE_PP_SEG_ROWS"); v && *v) rf_rows = std::max<int64_t>(1, std::atoll(v));
    if (const char* rf = std::getenv("RADIANCE_PP_READY_FIRST"); rf && !std::strcmp(rf, "1")) {
        ready_first = !tail_ahead_;
        RAD_INFO("two-node pipeline: RADIANCE_PP_READY_FIRST=1 %s", ready_first
                     ? fmt("-- a later lane step's tail goes before a pass of >= %lld rows whose X2 has not landed (tp8 dev "
                           "M4)", (long long)rf_rows).c_str()
                     : "is not taken with RADIANCE_PP_TAIL_AHEAD=1");
    }
    /* tp8 dev D10 (RADIANCE_PP_ISSUE_AHEAD=1, both nodes; not with TAIL_AHEAD, not with FOLLOWER_EMBED): an undrafted lane's
     * next DECODE step is planned and its head part released BEFORE this step's collect -- one machine's issue-ahead
     * (Scheduler::step_next: every row one position on, the card's ADV_TRUNK writing its token ids behind this step's
     * sampler), the frame shipped at once -- so the lane chain is the cards' (sampler -> the head part -> X1 -> the follower's
     * pass, which waits on its card for X1) and not the leader's host path (collect, commit, plan, build, ship) nor the
     * follower's start on the frame (PROFILE 20:58: ~2.2 ms of each node's ~2.7 ms idle a 19.0 ms lane step at c32). Only
     * where the follower reads no token id: the follower refuses the switch where its pass does (deepseek41's Engram layers,
     * its own embedding). Taken per tail only for a lane step of decode rows alone, none drafted, nothing lent to or from its
     * lane, its drafter proposing nothing, no KV-copy check due; else the step after it is planned after the commit, as before.
     * Only the order of issue changes, never a row's step: the step issued ahead is the one step() would plan there. */
    bool ia = false;
    int64_t ia_n = 0, ia_skip = 0;
    if (const char* v = std::getenv("RADIANCE_PP_ISSUE_AHEAD"); v && !std::strcmp(v, "1")) {
        const char* fe = std::getenv("RADIANCE_PP_FOLLOWER_EMBED");
        const bool fe_on = fe && !std::strcmp(fe, "1");
        /* tp8 dev D10 round 2: beside the follower's own embedding when the plugin's X1 carries the step's token ids (the
         * leader's card's, sent by the head part after ADV_TRUNK) and the follower takes them before its lookup */
        const bool ids = pp_x1_ids();
        ia = !tail_ahead_ && (!fe_on || ids) && !embedding_;
        RAD_INFO("two-node pipeline: RADIANCE_PP_ISSUE_AHEAD=1 %s%s", ia
                     ? "-- an undrafted decode lane's next step is planned, its head part released and its frame shipped before "
                       "this step's collect (tp8 dev D10)"
                     : fe_on && !ids ? "is not taken with RADIANCE_PP_FOLLOWER_EMBED=1 (the follower would embed token ids it "
                                       "does not have yet)"
                                     : "is not taken with RADIANCE_PP_TAIL_AHEAD=1",
                 ia && ids ? "; X1 carries the step's token ids, which the follower takes before it reads any (round 2)" : "");
    }
    auto ia_ok = [&](int l, const RadBatch* bb, bool kv_due_now) {
        if (!ia || kv_due_now || lanes_.is_bg(l) || lanes_.inbox_n(l) > 0 || bb->phase != RAD_PHASE_DECODE) return false;
        if (block_rows_ > 0 || (serial_drafting() && serial_rows_ > 0)) return false;
        for (const StepEntry& e : sched_of(l).plan().e)
            if (e.home >= 0 || e.is_prefill || e.n_spec > 0 || !e.produces_token) return false;
        return true;
    };
    auto x2_landed = [&](int l) {
        const RadPpRing* x2 = pp_recv_[l * RAD_PP_N_RINGS + RAD_PP_X2];
        return x2 && __atomic_load_n(&x2->landed, __ATOMIC_ACQUIRE) > __atomic_load_n(&x2->taken[0], __ATOMIC_ACQUIRE);
    };
    /* a lane step that may go before an older one: decode rows of its own sequences only (they read committed KV; a prompt's
     * chunk may read an earlier chunk's still in flight, a lent piece its home's), and no piece lent to its lane waiting in
     * its inbox (a home lends to the lanes in flight in their commit order, and each takes its piece at its next plan) */
    auto carries_lent = [&](int l) {
        if (sched_of(l).plan().phase != RAD_PHASE_DECODE || lanes_.inbox_n(l) > 0) return true;
        for (const StepEntry& e : sched_of(l).plan().e)
            if (e.home >= 0) return true;
        return false;
    };
    /* tp8 dev D7: a lane with prompt rows to plan waits ONE round while another lane's decode step is in flight (and nothing
     * was lent to it), so that lane's next step goes to the follower ahead of the long MIXED pass: the follower then runs it
     * before the MIXED pass and the leader its tail during it (one more decode step a newcomer than without). */
    std::vector<char> held_once((size_t)n_lanes_, 0);
    /* tp8 dev 1b: per lane, whether its step in flight has its model part / its sampler part released */
    std::vector<char> m_out((size_t)n_lanes_, 0), s_out((size_t)n_lanes_, 0);
    rad::StallWatch step_watch("step", 20000);
    /* tp8 dev D8: RADIANCE_PP_NODE_IDLE=1 (either node) */
    std::unique_ptr<PpNodeIdle> ni_own;
    struct NiReset { Engine* e; ~NiReset() { e->pp_ni_ = nullptr; } } ni_reset{ this };
    pp_node_idle_open(&ni_own);
    /* tp8 loop P7: RADIANCE_PP_PF_TIMELINE=1 (the leader; the follower ignores it) */
    std::unique_ptr<PfTimeline> pf_own;
    struct PfReset { Engine* e; ~PfReset() { e->pf_tl_ = nullptr; } } pf_reset{ this };
    if (const char* v = std::getenv("RADIANCE_PP_PF_TIMELINE"); v && !std::strcmp(v, "1")) {
        pf_own = std::make_unique<PfTimeline>();
        pf_own->s = ranks_[0]->ctx.stream();
        pf_tl_ = pf_own.get();
        RAD_INFO("two-node pipeline: RADIANCE_PP_PF_TIMELINE=1 -- a lone prefill's chunks are timed on rank 0's card "
                 "(card-local events around every head and tail) with the host's time a chunk: one line a prompt at its "
                 "first token (tp8 loop P7)%s", tail_ahead_ ? "; with RADIANCE_PP_TAIL_AHEAD=1 a tail whose model part "
                 "went ahead is not timed" : "");
    }
    /* The lanes with a head issued, oldest first; their tails go in this order (the order of their
     * X2 on the follower's stream). */
    std::deque<std::pair<int, const RadBatch*>> fifo;
    /* tp8 loop M2 (the PROFILE of 20:58: 320-430 ms with no lane step issued after every MIXED step): the time between two
     * lane-step heads while a step was in flight; a gap over 100 ms is a stall, named with the tail committed last before
     * it (its lane, phase and rows). One line a minute while lane steps run, and the run's at stop. */
    struct Gaps {
        struct Win {
            uint64_t t0 = 0; int64_t steps0 = 0, n = 0; double ms = 0, max_ms = 0; std::string worst;
            /* tp8 loop P6: the leader's host time in prefill / MIXED lane steps (a non-token chunk's collect waits for
             * nothing, so a host that blocks here is what keeps two chunks from overlapping): plan, head issue, tail */
            int64_t nh = 0, nt = 0; double plan_ms = 0, head_ms = 0, tail_ms = 0, tail_max = 0, head_max = 0;
        };
        Win win, run;
        uint64_t last = 0;
        int tl = -1, tphase = -1;
        int64_t ttok = 0;
        static void add(Win& w, double g, const std::string& why) {
            ++w.n;
            w.ms += g;
            if (g > w.max_ms) { w.max_ms = g; w.worst = why; }
        }
        void head(bool none_in_flight, int64_t steps) {
            const uint64_t now = rad_mono_ns();
            for (Win* w : { &win, &run }) if (!w->t0) { w->t0 = now; w->steps0 = steps; }
            if (last && !none_in_flight) {
                const double g = (double)(now - last) / 1e6;
                if (g > 100.0) {
                    const std::string why = fmt("%.0f ms after lane %d's %s step of %lld rows", g, tl,
                                                tphase == RAD_PHASE_PREFILL ? "prefill"
                                                : tphase == RAD_PHASE_DECODE ? "decode" : "MIXED", (long long)ttok);
                    add(win, g, why);
                    add(run, g, why);
                }
            }
            last = now;
        }
        void tail(int lane, const RadBatch* b) { tl = lane; tphase = b->phase; ttok = b->n_tok; }
        void host_head(const RadBatch* b, uint64_t p0, uint64_t p1, uint64_t h1) {
            if (b->phase == RAD_PHASE_DECODE) return;
            for (Win* w : { &win, &run }) {
                ++w->nh; w->plan_ms += (double)(p1 - p0) / 1e6; w->head_ms += (double)(h1 - p1) / 1e6;
                w->head_max = std::max(w->head_max, (double)(h1 - p1) / 1e6);
            }
        }
        void host_tail(const RadBatch* b, uint64_t t0, uint64_t t1) {
            if (b->phase == RAD_PHASE_DECODE) return;
            for (Win* w : { &win, &run }) {
                ++w->nt; w->tail_ms += (double)(t1 - t0) / 1e6;
                w->tail_max = std::max(w->tail_max, (double)(t1 - t0) / 1e6);
            }
        }
        static void note(const Win& w, const char* what, int64_t steps) {
            if (!w.t0) return;
            const double wall = (double)(rad_mono_ns() - w.t0) / 1e9;
            const std::string host = w.nh || w.nt
                ? fmt("; host in %lld prefill / MIXED lane step(s): plan %.1f + head issue %.1f ms a step (longest %.1f), "
                      "tail %.1f ms a step (longest %.1f)", (long long)std::max(w.nh, w.nt), w.nh ? w.plan_ms / w.nh : 0.0,
                      w.nh ? w.head_ms / w.nh : 0.0, w.head_max, w.nt ? w.tail_ms / w.nt : 0.0, w.tail_max)
                : std::string();
            RAD_INFO("two-node pipeline: lane-step stalls over %s (%.1f s, %lld lane steps): %lld head gaps > 100 ms with a "
                     "step in flight, %.0f ms in all (%.1f %% of the wall)%s%s%s", what, wall, (long long)(steps - w.steps0),
                     (long long)w.n, w.ms, wall > 0 ? w.ms / 10.0 / wall : 0.0, w.worst.empty() ? "" : "; the longest ",
                     w.worst.c_str(), host.c_str());
        }
        void minute(int64_t steps) {
            if (!win.t0 || rad_mono_ns() - win.t0 < 60ull * 1000000000ull) return;
            note(win, "the last minute", steps);
            win = Win{};
        }
        void end(int64_t steps) const { note(run, "the run", steps); }
    } gaps;
    std::vector<char> busy((size_t)n_lanes_, 0);
    int next = 0;   /* the lane planned first on the next round */
    int64_t lane_steps = 0, rounds_two = 0;
    int64_t kv_check_at = 0;   /* tp8 S6: trunk steps shipped at the last KV-copy check */
    while (!stopping_.load(std::memory_order_acquire)) {
        step_watch.progress();
        if (pp_link_ && !pp_link_->alive()) {
            RAD_ERR("two-node pipeline: the link to the follower is down (%s): failing every request",
                    pp_link_->why().c_str());
            return fail_loop(RAD_E_IO);
        }
        lora_epoch_.fetch_add(1, std::memory_order_acq_rel);
        phase_t_ = std::chrono::steady_clock::now();
        /* tp8 dev 1b: the oldest lane step's model part went ahead: its sampler and drafter now, before any other release
         * (a head's would change the arena level its sampler reads the logits at) */
        if (tail_ahead_ && !fifo.empty()) {
            const auto [fl, fb] = fifo.front();
            if (m_out[(size_t)fl] && !s_out[(size_t)fl]) {
                if (const int ts = lane_tail_issue(fl, fb, so, cancel, t_on, true); ts < 0) return fail_loop(ts);
                s_out[(size_t)fl] = 1;
            }
        }
        /* 1. every lane with no step in flight plans, in turn; each one's head goes out at once. The lanes
         * that plan before `l` plans again, in that order, are the idle ones after it -- planned right
         * behind it, before anything else -- then those in flight, oldest first (their commits come
         * before l's). A newcomer's chunk planned by `l` lends its pieces to them in that order
         * (lanes.h), so on both nodes the pieces run in position order. */
        /* tp8 S6: a KV-copy check due waits for every lane step in flight (the head comment), then goes out
         * before the next build */
        const bool kv_due = pp_kv_check_every_ > 0 && pp_steps_shipped_ - kv_check_at >= pp_kv_check_every_;
        if (kv_due && fifo.empty()) {
            pp_kv_check_send();
            kv_check_at = pp_steps_shipped_;
        }
        std::vector<int> todo;
        if (!(kv_due && !fifo.empty())) {
            for (int k = 0; k < n_lanes_; ++k) todo.push_back((next + k) % n_lanes_);
            if (decode_first) {
                const std::vector<int> was = todo;
                /* idle lanes with decode rows alone first (a busy lane is skipped below wherever it stands), so a change of
                 * order is a decode lane put ahead of a lane with prompt rows */
                std::stable_partition(todo.begin(), todo.end(), [&](int l) { return !busy[(size_t)l] && !lanes_.prefill_pending(l); });
                if (todo != was) ++reordered;
            }
        }
        for (size_t at = 0; at < todo.size(); ++at) {
            const int l = todo[at];
            if (busy[(size_t)l]) continue;
            if (decode_first && !held_once[(size_t)l] && lanes_.prefill_pending(l) && lanes_.inbox_n(l) == 0) {
                bool decode_in_flight = false;
                for (const auto& f : fifo) decode_in_flight = decode_in_flight || !lanes_.prefill_pending(f.first);
                if (decode_in_flight) {
                    held_once[(size_t)l] = 1;
                    ++held;
                    continue;
                }
            }
            std::vector<int> idle_after;
            {
                std::vector<int> order;
                for (int j = 1; j < n_lanes_; ++j) {
                    const int m = (l + j) % n_lanes_;
                    if (!busy[(size_t)m]) { order.push_back(m); idle_after.push_back(m); }
                }
                for (const auto& f : fifo) order.push_back(f.first);
                lanes_.set_planners(std::move(order));
            }
            lane_select(l);
            const uint64_t p0 = rad_mono_ns();
            const RadBatch* b = lsched().step();
            const uint64_t p1 = rad_mono_ns();
            if (pf_tl_) pf_tl_->plan_ms = (double)(p1 - p0) / 1e6;
            if (const std::string why = lanes_.broken(); !why.empty()) {
                RAD_ERR("two-node pipeline: %s -- the engine stops", why.c_str());
                return fail_loop(RAD_E_STATE);
            }
            if (!b) continue;
            if (t_on) phase_mark(PH_PLAN);
            if (const int hs = lane_head(l, b); hs < 0) return fail_loop(hs);
            gaps.host_head(b, p0, p1, rad_mono_ns());
            gaps.head(fifo.empty(), lane_steps);
            busy[(size_t)l] = 1;
            held_once[(size_t)l] = 0;
            fifo.emplace_back(l, b);
            next = (l + 1) % n_lanes_;
            /* the idle lanes it may have lent to plan next, in its order */
            todo.insert(todo.begin() + (ptrdiff_t)at + 1, idle_after.begin(), idle_after.end());
        }
        if (fifo.empty()) {
            step_batch_.store(nullptr, std::memory_order_release);
            drain_completed();
            sched_.publish(/*idle=*/true);
            for (auto& ls : lane_sched_) ls->publish(/*idle=*/true);
            bool deferred = sched_.has_deferred();
            for (auto& ls : lane_sched_) deferred = deferred || ls->has_deferred();
            if (deferred) {
                for (auto& rk : ranks_) (void)rad_stream_sync(rk->ctx.stream());
                sched_.release_deferred();
                for (auto& ls : lane_sched_) ls->release_deferred();
            }
            tier_tick();
            expire_tick(rad_mono_ns());
            loan_tick(/*idle=*/true);
            if (idle < 200) ++idle;
            std::this_thread::sleep_for(std::chrono::microseconds(idle * 5));
            continue;
        }
        idle = 0;
        if (fifo.size() >= 2) ++rounds_two;
        /* 2. the oldest lane step's tail, its drafter, its commit (tp8 dev M4 ready-first: or a later one's, above) */
        size_t pick = 0;
        if (ready_first && fifo.size() >= 2 && fifo.front().second->n_tok >= rf_rows && !x2_landed(fifo.front().first)) {
            bool waited = false;
            for (;;) {
                if (x2_landed(fifo.front().first)) break;
                for (size_t i = 1; i < fifo.size() && !pick; ++i)
                    if (x2_landed(fifo[i].first) && !carries_lent(fifo[i].first)) pick = i;
                if (pick || stopping_.load(std::memory_order_acquire) || (pp_link_ && !pp_link_->alive())) break;
                waited = true;
                std::this_thread::sleep_for(std::chrono::microseconds(20));
            }
            if (waited) ++rf_waits;
            if (pick) ++rf_taken;
        }
        const auto [lane, b] = fifo[pick];
        fifo.erase(fifo.begin() + (ptrdiff_t)pick);
        const uint64_t tt0 = rad_mono_ns();   /* tp8 loop P6: the tail's host time (after a ready-first wait) */
        const RadBatch* nb = nullptr;         /* tp8 dev D10: the lane's next step, issued ahead */
        if (!tail_ahead_ && ia) {
            if (const int ts = lane_tail_issue(lane, b, so, cancel, t_on, false); ts < 0) return fail_loop(ts);
            if (ia_ok(lane, b, kv_due)) {
                lane_select(lane);
                /* the card's acceptance of this step, which ADV_TRUNK continues from: a block or serial drafter ran it
                 * behind the sampler (issue_*_drafter); without a drafter only a pipelining engine does */
                if (!block_drafting() && !serial_drafting() && !pipelining_)
                    if (const int ac = advance_all(ADV_ACCEPT, 0, false); ac < 0) return fail_loop(ac);
                nb = lsched().step_next();
                if (nb) {
                    /* its rows' token ids, positions and slots on the card behind this step's sampler (and drafter) */
                    if (const int as = advance_all(ADV_TRUNK, 0, false); as < 0) return fail_loop(as);
                    if (const int hs = lane_head(lane, nb); hs < 0) return fail_loop(hs);
                }
            }
            if (!nb) ++ia_skip;
            if (const int fs = lane_tail_finish(lane, b, so, t_on); fs < 0) return fail_loop(fs);
            if (nb) {
                lane_select(lane);
                lsched().promote_next();
                gaps.head(false, lane_steps);
                fifo.emplace_back(lane, nb);
                ++ia_n;
            }
        } else if (!tail_ahead_) {
            if (const int ts = lane_tail(lane, b, so, cancel, t_on); ts < 0) return fail_loop(ts);
        } else {
            if (!s_out[(size_t)lane])
                if (const int ts = lane_tail_issue(lane, b, so, cancel, t_on, m_out[(size_t)lane] != 0); ts < 0)
                    return fail_loop(ts);
            /* tp8 dev 1b: the next lane step's model part, in head order, behind this one's drafter and step end; its
             * sampler goes out at the top of the next round, with nothing released between */
            if (!fifo.empty() && !m_out[(size_t)fifo.front().first]) {
                if (const int ms = lane_tail_model(fifo.front().first, fifo.front().second); ms < 0) return fail_loop(ms);
                m_out[(size_t)fifo.front().first] = 1;
            }
            if (const int fs = lane_tail_finish(lane, b, so, t_on); fs < 0) return fail_loop(fs);
            m_out[(size_t)lane] = s_out[(size_t)lane] = 0;
        }
        busy[(size_t)lane] = nb ? 1 : 0;
        gaps.host_tail(b, tt0, rad_mono_ns());
        gaps.tail(lane, b);
        if (++lane_steps == 1 || lane_steps % 4096 == 0) {
            lanes_note(lane_steps, rounds_two);
            if (ia)
                RAD_INFO("two-node pipeline: %lld lane steps issued ahead of the collect before them, %lld tails without "
                         "(RADIANCE_PP_ISSUE_AHEAD=1, tp8 dev D10)", (long long)ia_n, (long long)ia_skip);
        }
        gaps.minute(lane_steps);
        if (pp_ni_) pp_ni_->minute();
    }
    lanes_note(lane_steps, rounds_two);
    gaps.end(lane_steps);
    if (pp_ni_) pp_ni_->stop();
    if (tail_ahead_)
        RAD_INFO("two-node pipeline: %lld lane steps' model parts were issued ahead of the lane before them's collect "
                 "(RADIANCE_PP_TAIL_AHEAD=1)", (long long)tail_ahead_n_);
    if (decode_first)
        RAD_INFO("two-node pipeline: %lld planning rounds put a decode lane before a lane with prompt rows, %lld times a lane "
                 "with prompt rows waited a round behind a decode lane in flight (RADIANCE_PP_DECODE_FIRST=1)",
                 (long long)reordered, (long long)held);
    if (ready_first)
        RAD_INFO("two-node pipeline: %lld lane steps' tails went before an older long pass whose X2 had not landed, %lld "
                 "waits for a first X2 (RADIANCE_PP_READY_FIRST=1)", (long long)rf_taken, (long long)rf_waits);
    if (ia)
        RAD_INFO("two-node pipeline: %lld lane steps issued ahead of the collect before them, %lld tails without "
                 "(RADIANCE_PP_ISSUE_AHEAD=1, tp8 dev D10)", (long long)ia_n, (long long)ia_skip);
    return RAD_OK;
}

/* The tail of lane `lane`'s step `b`, whose head is out: Engine::step_loop's step from its sampler's
 * staging on, released as the leader's part from stage_recv X2 (kIssueAll), then its drafter, its
 * collect and commit -- with lsched() this lane's scheduler throughout. */
int Engine::lane_tail(int lane, const RadBatch* b, StepOut& so, std::vector<uint64_t>& cancel, bool t_on) {
    if (const int is = lane_tail_issue(lane, b, so, cancel, t_on, false); is < 0) return is;
    return lane_tail_finish(lane, b, so, t_on);
}

/* tp8 dev 1b: the tail's model part alone (kIssueModel: X2's receive, layers t.., the head's logits; no sampler), issued
 * ahead of its sampler as Engine::issue_ahead issues one machine's next step. Nothing is released between it and its
 * sampler part (step_loop_lanes): the arena level and every buffer the model part wrote are the sampler's. */
int Engine::lane_tail_model(int lane, const RadBatch* b) {
    lane_select(lane);
    if (const int ps = prepare_release(b, /*switches=*/false); ps < 0) return ps;
    pp_set_pass(lane, RAD_PP_PART_TAIL);
    issue_what_.store(kIssueModel, std::memory_order_release);
    step_batch_.store(b, std::memory_order_release);
    const uint64_t t0 = pp_stamps_ ? rad_mono_ns() : 0;
    if (pp_ni_) pp_ni_->begin();      /* tp8 dev D8 */
    barrier_->arrive_and_wait();      /* release the ranks */
    barrier_->arrive_and_wait();      /* and wait for them */
    if (pp_ni_) pp_ni_->end();
    if (pp_stamps_) {
        const RadPpRing* x2 = pp_recv_[lane * RAD_PP_N_RINGS + RAD_PP_X2];
        const uint64_t t1 = rad_mono_ns();
        pp_stamp("tail", lane, b->step, x2 && x2->taken_ns > t0 ? x2->taken_ns : t0, t1);
    }
    pp_set_pass(lane, RAD_PP_PART_WHOLE);
    ++tail_ahead_n_;
    const int st = step_status_.load(std::memory_order_acquire);
    if (st < 0) RAD_ERR("lane %d step %d's model part, issued ahead, failed: %s -- the engine cannot continue", lane, b->step,
                        rad_strerror(st));
    return st;
}

/* The tail's release (its model part too unless `model_done`), its drafter's passes and its step end. */
int Engine::lane_tail_issue(int lane, const RadBatch* b, StepOut& so, std::vector<uint64_t>& cancel, bool t_on,
                            bool model_done) {
    lane_select(lane);
    const StepPlan& plan = lsched().plan();
    /* tp8 loop P7: this chunk's tail on the timeline (the default path: its model part not issued ahead) */
    PfTimeline::Prompt* pfp = nullptr;
    uint64_t pf_seq = 0, pft0 = 0;
    int pf_ci = -1;
    if (pf_tl_ && !model_done) {
        auto it = pf_tl_->by_step.find(b->step);
        if (it != pf_tl_->by_step.end()) {
            auto pit = pf_tl_->by_seq.find(it->second.first);
            if (pit != pf_tl_->by_seq.end()) {
                pfp = &pit->second;
                pf_seq = it->second.first;
                pf_ci = it->second.second;
                pft0 = rad_mono_ns();
            }
        }
    }
    so.reqs.clear();
    so.first_row.assign(plan.e.size(), -1);
    int32_t logit_row = 0;
    for (size_t i = 0; i < plan.e.size(); ++i) {
        const StepEntry& e = plan.e[i];
        if (!e.produces_token) continue;
        so.first_row[i] = logit_row;
        logit_row += 1 + e.n_spec;
        so.reqs.push_back(const_cast<Request*>(e.req));
    }
    so.n_rows = logit_row;
    if (const int ss = stage_scored_rows(plan, so, cancel); ss < 0) return ss;

    /* As step_loop's stage_sampler. */
    auto stage_sampler = [&]() -> int {
        if (so.reqs.empty() || embedding_) return RAD_OK;
        for (Request* rq : so.reqs) {
            if (!rq) continue;
            bool failed = false;
            for (auto& rk : ranks_) {
                if (rk->sampler.has_request(rq->id)) continue;
                std::string err;
                if (rk->sampler.begin_request(rq->id, rq->sp, &err) < 0) {
                    RAD_ERR("sampler: request %llu could not be started: %s; cancelling it",
                            (unsigned long long)rq->id, err.c_str());
                    failed = true;
                    break;
                }
            }
            if (failed) {
                for (auto& rk : ranks_) rk->sampler.end_request(rq->id);
                cancel.push_back(rq->id);
                continue;
            }
            rq->speculation_fixed = ranks_[0]->sampler.speculation_fixed(rq->id);
        }
        if (t_on) phase_mark(PH_BEGIN);
        if (mtp_couple_)
            for (const StepEntry& e : plan.e) {
                if (!e.produces_token || !e.ngram || e.n_spec <= 0) continue;
                ++couple_n_lookup_;
                for (auto& rk : ranks_) {
                    const int cs = rk->sampler.set_mtp_couple(e.seq, nullptr, 0);
                    if (cs < 0 && cs != RAD_E_NOTFOUND) return cs;
                }
            }
        for (auto& rk : ranks_) {
            const int s = rk->sampler.build_step(so.reqs, b->step);
            if (s < 0) { RAD_ERR("sampler build_step: %s", rad_strerror(s)); return s; }
            for (uint64_t id : rk->sampler.take_failed()) cancel.push_back(id);
        }
        if (t_on) phase_mark(PH_BUILD);
        return RAD_OK;
    };

    static const bool log_steps = std::getenv("RADIANCE_LOG_STEPS") != nullptr;
    if (log_steps) {
        std::string s = fmt("lane %d step %d %s n_seq=%lld n_tok=%lld", lane, b->step,
                            b->phase == RAD_PHASE_PREFILL ? "prefill" : b->phase == RAD_PHASE_DECODE ? "decode" : "MIXED",
                            (long long)b->n_seq, (long long)b->n_tok);
        for (int i = 0; i < b->n_seq && i < 8; ++i)
            s += fmt("  [%d] slot=%d q=%d ctx=%d%s", i, plan.e[(size_t)i].slot, plan.e[(size_t)i].n_tokens,
                     plan.e[(size_t)i].ctx_len, plan.e[(size_t)i].home >= 0 ? fmt(" (lane %d's)", plan.e[(size_t)i].home).c_str() : "");
        RAD_INFO("%s", s.c_str());
    }
    if (t_on) phase_mark(PH_PRE);

    {
        std::lock_guard<std::mutex> lk(build_mu_);
        build_ready_  = false;
        build_status_ = RAD_OK;
    }
    /* The tail's release: the arena level and the follower pools for its rows, no switch change. tp8 dev 1b: its model
     * part went ahead (lane_tail_model), so this release issues the sampler alone, at the level that part set. */
    if (!model_done) {
        if (const int ps = prepare_release(b, /*switches=*/false); ps < 0) return ps;
        pp_set_pass(lane, RAD_PP_PART_TAIL);
        issue_what_.store(embedding_ ? kIssueModel : kIssueAll, std::memory_order_release);
    } else {
        issue_what_.store(kIssueSampler, std::memory_order_release);
    }
    step_batch_.store(b, std::memory_order_release);
    const uint64_t t0 = pp_stamps_ ? rad_mono_ns() : 0;
    if (pfp) pf_tl_->mark(PfTimeline::kC, pf_seq, pf_ci);
    if (pp_ni_) pp_ni_->begin();      /* tp8 dev D8: the tail, its drafter and its step end */
    barrier_->arrive_and_wait();      /* release the ranks */
    int bst = RAD_OK;
    {
        bst = stage_sampler();
        {
            std::lock_guard<std::mutex> lk(build_mu_);
            build_ready_  = true;
            build_status_ = bst;
        }
        build_cv_.notify_all();
    }
    barrier_->arrive_and_wait();      /* wait for them */
    /* busy from X2's arrival (rank 0 took it) to the join; the wait for X2 before it is not */
    if (pp_stamps_ && !model_done) {
        const RadPpRing* x2 = pp_recv_[lane * RAD_PP_N_RINGS + RAD_PP_X2];
        const uint64_t t1 = rad_mono_ns();
        pp_stamp("tail", lane, b->step, x2 && x2->taken_ns > t0 ? x2->taken_ns : t0, t1);
    }
    /* The drafter's passes behind the tail cross nothing: a whole pass of the lane. */
    pp_set_pass(lane, RAD_PP_PART_WHOLE);
    if (t_on) phase_mark(PH_RANKS);
    /* tp8 S5 review: a row the sampler refused may be another lane's (a newcomer's last piece, which
     * samples its first token here): the cancel goes to the lane holding the request, or the request
     * would commit a token its sampler never staged. */
    for (uint64_t id : cancel) {
        Scheduler* o = lanes_.owner(id);
        (o ? *o : lsched()).cancel(id);
    }
    cancel.clear();
    if (bst < 0) return bst;
    int st = step_status_.load(std::memory_order_acquire);
    if (st < 0) {
        RAD_ERR("lane %d step %d failed: %s -- the engine cannot continue", lane, b->step, rad_strerror(st));
        return st;
    }
    if (!(block_drafting() && drafter_snap_)) RAD_TRY(save_checkpoints());
    if (t_on) phase_mark(PH_SAVE);
    st = block_drafting() ? issue_block_drafter(b->step, so) : issue_serial_drafter(b->step, so);
    if (st < 0) return st;
    if (pfp) {
        pf_tl_->mark(PfTimeline::kD, pf_seq, pf_ci);
        pfp->tissue += (double)(rad_mono_ns() - pft0) / 1e6;
        ++pfp->nt;
    }
    RAD_TRY(stream_probe_issue(b));
    RAD_TRY(mark_step_end(b, so));
    /* tp8 dev 1b: the scored rows' copies here, before another lane's model part can follow on the stream (a no-op
     * outside --kld-record; with nothing between, the copies sit where collect's score_rows put them) */
    if (cfg_.kld()) RAD_TRY(score_rows_issue(b, so));
    if (pp_ni_) pp_ni_->end();
    return RAD_OK;
}

/* The tail's collect, commit and drafter finish (lane_tail_issue ran; tp8 dev 1b: another lane's model part may have
 * been released since, so this lane is selected again). */
int Engine::lane_tail_finish(int lane, const RadBatch* b, StepOut& so, bool t_on) {
    lane_select(lane);
    const StepPlan& plan = lsched().plan();
    /* tp8 loop P7: whether this is a timeline chunk, and the one with its prompt's first token */
    const uint64_t pff0 = pf_tl_ ? rad_mono_ns() : 0;
    const bool pf_tok = pf_tl_ && pf_entry(b, plan) && plan.e[0].produces_token;
    int st = embedding_ ? collect_pooled(b, so) : collect_step(b, so);
    if (t_on) phase_mark(PH_COLLECT);
    if (st < 0) return st;
    if (pp_link_)
        for (int k = 0; k < n_lanes_ * RAD_PP_N_RINGS; ++k)
            for (RadPpRing* q : { pp_send_[k], pp_recv_[k] })
                if (q && __atomic_load_n(&q->abort, __ATOMIC_ACQUIRE)) {
                    RAD_ERR("two-node pipeline: lane %d step %d: ring %d was aborted (%s): the step's tokens are not used",
                            lane, b->step, k, pp_link_->alive() ? "a stage op gave up" : pp_link_->why().c_str());
                    return RAD_E_IO;
                }
    couple_verify_evidence(b->step);
    if (cfg_.kld()) RAD_TRY(score_rows_take(b, so));
    dump_buffers(ranks_[0]->ctx, b->step);
    static const int prof_every = [] {
        const char* v = std::getenv("RADIANCE_PROFILE_EVERY");
        const int n = (v && *v) ? (int)std::strtol(v, nullptr, 10) : 32;
        return n > 0 ? n : 32;
    }();
    if (cfg_.profile_ops && b->step > 0 && (b->step % prof_every) == 0)
        for (auto& rk : ranks_) rk->ctx.dump_profile();
    RAD_TRACE("lane %d step %d: %lld tok / %lld seq, %lld logit row(s) -> tokens %s", lane, b->step, (long long)b->n_tok,
              (long long)b->n_seq, (long long)b->n_out,
              so.token.empty() ? "(none)" : fmt("%d%s", so.token[0], so.token.size() > 1 ? " ..." : "").c_str());

    Scheduler::StepResult res{};
    res.n_seq      = (int)plan.e.size();
    res.n_accepted = so.accepted.data();
    res.token      = so.token.data();
    res.eos        = so.eos.data();
    if (!so.lp.empty()) {
        res.lp        = so.lp.data();
        res.first_row = so.first_row.data();
    }
    res.path = so.path.empty() ? nullptr : so.path.data();
    st = lsched().commit(res);
    if (t_on) phase_mark(PH_COMMIT);
    if (st < 0) { RAD_ERR("commit: %s", rad_strerror(st)); return st; }
    dconf_commit(plan, so, b->step);
    if (so.n_rows > 0) RAD_TRY(stream_probe_take(b, res.n_prefill_done > 0, res.n_finished > 0));
    if (so.early) {
        st = rad_event_sync(step_end_);
        if (st < 0) { RAD_ERR("step %d: waiting for its draft passes: %s", b->step, rad_strerror(st)); return st; }
        so.early = false;
    }
    st = block_drafting() ? finish_block_drafter(so) : finish_serial_drafter(so);
    if (st < 0) return st;
    lsched().step_finished();
    tier_tick();
    loan_tick(/*idle=*/false);
    const uint64_t now = rad_mono_ns();
    arena_tick(now);
    expire_tick(now);
    drain_completed();
    lsched().publish(/*idle=*/false);
    if (pf_tl_) pf_finish(pf_tl_, b, pff0, pf_tok);
    if (t_on) {
        phase_mark(PH_DRAFT);
        if (++phase_n_ == 128) {
            double tot = 0;
            for (int i = 0; i < PH_N; ++i) tot += phase_us_[i];
            RAD_DEBUG("lane step %s: %.2f ms = plan %.2f begin %.2f build %.2f pre %.2f ranks %.2f save %.2f collect %.2f "
                      "commit %.2f draft %.2f (mean/128)", b->phase == RAD_PHASE_DECODE ? "decode" : "other",
                      tot / 128000.0, phase_us_[PH_PLAN] / 128000.0, phase_us_[PH_BEGIN] / 128000.0,
                      phase_us_[PH_BUILD] / 128000.0, phase_us_[PH_PRE] / 128000.0, phase_us_[PH_RANKS] / 128000.0,
                      phase_us_[PH_SAVE] / 128000.0, phase_us_[PH_COLLECT] / 128000.0, phase_us_[PH_COMMIT] / 128000.0,
                      phase_us_[PH_DRAFT] / 128000.0);
            for (int i = 0; i < PH_N; ++i) phase_us_[i] = 0;
            phase_n_ = 0;
        }
    }
    return RAD_OK;
}

}  /* namespace rad */
