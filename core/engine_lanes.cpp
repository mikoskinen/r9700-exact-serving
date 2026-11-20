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
 *     differ by the step whose head ran and whose tail has not); with RADIANCE_PP_KV_CHECK_DEFER=1 (tp8 loop P37) a
 *     check due while prompt rows are in flight or waiting waits for a round without them (at most N more trunk
 *     steps), so it does not drain a cold prompt's pipeline -- it still goes out with no lane step in flight;
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
#include "sched/geometry.h"  /* tp8 loop P23: prefill_shared_tok */
#include "sched/tcopy.h"     /* tp8 loop D46 */

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
int pp_band_k();    /* engine_node.cpp: the band the plugin declared (tp8 loop P28) */
int pp_tail_band_k();   /* engine_node.cpp: the tail band the plugin declared (tp8 loop D39) */
int pp_tail_copy();     /* engine_node.cpp: the plugin's tail band copy form, 0 none (D46, core/sched/tcopy.h) */
std::vector<int32_t> pp_segments();   /* engine_node.cpp: the boundaries the plugin declared on this node (tp8 dev M4 / M9) */

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
    struct Chunk { int64_t rows = 0, ctx = 0; float gap_a = -1, head = -1, gap_c = -1, tail = -1, issue = -1; };   /* issue: tp8 loop P31 */
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
    double imax = -1;   /* tp8 loop P31: the longest head issue and where */
    size_t imax_at = 0;
    for (size_t k = 0; k < n; ++k) {
        const PfTimeline::Chunk& c = P.ch[k];
        rows += c.rows;
        if (c.issue > imax) { imax = c.issue; imax_at = k; }
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
             "head issue %.2f (the ranks' issue; the first chunk's %.2f, the longest %.2f at chunk %zu) ms, tail issue %.2f, collect + "
             "commit + drafter finish %.2f ms; events dropped %lld, unread %lld", (unsigned long long)seq, n, (long long)rows, head,
             nhead, rows > 0 ? head * 1000.0 / (double)rows : 0.0, tail, ntail, (double)last, ga + gc, ga, nga, gc, ngc,
             gmax < 0 ? 0.0 : gmax, gmax_at, gmax_head ? "head" : "tail", P.plan / nh, P.rel / nh, P.issue / nh,
             n ? (double)P.ch[0].issue : -1.0, imax, imax_at, P.tissue / nt, P.fin / nt, (long long)tl->dropped,
             (long long)tl->unread);
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
    if (const int64_t bgr = LaneSet::bg_rows_env(); bgr > 0) {   /* tp8 loop D47 */
        if (lanes_.bg_lane() >= 0 && lanes_.bg_rows(cfg_.max_seqs) < cfg_.max_seqs)
            RAD_INFO("two-node pipeline: RADIANCE_PP_BG_ROWS=%lld (tp8 loop D47) -- the BG lanes' step slabs carve their block tables "
                     "for %lld rows, not --max-num-seqs %lld, and a BG lane takes a lent newcomer only while its inbox holds fewer "
                     "than %lld (else the next BG lane, or its home as a MIXED step)", (long long)bgr, (long long)bgr,
                     (long long)cfg_.max_seqs, (long long)bgr);
        else
            RAD_INFO("two-node pipeline: RADIANCE_PP_BG_ROWS=%lld is not taken (tp8 loop D47): %s", (long long)bgr,
                     lanes_.bg_lane() < 0 ? "it needs RADIANCE_PP_BG_LANE" : "not below --max-num-seqs");
    }
    if (const int64_t lm = LaneSet::lane_rows_env(); lm > 0)   /* tp8 loop D47b */
        RAD_INFO("two-node pipeline: RADIANCE_PP_LANE_ROWS=%lld (tp8 loop D47b) -- a decode lane's step slab (lanes 1 ..; lane 0, the MERGE "
                 "target, keeps --max-num-seqs) carves its block tables for its own cards + %lld rows (%lld at --max-num-seqs %lld over %d "
                 "decode lanes), and takes a lent piece only while its inbox holds fewer than %lld", (long long)lm, (long long)lm,
                 (long long)LaneSet::lane_rows(cfg_.max_seqs / std::max(1, n_lanes_ - lanes_.bg_n()), lm, cfg_.max_seqs),
                 (long long)cfg_.max_seqs, n_lanes_ - lanes_.bg_n(), (long long)lm);
    if (lanes_.bg_lane() >= 0)
        RAD_INFO("two-node pipeline: a lent piece's commit opens its home request's drafter range as the home's own chunk would, so "
                 "a newcomer the BG lane ran whole drafts from its second token on (tp8 loop F22)");
    if (lanes_.bg_lane() >= 0 && lanes_.bg_split())
        RAD_INFO("two-node pipeline: RADIANCE_PP_BG_SPLIT=1 -- %s", lanes_.bg_n() >= 2
                     ? "a BG newcomer is cut at the 512-grid point nearest its middle (>= 128 rows a side) and its two pieces go to two "
                       "BG lanes in the planning order (tp8 dev M8)"
                     : "not taken: it needs RADIANCE_PP_BG_LANE=2 or more");
    if (lanes_.bg_hold_ms() > 0)
        RAD_INFO("two-node pipeline: RADIANCE_PP_BG_HOLD_MS=%lld -- a newcomer the BG lanes have no room for is left out of its "
                 "home's step (the step goes out with its decode rows alone) and planned again at the home's next step, for a BG "
                 "lane to take whole; after %lld ms of holding it runs at home as today's MIXED step (tp8 dev D11)",
                 (long long)lanes_.bg_hold_ms(), (long long)lanes_.bg_hold_ms());
    if (const char* bp = std::getenv("RADIANCE_PP_BG_PIECES"); bp && *bp)
        RAD_WARN("two-node pipeline: RADIANCE_PP_BG_PIECES=%s is ignored: the switch was removed 18.11. (tp8 dev D12, never on cards)", bp);
    /* tp8 loop P23s: the in-process switches pp_bg_room / pp_bg_join_ms start at the env's values (0 without a BG lane) */
    fn4::note_sched(fn4::kPpBgRoom, (int)lanes_.bg_room());
    fn4::note_sched(fn4::kPpBgJoinMs, (int)lanes_.bg_join_ms());
    /* tp8 MERGE: fn4 pp_merge starts at RADIANCE_PP_MERGE (0 unset), taken with two decode lanes or more */
    {
        int v = 0;
        if (const char* pm = std::getenv("RADIANCE_PP_MERGE"); pm && *pm) v = std::max(0, std::min(127, std::atoi(pm)));
        if (lanes_.decode_n() < 2) v = 0;
        pp_merge_.store(v, std::memory_order_relaxed);
        fn4::note_sched(fn4::kPpMerge, v);
        if (v)
            RAD_INFO("two-node pipeline: RADIANCE_PP_MERGE=%d (fn4 pp_merge, tp8 MERGE) -- %s%s", v,
                     (v & 2) ? "the decode lanes run as ONE lane (lane 0) always"
                             : (v & 1) ? "the decode lanes run as ONE lane (lane 0) while a BG lane has a step in flight, split back when none"
                                       : "no merge",
                     (v & 8) ? "; no BG step's leader parts in segments (RADIANCE_PP_LEAD_SEG declared or not)"
                     : (v & 4) ? "; a BG step's leader parts in segments only with the merge on (with RADIANCE_PP_LEAD_SEG)" : "");
        int ms = 0;
        if (const char* m = std::getenv("RADIANCE_PP_MERGE_MS"); m && *m) ms = std::max(0, std::min(60000, std::atoi(m)));
        pp_merge_ms_.store(ms, std::memory_order_relaxed);
        fn4::note_sched(fn4::kPpMergeMs, ms);
        if (v && ((v & 16) || ms))
            RAD_INFO("two-node pipeline: tp8 MERGE -- %s%s", (v & 16) ? "while merged a newcomer the BG lanes have no room for is held for "
                     "them (RADIANCE_PP_MERGE_HOLD_MS, D11's hold)" : "", ms ? fmt("; the lanes stay merged %d ms after the last BG work "
                     "(RADIANCE_PP_MERGE_MS, fn4 pp_merge_ms)", ms).c_str() : "");
    }
    if (lanes_.bg_room() > 0)
        RAD_INFO("two-node pipeline: RADIANCE_PP_BG_ROOM=%lld -- a BG lane's inbox (its next step) takes lent newcomers up to %lld "
                 "rows (the shared budget %lld otherwise), so the newcomers that arrive while a BG step runs go to the next one "
                 "together, each its own BI segment; fn4 ds41_bi 7 (lever PMM) runs their routed MoE as one pass a layer (tp8 loop "
                 "P23)", (long long)lanes_.bg_room(), (long long)std::min<int64_t>(lanes_.bg_room(), cfg_.max_tok),
                 (long long)std::min<int64_t>(cfg_.max_tok, prefill_shared_tok(cfg_)));
    else if (const char* br = std::getenv("RADIANCE_PP_BG_ROOM"); br && *br)
        RAD_WARN("two-node pipeline: RADIANCE_PP_BG_ROOM=%s is not taken: %s (tp8 loop P23)", br,
                 lanes_.bg_lane() < 0 ? "it needs RADIANCE_PP_BG_LANE" : "not 256 .. 65536 rows");
    if (lanes_.bg_join_ms() > 0)
        RAD_INFO("two-node pipeline: RADIANCE_PP_BG_JOIN_MS=%lld -- a newcomer past the shared budget's room joins a busy BG lane's "
                 "inbox only while that lane's step in flight is predicted to commit within %lld ms (its rows x the BG steps' measured "
                 "us a row); else it runs at home as a MIXED step, as without RADIANCE_PP_BG_ROOM (tp8 loop P23b)",
                 (long long)lanes_.bg_join_ms(), (long long)lanes_.bg_join_ms());
    else if (const char* bj = std::getenv("RADIANCE_PP_BG_JOIN_MS"); bj && *bj)
        RAD_WARN("two-node pipeline: RADIANCE_PP_BG_JOIN_MS=%s is not taken: %s (tp8 loop P23b)", bj,
                 lanes_.bg_room() <= 0 ? "it needs RADIANCE_PP_BG_ROOM (and RADIANCE_PP_BG_LANE)" : "not 1 .. 60000 ms");
    if (const char* bh = std::getenv("RADIANCE_PP_BG_HOLD_MS"); lanes_.bg_hold_ms() <= 0 && bh && *bh)
        RAD_WARN("two-node pipeline: RADIANCE_PP_BG_HOLD_MS=%s is not taken: %s (tp8 dev D11)", bh,
                 lanes_.bg_lane() < 0 ? "it needs RADIANCE_PP_BG_LANE" : "not 1 .. 60000 ms");
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
    if (lanes_.span_first() > 0)
        RAD_INFO("two-node pipeline: a lone cold prompt planned one checkpoint interval a chunk, of more than two intervals, runs "
                 "its first interval as %lld + the rest rows (RADIANCE_PP_SPAN_FIRST, tp8 loop P34): the follower's first pass "
                 "waits for a %lld-row head%s", (long long)lanes_.span_first(), (long long)lanes_.span_first(),
                 !lanes_.span_on() || lanes_.span_short() > 0 || lanes_.span_tail() > 0
                     ? "" : " -- inactive: RADIANCE_PP_SPAN=0, _SPAN_SHORT or _SPAN_TAIL names those prompts");
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

/* tp8 loop P28: RADIANCE_PP_BAND_MIN_SEQS=N (default 0: always) -- band passes only while the decode lanes' last steps decoded
 * >= N sequences together: the c32 direction moves newcomer work to the leader, which is the bound node at low load (c16: its
 * drafter), so a load-chosen band leaves those cells W(h, t)'s. Read with RADIANCE_PP_BAND; counts at the run's end. */
static int64_t g_band_min_seqs = 0, g_band_low = 0;
/* tp8 loop P28: a lane step's band k (0: not a band pass) -- a BG lane's step whose prompts are all fresh (no context, no
 * snapshot restore: the leader holds no band layer's prefix) */
int Engine::pp_band_step(int lane, const RadBatch* b) {
    if (!pp_band_on_ || !b || !lanes_.is_bg(lane) || b->n_tok > pp_band_max_rows_) return 0;
    const int k = pp_band_k();
    if (k <= 0) return 0;
    if (g_band_min_seqs > 0) {
        int64_t dec = 0;
        for (int l = 0; l < n_lanes_; ++l)
            if (!lanes_.is_bg(l)) dec += sched_of(l).decoded_seqs();
        if (dec < g_band_min_seqs) { ++g_band_low; return 0; }
    }
    const StepPlan& p = sched_of(lane).plan();
    if (p.e.empty()) return 0;
    for (const StepEntry& e : p.e)
        if (!e.is_prefill || e.ctx_len != 0 || e.ckpt_restore >= 0 || e.n_spec > 0) return 0;
    if (pp_band_off_) {   /* tp8 loop D36: fn4 pp_lead_seg + 256 -- this band step runs W(h, t)'s */
        ++pp_band_off_n_;
        return 0;
    }
    return k;
}

/* tp8 loop D35: a decode step shipped -- the gap since its lane's last decode collect (D10's issue-ahead ships before that
 * collect: those steps find no collect time and are not counted) */
void Engine::dch_note_head(int lane, const RadBatch* b) {
    if (!pp_link_ || lane < 0 || lane >= RAD_PP_MAX_LANES || !dch_collect_ns_[lane]) return;
    if (b->phase != RAD_PHASE_DECODE) { dch_collect_ns_[lane] = 0; return; }   /* a prompt step between: not a decode chain */
    const double g = (double)(rad_mono_ns() - dch_collect_ns_[lane]) / 1e6;
    dch_collect_ns_[lane] = 0;
    ++dch_n_;
    dch_ms_ += g;
    dch_max_ms_ = std::max(dch_max_ms_, g);
    ++dch_hist_[std::min<size_t>(511, (size_t)(g * 10.0))];
}
std::string Engine::dch_text() const {
    double p50 = 0;
    int64_t acc = 0;
    for (size_t i = 0; i < 512; ++i) {
        acc += dch_hist_[i];
        if (dch_n_ > 0 && 2 * acc >= dch_n_) { p50 = (double)i / 10.0 + 0.05; break; }
    }
    return fmt("%lld decode steps of the lanes: their collect -> the lane's next decode step shipped p50 %.2f / mean %.2f ms (max %.1f); the "
               "collect's wait for the card mean %.2f ms over %lld collects", (long long)dch_n_, p50, dch_n_ ? dch_ms_ / (double)dch_n_ : 0.0,
               dch_max_ms_, dch_wait_n_ ? dch_wait_ms_ / (double)dch_wait_n_ : 0.0, (long long)dch_wait_n_);
}

/* tp8 loop D39: a lane step's tail band j (0: not a tail band pass) -- P28's rule (a BG lane's step whose prompts are all fresh,
 * no context, no snapshot restore, no draft rows, <= RADIANCE_PP_BAND_MAX_ROWS rows) and no head band in the same step */
int Engine::pp_tband_step(int lane, const RadBatch* b, bool count) {
    if (!pp_tband_on_ || !b || b->n_tok > pp_band_max_rows_) return 0;
    const int j = pp_tail_band_k();
    if (j <= 0) return 0;
    const StepPlan& p = sched_of(lane).plan();
    if (p.e.empty()) return 0;
    /* fn4 pp_tband (-1 never requested = the start's RADIANCE_PP_TAIL_BAND value): 0 none, 1 a BG lane's step of fresh prompts, 2
     * (tp8 loop D39b) + a step of prompt rows only on ANY lane (a lone cold prompt plans on its home decode lane: lane_split_bg
     * lends a BG lane only a newcomer's whole rest beside decode rows) and the later chunks of a prompt whose every chunk so far
     * ran as a tail band pass */
    const int sv0 = fn4::get(fn4::kPpTband), sv1 = sv0 < 0 ? pp_tband_start_ : sv0;
    /* tp8 loop D50: 4 + 1024 x N -- the form (sv) and the decode load N above which a clean stretch's decode step bands too */
    const int sv = sv1 & 1023, tload = (sv1 >> 10) & 63;
    /* tp8 loop D46: 3 = 1's (a BG lane's steps below, as 2's) + a decode lane's step while a BG step is in flight, when every
     * sequence in it has its band state on the leader (the plugin's copy since its first row); drafts, context, MIXED rows all go.
     * tp8 loop D50: 4 = 3 in EVERY stretch (a held decode step is a band pass whether a BG step is in flight or not: at a cut one
     * layer below the published one, W(h, t + 1), that is W(h, t)'s decode -- the in-process control); 4 + 1024 x N (N 1 .. 63) =
     * 3's band in a BG stretch, and in a clean one only while the decode lanes' last steps decoded > N sequences (the load
     * --draft-short-from reads): a clean stretch at low load keeps the band layer on the follower */
    if (count && !lanes_.is_bg(lane)) ++pp_tband_dec_seen_;   /* tp8 loop D50: every decode lane's step judged here */
    if (sv >= 3 && !lanes_.is_bg(lane)) {
        if (!pp_tband_copy_) return 0;
        bool on = pp_tband_bg_;
        if (!on && sv == 4) {
            if (tload == 0) {
                on = true;
            } else {
                int dec = 0;
                for (int q = 0; q < n_lanes_; ++q)
                    if (!lanes_.is_bg(q)) dec += sched_of(q).decoded_seqs();
                on = dec > tload;
                if (!on && count) ++pp_tband_dec_low_;
            }
        }
        if (!on) return 0;
        if (!tcopy::held(pp_tcopy_seq_, pp_tband_copy_, p.e)) {
            if (count) ++pp_tband_dec_skip_;
            return 0;
        }
        if (count) {
            ++pp_tband_dec_n_;
            if (!pp_tband_bg_) ++pp_tband_dec_clean_;
        }
        return j;
    }
    if (!lanes_.is_bg(lane) && sv < 2) return 0;
    static const bool t_cont_only = [] {   /* RADIANCE_PP_TEST_TBAND_CONT_ONLY=1: later chunks only -- the negative control */
        const char* v = std::getenv("RADIANCE_PP_TEST_TBAND_CONT_ONLY");
        return v && !std::strcmp(v, "1");
    }();
    bool later = false;
    for (const StepEntry& e : p.e) {
        if (!e.is_prefill || e.ckpt_restore >= 0 || e.n_spec > 0) return 0;
        if (e.ctx_len != 0) {
            if (sv < 2 || (!t_cont_only && !pp_tband_seq_.count(e.seq))) return 0;
            later = true;
        } else if (t_cont_only) {
            return 0;
        }
    }
    /* fn4 pp_tband 0 (step 2c's in-process control): this fresh BG step runs W(h, t)'s */
    if (sv == 0) {
        if (count) ++pp_tband_off_n_;
        return 0;
    }
    if (count && later) ++pp_tband_cont_n_;
    return j;
}

/* tp8 loop D50: the tail band's counters, cumulative (a cell = two lines' difference; kit d50cell.py), at the D32 line's period
 * (RADIANCE_PP_BGRATE_LINE_S) and at the run's end -- with the plugin's copy only */
std::string Engine::tband_text(const char* when) const {
    const int sv0 = fn4::get(fn4::kPpTband);
    return fmt("two-node pipeline: tail band%s (tp8 loop D46 / D50): %lld decode lanes' steps judged, %lld ran as tail band passes (%lld "
               "with no BG step in flight), %lld kept W(h, t)'s for a sequence not held, %lld clean ones kept W(h, t)'s by the load rule; "
               "%lld lane steps ran as tail band passes in all; pp_tband %d", when, (long long)pp_tband_dec_seen_, (long long)pp_tband_dec_n_,
               (long long)pp_tband_dec_clean_, (long long)pp_tband_dec_skip_, (long long)pp_tband_dec_low_, (long long)pp_tband_n_,
               sv0 < 0 ? pp_tband_start_ : sv0);
}

/* tp8 loop D39b: after a step's head decided its tail band -- which sequences have run every chunk as a tail band pass */
void Engine::pp_tband_note(int lane, int tband) {
    if (!pp_tband_on_) return;
    if (pp_tband_copy_) tcopy::note(pp_tcopy_seq_, pp_tband_copy_, tband > 0, sched_of(lane).plan().e);   /* tp8 loop D46 */
    for (const StepEntry& e : sched_of(lane).plan().e) {
        if (!e.is_prefill) { pp_tband_seq_.erase(e.seq); continue; }
        if (tband > 0 && (e.ctx_len == 0 || pp_tband_seq_.count(e.seq))) pp_tband_seq_[e.seq] = 1;
        else pp_tband_seq_.erase(e.seq);
    }
    if (pp_tband_seq_.size() > 4096) pp_tband_seq_.clear();   /* prompts never finished: forgotten (they run W(h, t)'s) */
}

int Engine::lane_head(int lane, const RadBatch* b) {
    lane_select(lane);
    /* tp8 loop P7 (RADIANCE_PP_PF_TIMELINE=1): this chunk's head, bracketed on rank 0's stream */
    const StepEntry* pfe = pf_tl_ ? pf_entry(b, lsched().plan()) : nullptr;
    const uint64_t pf0 = pfe ? rad_mono_ns() : 0;
    clear_fresh_states();
    /* tp8 loop D36: the band the lane loop chose for this step, if it chose one (taken first: every return below is a failure) */
    const int band_pre = pp_band_pre_;
    pp_band_pre_ = -1;
    if (const int ps = prepare_release(b); ps < 0) return ps;
    if (!pp_trunk_pending_[lane]) {
        RAD_ERR("two-node pipeline: lane %d's step %d was released without its build", lane, b->step);
        return RAD_E_STATE;
    }
    pp_trunk_pending_[lane] = false;
    /* tp8 S6: a prefix hit's snapshot restores, on this node's streams before the head (the follower runs them from
     * the frame before its pass); the tail reads the restored state on the same stream */
    RAD_TRY(restore_checkpoints());
    /* tp8 loop P28: the frame says so; the head runs [0, h + k) */
    const int band = band_pre >= 0 ? band_pre : pp_band_step(lane, b);
    /* tp8 loop D39: a tail band (never beside a head band); the frame says so, the lane keeps j for its tail part */
    const int tband = band > 0 ? 0 : pp_tband_step(lane, b, true);
    pp_ship_band_ = band;
    pp_ship_tband_ = tband;
    const int ss = pp_ship(b, true, lane);
    pp_ship_band_ = 0;
    pp_ship_tband_ = 0;
    if (ss < 0) return ss;
    if ((size_t)lane < pp_tband_lane_.size()) pp_tband_lane_[(size_t)lane] = tband;
    if (tband) ++pp_tband_n_;
    pp_tband_note(lane, tband);   /* tp8 loop D39b */
    dch_note_head(lane, b);   /* tp8 loop D35 */
    if (band) ++pp_band_n_;
    pp_set_pass(lane, RAD_PP_PART_HEAD, 0, 0, band);
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
    hiss_want_ = pp_stamps_ != nullptr;   /* tp8 loop D37: rank 0 times its run_step of this head */
    hiss_t0_ = hiss_t1_ = 0;
    barrier_->arrive_and_wait();      /* release the ranks */
    barrier_->arrive_and_wait();      /* and wait for them */
    if (pp_ni_) pp_ni_->end();
    if (hiss_want_) {   /* tp8 loop D37: 'hiss' = rank 0's run_step entry .. return (the release is the 'head' stamp's t0) */
        hiss_want_ = false;
        pp_stamp("hiss", lane, b->step, hiss_t0_, hiss_t1_);
    }
    if (pfp) {
        pf_tl_->mark(PfTimeline::kB, pfe->seq, pf_ci);
        const double iss = (double)(rad_mono_ns() - pf1) / 1e6;
        pfp->issue += iss;
        if (pf_ci >= 0 && (size_t)pf_ci < pfp->ch.size()) pfp->ch[(size_t)pf_ci].issue = (float)iss;   /* tp8 loop P31 */
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
    if (lanes_.lane_rows_m() > 0)   /* tp8 loop D47b */
        RAD_INFO("two-node pipeline: a decode lane's rows (RADIANCE_PP_LANE_ROWS=%lld, tp8 loop D47b) turned away %lld lends over the run",
                 (long long)lanes_.lane_rows_m(), (long long)lanes_.lane_rows_kept());
    if (lanes_.bg_lane() >= 0)
        RAD_INFO("two-node pipeline: BG lane(s) from %d (RADIANCE_PP_BG_LANE, tp8 dev M8): %lld newcomers lent whole beside decode "
                 "rows (%lld rows; %lld of them in two pieces on the 512 grid, RADIANCE_PP_BG_SPLIT; %lld whole for want of two "
                 "lanes' room); kept home: a prefix restore %lld, the BG lanes' room %lld, not a planner %lld; drafter ranges opened "
                 "by a lent piece's commit %lld (on, tp8 loop F22)%s%s", lanes_.bg_lane(),
                 (long long)c.bg, (long long)c.bg_rows, (long long)c.bg_split, (long long)c.bg_skip[LaneSet::kBgSplitRoom],
                 (long long)c.bg_skip[LaneSet::kBgRestore], (long long)c.bg_skip[LaneSet::kBgRoom],
                 (long long)c.bg_skip[LaneSet::kBgOrder], (long long)c.bg_floor,
                 lanes_.bg_hold_ms() > 0
                     ? fmt("; held for the BG lanes' room (RADIANCE_PP_BG_HOLD_MS=%lld, tp8 dev D11): %lld newcomers over %lld "
                           "steps, %lld then lent to a BG lane, %lld ran at home after the hold ran out, held %.1f ms each on "
                           "average (the longest %.1f)", (long long)lanes_.bg_hold_ms(), (long long)c.hold_reqs,
                           (long long)c.hold_steps, (long long)c.hold_lent, (long long)c.hold_home,
                           c.hold_lent + c.hold_home > 0 ? (double)c.hold_us / 1000.0 / (double)(c.hold_lent + c.hold_home) : 0.0,
                           (double)c.hold_max_us / 1000.0).c_str()
                     : "",
                 lanes_.bg_rows(cfg_.max_seqs) < cfg_.max_seqs
                     ? fmt("; a BG lane's %lld-row slab turned away %lld lends (RADIANCE_PP_BG_ROWS, tp8 loop D47)",
                           (long long)lanes_.bg_rows(cfg_.max_seqs), (long long)lanes_.bg_rows_kept()).c_str()
                     : "");
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
        if (tail_ahead_ && ti_capable_)
            RAD_INFO("two-node pipeline: RADIANCE_PP_TAIL_AHEAD=1 with lever TI on this node -- a step that saves a snapshot "
                     "inside its chunk keeps its model part with its sampler (tp8 loop FINDINGS F43)");
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
    /* tp8 dev D10 round 2: why a tail went without (loop RESULTS lq60: 1,122 ahead vs 1,732 without at W(15,35)), so the next
     * round knows which rule costs the most -- 0 KV check due, 1 a BG lane, 2 lent pieces waiting in its inbox, 3 not a decode
     * step (MIXED / prefill), 4 its drafter proposing (block / serial rows this cycle), 5 a row lent / prefill / drafted / not
     * producing, 6 Scheduler::step_next declined (a request waiting to be admitted, a row ending, the pool, the hybrid) */
    enum { kIaKv, kIaBg, kIaInbox, kIaPhase, kIaDraft, kIaRow, kIaNext, kIaWhy };
    int64_t ia_why[kIaWhy] = {};
    uint64_t ia_line_ns = 0;
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
    /* tp8 dev M9 (RADIANCE_PP_LEAD_SEG=k, the leader; the plugin reads k and declares the boundaries; with RADIANCE_PP_BG_LANE,
     * not with TAIL_AHEAD): a BG lane's step of >= RADIANCE_PP_SEG_ROWS rows runs its leader parts in layer segments -- the
     * head [0, h) and the tail [t, n) at the boundaries the plugin declared on this node -- and between two of them the loop
     * plans, heads and tails the other lanes (a tail once its X2 has landed: a decode step of its own sequences, or the oldest
     * step in flight), so a newcomer's leader part no longer holds every decode part on the leader's stream behind it (loop
     * FINDINGS F28: the follower then waits for X1, the stall line names the BG step). The next segment goes out once the
     * last one ran on rank 0's card; the frame goes to the follower with the head's last segment. One BG step in segments at a
     * time (the boundary ring buffer is one a rank). Only the order of issue changes, never a step's composition: a segment
     * boundary saves and restores the pass's live set on each rank, as the follower's do (tp8 dev M4). */
    /* kLsHeadShip (tp8 loop P27, RADIANCE_PP_LEAD_SEG_SHIP=1): the head's last segment is issued, its frame still held until
     * that segment has run on rank 0's card (lship_ev[lane]); then kLsHeadDone */
    enum { kLsNone = 0, kLsHead, kLsHeadDone, kLsTail, kLsHeadShip };
    struct LSeg {
        int st = kLsNone; const RadBatch* b = nullptr; std::vector<std::pair<int, int>> parts; size_t k = 0; bool mixed = false;
        uint64_t ship_ns = 0;       /* P27: the last head segment's issue */
        /* tp8 loop D32: the step's pp_lead_seg form / stride at its plan; whole_tail: its tail runs whole (no tail boundary
         * kept, or form 2 found no decode lane busy when its X2 landed) */
        int form = 1, stride = 1;
        bool whole_tail = false;
        int band = 0;               /* tp8 loop D36: the step's band k (its head's last segment ends at h + k), 0 none */
        int lflags = 0;             /* tp8 loop D40: RAD_PP_LSEG_SP on its head segments (pp_lead_seg + 512) */
        int ring = 0;               /* tp8 loop D32: the plugin's boundary ring the step's segments use (its lane % rings) */
        int64_t ship_s0 = 0, ship_s1 = 0;   /* P27: frames shipped by then since the first segment / at the last one's issue */
    };
    std::vector<LSeg> lsg((size_t)n_lanes_);
    std::vector<int32_t> lseg_hb, lseg_tb;
    bool lseg = false;
    int64_t lseg_steps = 0, lseg_parts = 0, lseg_between = 0, lseg_polls = 0, lseg_inside = 0, lseg_msteps = 0;
    /* tp8 dev M9b (RADIANCE_PP_LEAD_SEG_MIXED=1, with LEAD_SEG): a decode lane's MIXED newcomer step (a newcomer kept home while the
     * BG lane is busy) runs its leader parts in segments too, when it carries and lends no piece; at most one pass holds the
     * plugin's boundary ring (its head or tail segments in flight) at a time -- a step that finds it held runs that part whole */
    bool lseg_mixed = false;
    if (const char* v = std::getenv("RADIANCE_PP_LEAD_SEG_MIXED"); v && !std::strcmp(v, "1")) lseg_mixed = true;
    /* host tests (RADIANCE_PP_LEAD_SEG_TEST_WAIT_MS): a BG segment goes out no sooner than this after the last one, so the
     * other lanes' tails find their place between them on a backend whose launches finish at issue */
    uint64_t lseg_wait_ns = 0, lseg_last_ns = 0;
    if (const char* v = std::getenv("RADIANCE_PP_LEAD_SEG_TEST_WAIT_MS"); v && *v)
        lseg_wait_ns = (uint64_t)std::max<int64_t>(0, std::atoll(v)) * 1000000ull;
    /* rank 0's events after a pass's segments, two in turn; RADIANCE_PP_LEAD_SEG_AHEAD=1 (default 0): a segment goes out once
     * the one before the last ran (one queued behind the running one, as the follower's SEG_AHEAD), so a newcomer's leader part
     * does not wait for the host between its segments; 0: once the last one ran */
    /* tp8 loop D32: two a lane (with RADIANCE_PP_LEAD_SEG_RINGS=2 two BG steps run their segments at once, each in turn on its own) */
    std::vector<RadEvent> lseg_ev((size_t)n_lanes_ * 2, nullptr);
    struct EvReset { std::vector<RadEvent>* v; ~EvReset() { for (RadEvent e : *v) if (e) rad_event_destroy(e); } } lseg_ev_reset{ &lseg_ev };
    auto lev = [&](int l, int64_t k) -> RadEvent& { return lseg_ev[(size_t)l * 2 + (size_t)(k & 1)]; };
    int lseg_ahead = 0;
    if (const char* v = std::getenv("RADIANCE_PP_LEAD_SEG_AHEAD"); v && !std::strcmp(v, "1")) lseg_ahead = 1;
    /* tp8 loop P27 (RADIANCE_PP_LEAD_SEG_SHIP=1, with LEAD_SEG; default 0 = the frame goes out with the last head segment's
     * issue, as M9 built it): the frame goes once that segment has RUN on rank 0's card -- its X1 is then sent, so the
     * follower's card does not sit in the BG pass's X1 receive (loop FINDINGS F47 item 4: p50 85-95 ms a BG newcomer, 4.5 % of
     * the c32 follower) while the leader finishes the head, and the decode frames issued between the head's segments reach
     * the follower ahead of it. Only when each frame goes changes, never a pass: the held frame is the one made at the first
     * segment (its release's PSK plan, clears, checkpoint words and switch table), as M9 holds it today between the first
     * and the last segment. Host tests: RADIANCE_PP_LEAD_SEG_SHIP_TEST_WAIT_MS holds it that long past the last issue too. */
    std::vector<RadEvent> lship_ev;
    struct EvVecReset { std::vector<RadEvent>* v; ~EvVecReset() { for (RadEvent e : *v) if (e) rad_event_destroy(e); } } lship_ev_reset{ &lship_ev };
    uint64_t lship_wait_ns = 0;
    int64_t lship_n = 0, lship_ahead = 0, lship_late = 0;
    uint64_t lship_ns_sum = 0, lship_ns_max = 0;
    lseg_ship_ = false;
    struct ShipReset { Engine* e; ~ShipReset() { e->lseg_ship_ = false; } } lship_reset{ this };
    if (const char* v = std::getenv("RADIANCE_PP_LEAD_SEG"); v && *v && std::atoi(v) > 0) {
        for (int32_t x : pp_segments()) {
            if (x > 0 && x < cfg_.pp_cut_h) lseg_hb.push_back(x);
            else if (x > cfg_.pp_cut_t && x < RAD_PP_SEG_END) lseg_tb.push_back(x);
        }
        lseg = lanes_.bg_lane() >= 0 && !tail_ahead_ && (!lseg_hb.empty() || !lseg_tb.empty());
        for (RadEvent& e : lseg_ev)
            if (lseg && rad_event_create(&e) < 0) {
                RAD_ERR("two-node pipeline: could not create the leader's segment event");
                return fail_loop(RAD_E_DEVICE);
            }
        if (const char* sv = std::getenv("RADIANCE_PP_LEAD_SEG_SHIP"); lseg && !lseg_hb.empty() && sv && !std::strcmp(sv, "1")) {
            lship_ev.assign((size_t)n_lanes_, nullptr);
            for (RadEvent& e : lship_ev)
                if (rad_event_create(&e) < 0) {
                    RAD_ERR("two-node pipeline: could not create the leader's ship events (RADIANCE_PP_LEAD_SEG_SHIP)");
                    return fail_loop(RAD_E_DEVICE);
                }
            lseg_ship_ = true;
            if (const char* w = std::getenv("RADIANCE_PP_LEAD_SEG_SHIP_TEST_WAIT_MS"); w && *w)
                lship_wait_ns = (uint64_t)std::max<int64_t>(0, std::atoll(w)) * 1000000ull;
        } else if (sv && !std::strcmp(sv, "1")) {
            RAD_WARN("two-node pipeline: RADIANCE_PP_LEAD_SEG_SHIP=1 is not taken: %s (tp8 loop P27)",
                     !lseg ? "it needs RADIANCE_PP_LEAD_SEG taken" : "no head boundary is declared on this node");
        }
        std::string hs, ts;
        for (int32_t x : lseg_hb) hs += fmt("%s%d", hs.empty() ? "" : ", ", (int)x);
        for (int32_t x : lseg_tb) ts += fmt("%s%d", ts.empty() ? "" : ", ", (int)x);
        if (lseg)
            RAD_INFO("two-node pipeline: RADIANCE_PP_LEAD_SEG=%s -- a BG lane's step%s of >= %lld rows runs its leader head in segments "
                     "at layers [%s] and its tail at [%s] of W(%d,%d), the other lanes' parts between them (tp8 dev M9)%s", v,
                     lseg_mixed ? " and a decode lane's MIXED newcomer step (RADIANCE_PP_LEAD_SEG_MIXED=1, M9b)" : "",
                     (long long)rf_rows, hs.empty() ? "none: whole" : hs.c_str(), ts.empty() ? "none: whole" : ts.c_str(),
                     cfg_.pp_cut_h, cfg_.pp_cut_t,
                     lseg_ahead ? "; a segment goes out once the one before the last ran (RADIANCE_PP_LEAD_SEG_AHEAD=1)" : "");
        if (lseg_ship_)
            RAD_INFO("two-node pipeline: RADIANCE_PP_LEAD_SEG_SHIP=1 -- a segmented head's frame goes to the follower once its last "
                     "segment has run on rank 0's card (its X1 sent), the frames issued meanwhile ahead of it (tp8 loop P27)%s",
                     lship_wait_ns ? fmt("; held >= %llu ms past the last issue (RADIANCE_PP_LEAD_SEG_SHIP_TEST_WAIT_MS)",
                                         (unsigned long long)(lship_wait_ns / 1000000ull)).c_str() : "");
        if (!lseg)
            RAD_WARN("two-node pipeline: RADIANCE_PP_LEAD_SEG=%s is not taken: %s (tp8 dev M9)", v,
                     lanes_.bg_lane() < 0 ? "it needs RADIANCE_PP_BG_LANE" : tail_ahead_ ? "not with RADIANCE_PP_TAIL_AHEAD=1"
                                                                                    : "the plugin declared no boundary on this node");
    }
    /* tp8 loop D32 (LSP, fn4 pp_lead_seg, the leader's; fn4_switch.h): form + 16 x stride -- 0 a BG step's leader parts whole
     * (BEST's program beside the declared boundaries), 1 M9 as built, 2 decode first; the start value RADIANCE_PP_LEAD_SEG_FORM
     * (default 1 = today's M9) where the boundaries were taken, else 0 (the switch then takes 0 only) */
    int lsp_start = 0;
    if (lseg) {
        lsp_start = 1;
        if (const char* v = std::getenv("RADIANCE_PP_LEAD_SEG_FORM"); v && *v) {
            const int x = std::atoi(v);
            /* tp8 loop D36: + 256 = no band passes; D40: + 8 = the tail whole, + 512 = SP in the head's segments */
            /* tp8 loop D49b: + 1024 x N = segments only while the decode lanes' last steps decoded <= N sequences */
            /* tp8 loop F81: + 65536 = no position-order guard */
            if (x >= 0 && x <= 131071 && (x & 7) <= 2) lsp_start = x;
            else RAD_WARN("two-node pipeline: RADIANCE_PP_LEAD_SEG_FORM=%s is not taken: form + 8 x tail-whole + 16 x stride (+ 256, "
                          "+ 512, + 1024 x N, + 65536), form 0 / 1 / 2 (tp8 loop D32 / D40 / D49b / F81); 1 (M9 as built)", v);
        }
        RAD_INFO("two-node pipeline: fn4 pp_lead_seg starts at %d (form %d%s, stride %d%s%s; RADIANCE_PP_LEAD_SEG_FORM, tp8 loop D32): 0 "
                 "= a BG step's leader parts whole, 1 = M9 as built, 2 = decode first (a decode tail whose X2 landed and its lane's "
                 "next head go before the BG step's next segment, one segment in the stream, parts whole while no decode lane is "
                 "busy); + 8 = the tail whole (D40); + 16 x s = every s-th declared boundary; + 256 = no band passes (D36); + 512 = "
                 "the head's segments may run the plugin's SP plan (D40, RADIANCE_DS41_SPHC_LSEG=1); + 1024 x N = segments only while "
                 "the decode lanes' last steps decoded <= N sequences, the parts whole above (D49b); a step carrying a prompt entry "
                 "that is not a whole fresh prompt runs its parts whole (the position-order guard, F81; + 65536 = none)%s", lsp_start, lsp_start & 7,
                 (lsp_start & 7) == 0 ? ": whole" : (lsp_start & 7) == 1 ? ": M9" : ": decode first",
                 std::max(1, (lsp_start >> 4) & 15), (lsp_start & 8) ? ", tail whole" : "", (lsp_start & 512) ? ", SP word" : "",
                 (lsp_start >> 10) ? fmt("; starts with N = %d", (lsp_start >> 10) & 63).c_str() : "");
    }
    fn4::lead_seg_register(lseg);
    auto lsp_now = [&]() {
        if (!lseg) return 0;
        const int v = fn4::get(fn4::kPpLeadSeg);
        return v < 0 ? lsp_start : v;   /* never requested: the start's */
    };
    /* the declared boundaries a stride keeps: every s-th */
    auto lsp_kept = [](const std::vector<int32_t>& bs, int stride) {
        std::vector<int32_t> o;
        int j = 0;
        for (int32_t x : bs)
            if (++j % std::max(1, stride) == 0) o.push_back(x);
        return o;
    };
    int64_t lsp_first = 0, lsp_whole = 0, lsp_wtail = 0;
    int64_t lsp_spseg = 0, lsp_tw8 = 0;   /* tp8 loop D40: head segments issued with the SP word; steps whose tail + 8 kept whole */
    int64_t lsp_load = 0;                 /* tp8 loop D49b: BG steps whose parts ran whole by + 1024 x N (decode load above N) */
    int64_t lsp_pos = 0;                  /* tp8 loop F81: steps whose parts ran whole by the position-order guard */
    /* tp8 loop D32 (RADIANCE_PP_LEAD_SEG_RINGS=2, the leader; its plugin reads the same env and declares a second ring): a BG
     * step's segments use ring lane % rings, so the two BG lanes of B2L (lanes 2 / 3) each segment while the other does */
    int lseg_rings = 1;
    if (const char* v = std::getenv("RADIANCE_PP_LEAD_SEG_RINGS"); lseg && v && !std::strcmp(v, "2")) {
        lseg_rings = 2;
        RAD_INFO("two-node pipeline: RADIANCE_PP_LEAD_SEG_RINGS=2 -- two boundary rings a card (the plugin's): a lane's BG step uses ring "
                 "lane %% 2, so two BG lanes run their leader parts in segments at once (tp8 loop D32)");
    }
    int64_t lsp_both = 0;   /* BG steps whose segments began while another ring's were in flight */
    /* tp8 loop P28 (RADIANCE_PP_BAND=1, the leader): fresh BG steps run as band passes of the plugin's band */
    pp_band_on_ = false;
    if (const char* bv = std::getenv("RADIANCE_PP_BAND"); bv && !std::strcmp(bv, "1")) {
        if (pp_band_k() > 0 && lanes_.bg_lane() >= 0) {
            pp_band_on_ = true;
            /* a newcomer's step, not a cold prompt's chunk (those are the cold cells' and run SP: RADIANCE_DS41_SPHC) */
            pp_band_max_rows_ = 2047;
            if (const char* mr = std::getenv("RADIANCE_PP_BAND_MAX_ROWS"); mr && *mr) pp_band_max_rows_ = std::max<int64_t>(1, std::atoll(mr));
            g_band_min_seqs = 0;
            g_band_low = 0;
            if (const char* ms = std::getenv("RADIANCE_PP_BAND_MIN_SEQS"); ms && *ms) g_band_min_seqs = std::max<int64_t>(0, std::atoll(ms));
            if (g_band_min_seqs > 0)
                RAD_INFO("two-node pipeline: RADIANCE_PP_BAND_MIN_SEQS=%lld -- band passes only while the decode lanes' last steps "
                         "decoded >= %lld sequences together (tp8 loop P28)", (long long)g_band_min_seqs, (long long)g_band_min_seqs);
            RAD_INFO("two-node pipeline: RADIANCE_PP_BAND=1 -- a BG lane's step of fresh prompts runs as a band pass: the head over "
                     "[0, %d), the follower from %d (the plugin's band %d; steps of <= %lld rows, RADIANCE_PP_BAND_MAX_ROWS; tp8 loop P28)",
                     cfg_.pp_cut_h + pp_band_k(), cfg_.pp_cut_h + pp_band_k(), pp_band_k(), (long long)pp_band_max_rows_);
        } else {
            RAD_WARN("two-node pipeline: RADIANCE_PP_BAND=1 is not taken: %s (tp8 loop P28)",
                     pp_band_k() <= 0 ? "the plugin declared no band (RADIANCE_DS41_PP_BAND=k on both nodes)"
                                      : "it needs RADIANCE_PP_BG_LANE");
        }
    }
    /* tp8 loop D39 (RADIANCE_PP_TAIL_BAND=1, the leader): fresh BG steps run as tail band passes of the plugin's tail band */
    pp_tband_on_ = false;
    pp_tband_copy_ = 0;
    pp_tband_lane_.assign((size_t)n_lanes_, 0);
    pp_tband_seq_.clear();
    pp_tcopy_seq_.clear();
    /* tp8 loop D50: 4 / 4 + 1024 x N (N 1 .. 63) as well (fn4 pp_tband's values) */
    auto tband_env_ok = [](const char* v) {
        if (!v || !*v) return false;
        for (const char* c = v; *c; ++c)
            if (*c < '0' || *c > '9') return false;
        const long x = std::strtol(v, nullptr, 10);
        return (x >= 1 && x <= 4) || (x > 1024 && x <= 4 + 1024 * 63 && (x & 1023) == 4);
    };
    if (const char* tv = std::getenv("RADIANCE_PP_TAIL_BAND"); tband_env_ok(tv)) {
        pp_tband_start_ = std::atoi(tv);   /* tp8 loop D39b: 2 = the later chunks of an all-band prompt too; D46: 3 = + decode lanes */
        if (pp_tail_band_k() > 0 && lanes_.bg_lane() >= 0 && !pp_band_on_) {
            pp_tband_on_ = true;
            if (!std::getenv("RADIANCE_PP_BAND")) {   /* P28's row cap, read here when RADIANCE_PP_BAND did not read it */
                pp_band_max_rows_ = 2047;
                if (const char* mr = std::getenv("RADIANCE_PP_BAND_MAX_ROWS"); mr && *mr) pp_band_max_rows_ = std::max<int64_t>(1, std::atoll(mr));
            }
            RAD_INFO("two-node pipeline: RADIANCE_PP_TAIL_BAND=%d -- a BG lane's step of fresh prompts runs as a tail band pass: the "
                     "follower over [%d, %d) (then the band layers' own KV stores), the leader's tail from %d (the plugin's tail band %d; "
                     "steps of <= %lld rows; tp8 loop D39)%s", pp_tband_start_, cfg_.pp_cut_h, cfg_.pp_cut_t - pp_tail_band_k(),
                     cfg_.pp_cut_t - pp_tail_band_k(), pp_tail_band_k(), (long long)pp_band_max_rows_,
                     pp_tband_start_ == 2 ? "; and so does a later chunk of a prompt whose every chunk so far ran as one (D39b)" : "");
            /* tp8 loop D46: with the plugin's copy, pp_tband 3 (a decode lane's step while a BG step is in flight) may be set */
            pp_tband_copy_ = pp_tail_copy();
            if (pp_tband_copy_)
                RAD_INFO("two-node pipeline: the plugin keeps the tail band layers' state on the leader for every sequence (its tail "
                         "copy, form %d: %s; tp8 loop D46): fn4 pp_tband 3 runs a decode lane's step as a tail band pass too while a BG "
                         "step is in flight%s", pp_tband_copy_, pp_tband_copy_ == 1 ? "every pass that is no tail band pass"
                         : "decode-only passes, a window of that many positions", (pp_tband_start_ & 1023) >= 3 ? " -- on from the start "
                         "(RADIANCE_PP_TAIL_BAND=3 / 4)" : "");
            if (pp_tband_copy_)
                RAD_INFO("two-node pipeline: fn4 pp_tband 4 runs a held decode lane's step as a tail band pass in EVERY stretch, 4 + 1024 x N "
                         "in a BG stretch and in a clean one only while the decode lanes' last steps decoded > N sequences: layer %d on the "
                         "leader for such a step, on the follower for the others (tp8 loop D50)%s", cfg_.pp_cut_t - pp_tail_band_k(),
                         (pp_tband_start_ & 1023) == 4 ? fmt(" -- RADIANCE_PP_TAIL_BAND=%d from the start (N = %d)", pp_tband_start_,
                                                             (pp_tband_start_ >> 10) & 63).c_str() : "");
            if ((pp_tband_start_ & 1023) >= 3 && !pp_tband_copy_) {
                RAD_WARN("two-node pipeline: RADIANCE_PP_TAIL_BAND=%d runs as 1: the plugin declared no tail band copy (tp8 loop D46: "
                         "RADIANCE_DS41_PP_TBAND_COPY=1 / RADIANCE_LLAMA_PP_TBAND_COPY=1 on both nodes)", pp_tband_start_);
                pp_tband_start_ = 1;
            }
        } else {
            RAD_WARN("two-node pipeline: RADIANCE_PP_TAIL_BAND=1 is not taken: %s (tp8 loop D39)",
                     pp_tail_band_k() <= 0 ? "the plugin declared no tail band" : pp_band_on_ ? "not beside RADIANCE_PP_BAND=1"
                                                                                          : "it needs RADIANCE_PP_BG_LANE");
        }
    }
    fn4::tband_register(pp_tband_on_ ? (pp_tband_copy_ > 0 ? 2 : 1) : 0);   /* tp8 loop D39 step 2c: fn4 pp_tband 1 needs it taken; D46: 3 the copy */
    /* tp8 loop D36 (RADIANCE_PP_LEAD_SEG_BAND=1, the leader; default unset = a band step's head whole, as P28 built it): a band
     * step's head in LSP's segments too -- [0, b1) .. [b_last, h + k) at the head boundaries the plugin declared, the band word on
     * every segment, the held frame a band frame -- so the k layers a band pass adds to the leader's BG head no longer hold every
     * decode head behind them (P28's kill: lq138, the follower's freed newcomer work came back as waiting). The same passes in the
     * same order on rank 0's stream: a band pass's arithmetic is P28's, its segments' boundaries M9's (their live sets of W(h, t)
     * hold what the band layers and X1 at h + k read below a boundary, ds41layers::band_crossing / lead_segment_crossing). */
    bool lseg_band = false;
    pp_band_seg_n_ = 0;
    pp_band_off_n_ = 0;
    pp_band_off_ = false;
    if (const char* v = std::getenv("RADIANCE_PP_LEAD_SEG_BAND"); v && !std::strcmp(v, "1")) {
        lseg_band = lseg && pp_band_on_ && !lseg_hb.empty();
        if (lseg_band)
            RAD_INFO("two-node pipeline: RADIANCE_PP_LEAD_SEG_BAND=1 -- a band step's head runs in segments too, the last one [%d, %d) "
                     "(tp8 loop D36)", (int)lseg_hb.back(), cfg_.pp_cut_h + pp_band_k());
        else
            RAD_WARN("two-node pipeline: RADIANCE_PP_LEAD_SEG_BAND=1 is not taken: %s (tp8 loop D36)",
                     !pp_band_on_ ? "it needs RADIANCE_PP_BAND=1 taken" : !lseg ? "it needs RADIANCE_PP_LEAD_SEG taken"
                                                                        : "no head boundary is declared on this node");
    }
    if (const char* sv = std::getenv("RADIANCE_PP_LEAD_SEG_SHIP"); !lseg_ship_ && sv && !std::strcmp(sv, "1") &&
        !(std::getenv("RADIANCE_PP_LEAD_SEG") && std::atoi(std::getenv("RADIANCE_PP_LEAD_SEG")) > 0))
        RAD_WARN("two-node pipeline: RADIANCE_PP_LEAD_SEG_SHIP=1 is not taken: it needs RADIANCE_PP_LEAD_SEG (tp8 loop P27)");
    uint64_t lseg_line_ns = rad_mono_ns();
    /* tp8 dev D11: RADIANCE_PP_BG_HOLD_MS's counts (run so far) once a minute, so a cell's window is the difference of two lines */
    uint64_t hold_line_ns = rad_mono_ns();
    /* tp8 loop P23: the BG lanes' steps (run so far) once a minute under RADIANCE_PP_BG_ROOM, so a cell's window is the difference
     * of two lines: steps, newcomers, steps with >= 2 newcomers (batched) and their newcomers / rows, >= 3, the widest step */
    uint64_t bgst_line_ns = rad_mono_ns();
    bool bgst_seen = false;   /* tp8 loop P23s: the room was on at some point (the switch) */
    uint64_t sb_line_ns = bgst_line_ns;   /* tp8 loop D18: the short block's minute line */
    auto bgst_line = [&](const char* when) {
        const LaneSet::Counts c = lanes_.counts();
        RAD_INFO("two-node pipeline: BG steps%s (RADIANCE_PP_BG_ROOM=%lld, tp8 loop P23): %lld steps with %lld newcomers; %lld steps "
                 "with >= 2 newcomers (%lld newcomers, %lld rows), %lld with >= 3; the widest %lld rows; kept home for the BG lanes' "
                 "room %lld%s", when, (long long)lanes_.bg_room(), (long long)c.bgst_n, (long long)c.bgst_newc, (long long)c.bgst_b2,
                 (long long)c.bgst_b2_newc, (long long)c.bgst_b2_rows, (long long)c.bgst_b3, (long long)c.bgst_max_rows,
                 (long long)c.bg_skip[LaneSet::kBgRoom],
                 lanes_.bg_join_ms() > 0
                     ? fmt("; past the shared room (RADIANCE_PP_BG_JOIN_MS=%lld, P23b): %lld joined a busy BG lane, %lld ran at home "
                           "(its step in flight not within %lld ms); BG steps %.2f us a row (plan to commit)",
                           (long long)lanes_.bg_join_ms(), (long long)c.bg_join_yes, (long long)c.bg_join_no,
                           (long long)lanes_.bg_join_ms(), (double)c.bg_us_row_x100 / 100.0).c_str()
                     : "");
    };
    auto hold_line = [&]() {
        const LaneSet::Counts c = lanes_.counts();
        RAD_INFO("two-node pipeline: RADIANCE_PP_BG_HOLD_MS=%lld so far (tp8 dev D11): %lld newcomers held over %lld steps, %lld "
                 "then lent to a BG lane, %lld ran at home after the hold ran out (held %.1f ms in all, the longest %.1f); BG "
                 "lanes took %lld newcomers, the BG lanes' room kept %lld home or held", (long long)lanes_.bg_hold_ms(),
                 (long long)c.hold_reqs, (long long)c.hold_steps, (long long)c.hold_lent, (long long)c.hold_home,
                 (double)c.hold_us / 1000.0, (double)c.hold_max_us / 1000.0, (long long)c.bg,
                 (long long)c.bg_skip[LaneSet::kBgRoom]);
    };
    auto lseg_line = [&](const char* when) {
        RAD_INFO("two-node pipeline: %lld BG steps ran their leader parts in %lld segments, %lld other lanes' tails while one "
                 "was in flight (%lld of them between two of its segments), %lld polls with nothing to issue "
                 "(RADIANCE_PP_LEAD_SEG, tp8 dev M9)%s%s; pp_lead_seg %d now (tp8 loop D32): %lld tails taken first while a BG "
                 "step was in segments, %lld BG steps ran their head whole and %lld their tail whole with no decode lane busy; %d "
                 "boundary ring(s), %lld BG parts began their segments beside another BG step's; %lld head segments carried "
                 "the SP word (+ 512), %lld BG steps kept their tail whole by + 8 (tp8 loop D40), %lld BG steps ran their parts "
                 "whole by the decode load rule (+ 1024 x N, tp8 loop D49b), %lld steps ran their parts whole by the position-order "
                 "guard (a prompt entry that is not a whole fresh prompt; + 65536 = no guard, tp8 loop F81)",
                 (long long)lseg_steps, (long long)lseg_parts, (long long)lseg_between,
                 (long long)lseg_inside, (long long)lseg_polls,
                 lseg_mixed ? fmt("; %lld MIXED newcomer steps of decode lanes among them (RADIANCE_PP_LEAD_SEG_MIXED=1, M9b)",
                                  (long long)lseg_msteps).c_str() : "", when, lsp_now(), (long long)lsp_first,
                 (long long)lsp_whole, (long long)lsp_wtail, lseg_rings, (long long)lsp_both, (long long)lsp_spseg,
                 (long long)lsp_tw8, (long long)lsp_load, (long long)lsp_pos);
        if (lseg_ship_)
            RAD_INFO("two-node pipeline: %lld segmented heads' frames shipped after their last segment ran (RADIANCE_PP_LEAD_SEG_SHIP=1, "
                     "tp8 loop P27)%s: held past the last segment's issue %.2f ms on average, the longest %.2f; %lld other frames went "
                     "ahead of them since their first segment, %lld of them after the last segment's issue", (long long)lship_n, when,
                     lship_n ? (double)lship_ns_sum / 1e6 / (double)lship_n : 0.0, (double)lship_ns_max / 1e6,
                     (long long)lship_ahead, (long long)lship_late);
    };
    /* the boundary ring is held while a pass's head or tail segments are in flight */
    auto lseg_ring_busy = [&](int ring) {
        for (const LSeg& q : lsg)
            if ((q.st == kLsHead || q.st == kLsTail) && q.ring == ring) return true;
        return false;
    };
    /* whether a pass's next segment (index s.k of its part) may go out: the one lseg_ahead before the last has run */
    auto lseg_seg_ok = [&](const LSeg& q, int l) {
        const int64_t w = (int64_t)q.k - 1 - (q.form == 2 ? 0 : lseg_ahead);   /* tp8 loop D32: form 2 one at a time */
        if (w >= 0 && rad_event_query(lev(l, w)) == 0) return false;
        if (q.form != 2 || lseg_rings < 2) return true;
        /* tp8 loop D32, two rings, form 2: one BG segment in the stream over both BG steps (each other step's last one ran) */
        for (int o = 0; o < n_lanes_; ++o) {
            const LSeg& r = lsg[(size_t)o];
            if (o != l && (r.st == kLsHead || r.st == kLsTail) && r.k >= 1 && rad_event_query(lev(o, (int64_t)r.k - 1)) == 0) return false;
        }
        return true;
    };
    auto lseg_busy = [&]() {
        for (const LSeg& s : lsg)
            if (s.st != kLsNone) return true;
        return false;
    };
    auto ia_ok = [&](int l, const RadBatch* bb, bool kv_due_now) {
        if (!ia) return false;
        const int why = kv_due_now ? kIaKv : lanes_.is_bg(l) ? kIaBg : lanes_.inbox_n(l) > 0 ? kIaInbox
                        : bb->phase != RAD_PHASE_DECODE ? kIaPhase
                        : (block_rows_ > 0 || (serial_drafting() && serial_rows_ > 0)) ? kIaDraft : -1;
        if (why >= 0) { ++ia_why[why]; return false; }
        for (const StepEntry& e : sched_of(l).plan().e)
            if (e.home >= 0 || e.is_prefill || e.n_spec > 0 || !e.produces_token) { ++ia_why[kIaRow]; return false; }
        return true;
    };
    auto ia_line = [&](const char* when) {
        RAD_INFO("two-node pipeline: %lld lane steps issued ahead of the collect before them, %lld tails without "
                 "(RADIANCE_PP_ISSUE_AHEAD=1, tp8 dev D10)%s -- without: KV check due %lld, a BG lane %lld, lent pieces in its "
                 "inbox %lld, a MIXED / prefill step %lld, its drafter proposing %lld, a row lent / prefill / drafted %lld, "
                 "step_next declined %lld", (long long)ia_n, (long long)ia_skip, when, (long long)ia_why[kIaKv],
                 (long long)ia_why[kIaBg], (long long)ia_why[kIaInbox], (long long)ia_why[kIaPhase],
                 (long long)ia_why[kIaDraft], (long long)ia_why[kIaRow], (long long)ia_why[kIaNext]);
    };
    auto x2_landed = [&](int l) {
        const RadPpRing* x2 = pp_recv_[l * RAD_PP_N_RINGS + RAD_PP_X2];
        return x2 && __atomic_load_n(&x2->landed, __ATOMIC_ACQUIRE) > __atomic_load_n(&x2->taken[0], __ATOMIC_ACQUIRE);
    };
    /* a lane step that may go before an older one: decode rows of its own sequences only (they read committed KV; a prompt's
     * chunk may read an earlier chunk's still in flight, a lent piece its home's), and no piece lent to its lane waiting in
     * its inbox (a home lends to the lanes in flight in their commit order, and each takes its piece at its next plan) */
    /* tp8 dev M9b: a step whose head ran (whole or in segments) and whose X2 has landed waits for the boundary ring for its tail
     * segments: a new head then runs whole rather than take the ring before it */
    auto lseg_tail_waits = [&](int ring) {
        if (lseg_tb.empty()) return false;
        for (int q = 0; q < n_lanes_; ++q)
            if (lsg[(size_t)q].st == kLsHeadDone && !lsg[(size_t)q].whole_tail && lsg[(size_t)q].ring == ring && x2_landed(q))
                return true;
        return false;
    };
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
    /* tp8 loop D32: a decode lane has a step in flight (form 2 segments a BG step's parts only then: a lone prompt -- the cold
     * cells -- keeps its one launch chain) */
    auto decode_live = [&]() {
        for (int q = 0; q < n_lanes_; ++q)
            if (!lanes_.is_bg(q) && busy[(size_t)q]) return true;
        return false;
    };
    /* tp8 loop D49b: the decode load pp_lead_seg + 1024 x N is judged by -- the sequences the decode lanes' last steps decoded
     * (Scheduler::decoded_seqs, the load --draft-short-from reads) */
    auto decode_seqs = [&]() {
        int n = 0;
        for (int q = 0; q < n_lanes_; ++q)
            if (!lanes_.is_bg(q)) n += sched_of(q).decoded_seqs();
        return n;
    };
    /* ... and whether a step that is no BG step in segments may take its tail now (the pick below's rule for those) */
    auto decode_ready = [&]() {
        bool passed = false;
        for (const auto& f : fifo) {
            if (lsg[(size_t)f.first].st != kLsNone) { passed = true; continue; }
            if (!x2_landed(f.first)) {
                if (!passed) return false;
                continue;
            }
            if (!passed || !carries_lent(f.first)) return true;
        }
        return false;
    };
    /* tp8 loop D32: the decode rate with a BG step in flight vs without (the quantity LSP / R9 / MERGE move; the MERGE line's
     * definitions, so the numbers compare across images): a round with work in flight is BG time while a BG lane has a step in
     * flight or a newcomer in its inbox; a decode lane's commit counts its tokens (each produced row + its accepted drafts) by the
     * state at the commit; a gap > 100 ms between decode commits with a BG step in flight is a stop. One line a minute
     * (RADIANCE_PP_BGRATE_LINE_S) and at the run's end, cumulative: a cell = two lines' difference. Only with a BG lane. */
    struct Bgr {
        uint64_t last = 0, last_dec = 0;
        bool last_bg = false;
        double bg_s = 0, nobg_s = 0, stop_ms = 0;
        int64_t tok_bg = 0, tok_nobg = 0, stops = 0, bg_commits = 0;
    } bgr;
    const bool bgr_on = lanes_.bg_lane() >= 0;
    uint64_t bgr_every = 60000000000ull, bgr_line_ns = rad_mono_ns(), dch_line_ns = 0;
    if (const char* v = std::getenv("RADIANCE_PP_BGRATE_LINE_S"); v && *v && std::atoi(v) > 0) bgr_every = (uint64_t)std::atoi(v) * 1000000000ull;
    auto bgr_line = [&](const char* when) {
        const double r_bg = bgr.bg_s > 0 ? (double)bgr.tok_bg / bgr.bg_s : 0.0, r_no = bgr.nobg_s > 0 ? (double)bgr.tok_nobg / bgr.nobg_s : 0.0;
        RAD_INFO("two-node pipeline: decode beside BG steps%s (tp8 loop D32): decode tokens with a BG step in flight %lld in %.1f s = %.1f tok/s, "
                 "without %lld in %.1f s = %.1f tok/s, ratio %.3f; decode-commit gaps > 100 ms with a BG step in flight %lld (%.0f ms), BG "
                 "steps committed %lld (%.1f ms a BG step); pp_lead_seg %d", when, (long long)bgr.tok_bg, bgr.bg_s, r_bg,
                 (long long)bgr.tok_nobg, bgr.nobg_s, r_no, r_no > 0 ? r_bg / r_no : 0.0, (long long)bgr.stops, bgr.stop_ms,
                 (long long)bgr.bg_commits, bgr.bg_commits ? bgr.stop_ms / (double)bgr.bg_commits : 0.0, lsp_now());
    };
    int next = 0;   /* the lane planned first on the next round */
    int64_t lane_steps = 0, rounds_two = 0;
    int64_t kv_check_at = 0;   /* tp8 S6: trunk steps shipped at the last KV-copy check */
    /* tp8 loop P37: the due check's deferral / drain with prompt rows in flight (the step it began at, -1 = none) + counts */
    int64_t kv_defer_from = -1, kv_drain_from = -1, kv_deferred = 0, kv_drained = 0;
    bool kv_draining = false;
    /* tp8 MERGE (fn4 pp_merge, core/sched/lanes.h merge_into / split_from): while wanted, decode lanes 1 .. nd-1 are parked and
     * lane 0 runs every decode sequence. A merge first DRAINS (lane 0 waits until no decode lane has a step in flight), then
     * moves the parked lanes' requests and free cards to lane 0; at each of lane 0's plans while merged a parked lane's
     * stragglers (a newcomer whose piece was in a BG lane) follow. A split happens at lane 0's plan with no decode lane in flight.
     * The counts are cumulative (a cell = two minute lines' difference). */
    const bool merge_ok = lanes_.decode_n() >= 2 && !tail_ahead_ && !ia;
    if (!merge_ok && pp_merge_.load(std::memory_order_relaxed))
        RAD_WARN("two-node pipeline: fn4 pp_merge is not taken: %s (tp8 MERGE)",
                 lanes_.decode_n() < 2 ? "fewer than two decode lanes" : "not with RADIANCE_PP_TAIL_AHEAD / RADIANCE_PP_ISSUE_AHEAD");
    struct Merge {
        bool on = false, draining = false;
        uint64_t t_want = 0, t_on = 0, last = 0, last_dec = 0;
        bool last_bg = false, last_work = false;
        int64_t merges = 0, splits = 0, drains = 0, sweeps = 0;
        double drain_ms = 0, merged_s = 0, wall_s = 0, bg_s = 0, nobg_s = 0, stop_ms = 0;
        int64_t tok_bg = 0, tok_nobg = 0, steps_m = 0, tok_m = 0, rows_m = 0, steps_s = 0, tok_s = 0, rows_s = 0;
        int64_t stops = 0, bg_commits = 0, moved_in = 0, moved_out = 0, cards = 0, w_lent = 0, w_ctx = 0, w_other = 0;
        int64_t relay_steps = 0;   /* bit 32: BG steps a parked lane ran */
        void note(const LaneSet::MoveCounts& c, bool in) {
            (in ? moved_in : moved_out) += c.moved;
            cards += c.cards;
            w_lent += c.lent;
            w_ctx += c.ctx;
            w_other += c.other;
        }
    } mg;
    std::vector<char> lane_merged((size_t)n_lanes_, 0);   /* the lane's step in flight was planned merged */
    std::vector<char> lane_relay((size_t)n_lanes_, 0);    /* ... was a relay lane's BG step (bit 32) */
    uint64_t mg_line_ns = rad_mono_ns();
    /* the counts run from the start (pp_merge 0 too), so the start's own split-lane reading is the A of an in-process A / B */
    const bool mg_seen = merge_ok;
    /* bit 16: D11's hold while merged, this long at most (then the newcomer runs at home as today's MIXED step) */
    int64_t mg_hold_ms = 1500;
    if (const char* v = std::getenv("RADIANCE_PP_MERGE_HOLD_MS"); v && *v && std::atoi(v) > 0) mg_hold_ms = std::min(60000, std::atoi(v));
    uint64_t mg_bg_last = 0;   /* the last round with BG work (pp_merge_ms' clock) */
    int64_t mg_held_rounds = 0, mg_hyst_rounds = 0;
    int64_t mg_tails_first = 0;   /* bit 64: decode tails taken before a long pass whose X2 had landed */
    int mg_pm_last = pp_merge_.load(std::memory_order_relaxed);
    /* the line's period (RADIANCE_PP_MERGE_LINE_S, default 60 s); it prints at every pp_merge change too, so a switch arm's
     * window is the difference of the lines at its two ends */
    uint64_t mg_line_every = 60000000000ull;
    if (const char* v = std::getenv("RADIANCE_PP_MERGE_LINE_S"); v && *v && std::atoi(v) > 0)
        mg_line_every = (uint64_t)std::atoi(v) * 1000000000ull;
    auto mg_line = [&](const char* when) {
        const double r_bg = mg.bg_s > 0 ? (double)mg.tok_bg / mg.bg_s : 0.0, r_no = mg.nobg_s > 0 ? (double)mg.tok_nobg / mg.nobg_s : 0.0;
        RAD_INFO("two-node pipeline: merge%s (fn4 pp_merge=%d, tp8 MERGE): %lld merges (%lld drains, %.1f ms), %lld splits, merged %.1f "
                 "s of %.1f s; decode lane steps merged %lld (%lld tokens, %lld rows) / split %lld (%lld tokens, %lld rows); decode "
                 "tokens with a BG step in flight %lld in %.1f s = %.1f tok/s, without %lld in %.1f s = %.1f tok/s, ratio %.3f; "
                 "decode-commit gaps > 100 ms with a BG step in flight %lld (%.0f ms), BG steps committed %lld (%.1f ms a BG step); "
                 "requests moved in %lld / out %lld (+ %lld at later plans), free cards %lld, left behind: lent %lld, kept context rows "
                 "%lld, in a step %lld; held for the BG lanes %lld newcomers over %lld steps (%lld lent later, %lld home after the hold), "
                 "rounds merged by a hold %lld / by pp_merge_ms=%d %lld; relay lane BG steps %lld; decode tails before a landed long pass %lld", when, pp_merge_.load(std::memory_order_relaxed), (long long)mg.merges,
                 (long long)mg.drains, mg.drain_ms, (long long)mg.splits, mg.merged_s, mg.wall_s, (long long)mg.steps_m,
                 (long long)mg.tok_m, (long long)mg.rows_m, (long long)mg.steps_s, (long long)mg.tok_s, (long long)mg.rows_s,
                 (long long)mg.tok_bg, mg.bg_s, r_bg, (long long)mg.tok_nobg, mg.nobg_s, r_no, r_no > 0 ? r_bg / r_no : 0.0,
                 (long long)mg.stops, mg.stop_ms, (long long)mg.bg_commits, mg.bg_commits ? mg.stop_ms / (double)mg.bg_commits : 0.0,
                 (long long)mg.moved_in, (long long)mg.moved_out, (long long)mg.sweeps, (long long)mg.cards, (long long)mg.w_lent,
                 (long long)mg.w_ctx, (long long)mg.w_other, (long long)lanes_.counts().hold_reqs, (long long)lanes_.counts().hold_steps,
                 (long long)lanes_.counts().hold_lent, (long long)lanes_.counts().hold_home, (long long)mg_held_rounds,
                 pp_merge_ms_.load(std::memory_order_relaxed), (long long)mg_hyst_rounds, (long long)mg.relay_steps, (long long)mg_tails_first);
    };
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
        /* tp8 loop P37: whether prompt rows are in flight or waiting (a lone prompt's chunks, a newcomer): a drain now
         * puts one pass of bubble on both nodes inside that prompt (lq212: 134-251 ms in 5 of 12 colds). Without
         * RADIANCE_PP_KV_CHECK_DEFER the check still drains at once and the line below says so; with it the check waits
         * for a round without them, at most every more trunk steps (then it drains as before). */
        bool kv_hold = kv_due && !fifo.empty();
        if (kv_hold && !kv_draining) {   /* a drain once begun runs to its end (a newcomer arriving meanwhile waits) */
            bool prompt_rows = false;
            for (const auto& f : fifo) prompt_rows = prompt_rows || f.second->phase != RAD_PHASE_DECODE;
            for (int k = 0; k < n_lanes_ && !prompt_rows; ++k) prompt_rows = lanes_.prefill_pending(k) || lanes_.inbox_n(k) > 0;
            if (prompt_rows && pp_kv_check_defer_ && pp_steps_shipped_ - kv_check_at < 2 * (int64_t)pp_kv_check_every_) {
                kv_hold = false;
                if (kv_defer_from < 0) kv_defer_from = pp_steps_shipped_;
            } else {
                kv_draining = true;
                if (prompt_rows) kv_drain_from = pp_steps_shipped_;
            }
        }
        if (kv_due && fifo.empty()) {
            pp_kv_check_send();
            if (kv_defer_from >= 0 || kv_drain_from >= 0)
                RAD_INFO("two-node pipeline: the KV-copy check due after trunk step %lld went out after step %lld: %s (tp8 loop P37; "
                         "%lld deferred, %lld drained with prompt rows in flight so far)",
                         (long long)(kv_check_at + pp_kv_check_every_), (long long)pp_steps_shipped_,
                         kv_drain_from >= 0
                             ? (pp_kv_check_defer_ ? "drained the pipeline with prompt rows in flight at the deferral's bound "
                                                     "(no lane planned until every tail was in)"
                                                   : "drained the pipeline with prompt rows in flight (no lane planned until "
                                                     "every tail was in: one pass of bubble on both nodes inside a prompt; "
                                                     "RADIANCE_PP_KV_CHECK_DEFER=1 defers it)")
                             : "deferred while prompt rows were in flight (RADIANCE_PP_KV_CHECK_DEFER=1)",
                         (long long)(kv_deferred + (kv_drain_from < 0 ? 1 : 0)), (long long)(kv_drained + (kv_drain_from >= 0 ? 1 : 0)));
            if (kv_drain_from >= 0) ++kv_drained;
            else if (kv_defer_from >= 0) ++kv_deferred;
            kv_defer_from = kv_drain_from = -1;
            kv_draining = false;
            kv_check_at = pp_steps_shipped_;
            kv_hold = false;
        }
        /* tp8 MERGE: the state for this round -- the time since the last round goes to the last round's state; merge when wanted
         * (lane 0 then waits for the drain), the drain's end or a straggler sweep when lane 0 is idle, a split when not wanted */
        const int pm = merge_ok ? pp_merge_.load(std::memory_order_relaxed) : 0;
        bool hold0 = false;   /* lane 0 does not plan this round (the drain) */
        if (merge_ok && pm != mg_pm_last) {   /* a switch change: the counts so far, under the value that ran until now */
            if (mg_seen) mg_line(fmt(" so far (pp_merge %d -> %d)", mg_pm_last, pm).c_str());
            mg_pm_last = pm;
            mg_line_ns = rad_mono_ns();
        }
        if (merge_ok) {
            const uint64_t now = rad_mono_ns();
            bool bg_busy = false;
            for (int l = 0; l < n_lanes_; ++l) {
                if (lanes_.is_bg(l) && (busy[(size_t)l] || lanes_.inbox_n(l) > 0)) bg_busy = true;
                if (lane_relay[(size_t)l] && busy[(size_t)l]) bg_busy = true;   /* bit 32: a relay lane's BG step in flight */
                if (mg.on && l > 0 && l < lanes_.decode_n() && (pm & 32) && lanes_.inbox_n(l) > 0) bg_busy = true;
            }
            /* only time with work in flight counts (the pipeline idle between cells is neither state) */
            if (mg.last && mg.last_work) {
                const double dt = (double)(now - mg.last) / 1e9;
                mg.wall_s += dt;
                (mg.last_bg ? mg.bg_s : mg.nobg_s) += dt;
                if (mg.on) mg.merged_s += dt;
            }
            mg.last = now;
            mg.last_bg = bg_busy;
            /* bit 16: D11's hold acts while merged (LaneSet::bg_hold_ms), and a held newcomer is BG work: the lanes stay merged */
            lanes_.set_merge_hold_ms((pm & 16) && (pm & 3) ? mg_hold_ms : 0);
            bool held = false;
            if ((pm & 16) && (pm & 3) && !bg_busy) {
                held = lanes_.held_now() > 0;
                if (held) ++mg_held_rounds;
            }
            if (bg_busy || held) mg_bg_last = now;
            /* pp_merge_ms: merged that long after the last BG work */
            const uint64_t hyst = (uint64_t)pp_merge_ms_.load(std::memory_order_relaxed) * 1000000ull;
            const bool lingers = mg.on && !bg_busy && !held && hyst > 0 && mg_bg_last && now - mg_bg_last < hyst;
            if (lingers) ++mg_hyst_rounds;
            const bool want = (pm & 2) || ((pm & 1) && (bg_busy || held || lingers));
            bool dec_busy = false;   /* a decode lane has a step in flight */
            for (int l = 0; l < lanes_.decode_n(); ++l) dec_busy = dec_busy || busy[(size_t)l];
            if (want && !mg.on) {
                mg.on = true;
                mg.draining = true;
                mg.t_want = now;
                ++mg.merges;
                mg.note(lanes_.merge_into(0), true);   /* pick() names lane 0 from now; idle lanes' requests move at once */
            }
            if (mg.on && mg.draining) {
                if (dec_busy) {
                    hold0 = true;
                } else if (want) {
                    mg.note(lanes_.merge_into(0), true);
                    mg.draining = false;
                    mg.t_on = now;
                    ++mg.drains;
                    mg.drain_ms += (double)(now - mg.t_want) / 1e6;
                    pp_stamp("merge", 0, (int)lane_steps, mg.t_want, now);
                } else {   /* the BG work ended before the drain did: back to the split lanes */
                    mg.drain_ms += (double)(now - mg.t_want) / 1e6;
                    mg.note(lanes_.split_from(0), false);
                    mg.on = mg.draining = false;
                    ++mg.splits;
                    pp_stamp("split", 0, (int)lane_steps, mg.t_want, now);
                }
            } else if (mg.on && !want && !dec_busy) {
                mg.note(lanes_.split_from(0), false);
                mg.on = false;
                ++mg.splits;
                pp_stamp("split", 0, (int)lane_steps, mg.t_on ? mg.t_on : mg.t_want, now);
            } else if (mg.on && !busy[0] && lanes_.parked_live() > 0) {
                const LaneSet::MoveCounts c = lanes_.merge_into(0);   /* stragglers: a newcomer's piece came back */
                mg.sweeps += c.moved;
                mg.cards += c.cards;
            }
        }
        /* tp8 MERGE bit 32: while merged (drained), the parked lanes with nothing of their own take whole newcomers as BG lanes */
        const bool relay = merge_ok && (pm & 32) && mg.on && !mg.draining;
        lanes_.set_relay(relay);
        pp_relay_mask_ = 0;
        if (relay)
            for (int l = 1; l < lanes_.decode_n() && l < 32; ++l) pp_relay_mask_ |= 1u << l;
        std::vector<int> todo;
        if (!kv_hold) {
            for (int k = 0; k < n_lanes_; ++k) todo.push_back((next + k) % n_lanes_);
            if (decode_first) {
                const std::vector<int> was = todo;
                /* idle lanes with decode rows alone first (a busy lane is skipped below wherever it stands), so a change of
                 * order is a decode lane put ahead of a lane with prompt rows */
                std::stable_partition(todo.begin(), todo.end(), [&](int l) { return !busy[(size_t)l] && !lanes_.prefill_pending(l); });
                if (todo != was) ++reordered;
            }
        }
        /* tp8 MERGE: a parked decode lane plans nothing and takes nothing (no planner order names it) */
        auto parked = [&](int m) { return mg.on && m != 0 && m < lanes_.decode_n(); };
        /* bit 32: a parked lane plans only its inbox (a whole newcomer lent to it) and only while it holds nothing of its own */
        auto relay_ready = [&](int m) { return relay && parked(m) && lanes_.inbox_n(m) > 0 && lanes_.live(m) == 0; };
        for (size_t at = 0; at < todo.size(); ++at) {
            const int l = todo[at];
            if (busy[(size_t)l]) continue;
            if ((parked(l) && !relay_ready(l)) || (l == 0 && hold0)) continue;
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
                std::vector<int> order, relay_idle;
                for (int j = 1; j < n_lanes_; ++j) {
                    const int m = (l + j) % n_lanes_;
                    if (m == 0 && hold0) continue;   /* tp8 MERGE: it will not plan next */
                    if (parked(m)) {   /* tp8 MERGE: only as a relay lane (bit 32), after the idle BG lanes */
                        if (relay && !busy[(size_t)m] && lanes_.live(m) == 0) relay_idle.push_back(m);
                        continue;
                    }
                    if (!busy[(size_t)m]) { order.push_back(m); idle_after.push_back(m); }
                }
                for (int m : relay_idle) { order.push_back(m); idle_after.push_back(m); }
                for (const auto& f : fifo)
                    if (!parked(f.first) || (relay && lane_relay[(size_t)f.first])) order.push_back(f.first);
                lanes_.set_planners(std::move(order));
            }
            lane_select(l);
            /* tp8 dev M9b: the inboxes before this plan (a plan that lends grows one: then its step is not segmented) */
            std::vector<size_t> inbox0;
            if (lseg_mixed)
                for (int m2 = 0; m2 < n_lanes_; ++m2) inbox0.push_back(lanes_.inbox_n(m2));
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
            if (pp_tband_copy_) {   /* tp8 loop D46: the stretch this step's head sees (D32 / D45's: a BG lane busy or holding a newcomer) */
                bool bgb = false;
                for (int q = 0; q < n_lanes_; ++q)
                    if (lanes_.is_bg(q) && (busy[(size_t)q] || lanes_.inbox_n(q) > 0)) bgb = true;
                pp_tband_bg_ = bgb;
            }
            /* tp8 dev M9: a BG lane's step (M9b, RADIANCE_PP_LEAD_SEG_MIXED=1: also a decode lane's MIXED newcomer step that
             * carries and lends no piece): its head's first segment only, or its whole head where no head boundary is declared
             * or another pass holds the boundary ring (then its tail may still run in segments) */
            /* tp8 loop P28: a band step's head runs whole -- tp8 loop D36 (RADIANCE_PP_LEAD_SEG_BAND=1): in segments too, its last
             * one [s, h + k); the step's band is decided here once and handed to lane_head when its head runs whole after all */
            const int lsv = lsp_now(), lform = lsv & 7, lstride = std::max(1, (lsv >> 4) & 15);   /* tp8 loop D32 */
            const bool ltail_whole = (lsv & 8) != 0;   /* tp8 loop D40: + 8 = the tail whole */
            const int llflags = (lsv & 512) ? RAD_PP_LSEG_SP : 0;   /* tp8 loop D40: + 512 = the SP word on the head's segments */
            pp_band_off_ = (lsv & 256) != 0;   /* tp8 loop D36: + 256 = no band passes (pp_band_step reads it) */
            const bool relay_step = relay && parked(l);   /* tp8 MERGE bit 32: this step is a relay lane's BG step */
            /* D32's pp_lead_seg decides HOW a BG step's leader parts go (form / stride); MERGE's pp_merge bits 4 / 8 WHETHER (4: only with
             * the merge on, 8: never) */
            bool seg_step = lseg && lform != 0 && b->n_tok >= rf_rows && (lanes_.is_bg(l) || relay_step) && !(pm & 8) &&
                            (!(pm & 4) || (pm & 3) != 0);
            const int sband = seg_step ? pp_band_step(l, b) : -1;
            if (sband > 0 && !lseg_band) seg_step = false;
            if (seg_step && sband <= 0 && pp_tband_step(l, b) > 0) seg_step = false;   /* tp8 loop D39: a tail band step runs whole */
            if (seg_step && lform == 2 && !decode_live()) {   /* D32: no decode lane busy -> whole, as BEST */
                seg_step = false;
                ++lsp_whole;
            }
            /* tp8 loop D49b: + 1024 x N -- above N decoded sequences the parts run whole (form 0's program for this step) */
            const int lrows = (lsv >> 10) & 63;
            const bool load_whole = lseg && lform != 0 && lrows > 0 && decode_seqs() > lrows;
            if (seg_step && load_whole) {
                seg_step = false;
                ++lsp_load;
            }
            /* tp8 loop F81 (= QUEUE F75g, LSP's own position-order guard): a step in segments lets later heads / tails go between
             * its segments on rank 0's stream, so it may carry only rows no other step carries -- every prompt entry a WHOLE FRESH
             * prompt (context 0, its last row the prompt's last, no snapshot / cone replay). A piece of a longer prompt (the rest a
             * home lane lent, P23s's batched BG steps carry them: lq225 'all' step 43611 = 696 rows at context 30720 beside two
             * newcomers) runs its parts whole, in issue order = position order as every whole head. + 65536 = no guard (the F75 /
             * F81 form, the card's negative control). */
            const bool pos_guard = !(lsv & 65536);
            auto whole_fresh = [&]() {
                for (const StepEntry& e : lsched().plan().e)
                    if (e.is_prefill && (e.ctx_len != 0 || !e.produces_token || e.ckpt_restore >= 0 || e.cr_replay)) return false;
                return true;
            };
            if (seg_step && pos_guard && !whole_fresh()) {
                seg_step = false;
                ++lsp_pos;
            }
            if (lseg && lform != 0 && lseg_mixed && !seg_step && !load_whole && !(pm & 8) && (lform == 1 || !lanes_.is_bg(l)) &&
                b->n_tok >= rf_rows && b->phase == RAD_PHASE_MIXED && (!pos_guard || whole_fresh())) {
                seg_step = true;
                for (const StepEntry& e : lsched().plan().e) seg_step = seg_step && e.home < 0;
                for (int m2 = 0; m2 < n_lanes_; ++m2) seg_step = seg_step && lanes_.inbox_n(m2) <= inbox0[(size_t)m2];
                seg_step = seg_step && lanes_.inbox_n(l) == 0;
            }
            if (seg_step) {
                LSeg& s = lsg[(size_t)l];
                s = LSeg{};
                s.b = b;
                s.mixed = !lanes_.is_bg(l) && !relay_step;
                s.form = lform;
                s.stride = lstride;
                s.whole_tail = ltail_whole || lsp_kept(lseg_tb, lstride).empty();
                if (ltail_whole && !lsp_kept(lseg_tb, lstride).empty()) ++lsp_tw8;   /* tp8 loop D40 */
                s.lflags = llflags;
                s.ring = l % lseg_rings;
                s.band = std::max(0, sband);   /* tp8 loop D36 */
                const std::vector<int32_t> hb = lsp_kept(lseg_hb, lstride);
                if (!hb.empty() && !lseg_ring_busy(s.ring) && !lseg_tail_waits(s.ring)) {
                    if (lseg_rings > 1 && lseg_ring_busy(1 - s.ring)) ++lsp_both;
                    int a = 0;
                    for (int32_t x : hb) { s.parts.emplace_back(a, (int)x); a = x; }
                    s.parts.emplace_back(a, cfg_.pp_cut_h + s.band);   /* tp8 loop D36: a band step's head ends at h + k */
                    if (const int hs = lane_head_seg(l, b, s.parts[0].first, s.parts[0].second, true, false, s.band, s.lflags); hs < 0)
                        return fail_loop(hs);
                    if (s.lflags) ++lsp_spseg;   /* tp8 loop D40 */
                    if (rad_event_record(lev(l, 0), ranks_[0]->ctx.stream()) < 0) return fail_loop(RAD_E_DEVICE);
                    s.ship_s0 = pp_steps_shipped_;   /* tp8 loop P27 */
                    s.k = 1;
                    s.st = kLsHead;
                    ++lseg_parts;
                    lseg_last_ns = rad_mono_ns();
                } else {
                    pp_band_pre_ = sband;   /* tp8 loop D36: the band decided above (-1: none decided) */
                    if (const int hs = lane_head(l, b); hs < 0) return fail_loop(hs);
                    s.st = kLsHeadDone;
                }
                ++lseg_steps;
                if (s.mixed) ++lseg_msteps;
            } else {
                pp_band_pre_ = sband;   /* tp8 loop D36: the band decided above (-1: lane_head decides) */
                if (const int hs = lane_head(l, b); hs < 0) return fail_loop(hs);
            }
            gaps.host_head(b, p0, p1, rad_mono_ns());
            gaps.head(fifo.empty(), lane_steps);
            lane_merged[(size_t)l] = mg.on && l == 0 ? 1 : 0;   /* tp8 MERGE */
            lane_relay[(size_t)l] = relay_step ? 1 : 0;
            if (relay_step) ++mg.relay_steps;
            busy[(size_t)l] = 1;
            held_once[(size_t)l] = 0;
            fifo.emplace_back(l, b);
            next = (l + 1) % n_lanes_;
            /* the idle lanes it may have lent to plan next, in its order */
            todo.insert(todo.begin() + (ptrdiff_t)at + 1, idle_after.begin(), idle_after.end());
        }
        mg.last_work = !fifo.empty();   /* tp8 MERGE: the next round's time counts only with work in flight */
        if (fifo.empty()) {
            mg.last_dec = 0;   /* tp8 MERGE: an idle pipeline is no decode stop */
            bgr.last = 0;   /* tp8 loop D32: idle time is neither state */
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
        if (bgr_on) {   /* tp8 loop D32: this round's time by the state at its start */
            const uint64_t now = rad_mono_ns();
            bool bg_busy = false;
            for (int q = 0; q < n_lanes_; ++q)
                if ((lanes_.is_bg(q) && (busy[(size_t)q] || lanes_.inbox_n(q) > 0)) || (lane_relay[(size_t)q] && busy[(size_t)q]))
                    bg_busy = true;   /* tp8 MERGE: a relay lane's BG step counts as one */
            if (bgr.last) (bgr.last_bg ? bgr.bg_s : bgr.nobg_s) += (double)(now - bgr.last) / 1e9;
            bgr.last = now;
            bgr.last_bg = bg_busy;
            if (pp_ni_) pp_ni_->bg = bg_busy;   /* tp8 loop D45: the node-idle line's stretch (RADIANCE_PP_NODE_IDLE=2), the same state */
        }
        const bool two_in_flight = fifo.size() >= 2;
        /* 2. the oldest lane step's tail, its drafter, its commit (tp8 dev M4 ready-first: or a later one's, above) */
        size_t pick = 0;
        bool lseg_last = false;   /* tp8 dev M9: the pick is a BG step's last tail segment (with its sampler) */
        if (lseg && lseg_busy()) {
            /* tp8 dev M9: the BG step's next segment once its last one ran; its tail's first once its X2 landed */
            int adv = 0;
            /* tp8 loop D32 form 2: a step's tail that may go now (a decode tail whose X2 landed) goes before any BG segment due;
             * its lane's next head then goes at the next round's top, before that segment too */
            bool dfirst = false;
            for (const LSeg& q : lsg)
                if (q.st != kLsNone && q.form == 2) { dfirst = decode_ready(); break; }
            if (dfirst) ++lsp_first;
            for (int l = 0; l < n_lanes_ && adv == 0 && !dfirst; ++l) {
                LSeg& s = lsg[(size_t)l];
                if (s.st == kLsHead || (s.st == kLsTail && s.k + 1 < s.parts.size())) {
                    if (!lseg_seg_ok(s, l)) continue;
                    if (lseg_wait_ns && rad_mono_ns() - lseg_last_ns < lseg_wait_ns) continue;
                    const int s0 = s.parts[s.k].first, s1 = s.parts[s.k].second;
                    const bool last = s.st == kLsHead && s.k + 1 == s.parts.size();
                    const int rs = s.st == kLsHead ? lane_head_seg(l, s.b, s0, s1, false, last, s.band, s.lflags) : lane_tail_seg(l, s.b, s0, s1);
                    if (rs < 0) return fail_loop(rs);
                    if (s.st == kLsHead && s.lflags) ++lsp_spseg;   /* tp8 loop D40 */
                    ++lseg_parts;
                    lseg_last_ns = rad_mono_ns();
                    ++adv;
                    if (last && lseg_ship_) {
                        /* tp8 loop P27: the frame waits for this segment's run (its X1 sent) */
                        if (rad_event_record(lship_ev[(size_t)l], ranks_[0]->ctx.stream()) < 0) return fail_loop(RAD_E_DEVICE);
                        s.st = kLsHeadShip;
                        s.ship_ns = rad_mono_ns();
                        s.ship_s1 = pp_steps_shipped_;
                        s.parts.clear();
                        s.k = 0;
                    } else if (last) {
                        s.st = kLsHeadDone;
                        s.parts.clear();
                        s.k = 0;
                    } else {
                        if (rad_event_record(lev(l, (int64_t)s.k), ranks_[0]->ctx.stream()) < 0) return fail_loop(RAD_E_DEVICE);
                        ++s.k;
                    }
                } else if (s.st == kLsHeadShip) {
                    /* tp8 loop P27: ship the held frame once the head's last segment has run */
                    const uint64_t now = rad_mono_ns();
                    if (lship_wait_ns && now - s.ship_ns < lship_wait_ns) continue;
                    const int q = rad_event_query(lship_ev[(size_t)l]);
                    if (q < 0) return fail_loop(RAD_E_DEVICE);
                    if (q == 0) continue;
                    if (const int rs = pp_ship_release(l); rs < 0) return fail_loop(rs);
                    const uint64_t d = now - s.ship_ns;
                    ++lship_n;
                    lship_ns_sum += d;
                    lship_ns_max = std::max(lship_ns_max, d);
                    lship_ahead += pp_steps_shipped_ - 1 - s.ship_s0;
                    lship_late += pp_steps_shipped_ - 1 - s.ship_s1;
                    s.st = kLsHeadDone;
                    ++adv;
                } else if (s.st == kLsHeadDone && !s.whole_tail && x2_landed(l) && !lseg_ring_busy(s.ring)) {
                    if (s.form == 2 && !decode_live()) {   /* tp8 loop D32: no decode lane busy -> the tail whole */
                        s.whole_tail = true;
                        ++lsp_wtail;
                        continue;
                    }
                    int a = cfg_.pp_cut_t;
                    for (int32_t x : lsp_kept(lseg_tb, s.stride)) { s.parts.emplace_back(a, (int)x); a = x; }
                    s.parts.emplace_back(a, RAD_PP_SEG_END);
                    if (const int rs = lane_tail_seg(l, s.b, s.parts[0].first, s.parts[0].second); rs < 0) return fail_loop(rs);
                    if (lseg_rings > 1 && lseg_ring_busy(1 - s.ring)) ++lsp_both;
                    if (rad_event_record(lev(l, 0), ranks_[0]->ctx.stream()) < 0) return fail_loop(RAD_E_DEVICE);
                    s.k = 1;
                    s.st = kLsTail;
                    ++lseg_parts;
                    lseg_last_ns = rad_mono_ns();
                    ++adv;
                }
            }
            /* the tail to take now: the BG step's last segment (its previous one ran) or its whole tail (no tail boundary,
             * X2 landed); else the oldest other step whose X2 landed -- past a BG step only a decode step of its own
             * sequences (ready-first's rule); none: back to the top (plan, the next segment) */
            size_t want = SIZE_MAX;
            bool passed = false;
            for (size_t i = 0; i < fifo.size() && want == SIZE_MAX; ++i) {
                const int fl = fifo[i].first;
                const LSeg& s = lsg[(size_t)fl];
                if (s.st == kLsTail && s.k + 1 == s.parts.size()) {
                    if (lseg_seg_ok(s, fl)) { want = i; lseg_last = true; }
                    passed = true;
                    continue;
                }
                if (s.st == kLsHeadDone && s.whole_tail) {
                    if (x2_landed(fl) && !passed) want = i;
                    passed = true;
                    continue;
                }
                if (s.st != kLsNone) { passed = true; continue; }
                if (!x2_landed(fl)) {
                    if (!passed) break;   /* the oldest step: its tail first, once its X2 lands */
                    continue;
                }
                if (!passed || !carries_lent(fl)) want = i;
            }
            if (want == SIZE_MAX) {
                if (adv == 0) {
                    ++lseg_polls;
                    std::this_thread::sleep_for(std::chrono::microseconds(20));
                }
                continue;
            }
            pick = want;
            if (lsg[(size_t)fifo[pick].first].st == kLsNone) {
                ++lseg_between;
                /* ... and inside a BG part: between two of its segments (not while it waits for its X2) */
                for (const LSeg& s : lsg)
                    if (s.st == kLsHead || (s.st == kLsTail && s.k > 0)) { ++lseg_inside; break; }
            }
            if (lseg_last) ++lseg_parts;
        } else if (ready_first && fifo.size() >= 2 && fifo.front().second->n_tok >= rf_rows &&
                   ((pm & 64) || !x2_landed(fifo.front().first))) {
            /* tp8 MERGE bit 64 (decode tails first): a later decode-only step whose X2 landed goes before the long pass even when the
             * long pass's X2 landed too -- its tail is a few ms on rank 0's stream where the long pass's tail + collect hold it ~100 ms
             * (lq187: the merged lane's tail taken 100-600 ms after its head behind BG tails / heads). Only the order of tails changes,
             * as ready-first's: a decode step of its own sequences reads only committed KV. */
            bool waited = false;
            for (;;) {
                if (!(pm & 64) && x2_landed(fifo.front().first)) break;
                for (size_t i = 1; i < fifo.size() && !pick; ++i)
                    if (x2_landed(fifo[i].first) && !carries_lent(fifo[i].first)) pick = i;
                if (pick) { if (x2_landed(fifo.front().first)) ++mg_tails_first; break; }
                if ((pm & 64) && x2_landed(fifo.front().first)) break;
                if (stopping_.load(std::memory_order_acquire) || (pp_link_ && !pp_link_->alive())) break;
                waited = true;
                std::this_thread::sleep_for(std::chrono::microseconds(20));
            }
            if (waited) ++rf_waits;
            if (pick) ++rf_taken;
        }
        if (two_in_flight) ++rounds_two;
        const auto [lane, b] = fifo[pick];
        fifo.erase(fifo.begin() + (ptrdiff_t)pick);
        /* tp8 dev M9: a BG step's last tail segment goes out with its sampler (lane_tail_issue reads tail_seg0_) */
        tail_seg0_ = lseg_last ? lsg[(size_t)lane].parts.back().first : 0;
        if (lsg[(size_t)lane].st != kLsNone) lsg[(size_t)lane] = LSeg{};
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
                if (!nb) ++ia_why[kIaNext];
                if (nb) {
                    /* its rows' token ids, positions and slots on the card behind this step's sampler (and drafter) */
                    if (const int as = advance_all(ADV_TRUNK, 0, false); as < 0) return fail_loop(as);
                    pp_ship_ahead_ = true;   /* round 2: the frame says so (the follower waits for its X1 off the card) */
                    const int hs = lane_head(lane, nb);
                    pp_ship_ahead_ = false;
                    if (hs < 0) return fail_loop(hs);
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
             * sampler goes out at the top of the next round, with nothing released between. tp8 loop FINDINGS F43: not
             * for a step whose save patches a snapshot inside its chunk (lever TI): the patch copies the CONV window from
             * the ids the rank's LAST part mapped, and a model part issued ahead has other lanes' parts after it */
            auto ti_inside = [&](int l) {
                if (!ti_capable_) return false;
                for (const StepEntry& e : sched_of(l).plan().e)
                    if (e.ckpt_slot >= 0 && e.ckpt_tok >= 0 && e.ckpt_tok < e.n_tokens - 1) return true;
                return false;
            };
            if (!fifo.empty() && !m_out[(size_t)fifo.front().first] && !ti_inside(fifo.front().first)) {
                if (const int ms = lane_tail_model(fifo.front().first, fifo.front().second); ms < 0) return fail_loop(ms);
                m_out[(size_t)fifo.front().first] = 1;
            }
            if (const int fs = lane_tail_finish(lane, b, so, t_on); fs < 0) return fail_loop(fs);
            m_out[(size_t)lane] = s_out[(size_t)lane] = 0;
        }
        tail_seg0_ = 0;
        busy[(size_t)lane] = nb ? 1 : 0;
        if (bgr_on) {   /* tp8 loop D32: the decode tokens this commit emitted, by the state at the commit (MERGE's relay lane = BG) */
            const uint64_t now = rad_mono_ns();
            if (lanes_.is_bg(lane) || lane_relay[(size_t)lane]) {
                ++bgr.bg_commits;
            } else {
                int64_t tok = 0;
                const StepPlan& pl = sched_of(lane).plan();
                for (size_t i = 0; i < pl.e.size(); ++i) {
                    const StepEntry& e = pl.e[i];
                    if (e.home >= 0 || e.is_prefill || !e.produces_token) continue;
                    tok += 1 + (i < so.accepted.size() ? std::max<int32_t>(0, so.accepted[i]) : 0);
                }
                bool bg_now = false;
                for (int q = 0; q < n_lanes_; ++q)
                    if ((lanes_.is_bg(q) && (busy[(size_t)q] || lanes_.inbox_n(q) > 0)) || (lane_relay[(size_t)q] && busy[(size_t)q]))
                        bg_now = true;
                (bg_now ? bgr.tok_bg : bgr.tok_nobg) += tok;
                if (tok > 0) {
                    if (bgr.last_dec && bg_now && now - bgr.last_dec > 100000000ull) {
                        ++bgr.stops;
                        bgr.stop_ms += (double)(now - bgr.last_dec) / 1e6;
                    }
                    bgr.last_dec = now;
                }
            }
            if (now - bgr_line_ns >= bgr_every) {
                bgr_line(" so far");
                if (pp_tband_copy_) RAD_INFO("%s", tband_text(" so far").c_str());   /* tp8 loop D50: the tail band by cell */
                bgr_line_ns = now;
            }
        }
        if (pp_link_ && dch_n_ > 0) {   /* tp8 loop D35: the decode chain's host part, at the D32 line's period */
            const uint64_t now = rad_mono_ns();
            if (!dch_line_ns) dch_line_ns = now;
            if (now - dch_line_ns >= bgr_every) {
                RAD_INFO("two-node pipeline: decode chain on the leader host so far (tp8 loop D35): %s", dch_text().c_str());
                dch_line_ns = now;
            }
        }
        /* tp8 MERGE: the decode tokens this commit emitted (each decode row's sampled token + its accepted drafts), by the state
         * at the commit (a BG step in flight or not) and by the lane form the step ran in; a gap > 100 ms since the last decode
         * commit while a BG step is in flight is a full stop */
        if (mg_seen && merge_ok) {
            const uint64_t now = rad_mono_ns();
            if (lanes_.is_bg(lane) || lane_relay[(size_t)lane]) {
                ++mg.bg_commits;
            } else {
                int64_t tok = 0, rows = 0;
                const StepPlan& pl = sched_of(lane).plan();
                for (size_t i = 0; i < pl.e.size(); ++i) {
                    const StepEntry& e = pl.e[i];
                    if (e.home >= 0 || e.is_prefill || !e.produces_token) continue;
                    rows += e.n_tokens;
                    tok += 1 + (i < so.accepted.size() ? std::max<int32_t>(0, so.accepted[i]) : 0);
                }
                bool bg_now = false;
                for (int q = 0; q < n_lanes_; ++q)
                    if ((lanes_.is_bg(q) && (busy[(size_t)q] || lanes_.inbox_n(q) > 0)) || (lane_relay[(size_t)q] && busy[(size_t)q]))
                        bg_now = true;
                (bg_now ? mg.tok_bg : mg.tok_nobg) += tok;
                if (lane_merged[(size_t)lane]) { ++mg.steps_m; mg.tok_m += tok; mg.rows_m += rows; }
                else                           { ++mg.steps_s; mg.tok_s += tok; mg.rows_s += rows; }
                if (tok > 0) {
                    if (mg.last_dec && bg_now && now - mg.last_dec > 100000000ull) {
                        ++mg.stops;
                        mg.stop_ms += (double)(now - mg.last_dec) / 1e6;
                    }
                    mg.last_dec = now;
                }
            }
            if (now - mg_line_ns >= mg_line_every) {
                mg_line(" so far");
                mg_line_ns = now;
            }
        }
        lane_merged[(size_t)lane] = 0;
        lane_relay[(size_t)lane] = 0;
        gaps.host_tail(b, tt0, rad_mono_ns());
        gaps.tail(lane, b);
        if (++lane_steps == 1 || lane_steps % 4096 == 0) lanes_note(lane_steps, rounds_two);
        /* tp8 dev D10 round 2: the counts (run so far) once a minute, so a cell's window is the difference of two lines */
        if (ia) {
            const uint64_t now = rad_mono_ns();
            if (lane_steps == 1 || now - ia_line_ns >= 60000000000ull) {
                ia_line(lane_steps == 1 ? "" : " over the run so far");
                ia_line_ns = now;
            }
        }
        /* tp8 dev M9: the counts (run so far) once a minute, as D10's */
        if (lseg) {
            const uint64_t now = rad_mono_ns();
            if (now - lseg_line_ns >= bgr_every) {   /* tp8 loop D32: RADIANCE_PP_BGRATE_LINE_S's period (default 60 s, as before) */
                lseg_line(" so far");
                lseg_line_ns = now;
            }
        }
        if (lanes_.bg_hold_ms() > 0) {
            const uint64_t now = rad_mono_ns();
            if (now - hold_line_ns >= 60000000000ull) {
                hold_line();
                hold_line_ns = now;
            }
        }
        if (lanes_.bg_room() > 0 || bgst_seen) {   /* tp8 loop P23s: also once pp_bg_room was on */
            bgst_seen = true;
            const uint64_t now = rad_mono_ns();
            if (now - bgst_line_ns >= 60000000000ull) {
                bgst_line(" so far");
                bgst_line_ns = now;
            }
        }
        gaps.minute(lane_steps);
        if (pp_ni_) pp_ni_->minute();
        if (drafter_short_ > 0) {   /* tp8 loop D18: once a minute */
            const uint64_t now = rad_mono_ns();
            if (now - sb_line_ns >= 60000000000ull) {
                short_block_line(" so far");
                sb_line_ns = now;
            }
        }
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
    if (lseg) lseg_line("");
    if (bgr_on) bgr_line(" over the run");   /* tp8 loop D32 */
    if (bgr_on && pp_tband_copy_) RAD_INFO("%s", tband_text(" over the run").c_str());   /* tp8 loop D50 */
    if (pp_link_ && dch_n_ > 0)                /* tp8 loop D35 */
        RAD_INFO("two-node pipeline: decode chain on the leader host over the run (tp8 loop D35): %s", dch_text().c_str());
    if (pp_tband_on_) RAD_INFO("two-node pipeline: %lld lane steps ran as tail band passes (RADIANCE_PP_TAIL_BAND=1, tp8 loop D39); "
                               "%lld fresh BG steps ran W(h, t)'s under fn4 pp_tband 0; %lld of the tail band passes carried a later chunk "
                               "(pp_tband 2, D39b)", (long long)pp_tband_n_, (long long)pp_tband_off_n_, (long long)pp_tband_cont_n_);
    if (pp_tband_copy_) RAD_INFO("two-node pipeline: %lld decode lanes' steps ran as tail band passes while a BG step was in flight "
                                 "(pp_tband 3, tp8 loop D46); %lld such steps kept W(h, t)'s for a sequence whose band state the leader "
                                 "does not hold; %zu sequences held; %lld of the band passes had no BG step in flight (pp_tband 4, "
                                 "tp8 loop D50), %lld clean-stretch decode steps kept W(h, t)'s at a decode load <= N (4 + 1024 x N)",
                                 (long long)pp_tband_dec_n_, (long long)pp_tband_dec_skip_, pp_tcopy_seq_.size(),
                                 (long long)pp_tband_dec_clean_, (long long)pp_tband_dec_low_);
    if (pp_band_on_) RAD_INFO("two-node pipeline: %lld lane steps ran as band passes (RADIANCE_PP_BAND=1, tp8 loop P28); %lld fresh BG steps ran "
                              "W(h, t)'s under RADIANCE_PP_BAND_MIN_SEQS=%lld", (long long)pp_band_n_, (long long)g_band_low,
                              (long long)g_band_min_seqs);
    if (lseg_band) RAD_INFO("two-node pipeline: %lld band steps ran their head in segments (RADIANCE_PP_LEAD_SEG_BAND=1, tp8 loop D36)",
                            (long long)pp_band_seg_n_);
    if (pp_band_on_ && pp_band_off_n_ > 0)
        RAD_INFO("two-node pipeline: %lld fresh BG steps ran W(h, t)'s under pp_lead_seg + 256 (no band passes, tp8 loop D36)",
                 (long long)pp_band_off_n_);
    if (lanes_.bg_room() > 0 || bgst_seen) bgst_line(" over the run");
    if (mg_seen && merge_ok) mg_line(" over the run");   /* tp8 MERGE */
    if (ia) ia_line(" over the run");
    short_block_line(" over the run");   /* tp8 loop D18 */
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
    pp_set_pass(lane, RAD_PP_PART_TAIL, 0, 0, 0, 0, pp_tband_of(lane));   /* tp8 loop D39: from t - j on a tail band step */
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

/* tp8 dev M9: one segment [s0, s1) of a BG lane step's head part. The first takes what lane_head takes before its release
 * (the fresh states, the release with its switches, the prefix restores) and holds the step's frame; the last ships it
 * (the follower then issues its pass, which waits on its card for X1 as always) and ends with X1's send. */
int Engine::lane_head_seg(int lane, const RadBatch* b, int s0, int s1, bool first, bool last, int band, int lflags) {
    lane_select(lane);
    if (first) {
        clear_fresh_states();
        if (const int ps = prepare_release(b); ps < 0) return ps;
        if (!pp_trunk_pending_[lane]) {
            RAD_ERR("two-node pipeline: lane %d's step %d was released without its build", lane, b->step);
            return RAD_E_STATE;
        }
        pp_trunk_pending_[lane] = false;
        if ((size_t)lane < pp_tband_lane_.size()) pp_tband_lane_[(size_t)lane] = 0;   /* tp8 loop D39: a segmented step is no tail band step */
        pp_tband_note(lane, 0);   /* tp8 loop D39b: its prompts' later chunks run W(h, t)'s */
        RAD_TRY(restore_checkpoints());
        /* tp8 loop D36: a band step's held frame says so (flags bit 2: the follower's parts start at h + k) */
        pp_ship_band_ = band;
        const int hs = pp_ship_hold(b, lane);
        pp_ship_band_ = 0;
        if (hs < 0) return hs;
        if (band) {
            ++pp_band_n_;
            ++pp_band_seg_n_;
        }
    } else if (const int ps = prepare_release(b, /*switches=*/false); ps < 0) {
        return ps;
    }
    if (last && !lseg_ship_) RAD_TRY(pp_ship_release(lane));   /* tp8 loop P27: else the lane loop ships it after the run */
    /* tp8 loop D36: every segment of a band step is the band pass's (the plugin's X1 at h + k, the band layers in the last
     * segment [s, h + k)) */
    pp_set_pass(lane, RAD_PP_PART_HEAD, s0, s1, band, lflags);   /* tp8 loop D40: RAD_PP_LSEG_SP with pp_lead_seg + 512 */
    issue_what_.store(kIssueModel, std::memory_order_release);
    step_batch_.store(b, std::memory_order_release);
    const uint64_t t0 = pp_stamps_ ? rad_mono_ns() : 0;
    if (pp_ni_) pp_ni_->begin();      /* tp8 dev D8 */
    barrier_->arrive_and_wait();      /* release the ranks */
    barrier_->arrive_and_wait();      /* and wait for them */
    if (pp_ni_) pp_ni_->end();
    /* tp8 loop D32: a BG head segment's issue (host release to return; readers count 'hseg' apart from 'head') */
    if (pp_stamps_) pp_stamp("hseg", lane, b->step, t0, rad_mono_ns());
    pp_set_pass(lane, RAD_PP_PART_WHOLE);
    const int st = step_status_.load(std::memory_order_acquire);
    if (st < 0)
        RAD_ERR("lane %d step %d's head segment [%d, %d) failed: %s -- the engine cannot continue", lane, b->step, s0, s1,
                rad_strerror(st));
    return st;
}

/* tp8 dev M9: one model-part segment [s0, s1) of a BG lane step's tail (s0 == t: from X2's receive), not the last one (that
 * one goes with the sampler: lane_tail_issue from tail_seg0_) */
int Engine::lane_tail_seg(int lane, const RadBatch* b, int s0, int s1) {
    lane_select(lane);
    if (const int ps = prepare_release(b, /*switches=*/false); ps < 0) return ps;
    pp_set_pass(lane, RAD_PP_PART_TAIL, s0, s1, 0, 0, pp_tband_of(lane));
    issue_what_.store(kIssueModel, std::memory_order_release);
    step_batch_.store(b, std::memory_order_release);
    const uint64_t t0 = pp_stamps_ ? rad_mono_ns() : 0;
    if (pp_ni_) pp_ni_->begin();      /* tp8 dev D8 */
    barrier_->arrive_and_wait();      /* release the ranks */
    barrier_->arrive_and_wait();      /* and wait for them */
    if (pp_ni_) pp_ni_->end();
    if (pp_stamps_) pp_stamp("tseg", lane, b->step, t0, rad_mono_ns());   /* tp8 loop D32 */
    pp_set_pass(lane, RAD_PP_PART_WHOLE);
    const int st = step_status_.load(std::memory_order_acquire);
    if (st < 0)
        RAD_ERR("lane %d step %d's tail segment [%d, %d) failed: %s -- the engine cannot continue", lane, b->step, s0, s1,
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
        /* tp8 dev M9: the last of a BG step's tail segments (from boundary tail_seg0_ to the end) with its sampler */
        pp_set_pass(lane, RAD_PP_PART_TAIL, tail_seg0_, tail_seg0_ > 0 ? RAD_PP_SEG_END : 0, 0, 0, pp_tband_of(lane));   /* tp8 loop D39 */
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
    const uint64_t dcw0 = rad_mono_ns();   /* tp8 loop D35: the collect's wait */
    int st = embedding_ ? collect_pooled(b, so) : collect_step(b, so);
    if (t_on) phase_mark(PH_COLLECT);
    if (st < 0) return st;
    if (pp_link_ && b->phase == RAD_PHASE_DECODE && lane >= 0 && lane < RAD_PP_MAX_LANES) {
        const uint64_t now = rad_mono_ns();
        dch_wait_ms_ += (double)(now - dcw0) / 1e6;
        ++dch_wait_n_;
        dch_collect_ns_[lane] = now;
    }
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
