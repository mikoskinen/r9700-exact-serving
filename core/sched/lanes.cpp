/* lanes.cpp -- tp8 S5: a two-node pipeline's lanes (lanes.h): the set, the routing of requests, and
 * the scheduler's cross-lane halves (a newcomer's chunk split into one piece a lane). */
#include "lanes.h"

#include "server/sink.h"

#include <algorithm>
#include <iterator>
#include <chrono>
#include <cstdlib>
#include <cstring>

namespace rad {

namespace {
int64_t mono_us() {
    return (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  /* namespace */

/* ================================================================== the set */
void LaneSet::init(std::vector<Scheduler*> lanes) {
    lanes_ = std::move(lanes);
    mu_ = lanes_.empty() ? nullptr : &lanes_[0]->mu_;
    inbox_.assign(lanes_.size(), {});
    order_.clear();
    const char* rr = std::getenv("RADIANCE_PP_PICK");
    pick_rr_ = rr && !std::strcmp(rr, "rr");
    const char* on = std::getenv("RADIANCE_PP_SPLIT");
    split_on_ = !(on && !std::strcmp(on, "0"));
    if (const char* m = std::getenv("RADIANCE_PP_SPLIT_MIN"); m && *m) {
        const long long v = std::strtoll(m, nullptr, 10);
        if (v > 0) split_min_ = (int64_t)v;
    }
}

int LaneSet::pick() const {
    if (lanes_.size() <= 1) return 0;
    std::lock_guard<std::mutex> lk(*mu_);
    struct Load { int64_t live = 0, drafted = 0, free_cards = 0; };
    std::vector<Load> ld(lanes_.size());
    for (size_t l = 0; l < lanes_.size(); ++l) {
        const Scheduler& s = *lanes_[l];
        ld[l].live = (int64_t)s.running_.size() + (int64_t)s.waiting_.size();
        for (int32_t slot : s.running_) ld[l].drafted += s.reqs_[(size_t)slot]->n_draft;
        ld[l].free_cards = (int64_t)s.free_cards_.size();
    }
    auto better = [&](size_t a, size_t b) {   /* a strictly better than b */
        if (ld[a].live != ld[b].live) return ld[a].live < ld[b].live;
        if (ld[a].drafted != ld[b].drafted) return ld[a].drafted < ld[b].drafted;
        return ld[a].free_cards > ld[b].free_cards;
    };
    size_t best = 0;
    for (size_t l = 1; l < lanes_.size(); ++l)
        if (better(l, best)) best = l;
    /* RADIANCE_PP_PICK=rr: equally loaded lanes take new requests in turn (a request alone on an idle
     * engine goes to lane 0, 1, 2, ... in turn), so an identity run of requests one at a time covers
     * every lane. Default: the lowest index. */
    if (pick_rr_) {
        std::vector<size_t> tie;
        for (size_t l = 0; l < lanes_.size(); ++l)
            if (!better(best, l) && !better(l, best)) tie.push_back(l);
        best = tie[(size_t)(rr_.fetch_add(1, std::memory_order_relaxed) % (uint64_t)tie.size())];
    }
    return (int)best;
}

Scheduler* LaneSet::owner(uint64_t id) const {
    if (lanes_.empty()) return nullptr;
    std::lock_guard<std::mutex> lk(*mu_);
    for (Scheduler* s : lanes_)
        if (s->by_id_.count(id)) return s;
    return nullptr;
}

SchedMetrics LaneSet::observed_metrics() const {
    SchedMetrics m = lanes_.empty() ? SchedMetrics{} : lanes_[0]->observed_metrics();
    for (size_t l = 1; l < lanes_.size(); ++l) {
        const SchedMetrics o = lanes_[l]->observed_metrics();
        m.queue_depth += o.queue_depth;
        m.running += o.running;
        m.preempted_waiting += o.preempted_waiting;
        m.admitted += o.admitted;
        m.finished += o.finished;
        m.failed += o.failed;
        m.cancelled += o.cancelled;
        m.preemptions += o.preemptions;
        m.preempt_forced += o.preempt_forced;
        m.steps += o.steps;
        m.steps_prefill += o.steps_prefill;
        m.steps_mixed += o.steps_mixed;
        m.steps_decode += o.steps_decode;
        m.steps_held += o.steps_held;
        m.decode_rows_unspeculated += o.decode_rows_unspeculated;
        m.steps_ahead += o.steps_ahead;
        m.prefill_chunks_capped += o.prefill_chunks_capped;
        m.step_ns += o.step_ns;
        m.step_ns_decode += o.step_ns_decode;
        m.prompt_tokens += o.prompt_tokens;
        m.checkpoints_written += o.checkpoints_written;
        m.prefill_tokens += o.prefill_tokens;
        m.decode_tokens += o.decode_tokens;
        m.prefill_tps += o.prefill_tps;
        m.decode_tps += o.decode_tps;
        m.draft_tokens += o.draft_tokens;
        m.draft_accepted += o.draft_accepted;
        m.draft_refills += o.draft_refills;
        m.draft_refill_tokens += o.draft_refill_tokens;
        for (size_t d = 0; d < std::size(m.draft_pos_reached); ++d) {
            m.draft_pos_reached[d] += o.draft_pos_reached[d];
            m.draft_pos_accepted[d] += o.draft_pos_accepted[d];
        }
        for (size_t d = 0; d < std::size(m.draft_verify_depth); ++d) m.draft_verify_depth[d] += o.draft_verify_depth[d];
        m.lookup_proposals += o.lookup_proposals;
        m.lookup_held += o.lookup_held;
        m.lookup_held_drafts += o.lookup_held_drafts;
        m.lookup_held_hits += o.lookup_held_hits;
        if (o.step > m.step) m.step = o.step;
    }
    if (m.draft_tokens > 0) m.draft_acceptance = (float)((double)m.draft_accepted / (double)m.draft_tokens);
    return m;
}

int LaneSet::observed_requests(Scheduler::ReqStat* out, int max) const {
    int n = 0;
    for (Scheduler* s : lanes_) {
        if (n >= max) break;
        n += s->observed_requests(out + n, max - n);
    }
    return n;
}

int64_t LaneSet::inbox_rows_locked(int lane) const {
    int64_t n = 0;
    for (const Piece& p : inbox_[(size_t)lane]) n += p.e.n_tokens;
    return n;
}

void LaneSet::take_locked(int lane, std::vector<Piece>* out) {
    out->clear();
    out->swap(inbox_[(size_t)lane]);
}

/* ================================================================== the scheduler's halves */
/* At step()'s start: a sequence with a piece still waiting in another lane's inbox, or whose first
 * token is to come from one, is left out of this step's plan (out of running_ while step() plans; put
 * back by lane_plan_end); one whose pieces are in flight plans its next chunk behind them (its
 * n_computed seen past its lent rows while step() plans, so the chunk starts where they end). */
void Scheduler::lane_plan_begin() {
    lane_bump_.clear();
    lane_out_.clear();
    if (!lanes_ || n_lanes_ <= 1) return;
    for (size_t i = 0; i < running_.size();) {
        const int32_t slot = running_[i];
        SchedReq& s = *reqs_[(size_t)slot];
        if (s.req && (s.lent_pending > 0 || s.lent_tok || s.lent_hold)) {
            lane_out_.push_back(slot);
            running_.erase(running_.begin() + (ptrdiff_t)i);
            continue;
        }
        if (s.req && s.lent > 0) {
            s.req->n_computed += s.lent;
            lane_bump_.push_back({ slot, s.lent });
        }
        ++i;
    }
}

void Scheduler::lane_plan_end() {
    for (const auto& [slot, rows] : lane_bump_) {
        SchedReq& s = *reqs_[(size_t)slot];
        if (s.req) s.req->n_computed -= rows;
    }
    lane_bump_.clear();
    for (int32_t slot : lane_out_)
        if (reqs_[(size_t)slot] && reqs_[(size_t)slot]->req && reqs_[(size_t)slot]->running)
            queue_insert(running_, slot);
    lane_out_.clear();
}

/* The budget this lane's own plan may take: the pieces lent to it run in this step whatever it
 * plans, so they come first; a step carrying another lane's piece is a shared step (no span). */
int64_t Scheduler::lane_reserve(int64_t budget) {
    if (!lanes_ || n_lanes_ <= 1) return budget;
    const int64_t rows = lanes_->inbox_rows_locked(lane_);
    lane_reserved_ = rows;
    if (rows <= 0) return budget;
    /* a chunk wider than the shared budget is lent only to a lane with nothing else: its step is that one sequence */
    if (geo_.shared_tok > 0 && rows > geo_.shared_tok) return 0;
    if (cfg_.prefill_span > 1) {
        span_now_ = false;
        if (budget > mixed_now_ && mixed_now_ > 0) budget = mixed_now_;
    }
    return budget > rows ? budget - rows : 0;
}

/* The pieces other lanes lent this one, appended to its plan as prefill entries of their home lane. */
void Scheduler::lane_take_inbox() {
    if (!lanes_ || n_lanes_ <= 1) return;
    std::vector<LaneSet::Piece> got;
    lanes_->take_locked(lane_, &got);
    {
        /* Never: a lender lends only into a lane's room (lane_split_own), and the lane set it aside at the
         * step's start (lane_reserve). Named loudly should either rule break. */
        int64_t rows = 0;
        for (const LaneSet::Piece& p : got) rows += p.e.n_tokens;
        if (rows != lane_reserved_ || plan_.n_tok + rows > cfg_.max_tok)
            RAD_ERR("lane %d: %lld rows of other lanes' pieces (%zu) beside %lld of its own, %lld reserved at the step's "
                    "start, step budget %lld", lane_, (long long)rows, got.size(), (long long)plan_.n_tok,
                    (long long)lane_reserved_, (long long)cfg_.max_tok);
    }
    for (LaneSet::Piece& p : got) {
        if (Scheduler* home = lanes_->lane(p.home)) {
            auto it = home->by_id_.find(p.e.seq);
            if (it != home->by_id_.end() && home->reqs_[(size_t)it->second]) {
                SchedReq& s = *home->reqs_[(size_t)it->second];
                if (s.lent_pending > 0) --s.lent_pending;
                if (s.lent_next >= 0 && p.e.ctx_len != s.lent_next)
                    lanes_->note_broken_locked(fmt("lane %d took the piece [%d, %d) of request %llu (lane %d's) while the "
                                                   "piece at %lld was not yet taken: the pieces would run out of order",
                                                   lane_, (int)p.e.ctx_len, (int)(p.e.ctx_len + p.e.n_tokens),
                                                   (unsigned long long)p.e.seq, p.home, (long long)s.lent_next));
                s.lent_next = s.lent_pending > 0 ? (int64_t)p.e.ctx_len + p.e.n_tokens : -1;
            }
        }
        p.e.home = (int8_t)p.home;
        p.e.slot = -1;
        plan_.e.push_back(p.e);
        plan_.n_tok += p.e.n_tokens;
    }
}

/* A build that failed: the pieces other lanes lent this step wait in its inbox for its next one. */
void Scheduler::lane_return_pieces() {
    for (const StepEntry& e : plan_.e) {
        if (e.home < 0) continue;
        LaneSet::Piece p;
        p.home = e.home;
        p.e = e;
        p.e.home = -1;
        lanes_->lend_locked(lane_, p, /*count=*/false);
        if (Scheduler* home = lanes_->lane(e.home)) {
            auto it = home->by_id_.find(e.seq);
            if (it != home->by_id_.end() && home->reqs_[(size_t)it->second]) {
                SchedReq& s = *home->reqs_[(size_t)it->second];
                ++s.lent_pending;
                if (s.lent_next < 0 || e.ctx_len < s.lent_next) s.lent_next = e.ctx_len;
            }
        }
    }
}

/* The chunk an idle engine plans from position n (bi_chunk_plan's alone piece: the span where the request would span
 * alone, the interval grid, the tail, the prompt's end; lever TI's tail inside the chunk where the whole chunk fits a
 * step), with r's n_computed seen at n while it plans. */
int64_t Scheduler::lane_alone_chunk(const SchedReq& s, Request& r, int64_t target, int64_t n, int64_t* ckpt) {
    const int64_t had = r.n_computed;
    r.n_computed = n;
    const bool keep = span_now_;
    span_now_ = bi_chunk_ != 2 && cfg_.prefill_span > 1 &&
                (span_min_ == 0 || std::max<int64_t>((int64_t)r.prompt.size(), target) >= span_min_);
    const int64_t keep_ti = ti_limit_;
    ti_limit_ = cfg_.max_tok;
    int64_t ck = -1;
    const int64_t want = plan_chunk(n, chunk_end(r, target), cfg_.max_tok, &ck, r.checkpoint_at, &r);
    ti_limit_ = keep_ti;
    span_now_ = keep;
    psk_ckpt(s, target, &ck);   /* lever PSK design 2: a snapshot at a target only */
    r.n_computed = had;
    if (ckpt) *ckpt = ck;
    return want;
}

/* The newcomer's chunks over the lanes (tp8 S5; the one-process grid, tp8 M56): a prefill entry of this lane that is
 * exactly the chunk an idle engine plans from its start keeps its rows here, and the chunks an idle engine plans after
 * it -- whole, at the boundaries one process cuts the prompt at -- go one a lane to the lanes that plan next
 * (LaneSet::set_planners), in queue order, the one reaching the prompt's end taking its first token. So a cold prompt's
 * chunks alternate over the lanes and every boundary is one process's: no piece invents a cut (S4 hardware: cold
 * 16k / 64k / 130k texts differed with pieces cut inside one process's chunks). Not lent: a chunk past the prompt's end,
 * a cone replay, an embedding, a media or a scored request, a drafted entry; a chunk wider than the shared budget only
 * to a lane with nothing else to run (a step wider than it is one sequence); a chunk whose KV does not fit without
 * preempting. Where the leader's head writes state (LaneSet::head_state: a cut W(h > 0, t)), a chunk that takes a
 * snapshot is never followed by its sequence's next chunk before its commit (its home's own: nothing lent behind it;
 * a lent one: the last lent, its home holds the sequence until it commits). */
void Scheduler::lane_split_own() {
    if (!lanes_ || n_lanes_ <= 1 || !lanes_->split_on()) return;
    const std::vector<int>& order = lanes_->planners_locked();
    if (order.empty()) return;
    const bool hold = lanes_->head_state();
    for (StepEntry& e : plan_.e) {
        if (e.home >= 0 || !e.is_prefill || e.produces_token || e.slot < 0 || (size_t)e.slot >= reqs_.size()) continue;
        if (e.cr_replay || e.n_spec != 0) continue;
        if (hold && e.ckpt_slot >= 0 && e.ckpt_tok >= 0) continue;
        SchedReq& s = *reqs_[(size_t)e.slot];
        if (!s.req) continue;
        Request& r = *s.req;
        if (r.embed || r.media || r.token_scores || r.n_draft != 0 || s.lent_hold) continue;
        const int64_t target = s.seq_len();
        /* only an entry on the one-process grid: its own chunk is the one an idle engine plans from its start */
        if (lane_alone_chunk(s, r, target, e.ctx_len, nullptr) != e.n_tokens) continue;
        /* The committed end, below which no piece in flight reads: this step's bump undone. */
        int64_t real = r.n_computed;
        for (const auto& [slot, rows] : lane_bump_)
            if (slot == e.slot) real -= rows;
        int64_t p = (int64_t)e.ctx_len + e.n_tokens, rows = 0;
        bool tok = false, any = false;
        for (size_t j = 0; j < order.size() && p < target; ++j) {
            const int to = order[j];
            int64_t ck = -1;
            const int64_t n = lane_alone_chunk(s, r, target, p, &ck);
            if (n <= 0) break;
            const Scheduler* b = lanes_->lane(to);
            const bool wide = geo_.shared_tok > 0 && n > geo_.shared_tok;
            if (wide && (!b || !b->running_.empty() || !b->waiting_.empty() || lanes_->inbox_n_locked(to) > 0)) break;
            /* the borrower's entries: its own cards' beside the inbox (the S5 review's rule) */
            if (lanes_->inbox_rows_locked(to) + n > cfg_.max_tok ||
                (int64_t)lanes_->inbox_n_locked(to) + 1 + (cfg_.max_seqs + n_lanes_ - 1) / n_lanes_ > cfg_.max_seqs)
                break;
            if (kv_->ensure_from(r.id, p + n, p) < 0) break;   /* never preempt for a lent chunk */
            LaneSet::Piece pc;
            pc.home = lane_;
            pc.e = e;
            pc.e.ctx_len = (int32_t)p;
            pc.e.n_tokens = (int32_t)n;
            pc.e.produces_token = p + n >= target;
            pc.e.dev = false;
            pc.e.ckpt_restore = -1;
            pc.e.ckpt_slot = -1;
            pc.e.ckpt_tok = -1;
            pc.e.n_accepted = 1;
            /* its snapshot, reserved as the home's plan reserves one; commit_checkpoint at its commit (lane_piece_done) */
            if (ck > 0) reserve_ckpt(pc.e, r, ck, (int32_t)(ck - p - 1));
            lanes_->lend_locked(to, pc);
            s.lent += (int32_t)n;
            ++s.lent_pending;
            rows += n;
            any = true;
            p += n;
            if (pc.e.produces_token) { tok = true; break; }
            if (hold && pc.e.ckpt_slot >= 0) { s.lent_hold = true; break; }
        }
        if (!any) continue;
        lanes_->note_lent_locked(rows, tok);
        s.lent_tok = s.lent_tok || tok;
        s.lent_next = (int64_t)e.ctx_len + e.n_tokens;
        kv_->set_keep_floor(r.id, real);
    }
}

/* One of this lane's sequences' pieces, committed in the lane that ran it (its StepResult row
 * `i`): its rows join the prompt, and a piece that produced the first token hands it over as this
 * lane's own commit would. The lock is held. */
void Scheduler::lane_piece_done(const StepEntry& e, int i, const StepResult& res) {
    if (lanes_) lanes_->note_done_locked();
    auto it = by_id_.find(e.seq);
    if (it == by_id_.end() || !reqs_[(size_t)it->second]) return;
    SchedReq& s = *reqs_[(size_t)it->second];
    if (!s.req) return;
    Request& r = *s.req;
    s.lent = std::max<int32_t>(0, s.lent - e.n_tokens);
    if (e.produces_token) s.lent_tok = false;
    /* tp8 M56: a lent chunk's snapshot (lane_split_own), readable from its commit on, as commit() marks the home's */
    if (e.ckpt_slot >= 0 && e.ckpt_tok >= 0) {
        if (pc_ && r.state == ReqState::Running) pc_->commit_checkpoint(r, (int64_t)e.ctx_len + e.ckpt_tok + 1);
        s.lent_hold = false;
    }
    if (r.state != ReqState::Running) {
        /* Cancelled or failed while the piece ran: its blocks waited for it (in_step). */
        if (s.lent == 0) kv_->set_keep_floor(r.id, -1);
        sweep_deferred();
        return;
    }
    r.n_computed += e.n_tokens;
    kv_->set_keep_floor(r.id, s.lent > 0 ? r.n_computed : -1);
    kv_->trim_window(r.id, r.n_computed);
    server::sink_prompt_progress(r.sink, r.n_computed, (int32_t)r.prompt.size(), r.n_cached);
    if (r.n_computed >= (int64_t)r.prompt.size()) server::sink_prompt_stats(r.sink, (int32_t)r.prompt.size(), r.n_cached);
    if (!e.produces_token || !res.token) return;
    /* As commit()'s prefill entry that produces a token, then its finish. */
    const int32_t W = conv_window();
    bool overflowed = false, lp_failed = false;
    const bool eos = res.eos && res.eos[i] && !r.ignore_eos;
    if (!s.first_tok_us) s.first_tok_us = mono_us();
    r.output.push_back(res.token[i]);
    const TokenLogprobs* lp = res.lp && res.first_row && res.first_row[i] >= 0 && r.sp.n_probs >= 0
                                  ? &res.lp[res.first_row[i]] : nullptr;
    if (lp && lp->token < 0) lp_failed = true;
    else overflowed = !server::sink_push(r.sink, res.token[i], lp);
    s.conv_cursor = (int32_t)((s.conv_cursor + 1) % (W ? W : 1));
    s.n_draft = 0;
    r.n_draft = 0;
    if (overflowed) close_request(s, ReqState::Failed, "sink_overflow", false);
    else if (lp_failed)
        close_request(s, ReqState::Failed, "logprobs", false, true, RAD_E_STATE,
                      "the logprobs of a token it generated could not be computed");
    else if (server::sink_cancelled(r.sink)) close_request(s, ReqState::Cancelled, "cancelled", false);
    else if (r.max_tokens > 0 && (int64_t)r.output.size() >= r.max_tokens) {
        r.output.resize((size_t)r.max_tokens);
        close_request(s, ReqState::Finished, "length", false);
    } else if (ctx_limit_ > 0 && s.seq_len() >= ctx_limit_) close_request(s, ReqState::Finished, "length", false);
    else if (eos) close_request(s, ReqState::Finished, "stop", false);
    sweep_deferred();
}

}  /* namespace rad */
