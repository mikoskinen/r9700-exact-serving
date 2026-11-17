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

bool LaneSet::unlend_locked(int to, int home, uint64_t seq, int32_t ctx_len) {
    if (to < 0 || to >= (int)inbox_.size()) return false;
    std::vector<Piece>& in = inbox_[(size_t)to];
    for (size_t k = 0; k < in.size(); ++k)
        if (in[k].home == home && in[k].e.seq == seq && in[k].e.ctx_len == ctx_len) {
            in.erase(in.begin() + (ptrdiff_t)k);
            --lent_;
            return true;
        }
    return false;
}

/* ================================================================== the scheduler's halves */
/* At step()'s start: a sequence with a piece still waiting in another lane's inbox, or whose first
 * token is to come from one, is left out of this step's plan (out of running_ while step() plans; put
 * back by lane_plan_end); one whose pieces are in flight plans its next chunk behind them (its
 * n_computed seen past its lent rows while step() plans, so the chunk starts where they end). */
void Scheduler::lane_plan_begin() {
    lane_bump_.clear();
    lane_out_.clear();
    lane_lent_now_.clear();
    if (!lanes_ || n_lanes_ <= 1) return;
    for (size_t i = 0; i < running_.size();) {
        const int32_t slot = running_[i];
        SchedReq& s = *reqs_[(size_t)slot];
        if (s.req && (s.lent_pending > 0 || s.lent_tok)) {
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

/* Whether KV blocks this lane cannot preempt are held by running requests: another lane's, or this
 * lane's own left out of this step's plan (lane_plan_begin: pieces of them run elsewhere). Never on
 * one lane. The lock is held. */
bool Scheduler::lane_others_hold() const {
    if (!lanes_ || n_lanes_ <= 1) return false;
    if (!lane_out_.empty()) return true;
    for (int l = 0; l < lanes_->n(); ++l) {
        const Scheduler* o = lanes_->lane(l);
        if (o && o != this && !o->running_.empty()) return true;
    }
    return false;
}

/* The budget this lane's own plan may take: the pieces lent to it run in this step whatever it
 * plans, so they come first; a step carrying another lane's piece is a shared step (no span). */
int64_t Scheduler::lane_reserve(int64_t budget) {
    if (!lanes_ || n_lanes_ <= 1) return budget;
    const int64_t rows = lanes_->inbox_rows_locked(lane_);
    lane_reserved_ = rows;
    if (rows <= 0) return budget;
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

/* This lane's own prefill entries, each cut into one piece a lane in queue order: the first stays
 * here, the others go to the lanes that plan next (LaneSet::set_planners), the last of them taking
 * the entry's token if it produces one. Not cut: a chunk shorter than two pieces, one with a
 * checkpoint, a restore or a cone replay on it, an embedding, a media or a scored request. The
 * pieces after the first keep to the quantum grid and the minimum; the last takes the remainder. */
void Scheduler::lane_split_own() {
    if (!lanes_ || n_lanes_ <= 1 || !lanes_->split_on()) return;
    const std::vector<int>& order = lanes_->planners_locked();
    if (order.empty()) return;
    const int64_t Q = std::max<int64_t>(geo_.quantum, 1);
    const int64_t minp = (std::max<int64_t>(lanes_->split_min(), Q) + Q - 1) / Q * Q;
    /* A borrower's rows: the shared budget under a span (a step carrying another lane's piece is a
     * shared step), else the step's; its entries: its own cards' beside the inbox. */
    const int64_t rows_cap = cfg_.prefill_span > 1 && geo_.shared_tok > 0 ? std::min<int64_t>(cfg_.max_tok, geo_.shared_tok)
                                                                         : cfg_.max_tok;
    const int64_t lane_cards = (cfg_.max_seqs + n_lanes_ - 1) / n_lanes_;
    for (StepEntry& e : plan_.e) {
        if (e.home >= 0 || !e.is_prefill || e.slot < 0 || (size_t)e.slot >= reqs_.size()) continue;
        if (e.ckpt_slot >= 0 || e.ckpt_restore >= 0 || e.cr_replay || e.n_spec != 0) continue;
        SchedReq& s = *reqs_[(size_t)e.slot];
        if (!s.req) continue;
        const Request& r = *s.req;
        if (r.embed || r.media || r.token_scores || r.n_draft != 0) continue;
        const int64_t n = e.n_tokens;
        /* The most pieces whose every borrower has room for its piece beside what it was lent already:
         * a lane's step holds max_tok rows, its inbox first (lane_reserve), and a step carrying another
         * lane's piece is a shared step, so no piece is wider than the shared budget either. */
        int64_t k = std::min<int64_t>((int64_t)order.size() + 1, std::min<int64_t>(n_lanes_, n / minp));
        int64_t base = 0, last = 0;
        for (; k >= 2; --k) {
            base = n / k / Q * Q;
            if (base < minp) continue;
            last = n - (k - 1) * base;
            if (geo_.shared_tok > 0 && last > geo_.shared_tok) continue;
            bool room = true;
            for (int64_t j = 1; j < k && room; ++j) {
                const int to = order[(size_t)(j - 1)];
                /* tp8 S5 review: the borrower's step is a shared step (lane_reserve), and a step wider than
                 * the shared budget must be one sequence (RadBuildCtx::max_mixed_tok); its own plan counts
                 * its entries without the inbox, so the inbox leaves room for every card of the lane. */
                room = lanes_->inbox_rows_locked(to) + (j == k - 1 ? last : base) <= rows_cap &&
                       (int64_t)lanes_->inbox_n_locked(to) + 1 + lane_cards <= cfg_.max_seqs;
            }
            if (room) break;
        }
        if (k < 2) continue;
        const bool tok = e.produces_token;
        const int32_t c0 = e.ctx_len;
        /* The committed end, below which no piece in flight reads: this step's bump undone. */
        int64_t real = r.n_computed;
        for (const auto& [slot, rows] : lane_bump_)
            if (slot == e.slot) real -= rows;
        /* tp8 S5 review: what this cut changes, for lane_unsplit_own should the step's build fail */
        LaneLent undo;
        undo.slot = e.slot;
        undo.seq = e.seq;
        undo.lent = s.lent;
        undo.lent_pending = s.lent_pending;
        undo.lent_tok = s.lent_tok;
        undo.lent_next = s.lent_next;
        undo.rows = n - base;
        undo.tok = tok;
        for (int64_t j = 1; j < k; ++j) {
            LaneSet::Piece p;
            p.home = lane_;
            p.e = e;
            p.e.ctx_len = (int32_t)(c0 + j * base);
            p.e.n_tokens = (int32_t)(j == k - 1 ? last : base);
            p.e.produces_token = tok && j == k - 1;
            p.e.dev = false;
            lanes_->lend_locked(order[(size_t)(j - 1)], p);
            undo.to.emplace_back(order[(size_t)(j - 1)], p.e.ctx_len);
            s.lent += p.e.n_tokens;
            ++s.lent_pending;
        }
        lane_lent_now_.push_back(std::move(undo));
        lanes_->note_lent_locked(n - base, tok);
        s.lent_tok = s.lent_tok || tok;
        s.lent_next = (int64_t)c0 + base;
        e.n_tokens = (int32_t)base;
        e.produces_token = false;
        s.sched_tokens = (int32_t)base;
        plan_.n_tok -= n - base;
        kv_->set_keep_floor(r.id, real);
    }
}

/* A build that failed after lane_split_own cut this lane's chunks: nothing of the step ran, so the
 * pieces it lent come back out of the borrowers' inboxes and each request's lane fields are what they
 * were before the cut; the next step re-plans the chunk whole from the committed end (as on one lane,
 * "the next step re-schedules the same chunks"). Left lent, a borrower would run [c0 + base, ...) over
 * rows [c0, c0 + base) never computed and count them committed. The lock is held. */
void Scheduler::lane_unsplit_own() {
    if (!lanes_) return;
    for (const LaneLent& u : lane_lent_now_) {
        for (const auto& [to, at] : u.to)
            if (!lanes_->unlend_locked(to, lane_, u.seq, at))
                lanes_->note_broken_locked(fmt("lane %d: a failed build could not take back its piece at %d from lane %d's "
                                               "inbox", lane_, (int)at, to));
        lanes_->unnote_lent_locked(u.rows, u.tok);
        if (u.slot < 0 || (size_t)u.slot >= reqs_.size() || !reqs_[(size_t)u.slot]) continue;
        SchedReq& s = *reqs_[(size_t)u.slot];
        s.lent = u.lent;
        s.lent_pending = u.lent_pending;
        s.lent_tok = u.lent_tok;
        s.lent_next = u.lent_next;
        if (s.req && s.lent == 0) kv_->set_keep_floor(s.req->id, -1);
    }
    lane_lent_now_.clear();
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
