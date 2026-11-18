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
    const char* sp = std::getenv("RADIANCE_PP_SPAN");
    span_on_ = !(sp && !std::strcmp(sp, "0"));
    if (const char* st = std::getenv("RADIANCE_PP_SPAN_TAIL"); st && *st) {
        char* end = nullptr;
        const long long v = std::strtoll(st, &end, 10);
        span_tail_ = end && !*end && v > 0 ? (int64_t)v : 0;
    }
    if (const char* ss = std::getenv("RADIANCE_PP_SPAN_SHORT"); ss && *ss) {
        char* end = nullptr;
        const long long v = std::strtoll(ss, &end, 10);
        span_short_ = end && !*end && v > 0 ? (int64_t)v : 0;
    }
    const char* si = std::getenv("RADIANCE_PP_SPAN_INNER");
    span_inner_ = si && !std::strcmp(si, "1");
    const char* lb = std::getenv("RADIANCE_PP_LEND_BUSY");
    lend_busy_ = !(lb && !std::strcmp(lb, "0"));
    const char* lm = std::getenv("RADIANCE_PP_LEND_MIXED");
    lend_mixed_ = lm && !std::strcmp(lm, "1") ? 1 : lm && !std::strcmp(lm, "2") ? 2 : 0;
    bg_ = bg_env();
    const char* bs = std::getenv("RADIANCE_PP_BG_SPLIT");
    bg_split_ = bs && !std::strcmp(bs, "1");
    if (const char* m = std::getenv("RADIANCE_PP_SPLIT_MIN"); m && *m) {
        const long long v = std::strtoll(m, nullptr, 10);
        if (v > 0) split_min_ = (int64_t)v;
    }
}

int LaneSet::bg_env() {
    const char* v = std::getenv("RADIANCE_PP_BG_LANE");
    if (!v || !*v) return 0;
    char* end = nullptr;
    const long k = std::strtol(v, &end, 10);
    return end && !*end && k > 0 && k <= 3 ? (int)k : 0;
}

int LaneSet::pick() const {
    if (lanes_.size() <= 1) return 0;
    std::lock_guard<std::mutex> lk(*mu_);
    struct Load { int64_t live = 0, drafted = 0, free_cards = 0; };
    /* tp8 dev M8: the BG lane takes no request of its own */
    const size_t nl = lanes_.size() - (size_t)bg_n();
    if (nl <= 1) return 0;
    std::vector<Load> ld(nl);
    for (size_t l = 0; l < nl; ++l) {
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
    for (size_t l = 1; l < nl; ++l)
        if (better(l, best)) best = l;
    /* RADIANCE_PP_PICK=rr: equally loaded lanes take new requests in turn (a request alone on an idle
     * engine goes to lane 0, 1, 2, ... in turn), so an identity run of requests one at a time covers
     * every lane. Default: the lowest index. */
    if (pick_rr_) {
        std::vector<size_t> tie;
        for (size_t l = 0; l < nl; ++l)
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

bool LaneSet::prefill_pending(int l) const {
    if (l < 0 || l >= (int)lanes_.size()) return false;
    std::lock_guard<std::mutex> lk(*mu_);
    const Scheduler& s = *lanes_[(size_t)l];
    if (!s.waiting_.empty()) return true;
    for (int32_t slot : s.running_) {
        const SchedReq* q = slot >= 0 && (size_t)slot < s.reqs_.size() ? s.reqs_[(size_t)slot].get() : nullptr;
        if (q && q->req && q->req->n_computed + q->lent < (int64_t)q->req->prompt.size()) return true;
    }
    return false;
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
        /* tp8 M56: the prefix cache's hits are counted by the lane that admitted the request (the cache's own
         * counts -- evictions, expiries -- are the one cache's, lane 0's copy) */
        m.attn_cached_tokens += o.attn_cached_tokens;
        m.linear_cached_tokens += o.linear_cached_tokens;
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
        if (s.req && (s.lent_pending > 0 || s.lent_tok || s.lent_hold)) {
            if (s.lent_hold) lane_why(s, LaneSet::kHold);
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
    /* a chunk wider than the shared budget is lent only to a lane with nothing else: its step is that one sequence */
    if (geo_.shared_tok > 0 && rows > geo_.shared_tok) return 0;
    if (cfg_.prefill_span > 1) {
        span_now_ = false;
        if (budget > mixed_now_ && mixed_now_ > 0) budget = mixed_now_;
    }
    /* tp8 dev M5: a lent newcomer piece beside this lane's decoders keeps them a decode room (one process plans the piece
     * after the decode rows, over the budget where it must: bi_chunk_plan's last rule), within the shared budget */
    if (lanes_->lend_mixed() && budget <= rows && geo_.shared_tok > 0 && rows + lane_decode_room() <= geo_.shared_tok)
        return lane_decode_room();
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
    span_now_ = alone_spans(r, target);
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
    /* tp8 S5 review: a borrower's rows within the shared budget under a span -- a step carrying another lane's chunk
     * is a shared step and a step wider than the shared budget must be one sequence (RadBuildCtx::max_mixed_tok); two
     * lenders lending non-wide chunks into one borrower (three lanes or more) would pass max_tok alone */
    const int64_t rows_cap = cfg_.prefill_span > 1 && geo_.shared_tok > 0 ? std::min<int64_t>(cfg_.max_tok, geo_.shared_tok)
                                                                         : cfg_.max_tok;
    /* tp8 dev M8: the BG lanes (RADIANCE_PP_BG_LANE=k) -- whole newcomers beside decode rows go to them; they lend nothing */
    const bool bg = lanes_->bg_lane() >= 0;
    std::vector<size_t> bg_out;   /* this plan's entries lent whole to the BG lane, taken out after the walk */
    for (size_t ei = 0; ei < plan_.e.size(); ++ei) {
        StepEntry& e = plan_.e[ei];
        /* tp8 prefill P0: every chunk a lane plans of its own prompts, counted for the prompt's line */
        if (e.home < 0 && e.is_prefill && e.slot >= 0 && (size_t)e.slot < reqs_.size() && reqs_[(size_t)e.slot] &&
            reqs_[(size_t)e.slot]->req) {
            SchedReq& c = *reqs_[(size_t)e.slot];
            ++c.lst.home;
            if (!c.lst.t0_us) c.lst.t0_us = mono_us();
        }
        /* tp8 dev M8 (RADIANCE_PP_BG_LANE=1): a newcomer's whole rest beside decode rows, lent whole to the BG lane (an inner
         * tail snapshot -- lever TI -- goes with it and commits there, as a lent cold chunk's does) */
        if (bg && !lanes_->is_bg(lane_) && plan_.phase == RAD_PHASE_MIXED && e.home < 0 && e.is_prefill && e.produces_token &&
            e.slot >= 0 && (size_t)e.slot < reqs_.size() && reqs_[(size_t)e.slot] && reqs_[(size_t)e.slot]->req &&
            !e.cr_replay && e.n_spec == 0 && (e.ckpt_slot < 0 || e.ckpt_tok >= 0)) {
            if (lane_split_bg(e)) bg_out.push_back(ei);
            continue;
        }
        /* tp8 dev M5b (RADIANCE_PP_LEND_MIXED=2): a newcomer's whole rest beside decode rows, cut on the 512 grid */
        if (lanes_->lend_split() && plan_.phase == RAD_PHASE_MIXED && e.home < 0 && e.is_prefill && e.produces_token &&
            e.slot >= 0 && (size_t)e.slot < reqs_.size() && reqs_[(size_t)e.slot] && reqs_[(size_t)e.slot]->req &&
            !e.cr_replay && e.n_spec == 0 && e.ckpt_slot < 0 && e.ckpt_restore < 0) {
            lane_split_mixed(e);
            continue;
        }
        if (e.home >= 0 || !e.is_prefill || e.produces_token || e.slot < 0 || (size_t)e.slot >= reqs_.size()) continue;
        if (e.cr_replay || e.n_spec != 0) continue;
        if (hold && e.ckpt_slot >= 0 && e.ckpt_tok >= 0) {
            if (reqs_[(size_t)e.slot]) lane_why(*reqs_[(size_t)e.slot], LaneSet::kSnap);
            continue;
        }
        SchedReq& s = *reqs_[(size_t)e.slot];
        if (!s.req) continue;
        Request& r = *s.req;
        if (r.embed || r.media || r.token_scores || r.n_draft != 0 || s.lent_hold) continue;
        const int64_t target = s.seq_len();
        /* only an entry on the one-process grid: its own chunk is the one an idle engine plans from its start (tp8 dev M5:
         * or, under RADIANCE_PP_LEND_MIXED=1, a piece option D cut inside its alone piece [bi_beg, bi_end) beside decode
         * rows: the pieces lent after it are the ones one process cuts next, up to that alone piece's end) */
        bool dpiece = false;
        if (lane_alone_chunk(s, r, target, e.ctx_len, nullptr) != e.n_tokens) {
            dpiece = lanes_->lend_mixed() && bi_chunk_ == 3 && s.bi_beg <= e.ctx_len &&
                     (int64_t)e.ctx_len + e.n_tokens < s.bi_end && s.bi_end <= target;
            if (!dpiece) {
                lane_why(s, LaneSet::kOffGrid);
                continue;
            }
        }
        /* The committed end, below which no piece in flight reads: this step's bump undone. */
        int64_t real = r.n_computed;
        for (const auto& [slot, rows] : lane_bump_)
            if (slot == e.slot) real -= rows;
        int64_t p = (int64_t)e.ctx_len + e.n_tokens, rows = 0;
        bool tok = false, any = false;
        /* tp8 S5 review: the keep floor before the lend-time ensure below, not after it. ensure_from(r, p + n, p)
         * recycles a window group's blocks below p - window; with no floor yet (a sequence's first lend) that dropped
         * the blocks of this step's own chunk [e.ctx_len, p), which has not run: its keys and its queries' window
         * would be gone. The floor is the committed end, below which nothing in flight reads. */
        const bool fresh_floor = s.lent == 0;
        if (fresh_floor) kv_->set_keep_floor(r.id, real);
        /* tp8 S5 review: what this lend changes, for lane_unsplit_own should the step's build fail */
        LaneLent undo;
        undo.slot = e.slot;
        undo.seq = e.seq;
        undo.lent = s.lent;
        undo.lent_pending = s.lent_pending;
        undo.lent_tok = s.lent_tok;
        undo.lent_next = s.lent_next;
        undo.lent_hold = s.lent_hold;
        int why = -1;   /* tp8 prefill P0: why the walk stopped short of the prompt's end */
        size_t j = 0;
        for (; j < order.size() && p < (dpiece ? s.bi_end : target); ++j) {
            const int to = order[j];
            if (lanes_->is_bg(to)) continue;   /* tp8 dev M8: a cold prompt's chunks keep to the decode lanes */
            int64_t ck = -1;
            const int64_t n = dpiece ? lane_d_piece(s, p, &ck) : lane_alone_chunk(s, r, target, p, &ck);
            if (n <= 0) break;
            const Scheduler* b = lanes_->lane(to);
            if (!lanes_->lend_busy() && (!b || !b->running_.empty() || !b->waiting_.empty())) {
                why = LaneSet::kBusy;
                break;
            }
            const bool wide = geo_.shared_tok > 0 && n > geo_.shared_tok;
            if (wide && (!b || !b->running_.empty() || !b->waiting_.empty() || lanes_->inbox_n_locked(to) > 0)) {
                why = LaneSet::kWideBusy;
                break;
            }
            /* the borrower's entries: its own cards' beside the inbox (the S5 review's rule) */
            if (lanes_->inbox_rows_locked(to) + n > (wide ? cfg_.max_tok : rows_cap) ||
                (int64_t)lanes_->inbox_n_locked(to) + 1 + (cfg_.max_seqs + n_lanes_ - 1) / n_lanes_ > cfg_.max_seqs) {
                why = LaneSet::kRoom;
                break;
            }
            if (kv_->ensure_from(r.id, p + n, p) < 0) {   /* never preempt for a lent chunk */
                why = LaneSet::kKv;
                break;
            }
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
            undo.to.emplace_back(to, pc.e.ctx_len);
            s.lent += (int32_t)n;
            ++s.lent_pending;
            rows += n;
            any = true;
            p += n;
            if (pc.e.produces_token) { tok = true; break; }
            if (hold && pc.e.ckpt_slot >= 0) { s.lent_hold = true; why = LaneSet::kSnap; break; }
        }
        if (why < 0 && !tok && p < (dpiece ? s.bi_end : target) && j >= order.size()) why = LaneSet::kNoLane;
        if (dpiece && rows > 0) lanes_->note_dpiece_locked();
        if (why >= 0) lane_why(s, why);
        s.lst.lent += (int32_t)undo.to.size();
        if (!any) {
            if (fresh_floor) kv_->set_keep_floor(r.id, -1);
            continue;
        }
        undo.rows = rows;
        undo.tok = tok;
        lane_lent_now_.push_back(std::move(undo));
        lanes_->note_lent_locked(rows, tok);
        s.lent_tok = s.lent_tok || tok;
        s.lent_next = (int64_t)e.ctx_len + e.n_tokens;
        kv_->set_keep_floor(r.id, real);
    }
    if (!bg_out.empty()) lane_bg_drop(bg_out);
}

/* tp8 dev M8: entry e (a newcomer's whole rest beside this lane's decoders) lent whole to a BG lane -- the first one in this
 * plan's planning order (idle ones first) with room; else e stays (today's MIXED step). True when lent: the caller takes e out of the
 * plan (lane_bg_drop). The piece is e itself -- the rows, the snapshot (lever TI's inner tail: committed by lane_piece_done) --
 * so every boundary is the one this lane's own plan cut. */
bool Scheduler::lane_split_bg(StepEntry& e) {
    SchedReq& s = *reqs_[(size_t)e.slot];
    Request& r = *s.req;
    if (r.embed || r.media || r.token_scores || r.n_draft != 0 || s.lent_hold || s.lent_pending > 0 || s.lent > 0) return false;
    if (e.ckpt_restore >= 0) {
        lanes_->note_bg_skip_locked(LaneSet::kBgRestore);
        return false;
    }
    const std::vector<int>& order = lanes_->planners_locked();
    const int64_t n = e.n_tokens;
    const int64_t cap = geo_.shared_tok > 0 ? std::min<int64_t>(cfg_.max_tok, geo_.shared_tok) : cfg_.max_tok;
    auto room = [&](int l, int64_t rows) {
        return lanes_->inbox_rows_locked(l) + rows <= cap && (int64_t)lanes_->inbox_n_locked(l) + 1 <= cfg_.max_seqs;
    };
    /* the BG lanes in this plan's planning order (the order they take their inboxes in) */
    std::vector<int> bgo;
    for (int l : order)
        if (lanes_->is_bg(l)) bgo.push_back(l);
    /* tp8 dev M8 (RADIANCE_PP_BG_SPLIT=1, two BG lanes or more): the entry cut at the 512-grid point nearest its middle (>= 128
     * rows a side: M5b's cut, answer-neutral by the cards' lq21 lid576), the first piece to the first BG lane in the order with
     * room, the rest (its first token, the snapshot when it lies there) to a later one: piece 1's leader tail runs beside piece
     * 2's follower segments, and each segment is half a whole one's. Else: whole, to one BG lane. */
    int64_t x = 0;
    int to = -1, to2 = -1;
    if (lanes_->bg_split() && bgo.size() >= 2) {
        constexpr int64_t G = 512, kMin = 128;
        const int64_t c0 = e.ctx_len, end = c0 + n;
        for (int64_t g = (c0 + kMin + G - 1) / G * G; g <= end - kMin; g += G)
            if (!x || std::llabs(g - (c0 + end) / 2) < std::llabs(x - (c0 + end) / 2)) x = g;
        /* a snapshot inside the first piece keeps the entry whole (its next piece would wait for its commit at h > 0) */
        if (x && e.ckpt_slot >= 0 && c0 + e.ckpt_tok + 1 <= x) x = 0;
        if (x) {
            size_t k = 0;
            for (; k < bgo.size() && to < 0; ++k)
                if (room(bgo[k], x - c0)) to = bgo[k];
            for (; k < bgo.size() && to2 < 0; ++k)
                if (room(bgo[k], end - x)) to2 = bgo[k];
            if (to < 0 || to2 < 0) { x = 0; to = to2 = -1; lanes_->note_bg_skip_locked(LaneSet::kBgSplitRoom); }
        }
    }
    if (!x)
        for (int l : bgo)
            if (room(l, n)) { to = l; break; }
    if (to < 0) {
        lanes_->note_bg_skip_locked(!bgo.empty() ? LaneSet::kBgRoom : LaneSet::kBgOrder);
        return false;
    }
    /* The committed end, below which nothing in flight reads (this step's bump undone): the keep floor, as a lend's */
    int64_t real = r.n_computed;
    for (const auto& [slot, rows] : lane_bump_)
        if (slot == e.slot) real -= rows;
    kv_->set_keep_floor(r.id, real);
    LaneLent undo;
    undo.slot = e.slot;
    undo.seq = e.seq;
    undo.lent = s.lent;
    undo.lent_pending = s.lent_pending;
    undo.lent_tok = s.lent_tok;
    undo.lent_next = s.lent_next;
    undo.lent_hold = s.lent_hold;
    LaneSet::Piece pc;
    pc.home = lane_;
    pc.e = e;
    pc.e.dev = false;
    pc.e.ckpt_restore = -1;
    pc.e.n_accepted = 1;
    if (x) {   /* RADIANCE_PP_BG_SPLIT: [c, x) without a token or a snapshot, then [x, end) with both */
        LaneSet::Piece p1 = pc;
        p1.e.n_tokens = (int32_t)(x - e.ctx_len);
        p1.e.produces_token = false;
        p1.e.ckpt_slot = -1;
        p1.e.ckpt_tok = -1;
        lanes_->lend_locked(to, p1);
        undo.to.emplace_back(to, p1.e.ctx_len);
        ++s.lent_pending;
        pc.e.ctx_len = (int32_t)x;
        pc.e.n_tokens = (int32_t)(e.ctx_len + n - x);
        if (pc.e.ckpt_slot >= 0) pc.e.ckpt_tok = (int32_t)(e.ckpt_tok - (x - e.ctx_len));
        to = to2;
        lanes_->note_bg_split_locked();
    }
    lanes_->lend_locked(to, pc);
    undo.to.emplace_back(to, pc.e.ctx_len);
    s.lent += (int32_t)n;
    ++s.lent_pending;
    undo.rows = n;
    undo.tok = true;
    lane_lent_now_.push_back(std::move(undo));
    lanes_->note_lent_locked(n, true);
    lanes_->note_bg_locked(n);
    ++s.lst.lent;
    s.lent_tok = true;
    s.lent_next = e.ctx_len;
    /* at W(h > 0, t) a lent snapshot holds its sequence until its commit (lane_split_own's rule) -- it is held anyway (lent_tok) */
    if (lanes_->head_state() && pc.e.ckpt_slot >= 0) s.lent_hold = true;
    return true;
}

/* tp8 dev M8: the entries lane_split_bg lent whole leave this lane's plan (prompt entries sit after the decode entries, so
 * the decode rows' indices do not move); the entries behind them get their index again, and a step left with decode rows
 * alone goes out as a DECODE step (the counter step() bumped for it moved over). */
void Scheduler::lane_bg_drop(const std::vector<size_t>& at) {
    for (auto it = at.rbegin(); it != at.rend(); ++it) {
        const StepEntry& e = plan_.e[*it];
        if (e.slot >= 0 && (size_t)e.slot < reqs_.size() && reqs_[(size_t)e.slot]) {
            reqs_[(size_t)e.slot]->entry = -1;
            reqs_[(size_t)e.slot]->sched_tokens = 0;
        }
        plan_.n_tok -= e.n_tokens;
        plan_.e.erase(plan_.e.begin() + (ptrdiff_t)*it);
    }
    bool any_pf = false;
    for (size_t i = 0; i < plan_.e.size(); ++i) {
        const StepEntry& e = plan_.e[i];
        if (e.is_prefill) any_pf = true;
        if (e.home < 0 && e.slot >= 0 && (size_t)e.slot < reqs_.size() && reqs_[(size_t)e.slot]) reqs_[(size_t)e.slot]->entry = (int32_t)i;
    }
    if (!any_pf && plan_.phase == RAD_PHASE_MIXED) {
        plan_.phase = RAD_PHASE_DECODE;
        --m_.steps_mixed;
        ++m_.steps_decode;
    }
}

/* tp8 dev M5b: the home's whole-prompt MIXED entry e cut at the 512-grid point nearest its middle with >= 128 rows a side, the
 * rest lent to the first planner whose inbox is empty and that has no request waiting (a borrower carries one piece and
 * plans its own work beside it at the published budget); none such, or no grid point: e stays whole (today's plan). The
 * entry shrinks before the step's rows and the build are counted (Scheduler::step after lane_split_own). */
void Scheduler::lane_split_mixed(StepEntry& e) {
    SchedReq& s = *reqs_[(size_t)e.slot];
    Request& r = *s.req;
    if (r.embed || r.media || r.token_scores || r.n_draft != 0 || s.lent_hold || s.lent_pending > 0) return;
    constexpr int64_t G = 512, kMin = 128;
    const int64_t c = e.ctx_len, end = c + e.n_tokens;
    int64_t x = 0;
    for (int64_t g = (c + kMin + G - 1) / G * G; g <= end - kMin; g += G)
        if (!x || std::llabs(g - (c + end) / 2) < std::llabs(x - (c + end) / 2)) x = g;
    if (!x) return;
    const std::vector<int>& order = lanes_->planners_locked();
    int to = -1;
    for (int l : order) {
        const Scheduler* b = lanes_->lane(l);
        if (b && b->waiting_.empty() && lanes_->inbox_n_locked(l) == 0) { to = l; break; }
    }
    if (to < 0) {
        lane_why(s, LaneSet::kNoLane);
        return;
    }
    const int64_t n = end - x;
    if (lanes_->inbox_rows_locked(to) + n > (geo_.shared_tok > 0 ? std::min<int64_t>(cfg_.max_tok, geo_.shared_tok) : cfg_.max_tok)) {
        lane_why(s, LaneSet::kRoom);
        return;
    }
    int64_t real = r.n_computed;
    for (const auto& [slot, rows] : lane_bump_)
        if (slot == e.slot) real -= rows;
    const bool fresh_floor = s.lent == 0;
    if (fresh_floor) kv_->set_keep_floor(r.id, real);
    if (kv_->ensure_from(r.id, end, x) < 0) {
        if (fresh_floor) kv_->set_keep_floor(r.id, -1);
        lane_why(s, LaneSet::kKv);
        return;
    }
    LaneLent undo;
    undo.slot = e.slot;
    undo.seq = e.seq;
    undo.lent = s.lent;
    undo.lent_pending = s.lent_pending;
    undo.lent_tok = s.lent_tok;
    undo.lent_next = s.lent_next;
    undo.lent_hold = s.lent_hold;
    LaneSet::Piece pc;
    pc.home = lane_;
    pc.e = e;
    pc.e.ctx_len = (int32_t)x;
    pc.e.n_tokens = (int32_t)n;
    pc.e.produces_token = true;
    pc.e.dev = false;
    pc.e.ckpt_restore = -1;
    pc.e.ckpt_slot = -1;
    pc.e.ckpt_tok = -1;
    pc.e.n_accepted = 1;
    lanes_->lend_locked(to, pc);
    undo.to.emplace_back(to, pc.e.ctx_len);
    s.lent += (int32_t)n;
    ++s.lent_pending;
    undo.rows = n;
    undo.tok = true;
    lane_lent_now_.push_back(std::move(undo));
    lanes_->note_lent_locked(n, true);
    lanes_->note_dpiece_locked();
    ++s.lst.lent;
    s.lent_tok = true;
    s.lent_next = x;
    kv_->set_keep_floor(r.id, real);
    /* the home keeps [c, x): no token this step */
    e.n_tokens = (int32_t)(x - c);
    e.produces_token = false;
    plan_.n_tok -= n;
}

/* A build that failed after lane_split_own lent this lane's newcomers' next chunks: nothing of the step ran, so the
 * chunks it lent come back out of the borrowers' inboxes and each request's lane fields are what they were before the
 * lend; the next step plans and lends anew from the committed end (as on one lane, "the next step re-schedules the
 * same chunks"; the lent chunks' KV stays ensured). Left lent, a borrower would run [p, ...) over the home chunk's
 * rows [c, p) never computed and count them committed. The lock is held. */
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
        s.lent_hold = u.lent_hold;
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
    lane_prompt_note(s);
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

/* tp8 prefill P0: LaneSet::kWhyN reasons, in the set (the lanes line) and in the request's own line. */
static_assert(LaneSet::kWhyN == (int)(sizeof(SchedReq::LaneStats::why) / sizeof(int32_t)),
              "SchedReq::LaneStats::why holds one count a LaneSet reason");

void Scheduler::lane_why(SchedReq& s, int k) {
    if (!lanes_ || n_lanes_ <= 1 || k < 0 || k >= LaneSet::kWhyN) return;
    lanes_->note_why_locked(k);
    ++s.lst.why[k];
}

/* One leader line a prompt whose chunks went over the lanes (or were refused), at its first token: what the test side
 * reads from one cold-prompt run. The lock is held. */
void Scheduler::lane_prompt_note(SchedReq& s) {
    if (!lanes_ || n_lanes_ <= 1 || !s.req || !s.lst.t0_us) return;
    int32_t refused = 0;
    for (int k = 0; k < LaneSet::kWhyN; ++k) refused += s.lst.why[k];
    if (s.lst.lent == 0 && refused == 0) { s.lst.t0_us = 0; return; }
    std::string why;
    for (int k = 0; k < LaneSet::kWhyN; ++k)
        if (s.lst.why[k]) why += fmt("%s%s %d", why.empty() ? "" : ", ", LaneSet::why_name(k), (int)s.lst.why[k]);
    RAD_INFO("lanes: request %llu's prompt of %lld tokens: %d chunk(s) planned by its lane %d, %d lent to other lanes; "
             "%s; first chunk's plan to its first token %.1f ms", (unsigned long long)s.req->id,
             (long long)s.req->prompt.size(), (int)s.lst.home, lane_, (int)s.lst.lent,
             why.empty() ? "every chunk after a home chunk lent" : why.c_str(),
             (double)(mono_us() - s.lst.t0_us) / 1000.0);
    s.lst.t0_us = 0;
}

}  /* namespace rad */
