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
 *   - the prefix cache, its tiers and checkpoints stay off (tp8 S4's rule), so no restore crosses;
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

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <thread>

namespace rad {

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
    /* Every lane's batch is its own set of recordings (Ctx::pass_key reads its slab's pointers). */
    for (auto& rk : ranks_) rk->ctx.set_tape_lanes(n_lanes_);
    /* lane l holds cards [l * S / N, (l + 1) * S / N) (Scheduler::init) */
    const int64_t per = (cfg_.max_seqs + n_lanes_ - 1) / n_lanes_, least = cfg_.max_seqs / n_lanes_;
    RAD_INFO("two-node pipeline: %d lanes, each a scheduler with its own step slab on every card and at most %lld "
             "sequences (cards %lld-%lld a lane); the newcomer split %s (pieces of at least %lld rows on the %lld-row "
             "grid; RADIANCE_PP_SPLIT, RADIANCE_PP_SPLIT_MIN)", n_lanes_, (long long)per, (long long)least, (long long)per,
             lanes_.split_on() ? "on" : "off", (long long)lanes_.split_min(), (long long)std::max<int64_t>(geo_.quantum, 1));
    return RAD_OK;
}

/* A lane step's first leader pass: its build shipped to the follower, then the leader's part up to
 * and including stage_send X1, with no sampler (kIssueModel). */
int Engine::lane_head(int lane, const RadBatch* b) {
    lane_select(lane);
    clear_fresh_states();
    if (const int ps = prepare_release(b); ps < 0) return ps;
    if (!pp_trunk_pending_[lane]) {
        RAD_ERR("two-node pipeline: lane %d's step %d was released without its build", lane, b->step);
        return RAD_E_STATE;
    }
    pp_trunk_pending_[lane] = false;
    RAD_TRY(pp_ship(b, true, lane));
    pp_set_pass(lane, RAD_PP_PART_HEAD);
    issue_what_.store(kIssueModel, std::memory_order_release);
    step_batch_.store(b, std::memory_order_release);
    const uint64_t t0 = pp_stamps_ ? rad_mono_ns() : 0;
    barrier_->arrive_and_wait();      /* release the ranks */
    barrier_->arrive_and_wait();      /* and wait for them */
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
    RAD_INFO("two-node pipeline: %lld lane steps, %lld of them with another lane's step in flight at their tail; "
             "newcomer split: %lld pieces lent (%lld rows, %lld carrying a first token), %lld committed",
             (long long)lane_steps, (long long)rounds_two, (long long)c.pieces, (long long)c.rows,
             (long long)c.token_pieces, (long long)c.done);
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
    if (ctxs_.ready && ctxs_.on) {
        RAD_WARN("two-node pipeline with %d lanes: the depth-0 context skip (draft_ctxskip) keeps consecutive "
                 "steps' rows in one ring; lanes interleave their steps, so it is off", n_lanes_);
        ctxs_.on = 0;
    }
    const bool t_on = (int)log_level() >= (int)Log::Debug;
    rad::StallWatch step_watch("step", 20000);
    /* The lanes with a head issued, oldest first; their tails go in this order (the order of their
     * X2 on the follower's stream). */
    std::deque<std::pair<int, const RadBatch*>> fifo;
    std::vector<char> busy((size_t)n_lanes_, 0);
    int next = 0;   /* the lane planned first on the next round */
    int64_t lane_steps = 0, rounds_two = 0;
    while (!stopping_.load(std::memory_order_acquire)) {
        step_watch.progress();
        if (pp_link_ && !pp_link_->alive()) {
            RAD_ERR("two-node pipeline: the link to the follower is down (%s): failing every request",
                    pp_link_->why().c_str());
            return fail_loop(RAD_E_IO);
        }
        lora_epoch_.fetch_add(1, std::memory_order_acq_rel);
        phase_t_ = std::chrono::steady_clock::now();
        /* 1. every lane with no step in flight plans, in turn; each one's head goes out at once. The lanes
         * that plan before `l` plans again, in that order, are the idle ones after it -- planned right
         * behind it, before anything else -- then those in flight, oldest first (their commits come
         * before l's). A newcomer's chunk planned by `l` lends its pieces to them in that order
         * (lanes.h), so on both nodes the pieces run in position order. */
        std::vector<int> todo;
        for (int k = 0; k < n_lanes_; ++k) todo.push_back((next + k) % n_lanes_);
        for (size_t at = 0; at < todo.size(); ++at) {
            const int l = todo[at];
            if (busy[(size_t)l]) continue;
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
            const RadBatch* b = lsched().step();
            if (const std::string why = lanes_.broken(); !why.empty()) {
                RAD_ERR("two-node pipeline: %s -- the engine stops", why.c_str());
                return fail_loop(RAD_E_STATE);
            }
            if (!b) continue;
            if (t_on) phase_mark(PH_PLAN);
            if (const int hs = lane_head(l, b); hs < 0) return fail_loop(hs);
            busy[(size_t)l] = 1;
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
        /* 2. the oldest lane step's tail, its drafter, its commit */
        const auto [lane, b] = fifo.front();
        fifo.pop_front();
        if (const int ts = lane_tail(lane, b, so, cancel, t_on); ts < 0) return fail_loop(ts);
        busy[(size_t)lane] = 0;
        if (++lane_steps == 1 || lane_steps % 4096 == 0) lanes_note(lane_steps, rounds_two);
    }
    lanes_note(lane_steps, rounds_two);
    return RAD_OK;
}

/* The tail of lane `lane`'s step `b`, whose head is out: Engine::step_loop's step from its sampler's
 * staging on, released as the leader's part from stage_recv X2 (kIssueAll), then its drafter, its
 * collect and commit -- with lsched() this lane's scheduler throughout. */
int Engine::lane_tail(int lane, const RadBatch* b, StepOut& so, std::vector<uint64_t>& cancel, bool t_on) {
    lane_select(lane);
    const StepPlan& plan = lsched().plan();
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
    for (const StepEntry& e : plan.e)
        if (e.ckpt_restore >= 0) {
            RAD_ERR("two-node pipeline: step %d restores a snapshot (a prefix hit), which the pipeline does not carry",
                    b->step);
            return RAD_E_STATE;
        }

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
    /* The tail's release: the arena level and the follower pools for its rows, no switch change. */
    if (const int ps = prepare_release(b, /*switches=*/false); ps < 0) return ps;
    pp_set_pass(lane, RAD_PP_PART_TAIL);
    issue_what_.store(embedding_ ? kIssueModel : kIssueAll, std::memory_order_release);
    step_batch_.store(b, std::memory_order_release);
    const uint64_t t0 = pp_stamps_ ? rad_mono_ns() : 0;
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
    if (pp_stamps_) {
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
    RAD_TRY(stream_probe_issue(b));
    RAD_TRY(mark_step_end(b, so));
    st = embedding_ ? collect_pooled(b, so) : collect_step(b, so);
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
    if (cfg_.kld()) RAD_TRY(score_rows(b, so));
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
