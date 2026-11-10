/* draft.h -- the speculative draft-depth controller.
 *
 * Static by default: the depth is `--num-speculative-tokens`. A dynamic policy that gates on the
 * drafter's own top-1 confidences cannot work here, because the scheduler picks the depth before
 * the drafter runs; measured acceptance arrives a step late and pooled across sequences. With a
 * draft head whose cost is nearly flat in depth (one extra verify row on a weight-bound step), the
 * best depth is a property of the model pair and is known up front, so it is a flag. The
 * controller still keeps acceptance accounting, a reported metric (spec §14).
 *
 * `--draft-window adaptive` is for when that premise fails: with streamed weights (a MoE reading
 * experts per verify row, a serial head running one pass per round) each draft costs a head round
 * and a verify row, and the best depth depends on the text (deep on code and JSON, shallow on
 * prose). The adaptive policy has two halves:
 *
 *   Per sequence (DraftAdapt): a smoothed estimate of each draft position's conditional
 *   acceptance, fed by the verify outcomes, and the sequence's stop -- the deepest draft whose
 *   chance of being emitted still pays for its verify row (kDraftMinChance).
 *
 *   Per step (plan_step): the head depth D. A serial head runs as many rounds as the deepest
 *   window in the step, so D maximises sum_i E_i(min(D, stop_i)) / T(D, n), with
 *   E(d) = 1 + sum_k prod_{j<=k} a_j and T measured online (step_time). Each sequence then
 *   drafts min(D, stop_i).
 *
 * Both move only past a hold, and D also past a hysteresis, because a new window is a new pass
 * shape that must be recorded before it plays (runtime/ctx.h). `max_depth` stays the ceiling and
 * everything is sized for it.
 *
 * D pays for serial head rounds; a block drafter proposes its block in one pass and prompt lookup
 * runs none, so for those the policy prices only the verify rows (Scheduler::init): each sequence
 * verifies the first min(D, its stop) of its proposal, which on a MoE that reads each row's routed
 * experts is still the cost that matters.
 *
 * Depth 0 (Params::depth0: RADIANCE_DRAFT_DEPTH0=1 at start, or the fn4 switch draft_depth0) lets
 * the step choose D = 0 as well: no head pass and no verify row, each sequence emits one token, so
 * the step is a plain decode step (a block drafter's context pass still runs: it fills the
 * drafter's window cache, which a hole would leave unusable for a window). Under load a verify row
 * costs about what it buys -- each row reads its own routed experts -- and a plain step can emit
 * more tokens a millisecond than the best window. Its price is measured like any depth's, from
 * plain steps (step_time keyed 0), so D = 0 is taken only where the measured plain step beats the
 * window's measured yield by kHyst. While D is 0 nothing is verified, so nothing updates the
 * acceptance estimates or the window's step times; a probe (kProbeSteps cycles at the drafting
 * depth every probe_every cycles at 0) keeps both alive, and a plain probe (cycles at 0, the first
 * kPlainWarm plain steps not sampled) measures the plain step of a class that has none, or whose
 * last is older than kPlainStale probe intervals. The drafting side of the comparison is the lower
 * of the model's yield and the yield drafting steps measured (step_time's tokens).
 *
 * Probe back-off (Params::backoff: RADIANCE_DRAFT_PROBE_BACKOFF at start, or the fn4 switch
 * draft_probe_backoff = N): each probe at the drafting depth that ends with depth 0 still winning
 * doubles the interval to the next, up to probe_every << N; a move between 0 and drafting starts it
 * over at probe_every. A long run at 0 then pays few probes, at the price of finding a turn into
 * drafting text up to that many cycles late.
 *
 * The drafter's lend (Params::lend: RADIANCE_DRAFT_LEND at start declares it, the fn4 switch
 * draft_lend turns it on; lever DL, place/mover.h): once depth 0 has held lend_hold() cycles and
 * the probe interval is at least kLendInterval, lend_wanted() asks the engine to lend the drafter's
 * expert slab to the trunk; any move off 0 asks for it back. While lent a probe's drafter reads its
 * experts from host memory, so its drafting steps are no samples of a drafting step's cost (the
 * scheduler passes their lend state to step_time); plain steps while lent are kept apart
 * (lent_ratio_: their time over the plain step measured unlent), and the plain side of the
 * comparison is the lent one wherever it was measured, since staying at 0 means staying lent.
 *
 * It changes only how many drafts a step verifies. Acceptance follows core/sample/accept.h, so
 * every emitted token is the target's own; at temperature 0 the text matches a plain run as far
 * as the logits do, and a verify row's logits are computed at an M that moves with the window
 * (see accept.h).
 */
#pragma once
#include "rad_core.h"

namespace rad {

/* libr4d's decode kernel serves up to 64 query rows, the verify-step band (spec §10), so 64 is
 * the largest window any batch can carry. */
enum { RAD_SCHED_MAX_SPEC = 64 };

/* One sequence's half of the adaptive policy: the drafter's record on this sequence's text and the
 * resulting stop. Stored in the scheduler's per-request state; the controller does the arithmetic.
 *
 * `depth` 0 means not started: the first ask fills the estimate from the prior and sets the stop.
 * It belongs to the request, not its cache: preemption keeps it, a recycled slot resets it. */
struct DraftAdapt {
    int32_t  depth = 0;                       /* the sequence's stop; 0 = not started */
    int32_t  held  = 0;                       /* verifies since `depth` last moved */
    float    a[RAD_SCHED_MAX_SPEC] = {};      /* conditional acceptance at each draft position */
    uint16_t n[RAD_SCHED_MAX_SPEC] = {};      /* verifies that evaluated each one, saturating */
};

class DraftController {
public:
    struct Params {
        int  max_depth = 0;       /* the deployment's window; 0 disables speculation entirely */
        /* The adaptive policy: `--draft-window adaptive` with a serial drafter, decided by
         * Scheduler::init. Off, the depth is max_depth. */
        bool adapt = false;
        /* fn4: the drafter could run the adaptive policy (a serial or block head), whether or not
         * it was asked for at start -- set_adaptive() may turn it on later. */
        bool adapt_capable = false;
        /* Depth 0 (see the top of this file): the adaptive policy may also choose D = 0. Inert
         * while the policy is off. */
        bool depth0 = false;
        /* Cycles at D = 0 between probes; 0 is kProbeEvery. */
        int  probe_every = 0;
        /* Probe back-off: at most this many doublings of the interval (0 off, 1..kBackoffMax). */
        int  backoff = 0;
        /* The drafter's lend (lever DL): 0 off, 1 on after kLendHold cycles at 0, 2..4096 on after
         * that many. Acts only where the engine set the lend up (Scheduler::set_lend_view). */
        int  lend = 0;
        /* The lent ratio borrowed by a class with none of its own (kBorrowMin, kBorrowDamp). */
        bool borrow = false;
    };

    /* ---------------------------------------------------------- the policy's constants
     * Not configurable: each is a property of the policy, not of a deployment. The one
     * deployment property, step cost, is measured (step_time). */

    /* A window moves only for 2% more yield. One draft more or less moves E/T by several percent,
     * while a smoothed estimate wobbles about 1% per verify; a lower threshold would flip D
     * between near-equal neighbours and pay a recording each time. */
    static constexpr double kHyst = 0.02;
    /* And not within 8 verifies (steps, for D) of the last move. A pass shape is recorded on its
     * second sighting and played from the third (runtime/ctx.h), so a shape held 8 steps mostly
     * plays from the recording; 8 verifies at kHalfLife move an estimate about a third of the way
     * to a new rate, a trend rather than one rejection. */
    static constexpr int kHold = 8;
    /* Each position's estimate has a half-life of 16 verifies that evaluate it (about 50 tokens of
     * text): short enough to follow a reply from prose into code and back, long enough that one
     * rejection moves an estimate by about 4%. */
    static constexpr double kHalfLife = 16.0;
    /* An unmeasured position starts at 0.8, optimistic on purpose: a window that starts deep
     * learns every position, one that starts shallow never learns the deep ones. 0.8 lies between
     * typical first-position acceptance on prose and on code, so a wrong start corrects within a
     * hold. */
    static constexpr double kPrior = 0.8;
    /* Once 4 sequences have finished, new ones start from the mean of their final estimates
     * (adapt_retire): a running mean, then a moving average with a half-life of 32 sequences, so
     * the prior follows a change of traffic within a few dozen requests. A sequence counts only
     * after kHalfLife verifies; before that its estimate is mostly the prior itself. */
    static constexpr int    kPriorMinSeqs = 4;
    static constexpr double kPriorHalfLifeSeqs = 32.0;
    /* Never fewer than one draft: at zero no position is evaluated, so the estimate could never
     * recover. */
    static constexpr int kMinDepth = 1;
    /* A sequence stops at the first draft whose chance of being emitted, prod_{j<=d} a_j, is
     * under 1 in 10. Once the head runs D rounds for the deepest neighbour, a sequence's only
     * added cost is its verify row. At one sequence the step-cost prior (1 + 0.06 d) breaks even
     * near this chance (a draft pays while p_d / E exceeds the ~6% it adds, with E around 2), and
     * a row alone costs less than a round plus a row, so this is never stricter than D's own
     * trade. A constant because a row's share cannot be separated from the head's rounds in step
     * times keyed by depth. */
    static constexpr double kDraftMinChance = 0.1;
    /* Step times are kept per running-row class: 1, 2-3, 4-7, 8-15, 16+. Cost moves with how many
     * rows share the weight reads; power-of-two classes keep every key fed. */
    static constexpr int kCostClasses = 5;
    /* Depth 0's probes. Every 16 cycles at D = 0 the step drafts at the drafting depth for 2
     * cycles: the first proposes (its step verifies nothing), the second verifies and proposes, so
     * its step time is a full drafting step's, and the step after it verifies once more. At the
     * c4 cost ratio measured on DeepSeek-V4.1 (a d5 adaptive step 72.9 ms for 8.8 tokens, a plain
     * one 28.1 ms for 4) a probe costs about 15 % of two steps, about 1.7 % over 18 cycles. Each
     * probe moves a sequence's estimate by two verifies, so a turn from prose into a copied block
     * shows within a few probes. */
    static constexpr int kProbeEvery = 16;
    static constexpr int kProbeSteps = 2;
    /* A class's plain step is measured again when its last sample is older than 16 probe
     * intervals (256 cycles; at one sequence 2 slower steps in 256, under 1 %): a context that grows changes the plain step and the window's alike, but only the
     * window's is refreshed by the probes from D = 0. */
    static constexpr int kPlainStale = 16;
    /* The first plain steps after the head stops are not samples of a plain step: the first two
     * sightings of a pass shape run eagerly and record (runtime/ctx.h), several times slower than
     * a played one, and the step right after a move still carries the transition. Two such steps
     * in three samples put the first plain estimate near twice the plain step (gpu02 hc-depth0,
     * 10.11.2026: depth 0 never fired at c1 essay and c4, where a plain step pays 9-14 % more),
     * and with one sample a refresh the mean then took hundreds of cycles to come down -- while
     * drafting nothing else measures it. So the first kPlainWarm plain steps of every run at 0
     * are left out, and a plain probe is 1 + kPlainWarm cycles longer than the samples it takes:
     * 3 the first time in a class, 1 at a refresh. */
    static constexpr int kPlainWarm = 2;
    /* Probe back-off's ceiling: 8 doublings of 16 cycles is 4096, the most draft_probe takes. */
    static constexpr int kBackoffMax = 8;
    /* The drafter's lend. It starts once depth 0 has held 32 cycles (two probe intervals at the
     * default) and only while the probe interval is at least 64 cycles. A probe while lent reads the
     * drafter's experts over the link: on DeepSeek-V4.1 TP4 (f0-dplace d5ah vs bz d5a, 10.11.2026) a
     * drafting step took +11.9 ms at one essay sequence, +32.6 ms at c4 and +47.3 ms at c8, two of
     * them a probe; the lent slab is worth about 230 trunk units, ~1.0 / 3.0 / 4.6 ms a plain step by
     * the VRAM sweep's slope (0.42 / 1.3 / 2.0 ms per 100 units). Break-even is 21-25 cycles between
     * probes at every load; 64 keeps about 60 % of the gain, 128 about 80 %. */
    static constexpr int kLendHold = 32;
    static constexpr int kLendInterval = 64;
    /* A lent plain step's first 8 cycles after the slab moved are no samples: the fill's copies
     * (about 2 GiB a rank) are still crossing the link. */
    static constexpr int kLendWarm = 8;
    /* A lent plain step's ratio to the unlent one is held to [0.5, 1.25]: one stalled step must not
     * price the lend. */
    static constexpr double kLentRatioMin = 0.5;
    static constexpr double kLentRatioMax = 1.25;
    /* The borrow (fn4 draft_lend_borrow): a class with no lent ratio of its own prices its plain step
     * with a neighbour's, so D7 can choose 0 where the lend would pay before the lend ever ran there
     * (c4: the plain step it weighs is the VRAM-reduced NST 5 one). From the nearest class with at
     * least kBorrowMin ratio samples (half the cost half-life: past the running mean's first noise);
     * at equal distance the lower class (fewer rows). The lent gain, 1 - ratio, is taken as it is from
     * a lower class and damped by kBorrowDamp per class step from a higher one: by the VRAM sweep's
     * slope over the ~237 units the lend returns, the gain is 6.1 % of the plain step at one row (1.0
     * of 16.4 ms), 10.5 % at four (3.1 of 29.3) and 10.7 % at eight (4.7 of 44.05, d0attr) -- it never
     * shrinks toward more rows, so a lower class's ratio is never optimistic upward, while downward it
     * shrinks by 0.98 for the c8 -> c4 step and 0.58 over the two c4 -> c1 steps (0.76 a step). 0.75 a
     * step stays at or under both: c8's 10.7 % prices c4 at 8.0 %, under its 10.5 %. */
    static constexpr int64_t kBorrowMin = 16;
    static constexpr double  kBorrowDamp = 0.75;
    /* The lend's state as the scheduler hands it to step_time: a stable resident drafter (every
     * sample as before), a stable lent slab past its warm-up (plain samples into the lent table,
     * drafting ones none), or a transition (no sample). */
    enum LendSample { kLendResident = 0, kLendTransit = 1, kLendLent = 2 };
    int  configure(const Params& p);
    const Params& params() const { return p_; }

    /* The draft depth for the next step; 0 means do not speculate. Constant: under the adaptive
     * policy it is the ceiling, and window_for() is the window. */
    int  next_depth() const { return p_.max_depth; }

    /* ---------------------------------------------------------- the adaptive policy */
    bool adaptive() const { return adapt_; }
    /* fn4 (runtime/fn4_switch.h): the policy on or off between steps. On needs a drafter that can
     * run it (Params::adapt_capable); off leaves the estimates, which a later on resumes from. */
    bool can_adapt() const { return (p_.adapt || p_.adapt_capable) && p_.max_depth > 0; }
    bool set_adaptive(bool on) {
        adapt_ = on && can_adapt();
        return adapt_ == on;
    }
    /* Depth 0 (fn4 draft_depth0): on or off between steps. Off returns to the drafting depth at
     * once and ends a probe; the measurements stay. */
    bool depth0() const { return depth0_; }
    void set_depth0(bool on) {
        if (on && !depth0_) zero_held_ = kHold;
        depth0_ = on;
        if (!on) { zero_ = false; probe_left_ = 0; since_probe_ = 0; backoff_k_ = 0; }
    }
    int  probe_every() const { return probe_every_; }
    void set_probe_every(int n) { probe_every_ = n > 0 ? n : kProbeEvery; }
    /* Probe back-off (fn4 draft_probe_backoff): the most doublings, 0 off; a change starts over. */
    int  backoff() const { return backoff_max_; }
    void set_backoff(int n) {
        backoff_max_ = n < 0 ? 0 : n > kBackoffMax ? kBackoffMax : n;
        backoff_k_ = 0;
    }
    int  backoff_k() const { return backoff_k_; }
    /* Cycles at 0 between probes now: probe_every << backoff_k, at most 4096. */
    int  probe_interval() const {
        if (backoff_k_ <= 0) return probe_every_;
        const int64_t p = (int64_t)probe_every_ << backoff_k_;
        return p > 4096 ? 4096 : (int)p;
    }
    /* The drafter's lend (fn4 draft_lend): 0 off, 1 on after kLendHold cycles, 2..4096 after that
     * many, -1 lent at every step whatever the depth (an identity check of the lent placement: every
     * drafting step then reads the drafter's experts over the link). lend() is the switch as set,
     * lend_hold() the cycles (0 off or forced). */
    int  lend() const { return lend_sw_; }
    int  lend_hold() const { return lend_sw_ <= 0 ? 0 : lend_sw_ == 1 ? kLendHold : lend_sw_; }
    void set_lend(int v) { lend_sw_ = v < -1 ? 0 : v > 4096 ? 4096 : v; }
    /* Whether the engine should lend the drafter's slab for the cycle just planned: depth 0 holds
     * (not a probe's choice) since lend_hold() cycles, and probes are at least kLendInterval apart;
     * or always, under -1. */
    bool lend_wanted() const {
        if (lend_sw_ < 0) return true;
        return adapt_ && depth0_ && lend_hold() > 0 && zero_ && zero_held_ >= lend_hold() &&
               probe_interval() >= kLendInterval;
    }
    /* Plain steps measured while lent, for tests and the report. */
    int64_t lent_samples(int n_rows) const { return n_rows >= 1 ? lent_n_[cost_class(n_rows)] : 0; }
    /* The plain step's expected ms for `n_rows` under the lend where measured (else the unlent one);
     * under the borrow, a class with no lent ratio of its own takes a neighbour's (kBorrowMin).
     * `borrowed`, if set, says whether it did. */
    double plain_ms(int n_rows, bool* borrowed = nullptr) const;
    /* The borrow (fn4 draft_lend_borrow): on or off between steps. */
    bool borrow() const { return borrow_; }
    void set_borrow(bool on) { borrow_ = on; }
    /* Depth-0 decisions whose plain step was priced with a borrowed ratio. For the report. */
    int64_t borrow_cycles() const { return n_borrow_; }
    /* Whether the head drafts this cycle: false only while depth 0 holds D at 0, or a plain probe
     * runs. A block drafter's context pass runs either way; its query pass and every verify row
     * do not. */
    bool head_runs() const { return p_.max_depth > 0 && !zero_now(); }
    /* Depth 0's counters: cycles planned at D = 0, cycles of probes, and moves between 0 and the
     * drafting depth. For tests and the fn4 report. */
    int64_t zero_cycles() const { return n_zero_; }
    int64_t probe_cycles() const { return n_probe_; }
    int64_t zero_moves() const { return n_zero_moves_; }
    /* The window this sequence may draft into: min(D, its stop), starting its estimate on the
     * first ask; max_depth when the policy is off. Before any step chooses D, D is the ceiling. */
    int  window_for(DraftAdapt& s) const;
    /* As window_for() but never starts an estimate; 0 if not started. For reporting. */
    int  window_of(const DraftAdapt& s) const;
    /* One verify of this sequence: `drafted` proposed, a prefix of `accepted` taken. Updates the
     * estimate, and the stop once past the hold. Returns the stop. */
    int  adapt_observe(DraftAdapt& s, int drafted, int accepted) const;
    /* A finished sequence: its estimate joins the prior that new sequences start from.
     *
     * Per sequence, not per verify. Under this policy, pooled accepted(j) / reached(j) is biased:
     * only well-predicted texts get windows deep enough to measure deep positions. Here each
     * finished sequence counts once at every position, using its edge extrapolation where it never
     * reached (adapt_observe), so shallow texts pull deep positions down as much as deep ones hold
     * them up. */
    void adapt_retire(const DraftAdapt& s);
    /* The prior a sequence starting now is given at position `j`. For tests and logs. */
    double prior_at(int j) const;

    /* D for the step being opened. `seq` holds the estimates of the `n_seq` drafting sequences
     * (started here if needed), `n_rows` all decode rows (the rest emit one token and draft
     * nothing). Moves D past the hold and hysteresis; with nothing to draft, D stays. Returns D. */
    int  plan_step(DraftAdapt* const* seq, int n_seq, int n_rows);
    /* The step depth in force: 0 before the first step chose one, and while depth 0 holds D at 0
     * (head_runs() tells the two apart). */
    int  step_depth() const {
        if (zero_now()) return 0;
        return probe_left_ > 0 && probe_d_ > 0 ? probe_d_ : step_d_;
    }
    /* The deepest window a step at depth D carries: D capped at the deepest stop it was chosen
     * over; the ceiling before the first choice or with the policy off. Proposals from outside
     * the head (prompt lookup) are held to it, so they add no deeper pass shape. */
    int  step_window() const {
        if (zero_now()) return 0;
        return adapt_ && step_w_ > 0 ? step_w_ : p_.max_depth;
    }

    /* T(depth, n): a decode step with deepest window `depth` and `n_rows` decode rows took `ms`
     * since the end of the previous step. Fed by Scheduler::step_finished; adaptive policy only.
     * Depth 0, a plain step at D = 0, only under depth 0. */
    void step_time(int depth, int n_rows, double ms, double tokens = -1.0, int lend = kLendResident);
    /* Expected cost of a step at `depth` with `n_rows` rows: the key's mean (ms) if it has
     * samples; else the nearest measured key scaled by the prior's ratio (1 + 0.06 d); else the
     * unitless prior itself. */
    double step_cost(int depth, int n_rows) const;
    /* Samples behind a key: for tests and logs. */
    int64_t cost_samples(int depth, int n_rows) const;

    /* E(d) for `s`'s current estimate: expected tokens emitted by a verify of d drafts, the
     * verified token included. For tests and logs. */
    double yield(const DraftAdapt& s, int d) const;
    /* Histogram of verifies by window, counted only under the adaptive policy. */
    int64_t verified_at(int d) const {
        return (d >= 0 && d <= RAD_SCHED_MAX_SPEC) ? by_depth_[d] : 0;
    }

    /* One verify's outcome: `drafted` proposed, a prefix of `accepted` taken. Metrics only. */
    void observe(int drafted, int accepted);

    /* Metrics (spec §14). Cumulative and exact. */
    int64_t drafted()  const { return tot_drafted_; }
    int64_t accepted() const { return tot_accepted_; }
    float   acceptance() const {
        return tot_drafted_ ? (float)((double)tot_accepted_ / (double)tot_drafted_) : 0.0f;
    }

    /* Per-position acceptance, a diagnostic rather than a policy input. The pooled ratio cannot
     * tell a drafter that decays gradually from one that fails at a specific position.
     *
     * `reached`: every earlier position was accepted, so this one was evaluated. `hits`: and it
     * was accepted. Their ratio is the conditional acceptance. Unreached positions are not
     * charged, so a rejection at position 0 does not count against later positions. */
    int64_t reached(int j) const {
        return (j >= 0 && j < RAD_SCHED_MAX_SPEC) ? reach_[j] : 0;
    }
    int64_t hits(int j) const {
        return (j >= 0 && j < RAD_SCHED_MAX_SPEC) ? hit_[j] : 0;
    }
    /* The deepest position any draft has reached; bounds the meaningful part of both arrays. */
    int     depth_seen() const { return seen_; }

private:
    Params  p_;
    int64_t tot_drafted_ = 0, tot_accepted_ = 0;
    int64_t reach_[RAD_SCHED_MAX_SPEC] = {};
    int64_t hit_  [RAD_SCHED_MAX_SPEC] = {};
    int     seen_ = 0;

    /* The adaptive policy as configure() resolved it. */
    bool    adapt_ = false;
    double  alpha_ = 0.0;                         /* the smoothing step from kHalfLife */
    int64_t by_depth_[RAD_SCHED_MAX_SPEC + 1] = {};
    int32_t step_d_ = 0;                          /* D; 0 until a step chose one */
    int32_t step_w_ = 0;                          /* min(D, the deepest stop it saw) */
    int32_t step_held_ = 0;                       /* steps since D last moved */
    /* Depth 0. step_d_ stays the drafting depth (the policy above, unchanged); zero_ says D is 0
     * instead. A probe runs probe_left_ more cycles at probe_d_ (0: a plain probe from drafting;
     * else the drafting depth, from 0). */
    bool    depth0_ = false;
    int32_t probe_every_ = kProbeEvery;
    bool    zero_ = false;
    int32_t zero_held_ = 0;                       /* cycles since zero_ last moved */
    int32_t probe_left_ = 0;
    int32_t probe_d_ = 0;
    int32_t since_probe_ = 0;                     /* cycles at 0 since the last probe */
    int64_t cycles_ = 0;                          /* cycles planned */
    int64_t plain_at_[kCostClasses] = {};         /* cycles_ at each class's last plain sample */
    int64_t plain_probe_at_ = 0;                  /* cycles_ at the last plain probe's start */
    int32_t zero_run_ = 0;                        /* consecutive cycles the head has not run */
    /* The drafting steps' MEASURED tokens and time per class (moving averages, any depth): what
     * depth 0 is weighed against besides the estimates' E(D) / T(D), which can be optimistic
     * (a new sequence starts from the prior). */
    double  dtok_[kCostClasses] = {};
    double  dms_[kCostClasses] = {};
    int64_t dn_[kCostClasses] = {};
    int64_t n_zero_ = 0, n_probe_ = 0, n_zero_moves_ = 0;
    /* Probe back-off and the lend (see the top of this file). lent_ratio_: plain steps measured
     * while lent over the unlent plain mean at the time (a ratio follows a growing context as the
     * unlent mean does); lent_ms_: their mean, for a class with no unlent plain step. */
    int32_t backoff_max_ = 0, backoff_k_ = 0;
    int32_t lend_sw_ = 0;
    double  lent_ratio_[kCostClasses] = {};
    double  lent_ms_[kCostClasses] = {};
    int64_t lent_n_[kCostClasses] = {};
    int64_t lent_rn_[kCostClasses] = {};
    bool    borrow_ = false;
    int64_t n_borrow_ = 0;
    /* The ratio class `c` borrows (kBorrowMin, kBorrowDamp), or 0 when no neighbour has one. */
    double  borrowed_ratio(int c) const;
    bool    zero_now() const {
        return adapt_ && depth0_ && (probe_left_ > 0 ? probe_d_ == 0 : zero_);
    }
    /* The prior from finished sequences (adapt_retire), and how many are behind it. */
    float   prior_[RAD_SCHED_MAX_SPEC] = {};
    int64_t prior_n_ = 0;
    /* T(d, class): the mean step time of each key, and the steps behind it. */
    double  cost_ms_[RAD_SCHED_MAX_SPEC + 1][kCostClasses] = {};
    int64_t cost_n_ [RAD_SCHED_MAX_SPEC + 1][kCostClasses] = {};

    /* Depth 0's half of plan_step: tok[d] the expected tokens at each depth, `plain` the rows
     * that draft nothing. */
    void    plan_zero(const double* tok, double plain, int n_rows);
    /* Fill a sequence's estimate from the prior and set its first stop. */
    void    start(DraftAdapt& s) const;
    /* The deepest draft whose chance of being emitted is at least kDraftMinChance. */
    int     stop_of(const float* a) const;
    static int cost_class(int n_rows);
};

}  /* namespace rad */
