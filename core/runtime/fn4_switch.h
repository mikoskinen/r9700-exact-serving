/* fn4_switch.h -- runtime switches for in-process A/B blocks (test builds only).
 *
 * Ported to the DeepSeek-V4.1 line (ds41-gg, 10.11.2026) from fn4-c16 2d532ecc / 91f566cd: the core machinery only
 * (the registry, POST /admin/fn4, the apply at the step release, the epoch in the pass key, draft_adaptive,
 * epoch_bump), without fn4's qwen4exp and hc_* switches, so an A/B of an exact lever runs in one process and one
 * load instead of two starts (start-to-start spread is ~1-1.5 % on one config).
 *
 * A switch is set from the environment at start and may be changed while the server runs through POST /admin/fn4.
 * A request is checked whole before anything is queued: an unknown switch, a value the owner of the switch does not
 * take (the plugin that serves it, the deployment's drafter), or a switch nothing serves is refused in the response
 * (400, the reason by name), and nothing of that body is queued. An accepted change is queued by the server thread and
 * applied by the scheduler thread at the next step release (Engine::prepare_release), while every rank thread is
 * parked, so all ranks issue each step under one setting. Every applied change advances an epoch that Ctx::pass_key
 * mixes in, so no recorded pass from before the change is played after it. That holds for an architecture with a
 * pass-key callback and for one without (deepseek41 exports none; the core keys its decode, verify and draft passes
 * by the bucketed bound): the epoch is mixed into the core's own key after the callback's part, whichever applies.
 * At epoch 0 nothing is mixed in, so with the environment unset and no request every key is today's to the bit and
 * the same passes are recorded.
 *
 * GET reports what is APPLIED, read from its owner, never only what was asked:
 *   draft_adaptive  scheduler: 1 the adaptive draft window, 0 fixed -- the scheduler's own state (DraftController::
 *                   adaptive()), registered at its init and read back after every change; -1 when no scheduler has
 *                   registered. A request takes 1 / 0 / -1 (= back to --draft-window); 1 is refused when the
 *                   deployment's drafter cannot run the adaptive window ("draft_capable": 0).
 *   gg_gemv         libr4d (rad_fn4_*): gemm_nt_ggml's decode band (up to 48 rows), 0 the stock rows-a-wave / grid
 *                   rule, 1 the measured per-(type, N, K, M) form table (r4d_ds41_gg_forms.h), 2 lever D2's
 *                   multi-row form (r4d_ds41_ggml_mr.hip, gemm_nt_ggml and logits_gemm_ggml; what it declines runs
 *                   the stock rule); every form has the stock kernel's per-output arithmetic, so the bytes are the
 *                   same whichever runs. The value is the one libr4d applied (it reads RADIANCE_DS41_GG_GEMV at load
 *                   and takes only the forms it knows; another value there is the stock rule, named on stderr); -1
 *                   when no loaded object serves the switch. A request for a value the serving plugin does not know
 *                   is refused.
 *   draft_depth0    scheduler: 1 the adaptive window may choose depth 0 (no head pass, no verify row) where a measured
 *                   plain step beats the window (sched/draft.h), 0 not. RADIANCE_DRAFT_DEPTH0=1 at start. Acts while the
 *                   adaptive window runs; 1 is refused when the drafter cannot run that window. 2 (a test hook, also
 *                   RADIANCE_DRAFT_DEPTH0=2): depth 0 FORCED -- every cycle at 0 except the probes at the drafting depth
 *                   every draft_probe cycles, nothing measured or weighed; accepted wherever 1 is. Exact (it moves only
 *                   what a step verifies), so a one-sequence run exercises depth 0's paths.
 *   draft_probe     scheduler: cycles at depth 0 between probes at the drafting depth (2 .. 4096; 16 by default,
 *                   RADIANCE_DRAFT_PROBE at start).
 *   draft_hybrid    scheduler: 1 a sequence whose prompt lookup matches at least draft_hybrid_match tokens back verifies
 *                   the lookup's proposal instead of the head's (Scheduler::hybrid_), 0 the head's alone.
 *                   RADIANCE_DRAFT_HYBRID=1 at start; 1 is refused without a head drafting chains (a tree is
 *                   verified only on a step issued ahead, and the hybrid plans the steps behind a long match
 *                   after the commit).
 *   draft_hybrid_match  scheduler: that match length (1 .. 256; LookupGate::kLongMatch = 8 by default,
 *                   RADIANCE_DRAFT_HYBRID_MATCH at start).
 *   draft_probe_backoff scheduler (sched/draft.h): 0 off; N (1 .. 8) each probe at the drafting depth that
 *                   leaves depth 0 in place doubles the interval to the next, up to draft_probe << N (at most
 *                   4096); a move between 0 and drafting starts it over. RADIANCE_DRAFT_PROBE_BACKOFF at
 *                   start. Exact: it changes only how many drafts a step verifies. 1 refused when the
 *                   drafter cannot run the adaptive window.
 *   draft_lend      scheduler + every rank's mover (lever DL, place/mover.h): 0 off; 1 on, the drafter's
 *                   expert slab lent to the trunk's experts once depth 0 has held 32 cycles with probes at
 *                   least 64 cycles apart (draft_probe_backoff or draft_probe), taken back when drafting
 *                   resumes; 2 .. 4096 on, after that many cycles at 0; -1 lent at every step whatever the
 *                   depth (an identity check of the lent placement). RADIANCE_DRAFT_LEND set at start
 *                   (any value, 0 included) declares it: every drafter unit keeps a pinned host copy
 *                   (outside --host-pool-mib, the plan unchanged) and its slab is carved for the trunk;
 *                   the value is the start value. Unset: nothing declared, and 1 is refused. Exact:
 *                   placement only. The counters are GET's "lend" object (state_json).
 *   draft_lend_borrow  scheduler (lever DL, sched/draft.h kBorrowMin / kBorrowDamp): 1 a cost class with no lent
 *                   plain step of its own prices its plain step with the nearest class's lent ratio (16 samples
 *                   at least; a lower class's as it is, a higher one's gain damped 0.75 a class step), 0 not.
 *                   RADIANCE_DRAFT_LEND_BORROW=1 at start. 1 refused unless the lend is set up. Exact: it moves
 *                   only D7's choice of depth.
 *   drafter_mx      libr4d (rad_fn4_*): moe_gemm_ggml type=mxfp4 (the DSpark drafter's routed experts), 0 the stock
 *                   kernels, 1 the multi-row forms (r4d_ds41_mxfp4_mr.hip) in the stock path's summation order, the
 *                   same bits; RADIANCE_DS41_DRAFTER_MX=1 at load. -1 when no loaded object serves it.
 *   drafter_bf      libr4d (rad_fn4_*): gemm_nt bf16's decode band (the drafter's attention / main_proj / shared
 *                   expert), 0 the stock skinny WMMA kernel, 1 its prefetching form (r4d_gemm_bf16_nt_pf.hip), the
 *                   same chains and so the same bits; RADIANCE_DS41_DRAFTER_BF=1 at load.
 *   aql_acq         core (Ctx::haz_mark + the AQL replay): the address-based acquire rule in force, 0 off, 1 rule,
 *                   2 rule-w -- the switch once set, else RADIANCE_AQL_ACQ=rule | rule-w at start (ds41-ar-acq
 *                   4c9e6c2's start-time rule, now read through here). A change re-records every pass (the epoch),
 *                   since haz_mark marks the records at recording. RADIANCE_AQL_ACQ=none stays start-time only.
 *   idx_split       deepseek41 (rad_fn4_*; lever I1): the indexer row split's deal, 0 the plain deal (ceil(n / world)
 *                   runs a rank), 1 the balanced deal where the world does not divide a step's runs (an equal block of
 *                   rows a rank; arch/deepseek41/ds41_index_split.h). Exact: every row's selection is the same and the
 *                   gather lays it in the same place. RADIANCE_DS41_IDX_BALANCE=1 at start; 1 is refused where no rank
 *                   declared the row split. -1 when no loaded object serves it.
 *
 *   ds41_argather   deepseek41 (rad_fn4_*; lever D3): the MoE gather + all-reduce fusion's live row bound,
 *                   0..the bound the layers declared (RADIANCE_DS41_ARGATHER / _ARGATHER_MAX): a step of at most
 *                   that many rows issues moe_gather_all_reduce, a wider one moe_gather + all_reduce. Every form
 *                   is the same fold, so the bytes are the same at any value; a value past the declared bound is
 *                   refused. -1 when no loaded object serves it.
 *   ds41_shexp_lane deepseek41 (lever F4): 1 the shared expert on the second lane at decode / verify steps, 0 one
 *                   lane (RADIANCE_DS41_SHEXP_LANE at start); 1 is refused where no layer's arm is lane-safe.
 *   ds41_kvfold     deepseek41 (lever K1 fold c): 1 the window key's rmsnorm_rope_roundtrip_kv_store (norm, rope,
 *                   round trip and the cache store in one launch), 0 rmsnorm_rope_roundtrip + kv_store as before.
 *                   RADIANCE_DS41_KVFOLD=1 declares the fold beside the pair (and starts at 1); unset declares
 *                   nothing and 1 is refused. Exact: the same bytes in y and the cache.
 *   ds41_ofold      deepseek41 (lever K1 fold d): 1 attn_sparse_sink_rope_roundtrip (the attention output's inverse
 *                   rope and round trip in the attention's epilogue at decode heights), 0 attn_sparse_sink +
 *                   rope_rows_roundtrip. RADIANCE_DS41_OFOLD=1 declares it (starts at 1); unset: 1 refused. Exact.
 *   ds41_onefill    deepseek41 (lever K1 fold f): 1 the ungated shared expert's gate plane of ones filled once a
 *                   step (MoeFP8 shared_one_external), 0 by every MoE layer as before. RADIANCE_DS41_ONEFILL=1
 *                   declares the step's fill (starts at 1); unset: 1 refused. Exact: the gathers read the same ones.
 *
 *   ds41_qnfold     deepseek41 (lever K1 fold a): 1 the query's rmsnorm (q_a_norm) computed in q_b's GEMV activation
 *                   staging (rmsnorm_gemm_nt_ggml, decode band), 0 rmsnorm + gemm_nt_ggml as before.
 *                   RADIANCE_DS41_QNFOLD=1 declares the fold beside the pair (and starts at 1); unset declares nothing
 *                   and 1 is refused. Exact: the same bytes in q (and in the normed rows the indexer reads).
 *   ds41_swfold     deepseek41 (lever K1 fold e): 1 the shared expert's swiglu_clamped computed in its down GEMV's
 *                   activation staging (swiglu_clamped_gemm_nt_ggml, decode band), 0 swiglu_clamped + gemm_nt_ggml.
 *                   RADIANCE_DS41_SWFOLD=1 declares it (starts at 1); unset: 1 refused. Exact.
 *   draft_ctxskip   the core's (a windowed block drafter; the depth-0 context skip): 1 a block drafter's context pass
 *                   is skipped on a plain decode step (every row one decode token) while the head does not run (depth 0,
 *                   not a probe), the trunk's tapped rows kept in a ring on the card, and the drafter's window refilled
 *                   from them (context passes over the kept rows, the last `window` a sequence) before the head next
 *                   runs, on any step that cannot skip, when the ring is full, before a checkpoint carrying the drafter's
 *                   window, and after a switch back to 0; 0 the context pass every step as before.
 *                   RADIANCE_DRAFT_CTXSKIP=0 | 1 at start sets it up (the ring allocated) and starts at that value;
 *                   unset sets nothing up, the issue is today's, and 1 is refused. Exact for the target: the drafter
 *                   only proposes and every emitted token is the target's (core/engine_draft.cpp).
 *
 *   dspark_conf     engine (ds41-acc, lever C1): what a chain block drafter's confidence plane does
 *                   (rad_drafter_confidence; deepseek41 with RADIANCE_DS41_DSPARK_CONF): 0 nothing (the plugin
 *                   issues no head, the engine reads nothing: today's passes); 1 the reference rule -- each
 *                   sequence verifies the longest prefix of its proposal whose sigmoid(logit) >= 0.5, in place
 *                   of the adaptive window's stop; 5..95 the same at threshold v / 100; 101, 105..195 the same
 *                   rule capped by the window (min of the two; 101 = 0.5); 2 observe -- the head runs and its
 *                   calibration is counted (/stats draft_conf), nothing is cut. A gate waits for the query
 *                   pass before the step issued ahead is built (the cut is a row count). Exact: it changes how
 *                   many drafts a verify carries, never a proposal or an emitted token. Nonzero is refused
 *                   where no confidence plane is declared.
 *   dspark_alt      engine (ds41-acc, lever T1): 1 the drafter's alternative branches (rad_drafter_alternatives;
 *                   deepseek41 with RADIANCE_DS41_DSPARK_ALT) are computed and compared with the tokens the
 *                   target emits -- a shadow of a tree that branches at the first draft, counted under
 *                   /stats draft_conf.shadow, never verified; 0 nothing. 1 refused where none is declared.
 *   dspark_conf_rows engine (ds41-small: dspark_conf's load gate): 0 the confidence plane acts at every drafter
 *                   cycle (as ds41-acc measured it); N in 1..64 only at a cycle whose step decoded at most N
 *                   sequences -- above that the engine neither reads the plane nor waits for it (no cut, no host
 *                   wait before the step issued ahead), as at dspark_conf 0. Exact as dspark_conf: verify counts
 *                   only. Nonzero is refused where no confidence plane is declared.
 *
 * "epoch_bump": 1 in a request starts a new epoch without changing a switch, so an A/A block pair re-records
 * exactly as an A/B pair does.
 *
 * A plugin takes part by exporting (each optional but the first)
 *   extern "C" int rad_fn4_switch(const char* name, int value)    apply: 1 applied, 0 not its switch, < 0 refused
 *   extern "C" int rad_fn4_accepts(const char* name, int value)   check only: 1 it would apply it, 0 not its switch,
 *                                                                 < 0 its switch but not that value
 *   extern "C" int rad_fn4_value(const char* name, int* applied)  1 and the value it applies, 0 not its switch
 * The kernel and architecture plugins are dlopen'ed RTLD_LOCAL, so the core finds the hooks in every loaded object
 * through its own handle (fn4_switch.cpp). */
#ifndef RAD_RUNTIME_FN4_SWITCH_H
#define RAD_RUNTIME_FN4_SWITCH_H

#include <cstdint>
#include <string>

namespace rad {
namespace fn4 {

enum Switch { kDraftAdaptive = 0, kGgGemv, kAqlAcq, kDs41Argather, kDs41ShexpLane, kDs41KvFold, kDs41OFold,
              kDs41OneFill, kDraftDepth0, kDraftProbe, kDraftHybrid, kDraftHybridMatch, kDraftProbeBackoff,
              kDraftLend, kDraftLendBorrow, kDrafterMx, kDrafterBf,
              kIdxSplit, kDs41QnFold, kDs41SwFold, kDraftCtxSkip,
              kDsparkConf, kDsparkAlt, kDsparkConfRows, kSwitches };
/* The switches the scheduler owns: apply_pending hands their new values over, and the scheduler reports what it
 * runs (note_sched). */
bool sched_switch(int i);

const char* switch_name(int i);
int  switch_index(const std::string& name);   /* -1 when unknown */
int  value(int i);                            /* the applied value, as GET reports it */
uint64_t epoch();                             /* applied changes so far; 0 until the first */
/* The AQL acquire rule: 0 off, 1 rule, 2 rule-w -- the aql_acq switch once set, else RADIANCE_AQL_ACQ (rule |
 * rule-w) at start. aql_acq_none: RADIANCE_AQL_ACQ=none (the unsafe experiment; start-time only). */
int  aql_acq_rule();
bool aql_acq_none();

/* The scheduler, at its init: whether its drafter can run the adaptive window, and whether it does. */
void draft_register(bool capable, bool adaptive);
/* The scheduler, at its init: whether its drafter can run the hybrid (a head drafting chains beside prompt lookup). */
void hybrid_register(bool capable);
/* The scheduler thread, after it applied a draft_adaptive change: the window the scheduler now runs. */
void note_draft_adaptive(bool adaptive);
/* The scheduler, at its init and after it applied a change: the value it runs for one of its switches. */
void note_sched(int i, int value);
/* The scheduler's draft counters for the report (GET "draft"): cycles at depth 0, probe cycles, moves between 0 and
 * the drafting depth, hybrid proposals taken from the lookup and left to the head, steps not issued ahead for it. */
void note_draft_counts(int64_t zero, int64_t probe, int64_t moves, int64_t hyb_lookup, int64_t hyb_head,
                       int64_t hyb_unahead);
/* Lever DL. The scheduler, once the engine set the lend up: whether a slab can be lent (draft_lend=1 is
 * refused otherwise). */
void lend_register(bool capable);
/* The scheduler, after every step: cycles planned while lent; decode steps while lent, plain and drafting
 * (a probe reading the drafter's experts over the link), and steps while the slab moved, each with their wall
 * ms; the probe back-off's doublings now and the probe interval they give. */
void note_lend_sched(int64_t lent_cycles, int64_t plain_steps, int64_t plain_ms, int64_t probe_steps,
                     int64_t probe_ms, int64_t transit_steps, int64_t transit_ms, int64_t backoff_k,
                     int64_t probe_interval, int64_t borrow_cycles = 0);
/* The engine, at every step release: the lend's state over the ranks (0 resident, 1 moving, 2 lent), rank 0's
 * lends and recalls begun, trunk units in the drafter's memory, the drafter's units in VRAM and in all, and
 * the drafter's slab in MiB. */
void note_lend_engine(int64_t state, int64_t lends, int64_t reclaims, int64_t lent_units, int64_t draft_resident,
                      int64_t draft_units, int64_t slab_mib);
/* The engine, at its init: whether its drafter declared a confidence plane / alternative branches, and the
 * switches' values at start (0 when not capable). */
void dspark_register(bool conf_capable, bool alt_capable, int conf_start, int alt_start);
/* Whether `v` is a dspark_conf value (see above). */
bool dspark_conf_value_ok(int v);
/* The applied value of a core switch without the lock (the step path reads it each step); -1 unregistered. */
int  get(int i);
/* The scheduler, after every step: drafts verified and accepted, the head's and prompt lookup's apart (GET
 * "accept"; under draft_depth0 2 every head draft is a probe's). */
void note_draft_accept(int64_t head_drafted, int64_t head_accepted, int64_t lookup_drafted, int64_t lookup_accepted);
/* The depth-0 context skip (draft_ctxskip). The engine, at bring-up: whether it set the skip up (draft_ctxskip=1 is
 * refused otherwise; GET "ctxskip" only once registered). */
void ctxskip_register(bool capable);
/* The engine, after every block drafter's turn: steps whose context pass was skipped, refill passes and their
 * rows, kept rows dropped (a sequence finished or preempted, or older than the drafter's window), refills forced
 * by a full ring, rows kept now, and the ring's step slots and MiB a rank. */
void note_ctxskip(int64_t skipped, int64_t refill_passes, int64_t refill_rows, int64_t dropped_rows,
                  int64_t ring_full, int64_t pending_rows, int64_t ring_slots, int64_t ring_mib);

/* Whether a request may set `name` to `value` (epoch_bump included); false with the reason. */
bool check(const std::string& name, int value, std::string* err);
/* Server thread: queue a value; false (with the reason) for anything check() refuses. */
bool request(const std::string& name, int value, std::string* err);
/* Scheduler thread, every rank thread parked: apply what is queued. Returns true when anything
 * changed; sched[i] (kSwitches entries) is the value to set for each scheduler switch, kNoChange
 * where it did not change: draft_adaptive 1 adaptive, 0 fixed (a request for -1 arrives here as the
 * configured window), the others their value. */
constexpr int kNoChange = -2;
bool apply_pending(int step, int* sched);
/* {"switches": {...}, "pending": {...}, "epoch": n, "applied_step": n, "plugins": n, "draft_capable": n,
 *  "draft": {...}, "accept": {...}, "lend": {...}, "ctxskip": {...}} -- "lend" only once a scheduler registered
 *  (lend_register), "ctxskip" only once the engine did (ctxskip_register). */
std::string state_json();

}  // namespace fn4
}  // namespace rad

#endif
