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
 *   draft_lend_w    engine (lever Z0-W, Mover::draft_w_setup): 1 the drafter's static weights that only its query
 *                   pass reads (each block's q / o / shared expert / router / norms, the Markov walk's, the confidence
 *                   head) are lent to the trunk's experts while the slab's lend is wanted and a step runs no head,
 *                   recalled and copied back from a pinned host copy before a step whose head runs; 0 not.
 *                   RADIANCE_DRAFT_LEND_WEIGHTS at start declares it (needs RADIANCE_DRAFT_LEND); its value (1) starts
 *                   it on. 1 refused unless declared and set up on every rank. Exact: placement only -- the drafter
 *                   reads its weights' own bytes, copied back, at their own addresses. GET "lend" counts w_lends,
 *                   w_reclaims, w_lent_units, w_mib.
 *   arena_d0        engine (lever Z0-A, Engine::arena_step): 0 off -- decode and verify steps take the largest decode
 *                   rung of RADIANCE_DS41_ARENA_DECODE_ROWS (today's choice); 1 a decode step that verifies nothing and
 *                   runs no head (depth 0) takes the smallest rung holding its own rows (with "8,48": 8 at c8 depth 0
 *                   instead of 48), so the arena above it is lent to the expert slab as at NST 0. Starts 0. 1 refused
 *                   unless the variable declared a rung below the largest (arena_d0_register). Exact: placement only.
 *   draft_occ       scheduler (lever Z0-O, sched/draft.h DraftController::occ): 0 off (D7 as it was); 1 .. 64 = N: once
 *                   at least N decode rows have run draft_occ_k consecutive cycles, depth 0 holds with no probe; once
 *                   fewer than N have run draft_occ_k consecutive cycles, D7 decides again (its cost model and probes:
 *                   the 1 .. N-1 row region). Needs depth 0 on (draft_depth0 1); depth 0 forced (2) is left as it is.
 *                   Starts 0 (no environment). Above 0 refused without a drafter that runs the adaptive window. Exact:
 *                   it moves only D7's choice of depth. GET's "draft" object counts occ_cycles and occ_moves.
 *   draft_occ_k     scheduler (lever Z0-O, DraftController::occ_k): the consecutive cycles draft_occ counts each way,
 *                   1 .. 4096; starts 32 (DraftController::kOccK).
 *   place_lf        engine (lever LF, ds41-lf; Mover::fill): 0 off -- free lent slots (a KV / arena / drafter loan's) are
 *                   filled hottest pooled unit first by the heat engine's placement heat, as today; 1 by its long-memory
 *                   decode heat (the decode heat's credit at decay RADIANCE_PLACE_LF_DECAY, 0.998 a step: how often decode
 *                   routed an expert over the last few hundred steps), the same tie-break, slots and copies; every drop,
 *                   recall and swap as today. RADIANCE_PLACE_LF=1 at start starts it on. 1 refused unless every rank
 *                   places dynamically. Exact: placement only (which expert sits in which slot). /stats
 *                   experts.ranks[] carries the slot classes' hits / misses and each lender's occupancy, fills and drops.
 *   draft_lend_hyst scheduler (lever Z0-H, sched/draft.h DraftController::lend_hyst): 0 off (kHyst, 2 %, throughout);
 *                   1 .. 100: while the drafter's slab is lent or the lend is wanted, a move from depth 0 to drafting
 *                   needs the window's yield to beat the lent plain step's by that many percent (at least kHyst); a
 *                   move back to 0 and every move while not lent keep kHyst. Starts 0 (no environment). Above 0
 *                   refused unless the lend is set up (RADIANCE_DRAFT_LEND at start). Exact: it moves only D7's
 *                   choice of depth. GET's "lend" object counts the cycles it held 0 (hyst_holds).
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
 *   gg_pf           libr4d (rad_fn4_*; lever GPF): gemm_nt_ggml / logits_gemm_ggml's prefill WMMA rows (q5_K / q6_K /
 *                   q8_0 above the decode band), 0 the stock kernel (gg_wm_kernel), 1 the per-shape form table
 *                   (RADIANCE_DS41_GG_PF_TABLE at load, else the built-in one; a shape it does not name runs the
 *                   stock kernel), or a form code (r4d_ds41_ggml_pf.hip: 100 MT + 10 NT + WN / 4) for every shape.
 *                   Every form keeps the stock kernel's per-output WMMA chains and scale fmas, so the bytes are the
 *                   same; RADIANCE_DS41_GG_PF at load. -1 when no loaded object serves it.
 *   f32_pf          libr4d (rad_fn4_*; lever GPF-b): gemm_nt_f32's prefill rows (the router and the compressor's kv | gate
 *                   projection above 63 rows), 0 the stock tile kernel, 1 the tiled form (r4d_ds41_f32pf.hip, form 20),
 *                   20.. a form code. Every form keeps each output's 32 lane-strided fmaf chains and the xor butterfly,
 *                   so the bytes are the same for f32 and bf16 weight storage; RADIANCE_DS41_F32_PF at load.
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
 *   dspark_walk     deepseek41 (lever W1): 1 the DSpark Markov walk over the whole vocabulary (dspark_walk_step a
 *                   step on each rank's vocabulary shard + an all_gather of the keys; one rank: dflash_markov_greedy_f32),
 *                   0 the head's top 32 re-ranked (dflash_select) as before; 2 the walk of 1 with libr4d's form 1 (lever
 *                   W1b: succ staged through LDS, the same bits). RADIANCE_DS41_DSPARK_WALK=0 / 1 / 2 declares the walk
 *                   (and starts at that value); unset: 1 / 2 refused. Exact: proposals only, the trunk verifies.
 *
 *   ds41_bi         deepseek41 (lever BI, batch invariance): 1 a step's rows are issued in segments by their own
 *                   sequence's shape -- the decode rows as one segment (dense GEMMs, attention and the indexer score
 *                   in one launch up to 48 rows, the fused norms in launches of at most 32), each prefill sequence as its
 *                   own segment at its own rows (attention and the indexer through a two-entry cu view), the MoE one pass
 *                   a segment -- so a row's arithmetic is the one its sequence gets alone; 0 one launch over the step as
 *                   before; 2 (arm BI-P) = 1 + the routed MoE over the decode rows in passes of at most 16 (the EXL3
 *                   experts' decode band), so a c4 / c8 verify's rows keep their alone bytes too, at a MoE pass a piece.
 *                   3 / 4 = 1 / 2 with the cheap issue (the query / key chains op by op across the segments, the
 *                   per-row ops that never flip once over the step). RADIANCE_DS41_BI=0..4 declares it (and starts at that
 *                   value); unset: 1..4 refused. A step with
 *                   one segment runs today's issue either way. Changes the bytes of rows that shared a step with other
 *                   sequences (toward their alone bytes); a one-sequence step is unchanged.
 *   sched_bi_chunk  scheduler (lever BI, class (ii)): how a prefill's chunk boundaries are planned beside other work
 *                   (core/sched/scheduler.cpp bi_chunk_plan): 0 today's (budget cuts, busy-chunk share); 1 alone's
 *                   boundaries, the step budget --max-num-batched-tokens; 2 the checkpoint-interval grid, alone too;
 *                   3 / 4 alone's pieces, beside other work cut on the interval / RADIANCE_SCHED_BI_GRID (1024) grid
 *                   inside them, >= 128 rows a side. RADIANCE_SCHED_BI_CHUNK at start; unset = 0. The scheduler reads
 *                   it at every step it plans, and a change is applied at a step release (after that step was
 *                   planned), so the value takes effect from the next planned step: an A/B pins it, runs one short
 *                   request, then measures. A prefill in flight keeps nothing of the old value's plan except the rest
 *                   of an alone piece 3 / 4 cut in two, which runs whole under any value.
 *   sched_interleave scheduler (lever DST): --prefill-interleave's count, the decode-only steps (admitting nothing)
 *                   after a prefill chunk that ran beside decoding sequences and left its prompt unfinished
 *                   (Scheduler::step, interleave_left_). 0 .. 64 (the flag's range): 0 = no hold, a chunk may ride
 *                   every step; -1 = back to the flag (which may be auto). Unset: the flag; GET reports the flag's
 *                   value (-1 = auto) until a value is set. Read at every step planned; applied at a step release,
 *                   so it acts from the next planned step: a hold under way is cut to the new count when that is
 *                   smaller (0 ends it), a larger count arms after the next chunk. Timing only: it moves which
 *                   step carries a chunk, never a chunk's own plan rules (the chunk's rows may still differ with
 *                   what that step carries, as today's cuts beside decode rows do; sched_bi_chunk 3 / 4 keep them
 *                   on the grid).
 *   sched_mixed_tok scheduler (lever DST): the token budget of a step that is not a lone prefill's span step (the
 *                   shared budget, ChunkGeometry::shared_tok, 2112 at MNBT 4160 / interval 2048 / span 2): 128 ..
 *                   shared_tok, a multiple of the quantum; 0 = back to shared_tok (the start value; GET reports
 *                   the budget in force). Only under --prefill-span > 1 (refused otherwise) and only smaller: the
 *                   plugin's arenas were sized for shared_tok (RadBuildCtx::max_mixed_tok). Under sched_bi_chunk 0
 *                   the chunk beside decode rows is cut as today from the smaller budget (budget less the rows
 *                   ahead, the busy-chunk share; off the interval grid); under 3 / 4 (option D) the cut grid inside
 *                   alone's piece is min(G, the largest divisor of the interval, a multiple of the quantum, that
 *                   leaves 64 rows of the budget to decode rows) -- 2112 / 1088 / 576 give 2048 / 1024 / 512-row
 *                   mixed chunks on the grid, >= 128 rows a side; under 1 (A: the step budget stays MNBT) and 2
 *                   (B: whole intervals) it has no effect. Read at every step planned; a prefill in flight keeps
 *                   its position and takes its next chunk at the new budget (an alone piece 3 / 4 cut in two
 *                   keeps its piece; the rest is cut on the new grid).
 *   sched_short_bypass scheduler (lever DST): 0 off (today); N (a multiple of the quantum, up to
 *                   --max-num-batched-tokens; 1280 / 2112 the A/B values): on a step the interleave holds (decode rows
 *                   only, no admission today), a WAITING request is admitted when its uncached rest (after its prefix
 *                   hit, as an admission computes it) is at most N rows AND is one whole piece of its alone plan (the
 *                   chunk an idle engine plans from the same position: span, interval, tail checkpoint) AND fits what
 *                   the step has left; it runs that one piece, so its rows are cut exactly as alone. Everything else
 *                   waits as today (a request turned away for length is not asked again on held steps); running
 *                   prefills stay held, and the held step counts toward the hold as a decode-only step, so the long
 *                   prefill's chunks and their timing are unchanged by it (under sched_bi_chunk 4 a request no longer
 *                   waiting at the long prefill's next chunk takes no grid step from it: that cut can be the larger grid
 *                   point). Composes with sched_bi_chunk 0 .. 4 and ds41_bi's per-sequence segments. Read at every step
 *                   planned; GET "dst" counts the bypassed and the turned-away admissions.
 *   sched_bypass_total scheduler (lever DST, with sched_short_bypass N > 0): 0 = the bypass takes a request only when its
 *                   uncached rest is ONE alone piece (above); T (a multiple of the quantum, up to 65536; 4096 the A/B value)
 *                   = it takes a waiting request whose uncached rest is at most T rows and then runs it piece by piece,
 *                   held steps included, until its prompt is done: each piece the next piece of its alone plan when that
 *                   is at most N and fits the step (alone's boundaries and checkpoints); under sched_bi_chunk 3 / 4 an
 *                   alone piece that does not is cut on the checkpoint-interval grid inside it, >= 128 rows a side
 *                   (option D's own rule, so exact as D is: alone's span pieces of a long conversation's tail are 4096
 *                   rows, over a held step's ~2112). Under 0 / 1 / 2 such a request is turned away (an ordinary
 *                   admission). With either switch back to 0 a request taken this way continues as an ordinary prefill.
 *   ds41_psk        deepseek41 (lever PSK, prefill skip above the last cache source): 1 a prefill sequence's chunk runs
 *                   the layers past the last kv source (21..39 of the 1017's 40) over its last 160 + (39 - l) x 128 rows
 *                   only -- the rows the chunk's end reads through the layers' 128-key windows (its window state, the
 *                   logits row, the DSpark taps) -- every row through the last source; 0 every row as before.
 *                   RADIANCE_DS41_PSK=0 | 1 declares it (and starts at that value); unset: 1 refused, today's issue.
 *                   Changes no row the step's end reads (the cut rows' streams feed nothing later); a computed row
 *                   sees a launch of >= 160 rows, above every row-count form switch -- bytes to be measured.
 *                   Design 2 (ds41-psk2): 2 = anchors -- the scheduler names, per prefill sequence, the positions its
 *                   window state must be whole at (the prompt's end, its tail snapshot, every ds41_psk_k-th periodic
 *                   checkpoint position, checkpoint_at), reserves window snapshots there only, and hands each step's plan
 *                   to the plugin (rad_psk_plan); a chunk runs layers 21..39 only over those targets' cones (none: the
 *                   layers are skipped whole, and their experts not staged); 3 = 2 + an anchor at the first checkpoint
 *                   position at or past 32768. Needs RADIANCE_DS41_BI declared (2 / 3 refused otherwise). A request takes
 *                   the mode and K at admission and keeps them while it prefills.
 *   ds41_psk_k      scheduler (lever PSK design 2): the anchor stride K, 1 .. 4096 (default 8): the K-th, 2K-th, ...
 *                   periodic checkpoint position (the exact grid's points from its origin, else the interval's
 *                   multiples) is a target under ds41_psk 2 / 3. Read at a request's admission.
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
 *   draft_sample    engine (lever SD, sampled drafting, 11.11.2026; the user's GO: distribution-exact, NOT byte-identical
 *                   to non-speculative sampling at temperature > 0): 0 today's rule -- greedy drafts, a draft accepted
 *                   when it equals the target's own sampled draw at its RNG coordinate (core/sample/accept.h); 1 a sampled
 *                   request's drafts are drawn from the drafter's law Q (deepseek41: the W1 walk's eight best keys,
 *                   softmax at the request's temperature, x ~ Q at the request's seed, r4d_ds41_wsamp_math.h) and the
 *                   verify keeps x with probability min(1, P(x) / Q(x)), else draws from norm(max(0, P - Q)) (the
 *                   sampler's mtp_couple_verify); 2 .. 200 the same with Q at the request's temperature x v / 100.
 *                   A greedy request (temperature 0 / top_k 1) drafts and verifies as at 0 at every value, so
 *                   temperature 0 stays byte-identical. A prompt-lookup proposal never couples. Needs --draft-coupling
 *                   on and a drafter whose walk declares the coupling planes (refused otherwise); RADIANCE_DRAFT_SAMPLE
 *                   at start (unset: 0). The counters are GET's "sample" object.
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
              kDsparkConf, kDsparkAlt, kDsparkConfRows, kGgPf, kF32Pf, kDsparkWalk, kDs41Bi, kDraftSample,
              kDraftLendHyst, kArenaD0, kDraftLendW, kDraftOcc, kDraftOccK, kPlaceLf, kSchedBiChunk,
              kSchedInterleave, kSchedMixedTok, kSchedShortBypass, kSchedBypassTotal, kDs41Psk, kDs41PskK, kSwitches };
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
                       int64_t hyb_unahead, int64_t occ_cycles = 0, int64_t occ_moves = 0);
/* Lever DL. The scheduler, once the engine set the lend up: whether a slab can be lent (draft_lend=1 is
 * refused otherwise). */
void lend_register(bool capable);
/* The scheduler, after every step: cycles planned while lent; decode steps while lent, plain and drafting
 * (a probe reading the drafter's experts over the link), and steps while the slab moved, each with their wall
 * ms; the probe back-off's doublings now and the probe interval they give. */
void note_lend_sched(int64_t lent_cycles, int64_t plain_steps, int64_t plain_ms, int64_t probe_steps,
                     int64_t probe_ms, int64_t transit_steps, int64_t transit_ms, int64_t backoff_k,
                     int64_t probe_interval, int64_t borrow_cycles = 0, int64_t hyst_holds = 0);
/* The engine, at every step release (lever Z0-W): whether the weights are lent now, lends and recalls so far,
 * trunk units in their memory, and the lendable bytes in MiB (rank 0's). */
void note_lend_w(int64_t lent, int64_t lends, int64_t reclaims, int64_t lent_units, int64_t mib);
/* The engine, at every step release: the lend's state over the ranks (0 resident, 1 moving, 2 lent), rank 0's
 * lends and recalls begun, trunk units in the drafter's memory, the drafter's units in VRAM and in all, and
 * the drafter's slab in MiB. */
void note_lend_engine(int64_t state, int64_t lends, int64_t reclaims, int64_t lent_units, int64_t draft_resident,
                      int64_t draft_units, int64_t slab_mib);
/* The engine, at its init: whether its drafter declared a confidence plane / alternative branches, and the
 * switches' values at start (0 when not capable). */
void dspark_register(bool conf_capable, bool alt_capable, int conf_start, int alt_start);
/* Lever SD. The engine, at its init: whether --draft-coupling is live with a drafter that declared the coupling
 * planes (draft_sample >= 1 is refused otherwise), and the start value (RADIANCE_DRAFT_SAMPLE). */
void sample_register(bool capable, int start);
/* The engine, after every drafter cycle that ran the sampled walk: drafting rows staged sampled (mode 1) and greedy,
 * rounds bound to the verify, rounds the walk marked status -1, rows whose readback could not bind (cleared: the
 * ordinary rule applies to them), and prompt-lookup proposals whose coupling was cleared before a build. */
void note_sample(int64_t rows_sampled, int64_t rows_greedy, int64_t rounds_bound, int64_t rounds_malformed,
                 int64_t bind_fallback, int64_t lookup_cleared);
/* Whether `v` is a dspark_conf value (see above). */
bool dspark_conf_value_ok(int v);
/* The applied value of a core switch without the lock (the step path reads it each step); -1 unregistered. */
int  get(int i);
/* Lever BI's chunking (sched_bi_chunk): the switch once set, else RADIANCE_SCHED_BI_CHUNK (0 .. 4) at start, else 0.
 * The scheduler reads it at every step it plans. */
int  sched_bi_chunk();
/* Lever DST. The scheduler, at its init: --prefill-interleave as configured (-1 auto), the shared budget a non-lone step
 * is planned at (ChunkGeometry::shared_tok), whether a shared step exists (--prefill-span > 1), and the chunk quantum.
 * sched_interleave(): the switch once set (0 .. 64), else -1 (the flag applies). sched_mixed_tok(): the switch once set
 * (a budget), else 0 (shared_tok applies). The scheduler reads both at every step it plans. */
void sched_register(int interleave_flag, int64_t shared_tok, bool shared_capable, int64_t quantum, int64_t max_tok = 0);
int  sched_interleave();
int  sched_mixed_tok();
/* Lever DST, sched_short_bypass: the switch once set, else 0 (off). The scheduler reads it at every step it plans, and
 * after every step reports its counts (GET "dst"): requests the bypass admitted on a held step, and requests it turned
 * away because their uncached rest is longer than N or is not one piece of their alone plan. */
int  sched_short_bypass();
/* sched_bypass_total: the switch once set, else 0 (the bypass takes one-piece requests only). */
int  sched_bypass_total();
void note_sched_dst(int64_t bypassed, int64_t bypass_refused);
/* Lever PSK design 2: hand the scheduler's plan for trunk pass `step` to the plugin that cuts by it (rad_psk_plan, every
 * rank thread parked): per prefill sequence `ahead` and chunk rows, and the step's decode rows. Returns the first layer
 * the pass issues nothing from (the plugin's n_layer when every layer runs), -1 when no plugin takes a plan. */
int  psk_plan(int step, int64_t dec_tok, int n, const int32_t* ahead, const int32_t* rows);
/* The scheduler, after every step: drafts verified and accepted, the head's and prompt lookup's apart (GET
 * "accept"; under draft_depth0 2 every head draft is a probe's). */
void note_draft_accept(int64_t head_drafted, int64_t head_accepted, int64_t lookup_drafted, int64_t lookup_accepted);
/* The depth-0 context skip (draft_ctxskip). The engine, at bring-up: whether it set the skip up (draft_ctxskip=1 is
 * refused otherwise; GET "ctxskip" only once registered). */
void ctxskip_register(bool capable);
/* The engine, at bring-up (lever Z0-A): whether arena_d0 can act (a decode rung below the largest declared). */
void arena_d0_register(bool capable);
/* The engine, at bring-up (lever Z0-W): whether draft_lend_w can act (declared and set up on every rank). */
void draft_w_register(bool capable);
/* The engine, at bring-up (lever LF): whether place_lf can act (every rank places dynamically). */
void place_lf_register(bool capable);
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
