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
 *                   3 (ds41-int10, also RADIANCE_DRAFT_DEPTH0=3): depth 0 by OCCUPANCY ONLY -- draft_occ's rule decides first
 *                   as under 1 (0 at >= draft_occ rows for draft_occ_k cycles, the lend under it as under 1); where it does
 *                   not hold the cycle drafts, nothing measured or weighed (no plain-vs-window choice, no probe, no
 *                   back-off). Exact as 1 is; accepted wherever 1 is.
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
 *   ds41_psk_l20    deepseek41 (lever PSK-L20, ds41-pskl20): 1 = wherever lever PSK cuts (ds41_psk >= 1), the last kv
 *                   source (layer 20 of the 1017, the decoder's first layer: its global KV is projected from the
 *                   encoder's final hidden state) is split -- its collapse + norm, compressor and latent / index-key
 *                   stores over every row, its query, indexer, attention, MoE and window store over its cone
 *                   160 + (39 - 20) x 128 = 2592 rows as layers 21..39 over theirs (a planned pass with no row there
 *                   stages none of its experts); 0 = every row at layer 20 (today). 1 refused where lever PSK is not
 *                   declared or layer 20 carries an Engram block / DSpark tap. RADIANCE_DS41_PSK_L20=0 | 1: the start
 *                   value (unset: 0). Exact by the cone rule (deepseek41.cpp); bytes to be measured.
 *   dspark_pfctx    deepseek41 (lever PSK-DS, ds41-pskl20): 1 = a DSpark context pass (draft_pass -1) runs each
 *                   entry's last window + 1 (129) rows only -- the keys a query can still read -- in runs of >= 160 rows
 *                   (the form band of today's one launch), such a pass not recorded; 0 = every row (today). 1 refused
 *                   where no DSpark context pass is declared. RADIANCE_DS41_DSPARK_PFCTX=0 | 1: the start value (unset:
 *                   0). Exact for the target by construction (it reads no draft); the drafter's readable keys unchanged.
 *   ds41_psk_cr     deepseek41 + scheduler (lever PSK-CR, ds41-pskcr): 1 = a prefix hit of a request planned under
 *                   ds41_psk 2 / 3 / 4 that has no snapshot at the deepest grid point H under its match resumes there by
 *                   an exact CONE REPLAY: the request adopts the cached global KV (latent + index keys, every row) up to
 *                   H, starts at the grid point at or below H - reach (reach = 160 + 39 x 128 = 5152) with no state
 *                   restored, and its chunks up to H -- the cold run's own grid chunks -- run every layer only over H's
 *                   cone (layer l: the last 160 + (39 - l) x 128 rows before H) with no global-KV store (the batch maps
 *                   their full-group slots and linear-state slots to -1); the window keys, the Engram token window and
 *                   the DSpark taps at H come out as a cold prefill's. Then the prompt goes on from H as ever. Replay
 *                   chunks take no snapshot but one at H (ds41-pskcr2: the state the request goes on from, so the next
 *                   turn of the conversation resumes there instead of replaying again). No replay where the snapshot
 *                   the hit has (or the prompt's start) lies at most one grid step below H: resuming there recomputes
 *                   at most one chunk, cheaper than the replay (ds41-pskcr2). 0 = the anchor below (today).
 *                   RADIANCE_DS41_PSK_CR=0 | 1: the start value
 *                   (unset: 0). 1 refused where design 2 is not available. ds41_psk 4 = TAIL-ONLY anchors (the prompt's
 *                   end, its tail snapshot, checkpoint_at; no K-th grid point): a request admitted under 4 with
 *                   ds41_psk_cr 1 latches tail-only, under ds41_psk_cr 0 it plans as 2 (counted refused).
 *   ds41_engdd      deepseek41 (lever E1, ds41-e1; arch/deepseek41/ds41_engram_dedup.h): 1 an Engram layer's lookup, copy
 *                   and kv GEMM of a pass with a prefill sequence run over the step's UNIQUE n-gram id rows (the kv row is
 *                   a function of the row's ids alone, inference/model.py:353), and gather_rows copies every row's kv row
 *                   from its unique row; launches of <= 48 rows (the decode band) are issued as today, and nothing is
 *                   deduplicated when the unique rows would be <= 48 or fewer than 1/16 of the pooled rows repeat;
 *                   0 today's issue. RADIANCE_DS41_ENGDD=0 | 1 declares it (and starts at that value); unset: 1 refused,
 *                   the program and the issue are today's. A pass it may act on is not recorded. Exact: the same bytes
 *                   (gemm_nt_ggml's per-row form above the band). Live counters: /stats "engdd".
 *   sched_noa       scheduler (lever NOA, ds41-noa: a prefill's piece over the shared budget): 0 off (today); 1 / 2 on,
 *                   under sched_bi_chunk 3 / 4 (option D) with --prefill-span > 1 (refused without a shared step). On a
 *                   step that is neither a lone prefill's span step nor a step the interleave holds, a prefill that D
 *                   turns away because ANOTHER prefill already holds the step's shared budget -- a waiting request at
 *                   admission (queue order, the walk stops at the first that does not fit: none is overtaken), or a
 *                   running prefill (planned after every decode row of the running set) -- takes D's piece at the
 *                   step's --max-num-batched-tokens room instead (max_tok less every row planned, decode and draft rows
 *                   included): its alone piece whole, else the largest end on D's grid inside it with >= 128 rows on
 *                   both sides that fits (Scheduler::bi_chunk_plan, the same rule at another room; a request the bypass
 *                   runs: Scheduler::bypass_piece). 1 = at most ONE such piece a step; 2 = as many as max_tok and
 *                   --max-num-seqs hold (several newcomers' whole pieces in one step, with or without a long prompt
 *                   beside them). The step's first prefill is planned as before, so a long prompt's chunks are D's
 *                   chunks; every piece is one D (or the bypass) plans, so exact as D is. When even D's smallest piece
 *                   does not fit, the request waits that step (counted) and is marked; the next such step DISPLACES
 *                   every other running prefill (their next chunks move one step later, unchanged) so it runs as the
 *                   step's only prefill (D's own over-budget rule, up to max_tok); a displaced prefill is not displaced
 *                   again before it ran its next chunk. Never preempts. Read at every step planned; GET "noa" counts
 *                   admitted (waiting requests), pieces, rows, displaced (chunks moved), waited (did not fit), steps
 *                   (steps with a NOA piece) and every step with prefill rows by its prefill entries (pf_steps_1 / 2 / 3
 *                   / 4plus, NOA or not). 0 plans as unset.
 *   stage_short     engine (lever SA, ds41-sahb: the prefill stager armed for short steps): 0 off (today); N >= 1 a
 *                   trunk step with prefill rows and at least N tokens that runs at the span's shared arena level
 *                   (--prefill-span: every step of more than the largest smaller level's rows -- 1024 under
 *                   RADIANCE_ARENA_LEVEL_MAX=1024 -- up to the shared budget, 2112 on DeepSeek-V4.1) arms the prefill
 *                   stager with its ring at that level's own place (the one RADIANCE_STAGE_SPAN=0 lends to the slab):
 *                   the step's arena loan stops below the ring (the slab gives that room back for the step, recalled
 *                   like any level change), and the loan grows back over it at the next arena tick. Placement only:
 *                   a staged unit's GEMM reads the same bytes from the ring (stager.h). RADIANCE_STAGE_MIXED still
 *                   decides which steps with decode rows may arm. Refused where no rank has a stager or the plan has
 *                   no shared level. Read at every step release.
 *   stage_pick      engine (lever SA): which pooled units a stage_short step copies. 0 every pooled unit of each
 *                   routed layer (the stager's rule for full-plan steps); K = 1 .. 64 only those that one of the last K
 *                   trunk steps of the stager's size (>= its smallest pass) routed (the histograms the heat engine
 *                   credits); the rest are read in place as without the stager. Full-plan steps always stage every
 *                   pooled unit.
 *   place_hb        engine (lever HB, host-link balance): 0 off (today); R = 1 .. 4096 on every rank named by
 *                   RADIANCE_PLACE_HB_RANKS (a comma list; unset: every rank) the heat engine keeps the R units of each
 *                   slab class with the highest long-memory prefill heat (prefill rows' credit, decayed
 *                   RADIANCE_PLACE_HB_DECAY (0.8) a step with prefill rows) ranked above every unprotected resident:
 *                   promoted by ordinary swaps, never a victim while reserved; the decode-pinned core
 *                   (RADIANCE_HEAT_DECODE_PIN) stays above them. Needs the dual heat (RADIANCE_HEAT_PREFILL_DECAY) on
 *                   every rank. Placement only (the same bytes from whichever tier).
 *   stage_full_min  engine (lever SP, ds41-int9: the prefill stager at the FULL plan): 0 off (today: every full-plan step
 *                   -- on DeepSeek-V4.1 at --prefill-span 2 a lone step of 2113 .. 4160 rows: a new turn's tail, a lone
 *                   short prompt, a long prompt's spanned chunk -- arms the stager, which copies every pooled unit of
 *                   every routed layer through the ring, ~12.7 GB a rank); N = 1 .. 1048576 a full-plan step of fewer
 *                   than N tokens is left unarmed (sp_full_streams, stager.h): its pooled experts are read in place by
 *                   the grouped GEMMs (in-kernel streaming), as at every other level; N above the full plan's rows
 *                   (4161 on DeepSeek-V4.1) streams every full-plan step. Placement only: the same weight bytes, from
 *                   the host pool instead of the ring, the same kernel. Refused where no rank has a prefill stager.
 *                   Read at every step release.
 *   stage_full_pick engine (lever SP): which pooled units an ARMED full-plan step copies. 0 every pooled unit of each
 *                   routed layer (today); K = 1 .. 64 only those that one of the last K trunk steps of the stager's size
 *                   (>= its smallest pass) routed (lever SA's stamps from the histograms the heat engine credits); the
 *                   rest are read in place. Placement only. GET "sp" counts, per rank, full_steps / full_rows /
 *                   full_layers / full_units / full_bytes (every armed full-plan pass, any pick), full_pick_kept /
 *                   full_pick_dropped, stream_steps / stream_rows (full-plan passes stage_full_min left unarmed).
 *   mhc_fuse        deepseek41 (lever M1, ds41-int9b): 0 off (today); 1 a pcn_a / pcn_f launch (mhc_post_collapse_rmsnorm: the
 *                   post, the next connection's collapse and norm) of at least 512 rows is issued as
 *                   mhc_post_collapse_rmsnorm_mix -- the same op with that connection's mhc_mix over the streams it wrote, in
 *                   one op -- and the mix is not issued for those rows; N in 2 .. 1048576 the same from N rows.
 *                   RADIANCE_DS41_MHC_FUSE=0 | 1 | N declares it (the fused op beside every pair) and starts at that value;
 *                   unset declares nothing and >= 1 is refused. Exact: the two ops' bytes (libr4d pcn_mix_body; mhc_mix's
 *                   arithmetic is a row's own, so mixing some rows in the fused launch and the rest in the mix is too).
 *   ix_lds          libr4d (lever IX-LDSR2): 0 / 1 -- index_score's prefill WMMA form without keep, over MXFP4 keys, with a
 *                   workgroup's keys unpacked once into LDS and two rows a wave (16 a workgroup); the stock form's
 *                   per-output arithmetic, so the same bytes. RADIANCE_DS41_IX_LDS=1 at load starts it at 1.
 *   ix_ni           libr4d (lever IX-NI) + deepseek41 (its count): 0 / 1 -- index_score writes no -inf past a row's reach
 *                   (index_topk reads a row only below its reach) and index_candidates reads a row only below its reach (a
 *                   -inf never moves a block maximum, so the same keep words). RADIANCE_DS41_IX_NI=1 starts it at 1.
 *                   GET "i9b" carries the three levers' counters (i9b_stats).
 *   ds41_arcoded    libr4d (lever ARC, ds41-int10): 0 off (today: the four-rank exact two-shot at overlap 0 sends the plain
 *                   wire); N in 1 .. 1048576 -- such a launch of at least N rows (the operand's elements over its last axis)
 *                   sends the coded wire instead (~12.2 bits a bf16 value, r4d_ar_coded.h, until now only at overlap 1),
 *                   at 8 / 16 / 24 blocks of 1024 threads by the message (r4d_ar_arc.h: the ARW item's best grid a size),
 *                   where the coded form fits (bf16, whole-wave shards, the slot); the rest plain. Exact: the coded wire
 *                   decodes the same values, folded in the same order. RADIANCE_DS41_ARCODED=N at load starts it at N.
 *                   GET "arc" carries its counters (arc_stats).
 *   sched_row_cap   scheduler (lever CAP48, ds41-cap48, merged into ds41-int10 after ds41_arcoded: a cap on a step's decode rows): 0 off (today); N = 1 .. 4096 a
 *                   step's decode rows (every decoding sequence's own row + its drafts) are held to N by trimming drafts,
 *                   the deepest proposals first (equal depths: the later sequence in the plan's priority order first),
 *                   never a sequence's own row. At N = 48 every decode piece of the step stays in the decode band's
 *                   kernel forms (gemm_nt_ggml's <= 48-row GEMV, attention's decode band, index_score's scalar dot), whose
 *                   rows are batch-invariant. Prefill rows are not counted, and every prefill piece is planned as
 *                   without the cap (the trimmed rows' budget is not handed on). A step whose decoding sequences alone
 *                   exceed N (more than N at depth 0: --max-num-seqs > N) is left as planned (noted once). Both the step
 *                   planned after a commit and the one issued ahead are capped. Exact: only how many drafts a step
 *                   verifies moves, and the verify keeps the output (greedy, or a seeded sample whose rows the drafter
 *                   does not couple; under draft_sample >= 1 at T > 0 the coupled rows keep the target's law, as at
 *                   every other depth choice). RADIANCE_DECODE_ROW_CAP=N at start is the value until the switch is set
 *                   (unset / 0: off). Read at every step planned. GET "cap": the cap in force, steps_capped,
 *                   drafts_trimmed, ahead_capped (of them, steps issued ahead), steps_over (left: the sequences alone
 *                   exceed it), steps_gt48 (steps whose decode rows exceed 48, at any setting), max_rows (the most decode
 *                   rows a step carried), decode_steps (steps with decode rows).
 *   ds41_mix_wv     libr4d (lever 1U-MX, ds41-int11, appended after sched_row_cap): 0 off (today: mhc_mix's split kernel at
 *                   decode heights, <= 8 rows a launch, runs 8 waves a chunk, three of the 24 dots a wave); 12 / 16 / 24 the
 *                   same launch with that many waves a chunk (two / two / one dot a wave; the sum of squares on wave
 *                   24 % WV). Exact: a partial (row, chunk, dot) is the same sum in the same order whichever wave forms it
 *                   (mix_split_kernel's note: lane L the columns L, L + 32, .. from zero, unfused adds, r4d_wave_sum), and
 *                   the finish is unchanged. RADIANCE_DS41_MIX_WV=N at load starts it at N. GET "i11".mx counts the
 *                   launches it took (every rank).
 *   ds41_gq         deepseek41 (lever 1U-GQ = K1 fold b, ds41-int11): a mask -- 1 the query's q_a and the latent kv
 *                   projection (both GEMVs over the attention norm's output, one weight type a layer) as ONE launch, 2
 *                   attention_output_a's two group GEMVs (g0 + g1) as one launch, 3 both; 0 the separate launches.
 *                   gemm_nt_ggml_pair: lever D2's multi-row form with the two weights' rows side by side in one grid,
 *                   each output's operations exactly D2's (any form gives the same bytes), decode band only (above it,
 *                   or with D2 off, the two launches). RADIANCE_DS41_GQ=1|2|3 declares the pair ops (where a library
 *                   serves them) and starts at that value; unset declares nothing and >= 1 is refused.
 *   ds41_e2         deepseek41 (lever 1U-E2 = E2 / audit 6, ds41-int11): 1 the Engram kv projection at a decode-band
 *                   pass (<= 48 rows) computed by output columns over the ranks -- rank r the rows [r N/W, (r+1) N/W) of
 *                   the replicated weight (a view, nothing loaded twice) into a [rows, N/W] part -- then one
 *                   row-interleaved all_gather into engram_kv; 0 the whole GEMV on every rank. Exact: each output's dot is
 *                   gemm_nt_ggml's per-output sequence on a row slice, and the gather only copies.
 *                   RADIANCE_DS41_E2=1 declares the part, the GEMV and the gather (and starts at 1); unset declares
 *                   nothing and 1 is refused. GET "i11".gq / .e2 carry the two levers' issue counts (rank 0).
 *   sched_cb_hold_ms  scheduler (lever CBH, ds41-int10b bafca0c, merged into ds41-int11 after ds41_e2: co-batch hold): 0 off (today, byte
 *                   for byte); H = 1 .. 60000 the admission of waiting newcomers is held while at least two decoding
 *                   sequences run, so several newcomers start in ONE step (one link-bound host-expert read for 2-3 of
 *                   them). A hold covers fresh requests only (waiting, not preempted, no media, not an embedding, never
 *                   asked by an admission before); any other waiting request means no hold. It ends -- and the step
 *                   admits exactly as without the switch -- when sched_cb_hold_k newcomers wait, when fewer than two
 *                   sequences decode (an idle server is never held), or H ms after the oldest one's arrival. A request
 *                   is held at most once: every newcomer a release finds waiting is asked as today from then on.
 *                   While a hold stands, the step issued ahead (step_next) keeps running behind the step in flight, as
 *                   it does with nothing waiting. Admission order and every piece stay as sched_bi_chunk / sched_noa
 *                   plan them; only the step a newcomer is admitted in moves. Exact: scheduling only (under D3 + NOA
 *                   every piece is a whole alone piece wherever it runs). RADIANCE_SCHED_CB_HOLD_MS=H at start is the
 *                   value until the switch is set (unset / 0: off, nothing logged). Read at every step planned.
 *   sched_cb_hold_k   scheduler (lever CBH): 2 .. 64, the newcomers waiting that end a hold (default 2:
 *                   RADIANCE_SCHED_CB_HOLD_K at start, else 2). GET "cbh": in_force_ms, k, holds, by_k, by_timeout,
 *                   by_idle, by_other (a hold ended by a request that may not be held, the switch turned off, no
 *                   newcomer left, no admission possible), hold_ms (the holds' summed wall), held_steps (steps planned
 *                   while a hold stood, issued ahead included), cob_steps / cob_newcomers (steps that admitted >= 2
 *                   waiting requests, and those requests), single_steps (steps that admitted one) -- the last three at
 *                   any setting.
 *   gg_gemv_m16     libr4d (lever A4, ds41-a4; 64, after exl3_x3d_pf 63 = lever A10, ds41-pfx): 0 off (gemm_nt_ggml / its pair / folds a and e /
 *                   logits_gemm_ggml in the decode band take lever D2's form as with gg_gemv 2 alone); 1 under gg_gemv 2 a
 *                   call of M <= 16 rows takes the A4 table's form where it has an entry (the XP slab one ahead, the grid
 *                   fit's waves a block, H1R2, larger MT; libr4d r4d_ds41_ggml_mr.hip mr_choose). Exact: tiling, lane
 *                   map, loads and pipelining only, each output D2's operation sequence (the stock kernel's bytes, every
 *                   row count). RADIANCE_DS41_GG_M16=1 at load starts it; RADIANCE_DS41_GG_M16_FORMS=FILE adds a
 *                   measured table (r4d_selftest --perfm16) ahead of the compiled one.
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
              kSchedInterleave, kSchedMixedTok, kSchedShortBypass, kSchedBypassTotal, kDs41Psk, kDs41PskK,
              kDs41PskL20, kDsparkPfctx, kDs41PskCr, kDs41Engdd, kSchedNoa,
              kStageShort, kStagePick, kPlaceHb, kStageFullMin, kStageFullPick, kMhcFuse, kIxLds, kIxNi, kDs41Arcoded,
              kSchedRowCap, kDs41MixWv, kDs41Gq, kDs41E2, kSchedCbHoldMs, kSchedCbHoldK, kExl3X3d, kExl3X3dPf, kGgGemvM16, kSwitches };
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
/* Lever NOA (sched_noa): the switch once set, else 0 (off). The scheduler reads it at every step it plans and after every
 * step reports its counts (GET "noa", kNoaCounts values): waiting requests it admitted over the shared budget, its pieces
 * (admissions and running prefills) and their rows, chunks it displaced, prefills that did not fit even so, steps that
 * carried a NOA piece, and the steps with prefill rows by how many prefill entries they carried (1, 2, 3, 4 or more). */
int  sched_noa();
static constexpr int kNoaCounts = 10;   /* admitted, pieces, rows, displaced, waited, steps, pf_steps_1 .. 4plus */
void note_sched_noa(const int64_t* v);
/* Lever CAP48 (sched_row_cap): the switch once set, else RADIANCE_DECODE_ROW_CAP (0 .. 4096) at start, else 0 (off). The
 * scheduler reads it at every step it plans (step_next's step issued ahead takes the value of the step before it) and after
 * every step reports its counts (GET "cap", kCapCounts values): steps it capped, drafts it trimmed, how many of those steps
 * were issued ahead, steps it left because their decoding sequences alone exceed the cap, steps whose decode rows exceed 48,
 * the most decode rows a step carried, steps with decode rows. */
int  sched_row_cap();
static constexpr int kCapCounts = 7;   /* steps_capped, drafts_trimmed, ahead_capped, steps_over, steps_gt48, max_rows, decode_steps */
void note_sched_cap(const int64_t* v);
/* Lever CBH (sched_cb_hold_ms / sched_cb_hold_k): the switches once set, else RADIANCE_SCHED_CB_HOLD_MS (0 .. 60000; 0 off) /
 * RADIANCE_SCHED_CB_HOLD_K (2 .. 64; unset 2) at start. The scheduler reads them at every step it plans (the step issued ahead
 * takes the step before it's) and after every step reports its counts (GET "cbh", kCbhCounts values, in this order). */
int  sched_cb_hold_ms();
int  sched_cb_hold_k();
static constexpr int kCbhCounts = 10;   /* holds, by_k, by_timeout, by_idle, by_other, hold_ms, held_steps, cob_steps,
                                          cob_newcomers, single_steps */
void note_sched_cbh(const int64_t* v);
/* Lever PSK design 2: hand the scheduler's plan for trunk pass `step` to the plugin that cuts by it (rad_psk_plan, every
 * rank thread parked): per prefill sequence `ahead` and chunk rows, and the step's decode rows. Returns the first layer
 * the pass issues nothing from (the plugin's n_layer when every layer runs), -1 when no plugin takes a plan. */
int  psk_plan(int step, int64_t dec_tok, int n, const int32_t* ahead, const int32_t* rows, const int32_t* flags = nullptr);
/* Lever PSK-CR (ds41-pskcr): the rows before a replay's end a cone replay runs from (the plugin's deepest cone,
 * rad_psk_cr_reach: 160 + (n_layer - 1) x window, 5152 on the 1017), -1 when no plugin serves one. `flags` above: per
 * prefill sequence, bit 0 = the chunk is a cone replay (rad_psk_plan2; no global-KV store, every layer cut to its end's
 * cone). */
int  psk_cr_reach();
/* Lever PSK's live counters (ds41-int7-pskglue, /stats "psk"). The engine, at every step release: trunk passes it handed a
 * plan to, those that ended before the last layer, and the routed layers rank 0's stager did not stage for it. The
 * plugin's (rad_psk_stats, rank 0's): planned passes, passes with a cut, layers skipped whole, layer-rows not run at the
 * first nine cut layers and at the rest, plans dropped as not matching -- 0s and a return of 0 with no plugin serving it. */
void note_psk_engine(int64_t planned, int64_t ended_early, int64_t stager_skipped);
void psk_engine(int64_t out[3]);
/* out[0..5] as above; lever PSK-L20 (ds41-pskl20): [6] passes whose split layer was cut, [7] its rows not run in the cone
 * part, [8] passes it ran its all-row part only, [9] planned passes told to end at it (its experts not staged) -- 0s
 * from a plugin that writes 6. Returns how many the plugin wrote. */
/* Lever PSK-CR (ds41-pskcr): [10] replay passes the plugin cut, [11] their layer-rows run, [12] their layer-rows not run
 * below the first PSK layer, [13] kv-source compressor chains a replay pass skipped, [14] replay passes that were not
 * all-replay (decode rows or another prefill beside them: the chain ran, its stores masked) -- 0s from an older plugin. */
constexpr int kPskStats = 16;
int  psk_stats(int64_t out[kPskStats]);
/* Lever PSK-DS's live counters (ds41-pskl20, /stats "dspark_pfctx", the plugin's rad_dspark_pfctx_stats, rank 0's):
 * [0] dspark_pfctx (-1: no plugin serves it), [1] context passes cut, [2] their rows not run, [3] the rows they ran. */
int  dspark_pfctx_stats(int64_t out[4]);
/* Levers M1 / IX-LDSR2 / IX-NI's live counters (ds41-int9b, GET /admin/fn4 "i9b"), cumulative, counted at issue (a
 * recorded pass when it is recorded): out[0..5] deepseek41's rad_i9b_stats (rank 0): M1 declared, fused launches, their
 * rows, launches left unfused below the switch's rows, index_score runs issued under ix_ni, the plane bytes they leave
 * unwritten; out[6..9] libr4d's rad_ix_stats (every rank summed): LDS-form launches, their rows, index_score launches under
 * ix_ni, index_candidates launches under ix_ni. 0s where no plugin serves them. */
constexpr int kI9bStats = 10;
void i9b_stats(int64_t out[kI9bStats]);
/* Lever ARC's live counters (ds41-int10, GET /admin/fn4 "arc"; libr4d's rad_arc_stats, cumulative, launches of the
 * four-rank exact two-shot at overlap 0 counted when issued -- a recorded pass when it is recorded): [0] coded launches
 * (rank 0), [1] their rows, [2] plain launches (rank 0, any ds41_arcoded), [3] their rows, [4] plain launches below the
 * threshold, [5] plain launches at or above it (the coded form declined the message), [6] coded launches over every rank.
 * 0s where no plugin serves them. */
constexpr int kArcStats = 7;
void arc_stats(int64_t out[kArcStats]);
/* Levers 1U-MX / 1U-GQ / 1U-E2's live counters (ds41-int11, GET /admin/fn4 "i11"), cumulative, counted at issue (a recorded
 * pass when it is recorded): out[0..5] deepseek41's rad_i11_stats (rank 0): GQ layers declared, q_a + kv pair launches, out_a
 * g0 + g1 pair launches, E2 layers declared, E2 split launches, their rows; out[6..7] libr4d's rad_mx_stats (every rank):
 * mhc_mix launches in a wide-wave form, their rows. 0s where no plugin serves them. */
constexpr int kI11Stats = 8;
void i11_stats(int64_t out[kI11Stats]);
/* Lever E1's live counters (ds41-e1, /stats "engdd"; the plugin's rad_engdd_stats, rank 0's, cumulative): Engram layer
 * passes deduplicated, rows not computed in them, rows pooled in them, layer passes kept whole by the 48-row guard, layer
 * passes below the repeat share, steps with a deduplicated layer, layer passes with no launch above the band -- 0s and a
 * return of 0 with no plugin serving it. */
int  engdd_stats(int64_t out[7]);
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

/* Lever SA / HB (ds41-sahb). The engine, at bring-up: whether stage_short can act (a prefill stager and a span's shared
 * level on every rank) and whether place_hb can (the dual heat on every rank). */
void sahb_register(bool stage_capable, bool hb_capable);
/* The engine, at every step release (every rank thread parked): lever SA / HB's live counters (GET "sahb"), kSahbN
 * per rank for up to kSahbRanks ranks (see state_json). */
constexpr int kSahbN = 16, kSahbRanks = 8;
void note_sahb(int rank, const int64_t* v);
/* Lever SP (ds41-int9). The engine, at bring-up: whether stage_full_min / stage_full_pick can act (a prefill stager on
 * every rank). At every step release: lever SP's live counters (GET "sp"), kSpN per rank for up to kSahbRanks ranks. */
void sp_register(bool capable);
constexpr int kSpN = 9;
void note_sp(int rank, const int64_t* v);
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
