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
 *                   2 lever F4P (prefill-height passes). + 4 (tp8 loop D42 SDF): the shared arm's down issued ahead
 *                   of the routed down at decode-height passes whose arm is on lane 0; the same bytes.
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
 *                   0 today's issue. 2 (tp8 loop P48b) = 1, and where a launch takes the Engram twin (RADIANCE_DS41_TWIN
 *                   class 8) the dedup runs on it: launches of <= 64 rows kept, the unique rows on the int8 GEMM. RADIANCE_DS41_ENGDD=0 | 1 declares it (and starts at that value); unset: 1 refused,
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
 *                   Bit 1 (values 2 / 3, lever IXW): the decode band on the WMMA forms too. Bit 2 (values 4 .. 7, lever BFR,
 *                   tp8 loop P38): the WMMA forms' roundings as selects instead of a branch each -- the same bytes by
 *                   construction (ds_round_bf16_sel); RADIANCE_DS41_IX_LDS=N (1 .. 7) at load starts it at N.
 *   ix_ni          libr4d (lever IX-NI) + deepseek41 (its count): 0 / 1 -- index_score writes no -inf past a row's reach
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
 *   ar_xpush        libr4d (XPUSH, ds41-int13; ported from fn4-x3-sp3 808866c2, FN RSF 204; 63 here, after exl3_x3d): 0 off
 *                   (today: lockstep); 1 -- the four-rank exact two-shot's plain launch of a message of >= 1048576 elements
 *                   (ds41-int14c, FN 5f0352a8; was 131072; RADIANCE_FN4_AR_XPUSH_MIN at start overrides it) pushes its first
 *                   hop interleaved over the peers (R4D_AR8_FORM_XPUSH, r4d_ar_xpush.h: all three links at once); smaller
 *                   messages and the coded wires as before. Exact: the same stores, the same fold.
 *                   RADIANCE_FN4_AR_XPUSH=1 at load starts it at 1.
 *   place_swap_lent engine (lever SWL, ds41-swl; 65 here, after gg_gemv_m16; Mover::issue): 0 off (today: a heat
 *                   promotion takes only a free budgeted swap slot, else it starves -- experts.starved_slab -- and its unit
 *                   comes in later through the lent slots' fill); 1 a heat promotion may also take a free LENT slot (a KV /
 *                   arena / drafter loan's), the lowest free slot either way. A unit promoted there keeps its host copy, as
 *                   a fill's does, and is dropped by every demotion, reserve rebuild and recall the fill's units are.
 *                   RADIANCE_PLACE_SWAP_LENT=1 at start starts it on. 1 refused unless every rank places dynamically.
 *                   Exact: placement only. /stats experts.ranks[].lent.<lender>.swaps counts those promotions,
 *                   .place_swl the setting.
 *   heat_min_gain_x100  engine (lever L2, ds41-swl; 66 here): the heat guard's additive half, HeatParams::min_gain x 100,
 *                   0 .. 10000 (a swap needs hot >= cold x hysteresis + min_gain). Starts at RADIANCE_HEAT_MIN_GAIN x 100
 *                   (two decimals) or the shipped 150. Set on every rank's heat engine between steps (read by the next
 *                   step()). Refused unless every rank places dynamically. Exact: placement only.
 *   heat_hyst_x100  engine (lever L2, ds41-swl; 67 here): the guard's ratio, HeatParams::hysteresis x 100, 100 .. 1000; starts
 *                   at RADIANCE_HEAT_HYSTERESIS x 100 or the shipped 125. As heat_min_gain_x100 otherwise. /stats
 *                   experts.ranks[].heat_min_gain_x100 / heat_hyst_x100 read each rank's engine back.
 *   ds41_rfuse      deepseek41 (lever FUSE-R, ds41-fuse; 68 here, after heat_hyst_x100): the trunk MoE layers' router chain
 *                   (gemm_nt_f32 -> router_scored_topk -> moe_scatter: two dependent launch gaps a layer) in fewer launches,
 *                   each form the stock code with only the launch boundaries removed: 1 the selection and the sort as one
 *                   launch (router_scored_topk_scatter, no gate); 2 the projection too (router_gemm_f32_scored_topk_scatter:
 *                   the GEMM's last workgroup ranks and sorts) at a pass of <= 48 rows, else 1; 0 the separate ops (today).
 *                   RADIANCE_DS41_RFUSE=0 / 1 / 2 declares the forms beside every trunk router and starts at that value;
 *                   unset declares nothing and >= 1 is refused. Exact: every form writes the separate ops' bytes at any
 *                   row count. GET "fuse".rf_* counts the passes issued in each form (rank 0).
 *   ds41_mix_fin    libr4d (lever FUSE-M, ds41-fuse; 69 here): 1 mhc_mix at <= 48 rows in its one-launch FIN form (the
 *                   split launch's last workgroup of a row group runs mix_finish_kernel's row over the same partials:
 *                   the finish kernel's bytes, one launch fewer; not where ds41_mix_wv's wide form acts); 0 the split
 *                   and finish launches (today). RADIANCE_DS41_MIX_FIN=1 at load starts it on. GET "fuse".mixfin_*
 *                   counts its launches (every rank).
 *   draft_dcap      scheduler (SPB, ds41-etp-spb; 74 here after sched_tail_grid, 73 on ds41-etp-spb; sched/draft.h DraftController::dcap): 0 off
 *                   (today); D = 1 .. 64: once a cycle plans at least draft_dcap_n decode rows, the adaptive window's head
 *                   depth is held to at most D -- the choice is made over depths 1 .. D only, and a depth above D in force
 *                   moves down to that choice at once (no hold); under draft_dcap_n rows the window chooses as before (its
 *                   hold and hysteresis, so a move back up waits kHold cycles). On the ETP build at 8 sequences the window
 *                   chose depth 4 in some starts (40 rows, 2.66 ms a token) where depth 3 runs 32 rows at 2.48 ms a token
 *                   (etp/FOLLOWUP.md STEP 1). Starts 0. Above 0 refused without a drafter that runs the adaptive window.
 *                   Exact: it moves only which depth a step verifies. GET "dcap": cycles it bounded the choice, moves it
 *                   forced.
 *   draft_dcap_n    scheduler (SPB; 75 here, 74 on ds41-etp-spb): the decode rows from which draft_dcap holds, 1 .. 64; starts 8.
 *   pp_merge        engine, the two-node leader (tp8 MERGE; 95 here, after pp_lead_seg (82 on ds41-pp-merge); core/sched/lanes.h merge_into / split_from):
 *                   0 = today; bit 1: the decode lanes run as ONE lane (lane 0) while a BG lane has a step in flight or a newcomer in
 *                   its inbox (the other decode lanes park; their sequences and free cards move to lane 0 once idle), and split back
 *                   when none; bit 2: merged always; bit 4: with RADIANCE_PP_LEAD_SEG declared, a BG step's leader parts run in
 *                   segments only with the merge on (bit 1 / 2: every BG step then runs beside the merged lane); bit 8: no BG leader segments at all (so ONE start with RADIANCE_PP_LEAD_SEG declared
 *                   compares whole BG parts (8 / 9) with segments while merged (5)); bit 16: while merged a newcomer the BG lanes have
 *                   no room for is HELD out of the merged lane's step (D11's hold, RADIANCE_PP_MERGE_HOLD_MS, default 1500) instead of
 *                   a MIXED step of every sequence; bit 32 (the relay lane): while merged a parked decode lane with nothing of its own
 *                   takes whole newcomers as a third BG lane (after the idle BG lanes), so the next newcomer's head starts beside the
 *                   others' parts; its frames say so (StepHdr flags bit 8) and the follower segments them; bit 64 (decode tails first, with
 *                   RADIANCE_PP_READY_FIRST=1): a decode-only step whose X2 landed takes its tail before a long pass's even when that pass's X2
 *                   landed too (works with or without the merge bits). RADIANCE_PP_MERGE is its start value; taken only with two decode lanes or more
 *                   (else 0). The merged lane's occupancy rule (draft_occ) acts at once. Exact: only lane composition and the
 *                   drafting depth change (the build's composition invariance; depth moves only what a step verifies).
 *   pp_merge_ms     engine, the leader (tp8 MERGE; 96 here, 83 on ds41-pp-merge): the merge's hysteresis -- the lanes stay merged this many ms after the
 *                   last BG work ended (0 .. 60000; 0 = split at once); start value RADIANCE_PP_MERGE_MS. Exact (composition only).
 *   draft_cost_age  scheduler (DCA, ds41-drafter; 81 here, after stage_lone; sched/draft.h DraftController::cost_age): 0 off
 *                   (today); A = 16 .. 65536: a drafting depth's step-cost key last sampled more than A decode steps of its
 *                   scheduler ago does not price that depth while a fresh drafting key of the same row class exists (that
 *                   key by the prior's ratio stands in, as for an unmeasured depth). lq141 c16w: one lane held depth 2 on a
 *                   depth-3 key measured before drafter_mx 2. Starts 0. Exact: it moves only which depth a step verifies.
 *   draft_short_blk the engine (tp8 loop D23; 82 here, after draft_cost_age; Engine::block_shape): 0 (today) = the short block the
 *                   start chose (--draft-short-from's tier = the first block the drafter declared, or each
 *                   RADIANCE_DRAFT_SHORT_TIERS tier's); N >= 1 = the short block of N rows a sequence wherever --draft-short-from's
 *                   tier (or the TOP tier: the one for the highest load) would run its short block -- refused unless the
 *                   drafter declared N (deepseek41: RADIANCE_DS41_DSPARK_SHORT=2,3 declares 2 and 3) and a short block is
 *                   served at all. So one start runs depth 2 (2-row block) and depth 3 (3-row block) at lanes of 16 in
 *                   process. Starts 0. Exact: it moves only which declared block a drafter cycle runs (D18 / D19's rule);
 *                   the trunk verifies what it proposes. GET "short_blk": the declared lengths (mask), query passes at it.
 *   ds41_prefetch   deepseek41 (tp8 loop D24; 83 here, after draft_short_blk): which weights a decode-only pass of <= 64
 *                   rows reads into the on-die cache beside its all-reduces (libr4d `prefetch`, rad_arch.h prefetch_beside:
 *                   the second lane, or the chained form) -- bit 0 the mix fns (hc_f_fn beside the attention all-reduce,
 *                   the next layer's hc_a_fn beside the MoE all-reduce: the fused post + mix right after each reads it
 *                   first), bit 1 the router projection + the shared arm's gate_up beside the attention all-reduce, bit 2
 *                   the next layer's q_a / kv projections beside the MoE all-reduce; 0 = none (today). RADIANCE_DS41_PREFETCH
 *                   =0 .. 7 declares every read (each weight's first RADIANCE_DS41_PREFETCH_MB, default 8) and starts at
 *                   that value; unset declares nothing and >= 1 is refused (so is it where no library serves `prefetch`:
 *                   the host backend). Exact: the op reads weights and writes nothing a pass reads.
 *   ds41_attn_dec   libr4d (tp8 loop D26; 84 here, after ds41_prefetch): the decode band's attention form (attn_sparse_sink /
 *                   _rope_roundtrip and their gathered / unpacked forms, ds41_attn_dec_kernel) at the passes of more than 256
 *                   keys a row (the compressed layers: window 128 + top-k 512) where the rule takes form 4 (8 heads a
 *                   workgroup, 512 threads, tiled PV: more than 256 row-head pairs, i.e. c16 / c32's 24 / 48-row passes at 16
 *                   heads a rank) -- 0 = the rule (today), N = 1 .. 9 = form N - 1 there (0 .. 3 direct PV at 1024 threads
 *                   with 1 / 2 / 4 / 8 heads a workgroup, 4 .. 6 tiled at 512 threads with 8 / 4 / 2, 7 / 8 direct PV at 512
 *                   threads with 8 / 4; a form the LDS refuses takes the rule's next form). Every form sums each output in the same order: the same bytes (r4d_emu
 *                   test_ds41attn's D26 case). RADIANCE_DS41_ATTN_DEC=N starts at N. -1 when no loaded object serves it.
 *   ds41_sphc       deepseek41 (tp8 loop P25s; 85 here, after ds41_attn_dec): the SP route's row threshold in process -- 0
 *                   (start) = RADIANCE_DS41_SPHC's; N >= it: a lone prefill pass / a follower segment pass is sequence-
 *                   parallel from N rows (above --max-num-batched-tokens: none). Refused without RADIANCE_DS41_SPHC.
 *   ds41_sphc_attn  deepseek41 (tp8 loop P25s; 86): 0 .. RADIANCE_DS41_SPHC_ATTN (starts at it): 0 = an SP layer's q_a / kv
 *                   after xn's gather (P26 off), 1 P26, 2 + P26b, 3 + P26c (the kv / index sources of ratio > 1 without xn's
 *                   gather: the indexer's projection over the quarter, iw gathered; tp8 loop PREFILL round 38).
 *   ds41_sphc_seg   deepseek41 (tp8 loop P25s; 87): 0 .. RADIANCE_DS41_SPHC_SEG (starts at it): 0 = the follower's segment
 *                   passes whole (P25 off), 1 / 2 P25, 3 + P25b. Each only takes away what the env declared; exact at every
 *                   value (SP is the replicated pass's fold and bytes; X1 / X2 carry every row whichever side shards).
 *   pp_bg_room      engine, the two-node leader (tp8 loop P23s; 88 here, after ds41_sphc_seg): RADIANCE_PP_BG_ROOM in process --
 *                   0 = the shared budget (one ~1.1k newcomer a BG step), 256 .. 65536 rows a BG lane's inbox (lane_split_bg's
 *                   cap); starts at the env's; taken only with a BG lane (else 0). With ds41_bi 7 (PMM) a batched BG step runs its
 *                   newcomers' routed MoE as one pass a layer. Exact: multi-prefill steps are one process's (lever NOA / P23).
 *   pp_bg_join_ms   engine, the leader (tp8 loop P23s; 89): RADIANCE_PP_BG_JOIN_MS in process, 0 (off) or 1 .. 60000; with pp_bg_room.
 *   ds41_twin       deepseek41 (tp8 loop P30t; 90, after pp_bg_join_ms): mask + 256 x rows -- the twin classes (1 attn_q_a +
 *                   attn_kv, 2 attn_q_b, 4 attn_output_a + _b, 8 engram_kv, 16 the shared expert, 32 other) whose IQR twin ('<name>#pf' in the weight
 *                   sidecar) a PROMPT segment of >= rows rows (0 = RADIANCE_DS41_TWIN_ROWS) issues; decode rows keep the GGUF
 *                   launch. Starts at RADIANCE_DS41_TWIN; a subset of the env's classes (0 = every row GGUF). The follower takes
 *                   it with the leader's. Exact class: rule B (a prompt row's bytes are its own segment's).
 *   ds41_tsk        libr4d (lever TSK, ds41-drafter round 15; 91 here, after ds41_twin (85 before the merge of PREFILL's 85-90); 83 on ds41-drafter; r4d_ds41_sparse.hip):
 *                   0 off (today), 1 the compressing layers' decode-band attention (<= 48 rows) in the split-key WMMA form --
 *                   key chunks fixed by key index, a merge in ascending order with the o rope + fp8 round trip in it. Needs
 *                   RADIANCE_DS41_TSK=0 / 1 at load (the op's scratch sized; 1 starts it on). Rule B: deterministic and
 *                   batch-invariant against its own reference, not the decode band's bits. Where it serves a pass, ds41_attn_dec's
 *                   form choice does not apply (the decode band's forms are not issued).
 *   pp_fol_reserve  engine, the two-node FOLLOWER (tp8 loop IDEAS #9; 93 here, after draft_head_cut): 0 off, L = 1 .. 63 -- a BG
 *                   newcomer's body (a BG lane's split pass, RADIANCE_PP_BG_LANE) holds its parts from layer L until another BG
 *                   lane's frame arrives or RADIANCE_PP_FOL_RESERVE_MS (default 300) pass, every other frame running meanwhile.
 *                   RADIANCE_PP_FOL_RESERVE=L on the follower starts it at L; the follower takes the leader's request (its own
 *                   check: L a declared segment boundary inside W(h, t), a BG lane there; the leader checks the range only).
 *                   Order only: a pass's later segments go later, never another pass (rule A, as M4's segment gaps).
 *   pp_lead_seg     engine, the two-node LEADER (tp8 loop D32 = LSP; 94 here, after pp_fol_reserve): form + 16 x stride. Form 0 -- a BG
 *                   lane's step runs its leader head / tail whole (BEST's program beside the declared boundaries); 1 -- M9 as built
 *                   (RADIANCE_PP_LEAD_SEG_AHEAD as its env); 2 -- DECODE FIRST: a decode tail whose X2 landed goes before the BG
 *                   step's next segment (its lane's next head at the round's top, before that segment too), one BG segment in
 *                   the stream at a time, and a BG step runs its parts whole while no decode lane has a step in flight (a lone
 *                   prompt: the cold cells). Stride s (1 .. 15; 0 = 1): only every s-th declared boundary splits the part, so one
 *                   start with RADIANCE_PP_LEAD_SEG=1 compares k = 1 / 2 / 3 / 5 (v = 2, 34, 50, 82). Needs RADIANCE_PP_LEAD_SEG=k
 *                   on the leader at start (else only 0 is taken); its start value RADIANCE_PP_LEAD_SEG_FORM (default 1 =
 *                   today's M9 with the env set). Order of issue only, never a step's composition (rule A, as M9).
 *                   + 256 (tp8 loop D36): no band passes -- under RADIANCE_PP_BAND=1 a fresh BG step runs W(h, t)'s (the band layers
 *                   stay declared), so one start compares P28's band (with RADIANCE_PP_LEAD_SEG_BAND=1: its head in these segments)
 *                   with none: v = 2 vs 258, 0 vs 256.
 *                   + 1024 x N (tp8 loop D49b, N 1 .. 63): forms 1 / 2 segment a BG step only while the decode lanes' last steps
 *                   decoded <= N sequences (Scheduler::decoded_seqs summed over the decode lanes); above N the step's parts run
 *                   whole as under form 0 -- so ONE value segments at c8 (the leader's BG head stops both decode lanes there,
 *                   lq218) and keeps c16 / c32's whole parts (lq207: segments -1.7..-6.4 % there). 0 = no load rule.
 *                   Position-order guard (tp8 loop F81, on unless + 65536): a step whose prompt entries are not all WHOLE FRESH
 *                   prompts (context 0, the prompt's last row, no snapshot / replay) runs its parts whole -- a piece of a longer
 *                   prompt in segments let a later piece's head / tail run before its remaining segments (lq225 'all': layer 8's
 *                   compressed-KV copy differed). + 65536 = no guard (the F75 / F81 form, a negative control only).
 *                   + 131072 (tp8 loop D64): a TAIL BAND BG step (fn4 pp_tband 1 / 3 / 4 / 5: every fresh BG step at --pp-cut
 *                   10,33 with D50 / D51) runs its HEAD in these segments, its tail whole -- the held frame keeps the tail band
 *                   flag, the follower's parts end at t - j and the leader's tail starts there, as lane_head's band step; without
 *                   it such a step runs its parts whole (D39), so LSP never segmented a BG step beside D51. Order only (rule A).
 *                   + 262144 (tp8 loop D64b, with + 131072): that step's TAIL in segments too -- the first from t - j (X2's receive at
 *                   the band layer), then at the declared tail boundaries (all above t), the last with the sampler; without it the
 *                   tail runs whole.
 *   pp_seg_set      engine, the two-node FOLLOWER (tp8 loop D33; 95 here, after pp_lead_seg): which of the follower's declared
 *                   segment boundaries (RADIANCE_PP_SEG_LAYERS, at most RADIANCE_PP_SEG_MAX - 1 of them) split a pass: n + 64 x s,
 *                   n 0 .. 31, s 0 .. 7 -- every s-th boundary inside W(h, t) (0 = 1: all), then the first n of those (0: all
 *                   of them). 0 = every declared boundary (the start's program). With RADIANCE_PP_SEG_MAX=32 and SEG_LAYERS=1 at
 *                   W(10,32): 15 = 11..25 (the program before D33: [25,32) whole), 0 = 11..31, 128 = 12, 14 .. 30 (= SEG_LAYERS=2's
 *                   set). A body reserve's boundary (pp_fol_reserve) splits the pass whatever the set. RADIANCE_PP_SEG_SET=v on the
 *                   follower starts it at v; the follower takes the leader's request (its own check: boundaries declared; the
 *                   leader checks the range only). Order only: where a pass's segments end, never a pass's ops (rule A, M4).
 *   exl3_x3d_pf     exl3r4d (A10, ds41-x3d-pf 12.11.; tp8 loop PX1 on the served plugin line ds41-u14-band 4fd50e4; 96 here, after
 *                   pp_seg_set): 0 (today) or 1 -- K = 2's prefill-band (M > 16) map-form GEMM decodes a lane's eight trellis states
 *                   from ONE word pair (decode_frags_x3d, the decode band's X3D-F form): the same f16 weights, each output's WMMA
 *                   order unchanged -- the same bytes (Xeon GPU2: y at M 17 / 64 / 1,150 / 4,096 identical to 0). -1 when no loaded
 *                   object serves it (an exl3r4d before 4fd50e4).
 *   pp_rband        engine, the two-node LEADER (tp8 loop P41, the HEAD band on the follower; 100 here, after pp_merge_ms): v = p + 16 x
 *                   phase (+ 256): a lone prompt chunk i of a sequence with i % p == phase runs as a REVERSE pass -- the leader's head over
 *                   [0, h - 1) + layer h - 1's key chain, the follower from h - 1 (X1 of W(h - 1, t)); + 256: never the chunk before the
 *                   last; 0 = none; -1 = the start's RADIANCE_PP_RBAND_EVERY. Needs the plugin's reverse band (RADIANCE_DS41_PP_RBAND=1 on
 *                   both nodes, whose every prompt pass crosses layer h - 1's window keys); else no pass changes. Read at each lane step's
 *                   head; the same TP4 functions on either node (rule A), the follower's window copy whole for every reversed chunk.
 *   sched_srf       scheduler (tp8 loop P44, upstream 67cf57b8 'concurrent prefills take the span one at a time'; 101 here, after
 *                   pp_rband): 0 -- today's plan; 1 -- in each scheduler (one process, or each lane of a pipeline) running prompts with no
 *                   decode row and no admission possible take the step one at a time, the fewest prompt rows left to plan first: that
 *                   one is planned alone (its lone prefill's span / chunk plan), the others sit the step out (no share of it); the one
 *                   step after a request finished beside others is planned as today (a follow-up turn waits a shared step, not a
 *                   span); 2 -- 1, and across the decode lanes of a pipeline while no lane has a decode row: of every decode lane's
 *                   running prompts (rows left to plan) and waiting prompts (whole prompt) only the fewest plans prompt rows -- the
 *                   others sit out or stay queued, a waiting one is admitted into a step of its own -- and its chunks go over the
 *                   lanes as a lone cold prompt's (a wide span piece too: the borrowers' prompts sit out). Each prompt's chunks are
 *                   its lone plan (exact as a lone prompt's). Unset: the start's RADIANCE_SCHED_SRF (0 / 1 / 2).
 *   aql_hdp         core, the AQL backend (tp8 loop U1 = upstream 8f2e511f's first half; 102 here, after sched_srf): 0 -- a store
 *                   fence in front of a doorbell, as before; 1 -- the fence, then a store to the card's host data path flush register
 *                   (HSA_AMD_AGENT_INFO_HDP_FLUSH) in front of every doorbell (a live packet's publish and a played run's ring), so the
 *                   kernel arguments and chain records written into VRAM through the BAR are there before the packets that read them
 *                   are announced. Order of host writes only: no packet, argument or kernel changes (rule A). A card whose arguments
 *                   live in host memory, or whose driver gives no register, flushes nothing. Both nodes of a pipeline (a follower
 *                   switch). Unset: the start's RADIANCE_AQL_HDP_FLUSH (0 / 1).
 *   ds41_csplit     deepseek41 (tp8 loop P47; 103 here, after aql_hdp): 0 -- the candidate layer's indexer unsplit where its keep
 *                   words cross (the follower of W(10, 32): layer 20, every rank every row's full-width index_score), as before; 1 --
 *                   split there as at every other index layer, the keep rows all-gathered after it. Needs RADIANCE_DS41_CAND_SPLIT=1
 *                   at load on BOTH nodes (keep's pitch padded to 16-byte words, X2's keep part with it; the follower declares the
 *                   part + gather), which starts it at 1; without the env 1 is refused. Exact either way (lever I1's split + a word
 *                   gather). The follower takes it with the leader's; the leader holds nothing it changes.
 *   pp_send_ash     engine, BOTH nodes (tp8 loop P46; 105 here, after sched_lane_room): 1 -- a row-sharded payload (an SP pass's X1
 *                   on the leader, X2 on the follower under RADIANCE_DS41_SPHC_X2) of >= N rows: every rank packs its share into its
 *                   own card staging buffer on the pass's stream and a side stream of the card copies it into the send ring (the
 *                   split send's handshake unchanged), so the pass's stream goes on with the next pass while the share crosses PCIe;
 *                   0 -- every share on the pass's stream, as before. The ring's bytes are the same either way (a copy's order only).
 *                   Needs RADIANCE_PP_SEND_ASH=N at start (the staging buffers, kept by the budget on every rank; passes of >= N
 *                   rows issued live), which starts it at 1; without the env 1 is refused. The follower takes the leader's request.
 *   aql_waitk       core, the AQL backend, BOTH nodes (tp8 loop W1s, PREFILL round 42; 106 here, after pp_send_ash): lever WAITK
 *                   (DRAFTER round 49, reviewed in DECODE W1: core/device/aql.h) in process. 0 -- an event / ride wait is its
 *                   barrier-AND alone, as before; 1 -- an event wait (rad_event_wait) whose signal is still pending at enqueue gets a
 *                   one-wave wait kernel (an aside packet) on the waiting stream AHEAD of the unchanged barrier-AND, so the packet
 *                   processor never holds a pending barrier (one costs every other queue of its CP pipe ~+40 us a packet); 2 -- 1, and
 *                   ride waits (rad_ride_wait) too. Scheduling only: no kernel's bytes change (rule A). Read at every wait; nothing
 *                   is allocated, so 1 / 2 are taken without the env. Unset: the start's RADIANCE_AQL_WAITK (0 / 1 / 2, the parse of
 *                   rad::aql::waitk_mode_of). A follower switch (each node's own waits). A tape capture keeps its waits plain either way.
 *   aql_ika         core, the AQL backend, BOTH nodes (tp8 loop W3s, DECODE round 45; 107 here, after aql_waitk): lever IKA (DRAFTER
 *                   round 50, reviewed in DECODE W3: core/device/aql.h) in process. 0 -- every played packet keeps the fences it had
 *                   without the lever (the acquire rule's own marks, the packet release); 1 -- a self-acquiring kernel's played packet
 *                   (its module lists it in r4d_ika_kernels: libr4d / exl3r4d built with R4D_IKA) drops the packet release unless a
 *                   record or ride completes on it, and the acquire where haz_mark's IKA walk allows; the other packets take that
 *                   walk's rule marks. Both mark sets are made when a pass is recorded, so the switch picks at play: no re-recording,
 *                   and 0 is exactly the packets of a start without RADIANCE_AQL_IKA. Fences only: no kernel's bytes change (rule A).
 *                   2 (DRAFTER round 53, lever IKB) -- 1, and core's own device-local copies / fills (core/device/aql_blit.hip) are
 *                   LAUNCHED in their self-acquiring forms (the same loads / stores), so a routing copy also goes through haz_mark's
 *                   IKA walk on its exact ranges; the kernel is picked at launch, so a pass re-records (the epoch) as at every change.
 *                   Needs RADIANCE_AQL_IKA=1 at start (which kernels are self-acquiring is read at their first launch), which starts
 *                   it at 1; without the env 1 / 2 are refused and it reads 0. A follower switch (each node's own played packets).
                   3 (DRAFTER round 64, lever CBL) -- 2, and where lever CIK is on (RADIANCE_AQL_CIK=1 at start) those copies / fills
                   launch in their CHAIN-CAPABLE forms (aql_blit.hip *_cik: CIK's entry and exit), which the backend chains as it chains
                   CIK kernels; the same loads / stores (rule A); without CIK 3 launches as 2. Re-records as 2 does.
 *   sched_lane_room scheduler (tp8 loop D56; 104 here, after ds41_csplit): 0 -- a lane's admission room gates (a fresh prompt's whole
 *                   rest, a preempted request's whole recompute, beside every runner's rest of prompt + a block each) count the asking
 *                   lane's running requests, as before; 1 -- they count every lane's (each lane's running set and the ones its plan left
 *                   out), as one process's gate counts its one running set: a long prompt in one lane waits instead of being admitted
 *                   beside another lane's unfinished one (both then run out: a preemption by recompute). One lane / one process: the
 *                   same either way. Admission order only (rule A). Read at every gate the scheduler asks; the leader's (not a follower
 *                   switch, not handed over). Unset: the start's RADIANCE_SCHED_LANE_ROOM (0 / 1).
 *   ds41_rr1        deepseek41 (DRAFTER round 54, lever RR1; 108 here, after aql_ika): 1 -- after a step's ONE MoE pass from row 0,
 *                   the step's end routing report names the core's counts row (rad_route_counts), which that pass's own report
 *                   (right after its router) already filled with the same counts, so the core takes it in place: one copy
 *                   dispatch fewer a routed layer and step on the compute stream (trunk and DSpark layers); the report's count,
 *                   step and tape route step are as before. 0 -- both reports copy, as before. Host bookkeeping and one
 *                   op-less copy only: no kernel's bytes change (rule A). A plugin switch (the follower takes the leader's);
 *                   read at issue (a flip re-records). Unset: the start's RADIANCE_DS41_RR1 (0 / 1).
 *   dspark_mhr      deepseek41 (levers DMR / DML / DOF; 109 here, after ds41_rr1 -- tp8 loop R45): the DSpark drafter's mHC
 *                   forms 0..5 (1 lever DMR, 2 + fold c, 3 + pcn, 4 + DML (the late attention mix), 5 + DOF (o_rope + o_q in
 *                   one launch where RADIANCE_DS41_DSPARK_MHR=5 declared it)); the plugin serves the name, but before 109
 *                   the core did not list it, so every POST naming it was refused as an unknown switch (a kit arm with it
 *                   was skipped). Proposals only. A plugin switch (the follower takes the leader's); read at issue (a flip
 *                   re-records). Unset: the start's RADIANCE_DS41_DSPARK_MHR (0..5).
 *   dspark_hs       deepseek41 (lever DHS, DRAFTER round 57; 110 here -- tp8 loop R45): 1 -- at a DHD-cut DSpark query pass the
 *                   last stage's post-attention ops run on the S x H head rows only (gather_rows after its attention, the
 *                   stage's stream buffers swapped, the head reads the compacted rows); 0 -- the whole block's rows as before.
 *                   1 is taken only where RADIANCE_DS41_DSPARK_HS=0 / 1 declared its ops (refused where a rank's declare
 *                   failed; a node without the env takes the value and issues as before). Proposals only. A plugin switch;
 *                   read at issue. Unset: the start's RADIANCE_DS41_DSPARK_HS (0 / 1).
 *   host_spin       core, BOTH nodes (tp8 loop D61, DECODE round 48; 111 here, after dspark_hs): how a host thread waits on the
 *                   pipeline's critical path. v = N + 256 x M: N 0..255 = spin up to N x 100 us (a pause loop on the condition)
 *                   before blocking as before; 0 -- every wait blocks at once, as before (today's program); M 0..15 = the sites
 *                   (0 = every site): 1 the rank barrier (a release and a join: core/runtime/rank_barrier.cpp), 2 the leader's
 *                   wait for a lane's drafter passes (Engine::lane_tail_finish's step_end_), 4 a frame's way in (net::NodeLink:
 *                   the receive thread's socket read, polled without blocking, and recv's queue wait), 8 the lane loops' poll naps (sleep_for 20 / 50 / 100 us: a spin of that length, so a
 *                   poll lands every ~20 us instead of every ~80 with the default 50-us timer slack). Timing only: no issue
 *                   order, step, byte or plan rule reads it (rule A). Read at every wait. Unset: the start's RADIANCE_HOST_SPIN
 *                   (the same values; anything else = 0). A follower switch (each node's own threads); one machine takes it too
 *                   (its rank barrier).
 *   pp_bg_split     engine, the two-node LEADER's lanes (tp8 loop D62, DECODE round 49; 112 here, after host_spin): how a BG newcomer
 *                   is lent (Scheduler::lane_split_bg) with two BG lanes or more (RADIANCE_PP_BG_LANE >= 2): 0 -- whole, to one BG
 *                   lane (today's program); 1 -- tp8 dev M8's RADIANCE_PP_BG_SPLIT=1: cut at the 512-grid point nearest its middle
 *                   (>= 128 rows a side; answer-neutral: only which rows share a pass), the first piece to the first BG lane with
 *                   room, the rest to a later one; 2 -- as 1, but only where both lanes are IDLE (an empty inbox and no BG step in
 *                   flight), else whole (the low-load form: the follower starts on the first piece's body while the leader runs
 *                   the second piece's head). Read at each lend. Unset: the start's RADIANCE_PP_BG_SPLIT (0 / 1 / 2). The
 *                   leader's own (the follower skips it by name).
 *   aql_cik         core, the AQL backend, BOTH nodes (tp8 loop W9, DECODE round 50; 113 here, after pp_bg_split): lever CIK (DRAFTER
 *                   round 62) in process -- 1 = a chain-capable kernel (KernelRec::cik: RADIANCE_AQL_CIK=1 / 2 at start with lever IKA
 *                   and a module exporting r4d_ika_chain() = 1) chains: it waits in-kernel on its queue's counter and adds a lane at
 *                   its exit (with CHI / CTA where their env is on); 0 = it issues and plays as the IKA kernel it also is (its record
 *                   off: no wait, no add; the packets lever IKA gives it). Read at every live issue and every played packet (a tape
 *                   records the capability; a change bumps the epoch anyway). The plugins' named chained forms (RADIANCE_AQL_CIK=2)
 *                   are chosen at start and do not follow it. Unset: 1 where RADIANCE_AQL_CIK=1 / 2 at start, else 0; 1 needs that env
 *                   (which kernels are chain-capable is read at their first launch). Rule A: the launch-order mechanism only.
 *                   Lever CKW (DRAFTER round 108): N >= 2 = 1 for dispatches of at most N work-items only; a wider chain-capable
 *                   dispatch issues and plays as the IKA kernel it also is (RADIANCE_AQL_CIK_LANES=N's bound, in process).
 *   host_slack      core, BOTH nodes (tp8 loop D63, DECODE round 51; 114 here, after aql_cik): the TIMER SLACK of the host threads
 *                   that poll the pipeline's critical path with short sleeps -- the lane loops' naps (fn4::nap_us where host_spin
 *                   leaves them sleeping), the link's send proxy (its scan of the send rings after 64 idle scans), the verbs
 *                   receive-completion thread (after 1,024 idle polls) and the ring waits (ring_put / ring_take after 256 spins).
 *                   v = 1 .. 1000: each of those threads sets its own slack to v us (prctl PR_SET_TIMERSLACK) before its next
 *                   covered sleep, so a sleep_for(20 us) ends ~v us late instead of ~50 (the kernel's default slack for a normal
 *                   thread: 20 -> 82.8 us p50, tp8 loop D61's M); 0 = every thread back to its default slack (today's program: no
 *                   call is made while the switch was never above 0). No spin, no CPU held; the sleeps' lengths, counts and every
 *                   other rule unchanged (rule A: timing only). Once it was above 0, the covered sleeps are counted per site (asked
 *                   / slept us) and print beside host_spin's line. Read at every covered sleep. Unset: the start's RADIANCE_HOST_SLACK
 *                   (0 .. 1000; anything else = 0 with a line). A follower switch (each node's own threads); one machine takes it.
 *   pp_tband_bound  engine, BOTH nodes (tp8 loop D65, DECODE round 53; 115 here, after host_slack): fn4 pp_tband 5's WINDOW and
 *                   HYSTERESIS in process -- v = W + 10000 x H, W = 20 .. 5000 ms, H = 0 .. 500 permille: the leader judges the
 *                   follower's recent card working share against its own over the last W ms and bands once the follower's is
 *                   higher by > H (keeps the layer on the follower once the leader's is), a load report older than 4 W falls back
 *                   to 3's rule; the follower's load report covers its last W ms. -1 / 0 = the start's RADIANCE_PP_TBAND_BOUND_MS /
 *                   _H (defaults 200 / 20) on both, i.e. today's program (e.g. 200200 = the defaults as a value, 100 = 100 ms with
 *                   H 0, 200050 = 50 ms with H 20). Read at every decode step pp_tband 5 judges (the leader) and at every load
 *                   report (the follower: a follower switch, so both nodes use one window); inert at every other pp_tband. Where
 *                   the band layer runs only (rule A, D50's forms). One machine takes it (no pipeline: nothing reads it).
 *   sample_narrow   core, the sampler, BOTH nodes (DRAFTER track round 79, lever SNW; 116 here, after pp_tband_bound): 1 -- a
 *                   step whose every sampled row is a top-1 chain row (greedy over the split vocabulary, every row of a greedy
 *                   build), none asking for logprobs, no late temperature, no coupled verify, runs the vocabulary-parallel chain
 *                   at 2 candidates a rank (the served sample_topk, the gather of the first 2 pairs of its rows: 16 bytes a row
 *                   a peer, the merge of world x 2 pairs, the pick over 2) instead of the served 1024 (a gather of 8 KiB a row a
 *                   peer, a 4096-word merge a row); 0 = the served width. The same token by construction (each rank's argmax
 *                   word is in both sets; the merge's maximum is the global argmax, the lower id on a tie): rule A. Read at every
 *                   sampler run (live issue). Unset: 1 where RADIANCE_SAMPLE_NARROW=1 at start (which declares the narrow chain),
 *                   else 0; 1 needs that env. A follower switch (each node's own sampler); one machine takes it.
 *   exl3_x6w        exl3r4d (X6w, tp8 loop DECODE round 62 = QUEUE X6w-S; plugin ds41-x6w-rt; 117 here, after sample_narrow): 0
 *                   (today) or 1 -- the decode band's X3D-F (K = 2) and K25U (K = 2.5: gate_up G3, down U4) kernels with the K loop's
 *                   WMMAs replaced by per-row v_dot2_f32_f16 dots over 4-row windows of a run (each row's two half sums joined in
 *                   one order, moved into the WMMA's D layout; split-K, epilogue unchanged): a row's bytes depend on its own
 *                   activations and the weights alone, NOT the WMMA form's bytes -- rule B (user 18.11.; COORDINATOR 02:43
 *                   22.11.: its own reference, deterministic and the same alone and beside load). -1 when no loaded object serves
 *                   it (an exl3r4d before ds41-x6w-rt). A plugin switch: a pipeline's follower takes it from its leader.
 *                   X6w was dropped (QUEUE X6w-S); the same switch carries lever SWK (tp8 loop DECODE round 64 = QUEUE X12; plugin
 *                   ds41-swk 3602fc5a, which refuses 1): 2 -- the same kernels with the K loop's WMMA pairs as gfx1201's 2:4 sparse
 *                   v_swmmac_f32_16x16x32_f16, each token's K over two sparse-A rows, 8 tokens a pass, the rows' sums added once
 *                   and moved into the WMMA's D layout (rule B, its own reference); 3 -- 2 with the other index nibble order (the
 *                   card decides which of 2 / 3 is the right one; the wrong one is far off, not a rounding). The card named 2
 *                   (Xeon GPU2 slot s4, COORD 11:54 22.11.). 4 (tp8 loop DECODE round 67 = QUEUE X15; plugin ds41-swk
 *                   d4b42d9) -- 2 + the codebook fold's twins: with exl3_x3d 5 (X5F) the fold's kernels take the same sparse K loop
 *                   over the integer codes and apply the fold in SWK's row layout before the split-K sum (rule B, its own
 *                   reference); elsewhere 4 = 2; at 2 / 3 the fold keeps its own WMMA kernel. 5 (QUEUE X15b / X15c; plugin
 *                   ds41-swk 104f670 / b2cca12) -- 4 at K = 2 only: the K = 2.5 layers under exl3_x3d 5 run SWK's own twins (the
 *                   fold's K2.5 twins lost to them, Xeon GPU2 slots s6 / s7); the arm the cards run (ORDER 14:46 4 (a)).
 *   pp_tband       engine, the two-node LEADER (tp8 loop D39, the TAIL band; 97 here, after exl3_x3d_pf): 1 -- a BG lane's step of
 *                   fresh prompts runs as a tail band pass (the follower over [h, t - j) + layer t - j's key chain, the leader's tail
 *                   from t - j); 0 -- such a step runs W(h, t)'s (the band layer stays declared on the leader, so the KV pool is the
 *                   same in both arms); 2 (tp8 loop D39b) -- 1 + a step of prompt rows only on ANY lane (a lone cold prompt plans
 *                   on its home decode lane) + a prompt's LATER chunks while every chunk of it so far ran as a tail band pass (the
 *                   leader holds the band layer's keys of all its rows): the cold prompts' chunks, with RADIANCE_PP_BAND_MAX_ROWS
 *                   at or above the chunk (4160). Needs RADIANCE_PP_TAIL_BAND=1 or 2 on the leader at start (its start value; else only 0 is taken)
 *                   and the plugin's tail band (RADIANCE_DS41_PP_TBAND=1 on both nodes). Read at each BG step's head; never a pass's
 *                   ops (both forms issue the published ops: rule A, as P28's band).
 *                   3 (tp8 loop D46, needs the plugin's tail copy) -- 1 + a decode lane's step while a BG step is in flight when every
 *                   sequence of it is held (core/sched/tcopy.h). 4 (tp8 loop D50) -- 3 in EVERY stretch; 4 + 1024 x N (N 1 .. 63) -- 3,
 *                   and in a clean stretch only while the decode lanes' last steps decoded > N sequences. At --pp-cut h,t+1 a decode
 *                   band pass is W(h, t)'s decode: 4 is the in-process control of the published cut, 3 puts layer t on the follower in
 *                   the leader-bound clean stretches (lq222 D45: L 93-97 / F 82-90 % working at c4 .. c32) and on the leader in the
 *                   follower-bound BG ones, 4 + 1024 x N keeps it on the leader in clean stretches above N (c16 / c32).
 *                   5 (tp8 loop D51) -- a held decode step bands while the FOLLOWER's card worked more than the leader's over the last
 *                   RADIANCE_PP_TBAND_BOUND_MS (default 200) by > RADIANCE_PP_TBAND_BOUND_H permille (default 20; the follower's share
 *                   from its heartbeat load report, RADIANCE_PP_LOAD_REPORT_MS=N on the follower, RADIANCE_PP_NODE_IDLE on both), not
 *                   while the leader's did, else as the last step; no fresh report -> 3's rule. Where the band layer runs only (rule A).
 *   draft_head_cut  engine (DHD, ds41-drafter round 11; 92 here, after ds41_tsk (82 on ds41-drafter, after draft_cost_age); Engine::issue_block_drafter): 0 off (today),
 *                   1: with RADIANCE_DRAFT_HEAD_CUT=1 at start, a query pass at a block drafter's own block (every row
 *                   proposing, block = depth: DSpark) runs its head and walk on the first D rows of each block, D = the head
 *                   depth in force under the adaptive window; the verify of the step issued ahead reads those D. Starts 0.
 *                   Exact: the first D proposals are the same tokens (a row's head row and the walk's first D steps).
 *   exl3_dng        exl3r4d (SPB-G, ds41-etp-spb-plug; 76 here, 75 on ds41-etp-spb): 0 (today) or 2 / 4 / 8 -- the K = 2 (X3D-F) and K = 2.5 (K25U)
 *                   down kernels take that many 128-column blocks a workgroup (ceil(40 / G) x slots workgroups instead of
 *                   40 x slots), each block the one-block workgroup's code on the same data: every output byte is
 *                   today's. -1 when no loaded object serves it.
 *   ds41_moe1       deepseek41 (lever P, ds41-p1pass; 72 here after ds41_wmk, 71 on ds41-p1pass): 1 a step that is ONE decode segment of
 *                   17 .. 48 rows runs its MoE as ONE pass -- the router, the shared arm, the gather and the all-reduce
 *                   once, the routed EXL3 experts through moe_exl3_gate_up_dk / moe_exl3_down_dk (the decode kernels over
 *                   the whole pass: an expert decoded once), which write per (token, slot) the bytes of the <= 16-row
 *                   pieces ds41_bi 2 / 4 / 6 issue; 0 the pieces (today). A step with a prefill segment keeps the pieces.
 *                   RADIANCE_DS41_MOE1=0 / 1 declares the two ops beside every trunk layer's EXL3 pair and starts at
 *                   that value; unset declares nothing and 1 is refused. Exact (a decode row keeps its alone bytes).
 *   route_copy_ov   core (lever RCO, ds41-launch; 70 here (68 on ds41-launch), after ds41_mix_fin; Ctx::haz_mark at recording): 0 off (today:
 *                   a routed layer's report copy -- the card-side copy of its expert counts into the routing row,
 *                   Ctx::route_report_impl -- is an op-less dispatch, so haz_mark plays it ordered AND opaque, and the
 *                   dispatch after it waits for it to finish: on deepseek41 two such copies a routed layer, before the
 *                   routed GEMMs' prep and before the next layer's post-collapse norm, 80 a decode pass); 1 the copy is
 *                   recorded with its two ranges (reads the counts source, writes its routing row), so it stays ordered
 *                   behind whatever writes its source (the scatter) or anything opaque, and the next dispatch may start
 *                   unordered beside it when it touches neither range. Nothing in a pass reads a routing row but the
 *                   pass-end flush (route_flush: an op-less copy, ordered behind everything), and the acquire rule
 *                   treats the copy as before (a window it cannot see into). Launch mechanics only: the same kernels on
 *                   the same bytes; the routing rows and the heat engine's histograms are the same. A change re-records
 *                   every pass (the epoch), as aql_acq's does. RADIANCE_AQL_ROUTE_OV=1 at start starts it on. GET "rco":
 *                   in_force, and since start the passes recorded under it, the copies tracked and the dispatches after
 *                   one that play unordered (each recording counts once).
 *   ds41_f32dec     libr4d (lever SPD-F, ds41-etp-spd; 77 here after exl3_dng, 73 on ds41-etp-spd): the decode band's (<= 63 rows) f32
 *                   projections in wider tile forms -- 1 the router's fused launch (router_gemm_f32_scored_topk_scatter) and
 *                   gemm_nt_f32 (the compressor's kv | gate), 2 the router only, 3 gemm_nt_f32 only; 0 the stock rule (today).
 *                   Every form keeps each output's stock arithmetic: the same bytes. RADIANCE_DS41_F32DEC at load.
 *   exl3_xr         exl3r4d (lever XR, ds41-etp-xr, 73 there; SPEED-C order 3, ds41-etp-spc-plug; 78 here after ds41_f32dec):
 *                   gate_up's map form: the order of a chunk's groups -- 0 longest first (today), 1 VRAM-read first, 2 host-read
 *                   first, 3 INTERLEAVE (SPEED-C: the host-read entries spread evenly over the chunk; 11 / 19 / 27 over its first
 *                   3/4 / 1/2 / 1/4); bit 2 XR's ring copy (only with RADIANCE_DS41_XRING). The same entries in another order:
 *                   the same bytes. Needs the host pools at bringup (host_pools / rad_host_pool_ranges).
 *   ds41_wmk        libr4d (lever WMK, ds41-wmk2; 71 here, after route_copy_ov): 1 every dense GGUF GEMM of <= 48 rows
 *                   (gemm_nt_ggml, logits_gemm_ggml, gemm_nt_ggml_pair as one launch, the K1 folds a / e after their
 *                   producer's own launch) in the part-order WMMA form (gg_wk_kernel: K split into P = 8 / 4 / 2 / 1 parts
 *                   by K alone, each part the WMMA form's arithmetic, the parts' f32 partials added in part order and
 *                   rounded once). A ONE-TIME REFERENCE CHANGE (user OK 13.11.2026): the bytes are not the decode band's;
 *                   a row's bytes do not depend on M, its place or the launch geometry. Above 48 rows the stock WMMA /
 *                   GPF forms run as before. 0 = today's paths. RADIANCE_DS41_WMK=1 at load starts it on. GET "wmk":
 *                   in_force, launches / rows issued in the form, calls declined to today's path (every rank; recorded
 *                   passes replay without issuing).
 *   sched_tail_grid scheduler (lever G, ds41-tailgrid; 73 here, after ds41_moe1; Scheduler::tail_checkpoint): 0 off (today:
 *                   under --prefix-resume exact a prompt's tail snapshot is where its last chunk starts on the resume
 *                   grid -- every 4096 tokens from 2048 on the DS line -- so a follow-up turn re-prefills up to one grid
 *                   step of rows the cache already holds); F = 128 / 256 / 512 / 1024 / 2048: the tail snapshot of a plain
 *                   prompt of N tokens is at the deepest multiple of F at or below N - 128, when that lies past the grid
 *                   point (so the prompt's last chunk is cut there, each part >= 128 rows: the row count above which no
 *                   kernel form of the line depends on a piece's rows, option D's floor, sched_bi_chunk 3 / 4), and the
 *                   next turn resumes there: it re-prefills 128 .. 128 + F - 1 old rows instead of 1 .. 4096. The
 *                   snapshot is a chunk's end like every other (linear state, window and n-gram state as its last row
 *                   leaves them) and one of lever PSK's targets (its cone is run), so the hit restores what a cold run
 *                   holds there; from it the request runs to the end of the cold piece that holds it (cut where option D
 *                   cuts) and goes on on the cold run's grid. A request latches the value at its admission (a change
 *                   never moves the targets of a prompt in flight); inert (counted) where F is not a multiple of the
 *                   quantum and a divisor of the interval and the grid, or the resume is not exact. Not on a cone replay,
 *                   a media / embedding / scored request or a runtime window. RADIANCE_SCHED_TAIL_GRID=F at start.
 *                   GET "tailgrid": in_force, latched (requests), tails / tail_rows (fine tail snapshots planned and the
 *                   rows between each and its prompt's end), resumes / resume_rows_saved (hits resumed off the grid and
 *                   the rows below them the grid point would have re-prefilled), rejoins (plans from an interval
 *                   boundary off the grid kept inside the cold piece; counted a plan, not a step), inert (admissions
 *                   where F does not fit the geometry), declined (a fine snapshot within 128 rows of a prompt's end not
 *                   resumed at, or a request's fine tail dropped beside a shared head's snapshot).
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
 *   gg_gemv_m16     libr4d (lever A4, ds41-a4; 64 here, after ar_xpush): 0 off (gemm_nt_ggml / its pair / folds a and e /
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
#include <functional>
#include <string>
#include <vector>

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
              kSchedRowCap, kDs41MixWv, kDs41Gq, kDs41E2, kSchedCbHoldMs, kSchedCbHoldK, kExl3X3d,
              kArXpush, kGgGemvM16, kPlaceSwapLent, kHeatMinGainX100, kHeatHystX100, kDs41Rfuse, kDs41MixFin,
              kRouteCopyOv, kDs41Wmk, kDs41Moe1, kSchedTailGrid, kDraftDcap, kDraftDcapN, kExl3Dng, kDs41F32Dec,
              kExl3Xr, kSchedTailInner, kStageLone, kDraftCostAge, kDraftShortBlk, kDs41Prefetch, kDs41AttnDec, kDs41Sphc,
              kDs41SphcAttn, kDs41SphcSeg, kPpBgRoom, kPpBgJoinMs, kDs41Twin, kDs41Tsk, kDraftHeadCut,
              kPpFolReserve, kPpLeadSeg, kPpSegSet, kExl3X3dPf, kPpTband, kPpMerge, kPpMergeMs, kPpRband, kSchedSrf, kAqlHdp, kDs41Csplit, kSchedLaneRoom, kPpSendAsh, kAqlWaitk, kAqlIka, kDs41Rr1, kDsparkMhr, kDsparkHs, kHostSpin, kPpBgSplit, kAqlCik, kHostSlack, kPpTbandBound, kSampleNarrow, kExl3X6w, kSwitches };
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
/* Lever RCO (ds41-launch): the route_copy_ov switch once set, else RADIANCE_AQL_ROUTE_OV (1 = on) at start; read by
 * Ctx::route_report_impl and Ctx::haz_mark while a pass is recorded. note_rco adds one recording's counts (the
 * routing copies tracked, the dispatches after one left unordered) to GET "rco". */
int  route_copy_ov();
void note_rco(int64_t copies, int64_t freed);

/* The scheduler, at its init: whether its drafter can run the adaptive window, and whether it does. */
void draft_register(bool capable, bool adaptive);
/* tp8 loop D23 (draft_short_blk): the engine, once its short block is resolved at bring-up -- bit N set for each short
 * block of N rows (1 .. 63) the drafter declared, 0 when no short block is served (no drafter, --draft-short-from 0, no
 * declared block); and each query pass the switch moved (its rows). */
void short_register(uint64_t mask);
/* tp8 loop IDEAS #9 (pp_fol_reserve): the follower, at its loop's start -- bit L set for each layer L the body reserve may
 * start at (its declared segment boundaries inside W(h, t)), 0 when it has no BG lane / no segments; -1 until registered (the
 * leader never registers: it checks the range, the follower's verdict decides). */
void reserve_register(uint64_t mask);
/* tp8 loop D32 (pp_lead_seg): the leader's lane loop at its start -- whether it took leader boundaries (RADIANCE_PP_LEAD_SEG
 * with a BG lane); -1 until registered (one machine, the follower: the switch is the leader's, its range checked only). */
void lead_seg_register(bool capable);
/* tp8 loop D39 (pp_tband): the leader's lane loop at its start -- whether it took RADIANCE_PP_TAIL_BAND=1 (the plugin's tail band,
 * a BG lane); -1 until registered (one machine, the follower: the switch is the leader's, its range checked only). tp8 loop D46:
 * level 2 = the tail band WITH the plugin's copy (rad_pp_tail_copy): pp_tband 3 taken; 1 = without (3 refused). */
void tband_register(int level);
/* tp8 loop D33 (pp_seg_set): the follower, at its loop's start -- how many declared segment boundaries lie inside its W(h, t)
 * (0: none: RADIANCE_PP_SEG_LAYERS unset / one lane); -1 until registered (the leader never registers: it checks the range, the
 * follower's verdict decides). */
void segset_register(int n);
void note_short_blk(int64_t rows);
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
/* SPB (draft_dcap): the cycles whose depth choice the cap bounded, and the moves down it forced (GET "dcap"). */
void note_draft_dcap(int64_t cycles, int64_t moves);
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
/* Lever G (ds41-tailgrid, sched_tail_grid): the switch once set, else RADIANCE_SCHED_TAIL_GRID at start (0 = off). The
 * scheduler reads it when it admits a request (Request::tail_grid) and reports its counts every step (GET "tailgrid",
 * kTailGridCounts values: latched, tails, tail_rows, resumes, resume_rows_saved, rejoins, inert, declined). */
static constexpr int kTailGridCounts = 8;
/* tp8 loop P21 (b): 128 too -- kTailMinPiece, a multiple of the quantum and of every compressor ratio (DeepSeek's 4 / 128),
 * dividing the interval and the resume grid: the 64k / 128k cold cells' fine tails 242 / 193 rows before the end, not 370 / 321 */
inline bool tail_grid_ok(int v) { return v == 0 || v == 128 || v == 256 || v == 512 || v == 1024 || v == 2048; }
int  sched_tail_grid();
void note_sched_tail_grid(const int64_t* v);
/* Lever TI (ds41-pf2, sched_tail_inner; 79 here, after exl3_xr): 1 takes lever G's fine tail snapshot INSIDE the prompt's
 * last chunk instead of cutting the chunk there (Scheduler::plan_chunk), where the engine can save the snapshot's state at
 * that row (Engine::save_checkpoints: the window pages are held through the chunk; a CONV group's window is patched from
 * the plugin's rad_ckpt_conv_src; a LINEAR state is the chunk end's, which a resume at a multiple of every ratio never
 * reads -- deepseek41's compressor tails). 0 = today (the cut). RADIANCE_SCHED_TAIL_INNER=1 at start. */
int  sched_tail_inner();
/* tp8 loop P44 (sched_srf): the switch once set, else RADIANCE_SCHED_SRF at start (0 = off); read at every step planned. */
int  sched_srf();
/* tp8 loop D56 (sched_lane_room): the switch once set, else RADIANCE_SCHED_LANE_ROOM at start (0 = off); read at every room gate. */
int  sched_lane_room();
/* tp8 loop U1 (aql_hdp): the switch once set, else RADIANCE_AQL_HDP_FLUSH at start (0 = off); read at every doorbell. */
int  aql_hdp();
/* tp8 loop P46 (pp_send_ash): the switch once set, else 1 where RADIANCE_PP_SEND_ASH=N allocated the staging buffers (0 = off); the
 * engine writes it into this node's send rings (RadPpRing::ash_on) whenever a switch changes. */
int  pp_send_ash();
/* tp8 loop W1s (aql_waitk): the switch once set, else RADIANCE_AQL_WAITK at start (0 / 1 / 2; unset = 0); read at every event / ride
 * wait the AQL backend enqueues (rad::aql's wait_ahead). */
int  aql_waitk();
/* tp8 loop W3s (aql_ika): the switch once set, else 1 where RADIANCE_AQL_IKA=1 at start (0 = off); read at every played packet
 * (rad::aql's PlayRun::put): 1 = lever IKA's fences, 0 = the packets without the lever; 2 (lever IKB) = 1 + core's copies / fills
 * in their self-acquiring forms (read at their launch: rad::aql::ikb_live_now). */
int  aql_ika();
/* tp8 loop D61 (host_spin): the switch once set, else RADIANCE_HOST_SPIN at start (0 = every wait blocks at once, as before). */
int  host_spin();
/* tp8 loop W9 (aql_cik): the switch once set, else 1 where RADIANCE_AQL_CIK=1 / 2 at start (0 = off); read at every live issue and
 * played packet of a chain-capable kernel (rad::aql's dispatch / PlayRun::put, with lever CIK on). Lever CKW (DRAFTER round 108):
 * N >= 2 = CIK for dispatches of at most N work-items only (wider ones issue / play as IKA kernels). */
int  aql_cik();
/* The sites host_spin's M names (bits); kSpinAll = every site. */
static constexpr int kSpinBarrier = 1, kSpinDrafterEnd = 2, kSpinFrame = 4, kSpinNap = 8, kSpinAll = 15;
/* Not a site of its own: the link's receive thread reading the socket (site 4's bound), counted apart ('socket read'). */
static constexpr int kSpinSock = 16 | kSpinFrame;
/* How long a wait at `site` spins before it blocks, ns (0 = it blocks at once: host_spin's N is 0 or M leaves the site out). */
int64_t host_spin_ns(int site);
/* One wait at `site` ended: `caught` when the spin saw the condition (no block), `ns` the time it spun. Counted per site. */
void note_spin(int site, bool caught, uint64_t ns);
/* "host_spin V's waits: rank barrier N caught / M blocked (spun S ms), ..." for the sites with any wait since the last change;
 * empty while nothing spun. */
std::string spin_line();
/* spin_line's counts under host_spin = `label`, then zeroed (apply_pending prints them when host_spin changes). */
std::string spin_take(int label);
/* A pause-loop spin of `us` microseconds where host_spin_ns(kSpinNap) > 0, else std::this_thread::sleep_for(us) as before. */
void nap_us(int us);
/* tp8 loop D63 (host_slack): the switch once set, else RADIANCE_HOST_SLACK at start (0 = each thread keeps its default timer slack). */
int  host_slack();
/* The sleeps host_slack counts, one site each: the lane loops' naps, the link's send proxy, its receive-completion thread (verbs),
 * the ring waits (ring_put / ring_take). */
static constexpr int kSlackNap = 0, kSlackLinkSend = 1, kSlackLinkRecv = 2, kSlackRing = 3, kSlackSites = 4;
/* The calling thread's timer slack follows host_slack(): set (prctl PR_SET_TIMERSLACK, v x 1000 ns; 0 = the thread's default)
 * when the value differs from what this thread applied last; a no-op while the switch was never above 0. */
void slack_apply();
/* slack_apply(), then std::this_thread::sleep_for(us); counted at `site` (asked / slept) once host_slack was ever above 0. */
void slack_sleep_us(int us, int site);
/* "host_slack V's sleeps: lane naps N (asked A us, slept S us mean), link send proxy ..." for the sites with any sleep since
 * the last change; empty while nothing was counted. */
std::string slack_line();
/* slack_line's counts under host_slack = `label`, then zeroed (apply_pending prints them when host_slack changes). */
std::string slack_take(int label);
/* tp8 loop D65 (pp_tband_bound): true with the window (ms) and hysteresis (permille) once a request set them (v = W + 10000 x H);
 * false (both untouched) while unset / -1 / 0 = the start's RADIANCE_PP_TBAND_BOUND_MS / _H. */
bool pp_tband_bound(int* ms, int* h);
/* DRAFTER round 79 (sample_narrow, lever SNW): the switch once set, else RADIANCE_SAMPLE_NARROW at start (1 = the narrow
 * chain declared and issued for all-top-1 steps); sample_narrow_env is the start's env alone (the sampler's declare). */
int  sample_narrow();
int  sample_narrow_env();
/* Lever TI: the plugin's source of a CONV group's window at a row inside a step (rad_ckpt_conv_src): for rank `rank`, the
 * device address of the step's mapped token ids (one i32 a row, in the batch's row order) and the window's ids `ctx`
 * (conv_width - 1). 1 when a plugin serves it, else 0. */
int  ckpt_conv_src(int rank, const void** ids, int* ctx);
/* Lever SAL (ds41-pf2, stage_lone; 80 here, after sched_tail_inner): 1 = lever SA's stage_short arms the prefill stager only on a
 * step with no decode rows (a lone prefill's chunk at the span's shared level: a prompt shorter than the span's minimum, a long
 * prompt's first interval, a turn's tail), never beside decoding sequences; 0 = stage_short as built (any step it admits). The
 * engine's (handed over at a step release, as stage_short). */
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
/* Lever XR (ds41-etp-xr): hand rank `rank`'s pinned host pools ([lo, hi) as its card addresses them, Mover::pool_ranges)
 * to every loaded plugin that takes them (rad_host_pool_ranges; exl3r4d's gate_up map form tests its experts' codes
 * against them). Call with the rank's card current. Returns the plugins that took them. */
int  host_pools(int rank, int n, const uint64_t* lo, const uint64_t* hi);
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
 * ix_ni, index_candidates launches under ix_ni; out[10] (tp8 loop P38) WMMA-form index_score launches under lever BFR (ix_lds
 * bit 2). 0s where no plugin serves them. */
constexpr int kI9bStats = 11;
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
/* Levers FUSE-R / FUSE-M (ds41-fuse): deepseek41's rad_rfuse_stats [0..3] (forms declared, layers, passes in form 1, in
 * form 2; rank 0) then libr4d's rad_fuse_stats [4..7] (mhc_mix FIN launches under ds41_mix_fin, their rows, the fused
 * router's launches, their rows; every rank); 0s where no plugin serves them. */
constexpr int kFuseStats = 8;
void fuse_stats(int64_t out[kFuseStats]);
/* Lever WMK (ds41-wmk2): libr4d's rad_wmk_stats [0..3] (ds41_wmk now, part-order launches issued, their rows, calls declined
 * to today's path); 0s where no plugin serves it. */
constexpr int kWmkStats = 4;
void wmk_stats(int64_t out[kWmkStats]);
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
/* tp8 S5 review: a two-node pipeline leader with lanes, whose steps interleave in the ring the skip keeps:
 * draft_ctxskip=1 is refused naming that (GET "ctxskip" capable 2), so a run's switch probe drops it by name. */
void ctxskip_refuse_lanes();
/* The engine, at bring-up (lever Z0-A): whether arena_d0 can act (a decode rung below the largest declared). */
void arena_d0_register(bool capable);
/* The engine, at bring-up (lever Z0-W): whether draft_lend_w can act (declared and set up on every rank). */
void draft_w_register(bool capable);
/* The engine, at bring-up (lever LF): whether place_lf can act (every rank places dynamically). */
void place_lf_register(bool capable);
/* The engine, at bring-up (levers SWL / L2, ds41-swl): whether place_swap_lent, heat_min_gain_x100 and heat_hyst_x100
 * can act (every rank places dynamically: a heat engine and a mover that move). */
void swl_register(bool capable);
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
/* Whether a request may set `name` to `value` (epoch_bump included); false with the reason. On a
 * two-node pipeline's leader (set_peer_check) a plugin switch this node refuses is taken when the
 * other node takes it. */
bool check(const std::string& name, int value, std::string* err);

/* ---- tp8 S4: the two-node pipeline (review F2 / F3). Unused on one machine: no peer is set, the
 * request counts are only counted and no "peer" object is reported. ---- */
/* The switches a pipeline's follower takes from its leader: those that act on a node's own passes
 * or placement -- every plugin switch, aql_acq, route_copy_ov, place_lf, place_swap_lent,
 * heat_min_gain_x100, heat_hyst_x100, place_hb. The scheduler's and the drafter's are the leader's. */
bool follower_switch(int i);
/* Every request this node applied (apply_pending) per switch: how many so far, and the last value
 * asked (as asked, not as applied: a leader that holds no layer of a lever reports it 0). */
void requests(uint32_t* count, int32_t* value);
/* The follower: request every follower_switch whose count advanced past `taken` (updated); a value
 * this node refuses is skipped with its reason in `skipped` (a lever it does not declare), never a
 * failure. Returns how many were requested; apply_pending applies them. */
int  follow(const uint32_t* count, const int32_t* value, int n, uint32_t* taken, std::vector<std::string>* skipped);
/* The leader, once the link serves: asks the follower for its verdict on name = value (1 takes, -1
 * refuses with the reason, 0 no answer). */
using PeerCheck = std::function<int(const std::string&, int, std::string*)>;
void set_peer_check(PeerCheck f);
/* This node's own verdict, 1 or -1 (the follower's answer to the leader's ask). */
int  check_node(const std::string& name, int value, std::string* err);
/* The leader: the follower's applied values as it reported them (GET's "peer" object). */
void note_peer(const int32_t* v, int n);
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
