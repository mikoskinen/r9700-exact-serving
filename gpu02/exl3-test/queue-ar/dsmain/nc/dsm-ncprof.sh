#!/bin/bash
# dsm-ncprof.sh (12.11.2026, queue-ar/dsmain/nc, ds-main-port NC-profile builder) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r00zzz-cz-ncprof.sh. Log tag "RESULT xnc ", out runs/xdsm/ncprof (XNC_OUT).
# THE NEWCOMER (NC) PROFILE of the DS line (DeepSeek-V4.1 EXL3 2.25, gpu02 TP4) on the int7 image, ~28 min (estimate: two starts
#   at ~4.5 min, START U ~15 min in all with section (d) ~6.5, START P ~10, parsing ~2-3; budget <= 30).
# WHY (coordinator 12.11., existing logs): mb c8 (8 closed-loop streams, ~1099-token prompts, 512 out) reads 172 tok/s steady
#   (stack4 T 0: A 172.2 / 171.9 / 170.6 / 165.7, S 173.2 / 172.9 / 172.5 / 174.8); decode alone ~220 tok/s a decode-second, but
#   decode_share 0.784-0.791: each newcomer costs one non-decode step of ~0.61 s (stack4 arm A: 949 steps - 933 decode = 16 for 16
#   requests, 9.7 s of 45) -- 1099 tokens at ~1.8k tok/s vs 3.4-4.3k tok/s for a 136k prefill; alone TTFT 0.62-0.70 s, about the
#   same. c8 >= 200 tok/s needs <= ~0.25 s a newcomer, or the newcomer overlapped with decode.
# THREE QUESTIONS: (a) STAGING: the expert bytes / units a ~1.1k prefill step stages / streams host-to-device, the compute lane's
#   wait for them, the distinct experts it touches (hypothesis to TEST: 1.1k rows touch nearly every expert, so the step is
#   staging-bound -- the start log: "a step of more than 1024 tokens copies the pooled experts of 40 routed layers onto the
#   card", core/place/stager.cpp:123 arms at n_tok >= min_rows); (b) at c8, does the newcomer run as a MIXED step (the 7 decode
#   rows ride) or a pure prefill step; (c) the newcomer step's ms by class, per rank, + host remainder.
# (b) BY THE CODE (~/gpu/ds41/exl3/radiance-int7 = image source 0391a6e), to be confirmed by the step log: a MIXED step, the
#   decode rows ride. Scheduler::step() plans the running requests first (core/sched/scheduler.cpp:2601-2721: each decode row
#   1 + its drafts) and then admits waiting requests INTO THE SAME PLAN (:2807-2852, admit_one); a plan with prefill and decode
#   entries is RAD_PHASE_MIXED (:2993). None of the hold rules applies to a ~1.1k newcomer: --prefill-interleave 8 holds only
#   after a chunk that LEAVES ITS PROMPT UNFINISHED beside decoding sequences (:3009-3012 `continues`; the hold itself :2521 /
#   :2631 / :2813) -- a 1099-token prompt is one chunk that produces its token; --prefill-span 2 spans only a lone prefill
#   (:2538-2552) of >= 16384 tokens (span_min_, start log), so beside 7 runners the step is planned at the shared budget
#   (:2554) = sched_mixed_tok 2112 (core/sched/geometry.cpp:40: max(min(4160, 2048), 4160 - 2048)), and 7 x (1 + <= 5) +
#   1099 <= 2112; --busy-prefill-chunk auto never cuts under the stager's 1024 (:1821) and reserves only for a waiting
#   request; NST 5 rows mix ("A mixed step takes drafts like any other", :2557); issue-ahead stops while a request waits
#   (:3264), so the step after the newcomer's arrival is a full step() that admits it. ds41_bi 2 is the plugin's segmentation
#   (arch/deepseek41/deepseek41.cpp:3395: a step of more than one shape class runs in segments, the MoE a pass a segment) --
#   it changes the compute of the MIXED step, not its planning; sched_bi_chunk is 0 (stack4's applied value); PSK design 2
#   (psk_latch :1736 / psk_ckpt :1747) only moves snapshots. So the decode rows RIDE, but in a step as long as the prefill.
# DESIGN (coordinator's): image radiance-local:ds41-int7-0391a6ec-5c13e555-l (b54f9224a056). Start = stack4's NST 5 start exactly
#   (dsm-stack4.sh md5 21ce8c8e, its 2026-11-11 21:51:34 "RESULT xs4 s4-n5: start": stackenv + LEVENV + the DLW sidecar + Z0 words,
#   best_vars + MNBT 4160 NST 5 MAXSEQS 8 MML 524288 + --live --draft-window adaptive --draft-coupling on, the fixed pin FIX)
#   + RADIANCE_LOG_STEPS=1. Arms in process (POST /admin/fn4 pin + arm + epoch_bump, a nudge, the applied switches READ BACK):
#   S = {"ds41_bi": 2, "draft_sample": 85, "draft_occ": 6, "place_lf": 0, "ds41_psk": 2, "ds41_psk_k": 4} (stack4's S),
#   A = {"ds41_bi": 0, "draft_sample": 0, "draft_occ": 0, "place_lf": 0, "ds41_psk": 0} (stack4's A).
#   START U (UNPROFILED, docker logs -t: every step line with its receipt time): gate1 (restart + warm-up + fastcheck), modes,
#     start switches, warm-up answers vs stock; then arm S, then arm A: (1) ALONE: XNC_ALONE (12) fresh ~1.1k prompts one at a time,
#     one token each (ncone.py: mb.py's nonce-first prompt at ctx 1000, effort medium, T 0; /stats between prompts); (2) C8: one mb
#     c8 window exactly as stack4 (ncmb.py --mode phase = stack4/mbt.py's mb.phase_c at T 0, dur 60 warm 15, + stack4's 2 s
#     /stats + /admin/fn4 sampler; taps: per-request send / end, full /stats at the window's edges).
#   START P (PROFILED: --profile-ops, RADIANCE_PROFILE_EVERY 16 (XNC_EVERY), + RADIANCE_DEBUG_MISS_LOG=/stats/miss-p.log -- the per
#     (step, rank) live-expert log = distinct experts touched; arm S only): restart, warm-up answers only (no 16k / fastcheck: fewer
#     profiled steps, a smaller log), modes, arm S; (1) ALONE: XNC_PWARM (16) one-token prompts (a table dump falls inside), mark,
#     XNC_PALONE (32) more, mark -> the window holds only newcomer prefill steps (16-32 of them); (2) AT C8 BY DIFFERENCE: W1 = 8
#     closed-loop streams, gen 4096, warm 20 s, window 20 s, decode only, then the streams cut (the server cancels them: no drain);
#     W2 = the standard mb c8 (gen 512, ctx 1000), warm 15 s, window 40 s, cut.
#     DEVIATION (stated): every 16 and 16 + 32 prompts instead of every 8 and 24: at c8 a profiled step runs ~16 a second and a
#     dump (4 ranks x ~1.5k op rows) is ~0.9 MB, so every 8 over the c8 phases is ~250 dumps = ~230 MB of raw log (json-wrapped
#     ~350 MB) -- over the ~100 MB raw-log rule and near docker's own cap (compose/common.yaml json-file max-size 100m x 5 files:
#     past it the log's start, the ALONE window included, is gone). Every 16 with these windows is ~100 dumps, ~90-100 MB raw. docker stop closes the last table. Newcomer step ms a class = (W2 - W2's
#     decode steps x W1's per decode step) / W2's non-decode steps (an ESTIMATE, labelled) + a second reading inside W2 (the 8-step
#     dump intervals with one newcomer minus their decode steps x W2's decode-interval mean). + the coordinator's 12.11. split:
#     dspark_ctx (the DSpark prefill context pass over every trunk row: deepseek41.cpp:3594-3618 dsp_context, scheduler.cpp:4157
#     block_context) as its own class with its rows, mhc by op, Engram kv GEMM vs lookup / ngram ids, index_score vs index_topk.
#   SECTION (d), START U after the arms (coordinator 12.11.: D7 under a long prefill -- in r00zz's c4 + one cold 64k deepq prompt
#     the D7 depth choice (draft_depth0 1: cost model + probes) drafted in some runs and held depth 0 in others: ld-1D 608 drafted,
#     4.57 tok/step, 28.9 ms, TTFT 39.76 s, c4 30.8 tok/s vs ld-4D 8356 drafted, 10.17 tok/step, 59.2 ms, TTFT 56.71 s (+43 %),
#     c4 23.0, non-decode 43 -> 47 steps at 848 -> 959 ms; stack4's loaded runs all drafted, TTFT 41.8-55.2 s; D7 prices decode
#     steps only): arm S pins + X1 = draft_depth0 1 (today) / X2 = draft_depth0 2 (depth 0 forced: only the probes draft, every
#     draft_probe cycles under the back-off), order XNC_DORDER (1221 = ABBA). Each run: the pins read back, a prefix-cache reset,
#     /metrics, stack4's loaded probe exactly (stack4/pskld.py run --which deep --depth 64000 --ntok 512 --load --conc 4 --warm 10
#     --pf-tok 0: c4 mb streams at mb.py's sampling + deepq.py's fixed ~64k prompt, 512 greedy tokens), /metrics. Readings: probe
#     TTFT; c4 agg tok/s and per-stream p50 in the prefill window; TPOT p50 / p90; the /metrics deltas (decode steps, ms and tokens
#     a decode step, drafted / accepted, non-decode steps and their mean ms); the step log's MIXED / prefill / decode steps in the
#     run and the probe's chunks (pskld.py's find_chunks: sizes, boundaries off the 2048 grid, the decode rows and their drafted
#     tokens riding each chunk step, its wall). Guard: X2 drafted ~0 (<= max(800, 10 % of X1's)), pins read back, cold (cached 0),
#     status 200, no c4 error. Identity: every answer vs the first X1's (X1 vs X1 and X2 vs X2 too), the chunk boundaries compared;
#     a difference is reported, never hidden. DEVIATION (stated): the SAME fixed deepq prompt every run, made cold by POST
#     /reset_prefix_cache before it (stack4's and r00zz's ld-* method, guarded by cached == 0), not a distinct text a run -- the
#     identity X2 vs X1 needs the same prompt. "RESULT xnc (d) D7 under a long prefill: ..." with X2 vs X1 in % per reading.
# READINGS -> "RESULT xnc (a) / (b) / (c) / (d) / summary" lines (ncsteps.py report): (a) staged units / GB, copies, layers, h2d / link
#   staged / stream GB a newcomer step ALONE (/stats per prompt) and at c8 (window / non-decode steps + a least-squares split over
#   the 2 s samples), lane waits a step (stager vs other), distinct experts touched (miss log); (b) MIXED vs prefill counts, decode
#   rows riding each newcomer step, newcomer vs decode step wall at c8, S and A; (c) per rank, the ranked class table (ms a step,
#   share of wall) + host remainder (wall - device - lane waits), ALONE and at c8 by difference.
# GUARD (standing rule: a reading that silently did nothing = VIRHE, no reading): step-log lines in every window; profile tables
#   on all 4 ranks in every profiled window; /stats step counters moved; each arm's pins read back as applied; the newcomers
#   fresh (prompt ~1099 tokens, cached <= 64); every alone prompt exactly one prefill step. A failure prints VIRHE.
# Disk: >= XNC_DISKMIN_MB (11000) free on / before each start (the 10 GB guard + ~0.6 GB: docker's json log of the profiled start,
#   ~150-250 MB, lives until the container is removed); engine logs written gzipped only (never raw), the miss log gzipped.
# NO cc_sim, no evict, no power / host / fan settings; production compose files read only (--project-directory). Helpers
#   reused READ ONLY (lib-29.sh / lib-pr266.sh, mb.py, ds41_tests.py, stack2's yaml); copied functions from dsm-stack4.sh.
# Knobs: XNC_IMAGE, XNC_ITEM, XNC_OUT, XNC_UARMS ("S A"), XNC_ALONE (12), XNC_PWARM (16), XNC_PALONE (32), XNC_DUR (60), XNC_WARM (15),
#   XNC_EVERY (16), XNC_W1 (gen:warm:dur 4096:20:40), XNC_W2 (512:15:55), XNC_DISKMIN_MB (11000), XNC_P (1 | 0: no START P),
#   XNC_D (1 | 0: no section (d)), XNC_DORDER (1221), XNC_LDEPTH (64000),
#   XNC_SIDECAR, XNC_SIDESHA.
# DRYRUN=1 bash dsm-ncprof.sh : prints every command; checks the image literals, every helper (files, parse, selftests), the
#   sidecars, the yaml, the disk; no server, no GPU, no request to :8090, writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/../ds41-stack2-tp4.yaml                   # stack2's yaml, read only (env_file ${XS2_ENVFILE} carries every start word)
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh
O=${XNC_OUT:-$R/xdsm/ncprof}; export STATSDIR=$O/stats-u
RS4=$R/xdsm/stack4; RZ2=$R/xdsm/z0occ2
export IMAGE=${XNC_IMAGE:?set XNC_IMAGE}
ITEM=${XNC_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-ncprof}
UARMS=${XNC_UARMS:-S A}; NALONE=${XNC_ALONE:-12}; PWARMN=${XNC_PWARM:-16}; PALONEN=${XNC_PALONE:-32}; DUR=${XNC_DUR:-60}; WARMS=${XNC_WARM:-15}
EVERY=${XNC_EVERY:-16}; W1S=${XNC_W1:-4096:20:40}; W2S=${XNC_W2:-512:15:55}; DISKMIN=${XNC_DISKMIN_MB:-11000}; DOP=${XNC_P:-1}
DOD=${XNC_D:-1}; DORDER=${XNC_DORDER:-1221}; LDEPTH=${XNC_LDEPTH:-64000}
SIDE=${XNC_SIDECAR:-dspark-dlw-q8_0.rad}; SIDESHA=${XNC_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
IFS=: read -r W1GEN W1WARM W1DUR <<< "$W1S"; IFS=: read -r W2GEN W2WARM W2DUR <<< "$W2S"
case $DORDER in *[!12]*|"") echo "XNC_DORDER takes 1 (X1) / 2 (X2) (got '$DORDER')"; exit 1 ;; esac
for v in $NALONE $PWARMN $PALONEN $DUR $WARMS $EVERY $W1GEN $W1WARM $W1DUR $W2GEN $W2WARM $W2DUR $DISKMIN $DOP $DOD $LDEPTH; do case $v in ''|*[!0-9]*) echo "a numeric knob is not a number ('$v')"; exit 1 ;; esac; done
case " $UARMS " in *[!SA\ ]*) echo "XNC_UARMS takes S / A (got '$UARMS')"; exit 1 ;; esac
[ $PWARMN -ge $EVERY ] || { echo "XNC_PWARM ($PWARMN) must be >= XNC_EVERY ($EVERY): a table dump must fall inside the warm prompts"; exit 1; }
[ $W1WARM -lt $W1DUR ] && [ $W2WARM -lt $W2DUR ] && [ $WARMS -lt $DUR ] || { echo "a warm-up is not shorter than its phase"; exit 1; }
NCONE=$HD/ncone.py; NCMB=$HD/ncmb.py; NCS=$HD/ncsteps.py; OPNC=$HD/ds_opprof_nc.py
MBT=$HD/../stack4/mbt.py; OPP=$HD/../ds_opprof.py; PF1=$HD/../pf1.py; S4B=$HD/../stack4/dsm-stack4.sh   # references, read only
PL=$HD/../stack4/pskld.py                       # section (d): stack4's loaded probe, read only
MBP=$D/mb.py; TESTS=$D/ds41_tests.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
NV=0
xlog() { case "$*" in *VIRHE*) NV=$((NV + 1)) ;; esac; log "RESULT xnc $*"; }
# ---------------------------------------------------------------- copied from dsm-stack4.sh (md5 21ce8c8e; tag xs4 -> xnc)
GPTAB=q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422
SIDECAR=dspark-conf.f32
LEVENV="RADIANCE_DS41_GG_PF=1 RADIANCE_DS41_GG_PF_TABLE=$GPTAB RADIANCE_DS41_F32_PF=25 RADIANCE_DS41_ARENA_DECODE_ROWS=8,48
  RADIANCE_DS41_ROUTER_BF16=1 RADIANCE_DS41_EMB_RAM=1 RADIANCE_PLACE_SHADOW_SLOTS=16 RADIANCE_XPREFETCH=0 RADIANCE_DS41_DSPARK_WALK=1
  RADIANCE_DS41_ARGATHER_MAX=48 RADIANCE_DS41_QNFOLD=1 RADIANCE_DS41_SWFOLD=1 RADIANCE_DRAFT_CTXSKIP=0 RADIANCE_DS41_DSPARK_CONF=/models/$SIDECAR
  RADIANCE_DRAFT_LEND=0 RADIANCE_DRAFT_PROBE_BACKOFF=0 RADIANCE_DRAFT_LEND_BORROW=0 RADIANCE_DRAFT_SAMPLE=0 RADIANCE_WEIGHT_SIDECAR=/models/$SIDE
  RADIANCE_DS41_BI=0 RADIANCE_DRAFT_CTXSKIP_SLOTS=129 RADIANCE_DRAFT_LEND_WEIGHTS=0 RADIANCE_DS41_PSK=0"
LEVENV=$(echo $LEVENV)
LITS="gemm_nt_ggml's,prefill,rows,take,lever,GPF's,forms,(RADIANCE_DS41_GG_PF=1|gemm_nt_f32's,prefill,rows,take,lever,GPF-b's,tiled,form,(RADIANCE_DS41_F32_PF=25"
LITS="$LITS|48,tokens,use|a,rank),ram|expert,prefetch:,off"
LITS="$LITS|DSpark,lever,W1:,the,full-vocabulary,Markov,walk,declared,(dspark_walk_step,a,step,+,all_gather,over,4,rank(s)|fn4,dspark_walk=1,"
LITS="$LITS|one,kernel,up,to,48,tokens,a,step|RADIANCE_DS41_QNFOLD;,ds41_qnfold=1)|RADIANCE_DS41_SWFOLD;,ds41_swfold=1)|draft,context,skip,(RADIANCE_DRAFT_CTXSKIP=0|a,confidence,plane,(the,dspark_conf,gate"
LITS="$LITS|draft,lend:,set,up"
LITS="$LITS|DSpark,lever,SD:,the,sampled,full-vocabulary,walk,declared,(dspark_walk_sample_step,a,step,over,4,rank(s)|lever,SD:,fn4,draft_sample,0,at,start"
LITS="$LITS|weight,sidecar:,43,entries|read,from,the,weight,sidecar|lever,BI,declared,(RADIANCE_DS41_BI|now,0):,a,step,of,more"
LITS="$LITS|lever,Z0-A,set,up|activation,arena:,8,tokens,use|ring,of,129,steps|draft,weight,lend,(lever,Z0-W|mover:,lever,Z0-W:"
LITS="$LITS|lever,PSK,declared,(RADIANCE_DS41_PSK"
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": 8, "drafter_mx": 1, "gg_pf": 1, "f32_pf": 25, "dspark_walk": 1, "ds41_argather": 48, "idx_split": 1, "ds41_qnfold": 1, "ds41_swfold": 1, "draft_ctxskip": 1, "dspark_conf": 1, "dspark_conf_rows": 1, "draft_lend": 1, "draft_probe_backoff": 3, "draft_hybrid": 1}'
FIX=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update({"draft_lend_hyst": 0, "arena_d0": 0, "draft_lend_w": 0, "draft_occ_k": int(sys.argv[2])}); print(json.dumps(d))' "$FIX5" 32) || { echo "pin JSON"; exit 1; }
armj() {   # armj S|A: the arm's own switches (stack4's S at XS4_PSK 2 / XS4_PSKK 4 / XS4_LF sl / XS4_OCC 6 / XS4_DSA 85, and A)
  case $1 in
    A) echo '{"ds41_bi": 0, "draft_sample": 0, "draft_occ": 0, "place_lf": 0, "ds41_psk": 0}' ;;
    S) echo '{"ds41_bi": 2, "draft_sample": 85, "draft_occ": 6, "place_lf": 0, "ds41_psk": 2, "ds41_psk_k": 4}' ;;
  esac
}
pinj() {   # pinj ARM [EXTRAJSON ...]: the fixed pin + the arm + the extras as one POST body (a later one wins)
  python3 -c 'import json,sys
d = {}
for a in sys.argv[1:]: d.update(json.loads(a or "{}"))
print(json.dumps(d))' "$FIX" "$(armj $1)" "${@:2}"
}
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DS41_ARGATHER_MAX RADIANCE_DS41_SHEXP_LANE RADIANCE_DEBUG_MISS_LOG
  RADIANCE_DS41_KVFOLD RADIANCE_DS41_OFOLD RADIANCE_DS41_ONEFILL RADIANCE_DS41_PIPE
  RADIANCE_DRAFT_DEPTH0 RADIANCE_DRAFT_PROBE RADIANCE_DRAFT_HYBRID RADIANCE_DRAFT_HYBRID_MATCH
  RADIANCE_DRAFT_LEND RADIANCE_DRAFT_PROBE_BACKOFF RADIANCE_DRAFT_LEND_BORROW RADIANCE_DEBUG_VRAM
  RADIANCE_DS41_IDX_BALANCE RADIANCE_DS41_QNFOLD RADIANCE_DS41_SWFOLD RADIANCE_DRAFT_CTXSKIP RADIANCE_DS41_EMB_RAM
  RADIANCE_DS41_DSPARK_CONF RADIANCE_DS41_DSPARK_ALT RADIANCE_DS41_KVWIN_FP8 RADIANCE_DS41_DRAFTER_MX RADIANCE_DS41_DRAFTER_BF
  RADIANCE_DS41_GG_PF RADIANCE_DS41_GG_PF_TABLE RADIANCE_DS41_F32_PF RADIANCE_DS41_NVGATHER RADIANCE_DS41_DSPARK_WALK
  RADIANCE_DS41_ARENA_DECODE_ROWS RADIANCE_DS41_ROUTER_BF16 RADIANCE_PLACE_SHADOW_SLOTS RADIANCE_XPREFETCH
  RADIANCE_WEIGHT_OVERLAY RADIANCE_DS41_DA_TAPS RADIANCE_KLD_DUMP RADIANCE_DRAFT_SAMPLE RADIANCE_WEIGHT_SIDECAR
  RADIANCE_DS41_BI RADIANCE_DS41_BITRACE RADIANCE_DS41_BITRACE_PMAX RADIANCE_DRAFT_CTXSKIP_SLOTS RADIANCE_DRAFT_LEND_WEIGHTS
  RADIANCE_DS41_PSK RADIANCE_PLACE_LF RADIANCE_PLACE_LF_DECAY RADIANCE_LOG_STEPS"
stackenv() {   # 105's t3 arm + gg_gemv 2 + rule-w + the K1 folds + turnover 0 + LEVENV, every other switch unset
  local kv
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  export RADIANCE_PLACE_TURNOVER=0
  for kv in $LEVENV; do export "$kv"; done
}
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN | gzip > $O/engine-$1.log.gz"; return 0; }; docker logs $CN 2>&1 | gzip -1 > $O/engine-$1.log.gz || true; }
thermline() { type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"; }
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xnc $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { xlog "$1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  echo "$1 warm-up $r" >> $O/identity.txt
  case $r in IDENTICAL*) xlog "$1 warm-up answers vs stock ($(basename $2)): $r" ;; *) xlog "$1 warm-up answers vs stock: VIRHE $r" ;; esac
}
modes() {   # modes TAG: MODEWORDS in the container, mover 0 kept free x4, rule-w, K1, ARGATHER 48, every literal of LITS -> 0, else 1
  [ "$DRY" = 1 ] && { echo "+ modes $1: [$MODEWORDS] in the container's env; mover '0 more kept free' on 4 ranks; rule-w in every recorded pass; K1 folds declared =1; ARGATHER 48-row note; literals [$(echo "$LITS" | tr ',' ' ')]"; return 0; }
  local CE kv bad="" ok=1 NW NR KL AL nz lit found="" miss=""
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in RADIANCE_PLACE_TURNOVER=0 $MODEWORDS; do echo "$CE" | grep -qxF "$kv" || bad="$bad ${kv%%=*}"; done
  [ -z "$bad" ] || { xlog "$1: VIRHE start words not in the container:$bad"; ok=0; }
  nz=$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots \([0-9]+ shadow, 0 more kept free')
  [ "$nz" = 4 ] || { xlog "$1: VIRHE turnover 0 asked but $nz of 4 ranks keep 0 slots free"; ok=0; }
  NW=$(docker logs $CN 2>&1 | grep -c 'the acquire rule plays .*RADIANCE_AQL_ACQ=rule-w)'); NR=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_AQL_ACQ=rule)')
  [ "$NW" -gt 0 ] && [ "$NR" = 0 ] || { xlog "$1: VIRHE rule-w is not what the recorded passes play ($NW / $NR)"; ok=0; }
  KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-500)
  case "$KL" in *"ds41_kvfold=1"*"ds41_ofold=1"*"ds41_onefill=1"*) ;; *) xlog "$1: VIRHE the K1 switches do not start at 1: '${KL}'"; ok=0 ;; esac
  AL=$(docker logs $CN 2>&1 | grep -m1 'the MoE gather and its all-reduce as one kernel' | sed 's/^radiance: //' | cut -c1-300)
  case "$AL" in *"up to 48 tokens a step (RADIANCE_DS41_ARGATHER, RADIANCE_DS41_ARGATHER_MAX"*) ;; *) xlog "$1: VIRHE the ARGATHER note is not the 48-row one: '${AL}'"; ok=0 ;; esac
  IFS='|' read -ra lit <<< "$LITS"
  for l in "${lit[@]}"; do [ -n "$l" ] || continue
    if docker logs $CN 2>&1 | grep -qF -- "${l//,/ }"; then found="$found ${l%%,*}"; else miss="$miss | ${l//,/ }"; fi; done
  xlog "$1 modes: ${#lit[@]} literal(s), found [$(echo $found)]$([ -n "$miss" ] && echo " -- VIRHE missing:$miss") | $(docker logs $CN 2>&1 | grep -E "lever PSK|lever BI declared|prefill staging:|scheduler: --prefill|scheduler: --busy|chunk geometry" | sed 's/^radiance: //; s/^[A-Z] *//; s/^[A-Z]\[[0-9]\] //' | cut -c1-260 | sort -u | tr '\n' '|' | cut -c1-2400)"
  [ -z "$miss" ] || ok=0
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
arm() {   # arm S|A [EXTRAJSON]: the pin + the arm (+ extra) + epoch_bump, a nudge to apply it at a step
  fn4set "$(pinj $1 "${2:-}" '{"epoch_bump": 1}')"; nudge
}
gate1() {   # restart + ONE check; over the limit logged (in-process A/B: a slow state shows in every arm)
  local tag=$1 lim=$2 ms st=fast
  restart $tag || { keeplog $tag-fail; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && return 0
  awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}' && { st="SLOW (not retried)"; }
  echo "$st $ms" > $O/guard-$tag.txt
  xlog "guard $tag: $st, 1-user ${ms} ms/step (limit $lim; stack4 s4-n5 $(cat $RS4/guard-s4-n5.txt 2>/dev/null))"
}
# ---------------------------------------------------------------- this item's own functions
dfree() { df -BM --output=avail $HOME 2>/dev/null | tail -1 | tr -dc 0-9; }
diskok() {   # diskok TAG: >= DISKMIN MB free on / before a start
  local f; f=$(dfree)
  if [ "$DRY" = 1 ]; then echo "+ disk before $1: ${f:-?} MB free on / (a start needs >= $DISKMIN MB) -- $([ "${f:-0}" -ge $DISKMIN ] && echo OK now || echo "LOW now: the run would refuse this start")"; return 0; fi
  [ "${f:-0}" -ge $DISKMIN ] && return 0
  xlog "$1: VIRHE only ${f:-?} MB free on / (need >= $DISKMIN: the 10 GB guard + this item's logs) -- not started"; return 1
}
envfile() {   # envfile FILE WORDS...: the start words the yaml's env_file carries
  local f=$1; shift
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $* > $f"; else printf '%s\n' "$@" > $f; fi
}
envline() { env | grep -E '^(RADIANCE_|R4D_)' | grep -vE '(KEY|TOKEN|SECRET)' | sort | tr '\n' ' ' | cut -c1-2400; }
swstart() {   # swstart TAG: the switches at start are stack4's (switch note's start check)
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 at start: gg_gemv 2, aql_acq 2, K1 three 1, epoch 0, ds41_bi 0, ds41_psk 0, place_lf 0, draft_occ 0, draft_occ_k 32, draft_adaptive 1, draft_depth0 0, draft_capable 1, draft_sample 0 + sample.capable 1, sched_bi_chunk 0 / sched_interleave 8 / sched_mixed_tok 2112"; return 0; }
  local s0 r
  s0=$(fn4get)
  r=$(python3 -c 'import json,sys
s0 = json.loads(sys.argv[1]); w = s0["switches"]
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) and s0["epoch"] == 0
ok = ok and w.get("ds41_bi") == 0 and w.get("ds41_psk") == 0 and w.get("place_lf") == 0 and w.get("draft_occ") == 0 and w.get("draft_occ_k") == 32
ok = ok and w["draft_adaptive"] == 1 and w["draft_depth0"] == 0 and s0["draft_capable"] == 1 and w.get("draft_sample") == 0 and (s0.get("sample") or {}).get("capable") == 1
ok = ok and w.get("sched_bi_chunk") == 0 and w.get("sched_interleave") == 8 and w.get("sched_mixed_tok") == 2112
print(("OK" if ok else "FAIL") + " sched_bi_chunk %s sched_interleave %s sched_mixed_tok %s sched_short_bypass %s ds41_bi %s ds41_psk %s/%s" % (
    w.get("sched_bi_chunk"), w.get("sched_interleave"), w.get("sched_mixed_tok"), w.get("sched_short_bypass"), w.get("ds41_bi"), w.get("ds41_psk"), w.get("ds41_psk_k")))' "$s0" 2>&1 | cut -c1-300)
  case "$r" in OK*) xlog "$1 switches at start: $r"; return 0 ;; esac
  xlog "$1 switches at start: VIRHE $r | $(echo "$s0" | cut -c1-900)"; return 1
}
ARMOK=1
armset() {   # armset TAG ARM [EXTRAJSON]: the pin + the arm (+ extra) + epoch_bump, a nudge, the applied switches READ BACK
  local g r
  arm $2 "${3:-}"
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4: every switch of $(pinj $2 "${3:-}") applied, nothing pending -> RESULT xnc $1 arm $2 ${3:-} applied (read back)"; return 0; }
  g=$(fn4get)
  r=$(python3 -c 'import json,sys
g, w = json.loads(sys.argv[1]), json.loads(sys.argv[2])
bad = {k: g["switches"].get(k) for k, v in w.items() if g["switches"].get(k) != v}
s = g["switches"]
print(("OK" if not bad and not g["pending"] else "FAIL %s pending %s" % (bad, g["pending"])) + " | bi=%s ds=%s occ=%s lf=%s psk=%s/%s depth0=%s probe=%s epoch=%s applied_step=%s" % (
    s.get("ds41_bi"), s.get("draft_sample"), s.get("draft_occ"), s.get("place_lf"), s.get("ds41_psk"), s.get("ds41_psk_k"), s.get("draft_depth0"),
    s.get("draft_probe"), g.get("epoch"), g.get("applied_step")))' "$g" "$(pinj $2 "${3:-}")" 2>&1 | cut -c1-400)
  case "$r" in OK*) xlog "$1 arm $2 $(armj $2) ${3:-} applied (read back): $r"; return 0 ;; esac
  xlog "$1 arm $2: VIRHE the pin is not applied ($r) -- no reading for this arm"; ARMOK=0; return 1
}
alonex() {   # alonex TAG NAME N K0: N fresh one-token ~1.1k newcomers, one at a time (ncone.py; /stats between them)
  local t0=$(date +%s)
  run "(cd $D && timeout 1200 python3 -I -B $NCONE --mb $MBP --url $URL --n $3 --k0 $4 --tag $2 --out $O/$2-reqs.jsonl --stats $O/$2-stats.jsonl > $O/$2.txt 2>&1)"
  [ "$DRY" = 1 ] || xlog "$1 $2 ($3 fresh ~1.1k prompts one at a time, 1 token, T 0): $(grep -m1 '^NCONE ' $O/$2.txt | cut -c7- | cut -c1-300) | $(thermline $t0 $(date +%s))"
}
c8u() {   # c8u TAG ARM: one mb c8 window as stack4 (mbt.py's phase_c at T 0, dur DUR warm WARMS, the 2 s sampler) + the taps
  local nm=u-$2-c8 t0=$(date +%s)
  run "(cd $D && timeout $(( DUR + 900 )) python3 -I -B $NCMB --mb $MBP --mode phase --temperature 0 --n 8 --tag $nm --dur $DUR --warm $WARMS --reqs $O/$nm-reqs.jsonl --stats-prefix $O/$nm-st --samples $O/$nm-samp.jsonl > $O/$nm.txt 2>&1)"
  [ "$DRY" = 1 ] || xlog "$1 $nm (mb c8 at T 0 as stack4, dur $DUR warm $WARMS, + the 2 s /stats sampler): $(grep -m1 '^MB ' $O/$nm.txt | cut -c4- | cut -c1-500) | $(thermline $t0 $(date +%s))"
}
winp() {   # winp TAG NAME GEN WARM DUR: 8 closed-loop streams (ncmb.py --mode win), the window, the streams cut; then idle
  local nm=$2 i r=""
  run "(cd $D && timeout $(( $5 + 300 )) python3 -I -B $NCMB --mb $MBP --mode win --temperature 0 --n 8 --tag $nm --gen $3 --warm $4 --dur $5 --reqs $O/$nm-reqs.jsonl --stats-prefix $O/$nm-st > $O/$nm.txt 2>&1)"
  [ "$DRY" = 1 ] && { echo "+ GET /stats every 2 s until running 0 (<= 90 s: the cut streams are cancelled)"; return 0; }
  for i in $(seq 1 45); do r=$(curl -s -m 5 $URL/stats | python3 -c 'import json,sys; print(json.load(sys.stdin).get("running"))' 2>/dev/null); [ "$r" = 0 ] && break; sleep 2; done
  xlog "$1 $nm (PROFILED, 8 streams, gen $3, warm $4 s, window $(( $5 - $4 )) s, then cut): $(grep -m1 '^MB ' $O/$nm.txt | cut -c4- | cut -c1-500) | running after: $r"
}
marks() {   # marks REQS -> "T0:T1" (ncmb.py's two marks)
  [ "$DRY" = 1 ] && { echo "T0:T1"; return 0; }
  python3 -c 'import json,sys
m = sorted((d for d in (json.loads(l) for l in open(sys.argv[1]) if l.strip()) if d.get("ev") == "mark"), key=lambda d: d["k"])
print("%.6f:%.6f" % (m[0]["t"], m[1]["t"]))' $1 2>/dev/null
}
capture() {   # capture FILE: the engine log with the daemon's receipt times, gzipped (never written raw)
  run "docker logs -t $CN 2>&1 | gzip -1 > $1"
  [ "$DRY" = 1 ] || xlog "engine log $(basename $1): $(du -m $1 | cut -f1) MB gz, $(zcat $1 | wc -l) lines, $(zcat $1 | grep -cE ' step [0-9]+ (prefill|decode|MIXED) n_seq=') step lines, $(zcat $1 | grep -c 'per-op timing, rank') profile tables; $(dfree) MB free"
}
startenv() {   # startenv TAG [P]: stack4's dstart serve line + RADIANCE_LOG_STEPS=1 (+ the profiler and the miss log under P)
  stackenv; export RADIANCE_LOG_STEPS=1; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on"
  export XS2_ENVFILE=$O/$1.env
  MODEWORDS="$LEVENV RADIANCE_LOG_STEPS=1"
  if [ "${2:-}" = P ]; then
    export RADIANCE_DEBUG_MISS_LOG=/stats/miss-p.log PROFEVERY=$EVERY STATSDIR=$O/stats-p XARGS="$XARGS --profile-ops"
    MODEWORDS="$MODEWORDS RADIANCE_DEBUG_MISS_LOG=/stats/miss-p.log RADIANCE_PROFILE_EVERY=$EVERY"
  else
    export PROFEVERY= STATSDIR=$O/stats-u
  fi
  envfile $XS2_ENVFILE $LEVENV RADIANCE_LOG_STEPS=1 $([ "${2:-}" = P ] && echo RADIANCE_DEBUG_MISS_LOG=/stats/miss-p.log)
  mkd $STATSDIR
}
drafted() {   # drafted TAG: the DSpark drafter and the adaptive depth are up (stack4's check)
  [ "$DRY" = 1 ] && return 0
  local ok=1
  docker logs $CN 2>&1 | grep -q "DSpark drafter" || { xlog "$1: VIRHE no DSpark drafter line"; ok=0; }
  docker logs $CN 2>&1 | grep -q "draft depth: adaptive" || { xlog "$1: VIRHE no adaptive draft depth line"; ok=0; }
  [ $ok = 1 ]
}
# ---------------------------------------------------------------- section (d): msnap / pcreset / cmp copied from dsm-stack4.sh
msnap() { [ "$DRY" = 1 ] && { echo "+ GET /metrics > $O/m-$1.txt"; return 0; }; curl -s -m 20 $URL/metrics > $O/m-$1.txt 2>/dev/null; }
pcreset() { [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o $O/$1-reset.json -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  xlog "$1 prefix cache reset: HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"; }
mdeltax() {   # mdeltax A B -> JSON: stack4's mdelta (acceptance, drafted / accepted, non-decode steps + mean ms, decode steps,
  # prefill / decode tokens) + ms a decode step and tokens a decode step
  [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }
  python3 -c 'import json,re,sys
def rd(p):
    d = {}
    try:
        for l in open(p):
            m = re.match(r"^(radiance:[a-z_]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", l)
            if m: d[m.group(1)] = d.get(m.group(1), 0.0) + float(m.group(2))
    except OSError: pass
    return d
a, b = rd(sys.argv[1]), rd(sys.argv[2])
g = lambda k: b.get("radiance:" + k, 0.0) - a.get("radiance:" + k, 0.0)
nd = g("engine_steps_prefill_total") + g("engine_steps_mixed_total")
nds = g("engine_step_seconds_total") - g("engine_step_seconds_decode_total")
dt, da, ds = g("draft_tokens_total"), g("draft_accepted_total"), g("engine_steps_decode_total")
print(json.dumps(dict(acc=round(da / dt, 4) if dt > 0 else None, draft_tok=int(dt), draft_acc=int(da),
      nondecode_steps=int(nd), mixed_steps=int(g("engine_steps_mixed_total")), nondecode_ms=round(1000 * nds / nd, 1) if nd > 0 else None,
      decode_steps=int(ds), ms_decode_step=round(1000 * g("engine_step_seconds_decode_total") / ds, 2) if ds > 0 else None,
      tok_per_decode_step=round(g("decode_tokens_total") / ds, 3) if ds > 0 else None,
      prefill_tok=int(g("prefill_tokens_total")), decode_tok=int(g("decode_tokens_total")))))' $O/m-$1.txt $O/m-$2.txt 2>/dev/null || echo '{}'
}
ldx() {   # ldx TAG X K: arm S + draft_depth0 (X1 1 / X2 2), the pins read back, a prefix-cache reset, /metrics, c4 + one cold
  # ~LDEPTH deepq prompt (stack4's pskld.py --load, its loadx line), /metrics -> one line of d-runs.jsonl
  local tag=$1 x=$2 nm=d-$3$2 d0 t0 t1 j md t00=$(date +%s)
  d0=$([ $x = X2 ] && echo 2 || echo 1)
  armset $tag S "{\"draft_depth0\": $d0}" || return 1
  pcreset $tag-$nm
  msnap $nm-0; t0=$([ "$DRY" = 1 ] && echo T0 || date +%s.%N)
  run "(cd $D && timeout 2700 python3 -I $PL run --url $URL --tests $TESTS --mb $MBP --which deep --depth $LDEPTH --ntok 512 --load --conc 4 --warm 10 --pf-tok 0 --out $O/$nm.json --text $O/$nm.txt > $O/$nm.log 2>&1)"
  t1=$([ "$DRY" = 1 ] && echo T1 || date +%s.%N); msnap $nm-1
  [ "$DRY" = 1 ] && { echo "+ RESULT xnc $tag $nm (draft_depth0 $d0): probe TTFT, c4 in the prefill window, TPOT, /metrics deltas -> $O/d-runs.jsonl"; return 0; }
  j=$(grep -m1 '^LONGID ' $O/$nm.log | cut -c8-); md=$(mdeltax $nm-0 $nm-1)
  [ -n "$j" ] || { xlog "$tag $nm: VIRHE no LONGID line ($(tail -3 $O/$nm.log | tr '\n' ' ' | cut -c1-400))"; return 1; }
  python3 -c 'import json,sys
print(json.dumps(dict(name=sys.argv[1], x=sys.argv[2], depth0=int(sys.argv[3]), t0=float(sys.argv[4]), t1=float(sys.argv[5]),
      longid=json.loads(sys.argv[6]), md=json.loads(sys.argv[7] or "{}"), text=sys.argv[8])))' "$nm" "$x" "$d0" "$t0" "$t1" "$j" "$md" "$O/$nm.txt" >> $O/d-runs.jsonl
  xlog "$tag $nm (arm S + draft_depth0 $d0, c4 + one cold ~$LDEPTH deepq prompt, probe T 0): $(python3 -c 'import json,sys
d, m = json.loads(sys.argv[1]), json.loads(sys.argv[2] or "{}")
print("probe status %s, prompt %s tok, cached %s, TTFT %s s | prefill window %s s: c4 agg %s tok/s, per-stream p50 %s, TPOT p50 %s / p90 %s ms | drafted %s accepted %s, decode steps %s at %s ms, %s tok a step, non-decode %s at %s ms mean | c4 %s requests, %s errors" % (
    d.get("status"), d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("pf_win_s"), d.get("c4_pf_agg_tps"), d.get("c4_pf_stream_tps_p50"),
    d.get("tpot_pf_p50"), d.get("tpot_pf_p90"), m.get("draft_tok"), m.get("draft_acc"), m.get("decode_steps"), m.get("ms_decode_step"),
    m.get("tok_per_decode_step"), m.get("nondecode_steps"), m.get("nondecode_ms"), d.get("c4_requests"), d.get("c4_errors")))' "$j" "$md" 2>&1 | cut -c1-700) | $(thermline $t00 $(date +%s))"
}
dsect() {   # dsect TAG: section (d), the runs in DORDER (1 = X1, 2 = X2)
  local tag=$1 k=0 c
  [ "$DRY" = 1 ] || : > $O/d-runs.jsonl
  for c in $(echo $DORDER | grep -o .); do
    k=$((k + 1)); stopfile && return 0
    ldx $tag X$c $k
  done
  DDONE=1
}
UDONE=0; PDONE=0; DDONE=0
ustart() {   # START U: unprofiled; per arm ALONE then C8
  local tag=nc-u a k0=500
  diskok $tag || return 1
  startenv $tag
  xlog "$tag: start (START U, UNPROFILED: stack4's NST 5 start + RADIANCE_LOG_STEPS=1) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $FIX | arms $(for a in $UARMS; do echo "$a $(armj $a)"; done | tr '\n' ' ') | env: $(envline)"
  gate1 $tag 70 || { xlog "$tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  modes $tag || { xlog "$tag: VIRHE the modes are not the start's -> stopped"; down; return 1; }
  drafted $tag || { keeplog $tag-nodraft; down; return 1; }
  swstart $tag || { keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  for a in $UARMS; do
    stopfile && break
    armset $tag $a || continue
    alonex $tag u-$a-alone $NALONE $k0; k0=$((k0 + 100))
    stopfile && break
    c8u $tag $a
  done
  [ $DOD = 1 ] && { stopfile || dsect $tag; }
  capture $O/u-engine.log.gz
  [ "$DRY" = 1 ] || xlog "$tag: $(docker logs $CN 2>&1 | grep -cE '^E ') E lines | psk: $(docker logs $CN 2>&1 | grep 'lever PSK:' | tail -1 | sed 's/^radiance: deepseek41: //' | cut -c1-200)"
  down
  UDONE=1
}
TA="T0:T1"; TW1="T0:T1"; TW2="T0:T1"
pstart() {   # START P: profiled + the miss log; arm S; ALONE window, W1, W2
  local tag=nc-p ta0 ta1
  diskok $tag || return 1
  startenv $tag P
  xlog "$tag: start (START P, PROFILED --profile-ops every $EVERY + RADIANCE_DEBUG_MISS_LOG + RADIANCE_LOG_STEPS: not a speed) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $FIX | arm S $(armj S) | env: $(envline)"
  restart $tag || { keeplog $tag-fail; down; xlog "$tag: VIRHE not healthy -- no profile"; return 1; }
  run "mkdir -p $O/guard; (cd $D && timeout 600 python3 ds41_tests.py correct --only-q --effort none --outdir $O/guard > $O/guard/$tag-warm-q.txt 2>&1)"   # = check()'s warm-up line, no 16k / fastcheck
  modes $tag || { xlog "$tag: VIRHE the modes are not the start's -> stopped"; down; return 1; }
  drafted $tag || { keeplog $tag-nodraft; down; return 1; }
  swstart $tag || { keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  armset $tag S || { down; return 1; }
  alonex $tag p-S-warm $PWARMN 700                    # >= EVERY one-step prompts: a table dump falls inside them
  ta0=$([ "$DRY" = 1 ] && echo TA0 || date +%s.%N)
  alonex $tag p-S-alone $PALONEN 800
  ta1=$([ "$DRY" = 1 ] && echo TA1 || date +%s.%N)
  TA="$ta0:$ta1"
  stopfile || winp $tag p-w1 $W1GEN $W1WARM $W1DUR
  stopfile || winp $tag p-w2 $W2GEN $W2WARM $W2DUR
  run "docker stop -t 60 $CN > /dev/null 2>&1"        # the shutdown table and the miss log's closing summary
  capture $O/p-engine.log.gz
  run "gzip -1 -f $STATSDIR/miss-p.log"
  down
  TW1=$(marks $O/p-w1-reqs.jsonl); TW2=$(marks $O/p-w2-reqs.jsonl)
  [ "$DRY" = 1 ] || echo "alone $TA w1 $TW1 w2 $TW2" > $O/p-windows.txt
  PDONE=1
}
pyx() {   # pyx NAME CMD...: one parser run, its last line logged (VIRHE when it fails)
  local nm=$1 rc; shift
  [ "$DRY" = 1 ] && { echo "+ $* > $O/$nm.out"; return 0; }
  timeout 900 "$@" > $O/$nm.out 2>&1; rc=$?
  xlog "$nm: $(tail -1 $O/$nm.out | cut -c1-3500)$([ $rc = 0 ] || echo " -- VIRHE rc $rc")"
}
parse() {
  local a paw ref
  if [ $UDONE = 1 ] || [ "$DRY" = 1 ]; then
    for a in $UARMS; do
      pyx u-$a-alone python3 -I -B $NCS alone --log $O/u-engine.log.gz --reqs $O/u-$a-alone-reqs.jsonl --stats $O/u-$a-alone-stats.jsonl --tag u-$a-alone --label "U ALONE $a (unprofiled)" --json $O/u-$a-alone.json
      pyx u-$a-c8 python3 -I -B $NCS c8 --log $O/u-engine.log.gz --reqs $O/u-$a-c8-reqs.jsonl --tag u-$a-c8 --sa $O/u-$a-c8-st-w0.json --sb $O/u-$a-c8-st-w1.json --samples $O/u-$a-c8-samp.jsonl --mbline $O/u-$a-c8.txt --label "U c8 $a (unprofiled)" --json $O/u-$a-c8.json
    done
  fi
  if [ $PDONE = 1 ] || [ "$DRY" = 1 ]; then
    pyx p-alone python3 -I -B $NCS alone --log $O/p-engine.log.gz --reqs $O/p-S-alone-reqs.jsonl --stats $O/p-S-alone-stats.jsonl --tag p-S-alone --skip-mod $EVERY --label "P ALONE S (PROFILED)" --json $O/p-alone.json
    for a in w1 w2; do
      pyx p-$a python3 -I -B $NCS c8 --log $O/p-engine.log.gz --reqs $O/p-$a-reqs.jsonl --tag p-$a --sa $O/p-$a-st-w0.json --sb $O/p-$a-st-w1.json --mbline $O/p-$a.txt --skip-mod $EVERY --label "P $a S (PROFILED)" --json $O/p-$a.json
    done
    pyx p-miss python3 -I -B $NCS miss --file $O/stats-p/miss-p.log.gz --log $O/p-engine.log.gz --win alone:$TA --win w1:$TW1 --win w2:$TW2 --json $O/p-miss.json
    paw=$([ "$DRY" = 1 ] && echo MS || python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["step_ms"]["mean"])' $O/p-alone.json 2>/dev/null)
    pyx p-prof python3 -I -B $OPNC $O/p-engine.log.gz --win alone:$TA --win w1:$TW1 --win w2:$TW2 $([ -n "$paw" ] && echo "--wall alone:$paw") --diff nc_diff:w2:w1 --intervals nc_iv:w2 --every $EVERY --label nc-p --json $O/p-prof.json
  fi
  if [ $DOD = 1 ] && { [ $DDONE = 1 ] || [ "$DRY" = 1 ]; }; then
    pyx d-ld python3 -I -B $NCS ld --log $O/u-engine.log.gz --runs $O/d-runs.jsonl --pskld $PL --grid 2048 --json $O/d-ld.json
  fi
  ref=$([ "$DRY" = 1 ] && echo REF || python3 -c 'import json,statistics,sys
rs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(", ".join("%s %.1f" % (a, statistics.mean(r["res"]["c8"]["agg_dec_tps"] for r in rs if r["arm"] == a and r.get("ok"))) for a in "AS"))' $RS4/s4-n5-t0-blocks.jsonl 2>/dev/null)
  run "python3 -I -B $NCS report --in ua_S=$O/u-S-alone.json --in ua_A=$O/u-A-alone.json --in uc_S=$O/u-S-c8.json --in uc_A=$O/u-A-c8.json --in pa=$O/p-alone.json --in pw1=$O/p-w1.json --in pw2=$O/p-w2.json --in miss=$O/p-miss.json --in prof=$O/p-prof.json --in ld=$O/d-ld.json --ref-c8 '${ref:-n/a} (stack4 T 0 mean, cross-start)' > $O/report.txt 2>&1"
  [ "$DRY" = 1 ] && { echo "+ every line of $O/report.txt -> the log (RESULT xnc (a) / (b) / (c) / summary)"; return 0; }
  while IFS= read -r l; do case "$l" in "RESULT xnc "*) log "${l:0:3800}" ;; esac; done < $O/report.txt
  local nv; nv=$(( $(cat $O/report.txt 2>/dev/null | grep -c VIRHE) + NV ))
  [ $ARMOK = 1 ] || nv=$((nv + 1))
  [ $UDONE = 1 ] || nv=$((nv + 1))
  [ $DOP = 0 ] || [ $PDONE = 1 ] || nv=$((nv + 1))
  [ $DOD = 0 ] || [ $DDONE = 1 ] || nv=$((nv + 1))
  log "RESULT xnc guards: $([ "$nv" = 0 ] && echo "OK (step-log lines in every window, profile tables on 4 ranks, /stats counters moved, pins read back, newcomers fresh, one step a prompt, (d) cold + X2 drafted ~0; START U$([ $DOD = 1 ] && echo " + (d)")$([ $DOP = 1 ] && echo " and P") done)" || echo "VIRHE in $nv place(s) (START U done $UDONE, START P done $PDONE, section (d) done $DDONE) -- see the lines above; a reading with VIRHE is not a reading")"
}
trap 'down' EXIT
trap 'exit 143' TERM INT
# ---------------------------------------------------------------- main
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) | START U (unprofiled, step log) arms [$UARMS]: ALONE $NALONE one-token ~1.1k prompts + mb c8 T 0 dur $DUR warm $WARMS (as stack4) | START P$([ $DOP = 1 ] || echo " SKIPPED") (profiled every $EVERY + miss log, arm S): ALONE $PWARMN + $PALONEN, W1 8 streams gen $W1GEN warm $W1WARM dur $W1DUR, W2 mb c8 gen $W2GEN warm $W2WARM dur $W2DUR | section (d)$([ $DOD = 1 ] || echo " SKIPPED") D7 under a long prefill: arm S, draft_depth0 1 / 2 in order $DORDER, c4 + one cold ~$LDEPTH deepq prompt | ~28 min | (b) by the code: MIXED, the decode rows ride (scheduler.cpp:2601-2721 / 2807-2852 / 2993)"
IMGLITS="B|dspark_conf_rows, gg_pf, f32_pf, dspark_walk, ds41_bi, draft_sample, draft_lend_hyst, arena_d0, draft_lend_w, draft_occ, draft_occ_k, place_lf, sched_bi_chunk, sched_interleave, sched_mixed_tok,
B|sched_short_bypass, sched_bypass_total, ds41_psk
B|ds41_psk_k
B|reset_prefix_cache
B|radiance:engine_steps_mixed_total
B|radiance:engine_step_seconds_decode_total
B|radiance:client_disconnect_total
A|lever PSK declared (RADIANCE_DS41_PSK, fn4 ds41_psk, now
A|lever BI declared (RADIANCE_DS41_BI, fn4 ds41_bi, now
B|place_lf
B|RADIANCE_LOG_STEPS
B|n_seq=%lld n_tok=%lld
B|  [%d] slot=%d q=%d ctx=%d
B|RADIANCE_PROFILE_EVERY
B|per-op timing, rank %d.
B|RADIANCE_DEBUG_MISS_LOG
B|radiance-miss-log/1
B|step_ns_decode
B|steps_mixed
B|stage_layers
B|prefill staging: a %s ring at the top of the arena
A|RADIANCE_DS41_DSPARK_WALK
A|DSpark lever SD: the sampled full-vocabulary walk declared
B|RADIANCE_WEIGHT_SIDECAR
B|draft_occ takes 0 (off) .. 64
B|lever Z0-A set up
B|RADIANCE_DRAFT_CTXSKIP_SLOTS"
docker image inspect $IMAGE > /dev/null 2>&1 || { xlog "VIRHE image $IMAGE not on this host"; exit 1; }
IMGMISS=$(echo "$IMGLITS" | timeout 120 docker run --rm -i --network none --entrypoint sh $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels/libr4d.so; S=/opt/radiance/share/radiance/kernels/spechub.so; n=0
while IFS= read -r l; do [ -n "$l" ] || continue; n=$((n + 1)); w=${l%%|*}; s=${l#*|}
  case $w in B) f=$B ;; A) f=$A ;; K) f=$K ;; S) f=$S ;; *) f="$B $A $K" ;; esac
  o=0; for x in $f; do grep -qzF -- "$s" $x && { o=1; break; }; done
  [ $o = 1 ] || echo "MISS $w|$s"; done
test -s /opt/radiance/share/radiance/int7-PROVENANCE.txt || echo "MISS int7-PROVENANCE.txt"; echo "CHECKED $n"' 2>&1)
case "$IMGMISS" in "CHECKED "*) ;; *) xlog "VIRHE $IMAGE is not the int7 image this item reads -- missing: $(echo "$IMGMISS" | grep -v '^CHECKED' | tr '\n' ';' | cut -c1-900)"; exit 1 ;; esac
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)) is the int7 image: all $(echo "$IMGMISS" | sed 's/CHECKED //') literals present (the fn4 registry with the DST / PSK switches, the step-log format, the profiler's table, the miss log, /stats step / stager counters, the stager's ring note, client_disconnect) + int7-PROVENANCE -- OK"
HMISS=""
for f in "$LIB29" "$LIBPR" $Y $MBP $TESTS $D/fastcheck.py $NCONE $NCMB $NCS $OPNC $MBT $OPP $PF1 $S4B $PL; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $MBP $TESTS $D/fastcheck.py $NCONE $NCMB $NCS $OPNC $MBT $PL 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
ST1=$(timeout 60 python3 -I -B $NCONE --mb $MBP --selftest 2>&1 | tail -1); case "$ST1" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$NCONE(${ST1:0:300})" ;; esac
ST2=$(timeout 60 python3 -I -B $NCMB --mb $MBP --selftest 2>&1 | tail -1); case "$ST2" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$NCMB(${ST2:0:300})" ;; esac
ST3=$(timeout 120 python3 -I -B $NCS selftest 2>&1 | tail -1); case "$ST3" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$NCS(${ST3:0:300})" ;; esac
ST4=$(timeout 120 python3 -I -B $OPNC selftest 2>&1 | tail -1); case "$ST4" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$OPNC(${ST4:0:300})" ;; esac
ST5=$(timeout 60 python3 -I -B $MBT --mb $MBP --selftest 2>&1 | tail -1); case "$ST5" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$MBT(${ST5:0:300})" ;; esac
ST6=$(timeout 60 python3 -I -B $PL selftest --tests $TESTS --mb $MBP 2>&1 | tail -1); case "$ST6" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$PL(${ST6:0:300})" ;; esac
FNS="log run mkd restart check down stopfile best_vars desc serve_args xlog armj pinj stackenv keeplog thermline identw modes fn4get fn4set nudge arm gate1 dfree diskok envfile envline swstart armset alonex c8u winp marks capture startenv drafted ustart pstart pyx parse msnap pcreset mdeltax ldx dsect"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee sha256sum gzip zcat df du; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
for a in S A; do python3 -c 'import json,sys; json.loads(sys.argv[1])' "$(pinj $a '{"epoch_bump": 1}')" 2>/dev/null || HMISS="$HMISS json:$a"; done
for d0 in 1 2; do python3 -c 'import json,sys; d = json.loads(sys.argv[1]); sys.exit(0 if d["draft_depth0"] == int(sys.argv[2]) and d["ds41_bi"] == 2 else 1)' "$(pinj S "{\"draft_depth0\": $d0}" '{"epoch_bump": 1}')" $d0 2>/dev/null || HMISS="$HMISS json:S+draft_depth0=$d0"; done
[ "$(sha256sum $MODELDIR/$SIDECAR 2>/dev/null | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256 2>/dev/null)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR"
[ "$(sha256sum $MODELDIR/$SIDE 2>/dev/null | cut -c1-64)" = "$SIDESHA" ] || HMISS="$HMISS sha256:$MODELDIR/$SIDE(want ${SIDESHA:0:16})"
for w in XS2_ENVFILE 'RADIANCE_PROFILE_EVERY: ${PROFEVERY' 'RADIANCE_DEBUG_MISS_LOG:' '${STATSDIR:?set STATSDIR}:/stats'; do grep -qF "$w" $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no '$w')"; done
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config refuses it)"
HMISS=$(echo $HMISS)
[ -z "$HMISS" ] || { xlog "VIRHE helpers / inputs do not resolve: $HMISS"; exit 1; }
HMD5="ncone.py $(md5sum < $NCONE | cut -c1-8), ncmb.py $(md5sum < $NCMB | cut -c1-8), ncsteps.py $(md5sum < $NCS | cut -c1-8), ds_opprof_nc.py $(md5sum < $OPNC | cut -c1-8) (copy of ds_opprof.py $(md5sum < $OPP | cut -c1-8)), stack4/mbt.py $(md5sum < $MBT | cut -c1-8), stack4/pskld.py $(md5sum < $PL | cut -c1-8), dsm-stack4.sh $(md5sum < $S4B | cut -c1-8), mb.py $(md5sum < $MBP | cut -c1-8), pf1.py $(md5sum < $PF1 | cut -c1-8), lib-29.sh $(md5sum < $LIB29 | cut -c1-8), yaml $(md5sum < $Y | cut -c1-8)"
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing; ncone.py $ST1 | ncmb.py $ST2 | ncsteps.py $ST3 | ds_opprof_nc.py $ST4 | stack4/mbt.py $ST5 | stack4/pskld.py $ST6 | all $(echo $FNS | wc -w) functions resolve, tools on PATH, sidecars $SIDECAR + $SIDE sha256 OK, yaml accepted (env_file, RADIANCE_PROFILE_EVERY, RADIANCE_DEBUG_MISS_LOG, /stats mount) -- OK | md5: $HMD5"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { xlog "VIRHE $M225 missing or not verified"; exit 1; }
  for f in $WARM_REF_225 $RS4/s4-n5-t0-blocks.jsonl $RS4/guard-s4-n5.txt; do [ -s $f ] || xlog "note -- reference $f missing (not compared)"; done
  xlog "$ITEM: helpers md5: $HMD5"
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDE $WARM_REF_225 $RS4/s4-n5-t0-blocks.jsonl $RS4/guard-s4-n5.txt; do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  for a in S A; do echo "+ arm $a: $(armj $a) | POST body $(pinj $a '{"epoch_bump": 1}')"; done
  for d0 in 1 2; do echo "+ (d) X$d0: arm S + draft_depth0 $d0 | POST body $(pinj S "{\"draft_depth0\": $d0}" '{"epoch_bump": 1}')"; done
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(information here; the real run refuses to start beside one)"
fi
mkd $O $O/guard
[ "$DRY" = 1 ] || : > $O/identity.txt
stopfile || ustart
[ $DOP = 1 ] && { stopfile || pstart; }
parse
unset $SWITCHES
down
xlog "$ITEM: done -- $O ($(dfree) MB free)"
