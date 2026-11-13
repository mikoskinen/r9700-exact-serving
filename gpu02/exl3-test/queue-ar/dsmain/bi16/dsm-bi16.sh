#!/bin/bash
# dsm-bi16.sh (12.11.2026, queue-ar/dsmain/bi16, ds-main-port bi16 builder) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzz-bi16.sh. Log tag "RESULT xbi16 ", out runs/xdsm/bi16 (XB16_OUT). Made from COPIES of
# kv4/dsm-kv4.sh (f89a6cc2: the start words, LITS / FIX / DET, stackenv, armj / pinj / armset, modes, gate1, swstart, identw,
# the image / helper / sidecar / yaml checks; tag xkv4 -> xbi16) and dsm-bilid3.sh (af9f3529: the tracer's directory made
# before the start, its start-log checks under the tracer -- rule-w in the container env instead of the recorded-pass lines --
# and btcheck), + bi16probe.py (from stack4/loadid.py ee7d9663) + bi16trace.py (= bitrace2.py cbbd1852 + the probe match, the
# per-step decode rows, the op class, the independent flips, the extract, the step counter); the originals are not changed.
# QUESTION (coordinator 12.11.): kv4 n16 (this image, MAXSEQS 16, nvfp4, S5 pins incl. ds41_bi 2 + sched_bi_chunk 3) answered
#   9 / 20 (gs16) and 8 / 20 (lid16) greedy prompts differently under load than alone -> WHICH OP's rows change first when a
#   step carries > 48 decode rows? (kv4's bi48 parse read 'most 48' because the step line lists at most 8 sequences; the step
#   log's n_tok says lid16 had 519 of 667 decode steps with > 48 rows, up to 96, gs16 13 of 155.)
# CODE READING (int8 7e03384, deepseek41.cpp step_bi :4288-5003; bi16trace.py's class table): under ds41_bi 2 the decode rows
#   are ONE segment; issued over the whole segment: the query / key / index GEMMs q_a, kv_g, i_w, q_b or K1 fold a, i_q
#   (:4672-4706, lin_issue(..., n = s.n)), the indexer runs (:4788-4812, ds41_idx_runs(s.n, irows)), the attention
#   (attn_sparse_sink :4825-4848); in <= 48-row pieces: o_a, o_b, the Engram GEMM, the head (rows(kBiBand)); fused norms in
#   <= 32 (rows(kBiNorm)); the MoE in <= 16-row passes. gemm_nt_ggml takes the decode kernel only at M <= 48
#   (r4d_ds41_ggml.hip:950 / :3671), so a > 48-row segment's q_a would be the first op to change -- UNCONFIRMED: this item.
# ONE START t16 = kv4's n16 (int8, nvfp4, MAXSEQS 16, HOSTPOOL 19456, MML 524288, stack4's NST 5 start words + S5's
#   RADIANCE_DS41_ENGDD=0 + RADIANCE_LOG_STEPS=1 + NVG 48) + lever BI's tracer: RADIANCE_DS41_BITRACE=/stats/bt-t16 (the
#   start's STATSDIR, made 777 BEFORE the start), RADIANCE_DS41_BITRACE_PMAX=XB16_PMAX (224: the probes' 7-9 prompt tokens +
#   192 answer tokens; the streams' ~1000-token prompts are traced only below it), _LMAX XB16_LMAX (empty = every layer).
#   The tracer records no pass (eager issue = the recorded arithmetic, trace2 / trace3: alone IDENTICAL to d0) and syncs once
#   a step. Pin: kv4's fixed pin + S5 + sched_bi_chunk 3, every phase under arm D = S + DET (fixed depth 5: no depth 0 /
#   occupancy / hybrid / confidence cut -> 16 sequences x 6 rows = up to 96 decode rows), read back before each phase.
# PHASES (probes XB16_IDX = 9,8: 'The capital of Australia is' / 'SELECT name, COUNT(*) FROM orders', the two that differed
#   earliest in BOTH kv4 runs: p9 tok 2 / 2, p8 tok 5 / 3): tok (POST /tokenize: their ids, so the analyzer finds them);
#   a1 = each probe ALONE (greedyset.py's request, T 0, 192 tok); ld = each probe under XB16_CONC (15) mb closed-loop
#   streams (ctx 1000, gen 512, as lid16's; no newcomer prefill), XB16_WARM (10) s after they start, then the streams drain;
#   a2 = ALONE again (the control). Prefix cache reset before each.
# READINGS: per probe the FIRST DIFFERING OP loaded vs alone (bi16trace.py: point = layer.op, rank, position, the step's
#   geometry T / decode rows DT / sequences / prefill sequences / the row's q_len, alone and loaded), its CLASS (dense GEMM
#   over the whole segment, dense <= 48 pieces, attention, indexer, fused norm, MoE, other), the INDEPENDENT flips of its layer
#   (differing points whose traced inputs are equal: e.g. qr AND kvn = two GEMMs, not one), the loaded run's DT per step;
#   the step log's > 48-row decode steps in each window; answers ld vs a1 (B48), a2 vs a1, a1 vs kv4's untraced n16-a1 and
#   bz d0 (information).
# GUARDS (VIRHE): the tracer dumped (every rank's .bin after the warm-up + 1 request; the analyzer finds both probes in every
#   run); the loaded runs really had > 48-row steps (the trace's DT of the probe's own steps AND the step log's decode n_tok);
#   the alone reference reproduces itself (a2 vs a1: answers IDENTICAL and every traced row and point IDENTICAL).
# DUMP: ~20 kB a traced row (4 ranks x ~660 points x 8 B); expected ~150-300 MB raw (warm-up ~20, the probes' 6 runs ~40,
#   the streams' prompt rows below PMAX ~4.5 MB a stream request); cap XB16_BTCAP_MB (1500) checked after every phase (a
#   phase past it is skipped, VIRHE); after the analysis the raw .bin files are DELETED (kept: the .names, the analysis json,
#   a gzipped extract of every run's rows around the first divergence, < 5 MB); a failed analysis keeps them only when
#   <= XB16_KEEP_MB (600) and / keeps >= XB16_DISKMIN_MB (11000) + them free, else deletes them too (logged).
# ~12 min (estimate; budget XB16_BUDGET_MIN 25: ld is skipped and logged when it would end past it): start + check ~5,
#   modes / switches / tracer check 0.5, a1 0.5, ld ~3 (warm 10 s, 2 probes, the streams' drain), a2 0.5, compares 0.2,
#   capture + parse + cleanup ~1.5. Disk guard XB16_DISKMIN_MB 11000 before the start and after every phase.
# NO cc_sim, no evict, no power / host / fan settings; production compose files read only (--project-directory).
# Knobs: XB16_IMAGE, XB16_ITEM, XB16_OUT, XB16_PINS (auto | s4 | s5; auto = s5 for ds41-int8 / int9), XB16_IDX (9,8),
#   XB16_CONC (15), XB16_WARM (10), XB16_GEN (512), XB16_PMAX (224), XB16_LMAX (empty), XB16_BTCAP_MB (1500), XB16_KEEP_MB (600),
#   XB16_KEEPRAW (0; 1 keeps the raw files when the cap and the disk allow), XB16_BIC (3), XB16_NVG (48), XB16_HP (19456),
#   XB16_DISKMIN_MB (11000), XB16_BUDGET_MIN (25), XB16_SIDECAR, XB16_SIDESHA.
# DRYRUN=1 bash dsm-bi16.sh : prints every command; checks the image literals (kv4's + the tracer's), every helper (files,
#   parse, selftests), the sidecars, the yaml, the disk; no server, no GPU, no request to :8090, writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/../ds41-stack2-tp4.yaml                   # stack2's yaml, read only (env_file ${XS2_ENVFILE} carries the start words)
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh
O=${XB16_OUT:-$R/xdsm/bi16}
RK4=$R/xdsm/kv4; RD=$R/xdsm/draft
export IMAGE=${XB16_IMAGE:?set XB16_IMAGE}
ITEM=${XB16_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzz-bi16}
PINS=${XB16_PINS:-auto}; [ "$PINS" = auto ] && { case $IMAGE in *ds41-int8*|*ds41-int9*) PINS=s5 ;; *) PINS=s4 ;; esac; }
IDX=${XB16_IDX:-9,8}; CONC=${XB16_CONC:-15}; WARMS=${XB16_WARM:-10}; GEN=${XB16_GEN:-512}; CTX=1000
PMAX=${XB16_PMAX:-224}; LMAX=${XB16_LMAX:-}; BTCAP=${XB16_BTCAP_MB:-1500}; KEEPMB=${XB16_KEEP_MB:-600}; KEEPRAW=${XB16_KEEPRAW:-0}
BIC=${XB16_BIC:-3}; NVG=${XB16_NVG:-48}; HP=${XB16_HP:-19456}; DISKMIN=${XB16_DISKMIN_MB:-11000}; BUDGET=${XB16_BUDGET_MIN:-25}
SIDE=${XB16_SIDECAR:-dspark-dlw-q8_0.rad}; SIDESHA=${XB16_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
case $PINS in s4) DSPIN='{"ds41_bi": 2, "draft_sample": 85, "draft_occ": 6, "place_lf": 0, "ds41_psk": 2, "ds41_psk_k": 4}'; DPEX="" ;;
  s5) DSPIN='{"ds41_bi": 2, "draft_sample": 85, "draft_occ": 6, "place_lf": 0, "ds41_psk": 2, "ds41_psk_k": 8, "ds41_psk_l20": 1, "dspark_pfctx": 1, "ds41_engdd": 1, "sched_noa": 2, "sched_bi_chunk": 3}'
      DPEX="RADIANCE_DS41_ENGDD=0" ;;
  *) echo "XB16_PINS must be auto, s4 or s5 (got '$PINS')"; exit 1 ;; esac
SPIN=$DSPIN; PEXTRA=$DPEX
case "$IDX" in ''|*[!0-9,]*) echo "XB16_IDX: comma-separated greedy prompt indices 0..19 (got '$IDX')"; exit 1 ;; esac
NPROBE=$(echo "$IDX" | tr ',' '\n' | grep -c .)
[ $NPROBE -ge 1 ] && [ $NPROBE -le 2 ] || { echo "XB16_IDX: one or two probes (got $NPROBE)"; exit 1; }
for v in $CONC $WARMS $GEN $PMAX $BTCAP $KEEPMB $BIC $NVG $HP $DISKMIN $BUDGET; do case $v in ''|*[!0-9]*) echo "a numeric knob is not a number ('$v')"; exit 1 ;; esac; done
case "$LMAX" in ''|[0-9]*) ;; *) echo "XB16_LMAX takes a layer count or nothing (got '$LMAX')"; exit 1 ;; esac
case $KEEPRAW in 0|1) ;; *) echo "XB16_KEEPRAW takes 0 or 1"; exit 1 ;; esac
case $BIC in 0|1|2|3|4) ;; *) echo "XB16_BIC takes 0 .. 4 (got $BIC)"; exit 1 ;; esac
[ $NVG -le 48 ] || { echo "XB16_NVG is 0 (off) .. 48"; exit 1; }
[ $CONC -ge 9 ] || { echo "XB16_CONC below 9 cannot pass 48 decode rows at depth 5 (9 x 6 = 54)"; exit 1; }
[ $PMAX -ge 64 ] || { echo "XB16_PMAX below 64 would not hold the probes' first answer tokens"; exit 1; }
PRB=$HD/bi16probe.py; BT=$HD/bi16trace.py
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
MBP=$D/mb.py; TESTS=$D/ds41_tests.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
MT=192; T0=$(date +%s); NV=0; TAG=t16
xlog() { case "$*" in *VIRHE*) NV=$((NV + 1)) ;; esac; log "RESULT xbi16 $*"; }
# ---------------------------------------------------------------- copied from dsm-stack4.sh (21ce8c8e; tag xs4 -> xkv4), via kv4/dsm-kv4.sh f89a6cc2 (tag xkv4 -> xbi16)
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
# + this item's: the index keys packed in every start (stack4's RADIANCE_DS41_IDXKEY=mxfp4), the pool notes the capacity reads
LITS="$LITS|keys,mxfp4,(80,bytes,an,entry;,RADIANCE_DS41_IDXKEY)|kv:,the,pool,backs|kv:,the,cache,can,grow,to|VRAM,budget"
[ $PINS = s5 ] && LITS="$LITS|lever,E1,declared,(RADIANCE_DS41_ENGDD"
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": 8, "drafter_mx": 1, "gg_pf": 1, "f32_pf": 25, "dspark_walk": 1, "ds41_argather": 48, "idx_split": 1, "ds41_qnfold": 1, "ds41_swfold": 1, "draft_ctxskip": 1, "dspark_conf": 1, "dspark_conf_rows": 1, "draft_lend": 1, "draft_probe_backoff": 3, "draft_hybrid": 1}'
FIX=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update({"draft_lend_hyst": 0, "arena_d0": 0, "draft_lend_w": 0, "draft_occ_k": 32}); print(json.dumps(d))' "$FIX5") || { echo "pin JSON"; exit 1; }
# the > 48-row check's deterministic drafting overlay (dsm-pskl20.sh's DET): a fixed depth (5), no depth 0 / occupancy / lookup
# hybrid / confidence cut -- exact as every depth choice is (it moves only what a step verifies)
DET='{"draft_adaptive": 0, "draft_depth0": 0, "draft_occ": 0, "draft_hybrid": 0, "dspark_conf": 0}'
armj() {   # armj S|D: the S switches + sched_bi_chunk (D: + DET)
  python3 -c 'import json,sys
d = json.loads(sys.argv[1]); d["sched_bi_chunk"] = int(sys.argv[2])
if sys.argv[3] == "D": d.update(json.loads(sys.argv[4]))
print(json.dumps(d))' "$SPIN" "$BIC" "$1" "$DET"
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
  RADIANCE_DS41_PSK RADIANCE_PLACE_LF RADIANCE_PLACE_LF_DECAY RADIANCE_LOG_STEPS RADIANCE_DS41_ENGDD RADIANCE_DS41_PSK_L20
  RADIANCE_DS41_DSPARK_PFCTX RADIANCE_PLACE_HB_RANKS RADIANCE_PLACE_HB_DECAY"
stackenv() {   # 105's t3 arm + gg_gemv 2 + rule-w + the K1 folds + turnover 0 + LEVENV (+ PEXTRA under s5), every other switch unset
  local kv
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  export RADIANCE_PLACE_TURNOVER=0
  for kv in $LEVENV $PEXTRA; do export "$kv"; done
}
idrec() {   # idrec NAME RESULT [WHAT]: identity.txt + one RESULT line (a difference reported, not hidden). NAME: S: strict,
  # B48: the > 48-row measurement (a difference is the reading), I: information (cross-start / untraced, labelled)
  [ "$DRY" = 1 ] && { echo "+ identity $1 ${3:+($3)}"; return 0; }
  echo "$1 $2" >> $O/identity.txt
  case $2 in IDENTICAL*) xlog "identity $1${3:+ ($3)}: $2" ;;
    *) case $1 in S:*) xlog "identity $1${3:+ ($3)}: VIRHE $2" ;; *) xlog "identity $1${3:+ ($3)}: $2 -- reported, not hidden" ;; esac ;; esac
}
cmpx() {   # cmpx NAME A B: bi16probe.py cmp over the probes (byte for byte + the first differing char / token) -> identity.txt
  local r
  [ "$DRY" = 1 ] && { echo "+ python3 -I -B $PRB cmp $2 $3 --idx $IDX --url $URL --json $O/cmp-${1#*:}.json -> identity $1"; return 0; }
  if [ ! -s "$2" ] || [ ! -s "$3" ]; then xlog "$1: not compared (missing $2 or $3)"; echo "$1 not compared" >> $O/identity.txt; return 0; fi
  r=$(timeout 300 python3 -I -B $PRB cmp "$2" "$3" --idx $IDX --url $URL --json $O/cmp-${1#*:}.json --label "$1" 2>&1 | grep -m1 '^CMPLINE ' | cut -c9-)
  idrec "$1" "${r:-VIRHE compare failed}" "$(basename $2) vs $(basename $3)"
}
# ---------------------------------------------------------------- copied from kv4/dsm-kv4.sh f89a6cc2 (tag xkv4 -> xbi16)
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xbi16 $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { xlog "$1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  echo "S:$1-warmup $r" >> $O/identity.txt
  case $r in IDENTICAL*) xlog "$1 warm-up answers vs stock ($(basename $2)): $r" ;; *) xlog "$1 warm-up answers vs stock: VIRHE $r" ;; esac
}
modes() {   # modes TAG LAT: MODEWORDS in the container (RADIANCE_DS41_LATENT only under nvfp4), mover 0 kept free x4, K1, ARGATHER
  # 48, the latent / key literals, every literal of LITS; UNDER THE TRACER (dsm-bilid3.sh's rule): no pass is recorded, so
  # rule-w is read from the container env with no plain rule line, and the tracer ON line (its PMAX / LMAX) + 'recorded
  # passes: off' + no 'tracer cannot open' replace the recorded-pass lines -> 0, else 1
  [ "$DRY" = 1 ] && { echo "+ modes $1: [$MODEWORDS] in the container's env; 'latent $2 ($([ $2 = nvfp4 ] && echo 288 || echo 1024) bytes an entry; RADIANCE_DS41_LATENT)' in the start log; mover '0 more kept free' on 4 ranks; RADIANCE_AQL_ACQ=rule-w in the env + no plain rule line (no pass is recorded under the tracer); the tracer ON line with 'rows at positions 0..$((PMAX - 1)), layers below $(btlmax)', 'recorded passes: off', no 'tracer cannot open'; K1 folds =1; ARGATHER 48-row note; literals [$(echo "$LITS" | tr ',' ' ')]"; return 0; }
  local CE kv bad="" ok=1 NR KL AL LL nz lit found="" miss="" want TLN RO
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in RADIANCE_PLACE_TURNOVER=0 RADIANCE_AQL_ACQ=rule-w $MODEWORDS; do echo "$CE" | grep -qxF "$kv" || bad="$bad ${kv%%=*}"; done
  [ $2 = bf16 ] && echo "$CE" | grep -q '^RADIANCE_DS41_LATENT=' && bad="$bad RADIANCE_DS41_LATENT(set-under-bf16)"
  [ -z "$bad" ] || { xlog "$1: VIRHE start words not as asked:$bad"; ok=0; }
  want="latent $2 ($([ $2 = nvfp4 ] && echo 288 || echo 1024) bytes an entry; RADIANCE_DS41_LATENT)"
  LL=$(docker logs $CN 2>&1 | grep -m1 -o 'latent [a-z0-9]* ([0-9]* bytes an entry; RADIANCE_DS41_LATENT)')
  [ "$LL" = "$want" ] || { xlog "$1: VIRHE the latent mode read back is '$LL', asked '$want'"; ok=0; }
  nz=$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots \([0-9]+ shadow, 0 more kept free')
  [ "$nz" = 4 ] || { xlog "$1: VIRHE turnover 0 asked but $nz of 4 ranks keep 0 slots free"; ok=0; }
  NR=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_AQL_ACQ=rule)')
  TLN=$(docker logs $CN 2>&1 | grep -m1 "lever BI's tracer ON" | sed 's/^radiance: //' | cut -c1-400)
  RO=$(docker logs $CN 2>&1 | grep -c 'recorded passes: off, the program traces rows')
  if docker logs $CN 2>&1 | grep -q "lever BI's tracer cannot open"; then
    xlog "$1: VIRHE the tracer could not open its files ($(docker logs $CN 2>&1 | grep -m1 "tracer cannot open" | cut -c1-200)) -- it is off"; ok=0
  fi
  case "$TLN" in *"RADIANCE_DS41_BITRACE=/stats/bt-$1)"*"rows at positions 0..$((PMAX - 1)), layers below $(btlmax);"*) ;;
    *) xlog "$1: VIRHE no tracer ON line for /stats/bt-$1 with rows 0..$((PMAX - 1)), layers below $(btlmax): '$TLN'"; ok=0 ;; esac
  [ "$NR" = 0 ] && [ "$RO" -gt 0 ] || { xlog "$1: VIRHE under the tracer: a plain rule line ($NR) or no 'recorded passes: off' line ($RO)"; ok=0; }
  KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-500)
  case "$KL" in *"ds41_kvfold=1"*"ds41_ofold=1"*"ds41_onefill=1"*) ;; *) xlog "$1: VIRHE the K1 switches do not start at 1: '${KL}'"; ok=0 ;; esac
  AL=$(docker logs $CN 2>&1 | grep -m1 'the MoE gather and its all-reduce as one kernel' | sed 's/^radiance: //' | cut -c1-300)
  case "$AL" in *"up to 48 tokens a step (RADIANCE_DS41_ARGATHER, RADIANCE_DS41_ARGATHER_MAX"*) ;; *) xlog "$1: VIRHE the ARGATHER note is not the 48-row one: '${AL}'"; ok=0 ;; esac
  IFS='|' read -ra lit <<< "$LITS"
  for l in "${lit[@]}"; do [ -n "$l" ] || continue
    if docker logs $CN 2>&1 | grep -qF -- "${l//,/ }"; then found="$found ${l%%,*}"; else miss="$miss | ${l//,/ }"; fi; done
  xlog "$1 modes: latent read back '$LL', ${#lit[@]} literal(s), found [$(echo $found)]$([ -n "$miss" ] && echo " -- VIRHE missing:$miss") | tracer: '$TLN' | recorded passes off: $RO line(s), plain rule lines $NR | $(docker logs $CN 2>&1 | grep -E "kv: the pool backs|lever PSK declared|lever BI declared|lever E1|chunk geometry" | sed 's/^radiance: //; s/^[A-Z] *//; s/^[A-Z]\[[0-9]\] //' | cut -c1-330 | sort -u | tr '\n' '|' | cut -c1-2000)"
  [ -z "$miss" ] || ok=0
  echo "$1 modes $([ $ok = 1 ] && echo OK || echo VIRHE) | latent '$LL' | tracer '$TLN'" >> $O/guard.txt
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN | gzip > $O/engine-$1.log.gz"; return 0; }; docker logs $CN 2>&1 | gzip -1 > $O/engine-$1.log.gz || true; }
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
fn4code() { [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
arm() { fn4set "$(pinj $1 "${2:-}" '{"epoch_bump": 1}')"; nudge; }
ARMOK=1
armset() {   # armset TAG ARM: the pin + the arm + epoch_bump, a nudge, every switch READ BACK (else VIRHE, ARMOK 0)
  local g r
  arm $2
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4: every switch of $(pinj $2) applied, nothing pending -> RESULT xbi16 $1 arm $2 applied (read back)"; return 0; }
  g=$(fn4get)
  r=$(python3 -c 'import json,sys
g, w = json.loads(sys.argv[1]), json.loads(sys.argv[2])
bad = {k: g["switches"].get(k) for k, v in w.items() if g["switches"].get(k) != v}
s = g["switches"]
print(("OK" if not bad and not g["pending"] else "FAIL %s pending %s" % (bad, g["pending"])) + " | bi=%s psk=%s/%s bic=%s occ=%s ds=%s adapt=%s d0=%s hyb=%s conf=%s epoch=%s" % (
    s.get("ds41_bi"), s.get("ds41_psk"), s.get("ds41_psk_k"), s.get("sched_bi_chunk"), s.get("draft_occ"), s.get("draft_sample"), s.get("draft_adaptive"), s.get("draft_depth0"), s.get("draft_hybrid"), s.get("dspark_conf"), g.get("epoch")))' "$g" "$(pinj $2)" 2>&1 | cut -c1-400)
  case "$r" in OK*) xlog "$1 arm $2 applied (read back): $r"; return 0 ;; esac
  xlog "$1 arm $2: VIRHE the pin is not applied ($r) -- no reading under it"; echo "$1 pins VIRHE arm $2 $r" >> $O/guard.txt; ARMOK=0; return 1
}
gate1() {   # restart + ONE check; over the limit logged (the tracer's eager steps are slower: information, not a guard)
  local tag=$1 lim=$2 ms st=fast
  restart $tag || { keeplog $tag-fail; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && return 0
  awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}' && st="SLOW (not retried; eager issue under the tracer)"
  echo "$st $ms" > $O/guard-$tag.txt
  xlog "guard $tag: $st, 1-user ${ms} ms/step (limit $lim; kv4 n16 untraced $(cat $RK4/guard-n16.txt 2>/dev/null))"
}
dfree() { df -BM --output=avail $HOME 2>/dev/null | tail -1 | tr -dc 0-9; }
diskok() {   # diskok TAG: >= DISKMIN MB free on / before the start
  local f; f=$(dfree)
  if [ "$DRY" = 1 ]; then echo "+ disk before $1: ${f:-?} MB free on / (a start needs >= $DISKMIN MB) -- $([ "${f:-0}" -ge $DISKMIN ] && echo OK now || echo "LOW now: the run would refuse this start")"; return 0; fi
  [ "${f:-0}" -ge $DISKMIN ] && return 0
  xlog "$1: VIRHE only ${f:-?} MB free on / (need >= $DISKMIN: the 10 GB guard + this item's dumps) -- not started"; return 1
}
envfile() { local f=$1; shift; if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $* > $f"; else printf '%s\n' "$@" > $f; fi; }
envline() { env | grep -E '^(RADIANCE_|R4D_)' | grep -vE '(KEY|TOKEN|SECRET)' | sort | tr '\n' ' ' | cut -c1-2400; }
drafted() {   # drafted TAG: the DSpark drafter and the adaptive depth are up (stack4's check)
  [ "$DRY" = 1 ] && return 0
  local ok=1
  docker logs $CN 2>&1 | grep -q "DSpark drafter" || { xlog "$1: VIRHE no DSpark drafter line"; ok=0; }
  docker logs $CN 2>&1 | grep -q "draft depth: adaptive" || { xlog "$1: VIRHE no adaptive draft depth line"; ok=0; }
  [ $ok = 1 ]
}
pcreset() { [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o $O/$1-reset.json -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  xlog "$1 prefix cache reset: HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"; }
capture() {   # capture FILE: the engine log with the daemon's receipt times, gzipped (never written raw)
  run "docker logs -t $CN 2>&1 | gzip -1 > $1"
  [ "$DRY" = 1 ] || xlog "engine log $(basename $1): $(du -m $1 | cut -f1) MB gz, $(zcat $1 | wc -l) lines, $(zcat $1 | grep -cE ' step [0-9]+ (prefill|decode|MIXED) n_seq=') step lines; $(dfree) MB free"
}
swstart() {   # swstart TAG: the switches at start (stack4's start values; every switch the pins name present); ds41_psk 99 refused
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 at start: gg_gemv 2, aql_acq 2, K1 three 1, epoch 0, draft_capable 1, ds41_bi 0, ds41_psk 0, sched_bi_chunk 0, every switch of $(pinj D) present; POST ds41_psk 99 -> 400"; return 0; }
  local s0 c1 r
  s0=$(fn4get); c1=$(fn4code '{"ds41_psk": 99}')
  r=$(python3 -c 'import json,sys
s0 = json.loads(sys.argv[1]); w = s0["switches"]; want = json.loads(sys.argv[3])
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) and s0["epoch"] == 0
ok = ok and s0["draft_capable"] == 1 and w.get("ds41_bi") == 0 and w.get("ds41_psk") == 0 and w.get("sched_bi_chunk") == 0
miss = [k for k in want if k not in w]
ok = ok and not miss and sys.argv[2] == "400"
print(("OK" if ok else "FAIL") + " ds41_bi %s ds41_psk %s/%s sched_bi_chunk %s draft_occ %s draft_sample %s | ds41_psk 99 -> HTTP %s%s" % (
    w.get("ds41_bi"), w.get("ds41_psk"), w.get("ds41_psk_k"), w.get("sched_bi_chunk"), w.get("draft_occ"), w.get("draft_sample"), sys.argv[2],
    (" | not in the registry: %s" % miss) if miss else ""))' "$s0" "$c1" "$(pinj D)" 2>&1 | cut -c1-500)
  case "$r" in OK*) xlog "$1 switches at start: $r"; return 0 ;; esac
  xlog "$1 switches at start: VIRHE $r | $(echo "$s0" | cut -c1-900)"; return 1
}
# ---------------------------------------------------------------- this item's: the tracer, the probes, the readings
btlmax() { [ -n "$LMAX" ] && echo $LMAX || echo 1073741824; }   # deepseek41's default lmax = 1 << 30 (every layer)
BTW() { local w="RADIANCE_DS41_BITRACE=/stats/bt-$TAG RADIANCE_DS41_BITRACE_PMAX=$PMAX"; [ -n "$LMAX" ] && w="$w RADIANCE_DS41_BITRACE_LMAX=$LMAX"; echo $w; }
startenv() {   # startenv TAG LAT MS HP MML: kv4's startenv (stack4's dstart serve line + RADIANCE_LOG_STEPS=1 + NVG (+ nvfp4)) + the tracer words
  local tag=$1 lat=$2 ms=$3 hp=$4 mml=$5 nv=""
  stackenv; export RADIANCE_LOG_STEPS=1
  [ $NVG -gt 0 ] && { export RADIANCE_DS41_NVGATHER=$NVG; nv="RADIANCE_DS41_NVGATHER=$NVG"; }
  [ $lat = nvfp4 ] && export RADIANCE_DS41_LATENT=nvfp4      # the yaml's list passes it from the shell; unset under bf16
  export MODEL=$M225; HP29=$hp; best_vars
  export MNBT=4160 NST=5 MAXSEQS=$ms MML=$mml XARGS="$XARGS --live --draft-window adaptive --draft-coupling on"
  export XS2_ENVFILE=$O/$tag.env STATSDIR=$O/stats-$tag PROFEVERY=
  MODEWORDS="$LEVENV $PEXTRA RADIANCE_LOG_STEPS=1 $nv $(BTW) $([ $lat = nvfp4 ] && echo RADIANCE_DS41_LATENT=nvfp4)"
  MODEWORDS=$(echo $MODEWORDS)
  envfile $XS2_ENVFILE $LEVENV $PEXTRA RADIANCE_LOG_STEPS=1 $nv $(BTW)
  BTD=$STATSDIR/bt-$tag
  # the tracer's directory BEFORE the start (deepseek41 opens its files at the first step and turns the tracer off when it
  # cannot: bi-b-trace2 09:13), writable by the container
  if [ "$DRY" = 1 ]; then echo "+ mkdir -p $BTD && chmod 777 $BTD (before the start; must be empty)"
  else mkdir -p $BTD && chmod 777 $BTD; fi
}
btmb() { du -sm "$BTD" 2>/dev/null | cut -f1; }
btcheck() {   # btcheck TAG: after the warm-up and ONE more request, every rank's .bin has records and its .names is there
  local r miss="" env mnt
  [ "$DRY" = 1 ] && { echo "+ btcheck $1: a 4-token request, then $BTD/bt-r{0..3}-*.bin non-empty and bt-r{0..3}-*.names present (written and flushed each step)"; return 0; }
  nudge
  for r in 0 1 2 3; do
    ls $BTD/bt-r$r-*.bin > /dev/null 2>&1 && [ -s "$(ls $BTD/bt-r$r-*.bin 2>/dev/null | head -1)" ] || miss="$miss r$r.bin"
    ls $BTD/bt-r$r-*.names > /dev/null 2>&1 || miss="$miss r$r.names"
  done
  if [ -z "$miss" ]; then
    xlog "$1 tracer files: every rank writing ($(ls -la $BTD 2>/dev/null | awk 'NR > 1 && /bt-r/ {n++; b += $5} END {printf "%d files, %.1f MB", n, b / 1e6}') after the warm-up + 1 request; $(dfree) MB free)"
    echo "$1 tracer OK" >> $O/guard.txt
    return 0
  fi
  env=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null | grep '^RADIANCE_DS41_BITRACE')
  mnt=$(docker inspect -f '{{range .Mounts}}{{.Source}}->{{.Destination}} {{end}}' $CN 2>/dev/null)
  xlog "$1: VIRHE tracer wrote no records (missing:$miss) -- container path /stats/bt-$1, env [$(echo $env | tr '\n' ' ')], mounts [$mnt] (want $STATSDIR->/stats), engine: $(docker logs $CN 2>&1 | grep -E "tracer (ON|cannot open)" | head -2 | sed 's/^radiance: //' | tr '\n' '|' | cut -c1-400)"
  echo "$1 tracer VIRHE no records" >> $O/guard.txt
  return 1
}
BTOK=1
btroom() {   # btroom TAG PHASE: the dumps within XB16_BTCAP_MB and / above XB16_DISKMIN_MB, else VIRHE and BTOK 0 (later phases skipped)
  [ "$DRY" = 1 ] && { echo "+ after $2: du $BTD <= $BTCAP MB and >= $DISKMIN MB free on /, else the later phases are skipped (an error line)"; return 0; }
  local s f; s=$(btmb); f=$(dfree)
  xlog "$1 after $2: tracer files ${s:-?} MB, $f MB free"
  if [ "${s:-0}" -gt $BTCAP ] || [ "${f:-0}" -lt $DISKMIN ]; then
    xlog "$1 after $2: VIRHE the dumps ${s:-?} MB > cap $BTCAP or $f MB free < $DISKMIN -- the later phases are skipped"; BTOK=0; return 1
  fi
}
ptok() {   # ptok TAG: the probes' token ids (the analyzer finds their sequences by them)
  run "timeout 120 python3 -I -B $PRB tok --url $URL --gs $GS --idx $IDX --out $O/probes.json > $O/$1-tok.txt 2>&1"
  [ "$DRY" = 1 ] || xlog "$1 probes $IDX tokenized: $(grep -m1 '^TOK ' $O/$1-tok.txt | cut -c5-300)$(grep -q '"ok": true' $O/$1-tok.txt || echo " -- VIRHE no ids: the analyzer falls back to every prompt seen 3 times")"
}
palone() {   # palone TAG NAME: arm D read back, prefix-cache reset, the probes ALONE one at a time
  local tag=$1 nm=$2 t00=$(date +%s)
  armset $tag D || return 1
  pcreset $tag-$nm
  run "(cd $D && timeout 900 python3 -I -B $PRB alone --url $URL --gs $GS --idx $IDX --max-tokens $MT --out $O/$tag-$nm.jsonl > $O/$tag-$nm.txt 2>&1)"
  [ "$DRY" = 1 ] || xlog "$tag $nm (probes $IDX ALONE, T 0, $MT tok, arm D = S + DET: depth 5 fixed): $(grep '^PROBE ' $O/$tag-$nm.txt | tr '\n' ' ' | cut -c1-400)| $(( $(date +%s) - t00 )) s"
}
pload() {   # pload TAG: arm D read back, prefix-cache reset, each probe under CONC mb streams (no newcomer), then the drain
  local tag=$1 t00=$(date +%s) j
  armset $tag D || return 1
  pcreset $tag-ld
  run "(cd $D && timeout 1800 python3 -I -B $PRB load --url $URL --gs $GS --mb $MBP --idx $IDX --conc $CONC --warm $WARMS --ctx $CTX --gen $GEN --max-tokens $MT --out $O/$tag-ld.jsonl > $O/$tag-ld.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  j=$(grep -m1 '^LOADP ' $O/$tag-ld.txt | cut -c7-)
  [ -n "$j" ] || { xlog "$tag ld: VIRHE no LOADP line ($(tail -3 $O/$tag-ld.txt | tr '\n' ' ' | cut -c1-400))"; return 1; }
  xlog "$tag ld (probes $IDX each under $CONC mb streams (ctx $CTX, gen $GEN), warm $WARMS s, T 0, arm D: depth 5 fixed): $(python3 -c 'import json,sys
d = json.loads(sys.argv[1])
print("%s of %s answered | %s | streams %s requests done, %s errors, %s alive after the drain | window %s s, drain %s s | window steps %s, decode %s ms a step, %s tok a decode step" % (
    d["answered"], d["of"], " / ".join("p%s %s tok %s steps (decode %s, mixed %s) %.1f s" % (p["i"], p["tokens"], p["steps"], p["decode_steps"], p["mixed"], p["t1"] - p["t0"]) for p in d["probes"]),
    d["streams_done"], d["stream_errors"], d["streams_alive_after_join"], d["window_s"], d["drain_s"], d["window"].get("steps"),
    d["window"].get("ms_decode_step"), d["window"].get("tok_per_decode_step")))' "$j" 2>&1 | cut -c1-900) | $(( $(date +%s) - t00 )) s"
}
pwins() {   # pwins TAG: the step-log windows from the drivers' times (a1, ld, ld-p<i>, a2)
  [ "$DRY" = 1 ] && { echo "+ windows a1 / ld / ld-p<i> / a2 from $O/$1-{a1,ld,a2}.txt -> $O/$1-wins.txt"; return 0; }
  python3 -c 'import json,sys
o, tag = sys.argv[1], sys.argv[2]
def j(nm, pre):
    try:
        return next(json.loads(l[len(pre):]) for l in open("%s/%s-%s.txt" % (o, tag, nm)) if l.startswith(pre))
    except (OSError, StopIteration, ValueError):
        return None
for nm in ("a1", "a2"):
    d = j(nm, "ALONE ")
    if d and d["probes"]: print("%s:%.3f:%.3f" % (nm, min(p["t0"] for p in d["probes"]), max(p["t1"] for p in d["probes"])))
d = j("ld", "LOADP ")
if d:
    print("ld:%.3f:%.3f" % (d["t_start"], d["t_end"]))
    for p in d["probes"]: print("ld-p%d:%.3f:%.3f" % (p["i"], p["t0"], p["t1"]))' $O $1 > $O/$1-wins.txt 2>/dev/null
}
pparse() {   # pparse TAG: the step log's > 48-row decode steps a window, then the tracer analysis -> RESULT lines
  local tag=$1 l
  pwins $tag
  if [ "$DRY" = 1 ]; then
    echo "+ python3 -I -B $BT steps --log $O/$tag-engine.log.gz --win a1:T0:T1 --win ld:.. --win ld-p<i>:.. --win a2:.. --json $O/$tag-steps.json > $O/$tag-steps.out"
    echo "+ python3 -I -B $BT analyze $BTD --probes $O/probes.json --order a1,ld,a2 --json $O/$tag-bi16.json --extract $O/$tag-bi16-extract.json.gz > $O/$tag-bi16.txt -> every 'BI16 ' line + the BI16GUARD line to the log"
    return 0
  fi
  timeout 600 python3 -I -B $BT steps --log $O/$tag-engine.log.gz $(sed 's/^/--win /' $O/$tag-wins.txt | tr '\n' ' ') --json $O/$tag-steps.json > $O/$tag-steps.out 2>&1
  xlog "$tag step log: $(grep -m1 '^STEPLINE ' $O/$tag-steps.out | cut -c10-1800)"
  xlog "$tag tracer files before the analysis: $(btmb) MB in $(ls $BTD 2>/dev/null | wc -l) files"
  timeout 1500 python3 -I -B $BT analyze $BTD --probes $O/probes.json --order a1,ld,a2 --json $O/$tag-bi16.json --extract $O/$tag-bi16-extract.json.gz > $O/$tag-bi16.txt 2>&1
  while IFS= read -r l; do xlog "$tag $(echo "$l" | cut -c1-3200)"; done < <(grep -E '^BI16 ' $O/$tag-bi16.txt)
  grep -q '^BI16 summary' $O/$tag-bi16.txt || xlog "$tag analysis: VIRHE no summary ($(tail -3 $O/$tag-bi16.txt | tr '\n' ' ' | cut -c1-500))"
}
pguard() {   # pguard TAG: the three guards (tracer dumped; > 48 rows reached; the alone reference reproduces itself)
  [ "$DRY" = 1 ] && { echo "+ guards: BI16GUARD records > 0 and $NPROBE probe(s) found in a1 / ld / a2; every loaded run's own steps DT > 48 + the step log's ld window decode n_tok > 48; a2 vs a1 answers IDENTICAL and every traced row / point IDENTICAL -> RESULT xbi16 guards"; return 0; }
  local r l
  r=$(python3 -c 'import json,sys
o, tag, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
g, s, ident = {}, {}, ""
try: g = next(json.loads(l[10:]) for l in open("%s/%s-bi16.txt" % (o, tag)) if l.startswith("BI16GUARD "))
except (OSError, StopIteration, ValueError): pass
try: s = json.load(open("%s/%s-steps.json" % (o, tag)))
except (OSError, ValueError): pass
try: ident = open("%s/identity.txt" % o).read()
except OSError: pass
ld = s.get("ld", {})
a2txt = next((l for l in ident.splitlines() if l.startswith("S:a2-vs-a1")), "")
out = []
out.append(("tracer dumped", g.get("records", 0) > 0 and g.get("probes_found") == n and g.get("runs_ok") == n,
            "%s records, %s of %s probes found, %s with a1 / ld / a2" % (g.get("records"), g.get("probes_found"), n, g.get("runs_ok"))))
out.append(("> 48 decode rows reached", g.get("ld_gt48") == n and ld.get("decode_gt48", 0) > 0,
            "the probes own loaded steps: %s of %s runs with DT > 48; step log ld window: %s of %s decode steps with n_tok > 48 (most %s)" % (
                g.get("ld_gt48"), n, ld.get("decode_gt48"), ld.get("decode"), ld.get("decode_max"))))
out.append(("alone reproduces itself", g.get("control_n") == n and g.get("control_identical") == n and " IDENTICAL" in a2txt,
            "trace a2 vs a1 identical %s of %s; answers: %s" % (g.get("control_identical"), g.get("control_n"), a2txt[len("S:a2-vs-a1 "):][:120] or "-")))
for nm, ok, why in out:
    print("%s %s: %s" % ("OK" if ok else "VIRHE", nm, why))' $O $TAG $NPROBE 2>&1)
  while IFS= read -r l; do [ -n "$l" ] || continue; xlog "$TAG guard $l"; echo "$TAG guard $l" >> $O/guard.txt; done <<< "$r"
}
pclean() {   # pclean TAG: the raw dumps go after the analysis (kept: .names, the json, the extract); a failed analysis keeps them
  # only within XB16_KEEP_MB and the disk guard; XB16_KEEPRAW=1 keeps them within the cap and the disk guard
  [ "$DRY" = 1 ] && { echo "+ cleanup: analysis summary present -> rm $BTD/bt-r*-*.bin (keep .names, $TAG-bi16.json, $TAG-bi16-extract.json.gz); else keep only when <= $KEEPMB MB and >= $DISKMIN MB + them free (XB16_KEEPRAW=1: keep within $BTCAP MB)"; return 0; }
  local s f keep=0 why
  s=$(btmb); f=$(dfree)
  if grep -q '^BI16 summary' $O/$TAG-bi16.txt 2>/dev/null && [ -s $O/$TAG-bi16-extract.json.gz ]; then
    why="analysed (extract $(du -k $O/$TAG-bi16-extract.json.gz | cut -f1) kB)"; [ $KEEPRAW = 1 ] && [ "${s:-0}" -le $BTCAP ] && [ $((f - s)) -ge $DISKMIN ] && keep=1
  else
    why="analysis FAILED"; [ "${s:-0}" -le $KEEPMB ] && [ $((f - s)) -ge $DISKMIN ] && keep=1
  fi
  if [ $keep = 1 ]; then xlog "$TAG raw tracer files KEPT: ${s} MB in $BTD ($why; $f MB free)"; return 0; fi
  rm -f $BTD/bt-r*-*.bin
  xlog "$TAG raw tracer files DELETED: ${s} MB ($why); left in $BTD: $(du -sk $BTD 2>/dev/null | cut -f1) kB (.names); $(dfree) MB free"
}
trap 'down' EXIT
trap 'exit 143' TERM INT
# ---------------------------------------------------------------- main
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) pins $PINS | lever BI's tracer (first differing op loaded vs alone at > 48 decode rows) | one start $TAG = kv4's n16 (nvfp4, MAXSEQS 16) + $(BTW) | probes $IDX: a1 alone, ld under $CONC mb streams (ctx $CTX gen $GEN, warm $WARMS s), a2 alone | arm D $(armj D) | ~12 min (budget $BUDGET)"
IMGLITS="B|dspark_conf_rows, gg_pf, f32_pf, dspark_walk, ds41_bi, draft_sample, draft_lend_hyst, arena_d0, draft_lend_w, draft_occ, draft_occ_k, place_lf, sched_bi_chunk, sched_interleave, sched_mixed_tok,
B|ds41_psk
B|reset_prefix_cache
B|/tokenize
B|radiance:engine_steps_mixed_total
B|radiance:engine_step_seconds_decode_total
A|RADIANCE_DS41_PSK
A|lever PSK declared (RADIANCE_DS41_PSK, fn4 ds41_psk, now
B|place_lf
A|RADIANCE_DS41_KVFOLD
K|RADIANCE_DS41_GG_PF_TABLE
A|RADIANCE_DS41_DSPARK_WALK
A|RADIANCE_DS41_ARENA_DECODE_ROWS
A|lever BI declared (RADIANCE_DS41_BI, fn4 ds41_bi, now
B|RADIANCE_DRAFT_SAMPLE
A|DSpark lever SD: the sampled full-vocabulary walk declared
B|RADIANCE_WEIGHT_SIDECAR
B|read from the weight sidecar
B|draft_occ takes 0 (off) .. 64
B|lever Z0-A set up
B|RADIANCE_DRAFT_CTXSKIP_SLOTS
B|ds41_psk_k
A|latent %s (%lld bytes an entry; RADIANCE_DS41_LATENT)
A|RADIANCE_DS41_LATENT
K|RADIANCE_DS41_NVGATHER
A|lever BI stats: %lld segmented steps
A|steps with > 48 decode rows
B|RADIANCE_LOG_STEPS
B|n_seq=%lld n_tok=%lld
B|kv: the pool backs %.0f%% of the worst case
A|RADIANCE_DS41_BITRACE
A|RADIANCE_DS41_BITRACE_PMAX
A|RADIANCE_DS41_BITRACE_LMAX
A|lever BI's tracer ON (RADIANCE_DS41_BITRACE=
A|points a step at most, rows at positions %lld..%lld, layers below %d; passes not recorded
A|tracer cannot open
A|bi_rowhash
K|bi_rowhash
B|recorded passes: off, the program traces rows (bi_rowhash"
[ $PINS = s5 ] && IMGLITS="$IMGLITS
X|sched_noa
X|ds41_psk_l20
X|dspark_pfctx
X|ds41_engdd
A|RADIANCE_DS41_ENGDD"
docker image inspect $IMAGE > /dev/null 2>&1 || { xlog "VIRHE image $IMAGE not on this host"; exit 1; }
IMGMISS=$(echo "$IMGLITS" | timeout 120 docker run --rm -i --network none --entrypoint sh -e PINS=$PINS $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels/libr4d.so; S=/opt/radiance/share/radiance/kernels/spechub.so; n=0
while IFS= read -r l; do [ -n "$l" ] || continue; n=$((n + 1)); w=${l%%|*}; s=${l#*|}
  case $w in B) f=$B ;; A) f=$A ;; K) f=$K ;; S) f=$S ;; *) f="$B $A $K" ;; esac
  o=0; for x in $f; do grep -qzF -- "$s" $x && { o=1; break; }; done
  [ $o = 1 ] || echo "MISS $w|$s"; done
test -x /opt/radiance/bin/rad-sidecar || echo "MISS bin/rad-sidecar"
if [ "$PINS" = s4 ]; then test -s /opt/radiance/share/radiance/int7-PROVENANCE.txt || echo "MISS int7-PROVENANCE.txt"
else ls /opt/radiance/share/radiance/*PROVENANCE.txt > /dev/null 2>&1 || echo "MISS a PROVENANCE file"; fi; echo "CHECKED $n"' 2>&1)
case "$IMGMISS" in "CHECKED "*) ;; *) xlog "VIRHE $IMAGE does not carry what this item reads (pins $PINS) -- missing: $(echo "$IMGMISS" | grep -v '^CHECKED' | tr '\n' ';' | cut -c1-900)"; exit 1 ;; esac
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)), pins $PINS: all $(echo "$IMGMISS" | sed 's/CHECKED //') literals present (kv4's registry / BI / SD / Z0 / PSK / latent / NVG / BI stats / step log / pool + lever BI's tracer: env names, the ON line, 'cannot open', bi_rowhash in deepseek41.so + libr4d.so, the core's 'recorded passes: off'$([ $PINS = s5 ] && echo ", the S5 switches")) + bin/rad-sidecar + PROVENANCE -- OK"
HMISS=""
for f in "$LIB29" "$LIBPR" $Y $GS $MBP $TESTS $D/fastcheck.py $PRB $BT; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $GS $MBP $TESTS $D/fastcheck.py $PRB $BT 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
ST1=$(timeout 60 python3 -I -B $PRB selftest --gs $GS --mb $MBP 2>&1 | tail -1); case "$ST1" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$PRB(${ST1:0:300})" ;; esac
ST2=$(timeout 120 python3 -I -B $BT selftest 2>&1 | tail -1); case "$ST2" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$BT(${ST2:0:300})" ;; esac
FNS="log run mkd restart check down stopfile best_vars desc serve_args xlog armj pinj stackenv idrec cmpx identw modes keeplog fn4get fn4set fn4code nudge arm armset gate1 dfree diskok envfile envline drafted pcreset capture swstart btlmax BTW startenv btmb btcheck btroom ptok palone pload pwins pparse pguard pclean"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee sha256sum gzip zcat df du; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
for a in S D; do python3 -c 'import json,sys
d = json.loads(sys.argv[1]); a = sys.argv[2]; s = json.loads(sys.argv[3])
det = json.loads(sys.argv[5]) if a == "D" else {}
ok = d["gg_gemv"] == 2 and d["epoch_bump"] == 1 and d["sched_bi_chunk"] == int(sys.argv[4]) and d["ds41_bi"] == 2 and all(d[k] == v for k, v in s.items() if k != "sched_bi_chunk" and k not in det)
ok = ok and ((d["draft_adaptive"] == 0 and d["draft_depth0"] == 0 and d["draft_occ"] == 0 and d["draft_hybrid"] == 0 and d["dspark_conf"] == 0) if a == "D" else (d["draft_adaptive"] == 1 and d["draft_depth0"] == 1))
sys.exit(0 if ok else 1)' "$(pinj $a '{"epoch_bump": 1}')" $a "$SPIN" "$BIC" "$DET" 2>/dev/null || HMISS="$HMISS json:$a"; done
[ "$(sha256sum $MODELDIR/$SIDECAR 2>/dev/null | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256 2>/dev/null)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR"
[ "$(sha256sum $MODELDIR/$SIDE 2>/dev/null | cut -c1-64)" = "$SIDESHA" ] || HMISS="$HMISS sha256:$MODELDIR/$SIDE(want ${SIDESHA:0:16})"
for w in XS2_ENVFILE 'RADIANCE_DS41_LATENT:' '${STATSDIR:?set STATSDIR}:/stats'; do grep -qF "$w" $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no '$w')"; done
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$O/stats ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config refuses it)"
HMISS=$(echo $HMISS)
[ -z "$HMISS" ] || { xlog "VIRHE helpers / inputs do not resolve: $HMISS"; exit 1; }
HMD5="bi16probe.py $(md5sum < $PRB | cut -c1-8), bi16trace.py $(md5sum < $BT | cut -c1-8) (from bitrace2.py $(md5sum < $HD/../bitrace2.py 2>/dev/null | cut -c1-8)), greedyset.py $(md5sum < $GS | cut -c1-8), mb.py $(md5sum < $MBP | cut -c1-8), lib-29.sh $(md5sum < $LIB29 | cut -c1-8), yaml $(md5sum < $Y | cut -c1-8)"
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing; bi16probe.py $ST1 | bi16trace.py $ST2 | all $(echo $FNS | wc -w) functions resolve, tools on PATH, sidecars $SIDECAR + $SIDE sha256 OK, yaml accepted (env_file, RADIANCE_DS41_LATENT, /stats mount) -- OK | md5: $HMD5"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { xlog "VIRHE $M225 missing or not verified"; exit 1; }
  for f in $WARM_REF_225 $RD/dsd-d0-gs-c1.jsonl $RK4/n16-a1.jsonl $RK4/n16-gs16.jsonl $RK4/n16-lid16.jsonl $RK4/guard-n16.txt; do [ -s $f ] || xlog "note -- reference $f missing (not compared)"; done
  [ -e $O/stats-$TAG/bt-$TAG ] && [ -n "$(ls -A $O/stats-$TAG/bt-$TAG 2>/dev/null)" ] && { xlog "VIRHE $O/stats-$TAG/bt-$TAG is not empty (an earlier run's dumps): not mixed into this one -- move it away first"; exit 1; }
  xlog "$ITEM: helpers md5: $HMD5"
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDE $WARM_REF_225 $RD/dsd-d0-gs-c1.jsonl $RK4/n16-a1.jsonl $RK4/n16-gs16.jsonl $RK4/n16-lid16.jsonl $RK4/guard-n16.txt; do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  [ -e $O/stats-$TAG/bt-$TAG ] && [ -n "$(ls -A $O/stats-$TAG/bt-$TAG 2>/dev/null)" ] && echo "+ NOTE: $O/stats-$TAG/bt-$TAG is not empty now: the real run would refuse to start" || echo "+ tracer directory $O/stats-$TAG/bt-$TAG: absent or empty -- OK"
  for a in S D; do echo "+ arm $a: $(armj $a) | POST body $(pinj $a '{"epoch_bump": 1}')"; done
  echo "+ start $TAG: latent nvfp4, MAXSEQS 16, HOSTPOOL $HP MiB, MML 524288 (kv4's n16) + $(BTW)"
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(information here; the real run refuses to start beside one)"
fi
mkd $O $O/guard
[ "$DRY" = 1 ] || { : > $O/identity.txt; : > $O/guard.txt; }
run1() {   # the one start
  local el
  stopfile && { xlog "$TAG: sweep.stop present -- not started"; return 0; }
  diskok $TAG || return 1
  startenv $TAG nvfp4 16 $HP 524288
  xlog "$TAG: start (~12 min; latent nvfp4, MAXSEQS 16, HOSTPOOL $HP MiB, MML 524288; kv4 n16's start words + $(BTW)) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $FIX + D $(armj D) | env: $(envline)"
  gate1 $TAG 70 || { xlog "$TAG: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$TAG.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  modes $TAG nvfp4 || { xlog "$TAG: VIRHE the modes are not the start's -> stopped"; down; return 1; }
  drafted $TAG || { keeplog $TAG-nodraft; down; return 1; }
  swstart $TAG || { keeplog $TAG-switch; down; return 1; }
  identw $TAG $WARM_REF_225
  btcheck $TAG || { keeplog $TAG-notrace; down; return 1; }
  armset $TAG D || { keeplog $TAG-pin; down; return 1; }
  ptok $TAG
  stopfile || palone $TAG a1
  btroom $TAG a1
  el=$(( ($(date +%s) - T0) / 60 ))
  if [ $BTOK = 1 ] && ! stopfile; then
    if [ "$DRY" != 1 ] && [ $((el + 6)) -gt $BUDGET ]; then xlog "$TAG ld: VIRHE SKIPPED -- $el min elapsed + ~6 (ld + a2 + parse) > the budget $BUDGET min"
    else pload $TAG; btroom $TAG ld; fi
  fi
  [ $BTOK = 1 ] && ! stopfile && { palone $TAG a2; btroom $TAG a2; }
  cmpx S:a2-vs-a1 $O/$TAG-a1.jsonl $O/$TAG-a2.jsonl
  cmpx B48:ld-vs-a1 $O/$TAG-a1.jsonl $O/$TAG-ld.jsonl
  cmpx I:a1-vs-kv4-n16-a1 $RK4/n16-a1.jsonl $O/$TAG-a1.jsonl
  cmpx I:a1-vs-d0 $RD/dsd-d0-gs-c1.jsonl $O/$TAG-a1.jsonl
  cmpx I:ld-vs-kv4-lid16 $RK4/n16-lid16.jsonl $O/$TAG-ld.jsonl
  [ "$DRY" = 1 ] || xlog "$TAG: $(docker logs $CN 2>&1 | grep -cE '^E ') E lines | BI $(docker logs $CN 2>&1 | grep 'lever BI stats' | tail -1 | sed 's/^radiance: deepseek41: //' | cut -c1-200) | tracer $(btmb) MB"
  capture $O/$TAG-engine.log.gz
  down
  pparse $TAG
  pguard $TAG
  pclean
  xlog "$TAG: done ($(( ($(date +%s) - T0) / 60 )) min into the item)"
}
run1
[ "$DRY" = 1 ] || log "RESULT xbi16 guards (item): $([ "$NV" = 0 ] && [ $ARMOK = 1 ] && echo "OK (tracer dumped, > 48 rows reached, alone reproduced, pins read back)" || echo "VIRHE in $NV place(s) -- see the lines above; a reading with VIRHE is not a reading")"
unset $SWITCHES
down
xlog "$ITEM: done in $(( ($(date +%s) - T0) / 60 )) min -- $O ($(dfree) MB free)"
