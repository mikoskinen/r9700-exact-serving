#!/bin/bash
# dsm-dst2.sh (12.11.2026, queue-ar/dsmain/dst2, ds-main-port noa builder) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r00zzz-cz-tdst2.sh. Log tag "RESULT xdst2 ", out runs/xdsm/dst2 (XDT_OUT). = a COPY of dst/dsm-dst.sh 659b86c7
# (the original not edited): its start (r00z's = stack4's NST 5 start: the z0occ2 start + RADIANCE_DS41_PSK=0, + RADIANCE_LOG_STEPS=1),
# start checks, pin helpers, identity / metrics helpers and the probe's chunk tools kept; the cells, the load, the guard and the
# summary replaced (helpers dst2/dstld2.py = a copy of dst/dstld.py 24c2506f, dst2/dstguard2.py = a copy of dst/dstguard.py
# ab79e7a5, dst2/mbt.py = a copy of stack4/mbt.py 94db30da).
# WHY (r00zzz-c-dst readout 12.11.): under the kept chunking D3 (sched_bi_chunk 3) a short newcomer cannot take a partial piece
# beside the long prompt's 2048-row grid chunk, so it waited for the WHOLE long prefill (l18: turnover TTFT max 23.8 s vs 6.7 s
# in r00z); the DST bypass acts only on held steps, which need decode rows (l25 / l26 / l51 / l54 / l55: bypassed 0, refused 0).
# LEVER NOA (ds41-noa, fn4 sched_noa): a prefill D turns away beside another prefill takes D's piece (alone's whole, else its
# largest grid cut >= 128 rows a side) at the step's max_tok room (4160 less every planned row); 2 = as many as fit (several
# newcomers in one step); over max_tok: waits once, then displaces the long chunk one step (bounded). Exact as D is.
# ONE NST 5 START on the NOA image; PIN (every cell): r00z's pin (ds41_bi 2, draft_sample XDT_DS, Z0 switches 0, draft_occ 0 / k 32)
# + PSK design 2 k 4 (ds41_psk 2, ds41_psk_k 4) + interleave 8, budget 2112 + THE DRAFTING STATE PINNED: draft_depth0 2 (depth 0
# forced) and draft_probe 4096 (one probe a 4096 cycles), so c4 tok/s is not confounded by depth flips (lever DP0).
# CELLS (sched_bi_chunk / sched_noa / sched_short_bypass + total): A = D3 base (3 / 0 / 0), N = D3 + NOA (3 / XDT_NOA 2 / 0),
# Y = D3 + bypass (3 / 0 / 1280 + 4096: today's best), NY = D3 + NOA + bypass, Z = sched_bi_chunk 0 (r00z-like, NOT exact: identity
# is information there).
# LOADS: (l) DECODE ALIVE: ONE long greedy stream (4096 tokens, ignore_eos, started first: >= 1 decode stream through the whole
# prefill) + 3 mb.py closed-loop c4 streams (512 tokens, turnovers), 10 s later ONE cold ~64k deepq prompt (128 tokens), and
# THREE NEWCOMERS INSIDE ITS PREFILL WINDOW at +3 / +8 / +14 s (~1100 / ~1330 / ~4400 tokens, 64 greedy tokens; every prompt
# carries this item's nonce: the same bytes in every cell and alone, fresh -- the prefix cache is reset before every run and
# each must read cached 0); (n) NO DECODE on purpose: the probe and the three newcomers alone; (c8) the coordinator's c8 cell:
# mb c8 exactly as stack4 (stack4's mbt.py, temperature 0, dur 60 warm 15), then a deterministic c8 identity burst (8 workers x
# 2 requests of ~1000 tokens, 64 greedy tokens, ignore_eos: turnover waves of 8 newcomers).
# ORDER (ABBA over the whole item): l A N Y NY Z | c8 A N Z | n A N Y NY Z | n Z NY Y N A | c8 Z N A | l Z NY Y N A; the n mirror
# half is dropped when the item has run past XDT_BUDGET_MIN - 15 min at its start (logged).
# READINGS per run: newcomer TTFT p50 / max / each, c4 turnover TTFT p50 / max, in-request TPOT p50 / p90 / max of the decode
# streams, the long prompt's TTFT, c4 tok/s in the window, mixed (non-decode) step ms, noa / dst counter deltas, the probe's
# chunks; c8: agg_dec_tps, decode_share, non-decode step ms (= the newcomer steps' mean wall, /metrics), newcomers per step
# (step log: prefill entries >= 16 rows a step; fn4 noa.pf_steps_*), the burst's TTFT. IDENTITY: the long answer and every
# newcomer vs alone, the c8 burst vs alone -- strict under the D3 cells (A / N / Y / NY: a difference is a VIRHE finding),
# information under Z.
# GUARD (dstguard2.py; a VIRHE = no speed reading): pins read back before and after; PSK design 2's corrected counters grow
# (latched / planned_steps / cut_passes / rows_skipped_*), plan_mismatch stays (never layers_skipped); NOA counters grow in N / NY,
# stay 0 elsewhere; the bypass counters grow in Y / NY with decode rows (information in the no-decode case), stay 0 elsewhere;
# draft tokens ~0 under the pin; newcomers fresh; the long stream alive through the window; D's grid on the probe's chunks.
# ESTIMATE ~45 min (start ~5, checks + smoke ~4, alone references ~2, 10 l runs ~13, 6 c8 runs ~10, 10 n runs ~8, summary ~1);
# the item's budget XDT_BUDGET_MIN 70.
# NO cc_sim, no 2.9, no power / host / fan settings; production compose files read only. Exact (byte-identical) levers only.
# Knobs: XDT_IMAGE (required), XDT_ITEM, XDT_OUT, XDT_NOA (2), XDT_BYPASS (1280), XDT_BYPTOT (4096), XDT_IL (8), XDT_PSKV (2),
#   XDT_PSKK (4), XDT_LDEPTH (64000), XDT_NTOK (128), XDT_NEWC_AT (3,8,14), XDT_NEWC_CTX (1100,1330,4400), XDT_NEWC_TOK (64),
#   XDT_LONG_TOK (4096), XDT_CELLS_L / _N / _C8 (the halves: "A N Y NY Z" / "A N Y NY Z" / "A N Z"), XDT_C8_DUR (60), XDT_C8_WARM
#   (15), XDT_C8_REQS (2), XDT_C8_TOK (64), XDT_BUDGET_MIN (70), XDT_DS (1), XDT_PROBE (4096), XDT_SIDECAR, XDT_SIDESHA,
#   XDT_PSK_STATS, XDT_PSK_STATS_C8, XDT_PSK_ZERO.
# DRYRUN=1 bash dsm-dst2.sh : prints every command; checks the image literals (NOA + DST + PSK + BI chunk + the int5 levers),
# every helper (files, parse, dstld2.py / dstguard2.py / mbt.py selftests, deepq.py = dstld2's request), the sidecars, the yaml;
# no server, no GPU, writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/../ds41-stack2-tp4.yaml                   # stack2's yaml, read only (env_file ${XS2_ENVFILE} carries every start word)
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh
O=${XDT_OUT:-$R/xdsm/dst2}; export STATSDIR=$O/stats
RS3=$R/xdsm/stack3
export IMAGE=${XDT_IMAGE:?set XDT_IMAGE}
ITEM=${XDT_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-tdst2}
NOAV=${XDT_NOA:-2}; BYP=${XDT_BYPASS:-1280}; BYPTOT=${XDT_BYPTOT:-4096}; IL=${XDT_IL:-8}; PSKV=${XDT_PSKV:-2}; PSKK=${XDT_PSKK:-4}
LDEPTH=${XDT_LDEPTH:-64000}; NTOK=${XDT_NTOK:-128}; LONGTOK=${XDT_LONG_TOK:-4096}
NEWC_AT=${XDT_NEWC_AT:-3,8,14}; NEWC_CTX=${XDT_NEWC_CTX:-1100,1330,4400}; NEWC_TOK=${XDT_NEWC_TOK:-64}
CELLS_L=${XDT_CELLS_L:-A N Y NY Z}; CELLS_N=${XDT_CELLS_N:-A N Y NY Z}; CELLS_C8=${XDT_CELLS_C8:-A N Z}
C8DUR=${XDT_C8_DUR:-60}; C8WARM=${XDT_C8_WARM:-15}; C8REQS=${XDT_C8_REQS:-2}; C8TOK=${XDT_C8_TOK:-64}
BUDGET=${XDT_BUDGET_MIN:-70}; DS=${XDT_DS:-1}; PROBEV=${XDT_PROBE:-4096}; OCCK=32
PSKPATHS=${XDT_PSK_STATS:-psk.latched psk.planned_steps psk.cut_passes psk.rows_skipped_l21_29 psk.rows_skipped_l30_39}
PSKZERO=${XDT_PSK_ZERO:-psk.plan_mismatch}
# c8: the newcomers are ~1000-1280-token prefills; PSK's cone at layers 21..29 (160 + (39 - l) x 128 >= 1440 rows) covers them
# whole, so rows_skipped_l21_29 legitimately stays 0 there -- the c8 guard reads the other counters
PSKPATHS_C8=${XDT_PSK_STATS_C8:-psk.latched psk.planned_steps psk.cut_passes psk.rows_skipped_l30_39}
SIDE=${XDT_SIDECAR:-dspark-dlw-q8_0.rad}; SIDESHA=${XDT_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
case $NOAV in 1|2) ;; *) echo "XDT_NOA takes 1 or 2 (sched_noa in the NOA cells)"; exit 1 ;; esac
for v in $BYP $BYPTOT $IL $PSKV $PSKK $LDEPTH $NTOK $LONGTOK $NEWC_TOK $C8DUR $C8WARM $C8REQS $C8TOK $BUDGET $DS $PROBEV; do
  case $v in ''|*[!0-9]*) echo "XDT_* numbers: got '$v'"; exit 1 ;; esac; done
for v in $CELLS_L $CELLS_N $CELLS_C8; do case $v in A|N|Y|NY|Z) ;; *) echo "XDT_CELLS_*: A N Y NY Z (got '$v')"; exit 1 ;; esac; done
case "$NEWC_AT$NEWC_CTX" in *[!0-9.,]*) echo "XDT_NEWC_AT / XDT_NEWC_CTX: comma lists of numbers"; exit 1 ;; esac
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
DQ=$QA/deepq.py; PL=$HD/dstld2.py; GD=$HD/dstguard2.py; MBT=$HD/mbt.py; MBP=$D/mb.py; TESTS=$D/ds41_tests.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
T_ITEM=$(date +%s)
NONCE=tdst2-$T_ITEM            # the newcomers' / c8 burst's nonce: one for the whole item (alone and every cell)
xlog() { log "RESULT xdst2 $*"; }
# ---------------------------------------------------------------- r00z's start (the z0occ2 item's + RADIANCE_DS41_PSK=0) + RADIANCE_LOG_STEPS=1
GPTAB=q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422
SIDECAR=dspark-conf.f32
LEVENV="RADIANCE_DS41_GG_PF=1 RADIANCE_DS41_GG_PF_TABLE=$GPTAB RADIANCE_DS41_F32_PF=25 RADIANCE_DS41_ARENA_DECODE_ROWS=8,48
  RADIANCE_DS41_ROUTER_BF16=1 RADIANCE_DS41_EMB_RAM=1 RADIANCE_PLACE_SHADOW_SLOTS=16 RADIANCE_XPREFETCH=0 RADIANCE_DS41_DSPARK_WALK=1
  RADIANCE_DS41_ARGATHER_MAX=48 RADIANCE_DS41_QNFOLD=1 RADIANCE_DS41_SWFOLD=1 RADIANCE_DRAFT_CTXSKIP=0 RADIANCE_DS41_DSPARK_CONF=/models/$SIDECAR
  RADIANCE_DRAFT_LEND=0 RADIANCE_DRAFT_PROBE_BACKOFF=0 RADIANCE_DRAFT_LEND_BORROW=0 RADIANCE_DRAFT_SAMPLE=0 RADIANCE_WEIGHT_SIDECAR=/models/$SIDE
  RADIANCE_DS41_BI=0 RADIANCE_DRAFT_CTXSKIP_SLOTS=129 RADIANCE_DRAFT_LEND_WEIGHTS=0 RADIANCE_DS41_PSK=0"
LEVENV=$(echo $LEVENV)
STEPENV="RADIANCE_LOG_STEPS=1"
LITS="gemm_nt_ggml's,prefill,rows,take,lever,GPF's,forms,(RADIANCE_DS41_GG_PF=1|gemm_nt_f32's,prefill,rows,take,lever,GPF-b's,tiled,form,(RADIANCE_DS41_F32_PF=25"
LITS="$LITS|48,tokens,use|a,rank),ram|expert,prefetch:,off"
LITS="$LITS|DSpark,lever,W1:,the,full-vocabulary,Markov,walk,declared,(dspark_walk_step,a,step,+,all_gather,over,4,rank(s)|fn4,dspark_walk=1,"
LITS="$LITS|one,kernel,up,to,48,tokens,a,step|RADIANCE_DS41_QNFOLD;,ds41_qnfold=1)|RADIANCE_DS41_SWFOLD;,ds41_swfold=1)|draft,context,skip,(RADIANCE_DRAFT_CTXSKIP=0|a,confidence,plane,(the,dspark_conf,gate"
LITS="$LITS|draft,lend:,set,up"
LITS="$LITS|DSpark,lever,SD:,the,sampled,full-vocabulary,walk,declared,(dspark_walk_sample_step,a,step,over,4,rank(s)|lever,SD:,fn4,draft_sample,0,at,start"
LITS="$LITS|weight,sidecar:,43,entries|read,from,the,weight,sidecar|lever,BI,declared,(RADIANCE_DS41_BI|now,0):,a,step,of,more"
LITS="$LITS|lever,Z0-A,set,up|activation,arena:,8,tokens,use|ring,of,129,steps|draft,weight,lend,(lever,Z0-W|mover:,lever,Z0-W:"
LITS="$LITS|lever,PSK,declared,(RADIANCE_DS41_PSK|chunk,geometry:,a,prefill,alone,in,its,step,spans,up,to,2,intervals,from,its,second,interval,on;,every,other,step,is,planned,at,2112,tokens"
LITS="$LITS|scheduler:,--prefill-interleave,8:,after,a,prefill,chunk|scheduler:,--busy-prefill-chunk,auto:"
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": 8, "drafter_mx": 1, "gg_pf": 1, "f32_pf": 25, "dspark_walk": 1, "ds41_argather": 48, "idx_split": 1, "ds41_qnfold": 1, "ds41_swfold": 1, "draft_ctxskip": 1, "dspark_conf": 1, "dspark_conf_rows": 1, "draft_lend": 1, "draft_probe_backoff": 3, "draft_hybrid": 1}'
# dst2: the pin = r00z's (FIX5 + ds41_bi 2, draft_sample DS, Z0 switches 0, draft_occ 0 / k 32) + PSK design 2 k PSKK + the
# interleave and budget + THE DRAFTING STATE (draft_depth0 2: depth 0 forced, draft_probe PROBEV); the cell adds sched_bi_chunk,
# sched_noa, sched_short_bypass, sched_bypass_total
FIX=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update({"ds41_bi": 2, "draft_sample": int(sys.argv[2]), "draft_lend_hyst": 0, "arena_d0": 0, "draft_lend_w": 0, "draft_occ": 0, "draft_occ_k": int(sys.argv[3]), "draft_depth0": 2, "draft_probe": int(sys.argv[4]), "ds41_psk": int(sys.argv[5]), "ds41_psk_k": int(sys.argv[6]), "sched_interleave": int(sys.argv[7]), "sched_mixed_tok": 0}); print(json.dumps(d))' "$FIX5" "$DS" "$OCCK" "$PROBEV" "$PSKV" "$PSKK" "$IL") || { echo "pin JSON"; exit 1; }
cellj() {   # cellj A|N|Y|NY|Z: the cell's switches
  case $1 in
    A)  echo '{"sched_bi_chunk": 3, "sched_noa": 0, "sched_short_bypass": 0, "sched_bypass_total": 0}' ;;
    N)  echo "{\"sched_bi_chunk\": 3, \"sched_noa\": $NOAV, \"sched_short_bypass\": 0, \"sched_bypass_total\": 0}" ;;
    Y)  echo "{\"sched_bi_chunk\": 3, \"sched_noa\": 0, \"sched_short_bypass\": $BYP, \"sched_bypass_total\": $BYPTOT}" ;;
    NY) echo "{\"sched_bi_chunk\": 3, \"sched_noa\": $NOAV, \"sched_short_bypass\": $BYP, \"sched_bypass_total\": $BYPTOT}" ;;
    Z)  echo '{"sched_bi_chunk": 0, "sched_noa": 0, "sched_short_bypass": 0, "sched_bypass_total": 0}' ;;
  esac
}
cellbic() { case $1 in Z) echo 0 ;; *) echo 3 ;; esac; }
strict() { case $1 in Z) echo 0 ;; *) echo 1 ;; esac; }   # identity strict under the D3 cells, information under Z
pinj() {   # pinj CELLJSON [EXTRAJSON]: the pin + the cell + extra as one POST body
  local x=${2:-}; [ -n "$x" ] || x='{}'
  python3 -c 'import json,sys
d = {}
for a in sys.argv[1:]: d.update(json.loads(a))
print(json.dumps(d))' "$FIX" "$1" "$x"
}
DESC="stack4's / r00z's NST 5 start (the z0occ2 start: stack3 S-arm, stack2 NST 5 env, DSpark adaptive, W1, small n5 set, drafter_mx, lend + back-off 3, VRAM levers, --draft-coupling on, DLW sidecar $SIDE; ring 129, Z0-W declared, rungs 8,48; RADIANCE_DS41_PSK=0, RADIANCE_DS41_BI=0 declared) + $STEPENV; pin: ds41_bi 2, draft_sample $DS, Z0 switches 0, draft_occ 0 / k $OCCK, ds41_psk $PSKV k $PSKK, interleave $IL, budget 2112, draft_depth0 2 + draft_probe $PROBEV (drafting pinned off)"
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
  RADIANCE_DS41_PSK RADIANCE_SCHED_BI_CHUNK RADIANCE_SCHED_BI_GRID RADIANCE_LOG_STEPS RADIANCE_DS41_BITRACE_PMIN
  RADIANCE_DS41_BITRACE_LMAX"
stackenv() {   # 105's t3 arm + gg_gemv 2 + rule-w + the K1 folds + turnover 0 + LEVENV, every other switch unset (= dsm-psk.sh's)
  local kv
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  export RADIANCE_PLACE_TURNOVER=0
  for kv in $LEVENV; do export "$kv"; done
}
# ---------------------------------------------------------------- helpers = dsm-psk.sh 841ce920 lines 114-203 + 223-231 (modes also
# checks RADIANCE_LOG_STEPS=1; mdelta also reports the decode steps' mean ms)
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
thermline() { type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"; }
cmpfile() {   # cmpfile A B -> IDENTICAL (n chars) | DIFFERENT at char k (na vs nb chars) | not compared
  [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
idrec() {   # idrec NAME RESULT [WHAT]: identity.txt + one RESULT line (a difference reported, not hidden)
  echo "$1 $2" >> $O/identity.txt
  case $2 in IDENTICAL*) xlog "identity $1${3:+ ($3)}: $2" ;; *) xlog "identity $1${3:+ ($3)}: $2 -- reported, not hidden" ;; esac
}
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xpsk $1 warm-up identity vs $2"; return 0; }
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
modes() {   # modes TAG: the start words in the container, mover 0 kept free x4, rule-w, K1, ARGATHER 48, every literal of LITS -> 0, else 1
  [ "$DRY" = 1 ] && { echo "+ modes $1: LEVENV in the container's env; mover '0 more kept free' on 4 ranks; rule-w in every recorded pass; K1 folds declared =1; ARGATHER 48-row note; literals [$(echo "$LITS" | tr ',' ' ')]"; return 0; }
  local CE kv bad="" ok=1 NW NR KL AL nz lit found="" miss=""
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in RADIANCE_PLACE_TURNOVER=0 $LEVENV $STEPENV; do echo "$CE" | grep -qxF "$kv" || bad="$bad ${kv%%=*}"; done
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
  xlog "$1 modes: ${#lit[@]} literal(s), found [$(echo $found)]$([ -n "$miss" ] && echo " -- VIRHE missing:$miss") | $(docker logs $CN 2>&1 | grep -E "lever PSK|lever BI declared|lever SD|weight sidecar|lever Z0|activation arena|ring of|draft lend:|draft weight lend|no level for" | sed 's/^radiance: //; s/^[A-Z] *//' | cut -c1-260 | sort -u | tr '\n' '|' | cut -c1-2800)"
  [ -z "$miss" ] || ok=0
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
vramstart() {
  [ "$DRY" = 1 ] && { echo "+ GET /stats > $O/$1-stats0.json; RESULT xpsk $1 VRAM at start (experts per tier; vs stack3 s3-n5)"; return 0; }
  curl -s -m 30 $URL/stats > $O/$1-stats0.json 2>/dev/null
  xlog "$1 VRAM at start: experts $(python3 -c 'import json,sys
for p in sys.argv[1:]:
    try: s = json.load(open(p)); print({x.get("name"): x.get("units") for x in s.get("experts", {}).get("tiers", [])}, end=" ")
    except Exception as e: print("n/a", end=" ")' $O/$1-stats0.json $RS3/s3-n5-stats0.json 2>&1 | cut -c1-320) (this start, then stack3 s3-n5) | $(docker logs $CN 2>&1 | grep -E '^I\[0\] mover:|checkpoint slots|weight sidecar:' | sed 's/^radiance: //; s/^[A-Z]\[0\] //; s/^[A-Z] *//' | cut -c1-220 | sort -u | tr '\n' '|' | cut -c1-1500)"
}
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
fn4code() { [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null; }
pcreset() { [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o $O/$1-reset.json -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  xlog "$1 prefix cache reset: HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"; }
arm() {   # arm CELLJSON: the pin + the cell + epoch_bump, a nudge to apply it at a step (dsm-dst: a cell's JSON, quoted)
  fn4set "$(pinj "$1" '{"epoch_bump": 1}')"; nudge
}
msnap() { [ "$DRY" = 1 ] && { echo "+ GET /metrics > $O/m-$1.txt"; return 0; }; curl -s -m 20 $URL/metrics > $O/m-$1.txt 2>/dev/null; }
mdelta() {   # mdelta A B -> JSON: draft acceptance, non-decode steps and their mean ms, decode steps, prefill tokens over the window (+ the decode steps' mean ms: dsm-dst)
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
dt, da = g("draft_tokens_total"), g("draft_accepted_total")
ds_ = g("engine_step_seconds_decode_total"); dn_ = g("engine_steps_decode_total")
print(json.dumps(dict(acc=round(da / dt, 4) if dt > 0 else None, draft_tok=int(dt), draft_acc=int(da), decode_ms=round(1000 * ds_ / dn_, 1) if dn_ > 0 else None,
      nondecode_steps=int(nd), mixed_steps=int(g("engine_steps_mixed_total")), nondecode_ms=round(1000 * nds / nd, 1) if nd > 0 else None,
      decode_steps=int(g("engine_steps_decode_total")), prefill_tok=int(g("prefill_tokens_total")), decode_tok=int(g("decode_tokens_total")))))' $O/m-$1.txt $O/m-$2.txt 2>/dev/null || echo '{}'
}
pskstat() { [ "$DRY" = 1 ] && { echo "(dry)"; return 0; }; docker logs $CN 2>&1 | grep 'lever PSK:' | tail -1 | sed 's/^radiance: deepseek41: //' | cut -c1-300; }
gate1() {   # restart + ONE check; over the limit logged (in-process A/B: a slow state shows in every arm)
  local tag=$1 lim=$2 ms st=fast
  restart $tag || { keeplog $tag-fail; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && return 0
  awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}' && { st="SLOW (not retried)"; keeplog $tag-slow; }
  echo "$st $ms" > $O/guard-$tag.txt
  xlog "guard $tag: $st, 1-user ${ms} ms/step (limit $lim; stack3 s3-n5 $(cat $RS3/guard-s3-n5.txt 2>/dev/null))"
}
lslice() {   # lslice L0 L1 FILE: the step lines of the engine log between two line counts
  docker logs $CN 2>&1 | sed -n "$(($1 + 1)),${2}p" | grep -E ' step [0-9]+ (prefill|decode|MIXED) ' > $3
}
chunkline() {   # chunkline CHUNKSJSON GRID PROMPT -> "n chunks (m MIXED), sizes {..}, k end off the G grid, j under 128 rows"
  python3 -c 'import json,sys,collections
try: c = json.load(open(sys.argv[1]))
except Exception as e: print("no chunk list (%s)" % type(e).__name__); sys.exit(0)
if not c.get("ok"): print("no chunk list (%s)" % c.get("why")); sys.exit(0)
G, P = int(sys.argv[2]), int(sys.argv[3])
ends = [b + q for b, q in zip(c["boundaries"], c["q"])]
off = sum(1 for e in ends if e % G and e < P - 1)
small = sum(1 for q in c["q"][:-1] if q < 128)
h = collections.Counter(c["q"]).most_common(6)
print("%d chunks (%d MIXED), sizes {%s}, %d end(s) off the %d grid, %d non-final under 128 rows" % (c["n_chunks"], c["mixed_chunks"],
      ", ".join("%dx%d" % kv for kv in h), off, G, small))' "$@" 2>&1 | cut -c1-400
}
# ---------------------------------------------------------------- dst2: switches, smoke, alone references, runs
swnote() {   # swnote TAG: start values (sched_noa 0, the DST switches unset, sched_bi_chunk 0, ds41_psk 0 ...); refusals; two cells read back
  local s0 g c1 sets="" bad="" ck
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 at start (sched_noa 0, sched_bi_chunk 0, sched_short_bypass 0, sched_bypass_total 0, sched_interleave 8, sched_mixed_tok 2112, ds41_psk 0, ds41_bi 0, gg_gemv 2, aql_acq 2, K1 three 1, draft_sample 0, draft_capable 1, epoch 0, GET \"dst\" and \"noa\"); refusals sched_noa 3 / -1 -> 400; cells NY and A applied, the pin + cell read back"; return 0; }
  s0=$(fn4get)
  c1=$(fn4code '{"sched_noa": 3}')$(fn4code '{"sched_noa": -1}')
  for ck in NY A; do
    arm "$(cellj $ck)"; g=$(fn4get)
    python3 -c 'import json,sys
g, w = json.loads(sys.argv[1]), json.loads(sys.argv[2])
if w.get("sched_mixed_tok") == 0: w["sched_mixed_tok"] = 2112
sys.exit(0 if all(g["switches"].get(k) == v for k, v in w.items()) and not g["pending"] else 1)' "$g" "$(pinj "$(cellj $ck)")" 2>/dev/null || bad="$bad $ck"
    sets="$sets [$ck: $(echo "$g" | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print("bic=%s,noa=%s,byp=%s/%s,il=%s,mt=%s,psk=%s/%s,d0=%s,probe=%s,bi=%s,ep=%s" % (w.get("sched_bi_chunk"), w.get("sched_noa"), w.get("sched_short_bypass"), w.get("sched_bypass_total"), w.get("sched_interleave"), w.get("sched_mixed_tok"), w.get("ds41_psk"), w.get("ds41_psk_k"), w.get("draft_depth0"), w.get("draft_probe"), w.get("ds41_bi"), d.get("epoch")))' 2>/dev/null)]"
  done
  xlog "$1 switch note: at start $(echo "$s0" | cut -c1-1400) | sched_noa 3 / -1 -> HTTP $c1 | cells applied:$sets$([ -n "$bad" ] && echo " -- VIRHE not read back:$bad")"
  python3 -c 'import json,sys
s0 = json.loads(sys.argv[1]); w = s0["switches"]
ok = w.get("sched_noa") == 0 and w.get("sched_interleave") == 8 and w.get("sched_mixed_tok") == 2112 and w.get("sched_bi_chunk") == 0
ok = ok and w.get("ds41_psk") == 0 and w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill"))
ok = ok and w.get("ds41_bi") == 0 and s0["epoch"] == 0 and w["draft_adaptive"] == 1 and s0["draft_capable"] == 1 and w.get("draft_sample") == 0
ok = ok and w.get("sched_short_bypass") == 0 and w.get("sched_bypass_total") == 0 and "dst" in s0 and "noa" in s0
sys.exit(0 if ok else 1)' "$s0" 2>/dev/null && [ "$c1" = 400400 ] && [ -z "$bad" ]
}
smoke() {   # smoke TAG: cell NY (NOA + the bypass) -- the greedy set at conc 4; healthy -> 0
  local tag=$1 t0=$(date +%s) n
  arm "$(cellj NY)"
  run "(cd $D && timeout 600 python3 $GS $URL $O/$tag-smoke.jsonl --conc 4 --max-tokens 32 > $O/$tag-smoke.txt 2>&1)"
  [ "$DRY" = 1 ] && { echo "+ smoke $tag: cell NY $(cellj NY): greedy set conc 4 (32 tok) 20 of 20, /health 200"; return 0; }
  n=$(python3 -c 'import json,sys; print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l).get("text") is not None))' $O/$tag-smoke.jsonl 2>/dev/null || echo 0)
  if [ "$n" = 20 ] && curl -s -m 10 -o /dev/null -w '%{http_code}' $URL/health | grep -q 200; then
    xlog "$tag smoke (cell NY: NOA $NOAV + bypass $BYP / $BYPTOT, PSK $PSKV): greedy set conc 4 20 of 20, healthy, $(( $(date +%s) - t0 )) s | noa $(fn4get | python3 -c 'import json,sys; print(json.load(sys.stdin).get("noa"))' 2>/dev/null)"; return 0; fi
  xlog "$tag smoke: VIRHE $n of 20 answered or unhealthy | $(docker logs $CN 2>&1 | grep -E '^E |step fail|refus' | tail -3 | tr '\n' ' ' | cut -c1-500)"
  keeplog $tag-smoke; return 1
}
snap() {   # snap TAG-NAME K: /metrics, fn4 GET and /stats (K = 0 before, 1 after)
  msnap $1-$2
  [ "$DRY" = 1 ] && return 0
  fn4get > $O/$1-fn4-$2.json; curl -s -m 20 $URL/stats > $O/$1-stats-$2.json 2>/dev/null
}
loglines() { [ "$DRY" = 1 ] && { echo L; return 0; }; docker logs $CN 2>&1 | wc -l; }
pfsteps() {   # pfsteps STEPS.log -> "steps with prefill pieces (>= 16 rows) by count: 1 n1, 2 n2, 3+ n3; mean x"
  python3 -c 'import re,sys,collections
E = re.compile(r"\[(\d+)\] slot=(-?\d+) q=(\d+) ctx=(\d+)")
h = collections.Counter()
try:
    for l in open(sys.argv[1], errors="replace"):
        n = sum(1 for _, s, q, c in E.findall(l) if int(q) >= 16)
        if n: h[min(n, 3)] += 1
except OSError: pass
t = sum(h.values())
print("prefill steps %d: 1 piece %d, 2 %d, 3+ %d; mean %.2f" % (t, h[1], h[2], h[3], (sum(k * v for k, v in h.items()) / t) if t else 0))' "$1" 2>&1 | cut -c1-200
}
guard() {   # guard MODE CELL TAG-NAME [CHUNKS ALONECHUNKS] -> the guard's verdict line (GUARD ...)
  local mode=$1 ck=$2 tn=$3 chj=${4:-} alj=${5:-} md
  [ "$DRY" = 1 ] && { echo "OK (dry)"; return 0; }
  md=$(mdelta $tn-0 $tn-1)
  timeout 120 python3 -I $GD check --mode $mode --cell "$(cellj $ck | python3 -c 'import json,sys; d=json.load(sys.stdin); d["ds41_psk"]=int(sys.argv[1]); print(json.dumps(d))' $PSKV)" \
    --want "$(pinj "$(cellj $ck)")" --bic $(cellbic $ck) --grid 2048 --il $([ $mode = l ] && echo $IL || echo 0) \
    ${chj:+--chunks $chj} ${alj:+--alone $alj} --fn4a $O/$tn-fn4-0.json --fn4b $O/$tn-fn4-1.json --stats-a $O/$tn-stats-0.json \
    --stats-b $O/$tn-stats-1.json $([ $mode != c8 ] && echo "--info $O/$tn.json") --md "$md" --psk-paths "$([ $mode = c8 ] && echo "$PSKPATHS_C8" || echo "$PSKPATHS")" --psk-zero "$PSKZERO" 2>&1 \
    | grep -m1 '^GUARD ' | cut -c7- || true
}
newcid() {   # newcid ALONE.json LOADED.json -> IDENTICAL (n newcomers) | DIFFERENT: [k, ..] of n | not compared
  python3 -c 'import json,sys
try: a = {x["k"]: x for x in json.load(open(sys.argv[1]))}; l = {x["k"]: x for x in json.load(open(sys.argv[2]))}
except Exception as e: print("not compared (%s)" % type(e).__name__); sys.exit(0)
bad = [k for k in sorted(a) if not a[k].get("text") or (l.get(k) or {}).get("text") != a[k]["text"]]
print(("IDENTICAL (%d newcomers)" % len(a)) if a and not bad else ("DIFFERENT: newcomer(s) %s of %d" % (bad, len(a))))' "$1" "$2" 2>&1 | cut -c1-200
}
idrec2() {   # idrec2 CELL NAME RESULT WHAT: strict (a VIRHE finding) under D3 cells, information under Z
  echo "$2 $3" >> $O/identity.txt
  case $3 in IDENTICAL*) xlog "identity $2 ($4): $3" ;;
    *) if [ "$(strict $1)" = 1 ]; then xlog "identity $2 ($4): VIRHE $3 -- under D3 every answer must equal alone"; echo "$2" >> $O/identity-bad.txt
       else xlog "identity $2 ($4): $3 -- information (sched_bi_chunk 0 is not exact)"; fi ;;
  esac
}
NS=0
drun() {   # drun TAG CELL MODE(l|n): one 64k probe run -- l: the long stream + 3 c4 streams + the probe + 3 newcomers in its window;
  # n: the probe + the newcomers, no decode stream
  local tag=$1 ck=$2 mode=$3 t0=$(date +%s) nm tn l0 l1 j md pt c gr
  NS=$((NS + 1)); nm=$(printf 'r%02d' $NS)-$mode-$ck; tn=$tag-$nm
  arm "$(cellj $ck)"; pcreset $tn
  snap $tn 0; l0=$(loglines)
  run "(cd $D && timeout 1800 python3 -I $PL run --url $URL --tests $TESTS --mb $MBP --depth $LDEPTH --ntok $NTOK $([ $mode = l ] && echo "--load --conc 4 --warm 10 --long-tok $LONGTOK --stamps $O/$tn-stamps.json ")--newc-at $NEWC_AT --newc-ctx $NEWC_CTX --newc-tok $NEWC_TOK --nonce $NONCE --newc-out $O/$tn-newc.json --out $O/$tn.json --text $O/$tn.txt > $O/$tn.log 2>&1)"
  snap $tn 1
  run "sleep 2"
  if [ "$DRY" = 1 ]; then
    echo "+ step lines L0+1..L1 > $O/$tn-steps.log; python3 -I $PL chunks ... --json $O/$tn-chunks.json; python3 -I $GD check --mode $mode (pins, NOA, bypass, draft pin, PSK [$PSKPATHS] / [$PSKZERO], newcomers fresh$([ $mode = l ] && echo ', the long stream alive'), D grid); identity long + newcomers vs alone ($([ $(strict $ck) = 1 ] && echo strict || echo information)); RESULT xdst2 $tn (cell $ck $(cellj $ck))"
    return 0
  fi
  l1=$(loglines)
  lslice $l0 $l1 $O/$tn-steps.log
  j=$(grep -m1 '^LONGID ' $O/$tn.log | cut -c8-); md=$(mdelta $tn-0 $tn-1)
  [ -n "$j" ] || { xlog "$tn: VIRHE no LONGID line ($(tail -3 $O/$tn.log | tr '\n' ' ' | cut -c1-400))"; return 1; }
  pt=$(echo "$j" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("prompt_tokens") or 0)' 2>/dev/null || echo 0)
  timeout 300 python3 -I $PL chunks $O/$tn-steps.log --prompt $pt --json $O/$tn-chunks.json > /dev/null 2>&1
  c=$(chunkline $O/$tn-chunks.json 2048 $pt)
  gr=$(guard $mode $ck $tn $O/$tn-chunks.json $O/$tag-a-probe-chunks.json); [ -n "$gr" ] || gr="VIRHE the guard gave no verdict"
  case "$gr" in VIRHE*) xlog "$tn: $gr -- no speed reading for this run" ;; esac
  python3 -c 'import json,sys
d = json.loads(sys.argv[1]); d.update(json.loads(sys.argv[2]) if sys.argv[2].strip() else {})
d.update(name=sys.argv[3], cellkey=sys.argv[4], mode=sys.argv[5], cell=json.loads(sys.argv[6]), chunks=sys.argv[7], guard=sys.argv[8])
if sys.argv[8].startswith("VIRHE"): d.update(invalid=sys.argv[8])
for k, f in (("fn4a", sys.argv[9]), ("fn4b", sys.argv[10])):
    try: d[k] = {s: json.load(open(f)).get(s) for s in ("noa", "dst")}
    except Exception: pass
print(json.dumps(d))' "$j" "$md" "$tn" "$ck" "$mode" "$(cellj $ck)" "$c" "$gr" $O/$tn-fn4-0.json $O/$tn-fn4-1.json >> $O/runs.jsonl
  idrec2 $ck "$tn-long-vs-alone" "$(cmpfile $O/$tag-a-probe.txt $O/$tn.txt)" "the 64k answer vs alone"
  idrec2 $ck "$tn-newc-vs-alone" "$(newcid $O/$tag-a-newc.json $O/$tn-newc.json)" "the three newcomers vs each alone"
  xlog "$tn (cell $ck $(cellj $ck), $([ $mode = l ] && echo 'decode ALIVE: long stream + 3 c4 turnover streams' || echo 'NO decode stream'), 64k probe + newcomers at +$NEWC_AT s): $(python3 -c 'import json,sys
d, m = json.loads(sys.argv[1]), json.loads(sys.argv[2] or "{}")
print("probe TTFT %s s (prompt %s, cached %s) | newcomers TTFT p50 %s / max %s ms, each %s, prompts %s, cached %s | c4 turnovers %s, TTFT p50 %s / max %s ms | window %s s: c4 %s tok/s, TPOT in p50 %s / p90 %s / max %s ms, long stream alive %s | mixed %s, non-decode %s at %s ms, decode %s at %s ms, draft tok %s" % (
    d.get("ttft"), d.get("prompt_tokens"), d.get("cached"), d.get("newc_ttft_p50"), d.get("newc_ttft_max"), d.get("newc_ttft"), d.get("newc_prompts"),
    d.get("newc_cached"), d.get("turn_req_n"), d.get("turn_ttft_p50"), d.get("turn_ttft_max"), d.get("pf_win_s"), d.get("c4_pf_agg_tps"),
    d.get("tpot_in_p50"), d.get("tpot_in_p90"), d.get("tpot_in_max"), d.get("long_alive"), m.get("mixed_steps"), m.get("nondecode_steps"),
    m.get("nondecode_ms"), m.get("decode_steps"), m.get("decode_ms"), m.get("draft_tok")))' "$j" "$md" 2>&1 | cut -c1-1200) | noa $(python3 -c 'import json,sys
a, b = (json.load(open(f)).get("noa") or {} for f in sys.argv[1:3])
print({k: b.get(k, 0) - a.get(k, 0) for k in b})' $O/$tn-fn4-0.json $O/$tn-fn4-1.json 2>/dev/null) | dst $(python3 -c 'import json,sys
a, b = (json.load(open(f)).get("dst") or {} for f in sys.argv[1:3])
print({k: b.get(k, 0) - a.get(k, 0) for k in b})' $O/$tn-fn4-0.json $O/$tn-fn4-1.json 2>/dev/null) | guard: ${gr:0:500} | chunks: $c | $(thermline $t0 $(date +%s))"
}
c8run() {   # c8run TAG CELL: mb c8 as stack4 (mbt.py T 0, dur C8DUR warm C8WARM), then the deterministic c8 identity burst
  local tag=$1 ck=$2 t0=$(date +%s) nm tn l0 l1 mb gr ph id
  NS=$((NS + 1)); nm=$(printf 'r%02d' $NS)-c8-$ck; tn=$tag-$nm
  arm "$(cellj $ck)"; pcreset $tn
  snap $tn 0; l0=$(loglines)
  run "(cd $D && timeout 900 python3 -I -B $MBT --mb $MBP --temperature 0 --tag $tn --phases c8 --dur $C8DUR --warm $C8WARM > $O/$tn-mb.txt 2>&1)"
  pcreset $tn-id
  run "(cd $D && timeout 900 python3 -I $PL c8id --url $URL --tests $TESTS --mb $MBP --conc 8 --reqs $C8REQS --ctx 1000 --tok $C8TOK --nonce $NONCE --c8-out $O/$tn-c8id.json > $O/$tn-c8id.log 2>&1)"
  snap $tn 1
  if [ "$DRY" = 1 ]; then
    echo "+ step lines > $O/$tn-steps.log; newcomers a step (pfsteps); python3 -I $GD check --mode c8 (pins, NOA, bypass 0, draft pin, PSK [$PSKPATHS_C8] / [$PSKZERO]; mb errors 0, the burst answered); identity: python3 -I $PL cmpc8 $O/$tag-a-c8id.json $O/$tn-c8id.json ($([ $(strict $ck) = 1 ] && echo strict || echo information)); RESULT xdst2 $tn (cell $ck)"
    return 0
  fi
  l1=$(loglines)
  lslice $l0 $l1 $O/$tn-steps.log
  mb=$(grep -m1 '^MB ' $O/$tn-mb.txt | cut -c4-)
  [ -n "$mb" ] || { xlog "$tn: VIRHE no MB line ($(tail -3 $O/$tn-mb.txt | tr '\n' ' ' | cut -c1-300))"; return 1; }
  ph=$(pfsteps $O/$tn-steps.log)
  gr=$(guard c8 $ck $tn); [ -n "$gr" ] || gr="VIRHE the guard gave no verdict"
  # the load itself: mb's c8 requests and the burst answered (an error is a VIRHE: no speed reading)
  local me; me=$(echo "$mb" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("errors", 1) if d.get("requests") else "no requests")' 2>/dev/null || echo "?")
  [ "$me" = 0 ] || gr="VIRHE mb c8 errors: $me; $gr"
  grep -q '^C8ID burst' $O/$tn-c8id.log && ! grep -q 'VIRHE' $O/$tn-c8id.log || gr="VIRHE the c8 identity burst failed ($(tail -1 $O/$tn-c8id.log | cut -c1-200)); $gr"
  case "$gr" in VIRHE*) xlog "$tn: $gr -- no speed reading for this run" ;; esac
  id=$(timeout 60 python3 -I $PL cmpc8 $O/$tag-a-c8id.json $O/$tn-c8id.json 2>&1 | tail -1 | cut -c1-200)
  python3 -c 'import json,sys
d = json.loads(sys.argv[1])
dec_steps = d["dec_tok"] / d["tok_per_decode_step"] if d.get("tok_per_decode_step") else 0
nd = d.get("steps", 0) - dec_steps
d.update(nondecode_ms=round(1000 * d["busy_s"] * (1 - d["decode_share"]) / nd, 1) if nd > 0 else None, name=sys.argv[2], cellkey=sys.argv[3], mode="c8",
         cell=json.loads(sys.argv[4]), pfsteps=sys.argv[5], guard=sys.argv[6], identity=sys.argv[7])
if sys.argv[6].startswith("VIRHE"): d.update(invalid=sys.argv[6])
for k, f in (("fn4a", sys.argv[8]), ("fn4b", sys.argv[9])):
    try: d[k] = {s: json.load(open(f)).get(s) for s in ("noa", "dst")}
    except Exception: pass
print(json.dumps(d))' "$mb" "$tn" "$ck" "$(cellj $ck)" "$ph" "$gr" "$id" $O/$tn-fn4-0.json $O/$tn-fn4-1.json >> $O/runs.jsonl
  idrec2 $ck "$tn-c8id-vs-alone" "$id" "the c8 burst's 16 answers vs each alone"
  xlog "$tn (cell $ck $(cellj $ck), mb c8 T 0 dur $C8DUR warm $C8WARM + the c8 identity burst): $(tail -1 $O/runs.jsonl | python3 -c 'import json,sys
d = json.load(sys.stdin)
print("agg_dec_tps %s, decode_share %s, decode step %s ms, non-decode (newcomer) step %s ms, per-stream p50 %s, TTFT p50 %s s, requests %s / errors %s, preempt %s" % (
    d.get("agg_dec_tps"), d.get("decode_share"), d.get("ms_decode_step"), d.get("nondecode_ms"), d.get("per_stream_dec_p50"), d.get("ttft_p50"),
    d.get("requests"), d.get("errors"), d.get("preempt")))' 2>&1 | cut -c1-600) | $ph | noa $(python3 -c 'import json,sys
a, b = (json.load(open(f)).get("noa") or {} for f in sys.argv[1:3])
print({k: b.get(k, 0) - a.get(k, 0) for k in b})' $O/$tn-fn4-0.json $O/$tn-fn4-1.json 2>/dev/null) | burst $(grep -m1 '^C8ID' $O/$tn-c8id.log | cut -c1-200) | identity $id | guard: ${gr:0:400} | $(thermline $t0 $(date +%s))"
}
alones() {   # alones TAG: the alone references under cell A: the 64k probe (no load, no newcomers), the newcomers one by one, the c8 burst's prompts one by one
  local tag=$1 l0 l1 pt
  arm "$(cellj A)"; pcreset $tag-a-probe
  l0=$(loglines)
  run "(cd $D && timeout 1800 python3 -I $PL run --url $URL --tests $TESTS --mb $MBP --depth $LDEPTH --ntok $NTOK --out $O/$tag-a-probe.json --text $O/$tag-a-probe.txt > $O/$tag-a-probe.log 2>&1)"
  if [ "$DRY" != 1 ]; then
    l1=$(loglines); lslice $l0 $l1 $O/$tag-a-probe-steps.log
    pt=$(grep -m1 '^LONGID ' $O/$tag-a-probe.log | cut -c8- | python3 -c 'import json,sys; print(json.load(sys.stdin).get("prompt_tokens") or 0)' 2>/dev/null || echo 0)
    timeout 300 python3 -I $PL chunks $O/$tag-a-probe-steps.log --prompt $pt --json $O/$tag-a-probe-chunks.json > /dev/null 2>&1
    xlog "$tag alone 64k probe (cell A, PSK $PSKV): $(grep -m1 '^LONGID ' $O/$tag-a-probe.log | cut -c8- | python3 -c 'import json,sys; d=json.load(sys.stdin); print("status %s, prompt %s, cached %s, TTFT %s s, %s tok" % (d.get("status"), d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("tokens")))' 2>&1 | cut -c1-200) | chunks: $(chunkline $O/$tag-a-probe-chunks.json 2048 $pt)"
  fi
  pcreset $tag-a-newc
  run "(cd $D && timeout 900 python3 -I $PL newc-alone --url $URL --tests $TESTS --mb $MBP --newc-ctx $NEWC_CTX --newc-tok $NEWC_TOK --nonce $NONCE --newc-out $O/$tag-a-newc.json > $O/$tag-a-newc.log 2>&1)"
  [ "$DRY" = 1 ] || xlog "$tag newcomers alone: $(grep -m1 '^NEWC ' $O/$tag-a-newc.log | cut -c1-300)"
  pcreset $tag-a-c8id
  run "(cd $D && timeout 900 python3 -I $PL c8id-alone --url $URL --tests $TESTS --mb $MBP --conc 8 --reqs $C8REQS --ctx 1000 --tok $C8TOK --nonce $NONCE --c8-out $O/$tag-a-c8id.json > $O/$tag-a-c8id.log 2>&1)"
  [ "$DRY" = 1 ] || xlog "$tag c8 burst prompts alone: $(grep -m1 '^C8ID' $O/$tag-a-c8id.log | cut -c1-300)"
}
elapsed() { echo $(( ($(date +%s) - T_ITEM) / 60 )); }
plan() {   # plan TAG: the ABBA order over the whole item; the no-decode mirror half only while the budget allows
  local tag=$1 c rev
  rev() { echo $* | tr ' ' '\n' | tac | tr '\n' ' '; }
  xlog "$tag order: l [$CELLS_L] | c8 [$CELLS_C8] | n [$CELLS_N] | n [$(rev $CELLS_N)] (if <= $((BUDGET - 15)) min) | c8 [$(rev $CELLS_C8)] | l [$(rev $CELLS_L)]; NOA $NOAV, bypass $BYP / $BYPTOT, nonce $NONCE"
  for c in $CELLS_L; do stopfile && return 0; drun $tag $c l; done
  for c in $CELLS_C8; do stopfile && return 0; c8run $tag $c; done
  for c in $CELLS_N; do stopfile && return 0; drun $tag $c n; done
  if [ "$DRY" = 1 ] || [ $(elapsed) -le $((BUDGET - 15)) ]; then
    for c in $(rev $CELLS_N); do stopfile && return 0; drun $tag $c n; done
  else
    xlog "$tag: $(elapsed) min > $((BUDGET - 15)): the no-decode mirror half [$(rev $CELLS_N)] dropped (each no-decode cell sampled once)"
  fi
  for c in $(rev $CELLS_C8); do stopfile && return 0; c8run $tag $c; done
  for c in $(rev $CELLS_L); do stopfile && return 0; drun $tag $c l; done
}
summary() {
  [ "$DRY" = 1 ] && { echo "+ RESULT xdst2 summary: per (mode, cell) the means over its valid samples (l / n: newcomer TTFT p50 / max, turnover TTFT p50 / max, TPOT in p50 / p90 / max, probe TTFT, c4 tok/s, mixed ms, noa / dst deltas; c8: agg_dec_tps, decode_share, non-decode ms, newcomers a step) and vs cell A (%); INVALID runs; the identity table (strict under D3, information under Z)"; return 0; }
  python3 - "$O" > $O/summary.txt 2>&1 <<'PY'
import json, os, statistics, sys
o = sys.argv[1]
def rows(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except OSError: return []
def m(v):
    v = [x for x in v if isinstance(x, (int, float))]
    return statistics.mean(v) if v else None
def f(x, n=1): return "-" if x is None else format(x, ".%df" % n)
def pc(b, a): return "%+.1f %%" % ((b / a - 1) * 100) if a and b else "-"
def dl(r, s, k):
    a, b = ((r.get("fn4a") or {}).get(s) or {}), ((r.get("fn4b") or {}).get(s) or {})
    return (b.get(k) or 0) - (a.get(k) or 0) if b else None
R = rows(os.path.join(o, "runs.jsonl"))
for r in R:
    if r.get("invalid"): print("INVALID %s (no speed reading): %s" % (r.get("name"), r["invalid"][:400]))
ok = [r for r in R if not r.get("invalid")]
order = ["A", "N", "Y", "NY", "Z"]
for mode, K in (("l", [("newcomer TTFT p50", "newc_ttft_p50", 0), ("max", "newc_ttft_max", 0), ("turnover TTFT p50", "turn_ttft_p50", 0),
                       ("max", "turn_ttft_max", 0), ("TPOT in p50", "tpot_in_p50", 1), ("p90", "tpot_in_p90", 0), ("max", "tpot_in_max", 0),
                       ("long TTFT s", "ttft", 2), ("c4 tok/s", "c4_pf_agg_tps", 1), ("mixed ms", "nondecode_ms", 0)]),
                ("n", [("newcomer TTFT p50", "newc_ttft_p50", 0), ("max", "newc_ttft_max", 0), ("long TTFT s", "ttft", 2), ("mixed ms", "nondecode_ms", 0)]),
                ("c8", [("agg_dec_tps", "agg_dec_tps", 1), ("decode_share", "decode_share", 3), ("non-decode ms", "nondecode_ms", 0),
                        ("decode step ms", "ms_decode_step", 1), ("TTFT p50 s", "ttft_p50", 2)])):
    base = [r for r in ok if r.get("mode") == mode and r.get("cellkey") == "A"]
    for ck in order:
        rs = [r for r in ok if r.get("mode") == mode and r.get("cellkey") == ck]
        if not rs: continue
        parts = []
        for nm, k, n in K:
            a, b = m([r.get(k) for r in rs]), m([r.get(k) for r in base])
            parts.append("%s %s%s" % (nm, f(a, n), "" if ck == "A" else " (%s)" % pc(a, b)))
        cnt = "noa pieces +%s / admitted +%s / displaced +%s / waited +%s, bypassed +%s" % tuple(
            m([dl(r, s, k) for r in rs]) for s, k in (("noa", "pieces"), ("noa", "admitted"), ("noa", "displaced"), ("noa", "waited"), ("dst", "bypassed")))
        extra = (" | newcomers a step: " + " / ".join(r.get("pfsteps", "") for r in rs)) if mode == "c8" else ""
        print("CELL %s %s (n=%d; vs A): %s | %s%s | samples TTFT %s" % (mode, ck, len(rs), ", ".join(parts), cnt, extra,
              "/".join(f(r.get("ttft") if mode != "c8" else r.get("agg_dec_tps"), 2) for r in rs)))
try: ident = [l.strip() for l in open(os.path.join(o, "identity.txt")) if l.strip()]
except OSError: ident = []
bad = [l for l in ident if "IDENTICAL" not in l]
strict_bad = [l for l in bad if "-Z-" not in l.split()[0] and not l.split()[0].endswith("-Z")]
print("SUMMARY guard: %d run(s) invalid of %d || identity: %d compares, %d not identical (%d under the D3 cells: %s) || estimate to test (coordinator): c8 decode_share ~0.79 -> ~0.9, c8 ~172 -> ~195-205 tok/s" % (
      len(R) - len(ok), len(R), len(ident), len(bad), len(strict_bad), "; ".join(strict_bad)[:900] or "none"))
PY
  while IFS= read -r l; do xlog "${l:0:3000}"; done < $O/summary.txt
}
dstart() {   # the one NST 5 start
  local tag=ds-n5 ok
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on"
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV $STEPENV > $XS2_ENVFILE"; else printf '%s\n' $LEVENV $STEPENV > $XS2_ENVFILE; fi
  xlog "$tag: start ($DESC) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $FIX | cells $(for c in A N Y NY Z; do printf '%s=%s ' $c "$(cellj $c | tr -d ' ')"; done)| env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ' | cut -c1-2200)"
  gate1 $tag 70 || { xlog "$tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  modes $tag || { xlog "$tag: VIRHE the modes are not the start's -> stopped"; down; return 1; }
  if [ "$DRY" != 1 ]; then
    ok=1
    docker logs $CN 2>&1 | grep -q "DSpark drafter" || { xlog "$tag: VIRHE no DSpark drafter line"; ok=0; }
    docker logs $CN 2>&1 | grep -q "draft depth: adaptive" || { xlog "$tag: VIRHE no adaptive draft depth line"; ok=0; }
    docker logs $CN 2>&1 | grep -qE ' step [0-9]+ (prefill|decode|MIXED) n_seq=' || { xlog "$tag: VIRHE no step lines after the warm-up (RADIANCE_LOG_STEPS=1)"; ok=0; }
    [ $ok = 1 ] || { keeplog $tag-nodraft; down; return 1; }
  fi
  vramstart $tag
  swnote $tag || { xlog "$tag: VIRHE the switches are not as expected -> stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  smoke $tag || { xlog "$tag: VIRHE the smoke failed -> stopped"; down; return 1; }
  stopfile || alones $tag
  stopfile || plan $tag
  run "docker logs $CN 2>&1 | gzip -1 > $O/$tag-engine-full.log.gz"
  xlog "$tag: PSK stats at the end: $(pskstat) | noa $(fn4get | python3 -c 'import json,sys; print(json.load(sys.stdin).get("noa"))' 2>/dev/null) | $(docker logs $CN 2>&1 | grep -cE '^E ') E lines"
  down
  xlog "$tag: done ($(elapsed) min)"
}
trap 'down' EXIT
trap 'exit 143' TERM INT
# ---------------------------------------------------------------- main
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) | one NST 5 start: $DESC | cells l [$CELLS_L] c8 [$CELLS_C8] n [$CELLS_N], ABBA, budget $BUDGET min | alone probe / newcomers / c8 burst | no evict | estimate ~45 min"
IMGLITS="B|sched_interleave
B|sched_noa
B|sched_noa takes 0 (off), 1
B|pf_steps_4plus
B|pf_steps_1
B|sched_noa, epoch_bump)
B|draft_probe
B|sched_short_bypass
B|sched_short_bypass takes 0 (off)
B|sched_bypass_total
B|plan_mismatch
B|sched_mixed_tok
B|sched_bi_chunk
B|sched_interleave takes 0 .. 64
B|sched_mixed_tok takes 0 (back to the shared budget
B|RADIANCE_SCHED_BI_CHUNK
B|RADIANCE_LOG_STEPS
B|ds41_psk
B|reset_prefix_cache
B|radiance:engine_steps_mixed_total
B|radiance:engine_step_seconds_decode_total
B|radiance:draft_accepted_total
A|RADIANCE_DS41_PSK
A|lever PSK declared (RADIANCE_DS41_PSK, fn4 ds41_psk, now
A|RADIANCE_DS41_KVFOLD
K|RADIANCE_DS41_GG_PF_TABLE
A|RADIANCE_DS41_DSPARK_WALK
K|dspark_walk_step
A|RADIANCE_DS41_ARENA_DECODE_ROWS
A|lever BI declared (RADIANCE_DS41_BI, fn4 ds41_bi, now
B|RADIANCE_DRAFT_SAMPLE
K|dspark_walk_sample_step
A|DSpark lever SD: the sampled full-vocabulary walk declared
S|mtp_couple_verify
B|RADIANCE_WEIGHT_SIDECAR
B|read from the weight sidecar
B|draft_occ_k takes 1 .. 4096
B|RADIANCE_DRAFT_LEND_WEIGHTS
B|lever Z0-A set up
B|RADIANCE_DRAFT_CTXSKIP_SLOTS"
docker image inspect $IMAGE > /dev/null 2>&1 || { xlog "VIRHE image $IMAGE not on this host"; exit 1; }
IMGMISS=$(echo "$IMGLITS" | timeout 120 docker run --rm -i --network none --entrypoint sh $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels/libr4d.so; S=/opt/radiance/share/radiance/kernels/spechub.so; n=0
while IFS= read -r l; do [ -n "$l" ] || continue; n=$((n + 1)); w=${l%%|*}; s=${l#*|}
  case $w in B) f=$B ;; A) f=$A ;; K) f=$K ;; S) f=$S ;; *) f="$B $A $K" ;; esac
  o=0; for x in $f; do grep -qzF -- "$s" $x && { o=1; break; }; done
  [ $o = 1 ] || echo "MISS $w|$s"; done
test -x /opt/radiance/bin/rad-sidecar || echo "MISS bin/rad-sidecar"; echo "CHECKED $n"' 2>&1)
case "$IMGMISS" in "CHECKED "*) ;; *) xlog "VIRHE $IMAGE is not an image with lever NOA + DST + PSK + BI chunking -- missing: $(echo "$IMGMISS" | grep -v '^CHECKED' | tr '\n' ';' | cut -c1-900)"; exit 1 ;; esac
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)) carries lever NOA (sched_noa, GET \"noa\"), DST, sched_bi_chunk, ds41_psk and the int5 levers: all $(echo "$IMGMISS" | sed 's/CHECKED //') literals present + bin/rad-sidecar -- OK"
HMISS=""
for f in "$LIB29" "$LIBPR" $Y $GS $DQ $PL $GD $MBT $MBP $TESTS $D/fastcheck.py; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $GS $DQ $PL $GD $MBT $MBP $TESTS $D/fastcheck.py 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
ST=$(timeout 60 python3 -I $PL selftest --tests $TESTS --mb $MBP 2>&1 | tail -1); case "$ST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$PL(${ST:0:300})" ;; esac
GST=$(timeout 60 python3 -I $GD selftest 2>&1 | tail -1); case "$GST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$GD(${GST:0:300})" ;; esac
MST=$(timeout 60 python3 -I -B $MBT --mb $MBP --selftest 2>&1 | tail -1); case "$MST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$MBT(${MST:0:300})" ;; esac
DQP=$(timeout 60 python3 -I -c 'import importlib.util, json, sys
def ld(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
t = ld(sys.argv[2], "t"); pl = ld(sys.argv[1], "pl")
import ast
src = open(sys.argv[3]).read()
task = next((ast.literal_eval(n.value) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "task" for t in n.targets)), None)
ok = "CPT, OFF = 3.95, 1200" in src and "# deepq fixed prompt v1" in src and pl.DEEP_CPT == 3.95 and pl.DEEP_OFF == 1200 and task == pl.DEEP_TASK
d = pl.deep_payload(t, 64000, 128)
print("DQ OK" if ok and d["messages"][0]["content"].startswith("# deepq fixed prompt v1\n") and d["max_tokens"] == 128 else "DQ MISMATCH")' $PL $TESTS $DQ 2>&1 | tail -1)
case "$DQP" in "DQ OK") ;; *) HMISS="$HMISS deepq-vs-dstld2:(${DQP:0:200})" ;; esac
FNS="log run mkd restart check down stopfile best_vars desc serve_args xlog cellj cellbic strict pinj stackenv keeplog thermline cmpfile idrec identw modes vramstart fn4get fn4set nudge fn4code pcreset arm msnap mdelta pskstat gate1 lslice chunkline swnote smoke snap loglines pfsteps guard newcid idrec2 drun c8run alones elapsed plan summary dstart"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee sha256sum gzip tac; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
for c in A N Y NY Z; do python3 -c 'import json,sys; d=json.loads(sys.argv[1]); assert d["draft_depth0"] == 2 and d["ds41_psk"] >= 1 and "sched_noa" in d' "$(pinj "$(cellj $c)")" 2>/dev/null || HMISS="$HMISS json:$c"; done
[ "$(sha256sum $MODELDIR/$SIDECAR 2>/dev/null | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256 2>/dev/null)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR"
[ "$(sha256sum $MODELDIR/$SIDE 2>/dev/null | cut -c1-64)" = "$SIDESHA" ] || HMISS="$HMISS sha256:$MODELDIR/$SIDE(want ${SIDESHA:0:16})"
grep -q 'XS2_ENVFILE' $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no env_file)"
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config refuses it)"
HMISS=$(echo $HMISS)
[ -z "$HMISS" ] || { xlog "VIRHE helpers / inputs do not resolve: $HMISS"; exit 1; }
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing, dstguard2.py $GST, dstld2.py $ST, mbt.py $MST, deepq.py = dstld2.py's deep request ($DQP), all $(echo $FNS | wc -w) functions resolve, tools on PATH, every cell's pin parses (depth 0 forced, PSK on, sched_noa named), sidecars $SIDECAR + $SIDE sha256 OK, yaml accepted -- OK"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { xlog "VIRHE $M225 missing or not verified"; exit 1; }
  [ -s $WARM_REF_225 ] || xlog "note -- reference $WARM_REF_225 missing (not compared)"
  xlog "$ITEM: helpers md5: dsm-dst2.sh $(md5sum < $0 | cut -c1-8), dstguard2.py $(md5sum < $GD | cut -c1-8), dstld2.py $(md5sum < $PL | cut -c1-8), mbt.py $(md5sum < $MBT | cut -c1-8), deepq.py $(md5sum < $DQ | cut -c1-8), greedyset.py $(md5sum < $GS | cut -c1-8), mb.py $(md5sum < $MBP | cut -c1-8)"
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDE $WARM_REF_225 $RS3/s3-n5-stats0.json; do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  for c in A N Y NY Z; do echo "+ pin $c: $(pinj "$(cellj $c)")"; done
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(information here; the real run refuses to start beside one)"
fi
mkd $O $O/guard $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt
stopfile || dstart
summary
unset $SWITCHES
down
xlog "$ITEM: done -- $O ($(elapsed) min)"
