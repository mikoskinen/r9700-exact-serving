#!/bin/bash
# dsm-stack4.sh (11.11.2026, queue-ar/dsmain/stack4, ds-main-port stack3 builder) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r00zzz-b-stack4.sh. Log tag "RESULT xs4 ", out runs/xdsm/stack4 (XS4_OUT).
# THE DS STACK CONFIRMATION 4 on the ds41-int7 image (ds41-int5 283c7a1 (int3 + BI + DLW + SD + Z0) + ds41-lf d583904 (lever LF:
# fn4 place_lf) + ds41-dst b890cee (lever DST + sched_bi_chunk: left unset here -- the stack as designed) + ds41-psk2 at the commit
# r00zz tests (lever PSK design 2 + the indexer fix: fn4 ds41_psk 0 / 1 / 2 / 3, ds41_psk_k, RADIANCE_DS41_PSK); every switch unset
# = int5). ONE NST 5 START, everything in process: stack3's S start (stack2's NST 5 env + RADIANCE_DS41_BI=0,
# RADIANCE_DRAFT_SAMPLE=0, --draft-coupling on, the DLW q8_0 sidecar) + z0occ2's Z0 start words (rungs 8,48, ring 129,
# RADIANCE_DRAFT_LEND_WEIGHTS=0) + RADIANCE_DS41_PSK=0 (PSK declared, off at start). Fixed pin = stack3's NST 5 pin (draft_hybrid 1)
# + the Z0 switches 0 + draft_occ_k XS4_OCCK (32).
# ARMS (each = the fixed pin + its own switches, applied in process with an epoch bump):
#   A  base = stack3's arm A: ds41_bi 0, draft_sample 0, draft_occ 0, place_lf 0, ds41_psk 0 (the Z0 start words stay: start-time)
#   S  the stack: ds41_bi 2, draft_occ XS4_OCC (6), ds41_psk XS4_PSK (2 = design 2 at ds41_psk_k XS4_PSKK, r00zz's K; 1 = design 1
#      if design 2 fails identity and a fixed design 1 passed; 0 = no PSK), place_lf by XS4_LF (r00zzz-a-lf's verdict: 1 = LF in S,
#      0 = LF out; sl = its verdict not in: S has place_lf 0 and the SL arm below measures LF in the stack), draft_sample XS4_DSA
#      (85) -- at T 0 draft_sample does nothing to a greedy request, so S is the one stack arm there
#   8  = S at T 1 (S85, draft_sample XS4_DSA)     7  = S with draft_sample XS4_DSB (70) (S70)
#   L  = S85 + place_lf 1 (SL; only under XS4_LF=sl)
# Per start, in this order (decode before prefill):
#   gate1 -> modes (z0occ2's literals + lever PSK declared) -> VRAM at start -> switch note (start values; ds41_psk 2 / place_lf 2 /
#   draft_occ 65 refused; every arm applied at a step) -> warm-up answers vs stock -> smoke under 8 (greedy set conc 4 + mb c4 20 s
#   at T 1: every lever of the stack on) -> GUARDS (coordinator 11.11.: no silent lever): PSK >= 1 -- a cold ~XS4_GDEP (16000) deepq
#   prompt under A then S: a 'lever PSK: N cut steps' line, and under design 2 S's staged bytes <= 0.9 x A's and fewer snapshots
#   (/stats, r00zz's deltas); LF (XS4_LF 1 / sl) -- mb c8 30 s under the LF arm: experts.ranks[].lent.*.fills_lf > 0. A failed
#   guard: VIRHE, no speed reading for the arms that carry the lever (identity still runs) ->
#   T 0 BLOCKS XS4_ORDER0 (A S S A A S S A; XS4_LF=sl: A S L L S A A S L L S A; 4 a arm, mirrored): stepprobe essay / code + editprobe (T 0), mb c1 / c4 / c8 / pf at
#     temperature 0 (mbt.py), dur XS4_DUR 60 warm 15; per phase VRAM / lent units, depth-0 share (phase and 2 s), occ, SD counters
#   T 1 BLOCKS XS4_ORDER1 (A 7 8 8 7 A A 7 8 8 7 A; sl: A 7 8 L L 8 7 A A 7 8 L L 8 7 A; adjacent pairs A-7, 7-8, 8-L): sdprobe essay / code /
#     edit at the server default (temperature 1, seed 1234, thinking on), mb c4 / c8 at the server default
#   PREFILL A vs S in process, XS4_PORDER (ABBA = A S S A): cold TTFT at 136k and 64k (deepq.py's fixed prompt, a prefix-cache reset
#     before each, 512 greedy tokens; acceptance, non-decode step ms) -> the c4 + cold 64k decode-under-prefill reading, XS4_LORDER
#     (A S S A; pskld.py --load: mixed-step ms, c4 tok/s and per-stream p50 in the prefill window, TPOT, the probe's TTFT)
#   LOAD IDENTITY at T 0 under S: a1-A and a1-S (greedy set c1 alone) vs bz d0; lid-S (each greedy prompt under mb c4 + one fresh
#     ~32k prefill beside its decode: PSK cuts that prefill's rows, BI keeps the probe's decode rows their own segment) vs a1-S;
#     c4a-S / c4b-S (run-to-run) and c4a-S vs a1-S; a2-S (alone again); the 28k copy under S vs bz d0; every 136k cold answer (A and
#     S) vs dsm-225 and S vs A at each depth; the 64k answer under c4 load vs alone, same arm (a long prefill beside decode)
#     -- at sched_bi_chunk 0 (stack4 leaves it unset) a KNOWN class (ii) difference (2048-row mixed chunks vs alone's 4096),
#     labelled in the summary, not a failure (fixed by D3, r04); every other compare strict
# SUMMARY: per temperature, phase and arm mean +- t95; every arm vs A (pooled: the mirrored order cancels a linear drift; pairs
#   where adjacent: S-A (L-S) at T 0, 7-A / 8-7 (L-8) at T 1); S85 vs S70; SL vs S85 under sl; the targets (1u >= 90, c4 >= 160, c8 >= 200,
#   136k <= 35 s); cross-start references stack3 (n5 T 0 arm P, n5 T 1 arm S, n0 T 0 arm A) and z0occ2 (arm 6), labelled; the
#   prefill A / S table; the identity verdict.
# ~110 min (XS4_LF=sl: ~141), NO EVICT (the A / S prefill comparison is in process; the historical 136k 50.03 / 51.33 s are cross-start, page-cache
#   warm references, labelled): start + checks ~7, T 0 8 blocks x ~5 (stack3's n5 blocks: 298-305 s), T 1 12 x ~3, prefill ~13,
#   identity ~13. Cut XS4_EDIT=0 first (the 1u edit at both temperatures, ~-7 min) if a shorter run is wanted.
# NO cc_sim, no 2.9; no power / host / fan settings; production compose files read only. Exact: BI / LF / draft_occ / PSK by design
# (PSK's bytes are measured here and in r00z); SD at T > 0 has the user's GO (distribution-exact; at T 0 byte-identical).
# Knobs: XS4_IMAGE, XS4_ITEM, XS4_OUT, XS4_PSK (2 | 1 | 0: PSK in the S arms), XS4_PSKK (ds41_psk_k under 2), XS4_LF (0 | 1: place_lf
#   in the S arms; sl = the SL arm), XS4_OCC (6), XS4_OCCK (32), XS4_DSA (85), XS4_DSB (70),
#   XS4_ORDER0, XS4_ORDER1, XS4_DUR (60), XS4_WARM (15), XS4_EDIT (1 | 0), XS4_GDEP (16000: the PSK guard's prompt), XS4_DEPTHS ("135000 64000"), XS4_PORDER (ASSA),
#   XS4_LORDER (ASSA), XS4_LDEPTH (64000), XS4_SIDECAR, XS4_SIDESHA.
# DRYRUN=1 bash dsm-stack4.sh : prints every command; checks the image literals (the registry with place_lf + ds41_psk (+ ds41_psk_k
#   under XS4_PSK=2), BI / SD / DLW / Z0 / LF / PSK), every helper (files, parse, mbt.py / loadid.py / pskld.py selftests, s4blocks.py / sdprobe.py --help, deepq.py = pskld.py's
#   deep request), the sidecars, the yaml; no server, no GPU, writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/../ds41-stack2-tp4.yaml                   # stack2's yaml, read only (env_file ${XS2_ENVFILE} carries every start word)
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh
O=${XS4_OUT:-$R/xdsm/stack4}; export STATSDIR=$O/stats
RD=$R/xdsm/draft; RM=$R/xdsm; RS3=$R/xdsm/stack3; RZ2=$R/xdsm/z0occ2
export IMAGE=${XS4_IMAGE:?set XS4_IMAGE}
ITEM=${XS4_ITEM:-57-p4zyzzzzz-ds-r00zzz-b-stack4}
PSK=${XS4_PSK:-2}; PSKK=${XS4_PSKK:-}; LF=${XS4_LF:-0}; OCC=${XS4_OCC:-6}; OCCK=${XS4_OCCK:-32}; DSA=${XS4_DSA:-85}; DSB=${XS4_DSB:-70}
[ "${XS4_LF:-0}" = sl ] && { DO0=ASLLSAASLLSA; DO1=A78LL87AA78LL87A; } || { DO0=ASSAASSA; DO1=A7887AA7887A; }
ORDER0=${XS4_ORDER0-$DO0}; ORDER1=${XS4_ORDER1-$DO1}; DUR=${XS4_DUR:-60}; WARMS=${XS4_WARM:-15}; EDIT=${XS4_EDIT:-1}
GDEP=${XS4_GDEP:-16000}; DEPTHS=${XS4_DEPTHS:-135000 64000}; PORDER=${XS4_PORDER:-ASSA}; LORDER=${XS4_LORDER-ASSA}; LDEPTH=${XS4_LDEPTH:-64000}; NTOK=512
SIDE=${XS4_SIDECAR:-dspark-dlw-q8_0.rad}; SIDESHA=${XS4_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
OFF=0.5; PFT=32768; CONC=4; WARM=10; MT=192; SEED=1234
case $PSK in 0|1|2) ;; *) echo "XS4_PSK must be 0, 1 or 2 (got '$PSK')"; exit 1 ;; esac
[ $PSK = 2 ] && case $PSKK in ''|*[!0-9]*) echo "XS4_PSK=2 needs XS4_PSKK (ds41_psk_k, r00zz's K; got '$PSKK')"; exit 1 ;; esac
case $LF in 0|1|sl) ;; *) echo "XS4_LF must be 0, 1 or sl (got '$LF')"; exit 1 ;; esac
LFS=$([ $LF = 1 ] && echo 1 || echo 0)   # place_lf in the S arms
case $EDIT in 0|1) ;; *) echo "XS4_EDIT must be 0 or 1 (got '$EDIT')"; exit 1 ;; esac
case $ORDER0 in *[!ASL]*) echo "XS4_ORDER0 takes A / S (/ L) blocks (got '$ORDER0')"; exit 1 ;; esac
case $ORDER1 in *[!A87L]*) echo "XS4_ORDER1 takes A / 8 / 7 (/ L) blocks (got '$ORDER1')"; exit 1 ;; esac
[ $LF = sl ] || case "$ORDER0$ORDER1" in *L*) echo "the L (SL) arm runs only under XS4_LF=sl"; exit 1 ;; esac
case $PORDER in *[!AS]*|"") echo "XS4_PORDER takes A / S (got '$PORDER')"; exit 1 ;; esac
case $LORDER in *[!AS]*) echo "XS4_LORDER takes A / S or nothing (got '$LORDER')"; exit 1 ;; esac
for v in $OCC $OCCK $DSA $DSB $DUR $WARMS $LDEPTH $GDEP $DEPTHS; do case $v in ''|*[!0-9]*) echo "a numeric knob is not a number ('$v')"; exit 1 ;; esac; done
case " $DEPTHS " in *" $LDEPTH "*) ;; *) [ -z "$LORDER" ] || { echo "XS4_LDEPTH ($LDEPTH) must be one of XS4_DEPTHS ($DEPTHS): the loaded answer is compared with the alone one"; exit 1; } ;; esac
PH0=c1,$([ $EDIT = 1 ] && echo edit,)mbc1,c4,c8,pf; T1P=essay,code$([ $EDIT = 1 ] && echo ,edit)
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
LD=$HD/loadid.py; GGB=$HD/s4blocks.py; MBT=$HD/mbt.py; PRB=$HD/sdprobe.py; EP=$HD/editprobe.py; PL=$HD/pskld.py
DQ=$QA/deepq.py; MBP=$D/mb.py; TESTS=$D/ds41_tests.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
xlog() { log "RESULT xs4 $*"; }
# ---------------------------------------------------------------- the start (= dsm-psk.sh's: z0occ2's start + RADIANCE_DS41_PSK=0)
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
LITS="$LITS|lever,PSK,declared,(RADIANCE_DS41_PSK"   # the declare line's rest is design-specific (1 vs 2): checked by the PSK items
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": 8, "drafter_mx": 1, "gg_pf": 1, "f32_pf": 25, "dspark_walk": 1, "ds41_argather": 48, "idx_split": 1, "ds41_qnfold": 1, "ds41_swfold": 1, "draft_ctxskip": 1, "dspark_conf": 1, "dspark_conf_rows": 1, "draft_lend": 1, "draft_probe_backoff": 3, "draft_hybrid": 1}'
FIX=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update({"draft_lend_hyst": 0, "arena_d0": 0, "draft_lend_w": 0, "draft_occ_k": int(sys.argv[2])}); print(json.dumps(d))' "$FIX5" "$OCCK") || { echo "pin JSON"; exit 1; }
PSKJ="\"ds41_psk\": $PSK$([ $PSK = 2 ] && echo ", \"ds41_psk_k\": $PSKK")"
armj() {   # armj A|S|8|7|L: the arm's own switches (JSON)
  case $1 in
    A) echo '{"ds41_bi": 0, "draft_sample": 0, "draft_occ": 0, "place_lf": 0, "ds41_psk": 0}' ;;
    S|8) echo "{\"ds41_bi\": 2, \"draft_sample\": $DSA, \"draft_occ\": $OCC, \"place_lf\": $LFS, $PSKJ}" ;;
    7) echo "{\"ds41_bi\": 2, \"draft_sample\": $DSB, \"draft_occ\": $OCC, \"place_lf\": $LFS, $PSKJ}" ;;
    L) echo "{\"ds41_bi\": 2, \"draft_sample\": $DSA, \"draft_occ\": $OCC, \"place_lf\": 1, $PSKJ}" ;;
  esac
}
pinj() {   # pinj ARM [EXTRAJSON]: the fixed pin + the arm + extra as one POST body
  local x=${2:-}; [ -n "$x" ] || x='{}'
  python3 -c 'import json,sys
d = {}
for a in sys.argv[1:]: d.update(json.loads(a))
print(json.dumps(d))' "$FIX" "$(armj $1)" "$x"
}
DESC="NST 5 stack4 start (stack3 S: stack2 NST 5 env, DSpark adaptive, depth 0, hybrid 1, W1, small n5 set, drafter_mx, lend + back-off 3, VRAM levers, --draft-coupling on, DLW sidecar $SIDE; z0occ2's ring 129, Z0-W declared, rungs 8,48; PSK declared off) | fixed pin: stack3's + Z0 switches 0, draft_occ_k $OCCK | S: ds41_bi 2, draft_occ $OCC, ds41_psk $PSK$([ $PSK = 2 ] && echo " (k $PSKK)"), place_lf $LFS, draft_sample $DSA (S70: $DSB)$([ $LF = sl ] && echo ", SL = S85 + place_lf 1")"
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
  RADIANCE_DS41_PSK RADIANCE_PLACE_LF RADIANCE_PLACE_LF_DECAY"
# ---------------------------------------------------------------- copied from dsm-psk.sh (md5 841ce920; tag xpsk -> xs4, ORDER -> PORDER, arm B -> S)
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xs4 $1 warm-up identity vs $2"; return 0; }
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
  for kv in RADIANCE_PLACE_TURNOVER=0 $LEVENV; do echo "$CE" | grep -qxF "$kv" || bad="$bad ${kv%%=*}"; done
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
  [ "$DRY" = 1 ] && { echo "+ GET /stats > $O/$1-stats0.json; RESULT xs4 $1 VRAM at start (experts per tier; vs stack3 s3-n5)"; return 0; }
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
arm() {   # arm A|S|8|7|L: the pin + the arm + epoch_bump, a nudge to apply it at a step
  fn4set "$(pinj $1 '{"epoch_bump": 1}')"; nudge
}
msnap() { [ "$DRY" = 1 ] && { echo "+ GET /metrics > $O/m-$1.txt"; return 0; }; curl -s -m 20 $URL/metrics > $O/m-$1.txt 2>/dev/null; }
mdelta() {   # mdelta A B -> JSON: draft acceptance, non-decode steps and their mean ms, decode steps, prefill tokens over the window
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
print(json.dumps(dict(acc=round(da / dt, 4) if dt > 0 else None, draft_tok=int(dt), draft_acc=int(da),
      nondecode_steps=int(nd), mixed_steps=int(g("engine_steps_mixed_total")), nondecode_ms=round(1000 * nds / nd, 1) if nd > 0 else None,
      decode_steps=int(g("engine_steps_decode_total")), prefill_tok=int(g("prefill_tokens_total")), decode_tok=int(g("decode_tokens_total")))))' $O/m-$1.txt $O/m-$2.txt 2>/dev/null || echo '{}'
}
pskstat() { [ "$DRY" = 1 ] && { echo "(dry)"; return 0; }; echo "$(docker logs $CN 2>&1 | grep 'lever PSK:' | tail -1 | sed 's/^radiance: deepseek41: //' | cut -c1-300) | $(docker logs $CN 2>&1 | grep 'lever PSK design 2: [0-9]* planned' | tail -1 | sed 's/^I //' | cut -c1-300) | VIRHE lines $(docker logs $CN 2>&1 | grep -c 'lever PSK design 2: VIRHE')"; }
# ssnap / sdelta: r00zz's (dsm-psk2.sh 6ca0480a, copied) -- the /stats deltas a cold prefill leaves
ssnap() { [ "$DRY" = 1 ] && { echo "+ GET /stats > $O/st-$1.json"; return 0; }; curl -s -m 30 $URL/stats > $O/st-$1.json 2>/dev/null; }
sdelta() {   # sdelta A B -> JSON: checkpoints written, staged bytes (all ranks / rank 0), rank-0 staged layers, all-reduce bytes
  [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }
  python3 -c 'import json,sys
def rd(p):
    try: return json.load(open(p))
    except Exception: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
g = lambda d, *k: (lambda v: v if isinstance(v, (int, float)) else 0)(__import__("functools").reduce(lambda x, y: (x or {}).get(y) if isinstance(x, dict) else None, k, d))
print(json.dumps(dict(ckpts=int(g(b, "checkpoints_written") - g(a, "checkpoints_written")),
      staged_gb=round((g(b, "link", "staged_bytes") - g(a, "link", "staged_bytes")) / 1e9, 2),
      staged_gb_r0=round((g(b, "experts", "staged_bytes") - g(a, "experts", "staged_bytes")) / 1e9, 2),
      stage_layers_r0=int(g(b, "experts", "stage_layers") - g(a, "experts", "stage_layers")),
      ar_gb=round((g(b, "link", "ar_bytes") - g(a, "link", "ar_bytes")) / 1e9, 2))))' $O/st-$1.json $O/st-$2.json 2>/dev/null || echo '{}'
}
lffills() {   # lffills A B -> the lent slots' LF fills (sum over ranks and lenders of experts.ranks[].lent.*.fills_lf) between two /stats
  [ "$DRY" = 1 ] && { echo 0; return 0; }
  python3 -c 'import json,sys
def rd(p):
    try: return json.load(open(p))
    except Exception: return {}
def tot(d):
    return sum((l.get("fills_lf") or 0) for r in (d.get("experts") or {}).get("ranks") or [] for l in ((r.get("lent") or {}).values()) if isinstance(l, dict))
print(int(tot(rd(sys.argv[2])) - tot(rd(sys.argv[1]))))' $O/st-$1.json $O/st-$2.json 2>/dev/null || echo 0
}
gate1() {   # restart + ONE check; over the limit logged (in-process A/B: a slow state shows in every arm)
  local tag=$1 lim=$2 ms st=fast
  restart $tag || { keeplog $tag-fail; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && return 0
  awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}' && { st="SLOW (not retried)"; keeplog $tag-slow; }
  echo "$st $ms" > $O/guard-$tag.txt
  xlog "guard $tag: $st, 1-user ${ms} ms/step (limit $lim; stack3 s3-n5 $(cat $RS3/guard-s3-n5.txt 2>/dev/null))"
}
deepx() {   # deepx TAG ARM DEPTH NAME [noreset]: deepq.py's fixed prompt at DEPTH, 512 greedy tokens -> TTFT, cached, acceptance, ms
  local tag=$1 a=$2 dep=$3 nm=$4 t0=$(date +%s) j md
  arm $a
  [ "${5:-}" = noreset ] || pcreset $tag-$nm
  msnap $tag-$nm-0; ssnap $tag-$nm-0
  run "(cd $D && timeout 1800 python3 $DQ $O/$tag-$nm.txt $dep $NTOK > $O/$tag-$nm.out 2>&1)"
  msnap $tag-$nm-1; ssnap $tag-$nm-1
  [ "$DRY" = 1 ] && { echo "+ RESULT xs4 $tag $nm: arm $a depth $dep: TTFT, cached, acceptance, non-decode step ms, /stats deltas (snapshots, staged GB, all-reduce GB) -> $O/deep.jsonl"; return 0; }
  j=$(grep -m1 '^DEEPQ ' $O/$tag-$nm.out | cut -c7-)
  md=$(python3 -c 'import json,sys
d = json.loads(sys.argv[1] or "{}"); d.update(json.loads(sys.argv[2] or "{}")); print(json.dumps(d))' "$(mdelta $tag-$nm-0 $tag-$nm-1)" "$(sdelta $tag-$nm-0 $tag-$nm-1)" 2>/dev/null || mdelta $tag-$nm-0 $tag-$nm-1)
  [ -n "$j" ] || { xlog "$tag $nm: VIRHE no answer ($(tail -2 $O/$tag-$nm.out | tr '\n' ' ' | cut -c1-300))"; return 1; }
  python3 -c 'import json,sys
d = json.loads(sys.argv[1]); d.update(json.loads(sys.argv[2]) if sys.argv[2].strip() else {})
d.update(name=sys.argv[3], arm=sys.argv[4], depth=int(sys.argv[5]), reset=sys.argv[6] != "noreset", text=sys.argv[7])
print(json.dumps(d))' "$j" "$md" "$nm" "$a" "$dep" "${5:-reset}" "$O/$tag-$nm.txt" >> $O/deep.jsonl
  xlog "$tag $nm (arm $a $(armj $a), deepq depth $dep, $NTOK greedy tokens$([ "${5:-}" = noreset ] && echo ", NO reset")): $(python3 -c 'import json,sys
d, m = json.loads(sys.argv[1]), json.loads(sys.argv[2] or "{}")
print("status %s, prompt %s tok, cached %s, TTFT %s s (%s prefill tok/s), %s tok, %s chars | decode acceptance %s (%s of %s drafted) | non-decode steps %s, %s ms mean | snapshots %s, staged %s GB (rank 0 %s GB, %s layers), all-reduce %s GB" % (
    d.get("status"), d.get("prompt_tokens"), d.get("cached"), d.get("ttft"),
    round((d["prompt_tokens"] - (d.get("cached") or 0)) / d["ttft"], 1) if d.get("ttft") and d.get("prompt_tokens") else "-",
    d.get("tokens"), d.get("chars"), m.get("acc"), m.get("draft_acc"), m.get("draft_tok"), m.get("nondecode_steps"), m.get("nondecode_ms"),
    m.get("ckpts"), m.get("staged_gb"), m.get("staged_gb_r0"), m.get("stage_layers_r0"), m.get("ar_gb")))' "$j" "$md" 2>&1 | cut -c1-600) | psk: $(pskstat) | $(thermline $t0 $(date +%s))"
}
coldabba() {   # coldabba TAG: per depth, the arms in ORDER, each cold; identity: 136k vs dsm-225, B vs the first A of its depth
  local tag=$1 dep a k fa
  for dep in $DEPTHS; do
    k=0; fa=""
    for a in $(echo $PORDER | grep -o .); do
      k=$((k + 1)); stopfile && return 0
      deepx $tag $a $dep d$dep-$k$a
      [ "$DRY" = 1 ] && continue
      [ -n "$fa" ] || { [ $a = A ] && fa=$O/$tag-d$dep-$k$a.txt; }
      [ $dep = 135000 ] && idrec "deep136-$k$a-vs-dsm225" "$(cmpfile $O/$tag-d$dep-$k$a.txt $RM/deep-dsm-225.txt)" "136k answer vs dsm-225 NST 0 stock cold"
      [ -n "$fa" ] && [ "$O/$tag-d$dep-$k$a.txt" != "$fa" ] && idrec "d$dep-$k$a-vs-A" "$(cmpfile $fa $O/$tag-d$dep-$k$a.txt)" "depth $dep, vs the first A"
    done
  done
}
copyx() {   # copyx TAG ARM: the 28k copy (dsm-stack2.sh copyt's request) after a reset -> vs bz d0
  local tag=$1 t0=$(date +%s)
  arm $2; pcreset $tag-copy$2
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset 0 --tag=-$tag-$2 --outdir $O > $O/$tag-copy$2.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  xlog "$tag copy arm $2 (after a cold ~31k prompt): $(grep '^J ' $O/$tag-copy$2.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300) | $(thermline $t0 $(date +%s))"
  idrec "copy-$2-vs-d0" "$(cmpfile $O/copy-$tag-$2-run1.txt $RD/copy-dsd-d0-i1-run1.txt)" "28k copy vs bz d0 (NST 0 stock)"
}
loadx() {   # loadx TAG ARM NAME: c4 decode streams + ONE cold LDEPTH deepq prompt (pskld.py --load, no extra prefill)
  local tag=$1 a=$2 nm=$3 t0=$(date +%s) j md
  arm $a; pcreset $tag-$nm
  msnap $tag-$nm-0
  run "(cd $D && timeout 2700 python3 -I $PL run --url $URL --tests $TESTS --mb $MBP --which deep --depth $LDEPTH --ntok $NTOK --load --conc 4 --warm 10 --pf-tok 0 --out $O/$tag-$nm.json --text $O/$tag-$nm.txt > $O/$tag-$nm.log 2>&1)"
  msnap $tag-$nm-1
  [ "$DRY" = 1 ] && { echo "+ RESULT xs4 $tag $nm: arm $a: mixed-step ms, c4 tok/s in the prefill window (agg / per-stream p50), TPOT, TTFT -> $O/load.jsonl"; return 0; }
  j=$(grep -m1 '^LONGID ' $O/$tag-$nm.log | cut -c8-); md=$(mdelta $tag-$nm-0 $tag-$nm-1)
  [ -n "$j" ] || { xlog "$tag $nm: VIRHE no LONGID line ($(tail -3 $O/$tag-$nm.log | tr '\n' ' ' | cut -c1-400))"; return 1; }
  python3 -c 'import json,sys
d = json.loads(sys.argv[1]); d.update(json.loads(sys.argv[2]) if sys.argv[2].strip() else {}); d.update(name=sys.argv[3], arm=sys.argv[4], text=sys.argv[5])
print(json.dumps(d))' "$j" "$md" "$nm" "$a" "$O/$tag-$nm.txt" >> $O/load.jsonl
  xlog "$tag $nm (arm $a, c4 decode + one cold ~$LDEPTH-depth deepq prompt, T 0): $(python3 -c 'import json,sys
d, m = json.loads(sys.argv[1]), json.loads(sys.argv[2] or "{}")
print("probe status %s, prompt %s tok, cached %s, TTFT %s s | prefill window %s s: c4 %s tok, agg %s tok/s, per-stream p50 %s tok/s, TPOT p50 %s / p90 %s / max %s ms | mixed steps %s (non-decode %s at %s ms mean) | c4 %s requests, %s errors" % (
    d.get("status"), d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("pf_win_s"), d.get("c4_pf_tokens"), d.get("c4_pf_agg_tps"),
    d.get("c4_pf_stream_tps_p50"), d.get("tpot_pf_p50"), d.get("tpot_pf_p90"), d.get("tpot_pf_max"), m.get("mixed_steps"), m.get("nondecode_steps"),
    m.get("nondecode_ms"), d.get("c4_requests"), d.get("c4_errors")))' "$j" "$md" 2>&1 | cut -c1-700) | psk: $(pskstat) | $(thermline $t0 $(date +%s))"
}
loadabba() {   # loadabba TAG: the arms in LORDER; identity: each loaded 64k answer vs the same arm's alone 64k answer and vs A
  local tag=$1 a k al fa=""
  [ -n "$LORDER" ] || return 0
  k=0
  for a in $(echo $LORDER | grep -o .); do
    k=$((k + 1)); stopfile && return 0
    loadx $tag $a ld-$k$a
    [ "$DRY" = 1 ] && continue
    al=$(ls $O/$tag-d$LDEPTH-*$a.txt 2>/dev/null | head -1)
    [ -n "$al" ] && idrec "ld-$k$a-vs-alone$a" "$(cmpfile $al $O/$tag-ld-$k$a.txt)" "the ~$LDEPTH prompt under c4 load vs alone, same arm"
    [ -n "$fa" ] || { [ $a = A ] && fa=$O/$tag-ld-$k$a.txt; }
    [ -n "$fa" ] && [ "$O/$tag-ld-$k$a.txt" != "$fa" ] && idrec "ld-$k$a-vs-ldA" "$(cmpfile $fa $O/$tag-ld-$k$a.txt)" "loaded, vs the first loaded A"
  done
}
# ---------------------------------------------------------------- stack4's own functions
swnote() {   # swnote TAG: start values; refusals; every arm applied at a step, then A
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 at start (gg_gemv 2, aql_acq 2, K1 three 1, ds41_bi 0, ds41_psk 0, place_lf 0, draft_occ 0, draft_occ_k 32, draft_sample 0 + sample.capable 1, draft_capable 1, epoch 0); refusals ds41_psk 99 / place_lf 2 / draft_occ 65 -> 400; arms A $(pinj A) | S $(armj S) | 7 $(armj 7)$([ $LF = sl ] && echo " | L $(armj L)") applied at a step, then A"; return 0; }
  local s0 g c1 c2 c3 a sets="" bad=""
  s0=$(fn4get)
  c1=$(fn4code '{"ds41_psk": 99}'); c2=$(fn4code '{"place_lf": 2}'); c3=$(fn4code '{"draft_occ": 65}')
  for a in A S 7 $([ $LF = sl ] && echo L) A; do
    arm $a; g=$(fn4get)
    python3 -c 'import json,sys
g, w = json.loads(sys.argv[1]), json.loads(sys.argv[2])
sys.exit(0 if all(g["switches"].get(k) == v for k, v in w.items()) and not g["pending"] else 1)' "$g" "$(pinj $a)" 2>/dev/null || bad="$bad $a"
    sets="$sets $a:$(echo "$g" | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print("bi=%s,ds=%s,occ=%s,lf=%s,psk=%s/%s,ep=%s" % (w.get("ds41_bi"), w.get("draft_sample"), w.get("draft_occ"), w.get("place_lf"), w.get("ds41_psk"), w.get("ds41_psk_k"), d.get("epoch")))' 2>/dev/null)"
  done
  xlog "$1 switch note: at start $(echo "$s0" | cut -c1-1100) | ds41_psk 99 -> HTTP $c1, place_lf 2 -> HTTP $c2, draft_occ 65 -> HTTP $c3 | arms applied:$sets$([ -n "$bad" ] && echo " -- VIRHE not applied:$bad")"
  python3 -c 'import json,sys
s0 = json.loads(sys.argv[1]); w = s0["switches"]
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) and s0["epoch"] == 0
ok = ok and w.get("ds41_bi") == 0 and w.get("ds41_psk") == 0 and w.get("place_lf") == 0 and w.get("draft_occ") == 0 and w.get("draft_occ_k") == 32
ok = ok and w["draft_adaptive"] == 1 and w["draft_depth0"] == 0 and s0["draft_capable"] == 1 and w.get("draft_sample") == 0 and (s0.get("sample") or {}).get("capable") == 1
sys.exit(0 if ok else 1)' "$s0" 2>/dev/null && [ "$c1" = 400 ] && [ "$c2" = 400 ] && [ "$c3" = 400 ] && [ -z "$bad" ]
}
smoke() {   # smoke TAG: under 8 (L under sl: every lever on) -- the greedy set at conc 4 (mixed steps) + mb c4 20 s at T 1 (the coupled path)
  local tag=$1 t0=$(date +%s) n e sa=8
  [ $LF = sl ] && sa=L
  arm $sa
  run "(cd $D && timeout 600 python3 $GS $URL $O/$tag-smoke.jsonl --conc 4 --max-tokens 32 > $O/$tag-smoke.txt 2>&1)"
  run "(cd $D && timeout 600 python3 mb.py --tag xs4-smoke --phases c4 --dur 20 --warm 5 > $O/$tag-smoke-t1.txt 2>&1)"
  [ "$DRY" = 1 ] && { echo "+ smoke $tag: arm $sa $(armj $sa): greedy set conc 4 (32 tok) 20 of 20 + mb c4 20 s at T 1 with 0 errors, /health 200"; return 0; }
  n=$(python3 -c 'import json,sys; print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l).get("text") is not None))' $O/$tag-smoke.jsonl 2>/dev/null || echo 0)
  e=$(grep -m1 '^MB ' $O/$tag-smoke-t1.txt 2>/dev/null | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[3:]); print("%s requests %s errors, %s tok/s" % (d.get("requests"), d.get("errors"), d.get("agg_dec_tps")))' 2>/dev/null || echo "no MB line")
  if [ "$n" = 20 ] && case "$e" in *" 0 errors"*) true ;; *) false ;; esac && curl -s -m 10 -o /dev/null -w '%{http_code}' $URL/health | grep -q 200; then
    xlog "$tag smoke (arm $sa): greedy set conc 4 20 of 20, mb c4 T 1: $e, healthy, $(( $(date +%s) - t0 )) s | psk: $(pskstat)"; return 0; fi
  xlog "$tag smoke (arm $sa): VIRHE $n of 20 answered, mb c4 T 1: $e, or unhealthy | $(docker logs $CN 2>&1 | grep -E '^E |step fail|refus' | tail -3 | tr '\n' ' ' | cut -c1-500)"
  keeplog $tag-smoke; return 1
}
blocks() {   # blocks TAG KIND: t0 = XS4_ORDER0 (mb at temperature 0, stepprobe / editprobe), t1 = XS4_ORDER1 (sdprobe + mb at the server default)
  local tag=$1 k=$2 ord ph mbx="" args a
  if [ $k = t0 ]; then ord=$ORDER0; ph=$PH0; mbx="--mbtemp 0 --mbt $MBT"; else ord=$ORDER1; ph=t1,c4,c8; fi
  [ -n "$ord" ] || return 0
  args=""; for a in $(echo $ord | grep -o . | awk '!s[$0]++'); do args="$args --arm '$a=$(armj $a)'"; done
  run "timeout 10800 python3 -I $GGB --url $URL --mb-dir $D --edit $EP --sdprobe $PRB --seed $SEED --t1-prompts $T1P --label $tag-$k $args --order $ord --phases $ph --dur $DUR --warm $WARMS $mbx --fixed '$FIX' --out $O/$tag-$k-blocks.jsonl > $O/$tag-$k-blocks.txt 2>&1"
  [ "$DRY" = 1 ] && return 0
  while IFS= read -r l; do xlog "$tag $k ${l:0:1700}"; done < <(grep -E '^(BLOCK|T1|OCC|STATS|GGB \{"error)|Traceback' $O/$tag-$k-blocks.txt)
  xlog "$tag $k engine: $(docker logs $CN 2>&1 | grep 'lever BI stats' | tail -1 | sed 's/^radiance: //' | cut -c1-200) | psk: $(pskstat) | draft $(fn4get | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps(d.get("draft")), json.dumps(d.get("sample")))' 2>/dev/null | cut -c1-400) | $(docker logs $CN 2>&1 | grep -cE '^E ') E lines"
}
SPEEDOK=1
guard() {   # guard TAG: before any speed reading, evidence that the stack's silent levers act (coordinator 11.11.): PSK >= 1 must cut,
  # LF (XS4_LF 1 / sl) must fill lent slots. A failed check: VIRHE, and no speed reading for the arms that carry the lever
  # (PSK, or LF under XS4_LF=1: every stack arm -> no blocks / prefill readings at all; LF under sl: the SL arm leaves the orders)
  local tag=$1 r n la
  if [ $PSK -ge 1 ]; then
    stopfile && return 0
    deepx $tag A $GDEP guard-A
    deepx $tag S $GDEP guard-S
    if [ "$DRY" = 1 ]; then
      echo "+ PSK guard: a cold ~${GDEP}-depth prompt under A then S; the live /stats psk counters over S's prefill: $([ $PSK = 2 ] && echo "latched, planned_steps, ckpt_dropped, layers_skipped all grow and plan_mismatch does not" || echo "cut_passes grows") (an image without them: a 'lever PSK: N cut steps' line$([ $PSK = 2 ] && echo ", S's staged bytes <= 0.9 x A's and fewer snapshots")); one guard covers every stack arm (S / 8 / 7 / L share ds41_psk and k); otherwise the guard fails: no speed reading"
    else
      r=$(python3 -c 'import json,re,sys
rows = {}
for l in open(sys.argv[1]):
    d = json.loads(l)
    if d.get("name") in ("guard-A", "guard-S"): rows[d["name"]] = d
a, s, psk, logs = rows.get("guard-A") or {}, rows.get("guard-S") or {}, int(sys.argv[2]), sys.stdin.read()
def st(p):
    try: return json.load(open(p)).get("psk")
    except Exception: return None
p0, p1 = st(sys.argv[3]), st(sys.argv[4])
cuts = max([int(x) for x in re.findall(r"lever PSK: (\d+) cut steps", logs)] or [0])
why = []
if p0 is not None and p1 is not None:
    # the live /stats psk counters (ds41-int7-pskglue): what the first S prefill moved
    dl = {k: (p1.get(k) or 0) - (p0.get(k) or 0) for k in p1 if isinstance(p1.get(k), (int, float))}
    if psk == 2:
        for k in ("latched", "planned_steps", "ckpt_dropped", "layers_skipped"):
            if dl.get(k, 0) <= 0: why.append("%s did not grow (%+d)" % (k, dl.get(k, 0)))
        if dl.get("plan_mismatch", 0) != 0: why.append("plan_mismatch grew (%+d)" % dl["plan_mismatch"])
    elif dl.get("cut_passes", 0) <= 0: why.append("cut_passes did not grow (%+d)" % dl.get("cut_passes", 0))
    ev = "/stats psk deltas " + ", ".join("%s %+d" % (k, v) for k, v in sorted(dl.items()) if v)
else:
    # no live counters in this image: the engine log and the r00zz /stats deltas
    if cuts < 1: why.append("no lever PSK cut step logged")
    if psk == 2:
        sa, ss, ca, cs = a.get("staged_gb"), s.get("staged_gb"), a.get("ckpts"), s.get("ckpts")
        if not (isinstance(sa, (int, float)) and isinstance(ss, (int, float)) and sa > 0 and ss <= 0.9 * sa): why.append("staged %s GB vs A %s GB (want <= 0.9 x)" % (ss, sa))
        if not (isinstance(ca, int) and isinstance(cs, int) and cs < ca): why.append("snapshots %s vs A %s (want fewer)" % (cs, ca))
    ev = "no /stats psk object: log + staged / snapshot evidence"
print(("OK" if not why else "FAIL " + "; ".join(why)) + " | %s | cut steps logged %d; A staged %s GB snapshots %s TTFT %s s; S staged %s GB snapshots %s TTFT %s s" % (
    ev, cuts, a.get("staged_gb"), a.get("ckpts"), a.get("ttft"), s.get("staged_gb"), s.get("ckpts"), s.get("ttft")))' $O/deep.jsonl $PSK $O/st-$tag-guard-S-0.json $O/st-$tag-guard-S-1.json < <(docker logs $CN 2>&1 | grep 'lever PSK:') 2>&1 | cut -c1-700)
      echo "psk $r" >> $O/guard.txt
      case "$r" in OK*) xlog "$tag PSK guard (ds41_psk $PSK$([ $PSK = 2 ] && echo " k $PSKK")): $r" ;;
        *) xlog "$tag PSK guard (ds41_psk $PSK): VIRHE $r -- PSK did not cut: no speed reading for the stack arms"; SPEEDOK=0 ;; esac
    fi
  fi
  if [ $LF = 1 ] || [ $LF = sl ]; then
    stopfile && return 0
    la=$([ $LF = sl ] && echo L || echo S)
    arm $la; ssnap $tag-lf0
    run "(cd $D && timeout 600 python3 mb.py --tag xs4-lfguard --phases c8 --dur 30 --warm 5 > $O/$tag-lfguard.txt 2>&1)"
    ssnap $tag-lf1
    if [ "$DRY" = 1 ]; then echo "+ LF guard: arm $la $(armj $la), mb c8 30 s (draft_occ $OCC holds depth 0 at c8, the slab lent): experts.ranks[].lent.*.fills_lf summed > 0; otherwise the guard fails: $([ $LF = sl ] && echo "the SL arm leaves the orders" || echo "no speed reading")"
    else
      n=$(lffills $tag-lf0 $tag-lf1)
      echo "lf arm $la fills_lf $n" >> $O/guard.txt
      if [ "${n:-0}" -gt 0 ]; then xlog "$tag LF guard (arm $la): OK, $n LF fills in a 30 s c8 window | $(grep -m1 '^MB ' $O/$tag-lfguard.txt | cut -c4- | cut -c1-200)"
      elif [ $LF = sl ]; then xlog "$tag LF guard (arm $la): VIRHE 0 LF fills in a 30 s c8 window -- place_lf did nothing: the SL arm leaves the orders"; ORDER0=${ORDER0//L/}; ORDER1=${ORDER1//L/}
      else xlog "$tag LF guard (arm $la): VIRHE 0 LF fills in a 30 s c8 window -- place_lf did nothing: no speed reading for the stack arms"; SPEEDOK=0; fi
    fi
  fi
}
alone() {   # alone TAG ARM NAME: pin + epoch_bump, prefix-cache reset, greedyset.py's 20 prompts at c1 ALONE, temperature 0
  local tag=$1 t0=$(date +%s)
  arm $2; pcreset $tag-$3
  run "(cd $D && timeout 900 python3 $GS $URL $O/$tag-$3.jsonl --conc 1 --max-tokens $MT > $O/$tag-$3.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  xlog "$tag $3 (greedy set c1 ALONE, temperature 0, $MT tok, arm $2 $(armj $2)): $(tail -1 $O/$tag-$3.txt | cut -c1-200), $(( $(date +%s) - t0 )) s"
}
lidrun() {   # lidrun TAG ARM NAME: loadid.py run -- each greedy prompt under mb c4 + one fresh ~32k prefill beside its decode
  local tag=$1 t0=$(date +%s) j
  arm $2; pcreset $tag-$3
  run "(cd $D && timeout 2700 python3 -I $LD run --url $URL --gs $GS --mb $MBP --out $O/$tag-$3.jsonl --conc $CONC --warm $WARM --offset $OFF --pf-tok $PFT --max-tokens $MT > $O/$tag-$3.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  j=$(grep -m1 '^LOADID ' $O/$tag-$3.txt | cut -c8-)
  [ -n "$j" ] || { xlog "$tag $3: VIRHE no LOADID line ($(tail -3 $O/$tag-$3.txt | tr '\n' ' ' | cut -c1-400))"; return 1; }
  xlog "$tag $3 (each prompt under mb c4 + one fresh ~${PFT}-token prefill at +${OFF} s, temperature 0, arm $2): $(python3 -c 'import json,sys
d = json.loads(sys.argv[1])
print("%s of %s answered, probe %s s mean | prefills %s of %s cold, TTFT %s s mean | in flight %s %% (median) of a probe | mixed steps a probe min %s / median %s; probes without one %s | c4 %s requests, %s errors" % (
    d["answered"], d["of"], d["probe_s_mean"], d["pf_cold"], d["of"], d["pf_ttft_mean"], round(100 * (d["ov_frac_median"] or 0)), d["mixed_min"],
    d["mixed_median"], d["probes_without_mixed"] or "none", d["c4_requests_done"], d["c4_errors"]))' "$j" 2>&1 | cut -c1-700), $(( $(date +%s) - t0 )) s | psk: $(pskstat)"
}
c4rr() {   # c4rr TAG ARM NAME: pin + epoch_bump, prefix-cache reset, greedyset.py's 20 prompts at conc 4
  local tag=$1 t0=$(date +%s)
  arm $2; pcreset $tag-$3
  run "(cd $D && timeout 900 python3 $GS $URL $O/$tag-$3.jsonl --conc 4 --max-tokens $MT > $O/$tag-$3.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  xlog "$tag $3 (greedy set at conc 4, temperature 0, $MT tok, arm $2): $(tail -1 $O/$tag-$3.txt | cut -c1-200), $(( $(date +%s) - t0 )) s"
}
cmpx() {   # cmpx TAG NAME A B: loadid.py cmp (byte for byte + the first differing char / token) -> identity.txt
  local r
  [ "$DRY" = 1 ] && { echo "+ python3 -I $LD cmp --url $URL $3 $4 --json $O/$1-cmp-$2.json"; return 0; }
  if [ ! -s "$3" ] || [ ! -s "$4" ]; then xlog "$1 $2: not compared (missing $3 or $4)"; echo "$1 $2 not compared" >> $O/identity.txt; return 0; fi
  r=$(timeout 300 python3 -I $LD cmp --url $URL --json $O/$1-cmp-$2.json --label "$1 $2" "$3" "$4" 2>&1 | grep -m1 '^CMPLINE ' | cut -c9-)
  idrec "$1-$2" "${r:-VIRHE compare failed}" "$(basename $3) vs $(basename $4)"
}
ident() {   # ident TAG: the load identity at T 0 under S (and the base's alone line)
  local tag=$1
  stopfile || alone $tag A a1-A
  stopfile || alone $tag S a1-S
  stopfile || lidrun $tag S lid-S
  stopfile || c4rr $tag S c4a-S
  stopfile || c4rr $tag S c4b-S
  stopfile || alone $tag S a2-S
  stopfile || copyx $tag S
  cmpx $tag a1-A-vs-d0 $RD/dsd-d0-gs-c1.jsonl $O/$tag-a1-A.jsonl
  cmpx $tag a1-S-vs-d0 $RD/dsd-d0-gs-c1.jsonl $O/$tag-a1-S.jsonl
  cmpx $tag a1-S-vs-a1-A $O/$tag-a1-A.jsonl $O/$tag-a1-S.jsonl
  cmpx $tag lid-S-vs-a1-S $O/$tag-a1-S.jsonl $O/$tag-lid-S.jsonl
  cmpx $tag c4b-vs-c4a-S $O/$tag-c4a-S.jsonl $O/$tag-c4b-S.jsonl
  cmpx $tag c4a-vs-a1-S $O/$tag-a1-S.jsonl $O/$tag-c4a-S.jsonl
  cmpx $tag a2-vs-a1-S $O/$tag-a1-S.jsonl $O/$tag-a2-S.jsonl
}
summary() {
  [ "$DRY" = 1 ] && { echo "+ RESULT xs4 summary: per T / phase / arm mean +- t95, every arm vs A (pooled; adjacent pairs S-A (L-S) at T 0; 7-A, 8-7 (L-8) at T 1), S85 vs S70, SL vs S85 under sl, targets, cross-start refs (stack3 n5 P / S, n0 A; z0occ2 arm 6), prefill A / S (136k, 64k, loaded c4 + 64k), identity"; return 0; }
  python3 - "$O" "$RS3" "$RZ2" "$PORDER" "$LORDER" > $O/summary.txt 2>&1 <<'PY'
import json, math, os, statistics, sys
o, rs3, rz2, porder, lorder = sys.argv[1:6]
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
def rows(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except OSError: return []
def g(*path):
    def f(res):
        x = res
        for k in path:
            if not isinstance(x, dict): return None
            x = x.get(k)
        return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None
    return f
def ci(v):
    v = [x for x in v if x is not None]
    if not v: return None, None, 0
    m = statistics.mean(v)
    return m, (T95.get(len(v) - 1, 1.96) * statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else None), len(v)
def f1(x, n=1): return "-" if x is None else format(x, ".%df" % n)
def pc(a, b): return "%+.1f %%" % ((a / b - 1) * 100) if a and b else "-"
def mean_arm(rs, arm, get): return ci([get(r["res"]) for r in rs if r.get("ok") and r.get("arm") == arm])[0]
def pairs(rs, x, y, get):
    pr = []
    for p, q in zip(rs, rs[1:]):
        if {p.get("arm"), q.get("arm")} == {x, y} and p.get("ok") and q.get("ok"):
            px, py = (p, q) if p["arm"] == x else (q, p)
            gx, gy = get(px["res"]), get(py["res"])
            if gx and gy is not None: pr.append("%+.1f" % (100 * (gy / gx - 1)))
    return "/".join(pr) or "-"
NAME = {"A": "A", "S": "S", "8": "S85", "7": "S70", "L": "SL"}
s3 = {k: rows(os.path.join(rs3, "s3-n5-%s-blocks.jsonl" % k)) for k in ("t0", "t1")}
s30 = rows(os.path.join(rs3, "s3-n0-t0-blocks.jsonl"))
z2 = {k: rows(os.path.join(rz2, "z2-n5-%s-blocks.jsonl" % k)) for k in ("t0", "t1")}
KINDS = {
 "t0": ("T 0", "ASL", [("S", "A"), ("L", "S")],
        [("1u essay", g("c1", "essay", "decode_tps"), 90.0, 1), ("1u code", g("c1", "code", "decode_tps"), 90.0, 1), ("1u edit", g("edit", "decode_tps"), 90.0, 1),
         ("mb c1", g("mbc1", "agg_dec_tps"), 90.0, 1), ("c4", g("c4", "agg_dec_tps"), 160.0, 1), ("c4 p50", g("c4", "per_stream_dec_p50"), None, 1),
         ("c4 tok/step", g("c4", "tok_per_decode_step"), None, 2), ("c4 d0 share", g("statsph", "c4", "d0_share"), None, 3),
         ("c8", g("c8", "agg_dec_tps"), 200.0, 1), ("c8 p50", g("c8", "per_stream_dec_p50"), None, 1), ("c8 tok/step", g("c8", "tok_per_decode_step"), None, 2),
         ("c8 d0 share", g("statsph", "c8", "d0_share"), None, 3), ("pf prefill tok/s", g("pf", "prefill_tps"), None, 1),
         ("pf decode kept", g("pf", "decode_kept"), None, 3), ("pf ms/step during", g("pf", "ms_step_during"), None, 1)]),
 "t1": ("T 1 (seed 1234, thinking on; mb at the server default)", "A87L", [("7", "A"), ("8", "7"), ("L", "8")],
        [("1u essay", g("t1", "essay", "decode_tps"), 90.0, 1), ("1u code", g("t1", "code", "decode_tps"), 90.0, 1), ("1u edit", g("t1", "edit", "decode_tps"), 90.0, 1),
         ("1u essay tok/step", g("t1", "essay", "tokens_per_step"), None, 2), ("1u code tok/step", g("t1", "code", "tokens_per_step"), None, 2),
         ("c4", g("c4", "agg_dec_tps"), 160.0, 1), ("c4 p50", g("c4", "per_stream_dec_p50"), None, 1), ("c4 tok/step", g("c4", "tok_per_decode_step"), None, 2),
         ("c4 d0 share", g("statsph", "c4", "d0_share"), None, 3), ("c8", g("c8", "agg_dec_tps"), 200.0, 1), ("c8 p50", g("c8", "per_stream_dec_p50"), None, 1),
         ("c8 tok/step", g("c8", "tok_per_decode_step"), None, 2), ("c8 d0 share", g("statsph", "c8", "d0_share"), None, 3)])}
# cross-start: stack3 n5 (T 0 arm P, T 1 arm S), stack3 n0 T 0 arm A, z0occ2 arm 6 -- each read with the same getter where it has the phase
XREF = {"t0": [("stack3 n5 P", s3["t0"], "P"), ("stack3 n0 A", s30, "A"), ("z0occ2 O6", z2["t0"], "6")],
        "t1": [("stack3 n5 S", s3["t1"], "S"), ("z0occ2 O6", z2["t1"], "6")]}
def tgt(v, t):
    if t is None or v is None: return ""
    return "; >= %g: %s" % (t, "MET" if v >= t else "%.1f short" % (t - v))
out, best = [], {}
for k, (lab, arms, prs, M) in KINDS.items():
    rs = rows(os.path.join(o, "s4-n5-%s-blocks.jsonl" % k))
    if not rs: continue
    parts = []
    for nm, get, t, nd in M:
        mA = mean_arm(rs, "A", get)
        cells = []
        for a in arms:
            m, h, n = ci([get(r["res"]) for r in rs if r.get("ok") and r.get("arm") == a])
            if m is None: continue
            c = "%s %s%s" % (NAME[a], f1(m, nd), "" if h is None else " +- " + f1(h, nd))
            if a != "A" and mA: c += " (%s vs A)" % pc(m, mA)
            c += tgt(m, t)
            cells.append(c)
            if t and a != "A": best.setdefault("%s %s" % (nm, k.upper()), []).append((NAME[a], m, t))
        for x, y in prs:
            mx, my = mean_arm(rs, y, get), mean_arm(rs, x, get)
            if mx is not None and my is not None:
                cells.append("%s vs %s %s (pairs %s)" % (NAME[x], NAME[y], pc(my, mx), pairs(rs, y, x, get)))
        if t:
            xr = []
            for xl, xrs, xa in XREF[k]:
                v = [get(r["res"]) for r in xrs if r.get("ok") and r.get("arm") == xa and get(r["res"]) is not None]
                if v: xr.append("%s %s" % (xl, f1(statistics.mean(v), nd)))
            if xr: cells.append("[cross-start: %s]" % ", ".join(xr))
        if cells: parts.append("%s: %s" % (nm, " | ".join(cells)))
    flagged = sum(1 for r in rs if not r.get("ok"))
    l = "%s (%d blocks, %d flagged): %s" % (lab, len(rs), flagged, " || ".join(parts))
    out.append(l); print("ARMS " + l)
# prefill A / S
dp = [r for r in rows(os.path.join(o, "deep.jsonl")) if r.get("reset") and not r.get("name", "").startswith("guard")]
pp = []
for dep in sorted({r["depth"] for r in dp}, reverse=True):
    s = [r for r in dp if r["depth"] == dep]
    A, S = [r.get("ttft") for r in s if r["arm"] == "A"], [r.get("ttft") for r in s if r["arm"] == "S"]
    ta, tb = ci(A)[0], ci(S)[0]
    pr = []
    for p, q in zip(s, s[1:]):
        if {p["arm"], q["arm"]} == {"A", "S"} and p.get("ttft") and q.get("ttft"):
            a, b = (p, q) if p["arm"] == "A" else (q, p); pr.append("%+.1f" % (100 * (b["ttft"] / a["ttft"] - 1)))
    sg = lambda arm, k: ci([r.get(k) for r in s if r["arm"] == arm])[0]
    l = "cold %d (%s tok) TTFT: A %s s, S %s s, S vs A %s (pairs %s)%s; staged GB A %s S %s, snapshots A %s S %s" % (dep, s[0].get("prompt_tokens"), f1(ta, 2), f1(tb, 2), pc(tb, ta), "/".join(pr) or "-",
        "" if all(r.get("cached") == 0 for r in s) else " -- VIRHE a run was not cold", f1(sg("A", "staged_gb"), 1), f1(sg("S", "staged_gb"), 1), f1(sg("A", "ckpts"), 1), f1(sg("S", "ckpts"), 1))
    if dep == 135000:
        l += "; S vs the 35 s target: %s; cross-start (page-cache warm, no evict) stack3 n0 50.03 s (S %s), n5 51.33 s (S %s)" % (
            ("MET" if tb and tb <= 35 else ("%.1f s over" % (tb - 35) if tb else "-")), pc(tb, 50.03), pc(tb, 51.33))
        if tb: best.setdefault("136k TTFT s", []).append(("S", tb, 35.0))
    pp.append(l)
ld = rows(os.path.join(o, "load.jsonl"))
if ld:
    cells = []
    for nm, kk, nd in (("mixed-step ms", "nondecode_ms", 0), ("c4 agg tok/s in the prefill window", "c4_pf_agg_tps", 1),
                       ("c4 per-stream p50", "c4_pf_stream_tps_p50", 1), ("TPOT max ms", "tpot_pf_max", 0), ("probe TTFT s", "ttft", 2)):
        a, b = ci([r.get(kk) for r in ld if r["arm"] == "A"])[0], ci([r.get(kk) for r in ld if r["arm"] == "S"])[0]
        cells.append("%s A %s S %s (%s)" % (nm, f1(a, nd), f1(b, nd), pc(b, a)))
    pp.append("c4 + one cold ~64k prompt (%s): %s" % (lorder, " | ".join(cells)))
for l in pp: print("PREFILL " + l)
bl = []
for nm, vs in best.items():
    low = nm.startswith("136k")
    a, v, t = (min if low else max)(vs, key=lambda x: x[1])
    bl.append("%s best %s %s (%s)" % (nm, a, f1(v, 2 if low else 1), ("MET" if v <= t else "%.1f s over" % (v - t)) if low else ("MET" if v >= t else "%.1f short" % (t - v))))
import re
try: ident = [l.strip() for l in open(os.path.join(o, "identity.txt")) if l.strip()]
except OSError: ident = []
# the 64k prompt under c4 load vs alone runs at sched_bi_chunk 0 (2048-row mixed chunks vs alone's 4096): a known class (ii)
# difference, labelled, not a failure; a loaded S vs a loaded A compares A's ds41_bi 0 under load (not batch-invariant):
# information. Every other compare is strict.
known = [l for l in ident if re.match(r"ld-\d+[AS]-vs-alone[AS] ", l)]
info = [l for l in ident if re.match(r"ld-\d+S-vs-ldA ", l)]
strict = [l for l in ident if l not in known and l not in info]
bad = [l for l in strict if not ("IDENTICAL" in l and "differ" not in l)]
lab = lambda ls: "; ".join(l.split(" ")[0] + " " + ("IDENTICAL" if "IDENTICAL" in l and "differ" not in l else "differs") for l in ls)
try: guards = [l.strip() for l in open(os.path.join(o, "guard.txt")) if l.strip()]
except OSError: guards = []
print("SUMMARY targets: %s || guards: %s || identity (strict): %s || known class (ii) difference at sched_bi_chunk 0 (fixed by D3, r04): %s || information (A ds41_bi 0 under load): %s || prefill: %s || %s" % (
      " | ".join(bl) or "-", " | ".join(guards) or "none",
      ("IDENTICAL at every compare (%d)" % len(strict)) if strict and not bad else (("NOT IDENTICAL: " + "; ".join(bad))[:900] if strict else "none"),
      lab(known) or "-", lab(info) or "-", " | ".join(pp) or "-", " || ".join(out)))
PY
  while IFS= read -r l; do case "$l" in SUMMARY*) ;; *) xlog "${l:0:3000}" ;; esac; done < $O/summary.txt
  xlog "summary: $(grep -m1 '^SUMMARY ' $O/summary.txt | cut -c9- | cut -c1-6000)"
}
dstart() {   # the one NST 5 start
  local tag=s4-n5 ok
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on"
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV > $XS2_ENVFILE"; else printf '%s\n' $LEVENV > $XS2_ENVFILE; fi
  xlog "$tag: start ($DESC) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $FIX | arms A $(armj A) S $(armj S) 7 $(armj 7)$([ $LF = sl ] && echo " L $(armj L)") | env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ' | cut -c1-2200)"
  gate1 $tag 70 || { xlog "$tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  modes $tag || { xlog "$tag: VIRHE the modes are not the start's -> stopped"; down; return 1; }
  if [ "$DRY" != 1 ]; then
    ok=1
    docker logs $CN 2>&1 | grep -q "DSpark drafter" || { xlog "$tag: VIRHE no DSpark drafter line"; ok=0; }
    docker logs $CN 2>&1 | grep -q "draft depth: adaptive" || { xlog "$tag: VIRHE no adaptive draft depth line"; ok=0; }
    [ $ok = 1 ] || { keeplog $tag-nodraft; down; return 1; }
  fi
  vramstart $tag
  swnote $tag || { xlog "$tag: VIRHE the switches are not as expected -> stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  smoke $tag || { xlog "$tag: VIRHE the smoke failed -> stopped"; down; return 1; }
  stopfile || guard $tag
  if [ $SPEEDOK = 1 ]; then
    stopfile || blocks $tag t0
    stopfile || blocks $tag t1
    stopfile || coldabba $tag
    stopfile || loadabba $tag
  else
    xlog "$tag: VIRHE a guard failed (above): no T 0 / T 1 blocks, no prefill or loaded readings -- identity only"
  fi
  stopfile || ident $tag
  run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
  xlog "$tag: PSK stats at the end: $(pskstat) | $(docker logs $CN 2>&1 | grep -cE '^E ') E lines"
  down
  xlog "$tag: done"
}
trap 'down' EXIT
trap 'exit 143' TERM INT
# ---------------------------------------------------------------- main
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) | one NST 5 start: $DESC | T 0 order ${ORDER0:-none} phases $PH0 (mb T 0, dur $DUR warm $WARMS) | T 1 order ${ORDER1:-none} (sdprobe $T1P seed $SEED thinking on + mb c4 / c8 server default) | prefill A / S $PORDER at [$DEPTHS], loaded c4 + $LDEPTH ${LORDER:-none} | identity T 0 under S | no evict | targets 1u >= 90, c4 >= 160, c8 >= 200, 136k <= 35 s"
IMGLITS="B|dspark_conf_rows, gg_pf, f32_pf, dspark_walk, ds41_bi, draft_sample, draft_lend_hyst, arena_d0, draft_lend_w, draft_occ, draft_occ_k, place_lf, sched_bi_chunk, sched_interleave, sched_mixed_tok,
B|sched_short_bypass, sched_bypass_total, ds41_psk
B|ds41_psk
B|reset_prefix_cache
B|/tokenize
B|radiance:engine_steps_mixed_total
B|radiance:engine_step_seconds_decode_total
B|radiance:draft_accepted_total
A|RADIANCE_DS41_PSK
A|lever PSK declared (RADIANCE_DS41_PSK, fn4 ds41_psk, now
A|lever PSK: %lld cut steps
B|place_lf
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
B|draft_occ takes 0 (off) .. 64
B|draft_occ_k takes 1 .. 4096
B|occ_cycles
B|RADIANCE_DRAFT_LEND_WEIGHTS
B|lever Z0-A set up
B|RADIANCE_DRAFT_CTXSKIP_SLOTS"
[ $PSK = 2 ] && IMGLITS="$IMGLITS
B|ds41_psk_k"
docker image inspect $IMAGE > /dev/null 2>&1 || { xlog "VIRHE image $IMAGE not on this host"; exit 1; }
IMGMISS=$(echo "$IMGLITS" | timeout 120 docker run --rm -i --network none --entrypoint sh $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels/libr4d.so; S=/opt/radiance/share/radiance/kernels/spechub.so; n=0
while IFS= read -r l; do [ -n "$l" ] || continue; n=$((n + 1)); w=${l%%|*}; s=${l#*|}
  case $w in B) f=$B ;; A) f=$A ;; K) f=$K ;; S) f=$S ;; *) f="$B $A $K" ;; esac
  o=0; for x in $f; do grep -qzF -- "$s" $x && { o=1; break; }; done
  [ $o = 1 ] || echo "MISS $w|$s"; done
test -x /opt/radiance/bin/rad-sidecar || echo "MISS bin/rad-sidecar"; test -s /opt/radiance/share/radiance/int7-PROVENANCE.txt || echo "MISS int7-PROVENANCE.txt"; echo "CHECKED $n"' 2>&1)
case "$IMGMISS" in "CHECKED "*) ;; *) xlog "VIRHE $IMAGE is not the int7 image -- missing: $(echo "$IMGMISS" | grep -v '^CHECKED' | tr '\n' ';' | cut -c1-900)"; exit 1 ;; esac
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)) is the int7 image: all $(echo "$IMGMISS" | sed 's/CHECKED //') literals present (the registry with place_lf + the DST switches + ds41_psk$([ $PSK = 2 ] && echo " + ds41_psk_k"), BI / SD / DLW / Z0 / LF / DST / PSK, the /metrics counters) + bin/rad-sidecar + int7-PROVENANCE -- OK"
HMISS=""
for f in "$LIB29" "$LIBPR" $Y $GS $LD $GGB $MBT $PRB $EP $PL $DQ $MBP $TESTS $D/stepprobe.py $D/fastcheck.py; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $GS $LD $GGB $MBT $PRB $EP $PL $DQ $MBP $TESTS $D/stepprobe.py $D/fastcheck.py 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
ST=$(timeout 60 python3 -I $LD selftest --gs $GS --mb $MBP 2>&1 | tail -1); case "$ST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$LD(${ST:0:300})" ;; esac
MST=$(timeout 60 python3 -I -B $MBT --mb $MBP --selftest 2>&1 | tail -1); case "$MST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$MBT(${MST:0:300})" ;; esac
PST=$(timeout 60 python3 -I $PL selftest --tests $TESTS --mb $MBP 2>&1 | tail -1); case "$PST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$PL(${PST:0:300})" ;; esac
DQP=$(timeout 60 python3 -I -c 'import importlib.util, json, sys
def ld(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
t = ld(sys.argv[2], "t"); pl = ld(sys.argv[1], "pl")
import ast
src = open(sys.argv[3]).read()
task = next((ast.literal_eval(n.value) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "task" for t in n.targets)), None)
ok = "CPT, OFF = 3.95, 1200" in src and "# deepq fixed prompt v1" in src and pl.DEEP_CPT == 3.95 and pl.DEEP_OFF == 1200 and task == pl.DEEP_TASK
d = pl.deep_payload(t, 64000, 512)
print("DQ OK" if ok and d["messages"][0]["content"].startswith("# deepq fixed prompt v1\n") and d["max_tokens"] == 512 else "DQ MISMATCH")' $PL $TESTS $DQ 2>&1 | tail -1)
case "$DQP" in "DQ OK") ;; *) HMISS="$HMISS deepq-vs-pskld:(${DQP:0:200})" ;; esac
python3 -I -B $GGB --help > /dev/null 2>&1 || HMISS="$HMISS run:s4blocks.py--help"
(cd $D && python3 -I -B $PRB --help > /dev/null 2>&1) || HMISS="$HMISS run:sdprobe.py--help"
FNS="log run mkd restart check down stopfile best_vars desc serve_args xlog armj pinj stackenv keeplog thermline cmpfile idrec identw modes vramstart fn4get fn4set nudge fn4code pcreset arm msnap mdelta pskstat ssnap sdelta lffills guard gate1 deepx coldabba copyx loadx loadabba swnote smoke blocks alone lidrun c4rr cmpx ident summary dstart"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee sha256sum; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
for a in A S 8 7 L; do python3 -c 'import json,sys; json.loads(sys.argv[1])' "$(pinj $a)" 2>/dev/null || HMISS="$HMISS json:$a"; done
[ "$(sha256sum $MODELDIR/$SIDECAR 2>/dev/null | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256 2>/dev/null)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR"
[ "$(sha256sum $MODELDIR/$SIDE 2>/dev/null | cut -c1-64)" = "$SIDESHA" ] || HMISS="$HMISS sha256:$MODELDIR/$SIDE(want ${SIDESHA:0:16})"
grep -q 'XS2_ENVFILE' $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no env_file)"
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config refuses it)"
HMISS=$(echo $HMISS)
[ -z "$HMISS" ] || { xlog "VIRHE helpers / inputs do not resolve: $HMISS"; exit 1; }
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing, loadid.py $ST, mbt.py $MST, pskld.py $PST, deepq.py = pskld.py's deep request ($DQP), s4blocks.py / sdprobe.py --help, all $(echo $FNS | wc -w) functions resolve, tools on PATH, sidecars $SIDECAR + $SIDE sha256 OK, yaml accepted -- OK"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { xlog "VIRHE $M225 missing or not verified"; exit 1; }
  for f in $RD/dsd-d0-gs-c1.jsonl $RD/copy-dsd-d0-i1-run1.txt $RM/deep-dsm-225.txt $WARM_REF_225 $RS3/s3-n5-t0-blocks.jsonl $RS3/s3-n5-t1-blocks.jsonl $RZ2/z2-n5-t0-blocks.jsonl; do [ -s $f ] || xlog "note -- reference $f missing (not compared)"; done
  xlog "$ITEM: helpers md5: s4blocks.py $(md5sum < $GGB | cut -c1-8), mbt.py $(md5sum < $MBT | cut -c1-8), loadid.py $(md5sum < $LD | cut -c1-8), pskld.py $(md5sum < $PL | cut -c1-8), sdprobe.py $(md5sum < $PRB | cut -c1-8), deepq.py $(md5sum < $DQ | cut -c1-8), greedyset.py $(md5sum < $GS | cut -c1-8), mb.py $(md5sum < $MBP | cut -c1-8)"
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDE $RD/dsd-d0-gs-c1.jsonl $RD/copy-dsd-d0-i1-run1.txt $RM/deep-dsm-225.txt $WARM_REF_225 $RS3/s3-n5-stats0.json $RS3/s3-n5-t0-blocks.jsonl $RS3/s3-n5-t1-blocks.jsonl $RS3/s3-n0-t0-blocks.jsonl $RZ2/z2-n5-t0-blocks.jsonl $RZ2/z2-n5-t1-blocks.jsonl; do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  for a in A S 8 7 $([ $LF = sl ] && echo L); do echo "+ arm $a: $(armj $a)"; done
  echo "+ fixed pin: $FIX"
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(information here; the real run refuses to start beside one)"
fi
mkd $O $O/guard $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt
stopfile || dstart
summary
unset $SWITCHES
down
xlog "$ITEM: done -- $O"
