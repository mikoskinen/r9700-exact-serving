#!/bin/bash
# dsm-small.sh (10.11.2026, queue-ar/dsmain, ds-main-port small-levers agent) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-<x>-small.sh. Log tag xsm, out runs/xdsm/small. Helpers adapted from dsm-vramstack.sh / dsm-stack.sh (copied, tag
# xsm); drivers vlblocks.py (read only: D7 re-arm, settle block, per-phase /stats + draft counters) and xsmsum.py (the summary);
# yaml ds41-small-tp4.yaml (= ds41-vramstack-tp4.yaml with env_file ${XSM_ENVFILE}).
#
# WHAT: the exact SMALL levers of branch ds41-small, folded together (user 10.11.: no lever dropped for being small; exact small
#   wins are folded, not skipped), each an in-process switch (POST /admin/fn4), measured COMBINED: arm B (every lever on) vs arm A
#   (every lever off = the int line) in the same start, blocks ABBA. The levers (THE TABLE below, one a line):
#     d3   ARGATHER above 4 rows (moe_gather_all_reduce up to 48 rows; ds41_argather 48 vs 4, declared by ARGATHER_MAX=48)
#     i1   indexer row split balanced deal (idx_split 1 vs 0; acts on mixed steps / a prompt's last chunk only)
#     k1a  K1 fold a: rmsnorm(q_a_norm) in q_b's GEMV staging (ds41_qnfold; declared by RADIANCE_DS41_QNFOLD=1)
#     k1e  K1 fold e: swiglu_clamped in the shared down GEMV's staging (ds41_swfold; RADIANCE_DS41_SWFOLD=1)
#     ctx  depth-0 context-pass skip + window refill (draft_ctxskip; set up by RADIANCE_DRAFT_CTXSKIP=0; NST 5 only)
#     conf DSpark confidence gate (ds41-acc's dspark_conf, cherry-picked) under a load gate: dspark_conf 1 + dspark_conf_rows 1
#          (the gate only at one decoding sequence; hszz-confhead: 1u +1-6 %, c8 +3.6 %, c4 -4.0 % ungated); declared by
#          RADIANCE_DS41_DSPARK_CONF=/models/dspark-conf.f32 (the release's confidence head, MODELDIR, sha256 checked); NST 5
#   A table row whose literal the image lacks is skipped with a note (the item runs the rest). A start-env lever (kind env:
#     kvwin  KVWIN, ds41-kvfp8 a64aa1c merged: RADIANCE_DS41_KVWIN_FP8=1, kv_window + kv_dspark keys as fp8_e4m3 codes, exact)
#   cannot be an in-process arm: it gets a FOLLOW-UP START per NST it applies to (kvwin: NST 5 only) -- every fn4 lever on + the
#   env lever, blocks XSM_ENVORDER (BB) of arm B, phases as the main start; the 136k cold prefill (vs dsm-225); identity under B
#   and under B with ds41_kvfold 0 (the fold writes the packed cache); a prefix-cache hit (the 28k copy again, cached > 0) vs the
#   cold run and bz d0; resident units at start against the main start's -- compared with the main start's B blocks
#   (CROSS-START, labelled so). Its on-card selftest window_store_fp8_bf16 runs with the folds' before any server. c4 / c8
#   greedy identity is not run: the c4 batch-invariance floor (6 of 20 differ between identical runs) makes it no test.
# DESIGN, why two starts (the cheapest valid one): one NST 5 start covers all five switches, but the line's c4 / c8 configuration
#   is NST 0 (hx-stack's design: NST 0 c4 135.5 / c8 170.6 vs NST 5 114.6 / 147.2), and two levers act on shapes only NST 0 has:
#   K1 a / e at 1 and 4 decode rows (NST 5's c1 / c4 verify steps are 6 / ~24 rows, its c8 depth-0 steps 8), D3 at NST 0's c8
#   8-row steps (NST 5 c8 depth 0 has them too, c4 not). So:
#   START n0 (NST 0, levers d3 i1 k1a k1e): blocks ABBA, phases c1 (essay, code) / c4 / c8 (no edit: another 1-row decode, no
#     shape of its own), mb XSM_DUR 75 s; then the 136k COLD PREFILL under A and B (I1's case: the prompt's last partial chunk;
#     NST 0 = dsm-225's configuration, its reference TTFT 51.56 s and answer); identity per arm (28k copy + 20 greedy c1 vs bz d0,
#     deep vs dsm-225).
#   START n5 (DSpark NST 5 adaptive, D7 depth 0 + the hybrid (match 8) pinned, levers d3 i1 k1a k1e ctx): SETTLE block of A
#     (c1, c4, c8 at 30 s; left out), then ABBA, phases c1 (essay, code) / edit / c4 / c8, D7 RE-ARMED every block
#     (vlblocks --rearm); identity per arm (copy + greedy c1 vs bz d0); the context-skip checks when the image has the forced
#     depth-0 hook (XSM_CTXCHK).
#   PER-LEVER blocks only if the combined result is AMBIGUOUS (XSM_PERLEVER auto): a headline metric where B loses >= XSM_LOSS_PCT
#     (1.0 %) pooled and in every A / B pair (a consistent loss, not ABBA noise) -- a lever hurts and must be named. Then, in that start, leave-one-out arms ('1'..'9' = B with table lever k
#     off), order 1B2B..., on the losing phases only; summary per lever: (B - lever) vs B. XSM_PERLEVER 1 always, 0 never.
#   STAMPED c4 SCREEN of the confidence gate (XSM_STAMP auto: when lever conf runs; lead's request 10.11.): a short NST 5 start
#     with RADIANCE_AQL_STAMP=1 (every played pass on card 0 timed per dispatch; NOT a speed, not gated), every lever of B on,
#     then c4 windows (d0attrblocks.py window, read only; D7 re-armed before each): a settle (30 s), then W0 dspark_conf 0,
#     W1 1 (always, = hszz-confhead's arm), W2 101 (the 0.5 rule capped by the adaptive window: never more rows than the
#     window), W3 0 again (drift), 40 s each -> d0attrprof.py stamps (read only) -> per window ms a step by kernel family
#     group (dense GEMM, expert kernels, all-reduce, attention / indexer, other), card 0 idle a step, tok a step. At c4 the
#     load gate's N = 1 / 2 are W0 (no cut at 4 sequences) and N = 4 is W1, so they need no window of their own.
#   SELFTEST first (cards free, no server): the image's r4d_selftest on card 0, the fold cases (XSM_SELFTEST) -- an exactness
#     defect stops the item before any server.
# STACK (both starts, both arms): 105's t3 env + gg_gemv 2 + aql_acq 2 (rule-w) + K1 c / d / f declared and on (FIX0 / FIX5),
#   turnover 0 (XSM_BASEENV, = hx-stack's XST_VRAMENV); NST 5 adds draft_adaptive 1, draft_depth0 1, draft_hybrid 1 (match 8).
#   + drafter_mx 1 at NST 5 (lever DK, ds41-int2 merged into ds41-small; in process, same bits; as hx-stack's XST_DRAFTER_MX=1;
#   XSM_DRAFTER_MX=0 leaves it out). No lend (RADIANCE_DRAFT_LEND unset).
# SUMMARY (xsmsum.py): per start B vs A per metric (pooled %, ABBA pairs, the B value's distance to target); the 136k TTFT A vs B;
#   identity; per-lever lines when run; "RESULT xsm summary: ..." then "RESULT xsm <item>: done".
# NO cc_sim (user 10.11.), no 2.9. No power / host / fan settings; production compose files read only (--project-directory).
# XSM_IMAGE, XSM_ITEM, XSM_NST (n0 n5), XSM_ORDER (ABBA), XSM_DUR (75), XSM_SETTLE (1), XSM_PHASES0, XSM_PHASES5, XSM_DEEP (n0: the start
# that runs the 136k under both arms), XSM_PERLEVER (auto | 1 | 0), XSM_LOSS_PCT, XSM_SELFTEST, XSM_BASEENV, XSM_ON (all | names),
# XSM_TABLE_EXTRA, XSM_ENVORDER (BB), XSM_CTXCHK (auto | 0).
# DRYRUN=1 bash dsm-small.sh : prints every command; checks the image (docker image inspect + a no-device, no-network grep of its
#   files) and the yaml (docker compose config --quiet); no server, no GPU, nothing written.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-small-tp4.yaml
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=$R/xdsm/small; export STATSDIR=$O/stats
RD=$R/xdsm/draft                                # bz's d0 files (identity reference: copy + greedy set)
RM=$R/xdsm                                      # item a's dsm-225: deep-dsm-225.txt(.json) (136k cold reference)
export IMAGE=${XSM_IMAGE:-radiance-local:ds41-small-SHA8-5c13e555-l}
ITEM=${XSM_ITEM:-57-p4zyzzzzz-ds-small}
DUR=${XSM_DUR:-75}; NSTS=${XSM_NST:-n0 n5}; ORDER=${XSM_ORDER:-ABBA}; SETTLE=${XSM_SETTLE:-1}; MATCH=8; HYB=1
PH0=${XSM_PHASES0:-c1,c4,c8}; PH5=${XSM_PHASES5:-c1,edit,c4,c8}; DEEPNST=${XSM_DEEP:-n0}
STAMP=${XSM_STAMP:-auto}; SWDUR=${XSM_STAMP_DUR:-40}; PERLEVER=${XSM_PERLEVER:-auto}; LOSS=${XSM_LOSS_PCT:-1.0}; ENVORDER=${XSM_ENVORDER:-BB}; CTXCHK=${XSM_CTXCHK:-auto}
BASEENV=${XSM_BASEENV-RADIANCE_PLACE_TURNOVER=0}; ONSEL=${XSM_ON:-all}; DMX=${XSM_DRAFTER_MX:-1}
case $DMX in 0|1) ;; *) echo "XSM_DRAFTER_MX must be 0 or 1 (got '$DMX')"; exit 1 ;; esac
SELFTEST=${XSM_SELFTEST-rmsnorm_gemm_nt_ggml_qk256 rmsnorm_gemm_nt_ggml_qk32 swiglu_clamped_gemm_nt_ggml_qk256 swiglu_clamped_gemm_nt_ggml_qk32 window_store_fp8_bf16}
case $ORDER in *[!AB]*|"") echo "XSM_ORDER takes A / B letters (got '$ORDER')"; exit 1 ;; esac
case $ENVORDER in *[!B]*|"") echo "XSM_ENVORDER takes B letters (got '$ENVORDER')"; exit 1 ;; esac
case $PERLEVER in auto|0|1) ;; *) echo "XSM_PERLEVER must be auto, 0 or 1 (got '$PERLEVER')"; exit 1 ;; esac
case $SETTLE in 0|1) ;; *) echo "XSM_SETTLE must be 0 or 1 (got '$SETTLE')"; exit 1 ;; esac
case $CTXCHK in auto|0) ;; *) echo "XSM_CTXCHK must be auto or 0 (got '$CTXCHK')"; exit 1 ;; esac
case $STAMP in auto|0) ;; *) echo "XSM_STAMP must be auto or 0 (got '$STAMP')"; exit 1 ;; esac
for n in $NSTS; do case $n in n0|n5) ;; *) echo "XSM_NST takes n0 / n5 (got '$n')"; exit 1 ;; esac; done
case $DEEPNST in n0|n5) ;; *) echo "XSM_DEEP takes n0 / n5 (got '$DEEPNST')"; exit 1 ;; esac
for kv in $BASEENV; do case "$kv" in [A-Z]*=*) ;; *) echo "XSM_BASEENV takes NAME=VALUE words (got '$kv')"; exit 1 ;; esac; done

# ---------------------------------------------------------------- THE LEVER TABLE (one lever a line; a line starting '#' is off)
# name  kind  on                     off  nst  needs                     decl                              logok                         what
#   kind   fn4 = in-process switch(es): on = SWITCH=VALUE[,SWITCH=VALUE] (arm B), off = the arm-A value(s) in the same order (the
#                int line's behaviour); a decl value /models/FILE must exist in MODELDIR (a sidecar); decl = the start
#                env NAME=VALUE[,..] that declares / sets it up (both arms; '-' none) -> in the combined A / B arms
#          env = a start-time env NAME=VALUE[,..] (on), off '-', decl '-' -> a follow-up start (fn4 levers on + this env)
#   nst    n0 | n5 | any;  needs  a literal the image must carry (grep in bin/radiance, deepseek41.so, kernels/*.so; '-' none)
#   logok  a literal the start log must carry when declared (',' for a space; '-' none)
XSM_TABLE='
d3    fn4  ds41_argather=48    4  any  RADIANCE_DS41_ARGATHER_MAX  RADIANCE_DS41_ARGATHER_MAX=48  one,kernel,up,to,48,tokens,a,step  D3: moe_gather_all_reduce up to 48 rows (B) vs today'"'"'s 4 (A); same rank-partial rounding and rank-order fold
i1    fn4  idx_split=1         0  any  RADIANCE_DS41_IDX_BALANCE   -                               -                             I1: the indexer row split balanced deal (mixed steps / a prompt'"'"'s last chunk only)
k1a   fn4  ds41_qnfold=1       0  any  RADIANCE_DS41_QNFOLD        RADIANCE_DS41_QNFOLD=1          RADIANCE_DS41_QNFOLD;,ds41_qnfold=1) K1 fold a: rmsnorm(q_a_norm) in q_b'"'"'s GEMV staging
k1e   fn4  ds41_swfold=1       0  any  RADIANCE_DS41_SWFOLD        RADIANCE_DS41_SWFOLD=1          RADIANCE_DS41_SWFOLD;,ds41_swfold=1) K1 fold e: swiglu_clamped in the shared down GEMV'"'"'s staging
ctx   fn4  draft_ctxskip=1     0  n5   RADIANCE_DRAFT_CTXSKIP      RADIANCE_DRAFT_CTXSKIP=0        draft,context,skip,(RADIANCE_DRAFT_CTXSKIP=0                     depth-0 context-pass skip + the drafter window refill before the head runs
conf  fn4  dspark_conf=1,dspark_conf_rows=1  0,0  n5  dspark_conf_rows  RADIANCE_DS41_DSPARK_CONF=/models/dspark-conf.f32  a,confidence,plane,(the,dspark_conf,gate  DSpark confidence gate (ds41-acc C1) only where the step decoded <= 1 sequence (dspark_conf_rows 1); sidecar = the confidence head of the release
kvwin env  RADIANCE_DS41_KVWIN_FP8=1  -  n5  RADIANCE_DS41_KVWIN_FP8  -  lever,KVWIN:,the,window,caches,hold,each,key,as,fp8_e4m3,codes  KVWIN: kv_window + kv_dspark keys as fp8_e4m3 codes + a ue8m0 byte a block (ds41-kvfp8 a64aa1c), exact; a start env -> its own start against B
'
TABLE="$XSM_TABLE
${XSM_TABLE_EXTRA:-}"
LROWS=$(echo "$TABLE" | awk 'NF && $1 !~ /^#/')
lrow() { echo "$LROWS" | awk -v n="$1" '$1 == n {print; exit}'; }
lf() { lrow "$1" | awk -v k="$2" '{print $k}'; }
lwhat() { lrow "$1" | awk '{for (i = 1; i <= 8; i++) $i = ""; sub(/^ +/, ""); print}'; }
LNAMES=$(echo "$LROWS" | awk '{print $1}')
while read -r nm kd on off ns nd dc lo rest; do
  [ -n "${nm:-}" ] || continue
  [ -n "${lo:-}" ] || { echo "lever table: '$nm' has fewer than 8 fields"; exit 1; }
  case $kd in fn4|env) ;; *) echo "lever table: '$nm' kind '$kd' is not fn4 / env"; exit 1 ;; esac
  case $ns in n0|n5|any) ;; *) echo "lever table: '$nm' nst '$ns' is not n0 / n5 / any"; exit 1 ;; esac
  case $kd in
    fn4) for w in ${on//,/ }; do case $w in [a-z]*=-[0-9]*|[a-z]*=[0-9]*) ;; *) echo "lever table: '$nm' fn4 on '$w' is not switch=number"; exit 1 ;; esac; done
         for w in ${off//,/ }; do case $w in -[0-9]*|[0-9]*) ;; *) echo "lever table: '$nm' fn4 off '$w' is not a number"; exit 1 ;; esac; done
         [ $(echo ${on//,/ } | wc -w) = $(echo ${off//,/ } | wc -w) ] || { echo "lever table: '$nm' fn4 on / off counts differ"; exit 1; }
         [ "$dc" = - ] || for w in ${dc//,/ }; do case $w in [A-Z]*=*) ;; *) echo "lever table: '$nm' decl word '$w' is not NAME=VALUE"; exit 1 ;; esac; done ;;
    env) for w in ${on//,/ }; do case $w in [A-Z]*=*) ;; *) echo "lever table: '$nm' env word '$w' is not NAME=VALUE"; exit 1 ;; esac; done
         [ "$off" = - ] && [ "$dc" = - ] || { echo "lever table: '$nm' (env) takes '-' for off and decl"; exit 1; } ;;
  esac
  [ "$(echo "$LNAMES" | grep -cx "$nm")" = 1 ] || { echo "lever table: '$nm' appears twice"; exit 1; }
done <<< "$LROWS"
[ "$(echo "$LROWS" | awk '$2 == "fn4"' | wc -l)" -le 9 ] || { echo "lever table: at most 9 fn4 levers (leave-one-out arms are 1..9)"; exit 1; }
if [ "$ONSEL" = all ]; then ONL=$LNAMES
else ONL=""; for nm in $ONSEL; do lrow $nm | grep -q . || { echo "XSM_ON: no table lever '$nm'"; exit 1; }; ONL="$ONL $nm"; done; fi
appl() { local n=$(lf $1 5); [ $n = any ] || [ $n = $2 ]; }        # appl LEVER NST
fnsw() { lf $1 3 | tr ',' '\n' | cut -d= -f1 | xargs; }            # the fn4 lever's switch name(s)
fnpairs() {   # fnpairs LEVER on|off -> SWITCH=VALUE words
  local on=$(lf $1 3) off=$(lf $1 4) i=0 w
  for w in ${on//,/ }; do i=$((i + 1)); [ $2 = on ] && echo "$w" || echo "${w%%=*}=$(echo ${off//,/ } | cut -d' ' -f$i)"; done | xargs
}
# arm JSON over the fn4 levers given: on = the table's on values, off = the off values, leave-one-out "k" = on except lever k
armjson() {   # armjson on|off|lN LEVERS...
  local m=$1 nm a="" k=0; shift
  for nm in "$@"; do k=$((k + 1))
    case $m in on) a="$a $(fnpairs $nm on)" ;; off) a="$a $(fnpairs $nm off)" ;; l*) [ "l$k" = $m ] && a="$a $(fnpairs $nm off)" || a="$a $(fnpairs $nm on)" ;; esac
  done
  python3 -c 'import json,sys
print(json.dumps({w.split("=")[0]: int(w.split("=")[1]) for w in sys.argv[1:]}, separators=(",", ":")))' $a
}
declw() { local nm w; for nm in "$@"; do w=$(lf $nm 7); [ "$w" = - ] || echo "${w//,/ }"; done | xargs; }   # fn4 levers' start env
envon() { local nm; for nm in "$@"; do w=$(lf $nm 3); echo "${w//,/ }"; done | xargs; }                   # env levers' env
TENV=$(for nm in $LNAMES; do case $(lf $nm 2) in fn4) declw $nm ;; env) envon $nm ;; esac; done | tr ' ' '\n' | sed 's/=.*//' | sort -u | xargs)

GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
DAB=$HD/d0attrblocks.py; DAP=$HD/d0attrprof.py; PARSE_PID=""
EP=$HD/editprobe.py; VLB=$HD/vlblocks.py; VSB=$HD/vsblocks.py; GGB=$HD/ggblocksdz2.py; XSS=$HD/xsmsum.py; DQ=$QA/deepq.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
FIX0='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1}'
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": '"$MATCH"', "draft_hybrid": '"$HYB"'}'
[ "$DMX" = 1 ] && FIX5="${FIX5%\}}, \"drafter_mx\": 1}"   # lever DK in the NST 5 pin (as hx-stack's XST_DRAFTER_MX=1)
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DS41_ARGATHER_MAX RADIANCE_DS41_SHEXP_LANE RADIANCE_DEBUG_MISS_LOG
  RADIANCE_DS41_KVFOLD RADIANCE_DS41_OFOLD RADIANCE_DS41_ONEFILL RADIANCE_DS41_PIPE
  RADIANCE_DRAFT_DEPTH0 RADIANCE_DRAFT_PROBE RADIANCE_DRAFT_HYBRID RADIANCE_DRAFT_HYBRID_MATCH
  RADIANCE_DRAFT_LEND RADIANCE_DRAFT_PROBE_BACKOFF RADIANCE_DRAFT_LEND_BORROW RADIANCE_DEBUG_VRAM
  RADIANCE_DS41_IDX_BALANCE RADIANCE_DS41_QNFOLD RADIANCE_DS41_SWFOLD RADIANCE_DRAFT_CTXSKIP RADIANCE_DS41_EMB_RAM"
stackenv() {   # stackenv ENVWORDS: hx-stack's stack env (105's t3 arm + gg_gemv 2 + rule-w + K1 c/d/f declared and on) + ENVWORDS
  local kv
  unset $SWITCHES $TENV
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  for kv in $1; do export "$kv"; done
}
armset() {   # armset NST ARMJSON [EXTRAJSON]: the NST's fixed pin + the arm's switches (+ extra) as one POST body
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update(json.loads(sys.argv[2])); d.update(json.loads(sys.argv[3])); print(json.dumps(d))' \
    "$([ $1 = n0 ] && echo "$FIX0" || echo "$FIX5")" "$2" "${3:-{\}}"
}
# ---------------------------------------------------------------- helpers (dsm-vramstack.sh / dsm-stack.sh, copied; tag xsm)
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
gate() {   # gate TAG LIMIT: lib-29's gate + keeplog before the retry / after a second slow check
  local tag=$1 lim=$2 ms ms1 st=fast
  restart $tag || { keeplog $tag-fail; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && return 0
  if awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}'; then
    keeplog $tag-try1
    ms1=$ms; restart $tag || { keeplog $tag-fail; return 1; }; ms=$(check $tag 2); st="fast-after-retry (first $ms1)"
    awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}' && { st="SLOW-after-retry (first $ms1)"; keeplog $tag-try2; }
  fi
  echo "$st $ms" > $O/guard-$tag.txt
  log "RESULT xsm guard $tag: $st, 1-user ${ms} ms/step (limit $lim; NST 5 DSpark starts depth0b / hybrid 35.2-35.3, NST 0 bz dsd-d0 15.59)"
  case $st in SLOW*) return 2 ;; esac
}
gpurun() {   # gpurun NAME SECONDS docker-run-args...: bounded, named, removed even if its kernel hangs (dsm-attnfold2.sh's, copied)
  local name=$1 t=$2; shift 2
  timeout -k 15 $t docker run --rm --init --stop-timeout 10 --name $name "$@"; local rc=$?
  docker rm -f $name > /dev/null 2>&1 || true
  { [ $rc = 124 ] || [ $rc = 137 ]; } && echo "gpurun: $name stopped after $t s (rc $rc)" >&2
  return $rc
}
FOLLOW_PID=""
follow() { [ "$DRY" = 1 ] && { echo "+ docker logs -f $CN > $1/engine.log &"; return; }
  docker logs -f $CN > $1/engine.log 2>&1 & FOLLOW_PID=$!; sleep 2; }
unfollow() { [ -n "$FOLLOW_PID" ] && { kill $FOLLOW_PID 2>/dev/null; wait $FOLLOW_PID 2>/dev/null; FOLLOW_PID=""; }; return 0; }
trap 'unfollow; down' EXIT
trap 'exit 143' TERM INT
thermline() { type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"; }
cmpfile() {   # cmpfile A B -> IDENTICAL (n chars) | DIFFERENT at char k (na vs nb chars) | not compared
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xsm $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xsm $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  echo "warm-up-$1 $r" >> $O/identity.txt
  case $r in IDENTICAL*) log "RESULT xsm $1 warm-up answers vs stock ($(basename $2)): $r" ;;
    *) log "RESULT xsm $1 warm-up answers vs stock ($(basename $2)): VIRHE $r -- NOT EXACT vs the tuned3l line (the start's arm: every switch at its start value)" ;; esac
}
modes() {   # modes TAG "ENVWORDS" LEVERS...: the stack's notes (t3 indexer, rule-w, K1 c/d/f on, no shexp lane) + each declared
  # lever's env in the container and its log literal -> 0, else 1
  local tag=$1 ew=$2 MLF SL AL XL KL NW NR ok=1 kv nm lo CE bad=""; shift 2
  [ "$DRY" = 1 ] && { echo "+ modes $tag: indexer note (mxfp4, latent bf16, compressor tail bound, router selection+sort separate, AR_PCN separate at 0 layers), staging line, no shared-expert-lane note, rule-w in every recorded pass, the lever K1 note (c / d / f declared, switches at 1); env [${ew:-none}] in the container; log literals of [$*]: $(for nm in "$@"; do echo -n "$nm '$(lf $nm 8 | tr , ' ')' "; done); mover kept-free 0 on 4 ranks (turnover 0)"; return 0; }
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in $ew; do echo "$CE" | grep -qxF "$kv" || bad="$bad env:${kv%%=*}"; done
  for nm in "$@"; do lo=$(lf $nm 8); [ "$lo" = - ] && continue
    docker logs $CN 2>&1 | grep -qF -- "${lo//,/ }" || bad="$bad log:$nm('${lo//,/ }')"; done
  for kv in $ew; do
    docker logs $CN 2>&1 | grep -F -- "${kv%%=*}" | grep -qE 'is not used|is not a count|refused|cannot' && bad="$bad refused:${kv%%=*}($(docker logs $CN 2>&1 | grep -F -- "${kv%%=*}" | grep -m1 -E 'is not used|is not a count|refused|cannot' | cut -c1-160))"
  done
  [ -z "$bad" ] || { log "RESULT xsm $tag: VIRHE a lever did not reach the engine:$bad"; ok=0; }
  case " $ew " in *" RADIANCE_PLACE_TURNOVER=0 "*)
    nz=$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots \([0-9]+ shadow, 0 more kept free')
    [ "$nz" = 4 ] || { log "RESULT xsm $tag: VIRHE turnover 0 asked but $nz of 4 ranks keep 0 slots free"; ok=0; } ;;
  esac
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  AL=$(docker logs $CN 2>&1 | grep -m1 'the MoE gather and its all-reduce as one kernel' | sed 's/^radiance: //' | cut -c1-300)
  XL=$(docker logs $CN 2>&1 | grep -m1 'the shared expert on the second lane' | sed 's/^radiance: //' | cut -c1-300)
  KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-900)
  log "RESULT xsm $tag modes: $(echo "$MLF" | cut -c1-700) | staging: ${SL:-none} | argather: ${AL:-none} | K1: ${KL:-none} | env [${ew:-none}]"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate" "post-collapse-norm separate at 0 layers" "next collapse-norm separate at 0 layers"; do
    case "$MLF" in *"$w"*) ;; *) log "RESULT xsm $tag: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  [ -n "$AL" ] || { log "RESULT xsm $tag: VIRHE no ARGATHER note"; ok=0; }
  [ -z "$XL" ] || { log "RESULT xsm $tag: VIRHE a shared-expert-lane note (RADIANCE_DS41_SHEXP_LANE should be unset): '$XL'"; ok=0; }
  [ -n "$SL" ] || { log "RESULT xsm $tag: VIRHE no prefill staging line (levels refused?)"; ok=0; }
  case "$KL" in
    *"in one launch at 0 of"*|*"rope+round trip at 0 ("*|*"not declared at"*|*"at 0 MoE layers"*|"")
      log "RESULT xsm $tag: VIRHE the lever K1 note does not show folds c / d / f declared: '${KL}'"; ok=0 ;;
  esac
  case "$KL" in *"ds41_kvfold=1"*"ds41_ofold=1"*"ds41_onefill=1"*) ;; *) log "RESULT xsm $tag: VIRHE the K1 c/d/f switches do not start at 1: '${KL}'"; ok=0 ;; esac
  # folds a / e (when declared): the note names how many layers fold; 0 means the fold resolved nowhere
  KLF=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds')
  case "$KLF" in *"staging in 0 "*) log "RESULT xsm $tag: VIRHE a K1 a / e fold resolved at 0 layers: '$(echo "$KLF" | grep -oE "the (query norm|shared expert)[^,]*" | tr '\n' ' ')'"; ok=0 ;; esac
  [ -n "$KLF" ] && log "RESULT xsm $tag K1 a / e: $(echo "$KLF" | grep -oE "the (query norm|shared expert's clamped SwiGLU) folded [^)]*\)" | tr '\n' ' ' | cut -c1-500)"
  NW=$(docker logs $CN 2>&1 | grep -c 'the acquire rule plays .*RADIANCE_AQL_ACQ=rule-w)'); NR=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_AQL_ACQ=rule)')
  [ "$NW" -gt 0 ] && [ "$NR" = 0 ] || { log "RESULT xsm $tag: VIRHE rule-w is not what the recorded passes play ($NW rule-w, $NR rule)"; ok=0; }
  [ $ok = 1 ] || keeplog $tag-nomode
  [ $ok = 1 ]
}
vramunits() {   # vramunits FILE: the routed experts' VRAM / pinned units of a /stats snapshot
  python3 -c 'import json,sys
try: d = json.load(open(sys.argv[1]))
except Exception as e: print("n/a (%s)" % e); sys.exit(0)
t = {x.get("name"): x.get("units") for x in d.get("experts", {}).get("tiers", [])}
print("vram %s, pinned %s" % (t.get("vram"), t.get("pinned")))' "$1" 2>&1
}
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
fn4try() {   # fn4try JSON -> the HTTP status of the POST (200 applied / 400 refused); dry: 200
  [ "$DRY" = 1 ] && { echo "+ POST $URL/admin/fn4 $1 (status)" >&2; echo 200; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null
}
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
pcreset() {   # pcreset TAG: POST /reset_prefix_cache (loopback operator) so the next prompt is prefilled cold
  [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o $O/$1-reset.json -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  log "RESULT xsm $1 prefix cache reset: HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"
}
swnote() {   # swnote TAG NST ARM=JSON...: GET at start (the stack's values; every lever switch served, not -1); each arm's set
  # applied at a step; then back to the first arm
  local tag=$1 nst=$2 s0 s a j bad="" sw; shift 2
  sw=$(for a in $FNL; do fnsw $a; done | xargs)
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (expect gg_gemv 2, aql_acq 2, ds41_kvfold / ds41_ofold / ds41_onefill 1, $([ $nst = n0 ] && echo "draft_capable 0" || echo "draft_adaptive 1, draft_capable 1"), epoch 0, lever switches served (not -1): [$sw]); then per arm POST + a 4-token request -> applied: $(for a in "$@"; do echo -n "${a%%=*} $(armset $nst "${a#*=}") "; done)"; return 0; }
  s0=$(fn4get)
  python3 -c 'import json,sys
s0 = json.loads(sys.argv[1]); w = s0["switches"]; n0 = sys.argv[2] == "n0"
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) and s0["epoch"] == 0
ok = ok and (s0["draft_capable"] == 0 if n0 else (s0["draft_capable"] == 1 and w["draft_adaptive"] == 1))
ok = ok and all(w.get(k, -1) != -1 for k in sys.argv[3].split() if k)
sys.exit(0 if ok else 1)' "$s0" $nst "$sw" 2>/dev/null || bad="at start"
  for a in "$@"; do
    j=$(armset $nst "${a#*=}" '{"epoch_bump": 1}'); fn4set "$j"; nudge; s=$(fn4get)
    python3 -c 'import json,sys
s, want = json.loads(sys.argv[1]), json.loads(sys.argv[2]); want.pop("epoch_bump", None)
sys.exit(0 if all(s["switches"].get(k) == v for k, v in want.items()) and not s["pending"] and s["epoch"] >= 1 else 1)' "$s" "$j" 2>/dev/null || bad="$bad arm ${a%%=*}"
  done
  fn4set "$(armset $nst "${1#*=}" '{"epoch_bump": 1}')"; nudge
  log "RESULT xsm $tag switch note: at start $(echo "$s0" | cut -c1-1400) | after the arm checks: $(fn4get | cut -c1-1100)$([ -n "$bad" ] && echo " -- VIRHE not as expected:$bad")"
  [ -z "$bad" ]
}
copyt() {   # copyt TAG OFFSET: the copy test after a ~31k-token cold prompt (105's prompt: same offset -> same text)
  local t0=$(date +%s)
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset $2 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xsm $1 copy after a cold prompt: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300) | $(thermline $t0 $(date +%s))"
}
gsetc1() { run "(cd $D && timeout 900 python3 $GS $URL $O/$1-gs-c1.jsonl --conc 1 --max-tokens 192 > $O/$1-gs-c1.txt 2>&1)"; }
gcmp() { [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }; [ -s "$1" ] && [ -s "$2" ] || { echo "not compared (missing $1 or $2)"; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
idline() {   # idline KEY LABEL RESULT: identity.txt + one RESULT line (a difference is reported, not hidden)
  echo "$1 $3" >> $O/identity.txt
  case "$3" in IDENTICAL*|"not compared"*) log "RESULT xsm identity $2: $3" ;;
    *) log "RESULT xsm identity $2: VIRHE $3 -- every lever here is exact by construction, so this is a defect" ;; esac
}
deep() {   # deep TAG NST ARM ARMJSON: the arm's set, a reset, the 136k cold prefill (deepq.py, 512 greedy tokens) -> TTFT + the answer vs dsm-225
  local tag=$1-d$3 t0=$(date +%s) j
  fn4set "$(armset $2 "$4" '{"epoch_bump": 1}')"; nudge
  pcreset $tag
  run "(cd $D && timeout 1800 python3 $DQ $O/deep-$tag.txt 135000 512 > $O/deep-$tag.out 2>&1)"
  [ "$DRY" = 1 ] && { echo "+ RESULT xsm $tag 136k cold prefill: TTFT (cached must be 0) vs $RM/deep-dsm-225.txt.json; answer vs $RM/deep-dsm-225.txt"; return 0; }
  j=$(grep -m1 '^DEEPQ ' $O/deep-$tag.out | cut -c7-)
  [ -n "$j" ] || { log "RESULT xsm $tag 136k cold prefill: VIRHE no answer ($(tail -2 $O/deep-$tag.out | tr '\n' ' ' | cut -c1-300))"; return 1; }
  echo "$j" > $O/deep-$tag.json
  log "RESULT xsm $tag 136k cold prefill (deepq.py fixed prompt, 512 greedy tokens; arm $3): $(python3 -c 'import json,sys
d = json.loads(sys.argv[1])
try: r = json.load(open(sys.argv[2]))
except Exception: r = {}
t, rt = d.get("ttft"), r.get("ttft")
print("status %s, prompt %s tok, cached %s%s, TTFT %s s (dsm-225 cold %s s%s), %s prefill tok/s, %s tok" % (
    d.get("status"), d.get("prompt_tokens"), d.get("cached"), "" if d.get("cached") == 0 else " -- VIRHE not cold",
    t, rt, ", %+.1f %%" % ((t / rt - 1) * 100) if t and rt else "", round(d["prompt_tokens"] / t, 1) if t and d.get("prompt_tokens") else "-", d.get("tokens")))' "$j" $RM/deep-dsm-225.txt.json 2>&1 | cut -c1-400) | $(thermline $t0 $(date +%s))"
  idline deep-$tag "$tag 136k deep answer (cold) vs dsm-225 (NST 0, stock, cold)" "$(cmpfile $O/deep-$tag.txt $RM/deep-dsm-225.txt)"
}
ident() {   # ident TAG NST ARM ARMJSON [WHAT]: the arm's set, a reset, the 28k copy and the c1 greedy set vs bz d0
  local tag=$1-i$3 st
  fn4set "$(armset $2 "$4" '{"epoch_bump": 1}')"; nudge
  pcreset $tag
  copyt $tag 0; gsetc1 $tag
  [ "$DRY" = 1 ] && { echo "+ identity $tag (arm $3${5:+, $5}): 28k copy vs bz d0, greedy set c1 vs bz d0"; return 0; }
  st=$(fn4get)
  log "RESULT xsm $tag identity ran under: $(echo "$st" | cut -c1-1100)"
  idline copy-$tag "$tag 28k copy vs bz d0 (NST 0, stock)${5:+ [$5]}" "$(cmpfile $O/copy-$tag-run1.txt $RD/copy-dsd-d0-i1-run1.txt)"
  idline gs-$tag "$tag greedy set c1 vs bz d0${5:+ [$5]}" "$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$tag-gs-c1.jsonl)"
}
blocks() {   # blocks TAG NST ORDER SETTLEARM PHASES ARM=JSON...: vlblocks.py (NST 5: D7 re-armed every block; SETTLEARM '-' = none)
  local tag=$1 nst=$2 ord=$3 st=$4 ph=$5 args="" a fx; shift 5
  for a in "$@"; do args="$args --arm '$a'"; done
  [ $nst = n5 ] && args="$args --rearm"
  [ "$st" != - ] && args="$args --settle $st --settle-phases c1,c4,c8 --settle-dur 30"
  fx=$([ $nst = n0 ] && echo "$FIX0" || echo "$FIX5")
  run "timeout 9000 python3 -I $VLB --url $URL --mb-dir $D --edit $EP --label $tag $args --order $ord --phases $ph --dur $DUR --warm 10 --fixed '$fx' --ready-s 90 --sample-s 2 --out $O/$tag-blocks.jsonl > $O/$tag-blocks.txt 2>&1"
  [ "$DRY" = 1 ] && return 0
  while IFS= read -r l; do log "RESULT xsm $tag ${l:0:1600}"; done < <(grep -E '^(BLOCK|GGB \{"error)' $O/$tag-blocks.txt)
  log "RESULT xsm $tag counters a phase (block phase: vram units end, rank-0 hit %, missed a step, stream MB a step, draft zero / probe cycles): $(paste -d'|' <(grep '^VSX ' $O/$tag-blocks.txt | cut -c5-) <(grep '^VDL ' $O/$tag-blocks.txt | cut -c5-) | python3 -c 'import json,sys
out = []
for l in sys.stdin:
    a, b = l.rstrip("\n").split("|", 1)
    try: x = json.loads(a)
    except ValueError: continue
    if x.get("phase") not in ("c1", "edit", "c4", "c8"): continue
    out.append("b%s %s: %s u, hit %s, miss %s, stream %s" % (x.get("block"), x.get("phase"), x.get("vram_units_end"), x.get("hit_pct_rank0"), x.get("missed_step_rank0"), x.get("stream_MB_step")))
print("; ".join(out))' 2>&1 | cut -c1-3000)"
}
selftest() {   # the image's r4d_selftest on card 0, before any server: each fold against the rows it folds, byte for byte.
  # Per case: RAN when its output holds an "ok" / "FAIL" line (a --case child runs that row alone); a FAIL line is an EXACTNESS
  # DEFECT; no ok / FAIL line at all, or a "no smoke case" / "device phase" skip,
  # (rc 126 / 127 = the command did not run) is NOT RUN (packaging), never an exactness verdict;
  # rc 124 / 137 is a HANG; any other non-zero rc with only ok lines is an ERROR. Anything but every case clean stops the item.
  local DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ulimit memlock=-1 --ipc=host -e HIP_VISIBLE_DEVICES=0"
  local c rc nf nok nsk st nexact=0 npkg=0 nother=0 r
  [ -n "$SELFTEST" ] || { log "RESULT xsm selftest: none asked (XSM_SELFTEST empty)"; return 0; }
  if [ "$DRY" != 1 ]; then
    for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
    [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT xsm selftest: VIRHE a KFD client is running (after 60 s wait) -- NOT RUN"; return 1; }
  fi
  for c in $SELFTEST; do
    run "gpurun xsmst-$$-${c:0:12} 900 $DEV --entrypoint /opt/radiance/bin/r4d_selftest $IMAGE /opt/radiance/share/radiance/kernels/libr4d.so --case $c > $O/selftest-$c.txt 2>&1"
    rc=$?
    [ "$DRY" = 1 ] && continue
    # a --case child runs that one row's smoke test alone, so every ok / FAIL line in its output is the case's (a case may name
    # its checks in its own words, e.g. "lever KVWIN: ..."); "skip <row> -- no smoke case" / "skip device phase" = it did not run
    nf=$(grep -cE "^ *FAIL " $O/selftest-$c.txt); nok=$(grep -cE "^ *ok " $O/selftest-$c.txt)
    nsk=$(grep -cE "^ *skip +($c -- no smoke case|device phase)" $O/selftest-$c.txt)
    if [ $nf -gt 0 ] && [ "$(grep -E "^ *FAIL " $O/selftest-$c.txt | grep -cvE "absent")" = 0 ]; then
      st="NOT RUN (packaging: the case's rows / hooks absent: $(grep -m1 -E "^ *FAIL " $O/selftest-$c.txt | sed 's/^ *//' | cut -c1-200))"; npkg=$((npkg + 1))
    elif [ $nf -gt 0 ]; then st="EXACTNESS DEFECT ($nf FAIL line(s) from the case that ran)"; nexact=$((nexact + 1))
    elif [ $rc = 124 ] || [ $rc = 137 ]; then st="HANG (stopped by the 900 s limit, rc $rc; $nok ok line(s))"; nother=$((nother + 1))
    elif [ $nok = 0 ] || [ $nsk -gt 0 ]; then st="NOT RUN (packaging: rc $rc, no line from the case$([ $nsk -gt 0 ] && echo ", $nsk skip line(s)"); $(head -c 160 $O/selftest-$c.txt | tr '\n' ' '))"; npkg=$((npkg + 1))
    elif [ $rc != 0 ]; then st="ERROR (rc $rc after $nok ok line(s))"; nother=$((nother + 1))
    else st="CLEAN ($nok ok)"; fi
    log "RESULT xsm selftest $c: rc $rc, $nok ok, $nf FAIL -> $st | $(grep -m1 -E "^ *(ok|FAIL) " $O/selftest-$c.txt | sed 's/^ *//' | cut -c1-600)"
    while IFS= read -r l; do log "RESULT xsm selftest info $l"; done < <(grep -E '^ *info ' $O/selftest-$c.txt | sed 's/^ *info *//' | cut -c1-200)
  done
  [ "$DRY" = 1 ] && return 0
  if [ $nexact -gt 0 ]; then
    echo "selftest VIRHE exactness ($nexact case(s))" >> $O/identity.txt
    log "RESULT xsm selftest: VIRHE $nexact case(s) FAILED on the card -- an exactness defect, the engine phases do not run"; return 1
  elif [ $npkg -gt 0 ] || [ $nother -gt 0 ]; then
    echo "selftest NOT RUN ($npkg packaging, $nother hang / error)" >> $O/identity.txt
    log "RESULT xsm selftest: VIRHE selftest NOT RUN (packaging: $npkg case(s) produced no case line; hang / error: $nother) -- not an exactness verdict; the item stops"; return 1
  fi
  echo "selftest CLEAN" >> $O/identity.txt
  log "RESULT xsm selftest: CLEAN (every fold byte-identical to the rows it folds, on card 0)"
}
hitchk() {   # hitchk TAG NST ARMJSON: the 28k copy cold, then the same request again (a prefix-cache / checkpoint hit): both vs bz d0
  local tag=$1-hit
  fn4set "$(armset $2 "$3" '{"epoch_bump": 1}')"; nudge
  pcreset $tag
  copyt $tag-cold 0; copyt $tag-hit 0
  [ "$DRY" = 1 ] && { echo "+ identity $tag: the hit's copy (cached > 0) vs the cold one and vs bz d0"; return 0; }
  idline copy-$tag "$tag 28k copy on a prefix-cache hit (cached $(grep '^J ' $O/$tag-hit-copy.txt | head -1 | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()[2:]).get("cached"))' 2>/dev/null)) vs the cold run" "$(cmpfile $O/copy-$tag-hit-run1.txt $O/copy-$tag-cold-run1.txt)"
  idline copy-$tag-ref "$tag 28k copy on the hit vs bz d0" "$(cmpfile $O/copy-$tag-hit-run1.txt $RD/copy-dsd-d0-i1-run1.txt)"
}
ctxchk() {   # ctxchk TAG ARMJSON_B: the context skip under FORCED depth 0 at c1 (draft_depth0 2, the image's test hook; exact):
  # identity (copy + greedy c1 vs bz d0) with draft_ctxskip 1, then the refill check -- probe acceptance with the skip off vs
  # on (the c1 stepprobe essay / code, draft counters): equal = the refilled window drafts as the per-step one
  local tag=$1-cx b=$2 on off st
  [ "$CTXCHK" = 0 ] && { log "RESULT xsm $tag: XSM_CTXCHK=0 -- skipped"; return 0; }
  case " $FNL " in *" ctx "*) ;; *) return 0 ;; esac
  on=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d["draft_ctxskip"]=1; print(json.dumps(d, separators=(",", ":")))' "$b")
  off=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d["draft_ctxskip"]=0; print(json.dumps(d, separators=(",", ":")))' "$b")
  st=$(fn4try "$(armset n5 "$on" '{"draft_depth0": 2, "draft_hybrid": 0}')")
  [ "$st" = 200 ] || { log "RESULT xsm $tag: the image refuses draft_depth0 2 (HTTP $st) -- no forced depth-0 hook, the context-skip identity rests on the c4 / c8 blocks' depth-0 steps and the item's other identity runs"; fn4set "$(armset n5 "$b" '{"epoch_bump": 1}')"; nudge; return 0; }
  nudge
  pcreset $tag; copyt $tag-on 0; gsetc1 $tag-on
  if [ "$DRY" != 1 ]; then
    log "RESULT xsm $tag forced depth 0 + ctxskip 1 ran under: $(fn4get | cut -c1-1100)"
    idline copy-$tag "$tag 28k copy, forced depth 0 + draft_ctxskip 1, vs bz d0" "$(cmpfile $O/copy-$tag-on-run1.txt $RD/copy-dsd-d0-i1-run1.txt)"
    idline gs-$tag "$tag greedy set c1, forced depth 0 + draft_ctxskip 1, vs bz d0" "$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$tag-on-gs-c1.jsonl)"
  fi
  run "timeout 2400 python3 -I $VLB --url $URL --mb-dir $D --edit $EP --label $tag --arm 'A=$off' --arm 'B=$on' --order ABBA --phases c1 --dur $DUR --warm 10 --fixed '$(armset n5 '{}' '{"draft_depth0": 2, "draft_hybrid": 0}')' --ready-s 90 --sample-s 2 --out $O/$tag-blocks.jsonl > $O/$tag-blocks.txt 2>&1"
  [ "$DRY" = 1 ] || while IFS= read -r l; do log "RESULT xsm $tag refill check ${l:0:1200}"; done < <(grep -E '^BLOCK' $O/$tag-blocks.txt)
  [ "$DRY" = 1 ] || log "RESULT xsm $tag after the refill check: $(fn4get | python3 -c 'import json,sys
d = json.loads(sys.stdin.read()); print("accept %s | ctxskip %s" % (json.dumps(d.get("accept")), json.dumps(d.get("ctxskip"))))' 2>&1 | cut -c1-600)"
  fn4set "$(armset n5 "$b" '{"epoch_bump": 1, "draft_depth0": 1}')"; nudge
}
# ---------------------------------------------------------------- one start
FNL=""
xstart() {   # xstart NST KIND: KIND main = the fn4 levers' A / B start; env = a follow-up start with the env levers on
  local nst=$1 kind=$2 tag P lim ew arms ord st ph a fnl envl n k loss lo
  fnl=$(fnl $nst); envl=$([ $kind = env ] && envlv $nst)
  FNL=$fnl
  tag=sm-$nst$([ $kind = env ] && echo -env); P=$O/$tag
  ew=$(echo $BASEENV $(declw $fnl) $([ $kind = env ] && envon $envl))
  local A B; A="A=$(armjson off $fnl)"; B="B=$(armjson on $fnl)"
  if [ $kind = main ]; then arms="$A $B"; ord=$ORDER; else arms="$B"; ord=$ENVORDER; fi
  st=-; [ $nst = n5 ] && [ $SETTLE = 1 ] && st=$([ $kind = main ] && echo A || echo B)
  ph=$([ $nst = n0 ] && echo $PH0 || echo $PH5)
  stackenv "$ew"; export MODEL=$M225; HP29=19456; best_vars
  case $nst in
    n0) lim=30; export MNBT=4160 NST=0 XARGS="$XARGS --live" ;;
    n5) lim=70; export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive" ;;
  esac
  export XSM_ENVFILE=$O/$tag.env
  log "RESULT xsm $tag: start ($nst, $kind: fn4 levers [$fnl]$([ $kind = env ] && echo ", env levers [$envl]"); arms $arms; settle $st, order $ord, phases $ph) IMAGE=$IMAGE MODEL=$MODEL $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')| env file $XSM_ENVFILE [${ew:-empty}]"
  mkd $P
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $ew > $XSM_ENVFILE"; else printf '%s\n' $ew > $XSM_ENVFILE; fi
  gate $tag $lim; local rc=$?
  [ $rc = 1 ] && { log "RESULT xsm $tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  [ $rc = 2 ] && { log "RESULT xsm $tag: VIRHE over the gate limit after one restart -> start skipped"; down; return 1; }
  modes $tag "$ew" $fnl $envl || { log "RESULT xsm $tag: VIRHE the modes / lever notes are not as expected -> start stopped"; down; return 1; }
  if [ "$DRY" != 1 ]; then
    if [ $nst = n5 ]; then docker logs $CN 2>&1 | grep -q "DSpark drafter" && docker logs $CN 2>&1 | grep -q "draft depth: adaptive" \
      || { log "RESULT xsm $tag: VIRHE no DSpark drafter / adaptive depth line"; keeplog $tag-nodraft; down; return 1; }
    else docker logs $CN 2>&1 | grep -qiE "DSpark drafter|'dspark' proposes" && { log "RESULT xsm $tag: VIRHE a DSpark drafter line at NST 0"; keeplog $tag-nodraft; down; return 1; }; fi
  fi
  swnote $tag $nst $arms || { log "RESULT xsm $tag: VIRHE the switches are not as expected -> start stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  run "curl -s -m 15 $URL/stats > $O/$tag-stats0.json 2>/dev/null"
  [ "$DRY" = 1 ] || log "RESULT xsm $tag experts at start: $(vramunits $O/$tag-stats0.json) | bz d0: $(vramunits $RD/dsd-d0/stats-start.json)"
  follow $P
  stopfile || blocks $tag $nst $ord $st $ph $arms
  # per-lever leave-one-out blocks when the combined result loses somewhere (XSM_PERLEVER auto) or always (1)
  if [ $kind = main ] && ! stopfile && [ $PERLEVER != 0 ]; then
    if [ "$DRY" = 1 ]; then loss="(dry: decided from $O/$tag-blocks.jsonl by xsmsum.py --decide $LOSS)"
    else loss=$(python3 $XSS --decide $O/$tag-blocks.jsonl $LOSS $ph 2>&1 | tail -1); fi
    lo=$( [ $PERLEVER = 1 ] && echo $ph || echo "$loss" | sed -n 's/^LOSS phases=\([a-z0-9,]*\).*/\1/p')
    log "RESULT xsm $tag per-lever decision (XSM_PERLEVER $PERLEVER, loss threshold $LOSS %): ${loss:-none}"
    if [ -n "$lo" ] && [ "$lo" != none ]; then
      local la="" lord="" k=0
      for a in $fnl; do k=$((k + 1)); la="$la $k=$(armjson l$k $fnl)"; lord="$lord${k}B"; done
      log "RESULT xsm $tag per-lever leave-one-out: arms [$(k=0; for a in $fnl; do k=$((k + 1)); echo -n "$k=B-$a "; done)] order $lord phases $lo"
      blocks $tag-loo $nst $lord - $lo $B $la
    fi
  fi
  if [ $kind = main ] && [ $nst = $DEEPNST ]; then
    for a in $arms; do stopfile || deep $tag $nst ${a%%=*} "${a#*=}"; done
  fi
  [ $kind = env ] && ! stopfile && deep $tag $nst B "${B#*=}"
  for a in $arms; do stopfile || ident $tag $nst ${a%%=*} "${a#*=}"; done
  if [ $kind = env ]; then   # the env lever under both K1 fold c settings (it packs the cache the fold writes) and a prefix-cache hit
    stopfile || ident $tag $nst K "$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d["ds41_kvfold"]=0; print(json.dumps(d, separators=(",", ":")))' "${B#*=}")" "B with ds41_kvfold 0"
    stopfile || hitchk $tag $nst "${B#*=}"
  fi
  [ $kind = main ] && [ $nst = n5 ] && ! stopfile && ctxchk $tag "${B#*=}"
  unfollow
  run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
  down
  local pl="PLAN nst=$nst kind=$kind tag=$tag levers=$(echo $fnl | tr ' ' ',') env=$(echo ${envl:--} | tr ' ' ',') blocks=$O/$tag-blocks.jsonl loo=$O/$tag-loo-blocks.jsonl deepA=$O/deep-$tag-dA.json deepB=$O/deep-$tag-dB.json stats0=$O/$tag-stats0.json mainstats0=$O/sm-$nst-stats0.json ctx=$O/$tag-cx-blocks.jsonl"
  if [ "$DRY" = 1 ]; then echo "+ $pl >> $O/plan.txt"; else echo "$pl" >> $O/plan.txt; fi
  log "RESULT xsm $tag: done"
}
stampc4() {   # the stamped c4 screen of dspark_conf (W0 0, W1 1, W2 101, W3 0) on an NST 5 start with every B lever's env
  local tag=sm-stamp W=$O/stamp-win.jsonl fnl ew B0 w v ms wins="" k
  fnl=$(fnl n5); FNL=$fnl
  case " $fnl " in *" conf "*) ;; *) return 0 ;; esac
  ew=$(echo $BASEENV $(declw $fnl))
  B0=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update({"dspark_conf": 0, "dspark_conf_rows": 0}); print(json.dumps(d, separators=(",", ":")))' "$(armjson on $fnl)")
  stackenv "$ew"; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive" AQLSTAMP=1 STATSDIR=$O/stats-stamp XSM_ENVFILE=$O/$tag.env
  mkd $STATSDIR
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $ew > $XSM_ENVFILE"; else printf '%s\n' $ew > $XSM_ENVFILE; fi
  run "find $STATSDIR -maxdepth 1 -name 'stamps.csv*' -delete; rm -f $W"
  log "RESULT xsm $tag: start (RADIANCE_AQL_STAMP=1, every played pass on card 0; NOT a speed, not gated; B's levers [$fnl], conf switched per window) IMAGE=$IMAGE $(desc)"
  restart $tag || { keeplog $tag-fail; log "RESULT xsm $tag: VIRHE not healthy -- no stamped screen"; down; export AQLSTAMP= STATSDIR=$O/stats; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] || log "RESULT xsm $tag warm-up + 1-user fastcheck (stamped, not gated): $ms ms/step"
  run "timeout 900 python3 -I $DAB window --url $URL --mb-dir $D --label $tag-settle --phase c4 --dur 30 --warm 10 --reset --set '$(armset n5 "$B0" '{"epoch_bump": 1}')' --out $W > /dev/null 2>&1"
  k=0
  for v in 0 1 101 0; do
    stopfile && break
    run "timeout 900 python3 -I $DAB window --url $URL --mb-dir $D --label $tag-w$k --phase c4 --dur $SWDUR --warm 10 --reset --set '$(armset n5 "$B0" "{\"dspark_conf\": $v, \"epoch_bump\": 1}")' --out $W > /dev/null 2>&1"
    if [ "$DRY" != 1 ]; then
      w=$(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    try: d = json.loads(l)
    except ValueError: continue
    if d.get("label") == sys.argv[2]: print("%s:%s" % (d.get("t0"), d.get("t1")))' $W $tag-w$k 2>/dev/null | tail -1)
      [ -n "$w" ] && [ "$w" != "None:None" ] && wins="$wins --win w$k:$w"
      log "RESULT xsm $tag w$k (dspark_conf $v, c4 $SWDUR s): $(python3 -c 'import json,sys
r = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
d = next((x for x in reversed(r) if x.get("label") == sys.argv[2]), {})
m, w = d.get("mb") or {}, d.get("win") or {}
print("ok %s, applied dspark_conf %s; c4 %s tok/s, %s ms a decode step, %s tok a step; window %s decode steps, zero %s / probe %s cycles" % (
    d.get("ok"), (d.get("applied") or {}).get("dspark_conf"), m.get("agg_dec_tps"), m.get("ms_decode_step"), m.get("tok_per_decode_step"),
    w.get("decode_steps"), (w.get("draft") or {}).get("zero_cycles"), (w.get("draft") or {}).get("probe_cycles")))' $W $tag-w$k 2>&1 | cut -c1-400)"
    fi
    k=$((k + 1))
  done
  run "curl -s -m 15 $URL/stats > $O/$tag-stats-end.json 2>/dev/null"
  keeplog $tag
  down
  if [ "$DRY" = 1 ]; then echo "+ (background) nice python3 -I $DAP stamps $STATSDIR/stamps.csv --win w0:T0:T1 .. --win w3:T0:T1 --json $O/stamps.json; gzip stamps.csv"
  elif [ -s $STATSDIR/stamps.csv ] && [ -n "$wins" ]; then
    ( nice -n 10 timeout 2400 python3 -I $DAP stamps $STATSDIR/stamps.csv $wins --json $O/stamps.json --label xsm > $O/stamps.txt 2>&1; nice -n 10 gzip -f $STATSDIR/stamps.csv ) &
    PARSE_PID=$!
    log "RESULT xsm $tag: stamps parsed in the background (pid $PARSE_PID):$wins"
  else log "RESULT xsm $tag: VIRHE no stamps / windows ($(ls -la $STATSDIR 2>&1 | tail -2 | tr '\n' ' '), windows '$wins')"; fi
  export AQLSTAMP= STATSDIR=$O/stats
  local pl="PLAN nst=n5 kind=stamp tag=$tag win=$W stamps=$O/stamps.json"
  if [ "$DRY" = 1 ]; then echo "+ $pl >> $O/plan.txt"; else echo "$pl" >> $O/plan.txt; fi
}
fnl() { local nm; for nm in $ONOK; do appl $nm $1 && [ "$(lf $nm 2)" = fn4 ] && echo $nm; done | xargs; }
envlv() { local nm; for nm in $ONOK; do appl $nm $1 && [ "$(lf $nm 2)" = env ] && echo $nm; done | xargs; }
# ---------------------------------------------------------------- main
# preflight 1: every helper the body calls resolves (lib-29 / lib-pr266 and this file), DRYRUN too
HELPERS="log run mkd restart check down evict stopfile desc best_vars gpurun keeplog gate follow unfollow thermline cmpfile identw
  modes vramunits fn4get fn4set fn4try nudge pcreset swnote copyt gsetc1 gcmp idline deep ident blocks selftest hitchk ctxchk
  stampc4 xstart fnl envlv armjson armset stackenv declw envon fnsw fnpairs lf lrow lwhat appl"   # needok is defined beside its use below
MISSH=""; for h in $HELPERS; do type -t $h > /dev/null 2>&1 || MISSH="$MISSH $h"; done
[ -z "$MISSH" ] || { log "RESULT xsm: VIRHE helpers not defined:$MISSH -- packaging"; exit 1; }
[ "$DRY" = 1 ] && echo "+ helpers resolve: $(echo $HELPERS | wc -w) of $(echo $HELPERS | wc -w) ($(for h in gpurun restart check selftest; do echo -n "$h=$(type -t $h) "; done))"
IID=$(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null)
[ -n "$IID" ] || { log "RESULT xsm: VIRHE image $IMAGE not on this host"; exit 1; }
# preflight 2: the selftest binary runs (no device, no network) and its catalogue lists every case as a libr4d row
if [ -n "$SELFTEST" ]; then
  STCAT=$(timeout 300 docker run --rm --network none --entrypoint /opt/radiance/bin/r4d_selftest $IMAGE /opt/radiance/share/radiance/kernels/libr4d.so 2>&1)
  STMISS=""; for c in $SELFTEST; do echo "$STCAT" | grep -qE "^ +$c +op=" || STMISS="$STMISS $c"; done
  [ -n "$STCAT" ] && [ -z "$STMISS" ] || { log "RESULT xsm: VIRHE the image's r4d_selftest catalogue (no device) lacks case row(s):${STMISS:- (no output)} -- packaging"; exit 1; }
  [ "$DRY" = 1 ] && for c in $SELFTEST; do echo "+ selftest catalogue (no device): $(echo "$STCAT" | grep -m1 -E "^ +$c +op=" | sed -E 's/ +/ /g' | cut -c1-90)"; done
fi
NEEDS=$( { for nm in $LNAMES; do n=$(lf $nm 6); [ "$n" = - ] || echo "$n"; done; [ "$DMX" = 1 ] && echo drafter_mx; } | sort -u | xargs)
HAVE=$(timeout 180 docker run --rm --network none --entrypoint sh $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels; S=/opt/radiance/bin/r4d_selftest
s=yes; for l in draft_depth0 draft_hybrid_match reset_prefix_cache; do grep -qzF "$l" $B || s=no; done
grep -qzF RADIANCE_DS41_KVFOLD $A && grep -qzF rmsnorm_rope_roundtrip_kv_store $K/libr4d.so && grep -qzF r4d_gg_mr_pin $K/libr4d.so || s=no
echo "STACK $s"
for l in "$@"; do o=0; for f in $B $A $K/*.so; do grep -qzF -- "$l" "$f" && { o=1; break; }; done; echo "NEED $o $l"; done
for c in '"$SELFTEST"'; do grep -qzF -- "$c" $S && echo "CASE 1 $c" || echo "CASE 0 $c"; done' sh $NEEDS 2>/dev/null)
[ "$DMX" = 1 ] && { echo "$HAVE" | grep -qxF "NEED 1 drafter_mx" || { log "RESULT xsm: VIRHE XSM_DRAFTER_MX=1 but $IMAGE has no drafter_mx switch (ds41-int2's lever DK)"; exit 1; }; }
echo "$HAVE" | grep -qx "STACK yes" || { log "RESULT xsm: VIRHE $IMAGE is not a stack image (depth 0 / hybrid registry, reset_prefix_cache, K1 in deepseek41.so / libr4d, lever D2)"; exit 1; }
for c in $SELFTEST; do echo "$HAVE" | grep -qxF "CASE 1 $c" || { log "RESULT xsm: VIRHE $IMAGE's r4d_selftest has no case '$c'"; exit 1; }; done
needok() { local n=$(lf $1 6); [ "$n" = - ] || echo "$HAVE" | grep -qxF "NEED 1 $n"; }
ONOK=""; SKIP=""
for nm in $ONL; do if needok $nm; then ONOK="$ONOK $nm"; else SKIP="$SKIP $nm(lacks '$(lf $nm 6)')"; fi; done
ONOK=$(echo $ONOK)
log "RESULT xsm $ITEM: start IMAGE=$IMAGE (${IID:7:12}) | levers [${ONOK:-none}]$([ -n "$SKIP" ] && echo ", skipped:$SKIP") | starts [$NSTS] (+ a follow-up start per NST where an env lever applies), order $ORDER (settle $SETTLE at NST 5, D7 re-armed every NST 5 block), phases n0 $PH0 / n5 $PH5, mb $DUR s, 136k under A and B at $DEEPNST, per-lever $PERLEVER (loss $LOSS %), selftest [${SELFTEST:-none}], base env [$BASEENV], ctxchk $CTXCHK, stamp $STAMP, drafter_mx $DMX at NST 5, yaml $(basename $Y) | refs: in-process A (every lever off); identity bz d0 (copy, greedy c1), dsm-225 (136k deep, cold TTFT 51.56 s) | targets 1u >= 90, c4 >= 160, c8 >= 200 tok/s, 136k <= 35 s"
for nm in $ONOK; do echo "  lever $nm: $(lf $nm 2) on $(lf $nm 3) off $(lf $nm 4) [nst $(lf $nm 5), decl $(lf $nm 7)] $(lwhat $nm)"; done
STARTS=""
for nst in $NSTS; do
  [ -n "$(fnl $nst)" ] && STARTS="$STARTS $nst:main"
  [ -n "$(envlv $nst)" ] && STARTS="$STARTS $nst:env"
  [ -z "$(fnl $nst)$(envlv $nst)" ] && log "RESULT xsm $nst: no lever under test applies here -> no start"
done
MIN=2; case " $(fnl n5) $NSTS " in *" conf "*" n5 "*) [ $STAMP = auto ] && MIN=$((MIN + 10)) ;; esac
for s in $STARTS; do n=${s%%:*}; k=${s#*:}
  b=$([ $k = main ] && echo ${#ORDER} || echo ${#ENVORDER}); pb=$([ $n = n0 ] && echo 4 || echo 6)
  MIN=$(( MIN + 5 + b * pb + $([ $n = n5 ] && [ $SETTLE = 1 ] && echo 2 || echo 0) + $([ $k = main ] && [ $n = $DEEPNST ] && echo 3 || echo 0) + $([ $k = main ] && echo 5 || echo 10) + $([ $k = main ] && [ $n = n5 ] && echo 6 || echo 0) )); done
log "RESULT xsm plan: starts [${STARTS:- none}] (~$MIN min without per-lever blocks; each leave-one-out pair adds ~$((4 * 2))-$((6 * 2)) min on the losing phases)"
if [ "$DRY" != 1 ]; then
  [ -n "$STARTS" ] && docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xsm: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  for v in RADIANCE_DS41_GG_GEMV RADIANCE_AQL_ACQ RADIANCE_DS41_KVFOLD RADIANCE_DS41_OFOLD RADIANCE_DS41_ONEFILL; do grep -q "^ *$v:" $Y || { log "RESULT xsm: VIRHE $Y does not pass $v"; exit 1; }; done
  grep -q 'XSM_ENVFILE' $Y || { log "RESULT xsm: VIRHE $Y has no env_file entry"; exit 1; }
  for f in $GS $EP $VLB $VSB $GGB $XSS $DQ $DAB $DAP $D/stepprobe.py $D/mb.py $D/ds41_tests.py; do [ -e $f ] || { log "RESULT xsm: VIRHE $f missing"; exit 1; }; done
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xsm: VIRHE $M225 missing or not verified"; exit 1; }
  for nm in $ONOK; do for w in $(declw $nm); do case ${w#*=} in /models/*) f=$MODELDIR/${w#*=/models/}
    [ -s $f ] || { log "RESULT xsm: VIRHE lever $nm's sidecar $f missing"; exit 1; }
    [ -s $f.sha256 ] && { [ "$(sha256sum $f | cut -c1-64)" = "$(cut -c1-64 $f.sha256)" ] || { log "RESULT xsm: VIRHE lever $nm's sidecar $f does not match $f.sha256"; exit 1; }; } ;; esac; done; done
  for f in $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $RM/deep-dsm-225.txt $RM/deep-dsm-225.txt.json; do [ -s $f ] || log "RESULT xsm: note -- reference $f missing (not compared)"; done
else
  for f in $Y $GS $EP $VLB $VSB $GGB $XSS $DQ $DAB $DAP $D/stepprobe.py $D/mb.py $D/ds41_tests.py $MODELDIR/$M225 $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $RM/deep-dsm-225.txt $RM/deep-dsm-225.txt.json \
           $(for nm in $ONOK; do for w in $(declw $nm); do case ${w#*=} in /models/*) echo $MODELDIR/${w#*=/models/} ;; esac; done; done); do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  echo "+ image $IMAGE (${IID:7:12}): stack image OK; lever literals: $(echo "$HAVE" | grep '^NEED' | tr '\n' ';'); selftest cases: $(echo "$HAVE" | grep '^CASE' | tr '\n' ';')"
fi
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XSM_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || { log "RESULT xsm: VIRHE docker compose config refuses $Y"; exit 1; }
[ "$DRY" = 1 ] && echo "+ yaml $Y: docker compose config OK (env_file \${XSM_ENVFILE})"
if [ -z "$STARTS" ]; then
  log "RESULT xsm summary: nothing to measure on $IMAGE -- no lever under test applies$([ -n "$SKIP" ] && echo "; skipped:$SKIP"); no start, no GPU time"
  log "RESULT xsm $ITEM: done -- nothing run"
  exit 0
fi
mkd $O $O/guard $STATSDIR
[ "$DRY" = 1 ] || { : > $O/identity.txt; : > $O/plan.txt; }
selftest || exit 1
evict
for s in $STARTS; do stopfile && break; xstart ${s%%:*} ${s#*:}; done
[ $STAMP = auto ] && ! stopfile && case " $NSTS " in *" n5 "*) stampc4 ;; esac
[ -n "$PARSE_PID" ] && { wait $PARSE_PID 2>/dev/null; PARSE_PID=""; }
if [ "$DRY" = 1 ]; then
  echo "+ python3 $XSS $O/plan.txt $O/identity.txt $RM/deep-dsm-225.txt.json > $O/summary.txt -> RESULT xsm lines (per start B vs A per metric: pooled %, ABBA pairs, B's target distance; 136k A vs B; leave-one-out per lever; env-start B vs main B), RESULT xsm summary"
else
  python3 $XSS $O/plan.txt $O/identity.txt $RM/deep-dsm-225.txt.json > $O/summary.txt 2>&1
  while IFS= read -r l; do case "$l" in SUMMARY*) ;; *) log "RESULT xsm ${l:0:3000}" ;; esac; done < $O/summary.txt
  log "RESULT xsm summary: $(grep -m1 '^SUMMARY ' $O/summary.txt | cut -c9- | cut -c1-4500)"
fi
unset $SWITCHES $TENV
down
log "RESULT xsm $ITEM: done -- $O"
