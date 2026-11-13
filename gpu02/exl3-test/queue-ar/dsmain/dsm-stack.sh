#!/bin/bash
# dsm-stack.sh (10.11.2026, queue-ar/dsmain, ds-main-port) -- NOT a queue item; the body of 57-p4zyzzzzz-ds-hx-stack.sh.
# THE DS STACK CONFIRMATION: today's exact levers together on the ds41-stack image (XST_IMAGE, build/stack/assemble.sh:
# ds41-stack = ds41-k1 1d1f782 (ds41-main 3c33393 + ds41-d3 + ds41-gg 8b311fc + lever K1) + ds41-gg 1b8848a (D7 depth 0 with its
# estimator fix + the lookup / head hybrid), layered on the k1 image; exl3 plugins of 5c13e555), measured against the existing
# NST 0 + gg_gemv 2 reference blocks (no baseline re-run) and the stream's targets, in TWO STARTS (NST 0 first):
#   START 1, THE NST 0 STACK (arm N; XST_NST0=1, the default; 0 leaves it out): NST 0 (no drafter), gg_gemv 2, aql_acq 2, the
#     three K1 folds -- why: hcy-depth0b, depth 0 in an NST 5 server still cost -7.6 % essay / -9.9 % c4 / -13.8 % c8 against
#     plain NST 0 + D2, so today's best exact configuration for essay / c4 / c8 is NST 0 and for code NST 5: both are measured.
#     gate (30 ms) -> modes -> no drafter line -> switch note (ds41_kvfold 2 and draft_depth0 1 refused; N's set applied) ->
#     warm-up answers -> DECODE BLOCKS N N N N (XST_ORDER0) -> reset + the 136k cold prefill -> identity (copy + greedy set).
#   START 2, THE NST 5 STACK ARM (A): gg_gemv 2 (lever D2's multi-row form), aql_acq 2 (rule-w), ds41_kvfold 1 / ds41_ofold 1 /
#     ds41_onefill 1 (lever K1 folds c / d / f), DSpark NST 5 --draft-window adaptive (draft_adaptive 1), draft_depth0 1 (D7),
#     draft_hybrid HYB (default 0; match XST_MATCH = 8).   STACK_ARMS=both adds arm B = the same with draft_hybrid 1, A B B A.
# START 2 (= dsm-hybrid3.sh's DSpark start + lever K1's env; start 1 the same at NST 0 without --draft-window): EXL3 2.25, TP4, MML 524288, the t3 env and serve line of
#   dsm-draft.sh (105's t3 arm + lib-29 best_vars + MNBT 4160 + --live), NST 5 --draft-window adaptive; RADIANCE_DS41_GG_GEMV=2,
#   RADIANCE_AQL_ACQ=rule-w and RADIANCE_DS41_KVFOLD=1 / _OFOLD=1 / _ONEFILL=1 AT START (the folds must be declared at start;
#   the yaml ds41-stack-tp4.yaml passes them); the scheduler switches go in process (POST /admin/fn4), every arm switch pinned
#   in every block.
#   gate (70 ms) -> modes (the t3 notes, ARGATHER 4-row note, no shared-expert-lane note, rule-w in every recorded pass, the
#   lever K1 note: all three folds declared at > 0 layers and on) -> the DSpark / adaptive lines, routed experts in VRAM -> the
#   switch note (GET at start; ds41_kvfold 2 and draft_hybrid 2 refused; the stack set applied at a step, B's too in both
#   mode) -> warm-up answers vs stock -> DECODE BLOCKS (ggblocksdz2.py, read only; XST_ORDER, default AAAA for one arm, ABBA
#   for both; epoch_bump every block): stepprobe essay,code 512 (c1), essay by draft position, editprobe (~2.5k-token region
#   re-emitted, 1500 tok), mb c4 / c8 (XST_DUR 75 s, warm 10), the draft counters around each phase
#   -> PREFILL (after decode): POST /reset_prefix_cache, then ONE 136k COLD PREFILL -- deepq.py's fixed deep prompt (136216
#   tokens, the resume item's d1), 512 greedy tokens: TTFT (cached must be 0) + its answer vs runs/xdsm/deep-dsm-225.txt
#   -> IDENTITY under each arm (reset first): the 28k copy (a ~31k-token cold prompt) and the 20-prompt greedy set at c1 vs
#   bz's d0 (runs/xdsm/draft, NST 0 stock) -- the references depth0b / hybrid3 use -- byte for byte.
#   -> THE SUMMARY (after both starts): per arm and metric value (mean +- t95 over blocks), the reference, % difference,
#   distance to the target; one final "RESULT xst summary:" line -- per metric the NST 0 stack, the NST 5 stack and the best
#   of the two, each with the reference, % and the distance to the target -- then "RESULT xst <item>: done".
# THE EXACT VRAM ENV (10.11. evening, ds-main-port VRAM agent): XST_VRAMENV (NAME=VALUE ..., default RADIANCE_PLACE_TURNOVER=0;
#   empty = none) is exported at BOTH starts after the stack env. Turnover 0 = the mover keeps no lent slot free for swaps
#   (engine_bringup.cpp mover_turnover_from_env -> MoverConfig::turnover_slots -> mover.cpp n_turn): placement only, +32 routed
#   units a rank at NST 0 (+48 at NST 5: three slab classes), vramsweep T0 identity IDENTICAL (28k copy + 20 greedy c1 vs bz d0,
#   T150 vs T0). The other two audit levers named "env" have NO knob in this engine: shadow slots = RAD_PLACE_SHADOW_SLOTS
#   (core/place/planner.h:170, compile time) and the expert prefetch is on wherever a DMA engine is spare (engine_bringup.cpp
#   1842 / 4218, off only under --deterministic) -- both need code (item hxz-vramstack takes them as arms). The modes line
#   prints the mover's shadow / kept-free counts and the prefetch state, checks the env reached the container and, with
#   turnover 0, that every rank keeps 0 free. Each start writes <tag>-meta.txt (image id, VRAM env, NST, arms) for
#   hxz-vramstack, which reuses these starts as its OFF side when the image and the env match.
# REFERENCES (no re-run): NST 0 + gg_gemv 2 = gz-gemvacq's gg2-gemv2 B blocks (runs/xdsm/gg2/gg2-gemv2-blocks.jsonl: 1-user essay
#   62.5, code 64.7, c4 131.7 (p50 35.1), c8 170.0 (p50 22.3); the reference depth0b uses); 1-user edit at NST 0 from hybrid3's
#   N blocks when present (runs/xdsm/hybrid3/h2-nst0-blocks.jsonl), else none; 136k cold prefill TTFT = item a's dsm-225
#   (runs/xdsm/deep-dsm-225.txt.json, 51.56 s).  TARGETS: 1 user >= 90 tok/s, c4 >= 160, c8 >= 200, 136k prefill <= 35 s.
# ~45 min (evict ~1; NST 0 start ~20: start ~4, blocks ~13, prefill + identity ~3; NST 5 start ~22: start ~5, blocks ~14,
# prefill + identity ~3), ~47 min with both NST 5 arms; XST_NST0=0: ~23 / ~25 min. NO cc_sim (user 10.11.). No power / host /
# fan settings; production compose files read only (--project-directory). Exact levers only; nothing goes into production here.
# XST_NST0 (1 | 0), STACK_ARMS (stack | both), HYB (0 | 1: arm A's draft_hybrid), XST_ORDER, XST_ORDER0 (N blocks, default
# NNNN), XST_DUR, XST_MATCH, XST_IMAGE, XST_ITEM, XST_VRAMENV, XST_DRAFTER_MX.
# XST_DRAFTER_MX (0 | 1; default 0 = as before; 10.11.2026 evening, ds41-int2): 1 adds "drafter_mx": 1 (lever DK, the DSpark
#   drafter's live-experts-only MXFP4 grouped GEMM, libr4d's fn4 switch, nothing declared at start; ds41-drk a22a5df) to the NST 5
#   start's in-process pin (FIX; NST 0 has no drafter); same bits (hq-drafterkern identity IDENTICAL), so only time moves. It
#   needs an image whose registry has drafter_mx (ds41-int2: seventeen switches) -- checked with the image. Each start's
#   <tag>-meta.txt records its pin (pin=, sorted compact JSON), which hxz-vramstack compares before reusing a start.
# DRYRUN=1 bash dsm-stack.sh : prints every command and checks only that the image is on this host (docker image inspect + a
#   no-device, no-network grep of its files); no server, no GPU, nothing written.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-stack-tp4.yaml
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=$R/xdsm/stack; export STATSDIR=$O/stats
RD=$R/xdsm/draft                                # bz's d0 files (identity reference: copy + greedy set)
RG=$R/xdsm/gg2                                  # gz-gemvacq's files (NST 0, gg_gemv 2 blocks: the speed reference)
RM=$R/xdsm                                      # item a's dsm-225: deep-dsm-225.txt(.json) (136k cold reference)
RH=$R/xdsm/hybrid3                              # hybrid3's NST 0 blocks (the 1-user edit reference, when present)
export IMAGE=${XST_IMAGE:-radiance-local:ds41-stack-fe030c3b-5c13e555-l}
ITEM=${XST_ITEM:-57-p4zyzzzzz-ds-hx-stack}; ARMS=${STACK_ARMS:-stack}; HYB=${HYB:-0}; DUR=${XST_DUR:-75}; MATCH=${XST_MATCH:-8}
DMX=${XST_DRAFTER_MX:-0}
case $DMX in 0|1) ;; *) echo "XST_DRAFTER_MX must be 0 or 1 (got '$DMX')"; exit 1 ;; esac
case $ARMS in
  stack) ORDER=${XST_ORDER:-AAAA} ;;
  both)  ORDER=${XST_ORDER:-ABBA}; [ "$HYB" = 1 ] && { echo "STACK_ARMS=both runs A (draft_hybrid HYB) against B (draft_hybrid 1): HYB=1 makes them the same arm"; exit 1; } ;;
  *) echo "STACK_ARMS must be stack or both (got '$ARMS')"; exit 1 ;;
esac
case $HYB in 0|1) ;; *) echo "HYB must be 0 or 1 (got '$HYB')"; exit 1 ;; esac
case $ORDER in *[!AB]*|"") echo "XST_ORDER takes A / B blocks (got '$ORDER')"; exit 1 ;; esac
[ $ARMS = stack ] && case $ORDER in *B*) echo "XST_ORDER '$ORDER' names arm B, which runs only with STACK_ARMS=both"; exit 1 ;; esac
NST0=${XST_NST0:-1}; ORDER0=${XST_ORDER0:-NNNN}
case $NST0 in 0|1) ;; *) echo "XST_NST0 must be 1 (the NST 0 start included) or 0 (got '$NST0')"; exit 1 ;; esac
case $ORDER0 in *[!N]*|"") echo "XST_ORDER0 takes N blocks (got '$ORDER0')"; exit 1 ;; esac
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
EP=$HD/editprobe.py; GGB=$HD/ggblocksdz2.py; DQ=$QA/deepq.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
FIX='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": '"$MATCH"'}'
[ "$DMX" = 1 ] && FIX="${FIX%\}}, \"drafter_mx\": 1}"   # lever DK in the NST 5 pin (XST_DRAFTER_MX=1)
ARM_A='{"draft_hybrid": '"$HYB"'}'
ARM_B='{"draft_hybrid": 1}'
FIX0='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1}'   # the NST 0 start's pin (arm N = {})
armset() {   # armset ARMJSON [EXTRAJSON]: FIX + the arm's switches (+ extra, e.g. epoch_bump) as one POST body; arm N: FIX0 + extra
  local x=${2:-} base=$FIX; [ -n "$x" ] || x='{}'
  [ "$1" = N ] && { base=$FIX0; set -- '{}'; }
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update(json.loads(sys.argv[2])); d.update(json.loads(sys.argv[3])); print(json.dumps(d))' "$base" "$1" "$x"
}
DESC_N="NST 0 stack N: gg_gemv 2, rule-w, K1 folds c/d/f, no drafter"
DESC_A="NST 5 stack A: gg_gemv 2, rule-w, K1 folds c/d/f, DSpark NST 5 adaptive, depth 0, hybrid $HYB"
DESC_B="NST 5 stack B: the same with hybrid 1 (match $MATCH)"
[ "$DMX" = 1 ] && DESC_A="$DESC_A, drafter_mx 1"
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DS41_ARGATHER_MAX RADIANCE_DS41_SHEXP_LANE RADIANCE_DEBUG_MISS_LOG
  RADIANCE_DS41_KVFOLD RADIANCE_DS41_OFOLD RADIANCE_DS41_ONEFILL RADIANCE_DS41_PIPE
  RADIANCE_DRAFT_DEPTH0 RADIANCE_DRAFT_PROBE RADIANCE_DRAFT_HYBRID RADIANCE_DRAFT_HYBRID_MATCH"
# the exact VRAM env at both starts (empty = none): each word NAME=VALUE, NAME one of $SWITCHES (unset first, passed by the yaml)
VRAMENV=${XST_VRAMENV-RADIANCE_PLACE_TURNOVER=0}
for kv in $VRAMENV; do
  case "$kv" in [A-Z]*=*) ;; *) echo "XST_VRAMENV takes NAME=VALUE words (got '$kv')"; exit 1 ;; esac
  case " $(echo $SWITCHES) " in *" ${kv%%=*} "*) ;; *) echo "XST_VRAMENV: ${kv%%=*} is not in the stack's switch list (unset at start, passed by the yaml)"; exit 1 ;; esac
done
[ -n "$VRAMENV" ] && { DESC_N="$DESC_N, VRAM env $VRAMENV"; DESC_A="$DESC_A, VRAM env $VRAMENV"; }
stackenv() {   # 105's t3 arm + gg_gemv 2 + rule-w + the three K1 folds declared (and on) at start + XST_VRAMENV, every other switch unset
  local kv
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  for kv in $VRAMENV; do export "$kv"; done
}
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
  log "RESULT xst guard $tag: $st, 1-user ${ms} ms/step (limit $lim; NST 5 DSpark starts depth0b / hybrid 35.2-35.3, NST 0 bz dsd-d0 15.59)"
  case $st in SLOW*) return 2 ;; esac
}
writemeta() {   # writemeta TAG n0|n5: what this start ran, for hxz-vramstack's OFF-side reuse check (image id, VRAM env, NST, arms,
  # the start's in-process pin: N's at NST 0, A's at NST 5, sorted compact JSON -- drafter_mx included when pinned)
  local p; p=$(python3 -c 'import json,sys; print(json.dumps(json.loads(sys.argv[1]), sort_keys=True, separators=(",", ":")))' "$([ $2 = n0 ] && armset N || armset "$ARM_A")")
  local m="image=$(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null) vramenv=$(echo $VRAMENV | tr ' ' ',') nst=$2 arms=$ARMS hyb=$HYB match=$MATCH dur=$DUR pin=$p epoch=$(date +%s) item=$ITEM"
  [ "$DRY" = 1 ] && { echo "+ echo '$m' > $O/$1-meta.txt"; return 0; }
  echo "$m" > $O/$1-meta.txt
}
FOLLOW_PID=""
follow() { [ "$DRY" = 1 ] && { echo "+ docker logs -f $CN > $1/engine.log &"; return; }
  docker logs -f $CN > $1/engine.log 2>&1 & FOLLOW_PID=$!; sleep 2; }
unfollow() { [ -n "$FOLLOW_PID" ] && { kill $FOLLOW_PID 2>/dev/null; wait $FOLLOW_PID 2>/dev/null; FOLLOW_PID=""; }; return 0; }
trap 'unfollow; down' EXIT
trap 'exit 143' TERM INT
thermline() {   # thermline T0 T1: the window's full-power rows (heatsoak.sh thermwin) or the cards now
  type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"
}
cmpfile() {   # cmpfile A B -> IDENTICAL (n chars) | DIFFERENT at char k (na vs nb chars) | not compared
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xst $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xst $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  echo "warm-up-$1 $r" >> $O/identity.txt
  case $r in IDENTICAL*) log "RESULT xst $1 warm-up answers vs stock ($(basename $2)): $r (105 tuned3l: IDENTICAL)" ;;
    *) log "RESULT xst $1 warm-up answers vs stock ($(basename $2)): VIRHE $r -- NOT EXACT vs the tuned3l line" ;; esac
}
VKNOBS=""
vrammodes() {   # vrammodes TAG: the VRAM env in the container's env, the mover's shadow / kept-free counts a rank, the expert
  # prefetch state -> VKNOBS (the modes line prints it); 1 when the env is missing, a rank has no mover line, or turnover 0
  # was asked and a rank still keeps slots free
  local MV XP CE kv bad="" nz
  MV=$(docker logs $CN 2>&1 | grep -E '^I\[[0-9]\] mover: [0-9]+ slab slots' | sed -E 's/^I\[([0-9])\] mover: ([0-9]+) slab slots \(([0-9]+) shadow, ([0-9]+) more kept free.*/r\1 \3 shadow \4 kept free (\2 slots)/' | tr '\n' ',' | sed 's/,$//')
  XP=$(docker logs $CN 2>&1 | grep -cE '^I rank [0-9]+: expert prefetch on')
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in $VRAMENV; do echo "$CE" | grep -qxF "$kv" || bad="$bad ${kv%%=*}"; done
  VKNOBS="env [${VRAMENV:-none}]$([ -n "$bad" ] && echo " NOT in the container:$bad"); mover ${MV:-no line}; expert prefetch on at $XP of 4 ranks"
  [ -z "$bad" ] || { log "RESULT xst $1: VIRHE the VRAM env did not reach the container:$bad"; return 1; }
  [ "$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots')" = 4 ] || { log "RESULT xst $1: VIRHE no mover line on all 4 ranks: ${MV:-none}"; return 1; }
  case " $VRAMENV " in *" RADIANCE_PLACE_TURNOVER=0 "*)
    nz=$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots \([0-9]+ shadow, 0 more kept free')
    [ "$nz" = 4 ] || { log "RESULT xst $1: VIRHE turnover 0 asked but $nz of 4 ranks keep 0 slots free: $MV"; return 1; } ;;
  esac
  return 0
}
modes() {   # modes TAG: the t3 line's notes + rule-w + the lever K1 note (all three declared and on) + the VRAM knobs -> 0, else 1 (start stopped)
  [ "$DRY" = 1 ] && { echo "+ modes $1: indexer note (mxfp4, latent bf16, compressor tail bound, router selection+sort separate, AR_PCN separate at 0 layers), ARGATHER note up to 4, no shared-expert-lane note, staging line, rule-w in every recorded pass (no plain rule), the lever K1 note (kvfold / ofold at > 0 layers, ones at > 0 MoE layers, ds41_kvfold=1 ds41_ofold=1 ds41_onefill=1); VRAM env [${VRAMENV:-none}] in the container's env, the mover's shadow / kept-free counts on 4 ranks$(case " $VRAMENV " in *" RADIANCE_PLACE_TURNOVER=0 "*) echo " (each '0 more kept free')" ;; esac), expert prefetch state"; return 0; }
  local MLF SL AL XL KL IL NW NR ok=1
  vrammodes $1 || ok=0
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  AL=$(docker logs $CN 2>&1 | grep -m1 'the MoE gather and its all-reduce as one kernel' | sed 's/^radiance: //' | cut -c1-300)
  XL=$(docker logs $CN 2>&1 | grep -m1 'the shared expert on the second lane' | sed 's/^radiance: //' | cut -c1-300)
  KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-500)
  IL=$(grep -iE 'interleave' $O/start-$1.log 2>/dev/null | sed 's/^[A-Z] *//' | tr -s ' ' | head -3 | tr '\n' '|' | cut -c1-300)
  log "RESULT xst $1 modes: $(echo "$MLF" | cut -c1-900) | staging: ${SL:-none} | argather: ${AL:-none} | K1: ${KL:-none} | interleave: ${IL:-no line} | VRAM: ${VKNOBS:-none}"
  log "RESULT xst $1 prefix resume fix (R1, ds41-resfix cee0ee3): $(docker logs $CN 2>&1 | grep -c "later publisher's bytes win") startup line(s) naming it (information); hits that took a later publisher's bytes so far: $(docker logs $CN 2>&1 | grep -c 'take a later') line(s)"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate" "post-collapse-norm separate at 0 layers" "next collapse-norm separate at 0 layers"; do
    case "$MLF" in *"$w"*) ;; *) log "RESULT xst $1: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  case "$AL" in *"up to 4 tokens a step (RADIANCE_DS41_ARGATHER)"*) ;; *) log "RESULT xst $1: VIRHE the ARGATHER note is not today's 4-row one: '${AL}'"; ok=0 ;; esac
  [ -z "$XL" ] || { log "RESULT xst $1: VIRHE a shared-expert-lane note (RADIANCE_DS41_SHEXP_LANE should be unset): '$XL'"; ok=0; }
  [ -n "$SL" ] || { log "RESULT xst $1: VIRHE no prefill staging line (levels refused?)"; ok=0; }
  case "$KL" in
    *"in one launch at 0 of"*|*"rope+round trip at 0 ("*|*"not declared at"*|*"at 0 MoE layers"*|"")
      log "RESULT xst $1: VIRHE the lever K1 note does not show all three folds declared: '${KL}'"; ok=0 ;;
  esac
  case "$KL" in *"ds41_kvfold=1"*"ds41_ofold=1"*"ds41_onefill=1"*) ;; *) log "RESULT xst $1: VIRHE the K1 switches do not start at 1: '${KL}'"; ok=0 ;; esac
  # rule-w from the start (RADIANCE_AQL_ACQ=rule-w): every recorded pass says so; plain `rule` must not appear
  NW=$(docker logs $CN 2>&1 | grep -c 'the acquire rule plays .*RADIANCE_AQL_ACQ=rule-w)'); NR=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_AQL_ACQ=rule)')
  log "RESULT xst $1 acquire rule: $NW recorded-pass line(s) with RADIANCE_AQL_ACQ=rule-w, $NR with rule"
  [ "$NW" -gt 0 ] && [ "$NR" = 0 ] || { log "RESULT xst $1: VIRHE rule-w is not what the recorded passes play"; ok=0; }
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
draftlines() {   # draftlines TAG: every drafter / speculation line of the serve log, logged
  [ "$DRY" = 1 ] && { echo "+ RESULT xst $1 drafter lines"; return 0; }
  log "RESULT xst $1 drafter / speculation lines: $(docker logs $CN 2>&1 | grep -iE 'dspark|drafter|speculat|draft-window|draft window|draft depth|draft hybrid|verify' | sed 's/^[A-Z][^ ]* //' | sort -u | head -10 | tr '\n' '|' | cut -c1-1600)"
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
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
fn4code() {   # fn4code JSON: POST, print the HTTP status
  [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null
}
pcreset() {   # pcreset TAG: POST /reset_prefix_cache (loopback operator) so the next prompt is prefilled cold
  [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o $O/$1-reset.json -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  log "RESULT xst $1 prefix cache reset: HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"
}
swnote0() {   # swnote0 TAG: NST 0: GET at start (gg_gemv 2, aql_acq 2, the K1 three 1, no drafter); ds41_kvfold 2 and draft_depth0 1 refused; N's set applied
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (expect gg_gemv 2, aql_acq 2, ds41_kvfold / ds41_ofold / ds41_onefill 1, draft_capable 0, epoch 0); POST ds41_kvfold 2 -> 400, draft_depth0 1 -> 400; POST N's set $(armset N '{"epoch_bump": 1}') + a 4-token request -> applied"; return 0; }
  local s0 s1 c2 c3
  s0=$(fn4get); c2=$(fn4code '{"ds41_kvfold": 2}'); c3=$(fn4code '{"draft_depth0": 1}')
  fn4set "$(armset N '{"epoch_bump": 1}')"; nudge; s1=$(fn4get)
  log "RESULT xst $1 switch note: at start $s0 | ds41_kvfold 2 -> HTTP $c2, draft_depth0 1 -> HTTP $c3 | after N's set: $s1 | core log: $(docker logs $CN 2>&1 | grep -E 'fn4: switches' | tail -2 | sed 's/^[A-Z] *//' | tr '\n' '|' | cut -c1-400)"
  python3 -c 'import json,sys
s0, s1 = json.loads(sys.argv[1]), json.loads(sys.argv[2]); n = json.loads(sys.argv[3]); w = s0["switches"]
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) \
     and s0["draft_capable"] == 0 and s0["epoch"] == 0
ok = ok and all(s1["switches"].get(k) == v for k, v in n.items()) and not s1["pending"] and s1["epoch"] >= 1
sys.exit(0 if ok else 1)' "$s0" "$s1" "$(armset N)" 2>/dev/null && [ "$c2" = 400 ] && [ "$c3" = 400 ]
}
swnote() {   # swnote TAG: GET at start; ds41_kvfold 2 / draft_hybrid 2 refused; A's set (and B's in both mode) applied at a step
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (expect gg_gemv 2, aql_acq 2, ds41_kvfold / ds41_ofold / ds41_onefill 1, ds41_argather 4, draft_adaptive 1, draft_depth0 0, draft_hybrid 0, draft_capable 1, epoch 0); POST ds41_kvfold 2 -> 400, draft_hybrid 2 -> 400; POST A's set $(armset "$ARM_A")$([ $ARMS = both ] && echo " then B's $(armset "$ARM_B")"), each + a 4-token request -> applied"; return 0; }
  local s0 s1 s2 c2 c3
  s0=$(fn4get); c2=$(fn4code '{"ds41_kvfold": 2}'); c3=$(fn4code '{"draft_hybrid": 2}')
  fn4set "$(armset "$ARM_A")"; nudge; s1=$(fn4get)
  if [ $ARMS = both ]; then fn4set "$(armset "$ARM_B")"; nudge; s2=$(fn4get); else s2=$s1; fi
  log "RESULT xst $1 switch note: at start $s0 | ds41_kvfold 2 -> HTTP $c2, draft_hybrid 2 -> HTTP $c3 | after A's set: $s1$([ $ARMS = both ] && echo " | after B's set: $s2") | core log: $(docker logs $CN 2>&1 | grep -E 'fn4: switches' | tail -2 | sed 's/^[A-Z] *//' | tr '\n' '|' | cut -c1-400)"
  python3 -c 'import json,sys
s0, s1, s2 = (json.loads(x) for x in sys.argv[1:4]); a, b = json.loads(sys.argv[4]), json.loads(sys.argv[5]); both = sys.argv[6] == "both"
w = s0["switches"]
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) \
     and w["draft_adaptive"] == 1 and w["draft_depth0"] == 0 and w["draft_hybrid"] == 0 and s0["draft_capable"] == 1 and s0["epoch"] == 0
ok = ok and all(s1["switches"].get(k) == v for k, v in a.items()) and not s1["pending"] and s1["epoch"] >= 1
if both: ok = ok and all(s2["switches"].get(k) == v for k, v in b.items()) and not s2["pending"] and s2["epoch"] > s1["epoch"]
sys.exit(0 if ok else 1)' "$s0" "$s1" "$s2" "$(armset "$ARM_A")" "$(armset "$ARM_B")" $ARMS 2>/dev/null && [ "$c2" = 400 ] && [ "$c3" = 400 ]
}
blocks() {   # blocks TAG n0|n5: the decode blocks (ggblocksdz2.py, read only) and their lines
  local args="--arm 'A=$ARM_A'" ord=$ORDER fx=$FIX
  [ $ARMS = both ] && args="$args --arm 'B=$ARM_B' --pair A,B"
  [ $2 = n0 ] && { args="--arm 'N={}'"; ord=$ORDER0; fx=$FIX0; }
  run "timeout 5400 python3 -I $GGB --url $URL --mb-dir $D --edit $EP --label $1 $args --order $ord --dur $DUR --warm 10 --fixed '$fx' --out $O/$1-blocks.jsonl > $O/$1-blocks.txt 2>&1"
  [ "$DRY" = 1 ] && return 0
  while IFS= read -r l; do log "RESULT xst $1 ${l:0:1500}"; done < <(grep -E '^(BLOCK|GGB \{"error)' $O/$1-blocks.txt)
}
deep() {   # deep TAG ARM: the 136k cold prefill (deepq.py's fixed prompt, 512 greedy tokens) -> TTFT, cached, its answer vs dsm-225
  local t0=$(date +%s)
  pcreset $1-deep
  run "(cd $D && timeout 1800 python3 $DQ $O/deep-$1.txt 135000 512 > $O/deep-$1.out 2>&1)"
  [ "$DRY" = 1 ] && { echo "+ RESULT xst $1 136k cold prefill: TTFT (cached must be 0) vs $RM/deep-dsm-225.txt.json; answer vs $RM/deep-dsm-225.txt"; return 0; }
  local j r; j=$(grep -m1 '^DEEPQ ' $O/deep-$1.out | cut -c7-)
  [ -n "$j" ] || { log "RESULT xst $1 136k cold prefill: VIRHE no answer ($(tail -2 $O/deep-$1.out | tr '\n' ' ' | cut -c1-300))"; return 1; }
  echo "$j" > $O/deep-$1.json
  log "RESULT xst $1 136k cold prefill (deepq.py fixed prompt, 512 greedy tokens): $(python3 -c 'import json,sys
d = json.loads(sys.argv[1])
try: r = json.load(open(sys.argv[2]))
except Exception: r = {}
t, rt = d.get("ttft"), r.get("ttft")
print("status %s, prompt %s tok, cached %s%s, TTFT %s s (dsm-225 cold %s s%s), %s prefill tok/s, %s tok, %s chars" % (
    d.get("status"), d.get("prompt_tokens"), d.get("cached"), "" if d.get("cached") == 0 else " -- VIRHE not cold, the TTFT is not a cold prefill",
    t, rt, ", %+.1f %%" % ((t / rt - 1) * 100) if t and rt else "", round(d["prompt_tokens"] / t, 1) if t and d.get("prompt_tokens") else "-",
    d.get("tokens"), d.get("chars")))' "$j" $RM/deep-dsm-225.txt.json 2>&1 | cut -c1-400) | $(thermline $t0 $(date +%s))"
  r=$(cmpfile $O/deep-$1.txt $RM/deep-dsm-225.txt)
  echo "deep-$2 $r" >> $O/identity.txt
  case $r in IDENTICAL*) log "RESULT xst $1 identity: 136k deep answer (cold) vs dsm-225 (NST 0, stock, cold): $r" ;;
    *) log "RESULT xst $1 identity: DIFFERENT 136k deep answer (cold) vs dsm-225 (NST 0, stock, cold): $r -- reported, not hidden" ;; esac
}
copyt() {   # copyt TAG OFFSET: the copy test after a ~31k-token cold prompt (105's prompt: same offset -> same text)
  local t0=$(date +%s)
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset $2 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xst $1 copy after a cold prompt: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300) | $(thermline $t0 $(date +%s))"
}
gsetc1() { run "(cd $D && timeout 900 python3 $GS $URL $O/$1-gs-c1.jsonl --conc 1 --max-tokens 192 > $O/$1-gs-c1.txt 2>&1)"; }
gcmp() { [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }; [ -s "$1" ] && [ -s "$2" ] || { echo "not compared (missing $1 or $2)"; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
ident() {   # ident TAG ARM WHAT: the arm's set + epoch_bump, a prefix-cache reset, the 28k copy and the c1 greedy set vs bz d0
  local tag=$1 arm=$2 r
  fn4set "$(armset "$(case $arm in B) echo "$ARM_B" ;; N) echo N ;; *) echo "$ARM_A" ;; esac)" '{"epoch_bump": 1}')"
  pcreset $tag
  copyt $tag 0; gsetc1 $tag
  [ "$DRY" = 1 ] && { echo "+ identity ($3): 28k copy vs bz d0, greedy set c1 vs bz d0"; return 0; }
  log "RESULT xst $tag identity ran under: $(fn4get | cut -c1-600)"
  r=$(cmpfile $O/copy-$tag-run1.txt $RD/copy-dsd-d0-i1-run1.txt)
  echo "copy-$arm $r" >> $O/identity.txt
  case $r in IDENTICAL*) log "RESULT xst $tag identity ($3): 28k copy vs bz d0 (NST 0, stock): $r" ;;
    *) log "RESULT xst $tag identity ($3): DIFFERENT 28k copy vs bz d0 (NST 0, stock): $r -- reported, not hidden" ;; esac
  r=$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$tag-gs-c1.jsonl)
  echo "gs-$arm $r" >> $O/identity.txt
  case $r in IDENTICAL*) log "RESULT xst $tag identity ($3): greedy set c1 vs bz d0: $r" ;;
    *) log "RESULT xst $tag identity ($3): DIFFERENT greedy set c1 vs bz d0: $r -- reported, not hidden" ;; esac
}
summary() {   # summary: per arm and metric value / reference / % / distance to target; B vs A pairs in both mode; the final line
  [ "$DRY" = 1 ] && { echo "+ RESULT xst arm lines N / A$([ $ARMS = both ] && echo " / B") (value +- t95, ref, %, target distance), prefill lines, RESULT xst summary: per metric NST 0 stack / NST 5 stack / best of the two, each with ref, % and target distance (1u essay / code / edit, c4 agg + p50, c8 agg + p50, 136k TTFT), identity"; return 0; }
  python3 - $O/st-stack-blocks.jsonl $O/st-nst0-blocks.jsonl $RG/gg2-gemv2-blocks.jsonl $RH/h2-nst0-blocks.jsonl $O/deep-st-stack.json $O/deep-st-nst0.json $RM/deep-dsm-225.txt.json $O/identity.txt "$ARMS" "$NST0" "$DESC_N" "$DESC_A" "$DESC_B" > $O/summary.txt 2>&1 <<'PY'
import json, math, statistics, sys
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
def rows(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except OSError: return []
def js(p):
    try: return json.load(open(p))
    except Exception: return {}
n5, n0, gz, h2 = (rows(p) for p in sys.argv[1:5])
deep5, deep0, dref = js(sys.argv[5]), js(sys.argv[6]), js(sys.argv[7])
try: ident = [l.strip() for l in open(sys.argv[8]) if l.strip()]
except OSError: ident = []
arms, nst0 = sys.argv[9], sys.argv[10] == "1"
desc = {"N": sys.argv[11], "A": sys.argv[12], "B": sys.argv[13]}
src = {"N": n0, "A": n5, "B": n5}
def num(x): return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None
def c1(name, k):   # ggblocksdz2 (c1 keyed by prompt) and the older ggblocks (c1.rows)
    def f(res):
        c = res.get("c1") or {}
        if isinstance(c.get(name), dict): return num(c[name].get(k))
        return num(next((x.get(k) for x in (c.get("rows") or []) if x.get("prompt") == name), None))
    return f
def ph(p, k): return lambda res: num((res.get(p) or {}).get(k))
def vals(rs, arm, get): return [v for v in (get(r["res"]) for r in rs if r.get("ok") and r.get("arm") == arm) if v is not None]
def ci(v):
    if not v: return None, None
    m = statistics.mean(v)
    return m, (T95.get(len(v) - 1, 1.96) * statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else None)
# (name, getter, reference rows + arm, hard reference if the file is missing, target)
M = [("1u essay", c1("essay", "decode_tps"), (gz, "B"), 62.5, 90.0),
     ("1u code", c1("code", "decode_tps"), (gz, "B"), 64.7, 90.0),
     ("1u edit", ph("edit", "decode_tps"), (h2, "N"), None, 90.0),
     ("c4 aggregate", ph("c4", "agg_dec_tps"), (gz, "B"), 131.7, 160.0),
     ("c4 p50", ph("c4", "per_stream_dec_p50"), (gz, "B"), 35.1, None),
     ("c8 aggregate", ph("c8", "agg_dec_tps"), (gz, "B"), 170.0, 200.0),
     ("c8 p50", ph("c8", "per_stream_dec_p50"), (gz, "B"), 22.3, None)]
EXTRA = [("tok/step", {"1u essay": c1("essay", "tokens_per_step"), "1u code": c1("code", "tokens_per_step"), "1u edit": ph("edit", "tokens_per_step"),
                       "c4 aggregate": ph("c4", "tok_per_decode_step"), "c8 aggregate": ph("c8", "tok_per_decode_step")}),
         ("acc", {"1u essay": c1("essay", "acceptance"), "1u code": c1("code", "acceptance"), "1u edit": ph("edit", "acceptance"),
                  "c4 aggregate": ph("c4", "acceptance"), "c8 aggregate": ph("c8", "acceptance")}),
         ("ms/step", {"1u essay": c1("essay", "ms_per_decode_step"), "1u code": c1("code", "ms_per_decode_step"), "1u edit": ph("edit", "ms_per_decode_step"),
                      "c4 aggregate": ph("c4", "ms_decode_step"), "c8 aggregate": ph("c8", "ms_decode_step")})]
def f1(x, n=1): return "-" if x is None else format(x, ".%df" % n)
def pct(v, rv): return ("%+.1f %%" % ((v / rv - 1) * 100)) if v and rv else "-"
def tgt(v, t, low=False):
    if t is None or v is None: return ""
    if low:
        return "; target <= %g s: %s" % (t, "MET (%.1f s under)" % (t - v) if v <= t else "%.1f s over, %.1f %% less needed" % (v - t, (1 - t / v) * 100))
    return "; target >= %g: %s" % (t, "MET (+%.1f)" % (v - t) if v >= t else "%.1f short, +%.1f %% needed" % (t - v, (t / v - 1) * 100))
def one(v, rv, t, low=False): return "%s (ref %s, %s%s)" % (f1(v, 2 if low else 1), f1(rv, 2 if low else 1), pct(v, rv), tgt(v, t, low))
def ref(get, sref, hard):
    rs, arm = sref
    v = vals(rs, arm, get)
    return statistics.mean(v) if v else hard
present = [a for a in ("N", "A", "B") if any(r.get("arm") == a for r in src[a])]
cell, refs = {}, {}
for a in present:
    rs, parts = src[a], []
    for nm, get, sref, hard, t in M:
        v = vals(rs, a, get); m, h = ci(v)
        rv = ref(get, sref, hard); cell[(a, nm)] = m; refs[nm] = rv
        ex = ", ".join("%s %s" % (f1(statistics.mean(vals(rs, a, g[nm])) if vals(rs, a, g[nm]) else None, 2), lab) for lab, g in EXTRA if nm in g)
        parts.append("%s %s%s (n %d%s) vs ref %s (%s)%s" % (nm, f1(m), "" if h is None else " +- " + f1(h), len(v), ", " + ex if ex else "",
                     f1(rv), pct(m, rv), tgt(m, t)))
    print("ARM %s [%s]: %s" % (a, desc[a], "; ".join(parts)))
pairs = ""
if arms == "both" and "B" in present:
    out = []
    for nm, get, *_ in M:
        pr = []
        for j in range(0, len(n5) - 1, 2):
            x, y = n5[j], n5[j + 1]
            if {x.get("arm"), y.get("arm")} != {"A", "B"} or not (x.get("ok") and y.get("ok")): continue
            ra, rb = (x, y) if x["arm"] == "A" else (y, x)
            ga, gb = get(ra["res"]), get(rb["res"])
            if ga and gb is not None: pr.append(gb / ga)
        m, h = ci(pr)
        out.append("%s %s" % (nm, "-" if m is None else "%+.1f %%%s" % ((m - 1) * 100, "" if h is None else " +- %.1f" % (h * 100))))
    pairs = "NST 5 B vs A (pairs, t95): " + ", ".join(out)
    print("PAIRS " + pairs)
rt = num(dref.get("ttft"))
def cold(d): return num(d.get("ttft")) if d.get("cached") == 0 else None
for lab, d in (("NST 0 stack", deep0), ("NST 5 stack", deep5)):
    if d:
        print("PREFILL %s 136k cold prefill: TTFT %s%s, prompt %s tok, %s chars" % (lab, one(num(d.get("ttft")), rt, 35.0, True),
              "" if d.get("cached") == 0 else " [VIRHE cached %s: not cold, left out of the summary]" % d.get("cached"), d.get("prompt_tokens"), d.get("chars")))
lines = []
def best(vs, low=False):
    vs = [(k, v) for k, v in vs if v is not None]
    return (min if low else max)(vs, key=lambda kv: kv[1]) if vs else (None, None)
for nm, get, sref, hard, t in M:
    vn, va, rv = cell.get(("N", nm)), cell.get(("A", nm)), refs.get(nm, hard)
    bk, bv = best([("NST0", vn), ("NST5", va)])
    lines.append("%s: NST0 %s | NST5 %s | best %s %s" % (nm, one(vn, rv, t), one(va, rv, t), bk or "-", one(bv, rv, t)))
tn, ta = cold(deep0), cold(deep5)
bk, bv = best([("NST0", tn), ("NST5", ta)], True)
lines.append("136k cold TTFT s: NST0 %s | NST5 %s | best %s %s" % (one(tn, rt, 35.0, True), one(ta, rt, 35.0, True), bk or "-", one(bv, rt, 35.0, True)))
cnt = lambda rs: "%d blocks, %d flagged" % (len(rs), sum(1 for r in rs if not r.get("ok")))
s = " || ".join(lines) + " || identity: " + (", ".join(ident) if ident else "none recorded")
if arms == "both" and "B" in present:
    s += " || NST 5 B [%s]: %s | %s" % (desc["B"], "; ".join("%s %s" % (nm, one(cell.get(("B", nm)), refs.get(nm, hard), t)) for nm, _, _, hard, t in M), pairs)
s += " || NST0 [%s]: %s; NST5 [%s]: %s; refs NST 0 + gg_gemv 2 = gz gg2-gemv2 B blocks, 136k dsm-225%s" % (
    desc["N"], cnt(n0) if nst0 else "not run (XST_NST0=0)", desc["A"], cnt(n5), "" if h2 else "; 1u edit: no NST 0 reference yet")
print("SUMMARY " + s)
PY
  while IFS= read -r l; do case "$l" in SUMMARY*) ;; *) log "RESULT xst ${l:0:2500}" ;; esac; done < $O/summary.txt
  log "RESULT xst summary: $(grep -m1 '^SUMMARY ' $O/summary.txt | cut -c9- | cut -c1-4500)"
}
dstart() {   # dstart n0|n5: one start, its decode blocks, then the 136k cold prefill, then identity
  local s=$1 tag P rc ok lim
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  case $s in
    n0) tag=st-nst0; lim=30; export MNBT=4160 NST=0 XARGS="$XARGS --live" ;;
    n5) tag=st-stack; lim=70; export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive" ;;
  esac
  P=$O/$tag
  log "RESULT xst $tag: start ($([ $s = n0 ] && echo "$DESC_N" || echo "$DESC_A")) IMAGE=$IMAGE MODEL=$MODEL $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
  mkd $P
  gate $tag $lim; rc=$?
  [ $rc = 1 ] && { log "RESULT xst $tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  [ $rc = 2 ] && { log "RESULT xst $tag: VIRHE over the gate limit after one restart -> start skipped"; down; return 1; }
  modes $tag || { log "RESULT xst $tag: VIRHE the modes are not the t3 + rule-w + K1 line's -> start stopped"; down; return 1; }
  draftlines $tag
  if [ "$DRY" = 1 ]; then
    [ $s = n0 ] && echo "+ no DSpark drafter line at NST 0; VRAM expert units at start vs bz d0 / d5" || echo "+ DSpark drafter / adaptive depth line check; VRAM expert units at start vs bz d0 / d5"
  else
    ok=1
    if [ $s = n5 ]; then
      docker logs $CN 2>&1 | grep -q "DSpark drafter" || { log "RESULT xst $tag: VIRHE no DSpark drafter line"; ok=0; }
      docker logs $CN 2>&1 | grep -q "draft depth: adaptive" || { log "RESULT xst $tag: VIRHE no adaptive draft depth line"; ok=0; }
    else
      docker logs $CN 2>&1 | grep -qiE "DSpark drafter|'dspark' proposes" && { log "RESULT xst $tag: VIRHE a DSpark drafter line at NST 0"; ok=0; }
    fi
    curl -s -m 15 $URL/stats > $O/$tag-stats0.json 2>/dev/null
    log "RESULT xst $tag experts at start: $(vramunits $O/$tag-stats0.json) | bz d0 (NST 0): $(vramunits $RD/dsd-d0/stats-start.json) | bz d5 (NST 5): $(vramunits $RD/dsd-d5/stats-start.json)"
    [ $ok = 1 ] || { keeplog $tag-nodraft; down; return 1; }
  fi
  if [ $s = n0 ]; then swnote0 $tag; else swnote $tag; fi \
    || { log "RESULT xst $tag: VIRHE the switches are not as expected -> start stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  writemeta $tag $s
  follow $P
  stopfile || blocks $tag $s
  if [ $s = n0 ]; then
    if ! stopfile; then fn4set "$(armset N '{"epoch_bump": 1}')"; deep $tag N; fi
    stopfile || ident $tag-iN N "$DESC_N"
  else
    if ! stopfile; then fn4set "$(armset "$ARM_A" '{"epoch_bump": 1}')"; deep $tag A; fi
    stopfile || ident $tag-iA A "$DESC_A"
    [ $ARMS = both ] && ! stopfile && ident $tag-iB B "$DESC_B"
  fi
  unfollow
  run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
  down
  log "RESULT xst $tag: done"
}
# ---------------------------------------------------------------- main
log "RESULT xst $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) XST_NST0=$NST0$([ $NST0 = 1 ] && echo " (NST 0 start first: N order $ORDER0, fixed $FIX0)") | NST 5 start: STACK_ARMS=$ARMS HYB=$HYB order $ORDER (A $ARM_A$([ $ARMS = both ] && echo ", B $ARM_B"), fixed $FIX), mb $DUR s, yaml $(basename $Y), VRAM env [${VRAMENV:-none}] at both starts | refs gz-gemvacq gg2-gemv2 B (NST 0, gg_gemv 2: essay 62.5, code 64.7, c4 131.7, c8 170.0), dsm-225 deep (136k cold TTFT 51.56 s), identity bz d0 | targets 1u >= 90, c4 >= 160, c8 >= 200 tok/s, 136k <= 35 s"
# the image, on this host, read only (in DRYRUN too: docker image inspect + a no-device, no-network grep of its files; -z: the
# image's BusyBox grep reads a binary's NUL-separated strings whole, without it a literal past a NUL in a "line" is missed).
# The registry: the stack image's twelve switches or the ds41-int image's fifteen (+ draft_probe_backoff, draft_lend,
# draft_lend_borrow: ds41-dlend, unused here), whose unknown-switch message ends "draft_lend_borrow, epoch_bump" (10.11.2026),
# or the ds41-int2 image's seventeen (+ drafter_mx, drafter_bf: ds41-drk), "draft_lend_borrow, drafter_mx, drafter_bf, epoch_bump".
# XST_DRAFTER_MX=1 needs the seventeen and libr4d's drafter_mx (RADIANCE_DS41_DRAFTER_MX): a second read-only grep.
docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xst: VIRHE image $IMAGE not on this host"; exit 1; }
[ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c '{ grep -qzF "draft_hybrid_match, epoch_bump" /opt/radiance/bin/radiance || grep -qzF "draft_hybrid_match, draft_probe_backoff, draft_lend, draft_lend_borrow, epoch_bump" /opt/radiance/bin/radiance || grep -qzF "draft_hybrid_match, draft_probe_backoff, draft_lend, draft_lend_borrow, drafter_mx, drafter_bf, epoch_bump" /opt/radiance/bin/radiance; } && grep -qzF "ds41_onefill, draft_depth0, draft_probe, draft_hybrid," /opt/radiance/bin/radiance && grep -qzF "draft depth 0: a step drafts nothing" /opt/radiance/bin/radiance && grep -qzF "draft hybrid: a sequence whose prompt lookup" /opt/radiance/bin/radiance && grep -qzF reset_prefix_cache /opt/radiance/bin/radiance && grep -qzF "bytes win where no snapshot rests on the first copy" /opt/radiance/bin/radiance && grep -qzF RADIANCE_DS41_KVFOLD /opt/radiance/share/radiance/architectures/deepseek41.so && grep -qzF rmsnorm_rope_roundtrip_kv_store /opt/radiance/share/radiance/kernels/libr4d.so && grep -qzF r4d_gg_mr_pin /opt/radiance/share/radiance/kernels/libr4d.so && echo yes' 2>/dev/null)" = yes ] \
  || { log "RESULT xst: VIRHE $IMAGE is not the stack image (the twelve-, fifteen- or seventeen-switch registry, depth 0 / hybrid, reset_prefix_cache, the R1 prefix-resume fix, K1 in deepseek41.so / libr4d, lever D2)"; exit 1; }
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)) is the stack image (twelve-, fifteen- or seventeen-switch registry, depth 0 + hybrid, R1 prefix-resume fix, K1, D2) -- OK"
if [ "$DMX" = 1 ]; then
  [ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -qzF "draft_hybrid_match, draft_probe_backoff, draft_lend, draft_lend_borrow, drafter_mx, drafter_bf, epoch_bump" /opt/radiance/bin/radiance && grep -qzF RADIANCE_DS41_DRAFTER_MX /opt/radiance/share/radiance/kernels/libr4d.so && echo yes' 2>/dev/null)" = yes ] \
    || { log "RESULT xst: VIRHE XST_DRAFTER_MX=1 but $IMAGE has no drafter_mx (the seventeen-switch registry, libr4d's RADIANCE_DS41_DRAFTER_MX)"; exit 1; }
  [ "$DRY" = 1 ] && echo "+ image check: drafter_mx in the registry (seventeen switches) and in libr4d -- OK; NST 5 pin $FIX"
fi
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xst: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  [ -e $Y ] && grep -q '^ *RADIANCE_DS41_GG_GEMV:' $Y && grep -q '^ *RADIANCE_AQL_ACQ:' $Y && grep -q '^ *RADIANCE_DS41_KVFOLD:' $Y && grep -q '^ *RADIANCE_DS41_OFOLD:' $Y && grep -q '^ *RADIANCE_DS41_ONEFILL:' $Y \
    || { log "RESULT xst: VIRHE $Y missing or does not pass RADIANCE_DS41_GG_GEMV / RADIANCE_AQL_ACQ / the K1 folds"; exit 1; }
  for kv in $VRAMENV; do grep -q "^ *${kv%%=*}:" $Y || { log "RESULT xst: VIRHE $Y does not pass ${kv%%=*} (XST_VRAMENV)"; exit 1; }; done
  for f in $GS $EP $GGB $DQ $D/stepprobe.py $D/mb.py $D/ds41_tests.py; do [ -e $f ] || { log "RESULT xst: VIRHE $f missing"; exit 1; }; done
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xst: VIRHE $M225 missing or not verified"; exit 1; }
  for f in $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $RG/gg2-gemv2-blocks.jsonl $RM/deep-dsm-225.txt $RM/deep-dsm-225.txt.json; do [ -s $f ] || log "RESULT xst: note -- reference $f missing (not compared)"; done
else
  for f in $Y $GS $EP $GGB $DQ $D/stepprobe.py $D/mb.py $D/ds41_tests.py $MODELDIR/$M225 $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $RG/gg2-gemv2-blocks.jsonl $RM/deep-dsm-225.txt $RM/deep-dsm-225.txt.json; do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  for kv in ${VRAMENV:-}; do echo "+ VRAM env $kv: the yaml passes ${kv%%=*}: $(grep -c "^ *${kv%%=*}:" $Y)"; done
fi
mkd $O $O/guard $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt
evict
[ $NST0 = 1 ] && ! stopfile && dstart n0
stopfile || dstart n5
summary
unset $SWITCHES
down
log "RESULT xst $ITEM: done -- $O"
