#!/bin/bash
# 76-argather.sh (queue item, 6.11.2026 night, queue-ar) -- the MoE gather and its all-reduce in one kernel
# (moe_gather_all_reduce, RADIANCE_DS41_ARGATHER), A/B at the fin-best line with the chosen heat (dual 0.7 + pin 0.8).
# EXL3 2.25 bpw, TP4, MML 524288.
# What: at a decode step each of the 40 MoE layers runs moe_gather (2.5 us in the MTP5 stamps) and then the exact one-shot
# all-reduce, with a ~3.8 us dispatch gap between them. libr4d's ar_gather_hc_write_4rank kernel at hc 0 does both: the gather's
# words are the message, pushed and reduced in place. Same gather arithmetic (moe_gather's), same fold (fp32, ascending rank,
# one rounding) -> byte-identical; the identity check below holds it to that. Estimate: -40 dispatches a step, ~-0.15..-0.25
# ms/step at 1..4 decode rows (the fused op is declared up to 4 rows: there the unfused all-reduce is the one-shot too; 8 rows
# keeps the two kernels, so c8 is this item's control).
# Image X3ARG = radiance-local:ds41-exl3-117-ea7d3423-f2f3c3f7-idx-argather (9ae6f9f60a29): the 7e342763 heat image (the -idx
# kernels + bin/radiance of ds41-ar 7e342763) with libr4d.so + deepseek41.so of ds41-ar ea7d342 (no-GPU checks: 0 plugin
# refusals, rad-schemas lists moe_gather_all_reduce, ops_doc / plugin / shipped_plugins / arch / place tests pass, a TP1 host
# start declares the same 2202 ops).
#   arms (ARMS, default "off on"; both RADIANCE_HEAT_PREFILL_DECAY 0.7 + RADIANCE_HEAT_DECODE_PIN 0.8, fin-best, AQL stamps
#   one recorded pass in 16 into the arm's stats dir):
#     off  RADIANCE_DS41_ARGATHER unset (moe_gather + all_reduce)   on  RADIANCE_DS41_ARGATHER=on (fused up to 4 rows)
#   each: gate (23 ms); probe; fastcheck 384 x 2; mb c4; mb c8; ident (copy over ~143k, greedy); down; stampgaps.py on the
#   arm's stamps (the decode pass's wall / busy, median over its stamped passes -- the resolution a probe lacks).
# Identity: the two arms' copy outputs byte for byte (identcmp), and their warm-up answers (guard/<tag>-warm-q.txt) the same.
# Expected reading: on's 1-row decode pass wall ~0.15-0.25 ms under off's; c4 likewise per step; c8 equal (control); identical.
# 7.11. re-queue: the first run's on arm (ea7d3423) failed at its first fused launch -- DeepSeek-V4.1's expert weights are
# f32 and the fused shim took only bf16 (RAD_E_DTYPE; warm-up finish "error", then the server went down). ds41-ar
# c039395b reads f32 slot weights as moe_gather's F32 form does (fmaf). The off arm (19.96 ms, c4 96.9 tok/s 35.2 ms, c8
# 123.6 tok/s 55.5 ms, stamps wall 35.639 ms) need not run again: ARMS=on, compared against the first run's off files.
# Every arm now keeps its engine log (keeplog: engine-<tag>[-try1|-try2|-fail].log).
# ~36 min. DRYRUN=1 bash 76-argather.sh prints, writes nothing. Needs lib-ar.sh + ds41-ar-tp4.yaml (passes
# RADIANCE_DS41_ARGATHER: the updated copy) and stampgaps.py in queue/.
. "$(cd "$(dirname "$0")" && pwd)/../lib-ar.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-ar.sh" || { echo "lib-ar.sh not found"; exit 1; }
X3ARG=${X3ARG:-radiance-local:ds41-exl3-117-c039395b-f2f3c3f7-idx-argather}   # 7.11.: ds41-ar c039395b (f32 slot weights; ea7d3423 failed its on arm)
export IMAGE=$X3ARG
ARMS=${ARMS-"on"}   # 7.11. re-queue: the off arm's files are the first run's
SG=$QD/stampgaps.py; [ -e $SG ] || SG=$(cd "$(dirname "$0")" && pwd)/stampgaps.py
log "RESULT xar 76-argather: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) arms [$ARMS]"
imgcheck $IMAGE
mkd $O
evict
done_arms=""
for arm in $ARMS; do
  unset RADIANCE_DS41_ARGATHER RADIANCE_HEAT_PREFILL RADIANCE_HEAT_MOVES RADIANCE_PLACE_TURNOVER
  export RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8
  case $arm in off) ;; on) export RADIANCE_DS41_ARGATHER=on ;; *) log "RESULT xar 76: unknown arm $arm"; continue ;; esac
  t=xar-ag-$arm
  best_vars
  export STATSDIR=$O/stats-ag-$arm AQLSTAMP=16
  mkd $STATSDIR
  run "rm -f $STATSDIR/stamps.csv"
  gate $t 23; rc=$?
  [ $rc = 1 ] && { keeplog $t-fail; log "RESULT xar 76 $arm: VIRHE not healthy ($(desc)); logs $O/start-$t.log, $O/engine-$t-fail.log"; continue; }
  [ $rc = 2 ] && { log "RESULT xar 76 $arm: VIRHE slow twice -> skipped; engine logs $O/engine-$t-try1.log / -try2.log: $(grep -hE '^(E|F) |error|fault|abort' $O/engine-$t-try1.log 2>/dev/null | head -4 | tr '\n' ' ' | cut -c1-600)"; continue; }
  [ "$DRY" = 1 ] || log "RESULT xar 76 $arm mode: $(docker logs $CN 2>&1 | grep -E 'gather and its all-reduce|decode-hottest' | head -2 | cut -c1-200 | tr '\n' '|')"
  probe $t
  fcrun $t-fc1 "(1)"; fcrun $t-fc2 "(2)"
  for ph in c4 c8; do stopfile && break; mbp $t $ph; done
  stopfile || ident $t
  keeplog $t   # the arm's whole engine log (7.11.)
  down   # the stamp writer flushes at exit
  run "python3 $SG $STATSDIR/stamps.csv $t > $O/$t-stampgaps.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT xar 76 $arm stamps: $(grep -E 'pass of|wall ms|busy ms|summed gaps' $O/$t-stampgaps.txt | tr -s ' ' | tr '\n' ';' | cut -c1-700)"
  done_arms="$done_arms $arm"
  stopfile && break
done
unset AQLSTAMP RADIANCE_DS41_ARGATHER RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_DECODE_PIN
export STATSDIR=$O/stats
if [[ " $done_arms " == *" on "* ]] && { [ "$DRY" = 1 ] || [ -s $O/copy-xar-ag-off-run1.txt ]; }; then   # off: this run or the first
  identcmp xar-ag-off xar-ag-on
  [ "$DRY" = 1 ] || log "RESULT xar 76 warm-up answers off vs on: $(python3 -c 'import json,sys
rd = lambda f: {d.get("test"): (d.get("content"), d.get("finish")) for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if a.get(t) != b[t]]
print("IDENTICAL (%d answers)" % len(b) if b and not bad else "DIFFERENT: %s" % ", ".join(map(str, bad)) if b else "no answers")' $O/guard/xar-ag-off-warm-q.txt $O/guard/xar-ag-on-warm-q.txt 2>&1 | cut -c1-200)"
fi
down
log "RESULT xar 76-argather: done -- $O/xar-ag-{off,on}-* (probe, fc, c4, c8, copy, stampgaps)"
