#!/bin/bash
# 79-acqrule-tp4.sh (queue item, 6.11.2026 night) -- sizes the address-based acquire rule from the engine's own tape, and
# (optionally, ACQ_EXP=1, only after 78-cohbench) measures dropping the agent-scope acquire outright.
# Image radiance-local:ds41-exl3-117-1fb143a5-f2f3c3f7-fuse-c85088e7 (20b80d476a53): the fuse image (77) with bin/radiance of
# core-dbg 85088e7 = 9c7ce498 + RADIANCE_DEBUG_TAPE_DUMP=1 (haz_mark's TAPEDUMP lines: every recorded pass's dispatches,
# once a key, with the byte ranges each op reads and writes) + RADIANCE_AQL_ACQ=none (EXPERIMENT: a played packet's
# agent-scope acquire becomes none; system acquires stay). Both unset = the engine as before.
# A. dump: fin-best line, 2.25 bpw .rad, RADIANCE_DEBUG_TAPE_DUMP=1, gate 24 ms (stderr while recording), stepprobe (the
#    1-user decode passes) + mb c4 (the 4-row ones), engine log kept; tools/acqrule.py over it: per recorded pass the
#    ordered dispatches and how many the rule lets skip their acquire (a dispatch may skip iff none of the lines it reads
#    was both read and written since the last acquire; an op without ranges keeps it and leaves the window unknown).
#    gapbench (75): an agent acquire ~0.9 us a dispatch -> the skippable count x 0.9 us = the rule's step gain bound.
# B. (ACQ_EXP=1) two arms on the same image without the dump: agent (unset) vs none (RADIANCE_AQL_ACQ=none): gate 21 ms,
#    identity copy (~145k), the warm-up's greedy answers (ds41_tests.py correct --only-q, written by the gate's check),
#    stepprobe, mb c4; copies and answers byte for byte. ANY difference = unsafe (stale cached lines are read). No
#    difference does not make it safe (78's stale reads decide whether it can be).
# Estimate A ~15 min, B +30 min. DRYRUN=1 bash 79-acqrule-tp4.sh : prints, writes nothing. Needs lib-fix.sh (+ lib-117.sh,
# lib-pr266.sh), ds41-fix2-tp4.yaml (passes RADIANCE_DEBUG_TAPE_DUMP and RADIANCE_AQL_ACQ) and tools/acqrule.py beside it.
. "$(cd "$(dirname "$0")" && pwd)/../lib-fix.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-fix.sh" || { echo "lib-fix.sh not found"; exit 1; }
H79=$(cd "$(dirname "$0")" && pwd)
Y=$QD/ds41-fix2-tp4.yaml; [ -e $Y ] || Y=$H79/ds41-fix2-tp4.yaml
AR=$QD/tools/acqrule.py; [ -e $AR ] || AR=$H79/tools/acqrule.py
X3DB=${X3DB:-radiance-local:ds41-exl3-117-1fb143a5-f2f3c3f7-fuse-c85088e7}
export IMAGE=$X3DB
ACQ_EXP=${ACQ_EXP:-0}
P=$O/acqrule
log "RESULT xfix 79-acqrule: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) ACQ_EXP=$ACQ_EXP compose $Y"
imgcheck $IMAGE
if [ "$DRY" != 1 ]; then
  for v in RADIANCE_DEBUG_TAPE_DUMP RADIANCE_AQL_ACQ; do grep -q "^ *$v:" $Y || { log "RESULT xfix 79: VIRHE $Y does not pass $v"; exit 1; }; done
  [ -e $AR ] || { log "RESULT xfix 79: VIRHE acqrule.py missing"; exit 1; }
fi
mkd $O $STATSDIR $P
unset RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT RADIANCE_DS41_CTAIL RADIANCE_DS41_INDEX RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE \
      RADIANCE_DEBUG_TAPE_DUMP RADIANCE_AQL_ACQ
best_vars
# A. the dump
export RADIANCE_DEBUG_TAPE_DUMP=1
gate xacq-dump 24; rc=$?
[ $rc = 1 ] && { log "RESULT xfix 79: VIRHE dump arm not healthy ($(desc))"; unset RADIANCE_DEBUG_TAPE_DUMP; exit 1; }
probe xacq-dump
mbp xacq-dump c4
run "docker logs $CN > $P/dump-engine.log 2>&1"
down
unset RADIANCE_DEBUG_TAPE_DUMP
run "python3 $AR $P/dump-engine.log > $P/acqrule.txt 2>&1"
[ "$DRY" = 1 ] || log "RESULT xfix 79 acqrule: $(grep -c '^r[0-9]' $P/acqrule.txt) recorded passes; $(tail -1 $P/acqrule.txt | cut -c1-400) | gapbench: ~0.9 us an agent acquire"
stopfile && exit 0
[ "$ACQ_EXP" = 1 ] || { log "RESULT xfix 79-acqrule: done (dump only; ACQ_EXP=1 adds the agent / none arms)"; exit 0; }
# B. agent vs none
for arm in agent none; do
  if [ $arm = none ]; then export RADIANCE_AQL_ACQ=none; else unset RADIANCE_AQL_ACQ; fi
  gate xacq-$arm 21; rc=$?
  [ $rc = 1 ] && { log "RESULT xfix 79: VIRHE $arm arm not healthy ($(desc))"; unset RADIANCE_AQL_ACQ; exit 1; }
  [ $rc = 2 ] && { log "RESULT xfix 79: VIRHE $arm arm slow twice -> stopped"; down; unset RADIANCE_AQL_ACQ; exit 1; }
  if [ "$DRY" != 1 ]; then
    ev=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN | grep '^RADIANCE_AQL_ACQ=' | cut -d= -f2- || true)
    [ "$ev" = "$([ $arm = none ] && echo none)" ] || { log "RESULT xfix 79: VIRHE the $arm arm's container env says RADIANCE_AQL_ACQ='$ev' -> stopped"; down; unset RADIANCE_AQL_ACQ; exit 1; }
  fi
  ident xacq-$arm
  probe xacq-$arm
  mbp xacq-$arm c4
  run "docker logs $CN > $P/$arm-engine.log 2>&1"
  stopfile && { down; unset RADIANCE_AQL_ACQ; exit 0; }
done
unset RADIANCE_AQL_ACQ
down
identcmp xacq-agent xacq-none || log "RESULT xfix 79: VIRHE acquire none changed the identity copy -> NOT safe"
if [ "$DRY" != 1 ]; then
  r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in a if (a[t].get("content"), a[t].get("finish")) != (b.get(t, {}).get("content"), b.get(t, {}).get("finish"))]
print(("IDENTICAL: %d answers" % len(a)) if a and not bad else ("DIFFERENT: " + " ".join(map(str, bad)) if a else "no answers"))' \
     $O/guard/xacq-agent-warm-q.txt $O/guard/xacq-none-warm-q.txt 2>&1 | cut -c1-500)
  log "RESULT xfix 79 warm-up greedy answers agent vs none: $r"
fi
log "RESULT xfix 79-acqrule: done | ref $REF_BEST524"
