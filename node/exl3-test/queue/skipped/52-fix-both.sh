#!/bin/bash
# 52-fix-both.sh (queue item, 6.11.2026) -- both fixes together, the fable candidate: radiance-local:ds41-exl3-117-
# <idx-x3sk>-<exl3sk>-idx-splitk with RADIANCE_EXL3_SPLITK=1 (the indexer fixes are the image default).
#   1. the fin-best line at MML 524288, gated (21 ms): stepprobe, mb c4/c8/pf/turns, cc_sim u4 (refs x117-best524; cc_sim
#      6.11. EXL3 u4: TTFT p50/p95 3.88/6.45 s, 21.5 tok/s/agent, 71.9 total)
#   2. the 47 flags: the cold prefill set (refs 2402.2 / 2002.2 / 1522.3 tok/s) and onem.py 262144
# Run after 50 and 51 have passed (50's identity check, 51's copy check).
# Estimate ~60 min (2 starts, probe 1, mb 10, cc_sim 11, cold set ~4, onem ~6).
# DRYRUN=1 bash 52-fix-both.sh : prints, writes nothing.
. "$(cd "$(dirname "$0")" && pwd)/../lib-fix.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-fix.sh" || { echo "lib-fix.sh not found"; exit 1; }
export IMAGE=$X3BOTH
log "RESULT xfix 52-both: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19))"
imgcheck $IMAGE
mkd $O $STATSDIR
export RADIANCE_EXL3_SPLITK=1
best_vars
gate xfix-both 21; rc=$?
[ $rc = 1 ] && { log "RESULT xfix 52: VIRHE not healthy ($(desc))"; exit 1; }
[ $rc = 2 ] && { log "RESULT xfix 52: VIRHE slow state after one restart -> skipped"; down; exit 1; }
probe xfix-both
for p in c4 c8 pf turns; do stopfile && break; mbp xfix-both $p; done
stopfile || run "timeout 3300 $T/ccsim-exl3.sh 4 xfix-both > $O/xfix-both-ccsim-u4.out 2>&1"
run "docker logs $CN > $O/xfix-both-engine.log 2>&1"
log "RESULT xfix 52 best: done | ref $REF_BEST524"
stopfile && { down; exit 0; }
pfflags
gate xfix-both-pf 24 || { log "RESULT xfix 52 pf: VIRHE not healthy or slow"; down; exit 1; }
cold xfix-both-pf
deep xfix-both-pf 262144
run "docker logs $CN > $O/xfix-both-pf-engine.log 2>&1"
down
log "RESULT xfix 52-both: done"
