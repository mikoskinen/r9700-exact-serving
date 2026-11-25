#!/bin/bash
# 57-k-heat-t-ccref.sh (6.11. 21:30 coordinator; user: "Kyllä, pakotettu") -- the forced-length cc_sim reference: the base
# EXL3 2.25 bpw on the stock 1.17 EXL3 image (abf7f9a0, as 55), fin-best at MML 524288, cc_sim u4 with --gen-mode forced --seed 1
# (ccsim-exl3.sh default since 21:30). Every later DeepSeek cc_sim is forced too, so they compare with this row. ~22 min.
. "$(cd "$(dirname "$0")" && pwd)/../lib-117.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-117.sh" || { echo "lib-117.sh not found"; exit 1; }
TAG=x117-ccref-forced
log "RESULT x117 ccref-forced: start IMAGE=$IMAGE"
mkd $O
evict
best_vars
gate $TAG 21; rc=$?
[ $rc = 1 ] && { log "RESULT x117 ccref-forced: VIRHE not healthy ($(desc))"; exit 1; }
[ $rc = 2 ] && { log "RESULT x117 ccref-forced: VIRHE slow state after one restart -> skipped"; down; exit 1; }
probe $TAG
stopfile || run "timeout 3300 $T/ccsim-exl3.sh 4 $TAG > $O/$TAG-ccsim-u4.out 2>&1"
probe $TAG-after
run "docker logs $CN > $O/$TAG-engine.log 2>&1"
down
log "RESULT x117 ccref-forced: done"
