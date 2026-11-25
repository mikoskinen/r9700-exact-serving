#!/bin/bash
# 55-x117-ccstats.sh (6.11. 18.10; user: "jatka analysointia, radiancen korjauksia, testauksia jatkuvalla syötöllä") -- collect
# routed-expert statistics from the Claude Code load (not from short probe prompts: the 6.11. pin rows used probe-prompt stats
# and lost 6 %): EXL3 2.25 bpw on the 1.17 image, fin-best at MML 524288 + --expert-stats /stats/cc-x117.json, cc_sim u4.
# Output: STATSDIR/cc-x117.json for 56-x117-pin.sh. ~25 min.
. "$(cd "$(dirname "$0")" && pwd)/../lib-117.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-117.sh" || { echo "lib-117.sh not found"; exit 1; }
TAG=x117-ccstats
log "RESULT x117 55-ccstats: start IMAGE=$IMAGE"
mkd $O $STATSDIR
evict
best_vars; export XARGS="$XARGS --expert-stats /stats/cc-x117.json"
gate $TAG 24; rc=$?
[ $rc = 1 ] && { log "RESULT x117 55: VIRHE not healthy ($(desc))"; exit 1; }
[ $rc = 2 ] && { log "RESULT x117 55: VIRHE slow state after one restart -> skipped"; down; exit 1; }
stopfile || run "timeout 3300 $T/ccsim-exl3.sh 4 x117-ccstats > $O/x117-ccstats-ccsim-u4.out 2>&1"
run "docker logs $CN > $O/$TAG-engine.log 2>&1"
down
[ "$DRY" = 1 ] || log "RESULT x117 55-ccstats: done; stats $(ls -la $STATSDIR/cc-x117.json 2>&1 | awk "{print \$5, \$9}")"
