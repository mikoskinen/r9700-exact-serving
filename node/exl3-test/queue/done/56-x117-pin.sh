#!/bin/bash
# 56-x117-pin.sh (6.11. 18.10) -- S4 with Claude Code statistics: fin-best 524k + --expert-profile /stats/cc-x117.json
# --expert-pin-top 0.3 and 0.5 (decode-hot residency): gate, probe, mb c4/c8 each; cc_sim u4 for pin 0.3. Reference: x117-best524
# (guard 19.71, essay 47.7 / code 50.6, c4 95.0, c8 123.4) and the morning cc_sim u4 (71.9 total, 21.5 per agent). ~45 min.
. "$(cd "$(dirname "$0")" && pwd)/../lib-117.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-117.sh" || { echo "lib-117.sh not found"; exit 1; }
log "RESULT x117 56-pin: start IMAGE=$IMAGE"
[ "$DRY" = 1 ] || [ -s $STATSDIR/cc-x117.json ] || { log "RESULT x117 56: VIRHE no $STATSDIR/cc-x117.json (55 not run?)"; exit 1; }
for P in 0.3 0.5; do
  stopfile && break
  TAG=x117-pin${P/./}
  evict
  best_vars; export XARGS="$XARGS --expert-profile /stats/cc-x117.json --expert-pin-top $P"
  gate $TAG 24; rc=$?
  [ $rc != 0 ] && { log "RESULT x117 56 pin $P: VIRHE gate rc $rc -> next"; down; continue; }
  probe $TAG
  for p in c4 c8; do stopfile && break; mbp $TAG $p; done
  [ "$P" = 0.3 ] && { stopfile || run "timeout 3300 $T/ccsim-exl3.sh 4 $TAG > $O/$TAG-ccsim-u4.out 2>&1"; }
  run "docker logs $CN > $O/$TAG-engine.log 2>&1"
  down
done
log "RESULT x117 56-pin: done | ref x117-best524: guard 19.71, essay 47.7 / code 50.6, c4 95.0, c8 123.4; morning cc_sim u4 71.9 total / 21.5 per agent"
