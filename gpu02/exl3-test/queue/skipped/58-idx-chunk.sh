#!/bin/bash
# 58-idx-chunk.sh (6.11. 19.00 coordinator; user: "jatka tässä sessiossa testaamista, kehityksiä, profilointia. tehdään
# tästä nyt mahtava") -- EXL3 2.25 bpw on the indexer-fix image (-idx, 50: exact, prefill +19/+48 %), fin-best at MML 524288.
#   ccref  fin-best as 50's best line: probe, cc_sim u4 (the indexer fix in Claude Code load; 55 = the same load on the
#          base image), probe again after the load (the same-process slowdown seen in 50: 19.8 ms fresh, 22.4 ms after
#          135k+266k runs -- hypothesis: the KV-block expert loan is reclaimed once long sessions hold KV / prefix cache)
#   bpc2k  + --busy-prefill-chunk 2048 (50's pf phase: 4 decoders kept 22.7 % of their speed while a 31k prompt
#          prefilled, step 175 ms): probe, mb pf / turns / c4, cc_sim u4
#   bpc1k  + --busy-prefill-chunk 1024: probe, mb pf / turns / c4 (cc_sim only if the coordinator queues it)
# Refs: 50 xfix-idx-best524 (probe 47.7/50.6, c4 95.5, pf 1260.3 tok/s kept 0.231 step 175.4 ms, turns cold 28.67 hit 2.08 s).
# ~65 min. DRYRUN=1 bash 58-idx-chunk.sh : prints, writes nothing.
. "$(cd "$(dirname "$0")" && pwd)/../lib-fix.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-fix.sh" || { echo "lib-fix.sh not found"; exit 1; }
export IMAGE=$X3IDX
unset RADIANCE_DS41_INDEX
log "RESULT xfix 58-chunk: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19))"
imgcheck $IMAGE
mkd $O
evict
arm() {   # arm TAG EXTRA-FLAGS
  best_vars; export XARGS="$XARGS $2"
  gate $1 21; rc=$?
  [ $rc = 1 ] && { log "RESULT xfix 58 $1: VIRHE not healthy ($(desc))"; return 1; }
  [ $rc = 2 ] && { log "RESULT xfix 58 $1: VIRHE slow state after one restart -> skipped"; down; return 1; }
  probe $1
}
cc() { stopfile || run "timeout 3300 $T/ccsim-exl3.sh 4 $1 > $O/$1-ccsim-u4.out 2>&1"; }

if arm xidx-ccref ""; then
  cc xidx-ccref
  probe xidx-ccref-after
  run "docker logs $CN > $O/xidx-ccref-engine.log 2>&1"
fi
stopfile && { down; exit 0; }
if arm xidx-bpc2k "--busy-prefill-chunk 2048"; then
  for p in pf turns c4; do stopfile && break; mbp xidx-bpc2k $p; done
  cc xidx-bpc2k
  run "docker logs $CN > $O/xidx-bpc2k-engine.log 2>&1"
fi
stopfile && { down; exit 0; }
if arm xidx-bpc1k "--busy-prefill-chunk 1024"; then
  for p in pf turns c4; do stopfile && break; mbp xidx-bpc1k $p; done
  run "docker logs $CN > $O/xidx-bpc1k-engine.log 2>&1"
fi
down
log "RESULT xfix 58-chunk: done"
