#!/bin/bash
# 66-x3g-tp4.sh (queue item, 6.11.2026; after 63 has shown gu25 exact and faster on the card) -- the 2.5-bit gate_up
# prefetch fix on TP4: radiance-local:ds41-exl3-117-3ee8358a-5dd4abf3-x3g = the idx-splitk image (engine ds41-idx-x3sk
# 3ee8358a, the -idx indexer on by default) with exl3r4d.so from exl3-gu25 5dd4abf3 (gate_up at a half-integer K
# prefetches 3 k-steps, the 0-waits gone from its K loop; same bits). Layers 0-5 and 35-39 are 2.5-bit.
#   1. RADIANCE_EXL3_SPLITK unset (the PF fix alone; the engine is -idx's but for the ep_ranks plumbing, which is off),
#      the fin-best line at MML 524288, gated (21 ms): stepprobe essay/code 512, mb c4 / c8, the short copy (exact)
#   2. RADIANCE_EXL3_SPLITK=1 (E1/E2 + PF fix), gated: stepprobe, mb c4
# References: -idx fin-best (50, 6.11.): guard 19.72 ms, essay 47.7 / code 50.6 tok/s, c4 95.5, c8 123.4 tok/s;
# base x117-best524: 19.71 ms, 47.7 / 50.6, c4 95.0, c8 123.4.
# Expected (arithmetic from prof0 1u: gate_up 2.5-bit 147 us a call vs 2-bit 63.5; if 2.5-bit falls to 1.25 x 2-bit):
# -0.75 ms a 1-user step (essay/code ~+4 %), c4 ~-2 ms a step (~+5 %). 63's on-card numbers refine this.
# Estimate ~30 min (2 gated starts, probe 2, mb c4/c8/c4 ~10, copy 2).
# DRYRUN=1 bash 65-x3g-tp4.sh : prints, writes nothing. Needs lib-fix.sh (+ lib-117.sh, lib-pr266.sh, ds41-fix-tp4.yaml).
. "$(cd "$(dirname "$0")" && pwd)/../lib-fix.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-fix.sh" || { echo "lib-fix.sh not found"; exit 1; }
X3G=${X3G:-radiance-local:ds41-exl3-117-3ee8358a-5dd4abf3-x3g}
export IMAGE=$X3G
REF_IDXBEST="-idx fin-best 6.11. (50): guard 19.72 ms, essay 47.7 / code 50.6 tok/s, c4 95.5, c8 123.4"
log "RESULT xfix 66-x3g: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19))"
imgcheck $IMAGE
mkd $O $STATSDIR
best_vars
unset RADIANCE_EXL3_SPLITK
gate xfix-x3g 21; rc=$?
[ $rc = 1 ] && { log "RESULT xfix 66: VIRHE not healthy ($(desc))"; exit 1; }
[ $rc = 2 ] && { log "RESULT xfix 66: VIRHE slow state after one restart -> skipped"; down; exit 1; }
probe xfix-x3g
for p in c4 c8; do stopfile && break; mbp xfix-x3g $p; done
run "(cd $D && timeout 900 python3 ds41_tests.py copy --chars 6000 --max-tokens 2400 --runs 1 --tag=-xfix-x3g --outdir $O > $O/xfix-x3g-copy.txt 2>&1)"
[ "$DRY" = 1 ] || log "RESULT xfix 66 copy (6k chars, greedy): $(grep '^J ' $O/xfix-x3g-copy.txt | head -1 | cut -c3-300)"
run "docker logs $CN > $O/xfix-x3g-engine.log 2>&1"
log "RESULT xfix 66 PF fix alone: ref $REF_IDXBEST | $REF_BEST524"
stopfile && { down; exit 0; }
export RADIANCE_EXL3_SPLITK=1
gate xfix-x3g-sk 21; rc=$?
if [ $rc = 0 ]; then
  run "docker logs $CN 2>&1 | grep -m1 'EP split-K rules' > $O/xfix-x3g-sk-mode.txt"
  [ "$DRY" = 1 ] || log "RESULT xfix 66 mode: $(cat $O/xfix-x3g-sk-mode.txt | cut -c1-200)"
  probe xfix-x3g-sk
  mbp xfix-x3g-sk c4
  run "docker logs $CN > $O/xfix-x3g-sk-engine.log 2>&1"
else
  log "RESULT xfix 66 split-K arm: skipped (gate rc $rc)"
fi
unset RADIANCE_EXL3_SPLITK
down
log "RESULT xfix 66-x3g: done"
