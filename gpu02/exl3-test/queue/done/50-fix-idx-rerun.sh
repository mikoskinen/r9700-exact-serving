#!/bin/bash
# 50-fix-idx-rerun.sh (6.11. 17.55): first run stopped falsely -- ds41_tests.py rejected --tag -NAME (argparse took it for an option), no copy files, read as DIFFERENT. lib-fix.sh now passes --tag=-NAME.
# 50-fix-idx.sh (queue item, 6.11.2026) -- the indexer fixes on TP4: radiance-local:ds41-exl3-117-<idx>-f2f3c3f7-idx
# (engine ds41-idx: index_score scores only the candidate layer's kept blocks at layers 24/28/32/36, and a step's index
# runs are dealt over the 4 ranks with an all_gather of the selections; both exact, RADIANCE_DS41_INDEX for an A/B).
# Same .rad, plugins as today's image (exl3 f2f3c3f7).
#   1. evict; the 47 flags (MML 524288 MAXSEQS 4 NST 0 ckint 8192): RADIANCE_DS41_INDEX=off, gated (24 ms):
#      the identity copy over ~145k tokens, then onem.py 135000,262144 (decode ms/step at depth: no earlier number on
#      this model and engine, so this off arm is the reference for the deep-decode gain)
#   2. default (keep,split), gated (24 ms): the identity copy -> cmp with step 1 (must be IDENTICAL), the cold prefill
#      set (47's prompts; refs 2402.2 / 2002.2 / 1522.3 tok/s), onem.py 135000,262144, stepprobe essay/code 512
#   3. the fin-best line at MML 524288 (lib-117 best_vars), gated (21 ms): stepprobe, mb c4/c8/pf/turns (refs x117-best524)
# Expected (arithmetic from today's prefill profile, REPORT): 16k ~+2 %, 135k ~+23 %, 322k ~+50-55 % prefill; decode at
# 135k/262k a few % faster per step (layers 24-36 score <= 16384 positions); short-context decode unchanged (+-1 %).
# A DIFFERENT identity result stops the item before any measurement (the selections must be bit-identical).
# Estimate ~75 min (3 gated starts ~5 min each, ident 2 x ~2 min, onem 2 x ~6 min, cold set ~5 min, probe 1, mb 10).
# DRYRUN=1 bash 50-fix-idx.sh : prints, writes nothing. Needs lib-fix.sh, lib-117.sh, lib-pr266.sh, ds41-fix-tp4.yaml.
. "$(cd "$(dirname "$0")" && pwd)/../lib-fix.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-fix.sh" || { echo "lib-fix.sh not found"; exit 1; }
export IMAGE=$X3IDX
log "RESULT xfix 50-idx: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) MODEL=$MODELDIR/$MODEL"
imgcheck $IMAGE
mkd $O $STATSDIR
evict
pfflags
# 1. off: the identity reference and the deep-decode reference
export RADIANCE_DS41_INDEX=off
gate xfix-idx-off 24; rc=$?   # gated too: the deep-decode A/B needs both arms out of the slow state
[ $rc = 1 ] && { log "RESULT xfix 50: VIRHE off arm not healthy ($(desc))"; exit 1; }
[ $rc = 2 ] && { log "RESULT xfix 50: VIRHE off arm slow twice -> stopped"; down; exit 1; }
ident xfix-idx-off
stopfile && { down; exit 0; }
deep xfix-idx-off 135000,262144
run "docker logs $CN > $O/xfix-idx-off-engine.log 2>&1"
unset RADIANCE_DS41_INDEX
# 2. on (the image default)
gate xfix-idx 24; rc=$?
[ $rc = 1 ] && { log "RESULT xfix 50: VIRHE not healthy ($(desc))"; exit 1; }
[ $rc = 2 ] && { log "RESULT xfix 50: VIRHE slow state after one restart -> measurements skipped"; down; exit 1; }
run "docker logs $CN 2>&1 | grep -m2 'deepseek41 indexer' > $O/xfix-idx-mode.txt"
[ "$DRY" = 1 ] || log "RESULT xfix 50 mode: $(cat $O/xfix-idx-mode.txt | tr '\n' ' ')"
ident xfix-idx
identcmp xfix-idx-off xfix-idx || { log "RESULT xfix 50: VIRHE the indexer fixes changed the output -> stopped"; run "docker logs $CN > $O/xfix-idx-engine.log 2>&1"; down; exit 1; }
cold xfix-idx
stopfile && { down; exit 0; }
deep xfix-idx 135000,262144
probe xfix-idx
run "docker logs $CN > $O/xfix-idx-engine.log 2>&1"
stopfile && { down; exit 0; }
# 3. the fin-best line at MML 524288
best_vars
gate xfix-idx-best524 21; rc=$?
[ $rc = 1 ] && { log "RESULT xfix 50 best: VIRHE not healthy ($(desc))"; exit 1; }
[ $rc = 2 ] && { log "RESULT xfix 50 best: VIRHE slow state after one restart -> skipped"; down; exit 1; }
probe xfix-idx-best524
for p in c4 c8 pf turns; do stopfile && break; mbp xfix-idx-best524 $p; done
run "docker logs $CN > $O/xfix-idx-best524-engine.log 2>&1"
log "RESULT xfix 50 best: done | ref $REF_BEST524"
down
log "RESULT xfix 50-idx: done"
