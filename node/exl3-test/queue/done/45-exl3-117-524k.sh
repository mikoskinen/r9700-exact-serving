#!/bin/bash
# 45-exl3-117-524k.sh (6.11. 15.37; user: "ei 262k kiinnosta, vähintään 500k") = 45-exl3-117.sh with --max-model-len 524288 (lib-117 best_vars)
# 45-exl3-117.sh (queue item, 6.11.2026) -- the 1.17 EXL3 image (radiance-local:ds41-exl3-117-abf7f9a0-f2f3c3f7: PR #266 9c7ce498
# + our two EXL3 commits + radiance-exl3 ABI 35) on the EXL3 2.25 bpw .rad with the 6.11. fin-best line (host pool 19456, NST 0,
# ckint 2048, span 2, IL 8, MAXSEQS 8, MML 262144, RAM KV tier, ngram ram): the standard set as the 6.11. window, so every row has
# a same-model reference. Evict, gate (21 ms; 6.11. fast 19.2-19.56), stepprobe essay/code 512, mb c4/c8/pf/turns, cc_sim u4
# (u4 only), then the 3.11. morning flags (NST 5, ckint 8192, MAXSEQS 4, no RAM tier, unguarded as then): cold prefill
# 16k / 59k / 140k x 2.
# Estimate ~45 min (2 starts ~4 min each + warm-up; probe 1; mb 10.3; cc_sim 11; morning prefill 5).
# DRYRUN=1 bash 45-exl3-117.sh : prints, writes nothing. Needs lib-117.sh, lib-pr266.sh, ds41-117-tp4.yaml in queue/.
. "$(cd "$(dirname "$0")" && pwd)/../lib-117.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-117.sh" || { echo "lib-117.sh not found"; exit 1; }
TAG=x117-best524
log "RESULT x117 45-exl3-117: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) MODEL=$MODELDIR/$MODEL"
[ "$DRY" = 1 ] || docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT x117 45: VIRHE image $IMAGE not on gpu02"; exit 1; }
[ "$DRY" = 1 ] || [ -f $MODELDIR/$MODEL ] || { log "RESULT x117 45: VIRHE $MODELDIR/$MODEL missing"; exit 1; }
mkd $O $STATSDIR
evict
best_vars
gate $TAG 21; rc=$?
[ $rc = 1 ] && { log "RESULT x117 45: VIRHE not healthy ($(desc))"; exit 1; }
[ $rc = 2 ] && { log "RESULT x117 45: VIRHE slow state after one restart -> measurements skipped"; down; exit 1; }
probe $TAG
for p in c4 c8 pf turns; do stopfile && break; mbp $TAG $p; done
stopfile || run "timeout 3300 $T/ccsim-exl3.sh 4 x117-best524 > $O/x117-best524-ccsim-u4.out 2>&1"
run "docker logs $CN > $O/$TAG-engine.log 2>&1"
log "RESULT x117 45 best: done | ref $REF_X3_BEST"
stopfile && { down; exit 0; }
# the 3.11./6.11. morning flags, unguarded as then (DSpark 5 rounds are 45-53 ms, not the NST 0 gate)
export HOSTPOOL=19456 NGRAM=ram CKINT=8192 MNBT=8192 NST=5 MAXSEQS=4 MML=524288 XARGS=
restart x117-morning || { log "RESULT x117 45 morning: VIRHE not healthy"; exit 1; }
run "(cd $D && timeout 1800 python3 ds41_tests.py prefill --reps 2 > $O/x117-morning-prefill.txt 2>&1)"
[ "$DRY" = 1 ] || log "RESULT x117 45 prefill (morning flags: DSpark 5, ckint 8192, MAXSEQS 4, fresh): $(pfline $O/x117-morning-prefill.txt) | ref $REF_X3_MORNING"
run "docker logs $CN > $O/x117-morning-engine.log 2>&1"
down
log "RESULT x117 45-exl3-117: done"
