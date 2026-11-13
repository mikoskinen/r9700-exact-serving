#!/bin/bash
# 60-exl3-29.sh (chain-after queue item, prepared 6.11.2026, NOT run) -- the 6.11. standard set on the LibertAI EXL3 2.9 bpw .rad with the
# 1.17 EXL3 image (radiance-local:ds41-exl3-117-abf7f9a0-f2f3c3f7) and the x117-best524 serve line: TP4 exact, expert_tiered,
# --max-model-len 524288, MTP off (NST 0), MAXSEQS 8, ngram ram, RAM KV tier 84 GiB, ckint 2048 + span 2 + IL 8; host pool
# HP29 (default 31488, see lib-29.sh: 19456 cannot start with these experts).
# Steps: pre-checks (image, .rad, its .sha256ok from the verified copy), evict, gate (limit GATE29 = 30 ms), warm-up answers vs the
# 2.25's, stepprobe essay/code 512, mb c4 / c8 / pf / turns, cc_sim u4 (only), engine log, comparison line vs the 2.25 bpw
# (cmp29.py: x117-best524 rows, cc_sim from the 6.11. morning window), down.
# Estimate ~45 min: start ~5 min (322 GB .rad; 2.25: 232 s) x 1-2, warm-up 1, probe 1, mb ~14, cc_sim 11 (660 s), rest 3.
# DRYRUN=1 bash 60-exl3-29.sh : prints every command, runs and writes nothing. Speed only (no quality test).
# On gpu02 (install-29.sh stage): the support files lib-29.sh + cmp29.py (+ snapshots lib-pr266.sh, ds41-117-tp4.yaml) live in
# ~/gpu/ds41/exl3-test/queue-29/; lib-29.sh is looked up beside this file, in ../ (queue/ when chain-after runs it from
# queue/done/) and in that staging directory. lib-pr266.sh and ds41-117-tp4.yaml are taken from queue/ when present.
# Enqueue = copy this one file into ~/gpu/ds41/exl3-test/queue/ (install-29.sh enqueue).
H29="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$H29/lib-29.sh" "$H29/../lib-29.sh" "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
CMP=$K29/cmp29.py
TAG=x29-best524
log "RESULT x29 60-exl3-29: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) MODEL=$MODELDIR/$MODEL HP29=$HP29 GATE29=$GATE29 lib $(sha256sum $LIBPR | cut -c1-12) yaml $Y"
if [ "$DRY" != 1 ]; then
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT x29 60: VIRHE image $IMAGE not on gpu02"; exit 1; }
  [ -f $MODELDIR/$MODEL ] || { log "RESULT x29 60: VIRHE $MODELDIR/$MODEL missing"; exit 1; }
  [ -s $MODELDIR/$MODEL.sha256ok ] || { log "RESULT x29 60: VIRHE $MODELDIR/$MODEL.sha256ok missing (copy not verified)"; exit 1; }
  [ -n "$(docker ps --format '{{.Names}}' | grep -E '^radiance-prod')" ] && { log "RESULT x29 60: VIRHE production is running"; exit 1; }
fi
mkd $O $STATSDIR
evict
best_vars
gate $TAG $GATE29; rc=$?
[ $rc = 1 ] && { log "RESULT x29 60: VIRHE not healthy ($(desc)); start log $O/start-$TAG.log: $(grep -E '^E ' $O/start-$TAG.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; exit 1; }
[ $rc = 2 ] && { log "RESULT x29 60: VIRHE slow state after one restart -> measurements skipped"; down; exit 1; }
warmq $TAG
probe $TAG
for p in c4 c8 pf turns; do stopfile && break; mbp $TAG $p; done
stopfile || run "timeout 3300 $T/ccsim-exl3.sh 4 $TAG > $O/$TAG-ccsim-u4.out 2>&1"
run "docker logs $CN > $O/$TAG-engine.log 2>&1"
run "python3 $CMP $L >> $O/$TAG-cmp.txt 2>&1"
[ "$DRY" = 1 ] || log "$(tail -1 $O/$TAG-cmp.txt | cut -c1-1500)"
log "RESULT x29 60 best524: done | ref $REF_X117_524 || $REF_X3_MORNING_BEST"
down
log "RESULT x29 60-exl3-29: done"
