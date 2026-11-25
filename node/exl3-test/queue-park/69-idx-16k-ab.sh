#!/bin/bash
# 65-idx-16k-ab.sh (queue item, 6.11.2026) -- where the indexer's row split starts to pay: -idx's cold 16k was -3.3 %
# (2322.0 vs base 2402.2 tok/s) while 135k/322k gained +19 / +48 %. Candidates: the split's all_gather (16 MB of
# selections a layer at 8 indexer layers a chunk, the split on from 4096 live) or 50's non-fresh state (the 145k identity
# copy ran before its cold set). The keep restriction is ruled out by arithmetic (+5 % on 4 layers' index_score at 16k =
# ~1.6 ms of a ~3.4 s chunk) and is off below 32768 live in idx2b anyway.
# Image idx2b (ds41-idx2 528e5e06: the -idx indexer + idx2's kernels + RADIANCE_DS41_KEEP_MIN_LIVE / _SPLIT_MIN_LIVE).
# Three FRESH arms (restart + gate each, nothing else before the prompts), the 47 flags (MML 524288 MAXSEQS 4 NST 0
# ckint 8192 MNBT 8192), cold prompts 16k, 32k, 64k (ds41_tests.py prefill, nonce first, 0 cached):
#   A  default                              (keep from 32768 live, split from 4096)
#   B  RADIANCE_DS41_INDEX=keep             (no split)
#   C  RADIANCE_DS41_SPLIT_MIN_LIVE=65536   (split only from 64k live: none of these prompts)
#   (ARMS="A B C off" adds RADIANCE_DS41_INDEX=off: the base indexer in the same image)
# Reading: A < B at 16k/32k = the split costs there -> set its threshold where A and B cross (Claude Code first prompts are
# 20-60k); A = B = C = base 2402 at 16k = 50's -3.3 % was state, not the indexer.
# Estimate ~20 min (3 gated starts ~5 min, 16k+32k+64k ~1 min an arm).
# DRYRUN=1 bash 65-idx-16k-ab.sh : prints, writes nothing. Needs lib-fix.sh (+ lib-117.sh, lib-pr266.sh) and
# ds41-fix2-tp4.yaml (the threshold pass-throughs) in queue/.
. "$(cd "$(dirname "$0")" && pwd)/../lib-fix.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-fix.sh" || { echo "lib-fix.sh not found"; exit 1; }
Y=$QD/ds41-fix2-tp4.yaml; [ -e $Y ] || Y=$(cd "$(dirname "$0")" && pwd)/ds41-fix2-tp4.yaml
X3IDX2B=${X3IDX2B:-radiance-local:ds41-exl3-117-528e5e06-f2f3c3f7-idx2b}
export IMAGE=$X3IDX2B
ARMS=${ARMS:-"A B C"}
log "RESULT xfix 65-16k-ab: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) arms [$ARMS] compose $Y"
imgcheck $IMAGE
[ "$DRY" = 1 ] || grep -q RADIANCE_DS41_SPLIT_MIN_LIVE $Y || { log "RESULT xfix 65: VIRHE $Y has no threshold pass-through"; exit 1; }
mkd $O $STATSDIR
pfflags
for arm in $ARMS; do
  unset RADIANCE_DS41_INDEX RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_KEEP_MIN_LIVE
  case $arm in
    A) ;;
    B) export RADIANCE_DS41_INDEX=keep ;;
    C) export RADIANCE_DS41_SPLIT_MIN_LIVE=65536 ;;
    off) export RADIANCE_DS41_INDEX=off ;;
    *) log "RESULT xfix 65: unknown arm $arm"; continue ;;
  esac
  gate x16k-$arm 24; rc=$?
  [ $rc = 1 ] && { log "RESULT xfix 65: VIRHE arm $arm not healthy ($(desc))"; exit 1; }
  [ $rc = 2 ] && { log "RESULT xfix 65: arm $arm slow twice -> skipped"; continue; }
  run "docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' > $O/x16k-$arm-mode.txt"
  run "(cd $D && timeout 900 python3 ds41_tests.py prefill --depths 16384,32768,65536 --cpt 3.95 --reps 1 > $O/x16k-$arm-pf.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xfix 65 arm $arm [$(cut -c1-140 $O/x16k-$arm-mode.txt)] fresh cold: $(pfline $O/x16k-$arm-pf.txt) | vs base 16k 2402.2: $(pct "$(pfv $O/x16k-$arm-pf.txt 16384)" 2402.2); -idx 50 16k 2322.0 (after a 145k request)"
  run "docker logs $CN > $O/x16k-$arm-engine.log 2>&1"
  stopfile && break
done
unset RADIANCE_DS41_INDEX RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_KEEP_MIN_LIVE
down
log "RESULT xfix 65-16k-ab: done"
