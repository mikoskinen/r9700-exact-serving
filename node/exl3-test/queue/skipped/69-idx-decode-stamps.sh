#!/bin/bash
# 64-idx-decode-stamps.sh (queue item, 6.11.2026) -- why the -idx image's decode at depth did not get faster (50-fix-idx:
# 135k 28.1 -> 28.8 ms/step, 266k 30.3 -> 30.4, RADIANCE_DS41_INDEX off -> on), measured inside the RECORDED decode pass.
# --profile-ops cannot answer it: it turns the tape off (core/runtime/ctx.cpp:135 `!profiling_`), so every op runs alone
# and the overlap a played decode pass has is gone. RADIANCE_AQL_STAMP=N instead stamps every dispatch of one played pass in
# N on card 0 (start/end from the packet processor, nothing else changes: core/device/aql.cpp "dispatch timestamps") into
# $STATSDIR/stamps.csv; idxstamps.py splits the passes into onem.py's depths and gives, per depth, the decode pass's wall and
# busy time, every index_score dispatch's time in issue order (layers 2, 8, 14, 20, 24, 28, 32, 36), and the index_score
# time no other dispatch overlaps (the part on the pass's critical path).
#   image  -idx (X3IDX, the 50-fix-idx image), the 47 flags (MML 524288 MAXSEQS 4 NST 0 ckint 8192 MNBT 8192, no RAM tier)
#   arms   on (unset = keep,split), then off (RADIANCE_DS41_INDEX=off); each: restart + gate (24 ms), onem.py 135000,262144
#          (fresh prefill, 256 greedy tokens a depth, then the hit turn), stamps every 8th played pass
# Reading: the off arm's 24-36 dispatches should be ~0.39 ms at 266k (prof0 base, 243k: 388-395 us); on, the same four
# dispatches are what the restricted kernel costs in the engine. If they shrink but the wall does not, the indexer is
# overlapped (its exclusive time is small); if they do not shrink, the kernel is not faster on the engine's keep pattern.
# Estimate ~25 min (2 gated starts ~5 min, onem 135k+262k ~5-6 min an arm). ARMS="on" for one start (~12 min).
# DRYRUN=1 bash 64-idx-decode-stamps.sh : prints, writes nothing. Needs lib-fix.sh (+ lib-117.sh, lib-pr266.sh,
# ds41-fix-tp4.yaml) and idxstamps.py in queue/.
. "$(cd "$(dirname "$0")" && pwd)/../lib-fix.sh" 2>/dev/null || . "$(cd "$(dirname "$0")" && pwd)/lib-fix.sh" || { echo "lib-fix.sh not found"; exit 1; }
export IMAGE=$X3IDX
P=$O/idx-stamps
ARMS=${ARMS:-"on off"}
EVERY=${STAMP_EVERY:-8}
IS=$QD/idxstamps.py; [ -e $IS ] || IS=$(cd "$(dirname "$0")" && pwd)/idxstamps.py
log "RESULT xfix 64-idx-stamps: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) arms [$ARMS] stamp 1 pass in $EVERY"
imgcheck $IMAGE
[ "$DRY" = 1 ] || [ -e $IS ] || { log "RESULT xfix 64: VIRHE idxstamps.py missing"; exit 1; }
mkd $O $P
pfflags
for arm in $ARMS; do
  case $arm in
    on)  unset RADIANCE_DS41_INDEX ;;
    off) export RADIANCE_DS41_INDEX=off ;;
    *)   log "RESULT xfix 64: unknown arm $arm"; continue ;;
  esac
  export STATSDIR=$P/$arm AQLSTAMP=$EVERY
  mkd $STATSDIR
  run "rm -f $STATSDIR/stamps.csv"
  gate xidx-st-$arm 24; rc=$?
  [ $rc = 1 ] && { log "RESULT xfix 64: VIRHE $arm arm not healthy ($(desc))"; exit 1; }
  [ $rc = 2 ] && { log "RESULT xfix 64: VIRHE $arm arm slow twice -> skipped"; continue; }
  run "docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' > $P/$arm-mode.txt"
  deep xidx-st-$arm 135000,262144
  run "docker logs $CN > $P/$arm-engine.log 2>&1"
  down   # the stamp writer flushes at exit
  run "python3 $IS $STATSDIR/stamps.csv --last 3 > $P/$arm-stamps.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT xfix 64 $arm [$(cat $P/$arm-mode.txt 2>/dev/null | cut -c1-90)]: 135k / 262k / hit-turn: $(grep '^  segment' $P/$arm-stamps.txt | cut -c3- | tr '\n' '|' | cut -c1-1400)"
  stopfile && { unset AQLSTAMP RADIANCE_DS41_INDEX; exit 0; }
done
unset AQLSTAMP RADIANCE_DS41_INDEX
log "RESULT xfix 64-idx-stamps: done -- $P/{on,off}-stamps.txt (IS json lines), stamps.csv per arm"
