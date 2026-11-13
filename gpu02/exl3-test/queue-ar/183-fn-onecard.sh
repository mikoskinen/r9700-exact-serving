#!/bin/bash
# 183-fn-onecard.sh (queue item, 11.11.2026, queue-ar; gpu02, ONE card (HIP device 0), every radiance container DOWN; ~12 min)
# -- fn4's deferred one-card reads FOLDED into one queue slot (coordinator 11.11.: fewer items / starts; none dropped): the
# five bodies as they were queued, unchanged (md5 below), one after another from queue-ar, each with its own guards, logs and
# RESULT lines; one that fails does not stop the next:
#   171-fn-e8read.sh 20512d1a   E8' reading, M 32 / 64 (12.11. re-scope; was 5 M)          ~5 min
#   175-fn-gdwpe.sh  beb00ae5   E10b register-cap screen, DS M 4/8 + FN M 8/16 (was 10+3) ~6 min
#   (12.11. re-scope, user: quick screens) 174 gdctr DROPPED from this slot -- a counters reading, not a screen; 174's body kept
#   (12.11., coordinator: the queue faster) 176 pfcmp HELD -- the DS dsgd plugin question is the DS line's own now (exl3-main +
#   BI-dual); 155 hcread HELD -- a reading of today's decode hc_read forms, which hc_bi 2 (the BI base) replaces up to 64 rows
# DRYRUN=1 bash 183-fn-onecard.sh : each body's own dry run (prints every command, runs and writes nothing).
set -u
QA=$HOME/gpu/ds41/exl3-test/queue-ar; L=$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
DRY=${DRYRUN:-0}
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
B="171-fn-e8read.sh:20512d1a 175-fn-gdwpe.sh:beb00ae5"   # 12.11.: pfcmp + hcread held (stale); 12.11. re-scope: 174 gdctr dropped, 171 / 175 trimmed
log "RESULT fn4 183-fn-onecard: start -- $B"
for b in $B; do
  f=${b%%:*}; m=${b#*:}
  if [ ! -e $QA/$f ] || [ "$(md5sum < $QA/$f | cut -c1-8)" != $m ]; then log "RESULT fn4 183: VIRHE $f missing or not md5 $m -- skipped"; continue; fi
  log "RESULT fn4 183: body $f"
  ( cd $QA && DRYRUN=$DRY bash $QA/$f ); rc=$?
  [ $rc = 0 ] || log "RESULT fn4 183: $f exit $rc (the next body runs)"
done
log "RESULT fn4 183-fn-onecard: done"
