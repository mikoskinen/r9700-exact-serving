#!/bin/bash
# 104-keepawake-on.sh (queue item, 9.11.2026, queue-ar; gpu02; seconds) -- after 101 / 103 / 103b: start the read-only
# keep-awake poller (tools/keepawake.py, gpu_metrics on all 4 cards) as ONE background process for the rest of the test
# window, so every following FN / DS item runs out of the TP4 expert slow state (101: no poll 0/8 fast, 5 ms gpu_metrics
# 3/3 fast, c8 ~630 -> ~750 tok/s). Interval: X104_IVL, else 5 ms (101's proven one: the test window wants certainty, not
# the cheapest; 103's pick from tools/pick103.sh is only logged, for production sizing).
# setsid + nohup so it outlives the item; pid in runs/keepawake.pid, cost lines in ~/gpu/watch/keepawake.log.
# Stop: kill $(cat ~/gpu/ds41/exl3-test/runs/keepawake.pid). Writes nothing to sysfs. An already running poller is kept.
# DRYRUN=1 bash 104-keepawake-on.sh : prints the command, starts nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; P=$R/keepawake.pid; KL=$HOME/gpu/watch/keepawake.log
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/keepawake.py" ] && [ -e "$f/pick103.sh" ] && { TL=$f; break; }; done
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
[ -n "$TL" ] || { log "RESULT keepawake-on: FAILED -- tools/keepawake.py or pick103.sh not found"; exit 1; }
IVL=${X104_IVL:-5}; PICK=$(bash $TL/pick103.sh $O/103-table.txt)
if [ -s "$P" ] && kill -0 "$(cat $P)" 2>/dev/null; then log "RESULT keepawake-on: already running pid $(cat $P), kept"; exit 0; fi
CMD="setsid nohup python3 -I $TL/keepawake.py --interval-ms $IVL --node gpu_metrics --cards all --log $KL --report-s 600 > /dev/null 2>&1 < /dev/null &"
if [ "$DRY" = 1 ]; then echo "+ $CMD"; echo "+ echo \$! > $P"; log "RESULT keepawake-on: interval $IVL ms"; exit 0; fi
mkdir -p $(dirname $KL)
eval "$CMD"; echo $! > $P; sleep 5
if kill -0 "$(cat $P)" 2>/dev/null; then log "RESULT keepawake-on: started pid $(cat $P), gpu_metrics every $IVL ms on all cards (103's pick ${PICK} ms; 103 table: $(tr '\n' ';' < $O/103-table.txt 2>/dev/null | cut -c1-200)), log $KL"
else log "RESULT keepawake-on: FAILED -- poller exited at once: $(tail -3 $KL 2>/dev/null | tr '\n' ' ' | cut -c1-200)"; exit 1; fi
