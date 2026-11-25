#!/bin/bash
# chain-after.sh (6.11.2026) -- runs after chain-full.sh (which ends with production up): production down again (LiteLLM is
# still on xeon), evict, tempmon + 95 C watchdog, then every queue/NN-*.sh in name order (each once; the directory is re-read
# after every item, so items can be added while this runs). Empty queue + queue/HOLD present -> wait (re-check every 60 s);
# empty queue and no HOLD -> window.sh PHASES=down,produp and exit. Stop early: rm queue/HOLD + touch runs/sweep.stop.
set -u
T=$HOME/gpu/ds41/exl3-test; R=$T/runs; L=$R/ds41-exl3.log; Q=$T/queue; cd $T
log() { echo "$(date '+%F %T') $*" | tee -a $L; }
while pgrep -f "[c]hain-full\.sh" > /dev/null; do sleep 20; done
log "RESULT chain-after: chain-full.sh finished -> production down again for the queue"
LITELLM_SWITCHED=1 FREE_MIN_GB=60 PHASES=prodstop,evict ./window.sh > $R/chain-after-stop.out 2>&1
[ -n "$(docker ps --format '{{.Names}}' | grep -E '^radiance-prod')" ] && { log "RESULT chain-after: VIRHE production still running -> queue not run"; exit 1; }
CSV=$R/temps-exl3-$(date +%m%d).csv
rm -f $R/watchdog.stop $R/sweep.stop
nohup $HOME/gpu/bench/tempmon.sh $CSV > /dev/null 2>&1 &
nohup $T/watchdog-exl3.sh $CSV > /dev/null 2>&1 &
mkdir -p $Q/done
while true; do
  [ -e $R/sweep.stop ] && { log "RESULT chain-after: sweep.stop -> queue stopped"; break; }
  item=$(ls $Q/[0-9][0-9]-*.sh 2>/dev/null | sort | head -1)
  if [ -n "$item" ]; then
    n=$(basename $item); mv $item $Q/done/$n
    log "RESULT chain-after: queue item $n start"; bash $Q/done/$n > $R/queue-$n.out 2>&1; log "RESULT chain-after: queue item $n exit $?"
    continue
  fi
  [ -e $Q/HOLD ] || break
  sleep 60
done
PHASES=down,produp ./window.sh > $R/chain-after-up.out 2>&1
log "RESULT chain-after: done; production $(docker ps --format '{{.Names}} {{.Image}}' | grep prod | tr '\n' ' ')"
