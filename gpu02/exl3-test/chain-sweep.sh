#!/bin/bash
# chain-sweep.sh (5.11.2026) -- after window.sh (incl. its produp): production down again (LiteLLM still on xeon), evict, tempmon +
# 95 C watchdog, sweep-exl3.sh sweep-exl3a.cfg, then window.sh PHASES=down,produp (EXL3 down, watchdog off, production back + gates).
set -u
T=$HOME/gpu/ds41/exl3-test; R=$T/runs; L=$R/ds41-exl3.log; cd $T
log() { echo "$(date '+%F %T') $*" | tee -a $L; }
while pgrep -f "[e]xl3-test/window\.sh|[.]/window\.sh" > /dev/null; do sleep 15; done
log "RESULT chain-sweep: window.sh finished -> production down again for the sweep"
LITELLM_SWITCHED=1 FREE_MIN_GB=60 PHASES=prodstop,evict ./window.sh > $R/chain-sweep-stop.out 2>&1
[ -n "$(docker ps --format '{{.Names}}' | grep -E '^radiance-prod')" ] && { log "RESULT chain-sweep: VIRHE production still running -> no sweep"; exit 1; }
CSV=$R/temps-exl3-$(date +%m%d).csv   # same name as window.sh, so its ph_down stops this tempmon too
rm -f $R/watchdog.stop
nohup $HOME/gpu/bench/tempmon.sh $CSV > /dev/null 2>&1 &
nohup $T/watchdog-exl3.sh $CSV > /dev/null 2>&1 &
./sweep-exl3.sh $T/sweep-exl3a.cfg
PHASES=down,produp ./window.sh > $R/chain-sweep-up.out 2>&1
log "RESULT chain-sweep: done; production $(docker ps --format '{{.Names}} {{.Image}}' | grep prod | tr '\n' ' ')"
