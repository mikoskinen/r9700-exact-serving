#!/bin/bash
# chain-full.sh (6.11.2026; user: "mieluiten testataan kattavasti ja profiloidaan myös hidastuksia ja korjataan radiancea") --
# chain-sweep.sh for the full 2.25 bpw model: after window.sh (incl. its produp) production down again (LiteLLM still on xeon), evict,
# tempmon + 95 C watchdog, sweep-exl3.sh sweep-exl3c.cfg (= 3.11. sweep1 rows), then after-sweep.sh (profiling / 1M / capacity kit)
# only if runs/after-sweep.reviewed exists (coordinator review), then window.sh PHASES=down,produp.
set -u
T=$HOME/gpu/ds41/exl3-test; R=$T/runs; L=$R/ds41-exl3.log; cd $T
log() { echo "$(date '+%F %T') $*" | tee -a $L; }
while pgrep -f "[w]indow\.sh" > /dev/null; do sleep 15; done   # 6.11.: window started as "bash window.sh" (old pattern missed it)
log "RESULT chain-full: window.sh finished -> production down again for the sweep"
LITELLM_SWITCHED=1 FREE_MIN_GB=60 PHASES=prodstop,evict ./window.sh > $R/chain-full-stop.out 2>&1
[ -n "$(docker ps --format '{{.Names}}' | grep -E '^radiance-prod')" ] && { log "RESULT chain-full: VIRHE production still running -> no sweep"; exit 1; }
CSV=$R/temps-exl3-$(date +%m%d).csv   # same name as window.sh, so its ph_down stops this tempmon too
rm -f $R/watchdog.stop
nohup $HOME/gpu/bench/tempmon.sh $CSV > /dev/null 2>&1 &
nohup $T/watchdog-exl3.sh $CSV > /dev/null 2>&1 &
./sweep-exl3.sh $T/sweep-exl3c.cfg
if [ -x $T/after-sweep.sh ] && [ -e $R/after-sweep.reviewed ] && [ ! -e $R/sweep.stop ]; then
  log "RESULT chain-full: after-sweep.sh start"; $T/after-sweep.sh > $R/after-sweep.out 2>&1; log "RESULT chain-full: after-sweep.sh exit $?"
else log "RESULT chain-full: after-sweep.sh skipped (not reviewed / missing / sweep.stop)"; fi
PHASES=down,produp ./window.sh > $R/chain-full-up.out 2>&1
log "RESULT chain-full: done; production $(docker ps --format '{{.Names}} {{.Image}}' | grep prod | tr '\n' ' ')"
