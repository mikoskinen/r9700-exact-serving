#!/bin/bash
# watchdog-exl3.sh CSV -- test-window guard (= ~/gpu/ds41/ds41-watchdog.sh): if any card's junction or memory temperature in the
# tempmon CSV reaches 95 C, take the EXL3 test server down and log a RESULT VIRHE line. Stops when runs/watchdog.stop exists.
. "$(dirname "$0")/env.sh"; CSV=$1
while [ ! -e $R/watchdog.stop ]; do
  hot=$(tail -n 8 $CSV 2>/dev/null | awk -F, '$5 ~ /^[0-9]+$/ && ($5 >= 95 || $6 >= 95) {print; exit}')
  if [ -n "$hot" ]; then
    echo "$(date '+%F %T') RESULT watchdog: VIRHE temperature >= 95 C ($hot = time,pci,power,edge,junction,mem,fan,sclk); EXL3 server taken down" >> $L
    docker compose --project-directory $C -f $T/ds41-exl3-tp4.yaml down >> $R/watchdog.log 2>&1
    # 12.11. ds-main-port: items now start under per-item compose projects (ghost workaround), so also stop BY CONTAINER NAME
    docker stop -t 20 radiance-ds41-exl3 >> $R/watchdog.log 2>&1; docker rm -f radiance-ds41-exl3 >> $R/watchdog.log 2>&1
    # 13.11. coordinator: the Qwen test containers (radiance-fntest*) were not covered by this guard
    for cn in $(docker ps --format '{{.Names}}' | grep -E '^radiance-fntest'); do docker stop -t 20 $cn >> $R/watchdog.log 2>&1; done
    exit 1
  fi
  sleep 5
done
