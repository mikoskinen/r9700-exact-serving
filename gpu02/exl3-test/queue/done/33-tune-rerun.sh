#!/bin/bash
# 33-tune-rerun.sh (6.11. 13.20) -- re-run the 30-tune512 rows (32-tune512b run) that ended "VIRHE slow state": their gate retry
# never restarted the server (compose down failed without ARGS -> "healthy in 0 s", the same process measured twice). lib-pr266.sh
# is fixed (docker rm -f); 30-tune512.sh sources it fresh. ROWS limits the rows; the choice at the end still reads every row's files.
Q=$HOME/gpu/ds41/exl3-test/queue; L=$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
start=$(grep -n "pr266 30-tune512: start (base host pool 19456" $L | tail -1 | cut -d: -f1)
rows=$(tail -n +${start:-1} $L | grep -oE "pr266 30-tune512 [a-z0-9-]+: VIRHE slow state" | awk '{print $3}' | tr -d ':' | sort -u | paste -sd,)
echo "$(date '+%F %T') RESULT pr266 33-tune-rerun: rows to re-run: ${rows:-none}" | tee -a $L
[ -n "$rows" ] || exit 0
ROWS=$rows bash $Q/done/30-tune512.sh
