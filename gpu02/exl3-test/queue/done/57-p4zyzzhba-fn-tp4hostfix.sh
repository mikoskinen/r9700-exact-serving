#!/bin/bash
# 57-p4zyzzhba wrapper (9.11.2026): 103e with the plan A0 T T A1 X1 -- G dropped (KFD already disallows GFXOFF on GC 12
# while a process has queues, so a debugfs disallow changes only the gaps), X1 = pcie dpm manual + level 2 (Xeon's lock).
# Conditions are set by hand with sudo on each "103e want" line; see 103e-fn-tp4hostfix.sh.
export X103E_PLAN=${X103E_PLAN:-"A0 T T A1 X1"}
for f in "$(dirname "$0")/103e-fn-tp4hostfix.sh" "$HOME/gpu/ds41/exl3-test/queue-ar/103e-fn-tp4hostfix.sh"; do [ -e "$f" ] && exec bash "$f"; done
echo "$(date '+%F %T') RESULT fn4 103e: VIRHE 103e-fn-tp4hostfix.sh not found" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1
