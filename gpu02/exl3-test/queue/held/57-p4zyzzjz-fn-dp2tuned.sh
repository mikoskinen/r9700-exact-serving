#!/bin/bash
# 57-p4zyzzja wrapper (9.11.2026): 110 trimmed to four arms (~5 h): s15 reference, s15n (nearest on the production line,
# directly deployable), t3 (the 1.17-based tuned engine alone), t3hs (heat placement + nomix staging, the DS lessons combined).
export X110_ARMS=${X110_ARMS:-"s15 s15n t3 t3hs"}
for f in "$(dirname "$0")/110-fn-dp2tuned.sh" "$HOME/gpu/ds41/exl3-test/queue-ar/110-fn-dp2tuned.sh"; do [ -e "$f" ] && exec bash "$f"; done
echo "$(date '+%F %T') RESULT fn4 110: VIRHE 110-fn-dp2tuned.sh not found" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1
