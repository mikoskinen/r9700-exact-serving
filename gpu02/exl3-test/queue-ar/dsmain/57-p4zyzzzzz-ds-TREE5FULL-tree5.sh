#!/bin/bash
# 57-p4zyzzzzz-ds-<key>-tree5.sh (STAGED, NOT QUEUED; 11.11.2026, ds-main-port T1; rename to the wanted sort key when released;
# gpu02, all 4 cards, TP4, every radiance container DOWN; ~60 min, ~8 min when the oracle fails) -- LEVER T1, THE DSPARK TREE
# WITHIN THE EXL3 DECODE BAND (at most 15 nodes = 16 verify rows at one user): oracle first (or15 2 x 5, or14 3 x 3, forced),
# then, only when every check is IDENTICAL with branch trees verified, the main start: S screen T0 F1 F2 F3d3 F1d2 P2 P3d3 T0
# (c1 essay + code) with the T0-at-NST-20 cost vs hx-stack, AB ABBA T0 vs the pick (c1 essay / code / edit, c4, c8), ID under
# B and F2 vs bz d0. Body dsm-tree5.sh; tag "RESULT xtr5", out runs/xdsm/tree5.
# IMAGE radiance-local:ds41-tree-2fc434eb-5c13e555-l.
export XTR_IMAGE=${XTR_IMAGE:-radiance-local:ds41-tree-2fc434eb-5c13e555-l} XTR_ITEM=${XTR_ITEM:-$(basename "$0" .sh)}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-tree5.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xtr5: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
