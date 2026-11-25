#!/bin/bash
# 57-p4zyzzzzz-ds-iejzzzt-tree6.sh (queue item, 11.11.2026, ds-main-port T1 at the coordinator's request; right after iejzzz-sd;
# gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~40 min, ~5 min when the oracle fails) -- LEVER T1 SPEED AT
# THE SHAPE THAT PASSED THE ORACLE: B = P2 (2 branches x depth 5, priced by D7, at most 15 nodes = 16 verify rows at one user,
# inside the EXL3 decode band) vs A = dspark_tree 0, in-process ABBA (mb 75 s), on the stack2 NST 5 levers the tree image has
# (hx-stack pins + hybrid 1 + lever DL lend 1 / backoff 3 + the VRAM levers at start; int3's walk / drafter_mx / small set /
# GPF are not on this ds41-int-based image). Oracle (or15) first as the gate. Loads: 1u essay / code / edit at temperature 0
# and at the server default (no temperature sent, seed 1234), mb c4 / c8; identity under B at temperature 0.
# Body dsm-tree6.sh (yaml ds41-tree6-tp4.yaml, treeblocks2.py, da/daprobe.py, oracle.py, treetrace.py); tag "RESULT xtr6",
# out runs/xdsm/tree6. IMAGE radiance-local:ds41-tree-2fc434eb-5c13e555-l (ds41-tree 2fc434e = 9d0ce17 + b2e1064 the buffer
# declarations + 6b143a9 the /stats tree counters + the env-gated trace).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-iejzzzt-tree6.sh : prints every command; checks the image and every helper; writes nothing.
export XTR_IMAGE=${XTR_IMAGE:-radiance-local:ds41-tree-2fc434eb-5c13e555-l} XTR_ITEM=${XTR_ITEM:-$(basename "$0" .sh)}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-tree6.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xtr6: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
