#!/bin/bash
# 57-p4zyzzzzz-ds-ief-tree3.sh (queue item, 11.11.2026, ds-main-port T1 diagnosis; sorts right after the queue head i-e10, before
# iz-nvgf; gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~12 min when the oracle fails (3 oracle starts),
# ~60 min when it passes) -- LEVER T1, THE TREE ORACLE WITH A VERIFY-PATH TRACE. tree2 (11.11. 02:46, image ds41-tree-6b143a96)
# ran past the 10.11. crash but its oracle was DIFFERENT: first diff at token 39 of 256, 31 drafts accepted of 4480 (0.14 a step;
# the chain harness accepts 2.08 a step on the same ids). This item runs the oracle first with ds41-tree 2fc434e's
# RADIANCE_DEBUG_TREE_STEPS=400 trace and treetrace.py's readings (harness ids / staging / row 0 / rows on a correct path by
# depth / acceptance rule / FIRSTBAD), in three starts: or20t0 (tree declared, dspark_tree 0: the chain of 5 through the tree
# verify path), or20 (3 x 5 forced, fixed window), or20a (adaptive). The speed phases (S AB ID as tree2) run only when every
# oracle check is IDENTICAL with trees verified. Body dsm-tree3.sh, yaml ds41-tree3-tp4.yaml, treetrace.py; tag "RESULT xtr3",
# out runs/xdsm/tree3.
# IMAGE radiance-local:ds41-tree-2fc434eb-5c13e555-l (XTR_IMAGE; = 6b143a96 + the env-gated trace). The queue runner passes no
# environment: set any XTR_* on the export line before moving this file into queue/.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-ief-tree3.sh : prints every command; checks the image is here and every helper resolves.
export XTR_IMAGE=${XTR_IMAGE:-radiance-local:ds41-tree-2fc434eb-5c13e555-l} XTR_ITEM=${XTR_ITEM:-$(basename "$0" .sh)}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-tree3.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xtr3: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
