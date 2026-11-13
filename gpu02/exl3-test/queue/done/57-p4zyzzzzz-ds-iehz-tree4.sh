#!/bin/bash
# 57-p4zyzzzzz-ds-iehz-tree4.sh (queue item, 11.11.2026, ds-main-port T1 diagnosis; sorts right after the queue head; gpu02, all 4
# cards, TP4, every radiance container DOWN; ~16 min) -- THE VERIFY ROW-COUNT SWEEP, NO TREE: is a 21-row verify bit-identical
# to the 1-row decode? tree3 found the T1 oracle's first wrong token at one row (step 12, main-chain node 2) of a 21-row tree
# verify, with every other row on a correct path right and the same tree path at 6 rows IDENTICAL. Four starts on the tree
# image with the tree UNSET (= ds41-int), no draft head, --spec-ngram K --debug-draft-oracle (chain verifies of K + 1 rows of
# the plain run's ids): K = 20 / 12 / 8 / 5, one oracle check each vs bz d0. Body dsm-ngsweep.sh (yaml ds41-tree3-tp4.yaml,
# oracle.py); tag "RESULT xng", out runs/xdsm/ngsweep.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-iehz-tree4.sh : prints every command; checks the image and the helpers; writes nothing.
export XNG_IMAGE=${XNG_IMAGE:-radiance-local:ds41-tree-2fc434eb-5c13e555-l} XNG_ITEM=${XNG_ITEM:-$(basename "$0" .sh)}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-ngsweep.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xng: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
