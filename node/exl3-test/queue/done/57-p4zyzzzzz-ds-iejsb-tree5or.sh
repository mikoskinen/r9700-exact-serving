#!/bin/bash
# 57-p4zyzzzzz-ds-iejsb-tree5or.sh (queue item, 11.11.2026, ds-main-port T1; sorts right after the queue head iej-sampled; gpu02,
# all 4 cards, TP4, every radiance container DOWN; ~8 min) -- THE T1 TREE ORACLE AT MOST 16 VERIFY ROWS, ORACLE ONLY (no speed
# phase). The 21-row tree oracle's failure reproduced WITHOUT T1 (tree4: --spec-ngram 20 chain verifies, tree unset: DIFFERENT
# at token 39; 13 / 9 / 6 rows IDENTICAL): the EXL3 routed-expert GEMMs leave their decode band above 16 tokens a step. This
# confirms on the card that the tree path is exact inside it: or15 (2 branches x 5 forced, 16 rows) and or14 (3 x 3 forced,
# 15 rows), two checks each vs bz d0, the verify-path trace on (treetrace.py readings). Body dsm-tree5.sh with XTR_PHASES=OR;
# tag "RESULT xtr5", out runs/xdsm/tree5. The speed run (S AB ID within 15 nodes) is a separate wrapper, staged, not queued.
# IMAGE radiance-local:ds41-tree-2fc434eb-5c13e555-l (= 6b143a96 + the env-gated trace).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-iejsb-tree5or.sh : prints every command; checks the image and every helper; writes nothing.
export XTR_IMAGE=${XTR_IMAGE:-radiance-local:ds41-tree-2fc434eb-5c13e555-l} XTR_ITEM=${XTR_ITEM:-$(basename "$0" .sh)} XTR_PHASES=${XTR_PHASES-OR}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-tree5.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xtr5: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
