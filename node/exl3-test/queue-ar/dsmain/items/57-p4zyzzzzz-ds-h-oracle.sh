#!/bin/bash
# 57-p4zyzzzzz-ds-h-oracle.sh (queue item, 10.11.2026, ds41-gg agent; gpu02, all 4 cards, TP4, MML 524288, every radiance
# container DOWN; ~15 min) -- the verify-path oracle BESIDE THE DSPARK HEAD (bz's oracle start was refused: --spec-ngram
# with a declared head). The ds41-gg image's harness replaces each proposal the head publishes with the oracle's ids (30 %
# corrupted), so every verify runs at the head's 6-row shape with forced accepts and rejects: two starts, NST 5 and NST 5 +
# --draft-window adaptive, each oracle.py check (one prompt, 256 greedy ids, twice) against bz's d0 reference
# (runs/xdsm/draft/oracle-ids.txt, NST 0, main image): IDENTICAL is the pass; the harness must also have drafted and been
# rejected (draft_n > accepted > 0). Under the harness no step is issued ahead (not a speed). IMAGE
# radiance-local:ds41-gg-4d70490d-5c13e555-l (XGG_IMAGE).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-h-oracle.sh : prints every command, runs and writes nothing.
export XGG_IMAGE=${XGG_IMAGE:-radiance-local:ds41-gg-4d70490d-5c13e555-l}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-oracle.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xgh: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
