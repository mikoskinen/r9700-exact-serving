#!/bin/bash
# 57-p4zyzzzzz-ds-pz-walk3.sh (queue item, 10.11.2026, ds-main-port W1 agent; gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN;
# ~22 min) -- LEVER W1b, A SPEED A/B OF THE FULL-VOCABULARY WALK'S KERNEL (branch ds41-walk eecd93a; body dsm-walk3.sh): dspark_walk 1
# (form 0: a thread reads its own succ row, 16-byte pieces 512 B apart across a wave -- 32.4 / 39.2 / 49.9 us a step at S 1 / 4 / 8
# on the cards) vs dspark_walk 2 (form 1: each block's succ rows staged through LDS as whole 128-byte lines, then the same fma chain
# a thread). The same bits: every key equal on the emulator and in the card selftest, so tokens a step must match and only ms a
# round / tok/s move. Selftest (both forms vs the host walk, form 1's keys = form 0's, launch times of both forms at S 1 / 4 / 8,
# the key gather) -> one NST 5 start, hx-stack env + RADIANCE_DS41_DSPARK_WALK=1 -> smoke under B -> blocks A B B A (A walk 1,
# B walk 2; mb 45 s, c1 essay / code / edit, c4, c8) -> identity under B (28k copy + 20-prompt greedy set vs bz d0).
# Tags "RESULT xwk3 ", "RESULT xwk3 summary: ...", "... done"; out runs/xdsm/walk3.
# IMAGE radiance-local:ds41-walk-eecd93af-5c13e555-l (sha256:10fe458c21de).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-pz-walk3.sh : prints every command; checks the image and files (no server, no GPU), writes nothing.
export XWK_IMAGE=${XWK_IMAGE:-radiance-local:ds41-walk-eecd93af-5c13e555-l} XWK_ITEM=${XWK_ITEM:-57-p4zyzzzzz-ds-pz-walk3}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-walk3.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xwk3: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
