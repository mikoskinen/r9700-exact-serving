#!/bin/bash
# 57-p4zyzzzzz-ds-q-dascreen.sh (queue item, 11.11.2026, ds-main-port DA agent; gpu02, all 4 cards, TP4, MML 524288, every radiance
# container DOWN; ~14-16 min) -- LEVER DA SCREEN (drafter adaptation = self-distilling the DSpark drafter to the EXL3 2.25 target;
# nothing is trained here). The rejection anatomy of today's drafter on today's target (body dsm-dascreen.sh): one NST 5 start, the
# hx-stack env + RADIANCE_DS41_DSPARK_WALK=1 (W1), -v; fn4 fixed window 5, every proposal DSpark's; 20 prose + 6 code chat requests
# at c1 (the probe's essay / code first, then synthetic prompts; never a user transcript), 384 tokens, logprobs top 20 + each
# block's proposals from the -v log -> per block the first rejection classed by the TARGET's own distribution (A confident
# p_top1 >= 0.8 / B near tie within ln 2 / C other), acceptance by target confidence, and the bounds on tokens a step: now / +A /
# +A+B (DA's addressable upper bound) / distribution-matched / perfect 6.0, each as est. tok/s vs the 1-user target 90 tok/s.
# Exact: logprobs on vs off text, 28k copy + 20-prompt greedy set vs bz d0. Tags "RESULT xda2 ", "RESULT xda2 summary: ...",
# "... done"; out runs/xdsm/dascreen. IMAGE radiance-local:ds41-walk-eecd93af-5c13e555-l (sha256:10fe458c21de).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-q-dascreen.sh : prints every command; checks the image, every helper and file, runs daanat.py's
# selftest (CPU); no server, no GPU, writes nothing.
export XDA_IMAGE=${XDA_IMAGE:-radiance-local:ds41-walk-eecd93af-5c13e555-l} XDA_ITEM=${XDA_ITEM:-57-p4zyzzzzz-ds-q-dascreen}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-dascreen.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xda2: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
