#!/bin/bash
# 57-p4zyzzzzz-ds-hdz-candsweep.sh (queue item, 10.11.2026, ds-main-port; gpu02, all 4 cards; ~35-40 min) -- the zero-code
# pre-screen for W1 (an exact full-vocabulary DSpark Markov walk): the walk's candidates from the head's top 8 / 16 / 32
# (RADIANCE_DS41_DSPARK_CAND, read at declare -> one start an arm) at NST 5 fixed window, -v (proposal log for perpos.py, whose
# essay prompt is fixed since bzzz), per-position acceptance essay / code at 1 user, + mb c4 / c8 tokens a step. Output-exact
# (the walk only proposes). A steep 16 -> 32 slope on essay supports building W1 (research 10.11.: the Markov bias lifts tokens
# far below the head's top 32; top 16 -> 32 gave code 4.00 -> 4.52 tokens a step in bzzz). Image as bzzz (main 967d4eae-l).
# DRYRUN=1 bash ... : prints, runs and writes nothing.
export XDSM_IMAGE=${XDSM_IMAGE:-radiance-local:ds41-main-967d4eae-5c13e555-l} XDSG_ARMS=${XDSG_ARMS-"k8 k16 k32"}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-ddiag2.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xdsg: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
