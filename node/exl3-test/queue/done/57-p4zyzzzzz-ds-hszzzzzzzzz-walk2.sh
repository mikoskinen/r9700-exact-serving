#!/bin/bash
# 57-p4zyzzzzz-ds-hszzzzzzzzz-walk2.sh (queue item, 10.11.2026, ds-main-port W1 agent; gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN;
# ~45 min) -- LEVER W1, THE DSPARK MARKOV WALK OVER THE WHOLE VOCABULARY (branch ds41-walk 0281f7c = ds41-int c05de27 + W1; body
# dsm-walk.sh). Today's walk re-ranks the head's top 32; W1 walks every id as the reference / exllamav3 / vLLM do: a dspark_walk_step
# a step on each rank's vocabulary shard + one all_gather of a u64 key a sequence (RADIANCE_DS41_DSPARK_WALK=0 declares it at start,
# fn4 dspark_walk 0 / 1 in process). Output-exact by construction (proposals only). Steps: the GPU selftest of the kernel vs a host
# walk (+ a step's launch time at the TP4 shard); ONE NST 5 start with hx-stack's env (gg_gemv 2, aql_acq 2, K1 folds, D7 depth 0,
# hybrid 1 match 8, turnover 0, adaptive window), in-process blocks A B B A (A dspark_walk 0, B dspark_walk 1): c1 essay / code /
# edit with per-position acceptance, mb c4 / c8 75 s, + a fixed-window-5 essay / code position diagnostic; identity B then A (28k
# copy + 20-prompt greedy set c1 vs bz d0); the oracle start (NST 5 fixed window, walk 1, --debug-draft-oracle, 2 checks vs bz d0).
# RE-RUN after the fix (fix-failed-steps): 57-p4zyzzzzz-ds-hszzzzzz-walk (image ds41-walk-be5bf4c3) died at its first B query
# pass -- ar_gather_4rank refused the walk's 8-byte key row (row=2 f32, step 4251); ds41-walk 0281f7c gathers 16-byte key slots
# (test: r4d_emu_ds41walk holds the issued gather to libr4d's admission; the selftest runs ar_gather_4rank at that geometry on
# the cards first and stops the item if it fails; a walk smoke under B precedes the blocks).
# Tags "RESULT xwk ", "RESULT xwk summary: ...", "... done"; out runs/xdsm/walk2 (walk = the failed run).
# IMAGE radiance-local:ds41-walk-0281f7cc-5c13e555-l (sha256:554926e0b0b7).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-hszzzzzzzzz-walk2.sh : prints every command; checks the image and files (no server, no GPU), writes nothing.
export XWK_IMAGE=${XWK_IMAGE:-radiance-local:ds41-walk-0281f7cc-5c13e555-l} XWK_ITEM=${XWK_ITEM:-57-p4zyzzzzz-ds-hszzzzzzzzz-walk2} XWK_OUT=${XWK_OUT:-walk2}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-walk.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xwk: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
