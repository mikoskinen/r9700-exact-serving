#!/bin/bash
# 57-p4zyzzzzz-ds-s-tree.sh (queue item, 10.11.2026, ds41-acc agent, lever T1; gpu02, all 4 cards, TP4, MML 524288, every radiance
# container DOWN; ~30 min) -- a TREE AT POSITION 1 for DeepSeek-V4.1's DSpark, measured as a SHADOW: the walk's 2nd, 3rd and
# 4th best first drafts, each continued by the walk (RADIANCE_DS41_DSPARK_ALT=3, op dspark_alt), are compared verify by verify
# with the tokens the target emits; a verify whose chain took no draft and whose emitted token starts a branch would have
# emitted min(match, depth) more tokens under a tree of that branch (exact for greedy). The verify itself stays a chain: the
# trunk's compressor, indexer and Engram are not tree-aware, so no tree is verified. One start (DSpark NST 5 --draft-window
# adaptive, gg_gemv 2, dspark_conf 0), in-process blocks dspark_alt A 0 / B 1, ABBA: c1 essay / code / edit, mb c4 / c8, each
# with chain and tree tokens a step (1-3 branches x depth 1-5) and the branches' own cost (A vs B); identity under B vs bz d0.
# No cc_sim, no 2.9, no BetterBench. Nothing goes into a production line without the user's go.
# IMAGE radiance-local:ds41-acc-1f2f03b5-5c13e555-l (XAC_IMAGE).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-s-tree.sh : prints every command, runs and writes nothing.
export XAC_IMAGE=${XAC_IMAGE:-radiance-local:ds41-acc-1f2f03b5-5c13e555-l} XAC_MODE=tree
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-acc.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xac: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
