#!/bin/bash
# 57-p4zyzzzzz-ds-hszzzzzzzzzz-tree.sh (queue item, 10.11.2026, ds-main-port T1 agent; sort key picked at queue time to
# run right after hszzzzzzzzz-gpfab2; gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~55 min) -- LEVER T1, THE DSPARK
# TREE VERIFY AT POSITION 1 (branch ds41-tree on ds41-int c05de27; body dsm-tree.sh, yaml ds41-tree-tp4.yaml, driver
# treeblocks.py): the walk's 5-node main chain plus up to 3 alternative branches at position 1 (depth up to 5) verified as one
# exact tree (compressor, indexer, Engram, window attention tree-aware). ONE MAIN START = the hx-stack NST 5 start (t3 env,
# gg_gemv 2, aql_acq 2 rule-w, K1 folds, D7 depth 0 + adaptive window, hybrid 1 match 8, turnover 0) at NST 20 +
# RADIANCE_DS41_DSPARK_TREE=1 (start values RADIANCE_DS41_DSPARK_TREE_B 0 / _L 5 / _FORCE 0); in process dspark_tree /
# dspark_tree_depth / dspark_tree_force (POST /admin/fn4):
#   S   screen c1 essay + code, one block an arm: T0 (tree 0) F1 F2 F3 (1-3 branches x 5, forced) F1d2 (1 x 2, forced) P3 (3 x 5,
#       priced by D7) T0; the verify-row cost (ms a step vs rows a step, least squares); the pick (best geometric mean of essay
#       and code tok/s vs T0);
#   AB  ABBA A = T0 vs B = the pick, mb 75 s: c1 essay / code / edit, c4, c8;
#   ID  28k copy + 20-answer greedy set at c1 under B and F3 vs bz d0 (identity vs NST 0);
#   OR  the tree oracle: two starts (3 x 5 forced at start, NST 20, --debug-draft-oracle d0's ids; fixed window / adaptive), two
#       checks each, 256 greedy ids identical to NST 0's.
# Summary: tokens a step and tok/s B vs A, vs hx-stack's NST 5 stack, hd-hybrid and NST 0 + gg_gemv 2, distance to the targets
# (1u >= 90, c4 >= 160, c8 >= 200), the row cost, identity and oracle verdicts. Tag "RESULT xtr ", out runs/xdsm/tree.
# IMAGE radiance-local:ds41-tree-9d0ce178-5c13e555-l (XTR_IMAGE; id e7a1d92905ff). Set the env by editing the export line before moving this file
# into queue/ (the queue runner passes no environment). XTR_PHASES ("S AB ID OR"), XTR_PICK, XTR_SORDER, XTR_DUR, XTR_SGEN,
# XTR_HYB, XTR_DRAFTER_MX, XTR_TREE_B / _L / _FORCE, XTR_VRAMENV: see dsm-tree.sh.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-hszzzzzzzzzz-tree.sh : prints every command; checks the image is here (no server, no GPU), writes nothing.
export XTR_IMAGE=${XTR_IMAGE:-radiance-local:ds41-tree-9d0ce178-5c13e555-l} XTR_ITEM=${XTR_ITEM:-$(basename "$0" .sh)}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-tree.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xtr: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
