#!/bin/bash
# 57-p4zyzzzzz-ds-hxzzz-tree2.sh (queue item, 11.11.2026, ds-main-port T1 re-run; sorts right after hxzz-walk3, before
# j-shexplane; gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~55 min, ~10 min when the oracle fails) --
# LEVER T1 AGAIN, THE DSPARK TREE VERIFY AT POSITION 1, on the fixed image. The 10.11. item (57-p4zyzzzzz-ds-hszzzzzzzzzz-tree,
# runs/xdsm/tree) produced no data: the engine stopped at the first tree verify step in all three starts ("op 'qsa_tail_store'
# ... names buffer 'tree_slot_pos', which this op never declared"; the guard printed that as SLOW 999, the oracle as
# DIFFERENT ()). ds41-tree b2e1064 declares the two tree buffers the ordinary ops read on a tree verify (qsa_tail_store:
# tree_slot_pos; the ratio > 1 entry_store ops: comp_page_main); 6b143a9 adds the /stats tree counters (tree_verifies / _nodes /
# _wide / _branch_nodes / _branch_paths / _branch_accepted). Body dsm-tree2.sh = dsm-tree.sh + harness fixes (engine errors
# named as such, the tree-0 at NST 20 cost vs hx-stack NST 5 as RESULT lines, oracle ERROR vs DIFFERENT, /metrics + /stats tree
# counters, DRYRUN helper checks); yaml ds41-tree-tp4.yaml, driver treeblocks.py, oracle.py (unchanged).
# ORDER: the tree oracle FIRST (or20 fixed window / or20a adaptive, 3 x 5 forced, two checks each: IDENTICAL to bz d0's ids with
# drafts rejected and branch trees verified, /stats tree_wide > 0); then, only when every check passes, the main start: S screen
# (T0 F1 F2 F3 F1d2 P3 T0, c1 essay + code; the T0-at-NST-20 cost vs hx-stack NST 5), AB (ABBA T0 vs the pick, mb 75 s: c1
# essay / code / edit, c4, c8), ID (identity under B and F3 vs bz d0). Tag "RESULT xtr2 ", out runs/xdsm/tree2.
# IMAGE radiance-local:ds41-tree-6b143a96-5c13e555-l (XTR_IMAGE). The queue runner passes no environment: set any XTR_* on
# the export line before moving this file into queue/.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-hxzzz-tree2.sh : prints every command; checks the image is here and every helper resolves
# (no server, no GPU), writes nothing.
export XTR_IMAGE=${XTR_IMAGE:-radiance-local:ds41-tree-6b143a96-5c13e555-l} XTR_ITEM=${XTR_ITEM:-$(basename "$0" .sh)}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-tree2.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xtr2: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
