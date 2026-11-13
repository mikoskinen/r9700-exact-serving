#!/bin/bash
# 57-p4zyzzzzz-ds-hszzzzzzzzza-small.sh (queue item, 10.11.2026, ds-main-port small-levers agent; gpu02, all 4 cards, TP4, MML 524288,
# every radiance container DOWN; ~112 min (selftest ~3, NST 0 ~29, NST 5 ~42 + ctxchk ~9, KVWIN follow-up start ~25, stamped c4 conf screen ~10), more only if the combined result loses consistently somewhere: leave-one-out blocks)
# -- THE EXACT SMALL LEVERS, FOLDED (user 10.11.: no lever dropped for being small), on radiance-local:ds41-small-2b039035-5c13e555-l
# (branch ds41-small 2b03903 = ds41-int c05de27 + ds41-idxb (I1) + conf (ds41-acc's dspark_conf + the load gate dspark_conf_rows)
# + ds41-small-k1ae (K1 a / e) + ds41-small-ctx (depth-0 context skip) + ds41-int2 (drafter_mx) + ds41-kvfp8 (KVWIN); D3 was in
# int; body dsm-small.sh, log tag xsm, out runs/xdsm/small). Each fn4 lever is an in-process switch: arm B = every lever on, arm A
# = every lever off (= the int2 line), in the same start, blocks ABBA; KVWIN (a start env) is a follow-up NST 5 start against B.
#   selftest (card 0, no server): the new folds byte-compared on the card -- a defect stops the item;
#   START n0 (NST 0; d3 i1 k1a k1e): ABBA c1 essay/code, c4, c8 (mb 75 s); the 136k cold prefill under A and B; identity per arm;
#   START n5 (DSpark adaptive + D7 + hybrid + drafter_mx pinned; d3 i1 k1a k1e ctx conf): settle A, ABBA c1/edit/c4/c8, D7
#     re-armed every block; identity per arm; the context skip under forced depth 0 (draft_depth0 2): identity + refill acceptance.
#   START n5-env (KVWIN on + every fn4 lever on): settle B, B B; 136k; identity under B and B with ds41_kvfold 0; a prefix-cache hit.
#   STAMPED c4 screen (RADIANCE_AQL_STAMP=1, not a speed): dspark_conf 0 / 1 / 101 / 0 windows, ms a step by kernel family.
#   Stack in all: gg_gemv 2, aql_acq 2, K1 c/d/f, turnover 0 (+ D7 + hybrid + drafter_mx at NST 5). No cc_sim, no 2.9.
# The lever table in dsm-small.sh is data (one line a lever; kind env = a follow-up start against B).
# XSM_IMAGE, XSM_NST (n0 n5), XSM_ORDER (ABBA), XSM_DUR, XSM_PERLEVER (auto | 1 | 0), XSM_LOSS_PCT, XSM_DEEP, XSM_ON, XSM_CTXCHK, XSM_STAMP, XSM_DRAFTER_MX,
# XSM_TABLE_EXTRA, XSM_ENVORDER: set them by editing the export line below before moving this file into queue/ (the queue runner
# passes no environment).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-hszzzzzzzzza-small.sh : prints every command; checks the image and the yaml (no server, no GPU), writes nothing.
export XSM_IMAGE=${XSM_IMAGE:-radiance-local:ds41-small-2b039035-5c13e555-l} XSM_ITEM=${XSM_ITEM:-57-p4zyzzzzz-ds-hszzzzzzzzza-small}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-small.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xsm: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
