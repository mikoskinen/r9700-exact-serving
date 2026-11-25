#!/bin/bash
# 57-p4zyzzzzz-ds-<x>-gpfab.sh (queue item, 10.11.2026, lever GPF agent, branch ds41-gpf; gpu02, all 4 cards, TP4, MML 524288,
# every radiance container DOWN; ~45 min, or ~0 when the bench found no winning form) -- lever GPF IN PROCESS on the ds41-gpf image
# (= ds41-int2 745ce76 + lever GPF; body dsm-gpf.sh, log tag xgq, out runs/xdsm/gpfab). Runs right after the one-card bench item
# 57-p4zyzzzzz-ds-<x>-gpbench (tag xgp, runs/xdsm/gpbench) and reads its result: the form table = gpperf.txt's GPTAB line (the
# best byte-identical form a (type, shape) that beat the stock WMMA kernel by >= 1 % at M 4096), declared at start as
# RADIANCE_DS41_GG_PF_TABLE; GPTAB '-' (no form won) or a byte check that did not pass -> nothing runs.
#   One NST 0 start (the stack: t3 env + gg_gemv 2 + aql_acq 2 + K1 c / d / f, turnover 0), gg_pf 0 at start:
#   A B B A x 2 (A gg_pf 0 stock, B gg_pf 1 the table): mb c8 + fresh prefills ~16k / ~64k / ~128k a block; c4 ABBA; mb pf ABBA;
#   the 136k cold prefill under A and B (TTFT; answer vs dsm-225); identity A / B (28k copy + greedy c1 vs bz d0).
# XGQ_IMAGE, XGQ_TABLE (overrides the bench's GPTAB), XGQ_ROUNDS (2), XGQ_DUR (40), XGQ_PFD (16384,65536,131072). Set them by editing
# the export line below before moving this file into queue/ (the queue runner passes no environment).
# DRYRUN=1 [XGQ_TABLE=q6_K:8192x1280=422] bash 57-p4zyzzzzz-ds-<x>-gpfab.sh : prints every command; checks the image and the
# yaml (no server, no GPU), writes nothing.
export XGQ_IMAGE=${XGQ_IMAGE:-radiance-local:ds41-gpf-SHA8-5c13e555-l} XGQ_ROUNDS=${XGQ_ROUNDS:-2} XGQ_TABLE=${XGQ_TABLE:-}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-gpf.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xgq: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
