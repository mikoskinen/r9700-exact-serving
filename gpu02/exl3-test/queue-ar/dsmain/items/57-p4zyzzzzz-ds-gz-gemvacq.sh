#!/bin/bash
# 57-p4zyzzzzz-ds-gz-gemvacq.sh (queue item, 10.11.2026, ds41-gg agent, D1 + D2; gpu02, all 4 cards, TP4, MML 524288,
# every radiance container DOWN; ~90 min, ~65 with XGG_ORDER=ABBA) -- three exact levers A/B'd IN ONE PROCESS on one
# start of the ds41-gg D1 + D2 image: gg_gemv 0 / 1 (gemm_nt_ggml's decode band at the measured per-(type, N, K, M) form;
# the image's table, see its gg-PROVENANCE), gg_gemv 0 / 2 (lever D2, the multi-row form) and aql_acq 0 / 2 (the acquire
# rule, rule-w), each pair in blocks A B B A A B with the other switches pinned (POST /admin/fn4 {switch, pinned,
# epoch_bump}; each block stepprobe essay/code 512, mb c4 75 s, mb c8 75 s); per pair the B-vs-A summary per load point
# (aggregate and per-stream p50) pooled over blocks with t95; identity for every value (the 28k copy and the 20-prompt
# greedy set at c1, each vs all-0 byte for byte). The switch note checks the D1 reporting first (gg_gemv as libr4d
# applies it, 3 refused, 1 / 2 / 0 applied). NST 0 (a drafting phase would need a second start). Nothing goes into a
# production line without the user's go. IMAGE radiance-local:ds41-gg-8b311fcc-5c13e555-l (XGG_IMAGE).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-gz-gemvacq.sh : prints every command, runs and writes nothing.
export XGG_IMAGE=${XGG_IMAGE:-radiance-local:ds41-gg-8b311fcc-5c13e555-l}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-gemvacq.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xgg2: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
