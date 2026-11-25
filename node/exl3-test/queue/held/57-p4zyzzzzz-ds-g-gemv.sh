#!/bin/bash
# 57-p4zyzzzzz-ds-g-gemv.sh (queue item, 10.11.2026, ds41-gg lever 1 agent; gpu02, all 4 cards, TP4, MML 524288, every
# radiance container DOWN; ~70 min) -- lever 1 (gemm_nt_ggml's decode band at the measured per-(type, N, K, M) form: rows a
# wave and grid cap from one card's bench, 322 calls a decode step, 16 ms of a 44 ms c8 step) A/B'd IN ONE PROCESS on the
# ds41-gg image: one start at gg_gemv 0 (RADIANCE_DS41_GG_GEMV unset = the stock launch byte for byte), gate, modes, the
# switch note (the hook found, applied at a step), warm-up answers vs stock, state probe; then blocks A B B A A B (POST
# /admin/fn4 {gg_gemv, epoch_bump}; each block stepprobe essay/code 512, mb c4 75 s, mb c8 75 s) and the B-vs-A summary
# per load point (aggregate and per-stream p50) pooled over blocks; identity under each setting (the 28k copy, the
# 20-prompt greedy set at c1; B vs A byte for byte and vs the stock references); then one --profile-ops start, A / B
# windows, rank 0's gemm_nt_ggml us a call (PROFILED, not a speed). EXACT by construction (the forms move only which wave
# and block compute an output); NST 0, the t3 env and serve line of dsm-serve.sh. Nothing goes into a production line
# without the user's go. IMAGE radiance-local:ds41-gg-4d70490d-5c13e555-l (XGG_IMAGE).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-g-gemv.sh : prints every command, runs and writes nothing.
export XGG_IMAGE=${XGG_IMAGE:-radiance-local:ds41-gg-4d70490d-5c13e555-l}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-gemv.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xgg: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
