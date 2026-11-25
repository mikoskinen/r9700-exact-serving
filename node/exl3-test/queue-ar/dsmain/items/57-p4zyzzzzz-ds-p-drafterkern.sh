#!/bin/bash
# 57-p4zyzzzzz-ds-p-drafterkern.sh (queue item, 10.11.2026, ds41-drk agent; gpu02, all 4 cards, TP4, MML 524288, every radiance
# container DOWN; ~70-80 min) -- the DSpark drafter's kernels IN PROCESS on the ds41-drk image: one start, DSpark NST 5
# --draft-window adaptive, gg_gemv 2 (lever D2 + its measured table) from the start; gate, modes, drafter lines, the switch note
# (drafter_mx / drafter_bf 0 at start, 2 refused, 1 / 0 applied at a step), warm-up answers vs stock; then per switch blocks
# A B B A (A = 0, B = 1; the other switch 0 and gg_gemv 2 fixed; POST /admin/fn4 + epoch_bump): stepprobe essay / code (c1),
# an essay probe with acceptance by draft position, editprobe, mb c4 75 s, mb c8 75 s -- tokens a step and acceptance per arm
# logged (both forms keep the stock kernels' bits, so they must not move); identity with both switches on (the 28k copy, the
# c1 greedy set vs bz d0); per switch, logged when the start ends: B vs A pooled (pairs, t95) at c1 essay / code / edit, c4, c8.
#   drafter_mx  moe_gemm_ggml type=mxfp4 (the drafter's routed experts, 472 us a call x 6 a 1-user round) in multi-row forms
#   drafter_bf  gemm_nt bf16 (the drafter's attention / main_proj / shared expert, ~1.4 ms a step) with its next k steps in flight
# Nothing goes into a production line without the user's go. IMAGE radiance-local:ds41-drk-a22a5dfa-5c13e555-l (XDK_IMAGE).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-p-drafterkern.sh : prints every command, runs and writes nothing.
export XDK_IMAGE=${XDK_IMAGE:-radiance-local:ds41-drk-a22a5dfa-5c13e555-l}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-drafterkern.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xdk: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
