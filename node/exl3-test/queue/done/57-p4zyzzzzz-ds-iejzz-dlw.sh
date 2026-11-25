#!/bin/bash
# 57-p4zyzzzzz-ds-iejzz-dlw.sh (queue item, 11.11.2026, ds-main-port DLW agent; sorts right after the Z0 items (iej-z0ab and its
# follow-up), before iek-fn-x3arm; gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~75 min) -- LEVER DLW, A
# LIGHTER DSPARK DRAFTER (tag "RESULT xdlw ", out runs/xdsm/dlw, body queue-ar/dsmain/dlw/dsm-dlw.sh, summary dlwsum.py).
# The drafter's dense linears (fc + per stage q_a / q_b / kv / o_a / o_b / shared gate_up / down: plain bf16 in the container,
# ~0.43 GiB a rank) served from a weight sidecar (RADIANCE_WEIGHT_SIDECAR; image radiance-local:ds41-dlw-df860485-5c13e555-l (sha256:867d4364fb7e) =
# the ds41-int3 1116248a image + bin/radiance + bin/rad-sidecar of ds41-dlw; unset = int3, hostcheck graph dump IDENTICAL) as
# llama.cpp q8_0 (B, dspark-dlw-q8_0.rad) or q6_K (C, dspark-dlw-q6_K.rad, the trunk linears' own form), both beside the .rad.
# Only proposals change; verify keeps the output exact -- identity (28k copy + 20-prompt greedy set c1 vs bz d0) on B and C.
# CROSS-START A B C A (the sidecar is start-time), NST 5, hxzzy-stack2's env as it ran and its pin; per start one z0blocks.py
# block (c1 essay / code / essay by position / edit at temperature 0; mb c4 / c8 natural 75 s; per-phase VRAM units, lent units,
# hit, stream, depth-0 share) + the 1-user probe at the server-default sampling (seed 1234, essay / code / edit, 2 runs).
# NO cc_sim, no 2.9. No power / host / fan settings; production compose read only. Exact: proposals only (verify).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-iejzz-dlw.sh : prints every command; checks the image (its literals), every helper (files, parse,
# functions, --help / selftest), the sidecars (sha256) and the yaml; no server, no GPU, writes nothing.
export DLW_IMAGE=${DLW_IMAGE:-radiance-local:ds41-dlw-df860485-5c13e555-l} DLW_ITEM=${DLW_ITEM:-57-p4zyzzzzz-ds-iejzz-dlw} DLW_ORDER=${DLW_ORDER:-ABCA}
export DLW_SHA_Q8=${DLW_SHA_Q8:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
export DLW_SHA_Q6=${DLW_SHA_Q6:-d0f54212500ffc0317125a6d8fae1efd25113725b56d42d086f9d27ef734902b}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dlw/dsm-dlw.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xdlw: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
