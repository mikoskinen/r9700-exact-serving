#!/bin/bash
# 57-p4zyzzzzz-ds-i-argall.sh (queue item, 10.11.2026, ds41-d3 agent, lever D3; gpu02, all 4 cards, TP4, MML 524288, every
# radiance container DOWN; ~70 min) -- the MoE gather + its all-reduce as one kernel up to 48 rows
# (RADIANCE_DS41_ARGATHER_MAX=48: the one-shot below 7 rows, the new two-shot gather form from 7), A/B'd IN ONE PROCESS
# against today's 4-row bound by the switch ds41_argather (4 = A, 48 = B; blocks A B B A A B: stepprobe c1, mb c4, mb c8),
# after the image's r4d_selftest byte check of moe_gather_all_reduce_4rank (every M 1..48, 56, 64 at DeepSeek's shape, both
# transports, against the slot-order oracle; a FAIL stops the item before the start). Then the c4 greedy set twice under
# each value (each arm's run-to-run floor: D3 is the same bytes as today at every row count, so it CANNOT remove the 6-of-20
# c4 floor -- the item shows the floor under both), and identity (the 28k copy and the c1 greedy set under B and A vs bz's
# dsd-d0 files and B vs A). NST 0. Nothing goes into a production line without the user's go.
# IMAGE radiance-local:ds41-d3-c03c57d9-5c13e555-l (XD3_IMAGE).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-i-argall.sh : prints every command, runs and writes nothing.
export XD3_IMAGE=${XD3_IMAGE:-radiance-local:ds41-d3-c03c57d9-5c13e555-l}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-argall.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xd3: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
