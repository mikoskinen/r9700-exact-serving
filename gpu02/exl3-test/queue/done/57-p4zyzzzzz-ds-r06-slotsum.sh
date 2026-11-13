#!/bin/bash
# 57-p4zyzzzzz-ds-p-slotsum.sh (queue item, 10.11.2026, ds41-slot agent, lever D3s stage 1; gpu02, all 4 cards, TP4, MML 524288,
# every radiance container DOWN; ~80 min) -- the rank-independent SLOT-ORDER MoE sum (the user's decision 10.11.: changes the
# output once at rounding level, approved with a short quality check; unset = today's bytes). r4d_selftest byte checks first
# (moe_gather_all_reduce_slot_4rank: the oracle AND the same bytes under contiguous / scrambled / per-slot placement on four
# cards; moe_scatter_placed_i32; regressions) -- a FAIL stops the item. START 1 (RADIANCE_DS41_SLOTSUM=48, contiguous, the start
# on today's path): its cost in process (ds41_slotsum 0 / 1 blocks A B B A A B: stepprobe c1, mb c4, mb c8; then fresh prefill
# 16k A B B A and 136k A B, mb pf A B), the quality check (slotkl.py: teacher-forced KL of the next-token distributions, A / B /
# A again, top-20, 40 documents x 128 tokens; + greedy set and 28k copy, information) and --expert-stats for stage 2. START 2
# (+ RADIANCE_DS41_EP_PLACE scrambled 6 a rank a layer, _EP_PLACE2 the hottest 8 a layer replicated, host pool + the extras):
# the acceptance -- c1 greedy set, 28k copy and 136k deep answer BYTE-IDENTICAL under ds41_place 0 / 1 / 2. NST 0. No cc_sim,
# no BetterBench. Nothing goes into a production line without the user's go.
# IMAGE radiance-local:ds41-slot-4cf3501f-5c13e555-l (XSL_IMAGE).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-p-slotsum.sh : prints every command, runs and writes nothing.
export XSL_IMAGE=${XSL_IMAGE:-radiance-local:ds41-slot-4cf3501f-5c13e555-l}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-slotsum.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xsl: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
