#!/bin/bash
# 57-p4zyzzzzz-ds-q-epbal.sh (queue item, 10.11.2026, ds41-slot agent, levers D3c / F1 on lever D3s; gpu02, all 4 cards, TP4,
# MML 524288, every radiance container DOWN; ~2 h; AFTER item p: its start 1's --expert-stats are this item's routing input,
# else 6.11. cc-x117) -- expert-parallel balancing on the slot-order MoE sum (placement-invariant bytes, item p's acceptance),
# measured IN ONE PROCESS. Round 1 (start a): tables 0 contiguous, 1 F1 (the 8 hottest experts a layer on every rank, the
# least-loaded copy a step), 2 D3c (epplace.py solve: expected rank loads balanced by moving experts within 12 extras a rank a
# layer); blocks place 0 vs 1 and 0 vs 2 (A B B A A B: stepprobe c1, mb c4, mb c8) with each block's measured skew from
# RADIANCE_DEBUG_MISS_LOG. Round 2 (start b): round 1's best vs a re-solve at 16 rows with a replicated head of 4; then THE LINE:
# today (ds41_slotsum 0) vs the best (slot-order sum + the best table), and prefill today vs best (fresh 16k A B B A, 136k A B,
# mb pf A B). VRAM-honest: every copy off home is an extra expert unit (host pool grown by the extras' slots). NST 0. No cc_sim,
# no BetterBench. Nothing goes into a production line without the user's go.
# IMAGE radiance-local:ds41-slot-4cf3501f-5c13e555-l (XEB_IMAGE).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-q-epbal.sh : prints every command, runs and writes nothing.
export XEB_IMAGE=${XEB_IMAGE:-radiance-local:ds41-slot-4cf3501f-5c13e555-l}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-epbal.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xeb: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
