#!/bin/bash
# 57-p4zyzzzzz-ds-ha-k1prof.sh (queue item, 10.11.2026, ds41-k1 agent, lever K1 diagnosis; gpu02, all 4 cards, TP4, MML
# 524288, every radiance container DOWN; ~20 min, three starts) -- why lever K1's folds measured ~0 in gzzz-attnfold (all three
# 0 vs 1 in process: c1 +0.21 % +-2.27, c4 -0.22 % +-3.39, c8 +0.83 % +-1.13) while the kernel bench saved ~3-4 us a layer.
# The ds41-k1 image on the t3 line (ARGATHER 4, F4 off) with the folds declared, A (all K1 switches 0) vs B (1) in process:
#   P1 plain start: c1 stepprobe blocks A B B A A B -> ms a step (the truth: unprofiled, unstamped)
#   P2 RADIANCE_AQL_STAMP=1 start: c1 windows A1 B1 B2 A2 -> k1prof.py stamps: card 0's trunk pass wall, chain, and for the
#      folds' kernels device / exposed / idle-before / ordered / on-chain; the stamped step vs P1's
#   P3 --profile-ops start: c1 windows A, B -> k1prof.py ops: the folds' ops' us a call and lane waits (PROFILED)
#   summary: where the bench saving went -- gaps not on the path, hidden by unordered overlap, slower fused kernels, or
#   inconclusive -- each with its readings. No output comparisons (K1 is exact; gzzz-attnfold's identity holds for cold runs).
# IMAGE radiance-local:ds41-k1-1d1f782b-5c13e555-l (XK1P_IMAGE).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-ha-k1prof.sh : prints every command, runs and writes nothing.
export XK1P_IMAGE=${XK1P_IMAGE:-radiance-local:ds41-k1-1d1f782b-5c13e555-l}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-k1prof.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xk1p: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
