#!/bin/bash
# 57-p4zyzzzzz-ds-ka-pipeprof.sh (queue item, 10.11.2026, ds41-pipe agent; gpu02, all 4 cards, TP4, MML 524288, every radiance
# container DOWN; ~18 min) -- WHY THE PIPELINED PREFILL LOSES (f-pipe: fresh prefill B / A -10.6 / -13.0 / -12.9 % at 16k / 64k /
# 136k, exact; fz-span4: SP / S -3.6..-4.8 %): one image radiance-local:ds41-pipe-fd215624-5c13e555-l (XPP_IMAGE), two starts
# with --profile-ops (RADIANCE_PROFILE_EVERY 8): A = RADIANCE_DS41_PIPE unset, B = 1. t3 env, the 225 serve line.
# PHASES (queue-ar/dsmain/dsm-pipeprof.sh), per arm: restart, warm-up + fastcheck (PROFILED, not gated), warm-up answers vs
#   stock, the pipelining note (B only), one fresh ~64k prompt answered with one token, then the window: two more such
#   prompts (~34 lone prefill steps) between two /stats snapshots, docker stop; ds_opprof.py per arm, then ds_pipeprof.py
#   B vs A per op role, ms a prefill step a rank: ops beside the lane-1 all-reduce vs control ops, AR kernel time serial vs
#   lane form (floor / skew), the lane-1 gathers, the extra moe_scatter, lane-wait growth, staged / h2d GB a step.
#   PROFILED: not a speed. No mb / decode phases, no cc_sim.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-ka-pipeprof.sh : prints every command, runs and writes nothing.
export XPP_IMAGE=${XPP_IMAGE:-radiance-local:ds41-pipe-fd215624-5c13e555-l}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-pipeprof.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xpipeprof: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
