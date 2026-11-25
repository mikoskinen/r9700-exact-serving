#!/bin/bash
# 57-p4zyzzzzz-ds-hszzzzz-nvg.sh (queue item, 10.11.2026, ds-main-port nvlat agent; gpu02, all 4 cards, TP4, MML 524288, every radiance container
# DOWN; ~27 min, three starts) -- LEVER NVG A/B: the NVFP4 latent's decode attention as one unpack launch + the plain bf16
# kernel (RADIANCE_DS41_NVGATHER=16, libr4d of branch ds41-nvlat b12aca4d = ds41-int c05de27 + lever NVG) against today's packed
# kernel, both RADIANCE_DS41_LATENT=nvfp4, NST 0, the stack env (gg_gemv 2, aql_acq 2, K1 c/d/f, turnover 0):
#   A:plain (NVGATHER unset), B:plain (16), B:prof -- clean c1 short / ~31k / mb c8 45 s, the copies' TTFT, identity (copies +
#   20-prompt greedy set at c1 vs bz d0: must read IDENTICAL); B's per-op profile against item hszzz-nvlat's packed-kernel
#   profile (no re-run); then "RESULT xng summary" with the nvlat item's bf16 arm as context.
# WHY: item hszzz-nvlat + latb3: the packed decode attention 66.5 vs 15.7 us a call at 1 row = the latent's +1.7 ms a step.
# EXACT by construction; tests/r4d_emu test_ds41attn / test_ds41fold hold the gathered path to the bf16 cache's bytes.
# Body queue-ar/dsmain/dsm-nvg.sh, helpers nvlatprof.py / nvlatmet.py, yaml ds41-nvg-tp4.yaml (beside it). Output
# runs/xdsm/nvg/, log tag "RESULT xng ". No cc_sim, no 2.9. Nothing goes into a production line from here.
# IMAGE radiance-local:ds41-nvlat-b12aca4d-5c13e555-l (XNG_IMAGE); XNG_STARTS / XNG_ROWS / XNG_C8DUR.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-hszzzzz-nvg.sh : prints every command, checks the image and the yaml, runs and writes nothing.
export XNG_IMAGE=${XNG_IMAGE:-radiance-local:ds41-nvlat-b12aca4d-5c13e555-l} XNG_ITEM=${XNG_ITEM:-57-p4zyzzzzz-ds-hszzzzz-nvg}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-nvg.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xng: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
