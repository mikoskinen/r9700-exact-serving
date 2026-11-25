#!/bin/bash
# 57-p4zyzzzzz-ds-hszzz-nvlat.sh (queue item, 10.11.2026, ds-main-port nvlat agent; gpu02, all 4 cards, TP4, MML 524288, every
# radiance container DOWN; ~50 min, six starts) -- WHERE THE NVFP4 LATENT'S 1-USER DECODE COST GOES (LEVERS "NVFP4 latent" /
# "NVFP4 latent decode plateau": exact, the 1M-context capacity lever, 288 vs 1024 bytes an entry; cost 1 user +10-12 %
# (15.81 -> 17.71 ms), c8 -4..-6 %, prefill -15.1 %). On the integrated image, NST 0, the stack env (gg_gemv 2, aql_acq 2,
# K1 c/d/f, turnover 0): arm A = RADIANCE_DS41_LATENT unset (bf16), arm B = nvfp4 (read at declare: separate starts),
# ABBAAB over plain / stamped / profiled starts:
#   plain: clean c1 short (stepprobe essay,code 512, twice), c1 at ~31k (the 28k copy's decode, cold + prefix hit), mb c8 45 s,
#          the copy's cold TTFT; identity: the copies and the 20-prompt greedy set at c1 vs bz d0 (bf16) -- expected IDENTICAL;
#   stamp: RADIANCE_AQL_STAMP=1 windows short / long -> the critical path and the kernels' device / on-chain us a pass;
#   prof:  --profile-ops windows short / long / c8 -> per op device ms a step.
#   then "RESULT xnv summary": clean ms a step A / B, stamped critical path vs profiled device sum, kernels only in B.
# Body queue-ar/dsmain/dsm-nvlat.sh, helpers nvlatprof.py / nvlatmet.py, yaml ds41-nvlat-tp4.yaml (beside it). Output
# runs/xdsm/nvlat/, log tag "RESULT xnv ". Sorts right after -hszz-confhead (en_US). No cc_sim, no 2.9. Nothing goes into a
# production line from here. IMAGE radiance-local:ds41-int-c05de277-5c13e555-l (XNV_IMAGE); XNV_STARTS / XNV_C8DUR.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-hszzz-nvlat.sh : prints every command, checks the image and the yaml, runs and writes nothing.
export XNV_IMAGE=${XNV_IMAGE:-radiance-local:ds41-int-c05de277-5c13e555-l} XNV_ITEM=${XNV_ITEM:-57-p4zyzzzzz-ds-hszzz-nvlat}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-nvlat.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xnv: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
