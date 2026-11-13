#!/bin/bash
# 57-p4zyzzzzz-ds-hszzzz-nvlatp.sh (queue item, 10.11.2026, ds-main-port nvlat agent; gpu02, all 4 cards, TP4, MML 524288, every radiance container
# DOWN; ~12 min, two starts) -- THE PROFILED PAIR OF ITEM hszzz-nvlat, RE-RUN: its nv-aprof / nv-bprof starts stopped at the
# configuration guard ("rule-w is not what the recorded passes play": --profile-ops turns pass recording off,
# core/runtime/ctx.cpp:181, so no pass reports its acquire rule). Body dsm-nvlat2.sh = dsm-nvlat.sh + the guard skipping that
# check in a prof start (logged). Only A:prof (bf16 latent) and B:prof (RADIANCE_DS41_LATENT=nvfp4) on the integrated image,
# the same stack env (gg_gemv 2, aql_acq 2 = rule-w in the env of both arms, K1 c/d/f, turnover 0); windows short / long / c8
# -> per-op device ms a step B vs A (nvlatprof.py ops); the plain and stamped starts are NOT re-run (hszzz-nvlat's). Output
# runs/xdsm/nvlat-prof/ (+ op-*.json beside runs/xdsm/nvlat for the combined summary), log tag "RESULT xnv ". No cc_sim, no 2.9.
# IMAGE radiance-local:ds41-int-c05de277-5c13e555-l (XNV_IMAGE).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-hszzzz-nvlatp.sh : prints every command, checks the image and the yaml, runs and writes nothing.
export XNV_IMAGE=${XNV_IMAGE:-radiance-local:ds41-int-c05de277-5c13e555-l} XNV_ITEM=${XNV_ITEM:-57-p4zyzzzzz-ds-hszzzz-nvlatp} XNV_STARTS=${XNV_STARTS:-A:prof B:prof}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-nvlat2.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xnv: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
