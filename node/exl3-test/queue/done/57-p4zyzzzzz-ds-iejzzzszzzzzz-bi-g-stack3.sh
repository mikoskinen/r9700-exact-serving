#!/bin/bash
# 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-g-stack3.sh (queue item, 11.11.2026, ds-main-port stack3 builder; sorts after
# 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-f-pc and 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-g-longid (queued 13:16 in the slot after bi-f-pc),
# before 57-p4zyzzzzz-ds-iejzzzszzzzzz-fn-bitrace (C and en_US order); gpu02, all 4 cards,
# TP4, MML 524288, every radiance container DOWN; ~130 min) -- THE DS STACK CONFIRMATION 3, tag "RESULT xs3 ", out
# runs/xdsm/stack3, body queue-ar/dsmain/stack3/dsm-stack3.sh (+ s3blocks.py, mbt.py and copies of loadid.py / sdprobe.py /
# editprobe.py beside it). Image radiance-local:ds41-int4-cea53a40-5c13e555-l (482004322f11; ds41-int4 cea53a4 = ds41-int3 1116248
# + ds41-bi 2d9a1dc + ds41-dlw df86048 + ds41-sd 397608f; hostcheck: every switch unset = int3).
# In-process A/B in STACK CONFIRMATION 2's style: base arm A = stack2's switch set; stack arms P = + ds41_bi 2, C = + ds41_bi 4
# (+ draft_sample 1 at NST 5); T 1 arm S = ds41_bi XS3_T1BI + draft_sample 1. Starts n0 (NST 0) then n5 (NST 5 + --draft-coupling on
# + the DLW q8_0 sidecar for every arm; DLW's cross-start control = iejzz-dlw done + iel-dlwb queued, no baseline re-run here).
# Per start: T 0 blocks A P C C P A (stepprobe essay / code, editprobe, mb c1 (n5) / c4 / c8 / pf at temperature 0, per-phase VRAM /
# lend lines), n5 T 1 blocks A S S A (sdprobe seed 1234 thinking on, mb c4 / c8), 136k cold prefill, identity at T 0 (alone vs bz d0,
# loaded vs alone, c4 run-to-run, control; n0 values XS3_IDVALS0, n5 XS3_IDVALS5) + the 28k copy; summary vs A, vs hxzzy-stack2
# (cross-start) and the targets (1u >= 90, c4 >= 160, c8 >= 200, 136k <= 35 s).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-g-stack3.sh : prints every command; checks the image literals, every helper, the
# sidecars, the yaml; no server, no GPU, writes nothing.
export XS3_IMAGE=${XS3_IMAGE:-radiance-local:ds41-int4-cea53a40-5c13e555-l} XS3_ITEM=${XS3_ITEM:-57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-g-stack3}
export XS3_OUT=${XS3_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/stack3} XS3_STARTS=${XS3_STARTS:-n0 n5}
export XS3_ORDER=${XS3_ORDER:-APCCPA} XS3_T1ORDER=${XS3_T1ORDER-ASSA} XS3_T1BI=${XS3_T1BI:-2}
export XS3_PH0=${XS3_PH0:-c1,edit,c4,c8,pf} XS3_PH5=${XS3_PH5:-c1,edit,mbc1,c4,c8,pf} XS3_DUR=${XS3_DUR:-60} XS3_WARM=${XS3_WARM:-15}
export XS3_IDVALS0=${XS3_IDVALS0-4} XS3_IDVALS5=${XS3_IDVALS5-2 4}
export XS3_SIDECAR=${XS3_SIDECAR:-dspark-dlw-q8_0.rad} XS3_SIDESHA=${XS3_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/stack3/dsm-stack3.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xs3: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
