#!/bin/bash
# 57-p4zyzzzzz-ds-hszzzzzzzzz-gpfab2.sh (queue item, 10.11.2026, lever GPF agent, branch ds41-gpf edc9a9b; gpu02, all 4 cards, TP4,
# MML 524288, every radiance container DOWN; ~45 min) -- THE FOLLOW-UP A/B of levers GPF + GPF-b TOGETHER on the ds41-gpf edc9a9be
# image (body dsm-gpf2.sh = gpfab's dsm-gpf.sh with its own out dir; log tag xgq, out runs/xdsm/gpfab2). gpfab (running before
# this) switched gg_pf alone (its f32 form was not picked: gpsum.py skipped GPF-b's lines in a gg-less bench; fixed). Here:
#   A gg_pf 0 + f32_pf 0 (today) vs B gg_pf 1 with the M 4096 table of gpbench (GPTAB: the best byte-identical form a (type, shape)
#   >= 1 % over the stock at M 4096 -- q6_K q_b / o_b / shexp down (both widths) / engram_kv 426, q8_0 shexp down (both) 422) +
#   f32_pf 25 (gpbench-f32's GPF32FORM ab: x1.16-1.74 over the stock gemm_nt_f32 at every shape / storage / M, both byte checks
#   PASS). The rest as gpfab: NST 0 stack; ABBA x 2 mb c8 + fresh 16k / 64k / 128k; c4 ABBA; mb pf ABBA; 136k cold A / B; identity A / B.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-hszzzzzzzzz-gpfab2.sh : prints every command; checks the image and the yaml; writes nothing.
export XGQ_ITEM=${XGQ_ITEM:-57-p4zyzzzzz-ds-hszzzzzzzzz-gpfab2} XGQ_IMAGE=${XGQ_IMAGE:-radiance-local:ds41-gpf-edc9a9be-5c13e555-l} XGQ_ROUNDS=${XGQ_ROUNDS:-2} XGQ_OUT=${XGQ_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/gpfab2}
export XGQ_TABLE=${XGQ_TABLE:-q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422} XGQ_F32=${XGQ_F32:-25}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-gpf2.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xgq: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
