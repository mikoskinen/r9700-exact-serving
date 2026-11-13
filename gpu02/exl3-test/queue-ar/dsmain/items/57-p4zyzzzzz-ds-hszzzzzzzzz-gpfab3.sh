#!/bin/bash
# 57-p4zyzzzzz-ds-hszzzzzzzzz-gpfab3.sh (queue item, 10.11.2026, lever GPF agent, branch ds41-gpf edc9a9b; gpu02, all 4 cards, TP4,
# MML 524288, every radiance container DOWN; ~55 min) -- the A/B of levers GPF + GPF-b on the ds41-gpf edc9a9be image (body
# dsm-gpf3.sh; log tag xgq, out runs/xdsm/gpfab3). Replaces gpfab / gpfab2, which stopped at their start check on a false VIRHE
# (libr4d prints its GPF table line only at the first gg_pf rule call, not at start; dsm-gpf3.sh checks the env listing at start
# and the table line after the first B prefills).
#   A gg_pf 0 + f32_pf 0 (today); B gg_pf 1 with gpbench's M 4096 table (q6_K q_b / o_b / shexp down (both widths) / engram_kv
#   426, q8_0 shexp down (both) 422) + f32_pf 25 (gpbench-f32: x1.16-1.74 over the stock gemm_nt_f32, byte checks PASS);
#   C f32_pf 25 alone (one A C C A round of the same c8 + fresh prefills, so gg's and f32's shares separate).
#   NST 0 stack; A B B A x 2 mb c8 + fresh ~16k / ~64k / ~128k; A C C A x 1; c4 ABBA; mb pf ABBA; 136k cold A / B; identity A / B.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-hszzzzzzzzz-gpfab3.sh : prints every command; checks the image and the yaml; writes nothing.
export XGQ_ITEM=${XGQ_ITEM:-57-p4zyzzzzz-ds-hszzzzzzzzz-gpfab3} XGQ_IMAGE=${XGQ_IMAGE:-radiance-local:ds41-gpf-edc9a9be-5c13e555-l} XGQ_ROUNDS=${XGQ_ROUNDS:-2} XGQ_OUT=${XGQ_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/gpfab3}
export XGQ_TABLE=${XGQ_TABLE:-q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422} XGQ_F32=${XGQ_F32:-25} XGQ_ARMC=${XGQ_ARMC:-1}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-gpf3.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xgq: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
