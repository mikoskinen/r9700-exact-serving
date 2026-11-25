#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-cz-reng.sh (queue item, STAGED 12.11.2026, ds-main-port E1 builder for the coordinator; sorts after
# 57-p4zyzzzzz-ds-r00zzz-cz-qkbench and before 57-p4zyzzzzz-ds-r00zzz-d-slotsum in C and en_US order; gpu02, all 4 cards, TP4,
# MML 524288, every radiance container DOWN; ~30 min (estimate, <= 45), NO evict) -- LEVER E1, the Engram kv GEMM dedup in
# prefill (audit lever 5): tag "RESULT xe1 ", out runs/xdsm/e1, body queue-ar/dsmain/e1/dsm-e1.sh (+ e1q.py, code-body.txt and
# copies of stack4's pskld.py / mbt.py / loadid.py). NOT QUEUED: the coordinator queues it.
# Image XE1_IMAGE (ds41-e1 = ds41-int7-pskglue 5093ce9 + lever E1: fn4 ds41_engdd, RADIANCE_DS41_ENGDD, /stats "engdd"; libr4d's
# gather_rows 16-byte form above 48 rows + r4d_selftest's gather_rows_bf16 case).
# P0: the image's r4d_selftest --case gather_rows_bf16 on one card. ONE NST 5 START = stack4's (s4-n5) + RADIANCE_DS41_ENGDD=0;
# in process A = stack4's S {"ds41_bi": 2, "draft_sample": 85, "draft_occ": 6, "place_lf": 0, "ds41_psk": 2, "ds41_psk_k": 4} +
# ds41_engdd 0, E = + ds41_engdd 1 (each read back). COUNTER GUARD (E moves /stats engdd, A does not; else VIRHE and no speed
# reading) -> identity alone (greedy set c1) -> mb c8 A, E -> cold TTFT AEEA at 136k / 64k / 16k on deepq's prompt AND a
# realistic ~135k-token code prompt (our own repo sources) -> identity under load (c4 + a cold 64k probe, sched_bi_chunk 3 in
# both arms) -> identity on prefix hits (own texts per arm) -> summary.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-cz-reng.sh : prints every command; checks the image literals, every helper, the sidecars,
# the yaml; no server, no GPU, writes nothing.
export XE1_IMAGE=${XE1_IMAGE:-radiance-local:ds41-e1-920438e7-5c13e555-l} XE1_ITEM=${XE1_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-reng}
export XE1_OUT=${XE1_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/e1}
export XE1_DEPTHS=${XE1_DEPTHS:-135000 64000 16000} XE1_SRCS=${XE1_SRCS:-deep code} XE1_PORDER=${XE1_PORDER:-AEEA}
export XE1_NTOKD=${XE1_NTOKD:-512} XE1_NTOKC=${XE1_NTOKC:-256} XE1_GDEP=${XE1_GDEP:-16000} XE1_LDEPTH=${XE1_LDEPTH:-64000} XE1_LBIC=${XE1_LBIC:-3}
export XE1_HIT1=${XE1_HIT1:-64000} XE1_HIT2=${XE1_HIT2:-40000} XE1_HNTOK=${XE1_HNTOK:-256} XE1_DOFFE=${XE1_DOFFE:-100} XE1_DOFFA=${XE1_DOFFA:-300}
export XE1_DUR=${XE1_DUR:-60} XE1_WARM=${XE1_WARM:-15}
export XE1_SIDECAR=${XE1_SIDECAR:-dspark-dlw-q8_0.rad} XE1_SIDESHA=${XE1_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/e1/dsm-e1.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xe1: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
