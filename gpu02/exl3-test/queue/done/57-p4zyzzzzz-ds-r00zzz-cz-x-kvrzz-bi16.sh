#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzz-bi16.sh (queue item, STAGED 12.11.2026 by the ds-main-port bi16 builder for the coordinator;
# sorts after 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrz-sp and before 57-p4zyzzzzz-ds-r00zzz-cz-xz-fnsmoke2 / -d-slotsum in C and en_US
# order; gpu02, all 4 cards, TP4, every radiance container DOWN; ~12 min (estimate; budget XB16_BUDGET_MIN 25), NO evict) --
# LEVER BI AT > 48 DECODE ROWS: lever BI's engine tracer (RADIANCE_DS41_BITRACE) names the FIRST op whose output rows differ
# loaded vs alone when a step carries > 48 decode rows (kv4 n16: 'B48:gs16 9 of 20 differ', 'lid16 8 of 20'), tag
# "RESULT xbi16 ", out runs/xdsm/bi16, body queue-ar/dsmain/bi16/dsm-bi16.sh (from COPIES of kv4/dsm-kv4.sh f89a6cc2 and
# dsm-bilid3.sh af9f3529) + bi16probe.py (from stack4/loadid.py ee7d9663) + bi16trace.py (= bitrace2.py cbbd1852 + the probe
# match, per-step decode rows, op class, independent flips, extract, step counter); originals unchanged. NOT QUEUED: the
# coordinator queues it.
# IMAGE radiance-local:ds41-int8-7e033848-5c13e555-l (d75ba2d615b3; the image kv4 ran; its deepseek41.so carries the tracer:
# RADIANCE_DS41_BITRACE / _PMAX / _LMAX, the ON line, bi_rowhash; the core's 'recorded passes: off') with kv4 n16's start
# (nvfp4, MAXSEQS 16, HOSTPOOL 19456, MML 524288, S5 pins incl. ds41_bi 2 + sched_bi_chunk 3) and every phase under arm D
# (S + DET: fixed depth 5 -> 16 x 6 = up to 96 decode rows). ONE START t16: probes 9 / 8 (the earliest-differing in both kv4
# runs) a1 ALONE -> ld each under 15 mb streams -> a2 ALONE (control).
# READINGS: per probe the first differing op (layer.op, rank, position, the step's T / decode rows / sequences / prefill
# sequences / q_len, alone and loaded), its class (dense GEMM whole-segment / <= 48 pieces, attention, indexer, fused norm,
# MoE), the independent flips of its layer, the loaded run's decode rows a step; the step log's > 48-row steps.
# GUARDS (VIRHE): the tracer dumped; > 48 decode rows really reached (trace DT + step log); a2 = a1 (answers and every traced row).
# DUMP: ~150-300 MB raw expected (cap XB16_BTCAP_MB 1500 checked after every phase), DELETED after the analysis (kept: .names,
# the analysis json, a gzipped extract < 5 MB); disk guard XB16_DISKMIN_MB 11000 before the start and after every phase.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzz-bi16.sh : prints every command; checks the image literals, every helper
# (+ selftests), the sidecars, the yaml, the disk; no server, no GPU, no request to :8090, writes nothing.
export XB16_IMAGE=${XB16_IMAGE:-radiance-local:ds41-int8-7e033848-5c13e555-l} XB16_ITEM=${XB16_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzz-bi16}
export XB16_OUT=${XB16_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/bi16}
export XB16_PINS=${XB16_PINS:-auto} XB16_IDX=${XB16_IDX:-9,8} XB16_CONC=${XB16_CONC:-15} XB16_WARM=${XB16_WARM:-10} XB16_GEN=${XB16_GEN:-512}
export XB16_PMAX=${XB16_PMAX:-224} XB16_LMAX=${XB16_LMAX:-} XB16_BTCAP_MB=${XB16_BTCAP_MB:-1500} XB16_KEEP_MB=${XB16_KEEP_MB:-600} XB16_KEEPRAW=${XB16_KEEPRAW:-0}
export XB16_BIC=${XB16_BIC:-3} XB16_NVG=${XB16_NVG:-48} XB16_HP=${XB16_HP:-19456} XB16_DISKMIN_MB=${XB16_DISKMIN_MB:-11000} XB16_BUDGET_MIN=${XB16_BUDGET_MIN:-25}
export XB16_SIDECAR=${XB16_SIDECAR:-dspark-dlw-q8_0.rad} XB16_SIDESHA=${XB16_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/bi16/dsm-bi16.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xbi16: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
