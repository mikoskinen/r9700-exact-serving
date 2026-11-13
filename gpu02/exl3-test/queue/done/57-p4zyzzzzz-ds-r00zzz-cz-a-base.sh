#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-cz-a-base.sh (queue item, STAGED 13.11.2026 by the ds-main-port base builder for the coordinator; USER said go;
# sorts first among the r00zzz-cz items, before -cz-a-mclbench and -cz-mcl0probe, in C and en_US order; gpu02, all 4 cards, TP4,
# MML 524288, every radiance container DOWN; ~21 min estimate, budget XS5_BUDGET_MIN 25 (late phases skipped and logged), NO evict)
# -- ONE POST-RISER BASELINE START (cards re-seated: 83:00 on its own root complex, 03 and c3 behind risers), tag "RESULT xb1 ",
# out runs/xdsm/base1113, body queue-ar/dsmain/base/dsm-baseline.sh (a COPY of stack5/dsm-stack5.sh 63649414, unchanged; stack5's
# helpers read in place, read only). NOT QUEUED: the coordinator queues it.
# IMAGE radiance-local:ds41-int9-0cb6fd31-5c13e555-l (int9 = int8 7e03384 + KVR + SP); KVR OFF (RADIANCE_DS41_KV_REPLICATED unset,
# no declare line checked), SP OFF (stage_full_min / stage_full_pick 0 at start, read).
# stack5's start (NST 5, stack4's start words + RADIANCE_LOG_STEPS=1 + E1 declared + HB ranks 1,2) and its S5T arm EXACTLY as stack5
# ran it: fixed pin + ds41_bi 2, draft_sample 85 (SD), draft_occ 6, place_lf 0, sched_bi_chunk 3, sched_noa 2, ds41_psk 4 (tail-only)
# k 8, ds41_psk_cr 1, ds41_psk_l20 1, dspark_pfctx 1, ds41_engdd 1. Arms A / S5 are applied and read back in the switch note and
# get the cold ~16k guard (body checks); only S5T is measured.
# PHASES: start + modes + VRAM + RANK -> CARD MAP (start log) + switch note + warm-up vs stock + S5T smoke -> cold guard A / S / T at
# ~16k -> T 0: TWO S5T blocks (XS5_ORDER0=TT; each 1u essay / code + edit + mb c1 / c4 / c8 (old loop, nc/ncmb.py taps) + pf, dur 60
# warm 15) -> c8 newcomer share per block (nc/ncsteps.py: newcomer own step wall p50, MIXED / prefill, wait) -> T 1 none -> cold
# prefill S5T at 135000 / 64000 / 16000 (one each) -> no loaded runs -> no turn chains -> identity: greedy set c1 alone under S5T vs
# stack5's s5-n5-a1-T.jsonl -> summary (post vs pre-move stack5 S5T per metric). c8mix: offline later (harness/c8mix.py on
# runs/xdsm/base1113/t0-engine.log.gz). Per-rank expert time: NOT here (needs a profiled start).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-cz-a-base.sh : prints every command; checks the image literals, every helper (+ selftests),
# the sidecars, the yaml, the disk; no server, no GPU, writes nothing.
export XS5_IMAGE=${XS5_IMAGE:-radiance-local:ds41-int9-0cb6fd31-5c13e555-l} XS5_ITEM=${XS5_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-a-base}
export XS5_OUT=${XS5_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/base1113}
export XS5_T=1 XS5_X=0 XS5_PSKK=8 XS5_BIC=3 XS5_NOA=2 XS5_OCC=6 XS5_OCCK=32 XS5_DSA=85
export XS5_ORDER0=${XS5_ORDER0:-TT} XS5_ORDER1="" XS5_PORDER=${XS5_PORDER:-T} XS5_LORDER=""
export XS5_DUR=${XS5_DUR:-60} XS5_WARM=${XS5_WARM:-15} XS5_EDIT=1 XS5_GDEP=${XS5_GDEP:-16000}
export XS5_DEPTHS=${XS5_DEPTHS:-135000 64000 16000} XS5_LDEPTH=64000 XS5_CDEPTH=48000 XS5_CNTOK=256
export XS5_DISKMIN_MB=${XS5_DISKMIN_MB:-11000} XS5_BUDGET_MIN=${XS5_BUDGET_MIN:-25}
export XS5_SIDECAR=dspark-dlw-q8_0.rad XS5_SIDESHA=f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/base/dsm-baseline.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xb1: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
