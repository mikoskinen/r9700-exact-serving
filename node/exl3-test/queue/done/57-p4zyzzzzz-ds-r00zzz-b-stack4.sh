#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-b-stack4.sh (queue item, 11.11.2026, ds-main-port stack3 builder for the coordinator; coordinator order r00zz-psk2 ->
# r00zzz-a-lf -> r00zzz-b-stack4 -> r00zzz-c-dst -> the rest; sorts after r00zzz-a-lf, before r00zzz-c-dst (C and en_US)) -- THE DS STACK
# CONFIRMATION 4, tag "RESULT xs4 ", out runs/xdsm/stack4, body queue-ar/dsmain/stack4/dsm-stack4.sh (+ s4blocks.py = occ2blocks.py +
# --t1-prompts, mbt.py, pskld.py (r00z's), copies of loadid.py / sdprobe.py / editprobe.py). gpu02, all 4 cards, TP4, MML 524288, every
# radiance container DOWN; ~110 min (XS4_LF=sl ~141), NO evict. Image radiance-local:ds41-int7-0391a6ec-5c13e555-l (b54f9224a056; ds41-int7 0391a6e =  ds41-int5 +
# ds41-lf d583904 + ds41-dst b890cee + ds41-psk2 d6fcc24 + the glue 0391a6e (PSK design 2 on every admission path, /stats "psk"); DST switches and sched_bi_chunk left unset here).
# STAGED -- the coordinator queues it. SET AT QUEUE TIME from the results before it:
#   XS4_PSK / XS4_PSKK: 2 with k 4 (the coordinator's K, 11.11.), or 0 if design 2 cannot be made exact (not 1, unless a fixed
#     design 1 passed);
#   XS4_LF: 1 if r00zzz-a-lf wins, 0 if it loses; sl (below) = its verdict not in: S keeps place_lf 0 and the in-process SL arm runs.
# One NST 5 start (stack3's S start + z0occ2's Z0 start words + RADIANCE_DS41_PSK=0), in process: A = stack3's arm A (bi 0, sample 0,
# occ 0, lf 0, psk 0); S = ds41_bi 2, draft_occ 6 / k 32, ds41_psk XS4_PSK (k XS4_PSKK), place_lf by XS4_LF, draft_sample 85 (S85;
# S70 = 70); SL = S85 + place_lf 1 under XS4_LF=sl. T 0 blocks (1u essay / code / edit, mb c1 / c4 / c8 / pf at T 0), T 1 blocks
# (1u essay / code / edit seed 1234 thinking on, mb c4 / c8), prefill A / S ASSA (cold 136k / 64k TTFT + c4 + cold 64k), load
# identity at T 0 under S. Summary vs A, S85 vs S70 (SL vs S85), targets, stack3 / z0occ2 cross-start.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-b-stack4.sh : prints every command; checks the image literals, every helper, the sidecars, the
# yaml; no server, no GPU, writes nothing.
export XS4_IMAGE=${XS4_IMAGE:-radiance-local:ds41-int7-0391a6ec-5c13e555-l} XS4_ITEM=${XS4_ITEM:-57-p4zyzzzzz-ds-r00zzz-b-stack4}
export XS4_OUT=${XS4_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/stack4}
export XS4_PSK=${XS4_PSK:-2} XS4_PSKK=${XS4_PSKK:-4} XS4_LF=${XS4_LF:-sl}
export XS4_OCC=${XS4_OCC:-6} XS4_OCCK=${XS4_OCCK:-32} XS4_DSA=${XS4_DSA:-85} XS4_DSB=${XS4_DSB:-70}
export XS4_ORDER0=${XS4_ORDER0-} XS4_ORDER1=${XS4_ORDER1-} XS4_DUR=${XS4_DUR:-60} XS4_WARM=${XS4_WARM:-15} XS4_EDIT=${XS4_EDIT:-1}
export XS4_DEPTHS=${XS4_DEPTHS:-135000 64000} XS4_PORDER=${XS4_PORDER:-ASSA} XS4_LORDER=${XS4_LORDER-ASSA} XS4_LDEPTH=${XS4_LDEPTH:-64000}
export XS4_SIDECAR=${XS4_SIDECAR:-dspark-dlw-q8_0.rad} XS4_SIDESHA=${XS4_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
[ -n "$XS4_ORDER0" ] || unset XS4_ORDER0; [ -n "$XS4_ORDER1" ] || unset XS4_ORDER1   # empty = the body's default for XS4_LF
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/stack4/dsm-stack4.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xs4: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
