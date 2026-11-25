#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-cz-uzzzz-stack5.sh (queue item, STAGED 12.11.2026 by the ds-main-port stack5 builder for the coordinator; sorts
# after 57-p4zyzzzzz-ds-r00zzz-cz-uzzz-usahb and before 57-p4zyzzzzz-ds-r00zzz-cz-v-scr3 / 57-p4zyzzzzz-ds-r00zzz-d-slotsum in C and
# en_US order; gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~116 min estimate (budget XS5_BUDGET_MIN 150: T 1
# trimmed first), NO evict) -- THE DS STACK CONFIRMATION 5, tag "RESULT xs5 ", out runs/xdsm/stack5, body
# queue-ar/dsmain/stack5/dsm-stack5.sh (from COPIES of stack4/dsm-stack4.sh 21ce8c8e and stack4/s4blocks.py 0a2cca92 -> s5blocks.py;
# + s5ctr.py (counter guard), s5chain.py (3-turn chain on pskcr/pcr2.py's turn spec, read only); nc/ncmb.py + nc/ncsteps.py and
# stack4's drivers read only; originals unchanged). NOT QUEUED: the coordinator queues it.
# IMAGE radiance-local:ds41-int8-7e033848-5c13e555-l (ds41-int8 7e03384 = ds41-int7-pskglue 5093ce9 + ds41-pskl20 25aab56 + ds41-pskcr
# e9e529d + ds41-e1 920438e + ds41-noa 0f0294b + ds41-sahb 647830e + ds41-pskcr2 d4540a8 (PSK-CR turn-chain fix); fn4 after ds41_psk_k:
# ds41_psk_l20, dspark_pfctx, ds41_psk_cr, ds41_engdd, sched_noa, stage_short, stage_pick, place_hb), layered on the int7 image; every
# new switch unset = int7 (build/int8 hostcheck).
# ONE NST 5 START = stack4's (runs/ds41-exl3.log 2026-11-11 21:51:34 "RESULT xs4 s4-n5: start") + RADIANCE_LOG_STEPS=1 +
# RADIANCE_DS41_ENGDD=0 (E1 declared) + RADIANCE_PLACE_HB_RANKS=1,2 (HB). Arms in process (POST, nudge, READ BACK, counter baselines
# after the read-back): A = stack4's A pin, every new switch 0; S5 = sched_bi_chunk 3 + sched_noa 2 + ds41_psk 2 k 8 + ds41_psk_l20 1
# + dspark_pfctx 1 + ds41_engdd 1 + draft_sample 85 + draft_occ 6 + ds41_bi 2; S5T = S5 with ds41_psk 4 (tail-only) + ds41_psk_cr 1, ON
# (XS5_T=1): the image carries pskcr2's turn-chain fix d4540a8. S5X (S5 + SA / HB) DROPPED (XS5_X=0): usahb's verdict 12.11. -- c8
# ABBA B 175.3, SA 173.9, HB 173.1, SH 172.4 tok/s; 1u essay B 72.4, HB 68.7 (the switches stay in the image at 0). ABBA orders
# rebalanced for A / S5 / S5T: T 0 A S T T S A A S T T S A (4 blocks an arm), T 1 / prefill / loaded A S T T S A.
# Readings as stack4 (T 0 blocks 1u essay / code / edit, mb c1 / c4 / c8, pf at dur 60 warm 15; T 1 sdprobe; cold 135k / 64k; c4 + 64k)
# + the c8 newcomer share (step log, nc/ncsteps.py) + a 3-turn chain per arm (cached / TTFT / CR counters a turn); identity alone,
# under load, on hits (each hit vs the same prompt cold under A), strict; the COUNTER GUARD on every reading (s5ctr.py).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-cz-uzzzz-stack5.sh : prints every command; checks the image literals of every lever, every
# helper (+ selftests), the sidecars, the yaml, the disk; no server, no GPU, writes nothing.
export XS5_IMAGE=${XS5_IMAGE:-radiance-local:ds41-int8-7e033848-5c13e555-l} XS5_ITEM=${XS5_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-uzzzz-stack5}
export XS5_OUT=${XS5_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/stack5}
export XS5_T=${XS5_T:-1} XS5_X=${XS5_X:-0} XS5_PSKK=${XS5_PSKK:-8} XS5_BIC=${XS5_BIC:-3} XS5_NOA=${XS5_NOA:-2}
export XS5_OCC=${XS5_OCC:-6} XS5_OCCK=${XS5_OCCK:-32} XS5_DSA=${XS5_DSA:-85}
export XS5_SSN=${XS5_SSN:-1025} XS5_PICK=${XS5_PICK:-1} XS5_HBR=${XS5_HBR:-128} XS5_HBRANKS=${XS5_HBRANKS:-1,2}
export XS5_DUR=${XS5_DUR:-60} XS5_WARM=${XS5_WARM:-15} XS5_EDIT=${XS5_EDIT:-1} XS5_GDEP=${XS5_GDEP:-16000}
export XS5_DEPTHS=${XS5_DEPTHS:-135000 64000} XS5_LDEPTH=${XS5_LDEPTH:-64000} XS5_CDEPTH=${XS5_CDEPTH:-48000} XS5_CNTOK=${XS5_CNTOK:-256}
export XS5_DISKMIN_MB=${XS5_DISKMIN_MB:-11000} XS5_BUDGET_MIN=${XS5_BUDGET_MIN:-150}
export XS5_SIDECAR=${XS5_SIDECAR:-dspark-dlw-q8_0.rad} XS5_SIDESHA=${XS5_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
for v in XS5_ORDER0 XS5_ORDER1 XS5_PORDER XS5_LORDER; do [ -n "${!v-}" ] && export $v || unset $v; done   # empty = the body's default for XS5_T / XS5_X
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/stack5/dsm-stack5.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xs5: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
