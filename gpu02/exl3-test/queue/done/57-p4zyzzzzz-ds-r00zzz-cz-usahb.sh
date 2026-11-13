#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-cz-usahb.sh (queue item, STAGED 12.11.2026 by the ds-main-port sahb builder for the coordinator; sorts after
# 57-p4zyzzzzz-ds-r00zzz-cz-tdst2 and before 57-p4zyzzzzz-ds-r00zzz-d-slotsum in C and en_US order; gpu02, all 4 cards, TP4, MML
# 524288, every radiance container DOWN; ~45 min (estimate; budget XSB_BUDGET_MIN 55), NO evict) -- LEVERS SA + HB for the c8 target
# (c8 >= 200 tok/s; today ~172-175), tag "RESULT xsahb ", out runs/xdsm/sahb, body queue-ar/dsmain/sahb/dsm-sahb.sh (from COPIES of
# nc/dsm-ncprof-p.sh 9c20358f, nc/ncsteps.py 0a6e0ed2 (+ ncone.py / ncmb.py / ds_opprof_nc.py copied beside it, which it loads from
# its own directory) and stack4/dsm-stack4.sh 21ce8c8e; originals unchanged) + sahbsum.py (guards, summary). NOT QUEUED: the
# coordinator queues it.
# IMAGE radiance-local:ds41-sahb-647830e8-5c13e555-l (ds41-sahb 647830e = ds41-int7-pskglue 5093ce9 + ds41-noa 3f7fc59 (cherry-picked
# 95a901e: a recycled slot clears the bypass's stale bypass_no) + levers SA / HB), layered on the int7 image.
# WHY (r00zzz-cz-ncprof / -pz-ncprofp): each ~1.1k newcomer at c8 is one MIXED step of 0.58-0.61 s; the prefill stager never armed
# on it (RADIANCE_STAGE_SPAN=0 leaves the span's shared 2112-row level without a ring; 1025..2112-row steps run there), the
# host-pool experts (~3.8 GB a rank alone) stream in-kernel; r1 / r2 (83:00, 86:00 under the one 80: root complex) take ~30 % longer
# in the expert GEMMs for EQUAL streamed bytes, and r0 / r3 wait ~80 ms in collectives.
# LEVERS (fn4, all 0 = the int7 image's behaviour): SA = stage_short N (arm the stager for a step of >= N tokens at the span's shared
# level, its ring at that level's own place, the step's arena loan kept below it) + stage_pick K (0 every pooled unit / K the units
# the last K stager-size steps routed; K measured by an alone screen first); HB = place_hb R on RADIANCE_PLACE_HB_RANKS=1,2 (the R
# units a class with the highest long-memory prefill heat kept resident, below the decode-pinned core). Exact (address / placement).
# ONE NST 5 START U (stack4's start, runs/ds41-exl3.log 2026-11-11 21:51:34 "RESULT xs4 s4-n5: start") + RADIANCE_LOG_STEPS=1 +
# RADIANCE_PLACE_HB_RANKS=1,2; arms on stack4's S pins: B / SA / HB / SH (= SA + HB), ABBA B SA HB SH SH HB SA B; per block the 1u
# stepprobe essay / code and mb c8 exactly as stack4 (T 0, dur 60 warm 15): agg_dec_tps, decode_share, newcomer step wall (step
# log), staged / streamed GB a non-decode step, SA / HB / PSK counters. Identity at T 0: greedy set alone B vs SH (+ vs bz d0), SH
# under c8 load with ~1.1k newcomers vs alone, c4 + cold 64k probe B vs SH (+ vs stack4's loaded S, cross-start). START P
# (--profile-ops every 16 + miss log): per arm 16 + 16 alone newcomers -> per-rank experts / collective / dense ms.
# COUNTER GUARD: pins read back; each arm's counters move only where they apply (short passes only under SA, the reserve only under
# HB on ranks 1, 2); PSK on the corrected /stats psk counters (latched / planned_steps / cut_passes grow, plan_mismatch 0; never
# layers_skipped); baselines taken after the switch read-back. Otherwise VIRHE, no speed reading for that block.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-cz-usahb.sh : prints every command; checks the image literals, every helper (+ selftests), the
# sidecars, the yaml, the disk; no server, no GPU, no request to :8090, writes nothing.
export XSB_IMAGE=${XSB_IMAGE:-radiance-local:ds41-sahb-647830e8-5c13e555-l} XSB_ITEM=${XSB_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-usahb}
export XSB_OUT=${XSB_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/sahb}
export XSB_SSN=${XSB_SSN:-1025} XSB_HBR=${XSB_HBR:-128} XSB_HBRANKS=${XSB_HBRANKS:-1,2} XSB_PICKS=${XSB_PICKS:-0 1 2} XSB_PICK_DEFAULT=${XSB_PICK_DEFAULT:-1}
export XSB_SCREEN_N=${XSB_SCREEN_N:-8} XSB_ORDER=${XSB_ORDER:-B SA HB SH SH HB SA B} XSB_DUR=${XSB_DUR:-60} XSB_WARM=${XSB_WARM:-15}
export XSB_IDENT=${XSB_IDENT:-1} XSB_LDEPTH=${XSB_LDEPTH:-64000} XSB_P=${XSB_P:-1} XSB_PARMS=${XSB_PARMS:-B SA HB SH}
export XSB_PWARM=${XSB_PWARM:-16} XSB_PALONE=${XSB_PALONE:-16} XSB_EVERY=${XSB_EVERY:-16} XSB_DISKMIN_MB=${XSB_DISKMIN_MB:-11000}
export XSB_BUDGET_MIN=${XSB_BUDGET_MIN:-55}
export XSB_SIDECAR=${XSB_SIDECAR:-dspark-dlw-q8_0.rad} XSB_SIDESHA=${XSB_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/sahb/dsm-sahb.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xsahb: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
