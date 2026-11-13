#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzzzzz-1ub.sh (queue item, STAGED 12.11.2026 by the int11 builder's XP fork for ds-main-port; sorts
# after -cz-x-kvrzzzzzz-1u and before -cz-xz-* in C and en_US order; gpu02, all 4 cards, TP4, MML 524288, every radiance container
# DOWN; ~22 min estimate, budget X1UB_BUDGET_MIN 24 (a late section skipped and logged), NO evict) -- LEVER 1U-XP (the core's
# next-layer expert prefetch for DeepSeek-V4.1's EXL3 experts, deepseek41 fn4 ds41_xp), tag "RESULT x1ub ", out runs/xdsm/1ub,
# body queue-ar/dsmain/1ub/dsm-1ub.sh (from a COPY of arc/dsm-arc.sh; + 1ub/rowms.py). NOT QUEUED: the coordinator queues it.
# IMAGE radiance-local:ds41-int11b-84f682aa-5c13e555-l (cd436f2952ec; = the int11 image d3569104 + a layer of bin/radiance and deepseek41.so
# from ds41-int11b 84f682a: ec852d2 lever 1U-XP (+ 84f682a the route export a weak reference), fn4 ds41_xp 62 after CBH, kSwitches 63; GET /admin/fn4
# "xp"; RADIANCE_DS41_XP=1 links the layers; 0 / unset = the int11 image).
# ONE START = the 1u item's (the ARC screen's + RADIANCE_DS41_GQ=3 / _E2=1) + RADIANCE_XPREFETCH=1 (the ring, both arms) +
# RADIANCE_DS41_XP=1. BASE PIN = the 1u item's base (S5T, draft_sample 0, M1 on, draft_depth0 3, sched_row_cap 48, ds41_arcoded 0,
# the int11 levers 0). Arms in process (fn4, read back): B ds41_xp 0, C ds41_xp 1. Readings: (a) 1u reads stepprobe essay,code 512 T 0 B C B C (two a arm, reference first;
# tok/s, ms / tokens a step, the xp counters, stream MB, ms by verify rows from the step log), (b) mb c4 on the fixed schedule
# (1x4, seeded) B C B C + c8mix, (c) greedy-set identity C vs B (a difference = VIRHE). Guards: switch read back, 0 request errors,
# the XP counters (C plans copies, B none).
# X1UB_SW=0 = a mock run (queue-ar/tools/mockrad): no int9 .. int11b switch posted, the counters INFO, a "VIRHE X1UB_SW=0" marker.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzzzzz-1ub.sh : prints every command; checks the image literals (int9 .. int11b),
# every helper (+ selftests), the sidecars, the yaml, the disk; no server, no GPU, writes nothing.
export X1UB_IMAGE=${X1UB_IMAGE:-radiance-local:ds41-int11b-84f682aa-5c13e555-l} X1UB_ITEM=${X1UB_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzzzzz-1ub}
export X1UB_OUT=${X1UB_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/1ub} X1UB_SW=${X1UB_SW:-1}
export X1UB_EORDER=${X1UB_EORDER-BCBC} X1UB_EPROMPTS=${X1UB_EPROMPTS:-essay,code} X1UB_EGEN=${X1UB_EGEN:-512} X1UB_C4ORDER=${X1UB_C4ORDER-BCBC} X1UB_C4GENS=${X1UB_C4GENS:-2}
export X1UB_D=${X1UB_D:-1} X1UB_DDUR=${X1UB_DDUR:-30} X1UB_DWARM=${X1UB_DWARM:-10} X1UB_IDN=${X1UB_IDN:-1}
export X1UB_MHC=${X1UB_MHC:-1} X1UB_D0=${X1UB_D0:-3} X1UB_CAP=${X1UB_CAP:-48} X1UB_BASEX=${X1UB_BASEX:-} X1UB_SPN=${X1UB_SPN:-0} X1UB_SPK=${X1UB_SPK:-0}
export X1UB_DISKMIN_MB=${X1UB_DISKMIN_MB:-11000} X1UB_BUDGET_MIN=${X1UB_BUDGET_MIN:-24}
export X1UB_SIDECAR=${X1UB_SIDECAR:-dspark-dlw-q8_0.rad} X1UB_SIDESHA=${X1UB_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/1ub/dsm-1ub.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT x1ub: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B
