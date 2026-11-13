#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzzz-arc.sh (queue item, STAGED 12.11.2026 by the ds-main-port ARC builder for the coordinator;
# sorts after -cz-x-kvrzzzz-k-cap48r and -cz-x-kvrzzzz-l-i9p and before -cz-xz-fnsmoke2 in C and en_US order; gpu02, all 4 cards,
# TP4, MML 524288, every radiance container DOWN; ~25 min estimate, budget XARC_BUDGET_MIN 28 (a late section skipped and logged),
# NO evict) -- LEVER ARC (the coded all-reduce wire at overlap 0, libr4d fn4 ds41_arcoded), tag "RESULT xarc ", out runs/xdsm/arc,
# body queue-ar/dsmain/arc/dsm-arc.sh (from a COPY of i9b/dsm-i9b.sh 64d65518; i9b's i9bctr.py and d0ab's d0blocks.py read in
# place, unchanged). NOT QUEUED: the coordinator queues it.
# IMAGE radiance-local:ds41-int10-f6ef7cc3-5c13e555-l (ds41-int10 f6ef7cc = ds41-int9b f119c56 + ea613fc lever ARC (fn4
# ds41_arcoded 55; GET /admin/fn4 "arc"; RADIANCE_DS41_ARCODED at load; 0 / unset = the int9b image) + 14f34a3 draft_depth0 3
# (depth 0 by occupancy only) + f6ef7cc the merge of ds41-cap48 e5acfbc (lever CAP48, fn4 sched_row_cap 56; kSwitches 57)).
# P0 (ALL FOUR cards, no server, before the start): r4d_selftest --case ar_arcoded / ar_twoshot_4rank_exact /
# ar_twoshot_4rank_exact_lane; a failure stops the item.
# ONE START = stack5's + RADIANCE_DS41_MHC_FUSE=0 (M1 declared); KVR / SP off. BASE PIN = stack5's S5T, draft_sample 0, SP 0 / 0,
# M1 on (mhc_fuse 1), ix_lds 0, ix_ni 0, draft_depth0 3, sched_row_cap 48 (CAP48), draft_lend 1. Arms in process (fn4, read back): B ds41_arcoded 0, C
# ds41_arcoded 1024. Readings: the D block (mb c4 + mb c8 old loop under B: c4 zero_cycles ~0, c8 occ rule, CAP48 counters), cold 135k / 64k TTFT
# B C C B (512 greedy tokens), c8 B C C B on the fixed-schedule harness (newcomer step + agg), identity (greedy set c1 C vs B; c4
# B vs d0ab A), 1u essay T 0 C B. Guards: switch read back, 0 request errors, the ARC counters (C coded > 0, B none), byte
# identity (cold answers + greedy set; a difference = VIRHE).
# XARC_SW=0 = a mock run (queue-ar/tools/mockrad): no int9 / int9b / int10 switch posted, P0 not fatal, a "VIRHE XARC_SW=0" marker.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzzz-arc.sh : prints every command; checks the image literals (int9 + int9b +
# int10), every helper (+ selftests), the sidecars, the yaml, the disk; no server, no GPU, writes nothing.
export XARC_IMAGE=${XARC_IMAGE:-radiance-local:ds41-int10-f6ef7cc3-5c13e555-l} XARC_ITEM=${XARC_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzzz-arc}
export XARC_OUT=${XARC_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/arc} XARC_SW=${XARC_SW:-1}
export XARC_N=${XARC_N:-1024} XARC_COLD="${XARC_COLD:-135000B 135000C 64000C 64000B}" XARC_NTOK=${XARC_NTOK:-512}
export XARC_C8ORDER=${XARC_C8ORDER-BCCB} XARC_C8DUR=${XARC_C8DUR:-30} XARC_C8WARM=${XARC_C8WARM:-10} XARC_C8GENS=${XARC_C8GENS:-2}
export XARC_EORDER=${XARC_EORDER-CB} XARC_EGEN=${XARC_EGEN:-1024} XARC_IDN=${XARC_IDN:-1} XARC_C4ID=${XARC_C4ID:-1}
export XARC_MHC=${XARC_MHC:-1} XARC_D0=${XARC_D0:-3} XARC_CAP=${XARC_CAP:-48} XARC_BASEX=${XARC_BASEX:-} XARC_SPN=${XARC_SPN:-0} XARC_SPK=${XARC_SPK:-0}
export XARC_D=${XARC_D:-1} XARC_DDUR=${XARC_DDUR:-30} XARC_DWARM=${XARC_DWARM:-10}
export XARC_P0="${XARC_P0-ar_arcoded:29:420 ar_twoshot_4rank_exact:17:150 ar_twoshot_4rank_exact_lane:24:150}"
export XARC_DISKMIN_MB=${XARC_DISKMIN_MB:-11000} XARC_BUDGET_MIN=${XARC_BUDGET_MIN:-28}
export XARC_SIDECAR=${XARC_SIDECAR:-dspark-dlw-q8_0.rad} XARC_SIDESHA=${XARC_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/arc/dsm-arc.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xarc: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard (ghost 622b5b7a9e64)
bash $B
