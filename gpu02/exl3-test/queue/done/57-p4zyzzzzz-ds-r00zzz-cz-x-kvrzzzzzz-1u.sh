#!/bin/bash
# 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzzzz-1u.sh (queue item, STAGED 12.11.2026 by the int11 builder for ds-main-port / the coordinator;
# sorts after -cz-x-kvrzzzzz-arc and before -cz-xz-* in C and en_US order; gpu02, all 4 cards, TP4, MML 524288, every radiance
# container DOWN; ~34 min unskipped, budget X1U_BUDGET_MIN 35 (late sections skipped and logged: second c8 pairs and MX12 first),
# NO evict) -- THE ALL-SWITCH SCREEN of int11, ONE start: 1u levers (1U-MX ds41_mix_wv, 1U-GQ ds41_gq, 1U-E2 ds41_e2, 1U-F4
# ds41_shexp_lane, 1U-C1T dspark_conf 50 / 40 / 30; 1U-CHN SKIPPED: no runtime switch) + lever CBH (sched_cb_hold_ms 0 / 250 / 500,
# k 2, c8 on two schedules), tag "RESULT x1u ", out runs/xdsm/i11u, body queue-ar/dsmain/1u/dsm-1u.sh (from a COPY of
# arc/dsm-arc.sh; + 1u/u1sum.py: step-log ms a step by verify rows, the card-0 stamps, the summary). NOT QUEUED.
# IMAGE radiance-local:ds41-int11-d3569104-5c13e555-l (ds41-int11 d356910 = int10 f6ef7cc + 0446636 (fn4 ds41_mix_wv 57 / ds41_gq
# 58 / ds41_e2 59) + aababf1 + the CBH merge beb243b (sched_cb_hold_ms 60 / sched_cb_hold_k 61) + the GQ merge 63485b9; kSwitches 62).
# P0 (CARD 0, no server): r4d_selftest --case mix_wv / gq_pair; a failed case SKIPS its lever's arms (VIRHE logged), not the item.
# ONE START = the ARC screen's (stack5's + RADIANCE_DS41_MHC_FUSE=0) + RADIANCE_DS41_GQ=3 + RADIANCE_DS41_E2=1 + RADIANCE_AQL_STAMP=17
# (card 0, every 17th pass: the 1U-HG / 1U-DQ stamped read, ~0.5 % on every arm alike); KVR / SP off; ARC parked (unset, pinned 0).
# BASE PIN = stack5's S5T, draft_sample 0, M1 on (mhc_fuse 1), ix_lds / ix_ni 0, draft_depth0 3, sched_row_cap 48, ds41_arcoded 0,
# dspark_conf 1, dspark_conf_rows 1, every new lever 0, sched_cb_hold_ms 0 / k 2. In process (fn4, read back):
#   1u reads = base1113's (stepprobe essay,code 512, T 0), the all-off O FIRST, two an arm: MX O MX O MX (ds41_mix_wv 24), GQ (3),
#   E2 (1), F4 (ds41_shexp_lane 1); identity c1 NH (all on + hold 500) vs O and c8 H500 vs O (its holds printed), strict; C1T C50
#   C40 C30 C30 C40 C50; mb on the FIXED-SCHEDULE harness: c8 S:O S:N S:H250 S:H500 C:O C:H250 C:H500 (--gens 2; S = default spacing:
#   CBH's cost side, C = --spacing 6: an upper bound on gain; + a second pair only when |X vs O| < 1.0 %, or c8mix DIFFERS for S:N); c4: one discarded warm-up block, then O O (the
#   "O c4 confirm" read vs int10's 165.7 and the 160 target) and N; c1 O N N O; MX12 O MX12 last.
# Guards: switch read back, 0 request errors, the i11 counters (1u blocks), the cbh counters (mb blocks), byte identity.
# X1U_SW=0 = a mock run (queue-ar/tools/mockrad): no int9b / int10 / int11 switch posted, P0 not fatal, a "MOCK-RUN X1U_SW=0" marker.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzzzz-1u.sh : prints every command; checks the image literals (int9 + int9b + int10 +
# int11 incl. CBH / GQ), every helper (+ selftests), the sidecars, the yaml, the disk; no server, no GPU, writes nothing.
export X1U_IMAGE=${X1U_IMAGE:-radiance-local:ds41-int11-d3569104-5c13e555-l} X1U_ITEM=${X1U_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzzzz-1u}
export X1U_OUT=${X1U_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/i11u} X1U_SW=${X1U_SW:-1}
export X1U_PLAN=${X1U_PLAN-MX:O,MX,O,MX GQ:O,GQ,O,GQ E2:O,E2,O,E2 F4:O,F4,O,F4} X1U_PLAN2=${X1U_PLAN2-C1T:C50,C40,C30,C30,C40,C50} X1U_PLAN3=${X1U_PLAN3-MX12:O,MX12}
export X1U_EPROMPTS=${X1U_EPROMPTS:-essay,code} X1U_EGEN=${X1U_EGEN:-512} X1U_MXV=${X1U_MXV:-24} X1U_ARC=${X1U_ARC:-0} X1U_STAMP=${X1U_STAMP:-17}
export X1U_IDN=${X1U_IDN:-1} X1U_C4ID=${X1U_C4ID:-1} X1U_CBHK=${X1U_CBHK:-2} X1U_MHC=${X1U_MHC:-1} X1U_D0=${X1U_D0:-3} X1U_CAP=${X1U_CAP:-48} X1U_D=${X1U_D:-1}
export X1U_C8PLAN=${X1U_C8PLAN-S:O S:N S:H250 S:H500 C:O C:H250 C:H500} X1U_C8WIN=${X1U_C8WIN:-35} X1U_C8WARM=${X1U_C8WARM:-10} X1U_CSPACING=${X1U_CSPACING:-6} X1U_C8GENS=${X1U_C8GENS:-2}
export X1U_C8X=${X1U_C8X:-1} X1U_C8TH=${X1U_C8TH:-1.0} X1U_C4ORDER=${X1U_C4ORDER-OON} X1U_C4WIN=${X1U_C4WIN:-20} X1U_C4WARM=${X1U_C4WARM:-8}
export X1U_C1ORDER=${X1U_C1ORDER-ONNO} X1U_C1WIN=${X1U_C1WIN:-20} X1U_C1WARM=${X1U_C1WARM:-6}
export X1U_P0="${X1U_P0-mix_wv:18:600 gq_pair:60:600}"
export X1U_DISKMIN_MB=${X1U_DISKMIN_MB:-11000} X1U_BUDGET_MIN=${X1U_BUDGET_MIN:-35}
export X1U_SIDECAR=${X1U_SIDECAR:-dspark-dlw-q8_0.rad} X1U_SIDESHA=${X1U_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/1u/dsm-1u.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT x1u: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard (ghost 622b5b7a9e64)
bash $B
