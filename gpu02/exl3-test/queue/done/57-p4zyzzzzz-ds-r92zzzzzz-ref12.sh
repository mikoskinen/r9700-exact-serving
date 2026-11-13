#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12.sh (queue item, STAGED 12.11.2026 by the int12 builder for ds-main-port / the coordinator: FINAL
# pins per the c4lev verdict (16-row bucket: no lever clearly helps or hurts c4; F4G dropped): MX 0, F4 0, MX12 confirm INFO -- NOT QUEUED; sorts after -r92zzzzz-c4lev in C and en_US order; gpu02, all 4
# cards, TP4, MML 524288, every radiance container DOWN; ~22 min estimate, budget X12_BUDGET_MIN 35 (a section past it skipped, logged),
# NO evict) -- "NEW DS REFERENCE (int12)": the int12 reference screen. Tag "RESULT xref12 ", out runs/xdsm/ref12, body
# queue-ar/dsmain/ref12/dsm-ref12.sh (a COPY of 1u/dsm-1u.sh 58fad99d; + ref12/ref12sum.py). The COPY PATH: no ngshare, no ATTACH.
# IMAGE X12_IMAGE (default radiance-local:ds41-int12-6f3e70df-579d0bc2-l: int11 + the X3D-F merge, fn4 exl3_x3d 62, kSwitches 63,
# kernels/exl3r4d.so = ds41-x3d 579d0bc md5 9823cdde; a variable: the rows-gated F4 commit (F4G, literal "lever F4G", X12_F4G=1) may
# change the engine sha).
# ONE FRESH START = the int11 screen's (stack5's + LOG_STEPS=1, ENGDD=0, MHC_FUSE=0, HB ranks 1,2, GQ=3 / E2=1 declared) +
# RADIANCE_DS41_KV_REPLICATED=1 (KVR); no stamping, no P0. THE REFERENCE PIN (read back): S5T, draft_sample 0, M1 on (mhc_fuse 1),
# draft_depth0 3, sched_row_cap 48, exl3_x3d 1, ds41_e2 1, ds41_gq 3, ds41_mix_wv X12_MX (0), ds41_shexp_lane X12_F4 (0), SP 0, ARC 0,
# CBH 0. READINGS: 1u essay / code 512 (one discarded warm-up each + two warm reads each; stream MB / promotions; ms a step by verify
# rows); MX12 confirm (X12_MXCONFIRM=1); identity a1 vs int11 O (runs/xdsm/i11u/x1u-a-O.jsonl) + gs16 vs a1 (strict, T 0); mb c8 one
# ~35 s block, c4 one discarded warm-up + two, c1 two (the fixed-schedule harness); cold 64k / 136k (TTFT). SUMMARY: the TABLE vs the
# targets (1u >= 90, c4 >= 160, c8 >= 200, 136k <= 35 s) and int11's O numbers.
# X12_SW=0 = a mock run (queue-ar/tools/mockrad). DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzzz-ref12.sh : prints every command; checks the
# image literals (int9 .. int12, the plugin md5, + F4G with X12_F4G=1), every helper (+ selftests), the sidecars, the yaml, the disk.
export X12_IMAGE=${X12_IMAGE:-radiance-local:ds41-int12-6f3e70df-579d0bc2-l} X12_ITEM=${X12_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12}
export X12_OUT=${X12_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/ref12} X12_SW=${X12_SW:-1}
export X12_MX=${X12_MX:-0} X12_F4=${X12_F4:-0} X12_F4G=${X12_F4G:-0} X12_MXCONFIRM=${X12_MXCONFIRM:-1} X12_GSCONC=${X12_GSCONC:-16} X12_MAXSEQS=${X12_MAXSEQS:-}
export X12_EGEN=${X12_EGEN:-512} X12_NTOK=${X12_NTOK:-64} X12_DEPTHS="${X12_DEPTHS:-64000 136000}" X12_PLUGMD5=${X12_PLUGMD5:-9823cdde}
export X12_C8WIN=${X12_C8WIN:-35} X12_C8WARM=${X12_C8WARM:-10} X12_C4WIN=${X12_C4WIN:-20} X12_C4WARM=${X12_C4WARM:-8} X12_C1WIN=${X12_C1WIN:-20} X12_C1WARM=${X12_C1WARM:-6}
export X12_DISKMIN_MB=${X12_DISKMIN_MB:-11000} X12_BUDGET_MIN=${X12_BUDGET_MIN:-35}
export X12_SIDECAR=${X12_SIDECAR:-dspark-dlw-q8_0.rad} X12_SIDESHA=${X12_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/ref12/dsm-ref12.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xref12: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B
