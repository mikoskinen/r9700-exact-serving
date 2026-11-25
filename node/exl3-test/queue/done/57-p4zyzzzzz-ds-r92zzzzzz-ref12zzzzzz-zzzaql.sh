#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzzaql.sh (queue item, STAGED 13.11.2026 by a builder for ds-main-port / the coordinator -- NOT
# QUEUED by the builder; sorts after -r92zzzzzz-ref12zzzzzz-zzk25s and before -r92zzzzzz-ref12zzzzzzzzz-fn-* in C and en_US order;
# gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~12-14 min estimate, budget ZQA_BUDGET_MIN 16 for the sections,
# <= 20 with the summary; NO evict, NO build) -- THE AQL-STAMPED c8 CRITICAL-PATH READ on THE DS BASE int14 (measurement only): how
# much of the real c8 decode step is IDLE on card 0's critical path, by class (dependent-launch gaps / lane waits / host / AR wait /
# other), steady steps and the 10 right after a newcomer apart -> "launch / sync levers worth >= 1 ms: YES / NO". The ha-k1prof /
# hy-z0prof method (LEVERS K1, Z0, 1U-HG): RADIANCE_AQL_STAMP=1 (every played pass on card 0 timed per dispatch), compared with the
# UNSTAMPED twin = ref14's c8-1-S-O / c8-2-S-O blocks (same image, pin, harness, MAXSEQS 16). prof2's PROFILED numbers are INFO.
# IMAGE radiance-local:ds41-int14c-678c176f-579d0bc2-l (a7ca6bcdf07b), THE int14 PIN exactly (ref14's arm O), start / pin / table code
# from ref14/dsm-ref14.sh (74981d12). ONE FRESH START, stamped (ngshare, no ATTACH, MAXSEQS 16, compose guard), the pin read back on
# fn4 and every rank; the warm-up identity (a difference STOPS); 2 stamped c8 blocks (ncmb3 fixed 1x8, schedule S, 10 + 35 s, T 0);
# the summary (aqlcp.py). Tag "RESULT xzaql ", summary lines "AQL c8 ...", out runs/xdsm/zzzaql, body
# queue-ar/dsmain/zzzaql/dsm-zzzaql.sh (a COPY of ref14/dsm-ref14.sh made by zzzaql/mkbody.py) + zzzaql/aqlcp.py. The stamp dump
# (~1-2 GB raw) is gzipped once summarised. ZQA_SW=0 = a mock run (queue-ar/tools/mockrad; zzzaql/mockrun-zzzaql.sh: SYNTHETIC stamps).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzzaql.sh : prints every command; checks the image literals (int9 .. int14c, int12's
# plugin md5), every helper (+ selftests, aqlcp.py's), the tables' md5s, the ngramshare files, the sidecars, the yamls, the twin, the disk.
export ZQA_IMAGE=${ZQA_IMAGE:-radiance-local:ds41-int14c-678c176f-579d0bc2-l} ZQA_ITEM=${ZQA_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzzaql}
export ZQA_OUT=${ZQA_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/zzzaql} ZQA_SW=${ZQA_SW:-1} ZQA_STAMP=${ZQA_STAMP:-1} ZQA_NC=${ZQA_NC:-10} ZQA_MIN_STEPS=${ZQA_MIN_STEPS:-200}
export ZQA_MX=${ZQA_MX:-0} ZQA_F4=${ZQA_F4:-0} ZQA_F4G=${ZQA_F4G:-0} ZQA_MXCONFIRM=${ZQA_MXCONFIRM:-0} ZQA_GSCONC=${ZQA_GSCONC:-16} ZQA_MAXSEQS=${ZQA_MAXSEQS:-16}
export ZQA_PLUGMD5=${ZQA_PLUGMD5:-9823cdde} ZQA_SWL=${ZQA_SWL:-1} ZQA_SWLG=${ZQA_SWLG:-75} ZQA_XPMIN=${ZQA_XPMIN:-1048576}
export ZQA_A4MD5=${ZQA_A4MD5:-d55ad354} ZQA_A4N=${ZQA_A4N:-95} ZQA_A11MD5=${ZQA_A11MD5:-8e4fc84e} ZQA_A11NE=${ZQA_A11NE:-8} ZQA_A11ID="${ZQA_A11ID-1100 1300}"
export ZQA_C8N=${ZQA_C8N:-2} ZQA_C8WIN=${ZQA_C8WIN:-35} ZQA_C8WARM=${ZQA_C8WARM:-10}
export ZQA_DISKMIN_MB=${ZQA_DISKMIN_MB:-12000} ZQA_BUDGET_MIN=${ZQA_BUDGET_MIN:-16}
export ZQA_SIDECAR=${ZQA_SIDECAR:-dspark-dlw-q8_0.rad} ZQA_SIDESHA=${ZQA_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/zzzaql/dsm-zzzaql.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xzaql: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B
