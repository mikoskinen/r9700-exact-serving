#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yzint14c.sh (queue item, 12.11.2026 by the int14c builder for ds-main-port; sorts after
# -r92zzzzzz-ref12zzzzzz-yync / -yswl and before -r92zzzzzz-ref12zzzzzzzzz-fn-* in C and en_US order; gpu02, all 4 cards, TP4, MML 524288,
# every radiance container DOWN; ~38 min estimate, budget YZI_BUDGET_MIN 45 (a section past it skipped, logged; the SWL lever reads first),
# NO evict) -- THE DS BASE-CANDIDATE SCREEN of image int14c (= int12 bf8f557 + XPUSH + A4 + A9 + A11 + SWL / L2, each behind its own
# switch, every default = int12; branch ds41-int14c):
#   P0 (cards by chip UUID; a failure drops only that lever): XPUSH r4d_selftest ar_twoshot / ar_oneshot exact at ar_xpush 0 / 1 on the four
#   cards + the gate (XPUSH from 1048576 elements: ar_xpush_perf 48 / 256 / 1024 rows x 5120 bit for bit at 0 / 1); A4 --mrcheck16 with yd2m's table (yzint14c/m16-forms.txt, md5 d55ad354, 95 lines, not re-benched); A11 --gpcheck M 1000..1537 +
#   --perfgp -> ync/nctab.py (the ranged entries).
#   ONE FRESH START (ref12's pins, MAXSEQS 16, ngshare, no ATTACH, compose guard): 1 IDENTITY FIRST -- a1-ALL / a1-B vs ref12's a1, gs16-ALL
#   vs a1-ALL, A11 in range, STRICT staggered ld-ALL vs a1-ALL and ld-A9 vs a1-B; ANY difference -> VIRHE and the item STOPS before the
#   speed blocks; 2 per-lever reads (c8 B A9 A11 B newcomer ms; A4 c16 fixed + CAP48 B / A4; SWL switched / warm essays B / SWL); 3 all-on
#   reads (c8 B ALL ALL B, c16 ALL, 1u essay B / ALL, cold 136k B / ALL). Summary yzint14c/yzsum.py: one DECISION table vs B + int12 INFO.
# Tag "RESULT yzi ", out runs/xdsm/yzint14c/x1, body queue-ar/dsmain/yzint14c/dsm-yzint14c.sh (a COPY of ync/dsm-ync.sh made by
# yzint14c/mkbody.py) + yzint14c/yzsum.py + yzint14c/m16-forms.txt; ync/nctab.py, yswl/resprobe.py, prof12/p12win.py, harness/ncmb3.py,
# cap48/ncmb.py, stack4/loadid.py, tools/ngramshare read in place. YZI_SW=0 = a mock run (queue-ar/tools/mockrad; yzint14c/mockrun-yzint14c.sh).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yzint14c.sh : prints every command; checks the image literals (int9 .. int12 + int14c,
# int12's plugin md5), every helper (+ selftests), the A4 table, the ngramshare files, the sidecars, the yamls, the disk.
export YZI_IMAGE=${YZI_IMAGE:-radiance-local:ds41-int14c-678c176f-579d0bc2-l} YZI_ITEM=${YZI_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yzint14c}
export YZI_OUT=${YZI_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/yzint14c/x1} YZI_SW=${YZI_SW:-1}
export YZI_MX=${YZI_MX:-0} YZI_F4=${YZI_F4:-0} YZI_F4G=${YZI_F4G:-0} YZI_MXCONFIRM=${YZI_MXCONFIRM:-0} YZI_GSCONC=${YZI_GSCONC:-16} YZI_MAXSEQS=${YZI_MAXSEQS:-16}
export YZI_PLUGMD5=${YZI_PLUGMD5:-9823cdde} YZI_P0_UUID=${YZI_P0_UUID:-GPU-dbba5c851cfef7b3}
export YZI_A11CMS=${YZI_A11CMS:-1000,1104,1105,1216,1408,1535,1536,1537} YZI_A11MS=${YZI_A11MS:-1024,1104,1216,1408} YZI_A11IT=${YZI_A11IT:-30}
export YZI_A11_RANGE=${YZI_A11_RANGE:-1000-1536} YZI_A11_MIN_PCT=${YZI_A11_MIN_PCT:-1.0} YZI_A11ID="${YZI_A11ID-1100 1300}"
export YZI_SWL=${YZI_SWL:-1} YZI_SWLG=${YZI_SWLG:-75} YZI_SWLREADS=${YZI_SWLREADS:-1} YZI_A4MD5=${YZI_A4MD5:-d55ad354} YZI_A4N=${YZI_A4N:-95}
export YZI_XPMIN=${YZI_XPMIN:-1048576} YZI_XPROWS=${YZI_XPROWS:-48,256,1024}   # XPUSH from 1048576 elements (coordinator 13.11., FN 5f0352a8)
export YZI_C8ORDER1=${YZI_C8ORDER1-BAGB} YZI_C8ORDER2=${YZI_C8ORDER2-BZZB} YZI_U1N=${YZI_U1N:-3} YZI_CDEPTHS="${YZI_CDEPTHS-136000}" YZI_DECTH=${YZI_DECTH:-1.0}
export YZI_C8WIN=${YZI_C8WIN:-35} YZI_C8WARM=${YZI_C8WARM:-10} YZI_LDCONC=${YZI_LDCONC:-4} YZI_LDPF=${YZI_LDPF:-8192}
export YZI_DISKMIN_MB=${YZI_DISKMIN_MB:-11000} YZI_BUDGET_MIN=${YZI_BUDGET_MIN:-45}
export YZI_SIDECAR=${YZI_SIDECAR:-dspark-dlw-q8_0.rad} YZI_SIDESHA=${YZI_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/yzint14c/dsm-yzint14c.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT yzi: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B
