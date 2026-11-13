#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zref14.sh (queue item, STAGED 12.11.2026 by a builder for ds-main-port / the coordinator -- NOT
# QUEUED by the builder; sorts after -r92zzzzzz-ref12zzzzzz-yzint14c and before -r92zzzzzz-ref12zzzzzzzzz-fn-* in C and en_US order;
# gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~28 min estimate, budget R14_BUDGET_MIN 35 for the sections (a
# section past it skipped, logged; c1, then c4's 2nd block go first), <= 38 with the summary; NO evict) -- THE int14 REFERENCE READ
# (ref14): the zero point of THE NEW DS BASE int14 (TESTING.md 12.11. "NEW DS BASE int14"; LEVERS.md "INT14C DS BASE CANDIDATE").
# IMAGE radiance-local:ds41-int14c-678c176f-579d0bc2-l (a7ca6bcdf07b). THE int14 PIN = ref12's pins + ar_xpush 1, gg_gemv_m16 1,
# ds41_bi 6, gg_pf 2, place_swap_lent 1, heat_min_gain_x100 75; start env RADIANCE_FN4_AR_XPUSH_MIN=1048576; A4's table
# ref14/m16-forms.txt (= yzint14c's d55ad354) and A11's ranged entries ref14/a11-entries.txt (= yzint14c's P0 p0-tab.entries 8e4fc84e),
# NOT re-benched. ONE FRESH START (ngshare, no ATTACH, MAXSEQS 16, compose guard), int14 pinned by fn4 right after it and READ BACK on
# fn4 and on every rank. READINGS (ref12's set): 1 identity (warm-up vs stock; a1 vs yzint14c's a1-Z + ref12's a1; gs16 vs a1; A11 in
# range; strict staggered ld vs a1) -- ANY difference: VIRHE and STOP before speed; 2 1u essay warm (1 discarded + 3 back to back) +
# the switched essay x2; 3 c8 WORK-BASED (--window gens, in-process B Z Z B, B = every int14 lever at 0 = int12 bytes: coordinator
# 12.11. method fix), c8 35 s x2 (INFO), c16 fixed x2, c16 CAP48; 4 cold 64k / 136k (answers vs yzint14c / ref12); 5 c4 (warm-up + 2),
# c1. Newcomer readings = the MATCHED-PROMPT sums (ref14/ncmatch.py), not p12win.py's per-step mean. Tag "RESULT xref14 ", summary
# lines "REF14 ...", out runs/xdsm/ref14, body queue-ar/dsmain/ref14/dsm-ref14.sh (a COPY of yzint14c/dsm-yzint14c.sh 2a58559d made by
# ref14/mkbody.py) + ref14/r14sum.py + ref14/ncmatch.py + the two tables. R14_SW=0 = a mock run (queue-ar/tools/mockrad;
# ref14/mockrun-ref14.sh). DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zref14.sh : prints every command; checks the image
# literals (int9 .. int12 + int14c, int12's plugin md5), every helper (+ selftests), the tables' md5s, the ngramshare files, the
# sidecars, the yamls, the disk.
export R14_IMAGE=${R14_IMAGE:-radiance-local:ds41-int14c-678c176f-579d0bc2-l} R14_ITEM=${R14_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zref14}
export R14_OUT=${R14_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/ref14} R14_SW=${R14_SW:-1}
export R14_MX=${R14_MX:-0} R14_F4=${R14_F4:-0} R14_F4G=${R14_F4G:-0} R14_MXCONFIRM=${R14_MXCONFIRM:-0} R14_GSCONC=${R14_GSCONC:-16} R14_MAXSEQS=${R14_MAXSEQS:-16}
export R14_PLUGMD5=${R14_PLUGMD5:-9823cdde} R14_SWL=${R14_SWL:-1} R14_SWLG=${R14_SWLG:-75} R14_XPMIN=${R14_XPMIN:-1048576}
export R14_A4MD5=${R14_A4MD5:-d55ad354} R14_A4N=${R14_A4N:-95} R14_A11MD5=${R14_A11MD5:-8e4fc84e} R14_A11NE=${R14_A11NE:-8} R14_A11ID="${R14_A11ID-1100 1300}"
export R14_U1N=${R14_U1N:-3} R14_SWN=${R14_SWN:-2} R14_C8GN=${R14_C8GN:-2} R14_C8GORDER=${R14_C8GORDER-BZZB} R14_C8N=${R14_C8N:-2} R14_C16N=${R14_C16N:-2}
export R14_C4N=${R14_C4N:-2} R14_C4WARMUP=${R14_C4WARMUP:-1} R14_C1N=${R14_C1N:-1} R14_CDEPTHS="${R14_CDEPTHS-64000 136000}"
export R14_C8WIN=${R14_C8WIN:-35} R14_C8WARM=${R14_C8WARM:-10} R14_C4WIN=${R14_C4WIN:-35} R14_C4WARM=${R14_C4WARM:-8} R14_C1WIN=${R14_C1WIN:-35} R14_C1WARM=${R14_C1WARM:-6}
export R14_LDCONC=${R14_LDCONC:-4} R14_LDPF=${R14_LDPF:-8192} R14_DISKMIN_MB=${R14_DISKMIN_MB:-11000} R14_BUDGET_MIN=${R14_BUDGET_MIN:-35}
export R14_SIDECAR=${R14_SIDECAR:-dspark-dlw-q8_0.rad} R14_SIDESHA=${R14_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/ref14/dsm-ref14.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xref14: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B
