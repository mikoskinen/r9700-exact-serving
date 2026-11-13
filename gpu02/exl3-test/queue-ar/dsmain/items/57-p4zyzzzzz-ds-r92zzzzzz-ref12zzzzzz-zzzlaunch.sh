#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzzlaunch.sh (queue item, STAGED 13.11.2026 by a builder for ds-main-port / the coordinator -- NOT
# QUEUED by the builder; sorts after -r92zzzzzz-ref12zzzzzz-zzzfuse and before -r92zzzzzz-ref12zzzzzzzzz-fn-* in C order (en_US: see the
# hand-back); gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~19 min estimate, budget LCH_BUDGET_MIN 30 for the
# sections (a section past it skipped, logged), <= 35 with the summary; NO evict) -- LEVER RCO IN-ENGINE SCREEN (core fn4 route_copy_ov:
# a routing report's copy recorded with its ranges, so the dispatch after it may play unordered; branch ds41-launch 0f65666).
# IMAGE radiance-local:ds41-launch-0f65666a-38b88ec6-l (af80cdf344f9) = THE int15 IMAGE (ds41-k25u, bc91c460c13f) + bin/radiance from
# ds41-launch 0f65666 only, THE int15 PIN (zzk25s's Z: the int14 pin + exl3_x3d 2), MAXSEQS 16, ONE FRESH START (ngshare, no ATTACH,
# compose guard) with RADIANCE_AQL_STAMP=16 (one played pass in 16 on card 0 timed; both arms). ARMS in-process: B = the int15 pin,
# route_copy_ov 0 (int15 exactly); Z = ALL = the int15 pin + route_copy_ov 1 (lever RCO). GET /admin/fn4 route_copy_ov + exl3_x3d asserted
# before every block (0 / 2 under B, 1 / 2 under Z) and GET "rco" (recordings / copies / freed) logged. Refusal probes exl3_x3d 5 and
# route_copy_ov 2 -> 400. 1 IDENTITY FIRST under ALL-on: a1-B and a1-Z vs ref12's a1, gs16-Z vs a1-Z, STRICT staggered ld-Z vs a1-Z --
# ANY difference: VIRHE and STOP before speed; 2 c8 WORK-BASED (harness gens window, gens 2) B Z Z B, clock pairs around each; 3 c16 gens
# 1: B, Z; 4 1u essay: B (1 discarded + 3) then Z (1 discarded + 3). Readings (zzzlaunch/lchsum.py, lines "LCH ..."): decode-step p50 from
# the step log over all decode steps + the full-row bucket, block separation (THE JUDGED READING), c8 gens agg, matched-prompt newcomer
# sums, the cycle model, c16 / 1u Z vs B; the int15 zero point (zzk25s Z) INFO. THE AQL READ (zzzlaunch/lchaql.py, lines "LCHAQL ..."): per
# arm over the c8 gens blocks' stamped passes, inside the pass: span / busy / idle by class and the launch-gap site table, B vs Z with
# lever RCO's four sites and its engagement. Tag "RESULT xlch ", out runs/xdsm/zzzlaunch, body queue-ar/dsmain/zzzlaunch/dsm-zzzlaunch.sh
# (a COPY of zzk25s/dsm-zzk25s.sh abbf540f made by zzzlaunch/mkbody.py). LCH_SW=0 = a mock run (queue-ar/tools/mockrad;
# zzzlaunch/mockrun-zzzlaunch.sh). DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzzlaunch.sh : prints every command; checks the
# image literals (int9 .. int14c + route_copy_ov), the K25U plugin md5, every helper (+ selftests), the tables' and steps.py's md5s,
# the ngramshare files, the sidecars, the yamls, the disk.
export LCH_IMAGE=${LCH_IMAGE:-radiance-local:ds41-launch-0f65666a-38b88ec6-l} LCH_ITEM=${LCH_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzzlaunch}
export LCH_OUT=${LCH_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/zzzlaunch} LCH_SW=${LCH_SW:-1}
export LCH_MX=${LCH_MX:-0} LCH_F4=${LCH_F4:-0} LCH_F4G=${LCH_F4G:-0} LCH_MXCONFIRM=${LCH_MXCONFIRM:-0} LCH_GSCONC=${LCH_GSCONC:-16} LCH_MAXSEQS=${LCH_MAXSEQS:-16}
export LCH_PLUGMD5=${LCH_PLUGMD5:-7b2f66d6} LCH_SWL=${LCH_SWL:-1} LCH_SWLG=${LCH_SWLG:-75} LCH_XPMIN=${LCH_XPMIN:-1048576}
export LCH_A4MD5=${LCH_A4MD5:-d55ad354} LCH_A4N=${LCH_A4N:-95} LCH_A11MD5=${LCH_A11MD5:-8e4fc84e} LCH_A11NE=${LCH_A11NE:-8}
export LCH_XPIN=${LCH_XPIN:-2} LCH_XREFUSE=${LCH_XREFUSE:-5} LCH_STAMP=${LCH_STAMP:-16} LCH_GORDER=${LCH_GORDER-B Z Z B} LCH_C8GN=${LCH_C8GN:-2}
export LCH_C16N=${LCH_C16N:-1} LCH_C16GN=${LCH_C16GN:-1} LCH_C16ARMS=${LCH_C16ARMS-B Z} LCH_POST=${LCH_POST:-10} LCH_IDARMS=${LCH_IDARMS-B Z} LCH_GSARM=${LCH_GSARM:-Z} LCH_LDARM=${LCH_LDARM:-Z}
export LCH_1U=${LCH_1U:-1} LCH_1UARMS=${LCH_1UARMS-B Z} LCH_U1N=${LCH_U1N:-3} LCH_ESSAY_MD5=${LCH_ESSAY_MD5:-47a39dac581a}
export LCH_C8WARM=${LCH_C8WARM:-10} LCH_LDCONC=${LCH_LDCONC:-4} LCH_LDPF=${LCH_LDPF:-8192} LCH_DISKMIN_MB=${LCH_DISKMIN_MB:-11000} LCH_BUDGET_MIN=${LCH_BUDGET_MIN:-30}
export LCH_STPMD5=${LCH_STPMD5:-569fd59b}
export LCH_SIDECAR=${LCH_SIDECAR:-dspark-dlw-q8_0.rad} LCH_SIDESHA=${LCH_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/zzzlaunch/dsm-zzzlaunch.sh
if [ ! -e $B ]; then
  [ "${DRYRUN:-0}" = 1 ] && { echo "DRYRUN: $B MISSING"; exit 1; }
  echo "$(date '+%F %T') RESULT xlch: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1
fi
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B
