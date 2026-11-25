#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzk25s.sh (queue item, STAGED 13.11.2026 by a builder for ds-main-port / the coordinator -- NOT
# QUEUED by the builder; sorts after -r92zzzzzz-ref12zzzzzz-zzk25 and before -r92zzzzzz-ref12zzzzzzzzz-fn-* in C and en_US order;
# gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~17 min estimate, budget K25S_BUDGET_MIN 28 for the sections (a
# section past it skipped, logged), <= 30 with the summary; NO evict, NO build) -- LEVER K25U IN-ENGINE SCREEN (the cheaper K = 2.5
# trellis decode; plugin ds41-k25u 38b88ec, exl3r4d.so md5 7b2f66d6). IMAGE radiance-local:ds41-k25u-678c176f-38b88ec6-l (bc91c460c13f)
# = the int14 image + the K25U exl3r4d.so, THE int14 PIN (= ref14's O), MAXSEQS 16, ONE FRESH START (ngshare, no ATTACH, compose guard).
# ARMS in-process: B = the int14 pin, exl3_x3d 1 (the int14 kernels exactly); Z = the int14 pin + exl3_x3d 2 (X3D + K2.5 gate_up G3 +
# down U4). GET /admin/fn4 exl3_x3d asserted before every block (1 under B, 2 under Z). The int12 / int14 bodies' exl3_x3d 2 -> 400
# refusal probe is MOVED to exl3_x3d 5 (2 is valid on this plugin). 1 IDENTITY FIRST: a1-B and a1-Z vs ref12's a1, gs16-Z vs a1-Z,
# STRICT staggered ld-Z vs a1-Z -- ANY difference: VIRHE and STOP before speed; 2 c8 WORK-BASED (harness gens window, gens 2) B Z Z B;
# 3 c16 gens 1: B, Z; 4 1u essay: B (1 discarded + 3) then Z (1 discarded + 3). Readings: decode-step p50 from the step log over all
# decode steps (+ full-row bucket, block separation), matched-prompt newcomer sums (ncmatch.py), agg, the cycle model, 1u tok/s / ms a
# step at matched residency; int14 references (ref14 c8 gens 190.1, c16 226.7 / CAP48 307.6, 1u 89.85) INFO (zzk25s/k25sum.py, lines
# "K25S ..."). Tag "RESULT xk25s ", out runs/xdsm/zzk25s, body queue-ar/dsmain/zzk25s/dsm-zzk25s.sh (a COPY of zsa14/dsm-zsa14.sh
# fe5c6c3b made by zzk25s/mkbody.py). K25S_SW=0 = a mock run (queue-ar/tools/mockrad; zzk25s/mockrun-zzk25s.sh).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzk25s.sh : prints every command; checks the image literals (int9 .. int14c), the
# K25U plugin md5, every helper (+ selftests), the tables' and steps.py's md5s, the ngramshare files, the sidecars, the yamls, the disk.
export K25S_IMAGE=${K25S_IMAGE:-radiance-local:ds41-k25u-678c176f-38b88ec6-l} K25S_ITEM=${K25S_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzk25s}
export K25S_OUT=${K25S_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/zzk25s} K25S_SW=${K25S_SW:-1}
export K25S_MX=${K25S_MX:-0} K25S_F4=${K25S_F4:-0} K25S_F4G=${K25S_F4G:-0} K25S_MXCONFIRM=${K25S_MXCONFIRM:-0} K25S_GSCONC=${K25S_GSCONC:-16} K25S_MAXSEQS=${K25S_MAXSEQS:-16}
export K25S_PLUGMD5=${K25S_PLUGMD5:-7b2f66d6} K25S_SWL=${K25S_SWL:-1} K25S_SWLG=${K25S_SWLG:-75} K25S_XPMIN=${K25S_XPMIN:-1048576}
export K25S_A4MD5=${K25S_A4MD5:-d55ad354} K25S_A4N=${K25S_A4N:-95} K25S_A11MD5=${K25S_A11MD5:-8e4fc84e} K25S_A11NE=${K25S_A11NE:-8}
export K25S_XB=${K25S_XB:-1} K25S_XZ=${K25S_XZ:-2} K25S_XREFUSE=${K25S_XREFUSE:-5} K25S_GORDER=${K25S_GORDER-B Z Z B} K25S_C8GN=${K25S_C8GN:-2}
export K25S_C16N=${K25S_C16N:-1} K25S_C16GN=${K25S_C16GN:-1} K25S_C16ARMS=${K25S_C16ARMS-B Z} K25S_POST=${K25S_POST:-10} K25S_IDARMS=${K25S_IDARMS-B Z} K25S_GSARM=${K25S_GSARM:-Z} K25S_LDARM=${K25S_LDARM:-Z}
export K25S_1U=${K25S_1U:-1} K25S_1UARMS=${K25S_1UARMS-B Z} K25S_U1N=${K25S_U1N:-3} K25S_ESSAY_MD5=${K25S_ESSAY_MD5:-47a39dac581a}
export K25S_C8WARM=${K25S_C8WARM:-10} K25S_LDCONC=${K25S_LDCONC:-4} K25S_LDPF=${K25S_LDPF:-8192} K25S_DISKMIN_MB=${K25S_DISKMIN_MB:-11000} K25S_BUDGET_MIN=${K25S_BUDGET_MIN:-28}
export K25S_STPMD5=${K25S_STPMD5:-569fd59b}
export K25S_SIDECAR=${K25S_SIDECAR:-dspark-dlw-q8_0.rad} K25S_SIDESHA=${K25S_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/zzk25s/dsm-zzk25s.sh
if [ ! -e $B ]; then
  [ "${DRYRUN:-0}" = 1 ] && { echo "DRYRUN: $B MISSING"; exit 1; }
  echo "$(date '+%F %T') RESULT xk25s: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1
fi
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B
