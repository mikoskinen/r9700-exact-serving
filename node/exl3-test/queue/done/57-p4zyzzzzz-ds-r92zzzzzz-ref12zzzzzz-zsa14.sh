#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zsa14.sh (queue item, STAGED 12.11.2026 by a builder for ds-main-port / the coordinator -- NOT
# QUEUED by the builder; sorts after -r92zzzzzz-ref12zzzzzz-zref14 and before -r92zzzzzz-ref12zzzzzzzzz-fn-* in C and en_US order;
# gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~24 min estimate, budget SA14_BUDGET_MIN 28 for the sections (a
# section past it skipped, logged; the c16 pair goes first), <= 30 with the summary; NO evict, NO build) -- LEVER SA RE-READ ON int14
# (zero-build screen): fn4 stage_short / stage_pick (the prefill stager armed for newcomer-sized steps at the span's shared level),
# already in int14 and set up off at start. IMAGE radiance-local:ds41-int14c-678c176f-579d0bc2-l (a7ca6bcdf07b), THE int14 PIN (=
# ref14's O), MAXSEQS 16, ONE FRESH START (ngshare, no ATTACH, compose guard). ARMS in-process: B = int14 (SA off), S1 = stage_short
# 1025 + stage_pick 1, S2 = stage_short 1025 + stage_pick 2 (the pre-move fastest pick). 1 IDENTITY FIRST: a1 under B / S1 / S2 vs ref12's
# a1, gs16 under S2 vs a1-S2, strict staggered ld under S2 vs a1-S2 (+ SA armed-step count) -- ANY difference: VIRHE and STOP before
# speed; 2 c8 WORK-BASED (harness gens window, gens 2) B S1 S2 S2 S1 B; 3 c16 gens 1: B + the better SA arm. Newcomer readings = the
# MATCHED-PROMPT sums (zsa14/ncmatch.py = ref14's), decode p50 over all decode steps and the 10 after each newcomer, staged / stream /
# h2d GB, a cycle model per arm (zsa14/sa14sum.py, lines "SA14 ..."); steps.py raw listings. Tag "RESULT xsa14 ", out runs/xdsm/zsa14,
# body queue-ar/dsmain/zsa14/dsm-zsa14.sh (a COPY of ref14/dsm-ref14.sh 74981d12 made by zsa14/mkbody.py). SA14_SW=0 = a mock run
# (queue-ar/tools/mockrad; zsa14/mockrun-zsa14.sh). DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zsa14.sh : prints every command;
# checks the image literals (int9 .. int12 + int14c, int12's plugin md5), every helper (+ selftests), the tables' and steps.py's md5s,
# the ngramshare files, the sidecars, the yamls, the disk.
export SA14_IMAGE=${SA14_IMAGE:-radiance-local:ds41-int14c-678c176f-579d0bc2-l} SA14_ITEM=${SA14_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zsa14}
export SA14_OUT=${SA14_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/zsa14} SA14_SW=${SA14_SW:-1}
export SA14_MX=${SA14_MX:-0} SA14_F4=${SA14_F4:-0} SA14_F4G=${SA14_F4G:-0} SA14_MXCONFIRM=${SA14_MXCONFIRM:-0} SA14_GSCONC=${SA14_GSCONC:-16} SA14_MAXSEQS=${SA14_MAXSEQS:-16}
export SA14_PLUGMD5=${SA14_PLUGMD5:-9823cdde} SA14_SWL=${SA14_SWL:-1} SA14_SWLG=${SA14_SWLG:-75} SA14_XPMIN=${SA14_XPMIN:-1048576}
export SA14_A4MD5=${SA14_A4MD5:-d55ad354} SA14_A4N=${SA14_A4N:-95} SA14_A11MD5=${SA14_A11MD5:-8e4fc84e} SA14_A11NE=${SA14_A11NE:-8}
export SA14_SSN=${SA14_SSN:-1025} SA14_PICK1=${SA14_PICK1:-1} SA14_PICK2=${SA14_PICK2:-2} SA14_GORDER=${SA14_GORDER-B S1 S2 S2 S1 B} SA14_C8GN=${SA14_C8GN:-2}
export SA14_C16N=${SA14_C16N:-1} SA14_C16GN=${SA14_C16GN:-1} SA14_POST=${SA14_POST:-10} SA14_IDARMS=${SA14_IDARMS-B S1 S2} SA14_GSARM=${SA14_GSARM:-S2} SA14_LDARM=${SA14_LDARM:-S2}
export SA14_C8WARM=${SA14_C8WARM:-10} SA14_LDCONC=${SA14_LDCONC:-4} SA14_LDPF=${SA14_LDPF:-8192} SA14_DISKMIN_MB=${SA14_DISKMIN_MB:-11000} SA14_BUDGET_MIN=${SA14_BUDGET_MIN:-28}
export SA14_STPMD5=${SA14_STPMD5:-569fd59b}
export SA14_SIDECAR=${SA14_SIDECAR:-dspark-dlw-q8_0.rad} SA14_SIDESHA=${SA14_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/zsa14/dsm-zsa14.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xsa14: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B
