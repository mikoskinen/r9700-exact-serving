#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zrq14.sh (queue item, STAGED 13.11.2026 by a builder for ds-main-port / the coordinator -- NOT
# QUEUED by the builder; sorts after -r92zzzzzz-ref12zzzzzz-zref14 and before -r92zzzzzz-ref12zzzzzz-zsa14 / -zzk25 in C and en_US order;
# gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~16.5 min without start 2, ~24.5 with it; budget ZQ14_BUDGET1 20
# for start 1's sections, ZQ14_BUDGET_MIN 31 for all, <= 32 with the summary; NO evict) -- REF14 BELOW REF12: int14 or MAXSEQS 16 or
# the block window? IMAGE radiance-local:ds41-int14c-678c176f-579d0bc2-l (a7ca6bcdf07b), THE int14 PIN (ref14's O), A4 table + A11 ranged
# entries (ref14's copies, md5-checked). START 1: MAXSEQS 16, ONE FRESH START (ngshare, no ATTACH, compose guard): identity a1-B / a1-Z vs
# ref12's a1 (strict; ANY difference: VIRHE and STOP), c1 B Z Z B + a 35 s Z bridge, c4 warm-up + B Z Z B + a 35 s Z bridge (c1 / c4 in
# ref12's 20 s windows: ref14 read 35 s), 1u essay B (1 + 3), Z (1 + 3), B x3; B = every int14 lever at 0 (int12 bytes), Z = int14.
# THE GATE (logged either way): c1 B mean < 121.0 OR c4 B mean < 170.0 -> START 2: MAXSEQS 8, B only, a1-B8 vs ref12, c1 x2, c4 warm-up
# + 2, 1u 1 + 3. Every block / read: tok/s, tok a step, ms a step, acceptance, step-log row buckets, depth-0 share, q histogram, draft
# controller deltas (GET /admin/fn4 + /stats). Tag "RESULT xzq14 ", summary lines "ZQ14 ...", out runs/xdsm/zrq14, body
# queue-ar/dsmain/zrq14/dsm-zrq14.sh (a COPY of ref14/dsm-ref14.sh 74981d12 made by zrq14/mkbody.py) + zrq14/zq14sum.py + the two
# tables. ZQ14_SW=0 = a mock run (queue-ar/tools/mockrad; zrq14/mockrun-zrq14.sh). DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zrq14.sh :
# prints every command; checks the image literals (int9 .. int12 + int14c, int12's plugin md5), every helper (+ selftests), the tables'
# md5s, the ngramshare files, the sidecars, the yamls, the disk.
export ZQ14_IMAGE=${ZQ14_IMAGE:-radiance-local:ds41-int14c-678c176f-579d0bc2-l} ZQ14_ITEM=${ZQ14_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zrq14}
export ZQ14_OUT=${ZQ14_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/zrq14} ZQ14_SW=${ZQ14_SW:-1}
export ZQ14_MAXSEQS=${ZQ14_MAXSEQS:-16} ZQ14_MAXSEQS2=${ZQ14_MAXSEQS2:-8} ZQ14_START2=${ZQ14_START2:-auto} ZQ14_GATE_C1=${ZQ14_GATE_C1:-121.0} ZQ14_GATE_C4=${ZQ14_GATE_C4:-170.0}
export ZQ14_C1ARMS=${ZQ14_C1ARMS-BZZB} ZQ14_C4ARMS=${ZQ14_C4ARMS-BZZB} ZQ14_U1N=${ZQ14_U1N:-3} ZQ14_C1N2=${ZQ14_C1N2:-2} ZQ14_C4N2=${ZQ14_C4N2:-2}
export ZQ14_C1WIN=${ZQ14_C1WIN:-20} ZQ14_C1WARM=${ZQ14_C1WARM:-6} ZQ14_C4WIN=${ZQ14_C4WIN:-20} ZQ14_C4WARM=${ZQ14_C4WARM:-8} ZQ14_BRIDGE=${ZQ14_BRIDGE:-1} ZQ14_BRIDGEWIN=${ZQ14_BRIDGEWIN:-35}
export ZQ14_PLUGMD5=${ZQ14_PLUGMD5:-9823cdde} ZQ14_SWL=${ZQ14_SWL:-1} ZQ14_SWLG=${ZQ14_SWLG:-75} ZQ14_XPMIN=${ZQ14_XPMIN:-1048576}
export ZQ14_A4MD5=${ZQ14_A4MD5:-d55ad354} ZQ14_A4N=${ZQ14_A4N:-95} ZQ14_A11MD5=${ZQ14_A11MD5:-8e4fc84e} ZQ14_A11NE=${ZQ14_A11NE:-8}
export ZQ14_DISKMIN_MB=${ZQ14_DISKMIN_MB:-11000} ZQ14_BUDGET1=${ZQ14_BUDGET1:-20} ZQ14_BUDGET_MIN=${ZQ14_BUDGET_MIN:-31} ZQ14_S2EST=${ZQ14_S2EST:-9}
export ZQ14_SIDECAR=${ZQ14_SIDECAR:-dspark-dlw-q8_0.rad} ZQ14_SIDESHA=${ZQ14_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/zrq14/dsm-zrq14.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xzq14: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B
