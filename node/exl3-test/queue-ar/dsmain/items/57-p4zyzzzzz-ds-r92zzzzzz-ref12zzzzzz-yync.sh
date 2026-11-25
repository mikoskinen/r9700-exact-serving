#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yync.sh (queue item, 12.11.2026 by the ds41-nc builder for ds-main-port; sorts after
# -r92zzzzzz-ref12zzzzzz-yswl and before -r92zzzzzz-ref12zzzzzzzzz-fn-* in C and en_US order; gpu02, all 4 cards, TP4, MML 524288, every
# radiance container DOWN; ~35 min estimate, budget YNC_BUDGET_MIN 42 (a section past it skipped, logged), NO evict) -- THE NEWCOMER-STEP
# SCREEN of levers A9 + A11 on the ds41-nc image (= int12 bf8f557 + deepseek41.so / libr4d.so only; no new switch name):
#   A9  = fn4 ds41_bi 6: ref12's ds41_bi 2 + the cheap issue (3 / 4) + ONE MoE all-reduce over a mixed step's rows (ds41-bi 7cfb77b's BI-d
#         on int12; the exact wire's per-element fold, so every row keeps its bytes);
#   A11 = fn4 gg_pf 2: the GPF dense prefill table's ROW-RANGED entries "type:NxK@LO-HI=form" for the newcomer's 1000-1536-row launches
#         (int12's 7 entries are M-blind, chosen at 4096 / 2048 rows), built by this item's P0 bench; the same bytes (GPF's forms).
# P0 ON ONE CARD (chip UUID GPU-dbba5c851cfef7b3, no server): r4d_selftest --gpcheck at the range's heights (every GPF form vs the stock
# kernel, byte for byte; a failure drops A11, A9 goes on), then --perfgp -> ync/nctab.py (the ranged entries: >= 1 % vs int12's pick).
# ONE FRESH START, ref12's pins exactly (read back; ds41_bi 2, gg_pf 1), MAXSEQS 16, the copy path (no ngshare, no ATTACH); arms in-process
# B (ref12) / A (A9) / G (A11) / N (both). READINGS: identity a1 + gs16 under every arm (strict) + a1-B vs ref12's a1 + A11 in range
# (deepq 1100 / 1300 cold, B vs G); the STRICT staggered check under A (and N within the budget); c8 ncmb3 35 s blocks B A G N G A B with
# the P12SPLIT NEWCOMER ms a step (the target) and decode ms a step (guard: not above B's by > 1 %); BEST by c8; c16 (ncmb3 at 16, cap 48,
# 35 s) B + BEST; cold 64k / 136k B + BEST (answers compared). NCSUM (ync/ncsum.py): every reading per arm vs B (values, % difference).
# Tag "RESULT ync ", out runs/xdsm/ync/x1, body queue-ar/dsmain/ync/dsm-ync.sh (a COPY of yd2m/dsm-yd2m.sh made by ync/mkbody.py) +
# ync/nctab.py + ync/ncsum.py; prof12/p12win.py, harness/ncmb3.py, stack4/loadid.py read in place. YNC_SW=0 = a mock run
# (queue-ar/tools/mockrad; ync/mockrun-ync.sh). DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yync.sh : prints every command;
# checks the image literals (int9 .. int12 + nc, int12's plugin md5), every helper (+ selftests), the sidecars, the yaml, the disk.
export YNC_IMAGE=${YNC_IMAGE:-radiance-local:ds41-nc-919a74bf-579d0bc2-l} YNC_ITEM=${YNC_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yync}
export YNC_OUT=${YNC_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/ync/x1} YNC_SW=${YNC_SW:-1}
export YNC_MX=${YNC_MX:-0} YNC_F4=${YNC_F4:-0} YNC_F4G=${YNC_F4G:-0} YNC_MXCONFIRM=${YNC_MXCONFIRM:-0} YNC_GSCONC=${YNC_GSCONC:-16} YNC_MAXSEQS=${YNC_MAXSEQS:-16}
export YNC_PLUGMD5=${YNC_PLUGMD5:-9823cdde} YNC_P0_UUID=${YNC_P0_UUID:-GPU-dbba5c851cfef7b3}
export YNC_A11CMS=${YNC_A11CMS:-1000,1104,1105,1216,1408,1535,1536,1537} YNC_A11MS=${YNC_A11MS:-1024,1104,1216,1408} YNC_A11IT=${YNC_A11IT:-30}
export YNC_A11_RANGE=${YNC_A11_RANGE:-1000-1536} YNC_A11_MIN_PCT=${YNC_A11_MIN_PCT:-1.0} YNC_A11ID="${YNC_A11ID-1100 1300}"
export YNC_C8ORDER=${YNC_C8ORDER-BAGNGAB} YNC_C16ORDER=${YNC_C16ORDER-BX} YNC_CDEPTHS="${YNC_CDEPTHS-64000 136000}" YNC_DECTH=${YNC_DECTH:-1.0}
export YNC_C8WIN=${YNC_C8WIN:-35} YNC_C8WARM=${YNC_C8WARM:-10} YNC_LDCONC=${YNC_LDCONC:-4} YNC_LDPF=${YNC_LDPF:-8192} YNC_STAGN=${YNC_STAGN:-1} YNC_STAGB=${YNC_STAGB:-0}
export YNC_DISKMIN_MB=${YNC_DISKMIN_MB:-11000} YNC_BUDGET_MIN=${YNC_BUDGET_MIN:-42}
export YNC_SIDECAR=${YNC_SIDECAR:-dspark-dlw-q8_0.rad} YNC_SIDESHA=${YNC_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/ync/dsm-ync.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT ync: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B
