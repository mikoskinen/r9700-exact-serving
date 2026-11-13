#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-xpush.sh (queue item, 12.11.2026 by the XPUSH builder for ds-main-port; sorts after
# -r92zzzzzz-ref12zzzzz-res in C and en_US order; gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~23 min
# estimate, budget XPU_BUDGET_MIN 28 (a section past it skipped, logged), NO evict) -- XPUSH SCREEN on int13 (ds41-int13 c633a47 =
# int12 + XPUSH ported from FN fn4-x3-sp3 808866c2: libr4d fn4 ar_xpush, the four-rank exact two-shot's first hop interleaved over the
# peers from 131072 elements; the same bytes). P0 ON FOUR CARDS before the start: r4d_selftest --case ar_twoshot_4rank_exact and
# ar_oneshot_4rank_exact under RADIANCE_FN4_AR_XPUSH 0 and 1 (== the fp32 fold, bit for bit; the same ok lines; the XPUSH line only
# at 1) -- a failure STOPS the item -- + --case ar_xpush_perf (INFO: the two forms timed at 256 / 1024 / 2048 / 4096 rows x 5120).
# ONE START, ref12's pins exactly (S5T, draft_sample 0, M1, draft_depth0 3, CAP48 via fn4, exl3_x3d 1, E2 1, GQ 3, KVR=1; MX / F4 /
# SP / ARC / CBH / XPREFETCH 0) + ar_xpush 0 in the base pin, MAXSEQS 16, the copy path (no ngshare, no ATTACH); arms in-process
# B = ar_xpush 0, C = ar_xpush 1 (read back). READINGS: identity (greedy set c1 alone + gs16 at conc 16 under cap 48, B then C,
# strict); cold deepq 64k / 136k TTFT B C C B, 64 greedy tokens, answers byte-compared; c8 newcomer-step ms on the fixed harness
# (ncmb3 fixed prompts reset per block, 35 s windows) B C C B with the p12win wall split (int12: 538.2 ms a newcomer step, 20 % of
# the window) + the decode guard (C's c8 decode-step ms not above B's by > 1 %); c16 on the same harness (n 16) B then C: agg,
# decode_share, newcomer ms, > 48-row steps (0). XPSUM: C vs B per reading. Tag "RESULT xpu ", out runs/xdsm/xpush/x1, body
# queue-ar/dsmain/xpush/dsm-xpush.sh (a COPY of prof12/dsm-prof1.sh) + xpush/xpsum.py; prof12/p12win.py read in place.
# XPU_SW=0 = a mock run (queue-ar/tools/mockrad; xpush/mockrun-xpush.sh). DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-xpush.sh
# : prints every command; checks the image literals (int9 .. int13, the plugin md5), every helper (+ selftests), the sidecars, the
# yaml, the disk.
export XPU_IMAGE=${XPU_IMAGE:-radiance-local:ds41-int13-c633a474-579d0bc2-l} XPU_ITEM=${XPU_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-xpush}
export XPU_OUT=${XPU_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/xpush/x1} XPU_SW=${XPU_SW:-1}
export XPU_MX=${XPU_MX:-0} XPU_F4=${XPU_F4:-0} XPU_F4G=${XPU_F4G:-0} XPU_MXCONFIRM=${XPU_MXCONFIRM:-0} XPU_GSCONC=${XPU_GSCONC:-16} XPU_MAXSEQS=${XPU_MAXSEQS:-16}
export XPU_PLUGMD5=${XPU_PLUGMD5:-9823cdde} XPU_NTOK=${XPU_NTOK:-64} XPU_DEPTHS="${XPU_DEPTHS:-64000 136000}"
export XPU_CORDER=${XPU_CORDER-BCCB} XPU_C8ORDER=${XPU_C8ORDER-BCCB} XPU_C16ORDER=${XPU_C16ORDER-BC} XPU_DECTH=${XPU_DECTH:-1.0}
export XPU_C8WIN=${XPU_C8WIN:-35} XPU_C8WARM=${XPU_C8WARM:-10} XPU_P0MIN=${XPU_P0MIN:-12} XPU_PERFROWS=${XPU_PERFROWS:-256,1024,2048,4096}
export XPU_DISKMIN_MB=${XPU_DISKMIN_MB:-11000} XPU_BUDGET_MIN=${XPU_BUDGET_MIN:-28}
export XPU_SIDECAR=${XPU_SIDECAR:-dspark-dlw-q8_0.rad} XPU_SIDESHA=${XPU_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/xpush/dsm-xpush.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xpu: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B
