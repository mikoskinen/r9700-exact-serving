#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-ypfx.sh (queue item, STAGED 12.11.2026 by the A10 builder for ds-main-port -- NOT QUEUED by the
# builder; sorts after -r92zzzzzz-ref12zzzzz-res and -r92zzzzzz-ref12zzzzzz-xpush in C and en_US order; gpu02, all 4 cards, TP4, MML 524288,
# every radiance container DOWN; ~25 min estimate, budget YPF_BUDGET_MIN 40 (a section past it skipped, logged), NO evict) -- LEVER A10
# SCREEN (X3D in the M > 16 band) on ds41-pfx (radiance 8b52bd9 = int12 bf8f557 + fn4 exl3_x3d_pf 63; exl3r4d ds41-x3d-pf 90b1fef =
# 579d0bc + the prefill band's K = 2 map-form gate_up / down GEMM reading a lane's eight trellis states from one word pair, the same
# f16 weights and WMMA order: the same bytes). P0 ON ONE CARD (chip GPU-dbba5c851cfef7b3) before the start: ypfx/bin/pfxprobe, the
# int12 plugin (OLD) vs the image's (NEW) x3d 0 / 1, h' and y byte for byte at K 2 / 2.5, M 1 / 17 / 64 / 512 / 2048 (8 local
# experts and an EP4 rank's 96 of 384) + the fn4 hooks + an op-level TIME (INFO); a failure STOPS the item. ONE START, ref12's pins
# exactly (S5T, draft_sample 0, M1, draft_depth0 3, CAP48 via fn4, exl3_x3d 1, E2 1, GQ 3, KVR=1; MX / F4 / SP / ARC / CBH / XPREFETCH
# 0) + exl3_x3d_pf 0 in the base pin, MAXSEQS 16, the copy path (no ngshare, no ATTACH); arms in-process B = exl3_x3d_pf 0, C = 1
# (read back). READINGS: identity (greedy set c1 alone + gs16 at conc 16 under cap 48, B then C, strict; a1-B / a1-C vs ref12's a1,
# gs16-C vs a1-C, warm-up vs stock); cold deepq 64k / 136k TTFT B C C B, 64 greedy tokens, answers byte-compared; c8 newcomer-step
# ms on the fixed harness (ncmb3 fixed prompts reset per block, 35 s windows) B C C B with the p12win wall split (int12: 538.2 ms a
# newcomer step, 20 % of the window) + the decode guard (C's c8 decode-step ms not above B's by > 1 %); c16 on the same harness
# (n 16) B then C. YPSUM: C vs B per reading. Tag "RESULT ypf ", out runs/xdsm/ypfx/y1, body queue-ar/dsmain/ypfx/dsm-ypfx.sh (a COPY
# of xpush/dsm-xpush.sh) + ypfx/ypsum.py + ypfx/bin (pfxprobe, exl3r4d-int12.so; MD5SUMS); prof12/p12win.py read in place.
# YPF_SW=0 = a mock run (queue-ar/tools/mockrad; ypfx/mockrun-ypfx.sh). DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-ypfx.sh
# : prints every command; checks the image literals (int9 .. int12 + A10's, the plugin md5), the P0 binaries, every helper (+
# selftests), the sidecars, the yaml, the disk.
export YPF_IMAGE=${YPF_IMAGE:-radiance-local:ds41-pfx-8b52bd9e-90b1fef5-l} YPF_ITEM=${YPF_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-ypfx}
export YPF_OUT=${YPF_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/ypfx/y1} YPF_SW=${YPF_SW:-1}
export YPF_MX=${YPF_MX:-0} YPF_F4=${YPF_F4:-0} YPF_F4G=${YPF_F4G:-0} YPF_MXCONFIRM=${YPF_MXCONFIRM:-0} YPF_GSCONC=${YPF_GSCONC:-16} YPF_MAXSEQS=${YPF_MAXSEQS:-16}
export YPF_PLUGMD5=${YPF_PLUGMD5:-bb2d91f6} YPF_NTOK=${YPF_NTOK:-64} YPF_DEPTHS="${YPF_DEPTHS:-64000 136000}"
export YPF_CORDER=${YPF_CORDER-BCCB} YPF_C8ORDER=${YPF_C8ORDER-BCCB} YPF_C16ORDER=${YPF_C16ORDER-BC} YPF_DECTH=${YPF_DECTH:-1.0}
export YPF_C8WIN=${YPF_C8WIN:-35} YPF_C8WARM=${YPF_C8WARM:-10} YPF_CARDUUID=${YPF_CARDUUID:-GPU-dbba5c851cfef7b3} YPF_P0TMO=${YPF_P0TMO:-600}
export YPF_DISKMIN_MB=${YPF_DISKMIN_MB:-11000} YPF_BUDGET_MIN=${YPF_BUDGET_MIN:-40}
export YPF_SIDECAR=${YPF_SIDECAR:-dspark-dlw-q8_0.rad} YPF_SIDESHA=${YPF_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/ypfx/dsm-ypfx.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT ypf: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B
