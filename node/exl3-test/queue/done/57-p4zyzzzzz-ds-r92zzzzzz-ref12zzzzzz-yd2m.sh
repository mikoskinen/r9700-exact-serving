#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yd2m.sh (queue item, 12.11.2026 by the A4 builder for ds-main-port; sorts after
# -r92zzzzzz-ref12zzzzzz-xpush and -r92zzzzzz-ref12zzzzz-res in C and en_US order; gpu02, all 4 cards, TP4, MML 524288, every radiance
# container DOWN; ~40 min estimate, budget YDM_BUDGET_MIN 50 (a section past it skipped, logged), NO evict) -- LEVER A4 SCREEN (+ lever
# A10 as an extra in-process arm P) on the ds41-a4 image (= int12 bf8f557 + A4: libr4d fn4 gg_gemv_m16, D2's decode-band GEMV forms for
# M <= 16 with the slab one ahead (XP), the grid fit, H1R2, MT up to 16; + the merge of ds41-pfx 8b52bd9: fn4 exl3_x3d_pf, A10's exl3r4d
# 90b1fef bb2d91f6; the same per-output operations: the same bytes). A10: its pfxprobe P0 on the same card (a failure drops P only),
# cold 136k B then P (answers compared), P blocks at both ends of the c8 order (P B C C B P), a1 / gs16 under P, and the verdict line
# "A10 joins the stack if 136k TTFT >= 1 % faster OR newcomer step >= 15 ms faster". P0 ON ONE CARD (by chip UUID) before the start: the GATE
# BENCH r4d_selftest --perfm16 (D2 vs the A4 candidates, M 5..16, two passes -> the table) then THE BYTE CHECK r4d_selftest --mrcheck16
# with that table (every A4 form + the switch vs the stock band, M 1..16, plain / pair / folds) -- a failure STOPS the item; an empty
# table or a c8 bench saving under YDM_GATE_US (250 us a step) = A4's GATE: no A4 screen (YDM_FORCE=1 screens anyway), and THE A10 PART
# RUNS ALONE by default ("A4 gate NOT passed (...); A10 part runs alone": B / P identity, staggered B, cold 136k, c8 P B B P, the A10
# line; YDM_A10_ALONE=0 skips it). ONE START, ref12's
# pins exactly (S5T, draft_sample 0, M1, draft_depth0 3, CAP48 via fn4, exl3_x3d 1, E2 1, GQ 3, KVR=1; MX / F4 / SP / ARC / CBH /
# XPREFETCH 0) + gg_gemv_m16 0 in the base pin + RADIANCE_DS41_GG_M16_FORMS=/stats/m16-forms.txt (the P0 table), MAXSEQS 16, the copy
# path (no ngshare, no ATTACH); arms in-process B = gg_gemv_m16 0, C = gg_gemv_m16 1 (read back). READINGS: identity (warm-up vs stock;
# a1 + gs16 under B, C, P: a1 C vs B, gs16 C vs B, gs16 vs a1 under B, C, P, a1 B / C vs ref12's a1, strict; a STAGGERED-arrival check
# under B (4 streams + a fresh 8k prefill joining each probe: its decode rides MIXED steps; ld-B vs a1-B strict, a difference = VIRHE,
# the item goes on) and under C at the end when the budget allows); c8 on the fixed harness
# (ncmb3 fixed prompts reset per block, 35 s windows) P B C C B P with the p12win wall split (decode ms a step and rows a step = the target;
# int12: 36.18 ms at 8.47 rows, agg 184.0) + the no-regression guard (C's c8 decode-step ms not above B's by > 1 %); the 1u essay
# confirm C then B (1 discarded warm-up + 3 reads each); c16 on the same harness (n 16) B then C. YDSUM: C vs B per reading + the P0
# gate lines. Tag "RESULT ydm ", out runs/xdsm/yd2m/x1, body queue-ar/dsmain/yd2m/dsm-yd2m.sh (a COPY of xpush/dsm-xpush.sh) +
# yd2m/ydsum.py; prof12/p12win.py read in place. YDM_SW=0 = a mock run (queue-ar/tools/mockrad; yd2m/mockrun-yd2m.sh).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yd2m.sh : prints every command; checks the image literals (int9 .. int12 + A4, the
# plugin md5), every helper (+ selftests), the sidecars, the yaml, the disk.
export YDM_IMAGE=${YDM_IMAGE:-radiance-local:ds41-a4-578d0077-90b1fef5-l} YDM_ITEM=${YDM_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yd2m}
export YDM_OUT=${YDM_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/yd2m/x1} YDM_SW=${YDM_SW:-1}
export YDM_MX=${YDM_MX:-0} YDM_F4=${YDM_F4:-0} YDM_F4G=${YDM_F4G:-0} YDM_MXCONFIRM=${YDM_MXCONFIRM:-0} YDM_GSCONC=${YDM_GSCONC:-16} YDM_MAXSEQS=${YDM_MAXSEQS:-16}
export YDM_PLUGMD5=${YDM_PLUGMD5:-bb2d91f6} YDM_A10=${YDM_A10:-1} YDM_PDEPTHS=${YDM_PDEPTHS-136000} YDM_P0_UUID=${YDM_P0_UUID:-GPU-dbba5c851cfef7b3}
export YDM_PERFMS=${YDM_PERFMS:-5,6,7,8,9,10,11,12,13,14,15,16} YDM_PERFIT=${YDM_PERFIT:-100} YDM_GATE_US=${YDM_GATE_US:-250} YDM_FORCE=${YDM_FORCE:-0}
export YDM_C8ORDER=${YDM_C8ORDER-PBCCBP} YDM_U1ORDER=${YDM_U1ORDER-CB} YDM_C16ORDER=${YDM_C16ORDER-BC} YDM_NREADS=${YDM_NREADS:-3} YDM_DECTH=${YDM_DECTH:-1.0}
export YDM_C8WIN=${YDM_C8WIN:-35} YDM_C8WARM=${YDM_C8WARM:-10} YDM_A10_ALONE=${YDM_A10_ALONE:-1} YDM_LDCONC=${YDM_LDCONC:-4} YDM_LDPF=${YDM_LDPF:-8192} YDM_STAGC=${YDM_STAGC:-1}
export YDM_DISKMIN_MB=${YDM_DISKMIN_MB:-11000} YDM_BUDGET_MIN=${YDM_BUDGET_MIN:-50}
export YDM_SIDECAR=${YDM_SIDECAR:-dspark-dlw-q8_0.rad} YDM_SIDESHA=${YDM_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/yd2m/dsm-yd2m.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT ydm: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B
