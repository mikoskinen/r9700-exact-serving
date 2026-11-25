#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yll.sh (queue item, STAGED 12.11.2026 by the A3 / A5 builder for ds-main-port -- NOT QUEUED by the
# builder; sorts after -r92zzzzzz-ref12zzzzz-res and -r92zzzzzz-ref12zzzzzz-xpush, before -ypfx / -zepb, in C and en_US order; gpu02, all
# 4 cards, TP4, MML 524288, every radiance container DOWN; ~28 min estimate, budget YLL_BUDGET_MIN 30 (a section past it skipped, logged),
# NO evict) -- LEVERS A3 (LL all-reduce at decode sizes, libr4d fn4 ar_ll) + A5 (chained launches, core fn4 chain_ds41) + A10 (X3D in the
# M > 16 band, plugin fn4 exl3_x3d_pf; rides here instead of its own ypfx item) on ds41-ll (radiance = int12 bf8f557 + A3 / A5 + the merge
# of ds41-pfx 8b52bd9; exl3r4d = ds41-x3d-pf 90b1fef). Every new switch 0 = int12. P0 BEFORE THE START: A3 ON FOUR CARDS (r4d_selftest
# --case ar_ll: ar_ll 0 vs 1 byte for byte vs the fp32 fold, the gather's oracle, 2 x 1000 stress launches, us a call at T 1 / 8 / 16 / 48
# -- a failure STOPS the item) + A10 ON ONE CARD (chip GPU-dbba5c851cfef7b3, ypfx/bin/pfxprobe -- a failure drops only arm P). ONE START,
# ref12's pins exactly (S5T, draft_sample 0, M1, draft_depth0 3, CAP48 via fn4, exl3_x3d 1, E2 1, GQ 3, KVR=1; MX / F4 / SP / ARC / CBH /
# XPREFETCH 0) + RADIANCE_AR_LL=0 RADIANCE_CHAIN_DS41=0 (capable, off), MAXSEQS 16, the copy path (no ngshare, no ATTACH); arms in-process
# B = int12 (ar_ll 0, chain_ds41 0, exl3_x3d_pf 0), C = ar_ll 1, D = chain_ds41 1 (read back; TRIMMED 12.11.: no E, and A10 NOT read
# here -- the A4 item yd2m carries it: no pfxprobe P0, no arm P, no cold; the image contains A10 at exl3_x3d_pf 0). READINGS: identity
# (a1 + gs16 under B C D, strict: a1 vs ref12's a1, gs16 vs a1); STAGGERED arrivals (stack4/loadid.py as zepb: 4 decode streams + a
# fresh 8k prefill joining each probe, strict vs the arm's a1) under C, under B at the end within the budget; c8 ncmb3 fixed prompts 35 s
# B C D D C B with the P12SPLIT wall split and the arll / chain counters a block (the wire proof); the 1u essay confirm B / C (1
# discarded + 3); c16 on the same harness, B then C. YLLSUM: every arm vs B (A10's guard / rule lines only with YLL_A10=1), A10's
# decode guard and its rule (136k TTFT >= 1 % faster OR newcomer step >= 15 ms faster: PASS / NOT). Tag "RESULT yll ", out
# runs/xdsm/yll/l1, body queue-ar/dsmain/yll/dsm-yll.sh (a COPY of ypfx/dsm-ypfx.sh) + yll/yllsum.py; prof12/p12win.py and ypfx/bin read
# in place. YLL_SW=0 = a mock run (queue-ar/tools/mockrad; yll/mockrun-yll.sh). DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yll.sh
# : prints every command; checks the image literals (int9 .. int12 + A10's + A3 / A5's, the plugin md5), the P0 binaries, every helper
# (+ selftests), the sidecars, the yaml, the disk.
export YLL_IMAGE=${YLL_IMAGE:-radiance-local:ds41-ll-1b68d342-90b1fef5-l} YLL_ITEM=${YLL_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yll}
export YLL_OUT=${YLL_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/yll/l1} YLL_SW=${YLL_SW:-1}
export YLL_MX=${YLL_MX:-0} YLL_F4=${YLL_F4:-0} YLL_F4G=${YLL_F4G:-0} YLL_MXCONFIRM=${YLL_MXCONFIRM:-0} YLL_GSCONC=${YLL_GSCONC:-16} YLL_MAXSEQS=${YLL_MAXSEQS:-16}
export YLL_PLUGMD5=${YLL_PLUGMD5:-bb2d91f6} YLL_NTOK=${YLL_NTOK:-64} YLL_DEPTHS="${YLL_DEPTHS-}" YLL_LLV=${YLL_LLV:-1} YLL_A10=${YLL_A10:-0}
export YLL_CORDER=${YLL_CORDER-} YLL_C8ORDER=${YLL_C8ORDER-BCDDCB} YLL_C16ORDER=${YLL_C16ORDER-BC} YLL_U1ORDER=${YLL_U1ORDER-BC} YLL_DECTH=${YLL_DECTH:-1.0}
export YLL_IDARMS=${YLL_IDARMS-BCD} YLL_STAGARMS=${YLL_STAGARMS-CB} YLL_LDPF=${YLL_LDPF:-8192} YLL_LDCONC=${YLL_LDCONC:-4}
export YLL_C8WIN=${YLL_C8WIN:-35} YLL_C8WARM=${YLL_C8WARM:-10} YLL_NREADS=${YLL_NREADS:-3} YLL_CARDUUID=${YLL_CARDUUID:-GPU-dbba5c851cfef7b3} YLL_P0TMO=${YLL_P0TMO:-600}
export YLL_LLP0TMO=${YLL_LLP0TMO:-900} YLL_LLP0MIN=${YLL_LLP0MIN:-49}
export YLL_DISKMIN_MB=${YLL_DISKMIN_MB:-11000} YLL_BUDGET_MIN=${YLL_BUDGET_MIN:-30}
export YLL_SIDECAR=${YLL_SIDECAR:-dspark-dlw-q8_0.rad} YLL_SIDESHA=${YLL_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/yll/dsm-yll.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT yll: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B
