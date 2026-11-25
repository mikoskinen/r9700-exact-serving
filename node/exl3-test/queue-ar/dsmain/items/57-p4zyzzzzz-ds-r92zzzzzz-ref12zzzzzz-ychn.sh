#!/bin/bash
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-ychn.sh (queue item, STAGED 12.11.2026 ~20:00 UTC+2 by the A3 / A5 builder for ds-main-port -- NOT
# QUEUED by the builder; sorts after -r92zzzzzz-ref12zzzzzz-xpush and before -yd2m / -yll / -ypfx / -zepb in C and en_US order; gpu02, all 4
# cards, TP4, MML 524288, every radiance container DOWN; ~13 min estimate, budget YLL_BUDGET_MIN 20 (a section past it skipped, logged),
# NO evict) -- LEVER A5 ONLY (chained launches, core fn4 chain_ds41) on ds41-ll (image 8cb19c36b0f0): the CHN-only form of the yll body,
# for when A3 (LL) is parked after its P0 (12.11. 19:38: LL 2.6 .. 29 x slower a call, bytes exact). No A3 P0, no LL arm, no A10.
# ONE START, ref12's pins exactly (S5T, draft_sample 0, M1, draft_depth0 3, CAP48 via fn4, exl3_x3d 1, E2 1, GQ 3, KVR=1; MX / F4 / SP /
# ARC / CBH / XPREFETCH 0) + RADIANCE_AR_LL=0 RADIANCE_CHAIN_DS41=0 (capable, off), MAXSEQS 16, the copy path (no ngshare, no ATTACH); arms
# in-process B = int12 (chain_ds41 0), D = chain_ds41 1 (read back). READINGS: identity (a1 + gs16 under B and D, strict: a1 vs ref12's
# a1, gs16 vs a1; D's gs16 window must move GET "chain" (chained launches > 0) else VIRHE and D's c8 blocks SKIPPED); c8 ncmb3 fixed
# prompts 35 s B D D B with the P12SPLIT wall split and the chain counters a block (the wire proof). YLLSUM: D vs B. Tag "RESULT yll ",
# out runs/xdsm/yll/c1, body queue-ar/dsmain/yll/dsm-yll.sh + yll/yllsum.py (the yll item's own, under these knobs).
# YLL_SW=0 = a mock run (yll/mockrun-ychn.sh). DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-ychn.sh : prints every command.
export YLL_IMAGE=${YLL_IMAGE:-radiance-local:ds41-ll-1b68d342-90b1fef5-l} YLL_ITEM=${YLL_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-ychn}
export YLL_OUT=${YLL_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/yll/c1} YLL_SW=${YLL_SW:-1}
export YLL_MX=${YLL_MX:-0} YLL_F4=${YLL_F4:-0} YLL_F4G=${YLL_F4G:-0} YLL_MXCONFIRM=${YLL_MXCONFIRM:-0} YLL_GSCONC=${YLL_GSCONC:-16} YLL_MAXSEQS=${YLL_MAXSEQS:-16}
export YLL_PLUGMD5=${YLL_PLUGMD5:-bb2d91f6} YLL_NTOK=${YLL_NTOK:-64} YLL_DEPTHS="${YLL_DEPTHS-}" YLL_LLV=${YLL_LLV:-1} YLL_A10=${YLL_A10:-0} YLL_LLP0=${YLL_LLP0:-0}
export YLL_CORDER=${YLL_CORDER-} YLL_C8ORDER=${YLL_C8ORDER-BDDB} YLL_C16ORDER=${YLL_C16ORDER-} YLL_U1ORDER=${YLL_U1ORDER-} YLL_DECTH=${YLL_DECTH:-1.0}
export YLL_IDARMS=${YLL_IDARMS-BD} YLL_STAGARMS=${YLL_STAGARMS-} YLL_LDPF=${YLL_LDPF:-8192} YLL_LDCONC=${YLL_LDCONC:-4}
export YLL_C8WIN=${YLL_C8WIN:-35} YLL_C8WARM=${YLL_C8WARM:-10} YLL_NREADS=${YLL_NREADS:-3} YLL_CARDUUID=${YLL_CARDUUID:-GPU-dbba5c851cfef7b3} YLL_P0TMO=${YLL_P0TMO:-600}
export YLL_LLP0TMO=${YLL_LLP0TMO:-900} YLL_LLP0MIN=${YLL_LLP0MIN:-49}
export YLL_DISKMIN_MB=${YLL_DISKMIN_MB:-11000} YLL_BUDGET_MIN=${YLL_BUDGET_MIN:-20}
export YLL_SIDECAR=${YLL_SIDECAR:-dspark-dlw-q8_0.rad} YLL_SIDESHA=${YLL_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/yll/dsm-yll.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT yll: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard
bash $B
