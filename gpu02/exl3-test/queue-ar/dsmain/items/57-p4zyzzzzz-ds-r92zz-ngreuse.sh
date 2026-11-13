#!/bin/bash
# 57-p4zyzzzzz-ds-r92zz-ngreuse.sh (queue item, STAGED 12.11.2026 by the ds-main-port ARC builder for the coordinator; sorts after
# 57-p4zyzzzzz-ds-r92z-spread and before 57-p4zyzzzzz-ds-r93-* in C and en_US order; gpu02, all 4 cards, TP4, MML 524288, every
# radiance container DOWN; ~21 min estimate, budget XNGR_BUDGET_MIN 22 (a reading past it skipped and logged), NO evict) -- does the
# n-gram shared-tmpfs path (tools/ngramshare, --ngram-placement ram=/ngram) give the copy path's SPEED? If within the spread, every
# DS item may reuse the tables (-83 s a start). Tag "RESULT xngr ", out runs/xdsm/ngreuse, body queue-ar/dsmain/ngreuse/
# dsm-ngreuse.sh (from a COPY of spread/dsm-spread.sh b894cbb7). NOT QUEUED: the coordinator queues it.
# IMAGE radiance-local:ds41-int10-f6ef7cc3-5c13e555-l (0a60fffd3b3c). THREE STARTS on the spread base (stack5's + MHC_FUSE=0, KVR /
# SP off; the pin S5T, draft_sample 0, SP 0, mhc_fuse 1, IX off, draft_depth0 3, sched_row_cap 48, ds41_arcoded 0):
#   A the copy path (ds41-stack2-tp4.yaml, NGRAM=ram): c8 x 2 (time window 10..45 s = ~35 s), c1 (gens 6), cold 64k (64 tok);
#   B the ngshare opt-in as its README says (. ngshare.sh: ds41-ngshare-tp4.yaml, NGRAM=ram=/ngram, keeper) = the FIRST FILL:
#     c8 x 1, c1, cold 64k;
#   C ngshare REUSE ("reused (probes match)"): start-to-healthy seconds + c1.
#   Per start the healthy seconds and the n-gram placement lines (copied / filled / reused, s). The end: ngram-shm-cleanup.sh
#   (also on an early exit after B started). Summary: B vs A % (c8 / c1 / 64k TTFT) against A's c8 pair spread, start seconds
#   A / B / C, "VERDICT reuse within spread: yes / no".
# XNGR_SW=0 = a mock run (queue-ar/tools/mockrad): no int9 / int9b / int10 switch posted, a "VIRHE XNGR_SW=0" marker.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r92zz-ngreuse.sh : prints every command; checks the image literals, every helper (+ selftests), the
# ngramshare files and both yamls, the sidecars, the disk; no server, no GPU, writes nothing.
export XNGR_IMAGE=${XNGR_IMAGE:-radiance-local:ds41-int10-f6ef7cc3-5c13e555-l} XNGR_ITEM=${XNGR_ITEM:-57-p4zyzzzzz-ds-r92zz-ngreuse}
export XNGR_OUT=${XNGR_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/ngreuse} XNGR_SW=${XNGR_SW:-1}
export XNGR_C8DUR=${XNGR_C8DUR:-45} XNGR_C8WARM=${XNGR_C8WARM:-10} XNGR_DUR=${XNGR_DUR:-30} XNGR_WARM=${XNGR_WARM:-10} XNGR_GENS1=${XNGR_GENS1:-6}
export XNGR_DEPTH=${XNGR_DEPTH:-64000} XNGR_NTOK=${XNGR_NTOK:-64} XNGR_C=${XNGR_C:-1}
export XNGR_MHC=${XNGR_MHC:-1} XNGR_D0=${XNGR_D0:-3} XNGR_CAP=${XNGR_CAP:-48} XNGR_SPN=${XNGR_SPN:-0} XNGR_SPK=${XNGR_SPK:-0}
export XNGR_DISKMIN_MB=${XNGR_DISKMIN_MB:-11000} XNGR_BUDGET_MIN=${XNGR_BUDGET_MIN:-22}
export XNGR_SIDECAR=${XNGR_SIDECAR:-dspark-dlw-q8_0.rad} XNGR_SIDESHA=${XNGR_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/ngreuse/dsm-ngreuse.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xngr: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard (ghost 622b5b7a9e64)
bash $B
