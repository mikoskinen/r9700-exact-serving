#!/bin/bash
# 57-p4zyzzzzz-ds-r92z-spread.sh (queue item, STAGED 12.11.2026 by the ds-main-port ARC builder for the coordinator; sorts after
# 57-p4zyzzzzz-ds-r92-ccfill and before 57-p4zyzzzzz-ds-r93-da-f-fidelity in C and en_US order; gpu02, all 4 cards, TP4, MML 524288,
# every radiance container DOWN; ~13 min estimate, budget XSPR_BUDGET_MIN 15 (a block past it skipped and logged), NO evict) --
# the B-vs-B SPREAD of c4 / c1 (and c8) mb blocks on the fixed-schedule harness, for TESTING.md's block rules, + the DA harness's
# live marker self-check; tag "RESULT xspr ", out runs/xdsm/spread, body queue-ar/dsmain/spread/dsm-spread.sh (from a COPY of
# arc/dsm-arc.sh e783f84c) + spread/damark.py (imports da/daheld.py, executes its marker lines). NOT QUEUED: the coordinator
# queues it.
# IMAGE radiance-local:ds41-int10-f6ef7cc3-5c13e555-l (0a60fffd3b3c). ONE START on the ARC screen's base: stack5's + MHC_FUSE=0
# (M1 declared), KVR / SP off; the pin S5T, draft_sample 0, SP 0 / 0, mhc_fuse 1, ix_lds 0, ix_ni 0, draft_depth0 3,
# sched_row_cap 48, ds41_arcoded 0 -- posted once, read back, held (no arms, no P0).
# BLOCKS (T 0, harness/ncmb2.py fixed schedule 1xN seed 1, window gens, dur 30 warm 10): order 4 1 8 4 1 8 4 1 = c4 x 3, c1 x 3,
# c8 x 2 (c4 gens 2 = 8 newcomer steps ~28 s, c1 gens 6 = 6 requests ~30 s, c8 gens 2 = 16 newcomer steps ~45 s); per block agg
# tok/s, per-stream p50, TTFT p50 / p90, ms / tok a decode step, acceptance, decode share, the mix line (c8mix.py). SUMMARY: per
# phase mean + max-min spread %, "VERDICT cN spread X % (one block per arm OK if <= ~1 %)". DA: "RESULT xspr da-marker: <marker>
# -> N of 192 docs" (no /completion; the corpus read only).
# XSPR_SW=0 = a mock run (queue-ar/tools/mockrad): no int9 / int9b / int10 switch posted, a "VIRHE XSPR_SW=0" marker.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r92z-spread.sh : prints every command; checks the image literals, every helper (+ selftests), the
# corpus, the sidecars, the yaml, the disk; no server, no GPU, writes nothing.
export XSPR_IMAGE=${XSPR_IMAGE:-radiance-local:ds41-int10-f6ef7cc3-5c13e555-l} XSPR_ITEM=${XSPR_ITEM:-57-p4zyzzzzz-ds-r92z-spread}
export XSPR_OUT=${XSPR_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/spread} XSPR_SW=${XSPR_SW:-1}
export XSPR_ORDER="${XSPR_ORDER-4 1 8 4 1 8 4 1}" XSPR_DUR=${XSPR_DUR:-30} XSPR_WARM=${XSPR_WARM:-10}
export XSPR_GENS4=${XSPR_GENS4:-2} XSPR_GENS1=${XSPR_GENS1:-6} XSPR_GENS8=${XSPR_GENS8:-2}
export XSPR_MHC=${XSPR_MHC:-1} XSPR_D0=${XSPR_D0:-3} XSPR_CAP=${XSPR_CAP:-48} XSPR_SPN=${XSPR_SPN:-0} XSPR_SPK=${XSPR_SPK:-0}
export XSPR_DA=${XSPR_DA:-1} XSPR_CORPUS=${XSPR_CORPUS:-$HOME/gpu/ds41/exl3-test/runs/xdsm/da/stats/corpus/corpus-gen.jsonl}
export XSPR_DISKMIN_MB=${XSPR_DISKMIN_MB:-11000} XSPR_BUDGET_MIN=${XSPR_BUDGET_MIN:-15}
export XSPR_SIDECAR=${XSPR_SIDECAR:-dspark-dlw-q8_0.rad} XSPR_SIDESHA=${XSPR_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/spread/dsm-spread.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xspr: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }   # 12.11. compose guard (ghost 622b5b7a9e64)
bash $B
