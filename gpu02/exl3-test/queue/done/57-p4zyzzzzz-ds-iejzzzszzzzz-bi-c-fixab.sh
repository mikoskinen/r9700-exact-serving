#!/bin/bash
# 57-p4zyzzzzzzzzzz-bi-c-fixab.sh (queue item, 11.11.2026, ds41-bi agent for ds-main-port, lever BI; sorts after
# 57-p4zyzzzzzzzzzz-bi-b-trace; gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~100 min) -- LEVER BI FIX A/B
# = THE ACCEPTANCE TEST (tag "RESULT xbil ", out runs/xdsm/bifix, body queue-ar/dsmain/dsm-bilid.sh). Image radiance-local:ds41-bi-d7ea2abd-5c13e555-l
# (ds41-bi: ds41-int3 1116248 + the tracer + ds41_bi). Two starts (stack2's env, serve line and pins: n0 NST 0, n5 NST 5),
# each with RADIANCE_DS41_BI=0 (declared, starts 0) and in process:
#   ds41_bi 0: a1 (greedy set alone), ld (each prompt under mb c4 + a fresh ~32k prefill), c4a / c4b (the set at conc 4 twice);
#   ds41_bi 1: the same four; then a2 (alone again at 0).
# Compares: ld vs a1 and c4b vs c4a at each value (TARGET under ds41_bi 1: 20 / 20 IDENTICAL), c4a vs a1, a1-s1 vs a1-s0 and
# each arm's a1 vs bz d0 (the alone line must stay byte-identical: ds41_bi is CLASS (i), alone-identical by construction, the only
# lever in this item), a2 vs a1 (control). Then the SPEED A/B in process (coordinator: speed per arm): mb c1, c4, c8 (decode, no
# prefill: one segment, step() unchanged under 1) and pf (4 streams + a fresh 32k prefill: the mixed steps BI segments), blocks
# 0 1 1 0, 60 s each phase; summary per phase B vs A in %.
# Residual diffs under 1 are expected from the EXL3 routed MoE (fn4's: the decode / prefill forms at 16 tokens and gate_up's
# split-K at 2 tokens) -- this item names them by count, the tracer item (bi-b-trace) by op. NO cc_sim, no 2.9.
# DRYRUN=1 bash 57-p4zyzzzzzzzzzz-bi-c-fixab.sh : prints every command; checks the image (its literals incl. RADIANCE_DS41_BI),
# every helper (files, parse, selftests, functions), the sidecar and the yaml; no server, no GPU, writes nothing.
export XBIL_IMAGE=${XBIL_IMAGE:-radiance-local:ds41-bi-d7ea2abd-5c13e555-l} XBIL_ITEM=${XBIL_ITEM:-57-p4zyzzzzzzzzzz-bi-c-fixab}
export XBIL_OUT=${XBIL_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/bifix} XBIL_STARTS=${XBIL_STARTS:-n0 n5}
export XBIL_EXTRAENV=${XBIL_EXTRAENV:-RADIANCE_DS41_BI=0} XBIL_SW=${XBIL_SW:-ds41_bi} XBIL_SWVALS=${XBIL_SWVALS:-0 1}
export XBIL_C4RR=${XBIL_C4RR:-1} XBIL_TRACE=${XBIL_TRACE:-0} XBIL_CTRL=${XBIL_CTRL:-1}
export XBIL_MB=${XBIL_MB:-c1,c4,c8,pf} XBIL_MBORDER=${XBIL_MBORDER:-0 1 1 0} XBIL_MBDUR=${XBIL_MBDUR:-60} XBIL_MBWARM=${XBIL_MBWARM:-15}
export XBIL_OFFSET=${XBIL_OFFSET:-0.5} XBIL_PFTOK=${XBIL_PFTOK:-32768}
export XBIL_LITS="${XBIL_LITS:-A|RADIANCE_DS41_BI
A|lever BI declared
B|ds41_bi}"
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-bilid.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xbil: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
