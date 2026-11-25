#!/bin/bash
# 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-e-cheap.sh (queue item, 11.11.2026, ds41-bi agent for ds-main-port, lever BI-c: the COST of the
# segmented step; sorts right after 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-d-parm, before 57-p4zyzzzzz-ds-iejzzzszzzzzz-fn-bitrace (C and en_US order); gpu02, all 4
# cards, TP4, MML 524288, every radiance container DOWN; ~90 min) -- tag "RESULT xbil ", out runs/xdsm/bicheap, body
# queue-ar/dsmain/dsm-bilid5.sh. Image radiance-local:ds41-bi-2d9a1dc9-5c13e555-l (ds41-bi 2d9a1dc9: ds41_bi 3 = 1 + the cheap issue
# C: the query / key / index chains op by op across the segments, each weight read twice back to back, and the never-flipping
# per-row ops once over the step). WHY: bi-c-fixab NST 0: ds41_bi 1 = full BI (20 / 20 everywhere) at c1 -0.08 %, c4 -0.95 %,
# c8 -4.59 %, mb pf ms a mixed step 117.75 -> 131.15 (+13.4 ms, +11.4 %); target <= 2 % a mixed step, c8 within 1 %.
# THREE STARTS, stack2's env and pins, RADIANCE_DS41_BI=0 declared:
#   p0  NST 0, --profile-ops (PROFILED, not a speed; rule-w replaced by the profile sync, z0prof2): one mb pf window at ds41_bi
#       0, 1, 3 -> ds_opprof.py per window -> per value vs 0: device ms, lane waits and rank-0 launches ADDED A MIXED STEP, by class
#       (experts, dense_gemm, collective, attention, indexer, mhc, ...) and by op: where the +13.4 ms goes;
#   n0, n5  unprofiled: smoke under 3; identity for the new arm (ds41_bi 3: a1 vs bz d0 = the alone line, ld vs a1 with c4 + a
#       fresh 32k prefill, c4 run-to-run; a2 control); then mb c4 / c8 / pf blocks 0 1 3 3 1 0 (60 s a phase): 3 and 1 vs 0
#       in % (c1 is not run: no arm touches a 1-row step). bi 0 / bi 1 identity is bi-c-fixab's (no baseline re-run).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-e-cheap.sh : prints every command; checks the image (its literals incl. the arm-C
# ones), every helper (files incl. ds_opprof.py, parse, selftests, every function via type -t), the sidecar and the yaml.
export XBIL_IMAGE=${XBIL_IMAGE:-radiance-local:ds41-bi-2d9a1dc9-5c13e555-l} XBIL_ITEM=${XBIL_ITEM:-57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-e-cheap}
export XBIL_OUT=${XBIL_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/bicheap} XBIL_STARTS=${XBIL_STARTS:-p0 n0 n5}
export XBIL_EXTRAENV=${XBIL_EXTRAENV:-RADIANCE_DS41_BI=0} XBIL_SW=${XBIL_SW:-ds41_bi} XBIL_SWVALS=${XBIL_SWVALS:-0 1 3}
export XBIL_IDVALS=${XBIL_IDVALS:-3} XBIL_PROFVALS=${XBIL_PROFVALS:-0 1 3}
export XBIL_C4RR=${XBIL_C4RR:-1} XBIL_TRACE=${XBIL_TRACE:-0} XBIL_CTRL=${XBIL_CTRL:-1}
export XBIL_MB=${XBIL_MB:-c4,c8,pf} XBIL_MBORDER=${XBIL_MBORDER:-0 1 3 3 1 0} XBIL_MBDUR=${XBIL_MBDUR:-60} XBIL_MBWARM=${XBIL_MBWARM:-15}
export XBIL_OFFSET=${XBIL_OFFSET:-0.5} XBIL_PFTOK=${XBIL_PFTOK:-32768}
export XBIL_LITS="${XBIL_LITS:-A|RADIANCE_DS41_BI
A|lever BI declared
A|lever BI stats
B|ds41_bi}"
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-bilid5.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xbil: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
