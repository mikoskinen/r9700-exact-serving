#!/bin/bash
# 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-h-onear.sh (queue item, 11.11.2026, ds-main-port BI agent, lever BI-d; sorts right after
# 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-g-longid, before 57-p4zyzzzzz-ds-iejzzzszzzzzz-fn-bitrace (C and en_US order); gpu02, all 4 cards,
# TP4, MML 524288, every radiance container DOWN; ~120 min) -- tag "RESULT xbil ", out runs/xdsm/bionear, body
# queue-ar/dsmain/dsm-bilid7.sh 92a3997e (unchanged; the long-prompt starts are not run here). Image
# radiance-local:ds41-bi-7cfb77b1-5c13e555-l (= 2d9a1dc + 7cfb77b: ds41_bi 5 = 3 + the MoE's all-reduce whole, 6 = 4 + the same).
# WHY (coordinator 11.11., BI-d): bi-e-cheap's profile put collective +5.84 ms of bi 1's +17.15 ms device a mixed step (all_reduce
# +12.9 launches, ~0.46 ms each; ARGATHER lost: gather_ar +1.24). 5 / 6 segment the MoE compute but gather without reducing and
# issue ONE all-reduce over the step's rows (exact wire: each element sums 0 + r0 + r1 + r2 + r3 in fp32 at any message size).
# At NST 0 the decode segment is <= 8 rows, so 4 issues what 3 does and 6 what 5 does (one value list serves both NST).
# FOUR STARTS, stack2 env and pins, RADIANCE_DS41_BI=0 declared:
#   p0 / p5  NST 0 / NST 5 stack2, --profile-ops (PROFILED, not a speed): one mb pf window at ds41_bi 0, 4, 6 -> ds_opprof.py ->
#            per value vs 0: device ms, lane waits, rank-0 launches added a MIXED step, by class and op (6 vs 4 = the one AR);
#   n0 / n5  unprofiled: smoke; identity for 6 (a1 vs bz d0, ld vs a1 with c4 + a fresh 32k prefill, c4 run-to-run; a2
#            control; at NST 0 this is value 5's code); mb c4 / c8 / pf blocks 0 4 6 6 4 0 (60 s a phase): 4 and 6 vs 0 in %.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-h-onear.sh : prints every command; checks the image literals, every helper, sidecar, yaml.
export XBIL_IMAGE=${XBIL_IMAGE:-radiance-local:ds41-bi-7cfb77b1-5c13e555-l} XBIL_ITEM=${XBIL_ITEM:-57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-h-onear}
export XBIL_OUT=${XBIL_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/bionear} XBIL_STARTS=${XBIL_STARTS:-p0 p5 n0 n5}
export XBIL_EXTRAENV=${XBIL_EXTRAENV:-RADIANCE_DS41_BI=0} XBIL_SW=${XBIL_SW:-ds41_bi} XBIL_SWVALS=${XBIL_SWVALS:-0 4 6}
export XBIL_IDVALS=${XBIL_IDVALS:-6} XBIL_PROFVALS=${XBIL_PROFVALS:-0 4 6}
export XBIL_C4RR=${XBIL_C4RR:-1} XBIL_TRACE=${XBIL_TRACE:-0} XBIL_CTRL=${XBIL_CTRL:-1}
export XBIL_MB=${XBIL_MB:-c4,c8,pf} XBIL_MBORDER=${XBIL_MBORDER:-0 4 6 6 4 0} XBIL_MBDUR=${XBIL_MBDUR:-60} XBIL_MBWARM=${XBIL_MBWARM:-15}
export XBIL_OFFSET=${XBIL_OFFSET:-0.5} XBIL_PFTOK=${XBIL_PFTOK:-32768}
export XBIL_LITS="${XBIL_LITS:-A|RADIANCE_DS41_BI
A|lever BI declared
A|steps with one MoE all-reduce
B|ds41_bi}"
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-bilid7.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xbil: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
