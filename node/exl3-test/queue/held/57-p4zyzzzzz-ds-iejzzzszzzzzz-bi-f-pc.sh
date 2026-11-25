#!/bin/bash
# 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-f-pc.sh (queue item, 11.11.2026, ds-main-port, lever BI-c on top of arm P; sorts right after
# 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-e-cheap, before 57-p4zyzzzzz-ds-iejzzzszzzzzz-fn-bitrace (C and en_US order); gpu02, all 4 cards,
# TP4, MML 524288, every radiance container DOWN; ~60 min) -- tag "RESULT xbil ", out runs/xdsm/bipc, body
# queue-ar/dsmain/dsm-bilid6.sh (= dsm-bilid5.sh b2ffbcf5 + the profiled NST 5 start p5). Image radiance-local:ds41-bi-2d9a1dc9-5c13e555-l
# (ds41_bi 2 = arm P, the MoE decode segment in passes of <= 16 rows; ds41_bi 4 = the arm-C cheap issue on top of P).
# WHY: bi-d-parm NST 5: P = full BI with drafting and no reference change (20 / 20 everywhere); cost P vs B c1 +0.9 %, c4 +2.9 %,
# c8 +4.8 %, pf -2.0 %; vs bi 0 cross-run c8 about -1.3 %. Coordinator: the cheap work builds on P; A/B 0 vs 2 vs 4 in process.
# TWO STARTS, NST 5 only (at NST 0 the decode segment is <= 8 rows, so 2 issues what 1 does and 4 what 3 does; bi-e-cheap's
# p0 / n0 give NST 0 -- no baseline re-run); stack2 env and pins, RADIANCE_DS41_BI=0 declared:
#   p5  NST 5 stack2 A, --profile-ops (PROFILED, not a speed; rule-w replaced by the profile sync): one mb pf window at ds41_bi
#       0, 2, 4 -> ds_opprof.py -> per value vs 0: device ms, lane waits, rank-0 launches added a MIXED step, by class and op;
#   n5  unprofiled: smoke under 4; identity for 4 (a1 vs bz d0, ld vs a1 with c4 + a fresh 32k prefill, c4 run-to-run; a2
#       control; the identity of 2 is bi-d-parm's); mb c4 / c8 / pf blocks 0 2 4 4 2 0 (60 s a phase): 2 and 4 vs 0 in %.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-f-pc.sh : prints every command; checks the image literals, every helper, sidecar, yaml.
export XBIL_IMAGE=${XBIL_IMAGE:-radiance-local:ds41-bi-2d9a1dc9-5c13e555-l} XBIL_ITEM=${XBIL_ITEM:-57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-f-pc}
export XBIL_OUT=${XBIL_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/bipc} XBIL_STARTS=${XBIL_STARTS:-p5 n5}
export XBIL_EXTRAENV=${XBIL_EXTRAENV:-RADIANCE_DS41_BI=0} XBIL_SW=${XBIL_SW:-ds41_bi} XBIL_SWVALS=${XBIL_SWVALS:-0 2 4}
export XBIL_IDVALS=${XBIL_IDVALS:-4} XBIL_PROFVALS=${XBIL_PROFVALS:-0 2 4}
export XBIL_C4RR=${XBIL_C4RR:-1} XBIL_TRACE=${XBIL_TRACE:-0} XBIL_CTRL=${XBIL_CTRL:-1}
export XBIL_MB=${XBIL_MB:-c4,c8,pf} XBIL_MBORDER=${XBIL_MBORDER:-0 2 4 4 2 0} XBIL_MBDUR=${XBIL_MBDUR:-60} XBIL_MBWARM=${XBIL_MBWARM:-15}
export XBIL_OFFSET=${XBIL_OFFSET:-0.5} XBIL_PFTOK=${XBIL_PFTOK:-32768}
export XBIL_LITS="${XBIL_LITS:-A|RADIANCE_DS41_BI
A|lever BI declared
A|lever BI stats
B|ds41_bi}"
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-bilid6.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xbil: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
