#!/bin/bash
# 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-d-parm.sh (queue item, 11.11.2026, ds41-bi agent for ds-main-port, lever BI ARM P; sorts right after
# 57-p4zyzzzzz-ds-iejzzzszzzzz-bi-c-fixab (running when this was queued), before 57-p4zyzzzzz-ds-iejzzzt-tree6 (C and en_US
# order); gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~60 min) -- LEVER BI: arm P vs arm B at NST 5
# (tag "RESULT xbil ", out runs/xdsm/biparm, body queue-ar/dsmain/dsm-bilid4.sh = dsm-bilid.sh + the engine's BI stats line a mb
# block). Image radiance-local:ds41-bi-38c1331b-5c13e555-l (ds41-bi 38c1331: ds41_bi 2 = 1 + the routed MoE over the decode
# segment in passes of <= 16 rows, the EXL3 decode band). WHY: trace3 NST 5 found 3 first divergences in PURE decode steps
# (c4 NST 5 verify, ~24 rows: the decode segment crosses the EXL3 16-token band; alone <= 6 rows stay in it), which ds41_bi 1
# cannot reach; P keeps the alone output byte-identical (class (i)). The alternative, fn4's exl3_bi, is class (ii).
# ONE start, NST 5 (stack2 A) with RADIANCE_DS41_BI=0 (declared, starts 0): smoke under 2; arms A ds41_bi 0, B 1, P 2, each:
# a1 alone, ld (c4 + a fresh ~32k prefill), c4a / c4b (run-to-run); a2 control; compares per arm: a1 vs bz d0 (must be 0 / 20),
# ld vs a1, c4b vs c4a, c4a vs a1, a1 vs a1-s0; then mb c1, c4, c8, pf blocks B P P B (60 s a phase) -> P vs B in %, and after
# each block the engine's cumulative "lever BI stats" (MoE passes a layer per segmented step). NST 0 is not run: its decode
# segment is <= 8 rows (MAXSEQS 8 x 1), so P issues exactly B there (cost and bytes = bi-c-fixab's B arm).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-d-parm.sh : prints every command; checks the image (its literals incl. the arm-P
# ones), every helper, the sidecar and the yaml; no server, no GPU, writes nothing.
export XBIL_IMAGE=${XBIL_IMAGE:-radiance-local:ds41-bi-38c1331b-5c13e555-l} XBIL_ITEM=${XBIL_ITEM:-57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-d-parm}
export XBIL_OUT=${XBIL_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/biparm} XBIL_STARTS=${XBIL_STARTS:-n5}
export XBIL_EXTRAENV=${XBIL_EXTRAENV:-RADIANCE_DS41_BI=0} XBIL_SW=${XBIL_SW:-ds41_bi} XBIL_SWVALS=${XBIL_SWVALS:-0 1 2}
export XBIL_C4RR=${XBIL_C4RR:-1} XBIL_TRACE=${XBIL_TRACE:-0} XBIL_CTRL=${XBIL_CTRL:-1}
export XBIL_MB=${XBIL_MB:-c1,c4,c8,pf} XBIL_MBORDER=${XBIL_MBORDER:-1 2 2 1} XBIL_MBDUR=${XBIL_MBDUR:-60} XBIL_MBWARM=${XBIL_MBWARM:-15}
export XBIL_OFFSET=${XBIL_OFFSET:-0.5} XBIL_PFTOK=${XBIL_PFTOK:-32768}
export XBIL_LITS="${XBIL_LITS:-A|RADIANCE_DS41_BI
A|lever BI declared
A|is not 0 / off, 1 / on or 2: lever BI is not declared
A|lever BI stats
B|ds41_bi}"
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-bilid4.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xbil: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
