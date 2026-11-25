#!/bin/bash
# 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-gy-c8id.sh (queue item, 11.11.2026, ds-main-port, lever BI: identity under the HEAVIEST load;
# sorts right after 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-g-stack3, before 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-gz-dual (C and en_US order);
# gpu02, all 4 cards, TP4, MML 524288, every radiance container DOWN; ~35 min) -- tag "RESULT xbil ", out runs/xdsm/bic8, body
# queue-ar/dsmain/dsm-bilid7.sh (unchanged). Image radiance-local:ds41-bi-2d9a1dc9-5c13e555-l (ds41_bi 2 = arm P, 4 = C on P).
# WHY: every loaded identity so far ran at c4 (+ the probe): at NST 5 that is at most ~5 sequences x (1 + drafts) rows, so the
# segment splits above 32 rows (fused norms) and above 48 rows (gemm_nt_ggml D2 / WMMA) were not reached on the card; NST 5 c8
# steps reach 54 decode rows (the BI agent's stats). Here the 20 greedy prompts each run under 6 closed-loop streams + the probe
# + one fresh ~32k prefill = 8 sequences (MAXSEQS 8, nothing queued): decode segments up to 8 x (1 + window) rows.
# ONE START, NST 5 stack2 A (NST 0 has at most 8 decode rows), RADIANCE_DS41_BI=0 declared: smoke under 4; per value 2 and 4:
# a1 (alone vs bz d0), ld (XBIL_CONC=6 + the probe + the prefill), then a2 control; no mb, no c4 run-to-run (both done at c4).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-gy-c8id.sh : prints every command; checks the image literals, every helper, sidecar, yaml.
export XBIL_IMAGE=${XBIL_IMAGE:-radiance-local:ds41-bi-2d9a1dc9-5c13e555-l} XBIL_ITEM=${XBIL_ITEM:-57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-gy-c8id}
export XBIL_OUT=${XBIL_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/bic8} XBIL_STARTS=${XBIL_STARTS:-n5} XBIL_CONC=${XBIL_CONC:-6}
export XBIL_EXTRAENV=${XBIL_EXTRAENV:-RADIANCE_DS41_BI=0} XBIL_SW=${XBIL_SW:-ds41_bi} XBIL_SWVALS=${XBIL_SWVALS:-2 4}
export XBIL_IDVALS=${XBIL_IDVALS:-2 4} XBIL_PROFVALS=${XBIL_PROFVALS:-} XBIL_LGVALS=${XBIL_LGVALS:-}
export XBIL_C4RR=${XBIL_C4RR:-0} XBIL_TRACE=${XBIL_TRACE:-0} XBIL_CTRL=${XBIL_CTRL:-1}
export XBIL_MB=${XBIL_MB:-} XBIL_MBORDER=${XBIL_MBORDER:-2 4} XBIL_MBDUR=${XBIL_MBDUR:-60} XBIL_MBWARM=${XBIL_MBWARM:-15}
export XBIL_OFFSET=${XBIL_OFFSET:-0.5} XBIL_PFTOK=${XBIL_PFTOK:-32768}
export XBIL_LITS="${XBIL_LITS:-A|RADIANCE_DS41_BI
A|lever BI declared
A|lever BI stats
B|RADIANCE_LOG_STEPS
B|ds41_bi}"
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-bilid7.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xbil: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
