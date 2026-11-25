#!/bin/bash
# 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-g-longid.sh (queue item, 11.11.2026, ds-main-port, lever BI class (ii) check; sorts right after
# 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-f-pc, before 57-p4zyzzzzz-ds-iejzzzszzzzzz-fn-bitrace (C and en_US order); gpu02, all 4 cards,
# TP4, MML 524288, every radiance container DOWN; ~50 min) -- tag "RESULT xbil ", out runs/xdsm/bilong, body
# queue-ar/dsmain/dsm-bilid7.sh (= dsm-bilid6.sh + the long-prompt starts l0 / l5, driver longid.py). Image
# radiance-local:ds41-bi-2d9a1dc9-5c13e555-l (ds41_bi 2 = arm P, 4 = arm C on P).
# WHY (coordinator 11.11.): every loadid probe is a 10-16-token prompt, so long-prompt chunk boundaries alone vs under load are
# untested. Before DS is called batch-invariant: the 28k copy (~31k-token prompt, 900 tok) and the 136k deep answer (deepq.py's
# fixed 136216-token prompt, 512 tok), each ALONE and UNDER LOAD (mb c4 streams + one fresh ~32k prefill at +0.5 s), at
# ds41_bi 2 and 4, NST 0 (l0) and NST 5 (l5); byte compare loaded vs alone; alone vs today's references (bz d0 copy, dsm-225
# deep); the probe's prefill chunk starts / sizes from RADIANCE_LOG_STEPS=1 (set ONLY in these starts: a log line a step) for
# each run, alone vs loaded. A difference = the class (ii) case: count + boundaries go to the coordinator (a reference change
# for it is the user's decision, not before).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-g-longid.sh : prints every command; checks the image literals (+ RADIANCE_LOG_STEPS),
# every helper (longid.py parse + selftest, the two references), sidecar, yaml.
export XBIL_IMAGE=${XBIL_IMAGE:-radiance-local:ds41-bi-2d9a1dc9-5c13e555-l} XBIL_ITEM=${XBIL_ITEM:-57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-g-longid}
export XBIL_OUT=${XBIL_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/bilong} XBIL_STARTS=${XBIL_STARTS:-l0 l5}
export XBIL_EXTRAENV=${XBIL_EXTRAENV:-RADIANCE_DS41_BI=0} XBIL_SW=${XBIL_SW:-ds41_bi} XBIL_SWVALS=${XBIL_SWVALS:-2 4}
export XBIL_IDVALS=${XBIL_IDVALS:-2 4} XBIL_PROFVALS=${XBIL_PROFVALS:-} XBIL_LGVALS=${XBIL_LGVALS:-2 4}
export XBIL_C4RR=${XBIL_C4RR:-1} XBIL_TRACE=${XBIL_TRACE:-0} XBIL_CTRL=${XBIL_CTRL:-1}
export XBIL_MB=${XBIL_MB:-c4,c8,pf} XBIL_MBORDER=${XBIL_MBORDER:-2 4} XBIL_MBDUR=${XBIL_MBDUR:-60} XBIL_MBWARM=${XBIL_MBWARM:-15}
export XBIL_OFFSET=${XBIL_OFFSET:-0.5} XBIL_PFTOK=${XBIL_PFTOK:-32768}
export XBIL_LITS="${XBIL_LITS:-A|RADIANCE_DS41_BI
A|lever BI declared
A|lever BI stats
B|RADIANCE_LOG_STEPS
B|ds41_bi}"
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-bilid7.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xbil: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
