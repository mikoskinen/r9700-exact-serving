#!/bin/bash
# 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-gyz-chunk.sh (queue item, 11.11.2026, ds-main-port BI agent, lever BI class (ii); sorts right
# after 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-gy-c8id, before 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-gz-dual (C and en_US order); gpu02, all 4
# cards, TP4, MML 524288, every radiance container DOWN; ~100 min) -- tag "RESULT xbil ", out runs/xdsm/bichunk, body
# queue-ar/dsmain/dsm-bilid8.sh (= dsm-bilid7.sh + the class (ii) starts; drivers longid2.py, bitrace2.py). Image IMG below:
# ds41-bi 3d19a2c (= 5bb7ce0 + RADIANCE_SCHED_BI_CHUNK 1 / 2 / 3 / 4 + the tracer's PMIN / LMAX) with fn4-dsdual 8d738e2's plugins.
# WHY (coordinator 11.11., bi-g-longid lg-nst0 ds41_bi 2): the 136k deep answer differs loaded vs alone (char 60) while its
# chunks are 2048 / 4096 alone and 2048 / 1024 / odd pieces (1824 / 224, 1920 / 128, 2016 / 32, 1936 / 112) loaded; the
# copy's (all 2048 loaded) answer is identical. Policy (core/sched): a step that is not one lone prefill is planned at
# ChunkGeometry::shared_tok (geometry.cpp prefill_shared_tok: 4160 - 2048 = 2112) instead of the span's 4096
# (scheduler.cpp step(): the span block), the chunk is cut by the budget left after the decode rows and the other prefill
# (plan_chunk + busy_cap: a share for the second prefill, the "span - floor" halving) and then aligned to the 32-row quantum.
# SIX STARTS, NST 0 stack2 N, ds41_bi 2 (as bi-g-longid), RADIANCE_DS41_BI=0 declared, RADIANCE_LOG_STEPS=1:
#   lt  today's scheduler + lever BI's tracer over every row of the deep prompt at layer 0 (+ the head): deep alone, deep
#       loaded -> bitrace2.py (background, beside the next starts): the FIRST op whose row differs and both steps' shapes;
#   lx  today's scheduler: copy + deep alone / loaded, the c4 streams' TPOT while the probe runs, the second prefill's TTFT;
#   la  RADIANCE_SCHED_BI_CHUNK=1 (option A: alone's boundaries under load, the step budget --max-num-batched-tokens);
#   lb  RADIANCE_SCHED_BI_CHUNK=2 (option B: the 2048 grid, alone too);
#   lc  RADIANCE_SCHED_BI_CHUNK=3 / ld =4 (option D: a lone prefill takes alone's piece; beside other work, in the shared budget
#       as today, a chunk ends at alone's piece end or on a grid G inside it -- lc G 2048 (the interval), ld G 1024 --
#       leaving >= 128 rows on each side): each identity loaded vs alone, alone vs today's references (copy bz d0, deep
#       dsm-225: B changes them only if 2048- vs 4096-row chunks do), alone TTFT, TPOT, the second prefill's TTFT.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-gyz-chunk.sh : prints every command; checks the image literals, every helper
#   (longid2.py and bitrace2.py selftests included), the references, sidecar, yaml.
IMG=${XBIL_IMAGE:-radiance-local:ds41-bi-3d19a2c0-8d738e2-l}
export XBIL_IMAGE=$IMG XBIL_ITEM=${XBIL_ITEM:-57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-gyz-chunk}
export XBIL_OUT=${XBIL_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/bichunk} XBIL_STARTS=${XBIL_STARTS:-lt lx la lb lc ld}
export XBIL_EXTRAENV=${XBIL_EXTRAENV:-RADIANCE_DS41_BI=0} XBIL_SW=${XBIL_SW:-ds41_bi} XBIL_SWVALS=${XBIL_SWVALS:-2}
export XBIL_IDVALS=${XBIL_IDVALS:-2} XBIL_LGVALS=${XBIL_LGVALS:-2} XBIL_PROFVALS=${XBIL_PROFVALS:-}
export XBIL_C4RR=0 XBIL_TRACE=0 XBIL_CTRL=0 XBIL_MB=""
export XBIL_CPMAX=${XBIL_CPMAX:-140000} XBIL_CLMAX=${XBIL_CLMAX:-1}
export XBIL_OFFSET=${XBIL_OFFSET:-0.5} XBIL_PFTOK=${XBIL_PFTOK:-32768}
export XBIL_LITS="${XBIL_LITS:-A|RADIANCE_DS41_BI
A|lever BI declared
A|RADIANCE_DS41_BITRACE_LMAX
A|layers below
B|RADIANCE_SCHED_BI_CHUNK
B|RADIANCE_SCHED_BI_GRID
B|lever BI chunking
B|RADIANCE_LOG_STEPS
B|ds41_bi}"
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-bilid8.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xbil: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
