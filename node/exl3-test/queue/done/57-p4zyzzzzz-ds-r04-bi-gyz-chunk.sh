#!/bin/bash
# 57-p4zyzzzzz-ds-r04-bi-gyz-chunk.sh (queue item, 11.11.2026, ds-main-port BI agent, lever BI class (ii); the queue slot r04 the
# coordinator gave bi-gyz-chunk d7bc671a, swapped in place for this two-start version; gpu02, all 4 cards, TP4, MML 524288,
# every radiance container DOWN; ~55 min) -- tag "RESULT xbil ", out runs/xdsm/bichunk, body queue-ar/dsmain/dsm-bilid9.sh
# (= dsm-bilid8.sh + the in-process option start lm; drivers longid2.py, bitrace2.py). Image IMG below: ds41-bi 49989ab
# (= 3d19a2c + the core fn4 switch sched_bi_chunk 0 .. 4, read at every step planned) with fn4-dsdual 8d738e2's plugins.
# WHY (coordinator 11.11., bi-g-longid): the 136k deep answer differs loaded vs alone at NST 0 bi 2 / bi 4 and NST 5 bi 4, where the
# loaded chunks include 16 / 32 / 112-row pieces; identical where they are 1024 / 2048 only. Policy (core/sched): a step that
# is not one lone prefill is planned at shared_tok 2112 (geometry.cpp:37, scheduler.cpp step()), busy_cap's share and its
# "span - floor" halving, the 32-row quantum cut. The options (sched_bi_chunk): 0 today; 1 A (alone's boundaries, budget
# --max-num-batched-tokens); 2 B (the 2048 grid, alone too); 3 D (alone's pieces cut on the 2048 grid beside other work,
# >= 128 rows a side); 4 D on the 1024 grid (one grid step left to a prefill behind).
# TWO STARTS, NST 0 stack2 N, ds41_bi 2 in the pin (XBIL_PINX), RADIANCE_DS41_BI=0 declared, RADIANCE_LOG_STEPS=1:
#   lt  sched_bi_chunk 0 + lever BI's tracer over every row of the deep prompt at layer 0 (+ the head): deep alone, deep
#       loaded -> bitrace2.py (background, beside lm): the FIRST op whose row differs and both steps' shapes; skipped with a
#       VIRHE under 5 GB free; the raw trace files (runs/xdsm/bichunk/stats) removed once read. Its own start: the tracer
#       turns recorded passes off, which changes the timing.
#   lm  ONE start, sched_bi_chunk pinned 0 1 2 3 4 then 0 again (the closing control: answers byte for byte, TTFT drift) in
#       process: per option the pin + epoch_bump and one 4-token request (the change lands at a step release, after that
#       step was planned), the applied value checked, then copy + deep alone / under load (mb c4 + one fresh ~32k prefill at
#       +0.5 s), a prefix-cache reset before each: identity loaded vs alone, alone vs today's references (copy bz d0, deep
#       dsm-225), chunk lists, TTFT, the c4 streams' TPOT p50 / p90 / p99 / max while the probe runs, the second prefill's TTFT.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-r04-bi-gyz-chunk.sh : prints every command; checks the image literals, every helper (longid2.py and
#   bitrace2.py selftests included), the references, sidecar, yaml.
IMG=${XBIL_IMAGE:-radiance-local:ds41-bi-49989aba-8d738e2-l}
export XBIL_IMAGE=$IMG XBIL_ITEM=${XBIL_ITEM:-57-p4zyzzzzz-ds-r04-bi-gyz-chunk}
export XBIL_OUT=${XBIL_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/bichunk} XBIL_STARTS=${XBIL_STARTS:-lt lm}
export XBIL_EXTRAENV=${XBIL_EXTRAENV:-RADIANCE_DS41_BI=0} XBIL_PINX=${XBIL_PINX:-'{"ds41_bi": 2}'}
export XBIL_SW=${XBIL_SW:-sched_bi_chunk} XBIL_SWVALS=${XBIL_SWVALS:-0 1 2 3 4} XBIL_IDVALS=${XBIL_IDVALS:-0}
export XBIL_LGVALS=${XBIL_LGVALS:-0} XBIL_MVALS=${XBIL_MVALS:-0 1 2 3 4 0} XBIL_PROFVALS=${XBIL_PROFVALS:-}
export XBIL_C4RR=0 XBIL_TRACE=0 XBIL_CTRL=0 XBIL_MB=""
export XBIL_CPMAX=${XBIL_CPMAX:-140000} XBIL_CLMAX=${XBIL_CLMAX:-1}
export XBIL_OFFSET=${XBIL_OFFSET:-0.5} XBIL_PFTOK=${XBIL_PFTOK:-32768}
export XBIL_LITS="${XBIL_LITS:-A|RADIANCE_DS41_BI
A|lever BI declared
A|RADIANCE_DS41_BITRACE_LMAX
A|layers below
B|sched_bi_chunk
B|RADIANCE_SCHED_BI_CHUNK
B|RADIANCE_SCHED_BI_GRID
B|RADIANCE_LOG_STEPS
B|ds41_bi}"
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-bilid9.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xbil: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
