#!/bin/bash
# 113c-fn-tp4c0A.sh (queue item, 9.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~55 min, one arm)
# -- the new candidate line. 113's rmnIc0 (main 451b136 with chain_pays 0, --prefill-interleave 0, nearest) brought the
# cc_sim u16 draft acceptance back to 0.351 (1.15.1 k4n 0.354; main chained 0.259) and total gen to 291.0 tok/s (main
# chained 223.4, k4n 242.3), gen/agent 77.8, TTFT p50 0.50 s, 94.3 % of passes played; mb c4 / c8 / c16 561.9 / 745.2 /
# 954.7 (chained 562.2 / 754.1 / 958.6). e2e9416's chained launches with the MTP drafter lower acceptance on the
# warm-prefix agent load (cold prompts: 0.60-0.63 either way).
#   rmnIc0A  radiance-local:main-451b136-c0 + --prefill-interleave 0 + --prefix-resume nearest + --draft-window adaptive
# = 93i-fn-tp4ab.sh beside this file with ARMS="rmnIc0A" X93_REF=k4n (106's rows) X93_EXTRA_PAIRS="k4nA:rmnIc0A
# rmnIA:rmnIc0A rmnIc0:rmnIc0A" (107's k4nA, 109b's rmnIA, 113's rmnIc0) X93_ITEM=113c-fn-tp4c0A. Phases as 109b's rmnIA:
# gate, slow-state probe, produp + probe, mb c4 / c8 / c16 (+ engine stats), prefill 16k + fresh 64k / 128k, the 33k copy
# test, mb pf, BetterBench, cc_sim u16 (+ engine stats); the resolved --prefill-interleave logged.
# DRYRUN=1 bash 113c-fn-tp4c0A.sh : prints every command, runs and writes nothing.
H=$(cd "$(dirname "$0")" && pwd)
export ARMS=${ARMS:-"rmnIc0A"} X93_REF=${X93_REF:-k4n} X93_ITEM=${X93_ITEM:-113c-fn-tp4c0A} X93_HANG428=0
export X93_EXTRA_PAIRS=${X93_EXTRA_PAIRS:-"k4nA:rmnIc0A rmnIA:rmnIc0A rmnIc0:rmnIc0A"}
for f in "$H/93i-fn-tp4ab.sh" "$HOME/gpu/ds41/exl3-test/queue-ar/93i-fn-tp4ab.sh"; do [ -e "$f" ] && exec bash "$f"; done
echo "$(date '+%F %T') RESULT fn4 113c-fn-tp4c0A: VIRHE 93i-fn-tp4ab.sh not found beside $H" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
exit 1
