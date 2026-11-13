#!/bin/bash
# 109b-fn-tp4main2.sh (queue item, 9.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~1 h 50 min, two
# arms) -- 109's rmn (main 451b136) won decode (mb c4 / c8 / c16 559.7 / 748.7 / 954.4 tok/s, +6.5 / +1.4 / +0.1 % vs k4n;
# BetterBench 286.8; copy IDENTICAL; #428 no hang) but starved prefill: mb pf 2756.5 tok/s TTFT 11.64 s (k4n 6102.8 /
# 5.26 s) and cc_sim u16 TTFT p50 / p95 / max 1.10 / 4.88 / 21.4 s (k4n 0.56 / 1.84 / 6.97), total gen 203.7 vs 242.3,
# 18485 decode steps vs ~12-13k. Cause: main's served default --prefill-interleave auto:1 (a583046: up to 1.0 x a chunk
# step's time of decode-only steps after every unfinished prefill chunk while others decode). 1.15.1 resolves 0 on the k4n
# line (its start log lists no interleave; 0 is its default, "every step may carry a chunk"), so:
#   rmnI   main 451b136 + --prefill-interleave 0 (X93_INTERLEAVE), the k4n line otherwise
#   rmnIA  rmnI + --draft-window adaptive (the candidate line)
# = 93f-fn-tp4ab.sh beside this file with ARMS="rmnI rmnIA" X93_REF=k4n (106's rows) X93_EXTRA_PAIRS="rmn:rmnI k4nA:rmnIA
# rmnI:rmnIA" (109's rmn: the interleave's own effect; 107's k4nA: the adaptive window on 1.15.1 vs main; rmnI -> rmnIA:
# the window on main) X93_HANG428=0 X93_ITEM=109b-fn-tp4main2. Phases as 109 without the #428 check: gate, slow-state probe,
# produp + probe, mb c4 / c8 / c16 (+ engine stats), prefill 16k + fresh 64k / 128k, the 33k copy test, mb pf, BetterBench,
# cc_sim u16 (+ engine stats); each arm logs the scheduler's resolved --prefill-interleave from its start log.
# DRYRUN=1 bash 109b-fn-tp4main2.sh : prints every command, runs and writes nothing.
H=$(cd "$(dirname "$0")" && pwd)
export ARMS=${ARMS:-"rmnI rmnIA"} X93_REF=${X93_REF:-k4n} X93_ITEM=${X93_ITEM:-109b-fn-tp4main2} X93_HANG428=${X93_HANG428:-0}
export X93_EXTRA_PAIRS=${X93_EXTRA_PAIRS:-"rmn:rmnI k4nA:rmnIA rmnI:rmnIA"}
for f in "$H/93f-fn-tp4ab.sh" "$HOME/gpu/ds41/exl3-test/queue-ar/93f-fn-tp4ab.sh"; do [ -e "$f" ] && exec bash "$f"; done
echo "$(date '+%F %T') RESULT fn4 109b-fn-tp4main2: VIRHE 93f-fn-tp4ab.sh not found beside $H" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
exit 1
