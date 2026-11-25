#!/bin/bash
# 99-fn-tp4m4.sh (queue item, 9.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~1 h 50 min for the two
# arms) -- 93's phases for the pipelined prefill step and for it on PR #396, each paired against the k4s arm of the night's
# 93 run instead of running k4s again (as 97 does for d4t):
#   m4a   TP4 all_vram MNBT 4096 --prefill-span 2 (step_pipelined: 2 pieces of 2048, all-reduces on the second lane)
#   p4m4  PR #396 (radiance-local:1.15.1-pr396-0ae1545-stampcards) + the same MNBT / span: the two measured TP4 levers
# Phases per arm: gate, produp pair + probe, mb c4 / c8 / c16, prefill 16k + fresh 64k / 128k, mb pf, BetterBench (its own
# prefill sweep 2k-64k), cc_sim u16 fixed-work trace-seeded, card-0 slow-state stamps (c8), temperature guard; then
# cmparm / cc_sim.py --pair / warm-up answers against runs/fn4/fn4-k4s-* and the newest ccsim-*-fn4-k4s-u16.jsonl.
# WHY (95, all three m arms in the same slow state, mb c8 620-631): prefill 16k / 64k / 128k m2 serial 7828 / 7746 / 7606
# tok/s, m4 8496 / 8508 / 8391 (+8.5 / +9.8 / +10.3 %), m8 8355 / 8275 / 8167 (+6.7..+7.4 %); pf beside 4 decoders 5769 /
# 5714 / 5787 (no change); 29k-token copy TTFT 3.74 / 3.42 / 3.51 s; copy exact on all three. p4a (92): c4 +4.2 %, c8 +0.9
# %, c16 +0.8 %, 1 user -1..-6 % -- decode; m4a is prefill; the two touch different steps, so they should add.
# Exactness: m4a = stock per the code's own claim (a pipelined piece computes a span-1 chunk's bytes; 95 checks the 29k
# greedy copy output m4 vs m2 byte for byte); p4m4 inherits PR #396's non-byte-identical hc_read forms (bf16 WMMA order).
# = 93b-fn-tp4ab.sh (slow-state aware: state probe + restart, like-state pairs, a fast k4sf when the stored k4s ran slow) beside this file with ARMS="m4a p4m4" X93_REF=k4s X93_ITEM=99-fn-tp4m4 (all overridable).
# DRYRUN=1 bash 99-fn-tp4m4.sh : prints every command, runs and writes nothing.
H=$(cd "$(dirname "$0")" && pwd)
export ARMS=${ARMS:-"m4a p4m4"} X93_REF=${X93_REF:-k4s} X93_ITEM=${X93_ITEM:-99-fn-tp4m4}
for f in "$H/93b-fn-tp4ab.sh" "$HOME/gpu/ds41/exl3-test/queue-ar/93b-fn-tp4ab.sh"; do [ -e "$f" ] && exec bash "$f"; done   # 93b: slow-state aware
echo "$(date '+%F %T') RESULT fn4 99-fn-tp4m4: VIRHE 93b-fn-tp4ab.sh not found beside $H" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
exit 1
