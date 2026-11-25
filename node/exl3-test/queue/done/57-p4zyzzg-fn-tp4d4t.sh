#!/bin/bash
# 97-fn-tp4d4t.sh (queue item, 9.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~50 min) -- 93's
# phases for the d4t arm alone (TP4 expert_tiered --host-pool-mib 16384: 4 queues, prefetch, heat engine), paired against
# the k4s arm of the night's 93 run (57-p4zyy...) instead of running k4s again: mb c4 / c8 / c16, BetterBench, cc_sim u16
# fixed-work trace-seeded, card-0 slow-state stamps, temperature guard; then cmparm, cc_sim.py --pair and the warm-up
# answers against runs/fn4/fn4-k4s-* and the newest runs/ccsim-*-fn4-k4s-u16.jsonl (logged with its time; a missing
# reference run is a VIRHE line, the d4t numbers still land).
# WHY: 91's d4t ran c8 / c16 within -0.6 / -1.0 % of all_vram (c4 -5.4 %) with KV 26.95 GiB a card (14.3 full 262k sessions,
# 89 % of the 16 x 262k worst case) against 12.77 GiB (6.5, 41 %); mb's short contexts leave the KV mostly empty and lent
# to the experts, so the long-context cc load (prompts p50 ~70k, max ~103k, 16 users) is the test of the capacity lever.
# = 93b-fn-tp4ab.sh (slow-state aware: state probe + restart, like-state pairs, a fast k4sf when the stored k4s ran slow) beside this file with ARMS=d4t X93_REF=k4s X93_ITEM=97-fn-tp4d4t (both overridable).
# DRYRUN=1 bash 97-fn-tp4d4t.sh : prints every command, runs and writes nothing.
H=$(cd "$(dirname "$0")" && pwd)
L97=$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
# 93's own pick may already have been d4t (then 93 ran k4s + d4t in one window): nothing to add
if [ "${X97_FORCE:-0}" != 1 ] && [ "${DRYRUN:-0}" != 1 ] && grep -q "RESULT fn4 93-fn-tp4ab pick: WINNER d4t" $L97 2>/dev/null; then
  echo "$(date '+%F %T') RESULT fn4 97-fn-tp4d4t: skipped -- 93's pick was d4t, so 93 already paired it against k4s (X97_FORCE=1 runs it anyway)" | tee -a $L97
  exit 0
fi
export ARMS=${ARMS:-d4t} X93_REF=${X93_REF:-k4s} X93_ITEM=${X93_ITEM:-97-fn-tp4d4t}
for f in "$H/93b-fn-tp4ab.sh" "$HOME/gpu/ds41/exl3-test/queue-ar/93b-fn-tp4ab.sh"; do [ -e "$f" ] && exec bash "$f"; done   # 93b: slow-state aware
echo "$(date '+%F %T') RESULT fn4 97-fn-tp4d4t: VIRHE 93b-fn-tp4ab.sh not found beside $H" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
exit 1
