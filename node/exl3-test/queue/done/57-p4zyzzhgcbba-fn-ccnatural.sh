#!/bin/bash
# 118-fn-ccnatural.sh (queue item, 9.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~4 h for the
# first four arms, ~5 h with e4) -- the cc_sim pairs again in NATURAL generation mode.
# WHY: every cc_sim pair so far ran --gen-mode forced (ignore_eos + lognormal lengths). cc_sim.py's header: on Radiance the
# tokens after the model's natural stop are junk that collapses MTP draft acceptance (smoke 2.11.: 0.02-0.10 vs 0.4-0.75)
# and halves per-stream decode -- so the forced-mode acceptances (0.26-0.47) and the gains that rest on acceptance (the
# adaptive window, main's chained-launch loss 0.259 vs 0.351, EXL3's 0.269) may be artefacts or exaggerated. Natural mode:
# no ignore_eos, max_tokens = --gen-max, the model stops at its own tool call; greedy (--temperature 0) and
# --trace-seeded, so the lengths differ only where the outputs differ and exact arms still pair request by request.
# ARMS (in order; a "-n" arm runs that line under a tag of its own, so the forced-mode files stay):
#   k4n-n     1.15.1 (stampcards c0cbe5a) + --prefix-resume nearest                      (the reference)
#   k4nA-n    + --draft-window adaptive
#   rmnIc0-n  main 451b136 with chain_pays 0 (main-451b136-c0) + --prefill-interleave 0 + nearest
#   rmnI-n    main 451b136 as built (chained launches) + --prefill-interleave 0 + nearest
#   e4-n      EXL3 4.05 (TP4 + EP4, the -fnx3 image) + nearest -- last, "if time" (touch runs/sweep.stop to end before it)
# PER ARM (93j-fn-tp4ab.sh with X93_CC_ONLY=1): gate, slow-state probe (mb c8 45 s; a slow process restarted up to twice),
# cc_sim u16 --turns 15 --warmup-turns 3 --effort xhigh,medium --gen-mode natural --seed 1 --cpt-fixed --temperature 0
# --trace-seeded (--duration 3000 as the safety cap), the engine's speculation counters around it (msnap: drafts verified
# / accepted a step, acceptance, verify rows a step, passes played). No mb / prefill / copy / BetterBench (have them).
# PAIRS (cc_sim.py --pair + warm-up answers): k4nA-n, rmnIc0-n, rmnI-n, e4-n each vs k4n-n; rmnI-n vs rmnIc0-n.
# The upstream issue draft on main's chained launches waits for these numbers.
# DRYRUN=1 bash 118-fn-ccnatural.sh : prints every command, runs and writes nothing.
H=$(cd "$(dirname "$0")" && pwd)
export ARMS=${X118_ARMS:-"k4n-n k4nA-n rmnIc0-n rmnI-n e4-n"}
export X93_CC_ONLY=1 X93_REF=${X93_REF:-k4n-n} X93_REF_RERUN=0 X93_HANG428=0 X93_LACC=0 X93_PROF16=0
export X93_CCARGS=${X93_CCARGS:-"--duration 3000 --turns 15 --warmup-turns 3 --effort xhigh,medium --gen-mode natural --seed 1 --cpt-fixed --temperature 0 --trace-seeded"}
export X93_EXTRA_PAIRS=${X93_EXTRA_PAIRS:-"rmnIc0-n:rmnI-n"} X93_ITEM=${X93_ITEM:-118-fn-ccnatural}
for f in "$H/93j-fn-tp4ab.sh" "$HOME/gpu/ds41/exl3-test/queue-ar/93j-fn-tp4ab.sh"; do [ -e "$f" ] && exec bash "$f"; done
echo "$(date '+%F %T') RESULT fn4 118-fn-ccnatural: VIRHE 93j-fn-tp4ab.sh not found beside $H" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
exit 1
