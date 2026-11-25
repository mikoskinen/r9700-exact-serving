#!/bin/bash
# 116-fn-tp4exl3.sh (queue item, 9.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~2 h, two arms) --
# can Flash-Next EXL3 4.05 (~/models/flashnext-exl3/qwen3.8-flash-next-exl3.rad, 112.25 GiB with the n-gram table; KL
# 0.0467 vs production iq4r's 0.0537) on FOUR cards give better quality AND equal or better speed than the TP4 iq4r
# candidate k4nA (1.15.1 + nearest + adaptive: c4 / c8 / c16 534.6 / 742.6 / 957.7 tok/s, cc gen/agent 81.4)?
# IMAGE radiance-local:ds41-exl3-117-390664b3-fa0e9580-tuned3-hd4a11cab-fnx3 (X93_IMAGE_X3) = -fnx2 (ds41-fnx 996e4e4:
# EXL3 experts whole per rank, expert parallel) + the EP4 fix: at four ranks layers 46 / 47 leave a rank 127 EXL3 experts
# (its 128 minus one plain bf16 expert: L46 keeps 290 / 392, L47 122 / 143 / 399 / 445 bf16), which -fnx2's declare refuses
# ("they come in pairs on a rank") although the EXL3 ops take ceil / floor parity tables (exl3_class_entries); -fnx3 serves
# an odd count (a 4-fake-device host start of -fnx2 stops exactly there). Experts at EP4: 128 x 2.37 MiB x 48 layers =
# 14.2 GiB a card (iq4r TP4: a quarter of every expert, 14.5 GiB), so all_vram at MML 262144 fits as k4n does.
# EP: a step's expert time is the slowest rank's (routing imbalance; TP slices every expert evenly), and the ds41-fork
# binary (1.17.0 base, tuned3) stamps card 0 only -- the per-card EP balance shows in the c16 gap / split lines of card 0
# and the all-reduce waits, not per card.
#   e4    EXL3 TP4 + EP4, k4n's settings (all_vram, MML 262144, MAXSEQS 16, MNBT 2048, MTP 3, --prefix-resume nearest)
#   e4A   e4 + --draft-window adaptive
# = 93h-fn-tp4ab.sh beside this file with ARMS="e4 e4A" X93_REF=k4nA (107's rows) X93_EXTRA_PAIRS="k4n:e4 k4n:e4A e4:e4A"
# X93_PROF16=1 (the c16 window's card-0 split and gaps) X93_STATE_TRIES=0 (the slow-state thresholds are iq4r's: the state
# is logged, no restart) FN_GATE_MS=40 (the 21 ms 1-user gate is iq4r TP4's) X93_ITEM=116-fn-tp4exl3. Phases: gate, state
# probe, produp + probe, mb c4 / c8 / c16 (+ engine stats), prefill 16k + fresh 64k / 128k, the 33k copy test
# (information: another format), mb pf, BetterBench, cc_sim u16 (+ engine stats); pairs vs k4nA / k4n; 210 W, host tuning
# and the window poller as they are.
# DRYRUN=1 bash 116-fn-tp4exl3.sh : prints every command, runs and writes nothing.
H=$(cd "$(dirname "$0")" && pwd)
export ARMS=${ARMS:-"e4 e4A"} X93_REF=${X93_REF:-k4nA} X93_ITEM=${X93_ITEM:-116-fn-tp4exl3} X93_HANG428=0
export X93_EXTRA_PAIRS=${X93_EXTRA_PAIRS:-"k4n:e4 k4n:e4A e4:e4A"} X93_PROF16=${X93_PROF16:-1} X93_STATE_TRIES=${X93_STATE_TRIES:-0} FN_GATE_MS=${FN_GATE_MS:-40}
for f in "$H/93h-fn-tp4ab.sh" "$HOME/gpu/ds41/exl3-test/queue-ar/93h-fn-tp4ab.sh"; do [ -e "$f" ] && exec bash "$f"; done
echo "$(date '+%F %T') RESULT fn4 116-fn-tp4exl3: VIRHE 93h-fn-tp4ab.sh not found beside $H" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
exit 1
