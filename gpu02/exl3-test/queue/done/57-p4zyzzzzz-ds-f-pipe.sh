#!/bin/bash
# 57-p4zyzzzzz-ds-f-pipe.sh (queue item, 10.11.2026, ds41-pipe agent; gpu02, all 4 cards, TP4, MML 524288, every radiance
# container DOWN; ~45 min) -- DeepSeek-V4.1 Flash EXL3 2.25 bpw: the pipelined prefill (RADIANCE_DS41_PIPE=1: a lone
# prefill step wider than the span's shared budget in two pieces, the attention and MoE all-reduces on the second lane
# beside the other piece's work; Flash-Next's step_pipelined ported to deepseek41) vs the serial step, ONE image, two
# starts: A = RADIANCE_DS41_PIPE unset (= ds41-main 967d4ea byte for byte), B = 1. t3 env, 225 serve line. No cc_sim.
# IMAGE radiance-local:ds41-pipe-fd215624-5c13e555-l (XPIPE_IMAGE) = radiance-local:ds41-main-967d4eae-5c13e555-l + the
#   deepseek41.so of ds41-pipe fd215624 (workstation build/pipe/{build-in-env,assemble}.sh): c490dc6 the pipelined step +
#   fd21562 the whole-step routing histogram for the heat engine (B places experts from the same evidence as A).
# PHASES (queue-ar/dsmain/dsm-pipe.sh), per arm, decode before prefill (TESTING 9.11.): gate, modes (the pipelining note
#   in B only), warm-up answers vs stock, mb c4, mb c8, THEN fresh prefill 16k / 64k / 136k (a thermal line each), mb pf,
#   i1 copy (~31k cold), deep answer (136k); per-phase /stats; then B vs A (% differences) and identity B vs A byte for
#   byte (warm-up, copy, deep: the lever is exact by construction -- a difference is a VIRHE line).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-f-pipe.sh : prints every command, runs and writes nothing.
export XPIPE_IMAGE=${XPIPE_IMAGE:-radiance-local:ds41-pipe-fd215624-5c13e555-l}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-pipe.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xpipe: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
