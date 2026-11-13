#!/bin/bash
# 57-p4zyzzzzz-ds-bzzz-ddiag.sh (queue item, 10.11.2026, ds-main-port agent; gpu02, all 4 cards, TP4, MML 524288; ~30 min) --
# the cheap DSpark drafter diagnostics after the audit (coordinator 10.11.; no 2.9 runs): NST 5 on the main line with the
# Markov walk's candidates 16 vs 32 (RADIANCE_DS41_DSPARK_CAND; the reference biases the whole vocabulary), each arm: per-
# position acceptance at 1 user for an essay and a code prompt (the -v proposal log replayed against the reply ids,
# perpos.py: accepted per draft position and the conditional acceptance), then mb c4 / c8 tokens a step and acceptance.
# -v floods the log, so timings are not read here (tokens a step is the metric). If top 32 raises tokens a step, it becomes
# the drafter default on the next image. IMAGE radiance-local:ds41-main-967d4eae-5c13e555-l = the main image with
# deepseek41.so of ds41-main 967d4ea (the walk's override + its stderr note; router fusion default off, set 0 anyway).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-bzzz-ddiag.sh : prints every command, runs and writes nothing.
export XDSM_IMAGE=${XDSM_IMAGE:-radiance-local:ds41-main-967d4eae-5c13e555-l}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-ddiag.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xdsg: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
