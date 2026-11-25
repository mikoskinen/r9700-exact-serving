#!/bin/bash
# 57-p4zyzzzzz-ds-f1-onem.sh (queue item, 10.11.2026, ds-main-port agent; gpu02, all 4 cards, TP4, every radiance container
# DOWN; ~55 min) -- the 1M-context target (user 10.11.: "1M context") on the main line: the t3 arm at MML 1048576 with the
# production MAXSEQS 8 (arm 225m, body dsm-onem.sh = dsm-serve.sh + the 225m arm): gate, modes, warm-up answers vs stock,
# slow-state probe, probe (1 user), mb c4 / c8 (decode at short context under MML 1M), the 28k copy (identity vs tuned3l
# and vs item a's dsm-225 at MML 524288), then onem.py: fresh prompts of ~524k and ~980k tokens (TTFT, prefill tok/s, 256
# greedy tokens at depth: ms a decode step) and a cache-hit turn on the deepest. History: 6.11. 1M probe on the 1.17 line
# (before the indexer fix and tuned3, MAXSEQS 4) ran to 700k tokens: 973 tok/s, 32.9 ms a step, hit turn 5.67 s; never on
# main / t3 / MAXSEQS 8. Measurement only (no code change). No cc_sim, no 2.9, no BetterBench.
# IMAGE radiance-local:ds41-main-e736c019-5c13e555 (item a's). DRYRUN=1 bash ... : prints, runs and writes nothing.
export XDSM_IMAGE=${XDSM_IMAGE:-radiance-local:ds41-main-e736c019-5c13e555}
export XDSM_ARMS=${XDSM_ARMS:-225m} XDSM_BB=${XDSM_BB:-none} XDSM_GAP=${XDSM_GAP:-} XDSM_PROF=${XDSM_PROF:-} XDSM_SHORT=${XDSM_SHORT:-}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-onem.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xdsm: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || [ -s $HOME/gpu/ds41/exl3-test/runs/xdsm/guard-dsm-225.txt ] || { echo "$(date '+%F %T') RESULT xdsm onem: VIRHE no dsm-225 reference arm -- not starting" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
