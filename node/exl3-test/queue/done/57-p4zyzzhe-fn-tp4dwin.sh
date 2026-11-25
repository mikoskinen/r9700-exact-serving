#!/bin/bash
# 107-fn-tp4dwin.sh (queue item, 9.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~50 min, one arm)
# -- the adaptive draft window on the k4n line, param only. 106's cc_sim engine line: draft acceptance 0.354 on the Claude
# Code replay (mb ~0.67); at 16 users ~52 verify rows a step, so rejected drafts cost expert-kernel rows.
#   k4nA  k4n (stock 1.15.1 + --prefix-resume nearest, TP4 all_vram MML 262144 MAXSEQS 16 MNBT 2048, MTP 3) +
#         --draft-window adaptive: depth 3 becomes the ceiling; each sequence's stop follows its measured per-position
#         acceptance (the deepest draft whose chance of being emitted pays for its verify row) and the step's head depth D
#         maximises expected tokens / measured step time (core/sched/draft.h; Radiance #429: +3 % / -1 % on iq4r). Stock
#         code, so greedy text is unchanged; the copy test checks it byte for byte.
# = 93d-fn-tp4ab.sh beside this file with ARMS="k4nA" X93_REF=k4n (106's k4n rows: mb c4/c8/c16, prefill, BetterBench,
# cc_sim u16 trace-seeded jsonl -- no baseline re-run) X93_ITEM=107-fn-tp4dwin. Phases: gate, slow-state probe (+ restart),
# produp + probe, mb c4 / c8 / c16 (+ engine stats each), prefill 16k + fresh 64k / 128k, 33k copy test vs 95's m2 (byte for
# byte), mb pf, BetterBench, cc_sim u16 (+ engine stats); then k4nA vs k4n: cmparm, cc_sim --pair, warm-up answers.
# DRYRUN=1 bash 107-fn-tp4dwin.sh : prints every command, runs and writes nothing.
H=$(cd "$(dirname "$0")" && pwd)
export ARMS=${ARMS:-"k4nA"} X93_REF=${X93_REF:-k4n} X93_ITEM=${X93_ITEM:-107-fn-tp4dwin}
for f in "$H/93d-fn-tp4ab.sh" "$HOME/gpu/ds41/exl3-test/queue-ar/93d-fn-tp4ab.sh"; do [ -e "$f" ] && exec bash "$f"; done
echo "$(date '+%F %T') RESULT fn4 107-fn-tp4dwin: VIRHE 93d-fn-tp4ab.sh not found beside $H" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
exit 1
