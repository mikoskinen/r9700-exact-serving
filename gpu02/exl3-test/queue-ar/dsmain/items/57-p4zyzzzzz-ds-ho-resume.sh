#!/bin/bash
# 57-p4zyzzzzz-ds-ho-resume.sh (queue item, 10.11.2026, ds41 resume agent, lever R1 = correctness; gpu02, all 4 cards, TP4,
# MML 524288, every radiance container DOWN; ~20 min) -- does a prefix-cache resume answer exactly as the cold prefill of the
# same prompt does? One reading says no (10.11. gzzz-attnfold arm A: the 136k deep prompt resumed at 133120 of 136216 differs
# from its cold answer at char 170); the engine claims "prefix resume: exact". ONE START of the main line (ds41-gg-8b311fcc,
# gg_gemv 0 = today's arithmetic, t3 env, dsm-serve.sh's serve line, NST 0), sequential requests on an idle server, cached
# tokens + logprobs (top 2) + the /sessions tier counters per request: deep cold / full hit / second hit, a partial hit at the
# interval checkpoint 129024 (the deep prompt with its last ~4k tokens other text) and that prompt cold after
# /reset_prefix_cache, deep back as a partial hit, a fresh-nonce 136k cold + hit, 36k and 74k cold + hit (the indexer's keep
# threshold), the 28k copy cold + hit, a late hit, and deep cold again after a reset (the cold-vs-cold control). Every pair
# byte-compared (resume_ho.py cmp: text, first differing token, logprob differences) and against runs/xdsm/deep-dsm-225.txt.
# Body: queue-ar/dsmain/dsm-resume.sh; driver queue-ar/dsmain/resume_ho.py. Output runs/xdsm/resume/, log tag xho.
# Sorts after 57-p4zyzzzzz-ds-h-oracle and before -hp-drkbench under the runner's en_US sort ("horesume" > "horacle").
# IMAGE radiance-local:ds41-gg-8b311fcc-5c13e555-l (XHO_IMAGE). No cc_sim. Nothing goes into a production line from here.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-ho-resume.sh : prints every command, runs and writes nothing.
export XHO_IMAGE=${XHO_IMAGE:-radiance-local:ds41-gg-8b311fcc-5c13e555-l} XHO_ITEM=${XHO_ITEM:-57-p4zyzzzzz-ds-ho-resume}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-resume.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xho: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
