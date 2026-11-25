#!/bin/bash
# 57-p4zyzzzzz-ds-bz-draft.sh (queue item, 10.11.2026, ds-main-port agent; gpu02, all 4 cards, TP4, MML 524288, every
# radiance container DOWN; ~65 min) -- the DeepSeek DSpark drafter (the model's own 3-stage block drafter, block 5) sized
# and profiled on the main image, EXL3 2.25 (user 10.11.: "mtp ehka tata voisi profiloida ja kehittaa, adaptiivinen jne";
# coordinator: measure, profile the draft step, identity reported): arms d0 (NST 0) / d5 (NST 5) / d5a (NST 5 + --draft-window
# adaptive), each gate, probe (essay / code acceptance), mb c4 / c8 (aggregate + per-stream p50), 28k copy, greedy set c1;
# then a --profile-ops d5 start (1-user + c4 windows: the drafter's blocks vs the trunk's decode + verify rows, lane waits,
# host-link MB a step) and the verify-path oracle (--spec-ngram 5 --debug-draft-oracle: the 6-row verify must reproduce
# d0's text). Drafting output may differ from NST 0 at a near-tie: logged as DIFFERENT, not hidden; nothing goes into a
# production line without the user's go. IMAGE radiance-local:ds41-main-e736c019-5c13e555 (XDSM_IMAGE).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-bz-draft.sh : prints every command, runs and writes nothing.
export XDSM_IMAGE=${XDSM_IMAGE:-radiance-local:ds41-main-e736c019-5c13e555}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-draft.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xdsd: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
