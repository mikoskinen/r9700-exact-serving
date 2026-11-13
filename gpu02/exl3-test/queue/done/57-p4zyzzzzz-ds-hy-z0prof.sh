#!/bin/bash
# 57-p4zyzzzzz-ds-hy-z0prof.sh (queue item, 11.11.2026, ds-main-port Z0 agent; gpu02, all 4 cards, TP4, MML 524288, every radiance
# container DOWN; ~34 min) -- LEVER Z0 (depth-0 parity) STEP 1, THE PROFILE (tag "RESULT xz0 ", out runs/xdsm/z0prof; body
# queue-ar/dsmain/dsm-z0prof.sh, analysers d0attrprof.py / d0attrblocks.py read-only + z0prof.py). Where a depth-0 step of the
# NST 5 stack2 server spends the time an NST 0 stack2 step does not (stack2: c4 128.9 vs 138.5, c8 172.9 vs 180.3 tok/s).
# FOUR STARTS of radiance-local:ds41-int3-1116248a-5c13e555-l (sha256:e19c28ce372f; stack2's image), stack2's env and pins:
#   n5stamp  NST 5 RADIANCE_AQL_STAMP=1: c4 natural (stack2's pin), a 1-user essay, c4 natural again, c8 natural, then c4 / c8
#            with depth 0 FORCED (fn4 draft_depth0 2, draft_probe 4096; the lend engaged);
#   n0stamp  NST 0 RADIANCE_AQL_STAMP=1: c4, c8;
#   n5prof   NST 5 --profile-ops: c4 / c8 forced depth 0 (per rank, synchronised: not a speed);
#   n0prof   NST 0 --profile-ops: c4, c8.
# Every window: /stats + the draft counters + GET /admin/fn4's lend object at its ends. Analysis: stamps per load (step period,
# trunk kernel families / classes, other pass kinds, card idle) vs NST 0; profile per rank by class, per-rank all-reduce wait
# (floor / above-floor / last arrival), depth 0 vs NST 0 per class; the ranking ("RESULT xz0 rank ..."), "RESULT xz0 summary:".
# NO cc_sim, no 2.9, no identity (nothing here changes output). No power / host / fan settings; production compose read only.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-hy-z0prof.sh : prints every command; checks the image, every helper (files, parse, functions,
# tools, z0prof.py selftest), the sidecar and the yaml; no server, no GPU, writes nothing.
export XZ0_IMAGE=${XZ0_IMAGE:-radiance-local:ds41-int3-1116248a-5c13e555-l} XZ0_ITEM=${XZ0_ITEM:-57-p4zyzzzzz-ds-hy-z0prof}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-z0prof.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xz0: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
