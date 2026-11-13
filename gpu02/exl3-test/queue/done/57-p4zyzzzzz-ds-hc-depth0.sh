#!/bin/bash
# 57-p4zyzzzzz-ds-q-depth0.sh (queue item, 10.11.2026, ds41-gg agent; gpu02, all 4 cards, TP4, MML 524288, every radiance
# container DOWN; ~45-55 min) -- lever D7, DEPTH 0 under the adaptive window (no DSpark query pass, no verify row where a measured plain step pays more; 2-cycle probes every 16 at 0), in process on the ds41-gg 4675795f image: ONE START, DSpark NST 5
# --draft-window adaptive, gg_gemv 2 from the start and pinned in every block; gate, modes (t3 notes, DSpark line, routed
# experts in VRAM vs bz d0 / d5), the switch note (draft_depth0 2 refused, 1 / 0 applied at a step), warm-up answers vs stock,
# then blocks A B B A (A = draft_depth0 0, B = 1; POST /admin/fn4 + epoch_bump): stepprobe essay/code, an essay-only probe with
# acceptance by draft position (/stats), editprobe, mb c4 75 s, mb c8 75 s, the scheduler's draft counters around each
# phase; identity under B (the 28k copy, the c1 greedy set vs bz d0); logged when it ends: B vs A pooled (pairs, t95)
# at c1 essay / code / edit, c4, c8; one line A / B vs NST 0 gg_gemv 2 (gz-gemvacq's gg2-gemv2 B blocks, no re-run).
# Nothing goes into a production line without the user's go. IMAGE radiance-local:ds41-gg-4675795f-5c13e555-l
# (XGG_IMAGE); XDZ_ORDER / XDZ_DUR.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-q-depth0.sh : prints every command, runs and writes nothing.
export XGG_IMAGE=${XGG_IMAGE:-radiance-local:ds41-gg-4675795f-5c13e555-l}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-depth0.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT x: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
