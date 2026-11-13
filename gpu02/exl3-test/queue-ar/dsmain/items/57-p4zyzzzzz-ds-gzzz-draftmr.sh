#!/bin/bash
# 57-p4zyzzzzz-ds-gzzz-draftmr.sh (queue item, 10.11.2026, ds41-gg agent; gpu02, all 4 cards, TP4, MML 524288, every
# radiance container DOWN; ~60-70 min) -- DRAFTING x lever D2 (the decode band's multi-row form, gg_gemv 2, the measured
# MR table) in process on the ds41-gg 8b311fcc image: two starts, (1) DSpark NST 5 --draft-window adaptive, (2) prompt
# lookup alone --spec-ngram 5 --draft-window adaptive (the head not declared, loaded or placed); each start gate, modes
# (drafter lines, routed experts in VRAM vs bz d0 / d5), the switch note, warm-up answers vs stock, then blocks A B B A
# (A = gg_gemv 0, B = 2; POST /admin/fn4 + epoch_bump): stepprobe essay/code, an essay-only probe with acceptance by draft
# position (/stats), editprobe, mb c4 75 s, mb c8 75 s; identity under gg_gemv 2 (the 28k copy, the c1 greedy set vs bz
# d0); per start, logged when it ends: B vs A pooled (pairs, t95) at c1 essay / code / edit, c4, c8; A / B vs NST 0
# (gz-gemvacq's gg2-gemv2 blocks, no re-run); A vs bz d0 / d5a. Covers hz-ngram's phases at start 2 (mb in 75 s
# blocks, identity under gg_gemv 2). Nothing goes into a production line without the user's go. IMAGE
# radiance-local:ds41-gg-8b311fcc-5c13e555-l (XGG_IMAGE).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-gzzz-draftmr.sh : prints every command, runs and writes nothing.
export XGG_IMAGE=${XGG_IMAGE:-radiance-local:ds41-gg-8b311fcc-5c13e555-l}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-draftmr.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xdm: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
