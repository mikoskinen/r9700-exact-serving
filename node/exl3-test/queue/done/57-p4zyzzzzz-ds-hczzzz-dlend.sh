#!/bin/bash
# 57-p4zyzzzzz-ds-hczzzz-dlend.sh (queue item, 10.11.2026, ds-main-port VRAM agent; gpu02, all 4 cards, TP4, MML 524288, every
# radiance container DOWN; ~33 min) -- LEVER DL (the drafter's lend) A / B / C IN PROCESS, one DSpark NST 5 start (D7 depth 0 on,
# the stack env: gg_gemv 2, rule-w, K1) on radiance-local:ds41-dlend-9461f81e-5c13e555-l (ds41-stack 57f11bd + lever DL; body
# dsm-dlend.sh, log tag xdl, out runs/xdsm/dlend): A = draft_lend 0 / back-off 0 (today's D7), B = draft_lend 1 / back-off 3,
# C = draft_lend 0 / back-off 3; ABCCBA, mb 75 s, c1 essay / code / edit, c4, c8; per phase lend + /stats counters; the slab's
# reclaim waited before every A / C block; identity (28k copy + 20 greedy c1 vs bz d0) under draft_lend -1 (lent at every step)
# and under B; no prefill phase (decode-only lever). Sorts before 57-p4zyzzzzz-ds-hd-hybrid (en_US: ...dshczzzzdlend < ...dshdhybrid).
# VDL_IMAGE, VDL_ORDER (ABCCBA), VDL_DUR (75), VDL_VRAMENV (none; e.g. RADIANCE_PLACE_TURNOVER=0). Set them by editing the
# export line below before moving this file into queue/ (the queue runner passes no environment).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-hczzzz-dlend.sh : prints every command; checks the image and the yaml (no server, no GPU), writes nothing.
export VDL_IMAGE=${VDL_IMAGE:-radiance-local:ds41-dlend-9461f81e-5c13e555-l} VDL_ORDER=${VDL_ORDER:-ABCCBA} VDL_DUR=${VDL_DUR:-75} VDL_VRAMENV=${VDL_VRAMENV:-}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-dlend.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xdl: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
