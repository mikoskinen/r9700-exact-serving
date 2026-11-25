#!/bin/bash
# 57-p4zyzzzzz-ds-hczz-d0attr.sh (queue item, 10.11.2026, ds-main-port; gpu02, all 4 cards, TP4, MML 524288, every radiance
# container DOWN; ~37 min, 3 starts; XDA_STARTS="S1 S3" ~29) -- WHY A DEPTH-0 STEP OF AN NST 5 SERVER COSTS MORE THAN A PLAIN NST 0 STEP (hcy-depth0b:
# c8 53.2 ms a decode step / 146.9 tok/s vs NST 0 gg_gemv 2 38.9 ms / 170.0; c1 essay 19.45 vs ~16 ms), on the ds41-gg 1b8848a2
# image, DSpark NST 5 --draft-window adaptive, gg_gemv 2, draft_adaptive 1, draft_depth0 1, draft_hybrid 0:
#   S1 clean (gated): in-process blocks over draft_probe A B B A (A = 16 today, B = 4096 = a pure depth-0 plain step), depth 0
#      re-armed (0 then 1 at a step) before every block; essay probe (c1, acceptance by position), mb c4 / c8 75 s with /stats
#      + draft counters at the window's ends (depth-0 / probe share, VRAM units, rank-0 hit, missed and host-stream MB a step);
#      identity under P4096 (28k copy, 20-prompt greedy set c1 vs bz d0, stock);
#   S2 stamped (RADIANCE_AQL_STAMP=1) and S3 op-profiled (--profile-ops, PROFILE_EVERY 16): for gg_gemv 0 (the NST 0
#      references' forms) and then 2, a P16 settle at c8 and a P4096 c8 window -> pass kinds, the drafter's context pass, card idle a step,
#      kernel families / ops by class, against dsm-225-gap's stamps and dsm-225-prof's op profile (NST 0, c8, stock forms; no
#      NST 0 re-run);
#   then "RESULT xda summary": c8 / c4 / c1 ms a step P4096 vs NST 0 (gz-gemvacq gg2-gemv2) and P16 vs P4096, and the c8
#   attribution (probe share, context pass, expert misses, between-pass host time, remainder).
# Body queue-ar/dsmain/dsm-d0attr.sh, drivers d0attrblocks.py / d0attrprof.py (beside it). Output runs/xdsm/d0attr/, log tag
# "RESULT xda ". Sorts after -hcz-resume2 (and -hcz-resume2z-fn-fnres) and before -hd-hybrid under the runner's sort (en_US).
# No cc_sim, no 2.9. Nothing goes into a production line from here. IMAGE radiance-local:ds41-gg-1b8848a2-5c13e555-l
# (XDA_IMAGE); XDA_ORDER / XDA_DUR / XDA_PA / XDA_PB / XDA_SDUR / XDA_WDUR / XDA_STARTS.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-hczz-d0attr.sh : prints every command, runs the read-only preflight, writes nothing.
export XDA_IMAGE=${XDA_IMAGE:-radiance-local:ds41-gg-1b8848a2-5c13e555-l} XDA_ITEM=${XDA_ITEM:-57-p4zyzzzzz-ds-hczz-d0attr}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-d0attr.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xda: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
