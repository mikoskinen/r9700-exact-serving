#!/bin/bash
# 11.11. ds-main-port: n5-only rerun of hxz-vramstack (vl-n5-on stopped on a false log-literal VIRHE: logok draft,lend:,set,up,off lost its comma; body fixed 12b9268f)
# 57-p4zyzzzzz-ds-hxzya-vramstack5.sh (queue item, 10.11.2026, ds-main-port VRAM agent; gpu02, all 4 cards, TP4, MML 524288, every
# radiance container DOWN; ~52 min) -- THE STACKED EXACT VRAM LEVERS + LEVER DL on the integrated image
# radiance-local:ds41-int2-745ce760-5c13e555-l (ds41-int2 745ce76 = ds41-int c05de27 (ds41-stack fe030c3 (R1) + ds41-vram 92746a8 +
# ds41-dlend 3f0ac9c) + ds41-drk a22a5df; body dsm-vramstack.sh, log tag xvl, out runs/xdsm/vramstack). Runs right after
# 57-p4zyzzzzz-ds-hx-stack, whose NST 0 / NST 5 starts on the same image (turnover 0) and pins (NST 5: drafter_mx 1 in both,
# VSL_DRAFTER_MX=1 = hx-stack's XST_DRAFTER_MX=1) are the OFF sides (en_US sort ignores hyphens: ...dshxstack <
# ...dshxzvramstack < ...dsjshexplane).
#   NST 0 ON start: arena 8 rows, router / compressor bf16, embedding to host, shadow 16, expert prefetch off (+ turnover 0,
#     RADIANCE_DEBUG_VRAM=1 logs only); V x 4 blocks; 136k cold prefill; identity.
#   NST 5 ON start: arena 48 rows + the same + lever DL declared; D7 + the hybrid (match 8) pinned on as in hx-stack (HYB=1);
#     settle A, then A B D D B A in process (A = DL off, B = lend + back-off 3, D = B + borrow; they differ only in the lend
#     switches), D7 re-armed every block, the slab's reclaim waited before A blocks; 136k under A;
#     identity under A, D and draft_lend -1.
# VSL_IMAGE, VSL_DRAFTER_MX (1 | 0), VSL_OFF (auto | own | hx; hx = never an own OFF start), VSL_HYB (1), VSL_BLOCKS0, VSL_FNARMS, VSL_FNORDER, VSL_FNIDENT,
# VSL_SETTLE, VSL_ONDEBUG, VSL_NST, VSL_BASE, VSL_ON. Set them by editing the export line below before moving this file into
# queue/ (the queue runner passes no environment).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-hxzya-vramstack5.sh : prints every command; checks the image and the yaml (no server, no GPU),
# writes nothing.
export VSL_IMAGE=${VSL_IMAGE:-radiance-local:ds41-int2-745ce760-5c13e555-l} VSL_NST=${VSL_NST:-n5} VSL_BASE=${VSL_BASE-t0} VSL_ON=${VSL_ON:-all} VSL_OFF=${VSL_OFF:-auto} VSL_DRAFTER_MX=${VSL_DRAFTER_MX:-1}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-vramstack.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xvl: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
