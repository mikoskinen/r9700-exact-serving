#!/bin/bash
# 57-p4zyzzzzz-ds-hx-stack.sh (queue item, 10.11.2026, ds-main-port; gpu02, all 4 cards, TP4, MML 524288, every radiance
# container DOWN; ~45 min, ~47 with both NST 5 arms, ~23 / ~25 with XST_NST0=0) -- THE DS STACK CONFIRMATION on the ds41-int2
# 745ce760 image (ds41-int c05de27 = ds41-stack fe030c3 (ds41-k1 1d1f782 + ds41-gg 1b8848a + ds41-resfix cee0ee3, the R1 prefix-resume
# fix) + ds41-vram 92746a8 + ds41-dlend 3f0ac9c, their envs unset = the stack image; + ds41-drk a22a5df, lever DK: drafter_mx pinned
# at NST 5 (XST_DRAFTER_MX=1), drafter_bf never; body dsm-stack.sh). TWO STARTS, each with RADIANCE_DS41_GG_GEMV=2 +
# RADIANCE_AQL_ACQ=rule-w + RADIANCE_DS41_KVFOLD / _OFOLD / _ONEFILL=1 at start and the stack pinned in process every block:
#   start 1 (XST_NST0=1, default; 0 leaves it out): the NST 0 stack N = gg_gemv 2, aql_acq 2, ds41_kvfold 1, ds41_ofold 1,
#     ds41_onefill 1, no drafter; blocks N N N N;
#   start 2: DSpark NST 5 --draft-window adaptive, the NST 5 stack A = the same + draft_adaptive 1, draft_depth0 1, draft_hybrid
#     HYB (default 0), drafter_mx 1 (XST_DRAFTER_MX=1: lever DK, in process, same bits; hq-drafterkern 1u essay +10.7 %, code
#     +4.2 %, edit +5.4 %, identity IDENTICAL); STACK_ARMS=both adds B = the same with draft_hybrid 1, blocks A B B A (default one
#     arm, A A A A).
# In each start: decode blocks (stepprobe essay / code (c1), editprobe, mb c4 / c8 75 s), then one 136k cold prefill (deepq.py's
# prompt, TTFT), then identity (the 28k copy + the c1 greedy set vs bz d0; the deep answer vs dsm-225). Summary: per metric the
# NST 0 stack, the NST 5 stack and the best of the two vs NST 0 + gg_gemv 2 (gz gg2-gemv2 blocks: essay 62.5, code 64.7, c4
# 131.7, c8 170.0; 136k 51.56 s) and the targets (1u >= 90, c4 >= 160, c8 >= 200, 136k <= 35 s).
# IMAGE radiance-local:ds41-int2-745ce760-5c13e555-l (XST_IMAGE; 10.11. evening was ds41-int-c05de277-5c13e555-l, before that ds41-stack-fe030c3b-5c13e555-l; the earlier 57f11bd2 image lacks the R1 fix); XST_DRAFTER_MX (1 | 0); XST_NST0 (1 | 0), STACK_ARMS (stack | both), HYB (0 | 1),
# XST_ORDER, XST_ORDER0, XST_DUR, XST_MATCH (default 8), XST_VRAMENV (the exact VRAM env at both starts, default
# RADIANCE_PLACE_TURNOVER=0 = turnover 0, +32 routed units a rank, vramsweep T0 IDENTICAL; empty = none). Set them by editing the
# export line below before moving this file into queue/ (the queue runner passes no environment).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-hx-stack.sh : prints every command; checks the image is here (no server, no GPU), writes nothing.
export XST_IMAGE=${XST_IMAGE:-radiance-local:ds41-int2-745ce760-5c13e555-l} XST_NST0=${XST_NST0:-1} STACK_ARMS=${STACK_ARMS:-stack} HYB=${HYB:-1} XST_DRAFTER_MX=${XST_DRAFTER_MX:-1}
export XST_VRAMENV=${XST_VRAMENV-RADIANCE_PLACE_TURNOVER=0}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/dsm-stack.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xst: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
