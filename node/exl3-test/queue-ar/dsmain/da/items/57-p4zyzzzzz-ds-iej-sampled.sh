#!/bin/bash
# 57-p4zyzzzzz-ds-iej-sampled.sh (queue item, 11.11.2026, ds-main-port DA agent at the coordinator's request; gpu02, all 4 cards;
# every radiance container DOWN; ~25-30 min) -- WHAT DRAFTING GIVES REAL USERS TODAY: 1-user essay / code / edit at the
# server-default sampling (no temperature sent = the engine's temperature 1; per-request seed 1234) and at temperature 0, on the int3
# image with the stack2 env as it ran: one NST 5 start ABBA in process (A = stack2 drafting, B = draft_depth0 2 forced depth 0)
# + one NST 0 start (N N); identity at temperature 0 (28k copy + greedy set vs bz d0) in each start. Body
# queue-ar/dsmain/da/dsm-sam.sh (its header). Tag "RESULT xsam ", out runs/xdsm/sampled. IMAGE
# radiance-local:ds41-int3-1116248a-5c13e555-l (sha256:e19c28ce372f). No cc_sim, no 2.9.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-iej-sampled.sh : prints every command; checks the image, every helper; no server, no GPU.
export SAM_IMAGE=${SAM_IMAGE:-radiance-local:ds41-int3-1116248a-5c13e555-l} SAM_ITEM=${SAM_ITEM:-57-p4zyzzzzz-ds-iej-sampled}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/da/dsm-sam.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xsam: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
