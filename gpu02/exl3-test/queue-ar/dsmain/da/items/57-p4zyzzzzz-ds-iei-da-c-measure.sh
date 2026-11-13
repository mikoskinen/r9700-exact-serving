#!/bin/bash
# 57-p4zyzzzzz-ds-iei-da-c-measure.sh (queue item, 11.11.2026, ds-main-port DA agent; gpu02, all 4 cards; every radiance container
# DOWN; ~75 min) -- LEVER DA PILOT, phase measure (renamed by the coordinator from 57-p4zyzzzzzzzz-da-c-measure.sh, md5 1817cca1;
# 11.11. 04:30 UTC its body moved to dsm-da2.sh: the stack2 env as stack2 ran (hybrid 1, lend 1 + backoff 3, VRAM levers), and per
# arm the server-default sampling 1-user probe (seed 1234, essay / code / edit x2) + mb c4 / c8; starts B A B2; 05:30 UTC + arm X,
# an external fine-tune of the drafter (HF Avifenesh/...-drafter-ft, verified + converted: stats/ovl/ext.ovl): starts B A X B2). Body
# queue-ar/dsmain/da/dsm-da2.sh measure (its header). Runs after ieg-da-a-dump and ieh-da-b-train; finds runs/xdsm/da/stats/ovl/
# best.ovl or logs VIRHE and stops at once. Tag "RESULT xdap ", out runs/xdsm/da. IMAGE radiance-local:ds41-da-bd2caa0c-5c13e555-l
# (sha256:561f4fbc60e3; ds41-da bd2caa0 = ds41-int3 1116248 + lever DA). Public / synthetic text only; no cc_sim, no 2.9, no cloud.
# DRYRUN=1 bash 57-p4zyzzzzz-ds-iei-da-c-measure.sh : prints every command; checks the image, every helper, the inputs; no server, no GPU.
export DAP_IMAGE=${DAP_IMAGE:-radiance-local:ds41-da-bd2caa0c-5c13e555-l} DAP_ITEM=${DAP_ITEM:-57-p4zyzzzzz-ds-iei-da-c-measure}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/da/dsm-da2.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xdap: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B measure
