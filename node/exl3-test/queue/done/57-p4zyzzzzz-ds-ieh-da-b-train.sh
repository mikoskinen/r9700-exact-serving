#!/bin/bash
# 57-p4zyzzzzzzzz-da-b-train.sh (queue item, 11.11.2026, ds-main-port DA agent; gpu02, all 4 cards; every radiance container DOWN; ~32 min)
# -- LEVER DA PILOT, phase train (drafter adaptation = self-distilling the DSpark drafter to the EXL3 2.25 target). Body
# queue-ar/dsmain/da/dsm-da.sh train (its header says what each phase runs). The three phases run in this order (a-dump, b-train,
# c-measure); a later phase finds its input in runs/xdsm/da/stats or logs VIRHE and stops at once. Tag "RESULT xdap ", out
# runs/xdsm/da. IMAGE radiance-local:ds41-da-bd2caa0c-5c13e555-l (sha256:561f4fbc60e3; ds41-da bd2caa0 = ds41-int3 1116248 + lever DA); trainer rocm/dev-ubuntu-24.04:latest +
# ~/gpu/ds41/da/torchenv (torch 2.10.0+rocm7.13 TheRock gfx120X). Public / synthetic text only; no cc_sim, no 2.9, no cloud.
# DRYRUN=1 bash 57-p4zyzzzzzzzz-da-b-train.sh : prints every command; checks the image, every helper, the inputs; no server, no GPU.
export DAP_IMAGE=${DAP_IMAGE:-radiance-local:ds41-da-bd2caa0c-5c13e555-l} DAP_ITEM=${DAP_ITEM:-57-p4zyzzzzzzzz-da-b-train}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/da/dsm-da.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xdap: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B train
