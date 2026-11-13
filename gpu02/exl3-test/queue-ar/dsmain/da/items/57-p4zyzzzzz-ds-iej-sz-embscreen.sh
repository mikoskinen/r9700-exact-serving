#!/bin/bash
# 57-p4zyzzzzz-ds-iej-sz-embscreen.sh (queue item, 11.11.2026, ds-main-port DA agent at the coordinator's request; gpu02, 4 cards,
# no server; ~8 min) -- THE DRAFTER-EMBEDDING SCREEN, offline: the bf16 port (validated vs the engine, first-token agreement 0.942)
# on the DA pilot's KL dump (kept by iei for this item), held-out documents: the container's q6_K token table vs the release's BF16
# embed.weight, for the container's drafter and the pilot's best.ovl (4 runs, one card each); then the dump and the table are
# deleted. Runs after iei-da-c-measure and iej-sampled. Body queue-ar/dsmain/da/dsm-emb.sh. Tag "RESULT xdae ".
# DRYRUN=1 bash 57-p4zyzzzzz-ds-iej-sz-embscreen.sh : prints every command; checks the trainer image, torchenv, every helper.
export DAE_ITEM=${DAE_ITEM:-57-p4zyzzzzz-ds-iej-sz-embscreen}
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/da/dsm-emb.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xdae: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
bash $B
