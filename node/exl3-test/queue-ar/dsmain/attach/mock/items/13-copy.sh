#!/bin/bash
# 13-copy.sh (ATTACH MOCK item, 12.11.2026, queue-ar/dsmain/attach/mock; run by mockchain.sh under queue-ar/tools/mockrad ONLY,
# never queued) -- ATTACH + KEEP, the COPY path (no ngshare, otherwise = 12): its own start; queue empty -> down
# ATTACH-SIG: SIGONLY=1 DRYRUN=1 bash <this item> prints its start signature (queue-ar/dsmain/attach/attach.sh)
export ATTACH=${ATTACH:-1} KEEP=${KEEP:-1}
export ATM_TAG=cpy ATM_IMAGE=radiance-local:mock-int10 ATM_MNBT=4160 ATM_PIN='{"ds41_psk": 2}'
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/attach/mock/dsm-attmock.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xatm: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }
bash $B
