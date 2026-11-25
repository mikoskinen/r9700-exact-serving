#!/bin/bash
# 06-zz-plain.sh (ATTACH MOCK item, 12.11.2026, queue-ar/dsmain/attach/mock; run by mockchain.sh under queue-ar/tools/mockrad ONLY,
# never queued) -- NO ATTACH, injected by mockchain.sh right after 06 (the coordinator race): its cguard must stop the server kept for 07
export ATM_TAG=zzplain ATM_IMAGE=radiance-local:mock-int10 ATM_MNBT=4096 ATM_PIN='{"ds41_psk": 2}'
B=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/attach/mock/dsm-attmock.sh
[ -e $B ] || { echo "$(date '+%F %T') RESULT xatm: VIRHE $B missing" | tee -a $HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log; exit 1; }
[ "${DRYRUN:-0}" = 1 ] || { . $HOME/gpu/ds41/exl3-test/queue-ar/dsmain/compose-guard.sh && cguard; }
bash $B
