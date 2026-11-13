#!/bin/bash
# mockrun-xpush.sh NAME: the XPUSH screen under queue-ar/tools/mockrad (XPU_SW=0, short knobs; every stage runs once: P0 (not emulated:
# logged as the mock's), identity B / C, cold B C at two small depths, c8 B C, c16 B C, the summary)
cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad || exit 1
exec ./mockrun.sh --name ${1:-xpu-m} --timeout 1500 --env XPU_SW=0 --env XPU_C8WIN=8 --env XPU_C8WARM=4 --env "XPU_DEPTHS=12000 6000" \
  --env XPU_CORDER=BC --env XPU_C8ORDER=BC --env XPU_C16ORDER=BC --env XPU_BUDGET_MIN=60 items/57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-xpush.sh
