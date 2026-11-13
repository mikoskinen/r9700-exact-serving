#!/bin/bash
# mockrun-ypfx.sh NAME: the A10 screen under queue-ar/tools/mockrad (YPF_SW=0, short knobs; every stage runs once: P0 (not emulated:
# logged as the mock's), identity B / C, cold B C at two small depths, c8 B C, c16 B C, the summary) -- = xpush/mockrun-xpush.sh
cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad || exit 1
exec ./mockrun.sh --name ${1:-ypf-m} --timeout 1500 --env YPF_SW=0 --env YPF_C8WIN=8 --env YPF_C8WARM=4 --env "YPF_DEPTHS=12000 6000" \
  --env YPF_CORDER=BC --env YPF_C8ORDER=BC --env YPF_C16ORDER=BC --env YPF_BUDGET_MIN=60 items/57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-ypfx.sh
