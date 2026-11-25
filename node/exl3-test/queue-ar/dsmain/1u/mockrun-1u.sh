#!/bin/bash
# mockrun-1u.sh NAME: the 1u all-switch item under queue-ar/tools/mockrad (X1U_SW=0, short knobs; every stage runs once)
cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad || exit 1
exec ./mockrun.sh --name ${1:-1u-m} --timeout 1500 --env X1U_SW=0 --env X1U_IMAGE=radiance-local:ds41-int10-f6ef7cc3-5c13e555-l \
  --env "X1U_PLAN=MX:O,MX GQ:O,GQ E2:O,E2 F4:O,F4" --env X1U_PLAN2=C1T:C50,C40 --env X1U_PLAN3=MX12:O,MX12 --env X1U_EGEN=128 \
  --env "X1U_C8PLAN=S:O S:N S:H250 C:O C:H500" --env X1U_C8GENS=1 --env X1U_C8WARM=4 --env X1U_C4WIN=6 --env X1U_C4WARM=3 \
  --env X1U_C1WIN=6 --env X1U_C1WARM=3 --env X1U_BUDGET_MIN=120 \
  items/57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzzzz-1u.sh
