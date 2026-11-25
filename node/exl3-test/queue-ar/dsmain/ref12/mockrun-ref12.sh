#!/bin/bash
# mockrun-ref12.sh NAME: the int12 reference item under queue-ar/tools/mockrad (X12_SW=0, short knobs; every stage runs once)
cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad || exit 1
exec ./mockrun.sh --name ${1:-r12-m} --timeout 1500 --env X12_SW=0 --env X12_EGEN=128 --env "X12_DEPTHS=6000 12000" --env X12_NTOK=16 \
  --env X12_C8WIN=8 --env X12_C8WARM=4 --env X12_C4WIN=6 --env X12_C4WARM=3 --env X12_C1WIN=6 --env X12_C1WARM=3 --env X12_MXCONFIRM=1 \
  --env X12_BUDGET_MIN=120 items/57-p4zyzzzzz-ds-r92zzzzzz-ref12.sh
