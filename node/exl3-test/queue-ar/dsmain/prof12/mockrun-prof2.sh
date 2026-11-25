#!/bin/bash
# mockrun-prof2.sh NAME: prof START 2 (profiled) under queue-ar/tools/mockrad (XP2_SW=0, short knobs; every window and parser runs once)
cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad || exit 1
exec ./mockrun.sh --name ${1:-p2-m} --timeout 1500 --env XP2_SW=0 --env XP2_EGEN=128 --env XP2_C8WIN=8 --env XP2_C8WARM=4 \
  --env XP2_C4WIN=6 --env XP2_C4WARM=3 --env XP2_C16DUR=10 --env XP2_C16WARM=4 --env XP2_BUDGET_MIN=60 items/57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzz-prof2.sh
