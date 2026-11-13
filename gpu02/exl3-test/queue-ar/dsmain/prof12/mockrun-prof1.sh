#!/bin/bash
# mockrun-prof1.sh NAME: prof START 1 under queue-ar/tools/mockrad (XP1_SW=0, short knobs; every stage runs once)
cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad || exit 1
exec ./mockrun.sh --name ${1:-p1-m} --timeout 1200 --env XP1_SW=0 --env XP1_EGEN=128 --env XP1_C8WIN=8 --env XP1_C8WARM=4 \
  --env XP1_C16DUR=10 --env XP1_C16WARM=4 --env XP1_BUDGET_MIN=60 items/57-p4zyzzzzz-ds-r92zzzzzz-ref12zzz-prof1.sh
