#!/bin/bash
# mockrun-ychn.sh NAME: the CHN-only form (ychn) under queue-ar/tools/mockrad (YLL_SW=0, short knobs: identity B D, c8 B D, the summary)
cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad || exit 1
exec ./mockrun.sh --name ${1:-ychn-m} --timeout 1500 --env YLL_SW=0 --env YLL_C8WIN=8 --env YLL_C8WARM=4 --env YLL_C8ORDER=BD --env YLL_BUDGET_MIN=60 \
  items/57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-ychn.sh
