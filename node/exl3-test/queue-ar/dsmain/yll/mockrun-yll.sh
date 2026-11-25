#!/bin/bash
# mockrun-yll.sh NAME: the A3 / A5 + A10 screen under queue-ar/tools/mockrad (YLL_SW=0, short knobs; every stage runs once: P0 (not
# emulated: logged as the mock's), identity under B C D, staggered C (+ B at the end), c8 B C D, 1u B C, c16 B C, the summary; no A10)
cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad || exit 1
exec ./mockrun.sh --name ${1:-yll-m} --timeout 1800 --env YLL_SW=0 --env YLL_C8WIN=8 --env YLL_C8WARM=4 \
  --env YLL_C8ORDER=BCD --env YLL_C16ORDER=BC --env YLL_U1ORDER=BC --env YLL_NREADS=1 --env YLL_LDPF=2048 --env YLL_BUDGET_MIN=60 \
  items/57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yll.sh
