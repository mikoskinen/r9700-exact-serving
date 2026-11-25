#!/bin/bash
# mockrun-yd2m.sh NAME [--env K=V ...]: the A4 screen under queue-ar/tools/mockrad (YDM_SW=0, short knobs; every stage runs once: P0 (not
# emulated: logged as the mock's), identity B / C / P, staggered B (and C), cold B P, c8 B C P, 1u C B, c16 B C, the summary).
# "--env YDM_MOCK_GATE=1" runs the A4-gate path (A10 alone: B / P only).
cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad || exit 1
NAME=${1:-ydm-m}; shift
exec ./mockrun.sh --name $NAME --timeout 1500 --env YDM_SW=0 --env YDM_C8WIN=8 --env YDM_C8WARM=4 --env YDM_NREADS=1 \
  --env YDM_C8ORDER=BCP --env YDM_PDEPTHS=6000 --env YDM_LDPF=2048 \
  --env YDM_U1ORDER=CB --env YDM_C16ORDER=BC --env YDM_BUDGET_MIN=60 "$@" items/57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yd2m.sh
