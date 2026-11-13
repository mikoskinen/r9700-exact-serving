#!/bin/bash
# mockrun-ync.sh NAME [--env K=V ...]: the A9 / A11 screen under queue-ar/tools/mockrad (YNC_SW=0, short knobs; every stage runs once: P0
# (not emulated: logged as the mock's, a stand-in A11 entry), identity B A G N + A11 in range, staggered A (and N), c8 B A G N, BEST, c16
# B X, cold B X, the summary).
cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad || exit 1
NAME=${1:-ync-m}; shift
exec ./mockrun.sh --name $NAME --timeout 1800 --env YNC_SW=0 --env YNC_C8WIN=8 --env YNC_C8WARM=4 \
  --env YNC_C8ORDER=BAGNB --env "YNC_CDEPTHS=6000" --env "YNC_A11ID=1100" --env YNC_LDPF=2048 \
  --env YNC_C16ORDER=BX --env YNC_BUDGET_MIN=80 "$@" items/57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yync.sh
