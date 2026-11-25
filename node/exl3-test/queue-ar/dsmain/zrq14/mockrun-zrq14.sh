#!/bin/bash
# mockrun-zrq14.sh NAME [--fault ...]: zrq14 under queue-ar/tools/mockrad (ZQ14_SW=0, short knobs; every stage runs once: start 1 identity
# (warm-up, a1-B / a1-Z vs ref12), c1 B Z Z B + bridge, c4 warm-up + B Z Z B + bridge, 1u B / Z / B, THE GATE forced to RUN, start 2 (a1-B8,
# c1, c4 warm-up + 1, 1u), p12win rows per start, zq14sum.py and the summary; the ngshare keeper / cleanup on sandbox paths).
# MOCK SANDBOX RULE (TESTING.md): check `df -i /tmp` (< ~80 %) and `ls /tmp/mockrad` (<= 2 sandboxes, everyone's) first; delete the
# sandbox (/tmp/mockrad/NAME) as soon as its report is read.
cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad || exit 1
n=${1:-zq14-m}; shift
exec ./mockrun.sh --name $n --timeout 3000 --env ZQ14_SW=0 --env ZQ14_EGEN=128 --env ZQ14_C1WIN=6 --env ZQ14_C1WARM=3 --env ZQ14_C4WIN=6 --env ZQ14_C4WARM=3 \
  --env ZQ14_BRIDGEWIN=9 --env ZQ14_U1N=2 --env ZQ14_C1N2=1 --env ZQ14_C4N2=1 --env ZQ14_START2=1 --env ZQ14_BUDGET1=100 --env ZQ14_BUDGET_MIN=120 \
  --env NGSHARE_DIR=/tmp/mockrad/$n/ngdir --env NGSHARE_RUN=/tmp/mockrad/$n/ngrun --env NGSHARE_LOG=/tmp/mockrad/$n/ngshare.log \
  "$@" items/57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zrq14.sh
