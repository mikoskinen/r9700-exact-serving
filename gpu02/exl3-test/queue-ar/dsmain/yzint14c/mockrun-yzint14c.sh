#!/bin/bash
# mockrun-yzint14c.sh NAME [--fault ...]: the int14c base-candidate screen under queue-ar/tools/mockrad (YZI_SW=0, short knobs; every stage
# runs once: P0 (not emulated: logged as the mock's, a stand-in A11 entry), identity (a1 Z / B, gs16 Z, A11 in range, staggered Z + A), c8
# B A G B, A4 c16 fixed + CAP48 B / C, SWL reads B / W, c8 B Z Z B, c16 Z, 1u B / Z, cold B / Z, the summary; the ngshare keeper / cleanup on
# sandbox paths). The STOP path: mockrun-yzint14c.sh yzi-stop --fault differ=ds41_shexp_lane:1 (Z's answers differ -> the identity gate).
cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad || exit 1
n=${1:-yzi-m}; shift
exec ./mockrun.sh --name $n --timeout 3000 --env YZI_SW=0 --env YZI_EGEN=128 --env YZI_C8WIN=8 --env YZI_C8WARM=4 --env YZI_BUDGET_MIN=120 \
  --env YZI_LDPF=2048 --env "YZI_A11ID=1100" --env "YZI_CDEPTHS=6000" --env YZI_U1N=2 --env YZI_C16DUR=8 --env YZI_C16WARM=4 \
  --env NGSHARE_DIR=/tmp/mockrad/$n/ngdir --env NGSHARE_RUN=/tmp/mockrad/$n/ngrun --env NGSHARE_LOG=/tmp/mockrad/$n/ngshare.log \
  "$@" items/57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yzint14c.sh
