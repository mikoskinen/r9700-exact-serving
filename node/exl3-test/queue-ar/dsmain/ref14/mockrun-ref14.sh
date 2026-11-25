#!/bin/bash
# mockrun-ref14.sh NAME [--fault ...]: the int14 reference read under queue-ar/tools/mockrad (R14_SW=0, short knobs; every stage runs
# once: identity (warm-up, a1 vs yzint14c / ref12, gs16, A11 in range, staggered), 1u warm + switched, c8 work-based B Z Z B (gens 1),
# c8 35 s, c16 fixed, CAP48, cold, c4 warm-up + 2, c1, the ncmatch sets and the summary; the ngshare keeper / cleanup on sandbox paths).
# The STOP path: mockrun-ref14.sh r14-stop --swap gpu/ds41/exl3-test/runs/xdsm/yzint14c/x1/yzi-a1-Z.jsonl=gpu/ds41/exl3-test/queue-ar/dsmain/ref14/mock-stop-a1Z.jsonl
#   (a doctored copy of yzint14c's a1-Z, answer 1 changed: a1-O differs from it -> the identity gate STOPS before every speed block; the mock
#   replays the recorded greedy-set answers, so --fault differ cannot reach a1 / gs16 / the staggered probes).
cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad || exit 1
n=${1:-r14-m}; shift
exec ./mockrun.sh --name $n --timeout 3000 --env R14_SW=0 --env R14_EGEN=128 --env R14_C8WIN=8 --env R14_C8WARM=4 --env R14_C4WIN=6 --env R14_C4WARM=3 \
  --env R14_C1WIN=6 --env R14_C1WARM=3 --env R14_BUDGET_MIN=120 --env R14_LDPF=2048 --env "R14_A11ID=1100" --env "R14_CDEPTHS=6000 12000" --env R14_U1N=2 \
  --env R14_SWN=1 --env R14_C16DUR=8 --env R14_C16WARM=4 --env R14_C8GN=1 \
  --env NGSHARE_DIR=/tmp/mockrad/$n/ngdir --env NGSHARE_RUN=/tmp/mockrad/$n/ngrun --env NGSHARE_LOG=/tmp/mockrad/$n/ngshare.log \
  "$@" items/57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zref14.sh
