#!/bin/bash
# mockrun-zzk25s.sh NAME [--fault ...]: the zzk25s lever-K25U screen under queue-ar/tools/mockrad (K25S_SW=0, short knobs; every stage runs
# once: identity (warm-up, a1 B / Z vs ref12, gs16-Z, staggered ld-Z), c8 work-based B Z Z B (gens 1), c16 B / Z (gens 1), 1u essay B / Z
# (one discarded + 1 each), ncmatch / steps.py / k25sum.py and the summary; the ngshare keeper / cleanup on sandbox paths).
# The STOP path: mockrun-zzk25s.sh k25s-stop --fault differ=ds41_bi (every arm's answers differ from ref12's a1 -> the identity gate).
# MOCK SANDBOX RULE (TESTING.md): check `df -i /tmp` (< ~80 %) and `ls /tmp/mockrad` (<= 2 sandboxes, everyone's) first; delete the
# sandbox (/tmp/mockrad/NAME) as soon as its report is read.
cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad || exit 1
n=${1:-k25s-m}; shift
exec ./mockrun.sh --name $n --timeout 3000 --env K25S_SW=0 --env K25S_C8WARM=4 --env K25S_BUDGET_MIN=120 --env K25S_LDPF=2048 --env K25S_C8GN=1 --env K25S_U1N=1 \
  --env NGSHARE_DIR=/tmp/mockrad/$n/ngdir --env NGSHARE_RUN=/tmp/mockrad/$n/ngrun --env NGSHARE_LOG=/tmp/mockrad/$n/ngshare.log \
  "$@" items/57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzk25s.sh
