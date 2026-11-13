#!/bin/bash
# mockrun-zzzlaunch.sh NAME [--fault ...]: the zzzlaunch lever-RCO screen under queue-ar/tools/mockrad (LCH_SW=0, short knobs; every stage
# runs once: identity (warm-up, a1 B / Z vs ref12, gs16-Z, staggered ld-Z), c8 work-based B Z Z B (gens 1, clock pairs), c16 B / Z (gens 1),
# 1u essay B / Z (one discarded + 1 each), ncmatch / steps.py / lchsum.py, the AQL read over aqlcp.py synth's SYNTHETIC stamps (lchaql.py)
# and the summary; the ngshare keeper / cleanup on sandbox paths).
# The STOP path: mockrun-zzzlaunch.sh lch-stop --fault differ=ds41_bi (every arm's answers differ from ref12's a1 -> the identity gate).
# MOCK SANDBOX RULE (TESTING.md): check `df -i /tmp` (< ~80 %) and `ls /tmp/mockrad` (<= 2 sandboxes, everyone's) first; delete the
# sandbox (/tmp/mockrad/NAME) as soon as its report is read.
cd ~/gpu/ds41/exl3-test/queue-ar/tools/mockrad || exit 1
n=${1:-lch-m}; shift
exec ./mockrun.sh --name $n --timeout 3000 --env LCH_SW=0 --env LCH_C8WARM=4 --env LCH_BUDGET_MIN=120 --env LCH_LDPF=2048 --env LCH_C8GN=1 --env LCH_U1N=1 \
  --env NGSHARE_DIR=/tmp/mockrad/$n/ngdir --env NGSHARE_RUN=/tmp/mockrad/$n/ngrun --env NGSHARE_LOG=/tmp/mockrad/$n/ngshare.log \
  "$@" items/57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzzlaunch.sh
