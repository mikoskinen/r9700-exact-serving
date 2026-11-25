#!/bin/bash
# 75-gapbench.sh (6.11.2026; GPU window on gpu02, ONE card, every radiance container DOWN; ~2 min) -- what one ordered
# AQL dispatch costs by itself. The engine's played tape waits p50 3.76 us before every ordered dispatch (AQL stamps,
# runs/x117 mtp5-stamp: 937 ordered dispatches = 3.78 ms of a 40.8 ms verify pass, 95 % of gaps in 3-4.5 us, the same
# after every kind of kernel). gapbench (src/gapbench) writes N tiny kernels as AQL packets on ONE queue the way
# core/device/aql.cpp plays them (barrier bit, agent acquire/release, profiling on), one doorbell, wall clock to the last
# completion; each arm changes one thing:
#   bar_agent          today's played packet                      bar_agent_sig   + a completion signal on every packet
#   bar_system(_sig)   system-scope fences                        bar_none(_sig)  no fence scope (TIMING ONLY)
#   bar_acqnone / bar_relnone  one side's scope off               nobar_agent     no barrier bit, no wait (launch rate)
#   chain_agent(_sig)  no barrier bit, the kernel waits in its prologue on a device counter (r4d_chain.h's form)
#   chain_none         the same with no fence scope
#   *_w                128 workgroups of 256 instead of 1 of 64   *_busy          each kernel spins ~10 us
# Reading: bar_agent - nobar_agent = the barrier's wait; bar_agent - bar_none = the cache maintenance of agent scope;
# bar_agent_sig - bar_agent = what a per-packet signal (RADIANCE_AQL_STAMP) adds; chain_agent vs bar_agent = what
# chained launches would save a dispatch; *_busy - 10 us = what is left when the processor has a kernel's time to set
# up the next. The _sig arms also give the profiling gap start(i) - end(i-1), the stamps' own measure.
# DRYRUN=1 bash 75-gapbench.sh : prints, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
B=${X3G_BIN:-$Q/bin-gap}
O=${X3G_OUT:-$HOME/gpu/ds41/exl3-test/runs/gapbench}
L=${X3G_LOG:-$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log}
IMAGE=${X3G_IMAGE:-radiance-local:ds41-exl3-117-abf7f9a0-f2f3c3f7}
CARD=${CARD:-0}
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$CARD"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT gapbench: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT gapbench: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  for f in gapbench gapbench-tiny.hsaco; do [ -e "$B/$f" ] || { log "RESULT gapbench: VIRHE $B/$f missing"; exit 1; }; done
  mkdir -p "$O"
else
  echo "+ mkdir -p $O"
fi
log "RESULT gapbench: start IMAGE=$IMAGE card $CARD ($(md5sum $B/gapbench 2>/dev/null | cut -c1-8) / $(md5sum $B/gapbench-tiny.hsaco 2>/dev/null | cut -c1-8)) -> $O"
for n in 1000 3000; do
  run "timeout 600 docker run --rm $DEV -v $B:/b:ro --entrypoint /b/gapbench $IMAGE /b/gapbench-tiny.hsaco $n 7 > $O/gapbench-n$n.txt 2>&1"
  [ "$DRY" = 1 ] || { grep '^RESULT gapbench' $O/gapbench-n$n.txt | sed "s/^RESULT gapbench/RESULT gapbench N=$n/" | while read -r l; do log "$l"; done
                      grep -q '^RESULT gapbench check' $O/gapbench-n$n.txt || log "RESULT gapbench N=$n: VIRHE no check line ($(tail -2 $O/gapbench-n$n.txt | tr '\n' ' ' | cut -c1-300))"; }
done
log "RESULT gapbench: done -- $O"
