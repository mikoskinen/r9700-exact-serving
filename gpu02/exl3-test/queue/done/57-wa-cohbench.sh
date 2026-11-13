#!/bin/bash
# 78-cohbench.sh (6.11.2026; GPU window on gpu02, ONE card, every radiance container DOWN; ~1 min) -- can a kernel whose AQL
# packet carries NO acquire fence read a stale cached line on gfx1201? (gapbench 75: the agent-scope acquire costs ~0.9 us an
# ordered dispatch, ~0.9 ms a step at ~1000 of them; dropping it is only safe if a consumer cannot see stale data.)
# cohbench (src/gapbench/coh.hip, cohbench.cpp), per trial on one queue, barrier-bit packets: reader (every CU caches X:
# vector loads, or s_load in the scalar arms), writer (X = tag + i, a new tag a trial), check (counts elements that are
# not tag + i). Arms: the check's acquire agent (today) / none, and the writer's too (none_both); vector and scalar.
# A nonzero count under none: a consumer CAN read a stale line through that cache -- the scope cannot be dropped for a
# consumer reading its producer's output through it. Zero bounds how often; it does not prove safety.
# Runs 300 trials x 3 sizes of X (2 KiB, 64 KiB, 1 MiB) with 512 blocks.
# DRYRUN=1 bash 78-cohbench.sh : prints, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
B=${X3G_BIN:-$Q/bin-gap}
O=${X3G_OUT:-$HOME/gpu/ds41/exl3-test/runs/cohbench}
L=${X3G_LOG:-$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log}
IMAGE=${X3G_IMAGE:-radiance-local:ds41-exl3-117-abf7f9a0-f2f3c3f7}
CARD=${CARD:-0}
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$CARD"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
# gpurun NAME SECONDS docker-run-args...: a bounded, named container, removed even if its kernel hangs (timeout -k sends
# SIGKILL to the CLI after 15 s; --init so PID 1 takes SIGTERM; docker rm -f ends the container whatever the CLI did).
gpurun() {
  local name=$1 t=$2; shift 2
  timeout -k 15 $t docker run --rm --init --stop-timeout 10 --name $name "$@"; local rc=$?
  docker rm -f $name > /dev/null 2>&1 || true
  { [ $rc = 124 ] || [ $rc = 137 ]; } && echo "gpurun: $name stopped after $t s (rc $rc)" >&2
  return $rc
}
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT cohbench: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT cohbench: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  for f in cohbench cohbench-coh.hsaco; do [ -e "$B/$f" ] || { log "RESULT cohbench: VIRHE $B/$f missing"; exit 1; }; done
  mkdir -p "$O"
else
  echo "+ mkdir -p $O"
fi
log "RESULT cohbench: start IMAGE=$IMAGE card $CARD ($(md5sum $B/cohbench 2>/dev/null | cut -c1-8) / $(md5sum $B/cohbench-coh.hsaco 2>/dev/null | cut -c1-8)) -> $O"
for n in 512 16384 262144; do
  run "gpurun cohbench-$$-$n 300 $DEV -v $B:/b:ro --entrypoint /b/cohbench $IMAGE /b/cohbench-coh.hsaco 300 $n 512 > $O/cohbench-n$n.txt 2>&1"
  [ "$DRY" = 1 ] || { grep '^RESULT cohbench' $O/cohbench-n$n.txt | sed "s/^RESULT cohbench/RESULT cohbench X=$n/" | while read -r l; do log "$l"; done
                      [ "$(grep -c '^RESULT cohbench .*stale elements' $O/cohbench-n$n.txt)" = 6 ] || log "RESULT cohbench X=$n: VIRHE not six arms ($(tail -2 $O/cohbench-n$n.txt | tr '\n' ' ' | cut -c1-300))"; }
done
log "RESULT cohbench: done -- $O"
