#!/bin/bash
# 82-cohbench2.sh (6.11.2026; GPU window on gpu02, ONE card, every radiance container DOWN; ~1.5 min) -- 78 plus three
# write-after-write arms (src/gapbench/cohbench2.cpp, coh2.hip): writer (X = tag - 1 + i), writer_rev (X = tag + i, the
# block order reversed, so another CU writes each element), check; ww_none has no invalidation between the two writes
# and the check, ww_check_agent only the check's (control, must be 0). ww_none nonzero: a CU serves a line it STORED
# after another CU rewrote it (write-allocate) -- the acquire rule (core-acq) must then treat lines written twice in a
# window as risky; zero: the rule may count only lines read and then written (acqrule.py's form). The 78 text: can a kernel whose AQL
# packet carries NO acquire fence read a stale cached line on gfx1201? (gapbench 75: the agent-scope acquire costs ~0.9 us an
# ordered dispatch, ~0.9 ms a step at ~1000 of them; dropping it is only safe if a consumer cannot see stale data.)
# cohbench (src/gapbench/coh.hip, cohbench.cpp), per trial on one queue, barrier-bit packets: reader (every CU caches X:
# vector loads, or s_load in the scalar arms), writer (X = tag + i, a new tag a trial), check (counts elements that are
# not tag + i). Arms: the check's acquire agent (today) / none, and the writer's too (none_both); vector and scalar.
# A nonzero count under none: a consumer CAN read a stale line through that cache -- the scope cannot be dropped for a
# consumer reading its producer's output through it. Zero bounds how often; it does not prove safety.
# Runs 300 trials x 3 sizes of X (2 KiB, 64 KiB, 1 MiB) with 512 blocks.
# DRYRUN=1 bash 82-cohbench2.sh : prints, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
B=${X3G_BIN:-$Q/bin-gap}
O=${X3G_OUT:-$HOME/gpu/ds41/exl3-test/runs/cohbench2}
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
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT cohbench2: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT cohbench2: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  for f in cohbench2 cohbench2-coh.hsaco; do [ -e "$B/$f" ] || { log "RESULT cohbench2: VIRHE $B/$f missing"; exit 1; }; done
  mkdir -p "$O"
else
  echo "+ mkdir -p $O"
fi
log "RESULT cohbench2: start IMAGE=$IMAGE card $CARD ($(md5sum $B/cohbench2 2>/dev/null | cut -c1-8) / $(md5sum $B/cohbench2-coh.hsaco 2>/dev/null | cut -c1-8)) -> $O"
for n in 512 16384 262144; do
  run "gpurun cohbench2-$$-$n 300 $DEV -v $B:/b:ro --entrypoint /b/cohbench2 $IMAGE /b/cohbench2-coh.hsaco 300 $n 512 > $O/cohbench2-n$n.txt 2>&1"
  [ "$DRY" = 1 ] || { grep '^RESULT cohbench' $O/cohbench2-n$n.txt | sed "s/^RESULT cohbench/RESULT cohbench2 X=$n/" | while read -r l; do log "$l"; done
                      [ "$(grep -c '^RESULT cohbench .*stale elements' $O/cohbench2-n$n.txt)" = 9 ] || log "RESULT cohbench2 X=$n: VIRHE not nine arms ($(tail -2 $O/cohbench2-n$n.txt | tr '\n' ' ' | cut -c1-300))"; }
done
log "RESULT cohbench2: done -- $O"
