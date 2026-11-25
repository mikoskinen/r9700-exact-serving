#!/bin/bash
# 80-arpcn-selftest.sh (6.11.2026 night; GPU window on gpu02, ALL FOUR cards, every radiance container DOWN, no server) --
# the rank-symmetric host test of fusion 2 before any engine run: r4d_selftest's ar_mhc_post_collapse_rmsnorm_4rank case
# (libr4d of ds41-pcnw ce26b61): four rank threads, one card each, run the fused collective twice on the same inputs;
# the oracle is x folded in ascending rank order and rounded once (ar_vsum), then the single-card
# mhc_post_collapse_rmsnorm_bf16 row over that x on card 0. Every rank's x, out, xc and y must be those bytes, both times
# (the row counters reset themselves). Then the two cases that share its instance and protocol, as regressions:
# ar_oneshot_4rank_exact and ar_hc_write_4rank.
# Hang protection: each case is its own process in its own named container under gpurun (a hard limit, SIGKILL, then
# docker rm -f); the kernel's only wait is the one-shot's handshake, which traps past R4D_AR_SPIN_MAX.
# Inputs: bin/r4d_selftest-arpcn (built beside the libr4d it tests) and the image's libr4d.so.
# DRYRUN=1 bash 80-arpcn-selftest.sh : prints, runs and writes nothing. Estimate ~3 min.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
B=${X3G_BIN:-$Q/bin-gap}
O=${X3G_OUT:-$HOME/gpu/ds41/exl3-test/runs/arpcn-selftest}
L=${X3G_LOG:-$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log}
IMAGE=${X3AP:-radiance-local:ds41-exl3-117-ce26b611-f2f3c3f7-arpcn}
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ulimit memlock=-1 --ipc=host -e HIP_VISIBLE_DEVICES=0,1,2,3"
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
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT arpcn-selftest: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT arpcn-selftest: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  [ -x "$B/r4d_selftest-arpcn" ] || { log "RESULT arpcn-selftest: VIRHE $B/r4d_selftest-arpcn missing"; exit 1; }
  docker image inspect "$IMAGE" > /dev/null 2>&1 || { log "RESULT arpcn-selftest: VIRHE image $IMAGE not here"; exit 1; }
  mkdir -p "$O"
else
  echo "+ mkdir -p $O"
fi
log "RESULT arpcn-selftest: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) selftest $(md5sum $B/r4d_selftest-arpcn 2>/dev/null | cut -c1-8) -> $O"
for c in ar_mhc_post_collapse_rmsnorm_4rank ar_oneshot_4rank_exact ar_hc_write_4rank; do
  run "gpurun st-$$-${c:0:12} 600 $DEV -v $B:/b:ro --entrypoint /b/r4d_selftest-arpcn $IMAGE /opt/radiance/share/radiance/kernels/libr4d.so --case $c > $O/$c.txt 2>&1"
  rc=$?
  if [ "$DRY" != 1 ]; then
    nf=$(grep -c 'FAIL' $O/$c.txt); nok=$(grep -cE '^ *(ok|PASS)' $O/$c.txt); ns=$(grep -ciE '^ *skip' $O/$c.txt)
    log "RESULT arpcn-selftest $c: rc $rc, $nok ok, $nf FAIL, $ns skip$([ $nf -gt 0 ] && echo " -- first: $(grep -m1 FAIL $O/$c.txt | cut -c1-300)")$([ $rc = 124 ] || [ $rc = 137 ] && echo ' -- STOPPED by the limit (hang)')"
  fi
done
log "RESULT arpcn-selftest: done -- $O"
