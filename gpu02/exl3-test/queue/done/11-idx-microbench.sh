#!/bin/bash
# 11-idx-microbench.sh (6.11.2026; GPU window on gpu02, ONE card, production and every other radiance container DOWN) --
# the indexer kernels of the -idx image on one card at real contexts: ds41idx_bench (src/ds41idx_bench.cpp) loads the
# image's libr4d.so and, per case, scores a run of rows against ctx positions without keep (what deepseek41 issued at
# layers 24/28/32/36 before) and with keep (ds41-idx), selects top-512 through keep from both, checks on the device's
# results that every kept score is bit-identical, every other restricted score -inf and the selections identical, and
# times both forms (hipEvent medians).
# Cases: WMMA form (256-row runs, the engine's run height at MML 524288) at ctx 16k / 135k / 322k / 524k, kept blocks
# clustered (25 % a recent run, the rest shared, 10 % per-row jitter) and fully scattered (worst tile pairing); the
# decode form (1 row) at 135k / 322k.
# Model expectation (today's profile, ratio-1 layer = 1.798 ms per 1k ctx per 8192-token chunk = 0.22 ns a position a
# row): unrestricted 256 rows x 322k ~ 18 ms; restricted ~0.6-1.2 ms (16k-33k positions a row) -> 5-10 %.
# A MISMATCH line is a failed exactness check on the card: stop and report before any TP4 run of the -idx image.
# DRYRUN=1 bash 11-idx-microbench.sh : prints every command, runs and writes nothing.
# Estimate ~5 min.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
B=${IDXB_BIN:-$Q/bin}
O=${IDXB_OUT:-$HOME/gpu/ds41/exl3-test/runs/idx-micro}
L=${IDXB_LOG:-$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log}
IMAGE=${IDXB_IMAGE:-radiance-local:ds41-exl3-117-b4624774-f2f3c3f7-idx}
CARD=${CARD:-0}
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
bench() {   # bench NAME ARGS...
  local n=$1; shift
  run "docker run --rm --network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined" \
      "-e HIP_VISIBLE_DEVICES=$CARD -v $B:/b:ro --entrypoint /b/ds41idx_bench $IMAGE $* > $O/$n.txt 2>&1"
  [ "$DRY" = 1 ] || log "$(grep -m1 '^RESULT ds41idx' $O/$n.txt || echo "RESULT ds41idx $n: VIRHE $(tail -2 $O/$n.txt | tr '\n' ' ')")"
}
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT idx-micro: VIRHE a radiance container is running -- not now"; exit 1; }
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT idx-micro: VIRHE a KFD client is running"; exit 1; }
  [ -x "$B/ds41idx_bench" ] || { log "RESULT idx-micro: VIRHE $B/ds41idx_bench missing"; exit 1; }
  docker image inspect "$IMAGE" > /dev/null 2>&1 || { log "RESULT idx-micro: VIRHE image $IMAGE not here"; exit 1; }
  mkdir -p "$O"
else
  echo "+ mkdir -p $O"
fi
log "RESULT idx-micro: start IMAGE=$IMAGE card $CARD"
for c in 16384 135000 322000 524000; do bench wmma-$c --ctx $c --rows 256; done
for c in 135000 322000; do bench wmma-$c-scatter --ctx $c --rows 256 --scatter 1; done
for c in 135000 322000; do bench dec-$c --ctx $c --rows 1; done
log "RESULT idx-micro: done ($O)"
