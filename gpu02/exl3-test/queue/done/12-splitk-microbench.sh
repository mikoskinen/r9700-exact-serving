#!/bin/bash
# 12-splitk-microbench.sh (6.11.2026; GPU window on gpu02, ONE card, production and every other radiance container DOWN)
# -- the EXL3 decode split-K rules E1 (down split-K 4 at one token, 2 at two) and E2 (gate_up split-K 2 at one token
# under EP) of radiance-exl3 branch exl3-splitk (kernels/exl3r4d/exl3r4d_sk.h), timed by perfmoe on DeepSeek V4.1
# Flash's per-rank geometry: 96 whole experts a rank of 384 (EP4), top-6, gate_up [4608, 5120], down [5120, 2304],
# h_rot 1, K 2 (layers 6-34) and 2.5 (0-5, 35-39). Routing: perfmoe --ep 4 --ep-rank -1 (each layer's draw over 384
# experts, the rank with the most live experts kept: the card the all-reduce after the MoE waits for).
#
# Arms on the SAME routing and the same perfmoe binary (ABAB inside perfmoe; arms run back to back):
#   base   exl3r4d-base.so   (f2f3c3f7, the shipped plugin; optional: skipped when absent)
#   sk0    exl3r4d-splitk.so without --ep-param   (must equal base: the refactor moved no launch)
#   sk1    exl3r4d-splitk.so with --ep-param       (ep_ranks 4: E1 + E2 engage)
#   nofit  exl3r4d-splitk-nofit.so with --ep-param (optional: a -DEXL3R4D_SK_DN_EP_FIT=0 build, down keeps split-K 4
#          at 1 token and 2 at 2-4 tokens even when a second round of workgroups runs)
# Ops: gate_up alone, down alone (the decode band's _m16 rows) and the chain (gate_up -> down, a layer), M = 1, 2, 4.
# Model expectation (PERF-REPORT §3, not measured): gate_up M=1 ~74 -> ~40 us, down M=1 ~48 -> ~17 us a layer;
# M = 2: down only (split-K 2); M = 4: no change (control).
#
# Inputs (copy to gpu02 before the window; nothing is built there):
#   $B/perfmoe                 the exl3-splitk device build's tests-gpu/perfmoe (it has --ep / --ep-param / --bits 2.5)
#   $B/exl3r4d-splitk.so       the exl3-splitk device build's exl3r4d.so
#   $B/exl3r4d-base.so         optional: f2f3c3f7's exl3r4d.so (radiance-local:ds41-exl3-117-abf7f9a0-f2f3c3f7)
#   $B/exl3r4d-splitk-nofit.so optional
#   IMAGE                      any 1.17 serving image on gpu02: only its ROCm runtime is used (perfmoe loads there)
# Device numerics are NOT this script: run tests/gpu/test_exl3r4d.py --suite device (deepseek_ep_splitk) for that.
#
# DRYRUN=1 bash 12-splitk-microbench.sh : prints every command, runs and writes nothing.
# Estimate ~12 min (2 K x (2 ops + chain) x 3 M x 3-4 arms, ~15-25 s each incl. warm-up and pool allocation).
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
B=${X3SK_BIN:-$Q/bin}
O=${X3SK_OUT:-$HOME/gpu/ds41/exl3-test/runs/splitk-micro}
L=${X3SK_LOG:-$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log}
IMAGE=${X3SK_IMAGE:-radiance-local:ds41-exl3-117-abf7f9a0-f2f3c3f7}
CARD=${CARD:-0}            # the HIP index of the one card used (0 = the first amdgpu device the container sees)
BITS=${BITS:-"2 2.5"}
MS=${MS:-"1 2 4"}
OPS=${OPS:-"gate_up down chain"}
GEOM="--experts 96 --topk 6 --hidden 5120 --width 2304 --h-rot 1 --ep 4 --ep-rank -1 --warm-ms 500 --iters 4 --reps 5"

log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
pm() {   # pm OUTFILE perfmoe-args...: one perfmoe run in IMAGE on card $CARD
  local out=$1; shift
  run "docker run --rm --network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined" \
      "-e HIP_VISIBLE_DEVICES=$CARD -v $B:/b:ro --entrypoint /b/perfmoe $IMAGE $* > $out 2>&1"
}
res() { grep -m1 '^RESULT exl3r4d' "$1" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="us"){print $(i-1); exit}}'; }

# Preconditions: the card idle (no KFD client, no radiance container), the inputs present.
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT splitk-micro: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done   # 6.11.: wait for the previous item's teardown
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT splitk-micro: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  for f in perfmoe exl3r4d-splitk.so; do [ -e "$B/$f" ] || { log "RESULT splitk-micro: VIRHE $B/$f missing"; exit 1; }; done
  docker image inspect "$IMAGE" > /dev/null 2>&1 || { log "RESULT splitk-micro: VIRHE image $IMAGE not here"; exit 1; }
  mkdir -p "$O"
else
  echo "+ mkdir -p $O"
fi
arms="sk0 sk1"
[ "$DRY" = 1 ] || [ -e "$B/exl3r4d-base.so" ] && arms="base $arms"
[ "$DRY" = 1 ] || [ -e "$B/exl3r4d-splitk-nofit.so" ] && arms="$arms nofit"
log "RESULT splitk-micro: start IMAGE=$IMAGE card $CARD arms [$arms] bits [$BITS] M [$MS] ops [$OPS] -> $O"

for bits in $BITS; do
  for op in $OPS; do
    for M in $MS; do
      line=""
      for arm in $arms; do
        case $arm in
          base)  so=exl3r4d-base.so;         xp="" ;;
          sk0)   so=exl3r4d-splitk.so;       xp="" ;;
          sk1)   so=exl3r4d-splitk.so;       xp="--ep-param" ;;
          nofit) so=exl3r4d-splitk-nofit.so; xp="--ep-param" ;;
        esac
        [ "$op" = chain ] && opa="--chain" || opa="--op $op"
        f=$O/$op-k$bits-m$M-$arm.txt
        pm "$f" --exl3 /b/$so $opa --M $M --bits $bits $GEOM $xp
        [ "$DRY" = 1 ] || line="$line $arm $(res "$f") us;"
      done
      [ "$DRY" = 1 ] || log "RESULT splitk-micro $op K=$bits M=$M:$line (model: gate_up M1 ~74 -> ~40, down M1 ~48 -> ~17 us)"
    done
  done
done
log "RESULT splitk-micro: done -- $O (per-arm perfmoe output; 'live experts a dispatch' in each file is the slowest rank's)"
