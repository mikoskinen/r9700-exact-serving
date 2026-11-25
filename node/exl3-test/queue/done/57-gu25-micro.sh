#!/bin/bash
# 63-gu25-micro.sh (6.11.2026; GPU window on gpu02, ONE card, every radiance container DOWN) -- the 2.5-bit gate_up fix of
# radiance-exl3 branch exl3-gu25 (5dd4abf, on exl3-splitk 9de0167): gate_up at a half-integer K prefetches 3 k-steps
# (EXL3_PF_GU_HALF) instead of 4. Static ISA (build/gu25/isa): at PF 4 the K2.5 gate_up K loop waits s_wait_loadcnt 0 six
# times an iteration (every load drained each 4 k-steps); at PF 3 no wait in the loop reaches 0. K = 2 and all down
# kernels are unchanged (identical loop statistics). Same instructions on the same operands in the same order -> same bits.
#
# Arms (perfmoe of the gu25 build; same routing, DeepSeek V4.1 Flash per-rank geometry, EP-4, slowest rank):
#   base   exl3r4d-base.so   (f2f3c3f7, today's shipped plugin)
#   gu     exl3r4d-gu25.so   without --ep-param (base's launch rules: PF 3 alone)
#   sk1    exl3r4d-splitk.so with --ep-param  (E1/E2 alone, 12's sk1 arm)
#   gusk1  exl3r4d-gu25.so   with --ep-param  (E1/E2 + PF 3)
# Ops: gate_up K2.5 M 1/2/4 (the fix), chain K2.5 M 1/2/4 (a layer), controls gate_up K2 M1 and down K2.5 M1 (base and
# gu must agree within noise). References (12-splitk-microbench 6.11., same card class): gate_up K2.5 M1 base 133.08 us,
# sk1 126.24; K2 M1 81.74; chain K2.5 M1 173.76 / sk1 163.11. Expected (arithmetic, not measured): gate_up K2.5 M1 toward
# 1.25 x K2's (~100 us), chain K2.5 M1 ~-30 us.
# Exactness on the card: perfmoe --chain --rand-data --dump-y (layer 0's chain y) base vs gu and sk1 vs gusk1 at K2.5 and
# K2, M 1 and 4: byte comparison (cmp). A DIFFERENT line means the fix changed a bit -- report, do not ship.
#
# Inputs (copy to gpu02 beside this file, bin/): perfmoe (gu25 build), exl3r4d-gu25.so, exl3r4d-base.so,
# exl3r4d-splitk.so (12's). IMAGE: any 1.17 serving image on gpu02 (only its ROCm runtime is used).
# DRYRUN=1 bash 63-gu25-micro.sh : prints every command, runs and writes nothing. Estimate ~10 min.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
B=${X3G_BIN:-$Q/bin-gu25}
O=${X3G_OUT:-$HOME/gpu/ds41/exl3-test/runs/gu25-micro}
L=${X3G_LOG:-$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log}
IMAGE=${X3G_IMAGE:-radiance-local:ds41-exl3-117-abf7f9a0-f2f3c3f7}
CARD=${CARD:-0}
GEOM="--experts 96 --topk 6 --hidden 5120 --width 2304 --h-rot 1 --ep 4 --ep-rank -1 --warm-ms 500 --iters 4 --reps 5"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
pm() {   # pm OUTFILE perfmoe-args...
  local out=$1; shift
  run "docker run --rm $DEV -v $B:/b:ro -v $O:/o --entrypoint /b/perfmoe $IMAGE $* > $out 2>&1"
}
res() { grep -m1 '^RESULT exl3r4d' "$1" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="us"){print $(i-1); exit}}'; }
arm() { case $1 in base) echo "--exl3 /b/exl3r4d-base.so" ;; gu) echo "--exl3 /b/exl3r4d-gu25.so" ;;
                   sk1) echo "--exl3 /b/exl3r4d-splitk.so --ep-param" ;; gusk1) echo "--exl3 /b/exl3r4d-gu25.so --ep-param" ;; esac; }

if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT gu25-micro: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done   # KFD teardown race
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT gu25-micro: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  for f in perfmoe exl3r4d-gu25.so exl3r4d-base.so exl3r4d-splitk.so; do [ -e "$B/$f" ] || { log "RESULT gu25-micro: VIRHE $B/$f missing"; exit 1; }; done
  docker image inspect "$IMAGE" > /dev/null 2>&1 || { log "RESULT gu25-micro: VIRHE image $IMAGE not here"; exit 1; }
  mkdir -p "$O"
else
  echo "+ mkdir -p $O"
fi
log "RESULT gu25-micro: start IMAGE=$IMAGE card $CARD bin $B ($(md5sum $B/exl3r4d-gu25.so 2>/dev/null | cut -c1-8)) -> $O"

# 1. exactness on the card: chain y bytes, base vs gu and sk1 vs gusk1
for k in 2.5 2; do for m in 1 4; do
  for a in base gu sk1 gusk1; do
    pm $O/dump-k$k-m$m-$a.txt $(arm $a) --chain --rand-data --dump-y /o/y-k$k-m$m-$a.bin --M $m --bits $k \
       --experts 96 --topk 6 --hidden 5120 --width 2304 --h-rot 1 --ep 4 --ep-rank -1 --warm-ms 50 --iters 1 --reps 1
  done
  if [ "$DRY" != 1 ]; then
    c1=$(cmp -s $O/y-k$k-m$m-base.bin $O/y-k$k-m$m-gu.bin && echo IDENTICAL || echo DIFFERENT)
    c2=$(cmp -s $O/y-k$k-m$m-sk1.bin $O/y-k$k-m$m-gusk1.bin && echo IDENTICAL || echo DIFFERENT)
    [ -s $O/y-k$k-m$m-base.bin ] || c1="MISSING (no dump: $(tail -1 $O/dump-k$k-m$m-base.txt))"
    log "RESULT gu25-micro exact K=$k M=$m: base vs gu $c1, sk1 vs gusk1 $c2 ($(stat -c %s $O/y-k$k-m$m-base.bin 2>/dev/null) bytes)"
  fi
done; done

# 2. timing
t() {   # t OP K M ARMS...
  local op=$1 k=$2 m=$3 line="" opa f a; shift 3
  [ "$op" = chain ] && opa="--chain" || opa="--op $op"
  for a in "$@"; do
    f=$O/$op-k$k-m$m-$a.txt
    pm $f $(arm $a) $opa --M $m --bits $k $GEOM
    [ "$DRY" = 1 ] || line="$line $a $(res $f) us;"
  done
  [ "$DRY" = 1 ] || log "RESULT gu25-micro $op K=$k M=$m:$line"
}
for m in 1 2 4; do t gate_up 2.5 $m base gu sk1 gusk1; done
for m in 1 2 4; do t chain 2.5 $m base gu sk1 gusk1; done
t gate_up 2 1 base gu
t down 2.5 1 base gu
log "RESULT gu25-micro: done -- $O (refs 12-splitk 6.11.: gate_up K2.5 M1 base 133.08 / sk1 126.24 us, K2 M1 81.74; chain K2.5 M1 173.76 / 163.11)"
