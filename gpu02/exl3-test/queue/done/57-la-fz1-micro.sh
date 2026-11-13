#!/bin/bash
# 74-fz1-micro.sh (6.11.2026; GPU window on gpu02, ONE card, every radiance container DOWN) -- gate_up's prep and GEMM
# as ONE launch at K != 4 (radiance-exl3 exl3-fz1 0c2a5c5 on exl3-gu25c 978e264: D18's single-launch kernel, tickets
# 0 .. nslot - 1 rotate their slot's rows once and publish a flag, the GEMM tickets wait on it; the K loop pinned with
# sched_group_barrier, which keeps it free of 0-waits: device ISA K2 0x12-0x14, K2.5 / K3 0x8-0xd, 174-176 VGPRs).
# Why: in the engine's played tape every ordered dispatch waits ~3.76 us behind the one before (AQL stamps, gpu02
# runs/x117 mtp5-stamp: p50 3.76, 95 % in 3-4.5 us), and gate_up's prep is one (prep_row 4.94 us + 3.7..6.3 us gap a
# layer). perfmoe's back-to-back launches see a smaller gap than the tape, so this item measures the kernel side and
# the exactness; the engine A/B decides the step.
# Arms (perfmoe of the gu25 build, the 57 EP-4 routing):
#   base   exl3r4d-gu25c.so (two dispatches: prep, then the GEMM)
#   fz1    exl3r4d-fz1.so (one launch)          fz1off  exl3r4d-fz1.so with RADIANCE_EXL3_FUSE1=0 (two dispatches again)
# 1. exactness: chain y bytes (--rand-data --dump-y) base vs fz1, base vs fz1off, at K 2, 2.5, 3 and M 1, 4, 8 -> IDENTICAL
# 2. times: gate_up and chain at K 2 / 2.5 / 3, M 1 / 2 / 4 / 8 (base, fz1); fz1off at M 1 and 4 (must equal base)
# 3. the fixed part a call: gate_up M1 at --hidden 1280 / 2560 / 5120, K3 and K2 (base, fz1): slope and intercept
# References: 57-l-gu25b-micro (6.11. evening; gate_up K3 / K2.5 with the 12-byte load, the same gu25c .so as base);
# 70-fmt-micro gate_up M1: K2 81.41, K2.5 111.19 (PF 3), K3 133.85 us.
# Inputs (bin/ beside this file): perfmoe, exl3r4d-gu25c.so, exl3r4d-fz1.so.
# DRYRUN=1 bash 74-fz1-micro.sh : prints, runs and writes nothing. Estimate ~20 min (~95 perfmoe runs).
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
B=${X3G_BIN:-$Q/bin-gu25}
O=${X3G_OUT:-$HOME/gpu/ds41/exl3-test/runs/fz1-micro}
L=${X3G_LOG:-$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log}
IMAGE=${X3G_IMAGE:-radiance-local:ds41-exl3-117-abf7f9a0-f2f3c3f7}
CARD=${CARD:-0}
G0="--experts 96 --topk 6 --width 2304 --h-rot 1 --ep 4 --ep-rank -1"
GEOM="$G0 --hidden 5120 --warm-ms 500 --iters 4 --reps 5"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
# pm OUT ARM-ENV ARGS...: one perfmoe run; ARM-ENV is "-e VAR=..." or "-"
pm() { local out=$1 ev=$2; shift 2; [ "$ev" = - ] && ev=""; run "docker run --rm $DEV $ev -v $B:/b:ro -v $O:/o --entrypoint /b/perfmoe $IMAGE $* > $out 2>&1"; }
res() { grep -m1 '^RESULT exl3r4d' "$1" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="us"){print $(i-1); exit}}'; }
so()  { case $1 in base) echo "--exl3 /b/exl3r4d-gu25c.so" ;; fz1|fz1off) echo "--exl3 /b/exl3r4d-fz1.so" ;; esac; }
env_of() { [ "$1" = fz1off ] && echo "-e RADIANCE_EXL3_FUSE1=0" || echo -; }
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT fz1-micro: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT fz1-micro: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  for f in perfmoe exl3r4d-gu25c.so exl3r4d-fz1.so; do [ -e "$B/$f" ] || { log "RESULT fz1-micro: VIRHE $B/$f missing"; exit 1; }; done
  docker image inspect "$IMAGE" > /dev/null 2>&1 || { log "RESULT fz1-micro: VIRHE image $IMAGE not here"; exit 1; }
  mkdir -p "$O"
else
  echo "+ mkdir -p $O"
fi
log "RESULT fz1-micro: start IMAGE=$IMAGE card $CARD bin $B (gu25c $(md5sum $B/exl3r4d-gu25c.so 2>/dev/null | cut -c1-8), fz1 $(md5sum $B/exl3r4d-fz1.so 2>/dev/null | cut -c1-8)) -> $O"
# 1. exactness
for k in 2 2.5 3; do for m in 1 4 8; do
  for a in base fz1 fz1off; do
    pm $O/dump-k$k-m$m-$a.txt "$(env_of $a)" $(so $a) --chain --rand-data --dump-y /o/y-k$k-m$m-$a.bin --M $m --bits $k $G0 --hidden 5120 --warm-ms 50 --iters 1 --reps 1
  done
  if [ "$DRY" != 1 ]; then
    c1=$(cmp -s $O/y-k$k-m$m-base.bin $O/y-k$k-m$m-fz1.bin && echo IDENTICAL || echo DIFFERENT)
    c2=$(cmp -s $O/y-k$k-m$m-base.bin $O/y-k$k-m$m-fz1off.bin && echo IDENTICAL || echo DIFFERENT)
    [ -s $O/y-k$k-m$m-fz1.bin ] || c1="MISSING ($(tail -1 $O/dump-k$k-m$m-fz1.txt))"
    log "RESULT fz1-micro exact K=$k M=$m: base vs fz1 $c1, base vs fz1off $c2 ($(stat -c %s $O/y-k$k-m$m-fz1.bin 2>/dev/null) bytes)"
  fi
done; done
# 2. times
t() {   # t OP K M ARMS...
  local op=$1 k=$2 m=$3 line="" opa f a; shift 3
  [ "$op" = chain ] && opa="--chain" || opa="--op $op"
  for a in "$@"; do
    f=$O/$op-k$k-m$m-$a.txt
    pm $f "$(env_of $a)" $(so $a) $opa --M $m --bits $k $GEOM
    [ "$DRY" = 1 ] || line="$line $a $(res $f) us;"
  done
  [ "$DRY" = 1 ] || log "RESULT fz1-micro $op K=$k M=$m:$line"
}
for k in 3 2.5 2; do
  for m in 1 2 4 8; do
    case $m in 1|4) arms="base fz1 fz1off" ;; *) arms="base fz1" ;; esac
    t gate_up $k $m $arms
    t chain $k $m base fz1
  done
  [ "$DRY" != 1 ] && [ -e "$HOME/gpu/ds41/exl3-test/queue/STOP" ] && { log "RESULT fz1-micro: STOP file -- stopped after K=$k"; exit 0; }
done
# 3. the fixed part a call: hidden sweep
for ak in "base 3" "fz1 3" "base 2" "fz1 2"; do
  set -- $ak; a=$1 k=$2 pts=""
  for hd in 1280 2560 5120; do
    f=$O/hid-k$k-$a-h$hd.txt
    pm $f "$(env_of $a)" $(so $a) --op gate_up --M 1 --bits $k $G0 --hidden $hd --warm-ms 300 --iters 4 --reps 5
    [ "$DRY" = 1 ] || pts="$pts $hd:$(res $f)"
  done
  [ "$DRY" = 1 ] || log "RESULT fz1-micro hidden gate_up K=$k M=1 $a:$pts | $(echo $pts | tr ' ' '\n' | awk -F: 'NF==2 && $2!="" {n++; x=$1/1000; y=$2; sx+=x; sy+=y; sxx+=x*x; sxy+=x*y} END {if (n>=2) {b=(n*sxy-sx*sy)/(n*sxx-sx*sx); printf "slope %.2f us a 1k hidden, intercept %.1f us", b, (sy-b*sx)/n}}')"
done
log "RESULT fz1-micro: done -- $O"
