#!/bin/bash
# 71-gu25b-micro.sh (6.11.2026; GPU window on gpu02, ONE card, every radiance container DOWN) -- the 2.5-bit gate_up's
# 12-byte tile load (radiance-exl3 exl3-gu25b 4670c85, on exl3-gu25 5dd4abf): a lane's three tile words as one
# raw_buffer_load_b96 instead of four b32 (+6 VALU a tile for the selects and one readlane), which brings PF 4 back
# without the K loop's 0-waits (device ISA: waits 0x7..0xf, 180 VGPRs, no spills). Down and K = 2 are unchanged. Same
# words into the same decode -> same bits (host check of all 32 lanes' selection).
# K = 3 (added 6.11. evening; exl3-gu25c 978e264 on 4670c85): K = 3's gate_up loop was K = 2.5's before the fix (40
# loads, 328 VALU, six s_wait_loadcnt 0 an iteration, 178 VGPRs) and 70-fmt-micro had it at 133.85 us (K2 81.41, K2.5
# 111.19 at PF 3, K4 80.08). Its runs end 24 bits apart, so the same 12-byte load serves it: device ISA 16 loads, 376
# VALU, waits 0x6-0xf, 180 VGPRs, no spills; every other kernel's device code is the same as 4670c85's (gu25b.so).
# b96_exact() proves each lane's selection at compile time (K 2, 2.5, 3).
# Arms (perfmoe of the gu25 build, same EP-4 routing as 57-gu25-micro):
#   base  exl3r4d-base.so (f2f3c3f7)   gu  exl3r4d-gu25.so (PF 3)   b96  exl3r4d-gu25b.so (b96 at K2.5, PF 4)
#   b96c  exl3r4d-gu25c.so (b96 at K2.5 and K3, PF 4)   b96sk1 / b96csk1  the same with --ep-param (E1/E2)
# 1. exactness: chain y bytes (--rand-data --dump-y) at K 2.5, 2 and 3, M 1 and 4: base vs b96, gu vs b96, base vs b96c,
#    b96 vs b96c -> IDENTICAL
# 2. times: gate_up K2.5 M 1/2/4, chain K2.5 M 1/2/4 (base gu b96 b96c b96sk1); gate_up K3 M 1/2/4, chain K3 M 1/2/4
#    (base b96c b96csk1); controls gate_up K2 M1 (base b96 b96c), down K2.5 M1 (base b96), down K3 M1 (base b96c)
# 3. the fixed part a call: gate_up M1 at --hidden 1280 / 2560 / 5120 (K2.5 base, gu, b96; K3 base, b96c; K2 base):
#    slope (us a 1k of hidden) and intercept by least squares over the three points -- do K2.5's and K3's fall to K2's?
# References 57-gu25-micro (6.11.): gate_up K2.5 M1 base 132.58 / gu 113.99 / sk1 127.29 / gusk1 110.40 us; M2 181.09 /
# 161.12; M4 284.35 / 256.81; chain K2.5 M1 175.61 / 155.30; gate_up K2 M1 82.31; 54 phase C K2 hidden 1280/2560/5120:
# 28.01 / 46.52 / 83.91 us. Target (VALU a tile equal at K2 and K2.5): gate_up K2.5 M1 ~1.0-1.1 x K2's 82 us; K3 at
# most K2.5-b96's time x 1.2 (its bytes).
# Inputs (bin/ beside this file): perfmoe, exl3r4d-base.so, exl3r4d-gu25.so, exl3r4d-gu25b.so, exl3r4d-gu25c.so.
# DRYRUN=1 bash 71-gu25b-micro.sh : prints, runs and writes nothing. Estimate ~19 min (97 perfmoe runs).
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
B=${X3G_BIN:-$Q/bin-gu25}
O=${X3G_OUT:-$HOME/gpu/ds41/exl3-test/runs/gu25b-micro}
L=${X3G_LOG:-$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log}
IMAGE=${X3G_IMAGE:-radiance-local:ds41-exl3-117-abf7f9a0-f2f3c3f7}
CARD=${CARD:-0}
G0="--experts 96 --topk 6 --width 2304 --h-rot 1 --ep 4 --ep-rank -1"
GEOM="$G0 --hidden 5120 --warm-ms 500 --iters 4 --reps 5"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
pm() { local out=$1; shift; run "docker run --rm $DEV -v $B:/b:ro -v $O:/o --entrypoint /b/perfmoe $IMAGE $* > $out 2>&1"; }
res() { grep -m1 '^RESULT exl3r4d' "$1" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="us"){print $(i-1); exit}}'; }
arm() { case $1 in base) echo "--exl3 /b/exl3r4d-base.so" ;; gu) echo "--exl3 /b/exl3r4d-gu25.so" ;;
                   b96) echo "--exl3 /b/exl3r4d-gu25b.so" ;; b96sk1) echo "--exl3 /b/exl3r4d-gu25b.so --ep-param" ;;
                   b96c) echo "--exl3 /b/exl3r4d-gu25c.so" ;; b96csk1) echo "--exl3 /b/exl3r4d-gu25c.so --ep-param" ;; esac; }
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT gu25b-micro: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT gu25b-micro: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  for f in perfmoe exl3r4d-base.so exl3r4d-gu25.so exl3r4d-gu25b.so exl3r4d-gu25c.so; do [ -e "$B/$f" ] || { log "RESULT gu25b-micro: VIRHE $B/$f missing"; exit 1; }; done
  docker image inspect "$IMAGE" > /dev/null 2>&1 || { log "RESULT gu25b-micro: VIRHE image $IMAGE not here"; exit 1; }
  mkdir -p "$O"
else
  echo "+ mkdir -p $O"
fi
log "RESULT gu25b-micro: start IMAGE=$IMAGE card $CARD bin $B (gu25b $(md5sum $B/exl3r4d-gu25b.so 2>/dev/null | cut -c1-8), gu25c $(md5sum $B/exl3r4d-gu25c.so 2>/dev/null | cut -c1-8)) -> $O"
# 1. exactness
for k in 2.5 2 3; do for m in 1 4; do
  for a in base gu b96 b96c; do
    pm $O/dump-k$k-m$m-$a.txt $(arm $a) --chain --rand-data --dump-y /o/y-k$k-m$m-$a.bin --M $m --bits $k $G0 --hidden 5120 --warm-ms 50 --iters 1 --reps 1
  done
  if [ "$DRY" != 1 ]; then
    c1=$(cmp -s $O/y-k$k-m$m-base.bin $O/y-k$k-m$m-b96.bin && echo IDENTICAL || echo DIFFERENT)
    c2=$(cmp -s $O/y-k$k-m$m-gu.bin $O/y-k$k-m$m-b96.bin && echo IDENTICAL || echo DIFFERENT)
    c3=$(cmp -s $O/y-k$k-m$m-base.bin $O/y-k$k-m$m-b96c.bin && echo IDENTICAL || echo DIFFERENT)
    c4=$(cmp -s $O/y-k$k-m$m-b96.bin $O/y-k$k-m$m-b96c.bin && echo IDENTICAL || echo DIFFERENT)
    [ -s $O/y-k$k-m$m-b96.bin ] || c1="MISSING ($(tail -1 $O/dump-k$k-m$m-b96.txt))"
    [ -s $O/y-k$k-m$m-b96c.bin ] || c3="MISSING ($(tail -1 $O/dump-k$k-m$m-b96c.txt))"
    log "RESULT gu25b-micro exact K=$k M=$m: base vs b96 $c1, gu vs b96 $c2, base vs b96c $c3, b96 vs b96c $c4 ($(stat -c %s $O/y-k$k-m$m-b96c.bin 2>/dev/null) bytes)"
  fi
done; done
# 2. times
t() {   # t OP K M ARMS...
  local op=$1 k=$2 m=$3 line="" opa f a; shift 3
  [ "$op" = chain ] && opa="--chain" || opa="--op $op"
  for a in "$@"; do
    f=$O/$op-k$k-m$m-$a.txt
    pm $f $(arm $a) $opa --M $m --bits $k $GEOM
    [ "$DRY" = 1 ] || line="$line $a $(res $f) us;"
  done
  [ "$DRY" = 1 ] || log "RESULT gu25b-micro $op K=$k M=$m:$line"
}
for m in 1 2 4; do t gate_up 2.5 $m base gu b96 b96c b96sk1; done
for m in 1 2 4; do t chain 2.5 $m base gu b96 b96c b96sk1; done
for m in 1 2 4; do t gate_up 3 $m base b96c b96csk1; done
for m in 1 2 4; do t chain 3 $m base b96c b96csk1; done
t gate_up 2 1 base b96 b96c
t down 2.5 1 base b96
t down 3 1 base b96c
# 3. the fixed part a call: hidden sweep
for ak in "base 2.5" "gu 2.5" "b96 2.5" "base 3" "b96c 3" "base 2"; do
  set -- $ak; a=$1 k=$2 pts=""
  for hd in 1280 2560 5120; do
    f=$O/hid-k$k-$a-h$hd.txt
    pm $f $(arm $a) --op gate_up --M 1 --bits $k $G0 --hidden $hd --warm-ms 300 --iters 4 --reps 5
    [ "$DRY" = 1 ] || pts="$pts $hd:$(res $f)"
  done
  [ "$DRY" = 1 ] || log "RESULT gu25b-micro hidden gate_up K=$k M=1 $a:$pts | $(echo $pts | tr ' ' '\n' | awk -F: 'NF==2 && $2!="" {n++; x=$1/1000; y=$2; sx+=x; sy+=y; sxx+=x*x; sxy+=x*y} END {if (n>=2) {b=(n*sxy-sx*sy)/(n*sxx-sx*sx); printf "slope %.2f us a 1k hidden, intercept %.1f us", b, (sy-b*sx)/n}}')"
done
log "RESULT gu25b-micro: done -- $O (refs 57-gu25-micro: gate_up K2.5 M1 base 132.58 / gu 113.99 us, K2 82.31; 70-fmt-micro K3 133.85)"
