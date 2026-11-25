#!/bin/bash
# 175-fn-gdwpe.sh (queue item, 11.11.2026, queue-ar; gpu02, ONE card (HIP device 0), every radiance container DOWN; ~15 min)
# -- E10b SCREEN 1: the fused decode chain with its registers capped (EXL3R4D_GD_WPE, exl3-main37 fn4-dsgd 18def52; the
# split kernels unchanged; static metadata: gdw12 120 VGPRs + 87-91 spilled, gdw16 96 + 118-139 spilled). 172: the DS variant is exact but slower than the split pair at every M; the static metadata puts
# the fused kernel at ~180 VGPRs (its down units at the gate_up path's budget) vs the split down's ~94. Arms: dsgd (no cap,
# 172's), gdw12 (<= 128 VGPRs), gdw16 (<= 96). DS EP4 rank geometry, K 2 and 2.5, limit 10, M 3 / 4 / 6 / 8 / 16, and
# Flash-Next's EP4 K4 at M 4 / 8 / 16 (where the uncapped fused op wins); perfmoe --gd = split vs fused, y byte for byte,
# two runs each. If a cap beats the split pair at M 3-8 on DS: in-engine A/Bs (DS c1 / c4 / c8, Flash-Next c4 / c16).
# DRYRUN=1 bash 175-fn-gdwpe.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
R=$HOME/gpu/ds41/exl3-test/runs; L=$R/ds41-exl3.log; O=$R/fn4/gdwpe
B=; for c in "$Q/bin-gdwpe" "$Q/../queue-ar/bin-gdwpe" "$HOME/gpu/ds41/exl3-test/queue-ar/bin-gdwpe"; do [ -z "$B" ] && [ -e "$c/perfmoe" ] && B=$(cd "$c" && pwd); done
B=${B:-$HOME/gpu/ds41/exl3-test/queue-ar/bin-gdwpe}
IMAGE=${X175_IMAGE:-radiance-local:c16-x3c}; IMGID=29e7b84dcba5
CARD=${CARD:-0}
GEO="--tp 1 --ep 4 --ep-rank -1 --h-rot 1 --experts 96 --topk 6 --hidden 5120 --width 2304 --pooled 0 --ceiling 640 --warm-ms 500 --chain --chain-band _m16 --rand-data"
TIM="--iters ${X175_ITERS:-10} --reps ${X175_REPS:-5}"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD --user $(id -u):$(id -g) --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) -e HOME=/tmp -w /tmp"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
pm() {   # pm NAME VARIANT perfmoe-args...
  local nm=$1 v=$2; shift 2
  local cn=fn-gdw-$$-$nm
  if [ "$DRY" = 1 ]; then echo "+ docker run ... $IMAGE perfmoe --exl3 /b/exl3r4d-$v.so $* > $O/$nm.txt"; return 0; fi
  timeout -k 15 300 docker run --rm --init --stop-timeout 10 --name $cn $DEV -v $B:/b:ro -v $O:/o --entrypoint /b/perfmoe $IMAGE \
    --exl3 /b/exl3r4d-$v.so "$@" > $O/$nm.txt 2>&1
  local rc=$?; docker rm -f $cn > /dev/null 2>&1; return $rc
}
usl() { local v; v=$(grep -m1 "^RESULT $2" $O/$1.txt 2>/dev/null | sed -nE 's/.*M=[0-9]+ +([0-9.]+) us.*/\1/p'); echo ${v:-NA}; }
log "RESULT fn4 175-fn-gdwpe: start IMAGE=$IMAGE card HIP $CARD bins $B (DS EP4 rank geometry, decode band)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 175: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ "$(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)" = $IMGID ] || { log "RESULT fn4 175: VIRHE image $IMAGE is not $IMGID"; exit 1; }
  ( cd $B && md5sum -c --quiet MD5SUMS ) || { log "RESULT fn4 175: VIRHE bin md5s ($B/MD5SUMS)"; exit 1; }
  rm -rf $O; mkdir -p $O; chmod 777 $O
fi
GEOF="--tp 1 --ep 4 --ep-rank -1 --h-rot 1 --experts 128 --topk 10 --hidden 2560 --width 640 --group 4 --group-mix 0.53 --pooled 0 --ceiling 640 --warm-ms 500 --chain --chain-band _m16 --rand-data"
one() {   # one NAME VARIANT GEO-ARGS...
  local nm=$1 v=$2; shift 2
  pm $nm-a $v "$@" $TIM --gd; pm $nm-b $v "$@" $TIM --gd
  [ "$DRY" = 1 ] || log "RESULT fn4 175 $nm ($v): split chain $(usl $nm-a 'exl3r4d chain') / $(usl $nm-b 'exl3r4d chain') us, fused gd $(usl $nm-a 'exl3r4d gd') / $(usl $nm-b 'exl3r4d gd') us | $(grep -hE 'GD-CHECK' $O/$nm-a.txt $O/$nm-b.txt | sort -u | head -2 | tr '\n' ' ' | cut -c1-120)"
}
for K in 2 2.5; do
  for M in 3 4 6 8 16; do
    for v in dsgd gdw12 gdw16; do one k$K-m$M-$v $v $GEO --bits $K --limit 10 --M $M; done
  done
done
for M in 4 8 16; do
  for v in dsgd gdw12 gdw16; do one fn-k4-m$M-$v $v $GEOF --bits 4 --M $M; done
done
log "RESULT fn4 175-fn-gdwpe: done"
