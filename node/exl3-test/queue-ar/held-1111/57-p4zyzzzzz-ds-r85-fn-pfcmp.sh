#!/bin/bash
# 176-fn-pfcmp.sh (queue item, 11.11.2026, queue-ar; gpu02, ONE card (HIP device 0), every radiance container DOWN; ~15 min)
# -- WHICH PLUGIN SLOWS DS PREFILL? (coordinator 11.11.: in i-e10, image ds41-e10-1d28b33c-76e24c49, the cold 28k copy's TTFT
# was 12.48 / 13.24 s at NST 0 (B / A, so not the e10_gd switch) and 14.12 / 15.31 at NST 5, against 10.27-11.32 s on every
# stack image overnight; decode unaffected.) One card, perfmoe b4ab7889 (the same harness for all three), DeepSeek V4.1's EP4
# rank geometry (96 experts of 384, top 6, hidden 5120, n_ff 2304), the PREFILL band chain (_pf: map, prep, gate_up, down) at
# M 2048 and 4096 (DS's MNBT 4160 pieces), K 2 and 2.5, three exl3r4d.so: stack (ds41-stack-fe030c3b-5c13e555-l's,
# 4097ea92: exl3-main 5c13e55), e10 (ds41-e10-1d28b33c-76e24c49-l's, af12c979: 5c13e55 + the fn4 commits + the DS variant, built
# in radiance-ds41-build), dsgd (bin-dsgd's 2c66cff3: the same sources built in radiance-build-c16), order stack e10 dsgd dsgd
# e10 stack; then the decode band chain at M 8 (stack vs e10). Measurement only.
# DRYRUN=1 bash 176-fn-pfcmp.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
R=$HOME/gpu/ds41/exl3-test/runs; L=$R/ds41-exl3.log; O=$R/fn4/pfcmp
B=; for c in "$Q/bin-pfcmp" "$Q/../queue-ar/bin-pfcmp" "$HOME/gpu/ds41/exl3-test/queue-ar/bin-pfcmp"; do [ -z "$B" ] && [ -e "$c/perfmoe" ] && B=$(cd "$c" && pwd); done
B=${B:-$HOME/gpu/ds41/exl3-test/queue-ar/bin-pfcmp}
IMAGE=${X176_IMAGE:-radiance-local:c16-x3c}; IMGID=29e7b84dcba5
CARD=${CARD:-0}
GEO="--tp 1 --ep 4 --ep-rank -1 --h-rot 1 --experts 96 --topk 6 --hidden 5120 --width 2304 --pooled 0 --ceiling 640 --warm-ms 500 --chain --rand-data"
TIM="--iters ${X176_ITERS:-3} --reps ${X176_REPS:-3}"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD --user $(id -u):$(id -g) --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) -e HOME=/tmp -w /tmp"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
pm() {   # pm NAME VARIANT perfmoe-args...
  local nm=$1 v=$2; shift 2
  local cn=fn-pfc-$$-$nm
  if [ "$DRY" = 1 ]; then echo "+ docker run ... $IMAGE perfmoe --exl3 /b/exl3r4d-$v.so $* > $O/$nm.txt"; return 0; fi
  timeout -k 15 300 docker run --rm --init --stop-timeout 10 --name $cn $DEV -v $B:/b:ro -v $O:/o --entrypoint /b/perfmoe $IMAGE \
    --exl3 /b/exl3r4d-$v.so "$@" > $O/$nm.txt 2>&1
  local rc=$?; docker rm -f $cn > /dev/null 2>&1; return $rc
}
usl() { local v; v=$(grep -m1 "^RESULT $2" $O/$1.txt 2>/dev/null | sed -nE 's/.*M=[0-9]+ +([0-9.]+) us.*/\1/p'); echo ${v:-NA}; }
err() { grep -m1 -iE "error|refus|die|fail" $O/$1.txt 2>/dev/null | cut -c1-100; }
log "RESULT fn4 176-fn-pfcmp: start IMAGE=$IMAGE card HIP $CARD bins $B (DS EP4 rank geometry: stack vs e10 vs dsgd exl3r4d.so)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 176: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ "$(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)" = $IMGID ] || { log "RESULT fn4 176: VIRHE image $IMAGE is not $IMGID"; exit 1; }
  ( cd $B && md5sum -c --quiet MD5SUMS ) || { log "RESULT fn4 176: VIRHE bin md5s ($B/MD5SUMS)"; exit 1; }
  rm -rf $O; mkdir -p $O; chmod 777 $O
fi
for K in 2 2.5; do
  for M in 2048 4096; do
    n=k$K-m$M
    i=0; for v in stack e10 dsgd dsgd e10 stack; do i=$((i + 1)); pm $n-$i-$v $v $GEO --chain-band _pf $TIM --bits $K --M $M; done
    [ "$DRY" = 1 ] || log "RESULT fn4 176 prefill chain _pf K $K M $M us: stack $(usl $n-1-stack 'exl3r4d chain') / $(usl $n-6-stack 'exl3r4d chain') | e10 $(usl $n-2-e10 'exl3r4d chain') / $(usl $n-5-e10 'exl3r4d chain') | dsgd $(usl $n-3-dsgd 'exl3r4d chain') / $(usl $n-4-dsgd 'exl3r4d chain') $(err $n-1-stack)$(err $n-2-e10)"
  done
done
for K in 2 2.5; do
  n=d-k$K-m8
  i=0; for v in stack e10 e10 stack; do i=$((i + 1)); pm $n-$i-$v $v $GEO --chain-band _m16 $TIM --bits $K --M 8; done
  [ "$DRY" = 1 ] || log "RESULT fn4 176 decode chain _m16 K $K M 8 us: stack $(usl $n-1-stack 'exl3r4d chain') / $(usl $n-4-stack 'exl3r4d chain') | e10 $(usl $n-2-e10 'exl3r4d chain') / $(usl $n-3-e10 'exl3r4d chain')"
done
log "RESULT fn4 176-fn-pfcmp: done"
