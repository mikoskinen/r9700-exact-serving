#!/bin/bash
# 172-fn-dsgd.sh (queue item, 11.11.2026, queue-ar; gpu02, ONE card (HIP device 0), every radiance container DOWN; ~15 min)
# -- THE DS E10 VARIANT'S GD-CHECK (coordinator 10.11., DS wants it once E10 confirmed: 156b c4 B/A 1.0503, ms/step -5 %).
# exl3-main37 fn4-dsgd a3e0f1c: exl3r4d_fused_kernel templated on gate_up / down K (<2,2>, <2.5,2.5>, the plain form, two
# classes) + interface v11 (moe_exl3_gate_up_down takes gate_up's SwiGLU clamp `limit`); perfmoe --bits 2.5 / --limit.
# DeepSeek V4.1's EP4 rank geometry (dsm-eport.sh's GEO_DS: 96 experts a rank of 384, top 6, hidden 5120, n_ff 2304), the
# decode band (_m16): for K 2 and 2.5, limit 10 (DS) and 0, M 3 / 4 / 6 / 8 / 12 / 16 (DS's 1-user round 6 rows, NST 0 c4 /
# c8 4-8 rows): perfmoe --gd = the split chain and the fused op on the same tables, routing and x, y byte for byte
# (GD-CHECK) + both timed (two runs each). The GD_DEBUG build logs every fused launch that falls back to the split
# kernels (a fallback would make the check vacuous): its stderr is counted. Image c16-x3c's runtime (the plugin and perfmoe
# come from bin-dsgd).
# DRYRUN=1 bash 172-fn-dsgd.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
R=$HOME/gpu/ds41/exl3-test/runs; L=$R/ds41-exl3.log; O=$R/fn4/dsgd
B=; for c in "$Q/bin-dsgd" "$Q/../queue-ar/bin-dsgd" "$HOME/gpu/ds41/exl3-test/queue-ar/bin-dsgd"; do [ -z "$B" ] && [ -e "$c/perfmoe" ] && B=$(cd "$c" && pwd); done
B=${B:-$HOME/gpu/ds41/exl3-test/queue-ar/bin-dsgd}
IMAGE=${X172_IMAGE:-radiance-local:c16-x3c}; IMGID=29e7b84dcba5
CARD=${CARD:-0}
GEO="--tp 1 --ep 4 --ep-rank -1 --h-rot 1 --experts 96 --topk 6 --hidden 5120 --width 2304 --pooled 0 --ceiling 640 --warm-ms 500 --chain --chain-band _m16 --rand-data"
TIM="--iters ${X172_ITERS:-10} --reps ${X172_REPS:-5}"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD --user $(id -u):$(id -g) --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) -e HOME=/tmp -w /tmp"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
pm() {   # pm NAME VARIANT perfmoe-args...
  local nm=$1 v=$2; shift 2
  local cn=fn-dsgd-$$-$nm
  if [ "$DRY" = 1 ]; then echo "+ docker run ... $IMAGE perfmoe --exl3 /b/exl3r4d-$v.so $* > $O/$nm.txt"; return 0; fi
  timeout -k 15 300 docker run --rm --init --stop-timeout 10 --name $cn $DEV -v $B:/b:ro -v $O:/o --entrypoint /b/perfmoe $IMAGE \
    --exl3 /b/exl3r4d-$v.so "$@" > $O/$nm.txt 2>&1
  local rc=$?; docker rm -f $cn > /dev/null 2>&1; return $rc
}
usl() { local v; v=$(grep -m1 "^RESULT $2" $O/$1.txt 2>/dev/null | sed -nE 's/.*M=[0-9]+ +([0-9.]+) us.*/\1/p'); echo ${v:-NA}; }
log "RESULT fn4 172-fn-dsgd: start IMAGE=$IMAGE card HIP $CARD bins $B (DS EP4 rank geometry, decode band)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 172: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ "$(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)" = $IMGID ] || { log "RESULT fn4 172: VIRHE image $IMAGE is not $IMGID"; exit 1; }
  ( cd $B && md5sum -c --quiet MD5SUMS ) || { log "RESULT fn4 172: VIRHE bin md5s ($B/MD5SUMS)"; exit 1; }
  rm -rf $O; mkdir -p $O; chmod 777 $O
fi
for K in 2 2.5; do
  for LIM in 10 0; do
    for M in 3 4 6 8 12 16; do
      nm=k$K-l$LIM-m$M
      pm $nm-a dsgd $GEO $TIM --bits $K --limit $LIM --M $M --gd
      pm $nm-b dsgd $GEO $TIM --bits $K --limit $LIM --M $M --gd
      pm $nm-dbg dsgddbg $GEO --iters 2 --reps 1 --bits $K --limit $LIM --M $M --gd
      [ "$DRY" = 1 ] || log "RESULT fn4 172 K $K limit $LIM M $M: split chain $(usl $nm-a 'exl3r4d chain') / $(usl $nm-b 'exl3r4d chain') us, fused gd $(usl $nm-a 'exl3r4d gd') / $(usl $nm-b 'exl3r4d gd') us | $(grep -hE 'GD-CHECK|WARM-CHECK|differ' $O/$nm-a.txt $O/$nm-b.txt | sort -u | head -3 | tr '\n' ' ' | cut -c1-200) | fused fallbacks (GD_DEBUG build): $(grep -c 'exl3r4d gd: fused rc' $O/$nm-dbg.txt) $(grep -m1 'exl3r4d gd: fused rc' $O/$nm-dbg.txt | cut -c1-120)$(grep -m1 -iE 'error|die|refus' $O/$nm-a.txt | cut -c1-160)"
    done
  done
done
log "RESULT fn4 172-fn-dsgd: done"
