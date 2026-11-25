#!/bin/bash
# 164-fn-dzcmp.sh (queue item, 10.11.2026, queue-ar; gpu02, ONE card (HIP device 0), every radiance container DOWN; ~10 min)
# Two one-card readings on c16-x3c's image (perfmoe c77a7c42 = exl3-main37 6f02e2d), no server:
# A) DZ -- the EXL3 decode-band dead-row zeroing (from 161: down at the EP4 rank geometry, stage 2 - stage 1 = +9.84 us a
#    dispatch at M 16, +0.75 at M 4, +0.47 at M 1; the grid holds min(T, n_expert) + 1 runs but a rank's live runs are
#    ~1/4 of that, so at M 16 ~104 spare runs a column block each scanned all ~120 dead rows x 16 units with an integer
#    modulo). exl3r4d-dz.so = the same source with EXL3R4D_DZ 1 (each spare visits only its own rows d = spare + j nspare;
#    the same stores). Stock vs DZ, A B B A, op down and op gate_up (decode-band rows), M 1 2 4 8 12 16, 161's geometry;
#    the stage-2 exit with DZ at M 4 / 16 (161's stage 1 / 2 beside it); identity: the decode-band chain's y (layer 0,
#    --split-seed, every slot incl. the zeroed dead rows) stock vs DZ, byte for byte, at every M.
# B) P2x's split -- where the 4096-row MoE pass saves (162: one rep each, iq4r chain 2068.46 vs 2 x 1228.45 us, EXL3 EP4
#    2572.03 vs 2 x 1440.50): chain / op gate_up / op down at M 2048 and 4096, iters 5 reps 3, iq4r (tp4 mod4 rank 0, libr4d
#    from the image) and EXL3 EP4 rank 0 (_pf rows); sizes the engine arm (the joint pass vs the exposed MoE all-reduce).
# Bins (queue-ar/bin-dz, MD5SUMS checked): perfmoe, exl3r4d-{stock,dz,stage2dz}.so (out33, one build).
# DRYRUN=1 bash 164-fn-dzcmp.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
R=$HOME/gpu/ds41/exl3-test/runs; L=$R/ds41-exl3.log; O=$R/fn4/dzcmp
B=; for c in "$Q/bin-dz" "$Q/../queue-ar/bin-dz" "$HOME/gpu/ds41/exl3-test/queue-ar/bin-dz"; do [ -z "$B" ] && [ -e "$c/perfmoe" ] && B=$(cd "$c" && pwd); done
B=${B:-$HOME/gpu/ds41/exl3-test/queue-ar/bin-dz}
IMAGE=${X164_IMAGE:-radiance-local:c16-x3c}; IMGID=29e7b84dcba5
R4D=/opt/radiance/share/radiance/kernels/libr4d.so
CARD=${CARD:-0}
GEO="--tp 1 --ep 4 --ep-rank -1 --h-rot 1 --bits 4 --group 4 --group-mix 0.53 --experts 128 --topk 10 --hidden 2560 --width 640 --pooled 0 --ceiling 640 --warm-ms 500"
TIM="--iters 20 --reps 5"
IDC="--chain --chain-band _m16 --layers 1 --pools 1 --split-seed --rand-data --iters 2 --reps 1"
PB="--bits 4 --topk 10 --hidden 2560 --width 640 --pooled 0 --h-rot 1 --rand-data --warm-ms 100 --iters 5 --reps 3"
GEO_I="--tp 4 --mod4 --rank 0 --experts 512 --r4d $R4D"
GEO_X="--tp 1 --ep 4 --ep-rank 0 --experts 128 --group 4 --group-mix 0.53"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD --user $(id -u):$(id -g) --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) -e HOME=/tmp -w /tmp"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
pm() {   # pm NAME VARIANT perfmoe-args...
  local nm=$1 v=$2; shift 2
  local cn=fn-dz-$$-$nm
  if [ "$DRY" = 1 ]; then echo "+ docker run ... $IMAGE perfmoe --exl3 /b/exl3r4d-$v.so $* > $O/$nm.txt"; return 0; fi
  timeout -k 15 300 docker run --rm --init --stop-timeout 10 --name $cn $DEV -v $B:/b:ro -v $O:/o --entrypoint /b/perfmoe $IMAGE \
    --exl3 /b/exl3r4d-$v.so "$@" > $O/$nm.txt 2>&1
  local rc=$?; docker rm -f $cn > /dev/null 2>&1; return $rc
}
# usl NAME LIB -> the first "M=<m>  <us> us" number of that library's RESULT line
usl() { local v; v=$(grep -m1 "^RESULT $2" $O/$1.txt 2>/dev/null | sed -nE 's/.*M=[0-9]+ +([0-9.]+) us.*/\1/p'); echo ${v:-NA}; }
avg2() { python3 -c "import sys
a=[float(x) for x in sys.argv[1:] if x!='NA']
print('%.2f' % (sum(a)/len(a)) if len(a)==len(sys.argv[1:]) else 'NA')" "$@"; }
pct() { python3 -c "import sys
a,b=sys.argv[1],sys.argv[2]
print('NA' if 'NA' in (a,b) else '%+.1f %%' % (100*(float(b)/float(a)-1)))" "$1" "$2"; }
log "RESULT fn4 164-fn-dzcmp: start IMAGE=$IMAGE card HIP $CARD bins $B"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 164: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ "$(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)" = $IMGID ] || { log "RESULT fn4 164: VIRHE image $IMAGE is not $IMGID"; exit 1; }
  ( cd $B && md5sum -c --quiet MD5SUMS ) || { log "RESULT fn4 164: VIRHE bin md5s ($B/MD5SUMS)"; exit 1; }
  rm -rf $O; mkdir -p $O; chmod 777 $O
fi
# ---- A) DZ
for M in 1 2 4 8 12 16; do
  for op in down gate_up; do
    row=exl3r4d_moe_${op}_m16
    pm $op$M-a1 stock $GEO $TIM --op $op --exl3-row $row --M $M
    pm $op$M-b1 dz    $GEO $TIM --op $op --exl3-row $row --M $M
    pm $op$M-b2 dz    $GEO $TIM --op $op --exl3-row $row --M $M
    pm $op$M-a2 stock $GEO $TIM --op $op --exl3-row $row --M $M
  done
  pm id$M-a stock $GEO $IDC --M $M --dump-y /o/id$M-a.y
  pm id$M-b dz    $GEO $IDC --M $M --dump-y /o/id$M-b.y
  if [ "$DRY" != 1 ]; then
    line="M $M:"
    for op in down gate_up; do
      a=$(avg2 $(usl $op$M-a1 exl3r4d) $(usl $op$M-a2 exl3r4d)); b=$(avg2 $(usl $op$M-b1 exl3r4d) $(usl $op$M-b2 exl3r4d))
      line="$line $op stock $a us ($(usl $op$M-a1 exl3r4d) / $(usl $op$M-a2 exl3r4d)) -> DZ $b ($(usl $op$M-b1 exl3r4d) / $(usl $op$M-b2 exl3r4d)) $(pct $a $b) |"
    done
    if [ -s $O/id$M-a.y ] && [ -s $O/id$M-b.y ]; then
      if cmp -s $O/id$M-a.y $O/id$M-b.y; then idr="IDENTICAL ($(stat -c %s $O/id$M-a.y) bytes)"; else idr="DIFFERENT ($(cmp -l $O/id$M-a.y $O/id$M-b.y | wc -l) bytes differ)"; fi
    else idr="NA (no dump: $(tail -1 $O/id$M-b.txt | cut -c1-120))"; fi
    log "RESULT fn4 164 DZ $line chain y stock vs DZ: $idr"
  fi
done
for M in 4 16; do pm s2dz$M stage2dz $GEO $TIM --op down --exl3-row exl3r4d_moe_down_m16 --M $M; done
[ "$DRY" = 1 ] || log "RESULT fn4 164 DZ stage-2 exit with DZ (down): M 4 $(usl s2dz4 exl3r4d) us, M 16 $(usl s2dz16 exl3r4d) us (161 stock: stage1 4.62 / 6.60, stage2 5.37 / 16.44)"
# ---- B) P2x split
for M in 2048 4096; do
  pm pi-ch$M stock $PB $GEO_I --chain --chain-band _pf --M $M
  pm pi-gu$M stock $PB $GEO_I --op gate_up --exl3-row exl3r4d_moe_gate_up_mod4_pf --M $M
  pm pi-dn$M stock $PB $GEO_I --op down --exl3-row exl3r4d_moe_down_mod4_pf --M $M
  pm px-ch$M stock $PB $GEO_X --chain --chain-band _pf --M $M
  pm px-gu$M stock $PB $GEO_X --op gate_up --exl3-row exl3r4d_moe_gate_up_pf --M $M
  pm px-dn$M stock $PB $GEO_X --op down --exl3-row exl3r4d_moe_down_pf --M $M
done
if [ "$DRY" != 1 ]; then
  sv() { python3 -c "import sys
a,b=sys.argv[1],sys.argv[2]
print('NA' if 'NA' in (a,b) else '%.1f us (%+.1f %%)' % (float(b)-2*float(a), 100*(float(b)/(2*float(a))-1)))" "$1" "$2"; }
  line=""
  for k in ch gu dn; do
    i2=$(usl pi-${k}2048 libr4d); i4=$(usl pi-${k}4096 libr4d); x2=$(usl px-${k}2048 exl3r4d); x4=$(usl px-${k}4096 exl3r4d)
    line="$line $k: iq4r libr4d 2048 $i2 / 4096 $i4 us, 4096 - 2 x 2048 = $(sv $i2 $i4); EXL3 EP4 2048 $x2 / 4096 $x4, $(sv $x2 $x4) |"
  done
  log "RESULT fn4 164 P2x split (one card, iters 5 reps 3; ch = the chain, gu / dn the ops):$line"
fi
log "RESULT fn4 164-fn-dzcmp: done"
