#!/bin/bash
# 171-fn-e8read.sh (queue item, 10.11.2026, queue-ar; gpu02, ONE card (HIP device 0), every radiance container DOWN; ~15 min)
# -- E8' READING: what PREP 2 (the map and the by-row prep in ONE launch, exl3r4d_pf_mapprep_kernel) loses against PREP 0
# (the map kernel, then the prep by entry). 153 B, EXL3 EP4 chain _pf: M32 256.07 vs 223.61 us (+14.5 %), M64 342.78 vs
# 320.38 (+7.0 %), y IDENTICAL. Candidates: workgroup 0 (256 threads: the map, then the LPT order sort) as the launch's
# long pole, or the by-row prep itself (each task rotates ONE row, the expert's suh read a row). Arms (exl3-main37 9ebbd24,
# one build): stock prep 0 / prep 2; abl1 prep 2 (the prep workgroups return at once -- TIMING ONLY, wrong output: the
# by-row prep's cost); abl2 prep 2 (workgroup 0 writes the map's own order, no LPT sort -- exact); t1024 prep 2 (the
# launch at 1024 threads: workgroup 0's map and order with 4x the threads -- exact). M 24 / 32 / 48 / 64 / 128 (decode c8 / c16
# verify at 32 / 64 rows), A B B A per arm against stock prep 0, then y byte for byte (layer 0, --split-seed) of every exact
# arm against stock prep 0. 153 B's geometry (EP4 rank -1, _pf chain).
# 12.11. RE-SCOPE (user: quick screens, ~5 min): M 32 / 64 only (the two heights 153 B measured the loss at; X171_MS), the
# identity at M 32 / 64 for the two exact candidates abl2 / t1024 only (stock prep 2's y is IDENTICAL by 153 B; X171_IDMS).
# DRYRUN=1 bash 171-fn-e8read.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
R=$HOME/gpu/ds41/exl3-test/runs; L=$R/ds41-exl3.log; O=$R/fn4/e8read
B=; for c in "$Q/bin-e8" "$Q/../queue-ar/bin-e8" "$HOME/gpu/ds41/exl3-test/queue-ar/bin-e8"; do [ -z "$B" ] && [ -e "$c/perfmoe" ] && B=$(cd "$c" && pwd); done
B=${B:-$HOME/gpu/ds41/exl3-test/queue-ar/bin-e8}
IMAGE=${X171_IMAGE:-radiance-local:c16-x3c}; IMGID=29e7b84dcba5
CARD=${CARD:-0}; CARDUUID=${CARDUUID:-GPU-dbba5c851cfef7b3}   # 12.11. riser move: pinned by chip (KFD device 0 before the move = c3:00, dbba..), not by enumeration order
GEO="--tp 1 --ep 4 --ep-rank -1 --h-rot 1 --chain --chain-band _pf --bits 4 --group 4 --group-mix 0.53 --experts 128 --topk 10 --hidden 2560 --width 640 --pooled 0 --ceiling 640 --rand-data --warm-ms 500"
TIM="--iters ${X171_ITERS:-10} --reps ${X171_REPS:-5}"
IDC="--layers 1 --pools 1 --split-seed --iters 2 --reps 1"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$CARDUUID -e HIP_VISIBLE_DEVICES=$CARD --user $(id -u):$(id -g) --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) -e HOME=/tmp -w /tmp"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
pm() {   # pm NAME VARIANT perfmoe-args...
  local nm=$1 v=$2; shift 2
  local cn=fn-e8-$$-$nm
  if [ "$DRY" = 1 ]; then echo "+ docker run ... $IMAGE perfmoe --exl3 /b/exl3r4d-$v.so $* > $O/$nm.txt"; return 0; fi
  timeout -k 15 300 docker run --rm --init --stop-timeout 10 --name $cn $DEV -v $B:/b:ro -v $O:/o --entrypoint /b/perfmoe $IMAGE \
    --exl3 /b/exl3r4d-$v.so "$@" > $O/$nm.txt 2>&1
  local rc=$?; docker rm -f $cn > /dev/null 2>&1; return $rc
}
usl() { local v; v=$(grep -m1 "^RESULT $2" $O/$1.txt 2>/dev/null | sed -nE 's/.*M=[0-9]+ +([0-9.]+) us.*/\1/p'); echo ${v:-NA}; }
avg2() { python3 -c "import sys
a=[float(x) for x in sys.argv[1:] if x!='NA']
print('%.2f' % (sum(a)/len(a)) if len(a)==len(sys.argv[1:]) else 'NA')" "$@"; }
pct() { python3 -c "import sys
a,b=sys.argv[1],sys.argv[2]
print('NA' if 'NA' in (a,b) else '%+.1f %%' % (100*(float(b)/float(a)-1)))" "$1" "$2"; }
log "RESULT fn4 171-fn-e8read: start IMAGE=$IMAGE card HIP $CARD bins $B"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 171: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ "$(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)" = $IMGID ] || { log "RESULT fn4 171: VIRHE image $IMAGE is not $IMGID"; exit 1; }
  ( cd $B && md5sum -c --quiet MD5SUMS ) || { log "RESULT fn4 171: VIRHE bin md5s ($B/MD5SUMS)"; exit 1; }
  rm -rf $O; mkdir -p $O; chmod 777 $O
fi
for M in ${X171_MS:-32 64}; do
  line="M $M:"
  pm s0a$M stock $GEO $TIM --M $M
  for arm in "s2:stock:2" "a1:abl1:2" "a2:abl2:2" "t2:t1024:2"; do
    nm=${arm%%:*}; rest=${arm#*:}; v=${rest%%:*}; pp=${rest#*:}
    pm $nm-b1-$M $v $GEO $TIM --M $M --fn4 exl3_pfprep=$pp
    pm $nm-b2-$M $v $GEO $TIM --M $M --fn4 exl3_pfprep=$pp
  done
  pm s0b$M stock $GEO $TIM --M $M
  if [ "$DRY" != 1 ]; then
    a=$(avg2 $(usl s0a$M exl3r4d) $(usl s0b$M exl3r4d))
    line="$line stock prep 0 $a us |"
    for arm in "s2:stock prep 2" "a1:abl1 prep 2 (no prep work, timing)" "a2:abl2 prep 2 (no LPT sort)" "t2:t1024 prep 2"; do
      nm=${arm%%:*}; lab=${arm#*:}
      b=$(avg2 $(usl $nm-b1-$M exl3r4d) $(usl $nm-b2-$M exl3r4d))
      line="$line $lab $b ($(pct $a $b)) |"
    done
    log "RESULT fn4 171 E8' M $M chain _pf us: $line"
  fi
done
# identity: layer 0's y of every exact arm against stock prep 0
for M in ${X171_IDMS:-32 64}; do
  pm id-s0-$M stock $GEO $IDC --M $M --dump-y /o/id-s0-$M.y
  pm id-a2-$M abl2 $GEO $IDC --M $M --fn4 exl3_pfprep=2 --dump-y /o/id-a2-$M.y
  pm id-t2-$M t1024 $GEO $IDC --M $M --fn4 exl3_pfprep=2 --dump-y /o/id-t2-$M.y
  if [ "$DRY" != 1 ]; then
    r=""
    for x in a2 t2; do
      if [ -s $O/id-s0-$M.y ] && cmp -s $O/id-s0-$M.y $O/id-$x-$M.y; then r="$r $x IDENTICAL;"; else r="$r $x DIFFERENT/missing ($(cmp $O/id-s0-$M.y $O/id-$x-$M.y 2>&1 | head -1 | cut -c1-80));"; fi
    done
    log "RESULT fn4 171 E8' identity M $M vs stock prep 0 ($(stat -c %s $O/id-s0-$M.y 2>/dev/null) bytes):$r"
  fi
done
log "RESULT fn4 171-fn-e8read: done"
