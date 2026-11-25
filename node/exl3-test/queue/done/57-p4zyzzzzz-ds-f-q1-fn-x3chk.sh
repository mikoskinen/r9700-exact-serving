#!/bin/bash
# 153-fn-x3chk.sh (queue item, 10.11.2026, queue-ar; gpu02, ONE card (HIP device 0), every radiance container DOWN; ~12 min)
# -- the EXL3 prefill-band chain on one card, perfmoe (tools/perfmoe.cpp, exl3-main37 e345bb0) at Flash-Next's EP4 rank
# geometry (128 experts of 512, top-10, hidden 2560, width 640, --ep 4 --ep-rank -1, h_rot 1, the 4-bit group mix):
#  A. 149's fault reproduced: the c16-x3b exl3r4d.so (7a84e33) with exl3_pfprep 2 (the mapprep kernel) -- expected to
#     die in the gate_up GEMM (the LPT order array never written); its prep-0 y kept as the reference.
#  B. the fix (e345bb0: the mapprep kernel builds the order array): prep 0 vs 2 at M 32 / 64 (c8 / c16 decode with MTP 3):
#     y byte for byte, chain us ABBA.
#  C. the shared routing map (interface v10, --x3map: gate_up writes the map, down skips its map kernel): off vs on, y byte
#     for byte at M 19 (gate_up's run form: its own map kernel for down), 20, 32, 64, 256, 512, 1024 (chunked gate_up),
#     chain us ABBA at 20 / 32 / 64 / 256; and map on + prep 2 together at 64.
#  D. the decode band (M 4 / 8 / 12 / 16 = c1-c4 with MTP 3): the split chain gate_up -> down (what the main-line EXL3
#     port issues) vs interface v5's fused moe_exl3_gate_up_down (exl3r4d_moe_gate_up_down_m16: one launch, down units
#     ordered behind their expert's gate_up units; the lab notes put the split down's c4 loss in its INTERCEPT -- a
#     dispatch's fixed cost and the drain, "we stream as fast or faster" -- and refuted deeper prefetch for down: L1 PF 4/6
#     +8..+13 %, S1a streamed +9.2 %, FU_PF_DN). perfmoe --gd: y byte for byte (GD-CHECK), then both timed ABAB in one
#     process. Exact by construction (the separate ops' bits).
# perfmoe times dependent launches on one stream (no host gaps between a layer's ops): the map kernel's own time shows,
# the launch gap only partly. A server A/B follows only if C wins here (the coordinator's rule).
# Bins (queue-ar/bin-x3c): perfmoe, exl3r4d-x3b.so (= c16-x3b's), exl3r4d-x3c.so (e345bb0); md5s checked.
# DRYRUN=1 bash 153-fn-x3chk.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
R=$HOME/gpu/ds41/exl3-test/runs; L=$R/ds41-exl3.log; O=$R/fn4/x3chk
B=; for c in "$Q/bin-x3c" "$Q/../queue-ar/bin-x3c" "$HOME/gpu/ds41/exl3-test/queue-ar/bin-x3c"; do [ -z "$B" ] && [ -e "$c/perfmoe" ] && B=$(cd "$c" && pwd); done
B=${B:-$HOME/gpu/ds41/exl3-test/queue-ar/bin-x3c}
IMAGE=${X153_IMAGE:-radiance-local:c16-x3b}; IMGID=5afffb316b5e
MD5_PM=2c9de966; MD5_OLD=a8cf303b; MD5_NEW=45a82e63
CARD=${CARD:-0}
GEO="--tp 1 --ep 4 --ep-rank -1 --h-rot 1 --chain --chain-band _pf --bits 4 --group 4 --group-mix 0.53 --experts 128 --topk 10 --hidden 2560 --width 640 --pooled 0 --ceiling 640 --rand-data --warm-ms 500"
TIM="--iters ${X153_ITERS:-10} --reps ${X153_REPS:-5}"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD --user $(id -u):$(id -g) --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) -e HOME=/tmp -w /tmp"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
pm() {   # pm NAME SO perfmoe-args... -> $O/NAME.txt, y in $O/NAME.y; returns perfmoe's rc
  local nm=$1 so=$2; shift 2
  local cn=fn-x3chk-$$-$nm
  if [ "$DRY" = 1 ]; then echo "+ docker run ... $IMAGE perfmoe --exl3 /b/$so $* --dump-y /o/$nm.y > $O/$nm.txt"; return 0; fi
  timeout -k 15 300 docker run --rm --init --stop-timeout 10 --name $cn $DEV -v $B:/b:ro -v $O:/o --entrypoint /b/perfmoe $IMAGE \
    --exl3 /b/$so "$@" --dump-y /o/$nm.y > $O/$nm.txt 2>&1
  local rc=$?; docker rm -f $cn > /dev/null 2>&1; return $rc
}
us() { grep -m1 '^RESULT' $O/$1.txt 2>/dev/null | sed -E 's/.*M=[0-9]+ +([0-9.]+) us.*/\1/' || echo NA; }
usl() { grep -m1 "^RESULT $2" $O/$1.txt 2>/dev/null | sed -E 's/.*M=[0-9]+ +([0-9.]+) us.*/\1/' || echo NA; }
same() { if [ -s $O/$1.y ] && [ -s $O/$2.y ] && cmp -s $O/$1.y $O/$2.y; then echo IDENTICAL; else echo "DIFFER($(stat -c %s $O/$1.y 2>/dev/null)/$(stat -c %s $O/$2.y 2>/dev/null))"; fi; }
med() { python3 -c 'import sys,statistics as s; v=[float(x) for x in sys.argv[1:] if x not in ("NA","")]; print("%.2f" % s.median(v) if v else "NA")' "$@"; }
log "RESULT fn4 153-fn-x3chk: start IMAGE=$IMAGE card HIP $CARD bins $B GEO '$GEO' $TIM"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 153: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ "$(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)" = $IMGID ] || { log "RESULT fn4 153: VIRHE image $IMAGE is not $IMGID"; exit 1; }
  [ "$(md5sum < $B/perfmoe | cut -c1-8)" = $MD5_PM ] && [ "$(md5sum < $B/exl3r4d-x3b.so | cut -c1-8)" = $MD5_OLD ] && [ "$(md5sum < $B/exl3r4d-x3c.so | cut -c1-8)" = $MD5_NEW ] \
    || { log "RESULT fn4 153: VIRHE bin md5s (perfmoe $MD5_PM, x3b $MD5_OLD, x3c $MD5_NEW) in $B"; exit 1; }
  rm -rf $O; mkdir -p $O
fi
# ---- A. reproduce (old .so)
pm a-old-p0 exl3r4d-x3b.so $GEO $TIM --M 64; ra0=$?
pm a-old-p2 exl3r4d-x3b.so $GEO $TIM --M 64 --fn4 exl3_pfprep=2; ra2=$?
[ "$DRY" = 1 ] || log "RESULT fn4 153 A (c16-x3b .so, M 64): prep 0 rc $ra0 $(us a-old-p0) us | prep 2 rc $ra2 ($( [ $ra2 != 0 ] && echo "REPRODUCED: $(grep -vE '^W[0-9]' $O/a-old-p2.txt | grep -iE 'fault|error|abort|fail' | head -2 | tr '\n' ' ' | cut -c1-300)" || echo "did NOT fail: $(us a-old-p2) us, y vs prep 0 $(same a-old-p0 a-old-p2)"))"
# ---- B. the fix: prep 0 vs 2, ABBA
for M in 32 64; do
  pm b$M-p0a exl3r4d-x3c.so $GEO $TIM --M $M; pm b$M-p2a exl3r4d-x3c.so $GEO $TIM --M $M --fn4 exl3_pfprep=2
  pm b$M-p2b exl3r4d-x3c.so $GEO $TIM --M $M --fn4 exl3_pfprep=2; pm b$M-p0b exl3r4d-x3c.so $GEO $TIM --M $M
  [ "$DRY" = 1 ] || log "RESULT fn4 153 B M $M (x3c .so): chain us prep 0 $(us b$M-p0a) / $(us b$M-p0b) median $(med $(us b$M-p0a) $(us b$M-p0b)), prep 2 $(us b$M-p2a) / $(us b$M-p2b) median $(med $(us b$M-p2a) $(us b$M-p2b)) | y prep 2 vs 0: $(same b$M-p0a b$M-p2a), $(same b$M-p0a b$M-p2b); prep 0 runs: $(same b$M-p0a b$M-p0b)$( [ $M = 64 ] && echo "; x3c vs x3b prep 0: $(same a-old-p0 b64-p0a)")"
done
# ---- C. the shared map: off vs on
for M in 20 32 64 256; do
  pm c$M-offa exl3r4d-x3c.so $GEO $TIM --M $M; pm c$M-ona exl3r4d-x3c.so $GEO $TIM --M $M --x3map
  pm c$M-onb exl3r4d-x3c.so $GEO $TIM --M $M --x3map; pm c$M-offb exl3r4d-x3c.so $GEO $TIM --M $M
  [ "$DRY" = 1 ] || log "RESULT fn4 153 C M $M: chain us map off $(us c$M-offa) / $(us c$M-offb) median $(med $(us c$M-offa) $(us c$M-offb)), on $(us c$M-ona) / $(us c$M-onb) median $(med $(us c$M-ona) $(us c$M-onb)) | y on vs off: $(same c$M-offa c$M-ona), $(same c$M-offa c$M-onb)$(grep -m1 -iE 'error|refused|fail' $O/c$M-ona.txt | cut -c1-160 | sed 's/^/ | /')"
done
for M in 19 512 1024; do
  pm c$M-off exl3r4d-x3c.so $GEO --iters 2 --reps 2 --M $M; pm c$M-on exl3r4d-x3c.so $GEO --iters 2 --reps 2 --M $M --x3map
  [ "$DRY" = 1 ] || log "RESULT fn4 153 C identity M $M: y on vs off $(same c$M-off c$M-on) (us off $(us c$M-off), on $(us c$M-on))$(grep -m1 -iE 'error|refused|fail' $O/c$M-on.txt | cut -c1-160 | sed 's/^/ | /')"
done
pm c64-bothb exl3r4d-x3c.so $GEO $TIM --M 64 --x3map --fn4 exl3_pfprep=2; pm c64-botha exl3r4d-x3c.so $GEO $TIM --M 64 --x3map --fn4 exl3_pfprep=2
[ "$DRY" = 1 ] || log "RESULT fn4 153 C M 64 map on + prep 2: chain us $(us c64-bothb) / $(us c64-botha) median $(med $(us c64-bothb) $(us c64-botha)) | y vs off: $(same c64-offa c64-bothb), $(same c64-offa c64-botha)"
# ---- D. the decode band: split chain vs the fused gate_up_down (v5)
GEOD=${GEO/--chain-band _pf/--chain-band _m16}
for M in 4 8 12 16; do
  pm d$M-a exl3r4d-x3c.so $GEOD $TIM --M $M --gd; pm d$M-b exl3r4d-x3c.so $GEOD $TIM --M $M --gd
  [ "$DRY" = 1 ] || log "RESULT fn4 153 D M $M: split chain $(usl d$M-a 'exl3r4d chain') / $(usl d$M-b 'exl3r4d chain') us, fused gd $(usl d$M-a 'exl3r4d gd') / $(usl d$M-b 'exl3r4d gd') us | $(grep -hE 'GD-CHECK|WARM-CHECK|differ' $O/d$M-a.txt $O/d$M-b.txt | sort -u | head -3 | tr '\n' ' ' | cut -c1-200)"
done
log "RESULT fn4 153-fn-x3chk: done"
