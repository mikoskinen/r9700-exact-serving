#!/bin/bash
# 162-fn-p2xcmp.sh (queue item, 10.11.2026, queue-ar; gpu02, ONE card (HIP device 0), every radiance container DOWN; ~8 min)
# -- P2x's exactness question (coordinator 10.11.): does each row's MoE output depend on the pass it is computed in? P2x
# would run the MoE over two pipelined pieces at once (4096 rows, stack 3's pieces are 2048) with the pieces' hc ops and
# all-reduces unchanged; that is byte-exact only if every slot's expert output is the same bytes in a 2048-row and in a
# 4096-row pass. perfmoe (exl3-main37 6f02e2d: --split-seed reseeds x and the weights after the routing draw, so the first
# 2048 tokens, their routing, x rows and weights are identical at M 4096 and M 2048; --dump-sorted / --dump-y / --dump-y-r4d)
# runs each library's routed-expert chain at both pass sizes; a script compares every slot of the first 2048 tokens.
#  iq4r: --tp 4 --mod4 --rank 0 (the served moe_gemm_w4a8_mod4 -> gated_had_quant_fp8 -> moe_gemm_w4a8_mod4 chain, libr4d
#        from the image) + exl3r4d's four-class _pf chain beside it;
#  EXL3: --tp 1 --ep 4 --ep-rank 0 (the EP4 rank's whole experts, exl3r4d's two-class _pf chain).
# RESULT a library: slots compared, slots differing, max |diff|. Identical everywhere -> P2x's engine arm is exact by
# construction for the MoE; any difference -> P2x is a non-exact change (user's go) or closed.
# DRYRUN=1 bash 162-fn-p2xcmp.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
R=$HOME/gpu/ds41/exl3-test/runs; L=$R/ds41-exl3.log; O=$R/fn4/p2xcmp
B=; for c in "$Q/bin-p2x" "$Q/../queue-ar/bin-p2x" "$HOME/gpu/ds41/exl3-test/queue-ar/bin-p2x"; do [ -z "$B" ] && [ -e "$c/perfmoe" ] && B=$(cd "$c" && pwd); done
B=${B:-$HOME/gpu/ds41/exl3-test/queue-ar/bin-p2x}
IMAGE=${X162_IMAGE:-radiance-local:c16-x3c}; IMGID=29e7b84dcba5
R4D=/opt/radiance/share/radiance/kernels/libr4d.so
CARD=${CARD:-0}
COMMON="--chain --chain-band _pf --bits 4 --topk 10 --hidden 2560 --width 640 --pooled 0 --rand-data --warm-ms 100 --iters 1 --reps 1 --layers 1 --pools 1 --split-seed --h-rot 1"
GEO_I="--tp 4 --mod4 --rank 0 --experts 512 --r4d $R4D"
GEO_X="--tp 1 --ep 4 --ep-rank 0 --experts 128 --group 4 --group-mix 0.53"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD --user $(id -u):$(id -g) --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) -e HOME=/tmp -w /tmp"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
pm() {   # pm NAME perfmoe-args...
  local nm=$1; shift
  if [ "$DRY" = 1 ]; then echo "+ docker run ... $IMAGE perfmoe --exl3 /b/exl3r4d.so $* > $O/$nm.txt"; return 0; fi
  timeout -k 15 600 docker run --rm --init --stop-timeout 10 --name fn-p2x-$$-$nm $DEV -v $B:/b:ro -v $O:/o --entrypoint /b/perfmoe $IMAGE \
    --exl3 /b/exl3r4d.so "$@" > $O/$nm.txt 2>&1
  local rc=$?; docker rm -f fn-p2x-$$-$nm > /dev/null 2>&1; return $rc
}
cmpy() {   # cmpy SORTED_A Y_A SORTED_B Y_B HALF_TOKENS -> "n slots, k differ, max |d|"
  python3 - "$@" <<'PY' 2>&1 | cut -c1-300
import sys, struct, array
sa, ya, sb, yb, half = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
topk, hid = 10, 2560
A = array.array("i"); A.frombytes(open(sa, "rb").read())
Bs = array.array("i"); Bs.frombytes(open(sb, "rb").read())
YA = open(ya, "rb").read(); YB = open(yb, "rb").read()
rb = hid * 2
pos = {s: j for j, s in enumerate(Bs) if s >= 0}
n = k = 0; md = 0.0
def bf(b, i): return struct.unpack("<f", b"\x00\x00" + b[2 * i:2 * i + 2])[0]
for i, s in enumerate(A):
    if s < 0 or s // topk >= half: continue
    j = pos.get(s)
    if j is None: continue
    ra, rbb = YA[i * rb:(i + 1) * rb], YB[j * rb:(j + 1) * rb]
    n += 1
    if ra != rbb:
        k += 1
        for e in range(hid):
            d = abs(bf(ra, e) - bf(rbb, e))
            if d > md: md = d
print("%d slots compared, %d differ%s" % (n, k, (", max |diff| %.4g" % md) if k else " (byte-identical)"))
PY
}
log "RESULT fn4 162-fn-p2xcmp: start IMAGE=$IMAGE card HIP $CARD bins $B"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 162: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ "$(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)" = $IMGID ] || { log "RESULT fn4 162: VIRHE image $IMAGE is not $IMGID"; exit 1; }
  ( cd $B && md5sum -c --quiet MD5SUMS ) || { log "RESULT fn4 162: VIRHE bin md5s ($B/MD5SUMS)"; exit 1; }
  rm -rf $O; mkdir -p $O; chmod 777 $O
fi
for M in 4096 2048; do
  pm i$M $COMMON $GEO_I --M $M --dump-sorted /o/i$M.sorted --dump-y /o/i$M.yx --dump-y-r4d /o/i$M.yr; ri=$?
  pm x$M $COMMON $GEO_X --M $M --dump-sorted /o/x$M.sorted --dump-y /o/x$M.yx; rx=$?
  [ "$DRY" = 1 ] || log "RESULT fn4 162 runs M $M: iq4r-geometry rc $ri ($(grep -m1 -E '^RESULT libr4d' $O/i$M.txt | cut -c1-90)) | EP4 rc $rx ($(grep -m1 -E '^RESULT exl3r4d' $O/x$M.txt | cut -c1-90))"
done
if [ "$DRY" != 1 ]; then
  log "RESULT fn4 162 P2x byte compare, first 2048 tokens, pass 4096 vs 2048: iq4r libr4d chain (moe_gemm_w4a8_mod4) $(cmpy $O/i4096.sorted $O/i4096.yr $O/i2048.sorted $O/i2048.yr 2048) | exl3r4d four-class chain $(cmpy $O/i4096.sorted $O/i4096.yx $O/i2048.sorted $O/i2048.yx 2048) | EXL3 EP4 exl3r4d chain $(cmpy $O/x4096.sorted $O/x4096.yx $O/x2048.sorted $O/x2048.yx 2048)"
fi
log "RESULT fn4 162-fn-p2xcmp: done"
