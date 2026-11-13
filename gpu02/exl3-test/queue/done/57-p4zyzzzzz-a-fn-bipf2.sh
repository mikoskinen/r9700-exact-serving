#!/bin/bash
# 178b-fn-bipf.sh (queue item, 11.11.2026, queue-ar; gpu02, ONE card (HIP device 0), every radiance container DOWN; ~20 min)
# -- BATCH INVARIANCE, THE PREFILL BAND ALONE (coordinator 11.11.: at NST 5 under load 11 / 20 differ, one at token 0 -- a
# prompt's own prefill output moves with its batch-mates). A prefill piece's rows inside passes of other heights: the first
# 2048 tokens at M 2048 vs 2064 (16 decode rows beside them) vs 4096 vs 4160 (DS's MNBT), and the first 512 at M 512 vs 2048
# -- perfmoe 1206b8b --split-seed (the extra tokens are the batch-mates; the first ones' routing, x and weights identical),
# --x-bf16, gate_up h and down y per slot. Flash-Next K4 and DeepSeek K 2 / 2.5. (162 found 2048 in 4096 byte-identical
# for Flash-Next's K4 chain with f16 x; this adds DS's K, bf16 x, the odd heights and h.)
# RE-RUN (11.11.): 177 / 178 took a perfmoe build without --limit / --bits 2.5 (every DS run refused "unknown argument --limit";
# Flash-Next's K4 part measured). bin-bi2: perfmoe 848ccdc (exl3-main37: --dump-h / --x-bf16 + --limit / --bits 2.5 +
# --check-args); every pm call runs perfmoe --check-args on its exact line first (in DRYRUN too: an unknown flag fails there).
# DRYRUN=1 bash 178b-fn-bipf2.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
R=$HOME/gpu/ds41/exl3-test/runs; L=$R/ds41-exl3.log; O=$R/fn4/bipf2
B=; for c in "$Q/bin-bi2" "$Q/../queue-ar/bin-bi2" "$HOME/gpu/ds41/exl3-test/queue-ar/bin-bi2"; do [ -z "$B" ] && [ -e "$c/perfmoe" ] && B=$(cd "$c" && pwd); done
B=${B:-$HOME/gpu/ds41/exl3-test/queue-ar/bin-bi2}
IMAGE=${X178b_IMAGE:-radiance-local:c16-x3c}; IMGID=29e7b84dcba5
CARD=${CARD:-0}
COM="--chain --layers 1 --pools 1 --split-seed --rand-data --x-bf16 --h-rot 1 --pooled 0 --ceiling 640 --warm-ms 100 --iters 2 --reps 1"
GFN="--tp 1 --ep 4 --ep-rank 0 --experts 128 --topk 10 --hidden 2560 --width 640 --group 4 --group-mix 0.53 --bits 4"
GDS="--tp 1 --ep 4 --ep-rank 0 --experts 96 --topk 6 --hidden 5120 --width 2304 --limit 10"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD --user $(id -u):$(id -g) --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) -e HOME=/tmp -w /tmp"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
pm() {   # pm NAME perfmoe-args...
  local nm=$1; shift
  local cn=fn-bipf-$$-$nm
  local ck; ck=$(timeout -k 5 60 docker run --rm --init --network none --name $cn-ck -v $B:/b:ro --entrypoint /b/perfmoe $IMAGE \
      --exl3 /b/exl3r4d-bi.so "$@" --dump-sorted /o/$nm.sorted --dump-h /o/$nm.h --dump-y /o/$nm.y --check-args 2>&1 | tail -1)
  if [ "$DRY" = 1 ]; then echo "+ [$ck] docker run ... $IMAGE perfmoe --exl3 /b/exl3r4d-bi.so $* > $O/$nm.txt"; return 0; fi
  case "$ck" in *"ARGS OK"*) ;; *) echo "perfmoe --check-args: $ck" > $O/$nm.txt; return 1 ;; esac
  timeout -k 15 300 docker run --rm --init --stop-timeout 10 --name $cn $DEV -v $B:/b:ro -v $O:/o --entrypoint /b/perfmoe $IMAGE \
    --exl3 /b/exl3r4d-bi.so "$@" --dump-sorted /o/$nm.sorted --dump-h /o/$nm.h --dump-y /o/$nm.y > $O/$nm.txt 2>&1
  local rc=$?; docker rm -f $cn > /dev/null 2>&1; return $rc
}
cmpslots() {   # cmpslots A B NTOK TOPK -> "h: n slots k differ max|d| | y: ..."
  python3 - "$O/$1" "$O/$2" "$3" "$4" <<'PY' 2>&1 | cut -c1-300
import os, struct, sys, array
a, b, ntok, topk = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
def rd(p, ext):
    try: return open(p + ext, "rb").read()
    except OSError: return None
sa, sb = rd(a, ".sorted"), rd(b, ".sorted")
if not sa or not sb: print("missing dumps (%s / %s)" % (os.path.basename(a), os.path.basename(b))); sys.exit()
A = array.array("i"); A.frombytes(sa); Bs = array.array("i"); Bs.frombytes(sb)
pos = {s: j for j, s in enumerate(Bs) if s >= 0}
out = []
for ext, dt in ((".h", "f16"), (".y", "bf16")):
    xa, xb = rd(a, ext), rd(b, ext)
    if not xa or not xb: out.append("%s missing" % ext[1:]); continue
    rba, rbb = len(xa) // len(A), len(xb) // len(Bs)
    rb = min(rba, rbb)
    n = k = 0; md = 0.0
    def val(buf, i):
        h = buf[2 * i:2 * i + 2]
        return struct.unpack("<e", h)[0] if dt == "f16" else struct.unpack("<f", b"\x00\x00" + h)[0]
    for i, s in enumerate(A):
        if s < 0 or s // topk >= ntok: continue
        j = pos.get(s)
        if j is None: continue
        ra, rbb_ = xa[i * rba:i * rba + rb], xb[j * rbb:j * rbb + rb]
        n += 1
        if ra != rbb_:
            k += 1
            for e in range(rb // 2):
                d = abs(val(ra, e) - val(rbb_, e))
                if d > md: md = d
    out.append("%s %d slots, %d differ%s" % (ext[1:], n, k, (", max|d| %.3g" % md) if k else ""))
print(" | ".join(out))
PY
}
log "RESULT fn4 178b-fn-bipf: start IMAGE=$IMAGE card HIP $CARD bins $B (the prefill band among batch-mates, h and y)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 178b: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ "$(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)" = $IMGID ] || { log "RESULT fn4 178b: VIRHE image $IMAGE is not $IMGID"; exit 1; }
  ( cd $B && md5sum -c --quiet MD5SUMS ) || { log "RESULT fn4 178b: VIRHE bin md5s ($B/MD5SUMS)"; exit 1; }
  rm -rf $O; mkdir -p $O; chmod 777 $O
fi
for sh in "ds2:$GDS --bits 2:6" "ds25:$GDS --bits 2.5:6"; do   # Flash-Next's K4 part measured in 178 (all 0 differ)
  nm=${sh%%:*}; rest=${sh#*:}; geo=${rest%:*}; tk=${rest##*:}
  for M in 512 2048 2064 4096 4160; do pm $nm-p$M $COM $geo --chain-band _pf --M $M; done
  if [ "$DRY" != 1 ]; then
    log "RESULT fn4 178b $nm prefill band, a piece among batch-mates: 2048 in 2064 [$(cmpslots $nm-p2048 $nm-p2064 2048 $tk)] ; in 4096 [$(cmpslots $nm-p2048 $nm-p4096 2048 $tk)] ; in 4160 [$(cmpslots $nm-p2048 $nm-p4160 2048 $tk)] ; 512 in 2048 [$(cmpslots $nm-p512 $nm-p2048 512 $tk)]"
  fi
done
log "RESULT fn4 178b-fn-bipf: done"
