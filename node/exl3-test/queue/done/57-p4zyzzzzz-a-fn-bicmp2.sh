#!/bin/bash
# 177b-fn-bicmp.sh (queue item, 11.11.2026, queue-ar; gpu02, ONE card (HIP device 0), every radiance container DOWN; ~20 min)
# -- BATCH INVARIANCE OF THE EXL3 EXPERT KERNELS (the user: answers changing under load are a bug). Above 16 tokens a step the
# routed-expert GEMMs switch from the decode band (exl3r4d.hip) to the prefill band (prefill.c); DeepSeek NST 0 under load
# gave 6 / 20 greedy answers off the alone run. Per row, byte for byte: perfmoe 1206b8b --chain --layers 1 --pools 1
# --split-seed (the first tokens' routing, x and weights identical at every M) --x-bf16 (the engine's input dtype)
# --dump-sorted / --dump-h (gate_up's output) / --dump-y (down's), every slot of the first n tokens matched by its slot id.
# Shapes: Flash-Next EP4 rank K4 (128 experts, top 10, 2560 / 640) and DeepSeek V4.1 EP4 rank K 2 / 2.5 (96, top 6,
# 5120 / 2304, limit 10). Runs: decode band M 1 / 4 / 8 / 16, prefill band M 24 / 48 / 2048 and (if its rows take them) M 8 /
# 16. Comparisons: decode-band M-invariance (M 1 / 8 inside M 16), prefill-band (24 / 48 inside 2048), the band switch at
# one M (decode 8 vs prefill 8, decode 16 vs prefill 16) and across it (decode 16 inside prefill 24 / 48 / 2048) -- h
# and y separately, so the gate_up and the down are told apart. Measurement only; the why from the kernels' code.
# RE-RUN (11.11.): 177 / 178 took a perfmoe build without --limit / --bits 2.5 (every DS run refused "unknown argument --limit";
# Flash-Next's K4 part measured). bin-bi2: perfmoe 848ccdc (exl3-main37: --dump-h / --x-bf16 + --limit / --bits 2.5 +
# --check-args); every pm call runs perfmoe --check-args on its exact line first (in DRYRUN too: an unknown flag fails there).
# DRYRUN=1 bash 177b-fn-bicmp2.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
R=$HOME/gpu/ds41/exl3-test/runs; L=$R/ds41-exl3.log; O=$R/fn4/bicmp2
B=; for c in "$Q/bin-bi2" "$Q/../queue-ar/bin-bi2" "$HOME/gpu/ds41/exl3-test/queue-ar/bin-bi2"; do [ -z "$B" ] && [ -e "$c/perfmoe" ] && B=$(cd "$c" && pwd); done
B=${B:-$HOME/gpu/ds41/exl3-test/queue-ar/bin-bi2}
IMAGE=${X177b_IMAGE:-radiance-local:c16-x3c}; IMGID=29e7b84dcba5
CARD=${CARD:-0}
COM="--chain --layers 1 --pools 1 --split-seed --rand-data --x-bf16 --h-rot 1 --pooled 0 --ceiling 640 --warm-ms 100 --iters 2 --reps 1"
GFN="--tp 1 --ep 4 --ep-rank 0 --experts 128 --topk 10 --hidden 2560 --width 640 --group 4 --group-mix 0.53 --bits 4"
GDS="--tp 1 --ep 4 --ep-rank 0 --experts 96 --topk 6 --hidden 5120 --width 2304 --limit 10"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD --user $(id -u):$(id -g) --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) -e HOME=/tmp -w /tmp"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
pm() {   # pm NAME perfmoe-args...
  local nm=$1; shift
  local cn=fn-bi-$$-$nm
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
log "RESULT fn4 177b-fn-bicmp: start IMAGE=$IMAGE card HIP $CARD bins $B (decode band vs prefill band per row, h and y)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 177b: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ "$(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)" = $IMGID ] || { log "RESULT fn4 177b: VIRHE image $IMAGE is not $IMGID"; exit 1; }
  ( cd $B && md5sum -c --quiet MD5SUMS ) || { log "RESULT fn4 177b: VIRHE bin md5s ($B/MD5SUMS)"; exit 1; }
  rm -rf $O; mkdir -p $O; chmod 777 $O
fi
for sh in "fn4:$GFN:10" "ds2:$GDS --bits 2:6" "ds25:$GDS --bits 2.5:6"; do   # Flash-Next again: M 2 / 3 are new
  nm=${sh%%:*}; rest=${sh#*:}; geo=${rest%:*}; tk=${rest##*:}
  for M in 1 2 3 4 8 16; do pm $nm-d$M $COM $geo --chain-band _m16 --M $M; done
  for M in 8 16 24 48 2048; do pm $nm-p$M $COM $geo --chain-band _pf --M $M; done
  if [ "$DRY" != 1 ]; then
    ok() { [ -s $O/$1.y ] && echo ok || echo "no dump ($(grep -m1 -iE 'error|die|refus|unsupported|perfmoe:' $O/$1.txt | cut -c1-90))"; }
    log "RESULT fn4 177b $nm runs: $(for r in d1 d2 d3 d4 d8 d16 p8 p16 p24 p48 p2048; do echo -n "$r $(ok $nm-$r); "; done)"
    log "RESULT fn4 177b $nm decode band, split-K 2 (M <= 2) vs 1: M1 in M3 [$(cmpslots $nm-d1 $nm-d3 1 $tk)] ; M1 in M16 [$(cmpslots $nm-d1 $nm-d16 1 $tk)] ; M2 in M3 [$(cmpslots $nm-d2 $nm-d3 2 $tk)] ; M2 in M16 [$(cmpslots $nm-d2 $nm-d16 2 $tk)] ; M1 in M2 [$(cmpslots $nm-d1 $nm-d2 1 $tk)]"
    log "RESULT fn4 177b $nm decode band M-invariance from 3: M3 in M16 [$(cmpslots $nm-d3 $nm-d16 3 $tk)] ; M4 in M8 [$(cmpslots $nm-d4 $nm-d8 4 $tk)] ; M8 in M16 [$(cmpslots $nm-d8 $nm-d16 8 $tk)]"
    log "RESULT fn4 177b $nm prefill band M-invariance: M24 in M2048 [$(cmpslots $nm-p24 $nm-p2048 24 $tk)] ; M48 in M2048 [$(cmpslots $nm-p48 $nm-p2048 48 $tk)] ; M8 in M24 [$(cmpslots $nm-p8 $nm-p24 8 $tk)]"
    log "RESULT fn4 177b $nm band switch at one M: decode 8 vs prefill 8 [$(cmpslots $nm-d8 $nm-p8 8 $tk)] ; decode 16 vs prefill 16 [$(cmpslots $nm-d16 $nm-p16 16 $tk)]"
    log "RESULT fn4 177b $nm across the switch: decode 16 in prefill 24 [$(cmpslots $nm-d16 $nm-p24 16 $tk)] ; in 48 [$(cmpslots $nm-d16 $nm-p48 16 $tk)] ; in 2048 [$(cmpslots $nm-d16 $nm-p2048 16 $tk)]"
  fi
done
log "RESULT fn4 177b-fn-bicmp: done"
