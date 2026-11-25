#!/bin/bash
# 182-fn-dual.sh (queue item, 11.11.2026, queue-ar; gpu02, ONE card (HIP device 0), every radiance container DOWN; ~20 min)
# -- fn4 BI-dual BYTE CHECK + COST (coordinator 11.11.: DS's ds41_bi segments a mixed step's MoE into a decode-band pass and
# a prefill-band pass, +5.6 ms of expert time a mixed step -- each touched expert read twice). exl3-main37 ea077c8 (interface
# v12): moe_exl3_down's optional `band` operand, i32 [M]; in an M > 16 launch a marked token's rows take EPIC's paired tile
# sums, the M <= 16 band's down arithmetic (177b: gate_up's h the same bits in both bands; down's y differs only by EPIC).
# PASS = in ONE prefill-band pass (M 48 / 2048) the marked tokens' h and y = the decode band's (M 16, exl3_bi 0) slot for
# slot, the unmarked ones' = the plain prefill band's, and the marked ones differ from the plain prefill band (the flag
# acts); contiguous (0:16) and split (0:4,8:12) marks. Flash-Next K4 and DS K 2 / 2.5 (limit 10). Then the cost (timed
# chain, A B B A): one dual pass over decode D + prefill P rows vs the segmented passes (decode pieces <= 16 + one prefill
# pass) and vs the plain prefill pass over D + P. perfmoe ea077c8 + exl3r4d dual (DZ 2), every line --check-args'd first.
# --- (the header below is 177c's, kept for its helpers) ---
# 177b-fn-bicmp.sh (queue item, 11.11.2026, queue-ar; gpu02, ONE card (HIP device 0), every radiance container DOWN; ~20 min)
# -- BI FIX CHECK: exl3-main37 43da335's switch exl3_bi (the decode band computes every row as the prefill band: down's
# epilogue without EPIC's paired sum, gate_up without the small-M split-K 2). 177b: gate_up's h the same bits in both bands
# from M 2 / 3, down's y not (EPIC), Flash-Next's M 1 not (split-K 2). Here every decode-band run takes --fn4 exl3_bi=1:
# PASS = every comparison 0 differ in h and y -- decode M 1 / 2 / 3 / 4 / 8 / 16 among themselves, decode vs prefill at M 8 /
# 16 and inside prefill 24 / 48 / 2048 -- and the prefill band with exl3_bi=1 = the prefill band without it (M 2048). Then
# the cost: the decode-band chain with exl3_bi 0 vs 1 at M 1 / 2 / 4 / 8 / 16 (A B B A, timed, the default geometry runs).
# Flash-Next K4 and DS K 2 / 2.5 (limit 10). perfmoe 848ccdc + exl3r4d bi3 (bin-dual), every line --check-args'd first.
# DRYRUN=1 bash 177c-fn-bifix.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
R=$HOME/gpu/ds41/exl3-test/runs; L=$R/ds41-exl3.log; O=$R/fn4/dual
B=; for c in "$Q/bin-dual" "$Q/../queue-ar/bin-dual" "$HOME/gpu/ds41/exl3-test/queue-ar/bin-dual"; do [ -z "$B" ] && [ -e "$c/perfmoe" ] && B=$(cd "$c" && pwd); done
B=${B:-$HOME/gpu/ds41/exl3-test/queue-ar/bin-dual}
IMAGE=${X182_IMAGE:-radiance-local:c16-x3c}; IMGID=29e7b84dcba5
CARD=${CARD:-0}
COM="--chain --layers 1 --pools 1 --split-seed --rand-data --x-bf16 --h-rot 1 --pooled 0 --ceiling 640 --warm-ms 100 --iters 2 --reps 1"
GFN="--tp 1 --ep 4 --ep-rank 0 --experts 128 --topk 10 --hidden 2560 --width 640 --group 4 --group-mix 0.53 --bits 4"
GDS="--tp 1 --ep 4 --ep-rank 0 --experts 96 --topk 6 --hidden 5120 --width 2304 --limit 10"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD --user $(id -u):$(id -g) --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) -e HOME=/tmp -w /tmp"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
pm() {   # pm NAME perfmoe-args...
  local nm=$1; shift
  local cn=fn-dual-$$-$nm
  local ck; ck=$(timeout -k 5 60 docker run --rm --init --network none --name $cn-ck -v $B:/b:ro --entrypoint /b/perfmoe $IMAGE \
      --exl3 /b/exl3r4d-dual.so "$@" --dump-sorted /o/$nm.sorted --dump-h /o/$nm.h --dump-y /o/$nm.y --check-args 2>&1 | tail -1)
  if [ "$DRY" = 1 ]; then echo "+ [$ck] docker run ... $IMAGE perfmoe --exl3 /b/exl3r4d-dual.so $* > $O/$nm.txt"; return 0; fi
  case "$ck" in *"ARGS OK"*) ;; *) echo "perfmoe --check-args: $ck" > $O/$nm.txt; return 1 ;; esac
  timeout -k 15 300 docker run --rm --init --stop-timeout 10 --name $cn $DEV -v $B:/b:ro -v $O:/o --entrypoint /b/perfmoe $IMAGE \
    --exl3 /b/exl3r4d-dual.so "$@" --dump-sorted /o/$nm.sorted --dump-h /o/$nm.h --dump-y /o/$nm.y > $O/$nm.txt 2>&1
  local rc=$?; docker rm -f $cn > /dev/null 2>&1; return $rc
}
cmpslots() {   # cmpslots A B LO HI TOPK -> "h: n slots k differ max|d| | y: ..." over tokens [LO, HI)
  python3 - "$O/$1" "$O/$2" "$3" "$4" "$5" <<'PY' 2>&1 | cut -c1-300
import os, struct, sys, array
a, b, lo, hi, topk = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
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
        if s < 0 or not (lo <= s // topk < hi): continue
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
tm() {   # tm NAME perfmoe-args...: as pm without the dumps
  local nm=$1; shift
  local cn=fn-dual-$$-$nm ck
  ck=$(timeout -k 5 60 docker run --rm --init --network none --name $cn-ck -v $B:/b:ro --entrypoint /b/perfmoe $IMAGE --exl3 /b/exl3r4d-dual.so "$@" --check-args 2>&1 | tail -1)
  if [ "$DRY" = 1 ]; then echo "+ [$ck] docker run ... perfmoe $* > $O/$nm.txt"; return 0; fi
  case "$ck" in *"ARGS OK"*) ;; *) echo "perfmoe --check-args: $ck" > $O/$nm.txt; return 1 ;; esac
  timeout -k 15 300 docker run --rm --init --stop-timeout 10 --name $cn $DEV -v $B:/b:ro --entrypoint /b/perfmoe $IMAGE \
    --exl3 /b/exl3r4d-dual.so "$@" > $O/$nm.txt 2>&1
  local rc=$?; docker rm -f $cn > /dev/null 2>&1; return $rc
}
log "RESULT fn4 182-fn-dual: start IMAGE=$IMAGE card HIP $CARD bins $B (one prefill-band pass with the v12 band vs the decode band and the plain prefill band, per slot)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 182: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ "$(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)" = $IMGID ] || { log "RESULT fn4 182: VIRHE image $IMAGE is not $IMGID"; exit 1; }
  ( cd $B && md5sum -c --quiet MD5SUMS ) || { log "RESULT fn4 182: VIRHE bin md5s ($B/MD5SUMS)"; exit 1; }
  rm -rf $O; mkdir -p $O; chmod 777 $O
fi
for sh in "fn4:$GFN:10" "ds2:$GDS --bits 2:6" "ds25:$GDS --bits 2.5:6"; do
  nm=${sh%%:*}; rest=${sh#*:}; geo=${rest%:*}; tk=${rest##*:}
  pm $nm-d4 $COM $geo --chain-band _m16 --M 4
  pm $nm-d16 $COM $geo --chain-band _m16 --M 16
  for M in 48 2048; do
    pm $nm-p$M $COM $geo --chain-band _pf --M $M
    pm $nm-b$M $COM $geo --chain-band _pf --M $M --band 0:16
  done
  pm $nm-s2048 $COM $geo --chain-band _pf --M 2048 --band 0:4,8:12
  if [ "$DRY" != 1 ]; then
    ok() { [ -s $O/$1.y ] && echo ok || echo "no dump ($(grep -m1 -iE 'error|die|refus|unsupported|perfmoe:' $O/$1.txt | cut -c1-90))"; }
    log "RESULT fn4 182 $nm runs: $(for r in d4 d16 p48 b48 p2048 b2048 s2048; do echo -n "$r $(ok $nm-$r); "; done)"
    log "RESULT fn4 182 $nm MARKED = decode band (want 0 differ): b48 [0,16) vs d16 [$(cmpslots $nm-b48 $nm-d16 0 16 $tk)] ; b2048 [0,16) vs d16 [$(cmpslots $nm-b2048 $nm-d16 0 16 $tk)] ; b48 [0,4) vs d4 [$(cmpslots $nm-b48 $nm-d4 0 4 $tk)] ; s2048 [0,4) vs d16 [$(cmpslots $nm-s2048 $nm-d16 0 4 $tk)] ; s2048 [8,12) vs d16 [$(cmpslots $nm-s2048 $nm-d16 8 12 $tk)]"
    log "RESULT fn4 182 $nm UNMARKED = plain prefill band (want 0 differ): b48 [16,48) vs p48 [$(cmpslots $nm-b48 $nm-p48 16 48 $tk)] ; b2048 [16,2048) vs p2048 [$(cmpslots $nm-b2048 $nm-p2048 16 2048 $tk)] ; s2048 [4,8) vs p2048 [$(cmpslots $nm-s2048 $nm-p2048 4 8 $tk)] ; s2048 [12,2048) vs p2048 [$(cmpslots $nm-s2048 $nm-p2048 12 2048 $tk)]"
    log "RESULT fn4 182 $nm control (the flag acts; y should differ, h not): b2048 [0,16) vs p2048 [$(cmpslots $nm-b2048 $nm-p2048 0 16 $tk)] ; d16 vs p2048 [0,16) [$(cmpslots $nm-d16 $nm-p2048 0 16 $tk)]"
  fi
done
# ---- the cost: one dual pass vs the segmented passes vs the plain pass (timed chain, the timing geometry), A B B A
usl() { local v; v=$(grep -m1 "^RESULT $2" $O/$1.txt 2>/dev/null | sed -nE 's/.*M=[0-9]+ +([0-9.]+) us.*/\1/p'); echo ${v:-NA}; }
TIMD="--chain --rand-data --x-bf16 --h-rot 1 --pooled 0 --ceiling 640 --warm-ms 500 --iters 10 --reps 5"
for sh in "fn4:${GFN/--ep-rank 0/--ep-rank -1}:16:2032" "ds2:${GDS/--ep-rank 0/--ep-rank -1} --bits 2:24:488" "ds25:${GDS/--ep-rank 0/--ep-rank -1} --bits 2.5:24:488" "ds2:${GDS/--ep-rank 0/--ep-rank -1} --bits 2:24:2024"; do
  nm=${sh%%:*}; rest=${sh#*:}; P=${rest##*:}; rest=${rest%:*}; D=${rest##*:}; geo=${rest%:*}; MT=$((D + P))
  for ab in a b c d; do
    if [ $ab = a ] || [ $ab = d ]; then
      tm t-$nm-$D-$P-dual-$ab $TIMD $geo --chain-band _pf --M $MT --band 0:$D
      tm t-$nm-$D-$P-plain-$ab $TIMD $geo --chain-band _pf --M $MT
    else
      tm t-$nm-$D-$P-plain-$ab $TIMD $geo --chain-band _pf --M $MT
      tm t-$nm-$D-$P-dual-$ab $TIMD $geo --chain-band _pf --M $MT --band 0:$D
    fi
  done
  tm t-$nm-$D-$P-segP $TIMD $geo --chain-band _pf --M $P
  tm t-$nm-$D-$P-seg16 $TIMD $geo --chain-band _m16 --M 16
  [ $D -gt 16 ] && tm t-$nm-$D-$P-segR $TIMD $geo --chain-band _m16 --M $((D - 16))
  if [ "$DRY" != 1 ]; then
    k='exl3r4d chain'
    seg=$(python3 -c 'import sys; v=[float(x) for x in sys.argv[1:] if x not in ("NA","")]; print("%.1f" % sum(v) if v else "NA")' $(usl t-$nm-$D-$P-segP "$k") $(usl t-$nm-$D-$P-seg16 "$k") $([ $D -gt 16 ] && usl t-$nm-$D-$P-segR "$k"))
    log "RESULT fn4 182 cost $nm decode $D + prefill $P rows, us a layer (chain): dual one pass $(usl t-$nm-$D-$P-dual-a "$k") / $(usl t-$nm-$D-$P-dual-b "$k") / $(usl t-$nm-$D-$P-dual-c "$k") / $(usl t-$nm-$D-$P-dual-d "$k") | plain one pass $(usl t-$nm-$D-$P-plain-a "$k") / $(usl t-$nm-$D-$P-plain-b "$k") / $(usl t-$nm-$D-$P-plain-c "$k") / $(usl t-$nm-$D-$P-plain-d "$k") | segmented (prefill $P + decode pieces <= 16) $seg = $(usl t-$nm-$D-$P-segP "$k") + $(usl t-$nm-$D-$P-seg16 "$k")$([ $D -gt 16 ] && echo " + $(usl t-$nm-$D-$P-segR "$k")")"
  fi
done
log "RESULT fn4 182-fn-dual: done"
