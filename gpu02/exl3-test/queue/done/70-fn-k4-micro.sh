#!/bin/bash
# 83-fn-exl3-k4-micro.sh (queue item, 7.11.2026, queue-ar; gpu02, ONE card, every radiance container DOWN; ~5 min) -- does
# the b96 tile-load fix (exl3-gu25b/gu25c: a lane's tile words as one raw_buffer_load_b96 at K 2.5 and K 3, PF 4) have a
# 4-bit counterpart for Qwen3.8-Flash-Next's EXL3 4.05 bpw experts?
# Source reading (radiance-exl3 gu25c 978e264, kernels/exl3r4d/exl3r4d.hip): b96_k<BITS>() is `BITS != 4 && ...`, so K 4
# keeps its own buffer form -- three raw_buffer_load_b32 a tile a lane (the words d-1, d, d+1, i.e. the same three
# consecutive words the b96 form loads, wrapping at the tile's ends) with no s_wait drains in its loop; K 2.5 / 3 had
# 40 loads and six s_wait_loadcnt 0 an iteration before the fix. So base and gu25c run the SAME K4 code: the K4 rows below
# are a control (equal within noise), and the question is whether K4's 3 x b32 leaves time on the table against the
# b96-loaded K 2.5 / 3 at the same shapes (GB/s a dispatch). If K4's rate is at or above b96 K3's, a K4 b96 variant has
# nothing to win; if well below, it is worth writing (one 12-byte load + the wrap readlane, as b96_holder does).
# Shapes: perfmoe's defaults ARE Flash-Next's (512 experts, top-10, hidden 2560, width 640, 48 layers), at --tp 2 (the
# rank's widths by parity, as the arch splits them): xeon's and gpu02's TP2.
# Arms: exl3r4d-base.so (f2f3c3f7 kernels) and exl3r4d-gu25c.so (x3c, md5 aa4703c9) from queue/done/bin-gu25, perfmoe of
# the gu25 build (md5 b088bd01); --bits 4 (the 4.05 bpw experts), 3 and 2.5 (the b96 forms, for the rate comparison);
# --op gate_up and down; M 1, 4, 16 (decode at 1 / 4 / 16 sequences, MTP 3 adds rows: M 16 ~ 4 seqs x 4 rows); all VRAM
# (--pooled 0). 2 x 3 x 2 x 3 = 36 runs, ~7 s each (73's pace) -> ~4.5 min.
# Result: per (K, op, M) base vs gu25c us and GB/s, and a summary: K4's GB/s against gu25c's K3 / K2.5 at each M.
# Every run bounded (timeout -k 15 300, --init, named, rm -f). Guards: no radiance container, KFD idle, bins (md5) + image.
# DRYRUN=1 bash 83-fn-exl3-k4-micro.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
B=${K4_BIN:-}
for c in "$Q/bin-gu25" "$HOME/gpu/ds41/exl3-test/queue/done/bin-gu25" "$HOME/gpu/ds41/exl3-test/queue/bin-gu25"; do
  [ -z "$B" ] && [ -e "$c/perfmoe" ] && [ -e "$c/exl3r4d-gu25c.so" ] && [ -e "$c/exl3r4d-base.so" ] && B=$(cd "$c" && pwd)
done
B=${B:-$HOME/gpu/ds41/exl3-test/queue/done/bin-gu25}
O=${K4_OUT:-$HOME/gpu/ds41/exl3-test/runs/fn-k4-micro}
L=${K4_LOG:-$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log}
IMAGE=${K4_IMAGE:-radiance-local:ds41-exl3-117-abf7f9a0-f2f3c3f7}
CARD=${CARD:-0}
GEOM="--tp 2 --experts 512 --topk 10 --hidden 2560 --width 640 --warm-ms 500 --iters 4 --reps 5 --pooled 0"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD"
declare -A LIB=([base]=exl3r4d-base.so [x3c]=exl3r4d-gu25c.so)
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
pm() {   # pm ARM OUTFILE perfmoe-args...
  local arm=$1 out=$2; shift 2
  local cn=k4m-$$-$RANDOM
  run "timeout -k 15 300 docker run --rm --init --stop-timeout 10 --name $cn $DEV -v $B:/b:ro --entrypoint /b/perfmoe $IMAGE --exl3 /b/${LIB[$arm]} $* > $out 2>&1; docker rm -f $cn > /dev/null 2>&1"
}
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT fn-k4: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT fn-k4: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  for f in perfmoe exl3r4d-base.so exl3r4d-gu25c.so; do [ -e "$B/$f" ] || { log "RESULT fn-k4: VIRHE $B/$f missing"; exit 1; }; done
  [ "$(md5sum "$B/exl3r4d-gu25c.so" | cut -c1-8)" = aa4703c9 ] || { log "RESULT fn-k4: VIRHE gu25c md5 is not aa4703c9"; exit 1; }
  [ "$(md5sum "$B/perfmoe" | cut -c1-8)" = b088bd01 ] || { log "RESULT fn-k4: VIRHE perfmoe md5 is not b088bd01"; exit 1; }
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT fn-k4: VIRHE image $IMAGE missing"; exit 1; }
  mkdir -p "$O"
else
  echo "+ mkdir -p $O"
fi
log "RESULT fn-k4: start IMAGE=$IMAGE card $CARD bin $B -> $O (Flash-Next TP2 shapes: $GEOM)"
for k in 4 3 2.5; do for op in gate_up down; do for m in 1 4 16; do for a in base x3c; do
  pm $a $O/$a-k$k-$op-m$m.txt --op $op --M $m --bits $k $GEOM
done; done; done; done
[ "$DRY" = 1 ] && { log "RESULT fn-k4: dry run done"; exit 0; }
python3 - "$O" <<'PY' | while read -r line; do log "$line"; done
import re, sys
O = sys.argv[1]
def res(f):
    try:
        for l in open(f):
            m = re.match(r"RESULT exl3r4d\s+\S*\s*\S+ .*?M=\d+\s+([\d.]+) us .*?([\d.]+) GB/s .*?([\d.]+) MB/dispatch", l)
            if m: return float(m.group(1)), float(m.group(2)), float(m.group(3))
    except OSError:
        pass
    return None
tab = {}
for k in ("4", "3", "2.5"):
    for op in ("gate_up", "down"):
        parts = []
        for M in ("1", "4", "16"):
            a, b = res(f"{O}/base-k{k}-{op}-m{M}.txt"), res(f"{O}/x3c-k{k}-{op}-m{M}.txt")
            if a and b:
                tab[(k, op, M)] = b
                parts.append(f"M{M} base {a[0]:.1f} us {a[1]:.0f} GB/s -> x3c {b[0]:.1f} us {b[1]:.0f} GB/s ({(b[0] / a[0] - 1) * 100:+.1f} %), {b[2]:.2f} MB")
            else:
                parts.append(f"M{M} no RESULT")
        print(f"RESULT fn-k4 K{k} {op} (Flash-Next TP2, all VRAM): " + " | ".join(parts))
summ = []
for op in ("gate_up", "down"):
    for M in ("1", "4", "16"):
        k4, k3, k25 = tab.get(("4", op, M)), tab.get(("3", op, M)), tab.get(("2.5", op, M))
        if k4 and k3 and k25:
            summ.append(f"{op} M{M}: K4 {k4[1]:.0f} GB/s vs b96 K3 {k3[1]:.0f} ({(k4[1] / k3[1] - 1) * 100:+.0f} %), K2.5 {k25[1]:.0f}")
print("RESULT fn-k4 summary (x3c, GB/s a dispatch; K4's code is the same in both builds): " + "; ".join(summ))
PY
log "RESULT fn-k4: done -- $O"
