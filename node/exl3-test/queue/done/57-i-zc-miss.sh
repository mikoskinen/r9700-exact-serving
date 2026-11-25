#!/bin/bash
# 73-zc-miss.sh (queue item, 6.11.2026, queue-ar; gpu02, ONE card, every radiance container DOWN) -- the zero-copy miss cost:
# how fast a decode expert GEMM reads an expert that lives in the pinned host pool (PERF-REPORT plan item 4). The format
# model (tools/fmtsim.py) fitted R = 25 GB/s effective from two engine points; this measures R directly, per unit size.
# perfmoe (gu25 build) --pooled P puts P % of each class's experts (the least routed) in pinned, mapped host memory, both
# libraries -- the engine's host-pool tier, read in place by the GEMM over PCIe. Same routing and geometry as 54 / 57
# (DeepSeek V4.1 Flash per-rank: 96 experts top-6, hidden 5120, width 2304, h_rot 1, EP-4, the slowest rank a layer).
#   unit sizes: exl3r4d --bits 2 (2.25's K2 layers, 8.4 MiB an expert), 2.5 (its K2.5 layers, 10.6), 3 (2.9's K3 layers, 12.7);
#               libr4d w4nl64a8h in the same runs (--r4d, the 4.25-bpw class, 17.9 MiB) -- the 4-bit expert for free
#   ops gate_up and down, M 1 and 4, pooled 0 (all VRAM) and 100 (every live expert from host), cold (48 layers x 2 pools:
#   every dispatch streams); plus hot (--layers 1 --pools 1: the same experts again -- does the card cache host reads?) at
#   M 1 for K2 and K3 gate_up. Host pages are pinned (hipHostMalloc mapped, written before timing), so there is no page-cache
#   cold state; "cold / hot" is the card's caches (L2 + 64 MB MALL).
# Result: per (lib, K, op, M): us at pooled 0 and 100, MB a dispatch, R_marginal = MB / (t100 - t0) and R_abs = MB / t100, as
# RESULT lines; a final RESULT gives R per unit size (gate_up + down chain = an expert's whole read) for fmtsim.py.
# Guards as 54 (14-exl3-bottleneck.sh): no radiance container, KFD idle (60 s wait), bin present. ~10 min (26 perfmoe runs).
# Inputs: bin/ beside this file (or $ZC_BIN): perfmoe and exl3r4d-gu25.so (queue-ar/bin = queue-r2/bin's, sha 7f154bd6 /
# 8078f231); IMAGE: any 1.17 serving image on gpu02 (its ROCm runtime and kernels/libr4d.so for --r4d).
# DRYRUN=1 bash 73-zc-miss.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
B=${ZC_BIN:-$(cd "$(dirname "$0")" && pwd)/bin-zc}
for c in "$Q/bin" "$Q/../bin" "$HOME/gpu/ds41/exl3-test/queue/bin"; do
  [ -z "$B" ] && [ -e "$c/perfmoe" ] && [ -e "$c/exl3r4d-gu25.so" ] && B=$(cd "$c" && pwd)
done
B=${B:-$Q/bin}
O=${ZC_OUT:-$HOME/gpu/ds41/exl3-test/runs/zc-miss}
L=${ZC_LOG:-$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log}
IMAGE=${ZC_IMAGE:-radiance-local:ds41-exl3-117-abf7f9a0-f2f3c3f7}
CARD=${CARD:-0}
R4D=/opt/radiance/share/radiance/kernels/libr4d.so
GEOM="--experts 96 --topk 6 --hidden 5120 --width 2304 --h-rot 1 --ep 4 --ep-rank -1 --warm-ms 500 --iters 4 --reps 5"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
pm() {   # pm OUTFILE perfmoe-args...
  local out=$1; shift
  run "docker run --rm $DEV -v $B:/b:ro --entrypoint /b/perfmoe $IMAGE --exl3 /b/exl3r4d-gu25.so --r4d $R4D $* > $out 2>&1"
}
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT zc-miss: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done   # KFD teardown race (6.11.)
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT zc-miss: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  for f in perfmoe exl3r4d-gu25.so; do [ -e "$B/$f" ] || { log "RESULT zc-miss: VIRHE $B/$f missing"; exit 1; }; done
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT zc-miss: VIRHE image $IMAGE missing"; exit 1; }
  mkdir -p "$O"
else
  echo "+ mkdir -p $O"
fi
log "RESULT zc-miss: start IMAGE=$IMAGE card $CARD bin $B ($(sha256sum $B/perfmoe 2>/dev/null | cut -c1-8) / $(sha256sum $B/exl3r4d-gu25.so 2>/dev/null | cut -c1-8)) -> $O"
for k in 2 2.5 3; do for op in gate_up down; do for m in 1 4; do for p in 0 100; do
  pm $O/k$k-$op-m$m-p$p.txt --op $op --M $m --bits $k $GEOM --pooled $p
done; done; done; done
for k in 2 3; do pm $O/k$k-gate_up-m1-p100-hot.txt --op gate_up --M 1 --bits $k $GEOM --pooled 100 --layers 1 --pools 1; done
[ "$DRY" = 1 ] && { log "RESULT zc-miss: dry run done"; exit 0; }
python3 - "$O" <<'PY' | while read -r line; do log "$line"; done
import os, re, sys
O = sys.argv[1]
def res(f):
    """{lib: (us, MB a dispatch)} from perfmoe's RESULT lines"""
    out = {}
    try:
        for l in open(f):
            m = re.match(r"RESULT (exl3r4d|libr4d)\s+\S*\s*\S+ .*?M=\d+\s+([\d.]+) us .*?([\d.]+) MB/dispatch", l)
            if m: out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    except OSError:
        pass
    return out
chain = {}
for k in ("2", "2.5", "3"):
    for op in ("gate_up", "down"):
        for M in ("1", "4"):
            a, b = res(f"{O}/k{k}-{op}-m{M}-p0.txt"), res(f"{O}/k{k}-{op}-m{M}-p100.txt")
            parts = []
            for lib in ("exl3r4d", "libr4d"):
                if lib in a and lib in b:
                    (t0, mb), (t1, mb1) = a[lib], b[lib]
                    rm = mb1 / (t1 - t0) * 1e3 if t1 > t0 else float("nan")
                    parts.append(f"{lib} {t0:.1f} -> {t1:.1f} us, {mb1:.2f} MB: R_marginal {rm:.1f} GB/s, R_abs {mb1 / t1 * 1e3:.1f} GB/s")
                    if lib == "exl3r4d" or k == "2":   # w4a8 rows repeat in every K's runs: K2's stand
                        key = (lib, k if lib == "exl3r4d" else "w4a8", M)
                        c = chain.setdefault(key, [0.0, 0.0, 0.0]); c[0] += t1 - t0; c[1] += mb1; c[2] += t1
            print(f"RESULT zc-miss K{k} {op} M{M} (pooled 0 -> 100, cold): " + (" | ".join(parts) or "no RESULT lines"))
for k in ("2", "3"):
    h, c = res(f"{O}/k{k}-gate_up-m1-p100-hot.txt"), res(f"{O}/k{k}-gate_up-m1-p100.txt")
    print(f"RESULT zc-miss K{k} gate_up M1 pooled 100 hot (1 layer, 1 pool) vs cold: " +
          " | ".join(f"{lib} hot {h[lib][0]:.1f} us vs cold {c[lib][0]:.1f} us" for lib in ("exl3r4d", "libr4d") if lib in h and lib in c))
summ = []
for (lib, k, M), (dt, mb, t1) in sorted(chain.items()):
    summ.append(f"{'EXL3 K' + k if lib == 'exl3r4d' else 'w4a8 4.25'} M{M}: {mb:.1f} MB, +{dt:.0f} us -> R {mb / dt * 1e3:.1f} GB/s")
print("RESULT zc-miss summary (gate_up + down, every live expert from host vs VRAM; fmtsim.py fitted 25 GB/s): " + "; ".join(summ))
PY
log "RESULT zc-miss: done -- $O"
