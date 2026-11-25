#!/bin/bash
# 78-pf-zc.sh (queue item, 7.11.2026, queue-ar; gpu02, ONE card, every radiance container DOWN) -- is the prefill's host
# stream exposed? 61b's /stats over cc_sim u4 (7.11. snapshots): 2.9 prefill/mixed steps stream 57.9 GB a step (all ranks,
# 14.5 GB a rank: every host-pinned expert, 1823 of 3840 units = 21.6 GiB a rank, is read once a chunk) and take 1.84 s a
# step by regression, 46 % of busy time (engine: 211 steps, 1.42 s each, against 2.25's 308 at 0.81 s). If those zero-copy
# reads are serial to the GEMM (exposed), 14.5 GB at 24 GB/s is ~0.60 s of the step and a layer-ahead DMA stage (or larger
# chunks) would take it back; if the prefill GEMM's compute hides them, it is ~0 and the 2.9 prefill gap is the K3 kernels.
# perfmoe (gu25 build, md5 b088bd01) times the PREFILL band (exl3r4d_moe_gate_up_pf -> down_pf, --chain --chain-band _pf)
# on 73's geometry (96 experts top-6, hidden 5120, width 2304, h_rot 1, EP-4, slowest rank) with the gu25c library (the
# 61b kernels, md5 aa4703c9) at chunk rows M 1024 and 4096 (2.9 runs 4096-token chunks: MNBT 4160, span 2), --pooled 0 /
# 50 / 100 (at prefill M every expert is live, so P % of the experts = P % of the bytes from host):
#   K 3 (2.9's 35 layers) and 2 (2.25's 29, the bytes control) at M 1024 and 4096; K 2.5 at M 4096 -> 15 runs
# First run is a smoke (K3 M1024 pooled 0): no RESULT line -> VIRHE + the tail of its output, stop (the _pf band has not
# been timed by perfmoe on gpu02 before).
# Result per (K, M): t at pooled 0 / 50 / 100, MB a layer, host MB, and exposure = (t_P - t_0) / (host MB / 24 GB/s):
# ~1 = the reads add serially (exposed), ~0 = hidden under compute; plus t_100 against max(t_0, MB / 24 GB/s).
# Time: --iters 2 --reps 3 (288 dispatches a run); the slowest (K3 M4096 pooled 100) ~1.3 GB a layer at ~24 GB/s =
# ~55 ms a dispatch -> ~16 s; 15 runs ~3-4 min, ~5 min with the guards. Every docker run bounded (timeout -k 15 300,
# --init, named, rm -f) as the coordinator's gpurun.
# Inputs: bin dir ($ZC_BIN, else bin-gu25 beside this file, ../bin-gu25, ~/gpu/ds41/exl3-test/queue/done/bin-gu25):
# perfmoe, exl3r4d-gu25c.so. IMAGE: 73's (ROCm runtime). DRYRUN=1 bash 78-pf-zc.sh : prints, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
B=${ZC_BIN:-}
for c in "$Q/bin-gu25" "$Q/../bin-gu25" "$HOME/gpu/ds41/exl3-test/queue/done/bin-gu25" "$HOME/gpu/ds41/exl3-test/queue/bin-gu25"; do
  [ -z "$B" ] && [ -e "$c/perfmoe" ] && [ -e "$c/exl3r4d-gu25c.so" ] && B=$(cd "$c" && pwd)
done
B=${B:-$Q/bin-gu25}
O=${ZC_OUT:-$HOME/gpu/ds41/exl3-test/runs/pf-zc}
L=${ZC_LOG:-$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log}
IMAGE=${ZC_IMAGE:-radiance-local:ds41-exl3-117-abf7f9a0-f2f3c3f7}
CARD=${CARD:-0}
GEOM="--experts 96 --topk 6 --hidden 5120 --width 2304 --h-rot 1 --ep 4 --ep-rank -1 --warm-ms 500 --iters 2 --reps 3"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
pm() {   # pm OUTFILE perfmoe-args...
  local out=$1; shift
  local cn=pfz-$$-$RANDOM   # bounded like gpurun (timeout -k, --init, named, rm -f): a hung perfmoe never blocks the queue
  run "timeout -k 15 300 docker run --rm --init --stop-timeout 10 --name $cn $DEV -v $B:/b:ro --entrypoint /b/perfmoe $IMAGE --exl3 /b/exl3r4d-gu25c.so --chain --chain-band _pf $* > $out 2>&1; docker rm -f $cn > /dev/null 2>&1"
}
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT pf-zc: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done   # KFD teardown race (6.11.)
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT pf-zc: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  for f in perfmoe exl3r4d-gu25c.so; do [ -e "$B/$f" ] || { log "RESULT pf-zc: VIRHE $B/$f missing"; exit 1; }; done
  [ "$(md5sum "$B/exl3r4d-gu25c.so" | cut -c1-8)" = aa4703c9 ] || { log "RESULT pf-zc: VIRHE $B/exl3r4d-gu25c.so md5 is not aa4703c9"; exit 1; }
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT pf-zc: VIRHE image $IMAGE missing"; exit 1; }
  mkdir -p "$O"
else
  echo "+ mkdir -p $O"
fi
log "RESULT pf-zc: start IMAGE=$IMAGE card $CARD bin $B (md5 perfmoe $(md5sum $B/perfmoe 2>/dev/null | cut -c1-8), gu25c $(md5sum $B/exl3r4d-gu25c.so 2>/dev/null | cut -c1-8)) -> $O"
pm $O/k3-m1024-p0.txt --M 1024 --bits 3 $GEOM --pooled 0
if [ "$DRY" != 1 ] && ! grep -q "^RESULT exl3r4d" $O/k3-m1024-p0.txt; then
  log "RESULT pf-zc: VIRHE smoke run (K3 M1024 pooled 0, prefill band chain) gave no RESULT line: $(tail -3 $O/k3-m1024-p0.txt | tr '\n' ' ' | cut -c1-400)"
  exit 0
fi
for k in 3 2; do for m in 1024 4096; do for p in 0 50 100; do
  [ "$k$m$p" = "310240" ] && continue
  pm $O/k$k-m$m-p$p.txt --M $m --bits $k $GEOM --pooled $p
done; done; done
for p in 0 50 100; do pm $O/k2.5-m4096-p$p.txt --M 4096 --bits 2.5 $GEOM --pooled $p; done
[ "$DRY" = 1 ] && { log "RESULT pf-zc: dry run done"; exit 0; }
python3 - "$O" <<'PY' | while read -r line; do log "$line"; done
import re, sys
O = sys.argv[1]
R = 24.0   # GB/s, 73 / 77's host-path rate at K3 (full pool, decode band)
def res(f):
    t = mb = None; frac = []
    try:
        for l in open(f):
            m = re.match(r"RESULT exl3r4d\s+\S*\s*\S+ .*?M=\d+\s+([\d.]+) us .*?([\d.]+) MB/dispatch", l)
            if m: t, mb = float(m.group(1)), float(m.group(2))
            m = re.match(r"perfmoe: pooled class \d+: (\d+) of (\d+) experts", l)
            if m: frac.append(int(m.group(1)) / int(m.group(2)))
    except OSError:
        pass
    return t, mb, (sum(frac) / len(frac) if frac else 0.0)
out = []
for k, M in (("3", 1024), ("3", 4096), ("2", 1024), ("2", 4096), ("2.5", 4096)):
    r = {p: res(f"{O}/k{k}-m{M}-p{p}.txt") for p in (0, 50, 100)}
    if r[0][0] is None:
        out.append(f"K{k} M{M}: no RESULT"); continue
    t0, mb, _ = r[0]; parts = [f"t0 {t0 / 1e3:.2f} ms, {mb:.0f} MB a layer"]
    for p in (50, 100):
        t, _, fr = r[p]
        if t is None: parts.append(f"p{p} no RESULT"); continue
        host = mb * (fr if fr else p / 100)
        ser = host / R   # ms if serial at R (MB / GB/s = ms)
        parts.append(f"p{p} {t / 1e3:.2f} ms (+{(t - t0) / 1e3:.2f}; host {host:.0f} MB = {ser:.2f} ms at {R:.0f} GB/s -> exposure {(t - t0) / 1e3 / ser:.2f})")
    t1 = r[100][0]
    if t1 is not None:
        parts.append(f"t100 vs max(t0, MB/R) {t1 / 1e3:.2f} / {max(t0 / 1e3, mb / R):.2f} ms, vs t0 + MB/R {(t0 + mb / R * 1e3) / 1e3:.2f}")
    out.append(f"K{k} M{M}: " + "; ".join(parts))
for o in out: print("RESULT pf-zc prefill band chain, " + o)
print("RESULT pf-zc reading: exposure ~1 -> a 2.9 prefill step's 14.5 GB a rank costs ~0.60 s serial (of 1.42 s): a layer-ahead DMA "
      "stage / larger chunks pay; ~0 -> hidden, the 2.9 prefill gap is the K3 prefill kernels (compare K3 vs K2 t0 per MB)")
PY
log "RESULT pf-zc: done -- $O"
