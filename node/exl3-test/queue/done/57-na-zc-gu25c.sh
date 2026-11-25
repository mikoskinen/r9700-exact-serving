#!/bin/bash
# 77-zc-gu25c.sh (queue item, 7.11.2026, queue-ar; gpu02, ONE card, every radiance container DOWN) -- the zero-copy miss
# cost again, with the b96 tile load. 73 (57-i-zc-miss, 6.11. 17:58) measured R with exl3r4d-gu25.so, whose K2.5 and K3
# gate_up still had the drained load loop (40 loads, six s_wait_loadcnt 0 an iteration): host-path R K2 28.1 / 28.0 GB/s
# (M1 / M4) but K2.5 23.1 / 19.5 and K3 24.3 / 21.7. If the slow host path at K2.5 / K3 was the loop (each wait exposes a
# full PCIe round trip) and not the bytes, gu25c (exl3-gu25c 978e264: one raw_buffer_load_b96 a lane, 16 loads, waits
# 0x6-0xf at K2.5 and K3) raises it -- a 2.9 gain (35 of 40 layers K3) already in the 61b image.
# Arms (perfmoe of the gu25 build, the same binary as 73, md5 b088bd01):
#   base  exl3r4d-base.so  (f2f3c3f7 kernels, md5 f480a439)
#   b96c  exl3r4d-gu25c.so (b96 at K2.5 and K3, PF 4, md5 aa4703c9 -- the kernels in the 2.9 tuned image)
# Method = 73: perfmoe --pooled P puts P % of each class's experts (the least routed) in pinned, mapped host memory, read
# in place by the GEMM over PCIe; same geometry and routing (DeepSeek V4.1 Flash per rank: 96 experts top-6, hidden 5120,
# width 2304, h_rot 1, EP-4, the slowest rank a layer), cold (48 layers x 2 pools).
#   1. full grid, both arms: K 2.5 and 3 (K 2 = control, its kernels are the same in both), gate_up and down (down is the
#      same in both = control), M 1 and 4, pooled 0 and 100 -> 48 runs
#   2. partial pool, both arms: K 2.5 and 3 gate_up M1 at pooled 25 (a few % of the rows from host, mixed with VRAM experts
#      in the same dispatch -- the engine's case at 96 % hit) -> 4 runs
#   libr4d w4nl64a8h rows ride along in every run (--r4d): the host-path drift control (73: w4a8 27.7 / 26.8 GB/s).
# Result RESULT lines: per (arm, K, op, M) us at pooled 0 / 100, MB a dispatch, R_marginal = MB / (t100 - t0), R_abs =
# MB / t100, the miss cost in us an expert; chain R (gate_up + down) per (arm, K, M) next to 73's; b96c vs base in %; the
# partial-pool R (host MB = MB x share of routed rows on host).
# Guards as 73: no radiance container, KFD idle (60 s wait), bins present + gu25c md5 aa4703c9, image present.
# Time: 73 ran 26 perfmoe runs in 3 min 10 s (7.3 s a run); 52 runs here -> ~6.5 min, ~8 min with the guards.
# Inputs: bin dir ($ZC_BIN, else bin-gu25 beside this file, ../bin-gu25, or ~/gpu/ds41/exl3-test/queue/done/bin-gu25 on
# gpu02): perfmoe, exl3r4d-base.so, exl3r4d-gu25c.so. IMAGE: the 1.17 serving image 73 used (ROCm runtime + libr4d.so).
# DRYRUN=1 bash 77-zc-gu25c.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
Q=$(cd "$(dirname "$0")" && pwd)
B=${ZC_BIN:-}
for c in "$Q/bin-gu25" "$Q/../bin-gu25" "$HOME/gpu/ds41/exl3-test/queue/done/bin-gu25" "$HOME/gpu/ds41/exl3-test/queue/bin-gu25"; do
  [ -z "$B" ] && [ -e "$c/perfmoe" ] && [ -e "$c/exl3r4d-gu25c.so" ] && [ -e "$c/exl3r4d-base.so" ] && B=$(cd "$c" && pwd)
done
B=${B:-$Q/bin-gu25}
O=${ZC_OUT:-$HOME/gpu/ds41/exl3-test/runs/zc-gu25c}
L=${ZC_LOG:-$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log}
IMAGE=${ZC_IMAGE:-radiance-local:ds41-exl3-117-abf7f9a0-f2f3c3f7}
CARD=${CARD:-0}
R4D=/opt/radiance/share/radiance/kernels/libr4d.so
GEOM="--experts 96 --topk 6 --hidden 5120 --width 2304 --h-rot 1 --ep 4 --ep-rank -1 --warm-ms 500 --iters 4 --reps 5"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD"
declare -A LIB=([base]=exl3r4d-base.so [b96c]=exl3r4d-gu25c.so)
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
pm() {   # pm ARM OUTFILE perfmoe-args...
  local arm=$1 out=$2; shift 2
  local cn=zcg-$$-$RANDOM   # 7.11. coordinator: bounded like gpurun (timeout -k, --init, named, rm -f)
  run "timeout -k 15 300 docker run --rm --init --stop-timeout 10 --name $cn $DEV -v $B:/b:ro --entrypoint /b/perfmoe $IMAGE --exl3 /b/${LIB[$arm]} --r4d $R4D $* > $out 2>&1; docker rm -f $cn > /dev/null 2>&1"
}
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT zc-gu25c: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done   # KFD teardown race (6.11.)
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT zc-gu25c: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  for f in perfmoe exl3r4d-base.so exl3r4d-gu25c.so; do [ -e "$B/$f" ] || { log "RESULT zc-gu25c: VIRHE $B/$f missing"; exit 1; }; done
  [ "$(md5sum "$B/exl3r4d-gu25c.so" | cut -c1-8)" = aa4703c9 ] || { log "RESULT zc-gu25c: VIRHE $B/exl3r4d-gu25c.so md5 is not aa4703c9"; exit 1; }
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT zc-gu25c: VIRHE image $IMAGE missing"; exit 1; }
  mkdir -p "$O"
else
  echo "+ mkdir -p $O"
fi
log "RESULT zc-gu25c: start IMAGE=$IMAGE card $CARD bin $B (md5 perfmoe $(md5sum $B/perfmoe 2>/dev/null | cut -c1-8), base $(md5sum $B/exl3r4d-base.so 2>/dev/null | cut -c1-8), gu25c $(md5sum $B/exl3r4d-gu25c.so 2>/dev/null | cut -c1-8)) -> $O"
for k in 2.5 3 2; do for op in gate_up down; do for m in 1 4; do for p in 0 100; do for a in base b96c; do
  pm $a $O/$a-k$k-$op-m$m-p$p.txt --op $op --M $m --bits $k $GEOM --pooled $p
done; done; done; done; done
for k in 2.5 3; do for a in base b96c; do
  pm $a $O/$a-k$k-gate_up-m1-p25.txt --op gate_up --M 1 --bits $k $GEOM --pooled 25
done; done
[ "$DRY" = 1 ] && { log "RESULT zc-gu25c: dry run done"; exit 0; }
python3 - "$O" <<'PY' | while read -r line; do log "$line"; done
import re, sys
O = sys.argv[1]
def res(f):
    """{lib: (us, MB a dispatch)}, live experts a dispatch, share of routed rows on host (mean over classes)"""
    out, live, share = {}, float("nan"), []
    try:
        for l in open(f):
            m = re.match(r"RESULT (exl3r4d|libr4d)\s+\S*\s*\S+ .*?M=\d+\s+([\d.]+) us .*?([\d.]+) MB/dispatch", l)
            if m: out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
            m = re.match(r"perfmoe: ([\d.]+) live experts a dispatch", l)
            if m: live = float(m.group(1))
            m = re.match(r"perfmoe: pooled class \d+: .*?([\d.]+)% of routed rows", l)
            if m: share.append(float(m.group(1)) / 100)
    except OSError:
        pass
    return out, live, (sum(share) / len(share) if share else float("nan"))
REF73 = {("2", "1"): 28.1, ("2", "4"): 28.0, ("2.5", "1"): 23.1, ("2.5", "4"): 19.5, ("3", "1"): 24.3, ("3", "4"): 21.7}
chain, w4 = {}, {}
for k in ("2.5", "3", "2"):
    for op in ("gate_up", "down"):
        for M in ("1", "4"):
            parts = []
            for a in ("base", "b96c"):
                (r0, live, _), (r1, _, _) = res(f"{O}/{a}-k{k}-{op}-m{M}-p0.txt"), res(f"{O}/{a}-k{k}-{op}-m{M}-p100.txt")
                if "exl3r4d" in r0 and "exl3r4d" in r1:
                    (t0, _), (t1, mb) = r0["exl3r4d"], r1["exl3r4d"]
                    rm = mb / (t1 - t0) * 1e3 if t1 > t0 else float("nan")
                    parts.append(f"{a} {t0:.1f} -> {t1:.1f} us, {mb:.2f} MB: R_marginal {rm:.1f} GB/s, R_abs {mb / t1 * 1e3:.1f} GB/s, "
                                 f"miss {(t1 - t0) / live:.0f} us an expert")
                    c = chain.setdefault((a, k, M), [0.0, 0.0, 0.0]); c[0] += t1 - t0; c[1] += mb; c[2] += t1
                else:
                    parts.append(f"{a} no RESULT lines")
                if "libr4d" in r0 and "libr4d" in r1:
                    (t0, _), (t1, mb) = r0["libr4d"], r1["libr4d"]
                    c = w4.setdefault(M, [0.0, 0.0]); c[0] += t1 - t0; c[1] += mb
            print(f"RESULT zc-gu25c K{k} {op} M{M} (pooled 0 -> 100, cold): " + " | ".join(parts))
summ = []
for k in ("2.5", "3", "2"):
    for M in ("1", "4"):
        r = {a: chain[(a, k, M)][1] / chain[(a, k, M)][0] * 1e3 for a in ("base", "b96c") if (a, k, M) in chain and chain[(a, k, M)][0] > 0}
        if len(r) == 2:
            summ.append(f"K{k} M{M}: base {r['base']:.1f} / b96c {r['b96c']:.1f} GB/s ({(r['b96c'] / r['base'] - 1) * 100:+.1f} %; 73 gu25 {REF73[(k, M)]})")
print("RESULT zc-gu25c summary R chain (gate_up + down, every live expert from host vs VRAM): " + "; ".join(summ))
print("RESULT zc-gu25c drift control w4a8 4.25 chain R (all runs, both arms; 73: M1 27.7, M4 26.8): " +
      "; ".join(f"M{M} {c[1] / c[0] * 1e3:.1f} GB/s" for M, c in sorted(w4.items()) if c[0] > 0))
parts = []
for k in ("2.5", "3"):
    for a in ("base", "b96c"):
        (r0, _, _), (r25, _, sh), (r1, _, _) = (res(f"{O}/{a}-k{k}-gate_up-m1-p{p}.txt") for p in (0, 25, 100))
        if all("exl3r4d" in r for r in (r0, r25, r1)):
            t0, tp, (t1, mb) = r0["exl3r4d"][0], r25["exl3r4d"][0], r1["exl3r4d"]
            hmb = mb * sh
            rp = hmb / (tp - t0) * 1e3 if tp > t0 else float("nan")
            parts.append(f"K{k} {a}: {sh * 100:.1f} % rows on host, {t0:.1f} -> {tp:.1f} us, {hmb:.2f} host MB -> R {rp:.1f} GB/s "
                         f"(full pool {mb / (t1 - t0) * 1e3:.1f})")
print("RESULT zc-gu25c partial pool (gate_up M1, pooled 25): " + " | ".join(parts))
PY
log "RESULT zc-gu25c: done -- $O"
