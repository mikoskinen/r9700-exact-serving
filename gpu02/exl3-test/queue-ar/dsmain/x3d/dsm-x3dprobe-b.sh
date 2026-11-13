#!/bin/bash
# dsm-x3dprobe-b.sh (12.11.2026, X3D-F builder; the body of items/57-p4zyzzzzz-ds-r00zzz-cz-x3d-g12b.sh) -- X3D-F gates 1 + 2 on
# ONE card (logical HIP index CARD, default 0; no PCI address / card order used), every radiance container DOWN, no
# server, no engine start, no compose; ~2-4 min expected, bounded at 8 min (probe 420 s).
# g12b = g12's body (dsm-x3dprobe.sh, unchanged) with: binaries from x3d/bin-b (the g12b plugin: the X3D decode of a
# step's tiles phase-ordered, decode_frags_x3d), tag x3dg12b, and the final verdict line computed from the numbers.
# WHAT: bin-b/x3dprobe (built on fedora-homeamd in radiance-ds41-build:main-e736c019 with the plugin's HIP flags, source
# ~/gpu/ds41/exl3/x3d/probe/x3dprobe.hip) in the int9 image. It loads the image's own shipped exl3r4d (5c13e55, sha256
# 642afedb..., checked below) as OLD and bin-b/exl3r4d-x3d.so (branch ds41-x3d, md5 in bin-b/MD5SUMS) as NEW, side by side:
#   BYTES  DS-shaped synthetic experts, K 2 and 2.5, M 1/2/3/4/8/16, gate_up (prep, h_rot 1, limit 10) + down: OLD vs
#          NEW exl3_x3d 0 vs NEW exl3_x3d 1, h' and y byte for byte, every sorted row (408 a variant)
#   HOOKS  NEW's rad_fn4_accepts / _switch / _value for exl3_x3d (0 / 1 taken, 2 refused, a foreign name 0)
#   TIME   EP4 rank geometry (96 local experts, top 6), K 2, M 1/2/4/8/16: gate_up / down GEMM us a call, ABBA x3, x3d 0 vs 1
# VERDICT  gate 1 PASS = 'X3D DONE ok' + exit 0 (every byte row identical, hooks ok); gate 2 PASS = for BOTH gate_up and down
#          the mean over M 1/2/4/8/16 of 'x3d1 vs x3d0' <= -X3D_GATE2_PCT (default 7.5) %; overall PASS = both gates. 7.5 = the
#          gate's ~10 % K-loop drop seen through the whole call (prep kernel + epilogue included; g12's down, -8.7 %, passed).
# Out runs/xdsm/x3dg12b (~4 MB: probe.txt + NEW x3d 1's e2e rows for mclcheck). Measurement only.
# GUARD (VIRHE): files / md5s, image id, the shipped plugin's sha256 inside the image, a radiance container or KFD client
# up, the probe's exit code, a missing "X3D DONE ok" (it says VIRHE on any byte difference or a hook mismatch).
# DRYRUN=1 bash dsm-x3dprobe-b.sh : prints every command, checks binaries / image / plugin hash; starts nothing GPU-side, writes nothing.
set -u
DRY=${DRYRUN:-0}
H=${X3D_QHOME:-$HOME/gpu/ds41/exl3-test/queue-ar/dsmain}
K=$H/x3d; KB=$K/bin-b
O=${X3D_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/x3dg12b}
L=${X3D_LOG:-$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log}
IMAGE=${X3D_IMAGE:-radiance-local:ds41-int9-0cb6fd31-5c13e555-l}; IMGID=9af3dddd556c
PLUGIN=/opt/radiance/share/radiance/kernels/exl3r4d.so
PLUGSHA=642afedb13bca37cdb6ac830b29244307dbd7cbe5c90bd2237f82b7a75efa215
CARD=${CARD:-0}
T="RESULT x3dg12b"
GP=${X3D_GATE2_PCT:-7.5}
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
gpurun() {   # gpurun NAME SECONDS docker-run-args...: bounded, named, removed even if a kernel hangs
  local name=$1 t=$2; shift 2
  timeout -k 15 $t docker run --rm --init --stop-timeout 10 --name $name "$@"; local rc=$?
  docker rm -f $name > /dev/null 2>&1 || true
  return $rc
}
miss=0
for f in x3dprobe exl3r4d-x3d.so; do
  m=$(md5sum < $KB/$f 2>/dev/null | cut -c1-32); want=$(awk -v f=$f '$2 == f {print $1}' $KB/MD5SUMS 2>/dev/null)
  if [ -z "$m" ] || [ "$m" != "$want" ]; then echo "  (check) $KB/$f MISSING or md5 ${m:-none} != ${want:-none}"; miss=$((miss + 1)); else echo "  (check) $f md5 ${m:0:8}"; fi
done
gid=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null | cut -c8-19)
if [ "$gid" != "$IMGID" ]; then echo "  (check) image $IMAGE id ${gid:-missing} != $IMGID"; miss=$((miss + 1)); else echo "  (check) image $IMAGE $gid"; fi
psha=$(timeout 120 docker run --rm --network none --entrypoint sh "$IMAGE" -c "sha256sum $PLUGIN" 2>/dev/null | cut -c1-64)
if [ "$psha" != "$PLUGSHA" ]; then echo "  (check) plugin sha256 ${psha:-none} != ${PLUGSHA:0:8}"; miss=$((miss + 1)); else echo "  (check) OLD plugin sha256 ${psha:0:8} (the shipped exl3r4d 5c13e55)"; fi
if [ "$DRY" = 1 ]; then
  echo "+ mkdir -p $O; chmod 777 $O"
  echo "+ gpurun x3dg12b-\$\$ 420 $DEV -v $KB:/b:ro -v $O:/o --entrypoint /b/x3dprobe $IMAGE /o --new /b/exl3r4d-x3d.so > $O/probe.txt 2>&1"
  echo "  (dry) $(docker ps --format '{{.Names}}' | grep -ci radiance) radiance container(s) up now (the run refuses while any is)"
  echo "DRYRUN: $miss MISSING (nothing started, nothing written)"
  [ $miss = 0 ] || exit 1
  exit 0
fi
[ $miss = 0 ] || { log "$T: VIRHE $miss check(s) failed (binary / image / plugin hash) -- not run"; exit 1; }
docker ps --format '{{.Names}}' | grep -qi radiance && { log "$T: VIRHE a radiance container is running -- not now"; exit 1; }
for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
[ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "$T: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
mkdir -p "$O"; chmod 777 "$O"
log "$T: start IMAGE=$IMAGE ($gid) OLD ${psha:0:8} NEW $(md5sum < $KB/exl3r4d-x3d.so | cut -c1-8) card HIP $CARD -> $O (~2-4 min, bounded 8 min)"
gpurun x3dg12b-$$ 420 $DEV -v $KB:/b:ro -v $O:/o --entrypoint /b/x3dprobe "$IMAGE" /o --new /b/exl3r4d-x3d.so > $O/probe.txt 2>&1; rc=$?
while IFS= read -r l; do log "$T ${l:0:400}"; done < <(grep -E '^X3D (DEV|HOOKS|BYTES|TIME|DONE|VIRHE)' $O/probe.txt)
grep -q '^X3D DONE ok' $O/probe.txt || log "$T: VIRHE no 'X3D DONE ok' (rc $rc: 124/137 = timeout, else a crash or a byte difference; tail: $(tail -2 $O/probe.txt | tr '\n' ' ' | cut -c1-300))"
[ $rc = 0 ] || log "$T: VIRHE x3dprobe exit $rc"
g1=FAIL; grep -q '^X3D DONE ok' $O/probe.txt && [ $rc = 0 ] && g1=PASS
# per op: n rows, mean of "x3d1 vs x3d0" %, mean of x3d1 vs old %, gate-2 PASS / FAIL (5 rows, mean <= -GP)
g2=$(awk -v G=$GP '/^X3D TIME (gate_up|down) +M/ {
    op = $3; o = ""; x = ""
    for (i = 1; i < NF; i++) { if ($i == "old") o = $(i + 1); if ($i == "x3d1" && $(i + 1) ~ /^[0-9]/) x = $(i + 1) }
    n[op]++; d[op] += $(NF - 1) + 0; if (o + 0 > 0) r[op] += 100 * (x / o - 1) }
  END { all = "PASS"
    for (k = 1; k <= 2; k++) { op = k == 1 ? "gate_up" : "down"
      if (n[op] != 5) { printf "%s %d/5 rows FAIL, ", op, n[op]; all = "FAIL"; continue }
      v = d[op] / 5 <= -G ? "PASS" : "FAIL"; if (v == "FAIL") all = "FAIL"
      printf "%s x3d1 vs x3d0 %+.1f %% (vs old %+.1f %%) %s, ", op, d[op] / 5, r[op] / 5, v }
    printf "-> %s", all }' $O/probe.txt)
case "$g2" in *"-> PASS") g2v=PASS ;; *) g2v=FAIL ;; esac
v=FAIL; [ $g1 = PASS ] && [ $g2v = PASS ] && v=PASS
log "$T: gate 1 (bytes + hooks) $g1; gate 2 (mean over M 1-16 <= -$GP %): $g2"
log "$T: done -- $O; verdict: $v (gate 1 $g1, gate 2 $g2v)"
