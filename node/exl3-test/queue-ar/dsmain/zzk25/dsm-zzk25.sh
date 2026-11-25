#!/bin/bash
# dsm-zzk25.sh (13.11.2026, K25U builder for ds-main-port; the body of items/57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzk25.sh) -- lever
# K25U's card checks on ONE card BY CHIP UUID (ROCR_VISIBLE_DEVICES=GPU-<unique_id>, default GPU-dbba5c851cfef7b3), every radiance
# container DOWN, no server, no engine start, no compose; ~3-5 min expected, bounded at 10 min (probe 540 s).
# WHAT: zzk25/bin/k25probe (built on fedora-homeamd in radiance-ds41-build:main-e736c019 with the plugin's HIP flags, source
# ~/gpu/ds41/exl3/x3d/k25u/probe/k25probe.hip) run in the int14 image (the DS base). It loads the image's own shipped exl3r4d
# (ds41-x3d 579d0bc, md5 9823cdde, checked below) as OLD and zzk25/bin/exl3r4d-k25u.so (branch ds41-k25u; md5 in bin/MD5SUMS; the
# SAME file the K25U image ships, sha256 checked against the image) as NEW, side by side:
#   HOOKS   NEW's rad_fn4_accepts / _switch / _value for exl3_x3d (0..4 taken and read back, 5 / -1 refused, a foreign name 0);
#           OLD refuses 2
#   BYTES   DS-shaped synthetic experts (mcl G set, 8 experts), K 2 and 2.5, M 1/2/3/4/8/16, gate_up (prep, h_rot 1, limit 10) +
#           down: OLD exl3_x3d 0 (reference) vs OLD 1 vs NEW 0 / 1 / 2 / 3 / 4 (+ NEW 2 again), h' and y byte for byte, every
#           sorted row (408 a variant); BYTES96: the EP4 timing geometry (96 local experts), K 2.5 and K 2, M 1-16, OLD 1 vs NEW 1..4
#   TIME    EP4 rank geometry (96 local experts of K 2.5, top 6), M 1/2/4/8/16: GEMM us a call; gate_up arms off (exl3_x3d 1) /
#           G3 (2) / G2D (3) / UB96 (4) in blocks A B C D D C B A x 3, down off (1) / U4 (2) A B B A x 3; medians; each arm vs off
# VERDICT  gate 1 PASS = 'K25 DONE ok' + exit 0 (every byte row identical at every switch value, hooks ok); speed: per arm the mean
#          over M 1/2/4/8/16 of 'vs off' (gate_up G3 / G2D / UB96, down U4) and the BEST gate_up value (the lowest mean) -> the
#          exl3_x3d value for the server arm. No speed threshold (the coordinator reads the numbers).
# Out runs/xdsm/zzk25 (~2 MB: probe.txt). Measurement only. Tag "RESULT zzk25 ".
# GUARD (VIRHE): files / md5s, both image ids, the OLD plugin's md5 in the int14 image, NEW = the K25U image's plugin (sha256), the
# card's unique_id in sysfs, disk >= ZZK_DISKMIN_MB (11000) MB free on / and /tmp inodes < 80 %, a radiance container or KFD
# client up, the probe's exit code, a missing "K25 DONE ok" (it says VIRHE on any byte difference or a hook mismatch).
# DRYRUN=1 bash dsm-zzk25.sh : prints every command, checks binaries / images / plugin hashes / the card; starts nothing GPU-side,
# writes nothing. ZZK_MOCK=1 (queue-ar/tools/mockrad only, never in the queue): the image-id / plugin-hash checks are logged, not
# counted (the mock's docker shim has fake ids and no image files), the KFD-client wait is skipped (the mock reads the real sysfs);
# the probe's GPU run is not emulated there.
set -u
DRY=${DRYRUN:-0}
H=${ZZK_QHOME:-$HOME/gpu/ds41/exl3-test/queue-ar/dsmain}
K=$H/zzk25; KB=$K/bin
O=${ZZK_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/zzk25}
L=${ZZK_LOG:-$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log}
IMAGE=${ZZK_IMAGE:-radiance-local:ds41-int14c-678c176f-579d0bc2-l}; IMGID=${ZZK_IMGID:-a7ca6bcdf07b}
K25IMAGE=${ZZK_K25IMAGE:-radiance-local:ds41-k25u-678c176f-38b88ec6-l}; K25IMGID=${ZZK_K25IMGID:-bc91c460c13f}
PLUGIN=/opt/radiance/share/radiance/kernels/exl3r4d.so
PLUGMD5=9823cddefe8c9b06baedeb3e8070b99c
UUID=${ZZK_UUID:-GPU-dbba5c851cfef7b3}
DISKMIN=${ZZK_DISKMIN_MB:-11000}
MOCK=${ZZK_MOCK:-0}
mk() { if [ "$MOCK" = 1 ]; then echo "  (mock: not counted) $*"; else echo "  (check) $*"; miss=$((miss + 1)); fi; }
T="RESULT zzk25"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$UUID"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
gpurun() {   # gpurun NAME SECONDS docker-run-args...: bounded, named, removed even if a kernel hangs
  local name=$1 t=$2; shift 2
  timeout -k 15 $t docker run --rm --init --stop-timeout 10 --name $name "$@"; local rc=$?
  docker rm -f $name > /dev/null 2>&1 || true
  return $rc
}
miss=0
for f in k25probe exl3r4d-k25u.so; do
  m=$(md5sum < $KB/$f 2>/dev/null | cut -c1-32); want=$(awk -v f=$f '$2 == f {print $1}' $KB/MD5SUMS 2>/dev/null)
  if [ -z "$m" ] || [ "$m" != "$want" ]; then echo "  (check) $KB/$f MISSING or md5 ${m:-none} != ${want:-none}"; miss=$((miss + 1)); else echo "  (check) $f md5 ${m:0:8}"; fi
done
gid=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null | cut -c8-19)
if [ "$gid" != "$IMGID" ]; then mk "image $IMAGE id ${gid:-missing} != $IMGID"; else echo "  (check) image $IMAGE $gid"; fi
kid=$(docker image inspect -f '{{.Id}}' "$K25IMAGE" 2>/dev/null | cut -c8-19)
if [ "$kid" != "$K25IMGID" ]; then mk "K25U image $K25IMAGE id ${kid:-missing} != $K25IMGID"; else echo "  (check) K25U image $K25IMAGE $kid"; fi
pmd5=$(timeout 120 docker run --rm --network none --entrypoint sh "$IMAGE" -c "md5sum $PLUGIN" 2>/dev/null | cut -c1-32)
if [ "$pmd5" != "$PLUGMD5" ]; then mk "OLD plugin md5 ${pmd5:-none} != ${PLUGMD5:0:8}"; else echo "  (check) OLD plugin md5 ${pmd5:0:8} (the int14 exl3r4d, ds41-x3d 579d0bc)"; fi
ksha=$(timeout 120 docker run --rm --network none --entrypoint sh "$K25IMAGE" -c "sha256sum $PLUGIN" 2>/dev/null | cut -c1-64)
nsha=$(sha256sum < $KB/exl3r4d-k25u.so 2>/dev/null | cut -c1-64)
if [ -z "$ksha" ] || [ "$ksha" != "$nsha" ]; then mk "NEW plugin sha256 ${nsha:0:12} != the K25U image's ${ksha:0:12}"; else echo "  (check) NEW plugin = the K25U image's exl3r4d (sha256 ${ksha:0:12})"; fi
nuid=$(cat /sys/class/drm/card*/device/unique_id 2>/dev/null | grep -cx "${UUID#GPU-}")
if [ "$nuid" != 1 ]; then echo "  (check) card $UUID: $nuid sysfs unique_id matches (want 1)"; miss=$((miss + 1)); else echo "  (check) card $UUID: in sysfs (unique_id)"; fi
free_mb=$(df -BM --output=avail / | tail -1 | tr -dc 0-9); ipct=$(df -i /tmp | awk 'NR==2 {print $5}' | tr -dc 0-9)
if [ "${free_mb:-0}" -lt $DISKMIN ] || [ "${ipct:-100}" -ge 80 ]; then echo "  (check) disk: / ${free_mb} MB free (min $DISKMIN), /tmp inodes ${ipct} % (max 79)"; miss=$((miss + 1)); else echo "  (check) disk: / ${free_mb} MB free, /tmp inodes ${ipct} %"; fi
if [ "$DRY" = 1 ]; then
  echo "+ mkdir -p $O; chmod 777 $O"
  echo "+ gpurun zzk25-\$\$ 540 $DEV -v $KB:/b:ro -v $O:/o --entrypoint /b/k25probe $IMAGE /o --new /b/exl3r4d-k25u.so > $O/probe.txt 2>&1"
  echo "+ RESULT zzk25 lines: DEV / HOOKS / BYTES / BYTES96 / TIME / DONE, then 'gate 1 (bytes + hooks) PASS|FAIL; speed: ...; best gate_up value N'"
  echo "  (dry) $(docker ps --format '{{.Names}}' | grep -ci radiance) radiance container(s) up now (the run refuses while any is)"
  echo "DRYRUN: $miss MISSING (nothing started, nothing written)"
  [ $miss = 0 ] || exit 1
  exit 0
fi
[ $miss = 0 ] || { log "$T: VIRHE $miss check(s) failed (binary / image / plugin hash / card / disk) -- not run"; exit 1; }
docker ps --format '{{.Names}}' | grep -qi radiance && { log "$T: VIRHE a radiance container is running -- not now"; exit 1; }
if [ "$MOCK" = 1 ]; then echo "  (mock: not checked) KFD clients: $(ls -A /sys/class/kfd/kfd/proc 2>/dev/null | wc -l)"; else
for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
[ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "$T: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
fi
mkdir -p "$O"; chmod 777 "$O"
log "$T: start IMAGE=$IMAGE ($gid) OLD ${pmd5:0:8} NEW $(md5sum < $KB/exl3r4d-k25u.so | cut -c1-8) (= $K25IMAGE $kid) card $UUID -> $O (~3-5 min, bounded 10 min)"
gpurun zzk25-$$ 540 $DEV -v $KB:/b:ro -v $O:/o --entrypoint /b/k25probe "$IMAGE" /o --new /b/exl3r4d-k25u.so > $O/probe.txt 2>&1; rc=$?
while IFS= read -r l; do log "$T ${l:0:420}"; done < <(grep -E '^K25 (DEV|HOOKS|BYTES|BYTES96|TIME|DONE|VIRHE)' $O/probe.txt)
grep -q '^K25 DONE ok' $O/probe.txt || log "$T: VIRHE no 'K25 DONE ok' (rc $rc: 124/137 = timeout, else a crash or a byte difference; tail: $(tail -2 $O/probe.txt | tr '\n' ' ' | cut -c1-300))"
[ $rc = 0 ] || log "$T: VIRHE k25probe exit $rc"
g1=FAIL; grep -q '^K25 DONE ok' $O/probe.txt && [ $rc = 0 ] && g1=PASS
# per arm: n rows and the mean of its "vs off" % over M; the best gate_up arm (lowest mean)
sp=$(awk '/^K25 TIME (gate_up|down) +M/ {
    op = $3; arm = ""
    for (i = 1; i <= NF; i++) {
      if ($i == "|") { arm = $(i + 1); continue }
      if (arm != "" && arm != "off" && $i ~ /^[-+][0-9.]+$/ && $(i + 1) == "%") { n[op "/" arm]++; d[op "/" arm] += $i + 0 }
    } }
  END { best = ""; bv = 1e9; val["G3"] = 2; val["G2D"] = 3; val["UB96"] = 4
    split("gate_up/G3 gate_up/G2D gate_up/UB96 down/U4", ks, " ")
    for (k = 1; k <= 4; k++) { key = ks[k]
      if (n[key] != 5) { printf "%s %d/5 rows, ", key, n[key]; continue }
      m = d[key] / 5; printf "%s %+.1f %%, ", key, m
      if (key ~ /^gate_up/ && m < bv) { bv = m; split(key, a, "/"); best = a[2] } }
    if (best != "") printf "-> best gate_up arm %s = exl3_x3d %d (%+.1f %%)", best, val[best], bv; else printf "-> best gate_up arm NONE" }' $O/probe.txt)
log "$T: gate 1 (bytes + hooks, every switch value) $g1; speed (mean over M 1-16 vs off = exl3_x3d 1): $sp"
log "$T: done -- $O; verdict: bytes $g1"
