#!/bin/bash
# dsm-mclprobe.sh (12.11.2026, MCL builder; the body of items/57-p4zyzzzzz-ds-r00zzz-cz-mcl0probe.sh) -- MCL gate G0 card probe.
# ONE card (logical HIP index CARD, default 0; any gfx1201 card -- the probe reads no PCI address and no card order), every
# radiance container DOWN, no server, no engine start, no evict; ~1 min expected, bounded at 5 min (probe 240 s).
# WHAT: bin/mclprobe (built on fedora-homeamd in radiance-ds41-build:main-e736c019 with the shipped plugin's own HIP flags,
# from ~/gpu/ds41/exl3/mcl/probe/mclprobe.hip) in the int9 image, whose /opt/radiance/share/radiance/kernels/exl3r4d.so IS the
# production EXL3 plugin (5c13e55, sha256 642afedb...; checked below). It writes, under OUT:
#   W  wmma_s0..s5.D  -- v_wmma_f32_16x16x16_f16 on crafted / random / cancellation / real-chain batteries (inputs are
#      regenerated offline): the WMMA's internal sum order and rounding, the one unknown of an exact CPU lane
#   E  vexp_map.bin (109 MB), vexp_exc.bin, vexp_refsamp.bin -- v_exp_f32 on every f32 with |x| < 1 vs an f64 reference
#   X  x_silu_{g,e,s}.bin (48 MB) + log lines: the f16 mix rounding, 1/d correctly rounded, silu as the plugin compiles it
#   G  e2e_k{2,21}_m{1,2,3,4,8,16}.{h,y,sorted} -- the SHIPPED kernels (dlopen of the image's exl3r4d.so: launch_prep +
#      launch_gemm gate_up / down, h_rot 1, limit 10, split-K 1) on synthetic DS-shaped experts (5120 / 2304, 8 experts)
# The verdict is OFFLINE (mclcheck analyze OUT on fedora-homeamd: the CPU reference vs these bytes); this item only measures.
# GUARD (VIRHE): files / md5s, image id, the plugin's sha256 inside the image, a radiance container or KFD client up, the
# probe's exit code, a missing "MCL DONE ok".
# DRYRUN=1 bash dsm-mclprobe.sh : prints every command, checks binary / image / plugin hash; starts nothing GPU-side, writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain
K=$H/mcl
O=${MCL0_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/mcl0}
L=$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
IMAGE=${MCL0_IMAGE:-radiance-local:ds41-int9-0cb6fd31-5c13e555-l}; IMGID=9af3dddd556c
PLUGIN=/opt/radiance/share/radiance/kernels/exl3r4d.so
PLUGSHA=642afedb13bca37cdb6ac830b29244307dbd7cbe5c90bd2237f82b7a75efa215
CARD=${CARD:-0}
T="RESULT mcl0"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
gpurun() {   # gpurun NAME SECONDS docker-run-args...: bounded, named, removed even if a kernel hangs
  local name=$1 t=$2; shift 2
  timeout -k 15 $t docker run --rm --init --stop-timeout 10 --name $name "$@"; local rc=$?
  docker rm -f $name > /dev/null 2>&1 || true
  return $rc
}
miss=0
m=$(md5sum < $K/bin/mclprobe 2>/dev/null | cut -c1-32); want=$(awk '$2 == "mclprobe" {print $1}' $K/bin/MD5SUMS 2>/dev/null)
if [ -z "$m" ] || [ "$m" != "$want" ]; then echo "  (check) $K/bin/mclprobe MISSING or md5 ${m:-none} != ${want:-none}"; miss=$((miss + 1)); else echo "  (check) mclprobe md5 ${m:0:8}"; fi
gid=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null | cut -c8-19)
if [ "$gid" != "$IMGID" ]; then echo "  (check) image $IMAGE id ${gid:-missing} != $IMGID"; miss=$((miss + 1)); else echo "  (check) image $IMAGE $gid"; fi
psha=$(timeout 120 docker run --rm --network none --entrypoint sh "$IMAGE" -c "sha256sum $PLUGIN" 2>/dev/null | cut -c1-64)
if [ "$psha" != "$PLUGSHA" ]; then echo "  (check) plugin sha256 ${psha:-none} != ${PLUGSHA:0:8}"; miss=$((miss + 1)); else echo "  (check) plugin sha256 ${psha:0:8} (the shipped exl3r4d 5c13e55)"; fi
if [ "$DRY" = 1 ]; then
  echo "+ mkdir -p $O; chmod 777 $O"
  echo "+ gpurun mcl0-\$\$ 240 $DEV -v $K/bin:/b:ro -v $O:/o --entrypoint /b/mclprobe $IMAGE /o > $O/probe.txt 2>&1"
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
log "$T: start IMAGE=$IMAGE ($gid) plugin ${psha:0:8} card HIP $CARD mclprobe ${m:0:8} -> $O (~1 min expected, bounded 5 min)"
gpurun mcl0-$$ 240 $DEV -v $K/bin:/b:ro -v $O:/o --entrypoint /b/mclprobe "$IMAGE" /o > $O/probe.txt 2>&1; rc=$?
while IFS= read -r l; do log "$T ${l:0:600}"; done < <(grep -E '^MCL (DEV|W|E|X|G|DONE|VIRHE)' $O/probe.txt)
grep -q '^MCL DONE ok' $O/probe.txt || log "$T: VIRHE no 'MCL DONE ok' (rc $rc: 124/137 = timeout, else a crash; tail: $(tail -2 $O/probe.txt | tr '\n' ' ' | cut -c1-300))"
[ $rc = 0 ] || log "$T: VIRHE mclprobe exit $rc"
log "$T: done -- $O ($(du -sh $O 2>/dev/null | cut -f1)); verdict offline: mclcheck analyze on fedora-homeamd (MCL builder)"
