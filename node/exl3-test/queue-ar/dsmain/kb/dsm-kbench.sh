#!/bin/bash
# dsm-kbench.sh (12.11.2026, ds-main-port kbench builder; the body of items/57-p4zyzzzzz-ds-r00zzz-cz-qkbench.sh) -- ONE card (CARD,
# default 0), every radiance container DOWN, no server; ~3-6 min (estimate), bounded at 13.5 min (kb2_ix 240 + 150 + 120 + 120 s, kb2_m1 180 s).
# Single-card kernel benches that decide whether two audit levers (ARCH-AUDIT-20261112: 7 = IX, 4 = M1) become real levers. Both
# binaries (kb/bin, built on fedora-homeamd's CPU in radiance-ds41-build:main-e736c019 = the int7 image's libr4d toolchain, by
# kb/build.sh from kb/src = ds41-int7 0391a6e libr4d + abi copied verbatim, and kb/bench) run in the int7 image (its HIP runtime
# and its own libr4d.so, loaded by dlopen as LIB = the production kernels through their real launch entries):
# (IX) kb2_ix: index_score's prefill WMMA form (r4d_ds41_sparse.hip ds41_index_score_wm_kernel / index_score_body 2801-2899) at
#   the engine's shapes (deepseek41.cpp 1393-1399, 2053-2077): a 4096-row step scored in 256-row runs (irows at MML 524288) into
#   the [256, width] f32 plane at pitch 524289, width 262145 (ratio 2: layers 2 / 8 / 14) / 524289 (ratio 1: layer 20; + keep:
#   layers 24-36, blocks of 8, 2048 kept + the newest), 32 heads x 128, MXFP4 keys (production IDXKEY=mxfp4), steps ending at
#   136k / 64k / 16k (run 7 timed: rows 1792..2047 of the step). Forms: LIB, STK (the copied stock), exact variants (V128 / EB256
#   / EB512 entries a workgroup, PF / PF512 key prefetch, R2 two rows a wave, LDS / LDSR2 keys unpacked once a workgroup into
#   LDS; keep: V1024 / KPF / KR2 / K256 / K256PF), NI (the stock kernel launched only up to the run's reach: the -inf dead region
#   is not written; exact for index_topk readers, NOT for layer 20's index_candidates), the fp8 forms F8R (rebased, one exponent
#   a vector: exact iff the fp8 WMMA adds as the bf16 one) and F8B (the audit's block dot: not exact by construction), timing-only
#   ablations (no WMMA / no epilogue / no unpack / no store) for the limiter, the bf16-key form. Every form's bytes vs LIB over
#   the whole plane (a sentinel-filled plane, each variant's own run flag), on real-like data and on stress data (block binades
#   spread +-5, so the 128-term dots round).
# (M1) kb2_m1: mhc_post_collapse_rmsnorm + mhc_mix (r4d_ds41_elem.hip 849-1070, 2053-2140) at 4096 / 8 / 1 rows, n 5120, hc 4,
#   real weights (kb/data: blk.9 hc_ffn_* for the pcn inputs, blk.10 hc_attn_* + attn_norm, read from the 2.25 .rad), and F1
#   (K1 = post + collapse + mix's projection / squares in mix_split_kernel's exact order, the stock mix_finish_kernel, K2 = the
#   norm in pcn's order) at 8 and 4 rows a workgroup; bytes of every output vs LIB.
# -> kb/kbsum2.py: CHECKS (every expected byte check ran with its own kernel; exact forms byte-identical; else VIRHE), IXT, IXLIM,
#    IXATTR, IXF8, IXEST, M1, M1EST -- logged as "RESULT xkb2 ...". Out runs/xdsm/kbench2 (ix-136k / ix-64k / ix-16k / ix-stress.txt,
#    m1.txt, kbsum2.txt).
# kb2_ix runs four times (136k / 64k / 16k timed + checked, then the stress checks), each its own bounded container.
# DRYRUN=1 bash dsm-kbench.sh : prints every command, checks the binaries / data / parser / image (md5s), starts nothing, writes
# nothing.
set -u
DRY=${DRYRUN:-0}
H=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain
K=$H/kb
O=${XKB2_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/kbench2}
L=$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
IMAGE=${XKB2_IMAGE:-radiance-local:ds41-int7-0391a6ec-5c13e555-l}
CARD=${CARD:-0}
TIX=${XKB2_TIX:-240 150 120 120}; TM1=${XKB2_TM1:-180}   # seconds: kb2_ix at 136k / 64k / 16k / stress, kb2_m1
IXARGS=${XKB2_IXARGS:-}; M1ARGS=${XKB2_M1ARGS:-}
T="RESULT xkb2"
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
gpurun() {   # gpurun NAME SECONDS docker-run-args...: bounded, named, removed even if a kernel hangs
  local name=$1 t=$2; shift 2
  timeout -k 15 $t docker run --rm --init --stop-timeout 10 --name $name "$@"; local rc=$?
  docker rm -f $name > /dev/null 2>&1 || true
  return $rc
}
# ---- the files the item needs, each with its md5 (bin/MD5SUMS, data/MD5SUMS)
miss=0
chk() {   # chk PATH [MD5]: present (and the md5 when given)
  if [ ! -e "$1" ]; then echo "  (check) $1 MISSING"; miss=$((miss + 1)); return 1; fi
  if [ -n "${2:-}" ]; then
    local m; m=$(md5sum < "$1" | cut -c1-32)
    if [ "$m" != "$2" ]; then echo "  (check) $1 md5 $m != $2 MISSING (stale copy)"; miss=$((miss + 1)); return 1; fi
  fi
  echo "  (check) $1 present${2:+, md5 ${2:0:8}}"
}
for f in kb2_ix kb2_m1; do chk $K/bin/$f "$(awk -v f=$f '$2 == f {print $1}' $K/bin/MD5SUMS 2>/dev/null)"; done
NDATA=0
for f in blk9_hc_ffn_fn blk9_hc_ffn_base blk9_hc_ffn_scale blk10_hc_attn_fn blk10_hc_attn_base blk10_hc_attn_scale blk10_attn_norm; do
  chk $K/data/$f.f32 "$(awk -v f=$f.f32 '$2 == f {print $1}' $K/data/MD5SUMS 2>/dev/null)" && NDATA=$((NDATA + 1))
done
chk $K/kbsum2.py
if docker image inspect "$IMAGE" > /dev/null 2>&1; then
  echo "  (check) image $IMAGE present ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19))"
else echo "  (check) image $IMAGE MISSING"; miss=$((miss + 1)); fi
if [ "$DRY" = 1 ]; then
  echo "+ mkdir -p $O"
  echo "  (dry) the item: $(docker ps --format '{{.Names}}' | grep -ci radiance) radiance container(s) up now (the run refuses while any is)"
else
  [ $miss = 0 ] || { log "$T: VIRHE $miss file(s) missing or stale (see the check lines) -- not run"; exit 1; }
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "$T: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "$T: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  mkdir -p "$O"
fi
log "$T: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) card $CARD kb2_ix $(md5sum < $K/bin/kb2_ix 2>/dev/null | cut -c1-8) kb2_m1 $(md5sum < $K/bin/kb2_m1 2>/dev/null | cut -c1-8) data $NDATA/7 real tensors -> $O (~3-6 min estimate, bounded 13.5 min)"
# kb2_ix in four bounded runs (a fault in one form loses one run's rest, not the item): 136k, 64k, 16k (timed + checked), stress (checked)
set -- $TIX
rc1=0
for part in "136k 139264 --no-stress ${1:-240}" "64k 65536 --no-stress ${2:-150}" "16k 16384 --no-stress ${3:-120}" "stress 65536 --stress-only ${4:-120}"; do
  set -- $part
  run "gpurun dskb2-\$\$-ix$1 $4 $DEV -v $K/bin:/b:ro --entrypoint /b/kb2_ix $IMAGE --ctx $2 $3 $IXARGS > $O/ix-$1.txt 2>&1"; r=$?
  [ $r = 0 ] || rc1=$r
  [ "$DRY" = 1 ] || log "$T ix-$1 (rc $r): $(grep -m1 '^KB2 DEV' $O/ix-$1.txt | cut -c9-100) | $(grep -m1 '^KB2 IXDONE' $O/ix-$1.txt || echo 'NO IXDONE: '"$(tail -1 $O/ix-$1.txt | cut -c1-200)")"
done
run "gpurun dskb2-\$\$-m1 $TM1 $DEV -v $K/bin:/b:ro -v $K/data:/kbdata:ro --entrypoint /b/kb2_m1 $IMAGE --weights /kbdata $M1ARGS > $O/m1.txt 2>&1"; rc2=$?
[ "$DRY" = 1 ] || log "$T m1 (rc $rc2): $(grep -m1 '^KB2 M1W' $O/m1.txt | cut -c5-160) | $(grep -m1 '^KB2 M1DONE' $O/m1.txt || echo 'NO M1DONE: '"$(tail -1 $O/m1.txt | cut -c1-200)")"
run "python3 -I $K/kbsum2.py $O > $O/kbsum2.txt 2>&1"
if [ "$DRY" != 1 ]; then
  cat $O/kbsum2.txt
  while IFS= read -r l; do log "$T ${l:0:2400}"; done < <(grep -E '^(CHECKS|IXT|IXLIM|IXATTR|IXF8|IXEST|M1 |M1EST|VIRHE)' $O/kbsum2.txt)
  grep -q '^CHECKS' $O/kbsum2.txt || log "$T: VIRHE kbsum2.py printed no CHECKS line ($(tail -2 $O/kbsum2.txt | tr '\n' ' ' | cut -c1-300))"
  [ $rc1 = 0 ] || log "$T: VIRHE a kb2_ix run exited $rc1 (1 = a VIRHE inside (a byte difference, a form that did not run), 124/137 = timeout, else a crash)"
  [ $rc2 = 0 ] || log "$T: VIRHE kb2_m1 exit $rc2 (1 = a VIRHE inside, 124/137 = timeout, else a crash)"
else
  echo "DRYRUN: $miss MISSING, 0 VIRHE (nothing started, nothing written)"
  [ $miss = 0 ] || exit 1
fi
log "$T: done -- $O"
