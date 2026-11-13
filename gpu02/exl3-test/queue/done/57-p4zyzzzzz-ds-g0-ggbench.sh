#!/bin/bash
# 57-p4zyzzzzz-ds-g0-ggbench.sh (queue item, 10.11.2026, ds41-gg lever 1 agent; gpu02, ONE card (CARD, default 0), every
# radiance container DOWN, no server; ~20 min) -- lever 1's form bench over the WHOLE decode band: gemm_nt_ggml's gemv
# kernel (M <= 48) at DeepSeek-V4.1-Flash's dense TP4 shapes -- attn_q_b 8192x1280, attn_output_b 5120x2048, attn_q_a
# 1280x5120, attn_output_a 1024x4096, attn_kv 512x5120, engram_kv 25600x6144, the shared expert at rank 0's width (gate_up
# 1536x5120, down 5120x768) and at ranks 1-3's (1024x5120, 5120x512) -- in q5_K / q6_K / q8_0 (the 2.25 .rad: attention
# q5_K x13 / q6_K x27 layers, shared expert q6_K x27 / q8_0 x13, engram_kv q6_K), at M = 1 2 3 4 5 6 8 12 16 24 32 48 (the
# DSpark verify shapes 6 / 12 / 24 / 48 among them). Forms: the stock rule, the measured table (gg_gemv 1; the stock launch
# where the table has no entry yet), and every pinned rows-a-wave 1 / 2 / 4 x grid cap 256 / 1024, each output compared with
# the rule's BYTE FOR BYTE (exact by construction: a difference is a defect and is logged VIRHE). Two passes: "rot" (the
# weight streamed from DRAM through rotated copies past 256 MiB, mean of 100) and "cold" (--cold 256: 256 MiB read before
# each launch, median of 100). Then tools/gg_forms.py (the ds41-gg tree's generator) writes the form table the engine would
# take from both passes (a form enters only where it was byte-identical in every run, is not the rule's own launch, and
# beat the rule by >= 3 % without losing in either pass) -> runs/xdsm/kbench-gg/r4d_ds41_gg_forms.h, to be committed on
# ds41-gg; and kbsum.py's GGSUM lines (rule vs the best pinned form) for both passes.
# Binaries (queue-ar/dsmain/bin): r4d_selftest-gg and libr4d-gg.so, built from ds41-gg (the image's own libr4d: the switch
# + the seeded table + the RPW 4 form + the pin hook); the HIP runtime from the image (XGG_IMAGE).
# DRYRUN=1 bash 57-p4zyzzzzz-ds-g0-ggbench.sh : prints, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
B=${XGG_BIN:-$HOME/gpu/ds41/exl3-test/queue-ar/dsmain/bin}
H=$HOME/gpu/ds41/exl3-test/queue-ar/dsmain
O=${XGG_OUT:-$HOME/gpu/ds41/exl3-test/runs/xdsm/kbench-gg}
L=$HOME/gpu/ds41/exl3-test/runs/ds41-exl3.log
IMAGE=${XGG_IMAGE:-radiance-local:ds41-gg-4d70490d-5c13e555-l}
CARD=${CARD:-0}
MS=${XGG_MS:-1,2,3,4,5,6,8,12,16,24,32,48}; TYPES=${XGG_TYPES:-q5_K,q6_K,q8_0}; IT=${XGG_ITERS:-100}
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=$CARD"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a "$L"; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
gpurun() {   # gpurun NAME SECONDS docker-run-args...: bounded, named, removed even if a kernel hangs
  local name=$1 t=$2; shift 2
  timeout -k 15 $t docker run --rm --init --stop-timeout 10 --name $name "$@"; local rc=$?
  docker rm -f $name > /dev/null 2>&1 || true
  return $rc
}
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qi radiance && { log "RESULT xgg bench: VIRHE a radiance container is running -- not now"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT xgg bench: VIRHE a KFD client is running (after 60 s wait)"; exit 1; }
  for f in r4d_selftest-gg libr4d-gg.so; do [ -e "$B/$f" ] || { log "RESULT xgg bench: VIRHE $B/$f missing"; exit 1; }; done
  for f in $H/kbsum.py $H/gg_forms.py; do [ -e $f ] || { log "RESULT xgg bench: VIRHE $f missing"; exit 1; }; done
  docker image inspect "$IMAGE" > /dev/null 2>&1 || { log "RESULT xgg bench: VIRHE image $IMAGE not here"; exit 1; }
  mkdir -p "$O"
else
  echo "+ mkdir -p $O"
fi
log "RESULT xgg bench: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) card $CARD bench $(md5sum $B/r4d_selftest-gg 2>/dev/null | cut -c1-8) lib $(md5sum $B/libr4d-gg.so 2>/dev/null | cut -c1-8) M [$MS] types $TYPES iters $IT -> $O"
run "gpurun ggb-$$-rot 3000 $DEV -v $B:/b:ro --entrypoint /b/r4d_selftest-gg $IMAGE /b/libr4d-gg.so --perfgg $MS $IT $TYPES > $O/gg-rot.txt 2>&1"
rc1=$?
run "gpurun ggb-$$-cold 4200 $DEV -v $B:/b:ro --entrypoint /b/r4d_selftest-gg $IMAGE /b/libr4d-gg.so --cold 256 --perfgg $MS $IT $TYPES > $O/gg-cold.txt 2>&1"
rc2=$?
if [ "$DRY" != 1 ]; then
  for p in rot cold; do
    n=$(grep -c '^GGPERF ' $O/gg-$p.txt); nd=$(grep -c 'OUTPUT DIFFERS' $O/gg-$p.txt); nf=$(grep -cE ' rc=-?[0-9]+;' $O/gg-$p.txt)
    log "RESULT xgg bench $p: $n GGPERF lines (of 360 expected), $nd with OUTPUT DIFFERS, $nf with a refused launch; $(head -1 $O/gg-$p.txt | cut -c1-200)"
    [ "$nd" = 0 ] || log "RESULT xgg bench $p: VIRHE a form's output differs from the rule's: $(grep 'OUTPUT DIFFERS' $O/gg-$p.txt | head -3 | cut -c1-300 | tr '\n' '|')"
  done
  [ $rc1 = 0 ] && [ $rc2 = 0 ] || log "RESULT xgg bench: VIRHE r4d_selftest exit rot $rc1 cold $rc2 (1 = a byte difference, else a crash or a timeout)"
fi
# kbsum's GGSUM lines (rule vs the best pinned form), one pass at a time (it reads every gg-*.txt in a directory)
for p in rot cold; do
  run "mkdir -p $O/sum-$p && cp $O/gg-$p.txt $O/sum-$p/ && python3 -I $H/kbsum.py $O/sum-$p > $O/kbsum-$p.txt 2>&1"
done
run "python3 -I $H/gg_forms.py --out $O/r4d_ds41_gg_forms.h $O/gg-rot.txt $O/gg-cold.txt > $O/gg_forms.txt 2>&1"
if [ "$DRY" != 1 ]; then
  log "RESULT xgg bench: generated table $(grep -m1 -oE '[0-9]+ entr(y|ies)' $O/r4d_ds41_gg_forms.h) -> $O/r4d_ds41_gg_forms.h (md5 $(md5sum < $O/r4d_ds41_gg_forms.h | cut -c1-8))"
  # the table vs the rule and vs the best pinned form, per M over every type and shape (rot pass), and the seeded table's own check
  log "RESULT xgg bench table (rot, seeded M 1/4/8 table in this libr4d) vs rule per M: $(python3 - $O/gg-rot.txt <<'PY' 2>&1 | cut -c1-1800
import re, sys
from collections import defaultdict
H = re.compile(r"GGPERF (\S+) (\S+) (\d+)x(\d+) M=(\d+)")
F = re.compile(r" (rule|table|r\dg\d+) ([\d.]+) us")
agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
for l in open(sys.argv[1]):
    h = H.match(l)
    if not h: continue
    f = {n: float(u) for n, u in F.findall(l)}
    if "rule" not in f: continue
    a = agg[int(h.group(5))]
    a[0] += f["rule"]; a[1] += f.get("table", f["rule"]); a[2] += min(v for k, v in f.items() if k != "table"); a[3] += 1
print("; ".join("M%d rule %.0f us, table %+.1f %%, best pinned %+.1f %% (%d keys)" % (m, v[0], 100 * (v[1] / v[0] - 1), 100 * (v[2] / v[0] - 1), v[3])
                for m, v in sorted(agg.items())))
PY
)"
  for p in rot cold; do
    log "RESULT xgg bench GGSUM $p: $(grep -c '^GGSUM' $O/kbsum-$p.txt) lines -> $O/kbsum-$p.txt"
  done
fi
log "RESULT xgg bench: done -- $O"
