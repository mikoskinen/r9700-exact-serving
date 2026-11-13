#!/bin/bash
# 215-fn-sdma2.sh (queue item, 12.11.2026, SDMA-AR agent; gpu02, all 4 cards, every radiance container DOWN; selftest only, <= 15 min)
# -- SDMA-AR: THE 21 MB ANOMALY + THE ENGINE PATH'S BYTES. Image radiance-local:c16-x3sdma2 (branch fn4-x3-sp3-sdma fd9db762 =
# fn4-x3-sp3 69917ad7 + r4d_ar_sdma.hip v2: slot sets by launch parity, a device-side chain between launches, RS / AG halves
# over any shard table; switch ar_sdma in libr4d + core; r4d_selftest --sdmaeng4, --sdmaar4 R4D_SDMA_TL, R4D_SDMA_MAXCOPY).
#  1. --sdmaar4 engine 1 with the per-launch timeline (R4D_SDMA_TL=1: chain wait | go1 -> hop 1 landed | fold | go2 -> hop 2
#     landed | assemble | gap, median us a rank) over 10.5 .. 42 MB (sdma1: 21 MB +39..+54 % against the CU xpush, 10.5 / 42 MB
#     +5..+13 %): where the 21 MB time goes.
#  2. the same at 21 MB alone (the context sized for 21 MB, not 42: the slot layout), and with 2 MiB / 1 MiB COPY packets
#     (R4D_SDMA_MAXCOPY; the default 0x3fffe0 cuts a 5.24 MB shard into 4 MiB - 32 B + 1 MiB + 32 B, unaligned).
#  3. --sdmaeng4 2048x2560 (the pipelined prefill's piece) and 4096x2560 (a step): the engine's rows (lane AR, SP raw RS,
#     SP raw AG) at sdma 0 (CU) vs 1 (copy engines) -- every output compared with the CU path's and the fold, a cross-stream
#     chain test, then each op's time alone and beside a VRAM-bound kernel stream (the stream's loss a collective).
# DRYRUN=1 bash 215-fn-sdma2.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log
IMG=${XSDMA_IMAGE:-radiance-local:c16-x3sdma2}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
CARDS=$C3,$C83,$C86,$C03
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
st() {   # st NAME TIMEOUT ENV... -- ARGS : r4d_selftest on the four cards -> $O/sdma2-NAME.txt
  local name=$1 to=$2; shift 2; local envs=()
  while [ "$1" != "--" ]; do envs+=(-e "$1"); shift; done; shift
  local f=$O/sdma2-$name.txt
  if [ "$DRY" = 1 ]; then echo "+ timeout -k 15 $to docker run --rm --init --name fn4-sdma2-$name --network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$CARDS ${envs[*]} --entrypoint /opt/radiance/bin/r4d_selftest $IMG /opt/radiance/share/radiance/kernels/libr4d.so $* > $f"; return 0; fi
  timeout -k 15 $to docker run --rm --init --name fn4-sdma2-$name-$$ --network none --device /dev/kfd --device /dev/dri \
    --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$CARDS "${envs[@]}" --entrypoint /opt/radiance/bin/r4d_selftest $IMG \
    /opt/radiance/share/radiance/kernels/libr4d.so "$@" > $f 2>&1
  local rc=$?; docker rm -f fn4-sdma2-$name-$$ > /dev/null 2>&1
  log "RESULT fn4 sdma2 $name: exit $rc ($f)"
  return 0
}
arsum() {   # arsum NAME : the --sdmaar4 rows and phases as RESULT lines
  [ "$DRY" = 1 ] && { echo "+ summarise $O/sdma2-$1.txt"; return 0; }
  local f=$O/sdma2-$1.txt
  grep -E 'rank [0-9]: rc|error word|absent' $f | head -6 | while IFS= read -r l; do log "RESULT fn4 sdma2 $1 note: $(echo "$l" | sed 's/^ *//' | cut -c1-200)"; done
  python3 - $f <<'PY' | while IFS= read -r l; do log "RESULT fn4 sdma2 $1 $l"; done
import json, sys
ph = {}
rows = []
for l in open(sys.argv[1], errors="replace"):
    if l.startswith("SDMAPH {"):
        d = json.loads(l[7:]); ph[(d["rows"], d["width"])] = d["phases_us"]
    elif l.startswith("SDMAAR {"):
        rows.append(json.loads(l[7:]))
for d in rows:
    p = ph.get((d["rows"], d["width"]))
    ps = ""
    if p:
        mx = [max(r[k] for r in p) for k in range(6)]
        ps = "; phases max over ranks: chain %.1f | hop1 %.1f | fold %.1f | hop2 %.1f | asm %.1f | gap %.1f us" % tuple(mx)
    print("%dx%d (%.2f MB): bad oop %d / in-place %d / vs CU %d%s; SDMA %.1f us vs CU %.1f us (%+.1f %%)%s" % (
        d["rows"], d["width"], d["mb"], d["bad_oop"], d["bad_inplace"], d["bad_vs_cu"], " FAILED" if d["failed"] else "",
        d["us_sdma"], d["us_cu"], 100.0 * (d["us_sdma"] - d["us_cu"]) / d["us_cu"] if d["us_cu"] else 0.0, ps))
PY
}
log "RESULT fn4 sdma2: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)), cards $CARDS"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 sdma2: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 sdma2: VIRHE image $IMG missing"; exit 1; }
  mkdir -p $O
fi
# 1. the sweep with the timeline
st tl 600 R4D_SDMA_ENGINE=1 R4D_SDMA_SCR=1 R4D_SDMA_CHECKS=100 R4D_SDMA_TL=1 -- --sdmaar4 4096x5120,5120x2560,4096x2560,4096x2048,3072x2560,2048x2560 100
arsum tl
# 2. 21 MB alone; 2 MiB and 1 MiB packets
st tl21 300 R4D_SDMA_ENGINE=1 R4D_SDMA_SCR=1 R4D_SDMA_CHECKS=100 R4D_SDMA_TL=1 -- --sdmaar4 4096x2560 100
arsum tl21
st mc2m 400 R4D_SDMA_ENGINE=1 R4D_SDMA_SCR=1 R4D_SDMA_CHECKS=100 R4D_SDMA_TL=1 R4D_SDMA_MAXCOPY=0x200000 -- --sdmaar4 4096x5120,4096x2560,2048x2560 100
arsum mc2m
st mc1m 400 R4D_SDMA_ENGINE=1 R4D_SDMA_SCR=1 R4D_SDMA_CHECKS=100 R4D_SDMA_TL=1 R4D_SDMA_MAXCOPY=0x100000 -- --sdmaar4 4096x5120,4096x2560,2048x2560 100
arsum mc1m
# 3. the engine path
st eng 900 R4D_SDMA_ENGINE=1 R4D_SDMA_CHECKS=200 -- --sdmaeng4 2048x2560,4096x2560 50
if [ "$DRY" != 1 ]; then
  f=$O/sdma2-eng.txt
  grep -E 'rank [0-9]: rc|NO COPY-ENGINE|absent|fn4 SDMA-AR' $f | head -6 | while IFS= read -r l; do log "RESULT fn4 sdma2 eng note: $(echo "$l" | sed 's/^ *//' | cut -c1-220)"; done
  python3 - $f <<'PY' | while IFS= read -r l; do log "RESULT fn4 sdma2 eng $l"; done
import json, sys
for l in open(sys.argv[1], errors="replace"):
    if l.startswith("SDMAENG {"):
        d = json.loads(l[8:])
        s = "%dx%d (%.2f MB): ctx %s, %d iterations: SDMA vs CU %d words differ, vs the fold %d, cross-stream %d%s" % (
            d["rows"], d["width"], d["mb"], d["ctx"], d["iters"], d["bad_vs_cu"], d["bad_vs_fold"], d["bad_xstream"], " FAILED" if d["failed"] else "")
        for k in ("ar", "rs", "ag"):
            o = d[k]
            s += " | %s alone CU %.1f / SDMA %.1f us, beside CU %.1f / SDMA %.1f us, stream loss CU %.1f / SDMA %.1f us" % (
                k, o["cu_us"], o["sdma_us"], o["cu_with_us"], o["sdma_with_us"], o["cu_loss_us"], o["sdma_loss_us"])
        print(s)
PY
fi
log "RESULT fn4 sdma2: done ($O/sdma2-*.txt)"
