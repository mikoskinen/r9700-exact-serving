#!/bin/bash
# 202-fn-arbw.sh (queue item, 12.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; <= 6 min) -- THE
# FOUR-RANK COLLECTIVES' BANDWIDTH AFTER THE RISER MOVE + SP-hc's QUARTER-READ COST (coordinator 12.11.: every AR number we have predates it; LEVERS PA /
# MAR: DS attention AR 3208 us, floor 2927, 42 MB). r4d_selftest --perfar4 from radiance-local:c16-x3sp4 (fn4-x3-sp3
# 15f6a6be) on the four cards (TP4 order): the exact one-shot and two-shot, the ti8 coded two-shot (lossy: this tree has
# no coded EXACT four-rank form), SP's reduce_scatter_rows / all_gather_rows raw 1 (exact) and raw 0 (ti8), at
# 4096x5120 (DS prefill, 42 MB), 2048x2560 / 4096x2560 (Qwen prefill piece / step), 16 / 48 rows x 2560 and x 5120 (decode).
# us a call (slowest rank, warm) and GB/s a rank a direction on the bytes sent (two-shot: 2 (N-1)/N S, both phases;
# one-shot (N-1) S; RS or AG alone (N-1)/N S); compare: bw4's P2P pair rate 25.5-27.6 GB/s (gpu02-results-1031.md).
# Then (one card, the first TP4 card GPU-dbba5c85) r4d_selftest --perfhcv: hc_read_e4m3 by PF2 stage at 2048 rows, at 512 rows with the K split pinned
# to 2048 (R4D_SELFTEST_KC_ROWS: SP-hc's quarter read) and at 512 unpinned (201: the quarter costs 2.5x the piece per row).
# DRYRUN=1 bash 202-fn-arbw.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log
IMG=${X202_IMAGE:-radiance-local:c16-x3sp4}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
CARDS=$C3,$C83,$C86,$C03
SIZES=${X202_SIZES:-4096x5120,2048x2560,4096x2560,16x2560,48x2560,16x5120,48x5120}
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
f=$O/202-arbw.txt
log "RESULT fn4 202-fn-arbw: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) r4d_selftest --perfar4 $SIZES 100, cards $CARDS"
if [ "$DRY" = 1 ]; then
  echo "+ timeout -k 15 300 docker run --rm ... -e ROCR_VISIBLE_DEVICES=$CARDS --entrypoint /opt/radiance/bin/r4d_selftest $IMG /opt/radiance/share/radiance/kernels/libr4d.so --perfar4 $SIZES 100 > $f"
  for x in "piece2048 2048 0" "quarter512-pinned 512 2048" "quarter512-own 512 0"; do echo "+ docker run ... -e R4D_SELFTEST_KC_ROWS=${x##* } r4d_selftest --perfhcv $x"; done
  log "RESULT fn4 202-fn-arbw: done"; exit 0
fi
docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 202: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 202: VIRHE image $IMG missing"; exit 1; }
mkdir -p $O
timeout -k 15 300 docker run --rm --init --name fn4-202-ar-$$ --network none --device /dev/kfd --device /dev/dri \
  --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$CARDS --entrypoint /opt/radiance/bin/r4d_selftest $IMG \
  /opt/radiance/share/radiance/kernels/libr4d.so --perfar4 $SIZES 100 > $f 2>&1
rc=$?; docker rm -f fn4-202-ar-$$ > /dev/null 2>&1
log "RESULT fn4 202 r4d_selftest --perfar4: exit $rc, $(grep -c '^AR4 ' $f) points; $(grep -m1 '^four-rank' $f | cut -c1-200)"
grep -E 'not served|absent|init rc|needs 4' $f | head -6 | while IFS= read -r l; do log "RESULT fn4 202 note: $(echo "$l" | sed 's/^ *//' | cut -c1-200)"; done
python3 - $f <<'PY' | while IFS= read -r l; do log "RESULT fn4 202 $l"; done
import json, sys
rows = {}
for l in open(sys.argv[1], errors="replace"):
    if l.startswith("AR4 "):
        d = json.loads(l[4:]); rows.setdefault((d["rows"], d["width"], d["mb"]), []).append(d)
for (r, w, mb), v in rows.items():
    print("%dx%d (%.2f MB): %s" % (r, w, mb, "; ".join("%s %.1f us %.2f GB/s%s" % (d["op"], d["us"], d["gbs_dir"],
          " (%d differ)" % d["bad"] if d["bad"] and "AR" in d["op"] and "ti8" not in d["op"] else "") for d in v)))
PY
hcv() {   # hcv LABEL M KC_ROWS
  local g=$O/202-hcv-$1.txt
  timeout -k 15 240 docker run --rm --init --name fn4-202-hcv-$$ --network none --device /dev/kfd --device /dev/dri \
    --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$C3 -e R4D_SELFTEST_KC_ROWS=$3 --entrypoint /opt/radiance/bin/r4d_selftest $IMG \
    /opt/radiance/share/radiance/kernels/libr4d.so --perfhcv $2 200 > $g 2>&1
  docker rm -f fn4-202-hcv-$$ > /dev/null 2>&1
  log "RESULT fn4 202 hc_read $1 (M $2, kc_rows $3): $(grep -E 'hc_read_e4m3 M=|PF2 stages' $g | sed 's/^ *//' | tr '\n' ' ' | cut -c1-900)"
}
hcv piece2048 2048 0
hcv quarter512-pinned 512 2048
hcv quarter512-own 512 0
log "RESULT fn4 202-fn-arbw: done ($f)"
