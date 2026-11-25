#!/bin/bash
# 204-fn-rsf.sh (queue item, 12.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; <= 10 min) -- RSF: WHY
# THE EXACT REDUCE-SCATTER IS THE SLOW HALF (202: RS raw 1.5-1.7x AG for the same bytes; the exact two-shot ~ RS + AG) and
# the FENCE COST + STALE-READ STRESS (the R9700 audit item 3), plus hc_bi 5's one-launch read by row count. r4d_selftest from
# radiance-local:c16-x3sp6 (fn4-x3-sp3 808866c2: the interleaved first hop R4D_AR8_FORM_XPUSH / RS raw xpush behind
# ar_xpush, drain_force, --perfar4 v2). (1) --perfar4 at 4096x5120 / 2048x2560 / 4096x2560 / 16 / 48 x 2560, 500 launches:
# exact two-shot lockstep vs xpush at drain 1 (the rule) / 2 / 3 / 0, the coded lane form, the true one-shot, RS raw
# lockstep vs xpush at each drain, AG raw, the ti8 pair; --timeline (the two-shot's per-block push / wait / reduce / tail).
# (2) the stress: R4D_SELFTEST_AR4_STRESS=1000 at 4096x5120 and 2048x2560 -- every launch of fresh instances checked against
# the fp32 fold (the slots' first reuse included), per op x xpush x drain. (3) --perfhcv M at 4 / 8 / 16 / 32 / 64 rows,
# R4D_SELFTEST_BI_DEC 0 vs 1 (one card): hc_bi 5's form G + B3 row groups against today's forms (200: hc_bi 5 still 8 %).
# DRYRUN=1 bash 204-fn-rsf.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log
IMG=${X204_IMAGE:-radiance-local:c16-x3sp6}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
CARDS=$C3,$C83,$C86,$C03
SIZES=${X204_SIZES:-4096x5120,2048x2560,4096x2560,16x2560,48x2560}
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
f=$O/204-rsf.txt; f2=$O/204-rsf-stress.txt
log "RESULT fn4 204-fn-rsf: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) r4d_selftest --perfar4 $SIZES 500 (+ stress 1000 at 4096x5120 / 2048x2560), cards $CARDS"
if [ "$DRY" = 1 ]; then
  echo "+ timeout -k 15 300 docker run --rm ... -e ROCR_VISIBLE_DEVICES=$CARDS r4d_selftest --timeline --perfar4 $SIZES 500 > $f; then -e R4D_SELFTEST_AR4_STRESS=1000 --perfar4 4096x5120,2048x2560 20 > $f2"
  for m in 4 8 16 32 64; do echo "+ docker run ... -e R4D_SELFTEST_BI_DEC=0/1 r4d_selftest --perfhcv $m 500"; done
  log "RESULT fn4 204-fn-rsf: done"; exit 0
fi
docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 204: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 204: VIRHE image $IMG missing"; exit 1; }
mkdir -p $O
timeout -k 15 420 docker run --rm --init --name fn4-204-ar-$$ --network none --device /dev/kfd --device /dev/dri \
  --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$CARDS --entrypoint /opt/radiance/bin/r4d_selftest $IMG \
  /opt/radiance/share/radiance/kernels/libr4d.so --timeline --perfar4 $SIZES 500 > $f 2>&1
rc=$?; docker rm -f fn4-204-ar-$$ > /dev/null 2>&1
log "RESULT fn4 204 r4d_selftest --perfar4: exit $rc, $(grep -c '^AR4 ' $f) points; $(grep -m1 '^four-rank' $f | cut -c1-200)"
grep -E 'not served|absent|init rc|needs 4' $f | head -6 | while IFS= read -r l; do log "RESULT fn4 204 note: $(echo "$l" | sed 's/^ *//' | cut -c1-200)"; done
grep -E '^ +timeline r0' $f | head -4 | while IFS= read -r l; do log "RESULT fn4 204 two-shot 4096x5120 $(echo "$l" | sed 's/^ *//' | cut -c1-300)"; done
timeout -k 15 420 docker run --rm --init --name fn4-204-st-$$ --network none --device /dev/kfd --device /dev/dri \
  --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$CARDS -e R4D_SELFTEST_AR4_STRESS=1000 --entrypoint /opt/radiance/bin/r4d_selftest $IMG \
  /opt/radiance/share/radiance/kernels/libr4d.so --perfar4 4096x5120,2048x2560 20 > $f2 2>&1
rc2=$?; docker rm -f fn4-204-st-$$ > /dev/null 2>&1
log "RESULT fn4 204 stress: exit $rc2, $(grep -c '^STRESS ' $f2) points"
grep '^STRESS ' $f2 | python3 -c 'import json,sys
rows = [json.loads(l[7:]) for l in sys.stdin]
for (r, w) in sorted({(d["rows"], d["width"]) for d in rows}, reverse=True):
    print("%dx%d: %s" % (r, w, "; ".join("%s%s drain %d: %d launches, rc %d, bad %d (%d launches, first %d)" % (d["op"], " xpush" if d["xpush"] else "", d["drain"], d["launches"], d["rc"], d["bad"], d["bad_launches"], d["first_bad"]) for d in rows if (d["rows"], d["width"]) == (r, w))))' 2>&1 | while IFS= read -r l; do log "RESULT fn4 204 stress $l"; done
python3 - $f <<'PY' | while IFS= read -r l; do log "RESULT fn4 204 $l"; done
import json, sys
rows = {}
for l in open(sys.argv[1], errors="replace"):
    if l.startswith("AR4 "):
        d = json.loads(l[4:]); rows.setdefault((d["rows"], d["width"], d["mb"]), []).append(d)
for (r, w, mb), v in rows.items():
    print("%dx%d (%.2f MB): %s" % (r, w, mb, "; ".join("%s %.1f us %.2f GB/s%s" % (d["op"], d["us"], d["gbs_dir"],
          " (%d differ)" % d["bad"] if d["bad"] and "AR" in d["op"] and "ti8" not in d["op"] else "") for d in v)))
PY
hcv() {   # hcv M BI_DEC
  local g=$O/204-hcv-m$1-bd$2.txt
  timeout -k 15 240 docker run --rm --init --name fn4-204-hcv-$$ --network none --device /dev/kfd --device /dev/dri \
    --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$C3 -e R4D_SELFTEST_BI_DEC=$2 --entrypoint /opt/radiance/bin/r4d_selftest $IMG \
    /opt/radiance/share/radiance/kernels/libr4d.so --perfhcv $1 500 > $g 2>&1
  docker rm -f fn4-204-hcv-$$ > /dev/null 2>&1
  log "RESULT fn4 204 hc_read M $1 bi_dec $2: $(grep -E 'hc_read_e4m3 M=' $g | sed 's/^ *//' | cut -c1-300)"
}
for m in 4 8 16 32 64; do hcv $m 0; hcv $m 1; done
log "RESULT fn4 204-fn-rsf: done ($f)"
