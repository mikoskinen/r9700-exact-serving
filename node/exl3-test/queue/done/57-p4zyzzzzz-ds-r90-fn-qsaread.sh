#!/bin/bash
# 163-fn-qsaread.sh (queue item, 10.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~12 min) -- P-2
# READING (coordinator 10.11.: size the q-tiled union walk by measurement). The prefill attention (attn_paged_gate_quant on
# the QSA sparse path: q_len-1 rows a (token, head), each row's own table of 512 selected 4-token blocks) is 1.61 s of rank
# 2's 128k lane (151) at ~21 TFLOPS: bound by the per-row KV table reads. A q-tiled form would walk the UNION of R
# consecutive rows' tables once (each row its own blocks in its own ascending order: exact). Its gain is the reduction
# factor R x (blocks a row) / (union of R rows' blocks), measured here: the server (radiance-local:c16-q, fn4-c16
# 7302cad9 = c16-sp + RADIANCE_FN4_QSADUMP) dumps, on rank 0, the selection of full-attention layers 3 / 15 / 27 / 39 for
# the pipelined prefill steps starting near 8k / 32k / 64k / 112k of one fresh 128k prompt; a script computes the union
# sizes for R = 2 / 4 / 8 / 16. iq4r .rad, stack 3's switches, compose fn-tp4z.yaml (FN4QSA=/stats/qsa). Reading only
# (the dump synchronises its steps). The kernel micro-benchmark follows only if the factor is large enough to matter.
# DRYRUN=1 bash 163-fn-qsaread.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41
YN=fn-tp4z.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
IMG=${X163_IMAGE:-radiance-local:c16-q}
TAG=fn4-qsa
export FNIMAGE=$IMG FNMODELDIR=$HOME/models/flashnext-iq4r FNMODEL=qwen3.8-next-flash-fp8-iq4r-moe.rad
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 NST=3 WIREKB=128 PMIN="" PMINB=""
export AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats163 FN4FD="" FN4SW="" FN4CH="" FN4XM="" FN4AL="" FN4SPHC="" FN4X3C="" PROFEVERY=""
export FN4NW=1 FN4GI=1 FN4NR=1 FN4WV=1 FN4RV=11 FN4SS=1 FN4UM=5 MNBT=4096 FN4QSA=/stats/qsa
export EXTRA="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0"
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
compose() { docker compose --project-directory $C -f $Y "$@"; }
down() { run "compose down > /dev/null 2>&1; docker rm -f $CN > /dev/null 2>&1"; }
log "RESULT fn4 163-fn-qsaread: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) QSA dump -> $STATSDIR/qsa"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 163: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 163: VIRHE image $IMG missing"; exit 1; }
  [ -e "$Y" ] && grep -q RADIANCE_FN4_QSADUMP $Y || { log "RESULT fn4 163: VIRHE $YN (with RADIANCE_FN4_QSADUMP) missing"; exit 1; }
  rm -rf $STATSDIR; mkdir -p $STATSDIR/qsa; chmod 777 $STATSDIR $STATSDIR/qsa
  trap 'down' EXIT
fi
if [ "$DRY" = 1 ]; then echo "+ FNIMAGE=$FNIMAGE FN4QSA=$FN4QSA compose -f $Y up -d; health <= 1200 s"
else
  down; t0=$(date +%s); compose up -d > $O/up-$TAG.txt 2>&1; ok=0
  for i in $(seq 1 240); do
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' $URL/health)" = 200 ] && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Status}}' $CN 2>/dev/null)" != running ] && break
    sleep 5
  done
  docker logs $CN > $O/start-$TAG.log 2>&1
  [ $ok = 1 ] || { log "RESULT fn4 163: VIRHE not healthy after $(( $(date +%s) - t0 )) s"; exit 1; }
fi
run "(cd $D && timeout 600 python3 ds41_tests.py correct --only-q --effort none --outdir $O/guard > $O/guard/$TAG-warm-q.txt 2>&1)"
run "(cd $D && timeout 1500 python3 ds41_tests.py prefill --depths 131072 --cpt 3.6 --calibrate 0 --reps 1 > $O/$TAG-pf128.txt 2>&1)"
run "docker logs $CN > $O/$TAG-engine.log 2>&1"
down
if [ "$DRY" != 1 ]; then
  log "RESULT fn4 163 QSA union sizes (R consecutive rows; factor = R x blocks a row / union = the KV table reads a q-tiled walk saves): $(python3 - $STATSDIR/qsa <<'PY' 2>&1 | cut -c1-1500
import array, glob, os, struct, sys, statistics as st
fs = sorted(glob.glob(os.path.join(sys.argv[1], "qsa-L*-p*.bin")))
if not fs: print("no dumps"); sys.exit()
out = []; allf = {r: [] for r in (2, 4, 8, 16)}
for f in fs:
    b = open(f, "rb").read()
    T, w, p0 = struct.unpack("<qqq", b[:24])
    ns = array.array("i"); ns.frombytes(b[24:24 + 4 * T])
    sel = array.array("i"); sel.frombytes(b[24 + 4 * T:24 + 4 * T + 4 * T * w])
    rows = [set(sel[i * w:i * w + max(0, min(ns[i], w))]) for i in range(T)]
    per = st.mean(len(r) for r in rows)
    fac = {}
    for R in (2, 4, 8, 16):
        v = []
        for g in range(0, T - R + 1, R):
            u = set().union(*rows[g:g + R]); n = sum(len(r) for r in rows[g:g + R])
            if u: v.append(n / len(u))
        fac[R] = st.mean(v) if v else 0; allf[R] += v
    out.append("%s pos %d: %d rows, %.0f blocks a row, factor R2 %.2f R4 %.2f R8 %.2f R16 %.2f" % (os.path.basename(f)[4:7], p0, T, per, fac[2], fac[4], fac[8], fac[16]))
print("; ".join(out[:16]) + " | ALL: " + " ".join("R%d %.2f" % (R, st.mean(v)) for R, v in allf.items() if v))
PY
)"
  log "RESULT fn4 163 the 128k prompt: $(grep -h '^J ' $O/$TAG-pf128.txt | cut -c1-200) (dump steps synchronise: not a speed)"
fi
log "RESULT fn4 163-fn-qsaread: done"
