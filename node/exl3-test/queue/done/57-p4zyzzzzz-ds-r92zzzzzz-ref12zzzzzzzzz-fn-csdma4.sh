#!/bin/bash
# 219-fn-sdma4.sh (queue item, 12.11.2026, SDMA-AR agent, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~25 min)
# -- THE DECODE QUESTION + THE HALF-RATE FIX CANDIDATE. 216: c16 decode at ar_sdma 0 945.3 / 923.8 tok/s against 213's S7
# 1002.0 / 981.0 (c16-x3sp16) and 211's 994 / 988: is it the SDMA declaration (context, engine-1 queue, ~84 MB a card), the
# switch set (216's base had hc_sp 2 + ar_xpush 1; xpush engages on c16 verify passes: 64 rows x 2560 >= 131072), the image,
# or the cards' heat (216 started at 54-57 C after 214 / 215; 211 / 213 at 39-42 C)?
#   D1 c16-x3sdma3, RADIANCE_FN4_AR_SDMA=0 (declared): c16 A B B A  S7 vs S7 + hc_sp 2 + ar_xpush 1 (+ ar_sdma 0)
#   D2 c16-x3sdma3, not declared:                       c16 A B B A  S7 vs S7 (A/A)
#   D3 c16-x3sp16 (the same tree without SDMA-AR):      c16 A B B A  S7 vs S7 (A/A)
#   D4 c16-x3sdma3, declared again (brackets the heat): c16 A B B A  S7 vs S7 (A/A)
#   every start logs the cards' temperatures; S7 = 213's set with hc_sp / ar_xpush / mtp_flat 0 written out.
#   R  r4d_selftest (c16-x3sdma3): R4D_SDMA_RR=1 MiB round-robin hops vs off, 6 alternated runs each at 21 / 10.5 MB (24 timing
#      windows an arm), 256 KiB 3 runs; --sdmaeng4 at RR 1 MiB (bytes + alone / beside).
# Disk: not started under 12 GB free on /; logs gzipped.
# DRYRUN=1 bash 219-fn-sdma4.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4zs.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/heatsoak.sh" ] && { TL=$f; break; }; done
IMG=${XSD4_IMAGE:-radiance-local:c16-x3sdma3}; IMG0=${XSD4_IMAGE0:-radiance-local:c16-x3sp16}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-sd4
export FNIMAGE=$IMG FNMODELDIR=${XSD4_X3_DIR:-$HOME/models/flashnext-exl3} FNMODEL=${XSD4_X3_MODEL:-qwen3.8-flash-next-exl3.rad}
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
EXTRA0="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0"
export EXTRA="$EXTRA0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats-sd4
export FN4BT="" FN4BTP="" FN4BTL="" FN4X3DUAL=1 FN4NR=1 FN4GI=1 FN4FD="" FN4NW=1 FN4SW="" FN4WV=1 FN4RV=11 FN4AL="" FN4UM=5 FN4SS=1 FN4CH="" FN4XM=1 FN4SPHC=1 FN4X3C=1 FN4QSA="" FN4P2X="" FN4MP="" FN4X3GD=1 FN4X3DZ=1 PROFEVERY="" FN4MTPF=1
export FN4SDMA=0 SDMAENG=1
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
compose() { docker compose --project-directory $C -f $Y "$@"; }
down() { run "compose down > /dev/null 2>&1; docker rm -f $CN > /dev/null 2>&1"; }
[ -n "$TL" ] && . $TL/heatsoak.sh
caps() { local d h out=""; for d in /sys/class/drm/card*/device; do [ -e $d/unique_id ] || continue; h=$(ls -d $d/hwmon/hwmon* 2>/dev/null | head -1); [ -n "$h" ] || continue
  out="$out $(basename $(readlink -f $d)):cap=$(( $(cat $h/power1_cap 2>/dev/null || echo 0) / 1000000 ))W,hot=$(( $(cat $h/temp2_input 2>/dev/null || echo 0) / 1000 ))C"; done; echo "$out"; }
st() {   # st NAME TIMEOUT ENV... -- ARGS : r4d_selftest on the four cards -> $O/sdma4-NAME.txt
  local name=$1 to=$2; shift 2; local envs=()
  while [ "$1" != "--" ]; do envs+=(-e "$1"); shift; done; shift
  local f=$O/sdma4-$name.txt
  if [ "$DRY" = 1 ]; then echo "+ timeout -k 15 $to docker run --rm --init --name fn4-sdma4-$name --network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$CARDS ${envs[*]} --entrypoint /opt/radiance/bin/r4d_selftest $IMG /opt/radiance/share/radiance/kernels/libr4d.so $* > $f"; return 0; fi
  timeout -k 15 $to docker run --rm --init --name fn4-sdma4-$name-$$ --network none --device /dev/kfd --device /dev/dri \
    --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$CARDS "${envs[@]}" --entrypoint /opt/radiance/bin/r4d_selftest $IMG \
    /opt/radiance/share/radiance/kernels/libr4d.so "$@" > $f 2>&1
  local rc=$?; docker rm -f fn4-sdma4-$name-$$ > /dev/null 2>&1
  [ $rc = 0 ] || log "RESULT fn4 sdma4 $name: exit $rc ($f)"
  return 0
}
tguard() { local d h j m n=0 hot
  while [ ! -e $O/$TAG.tstop ]; do hot=""
    for d in /sys/class/drm/card*/device; do [ -e $d/unique_id ] || continue; h=$(ls -d $d/hwmon/hwmon* 2>/dev/null | head -1); [ -n "$h" ] || continue
      j=$(( $(cat $h/temp2_input 2>/dev/null || echo 0) / 1000 )); m=$(( $(cat $h/temp3_input 2>/dev/null || echo 0) / 1000 ))
      { [ $j -ge 100 ] || [ $m -ge 96 ]; } && hot="$hot $(basename $(readlink -f $d)):j$j/m$m"; done
    if [ -n "$hot" ]; then n=$((n + 1)); else n=0; fi
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 sdma4: VIRHE temperature guard ($hot) -> blocks stopped"; return 0; }
    sleep 1; done; }
restart() {   # restart TAG -> 0 healthy
  local tag=$1 ok=0 t0 i S=$O/start-$1.log
  [ "$DRY" = 1 ] && { echo "+ FNIMAGE=$FNIMAGE FN4NR=$FN4NR FN4GI=$FN4GI FN4NW=$FN4NW FN4WV=$FN4WV FN4RV=$FN4RV FN4UM=$FN4UM FN4CH=$FN4CH EXTRA='$EXTRA' compose -f $Y up -d; health <= 1200 s; docker logs > $S"; return 0; }
  down; t0=$(date +%s); compose up -d > $O/up-$tag.txt 2>&1
  for i in $(seq 1 240); do
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' $URL/health)" = 200 ] && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Status}}' $CN 2>/dev/null)" != running ] && break
    sleep 5
  done
  docker logs $CN > $S 2>&1
  [ $ok = 1 ] || { log "RESULT fn4 sdma4 restart-$tag: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
  log "RESULT fn4 sdma4 restart-$tag: healthy in $(( $(date +%s) - t0 )) s; $(grep -m1 -E '^I kv: ' $S | cut -c3-160); fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300)"
}
probe8() {   # probe8 TAG -> "ms tok/s" of a 45 s mb c8
  [ "$DRY" = 1 ] && { echo "+ (cd $D && timeout 400 python3 mb.py --tag $1 --phases c8 --dur 45 --warm 10 > $O/$1-c8.txt)" >&2; echo "25.6 744"; return 0; }
  (cd $D && timeout 400 python3 mb.py --tag $1 --phases c8 --dur 45 --warm 10 > $O/$1-c8.txt 2>&1)
  grep '^MB ' $O/$1-c8.txt 2>/dev/null | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[3:]); print(d.get("ms_decode_step"), d.get("agg_dec_tps"))' 2>/dev/null || echo "0 0"
}
setsw() {   # setsw JSON: queue switches (they land at the next step release)
  run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"
}
# 12.11. 16:3x: re-run of the T byte check only; 198 called nudge/swcheck without defining them (T skipped in 0 s)
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
swcheck() {   # swcheck WHAT SWITCHES_JSON: every switch named there has that value now
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4: $2"; return 0; }
  local g; g=$(curl -s -m 10 $URL/admin/fn4)
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); w=json.loads(sys.argv[2]); sys.exit(0 if all(d["switches"].get(k) == v for k, v in w.items()) else 1)' "$g" "$2" 2>/dev/null \
    || { log "RESULT fn4 sdma4 $1: VIRHE the switches are not $2 ($(echo "$g" | cut -c1-400))"; return 1; }
}
alive() {   # alive STEP: /health 200, else log it (the T arm then yields no answers)
  [ "$DRY" = 1 ] && { echo "+ curl -m 5 $URL/health == 200"; return 0; }
  local code; code=$(curl -s -m 5 -o /dev/null -w %{http_code} $URL/health)
  [ "$code" = 200 ] && return 0
  log "RESULT fn4 sdma4 $1: VIRHE the server is not healthy (/health $code; $(docker inspect -f {{.State.Status}} exit {{.State.ExitCode}} $CN 2>/dev/null || echo gone))"; return 1
}
GS=; for f in "$H/tools/greedyset.py" "$H/../tools/greedyset.py" "$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py"; do [ -z "$GS" ] && [ -e "$f" ] && GS=$f; done
LD=$TL/fn-loadid.py; MBT=$TL/fnharness/mbt2.py; GL=$TL/greedylong.py; BTP=$TL/fn-bitrace.py
# 213's S7 (hc_readv 75 = G16), its collective switches written out; SB0 = 216's base (+ hc_sp 2 + ar_xpush 1), ar_sdma 0
S7='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 4, "hc_readv": 75, "hc_sp": 0, "mtp_flat": 0, "ar_xpush": 0}'
SB0='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 4, "hc_readv": 75, "hc_sp": 2, "mtp_flat": 0, "ar_xpush": 1, "ar_sdma": 0}'
pcr() { run "curl -s -m 60 -o /dev/null -X POST -d '' $URL/reset_prefix_cache"; }
sw() { setsw "$(echo "$1" | sed 's/}$/, "epoch_bump": 1}/')"; nudge; swcheck "$2" "$1"; }
glr() { pcr; run "(cd $D && GL_N=4 timeout 1800 python3 $GL $URL $O/$TAG-$1.jsonl --conc 1 --max-tokens 64 > $O/$TAG-$1.txt 2>&1)"; }
gsr() { pcr; run "(cd $D && timeout 900 python3 $GS $URL $O/$TAG-$1.jsonl --conc 1 --max-tokens 192 > $O/$TAG-$1.txt 2>&1)"; }
lc() { [ "$DRY" = 1 ] && { echo "+ cmp $1 $2"; return 0; }; log "RESULT fn4 sdma4 text $3: $(timeout 300 python3 -I $LD cmp --url $URL --label "$3" $O/$TAG-$1.jsonl $O/$TAG-$2.jsonl 2>&1 | grep -m1 '^CMPLINE ' | cut -c9- | cut -c1-400)"; }
abr() {   # abr NAME A B (extra abblocks2 args...)
  local nm=$1 A=$2 B=$3; shift 3
  run "timeout 2400 python3 -I $TL/abblocks2.py --url $URL --mb-dir $D --label $TAG-$nm --rounds 1 --a '$A' --b '$B' --names a,b --out $O/$TAG-blocks.jsonl $* > $O/$TAG-$nm.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 sdma4 $nm (A B B A): $(grep '^ABB ' $O/$TAG-$nm.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=4: "-" if v is None else format(v, ".%df" % n)
s = "decode tok/s B/A %s %s ms/step B/A %s (A %s | B %s)" % (f(d["tps_ratio"]), (d.get("sep") or {}).get("tps"), f(d.get("ms_ratio")), " ".join("%.1f" % x for x in d.get("A_blocks", [])), " ".join("%.1f" % x for x in d.get("B_blocks", []))) if d.get("rounds") else ""
for k, v in (d.get("pf") or {}).items():
    s += " | prefill %s B/A %s %s (A %s | B %s)" % (k, f(v["ratio"]), v.get("sep"), " ".join("%.0f" % x for x in v.get("A_blocks", [])), " ".join("%.0f" % x for x in v.get("B_blocks", [])))
print(s + " | %d blocks, %d flagged, %s s" % (d["blocks"], d["blocks_bad"], d["wall_s"]))' 2>&1 | cut -c1-700)"
  [ "$DRY" = 1 ] || log "RESULT fn4 sdma4 $nm blocks (tok/s, ms/step, acceptance): $(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("label") == sys.argv[2]: print("%s %s tok/s %s ms/step acc %s;" % (r["arm"], r.get("tps"), r.get("ms"), (r.get("mb") or {}).get("acceptance")), end=" ")' $O/$TAG-blocks.jsonl $TAG-$nm 2>&1 | cut -c1-300)"
}
log "RESULT fn4 sdma4-fn-sdma: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) IMAGE0=$IMG0 ($(docker image inspect -f '{{.Id}}' $IMG0 2>/dev/null | cut -c8-19)); D1-D4 c16 decode, then R (selftest) | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 sdma4: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 && docker image inspect $IMG0 > /dev/null 2>&1 || { log "RESULT fn4 sdma4: VIRHE image $IMG / $IMG0 missing"; exit 1; }
  [ -e "$Y" ] && [ -n "$TL" ] && [ -e $TL/abblocks2.py ] && [ -e $MBT ] && [ -e $LD ] && [ -e $GL ] && [ -e $BTP ] && [ -n "$GS" ] && [ -e $D/mb.py ] || { log "RESULT fn4 sdma4: VIRHE tools missing"; exit 1; }
  grep -q 'RADIANCE_FN4_MTPFLAT' $Y && grep -q 'RADIANCE_FN4_AR_SDMA' $Y || { log "RESULT fn4 sdma4: VIRHE $YN does not pass RADIANCE_FN4_MTPFLAT / _AR_SDMA"; exit 1; }
  mkdir -p $O $STATSDIR/bt-sp && chmod 777 $STATSDIR/bt-sp && find $STATSDIR/bt-sp -maxdepth 1 -name 'bt-r*' -delete
  find $O -maxdepth 1 -name "$TAG.abort" -delete; find $O -maxdepth 1 -name "$TAG.tstop" -delete
  find $O -maxdepth 1 \( -name "$TAG-*.jsonl" -o -name "$TAG-*.txt" \) -delete
  trap 'touch $O/$TAG.tstop 2>/dev/null; down' EXIT
  trap 'exit 143' TERM INT
  tguard & TGPID=$!
else
  echo "+ guards; tguard &"; TGPID=""
fi
export EXTRA="$EXTRA0 --tp-wire exact"
FREEG=$(df --output=avail -BG / 2>/dev/null | tail -1 | tr -dc 0-9)
[ "$DRY" != 1 ] && [ "${FREEG:-0}" -lt 12 ] && { log "RESULT fn4 sdma4: VIRHE only ${FREEG:-?} GB free on / -- not started"; exit 1; }
export FN4BT="" FN4BTP="" FN4BTL="" EXTRA="$EXTRA0 --tp-wire exact"
dstart() {   # dstart NAME IMAGE FN4SDMA : one start for a decode phase
  export FNIMAGE=$2 FN4SDMA=$3
  restart $TAG-$1 || return 1
  nudge
  [ "$DRY" = 1 ] || log "RESULT fn4 sdma4 $1 start: IMAGE=$2 AR_SDMA='$3'; $(grep -E 'fn4 SDMA-AR' $O/start-$TAG-$1.log | cut -c1-160 | tr '\n' ' ')| cards: $(caps)"
  return 0
}
dend() {   # dend NAME : the engine log kept (gzipped), the cards' temperatures, down
  [ "$DRY" = 1 ] || log "RESULT fn4 sdma4 $1 end: cards $(caps)"
  run "docker logs $CN > $O/$TAG-$1-engine.log 2>&1"; down
  run "gzip -f $O/$TAG-$1-engine.log $O/start-$TAG-$1.log 2>/dev/null; true"
}
if dstart D1 "$IMG" 0; then
  sw "$S7" "S7"
  abr d1 "$S7" "$SB0" --mbt2 $MBT --gens 1 --phase c16 --warm 8; alive "D1 c16 S7 vs S7 + hc_sp 2 + xpush"
  dend D1
fi
if dstart D2 "$IMG" ""; then
  sw "$S7" "S7"
  abr d2 "$S7" "$S7" --mbt2 $MBT --gens 1 --phase c16 --warm 8; alive "D2 c16 S7 A/A undeclared"
  dend D2
fi
if dstart D3 "$IMG0" ""; then
  sw "$S7" "S7"
  abr d3 "$S7" "$S7" --mbt2 $MBT --gens 1 --phase c16 --warm 8; alive "D3 c16 S7 A/A c16-x3sp16"
  dend D3
fi
if dstart D4 "$IMG" 0; then
  sw "$S7" "S7"
  abr d4 "$S7" "$S7" --mbt2 $MBT --gens 1 --phase c16 --warm 8; alive "D4 c16 S7 A/A declared"
  dend D4
fi
# ---- R: the round-robin hop option (selftest, every container down)
[ "$DRY" = 1 ] || find $O -maxdepth 1 -name 'sdma4-*.txt' -delete
for k in 1 2 3 4 5 6; do
  st off$k 240 R4D_SDMA_ENGINE=1 R4D_SDMA_CHECKS=50 R4D_SDMA_TL=1 -- --sdmaar4 4096x2560,2048x2560 100
  st rr1m$k 240 R4D_SDMA_ENGINE=1 R4D_SDMA_CHECKS=50 R4D_SDMA_TL=1 R4D_SDMA_RR=1048576 -- --sdmaar4 4096x2560,2048x2560 100
done
for k in 1 2 3; do
  st rr256k$k 240 R4D_SDMA_ENGINE=1 R4D_SDMA_CHECKS=50 R4D_SDMA_TL=1 R4D_SDMA_RR=262144 -- --sdmaar4 4096x2560,2048x2560 100
done
st eng 600 R4D_SDMA_ENGINE=1 R4D_SDMA_CHECKS=100 R4D_SDMA_RR=1048576 -- --sdmaeng4 2048x2560,4096x2560 50
if [ "$DRY" != 1 ]; then
  python3 - $O <<'PY' | while IFS= read -r l; do log "RESULT fn4 sdma4 $l"; done
import glob, json, os, re, sys
O = sys.argv[1]
arms = {}
bad = {}
for f in sorted(glob.glob(os.path.join(O, "sdma4-*.txt"))):
    m = re.match(r"sdma4-(off|rr1m|rr256k)(\d+)\.txt$", os.path.basename(f))
    if not m:
        continue
    arm = m.group(1)
    cur = None
    for l in open(f, errors="replace"):
        mm = re.match(r"\s+(\d+) x (\d+) bf16", l)
        if mm:
            cur = "%sx%s" % (mm.group(1), mm.group(2))
        if l.startswith("SDMAAR {"):
            d = json.loads(l[7:])
            bad[arm] = bad.get(arm, 0) + d["bad_oop"] + d["bad_inplace"] + d["bad_vs_cu"] + (1 if d["failed"] else 0)
            arms.setdefault((arm, cur), {}).setdefault("cu", []).append(d["us_cu"])
        if "first / second timing by rank" in l and cur:
            pairs = [p.split("/") for p in l.split(":")[1].split()]
            w1 = max(float(a) for a, b in pairs); w2 = max(float(b) for a, b in pairs)
            arms.setdefault((arm, cur), {}).setdefault("win", []).extend([w1, w2])
for (arm, size), d in sorted(arms.items()):
    w = d.get("win", []); cu = d.get("cu", [])
    if not w:
        continue
    lo = min(w); cum = sorted(cu)[len(cu) // 2] if cu else 0
    slow = [x for x in w if cum > 0 and x > 1.25 * cum]
    print("%s %s: %d windows, SDMA median %.1f us (min %.1f, max %.1f), slow (>1.25x the CU median) %d; CU median %.1f us; windows %s" % (
        arm, size, len(w), sorted(w)[len(w) // 2], lo, max(w), len(slow), cum, " ".join("%.0f" % x for x in w)))
print("bytes: bad / failed counts by arm %s" % json.dumps(bad))
PY
  f=$O/sdma4-eng.txt
  python3 - $f <<'PY' | while IFS= read -r l; do log "RESULT fn4 sdma4 eng rr1m $l"; done
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
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
log "RESULT fn4 sdma4-fn-sdma: done"
