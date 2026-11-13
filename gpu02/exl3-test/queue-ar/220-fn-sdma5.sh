#!/bin/bash
# 220-fn-sdma5.sh (queue item, 12.11.2026, SDMA-AR agent, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~32 min)
# -- WHICH PART OF A DECLARED SDMA-AR CONTEXT COSTS DECODE. 219: c16 S7 at ar_sdma 0 declared (D1 / D4) 922-953 tok/s vs
# undeclared (D2) 979-1004 and c16-x3sp16 (D3) 955-996 -- the declaration costs ~5 %. Image radiance-local:c16-x3sdma5
# (fn4-x3-sp3-sdma 5fae0bf2: knobs R4D_SDMA_QLAZY / _SCR / _QPRIO, R4D_SDMA_ENGINE), compose fn-tp4zs2.yaml (passes the knobs
# and RADIANCE_LOG_STEPS: the step log; tools/fnsteps.py's full-row bucket, 16 sequences x 64 rows). One start a variant, the
# cards cool at each start; every start: c16 A B B A of S7 (A/A), the KFD queues of the engine process by card and type
# (/sys/class/kfd/kfd/proc/PID/queues), the cards' temperatures.
#   U1 not declared | F declared (default: queue at build, engine 1, priority 15, fine-grained scratch) | Q declared, the queue
#   opened lazily (no queue at decode) | P declared, queue priority 7 | C declared, coarse scratch | E0 declared, the queue on
#   engine 0 | QX lazy, then one ar_sdma 1 64k prefill (opens the queue), then c16 again | U2 not declared (brackets)
# Disk: not started under 12 GB free on /; logs gzipped.
# DRYRUN=1 bash 220-fn-sdma5.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4zs2.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/heatsoak.sh" ] && { TL=$f; break; }; done
IMG=${XSD5_IMAGE:-radiance-local:c16-x3sdma5}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-sd5
export FNIMAGE=$IMG FNMODELDIR=${XSD5_X3_DIR:-$HOME/models/flashnext-exl3} FNMODEL=${XSD5_X3_MODEL:-qwen3.8-flash-next-exl3.rad}
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
EXTRA0="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0"
export EXTRA="$EXTRA0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats-sd5
export FN4BT="" FN4BTP="" FN4BTL="" FN4X3DUAL=1 FN4NR=1 FN4GI=1 FN4FD="" FN4NW=1 FN4SW="" FN4WV=1 FN4RV=11 FN4AL="" FN4UM=5 FN4SS=1 FN4CH="" FN4XM=1 FN4SPHC=1 FN4X3C=1 FN4QSA="" FN4P2X="" FN4MP="" FN4X3GD=1 FN4X3DZ=1 PROFEVERY="" FN4MTPF=1
export FN4SDMA=0 SDMAENG=1 SDMAQLAZY="" SDMASCR="" SDMARR="" SDMAQPRIO="" LOGSTEPS=1
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
compose() { docker compose --project-directory $C -f $Y "$@"; }
down() { run "compose down > /dev/null 2>&1; docker rm -f $CN > /dev/null 2>&1"; }
[ -n "$TL" ] && . $TL/heatsoak.sh
caps() { local d h out=""; for d in /sys/class/drm/card*/device; do [ -e $d/unique_id ] || continue; h=$(ls -d $d/hwmon/hwmon* 2>/dev/null | head -1); [ -n "$h" ] || continue
  out="$out $(basename $(readlink -f $d)):cap=$(( $(cat $h/power1_cap 2>/dev/null || echo 0) / 1000000 ))W,hot=$(( $(cat $h/temp2_input 2>/dev/null || echo 0) / 1000 ))C"; done; echo "$out"; }
tguard() { local d h j m n=0 hot
  while [ ! -e $O/$TAG.tstop ]; do hot=""
    for d in /sys/class/drm/card*/device; do [ -e $d/unique_id ] || continue; h=$(ls -d $d/hwmon/hwmon* 2>/dev/null | head -1); [ -n "$h" ] || continue
      j=$(( $(cat $h/temp2_input 2>/dev/null || echo 0) / 1000 )); m=$(( $(cat $h/temp3_input 2>/dev/null || echo 0) / 1000 ))
      { [ $j -ge 100 ] || [ $m -ge 96 ]; } && hot="$hot $(basename $(readlink -f $d)):j$j/m$m"; done
    if [ -n "$hot" ]; then n=$((n + 1)); else n=0; fi
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 sdma5: VIRHE temperature guard ($hot) -> blocks stopped"; return 0; }
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
  [ $ok = 1 ] || { log "RESULT fn4 sdma5 restart-$tag: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
  log "RESULT fn4 sdma5 restart-$tag: healthy in $(( $(date +%s) - t0 )) s; $(grep -m1 -E '^I kv: ' $S | cut -c3-160); fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300)"
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
    || { log "RESULT fn4 sdma5 $1: VIRHE the switches are not $2 ($(echo "$g" | cut -c1-400))"; return 1; }
}
alive() {   # alive STEP: /health 200, else log it (the T arm then yields no answers)
  [ "$DRY" = 1 ] && { echo "+ curl -m 5 $URL/health == 200"; return 0; }
  local code; code=$(curl -s -m 5 -o /dev/null -w %{http_code} $URL/health)
  [ "$code" = 200 ] && return 0
  log "RESULT fn4 sdma5 $1: VIRHE the server is not healthy (/health $code; $(docker inspect -f {{.State.Status}} exit {{.State.ExitCode}} $CN 2>/dev/null || echo gone))"; return 1
}
GS=; for f in "$H/tools/greedyset.py" "$H/../tools/greedyset.py" "$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py"; do [ -z "$GS" ] && [ -e "$f" ] && GS=$f; done
LD=$TL/fn-loadid.py; MBT=$TL/fnharness/mbt2.py; GL=$TL/greedylong.py; BTP=$TL/fn-bitrace.py
# 213's S7 (hc_readv 75 = G16), its collective switches written out
S7='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 4, "hc_readv": 75, "hc_sp": 0, "mtp_flat": 0, "ar_xpush": 0}'
pcr() { run "curl -s -m 60 -o /dev/null -X POST -d '' $URL/reset_prefix_cache"; }
sw() { setsw "$(echo "$1" | sed 's/}$/, "epoch_bump": 1}/')"; nudge; swcheck "$2" "$1"; }
glr() { pcr; run "(cd $D && GL_N=4 timeout 1800 python3 $GL $URL $O/$TAG-$1.jsonl --conc 1 --max-tokens 64 > $O/$TAG-$1.txt 2>&1)"; }
gsr() { pcr; run "(cd $D && timeout 900 python3 $GS $URL $O/$TAG-$1.jsonl --conc 1 --max-tokens 192 > $O/$TAG-$1.txt 2>&1)"; }
lc() { [ "$DRY" = 1 ] && { echo "+ cmp $1 $2"; return 0; }; log "RESULT fn4 sdma5 text $3: $(timeout 300 python3 -I $LD cmp --url $URL --label "$3" $O/$TAG-$1.jsonl $O/$TAG-$2.jsonl 2>&1 | grep -m1 '^CMPLINE ' | cut -c9- | cut -c1-400)"; }
abr() {   # abr NAME A B (extra abblocks2 args...)
  local nm=$1 A=$2 B=$3; shift 3
  run "timeout 2400 python3 -I $TL/abblocks2.py --url $URL --mb-dir $D --label $TAG-$nm --rounds 1 --a '$A' --b '$B' --names a,b --out $O/$TAG-blocks.jsonl $* > $O/$TAG-$nm.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 sdma5 $nm (A B B A): $(grep '^ABB ' $O/$TAG-$nm.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=4: "-" if v is None else format(v, ".%df" % n)
s = "decode tok/s B/A %s %s ms/step B/A %s (A %s | B %s)" % (f(d["tps_ratio"]), (d.get("sep") or {}).get("tps"), f(d.get("ms_ratio")), " ".join("%.1f" % x for x in d.get("A_blocks", [])), " ".join("%.1f" % x for x in d.get("B_blocks", []))) if d.get("rounds") else ""
for k, v in (d.get("pf") or {}).items():
    s += " | prefill %s B/A %s %s (A %s | B %s)" % (k, f(v["ratio"]), v.get("sep"), " ".join("%.0f" % x for x in v.get("A_blocks", [])), " ".join("%.0f" % x for x in v.get("B_blocks", [])))
print(s + " | %d blocks, %d flagged, %s s" % (d["blocks"], d["blocks_bad"], d["wall_s"]))' 2>&1 | cut -c1-700)"
  [ "$DRY" = 1 ] || log "RESULT fn4 sdma5 $nm blocks (tok/s, ms/step, acceptance): $(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("label") == sys.argv[2]: print("%s %s tok/s %s ms/step acc %s;" % (r["arm"], r.get("tps"), r.get("ms"), (r.get("mb") or {}).get("acceptance")), end=" ")' $O/$TAG-blocks.jsonl $TAG-$nm 2>&1 | cut -c1-300)"
}
log "RESULT fn4 sdma5-fn-sdma: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)); U1 F Q P C E0 QX U2 c16 decode, step log | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 sdma5: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 sdma5: VIRHE image $IMG missing"; exit 1; }
  [ -e "$Y" ] && [ -n "$TL" ] && [ -e $TL/abblocks2.py ] && [ -e $MBT ] && [ -e $LD ] && [ -e $GL ] && [ -e $BTP ] && [ -n "$GS" ] && [ -e $D/mb.py ] && [ -e $TL/fnsteps.py ] || { log "RESULT fn4 sdma5: VIRHE tools missing"; exit 1; }
  grep -q 'RADIANCE_FN4_MTPFLAT' $Y && grep -q 'R4D_SDMA_QPRIO' $Y && grep -q 'RADIANCE_LOG_STEPS' $Y || { log "RESULT fn4 sdma5: VIRHE $YN does not pass RADIANCE_FN4_MTPFLAT / R4D_SDMA_QPRIO / RADIANCE_LOG_STEPS"; exit 1; }
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
[ "$DRY" != 1 ] && [ "${FREEG:-0}" -lt 12 ] && { log "RESULT fn4 sdma5: VIRHE only ${FREEG:-?} GB free on / -- not started"; exit 1; }
export FN4BT="" FN4BTP="" FN4BTL="" EXTRA="$EXTRA0 --tp-wire exact"
kfdq() {   # kfdq WHAT : the engine process's KFD queues by (gpu id, type: 0 compute, 1 SDMA, ...)
  [ "$DRY" = 1 ] && { echo "+ KFD queues of the radiance process ($1)"; return 0; }
  local p q out=""
  for p in $(ls /sys/class/kfd/kfd/proc/ 2>/dev/null); do
    [ "$(cat /proc/$p/comm 2>/dev/null)" = radiance ] || continue
    out="$out pid $p: $(for q in /sys/class/kfd/kfd/proc/$p/queues/*; do echo "gpu$(cat $q/gpuid 2>/dev/null)/t$(cat $q/type 2>/dev/null)"; done | sort | uniq -c | awk '{printf "%s x%s, ", $2, $1}')"
  done
  log "RESULT fn4 sdma5 $1 KFD queues:${out:- none found}"
}
vstart() {   # vstart NAME FN4SDMA QLAZY SCR QPRIO ENGINE : one start for a variant
  export FNIMAGE=$IMG FN4SDMA=$2 SDMAQLAZY=$3 SDMASCR=$4 SDMAQPRIO=$5 SDMAENG=$6
  restart $TAG-$1 || return 1
  nudge
  [ "$DRY" = 1 ] || log "RESULT fn4 sdma5 $1 start: AR_SDMA='$2' QLAZY='$3' SCR='$4' QPRIO='$5' ENGINE='$6'; $(grep -E 'fn4 SDMA-AR' $O/start-$TAG-$1.log | cut -c1-160 | tr '\n' ' ')| cards: $(caps)"
  kfdq $1
  sw "$S7" "S7"
  return 0
}
vsteps() {   # vsteps NAME LABEL... : the engine log (docker logs -t), fnsteps per label, down, gzip
  local nm=$1; shift
  [ "$DRY" = 1 ] || log "RESULT fn4 sdma5 $nm end: cards $(caps)"
  run "docker logs -t $CN > $O/$TAG-$nm-engine.log 2>&1"
  down
  local lab args=""
  for lab in "$@"; do args="$args --label $TAG-$lab:16:64"; done
  run "timeout 300 python3 -I $TL/fnsteps.py $O/$TAG-$nm-engine.log --blocks $O/$TAG-blocks.jsonl $args --json $O/$TAG-$nm-steps.json > $O/$TAG-$nm-steps.txt 2>&1"
  [ "$DRY" = 1 ] || grep -E '^FNSTEPS |^FNSTEPSB ' $O/$TAG-$nm-steps.txt | cut -c1-400 | while IFS= read -r l; do log "RESULT fn4 sdma5 $nm $l"; done
  run "gzip -f $O/$TAG-$nm-engine.log $O/start-$TAG-$nm.log 2>/dev/null; true"
}
c16aa() { abr $1 "$S7" "$S7" --mbt2 $MBT --gens 1 --phase c16 --warm 8; alive "$1 c16 S7 A/A"; }
SX='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 4, "hc_readv": 75, "hc_sp": 2, "mtp_flat": 0, "ar_xpush": 1, "ar_sdma": 1}'
#            NAME FN4SDMA QLAZY SCR QPRIO ENGINE
vstart U1 "" "" "" "" ""  && { c16aa u1; vsteps U1 u1; }
vstart F 0 "" "" "" ""    && { c16aa f; vsteps F f; }
vstart Q 0 1 "" "" ""     && { c16aa q; vsteps Q q; }
vstart P 0 "" "" 7 ""     && { c16aa p; vsteps P p; }
vstart C 0 "" 0 "" ""     && { c16aa c; vsteps C c; }
vstart E0 0 "" "" "" 0    && { c16aa e0; vsteps E0 e0; }
if vstart QX 0 1 "" "" ""; then
  c16aa qx1
  # one SDMA prefill: the lazy queues open at the first copy-engine launch
  sw "$SX" "SX"; pcr; run "(cd $D && timeout 900 python3 ds41_tests.py prefill --depths 65536 --cpt 3.6 --calibrate 0 --reps 1 > $O/$TAG-qx-pf.txt 2>&1)"; alive "QX sdma prefill"
  kfdq QX-after-prefill
  sw "$S7" "S7"
  c16aa qx2
  vsteps QX qx1 qx2
fi
vstart U2 "" "" "" "" ""  && { c16aa u2; vsteps U2 u2; }
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
log "RESULT fn4 sdma5-fn-sdma: done"
