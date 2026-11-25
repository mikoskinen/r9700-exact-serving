#!/bin/bash
# 223-fn-sdma6.sh (queue item, 13.11.2026, SDMA-AR agent, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~35 min)
# -- SDMA-AR DECLARED vs NOT: c16 AND c8 ON THE STEP LOG, THEN THE PREFILL GAIN. csdma5 (12.11.): c16 full-row steps declared
# (default) 37.63 / 37.59 ms p50 vs not declared 37.54-37.64 (+0.05 %); 219's -5 % was request-turnover steps (~70 ms instead
# of 37 on ~10 % of the full-row steps in some starts / blocks, with or without a queue). Image radiance-local:c16-x3sdma6
# (fn4-x3-sp3-sdma 5235bd07 = fn4-x3-sp3 984f559e + SDMA-AR; its probe knobs unset = the default context), compose
# fn-tp4zs3.yaml (step log on). Every start: the KFD queues, the cards' temperatures; every block: fnsteps' full-row bucket
# (16 x 64 at c16, 8 x 32 at c8) and the slow full-row steps (> 1.33x the block's p50) counted.
#   U1 not declared: c16 and c8 A/A of S7
#   D  declared (RADIANCE_FN4_AR_SDMA=0): c16 and c8 A/A of S7; the long texts at ar_sdma 0 / 1 / 0; a warm-up 64k prefill and
#      fresh 64k / 128k A B B A ar_sdma 0 vs 1 (216's base); then c16 and c8 A/A of S7 again (the queues have run)
#   U2 not declared again (brackets)
# Target (coordinator 12.11.): declared == not declared within 0.5 % at c16 / c8 (full-row bucket), the prefill gain kept.
# Disk / inodes: not started under 12 GB free on / or above 95 % inodes used on /tmp; logs gzipped.
# DRYRUN=1 bash 223-fn-sdma6.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4zs3.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/heatsoak.sh" ] && { TL=$f; break; }; done
IMG=${XSD6_IMAGE:-radiance-local:c16-x3sdma6}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-sd6
export FNIMAGE=$IMG FNMODELDIR=${XSD6_X3_DIR:-$HOME/models/flashnext-exl3} FNMODEL=${XSD6_X3_MODEL:-qwen3.8-flash-next-exl3.rad}
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
EXTRA0="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0"
export EXTRA="$EXTRA0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats-sd6
export FN4BT="" FN4BTP="" FN4BTL="" FN4X3DUAL=1 FN4NR=1 FN4GI=1 FN4FD="" FN4NW=1 FN4SW="" FN4WV=1 FN4RV=11 FN4AL="" FN4UM=5 FN4SS=1 FN4CH="" FN4XM=1 FN4SPHC=1 FN4X3C=1 FN4QSA="" FN4P2X="" FN4MP="" FN4X3GD=1 FN4X3DZ=1 PROFEVERY="" FN4MTPF=1
export FN4SDMA=0 SDMAENG=1 SDMAQLAZY="" SDMASCR="" SDMARR="" SDMAQPRIO="" SDMAQIDLE="" LOGSTEPS=1
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
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 sdma6: VIRHE temperature guard ($hot) -> blocks stopped"; return 0; }
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
  [ $ok = 1 ] || { log "RESULT fn4 sdma6 restart-$tag: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
  log "RESULT fn4 sdma6 restart-$tag: healthy in $(( $(date +%s) - t0 )) s; $(grep -m1 -E '^I kv: ' $S | cut -c3-160); fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300)"
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
    || { log "RESULT fn4 sdma6 $1: VIRHE the switches are not $2 ($(echo "$g" | cut -c1-400))"; return 1; }
}
alive() {   # alive STEP: /health 200, else log it (the T arm then yields no answers)
  [ "$DRY" = 1 ] && { echo "+ curl -m 5 $URL/health == 200"; return 0; }
  local code; code=$(curl -s -m 5 -o /dev/null -w %{http_code} $URL/health)
  [ "$code" = 200 ] && return 0
  log "RESULT fn4 sdma6 $1: VIRHE the server is not healthy (/health $code; $(docker inspect -f {{.State.Status}} exit {{.State.ExitCode}} $CN 2>/dev/null || echo gone))"; return 1
}
GS=; for f in "$H/tools/greedyset.py" "$H/../tools/greedyset.py" "$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py"; do [ -z "$GS" ] && [ -e "$f" ] && GS=$f; done
LD=$TL/fn-loadid.py; MBT=$TL/fnharness/mbt2.py; GL=$TL/greedylong.py; BTP=$TL/fn-bitrace.py
# 213's S7 (hc_readv 75 = G16), its collective switches written out
S7='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 4, "hc_readv": 75, "hc_sp": 0, "mtp_flat": 0, "ar_xpush": 0}'
pcr() { run "curl -s -m 60 -o /dev/null -X POST -d '' $URL/reset_prefix_cache"; }
sw() { setsw "$(echo "$1" | sed 's/}$/, "epoch_bump": 1}/')"; nudge; swcheck "$2" "$1"; }
glr() { pcr; run "(cd $D && GL_N=4 timeout 1800 python3 $GL $URL $O/$TAG-$1.jsonl --conc 1 --max-tokens 64 > $O/$TAG-$1.txt 2>&1)"; }
gsr() { pcr; run "(cd $D && timeout 900 python3 $GS $URL $O/$TAG-$1.jsonl --conc 1 --max-tokens 192 > $O/$TAG-$1.txt 2>&1)"; }
lc() { [ "$DRY" = 1 ] && { echo "+ cmp $1 $2"; return 0; }; log "RESULT fn4 sdma6 text $3: $(timeout 300 python3 -I $LD cmp --url $URL --label "$3" $O/$TAG-$1.jsonl $O/$TAG-$2.jsonl 2>&1 | grep -m1 '^CMPLINE ' | cut -c9- | cut -c1-400)"; }
abr() {   # abr NAME A B (extra abblocks2 args...)
  local nm=$1 A=$2 B=$3; shift 3
  run "timeout 2400 python3 -I $TL/abblocks2.py --url $URL --mb-dir $D --label $TAG-$nm --rounds 1 --a '$A' --b '$B' --names a,b --out $O/$TAG-blocks.jsonl $* > $O/$TAG-$nm.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 sdma6 $nm (A B B A): $(grep '^ABB ' $O/$TAG-$nm.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=4: "-" if v is None else format(v, ".%df" % n)
s = "decode tok/s B/A %s %s ms/step B/A %s (A %s | B %s)" % (f(d["tps_ratio"]), (d.get("sep") or {}).get("tps"), f(d.get("ms_ratio")), " ".join("%.1f" % x for x in d.get("A_blocks", [])), " ".join("%.1f" % x for x in d.get("B_blocks", []))) if d.get("rounds") else ""
for k, v in (d.get("pf") or {}).items():
    s += " | prefill %s B/A %s %s (A %s | B %s)" % (k, f(v["ratio"]), v.get("sep"), " ".join("%.0f" % x for x in v.get("A_blocks", [])), " ".join("%.0f" % x for x in v.get("B_blocks", [])))
print(s + " | %d blocks, %d flagged, %s s" % (d["blocks"], d["blocks_bad"], d["wall_s"]))' 2>&1 | cut -c1-700)"
  [ "$DRY" = 1 ] || log "RESULT fn4 sdma6 $nm blocks (tok/s, ms/step, acceptance): $(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("label") == sys.argv[2]: print("%s %s tok/s %s ms/step acc %s;" % (r["arm"], r.get("tps"), r.get("ms"), (r.get("mb") or {}).get("acceptance")), end=" ")' $O/$TAG-blocks.jsonl $TAG-$nm 2>&1 | cut -c1-300)"
}
log "RESULT fn4 sdma6-fn-sdma: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)); U1 / D (declared) / U2 c16 + c8 on the step log, D prefill ABBA | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 sdma6: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 sdma6: VIRHE image $IMG missing"; exit 1; }
  [ -e "$Y" ] && [ -n "$TL" ] && [ -e $TL/abblocks2.py ] && [ -e $MBT ] && [ -e $LD ] && [ -e $GL ] && [ -e $BTP ] && [ -n "$GS" ] && [ -e $D/mb.py ] && [ -e $TL/fnsteps.py ] || { log "RESULT fn4 sdma6: VIRHE tools missing"; exit 1; }
  grep -q 'RADIANCE_FN4_MTPFLAT' $Y && grep -q 'R4D_SDMA_QIDLE_MS' $Y && grep -q 'RADIANCE_LOG_STEPS' $Y || { log "RESULT fn4 sdma6: VIRHE $YN does not pass RADIANCE_FN4_MTPFLAT / R4D_SDMA_QIDLE_MS / RADIANCE_LOG_STEPS"; exit 1; }
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
[ "$DRY" != 1 ] && [ "${FREEG:-0}" -lt 12 ] && { log "RESULT fn4 sdma6: VIRHE only ${FREEG:-?} GB free on / -- not started"; exit 1; }
FREEI=$(df --output=ipcent /tmp 2>/dev/null | tail -1 | tr -dc 0-9)
[ "$DRY" != 1 ] && [ "${FREEI:-100}" -gt 95 ] && { log "RESULT fn4 sdma6: VIRHE /tmp inodes ${FREEI}% used -- not started"; exit 1; }
export FN4BT="" FN4BTP="" FN4BTL="" EXTRA="$EXTRA0 --tp-wire exact"
kfdq() {   # kfdq WHAT : the engine process's KFD queues by (gpu id, type: 0 compute, 1 SDMA, ...)
  [ "$DRY" = 1 ] && { echo "+ KFD queues of the radiance process ($1)"; return 0; }
  local p q out=""
  for p in $(ls /sys/class/kfd/kfd/proc/ 2>/dev/null); do
    [ "$(cat /proc/$p/comm 2>/dev/null)" = radiance ] || continue
    out="$out pid $p: $(for q in /sys/class/kfd/kfd/proc/$p/queues/*; do echo "gpu$(cat $q/gpuid 2>/dev/null)/t$(cat $q/type 2>/dev/null)"; done | sort | uniq -c | awk '{printf "%s x%s, ", $2, $1}')"
  done
  log "RESULT fn4 sdma6 $1 KFD queues:${out:- none found}"
}
vstart() {   # vstart NAME FN4SDMA QLAZY SCR QPRIO ENGINE QIDLE : one start for a variant
  export FNIMAGE=$IMG FN4SDMA=$2 SDMAQLAZY=$3 SDMASCR=$4 SDMAQPRIO=$5 SDMAENG=$6 SDMAQIDLE=$7
  restart $TAG-$1 || return 1
  nudge
  [ "$DRY" = 1 ] || log "RESULT fn4 sdma6 $1 start: AR_SDMA='$2' QLAZY='$3' SCR='$4' QPRIO='$5' ENGINE='$6' QIDLE='$7'; $(grep -E 'fn4 SDMA-AR' $O/start-$TAG-$1.log | cut -c1-160 | tr '\n' ' ')| cards: $(caps)"
  kfdq $1
  sw "$S7" "S7"
  return 0
}
vsteps() {   # vsteps NAME LABEL... : the engine log (docker logs -t), fnsteps per label, down, gzip
  local nm=$1; shift
  [ "$DRY" = 1 ] || log "RESULT fn4 sdma6 $nm end: cards $(caps)"
  run "docker logs -t $CN > $O/$TAG-$nm-engine.log 2>&1"
  down
  local lab args=""
  for lab in "$@"; do args="$args --label $TAG-$lab"; done
  run "timeout 300 python3 -I $TL/fnsteps.py $O/$TAG-$nm-engine.log --blocks $O/$TAG-blocks.jsonl $args --json $O/$TAG-$nm-steps.json > $O/$TAG-$nm-steps.txt 2>&1"
  [ "$DRY" = 1 ] || grep -E '^FNSTEPS |^FNSTEPSB ' $O/$TAG-$nm-steps.txt | cut -c1-400 | while IFS= read -r l; do log "RESULT fn4 sdma6 $nm $l"; done
  [ "$DRY" = 1 ] || python3 -I - $O/$TAG-$nm-engine.log <<'PY' | while IFS= read -r l; do log "RESULT fn4 sdma6 $nm $l"; done
import re, statistics, sys
STEP = re.compile(r"^(\S+)\s+I step (\d+) (prefill|decode|MIXED) n_seq=(\d+) n_tok=(\d+)")
APPL = re.compile(r"^(\S+)\s+I fn4: switches .* applied at step (\d+)")
def ts(s):
    m = re.match(r"(\d+)-(\d+)-(\d+)T(\d+):(\d+):(\d+)(?:\.(\d+))?", s)
    return int(m.group(4)) * 3600 + int(m.group(5)) * 60 + int(m.group(6)) + float("0." + (m.group(7) or "0"))
steps, appl = [], []
for l in open(sys.argv[1], errors="replace"):
    m = STEP.match(l)
    if m: steps.append((int(m.group(2)), ts(m.group(1)), m.group(3), int(m.group(4)), int(m.group(5)))); continue
    m = APPL.match(l)
    if m: appl.append(int(m.group(2)))
dur = {a[0]: b[1] - a[1] for a, b in zip(steps, steps[1:]) if b[0] == a[0] + 1 and b[1] - a[1] < 5}
res = []
for i in range(len(appl) - 1):
    s0, s1 = appl[i], appl[i + 1]
    for ns, nt in ((16, 64), (8, 32)):
        full = [1000 * dur[st[0]] for st in steps if s0 <= st[0] < s1 and st[0] in dur and st[2] == "decode" and st[3] == ns and st[4] == nt]
        if len(full) < 50: continue
        p50 = statistics.median(full); slow = [d for d in full if d > 1.33 * p50]
        res.append("steps %d-%d %dx%d: %d/%d slow (> %.0f ms)%s" % (s0, s1, ns, nt, len(slow), len(full), 1.33 * p50,
                   " med %.0f ms" % statistics.median(slow) if slow else ""))
print("slow full-row steps by block: " + (" | ".join(res) if res else "none"))
PY
  run "gzip -f $O/$TAG-$nm-engine.log $O/start-$TAG-$nm.log 2>/dev/null; true"
}
c16aa() { abr $1 "$S7" "$S7" --mbt2 $MBT --gens 1 --phase c16 --warm 8; alive "$1 c16 S7 A/A"; }
c8aa() { abr $1 "$S7" "$S7" --mbt2 $MBT --gens 1 --phase c8 --warm 8; alive "$1 c8 S7 A/A"; }
SB='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 4, "hc_readv": 75, "hc_sp": 2, "mtp_flat": 0, "ar_xpush": 1, "ar_sdma": 0}'
SD='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 4, "hc_readv": 75, "hc_sp": 2, "mtp_flat": 0, "ar_xpush": 1, "ar_sdma": 1}'
#            NAME FN4SDMA QLAZY SCR QPRIO ENGINE QIDLE
vstart U1 "" "" "" "" "" "" && { c16aa u1c16; c8aa u1c8; vsteps U1 u1c16:16:64 u1c8:8:32; }
if vstart D 0 "" "" "" "" ""; then
  c16aa dc16a; c8aa dc8a
  sw "$SB" "SB" && glr t0; alive "D texts ar_sdma 0"
  sw "$SD" "SD" && glr t1; alive "D texts ar_sdma 1"
  sw "$SB" "SB" && glr t2; alive "D texts ar_sdma 0 again"
  lc t0 t1 "declared: ar_sdma 1 long, vs ar_sdma 0"; lc t0 t2 "control"
  pcr; run "(cd $D && timeout 900 python3 ds41_tests.py prefill --depths 65536 --cpt 3.6 --calibrate 0 --reps 1 > $O/$TAG-warm-pf.txt 2>&1)"; alive "D warm-up"
  abr dpf "$SB" "$SD" --no-mb --pf-depths 65536,131072; alive "D prefill ar_sdma 0 vs 1"
  kfdq D-after-prefill
  sw "$S7" "S7"
  c16aa dc16b; c8aa dc8b
  vsteps D dc16a:16:64 dc8a:8:32 dc16b:16:64 dc8b:8:32
fi
vstart U2 "" "" "" "" "" "" && { c16aa u2c16; c8aa u2c8; vsteps U2 u2c16:16:64 u2c8:8:32; }
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
log "RESULT fn4 sdma6-fn-sdma: done"
