#!/bin/bash
# 221-fn-biprof2.sh = 218 RERUN (13.11.2026): 218 lost every window but the last -- docker's json-file log keeps 5 x 100 MB and
# the profiled run wrote more, so `docker logs` at the end returned only the tail (2 of 17 applied lines). Here the log is FOLLOWED from
# the start (docker logs -f | gzip -1, the full log), the prefix cache is reset before every window (218: prefill_tok 8683 vs 9816
# between windows of the same requests), opdiff reads a window per table (--per-table: unequal step counts). Otherwise as 218:
# 218-fn-biprof.sh (queue item, 13.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~25 min) -- WHERE hc_bi AND
# att_bi COST AT c8 / c4 (214 on S7Q, mb A B B A: c8 hc_bi +5.5 % ms/step, att_bi +3.2 %; c4 hc_bi +1.4 %, att_bi +2.0 %; bi_seg /
# exl3_dual ~0) and the residual at c16, op by op: radiance-local:c16-x3sp17 (the engagement counter) on fn-tp4zu.yaml, ONE start
# with --profile-ops (a cumulative per-op table a rank every 4 steps; profiled steps synchronise: device ms / calls / us a call are
# the reading, not tok/s), in-process windows of one seeded mbt2 generation each (the same work every window):
#   c8  P S7, H S7 hc_bi 0, T S7 att_bi 0, O OFF, then O T H P (A B C D D C B A)   [13.11.: base S7 = hc_readv 75 (G16), not S7Q:
#   217 measured G16q +2.0 % a full-row step at c16 and c8 -- 214's per-lever numbers were taken on S7Q]
#   c16 P, O, O, P        c4  P, H, T
# windows as engine-log lines of each window's "applied at step ... (epoch E)" (exact, no wall clock), tools/opwin.py --ops .
# then tools/opdiff.py (per op name and key, arm vs P: delta ms / calls / us a call), plus the forms the counter saw a window.
# DRYRUN=1 bash 218-fn-biprof.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4zu.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/heatsoak.sh" ] && { TL=$f; break; }; done
IMG=${X218_IMAGE:-radiance-local:c16-x3sp17}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-bp2
export FNIMAGE=$IMG FNMODELDIR=${X218_X3_DIR:-$HOME/models/flashnext-exl3} FNMODEL=${X218_X3_MODEL:-qwen3.8-flash-next-exl3.rad}
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
EXTRA0="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0"
export EXTRA="$EXTRA0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats221
export FN4BT="" FN4BTP="" FN4BTL="" FN4X3DUAL=1 FN4NR=1 FN4GI=1 FN4FD="" FN4NW=1 FN4SW="" FN4WV=1 FN4RV=11 FN4AL="" FN4UM=5 FN4SS=1 FN4CH="" FN4XM=1 FN4SPHC=1 FN4X3C=1 FN4QSA="" FN4P2X="" FN4MP="" FN4X3GD=1 FN4X3DZ=1 PROFEVERY=4 FN4MTPF=1 LOGSTEPS=1 FN4FC=1
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
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 221: VIRHE temperature guard ($hot) -> blocks stopped"; return 0; }
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
  [ $ok = 1 ] || { log "RESULT fn4 221 restart-$tag: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
  log "RESULT fn4 221 restart-$tag: healthy in $(( $(date +%s) - t0 )) s; $(grep -m1 -E '^I kv: ' $S | cut -c3-160); fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300)"
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
    || { log "RESULT fn4 221 $1: VIRHE the switches are not $2 ($(echo "$g" | cut -c1-400))"; return 1; }
}
alive() {   # alive STEP: /health 200, else log it (the T arm then yields no answers)
  [ "$DRY" = 1 ] && { echo "+ curl -m 5 $URL/health == 200"; return 0; }
  local code; code=$(curl -s -m 5 -o /dev/null -w %{http_code} $URL/health)
  [ "$code" = 200 ] && return 0
  log "RESULT fn4 221 $1: VIRHE the server is not healthy (/health $code; $(docker inspect -f {{.State.Status}} exit {{.State.ExitCode}} $CN 2>/dev/null || echo gone))"; return 1
}
GS=; for f in "$H/tools/greedyset.py" "$H/../tools/greedyset.py" "$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py"; do [ -z "$GS" ] && [ -e "$f" ] && GS=$f; done
LD=$TL/fn-loadid.py; MBT=$TL/fnharness/mbt2.py; GL=$TL/greedylong.py; BTP=$TL/fn-bitrace.py
S0='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 4, "hc_sp": 0, "mtp_flat": 0, "ar_xpush": 0}'
S0X='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 4, "hc_sp": 0, "mtp_flat": 0, "ar_xpush": 1}'
S2='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 4, "hc_sp": 2, "mtp_flat": 0, "ar_xpush": 0}'
S2X='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 4, "hc_sp": 2, "mtp_flat": 0, "ar_xpush": 1}'
pcr() { run "curl -s -m 60 -o /dev/null -X POST -d '' $URL/reset_prefix_cache"; }
sw() { setsw "$(echo "$1" | sed 's/}$/, "epoch_bump": 1}/')"; nudge; swcheck "$2" "$1"; }
glr() { pcr; run "(cd $D && GL_N=4 timeout 1800 python3 $GL $URL $O/$TAG-$1.jsonl --conc 1 --max-tokens 64 > $O/$TAG-$1.txt 2>&1)"; }
gsr() { pcr; run "(cd $D && timeout 900 python3 $GS $URL $O/$TAG-$1.jsonl --conc 1 --max-tokens 192 > $O/$TAG-$1.txt 2>&1)"; }
lc() { [ "$DRY" = 1 ] && { echo "+ cmp $1 $2"; return 0; }; log "RESULT fn4 221 text $3: $(timeout 300 python3 -I $LD cmp --url $URL --label "$3" $O/$TAG-$1.jsonl $O/$TAG-$2.jsonl 2>&1 | grep -m1 '^CMPLINE ' | cut -c9- | cut -c1-400)"; }
abr() {   # abr NAME A B (extra abblocks2 args...)
  local nm=$1 A=$2 B=$3; shift 3
  run "timeout 2400 python3 -I $TL/abblocks2.py --url $URL --mb-dir $D --label $TAG-$nm --rounds 1 --a '$A' --b '$B' --names a,b --out $O/$TAG-blocks.jsonl $* > $O/$TAG-$nm.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 221 $nm (A B B A): $(grep '^ABB ' $O/$TAG-$nm.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=4: "-" if v is None else format(v, ".%df" % n)
s = "decode tok/s B/A %s %s ms/step B/A %s (A %s | B %s)" % (f(d["tps_ratio"]), (d.get("sep") or {}).get("tps"), f(d.get("ms_ratio")), " ".join("%.1f" % x for x in d.get("A_blocks", [])), " ".join("%.1f" % x for x in d.get("B_blocks", []))) if d.get("rounds") else ""
for k, v in (d.get("pf") or {}).items():
    s += " | prefill %s B/A %s %s (A %s | B %s)" % (k, f(v["ratio"]), v.get("sep"), " ".join("%.0f" % x for x in v.get("A_blocks", [])), " ".join("%.0f" % x for x in v.get("B_blocks", [])))
print(s + " | %d blocks, %d flagged, %s s" % (d["blocks"], d["blocks_bad"], d["wall_s"]))' 2>&1 | cut -c1-700)"
  [ "$DRY" = 1 ] || log "RESULT fn4 221 $nm blocks (tok/s, ms/step, acceptance): $(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("label") == sys.argv[2]: print("%s %s tok/s %s ms/step acc %s;" % (r["arm"], r.get("tps"), r.get("ms"), (r.get("mb") or {}).get("acceptance")), end=" ")' $O/$TAG-blocks.jsonl $TAG-$nm 2>&1 | cut -c1-300)"
}
log "RESULT fn4 221-fn-biprof2: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) exact wire + SPHC + MTPFLAT declared + step log + form counter + --profile-ops every 4 steps; c8 / c16 / c4 windows S7Q / no hc_bi / no att_bi / OFF | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 221: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 221: VIRHE image $IMG missing"; exit 1; }
  [ -e "$Y" ] && [ -n "$TL" ] && [ -e $TL/abblocks2.py ] && [ -e $MBT ] && [ -e $LD ] && [ -e $GL ] && [ -e $BTP ] && [ -n "$GS" ] && [ -e $D/mb.py ] && [ -e $TL/opwin.py ] && [ -e $TL/opdiff.py ] && [ -e $TL/opprof.py ] || { log "RESULT fn4 221: VIRHE tools missing"; exit 1; }
  grep -q 'RADIANCE_FN4_FORMCTR' $Y && grep -q 'RADIANCE_LOG_STEPS' $Y && grep -q 'RADIANCE_FN4_MTPFLAT' $Y || { log "RESULT fn4 221: VIRHE $YN does not pass RADIANCE_FN4_MTPFLAT / _FORMCTR / RADIANCE_LOG_STEPS"; exit 1; }
  mkdir -p $O $STATSDIR/bt-sp && chmod 777 $STATSDIR/bt-sp && find $STATSDIR/bt-sp -maxdepth 1 -name 'bt-r*' -delete
  find $O -maxdepth 1 -name "$TAG.abort" -delete; find $O -maxdepth 1 -name "$TAG.tstop" -delete
  find $O -maxdepth 1 \( -name "$TAG-*.jsonl" -o -name "$TAG-*.txt" \) -delete
  trap 'touch $O/$TAG.tstop 2>/dev/null; down' EXIT
  trap 'exit 143' TERM INT
  tguard & TGPID=$!
else
  echo "+ guards; tguard &"; TGPID=""
fi
export EXTRA="$EXTRA0 --tp-wire exact --profile-ops"
FREEG=$(df --output=avail -BG / 2>/dev/null | tail -1 | tr -dc 0-9)
[ "$DRY" != 1 ] && [ "${FREEG:-0}" -lt 12 ] && { log "RESULT fn4 221: VIRHE only ${FREEG:-?} GB free on / -- not started"; exit 1; }
S7Q='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 4, "hc_readv": 75}'
NOHC='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 0, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 4, "hc_readv": 75}'
NOATT='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 0, "arw_rows": 16, "sched_bi_chunk": 4, "hc_readv": 75}'
OFF='{"exl3_bi": 0, "exl3_dual": 0, "hc_bi": 0, "bi_seg": 0, "att_bi": 0, "arw_rows": 0, "sched_bi_chunk": 0, "hc_readv": 11}'
WE=""
win() {   # win LABEL SWITCHES PHASE: switch (+ epoch bump), note the epoch, one seeded mbt2 generation at PHASE
  [ -e $O/$TAG.abort ] && return 1
  sw "$2" "$1" || return 1
  local e n
  if [ "$DRY" = 1 ]; then e=0; else e=$(curl -s -m 5 $URL/admin/fn4 | python3 -c 'import json,sys; print(json.load(sys.stdin).get("epoch"))' 2>/dev/null); fi
  n=${3//[!0-9]/}
  pcr
  run "(cd $D && timeout 900 python3 $MBT --mb mb.py --temperature 0 --tag $TAG-$1 --phases $3 --pattern 1x$n --seed 1 --gens 1 --warm 8 > $O/$TAG-$1-mb.txt 2>&1)"
  WE="$WE $1:${e:-x}"
  [ "$DRY" = 1 ] || log "RESULT fn4 221 window $1 ($3, epoch ${e:-?}): $(grep '^MB ' $O/$TAG-$1-mb.txt | cut -c4- | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print("profiled agg %s tok/s, %s decode steps, dec_tok %s, prefill_tok %s, acceptance %s" % (d.get("agg_dec_tps"), d.get("steps"), d.get("dec_tok"), d.get("prefill_tok"), d.get("acceptance")))' 2>&1 | cut -c1-200)"
}
restart $TAG-S || exit 1
# the FULL log: follow it from the container's start (json-file keeps only 5 x 100 MB; 218 lost its windows to the rotation)
if [ "$DRY" = 1 ]; then echo "+ docker logs -f $CN | gzip -1 > $O/$TAG-engine.log.gz &"; CAPPID=""
else (docker logs -f $CN 2>&1 | gzip -1 > $O/$TAG-engine.log.gz) & CAPPID=$!; fi
nudge
[ "$DRY" = 1 ] || log "RESULT fn4 221 engine notes: $(docker logs $CN 2>&1 | grep -E 'chained launches:|profil' | sed 's/^[A-Z] //' | cut -c1-160 | tr '\n' ' ' | cut -c1-400)"
sw "$S7Q" "S7Q" || { down; exit 1; }
run "(cd $D && timeout 600 python3 $MBT --mb mb.py --temperature 0 --tag $TAG-warm --phases c8 --pattern 1x8 --seed 1 --gens 1 --warm 8 > $O/$TAG-warm-mb.txt 2>&1)"; alive "warm-up"
win P8a "$S7Q" c8; win H8a "$NOHC" c8; win T8a "$NOATT" c8; win O8a "$OFF" c8
win O8b "$OFF" c8; win T8b "$NOATT" c8; win H8b "$NOHC" c8; win P8b "$S7Q" c8; alive "c8 windows"
win P16a "$S7Q" c16; win O16a "$OFF" c16; win O16b "$OFF" c16; win P16b "$S7Q" c16; alive "c16 windows"
win P4a "$S7Q" c4; win H4a "$NOHC" c4; win T4a "$NOATT" c4; alive "c4 windows"
sw "$S7Q" "end"   # the last window's end: one more applied line
down
[ -n "$CAPPID" ] && { for i in $(seq 1 60); do kill -0 $CAPPID 2>/dev/null || break; sleep 2; done; kill $CAPPID 2>/dev/null; }
if [ "$DRY" = 1 ]; then echo "+ windows from the epochs [$WE] -> opwin.py --ops . --json; opdiff.py c8 (H / T / O vs P), c16 (O vs P), c4 (H / T vs P); fnsteps-free form counts a window"
else
  WINS=$(python3 - $O/$TAG-engine.log.gz $WE <<'PY'
import gzip, re, sys
f = sys.argv[1]; we = [x.split(":") for x in sys.argv[2:]]
ln = {}
for n, l in enumerate(gzip.open(f, "rt", errors="replace"), 1):
    m = re.search(r"fn4: switches .* applied at step \d+ \(epoch (\d+)\)", l)
    if m: ln[int(m.group(1))] = n
eps = sorted(ln)
out = []
for lab, e in we:
    if not e.isdigit() or int(e) not in ln: continue
    e = int(e); nxt = [x for x in eps if x > e]
    if not nxt: continue
    out += ["--win", "%s:%d:%d" % (lab, ln[e], ln[nxt[0]])]
print(" ".join(out))
PY
)
  log "RESULT fn4 221 windows as log lines: $WINS"
  timeout 1500 python3 -I $TL/opwin.py $O/$TAG-engine.log.gz $WINS --ops . --top 12 --json $O/$TAG-opwin.json > $O/$TAG-opwin.txt 2>&1
  grep '^OPWIN ' $O/$TAG-opwin.txt | grep ' rank 0:' | while IFS= read -r l; do log "RESULT fn4 221 $(echo "$l" | cut -c1-900)"; done
  timeout 300 python3 -I $TL/opdiff.py $O/$TAG-opwin.json --arm P8=P8a,P8b --arm H8=H8a,H8b --arm T8=T8a,T8b --arm O8=O8a,O8b --base P8 --top 12 --per-table > $O/$TAG-opdiff8.txt 2>&1
  timeout 300 python3 -I $TL/opdiff.py $O/$TAG-opwin.json --arm P16=P16a,P16b --arm O16=O16a,O16b --base P16 --top 12 --per-table > $O/$TAG-opdiff16.txt 2>&1
  timeout 300 python3 -I $TL/opdiff.py $O/$TAG-opwin.json --arm P4=P4a --arm H4=H4a --arm T4=T4a --base P4 --top 12 --per-table > $O/$TAG-opdiff4.txt 2>&1
  grep -h '^OPDIFF ' $O/$TAG-opdiff8.txt $O/$TAG-opdiff16.txt $O/$TAG-opdiff4.txt | grep -v 'by key' | grep ' rank 0 ' | while IFS= read -r l; do log "RESULT fn4 221 $(echo "$l" | cut -c1-1800)"; done
  grep -h '^Traceback\|VIRHE' $O/$TAG-opwin.txt $O/$TAG-opdiff*.txt | head -3 | while IFS= read -r l; do log "RESULT fn4 221 analysis: VIRHE $l"; done
  log "RESULT fn4 221 forms (counter, cumulative at the end, profiled = every step live): $(zcat $O/$TAG-engine.log.gz | grep 'fn4 formctr dev' | awk '{d[$4] = $0} END {for (k in d) print d[k]}' | sort | sed 's/^fn4 formctr //' | tr '\n' '|' | cut -c1-900)"
  log "RESULT fn4 221 engine: $(zcat $O/$TAG-engine.log.gz | grep -cE 'fn4: switches .* applied') applied changes (17 expected), $(zcat $O/$TAG-engine.log.gz | grep -cE 'fn4: .* refused') refused; E lines $(zcat $O/$TAG-engine.log.gz | grep -cE '^E '); $(zcat $O/$TAG-engine.log.gz | wc -l) log lines"
fi
run "gzip -f $O/start-$TAG-S.log 2>/dev/null; true"
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
log "RESULT fn4 221-fn-biprof2: done ($O/$TAG-opdiff8.txt / 16 / 4, $O/$TAG-opwin.json)"
