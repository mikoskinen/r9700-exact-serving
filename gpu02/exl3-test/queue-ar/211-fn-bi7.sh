#!/bin/bash
# 211-fn-bi7.sh (queue item, 12.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~40 min) -- c16 DECODE
# TOWARDS 1000 WITH FULL BI: radiance-local:c16-x3sp11 (fn4-x3-sp3 16d681cf: hc_bi 5 now reads the final read in one launch
# too, per-row kernels at every height; G16 = hc_readv bit 6, the 16-row form G; the att_bi record-key fix), 205's S6 set.
# S7 = S6 + hc_readv 75 (11 + G16). START (untraced): BI acceptance of S7 -- greedy-20 a1 / c4 / pf / c16 / a2, the LONG set
# at sched_bi_chunk 5 (La1 / Lc4 / Lpf); then mbt2 A B B A: c16 S6 vs S7 (G16), and OFF vs S7 at c4 / c8 / c16. START 2
# (--profile-ops every 4 steps, >= 12 GB free): mb c16 windows S7 / S7 hc_bi 0 / S7 att_bi 0 / OFF -> tools/opwin.py (supersedes 208).
# DRYRUN=1 bash 211-fn-bi7.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4zt.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/heatsoak.sh" ] && { TL=$f; break; }; done
IMG=${X211_IMAGE:-radiance-local:c16-x3sp11}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-bi7
export FNIMAGE=$IMG FNMODELDIR=${X211_X3_DIR:-$HOME/models/flashnext-exl3} FNMODEL=${X211_X3_MODEL:-qwen3.8-flash-next-exl3.rad}
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
EXTRA0="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0"
export EXTRA="$EXTRA0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats211
export FN4BT="" FN4BTP="" FN4BTL="" FN4X3DUAL=1 FN4NR=1 FN4GI=1 FN4FD="" FN4NW=1 FN4SW="" FN4WV=1 FN4RV=11 FN4AL="" FN4UM=5 FN4SS=1 FN4CH="" FN4XM=1 FN4SPHC=1 FN4X3C=1 FN4QSA="" FN4P2X="" FN4MP="" FN4X3GD=1 FN4X3DZ=1 PROFEVERY="" FN4MTPF=1
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
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 211: VIRHE temperature guard ($hot) -> blocks stopped"; return 0; }
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
  [ $ok = 1 ] || { log "RESULT fn4 211 restart-$tag: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
  log "RESULT fn4 211 restart-$tag: healthy in $(( $(date +%s) - t0 )) s; $(grep -m1 -E '^I kv: ' $S | cut -c3-160); fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300)"
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
    || { log "RESULT fn4 211 $1: VIRHE the switches are not $2 ($(echo "$g" | cut -c1-400))"; return 1; }
}
alive() {   # alive STEP: /health 200, else log it (the T arm then yields no answers)
  [ "$DRY" = 1 ] && { echo "+ curl -m 5 $URL/health == 200"; return 0; }
  local code; code=$(curl -s -m 5 -o /dev/null -w %{http_code} $URL/health)
  [ "$code" = 200 ] && return 0
  log "RESULT fn4 211 $1: VIRHE the server is not healthy (/health $code; $(docker inspect -f {{.State.Status}} exit {{.State.ExitCode}} $CN 2>/dev/null || echo gone))"; return 1
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
lc() { [ "$DRY" = 1 ] && { echo "+ cmp $1 $2"; return 0; }; log "RESULT fn4 211 text $3: $(timeout 300 python3 -I $LD cmp --url $URL --label "$3" $O/$TAG-$1.jsonl $O/$TAG-$2.jsonl 2>&1 | grep -m1 '^CMPLINE ' | cut -c9- | cut -c1-400)"; }
abr() {   # abr NAME A B (extra abblocks2 args...)
  local nm=$1 A=$2 B=$3; shift 3
  run "timeout 2400 python3 -I $TL/abblocks2.py --url $URL --mb-dir $D --label $TAG-$nm --rounds 1 --a '$A' --b '$B' --names a,b --out $O/$TAG-blocks.jsonl $* > $O/$TAG-$nm.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 211 $nm (A B B A): $(grep '^ABB ' $O/$TAG-$nm.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=4: "-" if v is None else format(v, ".%df" % n)
s = "decode tok/s B/A %s %s ms/step B/A %s (A %s | B %s)" % (f(d["tps_ratio"]), (d.get("sep") or {}).get("tps"), f(d.get("ms_ratio")), " ".join("%.1f" % x for x in d.get("A_blocks", [])), " ".join("%.1f" % x for x in d.get("B_blocks", []))) if d.get("rounds") else ""
for k, v in (d.get("pf") or {}).items():
    s += " | prefill %s B/A %s %s (A %s | B %s)" % (k, f(v["ratio"]), v.get("sep"), " ".join("%.0f" % x for x in v.get("A_blocks", [])), " ".join("%.0f" % x for x in v.get("B_blocks", [])))
print(s + " | %d blocks, %d flagged, %s s" % (d["blocks"], d["blocks_bad"], d["wall_s"]))' 2>&1 | cut -c1-700)"
  [ "$DRY" = 1 ] || log "RESULT fn4 211 $nm blocks (tok/s, ms/step, acceptance): $(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("label") == sys.argv[2]: print("%s %s tok/s %s ms/step acc %s;" % (r["arm"], r.get("tps"), r.get("ms"), (r.get("mb") or {}).get("acceptance")), end=" ")' $O/$TAG-blocks.jsonl $TAG-$nm 2>&1 | cut -c1-300)"
}
log "RESULT fn4 211-fn-bi7: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) exact wire + SPHC + MTPFLAT declared; T (traced byte check) then S (speed) | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 211: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 211: VIRHE image $IMG missing"; exit 1; }
  [ -e "$Y" ] && [ -n "$TL" ] && [ -e $TL/abblocks2.py ] && [ -e $MBT ] && [ -e $LD ] && [ -e $GL ] && [ -e $BTP ] && [ -n "$GS" ] && [ -e $D/mb.py ] || { log "RESULT fn4 211: VIRHE tools missing"; exit 1; }
  grep -q 'RADIANCE_FN4_MTPFLAT' $Y || { log "RESULT fn4 211: VIRHE $YN does not pass RADIANCE_FN4_MTPFLAT"; exit 1; }
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
[ "$DRY" != 1 ] && [ "${FREEG:-0}" -lt 12 ] && { log "RESULT fn4 211: VIRHE only ${FREEG:-?} GB free on / -- not started"; exit 1; }
S6='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 4, "hc_readv": 11}'
S7='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 4, "hc_readv": 75}'
S7L='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 5, "hc_readv": 75}'
OFF='{"exl3_bi": 0, "exl3_dual": 0, "hc_bi": 0, "bi_seg": 0, "att_bi": 0, "arw_rows": 0, "sched_bi_chunk": 0, "hc_readv": 11}'
LD=$TL/fn-loadid.py; GL=$TL/greedylong.py
gs() { pcr; run "(cd $D && timeout 1500 python3 $GS $URL $O/$TAG-$2.jsonl --conc $1 --max-tokens 192 > $O/$TAG-$2.txt 2>&1)"; }
lc1() { [ "$DRY" = 1 ] && { echo "+ cmp $1"; return 0; }; log "RESULT fn4 211 text $1: $(timeout 300 python3 -I $LD cmp --url $URL --label "$1" $O/$TAG-a1.jsonl $O/$TAG-$1.jsonl 2>&1 | grep -m1 '^CMPLINE ' | cut -c9- | cut -c1-500)"; }
gl() { pcr; run "(cd $D && timeout 1800 python3 $GL $URL $O/$TAG-L$2.jsonl --conc $1 --max-tokens 128 > $O/$TAG-L$2.txt 2>&1)"; }
export FN4BT="" FN4BTP="" FN4BTL=""
restart $TAG-S || exit 1
nudge
sw "$S7" "S7" || { down; exit 1; }
gs 1 a1; alive "a1"
gs 4 c4; alive "c4"
pcr; run "(cd $D && timeout 3600 python3 -I $LD run --url $URL --gs $GS --mb $D/mb.py --out $O/$TAG-pf.jsonl --conc 4 --warm 10 --offset 0.5 --pf-tok 16384 --max-tokens 192 > $O/$TAG-pf.txt 2>&1)"; alive "pf"
gs 16 c16; alive "c16"
gs 1 a2; alive "a2"
lc1 c4; lc1 pf; lc1 c16; lc1 a2
sw "$S7L" "S7L" || { down; exit 1; }
gl 1 a1; alive "La1"
gl 4 c4; alive "Lc4"
pcr; run "(cd $D && timeout 3600 python3 -I $LD run --url $URL --gs $GL --mb $D/mb.py --out $O/$TAG-Lpf.jsonl --conc 4 --warm 10 --offset 0.5 --pf-tok 16384 --max-tokens 128 --timeout 900 > $O/$TAG-Lpf.txt 2>&1)"; alive "Lpf"
for x in c4 pf; do [ "$DRY" = 1 ] || log "RESULT fn4 211 text LONG L$x vs La1 (sched_bi_chunk 5): $(timeout 300 python3 -I $LD cmp --url $URL --label "L$x" $O/$TAG-La1.jsonl $O/$TAG-L$x.jsonl 2>&1 | grep -m1 '^CMPLINE ' | cut -c9- | cut -c1-500)"; done
abr g16c16 "$S6" "$S7" --mbt2 $MBT --gens 1 --phase c16 --warm 8; alive "S c16 G16"
abr s7c4 "$OFF" "$S7" --mbt2 $MBT --gens 1 --phase c4 --warm 8; alive "S c4 OFF vs S7"
abr s7c8 "$OFF" "$S7" --mbt2 $MBT --gens 1 --phase c8 --warm 8; alive "S c8 OFF vs S7"
abr s7c16 "$OFF" "$S7" --mbt2 $MBT --gens 1 --phase c16 --warm 8; alive "S c16 OFF vs S7"
run "docker logs $CN > $O/$TAG-S-engine.log 2>&1"
[ "$DRY" = 1 ] || log "RESULT fn4 211 engine: $(grep -cE 'fn4: switches .* applied' $O/$TAG-S-engine.log) applied changes, $(grep -cE 'fn4: .* refused' $O/$TAG-S-engine.log) refused; E lines $(grep -cE '^E ' $O/$TAG-S-engine.log)"
down
# ---- START 2: --profile-ops every 4 steps, mb c16 windows
OPW=$TL/opwin.py
lines() { [ "$DRY" = 1 ] && { echo 0; return 0; }; docker logs $CN 2>&1 | wc -l; }
WINS=""
pwin() {   # pwin LABEL SWITCHES
  sw "$2" "$1" || return 1
  run "(cd $D && timeout 300 python3 mb.py --tag $TAG-p$1 --phases c16 --dur 6 --warm 4 > /dev/null 2>&1)"
  local lo hi; lo=$(lines)
  run "(cd $D && timeout 400 python3 mb.py --tag $TAG-p$1 --phases c16 --dur 14 --warm 2 > $O/$TAG-p$1-c16.txt 2>&1)"
  hi=$(lines); WINS="$WINS --win $1:$lo:$hi"
  [ "$DRY" = 1 ] || log "RESULT fn4 211 profile window $1 ($2): engine-log lines $lo..$hi"
}
# disk (coordinator 12.11.: gpu02 / reached 100 % at 17:09): the profiled start only with >= 12 GB free; logs gzipped after
FREEG=$(df --output=avail -BG / 2>/dev/null | tail -1 | tr -dc 0-9)
if [ "$DRY" != 1 ] && [ "${FREEG:-0}" -lt 12 ]; then
  log "RESULT fn4 211: VIRHE only ${FREEG:-?} GB free on / -- the profiled start (START 2) skipped"
else
export PROFEVERY=4 EXTRA="$EXTRA0 --tp-wire exact --profile-ops"
restart $TAG-P || exit 1
nudge
S7H0='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 0, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 4, "hc_readv": 75}'
S7A0='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 0, "arw_rows": 16, "sched_bi_chunk": 4, "hc_readv": 75}'
pwin S7 "$S7"; alive "profile S7"
pwin H0 "$S7H0"; alive "profile H0"
pwin A0 "$S7A0"; alive "profile A0"
pwin OF "$OFF"; alive "profile OFF"
run "docker logs $CN > $O/$TAG-P-engine.log 2>&1"
down
if [ "$DRY" != 1 ]; then
  timeout 900 python3 -I $OPW $O/$TAG-P-engine.log $WINS --json $O/$TAG-opwin.json > $O/$TAG-opwin.txt 2>&1
  grep '^OPWIN ' $O/$TAG-opwin.txt | while IFS= read -r l; do log "RESULT fn4 211 $(echo "$l" | cut -c1-1500)"; done
else
  echo "+ opwin.py $O/$TAG-P-engine.log --win S7 / H0 / A0 / OFF"
fi
fi
run "gzip -f $O/$TAG-P-engine.log $O/start-$TAG-P.log 2>/dev/null; true"
run "gzip -f $O/$TAG-S-engine.log $O/start-$TAG-S.log 2>/dev/null; true"
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
log "RESULT fn4 211-fn-bi7: done"
