#!/bin/bash
# 205-fn-bi6.sh (queue item, 12.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~30 min) -- THE NEW
# EXACT-UNDER-LOAD BASE S6 = 200's S5 with att_bi 1 (200: att_bi 2 -0.9 %, kept at 1) and arw_rows 16 (200: +4.9 %), on
# radiance-local:c16-x3sp6 (fn4-x3-sp3 808866c2 = fn4-x3 7bb0a401 + SP-hc / hc_sp 2 / RSF, all off here; SPHC declared).
# START 1 (unprofiled): BI acceptance as 200 -- greedy-20 a1 / c4 / pf / c16 / a2 and the LONG set at sched_bi_chunk 5
# (La1 / Lc4 / Lpf); then c16 mbt2 A B B A: OFF vs S6 (the remaining total). START 2 (--profile-ops every 4 steps): mb c16
# windows under S6, S6 with hc_bi 0 and S6 with att_bi 0 -- tools/opwin.py: hc_read (the bi_dec twin, the final read a
# sequence a launch, the tiled form) and the attention ops' calls / us / ms a step (hc_bi 5's remaining 8 %, att_bi's 3.6 %).
# DRYRUN=1 bash 205-fn-bi6.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4zt.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/heatsoak.sh" ] && { TL=$f; break; }; done
IMG=${X205_IMAGE:-radiance-local:c16-x3sp6}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-bi6
export FNIMAGE=$IMG FNMODELDIR=${X205_X3_DIR:-$HOME/models/flashnext-exl3} FNMODEL=${X205_X3_MODEL:-qwen3.8-flash-next-exl3.rad}
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
EXTRA0="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0"
export EXTRA="$EXTRA0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats205
export FN4BT="" FN4BTP="" FN4BTL="" FN4X3DUAL=1 FN4NR=1 FN4GI=1 FN4FD="" FN4NW=1 FN4SW="" FN4WV=1 FN4RV=11 FN4AL="" FN4UM=5 FN4SS=1 FN4CH="" FN4XM=1 FN4SPHC=1 FN4X3C=1 FN4QSA="" FN4P2X="" FN4MP="" FN4X3GD=1 FN4X3DZ=1 PROFEVERY=""
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
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 205: VIRHE temperature guard ($hot) -> blocks stopped"; return 0; }
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
  [ $ok = 1 ] || { log "RESULT fn4 205 restart-$tag: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
  log "RESULT fn4 205 restart-$tag: healthy in $(( $(date +%s) - t0 )) s; $(grep -m1 -E '^I kv: ' $S | cut -c3-160); fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300)"
}
probe8() {   # probe8 TAG -> "ms tok/s" of a 45 s mb c8
  [ "$DRY" = 1 ] && { echo "+ (cd $D && timeout 400 python3 mb.py --tag $1 --phases c8 --dur 45 --warm 10 > $O/$1-c8.txt)" >&2; echo "25.6 744"; return 0; }
  (cd $D && timeout 400 python3 mb.py --tag $1 --phases c8 --dur 45 --warm 10 > $O/$1-c8.txt 2>&1)
  grep '^MB ' $O/$1-c8.txt 2>/dev/null | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[3:]); print(d.get("ms_decode_step"), d.get("agg_dec_tps"))' 2>/dev/null || echo "0 0"
}
setsw() {   # setsw JSON: queue switches (they land at the next step release)
  run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"
}
GS=; for f in "$H/tools/greedyset.py" "$H/../tools/greedyset.py" "$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py"; do [ -z "$GS" ] && [ -e "$f" ] && GS=$f; done
alive() {   # alive WHAT: the server still answers, else logs + down + exit
  [ "$DRY" = 1 ] && return 0
  local code; code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' $URL/health)
  [ "$code" = 200 ] && return 0
  docker logs $CN > $O/$TAG-dead-engine.log 2>&1
  log "RESULT fn4 205: VIRHE the server is not healthy after $1 (/health $code; last E lines: $(grep -E '^E |trap|abort|HSA_STATUS' $O/$TAG-dead-engine.log | tail -3 | tr '\n' ' ' | cut -c1-400)) -> logs saved, down"
  down; exit 1
}
LD=$TL/fn-loadid.py
PFD=${X205_PFD:-16384,65536}
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
swcheck() {   # swcheck WHAT SWITCHES_JSON: every switch named there has that value now
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4: $2"; return 0; }
  local g; g=$(curl -s -m 10 $URL/admin/fn4)
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); w=json.loads(sys.argv[2]); sys.exit(0 if all(d["switches"].get(k) == v for k, v in w.items()) else 1)' "$g" "$2" 2>/dev/null \
    || { log "RESULT fn4 205 $1: VIRHE the switches are not $2 ($(echo "$g" | cut -c1-400))"; return 1; }
}
pcreset() { [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o /dev/null -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  [ "$c" = 200 ] || log "RESULT fn4 205 $1 prefix cache reset: HTTP $c"; }
c16blk() { [ -e $O/$TAG.abort ] && return 0; run "(cd $D && timeout 400 python3 mb.py --tag $TAG-$1 --phases c16 --dur 20 --warm 6 > $O/$TAG-$1-c16.txt 2>&1)"; }
pfblk() { [ -e $O/$TAG.abort ] && return 0; run "(cd $D && timeout 600 python3 ds41_tests.py prefill --depths $PFD --cpt 3.6 --calibrate 0 --reps 1 > $O/$TAG-$1-pf.txt 2>&1)"; }
gset() { run "(cd $D && timeout 900 python3 $GS $URL $O/$TAG-gs-$1.jsonl --conc 1 --max-tokens 192 > $O/$TAG-gs-$1.txt 2>&1)"; }
copyt() {   # copyt NAME: a FRESH 33k copy (the prefix cache reset first), greedy; output $O/copy-$TAG-NAME-run1.txt
  pcreset copy-$1
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset 0 --tag=-$TAG-$1 --outdir $O > $O/$TAG-$1-copy.txt 2>&1)"
}
cmpf() {   # cmpf FILE_A FILE_B LABEL
  if [ "$DRY" = 1 ]; then echo "+ cmp $1 $2"
  elif [ -s $1 ] && [ -s $2 ]; then
    if cmp -s $1 $2; then log "RESULT fn4 205 identity $3: IDENTICAL ($(wc -c < $2) bytes)"
    else log "RESULT fn4 205 identity $3: DIFFERENT from byte $(cmp $1 $2 2>&1 | grep -oE 'byte [0-9]+' | head -1)"; fi
  else log "RESULT fn4 205 identity $3: not compared (missing $1 or $2)"; fi
}
mpair() {   # mpair N: moe_pair N, landed (an epoch bump, a nudge, checked)
  setsw "{\"moe_pair\": $1, \"epoch_bump\": 1}"; nudge; swcheck "moe_pair $1" "{\"moe_pair\": $1}"
}
abpf() {   # abpf NAME A_JSON B_JSON: prefill-only blocks A B B A (one round = 2 blocks a side)
  [ -e $O/$TAG.abort ] && return 0
  run "timeout 1800 python3 -I $TL/abblocks2.py --url $URL --mb-dir $D --label $TAG-$1 --rounds 1 --a '$2' --b '$3' --pf-depths $PFD --no-mb --names off,$1 --out $O/$TAG-blocks.jsonl > $O/$TAG-$1.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 205 $1 prefill (B/A of prefill tok/s, A B B A, 2 blocks a side): $(grep '^ABB ' $O/$TAG-$1.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
for k, v in (d.get("pf") or {}).items():
    print("%s B/A %.4f %s (A %s | B %s tok/s);" % (k, v["ratio"] or 0, v.get("sep"), " ".join("%.0f" % x for x in v.get("A_blocks", [])), " ".join("%.0f" % x for x in v.get("B_blocks", []))), end=" ")
print("%d blocks, %d flagged, %s s" % (d["blocks"], d["blocks_bad"], d["wall_s"]))' 2>&1 | cut -c1-700)"
}
GS=; for f in "$H/tools/greedyset.py" "$H/../tools/greedyset.py" "$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py"; do [ -z "$GS" ] && [ -e "$f" ] && GS=$f; done
LD=$TL/fn-loadid.py; MBT=$TL/fnharness/mbt2.py
S6='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 4}'
S6L='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 5}'
S6H0='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 0, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 4}'
S6A0='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 0, "arw_rows": 16, "sched_bi_chunk": 4}'
OFF='{"exl3_bi": 0, "exl3_dual": 0, "hc_bi": 0, "bi_seg": 0, "att_bi": 0, "arw_rows": 0, "sched_bi_chunk": 0}'
pcr() { run "curl -s -m 60 -o /dev/null -X POST -d '' $URL/reset_prefix_cache"; }
gs() { pcr; run "(cd $D && timeout 1500 python3 $GS $URL $O/$TAG-$2.jsonl --conc $1 --max-tokens 192 > $O/$TAG-$2.txt 2>&1)"; }
lc() { [ "$DRY" = 1 ] && { echo "+ cmp $1"; return 0; }; log "RESULT fn4 205 text $1: $(timeout 300 python3 -I $LD cmp --url $URL --label "$1" $O/$TAG-a1.jsonl $O/$TAG-$1.jsonl 2>&1 | grep -m1 '^CMPLINE ' | cut -c9- | cut -c1-500)"; }
GL=$TL/greedylong.py
gl() { pcr; run "(cd $D && timeout 1800 python3 $GL $URL $O/$TAG-L$2.jsonl --conc $1 --max-tokens 128 > $O/$TAG-L$2.txt 2>&1)"; }
sw() { setsw "$(echo "$1" | sed 's/}$/, "epoch_bump": 1}/')"; nudge; swcheck "$2" "$1"; }
abc() {   # abc NAME PHASE A B
  run "timeout 2400 python3 -I $TL/abblocks2.py --url $URL --mb-dir $D --mbt2 $MBT --gens 1 --label $TAG-$1 --phase $2 --warm 8 --rounds 1 --a '$3' --b '$4' --names a,b --out $O/$TAG-blocks.jsonl > $O/$TAG-$1.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 205 $1 ($2, mbt2 fixed schedule, A B B A): $(grep '^ABB ' $O/$TAG-$1.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=4: "-" if v is None else format(v, ".%df" % n)
print("tok/s B/A %s %s | ms/step B/A %s | A blocks %s | B blocks %s | %d blocks, %d flagged, %s s" % (f(d["tps_ratio"]), (d.get("sep") or {}).get("tps"), f(d["ms_ratio"]),
    " ".join("%.1f" % x for x in d.get("A_blocks", [])), " ".join("%.1f" % x for x in d.get("B_blocks", [])), d["blocks"], d["blocks_bad"], d["wall_s"]))' 2>&1 | cut -c1-600)"
}
log "RESULT fn4 205-fn-bi6: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) exact wire; S6 = $S6 | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 205: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 205: VIRHE image $IMG missing"; exit 1; }
  [ -e "$Y" ] && [ -n "$TL" ] && [ -e $TL/abblocks2.py ] && [ -e $MBT ] && [ -e $LD ] && [ -n "$GS" ] && [ -e $GL ] && [ -e $D/mb.py ] || { log "RESULT fn4 205: VIRHE tools missing"; exit 1; }
  mkdir -p $O $STATSDIR
  find $O -maxdepth 1 -name "$TAG.abort" -delete; find $O -maxdepth 1 -name "$TAG.tstop" -delete
  find $O -maxdepth 1 \( -name "$TAG-*.jsonl" -o -name "$TAG-*.txt" \) -delete
  trap 'touch $O/$TAG.tstop 2>/dev/null; down' EXIT
  trap 'exit 143' TERM INT
  tguard & TGPID=$!
else
  echo "+ guards; tguard &"; TGPID=""
fi
export EXTRA="$EXTRA0 --tp-wire exact"
restart $TAG || exit 1
nudge
# (1) BI acceptance under S6
sw "$S6" "S6" || { down; exit 1; }
gs 1 a1; alive "a1"
gs 4 c4; alive "c4"
pcr; run "(cd $D && timeout 3600 python3 -I $LD run --url $URL --gs $GS --mb $D/mb.py --out $O/$TAG-pf.jsonl --conc 4 --warm 10 --offset 0.5 --pf-tok 16384 --max-tokens 192 > $O/$TAG-pf.txt 2>&1)"; alive "pf"
gs 16 c16; alive "c16"
gs 1 a2; alive "a2"
lc c4; lc pf; lc c16; lc a2
sw "$S6L" "S6L" || { down; exit 1; }
gl 1 a1; alive "La1"
gl 4 c4; alive "Lc4"
pcr; run "(cd $D && timeout 3600 python3 -I $LD run --url $URL --gs $GL --mb $D/mb.py --out $O/$TAG-Lpf.jsonl --conc 4 --warm 10 --offset 0.5 --pf-tok 16384 --max-tokens 128 --timeout 900 > $O/$TAG-Lpf.txt 2>&1)"; alive "Lpf"
for x in c4 pf; do [ "$DRY" = 1 ] || log "RESULT fn4 205 text LONG L$x vs La1 (sched_bi_chunk 5): $(timeout 300 python3 -I $LD cmp --url $URL --label "L$x" $O/$TAG-La1.jsonl $O/$TAG-L$x.jsonl 2>&1 | grep -m1 '^CMPLINE ' | cut -c9- | cut -c1-500)"; done
# (2) the remaining total at c16
abc total c16 "$OFF" "$S6"; alive "the total blocks"
run "docker logs $CN > $O/$TAG-engine.log 2>&1"
[ "$DRY" = 1 ] || log "RESULT fn4 205 engine: $(grep -cE 'fn4: switches .* applied' $O/$TAG-engine.log) applied changes, $(grep -cE 'fn4: .* refused' $O/$TAG-engine.log) refused; E lines $(grep -cE '^E ' $O/$TAG-engine.log)"
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
  [ "$DRY" = 1 ] || log "RESULT fn4 205 profile window $1 ($2): engine-log lines $lo..$hi"
}
# disk (coordinator 12.11.: gpu02 / reached 100 % at 17:09): the profiled start only with >= 12 GB free; logs gzipped after
FREEG=$(df --output=avail -BG / 2>/dev/null | tail -1 | tr -dc 0-9)
if [ "$DRY" != 1 ] && [ "${FREEG:-0}" -lt 12 ]; then
  log "RESULT fn4 205: VIRHE only ${FREEG:-?} GB free on / -- the profiled start (START 2) skipped"
else
export PROFEVERY=4 EXTRA="$EXTRA0 --tp-wire exact --profile-ops"
restart $TAG-P || exit 1
nudge
pwin S6 "$S6"; alive "profile S6"
pwin H0 "$S6H0"; alive "profile H0"
pwin A0 "$S6A0"; alive "profile A0"
run "docker logs $CN > $O/$TAG-P-engine.log 2>&1"
down
if [ "$DRY" != 1 ]; then
  timeout 900 python3 -I $OPW $O/$TAG-P-engine.log $WINS --json $O/$TAG-opwin.json > $O/$TAG-opwin.txt 2>&1
  grep '^OPWIN ' $O/$TAG-opwin.txt | while IFS= read -r l; do log "RESULT fn4 205 $(echo "$l" | cut -c1-1500)"; done
else
  echo "+ opwin.py $O/$TAG-P-engine.log --win S6 / H0 / A0"
fi
fi
run "gzip -f $O/$TAG-P-engine.log $O/$TAG-engine.log $O/start-$TAG.log $O/start-$TAG-P.log 2>/dev/null; true"
[ "$DRY" = 1 ] || log "RESULT fn4 205 logs gzipped; $(df --output=avail -BG / | tail -1 | tr -d ' ') free on /"
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
log "RESULT fn4 205-fn-bi6: done"
