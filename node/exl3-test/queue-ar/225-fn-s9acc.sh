#!/bin/bash
# 225-fn-s9acc.sh (queue item, 13.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~40 min) -- S9 ACCEPTANCE (coordinator
# 13.11.: S9 = S8 + att_bi 3; 223: full-row step c16 -4.9 % / c8 -3.5 % / c4 -2.0 %, mb c16 1030.5 with full BI) on radiance-local:c16-x3sp21
# (fn4-x3-sp3 a21ed9a1: x3sp20 + the tracer's 'blkp' / 'moep' for the block output before a reducing write; 224) with fn-tp4zu.yaml:
#   START T (traced, PMAX 256 / PLEN 128): the short set a1 / c4 / pf (each prompt under mb c4 + a fresh ~16k prefill at +0.5 s: the staggered
#     arrival) / c16 / a2 under S9 -> fn-bitrace.py analyze --labels c4,pf,c16 (+ --census) and texts vs a1.
#   START S (untraced, recorded passes): the short set under S9, then the LONG set at sched_bi_chunk 5 (La1 / Lc4 / Lpf: att_bi 3 changes the
#     multi-sequence decode path that broke 205) under S9L; texts vs a1 / La1.
# DRYRUN=1 bash 225-fn-s9acc.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4zu.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/heatsoak.sh" ] && { TL=$f; break; }; done
IMG=${X225_IMAGE:-radiance-local:c16-x3sp21}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-s9
export FNIMAGE=$IMG FNMODELDIR=${X225_X3_DIR:-$HOME/models/flashnext-exl3} FNMODEL=${X225_X3_MODEL:-qwen3.8-flash-next-exl3.rad}
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
EXTRA0="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0"
export EXTRA="$EXTRA0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats225
export FN4BT="" FN4BTP="" FN4BTL="" FN4X3DUAL=1 FN4NR=1 FN4GI=1 FN4FD="" FN4NW=1 FN4SW="" FN4WV=1 FN4RV=11 FN4AL="" FN4UM=5 FN4SS=1 FN4CH="" FN4XM=1 FN4SPHC=1 FN4X3C=1 FN4QSA="" FN4P2X="" FN4MP="" FN4X3GD=1 FN4X3DZ=1 PROFEVERY="" FN4MTPF=1 LOGSTEPS=1 FN4FC=1
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
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 225: VIRHE temperature guard ($hot) -> blocks stopped"; return 0; }
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
  [ $ok = 1 ] || { log "RESULT fn4 225 restart-$tag: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
  log "RESULT fn4 225 restart-$tag: healthy in $(( $(date +%s) - t0 )) s; $(grep -m1 -E '^I kv: ' $S | cut -c3-160); fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300)"
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
    || { log "RESULT fn4 225 $1: VIRHE the switches are not $2 ($(echo "$g" | cut -c1-400))"; return 1; }
}
alive() {   # alive STEP: /health 200, else log it (the T arm then yields no answers)
  [ "$DRY" = 1 ] && { echo "+ curl -m 5 $URL/health == 200"; return 0; }
  local code; code=$(curl -s -m 5 -o /dev/null -w %{http_code} $URL/health)
  [ "$code" = 200 ] && return 0
  log "RESULT fn4 225 $1: VIRHE the server is not healthy (/health $code; $(docker inspect -f {{.State.Status}} exit {{.State.ExitCode}} $CN 2>/dev/null || echo gone))"; return 1
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
lc() { [ "$DRY" = 1 ] && { echo "+ cmp $1 $2"; return 0; }; log "RESULT fn4 225 text $3: $(timeout 300 python3 -I $LD cmp --url $URL --label "$3" $O/$TAG-$1.jsonl $O/$TAG-$2.jsonl 2>&1 | grep -m1 '^CMPLINE ' | cut -c9- | cut -c1-400)"; }
abr() {   # abr NAME A B (extra abblocks2 args...)
  local nm=$1 A=$2 B=$3; shift 3
  run "timeout 2400 python3 -I $TL/abblocks2.py --url $URL --mb-dir $D --label $TAG-$nm --rounds 1 --a '$A' --b '$B' --names a,b --out $O/$TAG-blocks.jsonl $* > $O/$TAG-$nm.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 225 $nm (A B B A): $(grep '^ABB ' $O/$TAG-$nm.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=4: "-" if v is None else format(v, ".%df" % n)
s = "decode tok/s B/A %s %s ms/step B/A %s (A %s | B %s)" % (f(d["tps_ratio"]), (d.get("sep") or {}).get("tps"), f(d.get("ms_ratio")), " ".join("%.1f" % x for x in d.get("A_blocks", [])), " ".join("%.1f" % x for x in d.get("B_blocks", []))) if d.get("rounds") else ""
for k, v in (d.get("pf") or {}).items():
    s += " | prefill %s B/A %s %s (A %s | B %s)" % (k, f(v["ratio"]), v.get("sep"), " ".join("%.0f" % x for x in v.get("A_blocks", [])), " ".join("%.0f" % x for x in v.get("B_blocks", [])))
print(s + " | %d blocks, %d flagged, %s s" % (d["blocks"], d["blocks_bad"], d["wall_s"]))' 2>&1 | cut -c1-700)"
  [ "$DRY" = 1 ] || log "RESULT fn4 225 $nm blocks (tok/s, ms/step, acceptance): $(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("label") == sys.argv[2]: print("%s %s tok/s %s ms/step acc %s;" % (r["arm"], r.get("tps"), r.get("ms"), (r.get("mb") or {}).get("acceptance")), end=" ")' $O/$TAG-blocks.jsonl $TAG-$nm 2>&1 | cut -c1-300)"
}
log "RESULT fn4 225-fn-s9acc: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) exact wire + SPHC + MTPFLAT declared + step log + form counter; S9 = S8 + att_bi 3 acceptance: T traced short set, S untraced short + LONG | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 225: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 225: VIRHE image $IMG missing"; exit 1; }
  [ -e "$Y" ] && [ -n "$TL" ] && [ -e $TL/abblocks2.py ] && [ -e $MBT ] && [ -e $LD ] && [ -e $GL ] && [ -e $BTP ] && [ -n "$GS" ] && [ -e $D/mb.py ] && [ -e $TL/fnsteps.py ] && [ -e $BTP ] && [ -e $GL ] || { log "RESULT fn4 225: VIRHE tools missing"; exit 1; }
  grep -q 'RADIANCE_FN4_FORMCTR' $Y && grep -q 'RADIANCE_LOG_STEPS' $Y && grep -q 'RADIANCE_FN4_MTPFLAT' $Y || { log "RESULT fn4 225: VIRHE $YN does not pass RADIANCE_FN4_MTPFLAT / _FORMCTR / RADIANCE_LOG_STEPS"; exit 1; }
  mkdir -p $O $STATSDIR/bt-s8 && chmod 777 $STATSDIR/bt-s8 && find $STATSDIR/bt-s8 -maxdepth 1 -name 'bt-r*' -delete
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
[ "$DRY" != 1 ] && [ "${FREEG:-0}" -lt 12 ] && { log "RESULT fn4 225: VIRHE only ${FREEG:-?} GB free on / -- not started"; exit 1; }
fcl() {   # fcl WHAT: the newest engagement-counter line a device (cumulative) and the libr4d hc_readv it read
  [ "$DRY" = 1 ] && { echo "+ docker logs $CN | grep 'fn4 formctr dev' (newest a device)"; return 0; }
  log "RESULT fn4 225 forms $1: $(docker logs $CN 2>&1 | grep 'fn4 formctr dev' | awk '{d[$4] = $0} END {for (k in d) print d[k]}' | sort | sed 's/^fn4 formctr //' | tr '\n' '|' | cut -c1-900)"
}
S9='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 3, "arw_rows": 16, "sched_bi_chunk": 4, "hc_readv": 331}'
S9L='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 3, "arw_rows": 16, "sched_bi_chunk": 5, "hc_readv": 331}'
OFF='{"exl3_bi": 0, "exl3_dual": 0, "hc_bi": 0, "bi_seg": 0, "att_bi": 0, "arw_rows": 0, "sched_bi_chunk": 0, "hc_readv": 11}'
gs() { pcr; run "(cd $D && timeout 1500 python3 $GS $URL $O/$TAG-$2.jsonl --conc $1 --max-tokens 192 > $O/$TAG-$2.txt 2>&1)"; }
gl() { pcr; run "(cd $D && timeout 1800 python3 $GL $URL $O/$TAG-L$2.jsonl --conc $1 --max-tokens 128 > $O/$TAG-L$2.txt 2>&1)"; }
pfs() { pcr; run "(cd $D && timeout 3600 python3 -I $LD run --url $URL --gs $GS --mb $D/mb.py --out $O/$TAG-$1.jsonl --conc 4 --warm 10 --offset 0.5 --pf-tok 16384 --max-tokens 192 > $O/$TAG-$1.txt 2>&1)"; }
cmpx() {   # cmpx LABEL A B
  [ "$DRY" = 1 ] && { echo "+ cmp $2 $3 ($1)"; return 0; }
  log "RESULT fn4 225 text $1: $(timeout 300 python3 -I $LD cmp --url $URL --label "$1" $O/$TAG-$2.jsonl $O/$TAG-$3.jsonl 2>&1 | grep -m1 '^CMPLINE ' | cut -c9- | cut -c1-500)"
}
# ---------------- START T: traced short set
export FN4BT=/stats/bt-s8 FN4BTP=256 FN4BTL=128
restart $TAG-T || exit 1
nudge
[ "$DRY" = 1 ] || log "RESULT fn4 225 T engine notes: $(grep -E 'tracer ON|chained launches:' $O/start-$TAG-T.log | sed 's/^[A-Z] //' | cut -c1-180 | tr '\n' ' ' | cut -c1-400)"
sw "$S9" "T S9" || { down; exit 1; }
gs 1 Ta1; alive "T a1"
gs 4 Tc4; alive "T c4"
pfs Tpf; alive "T pf"
gs 16 Tc16; alive "T c16"
gs 1 Ta2; alive "T a2"
cmpx "T c4 vs a1" Ta1 Tc4; cmpx "T pf vs a1 (staggered)" Ta1 Tpf; cmpx "T c16 vs a1" Ta1 Tc16; cmpx "T a2 vs a1 (control)" Ta1 Ta2
fcl "T end"
run "docker logs $CN 2>&1 | gzip -1 > $O/$TAG-T-engine.log.gz"
down
if [ "$DRY" = 1 ]; then echo "+ python3 -I $BTP analyze $STATSDIR/bt-s8 --labels c4,pf,c16 --json $O/$TAG-bitrace.json > $O/$TAG-bitrace.txt"
else
  log "RESULT fn4 225 T tracer files: $(ls $STATSDIR/bt-s8/bt-r*.bin 2>/dev/null | wc -l) files, $(du -sm $STATSDIR/bt-s8 | cut -f1) MB"
  timeout 2400 python3 -I $BTP analyze $STATSDIR/bt-s8 --labels c4,pf,c16 --json $O/$TAG-bitrace.json > $O/$TAG-bitrace.txt 2>&1
  grep -E '^BITRACE summary' $O/$TAG-bitrace.txt | while IFS= read -r l; do log "RESULT fn4 225 T S9 traced $(echo "$l" | cut -c1-900)"; done
  timeout 2400 python3 -I $BTP analyze $STATSDIR/bt-s8 --labels c4,pf,c16 --census --json $O/$TAG-bitrace-census.json > $O/$TAG-bitrace-census.txt 2>&1
  grep -E '^BITRACE census' $O/$TAG-bitrace-census.txt | while IFS= read -r l; do log "RESULT fn4 225 T S9 traced $(echo "$l" | cut -c1-900)"; done
  grep -q '^BITRACE summary' $O/$TAG-bitrace.txt && grep -q '^BITRACE census' $O/$TAG-bitrace-census.txt && find $STATSDIR/bt-s8 -maxdepth 1 -name 'bt-r*.bin' -delete
  grep -q '^BITRACE summary' $O/$TAG-bitrace.txt || log "RESULT fn4 225 T bitrace: VIRHE no summary ($(tail -2 $O/$TAG-bitrace.txt | tr '\n' ' ' | cut -c1-300))"
fi
# ---------------- START S: untraced (recorded passes), short + long acceptance
export FN4BT="" FN4BTP="" FN4BTL=""
restart $TAG-S || exit 1
if [ "$DRY" = 1 ]; then echo "+ docker logs -f -t $CN | gzip -1 > $O/$TAG-S-engine.log.gz &"; CAPPID=""
else (docker logs -f -t $CN 2>&1 | gzip -1 > $O/$TAG-S-engine.log.gz) & CAPPID=$!; fi
nudge
sw "$S9" "S S9" || { down; exit 1; }
gs 1 a1; alive "a1"
gs 4 c4; alive "c4"
pfs pf; alive "pf"
gs 16 c16; alive "c16"
gs 1 a2; alive "a2"
cmpx "c4 vs a1" a1 c4; cmpx "pf vs a1 (staggered)" a1 pf; cmpx "c16 vs a1" a1 c16; cmpx "a2 vs a1 (control)" a1 a2
sw "$S9L" "S S9L" || { down; exit 1; }
gl 1 a1; alive "La1"
gl 4 c4; alive "Lc4"
pcr; run "(cd $D && timeout 3600 python3 -I $LD run --url $URL --gs $GL --mb $D/mb.py --out $O/$TAG-Lpf.jsonl --conc 4 --warm 10 --offset 0.5 --pf-tok 16384 --max-tokens 128 --timeout 900 > $O/$TAG-Lpf.txt 2>&1)"; alive "Lpf"
cmpx "LONG Lc4 vs La1 (sched_bi_chunk 5)" La1 Lc4; cmpx "LONG Lpf vs La1 (staggered, sched_bi_chunk 5)" La1 Lpf
fcl "S after acceptance"
sw "$S9" "S9 end"
down
[ -n "$CAPPID" ] && { for i in $(seq 1 60); do kill -0 $CAPPID 2>/dev/null || break; sleep 2; done; kill $CAPPID 2>/dev/null; }
[ "$DRY" = 1 ] || log "RESULT fn4 225 engine S: $(zcat $O/$TAG-S-engine.log.gz | grep -cE 'fn4: switches .* applied') applied changes, $(zcat $O/$TAG-S-engine.log.gz | grep -cE 'fn4: .* refused') refused; E lines $(zcat $O/$TAG-S-engine.log.gz | grep -cE ' E ')"
run "gzip -f $O/start-$TAG-T.log $O/start-$TAG-S.log 2>/dev/null; true"
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
log "RESULT fn4 225-fn-s9acc: done"
