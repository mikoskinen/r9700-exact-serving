#!/bin/bash
# 195-fn-dp2.sh (queue item, 12.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~15 min) -- QUICK EXL3
# DP2 SPEED READING (user 12.11.: its level vs TP4): two TP2 servers side by side, radiance-local:c16-x3bs at fnsmoke2's /
# wirecost2 arm A's config (X3CODES + exl3_gd 1, map 1 + dz 1, RADIANCE_FN4_X3DUAL=1 declared, BI switches 0, the line's
# wire wht6 min-kb 128, all_vram, MML 262144, MTP 3, MNBT 4096, span 2), compose fn-dp2x3.yaml (fn-tp4zt.yaml with PROJ / CN
# / PORT): server A = chips dbba5c85 + 27d49203 (03:00 + 48:00 after the riser move) on :8090, server B = chips 5943fea8 +
# 4d61f52c (83:00 + c3:00) on :8092. TP2 FLAGS THAT DIFFER from TP4: --tp 2 and MAXSEQS 8 a server (16 users = 8 + 8;
# X195_MAXSEQS). Readings (tools/mbu.py = mb.py with MB_URL), one block each, 20 s windows, warm 6:
#   1. mb c8 on BOTH servers at once = 16 users: the sum of agg decode tok/s (+ each server's ms/step)
#   2. mb c4 on both at once = 8 users
#   3. mb c1 on server A alone (B idle) = 1 user
#   4. server A: an 8k warm-up prefill (discarded, the start's first long prefill), then ONE fresh 64k prefill: TTFT, tok/s
# A start that fails is logged and the item finishes with what it has. No BI work. Power untouched. NO cc_sim.
# TP4 reference (today, no re-runs): wirecost2 A (191, c16-x3bs same config): c16 1009.8 / 994.8 tok/s (ms/step 34.5 / 34.3),
# 64k 8823 / 8799 tok/s.
# DRYRUN=1 bash 195-fn-dp2.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-dp2x3.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/heatsoak.sh" ] && { TL=$f; break; }; done
IMG=${X195_IMAGE:-radiance-local:c16-x3bs}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-dp2
export FNIMAGE=$IMG FNMODELDIR=${X195_X3_DIR:-$HOME/models/flashnext-exl3} FNMODEL=${X195_X3_MODEL:-qwen3.8-flash-next-exl3.rad}
export TP=2 CARDS="" PLACE=all_vram MML=262144 ROPE="" MAXSEQS=${X195_MAXSEQS:-8} MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
EXTRA0="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0"
export EXTRA="$EXTRA0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats195
export FN4BT="" FN4BTP="" FN4BTL="" FN4X3DUAL=1 FN4NR=1 FN4GI=1 FN4FD="" FN4NW=1 FN4SW="" FN4WV=1 FN4RV=11 FN4AL="" FN4UM=5 FN4SS=1 FN4CH="" FN4XM=1 FN4SPHC="" FN4X3C=1 FN4QSA="" FN4P2X="" FN4MP="" FN4X3GD=1 FN4X3DZ=1 PROFEVERY=""
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
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 195: VIRHE temperature guard ($hot) -> blocks stopped"; return 0; }
    sleep 1; done; }
MBU=$TL/mbu.py
compose2() { docker compose -p $1 --project-directory $C -f $Y "${@:2}"; }
down2() { run "PROJ=$1 CN=$1 PORT=$2 CARDS=x compose2 $1 down > /dev/null 2>&1; docker rm -f $1 > /dev/null 2>&1"; }
upsrv() {   # upsrv NAME PORT CARDS: compose up -d (no wait)
  [ "$DRY" = 1 ] && { echo "+ PROJ=$1 CN=$1 PORT=$2 CARDS=$3 TP=$TP MAXSEQS=$MAXSEQS FNIMAGE=$FNIMAGE EXTRA='$EXTRA' compose -p $1 -f $Y up -d"; return 0; }
  down2 $1 $2
  PROJ=$1 CN=$1 PORT=$2 CARDS=$3 compose2 $1 up -d > $O/up-$1.txt 2>&1
}
waitsrv() {   # waitsrv NAME PORT T0 -> 0 healthy
  [ "$DRY" = 1 ] && { echo "+ wait /health :$2 <= 1200 s"; return 0; }
  local ok=0 i S=$O/start-$1.log
  for i in $(seq 1 240); do
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:$2/health)" = 200 ] && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Status}}' $1 2>/dev/null)" != running ] && break
    sleep 5
  done
  docker logs $1 > $S 2>&1
  [ $ok = 1 ] || { log "RESULT fn4 195 start $1: VIRHE not healthy after $(( $(date +%s) - $3 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
  log "RESULT fn4 195 start $1 (:$2): healthy in $(( $(date +%s) - $3 )) s; $(grep -m1 -E '^I kv: ' $S | cut -c3-170)"
}
mbr() {   # mbr PORT NAME PHASE -> $O/$TAG-NAME.txt
  run "(cd $D && MB_URL=http://127.0.0.1:$1 timeout 400 python3 $MBU --tag $TAG-$2 --phases $3 --dur 20 --warm 6 > $O/$TAG-$2.txt 2>&1)"
}
mbv() { grep '^MB ' $O/$TAG-$1.txt 2>/dev/null | head -1 | cut -c4- | python3 -c 'import json,sys
try: d = json.loads(sys.stdin.read()); print("%s %s" % (d.get("agg_dec_tps"), d.get("ms_decode_step")))
except Exception: print("None None")'; }
nudge2() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' http://127.0.0.1:$1/completion > /dev/null"; }
SA=radiance-fndp2a; SB=radiance-fndp2b; PA=8090; PB=8092; CA=$C3,$C03; CB=$C86,$C83
log "RESULT fn4 195-fn-dp2: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) EXL3 DP2: A = dbba+27d4 :$PA, B = 5943+4d61 :$PB, TP2, MAXSEQS $MAXSEQS a server, fnsmoke2's config (wht6) | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 195: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 195: VIRHE image $IMG missing"; exit 1; }
  [ -e "$Y" ] && [ -n "$TL" ] && [ -e $MBU ] && [ -e $D/ds41_tests.py ] || { log "RESULT fn4 195: VIRHE $YN, tools/mbu.py or ds41_tests.py missing"; exit 1; }
  mkdir -p $O $STATSDIR
  find $O -maxdepth 1 -name "$TAG.abort" -delete; find $O -maxdepth 1 -name "$TAG.tstop" -delete
  find $O -maxdepth 1 -name "$TAG-*.txt" -delete
  trap 'touch $O/$TAG.tstop 2>/dev/null; down2 $SA $PA; down2 $SB $PB' EXIT
  trap 'exit 143' TERM INT
  tguard & TGPID=$!
else
  echo "+ guards; tguard &"; TGPID=""
fi
t0=$(date +%s)
upsrv $SA $PA $CA; upsrv $SB $PB $CB
okA=0; okB=0
waitsrv $SA $PA $t0 && okA=1
waitsrv $SB $PB $t0 && okB=1
[ $okA = 1 ] && nudge2 $PA; [ $okB = 1 ] && nudge2 $PB
if [ $okA = 1 ] && [ $okB = 1 ]; then
  mbr $PA c8a c8 & p1=$!; mbr $PB c8b c8 & p2=$!; wait $p1 $p2   # not a bare wait: tguard runs in the background
  mbr $PA c4a c4 & p1=$!; mbr $PB c4b c4 & p2=$!; wait $p1 $p2   # not a bare wait: tguard runs in the background
fi
if [ $okA = 1 ]; then
  mbr $PA c1a c1
  run "(cd $D && DS41_URL=http://127.0.0.1:$PA timeout 600 python3 ds41_tests.py prefill --depths 8192 --cpt 3.6 --calibrate 0 --reps 1 > $O/$TAG-pfwarm.txt 2>&1)"
  run "(cd $D && DS41_URL=http://127.0.0.1:$PA timeout 600 python3 ds41_tests.py prefill --depths 65536 --cpt 3.6 --calibrate 0 --reps 1 > $O/$TAG-pf64.txt 2>&1)"
fi
run "docker logs $SA > $O/$TAG-a-engine.log 2>&1; docker logs $SB > $O/$TAG-b-engine.log 2>&1"
down2 $SA $PA; down2 $SB $PB
if [ "$DRY" != 1 ]; then
  set -- $(mbv c8a); a8=$1; m8a=$2; set -- $(mbv c8b); b8=$1; m8b=$2
  set -- $(mbv c4a); a4=$1; m4a=$2; set -- $(mbv c4b); b4=$1; m4b=$2
  set -- $(mbv c1a); a1=$1; m1a=$2
  pf=$(grep -h '^J ' $O/$TAG-pf64.txt 2>/dev/null | python3 -c 'import json,sys
for l in sys.stdin:
    d = json.loads(l[2:])
    if d.get("test") == "prefill": print("%s tok (cached %s): TTFT %s s, %s tok/s" % (d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("prefill_tps")))' 2>&1 | head -1)
  sum() { python3 -c 'import sys
v = [float(x) for x in sys.argv[1:] if x not in ("None", "")]
print("%.1f" % sum(v) if len(v) == len(sys.argv[1:]) else "-")' "$@"; }
  log "RESULT fn4 195 DP2 (TP2 x2, wht6, MAXSEQS $MAXSEQS a server) vs TP4 (191 A, same image / config): 16 users (c8 + c8) $(sum $a8 $b8) tok/s (A $a8 @ $m8a ms/step, B $b8 @ $m8b) vs TP4 c16 1002.3 (34.4 ms) | 8 users (c4 + c4) $(sum $a4 $b4) tok/s (A $a4 @ $m4a, B $b4 @ $m4b) | 1 user (c1 on A) $a1 tok/s @ $m1a ms/step | 64k cold prefill on A: ${pf:-VIRHE no line} vs TP4 8811 tok/s"
fi
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
log "RESULT fn4 195-fn-dp2: done"
