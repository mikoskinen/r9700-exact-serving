#!/bin/bash
# 181-fn-wirecost.sh (queue item, 11.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~22 min) -- fn4 BI:
# THE COST OF THE EXACT AR WIRE on the EXL3 line (c16-x3arm, 173 x3arm's A env: X3CODES + exl3_gd 1). Why: at four ranks a
# message of 128 KiB+ (26+ rows at n_embd 2560) goes out on the lossy tiered-int8 two-shot, below it exact; and in that
# two-shot the owner's own contribution stays exact while the peers' are coded -- the owner of a row's elements follows the
# message size (shard = numel / 4) -- so a row's all-reduce bytes depend on what else is in the step: a BI breaker by
# construction. --tp-wire exact removes both; this prices it. Two starts A B (X181_STARTS; was A B A B) (A = the line's --tp-wire wht6 --tp-wire-min-kb
# 128, B = + --tp-wire exact, last flag wins in core/config.cpp), each mb.py c1 / c4 / c8 (45 s, warm 10) + pf (4 streams +
# one fresh ~32k prefill: TTFT). Reported per phase: B/A of ms a decode step, aggregate decode tok/s, the pf TTFT, each start's
# values. Each start also runs mb c16 at 160's load (dur 30, warm 8) first: the A starts give c16-x3arm's start-level c16 beside
# 160's 1031.7 / 1023.8 (173's 947.0 was the 20 s A/B block: a 14 s window, decode share 0.628 vs mb's 0.68).
# Measurement only, no stack change. Power untouched. NO cc_sim.
# DRYRUN=1 bash 181-fn-wirecost.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4zq.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/heatsoak.sh" ] && { TL=$f; break; }; done
IMG=${X181_IMAGE:-radiance-local:c16-x3arm}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-wire
export FNIMAGE=$IMG FNMODELDIR=${X181_X3_DIR:-$HOME/models/flashnext-exl3} FNMODEL=${X181_X3_MODEL:-qwen3.8-flash-next-exl3.rad}
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
EXTRA0="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0"
export EXTRA="$EXTRA0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats181
export FN4NR=1 FN4GI=1 FN4FD="" FN4NW=1 FN4SW="" FN4WV=1 FN4RV=11 FN4AL="" FN4UM=5 FN4SS=1 FN4CH="" FN4XM="" FN4SPHC="" FN4X3C=1 FN4QSA="" FN4P2X="" FN4MP="" FN4X3GD=1 FN4X3DZ="" PROFEVERY=""
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
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 181: VIRHE temperature guard ($hot) -> blocks stopped"; return 0; }
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
  [ $ok = 1 ] || { log "RESULT fn4 181 restart-$tag: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
  log "RESULT fn4 181 restart-$tag: healthy in $(( $(date +%s) - t0 )) s; $(grep -m1 -E '^I kv: ' $S | cut -c3-160); fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300)"
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
  log "RESULT fn4 181: VIRHE the server is not healthy after $1 (/health $code; last E lines: $(grep -E '^E |trap|abort|HSA_STATUS' $O/$TAG-dead-engine.log | tail -3 | tr '\n' ' ' | cut -c1-400)) -> logs saved, down"
  down; exit 1
}
mbrun() {   # mbrun NAME: mb.py c16 at 160's load (dur 30, warm 8), then c1,c4,c8,pf -> $O/$TAG-NAME.txt
  run "(cd $D && timeout 600 python3 mb.py --tag $TAG-$1 --phases c16 --dur 30 --warm 8 > $O/$TAG-$1.txt 2>&1)"
  run "(cd $D && timeout 1500 python3 mb.py --tag $TAG-$1 --phases c1,c4,c8,pf --dur 45 --warm 10 >> $O/$TAG-$1.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT fn4 181 $1: $(grep '^MB ' $O/$TAG-$1.txt | cut -c4- | python3 -c 'import json,sys
for l in sys.stdin:
    d = json.loads(l)
    if d.get("phase") == "pf": print("pf TTFT %s s (%s tok, %s tok/s);" % (d.get("prefill_ttft"), d.get("prefill_tokens"), d.get("prefill_tps")), end=" ")
    else: print("%s %s ms/step %s tok/s acc %s decode share %s;" % (d.get("phase"), d.get("ms_decode_step"), d.get("agg_dec_tps"), d.get("acceptance"), d.get("decode_share")), end=" ")' 2>&1 | cut -c1-600)"
}
log "RESULT fn4 181-fn-wirecost: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) the EXL3 line, A (wht6, min-kb 128) / B (+ --tp-wire exact), starts ${X181_STARTS:-a1 b1}, mb c16 (160 load) + c1 / c4 / c8 / pf | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 181: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 181: VIRHE image $IMG missing"; exit 1; }
  [ -e "$Y" ] && [ -n "$TL" ] && [ -e $D/mb.py ] || { log "RESULT fn4 181: VIRHE $YN, tools or mb.py missing"; exit 1; }
  mkdir -p $O $STATSDIR
  find $O -maxdepth 1 -name "$TAG.abort" -delete; find $O -maxdepth 1 -name "$TAG.tstop" -delete
  trap 'touch $O/$TAG.tstop 2>/dev/null; down' EXIT
  trap 'exit 143' TERM INT
  tguard & TGPID=$!
else
  echo "+ guards; tguard &"; TGPID=""
fi
for st in ${X181_STARTS:-a1 b1}; do   # coordinator 11.11.: fewer starts -- A then B (the wire is a serve flag)
  [ -e $O/$TAG.abort ] && break
  case $st in a*) export EXTRA="$EXTRA0" ;; b*) export EXTRA="$EXTRA0 --tp-wire exact" ;; esac
  restart $TAG-$st || continue
  run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"
  mbrun $st; alive "mb $st"
  run "docker logs $CN > $O/$TAG-$st-engine.log 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 181 $st wire: $(grep -m1 -iE 'tp-wire|wire:' $O/$TAG-$st-engine.log | cut -c1-200)"
  down
done
if [ "$DRY" != 1 ]; then
  log "RESULT fn4 181 B/A (exact wire vs the line's): $(python3 -c 'import json,sys,statistics
def rd(f):
    o = {}
    try:
        for l in open(f):
            if l.startswith("MB "):
                d = json.loads(l[3:]); o[d["phase"]] = d
    except OSError: pass
    return o
O, T = sys.argv[1], sys.argv[2]
r = {s: rd("%s/%s-%s.txt" % (O, T, s)) for s in ("a1", "b1", "a2", "b2")}
out = []
for ph, key in (("c1", "ms_decode_step"), ("c4", "ms_decode_step"), ("c8", "ms_decode_step"), ("c16", "ms_decode_step"), ("c1", "agg_dec_tps"), ("c4", "agg_dec_tps"), ("c8", "agg_dec_tps"), ("c16", "agg_dec_tps"), ("c16", "decode_share"), ("pf", "prefill_ttft")):
    a = [r[s][ph][key] for s in ("a1", "a2") if ph in r[s] and r[s][ph].get(key)]
    b = [r[s][ph][key] for s in ("b1", "b2") if ph in r[s] and r[s][ph].get(key)]
    if a and b:
        out.append("%s %s A %s B %s B/A %.4f" % (ph, key, "/".join(str(x) for x in a), "/".join(str(x) for x in b), statistics.mean(b) / statistics.mean(a)))
    else:
        out.append("%s %s: missing" % (ph, key))
print(" | ".join(out))' $O $TAG 2>&1 | cut -c1-1500)"
fi
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
log "RESULT fn4 181-fn-wirecost: done"
