#!/bin/bash
# 190-fn-iq4rscreen.sh (queue item, 12.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~15 min) -- QUICK
# SCREENS replacing r12 (188 = 165 p2x + 158 sphc in one start; user 12.11.: too slow -- the target metric only, A B B A with
# 2 blocks a side, short windows; pass = beyond the block spread). ONE start of radiance-local:c16-rf (fn4-c16 51a2ddc6), the
# iq4r .rad, stack 3's switches, RADIANCE_FN4_P2X=1 + RADIANCE_FN4_SPHC=1 declared (compose fn-tp4zp.yaml). Both levers act
# in PREFILL only, so their target metric is fresh-prefill tok/s (16k + 64k, abblocks2 --no-mb), each lever with the other off:
#   P2x  moe_pair 0 vs 1, 0 vs 2      SP-hc  hc_sp 0 vs 1
# Exactness: a FRESH 33k copy (prefix cache reset) per value, byte for byte against moe_pair 0 + hc_sp 0 -- the 20-prompt
# greedy set never reaches either path (prompts < 2048 tokens: one piece, no pipelined pair, no long prefill). Engine notes
# (P2x / SP-hc) say whether the paths ran.
# DROPPED (12.11. re-scope): the warm-up answer compare vs 128, the state probe, the 128 cold-copy cross-process compare, decode
# blocks (both levers leave decode untouched), the 128k depth, 3-round A B B A and the mb pf rounds.
# DRYRUN=1 bash 190-fn-iq4rscreen.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4zp.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/heatsoak.sh" ] && { TL=$f; break; }; done
IMG=${X190_IMAGE:-radiance-local:c16-rf}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-iq4scr
export FNIMAGE=$IMG FNMODELDIR=${X190_DIR:-$HOME/models/flashnext-iq4r} FNMODEL=${X190_MODEL:-qwen3.8-next-flash-fp8-iq4r-moe.rad}
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
EXTRA0="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0"
export EXTRA="$EXTRA0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats190
export FN4NR=1 FN4GI=1 FN4FD="" FN4NW=1 FN4SW="" FN4WV=1 FN4RV=11 FN4AL="" FN4UM=5 FN4SS=1 FN4CH="" FN4XM="" FN4SPHC=1 FN4P2X=1 FN4MP="" PROFEVERY=""
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
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 190: VIRHE temperature guard ($hot) -> blocks stopped"; return 0; }
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
  [ $ok = 1 ] || { log "RESULT fn4 190 restart-$tag: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
  log "RESULT fn4 190 restart-$tag: healthy in $(( $(date +%s) - t0 )) s; $(grep -m1 -E '^I kv: ' $S | cut -c3-160); fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300)"
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
  log "RESULT fn4 190: VIRHE the server is not healthy after $1 (/health $code; last E lines: $(grep -E '^E |trap|abort|HSA_STATUS' $O/$TAG-dead-engine.log | tail -3 | tr '\n' ' ' | cut -c1-400)) -> logs saved, down"
  down; exit 1
}
LD=$TL/fn-loadid.py
PFD=${X190_PFD:-16384,65536}
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
swcheck() {   # swcheck WHAT SWITCHES_JSON: every switch named there has that value now
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4: $2"; return 0; }
  local g; g=$(curl -s -m 10 $URL/admin/fn4)
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); w=json.loads(sys.argv[2]); sys.exit(0 if all(d["switches"].get(k) == v for k, v in w.items()) else 1)' "$g" "$2" 2>/dev/null \
    || { log "RESULT fn4 190 $1: VIRHE the switches are not $2 ($(echo "$g" | cut -c1-400))"; return 1; }
}
pcreset() { [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o /dev/null -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  [ "$c" = 200 ] || log "RESULT fn4 190 $1 prefix cache reset: HTTP $c"; }
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
    if cmp -s $1 $2; then log "RESULT fn4 190 identity $3: IDENTICAL ($(wc -c < $2) bytes)"
    else log "RESULT fn4 190 identity $3: DIFFERENT from byte $(cmp $1 $2 2>&1 | grep -oE 'byte [0-9]+' | head -1)"; fi
  else log "RESULT fn4 190 identity $3: not compared (missing $1 or $2)"; fi
}
mpair() {   # mpair N: moe_pair N, landed (an epoch bump, a nudge, checked)
  setsw "{\"moe_pair\": $1, \"epoch_bump\": 1}"; nudge; swcheck "moe_pair $1" "{\"moe_pair\": $1}"
}
abpf() {   # abpf NAME A_JSON B_JSON: prefill-only blocks A B B A (one round = 2 blocks a side)
  [ -e $O/$TAG.abort ] && return 0
  run "timeout 1800 python3 -I $TL/abblocks2.py --url $URL --mb-dir $D --label $TAG-$1 --rounds 1 --a '$2' --b '$3' --pf-depths $PFD --no-mb --names off,$1 --out $O/$TAG-blocks.jsonl > $O/$TAG-$1.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 190 $1 prefill (B/A of prefill tok/s, A B B A, 2 blocks a side): $(grep '^ABB ' $O/$TAG-$1.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
for k, v in (d.get("pf") or {}).items():
    print("%s B/A %.4f %s (A %s | B %s tok/s);" % (k, v["ratio"] or 0, v.get("sep"), " ".join("%.0f" % x for x in v.get("A_blocks", [])), " ".join("%.0f" % x for x in v.get("B_blocks", []))), end=" ")
print("%d blocks, %d flagged, %s s" % (d["blocks"], d["blocks_bad"], d["wall_s"]))' 2>&1 | cut -c1-700)"
}
log "RESULT fn4 190-fn-iq4rscreen: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) iq4r stack 3 + RADIANCE_FN4_P2X=1 + RADIANCE_FN4_SPHC=1, one start: fresh 33k copies + prefill $PFD blocks for moe_pair 0 vs 1, 0 vs 2 and hc_sp 0 vs 1 | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 190: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 190: VIRHE image $IMG missing"; exit 1; }
  [ -e "$Y" ] && [ -n "$TL" ] && [ -e $TL/abblocks2.py ] && [ -e $D/mb.py ] || { log "RESULT fn4 190: VIRHE $YN, tools/abblocks2.py or mb.py missing"; exit 1; }
  grep -q -- '--no-mb' $TL/abblocks2.py || { log "RESULT fn4 190: VIRHE abblocks2.py has no --no-mb"; exit 1; }
  mkdir -p $O $STATSDIR
  find $O -maxdepth 1 -name "$TAG.abort" -delete; find $O -maxdepth 1 -name "$TAG.tstop" -delete
  find $O -maxdepth 1 -name "copy-$TAG-*-run1.txt" -delete
  trap 'touch $O/$TAG.tstop 2>/dev/null; down' EXIT
  trap 'exit 143' TERM INT
  tguard & TGPID=$!
else
  echo "+ guards; tguard &"; TGPID=""
  echo "+ tools: TL=$TL abblocks2 --no-mb: $(grep -c -- '--no-mb' $TL/abblocks2.py)"
fi
sw2() {   # sw2 MP SP: moe_pair MP + hc_sp SP, landed (an epoch bump, a nudge, checked)
  setsw "{\"moe_pair\": $1, \"hc_sp\": $2, \"epoch_bump\": 1}"; nudge; swcheck "moe_pair $1 hc_sp $2" "{\"moe_pair\": $1, \"hc_sp\": $2}"
}
restart $TAG || exit 1
nudge
sw2 1 0 && copyt p1; alive "the copy (moe_pair 1)"
sw2 2 0 && copyt p2; alive "the copy (moe_pair 2)"
sw2 0 1 && copyt s1; alive "the copy (hc_sp 1)"
sw2 0 0 && copyt b0; alive "the copy (both off)"
cmpf $O/copy-$TAG-b0-run1.txt $O/copy-$TAG-p1-run1.txt "P2x fresh 33k copy, moe_pair 1 vs 0"
cmpf $O/copy-$TAG-b0-run1.txt $O/copy-$TAG-p2-run1.txt "P2x fresh 33k copy, moe_pair 2 vs 0"
cmpf $O/copy-$TAG-b0-run1.txt $O/copy-$TAG-s1-run1.txt "SP-hc fresh 33k copy, hc_sp 1 vs 0"
abpf p2x1 '{"moe_pair": 0, "hc_sp": 0}' '{"moe_pair": 1, "hc_sp": 0}'; alive "the P2x blocks (0 vs 1)"
abpf p2x2 '{"moe_pair": 0, "hc_sp": 0}' '{"moe_pair": 2, "hc_sp": 0}'; alive "the P2x blocks (0 vs 2)"
abpf sph '{"moe_pair": 0, "hc_sp": 0}' '{"moe_pair": 0, "hc_sp": 1}'; alive "the SP-hc blocks"
sw2 0 0
run "docker logs $CN > $O/$TAG-engine.log 2>&1"
[ "$DRY" = 1 ] || log "RESULT fn4 190 engine notes: $(grep -E 'fn4 (P2x|SP-hc)' $O/$TAG-engine.log | sed 's/^[A-Z] //' | cut -c1-160 | sort | uniq -c | head -6 | tr '\n' ' ' | cut -c1-700); $(grep -cE 'fn4: switches .* applied' $O/$TAG-engine.log) applied changes, $(grep -cE 'fn4: .* refused' $O/$TAG-engine.log) refused"
down
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
log "RESULT fn4 190-fn-iq4rscreen: done"
