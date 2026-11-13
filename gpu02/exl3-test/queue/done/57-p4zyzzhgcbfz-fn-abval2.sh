#!/bin/bash
# 128-fn-abval2.sh (queue item, 9.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~75 min) -- toward
# the user's decode target (mb c16 >= 1000 tok/s; main + F1n 983.4, -1.7 %): in-process A/B blocks (tools/abblocks.py,
# one server, POST /admin/fn4) of the decode levers against F1n ALONE, which is the reference arm A of every run:
#   noise      A = B (only the epoch moves): the in-process noise floor
#   normgate   F1n from 33 rows (hc_normwave 33): at <= 32 rows the stock norm is 2.0 us, nothing to win
#   fdv2       + F1a v2 (hc_fusedown 2): the folded down reduce with the low-register tail (the first tail's 226 VGPRs
#              cost 119's c16all -6.4 % at c8)
#   gather_inv + F2a      no_route + F2b      (119 could not separate them from noise: +0.5 / 0.0 % at c16)
#   stack      F1n + F1a v2 + F2a + F2b
# each at c16 (X128_R16 rounds, default 3) and c8 (X128_R8, default 2); rounds A B B A; a block = POST + epoch_bump, mb
# window 6 s warm + 20 s (decode only: no soak, TESTING.md 9.11.). IDENTITY in this process: the 33k copy cold, then
# with the stack and with F1n alone (both resuming from the cache), the two byte for byte.
# IMAGE radiance-local:c16-sw2 (fn4-c16 646ef42 = c16-sw + F1a v2 + the F1n row floor), started with
# RADIANCE_FN4_SWITCHABLE=1 and RADIANCE_FN4_HC_NORMWAVE=1 on the main-line base (126: MNBT 4096 + --prefill-span 2,
# --prefill-interleave 0 --prefix-resume nearest; TP4 all_vram MML 262144 MAXSEQS 16 MTP 3), compose fn-tp4s.yaml.
# Reading: a lever's B/A against the noise run's spread; the distance to 1000 tok/s from the stack's c16 B.
# Power is not touched; read-only temperature watcher (junction 100 C / mem 96 C for 10 s -> stop).
# DRYRUN=1 bash 128-fn-abval2.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4s.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/abblocks.py" ] && { TL=$f; break; }; done
IMG=${X128_IMAGE:-radiance-local:c16-sw2}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-abval2
export FNIMAGE=$IMG FNMODELDIR=$HOME/models/flashnext-iq4r FNMODEL=qwen3.8-next-flash-fp8-iq4r-moe.rad
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
export EXTRA="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats128
export FN4NR="" FN4GI="" FN4FD="" FN4NW=1 FN4SW=1
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
compose() { docker compose --project-directory $C -f $Y "$@"; }
down() { run "compose down > /dev/null 2>&1; docker rm -f $CN > /dev/null 2>&1"; }
caps() { local d h out=""; for d in /sys/class/drm/card*/device; do [ -e $d/unique_id ] || continue; h=$(ls -d $d/hwmon/hwmon* 2>/dev/null | head -1); [ -n "$h" ] || continue
  out="$out $(basename $(readlink -f $d)):cap=$(( $(cat $h/power1_cap 2>/dev/null || echo 0) / 1000000 ))W,hot=$(( $(cat $h/temp2_input 2>/dev/null || echo 0) / 1000 ))C"; done; echo "$out"; }
tguard() { local d h j m n=0 hot
  while [ ! -e $O/$TAG.tstop ]; do hot=""
    for d in /sys/class/drm/card*/device; do [ -e $d/unique_id ] || continue; h=$(ls -d $d/hwmon/hwmon* 2>/dev/null | head -1); [ -n "$h" ] || continue
      j=$(( $(cat $h/temp2_input 2>/dev/null || echo 0) / 1000 )); m=$(( $(cat $h/temp3_input 2>/dev/null || echo 0) / 1000 ))
      { [ $j -ge 100 ] || [ $m -ge 96 ]; } && hot="$hot $(basename $(readlink -f $d)):j$j/m$m"; done
    if [ -n "$hot" ]; then n=$((n + 1)); else n=0; fi
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 128: VIRHE temperature guard ($hot) -> blocks stopped"; return 0; }
    sleep 1; done; }
restart() {   # restart TAG -> 0 healthy
  local tag=$1 ok=0 t0 i S=$O/start-$1.log
  [ "$DRY" = 1 ] && { echo "+ FNIMAGE=$FNIMAGE FN4SW=$FN4SW EXTRA='$EXTRA' compose -f $Y up -d; health <= 1200 s; docker logs > $S"; return 0; }
  down; t0=$(date +%s); compose up -d > $O/up-$tag.txt 2>&1
  for i in $(seq 1 240); do
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' $URL/health)" = 200 ] && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Status}}' $CN 2>/dev/null)" != running ] && break
    sleep 5
  done
  docker logs $CN > $S 2>&1
  [ $ok = 1 ] || { log "RESULT fn4 128 restart-$tag: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
  log "RESULT fn4 128 restart-$tag: healthy in $(( $(date +%s) - t0 )) s; fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300)"
}
probe8() {   # probe8 TAG -> "ms tok/s" of a 45 s mb c8
  [ "$DRY" = 1 ] && { echo "+ (cd $D && timeout 400 python3 mb.py --tag $1 --phases c8 --dur 45 --warm 10 > $O/$1-c8.txt)" >&2; echo "25.6 744"; return 0; }
  (cd $D && timeout 400 python3 mb.py --tag $1 --phases c8 --dur 45 --warm 10 > $O/$1-c8.txt 2>&1)
  grep '^MB ' $O/$1-c8.txt 2>/dev/null | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[3:]); print(d.get("ms_decode_step"), d.get("agg_dec_tps"))' 2>/dev/null || echo "0 0"
}
setsw() {   # setsw JSON: queue switches (they land at the next step release)
  run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"
}
copyt() {   # copyt NAME: the 33k copy test (greedy), output $O/copy-$TAG-NAME-run1.txt
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset 0 --tag=-$TAG-$1 --outdir $O > $O/$TAG-$1-copy.txt 2>&1)"
}
abrun() {   # abrun NAME PHASE ROUNDS A_JSON B_JSON
  local nm=$1 ph=$2 rr=$3
  [ -e $O/$TAG.abort ] && return 0
  run "timeout 3600 python3 -I $TL/abblocks.py --url $URL --mb-dir $D --label $TAG-$nm-$ph --phase $ph --dur ${X128_DUR:-20} --warm ${X128_WARM:-6} --rounds $rr --a '$4' --b '$5' --names F1n,$nm --out $O/$TAG-blocks.jsonl > $O/$TAG-$nm-$ph.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 128 $nm $ph: $(grep '^ABB ' $O/$TAG-$nm-$ph.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=4: "-" if v is None else format(v, ".%df" % n)
print("tok/s B/A %s +- %s (t95, %d rounds: %s) | ms/step B/A %s +- %s | A %s B %s tok/s | %d blocks, %d flagged, %s s" % (
    f(d["tps_ratio"]), f(d["tps_ci95"]), len(d["rounds"]), " ".join(f(r["tps_ratio"]) for r in d["rounds"]),
    f(d["ms_ratio"]), f(d["ms_ci95"]), f(d["A_tps_mean"], 1), f(d["B_tps_mean"], 1), d["blocks"], d["blocks_bad"], d["wall_s"]))' 2>&1 | cut -c1-600)"
}
OFF='{"no_route": 0, "gather_inv": 0, "hc_fusedown": 0, "hc_normwave": 1}'   # F1n alone: the reference
STACK='{"no_route": 1, "gather_inv": 1, "hc_fusedown": 2, "hc_normwave": 1}'
log "RESULT fn4 128-fn-abval2: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) RADIANCE_FN4_SWITCHABLE=1 EXTRA '$EXTRA' | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 128: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 128: VIRHE image $IMG missing"; exit 1; }
  [ -e "$Y" ] && grep -q RADIANCE_FN4_SWITCHABLE $Y && [ -n "$TL" ] || { log "RESULT fn4 128: VIRHE $YN (with RADIANCE_FN4_SWITCHABLE) or tools/abblocks.py missing"; exit 1; }
  mkdir -p $O $O/guard $STATSDIR
  find $O -maxdepth 1 -name "$TAG.abort" -delete; find $O -maxdepth 1 -name "$TAG.tstop" -delete; find $O -maxdepth 1 -name "$TAG-blocks.jsonl" -delete
  trap 'touch $O/$TAG.tstop 2>/dev/null; down' EXIT
  trap 'exit 143' TERM INT
  tguard & TGPID=$!
else
  echo "+ guards; tguard &"; TGPID=""
fi
restart $TAG || exit 1
if [ "$DRY" != 1 ]; then
  (cd $D && timeout 600 python3 ds41_tests.py correct --only-q --effort none --outdir $O/guard > $O/guard/$TAG-warm-q.txt 2>&1)
  (cd $D && timeout 600 python3 ds41_tests.py prefill --depths 16384 --cpt 3.6 --calibrate 0 > $O/guard/$TAG-warm-pf.txt 2>&1)
  timeout 300 python3 $Q/prodfc.py $URL 384 > $O/guard/$TAG-fc1.txt 2>&1
  log "RESULT fn4 128 1-user: $(grep '^FC ' $O/guard/$TAG-fc1.txt | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()[3:])["ms_per_step"])' 2>/dev/null || echo n/a) ms/step (informational)"
fi
tries=0
while :; do
  set -- $(probe8 $TAG-st$tries); ms=$1
  st=$(awk -v m=$ms 'BEGIN{print (m <= 0) ? "unknown" : (m <= 26.8) ? "fast" : (m >= 28.5) ? "slow" : "between"}')
  log "RESULT fn4 128 state probe $tries: mb c8 $ms ms -> $st"
  { [ "$st" = slow ] && [ $tries -lt 2 ]; } || break
  tries=$((tries + 1)); restart $TAG-r$tries || exit 1
done
# identity across a flip, in this process: a cold run fills the prefix cache, then all on and all off both resume
# from it (a hit and a cold prompt are not byte-identical under --prefix-resume nearest), compared with each other
setsw "$OFF"; copyt cold
setsw "$STACK"; copyt all
setsw "$OFF"; copyt off
if [ "$DRY" = 1 ]; then echo "+ cmp $O/copy-$TAG-off-run1.txt $O/copy-$TAG-all-run1.txt"
else
  a=$O/copy-$TAG-off-run1.txt; b=$O/copy-$TAG-all-run1.txt
  if [ -s $a ] && [ -s $b ]; then
    if cmp -s $a $b; then log "RESULT fn4 128 identity: the 33k copy output with the stack (F1n + F1a v2 + F2a + F2b) is IDENTICAL to F1n alone in one process ($(wc -c < $b) bytes)"
    else log "RESULT fn4 128 identity: VIRHE DIFFERENT from byte $(cmp $a $b 2>&1 | grep -oE 'byte [0-9]+' | head -1)"; fi
  else log "RESULT fn4 128 identity: not compared (missing $a or $b)"; fi
  log "RESULT fn4 128 switch log: $(docker logs $CN 2>&1 | grep -E 'fn4: switches' | tail -3 | sed 's/^[A-Z] //' | tr '\n' '|' | cut -c1-500)"
fi
R16=${X128_R16:-3}; R8=${X128_R8:-2}
for ph in c16 c8; do
  rr=$R16; [ $ph = c8 ] && rr=$R8
  abrun noise $ph $rr "$OFF" "$OFF"
  abrun normgate $ph $rr "$OFF" '{"hc_normwave": 33}'
  abrun fdv2 $ph $rr "$OFF" '{"hc_fusedown": 2}'
  abrun gather_inv $ph $rr "$OFF" '{"gather_inv": 1}'
  abrun no_route $ph $rr "$OFF" '{"no_route": 1}'
  abrun stack $ph $rr "$OFF" "$STACK"
done
run "docker logs $CN > $O/$TAG-engine.log 2>&1"
[ "$DRY" = 1 ] || log "RESULT fn4 128 engine: $(grep -cE 'fn4: switches .* applied' $O/$TAG-engine.log) applied changes, $(grep -cE 'fn4: .* refused' $O/$TAG-engine.log) refused; $(grep -E 'refused' $O/$TAG-engine.log | grep fn4 | head -2 | tr '\n' '|' | cut -c1-300)"
down
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
log "RESULT fn4 128-fn-abval2: done"
