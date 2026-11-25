#!/bin/bash
# 88-fn-pupair.sh (queue item, 8.11.2026, queue-ar; gpu02, cards 83:00 + 86:00, every radiance container DOWN; ~10 min) --
# Flash-Next on the tuned3 image measured by the produp method, the one production's 14.8-15.22 ms gates come from, so the
# 1.17 / tuned3 engine and production 1.15.1 meet on one method and one card pair.
# WHY. FN-A's stock arm gated 17.83 ms (qgate method: correct --only-q + a 16k prefill + one prodfc) against production's
# 14.8-15.22; the same qgate method on stock 1.15.1 on 83+86 gave 17.71-17.82 (fXYB24-A, qwen-tp4-1103.log:411,544,586), so
# no engine gap there. The produp method (prodfc twice on a fresh server, the second kept) gave prod A 19.54 / 19.59 -> 15.11 /
# 14.80 ms on stock 1.15.1 (q/produp/fc-a-*.txt, q/roll500k/fc-a-*.txt). 1.17 no longer records Flash-Next's MTP verify and
# draft passes (qwen4exp pass_key returns 0 for them, upstream ded64c0 + aa14eea): the one engine change that could cost
# 0.3-1.0 ms in the produp state (an estimate, not measured).
# WHAT. fn-tp2.yaml (the production line, private n-gram table, no key), the tuned3 image, no tuned switch (stock env),
# MAXSEQS 16, HP 16384. After a fresh start: prodfc warm, prodfc measured (produp's pair; a measured run above 21 ms gets
# produp's one restart + warm2 + measured), prodfc measured twice more, then 84's qgate sequence for continuity; engine log.
# RULE. measured <= 15.4 ms: no engine gap at the produp state (stay as is); >= 15.8 ms: a gap -> the verify-tape patch
# (qwen4exp pass_key keys the 4-row verify and the 1-token draft passes) is worth its ~1.5-2 h; in between: unclear.
# References (no rerun): prod A produp pairs 19.54 -> 15.11, 19.59 -> 14.80 ms; prod gates 14.8-15.22; qgate stock 1.15.1
# 83+86 17.71-17.82, FN-A stock 17.83.
# Bounded: compose up/down of fn-tp2.yaml's own project (radiance-fntest), health <= 20 min, every client under timeout.
# Never touches radiance-prod-* or /dev/shm/ngram-*. DRYRUN=1 bash 88-fn-pupair.sh : prints every command, writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fnpu; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
for Y in "$H/fn-tp2.yaml" "$H/../fn-tp2.yaml" "$HOME/gpu/ds41/exl3-test/queue-ar/fn-tp2.yaml"; do [ -e "$Y" ] && break; done
CN=radiance-fntest; URL=http://127.0.0.1:8090; export DS41_URL=$URL
export FNIMAGE=${FN_IMAGE:-radiance-local:ds41-exl3-117-390664b3-fa0e9580-tuned3-hd4a11cab}
export FNMODELDIR=$HOME/models/flashnext-iq4r FNMODEL=qwen3.8-next-flash-fp8-iq4r-moe.rad MAXSEQS=16 HP=16384
LIM=${FN_GATE_MS:-21}
unset RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES \
      RADIANCE_PLACE_TURNOVER RADIANCE_STAGE_MIXED RADIANCE_STAGE_SPAN RADIANCE_ARENA_LEVEL_MAX RADIANCE_ARENA_HOLD_MS
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
compose() { docker compose --project-directory $C -f $Y "$@"; }
down() { run "compose down > /dev/null 2>&1; docker rm -f $CN > /dev/null 2>&1"; }
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
restart() {   # restart TAG
  local tag=$1 ok=0 t0 t1 i S=$O/start-$1.log
  [ "$DRY" = 1 ] && { echo "+ FNIMAGE=$FNIMAGE FNMODEL=$FNMODEL compose -f $Y up -d; health <= 1200 s; docker logs > $S"; return 0; }
  down; t0=$(date +%s)
  compose up -d > $O/up-$tag.txt 2>&1
  for i in $(seq 1 240); do
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' $URL/health)" = 200 ] && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Status}}' $CN 2>/dev/null)" != running ] && break
    sleep 5
  done
  t1=$(date +%s); docker logs $CN > $S 2>&1
  [ $ok = 1 ] || { log "RESULT fnpu restart-$tag: VIRHE not healthy after $((t1-t0)) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  log "RESULT fnpu restart-$tag: healthy in $((t1-t0)) s (IMAGE=$FNIMAGE MODEL=$FNMODEL MAXSEQS=${MAXSEQS:-16} HP=${HP:-16384}); $(grep -m1 -E '^I kv: ' $S | cut -c3-200); VRAM experts $(grep -m1 -E '^ +experts +[0-9.]+ GiB' $S | awk '{print $2, $3, $4, $5}'), KV $(grep -m1 -E '^ +kv cache +[0-9.]+ GiB' $S | awk '{print $3, $4}'); $(grep -m1 -oE 'pinned by this process [0-9.]* GiB' $S); E lines $(grep -cE '^E ' $S)"
}
check() {   # check TAG N -> ms/step of prodfc.py (999 when it fails)
  local tag=$1 n=$2
  [ "$DRY" = 1 ] && { echo "+ warm-up (correct --only-q, prefill 16384) + prodfc.py $URL 384 -> $O/guard/$tag-fc$n.txt" >&2; echo 0; return; }
  mkdir -p $O/guard
  (cd $D && timeout 600 python3 ds41_tests.py correct --only-q --effort none --outdir $O/guard > $O/guard/$tag-warm-q.txt 2>&1)
  (cd $D && timeout 600 python3 ds41_tests.py prefill --depths 16384 --cpt 3.6 --calibrate 0 > $O/guard/$tag-warm-pf.txt 2>&1)
  timeout 300 python3 $Q/prodfc.py $URL 384 > $O/guard/$tag-fc$n.txt 2>&1
  grep '^FC ' $O/guard/$tag-fc$n.txt | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()[3:])["ms_per_step"])' 2>/dev/null || echo 999
}
fc() {   # fc NAME -> "ms tps mem_busy" of one prodfc.py run (produp's gate())
  [ "$DRY" = 1 ] && { echo "+ timeout 300 python3 $Q/prodfc.py $URL 384 > $O/fc-$1.txt" >&2; echo "0 0 0"; return; }
  timeout 300 python3 $Q/prodfc.py $URL 384 > $O/fc-$1.txt 2>&1
  grep '^FC ' $O/fc-$1.txt | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[3:]); print(d["ms_per_step"], d["decode_tps"], d["mem_busy_avg"])' 2>/dev/null || echo "999 0 0"
}
log "RESULT fnpu 88-fn-pupair: start IMAGE=$FNIMAGE ($(docker image inspect -f '{{.Id}}' $FNIMAGE 2>/dev/null | cut -c8-19)) model $FNMODELDIR/$FNMODEL compose $Y (cards 83+86, stock env, MAXSEQS 16, HP 16384)"
if [ "$DRY" != 1 ]; then
  [ -e "$Y" ] || { log "RESULT fnpu: VIRHE fn-tp2.yaml not found"; exit 1; }
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fnpu: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  docker image inspect $FNIMAGE > /dev/null 2>&1 || { log "RESULT fnpu: VIRHE image $FNIMAGE missing"; exit 1; }
  [ -f $FNMODELDIR/$FNMODEL ] || { log "RESULT fnpu: VIRHE $FNMODELDIR/$FNMODEL missing"; exit 1; }
  [ -e $Q/prodfc.py ] || { log "RESULT fnpu: VIRHE $Q/prodfc.py missing"; exit 1; }
  mkdir -p $O $O/guard
else
  echo "+ mkdir -p $O $O/guard"
fi
tag=fnpu-stock
restart $tag || { keeplog $tag-fail; exit 1; }
fc warm > /dev/null; read ms tps mb < <(fc m1)
first=$ms
if [ "$DRY" != 1 ] && awk -v m=$ms -v g=$LIM 'BEGIN{exit !(m > g)}'; then
  keeplog $tag-slow1; log "RESULT fnpu: slow at start ($ms ms/step > $LIM) -> one restart (as produp)"
  restart $tag || { keeplog $tag-fail; exit 1; }
  fc warm2 > /dev/null; read ms tps mb < <(fc m1b)
fi
read ms2 tps2 mb2 < <(fc m2); read ms3 tps3 mb3 < <(fc m3)
qg=$(check $tag 1)
run "docker logs $CN > $O/$tag-engine.log 2>&1"
down
[ "$DRY" = 1 ] && { log "RESULT fnpu 88-fn-pupair: dry run done"; exit 0; }
wm=$(grep '^FC ' $O/fc-warm.txt | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()[3:])["ms_per_step"])' 2>/dev/null || echo n/a)
v="unclear (15.4 .. 15.8)"
awk -v m=$ms 'BEGIN{exit !(m <= 15.4)}' && v="no engine gap at the produp state (<= 15.4)"
awk -v m=$ms 'BEGIN{exit !(m >= 15.8)}' && v="GAP (>= 15.8): the verify-tape patch is worth doing"
log "RESULT fnpu 88-fn-pupair: produp pair warm $wm -> measured $ms ms/step $tps tok/s mem_busy $mb (first measured $first); more runs $ms2 / $ms3 ms; qgate sequence $qg ms | verdict: $v | refs: prod A (stock 1.15.1, 83+86) 19.54 -> 15.11, 19.59 -> 14.80; gates 14.8-15.22; qgate stock 1.15.1 83+86 17.71-17.82, FN-A 17.83 | recorded passes: $(grep -c 'recorded pass: draft pass' $O/$tag-engine.log) lines, $(grep -m3 -oE 'recorded pass: draft pass [0-9]+ at [0-9]+ token' $O/$tag-engine.log | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
