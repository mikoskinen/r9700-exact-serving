#!/bin/bash
# 151-fn-prof3.sh (queue item, 10.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~25 min) -- WHERE
# THE TIME GOES NOW: Flash-Next iq4r on stack 3 (radiance-local:c16-sw9, TP4 all_vram MML 262144 MAXSEQS 16 MTP 3, wht6), per rank and op class, decode then prefill, for the next round's levers (ranked by measured ms).
# Templates 120 (decode windows, msnap counters) and 127 (prefill windows, the shutdown table closing the last one).
# CONFIG = 147's stack 3 + --profile-ops: device event pairs around every op and each op's lane wait, a cumulative table a rank
# every RADIANCE_PROFILE_EVERY (4) steps (compose fn-tp4z.yaml). Profiled steps synchronise around every launch: the op
# SHARES, us a call and lane waits are the reading, not tok/s (unprofiled speeds: 147: c4 / c8 / c16 572.3 / 770.1 / 1022.5 tok/s, prefill 8479.9 / 8554.0 / 8482.5).
# PHASES (TESTING.md 9.11.: decode before prefill after a start, no soak time): start; warm-up answers; decode windows
# c16 (mb, 10 s warm + X151_DUR s, default 30); prefill: four fresh 16k prompts back to back, then one fresh 128k prompt; stop (the
# shutdown table closes the last window). tools/opprof.py subtracts each rank's tables around a window: device ms by class
# a rank, lane waits, the top ops; thermwin beside each window.
# Power is not touched (caps logged); read-only temperature watcher (junction 100 C / mem 96 C for 10 s -> stop).
# DRYRUN=1 bash 151-fn-prof3.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41
YN=fn-tp4z.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/opprof.py" ] && [ -e "$f/msnap.py" ] && [ -e "$f/heatsoak.sh" ] && { TL=$f; break; }; done
[ -n "$TL" ] && . $TL/heatsoak.sh   # thermnow / thermwin only
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-prof3; N=151
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 NST=3 WIREKB=128 PMIN="" PMINB=""
export AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats$N PROFEVERY=${X151_EVERY:-4} FN4FD="" FN4SW="" FN4CH="" FN4XM="" FN4AL=""
# the stack (93r's arm stk3): F1n, F2a, F2b, hcw, hc_readv 11 + W1b, hc_upmix 5, MNBT 4096 + span 2, interleave 0
export FN4NW=1 FN4GI=1 FN4NR=1 FN4WV=1 FN4RV=11 FN4SS=1 FN4UM=5 MNBT=4096
export EXTRA="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0 --profile-ops"
export FNIMAGE=${X151_IMAGE:-radiance-local:c16-sw9} FNMODELDIR=${X151_DIR:-$HOME/models/flashnext-iq4r} FNMODEL=${X151_MODEL:-qwen3.8-next-flash-fp8-iq4r-moe.rad}
DEC="c16"; PF128=1
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
compose() { docker compose --project-directory $C -f $Y "$@"; }
down() { run "compose down > /dev/null 2>&1; docker rm -f $CN > /dev/null 2>&1"; }
nl() { [ "$DRY" = 1 ] && echo L || docker logs $CN 2>&1 | wc -l; }
caps() { local d h out=""; for d in /sys/class/drm/card*/device; do [ -e $d/unique_id ] || continue; h=$(ls -d $d/hwmon/hwmon* 2>/dev/null | head -1); [ -n "$h" ] || continue
  out="$out $(basename $(readlink -f $d)):cap=$(( $(cat $h/power1_cap 2>/dev/null || echo 0) / 1000000 ))W,hot=$(( $(cat $h/temp2_input 2>/dev/null || echo 0) / 1000 ))C"; done; echo "$out"; }
tguard() { local d h j m n=0 hot
  while [ ! -e $O/$TAG.tstop ]; do hot=""
    for d in /sys/class/drm/card*/device; do [ -e $d/unique_id ] || continue; h=$(ls -d $d/hwmon/hwmon* 2>/dev/null | head -1); [ -n "$h" ] || continue
      j=$(( $(cat $h/temp2_input 2>/dev/null || echo 0) / 1000 )); m=$(( $(cat $h/temp3_input 2>/dev/null || echo 0) / 1000 ))
      { [ $j -ge 100 ] || [ $m -ge 96 ]; } && hot="$hot $(basename $(readlink -f $d)):j$j/m$m"; done
    if [ -n "$hot" ]; then n=$((n + 1)); else n=0; fi
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 $N: VIRHE temperature guard ($hot) -> windows stopped"; return 0; }
    sleep 1; done; }
log "RESULT fn4 $N-fn-prof3: start IMAGE=$FNIMAGE ($(docker image inspect -f '{{.Id}}' $FNIMAGE 2>/dev/null | cut -c8-19)) MODEL=$FNMODEL MNBT $MNBT EXTRA '$EXTRA' RADIANCE_PROFILE_EVERY=$PROFEVERY decode '$DEC' prefill 16k x4$( [ $PF128 = 1 ] && echo ' + 128k') | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 $N: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $FNIMAGE > /dev/null 2>&1 || { log "RESULT fn4 $N: VIRHE image $FNIMAGE missing"; exit 1; }
  [ -f $FNMODELDIR/$FNMODEL ] || { log "RESULT fn4 $N: VIRHE $FNMODELDIR/$FNMODEL missing"; exit 1; }
  [ -e "$Y" ] && grep -q RADIANCE_PROFILE_EVERY $Y && [ -n "$TL" ] || { log "RESULT fn4 $N: VIRHE $YN (with RADIANCE_PROFILE_EVERY) or tools/opprof.py / msnap.py / heatsoak.sh missing"; exit 1; }
  mkdir -p $O $O/guard $STATSDIR
  find $O -maxdepth 1 -name "$TAG.abort" -delete; find $O -maxdepth 1 -name "$TAG.tstop" -delete; find $O -maxdepth 1 -name "$TAG-win.txt" -delete
  trap 'touch $O/$TAG.tstop 2>/dev/null; down' EXIT
  trap 'exit 143' TERM INT
  tguard & TGPID=$!
else
  echo "+ guards (no radiance container, image, model, $YN with RADIANCE_PROFILE_EVERY, tools); trap: down; tguard &"; TGPID=""
fi
if [ "$DRY" = 1 ]; then echo "+ FNIMAGE=$FNIMAGE MNBT=$MNBT EXTRA='$EXTRA' PROFEVERY=$PROFEVERY compose -f $Y up -d; health <= 1200 s; docker logs > $O/start-$TAG.log"
else
  down; t0=$(date +%s); compose up -d > $O/up-$TAG.txt 2>&1; ok=0
  for i in $(seq 1 240); do
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' $URL/health)" = 200 ] && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Status}}' $CN 2>/dev/null)" != running ] && break
    sleep 5
  done
  docker logs $CN > $O/start-$TAG.log 2>&1
  [ $ok = 1 ] || { log "RESULT fn4 $N: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $O/start-$TAG.log | head -3 | tr '\n' ' ' | cut -c1-400)"; exit 1; }
  log "RESULT fn4 $N: healthy in $(( $(date +%s) - t0 )) s; $(grep -m1 -E '^I kv: ' $O/start-$TAG.log | cut -c3-160); fn4: $(grep -E 'RADIANCE_FN4' $O/start-$TAG.log | sed 's/^[A-Z] //' | sort -u | head -3 | tr '\n' '|' | cut -c1-300)"
fi
run "(cd $D && timeout 600 python3 ds41_tests.py correct --only-q --effort none --outdir $O/guard > $O/guard/$TAG-warm-q.txt 2>&1)"
# ---- decode windows
for p in $DEC; do
  [ -e $O/$TAG.abort ] && break
  run "timeout 30 python3 -I $TL/msnap.py snap $URL $O/$TAG-$p-msa.json > /dev/null 2>&1"
  l0=$(nl); t0=$(date +%s)
  run "(cd $D && timeout 900 python3 mb.py --tag $TAG --phases $p --dur ${X151_DUR:-30} --warm 10 > $O/$TAG-$p.txt 2>&1)"
  run "sleep 3"   # the table the last profiled step wrote
  l1=$(nl); t1=$(date +%s)
  run "timeout 30 python3 -I $TL/msnap.py snap $URL $O/$TAG-$p-msb.json > /dev/null 2>&1"
  run "echo '$p $l0 $l1 $t0 $t1' >> $O/$TAG-win.txt"
  [ "$DRY" = 1 ] || log "RESULT fn4 $N $p (PROFILED, synchronised: not a speed): $(grep '^MB ' $O/$TAG-$p.txt 2>/dev/null | cut -c4- | cut -c1-300) | engine: $(timeout 30 python3 -I $TL/msnap.py delta $O/$TAG-$p-msa.json $O/$TAG-$p-msb.json 2>&1 | cut -c1-400)"
done
# ---- prefill windows (fresh prompts; the last one is closed by the shutdown table)
if [ ! -e $O/$TAG.abort ]; then
  l0=$(nl); t0=$(date +%s)
  for k in 1 2 3 4; do run "(cd $D && timeout 900 python3 ds41_tests.py prefill --depths 16384 --cpt 3.6 --calibrate 0 --reps 1 > $O/$TAG-pf16-$k.txt 2>&1)"; done
  run "sleep 3"; l1=$(nl); t1=$(date +%s)
  if [ $PF128 = 1 ]; then
    run "echo 'pf16 $l0 $l1 $t0 $t1' >> $O/$TAG-win.txt"
    run "(cd $D && timeout 1500 python3 ds41_tests.py prefill --depths 131072 --cpt 3.6 --calibrate 0 --reps 1 > $O/$TAG-pf128.txt 2>&1)"
    t2=$(date +%s); run "echo 'pf128 $l1 - $t1 $t2' >> $O/$TAG-win.txt"
  else
    run "echo 'pf16 $l0 - $t0 $t1' >> $O/$TAG-win.txt"
  fi
fi
# stop first (the shutdown table closes the last window), read the whole log, then remove
run "docker stop -t 60 $CN > /dev/null 2>&1; docker logs $CN > $O/$TAG-engine.log 2>&1"
down
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
if [ "$DRY" = 1 ]; then echo "+ opprof.py $O/$TAG-engine.log --from L0 [--to L1] --label $TAG-<window> --json ... for each window in $O/$TAG-win.txt"
else
  ttft() { cat "$@" 2>/dev/null | python3 -c 'import json,sys
v = [json.loads(l[2:]) for l in sys.stdin if l.startswith("J ")]
print("; ".join("%s tok TTFT %s s %s tok/s" % (d.get("prompt_tokens"), d.get("ttft"), d.get("prefill_tps")) for d in v) or "none")' 2>/dev/null; }
  while read -r p l0 l1 ta tb; do
    if [ "$l1" = - ]; then to=""; else to="--to $l1"; fi
    timeout 600 python3 -I $TL/opprof.py $O/$TAG-engine.log --from $l0 $to --label $TAG-$p --json $O/$TAG-$p-op.json > $O/$TAG-$p-op.txt 2>&1
    case $p in pf16) extra="profiled prompts: $(ttft $O/$TAG-pf16-*.txt | cut -c1-300)" ;; pf128) extra="profiled prompt: $(ttft $O/$TAG-pf128.txt | cut -c1-200)" ;; *) extra="" ;; esac
    log "RESULT fn4 $N $p per-op device time (PROFILED): $(cut -c1-3000 $O/$TAG-$p-op.txt | tr '\n' ' ') | $extra | cards $(thermwin $ta $tb)"
  done < $O/$TAG-win.txt
fi
log "RESULT fn4 $N-fn-prof3: done"
