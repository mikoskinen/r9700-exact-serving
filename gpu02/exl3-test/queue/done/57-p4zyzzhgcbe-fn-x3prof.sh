#!/bin/bash
# 120-fn-x3prof.sh (queue item, 9.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~15 min) -- where
# Flash-Next EXL3 4.05 (TP4 + EP4, the -fnx3 image, 116's e4 line) spends its decode time, by op.
# WHY NOT STAMPS: the AQL stamps time recorded passes only, and the -fnx3 binary records none at decode: its qwen4exp
# pass key (1.17 / PR #266 base) answers 0 for every verify and draft pass (main fixed it, issue #402), so 116's e4 c16
# window issued 63712 passes live and played 0 (iq4r on 1.15.1 / main: ~94 % played) and its card-0 profile came out
# empty. So here --profile-ops: device event pairs around every op (Ctx::dump_profile, a cumulative table a rank every 32
# steps), which needs no recording. CAVEAT (the engine's own): profiled steps synchronise around every launch, so they
# are several times slower -- the op SHARES and us a call are the reading, not tok/s, and launch gaps are not seen.
# One start (no gate: profiled steps fail any ms limit), the warm-up answers, then a c4 and a c16 window (mb, 10 s warm +
# X120_DUR s, default 30) with the engine counters around each; down; tools/opprof.py subtracts each rank's last table
# before a window from its last one inside it -> device ms an op class a rank (EXL3 expert GEMMs + prep, other MoE,
# attention, delta-net, gated residual, GEMMs, collectives, other), the EXL3 ops by op and by layer (us a call: layers of
# another K show as another cluster; 90b's rad-info census names them).
# Power is not touched (caps logged); read-only temperature watcher (junction 100 C / mem 96 C for 10 s -> stop).
# DRYRUN=1 bash 120-fn-x3prof.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41
YN=fn-tp4r.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/opprof.py" ] && [ -e "$f/msnap.py" ] && { TL=$f; break; }; done
FNX3=${X120_IMAGE:-radiance-local:ds41-exl3-117-390664b3-fa0e9580-tuned3-hd4a11cab-fnx3}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-x3prof
export FNIMAGE=$FNX3 FNMODELDIR=${X120_DIR:-$HOME/models/flashnext-exl3} FNMODEL=${X120_MODEL:-qwen3.8-flash-next-exl3.rad}
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=2048 NST=3 WIREKB=128 PMIN="" PMINB=""
export EXTRA="--prefix-resume nearest --profile-ops" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats120 FN4NR="" FN4GI="" FN4FD="" FN4NW=""
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
compose() { docker compose --project-directory $C -f $Y "$@"; }
down() { run "compose down > /dev/null 2>&1; docker rm -f $CN > /dev/null 2>&1"; }
now_line() { docker logs $CN 2>&1 | wc -l; }
caps() { local d h out=""; for d in /sys/class/drm/card*/device; do [ -e $d/unique_id ] || continue; h=$(ls -d $d/hwmon/hwmon* 2>/dev/null | head -1); [ -n "$h" ] || continue
  out="$out $(basename $(readlink -f $d)):cap=$(( $(cat $h/power1_cap 2>/dev/null || echo 0) / 1000000 ))W,hot=$(( $(cat $h/temp2_input 2>/dev/null || echo 0) / 1000 ))C"; done; echo "$out"; }
tguard() { local d h j m n=0 hot
  while [ ! -e $O/$TAG.tstop ]; do hot=""
    for d in /sys/class/drm/card*/device; do [ -e $d/unique_id ] || continue; h=$(ls -d $d/hwmon/hwmon* 2>/dev/null | head -1); [ -n "$h" ] || continue
      j=$(( $(cat $h/temp2_input 2>/dev/null || echo 0) / 1000 )); m=$(( $(cat $h/temp3_input 2>/dev/null || echo 0) / 1000 ))
      { [ $j -ge 100 ] || [ $m -ge 96 ]; } && hot="$hot $(basename $(readlink -f $d)):j$j/m$m"; done
    if [ -n "$hot" ]; then n=$((n + 1)); else n=0; fi
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 120: VIRHE temperature guard ($hot) -> windows stopped"; return 0; }
    sleep 1; done; }
log "RESULT fn4 120-fn-x3prof: start IMAGE=$FNX3 ($(docker image inspect -f '{{.Id}}' $FNX3 2>/dev/null | cut -c8-19)) MODEL=$FNMODEL EXTRA '$EXTRA' (per-op device timing, no stamps) | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 120: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $FNX3 > /dev/null 2>&1 || { log "RESULT fn4 120: VIRHE image $FNX3 missing"; exit 1; }
  [ -f $FNMODELDIR/$FNMODEL ] || { log "RESULT fn4 120: VIRHE $FNMODELDIR/$FNMODEL missing"; exit 1; }
  [ -e "$Y" ] && [ -n "$TL" ] || { log "RESULT fn4 120: VIRHE $YN or tools/opprof.py missing"; exit 1; }
  mkdir -p $O $O/guard $STATSDIR
  find $O -maxdepth 1 -name "$TAG.abort" -delete; find $O -maxdepth 1 -name "$TAG.tstop" -delete; find $O -maxdepth 1 -name "$TAG-win.txt" -delete
  trap 'touch $O/$TAG.tstop 2>/dev/null; down' EXIT
  trap 'exit 143' TERM INT
  tguard & TGPID=$!
else
  echo "+ guards (no radiance container, image, model, $YN, tools); tguard &"; TGPID=""
fi
# start (no gate)
if [ "$DRY" = 1 ]; then echo "+ FNIMAGE=$FNIMAGE EXTRA='$EXTRA' compose -f $Y up -d; health <= 1200 s; docker logs > $O/start-$TAG.log"
else
  down; t0=$(date +%s); compose up -d > $O/up-$TAG.txt 2>&1; ok=0
  for i in $(seq 1 240); do
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' $URL/health)" = 200 ] && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Status}}' $CN 2>/dev/null)" != running ] && break
    sleep 5
  done
  docker logs $CN > $O/start-$TAG.log 2>&1
  [ $ok = 1 ] || { log "RESULT fn4 120: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $O/start-$TAG.log | head -3 | tr '\n' ' ' | cut -c1-400)"; exit 1; }
  log "RESULT fn4 120: healthy in $(( $(date +%s) - t0 )) s; $(grep -m1 -iE 'per-op timing|profile' $O/start-$TAG.log | sed 's/^[A-Z] //' | cut -c1-200)"
fi
run "(cd $D && timeout 600 python3 ds41_tests.py correct --only-q --effort none --outdir $O/guard > $O/guard/$TAG-warm-q.txt 2>&1)"
for p in c4 c16; do
  [ -e $O/$TAG.abort ] && break
  run "timeout 30 python3 -I $TL/msnap.py snap $URL $O/$TAG-$p-msa.json > /dev/null 2>&1"
  if [ "$DRY" = 1 ]; then l0=L0; else l0=$(now_line); fi
  run "(cd $D && timeout 900 python3 mb.py --tag $TAG --phases $p --dur ${X120_DUR:-30} --warm 10 > $O/$TAG-$p.txt 2>&1)"
  run "sleep 3"   # the table the last profiled step wrote
  if [ "$DRY" = 1 ]; then l1=L1; else l1=$(now_line); fi
  run "timeout 30 python3 -I $TL/msnap.py snap $URL $O/$TAG-$p-msb.json > /dev/null 2>&1"
  run "echo '$p $l0 $l1' >> $O/$TAG-win.txt"
  [ "$DRY" = 1 ] || log "RESULT fn4 120 $p (PROFILED, synchronised: not a speed): $(grep '^MB ' $O/$TAG-$p.txt 2>/dev/null | cut -c4- | cut -c1-300) | engine: $(timeout 30 python3 -I $TL/msnap.py delta $O/$TAG-$p-msa.json $O/$TAG-$p-msb.json 2>&1 | cut -c1-500)"
done
run "docker logs $CN > $O/$TAG-engine.log 2>&1"
down
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
if [ "$DRY" = 1 ]; then for p in c4 c16; do echo "+ opprof.py $O/$TAG-engine.log --from L0 --to L1 --label $TAG-$p --json $O/$TAG-$p-op.json"; done
else
  while read -r p l0 l1; do
    timeout 600 python3 -I $TL/opprof.py $O/$TAG-engine.log --from $l0 --to $l1 --label $TAG-$p --json $O/$TAG-$p-op.json > $O/$TAG-$p-op.txt 2>&1
    log "RESULT fn4 120 $p per-op device time: $(cut -c1-3000 $O/$TAG-$p-op.txt | tr '\n' ' ')"
  done < $O/$TAG-win.txt
fi
log "RESULT fn4 120-fn-x3prof: done"
