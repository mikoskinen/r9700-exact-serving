#!/bin/bash
# 127-fn-pfprof.sh (queue item, 9.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~20 min) -- where
# a Flash-Next TP4 PREFILL step spends its time, per rank and op class, so the gap to the user's prefill target (>= 8500
# tok/s on warm cards; main + pipelined prefill measured by 126) is ranked by measured ms.
# CONFIG: 126's mpB (radiance-local:c16-sw, F1n on, --prefill-interleave 0 --prefix-resume nearest, MNBT 4096 +
# --prefill-span 2) when it held decode there -- mpB's mb c8 and c16 within 1.5 % of mpA's (126's files) -- else 126's
# mpA (MNBT 2048); the choice and its numbers are logged. Plus --profile-ops: device event pairs around every op, and
# each op's lane wait (stream waits and issue delays before it started, in no op's time), a cumulative table a rank
# every RADIANCE_PROFILE_EVERY (4) steps (compose fn-tp4t.yaml). Profiled steps synchronise: the op SHARES, us a call and
# lane waits are the reading, not tok/s; the unprofiled prefill rate is 126's.
# PHASES (TESTING.md 9.11.: decode before prefill after a start, no soak time): start; mb c16 30 s (warms the cards);
# window 1: four fresh 16k prompts back to back; window 2: one fresh 128k prompt; stop (the shutdown table closes window
# 2). tools/opprof.py subtracts each rank's tables around a window: device ms by class (MoE experts, MoE other, GEMMs,
# attention, delta-net, gated residual, collectives, other) a rank, lane waits, the top ops; thermwin (junction +
# full-power sclk) beside each window; 126's unprofiled TTFT at the same depths beside it.
# Power is not touched; read-only temperature watcher (junction 100 C / mem 96 C for 10 s -> stop).
# DRYRUN=1 bash 127-fn-pfprof.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41
YN=fn-tp4t.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/opprof.py" ] && [ -e "$f/heatsoak.sh" ] && { TL=$f; break; }; done
[ -n "$TL" ] && . $TL/heatsoak.sh   # thermnow / thermwin only
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
IMG=${X127_IMAGE:-radiance-local:c16-sw}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-pfprof
export FNIMAGE=$IMG FNMODELDIR=$HOME/models/flashnext-iq4r FNMODEL=qwen3.8-next-flash-fp8-iq4r-moe.rad
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 NST=3 WIREKB=128 PMIN="" PMINB=""
export AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats127 FN4NR="" FN4GI="" FN4FD="" FN4NW=1 PROFEVERY=${X127_EVERY:-4}
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
compose() { docker compose --project-directory $C -f $Y "$@"; }
down() { run "compose down > /dev/null 2>&1; docker rm -f $CN > /dev/null 2>&1"; }
tguard() { local d h j m n=0 hot
  while [ ! -e $O/$TAG.tstop ]; do hot=""
    for d in /sys/class/drm/card*/device; do [ -e $d/unique_id ] || continue; h=$(ls -d $d/hwmon/hwmon* 2>/dev/null | head -1); [ -n "$h" ] || continue
      j=$(( $(cat $h/temp2_input 2>/dev/null || echo 0) / 1000 )); m=$(( $(cat $h/temp3_input 2>/dev/null || echo 0) / 1000 ))
      { [ $j -ge 100 ] || [ $m -ge 96 ]; } && hot="$hot $(basename $(readlink -f $d)):j$j/m$m"; done
    if [ -n "$hot" ]; then n=$((n + 1)); else n=0; fi
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 127: VIRHE temperature guard ($hot) -> stopped"; return 0; }
    sleep 1; done; }
mbv() {   # mbv FILE: "agg_dec_tps"
  grep '^MB ' $1 2>/dev/null | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()[3:]).get("agg_dec_tps") or 0)' 2>/dev/null || echo 0
}
# ---- 126's verdict: mpB held decode?
CFG=mpB; WHY="dry"
if [ "$DRY" != 1 ]; then
  a8=$(mbv $O/fn4-mpA-c8.txt); b8=$(mbv $O/fn4-mpB-c8.txt); a16=$(mbv $O/fn4-mpA-c16.txt); b16=$(mbv $O/fn4-mpB-c16.txt)
  WHY="126 mb c8 mpA $a8 / mpB $b8, c16 mpA $a16 / mpB $b16"
  if python3 -c "import sys; a8,b8,a16,b16=map(float,sys.argv[1:]); sys.exit(0 if a8>0 and a16>0 and b8>=0.985*a8 and b16>=0.985*a16 else 1)" $a8 $b8 $a16 $b16; then CFG=mpB
  else CFG=mpA; WHY="$WHY -> mpB did not hold decode (or 126 has no numbers): profiling mpA's config"; fi
fi
if [ $CFG = mpB ]; then export MNBT=4096 EXTRA="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0 --profile-ops"
else export MNBT=2048 EXTRA="--prefix-resume nearest --prefill-interleave 0 --profile-ops"; fi
log "RESULT fn4 127-fn-pfprof: config $CFG ($WHY) IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) MNBT $MNBT EXTRA '$EXTRA' RADIANCE_PROFILE_EVERY=$PROFEVERY"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 127: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 127: VIRHE image $IMG missing"; exit 1; }
  [ -e "$Y" ] && grep -q RADIANCE_PROFILE_EVERY $Y && [ -n "$TL" ] || { log "RESULT fn4 127: VIRHE $YN (with RADIANCE_PROFILE_EVERY) or tools/opprof.py / heatsoak.sh missing"; exit 1; }
  mkdir -p $O $STATSDIR
  find $O -maxdepth 1 -name "$TAG.abort" -delete; find $O -maxdepth 1 -name "$TAG.tstop" -delete; find $O -maxdepth 1 -name "$TAG-win.txt" -delete
  trap 'touch $O/$TAG.tstop 2>/dev/null; down' EXIT
  trap 'exit 143' TERM INT
  tguard & TGPID=$!
else
  echo "+ guards; trap: down; tguard &"; TGPID=""
fi
if [ "$DRY" = 1 ]; then echo "+ FNIMAGE=$FNIMAGE MNBT=$MNBT EXTRA='$EXTRA' PROFEVERY=$PROFEVERY compose -f $Y up -d; health <= 1200 s"
else
  down; t0=$(date +%s); compose up -d > $O/up-$TAG.txt 2>&1; ok=0
  for i in $(seq 1 240); do
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' $URL/health)" = 200 ] && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Status}}' $CN 2>/dev/null)" != running ] && break
    sleep 5
  done
  docker logs $CN > $O/start-$TAG.log 2>&1
  [ $ok = 1 ] || { log "RESULT fn4 127: VIRHE not healthy after $(( $(date +%s) - t0 )) s"; exit 1; }
  log "RESULT fn4 127: healthy in $(( $(date +%s) - t0 )) s; $(grep -m1 -E '^I kv: ' $O/start-$TAG.log | cut -c3-160)"
fi
nl() { [ "$DRY" = 1 ] && echo L || docker logs $CN 2>&1 | wc -l; }
run "(cd $D && timeout 600 python3 mb.py --tag $TAG-warm --phases c16 --dur 30 --warm 5 > $O/$TAG-warm-c16.txt 2>&1)"
l0=$(nl); t0=$(date +%s)
for k in 1 2 3 4; do run "(cd $D && timeout 900 python3 ds41_tests.py prefill --depths 16384 --cpt 3.6 --calibrate 0 --reps 1 > $O/$TAG-pf16-$k.txt 2>&1)"; done
run "sleep 3"; l1=$(nl); t1=$(date +%s)
run "(cd $D && timeout 1500 python3 ds41_tests.py prefill --depths 131072 --cpt 3.6 --calibrate 0 --reps 1 > $O/$TAG-pf128.txt 2>&1)"
t2=$(date +%s)
# stop first (the shutdown table closes window 2), read the whole log, then remove
run "docker stop -t 60 $CN > /dev/null 2>&1; docker logs $CN > $O/$TAG-engine.log 2>&1"
down
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
if [ "$DRY" = 1 ]; then echo "+ opprof.py $O/$TAG-engine.log --from L0 --to L1 (16k x4) / --from L1 (128k, to the shutdown table)"
else
  ttft() { cat "$@" 2>/dev/null | python3 -c 'import json,sys
v = [json.loads(l[2:]) for l in sys.stdin if l.startswith("J ")]
print("; ".join("%s tok TTFT %s s %s tok/s" % (d.get("prompt_tokens"), d.get("ttft"), d.get("prefill_tps")) for d in v) or "none")' 2>/dev/null; }
  timeout 600 python3 -I $TL/opprof.py $O/$TAG-engine.log --from $l0 --to $l1 --label $TAG-16k --json $O/$TAG-16k-op.json > $O/$TAG-16k-op.txt 2>&1
  log "RESULT fn4 127 16k x4 ($CFG, PROFILED): $(cut -c1-2600 $O/$TAG-16k-op.txt | tr '\n' ' ') | profiled prompts: $(ttft $O/$TAG-pf16-*.txt | cut -c1-300) | cards $(thermwin $t0 $t1) | 126 unprofiled $CFG 16k: $(ttft $O/fn4-$CFG-pf16.txt | cut -c1-200)"
  timeout 600 python3 -I $TL/opprof.py $O/$TAG-engine.log --from $l1 --label $TAG-128k --json $O/$TAG-128k-op.json > $O/$TAG-128k-op.txt 2>&1
  log "RESULT fn4 127 128k ($CFG, PROFILED): $(cut -c1-2600 $O/$TAG-128k-op.txt | tr '\n' ' ') | profiled prompt: $(ttft $O/$TAG-pf128.txt | cut -c1-200) | cards $(thermwin $t1 $t2) | 126 unprofiled $CFG 64k/128k: $(ttft $O/fn4-$CFG-pf64-128.txt | cut -c1-200)"
fi
log "RESULT fn4 127-fn-pfprof: done"
