#!/bin/bash
# 125-fn-kawake.sh (queue item, 9.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~20 min) -- does
# the window-wide keepawake poller (tools/keepawake.py, gpu_metrics every 5 ms, ~600 SMU reads a second) cost prefill or
# decode now that the host tuning (C2 off + governor performance) already prevents the slow state?
# Hint: 95's pipelined m4 (no poller, no host tuning) 54k / 120k prefill 8507.7 / 8391.1 tok/s; 99's m4a and 106's m4n
# (poller on) 8257 / 8170 and 8209 / 8138 (-3 %).
# ONE server on the k4n line (1.15.1 stampcards c0cbe5a + --prefix-resume nearest; TP4 all_vram MML 262144 MAXSEQS 16
# MNBT 2048 MTP 3), no engine restart between blocks: the poller is the switch -- A = on (as found), B = off (its pid
# killed); it is started again with the identical command line it was found with (/proc/<pid>/cmdline) and its pid
# written back to runs/keepawake.pid. Blocks in the order A B, B A, A B (X125_ROUNDS rounds, default 3), each: 3 s
# settle, prefill 16k + fresh 64k / 128k (ds41_tests prefill, a nonce first: every prompt cold), mb c16 60 s (10 s warm).
# Reading: per round B/A of prefill tok/s at each depth and of c16 agg_dec_tps; mean and spread over rounds.
# WINDOW-WIDE STATE: the poller runs again at the end and in the EXIT trap (with the found command line, or tools/
# keepawake.py's window default when none was running at the start -- logged).
# Power is not touched (caps logged); read-only temperature watcher (junction 100 C / mem 96 C for 10 s -> stop).
# DRYRUN=1 bash 125-fn-kawake.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4r.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/keepawake.py" ] && { TL=$f; break; }; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
IMG=${X125_IMAGE:-radiance-local:1.15.1-stampcards-c0cbe5a}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-kawake; KPID=$R/keepawake.pid
export FNIMAGE=$IMG FNMODELDIR=$HOME/models/flashnext-iq4r FNMODEL=qwen3.8-next-flash-fp8-iq4r-moe.rad
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=2048 NST=3 WIREKB=128 PMIN="" PMINB=""
export EXTRA="--prefix-resume nearest" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats125 FN4NR="" FN4GI="" FN4FD="" FN4NW=""
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
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 125: VIRHE temperature guard ($hot) -> blocks stopped"; return 0; }
    sleep 1; done; }
# ---- the poller
KCMD=()   # the command line it was found with
ka_alive() { local p; [ -s $KPID ] || return 1; p=$(cat $KPID); kill -0 "$p" 2>/dev/null && tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q keepawake.py; }
ka_stop() {
  [ "$DRY" = 1 ] && { echo "+ kill the poller (pid in $KPID)"; return 0; }
  ka_alive || return 0
  local p; p=$(cat $KPID); kill $p 2>/dev/null
  for i in $(seq 1 30); do kill -0 $p 2>/dev/null || break; sleep 0.1; done
  kill -0 $p 2>/dev/null && kill -9 $p 2>/dev/null
  return 0
}
ka_start() {
  [ "$DRY" = 1 ] && { echo "+ setsid nohup <the found keepawake command line> & -> $KPID"; return 0; }
  ka_alive && return 0
  setsid nohup "${KCMD[@]}" > /dev/null 2>&1 < /dev/null &
  echo $! > $KPID; sleep 1
  ka_alive || log "RESULT fn4 125: VIRHE the poller did not start again: ${KCMD[*]}"
}
pf_once() {   # pf_once NAME: fresh 16k, then fresh 64k + 128k -> "tps16 tps64 tps128"
  if [ "$DRY" = 1 ]; then echo "+ (cd $D && ds41_tests.py prefill --depths 16384 ...; --depths 65536,131072 ...) -> $O/$TAG-$1-pf*.txt" >&2; echo "0 0 0"; return 0; fi
  (cd $D && timeout 900 python3 ds41_tests.py prefill --depths 16384 --cpt 3.6 --calibrate 0 --reps 1 > $O/$TAG-$1-pf16.txt 2>&1)
  (cd $D && timeout 1500 python3 ds41_tests.py prefill --depths 65536,131072 --cpt 3.6 --calibrate 0 --reps 1 > $O/$TAG-$1-pf64-128.txt 2>&1)
  cat $O/$TAG-$1-pf16.txt $O/$TAG-$1-pf64-128.txt | python3 -c 'import json,sys
v = []
for l in sys.stdin:
    if l.startswith("J "):
        try: v.append(json.loads(l[2:]).get("prefill_tps"))
        except ValueError: pass
v = [x for x in v if x] + [0, 0, 0]
print(v[0], v[1], v[2])' 2>/dev/null || echo "0 0 0"
}
mb_once() {   # mb_once NAME -> "agg_dec_tps ms_decode_step"
  if [ "$DRY" = 1 ]; then echo "+ (cd $D && mb.py --tag $TAG-$1 --phases c16 --dur 60 --warm 10) -> $O/$TAG-$1-c16.txt" >&2; echo "0 0"; return 0; fi
  (cd $D && timeout 900 python3 mb.py --tag $TAG-$1 --phases c16 --dur ${X125_DUR:-60} --warm 10 > $O/$TAG-$1-c16.txt 2>&1)
  grep '^MB ' $O/$TAG-$1-c16.txt 2>/dev/null | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[3:]); print(d.get("agg_dec_tps"), d.get("ms_decode_step"))' 2>/dev/null || echo "0 0"
}
log "RESULT fn4 125-fn-kawake: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) EXTRA '$EXTRA' | poller: $( [ "$DRY" = 1 ] && echo dry || { ka_alive && echo "pid $(cat $KPID): $(tr '\0' ' ' < /proc/$(cat $KPID)/cmdline | cut -c1-200)" || echo "none running"; }) | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 125: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 125: VIRHE image $IMG missing"; exit 1; }
  [ -e "$Y" ] && [ -n "$TL" ] || { log "RESULT fn4 125: VIRHE $YN or tools/keepawake.py missing"; exit 1; }
  if ka_alive; then mapfile -d '' -t KCMD < /proc/$(cat $KPID)/cmdline
  else
    KCMD=(python3 -I $TL/keepawake.py --interval-ms 5 --node gpu_metrics --cards all --log $HOME/gpu/watch/keepawake.log)
    log "RESULT fn4 125: no window-wide poller was running -- starting the window default (${KCMD[*]}) as the on state; it stays on at the end"
    ka_start
  fi
  mkdir -p $O $O/guard $STATSDIR
  find $O -maxdepth 1 -name "$TAG.abort" -delete; find $O -maxdepth 1 -name "$TAG.tstop" -delete
  trap 'touch $O/$TAG.tstop 2>/dev/null; ka_start; log "RESULT fn4 125 exit: poller $(ka_alive && echo "running, pid $(cat $KPID)" || echo "NOT running")"; down' EXIT
  trap 'exit 143' TERM INT
  tguard & TGPID=$!
else
  echo "+ guards; save the poller's command line; trap: start the poller again + down; tguard &"; TGPID=""
fi
# ---- the server (no restart between blocks), the gate and the slow-state probe with the poller on
if [ "$DRY" = 1 ]; then echo "+ FNIMAGE=$FNIMAGE EXTRA='$EXTRA' compose -f $Y up -d; health <= 1200 s"
else
  down; t0=$(date +%s); compose up -d > $O/up-$TAG.txt 2>&1; ok=0
  for i in $(seq 1 240); do
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' $URL/health)" = 200 ] && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Status}}' $CN 2>/dev/null)" != running ] && break
    sleep 5
  done
  docker logs $CN > $O/start-$TAG.log 2>&1
  [ $ok = 1 ] || { log "RESULT fn4 125: VIRHE not healthy after $(( $(date +%s) - t0 )) s"; exit 1; }
  (cd $D && timeout 600 python3 ds41_tests.py correct --only-q --effort none --outdir $O/guard > $O/guard/$TAG-warm-q.txt 2>&1)
  (cd $D && timeout 600 python3 ds41_tests.py prefill --depths 16384 --cpt 3.6 --calibrate 0 > $O/guard/$TAG-warm-pf.txt 2>&1)
  timeout 300 python3 $Q/prodfc.py $URL 384 > $O/guard/$TAG-fc1.txt 2>&1
  log "RESULT fn4 125: healthy in $(( $(date +%s) - t0 )) s; 1-user $(grep '^FC ' $O/guard/$TAG-fc1.txt | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()[3:])["ms_per_step"])' 2>/dev/null || echo n/a) ms/step"
fi
ROUNDS=${X125_ROUNDS:-3}
RES=$O/$TAG-blocks.txt; run ": > $RES"
for r in $(seq 1 $ROUNDS); do
  order="A B"; [ $((r % 2)) = 0 ] && order="B A"
  for arm in $order; do
    [ -e $O/$TAG.abort ] && break 2
    if [ $arm = A ]; then ka_start; else ka_stop; fi
    run "sleep 3"
    st=$( [ "$DRY" = 1 ] && echo dry || { ka_alive && echo on || echo off; })
    set -- $(pf_once r$r$arm); p16=$1 p64=$2 p128=$3
    set -- $(mb_once r$r$arm); tps=$1 ms=$2
    run "echo 'r$r $arm $st $p16 $p64 $p128 $tps $ms' >> $RES"
    [ "$DRY" = 1 ] || log "RESULT fn4 125 round $r $arm (poller $st): prefill 16k / 64k / 128k $p16 / $p64 / $p128 tok/s, mb c16 $tps tok/s $ms ms/step"
  done
done
ka_start
[ "$DRY" = 1 ] || log "RESULT fn4 125 B (poller off) / A (on): $(python3 - $RES <<'PY' 2>&1 | cut -c1-900
import statistics, sys
rows = [l.split() for l in open(sys.argv[1]) if l.strip()]
by = {}
for r, arm, st, *v in rows:
    by.setdefault(r, {})[arm] = [float(x) if x not in ("None", "") else 0.0 for x in v]
names = ["prefill 16k", "prefill 64k", "prefill 128k", "mb c16 tok/s", "mb c16 ms/step"]
out = []
for i, n in enumerate(names):
    rat = [d["B"][i] / d["A"][i] for d in by.values() if "A" in d and "B" in d and d["A"][i] > 0 and d["B"][i] > 0]
    if not rat: out.append(f"{n}: -"); continue
    m = statistics.mean(rat); sd = statistics.stdev(rat) if len(rat) > 1 else 0.0
    out.append(f"{n} {m:.4f} (sd {sd:.4f}, rounds {' '.join(format(x, '.4f') for x in rat)})")
print("; ".join(out))
PY
)"
run "docker logs $CN > $O/$TAG-engine.log 2>&1"
down
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
[ "$DRY" = 1 ] || log "RESULT fn4 125-fn-kawake: done -- poller $(ka_alive && echo "running again, pid $(cat $KPID)" || echo "NOT running")"
