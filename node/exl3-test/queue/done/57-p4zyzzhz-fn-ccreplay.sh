#!/bin/bash
# 135-fn-ccreplay.sh (queue item, 10.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~60 min) -- the
# cc REPLAY method (coordinator 10.11.): 123 showed cc pairs with MTP on are invalid (outputs are not batch-invariant, so 9
# of 16 users diverged on their first turn and every later prompt was built from the run's own outputs: the same-config
# repeat moved per-stream decode -26.5 % at 9-16 streams, TTFT 2-8k +28 %, acceptance -4.3 %). tools/ccsim/cc_sim3.py:
#   ref  one natural run (16 users, 15 turns, first 3 excluded, trace-seeded, cpt-fixed, temp 0, effort xhigh,medium)
#        with --record: every reply kept in <ref>.outputs.jsonl
#   A1, A2  two --replay runs of ref: every request's messages built from ref's replies (identical prompts request for
#        request; only the run-id line differs, so no cross-run cache hits), issued OPEN-LOOP at ref's start times,
#        max_tokens = ref's length with the natural stop
#   pair  cc_sim3 --pair-replay A1 A2: paired per-request TTFT / decode tok/s / end-to-end (geometric mean B/A, t 95 %),
#        acceptance difference, the prompt-length check, work ratios -> the A/A NOISE FLOOR of the method.
# One server on the main-line stack (132's): radiance-local:c16-sw6, F1n + F2a + F2b + hcw + hc_readv 3 by environment
# (compose fn-tp4u.yaml), MNBT 4096 + --prefill-span 2, --prefill-interleave 0, --prefix-resume nearest; TP4 all_vram MML
# 262144 MAXSEQS 16 MTP 3 wht6. If the floor comes out around +-1 %, TESTING.md gets the method and 126b is re-queued as
# replay pairs (coordinator). Power untouched; read-only temperature watcher (junction 100 C / mem 96 C for 10 s -> stop).
# DRYRUN=1 bash 135-fn-ccreplay.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4u.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/abblocks2.py" ] && [ -e "$f/heatsoak.sh" ] && { TL=$f; break; }; done
IMG=${X135_IMAGE:-radiance-local:c16-sw6}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-ccrep
W8=${X135_W8:-4096x2560 2560x1536 3072x2560 256x2560}
export FNIMAGE=$IMG FNMODELDIR=$HOME/models/flashnext-iq4r FNMODEL=qwen3.8-next-flash-fp8-iq4r-moe.rad
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
export EXTRA="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats135
export FN4NR=1 FN4GI=1 FN4FD="" FN4NW=1 FN4SW="" FN4WV=1 FN4RV=3 FN4AL=""
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
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 135: VIRHE temperature guard ($hot) -> blocks stopped"; return 0; }
    sleep 1; done; }
restart() {   # restart TAG -> 0 healthy
  local tag=$1 ok=0 t0 i S=$O/start-$1.log
  [ "$DRY" = 1 ] && { echo "+ FNIMAGE=$FNIMAGE FN4NR=$FN4NR FN4GI=$FN4GI FN4NW=$FN4NW FN4WV=$FN4WV FN4RV=$FN4RV EXTRA='$EXTRA' compose -f $Y up -d; health <= 1200 s; docker logs > $S"; return 0; }
  down; t0=$(date +%s); compose up -d > $O/up-$tag.txt 2>&1
  for i in $(seq 1 240); do
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' $URL/health)" = 200 ] && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Status}}' $CN 2>/dev/null)" != running ] && break
    sleep 5
  done
  docker logs $CN > $S 2>&1
  [ $ok = 1 ] || { log "RESULT fn4 135 restart-$tag: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
  log "RESULT fn4 135 restart-$tag: healthy in $(( $(date +%s) - t0 )) s; $(grep -m1 -E '^I kv: ' $S | cut -c3-160); fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300)"
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
cmpf() {   # cmpf FILE_A FILE_B LABEL
  if [ "$DRY" = 1 ]; then echo "+ cmp $1 $2"
  elif [ -s $1 ] && [ -s $2 ]; then
    if cmp -s $1 $2; then log "RESULT fn4 135 identity $3: IDENTICAL ($(wc -c < $2) bytes)"
    else log "RESULT fn4 135 identity $3: VIRHE DIFFERENT from byte $(cmp $1 $2 2>&1 | grep -oE 'byte [0-9]+' | head -1)"; fi
  else log "RESULT fn4 135 identity $3: not compared (missing $1 or $2)"; fi
}
warmcmp() {   # warmcmp FILE_A FILE_B LABEL (informational)
  [ "$DRY" = 1 ] && { echo "+ warm-up answers $1 vs $2"; return 0; }
  [ -s $1 ] && [ -s $2 ] || { log "RESULT fn4 135 warm-up $3: not compared (missing $1 or $2)"; return 0; }
  log "RESULT fn4 135 warm-up $3: $(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if t in a and ((a[t].get("content") or "") != (b[t].get("content") or "") or a[t].get("finish") != b[t].get("finish"))]
n = len([t for t in b if t in a])
print(("IDENTICAL: %d answers" % n) if n and not bad else ("DIFFERENT: " + ", ".join(bad) if n else "no common answers"))' $1 $2 2>&1 | cut -c1-300)"
}
abrun() {   # abrun NAME PHASE ROUNDS A_JSON B_JSON [PF_DEPTHS]
  local nm=$1 ph=$2 rr=$3 pf=${6:-}
  [ -e $O/$TAG.abort ] && return 0
  run "timeout 5400 python3 -I $TL/abblocks2.py --url $URL --mb-dir $D --label $TAG-$nm-$ph --phase $ph --dur ${X135_DUR:-20} --warm ${X135_WARM:-6} --rounds $rr --a '$4' --b '$5' ${pf:+--pf-depths $pf} --names plain,$nm --out $O/$TAG-blocks.jsonl > $O/$TAG-$nm-$ph.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 135 $nm $ph${pf:+ + prefill $pf}: $(grep '^ABB ' $O/$TAG-$nm-$ph.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=4: "-" if v is None else format(v, ".%df" % n)
s = "decode tok/s B/A %s +- %s (t95, %d rounds: %s) | ms/step B/A %s +- %s | A %s B %s tok/s | %d blocks, %d flagged, %s s" % (
    f(d["tps_ratio"]), f(d["tps_ci95"]), len(d["rounds"]), " ".join(f(r["tps_ratio"]) for r in d["rounds"]),
    f(d["ms_ratio"]), f(d["ms_ci95"]), f(d["A_tps_mean"], 1), f(d["B_tps_mean"], 1), d["blocks"], d["blocks_bad"], d["wall_s"])
for k, v in (d.get("pf") or {}).items():
    s += " | prefill %s B/A %s +- %s (%s) A %s B %s tok/s" % (k, f(v["ratio"]), f(v["ci95"]), " ".join(f(x) for x in v["rounds"]), f(v["A_mean"], 1), f(v["B_mean"], 1))
print(s)' 2>&1 | cut -c1-1200)"
  [ "$DRY" = 1 ] || [ -z "$pf" ] || log "RESULT fn4 135 $nm $ph prefill windows' thermal lines: $(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("label") == sys.argv[2] and r.get("pf_therm"): print("r%d%s %s;" % (r["round"], r["arm"], r["pf_therm"]), end=" ")' $O/$TAG-blocks.jsonl $TAG-$nm-$ph 2>&1 | cut -c1-1500)"
}
CC3=$HOME/gpu/ds41/exl3-test/queue-ar/tools/ccsim/cc_sim3.py; CCW=$HOME/gpu/ds41/exl3-test/ccsim-exl3-turns.sh
CCBASE="--duration 3000 --turns ${X135_TURNS:-15} --warmup-turns 3 --effort xhigh,medium --seed 1 --cpt-fixed --temperature 0 --trace-seeded"
ccone() {   # ccone NAME EXTRA_ARGS -> sets OUTF (the run's jsonl)
  local nm=$1; shift
  if [ "$DRY" = 1 ]; then echo "+ CCARGS='$CCBASE $*' CCSIM=$CC3 timeout 3400 $CCW ${X135_USERS:-16} $TAG-$nm"; OUTF=$O/dry-$nm.jsonl; return 0; fi
  CCARGS="$CCBASE $*" CCSIM=$CC3 timeout 3400 $CCW ${X135_USERS:-16} $TAG-$nm > $O/$TAG-$nm-cc.out 2>&1
  OUTF=$(grep -oE 'CCSIM DONE \S+' $O/$TAG-$nm-cc.out | tail -1 | cut -d' ' -f3).jsonl
  [ -s "$OUTF" ] || { log "RESULT fn4 135 $nm: VIRHE no cc_sim output ($O/$TAG-$nm-cc.out: $(tail -3 $O/$TAG-$nm-cc.out | tr '\n' ' ' | cut -c1-300))"; return 1; }
  log "RESULT fn4 135 $nm: $OUTF | $(grep -E '^SUMMARY window|requests [0-9]+ ok|TTFT p50' $(dirname $OUTF)/$(basename $OUTF .jsonl).txt 2>/dev/null | tr -s ' ' | tr '\n' '|' | cut -c1-500)"
}
log "RESULT fn4 135-fn-ccreplay: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) the 132 stack, cc_sim3 $( [ "$DRY" = 1 ] || md5sum $CC3 | cut -c1-8) | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 135: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 135: VIRHE image $IMG missing"; exit 1; }
  [ -e "$Y" ] && grep -q RADIANCE_FN4_HC_READV $Y && [ -e $CC3 ] && [ -e $CCW ] || { log "RESULT fn4 135: VIRHE $YN, cc_sim3.py or the cc wrapper missing"; exit 1; }
  mkdir -p $O $O/guard $STATSDIR
  find $O -maxdepth 1 -name "$TAG.abort" -delete; find $O -maxdepth 1 -name "$TAG.tstop" -delete
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
fi
tries=0
while :; do
  set -- $(probe8 $TAG-st$tries); ms=$1
  st=$(awk -v m=$ms 'BEGIN{print (m <= 0) ? "unknown" : (m <= 26.8) ? "fast" : (m >= 28.5) ? "slow" : "between"}')
  log "RESULT fn4 135 state probe $tries: mb c8 $ms ms -> $st"
  { [ "$st" = slow ] && [ $tries -lt 2 ]; } || break
  tries=$((tries + 1)); restart $TAG-r$tries || exit 1
done
ccone ref --gen-mode natural --record || exit 1; REFJ=$OUTF
[ "$DRY" = 1 ] || [ -s ${REFJ%.jsonl}.outputs.jsonl ] || { log "RESULT fn4 135: VIRHE the reference wrote no outputs file"; exit 1; }
ccone a1 --replay $REFJ || exit 1; A1=$OUTF
ccone a2 --replay $REFJ || exit 1; A2=$OUTF
if [ "$DRY" = 1 ]; then echo "+ python3 $CC3 --pair-replay $A1 $A2"
else
  python3 $CC3 --pair-replay $A1 $A2 > $O/$TAG-pair.txt 2>&1
  log "RESULT fn4 135 A/A replay pair (the method's noise floor): $(grep '^PAIR_REPLAY' $O/$TAG-pair.txt | cut -c13- | cut -c1-1500)"
fi
run "docker logs $CN > $O/$TAG-engine.log 2>&1"
down
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
log "RESULT fn4 135-fn-ccreplay: done | ref $REFJ, A1 $A1, A2 $A2"
