#!/bin/bash
# 141-fn-replay-closed.sh (queue item, 10.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~75 min, three
# arms) -- the REAL-LOAD check of the final stack against the 1.15.1 k4n line, CLOSED-LOOP (coordinator 10.11., after 139):
# 139's open-loop pair was a saturation artefact (TESTING.md 10.11. replay validity: k4n fell behind the stack-recorded
# arrivals, 170 of 219 turns overlapped their previous reply, 2.57x the uncached tokens). Here cc_sim3 --replay-closed (md5
# 2512bb16): 135's reference prompts request for request, each user's next request when THIS run's previous reply ended plus
# the reference's gap. Arms in A B A order, each on a fresh server:
#   stack (a1, a2)  radiance-local:c16-sw8 (137's stack): F1n + F2a + F2b + hcw + hc_readv 3 + hc_upmix 5, MNBT 4096 +
#                   --prefill-span 2, --prefill-interleave 0, --prefix-resume nearest
#   k4n             radiance-local:1.15.1-stampcards-c0cbe5a, MNBT 2048, --prefix-resume nearest
# Both TP4 all_vram MML 262144 MAXSEQS 16 MTP 3 wht6 210 W, compose fn-tp4v.yaml, no AQL stamps. Per arm: restart, warm-up,
# slow-state probe, the closed-loop replay, engine counters. Pairs: a1 vs a2 (the closed-loop A/A floor, read first), then
# stack vs k4n for a1 and a2 (TTFT all / by uncached bucket, decode tok/s, acceptance, makespan, validity). Power untouched;
# read-only temperature watcher.
# DRYRUN=1 bash 141-fn-replay-closed.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4v.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/abblocks2.py" ] && [ -e "$f/heatsoak.sh" ] && { TL=$f; break; }; done
IMG=${X141_IMAGE:-radiance-local:c16-sw8}; IMGK=${X141_IMAGE_K4N:-radiance-local:1.15.1-stampcards-c0cbe5a}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-rpcl
W8=${X141_W8:-4096x2560 2560x1536 3072x2560 256x2560}
export FNIMAGE=$IMG FNMODELDIR=$HOME/models/flashnext-iq4r FNMODEL=qwen3.8-next-flash-fp8-iq4r-moe.rad
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
export EXTRA="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats141
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
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 141: VIRHE temperature guard ($hot) -> blocks stopped"; return 0; }
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
  [ $ok = 1 ] || { log "RESULT fn4 141 restart-$tag: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
  log "RESULT fn4 141 restart-$tag: healthy in $(( $(date +%s) - t0 )) s; $(grep -m1 -E '^I kv: ' $S | cut -c3-160); fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300)"
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
    if cmp -s $1 $2; then log "RESULT fn4 141 identity $3: IDENTICAL ($(wc -c < $2) bytes)"
    else log "RESULT fn4 141 identity $3: VIRHE DIFFERENT from byte $(cmp $1 $2 2>&1 | grep -oE 'byte [0-9]+' | head -1)"; fi
  else log "RESULT fn4 141 identity $3: not compared (missing $1 or $2)"; fi
}
warmcmp() {   # warmcmp FILE_A FILE_B LABEL (informational)
  [ "$DRY" = 1 ] && { echo "+ warm-up answers $1 vs $2"; return 0; }
  [ -s $1 ] && [ -s $2 ] || { log "RESULT fn4 141 warm-up $3: not compared (missing $1 or $2)"; return 0; }
  log "RESULT fn4 141 warm-up $3: $(python3 -c 'import json,sys
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
  run "timeout 5400 python3 -I $TL/abblocks2.py --url $URL --mb-dir $D --label $TAG-$nm-$ph --phase $ph --dur ${X141_DUR:-20} --warm ${X141_WARM:-6} --rounds $rr --a '$4' --b '$5' ${pf:+--pf-depths $pf} --names plain,$nm --out $O/$TAG-blocks.jsonl > $O/$TAG-$nm-$ph.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 141 $nm $ph${pf:+ + prefill $pf}: $(grep '^ABB ' $O/$TAG-$nm-$ph.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=4: "-" if v is None else format(v, ".%df" % n)
s = "decode tok/s B/A %s +- %s (t95, %d rounds: %s) | ms/step B/A %s +- %s | A %s B %s tok/s | %d blocks, %d flagged, %s s" % (
    f(d["tps_ratio"]), f(d["tps_ci95"]), len(d["rounds"]), " ".join(f(r["tps_ratio"]) for r in d["rounds"]),
    f(d["ms_ratio"]), f(d["ms_ci95"]), f(d["A_tps_mean"], 1), f(d["B_tps_mean"], 1), d["blocks"], d["blocks_bad"], d["wall_s"])
for k, v in (d.get("pf") or {}).items():
    s += " | prefill %s B/A %s +- %s (%s) A %s B %s tok/s" % (k, f(v["ratio"]), f(v["ci95"]), " ".join(f(x) for x in v["rounds"]), f(v["A_mean"], 1), f(v["B_mean"], 1))
print(s)' 2>&1 | cut -c1-1200)"
  [ "$DRY" = 1 ] || [ -z "$pf" ] || log "RESULT fn4 141 $nm $ph prefill windows' thermal lines: $(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("label") == sys.argv[2] and r.get("pf_therm"): print("r%d%s %s;" % (r["round"], r["arm"], r["pf_therm"]), end=" ")' $O/$TAG-blocks.jsonl $TAG-$nm-$ph 2>&1 | cut -c1-1500)"
}
CC3=$HOME/gpu/ds41/exl3-test/queue-ar/tools/ccsim/cc_sim3.py; CCW=$HOME/gpu/ds41/exl3-test/ccsim-exl3-turns.sh
CCBASE="--duration 3000 --turns ${X141_TURNS:-15} --warmup-turns 3 --effort xhigh,medium --seed 1 --cpt-fixed --temperature 0 --trace-seeded"
ccone() {   # ccone NAME EXTRA_ARGS -> sets OUTF (the run's jsonl)
  local nm=$1; shift
  if [ "$DRY" = 1 ]; then echo "+ CCARGS='$CCBASE $*' CCSIM=$CC3 timeout 3400 $CCW ${X141_USERS:-16} $TAG-$nm"; OUTF=$O/dry-$nm.jsonl; return 0; fi
  CCARGS="$CCBASE $*" CCSIM=$CC3 timeout 3400 $CCW ${X141_USERS:-16} $TAG-$nm > $O/$TAG-$nm-cc.out 2>&1
  OUTF=$(grep -oE 'CCSIM DONE \S+' $O/$TAG-$nm-cc.out | tail -1 | cut -d' ' -f3).jsonl
  [ -s "$OUTF" ] || { log "RESULT fn4 141 $nm: VIRHE no cc_sim output ($O/$TAG-$nm-cc.out: $(tail -3 $O/$TAG-$nm-cc.out | tr '\n' ' ' | cut -c1-300))"; return 1; }
  log "RESULT fn4 141 $nm: $OUTF | $(grep -E '^SUMMARY window|requests [0-9]+ ok|TTFT p50' $(dirname $OUTF)/$(basename $OUTF .jsonl).txt 2>/dev/null | tr -s ' ' | tr '\n' '|' | cut -c1-500)"
}
REFJ=${X141_REF:-$O/../ccsim-1110-0115-fn4-ccrep-ref-u16.jsonl}
[ -e "$REFJ" ] || REFJ=$R/ccsim-1110-0115-fn4-ccrep-ref-u16.jsonl
RV=3   # 138: hc_readv bit 2 lost; the stack is 137's
log "RESULT fn4 141-fn-replay-closed: start reference $REFJ | k4n $IMGK ($(docker image inspect -f '{{.Id}}' $IMGK 2>/dev/null | cut -c8-19)) vs stack $IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) hc_readv $RV + hc_upmix 5 | cc_sim3 $( [ "$DRY" = 1 ] || md5sum $CC3 | cut -c1-8) | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 141: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  for i in $IMG $IMGK; do docker image inspect $i > /dev/null 2>&1 || { log "RESULT fn4 141: VIRHE image $i missing"; exit 1; }; done
  [ -e "$Y" ] && grep -q RADIANCE_FN4_HC_UPMIX $Y && [ -e $CC3 ] && [ -e $CCW ] && [ -s "$REFJ" ] && [ -s "${REFJ%.jsonl}.outputs.jsonl" ] || { log "RESULT fn4 141: VIRHE $YN, cc_sim3.py, the cc wrapper or the reference ($REFJ + outputs) missing"; exit 1; }
  mkdir -p $O $O/guard $STATSDIR
  find $O -maxdepth 1 -name "$TAG.abort" -delete; find $O -maxdepth 1 -name "$TAG.tstop" -delete
  trap 'touch $O/$TAG.tstop 2>/dev/null; down' EXIT
  trap 'exit 143' TERM INT
  tguard & TGPID=$!
else
  echo "+ guards; tguard &"; TGPID=""
fi
armrun() {   # armrun NAME -> sets ARMJ
  local nm=$1 tries=0 ms st
  restart $TAG-$nm || return 1
  if [ "$DRY" != 1 ]; then
    (cd $D && timeout 600 python3 ds41_tests.py correct --only-q --effort none --outdir $O/guard > $O/guard/$TAG-$nm-warm-q.txt 2>&1)
    (cd $D && timeout 600 python3 ds41_tests.py prefill --depths 16384 --cpt 3.6 --calibrate 0 > $O/guard/$TAG-$nm-warm-pf.txt 2>&1)
  fi
  while :; do
    set -- $(probe8 $TAG-$nm-st$tries); ms=$1
    st=$(awk -v m=$ms 'BEGIN{print (m <= 0) ? "unknown" : (m <= 26.8) ? "fast" : (m >= 28.5) ? "slow" : "between"}')
    log "RESULT fn4 141 $nm state probe $tries: mb c8 $ms ms -> $st"
    { [ "$st" = slow ] && [ $tries -lt 2 ]; } || break
    tries=$((tries + 1)); restart $TAG-$nm-r$tries || return 1
  done
  ccone $nm --replay $REFJ --replay-closed || return 1; ARMJ=$OUTF
  run "docker logs $CN > $O/$TAG-$nm-engine.log 2>&1"
  down
}
STK='export FNIMAGE=$IMG MNBT=4096 EXTRA="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0" FN4NR=1 FN4GI=1 FN4FD="" FN4NW=1 FN4SW="" FN4WV=1 FN4RV=$RV FN4AL="" FN4UM=5'
K4N='export FNIMAGE=$IMGK MNBT=2048 EXTRA="--prefix-resume nearest" FN4NR="" FN4GI="" FN4FD="" FN4NW="" FN4SW="" FN4WV="" FN4RV="" FN4AL="" FN4UM=""'
eval "$STK"; armrun a1 || exit 1; JA1=$ARMJ
eval "$K4N"; armrun k4n || exit 1; JK=$ARMJ
eval "$STK"; armrun a2 || exit 1; JA2=$ARMJ
pair() {   # pair A B LABEL
  if [ "$DRY" = 1 ]; then echo "+ python3 $CC3 --pair-replay $1 $2"; return 0; fi
  python3 $CC3 --pair-replay $1 $2 > $O/$TAG-pair-$3.txt 2>&1
  log "RESULT fn4 141 closed-loop pair $3: $(grep '^PAIR_REPLAY' $O/$TAG-pair-$3.txt | cut -c13- | cut -c1-1700)"
}
pair $JA1 $JA2 "A/A (stack a1 vs a2, the closed-loop floor)"
pair $JK $JA1 "stack a1 vs k4n (B/A = stack / k4n)"
pair $JK $JA2 "stack a2 vs k4n (B/A = stack / k4n)"
[ "$DRY" = 1 ] || for nm in a1 k4n a2; do
  f=$(ls -t $R/ccsim-*-$TAG-$nm-u*.mdelta.txt $O/../ccsim-*-$TAG-$nm-u*.mdelta.txt 2>/dev/null | head -1)
  log "RESULT fn4 141 $nm engine counters: $( [ -n "$f" ] && grep -E 'DERIVED|prefill_tokens_total|engine_steps_mixed_total' $f | sed 's/{[^}]*}//; s/  */ /g' | tr '\n' '|' | cut -c1-400 || echo 'no mdelta')"
done
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
log "RESULT fn4 141-fn-replay-closed: done | a1 $JA1, k4n $JK, a2 $JA2"
