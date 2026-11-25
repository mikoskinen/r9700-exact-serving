#!/bin/bash
# 136-fn-upmix5.sh (queue item, 10.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~25 min; step 0
# alone ~3 min when the new form does not win on one card) -- PF2's up-mix with the CU freed (fn4 hc_upmix 5, coordinator
# 10.11.): the stock tile (RT 256, 16 channels, two m-tiles a wave) with the A operand read as WMMA fragments straight from
# L2 a slice ahead, only B through LDS (double-buffered, 17 KB instead of 43.5) and N computed per lane in registers, so
# two or three workgroups share a CU (133: the stock form ran one a CU at 59.7 TFLOP/s and ~268 GB/s; its 227 of 546 us a
# 2048-row hc_read). Exact: the same k-steps per accumulator over the same fragment values, the same mix sum order.
# STEP 0 (card c3:00 alone): r4d_selftest --case hc_read_e4m3 with the mix-counter slots freed (every hc_readv and
# hc_upmix case incl. form 5 at 512 / 700 / 2048 rows, both code types: byte-identical, or no server runs) and --cold 256
# --perfhcu 512 / 2048. The server part runs only if form 5 beats the stock up-mix at 2048 rows there.
# Then one server on the main-line stack (132's): radiance-local:c16-sw7 (fn4-c16 803f418 = c16-sw6 + form 5: 173 VGPRs, no scratch, 17 KB LDS, two workgroups a CU), F1n + F2a +
# F2b + hcw + hc_readv 3 by environment (compose fn-tp4u.yaml), MNBT 4096 + --prefill-span 2, --prefill-interleave 0,
# --prefix-resume nearest; TP4 all_vram MML 262144 MAXSEQS 16 MTP 3 wht6. IDENTITY (a) a cold 33k copy at hc_upmix 5
# against 128's cold copy, (b) in this process cache-resumed copies 0 vs 5. PREFILL blocks A B B A (mb c16 window, THEN
# fresh prefills 16k / 64k / 128k with a thermal line each; no soak), hc_upmix 0 vs 5, X136_RPF rounds (3).
# Power untouched; read-only temperature watcher (junction 100 C / mem 96 C for 10 s -> stop).
# DRYRUN=1 bash 136-fn-upmix5.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4u.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/abblocks2.py" ] && [ -e "$f/heatsoak.sh" ] && { TL=$f; break; }; done
IMG=${X136_IMAGE:-radiance-local:c16-sw7}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-upm5; REF=fn4-abval2
export FNIMAGE=$IMG FNMODELDIR=$HOME/models/flashnext-iq4r FNMODEL=qwen3.8-next-flash-fp8-iq4r-moe.rad
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
export EXTRA="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats136
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
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 136: VIRHE temperature guard ($hot) -> blocks stopped"; return 0; }
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
  [ $ok = 1 ] || { log "RESULT fn4 136 restart-$tag: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
  log "RESULT fn4 136 restart-$tag: healthy in $(( $(date +%s) - t0 )) s; $(grep -m1 -E '^I kv: ' $S | cut -c3-160); fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300)"
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
    if cmp -s $1 $2; then log "RESULT fn4 136 identity $3: IDENTICAL ($(wc -c < $2) bytes)"
    else log "RESULT fn4 136 identity $3: VIRHE DIFFERENT from byte $(cmp $1 $2 2>&1 | grep -oE 'byte [0-9]+' | head -1)"; fi
  else log "RESULT fn4 136 identity $3: not compared (missing $1 or $2)"; fi
}
warmcmp() {   # warmcmp FILE_A FILE_B LABEL (informational)
  [ "$DRY" = 1 ] && { echo "+ warm-up answers $1 vs $2"; return 0; }
  [ -s $1 ] && [ -s $2 ] || { log "RESULT fn4 136 warm-up $3: not compared (missing $1 or $2)"; return 0; }
  log "RESULT fn4 136 warm-up $3: $(python3 -c 'import json,sys
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
  run "timeout 5400 python3 -I $TL/abblocks2.py --url $URL --mb-dir $D --label $TAG-$nm-$ph --phase $ph --dur ${X136_DUR:-20} --warm ${X136_WARM:-6} --rounds $rr --a '$4' --b '$5' ${pf:+--pf-depths $pf} --names plain,$nm --out $O/$TAG-blocks.jsonl > $O/$TAG-$nm-$ph.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 136 $nm $ph${pf:+ + prefill $pf}: $(grep '^ABB ' $O/$TAG-$nm-$ph.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=4: "-" if v is None else format(v, ".%df" % n)
s = "decode tok/s B/A %s +- %s (t95, %d rounds: %s) | ms/step B/A %s +- %s | A %s B %s tok/s | %d blocks, %d flagged, %s s" % (
    f(d["tps_ratio"]), f(d["tps_ci95"]), len(d["rounds"]), " ".join(f(r["tps_ratio"]) for r in d["rounds"]),
    f(d["ms_ratio"]), f(d["ms_ci95"]), f(d["A_tps_mean"], 1), f(d["B_tps_mean"], 1), d["blocks"], d["blocks_bad"], d["wall_s"])
for k, v in (d.get("pf") or {}).items():
    s += " | prefill %s B/A %s +- %s (%s) A %s B %s tok/s" % (k, f(v["ratio"]), f(v["ci95"]), " ".join(f(x) for x in v["rounds"]), f(v["A_mean"], 1), f(v["B_mean"], 1))
print(s)' 2>&1 | cut -c1-1200)"
  [ "$DRY" = 1 ] || [ -z "$pf" ] || log "RESULT fn4 136 $nm $ph prefill windows' thermal lines: $(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("label") == sys.argv[2] and r.get("pf_therm"): print("r%d%s %s;" % (r["round"], r["arm"], r["pf_therm"]), end=" ")' $O/$TAG-blocks.jsonl $TAG-$nm-$ph 2>&1 | cut -c1-1500)"
}
log "RESULT fn4 136-fn-upmix5: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) F1n + F2a + F2b + hcw + hc_readv 3 EXTRA '$EXTRA' MNBT $MNBT | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
# ---- STEP 0: the selftests on one card, before any server
if [ "$DRY" = 1 ]; then
  echo "+ timeout -k 15 1500 docker run ... -e ROCR_VISIBLE_DEVICES=$C3 r4d_selftest $IMG libr4d.so --case hc_read_e4m3 > $O/136-selftest-hc_read_e4m3.txt"
  for m in ${X136_PERF_M:-512 2048}; do echo "+ timeout -k 15 600 docker run ... r4d_selftest $IMG libr4d.so --cold 256 --perfhcu $m > $O/136-perfhcu-m$m.txt"; done
else
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 136: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 136: VIRHE image $IMG missing"; exit 1; }
  [ -e "$Y" ] && grep -q RADIANCE_FN4_HC_READV $Y && [ -n "$TL" ] || { log "RESULT fn4 136: VIRHE $YN (with RADIANCE_FN4_HC_READV) or tools/abblocks2.py + heatsoak.sh missing"; exit 1; }
  mkdir -p $O $O/guard $STATSDIR
  bad=0
  f=$O/136-selftest-hc_read_e4m3.txt
  timeout -k 15 1500 docker run --rm --init --name fn4-136-st-$$ --network none --device /dev/kfd --device /dev/dri \
    --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$C3 --entrypoint /opt/radiance/bin/r4d_selftest $IMG \
    /opt/radiance/share/radiance/kernels/libr4d.so --case hc_read_e4m3 > $f 2>&1
  rc=$?; docker rm -f fn4-136-st-$$ > /dev/null 2>&1
  np=$(grep -cE '^ +ok ' $f); nf=$(grep -cE '^ +FAIL ' $f); ns=$(grep -cE '^ +skip ' $f)
  nr=$(grep -E '^ +ok ' $f | grep -c 'hc_readv forms'); nv=$(grep -E '^ +ok ' $f | grep -c 'hc_upmix forms')
  log "RESULT fn4 136 selftest hc_read_e4m3 (card c3:00, $IMG, slots freed): exit $rc, $np ok / $nf FAIL / $ns skip; hc_readv cases ok $nr of 24, hc_upmix cases (form 5 incl.) ok $nv of 14$( [ $nf -gt 0 ] && echo " | FAILS: $(grep -E '^ +FAIL ' $f | head -4 | sed 's/^ *//' | cut -c1-220 | tr '\n' '|')")"
  [ $nr -lt 24 ] && bad=1
  case $rc in 0|2) ;; *) bad=1 ;; esac
  [ $nf -gt 0 ] && bad=1
  [ $nv -lt 14 ] && bad=1
  for m in ${X136_PERF_M:-512 2048}; do
    f=$O/136-perfhcu-m$m.txt
    timeout -k 15 600 docker run --rm --init --name fn4-136-pf-$$ --network none --device /dev/kfd --device /dev/dri \
      --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$C3 --entrypoint /opt/radiance/bin/r4d_selftest $IMG \
      /opt/radiance/share/radiance/kernels/libr4d.so --cold 256 --perfhcu $m > $f 2>&1
    docker rm -f fn4-136-pf-$$ > /dev/null 2>&1
    log "RESULT fn4 136 perfhcu M=$m (card c3:00, cold 256 MiB): $(grep -E '^hc_upmix|FAIL' $f | sed 's/^ *//' | tr -s ' ' | tr '\n' '|' | cut -c1-900)"
  done
  [ $bad = 0 ] || { log "RESULT fn4 136: VIRHE the selftest failed, crashed or did not run every hc_upmix case -- no server runs ($O/136-selftest-*.txt)"; exit 1; }
  find $O -maxdepth 1 -name "$TAG.abort" -delete; find $O -maxdepth 1 -name "$TAG.tstop" -delete; find $O -maxdepth 1 -name "$TAG-blocks.jsonl" -delete
  trap 'touch $O/$TAG.tstop 2>/dev/null; down' EXIT
  trap 'exit 143' TERM INT
  tguard & TGPID=$!
fi
# the PF2 form for the prefill blocks: the fastest of 1-4 at 2048 rows on one card, if it beats the stock form there
PFV=$( [ "$DRY" = 1 ] && echo "5 (dry)" || python3 - $O/136-perfhcu-m2048.txt <<'PY' 2>/dev/null || echo "0 (not parsed)"
import re, sys
t = open(sys.argv[1]).read()
v = {int(a): float(b) for a, b in re.findall(r"v(\d) ([\d.]+) us", t)}
up = {int(a): float(b) for a, b in re.findall(r"v(\d) [\d.]+ us \(up-mix ([\d.]+)\)", t)}
if 0 not in v or 5 not in v: print("0 (no timing)"); sys.exit(0)
print(("5" if v[5] < v[0] else "0") + " (one card, 2048 rows, cold: hc_read v0 %.1f us (up-mix %.1f), v5 %.1f us (up-mix %.1f))" % (v[0], up.get(0, -1), v[5], up.get(5, -1)))
PY
)
PF=${X136_PF:-${PFV%% *}}
log "RESULT fn4 136 up-mix form 5 on one card: ${PFV#* }$( [ "$PF" = 0 ] && echo ' -- does not beat the stock form: no server part')"
[ "$PF" = 0 ] && [ "$DRY" != 1 ] && { log "RESULT fn4 136-fn-upmix5: done (step 0 only)"; exit 0; }
[ "$DRY" = 1 ] && { echo "+ guards; tguard &"; TGPID=""; }
restart $TAG || exit 1
# the base (hcw, hc_readv bit 0) comes from the environment (fn-tp4u.yaml)
if [ "$DRY" != 1 ]; then
  (cd $D && timeout 600 python3 ds41_tests.py correct --only-q --effort none --outdir $O/guard > $O/guard/$TAG-warm-q.txt 2>&1)
  (cd $D && timeout 600 python3 ds41_tests.py prefill --depths 16384 --cpt 3.6 --calibrate 0 > $O/guard/$TAG-warm-pf.txt 2>&1)
  timeout 300 python3 $Q/prodfc.py $URL 384 > $O/guard/$TAG-fc1.txt 2>&1
  log "RESULT fn4 136 1-user: $(grep '^FC ' $O/guard/$TAG-fc1.txt | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()[3:])["ms_per_step"])' 2>/dev/null || echo n/a) ms/step (informational)"
fi
warmcmp $O/guard/$REF-warm-q.txt $O/guard/$TAG-warm-q.txt "c16-sw7 (F1n F2a F2b hcw hc_readv 3, hc_upmix 0) vs 128's c16-sw2 F1n (informational)"
tries=0
while :; do
  set -- $(probe8 $TAG-st$tries); ms=$1
  st=$(awk -v m=$ms 'BEGIN{print (m <= 0) ? "unknown" : (m <= 26.8) ? "fast" : (m >= 28.5) ? "slow" : "between"}')
  log "RESULT fn4 136 state probe $tries: mb c8 $ms ms -> $st"
  { [ "$st" = slow ] && [ $tries -lt 2 ]; } || break
  tries=$((tries + 1)); restart $TAG-r$tries || exit 1
done
# IDENTITY. (a) cold vs cold across processes: the same 33k copy prompt with every new switch on against 128's cold
# run (F1n alone; F2a / F2b exact by 128's identity); (b) in this process, both resuming from the cache (a hit and a
# cold prompt are not byte-identical under --prefix-resume nearest): all new switches off vs on.
OFF='{"hc_upmix": 0}'; ON='{"hc_upmix": 5}'
setsw "$ON"; copyt coldnew
cmpf $O/copy-$REF-cold-run1.txt $O/copy-$TAG-coldnew-run1.txt "(a) cold 33k copy, hc_upmix 5 (c16-sw7, the 132 stack) vs 128's cold copy (c16-sw2, F1n)"
setsw "$OFF"; copyt off
setsw "$ON"; copyt on
cmpf $O/copy-$TAG-off-run1.txt $O/copy-$TAG-on-run1.txt "(b) cache-resumed 33k copy, hc_upmix 5 vs 0, one process"
setsw "$OFF"
[ "$DRY" = 1 ] || log "RESULT fn4 136 A/B start: fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300) | thermal now $(thermnow)"
abrun upmix5-pf c16 ${X136_RPF:-3} '{"hc_upmix": 0}' '{"hc_upmix": 5}' 16384,65536,131072
run "docker logs $CN > $O/$TAG-engine.log 2>&1"
[ "$DRY" = 1 ] || log "RESULT fn4 136 engine: $(grep -cE 'fn4: switches .* applied' $O/$TAG-engine.log) applied changes, $(grep -cE 'fn4: .* refused' $O/$TAG-engine.log) refused; $(grep -E 'refused' $O/$TAG-engine.log | grep fn4 | head -2 | tr '\n' '|' | cut -c1-300)"
down
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
log "RESULT fn4 136-fn-upmix5: done"
