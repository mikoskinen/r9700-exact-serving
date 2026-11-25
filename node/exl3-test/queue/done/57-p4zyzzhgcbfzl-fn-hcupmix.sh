#!/bin/bash
# 133-fn-hcupmix.sh (queue item, 10.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~55 min) --
# hc_read's up-mix at other tiles with the same bytes (fn4 hc_upmix) as in-process A/B blocks (tools/abblocks2.py, one
# server, POST /admin/fn4) on the main-line stack (132's): image radiance-local:c16-sw5 (fn4-c16 00ab19b = c16-sw4 +
# hc_upmix + ar_laneblocks + selftest --perfhcu), F1n + F2a + F2b + hcw + hc_readv 3 (bits 0 and 1: 131) by environment
# (compose fn-tp4u.yaml), MNBT 4096
# + --prefill-span 2, --prefill-interleave 0, --prefix-resume nearest; TP4 all_vram MML 262144 MAXSEQS 16 MTP 3 wht6.
# WHY (ranked against P3 by attainable wall, coordinator 10.11.): PF2's up-mix is 227 of 546 us a 2048-row call on one
# card (131 --perfhcv stages), 96 calls a piece; at ~130 us (13.4 GFLOP, ~53 MB): 128k ~-0.53 s, 16k x4 ~-0.25 s, plus
# the decode up-mix (33-64 rows, now the 32-row tile reading mix_up twice). P3 (i8a8 GEMMs): ~-0.47 / -0.22 s, no decode.
#   hc_upmix 1-4  PF2's up-mix: 1 RT 256 CH 16, h after the K loop (stock tile, fewer live registers than its 201
#                 VGPRs); 2 RT 128 CH 32 (64-byte runs of h, half the d re-staging); 3 = 2 with h after the loop;
#                 4 RT 128 CH 16. The fastest on one card at 2048 rows (STEP 0) is the prefill blocks' B arm, only if
#                 it beats the stock form there.
#   hc_upmix 8    the 33-64-row decode up-mix at the 64-row tile, pipelined (mix_up read once).
#   ar_laneblocks 8 / 16  the prefill all-reduce lane's ti8 two-shot at 8 / 16 blocks of 1024 threads instead of 4
#                 (the opposite of 131's ar_lanethreads 256, which lost 2.8-3.7 % in every round: the lane's throughput
#                 is exposed); exact (grid partition only; shards and the rank fold unchanged).
# STEP 0 (card c3:00 alone, ~6 min): r4d_selftest --case hc_read_e4m3 (incl. the hc_upmix cases at M 24 33 48 64 512 700
# 2048, both code types: byte-identical, kernels swapped exactly where served) and --cold 256 --perfhcu M at M 48 64 512
# 2048. Then one server: warm-up, slow-state probe, IDENTITY (a) a cold 33k copy with the chosen forms against 128's cold
# copy, (b) in this process cache-resumed copies, stock vs chosen. Blocks A B B A (POST + epoch_bump + mb 6 s warm + 20 s),
# no soak: DECODE first -- hc_upmix 0 vs 8 at c16 (X133_R16, 3) and c8 (X133_R8, 2) -- then PREFILL blocks (mb c16, THEN
# fresh prefills 16k / 64k / 128k with a thermal line each): hc_upmix 0 vs the chosen PF2 form (X133_RPF, 3), then
# the AR lane screen: ar_laneblocks 0 vs 8 and 0 vs 16 (X133_RAR, 2 rounds each), at the chosen PF2 form.
# Reading: single rounds against ~+-1.5 %. Power untouched; read-only temperature watcher (100 C / 96 C for 10 s).
# DRYRUN=1 bash 133-fn-hcupmix.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4u.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/abblocks2.py" ] && [ -e "$f/heatsoak.sh" ] && { TL=$f; break; }; done
IMG=${X133_IMAGE:-radiance-local:c16-sw5}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-hcupmix; REF=fn4-abval2
export FNIMAGE=$IMG FNMODELDIR=$HOME/models/flashnext-iq4r FNMODEL=qwen3.8-next-flash-fp8-iq4r-moe.rad
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
export EXTRA="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats133
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
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 133: VIRHE temperature guard ($hot) -> blocks stopped"; return 0; }
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
  [ $ok = 1 ] || { log "RESULT fn4 133 restart-$tag: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
  log "RESULT fn4 133 restart-$tag: healthy in $(( $(date +%s) - t0 )) s; $(grep -m1 -E '^I kv: ' $S | cut -c3-160); fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300)"
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
    if cmp -s $1 $2; then log "RESULT fn4 133 identity $3: IDENTICAL ($(wc -c < $2) bytes)"
    else log "RESULT fn4 133 identity $3: VIRHE DIFFERENT from byte $(cmp $1 $2 2>&1 | grep -oE 'byte [0-9]+' | head -1)"; fi
  else log "RESULT fn4 133 identity $3: not compared (missing $1 or $2)"; fi
}
warmcmp() {   # warmcmp FILE_A FILE_B LABEL (informational)
  [ "$DRY" = 1 ] && { echo "+ warm-up answers $1 vs $2"; return 0; }
  [ -s $1 ] && [ -s $2 ] || { log "RESULT fn4 133 warm-up $3: not compared (missing $1 or $2)"; return 0; }
  log "RESULT fn4 133 warm-up $3: $(python3 -c 'import json,sys
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
  run "timeout 5400 python3 -I $TL/abblocks2.py --url $URL --mb-dir $D --label $TAG-$nm-$ph --phase $ph --dur ${X133_DUR:-20} --warm ${X133_WARM:-6} --rounds $rr --a '$4' --b '$5' ${pf:+--pf-depths $pf} --names plain,$nm --out $O/$TAG-blocks.jsonl > $O/$TAG-$nm-$ph.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 133 $nm $ph${pf:+ + prefill $pf}: $(grep '^ABB ' $O/$TAG-$nm-$ph.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=4: "-" if v is None else format(v, ".%df" % n)
s = "decode tok/s B/A %s +- %s (t95, %d rounds: %s) | ms/step B/A %s +- %s | A %s B %s tok/s | %d blocks, %d flagged, %s s" % (
    f(d["tps_ratio"]), f(d["tps_ci95"]), len(d["rounds"]), " ".join(f(r["tps_ratio"]) for r in d["rounds"]),
    f(d["ms_ratio"]), f(d["ms_ci95"]), f(d["A_tps_mean"], 1), f(d["B_tps_mean"], 1), d["blocks"], d["blocks_bad"], d["wall_s"])
for k, v in (d.get("pf") or {}).items():
    s += " | prefill %s B/A %s +- %s (%s) A %s B %s tok/s" % (k, f(v["ratio"]), f(v["ci95"]), " ".join(f(x) for x in v["rounds"]), f(v["A_mean"], 1), f(v["B_mean"], 1))
print(s)' 2>&1 | cut -c1-1200)"
  [ "$DRY" = 1 ] || [ -z "$pf" ] || log "RESULT fn4 133 $nm $ph prefill windows' thermal lines: $(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("label") == sys.argv[2] and r.get("pf_therm"): print("r%d%s %s;" % (r["round"], r["arm"], r["pf_therm"]), end=" ")' $O/$TAG-blocks.jsonl $TAG-$nm-$ph 2>&1 | cut -c1-1500)"
}
log "RESULT fn4 133-fn-hcupmix: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) F1n + F2a + F2b + hcw + hc_readv 3 EXTRA '$EXTRA' MNBT $MNBT | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
# ---- STEP 0: the selftests on one card, before any server
if [ "$DRY" = 1 ]; then
  echo "+ timeout -k 15 1500 docker run ... -e ROCR_VISIBLE_DEVICES=$C3 r4d_selftest $IMG libr4d.so --case hc_read_e4m3 > $O/133-selftest-hc_read_e4m3.txt"
  for m in ${X133_PERF_M:-48 64 512 2048}; do echo "+ timeout -k 15 600 docker run ... r4d_selftest $IMG libr4d.so --cold 256 --perfhcu $m > $O/133-perfhcu-m$m.txt"; done
else
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 133: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 133: VIRHE image $IMG missing"; exit 1; }
  [ -e "$Y" ] && grep -q RADIANCE_FN4_HC_READV $Y && [ -n "$TL" ] || { log "RESULT fn4 133: VIRHE $YN (with RADIANCE_FN4_HC_READV) or tools/abblocks2.py + heatsoak.sh missing"; exit 1; }
  mkdir -p $O $O/guard $STATSDIR
  bad=0
  f=$O/133-selftest-hc_read_e4m3.txt
  timeout -k 15 1500 docker run --rm --init --name fn4-133-st-$$ --network none --device /dev/kfd --device /dev/dri \
    --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$C3 --entrypoint /opt/radiance/bin/r4d_selftest $IMG \
    /opt/radiance/share/radiance/kernels/libr4d.so --case hc_read_e4m3 > $f 2>&1
  rc=$?; docker rm -f fn4-133-st-$$ > /dev/null 2>&1
  np=$(grep -cE '^ +ok ' $f); nf=$(grep -cE '^ +FAIL ' $f); ns=$(grep -cE '^ +skip ' $f); nv=$(grep -E '^ +ok ' $f | grep -c 'hc_upmix forms')
  log "RESULT fn4 133 selftest hc_read_e4m3 (card c3:00, $IMG): exit $rc, $np ok / $nf FAIL / $ns skip; hc_upmix cases ok $nv of 14$( [ $nf -gt 0 ] && echo " | FAILS: $(grep -E '^ +FAIL ' $f | head -4 | sed 's/^ *//' | cut -c1-220 | tr '\n' '|')")"
  case $rc in 0|2) ;; *) bad=1 ;; esac
  [ $nf -gt 0 ] && bad=1
  [ $nv -lt 14 ] && bad=1
  for m in ${X133_PERF_M:-48 64 512 2048}; do
    f=$O/133-perfhcu-m$m.txt
    timeout -k 15 600 docker run --rm --init --name fn4-133-pf-$$ --network none --device /dev/kfd --device /dev/dri \
      --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$C3 --entrypoint /opt/radiance/bin/r4d_selftest $IMG \
      /opt/radiance/share/radiance/kernels/libr4d.so --cold 256 --perfhcu $m > $f 2>&1
    docker rm -f fn4-133-pf-$$ > /dev/null 2>&1
    log "RESULT fn4 133 perfhcu M=$m (card c3:00, cold 256 MiB): $(grep -E '^hc_upmix|FAIL' $f | sed 's/^ *//' | tr -s ' ' | tr '\n' '|' | cut -c1-900)"
  done
  [ $bad = 0 ] || { log "RESULT fn4 133: VIRHE the selftest failed, crashed or did not run every hc_upmix case -- no server runs ($O/133-selftest-*.txt)"; exit 1; }
  find $O -maxdepth 1 -name "$TAG.abort" -delete; find $O -maxdepth 1 -name "$TAG.tstop" -delete; find $O -maxdepth 1 -name "$TAG-blocks.jsonl" -delete
  trap 'touch $O/$TAG.tstop 2>/dev/null; down' EXIT
  trap 'exit 143' TERM INT
  tguard & TGPID=$!
fi
# the PF2 form for the prefill blocks: the fastest of 1-4 at 2048 rows on one card, if it beats the stock form there
PFV=$( [ "$DRY" = 1 ] && echo "2 (dry)" || python3 - $O/133-perfhcu-m2048.txt <<'PY' 2>/dev/null || echo "0 (not parsed)"
import re, sys
t = open(sys.argv[1]).read()
v = {int(a): float(b) for a, b in re.findall(r"v(\d) ([\d.]+) us", t)}
if 0 not in v or len(v) < 2: print("0 (no timing)"); sys.exit(0)
b = min((k for k in v if k), key=lambda k: v[k])
print(("%d" % b if v[b] < v[0] else "0") + " (one card, 2048 rows: " + ", ".join("v%d %.1f us" % (k, v[k]) for k in sorted(v)) + ")")
PY
)
PF=${X133_PF:-${PFV%% *}}
log "RESULT fn4 133 PF2 up-mix form for the prefill blocks: hc_upmix $PF ${PFV#* }$( [ "$PF" = 0 ] && echo ' -- no form beat the stock one: prefill blocks skipped')"
[ "$DRY" = 1 ] && { echo "+ guards; tguard &"; TGPID=""; }
restart $TAG || exit 1
# the base (hcw, hc_readv bit 0) comes from the environment (fn-tp4u.yaml)
if [ "$DRY" != 1 ]; then
  (cd $D && timeout 600 python3 ds41_tests.py correct --only-q --effort none --outdir $O/guard > $O/guard/$TAG-warm-q.txt 2>&1)
  (cd $D && timeout 600 python3 ds41_tests.py prefill --depths 16384 --cpt 3.6 --calibrate 0 > $O/guard/$TAG-warm-pf.txt 2>&1)
  timeout 300 python3 $Q/prodfc.py $URL 384 > $O/guard/$TAG-fc1.txt 2>&1
  log "RESULT fn4 133 1-user: $(grep '^FC ' $O/guard/$TAG-fc1.txt | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()[3:])["ms_per_step"])' 2>/dev/null || echo n/a) ms/step (informational)"
fi
warmcmp $O/guard/$REF-warm-q.txt $O/guard/$TAG-warm-q.txt "c16-sw5 (F1n F2a F2b hcw hc_readv 3, hc_upmix 0) vs 128's c16-sw2 F1n (informational)"
tries=0
while :; do
  set -- $(probe8 $TAG-st$tries); ms=$1
  st=$(awk -v m=$ms 'BEGIN{print (m <= 0) ? "unknown" : (m <= 26.8) ? "fast" : (m >= 28.5) ? "slow" : "between"}')
  log "RESULT fn4 133 state probe $tries: mb c8 $ms ms -> $st"
  { [ "$st" = slow ] && [ $tries -lt 2 ]; } || break
  tries=$((tries + 1)); restart $TAG-r$tries || exit 1
done
# IDENTITY. (a) cold vs cold across processes: the same 33k copy prompt with every new switch on against 128's cold
# run (F1n alone; F2a / F2b exact by 128's identity); (b) in this process, both resuming from the cache (a hit and a
# cold prompt are not byte-identical under --prefix-resume nearest): all new switches off vs on.
ONV=$(( ${PF:-0} + 8 ))
OFF='{"hc_upmix": 0}'; ON="{\"hc_upmix\": $ONV}"
setsw "$ON"; copyt coldnew
cmpf $O/copy-$REF-cold-run1.txt $O/copy-$TAG-coldnew-run1.txt "(a) cold 33k copy, hc_upmix $ONV (c16-sw5, the 132 stack, hc_readv 3) vs 128's cold copy (c16-sw2, F1n)"
setsw "$OFF"; copyt off
setsw "$ON"; copyt on
cmpf $O/copy-$TAG-off-run1.txt $O/copy-$TAG-on-run1.txt "(b) cache-resumed 33k copy, hc_upmix $ONV vs 0, one process"
setsw "$OFF"
[ "$DRY" = 1 ] || log "RESULT fn4 133 switch log: $(docker logs $CN 2>&1 | grep -E 'fn4: switches|fn4: .*refused' | tail -4 | sed 's/^[A-Z] //' | tr '\n' '|' | cut -c1-600)"
[ "$DRY" = 1 ] || log "RESULT fn4 133 A/B start: fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300) | thermal now $(thermnow)"
# DECODE blocks first, then the prefill blocks (mb window, then the prefills, in each block)
abrun upmix64-dec c16 ${X133_R16:-3} '{"hc_upmix": 0}' '{"hc_upmix": 8}'
abrun upmix64-dec c8 ${X133_R8:-2} '{"hc_upmix": 0}' '{"hc_upmix": 8}'
[ "${PF:-0}" != 0 ] && abrun upmixpf$PF-pf c16 ${X133_RPF:-3} '{"hc_upmix": 0}' "{\"hc_upmix\": $PF}" 16384,65536,131072
# the AR lane screen (cheap switch values), at the chosen PF2 form (or the stock one)
abrun arblocks8-pf c16 ${X133_RAR:-2} "{\"hc_upmix\": ${PF:-0}, \"ar_laneblocks\": 0}" "{\"hc_upmix\": ${PF:-0}, \"ar_laneblocks\": 8}" 16384,65536,131072
abrun arblocks16-pf c16 ${X133_RAR:-2} "{\"hc_upmix\": ${PF:-0}, \"ar_laneblocks\": 0}" "{\"hc_upmix\": ${PF:-0}, \"ar_laneblocks\": 16}" 16384,65536,131072
run "docker logs $CN > $O/$TAG-engine.log 2>&1"
[ "$DRY" = 1 ] || log "RESULT fn4 133 engine: $(grep -cE 'fn4: switches .* applied' $O/$TAG-engine.log) applied changes, $(grep -cE 'fn4: .* refused' $O/$TAG-engine.log) refused; $(grep -E 'refused' $O/$TAG-engine.log | grep fn4 | head -2 | tr '\n' '|' | cut -c1-300)"
down
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
log "RESULT fn4 133-fn-hcupmix: done"
