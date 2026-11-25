#!/bin/bash
# 158-fn-sphc.sh (queue item, 10.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~55 min) -- SP-hc,
# THE SEQUENCE-PARALLEL GATED RESIDUAL, SPEED (user 10.11.: "rakenna", then "laatutest myöhemmin, nyt vain nopeus"):
# with RADIANCE_FN4_SPHC=1 at start the trunk declares reduce_scatter_rows / all_gather_rows / hc_codes, and switch hc_sp 1
# runs each pipelined-prefill piece of >= 64 rows as: the block's reduce-scatter (each rank the sum of its quarter of the
# piece's rows), the connections' write and read on that quarter, the next block input all-gathered on the ti8 wire and
# its codes redone on every row; the wide stream gathered raw after the last layer. Decode and serial steps unchanged.
# NOT EXACT (the gathered input rides the lossy wire) -- the quality item (KL arms, BetterBench) is prepared, not queued.
# FREE SAFETY CHECKS only: the server stays healthy through every block, and the 33k copy under each arm (the copy's own
# exactness and ratio, and whether the two arms' outputs agree). Image radiance-local:c16-sp (fn4-c16 964510ae = stack
# 3's tree + the P-1 reading arms + the SP probe + SP-hc), iq4r .rad, stack 3's switches (F1n + F2a + F2b + hcw +
# hc_readv 11 + HC_SS + hc_upmix 5, MNBT 4096 + span 2, interleave 0, nearest), compose fn-tp4z.yaml (FN4SPHC=1). In
# process A B B A: decode mb c16 x3 (the unchanged check), PREFILL blocks (mb c16 window, then prefills 16k / 64k / 128k,
# thermal lines) x3, mb pf x3. 157 (the probe) is the reference for how much of the gain the collectives' new form keeps.
# Power untouched; read-only temperature watcher.
# DRYRUN=1 bash 158-fn-sphc.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4z.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/abblocks2.py" ] && [ -e "$f/heatsoak.sh" ] && { TL=$f; break; }; done
IMG=${X158_IMAGE:-radiance-local:c16-sp}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-sph; REF=fn4-abval2
export FNIMAGE=$IMG FNMODELDIR=${X158_DIR:-$HOME/models/flashnext-iq4r} FNMODEL=${X158_MODEL:-qwen3.8-next-flash-fp8-iq4r-moe.rad}
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
export EXTRA="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats158
export FN4NR=1 FN4GI=1 FN4FD="" FN4NW=1 FN4SW="" FN4WV=1 FN4RV=11 FN4AL="" FN4UM=5 FN4SS=1 FN4CH="" FN4XM="" FN4SPHC=1 PROFEVERY=""
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
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 158: VIRHE temperature guard ($hot) -> blocks stopped"; return 0; }
    sleep 1; done; }
restart() {   # restart TAG -> 0 healthy
  local tag=$1 ok=0 t0 i S=$O/start-$1.log
  [ "$DRY" = 1 ] && { echo "+ FNIMAGE=$FNIMAGE FN4NR=$FN4NR FN4GI=$FN4GI FN4NW=$FN4NW FN4WV=$FN4WV FN4RV=$FN4RV FN4UM=$FN4UM FN4CH=$FN4CH EXTRA='$EXTRA' compose -f $Y up -d; health <= 1200 s; docker logs > $S"; return 0; }
  down; t0=$(date +%s); compose up -d > $O/up-$tag.txt 2>&1
  for i in $(seq 1 240); do
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' $URL/health)" = 200 ] && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Status}}' $CN 2>/dev/null)" != running ] && break
    sleep 5
  done
  docker logs $CN > $S 2>&1
  [ $ok = 1 ] || { log "RESULT fn4 158 restart-$tag: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
  log "RESULT fn4 158 restart-$tag: healthy in $(( $(date +%s) - t0 )) s; $(grep -m1 -E '^I kv: ' $S | cut -c3-160); fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300)"
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
    if cmp -s $1 $2; then log "RESULT fn4 158 identity $3: IDENTICAL ($(wc -c < $2) bytes)"
    else log "RESULT fn4 158 identity $3: DIFFERENT (not an error: SP is not exact) from byte $(cmp $1 $2 2>&1 | grep -oE 'byte [0-9]+' | head -1)"; fi
  else log "RESULT fn4 158 identity $3: not compared (missing $1 or $2)"; fi
}
warmcmp() {   # warmcmp FILE_A FILE_B LABEL (informational)
  [ "$DRY" = 1 ] && { echo "+ warm-up answers $1 vs $2"; return 0; }
  [ -s $1 ] && [ -s $2 ] || { log "RESULT fn4 158 warm-up $3: not compared (missing $1 or $2)"; return 0; }
  log "RESULT fn4 158 warm-up $3: $(python3 -c 'import json,sys
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
  run "timeout 5400 python3 -I $TL/abblocks2.py --url $URL --mb-dir $D --label $TAG-$nm-$ph --phase $ph --dur ${X158_DUR:-20} --warm ${X158_WARM:-6} --rounds $rr --a '$4' --b '$5' ${pf:+--pf-depths $pf} --names plain,$nm --out $O/$TAG-blocks.jsonl > $O/$TAG-$nm-$ph.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 158 $nm $ph${pf:+ + prefill $pf}: $(grep '^ABB ' $O/$TAG-$nm-$ph.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=4: "-" if v is None else format(v, ".%df" % n)
s = "decode tok/s B/A %s +- %s (t95, %d rounds: %s) | ms/step B/A %s +- %s | A %s B %s tok/s | %d blocks, %d flagged, %s s" % (
    f(d["tps_ratio"]), f(d["tps_ci95"]), len(d["rounds"]), " ".join(f(r["tps_ratio"]) for r in d["rounds"]),
    f(d["ms_ratio"]), f(d["ms_ci95"]), f(d["A_tps_mean"], 1), f(d["B_tps_mean"], 1), d["blocks"], d["blocks_bad"], d["wall_s"])
for k, v in (d.get("pf") or {}).items():
    s += " | prefill %s B/A %s +- %s (%s) A %s B %s tok/s" % (k, f(v["ratio"]), f(v["ci95"]), " ".join(f(x) for x in v["rounds"]), f(v["A_mean"], 1), f(v["B_mean"], 1))
print(s)' 2>&1 | cut -c1-1200)"
  [ "$DRY" = 1 ] || [ -z "$pf" ] || log "RESULT fn4 158 $nm $ph prefill windows' thermal lines: $(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("label") == sys.argv[2] and r.get("pf_therm"): print("r%d%s %s;" % (r["round"], r["arm"], r["pf_therm"]), end=" ")' $O/$TAG-blocks.jsonl $TAG-$nm-$ph 2>&1 | cut -c1-1500)"
}
GS=; for f in "$H/tools/greedyset.py" "$H/../tools/greedyset.py" "$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py"; do [ -z "$GS" ] && [ -e "$f" ] && GS=$f; done
alive() {   # alive WHAT: the server still answers, else logs + down + exit
  [ "$DRY" = 1 ] && return 0
  local code; code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' $URL/health)
  [ "$code" = 200 ] && return 0
  docker logs $CN > $O/$TAG-dead-engine.log 2>&1
  log "RESULT fn4 158: VIRHE the server is not healthy after $1 (/health $code; last E lines: $(grep -E '^E |trap|abort|HSA_STATUS' $O/$TAG-dead-engine.log | tail -3 | tr '\n' ' ' | cut -c1-400)) -> logs saved, down"
  down; exit 1
}
gset() { run "(cd $D && timeout 900 python3 $GS $URL $O/$TAG-gs-$2.jsonl --conc $1 --max-tokens 192 > $O/$TAG-gs-$2.txt 2>&1)"; }
gcmp() { [ "$DRY" = 1 ] && { echo IDENTICAL; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
log "RESULT fn4 158-fn-sphc: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) iq4r stack 3, hc_sp 0 vs 1 in process (SP-hc; not exact) | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 158: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 158: VIRHE image $IMG missing"; exit 1; }
  [ -e "$Y" ] && [ -n "$TL" ] && [ -n "$GS" ] || { log "RESULT fn4 158: VIRHE $YN, tools/abblocks2.py or tools/greedyset.py missing"; exit 1; }
  mkdir -p $O $O/guard $STATSDIR
  find $O -maxdepth 1 -name "$TAG.abort" -delete; find $O -maxdepth 1 -name "$TAG.tstop" -delete; find $O -maxdepth 1 -name "$TAG-blocks.jsonl" -delete
  trap 'touch $O/$TAG.tstop 2>/dev/null; down' EXIT
  trap 'exit 143' TERM INT
  tguard & TGPID=$!
else
  echo "+ guards; tguard &"; TGPID=""
fi
# ---- STEP 0: the collectives' own selftest on the four cards (sp_rows_check), before any server
if [ "$DRY" = 1 ]; then echo "+ timeout -k 15 900 docker run ... (4 cards) r4d_selftest $FNIMAGE libr4d.so --case sp_rs_rows_4rank_ti8 > $O/158-selftest-sp.txt"
else
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 158: VIRHE a radiance container is running"; exit 1; }
  f=$O/158-selftest-sp.txt
  timeout -k 15 900 docker run --rm --init --name fn4-158-st-$$ --network none --device /dev/kfd --device /dev/dri \
    --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$CARDS --entrypoint /opt/radiance/bin/r4d_selftest $FNIMAGE \
    /opt/radiance/share/radiance/kernels/libr4d.so --case sp_rs_rows_4rank_ti8 > $f 2>&1
  rc=$?; docker rm -f fn4-158-st-$$ > /dev/null 2>&1
  np=$(grep -cE '^ +ok ' $f); nf=$(grep -cE '^ +FAIL ' $f); ns=$(grep -cE '^ +skip ' $f)
  log "RESULT fn4 158 selftest sp_rows_check (4 cards): exit $rc, $np ok / $nf FAIL / $ns skip | $(grep -E 'relative RMS' $f | sed 's/^ *//' | head -5 | tr '\n' '|' | cut -c1-700)$( [ $nf -gt 0 ] && echo " | FAILS: $(grep -E '^ +FAIL ' $f | head -3 | sed 's/^ *//' | cut -c1-200 | tr '\n' '|')")"
  { [ $nf -gt 0 ] || [ $np -lt 26 ]; } && { log "RESULT fn4 158: VIRHE the SP collectives' selftest failed or did not run every case -- no server ($f)"; exit 1; }
fi
restart $TAG || exit 1
if [ "$DRY" != 1 ]; then
  (cd $D && timeout 600 python3 ds41_tests.py correct --only-q --effort none --outdir $O/guard > $O/guard/$TAG-warm-q.txt 2>&1)
fi
alive "the warm-up"
tries=0
while :; do
  set -- $(probe8 $TAG-st$tries); ms=$1
  st=$(awk -v m=$ms 'BEGIN{print (m <= 0) ? "unknown" : "logged"}')   # logged; an in-process A/B carries any state to both arms
  log "RESULT fn4 158 state probe $tries: mb c8 $ms ms -> $st"
  { [ "$st" = slow ] && [ $tries -lt 2 ]; } || break
  tries=$((tries + 1)); restart $TAG-r$tries || exit 1
done
OFF='{"hc_sp": 0}'; ON='{"hc_sp": 1}'
setsw "$OFF"
# ---- the free safety checks: the 33k copy under each arm (its own exactness / ratio, and A vs B), the server alive
copyt off; alive "the copy (hc_sp 0)"
setsw "$ON"; copyt on; alive "the copy (hc_sp 1)"
setsw "$OFF"
if [ "$DRY" != 1 ]; then
  jx() { grep -h '^J ' "$@" 2>/dev/null | python3 -c 'import json,sys
for l in sys.stdin:
    d = json.loads(l[2:]); print("exact %s ratio %s ttft %s" % (d.get("exact"), d.get("ratio"), d.get("ttft")))' 2>/dev/null | head -1; }
  log "RESULT fn4 158 copy (33k prefix, 2400 chars): hc_sp 0 $(jx $O/$TAG-off-copy.txt) | hc_sp 1 $(jx $O/$TAG-on-copy.txt) | engine: $(docker logs $CN 2>&1 | grep -m2 -E 'fn4 SP-hc|hc_sp' | sed 's/^[A-Z] //' | tr '\n' '|' | cut -c1-300)"
  cmpf $O/copy-$TAG-off-run1.txt $O/copy-$TAG-on-run1.txt "cache-resumed 33k copy, hc_sp 1 vs 0 (SP is not exact: informational)"
fi
[ "$DRY" = 1 ] || log "RESULT fn4 158 A/B start: fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300) | thermal now $(thermnow)"
abrun sph-c16 c16 ${X158_RC16:-3} "$OFF" "$ON"
alive "the decode blocks"
abrun sph-pf c16 ${X158_RPF:-3} "$OFF" "$ON" 16384,65536,131072
alive "the prefill blocks"
# mb pf (a 128k-char prefill beside 4 decoders) A B B A x rounds: the switch, an epoch bump, a short settle, then mb pf;
# its prefill_tps a block (abblocks2 reads decode tok/s only)
for r in $(seq 1 ${X158_RMBPF:-3}); do
  for arm in A B B A; do
    [ -e $O/$TAG.abort ] && break 2
    [ $arm = A ] && cfg="$OFF" || cfg="$ON"
    run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$(echo "$cfg" | sed 's/}$/, "epoch_bump": 1}/')' $URL/admin/fn4 > /dev/null"
    run "(cd $D && timeout 900 python3 mb.py --tag $TAG-mbpf-r$r-$arm --phases pf --dur 20 --warm 6 > $O/$TAG-mbpf-r$r-$arm-\$(date +%s%N).txt 2>&1)"
  done
done
if [ "$DRY" != 1 ]; then
  log "RESULT fn4 158 mb pf (prefill_tps, A = hc_sp 0, B = 1): $(python3 - $O $TAG <<'PY' 2>&1 | cut -c1-900
import glob, json, os, sys
o, tag = sys.argv[1], sys.argv[2]
v = {"A": [], "B": []}
for f in sorted(glob.glob(os.path.join(o, tag + "-mbpf-r*-*.txt")), key=os.path.getmtime):
    arm = f.rsplit("-", 2)[-2]
    for l in open(f):
        if l.startswith("MB "):
            try: d = json.loads(l[3:])
            except ValueError: continue
            if d.get("prefill_tps"): v[arm].append(d["prefill_tps"])
a, b = v["A"], v["B"]
print("A %s | B %s | B/A %.4f" % (a, b, (sum(b) / len(b)) / (sum(a) / len(a))) if a and b else "A %s B %s (incomplete)" % (a, b))
PY
)"
fi
alive "the mb pf blocks"
setsw "$OFF"
run "docker logs $CN > $O/$TAG-engine.log 2>&1"
[ "$DRY" = 1 ] || log "RESULT fn4 158 engine: $(grep -cE 'fn4: switches .* applied' $O/$TAG-engine.log) applied changes, $(grep -cE 'fn4: .* refused' $O/$TAG-engine.log) refused"
down
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
log "RESULT fn4 158-fn-sphc: done"
