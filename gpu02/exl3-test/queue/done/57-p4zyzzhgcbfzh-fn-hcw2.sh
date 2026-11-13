#!/bin/bash
# 129-fn-hcw2.sh (queue item, 9.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~45 min) -- hc_write
# eight elements a thread (fn4 hcw, rebased from the 1.15.1 line's b9ebe5f onto the main line; same expression, same
# bytes) as an in-process A/B (tools/abblocks2.py, one server, POST /admin/fn4 hc_writevec 0 / 1) on the MAIN-LINE BASE:
# image radiance-local:c16-sw3 (fn4-c16 fd18c52 = c16-sw2 + the eight-wide form behind the hc_writevec switch + the
# selftest's --perfhcw), F1n on (RADIANCE_FN4_HC_NORMWAVE=1), RADIANCE_FN4_SWITCHABLE=1 (128's declaration), MNBT 4096 +
# --prefill-span 2, --prefill-interleave 0, --prefix-resume nearest; TP4 all_vram MML 262144 MAXSEQS 16 MTP 3 wht6;
# compose fn-tp4s.yaml. Replaces 100-fn-tp4hcw.sh (1.15.1 m4 line; left in held/).
# WHY (127, mpB prefill profile, rank 0): hc_write 1176 ms of the compute lane's 13879 ms at 128k (8.5 %; every call on
# the critical lane), ~210 us a 2048-row piece for 94.4 MB (y read, four streams of h read + written) = 450 GB/s of
# ~640; the floor 147 us. Decode c16: 17.7 us a call at the >= 26-row steps (~2.2 MB at 48 rows, ~124 GB/s).
# STEP 0 (card c3:00 alone, ~5 min): r4d_selftest --case hc_write_bf16 (the forms byte for byte at M 1 7 16 64 300 2048,
# n 1004 and a misaligned h kept on the plain form, the dispatched kernel checked) and --case hc_read_e4m3.
# Then one server: warm-up (answers compared with 128's, informational: c16-sw3 plain = c16-sw2), slow-state probe,
# IDENTITY (a) cold 33k copy with hc_writevec 1 against 128's cold copy (c16-sw2, the same server line) byte for byte,
# (b) in this process, both resuming from the cache: hc_writevec 0 vs 1. A/B blocks A = hc_writevec 0, B = 1, rounds
# A B B A, a block = POST + epoch_bump, then mb c16 (6 s warm + 20 s) THEN fresh prefills 16k / 64k / 128k (no soak,
# TESTING.md 9.11.; a thermal line per prefill window in the blocks file), X129_RPF rounds (3); then mb c8 decode-only
# blocks, X129_R8 rounds (2). Reading: prefill B/A per depth and decode B/A against 128's noise run.
# Power is not touched; read-only temperature watcher (junction 100 C / mem 96 C for 10 s -> stop).
# DRYRUN=1 bash 129-fn-hcw2.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4s.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/abblocks2.py" ] && [ -e "$f/heatsoak.sh" ] && { TL=$f; break; }; done
IMG=${X129_IMAGE:-radiance-local:c16-sw3}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-hcw2; REF=fn4-abval2
export FNIMAGE=$IMG FNMODELDIR=$HOME/models/flashnext-iq4r FNMODEL=qwen3.8-next-flash-fp8-iq4r-moe.rad
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
export EXTRA="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats129
export FN4NR="" FN4GI="" FN4FD="" FN4NW=1 FN4SW=1
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
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 129: VIRHE temperature guard ($hot) -> blocks stopped"; return 0; }
    sleep 1; done; }
restart() {   # restart TAG -> 0 healthy
  local tag=$1 ok=0 t0 i S=$O/start-$1.log
  [ "$DRY" = 1 ] && { echo "+ FNIMAGE=$FNIMAGE FN4NW=$FN4NW FN4SW=$FN4SW EXTRA='$EXTRA' compose -f $Y up -d; health <= 1200 s; docker logs > $S"; return 0; }
  down; t0=$(date +%s); compose up -d > $O/up-$tag.txt 2>&1
  for i in $(seq 1 240); do
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' $URL/health)" = 200 ] && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Status}}' $CN 2>/dev/null)" != running ] && break
    sleep 5
  done
  docker logs $CN > $S 2>&1
  [ $ok = 1 ] || { log "RESULT fn4 129 restart-$tag: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
  log "RESULT fn4 129 restart-$tag: healthy in $(( $(date +%s) - t0 )) s; $(grep -m1 -E '^I kv: ' $S | cut -c3-160); fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300)"
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
    if cmp -s $1 $2; then log "RESULT fn4 129 identity $3: IDENTICAL ($(wc -c < $2) bytes)"
    else log "RESULT fn4 129 identity $3: VIRHE DIFFERENT from byte $(cmp $1 $2 2>&1 | grep -oE 'byte [0-9]+' | head -1)"; fi
  else log "RESULT fn4 129 identity $3: not compared (missing $1 or $2)"; fi
}
warmcmp() {   # warmcmp FILE_A FILE_B LABEL (informational)
  [ "$DRY" = 1 ] && { echo "+ warm-up answers $1 vs $2"; return 0; }
  [ -s $1 ] && [ -s $2 ] || { log "RESULT fn4 129 warm-up $3: not compared (missing $1 or $2)"; return 0; }
  log "RESULT fn4 129 warm-up $3: $(python3 -c 'import json,sys
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
  run "timeout 5400 python3 -I $TL/abblocks2.py --url $URL --mb-dir $D --label $TAG-$nm-$ph --phase $ph --dur ${X129_DUR:-20} --warm ${X129_WARM:-6} --rounds $rr --a '$4' --b '$5' ${pf:+--pf-depths $pf} --names plain,$nm --out $O/$TAG-blocks.jsonl > $O/$TAG-$nm-$ph.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 129 $nm $ph${pf:+ + prefill $pf}: $(grep '^ABB ' $O/$TAG-$nm-$ph.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=4: "-" if v is None else format(v, ".%df" % n)
s = "decode tok/s B/A %s +- %s (t95, %d rounds: %s) | ms/step B/A %s +- %s | A %s B %s tok/s | %d blocks, %d flagged, %s s" % (
    f(d["tps_ratio"]), f(d["tps_ci95"]), len(d["rounds"]), " ".join(f(r["tps_ratio"]) for r in d["rounds"]),
    f(d["ms_ratio"]), f(d["ms_ci95"]), f(d["A_tps_mean"], 1), f(d["B_tps_mean"], 1), d["blocks"], d["blocks_bad"], d["wall_s"])
for k, v in (d.get("pf") or {}).items():
    s += " | prefill %s B/A %s +- %s (%s) A %s B %s tok/s" % (k, f(v["ratio"]), f(v["ci95"]), " ".join(f(x) for x in v["rounds"]), f(v["A_mean"], 1), f(v["B_mean"], 1))
print(s)' 2>&1 | cut -c1-1200)"
  [ "$DRY" = 1 ] || [ -z "$pf" ] || log "RESULT fn4 129 $nm $ph prefill windows' thermal lines: $(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("label") == sys.argv[2] and r.get("pf_therm"): print("r%d%s %s;" % (r["round"], r["arm"], r["pf_therm"]), end=" ")' $O/$TAG-blocks.jsonl $TAG-$nm-$ph 2>&1 | cut -c1-1500)"
}
OFF='{"hc_writevec": 0}'; ON='{"hc_writevec": 1}'
log "RESULT fn4 129-fn-hcw2: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) F1n RADIANCE_FN4_SWITCHABLE=1 EXTRA '$EXTRA' MNBT $MNBT | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
# ---- STEP 0: the selftests on one card, before any server
if [ "$DRY" = 1 ]; then
  for c in hc_write_bf16 hc_read_e4m3; do
    echo "+ timeout -k 15 1200 docker run --rm --init --name fn4-129-st-\$\$ --network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$C3 --entrypoint /opt/radiance/bin/r4d_selftest $IMG /opt/radiance/share/radiance/kernels/libr4d.so --case $c > $O/129-selftest-$c.txt"
  done
else
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 129: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 129: VIRHE image $IMG missing"; exit 1; }
  [ -e "$Y" ] && grep -q RADIANCE_FN4_SWITCHABLE $Y && [ -n "$TL" ] || { log "RESULT fn4 129: VIRHE $YN (with RADIANCE_FN4_SWITCHABLE) or tools/abblocks2.py + heatsoak.sh missing"; exit 1; }
  mkdir -p $O $O/guard $STATSDIR
  bad=0
  for c in hc_write_bf16 hc_read_e4m3; do
    f=$O/129-selftest-$c.txt
    timeout -k 15 1200 docker run --rm --init --name fn4-129-st-$$ --network none --device /dev/kfd --device /dev/dri \
      --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$C3 --entrypoint /opt/radiance/bin/r4d_selftest $IMG \
      /opt/radiance/share/radiance/kernels/libr4d.so --case $c > $f 2>&1
    rc=$?; docker rm -f fn4-129-st-$$ > /dev/null 2>&1
    np=$(grep -cE '^ +ok ' $f); nf=$(grep -cE '^ +FAIL ' $f); ns=$(grep -cE '^ +skip ' $f)
    log "RESULT fn4 129 selftest $c (card c3:00, $IMG): exit $rc, $np ok / $nf FAIL / $ns skip; fn4 lines: $(grep -E 'eight-wide|folded|wave norm' $f | grep -E ' ok | FAIL | skip ' | sed 's/^ *//' | cut -c1-140 | head -8 | tr '\n' '|' | cut -c1-1100)$( [ $nf -gt 0 ] && echo " | FAILS: $(grep -E '^ +FAIL ' $f | head -4 | sed 's/^ *//' | cut -c1-200 | tr '\n' '|')")"
    case $rc in 0|2) ;; *) bad=1 ;; esac
    [ $nf -gt 0 ] && bad=1
    [ "$c" = hc_write_bf16 ] && [ $np -lt 8 ] && bad=1   # the eight hc_write cases must have run (none skipped)
  done
  [ $bad = 0 ] || { log "RESULT fn4 129: VIRHE a selftest failed, crashed or skipped the hc_write cases -- no server runs ($O/129-selftest-*.txt)"; exit 1; }
  find $O -maxdepth 1 -name "$TAG.abort" -delete; find $O -maxdepth 1 -name "$TAG.tstop" -delete; find $O -maxdepth 1 -name "$TAG-blocks.jsonl" -delete
  trap 'touch $O/$TAG.tstop 2>/dev/null; down' EXIT
  trap 'exit 143' TERM INT
  tguard & TGPID=$!
fi
[ "$DRY" = 1 ] && { echo "+ guards; tguard &"; TGPID=""; }
restart $TAG || exit 1
if [ "$DRY" != 1 ]; then
  (cd $D && timeout 600 python3 ds41_tests.py correct --only-q --effort none --outdir $O/guard > $O/guard/$TAG-warm-q.txt 2>&1)
  (cd $D && timeout 600 python3 ds41_tests.py prefill --depths 16384 --cpt 3.6 --calibrate 0 > $O/guard/$TAG-warm-pf.txt 2>&1)
  timeout 300 python3 $Q/prodfc.py $URL 384 > $O/guard/$TAG-fc1.txt 2>&1
  log "RESULT fn4 129 1-user: $(grep '^FC ' $O/guard/$TAG-fc1.txt | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()[3:])["ms_per_step"])' 2>/dev/null || echo n/a) ms/step (informational)"
fi
warmcmp $O/guard/$REF-warm-q.txt $O/guard/$TAG-warm-q.txt "c16-sw3 hc_writevec 0 vs 128's c16-sw2 (informational)"
tries=0
while :; do
  set -- $(probe8 $TAG-st$tries); ms=$1
  st=$(awk -v m=$ms 'BEGIN{print (m <= 0) ? "unknown" : (m <= 26.8) ? "fast" : (m >= 28.5) ? "slow" : "between"}')
  log "RESULT fn4 129 state probe $tries: mb c8 $ms ms -> $st"
  { [ "$st" = slow ] && [ $tries -lt 2 ]; } || break
  tries=$((tries + 1)); restart $TAG-r$tries || exit 1
done
# IDENTITY. (a) cold vs cold across processes: the same 33k copy prompt with the eight-wide form against 128's cold run
# (c16-sw2, the plain form, the same server line). (b) in this process, both resuming from the cache (a hit and a cold
# prompt are not byte-identical under --prefix-resume nearest): the plain form against the eight-wide one.
setsw "$ON"; copyt coldvec
cmpf $O/copy-$REF-cold-run1.txt $O/copy-$TAG-coldvec-run1.txt "(a) cold 33k copy, hc_writevec 1 (c16-sw3) vs 128's cold copy (c16-sw2)"
setsw "$OFF"; copyt off
setsw "$ON"; copyt on
cmpf $O/copy-$TAG-off-run1.txt $O/copy-$TAG-on-run1.txt "(b) cache-resumed 33k copy, hc_writevec 1 vs 0 in one process"
[ "$DRY" = 1 ] || log "RESULT fn4 129 switch log: $(docker logs $CN 2>&1 | grep -E 'fn4: switches' | tail -3 | sed 's/^[A-Z] //' | tr '\n' '|' | cut -c1-500)"
[ "$DRY" = 1 ] || log "RESULT fn4 129 A/B start: thermal now $(thermnow)"
abrun hcw c16 ${X129_RPF:-3} "$OFF" "$ON" 16384,65536,131072
abrun hcw c8 ${X129_R8:-2} "$OFF" "$ON"
run "docker logs $CN > $O/$TAG-engine.log 2>&1"
[ "$DRY" = 1 ] || log "RESULT fn4 129 engine: $(grep -cE 'fn4: switches .* applied' $O/$TAG-engine.log) applied changes, $(grep -cE 'fn4: .* refused' $O/$TAG-engine.log) refused; $(grep -E 'refused' $O/$TAG-engine.log | grep fn4 | head -2 | tr '\n' '|' | cut -c1-300)"
down
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
log "RESULT fn4 129-fn-hcw2: done"
