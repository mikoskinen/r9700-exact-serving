#!/bin/bash
# 160-fn-x3codes.sh (queue item, 10.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~45 min) -- E12,
# from 152 (EXL3 per-op profile) read against 148's stamps and 147's iq4r ones: the EXL3 c16 pass is 35.9 ms vs iq4r's 34.0
# (+1.9 ms; c4 19.4 vs 17.6 ms a step, the same offset), and the difference is not the experts (per layer-pass equal) but
# quant_act_fp8<false, true> -- 1.64 ms of the EXL3 c16 pass, absent in iq4r's -- plus ~196 more dispatches a pass (+0.95 ms
# of launch gaps). Cause: the connections write int8 codes only when experts are rotated or bf16
# (qwen4exp_fp8.cpp codes_i8); EXL3 experts read the bf16 plane too but were not counted, so every int8 trunk linear
# quantises its input itself. RADIANCE_FN4_X3CODES=1 (fn4-x3 ee393a41) counts them. EXACT by construction:
# hc_quant_group<true> and quant_act_fp8_body<false, true> take one amax a 128-group of the same bf16 x, scale m * (1/127),
# rintf of the clamped quotient. Declare-time, so an A/B ACROSS RESTARTS, A B B A (compose fn-tp4z.yaml, FN4X3C):
# image radiance-local:c16-x3e (fn4-x3 ee393a41 = c16-x3d's engine + X3CODES; exl3r4d e345bb0), the EXL3 4.05 .rad, the
# x3stk line. Per server: warm-up, DECODE mb c16 / c8 / c4 (30 s, 8 s warm), then PREFILL fresh 16k / 64k / 128k.
# IDENTITY: the 33k copy cold on the first A and the first B server, and the warm-up answers: must be IDENTICAL.
# Target: EXL3 at iq4r's speed (c4 / c8 / c16 572.3 / 770.1 / 1022.5, prefill 8480 / 8554 / 8483). Power untouched.
# DRYRUN=1 bash 160-fn-x3codes.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4z.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/abblocks2.py" ] && [ -e "$f/heatsoak.sh" ] && { TL=$f; break; }; done
IMG=${X160_IMAGE:-radiance-local:c16-x3e}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-x3codes; REF=fn4-abval2
export FNIMAGE=$IMG FNMODELDIR=$HOME/models/flashnext-exl3 FNMODEL=qwen3.8-flash-next-exl3.rad
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
export EXTRA="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats160
export FN4NR=1 FN4GI=1 FN4FD="" FN4NW=1 FN4SW="" FN4WV=1 FN4RV=11 FN4AL="" FN4UM=5 FN4SS=1 FN4CH="" FN4XM="" FN4SPHC="" FN4X3C="" PROFEVERY=""
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
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 160: VIRHE temperature guard ($hot) -> blocks stopped"; return 0; }
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
  [ $ok = 1 ] || { log "RESULT fn4 160 restart-$tag: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
  log "RESULT fn4 160 restart-$tag: healthy in $(( $(date +%s) - t0 )) s; $(grep -m1 -E '^I kv: ' $S | cut -c3-160); fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300)"
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
    if cmp -s $1 $2; then log "RESULT fn4 160 identity $3: IDENTICAL ($(wc -c < $2) bytes)"
    else log "RESULT fn4 160 identity $3: VIRHE DIFFERENT from byte $(cmp $1 $2 2>&1 | grep -oE 'byte [0-9]+' | head -1)"; fi
  else log "RESULT fn4 160 identity $3: not compared (missing $1 or $2)"; fi
}
warmcmp() {   # warmcmp FILE_A FILE_B LABEL (informational)
  [ "$DRY" = 1 ] && { echo "+ warm-up answers $1 vs $2"; return 0; }
  [ -s $1 ] && [ -s $2 ] || { log "RESULT fn4 160 warm-up $3: not compared (missing $1 or $2)"; return 0; }
  log "RESULT fn4 160 warm-up $3: $(python3 -c 'import json,sys
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
  run "timeout 5400 python3 -I $TL/abblocks2.py --url $URL --mb-dir $D --label $TAG-$nm-$ph --phase $ph --dur ${X160_DUR:-20} --warm ${X160_WARM:-6} --rounds $rr --a '$4' --b '$5' ${pf:+--pf-depths $pf} --names plain,$nm --out $O/$TAG-blocks.jsonl > $O/$TAG-$nm-$ph.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 160 $nm $ph${pf:+ + prefill $pf}: $(grep '^ABB ' $O/$TAG-$nm-$ph.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=4: "-" if v is None else format(v, ".%df" % n)
s = "decode tok/s B/A %s +- %s (t95, %d rounds: %s) | ms/step B/A %s +- %s | A %s B %s tok/s | %d blocks, %d flagged, %s s" % (
    f(d["tps_ratio"]), f(d["tps_ci95"]), len(d["rounds"]), " ".join(f(r["tps_ratio"]) for r in d["rounds"]),
    f(d["ms_ratio"]), f(d["ms_ci95"]), f(d["A_tps_mean"], 1), f(d["B_tps_mean"], 1), d["blocks"], d["blocks_bad"], d["wall_s"])
for k, v in (d.get("pf") or {}).items():
    s += " | prefill %s B/A %s +- %s (%s) A %s B %s tok/s" % (k, f(v["ratio"]), f(v["ci95"]), " ".join(f(x) for x in v["rounds"]), f(v["A_mean"], 1), f(v["B_mean"], 1))
print(s)' 2>&1 | cut -c1-1200)"
  [ "$DRY" = 1 ] || [ -z "$pf" ] || log "RESULT fn4 160 $nm $ph prefill windows' thermal lines: $(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("label") == sys.argv[2] and r.get("pf_therm"): print("r%d%s %s;" % (r["round"], r["arm"], r["pf_therm"]), end=" ")' $O/$TAG-blocks.jsonl $TAG-$nm-$ph 2>&1 | cut -c1-1500)"
}
log "RESULT fn4 160-fn-x3codes: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) the EXL3 line on the main stack, X3CODES off / on (RADIANCE_FN4_X3CODES) across restarts A B B A | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 160: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 160: VIRHE image $IMG missing"; exit 1; }
  [ -e "$Y" ] && grep -q RADIANCE_FN4_X3CODES $Y || { log "RESULT fn4 160: VIRHE $YN (with RADIANCE_FN4_X3CODES) missing"; exit 1; }
  mkdir -p $O $O/guard $STATSDIR
  find $O -maxdepth 1 -name "$TAG.abort" -delete; find $O -maxdepth 1 -name "$TAG.tstop" -delete
  trap 'touch $O/$TAG.tstop 2>/dev/null; down' EXIT
  trap 'exit 143' TERM INT
  tguard & TGPID=$!
else
  echo "+ guards; tguard &"; TGPID=""
fi
ORDER=${X160_ORDER:-"off on on off"}
k=0; firstoff=""; firston=""
for arm in $ORDER; do
  [ -e $O/$TAG.abort ] && break
  k=$((k + 1)); nm=$arm$k
  if [ $arm = on ]; then export FN4X3C=1; else export FN4X3C=""; fi
  restart $TAG-$nm || exit 1
  [ "$DRY" = 1 ] || log "RESULT fn4 160 $nm: $(grep -m1 -iE 'EXL3 experts' $O/start-$TAG-$nm.log | sed 's/^[A-Z] //' | cut -c1-200)"
  if [ "$DRY" != 1 ]; then
    (cd $D && timeout 600 python3 ds41_tests.py correct --only-q --effort none --outdir $O/guard > $O/guard/$TAG-$nm-warm-q.txt 2>&1)
  fi
  if [ $arm = off ] && [ -z "$firstoff" ]; then copyt $nm; firstoff=$nm; fi
  if [ $arm = on ] && [ -z "$firston" ]; then copyt $nm; firston=$nm; fi
  run "(cd $D && timeout 900 python3 mb.py --tag $TAG-$nm --phases c16,c8,c4 --dur ${X160_DUR:-30} --warm ${X160_WARM:-8} > $O/$TAG-$nm-mb.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT fn4 160 $nm mb: $(grep '^MB ' $O/$TAG-$nm-mb.txt | cut -c4- | python3 -c 'import json,sys
for l in sys.stdin:
    d = json.loads(l); print("%s %s tok/s %s ms/step acc %s;" % (d.get("phase"), d.get("agg_dec_tps"), d.get("ms_decode_step"), d.get("acceptance")), end=" ")' 2>&1 | cut -c1-400)"
  run "(cd $D && timeout 1500 python3 ds41_tests.py prefill --depths 16384,65536,131072 --cpt 3.6 --calibrate 0 --reps 1 > $O/$TAG-$nm-pf.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT fn4 160 $nm prefill: $(python3 -c 'import json,sys; print("; ".join("%s tok %s tok/s" % (d.get("prompt_tokens"), d.get("prefill_tps")) for d in (json.loads(l[2:]) for l in open(sys.argv[1]) if l.startswith("J "))))' $O/$TAG-$nm-pf.txt 2>&1 | cut -c1-300)"
  run "docker logs $CN > $O/$TAG-$nm-engine.log 2>&1"
  down
done
cmpf $O/copy-$TAG-$firstoff-run1.txt $O/copy-$TAG-$firston-run1.txt "cold 33k copy, X3CODES on ($firston) vs off ($firstoff), separate processes"
warmcmp $O/guard/$TAG-$firstoff-warm-q.txt $O/guard/$TAG-$firston-warm-q.txt "X3CODES on vs off (cold, separate processes)"
[ "$DRY" = 1 ] || log "RESULT fn4 160 X3CODES on / off (B/A per round, then the geometric mean): $(python3 - $O $TAG "$ORDER" <<'PY' 2>&1 | cut -c1-900
import json, math, sys
O, tag, order = sys.argv[1], sys.argv[2], sys.argv[3].split()
runs = []
for k, arm in enumerate(order, 1):
    d = {}
    try:
        for l in open(f"{O}/{tag}-{arm}{k}-mb.txt"):
            if l.startswith("MB "):
                m = json.loads(l[3:]); d[m.get("phase")] = m
    except OSError:
        pass
    runs.append((arm, d))
out = []
for ph in ("c16", "c8", "c4"):
    rs = []
    for i in range(0, len(runs) - 1, 2):
        (a1, d1), (a2, d2) = runs[i], runs[i + 1]
        on, off = (d1, d2) if a1 == "on" else (d2, d1)
        x, y = on.get(ph, {}).get("agg_dec_tps"), off.get(ph, {}).get("agg_dec_tps")
        if x and y: rs.append(x / y)
    acc = [(d.get(ph, {}).get("acceptance"), a) for a, d in runs]
    accs = lambda w: [v for v, a in acc if a == w and v is not None]
    gm = math.exp(sum(map(math.log, rs)) / len(rs)) if rs else None
    out.append("%s %s (rounds %s; acceptance on %s off %s)" % (ph, "%.4f" % gm if gm else "-", " ".join("%.4f" % r for r in rs),
               "%.3f" % (sum(accs("on")) / len(accs("on"))) if accs("on") else "-", "%.3f" % (sum(accs("off")) / len(accs("off"))) if accs("off") else "-"))
pf = {}
for k, arm in enumerate(order, 1):
    try:
        for l in open(f"{O}/{tag}-{arm}{k}-pf.txt"):
            if l.startswith("J "):
                d = json.loads(l[2:])
                if d.get("prefill_tps"): pf.setdefault((arm, d.get("target")), []).append(d["prefill_tps"])
    except OSError:
        pass
for t in (16384, 65536, 131072):
    a, b = pf.get(("off", t)), pf.get(("on", t))
    if a and b: out.append("prefill %d %.4f (off %s on %s)" % (t, (sum(b) / len(b)) / (sum(a) / len(a)), a, b))
print("; ".join(out))
PY
)"
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
log "RESULT fn4 160-fn-x3codes: done"
