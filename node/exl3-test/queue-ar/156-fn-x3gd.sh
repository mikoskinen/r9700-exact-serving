#!/bin/bash
# 156-fn-x3gd.sh (queue item, 10.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~40 min) -- E10: the
# EXL3 decode band's fused chain. The main-line EXL3 port issues split moe_exl3_gate_up / moe_exl3_down at every M; the
# interface v5 op moe_exl3_gate_up_down (exl3r4d_moe_gate_up_down_m16: one launch, down units ordered behind their
# expert's gate_up units) serves passes of 3..16 tokens (c1-c4 with MTP 3). The lab notes put the split down's c4 loss in
# its INTERCEPT (a dispatch's fixed cost + drain), which the fused launch removes; deeper prefetch for down was refuted
# there. Runtime switch exl3_gd (core -> qwen4exp_x3's rad_fn4_switch -> MoeFP8 issues the fused op) 0 vs 1 in process.
# Exact: from 3 tokens both forms are split-K 1, the fused y is the split chain's bits (perfmoe GD-CHECK in 153 D).
# GATE: runs only if 153 D shows GD-CHECK 0 bytes differ at every M and the fused op faster than the split chain at M 16
# (mean of its two runs); else logs the reason and exits 0.
# Image radiance-local:c16-x3d (fn4-x3 0ddf16fa = c16-x3c's engine + exl3_gd; exl3r4d e345bb0), the EXL3 4.05 .rad, the
# x3stk line (F1n + F2a + F2b + hcw + hc_readv 11 + HC_SS + hc_upmix 5, MNBT 4096 + span 2, interleave 0, nearest),
# compose fn-tp4y.yaml, exl3_pfprep / exl3_mapshare 0. BYTE CHECKS first: greedyset c1 / c4 (c4 twice under 0) and the
# cache-resumed copy, 0 vs 1. Then DECODE blocks A B B A mb c4 x5, c2 x4 (c8 / c16 passes are 32 / 64 rows: untouched).
# Power untouched; read-only temperature watcher.
# DRYRUN=1 bash 156-fn-x3gd.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4y.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/abblocks2.py" ] && [ -e "$f/heatsoak.sh" ] && { TL=$f; break; }; done
IMG=${X156_IMAGE:-radiance-local:c16-x3d}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-x3g; REF=fn4-abval2
export FNIMAGE=$IMG FNMODELDIR=${X156_X3_DIR:-$HOME/models/flashnext-exl3} FNMODEL=${X156_X3_MODEL:-qwen3.8-flash-next-exl3.rad}
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
export EXTRA="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats156
export FN4NR=1 FN4GI=1 FN4FD="" FN4NW=1 FN4SW="" FN4WV=1 FN4RV=11 FN4AL="" FN4UM=5 FN4SS=1 FN4CH=""
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
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 156: VIRHE temperature guard ($hot) -> blocks stopped"; return 0; }
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
  [ $ok = 1 ] || { log "RESULT fn4 156 restart-$tag: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
  log "RESULT fn4 156 restart-$tag: healthy in $(( $(date +%s) - t0 )) s; $(grep -m1 -E '^I kv: ' $S | cut -c3-160); fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300)"
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
    if cmp -s $1 $2; then log "RESULT fn4 156 identity $3: IDENTICAL ($(wc -c < $2) bytes)"
    else log "RESULT fn4 156 identity $3: VIRHE DIFFERENT from byte $(cmp $1 $2 2>&1 | grep -oE 'byte [0-9]+' | head -1)"; fi
  else log "RESULT fn4 156 identity $3: not compared (missing $1 or $2)"; fi
}
warmcmp() {   # warmcmp FILE_A FILE_B LABEL (informational)
  [ "$DRY" = 1 ] && { echo "+ warm-up answers $1 vs $2"; return 0; }
  [ -s $1 ] && [ -s $2 ] || { log "RESULT fn4 156 warm-up $3: not compared (missing $1 or $2)"; return 0; }
  log "RESULT fn4 156 warm-up $3: $(python3 -c 'import json,sys
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
  run "timeout 5400 python3 -I $TL/abblocks2.py --url $URL --mb-dir $D --label $TAG-$nm-$ph --phase $ph --dur ${X156_DUR:-20} --warm ${X156_WARM:-6} --rounds $rr --a '$4' --b '$5' ${pf:+--pf-depths $pf} --names plain,$nm --out $O/$TAG-blocks.jsonl > $O/$TAG-$nm-$ph.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 156 $nm $ph${pf:+ + prefill $pf}: $(grep '^ABB ' $O/$TAG-$nm-$ph.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=4: "-" if v is None else format(v, ".%df" % n)
s = "decode tok/s B/A %s +- %s (t95, %d rounds: %s) | ms/step B/A %s +- %s | A %s B %s tok/s | %d blocks, %d flagged, %s s" % (
    f(d["tps_ratio"]), f(d["tps_ci95"]), len(d["rounds"]), " ".join(f(r["tps_ratio"]) for r in d["rounds"]),
    f(d["ms_ratio"]), f(d["ms_ci95"]), f(d["A_tps_mean"], 1), f(d["B_tps_mean"], 1), d["blocks"], d["blocks_bad"], d["wall_s"])
for k, v in (d.get("pf") or {}).items():
    s += " | prefill %s B/A %s +- %s (%s) A %s B %s tok/s" % (k, f(v["ratio"]), f(v["ci95"]), " ".join(f(x) for x in v["rounds"]), f(v["A_mean"], 1), f(v["B_mean"], 1))
print(s)' 2>&1 | cut -c1-1200)"
  [ "$DRY" = 1 ] || [ -z "$pf" ] || log "RESULT fn4 156 $nm $ph prefill windows' thermal lines: $(python3 -c 'import json,sys
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
  log "RESULT fn4 156: VIRHE the server is not healthy after $1 (/health $code; last E lines: $(grep -E '^E |trap|abort|HSA_STATUS' $O/$TAG-dead-engine.log | tail -3 | tr '\n' ' ' | cut -c1-400)) -> logs saved, down"
  down; exit 1
}
gset() { run "(cd $D && timeout 900 python3 $GS $URL $O/$TAG-gs-$2.jsonl --conc $1 --max-tokens 192 > $O/$TAG-gs-$2.txt 2>&1)"; }
gcmp() { [ "$DRY" = 1 ] && { echo IDENTICAL; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
log "RESULT fn4 156-fn-x3gd: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) the EXL3 line on the main stack, exl3_gd 0 vs 1 in process | power: $( [ "$DRY" = 1 ] && echo dry || caps)"

# ---- GATE on 153 D (one card): GD-CHECK 0 bytes differ everywhere, the fused op faster at M 16
if [ "$DRY" != 1 ]; then
  gv=$(grep -E "RESULT fn4 153 D " $L 2>/dev/null | python3 -c '
import re, sys
ls = [l for l in sys.stdin]
if not ls: print("NO153D"); sys.exit()
bad = [l for l in ls if re.search(r"FAILED|DIFFER|GD-CHECK y [1-9]", l) or "GD-CHECK" not in l]
v = {}
for l in ls:
    m = re.search(r"D M (\d+): split chain ([\d.NA]+) / ([\d.NA]+) us, fused gd ([\d.NA]+) / ([\d.NA]+) us", l)
    if m:
        try: v[m.group(1)] = ((float(m.group(2)) + float(m.group(3))) / 2, (float(m.group(4)) + float(m.group(5))) / 2)
        except ValueError: pass
ok = "16" in v and v["16"][1] < v["16"][0]
print("GO" if ok and not bad else "NOGO %s bad=%d" % (v, len(bad)))' 2>&1)
  case "$gv" in GO) log "RESULT fn4 156: gate 153 D passed" ;; *) log "RESULT fn4 156: gate 153 D NOT passed ($gv) -- no server A/B"; exit 0 ;; esac
fi
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 156: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 156: VIRHE image $IMG missing"; exit 1; }
  [ -e "$Y" ] && [ -n "$TL" ] && [ -n "$GS" ] || { log "RESULT fn4 156: VIRHE $YN, tools/abblocks2.py or tools/greedyset.py missing"; exit 1; }
  mkdir -p $O $O/guard $STATSDIR
  find $O -maxdepth 1 -name "$TAG.abort" -delete; find $O -maxdepth 1 -name "$TAG.tstop" -delete; find $O -maxdepth 1 -name "$TAG-blocks.jsonl" -delete
  trap 'touch $O/$TAG.tstop 2>/dev/null; down' EXIT
  trap 'exit 143' TERM INT
  tguard & TGPID=$!
else
  echo "+ guards; tguard &"; TGPID=""
fi
restart $TAG || exit 1
if [ "$DRY" != 1 ]; then
  (cd $D && timeout 600 python3 ds41_tests.py correct --only-q --effort none --outdir $O/guard > $O/guard/$TAG-warm-q.txt 2>&1)
fi
alive "the warm-up"
tries=0
while :; do
  set -- $(probe8 $TAG-st$tries); ms=$1
  st=$(awk -v m=$ms 'BEGIN{print (m <= 0) ? "unknown" : "logged"}')   # the iq4r thresholds do not apply to EXL3: logged, no restart
  log "RESULT fn4 156 state probe $tries: mb c8 $ms ms -> $st"
  { [ "$st" = slow ] && [ $tries -lt 2 ]; } || break
  tries=$((tries + 1)); restart $TAG-r$tries || exit 1
done
# BYTE CHECKS: off (c1, c4 twice), then gd 1 (c1, c4); the copy cache-resumed in this process, 0 vs 1
OFF='{"exl3_gd": 0}'; ON='{"exl3_gd": 1}'
setsw "$OFF"; copyt warm
gset 1 off-c1; alive "greedyset c1 (off)"; gset 4 off-c4; alive "greedyset c4 (off)"; gset 4 off-c4b; alive "greedyset c4 (off, second)"
copyt off; alive "the copy (off)"
setsw "$ON"
gset 1 on-c1; alive "greedyset c1 (gd 1)"; gset 4 on-c4; alive "greedyset c4 (gd 1)"
copyt on; alive "the copy (gd 1)"
setsw "$OFF"
if [ "$DRY" != 1 ]; then
  f4=$(gcmp $O/$TAG-gs-off-c4.jsonl $O/$TAG-gs-off-c4b.jsonl); c1=$(gcmp $O/$TAG-gs-off-c1.jsonl $O/$TAG-gs-on-c1.jsonl); c4=$(gcmp $O/$TAG-gs-off-c4.jsonl $O/$TAG-gs-on-c4.jsonl)
  log "RESULT fn4 156 byte checks exl3_gd 1 vs 0: greedyset c1 $c1 | c4 $c4 (off's own c4 floor: $f4)"
  bad=0; case "$c1" in IDENTICAL*) ;; *) bad=1 ;; esac
  case "$f4" in IDENTICAL*) case "$c4" in IDENTICAL*) ;; *) bad=1 ;; esac ;; esac
  cmpf $O/copy-$TAG-off-run1.txt $O/copy-$TAG-on-run1.txt "cache-resumed 33k copy, exl3_gd 1 vs 0, one process"
  grep -q "identity cache-resumed 33k copy, exl3_gd 1 vs 0.*VIRHE" $L 2>/dev/null && [ "$(grep 'identity cache-resumed 33k copy, exl3_gd 1 vs 0' $L | tail -1 | grep -c VIRHE)" = 1 ] && bad=1
  run "docker logs $CN > $O/$TAG-engine-bytes.log 2>&1"
  log "RESULT fn4 156 EXL3 experts (engine): $(grep -iE 'EXL3 experts|expert parallel|kept plain bf16' $O/$TAG-engine-bytes.log | sed 's/^[A-Z] //' | sort -u | head -3 | tr '\n' '|' | cut -c1-400)"
  [ $bad = 0 ] || { log "RESULT fn4 156: VIRHE the fused chain changed bytes where 0 agrees with itself; no speed blocks"; exit 1; }
else
  cmpf $O/copy-$TAG-off-run1.txt $O/copy-$TAG-on-run1.txt "cache-resumed 33k copy, exl3_gd 1 vs 0, one process"
fi
[ "$DRY" = 1 ] || log "RESULT fn4 156 A/B start: fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300) | thermal now $(thermnow)"
abrun x3g-c4 c4 ${X156_RC4:-5} "$OFF" "$ON"
alive "the c4 blocks"
abrun x3g-c2 c2 ${X156_RC2:-4} "$OFF" "$ON"
alive "the c2 blocks"
run "docker logs $CN > $O/$TAG-engine.log 2>&1"
[ "$DRY" = 1 ] || log "RESULT fn4 156 engine: $(grep -cE 'fn4: switches .* applied' $O/$TAG-engine.log) applied changes, $(grep -cE 'fn4: .* refused' $O/$TAG-engine.log) refused"
down
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
log "RESULT fn4 156-fn-x3gd: done"
