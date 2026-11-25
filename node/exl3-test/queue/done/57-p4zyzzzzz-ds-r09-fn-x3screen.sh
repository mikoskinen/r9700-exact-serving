#!/bin/bash
# 189-fn-x3screen.sh (queue item, 12.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~20 min) -- QUICK
# SCREENS (user 12.11.: too slow; one start each or merged, the target metric only, A B A B with 2 blocks a side, short
# windows; exactness = the 20-prompt greedy set alone on vs off; the block spread is a guide, judged with the size, direction and whether the path ran). MERGES r09 (181 wirecost)
# and r15 (166 x3pair): both on the EXL3 line's image radiance-local:c16-x3arm (fn4-x3 38aa3e7b, which carries P2x c02aa2c4)
# at 181's base (X3CODES + exl3_gd 1, compose fn-tp4zq.yaml), RADIANCE_FN4_P2X=1 declared in EVERY start (if that start
# fails, the starts fall back to undeclared and the P2x screen is skipped with a VIRHE line).
#  WIRE (serve flag, so A B A B = 4 starts): A = the line (--tp-wire wht6, min-kb 128), B = + --tp-wire exact. Each start:
#    one c16 decode block (mb c16, 20 s, warm 6; at c16 the MTP verify rows pass the 128 KiB lossy threshold) + one prefill
#    block (fresh 16k + 64k: the large messages). Exactness: greedy-20 alone in a1 vs b1.
#  P2x (in process, in start a1 after its wire blocks): moe_pair 0 vs 1 and 0 vs 2, prefill-only blocks A B B A (16k + 64k,
#    abblocks2 --no-mb). Exactness: a FRESH 33k copy (prefix cache reset) under moe_pair 0 / 1 / 2, byte for byte -- the
#    20-prompt greedy set never reaches P2x (its prompts are < 2048 tokens: one piece, no pipelined pair). The engine's
#    P2x notes say whether the pair ran.
# DROPPED (12.11. re-scope): 181's c1 / c4 / c8 / pf phases + the 160-load c16 + the 2 x 45 s windows; 166's 165 condition,
# 148 cross-process copy, decode blocks, 128k depth, 3-round A B B A and mb pf rounds. Power untouched. NO cc_sim.
# DRYRUN=1 bash 189-fn-x3screen.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4zq.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/heatsoak.sh" ] && { TL=$f; break; }; done
IMG=${X189_IMAGE:-radiance-local:c16-x3arm}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-x3scr
export FNIMAGE=$IMG FNMODELDIR=${X189_X3_DIR:-$HOME/models/flashnext-exl3} FNMODEL=${X189_X3_MODEL:-qwen3.8-flash-next-exl3.rad}
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
EXTRA0="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0"
export EXTRA="$EXTRA0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats189
export FN4NR=1 FN4GI=1 FN4FD="" FN4NW=1 FN4SW="" FN4WV=1 FN4RV=11 FN4AL="" FN4UM=5 FN4SS=1 FN4CH="" FN4XM="" FN4SPHC="" FN4X3C=1 FN4QSA="" FN4P2X=1 FN4MP="" FN4X3GD=1 FN4X3DZ="" PROFEVERY=""
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
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 189: VIRHE temperature guard ($hot) -> blocks stopped"; return 0; }
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
  [ $ok = 1 ] || { log "RESULT fn4 189 restart-$tag: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
  log "RESULT fn4 189 restart-$tag: healthy in $(( $(date +%s) - t0 )) s; $(grep -m1 -E '^I kv: ' $S | cut -c3-160); fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300)"
}
probe8() {   # probe8 TAG -> "ms tok/s" of a 45 s mb c8
  [ "$DRY" = 1 ] && { echo "+ (cd $D && timeout 400 python3 mb.py --tag $1 --phases c8 --dur 45 --warm 10 > $O/$1-c8.txt)" >&2; echo "25.6 744"; return 0; }
  (cd $D && timeout 400 python3 mb.py --tag $1 --phases c8 --dur 45 --warm 10 > $O/$1-c8.txt 2>&1)
  grep '^MB ' $O/$1-c8.txt 2>/dev/null | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[3:]); print(d.get("ms_decode_step"), d.get("agg_dec_tps"))' 2>/dev/null || echo "0 0"
}
setsw() {   # setsw JSON: queue switches (they land at the next step release)
  run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"
}
GS=; for f in "$H/tools/greedyset.py" "$H/../tools/greedyset.py" "$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py"; do [ -z "$GS" ] && [ -e "$f" ] && GS=$f; done
alive() {   # alive WHAT: the server still answers, else logs + down + exit
  [ "$DRY" = 1 ] && return 0
  local code; code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' $URL/health)
  [ "$code" = 200 ] && return 0
  docker logs $CN > $O/$TAG-dead-engine.log 2>&1
  log "RESULT fn4 189: VIRHE the server is not healthy after $1 (/health $code; last E lines: $(grep -E '^E |trap|abort|HSA_STATUS' $O/$TAG-dead-engine.log | tail -3 | tr '\n' ' ' | cut -c1-400)) -> logs saved, down"
  down; exit 1
}
LD=$TL/fn-loadid.py
PFD=${X189_PFD:-16384,65536}
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
swcheck() {   # swcheck WHAT SWITCHES_JSON: every switch named there has that value now
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4: $2"; return 0; }
  local g; g=$(curl -s -m 10 $URL/admin/fn4)
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); w=json.loads(sys.argv[2]); sys.exit(0 if all(d["switches"].get(k) == v for k, v in w.items()) else 1)' "$g" "$2" 2>/dev/null \
    || { log "RESULT fn4 189 $1: VIRHE the switches are not $2 ($(echo "$g" | cut -c1-400))"; return 1; }
}
pcreset() { [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o /dev/null -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  [ "$c" = 200 ] || log "RESULT fn4 189 $1 prefix cache reset: HTTP $c"; }
c16blk() { [ -e $O/$TAG.abort ] && return 0; run "(cd $D && timeout 400 python3 mb.py --tag $TAG-$1 --phases c16 --dur 20 --warm 6 > $O/$TAG-$1-c16.txt 2>&1)"; }
pfblk() { [ -e $O/$TAG.abort ] && return 0; run "(cd $D && timeout 600 python3 ds41_tests.py prefill --depths $PFD --cpt 3.6 --calibrate 0 --reps 1 > $O/$TAG-$1-pf.txt 2>&1)"; }
gset() { run "(cd $D && timeout 900 python3 $GS $URL $O/$TAG-gs-$1.jsonl --conc 1 --max-tokens 192 > $O/$TAG-gs-$1.txt 2>&1)"; }
copyt() {   # copyt NAME: a FRESH 33k copy (the prefix cache reset first), greedy; output $O/copy-$TAG-NAME-run1.txt
  pcreset copy-$1
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset 0 --tag=-$TAG-$1 --outdir $O > $O/$TAG-$1-copy.txt 2>&1)"
}
cmpf() {   # cmpf FILE_A FILE_B LABEL
  if [ "$DRY" = 1 ]; then echo "+ cmp $1 $2"
  elif [ -s $1 ] && [ -s $2 ]; then
    if cmp -s $1 $2; then log "RESULT fn4 189 identity $3: IDENTICAL ($(wc -c < $2) bytes)"
    else log "RESULT fn4 189 identity $3: DIFFERENT from byte $(cmp $1 $2 2>&1 | grep -oE 'byte [0-9]+' | head -1)"; fi
  else log "RESULT fn4 189 identity $3: not compared (missing $1 or $2)"; fi
}
mpair() {   # mpair N: moe_pair N, landed (an epoch bump, a nudge, checked)
  setsw "{\"moe_pair\": $1, \"epoch_bump\": 1}"; nudge; swcheck "moe_pair $1" "{\"moe_pair\": $1}"
}
abpf() {   # abpf NAME A_JSON B_JSON: prefill-only blocks A B B A (one round = 2 blocks a side)
  [ -e $O/$TAG.abort ] && return 0
  run "timeout 1800 python3 -I $TL/abblocks2.py --url $URL --mb-dir $D --label $TAG-$1 --rounds 1 --a '$2' --b '$3' --pf-depths $PFD --no-mb --names off,$1 --out $O/$TAG-blocks.jsonl > $O/$TAG-$1.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 189 $1 prefill (B/A of prefill tok/s, A B B A, 2 blocks a side): $(grep '^ABB ' $O/$TAG-$1.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
for k, v in (d.get("pf") or {}).items():
    print("%s B/A %.4f %s (A %s | B %s tok/s);" % (k, v["ratio"] or 0, v.get("sep"), " ".join("%.0f" % x for x in v.get("A_blocks", [])), " ".join("%.0f" % x for x in v.get("B_blocks", []))), end=" ")
print("%d blocks, %d flagged, %s s" % (d["blocks"], d["blocks_bad"], d["wall_s"]))' 2>&1 | cut -c1-700)"
}
log "RESULT fn4 189-fn-x3screen: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) the EXL3 line (X3CODES + exl3_gd 1), RADIANCE_FN4_P2X=1 declared; wire A (wht6) / B (exact) starts ${X189_STARTS:-a1 b1 a2 b2}: c16 block + prefill $PFD block each, greedy-20 a1 vs b1; P2x in a1: fresh 33k copies 0/1/2 + prefill blocks 0 vs 1, 0 vs 2 | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 189: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 189: VIRHE image $IMG missing"; exit 1; }
  [ -e "$Y" ] && [ -n "$TL" ] && [ -e $TL/abblocks2.py ] && [ -e $LD ] && [ -n "$GS" ] && [ -e $D/mb.py ] || { log "RESULT fn4 189: VIRHE $YN, tools (abblocks2 / fn-loadid / greedyset) or mb.py missing"; exit 1; }
  grep -q -- '--no-mb' $TL/abblocks2.py || { log "RESULT fn4 189: VIRHE abblocks2.py has no --no-mb"; exit 1; }
  mkdir -p $O $STATSDIR
  find $O -maxdepth 1 -name "$TAG.abort" -delete; find $O -maxdepth 1 -name "$TAG.tstop" -delete
  find $O -maxdepth 1 \( -name "$TAG-*-c16.txt" -o -name "$TAG-*-pf.txt" -o -name "$TAG-gs-*.jsonl" -o -name "copy-$TAG-*-run1.txt" \) -delete
  trap 'touch $O/$TAG.tstop 2>/dev/null; down' EXIT
  trap 'exit 143' TERM INT
  tguard & TGPID=$!
else
  echo "+ guards; tguard &"; TGPID=""
  echo "+ tools: TL=$TL LD=$LD GS=$GS abblocks2 --no-mb: $(grep -c -- '--no-mb' $TL/abblocks2.py)"
fi
P2XOK=1
for st in ${X189_STARTS:-a1 b1 a2 b2}; do
  [ -e $O/$TAG.abort ] && break
  case $st in a*) export EXTRA="$EXTRA0" ;; b*) export EXTRA="$EXTRA0 --tp-wire exact" ;; esac
  if ! restart $TAG-$st; then
    [ "$FN4P2X" = 1 ] || continue
    log "RESULT fn4 189 P2x: VIRHE start $st did not come up with RADIANCE_FN4_P2X=1 -- the starts go on undeclared, the P2x screen skipped"
    export FN4P2X=""; P2XOK=0; restart $TAG-$st-nop2x || continue
  fi
  nudge
  c16blk $st; alive "the c16 block ($st)"
  pfblk $st; alive "the prefill block ($st)"
  case $st in a1|b1) gset $st; alive "greedy-20 ($st)" ;; esac
  if [ $st = b1 ] && [ "$DRY" != 1 ]; then
    r=$(timeout 300 python3 -I $LD cmp --url $URL --json $O/$TAG-cmp-gs.json --label "gs-b1-vs-a1" $O/$TAG-gs-a1.jsonl $O/$TAG-gs-b1.jsonl 2>&1 | grep -m1 '^CMPLINE ' | cut -c9-)
    log "RESULT fn4 189 wire exactness: greedy-20 alone, exact wire (b1) vs the line's (a1): ${r:-VIRHE compare failed}"
  fi
  if [ $st = a1 ] && [ $P2XOK = 1 ]; then
    mpair 1 && copyt p1; alive "the copy (moe_pair 1)"
    mpair 2 && copyt p2; alive "the copy (moe_pair 2)"
    mpair 0 && copyt p0; alive "the copy (moe_pair 0)"
    cmpf $O/copy-$TAG-p0-run1.txt $O/copy-$TAG-p1-run1.txt "P2x fresh 33k copy, moe_pair 1 vs 0"
    cmpf $O/copy-$TAG-p0-run1.txt $O/copy-$TAG-p2-run1.txt "P2x fresh 33k copy, moe_pair 2 vs 0"
    abpf p2x1 '{"moe_pair": 0}' '{"moe_pair": 1}'; alive "the P2x blocks (0 vs 1)"
    abpf p2x2 '{"moe_pair": 0}' '{"moe_pair": 2}'; alive "the P2x blocks (0 vs 2)"
    mpair 0
    run "docker logs $CN > $O/$TAG-$st-engine.log 2>&1"
    [ "$DRY" = 1 ] || log "RESULT fn4 189 P2x engine notes: $(grep -E 'fn4 P2x' $O/$TAG-$st-engine.log | sed 's/^[A-Z] //' | sort | uniq -c | head -4 | tr '\n' ' ' | cut -c1-500); $(grep -cE 'fn4: switches .* applied' $O/$TAG-$st-engine.log) applied changes, $(grep -cE 'fn4: .* refused' $O/$TAG-$st-engine.log) refused"
  fi
  down
done
if [ "$DRY" != 1 ]; then
  log "RESULT fn4 189 WIRE B/A (exact vs the line's; per side the two starts' blocks; the block spread is a guide, judged with the size, direction and whether the path ran): $(python3 - $O $TAG <<'PY' 2>&1 | cut -c1-1200
import json, sys
o, t = sys.argv[1], sys.argv[2]
def mbv(s, k):
    try:
        for l in open("%s/%s-%s-c16.txt" % (o, t, s)):
            if l.startswith("MB "): return json.loads(l[3:]).get(k)
    except OSError: return None
def pfv(s):
    r = {}
    try:
        for l in open("%s/%s-%s-pf.txt" % (o, t, s)):
            if l.startswith("J "):
                d = json.loads(l[2:])
                if d.get("test") == "prefill" and d.get("status") == 200 and d.get("prefill_tps"): r[str(d.get("target"))] = d["prefill_tps"]
    except OSError: pass
    return r
def sep(a, b, hi=True):
    if len(a) < 2 or len(b) < 2: return "n<2"
    if min(b) > max(a): return "B>A" if hi else "B<A (better)"
    if max(b) < min(a): return "B<A" if hi else "B>A (worse)"
    return "overlap"
out = []
for k, lab, hi in (("agg_dec_tps", "c16 decode tok/s", True), ("ms_decode_step", "c16 ms/step", False)):
    a = [v for v in (mbv(s, k) for s in ("a1", "a2")) if v]; b = [v for v in (mbv(s, k) for s in ("b1", "b2")) if v]
    out.append("%s B/A %s %s (A %s | B %s)" % (lab, "%.4f" % ((sum(b) / len(b)) / (sum(a) / len(a))) if a and b else "-", sep(a, b, hi),
               " ".join("%.1f" % x for x in a), " ".join("%.1f" % x for x in b)))
pa = [pfv(s) for s in ("a1", "a2")]; pb = [pfv(s) for s in ("b1", "b2")]
for d in sorted({k for x in pa + pb for k in x}, key=int):
    a = [x[d] for x in pa if d in x]; b = [x[d] for x in pb if d in x]
    out.append("prefill %s tok/s B/A %s %s (A %s | B %s)" % (d, "%.4f" % ((sum(b) / len(b)) / (sum(a) / len(a))) if a and b else "-", sep(a, b),
               " ".join("%.0f" % x for x in a), " ".join("%.0f" % x for x in b)))
print(" | ".join(out))
PY
)"
fi
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
log "RESULT fn4 189-fn-x3screen: done"
