#!/bin/bash
# 115-fn-tp4gaps16.sh (queue item, 9.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~25 min) --
# what are the ~5.0-5.3 ms of per-pass gaps at c16 (111b, k4n, stamps on all cards: c3 5282, 83 5025, 86 5136, 03 5328 us
# of a 37.7 ms pass, 14 %)? Two arms, each a fresh start with EVERY played pass stamped (RADIANCE_AQL_STAMP=1), no gate
# (it would stamp minutes of 1-user decode), a 12 s mb c16 window and a 12 s mb c8 window after 8 s of warm-up each, then
# down and the analysis on the complete stamp files:
#   k4nG  1.15.1 (the stampcards image) on the k4n line, all four cards stamped
#   rmnG  main 451b136 on the rmnI line (--prefill-interleave 0, nearest); main stamps card 0 only (the per-card patch is
#         1.15.1's); main chains every kernel of a step of <= 32 rows (b0a58b5), so c8 (32 rows) is chained there and c16
#         (~52-64) is not -- the c8 window shows what chaining does to the same gaps
# tools/gapcat.py per card and window: inside the trunk (verify) pass every idle stretch charged to the dispatch after it --
# stretch (host work between two played stretches: host ops, table refreshes, routing reports, KV staging), queue (a
# cross-queue dependency: the collective queue / a lane), chained, barrier (a dependent launch behind the barrier bit),
# unordered (the queue was empty) -- ranked in us a pass, a gap-length histogram, the costliest kernel pairs; per pass
# kind the wall / busy / gaps and the dispatch counts (barrier bit, chained); and the idle time BETWEEN passes by the two
# passes' kinds (trunk > history > draft rounds > next trunk: the host step loop's share). k4nG also through rankskew.py
# (per rank). Engine stats around each window. Power is not touched (the cards' caps, 210 W expected, are logged).
# DRYRUN=1 bash 115-fn-tp4gaps16.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4p.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
FN91=${X91_IMAGE:-radiance-local:1.15.1-stampcards-c0cbe5a}; export FNIMAGE=$FN91
export FNMODELDIR=$HOME/models/flashnext-iq4r FNMODEL=qwen3.8-next-flash-fp8-iq4r-moe.rad
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
CARDS4=$C3,$C83,$C86,$C03; CARDS2=$C3,$C03
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/rankskew.py" ] && [ -e "$f/pwsum.py" ] && [ -e "$f/slowprobe.py" ] && [ -e "$f/msnap.py" ] && { TL=$f; break; }; done
ARMS=${X115_ARMS:-"k4nG rmnG"}; ITEM=115-fn-tp4gaps16   # empty: k4s + the X93_WINNER pick (default auto: tools/pick93.py over the run log)
BBROOT=${BB_ROOT:-$HOME/gpu/ext/betterbench}; BBDEPS=${BB_DEPS:-$HOME/gpu/ext/bb-deps}
BBRUN=; for f in "$H/tools/bb_run.py" "$H/../tools/bb_run.py" "$HOME/gpu/ds41/exl3-test/queue-ar/tools/bb_run.py"; do [ -e "$f" ] && { BBRUN=$f; break; }; done
CCSIM=${X93_CCSIM:-$HOME/gpu/ds41/exl3-test/tools/ccsim/cc_sim.py}; CCWRAP=${X93_CCWRAP:-$HOME/gpu/ds41/exl3-test/ccsim-exl3-turns.sh}
CCU=${X93_CCU:-16}; CCTO=${X93_CCTIMEOUT:-3300}
CCARGS=${X93_CCARGS:-"--duration 3000 --turns 15 --warmup-turns 3 --effort xhigh,medium --gen-mode forced --seed 1 --cpt-fixed --temperature 0 --trace-seeded"}
FNP4=${X92_PR396_IMAGE:-radiance-local:1.15.1-pr396-0ae1545-stampcards}
FNPMIN=${X93_PMIN_IMAGE:-radiance-local:1.15.1-pmin-47e859a-stampcards}
FN17=${X93_IMAGE_17:-radiance-local:1.17.0-src}; FNMAIN=${X93_IMAGE_MAIN:-radiance-local:main-451b136}
FNMAINC0=${X93_IMAGE_MAINC0:-radiance-local:main-451b136-c0}; FNMAINK0D=${X93_IMAGE_MAINK0D:-radiance-local:main-451b136-k0d}
FNPROD=${X93_IMAGE_PROD:-stilldeadcode/radiance:1.15.1}; PRODY=${X93_PRODYAML:-$C/flashnext-prod-dp-a.yaml}
Y2N=fn-tp2.yaml; for Y2 in "$H/$Y2N" "$H/../$Y2N" "$HOME/gpu/ds41/exl3-test/queue-ar/$Y2N"; do [ -e "$Y2" ] && break; done
DQ=; for f in "$H/deepq.py" "$H/../deepq.py" "$HOME/gpu/ds41/exl3-test/queue-ar/deepq.py"; do [ -e "$f" ] && { DQ=$f; break; }; done
LIM=${FN_GATE_MS:-21}
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
compose() { docker compose --project-directory $C -f $Y "$@"; }
down() { run "compose down > /dev/null 2>&1; docker rm -f $CN > /dev/null 2>&1"; }
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
restart() {   # restart TAG
  local tag=$1 ok=0 t0 t1 i S=$O/start-$1.log
  [ "$DRY" = 1 ] && { echo "+ FNIMAGE=$FNIMAGE TP=$TP CARDS=$CARDS PLACE=$PLACE MML=$MML ROPE='$ROPE' MAXSEQS=$MAXSEQS MNBT=$MNBT EXTRA='${EXTRA:-}' compose -f $Y up -d; health <= 1200 s; docker logs > $S"; return 0; }
  down; t0=$(date +%s)
  compose up -d > $O/up-$tag.txt 2>&1
  for i in $(seq 1 240); do
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' $URL/health)" = 200 ] && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Status}}' $CN 2>/dev/null)" != running ] && break
    sleep 5
  done
  t1=$(date +%s); docker logs $CN > $S 2>&1
  [ $ok = 1 ] || { log "RESULT fn4 restart-$tag: VIRHE not healthy after $((t1-t0)) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  log "RESULT fn4 restart-$tag: healthy in $((t1-t0)) s (IMAGE=$FNIMAGE TP=$TP cards $CARDS PLACE=$PLACE MML=$MML ${ROPE:-no rope flag} MAXSEQS=$MAXSEQS MNBT=$MNBT); $(grep -m1 -E '^I kv: ' $S | cut -c3-200); VRAM experts $(grep -m1 -E '^ +experts +[0-9.]+ GiB' $S | awk '{print $2, $3, $4, $5}'), KV $(grep -m1 -E '^ +kv cache +[0-9.]+ GiB' $S | awk '{print $3, $4}'); $(grep -m1 -oE 'pinned by this process [0-9.]* GiB' $S); E lines $(grep -cE '^E ' $S)"
}
check() {   # check TAG N -> ms/step of prodfc.py (999 when it fails)
  local tag=$1 n=$2
  [ "$DRY" = 1 ] && { echo "+ warm-up (correct --only-q, prefill 16384) + prodfc.py $URL 384 -> $O/guard/$tag-fc$n.txt" >&2; echo 0; return; }
  mkdir -p $O/guard
  (cd $D && timeout 600 python3 ds41_tests.py correct --only-q --effort none --outdir $O/guard > $O/guard/$tag-warm-q.txt 2>&1)
  (cd $D && timeout 600 python3 ds41_tests.py prefill --depths 16384 --cpt 3.6 --calibrate 0 > $O/guard/$tag-warm-pf.txt 2>&1)
  timeout 300 python3 $Q/prodfc.py $URL 384 > $O/guard/$tag-fc$n.txt 2>&1
  grep '^FC ' $O/guard/$tag-fc$n.txt | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()[3:])["ms_per_step"])' 2>/dev/null || echo 999
}
gate() {   # gate TAG: 0 fast / 1 not healthy / 2 slow twice
  local tag=$1 ms ms1 st=fast
  restart $tag || return 1
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && return 0
  if awk -v m=$ms -v g=$LIM 'BEGIN{exit !(m > g)}'; then
    keeplog $tag-try1; ms1=$ms
    restart $tag || return 1; ms=$(check $tag 2); st="fast-after-retry (first $ms1)"
    awk -v m=$ms -v g=$LIM 'BEGIN{exit !(m > g)}' && { st="SLOW-after-retry (first $ms1)"; keeplog $tag-try2; }
  fi
  echo "$st $ms" > $O/guard-$tag.txt
  log "RESULT fn4 guard $tag: $st, 1-user ${ms} ms/step (limit $LIM; qgate method: correct --only-q + 16k prefill + one fastcheck; same method stock 1.15.1 TP2 83+86 17.71-17.82, c3+03 16.60-16.74; produp-method prod gates 14.8-15.22 are not comparable)"
  case $st in SLOW*) return 2 ;; esac
}
probe() { run "(cd $D && timeout 900 python3 stepprobe.py essay,code 512 > $O/$1-probe.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT fn4 $1 probe (1 user, greedy, thinking off, 512 tok): $(python3 $D/sweepfmt.py probe $O/$1-probe.txt 2>&1 | cut -c1-300)"; }
mbp() { run "(cd $D && timeout 1200 python3 mb.py --tag fn4-$1 --phases $2 > $O/$1-$2.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT fn4 $1 mb $2: $(grep '^MB ' $O/$1-$2.txt 2>/dev/null | cut -c4- | cut -c1-600 || tail -1 $O/$1-$2.txt)"; }
copyt() { run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset $2 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT fn4 $1 copy after a cold prompt: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact")))' 2>&1 | cut -c1-200)"; }
warmcmp() {   # warmcmp FILE_A FILE_B LABEL
  [ "$DRY" = 1 ] && { echo "+ warm-up answers $1 vs $2"; return 0; }
  [ -s $1 ] && [ -s $2 ] || { log "RESULT fn4 warm-up $3: not compared (missing $1 or $2)"; return 0; }
  log "RESULT fn4 warm-up $3: $(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if t in a and ((a[t].get("content") or "") != (b[t].get("content") or "") or a[t].get("finish") != b[t].get("finish"))]
n = len([t for t in b if t in a])
print(("IDENTICAL: %d answers" % n) if n and not bad else ("DIFFERENT: " + ", ".join(bad) if n else "no common answers"))' $1 $2 2>&1 | cut -c1-300)"
}
cmparm() {   # cmparm A B: arm B against arm A
  [ "$DRY" = 1 ] && { echo "+ RESULT fn4 $2 vs $1"; return 0; }
  log "RESULT fn4 $2 vs $1: $(python3 - $O fn4-$1 fn4-$2 <<'PY' 2>&1 | cut -c1-1500
import json, re, sys
O, a, b = sys.argv[1:4]
def mb(tag, p):
    try:
        for l in open(f"{O}/{tag}-{p}.txt"):
            if l.startswith("MB "): return json.loads(l[3:])
    except OSError: pass
    return {}
def gd(tag):
    try: return float(open(f"{O}/guard-{tag}.txt").read().split()[-1])
    except Exception: return None
def pr(tag):
    try: t = open(f"{O}/{tag}-probe.txt").read()
    except OSError: return {}
    out = {}
    for name in ("essay", "code"):
        m = re.search(name + r".*?([\d.]+) ms/step", t)
        if m: out[name] = float(m.group(1))
    return out
pa, pb = pr(a), pr(b)
rows = [("guard ms", gd(a), gd(b)), ("essay ms", pa.get("essay"), pb.get("essay")), ("code ms", pa.get("code"), pb.get("code"))]
for c in ("c4", "c8", "c16"):
    rows += [(f"{c} tok/s", mb(a, c).get("agg_dec_tps"), mb(b, c).get("agg_dec_tps")), (f"{c} ms", mb(a, c).get("ms_decode_step"), mb(b, c).get("ms_decode_step"))]
rows += [("pf tok/s", mb(a, "pf").get("prefill_tps"), mb(b, "pf").get("prefill_tps")), ("pf TTFT", mb(a, "pf").get("prefill_ttft"), mb(b, "pf").get("prefill_ttft")),
         ("pf kept", mb(a, "pf").get("decode_kept"), mb(b, "pf").get("decode_kept")),
         ("turns cold s", mb(a, "turns").get("cold_ttft"), mb(b, "turns").get("cold_ttft")), ("turns hit s", mb(a, "turns").get("ttft_hit_mean"), mb(b, "turns").get("ttft_hit_mean"))]
out = []
for n, x, y in rows:
    out.append(f"{n} {x} -> {y} ({(y / x - 1) * 100:+.1f} %)" if x and y else f"{n} {x} -> {y}")
print("; ".join(out))
PY
)"
}
probe() { run "(cd $D && timeout 900 python3 stepprobe.py essay,code 512 > $O/$1-probe.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT fn4 $1 probe (1 user, greedy, thinking off, 512 tok): $(python3 $D/sweepfmt.py probe $O/$1-probe.txt 2>&1 | cut -c1-300)"; }
pupair() {   # pupair TAG: prodfc.py twice back to back (the produp method's pair)
  run "timeout 300 python3 $Q/prodfc.py $URL 384 > $O/guard/$1-pu1.txt 2>&1; timeout 300 python3 $Q/prodfc.py $URL 384 > $O/guard/$1-pu2.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 $1 produp-method pair: $(for n in 1 2; do grep '^FC ' $O/guard/$1-pu$n.txt | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[3:]); print(d.get("ms_per_step"), end=" ")' 2>/dev/null || echo -n "n/a "; done)ms/step"
}
now() { python3 -c 'import time; print(time.clock_gettime_ns(time.CLOCK_MONOTONIC))'; }   # the stamps' clock
caps() {   # caps: every card's power1_cap / power1_average (W) and the hotspot temperature, by PCI slot (read-only)
  local d h out=""
  for d in /sys/class/drm/card*/device; do
    [ -e $d/unique_id ] || continue
    h=$(ls -d $d/hwmon/hwmon* 2>/dev/null | head -1); [ -n "$h" ] || continue
    out="$out $(basename $(readlink -f $d)):cap=$(( $(cat $h/power1_cap 2>/dev/null || echo 0) / 1000000 ))W,avg=$(( $(cat $h/power1_average 2>/dev/null || echo 0) / 1000000 ))W,hot=$(( $(cat $h/temp2_input 2>/dev/null || echo 0) / 1000 ))C"
  done
  echo "$out"
}
# ---- power (the user's go 8.11. 22:4x: power1_cap is root:render 0664, mikoskinen is in render; no sudo, no password)
hwmons() {   # hwmons: the hwmon directory of each of the four cards, in CARDS4 order
  local u d
  for u in ${CARDS4//,/ }; do
    for d in /sys/class/drm/card*/device; do
      [ "$(cat $d/unique_id 2>/dev/null)" = "${u#GPU-}" ] && { ls -d $d/hwmon/hwmon* 2>/dev/null | head -1; break; }
    done
  done
}
odsnap() {   # odsnap HWMON: the card's fan curve and VDDGFX offset, one line (read-only)
  local d=$(readlink -f $1/../..)
  echo "$(tr '\n' ' ' < $d/gpu_od/fan_ctrl/fan_curve 2>/dev/null | cut -c1-200) | $(grep -A1 OD_VDDGFX_OFFSET $d/pp_od_clk_voltage 2>/dev/null | tr '\n' ' ')"
}
setcap() {   # setcap WATTS TAG: power1_cap on all four cards, read back; fan curve + VDDGFX offset compared before / after
  local w=$1 tag=$2 h i=0 v mn mx b4 af bad="" rb=""
  [ "$DRY" = 1 ] && { echo "+ power1_cap = ${w} W on all four cards ($(echo $HWM | wc -w) hwmons), read back; fan curve / VDDGFX offset before = after"; echo "${w} W" > /dev/null; return 0; }
  v=$((w * 1000000))
  for h in $HWM; do
    mn=$(cat $h/power1_cap_min); mx=$(cat $h/power1_cap_max)
    [ $v -ge $mn ] && [ $v -le $mx ] || { log "RESULT fn4 $tag power: VIRHE ${w} W outside [$((mn / 1000000)), $((mx / 1000000))] W on $h"; return 1; }
  done
  for h in $HWM; do
    b4=$(odsnap $h)
    echo $v > $h/power1_cap 2>> $O/$tag-cap.err || bad="$bad write:$h"
    sleep 1; af=$(odsnap $h)
    [ "$b4" = "$af" ] || bad="$bad od-changed:$h[$b4 -> $af]"
    rb="$rb $(( $(cat $h/power1_cap) / 1000000 ))"
  done
  echo "${w} W (read back:$rb)" > $O/$tag-cap.txt
  if [ -n "$bad" ]; then
    log "RESULT fn4 $tag power: VIRHE setting ${w} W: $bad -> restoring the start caps, no more power arms"
    restorecaps "$tag after a failed set"; POWEROK=0; return 1
  fi
  log "RESULT fn4 $tag power: power1_cap ${w} W on all four cards, read back${rb} W; fan curve and VDDGFX offset unchanged on every card ($(odsnap $(echo $HWM | cut -d' ' -f1) | cut -c1-160))"
}
restorecaps() {   # restorecaps WHY: the caps read at the item's start, back on every card (idempotent; the EXIT/TERM trap)
  local h i=0 rb="" c
  [ "$DRY" = 1 ] && { echo "+ restore power1_cap to the start values ($BASECAPS) on all four cards, read back"; return 0; }
  [ -n "${BASECAPS:-}" ] || return 0
  for h in $HWM; do
    i=$((i + 1)); c=$(echo $BASECAPS | cut -d' ' -f$i)
    [ -n "$c" ] && echo $c > $h/power1_cap 2>/dev/null
    rb="$rb $(( $(cat $h/power1_cap) / 1000000 ))"
  done
  log "RESULT fn4 power restore ($1): power1_cap back to the start values, read back${rb} W"
}
tguard() {   # tguard TAG: every 1 s, junction >= 100 C or mem >= 96 C on any card for 10 s -> restore the caps, flag the arm
  local tag=$1 h j m n=0 hot
  while [ ! -e $O/$tag.tstop ]; do
    hot=""
    for h in $HWM; do
      j=$(( $(cat $h/temp2_input 2>/dev/null || echo 0) / 1000 )); m=$(( $(cat $h/temp3_input 2>/dev/null || echo 0) / 1000 ))
      { [ $j -ge ${X91_TJ_MAX:-100} ] || [ $m -ge ${X91_TMEM_MAX:-96} ]; } && hot="$hot $(basename $(readlink -f $h/../..)):j$j/m$m"
    done
    if [ -n "$hot" ]; then n=$((n + 1)); else n=0; fi
    if [ $n -ge 10 ]; then
      : > $O/$tag.abort
      log "RESULT fn4 $tag: VIRHE temperature guard -- junction >= ${X91_TJ_MAX:-100} C or mem >= ${X91_TMEM_MAX:-96} C for 10 s ($hot) -> caps restored, arm aborted"
      restorecaps "$tag temperature guard"; return 0
    fi
    sleep 1
  done
}
windows() {   # windows TAG: per window of $O/<tag>-win.txt (name from to): rankskew + card-0 stampcat + gmsum
  local tag=$1 w f t
  [ "$DRY" = 1 ] && { echo "+ for each window in $O/$tag-win.txt: rankskew.py $STATSDIR/stamps.csv (+ .c1-.c3), stampcat.py (card 0), gmsum.py $O/$tag-gm.txt; gzip the stamps"; return 0; }
  while read -r w f t; do
    python3 $TL/rankskew.py $STATSDIR/stamps.csv --from $f --to $t --label $tag-$w --json $O/$tag-$w-rank.json > $O/$tag-$w-rank.txt 2>&1
    log "RESULT fn4 $tag $w per rank: $(cat $O/$tag-$w-rank.txt | tr -s ' ' | tr '\n' '|' | cut -c1-2400)"
    python3 $TL/stampcat.py $STATSDIR/stamps.csv $tag-$w --from $f --to $t --json $O/$tag-$w-stamps.json > $O/$tag-$w-stamps.txt 2>&1
    log "RESULT fn4 $tag $w card 0: $(grep -E 'kind records' $O/$tag-$w-stamps.txt | head -2 | tr '\n' '|' | cut -c1-900)"
    log "RESULT fn4 $tag $w $(python3 $TL/gmsum.py $O/$tag-gm.txt x --from $f --to $t --json $O/$tag-$w-gm.json 2>&1 | cut -d' ' -f2- | cut -c1-1500)"
  done < $O/$tag-win.txt
  gzip -f $STATSDIR/stamps.csv* 2>/dev/null || true
}
msn() {   # msn TAG PHASE a|b: the engine's counters (/metrics) and pass counts (/stats) into a snapshot
  run "timeout 30 python3 -I $TL/msnap.py snap $URL $O/$1-$2-ms$3.json > /dev/null 2>&1"
}
msd() {   # msd TAG PHASE: the snapshots' difference -> RESULT line
  [ "$DRY" = 1 ] && return 0
  log "RESULT fn4 $1 engine stats $2: $(timeout 30 python3 -I $TL/msnap.py delta $O/$1-$2-msa.json $O/$1-$2-msb.json 2>&1 | cut -c1-900)"
}
mbw() {   # mbw TAG PHASE: a short mb window (X115_DUR s after X115_WARM s) -> RESULT line
  run "(cd $D && timeout 600 python3 mb.py --tag fn4-$1 --phases $2 --dur ${X115_DUR:-12} --warm ${X115_WARM:-8} > $O/$1-$2.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT fn4 $1 mb $2 (${X115_DUR:-12} s): $(grep '^MB ' $O/$1-$2.txt 2>/dev/null | cut -c4- | cut -c1-400)"
}
gaps() {   # gaps TAG WINDOW T0 T1: gapcat.py on every stamped card, rankskew.py when all four cards stamped -> RESULT lines
  local c f
  if [ "$DRY" = 1 ]; then echo "+ gapcat.py $STATSDIR/stamps.csv(.c1 .c2 .c3) --from $3 --to $4 --json $O/$1-$2-gap<card>.json; rankskew.py (all cards)"; return 0; fi
  for f in $STATSDIR/stamps.csv $STATSDIR/stamps.csv.c1 $STATSDIR/stamps.csv.c2 $STATSDIR/stamps.csv.c3; do
    [ -s $f ] || continue
    c=0; case $f in *.c[0-9]) c=${f##*.c} ;; esac
    timeout 1200 python3 -I $TL/gapcat.py $f --from $3 --to $4 --label $1-$2-card$c --json $O/$1-$2-gap$c.json > $O/$1-$2-gap$c.txt 2>&1
    log "RESULT fn4 115 $1 $2 card $c gaps: $(tr -s ' ' < $O/$1-$2-gap$c.txt | tr '\n' '|' | cut -c1-2400)"
  done
  if [ -s $STATSDIR/stamps.csv.c3 ]; then
    timeout 1800 python3 $TL/rankskew.py $STATSDIR/stamps.csv --from $3 --to $4 --label $1-$2 --json $O/$1-$2-rank.json > $O/$1-$2-rank.txt 2>&1
    log "RESULT fn4 115 $1 $2 per rank (card 0 = c3:00, 1 = 83:00, 2 = 86:00, 3 = 03:00; us a pass): $(tr -s ' ' < $O/$1-$2-rank.txt | tr '\n' '|' | cut -c1-1600)"
  fi
}
arm() {   # arm NAME: restart with every pass stamped, a c16 and a c8 window, the gap analysis, down
  local a=$1 tag=fn4-$1 tgpid="" t0 t1 p
  export TP=4 CARDS=$CARDS4 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=2048 NST=3 WIREKB=128 NOOVL="" PROFEVERY="" PMIN="" PMINB=""
  export AQLSTAMP=1 STATSDIR=$O/stats115-$a
  case $a in
    k4nG) export FNIMAGE=$FN91 AQLCARDS=all EXTRA="--prefix-resume nearest" ;;                         # 1.15.1 (k4n line), all four cards stamped
    rmnG) export FNIMAGE=$FNMAIN AQLCARDS="" EXTRA="--prefix-resume nearest --prefill-interleave 0" ;;  # main 451b136 (rmnI line), card 0 only
    *)    log "RESULT fn4 115: VIRHE unknown arm '$a'"; return 1 ;;
  esac
  run "mkdir -p $STATSDIR; find $STATSDIR -maxdepth 1 -name 'stamps.csv*' -delete; find $O -maxdepth 1 -name '$tag.abort' -delete; find $O -maxdepth 1 -name '$tag.tstop' -delete"
  if [ "$DRY" = 1 ]; then echo "+ tguard $tag &"; else tguard $tag & tgpid=$!; fi
  log "RESULT fn4 $tag: start IMAGE=$FNIMAGE ($(docker image inspect -f '{{.Id}}' $FNIMAGE 2>/dev/null | cut -c8-19)) EXTRA '$EXTRA', RADIANCE_AQL_STAMP=1 (every played pass) cards ${AQLCARDS:-0 only}; power caps: $( [ "$DRY" = 1 ] && echo dry || caps)"
  restart $tag || { keeplog $tag-fail; down; run "touch $O/$tag.tstop"; return 1; }
  [ "$DRY" = 1 ] || log "RESULT fn4 $tag: engine $(grep -m1 -E 'chained launches' $O/start-$tag.log | sed 's/^[A-Z] //' | cut -c1-160) | $(grep -m1 -E 'prefill-interleave' $O/start-$tag.log | sed 's/^[A-Z] //' | tr -s ' ' | cut -c1-120)"
  for p in c16 c8; do
    [ -e $O/$tag.abort ] && break
    msn $tag $p a; t0=$(now); mbw $tag $p; t1=$(now); msn $tag $p b; msd $tag $p
    run "echo '$p $t0 $t1' >> $O/$tag-win.txt"
  done
  run "docker logs $CN > $O/$tag-engine.log 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 $tag stamping: $(grep -m2 -E 'RADIANCE_AQL_STAMP' $O/$tag-engine.log | sed 's/^[A-Z] //' | tr '\n' '|' | cut -c1-300)"
  down
  run "touch $O/$tag.tstop"; [ -n "$tgpid" ] && wait $tgpid 2>/dev/null
  # analysis after down: the stamp files are complete and the cards idle
  if [ "$DRY" = 1 ]; then gaps $tag c16 T0 T1; gaps $tag c8 T0 T1
  else while read -r p t0 t1; do gaps $tag $p $t0 $t1; done < $O/$tag-win.txt; fi
  run "du -sh $STATSDIR > $O/$tag-stampsize.txt 2>&1; gzip -f $STATSDIR/stamps.csv* 2>/dev/null || true"
  log "RESULT fn4 $tag: done"
}
log "RESULT fn4 $ITEM: start ARMS [$ARMS] | k4n $FN91 ($(docker image inspect -f '{{.Id}}' $FN91 2>/dev/null | cut -c8-19)), main $FNMAIN ($(docker image inspect -f '{{.Id}}' $FNMAIN 2>/dev/null | cut -c8-19)) | power stays at the cards' caps (210 W expected), host tuning and the window poller as they are"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  for i in $FN91 $FNMAIN; do docker image inspect $i > /dev/null 2>&1 || { log "RESULT fn4: VIRHE image $i missing"; exit 1; }; done
  [ -e "$Y" ] && grep -q RADIANCE_AQL_STAMP_CARDS $Y || { log "RESULT fn4: VIRHE $Y not found or without the stamp variables"; exit 1; }
  [ -n "$TL" ] && [ -e $TL/gapcat.py ] && [ -e $TL/rankskew.py ] && [ -e $TL/stampcat.py ] && [ -e $TL/msnap.py ] || { log "RESULT fn4: VIRHE tools missing (gapcat.py / rankskew.py / stampcat.py / msnap.py)"; exit 1; }
  [ -f $FNMODELDIR/$FNMODEL ] || { log "RESULT fn4: VIRHE $FNMODELDIR/$FNMODEL missing"; exit 1; }
  for u in ${CARDS4//,/ }; do grep -qi "${u#GPU-}" /sys/class/drm/card*/device/unique_id 2>/dev/null || { log "RESULT fn4: VIRHE card $u not found on this host"; exit 1; }; done
  mkdir -p $O $O/guard
  HWM=$(hwmons); BASECAPS=$(for h in $HWM; do cat $h/power1_cap; done | tr '\n' ' ' | sed 's/ $//')
  trap 'down' EXIT
  trap 'exit 143' TERM INT
else
  echo "+ mkdir -p $O $O/guard; trap: compose down"
  HWM="h0 h1 h2 h3"
fi
for a in $ARMS; do
  [ -e $R/sweep.stop ] && break
  arm $a
done
down
[ "$DRY" = 1 ] || log "RESULT fn4 $ITEM summary, card 0 trunk pass (us a pass: wall / gaps; gap sources): $(python3 - $O "$ARMS" <<'PY' 2>&1 | cut -c1-2400
import json, sys
O, arms = sys.argv[1], sys.argv[2].split()
out = []
for p in ("c16", "c8"):
    for a in arms:
        try:
            g = json.load(open(f"{O}/fn4-{a}-{p}-gap0.json"))
        except (OSError, ValueError):
            out.append(f"{p} {a}: no analysis"); continue
        k = max(g["kinds"], key=lambda x: int(x))
        d = g["kinds"][k]; t = g.get("trunk", {}).get("cats_us", {})
        out.append(f"{p} {a}: wall {d['wall_us']:.0f} gaps {d['gaps_us']:.0f} ({d['dispatches']:.0f} dispatches, {d['ordered']:.0f} barrier, {d['chained']:.0f} chained) | " +
                   " ".join(f"{c} {v:.0f}" for c, v in sorted(t.items(), key=lambda kv: -kv[1])) +
                   " | between passes: " + " ".join(f"{x} {v:.0f}" for x, v in sorted(g.get("between_us", {}).items())))
print(" ; ".join(out))
PY
)"
log "RESULT fn4 $ITEM: done"
