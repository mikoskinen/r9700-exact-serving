#!/bin/bash
# 103-fn-tp4awake.sh (queue item, 9.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~40 min for 10
# starts) -- follow-up of 101 IF gpu_metrics polling (101's B / D) keeps TP4 out of the expert slow state: how often must the
# host poll? 101's machinery (fresh start of the k4s line, 16k warm prefill, a 60 s mb c8 probe: fast <= 26.8 ms, slow >=
# 28.5) with the sampler at 5 / 20 / 100 / 500 ms (tools/gmsample.py's 4th argument), interleaved with no-sampler starts:
# PLAN (X103_PLAN) default "A B5 B500 A B20 B100 A B500 B5 A". The cheapest interval that stays fast is the production
# keep-awake candidate (a read-only sysfs poller beside the engine). Temperature guard on; nothing written to sysfs.
# DRYRUN=1 bash 103-fn-tp4awake.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4o.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
FN91=${X101_IMAGE:-radiance-local:1.15.1-stampcards-c0cbe5a}; export FNIMAGE=$FN91
export FNMODELDIR=$HOME/models/flashnext-iq4r FNMODEL=qwen3.8-next-flash-fp8-iq4r-moe.rad
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
CARDS4=$C3,$C83,$C86,$C03; CARDS2=$C3,$C03
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/slowprobe.py" ] && [ -e "$f/gmsample.py" ] && { TL=$f; break; }; done
PLAN=${X103_PLAN:-"A B5 B500 A B20 B100 A B500 B5 A"}; ARMS=$PLAN; ITEM=103-fn-tp4awake
FNP4=${X92_PR396_IMAGE:-radiance-local:1.15.1-pr396-0ae1545-stampcards}
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
probe8() {   # probe8 TAG: a 60 s mb c8 -> RESULT line; prints "ms tok/s"
  run "(cd $D && timeout 400 python3 mb.py --tag fn4-$1 --phases c8 --dur ${X101_DUR:-60} --warm 10 > $O/$1-c8.txt 2>&1)"
  [ "$DRY" = 1 ] && { echo "25.6 744"; return 0; }
  grep '^MB ' $O/$1-c8.txt 2>/dev/null | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[3:]); print(d.get("ms_decode_step"), d.get("agg_dec_tps"))' 2>/dev/null || echo "0 0"
}
setknob() {   # setknob FILE VALUE: write one sysfs knob on all four cards, read it back (fan curve + VDDGFX checked like setcap)
  local f=$1 v=$2 h d b4 af bad="" rb=""
  [ "$DRY" = 1 ] && { echo "+ $f = $v on all four cards, read back; fan curve / VDDGFX before = after"; return 0; }
  for h in $HWM; do
    d=$(readlink -f $h/../..)
    b4=$(odsnap $h)
    echo $v > $d/$f 2>> $O/knob.err || bad="$bad write:$d/$f"
    sleep 1; af=$(odsnap $h)
    [ "$b4" = "$af" ] || bad="$bad od-changed:$(basename $d)"
    rb="$rb $(head -c 60 $d/$f | grep -oE '[a-z_]+|[0-9]+ [A-Z_0-9]+ *\*' | head -1 | tr -d ' ')"
  done
  [ -z "$bad" ] && { log "RESULT fn4 103 knob: $f = $v on all four cards (read back:$rb); fan curve and VDDGFX unchanged"; return 0; }
  log "RESULT fn4 103 knob: VIRHE setting $f = $v:$bad -> knobs back to the start values, no more knob starts"; KNOBOK=0; restoreknobs "failed set"; return 1
}
restoreknobs() {   # restoreknobs WHY: pp_power_profile_mode and power_dpm_force_performance_level back to their start values
  [ "$DRY" = 1 ] && { echo "+ restore pp_power_profile_mode / power_dpm_force_performance_level to the start values"; return 0; }
  local h d i=0
  for h in $HWM; do
    i=$((i + 1)); d=$(readlink -f $h/../..)
    echo $(echo $BASEPROF | cut -d' ' -f$i) > $d/pp_power_profile_mode 2>/dev/null
    echo $(echo $BASELVL | cut -d' ' -f$i) > $d/power_dpm_force_performance_level 2>/dev/null
  done
  log "RESULT fn4 103 knob restore ($1): profile $(for h in $HWM; do grep -oE '^ *[0-9]+ +[A-Z_0-9]+ *\*' $(readlink -f $h/../..)/pp_power_profile_mode | awk '{print $1}' | tr '\n' ' '; done)| level $(for h in $HWM; do cat $(readlink -f $h/../..)/power_dpm_force_performance_level | tr '\n' ' '; done)"
}
vramused() { local h; for h in $HWM; do echo -n "$(( $(cat $(readlink -f $h/../..)/mem_info_vram_used) / 1048576 )) "; done; }
start1() {   # start1 N COND: one fresh start of the k4s line under condition COND, its c8 probe, down
  local n=$1 c=$2 tag=fn4-103s$1$2 gp="" gi=5 rc ms tps st=unknown t0 t1 sp=""
  export FNIMAGE=$FN91 TP=4 CARDS=$CARDS4 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=2048 EXTRA="" NST=3 WIREKB=128 NOOVL="" PROFEVERY=""
  export AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats103-$n$c
  case $c in
    A) ;;                                              # base: no stamps, no sampler
    B) gp=1 ;;                                         # gmsample only (5 ms gpu_metrics + 100 ms fclk/mclk on all cards)
    B[0-9]*) gp=1; gi=${c#B} ;;                        # 103: gmsample at gi ms (B20, B100, B500 ...)
    C) export AQLSTAMP=20 AQLCARDS=all ;;              # stamps on every card only
    D) export AQLSTAMP=20 AQLCARDS=all; gp=1 ;;        # both: 91 / 92's configuration
    E) export AQLSTAMP=20 ;;                           # stamps on card 0 only (87 s4 / 93's configuration)
    P) [ "$KNOBOK" = 1 ] || { log "RESULT fn4 $tag: skipped (knob failure earlier)"; return 1; }; setknob pp_power_profile_mode 5 || return 1 ;;   # COMPUTE profile
    H) [ "$KNOBOK" = 1 ] || { log "RESULT fn4 $tag: skipped (knob failure earlier)"; return 1; }; setknob power_dpm_force_performance_level high || return 1 ;;
    *) log "RESULT fn4: VIRHE unknown condition '$c'"; return 1 ;;
  esac
  run "rm -rf $STATSDIR; mkdir -p $STATSDIR; rm -f $O/$tag.abort $O/$tag.tstop"
  local gap=""; [ -n "$LASTDOWN" ] && gap=$(( $(date +%s) - LASTDOWN ))
  log "RESULT fn4 103 start $n ($c): gap since the last down ${gap:-n/a} s, VRAM used before up $( [ "$DRY" = 1 ] && echo dry || vramused)MiB, temps:$( [ "$DRY" = 1 ] && echo ' (dry)' || caps)"
  if [ "$DRY" = 1 ]; then echo "+ tguard $tag &"; else tguard $tag & TGPID=$!; fi
  restart $tag; rc=$?
  if [ $rc != 0 ]; then keeplog $tag-fail; log "RESULT fn4 $tag: VIRHE not healthy"; down; LASTDOWN=$(date +%s); run "touch $O/$tag.tstop"; wait $TGPID 2>/dev/null; case $c in P|H) restoreknobs "start $n end" ;; esac; return 1; fi
  if [ -n "$gp" ]; then
    if [ "$DRY" = 1 ]; then echo "+ timeout -k 5 600 python3 -I $TL/gmsample.py $O/$tag-gm.txt 590 $CARDS $gi &"
    else timeout -k 5 600 python3 -I $TL/gmsample.py $O/$tag-gm.txt 590 $CARDS $gi > /dev/null 2>&1 & sp=$!; fi
  fi
  run "(cd $D && timeout 600 python3 ds41_tests.py prefill --depths 16384 --cpt 3.6 --calibrate 0 --reps 1 > $O/$tag-pf16.txt 2>&1)"
  t0=$(now); set -- $(probe8 $tag); ms=$1; tps=$2; t1=$(now)
  [ -n "$sp" ] && { kill $sp 2>/dev/null; wait $sp 2>/dev/null; }
  if [ "$DRY" != 1 ]; then
    st=$(awk -v m=$ms -v f=${X101_FAST_MS:-26.8} -v s=${X101_SLOW_MS:-28.5} 'BEGIN{print (m <= 0) ? "unknown" : (m <= f) ? "fast" : (m >= s) ? "slow" : "between"}')
  fi
  run "docker logs $CN > $O/$tag-engine.log 2>&1"
  down; LASTDOWN=$(date +%s)
  local sp8=""
  case $c in C|D|E) [ "$DRY" = 1 ] || sp8=$(python3 $TL/slowprobe.py $STATSDIR/stamps.csv --from $t0 --to $t1 --mb $O/$tag-c8.txt 2>&1 | head -1) ;; esac
  run "touch $O/$tag.tstop"; [ -n "${TGPID:-}" ] && { wait $TGPID 2>/dev/null; TGPID=""; }
  case $c in P|H) restoreknobs "start $n end" ;; esac
  [ "$DRY" = 1 ] || log "RESULT fn4 103 start $n ($c): ${st} -- mb c8 ${ms} ms ${tps} tok/s${sp8:+ | card-0 stamps: $sp8}$( [ -n "$gp" ] && echo " | $(python3 $TL/gmsum.py $O/$tag-gm.txt x --from $t0 --to $t1 2>&1 | cut -d' ' -f2- | cut -c1-400)")"
  run "echo '$n $c $st $ms $tps' >> $O/103-table.txt"
  gzip -f $STATSDIR/stamps.csv* 2>/dev/null || true
}
log "RESULT fn4 $ITEM: start PLAN [$PLAN] IMAGE=$FNIMAGE ($(docker image inspect -f '{{.Id}}' $FNIMAGE 2>/dev/null | cut -c8-19)) compose $Y | slow-state hunt: fresh starts of the TP4 all_vram line, one condition each, a 60 s mb c8 probe each (fast <= ${X101_FAST_MS:-26.8} ms, slow >= ${X101_SLOW_MS:-28.5} ms)"
if [ "$DRY" != 1 ]; then
  [ -e "$Y" ] || { log "RESULT fn4: VIRHE $YN not found"; exit 1; }
  [ -n "$TL" ] && [ -e $TL/slowprobe.py ] && [ -e $TL/gmsample.py ] && [ -e $TL/gmsum.py ] || { log "RESULT fn4: VIRHE tools missing (slowprobe.py / gmsample.py / gmsum.py)"; exit 1; }
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  docker image inspect $FNIMAGE > /dev/null 2>&1 || { log "RESULT fn4: VIRHE image $FNIMAGE missing"; exit 1; }
  [ -f $FNMODELDIR/$FNMODEL ] || { log "RESULT fn4: VIRHE $FNMODELDIR/$FNMODEL missing"; exit 1; }
  for u in ${CARDS4//,/ }; do grep -qi "${u#GPU-}" /sys/class/drm/card*/device/unique_id 2>/dev/null || { log "RESULT fn4: VIRHE card $u not found on this host"; exit 1; }; done
  mkdir -p $O $O/guard; : > $O/103-table.txt
  HWM=$(hwmons); [ $(echo $HWM | wc -w) = 4 ] || { log "RESULT fn4: VIRHE hwmon of the four cards not found ($HWM)"; exit 1; }
  BASECAPS=$(for h in $HWM; do cat $h/power1_cap; done | tr '\n' ' ' | sed 's/ $//')
  BASEPROF=$(for h in $HWM; do grep -oE '^ *[0-9]+ +[A-Z_0-9]+ *\*' $(readlink -f $h/../..)/pp_power_profile_mode | awk '{print $1}'; done | tr '\n' ' ')
  BASELVL=$(for h in $HWM; do cat $(readlink -f $h/../..)/power_dpm_force_performance_level; done | tr '\n' ' ')
  case " $PLAN " in *P*|*H*) for h in $HWM; do d=$(readlink -f $h/../..); [ -w $d/pp_power_profile_mode ] && [ -w $d/power_dpm_force_performance_level ] || { log "RESULT fn4: VIRHE $d power profile / DPM level not writable"; exit 1; }; done ;; esac
  log "RESULT fn4 103 at start: power profile [$BASEPROF] DPM level [$BASELVL] caps $(for c in $BASECAPS; do echo -n "$((c / 1000000)) "; done)W, VRAM used $(vramused)MiB"
  trap 'restoreknobs "EXIT/TERM trap"; restorecaps "EXIT/TERM trap"; down' EXIT
  trap 'exit 143' TERM INT
else
  echo "+ mkdir -p $O $O/guard; read the start profile / DPM level / caps; trap: restore them + compose down"
  HWM="h0 h1 h2 h3"; BASECAPS="210000000 210000000 210000000 210000000"; BASEPROF="0 0 0 0"; BASELVL="auto auto auto auto"
fi
KNOBOK=1; LASTDOWN=""; TGPID=""; n=0
for c in $PLAN; do
  [ -e $R/sweep.stop ] && break
  n=$((n + 1)); start1 $n $c
done
down
[ "$DRY" = 1 ] || log "RESULT fn4 $ITEM table (start condition state ms tok/s): $(tr '\n' ';' < $O/103-table.txt) | conditions: A none, B gpu_metrics sampling at 5 ms, B<n> at n ms, C stamps on every card, D both, E card-0 stamps, P COMPUTE profile, H DPM high"
log "RESULT fn4 $ITEM: done"
