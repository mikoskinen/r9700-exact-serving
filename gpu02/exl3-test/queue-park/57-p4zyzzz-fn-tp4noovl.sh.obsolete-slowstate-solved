#!/bin/bash
# 94-fn-tp4noovl.sh (queue item, 8.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~35 min for the two
# arms) -- Flash-Next TP4: is the 8u expert slow state the queue's concurrent dispatches?
# WHY: in 87 s4's slow layers the shared-expert GEMMs that the overlap analysis lets start beside the routed experts
# (gemm_fp8a8 dispatched without the barrier bit) stretch from 22.5 -> 94 us and 4.5 -> 104 us and end together with the
# routed kernels, which stretch +45-70 us (gate_up) and +105 us (down). RADIANCE_AQL_NO_OVERLAP=1 (core patch, branch
# fn4-stamp-cards fbf436a, core/runtime/ctx.cpp: Ctx::haz_mark marks nothing) keeps every played dispatch's barrier bit:
# the same kernels, arguments and order, no two dispatches of a queue at once. Exact: byte-identical output to stock.
# Cost to expect: ~4.3 us a barrier for the ~370 dispatches a pass that run unordered today (~1.6 ms at 8u) plus the lost
# overlap (~27 us a MoE layer in fast layers, ~1.3 ms); the slow state costs ~3.2 ms (10.7 %) -- so this is a
# diagnostic first, a fix only if the slow share collapses and the net is positive.
#   o4a   91's d4a (TP4 all_vram MML 262144, per-rank stamps) with RADIANCE_AQL_NO_OVERLAP=1
#   o4w   + --tp-wire-min-kb 1024 (92's w4a lever)
# Per arm as 91's d4a (gate, 1 stream, mb c4 / c8 / c16, per-rank stamps, gmsample, rankskew / stampcat / gmsum); the image is
# stock 1.15.1 + the stamp + no-overlap binary (X94_IMAGE); fn-tp4o.yaml passes the knob. REFERENCE: 91's d4a / 92's w4a.
# DRYRUN=1 bash 94-fn-tp4noovl.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4o.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
FN91=${X94_IMAGE:-radiance-local:1.15.1-stampcards-fbf436a}; export FNIMAGE=$FN91
export FNMODELDIR=$HOME/models/flashnext-iq4r FNMODEL=qwen3.8-next-flash-fp8-iq4r-moe.rad
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
CARDS4=$C3,$C83,$C86,$C03; CARDS2=$C3,$C03
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/rankskew.py" ] && [ -e "$f/pwsum.py" ] && { TL=$f; break; }; done
ARMS=${ARMS:-"o4a o4w"}; ITEM=94-fn-tp4noovl
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
arm() {   # arm NAME
  local a=$1 tag=fn4-$1 rc t0 t1 gpid="" tgpid="" b=${1%%-p*} pw=""
  case $a in *-p[0-9]*) pw=${a##*-p} ;; esac
  export FNIMAGE=$FN91 TP=4 CARDS=$CARDS4 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=2048 EXTRA="" NST=3 WIREKB=128 NOOVL=""
  export AQLSTAMP=${X91_STAMP_EVERY:-20} AQLCARDS=$CARDSMODE STATSDIR=$O/stats-$a
  local PH="c4 c8 c16"
  case $b in
    d4a) ;;
    d4t) export PLACE=expert_tiered EXTRA="--host-pool-mib 16384" ;;
    d2t) export TP=2 CARDS=$CARDS2 PLACE=expert_tiered EXTRA="--host-pool-mib 16384" MML=524288 ROPE="--rope-scaling yarn:2"; PH="c8" ;;
    w4a) export WIREKB=${X92_WIREKB:-1024} ;;                     # 92: d4a with the exact fused one-shot up to 1 MiB
    p4a) export FNIMAGE=$FNP4 ;;                                  # 92: d4a on PR #396 (chained forms, hc_read G/S/B3)
    p4w) export FNIMAGE=$FNP4 WIREKB=${X92_WIREKB:-1024} ;;       # 92: both
    o4a) export NOOVL=1 ;;                                        # 94: d4a with every played dispatch ordered
    o4w) export NOOVL=1 WIREKB=${X92_WIREKB:-1024} ;;             # 94: + the exact wire floor
    *)   log "RESULT fn4: VIRHE unknown arm '$a'"; return 1 ;;
  esac
  run "rm -rf $STATSDIR; mkdir -p $STATSDIR; : > $O/$tag-win.txt; rm -f $O/$tag.abort $O/$tag.tstop"
  if [ -n "$pw" ]; then
    [ "$POWEROK" = 1 ] || { log "RESULT fn4 $tag: skipped -- an earlier power set failed (fan curve / VDDGFX / write)"; return 1; }
    setcap $pw $tag || return 1
  else
    run "echo 'start caps ($BASECAPS µW)' > $O/$tag-cap.txt"
  fi
  if [ "$DRY" = 1 ]; then echo "+ tguard $tag & (junction >= ${X91_TJ_MAX:-100} C / mem >= ${X91_TMEM_MAX:-96} C for 10 s -> restore caps, abort)"
  else tguard $tag & tgpid=$!; fi
  log "RESULT fn4 $tag: start IMAGE=$FNIMAGE TP=$TP cards $CARDS PLACE=$PLACE MML=$MML ${ROPE:-} MAXSEQS=$MAXSEQS MNBT=$MNBT NST=$NST ${EXTRA:-} wire-min-kb $WIREKB${NOOVL:+ NO_OVERLAP=$NOOVL} AQL_STAMP=$AQLSTAMP cards=$AQLCARDS | power:$( [ "$DRY" = 1 ] && echo ' (dry)' || caps)"
  gate $tag; rc=$?
  if [ $rc = 1 ] && [ "$AQLCARDS" = all ]; then   # the per-card stamp patch is new: one retry stamping card 0 only (stock behaviour)
    keeplog $tag-failcards
    log "RESULT fn4 $tag: VIRHE not healthy with RADIANCE_AQL_STAMP_CARDS=all ($(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -2 | tr '\n' ' ' | cut -c1-300); $O/engine-$tag-failcards.log) -> card 0 only from here on"
    CARDSMODE=""; export AQLCARDS=""; run "rm -rf $STATSDIR; mkdir -p $STATSDIR"; gate $tag; rc=$?
  fi
  [ $rc = 1 ] && { keeplog $tag-fail; log "RESULT fn4 $tag: VIRHE not healthy -- $O/start-$tag.log, $O/engine-$tag-fail.log: $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; armend $tag; return 1; }
  [ $rc = 2 ] && { log "RESULT fn4 $tag: VIRHE slow state after one restart -> arm skipped (engine logs $O/engine-$tag-try1.log / -try2.log)"; down; armend $tag; return 1; }
  [ "$DRY" = 1 ] || [ -z "$NOOVL" ] || log "RESULT fn4 $tag no-overlap: $(grep -c 'RADIANCE_AQL_NO_OVERLAP: every played dispatch keeps its barrier bit' $O/start-$tag.log) engine line(s) confirm it"
  [ "$DRY" = 1 ] || log "RESULT fn4 $tag start: $(grep -m1 -E '^ +hardware queues' $O/start-$tag.log | tr -s ' ' | cut -c1-120) | $(grep -m1 -E '^ +expert prefetch' $O/start-$tag.log | tr -s ' ' | cut -c1-100) | $(grep -m1 -E '^ +experts +[0-9.]+ GiB' $O/start-$tag.log | tr -s ' ' | cut -c1-140) | $(grep -m1 -E '^ +kv cache +[0-9.]+ GiB' $O/start-$tag.log | tr -s ' ' | cut -c1-140) | $(grep -c 'RADIANCE_AQL_STAMP times' $O/start-$tag.log) stamp line(s): $(grep -oE 'on card [0-9]+, into [^ ]+' $O/start-$tag.log | sort -u | tr '\n' ';')"
  # the sampler: from the 1-stream phase to the end of the last mb phase (bounded: 1500 s at most)
  if [ "$DRY" = 1 ]; then echo "+ timeout -k 5 1500 python3 -I $TL/gmsample.py $O/$tag-gm.txt 1490 $CARDS &"
  else timeout -k 5 1500 python3 -I $TL/gmsample.py $O/$tag-gm.txt 1490 $CARDS > /dev/null 2>&1 & gpid=$!; fi
  t0=$(now); pupair $tag; probe $tag; t1=$(now); run "echo 1u $t0 $t1 >> $O/$tag-win.txt"
  for p in $PH; do
    [ -e $R/sweep.stop ] && break
    [ -e $O/$tag.abort ] && { log "RESULT fn4 $tag: phases from $p on skipped (temperature guard)"; break; }
    t0=$(now); mbp $tag $p; t1=$(now); run "echo $p $t0 $t1 >> $O/$tag-win.txt"
  done
  [ -n "$gpid" ] && { kill $gpid 2>/dev/null; wait $gpid 2>/dev/null; }
  if [ "$DRY" != 1 ] && [ "$(docker inspect -f '{{.State.Status}}' $CN 2>/dev/null)" != running ]; then
    keeplog $tag-died
    log "RESULT fn4 $tag: VIRHE the engine stopped during the run (stamp cards '${AQLCARDS:-0}'): $(grep -E '^E |fault|abort|signal' $O/engine-$tag-died.log 2>/dev/null | tail -3 | tr '\n' ' ' | cut -c1-400)"
    [ "$AQLCARDS" = all ] && { CARDSMODE=""; log "RESULT fn4: later arms stamp card 0 only"; }
  fi
  run "docker logs $CN > $O/$tag-engine.log 2>&1"
  down   # exit writes the stamps still pending
  [ "$DRY" = 1 ] || log "RESULT fn4 $tag stamp files: $(ls -la $STATSDIR 2>&1 | awk 'NR>1 {print $5, $9}' | tr '\n' ';' | cut -c1-300)"
  armend $tag
  windows $tag
  log "RESULT fn4 $tag: done${pw:+ at ${pw} W}"
}
armend() {   # armend TAG: stop the temperature guard; a power arm's caps go back to the start values
  run "touch $O/$1.tstop"; [ -n "${tgpid:-}" ] && { wait $tgpid 2>/dev/null; tgpid=""; }
  [ -n "${pw:-}" ] && restorecaps "$1 end"
  return 0
}
log "RESULT fn4 $ITEM: start ARMS [$ARMS] IMAGE=$FNIMAGE ($(docker image inspect -f '{{.Id}}' $FNIMAGE 2>/dev/null | cut -c8-19)) model $FNMODELDIR/$FNMODEL compose $Y tools $TL"
if [ "$DRY" != 1 ]; then
  [ -e "$Y" ] || { log "RESULT fn4: VIRHE $(basename ${YN:-fn-tp4d.yaml}) not found"; exit 1; }
  grep -q 'RADIANCE_AQL_NO_OVERLAP' $Y || { log "RESULT fn4: VIRHE $Y has no RADIANCE_AQL_NO_OVERLAP (94 needs fn-tp4o.yaml)"; exit 1; }
  case " $ARMS " in *" w4"*|*" p4w"*|*" o4w"*) grep -q 'WIREKB' $Y || { log "RESULT fn4: VIRHE $Y has no \${WIREKB} (the wire-floor arms need fn-tp4w.yaml)"; exit 1; } ;; esac
  [ -n "$TL" ] && [ -e $TL/gmsample.py ] && [ -e $TL/gmsum.py ] && [ -e $TL/stampcat.py ] || { log "RESULT fn4: VIRHE tools missing (rankskew.py / gmsample.py / gmsum.py / stampcat.py beside the item)"; exit 1; }
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  docker image inspect $FNIMAGE > /dev/null 2>&1 || { log "RESULT fn4: VIRHE image $FNIMAGE missing"; exit 1; }
  case " $ARMS " in *" p4"*) docker image inspect $FNP4 > /dev/null 2>&1 || { log "RESULT fn4: VIRHE image $FNP4 missing (the PR #396 arms)"; exit 1; } ;; esac
  [ -f $FNMODELDIR/$FNMODEL ] || { log "RESULT fn4: VIRHE $FNMODELDIR/$FNMODEL missing"; exit 1; }
  [ -e $Q/prodfc.py ] || { log "RESULT fn4: VIRHE $Q/prodfc.py missing"; exit 1; }
  for u in ${CARDS4//,/ }; do grep -qi "${u#GPU-}" /sys/class/drm/card*/device/unique_id 2>/dev/null || { log "RESULT fn4: VIRHE card $u not found on this host"; exit 1; }; done
  mkdir -p $O $O/guard
  HWM=$(hwmons); [ $(echo $HWM | wc -w) = 4 ] || { log "RESULT fn4: VIRHE hwmon of the four cards not found ($HWM)"; exit 1; }
  BASECAPS=$(for h in $HWM; do cat $h/power1_cap; done | tr '\n' ' ' | sed 's/ $//')
  for h in $HWM; do [ -w $h/power1_cap ] || { case " $ARMS " in *-p[0-9]*) log "RESULT fn4: VIRHE $h/power1_cap not writable (render group?) -- power arms would fail"; exit 1 ;; esac; }; done
  trap 'restorecaps "EXIT/TERM trap"; down' EXIT
  trap 'exit 143' TERM INT
  log "RESULT fn4 power at start: power1_cap $(for c in $BASECAPS; do echo -n "$((c / 1000000)) "; done)W (c3 83 86 03); $(for h in $HWM; do echo -n "$(basename $(readlink -f $h/../..)) [$(odsnap $h | cut -c1-120)] "; done)"
  case " $(for c in $BASECAPS; do echo -n "$((c / 1000000)) "; done)" in *" 210 210 210 210 "*) ;; *) log "RESULT fn4 power at start: note -- not 210 W on every card; the restore writes these start values back" ;; esac
else
  echo "+ mkdir -p $O $O/guard"
  HWM="h0 h1 h2 h3"; BASECAPS="210000000 210000000 210000000 210000000"
  echo "+ trap: restore power1_cap to the start values + compose down on EXIT/TERM/INT"
fi
done_arms=""; CARDSMODE=${X91_CARDS:-all}; POWEROK=1
for a in $ARMS; do
  [ -e $R/sweep.stop ] && break
  arm $a && done_arms="$done_arms $a"
done
down
restorecaps "item end"
[ "$DRY" = 1 ] || log "RESULT fn4 power table: $(python3 $TL/pwsum.py $O $(for a in $done_arms; do echo fn4-$a; done) 2>&1 | tr '\n' '#' | sed 's/#/ ## /g' | cut -c1-3000)"
log "RESULT fn4 $ITEM: done --$done_arms | refs (87, stock 1.15.1, card 0): s4 TP4 all_vram 8u verify 30.15 ms, slow layers 32.6 %, c8 619.8 tok/s 30.8 ms; s3 TP3 34.92 ms 61.8 %; s2 TP2 c3+03 33.68 ms 0.9 %; t4 c4 470.9 / c8 604.8 / c16 785.9 tok/s"
