#!/bin/bash
# 143-fn-dp2stack.sh (queue item, 10.11.2026, queue-ar; gpu02, cards 83:00 + 86:00 used, every radiance container DOWN; ~25 min)
# -- coordinator 10.11.: the production candidate at production's shape is our main-line stack, not the 1.17.0-based DS
# engine of 110b's t3 / t3hs. One arm on 110b's harness and pair (TP2 83+86, the production dp-a serve line on the test port,
# expert_tiered hp 16 GiB, fp8 KV, wht6 min-kb 128, MML 524288 yarn:2, MAXSEQS 16, MTP 3; compose fn-tp2s.yaml = fn-tp2.yaml
# + MNBT and the fn4 switches):
#   dstk  radiance-local:c16-sw7 (fn4-c16 803f418) + the stack switches that act at TP2 -- F1n, hc_readv 3, hc_upmix 5 (hc_read
#         is replicated on every rank, so they act as at TP4), F2b (route reports off, rank-independent); hcw set too but
#         idle: at two ranks every write is the fused ar_hc_write (halo), so libr4d's hc_write never runs; F2a OFF: at two
#         ranks the MoE gather is folded into ar_gather_hc_write, which reads the sorted slots, not F2a's inverse table, so
#         F2a would only add the sort's inverse writes -- MNBT 4096 + --prefill-span 2 + --prefill-interleave 0 +
#         --prefix-resume nearest (at two ranks the span runs serially: the fused write takes every step's all-reduce).
# Phases (110b's arm): gate -> produp pair + probe -> mb c4 / c8 / c16 (+ engine counters) -> prefill 16k + fresh 64k /
# 128k -> the 33k copy -> mb pf -> engine lines -> down; BetterBench and cc_sim off (X143_BB / X110_CC). END: dstk vs 110b's
# s15 of today (its files in runs/fn4: mb c4 / c8 / c16 347.9 / 455.3 / 573.7, prefill 4826 / 4799 / 4815, mb pf 3542)
# and the identity lines against s15 (information: another engine).
# DRYRUN=1 bash 143-fn-dp2stack.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp2s.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest; URL=http://127.0.0.1:8090; export DS41_URL=$URL
FN91=${X91_IMAGE:-radiance-local:1.15.1-stampcards-c0cbe5a}; export FNIMAGE=$FN91
export FNMODELDIR=$HOME/models/flashnext-iq4r FNMODEL=qwen3.8-next-flash-fp8-iq4r-moe.rad
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
CARDS4=$C3,$C83,$C86,$C03; CARDS2=$C3,$C03
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/rankskew.py" ] && [ -e "$f/pwsum.py" ] && [ -e "$f/slowprobe.py" ] && [ -e "$f/msnap.py" ] && { TL=$f; break; }; done
ARMS=${X143_ARMS:-"dstk"}; ITEM=143-fn-dp2stack   # empty: k4s + the X93_WINNER pick (default auto: tools/pick93.py over the run log)
BBROOT=${BB_ROOT:-$HOME/gpu/ext/betterbench}; BBDEPS=${BB_DEPS:-$HOME/gpu/ext/bb-deps}
BBRUN=; for f in "$H/tools/bb_run.py" "$H/../tools/bb_run.py" "$HOME/gpu/ds41/exl3-test/queue-ar/tools/bb_run.py"; do [ -e "$f" ] && { BBRUN=$f; break; }; done
CCSIM=${X93_CCSIM:-$HOME/gpu/ds41/exl3-test/tools/ccsim/cc_sim.py}; CCWRAP=${X93_CCWRAP:-$HOME/gpu/ds41/exl3-test/ccsim-exl3-turns.sh}
CCU=${X93_CCU:-16}; CCTO=${X110_CCTIMEOUT:-4800}   # 110: one TP2 server, so the fixed work takes longer than at TP4
CCARGS=${X110_CCARGS:-"--duration 4200 --turns 15 --warmup-turns 3 --effort xhigh,medium --gen-mode forced --seed 1 --cpt-fixed --temperature 0 --trace-seeded"}
FNP4=${X92_PR396_IMAGE:-radiance-local:1.15.1-pr396-0ae1545-stampcards}
FNPMIN=${X93_PMIN_IMAGE:-radiance-local:1.15.1-pmin-47e859a-stampcards}
FNSTK=${X143_IMAGE:-radiance-local:c16-sw7}; FNS15=${X110_IMAGE_S15:-stilldeadcode/radiance:1.15.1}; FNT3=${X110_IMAGE_T3:-radiance-local:ds41-exl3-117-390664b3-fa0e9580-tuned3-hd4a11cab}
FN17=${X93_IMAGE_17:-radiance-local:1.17.0-src}; FNMAIN=${X93_IMAGE_MAIN:-radiance-local:main-451b136}
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
bbph() {   # bbph TAG NOTE...: BetterBench --quick + its prefill sweep + concurrency 1/2/4/8/16, thinking off (tools/bb_run.py)
  local tag=$1 nt="" n; shift
  for n in "$@"; do nt="$nt --note $n"; done
  run "(BETTERBENCH_HOME=$O/bbhome timeout 1800 python3 -I $BBRUN --bb-root $BBROOT --deps $BBDEPS -- run --endpoint $URL/v1 --model qwen3.8-flash-next --quick --config $O/bb-config.json --name $tag --out $O/bb-$tag/results.json$nt > $O/$tag-bb.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  log "RESULT fn4 $tag BetterBench: $(python3 -I $BBRUN --summary $O/bb-$tag/results.json 2>&1 | cut -c1-1400)"
  python3 -I $BBRUN --check $O/bb-$tag/results.json > /dev/null 2>&1 || log "RESULT fn4 $tag BetterBench: VIRHE single-stream answers carried reasoning text -- thinking was not off ($O/bb-$tag/results.json)"
}
pfrun() {   # pfrun TAG NAME DEPTHS: fresh prefills (ds41_tests prefill, nonce first) -> RESULT line
  run "(cd $D && timeout 1500 python3 ds41_tests.py prefill --depths $3 --cpt 3.6 --calibrate 0 --reps 1 > $O/$1-$2.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT fn4 $1 prefill $2: $(python3 -c 'import json,sys; print("; ".join("%s tok %s tok/s TTFT %s s" % (d.get("prompt_tokens"), d.get("prefill_tps"), d.get("ttft")) for d in (json.loads(l[2:]) for l in open(sys.argv[1]) if l.startswith("J "))))' $O/$1-$2.txt 2>&1 | cut -c1-400)"
}
ccrun() {   # ccrun TAG: one fixed-work, trace-seeded cc_sim run at $CCU users (the wrapper logs its own RESULT cc_sim line)
  run "CCARGS='$CCARGS' CCSIM='$CCSIM' timeout $CCTO $CCWRAP $CCU $1 > $O/$1-ccsim-u$CCU.out 2>&1"
}
ccpair93() {   # ccpair93 TAG_A TAG_B: cc_sim.py --pair over the two arms' newest runs
  [ "$DRY" = 1 ] && { echo "+ cc_sim.py --pair <$1 u$CCU jsonl> <$2 u$CCU jsonl>"; return 0; }
  local A B
  A=$(ls -t $R/ccsim-*-$1-u$CCU.jsonl 2>/dev/null | head -1); B=$(ls -t $R/ccsim-*-$2-u$CCU.jsonl 2>/dev/null | head -1)
  [ -n "$A" ] && [ -n "$B" ] || { log "RESULT fn4 $2 vs $1 cc_sim: not paired (missing run: '$A' '$B')"; return 0; }
  log "RESULT fn4 $2 vs $1 cc_sim u$CCU paired: $(timeout 120 python3 $CCSIM --pair $A $B 2>&1 | tr '\n' ';' | cut -c1-1200)"
}
msn() {   # msn TAG PHASE a|b: the engine's speculation counters (/metrics) and pass counts (/stats) into a snapshot
  run "timeout 30 python3 -I $TL/msnap.py snap $URL $O/$1-$2-ms$3.json > /dev/null 2>&1"
}
msd() {   # msd TAG PHASE: the snapshots' difference -> RESULT line (drafts verified / accepted a step, verify rows a step)
  [ "$DRY" = 1 ] && return 0
  log "RESULT fn4 $1 engine stats $2: $(timeout 30 python3 -I $TL/msnap.py delta $O/$1-$2-msa.json $O/$1-$2-msb.json 2>&1 | cut -c1-900)"
}
copycmp() {   # copycmp ARM: arm's greedy 33k-token copy output against the stock reference's byte for byte
  local r=$O/copy-fn4-${X93_COPYREF:-m2}-run1.txt b=$O/copy-fn4-$1-run1.txt
  if [ "$DRY" = 1 ]; then echo "+ cmp $r $b"
  elif [ -s $r ] && [ -s $b ]; then
    if cmp -s $r $b; then log "RESULT fn4 $1 vs ${X93_COPYREF:-m2} (95, stock 1.15.1) long-prompt copy output: IDENTICAL ($(wc -c < $b) bytes)"
    else log "RESULT fn4 $1 vs ${X93_COPYREF:-m2} (95, stock 1.15.1) long-prompt copy output: DIFFERENT from byte $(cmp $r $b 2>&1 | grep -oE 'byte [0-9]+' | head -1)"; fi
  else log "RESULT fn4 $1 vs ${X93_COPYREF:-m2} long-prompt copy output: not compared (missing $r or $b)"; fi
}
idcmp() {   # idcmp REF ARM GATE: warm-up answers + the 33k copy output, ARM against REF; GATE=1: DIFFERENT is a VIRHE
  local r=$1 a=$2 g=$3 w c
  if [ "$DRY" = 1 ]; then echo "+ warm-up answers + copy output: $a vs $r ($( [ $g = 1 ] && echo 'must be IDENTICAL' || echo information))"; return 0; fi
  warmcmp $O/guard/fn4-$r-warm-q.txt $O/guard/fn4-$a-warm-q.txt "$a vs $r"
  w=$(grep "RESULT fn4 warm-up $a vs $r:" $L | tail -1 | grep -oE 'IDENTICAL[^|]*|DIFFERENT[^|]*|not compared[^|]*|no common answers' | head -1)
  if [ -s $O/copy-fn4-$r-run1.txt ] && [ -s $O/copy-fn4-$a-run1.txt ]; then
    cmp -s $O/copy-fn4-$r-run1.txt $O/copy-fn4-$a-run1.txt && c="IDENTICAL ($(wc -c < $O/copy-fn4-$a-run1.txt) bytes)" || c="DIFFERENT from $(cmp $O/copy-fn4-$r-run1.txt $O/copy-fn4-$a-run1.txt 2>&1 | grep -oE 'byte [0-9]+' | head -1)"
  else c="not compared (missing copy output)"; fi
  if [ $g = 1 ] && { case "$w $c" in *DIFFERENT*) true ;; *) false ;; esac; }; then
    log "RESULT fn4 143 identity $a vs $r: VIRHE -- an exact arm differs: warm-up ${w:-?} | copy $c"
  else
    log "RESULT fn4 143 identity $a vs $r ($( [ $g = 1 ] && echo 'exact: must be IDENTICAL' || echo 'information: another engine')): warm-up ${w:-?} | copy $c"
  fi
}
HEATV="RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_PLACE_TURNOVER RADIANCE_STAGE_MIXED RADIANCE_STAGE_SPAN RADIANCE_ARENA_LEVEL_MAX RADIANCE_ARENA_HOLD_MS"
arm() {   # arm NAME: one arm of the production TP2 line (fn-tp2.yaml, 83+86), every phase, down
  local a=$1 tag=fn4-$1 rc tgpid="" sw=""
  unset $HEATV
  export FNIMAGE=$FNS15 MAXSEQS=16 HP=16384 MML=524288 EXTRA="" TP=2 CARDS="83+86 (fn-tp2s.yaml)" PLACE="expert_tiered hp16384" ROPE="yarn:2" MNBT=2048
  export FN4NR="" FN4GI="" FN4NW="" FN4WV="" FN4RV="" FN4UM=""
  case $a in
    dstk)  export FNIMAGE=$FNSTK MNBT=4096 EXTRA="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0" FN4NR=1 FN4NW=1 FN4WV=1 FN4RV=3 FN4UM=5 ;;
    s15)   ;;                                                                     # stock 1.15.1, production's flags
    s15n)  export EXTRA="--prefix-resume nearest" ;;
    t3)    export FNIMAGE=$FNT3 ;;                                                # the tuned3 image's own defaults (1.17.0 base)
    t3h)   export FNIMAGE=$FNT3 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16 ;;
    t3hs)  export FNIMAGE=$FNT3 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16 RADIANCE_STAGE_MIXED=0 ;;
    t3hsn) export FNIMAGE=$FNT3 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16 RADIANCE_STAGE_MIXED=0 EXTRA="--prefix-resume nearest" ;;
    *)     log "RESULT fn4 143: VIRHE unknown arm '$a'"; return 1 ;;
  esac
  for v in $HEATV FN4NR FN4GI FN4NW FN4WV FN4RV FN4UM; do [ -n "${!v:-}" ] && sw="$sw $v=${!v}"; done
  run "rm -f $O/$tag.abort $O/$tag.tstop"
  if [ "$DRY" = 1 ]; then echo "+ tguard $tag &"; else tguard $tag & tgpid=$!; fi
  log "RESULT fn4 $tag: start IMAGE=$FNIMAGE ($(docker image inspect -f '{{.Id}}' $FNIMAGE 2>/dev/null | cut -c8-19)) TP2 83+86 fn-tp2s.yaml (production dp-a line, test port) MNBT $MNBT ${EXTRA:-}${sw:- no engine switch} | window poller: $(kastate110)"
  gate $tag; rc=$?
  [ $rc = 1 ] && { keeplog $tag-fail; log "RESULT fn4 $tag: VIRHE not healthy -- $O/start-$tag.log: $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; run "touch $O/$tag.tstop"; return 1; }
  [ $rc = 2 ] && log "RESULT fn4 $tag: the 1-user gate stayed slow after one restart -- the arm runs on, its numbers flagged"
  pupair $tag; probe $tag
  local p
  for p in c4 c8 c16; do
    [ -e $R/sweep.stop ] && break; [ -e $O/$tag.abort ] && break
    msn $tag $p a; mbp $tag $p; msn $tag $p b; msd $tag $p
  done
  [ -e $R/sweep.stop ] || [ -e $O/$tag.abort ] || { pfrun $tag pf16 16384; pfrun $tag pf64-128 65536,131072; }
  [ -e $R/sweep.stop ] || [ -e $O/$tag.abort ] || copyt fn4-$a 0
  [ -e $R/sweep.stop ] || [ -e $O/$tag.abort ] || mbp $tag pf
  [ "${X143_BB:-0}" = 1 ] && [ ! -e $R/sweep.stop ] && [ ! -e $O/$tag.abort ] && bbph $tag image=$FNIMAGE tp=2 cards=83+86 placement=expert_tiered hp=16384 mml=524288 yarn=2 mtp=3 wire=wht6 arm=$a
  [ "${X110_CC:-0}" = 1 ] && { [ -e $R/sweep.stop ] || [ -e $O/$tag.abort ] || { msn $tag cc a; ccrun $tag; msn $tag cc b; msd $tag cc; }; }
  [ -e $O/$tag.abort ] && log "RESULT fn4 $tag: phases cut short by the temperature guard"
  run "docker logs $CN > $O/$tag-engine.log 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 $tag engine (heat / staging / arena / resume): $(grep -iE 'dual heat|decode pin|heat engine|prefill staging|arena level|prefix-resume|chained launches' $O/$tag-engine.log 2>/dev/null | sed 's/^[A-Z] //' | sort -u | head -6 | tr '\n' '|' | cut -c1-900)"
  down
  run "touch $O/$tag.tstop"; [ -n "$tgpid" ] && wait $tgpid 2>/dev/null
  log "RESULT fn4 $tag: done"
}
kastate110() {   # the window-wide poller's state (read-only; 110 neither starts nor stops it)
  [ "$DRY" = 1 ] && { echo "(dry)"; return 0; }
  local p; [ -s $R/keepawake.pid ] && p=$(cat $R/keepawake.pid) && kill -0 $p 2>/dev/null && tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q keepawake.py && { echo "alive pid $p"; return 0; }
  echo "NOT running (runs/keepawake.pid) -- this arm's state is unknown"
}
log "RESULT fn4 $ITEM: start ARMS [$ARMS] s15 $FNS15 ($(docker image inspect -f '{{.Id}}' $FNS15 2>/dev/null | cut -c8-19)) t3 $FNT3 ($(docker image inspect -f '{{.Id}}' $FNT3 2>/dev/null | cut -c8-19)) compose $Y cc_sim u$CCU [$CCARGS]"
if [ "$DRY" != 1 ]; then
  [ -e "$Y" ] || { log "RESULT fn4: VIRHE $Y not found"; exit 1; }
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  for i in $FNSTK; do docker image inspect $i > /dev/null 2>&1 || { log "RESULT fn4: VIRHE image $i missing"; exit 1; }; done
  [ -f $FNMODELDIR/$FNMODEL ] || { log "RESULT fn4: VIRHE $FNMODELDIR/$FNMODEL missing"; exit 1; }
  [ -e $Q/prodfc.py ] || { log "RESULT fn4: VIRHE $Q/prodfc.py missing"; exit 1; }
  [ -n "$TL" ] && [ -e $TL/msnap.py ] && [ -e $TL/servediff.py ] || { log "RESULT fn4: VIRHE tools missing beside the item (msnap.py / servediff.py)"; exit 1; }
  [ -e "$BBRUN" ] && [ -d "$BBROOT/betterbench" ] && [ -d "$BBDEPS/numpy" ] || { log "RESULT fn4: VIRHE BetterBench not ready (bb_run.py '$BBRUN', $BBROOT/betterbench, $BBDEPS/numpy)"; exit 1; }
  [ -e "$CCSIM" ] && grep -q -- '--pair' $CCSIM && grep -q -- '--trace-seeded' $CCSIM && [ -e "$CCWRAP" ] || { log "RESULT fn4: VIRHE cc_sim not ready ($CCSIM with --pair / --trace-seeded, wrapper $CCWRAP)"; exit 1; }
  mkdir -p $O $O/guard
  printf '%s\n' '{"concurrency_levels": [1, 2, 4, 8, 16], "concurrency_requests": 32, "prefill_runs": 4, "prefill_warmup": 1}' > $O/bb-config.json
  if [ -e "$PRODY" ]; then
    cp $PRODY $O/110-prod-dp-a.copy.yaml
    log "RESULT fn4 $ITEM serve line vs production (read-only copy of $PRODY): $(timeout 30 python3 -I $TL/servediff.py $O/110-prod-dp-a.copy.yaml $Y 2>&1 | cut -c1-700) | prod image line: $(grep -m1 -E '^ *image:' $O/110-prod-dp-a.copy.yaml | tr -s ' ' | cut -c1-120)"
  else
    log "RESULT fn4 $ITEM: production yaml $PRODY not found -- the serve line is fn-tp2.yaml's (the 4.11. prod dp-a copy), not re-checked"
  fi
  trap 'down' EXIT
  trap 'exit 143' TERM INT
else
  echo "+ mkdir -p $O $O/guard; bb-config.json; cp $PRODY $O/110-prod-dp-a.copy.yaml; servediff.py it vs $Y; trap: compose down"
fi
done_arms=""
for a in $ARMS; do
  [ -e $R/sweep.stop ] && break
  arm $a && done_arms="$done_arms $a"
done
down
# the arm vs 110b's s15 of today (its mb / probe / guard / copy / warm-up files in runs/fn4); identity is information (another engine)
for a in $done_arms; do
  cmparm s15 $a
  idcmp s15 $a 0
done
log "RESULT fn4 $ITEM: done --$done_arms | ref 110b s15 (10.11.): mb c4 / c8 / c16 347.9 / 455.3 / 573.7, prefill 4826 / 4799 / 4815, mb pf 3542"
