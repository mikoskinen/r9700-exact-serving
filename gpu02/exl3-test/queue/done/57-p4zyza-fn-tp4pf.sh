#!/bin/bash
# 95-fn-tp4pf.sh (queue item, 8.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~30 min for the five
# arms) -- Flash-Next TP4 prefill: the ~7.7k tok/s cap by step size, and where a prefill step's time goes per rank.
# WHY: TP3 and TP4 both cap near 7.7k tok/s (87: t4 16k 7727, cold 128k 7419.9; TP3 cold 128k 7158), while TP3 -> TP4 cuts
# each card's expert share by 25 %. The 3.11. TP4 profile's mixed windows: collectives 24.7 %, hc 20.6 % (replicated on
# every rank: hc_read's mixing GEMMs at 2048 rows), experts 29.7 %. A lone prefill step carries one checkpoint interval
# (2048) unless --prefill-span lets it carry more, so MNBT alone does not grow it.
#   m2   the production line: MNBT 2048, span 1                         (unprofiled: tok/s)
#   m4   MNBT 4096 --prefill-span 2 (2 x 2048, checkpoints unchanged)     (unprofiled)
#   m8   MNBT 8192 --prefill-span 4                                       (unprofiled)
#   q2   m2 with --profile-ops (per-op device timing, synchronised; RADIANCE_PROFILE_EVERY 100000 = only the exit table):
#        one cold 64k prefill, then docker stop (the exit writes the table), tools/opsum.py per rank by class
#   q8   m8 with --profile-ops, the same
# Unprofiled arms: gate (87's qgate method) -> 16k warm-up prefill -> fresh 64k and 128k prefills (ds41_tests prefill,
# nonce first; tok/s and TTFT) -> the long-context copy test at a fixed offset (a ~33k-token prompt, greedy, 900 tokens:
# m4 / m8 vs m2 byte for byte -- the pipelined step's exactness) -> mb pf (31k beside 4 decoders) -> mb c8 -> down. The
# engine log says whether a step was pipelined ("prefill steps wider than a MoE pass are pipelined"). Profiled arms:
# no gate (profiling slows every step), one fresh 64k prefill, stop, log, opsum. Exact: MNBT / span change only how a
# prefill is cut into steps (the sums inside a step are the same per token; the wire floor is unchanged, so a bigger step
# is on the same lossy ti8 wire as today's 2048-token step). Per-rank stamps are not used here (prefill steps are not
# played passes). Image: stock 1.15.1 code (the stamp / no-overlap binary with both off, X95_IMAGE).
# REFERENCES (no reruns): 87 t4 (TP4 all_vram MML 524288 yarn) pf 5728.8 tok/s beside 4 decoders, cold 128k 7419.9,
# warm-up 16k 7727; 3.11. TP4 sweep (exact wire, MNBT 8192, span 2, ckpt 4096) pf 7049.6, 16k 7883.
# DRYRUN=1 bash 95-fn-tp4pf.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4o.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
FN91=${X95_IMAGE:-radiance-local:1.15.1-stampcards-fbf436a}; export FNIMAGE=$FN91
export FNMODELDIR=$HOME/models/flashnext-iq4r FNMODEL=qwen3.8-next-flash-fp8-iq4r-moe.rad
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
CARDS4=$C3,$C83,$C86,$C03; CARDS2=$C3,$C03
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/rankskew.py" ] && [ -e "$f/pwsum.py" ] && { TL=$f; break; }; done
ARMS=${ARMS:-"m2 m4 m8 q2 q8"}; ITEM=95-fn-tp4pf
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
pfrun() {   # pfrun TAG NAME DEPTHS: fresh prefills (ds41_tests prefill, nonce first) -> RESULT line
  run "(cd $D && timeout 1500 python3 ds41_tests.py prefill --depths $3 --cpt 3.6 --calibrate 0 --reps 1 > $O/$1-$2.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT fn4 $1 prefill $2: $(python3 -c 'import json,sys; print("; ".join("%s tok %s tok/s TTFT %s s" % (d.get("prompt_tokens"), d.get("prefill_tps"), d.get("ttft")) for d in (json.loads(l[2:]) for l in open(sys.argv[1]) if l.startswith("J "))))' $O/$1-$2.txt 2>&1 | cut -c1-400)"
}
cpx() {   # cpx TAG: the long-context copy test at a fixed offset (a ~33k-token prompt, greedy, 900 tokens out): the same
  # prompt in every m arm, so m2 / m4 / m8 outputs compare byte for byte (a fresh server each: no prefix-cache carry-over)
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset 0 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT fn4 $1 copy (33k-token prompt, greedy): $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, TTFT %s s, exact %s" % (u.get("prompt_tokens"), d.get("ttft"), d.get("exact")))' 2>&1 | cut -c1-200)"
}
arm() {   # arm NAME
  local a=$1 tag=fn4-$1 rc tgpid="" pw=""
  export FNIMAGE=$FN91 TP=4 CARDS=$CARDS4 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=2048 EXTRA="" NST=3 WIREKB=128 NOOVL="" PROFEVERY=""
  export AQLSTAMP="" AQLCARDS="" STATSDIR=/tmp
  case $a in
    m2|q2) ;;
    m4)    export MNBT=4096 EXTRA="--prefill-span 2" ;;
    m8|q8) export MNBT=8192 EXTRA="--prefill-span 4" ;;
    *)     log "RESULT fn4: VIRHE unknown arm '$a'"; return 1 ;;
  esac
  case $a in q*) export EXTRA="${EXTRA:+$EXTRA }--profile-ops" PROFEVERY=100000 ;; esac
  log "RESULT fn4 $tag: start IMAGE=$FNIMAGE TP=$TP cards $CARDS PLACE=$PLACE MML=$MML MAXSEQS=$MAXSEQS MNBT=$MNBT NST=$NST ${EXTRA:-} | power:$( [ "$DRY" = 1 ] && echo ' (dry)' || caps)"
  # the temperature guard (91's): junction >= 100 C or mem >= 96 C on any card for 10 s -> the arm's remaining phases skipped
  run "rm -f $O/$tag.abort $O/$tag.tstop"
  if [ "$DRY" = 1 ]; then echo "+ tguard $tag &"; else tguard $tag & tgpid=$!; fi
  case $a in
  m*)
    gate $tag; rc=$?
    [ $rc = 1 ] && { keeplog $tag-fail; log "RESULT fn4 $tag: VIRHE not healthy -- $O/start-$tag.log: $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; armend $tag; return 1; }
    [ $rc = 2 ] && { log "RESULT fn4 $tag: VIRHE slow state after one restart -> arm skipped"; down; armend $tag; return 1; }
    [ "$DRY" = 1 ] || log "RESULT fn4 $tag start: $(grep -m1 -E '^I kv: the pool' $O/start-$tag.log | cut -c3-200) | $(grep -m1 -E '^I activation arena' $O/start-$tag.log | cut -c3-220) | $(grep -m1 -E '^ +kv cache +[0-9.]+ GiB' $O/start-$tag.log | tr -s ' ' | cut -c1-120)"
    pfrun $tag pf16 16384
    [ -e $O/$tag.abort ] || pfrun $tag pf64-128 65536,131072
    [ -e $O/$tag.abort ] || cpx $tag
    for p in pf c8; do [ -e $R/sweep.stop ] && break; [ -e $O/$tag.abort ] && { log "RESULT fn4 $tag: phases from $p on skipped (temperature guard)"; break; }; mbp $tag $p; done
    run "docker logs $CN > $O/$tag-engine.log 2>&1"
    # step_pipelined (qwen4exp_fp8.cpp: a step wider than a MoE pass of 2048 rows, all-reduces on the second lane) says so once
    [ "$DRY" = 1 ] || log "RESULT fn4 $tag prefill path: $(grep -m1 -oE 'prefill steps wider than a MoE pass are pipelined: [^\n]*' $O/$tag-engine.log | cut -c1-200 || true)$(grep -q 'are pipelined' $O/$tag-engine.log || echo 'serial (no pipelined-step line: step_serial)')"
    down ;;
  q*)
    restart $tag || { keeplog $tag-fail; log "RESULT fn4 $tag: VIRHE not healthy (profiled)"; down; armend $tag; return 1; }
    pfrun $tag pf64 65536
    # the exit table: stop (SIGTERM, the engine prints its last per-op table), then the log, then down
    run "docker stop -t 120 $CN > /dev/null 2>&1; docker logs $CN > $O/$tag-engine.log 2>&1"
    down
    [ "$DRY" = 1 ] && { echo "+ python3 $TL/opsum.py $O/$tag-engine.log $tag"; } || {
      python3 $TL/opsum.py $O/$tag-engine.log $tag > $O/$tag-ops.txt 2>&1
      log "RESULT fn4 $tag per-op (cold 64k prefill + start-up, cumulative, synchronised -- shares, not speed): $(tr -s ' ' < $O/$tag-ops.txt | tr '\n' '|' | cut -c1-2400)"; } ;;
  esac
  armend $tag
  log "RESULT fn4 $tag: done"
}
armend() {   # armend TAG: stop the temperature guard
  run "touch $O/$1.tstop"; [ -n "${tgpid:-}" ] && { wait $tgpid 2>/dev/null; tgpid=""; }
  return 0
}
log "RESULT fn4 $ITEM: start ARMS [$ARMS] IMAGE=$FNIMAGE ($(docker image inspect -f '{{.Id}}' $FNIMAGE 2>/dev/null | cut -c8-19)) compose $Y"
if [ "$DRY" != 1 ]; then
  [ -e "$Y" ] && grep -q 'RADIANCE_PROFILE_EVERY' $Y || { log "RESULT fn4: VIRHE $YN not found or without RADIANCE_PROFILE_EVERY"; exit 1; }
  [ -n "$TL" ] && [ -e $TL/opsum.py ] || { log "RESULT fn4: VIRHE tools/opsum.py missing beside the item"; exit 1; }
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  docker image inspect $FNIMAGE > /dev/null 2>&1 || { log "RESULT fn4: VIRHE image $FNIMAGE missing"; exit 1; }
  [ -f $FNMODELDIR/$FNMODEL ] || { log "RESULT fn4: VIRHE $FNMODELDIR/$FNMODEL missing"; exit 1; }
  [ -e $Q/prodfc.py ] || { log "RESULT fn4: VIRHE $Q/prodfc.py missing"; exit 1; }
  for u in ${CARDS4//,/ }; do grep -qi "${u#GPU-}" /sys/class/drm/card*/device/unique_id 2>/dev/null || { log "RESULT fn4: VIRHE card $u not found on this host"; exit 1; }; done
  mkdir -p $O $O/guard
  HWM=$(hwmons); [ $(echo $HWM | wc -w) = 4 ] || { log "RESULT fn4: VIRHE hwmon of the four cards not found ($HWM) -- the temperature guard needs them"; exit 1; }
  BASECAPS=$(for h in $HWM; do cat $h/power1_cap; done | tr '\n' ' ' | sed 's/ $//')   # the guard's restore writes these back (no power change here)
  trap 'down' EXIT
  trap 'exit 143' TERM INT
else
  echo "+ mkdir -p $O $O/guard; trap: compose down on EXIT/TERM/INT; hwmons for the temperature guard"
  HWM="h0 h1 h2 h3"; BASECAPS="210000000 210000000 210000000 210000000"
fi
done_arms=""
for a in $ARMS; do
  [ -e $R/sweep.stop ] && break
  arm $a && done_arms="$done_arms $a"
done
down
# exactness: a pipelined piece computes the bytes a span-1 chunk of the same rows does (qwen4exp_fp8.cpp's own claim), so
# m4 / m8's greedy copy output after the same 33k-token prompt must equal m2's byte for byte (the warm-up answers are short
# prompts that never pipeline: they check the rest of the path only)
for a in m4 m8; do case " $done_arms " in *" $a "*) case " $done_arms " in *" m2 "*)
  warmcmp $O/guard/fn4-m2-warm-q.txt $O/guard/fn4-$a-warm-q.txt "$a vs m2 (short prompts, not pipelined)"
  if [ "$DRY" = 1 ]; then echo "+ cmp $O/copy-fn4-m2-run1.txt $O/copy-fn4-$a-run1.txt"
  elif [ -s $O/copy-fn4-m2-run1.txt ] && [ -s $O/copy-fn4-$a-run1.txt ]; then
    if cmp -s $O/copy-fn4-m2-run1.txt $O/copy-fn4-$a-run1.txt; then log "RESULT fn4 $a vs m2 long-prompt copy output (pipelined prefill vs serial): IDENTICAL ($(wc -c < $O/copy-fn4-$a-run1.txt) bytes)"
    else log "RESULT fn4 $a vs m2 long-prompt copy output (pipelined prefill vs serial): DIFFERENT from byte $(cmp $O/copy-fn4-m2-run1.txt $O/copy-fn4-$a-run1.txt 2>&1 | grep -oE 'byte [0-9]+' | head -1) ($(wc -c < $O/copy-fn4-m2-run1.txt) vs $(wc -c < $O/copy-fn4-$a-run1.txt) bytes)"; fi
  else log "RESULT fn4 $a vs m2 long-prompt copy output: not compared (missing $O/copy-fn4-m2-run1.txt or $O/copy-fn4-$a-run1.txt)"; fi ;; esac ;; esac; done
[ "$DRY" = 1 ] || log "RESULT fn4 $ITEM table: $(for a in $done_arms; do case $a in m*) echo -n "$a MNBT $(case $a in m2) echo 2048;; m4) echo 4096;; m8) echo 8192;; esac): $(python3 -c 'import json,sys
r=[]
for f in sys.argv[1:]:
    try:
        for l in open(f):
            if l.startswith("J "): d=json.loads(l[2:]); r.append("%s tok %s tok/s" % (d.get("prompt_tokens"), d.get("prefill_tps")))
    except OSError: pass
print(", ".join(r))' $O/fn4-$a-pf16.txt $O/fn4-$a-pf64-128.txt 2>&1 | cut -c1-200); pf $(grep '^MB ' $O/fn4-$a-pf.txt 2>/dev/null | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[3:]); print(d.get("prefill_tps"), "tok/s, kept", d.get("decode_kept"))' 2>/dev/null) ## " ;; esac; done | cut -c1-1800)"
log "RESULT fn4 $ITEM: done --$done_arms | refs: 87 t4 pf 5728.8 tok/s beside 4 decoders, cold 128k 7419.9, 16k 7727; 3.11. TP4 sweep (exact, MNBT 8192 span 2 ckpt 4096) pf 7049.6, 16k 7883"
