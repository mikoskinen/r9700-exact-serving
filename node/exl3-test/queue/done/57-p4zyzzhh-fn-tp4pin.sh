#!/bin/bash
# 10.11.2026 (fn4 dev agent, coordinator audit): cc_sim OFF by default -- its forced-mode pair is invalid with MTP 3 (forced
# lengths collapse draft acceptance; natural / reflen pairs diverge, 123). Every mb / prefill / stamp phase unchanged. Original in held/.
# 112-fn-tp4pin.sh (queue item, 9.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~2 h: the probe
# ~2 min, k4nP0 and k4nPin ~50 min each with cc_sim, k4nPinX ~20 min without) -- the TP4 rank skew fixed by host-thread
# placement? 111b (k4n, stamps on all cards, fast state): per-card compute equal within 0.3 % (exposed experts 8513-8541,
# linear 1103-1128, attention 2157-2185 us a pass at c8) but the gaps differ (c8: c3 4880, 83 4691, 86 4748, 03 4973 us;
# c16: 5282 / 5025 / 5136 / 5328) and the last-to-arrive share follows them (03 46.8 %, c3 28.9 %, 83 12.0 %, 86 12.2 %
# at c8); ~1-1.5 % of the pass. Hypothesis: each rank's issuing thread lands on an arbitrary CCD (BIOS NPS1: numa_node -1
# for every card), so a rank whose thread is far from its card's IOD quadrant writes doorbells / has its packets and
# signals snooped across the die. Cards: c3:00 under c0:01.1, 83:00 under 80:01.1, 86:00 under 80:03.1, 03:00 under 00:01.1.
# Radiance 1.15.1 has no thread placement option: rank threads are plain std::threads of the one process (rank_loop,
# created in rank order, then the step loop's), unnamed and unpinned. IMAGE radiance-local:1.15.1-pin-7e4c36d-stampcards
# (X112_IMAGE) = the stampcards image + bin/radiance from branch fn4-pin 7e4c36d (on c0cbe5a): RADIANCE_RANK_CPUS="L0;L1;.."
# pins rank i's thread to CPU list Li (pthread_setaffinity_np), RADIANCE_STEP_CPUS the step thread, RADIANCE_THREAD_NAMES=1
# names them rad-rank<i> / rad-step; unset or empty = stock (nothing named or pinned). No sysfs write anywhere.
# THE MAP, measured first (cards idle): tools/cpulat (HIP) from one CPU of each of the 8 L3 groups (CCX) times an empty
# kernel launch + sync and a 4-byte D2H + sync on every card (5000 each) -> per card the CCDs (CCD k = CPUs 4k..4k+3 and
# 16+4k..16+4k+3) by latency; tools/pinplan.py turns that into RADIANCE_RANK_CPUS: near (each rank on its fastest CCD; two
# ranks wanting one CCD get one CCX each) and far (the slowest).
# ARMS (X112_ARMS) default "k4nP0 k4nPin k4nPinX", all on the k4n line (TP4 all_vram MML 262144 MAXSEQS 16 MNBT 2048 MTP 3
# --prefix-resume nearest, fn-tp4q.yaml), stamps every X112_STAMP_EVERY (10)th pass on ALL cards:
#   k4nP0    no pinning, threads named -- the reference re-run (106's k4n ran another image with card-0 stamps only, so
#            its gaps are not comparable); where the scheduler puts each rank thread is sampled (tools/threadcpu.py)
#   k4nPin   RADIANCE_RANK_CPUS = the near plan        k4nPinX  = the far plan (the effect's size; no cc_sim)
# Per arm: gate, state probe (mb c8 45 s; restart while slow, at most 2), the engine's placement lines, mb c4 / c8 / c16
# (+ engine stats; c8 / c16 through rankskew.py + cardlag.py: gaps, exposed classes, arrival lag and who is last, per
# card), the rank threads' CPUs during c8, prefill 16k, cc_sim u16 (X112_CC_ARMS "k4nP0 k4nPin"). END: Pin / PinX vs P0
# (cmparm, cc_sim --pair), the cc arms vs 106's k4n, and one line of gaps / lag / last share per arm and window.
# Host tuning (C2 off, governor performance) and the window poller are left as they are; their state is logged.
# DRYRUN=1 bash 112-fn-tp4pin.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4q.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
FN91=${X91_IMAGE:-radiance-local:1.15.1-stampcards-c0cbe5a}; export FNIMAGE=$FN91
export FNMODELDIR=$HOME/models/flashnext-iq4r FNMODEL=qwen3.8-next-flash-fp8-iq4r-moe.rad
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
CARDS4=$C3,$C83,$C86,$C03; CARDS2=$C3,$C03
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/rankskew.py" ] && [ -e "$f/pwsum.py" ] && [ -e "$f/slowprobe.py" ] && [ -e "$f/msnap.py" ] && { TL=$f; break; }; done
ARMS=${X112_ARMS:-"k4nP0 k4nPin k4nPinX"}; ITEM=112-fn-tp4pin   # empty: k4s + the X93_WINNER pick (default auto: tools/pick93.py over the run log)
BBROOT=${BB_ROOT:-$HOME/gpu/ext/betterbench}; BBDEPS=${BB_DEPS:-$HOME/gpu/ext/bb-deps}
BBRUN=; for f in "$H/tools/bb_run.py" "$H/../tools/bb_run.py" "$HOME/gpu/ds41/exl3-test/queue-ar/tools/bb_run.py"; do [ -e "$f" ] && { BBRUN=$f; break; }; done
CCSIM=${X93_CCSIM:-$HOME/gpu/ds41/exl3-test/tools/ccsim/cc_sim.py}; CCWRAP=${X93_CCWRAP:-$HOME/gpu/ds41/exl3-test/ccsim-exl3-turns.sh}
CCU=${X93_CCU:-16}; CCTO=${X93_CCTIMEOUT:-3300}
CCARGS=${X93_CCARGS:-"--duration 3000 --turns 15 --warmup-turns 3 --effort xhigh,medium --gen-mode forced --seed 1 --cpt-fixed --temperature 0 --trace-seeded"}
FNP4=${X92_PR396_IMAGE:-radiance-local:1.15.1-pr396-0ae1545-stampcards}
FNPMIN=${X93_PMIN_IMAGE:-radiance-local:1.15.1-pmin-47e859a-stampcards}
FNPIN=${X112_IMAGE:-radiance-local:1.15.1-pin-7e4c36d-stampcards}; FNS15=${X112_IMAGE_S15:-stilldeadcode/radiance:1.15.1}
RANKBDFS=c3:00.0,83:00.0,86:00.0,03:00.0   # the ranks' cards in CARDS4 (= ROCR = rank) order; the arms log the engine's own table
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
hostline() {   # hostline: governors, C2 disable count, the window poller (non-root reads, once)
  local n1 n0 g
  n1=$(grep -l '^1' /sys/devices/system/cpu/cpu*/cpuidle/state2/disable 2>/dev/null | wc -l)
  n0=$(ls /sys/devices/system/cpu/cpu*/cpuidle/state2/disable 2>/dev/null | wc -l)
  g=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c | awk '{printf "%s x%s ", $2, $1}')
  echo "C2 disabled on $n1/$n0 CPUs | governor ${g}| window poller $( [ -s $R/keepawake.pid ] && kill -0 $(cat $R/keepawake.pid) 2>/dev/null && echo "alive pid $(cat $R/keepawake.pid)" || echo 'NOT running')"
}
dockgpu() {   # dockgpu NAME TIMEOUT ARGS...: one bounded container with the cards (production's device / group setup)
  local n=$1 t=$2; shift 2
  run "timeout -k 10 $t docker run --rm --init --name $n --device /dev/kfd --device /dev/dri --group-add 44 --group-add 991 -u 1000:1000 --security-opt seccomp=unconfined --ulimit memlock=-1 $*"
}
probe8() {   # probe8 TAG DUR: an mb c8 -> "ms tok/s"
  [ "$DRY" = 1 ] && { echo "+ (cd $D && timeout 400 python3 mb.py --tag fn4-$1 --phases c8 --dur $2 --warm 10 > $O/$1-c8.txt)" >&2; echo "25.6 744"; return 0; }
  (cd $D && timeout 400 python3 mb.py --tag fn4-$1 --phases c8 --dur $2 --warm 10 > $O/$1-c8.txt 2>&1)
  grep '^MB ' $O/$1-c8.txt 2>/dev/null | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[3:]); print(d.get("ms_decode_step"), d.get("agg_dec_tps"))' 2>/dev/null || echo "0 0"
}
win() {   # win TAG WINDOW T0 T1: rankskew.py (every rank) and cardlag.py (83:00's rank) over the window -> RESULT lines
  if [ "$DRY" = 1 ]; then echo "+ rankskew.py + cardlag.py --card $FOCUS over $STATSDIR/stamps.csv(.c1 .c2 .c3) --from $3 --to $4 -> $O/$1-$2-{rank,lag}.{txt,json}"; return 0; fi
  python3 $TL/rankskew.py $STATSDIR/stamps.csv --from $3 --to $4 --label $1-$2 --json $O/$1-$2-rank.json > $O/$1-$2-rank.txt 2>&1
  python3 $TL/cardlag.py $STATSDIR/stamps.csv --card $FOCUS --from $3 --to $4 --label $1-$2 --json $O/$1-$2-lag.json > $O/$1-$2-lag.txt 2>&1
  log "RESULT fn4 112 $1 $2 per rank (card 0 = c3:00, 1 = 83:00, 2 = 86:00, 3 = 03:00; us a pass): $(tr -s ' ' < $O/$1-$2-rank.txt | tr '\n' '|' | cut -c1-2000)"
  log "RESULT fn4 112 $1 $2 arrival lag: $(tr -s ' ' < $O/$1-$2-lag.txt | grep -E 'card [0-9]: last|passes' | tr '\n' '|' | cut -c1-1500)"
}
# ---------------------------------------------------------------- the CCD <-> card map, measured
placement_map() {
  dockgpu fn4-112-cpulat 600 "-e ROCR_VISIBLE_DEVICES=$CARDS4 -v $TL:/t:ro --entrypoint /t/cpulat $FNS15 ${X112_LAT_REPS:-5000} > $O/112-cpulat.txt 2>&1"
  run "docker rm -f fn4-112-cpulat > /dev/null 2>&1"
  if [ "$DRY" = 1 ]; then
    echo "+ pinplan.py $O/112-cpulat.txt $RANKBDFS near / far -> RADIANCE_RANK_CPUS"
    CPUSNEAR="12-15,28-31;8-9,24-25;10-11,26-27;0-3,16-19"; CPUSFAR="0-1,16-17;4-7,20-23;2-3,18-19;12-15,28-31"; return 0
  fi
  grep -q '^CPULAT done' $O/112-cpulat.txt || { log "RESULT fn4 112 cpulat: VIRHE the latency probe did not finish: $(tail -3 $O/112-cpulat.txt | tr '\n' ' ' | cut -c1-300)"; return 1; }
  log "RESULT fn4 112 cpulat (empty kernel launch + sync, us median from one CPU of each L3 group; CCD k = CPUs 4k..4k+3 + SMT): $(grep -E '^NEAR' $O/112-cpulat.txt | tr '\n' '|' | cut -c1-1500)"
  log "RESULT fn4 112 cpulat d2h4 (4-byte D2H + sync, us median): $(grep '^LAT d2h4' $O/112-cpulat.txt | awk '{print $3 " " $6 " " $8}' | tr '\n' ';' | cut -c1-1200)"
  log "RESULT fn4 112 cpulat: $(grep '^CCDMAP' $O/112-cpulat.txt | cut -c1-600)"
  local pn pf
  pn=$(python3 -I $TL/pinplan.py $O/112-cpulat.txt $RANKBDFS near 2>&1); pf=$(python3 -I $TL/pinplan.py $O/112-cpulat.txt $RANKBDFS far 2>&1)
  CPUSNEAR=$(echo "$pn" | sed -n 's/^CPUS //p'); CPUSFAR=$(echo "$pf" | sed -n 's/^CPUS //p')
  log "RESULT fn4 112 placement: $(echo "$pn" | grep '^PLAN' | cut -c1-700) | $(echo "$pf" | grep '^PLAN' | cut -c1-700)"
  [ -n "$CPUSNEAR" ] && [ -n "$CPUSFAR" ] || { log "RESULT fn4 112 placement: VIRHE no plan -- the pinned arms are skipped"; return 1; }
}
arm() {   # arm NAME: one arm of the k4n line on the fn4-pin image, stamps on all four cards
  local a=$1 tag=fn4-$1 rc tgpid="" kp="" st ms tps tries=0 t0 t1 p
  export FNIMAGE=$FNPIN TP=4 CARDS=$CARDS4 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=2048 EXTRA="--prefix-resume nearest" NST=3 WIREKB=128 NOOVL="" PROFEVERY="" PMIN="" PMINB=""
  export AQLSTAMP=${X112_STAMP_EVERY:-10} AQLCARDS=all STATSDIR=$O/stats112-$a TNAMES=1 RANKCPUS="" STEPCPUS=""
  case $a in
    k4nP0)   ;;                                                  # names only: where the scheduler puts the rank threads
    k4nPin)  [ -n "$CPUSNEAR" ] || { log "RESULT fn4 $tag: skipped (no placement plan)"; return 1; }; export RANKCPUS=$CPUSNEAR ;;
    k4nPinX) [ -n "$CPUSFAR" ] || { log "RESULT fn4 $tag: skipped (no placement plan)"; return 1; }; export RANKCPUS=$CPUSFAR ;;
    *)       log "RESULT fn4 112: VIRHE unknown arm '$a'"; return 1 ;;
  esac
  run "mkdir -p $STATSDIR; find $STATSDIR -maxdepth 1 -name 'stamps.csv*' -delete; find $O -maxdepth 1 -name '$tag.abort' -delete; find $O -maxdepth 1 -name '$tag.tstop' -delete"
  if [ "$DRY" = 1 ]; then echo "+ tguard $tag &"; else tguard $tag & tgpid=$!; fi
  log "RESULT fn4 $tag: start IMAGE=$FNIMAGE ($(docker image inspect -f '{{.Id}}' $FNIMAGE 2>/dev/null | cut -c8-19)) k4n line, stamps every ${AQLSTAMP}th pass on all cards, RADIANCE_THREAD_NAMES=1 RADIANCE_RANK_CPUS='${RANKCPUS}'"
  gate $tag; rc=$?
  [ $rc = 1 ] && { keeplog $tag-fail; log "RESULT fn4 $tag: VIRHE not healthy -- $O/start-$tag.log: $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; run "touch $O/$tag.tstop"; return 1; }
  [ $rc = 2 ] && log "RESULT fn4 $tag: the 1-user gate stayed slow after one restart -- the arm runs on, its numbers flagged"
  if [ "$DRY" != 1 ]; then
    log "RESULT fn4 $tag thread placement (engine): $(grep -E 'thread placement' $O/start-$tag.log | sed 's/^[A-Z] //' | tr '\n' '|' | cut -c1-700) | ranks: $(grep -oE 'card 0000:[0-9a-f]{2}:00\.0 +rank [0-9]+' $O/start-$tag.log | tr -s ' ' | tr '\n' ',' | cut -c1-200)"
    kp=$(grep -oE 'card 0000:83:00\.0 +rank [0-9]+' $O/start-$tag.log | grep -oE '[0-9]+$' | head -1)
    [ -n "$kp" ] && FOCUS=$kp
  fi
  while :; do
    set -- $(probe8 $tag-st$tries 45); ms=$1; tps=$2
    st=$(awk -v m=$ms 'BEGIN{print (m <= 0) ? "unknown" : (m <= 26.8) ? "fast" : (m >= 28.5) ? "slow" : "between"}')
    log "RESULT fn4 $tag state probe $tries: mb c8 $ms ms $tps tok/s -> $st"
    { [ "$st" = slow ] && [ $tries -lt ${X112_STATE_TRIES:-2} ]; } || break
    tries=$((tries + 1)); restart $tag-r$tries || { down; run "touch $O/$tag.tstop"; return 1; }
  done
  run "echo '$st' > $O/$tag-state.txt"
  local tcp=""
  for p in c4 c8 c16; do
    [ -e $R/sweep.stop ] && break; [ -e $O/$tag.abort ] && break
    if [ $p = c8 ]; then
      if [ "$DRY" = 1 ]; then echo "+ python3 -I $TL/threadcpu.py \$(docker inspect -f '{{.State.Pid}}' $CN) 75 0.2 > $O/$tag-threadcpu.txt &"
      else timeout -k 5 120 python3 -I $TL/threadcpu.py $(docker inspect -f '{{.State.Pid}}' $CN) 75 0.2 > $O/$tag-threadcpu.txt 2>&1 & tcp=$!; fi
    fi
    msn $tag $p a; t0=$(now); mbp $tag $p; t1=$(now); msn $tag $p b; msd $tag $p
    case $p in c8|c16) win $tag $p $t0 $t1 ;; esac
  done
  [ -n "$tcp" ] && { wait $tcp 2>/dev/null; log "RESULT fn4 $tag rank threads' CPUs during c8 (host /proc, CCD k = CPUs 4k..4k+3 + SMT): $(grep -E '^THREADCPU rad-' $O/$tag-threadcpu.txt | cut -c11- | tr '\n' '|' | cut -c1-1200)"; }
  [ -e $R/sweep.stop ] || [ -e $O/$tag.abort ] || pfrun $tag pf16 16384
  case " ${X112_CC_ARMS:-} " in *" $a "*) [ -e $R/sweep.stop ] || [ -e $O/$tag.abort ] || { msn $tag cc a; ccrun $tag; msn $tag cc b; msd $tag cc; } ;; esac
  [ -e $O/$tag.abort ] && log "RESULT fn4 $tag: phases cut short by the temperature guard"
  run "docker logs $CN > $O/$tag-engine.log 2>&1"
  down
  run "touch $O/$tag.tstop"; [ -n "$tgpid" ] && wait $tgpid 2>/dev/null
  run "gzip -f $STATSDIR/stamps.csv* 2>/dev/null || true"
  log "RESULT fn4 $tag: done ($st)"
}
log "RESULT fn4 $ITEM: start ARMS [$ARMS] IMAGE $FNPIN ($(docker image inspect -f '{{.Id}}' $FNPIN 2>/dev/null | cut -c8-19)) compose $Y cc_sim u$CCU [$CCARGS] on [${X112_CC_ARMS:-}] | host: $( [ "$DRY" = 1 ] && echo dry || hostline)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  for i in $FNPIN $FNS15; do docker image inspect $i > /dev/null 2>&1 || { log "RESULT fn4: VIRHE image $i missing"; exit 1; }; done
  [ -e "$Y" ] && grep -q RADIANCE_RANK_CPUS $Y || { log "RESULT fn4: VIRHE $Y not found or without RADIANCE_RANK_CPUS"; exit 1; }
  [ -n "$TL" ] && [ -x $TL/cpulat ] && [ -e $TL/pinplan.py ] && [ -e $TL/threadcpu.py ] && [ -e $TL/rankskew.py ] && [ -e $TL/cardlag.py ] && [ -e $TL/msnap.py ] || { log "RESULT fn4: VIRHE tools missing (cpulat / pinplan.py / threadcpu.py / rankskew.py / cardlag.py / msnap.py)"; exit 1; }
  [ -f $FNMODELDIR/$FNMODEL ] || { log "RESULT fn4: VIRHE $FNMODELDIR/$FNMODEL missing"; exit 1; }
  [ -e $Q/prodfc.py ] || { log "RESULT fn4: VIRHE $Q/prodfc.py missing"; exit 1; }
  [ -e "$CCSIM" ] && grep -q -- '--pair' $CCSIM && [ -e "$CCWRAP" ] || { log "RESULT fn4: VIRHE cc_sim not ready ($CCSIM, $CCWRAP)"; exit 1; }
  for u in ${CARDS4//,/ }; do grep -qi "${u#GPU-}" /sys/class/drm/card*/device/unique_id 2>/dev/null || { log "RESULT fn4: VIRHE card $u not found on this host"; exit 1; }; done
  mkdir -p $O $O/guard
  trap 'down; docker rm -f fn4-112-cpulat > /dev/null 2>&1' EXIT
  trap 'exit 143' TERM INT
else
  echo "+ mkdir -p $O $O/guard; trap: compose down + the probe container removed"
fi
FOCUS=1; CPUSNEAR=""; CPUSFAR=""
placement_map
done_arms=""
for a in $ARMS; do
  [ -e $R/sweep.stop ] && break
  arm $a && done_arms="$done_arms $a"
done
down
# pairs: the pinned arms vs the unpinned one (same image, stamps, window); every cc arm also vs 106's k4n (stock image,
# card-0 stamps only: information on the profiling overhead and the run-to-run spread)
for a in $done_arms; do
  [ $a = k4nP0 ] && continue
  case " $done_arms " in *" k4nP0 "*) cmparm k4nP0 $a; ccpair93 fn4-k4nP0 fn4-$a ;; esac
done
for a in $done_arms; do case " ${X112_CC_ARMS:-} " in *" $a "*) ccpair93 fn4-k4n fn4-$a ;; esac; done
[ "$DRY" = 1 ] || log "RESULT fn4 $ITEM gaps a pass by arm (rankskew median us, cards c3 / 83 / 86 / 03) and arrival lag (cardlag mean us a pass / share last): $(python3 - $O "$done_arms" <<'PY' 2>&1 | cut -c1-2400
import json, sys
O, arms = sys.argv[1], sys.argv[2].split()
out = []
for p in ("c8", "c16"):
    for a in arms:
        try:
            r = json.load(open(f"{O}/fn4-{a}-{p}-rank.json")); l = json.load(open(f"{O}/fn4-{a}-{p}-lag.json"))
        except (OSError, ValueError):
            out.append(f"{p} {a}: no profile"); continue
        g = [c.get("gaps", 0) for c in r.get("card", [])]
        lag = [l["card"][str(c)]["lag_mean_us"] if str(c) in l["card"] else l["card"][c]["lag_mean_us"] for c in range(len(g))]
        ls = [l["card"][str(c)]["last_share"] if str(c) in l["card"] else l["card"][c]["last_share"] for c in range(len(g))]
        out.append(f"{p} {a}: gaps " + "/".join(f"{x:.0f}" for x in g) + f" (spread {max(g) - min(g):.0f}) | lag " +
                   "/".join(f"{x:.0f}" for x in lag) + " | last " + "/".join(f"{100 * x:.0f}%" for x in ls) +
                   f" | wall {l.get('wall_us', 0):.0f}")
print(" ; ".join(out))
PY
)"
log "RESULT fn4 $ITEM: done --$done_arms"
