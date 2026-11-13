#!/bin/bash
# 117-ds-gaps.sh (queue item, 9.11.2026, queue-ar; gpu02, all 4 cards, TP4, MML 524288; every radiance container DOWN; ~25 min)
# -- where DeepSeek-V4.1 Flash's decode step idles: 115's gap profile on the DS line. One arm, a fresh start with EVERY
# played pass stamped on card 0 (RADIANCE_AQL_STAMP=1; the ds41 binary stamps card 0 only), the warm-up and a 1-user
# fastcheck (logged, not gated: it is stamped), then a 12 s mb c4 and a 12 s mb c8 window after 8 s of warm-up each, a
# /stats + /metrics snapshot around each window, down, and the analysis on the complete stamp file.
# LINE: 105's 225:t3 (x79-225pa-t3) -- EXL3 2.25 bpw, image radiance-local:ds41-exl3-117-641197d8-fa0e9580-tuned3l-hd4a11cab
# (X117_IMAGE), host pool 19456, lib-29's best_vars + MNBT 4160 + --live, mxfp4 index keys, dual heat 0.7 + decode pin 0.8
# + moves 16, arena levels (LEVELS 1, LEVEL_MAX 1024, STAGE_SPAN 0), RADIANCE_DS41_ARGATHER=on, ROUTER_FUSE / MIX_FUSE /
# QLDS / AR_PCN 0 by name, latent bf16 (RADIANCE_DS41_LATENT unset); compose file ds41-shp-tp4.yaml (79's).
# PER WINDOW (c4, c8):
#   tools/gapcat.py (f4f1badc): inside the trunk pass every idle stretch by class (stretch / queue / chained / barrier /
#     unordered) in us a pass, the gap-length histogram, the costliest prev > next kernel pairs, per pass kind the wall /
#     busy / gaps and the dispatch counts (barrier bit = dependent launches, chained), the idle time between passes
#   tools/gapsize.py: the same-queue dependent-launch gap binned by the previous and by the next kernel's duration
#     (does it grow with the kernels around it?); FN c16 under the same stamps (115, k4nG / rmnG): median 4.28 / 4.12 us,
#     flat from <5 us to >100 us kernels, 0 % under 2 us
#   the host link: /stats deltas -- steps (decode / mixed / prefill), ms a decode step, routed hit (rank 0), expert
#     stream MB a step (all ranks, as /stats sums them) and GB/s a card while stepping, mover h2d MB a step; the card's
#     share of 111a's bwcc "h2d all" GB/s when that file exists; /metrics via msnap.py
#   beside them, unstamped, the same line's 105 rows when they exist (x79-225pa-t3 mb c4 / c8 and its per-phase /stats):
#     the stamping inflation of the step on DS (FN: +1.2 ms on 37.8, all cards stamped; card 0 alone none)
# KEEPAWAKE: the window-wide gpu_metrics poller (runs/keepawake.pid) is checked alive at start and end; if it is not
# running, this item starts its own (as 105: tools/keepawake.py, X117_KA_MS or tools/pick103.sh on 103's table, else 5 ms;
# pid runs/keepawake-117.pid) and stops it at the end and in the EXIT trap.
# TEMPERATURE: read-only watcher every 1 s (junction >= 100 C or mem >= 96 C on any card for 10 s -> the remaining windows
# are skipped, VIRHE). Power is not touched (caps logged; 210 W expected). Host tuning and the window poller as they are.
# Context (not re-run): 76 ARGATHER 1390 -> 1350 dispatches a pass, ordered 935 -> 894, stamped wall 35.64 -> 33.22 ms,
# unstamped guard 19.96 -> 19.85; 77 router + mix fusions (-43 / -80 ordered) guard 18.91 / 18.89, c4 99.0 -> 97.6;
# 75 gapbench ~1.5 us an unstamped ordered dispatch (tiny kernels); 94 FN NO_OVERLAP ~2.9 / 3.3 / 5.1 us an added barrier.
# DRYRUN=1 bash 117-ds-gaps.sh : prints every command, runs and writes nothing.
set -u
H117="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$H117/lib-29.sh" "$H117/../lib-29.sh" "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
for YS in "$H117/ds41-shp-tp4.yaml" "$H117/../ds41-shp-tp4.yaml" "$HOME/gpu/ds41/exl3-test/queue-ar/ds41-shp-tp4.yaml"; do [ -e "$YS" ] && break; done
[ -e "$YS" ] || { echo "ds41-shp-tp4.yaml not found"; exit 1; }
Y=$YS
TL=; for f in "$H117/tools" "$H117/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/gapcat.py" ] && [ -e "$f/gapsize.py" ] && [ -e "$f/msnap.py" ] && [ -e "$f/keepawake.py" ] && { TL=$f; break; }; done
O=$R/x117g; ITEM=117-ds-gaps; TAG=x117g-225-t3
IMG=${X117_IMAGE:-radiance-local:ds41-exl3-117-641197d8-fa0e9580-tuned3l-hd4a11cab}
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WINS=${X117_WINS:-"c4 c8"}
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
REF105=$R/x79/x79-225pa-t3          # 105's 225:t3 arm: mb files beside it, its /stats phases inside
BWCC=$R/fn4/111-bwcc.txt
now() { python3 -c 'import time; print(time.clock_gettime_ns(time.CLOCK_MONOTONIC))'; }   # the stamps' clock
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
caps() {   # every card's power1_cap / power1_average (W) and hotspot temperature by PCI slot (read-only)
  local d h out=""
  for d in /sys/class/drm/card*/device; do
    [ -e $d/unique_id ] || continue
    h=$(ls -d $d/hwmon/hwmon* 2>/dev/null | head -1); [ -n "$h" ] || continue
    out="$out $(basename $(readlink -f $d)):cap=$(( $(cat $h/power1_cap 2>/dev/null || echo 0) / 1000000 ))W,avg=$(( $(cat $h/power1_average 2>/dev/null || echo 0) / 1000000 ))W,hot=$(( $(cat $h/temp2_input 2>/dev/null || echo 0) / 1000 ))C"
  done
  echo "$out"
}
tguard() {   # tguard: junction >= 100 C or mem >= 96 C on any card for 10 s -> $O/$TAG.abort (read-only: no cap is touched)
  local d h j m n=0 hot
  while [ ! -e $O/$TAG.tstop ]; do
    hot=""
    for d in /sys/class/drm/card*/device; do
      [ -e $d/unique_id ] || continue
      h=$(ls -d $d/hwmon/hwmon* 2>/dev/null | head -1); [ -n "$h" ] || continue
      j=$(( $(cat $h/temp2_input 2>/dev/null || echo 0) / 1000 )); m=$(( $(cat $h/temp3_input 2>/dev/null || echo 0) / 1000 ))
      { [ $j -ge ${X117_TJ_MAX:-100} ] || [ $m -ge ${X117_TMEM_MAX:-96} ]; } && hot="$hot $(basename $(readlink -f $d)):j$j/m$m"
    done
    if [ -n "$hot" ]; then n=$((n + 1)); else n=0; fi
    if [ $n -ge 10 ]; then
      : > $O/$TAG.abort
      log "RESULT x117g $TAG: VIRHE temperature guard -- junction >= ${X117_TJ_MAX:-100} C or mem >= ${X117_TMEM_MAX:-96} C for 10 s ($hot) -> remaining windows skipped"
      return 0
    fi
    sleep 1
  done
}
kaalive() {   # kaalive PIDFILE: 0 when its pid runs keepawake.py
  local p; [ -s "$1" ] || return 1; p=$(cat "$1"); kill -0 "$p" 2>/dev/null && tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q keepawake.py
}
kastate() { local p; for p in $R/keepawake.pid $R/keepawake-117.pid; do kaalive $p && { echo "pid $(cat $p) ($(basename $p)): $(tr '\0' ' ' < /proc/$(cat $p)/cmdline | grep -oE -- '--interval-ms [0-9.]+|--node [a-z_0-9]+' | tr '\n' ' ')"; return 0; }; done; echo "none alive"; return 1; }
OWN=0
stopown() { [ "$OWN" = 1 ] && kaalive $R/keepawake-117.pid && { kill $(cat $R/keepawake-117.pid) 2>/dev/null; log "RESULT x117g keepawake: own poller stopped (pid $(cat $R/keepawake-117.pid))"; }; OWN=0; }
snapst() {   # snapst WINDOW a|b: /stats and /metrics now
  run "curl -s -m 15 $URL/stats > $O/$TAG-$1-st$2.json 2>/dev/null; timeout 30 python3 -I $TL/msnap.py snap $URL $O/$TAG-$1-ms$2.json > /dev/null 2>&1"
}
mbw() {   # mbw WINDOW: a short mb window (X117_DUR s after X117_WARM s)
  run "(cd $D && timeout 600 python3 mb.py --tag $TAG --phases $1 --dur ${X117_DUR:-12} --warm ${X117_WARM:-8} > $O/$TAG-$1.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT x117g $TAG mb $1 (${X117_DUR:-12} s, stamped): $(grep '^MB ' $O/$TAG-$1.txt 2>/dev/null | cut -c4- | cut -c1-400)"
}
linkst() {   # linkst WINDOW: the /stats delta of the window, and 105's unstamped phase of the same name beside it
  [ "$DRY" = 1 ] && { echo "+ RESULT x117g $TAG $1 host link: /stats delta ($O/$TAG-$1-st{a,b}.json) + bwcc h2d all ($BWCC) + 105's $REF105/stats-phases.txt $1"; return 0; }
  log "RESULT x117g $TAG $1 host link + steps: $(python3 - $O/$TAG-$1-sta.json $O/$TAG-$1-stb.json $BWCC $REF105/stats-phases.txt $1 <<'PY' 2>&1 | cut -c1-1400
import json, re, sys
fa, fb, bw, ref, win = sys.argv[1:6]
try:
    a, b = json.load(open(fa)), json.load(open(fb))
except Exception as e:
    print("no /stats snapshots (%s)" % e); sys.exit(0)
ea, eb = a.get("experts", {}), b.get("experts", {}); la, lb = a.get("link", {}), b.get("link", {})
world = lb.get("world") or 4
ds = b.get("steps", 0) - a.get("steps", 0); dd = b.get("steps_decode", 0) - a.get("steps_decode", 0)
dpm = (b.get("steps_prefill", 0) + b.get("steps_mixed", 0)) - (a.get("steps_prefill", 0) + a.get("steps_mixed", 0))
dns = b.get("step_ns", 0) - a.get("step_ns", 0); dnd = b.get("step_ns_decode", 0) - a.get("step_ns_decode", 0)
dstr = lb.get("stream_bytes", 0) - la.get("stream_bytes", 0); dh2d = lb.get("h2d_bytes", 0) - la.get("h2d_bytes", 0)
dr = eb.get("routed", 0) - ea.get("routed", 0); drr = eb.get("routed_res", 0) - ea.get("routed_res", 0)
out = []
if ds <= 0:
    print("no steps in the window"); sys.exit(0)
gbs = dstr / world / dns if dns else 0.0      # bytes / ns = GB/s, a card
out.append("%d steps (%d decode, %d prefill/mixed), %.2f ms a step, decode %.2f ms" % (ds, dd, dpm, dns / ds / 1e6, (dnd / dd / 1e6) if dd else float("nan")))
out.append("routed hit rank 0 %.2f %%" % (100.0 * drr / dr) if dr else "routed n/a")
out.append("expert stream %.1f MB a step (all %d ranks) = %.1f MB a card, %.2f GB/s a card while stepping" % (dstr / ds / 1e6, world, dstr / ds / world / 1e6, gbs))
out.append("mover h2d %.1f MB a step (all ranks)" % (dh2d / ds / 1e6))
try:
    for l in open(bw):
        if l.startswith("BW h2d all |"):
            v = [float(x) for x in re.findall(r"h2d \S+ ([0-9.]+)", l.split("|")[1])]
            if v:
                m = sum(v) / len(v)
                out.append("link share %.0f %% of bwcc h2d-all %.1f GB/s a card (min %.1f)" % (100.0 * gbs / m, m, min(v)))
            break
except OSError:
    out.append("no 111a bwcc file")
try:
    for l in open(ref):
        if l.startswith("PH "):
            d = json.loads(l[3:])
            if d.get("phase") == win:
                out.append("105 unstamped %s: %s ms a step, decode %s ms, stream %s MB a step, h2d %s MB a step, hit %s %%" % (
                    win, d.get("ms_step"), d.get("ms_decode_step"), d.get("stream_MB_step"), d.get("h2d_MB_step"), d.get("hit_pct_rank0")))
except OSError:
    out.append("no 105 stats phases")
print("; ".join(out))
PY
)"
  log "RESULT x117g $TAG $1 engine counters: $(timeout 30 python3 -I $TL/msnap.py delta $O/$TAG-$1-msa.json $O/$TAG-$1-msb.json 2>&1 | cut -c1-600)"
}
gaps() {   # gaps WINDOW T0 T1: gapcat + gapsize on card 0's stamps
  local f=$STATSDIR/stamps.csv
  if [ "$DRY" = 1 ]; then echo "+ gapcat.py $f --from $2 --to $3 --label $TAG-$1 --json $O/$TAG-$1-gap0.json; gapsize.py $f --from $2 --to $3 --json $O/$TAG-$1-gapsize.json"; return 0; fi
  [ -s $f ] || { log "RESULT x117g $TAG $1: VIRHE no stamp file $f"; return 1; }
  timeout 1800 python3 -I $TL/gapcat.py $f --from $2 --to $3 --label $TAG-$1-card0 --json $O/$TAG-$1-gap0.json > $O/$TAG-$1-gap0.txt 2>&1
  if ! [ -s $O/$TAG-$1-gap0.json ]; then   # e.g. the window selected no pass: say what the file holds
    log "RESULT x117g $TAG $1: VIRHE gapcat wrote no analysis ($(tail -2 $O/$TAG-$1-gap0.txt | tr '\n' ' ' | cut -c1-200)); stamps start_ns $(awk -F, 'NR==2{print $7}' $f) .. $(tail -1 $f | cut -d, -f7), window $2 .. $3"
  fi
  log "RESULT x117g $TAG $1 card 0 gaps: $(tr -s ' ' < $O/$TAG-$1-gap0.txt | tr '\n' '|' | cut -c1-2400)"
  timeout 1800 python3 -I $TL/gapsize.py $f --from $2 --to $3 --label $TAG-$1 --json $O/$TAG-$1-gapsize.json > $O/$TAG-$1-gapsize.txt 2>&1
  log "RESULT x117g $TAG $1 dependent-launch gap vs kernel size: $(cut -c1-1500 $O/$TAG-$1-gapsize.txt | tr '\n' ' ')"
}
log "RESULT x117g $ITEM: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) MODEL=$M225 windows [$WINS] | compose $Y | power stays at the cards' caps (210 W expected), host tuning and the window poller as they are"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT x117g: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT x117g: VIRHE image $IMG missing"; exit 1; }
  [ -n "$TL" ] || { log "RESULT x117g: VIRHE tools missing (gapcat.py / gapsize.py / msnap.py / keepawake.py)"; exit 1; }
  grep -q RADIANCE_DS41_ARGATHER $Y && grep -q RADIANCE_AQL_STAMP $Y || { log "RESULT x117g: VIRHE $Y lacks RADIANCE_DS41_ARGATHER / RADIANCE_AQL_STAMP"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT x117g: VIRHE $MODELDIR/$M225 missing or not verified"; exit 1; }
  mkdir -p $O
  if st=$(kastate); then
    log "RESULT x117g keepawake at start: window-wide poller alive -- $st"
  else
    ms=${X117_KA_MS:-$( [ -x $TL/pick103.sh ] && $TL/pick103.sh $R/fn4/103-table.txt 2>/dev/null || echo 5)}
    mkdir -p $HOME/gpu/watch
    setsid nohup python3 -I $TL/keepawake.py --interval-ms $ms --node gpu_metrics --cards all --log $HOME/gpu/watch/keepawake-117.log > /dev/null 2>&1 < /dev/null &
    echo $! > $R/keepawake-117.pid; OWN=1; sleep 2
    if kaalive $R/keepawake-117.pid; then log "RESULT x117g keepawake at start: no window-wide poller -> own poller started at $ms ms (pid $(cat $R/keepawake-117.pid), log ~/gpu/watch/keepawake-117.log)"
    else log "RESULT x117g: VIRHE the own keepawake did not start -- not running without it"; exit 1; fi
  fi
  trap 'touch $O/$TAG.tstop 2>/dev/null; down; stopown' EXIT
  trap 'exit 143' TERM INT
else
  echo "+ docker ps: no radiance container; image / tools / yaml / model checks; keepawake: $R/keepawake.pid alive, else own poller (tools/keepawake.py, pick103.sh) -> $R/keepawake-117.pid; trap: down + stop own poller"
fi
# ---- the arm: 105's 225:t3 environment
unset RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT \
      RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN \
      RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN \
      RADIANCE_DS41_ARGATHER RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ
export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
       RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
export IMAGE=$IMG MODEL=$M225
HP29=19456; best_vars; export MNBT=4160 XARGS="$XARGS --live"
export PROFEVERY= PROFSYNC= AQLSTAMP=1 STATSDIR=$O/stats-117
mkd $STATSDIR
run "find $STATSDIR -maxdepth 1 -name 'stamps.csv*' -delete; find $O -maxdepth 1 -name '$TAG-win.txt' -delete; find $O -maxdepth 1 -name '$TAG.abort' -delete; find $O -maxdepth 1 -name '$TAG.tstop' -delete"
TGPID=""
if [ "$DRY" = 1 ]; then echo "+ tguard &"; else tguard & TGPID=$!; fi
log "RESULT x117g $TAG: start IMAGE=$IMAGE MODEL=$MODEL $(desc) RADIANCE_AQL_STAMP=1 (every played pass, card 0) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ') | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
if ! restart $TAG; then
  keeplog $TAG-fail; down; run "touch $O/$TAG.tstop"
  log "RESULT x117g $ITEM: VIRHE not healthy -- done without windows"; exit 1
fi
if [ "$DRY" != 1 ]; then
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  log "RESULT x117g $TAG modes: $(echo "$MLF" | cut -c1-500) | $(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-200) | $(docker logs $CN 2>&1 | grep -m1 -E 'RADIANCE_AQL_STAMP|stamps' | cut -c1-200)"
  docker logs $CN 2>&1 | grep -q 'the MoE gather and its all-reduce as one kernel' || { keeplog $TAG-nomode; log "RESULT x117g $TAG: VIRHE no RADIANCE_DS41_ARGATHER note in the serve log -> stopped"; exit 1; }
  case "$MLF" in *"latent nvfp4"*) keeplog $TAG-nomode; log "RESULT x117g $TAG: VIRHE the indexer note says latent nvfp4 (t3 is bf16) -> stopped"; exit 1 ;; esac
fi
ms=$(check $TAG 1)
[ "$DRY" = 1 ] || log "RESULT x117g $TAG warm-up + 1-user fastcheck (STAMPED, not gated): ${ms} ms/step (105 / 76 unstamped guards for comparison: $(cat $R/x79/guard-x79-225pa-t3.txt 2>/dev/null || echo 'no 105 guard file'))"
if [ "$DRY" != 1 ] && [ -s $WARM_REF_225 ]; then
  log "RESULT x117g $TAG warm-up answers vs stock ($(basename $WARM_REF_225)): $(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$TAG-warm-q.txt $WARM_REF_225 2>&1 | cut -c1-300)"
fi
for p in $WINS; do
  [ -e $O/$TAG.abort ] && break
  stopfile && break
  snapst $p a
  if [ "$DRY" = 1 ]; then t0=T0; else t0=$(now); fi
  mbw $p
  if [ "$DRY" = 1 ]; then t1=T1; else t1=$(now); fi
  snapst $p b
  run "echo '$p $t0 $t1' >> $O/$TAG-win.txt"
  linkst $p
  [ "$DRY" = 1 ] || [ ! -s $REF105-$p.txt ] || log "RESULT x117g $TAG $p beside 105's unstamped x79-225pa-t3 $p: $(grep '^MB ' $REF105-$p.txt | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[3:]); print("agg_dec_tps %s, ms_decode_step %s" % (d.get("agg_dec_tps"), d.get("ms_decode_step")))' 2>&1 | cut -c1-200) vs stamped $(grep '^MB ' $O/$TAG-$p.txt 2>/dev/null | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[3:]); print("agg_dec_tps %s, ms_decode_step %s" % (d.get("agg_dec_tps"), d.get("ms_decode_step")))' 2>&1 | cut -c1-200)"
done
keeplog $TAG
[ "$DRY" = 1 ] || log "RESULT x117g $TAG stamping: $(grep -m3 -iE 'RADIANCE_AQL_STAMP|stamp' $O/engine-$TAG.log | sed 's/^[A-Z] //' | tr '\n' '|' | cut -c1-400)"
down   # the stamp writer flushes at exit
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
[ "$DRY" = 1 ] || log "RESULT x117g keepawake at end: $(kastate)"
stopown
# ---- analysis after down: the stamp file is complete and the cards idle
if [ "$DRY" = 1 ]; then for p in $WINS; do gaps $p T0 T1; done
else while read -r p t0 t1; do gaps $p $t0 $t1; done < $O/$TAG-win.txt; fi
run "du -sh $STATSDIR > $O/$TAG-stampsize.txt 2>&1; gzip -f $STATSDIR/stamps.csv* 2>/dev/null || true"
[ "$DRY" = 1 ] || log "RESULT x117g $ITEM summary, card 0 trunk pass (us a pass: wall / gaps; gap classes; dependent-launch gap): $(python3 - $O $TAG "$WINS" <<'PY' 2>&1 | cut -c1-2400
import json, sys
O, tag, wins = sys.argv[1], sys.argv[2], sys.argv[3].split()
out = []
for p in wins:
    try:
        g = json.load(open(f"{O}/{tag}-{p}-gap0.json"))
    except (OSError, ValueError):
        out.append(f"{p}: no analysis"); continue
    k = max(g["kinds"], key=lambda x: int(x))
    d = g["kinds"][k]; t = g.get("trunk", {}).get("cats_us", {})
    s = f"{p}: wall {d['wall_us']:.0f} gaps {d['gaps_us']:.0f} ({d['dispatches']:.0f} dispatches, {d['ordered']:.0f} barrier, {d['chained']:.0f} chained) | " + \
        " ".join(f"{c} {v:.0f}" for c, v in sorted(t.items(), key=lambda kv: -kv[1])) + \
        " | between passes: " + " ".join(f"{x} {v:.0f}" for x, v in sorted(g.get("between_us", {}).items()))
    try:
        z = json.load(open(f"{O}/{tag}-{p}-gapsize.json"))
        s += f" | dependent-launch gap median {z.get('median_gap_us')} us, {z.get('ordered_gaps_a_pass')} a pass"
    except (OSError, ValueError):
        pass
    out.append(s)
print(" ; ".join(out) + " | FN c16 (115, same stamps): dependent-launch gap 4.28 (k4nG) / 4.12 (rmnG) us")
PY
)"
log "RESULT x117g $ITEM: done"
