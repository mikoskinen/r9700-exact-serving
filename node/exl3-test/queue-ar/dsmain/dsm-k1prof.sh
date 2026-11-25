#!/bin/bash
# dsm-k1prof.sh (10.11.2026, queue-ar/dsmain, ds41-k1 agent, lever K1 diagnosis) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-ha-k1prof.sh. WHY lever K1's folds measured ~0 in gzzz-attnfold (in process, ABBAAB, all three 0 vs 1:
# c1 +0.21 % +-2.27, c4 -0.22 % +-3.39, c8 +0.83 % +-1.13, c8 ms a step 44.23 -> 43.87) while the kernel bench saved
# ~3 us (kvfold 8.07 -> 5.04 us at M 1) and ~4.4 us (ofold 20.52 -> 16.14 us at M 1) a layer back to back. Suspects
# (coordinator, hypotheses, unverified): (i) the ~3.8 us dependent-launch gaps are not on the critical path of an unprofiled
# replay (the recorded pass has ~1390 dispatches, ~500 unordered; the gap reading came from stamped / profiled runs);
# (ii) the folded kernels run beside other work, so removing them does not shorten the chain; (iii) the fused kernels lose in
# the engine what they save in the bench. Three starts of the ds41-k1 image (XK1P_IMAGE), the t3 line (ARGATHER on at 4,
# F4 off) with the three folds declared (RADIANCE_DS41_KVFOLD / _OFOLD / _ONEFILL = 1), each A/B'd IN PROCESS by the switches
# (A = all three 0 = today's ops, B = all three 1; ds41_argather 4, gg_gemv 0, aql_acq 0 pinned; every change bumps the
# epoch, so each block plays fresh recordings):
#   P1 TRUTH (plain start, gated): c1 stepprobe essay,code 512, blocks A B B A A B (XK1P_ORDER) -> ms a decode step and
#      tok/s a block, B vs A pooled and per pair. Unprofiled, unstamped: the reference for the other two.
#   P2 STAMPS (RADIANCE_AQL_STAMP=1: every pass on card 0 gets per-dispatch start / end; nothing else changes): windows
#      A1 B1 B2 A2, each a stepprobe essay 256 (c1) -> k1prof.py stamps per pair (A1 / B1, A2 / B2): the trunk pass's wall,
#      busy, idle, dispatches (ordered), the chain (walk back from the last end through the dispatch that ended last before
#      each start: its run + gaps), the time between passes; per kernel class of the folds (norm_rope_rt KV 0 / 1,
#      kv_store, attention dec ORT 0 / 1, rope_rt, bf16 fill) count, device us, EXPOSED us (running alone), IDLE-BEFORE us
#      (card idle right before it: the dependent-launch gap it pays), ordered, on the chain. The stamped pass + between
#      time is set against P1's unstamped ms a step: how much stamping inflates.
#   P3 PROFILE (--profile-ops, RADIANCE_PROFILE_EVERY 16; synchronised steps): windows A then B, each a stepprobe essay,code
#      256 -> k1prof.py ops: the folds' ops a step, us a call, lane wait a call (rmsnorm_rope_roundtrip[_kv_store], kv_store,
#      attn_sparse_sink[_rope_roundtrip], rope_rows_roundtrip, fill n=1 bf16), device + waits a step. PROFILED, not a speed.
#   SUMMARY: the measured change (P1) against what the removed launches cost on card 0's chain in A (P2: their exposed +
#   idle-before), what the fused kernels add (P2 device / exposed B vs A, P3 us a call), and the stamped / unstamped step;
#   each conclusion labelled with the readings behind it: "gaps not on the path", "hidden by unordered overlap", "slower
#   fused kernels", or "inconclusive".
# ~20 min (three starts of ~4-5 min + ~6 min of windows). Exact by construction (K1); no output is compared here.
# NO cc_sim, no 2.9. No power / host / fan settings; production compose files read only (--project-directory).
# DRYRUN=1 bash dsm-k1prof.sh : prints every command, runs and writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-k1-tp4.yaml
O=$R/xdsm/k1prof; export STATSDIR=$O/stats
export IMAGE=${XK1P_IMAGE:-radiance-local:ds41-k1-1d1f782b-5c13e555-l}
ITEM=${XK1P_ITEM:-57-p4zyzzzzz-ds-ha-k1prof}; ORDER=${XK1P_ORDER:-ABBAAB}; PHASES=${XK1P_PHASES:-P1 P2 P3}
KP=$HD/k1prof.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
PIN='"ds41_argather": 4, "gg_gemv": 0, "aql_acq": 0'
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DS41_ARGATHER_MAX RADIANCE_DS41_SHEXP_LANE RADIANCE_DEBUG_MISS_LOG
  RADIANCE_DS41_KVFOLD RADIANCE_DS41_OFOLD RADIANCE_DS41_ONEFILL"
k1env() {   # 105's t3 arm + the three K1 folds declared (and on at start), every other switch unset (= dsm-attnfold.sh's)
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0 \
         RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
}
modelvars225() { export MODEL=$M225; HP29=19456; LIM=23; best_vars; export MNBT=4160 XARGS="$XARGS --live"; }
now() { python3 -c 'import time; print(time.clock_gettime_ns(time.CLOCK_MONOTONIC))'; }   # the stamps' clock
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
gate() {   # gate TAG LIMIT (dsm-attnfold.sh's): restart + warm-up + fastcheck, one restart when slow
  local tag=$1 lim=$2 ms ms1 st=fast
  restart $tag || { keeplog $tag-fail; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && return 0
  if awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}'; then
    keeplog $tag-try1
    ms1=$ms; restart $tag || { keeplog $tag-fail; return 1; }; ms=$(check $tag 2); st="fast-after-retry (first $ms1)"
    awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}' && { st="SLOW-after-retry (first $ms1)"; keeplog $tag-try2; }
  fi
  echo "$st $ms" > $O/guard-$tag.txt
  log "RESULT guard $tag: $st, 1-user ${ms} ms/step (limit $lim; bz dsd-d0 main image 15.59)"
  case $st in SLOW*) return 2 ;; esac
}
trap 'down' EXIT
trap 'exit 143' TERM INT
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
allk1() { echo "{\"ds41_kvfold\": $1, \"ds41_ofold\": $1, \"ds41_onefill\": $1, $PIN, \"epoch_bump\": 1}"; }
arm() {   # arm A|B: the K1 switches 0 / 1 (pinned others), a fresh epoch, one short request to record the passes
  case $1 in A) fn4set "$(allk1 0)" ;; B) fn4set "$(allk1 1)" ;; esac
  nudge
}
k1check() {   # k1check TAG ARM: the switches applied as asked (GET reports the plugin's value) -> 0
  [ "$DRY" = 1 ] && return 0
  local v=0; [ $2 = B ] && v=1
  fn4get | python3 -c 'import json,sys
s = json.loads(sys.stdin.read()); w = s["switches"]; v = int(sys.argv[1])
sys.exit(0 if all(w.get(k) == v for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) and w.get("ds41_argather") == 4 and not s.get("pending") else 1)' $v \
    || { log "RESULT xk1p $1: VIRHE the K1 switches are not at arm $2 ($(fn4get | cut -c1-300))"; return 1; }
}
k1note() {   # k1note TAG: the lever K1 note (every fold declared) -> 0
  [ "$DRY" = 1 ] && { echo "+ k1note $1: 'lever K1 folds' note with 40 of 40 layers, all switches 1"; return 0; }
  local KL; KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-500)
  log "RESULT xk1p $1 K1 note: ${KL:-none}"
  case "$KL" in *"in one launch at 0 of"*|*"rope+round trip at 0 ("*|*"not declared at"*|"") log "RESULT xk1p $1: VIRHE not every fold declared"; return 1 ;; esac
}
spjson() {   # spjson FILE: stepprobe's J lines -> "ms tps" means
  python3 -c 'import json,statistics,sys
r = [json.loads(l[2:]) for l in open(sys.argv[1]) if l.startswith("J ")]
print("%.3f %.2f %d" % (statistics.mean(x["ms_per_decode_step"] for x in r), statistics.mean(x["decode_tps"] for x in r), len(r)) if r else "nan nan 0")' "$1" 2>/dev/null || echo "nan nan 0"
}
# ---------------------------------------------------------------- P1: the truth (plain start)
p1() {
  local tag=k1p-plain i a f
  k1env; modelvars225; export AQLSTAMP= PROFEVERY= STATSDIR=$O/stats
  log "RESULT xk1p P1: start (plain: no stamps, no profile) IMAGE=$IMAGE $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
  gate $tag $LIM; local rc=$?
  [ $rc = 1 ] && { log "RESULT xk1p P1: VIRHE not healthy"; return 1; }
  [ $rc = 2 ] && log "RESULT xk1p P1: note -- slow after one restart (read P1 with that in mind)"
  k1note $tag || return 1
  run "rm -f $O/p1-blocks.txt"
  for i in $(seq 0 $((${#ORDER} - 1))); do
    stopfile && break
    a=${ORDER:$i:1}; f=$O/p1-b$i$a.txt
    arm $a; k1check $tag $a || return 1
    run "(cd $D && timeout 900 python3 stepprobe.py essay,code 512 > $f 2>&1)"
    [ "$DRY" = 1 ] || { echo "$i $a $(spjson $f)" >> $O/p1-blocks.txt; log "RESULT xk1p P1 block $i $a: ms a step / tok/s / prompts $(spjson $f)"; }
  done
  [ "$DRY" = 1 ] || log "RESULT xk1p P1 summary (c1, unprofiled, in process, $ORDER): $(python3 - $O/p1-blocks.txt <<'PY' 2>&1 | tee $O/p1-summary.txt | cut -c1-600
import math, statistics, sys
rows = [l.split() for l in open(sys.argv[1]) if l.strip()]
A = [float(r[2]) for r in rows if r[1] == "A" and r[2] != "nan"]; B = [float(r[2]) for r in rows if r[1] == "B" and r[2] != "nan"]
pr = []
for j in range(0, len(rows) - 1, 2):
    x, y = rows[j], rows[j + 1]
    if x[1] != y[1] and "nan" not in (x[2], y[2]):
        a_, b_ = (float(x[2]), float(y[2])) if x[1] == "A" else (float(y[2]), float(x[2]))
        pr.append(b_ / a_)
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
h = T95.get(len(pr) - 1, 1.96) * statistics.stdev(pr) / math.sqrt(len(pr)) if len(pr) > 1 else None
print("ms a step A %.3f -> B %.3f (%+.2f %%; pairs %s; t95 +- %s %%) | P1A %.4f P1B %.4f" % (statistics.mean(A), statistics.mean(B),
      100 * (statistics.mean(B) / statistics.mean(A) - 1), " ".join("%+.2f" % (100 * (p - 1)) for p in pr),
      "-" if h is None else "%.2f" % (100 * h), statistics.mean(A), statistics.mean(B)))
PY
)"
  run "docker logs $CN > $O/$tag-engine.log 2>&1"
  down
}
# ---------------------------------------------------------------- P2: AQL stamps (card 0)
p2() {
  local tag=k1p-stamp w a t0 t1 f ms
  k1env; modelvars225; export AQLSTAMP=1 PROFEVERY= STATSDIR=$O/stats-stamp
  mkd $STATSDIR
  run "find $STATSDIR -maxdepth 1 -name 'stamps.csv*' -delete; rm -f $O/p2-win.txt"
  log "RESULT xk1p P2: start RADIANCE_AQL_STAMP=1 (every played pass on card 0 stamped) IMAGE=$IMAGE $(desc)"
  restart $tag || { keeplog $tag-fail; log "RESULT xk1p P2: VIRHE not healthy -- no stamp windows"; export AQLSTAMP= STATSDIR=$O/stats; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] || log "RESULT xk1p P2 warm-up + 1-user fastcheck (STAMPED, not gated): $ms ms/step"
  k1note $tag || { down; export AQLSTAMP= STATSDIR=$O/stats; return 1; }
  for w in A1 B1 B2 A2; do
    stopfile && break
    a=${w:0:1}; f=$O/p2-$w.txt
    arm $a; k1check $tag $a || break
    run "(cd $D && timeout 600 python3 stepprobe.py essay 128 > /dev/null 2>&1)"   # settle (untimed)
    if [ "$DRY" = 1 ]; then t0=T0; else t0=$(now); fi
    run "(cd $D && timeout 600 python3 stepprobe.py essay 256 > $f 2>&1)"
    if [ "$DRY" = 1 ]; then t1=T1; else t1=$(now); fi
    run "echo '$w $t0 $t1' >> $O/p2-win.txt"
    [ "$DRY" = 1 ] || log "RESULT xk1p P2 window $w (stamped c1): ms a step / tok/s / prompts $(spjson $f)"
  done
  keeplog $tag
  down   # the stamp writer flushes at exit
  f=$STATSDIR/stamps.csv
  if [ "$DRY" = 1 ]; then echo "+ k1prof.py stamps $f --win A:A1 --win B:B1 / --win A:A2 --win B:B2 --json $O/p2-pair1.json / p2-pair2.json"
  else
    [ -s $f ] || log "RESULT xk1p P2: VIRHE no stamp file $f"
    for pr in "1 A1 B1" "2 A2 B2"; do
      set -- $pr
      local wa wb; wa=$(awk -v w=$2 '$1 == w {print $2 ":" $3}' $O/p2-win.txt); wb=$(awk -v w=$3 '$1 == w {print $2 ":" $3}' $O/p2-win.txt)
      [ -n "$wa" ] && [ -n "$wb" ] || { log "RESULT xk1p P2 pair $1: no windows"; continue; }
      timeout 1800 python3 -I $KP stamps $f --win A:$wa --win B:$wb --label p2-pair$1 --json $O/p2-pair$1.json > $O/p2-pair$1.txt 2>&1
      while IFS= read -r l; do log "RESULT xk1p P2 pair $1 ${l:0:600}"; done < <(grep '^K1PROF p2' $O/p2-pair$1.txt | grep -v '^K1PROF {' | sed 's/^K1PROF p2-pair[12] stamps //')
    done
    run "gzip -f $STATSDIR/stamps.csv* 2>/dev/null || true"
  fi
  export AQLSTAMP= STATSDIR=$O/stats
}
# ---------------------------------------------------------------- P3: --profile-ops (PROFILED)
p3() {
  local tag=k1p-prof a l0 l1 f E=$O/p3-engine.log ms
  k1env; modelvars225; export AQLSTAMP= PROFEVERY=16 XARGS="$XARGS --profile-ops" STATSDIR=$O/stats-prof
  mkd $STATSDIR
  run "rm -f $O/p3-win.txt"
  log "RESULT xk1p P3: start --profile-ops RADIANCE_PROFILE_EVERY=16 (PROFILED: not a speed) IMAGE=$IMAGE $(desc)"
  restart $tag || { keeplog $tag-fail; log "RESULT xk1p P3: VIRHE not healthy -- no profile"; export PROFEVERY= STATSDIR=$O/stats; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] || log "RESULT xk1p P3 warm-up + 1-user fastcheck (PROFILED, not gated): $ms ms/step"
  k1note $tag || { down; export PROFEVERY= STATSDIR=$O/stats; return 1; }
  for a in A B; do
    stopfile && break
    f=$O/p3-$a.txt
    arm $a; k1check $tag $a || break
    run "(cd $D && timeout 600 python3 stepprobe.py essay 64 > /dev/null 2>&1)"   # past the switch's first recordings
    if [ "$DRY" = 1 ]; then l0=L0; else l0=$(docker logs $CN 2>&1 | wc -l); fi
    run "(cd $D && timeout 900 python3 stepprobe.py essay,code 256 > $f 2>&1)"
    run "sleep 2"
    if [ "$DRY" = 1 ]; then l1=L1; else l1=$(docker logs $CN 2>&1 | wc -l); fi
    run "echo '$a $l0 $l1' >> $O/p3-win.txt"
    [ "$DRY" = 1 ] || log "RESULT xk1p P3 window $a (PROFILED c1): ms a step / tok/s / prompts $(spjson $f)"
  done
  run "docker logs $CN > $E 2>&1"
  down
  if [ "$DRY" = 1 ]; then echo "+ k1prof.py ops $E --win A:LA0:LA1 --win B:LB0:LB1 --json $O/p3-ops.json"
  else
    local wa wb; wa=$(awk '$1 == "A" {print $2 ":" $3}' $O/p3-win.txt); wb=$(awk '$1 == "B" {print $2 ":" $3}' $O/p3-win.txt)
    if [ -n "$wa" ] && [ -n "$wb" ]; then
      timeout 1800 python3 -I $KP ops $E --win A:$wa --win B:$wb --label p3 --json $O/p3-ops.json > $O/p3-ops.txt 2>&1
      while IFS= read -r l; do log "RESULT xk1p P3 ${l:0:900}"; done < <(grep '^K1PROF p3 ops' $O/p3-ops.txt | sed 's/^K1PROF p3 ops //')
    else log "RESULT xk1p P3: no windows"; fi
    run "gzip -f $E"
  fi
  export PROFEVERY= STATSDIR=$O/stats
}
# ---------------------------------------------------------------- summary
summary() {
  [ "$DRY" = 1 ] && { echo "+ RESULT xk1p summary: P1 change vs the removed launches' chain cost (P2 A exposed + idle-before), the fused kernels' added cost (P2 / P3), stamped vs unstamped step -> labelled conclusions"; return 0; }
  log "RESULT xk1p summary: $(python3 - $O <<'PY' 2>&1 | cut -c1-2500
import json, os, re, sys
O = sys.argv[1]
def jl(f):
    try: return json.load(open(os.path.join(O, f)))
    except (OSError, ValueError): return None
out = []
p1 = None
try:
    t = open(os.path.join(O, "p1-summary.txt")).read()
    m = re.search(r"P1A ([\d.]+) P1B ([\d.]+)", t)
    if m: p1 = (float(m.group(1)), float(m.group(2)))
    out.append("P1 (unprofiled truth, c1): " + t.split(" | P1A")[0].strip())
except OSError:
    out.append("P1: none")
pairs = [x for x in (jl("p2-pair1.json"), jl("p2-pair2.json")) if x and "error" not in x.get("A", {}) and "error" not in x.get("B", {})]
def g(x, c, k): return ((x["per"].get(c) or {}).get(k) or 0.0)
if pairs:
    rem, add, dwall, dstep, nomgap, idle, expo, dev, stA, stB = [], [], [], [], [], [], [], [], [], []
    for p in pairs:
        A, B = p["A"], p["B"]
        # the launches a fold removes, from A: kv_store (fold c), the o rope_rt (fold d: A's rope_rt minus B's), the bf16 fills
        cut = {"kvstore": g(A, "kvstore", "n") - g(B, "kvstore", "n"), "ropert": g(A, "ropert", "n") - g(B, "ropert", "n"),
               "fill1": g(A, "fill1", "n") - g(B, "fill1", "n")}
        r_e = sum(g(A, c, "exposed") * n / max(1e-9, g(A, c, "n")) for c, n in cut.items())
        r_i = sum(g(A, c, "idle_before") * n / max(1e-9, g(A, c, "n")) for c, n in cut.items())
        r_d = sum(g(A, c, "dev") * n / max(1e-9, g(A, c, "n")) for c, n in cut.items())
        expo.append(r_e); idle.append(r_i); dev.append(r_d); rem.append(r_e + r_i); nomgap.append(3.8 * sum(cut.values()))
        # what the fused kernels add: B's fused classes' device time against A's kernels they replace
        a_norm40 = g(A, "kvnorm", "dev") - g(B, "kvnorm", "dev")       # A's window-key norms (B keeps the index key's)
        add.append(((g(B, "kvfold", "dev") - a_norm40) if g(B, "kvfold", "n") > 0 else 0.0) +
                   ((g(B, "attnfold", "dev") - g(A, "attn", "dev")) if g(B, "attnfold", "n") > 0 else 0.0))
        dwall.append(B["wall_us"] - A["wall_us"]); dstep.append(B["step_us"] - A["step_us"]); stA.append(A["step_us"]); stB.append(B["step_us"])
        out.append("P2 pair: removed launches %s a pass, device %.0f us, exposed %.0f + idle-before %.0f = %.0f us on card 0 "
                   "(nominal 3.8 us gaps x count = %.0f us); fused kernels add %+.0f us device a pass; stamped wall %+.0f us, step %+.0f us "
                   "(A step %.0f us)" % ({k: round(v, 1) for k, v in cut.items()}, r_d, r_e, r_i, r_e + r_i, nomgap[-1], add[-1],
                                         dwall[-1], dstep[-1], A["step_us"]))
    mean = lambda v: sum(v) / len(v)
    pred = mean(rem) - max(0.0, mean(add))
    if p1:
        out.append("stamping: stamped step A %.0f us vs unprofiled %.0f us (x%.3f)" % (mean(stA), 1000 * p1[0], mean(stA) / (1000 * p1[0])))
        out.append("P1 measured %+.0f us a step vs P2 prediction -(%.0f) us (removed exposed + idle %.0f, fused add %+.0f) vs stamped %+.0f us"
                   % (1000 * (p1[1] - p1[0]), pred, mean(rem), mean(add), mean(dstep)))
    labels = []
    if mean(idle) < 0.3 * mean(nomgap):
        labels.append("GAPS NOT ON THE PATH: the removed launches' idle-before is %.0f us a pass of %.0f nominal (3.8 us x launches)" % (mean(idle), mean(nomgap)))
    if mean(dev) > 0 and mean(expo) < 0.5 * mean(dev):
        labels.append("HIDDEN BY UNORDERED OVERLAP: the removed kernels run alone for %.0f of their %.0f us device a pass" % (mean(expo), mean(dev)))
    if mean(add) > 0.3 * mean(rem):
        labels.append("SLOWER FUSED KERNELS: the fused kernels add %+.0f us device a pass against %.0f us the removed launches cost on the chain" % (mean(add), mean(rem)))
    if p1 and abs(1000 * (p1[1] - p1[0])) < 0.5 * pred:
        labels.append("UNEXPLAINED: P1 moved %+.0f us, under half the stamped prediction (-%.0f us)" % (1000 * (p1[1] - p1[0]), pred))
    out.append("labels: " + (" ; ".join(labels) if labels else "inconclusive (no reading crosses its threshold)"))
else:
    out.append("P2: no stamp pairs")
p3 = jl("p3-ops.json")
if p3 and "A" in p3 and "B" in p3:
    for rk in ("0",):
        a, b = p3["A"].get(rk) or {}, p3["B"].get(rk) or {}
        if not a or not b or "ops" not in a or "ops" not in b: continue
        u = lambda x, k: (x["ops"].get(k) or {}).get("us_step", 0.0)
        sepc = u(a, "rmsnorm_rope_roundtrip") + u(a, "kv_store"); fc = u(b, "rmsnorm_rope_roundtrip") + u(b, "rmsnorm_rope_roundtrip_kv_store")
        sepd = u(a, "attn_sparse_sink") + u(a, "rope_rows_roundtrip"); fd = u(b, "attn_sparse_sink") + u(b, "attn_sparse_sink_rope_roundtrip") + u(b, "rope_rows_roundtrip")
        out.append("P3 rank 0 (PROFILED us a step): fold c ops %.0f -> %.0f, fold d ops %.0f -> %.0f, fills %.0f -> %.0f; device %.2f -> %.2f ms + waits %.2f -> %.2f ms a step"
                   % (sepc, fc, sepd, fd, u(a, "fill(n=1,bf16)"), u(b, "fill(n=1,bf16)"), a["device_ms_step"], b["device_ms_step"], a["wait_ms_step"], b["wait_ms_step"]))
print(" | ".join(out))
PY
)"
}
# ---------------------------------------------------------------- main
log "RESULT xk1p $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) phases [$PHASES], P1 order $ORDER, yaml $Y | gzzz-attnfold all-three: c1 +0.21 % +-2.27, c4 -0.22 % +-3.39, c8 +0.83 % +-1.13 (ms a step 44.23 -> 43.87)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xk1p: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xk1p: VIRHE image $IMAGE not on gpu02"; exit 1; }
  [ -e $Y ] && grep -q '^ *RADIANCE_DS41_KVFOLD:' $Y && grep -q '^ *RADIANCE_AQL_STAMP:' $Y && grep -q '^ *RADIANCE_PROFILE_EVERY:' $Y || { log "RESULT xk1p: VIRHE $Y missing or does not pass the K1 / stamp / profile variables"; exit 1; }
  for f in $KP $HD/ds_opprof.py $D/stepprobe.py; do [ -e $f ] || { log "RESULT xk1p: VIRHE $f missing"; exit 1; }; done
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xk1p: VIRHE $M225 missing or not verified"; exit 1; }
  [ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -q RADIANCE_DS41_KVFOLD /opt/radiance/share/radiance/architectures/deepseek41.so && grep -q ds41_onefill /opt/radiance/bin/radiance && echo yes' 2>/dev/null)" = yes ] \
    || { log "RESULT xk1p: VIRHE $IMAGE is not the K1 image"; exit 1; }
fi
mkd $O $O/guard $O/stats
evict
for ph in $PHASES; do
  stopfile && break
  case $ph in P1) p1 ;; P2) p2 ;; P3) p3 ;; *) log "RESULT xk1p: note -- unknown phase $ph" ;; esac
done
summary
unset $SWITCHES
export AQLSTAMP= PROFEVERY=
down
log "RESULT xk1p $ITEM: done -- $O"
