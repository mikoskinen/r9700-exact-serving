#!/bin/bash
# dsm-serve.sh (10.11.2026, queue-ar/dsmain, ds-main-port agent) -- NOT a queue item; the body of queue item
# 57-p4zyzzzzz-ds-main.sh. DeepSeek-V4.1 Flash EXL3 on the Radiance MAIN line with every DS tuning switch, against the
# tuned3l line (1.17.0 / PR #266 9c7ce498 base) on the same .rad files and settings. gpu02, all 4 cards, TP4, MML 524288.
# IMAGE (XDSM_IMAGE): radiance-local:ds41-main-<ds41-main sha8>-x3c (build/main/assemble.sh on the workstation)
#   = Codeberg PR #266 head 671acfa (upstream's own merge of main 451b136 into the deepseek41 PR, + 7938081 the packed
#     stored IQR expert plane / one-hipMalloc KV pool) + our ds41-lat 641197d, ds41-ar d4a11ca, the deepseekv41 name line,
#     ds41-t3x 67c9cc3 (router / mix fusion, AR_PCN), ds41-ar-acq 4c9e6c2 (AQL_ACQ rule): one tree, built clean by upstream's
#     Dockerfile; exl3r4d / exl3ref / exl3_quant of exl3-main (x3c fa0e958 kernels, pinned to plugin ABI 37).
# ENV = 105's t3 arm (x79-<m>pa-t3 on the tuned3l image 641197d8, 9.11.): RADIANCE_DS41_IDXKEY=mxfp4, dual heat 0.7 +
#   decode pin 0.8 + moves 16, arena levels (LEVELS 1, LEVEL_MAX 1024, STAGE_SPAN 0), RADIANCE_DS41_ARGATHER=on,
#   ROUTER_FUSE / MIX_FUSE / QLDS / AR_PCN 0 by name (ds41-main carries the router / mix fusions default ON, as tuned3x),
#   latent bf16, AQL_ACQ unset; lib-29 best_vars (expert_tiered, ngram ram, ckint 2048, span 2, --prefill-interleave 8
#   pinned -- main's new default auto:1 does not apply --, MAXSEQS 8, NST 0, RAM KV tier) + MNBT 4160 + --live.
#   225: host pool 19456, gate 23 ms; 29: host pool 31488, gate 30 ms.
# ORDER (user 10.11.: the DeepSeek line goes on with EXL3 2.25): the 225 arm, its gaps start, its profile start, then a
# SHORT 29 arm (XDSM_SHORT; identity + decode only). PER ARM (XDSM_ARMS, default "225"), decode before prefill (TESTING 9.11.):
#   gate (restart, warm-up incl. 16k prefill, 1-user fastcheck 384; one retry over the limit) -> modes (indexer note,
#   ARGATHER, staging, fusion / acquire notes; the arm stops when they are not the t3 line's) -> warm-up answers vs the
#   stock reference -> slow-state probe (mb c8 45 s: fast <= 1.07x, slow >= 1.14x 105's c8 ms a step; slow -> one restart)
#   -> probe (1 user, 512 tok) -> mb c4 -> mb c8 -> mb pf -> mb turns -> i1 copy (31k cold prompt, 900 greedy tokens)
#   -> deep answer (136k, 512 greedy tokens) -> BetterBench --quick (concurrency 1/2/4/8, thinking off; XDSM_BB) ->
#   per-phase /stats, live lines. A thermal line (thermwin) beside every prefill-side result.
# IDENTITY vs tuned3l (105, the same .rad + env): the warm-up answers vs stock (105: IDENTICAL), the i1 copy byte for byte
#   vs runs/x79/copy-x79-<m>pa-t3-i1-run1.txt, the deep answer vs runs/x79/deep-x79-<m>pa-t3.txt. A difference is logged
#   as VIRHE (not hidden) and the arm goes on (the numbers are still read, marked NOT EXACT).
# GAPS (XDSM_GAP, default 225): one fresh start with every played pass stamped on card 0 (RADIANCE_AQL_STAMP=1), warm-up,
#   12 s mb c4 / c8 windows -> tools/gapcat.py + gapsize.py: the dependent-launch gap split vs 117's tuned3l line
#   (x117g-225-t3: median 4.0 / 4.04 us, 771 / 811 same-queue ordered gaps a pass, wall 28144 / 43244 us a pass).
# PROFILE (XDSM_PROF, default 225): one fresh start with --profile-ops (RADIANCE_PROFILE_EVERY 16), no gate (profiled steps
#   fail any limit): mb c4 30 s, mb c8 30 s, one fresh 128k prompt -> queue-ar/dsmain/ds_opprof.py: device ms a step per
#   rank by class (experts, moe other, gather+AR, collectives, attention, indexer, mhc, dense GEMMs, embed/engram, norm),
#   lane waits, and rank 0's wall minus device minus waits (host issue + profiler syncs). PROFILED = not a speed.
# NO cc_sim (user 10.11.). No power / host / fan settings touched; production compose files read only (project dir).
# References (no re-runs): 105 (9.11., poller on; TESTING 125: no effect) x79-225pa-t3: guard 15.81, probe 16.7 / 16.3 ms,
#   c4 124.0 tok/s 27.6 ms, c8 153.8 43.1 ms, pf 1777.9 tok/s, turns cold 20.74 s; x79-29pa-t3: guard 20.6, probe 20.9 /
#   21.5, c4 80.5 40.0 ms, c8 101.0 64.7 ms, pf 1023.7. No DS BetterBench reference exists (new absolute numbers).
# DRYRUN=1 bash dsm-serve.sh : prints every command, runs and writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$QA/ds41-shp-tp4.yaml
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=$R/xdsm; export STATSDIR=$O/stats
RX=$R/x79                                       # 105's tuned3l reference files
export IMAGE=${XDSM_IMAGE:-radiance-local:ds41-main-MAINSHA-x3c}
ARMS=${XDSM_ARMS-225}; BBARMS=${XDSM_BB-225}; GAPARMS=${XDSM_GAP-225}; PROFARMS=${XDSM_PROF-225}   # set but empty = none
SHORTARMS=${XDSM_SHORT-29}   # user 10.11.: 2.25 is the DeepSeek line; 2.9 a short identity / perf arm, last
M29=deepseek-v4.1-flash-1017-exl3-2.9bpw.rad; M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_29=$R/x29/guard/x29-best524-warm-q.txt; WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
BBROOT=$HOME/gpu/ext/betterbench; BBDEPS=$HOME/gpu/ext/bb-deps; BBRUN=$TL/bb_run.py
DQ=$QA/deepq.py
OPP=$HD/ds_opprof.py
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ"
t3env() {   # 105's t3 arm, every other switch unset
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
}
modelvars() {   # modelvars 225|29 -> MODEL, HP29, LIM, WARMREF, C8REF (105's c8 ms a step)
  case $1 in
    225) export MODEL=$M225; HP29=19456; LIM=23; WARMREF=$WARM_REF_225; C8REF=43.1 ;;
    29)  export MODEL=$M29;  HP29=31488; LIM=$GATE29; WARMREF=$WARM_REF_29; C8REF=64.7 ;;
    *)   return 1 ;;
  esac
  best_vars; export MNBT=4160 XARGS="$XARGS --live"
}
now() { python3 -c 'import time; print(time.clock_gettime_ns(time.CLOCK_MONOTONIC))'; }   # the stamps' clock
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
gate() {   # gate TAG LIMIT: lib-29's gate + keeplog before the retry / after a second slow check
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
  log "RESULT guard $tag: $st, 1-user ${ms} ms/step (limit $lim; 105 tuned3l: 225 15.81 / 29 20.6)"
  case $st in SLOW*) return 2 ;; esac
}
FOLLOW_PID=""
follow() { [ "$DRY" = 1 ] && { echo "+ docker logs -f $CN > $1/engine.log &"; return; }
  docker logs -f $CN > $1/engine.log 2>&1 & FOLLOW_PID=$!; sleep 2; }
unfollow() { [ -n "$FOLLOW_PID" ] && { kill $FOLLOW_PID 2>/dev/null; wait $FOLLOW_PID 2>/dev/null; FOLLOW_PID=""; }; return 0; }
trap 'unfollow; down' EXIT
trap 'exit 143' TERM INT
mark() { [ "$DRY" = 1 ] && { echo "+ mark $1 -> $2/marks.txt"; return; }; echo "$1 $(wc -l < $2/engine.log) $(date +%s)" >> $2/marks.txt; }
snap() { run "curl -s -m 15 $URL/stats > $1/stats-$2.json 2>/dev/null; echo $2 >> $1/stats-order.txt"; }
ph() { mark $2 $1; snap $1 $2; }
thermline() {   # thermline T0 T1: the window's full-power rows (heatsoak.sh thermwin) or the cards now
  type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"
}
cmpfile() {   # cmpfile A B -> IDENTICAL (n chars) | DIFFERENT at char k (na vs nb chars) | missing
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xdsm $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xdsm $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  case $r in IDENTICAL*) log "RESULT xdsm $1 warm-up answers vs stock ($(basename $2)): $r (105 tuned3l: IDENTICAL)" ;;
    *) log "RESULT xdsm $1 warm-up answers vs stock ($(basename $2)): VIRHE $r -- NOT EXACT vs the tuned3l line" ;; esac
}
modes() {   # modes TAG: the serve log's notes must be the t3 line's -> 0, else 1 (arm stopped)
  [ "$DRY" = 1 ] && { echo "+ modes $1: indexer note (mxfp4, latent bf16, compressor tail bound, router selection+sort separate, AR_PCN separate at 0 layers), ARGATHER note, staging line, no acquire-rule line, interleave"; return 0; }
  local MLF SL AL IL ok=1
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  AL=$(docker logs $CN 2>&1 | grep -m3 -E 'activation arena' | cut -c1-200 | tr '\n' '|')
  IL=$(grep -iE 'interleave' $O/start-$1.log 2>/dev/null | sed 's/^[A-Z] *//' | tr -s ' ' | head -3 | tr '\n' '|' | cut -c1-300)
  log "RESULT xdsm $1 modes: $(echo "$MLF" | cut -c1-1100) | arena: $AL | staging: ${SL:-none} | interleave: ${IL:-no line}"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate" "post-collapse-norm separate at 0 layers" "next collapse-norm separate at 0 layers"; do
    case "$MLF" in *"$w"*) ;; *) log "RESULT xdsm $1: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  docker logs $CN 2>&1 | grep -q 'the MoE gather and its all-reduce as one kernel' || { log "RESULT xdsm $1: VIRHE no RADIANCE_DS41_ARGATHER note"; ok=0; }
  [ -n "$SL" ] || { log "RESULT xdsm $1: VIRHE no prefill staging line (levels refused?)"; ok=0; }
  docker logs $CN 2>&1 | grep -q 'the acquire rule plays' && { log "RESULT xdsm $1: VIRHE an acquire-rule line (RADIANCE_AQL_ACQ should be unset)"; ok=0; }
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
mbms() { grep '^MB ' $1 2>/dev/null | head -1 | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()[3:]).get("ms_decode_step", ""))' 2>/dev/null; }
stprobe() {   # stprobe TAG N: mb c8 45 s -> "fast|between|slow MS (ref REF)"
  run "(cd $D && timeout 400 python3 mb.py --tag xdsm-$1-st$2 --phases c8 --dur ${XDSM_STATE_DUR:-45} --warm 10 > $O/$1-stprobe$2.txt 2>&1)"
  [ "$DRY" = 1 ] && { echo "fast (dry)"; return 0; }
  local ms; ms=$(mbms $O/$1-stprobe$2.txt)
  [ -n "$ms" ] || { echo "unknown (no mb line)"; return 0; }
  awk -v m=$ms -v r=$C8REF 'BEGIN{s=(m <= 1.07*r) ? "fast" : (m >= 1.14*r) ? "slow" : "between"; printf "%s %s ms (105 tuned3l c8 %s ms, x%.3f)\n", s, m, r, m/r}'
}
copyt() {   # copyt TAG OFFSET: the copy test after a ~31k-token cold prompt (105's prompt: same offset -> same text)
  local t0=$(date +%s)
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset $2 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xdsm $1 copy after a cold prompt: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300) | $(thermline $t0 $(date +%s))"
}
deepq() {   # deepq TAG: the fixed ~135k prompt and a 512-token greedy answer (105's deepq.py)
  local t0=$(date +%s)
  run "(cd $D && timeout 1800 python3 $DQ $O/deep-$1.txt 135000 512 > $O/deep-$1.out 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xdsm $1 deep probe: $(grep -m1 '^DEEPQ ' $O/deep-$1.out | cut -c7- | cut -c1-200) | $(thermline $t0 $(date +%s))"
}
mbpf() {   # mbpf TAG PHASE: lib-pr266's mbp + a thermal line
  local t0=$(date +%s)
  mbp $1 $2
  [ "$DRY" = 1 ] || log "RESULT xdsm $1 mb $2 window $(thermline $t0 $(date +%s))"
}
bbph() {   # bbph TAG MODEL: BetterBench --quick (prefill sweep + concurrency 1/2/4/8, thinking off)
  run "(BETTERBENCH_HOME=$O/bbhome timeout 3000 python3 -I $BBRUN --bb-root $BBROOT --deps $BBDEPS -- run --endpoint $URL/v1 --model deepseek-v4.1-flash --quick --config $O/bb-config.json --name $1 --out $O/bb-$1/results.json --note image=$IMAGE --note model=$2 --note tp=4 --note mml=524288 --note env=t3 > $O/$1-bb.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  log "RESULT xdsm $1 BetterBench (no DS reference exists): $(python3 -I $BBRUN --summary $O/bb-$1/results.json 2>&1 | cut -c1-1400)"
  python3 -I $BBRUN --check $O/bb-$1/results.json > /dev/null 2>&1 || log "RESULT xdsm $1 BetterBench: VIRHE single-stream answers carried reasoning text -- thinking was not off"
}
statsum() {   # statsum DIR TAG: per-phase deltas of the /stats snapshots
  [ "$DRY" = 1 ] && { echo "+ RESULT xdsm $2 stats per phase from $1/stats-*.json"; return 0; }
  python3 - $1 > $1/stats-phases.txt 2>&1 <<'PY'
import json, os, sys
d = sys.argv[1]
order = [l.strip() for l in open(os.path.join(d, "stats-order.txt")) if l.strip()]
snaps = []
for p in order:
    try: snaps.append((p, json.load(open(os.path.join(d, f"stats-{p}.json")))))
    except Exception: pass
G = 1e9
for (pa, a), (pb, b) in zip(snaps, snaps[1:]):
    ea, eb = a.get("experts", {}), b.get("experts", {}); la, lb = a.get("link", {}), b.get("link", {})
    ds = b["steps"] - a["steps"]
    if ds <= 0: print(f"PH {pb}: no steps"); continue
    dpm = (b["steps_prefill"] + b["steps_mixed"]) - (a["steps_prefill"] + a["steps_mixed"]); dd = b["steps_decode"] - a["steps_decode"]
    dr = eb.get("routed", 0) - ea.get("routed", 0)
    o = {"phase": pb, "steps": ds, "prefill_mixed_steps": dpm, "decode_steps": dd,
         "ms_step": round((b["step_ns"] - a["step_ns"]) / ds / 1e6, 2),
         "ms_decode_step": round((b["step_ns_decode"] - a["step_ns_decode"]) / max(1, dd) / 1e6, 2) if dd else None,
         "ms_prefill_mixed_step": round(((b["step_ns"] - b["step_ns_decode"]) - (a["step_ns"] - a["step_ns_decode"])) / max(1, dpm) / 1e6, 1) if dpm else None,
         "hit_pct_rank0": round(100 * (eb.get("routed_res", 0) - ea.get("routed_res", 0)) / dr, 2) if dr else None,
         "stream_MB_step": round((lb.get("stream_bytes", 0) - la.get("stream_bytes", 0)) / ds / 1e6, 1),
         "h2d_MB_step": round((lb.get("h2d_bytes", 0) - la.get("h2d_bytes", 0)) / ds / 1e6, 1),
         "stage_layers": eb.get("stage_layers", 0) - ea.get("stage_layers", 0),
         "staged_GB": round((eb.get("staged_bytes", 0) - ea.get("staged_bytes", 0)) / G, 2),
         "stage_overflow": eb.get("stage_overflow", 0) - ea.get("stage_overflow", 0),
         "vram_units_end": next((t.get("units") for t in eb.get("tiers", []) if t.get("name") == "vram"), None)}
    print("PH " + json.dumps(o))
PY
  log "RESULT xdsm $2 stats per phase: $(grep '^PH ' $1/stats-phases.txt | cut -c4- | python3 -c 'import json,sys
out = []
for l in sys.stdin:
    try: d = json.loads(l)
    except Exception: continue
    out.append("%s: %s st (%s pf/mx), %s ms/st, dec %s ms, pf/mx %s ms, hit %s %%, stream %s MB/st, staged %s layers %s GB (%s overflow), vram %s" % (
        d["phase"], d["steps"], d["prefill_mixed_steps"], d["ms_step"], d["ms_decode_step"], d["ms_prefill_mixed_step"], d["hit_pct_rank0"],
        d["stream_MB_step"], d["stage_layers"], d["staged_GB"], d["stage_overflow"], d["vram_units_end"]))
print("; ".join(out))' 2>&1 | cut -c1-1800)"
}
cmpref() {   # cmpref MODEL TAG [REFDIR REFTAG NAME]: this arm vs a reference arm (default 105's x79-<m>pa-t3, tuned3l)
  local rd=${3:-$RX} rt=${4:-x79-$1pa-t3} nm=${5:-"tuned3l (105 x79-$1pa-t3, same .rad + env)"}
  [ "$DRY" = 1 ] && { echo "+ RESULT xdsm $2 vs $nm: guard, probe, c4, c8, pf, turns, copy / deep TTFT"; return 0; }
  log "RESULT xdsm $2 vs $nm: $(python3 - $O $2 $rd $rt <<'PY' 2>&1 | cut -c1-1600
import json, sys
O, a, RX, r = sys.argv[1:5]
def mb(d, tag, p):
    try:
        for l in open(f"{d}/{tag}-{p}.txt"):
            if l.startswith("MB "): return json.loads(l[3:])
    except OSError: pass
    return {}
def gd(d, tag):
    try: return float(open(f"{d}/guard-{tag}.txt").read().split()[-1])
    except Exception: return None
def pr(d, tag, which):
    try:
        for l in open(f"{d}/{tag}-probe.txt"):
            if l.startswith("J "):
                j = json.loads(l[2:])
                if j.get("prompt") == which: return j.get("ms_per_decode_step")
    except OSError: pass
    return None
def cp(d, tag):
    try:
        for l in open(f"{d}/{tag}-i1-copy.txt"):
            if l.startswith("J "): return json.loads(l[2:]).get("ttft")
    except OSError: pass
    return None
def dq(d, tag):
    try:
        for l in open(f"{d}/deep-{tag}.out"):
            if l.startswith("DEEPQ "): return json.loads(l[6:]).get("ttft")
    except OSError: pass
    return None
rows = [("guard ms", gd(RX, r), gd(O, a)), ("probe essay ms", pr(RX, r, "essay"), pr(O, a, "essay")), ("probe code ms", pr(RX, r, "code"), pr(O, a, "code")),
        ("c4 tok/s", mb(RX, r, "c4").get("agg_dec_tps"), mb(O, a, "c4").get("agg_dec_tps")), ("c4 ms", mb(RX, r, "c4").get("ms_decode_step"), mb(O, a, "c4").get("ms_decode_step")),
        ("c8 tok/s", mb(RX, r, "c8").get("agg_dec_tps"), mb(O, a, "c8").get("agg_dec_tps")), ("c8 ms", mb(RX, r, "c8").get("ms_decode_step"), mb(O, a, "c8").get("ms_decode_step")),
        ("pf tok/s", mb(RX, r, "pf").get("prefill_tps"), mb(O, a, "pf").get("prefill_tps")), ("pf TTFT s", mb(RX, r, "pf").get("prefill_ttft"), mb(O, a, "pf").get("prefill_ttft")),
        ("turns cold s", mb(RX, r, "turns").get("cold_ttft"), mb(O, a, "turns").get("cold_ttft")), ("turns hit s", mb(RX, r, "turns").get("ttft_hit_mean"), mb(O, a, "turns").get("ttft_hit_mean")),
        ("copy 28k TTFT s", cp(RX, r), cp(O, a)), ("deep 136k TTFT s", dq(RX, r), dq(O, a))]
out = []
for n, x, y in rows:
    if x is None or y is None: out.append(f"{n} {x} -> {y}"); continue
    x, y = float(x), float(y)
    out.append(f"{n} {x:g} -> {y:g} ({(y / x - 1) * 100:+.1f} %)" if x else f"{n} {x} -> {y}")
print("; ".join(out))
PY
)"
}
arm() {   # arm SPEC [short]: 225 | 29 = the t3 line on the main image; 225h | 29h = + RADIANCE_ARENA_HOLD_MS (XDSM_HOLD_MS,
  # 5000: held item 57-p5's onsh lever on today's stack; compared with this run's / an earlier run's dsm-<m> arm, no
  # BetterBench). "short" (user 10.11.: 2.25 is the DS line, 2.9 only a short identity / perf arm): gate, modes, warm-up
  # answers, state probe, probe, mb c4 / c8, the i1 copy (identity) -- no pf / turns / deep / BetterBench.
  local spec=$1 mo=${1%h} tag=dsm-$1 P=$O/dsm-$1 rc st r v="" short=${2:-}
  [ "$spec" != "$mo" ] && v=h
  t3env; modelvars $mo || { log "RESULT xdsm: VIRHE unknown model '$mo'"; return 1; }
  [ "$v" = h ] && export RADIANCE_ARENA_HOLD_MS=${XDSM_HOLD_MS:-5000}
  log "RESULT xdsm $tag: start IMAGE=$IMAGE MODEL=$MODEL HOSTPOOL=$HOSTPOOL MNBT=$MNBT env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
  if [ "$DRY" != 1 ]; then
    [ -f $MODELDIR/$MODEL ] && [ -s $MODELDIR/$MODEL.sha256ok ] || { log "RESULT xdsm $tag: VIRHE $MODELDIR/$MODEL missing or not verified"; return 1; }
  fi
  mkd $P; run "rm -f $P/marks.txt $P/stats-order.txt"
  gate $tag $LIM; rc=$?
  [ $rc = 1 ] && { log "RESULT xdsm $tag: VIRHE not healthy ($(desc)); start log $O/start-$tag.log: $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  [ $rc = 2 ] && { log "RESULT xdsm $tag: VIRHE slow state after one restart -> arm skipped"; down; return 1; }
  modes $tag || { log "RESULT xdsm $tag: VIRHE the modes are not the t3 line's -> arm stopped"; down; return 1; }
  identw $tag $WARMREF
  st=$(stprobe $tag 1)
  if [ "${st%% *}" = slow ]; then
    log "RESULT xdsm $tag state probe 1: $st -> one restart"
    gate $tag $LIM; rc=$?
    [ $rc = 0 ] || { log "RESULT xdsm $tag: VIRHE no healthy / fast start after the slow-state restart"; down; return 1; }
    modes $tag || { down; return 1; }
    st=$(stprobe $tag 2)
  fi
  log "RESULT xdsm $tag state: $st (slow-state probe mb c8 ${XDSM_STATE_DUR:-45} s; window poller retired, TESTING 9.11.)"
  follow $P; ph $P start
  probe $tag; ph $P probe
  for p in c4 c8; do stopfile && break; mbp $tag $p; ph $P $p; done
  [ -n "$short" ] || for p in pf turns; do stopfile && break; mbpf $tag $p; ph $P $p; done
  stopfile || { copyt $tag-i1 0; ph $P i1; }
  [ -n "$short" ] || stopfile || { deepq $tag; ph $P deep; }
  if [ "$DRY" = 1 ]; then echo "+ identity: cmp $O/copy-$tag-i1-run1.txt $RX/copy-x79-${mo}pa-t3-i1-run1.txt; cmp $O/deep-$tag.txt $RX/deep-x79-${mo}pa-t3.txt"
  else
    r=$(cmpfile $O/copy-$tag-i1-run1.txt $RX/copy-x79-${mo}pa-t3-i1-run1.txt)
    case $r in IDENTICAL*) log "RESULT xdsm $tag identity: i1 copy vs tuned3l (105 x79-${mo}pa-t3-i1): $r" ;;
      *) log "RESULT xdsm $tag identity: VIRHE i1 copy vs tuned3l (105 x79-${mo}pa-t3-i1): $r -- NOT EXACT" ;; esac
    if [ -z "$short" ]; then
      r=$(cmpfile $O/deep-$tag.txt $RX/deep-x79-${mo}pa-t3.txt)
      case $r in IDENTICAL*) log "RESULT xdsm $tag identity: deep answer vs tuned3l (105 x79-${mo}pa-t3): $r" ;;
        *) log "RESULT xdsm $tag identity: VIRHE deep answer vs tuned3l (105 x79-${mo}pa-t3): $r -- NOT EXACT at depth" ;; esac
    fi
  fi
  if [ "$v" = h ]; then   # the hold arm against the main line's own t3 arm (this run or an earlier one)
    if [ "$DRY" = 1 ]; then echo "+ identity: cmp vs $O/copy-dsm-$mo-i1-run1.txt, $O/deep-dsm-$mo.txt"
    else
      log "RESULT xdsm $tag identity vs the main t3 arm dsm-$mo: i1 copy $(cmpfile $O/copy-$tag-i1-run1.txt $O/copy-dsm-$mo-i1-run1.txt); deep $(cmpfile $O/deep-$tag.txt $O/deep-dsm-$mo.txt)"
    fi
  fi
  case " $BBARMS " in *" $spec "*) stopfile || { bbph $tag $MODEL; ph $P bb; } ;; esac
  unfollow
  run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
  statsum $P $tag
  run "python3 $T/profsum.py live $P > $P/live.txt 2>&1"
  down
  cmpref $mo $tag
  [ "$v" = h ] && cmpref $mo $tag $O dsm-$mo "the main t3 arm dsm-$mo (RADIANCE_ARENA_HOLD_MS unset = 2000 ms)"
  unset RADIANCE_ARENA_HOLD_MS
  log "RESULT xdsm $tag: done"
}
gaps() {   # gaps MODEL: card-0 stamps of every played pass, c4 / c8 12 s windows, gapcat + gapsize
  local mo=$1 tag=dsm-$1-gap f p t0 t1
  t3env; modelvars $mo || return 1
  export AQLSTAMP=1 STATSDIR=$O/stats-gap-$mo
  mkd $STATSDIR
  run "find $STATSDIR -maxdepth 1 -name 'stamps.csv*' -delete; rm -f $O/$tag-win.txt"
  log "RESULT xdsm $tag: start IMAGE=$IMAGE MODEL=$MODEL RADIANCE_AQL_STAMP=1 (every played pass, card 0) $(desc)"
  restart $tag || { keeplog $tag-fail; down; log "RESULT xdsm $tag: VIRHE not healthy -- no gap windows"; export AQLSTAMP= STATSDIR=$O/stats; return 1; }
  local ms; ms=$(check $tag 1)
  [ "$DRY" = 1 ] || log "RESULT xdsm $tag warm-up + 1-user fastcheck (STAMPED, not gated): $ms ms/step | 117 tuned3l stamped run: x117g-225-t3"
  for p in c4 c8; do
    stopfile && break
    run "curl -s -m 15 $URL/stats > $O/$tag-$p-sta.json 2>/dev/null"
    if [ "$DRY" = 1 ]; then t0=T0; else t0=$(now); fi
    run "(cd $D && timeout 600 python3 mb.py --tag xdsm-$tag --phases $p --dur 12 --warm 8 > $O/$tag-$p.txt 2>&1)"
    if [ "$DRY" = 1 ]; then t1=T1; else t1=$(now); fi
    run "curl -s -m 15 $URL/stats > $O/$tag-$p-stb.json 2>/dev/null; echo '$p $t0 $t1' >> $O/$tag-win.txt"
    [ "$DRY" = 1 ] || log "RESULT xdsm $tag mb $p (12 s, stamped): $(grep '^MB ' $O/$tag-$p.txt 2>/dev/null | cut -c4- | cut -c1-300)"
  done
  keeplog $tag
  down   # the stamp writer flushes at exit
  f=$STATSDIR/stamps.csv
  if [ "$DRY" = 1 ]; then for p in c4 c8; do echo "+ gapcat.py $f --from T0 --to T1 --label $tag-$p-card0 --json $O/$tag-$p-gap0.json; gapsize.py $f --from T0 --to T1 --label $tag-$p --json $O/$tag-$p-gapsize.json"; done
  else
    [ -s $f ] || log "RESULT xdsm $tag: VIRHE no stamp file $f"
    while read -r p t0 t1; do
      timeout 1800 python3 -I $TL/gapcat.py $f --from $t0 --to $t1 --label $tag-$p-card0 --json $O/$tag-$p-gap0.json > $O/$tag-$p-gap0.txt 2>&1
      log "RESULT xdsm $tag $p card 0 gaps: $(tr -s ' ' < $O/$tag-$p-gap0.txt | tr '\n' '|' | cut -c1-2000)"
      timeout 1800 python3 -I $TL/gapsize.py $f --from $t0 --to $t1 --label $tag-$p --json $O/$tag-$p-gapsize.json > $O/$tag-$p-gapsize.txt 2>&1
      log "RESULT xdsm $tag $p dependent-launch gap vs kernel size: $(cut -c1-1200 $O/$tag-$p-gapsize.txt | tr '\n' ' ')"
    done < $O/$tag-win.txt
    log "RESULT xdsm $tag summary vs 117 tuned3l (x117g-225-t3), card 0 trunk pass: $(python3 - $O $tag $R/x117g x117g-225-t3 <<'PY' 2>&1 | cut -c1-2000
import json, sys
O, tag, RO, rt = sys.argv[1:5]
def one(d, t, p):
    try: g = json.load(open(f"{d}/{t}-{p}-gap0.json"))
    except (OSError, ValueError): return None
    k = max(g["kinds"], key=lambda x: int(x)); x = g["kinds"][k]
    r = {"wall": x["wall_us"], "gaps": x["gaps_us"], "disp": x["dispatches"], "ordered": x["ordered"], "chained": x["chained"]}
    try:
        z = json.load(open(f"{d}/{t}-{p}-gapsize.json")); r["med"] = z.get("median_gap_us"); r["gpp"] = z.get("ordered_gaps_a_pass")
    except (OSError, ValueError): pass
    return r
out = []
for p in ("c4", "c8"):
    a, b = one(RO, rt, p), one(O, tag, p)
    if not b: out.append(f"{p}: no analysis"); continue
    def f(n):
        x, y = (a or {}).get(n), b.get(n)
        if x is None or y is None: return f"{n} {x} -> {y}"
        return f"{n} {x:g} -> {y:g} ({(float(y) / float(x) - 1) * 100:+.1f} %)" if x else f"{n} {x} -> {y}"
    out.append(f"{p}: " + ", ".join(f(n) for n in ("wall", "gaps", "disp", "ordered", "chained", "gpp", "med")))
print(" ; ".join(out) + " (wall / gaps us a trunk pass; disp = dispatches, ordered = barrier-bit; gpp = same-queue ordered gaps a pass; med = their median us)")
PY
)"
    run "gzip -f $STATSDIR/stamps.csv* 2>/dev/null || true"
  fi
  export AQLSTAMP= STATSDIR=$O/stats
  log "RESULT xdsm $tag: done"
}
prof() {   # prof MODEL: --profile-ops (RADIANCE_PROFILE_EVERY 16; a window's edges are the dumps before them, +-15 steps), c4 / c8 30 s windows and one fresh 128k prompt
  local mo=$1 tag=dsm-$1-prof l0 l1 l2 l3 t0 t1 t2 t3 E
  t3env; modelvars $mo || return 1
  export PROFEVERY=${XDSM_PROFEVERY:-16} XARGS="$XARGS --profile-ops" STATSDIR=$O/stats-prof-$mo
  mkd $STATSDIR
  E=$O/$tag-engine.log
  log "RESULT xdsm $tag: start IMAGE=$IMAGE MODEL=$MODEL --profile-ops RADIANCE_PROFILE_EVERY=$PROFEVERY (PROFILED: not a speed) $(desc)"
  restart $tag || { keeplog $tag-fail; down; log "RESULT xdsm $tag: VIRHE not healthy -- no profile"; export PROFEVERY= STATSDIR=$O/stats; return 1; }
  local ms; ms=$(check $tag 1)
  [ "$DRY" = 1 ] || log "RESULT xdsm $tag warm-up + 1-user fastcheck (PROFILED, not gated): $ms ms/step"
  nl() { [ "$DRY" = 1 ] && echo L || docker logs $CN 2>&1 | wc -l; }
  run "curl -s -m 15 $URL/stats > $O/$tag-c4-sta.json 2>/dev/null"; l0=$(nl); t0=$(date +%s)
  run "(cd $D && timeout 600 python3 mb.py --tag xdsm-$tag --phases c4 --dur 30 --warm 10 > $O/$tag-c4.txt 2>&1)"
  run "sleep 3; curl -s -m 15 $URL/stats > $O/$tag-c4-stb.json 2>/dev/null"; l1=$(nl); t1=$(date +%s)
  run "curl -s -m 15 $URL/stats > $O/$tag-c8-sta.json 2>/dev/null"; l1=$(nl)
  run "(cd $D && timeout 600 python3 mb.py --tag xdsm-$tag --phases c8 --dur 30 --warm 10 > $O/$tag-c8.txt 2>&1)"
  run "sleep 3; curl -s -m 15 $URL/stats > $O/$tag-c8-stb.json 2>/dev/null"; l2=$(nl); t2=$(date +%s)
  run "curl -s -m 15 $URL/stats > $O/$tag-pf128-sta.json 2>/dev/null"; l2=$(nl)
  run "(cd $D && timeout 1800 python3 ds41_tests.py prefill --depths 131072 --cpt 3.95 --calibrate 0 --reps 1 > $O/$tag-pf128.txt 2>&1)"
  run "sleep 3; curl -s -m 15 $URL/stats > $O/$tag-pf128-stb.json 2>/dev/null"; t3=$(date +%s)
  # stop first (the shutdown table closes the last window), read the whole log, then remove
  run "docker stop -t 60 $CN > /dev/null 2>&1; docker logs $CN > $E 2>&1"
  down
  if [ "$DRY" = 1 ]; then echo "+ ds_opprof.py $E --from L0 --to L1 (c4) / L1..L2 (c8) / L2.. (128k, to the shutdown table) --stats"
  else
    timeout 900 python3 -I $OPP $E --from $l0 --to $l1 --label $tag-c4 --json $O/$tag-c4-op.json --stats $O/$tag-c4-sta.json $O/$tag-c4-stb.json > $O/$tag-c4-op.txt 2>&1
    log "RESULT xdsm $tag c4 30 s (PROFILED, not a speed): $(cut -c1-3000 $O/$tag-c4-op.txt | tr '\n' ' ') | mb (profiled) $(grep '^MB ' $O/$tag-c4.txt | cut -c4- | cut -c1-160)"
    timeout 900 python3 -I $OPP $E --from $l1 --to $l2 --label $tag-c8 --json $O/$tag-c8-op.json --stats $O/$tag-c8-sta.json $O/$tag-c8-stb.json > $O/$tag-c8-op.txt 2>&1
    log "RESULT xdsm $tag c8 30 s (PROFILED, not a speed): $(cut -c1-3000 $O/$tag-c8-op.txt | tr '\n' ' ') | mb (profiled) $(grep '^MB ' $O/$tag-c8.txt | cut -c4- | cut -c1-160)"
    timeout 900 python3 -I $OPP $E --from $l2 --label $tag-pf128 --json $O/$tag-pf128-op.json --stats $O/$tag-pf128-sta.json $O/$tag-pf128-stb.json > $O/$tag-pf128-op.txt 2>&1
    log "RESULT xdsm $tag fresh 128k prefill (PROFILED, not a speed): $(cut -c1-3000 $O/$tag-pf128-op.txt | tr '\n' ' ') | prompt (profiled): $(grep '^J ' $O/$tag-pf128.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); print("%s tok TTFT %s s %s tok/s" % (d.get("prompt_tokens"), d.get("ttft"), d.get("prefill_tps")))' 2>&1 | cut -c1-160) | $(thermline $t2 $t3)"
    run "gzip -f $E"
  fi
  export PROFEVERY= STATSDIR=$O/stats
  log "RESULT xdsm $tag: done"
}
# ---------------------------------------------------------------- main
log "RESULT xdsm 57-p4zyzzzzz-ds-main: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) ARMS [$ARMS] BB [$BBARMS] GAP [$GAPARMS] PROF [$PROFARMS] SHORT [$SHORTARMS] yaml $Y | refs 105 tuned3l x79-225pa-t3: guard 15.81, c4 124.0, c8 153.8, pf 1777.9; x79-29pa-t3: guard 20.6, c4 80.5, c8 101.0, pf 1023.7; 117 gaps 4.0 us, 771 / 811 a pass"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xdsm: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xdsm: VIRHE image $IMAGE not on gpu02"; exit 1; }
  for v in $SWITCHES; do grep -q "^ *$v:" $Y || { log "RESULT xdsm: VIRHE $Y does not pass $v"; exit 1; }; done
  grep -q RADIANCE_AQL_STAMP $Y && grep -q RADIANCE_PROFILE_EVERY $Y || { log "RESULT xdsm: VIRHE $Y lacks RADIANCE_AQL_STAMP / RADIANCE_PROFILE_EVERY"; exit 1; }
  for v in RADIANCE_DS41_ARGATHER RADIANCE_DS41_AR_PCN RADIANCE_DS41_ROUTER_FUSE RADIANCE_STAGE_SPAN RADIANCE_ARENA_LEVEL_MAX RADIANCE_DS41_LEVELS RADIANCE_HEAT_DECODE_PIN; do
    [ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c "cat /opt/radiance/bin/radiance /opt/radiance/share/radiance/architectures/deepseek41.so | grep -c $v" 2>/dev/null)" -gt 0 ] 2>/dev/null \
      || { log "RESULT xdsm: VIRHE $IMAGE's engine / plugin does not know $v"; exit 1; }
  done
  for f in $TL/gapcat.py $TL/gapsize.py $BBRUN $DQ $OPP $T/profsum.py; do [ -e $f ] || { log "RESULT xdsm: VIRHE $f missing"; exit 1; }; done
  [ -d $BBROOT/betterbench ] && [ -d $BBDEPS/numpy ] || { log "RESULT xdsm: VIRHE BetterBench not ready ($BBROOT, $BBDEPS)"; exit 1; }
  for mo in $ARMS $SHORTARMS; do for f in $RX/copy-x79-${mo}pa-t3-i1-run1.txt $RX/deep-x79-${mo}pa-t3.txt; do [ -s $f ] || log "RESULT xdsm: note -- reference $f missing (identity not compared)"; done; done
fi
mkd $O $O/guard $STATSDIR
run "printf '%s\n' '{\"concurrency_levels\": [1, 2, 4, 8], \"concurrency_requests\": 32, \"prefill_runs\": 4, \"prefill_warmup\": 1}' > $O/bb-config.json"
evict
done_arms=""
for mo in $ARMS; do stopfile && break; arm $mo && done_arms="$done_arms $mo"; done
for mo in $GAPARMS; do stopfile && break; gaps $mo; done
for mo in $PROFARMS; do stopfile && break; prof $mo; done
for mo in $SHORTARMS; do stopfile && break; arm $mo short && done_arms="$done_arms $mo(short)"; done
unset $SWITCHES
down
log "RESULT xdsm 57-p4zyzzzzz-ds-main: done -- arms [$done_arms ] gap [$GAPARMS] prof [$PROFARMS]"
