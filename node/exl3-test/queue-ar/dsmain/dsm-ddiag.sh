#!/bin/bash
# dsm-ddiag.sh (10.11.2026, queue-ar/dsmain, ds-main-port agent) -- NOT a queue item; the body of 57-p4zyzzzzz-ds-bzzz-ddiag.sh.
# The cheap DSpark diagnostics (coordinator 10.11., after the drafter audit): per-position acceptance (essay / code, the -v
# proposal log replayed against the reply ids, perpos.py) and the Markov walk's candidates 16 vs 32
# (RADIANCE_DS41_DSPARK_CAND, image ...-d24cd1d) at NST 5, c1 / c4 / c8, tokens a step the main metric. -v makes the
# timings worthless here (the log); only acceptance and tokens a step are read. The header below is dsm-draft.sh's.
# The DeepSeek-V4.1 DSpark drafter (the checkpoint's own 3-stage block drafter, mtp.0..2: 128 experts top 3 each; it
# proposes its trained block of 5, so deepseek41 serves --num-speculative-tokens 5 or none) measured and profiled on the
# MAIN image, EXL3 2.25, TP4, MML 524288, the t3 env (user 10.11.: "mtp ehka tata voisi profiloida ja kehittaa,
# adaptiivinen jne"). 6.11. on the untuned 1.17 engine NST 5 lost at every load but 1-user code (c4 67.9 vs 99.0); this
# sizes it on today's stack and shows where its step goes.
# ARMS (XDSD_ARMS, default "d0 d5 d5a"; load points c1 / c4 / c8, user 10.11.: "1, 4, 8 riittaa"):
#   d0   NST 0 (the reference of this window; item a's dsm-225 is the same config earlier)
#   d5   --num-speculative-tokens 5 (the DSpark block)
#   d5a  d5 + --draft-window adaptive (each sequence's window follows its acceptance)
# PER ARM: gate (limit 23 ms d0, 70 ms with drafting: a drafting step emits 1-6 tokens), modes (the t3 notes + every
#   drafter / speculation line of the serve log), warm-up answers vs stock, probe (1 user, essay / code: tok/s, ms a step,
#   tokens a step, acceptance), mb c4, mb c8 (aggregate, per-stream p50, tokens a step, acceptance), the 28k copy (i1),
#   the 20-prompt greedy set at c1 (tools/greedyset.py), /stats per phase (expert stream MB a step = the host link, hit %).
#   d0 also writes the verify-path oracle's reference (oracle.py ref: one fixed prompt, 256 greedy ids).
# IDENTITY (reported, never hidden): d5 / d5a's copy and greedy answers vs d0's byte for byte (a different verify shape
#   may change greedy text at a near-tie: upstream GUIDE); the oracle start (XDSD_ORACLE=1): --spec-ngram 5 +
#   --debug-draft-oracle (prompt lookup proposes d0's own ids, 30 % corrupted, so 6-row verifies accept and reject at every
#   depth, no drafter) -- its text must be d0's: the verify path's exactness at the DSpark verify shape.
# PROFILE (XDSD_PROF=1): a d5 start with --profile-ops (RADIANCE_PROFILE_EVERY 16): a 1-user window (stepprobe essay,code)
#   and a c4 30 s window -> ds_opprof_draft.py: per rank the trunk's classes (its decode + verify rows) and the drafter's
#   (draft_*: its 3 blocks, head, dflash walk), lane waits, wall - device - waits; /stats around each window (host link MB a
#   step). PROFILED = not a speed.
# References (no re-runs): item a dsm-225 (main, NST 0): gate 15.59 ms, probe 16.8 / 16.2 ms, c4 122.3 (p50 31.8), c8 148.0
#   (p50 19.5); 6.11. NST 5 (1.17 untuned): essay 33.3 / code 67.2 tok/s (acc 0.10 / 0.51), c4 67.9, c8 74.4; adaptive
#   essay 41.2 / code 69.0, c4 75.7, c8 95.9.
# DRYRUN=1 bash dsm-draft.sh : prints every command, runs and writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-dsm-tp4.yaml   # = queue-ar/ds41-shp-tp4.yaml + RADIANCE_DS41_DSPARK_CAND
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=$R/xdsm/ddiag; export STATSDIR=$O/stats
RX=$R/x79                                       # 105's tuned3l reference files
export IMAGE=${XDSM_IMAGE:-radiance-local:ds41-main-MAINSHA-x3c}
ARMS=${XDSG_ARMS-"k16 k32"}; PP=$HD/perpos.py
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
ORC=$HD/oracle.py; OPPD=$HD/ds_opprof_draft.py
M29=deepseek-v4.1-flash-1017-exl3-2.9bpw.rad; M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_29=$R/x29/guard/x29-best524-warm-q.txt; WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
BBROOT=$HOME/gpu/ext/betterbench; BBDEPS=$HOME/gpu/ext/bb-deps; BBRUN=$TL/bb_run.py
DQ=$QA/deepq.py
OPP=$HD/ds_opprof.py
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND"
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

gsetc1() { run "(cd $D && timeout 900 python3 $GS $URL $O/$1-gs-c1.jsonl --conc 1 --max-tokens 192 > $O/$1-gs-c1.txt 2>&1)"; }
gcmp() { [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }; [ -s "$1" ] && [ -s "$2" ] || { echo "not compared (missing $1 or $2)"; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
draftlines() {   # draftlines TAG: every drafter / speculation line of the serve log, logged
  [ "$DRY" = 1 ] && { echo "+ RESULT xdsd $1 drafter lines"; return 0; }
  log "RESULT xdsd $1 drafter / speculation lines: $(docker logs $CN 2>&1 | grep -iE 'dspark|drafter|speculat|draft-window|draft window|verify' | sed 's/^[A-Z][^ ]* //' | sort -u | head -8 | tr '\n' '|' | cut -c1-1500)"
}
probelog() {   # probelog TAG: the probe's two prompts with tokens a step and acceptance
  [ "$DRY" = 1 ] && return 0
  log "RESULT xdsd $1 probe detail: $(grep '^J ' $O/$1-probe.txt | cut -c3- | python3 -c 'import json,sys
print("; ".join("%s %s tok/s %s ms/step %s tok/step acc %s" % (d.get("prompt"), d.get("decode_tps"), d.get("ms_per_decode_step"), d.get("tokens_per_step"), d.get("acceptance")) for d in map(json.loads, sys.stdin)))' 2>&1 | cut -c1-400)"
}
mbline() {   # mbline TAG PHASE: aggregate, per-stream p50, ms and tokens a step, acceptance
  [ "$DRY" = 1 ] && return 0
  log "RESULT xdsd $1 mb $2: $(grep '^MB ' $O/$1-$2.txt 2>/dev/null | cut -c4- | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print("aggregate %s tok/s, per-stream p50 %s, %s ms a decode step, %s tokens a step, acceptance %s, TTFT p50 %s s, %s requests, %s errors" % (d.get("agg_dec_tps"), d.get("per_stream_dec_p50"), d.get("ms_decode_step"), d.get("tok_per_decode_step"), d.get("acceptance"), d.get("ttft_p50"), d.get("requests"), d.get("errors")))' 2>&1 | cut -c1-400)"
}
karm() {   # karm k16|k32: NST 5 with -v, the walk's candidates from the arm's name
  local a=$1 tag=dsg-$1 P=$O/dsg-$1 rc k=${1#k} pr l0 n
  t3env; modelvars 225
  export NST=5 RADIANCE_DS41_DSPARK_CAND=$k XARGS="$XARGS -v"
  log "RESULT xdsg $tag: start IMAGE=$IMAGE NST 5 RADIANCE_DS41_DSPARK_CAND=$k -v (timings not read) $(desc)"
  mkd $P; run "rm -f $P/marks.txt $P/stats-order.txt"
  gate $tag 90; rc=$?
  [ $rc = 0 ] || { log "RESULT xdsg $tag: VIRHE no healthy start (rc $rc)"; keeplog $tag-fail; down; return 1; }
  if [ "$DRY" != 1 ]; then
    local note; note=$(docker logs $CN 2>&1 | grep -m1 -oE "Markov rank [0-9]+ walked over the head's top [0-9]+")
    log "RESULT xdsg $tag drafter note: ${note:-none}"
    case "$note" in *"top $k") ;; *) log "RESULT xdsg $tag: VIRHE the drafter does not report the walk over the top $k -> arm stopped"; keeplog $tag-nomode; down; return 1 ;; esac
  fi
  identw $tag $WARMREF
  for pr in essay code; do
    stopfile && break
    if [ "$DRY" = 1 ]; then echo "+ PERPOS_PROMPT=$pr perpos.py $URL send; docker logs since; perpos.py score"; continue; fi
    l0=$(docker logs $CN 2>&1 | wc -l)
    PERPOS_PROMPT=$pr PERPOS_N=384 timeout 900 python3 -I $PP $URL send $O/$tag-$pr.json > $O/$tag-$pr-send.txt 2>&1
    sleep 2; docker logs $CN 2>&1 | tail -n +$((l0 + 1)) > $O/$tag-$pr-log.txt
    PERPOS_PROMPT=$pr python3 -I $PP - score $O/$tag-$pr.json $O/$tag-$pr-log.txt > $O/$tag-$pr-perpos.txt 2>&1
    log "RESULT xdsg $tag 1 user $pr: $(grep -m1 '^PERPOS ' $O/$tag-$pr-perpos.txt | cut -c8- | cut -c1-400)"
  done
  for p in c4 c8; do stopfile && break; mbp $tag $p; mbline $tag $p; done
  run "docker logs $CN 2>&1 | grep -c 'proposes' > $O/$tag-proposal-lines.txt"
  down
  log "RESULT xdsg $tag: done"
}
# ---------------------------------------------------------------- main
log "RESULT xdsg 57-p4zyzzzzz-ds-bzzz-ddiag: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) ARMS [$ARMS] | refs bz d5 (NST 5, top 16, no -v)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xdsg: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xdsg: VIRHE image $IMAGE not on gpu02"; exit 1; }
  [ -e $Y ] && grep -q "^ *RADIANCE_DS41_DSPARK_CAND:" $Y || { log "RESULT xdsg: VIRHE $Y does not pass RADIANCE_DS41_DSPARK_CAND"; exit 1; }
  [ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -c RADIANCE_DS41_DSPARK_CAND /opt/radiance/share/radiance/architectures/deepseek41.so' 2>/dev/null)" -gt 0 ] 2>/dev/null || { log "RESULT xdsg: VIRHE $IMAGE's deepseek41.so lacks RADIANCE_DS41_DSPARK_CAND"; exit 1; }
  [ -e $PP ] || { log "RESULT xdsg: VIRHE $PP missing"; exit 1; }
fi
mkd $O $O/guard $STATSDIR
evict
done_arms=""
for a in $ARMS; do stopfile && break; karm $a && done_arms="$done_arms $a"; done
unset $SWITCHES
down
log "RESULT xdsg 57-p4zyzzzzz-ds-bzzz-ddiag: done -- arms [$done_arms ]"
