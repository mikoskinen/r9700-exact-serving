#!/bin/bash
# dsm-ngram.sh (10.11.2026, queue-ar/dsmain, ds41-gg agent) -- NOT a queue item; the body of 57-p4zyzzzzz-ds-hz-ngram.sh.
# DeepSeek-V4.1 with PROMPT LOOKUP (n-gram) DRAFTING ALONE -- the DSpark head not declared, loaded or placed -- on the
# ds41-gg image (XGG_IMAGE; its core passes RadBuildCtx::draft_head_off again under --spec-ngram, which the merge into
# main had dropped, so --spec-ngram was refused beside the head). EXL3 2.25, TP4, MML 524288, the t3 env and serve line of
# dsm-draft.sh (105's t3 arm + lib-29 best_vars + MNBT 4160 + --live). ARM n5a: --spec-ngram 5 --draft-window adaptive,
# --num-speculative-tokens unset (--spec-ngram K sets the window to K; 0 or another value is refused). Under lookup
# alone no step is issued ahead (lookup reads the committed sequence), and a proposal is verified only when its
# record says it pays at the measured step cost.
# PHASES (dsm-draft.sh's darm, decode before prefill-heavy): gate (limit 30 ms) -> modes (the t3 notes, the lookup-alone
#   speculation line, NO DSpark drafter line; the routed experts in VRAM at start vs bz's d0 2948 / d5 2731 units: the
#   head's experts must take none) -> warm-up answers vs stock -> probe essay / code at 1 user (stepprobe: tok/s, ms and
#   tokens a step, acceptance) -> editprobe at 1 user (an edit that re-emits a ~2.5k-token stdlib region, 1500 tokens:
#   tokens a step, acceptance, and the echo check, so a refusal or a summary is flagged) -> mb c4, mb c8 (aggregate,
#   per-stream p50, ms and tokens a step, acceptance; mb's default 200 s, as bz) -> the 28k copy (i1) -> the 20-prompt
#   greedy set at c1.
# IDENTITY vs bz's d0 (NST 0, main image; runs/xdsm/draft): the 28k copy and the greedy set c1, byte for byte. Lookup
#   only proposes and the trunk verifies, so a difference is reported as such (a near-tie at another verify shape), not
#   hidden.
# COMPARISON line vs bz's d0 (NST 0) and d5a (NST 5 adaptive) from their files, no re-runs: probe essay / code tok/s and
#   tokens a step, c4 / c8 aggregate, per-stream p50, tokens a step, acceptance. History: 6.11. (1.17 core) ngram5 at 1
#   user essay 45.9 / code 46.1 tok/s (~1.0 token a step), c4 91.4 (4.14 tok/step, acc 0.552) vs NST 0 98.4-103.8, c8
#   113.7 vs 123.5-129.1.
# NO cc_sim (user 10.11.). No power / host / fan settings; production compose files read only (--project-directory).
# DRYRUN=1 bash dsm-ngram.sh : prints every command, runs and writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-gg-tp4.yaml
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=$R/xdsm/ngram; export STATSDIR=$O/stats
RD=$R/xdsm/draft                                # bz's d0 / d5a files
export IMAGE=${XGG_IMAGE:-radiance-local:ds41-gg-d5bed91f-5c13e555-l}
ITEM=${XGN_ITEM:-57-p4zyzzzzz-ds-hz-ngram}
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
EP=$HD/editprobe.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV"
t3env() {   # 105's t3 arm, every other switch unset
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xgn $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xgn $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  case $r in IDENTICAL*) log "RESULT xgn $1 warm-up answers vs stock ($(basename $2)): $r (105 tuned3l: IDENTICAL)" ;;
    *) log "RESULT xgn $1 warm-up answers vs stock ($(basename $2)): VIRHE $r -- NOT EXACT vs the tuned3l line" ;; esac
}
modes() {   # modes TAG: the serve log's notes must be the t3 line's -> 0, else 1 (arm stopped)
  [ "$DRY" = 1 ] && { echo "+ modes $1: indexer note (mxfp4, latent bf16, compressor tail bound, router selection+sort separate, AR_PCN separate at 0 layers), ARGATHER note, staging line, no acquire-rule line, interleave"; return 0; }
  local MLF SL AL IL ok=1
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  AL=$(docker logs $CN 2>&1 | grep -m3 -E 'activation arena' | cut -c1-200 | tr '\n' '|')
  IL=$(grep -iE 'interleave' $O/start-$1.log 2>/dev/null | sed 's/^[A-Z] *//' | tr -s ' ' | head -3 | tr '\n' '|' | cut -c1-300)
  log "RESULT xgn $1 modes: $(echo "$MLF" | cut -c1-1100) | arena: $AL | staging: ${SL:-none} | interleave: ${IL:-no line}"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate" "post-collapse-norm separate at 0 layers" "next collapse-norm separate at 0 layers"; do
    case "$MLF" in *"$w"*) ;; *) log "RESULT xgn $1: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  docker logs $CN 2>&1 | grep -q 'the MoE gather and its all-reduce as one kernel' || { log "RESULT xgn $1: VIRHE no RADIANCE_DS41_ARGATHER note"; ok=0; }
  [ -n "$SL" ] || { log "RESULT xgn $1: VIRHE no prefill staging line (levels refused?)"; ok=0; }
  docker logs $CN 2>&1 | grep -q 'the acquire rule plays' && { log "RESULT xgn $1: VIRHE an acquire-rule line (RADIANCE_AQL_ACQ should be unset)"; ok=0; }
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
copyt() {   # copyt TAG OFFSET: the copy test after a ~31k-token cold prompt (105's prompt: same offset -> same text)
  local t0=$(date +%s)
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset $2 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xgn $1 copy after a cold prompt: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300) | $(thermline $t0 $(date +%s))"
}
statsum() {   # statsum DIR TAG: per-phase deltas of the /stats snapshots
  [ "$DRY" = 1 ] && { echo "+ RESULT xgn $2 stats per phase from $1/stats-*.json"; return 0; }
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
  log "RESULT xgn $2 stats per phase: $(grep '^PH ' $1/stats-phases.txt | cut -c4- | python3 -c 'import json,sys
out = []
for l in sys.stdin:
    try: d = json.loads(l)
    except Exception: continue
    out.append("%s: %s st (%s pf/mx), %s ms/st, dec %s ms, pf/mx %s ms, hit %s %%, stream %s MB/st, staged %s layers %s GB (%s overflow), vram %s" % (
        d["phase"], d["steps"], d["prefill_mixed_steps"], d["ms_step"], d["ms_decode_step"], d["ms_prefill_mixed_step"], d["hit_pct_rank0"],
        d["stream_MB_step"], d["stage_layers"], d["staged_GB"], d["stage_overflow"], d["vram_units_end"]))
print("; ".join(out))' 2>&1 | cut -c1-1800)"
}
gsetc1() { run "(cd $D && timeout 900 python3 $GS $URL $O/$1-gs-c1.jsonl --conc 1 --max-tokens 192 > $O/$1-gs-c1.txt 2>&1)"; }
gcmp() { [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }; [ -s "$1" ] && [ -s "$2" ] || { echo "not compared (missing $1 or $2)"; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
draftlines() {   # draftlines TAG: every drafter / speculation line of the serve log, logged
  [ "$DRY" = 1 ] && { echo "+ RESULT xgn $1 drafter lines"; return 0; }
  log "RESULT xgn $1 drafter / speculation lines: $(docker logs $CN 2>&1 | grep -iE 'dspark|drafter|speculat|draft-window|draft window|verify' | sed 's/^[A-Z][^ ]* //' | sort -u | head -8 | tr '\n' '|' | cut -c1-1500)"
}
probelog() {   # probelog TAG: the probe's two prompts with tokens a step and acceptance
  [ "$DRY" = 1 ] && return 0
  log "RESULT xgn $1 probe detail: $(grep '^J ' $O/$1-probe.txt | cut -c3- | python3 -c 'import json,sys
print("; ".join("%s %s tok/s %s ms/step %s tok/step acc %s" % (d.get("prompt"), d.get("decode_tps"), d.get("ms_per_decode_step"), d.get("tokens_per_step"), d.get("acceptance")) for d in map(json.loads, sys.stdin)))' 2>&1 | cut -c1-400)"
}
mbline() {   # mbline TAG PHASE: aggregate, per-stream p50, ms and tokens a step, acceptance
  [ "$DRY" = 1 ] && return 0
  log "RESULT xgn $1 mb $2: $(grep '^MB ' $O/$1-$2.txt 2>/dev/null | cut -c4- | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print("aggregate %s tok/s, per-stream p50 %s, %s ms a decode step, %s tokens a step, acceptance %s, TTFT p50 %s s, %s requests, %s errors" % (d.get("agg_dec_tps"), d.get("per_stream_dec_p50"), d.get("ms_decode_step"), d.get("tok_per_decode_step"), d.get("acceptance"), d.get("ttft_p50"), d.get("requests"), d.get("errors")))' 2>&1 | cut -c1-400)"
}
vramunits() {   # vramunits FILE: the routed experts' VRAM / pinned units of a /stats snapshot
  python3 -c 'import json,sys
try: d = json.load(open(sys.argv[1]))
except Exception as e: print("n/a (%s)" % e); sys.exit(0)
t = {x.get("name"): x.get("units") for x in d.get("experts", {}).get("tiers", [])}
print("vram %s, pinned %s" % (t.get("vram"), t.get("pinned")))' "$1" 2>&1
}
editlog() {   # editlog TAG: the edit probe's line
  [ "$DRY" = 1 ] && return 0
  log "RESULT xgn $1 editprobe (1 user, greedy, thinking off, ~2.5k-token stdlib region re-emitted with one docstring, 1500 tok): $(grep '^J ' $O/$1-edit.txt | cut -c3- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
print("%s tok/s, %s ms/step, %s tok/step, acceptance %s (%s drafted, %s accepted), echo ratio %s%s, %s tokens, finish %s, prompt %s tok, function %s" % (
    d.get("decode_tps"), d.get("ms_per_decode_step"), d.get("tokens_per_step"), d.get("acceptance"), d.get("drafted"), d.get("accepted"),
    d.get("echo_ratio"), "" if d.get("echo_ok") else " -- ECHO NOT: the reply does not re-emit the region (a refusal / summary / rewrite), not a copy measurement",
    d.get("tokens"), d.get("finish"), d.get("prompt_tokens"), d.get("function")))' 2>&1 | cut -c1-500)"
}
cmpbz() {   # cmpbz TAG: this arm vs bz's d0 (NST 0) and d5a (NST 5 adaptive), from their files
  [ "$DRY" = 1 ] && { echo "+ RESULT xgn $1 vs bz d0 / d5a: probe essay / code tok/s + tok/step, c4 / c8 aggregate, p50, tok/step, acceptance"; return 0; }
  log "RESULT xgn $1 vs bz d0 (NST 0) / d5a (NST 5 adaptive), main image, same env: $(python3 - $O $1 $RD <<'PY' 2>&1 | cut -c1-1800
import json, sys
O, a, RD = sys.argv[1:4]
def mb(d, tag, p):
    try:
        for l in open(f"{d}/{tag}-{p}.txt"):
            if l.startswith("MB "): return json.loads(l[3:])
    except OSError: pass
    return {}
def pr(d, tag, which, k):
    try:
        for l in open(f"{d}/{tag}-probe.txt"):
            if l.startswith("J "):
                j = json.loads(l[2:])
                if j.get("prompt") == which: return j.get(k)
    except OSError: pass
    return None
arms = [(O, a, "n5a"), (RD, "dsd-d0", "d0"), (RD, "dsd-d5a", "d5a")]
rows = []
for which in ("essay", "code"):
    rows.append("probe %s tok/s %s" % (which, " / ".join("%s %s" % (n, pr(d, t, which, "decode_tps")) for d, t, n in arms)))
    rows.append("%s tok/step %s" % (which, " / ".join("%s %s" % (n, pr(d, t, which, "tokens_per_step")) for d, t, n in arms)))
for p in ("c4", "c8"):
    for k, nm in (("agg_dec_tps", "aggregate"), ("per_stream_dec_p50", "p50"), ("tok_per_decode_step", "tok/step"), ("acceptance", "acc"), ("ms_decode_step", "ms/step")):
        vals = [(n, mb(d, t, p).get(k)) for d, t, n in arms]
        x = vals[0][1]
        def rel(y):
            try: return " (%+.1f %%)" % ((float(x) / float(y) - 1) * 100) if x is not None and y else ""
            except Exception: return ""
        rows.append("%s %s %s" % (p, nm, " / ".join("%s %s%s" % (n, v, rel(v) if i else "") for i, (n, v) in enumerate(vals))))
print("; ".join(rows) + " (n5a vs each in parentheses)")
PY
)"
}
# ---------------------------------------------------------------- main
log "RESULT xgn $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) arm n5a: --spec-ngram 5 --draft-window adaptive (lookup alone, the DSpark head left out) | refs bz d0 (NST 0) / d5a (NST 5 adaptive) on the main image"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xgn: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xgn: VIRHE image $IMAGE not on gpu02"; exit 1; }
  [ -e $Y ] || { log "RESULT xgn: VIRHE $Y missing"; exit 1; }
  for f in $GS $EP $D/stepprobe.py $D/mb.py $D/ds41_tests.py; do [ -e $f ] || { log "RESULT xgn: VIRHE $f missing"; exit 1; }; done
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xgn: VIRHE $M225 missing or not verified"; exit 1; }
  [ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -q "draft head is left out" /opt/radiance/bin/radiance && echo yes' 2>/dev/null)" = yes ] \
    || { log "RESULT xgn: VIRHE $IMAGE's engine does not leave a draft head out under --spec-ngram"; exit 1; }
  for f in $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $RD/dsd-d0-c4.txt $RD/dsd-d5a-c4.txt; do [ -s $f ] || log "RESULT xgn: note -- reference $f missing (not compared)"; done
fi
mkd $O $O/guard $STATSDIR
evict
TAG=gn-n5a; P=$O/$TAG
t3env; export MODEL=$M225; HP29=19456; best_vars
export MNBT=4160 NST= XARGS="$XARGS --live --spec-ngram 5 --draft-window adaptive"
log "RESULT xgn $TAG: start IMAGE=$IMAGE MODEL=$MODEL $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
mkd $P; run "rm -f $P/marks.txt $P/stats-order.txt"
gate $TAG 30; rc=$?
[ $rc = 1 ] && { log "RESULT xgn $TAG: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$TAG.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; exit 1; }
[ $rc = 2 ] && { log "RESULT xgn $TAG: VIRHE over the gate limit after one restart -> stopped"; exit 1; }
modes $TAG || { log "RESULT xgn $TAG: VIRHE the modes are not the t3 line's -> stopped"; exit 1; }
draftlines $TAG
if [ "$DRY" = 1 ]; then echo "+ lookup-alone line present, no DSpark drafter line; VRAM expert units at start vs d0 / d5"
else
  ok=1
  docker logs $CN 2>&1 | grep -q "speculation: prompt lookup only (--spec-ngram 5) -- the container's draft head is left out" || { log "RESULT xgn $TAG: VIRHE no lookup-alone speculation line"; ok=0; }
  docker logs $CN 2>&1 | grep -qiE "DSpark drafter|'dspark' proposes" && { log "RESULT xgn $TAG: VIRHE a DSpark drafter line: the head was declared"; ok=0; }
  run "curl -s -m 15 $URL/stats > $O/$TAG-stats0.json 2>/dev/null"
  log "RESULT xgn $TAG experts at start: $(vramunits $O/$TAG-stats0.json) | bz d0 (NST 0): $(vramunits $RD/dsd-d0/stats-start.json) | bz d5 (NST 5, head placed): $(vramunits $RD/dsd-d5/stats-start.json)"
  [ $ok = 1 ] || { keeplog $TAG-nolookup; exit 1; }
fi
identw $TAG $WARM_REF_225
follow $P; ph $P start
probe $TAG; probelog $TAG; ph $P probe
run "(cd $D && timeout 900 python3 -I $EP 9000 1500 7 > $O/$TAG-edit.txt 2>&1)"; editlog $TAG; ph $P edit
for p in c4 c8; do stopfile && break; mbp $TAG $p; mbline $TAG $p; ph $P $p; done
stopfile || { copyt $TAG-i1 0; ph $P i1; }
stopfile || { gsetc1 $TAG; ph $P gs1; }
if [ "$DRY" = 1 ]; then echo "+ identity: 28k copy vs bz d0, greedy set c1 vs bz d0"
else
  r=$(cmpfile $O/copy-$TAG-i1-run1.txt $RD/copy-dsd-d0-i1-run1.txt)
  case $r in IDENTICAL*) log "RESULT xgn $TAG identity: 28k copy vs bz d0 (NST 0): $r" ;;
    *) log "RESULT xgn $TAG identity: DIFFERENT 28k copy vs bz d0 (NST 0): $r -- lookup changed the greedy text (a near-tie at another verify shape; reported, not hidden)" ;; esac
  r=$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$TAG-gs-c1.jsonl)
  case $r in IDENTICAL*) log "RESULT xgn $TAG identity: greedy set c1 vs bz d0: $r" ;;
    *) log "RESULT xgn $TAG identity: DIFFERENT greedy set c1 vs bz d0: $r -- reported, not hidden" ;; esac
fi
unfollow
run "docker logs $CN > $O/$TAG-engine-full.log 2>&1"
statsum $P $TAG
down
cmpbz $TAG
unset $SWITCHES
down
log "RESULT xgn $ITEM: done -- $O"
