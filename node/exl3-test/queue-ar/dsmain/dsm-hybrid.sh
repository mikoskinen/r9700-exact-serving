#!/bin/bash
# dsm-hybrid.sh (10.11.2026, queue-ar/dsmain, ds41-gg agent) -- NOT a queue item; the body of 57-p4zyzzzzz-ds-t-hybrid.sh.
# THE LOOKUP / DSPARK HYBRID in process on the ds41-gg image (XGG_IMAGE, default radiance-local:ds41-gg-4675795f-5c13e555-l: 4675795's depth 0 + hybrid
# over 8b311fc's gg tables, lever D2's MR form, lookup alone, the DSpark top-32 walk). the hybrid (ds41-gg 4675795, Scheduler::hybrid_): a sequence whose prompt lookup matches >= XHY_MATCH tokens back
# (default 8 = LookupGate::kLongMatch; draftmr start 2 sizes it) verifies the lookup's proposal (the whole window) instead
# of DSpark's; the step behind a long match is planned after the commit (no step issued ahead). hz-ngram / draftmr S2:
# lookup alone wins the edit (re-emitted region), DSpark wins prose; the hybrid should keep DSpark's essay / code and take
# the edit toward lookup's. A = draft_hybrid 0, B = 1 (draft_hybrid_match pinned in both).
# ONE START, EXL3 2.25, TP4, MML 524288, the t3 env and serve line of dsm-draft.sh (105's t3 arm + lib-29 best_vars + MNBT
# 4160 + --live), DSpark --num-speculative-tokens 5 --draft-window adaptive, gg_gemv 2 from the start (RADIANCE_DS41_GG_GEMV=2)
# and pinned in every block: gate (70 ms) -> modes (the t3 notes, the DSpark line, routed experts in VRAM vs bz d0 / d5) ->
# the switch note (draft_hybrid as the scheduler reports it, 2 refused, 1 and back to 0 applied at a step) -> warm-up answers vs
# stock -> BLOCKS (ggblocksdz.py, XDZ_ORDER default A B B A, epoch_bump each block, edit in both arms): stepprobe
# essay,code 512 (c1), an essay-only stepprobe between two /stats snapshots (acceptance by draft position), editprobe (an
# edit re-emitting a ~2.5k-token region, 1500 tok), mb c4 and mb c8 (XDZ_DUR 75 s, warm 10), the scheduler's draft
# counters around every phase (cycles at 0, probe cycles, moves, hybrid lookup / head / not issued ahead) -> IDENTITY
# under B: the 28k copy (a ~31k-token cold prompt) and the 20-prompt greedy set at c1 vs bz's d0 (runs/xdsm/draft), byte
# for byte (a drafter only proposes; a difference is reported as such) -> the summary lines: B vs A pooled (pairs, t95) at
# c1 essay / code / edit (tok/s, tokens a step, acceptance, ms a step) and c4 / c8 (aggregate, per-stream p50, tokens a
# step, acceptance, ms a step); one line A / B vs NST 0 at gg_gemv 2 (gz-gemvacq's gg2-gemv2 B blocks, no re-run).
# NO cc_sim (user 10.11.). No power / host / fan settings; production compose files read only (--project-directory).
# DRYRUN=1 bash dsm-hybrid.sh : prints every command, runs and writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-gg-tp4.yaml
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=$R/xdsm/hybrid; export STATSDIR=$O/stats
RD=$R/xdsm/draft                                # bz's d0 / d5 / d5a files
RG=$R/xdsm/gg2                                  # gz-gemvacq's files (NST 0, gg_gemv 0 / 2 blocks)
export IMAGE=${XGG_IMAGE:-radiance-local:ds41-gg-4675795f-5c13e555-l}
ITEM=${XDZ_ITEM:-57-p4zyzzzzz-ds-t-hybrid}; ORDER=${XDZ_ORDER:-ABBA}; DUR=${XDZ_DUR:-75}; MATCH=${XHY_MATCH:-8}
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
EP=$HD/editprobe.py; GGB=$HD/ggblocksdz.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
SW=draft_hybrid
FIXED='{"gg_gemv": 2, "draft_adaptive": 1, "draft_depth0": 0, "draft_hybrid_match": '"$MATCH"'}'
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DRAFT_DEPTH0 RADIANCE_DRAFT_PROBE RADIANCE_DRAFT_HYBRID RADIANCE_DRAFT_HYBRID_MATCH"
t3env() {   # 105's t3 arm, every other switch unset; gg_gemv 2 from the start
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xhy $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xhy $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  case $r in IDENTICAL*) log "RESULT xhy $1 warm-up answers vs stock ($(basename $2)): $r (105 tuned3l: IDENTICAL)" ;;
    *) log "RESULT xhy $1 warm-up answers vs stock ($(basename $2)): VIRHE $r -- NOT EXACT vs the tuned3l line" ;; esac
}
modes() {   # modes TAG: the serve log's notes must be the t3 line's -> 0, else 1 (arm stopped)
  [ "$DRY" = 1 ] && { echo "+ modes $1: indexer note (mxfp4, latent bf16, compressor tail bound, router selection+sort separate, AR_PCN separate at 0 layers), ARGATHER note, staging line, no acquire-rule line, interleave"; return 0; }
  local MLF SL AL IL ok=1
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  AL=$(docker logs $CN 2>&1 | grep -m3 -E 'activation arena' | cut -c1-200 | tr '\n' '|')
  IL=$(grep -iE 'interleave' $O/start-$1.log 2>/dev/null | sed 's/^[A-Z] *//' | tr -s ' ' | head -3 | tr '\n' '|' | cut -c1-300)
  log "RESULT xhy $1 modes: $(echo "$MLF" | cut -c1-1100) | arena: $AL | staging: ${SL:-none} | interleave: ${IL:-no line}"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate" "post-collapse-norm separate at 0 layers" "next collapse-norm separate at 0 layers"; do
    case "$MLF" in *"$w"*) ;; *) log "RESULT xhy $1: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  docker logs $CN 2>&1 | grep -q 'the MoE gather and its all-reduce as one kernel' || { log "RESULT xhy $1: VIRHE no RADIANCE_DS41_ARGATHER note"; ok=0; }
  [ -n "$SL" ] || { log "RESULT xhy $1: VIRHE no prefill staging line (levels refused?)"; ok=0; }
  docker logs $CN 2>&1 | grep -q 'the acquire rule plays' && { log "RESULT xhy $1: VIRHE an acquire-rule line (RADIANCE_AQL_ACQ should be unset)"; ok=0; }
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
copyt() {   # copyt TAG OFFSET: the copy test after a ~31k-token cold prompt (105's prompt: same offset -> same text)
  local t0=$(date +%s)
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset $2 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xhy $1 copy after a cold prompt: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300) | $(thermline $t0 $(date +%s))"
}
statsum() {   # statsum DIR TAG: per-phase deltas of the /stats snapshots
  [ "$DRY" = 1 ] && { echo "+ RESULT xhy $2 stats per phase from $1/stats-*.json"; return 0; }
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
  log "RESULT xhy $2 stats per phase: $(grep '^PH ' $1/stats-phases.txt | cut -c4- | python3 -c 'import json,sys
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xhy $1 drafter lines"; return 0; }
  log "RESULT xhy $1 drafter / speculation lines: $(docker logs $CN 2>&1 | grep -iE 'dspark|drafter|speculat|draft-window|draft window|verify' | sed 's/^[A-Z][^ ]* //' | sort -u | head -8 | tr '\n' '|' | cut -c1-1500)"
}
probelog() {   # probelog TAG: the probe's two prompts with tokens a step and acceptance
  [ "$DRY" = 1 ] && return 0
  log "RESULT xhy $1 probe detail: $(grep '^J ' $O/$1-probe.txt | cut -c3- | python3 -c 'import json,sys
print("; ".join("%s %s tok/s %s ms/step %s tok/step acc %s" % (d.get("prompt"), d.get("decode_tps"), d.get("ms_per_decode_step"), d.get("tokens_per_step"), d.get("acceptance")) for d in map(json.loads, sys.stdin)))' 2>&1 | cut -c1-400)"
}
mbline() {   # mbline TAG PHASE: aggregate, per-stream p50, ms and tokens a step, acceptance
  [ "$DRY" = 1 ] && return 0
  log "RESULT xhy $1 mb $2: $(grep '^MB ' $O/$1-$2.txt 2>/dev/null | cut -c4- | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print("aggregate %s tok/s, per-stream p50 %s, %s ms a decode step, %s tokens a step, acceptance %s, TTFT p50 %s s, %s requests, %s errors" % (d.get("agg_dec_tps"), d.get("per_stream_dec_p50"), d.get("ms_decode_step"), d.get("tok_per_decode_step"), d.get("acceptance"), d.get("ttft_p50"), d.get("requests"), d.get("errors")))' 2>&1 | cut -c1-400)"
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
  log "RESULT xhy $1 editprobe (1 user, greedy, thinking off, ~2.5k-token stdlib region re-emitted with one docstring, 1500 tok): $(grep '^J ' $O/$1-edit.txt | cut -c3- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
print("%s tok/s, %s ms/step, %s tok/step, acceptance %s (%s drafted, %s accepted), echo ratio %s%s, %s tokens, finish %s, prompt %s tok, function %s" % (
    d.get("decode_tps"), d.get("ms_per_decode_step"), d.get("tokens_per_step"), d.get("acceptance"), d.get("drafted"), d.get("accepted"),
    d.get("echo_ratio"), "" if d.get("echo_ok") else " -- ECHO NOT: the reply does not re-emit the region (a refusal / summary / rewrite), not a copy measurement",
    d.get("tokens"), d.get("finish"), d.get("prompt_tokens"), d.get("function")))' 2>&1 | cut -c1-500)"
}
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
fn4code() {   # fn4code JSON: POST, print the HTTP status
  [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null
}

swnote() {   # swnote TAG: at start gg_gemv 2, adaptive, draft_hybrid 0; draft_hybrid 2 refused; 1 and back to 0 applied at a step
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (expect gg_gemv 2, draft_adaptive 1, draft_hybrid 0, draft_capable 1, epoch 0); POST draft_hybrid 2 -> 400; POST draft_hybrid 1 (hybrid: with draft_hybrid_match $MATCH) / draft_hybrid 0, each + a 4-token request -> applied"; return 0; }
  local s0 s1 s2 c2
  s0=$(fn4get); c2=$(fn4code '{"draft_hybrid": 2}')
  fn4set '{"draft_hybrid": 1, "draft_hybrid_match": '"$MATCH"'}'; nudge; s1=$(fn4get)
  fn4set '{"draft_hybrid": 0}'; nudge; s2=$(fn4get)
  log "RESULT xhy $1 switch note: at start $s0 | draft_hybrid 2 -> HTTP $c2 | after 1: $s1 | after 0: $s2"
  python3 -c 'import json,sys
s0, s1, s2 = (json.loads(x) for x in sys.argv[1:4])
w = s0["switches"]
ok = w["gg_gemv"] == 2 and w["draft_adaptive"] == 1 and w["draft_hybrid"] == 0 and s0["draft_capable"] == 1 and s0["epoch"] == 0 \
     and s1["switches"]["draft_hybrid"] == 1 and not s1["pending"] and s2["switches"]["draft_hybrid"] == 0 and not s2["pending"] and s2["epoch"] == 2
sys.exit(0 if ok else 1)' "$s0" "$s1" "$s2" 2>/dev/null && [ "$c2" = 400 ]
}
blocks() {   # blocks TAG: the A/B blocks (draft_hybrid 0 / 1, the rest pinned) and the per-block lines
  run "timeout 7200 python3 -I $GGB --url $URL --mb-dir $D --edit $EP --label $1 --switch $SW --a 0 --b 1 --order $ORDER --dur $DUR --warm 10 --fixed '$FIXED' --out $O/$1-blocks.jsonl > $O/$1-blocks.txt 2>&1"
  [ "$DRY" = 1 ] && return 0
  while IFS= read -r l; do log "RESULT xhy $1 ${l:0:1400}"; done < <(grep '^BLOCK ' $O/$1-blocks.txt)
}
summary() {   # summary TAG: B vs A pooled; A / B vs gz-gemvacq's NST 0 gg_gemv 2 blocks
  [ "$DRY" = 1 ] && { echo "+ RESULT xhy $1 summary: B (draft_hybrid 1) vs A (0) pooled + pairs + t95; A / B vs NST 0 gg_gemv 2 (gz gg2-gemv2 B blocks)"; return 0; }
  log "RESULT xhy $1 summary (DSpark NST 5 adaptive, gg_gemv 2; B = draft_hybrid 1 vs A = 0, in process, order $ORDER, mb $DUR s; fixed $FIXED): $(grep '^GGB ' $O/$1-blocks.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=2: "-" if v is None else format(v, ".%df" % n)
out = []
for k, v in d["metrics"].items():
    p = v.get("pooled")
    out.append("%s %s -> %s (%s %%; pairs %s; t95 +- %s %%)" % (k, f(v.get("A")), f(v.get("B")), "-" if p is None else "%+.2f" % (100 * (p - 1)),
               " ".join("%+.2f" % (100 * (x - 1)) for x in v.get("pairs", [])), "-" if v.get("ci95") is None else "%.2f" % (100 * v["ci95"])))
print("; ".join(out) + " | %d blocks, %d flagged, %s s" % (d["blocks"], d["blocks_bad"], d["wall_s"]))' 2>&1 | cut -c1-3600)"
  log "RESULT xhy $1 vs NST 0 gg_gemv 2 (gz-gemvacq's gg2-gemv2 B blocks, the same image line, no re-run): $(python3 - $O/$1-blocks.jsonl $RG/gg2-gemv2-blocks.jsonl <<'PY' 2>&1 | cut -c1-2400
import json, statistics, sys
def rows(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except OSError: return []
me, gz = rows(sys.argv[1]), rows(sys.argv[2])
def mean(rs, arm, get):
    v = [get(r["res"]) for r in rs if r.get("ok") and r.get("arm") == arm]
    v = [x for x in v if isinstance(x, (int, float))]
    return statistics.mean(v) if v else None
def c1m(name, k):
    return lambda res: (res.get("c1", {}).get(name) or {}).get(k)
def c1gz(name, k):
    return lambda res: next((x.get(k) for x in (res.get("c1", {}).get("rows") or []) if x.get("prompt") == name), None)
def ph(p, k):
    return lambda res: (res.get(p) or {}).get(k)
items = [("1u essay tok/s", c1m("essay", "decode_tps"), c1gz("essay", "decode_tps")),
         ("1u code tok/s", c1m("code", "decode_tps"), c1gz("code", "decode_tps")),
         ("1u edit tok/s", ph("edit", "decode_tps"), ph("edit", "decode_tps")),
         ("c4 aggregate", ph("c4", "agg_dec_tps"), ph("c4", "agg_dec_tps")), ("c4 p50", ph("c4", "per_stream_dec_p50"), ph("c4", "per_stream_dec_p50")),
         ("c8 aggregate", ph("c8", "agg_dec_tps"), ph("c8", "agg_dec_tps")), ("c8 p50", ph("c8", "per_stream_dec_p50"), ph("c8", "per_stream_dec_p50"))]
out = []
for nm, gm, gg in items:
    a, b, z = mean(me, "A", gm), mean(me, "B", gm), mean(gz, "B", gg)
    def r(x, y): return ("%+.1f %%" % ((x / y - 1) * 100)) if x and y else "-"
    out.append("%s: A %s / B %s vs NST0 %s (A %s, B %s)" % (nm, "-" if a is None else "%.1f" % a, "-" if b is None else "%.1f" % b,
               "-" if z is None else "%.1f" % z, r(a, z), r(b, z)))
print("; ".join(out) if gz else "no gz-gemvacq blocks file")
PY
)"
}
dstart() {   # dstart: the one start, its blocks, identity under B, its summary lines
  local tag=hy-dspark P=$O/hy-dspark rc ok
  t3env; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive"
  log "RESULT xhy $tag: start IMAGE=$IMAGE MODEL=$MODEL $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
  mkd $P
  gate $tag 70; rc=$?
  [ $rc = 1 ] && { log "RESULT xhy $tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  [ $rc = 2 ] && { log "RESULT xhy $tag: VIRHE over the gate limit after one restart -> start skipped"; down; return 1; }
  modes $tag || { log "RESULT xhy $tag: VIRHE the modes are not the t3 line's -> start stopped"; down; return 1; }
  draftlines $tag
  if [ "$DRY" = 1 ]; then echo "+ DSpark drafter line check; VRAM expert units at start vs d0 / d5"
  else
    ok=1
    docker logs $CN 2>&1 | grep -q "DSpark drafter" || { log "RESULT xhy $tag: VIRHE no DSpark drafter line"; ok=0; }
    docker logs $CN 2>&1 | grep -q "draft depth: adaptive" || { log "RESULT xhy $tag: VIRHE no adaptive draft depth line"; ok=0; }
    curl -s -m 15 $URL/stats > $O/$tag-stats0.json 2>/dev/null
    log "RESULT xhy $tag experts at start: $(vramunits $O/$tag-stats0.json) | bz d0 (NST 0): $(vramunits $RD/dsd-d0/stats-start.json) | bz d5 (NST 5, head placed): $(vramunits $RD/dsd-d5/stats-start.json)"
    [ $ok = 1 ] || { keeplog $tag-nodraft; down; return 1; }
  fi
  swnote $tag || { log "RESULT xhy $tag: VIRHE the switch is not as expected -> start stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  follow $P
  stopfile || blocks $tag
  if ! stopfile; then
    fn4set '{"draft_hybrid": 1, "draft_hybrid_match": '"$MATCH"', "gg_gemv": 2, "draft_adaptive": 1, "draft_depth0": 0, "epoch_bump": 1}'
    copyt $tag-i1 0; gsetc1 $tag
    [ "$DRY" = 1 ] || log "RESULT xhy $tag identity ran under: $(fn4get | cut -c1-400)"
    if [ "$DRY" = 1 ]; then echo "+ identity: 28k copy vs bz d0, greedy set c1 vs bz d0"
    else
      local r
      r=$(cmpfile $O/copy-$tag-i1-run1.txt $RD/copy-dsd-d0-i1-run1.txt)
      case $r in IDENTICAL*) log "RESULT xhy $tag identity (draft_hybrid 1, gg_gemv 2): 28k copy vs bz d0 (NST 0, stock): $r" ;;
        *) log "RESULT xhy $tag identity (draft_hybrid 1, gg_gemv 2): DIFFERENT 28k copy vs bz d0 (NST 0, stock): $r -- reported, not hidden" ;; esac
      r=$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$tag-gs-c1.jsonl)
      case $r in IDENTICAL*) log "RESULT xhy $tag identity (draft_hybrid 1, gg_gemv 2): greedy set c1 vs bz d0: $r" ;;
        *) log "RESULT xhy $tag identity (draft_hybrid 1, gg_gemv 2): DIFFERENT greedy set c1 vs bz d0: $r -- reported, not hidden" ;; esac
    fi
  fi
  unfollow
  run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
  down
  summary $tag
  log "RESULT xhy $tag: done"
}
# ---------------------------------------------------------------- main
log "RESULT xhy $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) switch $SW fixed $FIXED order $ORDER mb $DUR s | refs gz-gemvacq gg2-gemv2 (NST 0, gg_gemv 2 blocks), bz d0"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xhy: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xhy: VIRHE image $IMAGE not on gpu02"; exit 1; }
  [ -e $Y ] && grep -q '^ *RADIANCE_DS41_GG_GEMV:' $Y || { log "RESULT xhy: VIRHE $Y missing or does not pass RADIANCE_DS41_GG_GEMV"; exit 1; }
  for f in $GS $EP $GGB $D/stepprobe.py $D/mb.py $D/ds41_tests.py; do [ -e $f ] || { log "RESULT xhy: VIRHE $f missing"; exit 1; }; done
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xhy: VIRHE $M225 missing or not verified"; exit 1; }
  [ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -q "draft depth 0: a step drafts nothing" /opt/radiance/bin/radiance && grep -q "draft hybrid: a sequence whose prompt lookup" /opt/radiance/bin/radiance && grep -q r4d_gg_mr_pin /opt/radiance/share/radiance/kernels/libr4d.so && echo yes' 2>/dev/null)" = yes ] \
    || { log "RESULT xhy: VIRHE $IMAGE lacks depth 0 / the hybrid or lever D2"; exit 1; }
  for f in $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $RG/gg2-gemv2-blocks.jsonl; do [ -s $f ] || log "RESULT xhy: note -- reference $f missing (not compared)"; done
fi
mkd $O $O/guard $STATSDIR
evict
stopfile || dstart
unset $SWITCHES
down
log "RESULT xhy $ITEM: done -- $O"
