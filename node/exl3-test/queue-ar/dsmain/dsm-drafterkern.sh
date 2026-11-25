#!/bin/bash
# dsm-drafterkern.sh (10.11.2026, queue-ar/dsmain, ds41-drk agent) -- NOT a queue item; the body of 57-p4zyzzzzz-ds-p-drafterkern.sh.
# The DSpark drafter's kernels in process on the ds41-drk image (XDK_IMAGE; ds41-gg 8b311fc + ds41-drk: the core's fn4 switches
# drafter_mx / drafter_bf, libr4d's MXFP4 grouped-GEMM multi-row forms and bf16 skinny-GEMM prefetching form). Both keep the
# stock kernels' bits, so the drafter proposes the same tokens: tokens a step and acceptance must not move, only time. The
# drafter pass was ~5.5 ms real a 1-user round (profiled: moe_gemm_ggml[drafter] 472 us x 6 = 2.83 ms, gemm_nt 1.43 ms a step).
# ONE START, EXL3 2.25, TP4, MML 524288, dsm-draftmr.sh's t3 env and serve line (105's t3 arm + lib-29 best_vars + MNBT 4160 +
# --live), DSpark NST 5 --draft-window adaptive, RADIANCE_DS41_GG_GEMV=2 (lever D2 with its measured table) from the start:
#   gate (limit 70 ms) -> modes (the t3 notes) -> drafter lines, experts in VRAM vs bz d0 / d5 -> the switch note (drafter_mx /
#   drafter_bf 0 at start as libr4d reports them, 2 refused, 1 and back to 0 applied at a step, gg_gemv 2) -> warm-up answers
#   vs stock -> per switch in XDK_SWITCHES (default "drafter_mx drafter_bf"), blocks A B B A (XDK_ORDER; A = 0, B = 1, the other
#   switch 0 and gg_gemv 2 fixed; epoch_bump each block; ggblocksdr.py): stepprobe essay,code 512 (c1), an essay-only probe
#   with acceptance by draft position, editprobe (XDK_EDIT_A=0 skips it in A blocks), mb c4 and mb c8 (XDK_DUR 75 s, warm 10);
#   every block's line (tokens a step and acceptance per arm) logged; the switch's summary (B vs A pooled, pairs, t95) and
#   tokens a step / acceptance A vs B -> IDENTITY with both switches on (gg_gemv 2): the 28k copy and the 20-prompt greedy
#   set at c1 vs bz's d0 (runs/xdsm/draft), byte for byte.
# NO cc_sim. No power / host / fan settings; production compose files read only.
# DRYRUN=1 bash dsm-drafterkern.sh : prints every command, runs and writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-drk-tp4.yaml
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=$R/xdsm/drafterkern; export STATSDIR=$O/stats
RD=$R/xdsm/draft                                # bz's d0 / d5 / d5a files
RG=$R/xdsm/gg2                                  # gz-gemvacq's files (NST 0, gg_gemv 0 / 2 blocks)
export IMAGE=${XDK_IMAGE:-radiance-local:ds41-drk-a22a5dfa-5c13e555-l}
ITEM=${XDK_ITEM:-57-p4zyzzzzz-ds-p-drafterkern}; ORDER=${XDK_ORDER:-ABBA}; DUR=${XDK_DUR:-75}; EDITA=${XDK_EDIT_A:-1}
KSW=${XDK_SWITCHES-"drafter_mx drafter_bf"}
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
EP=$HD/editprobe.py; GGB=$HD/ggblocksdr.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DS41_DRAFTER_MX RADIANCE_DS41_DRAFTER_BF"
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xdk $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xdk $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  case $r in IDENTICAL*) log "RESULT xdk $1 warm-up answers vs stock ($(basename $2)): $r (105 tuned3l: IDENTICAL)" ;;
    *) log "RESULT xdk $1 warm-up answers vs stock ($(basename $2)): VIRHE $r -- NOT EXACT vs the tuned3l line" ;; esac
}
modes() {   # modes TAG: the serve log's notes must be the t3 line's -> 0, else 1 (arm stopped)
  [ "$DRY" = 1 ] && { echo "+ modes $1: indexer note (mxfp4, latent bf16, compressor tail bound, router selection+sort separate, AR_PCN separate at 0 layers), ARGATHER note, staging line, no acquire-rule line, interleave"; return 0; }
  local MLF SL AL IL ok=1
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  AL=$(docker logs $CN 2>&1 | grep -m3 -E 'activation arena' | cut -c1-200 | tr '\n' '|')
  IL=$(grep -iE 'interleave' $O/start-$1.log 2>/dev/null | sed 's/^[A-Z] *//' | tr -s ' ' | head -3 | tr '\n' '|' | cut -c1-300)
  log "RESULT xdk $1 modes: $(echo "$MLF" | cut -c1-1100) | arena: $AL | staging: ${SL:-none} | interleave: ${IL:-no line}"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate" "post-collapse-norm separate at 0 layers" "next collapse-norm separate at 0 layers"; do
    case "$MLF" in *"$w"*) ;; *) log "RESULT xdk $1: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  docker logs $CN 2>&1 | grep -q 'the MoE gather and its all-reduce as one kernel' || { log "RESULT xdk $1: VIRHE no RADIANCE_DS41_ARGATHER note"; ok=0; }
  [ -n "$SL" ] || { log "RESULT xdk $1: VIRHE no prefill staging line (levels refused?)"; ok=0; }
  docker logs $CN 2>&1 | grep -q 'the acquire rule plays' && { log "RESULT xdk $1: VIRHE an acquire-rule line (RADIANCE_AQL_ACQ should be unset)"; ok=0; }
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
copyt() {   # copyt TAG OFFSET: the copy test after a ~31k-token cold prompt (105's prompt: same offset -> same text)
  local t0=$(date +%s)
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset $2 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xdk $1 copy after a cold prompt: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300) | $(thermline $t0 $(date +%s))"
}
statsum() {   # statsum DIR TAG: per-phase deltas of the /stats snapshots
  [ "$DRY" = 1 ] && { echo "+ RESULT xdk $2 stats per phase from $1/stats-*.json"; return 0; }
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
  log "RESULT xdk $2 stats per phase: $(grep '^PH ' $1/stats-phases.txt | cut -c4- | python3 -c 'import json,sys
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xdk $1 drafter lines"; return 0; }
  log "RESULT xdk $1 drafter / speculation lines: $(docker logs $CN 2>&1 | grep -iE 'dspark|drafter|speculat|draft-window|draft window|verify' | sed 's/^[A-Z][^ ]* //' | sort -u | head -8 | tr '\n' '|' | cut -c1-1500)"
}
probelog() {   # probelog TAG: the probe's two prompts with tokens a step and acceptance
  [ "$DRY" = 1 ] && return 0
  log "RESULT xdk $1 probe detail: $(grep '^J ' $O/$1-probe.txt | cut -c3- | python3 -c 'import json,sys
print("; ".join("%s %s tok/s %s ms/step %s tok/step acc %s" % (d.get("prompt"), d.get("decode_tps"), d.get("ms_per_decode_step"), d.get("tokens_per_step"), d.get("acceptance")) for d in map(json.loads, sys.stdin)))' 2>&1 | cut -c1-400)"
}
mbline() {   # mbline TAG PHASE: aggregate, per-stream p50, ms and tokens a step, acceptance
  [ "$DRY" = 1 ] && return 0
  log "RESULT xdk $1 mb $2: $(grep '^MB ' $O/$1-$2.txt 2>/dev/null | cut -c4- | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print("aggregate %s tok/s, per-stream p50 %s, %s ms a decode step, %s tokens a step, acceptance %s, TTFT p50 %s s, %s requests, %s errors" % (d.get("agg_dec_tps"), d.get("per_stream_dec_p50"), d.get("ms_decode_step"), d.get("tok_per_decode_step"), d.get("acceptance"), d.get("ttft_p50"), d.get("requests"), d.get("errors")))' 2>&1 | cut -c1-400)"
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
  log "RESULT xdk $1 editprobe (1 user, greedy, thinking off, ~2.5k-token stdlib region re-emitted with one docstring, 1500 tok): $(grep '^J ' $O/$1-edit.txt | cut -c3- | python3 -c 'import json,sys
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
swnote() {   # swnote TAG: drafter_mx / drafter_bf at start (0, as libr4d reports them), 2 refused, 1 and back to 0 applied, gg_gemv 2
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (expect drafter_mx 0, drafter_bf 0, gg_gemv 2); per switch POST 2 -> 400, POST 1 / 0 each + a 4-token request -> applied"; return 0; }
  local s0 s1 s2 s3 s4 c3 c4
  s0=$(fn4get); c3=$(fn4code '{"drafter_mx": 2}'); c4=$(fn4code '{"drafter_bf": 2}')
  fn4set '{"drafter_mx": 1}'; nudge; s1=$(fn4get)
  fn4set '{"drafter_mx": 0}'; nudge; s2=$(fn4get)
  fn4set '{"drafter_bf": 1}'; nudge; s3=$(fn4get)
  fn4set '{"drafter_bf": 0}'; nudge; s4=$(fn4get)
  log "RESULT xdk $1 switch note: at start $s0 | drafter_mx 2 -> HTTP $c3, drafter_bf 2 -> HTTP $c4 | mx 1: $s1 | mx 0: $s2 | bf 1: $s3 | bf 0: $s4"
  python3 -c 'import json,sys
s = [json.loads(x) for x in sys.argv[1:6]]
w = lambda d, k: d["switches"].get(k)
ok = w(s[0], "drafter_mx") == 0 and w(s[0], "drafter_bf") == 0 and w(s[0], "gg_gemv") == 2 and s[0]["plugins"] >= 1 \
     and w(s[1], "drafter_mx") == 1 and w(s[2], "drafter_mx") == 0 and w(s[3], "drafter_bf") == 1 and w(s[4], "drafter_bf") == 0 \
     and all(not d["pending"] for d in s[1:]) and s[4]["epoch"] == s[0]["epoch"] + 4
sys.exit(0 if ok else 1)' "$s0" "$s1" "$s2" "$s3" "$s4" 2>/dev/null && [ "$c3" = 400 ] && [ "$c4" = 400 ]
}
blocks() {   # blocks TAG SWITCH: A/B blocks (SWITCH 0 / 1, the other drafter switch 0 and gg_gemv 2 fixed) and the per-block lines
  local tag=$1 sw=$2 other fx
  other=$([ $sw = drafter_mx ] && echo drafter_bf || echo drafter_mx)
  fx="{\"gg_gemv\": 2, \"$other\": 0}"
  run "timeout 7200 python3 -I $GGB --url $URL --mb-dir $D --edit $EP --label $tag-$sw --switch $sw --a 0 --b 1 --order $ORDER --dur $DUR --warm 10 $( [ "$EDITA" = 1 ] || echo --no-edit-a) --fixed '$fx' --out $O/$tag-$sw-blocks.jsonl > $O/$tag-$sw-blocks.txt 2>&1"
  [ "$DRY" = 1 ] && return 0
  while IFS= read -r l; do log "RESULT xdk $tag $sw ${l:0:900}"; done < <(grep '^BLOCK ' $O/$tag-$sw-blocks.txt)
}
summary() {   # summary TAG SWITCH: B (1) vs A (0) pooled, pairs, t95; tokens a step and acceptance A vs B (equal by construction)
  [ "$DRY" = 1 ] && { echo "+ RESULT xdk $1 $2 summary: B (1) vs A (0) pooled + pairs + t95; tokens a step / acceptance A vs B"; return 0; }
  log "RESULT xdk $1 $2 summary (B = $2 1 vs A = 0, gg_gemv 2, in process, order $ORDER, mb $DUR s): $(grep '^GGB ' $O/$1-$2-blocks.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=2: "-" if v is None else format(v, ".%df" % n)
out = []
for k, v in d["metrics"].items():
    p = v.get("pooled")
    out.append("%s %s -> %s (%s %%; pairs %s; t95 +- %s %%)" % (k, f(v.get("A")), f(v.get("B")), "-" if p is None else "%+.2f" % (100 * (p - 1)),
               " ".join("%+.2f" % (100 * (x - 1)) for x in v.get("pairs", [])), "-" if v.get("ci95") is None else "%.2f" % (100 * v["ci95"])))
print("; ".join(out) + " | %d blocks, %d flagged, %s s" % (d["blocks"], d["blocks_bad"], d["wall_s"]))' 2>&1 | cut -c1-3600)"
  log "RESULT xdk $1 $2 tokens a step / acceptance A vs B (the drafter's bits do not change, so these must match within noise of which steps ran): $(python3 - $O/$1-$2-blocks.jsonl <<'PY' 2>&1 | cut -c1-1500
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
def m(arm, get):
    v = [get(r["res"]) for r in rows if r.get("ok") and r.get("arm") == arm]
    v = [x for x in v if isinstance(x, (int, float))]
    return "-" if not v else "%.3f" % statistics.mean(v)
out = []
for nm, g in (("essay", lambda r: (r.get("c1", {}).get("essay") or {})), ("code", lambda r: (r.get("c1", {}).get("code") or {})),
              ("edit", lambda r: (r.get("edit") or {}))):
    out.append("%s tok/step A %s B %s acc A %s B %s" % (nm, m("A", lambda r, g=g: g(r).get("tokens_per_step")), m("B", lambda r, g=g: g(r).get("tokens_per_step")),
               m("A", lambda r, g=g: g(r).get("acceptance")), m("B", lambda r, g=g: g(r).get("acceptance"))))
for p in ("c4", "c8"):
    out.append("%s tok/step A %s B %s acc A %s B %s" % (p, m("A", lambda r, p=p: (r.get(p) or {}).get("tok_per_decode_step")),
               m("B", lambda r, p=p: (r.get(p) or {}).get("tok_per_decode_step")), m("A", lambda r, p=p: (r.get(p) or {}).get("acceptance")),
               m("B", lambda r, p=p: (r.get(p) or {}).get("acceptance"))))
print("; ".join(out))
PY
)"
}
dstart() {   # the one start: DSpark NST 5 adaptive, gg_gemv 2
  local tag=dk-dspark P=$O/dk-dspark rc ok s
  t3env; export MODEL=$M225; HP29=19456; best_vars
  export RADIANCE_DS41_GG_GEMV=2 MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive"
  log "RESULT xdk $tag: start IMAGE=$IMAGE MODEL=$MODEL $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
  mkd $P
  gate $tag 70; rc=$?
  [ $rc = 1 ] && { log "RESULT xdk $tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  [ $rc = 2 ] && { log "RESULT xdk $tag: VIRHE over the gate limit after one restart -> start skipped"; down; return 1; }
  modes $tag || { log "RESULT xdk $tag: VIRHE the modes are not the t3 line's -> start stopped"; down; return 1; }
  draftlines $tag
  if [ "$DRY" = 1 ]; then echo "+ drafter line check; VRAM expert units at start vs d0 / d5"
  else
    ok=1
    docker logs $CN 2>&1 | grep -q "DSpark drafter" || { log "RESULT xdk $tag: VIRHE no DSpark drafter line"; ok=0; }
    docker logs $CN 2>&1 | grep -q "multi-row form (lever D2; RADIANCE_DS41_GG_GEMV=2" || { log "RESULT xdk $tag: VIRHE no gg_gemv 2 line"; ok=0; }
    curl -s -m 15 $URL/stats > $O/$tag-stats0.json 2>/dev/null
    log "RESULT xdk $tag experts at start: $(vramunits $O/$tag-stats0.json) | bz d0 (NST 0): $(vramunits $RD/dsd-d0/stats-start.json) | bz d5 (NST 5, head placed): $(vramunits $RD/dsd-d5/stats-start.json)"
    [ $ok = 1 ] || { keeplog $tag-nodraft; down; return 1; }
  fi
  swnote $tag || { log "RESULT xdk $tag: VIRHE the drafter switches are not as expected -> start stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  follow $P
  for s in $KSW; do stopfile && break; blocks $tag $s; done
  if ! stopfile; then
    fn4set '{"gg_gemv": 2, "drafter_mx": 1, "drafter_bf": 1, "epoch_bump": 1}'
    copyt $tag-i1 0; gsetc1 $tag
    [ "$DRY" = 1 ] || log "RESULT xdk $tag identity ran under: $(fn4get | cut -c1-260)"
    if [ "$DRY" = 1 ]; then echo "+ identity (drafter_mx 1, drafter_bf 1, gg_gemv 2): 28k copy vs bz d0, greedy set c1 vs bz d0"
    else
      local r
      r=$(cmpfile $O/copy-$tag-i1-run1.txt $RD/copy-dsd-d0-i1-run1.txt)
      case $r in IDENTICAL*) log "RESULT xdk $tag identity (drafter_mx 1, drafter_bf 1, gg_gemv 2): 28k copy vs bz d0 (NST 0, stock): $r" ;;
        *) log "RESULT xdk $tag identity (drafter_mx 1, drafter_bf 1, gg_gemv 2): DIFFERENT 28k copy vs bz d0 (NST 0, stock): $r -- reported, not hidden" ;; esac
      r=$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$tag-gs-c1.jsonl)
      case $r in IDENTICAL*) log "RESULT xdk $tag identity (drafter_mx 1, drafter_bf 1, gg_gemv 2): greedy set c1 vs bz d0: $r" ;;
        *) log "RESULT xdk $tag identity (drafter_mx 1, drafter_bf 1, gg_gemv 2): DIFFERENT greedy set c1 vs bz d0: $r -- reported, not hidden" ;; esac
    fi
  fi
  unfollow
  run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
  down
  for s in $KSW; do summary $tag $s; done
  log "RESULT xdk $tag: done"
}
# ---------------------------------------------------------------- main
log "RESULT xdk $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) switches [$KSW] order $ORDER mb $DUR s edit in A blocks $EDITA | refs bz d0 / d5"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xdk: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xdk: VIRHE image $IMAGE not on gpu02"; exit 1; }
  [ -e $Y ] && grep -q '^ *RADIANCE_DS41_GG_GEMV:' $Y || { log "RESULT xdk: VIRHE $Y missing or does not pass RADIANCE_DS41_GG_GEMV"; exit 1; }
  for f in $GS $EP $GGB $D/stepprobe.py $D/mb.py $D/ds41_tests.py; do [ -e $f ] || { log "RESULT xdk: VIRHE $f missing"; exit 1; }; done
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xdk: VIRHE $M225 missing or not verified"; exit 1; }
  [ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -q drafter_mx /opt/radiance/bin/radiance && grep -q r4d_mx_mr_pin /opt/radiance/share/radiance/kernels/libr4d.so && grep -q r4d_bf_pf_pin /opt/radiance/share/radiance/kernels/libr4d.so && echo yes' 2>/dev/null)" = yes ] \
    || { log "RESULT xdk: VIRHE $IMAGE lacks the drafter switches or kernels"; exit 1; }
  for f in $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl; do [ -s $f ] || log "RESULT xdk: note -- reference $f missing (not compared)"; done
fi
mkd $O $O/guard $STATSDIR
evict
dstart
unset $SWITCHES
down
log "RESULT xdk $ITEM: done -- $O"
