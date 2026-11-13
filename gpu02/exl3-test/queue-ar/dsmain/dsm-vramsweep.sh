#!/bin/bash
# dsm-vramsweep.sh (10.11.2026, queue-ar/dsmain, DS VRAM-budget audit agent) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-pz-vramsweep.sh. Helpers copied from dsm-span4.sh (aa91b3f1), dsm-draftmr.sh (055848b7) and
# dsm-gemvacq.sh (e0507cb5); those bodies are not edited. Log tag xvs.
# DeepSeek-V4.1 Flash EXL3 2.25 bpw, gpu02, all 4 cards, TP4, MML 524288, NST 0, the t3 env + lib-29 best_vars + MNBT 4160 +
# --live (= every DS main-line item), on the ds41-gg image (XVS_IMAGE, default 8b311fcc: the image gz-gemvacq and draftmr
# ran), the stack switched on in process after the gate: gg_gemv 2 (D2) + aql_acq 2 (rule-w) + epoch_bump.
#
# WHAT: routed-expert residency at decode, by the one exact start-time knob that moves it without a build:
#   RADIANCE_PLACE_TURNOVER = the lent slots a slab class keeps FREE for swaps (engine_bringup.cpp mover_turnover_from_env;
#   mover.cpp: n_turn = min(value, 160) a class). Today (HEAT_MOVES 16, unset) = 3 x 16 - 32 = 16 a class = 32 kept
#   free (start log "(64 shadow, 32 more kept free out of a loan)"). At decode a rank holds 1699 owned + 981 KV-lent +
#   299 arena-lent - 32 kept free = 2948 routed units (dsm-225h /stats: probe 2948, c4 2943, c8 2941 of 3840).
#   ARMS (XVS_ARMS, default "T0 T150"; Tn = RADIANCE_PLACE_TURNOVER=n):
#     T0    the lever: 0 kept free -> +32 resident units a rank (~+0.29 GiB of experts), swaps land in the 32 shadow
#           slots a class only (move rate capped ~10 a step a class; measured promotions today 1.5 / 3.3 / 6.2 a step
#           at c1 / c4 / c8 on rank 0, both classes).
#     T150  the CALIBRATION (not a candidate): 300 kept free -> -268 units vs today (-300 vs T0) at the same move
#           budget (16 a step needs <= 48 free a class, so the extra free slots add no agility). T150 vs T0 gives
#           today's line's ms per resident expert at c1 / c4 / c8 -- the number that converts every VRAM lever that
#           needs a build (arena decode level, shadow slots, xprefetch ring, router bf16, ...) from GiB into ms.
#     T16   today's default -- NOT run unless named (TESTING rule 2: no baseline re-runs; no fresh-start reference
#           with gg_gemv 2 + aql_acq 2 exists, so T0's own gain is read as the T150-T0 slope x 32 units, and the gz
#           gg2-gemv2 blocks are quoted for information).
# EXACT CLAIM: placement only (which tier a unit is read from); the MoE sum folds slots in top-k order and rank
#   partials in rank order whatever the tier (LEVERS D3 correction 10.11.), and every earlier placement change kept the
#   bytes (e-hold arena loans, f0-dplace drafter on host: copy + 20 greedy IDENTICAL). Checked here per arm: warm-up
#   answers vs stock, the 28k copy and the 20-prompt greedy set at c1 vs bz's d0 (runs/xdsm/draft, main image, stock),
#   and T150 vs T0 byte for byte. A difference is a VIRHE line, not hidden.
# PER ARM, decode before prefill (TESTING 9.11.): gate (restart, warm-up incl. a 16k prefill, 1-user fastcheck; one
#   retry over 23 ms) -> modes (the t3 notes + the mover's "N more kept free" = 2 x min(n, 160)) -> state probe (mb c8
#   45 s at gg_gemv 0, vs 105's 43.1 ms; slow -> one restart) -> switch (gg_gemv 2, aql_acq 2, epoch_bump; GET must show
#   them applied) -> warm-up answers vs stock -> /stats + probe c1 (essay, code 512) -> mb c4 -> mb c8 -> fresh 136k
#   prefill -> 28k copy -> greedy set c1 -> identity vs bz d0 -> per-phase /stats (routed units in VRAM, rank-0 hit,
#   misses and promotions a step).
# THEN: T150 vs T0 (% and the slope: ms a step per 100 resident units at c1 / c4 / c8), T0 vs the gz gg2-gemv2 blocks
#   (information), identity summary. ~33 min (each arm ~4 min start + ~2 gate + 1 state + ~8 decode + ~3 prefill side).
# NO cc_sim (user 10.11.), no 2.9. No power / host / fan settings; production compose files read only (--project-directory).
# DRYRUN=1 bash dsm-vramsweep.sh : prints every command, runs and writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-gg-tp4.yaml
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=$R/xdsm/vramsweep; export STATSDIR=$O/stats
RD=$R/xdsm/draft                                # bz's d0 (main image, NST 0, stock) identity files + stats
RH=$R/xdsm/dsm-225h                             # e-hold's main-line arm: today's counters (2948 units at decode)
export IMAGE=${XVS_IMAGE:-radiance-local:ds41-gg-8b311fcc-5c13e555-l}
ITEM=${XVS_ITEM:-57-p4zyzzzzz-ds-pz-vramsweep}
ARMS=${XVS_ARMS:-T0 T150}
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
DEPTH=${XVS_DEPTH:-136000}
REF_GZ="gz gg2-gemv2 B blocks (NST 0, gg_gemv 2, aql_acq 0, in process, mb 75 s; 10.11. 12:11): c1 63.59 tok/s 15.72 ms, c4 131.73 tok/s 25.73 ms, c8 170.03 tok/s 38.93 ms"
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV"
# ---------------------------------------------------------------- helpers (dsm-span4 / draftmr / gemvacq, copied; tag xvs)
t3env() {   # 105's t3 arm, every other switch unset
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
}
modelvars225() {   # -> MODEL, HP29, LIM, WARMREF, C8REF (105's c8 ms a step); lib-29 best_vars (NST 0) + MNBT 4160 + --live
  export MODEL=$M225; HP29=19456; LIM=23; WARMREF=$WARM_REF_225; C8REF=43.1
  best_vars; export MNBT=4160 XARGS="$XARGS --live"
}
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
  log "RESULT guard $tag: $st, 1-user ${ms} ms/step (limit $lim; 105 tuned3l 225: 15.81; gg_gemv 0 at this point)"
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xvs $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xvs $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  case $r in IDENTICAL*) log "RESULT xvs $1 warm-up answers vs stock ($(basename $2)): $r (105 tuned3l: IDENTICAL)" ;;
    *) log "RESULT xvs $1 warm-up answers vs stock ($(basename $2)): VIRHE $r -- NOT EXACT vs the tuned3l line" ;; esac
}
modes() {   # modes TAG N: the t3 line's notes + the mover's kept-free count (2 x min(N, 160)) -> 0, else 1 (arm stopped)
  local want=$(( 2 * ($2 < 160 ? $2 : 160) ))
  [ "$DRY" = 1 ] && { echo "+ modes $1: indexer note (mxfp4, latent bf16, compressor tail bound, router selection+sort separate, AR_PCN separate at 0 layers), ARGATHER note, staging line, no acquire-rule line at start, interleave; mover line '(64 shadow, $want more kept free out of a loan)' on every rank"; return 0; }
  local MLF SL AL IL MV ok=1
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  AL=$(docker logs $CN 2>&1 | grep -E 'activation arena|lent out of the (KV cache|activation arena)' | sort -u | cut -c1-220 | tr '\n' '|')
  IL=$(grep -iE 'interleave' $O/start-$1.log 2>/dev/null | sed 's/^[A-Z] *//' | tr -s ' ' | head -3 | tr '\n' '|' | cut -c1-300)
  MV=$(docker logs $CN 2>&1 | grep -E '^I\[[0-9]\] mover: [0-9]+ slab slots' | cut -c1-160 | tr '\n' '|')
  log "RESULT xvs $1 modes: $(echo "$MLF" | cut -c1-1100) | arena / loans: $AL | staging: ${SL:-none} | interleave: ${IL:-no line} | mover: $MV"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate" "post-collapse-norm separate at 0 layers" "next collapse-norm separate at 0 layers"; do
    case "$MLF" in *"$w"*) ;; *) log "RESULT xvs $1: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  docker logs $CN 2>&1 | grep -q 'the MoE gather and its all-reduce as one kernel' || { log "RESULT xvs $1: VIRHE no RADIANCE_DS41_ARGATHER note"; ok=0; }
  [ -n "$SL" ] || { log "RESULT xvs $1: VIRHE no prefill staging line (levels refused?)"; ok=0; }
  docker logs $CN 2>&1 | grep -q 'the acquire rule plays' && { log "RESULT xvs $1: VIRHE an acquire-rule line at start (RADIANCE_AQL_ACQ should be unset; aql_acq 2 comes by switch)"; ok=0; }
  [ "$(docker logs $CN 2>&1 | grep -cE "^I\[[0-3]\] mover: [0-9]+ slab slots \(64 shadow, $want more kept free out of a loan\)")" = 4 ] \
    || { log "RESULT xvs $1: VIRHE the mover does not keep $want slots free on all 4 ranks (RADIANCE_PLACE_TURNOVER=$2 not applied?): $MV"; ok=0; }
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
stackon() {   # stackon TAG: gg_gemv 2 + aql_acq 2 + epoch_bump, applied at a step -> 0, else 1
  [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 {gg_gemv 2, aql_acq 2, epoch_bump 1} + a 4-token request; GET must show gg_gemv 2, aql_acq 2, nothing pending, plugins >= 1"; return 0; }
  local s0 s1
  s0=$(fn4get)
  fn4set '{"gg_gemv": 2, "aql_acq": 2, "epoch_bump": 1}'; nudge; s1=$(fn4get)
  log "RESULT xvs $1 stack switch: at start $s0 | after gg_gemv 2 + aql_acq 2: $s1"
  python3 -c 'import json,sys
s0, s1 = (json.loads(x) for x in sys.argv[1:3])
ok = s0["switches"]["gg_gemv"] == 0 and s0["switches"]["aql_acq"] == 0 and s1["switches"]["gg_gemv"] == 2 \
     and s1["switches"]["aql_acq"] == 2 and s1["plugins"] >= 1 and not s1["pending"]
sys.exit(0 if ok else 1)' "$s0" "$s1" 2>/dev/null
}
mbms() { grep '^MB ' $1 2>/dev/null | head -1 | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()[3:]).get("ms_decode_step", ""))' 2>/dev/null; }
stprobe() {   # stprobe TAG N: mb c8 45 s at gg_gemv 0 -> "fast|between|slow MS (ref REF)"; T150 expects ~+5-7 % (fewer units)
  [ "$DRY" = 1 ] && { echo "+ (cd $D && timeout 400 python3 mb.py --tag xvs-$1-st$2 --phases c8 --dur ${XVS_STATE_DUR:-45} --warm 10 > $O/$1-stprobe$2.txt)" >&2; echo "fast (dry)"; return 0; }
  run "(cd $D && timeout 400 python3 mb.py --tag xvs-$1-st$2 --phases c8 --dur ${XVS_STATE_DUR:-45} --warm 10 > $O/$1-stprobe$2.txt 2>&1)"
  local ms; ms=$(mbms $O/$1-stprobe$2.txt)
  [ -n "$ms" ] || { echo "unknown (no mb line)"; return 0; }
  awk -v m=$ms -v r=$C8REF 'BEGIN{s=(m <= 1.07*r) ? "fast" : (m >= 1.14*r) ? "slow" : "between"; printf "%s %s ms (105 tuned3l c8 %s ms, x%.3f)\n", s, m, r, m/r}'
}
copyt() {   # copyt TAG OFFSET: the copy test after a ~31k-token cold prompt (105's prompt: same offset -> same text)
  local t0=$(date +%s)
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset $2 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xvs $1 copy after a cold prompt: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300) | $(thermline $t0 $(date +%s))"
}
pfd() {   # pfd TAG DEPTH: one fresh prefill of DEPTH tokens (nonce first, cache_prompt off) and its thermal line
  local t0=$(date +%s)
  run "(cd $D && timeout 1800 python3 ds41_tests.py prefill --depths $2 --cpt 3.95 --calibrate 0 --reps 1 > $O/$1-pf$2.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xvs $1 fresh prefill $2: $(grep '^J ' $O/$1-pf$2.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); print("%s tok, cached %s, TTFT %s s, %s tok/s, %s" % (d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("prefill_tps"), d.get("status")))' 2>&1 | cut -c1-200) | $(thermline $t0 $(date +%s))"
}
gsetc1() { run "(cd $D && timeout 900 python3 $GS $URL $O/$1-gs-c1.jsonl --conc 1 --max-tokens 192 > $O/$1-gs-c1.txt 2>&1)"; }
gcmp() { [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }; [ -s "$1" ] && [ -s "$2" ] || { echo "not compared (missing $1 or $2)"; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
idline() {   # idline LABEL RESULT: IDENTICAL -> plain, else VIRHE
  case "$2" in IDENTICAL*) log "RESULT xvs identity $1: $2" ;; "not compared"*) log "RESULT xvs identity $1: $2" ;;
    *) log "RESULT xvs identity $1: VIRHE $2 -- placement and both stack switches are exact by construction, so this is a defect (or a reference from another line)" ;; esac
}
statsum() {   # statsum DIR TAG: per-phase deltas of the /stats snapshots (+ rank-0 routed / missed / promotions a step)
  [ "$DRY" = 1 ] && { echo "+ RESULT xvs $2 stats per phase from $1/stats-*.json"; return 0; }
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
    dr = eb.get("routed", 0) - ea.get("routed", 0); drr = eb.get("routed_res", 0) - ea.get("routed_res", 0)
    o = {"phase": pb, "steps": ds, "prefill_mixed_steps": dpm, "decode_steps": dd,
         "ms_step": round((b["step_ns"] - a["step_ns"]) / ds / 1e6, 2),
         "ms_decode_step": round((b["step_ns_decode"] - a["step_ns_decode"]) / max(1, dd) / 1e6, 2) if dd else None,
         "ms_prefill_mixed_step": round(((b["step_ns"] - b["step_ns_decode"]) - (a["step_ns"] - a["step_ns_decode"])) / max(1, dpm) / 1e6, 1) if dpm else None,
         "hit_pct_rank0": round(100 * drr / dr, 2) if dr else None,
         "routed_step_rank0": round(dr / ds, 1), "missed_step_rank0": round((dr - drr) / ds, 2),
         "promotions_step": round((eb.get("promotions", 0) - ea.get("promotions", 0)) / ds, 2),
         "stream_MB_step": round((lb.get("stream_bytes", 0) - la.get("stream_bytes", 0)) / ds / 1e6, 1),
         "h2d_MB_step": round((lb.get("h2d_bytes", 0) - la.get("h2d_bytes", 0)) / ds / 1e6, 1),
         "stage_layers": eb.get("stage_layers", 0) - ea.get("stage_layers", 0),
         "staged_GB": round((eb.get("staged_bytes", 0) - ea.get("staged_bytes", 0)) / G, 2),
         "stage_overflow": eb.get("stage_overflow", 0) - ea.get("stage_overflow", 0),
         "vram_units_end": next((t.get("units") for t in eb.get("tiers", []) if t.get("name") == "vram"), None)}
    print("PH " + json.dumps(o))
PY
  log "RESULT xvs $2 stats per phase: $(grep '^PH ' $1/stats-phases.txt | cut -c4- | python3 -c 'import json,sys
out = []
for l in sys.stdin:
    try: d = json.loads(l)
    except Exception: continue
    out.append("%s: %s st (%s pf/mx), %s ms/st, dec %s ms, pf/mx %s ms, vram %s units, hit %s %%, routed %s / missed %s a st (rank 0), promo %s a st, stream %s MB/st, staged %s layers %s GB" % (
        d["phase"], d["steps"], d["prefill_mixed_steps"], d["ms_step"], d["ms_decode_step"], d["ms_prefill_mixed_step"], d["vram_units_end"],
        d["hit_pct_rank0"], d["routed_step_rank0"], d["missed_step_rank0"], d["promotions_step"], d["stream_MB_step"], d["stage_layers"], d["staged_GB"]))
print("; ".join(out))' 2>&1 | cut -c1-2400)"
}
vramunits() {   # vramunits FILE: the routed experts' VRAM / pinned units of a /stats snapshot
  python3 -c 'import json,sys
try: d = json.load(open(sys.argv[1]))
except Exception as e: print("n/a (%s)" % e); sys.exit(0)
t = {x.get("name"): x.get("units") for x in d.get("experts", {}).get("tiers", [])}
print("vram %s, pinned %s" % (t.get("vram"), t.get("pinned")))' "$1" 2>&1
}
# ---------------------------------------------------------------- one arm
arm() {   # arm Tn: RADIANCE_PLACE_TURNOVER=n (lent slots a slab class keeps free), everything else today's
  local a=$1 n=${1#T} tag=vs-$1 P=$O/vs-$1 rc st r
  case $n in ''|*[!0-9]*) log "RESULT xvs: VIRHE arm $a is not T<count>"; return 1 ;; esac
  t3env; modelvars225; export RADIANCE_PLACE_TURNOVER=$n
  log "RESULT xvs $tag: start IMAGE=$IMAGE MODEL=$MODEL $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
  if [ "$DRY" != 1 ]; then
    [ -f $MODELDIR/$MODEL ] && [ -s $MODELDIR/$MODEL.sha256ok ] || { log "RESULT xvs $tag: VIRHE $MODELDIR/$MODEL missing or not verified"; return 1; }
  fi
  mkd $P; run "rm -f $P/marks.txt $P/stats-order.txt"
  gate $tag $LIM; rc=$?
  [ $rc = 1 ] && { log "RESULT xvs $tag: VIRHE not healthy ($(desc)); start log $O/start-$tag.log: $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  [ $rc = 2 ] && { log "RESULT xvs $tag: VIRHE slow state after one restart -> arm skipped"; down; return 1; }
  modes $tag $n || { log "RESULT xvs $tag: VIRHE the modes are not the expected ones -> arm stopped"; down; return 1; }
  st=$(stprobe $tag 1)
  if [ "${st%% *}" = slow ]; then
    log "RESULT xvs $tag state probe 1: $st -> one restart"
    gate $tag $LIM || { log "RESULT xvs $tag: VIRHE no healthy / fast start after the slow-state restart"; down; return 1; }
    modes $tag $n || { down; return 1; }
    st=$(stprobe $tag 2)
  fi
  log "RESULT xvs $tag state: $st (slow-state probe mb c8 ${XVS_STATE_DUR:-45} s, gg_gemv 0)$(case "${st%% *}" in fast) ;; between) [ "$a" = T150 ] && echo " -- T150 holds ~268 fewer units, ~+5-7 % expected here" || echo " -- NOT FAST: flag to the coordinator" ;; *) echo " -- NOT FAST: flag to the coordinator" ;; esac)"
  stackon $tag || { log "RESULT xvs $tag: VIRHE gg_gemv 2 + aql_acq 2 not applied (GET /admin/fn4) -> arm stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARMREF
  follow $P; ph $P start
  [ "$DRY" = 1 ] || log "RESULT xvs $tag experts at the first snapshot: $(vramunits $P/stats-start.json) | bz d0 (NST 0, main image) at start: $(vramunits $RD/dsd-d0/stats-start.json)"
  stopfile || { probe $tag; ph $P probe; }
  for p in c4 c8; do stopfile && break; mbp $tag $p; ph $P $p; done
  stopfile || { pfd $tag $DEPTH; ph $P pf$DEPTH; }
  stopfile || { copyt $tag-i1 0; ph $P i1; }
  stopfile || { gsetc1 $tag; ph $P gs; }
  if [ "$DRY" = 1 ]; then echo "+ identity: 28k copy vs bz d0 ($RD/copy-dsd-d0-i1-run1.txt), greedy set c1 vs bz d0 ($RD/dsd-d0-gs-c1.jsonl)"
  else
    log "RESULT xvs $tag identity ran under: $(fn4get | cut -c1-200)"
    idline "$tag 28k copy vs bz d0 (NST 0, main image, stock)" "$(cmpfile $O/copy-$tag-i1-run1.txt $RD/copy-dsd-d0-i1-run1.txt)"
    idline "$tag greedy set c1 vs bz d0" "$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$tag-gs-c1.jsonl)"
  fi
  unfollow
  run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
  statsum $P $tag
  down
  log "RESULT xvs $tag: done"
}
# ---------------------------------------------------------------- summary
summary() {   # summary REF NEW: NEW vs REF (% and the slope per 100 resident units), each vs the gz blocks, identity NEW vs REF
  local RT=$1 NT=$2
  [ "$DRY" = 1 ] && { echo "+ RESULT xvs $NT vs $RT: c1 essay / code tok/s + ms, c4 / c8 aggregate + p50 + ms, ${DEPTH} TTFT + tok/s, VRAM units / hit / missed a step per phase, slope ms per 100 units (c1 / c4 / c8); each vs $REF_GZ; identity $NT vs $RT: warm-up, 28k copy, greedy set c1"; return 0; }
  log "RESULT xvs $NT vs $RT, same image $IMAGE: $(RT=$RT NT=$NT DEPTH=$DEPTH python3 - $O <<'PY' 2>&1 | cut -c1-3000
import json, os, sys
O = sys.argv[1]; A, B, dep = os.environ["RT"], os.environ["NT"], os.environ["DEPTH"]
def mb(tag, p):
    try:
        for l in open(f"{O}/{tag}-{p}.txt"):
            if l.startswith("MB "): return json.loads(l[3:])
    except OSError: pass
    return {}
def pr(tag, which):
    try:
        for l in open(f"{O}/{tag}-probe.txt"):
            if l.startswith("J "):
                d = json.loads(l[2:])
                if str(d.get("prompt", "")).startswith(which): return d
    except OSError: pass
    return {}
def pf(tag):
    try:
        for l in open(f"{O}/{tag}-pf{dep}.txt"):
            if l.startswith("J "): return json.loads(l[2:])
    except OSError: pass
    return {}
def st(tag):
    out = {}
    try:
        for l in open(f"{O}/{tag}/stats-phases.txt"):
            if l.startswith("PH "):
                d = json.loads(l[3:]); out[d["phase"]] = d
    except (OSError, ValueError): pass
    return out
sa, sb = st(A), st(B)
rows = [("c1 essay tok/s", pr(A, "essay").get("decode_tps"), pr(B, "essay").get("decode_tps")),
        ("c1 essay ms", pr(A, "essay").get("ms_per_decode_step"), pr(B, "essay").get("ms_per_decode_step")),
        ("c1 code tok/s", pr(A, "code").get("decode_tps"), pr(B, "code").get("decode_tps")),
        ("c1 code ms", pr(A, "code").get("ms_per_decode_step"), pr(B, "code").get("ms_per_decode_step"))]
for p in ("c4", "c8"):
    rows += [(f"{p} tok/s", mb(A, p).get("agg_dec_tps"), mb(B, p).get("agg_dec_tps")),
             (f"{p} p50", mb(A, p).get("per_stream_dec_p50"), mb(B, p).get("per_stream_dec_p50")),
             (f"{p} ms", mb(A, p).get("ms_decode_step"), mb(B, p).get("ms_decode_step"))]
rows += [(f"{dep} TTFT s", pf(A).get("ttft"), pf(B).get("ttft")), (f"{dep} tok/s", pf(A).get("prefill_tps"), pf(B).get("prefill_tps"))]
out = []
for n, x, y in rows:
    if x is None or y is None: out.append(f"{n} {x} -> {y}"); continue
    x, y = float(x), float(y)
    out.append(f"{n} {x:g} -> {y:g} ({(y / x - 1) * 100:+.1f} %)" if x else f"{n} {x} -> {y}")
for p in ("probe", "c4", "c8", f"pf{dep}"):
    a, b = sa.get(p, {}), sb.get(p, {})
    out.append(f"{p} units {a.get('vram_units_end')} -> {b.get('vram_units_end')}, hit {a.get('hit_pct_rank0')} -> {b.get('hit_pct_rank0')} %, missed {a.get('missed_step_rank0')} -> {b.get('missed_step_rank0')} a step")
# slope: ms a step per 100 resident units (B - A ms over A - B units), the c1 ms = mean of essay and code
def ms1(tag):
    v = [pr(tag, w).get("ms_per_decode_step") for w in ("essay", "code")]
    v = [float(x) for x in v if x is not None]
    return sum(v) / len(v) if v else None
for nm, ph, ma, mb_ in (("c1", "probe", ms1(A), ms1(B)), ("c4", "c4", mb(A, "c4").get("ms_decode_step"), mb(B, "c4").get("ms_decode_step")),
                        ("c8", "c8", mb(A, "c8").get("ms_decode_step"), mb(B, "c8").get("ms_decode_step"))):
    ua, ub = sa.get(ph, {}).get("vram_units_end"), sb.get(ph, {}).get("vram_units_end")
    if None in (ma, mb_, ua, ub) or ua == ub: out.append(f"slope {nm}: n/a"); continue
    s = (float(mb_) - float(ma)) / (float(ua) - float(ub)) * 100
    out.append(f"slope {nm}: {s:+.3f} ms a step per 100 resident units ({(s / float(ma)) * 100:+.2f} % of {float(ma):.2f} ms; units {ua} vs {ub})")
print("; ".join(out))
PY
)"
  for t in $RT $NT; do
    log "RESULT xvs $t vs $REF_GZ (information: in-process blocks there, fresh start + aql_acq 2 here): $(python3 -c 'import json,sys
O, t = sys.argv[1], sys.argv[2]
def mb(p):
    try:
        for l in open("%s/%s-%s.txt" % (O, t, p)):
            if l.startswith("MB "): return json.loads(l[3:])
    except OSError: pass
    return {}
def pr(w):
    try:
        for l in open("%s/%s-probe.txt" % (O, t)):
            if l.startswith("J "):
                d = json.loads(l[2:])
                if str(d.get("prompt", "")).startswith(w): return d
    except OSError: pass
    return {}
ref = {"c1": 15.72, "c4": 25.73, "c8": 38.93}
c1 = [pr(w).get("ms_per_decode_step") for w in ("essay", "code")]; c1 = [float(x) for x in c1 if x is not None]
v = {"c1": sum(c1) / len(c1) if c1 else None, "c4": mb("c4").get("ms_decode_step"), "c8": mb("c8").get("ms_decode_step")}
print("; ".join("%s %s ms vs %s (%s)" % (k, "-" if v[k] is None else "%.2f" % float(v[k]), ref[k], "-" if v[k] is None else "%+.1f %%" % ((float(v[k]) / ref[k] - 1) * 100)) for k in ("c1", "c4", "c8")))' $O $t 2>&1 | cut -c1-400)"
  done
  local w c g
  w=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): (d.get("content"), d.get("finish")) for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return None
a, b = rd(sys.argv[1]), rd(sys.argv[2])
print("not compared" if a is None or b is None else ("IDENTICAL (%d answers)" % len(a) if a == b else "DIFFERENT"))' $O/guard/$RT-warm-q.txt $O/guard/$NT-warm-q.txt 2>&1)
  c=$(cmpfile $O/copy-$NT-i1-run1.txt $O/copy-$RT-i1-run1.txt)
  g=$(gcmp $O/$RT-gs-c1.jsonl $O/$NT-gs-c1.jsonl)
  case "$w|$c|$g" in IDENTICAL*\|IDENTICAL*\|IDENTICAL*) log "RESULT xvs identity $NT vs $RT: EXACT -- warm-up $w; 28k copy $c; greedy set c1 $g" ;;
    *DIFFERENT*) log "RESULT xvs identity $NT vs $RT: VIRHE NOT EXACT -- warm-up $w; 28k copy $c; greedy set c1 $g" ;;
    *) log "RESULT xvs identity $NT vs $RT: INCOMPLETE (no difference found, a part not compared) -- warm-up $w; 28k copy $c; greedy set c1 $g" ;; esac
}
# ---------------------------------------------------------------- main
log "RESULT xvs $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) arms [$ARMS] (Tn = RADIANCE_PLACE_TURNOVER=n; today 16 = 32 kept free, 2948 units at decode), fresh prefill $DEPTH, yaml $Y | refs: $REF_GZ; dsm-225h counters probe / c4 / c8: 2948 / 2943 / 2941 units, hit 98.25 / 97.82 / 97.77 %; identity vs bz d0"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xvs: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xvs: VIRHE image $IMAGE not on gpu02"; exit 1; }
  for v in $SWITCHES; do grep -q "^ *$v:" $Y || { log "RESULT xvs: VIRHE $Y does not pass $v"; exit 1; }; done
  [ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -c RADIANCE_PLACE_TURNOVER /opt/radiance/bin/radiance' 2>/dev/null)" -gt 0 ] 2>/dev/null \
    || { log "RESULT xvs: VIRHE $IMAGE's engine does not know RADIANCE_PLACE_TURNOVER"; exit 1; }
  [ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -q r4d_gg_mr_pin /opt/radiance/share/radiance/kernels/libr4d.so && echo yes' 2>/dev/null)" = yes ] \
    || { log "RESULT xvs: VIRHE $IMAGE lacks lever D2 (gg_gemv 2)"; exit 1; }
  for f in $GS $D/stepprobe.py $D/mb.py $D/ds41_tests.py; do [ -e $f ] || { log "RESULT xvs: VIRHE $f missing"; exit 1; }; done
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xvs: VIRHE $M225 missing or not verified"; exit 1; }
  for f in $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $RD/dsd-d0/stats-start.json; do [ -s $f ] || log "RESULT xvs: note -- reference $f missing (not compared)"; done
else
  echo "+ image $IMAGE on this host: $(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null || echo MISSING)"
  echo "+ yaml $Y passes RADIANCE_PLACE_TURNOVER: $(grep -c '^ *RADIANCE_PLACE_TURNOVER:' $Y 2>/dev/null), RADIANCE_DS41_GG_GEMV: $(grep -c '^ *RADIANCE_DS41_GG_GEMV:' $Y 2>/dev/null)"
  for f in $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $RD/dsd-d0/stats-start.json $GS $D/stepprobe.py $D/mb.py $D/ds41_tests.py; do echo "+ ref / tool $f: $([ -s $f ] && echo ok || echo MISSING)"; done
fi
mkd $O $O/guard $STATSDIR
evict
done_arms=""
for a in $ARMS; do stopfile && break; arm $a && done_arms="$done_arms $a"; done
has() { case " $done_arms " in *" $1 "*) return 0 ;; esac; return 1; }
has T0 && has T150 && summary vs-T0 vs-T150
has T16 && has T0 && summary vs-T16 vs-T0
unset $SWITCHES
down
log "RESULT xvs $ITEM: done -- arms [$done_arms ] $O"
