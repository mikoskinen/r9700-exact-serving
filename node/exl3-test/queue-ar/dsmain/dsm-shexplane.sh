#!/bin/bash
# dsm-shexplane.sh (10.11.2026, queue-ar/dsmain, ds41-d3 agent, lever F4) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-j-shexplane.sh. Lever F4: the MoE block's shared expert on the second lane (rad_lane 1) beside the
# router, the sort and the routed EXL3 experts at a decode / verify step (<= 64 rows, one pass), joined back in front of the
# gather; A/B'd IN ONE PROCESS on one start of the ds41-d3 image (XD3_IMAGE) by the switch ds41_shexp_lane (0 = one lane,
# today's issue; 1 = the lane). RADIANCE_DS41_SHEXP_LANE=1 sets the start value (the serve log's note says so).
#   Why it can pay: after the router the shared expert (gemm_nt_ggml gate_up, the clamped swiglu, down: bandwidth-bound,
#   ~43-50 us a layer at 1 user) and the routed experts (VALU-bound at small M) are independent. On one queue a recorded pass
#   already overlaps them stage by stage (haz_mark: shared gate_up beside routed gate_up, the activation beside routed down),
#   but the shared down then runs alone, and each barrier waits for the slower arm; on two queues the whole arm runs beside
#   the routed chain. Estimate -0.5..-1.2 ms a 1-user step (coordinator); the item measures it.
#   EXACTNESS: the two arms share nothing but their read of the normed input until the gather (the arm's buffers are
#   rad_buf_concurrent at declare, as for the one-queue overlap), the same ops run on the same operands, and the gather
#   combines them exactly as before. The bytes are the same at either value.
#   The moe_scatter fold into the selection (the rest of the F4 row) is NOT built: it exists as RADIANCE_DS41_ROUTER_FUSE=1
#   (router_scored_topk_scatter, selection + sort + gate fill in one launch), measured and rejected in 77 (c4 -1.0..-1.4 %).
# DeepSeek-V4.1 Flash EXL3 2.25, TP4, MML 524288, NST 0, the t3 env and serve line of dsm-serve.sh (105's t3 arm + lib-29
# best_vars + MNBT 4160 + --live), RADIANCE_DS41_ARGATHER=on (today's 4-row bound; RADIANCE_DS41_ARGATHER_MAX unset, so F4 is
# measured alone on today's line) + RADIANCE_DS41_SHEXP_LANE=1, every other switch unset.
# ORDER (decode before prefill, TESTING 9.11.): gate -> modes (the t3 notes + the shared-expert-lane note "40 of 40 layers
#   lane-safe") -> the switch note (GET at start: ds41_shexp_lane 1, ds41_argather 4, gg_gemv 0, aql_acq 0, epoch 0;
#   ds41_shexp_lane 2 refused, 0 then 1 applied at a step) -> warm-up answers vs stock -> slow-state probe
#   -> BLOCKS ds41_shexp_lane 0 / 1 (ggblocks2.py, XD3_ORDER default A B B A A B, gg_gemv / aql_acq 0 and ds41_argather 4
#     pinned): stepprobe essay/code 512 (c1), mb c4, mb c8 (XD3_DUR 75 s) -> B vs A % per load point, pooled, pairs, t95
#   -> c4 greedy set under B and under A (B vs A against A's own run-to-run floor, A twice)
#   -> IDENTITY (prefill-heavy, last): under B then A, the 28k copy and the greedy set at c1, each vs bz's dsd-d0 files and B
#     vs A, byte for byte. Prefill is untouched (a step wider than 64 rows keeps one lane): no 16k / 136k pair.
# PROFILED (coordinator 10.11., ds41-miss 1f4754e; NOT a speed): what host-tier expert misses cost on the critical path. One
#   more start of TODAY's path (ARGATHER on at 4, the shared arm on one lane) with RADIANCE_DEBUG_MISS_LOG=/stats/miss.log,
#   --profile-ops and RADIANCE_PROFILE_EVERY 16; windows: the 1-user stepprobe (essay, code 512), mb c4 30 s, mb c8 30 s, each
#   a slice of the miss log and the engine log; tools/missreg.py (dsmain copy) --split --per-layer per window and over the
#   whole run: per rank the moe_exl3 ms a step against the live experts missed (host / other) and resident a step -> the
#   fitted ms per missed expert (host, other) and per resident live expert, and the missed experts a step at c1 / c4 / c8.
#   The miss log's engine wiring has only run in host tests: an empty log or a missreg error is logged VIRHE and the item
#   carries on (XD3_MISS=0 skips the phase).
# Serve-log check at the end: the core's "uses scratch on lane" warnings (a lane-1 op taking scratch would cost the overlap,
#   not correctness) are counted.
# References (no re-runs): bz's dsd-d0 (main image, NST 0): probe 59.38 / 62.04 tok/s (16.84 / 16.12 ms), c4 122.7 (p50
#   31.6), c8 150.1 (p50 19.7); d5 profile: shexp gate_up 0.87 + down 0.97 ms a 1-user step (PROFILED, serialised).
# The image (XD3_IMAGE, build/d3/assemble.sh): ds41-d3 = ds41-gg fbd4c8a + D3 + F4, merged with ds41-gg d5bed91 (--spec-ngram
#   leaves a draft head out) and ds41-miss 1f4754e (RADIANCE_DEBUG_MISS_LOG), layered on radiance-local:ds41-gg-fbd4c8ac-5c13e555-l.
# NO cc_sim (user 10.11.). No power / host / fan settings; production compose files read only (--project-directory).
# DRYRUN=1 bash dsm-shexplane.sh : prints every command, runs and writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-d3-tp4.yaml
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=$R/xdsm/shexplane; export STATSDIR=$O/stats
RD=$R/xdsm/draft                                # bz's dsd-d0 (main image, NST 0) files
RM=$R/xdsm                                      # item a's dsm-225 (main image, NST 0) files
export IMAGE=${XD3_IMAGE:-radiance-local:ds41-d3-c03c57d9-5c13e555-l}
ITEM=${XD3_ITEM:-57-p4zyzzzzz-ds-j-shexplane}; MISS=${XD3_MISS:-1}; ORDER=${XD3_ORDER:-ABBAAB}; DUR=${XD3_DUR:-75}
GARA=4   # ds41_argather pinned at today's bound in every block
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
GGB=$HD/ggblocks2.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DS41_ARGATHER_MAX RADIANCE_DS41_SHEXP_LANE RADIANCE_DEBUG_MISS_LOG"
t3env() {   # 105's t3 arm + SHEXP_LANE=1, every other switch unset (gg_gemv 0, aql_acq 0, ds41_argather 4 at start)
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0 \
         RADIANCE_DS41_SHEXP_LANE=1
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
  log "RESULT guard $tag: $st, 1-user ${ms} ms/step (limit $lim; bz dsd-d0 main image 15.59)"
  case $st in SLOW*) return 2 ;; esac
}
FOLLOW_PID=""
follow() { [ "$DRY" = 1 ] && { echo "+ docker logs -f $CN > $1/engine.log &"; return; }
  docker logs -f $CN > $1/engine.log 2>&1 & FOLLOW_PID=$!; sleep 2; }
unfollow() { [ -n "$FOLLOW_PID" ] && { kill $FOLLOW_PID 2>/dev/null; wait $FOLLOW_PID 2>/dev/null; FOLLOW_PID=""; }; return 0; }
trap 'unfollow; down' EXIT
trap 'exit 143' TERM INT
cmpfile() {   # cmpfile A B -> IDENTICAL (n chars) | DIFFERENT at char k (na vs nb chars) | not compared
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xd3 $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xd3 $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  case $r in IDENTICAL*) log "RESULT xd3 $1 warm-up answers vs stock ($(basename $2)): $r" ;;
    *) log "RESULT xd3 $1 warm-up answers vs stock ($(basename $2)): VIRHE $r -- NOT EXACT" ;; esac
}
modes() {   # modes TAG: the serve log's notes must be the t3 line's + the shared-expert-lane note -> 0, else 1 (stopped)
  [ "$DRY" = 1 ] && { echo "+ modes $1: indexer note (mxfp4, latent bf16, compressor tail bound, router selection+sort separate, AR_PCN separate at 0 layers), ARGATHER note up to 4 (no ARGATHER_MAX), the shared-expert-lane note (40 of 40 layers lane-safe), staging line, no acquire-rule line"; return 0; }
  local MLF SL AL XL ok=1
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  AL=$(docker logs $CN 2>&1 | grep -m1 'the MoE gather and its all-reduce as one kernel' | sed 's/^radiance: //' | cut -c1-300)
  XL=$(docker logs $CN 2>&1 | grep -m1 'the shared expert on the second lane' | sed 's/^radiance: //' | cut -c1-300)
  log "RESULT xd3 $1 modes: $(echo "$MLF" | cut -c1-900) | staging: ${SL:-none} | argather: ${AL:-none} | shexp lane: ${XL:-none}"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate" "post-collapse-norm separate at 0 layers" "next collapse-norm separate at 0 layers"; do
    case "$MLF" in *"$w"*) ;; *) log "RESULT xd3 $1: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  case "$AL" in *"up to 4 tokens a step (RADIANCE_DS41_ARGATHER)"*) ;; *) log "RESULT xd3 $1: VIRHE the ARGATHER note is not today's 4-row one: '${AL}'"; ok=0 ;; esac
  case "$XL" in *" 0 of "*|"") log "RESULT xd3 $1: VIRHE no lane-safe layer in the shared-expert-lane note: '${XL}'"; ok=0 ;; esac
  [ -n "$SL" ] || { log "RESULT xd3 $1: VIRHE no prefill staging line (levels refused?)"; ok=0; }
  docker logs $CN 2>&1 | grep -q 'the acquire rule plays' && { log "RESULT xd3 $1: VIRHE an acquire-rule line (RADIANCE_AQL_ACQ should be unset)"; ok=0; }
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
fn4code() {   # fn4code JSON: POST, print the HTTP status
  [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null
}
swnote() {   # swnote TAG: the switches at start, ds41_shexp_lane 2 refused, 0 then 1 applied at a step
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (expect ds41_shexp_lane 1, ds41_argather 4, gg_gemv 0, aql_acq 0, epoch 0); POST ds41_shexp_lane 2 -> 400; POST ds41_shexp_lane 0 / 1, each + a 4-token request -> applied"; return 0; }
  local s0 s1 s2 c2
  s0=$(fn4get)
  c2=$(fn4code '{"ds41_shexp_lane": 2}')
  fn4set '{"ds41_shexp_lane": 0}'; nudge; s1=$(fn4get)
  fn4set '{"ds41_shexp_lane": 1}'; nudge; s2=$(fn4get)
  log "RESULT xd3 $1 switch note: at start $s0 | ds41_shexp_lane 2 -> HTTP $c2 | after 0: $s1 | after 1: $s2 | core log: $(docker logs $CN 2>&1 | grep -E 'fn4: switches' | tail -2 | sed 's/^[A-Z] *//' | tr '\n' '|' | cut -c1-400)"
  python3 -c 'import json,sys
s0, s1, s2 = (json.loads(x) for x in sys.argv[1:4])
w = lambda s, k: s["switches"].get(k)
ok = w(s0, "ds41_shexp_lane") == 1 and w(s0, "ds41_argather") == 4 and w(s0, "gg_gemv") == 0 and w(s0, "aql_acq") == 0 \
     and s0["epoch"] == 0 and w(s1, "ds41_shexp_lane") == 0 and w(s2, "ds41_shexp_lane") == 1 and not s1["pending"] \
     and not s2["pending"] and s2["epoch"] == 2
sys.exit(0 if ok else 1)' "$s0" "$s1" "$s2" 2>/dev/null && [ "$c2" = 400 ]
}
mbms() { grep '^MB ' $1 2>/dev/null | head -1 | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()[3:]).get("ms_decode_step", ""))' 2>/dev/null; }
stprobe() {   # stprobe TAG N: mb c8 45 s -> "fast|between|slow MS (ref REF)"
  [ "$DRY" = 1 ] && { echo "+ (cd $D && timeout 400 python3 mb.py --tag xd3-$1-st$2 --phases c8 --dur ${XD3_STATE_DUR:-45} --warm 10 > $O/$1-stprobe$2.txt)" >&2; echo "fast (dry)"; return 0; }
  run "(cd $D && timeout 400 python3 mb.py --tag xd3-$1-st$2 --phases c8 --dur ${XD3_STATE_DUR:-45} --warm 10 > $O/$1-stprobe$2.txt 2>&1)"
  local ms; ms=$(mbms $O/$1-stprobe$2.txt)
  [ -n "$ms" ] || { echo "unknown (no mb line)"; return 0; }
  awk -v m=$ms -v r=$C8REF 'BEGIN{s=(m <= 1.07*r) ? "fast" : (m >= 1.14*r) ? "slow" : "between"; printf "%s %s ms (105 tuned3l c8 %s ms, x%.3f)\n", s, m, r, m/r}'
}
copyt() {   # copyt TAG: the 28k copy after a ~31k-token prompt (105's prompt, offset 0 -- bz d0's i1)
  local t0; t0=$(date +%s)
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset 0 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xd3 $1 copy: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300)$(type thermwin > /dev/null 2>&1 && echo " | $(thermwin $t0 $(date +%s) 2>/dev/null | cut -c1-200)")"
}
gsetc1() { run "(cd $D && timeout 900 python3 $GS $URL $O/$1-gs-c1.jsonl --conc 1 --max-tokens 192 > $O/$1-gs-c1.txt 2>&1)"; }
gsetc4() { run "(cd $D && timeout 900 python3 $GS $URL $O/$1-gs-c4.jsonl --conc 4 --max-tokens 192 > $O/$1-gs-c4.txt 2>&1)"; }
gcmp() { [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }; [ -s "$1" ] && [ -s "$2" ] || { echo "not compared (missing $1 or $2)"; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
idline() {   # idline LABEL RESULT: IDENTICAL -> plain, else VIRHE (the lever is exact by construction)
  case "$2" in IDENTICAL*) log "RESULT xd3 identity $1: $2" ;; "not compared"*) log "RESULT xd3 identity $1: $2" ;;
    *) log "RESULT xd3 identity $1: VIRHE $2 -- F4 is exact by construction, so this is a defect (or a reference from another line)" ;; esac
}
floorline() {   # floorline LABEL RESULT: a c4 comparison, read against A's own floor
  log "RESULT xd3 c4 $1: $2"
}
blocks() {   # blocks TAG SWITCH A B FIXED: the A/B blocks over SWITCH (FIXED pinned in every block) and their summary
  local tag=$1 sw=$2 a=$3 b=$4 fx=$5
  run "timeout 7200 python3 -I $GGB --url $URL --mb-dir $D --label $tag --switch $sw --a $a --b $b --fixed '$fx' --order $ORDER --dur $DUR --warm 10 --out $O/$tag-blocks.jsonl > $O/$tag-blocks.txt 2>&1"
  [ "$DRY" = 1 ] && return 0
  while IFS= read -r l; do log "RESULT xd3 $tag ${l:0:400}"; done < <(grep '^BLOCK ' $O/$tag-blocks.txt)
  log "RESULT xd3 $tag summary (B = $sw $b vs A = $a, $fx pinned, in process, order $ORDER, mb $DUR s): $(grep '^GGB ' $O/$tag-blocks.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=2: "-" if v is None else format(v, ".%df" % n)
out = []
for k, v in d["metrics"].items():
    p = v.get("pooled")
    out.append("%s %s -> %s (%s %%; pairs %s; t95 +- %s %%)" % (k, f(v.get("A")), f(v.get("B")), "-" if p is None else "%+.2f" % (100 * (p - 1)),
               " ".join("%+.2f" % (100 * (x - 1)) for x in v.get("pairs", [])), "-" if v.get("ci95") is None else "%.2f" % (100 * v["ci95"])))
print("; ".join(out) + " | %d blocks, %d flagged, %s s" % (d["blocks"], d["blocks_bad"], d["wall_s"]))' 2>&1 | cut -c1-2400)"
}
# ---------------------------------------------------------------- main
log "RESULT xd3 $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) A ds41_shexp_lane 0 / B 1 (ds41_argather 4), order $ORDER mb $DUR s yaml $Y | $(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -m1 "^radiance" /opt/radiance/share/radiance/d3-PROVENANCE.txt' 2>/dev/null | cut -c1-200) | refs bz dsd-d0 (main image, NST 0): probe 59.38 / 62.04 tok/s, c4 122.7 (p50 31.6), c8 150.1 (p50 19.7)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xd3: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xd3: VIRHE image $IMAGE not on gpu02"; exit 1; }
  [ -e $Y ] && grep -q '^ *RADIANCE_DS41_ARGATHER_MAX:' $Y && grep -q '^ *RADIANCE_DS41_SHEXP_LANE:' $Y || { log "RESULT xd3: VIRHE $Y missing or does not pass RADIANCE_DS41_ARGATHER_MAX / RADIANCE_DS41_SHEXP_LANE"; exit 1; }
  for f in $GS $GGB $D/stepprobe.py $D/mb.py; do [ -e $f ] || { log "RESULT xd3: VIRHE $f missing"; exit 1; }; done
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xd3: VIRHE $M225 missing or not verified"; exit 1; }
  [ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -q RADIANCE_DS41_ARGATHER_MAX /opt/radiance/share/radiance/architectures/deepseek41.so && grep -q ds41_shexp_lane /opt/radiance/bin/radiance && grep -q r4d_ar_gather_twoshot_4rank /opt/radiance/share/radiance/kernels/libr4d.so && echo yes' 2>/dev/null)" = yes ] \
    || { log "RESULT xd3: VIRHE $IMAGE is not the D3 image (no RADIANCE_DS41_ARGATHER_MAX in deepseek41.so, ds41_shexp_lane in radiance or the two-shot gather in libr4d)"; exit 1; }
fi
mkd $O $O/guard $STATSDIR
evict
TAG=f4; P=$O/f4
t3env; modelvars225
log "RESULT xd3 $TAG: start IMAGE=$IMAGE MODEL=$MODEL NST=$NST $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
mkd $P
gate $TAG $LIM; rc=$?
[ $rc = 1 ] && { log "RESULT xd3 $TAG: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$TAG.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; exit 1; }
[ $rc = 2 ] && { log "RESULT xd3 $TAG: VIRHE slow state after one restart -> stopped"; exit 1; }
modes $TAG || { log "RESULT xd3 $TAG: VIRHE the modes are not as asked -> stopped"; exit 1; }
swnote $TAG || { log "RESULT xd3 $TAG: VIRHE the switches are not as expected (start ds41_shexp_lane 1 / argather 4 / epoch 0, 2 refused, 0 / 1 applied) -> stopped"; keeplog $TAG-switch; exit 1; }
identw $TAG $WARMREF
st=$(stprobe $TAG 1)
if [ "${st%% *}" = slow ]; then
  log "RESULT xd3 $TAG state probe 1: $st -> one restart"
  gate $TAG $LIM || { log "RESULT xd3 $TAG: VIRHE no healthy / fast start after the slow-state restart"; exit 1; }
  modes $TAG || exit 1
  swnote $TAG || { log "RESULT xd3 $TAG: VIRHE the switches are not as expected after the restart -> stopped"; exit 1; }
  st=$(stprobe $TAG 2)
fi
log "RESULT xd3 $TAG state: $st (slow-state probe mb c8 ${XD3_STATE_DUR:-45} s)$(case "${st%% *}" in fast) ;; *) echo " -- NOT FAST: flag to the coordinator" ;; esac)"
follow $P
stopfile || blocks $TAG-shx ds41_shexp_lane 0 1 '{"gg_gemv": 0, "aql_acq": 0, "ds41_argather": 4}'
# ---- c4: B, then A twice (A's own run-to-run floor)
if ! stopfile; then
  fn4set '{"ds41_shexp_lane": 1, "ds41_argather": 4, "gg_gemv": 0, "aql_acq": 0, "epoch_bump": 1}'; nudge
  gsetc4 $TAG-B1
  fn4set '{"ds41_shexp_lane": 0, "ds41_argather": 4, "gg_gemv": 0, "aql_acq": 0, "epoch_bump": 1}'; nudge
  gsetc4 $TAG-A1; gsetc4 $TAG-A2
  if [ "$DRY" = 1 ]; then echo "+ c4: B1 vs A1, A1 vs A2"
  else
    floorline "c4 greedy set, B (lane) vs A (one lane)" "$(gcmp $O/$TAG-A1-gs-c4.jsonl $O/$TAG-B1-gs-c4.jsonl)"
    floorline "c4 greedy set, A run 1 vs run 2 (today's floor)" "$(gcmp $O/$TAG-A1-gs-c4.jsonl $O/$TAG-A2-gs-c4.jsonl)"
  fi
fi
# ---- identity (the prefill-heavy part last): B on the cold copy prompt, then A
if ! stopfile; then
  for v in 'B:{"ds41_shexp_lane": 1, "ds41_argather": 4, "gg_gemv": 0, "aql_acq": 0, "epoch_bump": 1}' \
           'A:{"ds41_shexp_lane": 0, "ds41_argather": 4, "gg_gemv": 0, "aql_acq": 0, "epoch_bump": 1}'; do
    nm=${v%%:*}
    fn4set "${v#*:}"
    copyt $TAG-$nm; gsetc1 $TAG-$nm
    [ "$DRY" = 1 ] || log "RESULT xd3 $TAG identity $nm ran under: $(fn4get | cut -c1-240)"
  done
  if [ "$DRY" = 1 ]; then echo "+ identity: copy B / A vs bz dsd-d0, B vs A; greedy set c1 B / A vs dsd-d0, B vs A"
  else
    idline "28k copy B (lane, cold prompt) vs bz dsd-d0 (main image, NST 0)" "$(cmpfile $O/copy-$TAG-B-run1.txt $RD/copy-dsd-d0-i1-run1.txt)"
    idline "28k copy A (one lane) vs bz dsd-d0" "$(cmpfile $O/copy-$TAG-A-run1.txt $RD/copy-dsd-d0-i1-run1.txt)"
    idline "28k copy B vs A (in process)" "$(cmpfile $O/copy-$TAG-B-run1.txt $O/copy-$TAG-A-run1.txt)"
    idline "greedy set c1 B vs bz dsd-d0" "$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$TAG-B-gs-c1.jsonl)"
    idline "greedy set c1 A vs bz dsd-d0" "$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$TAG-A-gs-c1.jsonl)"
    idline "greedy set c1 B vs A (in process)" "$(gcmp $O/$TAG-A-gs-c1.jsonl $O/$TAG-B-gs-c1.jsonl)"
  fi
fi
# ---- PROFILED: host-tier misses on the critical path (one more start of today's path; see the header)
lines() { [ -s "$1" ] && wc -l < "$1" || echo 0; }
missfit() {   # missfit W MISS ENGINE: missreg over one window's slices -> its file and the RESULT lines
  local w=$1 out=$O/$TAG-miss-$1-missreg.txt
  run "timeout 600 python3 -I $HD/missreg.py $2 $3 --split --per-layer --every 16 > $out 2>&1"
  [ "$DRY" = 1 ] && return 0
  if ! grep -q 'interval(s): mean' $out; then
    log "RESULT xd3 miss PROFILED $w: VIRHE missreg gave no fit ($(grep -vE '^\s*$' $out | tail -2 | tr '\n' ' ' | cut -c1-300)) -- $out"
    return 0
  fi
  python3 - $out <<'PY' | while IFS= read -r l; do log "RESULT xd3 miss PROFILED $w $l"; done
import re, sys
rk, out = None, []
for ln in open(sys.argv[1]):
    m = re.match(r"\s*== rank (\d+)", ln)
    if m: rk = m.group(1); continue
    if ln.startswith("== all ranks pooled"): rk = "pooled"; continue
    if rk is None: continue
    m = re.search(r"(\d+) interval\(s\): mean ([\d.]+) ms a step, ([\d.]+) missed \(host ([\d.]+), other ([\d.]+)\) and ([\d.]+) resident", ln)
    if m:
        out.append(f"rank {rk}: {m.group(1)} intervals, moe_exl3 {m.group(2)} ms a step (PROFILED), missed {m.group(3)} a step "
                   f"(host {m.group(4)}, other {m.group(5)}), resident live {m.group(6)}")
        continue
    if "with intercept" in ln or ("no  " in ln and "intercept" in ln):
        out.append(f"rank {rk} fit: " + re.sub(r"\s+", " ", ln.strip()))
for o in out: print(o[:400])
PY
}
missphase() {
  local tag=$TAG-miss MS=$O/stats-miss w ml0 ml1 el0 el1
  t3env; unset RADIANCE_DS41_SHEXP_LANE; modelvars225
  export STATSDIR=$MS RADIANCE_DEBUG_MISS_LOG=/stats/miss.log PROFEVERY=16 XARGS="$XARGS --profile-ops"
  mkd $MS; run "rm -f $MS/miss.log"
  log "RESULT xd3 miss PROFILED: start (today's path: ARGATHER 4, one lane) IMAGE=$IMAGE $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
  restart $tag || { keeplog $tag-fail; log "RESULT xd3 miss PROFILED: VIRHE not healthy -- phase skipped ($O/engine-$tag-fail.log)"; return 0; }
  nudge
  for w in c1 c4 c8; do
    stopfile && break
    ml0=$(lines $MS/miss.log); el0=$( [ "$DRY" = 1 ] && echo 0 || docker logs $CN 2>&1 | wc -l)
    case $w in
      c1) run "(cd $D && timeout 900 python3 stepprobe.py essay,code 512 > $O/$tag-c1.txt 2>&1)" ;;
      *)  run "(cd $D && timeout 400 python3 mb.py --tag xd3-$tag-$w --phases $w --dur 30 --warm 5 > $O/$tag-$w.txt 2>&1)" ;;
    esac
    run "sleep 3"   # the log's writer thread writes once a second
    ml1=$(lines $MS/miss.log); el1=$( [ "$DRY" = 1 ] && echo 0 || docker logs $CN 2>&1 | wc -l)
    run "{ grep '^# ' $MS/miss.log | grep -v '^# profile' | head -20; sed -n '$((ml0 + 1)),${ml1}p' $MS/miss.log; } > $O/$tag-$w-miss.log 2>/dev/null"
    run "docker logs $CN 2>&1 | sed -n '$((el0 + 1)),${el1}p' > $O/$tag-$w-engine.log"
    [ "$DRY" = 1 ] || log "RESULT xd3 miss PROFILED $w window: miss log lines $ml0..$ml1, engine log lines $el0..$el1, $(grep -c '^# profile' $O/$tag-$w-miss.log) profile markers, $(grep -c 'per-op timing' $O/$tag-$w-engine.log) tables; $(case $w in c1) grep '^J ' $O/$tag-c1.txt | python3 -c 'import json,sys; print("stepprobe " + " / ".join("%.2f ms" % json.loads(l[2:])["ms_per_decode_step"] for l in sys.stdin))' 2>/dev/null ;; *) echo "mb ms a step $(mbms $O/$tag-$w.txt)" ;; esac) (PROFILED)"
  done
  run "docker stop -t 60 $CN > /dev/null 2>&1"   # graceful: the miss log's closing summary
  run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
  if [ "$DRY" != 1 ] && [ "$(lines $MS/miss.log)" -lt 3 ]; then
    log "RESULT xd3 miss PROFILED: VIRHE the miss log is empty or missing ($MS/miss.log: $(lines $MS/miss.log) lines; engine: $(grep -m2 -iE 'miss.log|miss log' $O/$tag-engine-full.log | tr '\n' ' ' | cut -c1-300)) -- the engine wiring did not write it"
  else
    [ "$DRY" = 1 ] || log "RESULT xd3 miss PROFILED: miss log $(lines $MS/miss.log) lines, header: $(grep -m1 '^# model' $MS/miss.log | cut -c1-200); closing: $(grep '^#' $MS/miss.log | tail -1 | cut -c1-200)"
    for w in c1 c4 c8; do missfit $w $O/$tag-$w-miss.log $O/$tag-$w-engine.log; done
    missfit all $MS/miss.log $O/$tag-engine-full.log
  fi
  down
  export PROFEVERY= ; unset RADIANCE_DEBUG_MISS_LOG
}
unfollow
run "docker logs $CN > $O/$TAG-engine-full.log 2>&1"
[ "$DRY" = 1 ] || log "RESULT xd3 $TAG engine log: $(grep -cE '^E ' $O/$TAG-engine-full.log) E lines, $(grep -c 'fn4: switches' $O/$TAG-engine-full.log) switch applications, scratch-join warnings $(grep -c 'uses scratch on lane' $O/$TAG-engine-full.log)"
down
if [ "$MISS" = 1 ] && ! stopfile; then missphase; fi
unset $SWITCHES
down
log "RESULT xd3 $ITEM: done -- $O"
