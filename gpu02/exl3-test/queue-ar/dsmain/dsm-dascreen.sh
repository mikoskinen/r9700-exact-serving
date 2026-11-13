#!/bin/bash
# dsm-dascreen.sh (11.11.2026, queue-ar/dsmain, ds-main-port DA agent) -- NOT a queue item; the body of 57-p4zyzzzzz-ds-q-dascreen.sh.
# LEVER DA (drafter adaptation: self-distil the DSpark drafter to the EXL3 2.25 target) -- THE SCREEN, NOT THE LEVER. Nothing is
# trained, no weight changes: this measures how much of today's prose miss a drafter matched to this target could win back AT
# MOST, before anyone spends card-hours or cloud money on training. Exact by construction (the drafter only proposes; logprobs
# and -v read, they change no token) and checked: logprobs on vs off text, 28k copy + 20-prompt greedy set vs bz d0.
# ONE NST 5 START, image XDA_IMAGE (default the W1 image radiance-local:ds41-walk-eecd93af-5c13e555-l; an int3 image can be passed),
#   the hx-stack env + RADIANCE_DS41_DSPARK_WALK=1 (lever W1 on: the full-vocabulary Markov walk), --draft-window adaptive, -v:
#   0. gate (informational: acceptance does not depend on the host state; one try, no retry) -> the W1 declared line, the env in
#      the container -> fn4: the FIXED WINDOW 5 with every proposal from DSpark ({"draft_adaptive": 0, "draft_depth0": 0,
#      "draft_hybrid": 0, "dspark_walk": 1} + gg_gemv 2, aql_acq 2, K1 three 1), applied at a step -> warm-up answers vs stock.
#   1. ANATOMY (daanat.py collect; ~4-5 min): 20 prose + 6 code prompts (the 1-user probe's essay and code first, then synthetic
#      ones; never a user transcript), c1, /v1/chat/completions, greedy, reasoning none, 384 tokens, logprobs + top 20; each
#      request's "drafter 'dspark': sequence S proposes d1 .. d5" lines (-v) from the followed log; then the first prose prompt
#      again without logprobs (text must be IDENTICAL).
#   2. IDENTITY under the screen's switches: prefix-cache reset, the 28k copy and the 20-prompt greedy set at c1 vs bz d0.
#   3. ANALYSIS (daanat.py analyse, CPU): the replay of perpos.py per request (validated against the engine's own accepted count),
#      conditional acceptance p1..p5, every block's first rejection classed by the TARGET's own distribution there:
#      A confident (p_top1 >= 0.8), B near tie (the drafter's token within ln 2 of the target's top 1: the quantisation's label
#      flips live here), C other (the target itself unsure); acceptance by target confidence; the drafter's token's rank in the
#      target's top 20; and the bounds on tokens a step (fixed window 5, chain model): now / +A / +A+B (= DA's addressable upper
#      bound in this screen) / distribution-matched (rejected positions credited p_top1) / perfect 6.0, each also as estimated
#      tok/s at today's round time (walk2 B: essay 64.54, code 104.0 tok/s), against the 1-user target (90 tok/s = x 1.395).
# Tags "RESULT xda2 ", last lines "RESULT xda2 summary: ..." and "RESULT xda2 <item>: done"; out runs/xdsm/dascreen.
# ~14-16 min (evict 1, start + warm-up 5-6, anatomy 4-5, identity 2, analysis < 1). NO cc_sim, no 2.9, no oracle, no training,
#   no BetterBench. No power / host / fan settings; production compose files read only (common.yaml via extends, as every item).
# XDA_IMAGE, XDA_ITEM, XDA_OUT (dascreen), XDA_PROSE (20), XDA_CODE (6), XDA_TOK (384).
# DRYRUN=1 bash dsm-dascreen.sh : prints every command; checks the image (docker image inspect + a no-device, no-network grep of
#   its files), every helper and reference file, runs daanat.py's selftest (CPU, writes nothing); no server, no GPU.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
Y=$HD/ds41-walk-tp4.yaml
O=$R/xdsm/${XDA_OUT:-dascreen}; export STATSDIR=$O/stats
RD=$R/xdsm/draft                                # bz's d0 files (identity reference: copy + greedy set)
export IMAGE=${XDA_IMAGE:-radiance-local:ds41-walk-eecd93af-5c13e555-l}
ITEM=${XDA_ITEM:-57-p4zyzzzzz-ds-q-dascreen}
NPROSE=${XDA_PROSE:-20}; NCODE=${XDA_CODE:-6}; NTOK=${XDA_TOK:-384}
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
DAA=$HD/daanat.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
SCREEN='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 0, "draft_depth0": 0, "draft_hybrid": 0, "dspark_walk": 1}'
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DS41_ARGATHER_MAX RADIANCE_DS41_SHEXP_LANE RADIANCE_DEBUG_MISS_LOG
  RADIANCE_DS41_KVFOLD RADIANCE_DS41_OFOLD RADIANCE_DS41_ONEFILL RADIANCE_DS41_PIPE
  RADIANCE_DRAFT_DEPTH0 RADIANCE_DRAFT_PROBE RADIANCE_DRAFT_HYBRID RADIANCE_DRAFT_HYBRID_MATCH RADIANCE_DS41_DSPARK_WALK"
walkenv() {   # = dsm-walk3.sh's: hx-stack's env (105's t3 arm + gg_gemv 2 + rule-w + the K1 folds + turnover 0) + the W1 walk at start
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  export RADIANCE_PLACE_TURNOVER=0 RADIANCE_DS41_DSPARK_WALK=$1
}
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
FOLLOW_PID=""
follow() { [ "$DRY" = 1 ] && { echo "+ docker logs -f $CN > $1/engine.log &"; return; }
  docker logs -f $CN > $1/engine.log 2>&1 & FOLLOW_PID=$!; sleep 2; }
unfollow() { [ -n "$FOLLOW_PID" ] && { kill $FOLLOW_PID 2>/dev/null; wait $FOLLOW_PID 2>/dev/null; FOLLOW_PID=""; }; return 0; }
trap 'unfollow; down' EXIT
trap 'exit 143' TERM INT
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
cmpfile() {   # cmpfile A B -> IDENTICAL (n chars) | DIFFERENT at char k (na vs nb chars) | not compared
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
identw() {   # identw TAG REFFILE: warm-up answers (check's ds41_tests correct --only-q) vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xda2 $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xda2 $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  echo "warm-up-$1 $r" >> $O/identity.txt
  log "RESULT xda2 $1 warm-up answers vs stock ($(basename $2)): $r"
}
modes() {   # modes TAG: the env reached the container, turnover 0 on 4 ranks, the W1 declared line with dspark_walk=1, -v on
  [ "$DRY" = 1 ] && { echo "+ modes $1: RADIANCE_DS41_DSPARK_WALK=1 and RADIANCE_PLACE_TURNOVER=0 in the container, 0 kept-free slots on 4 ranks, the W1 declared line (dspark_walk=1 over 4 ranks), debug lines present (-v)"; return 0; }
  local CE WL nz nd ok=1 kv
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in RADIANCE_PLACE_TURNOVER=0 RADIANCE_DS41_DSPARK_WALK=1 RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w; do
    echo "$CE" | grep -qxF "$kv" || { log "RESULT xda2 $1: VIRHE $kv not in the container"; ok=0; }; done
  nz=$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots \([0-9]+ shadow, 0 more kept free')
  WL=$(docker logs $CN 2>&1 | grep -m1 'DSpark lever W1.*fn4 dspark_walk=' | sed 's/^radiance: //' | cut -c1-400)
  nd=$(docker logs $CN 2>&1 | grep -cE '^D ')
  log "RESULT xda2 $1 modes: W1: ${WL:-none} | turnover 0 on $nz of 4 ranks | debug lines (-v) $nd | indexer: $(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //' | cut -c1-300)"
  case "$WL" in *"dspark_walk_step a step + all_gather over 4 rank(s)"*"dspark_walk=1 "*) ;; *) log "RESULT xda2 $1: VIRHE no W1 declared line with dspark_walk=1 over 4 ranks"; ok=0 ;; esac
  [ "$nd" -gt 0 ] || { log "RESULT xda2 $1: VIRHE no debug lines: -v did not reach the server (the anatomy needs the proposals)"; ok=0; }
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
draftlines() {
  [ "$DRY" = 1 ] && { echo "+ RESULT xda2 $1 drafter lines"; return 0; }
  log "RESULT xda2 $1 drafter / speculation lines: $(docker logs $CN 2>&1 | grep -iE 'dspark|speculat|draft-window|draft window|draft depth|draft hybrid' | grep -v ' proposes ' | sed 's/^[A-Z][^ ]* //' | sort -u | head -12 | tr '\n' '|' | cut -c1-1800)"
}
screenset() {   # the fixed window 5, every proposal DSpark's, the walk on; applied at a step (epoch moved, nothing pending)
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (expect dspark_walk 1, draft_capable 1); POST $SCREEN + epoch_bump; a 4-token request; GET -> applied, nothing pending"; return 0; }
  local s0 s1 body
  s0=$(fn4get)
  body=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d["epoch_bump"]=1; print(json.dumps(d))' "$SCREEN")
  fn4set "$body"; nudge; s1=$(fn4get)
  log "RESULT xda2 $1 switch note: at start $s0 | after the screen's set: $s1"
  python3 -c 'import json,sys
s0, s1, w = (json.loads(x) for x in sys.argv[1:4])
ok = s0.get("draft_capable") == 1 and s0["switches"].get("dspark_walk") == 1
ok = ok and all(s1["switches"].get(k) == v for k, v in w.items()) and not s1["pending"] and s1["epoch"] > s0["epoch"]
sys.exit(0 if ok else 1)' "$s0" "$s1" "$SCREEN" 2>/dev/null
}
pcreset() {
  [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o $O/$1-reset.json -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  log "RESULT xda2 $1 prefix cache reset: HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"
}
copyt() {   # copyt TAG OFFSET: the copy test after a ~28k-token cold prompt (105's prompt: same offset -> same text)
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset $2 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xda2 $1 copy after a cold prompt: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300)"
}
gcmp() { [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }; [ -s "$1" ] && [ -s "$2" ] || { echo "not compared (missing $1 or $2)"; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
ident() {   # ident TAG: under the screen's switches: a prefix-cache reset, the 28k copy and the c1 greedy set vs bz d0
  local tag=$1 r
  pcreset $tag
  copyt $tag 0
  run "(cd $D && timeout 900 python3 $GS $URL $O/$tag-gs-c1.jsonl --conc 1 --max-tokens 192 > $O/$tag-gs-c1.txt 2>&1)"
  [ "$DRY" = 1 ] && { echo "+ identity (screen): 28k copy vs bz d0, greedy set c1 vs bz d0"; return 0; }
  log "RESULT xda2 $tag identity ran under: $(fn4get | cut -c1-500)"
  r=$(cmpfile $O/copy-$tag-run1.txt $RD/copy-dsd-d0-i1-run1.txt)
  echo "copy $r" >> $O/identity.txt
  log "RESULT xda2 $tag identity (fixed window 5, walk 1, -v): 28k copy vs bz d0 (NST 0, stock): $r$(case $r in IDENTICAL*) ;; *) echo " -- reported, not hidden" ;; esac)"
  r=$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$tag-gs-c1.jsonl)
  echo "gs $r" >> $O/identity.txt
  log "RESULT xda2 $tag identity: greedy set c1 vs bz d0: $r$(case $r in IDENTICAL*) ;; *) echo " -- reported, not hidden" ;; esac)"
}
anatomy() {   # anatomy TAG DIR: daanat.py collect against the followed log, then the logprobs-off text check
  run "timeout 2400 python3 -I -B $DAA collect --url $URL --log $2/engine.log --out $O/da --prose $NPROSE --code $NCODE --max-tokens $NTOK --top 20 > $O/da-collect.txt 2>&1"
  [ "$DRY" = 1 ] && return 0
  log "RESULT xda2 $1 anatomy requests: $(python3 -c 'import json,sys
r=[json.loads(l[10:]) for l in open(sys.argv[1]) if l.startswith("DACOLLECT ")]
for k in ("prose","code"):
    x=[d for d in r if d["kind"]==k]
    if not x: continue
    print("%s %d requests, %d tokens, %d proposal lines, %.1f s, one sequence a window in %d, finish %s;" % (k, len(x), sum(d["tokens"] for d in x), sum(d["blocks"] for d in x), sum(d["s"] for d in x), sum(1 for d in x if d["seqs"]==1), ",".join(sorted(set(str(d["finish"]) for d in x)))), end=" ")' $O/da-collect.txt 2>&1 | cut -c1-600)$(grep -m1 -E 'Traceback|Error' $O/da-collect.txt | cut -c1-200)"
  local l; l=$(grep -m1 '^DALPID ' $O/da-collect.txt | cut -c8-)
  echo "logprobs-off ${l:-not run}" >> $O/identity.txt
  log "RESULT xda2 $1 logprobs on vs off, the probe essay's text: ${l:-not run}$(case "${l:-}" in IDENTICAL*) ;; *) echo " -- reported, not hidden" ;; esac)"
}
analysis() {
  [ "$DRY" = 1 ] && { echo "+ python3 -I -B $DAA analyse $O/da-rows.jsonl > $O/da-analyse.txt; RESULT xda2 lines (anatomy, classes, confidence, bounds), RESULT xda2 summary"; return 0; }
  python3 -I -B $DAA analyse $O/da-rows.jsonl > $O/da-analyse.txt 2>&1
  while IFS= read -r l; do log "RESULT xda2 ${l:7:2400}"; done < <(grep '^DALINE ' $O/da-analyse.txt)
  grep -qE '^DALINE ' $O/da-analyse.txt || log "RESULT xda2: VIRHE the analysis printed nothing: $(head -c 300 $O/da-analyse.txt | tr '\n' ' ')"
  log "RESULT xda2 summary: $(python3 -c 'import json,sys
d=None
for l in open(sys.argv[1]):
    if l.startswith("DASUM "): d=json.loads(l[6:])
ident=[l.strip() for l in open(sys.argv[2]) if l.strip()] if sys.argv[2] else []
out=[]
for k in ("prose","code"):
    x=(d or {}).get(k) or {}
    b=x.get("bounds")
    if not b: out.append("%s: no blocks" % k); continue
    f=lambda v: "%.3f" % v
    pc=lambda v: "%+.1f %%" % ((v/b["now"]-1)*100)
    rej=sum(x["A"])+sum(x["B"])+sum(x["C"])
    out.append("%s tokens a step (fixed 5) now %s (replay %s), +A %s (%s), +A+B %s (%s, DA addressable bound), dist %s (%s); p1 %s; first rejections A %d / B %d / C %d of %d; acc at p_top1 >= 0.9 %d/%d; needed for 90 tok/s ~%s" % (
        k, f(b["now"]), f(b["replay"]), f(b["+A"]), pc(b["+A"]), f(b["+A+B"]), pc(b["+A+B"]), f(b["dist"]), pc(b["dist"]), f(x["cond_now"][0]),
        sum(x["A"]), sum(x["B"]), sum(x["C"]), rej, x["acc_at_ptop1_ge_0.9"][0], x["acc_at_ptop1_ge_0.9"][1], f(x["needed_tokens_a_step_for_90"] or 0)))
print(" || ".join(out) + " || identity / checks: " + (", ".join(ident) or "none recorded"))' $O/da-analyse.txt $O/identity.txt 2>&1 | cut -c1-3000)"
}
# ---------------------------------------------------------------- main
log "RESULT xda2 $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) | lever DA screen: the rejection anatomy of DSpark on the 2.25 target | one NST 5 start, hx-stack env + RADIANCE_DS41_DSPARK_WALK=1, -v, fn4 fixed window 5 $SCREEN | $NPROSE prose + $NCODE code requests x $NTOK tokens, c1, logprobs top 20 | identity vs bz d0 | refs walk2 B (11.11.): essay 64.54 tok/s 1.90-1.93 tok/step p1 ~0.56, fixed-5 essay 2.04 tok/step"
docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xda2: VIRHE image $IMAGE not on this host"; exit 1; }
[ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -qzF "draft_hybrid_match" /opt/radiance/bin/radiance && grep -qzF "draft depth 0: a step drafts nothing" /opt/radiance/bin/radiance && grep -qzF reset_prefix_cache /opt/radiance/bin/radiance && grep -qzF "proposes%s" /opt/radiance/bin/radiance && grep -qzF "top_logprobs" /opt/radiance/bin/radiance && grep -qzF "dspark_walk" /opt/radiance/bin/radiance && grep -qzF RADIANCE_DS41_DSPARK_WALK /opt/radiance/share/radiance/architectures/deepseek41.so && grep -qzF RADIANCE_DS41_KVFOLD /opt/radiance/share/radiance/architectures/deepseek41.so && grep -qzF dspark_walk_step /opt/radiance/share/radiance/kernels/libr4d.so && grep -qzF sample_logprobs /opt/radiance/share/radiance/kernels/libr4d.so && echo yes' 2>/dev/null)" = yes ] \
  || { log "RESULT xda2: VIRHE $IMAGE is not a W1 image (the registry with dspark_walk, depth 0 / hybrid, reset_prefix_cache, the -v proposal line, top_logprobs, RADIANCE_DS41_DSPARK_WALK, K1, dspark_walk_step, sample_logprobs)"; exit 1; }
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)) is a W1 image with the -v proposal line and top_logprobs -- OK"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xda2: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  [ -e $Y ] && grep -q '^ *RADIANCE_DS41_DSPARK_WALK:' $Y && grep -q '^ *RADIANCE_DS41_GG_GEMV:' $Y && grep -q '^ *RADIANCE_PLACE_TURNOVER:' $Y \
    || { log "RESULT xda2: VIRHE $Y missing or does not pass RADIANCE_DS41_DSPARK_WALK / GG_GEMV / PLACE_TURNOVER"; exit 1; }
  for f in $GS $DAA $D/ds41_tests.py $D/fastcheck.py; do [ -e $f ] || { log "RESULT xda2: VIRHE $f missing"; exit 1; }; done
  python3 -I -B $DAA selftest > /dev/null 2>&1 || { log "RESULT xda2: VIRHE daanat.py selftest fails"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xda2: VIRHE $M225 missing or not verified"; exit 1; }
  for f in $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $WARM_REF_225; do [ -s $f ] || log "RESULT xda2: note -- reference $f missing (not compared)"; done
else
  for f in $Y $GS $DAA $D/ds41_tests.py $D/fastcheck.py $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $WARM_REF_225 $C/common.yaml; do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  echo "+ yaml passes RADIANCE_DS41_DSPARK_WALK: $(grep -c '^ *RADIANCE_DS41_DSPARK_WALK:' $Y), RADIANCE_PLACE_TURNOVER: $(grep -c '^ *RADIANCE_PLACE_TURNOVER:' $Y), RADIANCE_DS41_GG_GEMV: $(grep -c '^ *RADIANCE_DS41_GG_GEMV:' $Y)"
  echo "+ $(python3 -I -B $DAA selftest 2>&1 | cut -c1-160)"
  python3 -I -B $DAA --help > /dev/null 2>&1 && echo "+ daanat.py parses" || echo "+ MISSING: daanat.py does not run"
  python3 -I -B -c 'import ast,sys; ast.parse(open(sys.argv[1]).read())' $GS && echo "+ greedyset.py parses" || echo "+ MISSING: greedyset.py does not parse"
  for fn in log run mkd evict down stopfile restart check best_vars desc; do declare -F $fn > /dev/null && echo "+ helper $fn: defined" || echo "+ MISSING helper: $fn"; done
  echo "+ D=$D R=$R C=$C MODELDIR=$MODELDIR URL=$URL CN=$CN"
fi
mkd $O $O/guard $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt
evict
tag=da-n5; P=$O/$tag
walkenv 1; export MODEL=$M225; HP29=19456; best_vars
export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive -v"
log "RESULT xda2 $tag: start IMAGE=$IMAGE $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
mkd $P
if ! stopfile; then
  if ! restart $tag; then
    keeplog $tag-fail; log "RESULT xda2 $tag: VIRHE not healthy: $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500) -> nothing measured"; down
  else
    ms=$(check $tag 1)
    [ "$DRY" = 1 ] || { echo "informational $ms" > $O/guard-$tag.txt; log "RESULT xda2 guard $tag: 1-user ${ms} ms/step (informational, one try: acceptance does not depend on the host state; NST 5 -v; walk2 start 34.91 without -v)"; }
    if ! modes $tag; then
      log "RESULT xda2 $tag: VIRHE the start is not W1 + -v -> stopped"; down
    else
      draftlines $tag
      if ! screenset $tag; then
        log "RESULT xda2 $tag: VIRHE the screen's switches did not apply -> stopped"; keeplog $tag-switch; down
      else
        identw $tag $WARM_REF_225
        follow $P
        stopfile || anatomy $tag $P
        stopfile || ident $tag-i
        unfollow
        run "docker logs $CN 2>&1 | gzip -1 > $O/$tag-engine-full.log.gz"
        down
        log "RESULT xda2 $tag: done"
      fi
    fi
  fi
fi
analysis
unset $SWITCHES
down
log "RESULT xda2 $ITEM: done -- $O"
