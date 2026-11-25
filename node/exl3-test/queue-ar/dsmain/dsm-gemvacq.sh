#!/bin/bash
# dsm-gemvacq.sh (10.11.2026, queue-ar/dsmain, ds41-gg agent, D1) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-gz-gemvacq.sh. Three exact levers A/B'd IN ONE PROCESS on one start of the ds41-gg D1 + D2 image
# (XGG_IMAGE):
#   (1) gg_gemv 0 / 1 -- gemm_nt_ggml's decode band at the measured per-(type, N, K, M) form (rows a wave, grid cap),
#       the image's table (see its gg-PROVENANCE: the g0 bench's full M 1..48 table, or the seeded M 1/4/8 one);
#   (2) gg_gemv 0 / 2 -- lever D2, the decode band's multi-row form (every weight block dequantised once for all the
#       call's rows, gemm_nt_ggml and logits_gemm_ggml; its measured forms from gb-mrbench, 145 entries, see the
#       image's gg-PROVENANCE "mr table" line; the rule H 4, R = N >= 4096 ? 2 : 1 elsewhere);
#   (3) aql_acq 0 / 2 -- the address-based acquire rule, rule-w (the safe form: every line written in the window is
#       risky), the coordinator's in-process A/B of rule-w instead of a separate start.
# DeepSeek-V4.1 Flash EXL3 2.25, TP4, MML 524288, NST 0 (the drafting verify widths would need a second start at NST 5,
# not cheap here; NST is start-time), the t3 env and serve line of dsm-serve.sh (105's t3 arm + lib-29 best_vars + MNBT
# 4160 + --live); RADIANCE_DS41_GG_GEMV and RADIANCE_AQL_ACQ unset at start (gg_gemv 0, rule off).
# ORDER (decode before prefill, TESTING 9.11.): gate -> modes (the t3 notes, no acquire-rule line) -> the switch note
#   (GET /admin/fn4 at start: gg_gemv 0 as libr4d reports it, aql_acq 0, epoch 0; gg_gemv 3 refused in the response, a
#   value this libr4d does not know; gg_gemv 1, 2, then 0 applied at a step) -> warm-up answers vs stock -> slow-state
#   probe -> BLOCKS gg_gemv 0 / 1 (ggblocks2.py, XGG_ORDER default A B B A A B, aql_acq pinned 0) -> BLOCKS gg_gemv 0 / 2
#   (the same) -> BLOCKS aql_acq 0 / 2 (gg_gemv pinned 0); each block POST {switch, pinned, epoch_bump}, stepprobe
#   essay/code 512 (c1), mb c4 and mb c8 (XGG_DUR 75 s, warm 10) -> per pair the summary line: B vs A % per load point
#   (c1 tok/s and ms, c4 / c8 aggregate, per-stream p50 and ms a step), pooled over blocks, each pair's ratio, t95 ->
#   IDENTITY (prefill-heavy, last), every value: gg_gemv 1 on the cold copy prompt (the 28k copy, the 20-prompt greedy
#   set at c1), then gg_gemv 2, then aql_acq 2 (gg_gemv 0), then A (all 0); each vs that A byte for byte, and the copy
#   vs 105 tuned3l x79's i1 copy. XGG_ORDER=ABBA runs four blocks a pair instead of six (two pairs, a wide t95).
# Reading: in-process blocks share one start (start-to-start spread ~1-1.5 % does not enter); both levers are exact,
#   so they need less evidence (TESTING 9.11.). References (no re-runs): bz's dsd-d0 (main image, NST 0): probe 59.38 /
#   62.04 tok/s, c4 122.7 (p50 31.6), c8 150.1 (p50 19.7).
# NO cc_sim (user 10.11.). No power / host / fan settings; production compose files read only (--project-directory).
# DRYRUN=1 bash dsm-gemvacq.sh : prints every command, runs and writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-gg-tp4.yaml
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=$R/xdsm/gg2; export STATSDIR=$O/stats
RX=$R/x79                                       # 105's tuned3l reference files
RD=$R/xdsm/draft                                # bz's dsd-d0 (main image, NST 0) files
export IMAGE=${XGG_IMAGE:-radiance-local:ds41-gg-8b311fcc-5c13e555-l}
ITEM=${XGG_ITEM:-57-p4zyzzzzz-ds-gz-gemvacq}; ORDER=${XGG_ORDER:-ABBAAB}; DUR=${XGG_DUR:-75}
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
GGB=$HD/ggblocks2.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV"
t3env() {   # 105's t3 arm, every other switch unset (RADIANCE_DS41_GG_GEMV unset: gg_gemv 0 at start)
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xgg2 $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xgg2 $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  case $r in IDENTICAL*) log "RESULT xgg2 $1 warm-up answers vs stock ($(basename $2)): $r" ;;
    *) log "RESULT xgg2 $1 warm-up answers vs stock ($(basename $2)): VIRHE $r -- NOT EXACT at gg_gemv 0" ;; esac
}
modes() {   # modes TAG: the serve log's notes must be the t3 line's -> 0, else 1 (stopped)
  [ "$DRY" = 1 ] && { echo "+ modes $1: indexer note (mxfp4, latent bf16, compressor tail bound, router selection+sort separate, AR_PCN separate at 0 layers), ARGATHER note, staging line, no acquire-rule line"; return 0; }
  local MLF SL ok=1
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  log "RESULT xgg2 $1 modes: $(echo "$MLF" | cut -c1-900) | staging: ${SL:-none}"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate" "post-collapse-norm separate at 0 layers" "next collapse-norm separate at 0 layers"; do
    case "$MLF" in *"$w"*) ;; *) log "RESULT xgg2 $1: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  docker logs $CN 2>&1 | grep -q 'the MoE gather and its all-reduce as one kernel' || { log "RESULT xgg2 $1: VIRHE no RADIANCE_DS41_ARGATHER note"; ok=0; }
  [ -n "$SL" ] || { log "RESULT xgg2 $1: VIRHE no prefill staging line (levels refused?)"; ok=0; }
  docker logs $CN 2>&1 | grep -q 'the acquire rule plays' && { log "RESULT xgg2 $1: VIRHE an acquire-rule line (RADIANCE_AQL_ACQ should be unset)"; ok=0; }
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
swnote() {   # swnote TAG: the switches at start (as their owners report them), gg_gemv 3 refused, gg_gemv 1, 2, back to 0
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (expect gg_gemv 0, aql_acq 0, epoch 0, draft_capable 0); POST gg_gemv 3 -> 400; POST gg_gemv 1 / 2 / 0, each + a 4-token request -> applied, plugins >= 1"; return 0; }
  local s0 s1 s2 s3 tl c3
  s0=$(fn4get); tl=$(docker logs $CN 2>&1 | grep -cE 'measured form table|takes the multi-row form')
  c3=$(fn4code '{"gg_gemv": 3}')
  fn4set '{"gg_gemv": 1}'; nudge; s1=$(fn4get)
  fn4set '{"gg_gemv": 2}'; nudge; s2=$(fn4get)
  fn4set '{"gg_gemv": 0}'; nudge; s3=$(fn4get)
  log "RESULT xgg2 $1 switch note: at start $s0 (form-table / multi-row lines in the serve log: $tl) | gg_gemv 3 -> HTTP $c3 | after gg_gemv 1: $s1 | after 2: $s2 | after 0: $s3 | core log: $(docker logs $CN 2>&1 | grep -E 'fn4: switches' | tail -3 | sed 's/^[A-Z] *//' | tr '\n' '|' | cut -c1-500)"
  python3 -c 'import json,sys
s0, s1, s2, s3 = (json.loads(x) for x in sys.argv[1:5])
ok = s0["switches"]["gg_gemv"] == 0 and s0["switches"]["aql_acq"] == 0 and s0["epoch"] == 0 and s1["switches"]["gg_gemv"] == 1 \
     and s1["plugins"] >= 1 and s2["switches"]["gg_gemv"] == 2 and s3["switches"]["gg_gemv"] == 0 \
     and not s1["pending"] and not s2["pending"] and not s3["pending"] and s3["epoch"] == 3
sys.exit(0 if ok else 1)' "$s0" "$s1" "$s2" "$s3" 2>/dev/null && [ "$tl" = 0 ] && [ "$c3" = 400 ]
}
mbms() { grep '^MB ' $1 2>/dev/null | head -1 | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()[3:]).get("ms_decode_step", ""))' 2>/dev/null; }
stprobe() {   # stprobe TAG N: mb c8 45 s -> "fast|between|slow MS (ref REF)"
  [ "$DRY" = 1 ] && { echo "+ (cd $D && timeout 400 python3 mb.py --tag xgg2-$1-st$2 --phases c8 --dur ${XGG_STATE_DUR:-45} --warm 10 > $O/$1-stprobe$2.txt)" >&2; echo "fast (dry)"; return 0; }
  run "(cd $D && timeout 400 python3 mb.py --tag xgg2-$1-st$2 --phases c8 --dur ${XGG_STATE_DUR:-45} --warm 10 > $O/$1-stprobe$2.txt 2>&1)"
  [ "$DRY" = 1 ] && { echo "fast (dry)"; return 0; }
  local ms; ms=$(mbms $O/$1-stprobe$2.txt)
  [ -n "$ms" ] || { echo "unknown (no mb line)"; return 0; }
  awk -v m=$ms -v r=$C8REF 'BEGIN{s=(m <= 1.07*r) ? "fast" : (m >= 1.14*r) ? "slow" : "between"; printf "%s %s ms (105 tuned3l c8 %s ms, x%.3f)\n", s, m, r, m/r}'
}
copyt() {   # copyt TAG: the 28k copy after a ~31k-token prompt (105's prompt, offset 0)
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset 0 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xgg2 $1 copy: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300)"
}
gsetc1() { run "(cd $D && timeout 900 python3 $GS $URL $O/$1-gs-c1.jsonl --conc 1 --max-tokens 192 > $O/$1-gs-c1.txt 2>&1)"; }
gcmp() { [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }; [ -s "$1" ] && [ -s "$2" ] || { echo "not compared (missing $1 or $2)"; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
idline() {   # idline LABEL RESULT: IDENTICAL -> plain, else VIRHE
  case "$2" in IDENTICAL*) log "RESULT xgg2 identity $1: $2" ;; "not compared"*) log "RESULT xgg2 identity $1: $2" ;;
    *) log "RESULT xgg2 identity $1: VIRHE $2 -- both levers are exact by construction, so this is a defect (or a reference from another line)" ;; esac
}
blocks() {   # blocks TAG SWITCH A B FIXED: the A/B blocks over SWITCH (FIXED pinned in every block) and their summary
  local tag=$1 sw=$2 a=$3 b=$4 fx=$5
  run "timeout 7200 python3 -I $GGB --url $URL --mb-dir $D --label $tag --switch $sw --a $a --b $b --fixed '$fx' --order $ORDER --dur $DUR --warm 10 --out $O/$tag-blocks.jsonl > $O/$tag-blocks.txt 2>&1"
  [ "$DRY" = 1 ] && return 0
  while IFS= read -r l; do log "RESULT xgg2 $tag ${l:0:400}"; done < <(grep '^BLOCK ' $O/$tag-blocks.txt)
  log "RESULT xgg2 $tag summary (B = $sw $b vs A = $a, $fx pinned, in process, order $ORDER, mb $DUR s): $(grep '^GGB ' $O/$tag-blocks.txt | cut -c5- | python3 -c 'import json,sys
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
log "RESULT xgg2 $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) order $ORDER mb $DUR s yaml $Y | table: $(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -m1 "^table" /opt/radiance/share/radiance/gg-PROVENANCE.txt' 2>/dev/null | cut -c1-200) | refs bz dsd-d0 (main image, NST 0): probe 59.38 / 62.04 tok/s, c4 122.7 (p50 31.6), c8 150.1 (p50 19.7)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xgg2: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xgg2: VIRHE image $IMAGE not on gpu02"; exit 1; }
  [ -e $Y ] && grep -q '^ *RADIANCE_DS41_GG_GEMV:' $Y && grep -q '^ *RADIANCE_AQL_ACQ:' $Y || { log "RESULT xgg2: VIRHE $Y missing or does not pass RADIANCE_DS41_GG_GEMV / RADIANCE_AQL_ACQ"; exit 1; }
  for f in $GS $GGB $D/stepprobe.py $D/mb.py; do [ -e $f ] || { log "RESULT xgg2: VIRHE $f missing"; exit 1; }; done
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xgg2: VIRHE $M225 missing or not verified"; exit 1; }
  [ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -q rad_fn4_accepts /opt/radiance/share/radiance/kernels/libr4d.so && grep -q r4d_gg_mr_pin /opt/radiance/share/radiance/kernels/libr4d.so && grep -q draft_capable /opt/radiance/bin/radiance && echo yes' 2>/dev/null)" = yes ] \
    || { log "RESULT xgg2: VIRHE $IMAGE is not the D1 + D2 image (no rad_fn4_accepts / r4d_gg_mr_pin in libr4d, or no draft_capable in radiance)"; exit 1; }
fi
mkd $O $O/guard $STATSDIR
evict
TAG=gg2; P=$O/gg2
t3env; modelvars225
log "RESULT xgg2 $TAG: start IMAGE=$IMAGE MODEL=$MODEL NST=$NST $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
mkd $P
gate $TAG $LIM; rc=$?
[ $rc = 1 ] && { log "RESULT xgg2 $TAG: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$TAG.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; exit 1; }
[ $rc = 2 ] && { log "RESULT xgg2 $TAG: VIRHE slow state after one restart -> stopped"; exit 1; }
modes $TAG || { log "RESULT xgg2 $TAG: VIRHE the modes are not the t3 line's -> stopped"; exit 1; }
swnote $TAG || { log "RESULT xgg2 $TAG: VIRHE the switches are not as expected (start gg_gemv 0 / aql_acq 0 / epoch 0, gg_gemv 3 refused, 1 / 2 / 0 applied at a step, the hook found) -> stopped"; keeplog $TAG-switch; exit 1; }
identw $TAG $WARMREF
st=$(stprobe $TAG 1)
if [ "${st%% *}" = slow ]; then
  log "RESULT xgg2 $TAG state probe 1: $st -> one restart"
  gate $TAG $LIM || { log "RESULT xgg2 $TAG: VIRHE no healthy / fast start after the slow-state restart"; exit 1; }
  modes $TAG || exit 1
  swnote $TAG || { log "RESULT xgg2 $TAG: VIRHE the switches are not as expected after the restart -> stopped"; exit 1; }
  st=$(stprobe $TAG 2)
fi
log "RESULT xgg2 $TAG state: $st (slow-state probe mb c8 ${XGG_STATE_DUR:-45} s)$(case "${st%% *}" in fast) ;; *) echo " -- NOT FAST: flag to the coordinator" ;; esac)"
follow $P
stopfile || blocks $TAG-gemv1 gg_gemv 0 1 '{"aql_acq": 0}'
stopfile || blocks $TAG-gemv2 gg_gemv 0 2 '{"aql_acq": 0}'
stopfile || blocks $TAG-acq aql_acq 0 2 '{"gg_gemv": 0}'
# identity (the prefill-heavy part last), every value: gg_gemv 1 on the cold copy prompt, then gg_gemv 2, then aql_acq 2,
# then A (all 0)
if ! stopfile; then
  for v in "gemv1:{\"gg_gemv\": 1, \"aql_acq\": 0, \"epoch_bump\": 1}" "gemv2:{\"gg_gemv\": 2, \"aql_acq\": 0, \"epoch_bump\": 1}" \
           "acq2:{\"gg_gemv\": 0, \"aql_acq\": 2, \"epoch_bump\": 1}" "A:{\"gg_gemv\": 0, \"aql_acq\": 0, \"epoch_bump\": 1}"; do
    nm=${v%%:*}
    fn4set "${v#*:}"
    copyt $TAG-$nm; gsetc1 $TAG-$nm
    [ "$DRY" = 1 ] || log "RESULT xgg2 $TAG identity $nm ran under: $(fn4get | cut -c1-200)"
  done
  if [ "$DRY" = 1 ]; then echo "+ identity: copy gemv1 / gemv2 / acq2 vs A, gemv1 vs x79; greedy set c1 gemv1 / gemv2 / acq2 vs A, A vs dsd-d0"
  else
    idline "28k copy gg_gemv 1 (cold prompt) vs A (all 0, prompt cached)" "$(cmpfile $O/copy-$TAG-gemv1-run1.txt $O/copy-$TAG-A-run1.txt)"
    idline "28k copy gg_gemv 2 vs A" "$(cmpfile $O/copy-$TAG-gemv2-run1.txt $O/copy-$TAG-A-run1.txt)"
    idline "28k copy aql_acq 2 vs A" "$(cmpfile $O/copy-$TAG-acq2-run1.txt $O/copy-$TAG-A-run1.txt)"
    idline "28k copy gg_gemv 1 vs 105 tuned3l x79-225pa-t3-i1" "$(cmpfile $O/copy-$TAG-gemv1-run1.txt $RX/copy-x79-225pa-t3-i1-run1.txt)"
    idline "greedy set c1 gg_gemv 1 vs A" "$(gcmp $O/$TAG-A-gs-c1.jsonl $O/$TAG-gemv1-gs-c1.jsonl)"
    idline "greedy set c1 gg_gemv 2 vs A" "$(gcmp $O/$TAG-A-gs-c1.jsonl $O/$TAG-gemv2-gs-c1.jsonl)"
    idline "greedy set c1 aql_acq 2 vs A" "$(gcmp $O/$TAG-A-gs-c1.jsonl $O/$TAG-acq2-gs-c1.jsonl)"
    idline "greedy set c1 A vs bz dsd-d0" "$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$TAG-A-gs-c1.jsonl)"
    log "RESULT xgg2 $TAG acquire-rule lines in the serve log: $(docker logs $CN 2>&1 | grep -c 'the acquire rule plays') (recorded passes under aql_acq 2), e.g. $(docker logs $CN 2>&1 | grep -m1 'the acquire rule plays' | sed 's/^[A-Z] *//' | cut -c1-220)"
  fi
fi
unfollow
run "docker logs $CN > $O/$TAG-engine-full.log 2>&1"
down
unset $SWITCHES
down
log "RESULT xgg2 $ITEM: done -- $O"
