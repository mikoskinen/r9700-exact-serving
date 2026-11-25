#!/bin/bash
# dsm-gemv.sh (10.11.2026, queue-ar/dsmain, ds41-gg lever 1 agent) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-g-gemv.sh. Lever 1 (libr4d's gemm_nt_ggml decode band at a measured form a (type, N, K, M): rows a wave
# and grid cap from one card's bench; EXACT by construction, every form byte-compared in the bench) A/B'd IN ONE PROCESS:
# DeepSeek-V4.1 Flash EXL3 2.25, TP4, MML 524288, NST 0, the t3 env and serve line of dsm-serve.sh (105's t3 arm + lib-29
# best_vars + MNBT 4160 + --live), on the ds41-gg image (XGG_IMAGE) = the main -l image + bin/radiance (the core's runtime
# switches: POST /admin/fn4, applied at a step release with every rank parked, an epoch in the pass key) + libr4d (the
# switch gg_gemv: 0 the stock rule = today's launch byte for byte, 1 the form table; RADIANCE_DS41_GG_GEMV unset = 0).
# ORDER (decode before prefill, TESTING 9.11.): gate (restart, warm-up incl. 16k prefill, 1-user fastcheck; one retry over
#   23 ms) -> modes (the t3 notes) -> the switch note (GET /admin/fn4 at start: gg_gemv 0, epoch 0, no form-table line in
#   the serve log; then gg_gemv 1 and back to 0, each applied at a step: the hook is found in libr4d) -> warm-up answers vs
#   stock -> slow-state probe (mb c8 45 s vs 105's 43.1 ms) -> BLOCKS (ggblocks.py, XGG_ORDER default A B B A A B, A =
#   gg_gemv 0, B = 1): each block POST {gg_gemv, epoch_bump} then stepprobe essay/code 512 (c1), mb c4 and mb c8
#   (XGG_DUR 75 s, warm 10) -> the summary line: B vs A % per load point (c1 tok/s, c4 / c8 aggregate and per-stream p50,
#   ms a step), pooled over blocks, each pair's ratio and a t95 interval -> IDENTITY under each setting (the prefill-heavy
#   part, last): B first (the 28k copy after a cold ~31k prompt, the 20-prompt greedy set at c1), then A (the copy's
#   prompt now cached): B vs A byte for byte, and each vs the stock references (105 tuned3l x79-225pa-t3's i1 copy, bz's
#   dsd-d0 copy and greedy set on the main image) -> PROFILE (XGG_PROF=1): one --profile-ops start, A / B / A / B windows
#   (stepprobe warm-up, mb c4 30 s, mb c8 30 s each) -> ds_opprof.py: rank 0's gemm_nt_ggml us a call and dense-GEMM ms a
#   step per window (PROFILED: not a speed).
# Reading: in-process blocks share one start (start-to-start spread ~1-1.5 % does not enter); an exact lever needs less
#   evidence (TESTING 9.11.). References (no re-runs): bz's dsd-d0 (main image, NST 0): probe 59.38 / 62.04 tok/s (16.84 /
#   16.11 ms), c4 122.7 (p50 31.6, 28.1 ms), c8 150.1 (p50 19.7, 44.0 ms).
# NO cc_sim (user 10.11.). No power / host / fan settings; production compose files read only (--project-directory).
# DRYRUN=1 bash dsm-gemv.sh : prints every command, runs and writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-gg-tp4.yaml
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=$R/xdsm/gg; export STATSDIR=$O/stats
RX=$R/x79                                       # 105's tuned3l reference files
RD=$R/xdsm/draft                                # bz's dsd-d0 (main image, NST 0) files
export IMAGE=${XGG_IMAGE:-radiance-local:ds41-gg-4d70490d-5c13e555-l}
ITEM=${XGG_ITEM:-57-p4zyzzzzz-ds-g-gemv}; ORDER=${XGG_ORDER:-ABBAAB}; DUR=${XGG_DUR:-75}; GPROF=${XGG_PROF-1}
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
GGB=$HD/ggblocks.py; OPP=$HD/ds_opprof.py
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xgg $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xgg $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  case $r in IDENTICAL*) log "RESULT xgg $1 warm-up answers vs stock ($(basename $2)): $r" ;;
    *) log "RESULT xgg $1 warm-up answers vs stock ($(basename $2)): VIRHE $r -- NOT EXACT at gg_gemv 0" ;; esac
}
modes() {   # modes TAG: the serve log's notes must be the t3 line's -> 0, else 1 (stopped)
  [ "$DRY" = 1 ] && { echo "+ modes $1: indexer note (mxfp4, latent bf16, compressor tail bound, router selection+sort separate, AR_PCN separate at 0 layers), ARGATHER note, staging line, no acquire-rule line"; return 0; }
  local MLF SL ok=1
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  log "RESULT xgg $1 modes: $(echo "$MLF" | cut -c1-900) | staging: ${SL:-none}"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate" "post-collapse-norm separate at 0 layers" "next collapse-norm separate at 0 layers"; do
    case "$MLF" in *"$w"*) ;; *) log "RESULT xgg $1: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  docker logs $CN 2>&1 | grep -q 'the MoE gather and its all-reduce as one kernel' || { log "RESULT xgg $1: VIRHE no RADIANCE_DS41_ARGATHER note"; ok=0; }
  [ -n "$SL" ] || { log "RESULT xgg $1: VIRHE no prefill staging line (levels refused?)"; ok=0; }
  docker logs $CN 2>&1 | grep -q 'the acquire rule plays' && { log "RESULT xgg $1: VIRHE an acquire-rule line (RADIANCE_AQL_ACQ should be unset)"; ok=0; }
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
swnote() {   # swnote TAG: the switch at start, then 1 and back to 0, each applied at a step -> 0 ok, 1 not switchable
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (expect gg_gemv 0, epoch 0); POST gg_gemv 1 + a 4-token request -> gg_gemv 1, plugins >= 1; POST 0 -> 0"; return 0; }
  local s0 s1 s2 tl
  s0=$(fn4get); tl=$(docker logs $CN 2>&1 | grep -c 'measured form table')
  fn4set '{"gg_gemv": 1}'; nudge; s1=$(fn4get)
  fn4set '{"gg_gemv": 0}'; nudge; s2=$(fn4get)
  log "RESULT xgg $1 switch note: at start $s0 (form-table lines in the serve log: $tl) | after gg_gemv 1: $s1 | after 0: $s2 | core log: $(docker logs $CN 2>&1 | grep -E 'fn4: switches' | tail -2 | sed 's/^[A-Z] *//' | tr '\n' '|' | cut -c1-400)"
  python3 -c 'import json,sys
s0, s1, s2 = (json.loads(x) for x in sys.argv[1:4])
ok = s0["switches"]["gg_gemv"] == 0 and s0["epoch"] == 0 and s1["switches"]["gg_gemv"] == 1 and s1["plugins"] >= 1 \
     and s2["switches"]["gg_gemv"] == 0 and not s1["pending"] and not s2["pending"] and s2["epoch"] == 2
sys.exit(0 if ok else 1)' "$s0" "$s1" "$s2" 2>/dev/null && [ "$tl" = 0 ]
}
mbms() { grep '^MB ' $1 2>/dev/null | head -1 | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()[3:]).get("ms_decode_step", ""))' 2>/dev/null; }
stprobe() {   # stprobe TAG N: mb c8 45 s -> "fast|between|slow MS (ref REF)"
  [ "$DRY" = 1 ] && { echo "+ (cd $D && timeout 400 python3 mb.py --tag xgg-$1-st$2 --phases c8 --dur ${XGG_STATE_DUR:-45} --warm 10 > $O/$1-stprobe$2.txt)" >&2; echo "fast (dry)"; return 0; }
  run "(cd $D && timeout 400 python3 mb.py --tag xgg-$1-st$2 --phases c8 --dur ${XGG_STATE_DUR:-45} --warm 10 > $O/$1-stprobe$2.txt 2>&1)"
  [ "$DRY" = 1 ] && { echo "fast (dry)"; return 0; }
  local ms; ms=$(mbms $O/$1-stprobe$2.txt)
  [ -n "$ms" ] || { echo "unknown (no mb line)"; return 0; }
  awk -v m=$ms -v r=$C8REF 'BEGIN{s=(m <= 1.07*r) ? "fast" : (m >= 1.14*r) ? "slow" : "between"; printf "%s %s ms (105 tuned3l c8 %s ms, x%.3f)\n", s, m, r, m/r}'
}
copyt() {   # copyt TAG: the 28k copy after a ~31k-token prompt (105's prompt, offset 0)
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset 0 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xgg $1 copy: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300)"
}
gsetc1() { run "(cd $D && timeout 900 python3 $GS $URL $O/$1-gs-c1.jsonl --conc 1 --max-tokens 192 > $O/$1-gs-c1.txt 2>&1)"; }
gcmp() { [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }; [ -s "$1" ] && [ -s "$2" ] || { echo "not compared (missing $1 or $2)"; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
idline() {   # idline LABEL RESULT: IDENTICAL -> plain, else VIRHE
  case "$2" in IDENTICAL*) log "RESULT xgg identity $1: $2" ;; "not compared"*) log "RESULT xgg identity $1: $2" ;;
    *) log "RESULT xgg identity $1: VIRHE $2 -- the form table is exact by construction, so this is a defect (or a reference from another line)" ;; esac
}
blocks() {   # blocks TAG: the A/B blocks and their summary
  run "timeout 7200 python3 -I $GGB --url $URL --mb-dir $D --label $1 --switch gg_gemv --a 0 --b 1 --order $ORDER --dur $DUR --warm 10 --out $O/$1-blocks.jsonl > $O/$1-blocks.txt 2>&1"
  [ "$DRY" = 1 ] && return 0
  while IFS= read -r l; do log "RESULT xgg $1 ${l:0:400}"; done < <(grep '^BLOCK ' $O/$1-blocks.txt)
  log "RESULT xgg $1 summary (B = gg_gemv 1 vs A = 0, in process, order $ORDER, mb $DUR s): $(grep '^GGB ' $O/$1-blocks.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=2: "-" if v is None else format(v, ".%df" % n)
out = []
for k, v in d["metrics"].items():
    p = v.get("pooled")
    out.append("%s %s -> %s (%s %%; pairs %s; t95 +- %s %%)" % (k, f(v.get("A")), f(v.get("B")), "-" if p is None else "%+.2f" % (100 * (p - 1)),
               " ".join("%+.2f" % (100 * (x - 1)) for x in v.get("pairs", [])), "-" if v.get("ci95") is None else "%.2f" % (100 * v["ci95"])))
print("; ".join(out) + " | %d blocks, %d flagged, %s s" % (d["blocks"], d["blocks_bad"], d["wall_s"]))' 2>&1 | cut -c1-2400)"
}
gprof() {   # gprof: one --profile-ops start; A / B / A / B windows -> rank 0's gemm_nt_ggml us a call per window
  local tag=gg-prof E l0 l1 l2 arm v n=0 ms
  t3env; modelvars225
  export PROFEVERY=${XGG_PROFEVERY:-16} XARGS="$XARGS --profile-ops" STATSDIR=$O/stats-prof
  mkd $STATSDIR; E=$O/$tag-engine.log
  log "RESULT xgg $tag: start IMAGE=$IMAGE --profile-ops RADIANCE_PROFILE_EVERY=$PROFEVERY (PROFILED: not a speed) $(desc)"
  restart $tag || { keeplog $tag-fail; down; log "RESULT xgg $tag: VIRHE not healthy -- no profile"; export PROFEVERY= STATSDIR=$O/stats; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] || log "RESULT xgg $tag warm-up + 1-user fastcheck (PROFILED, not gated): $ms ms/step"
  nl() { [ "$DRY" = 1 ] && echo L || docker logs $CN 2>&1 | wc -l; }
  run ": > $O/$tag-win.txt"
  for arm in A B A B; do
    v=0; [ $arm = B ] && v=1; n=$((n + 1))
    fn4set "{\"gg_gemv\": $v, \"epoch_bump\": 1}"
    run "(cd $D && timeout 600 python3 stepprobe.py essay 128 > $O/$tag-w$n$arm-warm.txt 2>&1)"
    l0=$(nl)
    run "(cd $D && timeout 600 python3 mb.py --tag xgg-$tag-w$n$arm --phases c4 --dur 30 --warm 5 > $O/$tag-w$n$arm-c4.txt 2>&1)"
    run "sleep 3"; l1=$(nl)
    run "(cd $D && timeout 600 python3 mb.py --tag xgg-$tag-w$n$arm --phases c8 --dur 30 --warm 5 > $O/$tag-w$n$arm-c8.txt 2>&1)"
    run "sleep 3"; l2=$(nl)
    run "echo 'w$n$arm $v $l0 $l1 $l2 $(fn4get | tr -d " " | cut -c1-200)' >> $O/$tag-win.txt"
  done
  run "docker stop -t 60 $CN > /dev/null 2>&1; docker logs $CN > $E 2>&1"
  down
  if [ "$DRY" = 1 ]; then echo "+ ds_opprof.py $E per window (c4 L0..L1, c8 L1..L2) -> rank 0 gemm_nt_ggml us a call, dense_gemm ms a step"
  else
    while read -r w v l0 l1 l2 st; do
      timeout 900 python3 -I $OPP $E --from $l0 --to $l1 --label $tag-$w-c4 --json $O/$tag-$w-c4-op.json > $O/$tag-$w-c4-op.txt 2>&1
      timeout 900 python3 -I $OPP $E --from $l1 --to $l2 --label $tag-$w-c8 --json $O/$tag-$w-c8-op.json > $O/$tag-$w-c8-op.txt 2>&1
    done < $O/$tag-win.txt
    log "RESULT xgg $tag (PROFILED, not a speed; rank 0, gemm_nt_ggml us a call | dense_gemm ms a step): $(python3 - $O $tag <<'PY' 2>&1 | cut -c1-1600
import json, sys
O, tag = sys.argv[1:3]
out = []
for l in open(f"{O}/{tag}-win.txt"):
    w, v = l.split()[:2]
    for ph in ("c4", "c8"):
        try: d = json.load(open(f"{O}/{tag}-{w}-{ph}-op.json"))
        except Exception: out.append(f"{w} gg_gemv {v} {ph}: no table"); continue
        c, ms = d["rank0_by_op"].get("gemm_nt_ggml", [0, 0.0])[:2]
        s0 = d["summary"][sorted(d["summary"], key=int)[0]]
        out.append(f"{w} gg_gemv {v} {ph}: {1000 * ms / max(1, c):.1f} us x {c} calls | dense {s0['by_class_ms'].get('dense_gemm', 0) / max(1, s0['steps']):.2f} ms a step")
print("; ".join(out))
PY
)"
    run "gzip -f $E"
  fi
  export PROFEVERY= STATSDIR=$O/stats
  log "RESULT xgg $tag: done"
}
# ---------------------------------------------------------------- main
log "RESULT xgg $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) order $ORDER mb $DUR s prof $GPROF yaml $Y | refs bz dsd-d0 (main image, NST 0): probe 59.38 / 62.04 tok/s, c4 122.7 (p50 31.6), c8 150.1 (p50 19.7)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xgg: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xgg: VIRHE image $IMAGE not on gpu02"; exit 1; }
  [ -e $Y ] && grep -q '^ *RADIANCE_DS41_GG_GEMV:' $Y || { log "RESULT xgg: VIRHE $Y missing or does not pass RADIANCE_DS41_GG_GEMV"; exit 1; }
  for f in $GS $GGB $OPP $D/stepprobe.py $D/mb.py; do [ -e $f ] || { log "RESULT xgg: VIRHE $f missing"; exit 1; }; done
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xgg: VIRHE $M225 missing or not verified"; exit 1; }
  [ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -q rad_fn4_switch /opt/radiance/share/radiance/kernels/libr4d.so && grep -q admin/fn4 /opt/radiance/bin/radiance && echo yes' 2>/dev/null)" = yes ] \
    || { log "RESULT xgg: VIRHE $IMAGE has no fn4 switch in libr4d / radiance"; exit 1; }
fi
mkd $O $O/guard $STATSDIR
evict
TAG=gg; P=$O/gg
t3env; modelvars225
log "RESULT xgg $TAG: start IMAGE=$IMAGE MODEL=$MODEL NST=$NST $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
mkd $P
gate $TAG $LIM; rc=$?
[ $rc = 1 ] && { log "RESULT xgg $TAG: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$TAG.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; exit 1; }
[ $rc = 2 ] && { log "RESULT xgg $TAG: VIRHE slow state after one restart -> stopped"; exit 1; }
modes $TAG || { log "RESULT xgg $TAG: VIRHE the modes are not the t3 line's -> stopped"; exit 1; }
swnote $TAG || { log "RESULT xgg $TAG: VIRHE the switch is not as expected (start 0 / epoch 0 / hook found / applied at a step) -> stopped"; keeplog $TAG-switch; exit 1; }
identw $TAG $WARMREF
st=$(stprobe $TAG 1)
if [ "${st%% *}" = slow ]; then
  log "RESULT xgg $TAG state probe 1: $st -> one restart"
  gate $TAG $LIM || { log "RESULT xgg $TAG: VIRHE no healthy / fast start after the slow-state restart"; exit 1; }
  modes $TAG || exit 1
  swnote $TAG || { log "RESULT xgg $TAG: VIRHE the switch is not as expected after the restart -> stopped"; exit 1; }
  st=$(stprobe $TAG 2)
fi
log "RESULT xgg $TAG state: $st (slow-state probe mb c8 ${XGG_STATE_DUR:-45} s)$(case "${st%% *}" in fast) ;; *) echo " -- NOT FAST: flag to the coordinator" ;; esac)"
follow $P
stopfile || blocks $TAG
# identity (the prefill-heavy part last): B first, on a cold copy prompt; then A
if ! stopfile; then
  fn4set '{"gg_gemv": 1, "epoch_bump": 1}'
  copyt $TAG-B; gsetc1 $TAG-B
  [ "$DRY" = 1 ] || log "RESULT xgg $TAG identity runs under: $(fn4get | cut -c1-200)"
  fn4set '{"gg_gemv": 0, "epoch_bump": 1}'
  copyt $TAG-A; gsetc1 $TAG-A
  [ "$DRY" = 1 ] || log "RESULT xgg $TAG identity runs under: $(fn4get | cut -c1-200)"
  if [ "$DRY" = 1 ]; then echo "+ identity: copy B vs A, B vs x79 / dsd-d0; greedy set c1 B vs A, A vs dsd-d0"
  else
    idline "28k copy B (gg_gemv 1, cold prompt) vs A (0, prompt cached)" "$(cmpfile $O/copy-$TAG-B-run1.txt $O/copy-$TAG-A-run1.txt)"
    idline "28k copy B vs 105 tuned3l x79-225pa-t3-i1" "$(cmpfile $O/copy-$TAG-B-run1.txt $RX/copy-x79-225pa-t3-i1-run1.txt)"
    idline "28k copy B vs bz dsd-d0 (main image, NST 0)" "$(cmpfile $O/copy-$TAG-B-run1.txt $RD/copy-dsd-d0-i1-run1.txt)"
    idline "greedy set c1 B vs A" "$(gcmp $O/$TAG-A-gs-c1.jsonl $O/$TAG-B-gs-c1.jsonl)"
    idline "greedy set c1 A vs bz dsd-d0" "$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$TAG-A-gs-c1.jsonl)"
  fi
fi
unfollow
run "docker logs $CN > $O/$TAG-engine-full.log 2>&1"
down
[ "$GPROF" = 1 ] && ! stopfile && gprof
unset $SWITCHES
down
log "RESULT xgg $ITEM: done -- $O"
