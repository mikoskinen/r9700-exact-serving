#!/bin/bash
# dsm-argall.sh (10.11.2026, queue-ar/dsmain, ds41-d3 agent, lever D3) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-i-argall.sh. Lever D3: the MoE gather + its all-reduce as one kernel (moe_gather_all_reduce) at every
# decode-band row count, A/B'd IN ONE PROCESS on one start of the ds41-d3 image (XD3_IMAGE), plus libr4d's byte check.
#   What the lever is: today RADIANCE_DS41_ARGATHER=on fuses the gather and the all-reduce only up to 4 rows; c8, the mixed
#   steps and any drafting verify run moe_gather + all_reduce (37 extra launches a c8 step). RADIANCE_DS41_ARGATHER_MAX=48
#   declares the fused op up to 48 rows: the one-shot below the all-reduce's ts_min (7 rows of 5120), from it the new
#   two-shot form (the gather as the plain two-shot's phase-1 source; libr4d r4d_ar_gather_twoshot_kernel). The in-process
#   switch ds41_argather sets the live bound: 4 = today's issue (A), 48 = the lever (B); both forms stay declared.
#   EXACTNESS: every form gathers a token's slots ON ITS RANK in ascending slot order (fp32, fmaf with the f32 router
#   weights), rounds that rank partial (with the shared arm's) to bf16 and sums the four ranks' partials in fp32 in
#   ascending rank order, rounded once -- the arithmetic of moe_gather + all_reduce at every row count. So the lever changes
#   NO byte at any batch (A and B must give identical text everywhere).
#   BATCH INVARIANCE (coordinator's acceptance item "c4 greedy set IDENTICAL against itself"): the <= 4-row fused form is
#   NOT a rank-independent slot-order sum; it is moe_gather + all_reduce byte for byte (libr4d says so, r4d_fuse.cpp's
#   chain and this image's selftest hold it). So the 6-of-20 c4 floor does NOT come from ARGATHER's 4-row edge and D3
#   cannot remove it; the item measures the floor under both arms (c4 set twice each) to show it, and reports it as a
#   finding, not a failure. (Candidates in the agent's report: row-count-dependent forms elsewhere -- the act path's
#   workgroup width at 32 / 120 rows, gemm_nt_ggml's tiled form past the decode band, exl3r4d's run / map form at
#   2T > 3 n_expert -- crossed by the mixed steps a new prompt makes beside the decoding ones.)
# DeepSeek-V4.1 Flash EXL3 2.25, TP4, MML 524288, NST 0, the t3 env and serve line of dsm-serve.sh (105's t3 arm + lib-29
# best_vars + MNBT 4160 + --live), RADIANCE_DS41_ARGATHER=on + RADIANCE_DS41_ARGATHER_MAX=48, every other switch unset.
# ORDER (decode before prefill, TESTING 9.11.):
#   SELFTEST (no server, all four cards, before the start): the image's r4d_selftest --case moe_gather_all_reduce_4rank
#     (every M 1..48, 56, 64 at n 5120 / top_k 6 / f32 weights / the ungated shared arm, expert-parallel routing, three
#     launches a shape on one instance: two-shot / one-shot / two-shot from 7 rows, held byte for byte to the slot-order
#     oracle on every rank; generic arms; the one-shot fallback past the two-shot's slot list) + its _host twin +
#     ar_twoshot_4rank_exact / ar_oneshot_4rank_exact / ar_gather_hc_write_4rank (the instance's other rows) as regressions;
#     each case its own bounded container (gpurun). A FAIL stops the item before the start.
#   gate -> modes (the t3 notes + the ARGATHER note "up to 48 ... RADIANCE_DS41_ARGATHER_MAX") -> the switch note (GET
#     /admin/fn4 at start: ds41_argather 48, ds41_shexp_lane 0, gg_gemv 0, aql_acq 0, epoch 0; ds41_argather 49 refused
#     (past the declared bound), 4 then 48 applied at a step) -> warm-up answers vs stock -> slow-state probe
#   -> BLOCKS ds41_argather 4 / 48 (ggblocks2.py, XD3_ORDER default A B B A A B, gg_gemv / aql_acq / ds41_shexp_lane pinned
#     0): each block POST {switch, pinned, epoch_bump}, stepprobe essay/code 512 (c1), mb c4 and mb c8 (XD3_DUR 75 s, warm
#     10) -> the summary line: B vs A % per load point (c1 tok/s and ms, c4 / c8 aggregate, per-stream p50, ms a step)
#     pooled over blocks, each pair's ratio, t95
#   -> BATCH FLOOR: the 20-prompt greedy set at c4 twice under B (48) and twice under A (4): B1 vs B2, A1 vs A2 (each arm's
#     own run-to-run floor) and B1 vs A1
#   -> IDENTITY (prefill-heavy, last): under B then A, the 28k copy (a cold ~31k prompt) and the greedy set at c1; each vs
#     bz's dsd-d0 files (the main image, NST 0) and B vs A, byte for byte. No 16k / 136k prefill pair: D3 does not touch a
#     step wider than 48 rows (every prefill chunk), so prefill runs today's ops at both values.
# References (no re-runs): bz's dsd-d0 (main image, NST 0): probe 59.38 / 62.04 tok/s, c4 122.7 (p50 31.6), c8 150.1 (p50
#   19.7); d5 profile (c8 NST 0): all_reduce 78 calls a step (41 + 37 MoE), 145.9 us a call, plus 37 moe_gather.
# The image (XD3_IMAGE, build/d3/assemble.sh): ds41-d3 = ds41-gg fbd4c8a + D3 + F4, merged with ds41-gg d5bed91 (--spec-ngram
#   leaves a draft head out) and ds41-miss 1f4754e (RADIANCE_DEBUG_MISS_LOG), layered on radiance-local:ds41-gg-fbd4c8ac-5c13e555-l.
# NO cc_sim (user 10.11.). No power / host / fan settings; production compose files read only (--project-directory).
# DRYRUN=1 bash dsm-argall.sh : prints every command, runs and writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-d3-tp4.yaml
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=$R/xdsm/argall; export STATSDIR=$O/stats
RD=$R/xdsm/draft                                # bz's dsd-d0 (main image, NST 0) files
RM=$R/xdsm                                      # item a's dsm-225 (main image, NST 0) files
export IMAGE=${XD3_IMAGE:-radiance-local:ds41-d3-c03c57d9-5c13e555-l}
ITEM=${XD3_ITEM:-57-p4zyzzzzz-ds-i-argall}; ORDER=${XD3_ORDER:-ABBAAB}; DUR=${XD3_DUR:-75}
GARMAX=${XD3_MAX:-48}; GARA=${XD3_A:-4}
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
GGB=$HD/ggblocks2.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ulimit memlock=-1 --ipc=host -e HIP_VISIBLE_DEVICES=0,1,2,3"
CASES=${XD3_CASES:-"moe_gather_all_reduce_4rank moe_gather_all_reduce_4rank_host ar_twoshot_4rank_exact ar_oneshot_4rank_exact ar_gather_hc_write_4rank"}
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DS41_ARGATHER_MAX RADIANCE_DS41_SHEXP_LANE"
t3env() {   # 105's t3 arm + ARGATHER_MAX, every other switch unset (gg_gemv 0, aql_acq 0, shexp_lane 0 at start)
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0 \
         RADIANCE_DS41_ARGATHER_MAX=$GARMAX
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
gpurun() {   # gpurun NAME SECONDS docker-run-args...: bounded, named, removed even if its kernel hangs
  local name=$1 t=$2; shift 2
  timeout -k 15 $t docker run --rm --init --stop-timeout 10 --name $name "$@"; local rc=$?
  docker rm -f $name > /dev/null 2>&1 || true
  { [ $rc = 124 ] || [ $rc = 137 ]; } && echo "gpurun: $name stopped after $t s (rc $rc)" >&2
  return $rc
}
selftest() {   # the four-rank byte checks, each case its own bounded container; 0 = every case clean
  local c rc nf nok ns bad=0 r
  mkd $O/selftest
  [ "$DRY" = 1 ] || rm -f $O/selftest/SUMMARY.txt
  for c in $CASES; do
    run "gpurun st-$$-${c:0:20} 900 $DEV --entrypoint /opt/radiance/bin/r4d_selftest $IMAGE /opt/radiance/share/radiance/kernels/libr4d.so --case $c > $O/selftest/$c.txt 2>&1"
    rc=$?
    [ "$DRY" = 1 ] && continue
    nf=$(grep -c 'FAIL' $O/selftest/$c.txt); nok=$(grep -cE '^ *(ok|PASS)' $O/selftest/$c.txt); ns=$(grep -ciE '^ *skip' $O/selftest/$c.txt)
    { [ $rc = 0 ] && [ $nf = 0 ] && [ $nok -gt 0 ]; } || bad=$((bad + 1))
    r="$c: rc $rc, $nok ok, $nf FAIL, $ns skip$([ $nf -gt 0 ] && echo " -- first: $(grep -m1 FAIL $O/selftest/$c.txt | cut -c1-300)")$({ [ $rc = 124 ] || [ $rc = 137 ]; } && echo ' -- STOPPED by the limit (hang)')"
    echo "$r" >> $O/selftest/SUMMARY.txt; log "RESULT xd3 selftest $r"
    [ $c = moe_gather_all_reduce_4rank ] && log "RESULT xd3 selftest $c transports: $(grep -m1 'shapes took the two-shot' $O/selftest/$c.txt | sed 's/^ *//')"
  done
  [ "$DRY" = 1 ] && return 0
  [ $bad = 0 ] && { echo "D3 SELFTEST CLEAN" >> $O/selftest/SUMMARY.txt; return 0; }
  echo "D3 SELFTEST NOT CLEAN ($bad case(s))" >> $O/selftest/SUMMARY.txt; return 1
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
modes() {   # modes TAG: the serve log's notes must be the t3 line's + the ARGATHER_MAX note -> 0, else 1 (stopped)
  [ "$DRY" = 1 ] && { echo "+ modes $1: indexer note (mxfp4, latent bf16, compressor tail bound, router selection+sort separate, AR_PCN separate at 0 layers), ARGATHER note up to $GARMAX with RADIANCE_DS41_ARGATHER_MAX, staging line, no acquire-rule line, no shared-expert-lane line"; return 0; }
  local MLF SL AL ok=1
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  AL=$(docker logs $CN 2>&1 | grep -m1 'the MoE gather and its all-reduce as one kernel' | sed 's/^radiance: //' | cut -c1-400)
  log "RESULT xd3 $1 modes: $(echo "$MLF" | cut -c1-900) | staging: ${SL:-none} | argather: ${AL:-none}"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate" "post-collapse-norm separate at 0 layers" "next collapse-norm separate at 0 layers"; do
    case "$MLF" in *"$w"*) ;; *) log "RESULT xd3 $1: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  case "$AL" in *"up to $GARMAX tokens a step"*RADIANCE_DS41_ARGATHER_MAX*) ;; *) log "RESULT xd3 $1: VIRHE the ARGATHER note does not say up to $GARMAX / RADIANCE_DS41_ARGATHER_MAX: '${AL}'"; ok=0 ;; esac
  [ -n "$SL" ] || { log "RESULT xd3 $1: VIRHE no prefill staging line (levels refused?)"; ok=0; }
  docker logs $CN 2>&1 | grep -q 'the acquire rule plays' && { log "RESULT xd3 $1: VIRHE an acquire-rule line (RADIANCE_AQL_ACQ should be unset)"; ok=0; }
  docker logs $CN 2>&1 | grep -q 'the shared expert on the second lane' && { log "RESULT xd3 $1: VIRHE a shared-expert-lane line (RADIANCE_DS41_SHEXP_LANE should be unset)"; ok=0; }
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
swnote() {   # swnote TAG: the switches at start, ds41_argather past the bound refused, A then B applied at a step
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (expect ds41_argather $GARMAX, ds41_shexp_lane 0, gg_gemv 0, aql_acq 0, epoch 0); POST ds41_argather $((GARMAX + 1)) -> 400; POST ds41_argather $GARA / $GARMAX, each + a 4-token request -> applied"; return 0; }
  local s0 s1 s2 c9
  s0=$(fn4get)
  c9=$(fn4code "{\"ds41_argather\": $((GARMAX + 1))}")
  fn4set "{\"ds41_argather\": $GARA}"; nudge; s1=$(fn4get)
  fn4set "{\"ds41_argather\": $GARMAX}"; nudge; s2=$(fn4get)
  log "RESULT xd3 $1 switch note: at start $s0 | ds41_argather $((GARMAX + 1)) -> HTTP $c9 | after $GARA: $s1 | after $GARMAX: $s2 | core log: $(docker logs $CN 2>&1 | grep -E 'fn4: switches' | tail -2 | sed 's/^[A-Z] *//' | tr '\n' '|' | cut -c1-400)"
  python3 -c 'import json,sys
s0, s1, s2 = (json.loads(x) for x in sys.argv[1:4]); a, m = int(sys.argv[4]), int(sys.argv[5])
w = lambda s, k: s["switches"].get(k)
ok = w(s0, "ds41_argather") == m and w(s0, "ds41_shexp_lane") == 0 and w(s0, "gg_gemv") == 0 and w(s0, "aql_acq") == 0 \
     and s0["epoch"] == 0 and w(s1, "ds41_argather") == a and w(s2, "ds41_argather") == m and not s1["pending"] \
     and not s2["pending"] and s2["epoch"] == 2
sys.exit(0 if ok else 1)' "$s0" "$s1" "$s2" $GARA $GARMAX 2>/dev/null && [ "$c9" = 400 ]
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
    *) log "RESULT xd3 identity $1: VIRHE $2 -- D3 is exact by construction, so this is a defect (or a reference from another line)" ;; esac
}
floorline() {   # floorline LABEL RESULT: a run-to-run c4 comparison -- a finding either way
  log "RESULT xd3 batch floor $1: $2"
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
log "RESULT xd3 $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) A ds41_argather $GARA / B $GARMAX, order $ORDER mb $DUR s yaml $Y | $(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -m1 "^radiance" /opt/radiance/share/radiance/d3-PROVENANCE.txt' 2>/dev/null | cut -c1-200) | refs bz dsd-d0 (main image, NST 0): probe 59.38 / 62.04 tok/s, c4 122.7 (p50 31.6), c8 150.1 (p50 19.7)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xd3: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xd3: VIRHE image $IMAGE not on gpu02"; exit 1; }
  [ -e $Y ] && grep -q '^ *RADIANCE_DS41_ARGATHER_MAX:' $Y && grep -q '^ *RADIANCE_DS41_SHEXP_LANE:' $Y || { log "RESULT xd3: VIRHE $Y missing or does not pass RADIANCE_DS41_ARGATHER_MAX / RADIANCE_DS41_SHEXP_LANE"; exit 1; }
  for f in $GS $GGB $D/stepprobe.py $D/mb.py; do [ -e $f ] || { log "RESULT xd3: VIRHE $f missing"; exit 1; }; done
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xd3: VIRHE $M225 missing or not verified"; exit 1; }
  [ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -q RADIANCE_DS41_ARGATHER_MAX /opt/radiance/share/radiance/architectures/deepseek41.so && grep -q ds41_shexp_lane /opt/radiance/bin/radiance && grep -q r4d_ar_gather_twoshot_4rank /opt/radiance/share/radiance/kernels/libr4d.so && echo yes' 2>/dev/null)" = yes ] \
    || { log "RESULT xd3: VIRHE $IMAGE is not the D3 image (no RADIANCE_DS41_ARGATHER_MAX in deepseek41.so, ds41_shexp_lane in radiance or the two-shot gather in libr4d)"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT xd3: VIRHE a KFD client is running (after 60 s wait) -- not the selftest's cards"; exit 1; }
fi
mkd $O $O/guard $STATSDIR
# ---- the byte check first, with no server: a FAIL stops the item
if ! stopfile; then
  selftest || { log "RESULT xd3 $ITEM: VIRHE the selftest is not clean ($O/selftest/SUMMARY.txt) -> no serving phase"; exit 1; }
  [ "$DRY" = 1 ] || log "RESULT xd3 selftest: $(tail -1 $O/selftest/SUMMARY.txt)"
fi
evict
TAG=d3; P=$O/d3
t3env; modelvars225
log "RESULT xd3 $TAG: start IMAGE=$IMAGE MODEL=$MODEL NST=$NST $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
mkd $P
gate $TAG $LIM; rc=$?
[ $rc = 1 ] && { log "RESULT xd3 $TAG: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$TAG.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; exit 1; }
[ $rc = 2 ] && { log "RESULT xd3 $TAG: VIRHE slow state after one restart -> stopped"; exit 1; }
modes $TAG || { log "RESULT xd3 $TAG: VIRHE the modes are not as asked -> stopped"; exit 1; }
swnote $TAG || { log "RESULT xd3 $TAG: VIRHE the switches are not as expected (start ds41_argather $GARMAX / shexp 0 / epoch 0, $((GARMAX + 1)) refused, $GARA / $GARMAX applied) -> stopped"; keeplog $TAG-switch; exit 1; }
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
stopfile || blocks $TAG-arg ds41_argather $GARA $GARMAX '{"gg_gemv": 0, "aql_acq": 0, "ds41_shexp_lane": 0}'
# ---- the c4 floor: each arm's greedy set at c4 twice (decode-heavy; before the prefill-heavy identity)
if ! stopfile; then
  fn4set "{\"ds41_argather\": $GARMAX, \"gg_gemv\": 0, \"aql_acq\": 0, \"ds41_shexp_lane\": 0, \"epoch_bump\": 1}"; nudge
  gsetc4 $TAG-B1; gsetc4 $TAG-B2
  fn4set "{\"ds41_argather\": $GARA, \"gg_gemv\": 0, \"aql_acq\": 0, \"ds41_shexp_lane\": 0, \"epoch_bump\": 1}"; nudge
  gsetc4 $TAG-A1; gsetc4 $TAG-A2
  if [ "$DRY" = 1 ]; then echo "+ batch floor: c4 B1 vs B2, A1 vs A2, B1 vs A1"
  else
    floorline "c4 greedy set, B ($GARMAX) run 1 vs run 2 (the lever's own floor)" "$(gcmp $O/$TAG-B1-gs-c4.jsonl $O/$TAG-B2-gs-c4.jsonl)"
    floorline "c4 greedy set, A ($GARA, today) run 1 vs run 2 (today's floor)" "$(gcmp $O/$TAG-A1-gs-c4.jsonl $O/$TAG-A2-gs-c4.jsonl)"
    floorline "c4 greedy set, B run 1 vs A run 1" "$(gcmp $O/$TAG-A1-gs-c4.jsonl $O/$TAG-B1-gs-c4.jsonl)"
    log "RESULT xd3 batch floor reading: D3 cannot change these (every gather form is the same bytes); IDENTICAL within B with DIFFERENT within A would contradict that -- flag it"
  fi
fi
# ---- identity (the prefill-heavy part last): B on the cold copy prompt, then A
if ! stopfile; then
  for v in "B:{\"ds41_argather\": $GARMAX, \"gg_gemv\": 0, \"aql_acq\": 0, \"ds41_shexp_lane\": 0, \"epoch_bump\": 1}" \
           "A:{\"ds41_argather\": $GARA, \"gg_gemv\": 0, \"aql_acq\": 0, \"ds41_shexp_lane\": 0, \"epoch_bump\": 1}"; do
    nm=${v%%:*}
    fn4set "${v#*:}"
    copyt $TAG-$nm; gsetc1 $TAG-$nm
    [ "$DRY" = 1 ] || log "RESULT xd3 $TAG identity $nm ran under: $(fn4get | cut -c1-240)"
  done
  if [ "$DRY" = 1 ]; then echo "+ identity: copy B / A vs bz dsd-d0 and dsm-225, B vs A; greedy set c1 B / A vs dsd-d0, B vs A"
  else
    idline "28k copy B ($GARMAX, cold prompt) vs bz dsd-d0 (main image, NST 0)" "$(cmpfile $O/copy-$TAG-B-run1.txt $RD/copy-dsd-d0-i1-run1.txt)"
    idline "28k copy A ($GARA) vs bz dsd-d0" "$(cmpfile $O/copy-$TAG-A-run1.txt $RD/copy-dsd-d0-i1-run1.txt)"
    idline "28k copy B vs item a's dsm-225" "$(cmpfile $O/copy-$TAG-B-run1.txt $RM/copy-dsm-225-i1-run1.txt)"
    idline "28k copy B vs A (in process)" "$(cmpfile $O/copy-$TAG-B-run1.txt $O/copy-$TAG-A-run1.txt)"
    idline "greedy set c1 B vs bz dsd-d0" "$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$TAG-B-gs-c1.jsonl)"
    idline "greedy set c1 A vs bz dsd-d0" "$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$TAG-A-gs-c1.jsonl)"
    idline "greedy set c1 B vs A (in process)" "$(gcmp $O/$TAG-A-gs-c1.jsonl $O/$TAG-B-gs-c1.jsonl)"
  fi
fi
unfollow
run "docker logs $CN > $O/$TAG-engine-full.log 2>&1"
[ "$DRY" = 1 ] || log "RESULT xd3 $TAG engine log: $(grep -cE '^E ' $O/$TAG-engine-full.log) E lines, $(grep -c 'fn4: switches' $O/$TAG-engine-full.log) switch applications, scratch-join warnings $(grep -c 'uses scratch on lane' $O/$TAG-engine-full.log)"
down
unset $SWITCHES
down
log "RESULT xd3 $ITEM: done -- $O"
