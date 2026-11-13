#!/bin/bash
# dsm-slotsum.sh (10.11.2026, queue-ar/dsmain, ds41-slot agent, lever D3s stage 1) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-p-slotsum.sh. The rank-independent SLOT-ORDER MoE sum (user's decision 10.11.: build it; it changes today's
# output ONCE at rounding level -- approved with a short quality check; unset = today's arithmetic byte for byte).
#   What it is: RADIANCE_DS41_SLOTSUM=48 declares, beside today's MoE path, a decode-band path (a single-pass step of at most 48
#   rows): moe_scatter_placed (the sort over an expert placement table), the EXL3 pair over the rank's home experts and its
#   extras, and moe_gather_all_reduce_slot: for every element the ranks' shared-arm products in ascending rank order, then
#   every slot's routed row times its weight in ascending slot order, rounded once -- no rank partial, no rank-order fold, so
#   the bytes do not depend on which rank holds (or which replica serves) an expert. Prefill chunks (> 48 rows) keep today's
#   path, home placement and today's sum, so prefill is today's to the byte. Switches: ds41_slotsum (1 the slot path in the
#   band, 0 today's at every step), ds41_place (the table: 0 contiguous, 1 / 2 RADIANCE_DS41_EP_PLACE / _EP_PLACE2's).
# DeepSeek-V4.1 Flash EXL3 2.25, TP4, MML 524288, NST 0, the t3 env and serve line of dsm-serve.sh (105's t3 arm + lib-29 best_vars
# + MNBT 4160 + --live), RADIANCE_DS41_ARGATHER=on (today's 4-row fused gather in arm A; XSL_GARMAX sets the D3 bound instead).
# ORDER (decode before prefill, TESTING 9.11.):
#   SELFTEST (no server, all four cards): r4d_selftest --case moe_gather_all_reduce_slot_4rank (DeepSeek's shape at every M 1..48,
#     56, 64 and generic arms; contiguous / scrambled / per-slot placement, both transports, three launches a shape: every rank's y
#     = the slot-order oracle AND the same bytes under every placement) + moe_scatter_placed_i32 (the dispatch's reference sort,
#     every slot on exactly one rank, the contiguous table = moe_scatter_i32) + regressions moe_gather_all_reduce_4rank,
#     ar_oneshot_4rank_exact, ar_twoshot_4rank_exact. A FAIL stops the item.
#   START 1 "cost" (SLOTSUM=48, contiguous only, RADIANCE_DS41_SLOTSUM_ON=0 so the start runs today's arithmetic; --expert-stats
#     /stats/xs-cost.json for stage 2): gate -> modes (+ the D3s note: 40 of 40 trunk layers, tables 0 contiguous, 0 extras) ->
#     the switch note (start ds41_slotsum 0 / ds41_place 0; ds41_slotsum 2 and ds41_place 1 refused; 1 then 0 applied) -> warm-up
#     answers vs stock (today's path at start: IDENTICAL) -> slow-state probe
#     -> COST BLOCKS ds41_slotsum 0 / 1 (ggblocks2.py, A B B A A B, mb 75 s: stepprobe c1, mb c4, mb c8; ds41_place 0 and the
#        other switches pinned) -> B vs A per load point, per-stream p50 beside the aggregate, pooled, pairs, t95
#     -> PREFILL (after decode): fresh 16k A B B A, fresh 136k A B, mb pf A B (thermwin each; prefill steps keep today's path,
#        so B = A is the expectation)
#     -> QUALITY (<= ~10 min, in process): slotkl.py -- 40 stdlib-source documents continued greedily 128 tokens under A, B and A
#        again, top-20 logprobs: KL(A || B) over the teacher-forced positions (each document up to its first divergence), top-1
#        agreement, |dlogprob|, divergence positions; A2 vs A is the floor (must be identical at c1). + the c1 greedy set under B
#        and A, the 28k copy under B and A (information: A vs bz dsd-d0 must be IDENTICAL, B vs A shows the rounding change).
#   START 2 "invariance" (SLOTSUM=48, ON=1; EP_PLACE = scrambled: per layer and rank 6 experts of the others moved onto it;
#     EP_PLACE2 = replicated: the 8 hottest experts a layer (start 1's decode counts, else 6.11. cc-x117) on every rank; host pool
#     + the extras' slots): gate -> modes (tables 0 / 1 / 2, the extras a rank) -> switch note (ds41_place 3 refused, 2 / 1 / 0)
#     -> the c1 greedy set under ds41_place 0, 1, 2 (decode first) -> the 28k copy under 0, 1, 2 -> the 136k deep answer under 0,
#        1, 2 -> BYTE-IDENTICAL 1 vs 0 and 2 vs 0 for all three (else VIRHE: the lever's acceptance) -> today's path (slotsum 0)
#        on this start: greedy set + copy vs start 1's A (information: the extras change no byte of today's path).
# References (no re-runs): bz's dsd-d0 (main image, NST 0): probe 59.38 / 62.04 tok/s, c4 122.7 (p50 31.6), c8 150.1 (p50 19.7).
# The image (XSL_IMAGE, build/slot/assemble.sh): ds41-slot = ds41-d3 c03c57d + ds41-gg 8b311fc + D3s, layered on
# radiance-local:ds41-d3-c03c57d9-5c13e555-l.
# NO cc_sim, no BetterBench (user 10.11.). No power / host / fan settings; production compose files read only.
# DRYRUN=1 bash dsm-slotsum.sh : prints every command, runs and writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-slot-tp4.yaml
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=$R/xdsm/slotsum; export STATSDIR=$O/stats
RD=$R/xdsm/draft                                # bz's dsd-d0 (main image, NST 0) files
export IMAGE=${XSL_IMAGE:-radiance-local:ds41-slot-4cf3501f-5c13e555-l}
ITEM=${XSL_ITEM:-57-p4zyzzzzz-ds-p-slotsum}; ORDER=${XSL_ORDER:-ABBAAB}; DUR=${XSL_DUR:-75}
SLROWS=${XSL_ROWS:-48}; GARMAX=${XSL_GARMAX:-}
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
GGB=$HD/ggblocks2.py; KL=$HD/slotkl.py; EPP=$HD/epplace.py; DQ=$QA/deepq.py
STATS0=$HD/cc-x117.json                         # 6.11. cc-x117 decode counts (fallback for the replicated table)
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ulimit memlock=-1 --ipc=host -e HIP_VISIBLE_DEVICES=0,1,2,3"
CASES=${XSL_CASES:-"moe_gather_all_reduce_slot_4rank moe_scatter_placed_i32 moe_gather_all_reduce_4rank ar_oneshot_4rank_exact ar_twoshot_4rank_exact"}
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DS41_ARGATHER_MAX RADIANCE_DS41_SHEXP_LANE RADIANCE_DEBUG_MISS_LOG RADIANCE_DS41_SLOTSUM
  RADIANCE_DS41_SLOTSUM_ON RADIANCE_DS41_EP_PLACE RADIANCE_DS41_EP_PLACE2 RADIANCE_DS41_EP_PLACE_TAB"
t3env() {   # 105's t3 arm + ARGATHER on (+ XSL_GARMAX) + the slot path declared (start value $1), every other switch unset
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0 \
         RADIANCE_DS41_SLOTSUM=$SLROWS RADIANCE_DS41_SLOTSUM_ON=$1
  [ -n "$GARMAX" ] && export RADIANCE_DS41_ARGATHER_MAX=$GARMAX
  return 0
}
GARB=${GARMAX:-4}                                # ds41_argather's declared bound (pinned in every block)
modelvars225() {   # -> MODEL, HP29, LIM, WARMREF, C8REF; lib-29 best_vars (NST 0) + MNBT 4160 + --live (+ $1: more XARGS)
  export MODEL=$M225; HP29=${XSL_HP:-19456}; LIM=23; WARMREF=$WARM_REF_225; C8REF=43.1
  best_vars; export MNBT=4160 XARGS="$XARGS --live ${1:-}"
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
selftest() {   # the byte checks, each case its own bounded container; 0 = every case clean
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
    echo "$r" >> $O/selftest/SUMMARY.txt; log "RESULT xsl selftest $r"
    [ $c = moe_gather_all_reduce_slot_4rank ] && log "RESULT xsl selftest $c transports: $(grep -m1 'shapes took the two-shot' $O/selftest/$c.txt | sed 's/^ *//')"
  done
  [ "$DRY" = 1 ] && return 0
  [ $bad = 0 ] && { echo "D3S SELFTEST CLEAN" >> $O/selftest/SUMMARY.txt; return 0; }
  echo "D3S SELFTEST NOT CLEAN ($bad case(s))" >> $O/selftest/SUMMARY.txt; return 1
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xsl $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xsl $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  case $r in IDENTICAL*) log "RESULT xsl $1 warm-up answers vs stock ($(basename $2)): $r" ;;
    *) log "RESULT xsl $1 warm-up answers vs stock ($(basename $2)): VIRHE $r -- the start runs today's path (ds41_slotsum 0), which must be today's bytes" ;; esac
}
modes() {   # modes TAG WANT_TABLES WANT_EXTRAS: the t3 notes + the D3s note -> 0, else 1 (stopped)
  [ "$DRY" = 1 ] && { echo "+ modes $1: indexer note (mxfp4, latent bf16, compressor tail bound, router selection+sort separate, AR_PCN separate at 0 layers), staging line, no acquire-rule / shared-expert-lane line, the D3s note (up to $SLROWS rows, 40 of 40 trunk layers, tables '$2', extras '$3')"; return 0; }
  local MLF SL SN ok=1
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  SN=$(docker logs $CN 2>&1 | grep -m1 'the slot-order MoE sum (lever D3s)' | sed 's/^radiance: deepseek41: //' | cut -c1-900)
  log "RESULT xsl $1 modes: $(echo "$MLF" | cut -c1-700) | staging: ${SL:-none} | D3s: ${SN:-none}"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate" "post-collapse-norm separate at 0 layers" "next collapse-norm separate at 0 layers"; do
    case "$MLF" in *"$w"*) ;; *) log "RESULT xsl $1: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  case "$SN" in *"up to $SLROWS rows: 40 of 40 trunk layers"*"tables: $2"*"extra experts a rank $3"*) ;;
    *) log "RESULT xsl $1: VIRHE the D3s note is not 'up to $SLROWS rows: 40 of 40 trunk layers ... tables: $2 ... extra experts a rank $3': '${SN}'"; ok=0 ;; esac
  [ "$(docker logs $CN 2>&1 | grep -c 'did not resolve; the layer keeps')" = 0 ] || { log "RESULT xsl $1: VIRHE a layer's slot path did not resolve: $(docker logs $CN 2>&1 | grep -m1 'did not resolve' | cut -c1-300)"; ok=0; }
  [ -n "$SL" ] || { log "RESULT xsl $1: VIRHE no prefill staging line (levels refused?)"; ok=0; }
  docker logs $CN 2>&1 | grep -q 'the acquire rule plays' && { log "RESULT xsl $1: VIRHE an acquire-rule line (RADIANCE_AQL_ACQ should be unset)"; ok=0; }
  docker logs $CN 2>&1 | grep -q 'the shared expert on the second lane' && { log "RESULT xsl $1: VIRHE a shared-expert-lane line (RADIANCE_DS41_SHEXP_LANE should be unset)"; ok=0; }
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
FIX="\"gg_gemv\": 0, \"aql_acq\": 0, \"ds41_shexp_lane\": 0, \"ds41_argather\": $GARB"
arm() {   # arm SLOTSUM PLACE: POST both (+ the pinned switches and an epoch bump) and a 4-token request so they apply
  fn4set "{\"ds41_slotsum\": $1, \"ds41_place\": $2, $FIX, \"epoch_bump\": 1}"; nudge
}
swnote() {   # swnote TAG NTAB START_SLOT START_PLACE: the switches at start, past-the-end values refused, changes applied at a step
  local tag=$1 nt=$2 s0v=$3 p0v=$4
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (expect ds41_slotsum $s0v, ds41_place $p0v, epoch 0); POST ds41_slotsum 2 -> 400, ds41_place $nt -> 400; POST ds41_slotsum 1 / place $((nt - 1)), then 0 / 0 -> applied"; return 0; }
  local s0 s1 s2 c1 c2
  s0=$(fn4get)
  c1=$(fn4code '{"ds41_slotsum": 2}'); c2=$(fn4code "{\"ds41_place\": $nt}")
  fn4set "{\"ds41_slotsum\": 1, \"ds41_place\": $((nt - 1))}"; nudge; s1=$(fn4get)
  fn4set '{"ds41_slotsum": 0, "ds41_place": 0}'; nudge; s2=$(fn4get)
  log "RESULT xsl $tag switch note: at start $s0 | ds41_slotsum 2 -> HTTP $c1, ds41_place $nt -> HTTP $c2 | after 1/$((nt - 1)): $s1 | after 0/0: $s2 | core log: $(docker logs $CN 2>&1 | grep -E 'fn4: switches' | tail -2 | sed 's/^[A-Z] *//' | tr '\n' '|' | cut -c1-400)"
  python3 -c 'import json,sys
s0, s1, s2 = (json.loads(x) for x in sys.argv[1:4]); nt, sv, pv = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
w = lambda s, k: s["switches"].get(k)
ok = w(s0, "ds41_slotsum") == sv and w(s0, "ds41_place") == pv and s0["epoch"] == 0 and w(s1, "ds41_slotsum") == 1 \
     and w(s1, "ds41_place") == nt - 1 and w(s2, "ds41_slotsum") == 0 and w(s2, "ds41_place") == 0 and not s1["pending"] \
     and not s2["pending"] and s2["epoch"] == 2
sys.exit(0 if ok else 1)' "$s0" "$s1" "$s2" $nt $s0v $p0v 2>/dev/null && [ "$c1" = 400 ] && [ "$c2" = 400 ]
}
mbms() { grep '^MB ' $1 2>/dev/null | head -1 | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()[3:]).get("ms_decode_step", ""))' 2>/dev/null; }
stprobe() {   # stprobe TAG N: mb c8 45 s -> "fast|between|slow MS (ref REF)"
  [ "$DRY" = 1 ] && { echo "+ (cd $D && timeout 400 python3 mb.py --tag xsl-$1-st$2 --phases c8 --dur ${XSL_STATE_DUR:-45} --warm 10 > $O/$1-stprobe$2.txt)" >&2; echo "fast (dry)"; return 0; }
  run "(cd $D && timeout 400 python3 mb.py --tag xsl-$1-st$2 --phases c8 --dur ${XSL_STATE_DUR:-45} --warm 10 > $O/$1-stprobe$2.txt 2>&1)"
  local ms; ms=$(mbms $O/$1-stprobe$2.txt)
  [ -n "$ms" ] || { echo "unknown (no mb line)"; return 0; }
  awk -v m=$ms -v r=$C8REF 'BEGIN{s=(m <= 1.07*r) ? "fast" : (m >= 1.14*r) ? "slow" : "between"; printf "%s %s ms (105 tuned3l c8 %s ms, x%.3f)\n", s, m, r, m/r}'
}
copyt() {   # copyt TAG: the 28k copy after a ~31k-token prompt (105's prompt, offset 0 -- bz d0's i1)
  local t0; t0=$(date +%s)
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset 0 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xsl $1 copy: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300)$(type thermwin > /dev/null 2>&1 && echo " | $(thermwin $t0 $(date +%s) 2>/dev/null | cut -c1-200)")"
}
deepq() {   # deepq TAG: the fixed ~135k prompt and a 512-token greedy answer (105's deepq.py)
  local t0=$(date +%s)
  run "(cd $D && timeout 1800 python3 $DQ $O/deep-$1.txt 135000 512 > $O/deep-$1.out 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xsl $1 deep answer: $(grep -m1 '^DEEPQ ' $O/deep-$1.out | cut -c7- | cut -c1-200)$(type thermwin > /dev/null 2>&1 && echo " | $(thermwin $t0 $(date +%s) 2>/dev/null | cut -c1-200)")"
}
pfd() {   # pfd TAG DEPTH: one fresh prefill of DEPTH tokens (nonce first, cache_prompt off), thermal line
  local t0=$(date +%s)
  run "(cd $D && timeout 1800 python3 ds41_tests.py prefill --depths $2 --cpt 3.95 --calibrate 0 --reps 1 > $O/$1-pf$2.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xsl $1 fresh prefill $2: $(grep '^J ' $O/$1-pf$2.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); print("%s tok, cached %s, TTFT %s s, %s tok/s, %s" % (d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("prefill_tps"), d.get("status")))' 2>&1 | cut -c1-200)$(type thermwin > /dev/null 2>&1 && echo " | $(thermwin $t0 $(date +%s) 2>/dev/null | cut -c1-200)")"
}
mbpf() {   # mbpf TAG: mb pf (lib-pr266's mbp) + a thermal line
  local t0=$(date +%s)
  run "(cd $D && timeout 1500 python3 mb.py --tag xsl-$1 --phases pf > $O/$1-pf.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xsl $1 mb pf: $(grep '^MB ' $O/$1-pf.txt 2>/dev/null | cut -c4- | cut -c1-500)$(type thermwin > /dev/null 2>&1 && echo " | $(thermwin $t0 $(date +%s) 2>/dev/null | cut -c1-200)")"
}
gsetc1() { run "(cd $D && timeout 900 python3 $GS $URL $O/$1-gs-c1.jsonl --conc 1 --max-tokens 192 > $O/$1-gs-c1.txt 2>&1)"; }
gcmp() { [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }; [ -s "$1" ] && [ -s "$2" ] || { echo "not compared (missing $1 or $2)"; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
must() {   # must LABEL RESULT: the lever's acceptance -- IDENTICAL, else VIRHE
  case "$2" in IDENTICAL*) log "RESULT xsl invariance $1: $2" ;;
    *) log "RESULT xsl invariance $1: VIRHE $2 -- the slot-order sum must not depend on the placement"; INV_BAD=$((INV_BAD + 1)) ;; esac
}
info() { log "RESULT xsl identity $1: $2"; }
blocks() {   # blocks TAG SWITCH A B FIXED: the A/B blocks over SWITCH (FIXED pinned in every block) and their summary
  local tag=$1 sw=$2 a=$3 b=$4 fx=$5
  run "timeout 7200 python3 -I $GGB --url $URL --mb-dir $D --label $tag --switch $sw --a $a --b $b --fixed '$fx' --order $ORDER --dur $DUR --warm 10 --out $O/$tag-blocks.jsonl > $O/$tag-blocks.txt 2>&1"
  [ "$DRY" = 1 ] && return 0
  while IFS= read -r l; do log "RESULT xsl $tag ${l:0:400}"; done < <(grep '^BLOCK ' $O/$tag-blocks.txt)
  log "RESULT xsl $tag summary (B = $sw $b vs A = $a, $fx pinned, in process, order $ORDER, mb $DUR s): $(grep '^GGB ' $O/$tag-blocks.txt | cut -c5- | python3 -c 'import json,sys
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
log "RESULT xsl $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) SLOTSUM=$SLROWS ARGATHER on${GARMAX:+ MAX $GARMAX}, order $ORDER mb $DUR s yaml $Y | $(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -m1 "^radiance" /opt/radiance/share/radiance/slot-PROVENANCE.txt' 2>/dev/null | cut -c1-200) | refs bz dsd-d0 (main image, NST 0): probe 59.38 / 62.04 tok/s, c4 122.7 (p50 31.6), c8 150.1 (p50 19.7)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xsl: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xsl: VIRHE image $IMAGE not on gpu02"; exit 1; }
  [ -e $Y ] && grep -q '^ *RADIANCE_DS41_SLOTSUM:' $Y && grep -q '^ *RADIANCE_DS41_EP_PLACE2:' $Y || { log "RESULT xsl: VIRHE $Y missing or does not pass RADIANCE_DS41_SLOTSUM / _EP_PLACE2"; exit 1; }
  for f in $GS $GGB $KL $EPP $DQ $D/stepprobe.py $D/mb.py; do [ -e $f ] || { log "RESULT xsl: VIRHE $f missing"; exit 1; }; done
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xsl: VIRHE $M225 missing or not verified"; exit 1; }
  [ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -q RADIANCE_DS41_SLOTSUM /opt/radiance/share/radiance/architectures/deepseek41.so && grep -q ds41_slotsum /opt/radiance/bin/radiance && grep -q moe_gather_all_reduce_slot /opt/radiance/share/radiance/kernels/libr4d.so && echo yes' 2>/dev/null)" = yes ] \
    || { log "RESULT xsl: VIRHE $IMAGE is not the D3s image (no RADIANCE_DS41_SLOTSUM in deepseek41.so, ds41_slotsum in radiance or moe_gather_all_reduce_slot in libr4d)"; exit 1; }
  for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
  [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT xsl: VIRHE a KFD client is running (after 60 s wait) -- not the selftest's cards"; exit 1; }
fi
mkd $O $O/guard $STATSDIR $STATSDIR/place
# ---- the byte checks first, with no server: a FAIL stops the item
if ! stopfile; then
  selftest || { log "RESULT xsl $ITEM: VIRHE the selftest is not clean ($O/selftest/SUMMARY.txt) -> no serving phase"; exit 1; }
  [ "$DRY" = 1 ] || log "RESULT xsl selftest: $(tail -1 $O/selftest/SUMMARY.txt)"
fi
evict
# ================================================================ START 1: cost + quality (contiguous only)
TAG=cost; P=$O/cost
t3env 0; modelvars225 "--expert-stats /stats/xs-cost.json"
log "RESULT xsl $TAG: start IMAGE=$IMAGE MODEL=$MODEL NST=$NST $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
mkd $P
gate $TAG $LIM; rc=$?
[ $rc = 1 ] && { log "RESULT xsl $TAG: VIRHE not healthy ($(desc)); $(grep -E '^E |deepseek41: ' $O/start-$TAG.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; exit 1; }
[ $rc = 2 ] && { log "RESULT xsl $TAG: VIRHE slow state after one restart -> stopped"; exit 1; }
modes $TAG "0 contiguous;" "0..0" || { log "RESULT xsl $TAG: VIRHE the modes are not as asked -> stopped"; exit 1; }
swnote $TAG 1 0 0 || { log "RESULT xsl $TAG: VIRHE the switches are not as expected -> stopped"; keeplog $TAG-switch; exit 1; }
identw $TAG $WARMREF
st=$(stprobe $TAG 1)
if [ "${st%% *}" = slow ]; then
  log "RESULT xsl $TAG state probe 1: $st -> one restart"
  gate $TAG $LIM || { log "RESULT xsl $TAG: VIRHE no healthy / fast start after the slow-state restart"; exit 1; }
  modes $TAG "0 contiguous;" "0..0" || exit 1
  swnote $TAG 1 0 0 || { log "RESULT xsl $TAG: VIRHE the switches are not as expected after the restart -> stopped"; exit 1; }
  st=$(stprobe $TAG 2)
fi
log "RESULT xsl $TAG state: $st (slow-state probe mb c8 ${XSL_STATE_DUR:-45} s)$(case "${st%% *}" in fast) ;; *) echo " -- NOT FAST: flag to the coordinator" ;; esac)"
follow $P
# ---- the cost: today's sum (A) vs the slot-order sum (B), contiguous placement, in process; decode first
stopfile || blocks $TAG-sum ds41_slotsum 0 1 "{\"ds41_place\": 0, $FIX}"
# ---- prefill after decode: prefill steps keep today's path in both arms (B = A expected)
if ! stopfile; then
  NPF=0; for v in A:0 B:1 B:1 A:0; do NPF=$((NPF + 1)); arm ${v#*:} 0; pfd $TAG-${v%%:*}$NPF 16000; done
  for v in A:0 B:1; do arm ${v#*:} 0; pfd $TAG-${v%%:*} 136000; done
  for v in A:0 B:1; do arm ${v#*:} 0; mbpf $TAG-${v%%:*}; done
fi
# ---- the quality check (in process, <= ~10 min) and the information identities
if ! stopfile; then
  run "timeout 1500 python3 -I $KL --url $URL --a '{\"ds41_slotsum\": 0, \"ds41_place\": 0}' --b '{\"ds41_slotsum\": 1, \"ds41_place\": 0}' --out $O/kl --docs ${XSL_KL_DOCS:-40} --gen ${XSL_KL_GEN:-128} --aa > $O/kl.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT xsl $TAG quality (slotkl.py, teacher-forced KL A=today vs B=slot-order sum, top-20, c1): $(grep -m1 '^SLOTKL ' $O/kl.txt | cut -c8- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
if "error" in d: print("VIRHE " + d["error"]); sys.exit(0)
def s(k):
    v = d.get(k) or {}
    g = lambda x: "-" if v.get(x) is None else "%.3g" % v[x]
    fd = v.get("first_divergence") or []
    return "%s: %s positions, KL mean %s p50 %s p99 %s p99.9 %s max %s nats, top-1 %s, |dlogp| mean %s max %s, %s of %s docs identical, %s diverged (first at %s)" % (
        k, v.get("positions"), g("kl_mean"), g("kl_p50"), g("kl_p99"), g("kl_p999"), g("kl_max"), g("top1_agree"), g("dlogprob_mean"),
        g("dlogprob_max"), v.get("docs_identical"), v.get("docs"), v.get("docs_diverged"), ",".join(map(str, fd[:12])))
print(s("b_vs_a") + " | " + s("a2_vs_a") + " | %s s" % d.get("wall_s"))' 2>&1 | cut -c1-1500)"
  arm 1 0; gsetc1 $TAG-B; copyt $TAG-B
  arm 0 0; gsetc1 $TAG-A; copyt $TAG-A
  if [ "$DRY" != 1 ]; then
    info "c1 greedy set A (today, this image) vs bz dsd-d0 (must be IDENTICAL: today's path)" "$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$TAG-A-gs-c1.jsonl)"
    info "28k copy A vs bz dsd-d0 (must be IDENTICAL)" "$(cmpfile $O/copy-$TAG-A-run1.txt $RD/copy-dsd-d0-i1-run1.txt)"
    info "c1 greedy set B (slot-order sum) vs A (the approved rounding change)" "$(gcmp $O/$TAG-A-gs-c1.jsonl $O/$TAG-B-gs-c1.jsonl)"
    info "28k copy B vs A" "$(cmpfile $O/copy-$TAG-B-run1.txt $O/copy-$TAG-A-run1.txt)"
  fi
fi
unfollow
run "docker logs $CN > $O/$TAG-engine-full.log 2>&1"
[ "$DRY" = 1 ] || log "RESULT xsl $TAG engine log: $(grep -cE '^E ' $O/$TAG-engine-full.log) E lines, $(grep -c 'fn4: switches' $O/$TAG-engine-full.log) switch applications; expert stats $(ls -la $STATSDIR/xs-cost.json 2>/dev/null | awk '{print $5}') bytes"
down
# ================================================================ START 2: invariance across placements
if ! stopfile; then
  TAG=place; P=$O/place
  ST=$STATSDIR/xs-cost.json; [ -s $ST ] || ST=$STATS0
  run "python3 -I $EPP scramble --experts 384 --layers 0-39 --per-rank 6 --seed 11 -o $STATSDIR/place/scr6.place > $O/place-gen.txt 2>&1"
  run "python3 -I $EPP replicate --stats $ST --hot 8 -o $STATSDIR/place/rep8.place >> $O/place-gen.txt 2>&1"
  run "python3 -I $EPP extras --place $STATSDIR/place/scr6.place --place $STATSDIR/place/rep8.place >> $O/place-gen.txt 2>&1"
  if [ "$DRY" = 1 ]; then XMAX=700
  else XMAX=$(grep -m1 '^EPEXTRAS ' $O/place-gen.txt | cut -c10- | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["extras_max"])' 2>/dev/null || echo 0); fi
  # every extra a slot of the pinned host pool (or VRAM): 9.06 MB an expert at EXL3 2.25 -> 8.7 MiB, + 1 GiB margin
  HPX=$(( 19456 + (XMAX * 87 + 9) / 10 + 1024 ))
  log "RESULT xsl $TAG placements: $(grep '^EP' $O/place-gen.txt 2>/dev/null | cut -c1-300 | tr '\n' '|') -> host pool $HPX MiB (19456 + the extras)"
  t3env 1; XSL_HP=$HPX modelvars225
  export RADIANCE_DS41_EP_PLACE=/stats/place/scr6.place RADIANCE_DS41_EP_PLACE2=/stats/place/rep8.place RADIANCE_DS41_EP_PLACE_TAB=0
  log "RESULT xsl $TAG: start IMAGE=$IMAGE MODEL=$MODEL NST=$NST $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
  mkd $P
  INV_BAD=0
  gate $TAG $LIM; rc=$?
  [ $rc = 1 ] && { log "RESULT xsl $TAG: VIRHE not healthy ($(desc)); $(grep -E '^E |deepseek41: ' $O/start-$TAG.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; exit 1; }
  [ $rc = 2 ] && log "RESULT xsl $TAG: slow state after one restart -- the invariance checks are bytes, not speed: carrying on"
  modes $TAG "0 contiguous, 1 /stats/place/scr6.place (40 layers placed, 960 experts off home, 0 replicated), 2 /stats/place/rep8.place (40 layers placed, 0 experts off home, 320 replicated);" "" \
    || { log "RESULT xsl $TAG: VIRHE the modes are not as asked -> stopped"; exit 1; }
  swnote $TAG 3 1 0 || { log "RESULT xsl $TAG: VIRHE the switches are not as expected -> stopped"; keeplog $TAG-switch; exit 1; }
  follow $P
  for p in 0 1 2; do arm 1 $p; gsetc1 $TAG-p$p; done
  for p in 0 1 2; do arm 1 $p; copyt $TAG-p$p; done
  for p in 0 1 2; do arm 1 $p; deepq $TAG-p$p; done
  if [ "$DRY" = 1 ]; then echo "+ invariance: greedy set, 28k copy, 136k deep answer, ds41_place 1 vs 0 and 2 vs 0 (must be IDENTICAL); start 2 place 0 vs start 1 B"
  else
    must "c1 greedy set, scrambled (ds41_place 1) vs contiguous (0)" "$(gcmp $O/$TAG-p0-gs-c1.jsonl $O/$TAG-p1-gs-c1.jsonl)"
    must "c1 greedy set, replicated (2) vs contiguous (0)" "$(gcmp $O/$TAG-p0-gs-c1.jsonl $O/$TAG-p2-gs-c1.jsonl)"
    must "28k copy, scrambled vs contiguous" "$(cmpfile $O/copy-$TAG-p1-run1.txt $O/copy-$TAG-p0-run1.txt)"
    must "28k copy, replicated vs contiguous" "$(cmpfile $O/copy-$TAG-p2-run1.txt $O/copy-$TAG-p0-run1.txt)"
    must "136k deep answer, scrambled vs contiguous" "$(cmpfile $O/deep-$TAG-p1.txt $O/deep-$TAG-p0.txt)"
    must "136k deep answer, replicated vs contiguous" "$(cmpfile $O/deep-$TAG-p2.txt $O/deep-$TAG-p0.txt)"
    info "c1 greedy set, start 2 contiguous vs start 1 B (another start, extras declared: information)" "$(gcmp $O/cost-B-gs-c1.jsonl $O/$TAG-p0-gs-c1.jsonl)"
    info "28k copy, start 2 contiguous vs start 1 B" "$(cmpfile $O/copy-cost-B-run1.txt $O/copy-$TAG-p0-run1.txt)"
    log "RESULT xsl invariance summary: $([ $INV_BAD = 0 ] && echo 'BYTE-IDENTICAL across contiguous / scrambled / replicated placement (greedy set, 28k copy, 136k deep answer)' || echo "VIRHE $INV_BAD of 6 comparisons differ")"
  fi
  # today's path on this start (the extras declared, unused): its bytes vs start 1's A (information)
  arm 0 0; gsetc1 $TAG-today; copyt $TAG-today
  if [ "$DRY" != 1 ]; then
    info "today's path (ds41_slotsum 0) with the extras declared vs start 1 A: c1 greedy set" "$(gcmp $O/cost-A-gs-c1.jsonl $O/$TAG-today-gs-c1.jsonl)"
    info "today's path with the extras declared vs start 1 A: 28k copy" "$(cmpfile $O/copy-cost-A-run1.txt $O/copy-$TAG-today-run1.txt)"
  fi
  unfollow
  run "docker logs $CN > $O/$TAG-engine-full.log 2>&1"
  [ "$DRY" = 1 ] || log "RESULT xsl $TAG engine log: $(grep -cE '^E ' $O/$TAG-engine-full.log) E lines, $(grep -c 'fn4: switches' $O/$TAG-engine-full.log) switch applications; $(grep -m1 -E 'heat engine: [0-9]+ movable units' $O/$TAG-engine-full.log | cut -c1-120)"
  down
fi
unset $SWITCHES
down
log "RESULT xsl $ITEM: done -- $O"
