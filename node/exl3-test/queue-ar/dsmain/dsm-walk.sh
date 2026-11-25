#!/bin/bash
# dsm-walk.sh (10.11.2026, queue-ar/dsmain, ds-main-port W1 agent) -- NOT a queue item; the body of 57-p4zyzzzzz-ds-*-walk.sh.
# LEVER W1: THE DSPARK MARKOV WALK OVER THE WHOLE VOCABULARY (branch ds41-walk = ds41-int c05de27 + W1; image XWK_IMAGE =
# radiance-local:ds41-walk-<sha8>-5c13e555-l, build/walk/assemble.sh). Today the walk re-ranks the head's top 32 (row_topk +
# all_gather + merge + dflash_select); the reference / exllamav3 / vLLM walk every id. W1: dspark_walk_step, a step on each rank's
# vocabulary shard (fma chain over the Markov rank, u64 key = ordered score bits | ~id, one atomicMax a block) + one all_gather of
# S u64 keys a step; RADIANCE_DS41_DSPARK_WALK=0 at start declares it (switch dspark_walk starts 0), the fn4 switch flips it in
# process. Output-exact by construction (proposals only; the trunk verifies) -- checked anyway below.
# ONE NST 5 START with the hx-stack env (RADIANCE_DS41_GG_GEMV=2, RADIANCE_AQL_ACQ=rule-w, RADIANCE_DS41_KVFOLD / _OFOLD / _ONEFILL=1,
#   RADIANCE_PLACE_TURNOVER=0 + RADIANCE_DS41_DSPARK_WALK=0; --draft-window adaptive; in process every block: gg_gemv 2, aql_acq 2,
#   the K1 three 1, draft_adaptive 1, draft_depth0 1, draft_hybrid 1, draft_hybrid_match 8):
#   0. GPU SELFTEST before the server (the image's r4d_selftest --case dspark_walk_step_f32): the kernel over 1-4 shards vs a host
#      walk token for token (ties forced, the real vocabulary 129280 x rank 256 at 3 / 4 ranks) + one step's launch time at the TP4
#      shard (32320 columns) at 1 / 4 / 8 sequences (PERF lines) + ar_gather_4rank on the four cards at the walk's 16-byte key slot
#      geometry (S 1-8) and the 8-byte row of the failed run refused; a gather that fails here stops the item before the server.
#   FIX 10.11. (first run, runs/xdsm/walk): the walk gathered an 8-byte key a row and ar_gather_4rank refused it at the first B
#      query pass (step 4251, "row=2 f32: shape this kernel does not serve"); ds41-walk 0281f7c moves the keys to 16-byte slots.
#      This run writes runs/xdsm/walk2 (XWK_OUT) and smoke-tests the walk under B (c1 + a 4-request burst) before the blocks.
#   1. gate (70 ms) -> modes (= hx-stack's: t3 notes, ARGATHER 4, K1 declared and on, rule-w, turnover 0) -> the W1 declared line
#      -> switch note (GET: dspark_walk 0; dspark_walk 2 -> 400; A {dspark_walk 0} and B {dspark_walk 1} applied at a step)
#      -> warm-up answers vs stock.
#   2. IN-PROCESS DECODE BLOCKS A B B A (XWK_ORDER; wkblocks.py, ggblocksdz2's helpers read only): c1 essay / code (stepprobe 512),
#      edit (editprobe), mb c4 / c8 (XWK_DUR 75 s, warm 10), the draft-position counters (/stats draft_positions) around every
#      phase, and a diagnostic phase at the FIXED window of 5 (essay + code; draft_adaptive 0 / draft_depth0 0 for the phase,
#      positions only -- the walk's conditional acceptance at positions 2-5, as the hdz-candsweep k8 / k16 / k32 arms).
#   3. IDENTITY under B (walk 1) then A (walk 0), each after a prefix-cache reset: the 28k copy and the 20-prompt greedy set at
#      c1 vs bz's d0 (runs/xdsm/draft, NST 0 stock) byte for byte.
#   4. ORACLE (second start, ~6 min): NST 5 fixed window (the chain shape, 6-row verifies) + RADIANCE_DS41_DSPARK_WALK=1 (the walk
#      runs every step, its proposals then replaced by the oracle's ids, 30 % corrupted) + --debug-draft-oracle + --no-prefix-cache:
#      oracle.py check twice vs bz d0's ids (IDENTICAL, draft_n > accepted > 0).
#   SUMMARY: per metric A (walk 0, the in-process reference) and B (walk 1): mean, B/A pairs (t95), % difference; tokens a step,
#   ms a round (ms a decode step) and tok/s; per-position acceptance (accepted / reached) of both arms; the distance to the targets
#   (1 user >= 90 tok/s, c4 >= 160, c8 >= 200); identity, oracle, selftest. Last lines: "RESULT xwk summary: ..." and
#   "RESULT xwk <item>: done".
# Estimate (LEVERS W1, not a measurement): prose tokens a step +3..+10 %, cost +0.5-0.7 ms a round (the agent's own estimate for
# this design: ~0.2-0.35 ms a round at c1, nearly flat in S). ~45 min (evict 1, selftest 2, start 5, blocks 4 x ~5, identity 5,
# oracle start 6, summary). NO cc_sim, no 2.9. No power / host / fan settings; production compose files read only.
# XWK_IMAGE, XWK_ITEM, XWK_ORDER (A/B blocks, default ABBA), XWK_DUR (75), XWK_ORACLE (1 | 0), XWK_SELFTEST (1 | 0), XWK_FIXED5 (1 | 0).
# DRYRUN=1 bash dsm-walk.sh : prints every command; checks the image is here (docker image inspect + a no-device, no-network grep
#   of its files) and the files / references exist; no server, no GPU, nothing written.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-walk-tp4.yaml
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=$R/xdsm/${XWK_OUT:-walk}; export STATSDIR=$O/stats   # XWK_OUT: walk2 after the 10.11. gather fix (walk = the failed run)
RD=$R/xdsm/draft                                # bz's d0 files (identity reference: copy + greedy set; oracle ids)
export IMAGE=${XWK_IMAGE:-radiance-local:ds41-walk-WALKSHA-5c13e555-l}
ITEM=${XWK_ITEM:-57-p4zyzzzzz-ds-walk}; ORDER=${XWK_ORDER:-ABBA}; DUR=${XWK_DUR:-75}; MATCH=8
DO_ORACLE=${XWK_ORACLE:-1}; DO_SELFTEST=${XWK_SELFTEST:-1}; FIXED5=${XWK_FIXED5:-1}
case $ORDER in *[!AB]*|"") echo "XWK_ORDER takes A / B blocks (got '$ORDER')"; exit 1 ;; esac
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
EP=$HD/editprobe.py; WKB=$HD/wkblocks.py; GGB=$HD/ggblocksdz2.py; ORC=$HD/oracle.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
FIX='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid": 1, "draft_hybrid_match": '"$MATCH"'}'
ARM_A='{"dspark_walk": 0}'
ARM_B='{"dspark_walk": 1}'
armset() {   # armset ARMJSON [EXTRAJSON]: FIX + the arm's switches (+ extra) as one POST body
  local x=${2:-}; [ -n "$x" ] || x='{}'
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update(json.loads(sys.argv[2])); d.update(json.loads(sys.argv[3])); print(json.dumps(d))' "$FIX" "$1" "$x"
}
DESC_A="A: the hx-stack NST 5 set (gg_gemv 2, rule-w, K1 c/d/f, adaptive, depth 0, hybrid 1 match 8, turnover 0), dspark_walk 0 (the head's top 32 re-ranked)"
DESC_B="B: the same, dspark_walk 1 (the whole vocabulary)"
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DS41_ARGATHER_MAX RADIANCE_DS41_SHEXP_LANE RADIANCE_DEBUG_MISS_LOG
  RADIANCE_DS41_KVFOLD RADIANCE_DS41_OFOLD RADIANCE_DS41_ONEFILL RADIANCE_DS41_PIPE
  RADIANCE_DRAFT_DEPTH0 RADIANCE_DRAFT_PROBE RADIANCE_DRAFT_HYBRID RADIANCE_DRAFT_HYBRID_MATCH RADIANCE_DS41_DSPARK_WALK"
VRAMENV=RADIANCE_PLACE_TURNOVER=0
walkenv() {   # walkenv 0|1: hx-stack's env (105's t3 arm + gg_gemv 2 + rule-w + the K1 folds + turnover 0) + RADIANCE_DS41_DSPARK_WALK
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  export $VRAMENV RADIANCE_DS41_DSPARK_WALK=$1
}
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ulimit memlock=-1 --ipc=host -e HIP_VISIBLE_DEVICES=0,1,2,3"
gpurun() {   # gpurun NAME SECONDS docker-run-args...: bounded, named, removed even if its kernel hangs (= the selftest items')
  local name=$1 t=$2; shift 2
  timeout -k 15 $t docker run --rm --init --stop-timeout 10 --name $name "$@"; local rc=$?
  docker rm -f $name > /dev/null 2>&1 || true
  { [ $rc = 124 ] || [ $rc = 137 ]; } && echo "gpurun: $name stopped after $t s (rc $rc)" >&2
  return $rc
}
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
wgate() {   # wgate TAG LIMIT: lib-29's restart + check, one retry over the limit, keeplog
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
  log "RESULT xwk guard $tag: $st, 1-user ${ms} ms/step (limit $lim; NST 5 DSpark starts depth0b / hybrid 35.2-35.3)"
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xwk $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xwk $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  echo "warm-up-$1 $r" >> $O/identity.txt
  log "RESULT xwk $1 warm-up answers vs stock ($(basename $2)): $r"
}
vrammodes() {   # the VRAM env in the container, the mover's kept-free counts (turnover 0: 0 on every rank) -> VKNOBS
  local MV CE kv bad="" nz
  MV=$(docker logs $CN 2>&1 | grep -E '^I\[[0-9]\] mover: [0-9]+ slab slots' | sed -E 's/^I\[([0-9])\] mover: ([0-9]+) slab slots \(([0-9]+) shadow, ([0-9]+) more kept free.*/r\1 \3 shadow \4 kept free (\2 slots)/' | tr '\n' ',' | sed 's/,$//')
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in $VRAMENV RADIANCE_DS41_DSPARK_WALK=$1; do echo "$CE" | grep -qxF "$kv" || bad="$bad ${kv%%=*}"; done
  VKNOBS="env [$VRAMENV RADIANCE_DS41_DSPARK_WALK=$1]$([ -n "$bad" ] && echo " NOT in the container:$bad"); mover ${MV:-no line}"
  [ -z "$bad" ] || { log "RESULT xwk: VIRHE the env did not reach the container:$bad"; return 1; }
  nz=$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots \([0-9]+ shadow, 0 more kept free')
  [ "$nz" = 4 ] || { log "RESULT xwk: VIRHE turnover 0 asked but $nz of 4 ranks keep 0 slots free: $MV"; return 1; }
  return 0
}
modes() {   # modes TAG WALKENV: hx-stack's checks (t3 notes, ARGATHER 4, no shexp lane, staging, K1 declared + on, rule-w) + the W1 line
  [ "$DRY" = 1 ] && { echo "+ modes $1: hx-stack's notes (indexer mxfp4 / latent bf16 / tail bound / router separate / AR_PCN 0 layers, ARGATHER 4, no shexp lane, staging, K1 c/d/f declared and 1, rule-w in every recorded pass), turnover 0 on 4 ranks, RADIANCE_DS41_DSPARK_WALK=$2 in the container, the W1 declared line (dspark_walk=$2)"; return 0; }
  local MLF SL AL XL KL WL NW NR ok=1
  vrammodes $2 || ok=0
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  AL=$(docker logs $CN 2>&1 | grep -m1 'the MoE gather and its all-reduce as one kernel' | sed 's/^radiance: //' | cut -c1-300)
  XL=$(docker logs $CN 2>&1 | grep -m1 'the shared expert on the second lane' | sed 's/^radiance: //' | cut -c1-300)
  KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-500)
  WL=$(docker logs $CN 2>&1 | grep -m1 'DSpark lever W1.*fn4 dspark_walk=' | sed 's/^radiance: //' | cut -c1-400)
  log "RESULT xwk $1 modes: $(echo "$MLF" | cut -c1-600) | staging: ${SL:-none} | argather: ${AL:-none} | K1: ${KL:-none} | W1: ${WL:-none} | VRAM: ${VKNOBS:-none}"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate" "post-collapse-norm separate at 0 layers" "next collapse-norm separate at 0 layers"; do
    case "$MLF" in *"$w"*) ;; *) log "RESULT xwk $1: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  case "$AL" in *"up to 4 tokens a step (RADIANCE_DS41_ARGATHER)"*) ;; *) log "RESULT xwk $1: VIRHE the ARGATHER note is not today's 4-row one: '${AL}'"; ok=0 ;; esac
  [ -z "$XL" ] || { log "RESULT xwk $1: VIRHE a shared-expert-lane note: '$XL'"; ok=0; }
  [ -n "$SL" ] || { log "RESULT xwk $1: VIRHE no prefill staging line"; ok=0; }
  case "$KL" in *"ds41_kvfold=1"*"ds41_ofold=1"*"ds41_onefill=1"*) ;; *) log "RESULT xwk $1: VIRHE the K1 switches do not start at 1: '${KL}'"; ok=0 ;; esac
  case "$WL" in *"dspark_walk_step a step + all_gather over 4 rank(s)"*"dspark_walk=$2 "*) ;; *) log "RESULT xwk $1: VIRHE no W1 declared line with dspark_walk=$2 over 4 ranks: '${WL}'"; ok=0 ;; esac
  NW=$(docker logs $CN 2>&1 | grep -c 'the acquire rule plays .*RADIANCE_AQL_ACQ=rule-w)'); NR=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_AQL_ACQ=rule)')
  [ "$NW" -gt 0 ] && [ "$NR" = 0 ] || { log "RESULT xwk $1: VIRHE rule-w is not what the recorded passes play ($NW rule-w, $NR rule)"; ok=0; }
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
draftlines() {
  [ "$DRY" = 1 ] && { echo "+ RESULT xwk $1 drafter lines"; return 0; }
  log "RESULT xwk $1 drafter / speculation lines: $(docker logs $CN 2>&1 | grep -iE 'dspark|drafter|speculat|draft-window|draft window|draft depth|draft hybrid|oracle' | sed 's/^[A-Z][^ ]* //' | sort -u | head -12 | tr '\n' '|' | cut -c1-1800)"
}
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
fn4code() {
  [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null
}
pcreset() {
  [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o $O/$1-reset.json -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  log "RESULT xwk $1 prefix cache reset: HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"
}
swnote() {   # swnote TAG: GET at start (the stack env's values, dspark_walk 0); dspark_walk 2 refused; A's then B's set applied at a step
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (expect gg_gemv 2, aql_acq 2, K1 three 1, draft_adaptive 1, draft_capable 1, dspark_walk 0, epoch 0); POST dspark_walk 2 -> 400; POST A's set $(armset "$ARM_A") then B's $(armset "$ARM_B"), each + a 4-token request -> applied"; return 0; }
  local s0 s1 s2 c2
  s0=$(fn4get); c2=$(fn4code '{"dspark_walk": 2}')
  fn4set "$(armset "$ARM_A")"; nudge; s1=$(fn4get)
  fn4set "$(armset "$ARM_B")"; nudge; s2=$(fn4get)
  log "RESULT xwk $1 switch note: at start $s0 | dspark_walk 2 -> HTTP $c2 | after A's set: $s1 | after B's set: $s2 | core log: $(docker logs $CN 2>&1 | grep -E 'fn4: switches' | tail -2 | sed 's/^[A-Z] *//' | tr '\n' '|' | cut -c1-400)"
  python3 -c 'import json,sys
s0, s1, s2 = (json.loads(x) for x in sys.argv[1:4]); a, b = json.loads(sys.argv[4]), json.loads(sys.argv[5])
w = s0["switches"]
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) \
     and w["draft_adaptive"] == 1 and w.get("dspark_walk") == 0 and s0["draft_capable"] == 1 and s0["epoch"] == 0
ok = ok and all(s1["switches"].get(k) == v for k, v in a.items()) and not s1["pending"] and s1["epoch"] >= 1
ok = ok and all(s2["switches"].get(k) == v for k, v in b.items()) and not s2["pending"] and s2["epoch"] > s1["epoch"]
sys.exit(0 if ok else 1)' "$s0" "$s1" "$s2" "$(armset "$ARM_A")" "$(armset "$ARM_B")" 2>/dev/null && [ "$c2" = 400 ]
}
walksmoke() {   # walksmoke TAG: under B (the switch note leaves it set) the walk must survive real query passes before the blocks:
  # one 96-token code request (S 1), then four at once (S up to 4); health 200 and no refused op / failed query pass in the
  # serve log, else the start stops here (10.11.: the 4-token nudge drafted nothing, and the first B block died at step 4251)
  [ "$DRY" = 1 ] && { echo "+ walk smoke under B: a 96-token code request, then 4 at once (64 tokens): health 200, no refused op / failed query pass in the log; draft_n logged"; return 0; }
  local r dn h e i
  r=$(curl -s -m 300 -X POST -H 'Content-Type: application/json' -d '{"prompt": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n", "n_predict": 96, "temperature": 0}' $URL/completion 2>/dev/null)
  dn=$(echo "$r" | python3 -c 'import json,sys; print((json.load(sys.stdin).get("timings") or {}).get("draft_n", 0))' 2>/dev/null)
  for i in 1 2 3 4; do
    curl -s -m 300 -X POST -H 'Content-Type: application/json' -d '{"prompt": "Write a short story, number '$i', about a lighthouse keeper.", "n_predict": 64, "temperature": 0}' $URL/completion > /dev/null 2>&1 &
  done
  wait
  h=$(curl -s -m 5 -o /dev/null -w '%{http_code}' $URL/health 2>/dev/null)
  e=$(docker logs $CN 2>&1 | grep -cE "refused op|query pass of step .* failed|cannot continue")
  log "RESULT xwk $1 walk smoke under B (dspark_walk 1): health $h, $e refused-op / failed-pass line(s), draft_n of the code request ${dn:-?}$([ "${dn:-0}" = 0 ] && echo " (no draft at c1: the c4 burst is the check)") | fn4 $(fn4get | cut -c1-200)"
  [ "$h" = 200 ] && [ "$e" = 0 ] || { keeplog $1-smoke; return 1; }
}
blocks() {   # blocks TAG: the A / B decode blocks (wkblocks.py) and their lines
  run "timeout 7200 python3 -I -B $WKB --url $URL --mb-dir $D --edit $EP --label $1 --arm 'A=$ARM_A' --arm 'B=$ARM_B' --pair A,B --order $ORDER --dur $DUR --warm 10 --fixed '$FIX' $([ "$FIXED5" = 1 ] && echo --fixed5) --out $O/$1-blocks.jsonl > $O/$1-blocks.txt 2>&1"
  [ "$DRY" = 1 ] && return 0
  while IFS= read -r l; do log "RESULT xwk $1 ${l:0:2000}"; done < <(grep -E '^(BLOCK|WKB \{"error)' $O/$1-blocks.txt)
}
copyt() {   # copyt TAG OFFSET: the copy test after a ~31k-token cold prompt (105's prompt: same offset -> same text)
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset $2 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xwk $1 copy after a cold prompt: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300)"
}
gsetc1() { run "(cd $D && timeout 900 python3 $GS $URL $O/$1-gs-c1.jsonl --conc 1 --max-tokens 192 > $O/$1-gs-c1.txt 2>&1)"; }
gcmp() { [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }; [ -s "$1" ] && [ -s "$2" ] || { echo "not compared (missing $1 or $2)"; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
ident() {   # ident TAG ARM: the arm's set + epoch_bump, a prefix-cache reset, the 28k copy and the c1 greedy set vs bz d0
  local tag=$1 arm=$2 r
  fn4set "$(armset "$([ $arm = B ] && echo "$ARM_B" || echo "$ARM_A")" '{"epoch_bump": 1}')"
  pcreset $tag
  copyt $tag 0; gsetc1 $tag
  [ "$DRY" = 1 ] && { echo "+ identity ($arm): 28k copy vs bz d0, greedy set c1 vs bz d0"; return 0; }
  log "RESULT xwk $tag identity ran under: $(fn4get | cut -c1-500)"
  r=$(cmpfile $O/copy-$tag-run1.txt $RD/copy-dsd-d0-i1-run1.txt)
  echo "copy-$arm $r" >> $O/identity.txt
  log "RESULT xwk $tag identity ($arm, dspark_walk $([ $arm = B ] && echo 1 || echo 0)): 28k copy vs bz d0 (NST 0, stock): $r$(case $r in IDENTICAL*) ;; *) echo " -- reported, not hidden" ;; esac)"
  r=$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$tag-gs-c1.jsonl)
  echo "gs-$arm $r" >> $O/identity.txt
  log "RESULT xwk $tag identity ($arm): greedy set c1 vs bz d0: $r$(case $r in IDENTICAL*) ;; *) echo " -- reported, not hidden" ;; esac)"
}
selftest() {   # the image's r4d_selftest --case dspark_walk_step_f32 on the cards (no server)
  local f=$O/selftest-walk.txt rc nf nok
  run "gpurun st-$$-walk 900 $DEV --entrypoint /opt/radiance/bin/r4d_selftest $IMAGE /opt/radiance/share/radiance/kernels/libr4d.so --case dspark_walk_step_f32 > $f 2>&1"
  rc=$?
  [ "$DRY" = 1 ] && return 0
  nf=$(grep -c 'FAIL' $f); nok=$(grep -cE '^ *ok ' $f)
  local r="rc $rc, $nok ok, $nf FAIL$([ $nf -gt 0 ] && echo " -- first: $(grep -m1 FAIL $f | cut -c1-300)")"
  echo "selftest $r" >> $O/identity.txt
  log "RESULT xwk selftest dspark_walk_step_f32 (the kernel on the cards vs a host walk, 1-4 shards, ties, the real vocabulary): $r | $(grep -E '^ *(ok|FAIL) +dspark_walk_step_f32 V' $f | sed -E 's/^ *//' | cut -c1-140 | tr '\n' '|')"
  log "RESULT xwk selftest launch time of one walk step at the TP4 shard (warm): $(grep -E 'PERF +dspark_walk_step_f32' $f | sed -E 's/^ *PERF +dspark_walk_step_f32 one step, //' | tr '\n' '|' | cut -c1-600)"
  log "RESULT xwk selftest the walk's key gather on four cards (ar_gather_4rank at the 16-byte slot geometry, S 1-8; the 8-byte row refused): $(grep -E '^ *(ok|FAIL|skip) +dspark_walk_step_f32 gather' $f | sed -E 's/^ *//' | cut -c1-170 | tr '\n' '|')"
  [ $rc = 0 ] && [ $nf = 0 ] && [ $nok -gt 0 ]
}
oracle() {   # the oracle start: NST 5 fixed window (6-row verifies) + the walk on from the start, proposals replaced by the harness
  local tag=wk-oracle r i
  walkenv 1; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --no-prefix-cache --debug-draft-oracle /stats/oracle-ids.txt"
  log "RESULT xwk $tag: start IMAGE=$IMAGE NST 5 (fixed window) + RADIANCE_DS41_DSPARK_WALK=1 + --debug-draft-oracle (the DSpark head's proposals replaced) $(desc)"
  run "cp $RD/oracle-ids.txt $STATSDIR/oracle-ids.txt"
  restart $tag || { keeplog $tag-fail; down; log "RESULT xwk $tag: VIRHE not healthy: $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; echo "oracle not healthy" >> $O/identity.txt; return 1; }
  if [ "$DRY" != 1 ]; then
    log "RESULT xwk $tag harness lines: $(docker logs $CN 2>&1 | grep -E -- '--debug-draft-oracle|DSpark lever W1|speculation: ' | sed 's/^[A-Z] *//' | head -5 | tr '\n' '|' | cut -c1-900) | fn4 $(fn4get | cut -c1-300)"
    docker logs $CN 2>&1 | grep -q "every proposal of the draft head" && docker logs $CN 2>&1 | grep -q "beside the draft head 'dspark': no step is issued ahead" \
      || log "RESULT xwk $tag: VIRHE the serve log lacks the harness-beside-a-head lines"
    case "$(fn4get)" in *'"dspark_walk": 1'*) ;; *) log "RESULT xwk $tag: VIRHE dspark_walk is not 1 at the oracle start" ;; esac
  fi
  for i in 1 2; do
    run "timeout 900 python3 -I $ORC check $URL $STATSDIR/oracle-ids.txt $O/$tag-check$i.json > $O/$tag-check$i.txt 2>&1"
    [ "$DRY" = 1 ] && continue
    r=$(grep -m1 '^ORACLE' $O/$tag-check$i.txt | cut -c8- | cut -c1-300)
    case "$r" in
      *'"identical": true'*)
        case "$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); n, a = d.get("draft_n") or 0, d.get("draft_n_accepted") or 0; print("ok" if n > a > 0 else "vacuous")' "$r" 2>/dev/null)" in
          ok) echo "oracle-$i IDENTICAL" >> $O/identity.txt; log "RESULT xwk $tag check $i: IDENTICAL to d0's ids, the harness drafted and was rejected, the walk on ($r)" ;;
          *)  echo "oracle-$i VACUOUS" >> $O/identity.txt; log "RESULT xwk $tag check $i: IDENTICAL but VIRHE vacuous -- no rejected draft ($r)" ;;
        esac ;;
      *) echo "oracle-$i DIFFERENT" >> $O/identity.txt; log "RESULT xwk $tag check $i: VIRHE DIFFERENT from d0's ids ($r) -- reported, not hidden" ;;
    esac
  done
  keeplog $tag; down
  log "RESULT xwk $tag: done"
}
summary() {
  [ "$DRY" = 1 ] && { echo "+ RESULT xwk arm / pair / position lines, RESULT xwk summary: per metric A, B, B/A (pairs t95), %; tokens a step, ms a round, tok/s; positions; target distance; identity / oracle / selftest"; return 0; }
  python3 - $O/wk-n5-blocks.txt $O/identity.txt > $O/summary.txt 2>&1 <<'PY'
import json, sys
wkb = None
try:
    for l in open(sys.argv[1]):
        if l.startswith("WKB "): wkb = json.loads(l[4:])
except OSError: pass
try: ident = [l.strip() for l in open(sys.argv[2]) if l.strip()]
except OSError: ident = []
if not wkb or "metrics" not in wkb:
    print("SUMMARY no block summary (WKB line missing) || identity: " + ", ".join(ident)); sys.exit(0)
M = wkb["metrics"]
def f(x, n=2): return "-" if x is None else format(x, ".%df" % n)
def mean(nm, arm): return ((M.get(nm) or {}).get("arms") or {}).get(arm, {}).get("mean")
def pair(nm):
    d = M.get(nm) or {}
    m, h = d.get("pair_mean"), d.get("ci95")
    return "-" if m is None else "%+.1f %%%s" % ((m - 1) * 100, "" if h is None else " +- %.1f" % (h * 100))
def tgt(v, t):
    if v is None: return ""
    return "target >= %g: %s" % (t, "MET (+%.1f)" % (v - t) if v >= t else "%.1f short, +%.1f %% needed" % (t - v, (t / v - 1) * 100))
lines = []
for base, tps, tps_t, tpstep, ms, acc in (("c1 essay", "c1 essay tok/s", 90, "c1 essay tok/step", "c1 essay ms/step", "c1 essay acc"),
                                          ("c1 code", "c1 code tok/s", 90, "c1 code tok/step", "c1 code ms/step", "c1 code acc"),
                                          ("edit", "edit tok/s", 90, "edit tok/step", "edit ms/step", "edit acc"),
                                          ("c4", "c4 tok/s", 160, "c4 tok/step", "c4 ms/step", "c4 acc"),
                                          ("c8", "c8 tok/s", 200, "c8 tok/step", "c8 ms/step", "c8 acc")):
    a, b = mean(tps, "A"), mean(tps, "B")
    s = "%s: tok/s B %s vs A %s (%s; %s)" % (base, f(b, 1), f(a, 1), pair(tps), tgt(b, tps_t))
    s += ", tokens a step B %s vs A %s (%s)" % (f(mean(tpstep, "B")), f(mean(tpstep, "A")), pair(tpstep))
    s += ", ms a round B %s vs A %s (%s)" % (f(mean(ms, "B")), f(mean(ms, "A")), pair(ms))
    s += ", acc B %s vs A %s" % (f(mean(acc, "B"), 3), f(mean(acc, "A"), 3))
    if base in ("c4", "c8"):
        p = base + " p50"
        s += ", p50 B %s vs A %s (%s)" % (f(mean(p, "B"), 1), f(mean(p, "A"), 1), pair(p))
    lines.append(s)
    print("ARMS " + s)
for nm in ("fixed5 essay tok/step", "fixed5 code tok/step"):
    if nm in M:
        s = "%s (diagnostic, fixed window 5): B %s vs A %s (%s)" % (nm, f(mean(nm, "B")), f(mean(nm, "A")), pair(nm))
        lines.append(s); print("ARMS " + s)
P = wkb.get("positions") or {}
pl = []
for ph in ("essay", "code", "edit", "c4", "c8", "essay5", "code5"):
    parts = []
    for arm in ("A", "B"):
        ps = (P.get(arm) or {}).get(ph) or []
        parts.append("%s %s" % (arm, " ".join("p%d %s (%d/%d)" % (p["p"], f(p["accepted"] / p["reached"], 3) if p["reached"] else "-", p["accepted"], p["reached"]) for p in ps) or "-"))
    s = "%s: %s" % (ph, " | ".join(parts))
    pl.append(s); print("POS " + s)
print("SUMMARY " + " || ".join(lines) + " || accepted / reached by draft position (conditional acceptance): " + " ;; ".join(pl)
      + " || blocks %d (%d flagged), order %s || identity / oracle / selftest: %s" % (wkb.get("blocks", 0), wkb.get("blocks_bad", 0), wkb.get("order"), ", ".join(ident) or "none recorded"))
PY
  while IFS= read -r l; do case "$l" in SUMMARY*) ;; *) log "RESULT xwk ${l:0:2500}" ;; esac; done < $O/summary.txt
  log "RESULT xwk summary: $(grep -m1 '^SUMMARY ' $O/summary.txt | cut -c9- | cut -c1-6000)"
}
# ---------------------------------------------------------------- main
log "RESULT xwk $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) | one NST 5 start, hx-stack env + RADIANCE_DS41_DSPARK_WALK=0, blocks $ORDER (A $ARM_A, B $ARM_B, fixed $FIX), mb $DUR s, fixed-window diagnostic $FIXED5 | selftest $DO_SELFTEST, oracle $DO_ORACLE | identity vs bz d0 | targets 1u >= 90, c4 >= 160, c8 >= 200 | estimate (LEVERS W1): prose tok/step +3..+10 %, +0.5-0.7 ms a round"
docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xwk: VIRHE image $IMAGE not on this host"; exit 1; }
[ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -qzF "draft_lend_borrow, dspark_walk, epoch_bump" /opt/radiance/bin/radiance && grep -qzF "draft depth 0: a step drafts nothing" /opt/radiance/bin/radiance && grep -qzF "draft hybrid: a sequence whose prompt lookup" /opt/radiance/bin/radiance && grep -qzF reset_prefix_cache /opt/radiance/bin/radiance && grep -qzF "bytes win where no snapshot rests on the first copy" /opt/radiance/bin/radiance && grep -qzF RADIANCE_DS41_KVFOLD /opt/radiance/share/radiance/architectures/deepseek41.so && grep -qzF RADIANCE_DS41_DSPARK_WALK /opt/radiance/share/radiance/architectures/deepseek41.so && grep -qzF dspark_walk_step /opt/radiance/share/radiance/kernels/libr4d.so && grep -qzF rmsnorm_rope_roundtrip_kv_store /opt/radiance/share/radiance/kernels/libr4d.so && grep -qzF r4d_gg_mr_pin /opt/radiance/share/radiance/kernels/libr4d.so && grep -qzF dspark_walk_step_f32 /opt/radiance/bin/r4d_selftest && grep -qzF "dspark_walk_step_f32 gather: ar_gather_4rank moves" /opt/radiance/bin/r4d_selftest && echo yes' 2>/dev/null)" = yes ] \
  || { log "RESULT xwk: VIRHE $IMAGE is not the walk image (the registry with dspark_walk, depth 0 / hybrid, reset_prefix_cache, R1, K1, D2, RADIANCE_DS41_DSPARK_WALK, dspark_walk_step + its selftest case with the 16-byte-slot gather)"; exit 1; }
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)) is the walk image -- OK"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xwk: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  [ -e $Y ] && grep -q '^ *RADIANCE_DS41_DSPARK_WALK:' $Y && grep -q '^ *RADIANCE_DS41_GG_GEMV:' $Y && grep -q '^ *RADIANCE_PLACE_TURNOVER:' $Y \
    || { log "RESULT xwk: VIRHE $Y missing or does not pass RADIANCE_DS41_DSPARK_WALK / GG_GEMV / PLACE_TURNOVER"; exit 1; }
  for f in $GS $EP $WKB $GGB $ORC $D/stepprobe.py $D/mb.py $D/ds41_tests.py; do [ -e $f ] || { log "RESULT xwk: VIRHE $f missing"; exit 1; }; done
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xwk: VIRHE $M225 missing or not verified"; exit 1; }
  for f in $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $RD/oracle-ids.txt; do [ -s $f ] || log "RESULT xwk: note -- reference $f missing (not compared)"; done
else
  for f in $Y $GS $EP $WKB $GGB $ORC $D/stepprobe.py $D/mb.py $D/ds41_tests.py $MODELDIR/$M225 $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $RD/oracle-ids.txt $WARM_REF_225; do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  echo "+ yaml passes RADIANCE_DS41_DSPARK_WALK: $(grep -c '^ *RADIANCE_DS41_DSPARK_WALK:' $Y), RADIANCE_PLACE_TURNOVER: $(grep -c '^ *RADIANCE_PLACE_TURNOVER:' $Y)"
  python3 -I -B $WKB --help > /dev/null 2>&1 && echo "+ wkblocks.py parses (imports ggblocksdz2)" || echo "+ MISSING: wkblocks.py does not run"
fi
mkd $O $O/guard $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt
evict
if [ "$DO_SELFTEST" = 1 ] && ! stopfile; then
  if [ "$DRY" != 1 ]; then
    for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
    [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || log "RESULT xwk: note -- a KFD client is running before the selftest"
  fi
  selftest || log "RESULT xwk: VIRHE the selftest is not clean -- the walk's proposals may not be the reference walk's (measured anyway)"
  if [ "$DRY" != 1 ] && { grep -qE '^ *FAIL +dspark_walk_step_f32 gather' $O/selftest-walk.txt || ! grep -qE '^ *ok +dspark_walk_step_f32 gather: ar_gather_4rank moves' $O/selftest-walk.txt; }; then
    log "RESULT xwk: VIRHE the walk's key gather does not run on the four cards (selftest) -> no server start (it would fail as at step 4251)"
    summary; down; log "RESULT xwk $ITEM: done -- $O (stopped at the selftest)"; exit 1
  fi
fi
# ---- the NST 5 start: switch note, blocks, identity
tag=wk-n5; P=$O/$tag
walkenv 0; export MODEL=$M225; HP29=19456; best_vars
export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive"
log "RESULT xwk $tag: start ($DESC_A at start) IMAGE=$IMAGE $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
mkd $P
if ! stopfile; then
  wgate $tag 70; rc=$?
  if [ $rc != 0 ]; then
    log "RESULT xwk $tag: VIRHE $([ $rc = 2 ] && echo "over the gate limit after one restart" || echo "not healthy: $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)") -> start skipped"; down
  elif ! modes $tag 0; then
    log "RESULT xwk $tag: VIRHE the modes are not hx-stack's + W1 -> start stopped"; down
  else
    draftlines $tag
    if ! swnote $tag; then
      log "RESULT xwk $tag: VIRHE the switches are not as expected -> start stopped"; keeplog $tag-switch; down
    else
      identw $tag $WARM_REF_225
      if ! walksmoke $tag; then
        log "RESULT xwk $tag: VIRHE the full-vocabulary walk did not survive its smoke -> start stopped (blocks, identity skipped)"; down
        DO_ORACLE=0
      else
      follow $P
      stopfile || blocks $tag
      stopfile || ident $tag-iB B
      stopfile || ident $tag-iA A
      unfollow
      run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
      down
      log "RESULT xwk $tag: done"
      fi
    fi
  fi
fi
[ "$DO_ORACLE" = 1 ] && ! stopfile && oracle
summary
unset $SWITCHES
down
log "RESULT xwk $ITEM: done -- $O"
