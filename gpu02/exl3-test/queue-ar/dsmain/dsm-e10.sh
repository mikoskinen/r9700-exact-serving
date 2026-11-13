#!/bin/bash
# dsm-e10.sh (11.11.2026, queue-ar/dsmain, ds-main-port E10 agent) -- NOT a queue item; the body of 57-p4zyzzzzz-ds-*-e10.sh.
# LEVER D5 = E10, THE FUSED EXL3 DECODE CHAIN ON DEEPSEEK-V4.1: moe_exl3_gate_up_down (radiance-exl3 interface v11, the SwiGLU
# clamp `limit` on the fused op; exl3r4d's fused launch at K 2 / 2.5 for both projections, fn4-dsgd 1a1d429 / a3e0f1c) issued
# in the decode band (passes of 3..16 tokens) instead of the split moe_exl3_gate_up + moe_exl3_down -- the arch side ds41-e10
# 1d28b33 (= ds41-int2 745ce76 + the hook), A/B'd IN ONE PROCESS a start on the ds41-e10 image (XE10_IMAGE; see its
# e10-PROVENANCE): A = e10_gd 0 (the split pair, today's ops), B = e10_gd 1 (the fused op in its band). RADIANCE_DS41_E10_GD=1
# at start declares the fused op (unset declares nothing: the int2 image's graph, hostcheck). fn4's Flash-Next reading (156b):
# c4 +5.0 % +- 2.3 (ms/step -5 %), c2 flat. Exact by construction: from 3 tokens both forms are split-K 1, the fused y is the
# split chain's bits.
# P0 FIRST (one card, HIP 0, no server; ~6 min; XE10_P0=0 skips): the image's perfmoe-e10 --chain --gd at DeepSeek-V4.1's EP4
#   rank geometry (96 experts a rank of 384, top 6, hidden 5120, width 2304, --ep 4 --ep-rank -1, h_rot 1, random data):
#   GD-CHECK (the fused op's y against the split chain's, byte for byte; perfmoe exits 1 on a difference) at --limit 10 (the
#   model's swiglu_limit) for BOTH K pairs (--bits 2 = 29-30 layers, --bits 2.5 = layers 0-5 / 35-39) at M 3 / 4 / 6 / 8 / 12 /
#   16, then a forced clamp (--limit 0.25, M 8) whose chain y must differ from limit 10's (the clamp reaches the kernels) and
#   still pass GD-CHECK. Any GD-CHECK failure stops the item (no server phase). The chain-vs-gd us are logged (perfmoe ABAB).
# THEN TWO STARTS (the hx-stack item's stack env: gg_gemv 2, aql_acq 2 (rule-w), K1 folds c / d / f, RADIANCE_PLACE_TURNOVER=0;
#   EXL3 2.25, TP4, MML 524288, the t3 env and serve line, MNBT 4160, --live; + RADIANCE_DS41_E10_GD=1):
#   nst0   NST 0 (no drafter): c1 = 1-token passes (outside the band: a control), c4 / c8 = 4 / 8-token passes (in band).
#   nst5   NST 5 DSpark --draft-window adaptive + D7 (draft_depth0 1) + hybrid (draft_hybrid 1, match 8) + drafter_mx 1: c1
#          verify passes of up to 6 tokens (in band), c4 / c8 up to 24 / 48 (in band at depth 0 or short drafts).
# PER START: gate -> modes (the stack's notes + the E10 notes: the fused op declared at every EXL3 layer, both dtypes, none
#   refused) -> switch note (GET at start e10_gd 1; e10_gd 2 refused; A's and B's sets applied at a step) -> warm-up answers
#   -> BLOCKS (ggblocksdz2.py, XE10_ORDER default A B B A, --pair A,B; POST {fixed pin, e10_gd, epoch_bump} each block;
#   stepprobe essay,code 512 (c1), essay by draft position, editprobe, mb c4 / c8 XE10_DUR 75 s, warm 10) -> IDENTITY
#   (prefill-heavy, last): B then A, each after a prefix-cache reset: the 28k copy (a ~31k-token cold prompt) + the 20-prompt
#   greedy set at c1; B vs A byte for byte (STRICT both starts: an exact lever), each vs bz d0 (strict at NST 0, reported at
#   NST 5 as the stack item does) -> the start's B-vs-A line (pooled mean(B)/mean(A), each pair, t95 over pairs).
# Tags "RESULT xe10 ", "RESULT xe10 summary: ...", "RESULT xe10 <item>: done". Out runs/xdsm/e10.
# ~55 min (P0 ~6, evict ~1, nst0 ~22, nst5 ~25). NO cc_sim (user 10.11.). No power / host / fan settings; production
# compose files read only (--project-directory). Exact lever; nothing goes into production here.
# XE10_IMAGE, XE10_IMGID, XE10_ITEM, XE10_STARTS ("nst0 nst5"), XE10_ORDER (ABBA), XE10_DUR (75), XE10_P0 (1), XE10_VRAMENV
# (default RADIANCE_PLACE_TURNOVER=0).
# DRYRUN=1 bash dsm-e10.sh : prints every command; checks the image (on this host, its E10 files and strings, read only, no
#   device, no network), every file the item reads and that every shell helper resolves; no server, no GPU, nothing written.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-e10-tp4.yaml
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=$R/xdsm/e10; export STATSDIR=$O/stats
RD=$R/xdsm/draft                                # bz's d0 files (identity reference: copy + greedy set)
export IMAGE=${XE10_IMAGE:-radiance-local:ds41-e10-1d28b33c-76e24c49-l}
IMGID=${XE10_IMGID-3914f475365a}
ITEM=${XE10_ITEM:-57-p4zyzzzzz-ds-e10}; ORDER=${XE10_ORDER:-ABBA}; DUR=${XE10_DUR:-75}; STARTS=${XE10_STARTS-"nst0 nst5"}
case $ORDER in *[!AB]*|"") echo "XE10_ORDER takes A / B blocks (got '$ORDER')"; exit 1 ;; esac
case $ORDER in *A*B*|*B*A*) ;; *) echo "XE10_ORDER '$ORDER' needs both arms"; exit 1 ;; esac
for s in $STARTS; do case $s in nst0|nst5) ;; *) echo "XE10_STARTS takes nst0 / nst5 (got '$s')"; exit 1 ;; esac; done
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
EP=$HD/editprobe.py; GGB=$HD/ggblocksdz2.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
MATCH=8
FIX0='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1}'   # NST 0: the stack's N pin
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid": 1, "draft_hybrid_match": '"$MATCH"', "drafter_mx": 1}'
ARM_A='{"e10_gd": 0}'
ARM_B='{"e10_gd": 1}'
DESC0="NST 0 stack: gg_gemv 2, rule-w, K1 folds c/d/f, no drafter"
DESC5="NST 5 stack: gg_gemv 2, rule-w, K1 folds c/d/f, DSpark NST 5 adaptive, depth 0, hybrid 1 (match $MATCH), drafter_mx 1"
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DS41_ARGATHER_MAX RADIANCE_DS41_SHEXP_LANE RADIANCE_DEBUG_MISS_LOG
  RADIANCE_DS41_KVFOLD RADIANCE_DS41_OFOLD RADIANCE_DS41_ONEFILL RADIANCE_DS41_PIPE
  RADIANCE_DRAFT_DEPTH0 RADIANCE_DRAFT_PROBE RADIANCE_DRAFT_HYBRID RADIANCE_DRAFT_HYBRID_MATCH RADIANCE_DS41_E10_GD"
VRAMENV=${XE10_VRAMENV-RADIANCE_PLACE_TURNOVER=0}
for kv in $VRAMENV; do
  case "$kv" in [A-Z]*=*) ;; *) echo "XE10_VRAMENV takes NAME=VALUE words (got '$kv')"; exit 1 ;; esac
  case " $(echo $SWITCHES) " in *" ${kv%%=*} "*) ;; *) echo "XE10_VRAMENV: ${kv%%=*} is not in the switch list"; exit 1 ;; esac
done
[ -n "$VRAMENV" ] && { DESC0="$DESC0, VRAM env $VRAMENV"; DESC5="$DESC5, VRAM env $VRAMENV"; }
e10env() {   # 105's t3 arm + gg_gemv 2 + rule-w + the three K1 folds + the VRAM env (the hx-stack item's) + E10 declared (starts at 1)
  local kv
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  for kv in $VRAMENV; do export "$kv"; done
  export RADIANCE_DS41_E10_GD=1
}
armset() {   # armset FIXJSON ARMJSON [EXTRAJSON]: one POST body
  local x=${3:-}; [ -n "$x" ] || x='{}'
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update(json.loads(sys.argv[2])); d.update(json.loads(sys.argv[3])); print(json.dumps(d))' "$1" "$2" "$x"
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
  log "RESULT xe10 guard $tag: $st, 1-user ${ms} ms/step (limit $lim; NST 5 DSpark starts 35.2-35.3, NST 0 bz dsd-d0 15.59)"
  case $st in SLOW*) return 2 ;; esac
}
FOLLOW_PID=""
follow() { [ "$DRY" = 1 ] && { echo "+ docker logs -f $CN > $1/engine.log &"; return; }
  docker logs -f $CN > $1/engine.log 2>&1 & FOLLOW_PID=$!; sleep 2; }
unfollow() { [ -n "$FOLLOW_PID" ] && { kill $FOLLOW_PID 2>/dev/null; wait $FOLLOW_PID 2>/dev/null; FOLLOW_PID=""; }; return 0; }
trap 'unfollow; down' EXIT
trap 'exit 143' TERM INT
thermline() { type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"; }
cmpfile() {   # cmpfile A B -> IDENTICAL (n chars) | DIFFERENT at char k (na vs nb chars) | not compared
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xe10 $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xe10 $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  echo "warm-up-$1 $r" >> $O/identity.txt
  case $r in IDENTICAL*) log "RESULT xe10 $1 warm-up answers (e10_gd 1 from start) vs stock ($(basename $2)): $r" ;;
    *) log "RESULT xe10 $1 warm-up answers (e10_gd 1 from start) vs stock ($(basename $2)): VIRHE $r -- NOT EXACT vs the tuned3l line" ;; esac
}
VKNOBS=""
vrammodes() {   # = dsm-stack.sh's: the VRAM env in the container, the mover's counts a rank, the expert prefetch state
  local MV XP CE kv bad="" nz
  MV=$(docker logs $CN 2>&1 | grep -E '^I\[[0-9]\] mover: [0-9]+ slab slots' | sed -E 's/^I\[([0-9])\] mover: ([0-9]+) slab slots \(([0-9]+) shadow, ([0-9]+) more kept free.*/r\1 \3 shadow \4 kept free (\2 slots)/' | tr '\n' ',' | sed 's/,$//')
  XP=$(docker logs $CN 2>&1 | grep -cE '^I rank [0-9]+: expert prefetch on')
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in $VRAMENV RADIANCE_DS41_E10_GD=1; do echo "$CE" | grep -qxF "$kv" || bad="$bad ${kv%%=*}"; done
  VKNOBS="env [${VRAMENV:-none} RADIANCE_DS41_E10_GD=1]$([ -n "$bad" ] && echo " NOT in the container:$bad"); mover ${MV:-no line}; expert prefetch on at $XP of 4 ranks"
  [ -z "$bad" ] || { log "RESULT xe10 $1: VIRHE the env did not reach the container:$bad"; return 1; }
  [ "$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots')" = 4 ] || { log "RESULT xe10 $1: VIRHE no mover line on all 4 ranks: ${MV:-none}"; return 1; }
  case " $VRAMENV " in *" RADIANCE_PLACE_TURNOVER=0 "*)
    nz=$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots \([0-9]+ shadow, 0 more kept free')
    [ "$nz" = 4 ] || { log "RESULT xe10 $1: VIRHE turnover 0 asked but $nz of 4 ranks keep 0 slots free: $MV"; return 1; } ;;
  esac
  return 0
}
modes() {   # modes TAG: the stack line's notes (t3 indexer, ARGATHER 4 rows, K1 all three declared and on, rule-w, VRAM) + E10's
  [ "$DRY" = 1 ] && { echo "+ modes $1: indexer note, ARGATHER 4-row note, no shared-expert-lane note, staging line, rule-w in every recorded pass, the K1 note (all three declared, =1); VRAM env [${VRAMENV:-none}] + RADIANCE_DS41_E10_GD=1 in the container, mover on 4 ranks; E10: the core's 'resolve_bands op=moe_exl3_gate_up_down key=M lo=3 hi=16' once per 'resolve_bands op=moe_exl3_gate_up ' (every EXL3 layer of every rank, both K sets), and the kernel selection line's exl3r4d op bands above the int2 image's 160 (the device library serves the fused op, not exl3ref on the host)"; return 0; }
  local MLF SL AL XL KL NW NR XN EN EB ED NX ok=1
  vrammodes $1 || ok=0
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  AL=$(docker logs $CN 2>&1 | grep -m1 'the MoE gather and its all-reduce as one kernel' | sed 's/^radiance: //' | cut -c1-300)
  XL=$(docker logs $CN 2>&1 | grep -m1 'the shared expert on the second lane' | sed 's/^radiance: //' | cut -c1-300)
  KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-500)
  # the arch's per-layer notes go to --debug-graph only (hostcheck); in a served start the core's band lines show the declare
  XN=$(docker logs $CN 2>&1 | grep -c 'resolve_bands op=moe_exl3_gate_up key=M')
  EN=$(docker logs $CN 2>&1 | grep -c 'resolve_bands op=moe_exl3_gate_up_down key=M lo=3 hi=16')
  EB=$(docker logs $CN 2>&1 | grep -oE 'resolve_bands op=moe_exl3_gate_up_down key=M lo=[0-9]+ hi=[0-9]+' | sort | uniq -c | tr -s ' ' | tr '\n' '|')
  ED=$(docker logs $CN 2>&1 | grep -m1 -E '^I +kernel selection' | sed -E 's/^I +kernel selection +//' | cut -c1-200)
  NX=$(echo "$ED" | grep -oE 'exl3r4d [0-9]+' | awk '{print $2}')
  log "RESULT xe10 $1 modes: $(echo "$MLF" | cut -c1-700) | staging: ${SL:-none} | argather: ${AL:-none} | K1: ${KL:-none} | VRAM: ${VKNOBS:-none}"
  log "RESULT xe10 $1 E10 declare: $EN fused-op band line(s) (3..16) for $XN EXL3 gate_up band line(s) [${EB:-none}]; kernel selection: ${ED:-no line} (int2 image: exl3r4d 160, exl3ref (host) 160)"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate" "post-collapse-norm separate at 0 layers" "next collapse-norm separate at 0 layers"; do
    case "$MLF" in *"$w"*) ;; *) log "RESULT xe10 $1: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  case "$AL" in *"up to 4 tokens a step (RADIANCE_DS41_ARGATHER)"*) ;; *) log "RESULT xe10 $1: VIRHE the ARGATHER note is not today's 4-row one: '${AL}'"; ok=0 ;; esac
  [ -z "$XL" ] || { log "RESULT xe10 $1: VIRHE a shared-expert-lane note (RADIANCE_DS41_SHEXP_LANE should be unset): '$XL'"; ok=0; }
  [ -n "$SL" ] || { log "RESULT xe10 $1: VIRHE no prefill staging line (levels refused?)"; ok=0; }
  case "$KL" in
    *"in one launch at 0 of"*|*"rope+round trip at 0 ("*|*"not declared at"*|*"at 0 MoE layers"*|"")
      log "RESULT xe10 $1: VIRHE the lever K1 note does not show all three folds declared: '${KL}'"; ok=0 ;;
  esac
  case "$KL" in *"ds41_kvfold=1"*"ds41_ofold=1"*"ds41_onefill=1"*) ;; *) log "RESULT xe10 $1: VIRHE the K1 switches do not start at 1: '${KL}'"; ok=0 ;; esac
  NW=$(docker logs $CN 2>&1 | grep -c 'the acquire rule plays .*RADIANCE_AQL_ACQ=rule-w)'); NR=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_AQL_ACQ=rule)')
  [ "$NW" -gt 0 ] && [ "$NR" = 0 ] || { log "RESULT xe10 $1: VIRHE rule-w is not what the recorded passes play ($NW rule-w, $NR rule)"; ok=0; }
  # E10: the fused op's band at every EXL3 layer of every rank (both K sets: all 40 layers), served by the device library
  [ "$XN" -gt 0 ] && [ "$EN" = "$XN" ] || { log "RESULT xe10 $1: VIRHE the fused op's band (3..16) at $EN of $XN EXL3 layer band lines"; ok=0; }
  [ "${NX:-0}" -gt 160 ] || { log "RESULT xe10 $1: VIRHE exl3r4d serves ${NX:-no} op bands (int2: 160) -- the fused op is not on the device library: '$ED'"; ok=0; }
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
draftlines() {
  [ "$DRY" = 1 ] && { echo "+ RESULT xe10 $1 drafter lines"; return 0; }
  log "RESULT xe10 $1 drafter / speculation lines: $(docker logs $CN 2>&1 | grep -iE 'dspark|drafter|speculat|draft-window|draft window|draft depth|draft hybrid|verify' | sed 's/^[A-Z][^ ]* //' | sort -u | head -10 | tr '\n' '|' | cut -c1-1600)"
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
  log "RESULT xe10 $1 prefix cache reset: HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"
}
swnote() {   # swnote TAG FIX: GET at start (e10_gd 1, the stack pin's start values); e10_gd 2 refused; A's then B's set applied
  local tag=$1 fx=$2
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (expect e10_gd 1, gg_gemv 2, aql_acq 2, ds41_kvfold / ds41_ofold / ds41_onefill 1, epoch 0); POST e10_gd 2 -> 400; POST A's set $(armset "$fx" "$ARM_A") then B's $(armset "$fx" "$ARM_B"), each + a 4-token request -> applied"; return 0; }
  local s0 s1 s2 c2
  s0=$(fn4get); c2=$(fn4code '{"e10_gd": 2}')
  fn4set "$(armset "$fx" "$ARM_A")"; nudge; s1=$(fn4get)
  fn4set "$(armset "$fx" "$ARM_B")"; nudge; s2=$(fn4get)
  log "RESULT xe10 $tag switch note: at start $(echo "$s0" | cut -c1-700) | e10_gd 2 -> HTTP $c2 | after A's set: $(echo "$s1" | cut -c1-300) | after B's set: $(echo "$s2" | cut -c1-300) | core log: $(docker logs $CN 2>&1 | grep -E 'fn4: switches|fn4: .* refused' | tail -2 | sed 's/^[A-Z] *//' | tr '\n' '|' | cut -c1-400)"
  python3 -c 'import json,sys
s0, s1, s2 = (json.loads(x) for x in sys.argv[1:4]); a, b = json.loads(sys.argv[4]), json.loads(sys.argv[5])
w = s0["switches"]
ok = w.get("e10_gd") == 1 and w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) and s0["epoch"] == 0
ok = ok and all(s1["switches"].get(k) == v for k, v in a.items()) and not s1["pending"] and s1["epoch"] >= 1
ok = ok and all(s2["switches"].get(k) == v for k, v in b.items()) and not s2["pending"] and s2["epoch"] > s1["epoch"]
sys.exit(0 if ok else 1)' "$s0" "$s1" "$s2" "$(armset "$fx" "$ARM_A")" "$(armset "$fx" "$ARM_B")" 2>/dev/null && [ "$c2" = 400 ]
}
blocks() {   # blocks TAG FIX: the A / B blocks (ggblocksdz2.py) and their lines
  run "timeout 5400 python3 -I $GGB --url $URL --mb-dir $D --edit $EP --label $1 --arm 'A=$ARM_A' --arm 'B=$ARM_B' --pair A,B --order $ORDER --dur $DUR --warm 10 --fixed '$2' --out $O/$1-blocks.jsonl > $O/$1-blocks.txt 2>&1"
  [ "$DRY" = 1 ] && return 0
  while IFS= read -r l; do log "RESULT xe10 $1 ${l:0:1500}"; done < <(grep -E '^(BLOCK|GGB \{"error)' $O/$1-blocks.txt)
}
abline() {   # abline TAG DESC: B (e10_gd 1) vs A (e10_gd 0) per metric: A, B, pooled %, each pair, t95 over pairs -> one line
  [ "$DRY" = 1 ] && { echo "+ RESULT xe10 $1 B vs A: per metric A / B / pooled % / pairs / t95 (from the GGB line of $O/$1-blocks.txt)"; return 0; }
  log "RESULT xe10 $1 B (e10_gd 1) vs A (e10_gd 0) [$2; order $ORDER, mb $DUR s, in process]: $(grep '^GGB ' $O/$1-blocks.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=2: "-" if v is None else format(v, ".%df" % n)
keep = ("c1 essay tok/s", "c1 code tok/s", "c1 edit tok/s", "c1 essay ms/step", "c4 aggregate tok/s", "c4 per-stream p50", "c4 ms/step", "c4 tok/step",
        "c8 aggregate tok/s", "c8 per-stream p50", "c8 ms/step", "c8 tok/step")
out = []
for k in keep:
    v = d["metrics"].get(k)
    if not v: continue
    a, b = v["arms"].get("A", {}).get("mean"), v["arms"].get("B", {}).get("mean")
    if a is None and b is None: continue
    p = v.get("pooled")
    out.append("%s A %s B %s (%s %%; pairs %s; t95 +- %s %%)" % (k, f(a), f(b), "-" if p is None else "%+.2f" % (100 * (p - 1)),
               " ".join("%+.2f" % (100 * (x - 1)) for x in v.get("pairs", [])) or "-", "-" if v.get("ci95") is None else "%.2f" % (100 * v["ci95"])))
print("; ".join(out) + " | %d blocks, %d flagged, %s s" % (d["blocks"], d["blocks_bad"], d["wall_s"]))' 2>&1 | cut -c1-3500)"
}
copyt() {   # copyt TAG: the copy test after a ~31k-token cold prompt (105's prompt, offset 0)
  local t0=$(date +%s)
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset 0 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xe10 $1 copy after a cold prompt: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300) | $(thermline $t0 $(date +%s))"
}
gsetc1() { run "(cd $D && timeout 900 python3 $GS $URL $O/$1-gs-c1.jsonl --conc 1 --max-tokens 192 > $O/$1-gs-c1.txt 2>&1)"; }
gcmp() { [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }; [ -s "$1" ] && [ -s "$2" ] || { echo "not compared (missing $1 or $2)"; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
idline() {   # idline LABEL RESULT STRICT: IDENTICAL -> plain; else VIRHE (STRICT 1: an exact lever changed bytes) or reported
  echo "$1 $2" >> $O/identity.txt
  case "$2" in IDENTICAL*|"not compared"*) log "RESULT xe10 identity $1: $2" ;;
    *) if [ "${3:-1}" = 1 ]; then log "RESULT xe10 identity $1: VIRHE $2 -- E10 is exact by construction, so this is a defect"
       else log "RESULT xe10 identity $1: DIFFERENT $2 -- reported, not hidden (a drafting start vs an NST 0 reference)"; fi ;;
  esac
}
ident() {   # ident TAG FIX STRICT_D0: B then A, each after a reset: 28k copy + greedy c1; B vs A strict, each vs bz d0
  local tag=$1 fx=$2 sd=$3
  fn4set "$(armset "$fx" "$ARM_B" '{"epoch_bump": 1}')"; pcreset $tag-iB; copyt $tag-iB; gsetc1 $tag-iB
  [ "$DRY" = 1 ] || log "RESULT xe10 $tag identity B ran under: $(fn4get | cut -c1-600)"
  fn4set "$(armset "$fx" "$ARM_A" '{"epoch_bump": 1}')"; pcreset $tag-iA; copyt $tag-iA; gsetc1 $tag-iA
  [ "$DRY" = 1 ] && { echo "+ identity $tag: 28k copy and greedy set c1, B vs A (strict) and each vs bz d0 (strict $sd)"; return 0; }
  log "RESULT xe10 $tag identity A ran under: $(fn4get | cut -c1-600)"
  idline "$tag 28k copy B (e10_gd 1) vs A (e10_gd 0), both cold" "$(cmpfile $O/copy-$tag-iB-run1.txt $O/copy-$tag-iA-run1.txt)" 1
  idline "$tag greedy set c1 B vs A" "$(gcmp $O/$tag-iA-gs-c1.jsonl $O/$tag-iB-gs-c1.jsonl)" 1
  idline "$tag 28k copy B vs bz d0 (NST 0, stock)" "$(cmpfile $O/copy-$tag-iB-run1.txt $RD/copy-dsd-d0-i1-run1.txt)" $sd
  idline "$tag 28k copy A vs bz d0 (NST 0, stock)" "$(cmpfile $O/copy-$tag-iA-run1.txt $RD/copy-dsd-d0-i1-run1.txt)" $sd
  idline "$tag greedy set c1 B vs bz d0" "$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$tag-iB-gs-c1.jsonl)" $sd
  idline "$tag greedy set c1 A vs bz d0" "$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$tag-iA-gs-c1.jsonl)" $sd
}
estart() {   # estart nst0|nst5: one start, its blocks, identity, its B-vs-A line
  local s=$1 tag=e10-$1 P=$O/e10-$1 rc lim fx desc sd
  e10env; export MODEL=$M225; HP29=19456; best_vars
  case $s in
    nst0) lim=30; fx=$FIX0; desc=$DESC0; sd=1; export MNBT=4160 NST=0 XARGS="$XARGS --live" ;;
    nst5) lim=70; fx=$FIX5; desc=$DESC5; sd=0; export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive" ;;
  esac
  log "RESULT xe10 $tag: start ($desc; RADIANCE_DS41_E10_GD=1) IMAGE=$IMAGE MODEL=$MODEL $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
  mkd $P
  gate $tag $lim; rc=$?
  [ $rc = 1 ] && { log "RESULT xe10 $tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  [ $rc = 2 ] && { log "RESULT xe10 $tag: VIRHE over the gate limit after one restart -> start skipped"; down; return 1; }
  modes $tag || { log "RESULT xe10 $tag: VIRHE the modes are not the stack + E10 line's -> start stopped"; down; return 1; }
  draftlines $tag
  if [ "$DRY" != 1 ]; then
    if [ $s = nst5 ]; then
      docker logs $CN 2>&1 | grep -q "DSpark drafter" || { log "RESULT xe10 $tag: VIRHE no DSpark drafter line"; keeplog $tag-nodraft; down; return 1; }
    else
      docker logs $CN 2>&1 | grep -qiE "DSpark drafter|'dspark' proposes" && { log "RESULT xe10 $tag: VIRHE a DSpark drafter line at NST 0"; keeplog $tag-draft; down; return 1; }
    fi
  fi
  swnote $tag "$fx" || { log "RESULT xe10 $tag: VIRHE the switches are not as expected (e10_gd 1 at start, 2 refused, A / B applied at a step) -> start stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  follow $P
  stopfile || blocks $tag "$fx"
  stopfile || ident $tag "$fx" $sd
  unfollow
  run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
  [ "$DRY" = 1 ] || log "RESULT xe10 $tag engine: $(grep -cE 'fn4: switches .* applied' $O/$tag-engine-full.log) applied switch changes, $(grep -cE 'fn4: .* refused' $O/$tag-engine-full.log) refused, $(grep -cE '^E ' $O/$tag-engine-full.log) E lines"
  down
  abline $tag "$desc"
  log "RESULT xe10 $tag: done"
}
# ---------------------------------------------------------------- P0: one card, no server (XE10_P0=0 skips)
PM=/opt/radiance/bin/perfmoe-e10; X3=/opt/radiance/share/radiance/kernels/exl3r4d.so
GEO_DS="--tp 1 --ep 4 --ep-rank -1 --h-rot 1 --experts 96 --topk 6 --hidden 5120 --width 2304 --pooled 0 --ceiling 640 --warm-ms 500 --rand-data"
P0GD="--chain --chain-band _m16 --gd --iters 10 --reps 3"
pm() {   # pm NAME perfmoe-args... -> $O/p0/NAME.txt; perfmoe's rc
  local nm=$1; shift
  local cn=xe10-p0-$$-$nm
  if [ "$DRY" = 1 ]; then echo "+ docker run (card HIP 0) --entrypoint $PM $IMAGE --exl3 $X3 $* > $O/p0/$nm.txt"; return 0; fi
  timeout -k 15 300 docker run --rm --init --stop-timeout 10 --name $cn --network none --device /dev/kfd --device /dev/dri \
    --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0 --user $(id -u):$(id -g) --group-add $(getent group render | cut -d: -f3) \
    --group-add $(getent group video | cut -d: -f3) -e HOME=/tmp -w /tmp -v $O/p0:/o --entrypoint $PM $IMAGE --exl3 $X3 "$@" > $O/p0/$nm.txt 2>&1
  local rc=$?; docker rm -f $cn > /dev/null 2>&1; return $rc
}
usl() { local v; v=$(grep -m1 "^RESULT $2" $O/p0/$1.txt 2>/dev/null | sed -nE 's/.*M=[0-9]+ +([0-9.]+) us.*/\1/p'); echo ${v:-NA}; }
p0() {   # -> 0, or 1 when a GD-CHECK fails (the item stops)
  local b M nm rc bad=0 chk
  run "mkdir -p $O/p0; chmod 777 $O/p0"
  for b in 2 2.5; do
    for M in 3 4 6 8 12 16; do
      nm=gd-k$b-m$M
      pm $nm $GEO_DS $P0GD --bits $b --limit 10 --M $M; rc=$?
      [ "$DRY" = 1 ] && continue
      chk=$(grep -hE 'GD-CHECK|WARM-CHECK' $O/p0/$nm.txt | sort -u | head -3 | tr '\n' ' ' | cut -c1-240)
      if [ $rc = 0 ] && grep -qE 'GD-CHECK y 0 of [0-9]+ bytes differ' $O/p0/$nm.txt; then
        log "RESULT xe10 P0 K $b M $M limit 10: GD-CHECK PASS ($chk) | split chain $(usl $nm 'exl3r4d chain') us, fused gd $(usl $nm 'exl3r4d gd') us (one card, EP4 rank -1, perfmoe ABAB)"
      else
        bad=1; log "RESULT xe10 P0 K $b M $M limit 10: VIRHE GD-CHECK not passed (rc $rc): ${chk:-no GD-CHECK line} $(grep -vE '^W[0-9]' $O/p0/$nm.txt | grep -iE 'fail|error|abort|refus' | head -2 | tr '\n' ' ' | cut -c1-300)"
      fi
    done
  done
  # the clamp reaches the kernels: --limit 0.25 at M 8 (K 2) must still pass GD-CHECK and its chain y must differ from limit 10's
  pm clamp-l10 $GEO_DS --chain --chain-band _m16 --gd --iters 2 --reps 1 --layers 1 --pools 1 --bits 2 --limit 10 --M 8 --dump-y /o/clamp-l10.y; rc=$?
  pm clamp-l025 $GEO_DS --chain --chain-band _m16 --gd --iters 2 --reps 1 --layers 1 --pools 1 --bits 2 --limit 0.25 --M 8 --dump-y /o/clamp-l025.y
  local rc2=$?
  if [ "$DRY" != 1 ]; then
    local dif="NA"
    [ -s $O/p0/clamp-l10.y ] && [ -s $O/p0/clamp-l025.y ] && { cmp -s $O/p0/clamp-l10.y $O/p0/clamp-l025.y && dif=SAME || dif="DIFFER ($(cmp -l $O/p0/clamp-l10.y $O/p0/clamp-l025.y | wc -l) of $(stat -c %s $O/p0/clamp-l10.y) bytes)"; }
    if [ $rc = 0 ] && [ $rc2 = 0 ] && grep -qE 'GD-CHECK y 0 of' $O/p0/clamp-l025.txt; then
      log "RESULT xe10 P0 clamp (K 2, M 8): limit 0.25 GD-CHECK PASS ($(grep -hE 'GD-CHECK' $O/p0/clamp-l025.txt | head -1 | cut -c1-120)); chain y limit 0.25 vs 10: $dif$([ "$dif" = SAME ] && echo " -- the clamp did not bite at 0.25 (information)")"
    else
      bad=1; log "RESULT xe10 P0 clamp (K 2, M 8): VIRHE limit 10 rc $rc, limit 0.25 rc $rc2: $(grep -hE 'GD-CHECK|fail|error' $O/p0/clamp-l10.txt $O/p0/clamp-l025.txt | head -3 | tr '\n' ' ' | cut -c1-300)"
    fi
  fi
  return $bad
}
# ---------------------------------------------------------------- main
log "RESULT xe10 $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) starts [$STARTS] order $ORDER mb $DUR s | A $ARM_A, B $ARM_B | NST 0 pin $FIX0 | NST 5 pin $FIX5 | yaml $(basename $Y), VRAM env [${VRAMENV:-none}] + RADIANCE_DS41_E10_GD=1 | refs: identity bz d0; fn4 156b (Flash-Next) c4 +5.0 % +- 2.3"
# the image, on this host, read only (in DRYRUN too): the E10 core registry (eighteen switches), the arch's E10 strings, the
# v11 plugin's fused K-generic launch, perfmoe-e10 with --limit, the provenance file
docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xe10: VIRHE image $IMAGE not on this host"; exit 1; }
[ -z "$IMGID" ] || [ "$(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)" = "$IMGID" ] || { log "RESULT xe10: VIRHE image $IMAGE is not $IMGID"; exit 1; }
[ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -qzF "drafter_mx, drafter_bf, e10_gd, epoch_bump" /opt/radiance/bin/radiance && grep -qzF RADIANCE_DS41_E10_GD /opt/radiance/share/radiance/architectures/deepseek41.so && grep -qzF "E10 the EXL3 decode chain as one op" /opt/radiance/share/radiance/architectures/deepseek41.so && grep -qzF RADIANCE_DS41_KVFOLD /opt/radiance/share/radiance/architectures/deepseek41.so && grep -qzF r4d_gg_mr_pin /opt/radiance/share/radiance/kernels/libr4d.so && grep -qzF RADIANCE_DS41_DRAFTER_MX /opt/radiance/share/radiance/kernels/libr4d.so && grep -qzF -- "--limit" /opt/radiance/bin/perfmoe-e10 && grep -qzF "fn4-dsgd" /opt/radiance/share/radiance/e10-PROVENANCE.txt && echo yes' 2>/dev/null)" = yes ] \
  || { log "RESULT xe10: VIRHE $IMAGE is not the ds41-e10 image (registry with e10_gd, deepseek41's E10 strings, K1, D2 / drafter_mx in libr4d, perfmoe-e10 --limit, provenance)"; exit 1; }
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)) is the ds41-e10 image -- OK"
if [ "$DRY" = 1 ]; then   # every helper this body calls resolves (lib-29 / lib-pr266 / heatsoak / here), every file is there
  miss=""
  for f in restart check down evict desc best_vars mkd stopfile log run serve_args thermwin thermnow e10env armset keeplog gate follow unfollow thermline cmpfile identw vrammodes modes draftlines fn4get fn4set nudge fn4code pcreset swnote blocks abline copyt gsetc1 gcmp idline ident estart pm usl p0; do
    [ "$(type -t $f)" = function ] || miss="$miss $f"; done
  echo "+ helpers: ${miss:+MISSING:$miss}${miss:-all resolve}"
  for f in $Y $GS $EP $GGB $D/stepprobe.py $D/mb.py $D/ds41_tests.py $D/fastcheck.py $T/evict-exl3.py $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $WARM_REF_225 $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $C/common.yaml; do
    [ -e $f ] && echo "+ present: $f" || { echo "+ MISSING: $f"; miss="$miss $f"; }; done
  grep -q '^ *RADIANCE_DS41_E10_GD:' $Y && grep -q '^ *RADIANCE_DS41_GG_GEMV:' $Y && grep -q '^ *RADIANCE_AQL_ACQ:' $Y && grep -q '^ *RADIANCE_DS41_KVFOLD:' $Y && grep -q '^ *RADIANCE_PLACE_TURNOVER:' $Y \
    && echo "+ yaml passes RADIANCE_DS41_E10_GD, GG_GEMV, AQL_ACQ, the K1 folds, PLACE_TURNOVER" || { echo "+ yaml $Y does not pass the env"; miss="$miss yaml"; }
  [ -z "$miss" ] || { echo "DRYRUN: FAILED ($miss )"; exit 1; }
else
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xe10: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  [ -e $Y ] && grep -q '^ *RADIANCE_DS41_E10_GD:' $Y && grep -q '^ *RADIANCE_DS41_GG_GEMV:' $Y && grep -q '^ *RADIANCE_AQL_ACQ:' $Y && grep -q '^ *RADIANCE_DS41_KVFOLD:' $Y \
    || { log "RESULT xe10: VIRHE $Y missing or does not pass RADIANCE_DS41_E10_GD / GG_GEMV / AQL_ACQ / the K1 folds"; exit 1; }
  for kv in $VRAMENV; do grep -q "^ *${kv%%=*}:" $Y || { log "RESULT xe10: VIRHE $Y does not pass ${kv%%=*}"; exit 1; }; done
  for f in $GS $EP $GGB $D/stepprobe.py $D/mb.py $D/ds41_tests.py; do [ -e $f ] || { log "RESULT xe10: VIRHE $f missing"; exit 1; }; done
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xe10: VIRHE $M225 missing or not verified"; exit 1; }
  for f in $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl; do [ -s $f ] || log "RESULT xe10: note -- reference $f missing (not compared)"; done
fi
mkd $O $O/guard $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt
if [ "${XE10_P0:-1}" = 1 ]; then
  p0 || { log "RESULT xe10 $ITEM: VIRHE P0's GD-CHECK failed -- no server phases (the fused op is not the split chain's bits on the card)"; exit 1; }
fi
evict
done_starts=""
for s in $STARTS; do stopfile && break; estart $s && done_starts="$done_starts $s"; done
unset $SWITCHES
down
if [ "$DRY" != 1 ]; then
  log "RESULT xe10 summary: $(for s in $done_starts; do grep -h "RESULT xe10 e10-$s B (e10_gd 1) vs A" $L | tail -1 | sed -E 's/^.*RESULT xe10 //' | python3 -c 'import re,sys
l = sys.stdin.read().strip()
m = re.match(r"(e10-\S+) B \(e10_gd 1\) vs A \(e10_gd 0\) \[[^]]*\]: (.*)", l)
if not m: print(l[:300]); sys.exit()
keep = ("c1 essay tok/s", "c1 code tok/s", "c4 aggregate tok/s", "c4 ms/step", "c8 aggregate tok/s", "c8 ms/step")
parts = [p for p in m.group(2).split("; ") if p.startswith(keep)]
print("%s: %s" % (m.group(1), "; ".join(re.sub(r"; pairs [^;]*;", ";", p) for p in parts)))'; done | tr '\n' '|') identity: $(tr '\n' ',' < $O/identity.txt | cut -c1-900)"
fi
log "RESULT xe10 $ITEM: done -- starts [$done_starts ] $O"
