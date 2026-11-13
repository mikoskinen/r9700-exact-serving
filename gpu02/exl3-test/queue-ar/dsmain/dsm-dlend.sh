#!/bin/bash
# dsm-dlend.sh (10.11.2026, queue-ar/dsmain, ds-main-port VRAM agent) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-hczzzz-dlend.sh. Log tag xdl, out runs/xdsm/dlend. Helpers copied from dsm-vramstack.sh (676e2cc4, not edited);
# drivers dlblocks.py (vsblocks.py + ggblocksdz2.py, read only, + lever DL's readiness wait and lend counters a phase) and
# dlsum.py (the summary); yaml ds41-vramstack-tp4.yaml (the stack yaml + env_file: RADIANCE_DRAFT_LEND /
# RADIANCE_DRAFT_PROBE_BACKOFF are not in the stack yaml's list).
#
# LEVER DL, THE DRAFTER'S LEND, IN PROCESS (one start): the drafter's expert slab (96 units, one class) lent to the trunk's
#   experts (~235 trunk slots a rank) while depth 0 holds, taken back when drafting resumes; plus probe back-off. Image
#   radiance-local:ds41-dlend-9461f81e-5c13e555-l (VDL_IMAGE; ds41-dlend 9461f81 = ds41-stack 57f11bd + lever DL 6e02708 +
#   draft_lend -1). EXACT by design (placement only; drafting verified) -- checked here under draft_lend -1 (lent at every
#   step) and under B.
# START: EXL3 2.25, TP4, MML 524288, the t3 env + lib-29 best_vars + MNBT 4160 + --live, DSpark NST 5 --draft-window adaptive;
#   hx-stack's stack env (gg_gemv 2, rule-w, K1 folds declared and on) + RADIANCE_DRAFT_DEPTH0=1 (D7) + RADIANCE_DRAFT_LEND=0
#   (declares lever DL, off) + RADIANCE_DRAFT_PROBE_BACKOFF=0 (+ VDL_VRAMENV, default none: today's D7 placement, the DL agent's
#   base). gate (limit 70 ms) -> modes (t3 / rule-w / K1 notes; the env in the container; DSpark + adaptive depth lines; lever
#   DL: "mover: lever DL: <n> drafter units in <k> class(es) ... lends <N> trunk slots" on all 4 ranks, none "nothing to
#   lend", n / N printed vs the DL agent's 96 / ~235; "draft lend: set up, off"; GET /admin/fn4 lend.capable 1 -- else the
#   start stops, VIRHE) -> each arm's set applied at a step -> warm-up answers vs stock -> /stats at start.
# BLOCKS (dlblocks.py, mb VDL_DUR 75 s, warm 10; fixed pin gg_gemv 2, aql_acq 2, K1 1, draft_adaptive 1, draft_depth0 1,
#   draft_hybrid 0, match 8): A = {draft_lend 0, draft_probe_backoff 0} (today's D7), B = {draft_lend 1, back-off 3},
#   C = {draft_lend 0, back-off 3} (the back-off's own share); order VDL_ORDER (default ABCCBA: each arm's mean position equal,
#   so linear drift cancels in the pooled means; pairs inside ABC | CBA). Each block: a 4-token request applies the switch;
#   with draft_lend 0 a poll (a request between polls, <= 90 s) until lend.state 0 and draft_resident == draft_units (the slab
#   back after a B block); then stepprobe essay,code 512 (c1), essay by draft position, editprobe, mb c4, mb c8; per phase
#   /stats (VRAM units, rank-0 hit, missed a step, stream MB a step), the draft counters (zero / probe cycles, zero moves) and
#   lend (counter deltas, gauges, a 2 s sampler: lent share, lent units max). NO PREFILL PHASE (DL is decode-only).
# IDENTITY: draft_lend -1 (+ requests until lend.state 2), prefix-cache reset, the 28k copy + the 20-prompt greedy set at c1
#   vs bz d0 (lend state read again after: must still be 2); then the same under B; then back to A (reclaim waited).
# SUMMARY (dlsum.py): B vs A and C vs A pooled + pairs (t95) per metric; the depth-0 share, units, hit, stream, lend counters;
#   refs NST 0 + D2 (gz: 62.5 / 64.7 / 131.7 / 170.0), vramsweep T0 (65.1 / 67.2 / 134.0 / 169.5), hcy-depth0b B (57.7 / 90.3
#   / 118.7 / 146.7); targets 1u >= 90, c4 >= 160, c8 >= 200. "RESULT xdl summary: ..." then "RESULT xdl <item>: done".
# ~33 min: evict ~1, start ~5, gate ~2, checks ~1, 6 blocks ~21, identity -1 / B ~4, switch back ~0.5.
# NO cc_sim (user 10.11.), no 2.9. No power / host / fan settings; production compose files read only (--project-directory).
# VDL_IMAGE, VDL_ITEM, VDL_ORDER, VDL_DUR, VDL_VRAMENV (NAME=VALUE words at start, e.g. RADIANCE_PLACE_TURNOVER=0).
# DRYRUN=1 bash dsm-dlend.sh : prints every command; checks the image (docker image inspect + a no-device, no-network grep of
#   its files) and the yaml (docker compose config --quiet); no server, no GPU, nothing written.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-vramstack-tp4.yaml
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=$R/xdsm/dlend; export STATSDIR=$O/stats
RD=$R/xdsm/draft                                # bz's d0 files (identity reference: copy + greedy set)
export IMAGE=${VDL_IMAGE:-radiance-local:ds41-dlend-9461f81e-5c13e555-l}
ITEM=${VDL_ITEM:-57-p4zyzzzzz-ds-hczzzz-dlend}
DUR=${VDL_DUR:-75}; ORDER=${VDL_ORDER:-ABCCBA}; VRAMENV=${VDL_VRAMENV:-}; MATCH=8
case $ORDER in *[!ABC]*|"") echo "VDL_ORDER takes A / B / C blocks (got '$ORDER')"; exit 1 ;; esac
case $ORDER in *A*) ;; *) echo "VDL_ORDER needs arm A (the reference)"; exit 1 ;; esac
for kv in $VRAMENV; do case "$kv" in [A-Z]*=*) ;; *) echo "VDL_VRAMENV takes NAME=VALUE words (got '$kv')"; exit 1 ;; esac; done
ARM_A='{"draft_lend":0,"draft_probe_backoff":0}'
ARM_B='{"draft_lend":1,"draft_probe_backoff":3}'
ARM_C='{"draft_lend":0,"draft_probe_backoff":3}'
ARM_I='{"draft_lend":-1,"draft_probe_backoff":0}'
FIX='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": '"$MATCH"', "draft_hybrid": 0}'
DLENV="RADIANCE_DRAFT_DEPTH0=1 RADIANCE_DRAFT_LEND=0 RADIANCE_DRAFT_PROBE_BACKOFF=0 $VRAMENV"
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
EP=$HD/editprobe.py; DLB=$HD/dlblocks.py; VSB=$HD/vsblocks.py; GGB=$HD/ggblocksdz2.py; DLS=$HD/dlsum.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DS41_ARGATHER_MAX RADIANCE_DS41_SHEXP_LANE RADIANCE_DEBUG_MISS_LOG
  RADIANCE_DS41_KVFOLD RADIANCE_DS41_OFOLD RADIANCE_DS41_ONEFILL RADIANCE_DS41_PIPE
  RADIANCE_DRAFT_DEPTH0 RADIANCE_DRAFT_PROBE RADIANCE_DRAFT_HYBRID RADIANCE_DRAFT_HYBRID_MATCH
  RADIANCE_DRAFT_LEND RADIANCE_DRAFT_PROBE_BACKOFF"
stackenv() {   # hx-stack's stack env + lever DL's start env (+ VDL_VRAMENV), every other switch unset
  local kv
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  for kv in $DLENV; do export "$kv"; done
}
armset() {   # armset ARMJSON [EXTRAJSON]: FIX + the arm + extra as one POST body
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update(json.loads(sys.argv[2])); d.update(json.loads(sys.argv[3])); print(json.dumps(d))' "$FIX" "$1" "${2:-{\}}"
}
# ---------------------------------------------------------------- helpers (dsm-vramstack.sh, copied; tag xdl)
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
  log "RESULT xdl guard $tag: $st, 1-user ${ms} ms/step (limit $lim; NST 5 DSpark starts depth0b / hybrid 35.2-35.3)"
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xdl $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xdl $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  echo "warm-up-$1 $r" >> $O/identity.txt
  case $r in IDENTICAL*) log "RESULT xdl $1 warm-up answers vs stock ($(basename $2)): $r" ;;
    *) log "RESULT xdl $1 warm-up answers vs stock ($(basename $2)): VIRHE $r -- NOT EXACT vs the tuned3l line" ;; esac
}
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
lendget() {   # lendget KEY: GET /admin/fn4 lend.KEY (empty when absent)
  fn4get | python3 -c 'import json,sys
try: print(json.loads(sys.stdin.read()).get("lend", {}).get(sys.argv[1], ""))
except Exception: print("")' $1 2>/dev/null
}
lendwait() {   # lendwait STATE [RESIDENT_FULL]: requests until lend.state == STATE (and draft_resident == draft_units), <= 90 s
  [ "$DRY" = 1 ] && { echo "+ 4-token requests until lend.state == $1$([ "${2:-}" = full ] && echo " and draft_resident == draft_units") (<= 90 s)"; return 0; }
  local t0=$(date +%s) s
  while :; do
    nudge
    s=$(fn4get | python3 -c 'import json,sys
L = json.loads(sys.stdin.read()).get("lend", {})
print("ok" if L.get("state") == int(sys.argv[1]) and (sys.argv[2] != "full" or L.get("draft_resident") == L.get("draft_units")) else "no")' $1 "${2:-}" 2>/dev/null)
    [ "$s" = ok ] && return 0
    [ $(( $(date +%s) - t0 )) -gt 90 ] && return 1
    sleep 0.5
  done
}
pcreset() {   # pcreset TAG: POST /reset_prefix_cache (loopback operator) so the next prompt is prefilled cold
  [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o $O/$1-reset.json -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  log "RESULT xdl $1 prefix cache reset: HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"
}
copyt() {   # copyt TAG OFFSET: the copy test after a ~31k-token cold prompt (105's prompt: same offset -> same text)
  local t0=$(date +%s)
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset $2 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xdl $1 copy after a cold prompt: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300) | $(thermline $t0 $(date +%s))"
}
gsetc1() { run "(cd $D && timeout 900 python3 $GS $URL $O/$1-gs-c1.jsonl --conc 1 --max-tokens 192 > $O/$1-gs-c1.txt 2>&1)"; }
gcmp() { [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }; [ -s "$1" ] && [ -s "$2" ] || { echo "not compared (missing $1 or $2)"; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
idline() {   # idline KEY LABEL RESULT: identity.txt + one RESULT line (a difference is reported, not hidden)
  echo "$1 $3" >> $O/identity.txt
  case "$3" in IDENTICAL*|"not compared"*) log "RESULT xdl identity $2: $3" ;;
    *) log "RESULT xdl identity $2: VIRHE $3 -- lever DL is exact by design (placement only), so this is a defect" ;; esac
}
ident() {   # ident TAG ARMNAME ARMJSON STATE: the arm's set, requests until lend.state STATE (-1: lent), a reset, copy + greedy c1 vs bz d0
  local tag=$1-i$2 st
  fn4set "$(armset "$3" '{"epoch_bump": 1}')"
  [ -n "$4" ] && { lendwait $4 || log "RESULT xdl $tag: VIRHE lend.state never reached $4 within 90 s -- identity runs anyway, read the state line"; }
  pcreset $tag
  copyt $tag 0; gsetc1 $tag
  [ "$DRY" = 1 ] && { echo "+ identity $tag (arm $2): 28k copy vs bz d0, greedy set c1 vs bz d0; lend state after"; return 0; }
  st=$(fn4get)
  log "RESULT xdl $tag identity ran under: $(echo "$st" | cut -c1-900)"
  [ -n "$4" ] && [ "$(lendget state)" != "$4" ] && log "RESULT xdl $tag: note -- lend.state $(lendget state) after the identity runs (wanted $4 throughout)"
  idline copy-$tag "$tag 28k copy vs bz d0 (NST 0, stock)" "$(cmpfile $O/copy-$tag-run1.txt $RD/copy-dsd-d0-i1-run1.txt)"
  idline gs-$tag "$tag greedy set c1 vs bz d0" "$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$tag-gs-c1.jsonl)"
}
modes() {   # modes TAG: hx-stack's t3 + rule-w + K1 checks, the env in the container, the DSpark lines, lever DL's start lines -> 0, else 1
  local tag=$1
  [ "$DRY" = 1 ] && { echo "+ modes $tag: indexer note, ARGATHER 4, no shared-expert-lane note, staging line, rule-w, K1 three declared at 1; env [$DLENV] in the container env; DSpark drafter + adaptive depth lines; 4 x 'mover: lever DL: <n> drafter units in <k> class(es) ... lends <N> trunk slots' (none 'nothing to lend'; DL agent: 96 / 1 / ~235), 'draft lend: set up, off'"; return 0; }
  local MLF SL AL XL KL NW NR CE kv bad="" DL DN ok=1
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in $DLENV; do echo "$CE" | grep -qxF "$kv" || bad="$bad ${kv%%=*}"; done
  [ -z "$bad" ] || { log "RESULT xdl $tag: VIRHE not in the container's env:$bad"; ok=0; }
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  AL=$(docker logs $CN 2>&1 | grep -m1 'the MoE gather and its all-reduce as one kernel' | sed 's/^radiance: //' | cut -c1-300)
  XL=$(docker logs $CN 2>&1 | grep -m1 'the shared expert on the second lane' | sed 's/^radiance: //' | cut -c1-300)
  KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-500)
  DL=$(docker logs $CN 2>&1 | grep -E '^I\[[0-3]\] mover: lever DL: ' | sed -E 's/^I\[([0-9])\] mover: lever DL: ([0-9]+) drafter units in ([0-9]+) class.*lends ([0-9]+) trunk slots.*/r\1 \2 units \3 class \4 slots/' | tr '\n' ',' | sed 's/,$//')
  DN=$(docker logs $CN 2>&1 | grep -m1 -oE 'draft lend: [^;]*' | cut -c1-120)
  log "RESULT xdl $tag modes: $(echo "$MLF" | cut -c1-500) | staging: ${SL:-none} | argather: ${AL:-none} | K1: ${KL:-none} | env [$DLENV] | lever DL: ${DL:-no mover line} | ${DN:-no 'draft lend' line} | mover: $(docker logs $CN 2>&1 | grep -E '^I\[[0-9]\] mover: [0-9]+ slab slots' | sed -E 's/^I\[([0-9])\] mover: ([0-9]+) slab slots \(([0-9]+) shadow, ([0-9]+) more kept free.*/r\1 \3 shadow \4 free/' | tr '\n' ',' | sed 's/,$//') | prefetch on at $(docker logs $CN 2>&1 | grep -cE '^I rank [0-9]+: expert prefetch on') ranks"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate" "post-collapse-norm separate at 0 layers" "next collapse-norm separate at 0 layers"; do
    case "$MLF" in *"$w"*) ;; *) log "RESULT xdl $tag: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  case "$AL" in *"up to 4 tokens a step (RADIANCE_DS41_ARGATHER)"*) ;; *) log "RESULT xdl $tag: VIRHE the ARGATHER note is not today's 4-row one: '${AL}'"; ok=0 ;; esac
  [ -z "$XL" ] || { log "RESULT xdl $tag: VIRHE a shared-expert-lane note: '$XL'"; ok=0; }
  [ -n "$SL" ] || { log "RESULT xdl $tag: VIRHE no prefill staging line (levels refused?)"; ok=0; }
  case "$KL" in *"in one launch at 0 of"*|*"rope+round trip at 0 ("*|*"not declared at"*|*"at 0 MoE layers"*|"")
    log "RESULT xdl $tag: VIRHE the lever K1 note does not show all three folds declared: '${KL}'"; ok=0 ;; esac
  case "$KL" in *"ds41_kvfold=1"*"ds41_ofold=1"*"ds41_onefill=1"*) ;; *) log "RESULT xdl $tag: VIRHE the K1 switches do not start at 1: '${KL}'"; ok=0 ;; esac
  NW=$(docker logs $CN 2>&1 | grep -c 'the acquire rule plays .*RADIANCE_AQL_ACQ=rule-w)'); NR=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_AQL_ACQ=rule)')
  [ "$NW" -gt 0 ] && [ "$NR" = 0 ] || { log "RESULT xdl $tag: VIRHE rule-w is not what the recorded passes play ($NW rule-w, $NR rule)"; ok=0; }
  docker logs $CN 2>&1 | grep -q "DSpark drafter" && docker logs $CN 2>&1 | grep -q "draft depth: adaptive" || { log "RESULT xdl $tag: VIRHE no DSpark drafter / adaptive depth line"; ok=0; }
  [ "$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: lever DL: [0-9]+ drafter units')" = 4 ] || { log "RESULT xdl $tag: VIRHE no lever DL mover line on all 4 ranks: ${DL:-none}"; ok=0; }
  docker logs $CN 2>&1 | grep -qE 'mover: lever DL: .*nothing to lend' && { log "RESULT xdl $tag: VIRHE a rank has nothing to lend"; ok=0; }
  docker logs $CN 2>&1 | grep -q 'draft lend: set up, off' || { log "RESULT xdl $tag: VIRHE no 'draft lend: set up, off' line"; ok=0; }
  echo "$DL" | tr ',' '\n' | awk '{if ($2 != 96 || $4 != 1 || $6 < 200 || $6 > 270) bad = 1} END {exit bad ? 0 : 1}' \
    && log "RESULT xdl $tag: note -- lever DL's units / classes / slots ($DL) differ from the DL agent's 96 / 1 / ~235 (not a stop)"
  [ $ok = 1 ] || keeplog $tag-nomode
  [ $ok = 1 ]
}
swnote() {   # swnote TAG: GET at start (the stack's start values, draft_depth0 1, lend capable 1, draft_lend 0); each arm's set applied at a step
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (expect gg_gemv 2, aql_acq 2, K1 1, draft_adaptive 1, draft_depth0 1, draft_hybrid 0, draft_lend 0, draft_probe_backoff 0, draft_capable 1, lend.capable 1, epoch 0); per arm POST + a 4-token request -> applied: A $(armset "$ARM_A") B $(armset "$ARM_B") C $(armset "$ARM_C"); back to A + the reclaim wait"; return 0; }
  local s0 s j a bad=""
  s0=$(fn4get)
  python3 -c 'import json,sys
s0 = json.loads(sys.argv[1]); w = s0["switches"]; L = s0.get("lend") or {}
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) and s0["epoch"] == 0
ok = ok and s0["draft_capable"] == 1 and w["draft_adaptive"] == 1 and w["draft_depth0"] == 1 and w["draft_hybrid"] == 0
ok = ok and w.get("draft_lend") == 0 and w.get("draft_probe_backoff") == 0 and L.get("capable") == 1
sys.exit(0 if ok else 1)' "$s0" 2>/dev/null || bad="at start"
  for a in "$ARM_B" "$ARM_C" "$ARM_A"; do
    j=$(armset "$a" '{"epoch_bump": 1}'); fn4set "$j"; nudge; s=$(fn4get)
    python3 -c 'import json,sys
s, want = json.loads(sys.argv[1]), json.loads(sys.argv[2]); want.pop("epoch_bump", None)
sys.exit(0 if all(s["switches"].get(k) == v for k, v in want.items()) and not s["pending"] and s["epoch"] >= 1 else 1)' "$s" "$j" 2>/dev/null || bad="$bad arm $a"
  done
  lendwait 0 full || bad="$bad (no reclaim to state 0 after the arm checks)"
  log "RESULT xdl $1 switch note: at start $s0 | after B, C, A: $(fn4get | cut -c1-900)$([ -n "$bad" ] && echo " -- VIRHE not as expected:$bad")"
  [ -z "$bad" ]
}
blocks() {   # blocks TAG: dlblocks.py, its BLOCK lines, the readiness and lend lines
  run "timeout 5400 python3 -I $DLB --url $URL --mb-dir $D --edit $EP --label $1 --arm 'A=$ARM_A' --arm 'B=$ARM_B' --arm 'C=$ARM_C' --order $ORDER --dur $DUR --warm 10 --fixed '$FIX' --ready-s 90 --sample-s 2 --out $O/$1-blocks.jsonl > $O/$1-blocks.txt 2>&1"
  [ "$DRY" = 1 ] && return 0
  while IFS= read -r l; do log "RESULT xdl $1 ${l:0:1500}"; done < <(grep -E '^(BLOCK|GGB \{"error)' $O/$1-blocks.txt)
  log "RESULT xdl $1 readiness a block: $(grep '^VDLR ' $O/$1-blocks.txt | cut -c6- | python3 -c 'import json,sys
print("; ".join("b%s lend %s: %s in %s s (state %s, resident %s / %s)" % (d.get("block"), d.get("draft_lend"), "ready" if d.get("ok") else "NOT READY", d.get("wait_s"),
      d.get("state"), d.get("draft_resident"), d.get("draft_units")) for d in (json.loads(l) for l in sys.stdin if l.strip())))' 2>&1 | cut -c1-1500)"
  log "RESULT xdl $1 lend a phase (c4 / c8: lent share, lent units max, lends, reclaims, transit ms; vram units, hit, stream MB a step): $(paste -d'|' <(grep '^VDL ' $O/$1-blocks.txt | cut -c5-) <(grep '^VSX ' $O/$1-blocks.txt | cut -c5-) | python3 -c 'import json,sys
out = []
for l in sys.stdin:
    a, b = l.rstrip("\n").split("|", 1)
    try: d, x = json.loads(a), json.loads(b)
    except ValueError: continue
    if d.get("phase") not in ("c4", "c8"): continue
    out.append("b%s %s: lent %s, max %s u, lends %s, reclaims %s, transit %s ms; %s u, hit %s, stream %s" % (d.get("block"), d.get("phase"), d.get("s_lent_share"),
               d.get("s_lent_units_max"), d.get("lends"), d.get("reclaims"), d.get("transit_ms"), x.get("vram_units_end"), x.get("hit_pct_rank0"), x.get("stream_MB_step")))
print("; ".join(out))' 2>&1 | cut -c1-2400)"
}
# ---------------------------------------------------------------- main
IID=$(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null)
[ -n "$IID" ] || { log "RESULT xdl: VIRHE image $IMAGE not on this host"; exit 1; }
log "RESULT xdl $ITEM: start IMAGE=$IMAGE (${IID:7:12}) | lever DL in process, one NST 5 start: A $ARM_A (today's D7), B $ARM_B, C $ARM_C, order $ORDER, mb $DUR s, fixed $FIX | start env [$DLENV] | yaml $(basename $Y) | refs NST 0 + D2 gz (62.5 / 64.7 / 131.7 / 170.0), vramsweep T0 (c4 134.0, c8 169.5), hcy-depth0b B (57.7 / 90.3 / 118.7 / 146.7); identity bz d0 under draft_lend -1 and B | targets 1u >= 90, c4 >= 160, c8 >= 200"
# the image, read only (DRYRUN too): the stack (depth 0 / hybrid registry, reset_prefix_cache, K1, D2) + lever DL
[ "$(timeout 180 docker run --rm --network none --entrypoint sh $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels
for l in draft_depth0 draft_hybrid_match reset_prefix_cache draft_probe_backoff draft_lend RADIANCE_DRAFT_LEND RADIANCE_DRAFT_PROBE_BACKOFF "mover: lever DL: " "draft lend: "; do grep -qzF -- "$l" $B || exit 1; done
grep -qzF RADIANCE_DS41_KVFOLD $A && grep -qzF rmsnorm_rope_roundtrip_kv_store $K/libr4d.so && grep -qzF r4d_gg_mr_pin $K/libr4d.so && echo yes' 2>/dev/null)" = yes ] \
  || { log "RESULT xdl: VIRHE $IMAGE is not the stack + lever DL image (registry, reset_prefix_cache, K1, D2, draft_lend / draft_probe_backoff, the DL log lines)"; exit 1; }
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xdl: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  grep -q 'VSL_ENVFILE' $Y && grep -q '^ *RADIANCE_DRAFT_DEPTH0:' $Y && grep -q '^ *RADIANCE_DS41_GG_GEMV:' $Y || { log "RESULT xdl: VIRHE $Y lacks the env_file entry or the stack env"; exit 1; }
  for f in $GS $EP $DLB $VSB $GGB $DLS $D/stepprobe.py $D/mb.py $D/ds41_tests.py; do [ -e $f ] || { log "RESULT xdl: VIRHE $f missing"; exit 1; }; done
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xdl: VIRHE $M225 missing or not verified"; exit 1; }
  for f in $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl; do [ -s $f ] || log "RESULT xdl: note -- reference $f missing (not compared)"; done
else
  for f in $Y $GS $EP $DLB $VSB $GGB $DLS $D/stepprobe.py $D/mb.py $D/ds41_tests.py $MODELDIR/$M225 $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl; do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  echo "+ image $IMAGE (${IID:7:12}): the stack + lever DL -- OK"
fi
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x VSL_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || { log "RESULT xdl: VIRHE docker compose config refuses $Y"; exit 1; }
[ "$DRY" = 1 ] && echo "+ yaml $Y: docker compose config OK"
mkd $O $O/guard $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt
evict
tag=dl-n5; P=$O/$tag
stackenv; export MODEL=$M225; HP29=19456; best_vars
export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive"
export VSL_ENVFILE=$O/$tag.env
log "RESULT xdl $tag: start IMAGE=$IMAGE MODEL=$MODEL $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')| env file $VSL_ENVFILE [$DLENV]"
mkd $P
if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $DLENV > $VSL_ENVFILE"; else printf '%s\n' $DLENV > $VSL_ENVFILE; fi
gate $tag 70; rc=$?
[ $rc = 1 ] && { log "RESULT xdl $tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; exit 1; }
[ $rc = 2 ] && { log "RESULT xdl $tag: VIRHE over the gate limit after one restart -> item stopped"; exit 1; }
modes $tag || { log "RESULT xdl $tag: VIRHE the modes / lever DL start lines are not as expected -> item stopped"; exit 1; }
swnote $tag || { log "RESULT xdl $tag: VIRHE the switches / lend are not as expected (lend.capable must be 1) -> item stopped"; keeplog $tag-switch; exit 1; }
identw $tag $WARM_REF_225
run "curl -s -m 15 $URL/stats > $O/$tag-stats0.json 2>/dev/null"
follow $P
stopfile || blocks $tag
# identity: lent at every step (-1), then B, then back to A with the slab taken back
stopfile || ident $tag I "$ARM_I" 2
stopfile || ident $tag B "$ARM_B" ""
fn4set "$(armset "$ARM_A" '{"epoch_bump": 1}')"; lendwait 0 full || log "RESULT xdl $tag: note -- the slab was not back (state 0, all resident) within 90 s after the identity runs"
unfollow
run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
down
if [ "$DRY" = 1 ]; then
  echo "+ python3 $DLS $O/$tag-blocks.jsonl $O/identity.txt > $O/summary.txt -> RESULT xdl ARM / CMP (B vs A, C vs A: pooled + pairs, t95) / REF lines, RESULT xdl summary"
else
  python3 $DLS $O/$tag-blocks.jsonl $O/identity.txt > $O/summary.txt 2>&1
  while IFS= read -r l; do case "$l" in SUMMARY*) ;; *) log "RESULT xdl ${l:0:3000}" ;; esac; done < $O/summary.txt
  log "RESULT xdl summary: $(grep -m1 '^SUMMARY ' $O/summary.txt | cut -c9- | cut -c1-4500)"
fi
unset $SWITCHES
log "RESULT xdl $ITEM: done -- $O"
