#!/bin/bash
# dsm-dlw.sh (11.11.2026, queue-ar/dsmain/dlw, ds-main-port DLW agent) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-iel-dlw.sh. Tag "RESULT xdlw ", out runs/xdsm/dlw (DLW_OUT). LEVER DLW, A LIGHTER DSPARK DRAFTER: the drafter's
# dense linears (fc + per stage q_a / q_b / kv / o_a groups / o_b / shared gate_up / down; plain bf16 in the container, ~0.43 GiB
# a rank) served from a WEIGHT SIDECAR (RADIANCE_WEIGHT_SIDECAR, ds41-dlw df86048) as llama.cpp q8_0 (B) or q6_K (C, the trunk's
# own linears' form) -- gemm_nt_ggml over them, the same libr4d forms the trunk's q6_K linears run. Only proposals change; verify
# keeps the output exact (identity on B and C). Saves per rank ~196 MiB (q8_0) / ~248 MiB (q6_K) of static drafter weights
# (= ~21 / ~27 expert units at depth 0) and that share of the query pass's / context pass's weight reads.
# IMAGE radiance-local:ds41-dlw-<sha8>-5c13e555-l (= the ds41-int3 1116248a image + bin/radiance + bin/rad-sidecar; sidecar
# unset = int3, hostcheck: graph dump IDENTICAL, start log 0 differing lines). Sidecars beside the .rad (MODELDIR, /models in
# the container): dspark-dlw-q8_0.rad / dspark-dlw-q6_K.rad, written by rad-sidecar with build/dlw/dlw-<type>.recipe.
# CROSS-START (the sidecar is start-time), FOUR NST 5 starts with hxzzy-stack2's env AS IT RAN (dsm-sam.sh's stackenv: HYB=1,
# lend 1 + backoff 3, the VRAM levers, drafter_mx 1, walk 1, the small n5 set) and its pin, order A B C A:
#   A = no sidecar (today's drafter, bf16 linears); B = q8_0 sidecar; C = q6_K sidecar.
# Per start: restart + check + modes (the start words in the container; the sidecar's start-log + load lines under B / C, none
# under A) + /stats at start (routed units in VRAM, the lend) -> one block of z0blocks.py (copy, md5 f7a22af5: stepprobe essay,code
# 512 at c1 temperature 0, essay acceptance by position, editprobe, mb c4 / c8 natural --dur 75 --warm 10; per phase /stats
# experts (VRAM units, hit, stream) + the lend (lent units) + the draft counters (zero cycles = depth 0)) -> daprobe.py (copy, md5
# 550550f3) at the SERVER-DEFAULT sampling (no temperature sent; per-request seed 1234; essay / code / edit, 2 runs) -> under B
# and C: identity at temperature 0 (28k copy + 20-prompt greedy set c1 vs bz d0) -> down.
# Summary (dlwsum.py): per arm and phase the numbers, B / A and C / A with A = mean of the two A starts, A3 / A0 (drift).
# ~75 min (evict 1; a start ~5 + check ~1.5 + block ~7.5 + sampled ~3, identity ~2.5 under B / C).
# NO cc_sim, no 2.9; no power / host / fan settings; production compose files read only (common.yaml via extends).
# DRYRUN=1 bash dsm-dlw.sh : prints every command; checks the image (its literals), every helper (files present and parsing,
#   functions defined, the python helpers' --help), the sidecars (present, sha256), the yaml (docker compose config); no server,
#   no GPU, nothing written. A missing helper fails it (exit 1).
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
Y=$HD/../ds41-stack2-tp4.yaml
O=${DLW_OUT:-$R/xdsm/dlw}; export STATSDIR=$O/stats
RD=$R/xdsm/draft
export IMAGE=${DLW_IMAGE:?DLW_IMAGE}
ITEM=${DLW_ITEM:-57-p4zyzzzzz-ds-iel-dlw}
ORDER=${DLW_ORDER:-ABCA}
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
PRB=$HD/daprobe.py; GGB=$HD/z0blocks.py; EP=$HD/editprobe.py; SUM=$HD/dlwsum.py; SEED=1234; DUR=75
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
SIDECAR=dspark-conf.f32
SQ8=dspark-dlw-q8_0.rad; SQ6=dspark-dlw-q6_K.rad
SHA_Q8=${DLW_SHA_Q8:?DLW_SHA_Q8}; SHA_Q6=${DLW_SHA_Q6:?DLW_SHA_Q6}
GPTAB=q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": 8, "drafter_mx": 1, "gg_pf": 1, "f32_pf": 25, "dspark_walk": 1, "ds41_argather": 48, "idx_split": 1, "ds41_qnfold": 1, "ds41_swfold": 1, "draft_ctxskip": 1, "dspark_conf": 1, "dspark_conf_rows": 1, "draft_hybrid": 1, "draft_lend": 1, "draft_probe_backoff": 3}'
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DS41_ARGATHER_MAX RADIANCE_DS41_SHEXP_LANE RADIANCE_DEBUG_MISS_LOG
  RADIANCE_DS41_KVFOLD RADIANCE_DS41_OFOLD RADIANCE_DS41_ONEFILL RADIANCE_DS41_PIPE
  RADIANCE_DRAFT_DEPTH0 RADIANCE_DRAFT_PROBE RADIANCE_DRAFT_HYBRID RADIANCE_DRAFT_HYBRID_MATCH
  RADIANCE_DRAFT_LEND RADIANCE_DRAFT_PROBE_BACKOFF RADIANCE_DRAFT_LEND_BORROW RADIANCE_DEBUG_VRAM
  RADIANCE_DS41_IDX_BALANCE RADIANCE_DS41_QNFOLD RADIANCE_DS41_SWFOLD RADIANCE_DRAFT_CTXSKIP RADIANCE_DS41_EMB_RAM
  RADIANCE_DS41_DSPARK_CONF RADIANCE_DS41_DSPARK_ALT RADIANCE_DS41_KVWIN_FP8 RADIANCE_DS41_DRAFTER_MX RADIANCE_DS41_DRAFTER_BF
  RADIANCE_DS41_GG_PF RADIANCE_DS41_GG_PF_TABLE RADIANCE_DS41_F32_PF RADIANCE_DS41_NVGATHER RADIANCE_DS41_DSPARK_WALK
  RADIANCE_DS41_ARENA_DECODE_ROWS RADIANCE_DS41_ROUTER_BF16 RADIANCE_PLACE_SHADOW_SLOTS RADIANCE_XPREFETCH
  RADIANCE_WEIGHT_OVERLAY RADIANCE_DS41_DA_TAPS RADIANCE_KLD_DUMP RADIANCE_WEIGHT_SIDECAR"
xlog() { log "RESULT xdlw $*"; }
# stackenv SIDE: dsm-sam.sh's stackenv n5 (hxzzy-stack2 as it ran: VRAM levers all, lend declared, walk 1, the small n5 set,
# turnover 0) + RADIANCE_WEIGHT_SIDECAR=/models/SIDE when SIDE is given; the start words also go to $XS2_ENVFILE
stackenv() {
  local kv w
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  export RADIANCE_PLACE_TURNOVER=0
  w="RADIANCE_DS41_GG_PF=1 RADIANCE_DS41_GG_PF_TABLE=$GPTAB RADIANCE_DS41_F32_PF=25"
  w="$w RADIANCE_DS41_ROUTER_BF16=1 RADIANCE_DS41_EMB_RAM=1 RADIANCE_PLACE_SHADOW_SLOTS=16 RADIANCE_XPREFETCH=0"
  w="$w RADIANCE_DS41_ARENA_DECODE_ROWS=48 RADIANCE_DS41_DSPARK_WALK=1 RADIANCE_DS41_ARGATHER_MAX=48 RADIANCE_DS41_QNFOLD=1 RADIANCE_DS41_SWFOLD=1"
  w="$w RADIANCE_DRAFT_CTXSKIP=0 RADIANCE_DS41_DSPARK_CONF=/models/$SIDECAR RADIANCE_DRAFT_LEND=0 RADIANCE_DRAFT_PROBE_BACKOFF=0 RADIANCE_DRAFT_LEND_BORROW=0"
  [ -n "${1:-}" ] && w="$w RADIANCE_WEIGHT_SIDECAR=/models/$1"
  for kv in $w; do export "$kv"; done
  LEVENV=$w
}
sideof() { case $1 in B) echo $SQ8 ;; C) echo $SQ6 ;; *) echo "" ;; esac; }
writeenv() { [ "$DRY" = 1 ] && { echo "+ printf '%s\\n' $LEVENV > $1"; return 0; }; printf '%s\n' $LEVENV > $1; }
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
pinset() {   # pinset TAG JSON...: the merged switches + epoch_bump, applied at a step (epoch moved, nothing pending)
  local tag=$1 body s0 s1; shift
  body=$(python3 -c 'import json,sys
d={}
for a in sys.argv[1:]: d.update(json.loads(a))
d["epoch_bump"]=1; print(json.dumps(d))' "$@")
  [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $body; a 4-token request; GET -> applied, nothing pending"; return 0; }
  s0=$(fn4get)
  curl -s -m 10 -X POST -H 'Content-Type: application/json' -d "$body" $URL/admin/fn4 > /dev/null
  nudge; s1=$(fn4get)
  python3 -c 'import json,sys
s0, s1, w = (json.loads(x) for x in sys.argv[1:4])
w.pop("epoch_bump", None)
ok = all(s1["switches"].get(k) == v for k, v in w.items()) and not s1["pending"] and s1["epoch"] > s0["epoch"]
bad = [k for k, v in w.items() if s1["switches"].get(k) != v]
print("applied" if ok else "NOT applied: %s" % bad); sys.exit(0 if ok else 1)' "$s0" "$s1" "$body" > $O/pin-$tag.txt 2>&1
  local rc=$?
  xlog "$tag pin: $(cat $O/pin-$tag.txt) -- $(echo "$s1" | cut -c1-300)"
  return $rc
}
modes() {   # modes TAG ARM: the start words in the container, the W1 line, the sidecar's lines (B / C) or none (A)
  local side; side=$(sideof $2)
  [ "$DRY" = 1 ] && { echo "+ modes $1: the start words in the container, the W1 declared line over 4 ranks, $([ -n "$side" ] && echo "RADIANCE_WEIGHT_SIDECAR=/models/$side + 'weight sidecar: 43 entries' + 'read from the weight sidecar'" || echo "no sidecar word / line")"; return 0; }
  local CE ok=1 kv WL SL LL
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in RADIANCE_PLACE_TURNOVER=0 RADIANCE_DS41_GG_GEMV=2 RADIANCE_DS41_ROUTER_BF16=1 RADIANCE_DS41_DSPARK_WALK=1 RADIANCE_DRAFT_LEND=0 \
            $([ -n "$side" ] && echo RADIANCE_WEIGHT_SIDECAR=/models/$side); do
    echo "$CE" | grep -qxF "$kv" || { xlog "$1: VIRHE $kv not in the container"; ok=0; }; done
  [ -z "$side" ] && echo "$CE" | grep -q '^RADIANCE_WEIGHT_SIDECAR=' && { xlog "$1: VIRHE RADIANCE_WEIGHT_SIDECAR set under A"; ok=0; }
  WL=$(docker logs $CN 2>&1 | grep -m1 'DSpark lever W1.*fn4 dspark_walk=' | sed 's/^radiance: //' | cut -c1-200)
  case "$WL" in *"over 4 rank(s)"*"dspark_walk=1 "*) ;; *) xlog "$1: VIRHE no W1 declared line over 4 ranks"; ok=0 ;; esac
  SL=$(docker logs $CN 2>&1 | grep -m1 'weight sidecar: ' | sed 's/^radiance: //' | cut -c1-400)
  LL=$(docker logs $CN 2>&1 | grep -m1 'read from the weight sidecar' | sed 's/^radiance: //' | cut -c1-300)
  if [ -n "$side" ]; then
    case "$SL" in *"43 entries"*) ;; *) xlog "$1: VIRHE no 'weight sidecar: 43 entries' start line"; ok=0 ;; esac
    [ -n "$LL" ] || { xlog "$1: VIRHE no 'read from the weight sidecar' load line"; ok=0; }
  else
    [ -z "$SL$LL" ] || { xlog "$1: VIRHE a sidecar line under A: $SL $LL"; ok=0; }
  fi
  xlog "$1 modes: $WL | sidecar: ${SL:-none} | ${LL:-no sidecar load line} | lend: $(docker logs $CN 2>&1 | grep -m1 -i 'draft lend' | sed 's/^radiance: //' | cut -c1-200) | pool: $(docker logs $CN 2>&1 | grep -m4 'load: rank [0-9], mover took' | sed 's/^.*load: //' | cut -c1-110 | tr '\n' ';')"
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
stats0() {   # stats0 TAG: /stats + GET /admin/fn4 right after the check (routed units per tier, the lend)
  [ "$DRY" = 1 ] && { echo "+ GET $URL/stats > $O/$1-stats0.json; GET $URL/admin/fn4 > $O/$1-fn4-0.json"; return 0; }
  curl -s -m 30 $URL/stats > $O/$1-stats0.json 2>/dev/null; curl -s -m 10 $URL/admin/fn4 > $O/$1-fn4-0.json 2>/dev/null
  xlog "$1 at start: $(python3 -c 'import json,sys
try: s=json.load(open(sys.argv[1]))
except Exception as e: print("no /stats (%s)" % type(e).__name__); sys.exit(0)
try: g=json.load(open(sys.argv[2]))
except Exception: g={}
t={x.get("name"): x.get("units") for x in s.get("experts", {}).get("tiers", [])}
l=g.get("lend") or {}
print("routed units per tier %s; lend state %s lent %s" % (t, l.get("state"), l.get("lent_units")))' $O/$1-stats0.json $O/$1-fn4-0.json 2>&1 | cut -c1-400)"
}
cmpfile() {
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
gcmp() { [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }; [ -s "$1" ] && [ -s "$2" ] || { echo "not compared (missing $1 or $2)"; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
pcreset() { [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache"; return 0; }
  xlog "$1 prefix cache reset: HTTP $(curl -s -m 60 -o /dev/null -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)"; }
ident() {   # ident TAG: temperature 0 -- the 28k copy and the c1 greedy set vs bz d0
  local tag=$1 r
  pcreset $tag
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset 0 --tag=-$tag --outdir $O > $O/$tag-copy.txt 2>&1)"
  run "(cd $D && timeout 900 python3 $GS $URL $O/$tag-gs-c1.jsonl --conc 1 --max-tokens 192 > $O/$tag-gs-c1.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  r=$(cmpfile $O/copy-$tag-run1.txt $RD/copy-dsd-d0-i1-run1.txt); echo "$tag copy $r" >> $O/identity.txt
  xlog "$tag identity: 28k copy vs bz d0 (NST 0, stock): $r$(case $r in IDENTICAL*) ;; *) echo " -- reported, not hidden" ;; esac)"
  r=$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$tag-gs-c1.jsonl); echo "$tag gs $r" >> $O/identity.txt
  xlog "$tag identity: greedy set c1 vs bz d0: $r$(case $r in IDENTICAL*) ;; *) echo " -- reported, not hidden" ;; esac)"
}
blk() {   # blk TAG ARM: one z0blocks.py block under the stack2 pin (c1 essay / code, essay by position, edit, mb c4 / c8)
  run "timeout 3000 python3 -I $GGB --url $URL --mb-dir $D --edit $EP --label $1 --arm '$2={}' --order $2 --dur $DUR --warm 10 --fixed '$FIX5' --out $O/$1-blocks.jsonl > $O/$1-blocks.txt 2>&1"
  [ "$DRY" = 1 ] && return 0
  while IFS= read -r l; do xlog "$1 ${l:0:1500}"; done < <(grep -E '^(BLOCK|STATS|GGB \{"error)|Traceback|Error' $O/$1-blocks.txt)
}
sampled() {   # sampled TAG: the 1-user probe at the server-default sampling (no temperature sent; seed SEED), essay / code / edit x2
  run "(cd $D && timeout 1500 python3 -I -B $PRB --mode sampled --seed $SEED --runs 2 --tag $1 > $O/$1-samp.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  xlog "$1 sampled 1-user (server-default sampling, seed $SEED, 2 runs; mean, t95): $(grep '^PROBE ' $O/$1-samp.txt | cut -c7- | cut -c1-900) $(grep -m1 -E 'Traceback|Error' $O/$1-samp.txt | cut -c1-200)"
}
start() {   # start TAG ARM: restart + check (warm-up + fastcheck) + modes; 0 healthy
  local tag=$1 arm=$2 ms
  stackenv $(sideof $arm); export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive"
  export XS2_ENVFILE=$O/$tag.env; writeenv $XS2_ENVFILE
  xlog "$tag: start arm $arm IMAGE=$IMAGE $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ' | cut -c1-1600)"
  restart $tag || { keeplog $tag-fail; xlog "$tag: VIRHE not healthy: $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500) $(docker logs $CN 2>&1 | grep -m3 -E '^E |sidecar' | tr '\n' ' ' | cut -c1-500)"; return 1; }
  ms=$(check $tag 1); [ "$DRY" = 1 ] || xlog "$tag guard: 1-user $ms ms/step (informational; stack2 A 34-35)"
  modes $tag $arm || { down; return 1; }
}
trap 'down' EXIT
trap 'exit 143' TERM INT

# ---------------------------------------------------------------- checks
HMISS=""
for fn in log run mkd restart check down evict stopfile best_vars desc serve_args; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS lib:$fn"; done
for fn in xlog stackenv sideof writeenv keeplog fn4get nudge pinset modes stats0 cmpfile gcmp pcreset ident blk sampled start; do
  [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for f in $Y $PRB $GGB $EP $SUM $GS $D/ds41_tests.py $D/fastcheck.py $D/stepprobe.py $D/mb.py $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDECAR \
         $MODELDIR/$SQ8 $MODELDIR/$SQ6 $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl; do [ -e $f ] || HMISS="$HMISS file:$f"; done
for f in $PRB $GGB $EP $SUM $GS $D/ds41_tests.py $D/stepprobe.py $D/mb.py; do python3 -I -B -c 'import ast,sys; ast.parse(open(sys.argv[1]).read())' $f 2>/dev/null || HMISS="$HMISS parse:$f"; done
python3 -I -B $PRB --help > /dev/null 2>&1 || HMISS="$HMISS run:daprobe.py--help"
python3 -I -B $GGB --help > /dev/null 2>&1 || HMISS="$HMISS run:z0blocks.py--help"
python3 -I -B $SUM --selftest > /dev/null 2>&1 || HMISS="$HMISS run:dlwsum.py--selftest"
for s in "$SQ8 $SHA_Q8" "$SQ6 $SHA_Q6"; do set -- $s
  [ "$(sha256sum $MODELDIR/$1 2>/dev/null | cut -c1-64)" = "$2" ] || HMISS="$HMISS sha256:$1"; done
docker image inspect $IMAGE > /dev/null 2>&1 || HMISS="$HMISS image:$IMAGE"
IMGOK=$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -qzF RADIANCE_WEIGHT_SIDECAR /opt/radiance/bin/radiance && grep -qzF "read from the weight sidecar" /opt/radiance/bin/radiance && test -x /opt/radiance/bin/rad-sidecar && grep -qzF draft_depth0 /opt/radiance/bin/radiance && grep -qzF draft_lend /opt/radiance/bin/radiance && grep -qzF RADIANCE_DS41_ARENA_DECODE_ROWS /opt/radiance/share/radiance/architectures/deepseek41.so && grep -qzF RADIANCE_DS41_ROUTER_BF16 /opt/radiance/share/radiance/architectures/deepseek41.so && grep -qzF RADIANCE_DS41_DSPARK_WALK /opt/radiance/share/radiance/architectures/deepseek41.so && grep -qzF reset_prefix_cache /opt/radiance/bin/radiance && echo yes' 2>/dev/null)
[ "$IMGOK" = yes ] || HMISS="$HMISS image-literals:$IMAGE(sidecar / rad-sidecar / draft_depth0 / draft_lend / VRAM levers / W1)"
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config)"
case $ORDER in *[!ABC]*|"") HMISS="$HMISS order:$ORDER" ;; esac
if [ "$DRY" = 1 ]; then
  echo "+ item $ITEM, image $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)), out $O, order $ORDER (A none, B $SQ8 ${SHA_Q8:0:16}, C $SQ6 ${SHA_Q6:0:16}), seed $SEED, mb --dur $DUR; NST 5 pin $FIX5"
  [ -z "$HMISS" ] && echo "+ helpers: every file present and parsing, every function defined, the python helpers' --help / selftest, sidecars sha256 OK, image literals (sidecar / rad-sidecar / draft_depth0 / draft_lend / VRAM levers / W1), yaml accepted -- OK" \
                  || { echo "+ MISSING:$HMISS"; exit 1; }
else
  [ -z "$HMISS" ] || { xlog "$ITEM: VIRHE helpers missing:$HMISS"; exit 1; }
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "$ITEM: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
fi
mkd $O $O/guard $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt

# ---------------------------------------------------------------- run
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) | lever DLW: the DSpark drafter's dense linears from a weight sidecar, B q8_0 ($SQ8 ${SHA_Q8:0:16}) / C q6_K ($SQ6 ${SHA_Q6:0:16}) vs A bf16 (none) | cross-start $ORDER, NST 5 stack2 env + pin | per start: one z0blocks block (c1 essay / code / essay by position / edit at temperature 0, mb c4 / c8 natural, per-phase units / lend / depth 0) + the sampled 1-user probe (seed $SEED); identity on B / C"
evict
k=0
for arm in $(echo $ORDER | grep -o .); do
  tag=dlw-$k$arm
  if start $tag $arm; then
    stats0 $tag
    if pinset $tag "$FIX5"; then
      blk $tag $arm
      sampled $tag
      case $arm in B|C) ident $tag-i ;; esac
    else
      xlog "$tag: VIRHE the pin did not apply -> start skipped"
    fi
    run "docker logs $CN 2>&1 | gzip -1 > $O/$tag-engine-full.log.gz"
    down
  else
    xlog "$tag: VIRHE no healthy start -> arm $arm skipped"
  fi
  k=$((k + 1))
done
[ "$DRY" = 1 ] || { python3 -I -B $SUM $O $ORDER > $O/summary.txt 2>&1; while IFS= read -r l; do xlog "summary ${l:0:1800}"; done < $O/summary.txt; }
[ "$DRY" = 1 ] || xlog "summary identity: $(tr '\n' ',' < $O/identity.txt 2>/dev/null)"
unset $SWITCHES
xlog "$ITEM: done -- $O"
