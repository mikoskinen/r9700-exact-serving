#!/bin/bash
# dsm-sam.sh (11.11.2026, queue-ar/dsmain/da, ds-main-port DA agent; coordinator's request 04:20 UTC) -- NOT a queue item; the body
# of 57-p4zyzzzzz-ds-iej-sampled.sh. Tag "RESULT xsam ", out runs/xdsm/sampled (SAM_OUT). WHAT DRAFTING GIVES REAL USERS TODAY:
# real clients send no temperature, so the engine samples at its default (temperature 1); every drafting number so far is greedy.
# IMAGE the int3 image radiance-local:ds41-int3-1116248a-5c13e555-l, the stack2 env AS IT RAN (hxzzy-stack2: HYB=1, XS2_LEND=1,
# XS2_VRAM=all, drafter_mx 1, walk 1, the small n5 set; its NST 0 start's env for N):
#   START 1, NST 5 (-v off), ABBA in process (POST /admin/fn4 + epoch_bump a block, applied at a step):
#     A = drafting as stack2 runs it (the stack2 pin: adaptive, depth 0 1, hybrid 1, lend 1 + backoff 3, ...)
#     B = A + draft_depth0 2 (depth 0 FORCED: plain decode inside the NST 5 server, only the probes draft; exact)
#     each block: daprobe.py sampled (no temperature sent = the server default; per-request seed 1234; essay, code, edit) then
#     daprobe.py greedy (temperature 0, the reference); then identity under A (temperature 0): 28k copy + greedy set c1 vs bz d0.
#   START 2, NST 0 (stack2's n0 env and pin): two blocks N N of the same probes + identity.
#   Per arm and mode: decode tok/s, tokens a step, acceptance (mean and t95 over the blocks), B / A and N / A.
# ~25-30 min (evict 1; NST 5: start ~5, check ~1.5, 4 blocks ~6, identity ~2; NST 0: start ~4.5, check ~1.5, 2 blocks ~3,
# identity ~2). NO cc_sim, no 2.9; no power / host / fan settings; production compose files read only (common.yaml via extends).
# DRYRUN=1 bash dsm-sam.sh : prints every command; checks the image, every helper (files present and parsing, functions defined,
#   the image's literals), the yaml (docker compose config); no server, no GPU, nothing written. A missing helper fails it.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
Y=$HD/ds41-da-tp4.yaml
O=${SAM_OUT:-$R/xdsm/sampled}; export STATSDIR=$O/stats
RD=$R/xdsm/draft
export IMAGE=${SAM_IMAGE:-radiance-local:ds41-int3-1116248a-5c13e555-l}
ITEM=${SAM_ITEM:-57-p4zyzzzzz-ds-iej-sampled}
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
PRB=$HD/daprobe.py; SEED=1234
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
SIDECAR=dspark-conf.f32
GPTAB=q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": 8, "drafter_mx": 1, "gg_pf": 1, "f32_pf": 25, "dspark_walk": 1, "ds41_argather": 48, "idx_split": 1, "ds41_qnfold": 1, "ds41_swfold": 1, "draft_ctxskip": 1, "dspark_conf": 1, "dspark_conf_rows": 1, "draft_hybrid": 1, "draft_lend": 1, "draft_probe_backoff": 3}'
FIX0='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "gg_pf": 1, "f32_pf": 25}'
ARM_A='{}'; ARM_B='{"draft_depth0": 2}'
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
  RADIANCE_WEIGHT_OVERLAY RADIANCE_DS41_DA_TAPS RADIANCE_KLD_DUMP"
xlog() { log "RESULT xsam $*"; }
# stackenv n5|n0: dsm-stack2.sh's stackenv + levenv with XS2_VRAM=all, XS2_LEND=1 (n5), walk 1, the small n5 set, turnover 0;
# the start-time lever words also go to $XS2_ENVFILE (the yaml's env_file)
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
  if [ $1 = n5 ]; then
    w="$w RADIANCE_DS41_ARENA_DECODE_ROWS=48 RADIANCE_DS41_DSPARK_WALK=1 RADIANCE_DS41_ARGATHER_MAX=48 RADIANCE_DS41_QNFOLD=1 RADIANCE_DS41_SWFOLD=1"
    w="$w RADIANCE_DRAFT_CTXSKIP=0 RADIANCE_DS41_DSPARK_CONF=/models/$SIDECAR RADIANCE_DRAFT_LEND=0 RADIANCE_DRAFT_PROBE_BACKOFF=0 RADIANCE_DRAFT_LEND_BORROW=0"
  else
    w="$w RADIANCE_DS41_ARENA_DECODE_ROWS=8"
  fi
  for kv in $w; do export "$kv"; done
  LEVENV=$w
}
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
  xlog "$tag pin: $(cat $O/pin-$tag.txt) -- $(echo "$s1" | cut -c1-500)"
  return $rc
}
modes() {   # modes TAG n5|n0: the start words in the container (and the W1 line at NST 5)
  [ "$DRY" = 1 ] && { echo "+ modes $1: the start words in the container$([ $2 = n5 ] && echo ', the W1 declared line over 4 ranks')"; return 0; }
  local CE ok=1 kv WL
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in RADIANCE_PLACE_TURNOVER=0 RADIANCE_DS41_GG_GEMV=2 RADIANCE_DS41_ROUTER_BF16=1 $([ $2 = n5 ] && echo RADIANCE_DS41_DSPARK_WALK=1 RADIANCE_DRAFT_LEND=0); do
    echo "$CE" | grep -qxF "$kv" || { xlog "$1: VIRHE $kv not in the container"; ok=0; }; done
  if [ $2 = n5 ]; then
    WL=$(docker logs $CN 2>&1 | grep -m1 'DSpark lever W1.*fn4 dspark_walk=' | sed 's/^radiance: //' | cut -c1-200)
    case "$WL" in *"over 4 rank(s)"*"dspark_walk=1 "*) ;; *) xlog "$1: VIRHE no W1 declared line over 4 ranks"; ok=0 ;; esac
  fi
  xlog "$1 modes: ${WL:-NST 0, no drafter} | lend: $(docker logs $CN 2>&1 | grep -m1 -i 'draft lend' | sed 's/^radiance: //' | cut -c1-200)"
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
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
block() {   # block TAG ARM BLOCK#: the sampled probe (seed SEED) then the greedy one, one run each
  run "(cd $D && timeout 1200 python3 -I -B $PRB --mode sampled --seed $SEED --runs 1 --tag $1-$2$3 > $O/$1-b$3$2-samp.txt 2>&1; timeout 1200 python3 -I -B $PRB --mode greedy --runs 1 --tag $1-$2$3 > $O/$1-b$3$2-greedy.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  xlog "$1 block $3 arm $2: $(cat $O/$1-b$3$2-samp.txt $O/$1-b$3$2-greedy.txt | grep '^J ' | python3 -c 'import json,sys
for l in sys.stdin:
    d=json.loads(l[2:]); print("%s %s %s tok/s %s tok/step acc %s;" % (d["mode"], d["prompt"], d["decode_tps"], d["tokens_per_step"], d["acceptance"]), end=" ")' 2>&1 | cut -c1-900) $(grep -m1 -hE 'Traceback|Error' $O/$1-b$3$2-samp.txt $O/$1-b$3$2-greedy.txt | cut -c1-200)"
}
start() {   # start TAG n5|n0: restart + check (warm-up + fastcheck) + modes; 0 healthy
  local tag=$1 ms
  stackenv $2; export MODEL=$M225; HP29=19456; best_vars
  if [ $2 = n5 ]; then export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive"; else export MNBT=4160 NST=0 XARGS="$XARGS --live"; fi
  export XS2_ENVFILE=$O/$tag.env; writeenv $XS2_ENVFILE
  xlog "$tag: start IMAGE=$IMAGE $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ' | cut -c1-1500)"
  restart $tag || { keeplog $tag-fail; xlog "$tag: VIRHE not healthy: $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; return 1; }
  ms=$(check $tag 1); [ "$DRY" = 1 ] || xlog "$tag guard: 1-user $ms ms/step (informational)"
  modes $tag $2 || { down; return 1; }
}
trap 'down' EXIT
trap 'exit 143' TERM INT

# ---------------------------------------------------------------- checks
HMISS=""
for fn in log run mkd restart check down evict stopfile best_vars desc serve_args; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS lib:$fn"; done
for fn in xlog stackenv writeenv keeplog fn4get nudge pinset modes cmpfile gcmp pcreset ident block start; do
  [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for f in $Y $PRB $GS $D/ds41_tests.py $D/fastcheck.py $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDECAR \
         $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl; do [ -e $f ] || HMISS="$HMISS file:$f"; done
for f in $PRB $GS $D/ds41_tests.py; do python3 -I -B -c 'import ast,sys; ast.parse(open(sys.argv[1]).read())' $f 2>/dev/null || HMISS="$HMISS parse:$f"; done
python3 -I -B $PRB --help > /dev/null 2>&1 || HMISS="$HMISS run:daprobe.py--help"
docker image inspect $IMAGE > /dev/null 2>&1 || HMISS="$HMISS image:$IMAGE"
IMGOK=$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -qzF draft_depth0 /opt/radiance/bin/radiance && grep -qzF draft_lend /opt/radiance/bin/radiance && grep -qzF RADIANCE_DS41_ARENA_DECODE_ROWS /opt/radiance/share/radiance/architectures/deepseek41.so && grep -qzF RADIANCE_DS41_ROUTER_BF16 /opt/radiance/share/radiance/architectures/deepseek41.so && grep -qzF RADIANCE_DS41_DSPARK_WALK /opt/radiance/share/radiance/architectures/deepseek41.so && grep -qzF reset_prefix_cache /opt/radiance/bin/radiance && echo yes' 2>/dev/null)
[ "$IMGOK" = yes ] || HMISS="$HMISS image-literals:$IMAGE(draft_depth0 / draft_lend / VRAM levers / W1)"
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config)"
if [ "$DRY" = 1 ]; then
  echo "+ item $ITEM, image $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)), out $O, seed $SEED; NST 5 pin $FIX5; arms A $ARM_A, B $ARM_B (ABBA); NST 0 pin $FIX0"
  [ -z "$HMISS" ] && echo "+ helpers: every file present and parsing, every function defined, image literals (draft_depth0 / draft_lend / VRAM levers / W1), yaml accepted -- OK" \
                  || { echo "+ MISSING:$HMISS"; exit 1; }
else
  [ -z "$HMISS" ] || { xlog "$ITEM: VIRHE helpers missing:$HMISS"; exit 1; }
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "$ITEM: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
fi
mkd $O $O/guard $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt

# ---------------------------------------------------------------- run
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) | what drafting gives real users: server-default sampling (no temperature sent, seed $SEED) vs greedy | NST 5 stack2 (hybrid 1, lend 1 + backoff 3, VRAM levers, drafter_mx 1) ABBA A=stack2 drafting B=draft_depth0 2 (forced depth 0) + NST 0 stack2 N N | 1-user essay / code / edit | identity at temperature 0"
evict
if start sam-n5 n5; then
  k=0
  for arm in A B B A; do
    eval aj=\$ARM_$arm
    if pinset sam-n5-$k$arm "$FIX5" "$aj"; then block sam-n5 $arm $k; else xlog "sam-n5 block $k $arm: VIRHE the pin did not apply -> block skipped"; fi
    k=$((k + 1))
  done
  pinset sam-n5-ident "$FIX5" && ident sam-n5-iA
  run "docker logs $CN 2>&1 | gzip -1 > $O/sam-n5-engine-full.log.gz"
  down
fi
if start sam-n0 n0; then
  for k in 0 1; do
    if pinset sam-n0-${k}N "$FIX0"; then block sam-n0 N $k; else xlog "sam-n0 block $k N: VIRHE the pin did not apply"; fi
  done
  ident sam-n0-iN
  run "docker logs $CN 2>&1 | gzip -1 > $O/sam-n0-engine-full.log.gz"
  down
fi
[ "$DRY" = 1 ] || xlog "summary (mean over blocks, t95; sampled = server default + seed $SEED, greedy = temperature 0): $(python3 -c 'import glob,json,math,os,statistics,sys
O=sys.argv[1]; T95={1:12.706,2:4.303,3:3.182}
rows=[]
for f in glob.glob(os.path.join(O, "sam-n*-b*-*.txt")):
    arm=os.path.basename(f).split("-b")[1][1]
    for l in open(f):
        if l.startswith("J "): d=json.loads(l[2:]); d["arm"]=arm; rows.append(d)
def ci(v):
    v=[x for x in v if x is not None]
    if not v: return None
    m=statistics.mean(v); h=T95.get(len(v)-1,1.96)*statistics.stdev(v)/math.sqrt(len(v)) if len(v)>1 else 0.0
    return m,h
out=[]
for mode in ("sampled","greedy"):
    for p in ("essay","code","edit"):
        s=[]; base=None
        for arm in ("A","B","N"):
            x=[r for r in rows if r["mode"]==mode and r["prompt"]==p and r["arm"]==arm]
            t=ci([r["decode_tps"] for r in x]); st=ci([r["tokens_per_step"] for r in x])
            if not t: continue
            if arm=="A": base=t[0]
            s.append("%s %.1f+-%.1f tok/s %.2f tok/step%s" % (arm, t[0], t[1], st[0] if st else 0, "" if arm=="A" or not base else " (%+.1f %% vs A)" % (100*(t[0]/base-1))))
        if s: out.append("%s %s: %s" % (mode, p, ", ".join(s)))
print(" | ".join(out) or "none")' $O 2>&1 | cut -c1-2400) || identity: $(tr '\n' ',' < $O/identity.txt 2>/dev/null)"
unset $SWITCHES
xlog "$ITEM: done -- $O"
