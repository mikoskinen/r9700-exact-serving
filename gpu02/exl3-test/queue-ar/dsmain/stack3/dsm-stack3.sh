#!/bin/bash
# dsm-stack3.sh (11.11.2026, queue-ar/dsmain/stack3, ds-main-port stack3 builder) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-g-stack3.sh. Log tag "RESULT xs3 ", out runs/xdsm/stack3 (XS3_OUT).
# THE DS STACK CONFIRMATION 3 on the ds41-int4 image (ds41-int4 = ds41-int3 1116248 + ds41-bi 2d9a1dc (lever BI: ds41_bi 2 = arm P,
# 4 = arm C on P) + ds41-dlw df86048 (lever DLW: RADIANCE_WEIGHT_SIDECAR, the DSpark drafter's bf16 linears as q8_0) + ds41-sd
# 397608f (lever SD: fn4 draft_sample, --draft-coupling on); every switch unset = ds41-int3). Design = STACK CONFIRMATION 2
# (dsm-stack2.sh: stack2's env, serve line and pins, two starts NST 0 then NST 5, decode before prefill, 136k cold prefill,
# identity vs bz d0, targets) as an IN-PROCESS A/B: in each start the base arm is stack2's switch set, the stack arms add lever BI
# (ds41_bi 2 = P, then 4 = C; both are runtime values here because 57-...-bi-f-pc decides between them) and, at NST 5, lever SD
# (draft_sample 1). DLW is start-time: see "DLW" below.
#
# START n0 (NST 0, stack2 N's env + RADIANCE_DS41_BI=0 declared; no drafter, so no SD / DLW):
#   arms  A {ds41_bi 0}   P {ds41_bi 2}   C {ds41_bi 4}       (at NST 0 the decode segment is <= 8 rows: 2 issues as 1, 4 as 3)
# START n5 (DSpark NST 5 adaptive, stack2 A's env + RADIANCE_DS41_BI=0 + RADIANCE_DRAFT_SAMPLE=0 + --draft-coupling on +
#   RADIANCE_WEIGHT_SIDECAR=/models/dspark-dlw-q8_0.rad):
#   arms  A {ds41_bi 0, draft_sample 0}   P {ds41_bi 2, draft_sample 1}   C {ds41_bi 4, draft_sample 1}
#         T 1 arm S {ds41_bi XS3_T1BI (2), draft_sample 1} vs A
# Per start, in this order (decode before prefill, TESTING 9.11.):
#   gate1 -> modes (stack2's literals + lever BI declared at 0, at n5 lever SD declared + draft_sample 0 at start + the sidecar's
#   43 entries read) -> the start's VRAM / lend lines (/stats experts, rank-0 mover / lend lines) -> switch note (every arm's set
#   applied at a step; refusals: ds41_kvfold 2, ds41_bi 5, draft_sample 201 at n5 / 1 at n0) -> warm-up answers vs stock ->
#   SMOKE under C (greedy set at conc 4, mixed steps; at n5 also mb c4 20 s at T 1 = the coupled path) ->
#   T 0 BLOCKS (s3blocks.py, XS3_ORDER A P C C P A, pins in process + epoch_bump): stepprobe essay / code (c1) + editprobe at
#     T 0 (stack2's 1-user probes), mb c1 (n5) / c4 / c8 / pf at TEMPERATURE 0 (mbt.py: mb.py read only + {"temperature": 0}),
#     dur XS3_DUR 60 warm 15; per phase /stats + the lend (VRAM units and lent units sampled every 2 s, lends / reclaims /
#     lent cycles), the draft counters and the SD "sample" counters -> lever BI stats (engine)
#   -> T 1 BLOCKS (n5; XS3_T1ORDER A S S A): sdprobe essay / code at the server default (temperature 1, seed 1234) with thinking on
#     (reasoning_effort medium), mb c4 / c8 at the server default (mb.py as stack2 ran it) -> SD counters a phase
#   -> 136k COLD PREFILL under P (deepq.py's fixed prompt, 512 greedy tokens: TTFT + the answer vs dsm-225)
#   -> IDENTITY at T 0 (loadid.py / greedyset.py, dsm-bilid6.sh's machinery): a1-A (base arm alone vs bz d0: the sidecar /
#     coupling / BI declared change nothing), per value v of XS3_IDVALS0 / XS3_IDVALS5: a1-sv alone (vs bz d0 and vs a1-A),
#     ld-sv (each prompt under mb c4 + one fresh ~32k prefill), c4a-sv / c4b-sv (run-to-run), then a2 (alone again, control);
#     the 28k copy under P vs bz d0. Target: every compare IDENTICAL (20 / 20).
# SUMMARY: per start, arm and metric mean +- t95; P / C / S vs A in process (pooled; T 1 also the ABBA pairs); each vs stack2
#   (hxzzy-stack2, runs/xdsm/stack2, the same start's arm; CROSS-START, labelled; stack2's c4 / c8 were mb at the server default,
#   so they compare with the T 1 blocks, its c1 / edit at T 0 with the T 0 blocks); the targets 1u >= 90, c4 >= 160, c8 >= 200,
#   136k <= 35 s; the identity verdict; one "RESULT xs3 summary:" line.
# DLW (start-time, cannot be switched in process) -- DECISION: the n5 start runs WITH the q8_0 sidecar for every arm and this
#   item gives DLW NO cross-start control of its own. Its control is already measured / queued: iejzz-dlw (cross-start A B C A,
#   B q8_0: identity IDENTICAL, 1u code +2.6 %, c4 / c8 within noise) and iel-dlwb (A B A, queued after this item). Re-running a
#   no-sidecar start here would be a baseline re-run (TESTING rule 2) of ~35 min for a lever whose in-process A/B is impossible
#   anyway; instead the base arm A of n5 (stack2's switches on the sidecar start) is compared cross-start with hxzzy-stack2 A
#   (labelled cross-start), and the stack arms in process with A. At NST 0 the drafter is not declared, so no sidecar there.
# ~130 min (evict 1; n0 ~48: start ~5, T 0 blocks 6 x ~4.9, 136k ~1.5, identity ~12.5 with one value; n5 ~80: start ~7, T 0
#   blocks 6 x ~5.8, T 1 blocks 4 x ~3.1, 136k ~1.5, identity ~23.5 with two values). NO cc_sim (user 10.11.), no 2.9; no power /
#   host / fan settings; production compose files read only (--project-directory). Exact levers only; SD at T > 0 has the user's
#   GO (distribution-exact); at T 0 everything here must stay byte-identical.
# Knobs: XS3_IMAGE, XS3_ITEM, XS3_OUT, XS3_STARTS (n0 n5), XS3_ORDER (APCCPA), XS3_PH0 (c1,edit,c4,c8,pf), XS3_PH5
#   (c1,edit,mbc1,c4,c8,pf), XS3_DUR (60), XS3_WARM (15), XS3_T1ORDER (ASSA), XS3_T1BI (2), XS3_IDVALS0 (4), XS3_IDVALS5 (2 4),
#   XS3_SIDECAR (dspark-dlw-q8_0.rad), XS3_SIDESHA (its sha256), XS3_PFTOK (32768), XS3_OFFSET (0.5).
# DRYRUN=1 bash dsm-stack3.sh : prints every command; checks the image (its literals: the int4 registry, BI / SD / DLW / stack2
#   literals), every helper (files present and parsing, mbt.py / loadid.py selftests, s3blocks.py / sdprobe.py --help, every
#   function defined), the sidecars (sha256), the yaml (docker compose config); no server, no GPU, writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/../ds41-stack2-tp4.yaml                   # stack2's yaml, read only (env_file ${XS2_ENVFILE} carries every start word)
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=${XS3_OUT:-$R/xdsm/stack3}; export STATSDIR=$O/stats
RD=$R/xdsm/draft                                # bz's d0 files (identity reference: copy + greedy set, NST 0 stock)
RM=$R/xdsm                                      # item a's dsm-225: deep-dsm-225.txt(.json) (136k cold reference)
RS2=$R/xdsm/stack2                              # hxzzy-stack2's outputs (int3): the summary's cross-start comparison, read only
export IMAGE=${XS3_IMAGE:?set XS3_IMAGE}
ITEM=${XS3_ITEM:-57-p4zyzzzzz-ds-iejzzzszzzzzz-bi-g-stack3}
STARTS=${XS3_STARTS:-n0 n5}; ORDER=${XS3_ORDER:-APCCPA}; T1ORDER=${XS3_T1ORDER-ASSA}; T1BI=${XS3_T1BI:-2}
PH0=${XS3_PH0:-c1,edit,c4,c8,pf}; PH5=${XS3_PH5:-c1,edit,mbc1,c4,c8,pf}; DUR=${XS3_DUR:-60}; WARMS=${XS3_WARM:-15}
IDVALS0=${XS3_IDVALS0-4}; IDVALS5=${XS3_IDVALS5-2 4}
SIDE=${XS3_SIDECAR:-dspark-dlw-q8_0.rad}; SIDESHA=${XS3_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
OFF=${XS3_OFFSET:-0.5}; PFT=${XS3_PFTOK:-32768}; CONC=4; WARM=10; MT=192; SEED=1234
for s in $STARTS; do case $s in n0|n5) ;; *) echo "XS3_STARTS takes n0 / n5 (got '$s')"; exit 1 ;; esac; done
case $ORDER in *[!APC]*|"") echo "XS3_ORDER takes A / P / C blocks (got '$ORDER')"; exit 1 ;; esac
case $T1ORDER in *[!AS]*) echo "XS3_T1ORDER takes A / S blocks or nothing (got '$T1ORDER')"; exit 1 ;; esac
case $T1BI in 0|1|2|3|4) ;; *) echo "XS3_T1BI must be 0..4 (got '$T1BI')"; exit 1 ;; esac
for v in $IDVALS0 $IDVALS5; do case $v in 1|2|3|4) ;; *) echo "XS3_IDVALS0 / XS3_IDVALS5 take ds41_bi values 1..4 (got '$v')"; exit 1 ;; esac; done
for v in DUR WARMS OFF PFT; do case ${!v} in ''|*[!0-9.]*) echo "$v must be a number (got '${!v}')"; exit 1 ;; esac; done
for p in $(echo $PH0 $PH5 | tr ',' ' '); do case $p in c1|edit|mbc1|c4|c8|pf) ;; *) echo "XS3_PH0 / XS3_PH5: unknown phase '$p'"; exit 1 ;; esac; done
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
LD=$HD/loadid.py; GGB=$HD/s3blocks.py; MBT=$HD/mbt.py; PRB=$HD/sdprobe.py; EP=$HD/editprobe.py; DQ=$QA/deepq.py; MBP=$D/mb.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
xlog() { log "RESULT xs3 $*"; }
# ---------------------------------------------------------------- stack2's configuration as hxzzy-stack2 ran it (fixed; = dsm-bilid6.sh)
HYB=1; MATCH=8; DMX=1; GGPF=1; F32PF=25; WALK=1; SMALL5=1; LEND=1; BORROW=0; VRAMSEL=all
GPTAB=q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422
SIDECAR=dspark-conf.f32                         # MODELDIR's confidence head (the release's), /models/dspark-conf.f32 in the container
VTAB='
arena   RADIANCE_DS41_ARENA_DECODE_ROWS=8  RADIANCE_DS41_ARENA_DECODE_ROWS=48  activation,arena:,8,tokens,use  activation,arena:,48,tokens,use
bf16    RADIANCE_DS41_ROUTER_BF16=1        RADIANCE_DS41_ROUTER_BF16=1         -                               -
emb     RADIANCE_DS41_EMB_RAM=1            RADIANCE_DS41_EMB_RAM=1             a,rank),ram                     a,rank),ram
shadow  RADIANCE_PLACE_SHADOW_SLOTS=16     RADIANCE_PLACE_SHADOW_SLOTS=16      -                               -
xpoff   RADIANCE_XPREFETCH=0               RADIANCE_XPREFETCH=0                expert,prefetch:,off            expert,prefetch:,off
'
VNAMES=$(echo "$VTAB" | awk 'NF && $1 !~ /^#/ {print $1}' | xargs)
[ "$VRAMSEL" = all ] && VRAMSEL=$VNAMES
vf() { echo "$VTAB" | awk -v n="$1" -v k="$2" '$1 == n {print $k; exit}'; }   # vf LEVER FIELD
# levenv n0|n5: every start-time word of that start: stack2's + lever BI declared at 0 (+ at n5 lever SD at 0, the DLW sidecar)
levenv() {
  local v w=""
  [ $GGPF = 1 ] && w="$w RADIANCE_DS41_GG_PF=1 RADIANCE_DS41_GG_PF_TABLE=$GPTAB"
  [ $F32PF != 0 ] && w="$w RADIANCE_DS41_F32_PF=$F32PF"
  for v in $VRAMSEL; do w="$w $(vf $v $([ $1 = n0 ] && echo 2 || echo 3) | tr ',' ' ')"; done
  if [ $1 = n5 ]; then
    [ $WALK != 0 ] && w="$w RADIANCE_DS41_DSPARK_WALK=$WALK"
    [ $SMALL5 = 1 ] && w="$w RADIANCE_DS41_ARGATHER_MAX=48 RADIANCE_DS41_QNFOLD=1 RADIANCE_DS41_SWFOLD=1 RADIANCE_DRAFT_CTXSKIP=0 RADIANCE_DS41_DSPARK_CONF=/models/$SIDECAR"
    [ $LEND = 1 ] && w="$w RADIANCE_DRAFT_LEND=0 RADIANCE_DRAFT_PROBE_BACKOFF=0 RADIANCE_DRAFT_LEND_BORROW=0"
    w="$w RADIANCE_DRAFT_SAMPLE=0 RADIANCE_WEIGHT_SIDECAR=/models/$SIDE"
  fi
  w="$w RADIANCE_DS41_BI=0"
  echo $w
}
# levlits n0|n5: the start-log literals the start's levers must leave ('|'-separated; ',' in a literal = a space)
levlits() {
  local v l x=""
  [ $GGPF = 1 ] && x="$x|gemm_nt_ggml's,prefill,rows,take,lever,GPF's,forms,(RADIANCE_DS41_GG_PF=1"
  [ $F32PF != 0 ] && x="$x|gemm_nt_f32's,prefill,rows,take,lever,GPF-b's,tiled,form,(RADIANCE_DS41_F32_PF=$F32PF"
  for v in $VRAMSEL; do l=$(vf $v $([ $1 = n0 ] && echo 4 || echo 5)); [ "$l" = - ] || x="$x|$l"; done
  if [ $1 = n5 ]; then
    [ $WALK != 0 ] && x="$x|DSpark,lever,W1:,the,full-vocabulary,Markov,walk,declared,(dspark_walk_step,a,step,+,all_gather,over,4,rank(s)|fn4,dspark_walk=$WALK,"
    [ $SMALL5 = 1 ] && x="$x|one,kernel,up,to,48,tokens,a,step|RADIANCE_DS41_QNFOLD;,ds41_qnfold=1)|RADIANCE_DS41_SWFOLD;,ds41_swfold=1)|draft,context,skip,(RADIANCE_DRAFT_CTXSKIP=0|a,confidence,plane,(the,dspark_conf,gate"
    [ $LEND = 1 ] && x="$x|draft,lend:,set,up"
    x="$x|DSpark,lever,SD:,the,sampled,full-vocabulary,walk,declared,(dspark_walk_sample_step,a,step,over,4,rank(s)|lever,SD:,fn4,draft_sample,0,at,start"
    x="$x|weight,sidecar:,43,entries|read,from,the,weight,sidecar"
  fi
  x="$x|lever,BI,declared,(RADIANCE_DS41_BI|now,0):,a,step,of,more"
  echo "${x#|}"
}
# the pins: stack2's (dsm-stack2.sh as hxzzy-stack2 ran it: NST 0 FIX0, NST 5 FIX + draft_hybrid 1) + the stack3 arms
P0=""; [ $GGPF = 1 ] && P0="$P0, \"gg_pf\": 1"; [ $F32PF != 0 ] && P0="$P0, \"f32_pf\": $F32PF"
P5=""; [ $WALK != 0 ] && P5="$P5, \"dspark_walk\": $WALK"
[ $SMALL5 = 1 ] && P5="$P5, \"ds41_argather\": 48, \"idx_split\": 1, \"ds41_qnfold\": 1, \"ds41_swfold\": 1, \"draft_ctxskip\": 1, \"dspark_conf\": 1, \"dspark_conf_rows\": 1"
[ $LEND = 1 ] && P5="$P5, \"draft_lend\": 1, \"draft_probe_backoff\": 3"
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": '"$MATCH"'}'
[ "$DMX" = 1 ] && FIX5="${FIX5%\}}, \"drafter_mx\": 1}"
FIX5="${FIX5%\}}$P0$P5, \"draft_hybrid\": $HYB}"
FIX0="{\"gg_gemv\": 2, \"aql_acq\": 2, \"ds41_kvfold\": 1, \"ds41_ofold\": 1, \"ds41_onefill\": 1$P0}"
armj() {   # armj n0|n5 A|P|C|S|<bi value>: the arm's own switches (JSON); a digit = the stack arm at that ds41_bi
  local s=$1 a=$2 ds=""
  [ $s = n5 ] && ds=', "draft_sample": 1'
  case $a in
    A) [ $s = n5 ] && echo '{"ds41_bi": 0, "draft_sample": 0}' || echo '{"ds41_bi": 0}' ;;
    P) echo "{\"ds41_bi\": 2$ds}" ;;
    C) echo "{\"ds41_bi\": 4$ds}" ;;
    S) echo "{\"ds41_bi\": $T1BI$ds}" ;;
    [0-9]) echo "{\"ds41_bi\": $a$ds}" ;;
  esac
}
fixof() { [ $1 = n5 ] && echo "$FIX5" || echo "$FIX0"; }
pinj() {   # pinj n0|n5 ARM [EXTRAJSON]: the start's fixed pin + the arm + extra (e.g. epoch_bump) as one POST body
  local x=${3:-}; [ -n "$x" ] || x='{}'
  python3 -c 'import json,sys
d = {}
for a in sys.argv[1:]: d.update(json.loads(a))
print(json.dumps(d))' "$(fixof $1)" "$(armj $1 $2)" "$x"
}
DESC_N="NST 0 stack2 N (gg_gemv 2, rule-w, K1 folds c/d/f, gg_pf 1 (table), f32_pf 25, VRAM levers all, turnover 0, no drafter) + lever BI declared"
DESC_5="NST 5 stack2 A (+ DSpark NST 5 adaptive, depth 0, hybrid 1 (match 8), dspark_walk 1, small n5 set, drafter_mx 1, lend + back-off 3) + lever BI declared + lever SD (--draft-coupling on, draft_sample 0 at start) + lever DLW sidecar $SIDE"
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
  RADIANCE_WEIGHT_OVERLAY RADIANCE_DS41_DA_TAPS RADIANCE_KLD_DUMP RADIANCE_DRAFT_SAMPLE RADIANCE_WEIGHT_SIDECAR
  RADIANCE_DS41_BI RADIANCE_DS41_BITRACE RADIANCE_DS41_BITRACE_PMAX"
VRAMENV=RADIANCE_PLACE_TURNOVER=0
stackenv() {   # stackenv n0|n5: 105's t3 arm + gg_gemv 2 + rule-w + the three K1 folds declared (and on) at start + turnover 0
  # + the start's lever words (levenv), every other switch unset; LEVENV = those words (the env file's content)
  local kv
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  for kv in $VRAMENV; do export "$kv"; done
  LEVENV=$(levenv $1)
  for kv in $LEVENV; do export "$kv"; done
}
LEVENV=""
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
thermline() { type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"; }
cmpfile() {   # cmpfile A B -> IDENTICAL (n chars) | DIFFERENT at char k (na vs nb chars) | not compared
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xs3 $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { xlog "$1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  echo "$1 warm-up $r" >> $O/identity.txt
  case $r in IDENTICAL*) xlog "$1 warm-up answers vs stock ($(basename $2)): $r (105 tuned3l: IDENTICAL)" ;;
    *) xlog "$1 warm-up answers vs stock ($(basename $2)): VIRHE $r -- NOT EXACT vs the tuned3l line" ;; esac
}
VKNOBS=""
vrammodes() {   # vrammodes TAG: the start env in the container's env, the mover's shadow / kept-free counts a rank, the prefetch state
  local MV XP CE kv bad="" nz
  MV=$(docker logs $CN 2>&1 | grep -E '^I\[[0-9]\] mover: [0-9]+ slab slots' | sed -E 's/^I\[([0-9])\] mover: ([0-9]+) slab slots \(([0-9]+) shadow, ([0-9]+) more kept free.*/r\1 \3 shadow \4 kept free (\2 slots)/' | tr '\n' ',' | sed 's/,$//')
  XP=$(docker logs $CN 2>&1 | grep -cE '^I rank [0-9]+: expert prefetch on')
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in $VRAMENV $LEVENV; do echo "$CE" | grep -qxF "$kv" || bad="$bad ${kv%%=*}"; done
  VKNOBS="env [$VRAMENV] + lever env [$(echo $LEVENV | sed 's/RADIANCE_DS41_GG_PF_TABLE=[^ ]*/RADIANCE_DS41_GG_PF_TABLE=<table>/')]$([ -n "$bad" ] && echo " NOT in the container:$bad"); mover ${MV:-no line}; expert prefetch on at $XP of 4 ranks"
  [ -z "$bad" ] || { xlog "$1: VIRHE the VRAM / lever env did not reach the container:$bad"; return 1; }
  [ "$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots')" = 4 ] || { xlog "$1: VIRHE no mover line on all 4 ranks: ${MV:-none}"; return 1; }
  nz=$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots \([0-9]+ shadow, 0 more kept free')
  [ "$nz" = 4 ] || { xlog "$1: VIRHE turnover 0 asked but $nz of 4 ranks keep 0 slots free: $MV"; return 1; }
  return 0
}
modes() {   # modes TAG n0|n5: stack2's checks (dsm-bilid6.sh) + the stack3 levers' declare lines (levlits) -> 0, else 1
  local AR=4; [ $2 = n5 ] && [ $SMALL5 = 1 ] && AR=48
  [ "$DRY" = 1 ] && { echo "+ modes $1: lever literals [$(levlits $2 | tr ',' ' ')]; ARGATHER note up to $AR rows; the env file's words in the container's env; indexer note, staging line, no shared-expert-lane note, rule-w in every recorded pass, the lever K1 note (all three declared, =1), the mover's 0 kept free on 4 ranks"; return 0; }
  local MLF SL AL XL KL NW NR ok=1 lit LL
  vrammodes $1 || ok=0
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  AL=$(docker logs $CN 2>&1 | grep -m1 'the MoE gather and its all-reduce as one kernel' | sed 's/^radiance: //' | cut -c1-300)
  XL=$(docker logs $CN 2>&1 | grep -m1 'the shared expert on the second lane' | sed 's/^radiance: //' | cut -c1-300)
  KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-500)
  xlog "$1 modes: $(echo "$MLF" | cut -c1-600) | staging: ${SL:-none} | argather: ${AL:-none} | K1: ${KL:-none} | VRAM: ${VKNOBS:-none}"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate" "post-collapse-norm separate at 0 layers" "next collapse-norm separate at 0 layers"; do
    case "$MLF" in *"$w"*) ;; *) xlog "$1: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  if [ $AR = 48 ]; then
    case "$AL" in *"up to 48 tokens a step (RADIANCE_DS41_ARGATHER, RADIANCE_DS41_ARGATHER_MAX"*) ;; *) xlog "$1: VIRHE the ARGATHER note is not the 48-row one: '${AL}'"; ok=0 ;; esac
  else
    case "$AL" in *"up to 4 tokens a step (RADIANCE_DS41_ARGATHER)"*) ;; *) xlog "$1: VIRHE the ARGATHER note is not today's 4-row one: '${AL}'"; ok=0 ;; esac
  fi
  [ -z "$XL" ] || { xlog "$1: VIRHE a shared-expert-lane note (RADIANCE_DS41_SHEXP_LANE should be unset): '$XL'"; ok=0; }
  [ -n "$SL" ] || { xlog "$1: VIRHE no prefill staging line (levels refused?)"; ok=0; }
  case "$KL" in
    *"in one launch at 0 of"*|*"rope+round trip at 0 ("*|*"not declared at"*|*"at 0 MoE layers"*|"")
      xlog "$1: VIRHE the lever K1 note does not show all three folds declared: '${KL}'"; ok=0 ;;
  esac
  case "$KL" in *"ds41_kvfold=1"*"ds41_ofold=1"*"ds41_onefill=1"*) ;; *) xlog "$1: VIRHE the K1 switches do not start at 1: '${KL}'"; ok=0 ;; esac
  NW=$(docker logs $CN 2>&1 | grep -c 'the acquire rule plays .*RADIANCE_AQL_ACQ=rule-w)'); NR=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_AQL_ACQ=rule)')
  xlog "$1 acquire rule: $NW recorded-pass line(s) with RADIANCE_AQL_ACQ=rule-w, $NR with rule"
  [ "$NW" -gt 0 ] && [ "$NR" = 0 ] || { xlog "$1: VIRHE rule-w is not what the recorded passes play"; ok=0; }
  LL=$(levlits $2); IFS='|' read -ra lit <<< "$LL"
  local found="" miss=""
  for l in "${lit[@]}"; do [ -n "$l" ] || continue
    if docker logs $CN 2>&1 | grep -qF -- "${l//,/ }"; then found="$found ${l%%,*}"; else miss="$miss | ${l//,/ }"; fi; done
  xlog "$1 lever lines: ${#lit[@]} literal(s), found [$(echo $found)]$([ -n "$miss" ] && echo " -- VIRHE missing:$miss") | $(docker logs $CN 2>&1 | grep -E "lever BI declared|lever SD|weight sidecar|lever GPF|GPF-b|lever W1|context skip \(|confidence plane|activation arena|expert prefetch: off|draft lend:" | sed 's/^radiance: //; s/^[A-Z] *//' | cut -c1-240 | sort -u | tr '\n' '|' | cut -c1-2600)"
  [ -z "$miss" ] || ok=0
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
gpftab() {   # gpftab TAG: lever GPF's table line (parsed at the first gg_pf call): all entries, none skipped (logged)
  local n; n=$(echo "$GPTAB" | tr ',' '\n' | grep -c .)
  [ "$DRY" = 1 ] && { echo "+ gpftab $1: 'lever GPF form table from RADIANCE_DS41_GG_PF_TABLE: $n entries', no skipped-entry line"; return 0; }
  local tl sk; tl=$(docker logs $CN 2>&1 | grep -m1 'lever GPF form table from RADIANCE_DS41_GG_PF_TABLE' | sed 's/^radiance: //'); sk=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_DS41_GG_PF_TABLE entry')
  case "$tl" in *": $n entries"*) [ "$sk" = 0 ] && { xlog "$1 GPF table: $tl (all $n)"; return 0; } ;; esac
  xlog "$1 GPF table: VIRHE '${tl:-no table line}' with $sk skipped-entry line(s); the table has $n entries"
}
vramstart() {   # vramstart TAG: the start's VRAM / lend lines -- /stats experts per tier, rank-0 mover / lend lines, the KV line
  [ "$DRY" = 1 ] && { echo "+ GET /stats > $O/$1-stats0.json; RESULT xs3 $1 VRAM at start: experts per tier, rank-0 mover / lend lines, KV line (vs hxzzy-stack2's same start)"; return 0; }
  curl -s -m 30 $URL/stats > $O/$1-stats0.json 2>/dev/null
  xlog "$1 VRAM at start: experts $(python3 -c 'import json,sys
try: s = json.load(open(sys.argv[1]))
except Exception as e: print("n/a (%s)" % type(e).__name__); sys.exit(0)
print({x.get("name"): x.get("units") for x in s.get("experts", {}).get("tiers", [])})' $O/$1-stats0.json 2>&1 | cut -c1-200) (stack2 $(python3 -c 'import json,sys
try: s = json.load(open(sys.argv[1]))
except Exception: print("n/a"); sys.exit(0)
print({x.get("name"): x.get("units") for x in s.get("experts", {}).get("tiers", [])})' $RS2/$2-stats0.json 2>&1 | cut -c1-160)) | $(docker logs $CN 2>&1 | grep -E '^I\[0\] mover:|^I +kv cache|^I kv: the pool|draft lend: set up|weight sidecar:|load: [0-9]+ weight share' | sed 's/^radiance: //; s/^[A-Z]\[0\] //; s/^[A-Z] *//' | cut -c1-220 | sort -u | tr '\n' '|' | cut -c1-1800)"
}
draftlines() { [ "$DRY" = 1 ] && { echo "+ RESULT xs3 $1 drafter lines"; return 0; }
  xlog "$1 drafter / speculation lines: $(docker logs $CN 2>&1 | grep -iE 'dspark|drafter|speculat|draft-window|draft window|draft depth|draft hybrid|coupl' | sed 's/^[A-Z][^ ]* //' | sort -u | head -12 | tr '\n' '|' | cut -c1-1800)"; }
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
fn4code() {   # fn4code JSON: POST, print the HTTP status
  [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null
}
pcreset() {   # pcreset TAG: POST /reset_prefix_cache (loopback operator) so the next prompt is prefilled cold
  [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o $O/$1-reset.json -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  xlog "$1 prefix cache reset: HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"
}
swnote() {   # swnote TAG n0|n5: GET at start (stack2's start values, ds41_bi 0; at n5 draft_sample 0 with the coupling capable);
  # refusals (ds41_kvfold 2, ds41_bi 5; n5 draft_sample 201, n0 draft_sample 1 -- no coupled drafter); each arm's set (A, P, C,
  # and S at n5) applied at a step, then A again
  local s=$2
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 at start (gg_gemv 2, aql_acq 2, K1 three 1, ds41_bi 0$([ $s = n5 ] && echo ", draft_adaptive 1, draft_depth0 0, draft_hybrid 0, draft_capable 1, draft_sample 0, sample.capable 1" || echo ", draft_capable 0"), epoch 0); refusals ds41_kvfold 2 / ds41_bi 5 / draft_sample $([ $s = n5 ] && echo 201 || echo 1) -> 400; arms A $(pinj $s A) P $(pinj $s P) C $(pinj $s C)$([ $s = n5 ] && echo " S $(pinj $s S)") each applied at a step, then A"; return 0; }
  local s0 g c1 c2 c3 a sets="" bad=""
  s0=$(fn4get)
  c1=$(fn4code '{"ds41_kvfold": 2}'); c2=$(fn4code '{"ds41_bi": 5}')
  if [ $s = n5 ]; then c3=$(fn4code '{"draft_sample": 201}'); else c3=$(fn4code '{"draft_sample": 1}'); fi
  for a in A P C $([ $s = n5 ] && echo S) A; do
    fn4set "$(pinj $s $a '{"epoch_bump": 1}')"; nudge; g=$(fn4get)
    python3 -c 'import json,sys
g, w = json.loads(sys.argv[1]), json.loads(sys.argv[2])
sys.exit(0 if all(g["switches"].get(k) == v for k, v in w.items()) and not g["pending"] else 1)' "$g" "$(pinj $s $a)" 2>/dev/null || bad="$bad $a"
    sets="$sets $a:$(echo "$g" | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print("bi=%s,ds=%s,ep=%s" % (w.get("ds41_bi"), w.get("draft_sample"), d.get("epoch")))' 2>/dev/null)"
  done
  xlog "$2 switch note ($1): at start $(echo "$s0" | cut -c1-900) | ds41_kvfold 2 -> HTTP $c1, ds41_bi 5 -> HTTP $c2, draft_sample $([ $s = n5 ] && echo 201 || echo 1) -> HTTP $c3 | arms applied:$sets$([ -n "$bad" ] && echo " -- VIRHE not applied:$bad") | core log: $(docker logs $CN 2>&1 | grep -E 'fn4: switches' | tail -2 | sed 's/^[A-Z] *//' | tr '\n' '|' | cut -c1-400)"
  python3 -c 'import json,sys
s0, n5 = json.loads(sys.argv[1]), sys.argv[2] == "n5"; w = s0["switches"]
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) and w.get("ds41_bi") == 0 and s0["epoch"] == 0
if n5: ok = ok and w["draft_adaptive"] == 1 and w["draft_depth0"] == 0 and w["draft_hybrid"] == 0 and s0["draft_capable"] == 1 and w.get("draft_sample") == 0 and (s0.get("sample") or {}).get("capable") == 1
else: ok = ok and s0["draft_capable"] == 0
sys.exit(0 if ok else 1)' "$s0" $s 2>/dev/null && [ "$c1" = 400 ] && [ "$c2" = 400 ] && [ "$c3" = 400 ] && [ -z "$bad" ]
}
gate1() {   # gate1 TAG LIMIT: restart + ONE check (warm-up + fastcheck); over the limit is logged (identity does not depend on speed;
  # the speed blocks are in-process A/B, so a slow state shows in both arms -- the guard line is next to every result)
  local tag=$1 lim=$2 ms st=fast
  restart $tag || { keeplog $tag-fail; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && return 0
  awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}' && { st="SLOW (not retried)"; keeplog $tag-slow; }
  echo "$st $ms" > $O/guard-$tag.txt
  xlog "guard $tag: $st, 1-user ${ms} ms/step (limit $lim; hxzzy-stack2 st-nst0 13.97, st-stack 33.6)"
}
# ---------------------------------------------------------------- identity at T 0 (dsm-bilid6.sh's alone / loaded / c4rr / cmpx)
alone() {   # alone TAG START ARM NAME: pin + epoch_bump, a prefix-cache reset, greedyset.py's 20 prompts at c1 ALONE -> $O/TAG-NAME.jsonl
  local tag=$1 t0=$(date +%s)
  fn4set "$(pinj $2 $3 '{"epoch_bump": 1}')"; nudge
  pcreset $tag-$4
  run "(cd $D && timeout 900 python3 $GS $URL $O/$tag-$4.jsonl --conc 1 --max-tokens $MT > $O/$tag-$4.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  xlog "$tag $4 (greedy set c1 ALONE, temperature 0, $MT tok, arm $3 $(armj $2 $3)): $(tail -1 $O/$tag-$4.txt | cut -c1-200), $(( $(date +%s) - t0 )) s"
  swcheck $tag $2 $3 $4
}
swcheck() {   # swcheck TAG START ARM NAME: the arm's switches are what GET reports (else VIRHE: the phase did not run it)
  [ "$DRY" = 1 ] && { echo "+ check GET /admin/fn4: $(armj $2 $3) applied"; return 0; }
  local g; g=$(fn4get)
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); w=json.loads(sys.argv[2]); sys.exit(0 if all(d["switches"].get(k) == v for k, v in w.items()) else 1)' "$g" "$(armj $2 $3)" 2>/dev/null \
    || xlog "$1 $4: VIRHE $(armj $2 $3) not applied ($(echo "$g" | cut -c1-400)) -- this phase did not run it"
}
loaded() {   # loaded TAG START ARM NAME: pin + epoch_bump, a prefix-cache reset, loadid.py run (mb c4 + per probe one fresh prefill)
  local tag=$1 t0=$(date +%s) j nm=$4
  fn4set "$(pinj $2 $3 '{"epoch_bump": 1}')"; nudge
  pcreset $tag-$nm
  run "(cd $D && timeout 2700 python3 -I $LD run --url $URL --gs $GS --mb $MBP --out $O/$tag-$nm.jsonl --conc $CONC --warm $WARM --offset $OFF --pf-tok $PFT --max-tokens $MT > $O/$tag-$nm.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  swcheck $tag $2 $3 $nm
  j=$(grep -m1 '^LOADID ' $O/$tag-$nm.txt | cut -c8-)
  [ -n "$j" ] || { xlog "$tag $nm: VIRHE no LOADID line ($(tail -3 $O/$tag-$nm.txt | tr '\n' ' ' | cut -c1-400))"; return 1; }
  xlog "$tag $nm (the 20 prompts each under load: mb c4 + one fresh ~${PFT}-token prefill at +${OFF} s, temperature 0, $MT tok, arm $3): $(python3 -c 'import json,sys
d = json.loads(sys.argv[1]); w = d.get("window") or {}
print("%s of %s answered, probe %s s mean | prefills %s tok mean, %s of %s cold, TTFT %s s mean | in flight %s %% (median) of a probe | mixed steps a probe min %s, median %s; probes without one %s | c4 %s requests, %s errors%s" % (
    d["answered"], d["of"], d["probe_s_mean"], d["pf_prompt_mean"], d["pf_cold"], d["of"], d["pf_ttft_mean"],
    round(100 * (d["ov_frac_median"] or 0)), d["mixed_min"], d["mixed_median"], d["probes_without_mixed"] or "none",
    d["c4_requests_done"], d["c4_errors"], (" | VIRHE errors: %s" % d["errors"]) if d.get("errors") else ""))' "$j" 2>&1 | cut -c1-900), $(( $(date +%s) - t0 )) s"
}
c4rr() {   # c4rr TAG START ARM NAME: pin + epoch_bump, a prefix-cache reset, greedyset.py's 20 prompts at conc 4
  local tag=$1 t0=$(date +%s)
  fn4set "$(pinj $2 $3 '{"epoch_bump": 1}')"; nudge
  pcreset $tag-$4
  run "(cd $D && timeout 900 python3 $GS $URL $O/$tag-$4.jsonl --conc 4 --max-tokens $MT > $O/$tag-$4.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  xlog "$tag $4 (greedy set at conc 4, temperature 0, $MT tok, arm $3): $(tail -1 $O/$tag-$4.txt | cut -c1-200), $(( $(date +%s) - t0 )) s"
}
cmpx() {   # cmpx TAG NAME A B: loadid.py cmp (byte for byte + the first differing char / token via /tokenize) -> $O/TAG-cmp-NAME.json
  local r
  [ "$DRY" = 1 ] && { echo "+ python3 -I $LD cmp --url $URL $3 $4 --json $O/$1-cmp-$2.json"; return 0; }
  if [ ! -s "$3" ] || [ ! -s "$4" ]; then
    xlog "$1 $2: not compared (missing:$([ -s "$3" ] || echo " $3")$([ -s "$4" ] || echo " $4"))"; echo "$1 $2 not compared" >> $O/identity.txt; return 0
  fi
  r=$(timeout 300 python3 -I $LD cmp --url $URL --json $O/$1-cmp-$2.json --label "$1 $2" "$3" "$4" 2>&1 | grep -m1 '^CMPLINE ' | cut -c9-)
  echo "$1 $2 ${r:-compare failed}" >> $O/identity.txt
  xlog "$1 $2: ${r:-VIRHE compare failed} ($(basename $3) vs $(basename $4))"
}
copyt() {   # copyt TAG START ARM: the 28k copy after a ~31k-token cold prompt under the arm, vs bz d0
  local tag=$1 r t0=$(date +%s)
  fn4set "$(pinj $2 $3 '{"epoch_bump": 1}')"; nudge
  pcreset $tag-copy
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset 0 --tag=-$tag --outdir $O > $O/$tag-copy.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  r=$(cmpfile $O/copy-$tag-run1.txt $RD/copy-dsd-d0-i1-run1.txt)
  echo "$tag copy-$3 $r" >> $O/identity.txt
  xlog "$tag 28k copy under $3 vs bz d0 (NST 0, stock): $r$(case $r in IDENTICAL*) ;; *) echo " -- reported, not hidden" ;; esac) | $(grep '^J ' $O/$tag-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact")))' 2>&1 | cut -c1-200) | $(thermline $t0 $(date +%s))"
}
deep() {   # deep TAG START ARM: the 136k cold prefill (deepq.py's fixed prompt, 512 greedy tokens) -> TTFT + its answer vs dsm-225
  local t0=$(date +%s) j r
  fn4set "$(pinj $2 $3 '{"epoch_bump": 1}')"; nudge
  pcreset $1-deep
  run "(cd $D && timeout 1800 python3 $DQ $O/deep-$1.txt 135000 512 > $O/deep-$1.out 2>&1)"
  [ "$DRY" = 1 ] && { echo "+ RESULT xs3 $1 136k cold prefill: TTFT (cached must be 0) vs dsm-225 and hxzzy-stack2; answer vs $RM/deep-dsm-225.txt"; return 0; }
  j=$(grep -m1 '^DEEPQ ' $O/deep-$1.out | cut -c7-)
  [ -n "$j" ] || { xlog "$1 136k cold prefill: VIRHE no answer ($(tail -2 $O/deep-$1.out | tr '\n' ' ' | cut -c1-300))"; return 1; }
  echo "$j" > $O/deep-$1.json
  xlog "$1 136k cold prefill under $3 (deepq.py fixed prompt, 512 greedy tokens): $(python3 -c 'import json,sys
d = json.loads(sys.argv[1])
def js(p):
    try: return json.load(open(p))
    except Exception: return {}
r, s2 = js(sys.argv[2]), js(sys.argv[3])
t, rt, st = d.get("ttft"), r.get("ttft"), s2.get("ttft")
pc = lambda a, b: ", %+.1f %%" % ((a / b - 1) * 100) if a and b else ""
print("status %s, prompt %s tok, cached %s%s, TTFT %s s (dsm-225 %s s%s; hxzzy-stack2 same start %s s%s, cross-start), %s tok, %s chars" % (
    d.get("status"), d.get("prompt_tokens"), d.get("cached"), "" if d.get("cached") == 0 else " -- VIRHE not cold",
    t, rt, pc(t, rt), st, pc(t, st), d.get("tokens"), d.get("chars")))' "$j" $RM/deep-dsm-225.txt.json $RS2/deep-$4.json 2>&1 | cut -c1-400) | $(thermline $t0 $(date +%s))"
  r=$(cmpfile $O/deep-$1.txt $RM/deep-dsm-225.txt)
  echo "$1 deep-$3 $r" >> $O/identity.txt
  xlog "$1 136k deep answer (cold) vs dsm-225 (NST 0, stock, cold): $r$(case $r in IDENTICAL*) ;; *) echo " -- reported, not hidden" ;; esac)"
}
# ---------------------------------------------------------------- smoke, speed blocks
smoke() {   # smoke TAG START: under C first -- the greedy set at conc 4 (mixed steps), at n5 also mb c4 20 s at T 1 (the coupled path)
  local tag=$1 s=$2 t0=$(date +%s) n e=""
  fn4set "$(pinj $s C '{"epoch_bump": 1}')"; nudge
  run "(cd $D && timeout 600 python3 $GS $URL $O/$tag-smoke.jsonl --conc 4 --max-tokens 32 > $O/$tag-smoke.txt 2>&1)"
  [ $s = n5 ] && run "(cd $D && timeout 600 python3 mb.py --tag xs3-smoke-$tag --phases c4 --dur 20 --warm 5 > $O/$tag-smoke-t1.txt 2>&1)"
  [ "$DRY" = 1 ] && { echo "+ smoke $tag: C $(armj $s C): greedy set conc 4 (32 tok) 20 of 20$([ $s = n5 ] && echo " + mb c4 20 s at T 1 with 0 errors"), /health 200"; return 0; }
  swcheck $tag $s C smoke
  n=$(python3 -c 'import json,sys; print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l).get("text") is not None))' $O/$tag-smoke.jsonl 2>/dev/null || echo 0)
  [ $s = n5 ] && e=$(grep -m1 '^MB ' $O/$tag-smoke-t1.txt 2>/dev/null | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[3:]); print("%s requests %s errors, %s tok/s" % (d.get("requests"), d.get("errors"), d.get("agg_dec_tps")))' 2>/dev/null || echo "no MB line")
  if [ "$n" = 20 ] && { [ $s = n0 ] || case "$e" in *" 0 errors"*) true ;; *) false ;; esac; } && curl -s -m 10 -o /dev/null -w '%{http_code}' $URL/health | grep -q 200; then
    xlog "$tag smoke (C $(armj $s C)): greedy set conc 4 20 of 20$([ $s = n5 ] && echo ", mb c4 T 1: $e"), healthy, $(( $(date +%s) - t0 )) s | $(docker logs $CN 2>&1 | grep 'lever BI stats' | tail -1 | sed 's/^radiance: //' | cut -c1-200) | SD $(fn4get | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin).get("sample")))' 2>/dev/null | cut -c1-200)"
    return 0
  fi
  xlog "$tag smoke (C): VIRHE $n of 20 answered$([ $s = n5 ] && echo ", mb c4 T 1: $e"), or unhealthy | $(docker logs $CN 2>&1 | grep -E '^E |step fail|refus' | tail -3 | tr '\n' ' ' | cut -c1-500)"
  keeplog $tag-smoke
  return 1
}
blocks() {   # blocks TAG START KIND: KIND t0 = the T 0 A/B (XS3_ORDER over A / P / C; mb at temperature 0), t1 = the T 1 A/B (XS3_T1ORDER
  # over A / S; sdprobe sampled + mb at the server default) -- s3blocks.py, read only; its BLOCK / STATS lines logged
  local tag=$1 s=$2 k=$3 args ord ph mbx="" a
  if [ $k = t0 ]; then ord=$ORDER; ph=$([ $s = n5 ] && echo $PH5 || echo $PH0); mbx="--mbtemp 0 --mbt $MBT"
  else ord=$T1ORDER; ph=t1,c4,c8; fi
  [ -n "$ord" ] || return 0
  args=""; for a in $(echo $ord | grep -o . | awk '!s[$0]++'); do args="$args --arm '$a=$(armj $s $a)'"; done
  run "timeout 9000 python3 -I $GGB --url $URL --mb-dir $D --edit $EP --sdprobe $PRB --seed $SEED --label $tag-$k $args --order $ord --phases $ph --dur $DUR --warm $WARMS $mbx --fixed '$(fixof $s)' --out $O/$tag-$k-blocks.jsonl > $O/$tag-$k-blocks.txt 2>&1"
  [ "$DRY" = 1 ] && return 0
  while IFS= read -r l; do xlog "$tag $k ${l:0:1700}"; done < <(grep -E '^(BLOCK|STATS|GGB \{"error)|Traceback' $O/$tag-$k-blocks.txt)
  xlog "$tag $k engine: $(docker logs $CN 2>&1 | grep 'lever BI stats' | tail -1 | sed 's/^radiance: //' | cut -c1-260) (cumulative) | SD $(fn4get | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin).get("sample")))' 2>/dev/null | cut -c1-260) | $(docker logs $CN 2>&1 | grep -cE '^E ') E lines"
}
startsum() {   # startsum TAG: the start's identity compares in one line
  [ "$DRY" = 1 ] && { echo "+ RESULT xs3 $1 identity summary: each compare (n differ of 20, first diff char / token)"; return 0; }
  xlog "$1 identity summary: $(python3 - "$O" "$1" <<'PY' 2>&1 | head -c 2500
import glob, json, os, sys
o, tag = sys.argv[1], sys.argv[2]
parts = []
for f in sorted(glob.glob(os.path.join(o, tag + "-cmp-*.json"))):
    n = f[len(os.path.join(o, tag + "-cmp-")):-5]
    try: d = json.load(open(f))
    except Exception: parts.append("%s: not compared" % n); continue
    parts.append("%s: %s" % (n, "IDENTICAL (%d of %d)" % (d["n"], d["n"]) if d["identical"] else "%d of %d differ [%s]" % (d["n_diff"], d["n"],
                 ", ".join("p%d tok %s" % (e["i"], "-" if e["tok"] is None else e["tok"]) for e in d["diff"][:10]))))
print(" | ".join(parts) or "none")
PY
)"
}
dstart() {   # dstart n0|n5: one start: checks, smoke, T 0 blocks, (n5) T 1 blocks, 136k, identity, summary line
  local s=$1 tag lim ok v first="" s2
  stackenv $s; export MODEL=$M225; HP29=19456; best_vars
  case $s in
    n0) tag=s3-n0; lim=30; s2=st-nst0; export MNBT=4160 NST=0 XARGS="$XARGS --live" ;;
    n5) tag=s3-n5; lim=70; s2=st-stack; export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on" ;;
  esac
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV > $XS2_ENVFILE"; else printf '%s\n' $LEVENV > $XS2_ENVFILE; fi
  xlog "$tag: start ($([ $s = n0 ] && echo "$DESC_N" || echo "$DESC_5")) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $(fixof $s) | arms A $(armj $s A) P $(armj $s P) C $(armj $s C)$([ $s = n5 ] && echo " S $(armj $s S)") | env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ' | cut -c1-2000)"
  gate1 $tag $lim || { xlog "$tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  modes $tag $s || { xlog "$tag: VIRHE the modes are not stack3's -> start stopped"; down; return 1; }
  draftlines $tag
  if [ "$DRY" != 1 ]; then
    ok=1
    if [ $s = n5 ]; then
      docker logs $CN 2>&1 | grep -q "DSpark drafter" || { xlog "$tag: VIRHE no DSpark drafter line"; ok=0; }
      docker logs $CN 2>&1 | grep -q "draft depth: adaptive" || { xlog "$tag: VIRHE no adaptive draft depth line"; ok=0; }
    else
      docker logs $CN 2>&1 | grep -qiE "DSpark drafter|'dspark' proposes" && { xlog "$tag: VIRHE a DSpark drafter line at NST 0"; ok=0; }
    fi
    [ $ok = 1 ] || { keeplog $tag-nodraft; down; return 1; }
  fi
  vramstart $tag $s2
  swnote $tag $s || { xlog "$tag: VIRHE the switches are not as expected -> start stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  smoke $tag $s || { xlog "$tag: VIRHE the smoke under C failed -> start stopped"; down; return 1; }
  stopfile || blocks $tag $s t0
  [ $s = n5 ] && ! stopfile && blocks $tag $s t1
  stopfile || deep $tag $s P $s2
  gpftab $tag
  # identity at T 0 (after the speed blocks and the 136k: the loads do not heat a speed number)
  local IDV; IDV=$([ $s = n5 ] && echo "$IDVALS5" || echo "$IDVALS0")
  stopfile || alone $tag $s A a1-A
  for v in $IDV; do
    stopfile && break
    [ -n "$first" ] || first=$v
    alone $tag $s $v a1-s$v
    stopfile || loaded $tag $s $v ld-s$v
    stopfile || c4rr $tag $s $v c4a-s$v
    stopfile || c4rr $tag $s $v c4b-s$v
  done
  [ -n "$first" ] && ! stopfile && alone $tag $s $first a2
  stopfile || copyt $tag $s P
  cmpx $tag a1-A-vs-d0 $RD/dsd-d0-gs-c1.jsonl $O/$tag-a1-A.jsonl
  for v in $IDV; do
    cmpx $tag a1-s$v-vs-d0 $RD/dsd-d0-gs-c1.jsonl $O/$tag-a1-s$v.jsonl
    cmpx $tag a1-s$v-vs-a1-A $O/$tag-a1-A.jsonl $O/$tag-a1-s$v.jsonl
    cmpx $tag ld-vs-a1-s$v $O/$tag-a1-s$v.jsonl $O/$tag-ld-s$v.jsonl
    cmpx $tag c4b-vs-c4a-s$v $O/$tag-c4a-s$v.jsonl $O/$tag-c4b-s$v.jsonl
    cmpx $tag c4a-vs-a1-s$v $O/$tag-a1-s$v.jsonl $O/$tag-c4a-s$v.jsonl
  done
  [ -n "$first" ] && cmpx $tag a2-vs-a1-s$first $O/$tag-a1-s$first.jsonl $O/$tag-a2.jsonl
  startsum $tag
  run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
  down
  xlog "$tag: done"
}
summary() {   # summary: per start / kind / arm / metric mean +- t95; vs A in process; vs hxzzy-stack2 (cross-start); targets; identity
  [ "$DRY" = 1 ] && { echo "+ RESULT xs3 summary: per start (n0 / n5) and kind (T 0 A/P/C, T 1 A/S): metric mean +- t95, P / C / S vs A (pooled; T 1 pairs), vs hxzzy-stack2 same start (cross-start), targets (1u 90, c4 160, c8 200, 136k 35 s), identity verdict"; return 0; }
  python3 - "$O" "$RS2" "$RM/deep-dsm-225.txt.json" "$T1BI" > $O/summary.txt 2>&1 <<'PY'
import json, math, os, statistics, sys
o, rs2, dref, t1bi = sys.argv[1:5]
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
def rows(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except OSError: return []
def js(p):
    try: return json.load(open(p))
    except Exception: return {}
def num(x): return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None
def g(*path):
    def f(res):
        x = res
        for k in path:
            if not isinstance(x, dict): return None
            x = x.get(k)
        return num(x)
    return f
def ci(v):
    if not v: return None, None
    m = statistics.mean(v)
    return m, (T95.get(len(v) - 1, 1.96) * statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else None)
def vals(rs, arm, get): return [v for v in (get(r["res"]) for r in rs if r.get("ok") and r.get("arm") == arm) if v is not None]
def f1(x, n=1): return "-" if x is None else format(x, ".%df" % n)
def pct(v, rv): return ("%+.1f %%" % ((v / rv - 1) * 100)) if v and rv else "-"
def tgt(v, t, low=False):
    if t is None or v is None: return ""
    if low: return "; target <= %g s: %s" % (t, "MET" if v <= t else "%.1f s over" % (v - t))
    return "; target >= %g: %s" % (t, "MET (+%.1f)" % (v - t) if v >= t else "%.1f short (+%.1f %% needed)" % (t - v, (t / v - 1) * 100))
# (name, getter, target, stack2 getter or None) -- stack2's res: ggblocksdz2 (c1 essay / code, edit at T 0; c4 / c8 mb at the server default)
T0M = [("1u essay T0", g("c1", "essay", "decode_tps"), 90.0, g("c1", "essay", "decode_tps")),
       ("1u code T0", g("c1", "code", "decode_tps"), 90.0, g("c1", "code", "decode_tps")),
       ("1u edit T0", g("edit", "decode_tps"), 90.0, g("edit", "decode_tps")),
       ("mb c1 T0", g("mbc1", "agg_dec_tps"), 90.0, None),
       ("c4 T0", g("c4", "agg_dec_tps"), 160.0, None), ("c4 p50 T0", g("c4", "per_stream_dec_p50"), None, None),
       ("c8 T0", g("c8", "agg_dec_tps"), 200.0, None), ("c8 p50 T0", g("c8", "per_stream_dec_p50"), None, None),
       ("pf prefill tok/s T0", g("pf", "prefill_tps"), None, None), ("pf decode kept T0", g("pf", "decode_kept"), None, None),
       ("pf ms/step during T0", g("pf", "ms_step_during"), None, None)]
T1M = [("1u essay T1", g("t1", "essay", "decode_tps"), 90.0, None), ("1u code T1", g("t1", "code", "decode_tps"), 90.0, None),
       ("1u essay T1 tok/step", g("t1", "essay", "tokens_per_step"), None, None), ("1u code T1 tok/step", g("t1", "code", "tokens_per_step"), None, None),
       ("c4 T1", g("c4", "agg_dec_tps"), 160.0, g("c4", "agg_dec_tps")), ("c4 p50 T1", g("c4", "per_stream_dec_p50"), None, g("c4", "per_stream_dec_p50")),
       ("c4 tok/step T1", g("c4", "tok_per_decode_step"), None, g("c4", "tok_per_decode_step")),
       ("c8 T1", g("c8", "agg_dec_tps"), 200.0, g("c8", "agg_dec_tps")), ("c8 p50 T1", g("c8", "per_stream_dec_p50"), None, g("c8", "per_stream_dec_p50")),
       ("c8 tok/step T1", g("c8", "tok_per_decode_step"), None, g("c8", "tok_per_decode_step"))]
best = {}   # metric -> [(label, value)]
out = []
for tag, s2, s2arm in (("s3-n0", "st-nst0", "N"), ("s3-n5", "st-stack", "A")):
    ref2 = rows(os.path.join(rs2, s2 + "-blocks.jsonl"))
    for kind, M, arms in (("t0", T0M, "APC"), ("t1", T1M, "AS")):
        rs = rows(os.path.join(o, "%s-%s-blocks.jsonl" % (tag, kind)))
        if not rs: continue
        present = [a for a in arms if any(r.get("arm") == a for r in rs)]
        parts = []
        for nm, get, t, g2 in M:
            mA = ci(vals(rs, "A", get))[0]
            cells = []
            for a in present:
                v = vals(rs, a, get); m, h = ci(v)
                if m is None: continue
                c = "%s %s%s (n %d)" % (a, f1(m, 2 if m < 10 else 1), "" if h is None else " +- " + f1(h, 2 if m < 10 else 1), len(v))
                if a != "A" and mA: c += " %s vs A" % pct(m, mA)
                if kind == "t1" and a != "A":
                    pr = []
                    for j in range(0, len(rs) - 1, 2):
                        x, y = rs[j], rs[j + 1]
                        if {x.get("arm"), y.get("arm")} != {"A", a} or not (x.get("ok") and y.get("ok")): continue
                        ra, rb = (x, y) if x["arm"] == "A" else (y, x)
                        ga, gb = get(ra["res"]), get(rb["res"])
                        if ga and gb is not None: pr.append(gb / ga)
                    if pr: c += " (pairs %s)" % "/".join("%+.1f" % (100 * (q - 1)) for q in pr)
                if g2 is not None:
                    v2 = [x for x in (g2(r["res"]) for r in ref2 if r.get("ok") and r.get("arm") == s2arm) if x is not None]
                    if v2: c += " [stack2 %s %s, %s, cross-start]" % (s2arm, f1(statistics.mean(v2)), pct(m, statistics.mean(v2)))
                if t: c += tgt(m, t)
                cells.append(c)
                if t and a != "A": best.setdefault(nm.replace(" T0", "").replace(" T1", " T1"), []).append(("%s %s %s" % (tag[3:], kind, a), m))
            if cells: parts.append("%s: %s" % (nm, " | ".join(cells)))
        bad = sum(1 for r in rs if not r.get("ok"))
        line = "%s %s (%d blocks, %d flagged): %s" % (tag, kind, len(rs), bad, " || ".join(parts))
        out.append(line); print("ARMS " + line)
    d = js(os.path.join(o, "deep-%s.json" % tag)); r = js(dref); r2 = js(os.path.join(rs2, "deep-%s.json" % s2))
    t = num(d.get("ttft")) if d.get("cached") == 0 else None
    if d:
        l = "%s 136k cold TTFT %s s (dsm-225 %s s, %s; hxzzy-stack2 %s s, %s, cross-start)%s" % (tag, f1(t, 2), f1(num(r.get("ttft")), 2), pct(t, num(r.get("ttft"))),
            f1(num(r2.get("ttft")), 2), pct(t, num(r2.get("ttft"))), tgt(t, 35.0, True))
        out.append(l); print("PREFILL " + l)
        if t: best.setdefault("136k TTFT s", []).append((tag[3:], t))
bl = []
for nm, vs in best.items():
    low = nm.startswith("136k")
    k, v = (min if low else max)(vs, key=lambda x: x[1])
    tt = 35.0 if low else (90.0 if nm.startswith("1u") or nm.startswith("mb c1") else 160.0 if nm.startswith("c4") else 200.0)
    bl.append("%s best %s %s%s" % (nm, k, f1(v, 2 if low else 1), tgt(v, tt, low)))
try: ident = [l.strip() for l in open(os.path.join(o, "identity.txt")) if l.strip()]
except OSError: ident = []
badid = [l for l in ident if not ("IDENTICAL" in l and "differ" not in l)]
verdict = ("IDENTICAL at every compare (%d)" % len(ident)) if ident and not badid else ("NOT IDENTICAL: " + "; ".join(badid)[:900] if ident else "none recorded")
print("SUMMARY stack arms (P = ds41_bi 2, C = ds41_bi 4, S = ds41_bi %s; at n5 + draft_sample 1; DLW sidecar q8_0 in every n5 arm): %s || identity: %s || %s" % (
    t1bi, " | ".join(bl) or "-", verdict, " || ".join(out)))
PY
  while IFS= read -r l; do case "$l" in SUMMARY*) ;; *) xlog "${l:0:3000}" ;; esac; done < $O/summary.txt
  xlog "summary: $(grep -m1 '^SUMMARY ' $O/summary.txt | cut -c9- | cut -c1-6000)"
}
trap 'down' EXIT
trap 'exit 143' TERM INT
# ---------------------------------------------------------------- main
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) starts [$STARTS] | T 0 order $ORDER (n0 phases $PH0, n5 $PH5; mb at temperature 0, dur $DUR warm $WARMS) | T 1 order ${T1ORDER:-none} (n5: S = ds41_bi $T1BI + draft_sample 1; sdprobe seed $SEED thinking on + mb c4 / c8 server default) | identity T 0: n0 values [$IDVALS0], n5 [$IDVALS5] | DLW sidecar $SIDE (n5, every arm) | n0 pin $FIX0 | n5 pin $FIX5 | env n0 [$(levenv n0)] | env n5 [$(levenv n5)] | refs hxzzy-stack2 ($RS2, cross-start), bz d0, dsm-225 | targets 1u >= 90, c4 >= 160, c8 >= 200 tok/s, 136k <= 35 s"
# the image, on this host, read only (in DRYRUN too: docker image inspect + a no-device, no-network grep of its files; -z as dsm-stack2.sh):
# the int4 registry (29 switches: ..., dspark_walk, ds41_bi, draft_sample, epoch_bump), stack2's literals, levers BI / SD / DLW
# (B bin/radiance, A deepseek41.so, K libr4d.so, S spechub.so, X any of B / A / K).
IMGLITS="B|dspark_conf_rows, gg_pf, f32_pf, dspark_walk, ds41_bi, draft_sample, epoch_bump
B|ds41_onefill, draft_depth0, draft_probe, draft_hybrid,
B|draft depth 0: a step drafts nothing
B|draft hybrid: a sequence whose prompt lookup
B|reset_prefix_cache
B|bytes win where no snapshot rests on the first copy
B|/tokenize
B|radiance:engine_steps_mixed_total
A|RADIANCE_DS41_KVFOLD
K|rmsnorm_rope_roundtrip_kv_store
K|r4d_gg_mr_pin
K|RADIANCE_DS41_DRAFTER_MX
K|RADIANCE_DS41_GG_PF_TABLE
K|lever GPF form table from RADIANCE_DS41_GG_PF_TABLE
K|RADIANCE_DS41_F32_PF
A|RADIANCE_DS41_DSPARK_WALK
K|dspark_walk_step
A|RADIANCE_DS41_ARGATHER_MAX
A|RADIANCE_DS41_QNFOLD
A|RADIANCE_DS41_SWFOLD
A|RADIANCE_DS41_DSPARK_CONF
B|RADIANCE_DRAFT_CTXSKIP
B|a confidence plane (the dspark_conf gate
X|RADIANCE_DRAFT_LEND
X|RADIANCE_DRAFT_LEND_BORROW
A|RADIANCE_DS41_ARENA_DECODE_ROWS
A|RADIANCE_DS41_ROUTER_BF16
X|RADIANCE_DS41_EMB_RAM
X|RADIANCE_PLACE_SHADOW_SLOTS
X|RADIANCE_XPREFETCH
A|RADIANCE_DS41_BI
A|lever BI declared (RADIANCE_DS41_BI, fn4 ds41_bi, now
A|lever BI stats
B|RADIANCE_DRAFT_SAMPLE
B|lever SD: fn4 draft_sample
K|dspark_walk_sample_step
A|dspark_walk_sample_step
A|mtp_couple_qtemp
A|DSpark lever SD: the sampled full-vocabulary walk declared
S|mtp_couple_verify
B|RADIANCE_WEIGHT_SIDECAR
B|weight sidecar: %lld entries
B|read from the weight sidecar"
docker image inspect $IMAGE > /dev/null 2>&1 || { xlog "VIRHE image $IMAGE not on this host"; exit 1; }
IMGMISS=$(echo "$IMGLITS" | timeout 120 docker run --rm -i --network none --entrypoint sh $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels/libr4d.so; S=/opt/radiance/share/radiance/kernels/spechub.so; n=0
while IFS= read -r l; do [ -n "$l" ] || continue; n=$((n + 1)); w=${l%%|*}; s=${l#*|}
  case $w in B) f=$B ;; A) f=$A ;; K) f=$K ;; S) f=$S ;; *) f="$B $A $K" ;; esac
  o=0; for x in $f; do grep -qzF -- "$s" $x && { o=1; break; }; done
  [ $o = 1 ] || echo "MISS $w|$s"; done
test -x /opt/radiance/bin/rad-sidecar || echo "MISS bin/rad-sidecar"; echo "CHECKED $n"' 2>&1)
case "$IMGMISS" in "CHECKED "*) ;; *) xlog "VIRHE $IMAGE is not the int4 stack image -- missing: $(echo "$IMGMISS" | grep -v '^CHECKED' | tr '\n' ';' | cut -c1-900)"; exit 1 ;; esac
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)) is the int4 stack image: all $(echo "$IMGMISS" | sed 's/CHECKED //') literals present (the 29-switch registry with ds41_bi + draft_sample, stack2's, BI / SD / DLW) + bin/rad-sidecar -- OK"
# EVERY HELPER RESOLVES (DRYRUN and real): files present and parsing, the python helpers' selftest / --help, every function
# defined (type -t), tools on PATH, the sidecars' sha256, the yaml (env_file, stack names, docker compose config). Any miss: VIRHE.
HMISS=""
for f in "$LIB29" "$LIBPR" $Y $GS $LD $GGB $MBT $PRB $EP $DQ $MBP $D/stepprobe.py $D/ds41_tests.py $D/fastcheck.py $T/evict-exl3.py; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $GS $LD $GGB $MBT $PRB $EP $DQ $MBP $D/stepprobe.py $D/ds41_tests.py $D/fastcheck.py $T/evict-exl3.py 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
ST=$(timeout 60 python3 -I $LD selftest --gs $GS --mb $MBP 2>&1 | tail -1); case "$ST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$LD(${ST:0:300})" ;; esac
MST=$(timeout 60 python3 -I -B $MBT --mb $MBP --selftest 2>&1 | tail -1); case "$MST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$MBT(${MST:0:300})" ;; esac
python3 -I -B $GGB --help > /dev/null 2>&1 || HMISS="$HMISS run:s3blocks.py--help"
(cd $D && python3 -I -B $PRB --help > /dev/null 2>&1) || HMISS="$HMISS run:sdprobe.py--help"
FNS="log run mkd restart check down evict stopfile best_vars desc serve_args xlog vf levenv levlits armj fixof pinj stackenv keeplog thermline cmpfile
     identw vrammodes modes gpftab vramstart draftlines fn4get fn4set nudge fn4code pcreset swnote gate1 alone swcheck loaded c4rr cmpx copyt deep smoke blocks startsum dstart summary"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee sha256sum; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
for s in n0 n5; do for a in A P C S; do python3 -c 'import json,sys; json.loads(sys.argv[1])' "$(pinj $s $a)" 2>/dev/null || HMISS="$HMISS json:$s-$a"; done; done
if [ -s $MODELDIR/$SIDECAR ] && [ -s $MODELDIR/$SIDECAR.sha256 ]; then
  [ "$(sha256sum $MODELDIR/$SIDECAR | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR(sha256 differs from .sha256)"
else HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR(+.sha256)"; fi
case " $STARTS " in *" n5 "*) [ "$(sha256sum $MODELDIR/$SIDE 2>/dev/null | cut -c1-64)" = "$SIDESHA" ] || HMISS="$HMISS sha256:$MODELDIR/$SIDE(want ${SIDESHA:0:16})" ;; esac
grep -q 'XS2_ENVFILE' $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no env_file \${XS2_ENVFILE})"
for v in RADIANCE_DS41_GG_GEMV RADIANCE_AQL_ACQ RADIANCE_DS41_KVFOLD RADIANCE_DS41_OFOLD RADIANCE_DS41_ONEFILL RADIANCE_PLACE_TURNOVER; do grep -q "^ *$v:" $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(does not pass $v)"; done
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config refuses it)"
HMISS=$(echo $HMISS)
[ -z "$HMISS" ] || { xlog "VIRHE helpers / inputs do not resolve: $HMISS"; exit 1; }
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing, loadid.py $ST, mbt.py $MST, s3blocks.py / sdprobe.py --help, all $(echo $FNS | wc -w) functions resolve (type -t), tools on PATH, sidecars $SIDECAR + $SIDE sha256 OK, yaml $(basename $Y) (env_file \${XS2_ENVFILE}) accepted by docker compose config -- OK"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { xlog "VIRHE $M225 missing or not verified"; exit 1; }
  for f in $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $RM/deep-dsm-225.txt $RM/deep-dsm-225.txt.json $WARM_REF_225 $RS2/st-stack-blocks.jsonl $RS2/st-nst0-blocks.jsonl; do
    [ -s $f ] || xlog "note -- reference $f missing (not compared)"; done
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDE $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $RM/deep-dsm-225.txt $RM/deep-dsm-225.txt.json $WARM_REF_225 \
           $RS2/st-stack-blocks.jsonl $RS2/st-nst0-blocks.jsonl $RS2/deep-st-stack.json $RS2/deep-st-nst0.json $RS2/st-stack-stats0.json $RS2/st-nst0-stats0.json; do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  for s in n0 n5; do echo "+ $s pins: A $(pinj $s A) | P $(armj $s P) | C $(armj $s C) | S $(armj $s S)"; done
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(a radiance container running now is information here; the real run refuses to start beside one)"
fi
mkd $O $O/guard $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt
evict
for s in $STARTS; do stopfile && break; dstart $s; done
summary
unset $SWITCHES
down
xlog "$ITEM: done -- $O"
