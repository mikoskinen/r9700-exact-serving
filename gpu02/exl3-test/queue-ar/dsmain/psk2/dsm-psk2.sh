#!/bin/bash
# dsm-psk2.sh (11.11.2026, queue-ar/dsmain/psk2, ds-main-port PSK design-2 builder) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r00zz-psk2.sh. Log tag "RESULT xpsk2 ", out runs/xdsm/psk2 (XP2_OUT). = dsm-psk.sh 841ce920 (r00z's body, the
# same start and helpers; that file is not edited) with design 2's arms, hit sets, staging / snapshot counters and summary.
# LEVER PSK DESIGN 2 (anchors; coordinator + ds-main-port 11.11.): layers 21-39 run only the cones of a request's targets --
# the prompt's end, its tail snapshot, every ds41_psk_k-th checkpoint grid point (2048 + (i - 1) x 4096, i = K, 2K, ...),
# checkpoint_at -- and a chunk no cone reaches runs and stages none of them; window snapshots only at targets, so a prefix hit
# between anchors resumes at the anchor below. Image XP2_IMAGE = ds41-psk2 (ds41-psk 4d4da8e + design 2: fn4 ds41_psk 2 / 3,
# ds41_psk_k; unset and ds41_psk 0 / 1 = design 1's image, hostcheck). Estimate (agent, from r00z: design 1 136k 43.5 s, its cut
# layers staging-bound ~24 ms a layer-chunk): K8 ~31 s, K4 ~33 s at 136k.
# ONE NST 5 START = r00z's (the z0occ2 start + RADIANCE_DS41_PSK=0), in process. ARMS: A = ds41_psk 0 (today), D = ds41_psk 1
# (design 1), 4 = ds41_psk 2 + ds41_psk_k 4, 8 = ds41_psk 2 + ds41_psk_k 8 (each pin also sets ds41_psk_k; A / D take 8).
# Every probe at temperature 0, 512 greedy tokens, a prefix-cache reset before each cold prompt, NO EVICT (in-process A/B; r00z's
# ABBA pairs agreed to 0.1 %: no page-cache effect).
#   gate1 -> modes -> VRAM -> switch note (ds41_psk 0 / ds41_psk_k 8 at start; ds41_psk 4 and ds41_psk_k 0 refused; every arm
#   applied) -> warm-up vs stock -> smoke under 8 (greedy set conc 4: mixed steps with planned prefill segments, no VIRHE line)
#   -> COLD 136k (deepq.py's fixed prompt, 136216 tokens) in XP2_ORDER (A D 4 8 8 4 D A): TTFT, /stats deltas per run
#   (checkpoints written, staged bytes all ranks, rank-0 staged layers, all-reduce bytes), the engine's design-2 line; identity
#   136k vs dsm-225 and vs the first A. Right after the SECOND cold run of 8, 4 and A (cache = that arm's snapshots): the HIT
#   SET, no reset: deepq at 90000 / 58000 / 20000 (non-anchor for both K: A resumes at 88064 / 55296 / 18432, K4 at 79872 /
#   47104 / 14336, K8 at 63488 / 30720 / 0) and the 136k again (the tail snapshot, 133120): TTFT, cached; identity K hit vs A
#   hit at each depth (fallback), the tail hit vs A's cold 136k -> 28k copy and greedy set c1 under 8 vs bz d0 -> LOADED
#   (XP2_LORDER D 8 8 D): c4 decode + one cold ~64k prompt (pskld.py), as r00z for D and 8.
# SUMMARY: per arm 136k TTFT (mean, vs A, pairs, vs the 35 s target), snapshots / staged GB / all-reduce GB per cold run; per
#   depth the hit TTFT A / K4 / K8 and K's extra over A (worst, mean); the K4 vs K8 trade and the pick (rule: the lower 136k
#   TTFT unless it costs > 5 s more mean hit TTFT); loaded D vs 8; identity at every compare.
# ~28 min (start ~5, checks + smoke ~3, cold x8 ~8, hit sets x3 ~4, copy + greedy set ~1, loaded x4 ~6). NO cc_sim, no 2.9, no
# power / host / fan settings; production compose files read only. Exact by design (r00z: design 1 was not -- the indexer deal, fixed by
# cf7bbc7 in the image; r00y2: B == A at every trusted row); bytes measured here and reported, never enabled.
# Knobs: XP2_IMAGE, XP2_ITEM, XP2_OUT, XP2_ORDER (AD4884DA), XP2_LORDER (D88D), XP2_HITS ("90000 58000 20000"),
#   XP2_DS (1: draft_sample in the pin), XP2_SIDECAR, XP2_SIDESHA.
# DRYRUN=1 bash dsm-psk2.sh : prints every command; checks the image literals (int5 + PSK + design 2), every helper, the
# sidecars, the yaml; no server, no GPU, writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/../ds41-stack2-tp4.yaml                   # stack2's yaml, read only (env_file ${XS2_ENVFILE} carries every start word)
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh
O=${XP2_OUT:-$R/xdsm/psk2}; export STATSDIR=$O/stats
RD=$R/xdsm/draft; RM=$R/xdsm; RS3=$R/xdsm/stack3
export IMAGE=${XP2_IMAGE:?set XP2_IMAGE}
ITEM=${XP2_ITEM:-57-p4zyzzzzz-ds-r00zz-psk2}
ORDER=${XP2_ORDER:-AD4884DA}; LORDER=${XP2_LORDER-D88D}; DEPTHS=135000; LDEPTH=64000; HITS=${XP2_HITS:-90000 58000 20000}
DS=${XP2_DS:-1}; OCCK=32; NTOK=512
SIDE=${XP2_SIDECAR:-dspark-dlw-q8_0.rad}; SIDESHA=${XP2_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
case $ORDER in *[!AD48]*|"") echo "XP2_ORDER takes A / D / 4 / 8 blocks (got '$ORDER')"; exit 1 ;; esac
case $LORDER in *[!AD48]*) echo "XP2_LORDER takes A / D / 4 / 8 blocks or nothing (got '$LORDER')"; exit 1 ;; esac
for v in $DEPTHS $LDEPTH $HITS; do case $v in ''|*[!0-9]*) echo "depths must be numbers (got '$v')"; exit 1 ;; esac; done
case $DS in ''|*[!0-9]*) echo "XP2_DS must be a draft_sample value 0..200 (got '$DS')"; exit 1 ;; esac
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
DQ=$QA/deepq.py; PL=$HD/pskld.py; MBP=$D/mb.py; TESTS=$D/ds41_tests.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
xlog() { log "RESULT xpsk2 $*"; }
# ---------------------------------------------------------------- the z0occ2 item's start + RADIANCE_DS41_PSK=0
GPTAB=q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422
SIDECAR=dspark-conf.f32
LEVENV="RADIANCE_DS41_GG_PF=1 RADIANCE_DS41_GG_PF_TABLE=$GPTAB RADIANCE_DS41_F32_PF=25 RADIANCE_DS41_ARENA_DECODE_ROWS=8,48
  RADIANCE_DS41_ROUTER_BF16=1 RADIANCE_DS41_EMB_RAM=1 RADIANCE_PLACE_SHADOW_SLOTS=16 RADIANCE_XPREFETCH=0 RADIANCE_DS41_DSPARK_WALK=1
  RADIANCE_DS41_ARGATHER_MAX=48 RADIANCE_DS41_QNFOLD=1 RADIANCE_DS41_SWFOLD=1 RADIANCE_DRAFT_CTXSKIP=0 RADIANCE_DS41_DSPARK_CONF=/models/$SIDECAR
  RADIANCE_DRAFT_LEND=0 RADIANCE_DRAFT_PROBE_BACKOFF=0 RADIANCE_DRAFT_LEND_BORROW=0 RADIANCE_DRAFT_SAMPLE=0 RADIANCE_WEIGHT_SIDECAR=/models/$SIDE
  RADIANCE_DS41_BI=0 RADIANCE_DRAFT_CTXSKIP_SLOTS=129 RADIANCE_DRAFT_LEND_WEIGHTS=0 RADIANCE_DS41_PSK=0"
LEVENV=$(echo $LEVENV)
LITS="gemm_nt_ggml's,prefill,rows,take,lever,GPF's,forms,(RADIANCE_DS41_GG_PF=1|gemm_nt_f32's,prefill,rows,take,lever,GPF-b's,tiled,form,(RADIANCE_DS41_F32_PF=25"
LITS="$LITS|48,tokens,use|a,rank),ram|expert,prefetch:,off"
LITS="$LITS|DSpark,lever,W1:,the,full-vocabulary,Markov,walk,declared,(dspark_walk_step,a,step,+,all_gather,over,4,rank(s)|fn4,dspark_walk=1,"
LITS="$LITS|one,kernel,up,to,48,tokens,a,step|RADIANCE_DS41_QNFOLD;,ds41_qnfold=1)|RADIANCE_DS41_SWFOLD;,ds41_swfold=1)|draft,context,skip,(RADIANCE_DRAFT_CTXSKIP=0|a,confidence,plane,(the,dspark_conf,gate"
LITS="$LITS|draft,lend:,set,up"
LITS="$LITS|DSpark,lever,SD:,the,sampled,full-vocabulary,walk,declared,(dspark_walk_sample_step,a,step,over,4,rank(s)|lever,SD:,fn4,draft_sample,0,at,start"
LITS="$LITS|weight,sidecar:,43,entries|read,from,the,weight,sidecar|lever,BI,declared,(RADIANCE_DS41_BI|now,0):,a,step,of,more"
LITS="$LITS|lever,Z0-A,set,up|activation,arena:,8,tokens,use|ring,of,129,steps|draft,weight,lend,(lever,Z0-W|mover:,lever,Z0-W:"
LITS="$LITS|lever,PSK,declared,(RADIANCE_DS41_PSK|now,0):,a,prefill,sequence's,chunk,runs,layers,21..39,over,its,last,160|(layer,21:,2464|layer,39:,160);,every,row,through,layer,20"
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": 8, "drafter_mx": 1, "gg_pf": 1, "f32_pf": 25, "dspark_walk": 1, "ds41_argather": 48, "idx_split": 1, "ds41_qnfold": 1, "ds41_swfold": 1, "draft_ctxskip": 1, "dspark_conf": 1, "dspark_conf_rows": 1, "draft_lend": 1, "draft_probe_backoff": 3, "draft_hybrid": 1}'
FIX=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update({"ds41_bi": 2, "draft_sample": int(sys.argv[2]), "draft_lend_hyst": 0, "arena_d0": 0, "draft_lend_w": 0, "draft_occ": 0, "draft_occ_k": int(sys.argv[3])}); print(json.dumps(d))' "$FIX5" "$DS" "$OCCK") || { echo "pin JSON"; exit 1; }
armj() { case $1 in A) echo '{"ds41_psk": 0, "ds41_psk_k": 8}' ;; D) echo '{"ds41_psk": 1, "ds41_psk_k": 8}' ;;
                     4) echo '{"ds41_psk": 2, "ds41_psk_k": 4}' ;; 8) echo '{"ds41_psk": 2, "ds41_psk_k": 8}' ;; esac; }
armn() { case $1 in A) echo "A (today)" ;; D) echo "D1 (design 1)" ;; 4) echo "K4" ;; 8) echo "K8" ;; esac; }
pinj() {   # pinj ARM [EXTRAJSON]: the pin + the arm + extra as one POST body
  local x=${2:-}; [ -n "$x" ] || x='{}'
  python3 -c 'import json,sys
d = {}
for a in sys.argv[1:]: d.update(json.loads(a))
print(json.dumps(d))' "$FIX" "$(armj $1)" "$x"
}
DESC="NST 5 z0occ2 start (stack3 S-arm: stack2 NST 5 env, DSpark adaptive, depth 0, hybrid 1, W1, small n5 set, drafter_mx, lend + back-off 3, VRAM levers, ds41_bi 2 pinned, draft_sample $DS pinned, --draft-coupling on, DLW sidecar $SIDE; ring 129, Z0-W declared, rungs 8,48; Z0 switches 0, draft_occ_k $OCCK) + RADIANCE_DS41_PSK=0"
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
  RADIANCE_DS41_BI RADIANCE_DS41_BITRACE RADIANCE_DS41_BITRACE_PMAX RADIANCE_DRAFT_CTXSKIP_SLOTS RADIANCE_DRAFT_LEND_WEIGHTS
  RADIANCE_DS41_PSK"
stackenv() {   # 105's t3 arm + gg_gemv 2 + rule-w + the K1 folds + turnover 0 + LEVENV, every other switch unset
  local kv
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  export RADIANCE_PLACE_TURNOVER=0
  for kv in $LEVENV; do export "$kv"; done
}
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
thermline() { type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"; }
cmpfile() {   # cmpfile A B -> IDENTICAL (n chars) | DIFFERENT at char k (na vs nb chars) | not compared
  [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
idrec() {   # idrec NAME RESULT [WHAT]: identity.txt + one RESULT line (a difference reported, not hidden)
  echo "$1 $2" >> $O/identity.txt
  case $2 in IDENTICAL*) xlog "identity $1${3:+ ($3)}: $2" ;; *) xlog "identity $1${3:+ ($3)}: $2 -- reported, not hidden" ;; esac
}
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xpsk2 $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { xlog "$1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  echo "$1 warm-up $r" >> $O/identity.txt
  case $r in IDENTICAL*) xlog "$1 warm-up answers vs stock ($(basename $2)): $r" ;; *) xlog "$1 warm-up answers vs stock: VIRHE $r" ;; esac
}
modes() {   # modes TAG: the start words in the container, mover 0 kept free x4, rule-w, K1, ARGATHER 48, every literal of LITS -> 0, else 1
  [ "$DRY" = 1 ] && { echo "+ modes $1: LEVENV in the container's env; mover '0 more kept free' on 4 ranks; rule-w in every recorded pass; K1 folds declared =1; ARGATHER 48-row note; literals [$(echo "$LITS" | tr ',' ' ')]"; return 0; }
  local CE kv bad="" ok=1 NW NR KL AL nz lit found="" miss=""
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in RADIANCE_PLACE_TURNOVER=0 $LEVENV; do echo "$CE" | grep -qxF "$kv" || bad="$bad ${kv%%=*}"; done
  [ -z "$bad" ] || { xlog "$1: VIRHE start words not in the container:$bad"; ok=0; }
  nz=$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots \([0-9]+ shadow, 0 more kept free')
  [ "$nz" = 4 ] || { xlog "$1: VIRHE turnover 0 asked but $nz of 4 ranks keep 0 slots free"; ok=0; }
  NW=$(docker logs $CN 2>&1 | grep -c 'the acquire rule plays .*RADIANCE_AQL_ACQ=rule-w)'); NR=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_AQL_ACQ=rule)')
  [ "$NW" -gt 0 ] && [ "$NR" = 0 ] || { xlog "$1: VIRHE rule-w is not what the recorded passes play ($NW / $NR)"; ok=0; }
  KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-500)
  case "$KL" in *"ds41_kvfold=1"*"ds41_ofold=1"*"ds41_onefill=1"*) ;; *) xlog "$1: VIRHE the K1 switches do not start at 1: '${KL}'"; ok=0 ;; esac
  AL=$(docker logs $CN 2>&1 | grep -m1 'the MoE gather and its all-reduce as one kernel' | sed 's/^radiance: //' | cut -c1-300)
  case "$AL" in *"up to 48 tokens a step (RADIANCE_DS41_ARGATHER, RADIANCE_DS41_ARGATHER_MAX"*) ;; *) xlog "$1: VIRHE the ARGATHER note is not the 48-row one: '${AL}'"; ok=0 ;; esac
  IFS='|' read -ra lit <<< "$LITS"
  for l in "${lit[@]}"; do [ -n "$l" ] || continue
    if docker logs $CN 2>&1 | grep -qF -- "${l//,/ }"; then found="$found ${l%%,*}"; else miss="$miss | ${l//,/ }"; fi; done
  xlog "$1 modes: ${#lit[@]} literal(s), found [$(echo $found)]$([ -n "$miss" ] && echo " -- VIRHE missing:$miss") | $(docker logs $CN 2>&1 | grep -E "lever PSK|lever BI declared|lever SD|weight sidecar|lever Z0|activation arena|ring of|draft lend:|draft weight lend|no level for" | sed 's/^radiance: //; s/^[A-Z] *//' | cut -c1-260 | sort -u | tr '\n' '|' | cut -c1-2800)"
  [ -z "$miss" ] || ok=0
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
vramstart() {
  [ "$DRY" = 1 ] && { echo "+ GET /stats > $O/$1-stats0.json; RESULT xpsk2 $1 VRAM at start (experts per tier; vs stack3 s3-n5)"; return 0; }
  curl -s -m 30 $URL/stats > $O/$1-stats0.json 2>/dev/null
  xlog "$1 VRAM at start: experts $(python3 -c 'import json,sys
for p in sys.argv[1:]:
    try: s = json.load(open(p)); print({x.get("name"): x.get("units") for x in s.get("experts", {}).get("tiers", [])}, end=" ")
    except Exception as e: print("n/a", end=" ")' $O/$1-stats0.json $RS3/s3-n5-stats0.json 2>&1 | cut -c1-320) (this start, then stack3 s3-n5) | $(docker logs $CN 2>&1 | grep -E '^I\[0\] mover:|checkpoint slots|weight sidecar:' | sed 's/^radiance: //; s/^[A-Z]\[0\] //; s/^[A-Z] *//' | cut -c1-220 | sort -u | tr '\n' '|' | cut -c1-1500)"
}
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
fn4code() { [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null; }
pcreset() { [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o $O/$1-reset.json -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  xlog "$1 prefix cache reset: HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"; }
arm() {   # arm A|B: the pin + the arm + epoch_bump, a nudge to apply it at a step
  fn4set "$(pinj $1 '{"epoch_bump": 1}')"; nudge
}
msnap() { [ "$DRY" = 1 ] && { echo "+ GET /metrics > $O/m-$1.txt"; return 0; }; curl -s -m 20 $URL/metrics > $O/m-$1.txt 2>/dev/null; }
mdelta() {   # mdelta A B -> JSON: draft acceptance, non-decode steps and their mean ms, decode steps, prefill tokens over the window
  [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }
  python3 -c 'import json,re,sys
def rd(p):
    d = {}
    try:
        for l in open(p):
            m = re.match(r"^(radiance:[a-z_]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", l)
            if m: d[m.group(1)] = d.get(m.group(1), 0.0) + float(m.group(2))
    except OSError: pass
    return d
a, b = rd(sys.argv[1]), rd(sys.argv[2])
g = lambda k: b.get("radiance:" + k, 0.0) - a.get("radiance:" + k, 0.0)
nd = g("engine_steps_prefill_total") + g("engine_steps_mixed_total")
nds = g("engine_step_seconds_total") - g("engine_step_seconds_decode_total")
dt, da = g("draft_tokens_total"), g("draft_accepted_total")
print(json.dumps(dict(acc=round(da / dt, 4) if dt > 0 else None, draft_tok=int(dt), draft_acc=int(da),
      nondecode_steps=int(nd), mixed_steps=int(g("engine_steps_mixed_total")), nondecode_ms=round(1000 * nds / nd, 1) if nd > 0 else None,
      decode_steps=int(g("engine_steps_decode_total")), prefill_tok=int(g("prefill_tokens_total")), decode_tok=int(g("decode_tokens_total")))))' $O/m-$1.txt $O/m-$2.txt 2>/dev/null || echo '{}'
}
pskstat() { [ "$DRY" = 1 ] && { echo "(dry)"; return 0; }; echo "$(docker logs $CN 2>&1 | grep 'lever PSK:' | tail -1 | sed 's/^radiance: deepseek41: //' | cut -c1-300) | $(docker logs $CN 2>&1 | grep 'lever PSK design 2: [0-9]* planned' | tail -1 | sed 's/^I //' | cut -c1-300) | VIRHE lines $(docker logs $CN 2>&1 | grep -c 'lever PSK design 2: VIRHE')"; }
ssnap() { [ "$DRY" = 1 ] && { echo "+ GET /stats > $O/st-$1.json"; return 0; }; curl -s -m 30 $URL/stats > $O/st-$1.json 2>/dev/null; }
sdelta() {   # sdelta A B -> JSON: checkpoints written, staged bytes (all ranks / rank 0), rank-0 staged layers, all-reduce bytes
  [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }
  python3 -c 'import json,sys
def rd(p):
    try: return json.load(open(p))
    except Exception: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
g = lambda d, *k: (lambda v: v if isinstance(v, (int, float)) else 0)(__import__("functools").reduce(lambda x, y: (x or {}).get(y) if isinstance(x, dict) else None, k, d))
print(json.dumps(dict(ckpts=int(g(b, "checkpoints_written") - g(a, "checkpoints_written")),
      staged_gb=round((g(b, "link", "staged_bytes") - g(a, "link", "staged_bytes")) / 1e9, 2),
      staged_gb_r0=round((g(b, "experts", "staged_bytes") - g(a, "experts", "staged_bytes")) / 1e9, 2),
      stage_layers_r0=int(g(b, "experts", "stage_layers") - g(a, "experts", "stage_layers")),
      ar_gb=round((g(b, "link", "ar_bytes") - g(a, "link", "ar_bytes")) / 1e9, 2))))' $O/st-$1.json $O/st-$2.json 2>/dev/null || echo '{}'
}
swnote() {   # swnote TAG: start values; refusals; every arm applied at a step, then A
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 at start (ds41_psk 0, ds41_psk_k 8, ds41_bi 0, gg_gemv 2, aql_acq 2, K1 three 1, draft_sample 0, draft_capable 1, epoch 0); refusals ds41_psk 4 -> 400, ds41_psk_k 0 -> 400; arms A $(pinj A) | D $(armj D) | 4 $(armj 4) | 8 $(armj 8) applied, then A"; return 0; }
  local s0 g c1 c2 a sets="" bad=""
  s0=$(fn4get)
  c1=$(fn4code '{"ds41_psk": 4}'); c2=$(fn4code '{"ds41_psk_k": 0}')
  for a in A D 4 8 A; do
    arm $a; g=$(fn4get)
    python3 -c 'import json,sys
g, w = json.loads(sys.argv[1]), json.loads(sys.argv[2])
sys.exit(0 if all(g["switches"].get(k) == v for k, v in w.items()) and not g["pending"] else 1)' "$g" "$(pinj $a)" 2>/dev/null || bad="$bad $a"
    sets="$sets $a:$(echo "$g" | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print("psk=%s,k=%s,bi=%s,ds=%s,ep=%s" % (w.get("ds41_psk"), w.get("ds41_psk_k"), w.get("ds41_bi"), w.get("draft_sample"), d.get("epoch")))' 2>/dev/null)"
  done
  xlog "$1 switch note: at start $(echo "$s0" | cut -c1-1000) | ds41_psk 4 -> HTTP $c1, ds41_psk_k 0 -> HTTP $c2 | arms applied:$sets$([ -n "$bad" ] && echo " -- VIRHE not applied:$bad")"
  python3 -c 'import json,sys
s0 = json.loads(sys.argv[1]); w = s0["switches"]
ok = w.get("ds41_psk") == 0 and w.get("ds41_psk_k") == 8 and w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill"))
ok = ok and w.get("ds41_bi") == 0 and s0["epoch"] == 0 and w["draft_adaptive"] == 1 and s0["draft_capable"] == 1 and w.get("draft_sample") == 0
sys.exit(0 if ok else 1)' "$s0" 2>/dev/null && [ "$c1" = 400 ] && [ "$c2" = 400 ] && [ -z "$bad" ]
}
gate1() {   # restart + ONE check; over the limit logged (in-process A/B: a slow state shows in every arm)
  local tag=$1 lim=$2 ms st=fast
  restart $tag || { keeplog $tag-fail; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && return 0
  awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}' && { st="SLOW (not retried)"; keeplog $tag-slow; }
  echo "$st $ms" > $O/guard-$tag.txt
  xlog "guard $tag: $st, 1-user ${ms} ms/step (limit $lim; stack3 s3-n5 $(cat $RS3/guard-s3-n5.txt 2>/dev/null))"
}
smoke() {   # smoke TAG: under 8 -- the greedy set at conc 4 (mixed steps: BI segments, planned prefill segments); healthy -> 0
  local tag=$1 t0=$(date +%s) n
  arm 8
  run "(cd $D && timeout 600 python3 $GS $URL $O/$tag-smoke.jsonl --conc 4 --max-tokens 32 > $O/$tag-smoke.txt 2>&1)"
  [ "$DRY" = 1 ] && { echo "+ smoke $tag: arm 8 $(armj 8): greedy set conc 4 (32 tok) 20 of 20, /health 200, no design-2 plan-mismatch line"; return 0; }
  n=$(python3 -c 'import json,sys; print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l).get("text") is not None))' $O/$tag-smoke.jsonl 2>/dev/null || echo 0)
  if [ "$n" = 20 ] && curl -s -m 10 -o /dev/null -w '%{http_code}' $URL/health | grep -q 200 && \
     [ "$(docker logs $CN 2>&1 | grep -c 'lever PSK design 2: VIRHE')" = 0 ]; then
    xlog "$tag smoke (arm 8): greedy set conc 4 20 of 20, healthy, $(( $(date +%s) - t0 )) s | $(pskstat)"; return 0; fi
  xlog "$tag smoke (arm 8): VIRHE $n of 20 answered, unhealthy or a design-2 plan VIRHE line | $(docker logs $CN 2>&1 | grep -E '^E |step fail|refus' | tail -3 | tr '\n' ' ' | cut -c1-500)"
  keeplog $tag-smoke; return 1
}
deepx() {   # deepx TAG ARM DEPTH NAME [noreset]: deepq.py's fixed prompt at DEPTH, 512 greedy tokens -> TTFT, cached, acceptance, ms
  local tag=$1 a=$2 dep=$3 nm=$4 t0=$(date +%s) j md
  arm $a
  [ "${5:-}" = noreset ] || pcreset $tag-$nm
  msnap $tag-$nm-0; ssnap $tag-$nm-0
  run "(cd $D && timeout 1800 python3 $DQ $O/$tag-$nm.txt $dep $NTOK > $O/$tag-$nm.out 2>&1)"
  msnap $tag-$nm-1; ssnap $tag-$nm-1
  [ "$DRY" = 1 ] && { echo "+ RESULT xpsk2 $tag $nm: arm $a depth $dep: TTFT, cached, acceptance, non-decode step ms, /stats deltas (snapshots, staged GB, all-reduce GB) -> $O/deep.jsonl"; return 0; }
  j=$(grep -m1 '^DEEPQ ' $O/$tag-$nm.out | cut -c7-); md=$(mdelta $tag-$nm-0 $tag-$nm-1)
  md=$(python3 -c 'import json,sys
d = json.loads(sys.argv[1] or "{}"); d.update(json.loads(sys.argv[2] or "{}")); print(json.dumps(d))' "$md" "$(sdelta $tag-$nm-0 $tag-$nm-1)" 2>/dev/null || echo "$md")
  [ -n "$j" ] || { xlog "$tag $nm: VIRHE no answer ($(tail -2 $O/$tag-$nm.out | tr '\n' ' ' | cut -c1-300))"; return 1; }
  python3 -c 'import json,sys
d = json.loads(sys.argv[1]); d.update(json.loads(sys.argv[2]) if sys.argv[2].strip() else {})
d.update(name=sys.argv[3], arm=sys.argv[4], depth=int(sys.argv[5]), reset=sys.argv[6] != "noreset", text=sys.argv[7])
print(json.dumps(d))' "$j" "$md" "$nm" "$a" "$dep" "${5:-reset}" "$O/$tag-$nm.txt" >> $O/deep.jsonl
  xlog "$tag $nm (arm $(armn $a) $(armj $a), deepq depth $dep, $NTOK greedy tokens$([ "${5:-}" = noreset ] && echo ", NO reset")): $(python3 -c 'import json,sys
d, m = json.loads(sys.argv[1]), json.loads(sys.argv[2] or "{}")
print("status %s, prompt %s tok, cached %s, TTFT %s s (%s prefill tok/s), %s tok, %s chars | decode acceptance %s (%s of %s drafted) | non-decode steps %s, %s ms mean | snapshots %s, staged %s GB (rank 0 %s GB, %s layers), all-reduce %s GB" % (
    d.get("status"), d.get("prompt_tokens"), d.get("cached"), d.get("ttft"),
    round((d["prompt_tokens"] - (d.get("cached") or 0)) / d["ttft"], 1) if d.get("ttft") and d.get("prompt_tokens") else "-",
    d.get("tokens"), d.get("chars"), m.get("acc"), m.get("draft_acc"), m.get("draft_tok"), m.get("nondecode_steps"), m.get("nondecode_ms"),
    m.get("ckpts"), m.get("staged_gb"), m.get("staged_gb_r0"), m.get("stage_layers_r0"), m.get("ar_gb")))' "$j" "$md" 2>&1 | cut -c1-600) | psk: $(pskstat) | $(thermline $t0 $(date +%s))"
}
coldabba() {   # coldabba TAG: the 136k prompt cold in ORDER; identity vs dsm-225 and vs the first A; the hit set after the
  # second cold run of 8, 4 and A (the cache holds that arm's snapshots)
  local tag=$1 dep=135000 a k fa="" seen=""
  k=0
  for a in $(echo $ORDER | grep -o .); do
    k=$((k + 1)); stopfile && return 0
    deepx $tag $a $dep d$dep-$k$a
    seen="$seen$a"
    [ "$DRY" = 1 ] || {
      [ -n "$fa" ] || { [ $a = A ] && fa=$O/$tag-d$dep-$k$a.txt; }
      idrec "deep136-$k$a-vs-dsm225" "$(cmpfile $O/$tag-d$dep-$k$a.txt $RM/deep-dsm-225.txt)" "136k answer, arm $(armn $a), vs dsm-225 NST 0 stock cold"
      [ -n "$fa" ] && [ "$O/$tag-d$dep-$k$a.txt" != "$fa" ] && idrec "d$dep-$k$a-vs-A" "$(cmpfile $fa $O/$tag-d$dep-$k$a.txt)" "136k cold, arm $(armn $a), vs the first A"
    }
    case $a in A|4|8) [ "$(echo "$seen" | tr -cd "$a" | wc -c)" = 2 ] && hitset $tag $a ;; esac
  done
}
hitset() {   # hitset TAG ARM: no reset after ARM's cold 136k -- deepq at each HITS depth (descending), then the 136k again
  local tag=$1 a=$2 d
  for d in $HITS 135000; do
    stopfile && return 0
    deepx $tag $a $d hit$a-$d noreset
  done
}
hitident() {   # hitident TAG: each K arm's hit answer vs A's at the same depth (the fallback: today's bytes); tail hits vs A cold
  local tag=$1 a d ca
  [ "$DRY" = 1 ] && { echo "+ identity: hit4-D / hit8-D vs hitA-D at each depth; hitX-135000 (the tail snapshot) vs the first cold A"; return 0; }
  ca=$(ls $O/$tag-d135000-*A.txt 2>/dev/null | head -1)
  for a in 4 8 A; do
    for d in $HITS; do
      [ $a = A ] && continue
      idrec "hit$a-$d-vs-hitA" "$(cmpfile $O/$tag-hitA-$d.txt $O/$tag-hit$a-$d.txt)" "arm $(armn $a): the depth-$d prompt resumed below its divergence (fallback to the anchor below) vs A's resume"
    done
    idrec "hit$a-tail136-vs-coldA" "$(cmpfile ${ca:-/nonexistent} $O/$tag-hit$a-135000.txt)" "arm $(armn $a): the 136k resumed from its tail snapshot vs the cold A"
  done
}
copyx() {   # copyx TAG ARM: the 28k copy (dsm-stack2.sh copyt's request) after a reset -> vs bz d0
  local tag=$1 t0=$(date +%s)
  arm $2; pcreset $tag-copy$2
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset 0 --tag=-$tag-$2 --outdir $O > $O/$tag-copy$2.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  xlog "$tag copy arm $(armn $2) (after a cold ~31k prompt): $(grep '^J ' $O/$tag-copy$2.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300) | $(thermline $t0 $(date +%s))"
  idrec "copy-$2-vs-d0" "$(cmpfile $O/copy-$tag-$2-run1.txt $RD/copy-dsd-d0-i1-run1.txt)" "28k copy vs bz d0 (NST 0 stock)"
}
gsalone() {   # gsalone TAG ARM: greedy set c1 alone (192 tok, T 0) after a reset -> vs bz d0
  local tag=$1 t0=$(date +%s) r
  arm $2; pcreset $tag-gs$2
  run "(cd $D && timeout 900 python3 $GS $URL $O/$tag-gs$2.jsonl --conc 1 --max-tokens 192 > $O/$tag-gs$2.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  xlog "$tag greedy set c1 alone arm $(armn $2): $(tail -1 $O/$tag-gs$2.txt | cut -c1-200), $(( $(date +%s) - t0 )) s | psk: $(pskstat)"
  r=$([ -s $RD/dsd-d0-gs-c1.jsonl ] && python3 $GS --cmp $RD/dsd-d0-gs-c1.jsonl $O/$tag-gs$2.jsonl 2>&1 | cut -c1-300 || echo "not compared (no reference)")
  idrec "gs-$2-vs-d0" "$r" "greedy set c1 vs bz d0"
}
loadx() {   # loadx TAG ARM NAME: c4 decode streams + ONE cold LDEPTH deepq prompt (pskld.py --load, no extra prefill)
  local tag=$1 a=$2 nm=$3 t0=$(date +%s) j md
  arm $a; pcreset $tag-$nm
  msnap $tag-$nm-0
  run "(cd $D && timeout 2700 python3 -I $PL run --url $URL --tests $TESTS --mb $MBP --which deep --depth $LDEPTH --ntok $NTOK --load --conc 4 --warm 10 --pf-tok 0 --out $O/$tag-$nm.json --text $O/$tag-$nm.txt > $O/$tag-$nm.log 2>&1)"
  msnap $tag-$nm-1
  [ "$DRY" = 1 ] && { echo "+ RESULT xpsk2 $tag $nm: arm $a: mixed-step ms, c4 tok/s in the prefill window (agg / per-stream p50), TPOT, TTFT -> $O/load.jsonl"; return 0; }
  j=$(grep -m1 '^LONGID ' $O/$tag-$nm.log | cut -c8-); md=$(mdelta $tag-$nm-0 $tag-$nm-1)
  [ -n "$j" ] || { xlog "$tag $nm: VIRHE no LONGID line ($(tail -3 $O/$tag-$nm.log | tr '\n' ' ' | cut -c1-400))"; return 1; }
  python3 -c 'import json,sys
d = json.loads(sys.argv[1]); d.update(json.loads(sys.argv[2]) if sys.argv[2].strip() else {}); d.update(name=sys.argv[3], arm=sys.argv[4], text=sys.argv[5])
print(json.dumps(d))' "$j" "$md" "$nm" "$a" "$O/$tag-$nm.txt" >> $O/load.jsonl
  xlog "$tag $nm (arm $(armn $a), c4 decode + one cold ~$LDEPTH-depth deepq prompt, T 0): $(python3 -c 'import json,sys
d, m = json.loads(sys.argv[1]), json.loads(sys.argv[2] or "{}")
print("probe status %s, prompt %s tok, cached %s, TTFT %s s | prefill window %s s: c4 %s tok, agg %s tok/s, per-stream p50 %s tok/s, TPOT p50 %s / p90 %s / max %s ms | mixed steps %s (non-decode %s at %s ms mean) | c4 %s requests, %s errors" % (
    d.get("status"), d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("pf_win_s"), d.get("c4_pf_tokens"), d.get("c4_pf_agg_tps"),
    d.get("c4_pf_stream_tps_p50"), d.get("tpot_pf_p50"), d.get("tpot_pf_p90"), d.get("tpot_pf_max"), m.get("mixed_steps"), m.get("nondecode_steps"),
    m.get("nondecode_ms"), d.get("c4_requests"), d.get("c4_errors")))' "$j" "$md" 2>&1 | cut -c1-700) | psk: $(pskstat) | $(thermline $t0 $(date +%s))"
}
loadabba() {   # loadabba TAG: the arms in LORDER; identity: each loaded answer vs the first loaded one, and vs r00z's loaded A
  local tag=$1 a k fl=""
  [ -n "$LORDER" ] || return 0
  k=0
  for a in $(echo $LORDER | grep -o .); do
    k=$((k + 1)); stopfile && return 0
    loadx $tag $a ld-$k$a
    [ "$DRY" = 1 ] && continue
    [ -n "$fl" ] || { fl=$O/$tag-ld-$k$a.txt; continue; }
    idrec "ld-$k$a-vs-ld1" "$(cmpfile $fl $O/$tag-ld-$k$a.txt)" "the ~$LDEPTH prompt under c4 load, arm $(armn $a), vs the first loaded run"
  done
  [ "$DRY" = 1 ] || idrec "ld-1-vs-r00z-ldA" "$(cmpfile $R/xdsm/psk/ps-n5-ld-1A.txt $(ls $O/$tag-ld-1*.txt 2>/dev/null | head -1))" "loaded, the first run here vs r00z's loaded A (cross-start, ds41-psk image)"
}
summary() {
  [ "$DRY" = 1 ] && { echo "+ RESULT xpsk2 summary: per arm 136k TTFT (mean, vs A, pairs, vs 35 s), snapshots / staged GB / all-reduce GB a cold run; hit TTFT per depth A / K4 / K8 and K's extra over A (worst, mean); K4 vs K8 trade + pick; loaded D vs 8; identity"; return 0; }
  python3 - "$O" "$ORDER" "$LORDER" "$HITS" > $O/summary.txt 2>&1 <<'PY'
import json, os, statistics, sys
o, order, lorder, hits = sys.argv[1], sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4].split()]
NAME = {"A": "A", "D": "D1", "4": "K4", "8": "K8"}
def rows(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except OSError: return []
def m(v):
    v = [x for x in v if isinstance(x, (int, float))]
    return statistics.mean(v) if v else None
def f(x, n=2): return "-" if x is None else format(x, ".%df" % n)
def pc(b, a): return "%+.1f %%" % ((b / a - 1) * 100) if a and b else "-"
dp = rows(os.path.join(o, "deep.jsonl"))
cold = [r for r in dp if r.get("reset") and r["depth"] == 135000 and not r["name"].startswith("hit")]
parts = []
ta = m([r.get("ttft") for r in cold if r["arm"] == "A"])
res = {}
for a in "AD48":
    s = [r for r in cold if r["arm"] == a]
    if not s: continue
    t = m([r.get("ttft") for r in s]); res[a] = t
    # pairs: each run of arm a vs the A runs in ORDER order (first with first, last with last)
    A = [r for r in cold if r["arm"] == "A"]
    pr = "/".join("%+.1f" % ((x.get("ttft") / y.get("ttft") - 1) * 100) for x, y in zip(s, A) if x.get("ttft") and y.get("ttft")) if a != "A" else "-"
    l = "136k cold %s: TTFT %s s (runs %s), vs A %s (pairs %s), vs the 35 s target %s | per run: snapshots %s, staged %s GB all ranks (rank 0 %s layers), all-reduce %s GB | acceptance %s, non-decode step ms %s%s" % (
        NAME[a], f(t), "/".join(f(r.get("ttft")) for r in s), pc(t, ta) if a != "A" else "-", pr,
        ("MET (%.1f s under)" % (35 - t)) if t and t <= 35 else ("%.1f s over" % (t - 35) if t else "-"),
        "/".join(str(r.get("ckpts")) for r in s), f(m([r.get("staged_gb") for r in s]), 1), f(m([r.get("stage_layers_r0") for r in s]), 0),
        f(m([r.get("ar_gb") for r in s]), 1), f(m([r.get("acc") for r in s]), 3), f(m([r.get("nondecode_ms") for r in s]), 0),
        "" if all(r.get("cached") == 0 for r in s) else " -- VIRHE a run was not cold")
    parts.append(l); print("ARMS " + l)
hit = {(r["arm"], r["depth"]): r for r in dp if r["name"].startswith("hit")}
extra = {"4": [], "8": []}
for d in hits + [135000]:
    cells = []
    for a in "A48":
        r = hit.get((a, d))
        if not r: continue
        cells.append("%s cached %s of %s, TTFT %s s" % (NAME[a], r.get("cached"), r.get("prompt_tokens"), f(r.get("ttft"))))
    ra = hit.get(("A", d))
    for a in "48":
        r = hit.get((a, d))
        if r and ra and r.get("ttft") is not None and ra.get("ttft") is not None:
            x = r["ttft"] - ra["ttft"]
            if d != 135000: extra[a].append(x)
            cells.append("%s extra %+.2f s (%s tokens more recomputed)" % (NAME[a], x, (ra.get("cached") or 0) - (r.get("cached") or 0)))
    l = "hit at depth %d: %s" % (d, " | ".join(cells) or "none")
    parts.append(l); print("ARMS " + l)
for a in "48":
    if extra[a]:
        l = "%s hit cost over A at non-anchor depths %s: worst %+.2f s, mean %+.2f s" % (NAME[a], "/".join(str(d) for d in hits), max(extra[a]), statistics.mean(extra[a]))
        parts.append(l); print("ARMS " + l)
t4, t8 = res.get("4"), res.get("8")
h4 = statistics.mean(extra["4"]) if extra["4"] else None
h8 = statistics.mean(extra["8"]) if extra["8"] else None
if t4 and t8:
    pick = "8" if (t8 <= t4 and (h8 is None or h4 is None or h8 - h4 <= 5.0)) else "4"
    if t8 < t4 and h8 is not None and h4 is not None and h8 - h4 > 5.0: why = "K8 is %.2f s faster cold but its mean hit costs %.2f s more (> 5 s)" % (t4 - t8, h8 - h4)
    elif pick == "8": why = "K8 %.2f s vs K4 %.2f s cold, mean hit extra K8 %s s vs K4 %s s" % (t8, t4, f(h8), f(h4))
    else: why = "K4 %.2f s vs K8 %.2f s cold (K4 not slower), mean hit extra K4 %s s vs K8 %s s" % (t4, t8, f(h4), f(h8))
    l = "K pick (rule: the lower 136k TTFT unless it costs > 5 s more mean hit TTFT): K%s -- %s" % (pick, why)
    parts.append(l); print("ARMS " + l)
ld = rows(os.path.join(o, "load.jsonl"))
if ld:
    cells = []
    for nm, k, n in (("mixed-step ms", "nondecode_ms", 0), ("c4 agg tok/s in the prefill window", "c4_pf_agg_tps", 1),
                     ("TPOT p50 ms", "tpot_pf_p50", 1), ("probe TTFT s", "ttft", 2)):
        v = {a: m([r.get(k) for r in ld if r["arm"] == a]) for a in "D8"}
        cells.append("%s D1 %s K8 %s (K8 vs D1 %s)" % (nm, f(v["D"], n), f(v["8"], n), pc(v["8"], v["D"])))
    l = "loaded (c4 + one cold ~64k prompt, %s): %s" % (lorder, " | ".join(cells))
    parts.append(l); print("ARMS " + l)
try: ident = [l.strip() for l in open(os.path.join(o, "identity.txt")) if l.strip()]
except OSError: ident = []
bad = [l for l in ident if "IDENTICAL" not in l]
print("SUMMARY identity: %s || %s" % (("IDENTICAL at every compare (%d)" % len(ident)) if ident and not bad else (("NOT IDENTICAL (%d of %d): " % (len(bad), len(ident)) + "; ".join(bad))[:1200] if ident else "none"),
      " || ".join(parts)))
PY
  while IFS= read -r l; do case "$l" in SUMMARY*) ;; *) xlog "${l:0:3000}" ;; esac; done < $O/summary.txt
  xlog "summary: $(grep -m1 '^SUMMARY ' $O/summary.txt | cut -c9- | cut -c1-6000)"
}
dstart() {   # the one NST 5 start
  local tag=p2-n5 ok
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on"
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV > $XS2_ENVFILE"; else printf '%s\n' $LEVENV > $XS2_ENVFILE; fi
  xlog "$tag: start ($DESC) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $FIX | arms A $(armj A) D $(armj D) 4 $(armj 4) 8 $(armj 8) | env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ' | cut -c1-2200)"
  gate1 $tag 70 || { xlog "$tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  modes $tag || { xlog "$tag: VIRHE the modes are not the start's -> stopped"; down; return 1; }
  if [ "$DRY" != 1 ]; then
    ok=1
    docker logs $CN 2>&1 | grep -q "DSpark drafter" || { xlog "$tag: VIRHE no DSpark drafter line"; ok=0; }
    docker logs $CN 2>&1 | grep -q "draft depth: adaptive" || { xlog "$tag: VIRHE no adaptive draft depth line"; ok=0; }
    [ $ok = 1 ] || { keeplog $tag-nodraft; down; return 1; }
  fi
  vramstart $tag
  swnote $tag || { xlog "$tag: VIRHE the switches are not as expected -> stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  smoke $tag || { xlog "$tag: VIRHE the smoke under 8 failed -> stopped"; down; return 1; }
  stopfile || coldabba $tag
  stopfile || hitident $tag
  stopfile || copyx $tag 8
  stopfile || gsalone $tag 8
  stopfile || loadabba $tag
  run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
  xlog "$tag: PSK stats at the end: $(pskstat) | $(docker logs $CN 2>&1 | grep -cE '^E ') E lines"
  down
  xlog "$tag: done"
}
trap 'down' EXIT
trap 'exit 143' TERM INT
# ---------------------------------------------------------------- main
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) | one NST 5 start: $DESC | arms A $(armj A), D1 $(armj D), K4 $(armj 4), K8 $(armj 8) | cold 136k order $ORDER ($NTOK greedy tokens) | hit sets after the 2nd cold 8 / 4 / A at depths [$HITS] + the 136k tail | copy + greedy set c1 under K8 | loaded order ${LORDER:-none} (c4 + one cold depth-$LDEPTH prompt) | no evict | refs dsm-225 deep, bz d0 copy / greedy set, r00z (design 1: A 50.46 s, D1 43.48 s at 136k, same start)"
IMGLITS="B|dspark_conf_rows, gg_pf, f32_pf, dspark_walk, ds41_bi, draft_sample, draft_lend_hyst, arena_d0, draft_lend_w, draft_occ, draft_occ_k, ds41_psk, ds41_psk_k, epoch_bump
B|ds41_psk
B|ds41_psk_k takes 1 .. 4096
B|rad_psk_plan
B|lever PSK design 2: %lld planned passes
A|rad_psk_plan
A|lever PSK design 2 (anchors, RADIANCE_DS41_PSK=
A|lever PSK design 2: VIRHE
B|reset_prefix_cache
B|radiance:engine_steps_mixed_total
B|radiance:engine_step_seconds_decode_total
B|radiance:draft_accepted_total
A|RADIANCE_DS41_PSK
A|lever PSK declared (RADIANCE_DS41_PSK, fn4 ds41_psk, now
A|lever PSK: %lld cut steps
A|RADIANCE_DS41_KVFOLD
K|RADIANCE_DS41_GG_PF_TABLE
A|RADIANCE_DS41_DSPARK_WALK
K|dspark_walk_step
A|RADIANCE_DS41_ARENA_DECODE_ROWS
A|lever BI declared (RADIANCE_DS41_BI, fn4 ds41_bi, now
B|RADIANCE_DRAFT_SAMPLE
K|dspark_walk_sample_step
A|DSpark lever SD: the sampled full-vocabulary walk declared
S|mtp_couple_verify
B|RADIANCE_WEIGHT_SIDECAR
B|read from the weight sidecar
B|draft_occ_k takes 1 .. 4096
B|RADIANCE_DRAFT_LEND_WEIGHTS
B|lever Z0-A set up
B|RADIANCE_DRAFT_CTXSKIP_SLOTS"
docker image inspect $IMAGE > /dev/null 2>&1 || { xlog "VIRHE image $IMAGE not on this host"; exit 1; }
IMGMISS=$(echo "$IMGLITS" | timeout 120 docker run --rm -i --network none --entrypoint sh $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels/libr4d.so; S=/opt/radiance/share/radiance/kernels/spechub.so; n=0
while IFS= read -r l; do [ -n "$l" ] || continue; n=$((n + 1)); w=${l%%|*}; s=${l#*|}
  case $w in B) f=$B ;; A) f=$A ;; K) f=$K ;; S) f=$S ;; *) f="$B $A $K" ;; esac
  o=0; for x in $f; do grep -qzF -- "$s" $x && { o=1; break; }; done
  [ $o = 1 ] || echo "MISS $w|$s"; done
test -x /opt/radiance/bin/rad-sidecar || echo "MISS bin/rad-sidecar"; test -s /opt/radiance/share/radiance/psk2-PROVENANCE.txt || echo "MISS psk2-PROVENANCE.txt"; echo "CHECKED $n"' 2>&1)
case "$IMGMISS" in "CHECKED "*) ;; *) xlog "VIRHE $IMAGE is not the ds41-psk2 image -- missing: $(echo "$IMGMISS" | grep -v '^CHECKED' | tr '\n' ';' | cut -c1-900)"; exit 1 ;; esac
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)) is the ds41-psk2 image: all $(echo "$IMGMISS" | sed 's/CHECKED //') literals present (the 36-switch registry with ds41_psk + ds41_psk_k, lever PSK designs 1 + 2, rad_psk_plan both sides, the int5 levers, the /metrics counters) + bin/rad-sidecar + psk2-PROVENANCE -- OK"
HMISS=""
for f in "$LIB29" "$LIBPR" $Y $GS $DQ $PL $MBP $TESTS $D/fastcheck.py; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $GS $DQ $PL $MBP $TESTS $D/fastcheck.py 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
ST=$(timeout 60 python3 -I $PL selftest --tests $TESTS --mb $MBP 2>&1 | tail -1); case "$ST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$PL(${ST:0:300})" ;; esac
DQP=$(timeout 60 python3 -I -c 'import importlib.util, json, sys
def ld(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
t = ld(sys.argv[2], "t"); pl = ld(sys.argv[1], "pl")
import ast
src = open(sys.argv[3]).read()
# deepq.py builds its prompt as pskld.py deep_payload does: CPT 3.95, offset 1200, the same header and task (read by ast)
task = next((ast.literal_eval(n.value) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "task" for t in n.targets)), None)
ok = "CPT, OFF = 3.95, 1200" in src and "# deepq fixed prompt v1" in src and pl.DEEP_CPT == 3.95 and pl.DEEP_OFF == 1200 and task == pl.DEEP_TASK
d = pl.deep_payload(t, 64000, 512)
print("DQ OK" if ok and d["messages"][0]["content"].startswith("# deepq fixed prompt v1\n") and d["max_tokens"] == 512 else "DQ MISMATCH")' $PL $TESTS $DQ 2>&1 | tail -1)
case "$DQP" in "DQ OK") ;; *) HMISS="$HMISS deepq-vs-pskld:(${DQP:0:200})" ;; esac
FNS="log run mkd restart check down stopfile best_vars desc serve_args xlog armj armn pinj stackenv keeplog thermline cmpfile idrec identw modes vramstart fn4get fn4set nudge fn4code pcreset arm msnap mdelta pskstat ssnap sdelta swnote gate1 smoke deepx coldabba hitset hitident copyx gsalone loadx loadabba summary dstart"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee sha256sum; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
for a in A D 4 8; do python3 -c 'import json,sys; json.loads(sys.argv[1])' "$(pinj $a)" 2>/dev/null || HMISS="$HMISS json:$a"; done
[ "$(sha256sum $MODELDIR/$SIDECAR 2>/dev/null | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256 2>/dev/null)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR"
[ "$(sha256sum $MODELDIR/$SIDE 2>/dev/null | cut -c1-64)" = "$SIDESHA" ] || HMISS="$HMISS sha256:$MODELDIR/$SIDE(want ${SIDESHA:0:16})"
grep -q 'XS2_ENVFILE' $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no env_file)"
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config refuses it)"
HMISS=$(echo $HMISS)
[ -z "$HMISS" ] || { xlog "VIRHE helpers / inputs do not resolve: $HMISS"; exit 1; }
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing, pskld.py $ST, deepq.py = pskld.py's deep request ($DQP), all $(echo $FNS | wc -w) functions resolve, tools on PATH, sidecars $SIDECAR + $SIDE sha256 OK, yaml accepted -- OK"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { xlog "VIRHE $M225 missing or not verified"; exit 1; }
  for f in $RD/dsd-d0-gs-c1.jsonl $RD/copy-dsd-d0-i1-run1.txt $RM/deep-dsm-225.txt $WARM_REF_225 $R/xdsm/psk/ps-n5-ld-1A.txt; do [ -s $f ] || xlog "note -- reference $f missing (not compared)"; done
  xlog "$ITEM: helpers md5: pskld.py $(md5sum < $PL | cut -c1-8) (longid2.py $(md5sum < $HD/../longid2.py 2>/dev/null | cut -c1-8)), deepq.py $(md5sum < $DQ | cut -c1-8), greedyset.py $(md5sum < $GS | cut -c1-8), mb.py $(md5sum < $MBP | cut -c1-8)"
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDE $RD/dsd-d0-gs-c1.jsonl $RD/copy-dsd-d0-i1-run1.txt $RM/deep-dsm-225.txt $WARM_REF_225 $RS3/s3-n5-stats0.json $R/xdsm/psk/ps-n5-ld-1A.txt; do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  for a in A D 4 8; do echo "+ pin $a: $(pinj $a)"; done
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(information here; the real run refuses to start beside one)"
fi
mkd $O $O/guard $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt
stopfile || dstart
summary
unset $SWITCHES
down
xlog "$ITEM: done -- $O"
