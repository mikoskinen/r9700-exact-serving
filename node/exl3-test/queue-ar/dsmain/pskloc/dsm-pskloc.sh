#!/bin/bash
# dsm-pskloc.sh (11.11.2026, queue-ar/dsmain/pskloc, PSK cone-exactness localiser for ds-main-port) -- NOT a queue item; the body
# of 57-p4zyzzzzz-ds-r00y-pskloc.sh. Log tag "RESULT xpsl ", out runs/xdsm/pskloc (XPL_OUT). = psk/dsm-psk.sh 841ce920 (r00z's
# body: its start, pin, helpers) + lever BI's engine tracer (RADIANCE_DS41_BITRACE=/stats/bt-pl, _PMAX XPL_PMAX) + pskloc.py.
# WHY: r00z (19:14-19:27) -- lever PSK design 1 (ds41_psk 1) changes the alone answers at 136k / 64k / 16k (4096-row pure prefill
# chunks: layers 21-39 cut) while the loaded 64k (2048-row mixed chunks: layers 25-39 cut) is IDENTICAL to A. Which op, at which
# layer, first differs on a row the cut must leave exact?
# ONE NST 5 START = r00z's (the z0occ2 start + RADIANCE_DS41_PSK=0 + the tracer; XPL_EXTRA adds start words), in process: the arms
# of XPL_ORDER (A = ds41_psk 0, B = 1, C/D/E = XPL_ARMVALS), each a COLD deepq prompt at depth XPL_DEPTH (16000 = r00z's 15777-token
# prompt: 4096-row pure prefill chunks; its A / B answers are r00z's d16000 references, so the traced answers also check the
# tracer's neutrality), a prefix-cache reset before each, NTOK greedy tokens, temperature 0; every row of positions < XPL_PMAX
# fingerprinted at every traced op (h0 xn qr q kvn iq iw sel o oa attn_out h1 ffn_in router ids ew sout moe_out a layer, final,
# logits; eager, one sync a step). Then pskloc.py: per arm vs A the first (chunk, layer, op) differing on a TRUSTED row (the
# cone's pre-attention rows, its post-attention rows from e - C_l + 127; stale rows ignored), the window kvn / taps / logits.
# (Default depth 8000 / pmax 8192: two pure prefill chunks, 4096 rows then ~3.8k, every row traced, ~0.55 GB with the warm-up's
# first 8192 rows; XPL_DEPTH=16000 + XPL_PMAX=16384 is r00z's d16000 prompt with its untraced answers as references.)
# XPL_DEEP136 (arms, default none): the 136k deepq prompt cold under those arms vs dsm-225 (= r00z's A).
# NO cc_sim, no 2.9, no power / host / fan settings; production compose files read only. Measurement only.
# Knobs: XPL_IMAGE, XPL_ITEM, XPL_OUT, XPL_ORDER (AB), XPL_ARMVALS ("A=0 B=1"), XPL_CUT (pskloc.py --cut, "B=21:160:128"), XPL_PAIRS
#   (extra pskloc.py comparisons, e.g. "B:C"), XPL_PROV (the image's provenance file, psk-PROVENANCE.txt),
#   XPL_DEPTH (16000), XPL_NTOK (512), XPL_PMAX (16384), XPL_DEEP136 (""), XPL_EXTRA (""), XPL_LITS (extra image literals).
# DRYRUN=1 bash dsm-pskloc.sh : prints every command; checks the image literals, every helper (pskloc.py selftest), the sidecars,
# the yaml; no server, no GPU, writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/../ds41-stack2-tp4.yaml                   # stack2's yaml, read only (env_file ${XS2_ENVFILE} carries every start word)
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh
O=${XPL_OUT:-$R/xdsm/pskloc}; export STATSDIR=$O/stats
RP=$R/xdsm/psk   # r00z
RD=$R/xdsm/draft; RM=$R/xdsm; RS3=$R/xdsm/stack3
export IMAGE=${XPL_IMAGE:?set XPL_IMAGE}
ITEM=${XPL_ITEM:-57-p4zyzzzzz-ds-r00y-pskloc}
ORDER=${XPL_ORDER:-AB}; ARMVALS=${XPL_ARMVALS:-A=0 B=1}; CUT=${XPL_CUT:-B=21:160:128}; DEPTH=${XPL_DEPTH:-8000}
PMAX=${XPL_PMAX:-8192}; DEEP136=${XPL_DEEP136:-}; EXTRA=${XPL_EXTRA:-}; XLITS=${XPL_LITS:-}; PAIRS=${XPL_PAIRS:-}
DS=${XPS_DS:-1}; OCCK=32; NTOK=${XPL_NTOK:-512}
SIDE=${XPS_SIDECAR:-dspark-dlw-q8_0.rad}; SIDESHA=${XPS_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
case $ORDER in *[!A-E]*|"") echo "XPL_ORDER takes arm letters A..E (got '$ORDER')"; exit 1 ;; esac
case "$ORDER" in A*) ;; *) echo "XPL_ORDER must start with A (the reference)"; exit 1 ;; esac
for v in $DEPTH $PMAX $NTOK; do case $v in ''|*[!0-9]*) echo "XPL_DEPTH / XPL_PMAX / XPL_NTOK must be numbers (got '$v')"; exit 1 ;; esac; done
armval() { local kv; for kv in $ARMVALS; do [ "${kv%%=*}" = "$1" ] && { echo "${kv#*=}"; return 0; }; done; return 1; }
for a in $(echo $ORDER | grep -o .) $DEEP136; do armval $a > /dev/null || { echo "arm $a has no value in XPL_ARMVALS ($ARMVALS)"; exit 1; }; done
case $DS in ''|*[!0-9]*) echo "XPS_DS must be a draft_sample value 0..200 (got '$DS')"; exit 1 ;; esac
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
DQ=$QA/deepq.py; PL=$HD/pskloc.py; MBP=$D/mb.py; TESTS=$D/ds41_tests.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
xlog() { log "RESULT xpsl $*"; }
# ---------------------------------------------------------------- the z0occ2 item's start + RADIANCE_DS41_PSK=0
GPTAB=q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422
SIDECAR=dspark-conf.f32
LEVENV="RADIANCE_DS41_GG_PF=1 RADIANCE_DS41_GG_PF_TABLE=$GPTAB RADIANCE_DS41_F32_PF=25 RADIANCE_DS41_ARENA_DECODE_ROWS=8,48
  RADIANCE_DS41_ROUTER_BF16=1 RADIANCE_DS41_EMB_RAM=1 RADIANCE_PLACE_SHADOW_SLOTS=16 RADIANCE_XPREFETCH=0 RADIANCE_DS41_DSPARK_WALK=1
  RADIANCE_DS41_ARGATHER_MAX=48 RADIANCE_DS41_QNFOLD=1 RADIANCE_DS41_SWFOLD=1 RADIANCE_DRAFT_CTXSKIP=0 RADIANCE_DS41_DSPARK_CONF=/models/$SIDECAR
  RADIANCE_DRAFT_LEND=0 RADIANCE_DRAFT_PROBE_BACKOFF=0 RADIANCE_DRAFT_LEND_BORROW=0 RADIANCE_DRAFT_SAMPLE=0 RADIANCE_WEIGHT_SIDECAR=/models/$SIDE
  RADIANCE_DS41_BI=0 RADIANCE_DRAFT_CTXSKIP_SLOTS=129 RADIANCE_DRAFT_LEND_WEIGHTS=0 RADIANCE_DS41_PSK=0
  RADIANCE_DS41_BITRACE=/stats/bt-pl RADIANCE_DS41_BITRACE_PMAX=$PMAX $EXTRA"
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
LITS="$LITS|lever,BI's,tracer,ON,(RADIANCE_DS41_BITRACE=/stats/bt-pl):|rows,at,positions,below,$PMAX;,passes,not,recorded"
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": 8, "drafter_mx": 1, "gg_pf": 1, "f32_pf": 25, "dspark_walk": 1, "ds41_argather": 48, "idx_split": 1, "ds41_qnfold": 1, "ds41_swfold": 1, "draft_ctxskip": 1, "dspark_conf": 1, "dspark_conf_rows": 1, "draft_lend": 1, "draft_probe_backoff": 3, "draft_hybrid": 1}'
FIX=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update({"ds41_bi": 2, "draft_sample": int(sys.argv[2]), "draft_lend_hyst": 0, "arena_d0": 0, "draft_lend_w": 0, "draft_occ": 0, "draft_occ_k": int(sys.argv[3])}); print(json.dumps(d))' "$FIX5" "$DS" "$OCCK") || { echo "pin JSON"; exit 1; }
armj() { echo "{\"ds41_psk\": $(armval $1)}"; }
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xpsl $1 warm-up identity vs $2"; return 0; }
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
  # the tracer records no pass (dsm-bilid3.sh's rule): rule-w in the container env, no plain rule line, the tracer line
  NR=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_AQL_ACQ=rule)')
  docker logs $CN 2>&1 | grep -q "lever BI's tracer cannot open" && { xlog "$1: VIRHE the tracer could not open its files"; ok=0; }
  echo "$CE" | grep -qx 'RADIANCE_AQL_ACQ=rule-w' && [ "$NR" = 0 ] || { xlog "$1: VIRHE rule-w not in the container env or a plain rule line ($NR)"; ok=0; }
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
  [ "$DRY" = 1 ] && { echo "+ GET /stats > $O/$1-stats0.json; RESULT xpsk $1 VRAM at start (experts per tier; vs stack3 s3-n5)"; return 0; }
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
pskstat() { [ "$DRY" = 1 ] && { echo "(dry)"; return 0; }; docker logs $CN 2>&1 | grep 'lever PSK:' | tail -1 | sed 's/^radiance: deepseek41: //' | cut -c1-300; }
swnote() {   # swnote TAG: start values; every arm of ORDER (+ DEEP136) applied at a step, then A
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 at start (ds41_psk 0, ds41_bi 0, gg_gemv 2, aql_acq 2, K1 three 1, draft_sample 0, draft_capable 1, epoch 0); arms $(for a in $(echo $ORDER | grep -o . | sort -u) $DEEP136; do echo -n "$a $(armj $a) "; done)applied, then A"; return 0; }
  local s0 g a sets="" bad=""
  s0=$(fn4get)
  for a in $(echo $ORDER | grep -o . | sort -u) $DEEP136 A; do
    arm $a; g=$(fn4get)
    python3 -c 'import json,sys
g, w = json.loads(sys.argv[1]), json.loads(sys.argv[2])
sys.exit(0 if all(g["switches"].get(k) == v for k, v in w.items()) and not g["pending"] else 1)' "$g" "$(pinj $a)" 2>/dev/null || bad="$bad $a"
    sets="$sets $a:$(echo "$g" | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print("psk=%s,bi=%s,ds=%s,ep=%s" % (w.get("ds41_psk"), w.get("ds41_bi"), w.get("draft_sample"), d.get("epoch")))' 2>/dev/null)"
  done
  xlog "$1 switch note: at start $(echo "$s0" | cut -c1-1000) | arms applied:$sets$([ -n "$bad" ] && echo " -- VIRHE not applied:$bad")"
  python3 -c 'import json,sys
s0 = json.loads(sys.argv[1]); w = s0["switches"]
ok = w.get("ds41_psk") == 0 and w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill"))
ok = ok and w.get("ds41_bi") == 0 and s0["epoch"] == 0 and w["draft_adaptive"] == 1 and s0["draft_capable"] == 1 and w.get("draft_sample") == 0
sys.exit(0 if ok else 1)' "$s0" 2>/dev/null && [ -z "$bad" ]
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
deepx() {   # deepx TAG ARM DEPTH NAME [noreset]: deepq.py's fixed prompt at DEPTH, 512 greedy tokens -> TTFT, cached, acceptance, ms
  local tag=$1 a=$2 dep=$3 nm=$4 t0=$(date +%s) j md
  arm $a
  [ "${5:-}" = noreset ] || pcreset $tag-$nm
  msnap $tag-$nm-0
  run "(cd $D && timeout 1800 python3 $DQ $O/$tag-$nm.txt $dep $NTOK > $O/$tag-$nm.out 2>&1)"
  msnap $tag-$nm-1
  [ "$DRY" = 1 ] && { echo "+ RESULT xpsl $tag $nm: arm $a depth $dep: TTFT, cached, acceptance, non-decode step ms -> $O/deep.jsonl"; return 0; }
  j=$(grep -m1 '^DEEPQ ' $O/$tag-$nm.out | cut -c7-); md=$(mdelta $tag-$nm-0 $tag-$nm-1)
  [ -n "$j" ] || { xlog "$tag $nm: VIRHE no answer ($(tail -2 $O/$tag-$nm.out | tr '\n' ' ' | cut -c1-300))"; return 1; }
  python3 -c 'import json,sys
d = json.loads(sys.argv[1]); d.update(json.loads(sys.argv[2]) if sys.argv[2].strip() else {})
d.update(name=sys.argv[3], arm=sys.argv[4], depth=int(sys.argv[5]), reset=sys.argv[6] != "noreset", text=sys.argv[7])
print(json.dumps(d))' "$j" "$md" "$nm" "$a" "$dep" "${5:-reset}" "$O/$tag-$nm.txt" >> $O/deep.jsonl
  xlog "$tag $nm (arm $a $(armj $a), deepq depth $dep, $NTOK greedy tokens$([ "${5:-}" = noreset ] && echo ", NO reset")): $(python3 -c 'import json,sys
d, m = json.loads(sys.argv[1]), json.loads(sys.argv[2] or "{}")
print("status %s, prompt %s tok, cached %s, TTFT %s s (%s prefill tok/s), %s tok, %s chars | decode acceptance %s (%s of %s drafted) | non-decode steps %s, %s ms mean" % (
    d.get("status"), d.get("prompt_tokens"), d.get("cached"), d.get("ttft"),
    round((d["prompt_tokens"] - (d.get("cached") or 0)) / d["ttft"], 1) if d.get("ttft") and d.get("prompt_tokens") else "-",
    d.get("tokens"), d.get("chars"), m.get("acc"), m.get("draft_acc"), m.get("draft_tok"), m.get("nondecode_steps"), m.get("nondecode_ms")))' "$j" "$md" 2>&1 | cut -c1-500) | psk: $(pskstat) | $(thermline $t0 $(date +%s))"
}
btcheck() {   # btcheck TAG: after the start's warm-up + one request, every rank's .bin has records and its .names is there
  local d=$STATSDIR/bt-pl r miss=""
  [ "$DRY" = 1 ] && { echo "+ btcheck $1: a 4-token request, then $d/bt-r{0..3}-*.bin non-empty and bt-r{0..3}-*.names present"; return 0; }
  nudge
  for r in 0 1 2 3; do
    [ -s "$(ls $d/bt-r$r-*.bin 2>/dev/null | head -1)" ] || miss="$miss r$r.bin"
    ls $d/bt-r$r-*.names > /dev/null 2>&1 || miss="$miss r$r.names"
  done
  [ -z "$miss" ] && { xlog "$1 tracer files: every rank writing ($(ls -la $d 2>/dev/null | awk 'NR > 1 && /bt-r/ {n++; b += $5} END {printf "%d files, %.1f MB", n, b / 1e6}'))"; return 0; }
  xlog "$1: VIRHE tracer wrote no records (missing:$miss) -- env [$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null | grep '^RADIANCE_DS41_BITRACE' | tr '\n' ' ')], engine: $(docker logs $CN 2>&1 | grep -E "tracer (ON|cannot open)" | head -2 | sed 's/^radiance: //' | tr '\n' '|' | cut -c1-400)"
  return 1
}
tracerun() {   # tracerun TAG: the arms of ORDER, each a cold deepq prompt at DEPTH; identity vs r00z's d16000 answers and vs the first A
  local tag=$1 a k=0 fa="" ref
  for a in $(echo $ORDER | grep -o .); do
    k=$((k + 1)); stopfile && return 0
    deepx $tag $a $DEPTH tr-$k$a
    [ "$DRY" = 1 ] && continue
    [ -n "$fa" ] || { [ $a = A ] && fa=$O/$tag-tr-$k$a.txt; }
    [ -n "$fa" ] && [ "$O/$tag-tr-$k$a.txt" != "$fa" ] && idrec "tr-$k$a-vs-trA" "$(cmpfile $fa $O/$tag-tr-$k$a.txt)" "depth $DEPTH traced, vs the first traced A"
    if [ "$DEPTH" = 16000 ]; then
      ref=""; case $(armval $a) in 0) ref=$RP/ps-n5-d16000-1A.txt ;; 1) [ -z "$EXTRA" ] && ref=$RP/ps-n5-d16000-2B.txt ;; esac
      [ -n "$ref" ] && idrec "tr-$k$a-vs-r00z" "$(cmpfile $ref $O/$tag-tr-$k$a.txt)" "traced vs r00z's untraced d16000 answer of ds41_psk $(armval $a) (the tracer's neutrality)"
    fi
  done
  run "du -sh $STATSDIR/bt-pl > $O/$tag-trace-size.txt 2>&1"
}
deep136() {   # deep136 TAG: the 136k deepq prompt cold under each DEEP136 arm vs dsm-225 (= r00z's A, IDENTICAL 2 of 2)
  local tag=$1 a
  for a in $DEEP136; do
    stopfile && return 0
    deepx $tag $a 135000 d136-$a
    [ "$DRY" = 1 ] && continue
    idrec "d136-$a-vs-dsm225" "$(cmpfile $RM/deep-dsm-225.txt $O/$tag-d136-$a.txt)" "136k cold, ds41_psk $(armval $a), vs dsm-225 (r00z A: IDENTICAL)"
  done
}
btanalyze() {   # btanalyze TAG: pskloc.py over the tracer's files -> RESULT lines
  [ "$DRY" = 1 ] && { echo "+ python3 -I $PL analyze $STATSDIR/bt-pl --order $ORDER --cut '$CUT' --pairs '$PAIRS' --json $O/$1-pskloc.json > $O/$1-pskloc.txt"; return 0; }
  timeout 1800 python3 -I $PL analyze $STATSDIR/bt-pl --order $ORDER --cut "$CUT" --pairs "$PAIRS" --json $O/$1-pskloc.json > $O/$1-pskloc.txt 2>&1
  local l; while IFS= read -r l; do xlog "$1 $(echo "$l" | cut -c1-2500)"; done < <(grep -E '^PSKLOC' $O/$1-pskloc.txt)
  grep -q '^PSKLOC summary' $O/$1-pskloc.txt || xlog "$1 pskloc: VIRHE no summary ($(tail -3 $O/$1-pskloc.txt | tr '\n' ' ' | cut -c1-500))"
}
dstart() {   # the one NST 5 start
  local tag=pl-n5 ok
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on"
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV > $XS2_ENVFILE"; else printf '%s\n' $LEVENV > $XS2_ENVFILE; fi
  xlog "$tag: start ($DESC) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $FIX | arms $(for a in $(echo $ORDER | grep -o . | sort -u) $DEEP136; do echo -n "$a $(armj $a) "; done)| env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ' | cut -c1-2200)"
  if [ "$DRY" = 1 ]; then echo "+ mkdir -p $STATSDIR/bt-pl && chmod 777 $STATSDIR/bt-pl (before the start)"
  else mkdir -p $STATSDIR/bt-pl && chmod 777 $STATSDIR/bt-pl; fi
  gate1 $tag 70 || { xlog "$tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  modes $tag || { xlog "$tag: VIRHE the modes are not the start's -> stopped"; down; return 1; }
  if [ "$DRY" != 1 ]; then
    ok=1
    docker logs $CN 2>&1 | grep -q "DSpark drafter" || { xlog "$tag: VIRHE no DSpark drafter line"; ok=0; }
    docker logs $CN 2>&1 | grep -q "draft depth: adaptive" || { xlog "$tag: VIRHE no adaptive draft depth line"; ok=0; }
    [ $ok = 1 ] || { keeplog $tag-nodraft; down; return 1; }
  fi
  swnote $tag || { xlog "$tag: VIRHE the switches are not as expected -> stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  btcheck $tag || { keeplog $tag-tracer; down; return 1; }
  stopfile || tracerun $tag
  stopfile || deep136 $tag
  run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
  xlog "$tag: PSK stats at the end: $(pskstat) | $(docker logs $CN 2>&1 | grep -cE '^E ') E lines"
  down
  btanalyze $tag
  xlog "$tag: done"
}
trap 'down' EXIT
trap 'exit 143' TERM INT
# ---------------------------------------------------------------- main
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) | one NST 5 start + lever BI's tracer (pmax $PMAX): $DESC${EXTRA:+ + $EXTRA} | arms [$ARMVALS] in order $ORDER, each a cold deepq depth $DEPTH ($NTOK greedy tokens); 136k cold under [${DEEP136:-none}] | pskloc.py --cut '$CUT' | refs r00z d16000 A / B, dsm-225 deep"
IMGLITS="B|dspark_conf_rows, gg_pf, f32_pf, dspark_walk, ds41_bi, draft_sample, draft_lend_hyst, arena_d0, draft_lend_w, draft_occ, draft_occ_k, ds41_psk, epoch_bump
B|ds41_psk
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
PROV=${XPL_PROV:-psk-PROVENANCE.txt}
IMGLITS="$IMGLITS
A|lever BI's tracer ON (RADIANCE_DS41_BITRACE=
A|bi_rowhash$([ -n "$XLITS" ] && printf '\n%s' "$XLITS")"
IMGMISS=$(echo "$IMGLITS" | timeout 120 docker run --rm -i --network none --entrypoint sh -e PROV=$PROV $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels/libr4d.so; S=/opt/radiance/share/radiance/kernels/spechub.so; n=0
while IFS= read -r l; do [ -n "$l" ] || continue; n=$((n + 1)); w=${l%%|*}; s=${l#*|}
  case $w in B) f=$B ;; A) f=$A ;; K) f=$K ;; S) f=$S ;; *) f="$B $A $K" ;; esac
  o=0; for x in $f; do grep -qzF -- "$s" $x && { o=1; break; }; done
  [ $o = 1 ] || echo "MISS $w|$s"; done
test -x /opt/radiance/bin/rad-sidecar || echo "MISS bin/rad-sidecar"; test -s /opt/radiance/share/radiance/$PROV || echo "MISS $PROV"; echo "CHECKED $n"' 2>&1)
case "$IMGMISS" in "CHECKED "*) ;; *) xlog "VIRHE $IMAGE is not the ds41-psk image -- missing: $(echo "$IMGMISS" | grep -v '^CHECKED' | tr '\n' ';' | cut -c1-900)"; exit 1 ;; esac
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)) is the ds41-psk image: all $(echo "$IMGMISS" | sed 's/CHECKED //') literals present (the 35-switch registry with ds41_psk, lever PSK, the int5 levers, the /metrics counters) + bin/rad-sidecar + psk-PROVENANCE -- OK"
HMISS=""
for f in "$LIB29" "$LIBPR" $Y $GS $DQ $PL $MBP $TESTS $D/fastcheck.py; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $GS $DQ $PL $MBP $TESTS $D/fastcheck.py 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
ST=$(timeout 60 python3 -I $PL selftest 2>&1 | tail -1); case "$ST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$PL(${ST:0:300})" ;; esac
DQP="DQ OK"
FNS="log run mkd restart check down stopfile best_vars desc serve_args xlog armval armj pinj stackenv keeplog thermline cmpfile idrec identw modes fn4get fn4set nudge fn4code pcreset arm msnap mdelta pskstat swnote gate1 deepx btcheck tracerun deep136 btanalyze dstart"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee sha256sum; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
for a in $(echo $ORDER | grep -o . | sort -u) $DEEP136; do python3 -c 'import json,sys; json.loads(sys.argv[1])' "$(pinj $a)" 2>/dev/null || HMISS="$HMISS json:$a"; done
[ "$(sha256sum $MODELDIR/$SIDECAR 2>/dev/null | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256 2>/dev/null)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR"
[ "$(sha256sum $MODELDIR/$SIDE 2>/dev/null | cut -c1-64)" = "$SIDESHA" ] || HMISS="$HMISS sha256:$MODELDIR/$SIDE(want ${SIDESHA:0:16})"
grep -q 'XS2_ENVFILE' $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no env_file)"
grep -qE '^ *- \$\{STATSDIR:\?set STATSDIR\}:/stats *$' $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no STATSDIR:/stats mount for the tracer)"
[ "$DEPTH" = 16000 ] && for f in $RP/ps-n5-d16000-1A.txt $RP/ps-n5-d16000-2B.txt; do [ -s $f ] || HMISS="$HMISS ref:$f"; done
[ -n "$DEEP136" ] && { [ -s $RM/deep-dsm-225.txt ] || HMISS="$HMISS ref:$RM/deep-dsm-225.txt"; }
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config refuses it)"
HMISS=$(echo $HMISS)
[ -z "$HMISS" ] || { xlog "VIRHE helpers / inputs do not resolve: $HMISS"; exit 1; }
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing, pskloc.py $ST, all $(echo $FNS | wc -w) functions resolve, tools on PATH, sidecars $SIDECAR + $SIDE sha256 OK, yaml accepted -- OK"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { xlog "VIRHE $M225 missing or not verified"; exit 1; }
  [ -s $WARM_REF_225 ] || xlog "note -- reference $WARM_REF_225 missing (not compared)"
  xlog "$ITEM: helpers md5: pskloc.py $(md5sum < $PL | cut -c1-8), deepq.py $(md5sum < $DQ | cut -c1-8), body $(md5sum < $0 | cut -c1-8); disk free $(df -h / | awk 'NR == 2 {print $4}')"
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDE $RP/ps-n5-d16000-1A.txt $RP/ps-n5-d16000-2B.txt $RM/deep-dsm-225.txt $WARM_REF_225; do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  for a in $(echo $ORDER | grep -o . | sort -u) $DEEP136; do echo "+ pin $a: $(pinj $a)"; done
  echo "+ disk free: $(df -h / | awk 'NR == 2 {print $4}') (the trace: ~0.18 GB a long prompt at pmax 8192, the warm-up's included)"
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(information here; the real run refuses to start beside one)"
fi
mkd $O $O/guard $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt
# the disk: the trace takes ~0.18 GB a long prompt at pmax 8192 (4 ranks x ~5.6 kB a row; the warm-up's prefill too); refuse under 4 GB free
FREEK=$(df -k / | awk 'NR == 2 {print $4}')
[ "${FREEK:-0}" -ge 4194304 ] || { xlog "VIRHE only $((FREEK / 1024)) MB free on / -- not tracing"; exit 1; }
stopfile || dstart
unset $SWITCHES
down
xlog "$ITEM: done -- $O"
