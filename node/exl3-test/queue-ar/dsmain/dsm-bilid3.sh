#!/bin/bash
# dsm-bilid3.sh (= dsm-bilid.sh with the tracer directory made before the start; a separate name while bi-b-trace2 runs the old file) (11.11.2026, queue-ar/dsmain, ds41-bi, lever BI) -- NOT a queue item; the PERMANENT identity-under-load body
# (dsm-loadid.sh acb47bbc, generalised; that file stays untouched). Log tag "RESULT xbil ", out XBIL_OUT
# (default runs/xdsm/bilid). USER RULING 11.11.: an answer that changes under load vs alone is a BUG; this body is lever BI's
# acceptance test: target 20 / 20 IDENTICAL loaded vs alone (c4 + a concurrent ~32k prefill) and c4 run-to-run, NST 0 and NST 5.
# Per start (stack2's env, serve line and pins, as dsm-loadid.sh): the warm-up identity, then for each value v of XBIL_SWVALS
# (an in-process fn4 switch, XBIL_SW; empty = one pass with no switch):
#   a1-v  the 20-prompt greedy set c1 ALONE (greedyset.py, temperature 0, 192 tok);
#   ld-v  the same 20 prompts, each under load (loadid.py: mb c4 streams + one fresh ~XBIL_PFTOK prefill at +XBIL_OFFSET s);
#   c4a-v / c4b-v (XBIL_C4RR=1)  the greedy set at conc 4, twice (run-to-run);
# then a2 (XBIL_CTRL=1): alone again at the first value (control). Compares (loadid.py cmp, byte for byte + first differing
# char / token): ld-v vs a1-v (THE measurement), c4b-v vs c4a-v (run-to-run), c4a-v vs a1-v (c4 vs alone), a1-v vs a1-<first>
# (the alone line's identity across switch values: a BI fix must leave it byte-identical), a1 vs bz d0, a2 vs a1.
# XBIL_TRACE=1: lever BI's engine tracer on (RADIANCE_DS41_BITRACE=/stats/bt-<start>, _PMAX XBIL_PMAX, needs a ds41-bi image),
#   then bitrace.py analyze: per probe the first op whose row differs loaded vs alone, with both steps' shapes.
# Knobs: XBIL_IMAGE, XBIL_ITEM, XBIL_OUT, XBIL_STARTS ("n0 n5"), XBIL_OFFSET (0.5), XBIL_PFTOK (32768), XBIL_CONC (4), XBIL_WARM
#   (10), XBIL_MT (192), XBIL_CTRL (1|0), XBIL_VRAMENV (RADIANCE_PLACE_TURNOVER=0), XBIL_EXTRAENV (NAME=VALUE words added to both
#   starts' env file, e.g. RADIANCE_DS41_BI=1; each must reach the container), XBIL_PINX (JSON merged into both pins),
#   XBIL_SW + XBIL_SWVALS (switch name, values), XBIL_C4RR (0|1), XBIL_TRACE (0|1), XBIL_PMAX (256), XBIL_LITS (extra image
#   literals, "W|text" lines: B radiance, A deepseek41.so, K libr4d.so, X any).
# DRYRUN=1 bash dsm-bilid.sh : prints every command; checks the image (its literals), every helper (files, parse, loadid.py and
#   bitrace.py selftests, every function via type -t), the sidecar and the yaml; no server, no GPU, writes nothing.
# NO cc_sim, no 2.9. No power / host / fan settings; production compose files read only (--project-directory).
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-stack2-tp4.yaml                      # stack2's yaml, read only
O=${XBIL_OUT:-$R/xdsm/bilid}; export STATSDIR=$O/stats
RD=$R/xdsm/draft                                # bz's d0 files (the greedy set reference, NST 0 stock)
export IMAGE=${XBIL_IMAGE:?set XBIL_IMAGE}
ITEM=${XBIL_ITEM:-dsm-bilid}
STARTS=${XBIL_STARTS:-n0 n5}; OFF=${XBIL_OFFSET:-0.5}; PFT=${XBIL_PFTOK:-32768}; CONC=${XBIL_CONC:-4}; WARM=${XBIL_WARM:-10}
MT=${XBIL_MT:-192}; CTRL=${XBIL_CTRL:-1}
EXTRAENV=${XBIL_EXTRAENV:-}; PINX=${XBIL_PINX:-}; SW=${XBIL_SW:-}; SWVALS=${XBIL_SWVALS:-}; C4RR=${XBIL_C4RR:-0}
TRACE=${XBIL_TRACE:-0}; PMAX=${XBIL_PMAX:-256}; XLITS=${XBIL_LITS:-}
for kv in $EXTRAENV; do case "$kv" in [A-Z]*=*) ;; *) echo "XBIL_EXTRAENV takes NAME=VALUE words (got '$kv')"; exit 1 ;; esac; done
[ -z "$PINX" ] || python3 -c 'import json,sys; d=json.loads(sys.argv[1]); assert isinstance(d, dict)' "$PINX" 2>/dev/null || { echo "XBIL_PINX must be a JSON object (got '$PINX')"; exit 1; }
[ -z "$SWVALS" ] || [ -n "$SW" ] || { echo "XBIL_SWVALS needs XBIL_SW"; exit 1; }
for v in $SWVALS; do case $v in ''|*[!0-9]*) echo "XBIL_SWVALS takes integers (got '$v')"; exit 1 ;; esac; done
case "$C4RR$TRACE" in [01][01]) ;; *) echo "XBIL_C4RR / XBIL_TRACE must be 0 or 1"; exit 1 ;; esac
case $PMAX in ''|*[!0-9]*) echo "XBIL_PMAX must be a number"; exit 1 ;; esac
BT=$HD/bitrace.py
MB=${XBIL_MB:-}; MBORDER=${XBIL_MBORDER:-${SWVALS:-_}}; MBDUR=${XBIL_MBDUR:-60}; MBWARM=${XBIL_MBWARM:-15}
case "$MB" in *[!a-z0-9,]*) echo "XBIL_MB: a comma list of mb.py phases (got '$MB')"; exit 1 ;; esac
for v in $MBORDER; do case " ${SWVALS:-_} " in *" $v "*) ;; *) echo "XBIL_MBORDER: '$v' is not one of XBIL_SWVALS (${SWVALS:-_})"; exit 1 ;; esac; done
case "$MBDUR$MBWARM" in *[!0-9]*) echo "XBIL_MBDUR / XBIL_MBWARM must be numbers"; exit 1 ;; esac
REFDIR=${XBIL_REFDIR:-$R/xdsm/loadid}   # the untraced loadid's compares (iej-z0ad-loadid), for the tracer's reproduction check
for s in $STARTS; do case $s in n0|n5) ;; *) echo "XBIL_STARTS takes n0 / n5 (got '$s')"; exit 1 ;; esac; done
case $CTRL in 0|1) ;; *) echo "XBIL_CTRL must be 0 or 1 (got '$CTRL')"; exit 1 ;; esac
for v in OFF PFT CONC WARM MT; do case ${!v} in ''|*[!0-9.]*) echo "$v must be a number (got '${!v}')"; exit 1 ;; esac; done
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
LD=$HD/loadid.py
MBP=$D/mb.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
# ---------------------------------------------------------------- the stack2 configuration as hxzzy-stack2 ran it (fixed)
# (hxzzy-stack2's wrapper: HYB=1 XS2_DRAFTER_MX=1 XS2_GGPF=1 XS2_F32PF=25 XS2_WALK=1 XS2_SMALL5=1 XS2_VRAM=all XS2_LEND=1
#  XS2_BORROW=0, turnover 0, match 8; its st-nst0.env / st-stack.env / -meta.txt pins are what these produce)
ARMS=stack; HYB=1; MATCH=8; DMX=1; DUR=75
GGPF=1; F32PF=25; WALK=1; SMALL5=1; LEND=1; BORROW=0; VRAMSEL=all
GPTAB=q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422
SIDECAR=dspark-conf.f32                         # MODELDIR's confidence head (the release's), /models/dspark-conf.f32 in the container
# ---------------------------------------------------------------- copied from dsm-stack2.sh (VRAM table .. swnote; tag xs2 -> xbil)
# THE VRAM TABLE (hxz-vramstack's start levers, one a line): name  NST 0 env  NST 5 env  NST 0 log literal  NST 5 log literal
#   (env NAME=VALUE[,..]; a log literal with ',' for a space the start log must carry when on, '-' none; a literal holds no comma)
VTAB='
arena   RADIANCE_DS41_ARENA_DECODE_ROWS=8  RADIANCE_DS41_ARENA_DECODE_ROWS=48  activation,arena:,8,tokens,use  activation,arena:,48,tokens,use
bf16    RADIANCE_DS41_ROUTER_BF16=1        RADIANCE_DS41_ROUTER_BF16=1         -                               -
emb     RADIANCE_DS41_EMB_RAM=1            RADIANCE_DS41_EMB_RAM=1             a,rank),ram                     a,rank),ram
shadow  RADIANCE_PLACE_SHADOW_SLOTS=16     RADIANCE_PLACE_SHADOW_SLOTS=16      -                               -
xpoff   RADIANCE_XPREFETCH=0               RADIANCE_XPREFETCH=0                expert,prefetch:,off            expert,prefetch:,off
'
VNAMES=$(echo "$VTAB" | awk 'NF && $1 !~ /^#/ {print $1}' | xargs)
[ "$VRAMSEL" = all ] && VRAMSEL=$VNAMES
for v in $VRAMSEL; do case " $VNAMES " in *" $v "*) ;; *) echo "XS2_VRAM: no VRAM lever '$v' (have: $VNAMES, or all)"; exit 1 ;; esac; done
vf() { echo "$VTAB" | awk -v n="$1" -v k="$2" '$1 == n {print $k; exit}'; }   # vf LEVER FIELD
# levenv n0|n5: every start-time lever word of that start (exported by stackenv and written to the start's env file)
levenv() {
  local v w=""
  [ $GGPF = 1 ] && w="$w RADIANCE_DS41_GG_PF=1 RADIANCE_DS41_GG_PF_TABLE=$GPTAB"
  [ $F32PF != 0 ] && w="$w RADIANCE_DS41_F32_PF=$F32PF"
  for v in $VRAMSEL; do w="$w $(vf $v $([ $1 = n0 ] && echo 2 || echo 3) | tr ',' ' ')"; done
  if [ $1 = n5 ]; then
    [ $WALK != 0 ] && w="$w RADIANCE_DS41_DSPARK_WALK=$WALK"
    [ $SMALL5 = 1 ] && w="$w RADIANCE_DS41_ARGATHER_MAX=48 RADIANCE_DS41_QNFOLD=1 RADIANCE_DS41_SWFOLD=1 RADIANCE_DRAFT_CTXSKIP=0 RADIANCE_DS41_DSPARK_CONF=/models/$SIDECAR"
    [ $LEND = 1 ] && w="$w RADIANCE_DRAFT_LEND=0 RADIANCE_DRAFT_PROBE_BACKOFF=0 RADIANCE_DRAFT_LEND_BORROW=0"
  fi
  w="$w $EXTRAENV"
  [ $TRACE = 1 ] && w="$w RADIANCE_DS41_BITRACE=/stats/bt-$1 RADIANCE_DS41_BITRACE_PMAX=$PMAX"
  echo $w
}
# levlits n0|n5: the start-log literals the levers on at that start must leave ('|'-separated; ',' in a literal = a space)
levlits() {
  local v l x=""
  [ $GGPF = 1 ] && x="$x|gemm_nt_ggml's,prefill,rows,take,lever,GPF's,forms,(RADIANCE_DS41_GG_PF=1"
  [ $F32PF != 0 ] && x="$x|gemm_nt_f32's,prefill,rows,take,lever,GPF-b's,tiled,form,(RADIANCE_DS41_F32_PF=$F32PF"
  for v in $VRAMSEL; do l=$(vf $v $([ $1 = n0 ] && echo 4 || echo 5)); [ "$l" = - ] || x="$x|$l"; done
  if [ $1 = n5 ]; then
    [ $WALK != 0 ] && x="$x|DSpark,lever,W1:,the,full-vocabulary,Markov,walk,declared,(dspark_walk_step,a,step,+,all_gather,over,4,rank(s)|fn4,dspark_walk=$WALK,"
    [ $SMALL5 = 1 ] && x="$x|one,kernel,up,to,48,tokens,a,step|RADIANCE_DS41_QNFOLD;,ds41_qnfold=1)|RADIANCE_DS41_SWFOLD;,ds41_swfold=1)|draft,context,skip,(RADIANCE_DRAFT_CTXSKIP=0|a,confidence,plane,(the,dspark_conf,gate"
    [ $LEND = 1 ] && x="$x|draft,lend:,set,up"
  fi
  echo "${x#|}"
}
# the pins: hx-stack's + the int3 levers (NST 0: N = FIX0; NST 5: FIX + arm A / B)
P0=""; [ $GGPF = 1 ] && P0="$P0, \"gg_pf\": 1"; [ $F32PF != 0 ] && P0="$P0, \"f32_pf\": $F32PF"
P5=""; [ $WALK != 0 ] && P5="$P5, \"dspark_walk\": $WALK"
[ $SMALL5 = 1 ] && P5="$P5, \"ds41_argather\": 48, \"idx_split\": 1, \"ds41_qnfold\": 1, \"ds41_swfold\": 1, \"draft_ctxskip\": 1, \"dspark_conf\": 1, \"dspark_conf_rows\": 1"
[ $LEND = 1 ] && P5="$P5, \"draft_lend\": 1, \"draft_probe_backoff\": 3"
[ $BORROW = 1 ] && P5="$P5, \"draft_lend_borrow\": 1"
FIX='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": '"$MATCH"'}'
[ "$DMX" = 1 ] && FIX="${FIX%\}}, \"drafter_mx\": 1}"   # lever DK in the NST 5 pin (XS2_DRAFTER_MX=1)
FIX="${FIX%\}}$P0$P5}"
[ -n "$PINX" ] && FIX=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update(json.loads(sys.argv[2])); print(json.dumps(d))' "$FIX" "$PINX")
ARM_A='{"draft_hybrid": '"$HYB"'}'
ARM_B='{"draft_hybrid": 1}'
FIX0="{\"gg_gemv\": 2, \"aql_acq\": 2, \"ds41_kvfold\": 1, \"ds41_ofold\": 1, \"ds41_onefill\": 1$P0}"   # the NST 0 start's pin (arm N = {})
[ -n "$PINX" ] && FIX0=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update(json.loads(sys.argv[2])); print(json.dumps(d))' "$FIX0" "$PINX")
armset() {   # armset ARMJSON [EXTRAJSON]: FIX + the arm's switches (+ extra, e.g. epoch_bump) as one POST body; arm N: FIX0 + extra
  local x=${2:-} base=$FIX; [ -n "$x" ] || x='{}'
  [ "$1" = N ] && { base=$FIX0; set -- '{}'; }
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update(json.loads(sys.argv[2])); d.update(json.loads(sys.argv[3])); print(json.dumps(d))' "$base" "$1" "$x"
}
DESC_N="NST 0 stack2 N: gg_gemv 2, rule-w, K1 folds c/d/f$([ $GGPF = 1 ] && echo ", gg_pf 1 (table)")$([ $F32PF != 0 ] && echo ", f32_pf $F32PF"), no drafter"
DESC_A="NST 5 stack2 A: gg_gemv 2, rule-w, K1 folds c/d/f$([ $GGPF = 1 ] && echo ", gg_pf 1 (table)")$([ $F32PF != 0 ] && echo ", f32_pf $F32PF"), DSpark NST 5 adaptive, depth 0, hybrid $HYB$([ $WALK != 0 ] && echo ", dspark_walk $WALK")$([ $SMALL5 = 1 ] && echo ", small n5 set (argather 48, idx_split, qnfold, swfold, ctxskip, conf + conf_rows 1)")$([ $LEND = 1 ] && echo ", lend + back-off 3")$([ $BORROW = 1 ] && echo " + borrow")"
DESC_B="NST 5 stack2 B: the same with hybrid 1 (match $MATCH)"
[ "$DMX" = 1 ] && DESC_A="$DESC_A, drafter_mx 1"
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
  RADIANCE_DS41_ARENA_DECODE_ROWS RADIANCE_DS41_ROUTER_BF16 RADIANCE_PLACE_SHADOW_SLOTS RADIANCE_XPREFETCH"
# the exact VRAM env at both starts (empty = none): each word NAME=VALUE, NAME one of $SWITCHES (unset first, passed by the yaml)
VRAMENV=${XBIL_VRAMENV-RADIANCE_PLACE_TURNOVER=0}
for kv in $VRAMENV; do
  case "$kv" in [A-Z]*=*) ;; *) echo "XBIL_VRAMENV takes NAME=VALUE words (got '$kv')"; exit 1 ;; esac
  case " $(echo $SWITCHES) " in *" ${kv%%=*} "*) ;; *) echo "XBIL_VRAMENV: ${kv%%=*} is not in the stack's switch list (unset at start, passed by the yaml)"; exit 1 ;; esac
done
[ -n "$VRAMENV" ] && { DESC_N="$DESC_N, VRAM env $VRAMENV"; DESC_A="$DESC_A, VRAM env $VRAMENV"; }
[ -n "$VRAMSEL" ] && { DESC_N="$DESC_N, VRAM levers [$VRAMSEL]"; DESC_A="$DESC_A, VRAM levers [$VRAMSEL]"; }
stackenv() {   # stackenv n0|n5: 105's t3 arm + gg_gemv 2 + rule-w + the three K1 folds declared (and on) at start + XBIL_VRAMENV
  # + the start's int3 lever words (levenv), every other switch unset; LEVENV = those words (the env file's content)
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

identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xbil $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xbil $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  echo "warm-up-$1 $r" >> $O/identity.txt
  case $r in IDENTICAL*) log "RESULT xbil $1 warm-up answers vs stock ($(basename $2)): $r (105 tuned3l: IDENTICAL)" ;;
    *) log "RESULT xbil $1 warm-up answers vs stock ($(basename $2)): VIRHE $r -- NOT EXACT vs the tuned3l line" ;; esac
}
VKNOBS=""
vrammodes() {   # vrammodes TAG: the VRAM env in the container's env, the mover's shadow / kept-free counts a rank, the expert
  # prefetch state -> VKNOBS (the modes line prints it); 1 when the env is missing, a rank has no mover line, or turnover 0
  # was asked and a rank still keeps slots free
  local MV XP CE kv bad="" nz
  MV=$(docker logs $CN 2>&1 | grep -E '^I\[[0-9]\] mover: [0-9]+ slab slots' | sed -E 's/^I\[([0-9])\] mover: ([0-9]+) slab slots \(([0-9]+) shadow, ([0-9]+) more kept free.*/r\1 \3 shadow \4 kept free (\2 slots)/' | tr '\n' ',' | sed 's/,$//')
  XP=$(docker logs $CN 2>&1 | grep -cE '^I rank [0-9]+: expert prefetch on')
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in $VRAMENV $LEVENV; do echo "$CE" | grep -qxF "$kv" || bad="$bad ${kv%%=*}"; done
  VKNOBS="env [${VRAMENV:-none}] + lever env [$(echo $LEVENV | sed 's/RADIANCE_DS41_GG_PF_TABLE=[^ ]*/RADIANCE_DS41_GG_PF_TABLE=<table>/')]$([ -n "$bad" ] && echo " NOT in the container:$bad"); mover ${MV:-no line}; expert prefetch on at $XP of 4 ranks"
  [ -z "$bad" ] || { log "RESULT xbil $1: VIRHE the VRAM / lever env did not reach the container:$bad"; return 1; }
  [ "$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots')" = 4 ] || { log "RESULT xbil $1: VIRHE no mover line on all 4 ranks: ${MV:-none}"; return 1; }
  case " $VRAMENV " in *" RADIANCE_PLACE_TURNOVER=0 "*)
    nz=$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots \([0-9]+ shadow, 0 more kept free')
    [ "$nz" = 4 ] || { log "RESULT xbil $1: VIRHE turnover 0 asked but $nz of 4 ranks keep 0 slots free: $MV"; return 1; } ;;
  esac
  return 0
}
modes() {   # modes TAG n0|n5: the t3 line's notes + rule-w + the lever K1 note (all three declared and on) + the VRAM knobs + the int3
  # levers' declare lines (levlits) -> 0, else 1 (start stopped)
  local AR=4; [ $2 = n5 ] && [ $SMALL5 = 1 ] && AR=48
  [ "$DRY" = 1 ] && { echo "+ modes $1: int3 lever literals [$(levlits $2 | tr ',' ' ')]; ARGATHER note up to $AR rows; the env file's words in the container's env"; echo "+ modes $1: indexer note (mxfp4, latent bf16, compressor tail bound, router selection+sort separate, AR_PCN separate at 0 layers), ARGATHER note up to $AR, no shared-expert-lane note, staging line, rule-w in every recorded pass (no plain rule; under XBIL_TRACE=1 no pass is recorded: rule-w in the container env + the tracer and passes-off lines instead), the lever K1 note (kvfold / ofold at > 0 layers, ones at > 0 MoE layers, ds41_kvfold=1 ds41_ofold=1 ds41_onefill=1); VRAM env [${VRAMENV:-none}] in the container's env, the mover's shadow / kept-free counts on 4 ranks$(case " $VRAMENV " in *" RADIANCE_PLACE_TURNOVER=0 "*) echo " (each '0 more kept free')" ;; esac), expert prefetch state"; return 0; }
  local MLF SL AL XL KL IL NW NR ok=1 lit LL
  vrammodes $1 || ok=0
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  AL=$(docker logs $CN 2>&1 | grep -m1 'the MoE gather and its all-reduce as one kernel' | sed 's/^radiance: //' | cut -c1-300)
  XL=$(docker logs $CN 2>&1 | grep -m1 'the shared expert on the second lane' | sed 's/^radiance: //' | cut -c1-300)
  KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-500)
  IL=$(grep -iE 'interleave' $O/start-$1.log 2>/dev/null | sed 's/^[A-Z] *//' | tr -s ' ' | head -3 | tr '\n' '|' | cut -c1-300)
  log "RESULT xbil $1 modes: $(echo "$MLF" | cut -c1-900) | staging: ${SL:-none} | argather: ${AL:-none} | K1: ${KL:-none} | interleave: ${IL:-no line} | VRAM: ${VKNOBS:-none}"
  log "RESULT xbil $1 prefix resume fix (R1, ds41-resfix cee0ee3): $(docker logs $CN 2>&1 | grep -c "later publisher's bytes win") startup line(s) naming it (information); hits that took a later publisher's bytes so far: $(docker logs $CN 2>&1 | grep -c 'take a later') line(s)"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate" "post-collapse-norm separate at 0 layers" "next collapse-norm separate at 0 layers"; do
    case "$MLF" in *"$w"*) ;; *) log "RESULT xbil $1: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  if [ $AR = 48 ]; then
    case "$AL" in *"up to 48 tokens a step (RADIANCE_DS41_ARGATHER, RADIANCE_DS41_ARGATHER_MAX"*) ;; *) log "RESULT xbil $1: VIRHE the ARGATHER note is not the 48-row one (RADIANCE_DS41_ARGATHER_MAX=48): '${AL}'"; ok=0 ;; esac
  else
    case "$AL" in *"up to 4 tokens a step (RADIANCE_DS41_ARGATHER)"*) ;; *) log "RESULT xbil $1: VIRHE the ARGATHER note is not today's 4-row one: '${AL}'"; ok=0 ;; esac
  fi
  [ -z "$XL" ] || { log "RESULT xbil $1: VIRHE a shared-expert-lane note (RADIANCE_DS41_SHEXP_LANE should be unset): '$XL'"; ok=0; }
  [ -n "$SL" ] || { log "RESULT xbil $1: VIRHE no prefill staging line (levels refused?)"; ok=0; }
  case "$KL" in
    *"in one launch at 0 of"*|*"rope+round trip at 0 ("*|*"not declared at"*|*"at 0 MoE layers"*|"")
      log "RESULT xbil $1: VIRHE the lever K1 note does not show all three folds declared: '${KL}'"; ok=0 ;;
  esac
  case "$KL" in *"ds41_kvfold=1"*"ds41_ofold=1"*"ds41_onefill=1"*) ;; *) log "RESULT xbil $1: VIRHE the K1 switches do not start at 1: '${KL}'"; ok=0 ;; esac
  # rule-w from the start (RADIANCE_AQL_ACQ=rule-w): every recorded pass says so; plain `rule` must not appear
  NW=$(docker logs $CN 2>&1 | grep -c 'the acquire rule plays .*RADIANCE_AQL_ACQ=rule-w)'); NR=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_AQL_ACQ=rule)')
  log "RESULT xbil $1 acquire rule: $NW recorded-pass line(s) with RADIANCE_AQL_ACQ=rule-w, $NR with rule"
  if [ $TRACE = 1 ]; then   # the tracer records no pass (ctx.cpp: tape_on_ needs no bi_rowhash op, as !profiling_ for --profile-ops)
    local TL RO
    TL=$(docker logs $CN 2>&1 | grep -c "lever BI's tracer ON"); RO=$(docker logs $CN 2>&1 | grep -c 'recorded passes: off, the program traces rows')
    if docker logs $CN 2>&1 | grep -q "lever BI's tracer cannot open"; then
      log "RESULT xbil $1: VIRHE the tracer could not open its files ($(docker logs $CN 2>&1 | grep -m1 "tracer cannot open" | cut -c1-200)) -- it is off"; ok=0
    fi
    if docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null | grep -qx 'RADIANCE_AQL_ACQ=rule-w' && [ "$NR" = 0 ] \
       && [ "$TL" -gt 0 ] && [ "$RO" -gt 0 ]; then
      log "RESULT xbil $1: rule-w replaced by the tracer (not a speed): RADIANCE_DS41_BITRACE records no pass (core/runtime/ctx.cpp tape_on_ needs no bi_rowhash op), RADIANCE_AQL_ACQ=rule-w is in the container env, no plain rule line, the tracer and 'recorded passes: off' lines present ($TL / $RO)"
    else log "RESULT xbil $1: VIRHE under the tracer: RADIANCE_AQL_ACQ=rule-w not in the container env, a plain rule line ($NR), or no tracer / passes-off line ($TL / $RO)"; ok=0; fi
  else
    [ "$NW" -gt 0 ] && [ "$NR" = 0 ] || { log "RESULT xbil $1: VIRHE rule-w is not what the recorded passes play"; ok=0; }
  fi
  # the int3 levers' declare lines (levlits): each literal in the start log, else VIRHE
  LL=$(levlits $2); IFS='|' read -ra lit <<< "$LL"
  local found="" miss=""
  for l in "${lit[@]}"; do [ -n "$l" ] || continue
    if docker logs $CN 2>&1 | grep -qF -- "${l//,/ }"; then found="$found ${l%%,*}"; else miss="$miss | ${l//,/ }"; fi; done
  log "RESULT xbil $1 int3 lever lines: ${#lit[@]} literal(s), found [$(echo $found)]$([ -n "$miss" ] && echo " -- VIRHE missing:$miss") | $(docker logs $CN 2>&1 | grep -E "lever GPF|GPF-b|lever W1|context skip \(|confidence plane|ds41_qnfold=|ds41_swfold=|activation arena|expert prefetch: off|draft lend:" | sed 's/^radiance: //; s/^[A-Z] *//' | cut -c1-260 | sort -u | tr '\n' '|' | cut -c1-2400)"
  [ -z "$miss" ] || ok=0
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
gpftab() {   # gpftab TAG: lever GPF's table line (libr4d parses RADIANCE_DS41_GG_PF_TABLE at the first gg_pf call, a prefill row):
  # its entries = the table's, no skipped entry (logged; VIRHE when not, the item goes on: the forms are exact either way)
  [ $GGPF = 1 ] || return 0
  local n; n=$(echo "$GPTAB" | tr ',' '\n' | grep -c .)
  [ "$DRY" = 1 ] && { echo "+ gpftab $1: 'lever GPF form table from RADIANCE_DS41_GG_PF_TABLE: $n entries', no 'RADIANCE_DS41_GG_PF_TABLE entry' (skipped) line"; return 0; }
  local tl sk; tl=$(docker logs $CN 2>&1 | grep -m1 'lever GPF form table from RADIANCE_DS41_GG_PF_TABLE' | sed 's/^radiance: //'); sk=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_DS41_GG_PF_TABLE entry')
  case "$tl" in *": $n entries"*) [ "$sk" = 0 ] && { log "RESULT xbil $1 GPF table: $tl (all $n of XS2_GGPF_TABLE)"; return 0; } ;; esac
  log "RESULT xbil $1 GPF table: VIRHE '${tl:-no table line}' with $sk skipped-entry line(s); XS2_GGPF_TABLE has $n entries"
}
draftlines() {   # draftlines TAG: every drafter / speculation line of the serve log, logged
  [ "$DRY" = 1 ] && { echo "+ RESULT xbil $1 drafter lines"; return 0; }
  log "RESULT xbil $1 drafter / speculation lines: $(docker logs $CN 2>&1 | grep -iE 'dspark|drafter|speculat|draft-window|draft window|draft depth|draft hybrid|verify' | sed 's/^[A-Z][^ ]* //' | sort -u | head -10 | tr '\n' '|' | cut -c1-1600)"
}
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
  log "RESULT xbil $1 prefix cache reset: HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"
}
swnote0() {   # swnote0 TAG: NST 0: GET at start (gg_gemv 2, aql_acq 2, the K1 three 1, no drafter); ds41_kvfold 2 and draft_depth0 1 refused; N's set applied
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (expect gg_gemv 2, aql_acq 2, ds41_kvfold / ds41_ofold / ds41_onefill 1, draft_capable 0, epoch 0); POST ds41_kvfold 2 -> 400, draft_depth0 1 -> 400; POST N's set $(armset N '{"epoch_bump": 1}') + a 4-token request -> applied"; return 0; }
  local s0 s1 c2 c3
  s0=$(fn4get); c2=$(fn4code '{"ds41_kvfold": 2}'); c3=$(fn4code '{"draft_depth0": 1}')
  fn4set "$(armset N '{"epoch_bump": 1}')"; nudge; s1=$(fn4get)
  log "RESULT xbil $1 switch note: at start $s0 | ds41_kvfold 2 -> HTTP $c2, draft_depth0 1 -> HTTP $c3 | after N's set: $s1 | core log: $(docker logs $CN 2>&1 | grep -E 'fn4: switches' | tail -2 | sed 's/^[A-Z] *//' | tr '\n' '|' | cut -c1-400)"
  python3 -c 'import json,sys
s0, s1 = json.loads(sys.argv[1]), json.loads(sys.argv[2]); n = json.loads(sys.argv[3]); w = s0["switches"]
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) \
     and s0["draft_capable"] == 0 and s0["epoch"] == 0 and all(w.get(k, -1) != -1 for k in n)
ok = ok and all(s1["switches"].get(k) == v for k, v in n.items()) and not s1["pending"] and s1["epoch"] >= 1
sys.exit(0 if ok else 1)' "$s0" "$s1" "$(armset N)" 2>/dev/null && [ "$c2" = 400 ] && [ "$c3" = 400 ]
}
swnote() {   # swnote TAG: GET at start; ds41_kvfold 2 / draft_hybrid 2 refused; A's set (and B's in both mode) applied at a step
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (expect gg_gemv 2, aql_acq 2, ds41_kvfold / ds41_ofold / ds41_onefill 1, draft_adaptive 1, draft_depth0 0, draft_hybrid 0, draft_capable 1, epoch 0; every pinned switch served, not -1); POST ds41_kvfold 2 -> 400, draft_hybrid 2 -> 400; POST A's set $(armset "$ARM_A")$([ $ARMS = both ] && echo " then B's $(armset "$ARM_B")"), each + a 4-token request -> applied"; return 0; }
  local s0 s1 s2 c2 c3
  s0=$(fn4get); c2=$(fn4code '{"ds41_kvfold": 2}'); c3=$(fn4code '{"draft_hybrid": 2}')
  fn4set "$(armset "$ARM_A")"; nudge; s1=$(fn4get)
  if [ $ARMS = both ]; then fn4set "$(armset "$ARM_B")"; nudge; s2=$(fn4get); else s2=$s1; fi
  log "RESULT xbil $1 switch note: at start $s0 | ds41_kvfold 2 -> HTTP $c2, draft_hybrid 2 -> HTTP $c3 | after A's set: $s1$([ $ARMS = both ] && echo " | after B's set: $s2") | core log: $(docker logs $CN 2>&1 | grep -E 'fn4: switches' | tail -2 | sed 's/^[A-Z] *//' | tr '\n' '|' | cut -c1-400)"
  python3 -c 'import json,sys
s0, s1, s2 = (json.loads(x) for x in sys.argv[1:4]); a, b = json.loads(sys.argv[4]), json.loads(sys.argv[5]); both = sys.argv[6] == "both"
w = s0["switches"]
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) \
     and w["draft_adaptive"] == 1 and w["draft_depth0"] == 0 and w["draft_hybrid"] == 0 and s0["draft_capable"] == 1 and s0["epoch"] == 0
ok = ok and all(w.get(k, -1) != -1 for k in a)   # every pinned switch served at start (stack2: the int3 levers too)
ok = ok and all(s1["switches"].get(k) == v for k, v in a.items()) and not s1["pending"] and s1["epoch"] >= 1
if both: ok = ok and all(s2["switches"].get(k) == v for k, v in b.items()) and not s2["pending"] and s2["epoch"] > s1["epoch"]
sys.exit(0 if ok else 1)' "$s0" "$s1" "$s2" "$(armset "$ARM_A")" "$(armset "$ARM_B")" $ARMS 2>/dev/null && [ "$c2" = 400 ] && [ "$c3" = 400 ]
}
# ---------------------------------------------------------------- this item's own functions
trap 'down' EXIT
trap 'exit 143' TERM INT
gate1() {   # gate1 TAG LIMIT: lib-pr266's restart + ONE check (warm-up + fastcheck); over the limit is logged, not retried
  # (identity does not depend on speed; a slow state only stretches the load's timing, which the probe lines record)
  local tag=$1 lim=$2 ms st=fast
  restart $tag || { keeplog $tag-fail; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && return 0
  awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}' && { st="SLOW (not retried)"; keeplog $tag-slow; }
  echo "$st $ms" > $O/guard-$tag.txt
  log "RESULT xbil guard $tag: $st, 1-user ${ms} ms/step (limit $lim; hxzzy-stack2 st-nst0 13.97, st-stack 33.6)"
}
SWV=""   # the switch value of the phase running (XBIL_SWVALS), "" = none
pin() {   # pin N|A [EXTRA]: the start's in-process pin (+ XBIL_SW at the phase's value)
  local x=${2:-}; [ -n "$x" ] || x='{}'
  [ -n "$SWV" ] && x=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d[sys.argv[2]]=int(sys.argv[3]); print(json.dumps(d))' "$x" "$SW" "$SWV")
  if [ "$1" = N ]; then armset N "$x"; else armset "$ARM_A" "$x"; fi
}
alone() {   # alone TAG ARM NAME: pin + epoch_bump, a prefix-cache reset, greedyset.py's 20 prompts at c1 ALONE -> $O/TAG-NAME.jsonl
  local tag=$1 t0=$(date +%s)
  fn4set "$(pin $2 '{"epoch_bump": 1}')"
  pcreset $tag-$3
  run "(cd $D && timeout 900 python3 $GS $URL $O/$tag-$3.jsonl --conc 1 --max-tokens $MT > $O/$tag-$3.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  log "RESULT xbil $tag $3 (greedy set c1 ALONE, temperature 0, $MT tok): $(tail -1 $O/$tag-$3.txt | cut -c1-300), $(( $(date +%s) - t0 )) s | ran under: $(fn4get | cut -c1-700)"
  swcheck $tag $3
}
swcheck() {   # swcheck TAG NAME: under XBIL_SW, the switch applied is the phase's value (else VIRHE: the phase did not run it)
  [ -n "$SWV" ] || return 0
  [ "$DRY" = 1 ] && { echo "+ check GET /admin/fn4: $SW = $SWV"; return 0; }
  local g; g=$(fn4get)
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); sys.exit(0 if d["switches"].get(sys.argv[2]) == int(sys.argv[3]) else 1)' "$g" "$SW" "$SWV" 2>/dev/null \
    || log "RESULT xbil $1 $2: VIRHE $SW is not $SWV after the phase's pin ($(echo "$g" | cut -c1-400)) -- this phase did not run it"
}
loaded() {   # loaded TAG ARM [NAME]: pin + epoch_bump, a prefix-cache reset, loadid.py run (mb c4 + per probe one fresh prefill) -> $O/TAG-NAME.jsonl
  local tag=$1 t0=$(date +%s) j nm=${3:-ld}
  fn4set "$(pin $2 '{"epoch_bump": 1}')"
  pcreset $tag-$nm
  run "(cd $D && timeout 2700 python3 -I $LD run --url $URL --gs $GS --mb $MBP --out $O/$tag-$nm.jsonl --conc $CONC --warm $WARM --offset $OFF --pf-tok $PFT --max-tokens $MT > $O/$tag-$nm.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  j=$(grep -m1 '^LOADID ' $O/$tag-$nm.txt | cut -c8-)
  [ -n "$j" ] || { log "RESULT xbil $tag $nm: VIRHE no LOADID line ($(tail -3 $O/$tag-$nm.txt | tr '\n' ' ' | cut -c1-400))"; return 1; }
  log "RESULT xbil $tag $nm (the 20 prompts each under load: mb c4 + one fresh ~${PFT}-token prefill at +${OFF} s, temperature 0, $MT tok): $(python3 -c 'import json,sys
d = json.loads(sys.argv[1]); w = d.get("window") or {}
print("%s of %s answered, probe %s s mean | prefills %s tok mean, %s of %s cold, TTFT %s s mean (%s..%s) | in flight %s..%s %% (median) of a probe | mixed steps a probe min %s, median %s, total %s; probes without one %s | c4 %s requests done, %s errors | window %s s: %s ms/decode step, %s steps, %s prefill tok, %s decode tok%s" % (
    d["answered"], d["of"], d["probe_s_mean"], d["pf_prompt_mean"], d["pf_cold"], d["of"], d["pf_ttft_mean"], d["pf_ttft_min"], d["pf_ttft_max"],
    round(100 * (d["ov_frac_min"] or 0)), round(100 * (d["ov_frac_median"] or 0)), d["mixed_min"], d["mixed_median"], d["mixed_total"],
    d["probes_without_mixed"] or "none", d["c4_requests_done"], d["c4_errors"], d["window_s"], w.get("ms_decode_step"), w.get("steps"),
    w.get("prefill_tok"), w.get("dec_tok"), (" | VIRHE errors: %s" % d["errors"]) if d.get("errors") else ""))' "$j" 2>&1 | cut -c1-1200), $(( $(date +%s) - t0 )) s"
}
cmpx() {   # cmpx TAG NAME A B: loadid.py cmp (byte for byte + the first differing char / token via /tokenize) -> $O/TAG-cmp-NAME.json, logged
  local r
  [ "$DRY" = 1 ] && { echo "+ python3 -I $LD cmp --url $URL $3 $4 --json $O/$1-cmp-$2.json"; return 0; }
  if [ ! -s "$3" ] || [ ! -s "$4" ]; then
    log "RESULT xbil $1 $2: not compared (missing:$([ -s "$3" ] || echo " $3")$([ -s "$4" ] || echo " $4"))"; echo "$1 $2 not compared" >> $O/identity.txt; return 0
  fi
  r=$(timeout 300 python3 -I $LD cmp --url $URL --json $O/$1-cmp-$2.json --label "$1 $2" "$3" "$4" 2>&1 | grep -m1 '^CMPLINE ' | cut -c9-)
  echo "$1 $2 ${r:-compare failed}" >> $O/identity.txt
  log "RESULT xbil $1 $2: ${r:-VIRHE compare failed} ($(basename $3) vs $(basename $4))"
}
c4rr() {   # c4rr TAG ARM NAME: pin + epoch_bump, a prefix-cache reset, greedyset.py's 20 prompts at conc 4 -> $O/TAG-NAME.jsonl
  local tag=$1 t0=$(date +%s)
  fn4set "$(pin $2 '{"epoch_bump": 1}')"
  pcreset $tag-$3
  run "(cd $D && timeout 900 python3 $GS $URL $O/$tag-$3.jsonl --conc 4 --max-tokens $MT > $O/$tag-$3.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  log "RESULT xbil $tag $3 (greedy set at conc 4, temperature 0, $MT tok): $(tail -1 $O/$tag-$3.txt | cut -c1-300), $(( $(date +%s) - t0 )) s"
}
smoke() {   # smoke TAG ARM: under XBIL_SW, the LAST switch value first on a short mixed load (the greedy set at conc 4: prompts
  # prefilled beside other prompts' decode) -> 0 when every answer came back and the server is healthy, else 1 (start stopped)
  [ -n "$SWVALS" ] || return 0
  local v=${SWVALS##* } t0=$(date +%s) n
  SWV=$v
  fn4set "$(pin $2 '{"epoch_bump": 1}')"
  run "(cd $D && timeout 600 python3 $GS $URL $O/$1-smoke.jsonl --conc 4 --max-tokens 32 > $O/$1-smoke.txt 2>&1)"
  SWV=""
  [ "$DRY" = 1 ] && { echo "+ smoke $1: $SW=$v, greedy set at conc 4 (32 tok): every answer back, /health 200"; return 0; }
  n=$(python3 -c 'import json,sys; print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l).get("text") is not None))' $O/$1-smoke.jsonl 2>/dev/null || echo 0)
  if [ "$n" = 20 ] && curl -s -m 10 -o /dev/null -w '%{http_code}' $URL/health | grep -q 200; then
    log "RESULT xbil $1 smoke ($SW=$v, greedy set at conc 4, 32 tok, mixed steps): 20 of 20 answered, healthy, $(( $(date +%s) - t0 )) s | $(docker logs $CN 2>&1 | grep -m1 'lever BI' | sed 's/^radiance: //' | cut -c1-300)"
    return 0
  fi
  log "RESULT xbil $1 smoke ($SW=$v): VIRHE $n of 20 answered or unhealthy ($(tail -2 $O/$1-smoke.txt | tr '\n' ' ' | cut -c1-300)) | $(docker logs $CN 2>&1 | grep -E '^E |step fail|refus' | tail -3 | tr '\n' ' ' | cut -c1-500)"
  keeplog $1-smoke
  return 1
}
mbblocks() {   # mbblocks TAG ARM: the speed A/B in process (XBIL_MB phases, decode phases before pf), blocks in XBIL_MBORDER
  [ -n "$MB" ] || return 0
  local k=0 v
  for v in $MBORDER; do
    k=$((k + 1))
    if [ "$v" = _ ]; then SWV=""; else SWV=$v; fi
    fn4set "$(pin $2 '{"epoch_bump": 1}')"
    nudge
    swcheck $1 mb-b$k
    stopfile || run "(cd $D && timeout 2400 python3 $MBP --tag xbil-$1-b$k-s$v --phases $MB --dur $MBDUR --warm $MBWARM > $O/$1-mb-b$k-s$v.txt 2>&1)"
    [ "$DRY" = 1 ] || log "RESULT xbil $1 mb block $k ($SW=$v): $(grep '^MB ' $O/$1-mb-b$k-s$v.txt 2>/dev/null | cut -c4- | tr '\n' ' ' | cut -c1-1800)"
  done
  SWV=""
  mbsum $1
}
mbsum() {   # mbsum TAG: per mb phase and switch value the mean of its blocks, B vs the first value in %
  [ "$DRY" = 1 ] && { echo "+ RESULT xbil $1 mb summary: per phase and value (c: agg dec tok/s, ms a decode step; pf: prefill tok/s, decode kept, ms a step during), vs the first value in %"; return 0; }
  log "RESULT xbil $1 mb summary ($SW, blocks [$MBORDER], dur $MBDUR s): $(python3 - $O $1 <<'PY' 2>&1 | head -c 2500
import glob, json, os, re, statistics, sys
o, tag = sys.argv[1], sys.argv[2]
by = {}
for f in sorted(glob.glob(os.path.join(o, tag + "-mb-b*-s*.txt"))):
    v = re.search(r"-s([^-.]+)\.txt$", f).group(1)
    for l in open(f):
        if l.startswith("MB "):
            d = json.loads(l[3:]); by.setdefault(d["phase"], {}).setdefault(v, []).append(d)
keys = {"pf": ("prefill_tps", "decode_kept", "ms_step_during")}
out = []
for ph in sorted(by, key=lambda p: (p == "pf", p)):
    vals = by[ph]; vs = sorted(vals); base = vs[0]
    ks = keys.get(ph, ("agg_dec_tps", "ms_decode_step", "per_stream_dec_p50"))
    parts = []
    for k in ks:
        m = {v: statistics.mean(x[k] for x in vals[v] if x.get(k) is not None) for v in vs if any(x.get(k) is not None for x in vals[v])}
        if base in m and m[base]:
            parts.append("%s %s" % (k, ", ".join("%s=%.2f%s" % (v, m[v], "" if v == base else " (%+.2f %%)" % (100 * (m[v] / m[base] - 1))) for v in m)))
    out.append("%s [%s blocks] %s" % (ph, "/".join(str(len(vals[v])) for v in vs), "; ".join(parts)))
print(" | ".join(out) or "no MB lines")
PY
)"
}
btcheck() {   # btcheck TAG START: under XBIL_TRACE, after the warm-up and ONE more request: every rank's .bin has records and its
  # .names file is there (deepseek41 writes and flushes them every step: bt_flush fwrite + fflush, no shutdown needed); else 1
  [ $TRACE = 1 ] || return 0
  local d=$STATSDIR/bt-$2 r miss="" env mnt
  [ "$DRY" = 1 ] && { echo "+ btcheck $1: a 4-token request, then $d/bt-r{0..3}-*.bin non-empty and bt-r{0..3}-*.names present (written and flushed each step)"; return 0; }
  nudge
  for r in 0 1 2 3; do
    ls $d/bt-r$r-*.bin > /dev/null 2>&1 && [ -s "$(ls $d/bt-r$r-*.bin 2>/dev/null | head -1)" ] || miss="$miss r$r.bin"
    ls $d/bt-r$r-*.names > /dev/null 2>&1 || miss="$miss r$r.names"
  done
  if [ -z "$miss" ]; then
    log "RESULT xbil $1 tracer files: every rank writing ($(ls -la $d 2>/dev/null | awk 'NR > 1 && /bt-r/ {n++; b += $5} END {printf "%d files, %.1f MB", n, b / 1e6}') after the warm-up + 1 request)"
    return 0
  fi
  env=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null | grep '^RADIANCE_DS41_BITRACE')
  mnt=$(docker inspect -f '{{range .Mounts}}{{.Source}}->{{.Destination}} {{end}}' $CN 2>/dev/null)
  log "RESULT xbil $1: VIRHE tracer wrote no records (missing:$miss) -- container path /stats/bt-$2, env [$(echo $env | tr '\n' ' ')], mounts [$mnt] (want $STATSDIR->/stats), engine: $(docker logs $CN 2>&1 | grep -E "tracer (ON|cannot open)" | head -2 | sed 's/^radiance: //' | tr '\n' '|' | cut -c1-400)"
  return 1
}
btanalyze() {   # btanalyze TAG START: lever BI's tracer files of this start -> bitrace.py analyze (per probe the first diverging op)
  [ $TRACE = 1 ] || return 0
  [ "$DRY" = 1 ] && { echo "+ python3 -I $BT analyze $STATSDIR/bt-$2 --json $O/$1-bitrace.json > $O/$1-bitrace.txt"; return 0; }
  log "RESULT xbil $1 tracer files: $(ls -la $STATSDIR/bt-$2 2>/dev/null | awk 'NR > 1 {n++; b += $5} END {printf "%d files, %.1f MB", n, b / 1e6}')"
  timeout 1200 python3 -I $BT analyze $STATSDIR/bt-$2 --json $O/$1-bitrace.json > $O/$1-bitrace.txt 2>&1
  local l; while IFS= read -r l; do log "RESULT xbil $1 $(echo "$l" | cut -c1-1500)"; done < <(grep -E '^BITRACE' $O/$1-bitrace.txt)
  grep -q '^BITRACE summary' $O/$1-bitrace.txt || log "RESULT xbil $1 bitrace: VIRHE no summary ($(tail -3 $O/$1-bitrace.txt | tr '\n' ' ' | cut -c1-500))"
  # does the tracer still reproduce the bug? the traced loaded-vs-alone count beside the untraced loadid's (XBIL_REFDIR, iej-z0ad)
  local tc rc
  tc=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("0/%d" % d["n"] if d["identical"] else "%d/%d" % (d["n_diff"], d["n"]))' $O/$1-cmp-ld-vs-a1.json 2>/dev/null || echo "-")
  rc=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("0/%d" % d["n"] if d["identical"] else "%d/%d" % (d["n_diff"], d["n"]))' $REFDIR/$1-cmp-ld-vs-a1.json 2>/dev/null || echo "-")
  log "RESULT xbil $1 tracer vs untraced: loaded vs alone $tc differ under the tracer, $rc in the untraced run ($REFDIR); alone under the tracer vs bz d0: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("IDENTICAL" if d["identical"] else "%d of %d differ" % (d["n_diff"], d["n"]))' $O/$1-cmp-a1-vs-d0.json 2>/dev/null || echo "-") (eager vs recorded passes: the same arithmetic when IDENTICAL)"
}
startsum() {   # startsum TAG DESC: the start's summary line: every compare of the start (byte for byte), the load
  [ "$DRY" = 1 ] && { echo "+ RESULT xbil $1 summary: each compare (n differ of 20, first diff char / token each), the load"; return 0; }
  python3 - "$O" "$1" > $O/$1-summary.txt 2>&1 <<'PY'
import glob, json, os, sys
o, tag = sys.argv[1], sys.argv[2]
def js(f):
    try: return json.load(open(f))
    except Exception: return None
parts = []
for f in sorted(glob.glob(os.path.join(o, tag + "-cmp-*.json"))):
    n, d = f[len(os.path.join(o, tag + "-cmp-")):-5], js(f)
    if d is None: parts.append("%s: not compared" % n); continue
    if d["identical"]: parts.append("%s: IDENTICAL (%d of %d)" % (n, d["n"], d["n"])); continue
    parts.append("%s: %d of %d differ [%s]" % (n, d["n_diff"], d["n"], ", ".join("p%d tok %s%s" % (e["i"], "-" if e["tok"] is None else e["tok"],
                 "" if e.get("mixed_b") is None else " (mixed %s)" % e["mixed_b"]) for e in d["diff"][:12])))
for f in sorted(glob.glob(os.path.join(o, tag + "-ld*.txt"))):
    try: ld = json.loads(next(l for l in open(f) if l.startswith("LOADID "))[7:])
    except Exception: continue
    parts.append("load %s: c4 %s requests (%s errors), %s of %s prefills cold ~%s tok TTFT %s s, in flight median %s %%, mixed steps a probe min %s / median %s"
                 % (os.path.basename(f)[len(tag) + 1:-4], ld["c4_requests_done"], ld["c4_errors"], ld["pf_cold"], ld["of"], ld["pf_prompt_mean"],
                    ld["pf_ttft_mean"], round(100 * (ld["ov_frac_median"] or 0)), ld["mixed_min"], ld["mixed_median"]))
print(" | ".join(parts))
PY
  log "RESULT xbil $1 summary ($2): $(head -c 3000 $O/$1-summary.txt)"
}
overall() {   # overall: one line over the starts that ran: loaded vs alone and c4 run-to-run per switch value, the verdict
  [ "$DRY" = 1 ] && { echo "+ RESULT xbil summary: per start and switch value loaded vs alone, c4 run-to-run (n differ of 20), alone across values, control, verdict"; return 0; }
  log "RESULT xbil summary: $(python3 - $O "${SWVALS##* }" <<'PY' 2>&1 | head -c 3000
import glob, json, os, sys
o, last = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
def js(f):
    try: return json.load(open(f))
    except Exception: return None
f = lambda d: "-" if d is None else ("0/%d" % d["n"] if d["identical"] else "%d/%d" % (d["n_diff"], d["n"]))
# the verdict reads the last switch value's compares (the fix arm under XBIL_SW) and the alone line across values
parts, bad, ok = [], [], 0
for tag, lab in (("li-nst0", "NST 0"), ("li-stack", "NST 5")):
    cs = {os.path.basename(p)[len(tag) + 5:-5]: js(p) for p in glob.glob(os.path.join(o, tag + "-cmp-*.json"))}
    if not cs: continue
    keys = sorted(cs)
    parts.append("%s: %s" % (lab, ", ".join("%s %s" % (k, f(cs[k])) for k in keys)))
    for k in keys:
        meas = (k.startswith("ld-vs-a1") or k.startswith("c4b-vs-c4a")) and (not last or k.endswith("-s" + last))
        if last and k.startswith("a1-s") and "-vs-a1-s" in k: meas = True   # the alone line must not move
        if meas and cs[k] is not None:
            if cs[k]["identical"]: ok += 1
            else: bad.append("%s %s %s" % (lab, k, f(cs[k])))
v = ("nothing measured" if not parts else ("IDENTICAL UNDER LOAD AND RUN TO RUN%s, alone line unchanged, at every verdict compare (%d)" % (" under the last switch value " + last if last else "", ok)) if not bad
     else "ANSWERS CHANGE: " + "; ".join(bad))
print(" || ".join(parts + ["verdict: " + v]))
PY
)"
}
dstart() {   # dstart n0|n5: one start of the stack2 config: checks, a1, ld, a2, compares, summary
  local s=$1 tag lim arm desc ok
  stackenv $s; export MODEL=$M225; HP29=19456; best_vars
  case $s in
    n0) tag=li-nst0; lim=30; arm=N; desc=$DESC_N; export MNBT=4160 NST=0 XARGS="$XARGS --live" ;;
    n5) tag=li-stack; lim=70; arm=A; desc=$DESC_A; export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive" ;;
  esac
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV > $XS2_ENVFILE"; else printf '%s\n' $LEVENV > $XS2_ENVFILE; fi
  log "RESULT xbil $tag: start ($desc) IMAGE=$IMAGE MODEL=$MODEL $(desc) pin $(pin $arm) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
  # the tracer's directory BEFORE the start: deepseek41 opens its files at the first step (the warm-up) and turns the tracer off
  # when it cannot (bi-b-trace2, 09:13: the directory came after the start, "tracer cannot open ...: off")
  if [ $TRACE = 1 ]; then
    if [ "$DRY" = 1 ]; then echo "+ mkdir -p $STATSDIR/bt-$s && chmod 777 $STATSDIR/bt-$s (before the start)"
    else mkdir -p $STATSDIR/bt-$s && chmod 777 $STATSDIR/bt-$s; fi
  fi
  gate1 $tag $lim || { log "RESULT xbil $tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  modes $tag $s || { log "RESULT xbil $tag: VIRHE the modes are not stack2's -> start stopped"; down; return 1; }
  draftlines $tag
  if [ "$DRY" = 1 ]; then
    [ $s = n0 ] && echo "+ no DSpark drafter line at NST 0" || echo "+ DSpark drafter + adaptive depth lines"
  else
    ok=1
    if [ $s = n5 ]; then
      docker logs $CN 2>&1 | grep -q "DSpark drafter" || { log "RESULT xbil $tag: VIRHE no DSpark drafter line"; ok=0; }
      docker logs $CN 2>&1 | grep -q "draft depth: adaptive" || { log "RESULT xbil $tag: VIRHE no adaptive draft depth line"; ok=0; }
    else
      docker logs $CN 2>&1 | grep -qiE "DSpark drafter|'dspark' proposes" && { log "RESULT xbil $tag: VIRHE a DSpark drafter line at NST 0"; ok=0; }
    fi
    [ $ok = 1 ] || { keeplog $tag-nodraft; down; return 1; }
  fi
  if [ $s = n0 ]; then swnote0 $tag; else swnote $tag; fi \
    || { log "RESULT xbil $tag: VIRHE the switches are not as expected -> start stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  btcheck $tag $s || { log "RESULT xbil $tag: VIRHE the tracer wrote no records -> start stopped"; keeplog $tag-notrace; down; return 1; }
  smoke $tag $arm || { log "RESULT xbil $tag: VIRHE the smoke under $SW failed -> start stopped"; down; return 1; }
  local vals="${SWVALS:-_}" v sfx first=""
  for v in $vals; do
    if [ "$v" = _ ]; then SWV=""; sfx=""; else SWV=$v; sfx="-s$v"; fi
    [ -n "$first" ] || first=$sfx
    [ -n "$SWV" ] && log "RESULT xbil $tag phase $SW=$SWV: pin $(pin $arm)"
    stopfile || alone $tag $arm a1$sfx
    stopfile || loaded $tag $arm ld$sfx
    [ $C4RR = 1 ] && { stopfile || c4rr $tag $arm c4a$sfx; stopfile || c4rr $tag $arm c4b$sfx; }
  done
  SWV=${SWVALS%% *}
  [ $CTRL = 1 ] && ! stopfile && alone $tag $arm a2
  SWV=""
  gpftab $tag
  for v in $vals; do
    [ "$v" = _ ] && sfx="" || sfx="-s$v"
    cmpx $tag ld-vs-a1$sfx $O/$tag-a1$sfx.jsonl $O/$tag-ld$sfx.jsonl
    [ $C4RR = 1 ] && { cmpx $tag c4b-vs-c4a$sfx $O/$tag-c4a$sfx.jsonl $O/$tag-c4b$sfx.jsonl; cmpx $tag c4a-vs-a1$sfx $O/$tag-a1$sfx.jsonl $O/$tag-c4a$sfx.jsonl; }
    [ "$sfx" != "$first" ] && cmpx $tag a1$sfx-vs-a1$first $O/$tag-a1$first.jsonl $O/$tag-a1$sfx.jsonl
    cmpx $tag a1$sfx-vs-d0 $RD/dsd-d0-gs-c1.jsonl $O/$tag-a1$sfx.jsonl   # each arm's alone line vs today's (bz d0)
  done
  cmpx $tag ld-vs-d0 $RD/dsd-d0-gs-c1.jsonl $O/$tag-ld$first.jsonl
  [ $CTRL = 1 ] && cmpx $tag a2-vs-a1 $O/$tag-a1$first.jsonl $O/$tag-a2.jsonl
  mbblocks $tag $arm
  startsum $tag "$desc"
  btanalyze $tag $s
  run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
  down
  log "RESULT xbil $tag: done"
}
# ---------------------------------------------------------------- main
log "RESULT xbil $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) starts [$STARTS] | extra env [${EXTRAENV:-none}] | pin extra ${PINX:-none} | switch ${SW:-none} values [${SWVALS:-}] | c4 run-to-run $C4RR | tracer $TRACE (pmax $PMAX) | n0 pin $FIX0 | n5 pin $(pin A) | env n0 [$(levenv n0)] | env n5 [$(levenv n5)] | VRAM env [${VRAMENV:-none}] | per start: greedy set c1 alone, then each prompt under mb c4 + one fresh ~${PFT}-token prefill at +${OFF} s (warm ${WARM} s), $MT tok, temperature 0$([ $CTRL = 1 ] && echo ", control alone again"); compares ld vs a1, vs bz d0 ($RD/dsd-d0-gs-c1.jsonl)"
# the image, on this host, read only (in DRYRUN too: docker image inspect + a no-device, no-network grep of its files; -z as
# dsm-stack2.sh, copied with its literal list for this configuration):
IMGLITS="B|dspark_conf_rows, gg_pf, f32_pf, dspark_walk,
B|ds41_onefill, draft_depth0, draft_probe, draft_hybrid,
B|draft depth 0: a step drafts nothing
B|draft hybrid: a sequence whose prompt lookup
B|reset_prefix_cache
B|bytes win where no snapshot rests on the first copy
A|RADIANCE_DS41_KVFOLD
K|rmsnorm_rope_roundtrip_kv_store
K|r4d_gg_mr_pin"
[ "$DMX" = 1 ] && IMGLITS="$IMGLITS
K|RADIANCE_DS41_DRAFTER_MX"
[ $GGPF = 1 ] && IMGLITS="$IMGLITS
K|RADIANCE_DS41_GG_PF_TABLE
K|lever GPF form table from RADIANCE_DS41_GG_PF_TABLE"
[ $F32PF != 0 ] && IMGLITS="$IMGLITS
K|RADIANCE_DS41_F32_PF"
[ $WALK != 0 ] && IMGLITS="$IMGLITS
A|RADIANCE_DS41_DSPARK_WALK
K|dspark_walk_step"
[ $SMALL5 = 1 ] && IMGLITS="$IMGLITS
A|RADIANCE_DS41_ARGATHER_MAX
A|RADIANCE_DS41_QNFOLD
A|RADIANCE_DS41_SWFOLD
A|RADIANCE_DS41_DSPARK_CONF
B|RADIANCE_DRAFT_CTXSKIP
B|a confidence plane (the dspark_conf gate"
[ $LEND = 1 ] && IMGLITS="$IMGLITS
X|RADIANCE_DRAFT_LEND
X|RADIANCE_DRAFT_LEND_BORROW"
for v in $VRAMSEL; do for w in $(vf $v 2 | tr ',' ' ') $(vf $v 3 | tr ',' ' '); do IMGLITS="$IMGLITS
X|${w%%=*}"; done; done
# this item's driver needs two more: the server's /tokenize (first differing token) and the mixed-step counter
IMGLITS="$IMGLITS
B|/tokenize
B|radiance:engine_steps_mixed_total"
[ $TRACE = 1 ] && IMGLITS="$IMGLITS
K|bi_rowhash
A|RADIANCE_DS41_BITRACE
A|tracer ON (RADIANCE_DS41_BITRACE=
B|recorded passes: off, the program traces rows (bi_rowhash"
[ -n "$XLITS" ] && IMGLITS="$IMGLITS
$XLITS"
IMGLITS=$(echo "$IMGLITS" | awk '!seen[$0]++')
docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xbil: VIRHE image $IMAGE not on this host"; exit 1; }
IMGMISS=$(echo "$IMGLITS" | timeout 120 docker run --rm -i --network none --entrypoint sh $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels/libr4d.so; n=0
while IFS= read -r l; do [ -n "$l" ] || continue; n=$((n + 1)); w=${l%%|*}; s=${l#*|}
  case $w in B) f=$B ;; A) f=$A ;; K) f=$K ;; *) f="$B $A $K" ;; esac
  o=0; for x in $f; do grep -qzF -- "$s" $x && { o=1; break; }; done
  [ $o = 1 ] || echo "MISS $w|$s"; done; echo "CHECKED $n"' 2>&1)
case "$IMGMISS" in "CHECKED "*) ;; *) log "RESULT xbil: VIRHE $IMAGE is not the int3 stack image -- missing: $(echo "$IMGMISS" | grep -v '^CHECKED' | tr '\n' ';' | cut -c1-900)"; exit 1 ;; esac
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)) is the int3 stack image: all $(echo "$IMGMISS" | sed 's/CHECKED //') literals present (the 27-switch registry, depth 0 + hybrid, R1 prefix-resume fix, K1, D2, the levers switched on here) -- OK; NST 0 pin $FIX0; NST 5 pin $FIX"
# EVERY HELPER RESOLVES (DRYRUN and real): the files this body runs or sources are present, the python ones parse (ast, nothing
# written), loadid.py's selftest loads greedyset.py / mb.py read only and finds what it calls, every shell function this body
# calls is defined (`type -t`: lib-29.sh / lib-pr266.sh and this body), the tools on PATH; the sidecar matches its .sha256; the
# yaml carries the env_file entry and the stack names and docker compose accepts it. Any miss: VIRHE (a dry run fails too).
HMISS=""
for f in "$LIB29" "$LIBPR" $Y $GS $LD $BT $MBP $D/ds41_tests.py $D/fastcheck.py $T/evict-exl3.py; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $GS $LD $BT $MBP $D/ds41_tests.py $D/fastcheck.py $T/evict-exl3.py 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
ST=$(timeout 60 python3 -I $LD selftest --gs $GS --mb $MBP 2>&1 | tail -1)
case "$ST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$LD(${ST:0:300})" ;; esac
BST=$(timeout 60 python3 -I $BT selftest 2>&1 | tail -1)
case "$BST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$BT(${BST:0:300})" ;; esac
FNS="log run mkd restart check down evict stopfile best_vars desc serve_args keeplog vf levenv levlits armset stackenv identw vrammodes
     modes gpftab draftlines fn4get fn4set nudge fn4code pcreset swnote0 swnote gate1 pin alone swcheck smoke loaded c4rr mbblocks mbsum btcheck btanalyze cmpx startsum overall dstart"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee sha256sum; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
if [ $SMALL5 = 1 ]; then
  if [ -s $MODELDIR/$SIDECAR ] && [ -s $MODELDIR/$SIDECAR.sha256 ]; then
    [ "$(sha256sum $MODELDIR/$SIDECAR | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR(sha256 differs from .sha256)"
  else HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR(+.sha256)"; fi
fi
grep -q 'XS2_ENVFILE' $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no env_file \${XS2_ENVFILE})"
# the tracer's path (XBIL_TRACE): the yaml maps STATSDIR to /stats, and its words go through the env file (levenv)
if [ $TRACE = 1 ]; then
  grep -qE '^ *- \$\{STATSDIR:\?set STATSDIR\}:/stats *$' $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no STATSDIR:/stats mount for the tracer)"
  for s in $STARTS; do case " $(levenv $s) " in *" RADIANCE_DS41_BITRACE=/stats/bt-$s "*) ;; *) HMISS="$HMISS tracer:levenv($s) lacks RADIANCE_DS41_BITRACE=/stats/bt-$s" ;; esac; done
fi
for v in RADIANCE_DS41_GG_GEMV RADIANCE_AQL_ACQ RADIANCE_DS41_KVFOLD RADIANCE_DS41_OFOLD RADIANCE_DS41_ONEFILL; do grep -q "^ *$v:" $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(does not pass $v)"; done
for kv in $VRAMENV; do grep -q "^ *${kv%%=*}:" $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(does not pass ${kv%%=*}, XBIL_VRAMENV)"; done
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config refuses it)"
HMISS=$(echo $HMISS)
[ -z "$HMISS" ] || { log "RESULT xbil: VIRHE helpers / inputs do not resolve: $HMISS"; exit 1; }
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing, $ST, bitrace.py $BST, all $(echo $FNS | wc -w) functions resolve (type -t), tools on PATH, sidecar $SIDECAR sha256 OK, yaml $(basename $Y) (env_file \${XS2_ENVFILE}) accepted by docker compose config -- OK"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xbil: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xbil: VIRHE $M225 missing or not verified"; exit 1; }
  for f in $RD/dsd-d0-gs-c1.jsonl $WARM_REF_225; do [ -s $f ] || log "RESULT xbil: note -- reference $f missing (not compared)"; done
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $RD/dsd-d0-gs-c1.jsonl $WARM_REF_225; do [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(a radiance container running now is information here; the real run refuses to start beside one)"
fi
mkd $O $O/guard $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt
evict
for s in $STARTS; do stopfile && break; dstart $s; done
overall
unset $SWITCHES
down
log "RESULT xbil $ITEM: done -- $O"
