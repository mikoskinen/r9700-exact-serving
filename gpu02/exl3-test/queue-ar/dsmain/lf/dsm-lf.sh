#!/bin/bash
# dsm-lf.sh (11.11.2026, queue-ar/dsmain/lf, ds-main-port LF builder) -- NOT a queue item; the body of 57-p4zyzzzzz-ds-r06z-lf.sh.
# Log tag "RESULT xlf ", out runs/xdsm/lf (XLF_OUT).
# LEVER LF (coordinator 11.11.): z0occ's c8 at NST 5 depth 0 (draft_occ 3) is 173.2 vs NST 0's 182.0 (-5 %), ~100 fewer trunk units a
# rank, and runtime-lent units (DL, Z0-W) moved neither hit nor tok/s. Code reading (ds41-lf): lent slots are filled only by
# Mover::fill (balance(), every dispatch: every free lent slot past the working reserve, at once, hottest pooled unit first by the
# heat engine's placement heat -- proactive, not on misses; a swap's promotion never lands in a lent slot), and with no admission
# bar: the hottest pooled units by a ~14-step heat are the experts just missed. LF ranks that fill by a long-memory decode heat
# (decay 0.998, ~350-step half-life) instead -- placement only; + /stats per-slot-class counters (both arms) to read it.
# Image XLF_IMAGE = ds41-lf (ds41-int5 283c7a1 + lever LF + the slot-class counters; unset = int5: hostcheck graph + start log).
# ONE NST 5 START, everything in process: z0occ2's start (stack3's S-arm start: stack2's NST 5 env + RADIANCE_DS41_BI=0,
# RADIANCE_DRAFT_SAMPLE=0, --draft-coupling on, the DLW q8_0 sidecar; + RADIANCE_DRAFT_CTXSKIP_SLOTS=129,
# RADIANCE_DRAFT_LEND_WEIGHTS=0, RADIANCE_DS41_ARENA_DECODE_ROWS=8,48). Pin = z0occ2's pin (stack3's NST 5 pin + ds41_bi 2 +
# draft_sample XLF_DS + draft_lend_hyst 0 / arena_d0 0 / draft_lend_w 0) + draft_occ XLF_OCC (3) / draft_occ_k XLF_OCCK (32) in
# EVERY arm, so c4 / c8 hold depth 0 and the drafter's slab is lent (where lent slots exist).
# ARMS: A = place_lf 0 (today's fill), B = place_lf 1 (lever LF). RA (per-rank trunk-unit allotment) is not built.
#   gate1 -> modes (z0occ2's literals) -> VRAM / lend / slot classes at start -> switch note (start values; place_lf 2 /
#   draft_occ 65 / ds41_kvfold 2 refused; each arm applied) -> warm-up answers vs stock -> smoke under B (greedy set conc 4 +
#   mb c4 20 s at T 1) -> T 0 BLOCKS XLF_ORDER (ABBAABBA): mb c4 / c8 at temperature 0 (mbt.py) -> T 1 BLOCKS XLF_T1ORDER (ABBA):
#   mb c4 / c8 at the server default (temperature 1; draft_sample XLF_DS in the pin) -- lfblocks.py; per block and phase:
#   aggregate tok/s, per-stream p50, tokens a step, acceptance, ms a step, the depth-0 share, and PER RANK: hit %, misses / step,
#   trunk VRAM units, the slot classes' hits / step and misses / step by last slot, each lender's held / covered slots (sampled
#   every 2 s), fills (lf, hot) and drops (victim, reserve, recall) -> IDENTITY at T 0 under B: a1-B (greedy set c1 alone) vs
#   bz d0, ld-B (each prompt under mb c4 + one fresh ~32k prefill) vs a1-B. No 1u phases, no 136k, NO EVICT.
# SUMMARY: per temperature and phase, each arm mean +- t95; B vs A (ABBA pairs, %); per arm the per-rank misses / step and hit %,
#   the slot classes' hits a resident unit a step (trunk vs each lender: hits / step over held units) and the lenders' held share;
#   the reference z0occ NST 0 (int5's z0 image, c4 138.7 / c8 182.0, mb server default) and stack3 NST 0 (T 0: c4 138.5 / c8
#   184.6), both CROSS-START, labelled.
# ~45 min (start ~6.5, smoke ~1, 12 blocks x ~2.25 at dur 45, identity ~10.5). NO cc_sim, no 2.9; no power / host / fan settings;
# production compose files read only. Exact: place_lf moves only which expert sits in which lent slot (same kernel, same bytes);
# SD at T > 0 has the user's GO.
# Knobs: XLF_IMAGE, XLF_ITEM, XLF_OUT, XLF_ORDER (ABBAABBA), XLF_T1ORDER (ABBA; empty = none), XLF_DUR (45), XLF_WARM (10),
#   XLF_DS (1: draft_sample in the pin), XLF_OCC (3), XLF_OCCK (32), XLF_IDARM (B), XLF_SIDECAR, XLF_SIDESHA.
# DRYRUN=1 bash dsm-lf.sh : prints every command; checks the image literals (the 35-switch registry, BI / SD / DLW / Z0 / LF), every
# helper (files, parse, mbt.py / loadid.py selftests, lfblocks.py --help), the sidecars, the yaml; no server, no GPU, writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/../ds41-stack2-tp4.yaml                   # stack2's yaml, read only (env_file ${XS2_ENVFILE} carries every start word)
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh
O=${XLF_OUT:-$R/xdsm/lf}; export STATSDIR=$O/stats
RD=$R/xdsm/draft; RS3=$R/xdsm/stack3; RZO=$R/xdsm/z0occ
export IMAGE=${XLF_IMAGE:?set XLF_IMAGE}
ITEM=${XLF_ITEM:-57-p4zyzzzzz-ds-r06z-lf}
ORDER=${XLF_ORDER:-ABBAABBA}; T1ORDER=${XLF_T1ORDER-ABBA}; DUR=${XLF_DUR:-45}; WARMS=${XLF_WARM:-10}
DS=${XLF_DS:-1}; OCC=${XLF_OCC:-3}; OCCK=${XLF_OCCK:-32}; IDARM=${XLF_IDARM:-B}
SIDE=${XLF_SIDECAR:-dspark-dlw-q8_0.rad}; SIDESHA=${XLF_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
OFF=0.5; PFT=32768; CONC=4; WARM=10; MT=192
case $ORDER in *[!AB]*|"") echo "XLF_ORDER takes A / B blocks (got '$ORDER')"; exit 1 ;; esac
case $T1ORDER in *[!AB]*) echo "XLF_T1ORDER takes A / B blocks or nothing (got '$T1ORDER')"; exit 1 ;; esac
case $IDARM in A|B) ;; *) echo "XLF_IDARM takes A / B (got '$IDARM')"; exit 1 ;; esac
case $DS in ''|*[!0-9]*) echo "XLF_DS must be a draft_sample value 0..200 (got '$DS')"; exit 1 ;; esac
[ "$DS" -le 200 ] || { echo "XLF_DS must be 0..200 (got '$DS')"; exit 1; }
case $OCC in ''|*[!0-9]*) echo "XLF_OCC must be 1..64 (got '$OCC')"; exit 1 ;; esac
case $OCCK in ''|*[!0-9]*) echo "XLF_OCCK must be 1..4096 (got '$OCCK')"; exit 1 ;; esac
for v in DUR WARMS; do case ${!v} in ''|*[!0-9]*) echo "$v must be a number (got '${!v}')"; exit 1 ;; esac; done
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
LD=$HD/loadid.py; GGB=$HD/lfblocks.py; MBT=$HD/mbt.py; PRB=$HD/sdprobe.py; EP=$HD/editprobe.py; MBP=$D/mb.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
xlog() { log "RESULT xlf $*"; }
# ---------------------------------------------------------------- stack3's NST 5 start (= dsm-stack3.sh n5) + z0occ's start words
GPTAB=q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422
SIDECAR=dspark-conf.f32
LEVENV="RADIANCE_DS41_GG_PF=1 RADIANCE_DS41_GG_PF_TABLE=$GPTAB RADIANCE_DS41_F32_PF=25 RADIANCE_DS41_ARENA_DECODE_ROWS=8,48
  RADIANCE_DS41_ROUTER_BF16=1 RADIANCE_DS41_EMB_RAM=1 RADIANCE_PLACE_SHADOW_SLOTS=16 RADIANCE_XPREFETCH=0 RADIANCE_DS41_DSPARK_WALK=1
  RADIANCE_DS41_ARGATHER_MAX=48 RADIANCE_DS41_QNFOLD=1 RADIANCE_DS41_SWFOLD=1 RADIANCE_DRAFT_CTXSKIP=0 RADIANCE_DS41_DSPARK_CONF=/models/$SIDECAR
  RADIANCE_DRAFT_LEND=0 RADIANCE_DRAFT_PROBE_BACKOFF=0 RADIANCE_DRAFT_LEND_BORROW=0 RADIANCE_DRAFT_SAMPLE=0 RADIANCE_WEIGHT_SIDECAR=/models/$SIDE
  RADIANCE_DS41_BI=0 RADIANCE_DRAFT_CTXSKIP_SLOTS=129 RADIANCE_DRAFT_LEND_WEIGHTS=0"
LEVENV=$(echo $LEVENV)
# the start-log literals ('|'-separated; ',' = a space; a literal holds no comma): stack3 n5's (arena: the 48 rung) + z0occ's
LITS="gemm_nt_ggml's,prefill,rows,take,lever,GPF's,forms,(RADIANCE_DS41_GG_PF=1|gemm_nt_f32's,prefill,rows,take,lever,GPF-b's,tiled,form,(RADIANCE_DS41_F32_PF=25"
LITS="$LITS|48,tokens,use|a,rank),ram|expert,prefetch:,off"
LITS="$LITS|DSpark,lever,W1:,the,full-vocabulary,Markov,walk,declared,(dspark_walk_step,a,step,+,all_gather,over,4,rank(s)|fn4,dspark_walk=1,"
LITS="$LITS|one,kernel,up,to,48,tokens,a,step|RADIANCE_DS41_QNFOLD;,ds41_qnfold=1)|RADIANCE_DS41_SWFOLD;,ds41_swfold=1)|draft,context,skip,(RADIANCE_DRAFT_CTXSKIP=0|a,confidence,plane,(the,dspark_conf,gate"
LITS="$LITS|draft,lend:,set,up"
LITS="$LITS|DSpark,lever,SD:,the,sampled,full-vocabulary,walk,declared,(dspark_walk_sample_step,a,step,over,4,rank(s)|lever,SD:,fn4,draft_sample,0,at,start"
LITS="$LITS|weight,sidecar:,43,entries|read,from,the,weight,sidecar|lever,BI,declared,(RADIANCE_DS41_BI|now,0):,a,step,of,more"
LITS="$LITS|lever,Z0-A,set,up|activation,arena:,8,tokens,use|ring,of,129,steps|draft,weight,lend,(lever,Z0-W|mover:,lever,Z0-W:"
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": 8, "drafter_mx": 1, "gg_pf": 1, "f32_pf": 25, "dspark_walk": 1, "ds41_argather": 48, "idx_split": 1, "ds41_qnfold": 1, "ds41_swfold": 1, "draft_ctxskip": 1, "dspark_conf": 1, "dspark_conf_rows": 1, "draft_lend": 1, "draft_probe_backoff": 3, "draft_hybrid": 1}'
FIX=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update({"ds41_bi": 2, "draft_sample": int(sys.argv[2]), "draft_lend_hyst": 0, "arena_d0": 0, "draft_lend_w": 0, "draft_occ": int(sys.argv[3]), "draft_occ_k": int(sys.argv[4])}); print(json.dumps(d))' "$FIX5" "$DS" "$OCC" "$OCCK") || { echo "pin JSON"; exit 1; }
armj() { case $1 in A) echo '{"place_lf": 0}' ;; B) echo '{"place_lf": 1}' ;; esac; }
pinj() {   # pinj ARM [EXTRAJSON]: the pin + the arm + extra as one POST body
  local x=${2:-}; [ -n "$x" ] || x='{}'
  python3 -c 'import json,sys
d = {}
for a in sys.argv[1:]: d.update(json.loads(a))
print(json.dumps(d))' "$FIX" "$(armj $1)" "$x"
}
DESC="NST 5 z0occ2 start (stack3 S-arm: stack2 NST 5 env, DSpark adaptive, depth 0, hybrid 1, W1, small n5 set, drafter_mx, lend + back-off 3, VRAM levers, ds41_bi 2 pinned, draft_sample $DS pinned, --draft-coupling on, DLW sidecar $SIDE; z0occ's start words: ring 129, Z0-W declared, rungs 8,48), Z0-H / A / W 0, draft_occ $OCC / draft_occ_k $OCCK pinned in every arm"
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
  RADIANCE_PLACE_LF RADIANCE_PLACE_LF_DECAY"
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
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xlf $1 warm-up identity vs $2"; return 0; }
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
  xlog "$1 modes: ${#lit[@]} literal(s), found [$(echo $found)]$([ -n "$miss" ] && echo " -- VIRHE missing:$miss") | $(docker logs $CN 2>&1 | grep -E "lever BI declared|lever SD|weight sidecar|lever Z0|activation arena|ring of|draft lend:|draft weight lend|W activation arena|no level for" | sed 's/^radiance: //; s/^[A-Z] *//' | cut -c1-220 | sort -u | tr '\n' '|' | cut -c1-2600)"
  [ -z "$miss" ] || ok=0
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
vramstart() {
  [ "$DRY" = 1 ] && { echo "+ GET /stats > $O/$1-stats0.json; RESULT xlf $1 VRAM at start (experts per tier, rank-0 mover / lend lines; vs stack3 s3-n5) + per rank trunk units, slot classes, lenders' covered / held"; return 0; }
  curl -s -m 30 $URL/stats > $O/$1-stats0.json 2>/dev/null
  xlog "$1 VRAM at start: experts $(python3 -c 'import json,sys
for p in sys.argv[1:]:
    try: s = json.load(open(p)); print({x.get("name"): x.get("units") for x in s.get("experts", {}).get("tiers", [])}, end=" ")
    except Exception as e: print("n/a", end=" ")' $O/$1-stats0.json 2>&1 | cut -c1-160) (stack3 s3-n5 $(python3 -c 'import json,sys
try: s = json.load(open(sys.argv[1])); print({x.get("name"): x.get("units") for x in s.get("experts", {}).get("tiers", [])})
except Exception: print("n/a")' $RS3/s3-n5-stats0.json 2>&1 | cut -c1-160)) | ranks $(python3 -c 'import json,sys
try:
    s = json.load(open(sys.argv[1]))
    for r in s.get("experts", {}).get("ranks", []):
        l = r.get("lent") or {}
        print("r%s trunk %s slots %s lent %s lf %s;" % (r.get("rank"), r.get("trunk_vram"), "yes" if r.get("slots") else "NO", " ".join("%s %s/%s" % (k, v.get("held"), v.get("covered")) for k, v in l.items()), r.get("place_lf")), end=" ")
except Exception as e: print("n/a", e)' $O/$1-stats0.json 2>&1 | cut -c1-700) | $(docker logs $CN 2>&1 | grep -E '^I\[0\] mover:|draft lend: set up|weight sidecar:|draft weight lend' | sed 's/^radiance: //; s/^[A-Z]\[0\] //; s/^[A-Z] *//' | cut -c1-220 | sort -u | tr '\n' '|' | cut -c1-1800)"
}
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
fn4code() { [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null; }
pcreset() { [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o $O/$1-reset.json -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  xlog "$1 prefix cache reset: HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"; }
swnote() {   # swnote TAG: start values; refusals; each arm's set applied at a step, then A
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 at start (gg_gemv 2, aql_acq 2, K1 three 1, ds41_bi 0, draft_sample 0 + sample.capable 1, draft_occ 0, draft_occ_k 32, draft_lend_hyst 0, arena_d0 0, draft_lend_w 0, place_lf 0, draft_capable 1, epoch 0); refusals ds41_kvfold 2 / draft_occ 65 / place_lf 2 -> 400; arms A $(pinj A) | B $(armj B) applied, then A"; return 0; }
  local s0 g c1 c2 c3 a sets="" bad=""
  s0=$(fn4get)
  c1=$(fn4code '{"ds41_kvfold": 2}'); c2=$(fn4code '{"draft_occ": 65}'); c3=$(fn4code '{"place_lf": 2}')
  for a in A B A; do
    fn4set "$(pinj $a '{"epoch_bump": 1}')"; nudge; g=$(fn4get)
    python3 -c 'import json,sys
g, w = json.loads(sys.argv[1]), json.loads(sys.argv[2])
sys.exit(0 if all(g["switches"].get(k) == v for k, v in w.items()) and not g["pending"] else 1)' "$g" "$(pinj $a)" 2>/dev/null || bad="$bad $a"
    sets="$sets $a:$(echo "$g" | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print("lf=%s,occ=%s,k=%s,bi=%s,ds=%s,ep=%s" % (w.get("place_lf"), w.get("draft_occ"), w.get("draft_occ_k"), w.get("ds41_bi"), w.get("draft_sample"), d.get("epoch")))' 2>/dev/null)"
  done
  xlog "$1 switch note: at start $(echo "$s0" | cut -c1-1000) | ds41_kvfold 2 -> HTTP $c1, draft_occ 65 -> HTTP $c2, place_lf 2 -> HTTP $c3 | arms applied:$sets$([ -n "$bad" ] && echo " -- VIRHE not applied:$bad")"
  python3 -c 'import json,sys
s0 = json.loads(sys.argv[1]); w = s0["switches"]
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) and w.get("ds41_bi") == 0 and s0["epoch"] == 0
ok = ok and w["draft_adaptive"] == 1 and w["draft_depth0"] == 0 and s0["draft_capable"] == 1 and w.get("draft_sample") == 0 and (s0.get("sample") or {}).get("capable") == 1
ok = ok and w.get("draft_occ") == 0 and w.get("draft_occ_k") == 32 and w.get("draft_lend_hyst") == 0 and w.get("arena_d0") == 0 and w.get("draft_lend_w") == 0 and w.get("place_lf") == 0
sys.exit(0 if ok else 1)' "$s0" 2>/dev/null && [ "$c1" = 400 ] && [ "$c2" = 400 ] && [ "$c3" = 400 ] && [ -z "$bad" ]
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
smoke() {   # smoke TAG: under B -- the greedy set at conc 4 (mixed steps) + mb c4 20 s at T 1 (the coupled path); healthy -> 0
  local tag=$1 t0=$(date +%s) n e
  fn4set "$(pinj B '{"epoch_bump": 1}')"; nudge
  run "(cd $D && timeout 600 python3 $GS $URL $O/$tag-smoke.jsonl --conc 4 --max-tokens 32 > $O/$tag-smoke.txt 2>&1)"
  run "(cd $D && timeout 600 python3 mb.py --tag xlf-smoke --phases c4 --dur 20 --warm 5 > $O/$tag-smoke-t1.txt 2>&1)"
  [ "$DRY" = 1 ] && { echo "+ smoke $tag: arm B $(armj B): greedy set conc 4 (32 tok) 20 of 20 + mb c4 20 s at T 1 with 0 errors, /health 200"; return 0; }
  n=$(python3 -c 'import json,sys; print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l).get("text") is not None))' $O/$tag-smoke.jsonl 2>/dev/null || echo 0)
  e=$(grep -m1 '^MB ' $O/$tag-smoke-t1.txt 2>/dev/null | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[3:]); print("%s requests %s errors, %s tok/s" % (d.get("requests"), d.get("errors"), d.get("agg_dec_tps")))' 2>/dev/null || echo "no MB line")
  if [ "$n" = 20 ] && case "$e" in *" 0 errors"*) true ;; *) false ;; esac && curl -s -m 10 -o /dev/null -w '%{http_code}' $URL/health | grep -q 200; then
    xlog "$tag smoke (arm B): greedy set conc 4 20 of 20, mb c4 T 1: $e, healthy, $(( $(date +%s) - t0 )) s"; return 0; fi
  xlog "$tag smoke (arm B): VIRHE $n of 20 answered, mb c4 T 1: $e, or unhealthy | $(docker logs $CN 2>&1 | grep -E '^E |step fail|refus' | tail -3 | tr '\n' ' ' | cut -c1-500)"
  keeplog $tag-smoke; return 1
}
blocks() {   # blocks TAG KIND: t0 = XLF_ORDER with mb at temperature 0, t1 = XLF_T1ORDER at the server default; lfblocks.py
  local tag=$1 k=$2 ord mbx="" args a
  if [ $k = t0 ]; then ord=$ORDER; mbx="--mbtemp 0 --mbt $MBT"; else ord=$T1ORDER; fi
  [ -n "$ord" ] || return 0
  args=""; for a in $(echo $ord | grep -o . | awk '!s[$0]++'); do args="$args --arm '$a=$(armj $a)'"; done
  run "timeout 5400 python3 -I $GGB --url $URL --mb-dir $D --edit $EP --sdprobe $PRB --label $tag-$k $args --order $ord --phases c4,c8 --dur $DUR --warm $WARMS $mbx --fixed '$FIX' --out $O/$tag-$k-blocks.jsonl > $O/$tag-$k-blocks.txt 2>&1"
  [ "$DRY" = 1 ] && return 0
  while IFS= read -r l; do xlog "$tag $k ${l:0:3500}"; done < <(grep -E '^(BLOCK|OCC|RANKS|STATS|GGB \{"error)|Traceback' $O/$tag-$k-blocks.txt)
  xlog "$tag $k engine: $(docker logs $CN 2>&1 | grep 'lever BI stats' | tail -1 | sed 's/^radiance: //' | cut -c1-200) | draft $(fn4get | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps(d.get("draft")), json.dumps(d.get("sample")))' 2>/dev/null | cut -c1-400) | $(docker logs $CN 2>&1 | grep -cE '^E ') E lines"
}
alone() {   # alone TAG ARM NAME: pin + epoch_bump, prefix-cache reset, greedyset.py's 20 prompts at c1 ALONE, temperature 0
  local tag=$1 t0=$(date +%s)
  fn4set "$(pinj $2 '{"epoch_bump": 1}')"; nudge
  pcreset $tag-$3
  run "(cd $D && timeout 900 python3 $GS $URL $O/$tag-$3.jsonl --conc 1 --max-tokens $MT > $O/$tag-$3.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  xlog "$tag $3 (greedy set c1 ALONE, temperature 0, $MT tok, arm $2 $(armj $2)): $(tail -1 $O/$tag-$3.txt | cut -c1-200), $(( $(date +%s) - t0 )) s"
}
loaded() {   # loaded TAG ARM NAME: pin + epoch_bump, prefix-cache reset, loadid.py run (mb c4 + per probe one fresh ~32k prefill)
  local tag=$1 t0=$(date +%s) j
  fn4set "$(pinj $2 '{"epoch_bump": 1}')"; nudge
  pcreset $tag-$3
  run "(cd $D && timeout 2700 python3 -I $LD run --url $URL --gs $GS --mb $MBP --out $O/$tag-$3.jsonl --conc $CONC --warm $WARM --offset $OFF --pf-tok $PFT --max-tokens $MT > $O/$tag-$3.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  j=$(grep -m1 '^LOADID ' $O/$tag-$3.txt | cut -c8-)
  [ -n "$j" ] || { xlog "$tag $3: VIRHE no LOADID line ($(tail -3 $O/$tag-$3.txt | tr '\n' ' ' | cut -c1-400))"; return 1; }
  xlog "$tag $3 (each prompt under mb c4 + one fresh ~${PFT}-token prefill, temperature 0, arm $2): $(python3 -c 'import json,sys
d = json.loads(sys.argv[1])
print("%s of %s answered, probe %s s mean | prefills %s of %s cold, TTFT %s s mean | mixed steps a probe median %s | c4 %s requests, %s errors" % (
    d["answered"], d["of"], d["probe_s_mean"], d["pf_cold"], d["of"], d["pf_ttft_mean"], d["mixed_median"], d["c4_requests_done"], d["c4_errors"]))' "$j" 2>&1 | cut -c1-600), $(( $(date +%s) - t0 )) s"
}
cmpx() {   # cmpx TAG NAME A B: loadid.py cmp (byte for byte + first differing char / token) -> $O/TAG-cmp-NAME.json
  local r
  [ "$DRY" = 1 ] && { echo "+ python3 -I $LD cmp --url $URL $3 $4 --json $O/$1-cmp-$2.json"; return 0; }
  if [ ! -s "$3" ] || [ ! -s "$4" ]; then xlog "$1 $2: not compared (missing $3 or $4)"; echo "$1 $2 not compared" >> $O/identity.txt; return 0; fi
  r=$(timeout 300 python3 -I $LD cmp --url $URL --json $O/$1-cmp-$2.json --label "$1 $2" "$3" "$4" 2>&1 | grep -m1 '^CMPLINE ' | cut -c9-)
  echo "$1 $2 ${r:-compare failed}" >> $O/identity.txt
  xlog "$1 $2: ${r:-VIRHE compare failed} ($(basename $3) vs $(basename $4))"
}
summary() {   # per temperature / phase / arm: mean +- t95, B vs A (pairs), per-rank misses, slot classes per resident unit; refs; identity
  [ "$DRY" = 1 ] && { echo "+ RESULT xlf summary: per T / phase, arms A / B (agg tok/s, p50, tok/step, acc, ms/step, d0 share), B vs A (ABBA pairs), per rank misses / step + hit %, slot classes' hits a resident unit a step (trunk vs kv / arena / dslab / dw), lenders' held share, fills (lf / hot) and drops; vs z0occ NST 0 (c8 182.0) and stack3 NST 0 T 0, cross-start; identity"; return 0; }
  python3 - "$O" "$RS3/s3-n0-t0-blocks.jsonl" "$RZO/st-nst0-blocks.jsonl" > $O/summary.txt 2>&1 <<'PY'
import json, math, os, statistics, sys
o, r3, rz = sys.argv[1:4]
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
CLS = ("trunk", "kv", "arena", "dslab", "dw")
LND = ("kv", "arena", "dslab", "dw")
def rows(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except OSError: return []
def g(*path):
    def f(res):
        x = res
        for k in path:
            if not isinstance(x, dict): return None
            x = x.get(k)
        return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None
    return f
def ci(v):
    v = [x for x in v if x is not None]
    if not v: return None, None, 0
    m = statistics.mean(v)
    return m, (T95.get(len(v) - 1, 1.96) * statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else None), len(v)
def f1(x, n=1): return "-" if x is None else format(x, ".%df" % n)
def pc(a, b): return "%+.1f %%" % ((a / b - 1) * 100) if a and b else "-"
def refm(rs, arm, get):
    v = [get(r["res"]) for r in rs if r.get("ok") and r.get("arm") == arm and get(r["res"]) is not None]
    return statistics.mean(v) if v else None
def pairs(rs, get):
    pr = []
    for j in range(len(rs) - 1):
        p, q = rs[j], rs[j + 1]
        if {p.get("arm"), q.get("arm")} == {"A", "B"} and p.get("ok") and q.get("ok") and (j % 2 == 0):
            pa, pb = (p, q) if p["arm"] == "A" else (q, p)
            ga, gb = get(pa["res"]), get(pb["res"])
            if ga and gb is not None: pr.append(gb / ga)
    return pr
def rk_vals(r, ph):
    """per rank: misses / step, hit %, and per class hits per resident unit per step (held units sampled)."""
    sp = (r["res"].get("statsph") or {}).get(ph) or {}
    rs, sm = sp.get("ranks") or {}, sp.get("ranks_sampled") or {}
    out = {}
    for k, x in rs.items():
        s = sm.get(k) or {}
        hs = x.get("hit_step") or {}
        held = {ln: s.get(ln + "_held") for ln in LND}
        lent_total = sum(v for v in held.values() if isinstance(v, (int, float)))
        trunk_units = (s.get("trunk_vram") - lent_total) if isinstance(s.get("trunk_vram"), (int, float)) else None
        per = {"trunk": (hs.get("trunk") / trunk_units) if hs.get("trunk") is not None and trunk_units else None}
        for ln in LND:
            per[ln] = (hs.get(ln) / held[ln]) if hs.get(ln) is not None and held[ln] else None
        lt = x.get("lent") or {}
        out[k] = {"miss": x.get("miss_step"), "hit": x.get("hit_pct"), "per": per, "held": held,
                  "cov": {ln: s.get(ln + "_covered") for ln in LND}, "trunk_units": trunk_units,
                  "hit_step": hs, "miss_last": x.get("miss_last_step") or {},
                  "fills": {ln: (lt.get(ln) or {}).get("fills") for ln in LND}, "fills_lf": {ln: (lt.get(ln) or {}).get("fills_lf") for ln in LND},
                  "fills_hot": {ln: (lt.get(ln) or {}).get("fills_hot") for ln in LND},
                  "drops": {ln: sum(((lt.get(ln) or {}).get(c) or 0) for c in ("drop_victim", "drop_reserve", "drop_recall")) for ln in LND}}
    return out
ref3, refz = rows(r3), rows(rz)
out = []
for k, lab in (("t0", "T 0"), ("t1", "T 1 (server default, draft_sample in the pin)")):
    rs = rows(os.path.join(o, "lf-n5-%s-blocks.jsonl" % k))
    if not rs: continue
    ok = [r for r in rs if r.get("ok")]
    for ph in ("c4", "c8"):
        M = [("agg tok/s", g(ph, "agg_dec_tps"), 1), ("p50", g(ph, "per_stream_dec_p50"), 1), ("tok/step", g(ph, "tok_per_decode_step"), 2),
             ("acc", g(ph, "acceptance"), 3), ("ms/step", g(ph, "ms_decode_step"), 1), ("d0 share", g("statsph", ph, "d0_share"), 3),
             ("vram units", g("statsph", ph, "vram_units_sampled", "mean"), 0), ("lent units", g("statsph", ph, "lent_units_sampled", "mean"), 0)]
        parts = []
        for nm, get, nd in M:
            cells = []
            for a in ("A", "B"):
                m, h, n = ci([get(r["res"]) for r in ok if r["arm"] == a])
                if m is not None: cells.append("%s %s%s" % (a, f1(m, nd), "" if h is None else " +- " + f1(h, nd)))
            if nm in ("agg tok/s", "p50", "ms/step"):
                pr = pairs(rs, get); pm, ph_, pn = ci(pr)
                cells.append("B vs A %s (pairs %s)" % ("-" if pm is None else "%+.1f %%%s" % (100 * (pm - 1), "" if ph_ is None else " +- %.1f" % (100 * ph_)),
                                                      "/".join("%+.1f" % (100 * (x - 1)) for x in pr) or "-"))
            if nm == "agg tok/s":
                mA = ci([get(r["res"]) for r in ok if r["arm"] == "A"])[0]; mB = ci([get(r["res"]) for r in ok if r["arm"] == "B"])[0]
                rz_ = refm(refz, "N", get)
                cells.append("[z0occ NST 0 (mb server default) %s: A %s, B %s; cross-start]" % (f1(rz_), pc(mA, rz_), pc(mB, rz_)))
                if k == "t0":
                    rv = refm(ref3, "A", get)
                    cells.append("[stack3 NST 0 T 0 %s: A %s, B %s; cross-start]" % (f1(rv), pc(mA, rv), pc(mB, rv)))
            parts.append("%s: %s" % (nm, " | ".join(cells)))
        # per rank and slot class, per arm (means over the arm's good blocks)
        for a in ("A", "B"):
            per = [rk_vals(r, ph) for r in ok if r["arm"] == a]
            if not per or not per[0]: continue
            rkl = []
            for rr in sorted(per[0], key=lambda q: int(q)):
                vv = [p.get(rr) for p in per if p.get(rr)]
                mm = lambda f: ci([f(x) for x in vv])[0]
                rkl.append("r%s miss %s/st hit %s %% trunk units %s | hits/unit/st trunk %s %s | held/covered %s | fills %s (lf %s, hot %s) drops %s | miss/st by last slot %s" % (
                    rr, f1(mm(lambda x: x["miss"]), 2), f1(mm(lambda x: x["hit"]), 2), f1(mm(lambda x: x["trunk_units"]), 0),
                    f1(mm(lambda x: x["per"]["trunk"]), 4),
                    " ".join("%s %s" % (ln, f1(mm(lambda x, ln=ln: x["per"][ln]), 4)) for ln in LND),
                    " ".join("%s %s/%s" % (ln, f1(mm(lambda x, ln=ln: x["held"][ln]), 0), f1(mm(lambda x, ln=ln: x["cov"][ln]), 0)) for ln in LND),
                    "/".join(f1(mm(lambda x, ln=ln: x["fills"][ln]), 0) for ln in LND),
                    "/".join(f1(mm(lambda x, ln=ln: x["fills_lf"][ln]), 0) for ln in LND),
                    "/".join(f1(mm(lambda x, ln=ln: x["fills_hot"][ln]), 0) for ln in LND),
                    "/".join(f1(mm(lambda x, ln=ln: x["drops"][ln]), 0) for ln in LND),
                    " ".join("%s %s" % (c, f1(mm(lambda x, c=c: x["miss_last"].get(c)), 2)) for c in ("cold",) + CLS)))
            parts.append("PER RANK %s: %s" % (a, " ; ".join(rkl)))
        l = "%s %s (%d blocks, %d flagged): %s" % (lab, ph, len(rs), len(rs) - len(ok), " || ".join(parts))
        out.append(l); print("ARMS " + l)
try: ident = [l.strip() for l in open(os.path.join(o, "identity.txt")) if l.strip()]
except OSError: ident = []
bad = [l for l in ident if not ("IDENTICAL" in l and "differ" not in l)]
print("SUMMARY identity: %s || %s" % (("IDENTICAL at every compare (%d)" % len(ident)) if ident and not bad else ("NOT IDENTICAL: " + "; ".join(bad)[:800] if ident else "none"),
      " || ".join(x.split(" || PER RANK")[0] for x in out)))
PY
  while IFS= read -r l; do case "$l" in SUMMARY*) ;; *) xlog "${l:0:6000}" ;; esac; done < $O/summary.txt
  xlog "summary: $(grep -m1 '^SUMMARY ' $O/summary.txt | cut -c9- | cut -c1-5000)"
}
dstart() {   # the one NST 5 start
  local tag=lf-n5 ok
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on"
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV > $XS2_ENVFILE"; else printf '%s\n' $LEVENV > $XS2_ENVFILE; fi
  xlog "$tag: start ($DESC) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $FIX | arms A $(armj A) B $(armj B) | env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ' | cut -c1-2200)"
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
  stopfile || blocks $tag t0
  stopfile || blocks $tag t1
  stopfile || alone $tag $IDARM a1-$IDARM
  stopfile || loaded $tag $IDARM ld-$IDARM
  cmpx $tag a1-$IDARM-vs-d0 $RD/dsd-d0-gs-c1.jsonl $O/$tag-a1-$IDARM.jsonl
  cmpx $tag ld-vs-a1-$IDARM $O/$tag-a1-$IDARM.jsonl $O/$tag-ld-$IDARM.jsonl
  run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
  down
  xlog "$tag: done"
}
trap 'down' EXIT
trap 'exit 143' TERM INT
# ---------------------------------------------------------------- main
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) | one NST 5 start: $DESC | arms A place_lf 0, B place_lf 1 | T 0 order $ORDER, T 1 order ${T1ORDER:-none}, mb c4 / c8 dur $DUR warm $WARMS | identity T 0 under $IDARM | no evict, no 136k | refs z0occ NST 0 (c4 138.7 / c8 182.0, mb server default), stack3 NST 0 T 0 A (c4 138.5 / c8 184.6), cross-start"
IMGLITS="B|dspark_conf_rows, gg_pf, f32_pf, dspark_walk, ds41_bi, draft_sample, draft_lend_hyst, arena_d0, draft_lend_w, draft_occ, draft_occ_k, place_lf, epoch_bump
B|ds41_onefill, draft_depth0, draft_probe, draft_hybrid,
B|reset_prefix_cache
B|/tokenize
B|radiance:engine_steps_mixed_total
A|RADIANCE_DS41_KVFOLD
K|RADIANCE_DS41_GG_PF_TABLE
A|RADIANCE_DS41_DSPARK_WALK
K|dspark_walk_step
A|RADIANCE_DS41_ARENA_DECODE_ROWS
A|RADIANCE_DS41_BI
A|lever BI declared (RADIANCE_DS41_BI, fn4 ds41_bi, now
B|RADIANCE_DRAFT_SAMPLE
K|dspark_walk_sample_step
A|DSpark lever SD: the sampled full-vocabulary walk declared
S|mtp_couple_verify
B|RADIANCE_WEIGHT_SIDECAR
B|read from the weight sidecar
B|draft_occ takes 0 (off) .. 64
B|draft_occ_k takes 1 .. 4096
B|occ_cycles
B|draft_lend_hyst takes 0 (off) .. 100
B|arena_d0 takes 1 (on) or 0 (off)
B|draft_lend_w takes 1 (on) or 0 (off)
B|RADIANCE_DRAFT_LEND_WEIGHTS
B|lever Z0-A set up
B|draft_lent_units
B|trunk_vram
B|RADIANCE_DRAFT_CTXSKIP_SLOTS
B|place_lf takes 1 (lent slots filled by the long-memory decode heat) or 0 (off)
B|lent-slot fill (lever LF
B|RADIANCE_PLACE_LF_DECAY
B|fills_lf
B|drop_victim
B|drop_recall
B|place_lf=1 refused: "
docker image inspect $IMAGE > /dev/null 2>&1 || { xlog "VIRHE image $IMAGE not on this host"; exit 1; }
IMGMISS=$(echo "$IMGLITS" | timeout 120 docker run --rm -i --network none --entrypoint sh $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels/libr4d.so; S=/opt/radiance/share/radiance/kernels/spechub.so; n=0
while IFS= read -r l; do [ -n "$l" ] || continue; n=$((n + 1)); w=${l%%|*}; s=${l#*|}
  case $w in B) f=$B ;; A) f=$A ;; K) f=$K ;; S) f=$S ;; *) f="$B $A $K" ;; esac
  o=0; for x in $f; do grep -qzF -- "$s" $x && { o=1; break; }; done
  [ $o = 1 ] || echo "MISS $w|$s"; done
test -x /opt/radiance/bin/rad-sidecar || echo "MISS bin/rad-sidecar"; echo "CHECKED $n"' 2>&1)
case "$IMGMISS" in "CHECKED "*) ;; *) xlog "VIRHE $IMAGE is not the ds41-lf image -- missing: $(echo "$IMGMISS" | grep -v '^CHECKED' | tr '\n' ';' | cut -c1-900)"; exit 1 ;; esac
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)) is the ds41-lf image: all $(echo "$IMGMISS" | sed 's/CHECKED //') literals present (the 35-switch registry, BI / SD / DLW / Z0 / LF) + bin/rad-sidecar -- OK"
HMISS=""
for f in "$LIB29" "$LIBPR" $Y $GS $LD $GGB $MBT $PRB $EP $MBP $D/ds41_tests.py $D/fastcheck.py; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $GS $LD $GGB $MBT $PRB $EP $MBP $D/ds41_tests.py $D/fastcheck.py 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
ST=$(timeout 60 python3 -I $LD selftest --gs $GS --mb $MBP 2>&1 | tail -1); case "$ST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$LD(${ST:0:300})" ;; esac
MST=$(timeout 60 python3 -I -B $MBT --mb $MBP --selftest 2>&1 | tail -1); case "$MST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$MBT(${MST:0:300})" ;; esac
python3 -I -B $GGB --help > /dev/null 2>&1 || HMISS="$HMISS run:lfblocks.py--help"
FNS="log run mkd restart check down stopfile best_vars desc serve_args xlog armj pinj stackenv keeplog identw modes vramstart fn4get fn4set nudge fn4code pcreset swnote gate1 smoke blocks alone loaded cmpx summary dstart"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee sha256sum; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
for a in A B; do python3 -c 'import json,sys; json.loads(sys.argv[1])' "$(pinj $a)" 2>/dev/null || HMISS="$HMISS json:$a"; done
[ "$(sha256sum $MODELDIR/$SIDECAR 2>/dev/null | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256 2>/dev/null)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR"
[ "$(sha256sum $MODELDIR/$SIDE 2>/dev/null | cut -c1-64)" = "$SIDESHA" ] || HMISS="$HMISS sha256:$MODELDIR/$SIDE(want ${SIDESHA:0:16})"
grep -q 'XS2_ENVFILE' $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no env_file)"
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config refuses it)"
HMISS=$(echo $HMISS)
[ -z "$HMISS" ] || { xlog "VIRHE helpers / inputs do not resolve: $HMISS"; exit 1; }
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing, loadid.py $ST, mbt.py $MST, lfblocks.py --help, all $(echo $FNS | wc -w) functions resolve, tools on PATH, sidecars $SIDECAR + $SIDE sha256 OK, yaml accepted -- OK"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { xlog "VIRHE $M225 missing or not verified"; exit 1; }
  for f in $RD/dsd-d0-gs-c1.jsonl $WARM_REF_225 $RS3/s3-n0-t0-blocks.jsonl $RZO/st-nst0-blocks.jsonl; do [ -s $f ] || xlog "note -- reference $f missing (not compared)"; done
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDE $RD/dsd-d0-gs-c1.jsonl $WARM_REF_225 $RS3/s3-n0-t0-blocks.jsonl $RS3/s3-n5-stats0.json $RZO/st-nst0-blocks.jsonl; do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  for a in A B; do echo "+ pin $a: $(pinj $a)"; done
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(information here; the real run refuses to start beside one)"
fi
mkd $O $O/guard $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt
stopfile || dstart
summary
unset $SWITCHES
down
xlog "$ITEM: done -- $O"
