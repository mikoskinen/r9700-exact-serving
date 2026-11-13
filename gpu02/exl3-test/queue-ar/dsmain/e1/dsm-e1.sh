#!/bin/bash
# dsm-e1.sh (12.11.2026, queue-ar/dsmain/e1, ds-main-port E1 builder) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r00zzz-cz-reng.sh. Log tag "RESULT xe1 ", out runs/xdsm/e1 (XE1_OUT). Based on copies of dsm-stack4.sh
# (21ce8c8e: the start, LEVENV, the pin, modes / swnote / deepx / loadx / alone / cmpx) and dsm-psk2b.sh (421f3636: own texts
# for prefix hits, p2bq.py's construction in e1q.py); neither is edited.
# LEVER E1 (audit lever 5, LEVERS.md "E1 Engram kv GEMM dedup"; branch ds41-e1 off ds41-int7-pskglue 5093ce9): an Engram
# layer's lookup, copy and kv GEMM (q6_K 25600 x 6144, every rank, layers 1 and 14) over a prefill step's UNIQUE n-gram id rows,
# then gather_rows copies every row's kv row from its unique row. Exact by construction (inference/model.py:353 kv =
# wkv(embed(hash_ids)): the row is a function of its ids alone; gemm_nt_ggml above 48 rows has per-row arithmetic;
# arch/deepseek41/ds41_engram_dedup.h), with the 48-row guard (no pool of > 48-row launches deduplicated to <= 48 unique rows).
# Live /stats "engdd" counters. Switch fn4 ds41_engdd (RADIANCE_DS41_ENGDD=0 declares it, off at start).
# ONE NST 5 START = stack4's (runs/ds41-exl3.log 2026-11-11 21:51:34 "RESULT xs4 s4-n5: start": stack3's S start + z0occ2's Z0
# start words + RADIANCE_DS41_PSK=0, the stack2 yaml, MNBT 4160) on the E1 image + RADIANCE_DS41_ENGDD=0 (declares lever E1).
# ARMS in process (POST /admin/fn4, the fixed pin = stack4's + its S switches {"ds41_bi": 2, "draft_sample": 85, "draft_occ": 6,
# "place_lf": 0, "ds41_psk": 2, "ds41_psk_k": 4}): A = + ds41_engdd 0, E = + ds41_engdd 1. Every arm read back after its apply.
# Per start, in this order:
#   P0 one card, no server: the image's r4d_selftest --case gather_rows_bf16 (the 16-byte gather E1 issues and the 2-byte one,
#      byte for byte against the host) -- a FAIL stops the item (VIRHE)
#   start (gate1) -> modes (stack4's literals + 'lever E1 declared ... now 0') -> VRAM at start -> switch note (start values,
#   ds41_engdd 0; ds41_engdd 2 -> 400; A / E applied and read back) -> warm-up answers vs stock ->
#   COUNTER GUARD: a cold deepq ~XE1_GDEP (16000) prompt under A, then E: /stats engdd must move under E (passes, rows_deduped
#     > 0) and stay put under A; else VIRHE and no speed reading (identity still runs) ->
#   IDENTITY ALONE: the greedy set c1 (20 prompts, 192 tok, T 0) under A and E (strict E vs A) ->
#   mb c8 once per arm (stack4's: temperature 0 via mbt.py, dur XE1_DUR 60 warm XE1_WARM 15) -- no-regression check ->
#   COLD TTFT ABBA (XE1_PORDER AEEA) per depth XE1_DEPTHS (135000 64000 16000) x prompt XE1_SRCS: deepq's fixed prompt
#     (deepq.py, XE1_NTOKD 512 tok) AND the realistic code prompt (e1q.py --src code: ~135k tokens of our own repo sources,
#     code-body.txt md5 71fbba44, prefixes at 64k / 16k; XE1_NTOKC 256 tok); each run after a prefix-cache reset, cached 0
#     required; the engdd counters a run (rows deduped / pooled = the repeat share it found); identity: every run vs the
#     first A of its prompt (strict), deepq 136k A vs stack4's S 136k answer and vs dsm-225 (cross-start image check) ->
#   IDENTITY UNDER LOAD: c4 decode streams + one cold deepq XE1_LDEPTH (64000) probe (pskld.py --load) under A, then E,
#     both with sched_bi_chunk XE1_LBIC (3 = option D: the probe's chunks on alone's pieces, so the bytes do not depend on
#     timing): ld-E vs ld-A strict; each vs the alone 64k deepq answer (D's claim; information) -- ld-E vs ld-A turns to
#     information when A's own loaded answer differs from its alone one (the chunking then depended on timing, not E1) ->
#   IDENTITY ON HITS (own texts, e1q.py --src deep: deepq's request at SRC offset XE1_DOFFE 100 for E's hits, XE1_DOFFA 300 for
#     A's; ds41_tests.SRC holds 624 files on gpu02, a larger offset wraps to deepq's file 0): the hitting arm X: reset, cold XE1_HIT1 (64000), then NO reset XE1_HIT2 (40000) -- a prefix hit resuming at X's own
#     snapshot (cached > 0 required); then the other arm Y: reset, the same XE1_HIT2 prompt cold (cached 0: a reset drops the
#     attention blocks, so no record is used); X's hit answer vs Y's cold answer strict. E then A as X.
# SUMMARY: TTFT per (prompt, depth) A / E mean, E vs A % (pairs), the repeat share and rows deduped a run, s saved at 136k;
#   mb c8 E vs A; the counter guard; identity (strict / information).
# ~30 min (estimate; <= 45 worst): P0 ~1, start + checks ~6, guard ~1, alone ~2.5, mb c8 ~3.5, cold ABBA ~10.5 (136k deepq 4 x
#   ~44 s, code 4 x ~39 s, 64k 8 x ~22 s, 16k 8 x ~11 s), loaded ~2, hits ~2.5, summary ~0.5. NO evict, no cc_sim, no 2.9, no
#   power / host / fan settings; production compose files read only. Exact by design; bytes measured and reported.
# Knobs: XE1_IMAGE, XE1_ITEM, XE1_OUT, XE1_DEPTHS, XE1_SRCS (deep code), XE1_PORDER (AEEA), XE1_NTOKD (512), XE1_NTOKC (256),
#   XE1_GDEP (16000), XE1_LDEPTH (64000), XE1_LBIC (3), XE1_HIT1 (64000), XE1_HIT2 (40000), XE1_HNTOK (256), XE1_DOFFE (100),
#   XE1_DOFFA (300), XE1_DUR (60), XE1_WARM (15), XE1_SIDECAR, XE1_SIDESHA.
# DRYRUN=1 bash dsm-e1.sh : prints every command; checks the image literals (the registry with ds41_engdd, lever E1's
#   literals and /stats "engdd", libr4d's 16-byte gather, r4d_selftest's gather case, e1-PROVENANCE), every helper (files,
#   parse, selftests of e1q.py / loadid.py / mbt.py / pskld.py, deepq.py = pskld.py's deep request, the code body's md5), the
#   sidecars, the yaml; no server, no GPU, writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/../ds41-stack2-tp4.yaml                   # stack2's yaml, read only (env_file ${XS2_ENVFILE} carries every start word)
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh
O=${XE1_OUT:-$R/xdsm/e1}; export STATSDIR=$O/stats
RD=$R/xdsm/draft; RM=$R/xdsm; RS4=$R/xdsm/stack4
export IMAGE=${XE1_IMAGE:?set XE1_IMAGE}
ITEM=${XE1_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-reng}
DEPTHS=${XE1_DEPTHS:-135000 64000 16000}; SRCS=${XE1_SRCS:-deep code}; PORDER=${XE1_PORDER:-AEEA}
NTOKD=${XE1_NTOKD:-512}; NTOKC=${XE1_NTOKC:-256}; GDEP=${XE1_GDEP:-16000}; LDEPTH=${XE1_LDEPTH:-64000}; LBIC=${XE1_LBIC:-3}
HIT1=${XE1_HIT1:-64000}; HIT2=${XE1_HIT2:-40000}; HNTOK=${XE1_HNTOK:-256}; DOFFE=${XE1_DOFFE:-100}; DOFFA=${XE1_DOFFA:-300}
DUR=${XE1_DUR:-60}; WARMS=${XE1_WARM:-15}
SIDE=${XE1_SIDECAR:-dspark-dlw-q8_0.rad}; SIDESHA=${XE1_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
OCC=6; OCCK=32; DSA=85; PSKK=4; MT=192; CODEMD5=71fbba44
case $PORDER in *[!AE]*|"") echo "XE1_PORDER takes A / E (got '$PORDER')"; exit 1 ;; esac
for s in $SRCS; do case $s in deep|code) ;; *) echo "XE1_SRCS takes deep / code (got '$s')"; exit 1 ;; esac; done
for v in $DEPTHS $NTOKD $NTOKC $GDEP $LDEPTH $LBIC $HIT1 $HIT2 $HNTOK $DOFFE $DOFFA $DUR $WARMS; do
  case $v in ''|*[!0-9]*) echo "a numeric knob is not a number ('$v')"; exit 1 ;; esac; done
[ "$HIT2" -lt "$HIT1" ] || { echo "XE1_HIT2 ($HIT2) must be below XE1_HIT1 ($HIT1): the hit resumes inside the cold run's prefix"; exit 1; }
[ "$DOFFE" != "$DOFFA" ] || { echo "XE1_DOFFE and XE1_DOFFA must differ (own texts)"; exit 1; }
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
LD=$HD/loadid.py; MBT=$HD/mbt.py; PL=$HD/pskld.py; EQ=$HD/e1q.py; CODE=$HD/code-body.txt
DQ=$QA/deepq.py; MBP=$D/mb.py; TESTS=$D/ds41_tests.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
xlog() { log "RESULT xe1 $*"; }
# ---------------------------------------------------------------- the start (= dsm-stack4.sh's + RADIANCE_DS41_ENGDD=0)
GPTAB=q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422
SIDECAR=dspark-conf.f32
LEVENV="RADIANCE_DS41_GG_PF=1 RADIANCE_DS41_GG_PF_TABLE=$GPTAB RADIANCE_DS41_F32_PF=25 RADIANCE_DS41_ARENA_DECODE_ROWS=8,48
  RADIANCE_DS41_ROUTER_BF16=1 RADIANCE_DS41_EMB_RAM=1 RADIANCE_PLACE_SHADOW_SLOTS=16 RADIANCE_XPREFETCH=0 RADIANCE_DS41_DSPARK_WALK=1
  RADIANCE_DS41_ARGATHER_MAX=48 RADIANCE_DS41_QNFOLD=1 RADIANCE_DS41_SWFOLD=1 RADIANCE_DRAFT_CTXSKIP=0 RADIANCE_DS41_DSPARK_CONF=/models/$SIDECAR
  RADIANCE_DRAFT_LEND=0 RADIANCE_DRAFT_PROBE_BACKOFF=0 RADIANCE_DRAFT_LEND_BORROW=0 RADIANCE_DRAFT_SAMPLE=0 RADIANCE_WEIGHT_SIDECAR=/models/$SIDE
  RADIANCE_DS41_BI=0 RADIANCE_DRAFT_CTXSKIP_SLOTS=129 RADIANCE_DRAFT_LEND_WEIGHTS=0 RADIANCE_DS41_PSK=0 RADIANCE_DS41_ENGDD=0"
LEVENV=$(echo $LEVENV)
LITS="gemm_nt_ggml's,prefill,rows,take,lever,GPF's,forms,(RADIANCE_DS41_GG_PF=1|gemm_nt_f32's,prefill,rows,take,lever,GPF-b's,tiled,form,(RADIANCE_DS41_F32_PF=25"
LITS="$LITS|48,tokens,use|a,rank),ram|expert,prefetch:,off"
LITS="$LITS|DSpark,lever,W1:,the,full-vocabulary,Markov,walk,declared,(dspark_walk_step,a,step,+,all_gather,over,4,rank(s)|fn4,dspark_walk=1,"
LITS="$LITS|one,kernel,up,to,48,tokens,a,step|RADIANCE_DS41_QNFOLD;,ds41_qnfold=1)|RADIANCE_DS41_SWFOLD;,ds41_swfold=1)|draft,context,skip,(RADIANCE_DRAFT_CTXSKIP=0|a,confidence,plane,(the,dspark_conf,gate"
LITS="$LITS|draft,lend:,set,up"
LITS="$LITS|DSpark,lever,SD:,the,sampled,full-vocabulary,walk,declared,(dspark_walk_sample_step,a,step,over,4,rank(s)|lever,SD:,fn4,draft_sample,0,at,start"
LITS="$LITS|weight,sidecar:,43,entries|read,from,the,weight,sidecar|lever,BI,declared,(RADIANCE_DS41_BI|now,0):,a,step,of,more"
LITS="$LITS|lever,Z0-A,set,up|activation,arena:,8,tokens,use|ring,of,129,steps|draft,weight,lend,(lever,Z0-W|mover:,lever,Z0-W:"
LITS="$LITS|lever,PSK,declared,(RADIANCE_DS41_PSK"
LITS="$LITS|lever,E1,declared,(RADIANCE_DS41_ENGDD|now,0):,the,Engram,lookup,/,copy,/,kv,GEMM,(25600,x,6144)|rows,at,2,Engram,layer(s)"
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": 8, "drafter_mx": 1, "gg_pf": 1, "f32_pf": 25, "dspark_walk": 1, "ds41_argather": 48, "idx_split": 1, "ds41_qnfold": 1, "ds41_swfold": 1, "draft_ctxskip": 1, "dspark_conf": 1, "dspark_conf_rows": 1, "draft_lend": 1, "draft_probe_backoff": 3, "draft_hybrid": 1}'
FIX=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update({"draft_lend_hyst": 0, "arena_d0": 0, "draft_lend_w": 0, "draft_occ_k": int(sys.argv[2])}); print(json.dumps(d))' "$FIX5" "$OCCK") || { echo "pin JSON"; exit 1; }
SJ="\"ds41_bi\": 2, \"draft_sample\": $DSA, \"draft_occ\": $OCC, \"place_lf\": 0, \"ds41_psk\": 2, \"ds41_psk_k\": $PSKK"
armj() {   # armj A|E: stack4's S switches + ds41_engdd
  case $1 in
    A) echo "{$SJ, \"ds41_engdd\": 0}" ;;
    E) echo "{$SJ, \"ds41_engdd\": 1}" ;;
  esac
}
pinj() {   # pinj ARM [EXTRAJSON]: the fixed pin + the arm + extra as one POST body
  local x=${2:-}; [ -n "$x" ] || x='{}'
  python3 -c 'import json,sys
d = {}
for a in sys.argv[1:]: d.update(json.loads(a))
print(json.dumps(d))' "$FIX" "$(armj $1)" "$x"
}
DESC="NST 5 stack4 start (stack3 S: stack2 NST 5 env, DSpark adaptive, depth 0, hybrid 1, W1, small n5 set, drafter_mx, lend + back-off 3, VRAM levers, --draft-coupling on, DLW sidecar $SIDE; z0occ2's ring 129, Z0-W declared, rungs 8,48; PSK declared off) + RADIANCE_DS41_ENGDD=0 (lever E1 declared, off) | fixed pin: stack3's + Z0 switches 0, draft_occ_k $OCCK | A: stack4's S ($SJ) + ds41_engdd 0, E: + ds41_engdd 1"
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
  RADIANCE_DS41_PSK RADIANCE_PLACE_LF RADIANCE_PLACE_LF_DECAY RADIANCE_DS41_ENGDD"
# ---------------------------------------------------------------- copied from dsm-stack4.sh (21ce8c8e; tag xs4 -> xe1, RS3 -> RS4)
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xe1 $1 warm-up identity vs $2"; return 0; }
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
  xlog "$1 modes: ${#lit[@]} literal(s), found [$(echo $found)]$([ -n "$miss" ] && echo " -- VIRHE missing:$miss") | $(docker logs $CN 2>&1 | grep -E "lever E1|lever PSK|lever BI declared|lever SD|weight sidecar|lever Z0|activation arena|ring of|draft lend:|draft weight lend|no level for" | sed 's/^radiance: //; s/^[A-Z] *//' | cut -c1-300 | sort -u | tr '\n' '|' | cut -c1-3000)"
  [ -z "$miss" ] || ok=0
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
vramstart() {
  [ "$DRY" = 1 ] && { echo "+ GET /stats > $O/$1-stats0.json; RESULT xe1 $1 VRAM at start (experts per tier; vs stack4 s4-n5)"; return 0; }
  curl -s -m 30 $URL/stats > $O/$1-stats0.json 2>/dev/null
  xlog "$1 VRAM at start: experts $(python3 -c 'import json,sys
for p in sys.argv[1:]:
    try: s = json.load(open(p)); print({x.get("name"): x.get("units") for x in s.get("experts", {}).get("tiers", [])}, end=" ")
    except Exception as e: print("n/a", end=" ")' $O/$1-stats0.json $RS4/s4-n5-stats0.json 2>&1 | cut -c1-320) (this start, then stack4 s4-n5) | $(docker logs $CN 2>&1 | grep -E '^I\[0\] mover:|checkpoint slots|weight sidecar:|activation arena' | sed 's/^radiance: //; s/^[A-Z]\[0\] //; s/^[A-Z] *//' | cut -c1-220 | sort -u | tr '\n' '|' | cut -c1-1500)"
}
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
fn4code() { [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null; }
pcreset() { [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o $O/$1-reset.json -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  xlog "$1 prefix cache reset: HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"; }
ARMOK=1
arm() {   # arm A|E [EXTRAJSON]: the pin + the arm (+ extra) + epoch_bump, a nudge to apply it at a step, then READ BACK
  local want g
  want=$(pinj $1 "${2:-}")
  fn4set "$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d["epoch_bump"]=1; print(json.dumps(d))' "$want")"; nudge
  [ "$DRY" = 1 ] && return 0
  g=$(fn4get)
  python3 -c 'import json,sys
g, w = json.loads(sys.argv[1]), json.loads(sys.argv[2])
sys.exit(0 if all(g["switches"].get(k) == v for k, v in w.items()) and not g["pending"] else 1)' "$g" "$want" 2>/dev/null \
    || { xlog "VIRHE arm $1${2:+ + $2} not read back: $(echo "$g" | cut -c1-600)"; ARMOK=0; }
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
pskstat() { [ "$DRY" = 1 ] && { echo "(dry)"; return 0; }; echo "$(docker logs $CN 2>&1 | grep 'lever PSK:' | tail -1 | sed 's/^radiance: deepseek41: //' | cut -c1-200) | VIRHE lines $(docker logs $CN 2>&1 | grep -c 'lever PSK design 2: VIRHE')"; }
ssnap() { [ "$DRY" = 1 ] && { echo "+ GET /stats > $O/st-$1.json"; return 0; }; curl -s -m 30 $URL/stats > $O/st-$1.json 2>/dev/null; }
gate1() {   # restart + ONE check; over the limit logged (in-process A/B: a slow state shows in every arm)
  local tag=$1 lim=$2 ms st=fast
  restart $tag || { keeplog $tag-fail; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && return 0
  awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}' && { st="SLOW (not retried)"; keeplog $tag-slow; }
  echo "$st $ms" > $O/guard-$tag.txt
  xlog "guard $tag: $st, 1-user ${ms} ms/step (limit $lim; stack4 s4-n5 $(cat $RS4/guard-s4-n5.txt 2>/dev/null))"
}
alone() {   # alone TAG ARM NAME: pin + epoch_bump, prefix-cache reset, greedyset.py's 20 prompts at c1 ALONE, temperature 0
  local tag=$1 t0=$(date +%s)
  arm $2; pcreset $tag-$3; ssnap $tag-$3-0
  run "(cd $D && timeout 900 python3 $GS $URL $O/$tag-$3.jsonl --conc 1 --max-tokens $MT > $O/$tag-$3.txt 2>&1)"
  ssnap $tag-$3-1
  [ "$DRY" = 1 ] && return 0
  xlog "$tag $3 (greedy set c1 ALONE, temperature 0, $MT tok, arm $2 $(armj $2)): $(tail -1 $O/$tag-$3.txt | cut -c1-200), $(( $(date +%s) - t0 )) s | engdd $(edelta $tag-$3-0 $tag-$3-1)"
}
cmpx() {   # cmpx TAG NAME A B: loadid.py cmp (byte for byte + the first differing char / token) -> identity.txt
  local r
  [ "$DRY" = 1 ] && { echo "+ python3 -I $LD cmp --url $URL $3 $4 --json $O/$1-cmp-$2.json"; return 0; }
  if [ ! -s "$3" ] || [ ! -s "$4" ]; then xlog "$1 $2: not compared (missing $3 or $4)"; echo "$1 $2 not compared" >> $O/identity.txt; return 0; fi
  r=$(timeout 300 python3 -I $LD cmp --url $URL --json $O/$1-cmp-$2.json --label "$1 $2" "$3" "$4" 2>&1 | grep -m1 '^CMPLINE ' | cut -c9-)
  idrec "$1-$2" "${r:-VIRHE compare failed}" "$(basename $3) vs $(basename $4)"
}
# ---------------------------------------------------------------- lever E1's own functions
edelta() {   # edelta A B -> JSON: /stats engdd deltas (passes, rows_deduped, rows_pooled, guard48, low_share, steps, no_pool) + share
  [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }
  python3 -c 'import json,sys
def rd(p):
    try: return json.load(open(p)).get("engdd")
    except Exception: return None
a, b = rd(sys.argv[1]), rd(sys.argv[2])
if a is None or b is None: print(json.dumps({"missing": True})); sys.exit(0)
d = {k: int((b.get(k) or 0) - (a.get(k) or 0)) for k in ("passes", "rows_deduped", "rows_pooled", "guard48", "low_share", "steps", "no_pool")}
d["share"] = round(d["rows_deduped"] / d["rows_pooled"], 4) if d["rows_pooled"] else None
print(json.dumps(d))' $O/st-$1.json $O/st-$2.json 2>/dev/null || echo '{"missing": true}'
}
estat() { [ "$DRY" = 1 ] && { echo "(dry)"; return 0; }; docker logs $CN 2>&1 | grep 'lever E1:' | tail -1 | sed 's/^radiance: deepseek41: //' | cut -c1-400; }
rec() {   # rec NAME ARM SRC DEPTH RESETFLAG TEXT JSONLINE MDELTA EDELTA -> deep.jsonl
  python3 -c 'import json,sys
d = json.loads(sys.argv[7] or "{}")
for x in (sys.argv[8], sys.argv[9]):
    try: e = json.loads(x)
    except Exception: e = {}
    if isinstance(e, dict): d.update({k if k not in d else "m_" + k: v for k, v in e.items()})
try: d["engdd"] = json.loads(sys.argv[9])
except Exception: d["engdd"] = None
d.update(name=sys.argv[1], arm=sys.argv[2], src=sys.argv[3], depth=int(sys.argv[4]), reset=sys.argv[5] == "reset", text=sys.argv[6])
print(json.dumps(d))' "$@" >> $O/deep.jsonl
}
runline() {   # runline JSONLINE MDELTA EDELTA -> one readable summary of a run
  python3 -c 'import json,sys
d = json.loads(sys.argv[1] or "{}")
try: m = json.loads(sys.argv[2] or "{}")
except Exception: m = {}
try: e = json.loads(sys.argv[3] or "{}")
except Exception: e = {}
pt, ca, tt = d.get("prompt_tokens"), d.get("cached"), d.get("ttft")
print("status %s, prompt %s tok, cached %s, TTFT %s s (%s prefill tok/s), %s tok, %s chars | non-decode steps %s, %s ms mean, acceptance %s | engdd: %s passes, %s of %s pooled rows deduped (share %s), guard48 %s, low_share %s, no_pool %s" % (
    d.get("status"), pt, ca, tt, round((pt - (ca or 0)) / tt, 1) if tt and pt else "-", d.get("tokens"), d.get("chars"),
    m.get("nondecode_steps"), m.get("nondecode_ms"), m.get("acc"), e.get("passes"), e.get("rows_deduped"), e.get("rows_pooled"),
    e.get("share"), e.get("guard48"), e.get("low_share"), e.get("no_pool")))' "$1" "$2" "$3" 2>&1 | cut -c1-700
}
deepx() {   # deepx TAG ARM DEPTH NAME [noreset]: deepq.py's fixed prompt at DEPTH, NTOKD greedy tokens
  local tag=$1 a=$2 dep=$3 nm=$4 t0=$(date +%s) j md ed
  arm $a
  [ "${5:-}" = noreset ] || pcreset $tag-$nm
  msnap $tag-$nm-0; ssnap $tag-$nm-0
  run "(cd $D && timeout 1800 python3 $DQ $O/$tag-$nm.txt $dep $NTOKD > $O/$tag-$nm.out 2>&1)"
  msnap $tag-$nm-1; ssnap $tag-$nm-1
  [ "$DRY" = 1 ] && { echo "+ RESULT xe1 $tag $nm: arm $a deepq depth $dep: TTFT, cached, /stats engdd deltas -> $O/deep.jsonl"; return 0; }
  j=$(grep -m1 '^DEEPQ ' $O/$tag-$nm.out | cut -c7-); md=$(mdelta $tag-$nm-0 $tag-$nm-1); ed=$(edelta $tag-$nm-0 $tag-$nm-1)
  [ -n "$j" ] || { xlog "$tag $nm: VIRHE no answer ($(tail -2 $O/$tag-$nm.out | tr '\n' ' ' | cut -c1-300))"; return 1; }
  rec "$nm" "$a" deep "$dep" "${5:-reset}" "$O/$tag-$nm.txt" "$j" "$md" "$ed"
  xlog "$tag $nm (arm $a, deepq depth $dep, $NTOKD greedy tokens$([ "${5:-}" = noreset ] && echo ", NO reset")): $(runline "$j" "$md" "$ed") | $(thermline $t0 $(date +%s))"
}
eqx() {   # eqx TAG ARM SRC DEPTH NAME NTOK [noreset] [DOFF]: e1q.py's request (code prompt, or deepq's on text DOFF)
  local tag=$1 a=$2 src=$3 dep=$4 nm=$5 nt=$6 nr=${7:-reset} doff=${8:-0} t0=$(date +%s) j md ed sa
  arm $a
  [ "$nr" = noreset ] || pcreset $tag-$nm
  msnap $tag-$nm-0; ssnap $tag-$nm-0
  if [ $src = code ]; then sa="--src code --file $CODE"; else sa="--src deep --pskld $PL --doff $doff"; fi
  run "(cd $D && timeout 1800 python3 -I $EQ run --tests $TESTS --url $URL $sa --depth $dep --ntok $nt --out $O/$tag-$nm.json --text $O/$tag-$nm.txt > $O/$tag-$nm.out 2>&1)"
  msnap $tag-$nm-1; ssnap $tag-$nm-1
  [ "$DRY" = 1 ] && { echo "+ RESULT xe1 $tag $nm: arm $a $src depth $dep$([ $src = deep ] && echo " text $doff"): TTFT, cached, /stats engdd deltas -> $O/deep.jsonl"; return 0; }
  j=$(grep -m1 '^E1Q ' $O/$tag-$nm.out | cut -c5-); md=$(mdelta $tag-$nm-0 $tag-$nm-1); ed=$(edelta $tag-$nm-0 $tag-$nm-1)
  [ -n "$j" ] || { xlog "$tag $nm: VIRHE no answer ($(tail -2 $O/$tag-$nm.out | tr '\n' ' ' | cut -c1-300))"; return 1; }
  rec "$nm" "$a" "$src$([ $src = deep ] && echo "@$doff")" "$dep" "$nr" "$O/$tag-$nm.txt" "$j" "$md" "$ed"
  xlog "$tag $nm (arm $a, $src$([ $src = deep ] && echo " text $doff") depth $dep, $nt greedy tokens$([ "$nr" = noreset ] && echo ", NO reset")): $(runline "$j" "$md" "$ed") | $(thermline $t0 $(date +%s))"
}
SPEEDOK=1
guard() {   # guard TAG: E must move the dedup counters on a cold prefill, A must not; arms read back (arm()); else no speed reading
  local tag=$1 r
  stopfile && return 0
  deepx $tag A $GDEP guard-A
  deepx $tag E $GDEP guard-E
  if [ "$DRY" = 1 ]; then echo "+ COUNTER GUARD: /stats engdd over guard-A all 0 (passes, rows_deduped, rows_pooled, guard48, low_share, steps, no_pool) and over guard-E passes > 0, rows_deduped > 0, steps > 0; arms read back; else the guard fails: no speed reading"; return 0; fi
  r=$(python3 -c 'import json,sys
rows = {}
for l in open(sys.argv[1]):
    d = json.loads(l)
    if d.get("name") in ("guard-A", "guard-E"): rows[d["name"]] = d
a, e = (rows.get("guard-A") or {}).get("engdd"), (rows.get("guard-E") or {}).get("engdd")
why = []
if not isinstance(a, dict) or a.get("missing"): why.append("no /stats engdd for A")
elif any(a.get(k) for k in ("passes", "rows_deduped", "rows_pooled", "guard48", "low_share", "steps", "no_pool")): why.append("A moved the counters %s" % a)
if not isinstance(e, dict) or e.get("missing"): why.append("no /stats engdd for E")
elif not (e.get("passes", 0) > 0 and e.get("rows_deduped", 0) > 0 and e.get("steps", 0) > 0): why.append("E did not deduplicate %s" % e)
print(("OK" if not why else "FAIL " + "; ".join(why)) + " | A %s | E %s" % (a, e))' $O/deep.jsonl 2>&1 | cut -c1-900)
  [ $ARMOK = 1 ] || r="FAIL an arm was not read back (above) | $r"
  echo "engdd $r" >> $O/guard.txt
  case "$r" in OK*) xlog "$tag COUNTER GUARD: $r | $(estat)" ;;
    *) xlog "$tag COUNTER GUARD: VIRHE $r -- no speed reading (identity still runs)"; SPEEDOK=0 ;; esac
}
mbc8() {   # mbc8 TAG ARM: mb c8 at temperature 0 (stack4's mbt.py, dur DUR warm WARMS) -> mb.jsonl + the engdd deltas
  local tag=$1 a=$2 t0=$(date +%s) j ed md
  arm $a; msnap $tag-c8$a-0; ssnap $tag-c8$a-0
  run "(cd $D && timeout 900 python3 -I -B $MBT --mb $MBP --temperature 0 --tag xe1-c8$a --phases c8 --dur $DUR --warm $WARMS > $O/$tag-c8$a.txt 2>&1)"
  msnap $tag-c8$a-1; ssnap $tag-c8$a-1
  [ "$DRY" = 1 ] && { echo "+ RESULT xe1 $tag mb c8 arm $a: agg / per-stream tok/s, TTFT, engdd deltas -> $O/mb.jsonl"; return 0; }
  j=$(grep -m1 '^MB ' $O/$tag-c8$a.txt | cut -c4-); ed=$(edelta $tag-c8$a-0 $tag-c8$a-1); md=$(mdelta $tag-c8$a-0 $tag-c8$a-1)
  [ -n "$j" ] || { xlog "$tag mb c8 arm $a: VIRHE no MB line ($(tail -2 $O/$tag-c8$a.txt | tr '\n' ' ' | cut -c1-300))"; return 1; }
  python3 -c 'import json,sys
d = json.loads(sys.argv[1]); d.update(arm=sys.argv[2], engdd=json.loads(sys.argv[3] or "{}"), metrics=json.loads(sys.argv[4] or "{}")); print(json.dumps(d))' "$j" "$a" "$ed" "$md" >> $O/mb.jsonl
  xlog "$tag mb c8 arm $a (T 0, dur $DUR warm $WARMS): $(echo "$j" | cut -c1-500) | engdd $ed | $(thermline $t0 $(date +%s))"
}
coldabba() {   # coldabba TAG: per depth, per prompt, the arms in PORDER, each cold; identity vs the first A of the same prompt
  local tag=$1 dep src a k fa nm f
  for dep in $DEPTHS; do
    for src in $SRCS; do
      k=0; fa=""
      for a in $(echo $PORDER | grep -o .); do
        k=$((k + 1)); stopfile && return 0
        nm=c$src$dep-$k$a
        if [ $src = deep ]; then deepx $tag $a $dep $nm; else eqx $tag $a code $dep $nm $NTOKC; fi
        [ "$DRY" = 1 ] && continue
        [ -n "$fa" ] || { [ $a = A ] && fa=$O/$tag-$nm.txt; }
        [ -n "$fa" ] && [ "$O/$tag-$nm.txt" != "$fa" ] && idrec "$src$dep-$k$a-vs-A" "$(cmpfile $fa $O/$tag-$nm.txt)" "$src depth $dep, vs the first A"
        if [ $src = deep ] && [ $dep = 135000 ] && [ $a = A ] && [ $k = 1 ]; then   # the image check, cross-start
          idrec "x-deep136-1A-vs-dsm225" "$(cmpfile $O/$tag-$nm.txt $RM/deep-dsm-225.txt)" "cross-start: 136k answer vs dsm-225 NST 0 stock cold"
          f=$(ls $RS4/s4-n5-d135000-*S.txt 2>/dev/null | head -1)
          [ -n "$f" ] && idrec "x-deep136-1A-vs-stack4S" "$(cmpfile $O/$tag-$nm.txt $f)" "cross-start: 136k answer vs stack4's S ($(basename $f))" \
                      || xlog "$tag: no stack4 S 136k answer under $RS4 (not compared)"
        fi
      done
    done
  done
}
loadx() {   # loadx TAG ARM NAME: c4 decode streams + ONE cold LDEPTH deepq prompt (pskld.py --load), sched_bi_chunk LBIC pinned
  local tag=$1 a=$2 nm=$3 t0=$(date +%s) j md ed
  arm $a "{\"sched_bi_chunk\": $LBIC}"; nudge; pcreset $tag-$nm
  msnap $tag-$nm-0; ssnap $tag-$nm-0
  run "(cd $D && timeout 2700 python3 -I $PL run --url $URL --tests $TESTS --mb $MBP --which deep --depth $LDEPTH --ntok $NTOKD --load --conc 4 --warm 10 --pf-tok 0 --out $O/$tag-$nm.json --text $O/$tag-$nm.txt > $O/$tag-$nm.log 2>&1)"
  msnap $tag-$nm-1; ssnap $tag-$nm-1
  [ "$DRY" = 1 ] && { echo "+ RESULT xe1 $tag $nm: arm $a + sched_bi_chunk $LBIC: the probe's TTFT / cached, mixed steps, c4 in the prefill window, engdd deltas -> $O/load.jsonl"; return 0; }
  j=$(grep -m1 '^LONGID ' $O/$tag-$nm.log | cut -c8-); md=$(mdelta $tag-$nm-0 $tag-$nm-1); ed=$(edelta $tag-$nm-0 $tag-$nm-1)
  [ -n "$j" ] || { xlog "$tag $nm: VIRHE no LONGID line ($(tail -3 $O/$tag-$nm.log | tr '\n' ' ' | cut -c1-400))"; return 1; }
  python3 -c 'import json,sys
d = json.loads(sys.argv[1]); d.update(metrics=json.loads(sys.argv[2] or "{}"), engdd=json.loads(sys.argv[3] or "{}"), name=sys.argv[4], arm=sys.argv[5], text=sys.argv[6])
print(json.dumps(d))' "$j" "$md" "$ed" "$nm" "$a" "$O/$tag-$nm.txt" >> $O/load.jsonl
  xlog "$tag $nm (arm $a + sched_bi_chunk $LBIC, c4 decode + one cold ~$LDEPTH-depth deepq prompt, T 0): $(python3 -c 'import json,sys
d, m = json.loads(sys.argv[1]), json.loads(sys.argv[2] or "{}")
print("probe status %s, prompt %s tok, cached %s, TTFT %s s | mixed steps %s (non-decode %s at %s ms mean) | c4 %s requests, %s errors" % (
    d.get("status"), d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), m.get("mixed_steps"), m.get("nondecode_steps"),
    m.get("nondecode_ms"), d.get("c4_requests"), d.get("c4_errors")))' "$j" "$md" 2>&1 | cut -c1-500) | engdd $ed | $(thermline $t0 $(date +%s))"
}
loadid() {   # loadid TAG: ld-A then ld-E; ld-E vs ld-A strict; each vs the alone LDEPTH deepq answer (D's claim, labelled)
  local tag=$1 al
  stopfile || loadx $tag A ld-A
  stopfile || loadx $tag E ld-E
  arm A   # sched_bi_chunk back to the start's 0 with the pin (the next blocks run without it)
  fn4set '{"sched_bi_chunk": 0, "epoch_bump": 1}'; nudge
  [ "$DRY" = 1 ] && return 0
  idrec "ld-E-vs-ld-A" "$(cmpfile $O/$tag-ld-A.txt $O/$tag-ld-E.txt)" "the ~$LDEPTH probe under c4 load, sched_bi_chunk $LBIC, E vs A"
  al=$(ls $O/$tag-cdeep$LDEPTH-1A.txt 2>/dev/null | head -1)
  if [ -n "$al" ]; then
    idrec "dl-ld-A-vs-alone" "$(cmpfile $al $O/$tag-ld-A.txt)" "information (sched_bi_chunk $LBIC's claim): the loaded A answer vs the alone A answer"
    idrec "dl-ld-E-vs-alone" "$(cmpfile $al $O/$tag-ld-E.txt)" "information: the loaded E answer vs the alone A answer"
  fi
}
hitset() {   # hitset TAG X Y DOFF: arm X on text DOFF: reset, cold HIT1, NO reset HIT2 (a hit at X's own snapshot); arm Y: reset,
  # HIT2 cold on the same text; X's hit answer vs Y's cold answer (strict); the hit must have resumed (cached > 0), the cold not
  local tag=$1 x=$2 y=$3 doff=$4 r
  stopfile && return 0
  eqx $tag $x deep $HIT1 h$x-cold$HIT1 $HNTOK reset $doff
  eqx $tag $x deep $HIT2 h$x-hit$HIT2 $HNTOK noreset $doff
  eqx $tag $y deep $HIT2 h$x-ref$y$HIT2 $HNTOK reset $doff
  [ "$DRY" = 1 ] && return 0
  r=$(python3 -c 'import json,sys
rows = {d.get("name"): d for d in (json.loads(l) for l in open(sys.argv[1]))}
h, c = rows.get(sys.argv[2]) or {}, rows.get(sys.argv[3]) or {}
print("hit cached %s of %s tokens, the reference cold cached %s%s" % (h.get("cached"), h.get("prompt_tokens"), c.get("cached"),
      "" if (h.get("cached") or 0) > 0 and c.get("cached") == 0 else " -- VIRHE the hit did not resume or the reference was not cold"))' $O/deep.jsonl h$x-hit$HIT2 h$x-ref$y$HIT2 2>&1 | cut -c1-300)
  xlog "$tag hits arm $x on text $doff: $r"
  idrec "hit-$x-vs-cold$y" "$(cmpfile $O/$tag-h$x-ref$y$HIT2.txt $O/$tag-h$x-hit$HIT2.txt)" "text $doff depth $HIT2: arm $x resumed at its own snapshot vs arm $y cold ($r)"
}
swnote() {   # swnote TAG: start values; ds41_engdd 2 refused; A and E applied and read back (arm()), then A
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 at start (gg_gemv 2, aql_acq 2, K1 three 1, ds41_bi 0, ds41_psk 0, place_lf 0, draft_occ 0, draft_occ_k 32, draft_sample 0, ds41_engdd 0, epoch 0); ds41_engdd 2 -> 400; arms A $(pinj A) | E $(armj E) applied at a step and read back, then A"; return 0; }
  local s0 c1 a sets=""
  s0=$(fn4get)
  c1=$(fn4code '{"ds41_engdd": 2}')
  for a in A E A; do arm $a; sets="$sets $a:engdd=$(fn4get | python3 -c 'import json,sys; d=json.load(sys.stdin); print("%s,psk=%s/%s,bi=%s,ep=%s" % (d["switches"].get("ds41_engdd"), d["switches"].get("ds41_psk"), d["switches"].get("ds41_psk_k"), d["switches"].get("ds41_bi"), d.get("epoch")))' 2>/dev/null)"; done
  xlog "$1 switch note: at start $(echo "$s0" | cut -c1-1200) | ds41_engdd 2 -> HTTP $c1 | arms applied and read back:$sets$([ $ARMOK = 1 ] || echo " -- VIRHE an arm was not read back")"
  python3 -c 'import json,sys
s0 = json.loads(sys.argv[1]); w = s0["switches"]
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) and s0["epoch"] == 0
ok = ok and w.get("ds41_bi") == 0 and w.get("ds41_psk") == 0 and w.get("place_lf") == 0 and w.get("draft_occ") == 0 and w.get("draft_occ_k") == 32
ok = ok and w["draft_adaptive"] == 1 and w["draft_depth0"] == 0 and s0["draft_capable"] == 1 and w.get("draft_sample") == 0 and w.get("ds41_engdd") == 0
sys.exit(0 if ok else 1)' "$s0" 2>/dev/null && [ "$c1" = 400 ] && [ $ARMOK = 1 ]
}
p0sel() {   # p0sel: one card, no server -- the image's r4d_selftest --case gather_rows_bf16
  local r rc
  run "gpurun xe1-gr-$$ 600 $DEV --entrypoint /opt/radiance/bin/r4d_selftest $IMAGE /opt/radiance/share/radiance/kernels/libr4d.so --case gather_rows_bf16 > $O/p0-gather.txt 2>&1"; rc=$?
  [ "$DRY" = 1 ] && { echo "+ RESULT xe1 P0 gather_rows_bf16 on one card: 7 byte checks + the launch rows, 0 FAIL, else a failure line and the item stops"; return 0; }
  r="$(grep -c '^  ok' $O/p0-gather.txt) ok, $(grep -c '^  FAIL' $O/p0-gather.txt) FAIL (rc $rc): $(grep -E 'byte for byte' $O/p0-gather.txt | sed 's/^ *//' | cut -c1-120 | tr '\n' ';' | cut -c1-900)"
  if [ $rc = 0 ] && [ "$(grep -c '^  FAIL' $O/p0-gather.txt)" = 0 ] && [ "$(grep -c 'byte for byte' $O/p0-gather.txt)" -ge 7 ]; then xlog "P0 gather_rows_bf16 on one card (r4d_selftest --case): OK $r"; return 0; fi
  xlog "P0 gather_rows_bf16 on one card (r4d_selftest --case): VIRHE $r"; return 1
}
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ulimit memlock=-1 --ipc=host -e HIP_VISIBLE_DEVICES=0"
gpurun() {   # gpurun NAME SECONDS docker-run-args...: bounded, named, removed even if its kernel hangs (dsm-bidual.sh's, copied)
  local name=$1 t=$2; shift 2
  timeout -k 15 $t docker run --rm --init --stop-timeout 10 --name $name "$@"; local rc=$?
  docker rm -f $name > /dev/null 2>&1 || true
  { [ $rc = 124 ] || [ $rc = 137 ]; } && echo "gpurun: $name stopped after $t s (rc $rc)" >&2
  return $rc
}
summary() {
  [ "$DRY" = 1 ] && { echo "+ RESULT xe1 summary: TTFT per prompt / depth A / E mean, E vs A % (pairs), repeat share and rows deduped a run, s saved at 136k; mb c8 E vs A; the counter guard; identity (strict / information)"; return 0; }
  python3 - "$O" "$PORDER" > $O/summary.txt 2>&1 <<'PY'
import json, os, statistics, sys
o, porder = sys.argv[1:3]
def rows(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except OSError: return []
def mean(v):
    v = [x for x in v if isinstance(x, (int, float))]
    return statistics.mean(v) if v else None
def f(x, n=2): return "-" if x is None else format(x, ".%df" % n)
def pc(a, b): return "%+.2f %%" % ((a / b - 1) * 100) if a and b else "-"
dp = [r for r in rows(os.path.join(o, "deep.jsonl")) if r.get("name", "").startswith("c")]
pp, saved = [], {}
for (src, dep) in sorted({(r["src"], r["depth"]) for r in dp}, key=lambda x: (-x[1], x[0])):
    s = [r for r in dp if r["src"] == src and r["depth"] == dep]
    ta, te = mean([r.get("ttft") for r in s if r["arm"] == "A"]), mean([r.get("ttft") for r in s if r["arm"] == "E"])
    pr = []
    for p, q in zip(s, s[1:]):
        if {p["arm"], q["arm"]} == {"A", "E"} and p.get("ttft") and q.get("ttft"):
            a, b = (p, q) if p["arm"] == "A" else (q, p); pr.append("%+.2f" % (100 * (b["ttft"] / a["ttft"] - 1)))
    ee = [r.get("engdd") or {} for r in s if r["arm"] == "E"]
    ea = [r.get("engdd") or {} for r in s if r["arm"] == "A"]
    dd, pl = mean([e.get("rows_deduped") for e in ee]), mean([e.get("rows_pooled") for e in ee])
    ndm_a, ndm_e = mean([r.get("nondecode_ms") for r in s if r["arm"] == "A"]), mean([r.get("nondecode_ms") for r in s if r["arm"] == "E"])
    cold = all(r.get("cached") == 0 for r in s)
    amoved = any(any((e.get(k) or 0) for k in ("passes", "rows_deduped", "rows_pooled")) for e in ea)
    l = "%s %d (%s tok): TTFT A %s s, E %s s, E vs A %s (pairs %s), E - A %s s | non-decode step A %s ms, E %s ms | E deduped %s of %s pooled rows a run (share %s), passes %s, guard48 %s, low_share %s%s%s" % (
        src, dep, s[0].get("prompt_tokens"), f(ta), f(te), pc(te, ta), "/".join(pr) or "-", f((te - ta) if ta and te else None),
        f(ndm_a, 1), f(ndm_e, 1), f(dd, 0), f(pl, 0), f(dd / pl if dd and pl else None, 3), f(mean([e.get("passes") for e in ee]), 1),
        f(mean([e.get("guard48") for e in ee]), 1), f(mean([e.get("low_share") for e in ee]), 1),
        "" if cold else " -- VIRHE a run was not cold", " -- VIRHE A moved the counters" if amoved else "")
    if ta and te: saved["%s %d" % (src, dep)] = ta - te
    pp.append(l)
for l in pp: print("TTFT " + l)
mb = rows(os.path.join(o, "mb.jsonl"))
mbl = "-"
if mb:
    g = lambda arm, k: mean([r.get(k) for r in mb if r.get("arm") == arm])
    mbl = "c8 agg tok/s A %s E %s (%s), per-stream p50 A %s E %s (%s), TTFT p50 A %s E %s | E deduped %s of %s pooled rows (share %s)" % (
        f(g("A", "agg_dec_tps"), 1), f(g("E", "agg_dec_tps"), 1), pc(g("E", "agg_dec_tps"), g("A", "agg_dec_tps")),
        f(g("A", "per_stream_dec_p50"), 1), f(g("E", "per_stream_dec_p50"), 1), pc(g("E", "per_stream_dec_p50"), g("A", "per_stream_dec_p50")),
        f(g("A", "ttft_p50")), f(g("E", "ttft_p50")),
        f(mean([(r.get("engdd") or {}).get("rows_deduped") for r in mb if r.get("arm") == "E"]), 0),
        f(mean([(r.get("engdd") or {}).get("rows_pooled") for r in mb if r.get("arm") == "E"]), 0),
        f(mean([(r.get("engdd") or {}).get("share") for r in mb if r.get("arm") == "E"]), 3))
    print("MB " + mbl)
try: ident = [l.strip() for l in open(os.path.join(o, "identity.txt")) if l.strip()]
except OSError: ident = []
info = [l for l in ident if l.split(" ")[0].startswith(("dl-", "x-"))]
# the loaded E vs A compare is strict unless A's own loaded answer differs from its alone one too (the probe's chunking then
# depended on timing under sched_bi_chunk 3 -- not lever E1): then it is information, labelled
same = lambda l: "IDENTICAL" in l and "differ" not in l
lda = [l for l in ident if l.startswith("dl-ld-A-vs-alone ")]
if lda and not same(lda[0]):
    info += [l for l in ident if l.startswith("ld-E-vs-ld-A ")]
strict = [l for l in ident if l not in info]
bad = [l for l in strict if not ("IDENTICAL" in l and "differ" not in l)]
lab = lambda ls: "; ".join(l.split(" ")[0] + " " + ("IDENTICAL" if "IDENTICAL" in l and "differ" not in l else "differs") for l in ls)
try: guards = [l.strip() for l in open(os.path.join(o, "guard.txt")) if l.strip()]
except OSError: guards = []
print("SUMMARY lever E1 (ds41_engdd 1 vs 0, in process on stack4's S) || guards: %s || identity (strict, E vs A alone / under load / on hits, run-to-run): %s || information (cross-start image check, loaded vs alone): %s || TTFT: %s || s saved a cold request (A - E): %s || mb c8: %s" % (
      " | ".join(guards) or "none",
      ("IDENTICAL at every compare (%d)" % len(strict)) if strict and not bad else (("NOT IDENTICAL: " + "; ".join(bad))[:900] if strict else "none"),
      lab(info) or "-", " | ".join(pp) or "-", ", ".join("%s %+.2f" % (k, v) for k, v in saved.items()) or "-", mbl))
PY
  while IFS= read -r l; do case "$l" in SUMMARY*) ;; *) xlog "${l:0:3000}" ;; esac; done < $O/summary.txt
  xlog "summary: $(grep -m1 '^SUMMARY ' $O/summary.txt | cut -c9- | cut -c1-6000)"
}
dstart() {   # the one NST 5 start
  local tag=e1-n5 ok
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on"
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV > $XS2_ENVFILE"; else printf '%s\n' $LEVENV > $XS2_ENVFILE; fi
  xlog "$tag: start ($DESC) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $FIX | arms A $(armj A) E $(armj E) | env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ' | cut -c1-2200)"
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
  stopfile || guard $tag
  stopfile || alone $tag A a1-A
  stopfile || alone $tag E a1-E
  cmpx $tag a1-E-vs-a1-A $O/$tag-a1-A.jsonl $O/$tag-a1-E.jsonl
  if [ $SPEEDOK = 1 ]; then
    stopfile || mbc8 $tag A
    stopfile || mbc8 $tag E
  else
    xlog "$tag: VIRHE the counter guard failed (above): no mb c8, the cold runs one A and one E each (identity only, no TTFT verdict)"; PORDER=AE
  fi
  stopfile || coldabba $tag
  stopfile || loadid $tag
  stopfile || hitset $tag E A $DOFFE
  stopfile || hitset $tag A E $DOFFA
  run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
  xlog "$tag: lever E1 at the end: $(estat) | PSK $(pskstat) | $(docker logs $CN 2>&1 | grep -cE '^E ') E lines"
  down
  xlog "$tag: done"
}
trap 'down' EXIT
trap 'exit 143' TERM INT
# ---------------------------------------------------------------- main
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) | ~30 min (estimate, <= 45) | P0 gather_rows_bf16 on one card | one NST 5 start: $DESC | guard deepq $GDEP A/E | alone greedy set c1 A/E | mb c8 A, E (T 0, dur $DUR warm $WARMS) | cold $PORDER at [$DEPTHS] x [$SRCS] (deepq $NTOKD tok, code $NTOKC tok) | loaded c4 + deepq $LDEPTH (sched_bi_chunk $LBIC) A, E | hits $HIT1 -> $HIT2 on own texts $DOFFE (E) / $DOFFA (A) | no evict"
IMGLITS="B|sched_short_bypass, sched_bypass_total, ds41_psk, ds41_psk_k, ds41_engdd, epoch_bump)
B|ds41_engdd
B|rows_deduped
B|rad_engdd_stats
B|guard48
B|reset_prefix_cache
B|/tokenize
B|radiance:engine_steps_mixed_total
B|radiance:engine_step_seconds_decode_total
B|radiance:draft_accepted_total
B|sched_bi_chunk
B|place_lf
A|RADIANCE_DS41_ENGDD
A|lever E1 declared (RADIANCE_DS41_ENGDD, fn4 ds41_engdd, now
A|lever E1: %lld Engram layer passes deduplicated
A|rad_engdd_stats
A|ds41_engdd
A|engram_kv_u
A|RADIANCE_DS41_PSK
A|lever PSK declared (RADIANCE_DS41_PSK, fn4 ds41_psk, now
A|RADIANCE_DS41_KVFOLD
K|RADIANCE_DS41_GG_PF_TABLE
K|gather_rows_w16_kernel
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
B|draft_occ takes 0 (off) .. 64
B|occ_cycles
B|RADIANCE_DRAFT_LEND_WEIGHTS
B|lever Z0-A set up
B|RADIANCE_DRAFT_CTXSKIP_SLOTS
T|gather_rows_bf16 byte for byte"
docker image inspect $IMAGE > /dev/null 2>&1 || { xlog "VIRHE image $IMAGE not on this host"; exit 1; }
IMGMISS=$(echo "$IMGLITS" | timeout 120 docker run --rm -i --network none --entrypoint sh $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels/libr4d.so; S=/opt/radiance/share/radiance/kernels/spechub.so; T=/opt/radiance/bin/r4d_selftest; n=0
while IFS= read -r l; do [ -n "$l" ] || continue; n=$((n + 1)); w=${l%%|*}; s=${l#*|}
  case $w in B) f=$B ;; A) f=$A ;; K) f=$K ;; S) f=$S ;; T) f=$T ;; *) f="$B $A $K" ;; esac
  o=0; for x in $f; do grep -qzF -- "$s" $x && { o=1; break; }; done
  [ $o = 1 ] || echo "MISS $w|$s"; done
test -x /opt/radiance/bin/rad-sidecar || echo "MISS bin/rad-sidecar"; test -s /opt/radiance/share/radiance/e1-PROVENANCE.txt || echo "MISS e1-PROVENANCE.txt"; echo "CHECKED $n"' 2>&1)
case "$IMGMISS" in "CHECKED "*) ;; *) xlog "VIRHE $IMAGE is not the E1 image -- missing: $(echo "$IMGMISS" | grep -v '^CHECKED' | tr '\n' ';' | cut -c1-900)"; exit 1 ;; esac
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)) is the E1 image: all $(echo "$IMGMISS" | sed 's/CHECKED //') literals present (the registry with ds41_engdd, /stats engdd, lever E1's declare / stats lines, engram_kv_u, libr4d's gather_rows_w16_kernel, r4d_selftest's gather case, stack4's PSK / BI / SD / DLW / Z0 / LF literals) + bin/rad-sidecar + e1-PROVENANCE -- OK"
HMISS=""
for f in "$LIB29" "$LIBPR" $Y $GS $LD $MBT $PL $EQ $CODE $DQ $MBP $TESTS $D/stepprobe.py $D/fastcheck.py; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $GS $LD $MBT $PL $EQ $DQ $MBP $TESTS $D/stepprobe.py $D/fastcheck.py 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
ST=$(timeout 60 python3 -I $LD selftest --gs $GS --mb $MBP 2>&1 | tail -1); case "$ST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$LD(${ST:0:300})" ;; esac
MST=$(timeout 60 python3 -I -B $MBT --mb $MBP --selftest 2>&1 | tail -1); case "$MST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$MBT(${MST:0:300})" ;; esac
PST=$(timeout 60 python3 -I $PL selftest --tests $TESTS --mb $MBP 2>&1 | tail -1); case "$PST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$PL(${PST:0:300})" ;; esac
EST=$(timeout 60 python3 -I $EQ selftest --tests $TESTS --pskld $PL --file $CODE --md5 $CODEMD5 --doff $DOFFE --doff2 $DOFFA 2>&1 | tail -1); case "$EST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$EQ(${EST:0:300})" ;; esac
DQP=$(timeout 60 python3 -I -c 'import importlib.util, json, sys
def ld(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
t = ld(sys.argv[2], "t"); pl = ld(sys.argv[1], "pl")
import ast
src = open(sys.argv[3]).read()
task = next((ast.literal_eval(n.value) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "task" for t in n.targets)), None)
ok = "CPT, OFF = 3.95, 1200" in src and "# deepq fixed prompt v1" in src and pl.DEEP_CPT == 3.95 and pl.DEEP_OFF == 1200 and task == pl.DEEP_TASK
d = pl.deep_payload(t, 64000, 512)
print("DQ OK" if ok and d["messages"][0]["content"].startswith("# deepq fixed prompt v1\n") and d["max_tokens"] == 512 else "DQ MISMATCH")' $PL $TESTS $DQ 2>&1 | tail -1)
case "$DQP" in "DQ OK") ;; *) HMISS="$HMISS deepq-vs-pskld:(${DQP:0:200})" ;; esac
FNS="log run mkd restart check down stopfile best_vars desc serve_args xlog armj pinj stackenv keeplog thermline cmpfile idrec identw modes vramstart fn4get fn4set nudge fn4code pcreset arm msnap mdelta pskstat ssnap gate1 alone cmpx edelta estat rec runline deepx eqx guard mbc8 coldabba loadx loadid hitset swnote p0sel gpurun summary dstart"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee sha256sum md5sum; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
for a in A E; do python3 -c 'import json,sys; json.loads(sys.argv[1])' "$(pinj $a "{\"sched_bi_chunk\": $LBIC}")" 2>/dev/null || HMISS="$HMISS json:$a"; done
[ "$(sha256sum $MODELDIR/$SIDECAR 2>/dev/null | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256 2>/dev/null)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR"
[ "$(sha256sum $MODELDIR/$SIDE 2>/dev/null | cut -c1-64)" = "$SIDESHA" ] || HMISS="$HMISS sha256:$MODELDIR/$SIDE(want ${SIDESHA:0:16})"
grep -q 'XS2_ENVFILE' $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no env_file)"
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config refuses it)"
HMISS=$(echo $HMISS)
[ -z "$HMISS" ] || { xlog "VIRHE helpers / inputs do not resolve: $HMISS"; exit 1; }
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing, loadid.py $ST, mbt.py $MST, pskld.py $PST, e1q.py $EST, deepq.py = pskld.py's deep request ($DQP), all $(echo $FNS | wc -w) functions resolve, tools on PATH, sidecars $SIDECAR + $SIDE sha256 OK, yaml accepted -- OK"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { xlog "VIRHE $M225 missing or not verified"; exit 1; }
  for f in $RD/dsd-d0-gs-c1.jsonl $RM/deep-dsm-225.txt $WARM_REF_225 $RS4/s4-n5-stats0.json; do [ -s $f ] || xlog "note -- reference $f missing (not compared)"; done
  xlog "$ITEM: helpers md5: e1q.py $(md5sum < $EQ | cut -c1-8), code-body.txt $(md5sum < $CODE | cut -c1-8), pskld.py $(md5sum < $PL | cut -c1-8), mbt.py $(md5sum < $MBT | cut -c1-8), loadid.py $(md5sum < $LD | cut -c1-8), deepq.py $(md5sum < $DQ | cut -c1-8), greedyset.py $(md5sum < $GS | cut -c1-8), mb.py $(md5sum < $MBP | cut -c1-8)"
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDE $RD/dsd-d0-gs-c1.jsonl $RM/deep-dsm-225.txt $WARM_REF_225 $RS4/s4-n5-stats0.json $RS4/guard-s4-n5.txt; do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  echo "+ stack4 S 136k answer (cross-start image check, written by stack4's prefill phase): $(ls $RS4/s4-n5-d135000-*S.txt 2>/dev/null | head -1 || true)"
  for a in A E; do echo "+ arm $a: $(pinj $a)"; done
  echo "+ loaded arms: + {\"sched_bi_chunk\": $LBIC}"
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(information here; the real run refuses to start beside one)"
fi
mkd $O $O/guard $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt
if stopfile; then :; elif p0sel; then dstart; else xlog "$ITEM: VIRHE P0 failed -- the server part is not run"; fi
summary
unset $SWITCHES
down
xlog "$ITEM: done -- $O"
