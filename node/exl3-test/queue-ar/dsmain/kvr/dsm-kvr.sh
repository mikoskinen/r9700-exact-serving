#!/bin/bash
# dsm-kvr.sh (12.11.2026, queue-ar/dsmain/kvr, ds-main-port kvr builder) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r00zzz-cz-x-kvr.sh. Log tag "RESULT xkvr ", out runs/xdsm/kvr (XKVR_OUT). Made from COPIES of
# stack4/dsm-stack4.sh (21ce8c8e: the start words, LITS / FIX / the S arm, stackenv, modes, identw, cmpfile / idrec / cmpx,
# the alone identity, the image / helper / sidecar / yaml checks), stack4/s4blocks.py (0a2cca92), stack4/mbt.py (94db30da),
# stack4/editprobe.py (96a6b775: s4blocks.py takes its path), stack4/loadid.py (ee7d9663) and dsmain/resume_ho.py (b30dab40:
# the cold / hit driver with the /sessions tier counters), all copied beside this file; the originals are not changed.
# + kvrsum.py (the tier lines, the hit's counters, RADIANCE_DEBUG_TIER_VERIFY's lines, the report).
# LEVER KVR: RADIANCE_DS41_KV_REPLICATED=1 (a start-time env of the kvr image, read at declare; a fn4 switch cannot change it:
#   the host slot's layout and the host arena are fixed at bringup) declares deepseek41's prefix-cached groups kv_index_r1/_r2
#   and kv_compressed_r1/_r2 rad_kv_group_replicated: a host-tier slot holds them once (rank 0 writes, every rank reads it
#   back) instead of once a rank. Storage only: the program, the ops and every rank's own VRAM copy are unchanged. TP4 today:
#   21.56 KiB a 2-token entry = 4 x 5.39 KiB; under KVR 5.39 KiB, so the 84 GiB tier holds 32.7M tokens instead of 8.17M.
#   Declared only when no rank is attention-zero (aw == world; 4 of 4 at TP4). The window (kv_window), the DSpark ring
#   (kv_dspark) and the compressor tail (kv_comp_tail_r2, linear) are not in a tier entry (the window and the tail travel in
#   each rank's checkpoint slot, the ring in none) and keep their layout.
# STARTS (XKVR_ORDER, default "A B V"; each a separate start of stack4's NST 5 start, runs/ds41-exl3.log 2026-11-11 21:51:34
#   "RESULT xs4 s4-n5: start", on the kvr image, stack4's S pins):
#     A  KVR off (RADIANCE_DS41_KV_REPLICATED unset = the int7 image's slots, byte for byte)
#     B  KVR on  (RADIANCE_DS41_KV_REPLICATED=1)
#     V  KVR on + RADIANCE_DEBUG_TIER_VERIFY=1 (every follower hashes the replicated bytes it would have written and the
#        leader compares them with its window; a leader-side round-trip hash of every slot) -- no speed readings: the hashes
#        run on the engine thread
#   Per start: gate (restart + warm-up + fastcheck), modes (stack4's words + the KVR / verify words read back from the
#   container), the switches at start, warm-up answers vs stock, the S pin read back ->
#   (b) CAPACITY: the start log's tier lines (the transfer entry, the tier's entries / tokens / slot, the 'held once a slot'
#       line, deepseek41's 'lever KVR declared' line) + /sessions occupancy (host_cap_bytes; /stats carries no tier figure);
#   (a) A HIT FROM THE HOST TIER vs the cold run of the same text: prefix-cache reset, a cold nonce prompt (deepq's text at
#       XKVR_TDEPTH 32000, XKVR_TTOK 256 greedy tokens, logprobs on; resume_ho.py), then GET /sessions every 2 s (at most
#       XKVR_DROPWAIT 60 s) until moved.dropped has grown by 90 % of the cold's entries (its copies landed and its VRAM was
#       given up to the expert loan: the 10.11. resume runs saw a 136k chain promoted whole on its hit); if not, XKVR_FILL
#       distinct nonce prompts at the same depth push it out and the poll runs again; then the same prompt again -> its
#       window's promoted entries and restores (/sessions), cached tokens, the text / token ids / logprobs vs the cold;
#       host bytes an entry = /sessions occupancy grown / entries copied;
#   (c) SPEED (A and B only; V's hashes cost engine time): s4blocks.py --phases c1,c8 at T 0 (mbt.py), XKVR_BLOCKS blocks of
#       the 1u stepprobe essay / code and mb c8 dur XKVR_DUR 60 warm XKVR_WARM 15 under the S pin;
#   (d) IDENTITY ALONE: greedyset.py's 20 prompts at c1 alone, T 0, 192 tok, S pin (prefix-cache reset first).
#   COMPARES (strict, S:): a1 of every start vs every earlier one and vs bz d0 and stack4's a1-S; the hit vs the cold of its
#   start; the cold and the hit vs the first start's (KVR moves no arithmetic).
# GUARD (VIRHE, no hiding): the env read back from the container (A: no KVR word; B / V: RADIANCE_DS41_KV_REPLICATED=1;
#   V: + RADIANCE_DEBUG_TIER_VERIFY=1); the tier lines (A: one window a rank, the slot 4 rank entries; B / V: the four groups
#   named replicated and held once, the slot quartered to 1 rank entry, the host tier's tokens x4 against the 'rather than'
#   figure and A's); the hit's /sessions counters moved (promoted >= 1, restores >= 1, cached > 0); V: no 'differ between
#   rank', no 'came back different', no refused verify scratch; the S pin read back. Speed is cross-start (A vs B), labelled.
# ~33 min (estimate; XKVR_BUDGET_MIN 45 -- a start that would end past it is skipped and logged): A / B ~11.5 each (start +
#   check ~5, modes / capacity 0.5, tier hit ~1.5, speed 2 blocks ~3.7, a1 ~1, compares / capture ~0.5), V ~7.5; + preflight /
#   report ~1. Disk guard XKVR_DISKMIN_MB 11000 before each start. ABBA across starts costs two more starts (XKVR_ORDER="A B B A V",
#   ~56 min: raise XKVR_BUDGET_MIN).
# NO cc_sim, no evict, no power / host / fan settings; production compose files read only (--project-directory). Exact.
# Knobs: XKVR_IMAGE, XKVR_ITEM, XKVR_OUT, XKVR_ORDER ("A B V"), XKVR_BLOCKS (2), XKVR_DUR (60), XKVR_WARM (15), XKVR_TDEPTH (32000),
#   XKVR_TTOK (256), XKVR_DROPWAIT (60), XKVR_FILL (2), XKVR_DISKMIN_MB (11000), XKVR_BUDGET_MIN (45), XKVR_SIDECAR, XKVR_SIDESHA.
# DRYRUN=1 bash dsm-kvr.sh : prints every command; checks the image literals (stack4's + lever KVR's + the core's tier lines),
#   every helper (files, parse, selftests), the sidecars, the yaml, the disk; no server, no GPU, no request to :8090, writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/../ds41-stack2-tp4.yaml                   # stack2's yaml, read only (env_file ${XS2_ENVFILE} carries every start word)
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh
O=${XKVR_OUT:-$R/xdsm/kvr}
RS4=$R/xdsm/stack4; RD=$R/xdsm/draft; RM=$R/xdsm
export IMAGE=${XKVR_IMAGE:?set XKVR_IMAGE}
ITEM=${XKVR_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-x-kvr}
ORDER=${XKVR_ORDER:-A B V}; NBLK=${XKVR_BLOCKS:-2}; DUR=${XKVR_DUR:-60}; WARMS=${XKVR_WARM:-15}
TDEP=${XKVR_TDEPTH:-32000}; TTOK=${XKVR_TTOK:-256}; DROPWAIT=${XKVR_DROPWAIT:-60}; NFILL=${XKVR_FILL:-2}
DISKMIN=${XKVR_DISKMIN_MB:-11000}; BUDGET=${XKVR_BUDGET_MIN:-45}
SIDE=${XKVR_SIDECAR:-dspark-dlw-q8_0.rad}; SIDESHA=${XKVR_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
for v in $NBLK $DUR $WARMS $TDEP $TTOK $DROPWAIT $NFILL $DISKMIN $BUDGET; do case $v in ''|*[!0-9]*) echo "a numeric knob is not a number ('$v')"; exit 1 ;; esac; done
for s in $ORDER; do case $s in A|B|V) ;; *) echo "XKVR_ORDER takes A (KVR off), B (KVR on), V (KVR on + TIER_VERIFY) (got '$s')"; exit 1 ;; esac; done
[ -n "$ORDER" ] || { echo "XKVR_ORDER is empty"; exit 1; }
[ $WARMS -lt $DUR ] || { echo "the warm-up is not shorter than the phase"; exit 1; }
[ $NBLK -ge 1 ] && [ $TDEP -ge 4096 ] && [ $TTOK -ge 16 ] || { echo "XKVR_BLOCKS >= 1, XKVR_TDEPTH >= 4096, XKVR_TTOK >= 16"; exit 1; }
GGB=$HD/s4blocks.py; MBT=$HD/mbt.py; EP=$HD/editprobe.py; LD=$HD/loadid.py; RH=$HD/resume_ho.py; SUM=$HD/kvrsum.py
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
MBP=$D/mb.py; TESTS=$D/ds41_tests.py; SPB=$D/stepprobe.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
MT=192; NONCE=kvr1; T0=$(date +%s); NV=0
xlog() { case "$*" in *VIRHE*) NV=$((NV + 1)) ;; esac; log "RESULT xkvr $*"; }
# ---------------------------------------------------------------- copied from dsm-stack4.sh (21ce8c8e; tag xs4 -> xkvr)
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
LITS="$LITS|lever,PSK,declared,(RADIANCE_DS41_PSK"
# + this item's: the tier lines every start prints (the capacity parse reads them; their KVR shape is kvrsum.py cap's)
LITS="$LITS|kv,tier,transfer:,4,paged,group(s)|kv,tiers:,a,finished,conversation's,KV,is,copied,to,host,memory,at,once,(84.00,GiB"
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": 8, "drafter_mx": 1, "gg_pf": 1, "f32_pf": 25, "dspark_walk": 1, "ds41_argather": 48, "idx_split": 1, "ds41_qnfold": 1, "ds41_swfold": 1, "draft_ctxskip": 1, "dspark_conf": 1, "dspark_conf_rows": 1, "draft_lend": 1, "draft_probe_backoff": 3, "draft_hybrid": 1}'
FIX=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update({"draft_lend_hyst": 0, "arena_d0": 0, "draft_lend_w": 0, "draft_occ_k": 32}); print(json.dumps(d))' "$FIX5") || { echo "pin JSON"; exit 1; }
# stack4's S arm at its run's values (XS4_PSK 2, XS4_PSKK 4, XS4_LF sl -> place_lf 0, XS4_OCC 6, XS4_DSA 85)
SPIN='{"ds41_bi": 2, "draft_sample": 85, "draft_occ": 6, "place_lf": 0, "ds41_psk": 2, "ds41_psk_k": 4}'
armj() { echo "$SPIN"; }
pinj() {   # pinj S [EXTRAJSON ...]: the fixed pin + the S arm + the extras as one POST body (a later one wins)
  python3 -c 'import json,sys
d = {}
for a in sys.argv[1:]: d.update(json.loads(a or "{}"))
print(json.dumps(d))' "$FIX" "$SPIN" "${@:2}"
}
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
  RADIANCE_DS41_PSK RADIANCE_PLACE_LF RADIANCE_PLACE_LF_DECAY RADIANCE_LOG_STEPS RADIANCE_DS41_KV_REPLICATED RADIANCE_DEBUG_TIER_VERIFY"
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
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN | gzip > $O/engine-$1.log.gz"; return 0; }; docker logs $CN 2>&1 | gzip -1 > $O/engine-$1.log.gz || true; }
thermline() { type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"; }
cmpfile() {   # cmpfile A B -> IDENTICAL (n chars) | DIFFERENT at char k (na vs nb chars) | not compared
  [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
idrec() {   # idrec NAME RESULT [WHAT]: identity.txt + one RESULT line (a difference reported, not hidden); NAME S: strict, I: information
  [ "$DRY" = 1 ] && { echo "+ identity $1 ${3:+($3)}"; return 0; }
  echo "$1 $2" >> $O/identity.txt
  case $2 in IDENTICAL*) xlog "identity $1${3:+ ($3)}: $2" ;; *) xlog "identity $1${3:+ ($3)}: $2 -- VIRHE reported, not hidden" ;; esac
}
cmpx() {   # cmpx NAME A B: loadid.py cmp (byte for byte + the first differing char / token) -> identity.txt
  local r
  [ "$DRY" = 1 ] && { echo "+ python3 -I $LD cmp --url $URL $2 $3 --json $O/cmp-${1#*:}.json -> identity $1"; return 0; }
  if [ ! -s "$2" ] || [ ! -s "$3" ]; then xlog "$1: not compared (missing $2 or $3)"; echo "$1 not compared" >> $O/identity.txt; return 0; fi
  r=$(timeout 300 python3 -I -B $LD cmp --url $URL --json $O/cmp-${1#*:}.json --label "$1" "$2" "$3" 2>&1 | grep -m1 '^CMPLINE ' | cut -c9-)
  idrec "$1" "${r:-VIRHE compare failed}" "$(basename $2) vs $(basename $3)"
}
rhcmp() {   # rhcmp NAME A B [LP]: resume_ho.py cmp -- the text, the token ids, the logprobs -> identity.txt. LP 1 (the hit vs its
  # cold, the prefix-resume contract): IDENTICAL only with every logprob equal too; LP 0 (across starts): the text decides, the
  # logprobs are said (a cross-start logprob difference with the same text is information, not a KVR finding: no restore is in a cold)
  local r lp=${4:-1}
  [ "$DRY" = 1 ] && { echo "+ python3 -I -B $RH cmp $2 $3 -> identity $1 (text + token ids$([ $lp = 1 ] && echo " + every logprob") decide)"; return 0; }
  if [ ! -s "$2" ] || [ ! -s "$3" ]; then xlog "$1: not compared (missing $2 or $3)"; echo "$1 not compared" >> $O/identity.txt; return 0; fi
  r=$(cd $D && timeout 120 python3 -I -B $RH cmp "$2" "$3" 2>&1 | grep -m1 '^CMP ' | cut -c5- | python3 -c 'import json,sys
d, lp = json.load(sys.stdin), sys.argv[1] == "1"
txt = d.get("text") or "not compared"
tok = "not compared" if "first_token_diff" not in d else "identical" if d["first_token_diff"] is None else "first differ at %s" % d["first_token_diff"]
if "first_lp_diff_at" in d:
    lps = ("equal on all %s tokens" % d.get("lp_equal_tokens_before")) if d["first_lp_diff_at"] is None else \
          ("first differ at token %s (token 0 |diff| %s, max %s before the first differing token)" % (d["first_lp_diff_at"], d.get("lp_diff_tok0"), d.get("lp_diff_max_before")))
else:
    lps = d.get("lp") or "not compared"
same = txt.startswith("IDENTICAL") and tok in ("identical", "not compared")
if lp and same and lps.startswith("first differ"): print("LOGPROBS DIFFER (%s) with the text %s" % (lps, txt))
else: print("%s; token ids %s; logprobs %s" % (txt, tok, lps))' $lp 2>&1 | cut -c1-500)
  idrec "$1" "${r:-VIRHE compare failed}" "$(basename $2) vs $(basename $3)$([ $lp = 1 ] && echo ", logprobs strict")"
}
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xkvr $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { xlog "$1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  echo "S:$1-warmup $r" >> $O/identity.txt
  case $r in IDENTICAL*) xlog "$1 warm-up answers vs stock ($(basename $2)): $r" ;; *) xlog "$1 warm-up answers vs stock: VIRHE $r" ;; esac
}
modes() {   # modes TAG MODE: the start words + this mode's KVR / verify words in the container (and the others NOT in it), mover 0 kept
  # free x4, rule-w, K1, ARGATHER 48, every literal of LITS -> 0, else 1
  local tag=$1 md=$2 want="" nowant="" CE kv bad="" ok=1 NW NR KL AL nz lit found="" miss=""
  case $md in A) nowant="RADIANCE_DS41_KV_REPLICATED RADIANCE_DEBUG_TIER_VERIFY" ;; B) want="RADIANCE_DS41_KV_REPLICATED=1"; nowant="RADIANCE_DEBUG_TIER_VERIFY" ;;
    V) want="RADIANCE_DS41_KV_REPLICATED=1 RADIANCE_DEBUG_TIER_VERIFY=1" ;; esac
  [ "$DRY" = 1 ] && { echo "+ modes $tag ($md): LEVENV$([ -n "$want" ] && echo " + $want") in the container's env$([ -n "$nowant" ] && echo ", NOT $nowant"); mover '0 more kept free' on 4 ranks; rule-w in every recorded pass; K1 folds =1; ARGATHER 48-row note; literals [$(echo "$LITS" | tr ',' ' ')]"; return 0; }
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  echo "$CE" | grep -E '^(RADIANCE_|R4D_)' | grep -vE '(KEY|TOKEN|SECRET)' > $O/$tag-cenv.txt
  for kv in RADIANCE_PLACE_TURNOVER=0 $LEVENV $want; do echo "$CE" | grep -qxF "$kv" || bad="$bad ${kv%%=*}"; done
  for kv in $nowant; do echo "$CE" | grep -q "^$kv=" && bad="$bad $kv(set-in-$md)"; done
  [ -z "$bad" ] || { xlog "$tag: VIRHE start words not as asked:$bad"; ok=0; }
  nz=$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots \([0-9]+ shadow, 0 more kept free')
  [ "$nz" = 4 ] || { xlog "$tag: VIRHE turnover 0 asked but $nz of 4 ranks keep 0 slots free"; ok=0; }
  NW=$(docker logs $CN 2>&1 | grep -c 'the acquire rule plays .*RADIANCE_AQL_ACQ=rule-w)'); NR=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_AQL_ACQ=rule)')
  [ "$NW" -gt 0 ] && [ "$NR" = 0 ] || { xlog "$tag: VIRHE rule-w is not what the recorded passes play ($NW / $NR)"; ok=0; }
  KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-500)
  case "$KL" in *"ds41_kvfold=1"*"ds41_ofold=1"*"ds41_onefill=1"*) ;; *) xlog "$tag: VIRHE the K1 switches do not start at 1: '${KL}'"; ok=0 ;; esac
  AL=$(docker logs $CN 2>&1 | grep -m1 'the MoE gather and its all-reduce as one kernel' | sed 's/^radiance: //' | cut -c1-300)
  case "$AL" in *"up to 48 tokens a step (RADIANCE_DS41_ARGATHER, RADIANCE_DS41_ARGATHER_MAX"*) ;; *) xlog "$tag: VIRHE the ARGATHER note is not the 48-row one: '${AL}'"; ok=0 ;; esac
  IFS='|' read -ra lit <<< "$LITS"
  for l in "${lit[@]}"; do [ -n "$l" ] || continue
    if docker logs $CN 2>&1 | grep -qF -- "${l//,/ }"; then found="$found ${l%%,*}"; else miss="$miss | ${l//,/ }"; fi; done
  xlog "$tag modes ($md): ${#lit[@]} literal(s), found [$(echo $found)]$([ -n "$miss" ] && echo " -- VIRHE missing:$miss") | KVR / verify words in the container: $(grep -E 'KV_REPLICATED|TIER_VERIFY' $O/$tag-cenv.txt | tr '\n' ' ')| $(docker logs $CN 2>&1 | grep -E "lever KVR|kv tier transfer|kv tiers: |lever PSK declared|lever BI declared" | sed 's/^radiance: //; s/^[A-Z] *//; s/^[A-Z]\[[0-9]\] //' | cut -c1-330 | sort -u | tr '\n' '|' | cut -c1-2600)"
  [ -z "$miss" ] || ok=0
  echo "$tag modes $([ $ok = 1 ] && echo OK || echo VIRHE) ($md)" >> $O/guard.txt
  [ $ok = 1 ] || keeplog $tag-nomode
  [ $ok = 1 ]
}
# ---------------------------------------------------------------- copied from dsm-kv4.sh's copies of stack4 (the read-back, disk, snapshots)
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
fn4code() { [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
arm() { fn4set "$(pinj S '{"epoch_bump": 1}')"; nudge; }
ARMOK=1
armset() {   # armset TAG: the pin + S + epoch_bump, a nudge, every switch READ BACK (else VIRHE, ARMOK 0)
  local g r
  arm
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4: every switch of $(pinj S) applied, nothing pending -> RESULT xkvr $1 arm S applied (read back)"; return 0; }
  g=$(fn4get)
  r=$(python3 -c 'import json,sys
g, w = json.loads(sys.argv[1]), json.loads(sys.argv[2])
bad = {k: g["switches"].get(k) for k, v in w.items() if g["switches"].get(k) != v}
s = g["switches"]
print(("OK" if not bad and not g["pending"] else "FAIL %s pending %s" % (bad, g["pending"])) + " | bi=%s psk=%s/%s occ=%s ds=%s lf=%s adapt=%s d0=%s epoch=%s" % (
    s.get("ds41_bi"), s.get("ds41_psk"), s.get("ds41_psk_k"), s.get("draft_occ"), s.get("draft_sample"), s.get("place_lf"), s.get("draft_adaptive"), s.get("draft_depth0"), g.get("epoch")))' "$g" "$(pinj S)" 2>&1 | cut -c1-400)
  case "$r" in OK*) xlog "$1 arm S applied (read back): $r"; return 0 ;; esac
  xlog "$1 arm S: VIRHE the pin is not applied ($r) -- no reading under it"; echo "$1 pins VIRHE $r" >> $O/guard.txt; ARMOK=0; return 1
}
gate1() {   # restart + ONE check; over the limit logged (a slow state shows in the 1-user ms of every start)
  local tag=$1 lim=$2 ms st=fast
  restart $tag || { keeplog $tag-fail; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && return 0
  awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}' && st="SLOW (not retried)"
  echo "$st $ms" > $O/guard-$tag.txt
  xlog "guard $tag: $st, 1-user ${ms} ms/step (limit $lim; stack4 s4-n5 $(cat $RS4/guard-s4-n5.txt 2>/dev/null))"
}
dfree() { df -BM --output=avail $HOME 2>/dev/null | tail -1 | tr -dc 0-9; }
diskok() {   # diskok TAG: >= DISKMIN MB free on / before a start
  local f; f=$(dfree)
  if [ "$DRY" = 1 ]; then echo "+ disk before $1: ${f:-?} MB free on / (a start needs >= $DISKMIN MB) -- $([ "${f:-0}" -ge $DISKMIN ] && echo OK now || echo "LOW now: the run would refuse this start")"; return 0; fi
  [ "${f:-0}" -ge $DISKMIN ] && return 0
  xlog "$1: VIRHE only ${f:-?} MB free on / (need >= $DISKMIN: the 10 GB guard + this item's logs) -- not started"; return 1
}
envfile() { local f=$1; shift; if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $* > $f"; else printf '%s\n' "$@" > $f; fi; }
envline() { env | grep -E '^(RADIANCE_|R4D_)' | grep -vE '(KEY|TOKEN|SECRET)' | sort | tr '\n' ' ' | cut -c1-2400; }
drafted() {   # drafted TAG: the DSpark drafter and the adaptive depth are up (stack4's check)
  [ "$DRY" = 1 ] && return 0
  local ok=1
  docker logs $CN 2>&1 | grep -q "DSpark drafter" || { xlog "$1: VIRHE no DSpark drafter line"; ok=0; }
  docker logs $CN 2>&1 | grep -q "draft depth: adaptive" || { xlog "$1: VIRHE no adaptive draft depth line"; ok=0; }
  [ $ok = 1 ]
}
pyx() {   # pyx NAME CMD...: one parser run, its last line logged (VIRHE when it fails)
  local nm=$1 rc; shift
  [ "$DRY" = 1 ] && { echo "+ $* > $O/$nm.out"; return 0; }
  timeout 900 "$@" > $O/$nm.out 2>&1; rc=$?
  xlog "$nm: $(tail -1 $O/$nm.out | cut -c1-3500)$([ $rc = 0 ] || echo " -- VIRHE rc $rc")"
}
pcreset() { [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o $O/$1-reset.json -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  xlog "$1 prefix cache reset: HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"; }
sess() { [ "$DRY" = 1 ] && { echo "+ GET /sessions > $O/$1.json"; return 0; }; curl -s -m 30 $URL/sessions > $O/$1.json 2>/dev/null; }
capture() {   # capture FILE: the engine log, gzipped (never written raw)
  run "docker logs $CN 2>&1 | gzip -1 > $1"
  [ "$DRY" = 1 ] || xlog "engine log $(basename $1): $(du -m $1 | cut -f1) MB gz, $(zcat $1 | wc -l) lines; $(dfree) MB free"
}
swstart() {   # swstart TAG: the switches at start (stack4's start values; every switch the S pin names present); ds41_psk 99 refused
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 at start: gg_gemv 2, aql_acq 2, K1 three 1, epoch 0, draft_capable 1, ds41_bi 0, ds41_psk 0, every switch of $(pinj S) present; POST ds41_psk 99 -> 400"; return 0; }
  local s0 c1 r
  s0=$(fn4get); c1=$(fn4code '{"ds41_psk": 99}')
  r=$(python3 -c 'import json,sys
s0 = json.loads(sys.argv[1]); w = s0["switches"]; want = json.loads(sys.argv[3])
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) and s0["epoch"] == 0
ok = ok and s0["draft_capable"] == 1 and w.get("ds41_bi") == 0 and w.get("ds41_psk") == 0
miss = [k for k in want if k not in w]
ok = ok and not miss and sys.argv[2] == "400"
print(("OK" if ok else "FAIL") + " ds41_bi %s ds41_psk %s/%s draft_occ %s draft_sample %s | ds41_psk 99 -> HTTP %s%s" % (
    w.get("ds41_bi"), w.get("ds41_psk"), w.get("ds41_psk_k"), w.get("draft_occ"), w.get("draft_sample"), sys.argv[2],
    (" | not in the registry: %s" % miss) if miss else ""))' "$s0" "$c1" "$(pinj S)" 2>&1 | cut -c1-500)
  case "$r" in OK*) xlog "$1 switches at start: $r"; return 0 ;; esac
  xlog "$1 switches at start: VIRHE $r | $(echo "$s0" | cut -c1-900)"; return 1
}
# ---------------------------------------------------------------- lever KVR's readings
startenv() {   # startenv TAG MODE: stack4's dstart serve line + this mode's start words in the env file
  local tag=$1 md=$2 kw=""
  stackenv
  case $md in B) kw="RADIANCE_DS41_KV_REPLICATED=1" ;; V) kw="RADIANCE_DS41_KV_REPLICATED=1 RADIANCE_DEBUG_TIER_VERIFY=1" ;; esac
  export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on"
  export XS2_ENVFILE=$O/$tag.env STATSDIR=$O/stats-$tag
  envfile $XS2_ENVFILE $LEVENV $kw
  mkd $STATSDIR
  KW=$kw
}
capx() {   # capx TAG MODE: (b) the start log's tier lines + /sessions occupancy + the container env -> cap-TAG.json, the guard
  local tag=$1 md=$2 r
  sess $tag-ses-start
  if [ "$DRY" = 1 ]; then echo "+ python3 -I -B $SUM cap --log $O/start-$tag.log --sessions $O/$tag-ses-start.json --env $O/$tag-cenv.txt --tag $tag --mode $md$([ -s $O/cap-A.json ] && echo " --ref $O/cap-A.json") --json $O/cap-$tag.json -> RESULT xkvr cap-$tag (A: one window a rank, 4 rank entries a slot; B / V: the four groups held once, the slot quartered, tokens x4, the env word read back)"; return 0; fi
  r=$(timeout 120 python3 -I -B $SUM cap --log $O/start-$tag.log --sessions $O/$tag-ses-start.json --env $O/$tag-cenv.txt --tag $tag --mode $md \
        $([ -s $O/cap-A.json ] && [ $tag != A ] && echo "--ref $O/cap-A.json") --json $O/cap-$tag.json 2>&1 | tail -1 | cut -c1-1800)
  case "$r" in "CAP OK"*) xlog "cap-$tag: $r"; echo "cap $tag OK" >> $O/guard.txt ;;
    *) xlog "cap-$tag: VIRHE $r"; echo "cap $tag VIRHE ${r:0:300}" >> $O/guard.txt ;; esac
}
thrun() {   # thrun TAG NAME NONCE NTOK: resume_ho.py run (deepq's text at TDEP behind '# nonce NONCE', greedy, logprobs top 2)
  local tag=$1 nm=$2
  run "(cd $D && timeout 900 python3 -I $RH run $O/$tag-$nm.txt nonce --nonce $3 --depth $TDEP --tokens $4 --top 2 > $O/$tag-$nm.log 2>&1)"
  [ "$DRY" = 1 ] || xlog "$tag $nm (nonce $3, deepq depth $TDEP, $4 greedy tokens): $(grep -m1 '^RHO ' $O/$tag-$nm.log | cut -c5- | python3 -c 'import json,sys
d = json.load(sys.stdin); a, b = d.get("tiers_before") or {}, d.get("tiers_after") or {}
dl = lambda k: (b.get(k) - a.get(k)) if isinstance(a.get(k), (int, float)) and isinstance(b.get(k), (int, float)) else "?"
print("status %s, prompt %s tok, cached %s, TTFT %s s, %s tok, md5 %s, lp %s | tier: promoted +%s, restores +%s (+%s ms), to_host +%s, dropped +%s" % (
    d.get("status"), d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("tokens"), d.get("md5"), d.get("lp"),
    dl("promoted"), dl("restores"), dl("restore_ms"), dl("to_host"), dl("dropped")))' 2>&1 | cut -c1-500)"
}
dropped_since() {   # dropped_since A.json B.json -> moved.dropped grown between two /sessions snapshots
  python3 -c 'import json,sys
def g(p):
    try: return (json.load(open(p)).get("moved") or {}).get("dropped") or 0
    except Exception: return 0
print(g(sys.argv[2]) - g(sys.argv[1]))' "$1" "$2" 2>/dev/null || echo 0
}
dropwait() {   # dropwait TAG NEED -> 0 when moved.dropped grew by >= NEED since TAG-th-s0 within DROPWAIT s (GET /sessions every 2 s)
  local tag=$1 need=$2 k=0 dd=0
  [ "$DRY" = 1 ] && { echo "+ GET /sessions every 2 s, at most $DROPWAIT s, until moved.dropped grew by >= 90 % of the cold's entries"; return 0; }
  while :; do
    curl -s -m 10 $URL/sessions > $O/$tag-th-s1.json 2>/dev/null
    dd=$(dropped_since $O/$tag-th-s0.json $O/$tag-th-s1.json)
    [ "${dd:-0}" -ge $need ] && break
    [ $k -ge $DROPWAIT ] && break
    sleep 2; k=$((k + 2))
  done
  xlog "$tag tier hit: $dd entries dropped from VRAM after $k s (want >= $need)"
  [ "${dd:-0}" -ge $need ]
}
tierhit() {   # tierhit TAG: (a) a cold nonce prompt, its VRAM copies off the card, the same prompt again from the host tier
  local tag=$1 need f r
  armset $tag || return 1
  pcreset $tag-th
  sess $tag-th-s0
  thrun $tag th-cold $NONCE $TTOK
  if [ "$DRY" = 1 ]; then need=1
  else
    need=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(max(1, int(0.9 * (d.get("prompt_tokens") or 0) / 2)))' $O/$tag-th-cold.txt.json 2>/dev/null || echo 1)
  fi
  if ! dropwait $tag $need; then
    for f in $(seq 1 $NFILL); do thrun $tag th-fill$f kvr-fill$f 16; done
    dropwait $tag $need || xlog "$tag tier hit: the cold's entries are still in VRAM after $NFILL filler prompts -- the hit below may come from VRAM (the guard decides)"
  fi
  thrun $tag th-hit $NONCE $TTOK
  sess $tag-th-s2
  if [ "$DRY" = 1 ]; then echo "+ python3 -I -B $SUM hit --cold $O/$tag-th-cold.txt.json --hit $O/$tag-th-hit.txt.json --s0 / --s1 / --s2 $O/$tag-th-s{0,1,2}.json -> RESULT xkvr hit-$tag (promoted >= 1, restores >= 1, cached > 0; host bytes an entry)"
  else
    r=$(timeout 120 python3 -I -B $SUM hit --cold $O/$tag-th-cold.txt.json --hit $O/$tag-th-hit.txt.json --s0 $O/$tag-th-s0.json --s1 $O/$tag-th-s1.json \
          --s2 $O/$tag-th-s2.json --tag $tag --json $O/hit-$tag.json 2>&1 | tail -1 | cut -c1-1200)
    case "$r" in "HIT OK"*) xlog "hit-$tag: $r"; echo "hit $tag OK" >> $O/guard.txt ;;
      *) xlog "hit-$tag: VIRHE $r"; echo "hit $tag VIRHE ${r:0:300}" >> $O/guard.txt ;; esac
  fi
  rhcmp S:th-hit-vs-cold-$tag $O/$tag-th-cold.txt $O/$tag-th-hit.txt 1
}
speedx() {   # speedx TAG: (c) NBLK blocks of the 1u stepprobe essay / code + mb c8 at T 0 under S (s4blocks.py, one arm)
  local tag=$1 ord t00=$(date +%s)
  ord=$(printf 'S%.0s' $(seq 1 $NBLK))
  run "timeout 5400 python3 -I $GGB --url $URL --mb-dir $D --edit $EP --label $tag-sp --arm 'S=$SPIN' --order $ord --phases c1,c8 --dur $DUR --warm $WARMS --mbtemp 0 --mbt $MBT --fixed '$FIX' --out $O/$tag-sp-blocks.jsonl > $O/$tag-sp-blocks.txt 2>&1"
  [ "$DRY" = 1 ] && return 0
  while IFS= read -r l; do xlog "$tag sp ${l:0:1500}"; done < <(grep -E '^(BLOCK|GGB \{"error)|Traceback' $O/$tag-sp-blocks.txt)
  xlog "$tag sp: $(grep -c . $O/$tag-sp-blocks.jsonl 2>/dev/null) block(s) in $(( $(date +%s) - t00 )) s | $(thermline $t00 $(date +%s))"
}
alone() {   # alone TAG: (d) the S pin read back, prefix-cache reset, greedyset.py's 20 prompts at c1 ALONE, temperature 0
  local tag=$1 t00=$(date +%s)
  armset $tag || return 1
  pcreset $tag-a1
  run "(cd $D && timeout 900 python3 $GS $URL $O/$tag-a1.jsonl --conc 1 --max-tokens $MT > $O/$tag-a1.txt 2>&1)"
  [ "$DRY" = 1 ] || xlog "$tag a1 (greedy set c1 ALONE, T 0, $MT tok, arm S): $(tail -1 $O/$tag-a1.txt | cut -c1-200), $(( $(date +%s) - t00 )) s"
}
verifyx() {   # verifyx TAG: V's RADIANCE_DEBUG_TIER_VERIFY lines over the whole engine log
  local r
  [ "$DRY" = 1 ] && { echo "+ python3 -I -B $SUM verify --log $O/$1-engine.log.gz -> 0 'differ between rank', 0 'came back different', no refused scratch"; return 0; }
  r=$(timeout 300 python3 -I -B $SUM verify --log $O/$1-engine.log.gz --tag $1 2>&1 | tail -1 | cut -c1-600)
  case "$r" in "VERIFY OK"*) xlog "verify-$1: $r | entries copied under the check: $(python3 -c 'import json,sys
def g(p):
    try: return (json.load(open(p)).get("moved") or {}).get("to_host")
    except Exception: return None
print(g(sys.argv[1]))' $O/$1-ses-end.json 2>/dev/null) (each hashed by the 3 followers against rank 0's window)"; echo "verify $1 OK" >> $O/guard.txt ;;
    *) xlog "verify-$1: VIRHE $r"; echo "verify $1 VIRHE ${r:0:300}" >> $O/guard.txt ;; esac
}
idstart() {   # idstart TAG: this start's compares while its server is up (loadid.py cmp tokenizes the first differing token)
  local t=$1 x
  cmpx S:a1-$t-vs-d0 $RD/dsd-d0-gs-c1.jsonl $O/$t-a1.jsonl
  cmpx S:a1-$t-vs-stack4-a1S $RS4/s4-n5-a1-S.jsonl $O/$t-a1.jsonl
  for x in $DONE; do cmpx S:a1-$t-vs-a1-$x $O/$x-a1.jsonl $O/$t-a1.jsonl; done
  if [ -n "$REF" ]; then
    rhcmp S:th-cold-$t-vs-th-cold-$REF $O/$REF-th-cold.txt $O/$t-th-cold.txt 0
    rhcmp S:th-hit-$t-vs-th-hit-$REF $O/$REF-th-hit.txt $O/$t-th-hit.txt 0
  fi
  return 0
}
est() { case $1 in V) echo 8 ;; *) echo $(( 8 + (NBLK * (DUR + WARMS + 35) + 59) / 60 )) ;; esac; }
DONE=""; REF=""
kstart() {   # kstart TAG: one start (TAG = its mode, A / B / V; a repeated mode is TAG2, TAG3 ...), its readings, its compares
  local md=$1 tag=$1 n=2 e el
  while case " $DONE " in *" $tag "*) true ;; *) false ;; esac; do tag=$md$n; n=$((n + 1)); done
  e=$(est $md); el=$(( ($(date +%s) - T0) / 60 ))
  if [ "$DRY" != 1 ] && [ $((el + e)) -gt $BUDGET ]; then xlog "$tag SKIPPED: $el min elapsed + ~$e min > the budget $BUDGET min"; return 0; fi
  diskok $tag || return 1
  startenv $tag $md
  xlog "$tag: start (~$e min; mode $md: $(case $md in A) echo "KVR off" ;; B) echo "KVR on" ;; V) echo "KVR on + RADIANCE_DEBUG_TIER_VERIFY" ;; esac)${KW:+, env file + $KW}; stack4's NST 5 start) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $FIX + S $SPIN | env: $(envline)"
  gate1 $tag 70 || { xlog "$tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  modes $tag $md || { xlog "$tag: VIRHE the modes are not the start's -> stopped"; down; return 1; }
  drafted $tag || { keeplog $tag-nodraft; down; return 1; }
  swstart $tag || { keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  armset $tag || { keeplog $tag-pin; down; return 1; }
  capx $tag $md
  stopfile || tierhit $tag
  [ $md = V ] || { stopfile || speedx $tag; }
  stopfile || alone $tag
  idstart $tag
  sess $tag-ses-end
  [ "$DRY" = 1 ] || xlog "$tag: $(docker logs $CN 2>&1 | grep -cE '^E ') E lines | /sessions at the end: $(python3 -c 'import json,sys
d = json.load(open(sys.argv[1])); m, o = d.get("moved") or {}, d.get("occupancy") or {}
print("to_host %s dropped %s promoted %s host_drops %s failures %s | host %s of %s bytes" % (m.get("to_host"), m.get("dropped"), m.get("promoted"),
      m.get("host_drops"), m.get("failures"), o.get("host_bytes"), o.get("host_cap_bytes")))' $O/$tag-ses-end.json 2>&1 | cut -c1-300)"
  capture $O/$tag-engine.log.gz
  [ $md = V ] && verifyx $tag
  down
  [ -n "$REF" ] || REF=$tag
  DONE="$DONE $tag"
  xlog "$tag: done ($(( ($(date +%s) - T0) / 60 )) min into the item)"
}
report() {
  run "python3 -I -B $SUM report --dir $O --starts '$DONE' > $O/report.txt 2>&1"
  [ "$DRY" = 1 ] && { echo "+ every RESULT line of $O/report.txt -> the log (capacity per start and vs A, the tier hit per start, speed per start and B vs A cross-start, identity, guards)"; return 0; }
  while IFS= read -r l; do case "$l" in "RESULT xkvr "*) log "${l:0:3800}" ;; esac; done < $O/report.txt
  local nv; nv=$(( $(grep -c VIRHE $O/report.txt 2>/dev/null) + NV ))
  [ $ARMOK = 1 ] || nv=$((nv + 1))
  log "RESULT xkvr guards (item): $([ "$nv" = 0 ] && echo "OK (env read back, tier lines as the mode asks, hits from the host tier, pins read back$(case " $DONE " in *" V"*) echo ", TIER_VERIFY clean" ;; esac))" || echo "VIRHE in $nv place(s) -- see the lines above; a reading with VIRHE is not a reading") | starts done:$DONE of [$ORDER]"
}
trap 'down' EXIT
trap 'exit 143' TERM INT
# ---------------------------------------------------------------- main
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) | lever KVR (RADIANCE_DS41_KV_REPLICATED=1: deepseek41's cached groups held once a host-tier slot) | starts [$ORDER] (A off, B on, V on + RADIANCE_DEBUG_TIER_VERIFY) of stack4's NST 5 start | S pin $SPIN | tier hit nonce $NONCE at depth $TDEP, $TTOK tok, drop wait $DROPWAIT s, $NFILL filler(s) | speed (A / B) $NBLK block(s) of 1u essay / code + mb c8 T 0 dur $DUR warm $WARMS | a1 identity | ~33 min (budget $BUDGET)"
IMGLITS="B|dspark_conf_rows, gg_pf, f32_pf, dspark_walk, ds41_bi, draft_sample, draft_lend_hyst, arena_d0, draft_lend_w, draft_occ, draft_occ_k, place_lf, sched_bi_chunk, sched_interleave, sched_mixed_tok,
B|ds41_psk
B|ds41_psk_k
B|reset_prefix_cache
B|/tokenize
B|/sessions
A|RADIANCE_DS41_PSK
A|lever PSK declared (RADIANCE_DS41_PSK, fn4 ds41_psk, now
B|place_lf
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
B|draft_occ takes 0 (off) .. 64
B|RADIANCE_DRAFT_LEND_WEIGHTS
B|lever Z0-A set up
B|RADIANCE_DRAFT_CTXSKIP_SLOTS
A|RADIANCE_DS41_KV_REPLICATED
A|lever KVR declared (RADIANCE_DS41_KV_REPLICATED=1): %s hold the same bytes on
A|lever KVR not declared (RADIANCE_DS41_KV_REPLICATED=1)
A|lever KVR: %s replicated on every rank
B|is the same on every rank and held once a slot: %s a slot rather than %s
B|of it replicated on every rank and held once a slot)
B|kv tier transfer: %zu paged group(s), %s an entry%s%s%s%s, batches of %d
B|copied to host memory at once
B|RADIANCE_DEBUG_TIER_VERIFY
B|differ between rank 0
B|came back different
B|no pinned host memory (%s) for this
B|host_cap_bytes
B|promoted"
docker image inspect $IMAGE > /dev/null 2>&1 || { xlog "VIRHE image $IMAGE not on this host"; exit 1; }
IMGMISS=$(echo "$IMGLITS" | timeout 120 docker run --rm -i --network none --entrypoint sh $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels/libr4d.so; S=/opt/radiance/share/radiance/kernels/spechub.so; n=0
while IFS= read -r l; do [ -n "$l" ] || continue; n=$((n + 1)); w=${l%%|*}; s=${l#*|}
  case $w in B) f=$B ;; A) f=$A ;; K) f=$K ;; S) f=$S ;; *) f="$B $A $K" ;; esac
  o=0; for x in $f; do grep -qzF -- "$s" $x && { o=1; break; }; done
  [ $o = 1 ] || echo "MISS $w|$s"; done
test -x /opt/radiance/bin/rad-sidecar || echo "MISS bin/rad-sidecar"; test -s /opt/radiance/share/radiance/int7-PROVENANCE.txt || echo "MISS int7-PROVENANCE.txt"
test -s /opt/radiance/share/radiance/kvr-PROVENANCE.txt || echo "MISS kvr-PROVENANCE.txt"; echo "CHECKED $n"' 2>&1)
case "$IMGMISS" in "CHECKED "*) ;; *) xlog "VIRHE $IMAGE is not the kvr image -- missing: $(echo "$IMGMISS" | grep -v '^CHECKED' | tr '\n' ';' | cut -c1-900)"; exit 1 ;; esac
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)) is the kvr image: all $(echo "$IMGMISS" | sed 's/CHECKED //') literals present (stack4's registry / BI / SD / DLW / Z0 / PSK, lever KVR's env and declare lines, the core's replicated tier lines and TIER_VERIFY lines, /sessions' counters) + bin/rad-sidecar + int7- and kvr-PROVENANCE -- OK"
HMISS=""
for f in "$LIB29" "$LIBPR" $Y $GS $GGB $MBT $EP $LD $RH $SUM $MBP $TESTS $SPB $D/fastcheck.py; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $GS $GGB $MBT $EP $LD $RH $SUM $MBP $TESTS $SPB $D/fastcheck.py 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
ST1=$(timeout 60 python3 -I -B $LD selftest --gs $GS --mb $MBP 2>&1 | tail -1); case "$ST1" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$LD(${ST1:0:300})" ;; esac
ST2=$(timeout 60 python3 -I -B $MBT --mb $MBP --selftest 2>&1 | tail -1); case "$ST2" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$MBT(${ST2:0:300})" ;; esac
ST3=$(timeout 60 python3 -I -B $SUM selftest 2>&1 | tail -1); case "$ST3" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$SUM(${ST3:0:300})" ;; esac
python3 -I -B $GGB --help > /dev/null 2>&1 || HMISS="$HMISS run:s4blocks.py--help"
# resume_ho.py builds the nonce prompt offline (deepq's text at TDEP behind '# nonce NONCE'), with run / cmp present
RHP=$(cd $D && timeout 60 python3 -I -B -c 'import importlib.util, sys
s = importlib.util.spec_from_file_location("rh", sys.argv[1]); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
nonce = sys.argv[3]
sys.argv = ["rh", "run", "x", "nonce", "--nonce", nonce, "--depth", sys.argv[2]]
p = m.prompt("nonce"); c = p[0]["content"]
ok = c.startswith("# nonce %s\n# deepq fixed prompt v1\n" % nonce) and all(callable(getattr(m, f, None)) for f in ("cmd_run", "cmd_cmp", "tiers", "chat_lp"))
print("RH OK %d chars" % len(c) if ok else "RH MISMATCH")' $RH $TDEP $NONCE 2>&1 | tail -1)
case "$RHP" in "RH OK"*) ;; *) HMISS="$HMISS resume_ho:(${RHP:0:200})" ;; esac
FNS="log run mkd restart check down stopfile best_vars desc serve_args xlog armj pinj stackenv keeplog thermline cmpfile idrec cmpx rhcmp identw modes fn4get fn4set fn4code nudge arm armset gate1 dfree diskok envfile envline drafted pyx pcreset sess capture swstart startenv capx thrun dropped_since dropwait tierhit speedx alone verifyx idstart est kstart report"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee sha256sum gzip zcat df du seq; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
python3 -c 'import json,sys; d = json.loads(sys.argv[1]); s = json.loads(sys.argv[2]); sys.exit(0 if d["gg_gemv"] == 2 and d["epoch_bump"] == 1 and all(d[k] == v for k, v in s.items()) else 1)' "$(pinj S '{"epoch_bump": 1}')" "$SPIN" 2>/dev/null || HMISS="$HMISS json:S"
[ "$(sha256sum $MODELDIR/$SIDECAR 2>/dev/null | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256 2>/dev/null)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR"
[ "$(sha256sum $MODELDIR/$SIDE 2>/dev/null | cut -c1-64)" = "$SIDESHA" ] || HMISS="$HMISS sha256:$MODELDIR/$SIDE(want ${SIDESHA:0:16})"
for w in XS2_ENVFILE '${STATSDIR:?set STATSDIR}:/stats'; do grep -qF "$w" $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no '$w')"; done
grep -qE 'RADIANCE_DS41_KV_REPLICATED|RADIANCE_DEBUG_TIER_VERIFY' $Y 2>/dev/null && HMISS="$HMISS yaml:$Y(names a KVR word: the env file alone must carry it)"
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$O/stats ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config refuses it)"
HMISS=$(echo $HMISS)
[ -z "$HMISS" ] || { xlog "VIRHE helpers / inputs do not resolve: $HMISS"; exit 1; }
HMD5="s4blocks.py $(md5sum < $GGB | cut -c1-8) (stack4's $(md5sum < $HD/../stack4/s4blocks.py 2>/dev/null | cut -c1-8)), mbt.py $(md5sum < $MBT | cut -c1-8) (stack4's $(md5sum < $HD/../stack4/mbt.py 2>/dev/null | cut -c1-8)), editprobe.py $(md5sum < $EP | cut -c1-8), loadid.py $(md5sum < $LD | cut -c1-8) (stack4's $(md5sum < $HD/../stack4/loadid.py 2>/dev/null | cut -c1-8)), resume_ho.py $(md5sum < $RH | cut -c1-8) (dsmain's $(md5sum < $HD/../resume_ho.py 2>/dev/null | cut -c1-8)), kvrsum.py $(md5sum < $SUM | cut -c1-8), greedyset.py $(md5sum < $GS | cut -c1-8), stepprobe.py $(md5sum < $SPB | cut -c1-8), mb.py $(md5sum < $MBP | cut -c1-8), lib-29.sh $(md5sum < $LIB29 | cut -c1-8), yaml $(md5sum < $Y | cut -c1-8)"
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing; loadid.py $ST1 | mbt.py $ST2 | kvrsum.py $ST3 | s4blocks.py --help | resume_ho.py $RHP (nonce $NONCE at depth $TDEP) | all $(echo $FNS | wc -w) functions resolve, tools on PATH, sidecars $SIDECAR + $SIDE sha256 OK, yaml accepted (env_file, /stats mount, no KVR word of its own) -- OK | md5: $HMD5"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { xlog "VIRHE $M225 missing or not verified"; exit 1; }
  for f in $WARM_REF_225 $RD/dsd-d0-gs-c1.jsonl $RS4/s4-n5-a1-S.jsonl $RS4/guard-s4-n5.txt; do [ -s $f ] || xlog "note -- reference $f missing (not compared)"; done
  xlog "$ITEM: helpers md5: $HMD5"
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDE $WARM_REF_225 $RD/dsd-d0-gs-c1.jsonl $RS4/s4-n5-a1-S.jsonl $RS4/guard-s4-n5.txt; do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  echo "+ arm S: $SPIN | POST body $(pinj S '{"epoch_bump": 1}')"
  for s in $ORDER; do echo "+ start $s: $(case $s in A) echo "KVR off: env file = LEVENV" ;; B) echo "KVR on: env file = LEVENV + RADIANCE_DS41_KV_REPLICATED=1" ;; V) echo "KVR on + verify: env file = LEVENV + RADIANCE_DS41_KV_REPLICATED=1 RADIANCE_DEBUG_TIER_VERIFY=1 (no speed blocks)" ;; esac) (~$(est $s) min)"; done
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(information here; the real run refuses to start beside one)"
fi
mkd $O $O/guard
[ "$DRY" = 1 ] || { : > $O/identity.txt; : > $O/guard.txt; }
for s in $ORDER; do stopfile && break; kstart $s; done
report
unset $SWITCHES
down
xlog "$ITEM: done in $(( ($(date +%s) - T0) / 60 )) min -- $O ($(dfree) MB free)"
