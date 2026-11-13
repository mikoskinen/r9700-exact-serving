#!/bin/bash
# dsm-pskl20.sh (12.11.2026, queue-ar/dsmain/pskl20, ds-main-port PSK-L20/DS builder) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r00zzz-cz-pskl20.sh. Log tag "RESULT xpl20 ", out runs/xdsm/pskl20 (XPL_OUT). Copied from dsm-stack4.sh
# (21ce8c8e: the start, LEVENV, the fixed pin, modes / swnote / smoke / deepx / loadx / alone / cmpx / msnap / mdelta / ssnap) and
# dsm-psk2b.sh (421f3636: qx on p2bq.py's own texts, cachedof, the hit design); neither file is edited.
# WHY (ARCH-AUDIT-20261112 levers 1 + 2, the coordinator 12.11.): two EXACT levers in one image (ds41-pskl20 = ds41-int7-pskglue
# 5093ce9 + both levers, fn4 registry 42 -> 44 switches, defaults 0 = today):
#   L  = ds41_psk_l20 1 (PSK-L20): layer 20 split -- collapse + norm, compressor, latent / index-key stores over every row; query,
#        indexer, attention, MoE, window store over C_20 = 2592 rows (the decoder's global KV comes from layer 20's input).
#        Estimate -1.5..-2.5 s at 136k cold (K4 33.25 s -> ~31 s); planned chunks with no layer-20 cone row stage none of layer
#        20's experts.
#   D  = dspark_pfctx 1 (PSK-DS): a DSpark context pass runs each entry's last window + 1 (129) rows only (runs >= 160 rows):
#        estimate -0.3..-0.45 s at 136k, and the c8 newcomer step (lever NC) lighter.
#   LD = both. B = neither (= stack4's S arm).
# ONE NST 5 START = stack4's start exactly (runs/ds41-exl3.log 2026-11-11 21:51:34 "RESULT xs4 s4-n5: start": stack3 S env +
#   z0occ2's Z0 words + RADIANCE_DS41_PSK=0, MNBT 4160, --live --draft-window adaptive --draft-coupling on) + RADIANCE_LOG_STEPS=1;
#   the engine log captured by docker logs -t (gzipped). Arms in process (POST /admin/fn4: stack4's fixed pin + S's pins
#   {ds41_bi 2, draft_sample 85, draft_occ 6, place_lf 0, ds41_psk 2, ds41_psk_k 4} + the arm's two switches + epoch_bump, a
#   nudge, the applied switches READ BACK every time).
# ORDER: gate1 -> modes (stack4's literals + the PSK-L20 declare line) -> VRAM -> switch note (start values 0; ds41_psk_l20 2 /
#   dspark_pfctx 2 refused; B / L / D / LD applied) -> warm-up answers -> smoke under LD -> COUNTER GUARD (a cold ~16k prompt a
#   arm, each on its own text: design 2's latched / planned_steps / ckpt_dropped / layers_skipped grow and plan_mismatch does not in
#   every arm; L / LD move l20_cut_passes + l20_rows_skipped, B / D no l20 counter; D / LD move dspark_pfctx passes + rows_skipped,
#   B / L none. A failure = VIRHE and no speed reading) -> DRAFTER CHECK (a deterministic greedy NST 5 pin: draft_adaptive 0,
#   draft_depth0 0, draft_occ 0, draft_hybrid 0, dspark_conf 0; 3 prompts of 2k / 8k / 16k, 256 tokens, each cold, arms B D LD LD D
#   B: every run's answer and its drafted / accepted counts equal the first B run's = the drafter's readable keys unchanged) ->
#   MB c8 ABBA (stack4's: mbt.py temperature 0, dur 60 warm 15; B L D LD LD D L B; agg_dec_tps, decode_share, and from the step log
#   the newcomer steps' wall -- a step carrying a fresh ~1k prompt's prefill -- per block) -> COLD TTFT ABBA at 136k / 64k / 16k
#   (B L D LD LD D L B, every run a FRESH DISTINCT text: p2bq.py's offsets, a reset before each, 64 greedy tokens; /stats deltas a
#   run) -> IDENTITY: alone greedy set c1 a arm (vs B; B vs bz d0 information); deepq's fixed 136k prompt a arm, 512 tokens (vs B
#   and vs dsm-225); c8 newcomers = the greedy set at conc 8 a arm (vs B alone); loaded = c4 + a cold 64k probe a arm (pskld.py
#   --load; the probe's chunks from the step log: vs loaded B strict when the boundaries match, else class (ii) information at
#   sched_bi_chunk 0); HITS (psk2b's corrected design: L / D / LD each on its OWN text -- reset, cold 64k, hits at 40000 / 20000
#   and the 64k tail with no reset, the cached count must be the anchor below (K4: 30720 / 14336 / the tail snapshot); then B on
#   that text after a reset each: cold 64k / 40000 / 20000 = the references) -> SUMMARY.
# ~57 min (estimate): start + checks + smoke ~9, guard ~2, drafter check ~3, mb c8 8 x ~85 s ~11.5, TTFT 24 runs ~9, alone 4 x
#   ~1 ~4, fixed 136k 4 x ~45 s ~3, conc 8 4 x ~45 s ~3, loaded 4 x ~65 s ~4.5, hits 3 x ~85 s ~4.5, log + summary ~1.5.
#   Cut XPL_MORDER to BLDXXDLB's half (BLDX) first (-6 min) if a shorter run is wanted.
# NO cc_sim, no evict, no power / host / fan settings; production compose files read only (--project-directory). Exact by design
#   (both levers): bytes measured and reported, a difference is reported, never hidden.
# Disk: >= XPL_DISKMIN_MB (11000) free on / before the start (the 10 GB guard; docker's json log of a RADIANCE_LOG_STEPS start lives
#   until the container is removed); engine logs written gzipped only.
# Knobs: XPL_IMAGE, XPL_ITEM, XPL_OUT, XPL_PSKK (4), XPL_OCC (6), XPL_OCCK (32), XPL_DSA (85), XPL_TORDER / XPL_MORDER (BLDXXDLB),
#   XPL_AORDER (BDXXDB), XPL_DEPTHS ("135000 64000 16000"), XPL_TNTOK (64), XPL_GDEP (16000), XPL_LDEPTH (64000), XPL_LNTOK (128),
#   XPL_HDEP (64000), XPL_HITS ("40000 20000"), XPL_ADEPS ("2000 8000 16000"), XPL_ANTOK (256), XPL_DUR (60), XPL_WARM (15),
#   XPL_DISKMIN_MB (11000), XPL_SIDECAR, XPL_SIDESHA.
# DRYRUN=1 bash dsm-pskl20.sh : prints every command; checks the image literals (the 44-switch registry, PSK-L20 / PSK-DS's
#   literals and exports), every helper (files, parse, selftests), the text offsets, the sidecars, the yaml; no server, no GPU,
#   writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/../ds41-stack2-tp4.yaml                   # stack2's yaml, read only (env_file ${XS2_ENVFILE} carries every start word)
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh
O=${XPL_OUT:-$R/xdsm/pskl20}; export STATSDIR=$O/stats
RD=$R/xdsm/draft; RM=$R/xdsm; RS4=$R/xdsm/stack4
export IMAGE=${XPL_IMAGE:?set XPL_IMAGE}
ITEM=${XPL_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-pskl20}
PSKK=${XPL_PSKK:-4}; OCC=${XPL_OCC:-6}; OCCK=${XPL_OCCK:-32}; DSA=${XPL_DSA:-85}
TORDER=${XPL_TORDER:-BLDXXDLB}; MORDER=${XPL_MORDER-BLDXXDLB}; AORDER=${XPL_AORDER-BDXXDB}
DEPTHS=${XPL_DEPTHS:-135000 64000 16000}; TNTOK=${XPL_TNTOK:-64}; GDEP=${XPL_GDEP:-16000}; GNTOK=64
FDEP=135000; NTOK=512; LDEPTH=${XPL_LDEPTH:-64000}; LNTOK=${XPL_LNTOK:-128}
HDEP=${XPL_HDEP:-64000}; HITS=${XPL_HITS:-40000 20000}; HNTOK=64; ADEPS=${XPL_ADEPS:-2000 8000 16000}; ANTOK=${XPL_ANTOK:-256}
DUR=${XPL_DUR:-60}; WARMS=${XPL_WARM:-15}; DISKMIN=${XPL_DISKMIN_MB:-11000}
SIDE=${XPL_SIDECAR:-dspark-dlw-q8_0.rad}; SIDESHA=${XPL_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
MT=192; SEED=1234
# the text offsets (ds41_tests.SRC file index, < its 624 files; p2bq.py --doff): guard 10.., drafter check 20.., TTFT 30.. (8 a
# depth, 24), hits 300.. (one a hit arm); deepq.py's fixed prompt is offset 1200 = file 0
GOFF=10; AOFF=20; TOFF=30; HOFF=300
case $TORDER in *[!BLDX]*|"") echo "XPL_TORDER takes B / L / D / X (= LD) (got '$TORDER')"; exit 1 ;; esac
case $MORDER in *[!BLDX]*) echo "XPL_MORDER takes B / L / D / X or nothing (got '$MORDER')"; exit 1 ;; esac
case $AORDER in *[!BDX]*) echo "XPL_AORDER takes B / D / X or nothing (got '$AORDER')"; exit 1 ;; esac
case $AORDER in ""|B*) ;; *) echo "XPL_AORDER must start with B (the drafter check's reference run)"; exit 1 ;; esac
for v in $PSKK $OCC $OCCK $DSA $DEPTHS $TNTOK $GDEP $LDEPTH $LNTOK $HDEP $HITS $ADEPS $ANTOK $DUR $WARMS $DISKMIN; do
  case $v in ''|*[!0-9]*) echo "a numeric knob is not a number ('$v')"; exit 1 ;; esac; done
[ ${#TORDER} -le 8 ] || { echo "XPL_TORDER: at most 8 runs a depth (8 text offsets each)"; exit 1; }
for h in $HITS; do [ $h -lt $HDEP ] || { echo "XPL_HITS must be below XPL_HDEP"; exit 1; }; done
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
LD=$HD/loadid.py; MBT=$HD/mbt.py; PL=$HD/pskld.py; P2=$HD/p2bq.py; PX=$HD/pl20.py; NCS=$HD/../nc/ncsteps.py
DQ=$QA/deepq.py; MBP=$D/mb.py; TESTS=$D/ds41_tests.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
xlog() { log "RESULT xpl20 $*"; }
# ---------------------------------------------------------------- the start (= dsm-stack4.sh's + RADIANCE_LOG_STEPS=1)
GPTAB=q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422
SIDECAR=dspark-conf.f32
LEVENV="RADIANCE_DS41_GG_PF=1 RADIANCE_DS41_GG_PF_TABLE=$GPTAB RADIANCE_DS41_F32_PF=25 RADIANCE_DS41_ARENA_DECODE_ROWS=8,48
  RADIANCE_DS41_ROUTER_BF16=1 RADIANCE_DS41_EMB_RAM=1 RADIANCE_PLACE_SHADOW_SLOTS=16 RADIANCE_XPREFETCH=0 RADIANCE_DS41_DSPARK_WALK=1
  RADIANCE_DS41_ARGATHER_MAX=48 RADIANCE_DS41_QNFOLD=1 RADIANCE_DS41_SWFOLD=1 RADIANCE_DRAFT_CTXSKIP=0 RADIANCE_DS41_DSPARK_CONF=/models/$SIDECAR
  RADIANCE_DRAFT_LEND=0 RADIANCE_DRAFT_PROBE_BACKOFF=0 RADIANCE_DRAFT_LEND_BORROW=0 RADIANCE_DRAFT_SAMPLE=0 RADIANCE_WEIGHT_SIDECAR=/models/$SIDE
  RADIANCE_DS41_BI=0 RADIANCE_DRAFT_CTXSKIP_SLOTS=129 RADIANCE_DRAFT_LEND_WEIGHTS=0 RADIANCE_DS41_PSK=0"
LEVENV=$(echo $LEVENV)
STARTENV="$LEVENV RADIANCE_LOG_STEPS=1"
LITS="gemm_nt_ggml's,prefill,rows,take,lever,GPF's,forms,(RADIANCE_DS41_GG_PF=1|gemm_nt_f32's,prefill,rows,take,lever,GPF-b's,tiled,form,(RADIANCE_DS41_F32_PF=25"
LITS="$LITS|48,tokens,use|a,rank),ram|expert,prefetch:,off"
LITS="$LITS|DSpark,lever,W1:,the,full-vocabulary,Markov,walk,declared,(dspark_walk_step,a,step,+,all_gather,over,4,rank(s)|fn4,dspark_walk=1,"
LITS="$LITS|one,kernel,up,to,48,tokens,a,step|RADIANCE_DS41_QNFOLD;,ds41_qnfold=1)|RADIANCE_DS41_SWFOLD;,ds41_swfold=1)|draft,context,skip,(RADIANCE_DRAFT_CTXSKIP=0|a,confidence,plane,(the,dspark_conf,gate"
LITS="$LITS|draft,lend:,set,up"
LITS="$LITS|DSpark,lever,SD:,the,sampled,full-vocabulary,walk,declared,(dspark_walk_sample_step,a,step,over,4,rank(s)|lever,SD:,fn4,draft_sample,0,at,start"
LITS="$LITS|weight,sidecar:,43,entries|read,from,the,weight,sidecar|lever,BI,declared,(RADIANCE_DS41_BI|now,0):,a,step,of,more"
LITS="$LITS|lever,Z0-A,set,up|activation,arena:,8,tokens,use|ring,of,129,steps|draft,weight,lend,(lever,Z0-W|mover:,lever,Z0-W:"
LITS="$LITS|lever,PSK,declared,(RADIANCE_DS41_PSK"
LITS="$LITS|now,0;,RADIANCE_DS41_PSK_L20):,layer,20,split,under,lever,PSK"   # the new declare line (lever PSK-L20 (fn4 ds41_psk_l20, ...)
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": 8, "drafter_mx": 1, "gg_pf": 1, "f32_pf": 25, "dspark_walk": 1, "ds41_argather": 48, "idx_split": 1, "ds41_qnfold": 1, "ds41_swfold": 1, "draft_ctxskip": 1, "dspark_conf": 1, "dspark_conf_rows": 1, "draft_lend": 1, "draft_probe_backoff": 3, "draft_hybrid": 1}'
FIX=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update({"draft_lend_hyst": 0, "arena_d0": 0, "draft_lend_w": 0, "draft_occ_k": int(sys.argv[2])}); print(json.dumps(d))' "$FIX5" "$OCCK") || { echo "pin JSON"; exit 1; }
SPIN="\"ds41_bi\": 2, \"draft_sample\": $DSA, \"draft_occ\": $OCC, \"place_lf\": 0, \"ds41_psk\": 2, \"ds41_psk_k\": $PSKK"
# the drafter check's deterministic overlay: a fixed depth (5), no depth 0 / occupancy / lookup hybrid / confidence cut
DET='{"draft_adaptive": 0, "draft_depth0": 0, "draft_occ": 0, "draft_hybrid": 0, "dspark_conf": 0}'
armj() {   # armj B|L|D|X: S's pins + the arm's two switches (JSON)
  case $1 in
    B) echo "{$SPIN, \"ds41_psk_l20\": 0, \"dspark_pfctx\": 0}" ;;
    L) echo "{$SPIN, \"ds41_psk_l20\": 1, \"dspark_pfctx\": 0}" ;;
    D) echo "{$SPIN, \"ds41_psk_l20\": 0, \"dspark_pfctx\": 1}" ;;
    X) echo "{$SPIN, \"ds41_psk_l20\": 1, \"dspark_pfctx\": 1}" ;;
  esac
}
armn() { case $1 in B) echo B ;; L) echo L ;; D) echo D ;; X) echo LD ;; esac; }
pinj() {   # pinj ARM [EXTRAJSON]: the fixed pin + the arm + extra as one POST body
  local x=${2:-}; [ -n "$x" ] || x='{}'
  python3 -c 'import json,sys
d = {}
for a in sys.argv[1:]: d.update(json.loads(a))
print(json.dumps(d))' "$FIX" "$(armj $1)" "$x"
}
DESC="NST 5 stack4 start (stack3 S: stack2 NST 5 env, DSpark adaptive, depth 0, hybrid 1, W1, small n5 set, drafter_mx, lend + back-off 3, VRAM levers, --draft-coupling on, DLW sidecar $SIDE; z0occ2's ring 129, Z0-W declared, rungs 8,48; PSK declared off) + RADIANCE_LOG_STEPS=1 | fixed pin: stack4's (stack3's + Z0 switches 0, draft_occ_k $OCCK) | S pins: ds41_bi 2, draft_sample $DSA, draft_occ $OCC, place_lf 0, ds41_psk 2 (k $PSKK) | B: ds41_psk_l20 0 dspark_pfctx 0, L: 1 0, D: 0 1, LD: 1 1"
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
  RADIANCE_DS41_PSK RADIANCE_PLACE_LF RADIANCE_PLACE_LF_DECAY RADIANCE_LOG_STEPS RADIANCE_DS41_PSK_L20 RADIANCE_DS41_DSPARK_PFCTX"
stackenv() {   # 105's t3 arm + gg_gemv 2 + rule-w + the K1 folds + turnover 0 + LEVENV, every other switch unset (stack4's)
  local kv
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  export RADIANCE_PLACE_TURNOVER=0
  for kv in $STARTENV; do export "$kv"; done
}
dfree() { df -Pm / | awk 'NR==2{print $4}'; }
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN | gzip > $O/engine-$1.log.gz"; return 0; }; docker logs $CN 2>&1 | gzip -1 > $O/engine-$1.log.gz || true; }
capture() {   # capture FILE: the engine log with the daemon's receipt times, gzipped (never written raw)
  run "docker logs -t $CN 2>&1 | gzip -1 > $1"
  [ "$DRY" = 1 ] || xlog "engine log $(basename $1): $(du -m $1 | cut -f1) MB gz, $(zcat $1 | grep -cE ' step [0-9]+ (prefill|decode|MIXED) n_seq=') step lines; $(dfree) MB free"
}
thermline() { type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"; }
nowt() { date +%s.%N; }
cmpfile() {   # cmpfile A B -> IDENTICAL (n chars) | DIFFERENT at char k (na vs nb chars) | not compared
  [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
idrec() {   # idrec NAME RESULT [WHAT]: identity.txt + one RESULT line (a difference reported, not hidden)
  [ "$DRY" = 1 ] && { echo "+ identity $1${3:+ ($3)}"; return 0; }
  echo "$1 $2" >> $O/identity.txt
  case $2 in IDENTICAL*) xlog "identity $1${3:+ ($3)}: $2" ;; *) xlog "identity $1${3:+ ($3)}: $2 -- reported, not hidden" ;; esac
}
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference (stack4's)
  [ "$DRY" = 1 ] && { echo "+ RESULT xpl20 $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { xlog "$1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  echo "info-warm $r" >> $O/identity.txt
  case $r in IDENTICAL*) xlog "$1 warm-up answers vs stock ($(basename $2)): $r" ;; *) xlog "$1 warm-up answers vs stock: VIRHE $r" ;; esac
}
modes() {   # modes TAG: the start words in the container, mover 0 kept free x4, rule-w, K1, ARGATHER 48, every literal of LITS (stack4's)
  [ "$DRY" = 1 ] && { echo "+ modes $1: STARTENV in the container's env; mover '0 more kept free' on 4 ranks; rule-w in every recorded pass; K1 folds declared =1; ARGATHER 48-row note; literals [$(echo "$LITS" | tr ',' ' ')]"; return 0; }
  local CE kv bad="" ok=1 NW NR KL AL nz lit found="" miss=""
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in RADIANCE_PLACE_TURNOVER=0 $STARTENV; do echo "$CE" | grep -qxF "$kv" || bad="$bad ${kv%%=*}"; done
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
  [ "$(docker logs $CN 2>&1 | grep -cE ' step [0-9]+ (prefill|decode|MIXED) n_seq=')" -gt 0 ] || { xlog "$1: VIRHE RADIANCE_LOG_STEPS=1 but no step line in the log"; ok=0; }
  xlog "$1 modes: ${#lit[@]} literal(s), found [$(echo $found)]$([ -n "$miss" ] && echo " -- VIRHE missing:$miss") | $(docker logs $CN 2>&1 | grep -E "lever PSK|lever BI declared|lever SD|weight sidecar|lever Z0|activation arena|ring of|draft lend:|draft weight lend|no level for|draft floor|draft refill" | sed 's/^radiance: //; s/^[A-Z] *//' | cut -c1-300 | sort -u | tr '\n' '|' | cut -c1-3200)"
  [ -z "$miss" ] || ok=0
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
vramstart() {
  [ "$DRY" = 1 ] && { echo "+ GET /stats > $O/$1-stats0.json; RESULT xpl20 $1 VRAM at start (experts per tier; vs stack4 s4-n5)"; return 0; }
  curl -s -m 30 $URL/stats > $O/$1-stats0.json 2>/dev/null
  xlog "$1 VRAM at start: experts $(python3 -c 'import json,sys
for p in sys.argv[1:]:
    try: s = json.load(open(p)); print({x.get("name"): x.get("units") for x in s.get("experts", {}).get("tiers", [])}, end=" ")
    except Exception as e: print("n/a", end=" ")' $O/$1-stats0.json $RS4/s4-n5-stats0.json 2>&1 | cut -c1-320) (this start, then stack4 s4-n5) | /stats psk + dspark_pfctx at start: $(python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); print(json.dumps(s.get("psk")), json.dumps(s.get("dspark_pfctx")))' $O/$1-stats0.json 2>&1 | cut -c1-900)"
}
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
fn4code() { [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null; }
pcreset() { [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o $O/$1-reset.json -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  [ "$c" = 200 ] || xlog "$1 prefix cache reset: VIRHE HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"; }
ARMBAD=0
arm() {   # arm B|L|D|X [EXTRAJSON]: the pin + the arm (+ extra) + epoch_bump, a nudge to apply it at a step, the switches READ BACK
  local want g
  want=$(pinj $1 "${2:-}")
  fn4set "$(pinj $1 "$(python3 -c 'import json,sys; d=json.loads(sys.argv[1] or "{}"); d["epoch_bump"]=1; print(json.dumps(d))' "${2:-}")")"; nudge
  [ "$DRY" = 1 ] && return 0
  g=$(fn4get)
  python3 -c 'import json,sys
g, w = json.loads(sys.argv[1]), json.loads(sys.argv[2])
sys.exit(0 if all(g["switches"].get(k) == v for k, v in w.items()) and not g["pending"] else 1)' "$g" "$want" 2>/dev/null && return 0
  ARMBAD=$((ARMBAD + 1))
  xlog "VIRHE arm $(armn $1)${2:+ + $2}: the applied switches are not the pin: $(echo "$g" | cut -c1-900)"
}
msnap() { [ "$DRY" = 1 ] && { echo "+ GET /metrics > $O/m-$1.txt"; return 0; }; curl -s -m 20 $URL/metrics > $O/m-$1.txt 2>/dev/null; }
mdelta() {   # mdelta A B -> JSON: draft acceptance, non-decode steps and their mean ms, decode steps, prefill tokens over the window (stack4's)
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
ssnap() { [ "$DRY" = 1 ] && { echo "+ GET /stats > $O/st-$1.json"; return 0; }; curl -s -m 30 $URL/stats > $O/st-$1.json 2>/dev/null; }
sdelta() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; python3 -I $PX sdelta $O/st-$1.json $O/st-$2.json 2>/dev/null || echo '{}'; }
pskstat() { [ "$DRY" = 1 ] && { echo "(dry)"; return 0; }; echo "$(docker logs $CN 2>&1 | grep 'lever PSK:' | tail -1 | sed 's/^radiance: deepseek41: //' | cut -c1-260) | $(docker logs $CN 2>&1 | grep 'lever PSK-DS:' | tail -1 | sed 's/^radiance: deepseek41: //' | cut -c1-200) | VIRHE lines $(docker logs $CN 2>&1 | grep -c 'lever PSK design 2: VIRHE')"; }
jmerge() { python3 -c 'import json,sys
d = {}
for a in sys.argv[1:]:
    try: d.update(json.loads(a or "{}"))
    except ValueError: pass
print(json.dumps(d))' "$@"; }
gate1() {   # restart + ONE check; over the limit logged (in-process A/B: a slow state shows in every arm) (stack4's)
  local tag=$1 lim=$2 ms st=fast
  restart $tag || { keeplog $tag-fail; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && return 0
  awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}' && { st="SLOW (not retried)"; keeplog $tag-slow; }
  echo "$st $ms" > $O/guard-$tag.txt
  xlog "guard $tag: $st, 1-user ${ms} ms/step (limit $lim; stack4 s4-n5 $(cat $RS4/guard-s4-n5.txt 2>/dev/null))"
}
# ---------------------------------------------------------------- one request, recorded
qrun() {   # qrun TAG ARM KIND NAME DEPTH NTOK DOFF RESET [EXTRAJSON]: arm, (reset), one deep request on text DOFF (p2bq.py; DOFF
  # "deepq" = deepq.py's fixed prompt), /metrics + /stats deltas -> deep.jsonl + a RESULT line
  local tag=$1 a=$2 kind=$3 nm=$4 dep=$5 nt=$6 off=$7 rs=$8 x=${9:-} t0=$(date +%s) j md sd
  arm $a "$x"
  [ "$rs" = reset ] && pcreset $tag-$nm
  msnap $tag-$nm-0; ssnap $tag-$nm-0
  if [ "$off" = deepq ]; then
    run "(cd $D && timeout 1800 python3 $DQ $O/$tag-$nm.txt $dep $nt > $O/$tag-$nm.out 2>&1)"
  else
    run "(cd $D && timeout 1800 python3 -I $P2 run --pskld $PL --url $URL --tests $TESTS --depth $dep --ntok $nt --doff $off --out $O/$tag-$nm.json --text $O/$tag-$nm.txt > $O/$tag-$nm.out 2>&1)"
  fi
  msnap $tag-$nm-1; ssnap $tag-$nm-1
  [ "$DRY" = 1 ] && { echo "+ RESULT xpl20 $tag $nm: arm $(armn $a) $kind depth $dep text $off, $nt tok$([ "$rs" = reset ] || echo ", NO reset")${x:+, $x}: TTFT, cached, /stats deltas (psk + l20 + pfctx counters, staged GB, snapshots) -> $O/deep.jsonl"; return 0; }
  if [ "$off" = deepq ]; then j=$(grep -m1 '^DEEPQ ' $O/$tag-$nm.out | cut -c7-); else j=$(grep -m1 '^LONGID ' $O/$tag-$nm.out | cut -c8-); fi
  md=$(mdelta $tag-$nm-0 $tag-$nm-1); sd=$(sdelta $tag-$nm-0 $tag-$nm-1)
  [ -n "$j" ] || { xlog "$tag $nm: VIRHE no answer ($(tail -2 $O/$tag-$nm.out | tr '\n' ' ' | cut -c1-300))"; return 1; }
  python3 -c 'import json,sys
d = {}
for x in sys.argv[1:4]: d.update(json.loads(x or "{}"))
d.update(name=sys.argv[4], arm=sys.argv[5], kind=sys.argv[6], depth=int(sys.argv[7]), doff=sys.argv[8], reset=sys.argv[9] == "reset", text=sys.argv[10])
print(json.dumps(d))' "$j" "$md" "$sd" "$nm" "$a" "$kind" "$dep" "$off" "$rs" "$O/$tag-$nm.txt" >> $O/deep.jsonl
  xlog "$tag $nm (arm $(armn $a), $kind, depth $dep text $off, $nt greedy tokens$([ "$rs" = reset ] || echo ", NO reset")${x:+, overlay $x}): $(python3 -c 'import json,sys
d, m, s = json.loads(sys.argv[1]), json.loads(sys.argv[2] or "{}"), json.loads(sys.argv[3] or "{}")
print("status %s, prompt %s tok, cached %s, TTFT %s s, %s tok, %s chars | drafted %s accepted %s | non-decode steps %s, %s ms mean | snapshots %s, staged %s GB (r0 %s, %s layers) | psk planned %s cut %s skipped-whole %s mismatch %s | L20 cut %s rows %s cone-empty %s not-staged %s | pfctx passes %s rows %s (ran %s)" % (
    d.get("status"), d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("tokens"), d.get("chars"), m.get("draft_tok"), m.get("draft_acc"),
    m.get("nondecode_steps"), m.get("nondecode_ms"), s.get("ckpts"), s.get("staged_gb"), s.get("staged_gb_r0"), s.get("stage_layers_r0"),
    s.get("psk_planned_steps"), s.get("psk_cut_passes"), s.get("psk_layers_skipped"), s.get("psk_plan_mismatch"), s.get("psk_l20_cut_passes"),
    s.get("psk_l20_rows_skipped"), s.get("psk_l20_cone_empty"), s.get("psk_l20_layers_not_staged"), s.get("pfctx_passes"), s.get("pfctx_rows_skipped"),
    s.get("pfctx_rows_run")))' "$j" "$md" "$sd" 2>&1 | cut -c1-900) | $(thermline $t0 $(date +%s))"
}
cachedof() { [ "$DRY" = 1 ] && { echo 0; return 0; }; python3 -c 'import json,sys
r = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
x = [d for d in r if d.get("name") == sys.argv[2]]
print(x[-1].get(sys.argv[3]) if x else -1)' $O/deep.jsonl "$1" "${2:-cached}" 2>/dev/null || echo -1; }
# ---------------------------------------------------------------- the start's checks
swnote() {   # swnote TAG: start values; refusals; every arm applied at a step and read back, then B
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 at start (ds41_psk_l20 0, dspark_pfctx 0, ds41_psk 0, ds41_psk_k 8, ds41_bi 0, draft_occ_k 32, epoch 0); refusals ds41_psk_l20 2 / dspark_pfctx 2 / ds41_psk 99 -> 400; arms B $(armj B) | L $(armj L) | D $(armj D) | LD $(armj X) applied + read back, then B"; return 0; }
  local s0 c1 c2 c3 a b0=$ARMBAD
  s0=$(fn4get)
  c1=$(fn4code '{"ds41_psk_l20": 2}'); c2=$(fn4code '{"dspark_pfctx": 2}'); c3=$(fn4code '{"ds41_psk": 99}')
  for a in B L D X B; do arm $a; done
  xlog "$1 switch note: at start $(echo "$s0" | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print("ds41_psk_l20 %s dspark_pfctx %s ds41_psk %s/%s ds41_bi %s draft_occ_k %s epoch %s" % (w.get("ds41_psk_l20"), w.get("dspark_pfctx"), w.get("ds41_psk"), w.get("ds41_psk_k"), w.get("ds41_bi"), w.get("draft_occ_k"), d.get("epoch")))' 2>&1 | cut -c1-300) | ds41_psk_l20 2 -> HTTP $c1, dspark_pfctx 2 -> HTTP $c2, ds41_psk 99 -> HTTP $c3 | arms B L D LD B applied, $((ARMBAD - b0)) not as pinned"
  python3 -c 'import json,sys
s0 = json.loads(sys.argv[1]); w = s0["switches"]
ok = w.get("ds41_psk_l20") == 0 and w.get("dspark_pfctx") == 0 and w.get("ds41_psk") == 0 and w.get("ds41_psk_k") == 8 and w.get("ds41_bi") == 0
ok = ok and w["gg_gemv"] == 2 and w["aql_acq"] == 2 and s0["epoch"] == 0 and w.get("draft_occ_k") == 32 and s0["draft_capable"] == 1
sys.exit(0 if ok else 1)' "$s0" 2>/dev/null && [ "$c1" = 400 ] && [ "$c2" = 400 ] && [ "$c3" = 400 ] && [ $ARMBAD = $b0 ]
}
smoke() {   # smoke TAG: under LD (both levers on): the greedy set at conc 4 (mixed steps) + mb c4 20 s at T 1 (stack4's)
  local tag=$1 t0=$(date +%s) n e
  arm X
  run "(cd $D && timeout 600 python3 $GS $URL $O/$tag-smoke.jsonl --conc 4 --max-tokens 32 > $O/$tag-smoke.txt 2>&1)"
  run "(cd $D && timeout 600 python3 mb.py --tag xpl20-smoke --phases c4 --dur 20 --warm 5 > $O/$tag-smoke-t1.txt 2>&1)"
  [ "$DRY" = 1 ] && { echo "+ smoke $tag: arm LD $(armj X): greedy set conc 4 (32 tok) 20 of 20 + mb c4 20 s at T 1 with 0 errors, /health 200, no design-2 plan-mismatch line"; return 0; }
  n=$(python3 -c 'import json,sys; print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l).get("text") is not None))' $O/$tag-smoke.jsonl 2>/dev/null || echo 0)
  e=$(grep -m1 '^MB ' $O/$tag-smoke-t1.txt 2>/dev/null | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[3:]); print("%s requests %s errors, %s tok/s" % (d.get("requests"), d.get("errors"), d.get("agg_dec_tps")))' 2>/dev/null || echo "no MB line")
  if [ "$n" = 20 ] && case "$e" in *" 0 errors"*) true ;; *) false ;; esac && curl -s -m 10 -o /dev/null -w '%{http_code}' $URL/health | grep -q 200 && \
     [ "$(docker logs $CN 2>&1 | grep -c 'lever PSK design 2: VIRHE')" = 0 ]; then
    xlog "$tag smoke (arm LD): greedy set conc 4 20 of 20, mb c4 T 1: $e, healthy, $(( $(date +%s) - t0 )) s | psk: $(pskstat)"; return 0; fi
  xlog "$tag smoke (arm LD): VIRHE $n of 20 answered, mb c4 T 1: $e, unhealthy or a design-2 VIRHE line | $(docker logs $CN 2>&1 | grep -E '^E |step fail|refus|VIRHE' | tail -3 | tr '\n' ' ' | cut -c1-500)"
  keeplog $tag-smoke; return 1
}
SPEEDOK=1
guard() {   # guard TAG: the counter guard -- a cold ~GDEP prompt a arm on its own text, /stats deltas checked (pl20.py guard)
  local tag=$1 a i=0 r
  for a in B L D X; do
    stopfile && return 0
    qrun $tag $a guard guard-$a $GDEP $GNTOK $((GOFF + i)) reset
    i=$((i + 1))
    if [ "$DRY" = 1 ]; then echo "+ guard arm $(armn $a): pl20.py guard $a <its /stats deltas> (design 2's counters grow in every arm, plan_mismatch 0; l20 counters only in L / LD; pfctx counters only in D / LD)"; continue; fi
    r=$(python3 -I $PX guard $a "$(sdelta $tag-guard-$a-0 $tag-guard-$a-1)" 2>&1 | cut -c1-900)
    echo "$(armn $a) $r" >> $O/guard.txt
    case "$r" in OK*) xlog "$tag counter guard arm $(armn $a): $r" ;;
      *) xlog "$tag counter guard arm $(armn $a): VIRHE $r -- a lever that silently does nothing (or one that moved where it must not): no speed reading"; SPEEDOK=0 ;; esac
  done
  [ $ARMBAD = 0 ] || { xlog "$tag: VIRHE $ARMBAD arm switch(es) not applied as pinned -- no speed reading"; SPEEDOK=0; }
}
acccheck() {   # acccheck TAG: the drafter's state -- deterministic greedy NST 5 pin (DET), AORDER, the same 3 texts a run
  local tag=$1 a k=0 dep i nm
  [ -n "$AORDER" ] || return 0
  for a in $(echo $AORDER | grep -o .); do
    k=$((k + 1)); i=0
    for dep in $ADEPS; do
      stopfile && return 0
      nm=acc-$k$a-$dep
      qrun $tag $a acc $nm $dep $ANTOK $((AOFF + i)) reset "$DET"
      i=$((i + 1))
      [ "$DRY" = 1 ] && continue
      python3 -c 'import json,sys
r = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
x = [d for d in r if d.get("name") == sys.argv[2]][-1]
print(json.dumps(dict(arm=sys.argv[3], run=int(sys.argv[4]), prompt=sys.argv[5], text=x.get("text"), draft_tok=x.get("draft_tok"), draft_acc=x.get("draft_acc"))))' $O/deep.jsonl $nm $a $k "d$dep" >> $O/acc.jsonl 2>/dev/null
    done
  done
  [ "$DRY" = 1 ] && { echo "+ pl20.py acc $O/acc.jsonl > $O/acc-cmp.txt: every run's answer and drafted / accepted counts vs the first B run's -> $O/acc.txt"; return 0; }
  python3 -I $PX acc $O/acc.jsonl > $O/acc-cmp.txt 2>&1
  tail -1 $O/acc-cmp.txt > $O/acc.txt
  while IFS= read -r l; do xlog "$tag drafter check: ${l:0:600}"; done < $O/acc-cmp.txt
}
mbblocks() {   # mbblocks TAG: mb c8 at temperature 0 (stack4's mbt.py, dur DUR warm WARMS), MORDER; marks for the step log
  local tag=$1 a k=0 nm t0 t1 j sd
  [ -n "$MORDER" ] || return 0
  for a in $(echo $MORDER | grep -o .); do
    k=$((k + 1)); nm=mb-$k$a
    stopfile && return 0
    arm $a
    ssnap $tag-$nm-0; t0=$(nowt)
    run "(cd $D && timeout 900 python3 -I -B $MBT --mb mb.py --temperature 0 --tag xpl20-$nm --phases c8 --dur $DUR --warm $WARMS > $O/$tag-$nm.txt 2>&1)"
    t1=$(nowt); ssnap $tag-$nm-1
    [ "$DRY" = 1 ] && { echo "+ RESULT xpl20 $tag $nm: arm $(armn $a): mb c8 T 0 agg_dec_tps, decode_share, /stats deltas; marks [t0, t1] -> $O/marks.jsonl, $O/mb.jsonl"; continue; }
    # the step window = mb's own measuring window [t0 + warm, t0 + dur] (its streams drain after it)
    echo "{\"name\": \"$nm\", \"arm\": \"$a\", \"t0\": $t0, \"t1\": $(awk -v a=$t0 -v d=$DUR 'BEGIN{printf "%.6f", a + d}'), \"end\": $t1}" >> $O/marks.jsonl
    j=$(grep -m1 '^MB ' $O/$tag-$nm.txt | cut -c4-); sd=$(sdelta $tag-$nm-0 $tag-$nm-1)
    [ -n "$j" ] || { xlog "$tag $nm: VIRHE no MB line ($(tail -2 $O/$tag-$nm.txt | tr '\n' ' ' | cut -c1-300))"; continue; }
    jmerge "$j" "$sd" "{\"name\": \"$nm\", \"arm\": \"$a\", \"kind\": \"mb\"}" >> $O/mb.jsonl
    xlog "$tag $nm (arm $(armn $a), mb c8 T 0, dur $DUR warm $WARMS): $(python3 -c 'import json,sys
d, s = json.loads(sys.argv[1]), json.loads(sys.argv[2] or "{}")
print("agg %s tok/s, per-stream p50 %s, decode_share %s, ms/decode step %s, tok/step %s, acceptance %s, TTFT p50 %s s, %s requests %s errors | L20 rows %s | pfctx passes %s rows skipped %s ran %s" % (
    d.get("agg_dec_tps"), d.get("per_stream_dec_p50"), d.get("decode_share"), d.get("ms_decode_step"), d.get("tok_per_decode_step"), d.get("acceptance"),
    d.get("ttft_p50"), d.get("requests"), d.get("errors"), s.get("psk_l20_rows_skipped"), s.get("pfctx_passes"), s.get("pfctx_rows_skipped"), s.get("pfctx_rows_run")))' "$j" "$sd" 2>&1 | cut -c1-600)"
  done
}
ttftabba() {   # ttftabba TAG: per depth, TORDER, each run a fresh distinct text (TOFF + 8 x depth index + run), a reset each
  local tag=$1 dep di=0 a k
  for dep in $DEPTHS; do
    k=0
    for a in $(echo $TORDER | grep -o .); do
      stopfile && return 0
      qrun $tag $a ttft t$dep-$((k + 1))$a $dep $TNTOK $((TOFF + 8 * di + k)) reset
      k=$((k + 1))
    done
    di=$((di + 1))
  done
}
alonex() {   # alonex TAG ARM NAME [CONC]: the greedy set (20 prompts, MT tok, T 0) at conc CONC (1 = alone), after a reset
  local tag=$1 c=${4:-1} t0=$(date +%s)
  arm $2; pcreset $tag-$3
  run "(cd $D && timeout 900 python3 $GS $URL $O/$tag-$3.jsonl --conc $c --max-tokens $MT > $O/$tag-$3.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  xlog "$tag $3 (greedy set at conc $c, temperature 0, $MT tok, arm $(armn $2)): $(tail -1 $O/$tag-$3.txt | cut -c1-200), $(( $(date +%s) - t0 )) s"
}
cmpx() {   # cmpx TAG NAME A B: loadid.py cmp (byte for byte + the first differing char / token) -> identity.txt (stack4's)
  local r
  [ "$DRY" = 1 ] && { echo "+ python3 -I $LD cmp --url $URL $3 $4 --json $O/$1-cmp-$2.json"; return 0; }
  if [ ! -s "$3" ] || [ ! -s "$4" ]; then xlog "$1 $2: not compared (missing $3 or $4)"; echo "$2 not compared" >> $O/identity.txt; return 0; fi
  r=$(timeout 300 python3 -I $LD cmp --url $URL --json $O/$1-cmp-$2.json --label "$1 $2" "$3" "$4" 2>&1 | grep -m1 '^CMPLINE ' | cut -c9-)
  idrec "$2" "${r:-VIRHE compare failed}" "$(basename $3) vs $(basename $4)"
}
loadx() {   # loadx TAG ARM NAME: c4 decode streams + ONE cold LDEPTH deepq prompt (pskld.py --load), the probe's chunks from the step log
  local tag=$1 a=$2 nm=$3 t0=$(date +%s) j md sd s0 s1
  arm $a; pcreset $tag-$nm
  msnap $tag-$nm-0; ssnap $tag-$nm-0; s0=$(date -u +%Y-%m-%dT%H:%M:%S)
  run "(cd $D && timeout 2700 python3 -I $PL run --url $URL --tests $TESTS --mb $MBP --which deep --depth $LDEPTH --ntok $LNTOK --load --conc 4 --warm 10 --pf-tok 0 --out $O/$tag-$nm.json --text $O/$tag-$nm.txt > $O/$tag-$nm.log 2>&1)"
  msnap $tag-$nm-1; ssnap $tag-$nm-1; s1=$(date -u -d '+2 sec' +%Y-%m-%dT%H:%M:%S)
  run "docker logs --since ${s0}Z --until ${s1}Z $CN 2>&1 | grep -E ' n_seq=' > $O/$tag-$nm.steps; python3 -I $PL chunks $O/$tag-$nm.steps --prompt \$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get(\"prompt_tokens\") or 0)' $O/$tag-$nm.json 2>/dev/null || echo 0) --json $O/$tag-$nm.chunks.json > $O/$tag-$nm.chunks.txt 2>&1; gzip -1 -f $O/$tag-$nm.steps"
  [ "$DRY" = 1 ] && { echo "+ RESULT xpl20 $tag $nm: arm $(armn $a): probe TTFT, mixed-step ms, c4 tok/s in the prefill window, the probe's chunks -> $O/load.jsonl"; return 0; }
  j=$(grep -m1 '^LONGID ' $O/$tag-$nm.log | cut -c8-); md=$(mdelta $tag-$nm-0 $tag-$nm-1); sd=$(sdelta $tag-$nm-0 $tag-$nm-1)
  [ -n "$j" ] || { xlog "$tag $nm: VIRHE no LONGID line ($(tail -3 $O/$tag-$nm.log | tr '\n' ' ' | cut -c1-400))"; return 1; }
  jmerge "$j" "$md" "$sd" "{\"name\": \"$nm\", \"arm\": \"$a\", \"kind\": \"load\", \"text\": \"$O/$tag-$nm.txt\", \"chunks\": \"$O/$tag-$nm.chunks.json\"}" >> $O/load.jsonl
  xlog "$tag $nm (arm $(armn $a), c4 decode + one cold ~$LDEPTH deepq prompt, $LNTOK tok, T 0): $(python3 -c 'import json,sys
d, m, s = json.loads(sys.argv[1]), json.loads(sys.argv[2] or "{}"), json.loads(sys.argv[3] or "{}")
print("probe status %s, prompt %s tok, cached %s, TTFT %s s | prefill window %s s: c4 agg %s tok/s, TPOT p90 %s ms | mixed steps %s (non-decode %s at %s ms mean) | L20 rows %s | pfctx rows %s | c4 %s requests, %s errors" % (
    d.get("status"), d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("pf_win_s"), d.get("c4_pf_agg_tps"), d.get("tpot_pf_p90"),
    m.get("mixed_steps"), m.get("nondecode_steps"), m.get("nondecode_ms"), s.get("psk_l20_rows_skipped"), s.get("pfctx_rows_skipped"), d.get("c4_requests"), d.get("c4_errors")))' "$j" "$md" "$sd" 2>&1 | cut -c1-600) | chunks: $(head -c 300 $O/$tag-$nm.chunks.txt 2>/dev/null | tr '\n' ' ')"
}
loadcmp() {   # loadcmp TAG: every loaded probe vs loaded B -- strict where the probe's chunk boundaries are B's, else class (ii) information
  local tag=$1 a cb r
  [ "$DRY" = 1 ] && { echo "+ loaded L / D / LD probe vs loaded B: python3 -I $PL cmpchunks B.chunks.json X.chunks.json -> SAME BOUNDARIES: strict compare, else 'ld-' (class (ii) information)"; return 0; }
  for a in L D X; do
    [ -s $O/$tag-ld-$a.txt ] || continue
    cb=$(python3 -I $PL cmpchunks $O/$tag-ld-B.chunks.json $O/$tag-ld-$a.chunks.json 2>&1 | head -1 | cut -c1-200)
    r=$(cmpfile $O/$tag-ld-B.txt $O/$tag-ld-$a.txt)
    case "$cb" in "SAME BOUNDARIES"*) idrec "load-$(armn $a)-vs-loadB" "$r" "the ~$LDEPTH probe under c4, chunks $cb" ;;
      *) idrec "ld-$(armn $a)-vs-loadB" "$r" "the ~$LDEPTH probe under c4, chunks differ from B's ($cb): class (ii) at sched_bi_chunk 0, information" ;; esac
  done
}
want_anchor() {   # want_anchor DEPTH: K's anchor below a hit's match on the 2048-then-4096 chunk grid ("" unknown)
  python3 -c 'import sys
k, d = int(sys.argv[1]), int(sys.argv[2])
ends = [2048 + 4096 * i for i in range(0, 64)]
anch = [e for i, e in enumerate(ends) if (i + 1) % k == 0 and e <= d]
print(anch[-1] if anch else 0)' $PSKK $1 2>/dev/null; }
hitblock() {   # hitblock TAG ARM DOFF: the arm on its own text -- reset, cold HDEP, hits (no reset) + the tail; then B's cold references
  local tag=$1 a=$2 off=$3 h c w n tl an=$(armn $2)
  stopfile && return 0
  qrun $tag $a hitcold h$an-c$HDEP $HDEP $HNTOK $off reset
  for h in $HITS; do
    stopfile && return 0
    qrun $tag $a hit h$an-$h $h $HNTOK $off noreset
    [ "$DRY" = 1 ] && continue
    c=$(cachedof h$an-$h); w=$(want_anchor $h)
    if [ "$c" = "$w" ]; then xlog "hit $an depth $h: cached $c = the K$PSKK anchor below its match"
    else xlog "hit $an depth $h: VIRHE cached $c, expected the K$PSKK anchor $w"; echo "hit-$an-$h VIRHE cached $c want $w" >> $O/guard.txt; fi
  done
  stopfile && return 0
  qrun $tag $a hit h$an-tail $HDEP $HNTOK $off noreset
  if [ "$DRY" != 1 ]; then
    n=$(cachedof h$an-c$HDEP prompt_tokens); tl=$(( 2048 + (n - 1 - 2048) / 4096 * 4096 )); c=$(cachedof h$an-tail)
    if [ "$c" = "$tl" ]; then xlog "hit $an tail: cached $c = the cold run's tail snapshot"; else xlog "hit $an tail: VIRHE cached $c, expected $tl"; echo "hit-$an-tail VIRHE cached $c want $tl" >> $O/guard.txt; fi
  fi
  for h in $HDEP $HITS; do stopfile && return 0; qrun $tag B hitcold h$an-ref$h $h $HNTOK $off reset; done
  [ "$DRY" = 1 ] && { echo "+ identity: h$an-c$HDEP vs h$an-ref$HDEP (cold, same text), h$an-D vs h$an-refD (hit vs B cold), h$an-tail vs h$an-ref$HDEP"; return 0; }
  idrec "hit-$an-cold$HDEP-vs-B" "$(cmpfile $O/$tag-h$an-ref$HDEP.txt $O/$tag-h$an-c$HDEP.txt)" "$an cold $HDEP vs B cold, text $off"
  for h in $HITS; do
    idrec "hit-$an-$h-vs-B" "$(cmpfile $O/$tag-h$an-ref$h.txt $O/$tag-h$an-$h.txt)" "$an: depth $h resumed at its anchor vs B's cold answer (text $off)"
  done
  idrec "hit-$an-tail-vs-B" "$(cmpfile $O/$tag-h$an-ref$HDEP.txt $O/$tag-h$an-tail.txt)" "$an: the $HDEP prompt resumed at its tail snapshot vs B's cold answer (text $off)"
}
ident() {   # ident TAG: alone / fixed 136k / c8 newcomers / loaded / hits
  local tag=$1 a i=0
  for a in B L D X; do stopfile || alonex $tag $a a1-$a 1; done
  for a in L D X; do cmpx $tag a1-$(armn $a)-vs-B $O/$tag-a1-B.jsonl $O/$tag-a1-$a.jsonl; done
  cmpx $tag info-a1-B-vs-d0 $RD/dsd-d0-gs-c1.jsonl $O/$tag-a1-B.jsonl
  for a in B L D X; do stopfile || qrun $tag $a fixed f136-$a $FDEP $NTOK deepq reset; done
  if [ "$DRY" != 1 ]; then
    for a in B L D X; do idrec "f136-$(armn $a)-vs-dsm225" "$(cmpfile $RM/deep-dsm-225.txt $O/$tag-f136-$a.txt)" "deepq's fixed 136k, 512 tok, vs dsm-225 NST 0 stock"; done
    for a in L D X; do idrec "f136-$(armn $a)-vs-B" "$(cmpfile $O/$tag-f136-B.txt $O/$tag-f136-$a.txt)" "deepq's fixed 136k vs B, same start"; done
  fi
  for a in B L D X; do stopfile || alonex $tag $a c8-$a 8; done
  for a in B L D X; do cmpx $tag c8-$(armn $a)-vs-a1B $O/$tag-a1-B.jsonl $O/$tag-c8-$a.jsonl; done
  for a in B L D X; do stopfile || loadx $tag $a ld-$a; done
  loadcmp $tag
  for a in L D X; do stopfile || hitblock $tag $a $((HOFF + i)); i=$((i + 1)); done
}
steplog() {   # steplog TAG: the engine log captured once (docker logs -t, gzipped), the mb blocks' step windows (pl20.py steps)
  local tag=$1
  capture $O/$tag-engine-full.log.gz
  [ -n "$MORDER" ] || return 0
  run "python3 -I $PX steps --ncsteps $NCS --log $O/$tag-engine-full.log.gz --marks $O/marks.jsonl --warm $WARMS > $O/steps.txt 2>&1"
  [ "$DRY" = 1 ] && { echo "+ RESULT xpl20 $tag steps: per mb block the newcomer steps' wall p50 / mean, MIXED / decode walls -> $O/steps.jsonl"; return 0; }
  grep '^PL20STEPS ' $O/steps.txt | cut -c11- > $O/steps.jsonl
  while IFS= read -r l; do xlog "$tag steps ${l:0:700}"; done < $O/steps.jsonl
}
summary() {
  [ "$DRY" = 1 ] && { echo "+ python3 -I $PX summary --out $O: GUARD, COUNTERS, TTFT per depth / arm (vs B, adjacent pairs), MB c8 (agg, decode_share, newcomer step wall), DRAFTER, IDENTITY"; return 0; }
  python3 -I $PX summary --out $O > $O/summary.txt 2>&1
  while IFS= read -r l; do xlog "summary: ${l:0:3000}"; done < $O/summary.txt
}
dstart() {   # the one NST 5 start
  local tag=p20-n5 ok
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on"
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $STARTENV > $XS2_ENVFILE"; else printf '%s\n' $STARTENV > $XS2_ENVFILE; fi
  xlog "$tag: start ($DESC) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $FIX | arms B $(armj B) L $(armj L) D $(armj D) LD $(armj X) | env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ' | cut -c1-2300)"
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
  smoke $tag || { xlog "$tag: VIRHE the smoke failed -> stopped"; down; return 1; }
  stopfile || guard $tag
  stopfile || acccheck $tag
  if [ $SPEEDOK = 1 ]; then
    stopfile || mbblocks $tag
    stopfile || ttftabba $tag
  else
    xlog "$tag: VIRHE a guard failed (above): no mb c8 / TTFT readings -- identity only"
  fi
  stopfile || ident $tag
  steplog $tag
  if [ "$DRY" = 1 ]; then echo "+ RESULT xpl20 $tag: PSK stats at the end (the log's last PSK / PSK-DS lines, /stats psk + dspark_pfctx, E lines, arms not as pinned)"
  else xlog "$tag: PSK stats at the end: $(pskstat) | /stats psk $(curl -s -m 30 $URL/stats 2>/dev/null | python3 -c 'import json,sys; s=json.load(sys.stdin); print(json.dumps(s.get("psk")), json.dumps(s.get("dspark_pfctx")))' 2>&1 | cut -c1-900) | $(docker logs $CN 2>&1 | grep -cE '^E ') E lines | arms not as pinned: $ARMBAD"; fi
  down
  xlog "$tag: done"
}
trap 'down' EXIT
trap 'exit 143' TERM INT
# ---------------------------------------------------------------- main
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) | one NST 5 start: $DESC | guard B L D LD at ${GDEP} | drafter check $AORDER at [$ADEPS] | mb c8 T 0 ${MORDER:-none} (dur $DUR warm $WARMS) | TTFT $TORDER at [$DEPTHS] fresh texts | identity: alone, fixed 136k, conc 8, c4 + ${LDEPTH} probe, hits L / D / LD at $HDEP -> [$HITS] + tail | no evict | ~57 min (estimate)"
IMGLITS="B|sched_bypass_total, ds41_psk, ds41_psk_k, ds41_psk_l20, dspark_pfctx, epoch_bump
B|ds41_psk_l20
B|dspark_pfctx
B|rad_dspark_pfctx_stats
B|l20_layers_not_staged
B|l20_rows_skipped
B|l20_cone_empty
B|rows_run
B|ds41_psk_k takes 1 .. 4096
B|RADIANCE_LOG_STEPS
B|reset_prefix_cache
B|radiance:engine_steps_mixed_total
B|radiance:draft_accepted_total
A|rad_psk_plan
A|rad_psk_stats
A|rad_dspark_pfctx_stats
A|lever PSK-L20 (fn4 ds41_psk_l20, now %d; RADIANCE_DS41_PSK_L20): %s
A|lever PSK-DS: %lld context passes cut
A|RADIANCE_DS41_PSK_L20
A|RADIANCE_DS41_DSPARK_PFCTX
A|ds41_psk_l20
A|dspark_pfctx
A|lever PSK design 2: VIRHE
A|lever PSK declared (RADIANCE_DS41_PSK, fn4 ds41_psk, now
A|lever PSK: %lld cut steps
A|RADIANCE_DS41_KVFOLD
K|RADIANCE_DS41_GG_PF_TABLE
A|RADIANCE_DS41_DSPARK_WALK
K|dspark_walk_step
A|lever BI declared (RADIANCE_DS41_BI, fn4 ds41_bi, now
B|RADIANCE_DRAFT_SAMPLE
A|DSpark lever SD: the sampled full-vocabulary walk declared
S|mtp_couple_verify
B|RADIANCE_WEIGHT_SIDECAR
B|draft_occ takes 0 (off) .. 64
B|RADIANCE_DRAFT_LEND_WEIGHTS
B|lever Z0-A set up
B|RADIANCE_DRAFT_CTXSKIP_SLOTS"
docker image inspect $IMAGE > /dev/null 2>&1 || { xlog "VIRHE image $IMAGE not on this host"; exit 1; }
IMGMISS=$(echo "$IMGLITS" | timeout 120 docker run --rm -i --network none --entrypoint sh $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels/libr4d.so; S=/opt/radiance/share/radiance/kernels/spechub.so; n=0
while IFS= read -r l; do [ -n "$l" ] || continue; n=$((n + 1)); w=${l%%|*}; s=${l#*|}
  case $w in B) f=$B ;; A) f=$A ;; K) f=$K ;; S) f=$S ;; *) f="$B $A $K" ;; esac
  o=0; for x in $f; do grep -qzF -- "$s" $x && { o=1; break; }; done
  [ $o = 1 ] || echo "MISS $w|$s"; done
test -x /opt/radiance/bin/rad-sidecar || echo "MISS bin/rad-sidecar"; test -s /opt/radiance/share/radiance/pskl20-PROVENANCE.txt || echo "MISS pskl20-PROVENANCE.txt"; echo "CHECKED $n"' 2>&1)
case "$IMGMISS" in "CHECKED "*) ;; *) xlog "VIRHE $IMAGE is not the ds41-pskl20 image -- missing: $(echo "$IMGMISS" | grep -v '^CHECKED' | tr '\n' ';' | cut -c1-900)"; exit 1 ;; esac
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)) is the ds41-pskl20 image: all $(echo "$IMGMISS" | sed 's/CHECKED //') literals present (the 44-switch registry ending ds41_psk_l20, dspark_pfctx; PSK-L20 / PSK-DS's lines, env names and exports; /stats l20_* and dspark_pfctx; stack4's int7 levers) + bin/rad-sidecar + pskl20-PROVENANCE -- OK"
HMISS=""
for f in "$LIB29" "$LIBPR" $Y $GS $LD $MBT $PL $P2 $PX $NCS $DQ $MBP $TESTS $D/fastcheck.py; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -B -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $GS $LD $MBT $PL $P2 $PX $NCS $DQ $MBP $TESTS $D/fastcheck.py 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
ST=$(timeout 60 python3 -I -B $LD selftest --gs $GS --mb $MBP 2>&1 | tail -1); case "$ST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$LD(${ST:0:300})" ;; esac
MST=$(timeout 60 python3 -I -B $MBT --mb $MBP --selftest 2>&1 | tail -1); case "$MST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$MBT(${MST:0:300})" ;; esac
PST=$(timeout 60 python3 -I -B $PL selftest --tests $TESTS --mb $MBP 2>&1 | tail -1); case "$PST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$PL(${PST:0:300})" ;; esac
P2T=$(timeout 60 python3 -I -B $P2 selftest --pskld $PL --tests $TESTS 2>&1 | tail -1); case "$P2T" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$P2(${P2T:0:300})" ;; esac
PXT=$(timeout 60 python3 -I -B $PX selftest --ncsteps $NCS 2>&1 | tail -1); case "$PXT" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$PX(${PXT:0:300})" ;; esac
NCT=$(timeout 120 python3 -I -B $NCS selftest 2>&1 | tail -1); case "$NCT" in *"SELFTEST OK"*|*"selftest OK"*|*" OK"*) ;; *) HMISS="$HMISS selftest:$NCS(${NCT:0:300})" ;; esac
OFFS=$(for i in 0 1 2 3; do printf "%s," $((GOFF + i)); done; for i in 0 1 2; do printf "%s," $((AOFF + i)); done; for i in $(seq 0 23); do printf "%s," $((TOFF + i)); done; for i in 0 1 2; do printf "%s," $((HOFF + i)); done; echo 1200)
TXT=$(cd $D && timeout 120 python3 -I -B $PX texts --pskld $PL --tests $TESTS --doffs $OFFS 2>&1 | tail -1); case "$TXT" in "SELFTEST OK"*) ;; *) HMISS="$HMISS texts:(${TXT:0:300})" ;; esac
DQP=$(timeout 60 python3 -I -B -c 'import importlib.util, json, sys
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
WA=$(want_anchor 40000):$(want_anchor 20000); [ "$WA" = "30720:14336" ] || HMISS="$HMISS want_anchor(40000/20000 at K$PSKK = $WA, K4 wants 30720:14336)"
FNS="log run mkd restart check down stopfile best_vars desc serve_args xlog armj armn pinj stackenv dfree keeplog capture thermline nowt cmpfile idrec identw modes vramstart fn4get fn4set nudge fn4code pcreset arm msnap mdelta ssnap sdelta pskstat jmerge gate1 qrun cachedof swnote smoke guard acccheck mbblocks ttftabba alonex cmpx loadx loadcmp want_anchor hitblock ident steplog summary dstart"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee sha256sum gzip zcat df du; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
for a in B L D X; do python3 -c 'import json,sys; json.loads(sys.argv[1])' "$(pinj $a)" 2>/dev/null || HMISS="$HMISS json:$a"; python3 -c 'import json,sys; json.loads(sys.argv[1])' "$(pinj $a "$DET")" 2>/dev/null || HMISS="$HMISS json:$a+DET"; done
[ "$(sha256sum $MODELDIR/$SIDECAR 2>/dev/null | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256 2>/dev/null)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR"
[ "$(sha256sum $MODELDIR/$SIDE 2>/dev/null | cut -c1-64)" = "$SIDESHA" ] || HMISS="$HMISS sha256:$MODELDIR/$SIDE(want ${SIDESHA:0:16})"
grep -q 'XS2_ENVFILE' $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no env_file)"
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config refuses it)"
HMISS=$(echo $HMISS)
[ -z "$HMISS" ] || { xlog "VIRHE helpers / inputs do not resolve: $HMISS"; exit 1; }
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing, loadid.py $ST, mbt.py $MST, pskld.py $PST, p2bq.py $P2T, pl20.py $PXT, ncsteps.py ($NCT), text offsets $TXT, deepq.py = pskld.py's deep request ($DQP), K$PSKK anchors 40000 / 20000 -> $WA, all $(echo $FNS | wc -w) functions resolve, tools on PATH, sidecars $SIDECAR + $SIDE sha256 OK, yaml accepted -- OK"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ "$(dfree)" -ge "$DISKMIN" ] || { xlog "VIRHE $(dfree) MB free on / < $DISKMIN MB: not starting (the 10 GB guard + docker's step log)"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { xlog "VIRHE $M225 missing or not verified"; exit 1; }
  for f in $RD/dsd-d0-gs-c1.jsonl $RM/deep-dsm-225.txt $WARM_REF_225 $RS4/s4-n5-stats0.json; do [ -s $f ] || xlog "note -- reference $f missing (not compared)"; done
  xlog "$ITEM: helpers md5: pl20.py $(md5sum < $PX | cut -c1-8), ncsteps.py $(md5sum < $NCS | cut -c1-8) (read only), mbt.py $(md5sum < $MBT | cut -c1-8), loadid.py $(md5sum < $LD | cut -c1-8), pskld.py $(md5sum < $PL | cut -c1-8), p2bq.py $(md5sum < $P2 | cut -c1-8), deepq.py $(md5sum < $DQ | cut -c1-8), greedyset.py $(md5sum < $GS | cut -c1-8), mb.py $(md5sum < $MBP | cut -c1-8) | $(dfree) MB free on /"
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDE $RD/dsd-d0-gs-c1.jsonl $RM/deep-dsm-225.txt $WARM_REF_225 $RS4/s4-n5-stats0.json; do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  for a in B L D X; do echo "+ arm $(armn $a): $(pinj $a)"; done
  echo "+ drafter check overlay: $DET"
  echo "+ disk: $(dfree) MB free on / (the real run refuses to start below $DISKMIN)"
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(information here; the real run refuses to start beside one)"
fi
mkd $O $O/guard $STATSDIR
[ "$DRY" = 1 ] || { : > $O/identity.txt; : > $O/guard.txt; }
stopfile || dstart
summary
unset $SWITCHES
down
xlog "$ITEM: done -- $O"
