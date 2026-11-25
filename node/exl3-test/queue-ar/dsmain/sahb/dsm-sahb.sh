#!/bin/bash
# dsm-sahb.sh (12.11.2026, queue-ar/dsmain/sahb, ds-main-port sahb builder) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r00zzz-cz-usahb.sh. Log tag "RESULT xsahb ", out runs/xdsm/sahb (XSB_OUT). Made from COPIES of
# nc/dsm-ncprof-p.sh (9c20358f: the start, modes / switch / arm functions, alonex / c8u / capture, START P), nc/ncsteps.py
# (0a6e0ed2, copied beside this file with ncone.py / ncmb.py / ds_opprof_nc.py, which it loads from its own directory) and
# stack4/dsm-stack4.sh (21ce8c8e: cmpfile / idrec / cmpx / the greedy-set alone, loadid and loaded-probe identity); the originals
# are not changed. + sahbsum.py (this item's guards and summary).
# LEVERS SA / HB on image ds41-sahb (ds41-int7-pskglue 5093ce9 + ds41-noa 3f7fc59 + 647830e): fn4 stage_short N / stage_pick K
#   (lever SA: the prefill stager armed for a step of 1025 .. 2112 tokens at the span's shared level, ring at that level's own
#   place, the step's arena loan kept below it; K 0 = every pooled unit of a layer, K >= 1 = only the pooled units one of the
#   last K stager-size steps routed) and place_hb R (lever HB: on the ranks RADIANCE_PLACE_HB_RANKS names -- 1, 2 here = 83:00
#   and 86:00 under the one 80: root complex, which stream the same bytes ~30 % slower -- the R units a class with the highest
#   long-memory prefill heat are kept resident above every unprotected resident, below the decode-pinned core). Both exact
#   (placement / address only: the same bytes, the same kernels); every switch 0 = the int7 image's staging and placement.
# WHY (r00zzz-cz-ncprof / -pz-ncprofp, 12.11.): at mb c8 each ~1.1k newcomer is ONE MIXED step of 0.58-0.61 s (decode 35 ms),
#   decode_share ~0.79; profiled alone 627 ms: experts 236 / 315 / 307 / 236 ms, collective 198 / 117 / 129 / 203 ms (ranks 0 / 3
#   wait for 1 / 2), the stager never armed (RADIANCE_STAGE_SPAN=0 leaves the 2112-row level without a ring), host-pool ~425
#   units ~3.8 GB a rank streamed in-kernel. Equal host-pool units / stream bytes on every rank (424.6 / 424.2 / 416.9 / 425;
#   7.19 / 7.15 / 7.10 / 7.18 GB) and equal decode experts (13.0-13.2 ms): the r1 / r2 excess is their link, not their bytes.
# DESIGN (coordinator 12.11.): ONE NST 5 START U = stack4's start (runs/ds41-exl3.log 2026-11-11 21:51:34 "RESULT xs4 s4-n5:
#   start") on the new image + RADIANCE_LOG_STEPS=1 + RADIANCE_PLACE_HB_RANKS=1,2 (read only under place_hb > 0). Every arm =
#   stack4's fixed pin + stack4's S + its own switches, applied in process (POST + epoch_bump, a nudge, READ BACK; each block's
#   counter baseline taken AFTER the read-back -- the PSK-CR lesson: a request admitted before the switch latches the old arm):
#     B  = stage_short 0, stage_pick 0, place_hb 0          SA = stage_short XSB_SSN (1025), stage_pick K*, place_hb 0
#     HB = stage_short 0, place_hb XSB_HBR (128)              SH = SA + HB
#   (1) SCREEN (gate first, alone, unprofiled): XSB_SCREEN_N (8) fresh one-token ~1.1k prompts (nc/ncone.py's) under B, stage_pick
#       0 / 1 / 2 (stage_short on), B; K* = the fastest clean pick by the /stats step wall (default XSB_PICK_DEFAULT 1) -- 79
#       (7.11.) found the whole pooled set (pick 0) a LOSS for 1-2k-row steps, so the pick is measured, not assumed;
#   (2) ABBA blocks XSB_ORDER (B SA HB SH SH HB SA B): per block the 1u stepprobe essay / code (T 0, 512 tok = stack4's c1
#       phase), then mb c8 exactly as stack4 (mbt.py's phase_c at T 0, dur 60 warm 15, through ncmb.py --mode phase: the 2 s
#       sampler + per-request taps + /stats at the window's edges) -> agg_dec_tps, decode_share, the newcomer step wall from
#       the step log (ncsteps.py c8), staged / streamed GB a non-decode step, the fn4 "sahb" counters per rank, PSK's;
#   (3) IDENTITY at T 0: the greedy set c1 alone under B and SH (vs each other and vs bz d0); under SH each greedy prompt at
#       c8 load (stack4/loadid.py: 6 mb streams + one fresh ~1.1k newcomer at +0.5 s beside its decode = SA's MIXED steps) vs
#       alone; the load case c4 + one cold ~XSB_LDEPTH deepq prompt (stack4/pskld.py --load) under B and SH (strict, same
#       load) and each vs stack4's loaded S answer (cross-start, labelled);
#   START P (XSB_P 1): --profile-ops every XSB_EVERY (16) + RADIANCE_DEBUG_MISS_LOG, arms B SA HB SH in that order (SH after HB
#       keeps HB's reserve: a staged pass swaps nothing), each XSB_PWARM (16) warm + XSB_PALONE (16) measured one-token ~1.1k
#       prompts -> per rank experts / collective / dense ms a newcomer step (ds_opprof_nc.py, a window an arm), per-rank
#       host-pool units (miss log). Not a speed: the per-rank split is the reading.
# COUNTER GUARD (sahbsum.py block / pick; else VIRHE and NO speed reading from that block): pins read back before and after;
#   SA arms: short_steps and short_units grow on every rank, non-SA arms: short_steps +0; HB arms: hb_reserve = R on ranks 1, 2
#   and 0 elsewhere, hb_reserved > 0 on 1, 2; non-HB arms: hb_reserve 0, hb_promotions +0; PSK design 2 (pinned in S) on the
#   corrected /stats psk counters: latched, planned_steps, cut_passes grow, plan_mismatch +0 (never layers_skipped: SA arms
#   more passes, so the stager's layers_skipped grows by design); c8 clean; both 1u prompts.
# ~45 min (estimate; budget XSB_BUDGET_MIN 55: START P is skipped, logged, if START U ran past budget - 10): START U start +
#   checks ~6, screen ~2, 8 blocks x ~2 (1u ~20 s + mb c8 ~85 s) = ~16, identity ~7, START P ~9 (start ~4.5 + 128 profiled prompts), parsing ~2.
# NO cc_sim, no evict, no power / host / fan settings; production compose files read only (--project-directory). Exact by
#   construction; identity reported, never hidden.
# Knobs: XSB_IMAGE, XSB_ITEM, XSB_OUT, XSB_SSN (1025), XSB_HBR (128), XSB_HBRANKS (1,2), XSB_PICKS ("0 1 2"), XSB_PICK_DEFAULT (1),
#   XSB_SCREEN_N (8), XSB_ORDER ("B SA HB SH SH HB SA B"), XSB_DUR (60), XSB_WARM (15), XSB_IDENT (1 | 0), XSB_LDEPTH (64000),
#   XSB_P (1 | 0), XSB_PARMS ("B SA HB SH"), XSB_PWARM (16), XSB_PALONE (16), XSB_EVERY (16), XSB_DISKMIN_MB (11000),
#   XSB_BUDGET_MIN (55), XSB_SIDECAR, XSB_SIDESHA.
# DRYRUN=1 bash dsm-sahb.sh : prints every command; checks the image literals (the SA / HB switches, counters and start note),
#   every helper (files, parse, selftests), the sidecars, the yaml, the disk; no server, no GPU, no request to :8090, writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/../ds41-stack2-tp4.yaml                   # stack2's yaml, read only (env_file ${XS2_ENVFILE} carries every start word)
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh
O=${XSB_OUT:-$R/xdsm/sahb}; export STATSDIR=$O/stats-u
RS4=$R/xdsm/stack4; RD=$R/xdsm/draft
export IMAGE=${XSB_IMAGE:?set XSB_IMAGE}
ITEM=${XSB_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-usahb}
SSN=${XSB_SSN:-1025}; HBR=${XSB_HBR:-128}; HBRANKS=${XSB_HBRANKS:-1,2}; PICKS=${XSB_PICKS:-0 1 2}; PICKDEF=${XSB_PICK_DEFAULT:-1}
SCN=${XSB_SCREEN_N:-8}; ORDER=${XSB_ORDER:-B SA HB SH SH HB SA B}; DUR=${XSB_DUR:-60}; WARMS=${XSB_WARM:-15}; DOI=${XSB_IDENT:-1}
LDEPTH=${XSB_LDEPTH:-64000}; DOP=${XSB_P:-1}; PARMS=${XSB_PARMS:-B SA HB SH}; PWARMN=${XSB_PWARM:-16}; PALONEN=${XSB_PALONE:-16}
EVERY=${XSB_EVERY:-16}; DISKMIN=${XSB_DISKMIN_MB:-11000}; BUDGET=${XSB_BUDGET_MIN:-55}
SIDE=${XSB_SIDECAR:-dspark-dlw-q8_0.rad}; SIDESHA=${XSB_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
for v in $SSN $HBR $PICKS $PICKDEF $SCN $DUR $WARMS $DOI $LDEPTH $DOP $PWARMN $PALONEN $EVERY $DISKMIN $BUDGET; do case $v in ''|*[!0-9]*) echo "a numeric knob is not a number ('$v')"; exit 1 ;; esac; done
for a in $ORDER $PARMS; do case $a in B|SA|HB|SH) ;; *) echo "XSB_ORDER / XSB_PARMS take B / SA / HB / SH (got '$a')"; exit 1 ;; esac; done
case $HBRANKS in *[!0-9,]*|"") echo "XSB_HBRANKS takes a comma list of ranks (got '$HBRANKS')"; exit 1 ;; esac
[ $SSN -ge 1 ] && [ $SSN -le 65536 ] && [ $HBR -ge 1 ] && [ $HBR -le 4096 ] || { echo "XSB_SSN 1 .. 65536, XSB_HBR 1 .. 4096"; exit 1; }
for k in $PICKS $PICKDEF; do [ $k -le 64 ] || { echo "a stage_pick is 0 .. 64 (got $k)"; exit 1; }; done
[ $PWARMN -ge $EVERY ] || { echo "XSB_PWARM ($PWARMN) must be >= XSB_EVERY ($EVERY): a table dump must fall inside the warm prompts"; exit 1; }
[ $WARMS -lt $DUR ] || { echo "the warm-up is not shorter than the phase"; exit 1; }
NCONE=$HD/ncone.py; NCMB=$HD/ncmb.py; NCS=$HD/ncsteps.py; OPNC=$HD/ds_opprof_nc.py; SUM=$HD/sahbsum.py
MBT=$HD/../stack4/mbt.py; S4B=$HD/../stack4/dsm-stack4.sh; NCB=$HD/../nc/dsm-ncprof-p.sh   # references, read only
PL=$HD/../stack4/pskld.py; LD=$HD/../stack4/loadid.py                                       # stack4's drivers, read only
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
MBP=$D/mb.py; TESTS=$D/ds41_tests.py; SPB=$D/stepprobe.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
MT=192; T0=$(date +%s)
NV=0
xlog() { case "$*" in *VIRHE*) NV=$((NV + 1)) ;; esac; log "RESULT xsahb $*"; }
# ---------------------------------------------------------------- copied from dsm-ncprof-p.sh (9c20358f; = stack4's start; tag xnc -> xsahb)
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
# + this item's: lever SA / HB set up on every rank, SA able to arm 1025 .. 2112-token steps, HB on ranks 1 and 2
LITS="$LITS|lever,SA,/,HB,(ds41-sahb),set,up|stage_short,can,arm,the,prefill,stager,for,a,step,of,1025,..,2112,tokens|place_hb,can,reserve,on,rank,1"
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": 8, "drafter_mx": 1, "gg_pf": 1, "f32_pf": 25, "dspark_walk": 1, "ds41_argather": 48, "idx_split": 1, "ds41_qnfold": 1, "ds41_swfold": 1, "draft_ctxskip": 1, "dspark_conf": 1, "dspark_conf_rows": 1, "draft_lend": 1, "draft_probe_backoff": 3, "draft_hybrid": 1}'
FIX=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update({"draft_lend_hyst": 0, "arena_d0": 0, "draft_lend_w": 0, "draft_occ_k": int(sys.argv[2])}); print(json.dumps(d))' "$FIX5" 32) || { echo "pin JSON"; exit 1; }
SPIN='{"ds41_bi": 2, "draft_sample": 85, "draft_occ": 6, "place_lf": 0, "ds41_psk": 2, "ds41_psk_k": 4}'   # stack4's S (ncprof's S)
PICK=$PICKDEF
armj() {   # armj B|SA|HB|SH|P<k>: the arm's own switches (on top of stack4's S)
  case $1 in
    B)  echo '{"stage_short": 0, "stage_pick": 0, "place_hb": 0}' ;;
    SA) echo "{\"stage_short\": $SSN, \"stage_pick\": $PICK, \"place_hb\": 0}" ;;
    HB) echo "{\"stage_short\": 0, \"stage_pick\": 0, \"place_hb\": $HBR}" ;;
    SH) echo "{\"stage_short\": $SSN, \"stage_pick\": $PICK, \"place_hb\": $HBR}" ;;
    P*) echo "{\"stage_short\": $SSN, \"stage_pick\": ${1#P}, \"place_hb\": 0}" ;;
  esac
}
pinj() {   # pinj ARM [EXTRAJSON ...]: the fixed pin + stack4's S + the arm + the extras as one POST body (a later one wins)
  python3 -c 'import json,sys
d = {}
for a in sys.argv[1:]: d.update(json.loads(a or "{}"))
print(json.dumps(d))' "$FIX" "$SPIN" "$(armj $1)" "${@:2}"
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
  RADIANCE_DS41_PSK RADIANCE_PLACE_LF RADIANCE_PLACE_LF_DECAY RADIANCE_LOG_STEPS RADIANCE_PLACE_HB_RANKS RADIANCE_PLACE_HB_DECAY"
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
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xsahb $1 warm-up identity vs $2"; return 0; }
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
modes() {   # modes TAG: MODEWORDS in the container, mover 0 kept free x4, rule-w, K1, ARGATHER 48, every literal of LITS -> 0, else 1
  [ "$DRY" = 1 ] && { echo "+ modes $1: [$MODEWORDS] in the container's env; mover '0 more kept free' on 4 ranks; rule-w in every recorded pass; K1 folds declared =1; ARGATHER 48-row note; literals [$(echo "$LITS" | tr ',' ' ')]"; return 0; }
  local CE kv bad="" ok=1 NW NR KL AL nz lit found="" miss=""
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in RADIANCE_PLACE_TURNOVER=0 $MODEWORDS; do echo "$CE" | grep -qxF "$kv" || bad="$bad ${kv%%=*}"; done
  [ -z "$bad" ] || { xlog "$1: VIRHE start words not in the container:$bad"; ok=0; }
  nz=$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots \([0-9]+ shadow, 0 more kept free')
  [ "$nz" = 4 ] || { xlog "$1: VIRHE turnover 0 asked but $nz of 4 ranks keep 0 slots free"; ok=0; }
  NW=$(docker logs $CN 2>&1 | grep -c 'the acquire rule plays .*RADIANCE_AQL_ACQ=rule-w)'); NR=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_AQL_ACQ=rule)')
  if [ -n "${PROFEVERY:-}" ]; then   # --profile-ops records no pass (ctx.cpp tape_on_ needs !profiling_): env rule-w, no plain rule line
    if echo "$CE" | grep -qx "RADIANCE_AQL_ACQ=rule-w" && [ "$NR" = 0 ]; then xlog "$1: rule-w replaced by the profile sync (not a speed): --profile-ops records no pass, RADIANCE_AQL_ACQ=rule-w in the container env, no plain rule line ($NW / $NR)"
    else xlog "$1: VIRHE RADIANCE_AQL_ACQ=rule-w not in the container env (or a plain rule line) under --profile-ops ($NW / $NR)"; ok=0; fi
  else
    [ "$NW" -gt 0 ] && [ "$NR" = 0 ] || { xlog "$1: VIRHE rule-w is not what the recorded passes play ($NW / $NR)"; ok=0; }
  fi
  KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-500)
  case "$KL" in *"ds41_kvfold=1"*"ds41_ofold=1"*"ds41_onefill=1"*) ;; *) xlog "$1: VIRHE the K1 switches do not start at 1: '${KL}'"; ok=0 ;; esac
  AL=$(docker logs $CN 2>&1 | grep -m1 'the MoE gather and its all-reduce as one kernel' | sed 's/^radiance: //' | cut -c1-300)
  case "$AL" in *"up to 48 tokens a step (RADIANCE_DS41_ARGATHER, RADIANCE_DS41_ARGATHER_MAX"*) ;; *) xlog "$1: VIRHE the ARGATHER note is not the 48-row one: '${AL}'"; ok=0 ;; esac
  IFS='|' read -ra lit <<< "$LITS"
  for l in "${lit[@]}"; do [ -n "$l" ] || continue
    if docker logs $CN 2>&1 | grep -qF -- "${l//,/ }"; then found="$found ${l%%,*}"; else miss="$miss | ${l//,/ }"; fi; done
  xlog "$1 modes: ${#lit[@]} literal(s), found [$(echo $found)]$([ -n "$miss" ] && echo " -- VIRHE missing:$miss") | $(docker logs $CN 2>&1 | grep -E "lever SA / HB|lever PSK|lever BI declared|prefill staging:|activation arena: lends|scheduler: --prefill|chunk geometry" | sed 's/^radiance: //; s/^[A-Z] *//; s/^[A-Z]\[[0-9]\] //' | cut -c1-330 | sort -u | tr '\n' '|' | cut -c1-2600)"
  [ -z "$miss" ] || ok=0
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
fn4code() { [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
arm() {   # arm ARM [EXTRAJSON]: the pin + the arm (+ extra) + epoch_bump, a nudge to apply it at a step
  fn4set "$(pinj $1 "${2:-}" '{"epoch_bump": 1}')"; nudge
}
gate1() {   # restart + ONE check; over the limit logged (in-process A/B: a slow state shows in every arm)
  local tag=$1 lim=$2 ms st=fast
  restart $tag || { keeplog $tag-fail; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && return 0
  awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}' && { st="SLOW (not retried)"; }
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
envfile() {   # envfile FILE WORDS...: the start words the yaml's env_file carries
  local f=$1; shift
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $* > $f"; else printf '%s\n' "$@" > $f; fi
}
envline() { env | grep -E '^(RADIANCE_|R4D_)' | grep -vE '(KEY|TOKEN|SECRET)' | sort | tr '\n' ' ' | cut -c1-2400; }
ARMOK=1
armset() {   # armset TAG ARM [EXTRAJSON]: the pin + the arm (+ extra) + epoch_bump, a nudge, the applied switches READ BACK
  local g r
  arm $2 "${3:-}"
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4: every switch of $(pinj $2 "${3:-}") applied, nothing pending -> RESULT xsahb $1 arm $2 ${3:-} applied (read back)"; return 0; }
  g=$(fn4get)
  r=$(python3 -c 'import json,sys
g, w = json.loads(sys.argv[1]), json.loads(sys.argv[2])
bad = {k: g["switches"].get(k) for k, v in w.items() if g["switches"].get(k) != v}
s = g["switches"]
print(("OK" if not bad and not g["pending"] else "FAIL %s pending %s" % (bad, g["pending"])) + " | bi=%s psk=%s/%s stage_short=%s stage_pick=%s place_hb=%s epoch=%s applied_step=%s" % (
    s.get("ds41_bi"), s.get("ds41_psk"), s.get("ds41_psk_k"), s.get("stage_short"), s.get("stage_pick"), s.get("place_hb"), g.get("epoch"), g.get("applied_step")))' "$g" "$(pinj $2 "${3:-}")" 2>&1 | cut -c1-400)
  case "$r" in OK*) xlog "$1 arm $2 $(armj $2) ${3:-} applied (read back): $r"; return 0 ;; esac
  xlog "$1 arm $2: VIRHE the pin is not applied ($r) -- no reading for this arm"; ARMOK=0; return 1
}
alonex() {   # alonex TAG NAME N K0: N fresh one-token ~1.1k newcomers, one at a time (ncone.py; /stats between them)
  local t0=$(date +%s)
  run "(cd $D && timeout 1200 python3 -I -B $NCONE --mb $MBP --url $URL --n $3 --k0 $4 --tag $2 --out $O/$2-reqs.jsonl --stats $O/$2-stats.jsonl > $O/$2.txt 2>&1)"
  [ "$DRY" = 1 ] || xlog "$1 $2 ($3 fresh ~1.1k prompts one at a time, 1 token, T 0): $(grep -m1 '^NCONE ' $O/$2.txt | cut -c7- | cut -c1-300) | $(thermline $t0 $(date +%s))"
}
capture() {   # capture FILE: the engine log with the daemon's receipt times, gzipped (never written raw)
  run "docker logs -t $CN 2>&1 | gzip -1 > $1"
  [ "$DRY" = 1 ] || xlog "engine log $(basename $1): $(du -m $1 | cut -f1) MB gz, $(zcat $1 | wc -l) lines, $(zcat $1 | grep -cE ' step [0-9]+ (prefill|decode|MIXED) n_seq=') step lines, $(zcat $1 | grep -c 'per-op timing, rank') profile tables; $(dfree) MB free"
}
startenv() {   # startenv TAG [P]: stack4's dstart serve line + RADIANCE_LOG_STEPS=1 + RADIANCE_PLACE_HB_RANKS (+ profiler, miss log under P)
  stackenv; export RADIANCE_LOG_STEPS=1 RADIANCE_PLACE_HB_RANKS=$HBRANKS; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on"
  export XS2_ENVFILE=$O/$1.env
  MODEWORDS="$LEVENV RADIANCE_LOG_STEPS=1 RADIANCE_PLACE_HB_RANKS=$HBRANKS"
  if [ "${2:-}" = P ]; then
    export RADIANCE_DEBUG_MISS_LOG=/stats/miss-p.log PROFEVERY=$EVERY STATSDIR=$O/stats-p XARGS="$XARGS --profile-ops"
    MODEWORDS="$MODEWORDS RADIANCE_DEBUG_MISS_LOG=/stats/miss-p.log RADIANCE_PROFILE_EVERY=$EVERY"
  else
    export PROFEVERY= STATSDIR=$O/stats-u
  fi
  envfile $XS2_ENVFILE $LEVENV RADIANCE_LOG_STEPS=1 RADIANCE_PLACE_HB_RANKS=$HBRANKS $([ "${2:-}" = P ] && echo RADIANCE_DEBUG_MISS_LOG=/stats/miss-p.log)
  mkd $STATSDIR
}
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
# ---------------------------------------------------------------- copied from dsm-stack4.sh (21ce8c8e; tag xs4 -> xsahb)
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
cmpx() {   # cmpx TAG NAME A B: loadid.py cmp (byte for byte + the first differing char / token) -> identity.txt
  local r
  [ "$DRY" = 1 ] && { echo "+ python3 -I $LD cmp --url $URL $3 $4 --json $O/$1-cmp-$2.json"; return 0; }
  if [ ! -s "$3" ] || [ ! -s "$4" ]; then xlog "$1 $2: not compared (missing $3 or $4)"; echo "$1 $2 not compared" >> $O/identity.txt; return 0; fi
  r=$(timeout 300 python3 -I $LD cmp --url $URL --json $O/$1-cmp-$2.json --label "$1 $2" "$3" "$4" 2>&1 | grep -m1 '^CMPLINE ' | cut -c9-)
  idrec "$1-$2" "${r:-VIRHE compare failed}" "$(basename $3) vs $(basename $4)"
}
galone() {   # galone TAG ARM NAME: pin + arm read back, prefix-cache reset, greedyset.py's 20 prompts at c1 ALONE, temperature 0
  local tag=$1 t0=$(date +%s)
  armset $tag $2 || return 1
  pcreset $tag-$3
  run "(cd $D && timeout 900 python3 $GS $URL $O/$tag-$3.jsonl --conc 1 --max-tokens $MT > $O/$tag-$3.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  xlog "$tag $3 (greedy set c1 ALONE, temperature 0, $MT tok, arm $2 $(armj $2)): $(tail -1 $O/$tag-$3.txt | cut -c1-200), $(( $(date +%s) - t0 )) s"
}
lidc8() {   # lidc8 TAG ARM NAME: loadid.py run -- each greedy prompt under 6 mb streams + one fresh ~1.1k newcomer at +0.5 s (= c8)
  local tag=$1 t0=$(date +%s) j s0 s1
  armset $tag $2 || return 1
  pcreset $tag-$3
  s0=$(fn4get)
  run "(cd $D && timeout 2700 python3 -I $LD run --url $URL --gs $GS --mb $MBP --out $O/$tag-$3.jsonl --conc 6 --warm 10 --offset 0.5 --pf-tok 1100 --max-tokens $MT > $O/$tag-$3.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  s1=$(fn4get)
  j=$(grep -m1 '^LOADID ' $O/$tag-$3.txt | cut -c8-)
  [ -n "$j" ] || { xlog "$tag $3: VIRHE no LOADID line ($(tail -3 $O/$tag-$3.txt | tr '\n' ' ' | cut -c1-400))"; return 1; }
  xlog "$tag $3 (each greedy prompt under 6 mb streams + one fresh ~1.1k newcomer at +0.5 s beside its decode, T 0, arm $2): $(python3 -c 'import json,sys
d = json.loads(sys.argv[1])
print("%s of %s answered, probe %s s mean | newcomers %s of %s cold, TTFT %s s mean | mixed steps a probe min %s / median %s; probes without one %s | streams %s requests, %s errors" % (
    d["answered"], d["of"], d["probe_s_mean"], d["pf_cold"], d["of"], d["pf_ttft_mean"], d["mixed_min"], d["mixed_median"],
    d["probes_without_mixed"] or "none", d["c4_requests_done"], d["c4_errors"]))' "$j" 2>&1 | cut -c1-600) | SA under it: rank 0 short passes +$(python3 -c 'import json,sys
a, b = json.loads(sys.argv[1]), json.loads(sys.argv[2])
r = lambda g: {x["rank"]: x for x in (g.get("sahb") or {}).get("ranks") or []}
x, y = r(a).get(0) or {}, r(b).get(0) or {}
print("%s, units +%s" % ((y.get("short_steps") or 0) - (x.get("short_steps") or 0), (y.get("short_units") or 0) - (x.get("short_units") or 0)))' "$s0" "$s1" 2>/dev/null) | $(( $(date +%s) - t0 )) s"
}
ldx() {   # ldx TAG ARM NAME: c4 decode streams + ONE cold LDEPTH deepq prompt (stack4/pskld.py --load), 512 greedy tokens
  local tag=$1 nm=$3 t00=$(date +%s) j
  armset $tag $2 || return 1
  pcreset $tag-$nm
  run "(cd $D && timeout 2700 python3 -I $PL run --url $URL --tests $TESTS --mb $MBP --which deep --depth $LDEPTH --ntok 512 --load --conc 4 --warm 10 --pf-tok 0 --out $O/$tag-$nm.json --text $O/$tag-$nm.txt > $O/$tag-$nm.log 2>&1)"
  [ "$DRY" = 1 ] && return 0
  j=$(grep -m1 '^LONGID ' $O/$tag-$nm.log | cut -c8-)
  [ -n "$j" ] || { xlog "$tag $nm: VIRHE no LONGID line ($(tail -3 $O/$tag-$nm.log | tr '\n' ' ' | cut -c1-400))"; return 1; }
  xlog "$tag $nm (arm $2 $(armj $2), c4 + one cold ~$LDEPTH deepq prompt, T 0): $(python3 -c 'import json,sys
d = json.loads(sys.argv[1])
print("probe status %s, prompt %s tok, cached %s, TTFT %s s | prefill window %s s: c4 agg %s tok/s, per-stream p50 %s, TPOT p50 %s / p90 %s ms | c4 %s requests, %s errors" % (
    d.get("status"), d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("pf_win_s"), d.get("c4_pf_agg_tps"), d.get("c4_pf_stream_tps_p50"),
    d.get("tpot_pf_p50"), d.get("tpot_pf_p90"), d.get("c4_requests"), d.get("c4_errors")))' "$j" 2>&1 | cut -c1-600) | $(thermline $t00 $(date +%s))"
}
# ---------------------------------------------------------------- this item's own functions
swstart() {   # swstart TAG: the switches at start (stack4's + SA / HB at 0 and capable); refusals of out-of-range values
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 at start: gg_gemv 2, aql_acq 2, K1 three 1, epoch 0, ds41_bi 0, ds41_psk 0, place_lf 0, draft_occ 0, draft_occ_k 32, draft_adaptive 1, draft_depth0 0, draft_sample 0, sched_bi_chunk 0 / sched_interleave 8 / sched_mixed_tok 2112, stage_short 0, stage_pick 0, place_hb 0, sahb.stage_capable 1, sahb.hb_capable 1, sahb ranks 4; POST stage_short 65537 / place_hb 4097 / stage_pick 65 -> 400"; return 0; }
  local s0 r c1 c2 c3
  s0=$(fn4get)
  c1=$(fn4code '{"stage_short": 65537}'); c2=$(fn4code '{"place_hb": 4097}'); c3=$(fn4code '{"stage_pick": 65}')
  r=$(python3 -c 'import json,sys
s0 = json.loads(sys.argv[1]); w = s0["switches"]; sa = s0.get("sahb") or {}
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) and s0["epoch"] == 0
ok = ok and w.get("ds41_bi") == 0 and w.get("ds41_psk") == 0 and w.get("place_lf") == 0 and w.get("draft_occ") == 0 and w.get("draft_occ_k") == 32
ok = ok and w["draft_adaptive"] == 1 and w["draft_depth0"] == 0 and s0["draft_capable"] == 1 and w.get("draft_sample") == 0
ok = ok and w.get("sched_bi_chunk") == 0 and w.get("sched_interleave") == 8 and w.get("sched_mixed_tok") == 2112
ok = ok and w.get("stage_short") == 0 and w.get("stage_pick") == 0 and w.get("place_hb") == 0
ok = ok and sa.get("stage_capable") == 1 and sa.get("hb_capable") == 1 and len(sa.get("ranks") or []) == 4
ok = ok and sys.argv[2:5] == ["400", "400", "400"]
print(("OK" if ok else "FAIL") + " stage_short %s stage_pick %s place_hb %s capable %s/%s ranks %d | refusals %s | sched_bi_chunk %s ds41_bi %s ds41_psk %s/%s" % (
    w.get("stage_short"), w.get("stage_pick"), w.get("place_hb"), sa.get("stage_capable"), sa.get("hb_capable"), len(sa.get("ranks") or []),
    "/".join(sys.argv[2:5]), w.get("sched_bi_chunk"), w.get("ds41_bi"), w.get("ds41_psk"), w.get("ds41_psk_k")))' "$s0" "$c1" "$c2" "$c3" 2>&1 | cut -c1-400)
  case "$r" in OK*) xlog "$1 switches at start: $r"; return 0 ;; esac
  xlog "$1 switches at start: VIRHE $r | $(echo "$s0" | cut -c1-900)"; return 1
}
snap() {   # snap NAME: GET /admin/fn4 and the full /stats -> NAME-fn4.json / NAME-st.json
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 > $O/$1-fn4.json; GET /stats > $O/$1-st.json"; return 0; }
  curl -s -m 10 $URL/admin/fn4 > $O/$1-fn4.json 2>/dev/null; curl -s -m 30 $URL/stats > $O/$1-st.json 2>/dev/null
}
SCREENRUNS=""; SCK=900
screen() {   # screen TAG: B, P<k> for k in PICKS, B -- SCN alone newcomers each; the fn4 counters around each
  local tag=$1 nm i=0 a
  for a in B $(for k in $PICKS; do echo P$k; done) B; do
    stopfile && return 0
    i=$((i + 1)); nm=s$i-$a
    armset $tag $a || continue
    snap $nm-0
    alonex $tag $nm $SCN $SCK; SCK=$((SCK + 50))
    snap $nm-1
    SCREENRUNS="$SCREENRUNS $nm=$([ $a = B ] && echo B || echo $a):$O/$nm.json:$O/$nm-0-fn4.json:$O/$nm-1-fn4.json"
  done
}
pickit() {   # pickit: the screen's engine log, ncsteps.py alone a run, sahbsum.py pick -> PICK
  local s k
  capture $O/u-engine-screen.log.gz
  for s in $SCREENRUNS; do
    k=${s%%=*}
    pyx $k python3 -I -B $NCS alone --log $O/u-engine-screen.log.gz --reqs $O/$k-reqs.jsonl --stats $O/$k-stats.jsonl --tag $k --label "screen $k" --json $O/$k.json
  done
  [ "$DRY" = 1 ] && { echo "+ python3 -I -B $SUM pick $(for s in $SCREENRUNS; do echo "--run $s"; done | tr '\n' ' ')--default $PICKDEF --json $O/screen.json -> PICK k (SA / SH run stage_pick k)"; return 0; }
  timeout 300 python3 -I -B $SUM pick $(for s in $SCREENRUNS; do echo "--run $s"; done) --default $PICKDEF --json $O/screen.json > $O/screen.out 2>&1
  k=$(grep -m1 '^PICK ' $O/screen.out | cut -c6-)
  case "$k" in ''|*[!0-9]*) xlog "screen: VIRHE no pick ($(tail -2 $O/screen.out | tr '\n' ' ' | cut -c1-300)); stage_pick $PICKDEF (the default)"; k=$PICKDEF ;; esac
  PICK=$k
  xlog "$(grep -m1 '^SCREEN ' $O/screen.out | cut -c1-3500)"
}
BLOCKS=""
block() {   # block TAG I ARM: arm read back, then (baseline AFTER the read-back) 1u stepprobe essay / code, mb c8 as stack4, snapshots
  local tag=$1 i=$2 a=$3 nm=b$2-$3 t0=$(date +%s)
  armset $tag $a || { BLOCKS="$BLOCKS $nm:FAILPIN"; return 1; }
  [ "$DRY" = 1 ] || pinj $a > $O/$nm-want.json
  snap $nm-0
  run "(cd $D && timeout 900 python3 $SPB essay,code 512 > $O/$nm-c1.txt 2>&1)"
  run "(cd $D && timeout $(( DUR + 900 )) python3 -I -B $NCMB --mb $MBP --mode phase --temperature 0 --n 8 --tag $nm-c8 --dur $DUR --warm $WARMS --reqs $O/$nm-c8-reqs.jsonl --stats-prefix $O/$nm-c8-st --samples $O/$nm-c8-samp.jsonl > $O/$nm-c8.txt 2>&1)"
  snap $nm-1
  BLOCKS="$BLOCKS $nm"
  [ "$DRY" = 1 ] || xlog "$tag block $i arm $a $(armj $a): 1u $(grep '^J ' $O/$nm-c1.txt | python3 -c 'import json,sys
print(" / ".join("%s %s tok/s %s ms" % (d.get("prompt"), d.get("decode_tps"), d.get("ms_per_decode_step")) for d in (json.loads(l[2:]) for l in sys.stdin)))' 2>/dev/null) | c8 $(grep -m1 '^MB ' $O/$nm-c8.txt | cut -c4- | cut -c1-400) | $(thermline $t0 $(date +%s))"
}
ablocks() {   # ablocks TAG: the ABBA order
  local i=0 a
  for a in $ORDER; do
    stopfile && return 0
    i=$((i + 1))
    block $1 $i $a
  done
}
ident() {   # ident TAG: identity at T 0 -- alone B / SH, SH under c8 load with newcomers, the c4 + cold 64k load case B / SH
  local tag=$1 s4
  stopfile || galone $tag B a1-B
  stopfile || galone $tag SH a1-SH
  stopfile || lidc8 $tag SH lid8-SH
  stopfile || ldx $tag B ld-B
  stopfile || ldx $tag SH ld-SH
  cmpx $tag a1-SH-vs-a1-B $O/$tag-a1-B.jsonl $O/$tag-a1-SH.jsonl
  cmpx $tag a1-B-vs-d0 $RD/dsd-d0-gs-c1.jsonl $O/$tag-a1-B.jsonl
  cmpx $tag lid8-SH-vs-a1-B $O/$tag-a1-B.jsonl $O/$tag-lid8-SH.jsonl
  cmpx $tag lid8-SH-vs-a1-SH $O/$tag-a1-SH.jsonl $O/$tag-lid8-SH.jsonl
  [ "$DRY" = 1 ] && { echo "+ cmpfile $O/$tag-ld-B.txt $O/$tag-ld-SH.txt; each vs $RS4/s4-n5-ld-2S.txt (cross-start)"; return 0; }
  idrec "$tag-ld-SH-vs-ld-B" "$(cmpfile $O/$tag-ld-B.txt $O/$tag-ld-SH.txt)" "the ~$LDEPTH prompt under c4 load, SH vs B (same load, same chunks)"
  s4=$(ls $RS4/s4-n5-ld-*S.txt 2>/dev/null | head -1)
  [ -n "$s4" ] && idrec "$tag-ld-B-vs-stack4-ldS" "$(cmpfile $s4 $O/$tag-ld-B.txt)" "cross-start: stack4's loaded S answer (int7 image), labelled"
  [ -n "$s4" ] && idrec "$tag-ld-SH-vs-stack4-ldS" "$(cmpfile $s4 $O/$tag-ld-SH.txt)" "cross-start: stack4's loaded S answer (int7 image), labelled"
}
UDONE=0; PDONE=0; IDONE=0
ustart() {   # START U: unprofiled; screen, ABBA blocks, identity
  local tag=sb-u
  diskok $tag || return 1
  startenv $tag
  xlog "$tag: start (START U, UNPROFILED: stack4's NST 5 start + RADIANCE_LOG_STEPS=1 + RADIANCE_PLACE_HB_RANKS=$HBRANKS) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $FIX + S $SPIN | arms B $(armj B) SA $(armj SA) HB $(armj HB) SH $(armj SH) (stage_pick from the screen) | env: $(envline)"
  gate1 $tag 70 || { xlog "$tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  modes $tag || { xlog "$tag: VIRHE the modes are not the start's -> stopped"; down; return 1; }
  drafted $tag || { keeplog $tag-nodraft; down; return 1; }
  swstart $tag || { keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  stopfile || screen $tag
  stopfile || pickit
  xlog "$tag: stage_pick $PICK for SA / SH | arms SA $(armj SA) SH $(armj SH)"
  stopfile || ablocks $tag
  [ $DOI = 1 ] && { stopfile || { ident $tag; IDONE=1; }; }
  capture $O/u-engine.log.gz
  [ "$DRY" = 1 ] || xlog "$tag: $(docker logs $CN 2>&1 | grep -cE '^E ') E lines | stager: $(docker logs $CN 2>&1 | grep -c 'prefill staging') notes | $(docker logs $CN 2>&1 | grep -E 'fn4: switches .*(stage_short|place_hb)' | tail -2 | sed 's/^radiance: //' | cut -c1-200 | tr '\n' '|')"
  down
  UDONE=1
}
PW=""
pstart() {   # START P: profiled + the miss log; per arm XSB_PWARM warm + XSB_PALONE measured alone newcomers
  local tag=sb-p a ta0 ta1 k0=1500
  diskok $tag || return 1
  startenv $tag P
  xlog "$tag: start (START P, PROFILED --profile-ops every $EVERY + RADIANCE_DEBUG_MISS_LOG + RADIANCE_LOG_STEPS: not a speed) IMAGE=$IMAGE MODEL=$MODEL $(desc) | arms [$PARMS] stage_pick $PICK | env: $(envline)"
  restart $tag || { keeplog $tag-fail; down; xlog "$tag: VIRHE not healthy -- no profile"; return 1; }
  run "mkdir -p $O/guard; (cd $D && timeout 600 python3 ds41_tests.py correct --only-q --effort none --outdir $O/guard > $O/guard/$tag-warm-q.txt 2>&1)"   # = check()'s warm-up line, no 16k / fastcheck
  modes $tag || { xlog "$tag: VIRHE the modes are not the start's -> stopped"; down; return 1; }
  drafted $tag || { keeplog $tag-nodraft; down; return 1; }
  swstart $tag || { keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  for a in $PARMS; do
    stopfile && break
    armset $tag $a || continue
    alonex $tag p-$a-warm $PWARMN $k0; k0=$((k0 + 50))   # >= EVERY one-step prompts: a table dump falls inside them (HB: the reserve fills)
    snap p-$a-0
    ta0=$([ "$DRY" = 1 ] && echo TA0 || date +%s.%N)
    alonex $tag p-$a-alone $PALONEN $k0; k0=$((k0 + 50))
    ta1=$([ "$DRY" = 1 ] && echo TA1 || date +%s.%N)
    snap p-$a-1
    PW="$PW $a:$ta0:$ta1"
  done
  run "docker stop -t 60 $CN > /dev/null 2>&1"        # the shutdown table and the miss log's closing summary
  capture $O/p-engine.log.gz
  run "gzip -1 -f $STATSDIR/miss-p.log"
  down
  [ "$DRY" = 1 ] || echo "$PW" > $O/p-windows.txt
  PDONE=1
}
parse() {
  local b a bf blk="" wins="" walls="" w nm ms
  if [ $UDONE = 1 ] || [ "$DRY" = 1 ]; then
    for b in $BLOCKS; do
      case $b in *:FAILPIN) xlog "block ${b%%:*}: VIRHE pins not applied -- no reading"; continue ;; esac
      a=${b#*-}
      pyx $b-c8 python3 -I -B $NCS c8 --log $O/u-engine.log.gz --reqs $O/$b-c8-reqs.jsonl --tag $b-c8 --sa $O/$b-c8-st-w0.json --sb $O/$b-c8-st-w1.json --samples $O/$b-c8-samp.jsonl --mbline $O/$b-c8.txt --label "block $b (unprofiled)" --json $O/$b-c8.json
      pyx $b python3 -I -B $SUM block --name $b --arm $a --want $O/$b-want.json --pre-fn4 $O/$b-0-fn4.json --post-fn4 $O/$b-1-fn4.json --pre-st $O/$b-0-st.json --post-st $O/$b-1-st.json --c8 $O/$b-c8.json --c1 $O/$b-c1.txt --sa $(case $a in SA|SH) echo 1 ;; *) echo 0 ;; esac) --hb $(case $a in HB|SH) echo $HBR ;; *) echo 0 ;; esac) --hb-ranks $HBRANKS --json $O/$b.json
      blk="$blk --block $O/$b.json"
    done
  fi
  if [ $PDONE = 1 ] || [ "$DRY" = 1 ]; then
    for w in ${PW:-B:TA0:TA1 SA:TA0:TA1 HB:TA0:TA1 SH:TA0:TA1}; do
      nm=${w%%:*}
      pyx p-$nm python3 -I -B $NCS alone --log $O/p-engine.log.gz --reqs $O/p-$nm-alone-reqs.jsonl --stats $O/p-$nm-alone-stats.jsonl --tag p-$nm-alone --skip-mod $EVERY --label "P ALONE $nm (PROFILED)" --json $O/p-$nm.json
      ms=$([ "$DRY" = 1 ] && echo MS || python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["step_ms"]["mean"])' $O/p-$nm.json 2>/dev/null)
      wins="$wins --win $w"; [ -n "$ms" ] && walls="$walls --wall $nm:$ms"
    done
    pyx p-miss python3 -I -B $NCS miss --file $O/stats-p/miss-p.log.gz --log $O/p-engine.log.gz $wins --json $O/p-miss.json
    pyx p-prof python3 -I -B $OPNC $O/p-engine.log.gz $wins $walls --every $EVERY --label sb-p --json $O/p-prof.json
  fi
  local ref
  ref=$([ "$DRY" = 1 ] && echo REF || python3 -c 'import json,statistics,sys
rs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(", ".join("stack4 T 0 c8 %s %.1f tok/s" % (a, statistics.mean(r["res"]["c8"]["agg_dec_tps"] for r in rs if r["arm"] == a and r.get("ok"))) for a in "AS"))' $RS4/s4-n5-t0-blocks.jsonl 2>/dev/null)
  run "python3 -I -B $SUM report $blk $([ $PDONE = 1 ] || [ "$DRY" = 1 ] && echo "--prof $O/p-prof.json") --screen $O/screen.json --ident $O/identity.txt --ref '${ref:-n/a}; ncprof (int7, cross-start): c8 S newcomer step p50 581.5 ms, decode_share 0.793; profiled alone experts 236 / 315 / 307 / 236 ms, collective 198 / 117 / 129 / 203 ms' > $O/report.txt 2>&1"
  [ "$DRY" = 1 ] && { echo "+ every line of $O/report.txt -> the log (RESULT xsahb summary / profiled / identity)"; return 0; }
  while IFS= read -r l; do case "$l" in "RESULT xsahb "*) log "${l:0:3800}" ;; esac; done < $O/report.txt
  local nv; nv=$(( $(grep -c VIRHE $O/report.txt 2>/dev/null) + NV ))
  [ $ARMOK = 1 ] || nv=$((nv + 1))
  [ $UDONE = 1 ] || nv=$((nv + 1))
  [ $DOP = 0 ] || [ $PDONE = 1 ] || nv=$((nv + 1))
  [ $DOI = 0 ] || [ $IDONE = 1 ] || nv=$((nv + 1))
  log "RESULT xsahb guards: $([ "$nv" = 0 ] && echo "OK (pins read back, the SA / HB / PSK counters moved only where they apply in every block, c8 windows clean, step-log lines, identity run$([ $DOP = 1 ] && echo ", profile tables in every arm window"))" || echo "VIRHE in $nv place(s) (START U done $UDONE, identity done $IDONE, START P done $PDONE) -- see the lines above; a reading with VIRHE is not a reading")"
}
trap 'down' EXIT
trap 'exit 143' TERM INT
# ---------------------------------------------------------------- main
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) | ONE START U (stack4's NST 5 start + RADIANCE_LOG_STEPS=1 + RADIANCE_PLACE_HB_RANKS=$HBRANKS): screen B / stage_pick [$PICKS] / B ($SCN alone newcomers each) -> K; ABBA [$ORDER] = 1u essay / code + mb c8 T 0 dur $DUR warm $WARMS a block; identity$([ $DOI = 1 ] || echo " SKIPPED") (greedy set alone B / SH, under c8 + newcomers SH, c4 + cold ~$LDEPTH B / SH) | START P$([ $DOP = 1 ] || echo " SKIPPED") (profiled every $EVERY + miss log): [$PARMS] $PWARMN + $PALONEN alone newcomers an arm | stage_short $SSN, place_hb $HBR on ranks $HBRANKS | ~45 min (budget $BUDGET)"
IMGLITS="B|dspark_conf_rows, gg_pf, f32_pf, dspark_walk, ds41_bi, draft_sample, draft_lend_hyst, arena_d0, draft_lend_w, draft_occ, draft_occ_k, place_lf, sched_bi_chunk, sched_interleave, sched_mixed_tok,
B|sched_short_bypass, sched_bypass_total, ds41_psk
B|ds41_psk_k, stage_short, stage_pick, place_hb,
B|stage_short takes 0 (off) or 1 .. 65536
B|stage_pick takes 0 (every pooled unit)
B|place_hb takes 0 (off) or 1 .. 4096
B|lever SA / HB (ds41-sahb) set up, off: stage_short %s; place_hb %s
B|can arm the prefill stager for a step of %lld .. %lld tokens at the span's shared level
B|can reserve on %s (long prefill heat decay %.2f)
B|RADIANCE_PLACE_HB_RANKS
B|RADIANCE_PLACE_HB_DECAY
B|\"sahb\": {\"stage_capable\":
B|short_steps
B|pick_dropped
B|expect_sum_units
B|hb_promotions
B|sa_recall_bytes
B|reset_prefix_cache
B|radiance:engine_steps_mixed_total
B|radiance:engine_step_seconds_decode_total
A|lever PSK declared (RADIANCE_DS41_PSK, fn4 ds41_psk, now
A|lever BI declared (RADIANCE_DS41_BI, fn4 ds41_bi, now
B|RADIANCE_LOG_STEPS
B|n_seq=%lld n_tok=%lld
B|  [%d] slot=%d q=%d ctx=%d
B|RADIANCE_PROFILE_EVERY
B|per-op timing, rank %d.
B|RADIANCE_DEBUG_MISS_LOG
B|radiance-miss-log/1
B|steps_mixed
B|stage_layers
B|prefill staging: a %s ring at the top of the arena
A|RADIANCE_DS41_DSPARK_WALK
A|DSpark lever SD: the sampled full-vocabulary walk declared
B|RADIANCE_WEIGHT_SIDECAR
B|lever Z0-A set up
B|RADIANCE_DRAFT_CTXSKIP_SLOTS"
docker image inspect $IMAGE > /dev/null 2>&1 || { xlog "VIRHE image $IMAGE not on this host"; exit 1; }
IMGMISS=$(echo "$IMGLITS" | timeout 120 docker run --rm -i --network none --entrypoint sh $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels/libr4d.so; S=/opt/radiance/share/radiance/kernels/spechub.so; n=0
while IFS= read -r l; do [ -n "$l" ] || continue; n=$((n + 1)); w=${l%%|*}; s=${l#*|}
  case $w in B) f=$B ;; A) f=$A ;; K) f=$K ;; S) f=$S ;; *) f="$B $A $K" ;; esac
  o=0; for x in $f; do grep -qzF -- "$s" $x && { o=1; break; }; done
  [ $o = 1 ] || echo "MISS $w|$s"; done
test -s /opt/radiance/share/radiance/int7-PROVENANCE.txt || echo "MISS int7-PROVENANCE.txt"
test -s /opt/radiance/share/radiance/sahb-PROVENANCE.txt || echo "MISS sahb-PROVENANCE.txt"; echo "CHECKED $n"' 2>&1)
case "$IMGMISS" in "CHECKED "*) ;; *) xlog "VIRHE $IMAGE is not the ds41-sahb image this item reads -- missing: $(echo "$IMGMISS" | grep -v '^CHECKED' | tr '\n' ';' | cut -c1-900)"; exit 1 ;; esac
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)) is the ds41-sahb image: all $(echo "$IMGMISS" | sed 's/CHECKED //') literals present (the fn4 registry with stage_short / stage_pick / place_hb after ds41_psk_k, their ranges, the SA / HB start note, GET \"sahb\" counters, RADIANCE_PLACE_HB_RANKS / _DECAY, the step-log format, the profiler's table, the miss log, /stats step / stager counters) + int7-PROVENANCE + sahb-PROVENANCE -- OK"
HMISS=""
for f in "$LIB29" "$LIBPR" $Y $MBP $TESTS $SPB $GS $D/fastcheck.py $NCONE $NCMB $NCS $OPNC $SUM $MBT $S4B $NCB $PL $LD $QA/deepq.py; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $MBP $TESTS $SPB $GS $D/fastcheck.py $NCONE $NCMB $NCS $OPNC $SUM $MBT $PL $LD 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
ST1=$(timeout 60 python3 -I -B $NCONE --mb $MBP --selftest 2>&1 | tail -1); case "$ST1" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$NCONE(${ST1:0:300})" ;; esac
ST2=$(timeout 60 python3 -I -B $NCMB --mb $MBP --selftest 2>&1 | tail -1); case "$ST2" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$NCMB(${ST2:0:300})" ;; esac
ST3=$(timeout 120 python3 -I -B $NCS selftest 2>&1 | tail -1); case "$ST3" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$NCS(${ST3:0:300})" ;; esac
ST4=$(timeout 120 python3 -I -B $OPNC selftest 2>&1 | tail -1); case "$ST4" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$OPNC(${ST4:0:300})" ;; esac
ST5=$(timeout 60 python3 -I -B $SUM selftest 2>&1 | tail -1); case "$ST5" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$SUM(${ST5:0:300})" ;; esac
ST6=$(timeout 60 python3 -I -B $PL selftest --tests $TESTS --mb $MBP 2>&1 | tail -1); case "$ST6" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$PL(${ST6:0:300})" ;; esac
ST7=$(timeout 60 python3 -I -B $LD selftest --gs $GS --mb $MBP 2>&1 | tail -1); case "$ST7" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$LD(${ST7:0:300})" ;; esac
FNS="log run mkd restart check down stopfile best_vars desc serve_args xlog armj pinj stackenv keeplog thermline identw modes fn4get fn4set fn4code nudge arm gate1 dfree diskok envfile envline armset alonex capture startenv drafted pyx pcreset cmpfile idrec cmpx galone lidc8 ldx swstart snap screen pickit block ablocks ident ustart pstart parse"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee sha256sum gzip zcat df du; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
for a in B SA HB SH $(for k in $PICKS; do echo P$k; done); do python3 -c 'import json,sys
d = json.loads(sys.argv[1]); a = sys.argv[2]
want = {"B": (0, 0), "SA": (1, 0), "HB": (0, 1), "SH": (1, 1)}.get(a, (1, 0))
ok = d["ds41_bi"] == 2 and d["ds41_psk"] == 2 and d["ds41_psk_k"] == 4 and d["gg_gemv"] == 2 and d["epoch_bump"] == 1
ok = ok and (d["stage_short"] > 0) == bool(want[0]) and (d["place_hb"] > 0) == bool(want[1]) and 0 <= d["stage_pick"] <= 64
sys.exit(0 if ok else 1)' "$(pinj $a '{"epoch_bump": 1}')" $a 2>/dev/null || HMISS="$HMISS json:$a"; done
[ "$(sha256sum $MODELDIR/$SIDECAR 2>/dev/null | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256 2>/dev/null)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR"
[ "$(sha256sum $MODELDIR/$SIDE 2>/dev/null | cut -c1-64)" = "$SIDESHA" ] || HMISS="$HMISS sha256:$MODELDIR/$SIDE(want ${SIDESHA:0:16})"
for w in XS2_ENVFILE 'RADIANCE_PROFILE_EVERY: ${PROFEVERY' 'RADIANCE_DEBUG_MISS_LOG:' '${STATSDIR:?set STATSDIR}:/stats'; do grep -qF "$w" $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no '$w')"; done
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config refuses it)"
HMISS=$(echo $HMISS)
[ -z "$HMISS" ] || { xlog "VIRHE helpers / inputs do not resolve: $HMISS"; exit 1; }
HMD5="ncone.py $(md5sum < $NCONE | cut -c1-8), ncmb.py $(md5sum < $NCMB | cut -c1-8), ncsteps.py $(md5sum < $NCS | cut -c1-8) (copy of nc/ncsteps.py $(md5sum < $HD/../nc/ncsteps.py | cut -c1-8)), ds_opprof_nc.py $(md5sum < $OPNC | cut -c1-8), sahbsum.py $(md5sum < $SUM | cut -c1-8), stack4/loadid.py $(md5sum < $LD | cut -c1-8), stack4/pskld.py $(md5sum < $PL | cut -c1-8), stack4/mbt.py $(md5sum < $MBT | cut -c1-8), dsm-stack4.sh $(md5sum < $S4B | cut -c1-8), nc/dsm-ncprof-p.sh $(md5sum < $NCB | cut -c1-8), greedyset.py $(md5sum < $GS | cut -c1-8), stepprobe.py $(md5sum < $SPB | cut -c1-8), mb.py $(md5sum < $MBP | cut -c1-8), lib-29.sh $(md5sum < $LIB29 | cut -c1-8), yaml $(md5sum < $Y | cut -c1-8)"
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing; ncone.py $ST1 | ncmb.py $ST2 | ncsteps.py $ST3 | ds_opprof_nc.py $ST4 | sahbsum.py $ST5 | stack4/pskld.py $ST6 | stack4/loadid.py $ST7 | all $(echo $FNS | wc -w) functions resolve, tools on PATH, sidecars $SIDECAR + $SIDE sha256 OK, yaml accepted (env_file, RADIANCE_PROFILE_EVERY, RADIANCE_DEBUG_MISS_LOG, /stats mount) -- OK | md5: $HMD5"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { xlog "VIRHE $M225 missing or not verified"; exit 1; }
  for f in $WARM_REF_225 $RS4/s4-n5-t0-blocks.jsonl $RS4/guard-s4-n5.txt $RD/dsd-d0-gs-c1.jsonl; do [ -s $f ] || xlog "note -- reference $f missing (not compared)"; done
  xlog "$ITEM: helpers md5: $HMD5"
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDE $WARM_REF_225 $RS4/s4-n5-t0-blocks.jsonl $RS4/guard-s4-n5.txt $RD/dsd-d0-gs-c1.jsonl $(ls $RS4/s4-n5-ld-*S.txt 2>/dev/null | head -1); do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  for a in B SA HB SH $(for k in $PICKS; do echo P$k; done); do echo "+ arm $a: $(armj $a) | POST body $(pinj $a '{"epoch_bump": 1}')"; done
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(information here; the real run refuses to start beside one)"
fi
mkd $O $O/guard
[ "$DRY" = 1 ] || : > $O/identity.txt
stopfile || ustart
if [ $DOP = 1 ]; then
  el=$(( ($(date +%s) - T0) / 60 ))
  if [ "$DRY" != 1 ] && [ $el -gt $((BUDGET - 10)) ]; then xlog "START P SKIPPED: START U took $el min (budget $BUDGET - 10) -- no per-rank profile"
  else stopfile || pstart; fi
fi
parse
unset $SWITCHES
down
xlog "$ITEM: done in $(( ($(date +%s) - T0) / 60 )) min -- $O ($(dfree) MB free)"
