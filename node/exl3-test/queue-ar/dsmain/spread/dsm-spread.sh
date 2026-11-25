#!/bin/bash
# dsm-spread.sh (12.11.2026, queue-ar/dsmain/spread, ds-main-port ARC builder) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r92z-spread.sh. Log tag "RESULT xspr ", out runs/xdsm/spread (XSPR_OUT). Made from a COPY of arc/dsm-arc.sh
# (e783f84c: the start, LEVENV / LITS, the base pin, stackenv .. gate1, budok, readback, the harness block runner and its c8mix
# parse; tag xarc -> xspr); the original is not changed. Helpers read only: harness/ncmb2.py + mbt2.py + c8mix.py, nc/ncmb.py
# (the fallback only), i9b/i9bctr.py (counter deltas, INFO only), da/daheld.py (IMPORTED by spread/damark.py, not copied),
# mb.py, ds41_tests.py.
# PURPOSE (the coordinator 12.11.): the B-vs-B SPREAD of c4 and c1 mb blocks on the fixed-schedule harness -- if tight, the new
# TESTING.md block rules ("block counts for DS screens") may run c4 / c1 at one block a arm -- and c8 again on this base (the
# 0.2-0.5 % spread seen so far). Plus ONE live-server self-check of the DA harness's fixed marker derivation.
# ONE START on the int10 image, the ARC screen's base: stack5's start + RADIANCE_DS41_MHC_FUSE=0 (M1 declared); KVR / SP off.
#   PIN = stack5's S5T, draft_sample 0, SP 0 / 0, M1 on (mhc_fuse 1), ix_lds 0, ix_ni 0, draft_depth0 3, sched_row_cap 48,
#   ds41_arcoded 0 (the fixed pin: stack4's + ds41_psk_k 8, draft_lend 1). NO ARMS: the pin is posted once (+ epoch_bump, read
#   back) and held. No P0 (no selftest: nothing new is screened).
# BLOCKS (all T 0 on the harness: harness/ncmb2.py --mode phase --sched fixed --pattern 1xN --seed 1 --window gens --edge 0.5
#   --ignore-eos 1, --dur XSPR_DUR 30 --warm XSPR_WARM 10; the window is XSPR_GENS<N> x N newcomer steps, so its length is set by
#   the generations: c8 gens 2 = 16 newcomer steps ~45 s, c4 gens 2 = 8 steps ~28 s, c1 gens 6 = 6 requests ~30 s), in
#   XSPR_ORDER "4 1 8 4 1 8 4 1" (c4 x 3, c1 x 3, c8 x 2, interleaved: a phase's blocks sit minutes apart, as a screen's
#   same-arm blocks do). Per block: the MB line (agg tok/s, per-stream p50, TTFT p50, ms and tokens a decode step,
#   acceptance, decode share, the client mix), TTFT p50 / p90 over the window's requests (the taps), the counters (INFO),
#   then c8mix.py block on the engine log (the mix line, the newcomer step walls; INFO for c4 / c1).
# SUMMARY: per phase the mean and the max-min spread in % of the mean of each metric, and ONE verdict line a phase:
#   "c4 spread X % (one block per arm OK if <= ~1 %)" on agg tok/s (the per-stream p50 and the ms a decode step beside).
# DA LIVE CHECK (after the blocks, ~10 s): spread/damark.py imports da/daheld.py and executes ITS marker lines (the 12.11. fix in
#   run(): /tokenize messages + reasoning_effort none minus the reference text's own ids) against this server, then counts how
#   many of runs/xdsm/da/stats/corpus/corpus-gen.jsonl's 192 documents get a usable prefix with that marker (daheld.run's own
#   loop and skip condition, executed). "RESULT xspr da-marker: <marker> -> N of 192 docs". No /completion call; the corpus
#   is only read.
# GUARDS: the pin read back + 0 request errors (a VIRHE). Exact by design (nothing is switched).
# ~13 min estimate (start + checks ~5.5, pin ~0.3, blocks 3 x ~40 s + 3 x ~45 s + 2 x ~60 s = ~6.3, c8mix parse ~0.5, DA ~0.2,
#   end ~0.5); BUDGET XSPR_BUDGET_MIN 15 (a block that would end past it is skipped, logged). NO evict, no cc_sim, no power /
#   host / fan settings; production compose files read only.
# MOCK: XSPR_SW=0 = a server without the int9 / int9b / int10 switches (queue-ar/tools/mockrad): the pin posts none of them and
#   keeps draft_depth0 1; the start line carries a "VIRHE XSPR_SW=0" marker (a mock run only).
# Knobs: XSPR_IMAGE, XSPR_ITEM, XSPR_OUT, XSPR_SW (1), XSPR_ORDER ("4 1 8 4 1 8 4 1"), XSPR_DUR (30), XSPR_WARM (10), XSPR_GENS4 (2),
#   XSPR_GENS1 (6), XSPR_GENS8 (2), XSPR_MHC (1), XSPR_D0 (3), XSPR_CAP (48), XSPR_SPN (0) / XSPR_SPK (0), XSPR_DA (1),
#   XSPR_CORPUS (runs/xdsm/da/stats/corpus/corpus-gen.jsonl), XSPR_DISKMIN_MB (11000), XSPR_BUDGET_MIN (15), XSPR_SIDECAR,
#   XSPR_SIDESHA; XSPR_DRY_ALLOW_OLD=1 (DRYRUN only).
# DRYRUN=1 bash dsm-spread.sh : prints every command; checks the image literals (int9 + int9b + int10), every helper (files,
#   parse, selftests incl. damark.py's against daheld.py), the corpus, the sidecars, the yaml, the disk; no server, no GPU,
#   writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/../ds41-stack2-tp4.yaml                   # stack2's yaml, read only (env_file ${XS2_ENVFILE})
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh
O=${XSPR_OUT:-$R/xdsm/spread}; export STATSDIR=$O/stats
RD=$R/xdsm/draft; RM=$R/xdsm; RS5=$R/xdsm/stack5
export IMAGE=${XSPR_IMAGE:?set XSPR_IMAGE}
ITEM=${XSPR_ITEM:-57-p4zyzzzzz-ds-r92z-spread}
PSKK=8; BIC=3; NOA=2; OCC=6; OCCK=32; DSA=0; HBR=0; HBRANKS=1,2      # stack5's S5T values, draft_sample 0 (i9b's base)
SW=${XSPR_SW:-1}; ORDER=${XSPR_ORDER-4 1 8 4 1 8 4 1}; DUR=${XSPR_DUR:-30}; WARM=${XSPR_WARM:-10}
GENS4=${XSPR_GENS4:-2}; GENS1=${XSPR_GENS1:-6}; GENS8=${XSPR_GENS8:-2}
MHC=${XSPR_MHC:-1}; D0=${XSPR_D0:-3}; CAP=${XSPR_CAP:-48}; SPN=${XSPR_SPN:-0}; SPK=${XSPR_SPK:-0}; ARCN=0; BASEX=""
DODA=${XSPR_DA:-1}; CORPUS=${XSPR_CORPUS:-$R/xdsm/da/stats/corpus/corpus-gen.jsonl}
DISKMIN=${XSPR_DISKMIN_MB:-11000}; BUDGET=${XSPR_BUDGET_MIN:-15}; ALLOWOLD=${XSPR_DRY_ALLOW_OLD:-0}
SIDE=${XSPR_SIDECAR:-dspark-dlw-q8_0.rad}; SIDESHA=${XSPR_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
MT=192; SEED=1234
case $SW in 0|1) ;; *) echo "XSPR_SW is 1 (the item) or 0 (a server without the int9 / int9b / int10 switches: the mock) (got '$SW')"; exit 1 ;; esac
for v in $DUR $WARM $GENS4 $GENS1 $GENS8 $MHC $D0 $CAP $SPN $SPK $DODA $DISKMIN $BUDGET; do case $v in ''|*[!0-9]*) echo "a numeric knob is not a number ('$v')"; exit 1 ;; esac; done
for p in $ORDER; do case $p in 1|4|8) ;; *) echo "XSPR_ORDER takes the phases 1 4 8 (c1 / c4 / c8) (got '$p')"; exit 1 ;; esac; done
[ $WARM -lt $DUR ] || { echo "the warm-up is not shorter than the window"; exit 1; }
[ $GENS4 -ge 1 ] && [ $GENS1 -ge 1 ] && [ $GENS8 -ge 1 ] || { echo "XSPR_GENS* >= 1"; exit 1; }
[ $D0 -le 3 ] && [ $CAP -le 4096 ] && [ $MHC -le 1048576 ] || { echo "XSPR_D0 0..3, XSPR_CAP 0..4096, XSPR_MHC 0..1048576"; exit 1; }
[ $SPN -le 1048576 ] && [ $SPK -le 64 ] || { echo "XSPR_SPN 0..1048576, XSPR_SPK 0..64"; exit 1; }
CTR=$HD/../i9b/i9bctr.py                                                                            # i9b's (INFO only), read only
NCMB=$HD/../nc/ncmb.py                                                                              # nc's, read only (the fallback only)
MBT2=$HD/../harness/mbt2.py; NCMB2=$HD/../harness/ncmb2.py; C8MIX=$HD/../harness/c8mix.py          # the harness (read only)
DAH=$HD/../da/daheld.py; DAM=$HD/damark.py                                                          # daheld.py imported by damark.py (read only)
DQ=$QA/deepq.py; MBP=$D/mb.py; TESTS=$D/ds41_tests.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
T00=$(date +%s); NV=0
xlog() { case "$*" in *VIRHE*) NV=$((NV + 1)) ;; esac; log "RESULT xspr $*"; }
elapsed() { echo $(( ($(date +%s) - T00) / 60 )); }
# ---------------------------------------------------------------- the start (= stack5's + RADIANCE_DS41_MHC_FUSE=0; KVR off)
GPTAB=q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422
SIDECAR=dspark-conf.f32
LEVENV="RADIANCE_DS41_GG_PF=1 RADIANCE_DS41_GG_PF_TABLE=$GPTAB RADIANCE_DS41_F32_PF=25 RADIANCE_DS41_ARENA_DECODE_ROWS=8,48
  RADIANCE_DS41_ROUTER_BF16=1 RADIANCE_DS41_EMB_RAM=1 RADIANCE_PLACE_SHADOW_SLOTS=16 RADIANCE_XPREFETCH=0 RADIANCE_DS41_DSPARK_WALK=1
  RADIANCE_DS41_ARGATHER_MAX=48 RADIANCE_DS41_QNFOLD=1 RADIANCE_DS41_SWFOLD=1 RADIANCE_DRAFT_CTXSKIP=0 RADIANCE_DS41_DSPARK_CONF=/models/$SIDECAR
  RADIANCE_DRAFT_LEND=0 RADIANCE_DRAFT_PROBE_BACKOFF=0 RADIANCE_DRAFT_LEND_BORROW=0 RADIANCE_DRAFT_SAMPLE=0 RADIANCE_WEIGHT_SIDECAR=/models/$SIDE
  RADIANCE_DS41_BI=0 RADIANCE_DRAFT_CTXSKIP_SLOTS=129 RADIANCE_DRAFT_LEND_WEIGHTS=0 RADIANCE_DS41_PSK=0
  RADIANCE_LOG_STEPS=1 RADIANCE_DS41_ENGDD=0 RADIANCE_PLACE_HB_RANKS=$HBRANKS
  RADIANCE_DS41_MHC_FUSE=0"
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
LITS="$LITS|now,0;,RADIANCE_DS41_PSK_L20):,layer,20,split,under,lever,PSK"
LITS="$LITS|now,0;,RADIANCE_DS41_PSK_CR):,a,planned,prefix,hit,with,no,snapshot,at,its,grid,point"
LITS="$LITS|lever,SA,/,HB,(ds41-sahb),set,up|stage_short,can,arm,the,prefill,stager,for,a,step,of,1025,..,2112,tokens|place_hb,can,reserve,on,rank,1"
# + lever M1's declare line (int9b; KVR off here: its lines are not looked for, readback checks they are absent)
LITS="$LITS|lever,M1,(mhc_fuse),declared"
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": 8, "drafter_mx": 1, "gg_pf": 1, "f32_pf": 25, "dspark_walk": 1, "ds41_argather": 48, "idx_split": 1, "ds41_qnfold": 1, "ds41_swfold": 1, "draft_ctxskip": 1, "dspark_conf": 1, "dspark_conf_rows": 1, "draft_lend": 1, "draft_probe_backoff": 3, "draft_hybrid": 1}'
FIX=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update({"draft_lend_hyst": 0, "arena_d0": 0, "draft_lend_w": 0, "draft_occ_k": int(sys.argv[2]), "ds41_psk_k": int(sys.argv[3])}); print(json.dumps(d))' "$FIX5" "$OCCK" "$PSKK") || { echo "pin JSON"; exit 1; }
S5J="\"ds41_bi\": 2, \"draft_sample\": $DSA, \"draft_occ\": $OCC, \"place_lf\": 0, \"sched_bi_chunk\": $BIC, \"sched_noa\": $NOA, \"ds41_psk_l20\": 1, \"dspark_pfctx\": 1, \"ds41_engdd\": 1"
# ---------------------------------------------------------------- the base pin (S5T, draft_sample 0, SP pinned, M1 on, IX off) and the arms
SPJ=", \"stage_full_min\": $SPN, \"stage_full_pick\": $SPK"; [ $SW = 1 ] || SPJ=""        # XSPR_SW=0: the mock has no int9 switch
TJ="{$S5J, \"ds41_psk\": 4, \"ds41_psk_cr\": 1, \"stage_short\": 0, \"stage_pick\": 0, \"place_hb\": 0$SPJ}"   # = stack5's armj T, draft_sample 0, + SP
MJ="{\"mhc_fuse\": $MHC, \"ix_lds\": 0, \"ix_ni\": 0, \"draft_depth0\": $D0, \"sched_row_cap\": $CAP}"; [ $SW = 1 ] || MJ='{}'   # i9b's "M" arm (M1 on, IX-LDSR2 / IX-NI off) + draft_depth0 3 + CAP48 (int10)
armj() {   # armj B|C: the arm's ds41_arcoded (JSON) on top of the base; {} under XSPR_SW=0 (the mock)
  [ $SW = 1 ] || { echo '{}'; return 0; }
  case $1 in
    B) echo '{"ds41_arcoded": 0}' ;;
    C) echo "{\"ds41_arcoded\": $ARCN}" ;;
  esac
}
NAMEOF() { case $1 in B) echo "B (plain wire)" ;; C) echo "C (coded wire >= $ARCN rows)" ;; esac; }
pinj() {   # pinj ARM [EXTRAJSON]: the fixed pin + the base (S5T, draft_sample 0, SP, M1 on, IX off, XSPR_BASEX) + the arm + extra as one POST body
  local x=${2:-} bx=$BASEX; [ -n "$x" ] || x='{}'; [ -n "$bx" ] || bx='{}'
  python3 -c 'import json,sys
d = {}
for a in sys.argv[1:]: d.update(json.loads(a))
print(json.dumps(d))' "$FIX" "$TJ" "$MJ" "$bx" "$(armj $1)" "$x"
}
DESC="NST 5 stack5 start (stack4's + RADIANCE_LOG_STEPS=1, E1 declared off, HB ranks $HBRANKS) + RADIANCE_DS41_MHC_FUSE=0 (M1 declared) | KVR OFF (RADIANCE_DS41_KV_REPLICATED unset), SP off | base pin = stack5's S5T with draft_sample 0 + SP pinned (stage_full_min $SPN, stage_full_pick $SPK) + M1 on / IX off / draft_depth0 $D0 (int10: depth 0 by occupancy only) / sched_row_cap $CAP (CAP48) $MJ + the fixed pin (ds41_psk_k $PSKK; draft_lend 1)$([ -n "$BASEX" ] && echo " + XSPR_BASEX $BASEX") | NO ARMS: the pin with ds41_arcoded 0 held throughout"
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
  RADIANCE_DS41_PSK RADIANCE_PLACE_LF RADIANCE_PLACE_LF_DECAY RADIANCE_LOG_STEPS RADIANCE_DS41_ENGDD RADIANCE_PLACE_HB_RANKS
  RADIANCE_PLACE_HB_DECAY RADIANCE_DS41_PSK_L20 RADIANCE_DS41_DSPARK_PFCTX RADIANCE_DS41_PSK_CR RADIANCE_SCHED_BI_CHUNK
  RADIANCE_SCHED_BI_GRID RADIANCE_PROFILE_EVERY RADIANCE_DS41_KV_REPLICATED RADIANCE_DEBUG_TIER_VERIFY
  RADIANCE_DS41_MHC_FUSE RADIANCE_DS41_IX_LDS RADIANCE_DS41_IX_NI RADIANCE_DS41_ARCODED RADIANCE_DECODE_ROW_CAP"
# ---------------------------------------------------------------- copied from stack5/dsm-stack5.sh (63649414; tag xs5 -> xspr): stackenv .. gate1 (modes: + nc/dsm-ncprof-p.sh's rule-w check under --profile-ops)
# (vramstart / gate1 show stack5's s5-n5 start beside this one)
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
capture() {   # capture FILE: the engine log with the daemon's receipt times (ncsteps.py reads them), gzipped (never written raw)
  run "docker logs -t $CN 2>&1 | gzip -1 > $1"
  [ "$DRY" = 1 ] || xlog "engine log $(basename $1): $(du -m $1 | cut -f1) MB gz, $(zcat $1 | grep -cE ' step [0-9]+ (prefill|decode|MIXED) n_seq=') step lines; $(dfree) MB free"
}
thermline() { type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"; }
dfree() { df -BM --output=avail $HOME 2>/dev/null | tail -1 | tr -dc 0-9; }
diskok() {   # diskok TAG: >= DISKMIN MB free on / before the start (sahb's)
  local f; f=$(dfree)
  if [ "$DRY" = 1 ]; then echo "+ disk before $1: ${f:-?} MB free on / (the start needs >= $DISKMIN MB) -- $([ "${f:-0}" -ge $DISKMIN ] && echo OK now || echo "LOW now: the run would refuse the start")"; return 0; fi
  [ "${f:-0}" -ge $DISKMIN ] && return 0
  xlog "$1: VIRHE only ${f:-?} MB free on / (need >= $DISKMIN: the 10 GB guard + this item's logs) -- not started"; return 1
}
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xspr $1 warm-up identity vs $2"; return 0; }
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
modes() {   # modes TAG: the start words in the container, mover 0 kept free x4, rule-w, K1, ARGATHER 48, step-log lines, every literal of LITS -> 0, else 1
  [ "$DRY" = 1 ] && { echo "+ modes $1: LEVENV in the container's env; mover '0 more kept free' on 4 ranks; rule-w in every recorded pass; K1 folds declared =1; ARGATHER 48-row note; step-log lines (RADIANCE_LOG_STEPS=1); literals [$(echo "$LITS" | tr ',' ' ')]"; return 0; }
  local CE kv bad="" ok=1 NW NR KL AL nz lit found="" miss=""
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in RADIANCE_PLACE_TURNOVER=0 $LEVENV; do echo "$CE" | grep -qxF "$kv" || bad="$bad ${kv%%=*}"; done
  [ -z "$bad" ] || { xlog "$1: VIRHE start words not in the container:$bad"; ok=0; }
  nz=$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots \([0-9]+ shadow, 0 more kept free')
  [ "$nz" = 4 ] || { xlog "$1: VIRHE turnover 0 asked but $nz of 4 ranks keep 0 slots free"; ok=0; }
  NW=$(docker logs $CN 2>&1 | grep -c 'the acquire rule plays .*RADIANCE_AQL_ACQ=rule-w)'); NR=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_AQL_ACQ=rule)')
  if [ -n "${PROFEVERY:-}" ]; then   # --profile-ops records no pass (ctx.cpp tape_on_ needs !profiling_): env rule-w, no plain rule line (nc/dsm-ncprof-p.sh)
    if echo "$CE" | grep -qx "RADIANCE_AQL_ACQ=rule-w" && [ "$NR" = 0 ]; then xlog "$1: rule-w replaced by the profile sync (not a speed): --profile-ops records no pass, RADIANCE_AQL_ACQ=rule-w in the container env, no plain rule line ($NW / $NR)"
    else xlog "$1: VIRHE RADIANCE_AQL_ACQ=rule-w not in the container env (or a plain rule line) under --profile-ops ($NW / $NR)"; ok=0; fi
  else
    [ "$NW" -gt 0 ] && [ "$NR" = 0 ] || { xlog "$1: VIRHE rule-w is not what the recorded passes play ($NW / $NR)"; ok=0; }
  fi
  KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-500)
  case "$KL" in *"ds41_kvfold=1"*"ds41_ofold=1"*"ds41_onefill=1"*) ;; *) xlog "$1: VIRHE the K1 switches do not start at 1: '${KL}'"; ok=0 ;; esac
  AL=$(docker logs $CN 2>&1 | grep -m1 'the MoE gather and its all-reduce as one kernel' | sed 's/^radiance: //' | cut -c1-300)
  case "$AL" in *"up to 48 tokens a step (RADIANCE_DS41_ARGATHER, RADIANCE_DS41_ARGATHER_MAX"*) ;; *) xlog "$1: VIRHE the ARGATHER note is not the 48-row one: '${AL}'"; ok=0 ;; esac
  [ "$(docker logs $CN 2>&1 | grep -cE ' step [0-9]+ (prefill|decode|MIXED) n_seq=')" -gt 0 ] || { xlog "$1: VIRHE RADIANCE_LOG_STEPS=1 but no step line in the log"; ok=0; }
  IFS='|' read -ra lit <<< "$LITS"
  for l in "${lit[@]}"; do [ -n "$l" ] || continue
    if docker logs $CN 2>&1 | grep -qF -- "${l//,/ }"; then found="$found ${l%%,*}"; else miss="$miss | ${l//,/ }"; fi; done
  xlog "$1 modes: ${#lit[@]} literal(s), found [$(echo $found)]$([ -n "$miss" ] && echo " -- VIRHE missing:$miss") | $(docker logs $CN 2>&1 | grep -E "lever PSK|lever E1|lever SA / HB|lever BI declared|lever SD|lever KVR|lever M1|held once a slot|weight sidecar|lever Z0|activation arena|ring of|draft lend:|draft weight lend|no level for|prefill staging:" | sed 's/^radiance: //; s/^[A-Z] *//; s/^[A-Z]\[[0-9]\] //' | cut -c1-260 | sort -u | tr '\n' '|' | cut -c1-3600)"
  [ -z "$miss" ] || ok=0
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
vramstart() {
  [ "$DRY" = 1 ] && { echo "+ GET /stats > $O/$1-stats0.json; RESULT xspr $1 VRAM at start (experts per tier; vs stack4 s4-n5)"; return 0; }
  curl -s -m 30 $URL/stats > $O/$1-stats0.json 2>/dev/null
  xlog "$1 VRAM at start: experts $(python3 -c 'import json,sys
for p in sys.argv[1:]:
    try: s = json.load(open(p)); print({x.get("name"): x.get("units") for x in s.get("experts", {}).get("tiers", [])}, end=" ")
    except Exception as e: print("n/a", end=" ")' $O/$1-stats0.json $RS5/s5-n5-stats0.json 2>&1 | cut -c1-320) (this start, then stack5 s5-n5) | $(docker logs $CN 2>&1 | grep -E '^I\[0\] mover:|checkpoint slots|weight sidecar:' | sed 's/^radiance: //; s/^[A-Z]\[0\] //; s/^[A-Z] *//' | cut -c1-220 | sort -u | tr '\n' '|' | cut -c1-1500)"
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
arm() {   # arm ARM: the pin + the arm + epoch_bump, a nudge, the switches READ BACK (polled <= 30 s) -> 0, else VIRHE + 1
  local g r i
  fn4set "$(pinj $1 '{"epoch_bump": 1}')"; nudge
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4: every switch of arm $1 applied, nothing pending (polled <= 30 s, a nudge a try)"; return 0; }
  for i in $(seq 1 30); do
    g=$(fn4get)
    r=$(python3 -c 'import json,sys
g, w = json.loads(sys.argv[1]), json.loads(sys.argv[2])
bad = {k: g["switches"].get(k) for k, v in w.items() if g["switches"].get(k) != v}
print("OK" if not bad and not g["pending"] else "FAIL %s pending %s" % (bad, g["pending"]))' "$g" "$(pinj $1)" 2>&1 | cut -c1-400)
    case "$r" in OK*) return 0 ;; esac
    sleep 1; nudge
  done
  xlog "arm $1: VIRHE the pin is not applied after 30 s ($r) -- no reading for this arm here"; ARMOK=0; return 1
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
pskstat() {   # the live lever counters (cumulative, /stats): PSK, PSK-L20, PSK-CR, PSK-DS, E1
  [ "$DRY" = 1 ] && { echo "(dry)"; return 0; }
  curl -s -m 20 $URL/stats 2>/dev/null | python3 -c 'import json,sys
s = json.load(sys.stdin); p = s.get("psk") or {}; e = s.get("engdd") or {}; f = s.get("dspark_pfctx") or {}
print("psk mode %s k %s latched %s planned %s cut %s rows %s layers %s mismatch %s | l20 %s / %s | cr %s hits %s not_needed %s fb %s tail %s | pfctx %s / %s | engdd %s / %s" % (
    p.get("mode"), p.get("k"), p.get("latched"), p.get("planned_steps"), p.get("cut_passes"), (p.get("rows_skipped_l21_29") or 0) + (p.get("rows_skipped_l30_39") or 0),
    p.get("layers_skipped"), p.get("plan_mismatch"), p.get("l20_cut_passes"), p.get("l20_rows_skipped"), p.get("cr"), p.get("cr_hits"), p.get("cr_not_needed"),
    p.get("cr_fallbacks"), p.get("tail_only_latched"), f.get("passes"), f.get("rows_skipped"), e.get("passes"), e.get("rows_deduped")))' 2>/dev/null | cut -c1-500
}
ssnap() {   # ssnap NAME: the full /stats and GET /admin/fn4 (the counter guard's snapshot)
  [ "$DRY" = 1 ] && { echo "+ GET /stats > $O/st-$1.json; GET /admin/fn4 > $O/fn-$1.json"; return 0; }
  curl -s -m 30 $URL/stats > $O/st-$1.json 2>/dev/null; curl -s -m 10 $URL/admin/fn4 > $O/fn-$1.json 2>/dev/null
}
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
gate1() {   # restart + ONE check; over the limit logged (in-process A/B: a slow state shows in every arm)
  local tag=$1 lim=$2 ms st=fast
  restart $tag || { keeplog $tag-fail; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && return 0
  awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}' && { st="SLOW (not retried)"; keeplog $tag-slow; }
  echo "$st $ms" > $O/guard-$tag.txt
  xlog "guard $tag: $st, 1-user ${ms} ms/step (limit $lim; stack5 s5-n5 $(cat $RS5/guard-s5-n5.txt 2>/dev/null))"
}
# ---------------------------------------------------------------- copied from stack5/dsm-stack5.sh (63649414): alone, cmpx
alone() {   # alone TAG ARM NAME: pin read back, prefix-cache reset, greedyset.py's 20 prompts at c1 ALONE, temperature 0 (20 of 20 or a VIRHE)
  local tag=$1 t0=$(date +%s) n
  arm $2 || return 1
  pcreset $tag-$3
  ssnap $tag-$3-0
  run "(cd $D && timeout 900 python3 $GS $URL $O/$tag-$3.jsonl --conc 1 --max-tokens $MT > $O/$tag-$3.txt 2>&1)"
  ssnap $tag-$3-1
  [ "$DRY" = 1 ] && { echo "+ RESULT xspr $tag $3: greedy set c1 alone under $(NAMEOF $2) $(armj $2): 20 of 20 answered (else a request-error line); counters INFO"; return 0; }
  n=$(python3 -c 'import json,sys; print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l).get("text") is not None))' $O/$tag-$3.jsonl 2>/dev/null || echo 0)
  xlog "$tag $3 (greedy set c1 ALONE, temperature 0, $MT tok, arm $(NAMEOF $2) $(armj $2)): $n of 20 answered$([ "$n" = 20 ] || echo " -- VIRHE request errors"), $(tail -1 $O/$tag-$3.txt | cut -c1-160), $(( $(date +%s) - t0 )) s | INFO counters: $(ctrinfo $tag-$3-0 $tag-$3-1)"
}
cmpx() {   # cmpx TAG NAME A B: loadid.py cmp (byte for byte + the first differing char / token) -> identity.txt
  local r
  [ "$DRY" = 1 ] && { echo "+ python3 -I $LD cmp --url $URL $3 $4 --json $O/$1-cmp-$2.json"; return 0; }
  if [ ! -s "$3" ] || [ ! -s "$4" ]; then xlog "$1 $2: not compared (missing $3 or $4)"; echo "$1-$2 not compared" >> $O/identity.txt; return 0; fi
  r=$(timeout 300 python3 -I $LD cmp --url $URL --json $O/$1-cmp-$2.json --label "$1 $2" "$3" "$4" 2>&1 | grep -m1 '^CMPLINE ' | cut -c9-)
  idrec "$1-$2" "${r:-VIRHE compare failed}" "$(basename $3) vs $(basename $4)"
}
# ---------------------------------------------------------------- arc/dsm-arc.sh's own (copied): budok, ctrinfo, arcnow, capd, drafted, readback
budok() {   # budok NAME EST_MIN [LIMIT]: 0 = run the section; 1 = skip it (the time now + its estimate would pass the budget), logged (sp's)
  local lim=${3:-$BUDGET}
  [ "$DRY" = 1 ] && { echo "+ budget: $1 runs if elapsed + ~$2 min <= $lim min, else skipped (logged)"; return 0; }
  [ $(( $(elapsed) + $2 )) -le $lim ] && return 0
  xlog "$1 SKIPPED (budget): $(elapsed) min elapsed + ~$2 min > $lim min"; return 1
}
ctrinfo() {   # ctrinfo A B: every counter delta between the snapshots (i9b/i9bctr.py, no need / forbid: INFORMATION, never a VIRHE)
  [ "$DRY" = 1 ] && { echo "(dry: python3 -I -B $CTR guard --st0 st-$1 --st1 st-$2 --fn0 fn-$1 --fn1 fn-$2, printed as INFO)"; return 0; }
  timeout 60 python3 -I -B $CTR guard --st0 $O/st-$1.json --st1 $O/st-$2.json --fn0 $O/fn-$1.json --fn1 $O/fn-$2.json 2>&1 | tail -1 | sed 's/^OK | //' | cut -c1-900
}
arcnow() {   # arcnow: GET /admin/fn4's ds41_arcoded + the int9b switches + draft_depth0 and the "arc" counters, one line
  [ "$DRY" = 1 ] && { echo "(dry)"; return 0; }
  fn4get | python3 -c 'import json,sys
d = json.load(sys.stdin); w = d.get("switches") or {}
print("ds41_arcoded %s sched_row_cap %s mhc_fuse %s ix_lds %s ix_ni %s draft_depth0 %s | arc %s | cap %s" % (w.get("ds41_arcoded"), w.get("sched_row_cap"), w.get("mhc_fuse"), w.get("ix_lds"), w.get("ix_ni"), w.get("draft_depth0"), json.dumps(d.get("arc")), json.dumps(d.get("cap"))))' 2>/dev/null | cut -c1-600
}
capd() {   # capd A B -> one line: the CAP48 "cap" deltas between GET /admin/fn4 snapshots A and B (in_force: B's)
  [ "$DRY" = 1 ] && { echo "(dry)"; return 0; }
  python3 -c 'import json,sys
def rd(p):
    try: return json.load(open(p)).get("cap")
    except Exception: return None
a, b = rd(sys.argv[1]) or {}, rd(sys.argv[2])
if b is None: print("no \"cap\" in GET /admin/fn4"); sys.exit(0)
d = {k: (b.get(k, 0) - a.get(k, 0)) for k in ("steps_capped", "drafts_trimmed", "ahead_capped", "steps_over", "steps_gt48", "decode_steps") if isinstance(b.get(k), (int, float))}
print("cap in_force %s, max_rows %s | %s" % (b.get("in_force"), b.get("max_rows"), ", ".join("%s +%s" % kv for kv in d.items())))' $O/fn-$1.json $O/fn-$2.json 2>/dev/null | cut -c1-400
}
drafted() {   # drafted TAG: the DSpark drafter and the adaptive depth are up (stack4's check)
  [ "$DRY" = 1 ] && return 0
  local ok=1
  docker logs $CN 2>&1 | grep -q "DSpark drafter" || { xlog "$1: VIRHE no DSpark drafter line"; ok=0; }
  docker logs $CN 2>&1 | grep -q "draft depth: adaptive" || { xlog "$1: VIRHE no adaptive draft depth line"; ok=0; }
  [ $ok = 1 ] || keeplog $1-nodraft
  [ $ok = 1 ]
}
readback() {   # readback TAG: the start's words read back -- MHC_FUSE=0 in the env, M1's declare line; KVR and ARCODED NOT in the env, no KVR declare line
  [ "$DRY" = 1 ] && { echo "+ readback $1: RADIANCE_DS41_MHC_FUSE=0 in the container env and the 'lever M1 (mhc_fuse) declared' line; RADIANCE_DS41_KV_REPLICATED and RADIANCE_DS41_ARCODED NOT in the env, no 'lever KVR declared' line (KVR off; ARC off at start)"; return 0; }
  local ce e kv ac cp d m ok=1
  ce=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  e=$(echo "$ce" | grep -cx 'RADIANCE_DS41_MHC_FUSE=0'); kv=$(echo "$ce" | grep -c '^RADIANCE_DS41_KV_REPLICATED='); ac=$(echo "$ce" | grep -c '^RADIANCE_DS41_ARCODED='); cp=$(echo "$ce" | grep -c '^RADIANCE_DECODE_ROW_CAP=.')
  d=$(docker logs $CN 2>&1 | grep -m1 'lever KVR declared' | sed 's/^radiance: //' | cut -c1-200)
  m=$(docker logs $CN 2>&1 | grep -m1 'lever M1 (mhc_fuse) declared' | sed 's/^radiance: //; s/^[A-Z]\[[0-9]\] //; s/^[A-Z] *//' | cut -c1-400)
  [ "$e" = 1 ] && [ "$kv" = 0 ] && [ "$ac" = 0 ] && [ "$cp" = 0 ] && [ -z "$d" ] && [ -n "$m" ] || ok=0
  xlog "$1 read back: env MHC_FUSE=0 $e of 1, KV_REPLICATED $kv (want 0), ARCODED $ac (want 0), DECODE_ROW_CAP $cp (want 0: CAP48 comes by fn4) | KVR declare line: ${d:-none (off)} | M1: ${m:-NONE}$([ $ok = 1 ] || echo " -- VIRHE the start's words are not read back")"
  [ $ok = 1 ]
}
pinset() {   # pinset TAG: start values (read only, INFO + the int10 switches 0 at start), then the pin ONCE (+ epoch_bump), read back
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 at start (INFO: the switches, \"arc\", \"cap\"); POST the pin once ($(pinj B | cut -c1-160)...) + epoch_bump, a nudge, every switch READ BACK (polled <= 30 s); no arm after it"; return 0; }
  local s0 g
  s0=$(fn4get)
  arm B || return 1
  g=$(fn4get)
  xlog "$1 pin: at start $(echo "$s0" | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print("switches %s | epoch %s" % (json.dumps({k: w.get(k) for k in ("ds41_arcoded", "sched_row_cap", "mhc_fuse", "ix_lds", "ix_ni", "draft_depth0", "draft_lend", "draft_occ", "stage_full_min", "draft_sample", "ds41_psk", "ds41_bi")}), d.get("epoch")))' 2>/dev/null | cut -c1-700) | after the pin (read back): $(echo "$g" | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print("switches %s | epoch %s" % (json.dumps({k: w.get(k) for k in ("ds41_arcoded", "sched_row_cap", "mhc_fuse", "ix_lds", "ix_ni", "draft_depth0", "draft_lend", "draft_occ", "stage_full_min", "draft_sample", "ds41_psk", "ds41_bi")}), d.get("epoch")))' 2>/dev/null | cut -c1-700)"
}
blk() {   # blk I N: one mb cN block at T 0 on the harness's fixed, seeded schedule (pattern 1xN, gens XSPR_GENSN); the MB line, TTFT p50 / p90
          # over the window's requests (the taps), the /metrics deltas, the counters (INFO); 0 request errors
  local i=$1 n=$2 nm=c$2-$1 base=$O/blk/c$2-$1 t0 j gens tool
  case $n in 1) gens=$GENS1 ;; 4) gens=$GENS4 ;; 8) gens=$GENS8 ;; esac
  budok "block $nm (c$n)" $(( n == 8 ? 2 : 1 )) || return 0
  ssnap $nm-0; msnap $nm-0
  t0=$(date +%s)
  if [ -s $NCMB2 ] && [ -s $MBT2 ]; then
    tool=ncmb2
    run "(cd $D && timeout $((DUR + WARM + 900)) python3 -I -B $NCMB2 --mb $MBP --mode phase --n $n --temperature 0 --tag xspr-$nm --dur $DUR --warm $WARM --reqs $base-reqs.jsonl --stats-prefix $base-st --samples $base-samp.jsonl --sched fixed --pattern 1x$n --seed 1 --window gens --gens $gens --edge 0.5 --ignore-eos 1 > $base.txt 2>&1)"
  else
    tool=ncmb
    run "(cd $D && timeout $((DUR + WARM + 900)) python3 -I -B $NCMB --mb $MBP --mode phase --n $n --temperature 0 --tag xspr-$nm --dur $DUR --warm $WARM --reqs $base-reqs.jsonl --stats-prefix $base-st --samples $base-samp.jsonl > $base.txt 2>&1)"
  fi
  ssnap $nm-1; msnap $nm-1
  [ "$DRY" = 1 ] && { echo "+ RESULT xspr block $nm (c$n, T 0 via $([ -s $NCMB2 ] && echo "harness/ncmb2.py: fixed schedule 1x$n seed 1, window gens $gens ($((gens * n)) newcomer steps), edge 0.5, ignore_eos" || echo "nc/ncmb.py: the old loop")): agg tok/s, per-stream p50, TTFT p50 / p90, ms / tok a decode step, acceptance, decode share, the client mix, errors 0 (else a request-error line) -> $O/blocks.jsonl"; return 0; }
  j=$(grep -m1 '^MB ' $base.txt | cut -c4-)
  [ -n "$j" ] || { xlog "block $nm: VIRHE request error: no MB line ($(tail -3 $base.txt | tr '\n' ' ' | cut -c1-300))"; return 1; }
  python3 -c 'import json,sys
d = json.loads(sys.argv[1]); m = json.loads(sys.argv[5] or "{}")
tt = []; mk = []
try:
    for l in open(sys.argv[4]):
        r = json.loads(l)
        if r.get("ev") == "mark": mk.append(r["t"])
        elif r.get("ev") == "done" and r.get("ttft") is not None and not r.get("err"): tt.append((r.get("t_first") or 0, r["ttft"]))
except OSError: pass
w = [x for t, x in tt if len(mk) >= 2 and mk[0] <= t <= mk[-1]] or [x for _, x in tt]
w.sort()
q = lambda p: round(w[min(len(w) - 1, int(p * (len(w) - 1) + 0.5))], 3) if w else None
print(json.dumps(dict(name=sys.argv[2], phase=int(sys.argv[3]), tool=sys.argv[6], agg_dec_tps=d.get("agg_dec_tps"), per_stream_dec_p50=d.get("per_stream_dec_p50"),
      ttft_p50=d.get("ttft_p50"), ttft_p50_taps=q(0.5), ttft_p90_taps=q(0.9), ttft_n=len(w), ms_decode_step=d.get("ms_decode_step"), tok_per_decode_step=d.get("tok_per_decode_step"),
      acceptance=d.get("acceptance"), decode_share=d.get("decode_share"), requests=d.get("requests"), errors=d.get("errors"), window_s=d.get("window_s"),
      mix_client=d.get("mix_client"), window_timeout=d.get("window_timeout"), metrics=m)))' "$j" "$nm" "$n" "$base-reqs.jsonl" "$(mdelta $nm-0 $nm-1)" "$tool" >> $O/blocks.jsonl
  xlog "block $nm (c$n, T 0 via $tool, gens $gens, $(( $(date +%s) - t0 )) s): $(tail -1 $O/blocks.jsonl | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
print("agg %s tok/s, per-stream p50 %s, TTFT p50 %s s (taps p50 %s / p90 %s, n %s), %s ms / %s tok a decode step, acceptance %s, decode_share %s, window %s s, %s requests, %s errors%s | client mix %s" % (
    d.get("agg_dec_tps"), d.get("per_stream_dec_p50"), d.get("ttft_p50"), d.get("ttft_p50_taps"), d.get("ttft_p90_taps"), d.get("ttft_n"), d.get("ms_decode_step"), d.get("tok_per_decode_step"),
    d.get("acceptance"), d.get("decode_share"), d.get("window_s"), d.get("requests"), d.get("errors"),
    "" if d.get("errors") == 0 and (d.get("requests") or 0) > 0 else " -- VIRHE request errors", json.dumps(d.get("mix_client"))[:200]))' 2>&1 | cut -c1-800) | INFO $(capd $nm-0 $nm-1) | INFO counters: $(ctrinfo $nm-0 $nm-1 | cut -c1-400)"
}
blocks() {   # the blocks in XSPR_ORDER, one engine-log capture, then c8mix.py block per block (the mix line; INFO)
  local p i4=0 i1=0 i8=0 k b
  [ -s $NCMB2 ] && [ -s $MBT2 ] && xlog "blocks: harness/mbt2.py through harness/ncmb2.py (fixed, seeded schedule, pattern 1xN, seed 1, window gens: c1 $GENS1 / c4 $GENS4 / c8 $GENS8 generations, edge 0.5, ignore_eos); order [$ORDER]; mbt2.py $(md5sum < $MBT2 | cut -c1-8), ncmb2.py $(md5sum < $NCMB2 | cut -c1-8), c8mix.py $(md5sum < $C8MIX 2>/dev/null | cut -c1-8)" \
    || xlog "blocks: harness/mbt2.py / ncmb2.py NOT present at run time -- nc/ncmb.py runs the blocks (old loop)"
  for p in $ORDER; do
    stopfile && return 0
    case $p in 1) i1=$((i1 + 1)); k=$i1 ;; 4) i4=$((i4 + 1)); k=$i4 ;; 8) i8=$((i8 + 1)); k=$i8 ;; esac
    blk $k $p
  done
  capture $O/blk-engine.log.gz
  [ "$DRY" = 1 ] && { echo "+ per block: python3 -I -B $C8MIX block --log $O/blk-engine.log.gz --reqs $O/blk/<b>-reqs.jsonl --tag xspr-<b> --mbline $O/blk/<b>.txt --label <b> --json $O/blk/<b>-mix.json (the mix line, INFO)"; return 0; }
  [ -s $C8MIX ] || { xlog "blocks: c8mix.py absent -- no mix lines"; return 0; }
  for b in $(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    if l.strip(): print(json.loads(l)["name"])' $O/blocks.jsonl 2>/dev/null); do
    [ -s $O/blk/$b-reqs.jsonl ] || continue
    timeout 300 python3 -I -B $C8MIX block --log $O/blk-engine.log.gz --reqs $O/blk/$b-reqs.jsonl --tag xspr-$b --mbline $O/blk/$b.txt --label $b --json $O/blk/$b-mix.json > $O/blk/$b-mix.out 2>&1
    xlog "mix $b (INFO): $(grep -m1 '^C8MIX ' $O/blk/$b-mix.out | cut -c7- | sed 's/ -- VIRHE / -- c8mix guard (INFO): /' | cut -c1-700)$(grep -q '^C8MIX ' $O/blk/$b-mix.out || echo "no C8MIX line ($(tail -1 $O/blk/$b-mix.out | cut -c1-200))")"
  done
}
dalive() {   # the DA harness's live self-check (read only): daheld.py's own marker lines on this server + the usable-prefix count over the corpus
  run "timeout 120 python3 -I -B $DAM check --daheld $DAH --url $URL --corpus $CORPUS > $O/damark.txt 2>&1"
  [ "$DRY" = 1 ] && { echo "+ RESULT xspr da-marker: <marker> -> N of $(grep -c . $CORPUS 2>/dev/null) docs (damark.py check: daheld.py's marker lines executed against this server, its prefix loop over $(basename $CORPUS); no /completion)"; return 0; }
  local j
  j=$(grep -m1 '^DAMARK ' $O/damark.txt | cut -c8-)
  [ -n "$j" ] || { xlog "da-marker: VIRHE no DAMARK line ($(tail -3 $O/damark.txt | tr '\n' ' ' | cut -c1-400))"; return 1; }
  xlog "da-marker: $(echo "$j" | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
print("%s -> %s of %s docs | skipped %s%s | prompt lengths %s..%s | daheld.py %s, its marker lines executed (%s lines)" % (d["marker"], d["usable"], d["docs"], d["skipped"],
      (" (first: %s)" % ", ".join(str(x) for x in d["first_skips"])) if d["first_skips"] else "", d.get("prompt_len_min"), d.get("prompt_len_max"), sys.argv[1], d.get("derivation_lines")))' "$(md5sum < $DAH | cut -c1-8)" 2>&1 | cut -c1-600)"
}
summary() {   # per phase: mean and max-min spread (% of the mean) of each metric; ONE verdict line a phase on agg tok/s
  [ "$DRY" = 1 ] && { echo "+ RESULT xspr SPREAD lines: per phase c4 / c1 / c8 the blocks' agg tok/s, per-stream p50, TTFT p50 / p90, ms / tok a decode step, acceptance, decode share -- mean and max-min spread %; VERDICT 'cN spread X % (one block per arm OK if <= ~1 %)'; SUMMARY"; return 0; }
  python3 - "$O" > $O/summary.txt 2>&1 <<'PY'
import json, os, statistics, sys
o = sys.argv[1]
try: rows = [json.loads(l) for l in open(os.path.join(o, "blocks.jsonl")) if l.strip()]
except OSError: rows = []
rows = [r for r in rows if r.get("errors") == 0 and (r.get("requests") or 0) > 0]
M = [("agg_dec_tps", "agg tok/s", 1), ("per_stream_dec_p50", "per-stream p50", 1), ("ttft_p50_taps", "TTFT p50 s", 3), ("ttft_p90_taps", "TTFT p90 s", 3),
     ("ms_decode_step", "ms a decode step", 2), ("tok_per_decode_step", "tok a decode step", 2), ("acceptance", "acceptance", 3), ("decode_share", "decode share", 3)]
def st(v):
    v = [x for x in v if isinstance(x, (int, float))]
    if not v: return None, None
    m = statistics.mean(v)
    return m, (100.0 * (max(v) - min(v)) / m if len(v) >= 2 and m else None)
out, sm = [], []
for n in (4, 1, 8):
    x = [r for r in rows if r["phase"] == n]
    if not x: out.append("SPREAD c%d: no block" % n); continue
    parts = []
    for k, nm, d in M:
        m, s = st([r.get(k) for r in x])
        parts.append("%s mean %s spread %s" % (nm, "-" if m is None else "%.*f" % (d, m), "-" if s is None else "%.2f %%" % s))
    out.append("SPREAD c%d (%d blocks: %s): %s" % (n, len(x), ", ".join("%s %s" % (r["name"], r.get("agg_dec_tps")) for r in x), " | ".join(parts)))
    m, s = st([r.get("agg_dec_tps") for r in x])
    _, sp = st([r.get("per_stream_dec_p50") for r in x]); _, sms = st([r.get("ms_decode_step") for r in x])
    if s is None: v = "c%d spread: fewer than 2 blocks -- no reading" % n
    else:
        ok = s <= 1.0
        v = "c%d spread %.2f %% (one block per arm OK if <= ~1 %%): %s -- agg tok/s over %d blocks, mean %.1f; per-stream p50 spread %s, ms a decode step spread %s" % (
            n, s, "within" if ok else "NOT within", len(x), m, "-" if sp is None else "%.2f %%" % sp, "-" if sms is None else "%.2f %%" % sms)
    out.append("VERDICT " + v); sm.append(v.split(":")[0] + (": %s" % ("OK" if s is not None and s <= 1.0 else "not OK" if s is not None else "-")))
try:
    dm = [json.loads(l[7:]) for l in open(os.path.join(o, "damark.txt")) if l.startswith("DAMARK ")]
    if dm: out.append("DA marker %s -> %s of %s docs usable" % (dm[0]["marker"], dm[0]["usable"], dm[0]["docs"])); sm.append("da-marker %s of %s" % (dm[0]["usable"], dm[0]["docs"]))
except OSError: pass
for l in out: print(l)
print("SUMMARY " + " || ".join(sm) + " || " + " || ".join(out)[:4000])
PY
  while IFS= read -r l; do case "$l" in SUMMARY*) ;; *) xlog "${l:0:3000}" ;; esac; done < $O/summary.txt
  xlog "summary: $(grep -m1 '^SUMMARY ' $O/summary.txt | cut -c9- | cut -c1-5000)"
}
essay2() {   # 12.11. (coordinator): base1113's own 1u read -- s5blocks' c1 phase = stepprobe "essay,code" 512 at T 0 -- XSPR_EBLK (2) blocks
  local k j
  for k in $(seq 1 ${XSPR_EBLK:-2}); do
    if [ "$DRY" = 1 ]; then echo "+ (cd $D && timeout 600 python3 $D/stepprobe.py essay,code 512 > $O/essay-$k.txt) -> RESULT xspr essay512 $k: the J lines (decode tok/s, ms / tok a decode step, acceptance)"; continue; fi
    (cd $D && timeout 600 python3 $D/stepprobe.py essay,code 512 > $O/essay-$k.txt 2>&1)
    j=$(grep '^J ' $O/essay-$k.txt | cut -c3- | tr '\n' ' ' | cut -c1-900)
    [ -n "$j" ] && xlog "essay512 $k (base1113's c1 read: stepprobe essay,code 512, T 0, 1u; vs base1113 S5T essay 76.9 / 73.8, code 123.8 / 124.2): $j" || xlog "essay512 $k: VIRHE request error ($(tail -2 $O/essay-$k.txt | tr '\n' ' ' | cut -c1-200))"
  done
}
dstart() {   # the start, the pin, the blocks, the DA check
  local tag=spr-n5
  diskok $tag || return 1
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on" PROFEVERY=
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV > $XS2_ENVFILE"; else printf '%s\n' $LEVENV > $XS2_ENVFILE; fi
  xlog "$tag: start (SPREAD; $DESC) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $(pinj B | cut -c1-900) | env: $(env | grep -E '^(RADIANCE_|R4D_)' | grep -vE '(KEY|TOKEN|SECRET)' | sort | tr '\n' ' ' | cut -c1-2400)"
  gate1 $tag 70 || { xlog "$tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  modes $tag || { xlog "$tag: VIRHE the modes are not the start's -> stopped"; down; return 1; }
  drafted $tag || { down; return 1; }
  readback $tag || { keeplog $tag-readback; down; return 1; }
  vramstart $tag
  pinset $tag || { xlog "$tag: VIRHE the pin is not applied -> stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  stopfile || blocks
  stopfile || essay2
  stopfile || { [ $DODA = 1 ] && dalive; }
  capture $O/$tag-engine-full.log.gz
  xlog "$tag: at the end: $(pskstat) | $(arcnow) | $(docker logs --since 6h $CN 2>&1 | grep -cE '^E ') E lines | $(elapsed) min"
  down
  xlog "$tag: done"
}
trap 'down' EXIT
trap 'exit 143' TERM INT
# ---------------------------------------------------------------- main
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19))$([ $SW = 1 ] || echo " | VIRHE XSPR_SW=0: NO mhc_fuse / ix_lds / ix_ni / draft_depth0 3 / stage_full_* / ds41_arcoded / sched_row_cap posted (a mock run only)") | SPREAD start: $DESC | blocks [$ORDER] (c1 gens $GENS1, c4 gens $GENS4, c8 gens $GENS8; dur $DUR warm $WARM; T 0, fixed schedule) | DA live check $([ $DODA = 1 ] && echo on || echo off) | guards: read back + 0 request errors | budget $BUDGET min | no evict"
IMGLITS="B|sched_short_bypass, sched_bypass_total, ds41_psk, ds41_psk_k, ds41_psk_l20, dspark_pfctx,
B|stage_full_min
B|stage_full_pick
B|stage_full_min takes 0 (off) or 1 .. 1048576
B|full_pick_dropped
B|ds41_psk_cr
B|ds41_engdd
B|sched_noa
B|stage_short
B|stage_short takes 0 (off) or 1 .. 65536
B|cr_end_snapshots
B|tail_only_latched
B|lever SA / HB (ds41-sahb) set up, off: stage_short %s; place_hb %s
B|RADIANCE_PLACE_HB_RANKS
B|RADIANCE_LOG_STEPS
B|RADIANCE_PROFILE_EVERY
B|per-op timing, rank
B|n_seq=%lld n_tok=%lld
B|reset_prefix_cache
B|held once a slot
B|radiance:engine_steps_mixed_total
B|radiance:engine_step_seconds_decode_total
A|RADIANCE_DS41_KV_REPLICATED
A|lever KVR declared (RADIANCE_DS41_KV_REPLICATED=1)
A|RADIANCE_DS41_PSK
A|lever PSK declared (RADIANCE_DS41_PSK, fn4 ds41_psk, now
A|lever PSK-L20 (fn4 ds41_psk_l20, now %d; RADIANCE_DS41_PSK_L20): %s
A|lever PSK-CR (fn4 ds41_psk_cr, now %d; RADIANCE_DS41_PSK_CR): %s
A|lever E1 declared (RADIANCE_DS41_ENGDD, fn4 ds41_engdd, now
K|gather_rows_w16_kernel
A|lever BI declared (RADIANCE_DS41_BI, fn4 ds41_bi, now
A|DSpark lever SD: the sampled full-vocabulary walk declared
B|RADIANCE_WEIGHT_SIDECAR
B|lever Z0-A set up"
# int9b's own (VIRHE when missing; SKIP only under DRYRUN=1 + XSPR_DRY_ALLOW_OLD=1, a dry run against an image without them)
IMGLITS9B="B|stage_full_pick, mhc_fuse, ix_lds, ix_ni, ds41_arcoded, sched_row_cap, epoch_bump)
B|mhc_fuse
B|ix_lds
B|ix_ni
B|lds_launches
B|ni_cand_launches
B|dead_bytes
A|RADIANCE_DS41_MHC_FUSE
A|lever M1 (mhc_fuse) declared
A|mhc_post_collapse_rmsnorm_mix
K|mhc_post_collapse_rmsnorm_mix
K|ix_lds
K|ix_ni
R|mhc_pcn_mix
R|ix_lds
R|ix_ni
P|int9b-PROVENANCE.txt"
# int10's own (VIRHE when missing; SKIP only under DRYRUN=1 + XSPR_DRY_ALLOW_OLD=1): lever ARC + draft_depth0 3
IMGLITS10="B|ds41_arcoded
B|coded_all_ranks
B|sched_row_cap
B|RADIANCE_DECODE_ROW_CAP
B|steps_gt48
B|3 (occupancy only)
B|draft depth 0 by OCCUPANCY ONLY
K|ds41_arcoded
K|RADIANCE_DS41_ARCODED
K|rad_arc_stats
K|ar_twoshot_4rank_exact (coded, ds41_arcoded)
R|ar_arcoded
R|rad_arc_stats
P|int10-PROVENANCE.txt"
docker image inspect $IMAGE > /dev/null 2>&1 || { xlog "VIRHE image $IMAGE not on this host"; exit 1; }
IMGMISS=$( { echo "$IMGLITS"; echo "$IMGLITS9B" | sed 's/^/9/'; echo "$IMGLITS10" | sed 's/^/9/'; } | timeout 120 docker run --rm -i --network none --entrypoint sh $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels/libr4d.so; R=/opt/radiance/bin/r4d_selftest; n=0
while IFS= read -r l; do [ -n "$l" ] || continue; n=$((n + 1)); g=""; case $l in 9*) g=9; l=${l#9} ;; esac; w=${l%%|*}; s=${l#*|}
  if [ $w = P ]; then test -s /opt/radiance/share/radiance/$s || echo "MISS$g $w|$s"; continue; fi
  case $w in B) f=$B ;; A) f=$A ;; K) f=$K ;; R) f=$R ;; *) f="$B $A $K" ;; esac
  o=0; for x in $f; do grep -qzF -- "$s" $x && { o=1; break; }; done
  [ $o = 1 ] || echo "MISS$g $w|$s"; done
test -x /opt/radiance/bin/rad-sidecar || echo "MISS bin/rad-sidecar"; for p in int7 int8 int9; do test -s /opt/radiance/share/radiance/$p-PROVENANCE.txt || echo "MISS $p-PROVENANCE.txt"; done; echo "CHECKED $n"' 2>&1)
case "$IMGMISS" in *"CHECKED "*) ;; *) xlog "VIRHE the image literal check did not run on $IMAGE: $(echo "$IMGMISS" | tail -3 | tr '\n' ';' | cut -c1-500)"; exit 1 ;; esac
MISS8=$(echo "$IMGMISS" | grep '^MISS ' | tr '\n' ';'); MISS9=$(echo "$IMGMISS" | grep '^MISS9 ' | sed 's/^MISS9 //' | tr '\n' ';')
[ -z "$MISS8" ] || { xlog "VIRHE $IMAGE is not the int9 image or later -- missing: $(echo "$MISS8" | cut -c1-900)"; exit 1; }
if [ -n "$MISS9" ]; then
  if [ "$DRY" = 1 ] && [ "$ALLOWOLD" = 1 ]; then
    echo "+ image check: SKIP (image lacks int9b / int10 literals; XSPR_DRY_ALLOW_OLD=1, dry run only): missing $(echo "$MISS9" | tr ';' '\n' | grep -c .): $(echo "$MISS9" | cut -c1-700)"
  else
    xlog "VIRHE $IMAGE is not the int10 image -- missing: $(echo "$MISS9" | cut -c1-900)"; exit 1
  fi
fi
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)): all $(echo "$IMGLITS" | grep -c .) int9 / KVR / profiler literals present$([ -z "$MISS9" ] && echo " + all $(echo "$IMGLITS9B" | grep -c .) int9b literals (the registry tail now ending in ds41_arcoded, sched_row_cap) + all $(echo "$IMGLITS10" | grep -c .) int10 literals (ds41_arcoded, GET arc counters, draft_depth0 3 and its start line, libr4d's ARC env / stats / coded launch name, r4d_selftest's ar_arcoded, int10-PROVENANCE)") + bin/rad-sidecar + int7 / int8 / int9 PROVENANCE -- OK"
HMISS=""
for f in "$LIB29" "$LIBPR" $Y $CTR $NCMB $MBP $TESTS $D/fastcheck.py $DAH $DAM $CORPUS; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $CTR $NCMB $MBP $TESTS $D/fastcheck.py $DAH $DAM 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
CTT_=$(timeout 60 python3 -I -B $CTR selftest 2>&1 | tail -1); case "$CTT_" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$CTR(${CTT_:0:300})" ;; esac
DMT=$(timeout 60 python3 -I -B $DAM selftest --daheld $DAH 2>&1 | tail -1); case "$DMT" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$DAM(${DMT:0:300})" ;; esac
NDOC=$(python3 -I -c 'import json,sys; print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l).get("ids")))' $CORPUS 2>/dev/null || echo 0)
[ "${NDOC:-0}" -gt 0 ] || HMISS="$HMISS corpus:$CORPUS(no docs with ids)"
NMT=$(timeout 60 python3 -I -B $NCMB --mb $MBP --selftest 2>&1 | tail -1); case "$NMT" in *OK*) ;; *) HMISS="$HMISS selftest:$NCMB(${NMT:0:300})" ;; esac
if [ -s $NCMB2 ] && [ -s $MBT2 ]; then   # the harness (present: it runs the c8 windows; absent at run time: ncmb.py, logged)
  M2T=$(timeout 60 python3 -I -B $MBT2 --mb $MBP --selftest 2>&1 | tail -1); case "$M2T" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$MBT2(${M2T:0:300})" ;; esac
  N2T=$(timeout 60 python3 -I -B $NCMB2 --mb $MBP --selftest 2>&1 | tail -1); case "$N2T" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$NCMB2(${N2T:0:300})" ;; esac
  CMT=$(timeout 60 python3 -I -B $C8MIX selftest 2>&1 | tail -1); case "$CMT" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$C8MIX(${CMT:0:300})" ;; esac
else M2T="absent"; N2T="absent"; CMT="absent"; fi
FNS="log run mkd restart check down stopfile best_vars desc serve_args xlog elapsed armj NAMEOF pinj stackenv keeplog capture thermline dfree diskok cmpfile idrec identw modes vramstart fn4get fn4set nudge fn4code pcreset arm msnap mdelta pskstat ssnap sdelta gate1 budok ctrinfo arcnow capd drafted readback pinset blk blocks dalive summary dstart"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee sha256sum gzip zcat; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
for a in B; do python3 -c 'import json,sys; d=json.loads(sys.argv[1]); sw = sys.argv[2] == "1"; a = sys.argv[3]
assert d["ds41_psk"] == 4 and d["ds41_psk_cr"] == 1 and d["draft_sample"] == 0 and d["draft_lend"] == 1
if sw: assert d["mhc_fuse"] == int(sys.argv[4]) and d["ix_lds"] == 0 and d["ix_ni"] == 0 and d["draft_depth0"] == int(sys.argv[5]) and d["ds41_arcoded"] == (0 if a == "B" else int(sys.argv[6])) and d["stage_full_min"] == int(sys.argv[7]) and d["sched_row_cap"] == int(sys.argv[8])
else: assert d["draft_depth0"] == 1 and not any(k in d for k in ("mhc_fuse", "ix_lds", "ix_ni", "ds41_arcoded", "stage_full_min", "stage_full_pick", "sched_row_cap"))' "$(pinj $a)" $SW $a $MHC $D0 $ARCN $SPN $CAP 2>/dev/null || HMISS="$HMISS json:$a"; done
[ "$(sha256sum $MODELDIR/$SIDECAR 2>/dev/null | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256 2>/dev/null)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR"
[ "$(sha256sum $MODELDIR/$SIDE 2>/dev/null | cut -c1-64)" = "$SIDESHA" ] || HMISS="$HMISS sha256:$MODELDIR/$SIDE(want ${SIDESHA:0:16})"
grep -qF XS2_ENVFILE $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no XS2_ENVFILE)"
grep -qE 'RADIANCE_DS41_KV_REPLICATED|RADIANCE_DS41_MHC_FUSE|RADIANCE_DS41_IX_|RADIANCE_DS41_ARCODED|RADIANCE_DECODE_ROW_CAP' $Y 2>/dev/null && HMISS="$HMISS yaml:$Y(names a lever word: they come from the env file only)"
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config refuses it)"
case "$LEVENV" in *RADIANCE_DS41_MHC_FUSE=0*) ;; *) HMISS="$HMISS levenv:no-M1-declare" ;; esac
case "$LEVENV" in *RADIANCE_DS41_KV_REPLICATED*|*RADIANCE_DS41_ARCODED*|*RADIANCE_DECODE_ROW_CAP*) HMISS="$HMISS levenv:KVR-ARC-or-CAP-at-start" ;; esac
HMISS=$(echo $HMISS)
[ -z "$HMISS" ] || { xlog "VIRHE helpers / inputs do not resolve: $HMISS"; exit 1; }
HMD5="i9b/i9bctr.py $(md5sum < $CTR | cut -c1-8) (INFO only), nc/ncmb.py $(md5sum < $NCMB | cut -c1-8), harness mbt2.py $(md5sum < $MBT2 2>/dev/null | cut -c1-8) / ncmb2.py $(md5sum < $NCMB2 2>/dev/null | cut -c1-8) / c8mix.py $(md5sum < $C8MIX 2>/dev/null | cut -c1-8), da/daheld.py $(md5sum < $DAH | cut -c1-8) (imported), spread/damark.py $(md5sum < $DAM | cut -c1-8), mb.py $(md5sum < $MBP | cut -c1-8), ds41_tests.py $(md5sum < $TESTS | cut -c1-8), corpus $(basename $CORPUS) $(md5sum < $CORPUS | cut -c1-8) ($NDOC docs), dsm-spread.sh $(md5sum < $0 | cut -c1-8) (from arc/dsm-arc.sh $(md5sum < $HD/../arc/dsm-arc.sh | cut -c1-8)), lib-29.sh $(md5sum < $LIB29 | cut -c1-8), yaml $(md5sum < $Y | cut -c1-8)"
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing; i9bctr.py $CTT_ | ncmb.py $NMT | damark.py $DMT | corpus $NDOC docs | harness: mbt2.py $M2T / ncmb2.py $N2T / c8mix.py $CMT | all $(echo $FNS | wc -w) functions resolve, tools on PATH, the pin (S5T, draft_sample 0, draft_lend 1$([ $SW = 1 ] && echo ", SP 0, M1 on, IX off, draft_depth0 $D0, sched_row_cap $CAP, ds41_arcoded 0" || echo "; XSPR_SW=0: no int9 / int9b / int10 switch")), sidecars $SIDECAR + $SIDE sha256 OK, yaml accepted, LEVENV with MHC_FUSE=0, without KVR / ARC / CAP -- OK | blocks: $([ -s $NCMB2 ] && [ -s $MBT2 ] && echo "harness/ncmb2.py + mbt2.py (fixed schedule)" || echo "nc/ncmb.py (harness absent)") | md5: $HMD5"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { xlog "VIRHE $M225 missing or not verified"; exit 1; }
  [ -s $WARM_REF_225 ] || xlog "note -- reference $WARM_REF_225 missing (not compared)"
  xlog "$ITEM: helpers md5: $HMD5"
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDE; do [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  for f in $WARM_REF_225 $RS5/s5-n5-stats0.json $RS5/guard-s5-n5.txt; do [ -e $f ] && echo "+ present (reference): $f" || echo "+ absent (reference, information only): $f"; done
  echo "+ the pin (base + fixed, held; no arms): $(pinj B)"
  echo "+ blocks [$ORDER] (c1 gens $GENS1, c4 gens $GENS4, c8 gens $GENS8; dur $DUR warm $WARM; T 0) | DA live check $DODA ($(basename $CORPUS), $NDOC docs)"
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(information here; the real run refuses to start beside one)"
  echo "+ disk now: $(dfree) MB free on / (guard $DISKMIN MB)"
fi
mkd $O $O/guard $O/blk $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt
stopfile || dstart
summary
unset $SWITCHES
down
xlog "$ITEM: done ($(elapsed) min) -- $O | $([ $NV = 0 ] && echo "no VIRHE line" || echo "$NV VIRHE line(s): see above")"

