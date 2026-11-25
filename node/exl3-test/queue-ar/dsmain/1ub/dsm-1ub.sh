#!/bin/bash
# dsm-1ub.sh (12.11.2026, queue-ar/dsmain/1ub, the int11 builder's XP fork) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzzzzz-1ub.sh. Log tag "RESULT x1ub ", out runs/xdsm/1ub (X1UB_OUT). Made from a COPY of
# arc/dsm-arc.sh (the start, LEVENV / LITS, the fixed pin and the S5T base, stackenv .. gate1, alone, cmpx, budok, the image
# literal check; tag xarc -> x1ub); the original is not changed. Its cold / c8 / P0 sections are not run here. Helpers read only:
# harness/ncmb2.py + mbt2.py + c8mix.py (c4 on the fixed schedule, the co-batch mix), stack4/loadid.py (greedy-set compare),
# stepprobe.py, mb.py, greedyset.py, ds41_tests.py, i9b/i9bctr.py (INFO); 1ub/rowms.py (this item's: ms a decode step by verify
# rows from the step log).
# LEVER 1U-XP on the ds41-int11b image (ds41-int11 + the core's next-layer expert prefetch wired for DeepSeek-V4.1's EXL3 experts;
# fn4 ds41_xp appended after CBH's sched_cb_hold_k (62; kSwitches 63); GET /admin/fn4 "xp": ranks / copied / routed / planned / dropped / rests):
# at every trunk MoE layer of a pass of <= 16 rows, layer l+1's bf16 router over layer l's router input, ranked by DeepSeek-V4's
# selection key (sqrt(softplus(logit)) + the correction bias), names the likeliest experts; the two best that sit in the pinned
# host pool are copied by DMA into a ring (3 groups x 2 slots of the largest pool unit) behind layer l's routed down GEMM, and
# layer l+1's EXL3 table entries are pointed at a copy (CAS) only once its copy landed, restored after l+1's GEMMs. Placement
# only: a copy holds the pool slot's bytes, the same kernels read the same bytes -- exact by construction.
#   B  ds41_xp 0 (no prediction; the ring allocated, the marks / restores run)
#   C  ds41_xp 1 (the prefetch predicts, copies, redirects)
# THE START = the 1u item's (the ARC screen's stack5 start + RADIANCE_DS41_MHC_FUSE=0 + the int11 levers' declares
#   RADIANCE_DS41_GQ=3 / RADIANCE_DS41_E2=1) with RADIANCE_XPREFETCH=1 (the core's ring and DMA queue: in BOTH arms) and
#   RADIANCE_DS41_XP=1 (the layers linked). BASE PIN = the ARC screen's (S5T, draft_sample 0, SP 0 / 0, M1 on, IX off,
#   draft_depth0 3, sched_row_cap 48, the fixed pin) + ds41_arcoded 0, ds41_mix_wv 0, ds41_gq 0, ds41_e2 0, ds41_shexp_lane 0 (the
#   int10 forms: only ds41_xp moves). NOTE: RADIANCE_XPREFETCH=1 costs the slab its ring (VA "xpoff": +11 / +16 units at 0), in
#   both arms: B here is NOT the 1u item's base; the lever's reading is C vs B in process.
# READINGS (in process, every arm read back after each POST; TESTING.md 12.11. block rules for DS screens):
#   (a) 1u READS: stepprobe X1UB_EPROMPTS (essay,code: base1113's fixed prompts) X1UB_EGEN (512) at T 0, X1UB_EORDER B C B C (two
#       short reads a arm, the reference arm first): per prompt decode tok/s, ms a step, tokens a step;
#       the "xp" deltas (copied / routed = the copies a step used / planned / dropped / rests), the streamed host MB
#       (/stats link.stream_bytes: the heat engine's miss estimate -- it does not see the ring) and the routed / resident expert
#       counts; then ms a decode step by verify-row count a window and a arm from the step log (1ub/rowms.py).
#   (b) c4: mb c4 at T 0 on the harness's fixed, seeded schedule (ncmb2.py --sched fixed --pattern 1x4 --seed 1 --window gens
#       --gens X1UB_C4GENS), X1UB_C4ORDER B C B C (two a arm), c8mix.py a block + the B-vs-B mix check; the xp deltas a block.
#       (A pass of more than 16 rows predicts nothing: at c4 the prefetch acts on the steps of <= 16 rows only.)
#   (c) IDENTITY: greedyset.py's 20 prompts at c1 alone under B then C, text + token ids strict (loadid.py cmp); a difference = VIRHE.
#   GUARDS: switch read back + 0 request errors; the XP COUNTERS a reading: a C reading plans copies (or rests: the prefetch's own
#     payoff bar paused it), else "path did not run" VIRHE; a B reading plans none, else VIRHE.
# ~22 min estimate (start + checks ~5.5, switch note ~0.5, (a) 4 x ~20 s + rows ~1, (b) 4 x ~60 s + mixes ~5, (c) 2 x ~65 s +
#   ~20 s = ~2.6, end ~0.5). BUDGET X1UB_BUDGET_MIN 24 (a section that would end past it is skipped, logged). NO evict, no
#   cc_sim, no power / host / fan settings; production compose files read only.
# MOCK: X1UB_SW=0 = a server without the int9 .. int11b switches (queue-ar/tools/mockrad): no such switch posted, the XP counters
#   INFO; the start line carries a "VIRHE X1UB_SW=0" marker (a mock run only).
# Knobs: X1UB_IMAGE, X1UB_ITEM, X1UB_OUT, X1UB_SW (1), X1UB_EORDER (BCBC), X1UB_EPROMPTS (essay,code), X1UB_EGEN (512), X1UB_C4ORDER (BCBC), X1UB_C4GENS (2),
#   X1UB_DDUR (30) / X1UB_DWARM (10) (the c4 window), X1UB_D (1: the c4 blocks), X1UB_IDN (1), X1UB_MHC (1), X1UB_D0 (3), X1UB_CAP (48),
#   X1UB_BASEX ('' = none; JSON pins merged into the base, never ds41_xp), X1UB_SPN (0) / X1UB_SPK (0), X1UB_DISKMIN_MB (11000),
#   X1UB_BUDGET_MIN (24), X1UB_SIDECAR, X1UB_SIDESHA; X1UB_DRY_ALLOW_OLD=1 (DRYRUN only).
# DRYRUN=1 bash dsm-1ub.sh : prints every command; checks the image literals (int9 .. int11b), every helper (files, parse,
#   selftests), the sidecars, the yaml, the disk; no server, no GPU, writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/../ds41-stack2-tp4.yaml                   # stack2's yaml, read only (env_file ${XS2_ENVFILE})
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh
O=${X1UB_OUT:-$R/xdsm/1ub}; export STATSDIR=$O/stats
RD=$R/xdsm/draft; RM=$R/xdsm; RS5=$R/xdsm/stack5
export IMAGE=${X1UB_IMAGE:?set X1UB_IMAGE}
ITEM=${X1UB_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzzzzz-1ub}
PSKK=8; BIC=3; NOA=2; OCC=6; OCCK=32; DSA=0; HBR=0; HBRANKS=1,2      # stack5's S5T values, draft_sample 0 (i9b's base)
SW=${X1UB_SW:-1}; ARCN=${X1UB_N:-1024}; COLD=${X1UB_COLD:-135000B 135000C 64000C 64000B}; NTOK=${X1UB_NTOK:-512}
C8ORDER=${X1UB_C8ORDER:-BCCB}; C8DUR=${X1UB_C8DUR:-30}; C8WARM=${X1UB_C8WARM:-10}; C8GENS=${X1UB_C8GENS:-2}
EORDER=${X1UB_EORDER-BCBC}; EPROMPTS=${X1UB_EPROMPTS:-essay,code}; C4ORDER=${X1UB_C4ORDER-BCBC}; C4GENS=${X1UB_C4GENS:-2}; ROWMS=$HD/rowms.py; EGEN=${X1UB_EGEN:-512}; DOID=${X1UB_IDN:-1}
MHC=${X1UB_MHC:-1}; D0=${X1UB_D0:-3}; CAP=${X1UB_CAP:-48}; BASEX=${X1UB_BASEX:-}; SPN=${X1UB_SPN:-0}; SPK=${X1UB_SPK:-0}
DOD=${X1UB_D:-1}; DDUR=${X1UB_DDUR:-30}; DWARM=${X1UB_DWARM:-10}; DOC4=${X1UB_C4ID:-0}
P0=${X1UB_P0-}
DISKMIN=${X1UB_DISKMIN_MB:-11000}; BUDGET=${X1UB_BUDGET_MIN:-24}; ALLOWOLD=${X1UB_DRY_ALLOW_OLD:-0}
SIDE=${X1UB_SIDECAR:-dspark-dlw-q8_0.rad}; SIDESHA=${X1UB_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
MT=192; SEED=1234
case $SW in 0|1) ;; *) echo "X1UB_SW is 1 (the item) or 0 (a server without the int9 / int9b / int10 switches: the mock) (got '$SW')"; exit 1 ;; esac
for v in $ARCN $NTOK $C8DUR $C8WARM $C8GENS $EGEN $DOID $MHC $D0 $CAP $DOD $DDUR $DWARM $DOC4 $SPN $SPK $DISKMIN $BUDGET; do case $v in ''|*[!0-9]*) echo "a numeric knob is not a number ('$v')"; exit 1 ;; esac; done
[ $ARCN -ge 1 ] && [ $ARCN -le 1048576 ] || { echo "X1UB_N 1..1048576 (ds41_arcoded's on values)"; exit 1; }
for c in $COLD; do case $c in [0-9]*[BC]) case ${c%?} in *[!0-9]*) echo "X1UB_COLD takes DEPTH followed by B or C (got '$c')"; exit 1 ;; esac ;; *) echo "X1UB_COLD takes DEPTH followed by B or C (got '$c')"; exit 1 ;; esac; done
case $C8ORDER in *[!BC]*) echo "X1UB_C8ORDER takes B and C (got '$C8ORDER')"; exit 1 ;; esac
case $C8ORDER in B*B|"") ;; *) echo "X1UB_C8ORDER must start and end with B (the B-vs-B mix check; got '$C8ORDER')"; exit 1 ;; esac
case $EORDER in *[!BC]*) echo "X1UB_EORDER takes B and C (got '$EORDER')"; exit 1 ;; esac
case $C4ORDER in *[!BC]*) echo "X1UB_C4ORDER takes B and C (got '$C4ORDER')"; exit 1 ;; esac
[ $C8WARM -lt $C8DUR ] || { echo "the c8 warm-up is not shorter than the window"; exit 1; }
[ $MHC -le 1048576 ] || { echo "X1UB_MHC 0..1048576 (mhc_fuse's values)"; exit 1; }
[ $SPN -le 1048576 ] && [ $SPK -le 64 ] || { echo "X1UB_SPN 0..1048576, X1UB_SPK 0..64"; exit 1; }
[ $EGEN -ge 64 ] || { echo "X1UB_EGEN >= 64"; exit 1; }
[ $D0 -le 3 ] || { echo "X1UB_D0 0..3 (draft_depth0's values; 3 = occupancy only, int10)"; exit 1; }
[ $CAP -le 4096 ] || { echo "X1UB_CAP 0..4096 (sched_row_cap's values; 48 = CAP48)"; exit 1; }
[ $DWARM -lt $DDUR ] || { echo "the D block's warm-up is not shorter than its phase"; exit 1; }
for c in $P0; do case $c in *:*[!0-9:]*|*::*|*:*:*:*) echo "X1UB_P0 takes case:min_ok:seconds (got '$c')"; exit 1 ;; esac; done
if [ -n "$BASEX" ]; then python3 -c 'import json,sys; d=json.loads(sys.argv[1]); assert isinstance(d, dict) and "ds41_xp" not in d' "$BASEX" 2>/dev/null \
  || { echo "X1UB_BASEX must be a JSON object of extra base pins without ds41_xp (got '$BASEX')"; exit 1; }; fi
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
S4=$HD/../stack4
LD=$S4/loadid.py                                                                                    # stack4's, read only
CTR=$HD/../i9b/i9bctr.py                                                                            # i9b's (INFO only), read only
NCMB=$HD/../nc/ncmb.py                                                                              # nc's, read only (the fallback only)
MBT2=$HD/../harness/mbt2.py; NCMB2=$HD/../harness/ncmb2.py; C8MIX=$HD/../harness/c8mix.py          # the harness (read only): c8 on mbt2's fixed schedule
DQ=$QA/deepq.py; MBP=$D/mb.py; TESTS=$D/ds41_tests.py; SPB=$D/stepprobe.py
MBT=$S4/mbt.py; EP=$S4/editprobe.py; PRB=$S4/sdprobe.py                                            # stack4's, read only (the D block's d0blocks.py args)
D0B=$HD/../d0ab/d0blocks.py                                                                         # d0ab's (+ its s5ctr.py beside it), read only
C4REF=${X1UB_C4REF:-$R/xdsm/d0ab/s5-n5-c4-A.jsonl}                                                 # d0ab's greedy set at conc 4 under A (int9, 12.11. 12:23)
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
T00=$(date +%s); NV=0
xlog() { case "$*" in *VIRHE*) NV=$((NV + 1)) ;; esac; log "RESULT x1ub $*"; }
elapsed() { echo $(( ($(date +%s) - T00) / 60 )); }
# ---------------------------------------------------------------- the start (= stack5's + RADIANCE_DS41_MHC_FUSE=0; KVR off)
GPTAB=q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422
SIDECAR=dspark-conf.f32
LEVENV="RADIANCE_DS41_GG_PF=1 RADIANCE_DS41_GG_PF_TABLE=$GPTAB RADIANCE_DS41_F32_PF=25 RADIANCE_DS41_ARENA_DECODE_ROWS=8,48
  RADIANCE_DS41_ROUTER_BF16=1 RADIANCE_DS41_EMB_RAM=1 RADIANCE_PLACE_SHADOW_SLOTS=16 RADIANCE_XPREFETCH=1 RADIANCE_DS41_DSPARK_WALK=1
  RADIANCE_DS41_ARGATHER_MAX=48 RADIANCE_DS41_QNFOLD=1 RADIANCE_DS41_SWFOLD=1 RADIANCE_DRAFT_CTXSKIP=0 RADIANCE_DS41_DSPARK_CONF=/models/$SIDECAR
  RADIANCE_DRAFT_LEND=0 RADIANCE_DRAFT_PROBE_BACKOFF=0 RADIANCE_DRAFT_LEND_BORROW=0 RADIANCE_DRAFT_SAMPLE=0 RADIANCE_WEIGHT_SIDECAR=/models/$SIDE
  RADIANCE_DS41_BI=0 RADIANCE_DRAFT_CTXSKIP_SLOTS=129 RADIANCE_DRAFT_LEND_WEIGHTS=0 RADIANCE_DS41_PSK=0
  RADIANCE_LOG_STEPS=1 RADIANCE_DS41_ENGDD=0 RADIANCE_PLACE_HB_RANKS=$HBRANKS
  RADIANCE_DS41_MHC_FUSE=0 RADIANCE_DS41_GQ=3 RADIANCE_DS41_E2=1 RADIANCE_DS41_XP=1"
LEVENV=$(echo $LEVENV)
LITS="gemm_nt_ggml's,prefill,rows,take,lever,GPF's,forms,(RADIANCE_DS41_GG_PF=1|gemm_nt_f32's,prefill,rows,take,lever,GPF-b's,tiled,form,(RADIANCE_DS41_F32_PF=25"
LITS="$LITS|48,tokens,use|a,rank),ram|expert,prefetch,on:,2,copies,a,routed,layer,by,DMA"
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
# + lever 1U-XP's link line (int11b) and the int11 levers' declare lines (the 1u item's start)
# 12.11. fix (ds-main-port): LITS uses , for a space, so a literal cannot hold the log line's own commas -- comma-free pieces
LITS="$LITS|lever,1U-XP,(ds41_xp|now,1),linked:|lever,1U-GQ,(ds41_gq|lever,1U-E2,(ds41_e2"
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": 8, "drafter_mx": 1, "gg_pf": 1, "f32_pf": 25, "dspark_walk": 1, "ds41_argather": 48, "idx_split": 1, "ds41_qnfold": 1, "ds41_swfold": 1, "draft_ctxskip": 1, "dspark_conf": 1, "dspark_conf_rows": 1, "draft_lend": 1, "draft_probe_backoff": 3, "draft_hybrid": 1}'
FIX=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update({"draft_lend_hyst": 0, "arena_d0": 0, "draft_lend_w": 0, "draft_occ_k": int(sys.argv[2]), "ds41_psk_k": int(sys.argv[3])}); print(json.dumps(d))' "$FIX5" "$OCCK" "$PSKK") || { echo "pin JSON"; exit 1; }
S5J="\"ds41_bi\": 2, \"draft_sample\": $DSA, \"draft_occ\": $OCC, \"place_lf\": 0, \"sched_bi_chunk\": $BIC, \"sched_noa\": $NOA, \"ds41_psk_l20\": 1, \"dspark_pfctx\": 1, \"ds41_engdd\": 1"
# ---------------------------------------------------------------- the base pin (S5T, draft_sample 0, SP pinned, M1 on, IX off) and the arms
SPJ=", \"stage_full_min\": $SPN, \"stage_full_pick\": $SPK"; [ $SW = 1 ] || SPJ=""        # X1UB_SW=0: the mock has no int9 switch
TJ="{$S5J, \"ds41_psk\": 4, \"ds41_psk_cr\": 1, \"stage_short\": 0, \"stage_pick\": 0, \"place_hb\": 0$SPJ}"   # = stack5's armj T, draft_sample 0, + SP
MJ="{\"mhc_fuse\": $MHC, \"ix_lds\": 0, \"ix_ni\": 0, \"draft_depth0\": $D0, \"sched_row_cap\": $CAP, \"ds41_arcoded\": 0, \"ds41_mix_wv\": 0, \"ds41_gq\": 0, \"ds41_e2\": 0, \"ds41_shexp_lane\": 0}"; [ $SW = 1 ] || MJ='{}'   # i9b's "M" arm (M1 on, IX-LDSR2 / IX-NI off) + draft_depth0 3 + CAP48 (int10)
armj() {   # armj B|C: the arm's ds41_xp (JSON) on top of the base; {} under X1UB_SW=0 (the mock)
  [ $SW = 1 ] || { echo '{}'; return 0; }
  case $1 in
    B) echo '{"ds41_xp": 0}' ;;
    C) echo '{"ds41_xp": 1}' ;;
  esac
}
NAMEOF() { case $1 in B) echo "B (no prediction, the ring allocated)" ;; C) echo "C (next-layer expert prefetch)" ;; esac; }
pinj() {   # pinj ARM [EXTRAJSON]: the fixed pin + the base (S5T, draft_sample 0, SP, M1 on, IX off, X1UB_BASEX) + the arm + extra as one POST body
  local x=${2:-} bx=$BASEX; [ -n "$x" ] || x='{}'; [ -n "$bx" ] || bx='{}'
  python3 -c 'import json,sys
d = {}
for a in sys.argv[1:]: d.update(json.loads(a))
print(json.dumps(d))' "$FIX" "$TJ" "$MJ" "$bx" "$(armj $1)" "$x"
}
DESC="NST 5 stack5 start (stack4's + RADIANCE_LOG_STEPS=1, E1 declared off, HB ranks $HBRANKS) + RADIANCE_DS41_MHC_FUSE=0 (M1 declared) + the 1u item's RADIANCE_DS41_GQ=3 / RADIANCE_DS41_E2=1 (declared, pinned 0) + RADIANCE_XPREFETCH=1 (the ring) + RADIANCE_DS41_XP=1 (lever 1U-XP linked) | KVR OFF (RADIANCE_DS41_KV_REPLICATED unset), SP off | base pin = stack5's S5T with draft_sample 0 + SP pinned (stage_full_min $SPN, stage_full_pick $SPK) + M1 on / IX off / draft_depth0 $D0 (int10: depth 0 by occupancy only) / sched_row_cap $CAP (CAP48) $MJ + the fixed pin (ds41_psk_k $PSKK; draft_lend 1)$([ -n "$BASEX" ] && echo " + X1UB_BASEX $BASEX") | arms B $(armj B), C $(armj C)"
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
  RADIANCE_DS41_MHC_FUSE RADIANCE_DS41_IX_LDS RADIANCE_DS41_IX_NI RADIANCE_DS41_ARCODED RADIANCE_DECODE_ROW_CAP
  RADIANCE_DS41_MIX_WV RADIANCE_DS41_GQ RADIANCE_DS41_E2 RADIANCE_DS41_XP"
# ---------------------------------------------------------------- copied from stack5/dsm-stack5.sh (63649414; tag xs5 -> xarc): stackenv .. gate1 (modes: + nc/dsm-ncprof-p.sh's rule-w check under --profile-ops)
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
  [ "$DRY" = 1 ] && { echo "+ RESULT x1ub $1 warm-up identity vs $2"; return 0; }
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
  [ "$DRY" = 1 ] && { echo "+ GET /stats > $O/$1-stats0.json; RESULT x1ub $1 VRAM at start (experts per tier; vs stack4 s4-n5)"; return 0; }
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
  [ "$DRY" = 1 ] && { echo "+ RESULT x1ub $tag $3: greedy set c1 alone under $(NAMEOF $2) $(armj $2): 20 of 20 answered (else a request-error line); counters INFO"; return 0; }
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
# ---------------------------------------------------------------- this item's own functions
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
arcnow() {   # arcnow: GET /admin/fn4's ds41_xp + the base switches and the "xp" counters, one line
  [ "$DRY" = 1 ] && { echo "(dry)"; return 0; }
  fn4get | python3 -c 'import json,sys
d = json.load(sys.stdin); w = d.get("switches") or {}
print("ds41_xp %s ds41_gq %s ds41_e2 %s ds41_mix_wv %s ds41_arcoded %s sched_row_cap %s mhc_fuse %s draft_depth0 %s | xp %s | cap %s" % (w.get("ds41_xp"), w.get("ds41_gq"), w.get("ds41_e2"), w.get("ds41_mix_wv"), w.get("ds41_arcoded"), w.get("sched_row_cap"), w.get("mhc_fuse"), w.get("draft_depth0"), json.dumps(d.get("xp")), json.dumps(d.get("cap"))))' 2>/dev/null | cut -c1-700
}
arcd() {   # arcd A B -> JSON: the "xp" counter deltas (GET /admin/fn4 snapshots) + the streamed host bytes and the routed /
          # resident expert counts (/stats experts.ranks, link.stream_bytes) between snapshots A and B
  [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }
  python3 -c 'import json,sys
def rd(p):
    try: return json.load(open(p))
    except Exception: return {}
fa, fb, sa, sb = rd(sys.argv[1]), rd(sys.argv[2]), rd(sys.argv[3]), rd(sys.argv[4])
a, b = fa.get("xp") or {}, fb.get("xp")
o = {"present": b is not None, "ranks": (b or {}).get("ranks")}
for k in ("copied", "routed", "planned", "dropped", "rests"):
    x, y = a.get(k, 0), (b or {}).get(k, 0)
    o[k] = (y - x) if isinstance(x, (int, float)) and isinstance(y, (int, float)) else None
def g(d, *ks):
    for k in ks: d = d.get(k) if isinstance(d, dict) else None
    return d if isinstance(d, (int, float)) else 0
o["stream_mb"] = round((g(sb, "link", "stream_bytes") - g(sa, "link", "stream_bytes")) / 1e6, 1)
def rk(d, k): return sum((r.get(k) or 0) for r in ((d.get("experts") or {}).get("ranks") or []) if isinstance(r, dict))
o["exp_routed"] = rk(sb, "routed") - rk(sa, "routed"); o["exp_routed_res"] = rk(sb, "routed_res") - rk(sa, "routed_res")
print(json.dumps(o))' $O/fn-$1.json $O/fn-$2.json $O/st-$1.json $O/st-$2.json 2>/dev/null || echo '{}'
}
arcg() {   # arcg ARM DELTAJSON WHAT: the XP counter guard of one reading -> "" (ok) or a VIRHE text; X1UB_SW=0: INFO only
  [ "$DRY" = 1 ] && return 0
  python3 -c 'import json,sys
d = json.loads(sys.argv[2] or "{}"); a = sys.argv[1]; sw = sys.argv[4] == "1"
pl, rs = d.get("planned"), d.get("rests")
if not sw: print("INFO (X1UB_SW=0) xp %s" % json.dumps(d)); sys.exit(0)
if not d.get("present"): print("VIRHE GET /admin/fn4 has no \"xp\" counters (not the int11b image?)"); sys.exit(0)
if a == "C" and not ((pl or 0) > 0 or (rs or 0) > 0): print("VIRHE path did not run: no copy planned (and no rest) in a C reading (%s)" % sys.argv[3]); sys.exit(0)
if a == "B" and (pl or 0) != 0: print("VIRHE %s copies planned in a B (OFF) reading" % pl); sys.exit(0)' "$1" "$2" "$3" $SW 2>&1 | head -1
}
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ulimit memlock=-1 --ipc=host -e HIP_VISIBLE_DEVICES=0,1,2,3"
gpurun() {   # gpurun NAME SECONDS docker-run-args...: bounded, named, removed even if its kernel hangs (e1/dsm-e1.sh's, copied)
  local name=$1 t=$2; shift 2
  timeout -k 15 $t docker run --rm --init --stop-timeout 10 --name $name "$@"; local rc=$?
  docker rm -f $name > /dev/null 2>&1 || true
  { [ $rc = 124 ] || [ $rc = 137 ]; } && echo "gpurun: $name stopped after $t s (rc $rc)" >&2
  return $rc
}
p0sel() {   # p0sel: ALL FOUR cards, no server -- the image's r4d_selftest --case ar_arcoded / ar_twoshot_4rank_exact / ar_twoshot_4rank_exact_lane
  local c cs nmin tmo r rc f bad=0 nok nfail nskip
  for c in $P0; do
    cs=${c%%:*}; nmin=$(echo $c | cut -d: -f2); tmo=$(echo $c | cut -d: -f3); f=$O/p0-$cs.txt
    run "gpurun xarc-p0-$cs-$$ $tmo $DEV --entrypoint /opt/radiance/bin/r4d_selftest $IMAGE /opt/radiance/share/radiance/kernels/libr4d.so --case $cs > $f 2>&1"; rc=$?
    if [ "$DRY" = 1 ]; then echo "+ RESULT x1ub P0 $cs on 4 cards: rc 0, 0 FAIL, 0 skip, >= $nmin ok lines (bounded at $tmo s), else a failure line and the item stops (no server start); info lines logged"; continue; fi
    nok=$(grep -c '^  ok' $f); nfail=$(grep -c '^  FAIL' $f); nskip=$(grep -c '^  skip' $f)
    r="$nok ok (need $nmin), $nfail FAIL, $nskip skip, rc $rc"
    while IFS= read -r l; do xlog "P0 $cs info: ${l:0:600}"; done < <(grep -E '^  info' $f | head -8)
    if [ $rc = 0 ] && [ "$nfail" = 0 ] && [ "$nskip" = 0 ] && [ "$nok" -ge $nmin ]; then
      xlog "P0 $cs on 4 cards (r4d_selftest --case): OK $r | $(grep -E '^  ok' $f | sed 's/^ *ok *//' | cut -c1-120 | tr '\n' ';' | cut -c1-1800)"
    else
      bad=1; xlog "P0 $cs on 4 cards (r4d_selftest --case): VIRHE $r | $(grep -E '^  FAIL|^  skip|FATAL|error|Error|fault|abort|stopped after' $f | head -6 | sed 's/^ *//' | cut -c1-240 | tr '\n' ';' | cut -c1-1400)"
      break   # the next case does not start beside a failure (a hang or a lost card is not stacked)
    fi
  done
  [ $bad = 0 ]
}
drafted() {   # drafted TAG: the DSpark drafter and the adaptive depth are up (stack4's check)
  [ "$DRY" = 1 ] && return 0
  local ok=1
  docker logs $CN 2>&1 | grep -q "DSpark drafter" || { xlog "$1: VIRHE no DSpark drafter line"; ok=0; }
  docker logs $CN 2>&1 | grep -q "draft depth: adaptive" || { xlog "$1: VIRHE no adaptive draft depth line"; ok=0; }
  [ $ok = 1 ] || keeplog $1-nodraft
  [ $ok = 1 ]
}
readback() {   # readback TAG: the start's words read back -- MHC_FUSE=0, GQ=3, E2=1, XP=1, XPREFETCH=1 in the env, M1's declare
            # line and lever 1U-XP's link line; KVR, ARCODED and DECODE_ROW_CAP NOT in the env, no KVR declare line
  [ "$DRY" = 1 ] && { echo "+ readback $1: RADIANCE_DS41_MHC_FUSE=0, RADIANCE_DS41_GQ=3, RADIANCE_DS41_E2=1, RADIANCE_DS41_XP=1, RADIANCE_XPREFETCH=1 in the container env, the 'lever M1 (mhc_fuse) declared' and 'lever 1U-XP ... linked' lines and 'expert prefetch on' on 4 ranks; RADIANCE_DS41_KV_REPLICATED / _ARCODED / RADIANCE_DECODE_ROW_CAP NOT in the env, no 'lever KVR declared' line"; return 0; }
  local ce e kv ac cp d m x xl xo ok=1
  ce=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  e=$(echo "$ce" | grep -cxE 'RADIANCE_DS41_MHC_FUSE=0|RADIANCE_DS41_GQ=3|RADIANCE_DS41_E2=1|RADIANCE_DS41_XP=1|RADIANCE_XPREFETCH=1')
  kv=$(echo "$ce" | grep -c '^RADIANCE_DS41_KV_REPLICATED='); ac=$(echo "$ce" | grep -c '^RADIANCE_DS41_ARCODED='); cp=$(echo "$ce" | grep -c '^RADIANCE_DECODE_ROW_CAP=.')
  d=$(docker logs $CN 2>&1 | grep -m1 'lever KVR declared' | sed 's/^radiance: //' | cut -c1-200)
  m=$(docker logs $CN 2>&1 | grep -m1 'lever M1 (mhc_fuse) declared' | sed 's/^radiance: //; s/^[A-Z]\[[0-9]\] //; s/^[A-Z] *//' | cut -c1-300)
  xl=$(docker logs $CN 2>&1 | grep -m1 'lever 1U-XP (ds41_xp' | sed 's/^radiance: //; s/^[A-Z]\[[0-9]\] //; s/^[A-Z] *//' | cut -c1-400)
  xo=$(docker logs $CN 2>&1 | grep -c 'expert prefetch on:')
  # the XP link line and the four ranks' "expert prefetch on": required on the real image; INFO under X1UB_SW=0 (the mock emulates neither)
  case "$xl" in *" linked: "*) ;; *) [ $SW = 0 ] || ok=0 ;; esac
  [ "$xo" = 4 ] || [ $SW = 0 ] || ok=0
  [ "$e" = 5 ] && [ "$kv" = 0 ] && [ "$ac" = 0 ] && [ "$cp" = 0 ] && [ -z "$d" ] && [ -n "$m" ] || ok=0
  xlog "$1 read back: env MHC_FUSE=0 / GQ=3 / E2=1 / XP=1 / XPREFETCH=1 $e of 5, KV_REPLICATED $kv (want 0), ARCODED $ac (want 0), DECODE_ROW_CAP $cp (want 0: CAP48 comes by fn4) | KVR declare line: ${d:-none (off)} | M1: ${m:-NONE} | XP: ${xl:-NONE} | 'expert prefetch on' on $xo of 4 ranks$([ $ok = 1 ] || echo " -- VIRHE the start's words are not read back")"
  [ $ok = 1 ]
}
swnote() {   # swnote TAG: start values; refusals; every arm read back (base: M1 on, IX off, draft_depth0 3, SP 0)
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 at start (gg_gemv 2, aql_acq 2, K1 three 1, epoch 0, draft_capable 1, draft_sample 0; ds41_psk 0, sched_bi_chunk / sched_noa / ds41_psk_l20 / dspark_pfctx / ds41_engdd / ds41_psk_cr / stage_short / stage_pick / place_hb 0$([ $SW = 1 ] && echo "; stage_full_min / stage_full_pick / mhc_fuse / ix_lds / ix_ni / ds41_arcoded / sched_row_cap 0; ds41_xp 1 (its env; ds41_gq / ds41_e2 logged); \"xp\" present" || echo "; int9 / int9b / int10 NOT CHECKED (X1UB_SW=0)")); refusals stage_short 65537$([ $SW = 1 ] && echo " / ds41_xp 2 / ds41_xp -1 / draft_depth0 4 / sched_row_cap 4097") -> 400; arms B C B each applied and read back ($(pinj B | cut -c1-200)...)"; return 0; }
  local s0 g c1=400 c2=400 c3=400 c5=400 c4 a sets="" bad=""
  s0=$(fn4get)
  c4=$(fn4code '{"stage_short": 65537}')
  [ $SW = 1 ] && { c1=$(fn4code '{"ds41_xp": -1}'); c2=$(fn4code '{"ds41_xp": 2}'); c3=$(fn4code '{"draft_depth0": 4}'); c5=$(fn4code '{"sched_row_cap": 4097}'); }
  for a in B C B; do
    arm $a || bad="$bad $a"; g=$(fn4get)
    sets="$sets $a:$(echo "$g" | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print("xp=%s,cap=%s,mhc=%s,lds=%s,ni=%s,d0=%s,lend=%s,occ=%s,ds=%s,psk=%s/%s,cr=%s,bi=%s,sfm=%s,sfp=%s,ep=%s" % (w.get("ds41_xp"), w.get("sched_row_cap"), w.get("mhc_fuse"), w.get("ix_lds"), w.get("ix_ni"), w.get("draft_depth0"), w.get("draft_lend"), w.get("draft_occ"), w.get("draft_sample"), w.get("ds41_psk"), w.get("ds41_psk_k"), w.get("ds41_psk_cr"), w.get("ds41_bi"), w.get("stage_full_min"), w.get("stage_full_pick"), d.get("epoch")))' 2>/dev/null)"
  done
  xlog "$1 switch note: at start $(echo "$s0" | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print("switches %s | xp %s | epoch %s" % (json.dumps({k: w.get(k) for k in ("ds41_xp", "ds41_gq", "ds41_e2", "ds41_mix_wv", "ds41_arcoded", "sched_row_cap", "mhc_fuse", "ix_lds", "ix_ni", "draft_depth0", "stage_full_min", "stage_full_pick", "draft_sample", "ds41_psk", "ds41_bi", "gg_gemv", "aql_acq")}), json.dumps(d.get("xp")), d.get("epoch")))' 2>/dev/null | cut -c1-1200) | stage_short 65537 -> HTTP $c4$([ $SW = 1 ] && echo ", ds41_xp -1 -> HTTP $c1, ds41_xp 2 -> HTTP $c2, draft_depth0 4 -> HTTP $c3, sched_row_cap 4097 -> HTTP $c5" || echo " | int9 / int9b / int10 NOT CHECKED (X1UB_SW=0)") | arms applied (read back):$sets$([ -n "$bad" ] && echo " -- VIRHE not applied:$bad")"
  python3 -c 'import json,sys
s0 = json.loads(sys.argv[1]); sw = sys.argv[2] == "1"; w = s0["switches"]; ac = s0.get("xp")
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) and s0["epoch"] == 0
ok = ok and w["draft_adaptive"] == 1 and s0["draft_capable"] == 1 and w.get("ds41_psk") == 0 and w.get("draft_sample") == 0
ok = ok and all(w.get(k) == 0 for k in ("sched_bi_chunk", "sched_noa", "ds41_psk_l20", "dspark_pfctx", "ds41_engdd", "ds41_psk_cr", "stage_short", "stage_pick", "place_hb"))
if sw:
    ok = ok and all(w.get(k) == 0 for k in ("stage_full_min", "stage_full_pick", "mhc_fuse", "ix_lds", "ix_ni", "ds41_arcoded", "sched_row_cap"))
    ok = ok and w.get("ds41_xp") == 1 and isinstance(ac, dict)   # ds41_gq / ds41_e2 read 3 / 1 where declared, 0 where not (logged only)
sys.exit(0 if ok else 1)' "$s0" $SW 2>/dev/null && [ "$c1" = 400 ] && [ "$c2" = 400 ] && [ "$c3" = 400 ] && [ "$c4" = 400 ] && [ "$c5" = 400 ] && [ -z "$bad" ]
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
c4x() {   # c4x I ARM: one mb c4 block at T 0 on the harness's fixed, seeded schedule (harness/ncmb2.py --sched fixed --pattern 1x4
          # --seed 1, window gens C4GENS, edge 0.5, ignore_eos: TESTING.md 12.11. block rules for DS screens); 0 request errors; the XP guard
  local i=$1 a=$2 nm=c4-$1$2 t0 base=$O/c8/c4-$1$2 j ad g
  arm $a || return 1
  ssnap $nm-0
  t0=$(date +%s)
  run "(cd $D && timeout $((DDUR + DWARM + 900)) python3 -I -B $NCMB2 --mb $MBP --mode phase --n 4 --temperature 0 --tag x1ub-$nm --dur $DDUR --warm $DWARM --reqs $base-reqs.jsonl --stats-prefix $base-st --samples $base-samp.jsonl --sched fixed --pattern 1x4 --seed 1 --window gens --gens $C4GENS --edge 0.5 --ignore-eos 1 > $base.txt 2>&1)"
  ssnap $nm-1
  [ "$DRY" = 1 ] && { echo "+ RESULT x1ub c4 $nm ($(NAMEOF $a) $(armj $a), mb c4 T 0 via harness/ncmb2.py: fixed schedule 1x4 seed 1, gens $C4GENS, edge 0.5, ignore_eos): MB agg_dec_tps / decode_share / per-stream p50, errors 0; xp counters + stream MB; the XP guard -> $O/c4.jsonl"; return 0; }
  j=$(grep -m1 '^MB ' $base.txt | cut -c4-)
  [ -n "$j" ] || { xlog "c4 $nm: VIRHE request error: no MB line ($(tail -3 $base.txt | tr '\n' ' ' | cut -c1-300))"; return 1; }
  ad=$(arcd $nm-0 $nm-1); g=$(arcg $a "$ad" "c4 block $i")
  python3 -c 'import json,sys
d = json.loads(sys.argv[1])
print(json.dumps(dict(name=sys.argv[2], arm=sys.argv[3], agg_dec_tps=d.get("agg_dec_tps"), decode_share=d.get("decode_share"), per_stream_dec_p50=d.get("per_stream_dec_p50"),
      ttft_p50=d.get("ttft_p50"), requests=d.get("requests"), errors=d.get("errors"), sched=d.get("sched"), window=d.get("window"), xp=json.loads(sys.argv[4] or "{}"), guard=sys.argv[5])))' \
    "$j" "$nm" "$a" "$ad" "$g" >> $O/c4.jsonl
  xlog "c4 $nm ($(NAMEOF $a) $(armj $a), mb c4 T 0 fixed 1x4, $(( $(date +%s) - t0 )) s): $(tail -1 $O/c4.jsonl | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
print("agg %s tok/s, decode_share %s, per-stream p50 %s, TTFT p50 %s, %s requests, %s errors%s" % (d.get("agg_dec_tps"), d.get("decode_share"), d.get("per_stream_dec_p50"), d.get("ttft_p50"),
      d.get("requests"), d.get("errors"), "" if d.get("errors") == 0 and (d.get("requests") or 0) > 0 else " -- VIRHE request errors"))' 2>&1 | cut -c1-500) | xp $ad$([ -n "$g" ] && echo " -- $g") | INFO $(capd $nm-0 $nm-1)"
}
dblock() {   # (b) the c4 blocks in C4ORDER (B C B C: two a arm), then one engine-log capture; per block c8mix.py block (the co-batch mix and
          # the newcomer step walls), c8mix.py compare B1 vs the last B (the B-vs-B mix check) and over every block (INFO)
  local i=0 a n=${#C4ORDER} ins="" b1="" bl="" b
  [ -n "$C4ORDER" ] || return 0
  [ -s $NCMB2 ] && [ -s $MBT2 ] || { xlog "c4: VIRHE harness/ncmb2.py / mbt2.py absent -- the c4 blocks are not run (the block rules ask for the fixed schedule)"; return 0; }
  xlog "c4: harness/mbt2.py through harness/ncmb2.py (TESTING.md 12.11. block rules for DS screens: fixed, seeded schedule, pattern 1x4, seed 1, window gens $C4GENS, edge 0.5, ignore_eos; order $C4ORDER = two blocks a arm; mbt2.py $(md5sum < $MBT2 | cut -c1-8), ncmb2.py $(md5sum < $NCMB2 | cut -c1-8), c8mix.py $(md5sum < $C8MIX 2>/dev/null | cut -c1-8))"
  for a in $(echo $C4ORDER | grep -o .); do i=$((i + 1)); stopfile && return 0; c4x $i $a; done
  capture $O/c4-engine.log.gz
  [ "$DRY" = 1 ] && { echo "+ per block: python3 -I -B $C8MIX block --log $O/c4-engine.log.gz --reqs $O/c8/<b>-reqs.jsonl --tag x1ub-<b> --mbline $O/c8/<b>.txt --label <b> --json $O/c8/<b>-mix.json; then $C8MIX compare (the first vs the last B: MATCH = the C-vs-B delta is read; else INCONCLUSIVE (mix) with the numbers) + over every block (INFO)"; return 0; }
  [ -s $C8MIX ] || { xlog "c4: c8mix.py absent -- no mix reading"; return 0; }
  i=0
  for a in $(echo $C4ORDER | grep -o .); do
    i=$((i + 1)); b=c4-$i$a
    [ -s $O/c8/$b-reqs.jsonl ] || { xlog "c4 $b: no taps (block not run)"; continue; }
    timeout 300 python3 -I -B $C8MIX block --log $O/c4-engine.log.gz --reqs $O/c8/$b-reqs.jsonl --tag x1ub-$b --mbline $O/c8/$b.txt --label $b --json $O/c8/$b-mix.json > $O/c8/$b-mix.out 2>&1
    xlog "c4 mix $b ($(NAMEOF $a)): $(grep -m1 '^C8MIX ' $O/c8/$b-mix.out | cut -c7- | sed 's/ -- VIRHE / -- c8mix guard (INFO): /' | cut -c1-800)"
    [ -s $O/c8/$b-mix.json ] && ins="$ins --in $b=$O/c8/$b-mix.json"
    [ $a = B ] && [ -s $O/c8/$b-mix.json ] && { [ -n "$b1" ] || b1=$b; bl=$b; }
  done
  if [ -n "$b1" ] && [ "$bl" != "$b1" ]; then
    timeout 120 python3 -I -B $C8MIX compare --in $b1=$O/c8/$b1-mix.json --in $bl=$O/c8/$bl-mix.json > $O/c8/c4-bb-compare.out 2>&1
    xlog "c4 B-vs-B mix check ($b1 vs $bl): $(grep "^C8MIXCMP $bl " $O/c8/c4-bb-compare.out | cut -c10- | cut -c1-500)"
  else xlog "c4 B-vs-B mix check: not possible (fewer than two B blocks with a mix)"; fi
  [ -n "$ins" ] && timeout 120 python3 -I -B $C8MIX compare $ins > $O/c8/c4-all-compare.out 2>&1
  while IFS= read -r l; do xlog "c4 mix compare (INFO): ${l:9:600}"; done < <(grep '^C8MIXCMP ' $O/c8/c4-all-compare.out 2>/dev/null)
}
coldx() {   # coldx K DEPTH ARM: deepq's fixed prompt at DEPTH after a reset, NTOK greedy tokens -> TTFT, cached, chunk ms; the ARC guard
  local k=$1 dep=$2 a=$3 nm=cold-$1-$2$3 t0=$(date +%s) j md ad g
  arm $a || return 1
  pcreset $nm
  msnap $nm-0; ssnap $nm-0
  run "(cd $D && timeout 1200 python3 $DQ $O/$nm.txt $dep $NTOK > $O/$nm.out 2>&1)"
  msnap $nm-1; ssnap $nm-1
  [ "$DRY" = 1 ] && { echo "+ RESULT x1ub cold $nm: $(NAMEOF $a) $(armj $a), deepq $dep cold, $NTOK tok: TTFT, cached, chunk ms (status 200 or a request-error line); the ARC guard (C: coded launches > 0 on every rank; B: none) -> $O/cold.jsonl"; return 0; }
  j=$(grep -m1 '^DEEPQ ' $O/$nm.out | cut -c7-)
  [ -n "$j" ] || { xlog "cold $nm: VIRHE request error: no answer ($(tail -2 $O/$nm.out | tr '\n' ' ' | cut -c1-300))"; return 1; }
  md=$(mdelta $nm-0 $nm-1); ad=$(arcd $nm-0 $nm-1); g=$(arcg $a "$ad" "cold $dep")
  python3 -c 'import json,sys
d = json.loads(sys.argv[1]); m = json.loads(sys.argv[2] or "{}")
d.update(nondecode_ms=m.get("nondecode_ms"), nondecode_steps=m.get("nondecode_steps"), depth=int(sys.argv[3]), arm=sys.argv[4], slot=int(sys.argv[5]), name=sys.argv[6],
         text=sys.argv[7], arc=json.loads(sys.argv[8] or "{}"), guard=sys.argv[9])
print(json.dumps(d))' "$j" "$md" "$dep" "$a" "$k" "$nm" "$O/$nm.txt" "$ad" "$g" >> $O/cold.jsonl
  xlog "cold $nm ($(NAMEOF $a) $(armj $a), deepq $dep cold after a reset, $NTOK greedy tokens): $(python3 -c 'import json,sys
d, m = json.loads(sys.argv[1]), json.loads(sys.argv[2] or "{}")
print("status %s, prompt %s tok, cached %s, TTFT %s s (%s prefill tok/s), %s tok | non-decode steps %s, %s ms mean%s%s" % (d.get("status"), d.get("prompt_tokens"), d.get("cached"), d.get("ttft"),
      round((d["prompt_tokens"] - (d.get("cached") or 0)) / d["ttft"], 1) if d.get("ttft") and d.get("prompt_tokens") else "-", d.get("tokens"), m.get("nondecode_steps"), m.get("nondecode_ms"),
      "" if d.get("status") == 200 else " -- VIRHE request error (status %s)" % d.get("status"), "" if not d.get("cached") else " -- NOT COLD (cached %s): left out of the TTFT line" % d.get("cached")))' "$j" "$md" 2>&1 | cut -c1-500) | arc $ad$([ -n "$g" ] && echo " -- $g") | INFO counters: $(ctrinfo $nm-0 $nm-1 | cut -c1-400) | $(thermline $t0 $(date +%s))"
}
colds() {   # (a) the cold runs in X1UB_COLD order; per depth the C answer vs the B answer, byte for byte (a difference = VIRHE)
  local c k=0 dep a fb fc r
  budok "cold prefill (a)" $(( 2 + $(echo $COLD | wc -w) )) || return 0
  for c in $COLD; do k=$((k + 1)); stopfile && return 0; coldx $k ${c%?} ${c#${c%?}}; done
  for dep in $(for c in $COLD; do echo ${c%?}; done | awk '!s[$0]++'); do
    fb=$(ls $O/cold-*-${dep}B.txt 2>/dev/null | head -1); fc=$(ls $O/cold-*-${dep}C.txt 2>/dev/null | head -1)
    [ "$DRY" = 1 ] && { echo "+ identity cold $dep C vs B: cmpfile (byte for byte; a difference = VIRHE)"; continue; }
    if [ -z "$fb" ] || [ -z "$fc" ]; then xlog "identity cold $dep C vs B: VIRHE not compared (missing an answer)"; echo "cold-$dep-C-vs-B not compared" >> $O/identity.txt; continue; fi
    r=$(cmpfile $fb $fc)
    case "$r" in IDENTICAL*) idrec "cold-$dep-C-vs-B" "$r" "$(basename $fc) vs $(basename $fb)" ;;
      *) idrec "cold-$dep-C-vs-B" "VIRHE $r" "$(basename $fc) vs $(basename $fb)" ;; esac
  done
}
c8x() {   # c8x I ARM: one mb c8 block at temperature 0 on the harness's fixed, seeded schedule (harness/ncmb2.py, TESTING.md "12.11.2026
          # c8 block mix": 16 newcomer steps a block, gens 2); 0 request errors; the ARC guard. Without the harness: nc/ncmb.py, logged.
  local i=$1 a=$2 nm=c8-$1$2 t0 base=$O/c8/c8-$1$2 j tool ad g
  arm $a || return 1
  ssnap $nm-0
  t0=$(date +%s)
  if [ -s $NCMB2 ] && [ -s $MBT2 ]; then
    tool=ncmb2
    run "(cd $D && timeout $((C8DUR + C8WARM + 900)) python3 -I -B $NCMB2 --mb $MBP --mode phase --n 8 --temperature 0 --tag xarc-$nm --dur $C8DUR --warm $C8WARM --reqs $base-reqs.jsonl --stats-prefix $base-st --samples $base-samp.jsonl --sched fixed --pattern 1x8 --seed 1 --window gens --gens $C8GENS --edge 0.5 --ignore-eos 1 > $base.txt 2>&1)"
  else
    tool=ncmb
    run "(cd $D && timeout $((C8DUR + C8WARM + 900)) python3 -I -B $NCMB --mb $MBP --mode phase --n 8 --temperature 0 --tag xarc-$nm --dur $C8DUR --warm $C8WARM --reqs $base-reqs.jsonl --stats-prefix $base-st --samples $base-samp.jsonl > $base.txt 2>&1)"
  fi
  ssnap $nm-1
  [ "$DRY" = 1 ] && { echo "+ RESULT x1ub c8 $nm ($(NAMEOF $a) $(armj $a), mb c8 T 0 via $([ -s $NCMB2 ] && echo "harness/ncmb2.py: fixed schedule 1x8 seed 1, gens $C8GENS, edge 0.5, ignore_eos" || echo "nc/ncmb.py: the old loop")): MB agg_dec_tps / decode_share / per-stream p50 / the client mix, errors 0 (else a request-error line); the ARC guard -> $O/c8.jsonl"; return 0; }
  j=$(grep -m1 '^MB ' $base.txt | cut -c4-)
  [ -n "$j" ] || { xlog "c8 $nm: VIRHE request error: no MB line ($(tail -3 $base.txt | tr '\n' ' ' | cut -c1-300))"; return 1; }
  ad=$(arcd $nm-0 $nm-1); g=$(arcg $a "$ad" "c8 block $i")
  python3 -c 'import json,sys
d = json.loads(sys.argv[1])
print(json.dumps(dict(name=sys.argv[2], arm=sys.argv[3], tool=sys.argv[5], agg_dec_tps=d.get("agg_dec_tps"), decode_share=d.get("decode_share"), per_stream_dec_p50=d.get("per_stream_dec_p50"),
      ttft_p50=d.get("ttft_p50"), requests=d.get("requests"), errors=d.get("errors"), temperature=d.get("temperature"), sched=d.get("sched"), window=d.get("window"),
      window_timeout=d.get("window_timeout"), mix_client=d.get("mix_client"), arc=json.loads(sys.argv[4] or "{}"), guard=sys.argv[6])))' \
    "$j" "$nm" "$a" "$ad" "$tool" "$g" >> $O/c8.jsonl
  xlog "c8 $nm ($(NAMEOF $a) $(armj $a), mb c8 T 0 via $tool, $(( $(date +%s) - t0 )) s): $(tail -1 $O/c8.jsonl | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
print("agg %s tok/s, decode_share %s, per-stream p50 %s, TTFT p50 %s, %s requests, %s errors%s | sched %s, window %s%s, client mix %s" % (d.get("agg_dec_tps"), d.get("decode_share"), d.get("per_stream_dec_p50"), d.get("ttft_p50"),
      d.get("requests"), d.get("errors"), "" if d.get("errors") == 0 and (d.get("requests") or 0) > 0 else " -- VIRHE request errors", d.get("sched") or "mb", json.dumps(d.get("window"))[:120],
      (" (window timeout: %s)" % d.get("window_timeout")) if d.get("window_timeout") else "", json.dumps(d.get("mix_client"))[:200]))' 2>&1 | cut -c1-700) | arc $ad$([ -n "$g" ] && echo " -- $g") | INFO $(capd $nm-0 $nm-1) | INFO counters: $(ctrinfo $nm-0 $nm-1 | cut -c1-400)"
}
c8s() {   # (b) the c8 blocks in C8ORDER (B C C B), then one engine-log capture; per block c8mix.py block (the co-batch mix and the newcomer
          # step walls), c8mix.py compare B1 vs B4 (the B-vs-B mix check) and over every block (INFO)
  local i=0 a n=${#C8ORDER} ins="" b1="" bl="" b
  [ -n "$C8ORDER" ] || return 0
  budok "c8 (b)" $(( n * 90 / 60 + 1 )) || return 0
  if [ -s $NCMB2 ] && [ -s $MBT2 ]; then xlog "c8: harness/mbt2.py used through harness/ncmb2.py (TESTING.md 12.11.2026 c8 block mix: fixed, seeded schedule, pattern 1x8, seed 1, window gens $C8GENS = 16 newcomer steps a block, edge 0.5, ignore_eos; order $C8ORDER, blocks 1 and $n the B-vs-B mix check; mbt2.py $(md5sum < $MBT2 | cut -c1-8), ncmb2.py $(md5sum < $NCMB2 | cut -c1-8), c8mix.py $(md5sum < $C8MIX 2>/dev/null | cut -c1-8))"
  else xlog "c8: harness/mbt2.py / ncmb2.py NOT present at run time -- nc/ncmb.py runs the blocks (mb's closed loop, a time window $C8WARM..$C8DUR s; old-loop numbers)"; fi
  for a in $(echo $C8ORDER | grep -o .); do i=$((i + 1)); stopfile && return 0; c8x $i $a; done
  capture $O/c8-engine.log.gz
  [ "$DRY" = 1 ] && { echo "+ per block: python3 -I -B $C8MIX block --log $O/c8-engine.log.gz --reqs $O/c8/<b>-reqs.jsonl --tag xarc-<b> --mbline $O/c8/<b>.txt --label <b> --json $O/c8/<b>-mix.json; then $C8MIX compare --in B1=$O/c8/c8-1B-mix.json --in B$n=$O/c8/c8-${n}B-mix.json (MATCH: the C-vs-B delta is read; else INCONCLUSIVE (mix) with the numbers) + compare over every block (INFO)"; return 0; }
  [ -s $C8MIX ] || { xlog "c8: c8mix.py absent -- no mix reading, the C-vs-B delta is printed as INCONCLUSIVE (mix)"; return 0; }
  i=0
  for a in $(echo $C8ORDER | grep -o .); do
    i=$((i + 1)); b=c8-$i$a
    [ -s $O/c8/$b-reqs.jsonl ] || { xlog "c8 $b: no taps (block not run)"; continue; }
    timeout 300 python3 -I -B $C8MIX block --log $O/c8-engine.log.gz --reqs $O/c8/$b-reqs.jsonl --tag xarc-$b --mbline $O/c8/$b.txt --label $b --json $O/c8/$b-mix.json > $O/c8/$b-mix.out 2>&1
    xlog "c8 mix $b ($(NAMEOF $a)): $(grep -m1 '^C8MIX ' $O/c8/$b-mix.out | cut -c7- | sed 's/ -- VIRHE / -- c8mix guard (INFO): /' | cut -c1-800)"
    [ -s $O/c8/$b-mix.json ] && ins="$ins --in $b=$O/c8/$b-mix.json"
    [ $a = B ] && [ -s $O/c8/$b-mix.json ] && { [ -n "$b1" ] || b1=$b; bl=$b; }
  done
  if [ -n "$b1" ] && [ "$bl" != "$b1" ]; then
    timeout 120 python3 -I -B $C8MIX compare --in $b1=$O/c8/$b1-mix.json --in $bl=$O/c8/$bl-mix.json > $O/c8/bb-compare.out 2>&1
    xlog "c8 B-vs-B mix check ($b1 vs $bl): $(grep "^C8MIXCMP $bl " $O/c8/bb-compare.out | cut -c10- | cut -c1-500)"
  else xlog "c8 B-vs-B mix check: not possible (fewer than two B blocks with a mix)"; fi
  [ -n "$ins" ] && timeout 120 python3 -I -B $C8MIX compare $ins > $O/c8/all-compare.out 2>&1
  while IFS= read -r l; do xlog "c8 mix compare (INFO): ${l:9:600}"; done < <(grep '^C8MIXCMP ' $O/c8/all-compare.out 2>/dev/null)
}
essayx() {   # (a) ONE 1u essay at T 0 under ARM (stepprobe essay EGEN): decode tok/s, ms a step, tok a step; the XP counters, the
           # streamed host MB and the routed / resident experts over it; the guard (C plans copies, B none)
  local a=$1 i=$2 nm=essay-$1$2 j ad gd
  arm $a || return 1
  ssnap $nm-0
  local w0 w1; w0=$(date +%s.%N)
  run "(cd $D && timeout 900 python3 $SPB $EPROMPTS $EGEN > $O/$nm.txt 2>&1)"
  w1=$(date +%s.%N)
  [ "$DRY" = 1 ] || echo "{\"name\": \"$nm\", \"arm\": \"$a\", \"t0\": $w0, \"t1\": $w1}" >> $O/essay-windows.jsonl
  ssnap $nm-1
  [ "$DRY" = 1 ] && { echo "+ RESULT x1ub read $a ($(NAMEOF $a)): stepprobe $EPROMPTS $EGEN at T 0 (fixed prompts, fixed length): a J line a prompt -- decode tok/s, ms / tok a decode step, acceptance; xp copied / routed / planned / dropped / rests, stream MB, routed / resident over the read -> $O/essay.jsonl; guard: C plans > 0 (or rests), B plans 0"; return 0; }
  local np nj; np=$(echo $EPROMPTS | tr ',' '\n' | grep -c .); nj=$(grep -c '^J ' $O/$nm.txt)
  [ "$nj" = "$np" ] || { xlog "read $a$i: VIRHE request error: $nj of $np J lines ($(tail -3 $O/$nm.txt | tr '\n' ' ' | cut -c1-300))"; return 1; }
  ad=$(arcd $nm-0 $nm-1); gd=$(arcg $a "$ad" "read $a$i")
  grep '^J ' $O/$nm.txt | cut -c3- | python3 -c 'import json,sys
xp = json.loads(sys.argv[2] or "{}")
for k, l in enumerate(sys.stdin):
    d = json.loads(l); d.update(arm=sys.argv[1], read=sys.argv[4], xp=xp if k == 0 else {}, guard=sys.argv[3]); print(json.dumps(d))' "$a" "$ad" "$gd" "$nm" >> $O/essay.jsonl
  xlog "read $a$i ($(NAMEOF $a) $(armj $a), 1u, stepprobe $EPROMPTS $EGEN, T 0): $(grep '^J ' $O/$nm.txt | cut -c3- | python3 -c 'import json,sys
print(" | ".join("%s %s tok/s, %s ms a step, %s tok a step, acc %s" % (d.get("prompt"), d.get("decode_tps"), d.get("ms_per_decode_step"), d.get("tokens_per_step"), d.get("acceptance")) for d in map(json.loads, sys.stdin)))' 2>&1 | cut -c1-500) | xp $ad$([ -n "$gd" ] && echo " | $gd")"
}
rowsec() {   # after the essays: one engine-log capture, then 1ub/rowms.py -- ms a decode step by verify-row count a window and a arm
  capture $O/essay-engine.log.gz
  [ "$DRY" = 1 ] && { echo "+ python3 -I -B $ROWMS $O/essay-engine.log.gz $O/essay-windows.jsonl > $O/rowms.txt: ROWMS a window (decode steps n_seq 1, consecutive: ms by n_tok = verify rows; rows_mean) + ROWMSARM a arm"; return 0; }
  [ -s $O/essay-windows.jsonl ] || { xlog "rows: no essay window recorded"; return 0; }
  timeout 300 python3 -I -B $ROWMS $O/essay-engine.log.gz $O/essay-windows.jsonl > $O/rowms.txt 2>&1
  while IFS= read -r l; do xlog "rows ${l:0:900}"; done < <(grep -E '^ROWMS|Traceback' $O/rowms.txt)
}
idcheck() {   # (d) exactness: the greedy set alone under B then C (strict: VIRHE on a difference); the greedy set at conc 4 under B vs d0ab's A
  local l
  budok "identity (d)" 3 || return 0
  alone x1ub B a-B
  alone x1ub C a-C
  cmpx x1ub C-vs-B $O/x1ub-a-B.jsonl $O/x1ub-a-C.jsonl
  if [ "$DRY" != 1 ]; then
    l=$(grep "^x1ub-C-vs-B " $O/identity.txt 2>/dev/null | tail -1)
    case "$l" in *IDENTICAL*) case "$l" in *differ*|*DIFFER*) xlog "identity greedy set C vs B: VIRHE $l" ;; esac ;;
      *) xlog "identity greedy set C vs B: VIRHE ${l:-not compared} (a prefetched copy holds the pool slot's bytes: the text must match)" ;; esac
  fi
  [ "$DOC4" = 1 ] || return 0
  c4rr arc B c4-B
  cmpx arc c4-B-vs-d0ab-A $C4REF $O/arc-c4-B.jsonl
  [ "$DRY" = 1 ] && return 0
  l=$(grep "^arc-c4-B-vs-d0ab-A " $O/identity.txt 2>/dev/null | tail -1)
  case "$l" in *IDENTICAL*) case "$l" in *differ*|*DIFFER*) xlog "identity c4 B vs d0ab A: VIRHE $l" ;; esac ;;
    *) xlog "identity c4 B vs d0ab A (cross-start: int9 / draft_depth0 1 / M1 off there, int10 / draft_depth0 3 / M1 on here -- all exact): VIRHE ${l:-not compared}" ;; esac
}
c4rr() {   # c4rr TAG ARM NAME: pin read back, prefix-cache reset, greedyset.py's 20 prompts at conc 4 (d0ab's)
  local tag=$1 t0=$(date +%s)
  arm $2 || return 1; pcreset $tag-$3
  run "(cd $D && timeout 900 python3 $GS $URL $O/$tag-$3.jsonl --conc 4 --max-tokens $MT > $O/$tag-$3.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  xlog "$tag $3 (greedy set at conc 4, temperature 0, $MT tok, arm $(NAMEOF $2)): $(tail -1 $O/$tag-$3.txt | cut -c1-200), $(( $(date +%s) - t0 )) s"
}
summary() {   # the readings, one line each: 1u essays (C vs B), rows, c4 (C vs B, the mix check), identity, counters; a SUMMARY line
  [ "$DRY" = 1 ] && { echo "+ RESULT x1ub lines: ESSAY (decode tok/s, ms a step, tokens a step a arm, C vs B %; the xp copies used a step and their MB), ROWS (ms a step at 1 / 2 / 3 verify rows a arm), C4 (agg tok/s a arm, C vs B %, read only when the B-vs-B mixes MATCH), IDENTITY, COUNTERS, SUMMARY -- numbers, no verdict words beyond the guards"; return 0; }
  python3 - "$O" > $O/summary.txt 2>&1 <<'PY'
import json, os, re, statistics, sys
o = sys.argv[1]
def rows(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except OSError: return []
def mean(v):
    v = [x for x in v if isinstance(x, (int, float))]
    return statistics.mean(v) if v else None
def pc(a, b): return 100.0 * (a / b - 1) if a is not None and b else None
def f(x, k=2, pct=False): return "-" if x is None else ("%+.*f %%" % (k, x) if pct else "%.*f" % (k, x))
def num(d, *ks):
    for k in ks:
        v = d.get(k)
        if isinstance(v, (int, float)): return v
    return None
out, sm = [], []
UNIT_MB = 9.65   # one DS EXL3 expert unit (gate_up + down planes, ~9.2 MiB) -- the MB a used copy keeps off the link
es = rows(os.path.join(o, "essay.jsonl"))
if es:
    for pr in sorted({e.get("prompt") for e in es}):
        per = {}
        for a in "BC":
            r = [e for e in es if e.get("arm") == a and e.get("prompt") == pr]
            per[a] = (mean([num(e, "decode_tps") for e in r]), mean([num(e, "ms_per_decode_step") for e in r]),
                      mean([num(e, "tokens_per_step") for e in r]), [num(e, "decode_tps") for e in r])
        b, c = per["B"], per["C"]
        out.append("READ %s 1u (T 0, %d B / %d C reads): decode C %s tok/s vs B %s = %s (reads B %s, C %s) | ms a step C %s vs B %s | tokens a step C %s vs B %s" % (
            pr, len(b[3]), len(c[3]), f(c[0]), f(b[0]), f(pc(c[0], b[0]), 2, True), [v for v in b[3] if v], [v for v in c[3] if v],
            f(c[1]), f(b[1]), f(c[2], 3), f(b[2], 3)))
        sm.append("%s %s" % (pr, f(pc(c[0], b[0]), 2, True)))
    for a in "BC":
        x = [e.get("xp") or {} for e in es if e.get("arm") == a]
        cop, rou, pla = sum(d.get("copied") or 0 for d in x), sum(d.get("routed") or 0 for d in x), sum(d.get("planned") or 0 for d in x)
        steps = sum((num(e, "decode_steps") or 0) for e in es if e.get("arm") == a)
        smb = sum(d.get("stream_mb") or 0 for d in x)
        out.append("XP %s reads: %s planned, %s of %s restored copies routed (%s)%s | stream MB (heat estimate, the ring not seen) %s over %d decode steps" % (
            a, pla, rou, cop, (f(100.0 * rou / cop, 1) + " %") if cop else "-",
            (", ~%.1f MB a step kept off the link" % (rou * UNIT_MB / steps)) if steps else "", f(smb, 0), steps))
try:
    rl = [json.loads(l[9:]) for l in open(os.path.join(o, "rowms.txt")) if l.startswith("ROWMSARM ")]
except OSError:
    rl = []
if rl:
    ra = {r["arm"]: r for r in rl}
    def rw(a, k): return ((ra.get(a) or {}).get("rows") or {}).get(str(k), {}).get("p50")
    out.append("ROWS (ms a decode step by verify rows, p50; step log): " + " | ".join("%d row(s) C %s vs B %s" % (k, f(rw("C", k)), f(rw("B", k))) for k in (1, 2, 3)) +
               " | rows a step C %s vs B %s" % (f((ra.get("C") or {}).get("rows_mean"), 3), f((ra.get("B") or {}).get("rows_mean"), 3)))
c4 = [r for r in rows(os.path.join(o, "c4.jsonl")) if r.get("errors") == 0 and (r.get("requests") or 0) > 0]
if c4:
    tb = mean([r.get("agg_dec_tps") for r in c4 if r["arm"] == "B"]); tc = mean([r.get("agg_dec_tps") for r in c4 if r["arm"] == "C"])
    mix = ""
    try:
        m = [l for l in open(os.path.join(o, "c8", "c4-bb-compare.out")) if l.startswith("C8MIXCMP ")]
        mix = m[-1][9:].strip()[:200] if m else ""
    except OSError: pass
    ok = "MATCH" in mix
    xc = [r.get("xp") or {} for r in c4 if r["arm"] == "C"]
    out.append("C4 (fixed schedule 1x4, T 0, %d blocks): agg C %s vs B %s tok/s = %s%s | blocks %s | C: %s copies planned, %s of %s restored routed | B-vs-B mix: %s" % (
        len(c4), f(tc), f(tb), f(pc(tc, tb), 2, True), "" if ok else " (INCONCLUSIVE (mix): read with the mix check)",
        ["%s %s" % (r["name"], r.get("agg_dec_tps")) for r in c4], sum(d.get("planned") or 0 for d in xc), sum(d.get("routed") or 0 for d in xc),
        sum(d.get("copied") or 0 for d in xc), mix or "-"))
    sm.append("c4 %s%s" % (f(pc(tc, tb), 2, True), "" if ok else " (mix?)"))
try:
    ids = [l.strip() for l in open(os.path.join(o, "identity.txt")) if l.strip()]
except OSError:
    ids = []
out.append("IDENTITY: " + (" | ".join(ids)[:900] if ids else "none recorded"))
bad = [l for l in ids if "IDENTICAL" not in l or re.search("differ", l, re.I)]
sm.append("identity %s" % ("IDENTICAL" if ids and not bad else ("VIRHE" if ids else "-")))
for l in out: print(l)
print("SUMMARY " + " | ".join(sm))
PY
  while IFS= read -r l; do xlog "$l"; done < $O/summary.txt
}
dstart() {   # P0, then the screen start
  local tag=x1ub-n5 e k=0
  diskok $tag || return 1
  if ! p0sel; then
    if [ $SW = 1 ]; then xlog "P0: VIRHE a four-card selftest failed -> the item stops (no server start)"; return 1; fi   # (no P0 by default: lever 1U-XP has no kernel of its own to hold to a stock form)
    xlog "P0: failed under X1UB_SW=0 (a mock run: docker run --device is not emulated) -- the mock goes on"
  fi
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on" PROFEVERY=
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV > $XS2_ENVFILE"; else printf '%s\n' $LEVENV > $XS2_ENVFILE; fi
  xlog "$tag: start (SCREEN; $DESC) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $(pinj B | cut -c1-900) | env: $(env | grep -E '^(RADIANCE_|R4D_)' | grep -vE '(KEY|TOKEN|SECRET)' | sort | tr '\n' ' ' | cut -c1-2400)"
  gate1 $tag 70 || { xlog "$tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  modes $tag || { xlog "$tag: VIRHE the modes are not the start's -> stopped"; down; return 1; }
  drafted $tag || { down; return 1; }
  readback $tag || { keeplog $tag-readback; down; return 1; }
  vramstart $tag
  swnote $tag || { xlog "$tag: VIRHE the switches are not as expected -> stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  stopfile || { [ -n "$EORDER" ] && for e in $(echo $EORDER | grep -o .); do k=$((k + 1)); stopfile || { budok "essay $e$k (a)" 1 && essayx $e $k; }; done; }
  stopfile || { [ -n "$EORDER" ] && rowsec; }
  stopfile || { [ $DOD = 1 ] && [ -n "$C4ORDER" ] && budok "c4 blocks (b)" $(( ${#C4ORDER} * (DDUR + 15) / 60 + 1 )) && dblock; }
  stopfile || { [ $DOID = 1 ] && idcheck; }
  capture $O/$tag-engine-full.log.gz
  xlog "$tag: at the end: $(pskstat) | $(arcnow) | prefetch lines: $(docker logs --since 6h $CN 2>&1 | grep -E 'expert prefetch(:| resting)' | sed 's/^radiance: //; s/^[A-Z]\[[0-9]\] //; s/^[A-Z] *//' | cut -c1-200 | sort -u | head -4 | tr '\n' '|') | $(docker logs --since 6h $CN 2>&1 | grep -cE '^E ') E lines | $(elapsed) min"
  down
  xlog "$tag: done"
}
trap 'down' EXIT
trap 'exit 143' TERM INT
# ---------------------------------------------------------------- main
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19))$([ $SW = 1 ] || echo " | VIRHE X1UB_SW=0: NO int9 / int9b / int10 / int11 / int11b switch posted, the XP counters INFO (a mock run only)") | SCREEN start: $DESC | (a) 1u essays T 0 [$EORDER] ($EGEN tok) | (b) mb c4 [$C4ORDER] dur $DDUR warm $DWARM | (c) identity greedy set c1 C vs B $([ $DOID = 1 ] && echo on || echo off) | guards: read back + 0 request errors + the XP counters (C plans copies, B none) + byte identity | budget $BUDGET min | no evict"
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
# int9b's own (VIRHE when missing; SKIP only under DRYRUN=1 + X1UB_DRY_ALLOW_OLD=1, a dry run against an image without them)
IMGLITS9B="B|stage_full_pick, mhc_fuse, ix_lds, ix_ni, ds41_arcoded, sched_row_cap, ds41_mix_wv, ds41_gq, ds41_e2, sched_cb_hold_ms, sched_cb_hold_k, ds41_xp, epoch_bump)
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
# int10's own (VIRHE when missing; SKIP only under DRYRUN=1 + X1UB_DRY_ALLOW_OLD=1): lever ARC + draft_depth0 3
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
P|int10-PROVENANCE.txt
B|ds41_mix_wv
B|\"i11\": {\"gq\": {\"declared\":
A|lever 1U-GQ (ds41_gq, RADIANCE_DS41_GQ, now %d)
A|lever 1U-E2 (ds41_e2, RADIANCE_DS41_E2, now %d)
K|rad_mx_stats
P|int11-PROVENANCE.txt
B|ds41_xp
B|rad_expert_prefetch_route
B|\"xp\": {\"ranks\":
B|expert prefetch on: %d copies a routed layer by DMA
A|lever 1U-XP (ds41_xp, RADIANCE_DS41_XP, now %d)
A|RADIANCE_DS41_XP
P|int11b-PROVENANCE.txt"
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
    echo "+ image check: SKIP (image lacks int9b / int10 literals; X1UB_DRY_ALLOW_OLD=1, dry run only): missing $(echo "$MISS9" | tr ';' '\n' | grep -c .): $(echo "$MISS9" | cut -c1-700)"
  else
    xlog "VIRHE $IMAGE is not the int11b image -- missing: $(echo "$MISS9" | cut -c1-900)"; exit 1
  fi
fi
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)): all $(echo "$IMGLITS" | grep -c .) int9 / KVR / profiler literals present$([ -z "$MISS9" ] && echo " + all $(echo "$IMGLITS9B" | grep -c .) int9b literals (the registry tail now ending in ds41_arcoded, sched_row_cap) + all $(echo "$IMGLITS10" | grep -c .) int10 literals (ds41_arcoded, GET arc counters, draft_depth0 3 and its start line, libr4d's ARC env / stats / coded launch name, r4d_selftest's ar_arcoded, int10-PROVENANCE)") + bin/rad-sidecar + int7 / int8 / int9 PROVENANCE -- OK"
HMISS=""
for f in "$LIB29" "$LIBPR" $Y $CTR $GS $LD $NCMB $DQ $MBP $TESTS $SPB $D/fastcheck.py $D0B $HD/../d0ab/s5ctr.py $MBT $EP $PRB; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $CTR $GS $LD $NCMB $DQ $MBP $TESTS $SPB $D/fastcheck.py $D0B $MBT $EP $PRB 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
CTT_=$(timeout 60 python3 -I -B $CTR selftest 2>&1 | tail -1); case "$CTT_" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$CTR(${CTT_:0:300})" ;; esac
ST=$(timeout 60 python3 -I $LD selftest --gs $GS --mb $MBP 2>&1 | tail -1); case "$ST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$LD(${ST:0:300})" ;; esac
NMT=$(timeout 60 python3 -I -B $NCMB --mb $MBP --selftest 2>&1 | tail -1); case "$NMT" in *OK*) ;; *) HMISS="$HMISS selftest:$NCMB(${NMT:0:300})" ;; esac
if [ -s $NCMB2 ] && [ -s $MBT2 ]; then   # the harness (present: it runs the c8 windows; absent at run time: ncmb.py, logged)
  M2T=$(timeout 60 python3 -I -B $MBT2 --mb $MBP --selftest 2>&1 | tail -1); case "$M2T" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$MBT2(${M2T:0:300})" ;; esac
  N2T=$(timeout 60 python3 -I -B $NCMB2 --mb $MBP --selftest 2>&1 | tail -1); case "$N2T" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$NCMB2(${N2T:0:300})" ;; esac
  CMT=$(timeout 60 python3 -I -B $C8MIX selftest 2>&1 | tail -1); case "$CMT" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$C8MIX(${CMT:0:300})" ;; esac
else M2T="absent"; N2T="absent"; CMT="absent"; fi
python3 -I -B $D0B --help > /dev/null 2>&1 || HMISS="$HMISS run:d0blocks.py--help"
grep -q '"essay"' $TESTS || HMISS="$HMISS stepprobe-prompts:$TESTS"
RMT=$(timeout 60 python3 -I -B $ROWMS selftest 2>&1 | tail -1); case "$RMT" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$ROWMS(${RMT:0:300})" ;; esac
[ -s $NCMB2 ] && [ -s $MBT2 ] && [ -s $C8MIX ] || HMISS="$HMISS harness:ncmb2/mbt2/c8mix(the block rules need the fixed schedule)"
FNS="log run mkd restart check down stopfile best_vars desc serve_args xlog elapsed armj NAMEOF pinj stackenv keeplog capture thermline dfree diskok cmpfile idrec identw modes vramstart fn4get fn4set nudge fn4code pcreset arm msnap mdelta pskstat ssnap sdelta gate1 alone cmpx budok ctrinfo arcnow arcd arcg gpurun p0sel drafted readback swnote capd dblock coldx colds c8x c8s essayx rowsec c4x idcheck c4rr summary dstart"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee sha256sum gzip zcat; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
for a in B C; do python3 -c 'import json,sys; d=json.loads(sys.argv[1]); sw = sys.argv[2] == "1"; a = sys.argv[3]
assert d["ds41_psk"] == 4 and d["ds41_psk_cr"] == 1 and d["draft_sample"] == 0 and d["draft_lend"] == 1
if sw: assert d["mhc_fuse"] == int(sys.argv[4]) and d["ix_lds"] == 0 and d["ix_ni"] == 0 and d["draft_depth0"] == int(sys.argv[5]) and d["ds41_xp"] == (0 if a == "B" else 1) and d["stage_full_min"] == int(sys.argv[6]) and d["sched_row_cap"] == int(sys.argv[7]) and all(d[k] == 0 for k in ("ds41_arcoded", "ds41_mix_wv", "ds41_gq", "ds41_e2", "ds41_shexp_lane"))
else: assert d["draft_depth0"] == 1 and not any(k in d for k in ("mhc_fuse", "ix_lds", "ix_ni", "ds41_arcoded", "stage_full_min", "stage_full_pick", "sched_row_cap", "ds41_xp"))' "$(pinj $a)" $SW $a $MHC $D0 $SPN $CAP 2>/dev/null || HMISS="$HMISS json:$a"; done
[ "$(sha256sum $MODELDIR/$SIDECAR 2>/dev/null | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256 2>/dev/null)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR"
[ "$(sha256sum $MODELDIR/$SIDE 2>/dev/null | cut -c1-64)" = "$SIDESHA" ] || HMISS="$HMISS sha256:$MODELDIR/$SIDE(want ${SIDESHA:0:16})"
grep -qF XS2_ENVFILE $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no XS2_ENVFILE)"
grep -qE 'RADIANCE_DS41_KV_REPLICATED|RADIANCE_DS41_MHC_FUSE|RADIANCE_DS41_IX_|RADIANCE_DS41_ARCODED|RADIANCE_DECODE_ROW_CAP|RADIANCE_DS41_XP|RADIANCE_DS41_GQ|RADIANCE_DS41_E2|RADIANCE_XPREFETCH' $Y 2>/dev/null && HMISS="$HMISS yaml:$Y(names a lever word: they come from the env file only)"
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config refuses it)"
case "$LEVENV" in *RADIANCE_DS41_MHC_FUSE=0*) ;; *) HMISS="$HMISS levenv:no-M1-declare" ;; esac
case "$LEVENV" in *RADIANCE_DS41_XP=1*) ;; *) HMISS="$HMISS levenv:no-XP-link" ;; esac
case "$LEVENV" in *RADIANCE_XPREFETCH=1*) ;; *) HMISS="$HMISS levenv:no-XPREFETCH=1" ;; esac
case "$LEVENV" in *RADIANCE_DS41_KV_REPLICATED*|*RADIANCE_DS41_ARCODED*|*RADIANCE_DECODE_ROW_CAP*) HMISS="$HMISS levenv:KVR-ARC-or-CAP-at-start" ;; esac
HMISS=$(echo $HMISS)
[ -z "$HMISS" ] || { xlog "VIRHE helpers / inputs do not resolve: $HMISS"; exit 1; }
if [ $DOC4 = 1 ] && [ ! -s $C4REF ]; then xlog "note -- the c4 identity reference $C4REF is missing: the c4 greedy set vs d0ab A is not run"; DOC4=0; fi
HMD5="i9b/i9bctr.py $(md5sum < $CTR | cut -c1-8) (INFO only), d0ab/d0blocks.py $(md5sum < $D0B | cut -c1-8) + s5ctr.py $(md5sum < $HD/../d0ab/s5ctr.py | cut -c1-8), stack4/loadid.py $(md5sum < $LD | cut -c1-8), stack4/mbt.py $(md5sum < $MBT | cut -c1-8), nc/ncmb.py $(md5sum < $NCMB | cut -c1-8), harness mbt2.py $(md5sum < $MBT2 2>/dev/null | cut -c1-8) / ncmb2.py $(md5sum < $NCMB2 2>/dev/null | cut -c1-8) / c8mix.py $(md5sum < $C8MIX 2>/dev/null | cut -c1-8), deepq.py $(md5sum < $DQ | cut -c1-8), mb.py $(md5sum < $MBP | cut -c1-8), stepprobe.py $(md5sum < $SPB | cut -c1-8), greedyset.py $(md5sum < $GS | cut -c1-8), ds41_tests.py $(md5sum < $TESTS | cut -c1-8), dsm-1ub.sh $(md5sum < $0 | cut -c1-8) (from arc/dsm-arc.sh $(md5sum < $HD/../arc/dsm-arc.sh | cut -c1-8)), lib-29.sh $(md5sum < $LIB29 | cut -c1-8), yaml $(md5sum < $Y | cut -c1-8)"
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing; i9bctr.py $CTT_ | loadid.py $ST | ncmb.py $NMT | d0blocks.py --help OK | harness: mbt2.py $M2T / ncmb2.py $N2T / c8mix.py $CMT | all $(echo $FNS | wc -w) functions resolve, tools on PATH, both arms' pins (S5T, draft_sample 0, draft_lend 1$([ $SW = 1 ] && echo ", SP 0, M1 on, IX off, draft_depth0 $D0, sched_row_cap $CAP, ds41_arcoded / ds41_mix_wv / ds41_gq / ds41_e2 / ds41_shexp_lane 0, ds41_xp 0 / 1" || echo "; X1UB_SW=0: no int9 .. int11b switch")), sidecars $SIDECAR + $SIDE sha256 OK, yaml accepted (env_file, no lever word), LEVENV with MHC_FUSE=0 + GQ=3 + E2=1 + XP=1 + XPREFETCH=1, without KVR / ARC -- OK | md5: $HMD5"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { xlog "VIRHE $M225 missing or not verified"; exit 1; }
  [ -s $WARM_REF_225 ] || xlog "note -- reference $WARM_REF_225 missing (not compared)"
  xlog "$ITEM: helpers md5: $HMD5"
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDE; do [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  for f in $WARM_REF_225 $RS5/s5-n5-stats0.json $RS5/guard-s5-n5.txt $C4REF; do [ -e $f ] && echo "+ present (reference): $f" || echo "+ absent (reference, information only): $f"; done
  for a in B C; do echo "+ arm $a ($(NAMEOF $a)): $(armj $a)"; done
  echo "+ base pin + fixed pin: $(pinj B)"
  echo "+ readings: (a) essays [$EORDER] $EGEN tok, T 0 | (b) c4 [$C4ORDER] dur $DDUR warm $DWARM (d0blocks.py, one invocation) | (c) identity $DOID (greedy set c1 C vs B)"
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(information here; the real run refuses to start beside one)"
  echo "+ disk now: $(dfree) MB free on / (guard $DISKMIN MB)"
fi
mkd $O $O/guard $O/c8 $O/snap $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt
stopfile || dstart
summary
unset $SWITCHES
down
xlog "$ITEM: done ($(elapsed) min) -- $O | $([ $NV = 0 ] && echo "no VIRHE line" || echo "$NV VIRHE line(s): see above")"
