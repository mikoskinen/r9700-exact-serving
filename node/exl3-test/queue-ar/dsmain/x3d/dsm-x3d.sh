#!/bin/bash
# dsm-x3d.sh (12.11.2026, queue-ar/dsmain/x3d, the X3D-F builder for ds-main-port) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r00zzz-cz-x3d-g3.sh. Log tag "RESULT xx3d ", out runs/xdsm/x3d (X3DS_OUT). Made from a COPY of
# ngreuse/dsm-ngreuse.sh (92204f57: the start, LEVENV / LITS, the base pin, stackenv .. gate1, readback, the harness block runner
# blk, coldx, startinfo, the image literal check; tag xngr -> xx3d) + arc/dsm-arc.sh's in-process arm pieces (swnote, essayx,
# idcheck, the cold answers C vs B, the c8 mix check); the originals are not changed. Helpers read only: harness/ncmb2.py +
# mbt2.py + c8mix.py, nc/ncmb.py (the fallback only), stack4/loadid.py, i9b/i9bctr.py (INFO), deepq.py, mb.py, stepprobe.py,
# greedyset.py, ds41_tests.py; tools/ngramshare/ngshare.sh SOURCED as its README says (TESTING.md 12.11.: n-gram reuse for all
# new DS items).
# GATE 3 OF X3D-F (the engine port): the int10x image = int10 (f6ef7cc) + ds41-x3d-eng10 0488213 (fn4 exl3_x3d 57, a plugin
# switch, kSwitches 58) + the ds41-x3d exl3r4d 579d0bc (K = 2's decode reads a lane's eight trellis states from ONE word pair,
# the decode of a step's tiles phase-ordered; the same f16 weights by construction: g12b 408 / 408 rows identical on the card,
# gate_up -7.3 % / down -9.0 % a GEMM call at M 1-16). The switch only moves the K = 2 decode-band GEMMs (M <= 16, split-K 1).
#   B  exl3_x3d 0 (the shipped decode; the plugin's other 902 kernels are ISA-identical to 5c13e55's)
#   C  exl3_x3d 1 (X3D-F)
# ONE START = the ngreuse / spread base (stack5's + RADIANCE_DS41_MHC_FUSE=0; KVR / SP off) on the ngshare path (ds41-ngshare-
#   tp4.yaml, NGRAM=ram=/ngram: fill or reuse, whichever the tables allow). BASE PIN = S5T, draft_sample 0, SP 0 / 0, mhc_fuse 1,
#   ix_lds 0, ix_ni 0, draft_depth0 3, sched_row_cap 48, ds41_arcoded 0; only exl3_x3d moves. Every reading is preceded by the arm's
#   POST (pin + arm + epoch_bump, read back), so both arms pay the same re-record.
# READINGS (TESTING.md 12.11. block rules for DS screens; T 0; harness/ncmb2.py fixed schedule 1xN seed 1):
#   identity B (greedyset.py's 20 prompts c1 alone) | c4 warm-up (B, discarded) | c8 B, c8 C (one block a arm, time window
#   X3DS_C8WARM 10 .. X3DS_C8DUR 45 = ~35 s) | c4 C, c4 B (gens X3DS_GENS4 2, one a arm after the warm-up) | c1 B C C B (gens
#   X3DS_GENS1 6, two a arm) | 1u essay B C C B (stepprobe essay X3DS_EGEN 512, two reads a arm) | cold 64k B, C (deepq
#   X3DS_DEPTH 64000 after a reset, X3DS_NTOK 64 tokens: TTFT; the two answers compared byte for byte) | identity C (the greedy set
#   again, C vs B text + token ids strict: loadid.py cmp). Then the engine log once: c8mix.py block for c8 B / C + compare (a c8
#   delta < 2 % with differing mixes is INCONCLUSIVE).
# SUMMARY: C vs B % a reading (c8 / c4 agg, per-stream p50, ms a decode step; c1 agg mean of two + its B pair spread; essay decode
#   tok/s mean of two + ms a step; 64k TTFT), identity, and INFO: B against int10's own numbers on the same harness (ngreuse
#   B-c8 / B-c1 / C-c1 / B-cold on the ngshare path; spread c4-2 / c4-3 and essay-1 / -2 on the copy path) -- not a gate.
#   "VERDICT x3d: C vs B ..." with the identity verdict.
# GUARDS: exl3_x3d served (0 at start, 2 refused, B / C read back), the pin read back, 0 request errors, identity (a difference =
#   VIRHE). The plugin has no counter for the X3D kernels: that the path ran shows only as the C-vs-B time (INFO).
# ~24 min estimate (start ~3.5-5.5 (reuse / fill) + checks ~1.5, switch note ~0.5, identity 2 x ~65 s, c4 3 x ~50 s, c8 2 x
#   ~60 s, c1 4 x ~45 s, essay 4 x ~15 s, cold 2 x ~20 s, mixes ~1, arm posts 17 x ~3 s); BUDGET X3DS_BUDGET_MIN 28 (a reading
#   that would end past it is skipped, logged). NO evict, no cc_sim, no power / host / fan settings; production compose files
#   read only. The n-gram tables stay in /dev/shm/ds41ng for the next item (ngshare.sh's design; X3DS_NGCLEAN=1 cleans them).
# MOCK: X3DS_SW=0 = a server without the int9 .. int10x switches (queue-ar/tools/mockrad): no such switch posted; the start line
#   carries a "VIRHE X3DS_SW=0" marker (a mock run only).
# Knobs: X3DS_IMAGE, X3DS_ITEM, X3DS_OUT, X3DS_SW (1), X3DS_C8DUR (45), X3DS_C8WARM (10), X3DS_DUR (30), X3DS_WARM (10),
#   X3DS_GENS1 (6), X3DS_GENS4 (2), X3DS_EGEN (512), X3DS_DEPTH (64000), X3DS_NTOK (64), X3DS_MHC (1), X3DS_D0 (3), X3DS_CAP (48),
#   X3DS_SPN (0) / X3DS_SPK (0), X3DS_DISKMIN_MB (11000), X3DS_BUDGET_MIN (28), X3DS_NGCLEAN (0), X3DS_PLUGSHA (the int10x
#   exl3r4d's sha256), X3DS_SIDECAR, X3DS_SIDESHA; X3DS_DRY_ALLOW_OLD=1 (DRYRUN only).
# DRYRUN=1 bash dsm-x3d.sh : prints every command (the ngshare steps too); checks the image literals (int9 .. int10x) and the
#   plugin's sha256, every helper (files, parse, selftests), tools/ngramshare, the sidecars, the yaml, the disk; no server, no
#   GPU, writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/../ds41-stack2-tp4.yaml                   # stack2's yaml, read only (env_file ${XS2_ENVFILE}); ngshare.sh switches it (main)
NGS=$QA/tools/ngramshare
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh
O=${X3DS_OUT:-$R/xdsm/x3d}; export STATSDIR=$O/stats
RD=$R/xdsm/draft; RM=$R/xdsm; RS5=$R/xdsm/stack5
export IMAGE=${X3DS_IMAGE:?set X3DS_IMAGE}
ITEM=${X3DS_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-x3d-g3}
PSKK=8; BIC=3; NOA=2; OCC=6; OCCK=32; DSA=0; HBR=0; HBRANKS=1,2      # stack5's S5T values, draft_sample 0 (i9b's base)
SW=${X3DS_SW:-1}; C8DUR=${X3DS_C8DUR:-45}; C8WARM=${X3DS_C8WARM:-10}; DUR=${X3DS_DUR:-30}; WARM=${X3DS_WARM:-10}; GENS1=${X3DS_GENS1:-6}
GENS4=${X3DS_GENS4:-2}; EGEN=${X3DS_EGEN:-512}; DEPTH=${X3DS_DEPTH:-64000}; NTOK=${X3DS_NTOK:-64}; NGC=${X3DS_NGCLEAN:-0}
MHC=${X3DS_MHC:-1}; D0=${X3DS_D0:-3}; CAP=${X3DS_CAP:-48}; SPN=${X3DS_SPN:-0}; SPK=${X3DS_SPK:-0}; ARCN=0; BASEX=""
DISKMIN=${X3DS_DISKMIN_MB:-11000}; BUDGET=${X3DS_BUDGET_MIN:-28}; ALLOWOLD=${X3DS_DRY_ALLOW_OLD:-0}
PLUGSHA=${X3DS_PLUGSHA:-4245da82106e56c1a1e9b7f58221b2cf9af1b79c8e5eefc2628d5ba0c052ad5a}
SIDE=${X3DS_SIDECAR:-dspark-dlw-q8_0.rad}; SIDESHA=${X3DS_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
MT=192; SEED=1234
case $SW in 0|1) ;; *) echo "X3DS_SW is 1 (the item) or 0 (a server without the int9 .. int10x switches: the mock) (got '$SW')"; exit 1 ;; esac
for v in $C8DUR $C8WARM $DUR $WARM $GENS1 $GENS4 $EGEN $DEPTH $NTOK $NGC $MHC $D0 $CAP $SPN $SPK $DISKMIN $BUDGET; do case $v in ''|*[!0-9]*) echo "a numeric knob is not a number ('$v')"; exit 1 ;; esac; done
[ $C8WARM -lt $C8DUR ] && [ $WARM -lt $DUR ] || { echo "a warm-up is not shorter than its window"; exit 1; }
[ $GENS1 -ge 1 ] && [ $GENS4 -ge 1 ] && [ $NTOK -ge 1 ] && [ $EGEN -ge 64 ] || { echo "X3DS_GENS1 / X3DS_GENS4 >= 1, X3DS_NTOK >= 1, X3DS_EGEN >= 64"; exit 1; }
[ $D0 -le 3 ] && [ $CAP -le 4096 ] && [ $MHC -le 1048576 ] || { echo "X3DS_D0 0..3, X3DS_CAP 0..4096, X3DS_MHC 0..1048576"; exit 1; }
[ $SPN -le 1048576 ] && [ $SPK -le 64 ] || { echo "X3DS_SPN 0..1048576, X3DS_SPK 0..64"; exit 1; }
case $PLUGSHA in [0-9a-f]*) [ ${#PLUGSHA} = 64 ] || { echo "X3DS_PLUGSHA is a sha256 (64 hex)"; exit 1; } ;; *) echo "X3DS_PLUGSHA is a sha256 (64 hex)"; exit 1 ;; esac
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
S4=$HD/../stack4
LD=$S4/loadid.py                                                                                    # stack4's, read only
CTR=$HD/../i9b/i9bctr.py                                                                            # i9b's (INFO only), read only
NCMB=$HD/../nc/ncmb.py                                                                              # nc's, read only (the fallback only)
MBT2=$HD/../harness/mbt2.py; NCMB2=$HD/../harness/ncmb2.py; C8MIX=$HD/../harness/c8mix.py          # the harness (read only)
DQ=$QA/deepq.py; MBP=$D/mb.py; TESTS=$D/ds41_tests.py; SPB=$D/stepprobe.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
T00=$(date +%s); NV=0; NGB=0; NGCLEAN=0
xlog() { case "$*" in *VIRHE*) NV=$((NV + 1)) ;; esac; log "RESULT xx3d $*"; }
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
SPJ=", \"stage_full_min\": $SPN, \"stage_full_pick\": $SPK"; [ $SW = 1 ] || SPJ=""        # X3DS_SW=0: the mock has no int9 switch
TJ="{$S5J, \"ds41_psk\": 4, \"ds41_psk_cr\": 1, \"stage_short\": 0, \"stage_pick\": 0, \"place_hb\": 0$SPJ}"   # = stack5's armj T, draft_sample 0, + SP
MJ="{\"mhc_fuse\": $MHC, \"ix_lds\": 0, \"ix_ni\": 0, \"draft_depth0\": $D0, \"sched_row_cap\": $CAP, \"ds41_arcoded\": 0}"; [ $SW = 1 ] || MJ='{}'   # i9b's "M" arm (M1 on, IX-LDSR2 / IX-NI off) + draft_depth0 3 + CAP48 + ARC off (int10)
armj() {   # armj B|C: the arm's exl3_x3d (JSON) on top of the base; {} under X3DS_SW=0 (the mock)
  [ $SW = 1 ] || { echo '{}'; return 0; }
  case $1 in
    B) echo '{"exl3_x3d": 0}' ;;
    C) echo '{"exl3_x3d": 1}' ;;
  esac
}
NAMEOF() { case $1 in B) echo "B (exl3_x3d 0: the shipped K = 2 decode)" ;; C) echo "C (exl3_x3d 1: X3D-F)" ;; W) echo "W (B's pin: the discarded c4 warm-up)" ;; esac; }
pinj() {   # pinj ARM [EXTRAJSON]: the fixed pin + the base (S5T, draft_sample 0, SP, M1 on, IX off, X3DS_BASEX) + the arm + extra as one POST body
  local x=${2:-} bx=$BASEX; [ -n "$x" ] || x='{}'; [ -n "$bx" ] || bx='{}'
  python3 -c 'import json,sys
d = {}
for a in sys.argv[1:]: d.update(json.loads(a))
print(json.dumps(d))' "$FIX" "$TJ" "$MJ" "$bx" "$(armj $1)" "$x"
}
DESC="NST 5 stack5 start (stack4's + RADIANCE_LOG_STEPS=1, E1 declared off, HB ranks $HBRANKS) + RADIANCE_DS41_MHC_FUSE=0 (M1 declared) | KVR OFF (RADIANCE_DS41_KV_REPLICATED unset), SP off | base pin = stack5's S5T with draft_sample 0 + SP pinned (stage_full_min $SPN, stage_full_pick $SPK) + M1 on / IX off / draft_depth0 $D0 (int10: depth 0 by occupancy only) / sched_row_cap $CAP (CAP48) $MJ + the fixed pin (ds41_psk_k $PSKK; draft_lend 1)$([ -n "$BASEX" ] && echo " + X3DS_BASEX $BASEX") | ARMS in process (fn4, read back before every reading): B exl3_x3d 0, C exl3_x3d 1"
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
# ---------------------------------------------------------------- copied from stack5/dsm-stack5.sh (63649414; tag xs5 -> xx3d): stackenv .. gate1 (modes: + nc/dsm-ncprof-p.sh's rule-w check under --profile-ops)
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xx3d $1 warm-up identity vs $2"; return 0; }
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
  [ "$DRY" = 1 ] && { echo "+ GET /stats > $O/$1-stats0.json; RESULT xx3d $1 VRAM at start (experts per tier; vs stack4 s4-n5)"; return 0; }
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xx3d $tag $3: greedy set c1 alone under $(NAMEOF $2) $(armj $2): 20 of 20 answered (else a request-error line); counters INFO"; return 0; }
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
arcnow() {   # arcnow: GET /admin/fn4's exl3_x3d + ds41_arcoded + the int9b switches + draft_depth0 and the "arc" counters, one line
  [ "$DRY" = 1 ] && { echo "(dry)"; return 0; }
  fn4get | python3 -c 'import json,sys
d = json.load(sys.stdin); w = d.get("switches") or {}
print("exl3_x3d %s ds41_arcoded %s sched_row_cap %s mhc_fuse %s ix_lds %s ix_ni %s draft_depth0 %s | arc %s | cap %s" % (w.get("exl3_x3d"), w.get("ds41_arcoded"), w.get("sched_row_cap"), w.get("mhc_fuse"), w.get("ix_lds"), w.get("ix_ni"), w.get("draft_depth0"), json.dumps(d.get("arc")), json.dumps(d.get("cap"))))' 2>/dev/null | cut -c1-600
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
pinset() {   # pinset TAG: the pin under B (+ epoch_bump), read back
  [ "$DRY" = 1 ] && { echo "+ POST the pin once ($(pinj B | cut -c1-160)...) + epoch_bump, a nudge, every switch READ BACK (polled <= 30 s)"; return 0; }
  local g
  arm B || return 1
  g=$(fn4get)
  xlog "$1 pin (read back): $(echo "$g" | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print("switches %s | epoch %s" % (json.dumps({k: w.get(k) for k in ("exl3_x3d", "ds41_arcoded", "sched_row_cap", "mhc_fuse", "ix_lds", "ix_ni", "draft_depth0", "draft_lend", "draft_occ", "stage_full_min", "draft_sample", "ds41_psk", "ds41_bi")}), d.get("epoch")))' 2>/dev/null | cut -c1-700)"
}
startinfo() {   # startinfo TAG PATH: start-to-healthy seconds (restart's RESULT line) + the n-gram placement lines (copied / filled / reused, s)
  local tag=$1 p=$2 h ng s
  [ "$DRY" = 1 ] && { echo "+ RESULT xx3d start $tag ($p): healthy in N s (restart's line), the n-gram placement lines (copied / filled / reused + seconds) from $O/start-$tag.log -> $O/starts.jsonl"; return 0; }
  h=$(grep "RESULT restart-pr266-$tag: healthy in" $L 2>/dev/null | tail -1 | grep -oE 'healthy in [0-9]+ s' | grep -oE '[0-9]+')
  ng=$(grep -E 'placement: n-gram table' $O/start-$tag.log 2>/dev/null | sed -E 's/^[A-Z](\[[0-9]+\])? *//' | cut -c1-200 | sort | uniq -c | sed 's/^ *//' | tr '\n' '|')
  s=$(grep -E 'placement: n-gram table' $O/start-$tag.log 2>/dev/null | grep -oE 'in [0-9.]+ s' | grep -oE '[0-9.]+' | awk '{s+=$1} END{printf "%.1f", s+0}')
  python3 -c 'import json,sys; print(json.dumps(dict(tag=sys.argv[1], path=sys.argv[2], healthy_s=int(sys.argv[3]) if sys.argv[3] else None, ngram_s=float(sys.argv[4] or 0), ngram_lines=sys.argv[5][:900],
      reused=("reused" in sys.argv[5]), filled=("filled" in sys.argv[5]), copied=("copied" in sys.argv[5]))))' "$tag" "$p" "${h:-}" "${s:-0}" "${ng:-}" >> $O/starts.jsonl
  xlog "start $tag ($p): healthy in ${h:-?} s | n-gram placement (ranks x line): ${ng:-none} | n-gram seconds summed over the lines ${s:-0} s$(case "$p:$ng" in B*:*filled*|C*:*reused*|A*:*copied*|ngshare:*filled*|ngshare:*reused*) ;; *) echo " -- NOTE: not the expected placement for $p (A copied / B filled / C reused / ngshare filled or reused)" ;; esac)"
}
blk() {   # blk NAME N GENS|time: one mb cN block at T 0 on the harness's fixed, seeded schedule (pattern 1xN); "time" = the window
          # [X3DS_C8WARM, X3DS_C8DUR] (~35 s), else --window gens GENS; the MB line, TTFT p50 / p90 (taps), /metrics deltas; 0 errors
  local nm=$1 n=$2 g=$3 base=$O/blk/$1 t0 j tool win
  budok "block $nm (c$n)" $(( n == 8 ? 2 : 1 )) || return 0
  if [ "$g" = time ]; then win="--window time --dur $C8DUR --warm $C8WARM"; else win="--window gens --gens $g --dur $DUR --warm $WARM"; fi
  ssnap $nm-0; msnap $nm-0
  t0=$(date +%s)
  if [ -s $NCMB2 ] && [ -s $MBT2 ]; then
    tool=ncmb2
    run "(cd $D && timeout $((C8DUR + DUR + 900)) python3 -I -B $NCMB2 --mb $MBP --mode phase --n $n --temperature 0 --tag xx3d-$nm --reqs $base-reqs.jsonl --stats-prefix $base-st --samples $base-samp.jsonl --sched fixed --pattern 1x$n --seed 1 $win --edge 0.5 --ignore-eos 1 > $base.txt 2>&1)"
  else
    tool=ncmb
    run "(cd $D && timeout $((C8DUR + DUR + 900)) python3 -I -B $NCMB --mb $MBP --mode phase --n $n --temperature 0 --tag xx3d-$nm --dur $DUR --warm $WARM --reqs $base-reqs.jsonl --stats-prefix $base-st --samples $base-samp.jsonl > $base.txt 2>&1)"
  fi
  ssnap $nm-1; msnap $nm-1
  [ "$DRY" = 1 ] && { echo "+ RESULT xx3d block $nm (c$n, T 0 via $([ -s $NCMB2 ] && echo "harness/ncmb2.py: fixed schedule 1x$n seed 1, $win" || echo "nc/ncmb.py: the old loop")): agg tok/s, per-stream p50, TTFT p50 / p90, ms / tok a decode step, acceptance, decode share, the client mix, errors 0 -> $O/blocks.jsonl"; return 0; }
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
print(json.dumps(dict(name=sys.argv[2], phase=int(sys.argv[3]), start=sys.argv[2].split("-")[0], tool=sys.argv[6], agg_dec_tps=d.get("agg_dec_tps"), per_stream_dec_p50=d.get("per_stream_dec_p50"),
      ttft_p50=d.get("ttft_p50"), ttft_p50_taps=q(0.5), ttft_p90_taps=q(0.9), ttft_n=len(w), ms_decode_step=d.get("ms_decode_step"), tok_per_decode_step=d.get("tok_per_decode_step"),
      acceptance=d.get("acceptance"), decode_share=d.get("decode_share"), requests=d.get("requests"), errors=d.get("errors"), window_s=d.get("window_s"), window=d.get("window"),
      mix_client=d.get("mix_client"), window_timeout=d.get("window_timeout"), metrics=m)))' "$j" "$nm" "$n" "$base-reqs.jsonl" "$(mdelta $nm-0 $nm-1)" "$tool" >> $O/blocks.jsonl
  xlog "block $nm (c$n, T 0 via $tool, $([ "$g" = time ] && echo "time window $C8WARM..$C8DUR s" || echo "gens $g"), $(( $(date +%s) - t0 )) s): $(tail -1 $O/blocks.jsonl | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
print("agg %s tok/s, per-stream p50 %s, TTFT p50 %s s (taps p50 %s / p90 %s, n %s), %s ms / %s tok a decode step, acceptance %s, decode_share %s, window %s s, %s requests, %s errors%s | client mix %s" % (
    d.get("agg_dec_tps"), d.get("per_stream_dec_p50"), d.get("ttft_p50"), d.get("ttft_p50_taps"), d.get("ttft_p90_taps"), d.get("ttft_n"), d.get("ms_decode_step"), d.get("tok_per_decode_step"),
    d.get("acceptance"), d.get("decode_share"), d.get("window_s"), d.get("requests"), d.get("errors"),
    "" if d.get("errors") == 0 and (d.get("requests") or 0) > 0 else " -- VIRHE request errors", json.dumps(d.get("mix_client"))[:200]))' 2>&1 | cut -c1-800) | INFO $(capd $nm-0 $nm-1)"
}
coldx() {   # coldx NAME: deepq's fixed prompt at X3DS_DEPTH (64000) after a prefix-cache reset, X3DS_NTOK greedy tokens -> TTFT
  local nm=$1 j md
  budok "cold $nm" 1 || return 0
  pcreset $nm
  msnap $nm-0
  run "(cd $D && timeout 900 python3 $DQ $O/$nm.txt $DEPTH $NTOK > $O/$nm.out 2>&1)"
  msnap $nm-1
  [ "$DRY" = 1 ] && { echo "+ RESULT xx3d cold $nm: deepq $DEPTH cold, $NTOK tok: TTFT, cached, the non-decode step ms -> $O/cold.jsonl"; return 0; }
  j=$(grep -m1 '^DEEPQ ' $O/$nm.out | cut -c7-)
  [ -n "$j" ] || { xlog "cold $nm: VIRHE request error: no answer ($(tail -2 $O/$nm.out | tr '\n' ' ' | cut -c1-300))"; return 1; }
  md=$(mdelta $nm-0 $nm-1)
  python3 -c 'import json,sys; d = json.loads(sys.argv[1]); m = json.loads(sys.argv[2] or "{}"); d.update(name=sys.argv[3], start=sys.argv[3].split("-")[0], nondecode_ms=m.get("nondecode_ms"), nondecode_steps=m.get("nondecode_steps")); print(json.dumps(d))' "$j" "$md" "$nm" >> $O/cold.jsonl
  xlog "cold $nm (deepq $DEPTH after a reset, $NTOK greedy tokens): $(tail -1 $O/cold.jsonl | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
print("status %s, prompt %s tok, cached %s, TTFT %s s, %s tok | non-decode steps %s, %s ms mean%s%s" % (d.get("status"), d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("tokens"),
      d.get("nondecode_steps"), d.get("nondecode_ms"), "" if d.get("status") == 200 else " -- VIRHE request error", "" if not d.get("cached") else " -- NOT COLD (cached %s)" % d.get("cached")))' 2>&1 | cut -c1-400)"
}
# ---------------------------------------------------------------- the X3D screen's own (gate 3 of X3D-F)
swnote() {   # swnote TAG: exl3_x3d at start (0: the plugin serves it; -1 = no plugin serves it -> VIRHE); exl3_x3d 2 and -1 refused
             # (400); the arms B C B each applied and read back
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 at start: exl3_x3d 0 (served by the ds41-x3d exl3r4d; -1 = not served -> VIRHE)$([ $SW = 1 ] || echo " -- X3DS_SW=0: NOT CHECKED"), ds41_arcoded 0, epoch 0; POST exl3_x3d 2 / -1 -> 400; arms B C B each applied and read back ($(pinj C | cut -c1-120)...)"; return 0; }
  local s0 g c1=400 c2=400 a sets="" bad=""
  s0=$(fn4get)
  [ $SW = 1 ] && { c1=$(fn4code '{"exl3_x3d": 2}'); c2=$(fn4code '{"exl3_x3d": -1}'); }
  for a in B C B; do
    arm $a || bad="$bad $a"; g=$(fn4get)
    sets="$sets $a:$(echo "$g" | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print("x3d=%s,arc=%s,cap=%s,mhc=%s,d0=%s,ep=%s" % (w.get("exl3_x3d"), w.get("ds41_arcoded"), w.get("sched_row_cap"), w.get("mhc_fuse"), w.get("draft_depth0"), d.get("epoch")))' 2>/dev/null)"
  done
  xlog "$1 switch note: at start $(echo "$s0" | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print("switches %s | epoch %s" % (json.dumps({k: w.get(k) for k in ("exl3_x3d", "ds41_arcoded", "sched_row_cap", "mhc_fuse", "ix_lds", "ix_ni", "draft_depth0", "draft_sample", "gg_gemv", "aql_acq")}), d.get("epoch")))' 2>/dev/null | cut -c1-900) | exl3_x3d 2 -> HTTP $c1, exl3_x3d -1 -> HTTP $c2$([ $SW = 1 ] || echo " (X3DS_SW=0: not posted)") | arms applied (read back):$sets$([ -n "$bad" ] && echo " -- VIRHE not applied:$bad")"
  python3 -c 'import json,sys
s0 = json.loads(sys.argv[1]); sw = sys.argv[2] == "1"; w = s0["switches"]
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and s0["epoch"] == int(__import__("os").environ.get("ATTACH_EPOCH0") or 0) and w.get("draft_sample") == 0
if sw: ok = ok and w.get("exl3_x3d") == 0 and w.get("ds41_arcoded") == 0
sys.exit(0 if ok else 1)' "$s0" $SW 2>/dev/null && [ "$c1" = 400 ] && [ "$c2" = 400 ] && [ -z "$bad" ] && return 0
  xlog "$1: VIRHE the exl3_x3d switch is not served as asked (0 at start, 2 / -1 refused, B / C / B applied) -- stopped"; return 1
}
ablk() {   # ablk NAME ARM N GENS|time: the arm (pin + arm + epoch_bump, read back), then blk NAME N GENS|time
  local nm=$1 a=$2
  arm $a || { xlog "block $nm: VIRHE arm $a not applied -- no reading"; return 1; }
  [ "$DRY" = 1 ] && echo "+ arm $a = $(NAMEOF $a) $(armj $a) applied"
  blk $nm $3 $4
}
acold() {   # acold ARM: the arm, then coldx ARM-cold (deepq X3DS_DEPTH after a reset, X3DS_NTOK greedy tokens: TTFT); the answer is kept
  arm $1 || { xlog "cold $1-cold: VIRHE arm $1 not applied -- no reading"; return 1; }
  coldx $1-cold
}
coldid() {   # the two cold answers (B-cold.txt, C-cold.txt), byte for byte (a difference = VIRHE)
  local r
  [ "$DRY" = 1 ] && { echo "+ identity cold $DEPTH C vs B: cmpfile $O/C-cold.txt $O/B-cold.txt (byte for byte; a difference = VIRHE)"; return 0; }
  if [ ! -s $O/B-cold.txt ] || [ ! -s $O/C-cold.txt ]; then xlog "identity cold $DEPTH C vs B: not compared (missing an answer)"; echo "cold-$DEPTH-C-vs-B not compared" >> $O/identity.txt; return 0; fi
  r=$(cmpfile $O/B-cold.txt $O/C-cold.txt)
  case "$r" in IDENTICAL*) idrec "cold-$DEPTH-C-vs-B" "$r" "C-cold.txt vs B-cold.txt" ;; *) idrec "cold-$DEPTH-C-vs-B" "VIRHE $r" "C-cold.txt vs B-cold.txt" ;; esac
}
essayx() {   # essayx ARM I: ONE 1u essay at T 0 under ARM (stepprobe essay X3DS_EGEN): decode tok/s, ms a step, tok a step -> essay.jsonl
  local a=$1 i=$2 nm=essay-$1-$2 j
  budok "essay $a-$i" 1 || return 0
  arm $a || { xlog "essay $a-$i: VIRHE arm $a not applied -- no reading"; return 1; }
  ssnap $nm-0
  run "(cd $D && timeout 900 python3 $SPB essay $EGEN > $O/$nm.txt 2>&1)"
  ssnap $nm-1
  [ "$DRY" = 1 ] && { echo "+ RESULT xx3d essay $a-$i ($(NAMEOF $a) $(armj $a)): stepprobe essay $EGEN at T 0: decode tok/s, ms / tok a decode step, acceptance (a J line or a request-error line) -> $O/essay.jsonl"; return 0; }
  j=$(grep -m1 '^J ' $O/$nm.txt | cut -c3-)
  [ -n "$j" ] || { xlog "essay $a-$i: VIRHE request error: no J line ($(tail -3 $O/$nm.txt | tr '\n' ' ' | cut -c1-300))"; return 1; }
  python3 -c 'import json,sys; d = json.loads(sys.argv[1]); d.update(arm=sys.argv[2], slot=int(sys.argv[3]), name=sys.argv[4]); print(json.dumps(d))' "$j" "$a" "$i" "$nm" >> $O/essay.jsonl
  xlog "essay $a-$i ($(NAMEOF $a) $(armj $a), 1u, stepprobe essay $EGEN, T 0): $(echo "$j" | cut -c1-400)"
}
idchk() {   # idchk: the greedy sets a-B and a-C (alone, c1, T 0), C vs B text + token ids strict (loadid.py cmp; a difference = VIRHE)
  local l
  cmpx x3d C-vs-B $O/x3d-a-B.jsonl $O/x3d-a-C.jsonl
  [ "$DRY" = 1 ] && return 0
  l=$(grep "^x3d-C-vs-B " $O/identity.txt 2>/dev/null | tail -1)
  case "$l" in *IDENTICAL*) case "$l" in *differ*|*DIFFER*) xlog "identity greedy set C vs B: VIRHE $l" ;; esac ;;
    *) xlog "identity greedy set C vs B: VIRHE ${l:-not compared} (X3D-F decodes the same f16 weights: the text must match)" ;; esac
}
c8mixes() {   # after the engine-log capture: c8mix.py block for the c8 blocks (B-c8, C-c8) + compare (MATCH: the delta is read; else
              # a c8 delta < 2 % is INCONCLUSIVE (mix))
  local b ins=""
  [ "$DRY" = 1 ] && { echo "+ per c8 block: python3 -I -B $C8MIX block --log $O/x3d-engine.log.gz --reqs $O/blk/<b>-reqs.jsonl --tag xx3d-<b> --mbline $O/blk/<b>.txt --label <b> --json $O/blk/<b>-mix.json; then $C8MIX compare --in B-c8=... --in C-c8=..."; return 0; }
  [ -s $C8MIX ] || { xlog "c8: c8mix.py absent -- no mix reading (the c8 delta is INCONCLUSIVE (mix) when under 2 %)"; return 0; }
  for b in B-c8 C-c8; do
    [ -s $O/blk/$b-reqs.jsonl ] || { xlog "c8 mix $b: no taps (block not run)"; continue; }
    timeout 300 python3 -I -B $C8MIX block --log $O/x3d-engine.log.gz --reqs $O/blk/$b-reqs.jsonl --tag xx3d-$b --mbline $O/blk/$b.txt --label $b --json $O/blk/$b-mix.json > $O/blk/$b-mix.out 2>&1
    xlog "c8 mix $b: $(grep -m1 '^C8MIX ' $O/blk/$b-mix.out | cut -c7- | sed 's/ -- VIRHE / -- c8mix guard (INFO): /' | cut -c1-800)"
    [ -s $O/blk/$b-mix.json ] && ins="$ins --in $b=$O/blk/$b-mix.json"
  done
  if [ -s $O/blk/B-c8-mix.json ] && [ -s $O/blk/C-c8-mix.json ]; then
    timeout 120 python3 -I -B $C8MIX compare $ins > $O/blk/c8-compare.out 2>&1
    grep '^C8MIXCMP ' $O/blk/c8-compare.out > $O/c8mix-compare.txt 2>/dev/null
    xlog "c8 mix compare B-c8 vs C-c8: $(grep '^C8MIXCMP ' $O/blk/c8-compare.out | cut -c10- | tr '\n' '|' | cut -c1-600)"
  else xlog "c8 mix compare: not possible (a c8 block without a mix)"; fi
}
xstart() {   # xstart TAG: the one start (gate1: restart + the warm-up check), its checks, the switch note, the pin under B
  local tag=$1
  diskok $tag || return 1
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on" PROFEVERY=
  export NGRAM=ram=/ngram   # ngshare.sh's restart sets it too; here it shows in the start line
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV > $XS2_ENVFILE"; else printf '%s\n' $LEVENV > $XS2_ENVFILE; fi
  xlog "$tag: start (ngshare: ds41-ngshare-tp4.yaml, NGRAM=ram=/ngram, keeper; fill or reuse; $DESC) Y=$(basename $Y) IMAGE=$IMAGE MODEL=$MODEL $(desc)"
  gate1 $tag 70 || { xlog "$tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  startinfo $tag ngshare
  modes $tag || { xlog "$tag: VIRHE the modes are not the start's -> stopped"; down; return 1; }
  drafted $tag || { down; return 1; }
  readback $tag || { keeplog $tag-readback; down; return 1; }
  vramstart $tag
  swnote $tag || { keeplog $tag-switch; down; return 1; }
  pinset $tag || { xlog "$tag: VIRHE the pin is not applied -> stopped"; keeplog $tag-switch; down; return 1; }
}
readings() {   # the readings in their fixed order (module header); each arm posted (read back) before its reading
  local s
  for s in "alone x3d B a-B" "ablk W-c4 B 4 $GENS4" "ablk B-c8 B 8 time" "ablk C-c8 C 8 time" "ablk C-c4 C 4 $GENS4" \
           "ablk B-c4 B 4 $GENS4" "ablk B-c1-1 B 1 $GENS1" "ablk C-c1-1 C 1 $GENS1" "ablk C-c1-2 C 1 $GENS1" "ablk B-c1-2 B 1 $GENS1" \
           "essayx B 1" "essayx C 1" "essayx C 2" "essayx B 2" "acold B" "acold C" "alone x3d C a-C"; do
    stopfile && { xlog "stop file: the readings end before '$s'"; return 0; }
    case $s in alone*) budok "identity (${s##* })" 2 || continue ;; esac
    $s
  done
  idchk
  coldid
}
ngclean() {   # the end: ngram-shm-cleanup.sh (keeper stop + tables and .lock files removed), logged; /dev/shm before / after
  local r
  NGCLEAN=1
  run "docker rm -f $CN > /dev/null 2>&1"
  if [ "$DRY" = 1 ]; then echo "+ $NGS/ngram-shm-cleanup.sh   (keeper stop + /dev/shm/ds41ng/ngram-*.tab + .lock removed; logs to runs/queue-changes.log)"; return 0; fi
  local b; b=$(df -B1 --output=used /dev/shm 2>/dev/null | tail -1 | awk '{printf "%.1f GiB", $1 / 1073741824}')
  r=$("$NGS/ngram-shm-cleanup.sh" 2>&1 | tail -3 | tr '\n' ' ' | cut -c1-600)
  xlog "ngshare cleanup: $r | /dev/shm used $b -> $(df -B1 --output=used /dev/shm 2>/dev/null | tail -1 | awk '{printf "%.1f GiB", $1 / 1073741824}') | left: $(ls /dev/shm/ds41ng 2>/dev/null | tr '\n' ' ' | cut -c1-200)"
}
summary() {   # C vs B a reading; identity; INFO: B against int10's own numbers (ngreuse / spread, the same harness); the verdict line
  [ "$DRY" = 1 ] && { echo "+ RESULT xx3d lines: C vs B % for c8 / c4 (agg, per-stream p50, ms a decode step; the c8 mix compare), c1 (agg mean of two a arm + the pair spreads), essay (decode tok/s mean of two + ms a step), 64k TTFT; identity; INFO B vs int10 (ngreuse B-c8 / B-c1 / C-c1 / B-cold, spread c4-2 / c4-3 / essay-1 / -2); VERDICT x3d"; return 0; }
  python3 - "$O" "$RM" "$NV" > $O/summary.txt 2>&1 <<'PY'
import json, os, re, statistics, sys
o, rm, nv = sys.argv[1], sys.argv[2], int(sys.argv[3])
def rows(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except (OSError, ValueError): return []
def jline(p):
    try:
        for l in open(p):
            if l.startswith("J "): return json.loads(l[2:])
    except (OSError, ValueError): pass
    return None
ok = lambda r: r.get("errors") == 0 and (r.get("requests") or 0) > 0
bl = [r for r in rows(os.path.join(o, "blocks.jsonl")) if ok(r)]
es = [r for r in rows(os.path.join(o, "essay.jsonl")) if isinstance(r.get("decode_tps"), (int, float))]
cold = [r for r in rows(os.path.join(o, "cold.jsonl")) if r.get("status") == 200 and not r.get("cached")]
def num(rs, k):
    v = [r[k] for r in rs if isinstance(r.get(k), (int, float))]
    return statistics.mean(v) if v else None
def pc(a, b): return 100.0 * (a / b - 1) if a is not None and b else None
def f(x, k=2, p=False): return "-" if x is None else ("%+.*f %%" % (k, x) if p else "%.*f" % (k, x))
def spread(v): return 100.0 * (max(v) - min(v)) / statistics.mean(v) if len(v) >= 2 else None
get = lambda arm, ph: [r for r in bl if r.get("start") == arm and r.get("phase") == ph]
out = []; dl = {}
mixcmp = ""
try: mixcmp = open(os.path.join(o, "c8mix-compare.txt")).read()
except OSError: pass
mixs = "MATCH" if "MATCH" in mixcmp and "DIFFER" not in mixcmp else ("DIFFERS" if "DIFFER" in mixcmp else "no reading")
for ph, nm in ((8, "C8"), (4, "C4")):
    b, c = get("B", ph), get("C", ph)
    d = pc(num(c, "agg_dec_tps"), num(b, "agg_dec_tps")); dl[nm] = d
    extra = ""
    if ph == 8:
        extra = " | mix (c8mix compare) %s%s" % (mixs, " -> INCONCLUSIVE (mix): |delta| < 2 %% with differing mixes" if d is not None and abs(d) < 2 and mixs != "MATCH" else "")
        extra += " | client mix B %s / C %s" % (json.dumps((b[0].get("mix_client") if b else None))[:90], json.dumps((c[0].get("mix_client") if c else None))[:90])
    if ph == 4:
        w = get("W", 4)
        extra = " | the discarded warm-up W-c4 %s tok/s" % (f(num(w, "agg_dec_tps"), 1))
    out.append("%s C vs B (one block a arm): agg %s vs %s tok/s = %s | per-stream p50 %s vs %s = %s | ms a decode step %s vs %s = %s | tok a decode step %s vs %s | acceptance %s vs %s%s" % (
        nm, f(num(c, "agg_dec_tps"), 1), f(num(b, "agg_dec_tps"), 1), f(d, 2, True), f(num(c, "per_stream_dec_p50"), 1), f(num(b, "per_stream_dec_p50"), 1),
        f(pc(num(c, "per_stream_dec_p50"), num(b, "per_stream_dec_p50")), 2, True), f(num(c, "ms_decode_step")), f(num(b, "ms_decode_step")),
        f(pc(num(c, "ms_decode_step"), num(b, "ms_decode_step")), 2, True), f(num(c, "tok_per_decode_step")), f(num(b, "tok_per_decode_step")),
        f(num(c, "acceptance"), 3), f(num(b, "acceptance"), 3), extra))
b1, c1 = get("B", 1), get("C", 1)
d = pc(num(c1, "agg_dec_tps"), num(b1, "agg_dec_tps")); dl["C1"] = d
out.append("C1 C vs B (two blocks a arm, B C C B): agg %s (%s) vs %s (%s) tok/s = %s | pair spread B %s, C %s | ms a decode step %s vs %s = %s | tok a decode step %s vs %s" % (
    f(num(c1, "agg_dec_tps"), 1), " / ".join(str(r["agg_dec_tps"]) for r in c1), f(num(b1, "agg_dec_tps"), 1), " / ".join(str(r["agg_dec_tps"]) for r in b1), f(d, 2, True),
    f(spread([r["agg_dec_tps"] for r in b1]), 2) + " %", f(spread([r["agg_dec_tps"] for r in c1]), 2) + " %", f(num(c1, "ms_decode_step")), f(num(b1, "ms_decode_step")),
    f(pc(num(c1, "ms_decode_step"), num(b1, "ms_decode_step")), 2, True), f(num(c1, "tok_per_decode_step")), f(num(b1, "tok_per_decode_step"))))
eb, ec = [r for r in es if r.get("arm") == "B"], [r for r in es if r.get("arm") == "C"]
d = pc(num(ec, "decode_tps"), num(eb, "decode_tps")); dl["ESSAY"] = d
out.append("ESSAY 1u C vs B (two reads a arm, B C C B; stepprobe essay): decode %s (%s) vs %s (%s) tok/s = %s | ms a decode step %s vs %s = %s | tokens a step %s vs %s | pair spread B %s, C %s" % (
    f(num(ec, "decode_tps"), 2), " / ".join(str(r["decode_tps"]) for r in ec), f(num(eb, "decode_tps"), 2), " / ".join(str(r["decode_tps"]) for r in eb), f(d, 2, True),
    f(num(ec, "ms_per_decode_step")), f(num(eb, "ms_per_decode_step")), f(pc(num(ec, "ms_per_decode_step"), num(eb, "ms_per_decode_step")), 2, True),
    f(num(ec, "tokens_per_step")), f(num(eb, "tokens_per_step")), f(spread([r["decode_tps"] for r in eb]), 2) + " %", f(spread([r["decode_tps"] for r in ec]), 2) + " %"))
cb, cc = [r for r in cold if r.get("start") == "B"], [r for r in cold if r.get("start") == "C"]
d = pc(num(cc, "ttft"), num(cb, "ttft")); dl["64k TTFT"] = d
out.append("COLD 64k C vs B (one each): TTFT %s vs %s s = %s | non-decode step %s vs %s ms" % (f(num(cc, "ttft")), f(num(cb, "ttft")), f(d, 2, True), f(num(cc, "nondecode_ms"), 1), f(num(cb, "nondecode_ms"), 1)))
idl = []
try: idl = [l.strip() for l in open(os.path.join(o, "identity.txt")) if l.strip()]
except OSError: pass
want = [l for l in idl if l.startswith("x3d-C-vs-B ") or l.startswith("cold-")]   # the C-vs-B comparisons only (not the warm-up line)
idbad = [l for l in want if "IDENTICAL" not in l or "differ" in l.lower()]
if len(want) < 2: idbad = idbad or ["%d of 2 C-vs-B comparisons present" % len(want)]
idv = "IDENTICAL (%s)" % "; ".join(l[:120] for l in want) if want and not idbad else ("VIRHE: %s" % "; ".join(l[:160] for l in (idbad or ["not compared"])))
out.append("IDENTITY C vs B: %s" % idv)
# INFO: B on this start against int10's own numbers on the same harness (other starts, other days' state: not a gate)
ng = [r for r in rows(os.path.join(rm, "ngreuse", "blocks.jsonl")) if ok(r)]
ngc = [r for r in rows(os.path.join(rm, "ngreuse", "cold.jsonl")) if r.get("status") == 200 and not r.get("cached")]
sp = [r for r in rows(os.path.join(rm, "spread", "blocks.jsonl")) if ok(r)]
sess = [x for x in (jline(os.path.join(rm, "spread", "essay-%d.txt" % i)) for i in (1, 2)) if x]
r8 = [r for r in ng if r.get("name") == "B-c8"]; r1 = [r for r in ng if r.get("name") in ("B-c1", "C-c1")]
rc = [r for r in ngc if r.get("name") == "B-cold"]; r4 = [r for r in sp if r.get("name") in ("c4-2", "c4-3")]
out.append("INFO B vs int10 (0a60fffd, same harness, other starts; not a gate): c8 %s vs ngreuse B-c8 %s = %s | c1 %s vs ngreuse B-c1 / C-c1 %s = %s | 64k TTFT %s vs ngreuse B-cold %s = %s | c4 %s vs spread c4-2 / c4-3 %s = %s (copy path) | essay %s vs spread essay-1 / -2 %s = %s (copy path)" % (
    f(num(get("B", 8), "agg_dec_tps"), 1), f(num(r8, "agg_dec_tps"), 1), f(pc(num(get("B", 8), "agg_dec_tps"), num(r8, "agg_dec_tps")), 2, True),
    f(num(b1, "agg_dec_tps"), 1), f(num(r1, "agg_dec_tps"), 1), f(pc(num(b1, "agg_dec_tps"), num(r1, "agg_dec_tps")), 2, True),
    f(num(cb, "ttft")), f(num(rc, "ttft")), f(pc(num(cb, "ttft"), num(rc, "ttft")), 2, True),
    f(num(get("B", 4), "agg_dec_tps"), 1), f(num(r4, "agg_dec_tps"), 1), f(pc(num(get("B", 4), "agg_dec_tps"), num(r4, "agg_dec_tps")), 2, True),
    f(num(eb, "decode_tps"), 2), f(num(sess, "decode_tps"), 2), f(pc(num(eb, "decode_tps"), num(sess, "decode_tps")), 2, True)))
miss = [k for k, v in dl.items() if v is None]
out.append("VERDICT x3d (C = exl3_x3d 1 vs B = 0, one start): %s | identity %s | %s%s" % (
    ", ".join("%s %s" % (k, f(v, 2, True)) for k, v in dl.items()), "IDENTICAL" if want and not idbad else "VIRHE", "no VIRHE line" if nv == 0 else "%d VIRHE line(s)" % nv,
    (" | no reading: " + ", ".join(miss)) if miss else ""))
for l in out: print(l)
print("SUMMARY " + " || ".join(out)[:5000])
PY
  while IFS= read -r l; do case "$l" in SUMMARY*) ;; *) xlog "${l:0:3000}" ;; esac; done < $O/summary.txt
  xlog "summary: $(grep -m1 '^SUMMARY ' $O/summary.txt | cut -c9- | cut -c1-5000)"
}
trap 'down; [ $NGB = 1 ] && [ $NGC = 1 ] && [ $NGCLEAN = 0 ] && ngclean' EXIT   # X3DS_NGCLEAN=1: an early exit frees /dev/shm too
trap 'exit 143' TERM INT
# ---------------------------------------------------------------- main
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19))$([ $SW = 1 ] || echo " | VIRHE X3DS_SW=0: NO exl3_x3d / mhc_fuse / ix_lds / ix_ni / draft_depth0 3 / stage_full_* / ds41_arcoded / sched_row_cap posted (a mock run only)") | X3D-F gate 3: ONE start (ngshare) on $DESC | readings: identity B; c4 warm-up; c8 B C (time window $C8WARM..$C8DUR s); c4 C B (gens $GENS4); c1 B C C B (gens $GENS1); essay B C C B ($EGEN); cold $DEPTH B C ($NTOK tok); identity C | budget $BUDGET min | no evict | n-gram tables $([ $NGC = 1 ] && echo "cleaned at the end" || echo "kept for the next item")"
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
# int9b's own (VIRHE when missing; SKIP only under DRYRUN=1 + X3DS_DRY_ALLOW_OLD=1, a dry run against an image without them)
IMGLITS9B="B|stage_full_pick, mhc_fuse, ix_lds, ix_ni, ds41_arcoded, sched_row_cap, exl3_x3d, epoch_bump)
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
# int10's own (VIRHE when missing; SKIP only under DRYRUN=1 + X3DS_DRY_ALLOW_OLD=1): lever ARC + draft_depth0 3
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
# int10x's own (VIRHE when missing; SKIP only under DRYRUN=1 + X3DS_DRY_ALLOW_OLD=1): fn4 exl3_x3d (engine) + the plugin's hooks
IMGLITSX="B|exl3_x3d
X|rad_fn4_switch
X|rad_fn4_accepts
X|rad_fn4_value
X|exl3_x3d
P|int10x-PROVENANCE.txt"
docker image inspect $IMAGE > /dev/null 2>&1 || { xlog "VIRHE image $IMAGE not on this host"; exit 1; }
IMGMISS=$( { echo "$IMGLITS"; echo "$IMGLITS9B" | sed 's/^/9/'; echo "$IMGLITS10" | sed 's/^/9/'; echo "$IMGLITSX" | sed 's/^/9/'; } | timeout 120 docker run --rm -i --network none --entrypoint sh $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels/libr4d.so; R=/opt/radiance/bin/r4d_selftest; X=/opt/radiance/share/radiance/kernels/exl3r4d.so; n=0
while IFS= read -r l; do [ -n "$l" ] || continue; n=$((n + 1)); g=""; case $l in 9*) g=9; l=${l#9} ;; esac; w=${l%%|*}; s=${l#*|}
  if [ $w = P ]; then test -s /opt/radiance/share/radiance/$s || echo "MISS$g $w|$s"; continue; fi
  case $w in B) f=$B ;; A) f=$A ;; K) f=$K ;; R) f=$R ;; X) f=$X ;; *) f="$B $A $K" ;; esac
  o=0; for x in $f; do grep -qzF -- "$s" $x && { o=1; break; }; done
  [ $o = 1 ] || echo "MISS$g $w|$s"; done
test -x /opt/radiance/bin/rad-sidecar || echo "MISS bin/rad-sidecar"; for p in int7 int8 int9; do test -s /opt/radiance/share/radiance/$p-PROVENANCE.txt || echo "MISS $p-PROVENANCE.txt"; done; echo "PLUGSHA $(sha256sum $X | cut -c1-64)"; echo "CHECKED $n"' 2>&1)
case "$IMGMISS" in *"CHECKED "*) ;; *) xlog "VIRHE the image literal check did not run on $IMAGE: $(echo "$IMGMISS" | tail -3 | tr '\n' ';' | cut -c1-500)"; exit 1 ;; esac
MISS8=$(echo "$IMGMISS" | grep '^MISS ' | tr '\n' ';'); MISS9=$(echo "$IMGMISS" | grep '^MISS9 ' | sed 's/^MISS9 //' | tr '\n' ';')
[ -z "$MISS8" ] || { xlog "VIRHE $IMAGE is not the int9 image or later -- missing: $(echo "$MISS8" | cut -c1-900)"; exit 1; }
if [ -n "$MISS9" ]; then
  if [ "$DRY" = 1 ] && [ "$ALLOWOLD" = 1 ]; then
    echo "+ image check: SKIP (image lacks int9b / int10 literals; X3DS_DRY_ALLOW_OLD=1, dry run only): missing $(echo "$MISS9" | tr ';' '\n' | grep -c .): $(echo "$MISS9" | cut -c1-700)"
  else
    xlog "VIRHE $IMAGE is not the int10x image -- missing: $(echo "$MISS9" | cut -c1-900)"; exit 1
  fi
fi
PSHA=$(echo "$IMGMISS" | grep -m1 '^PLUGSHA ' | cut -c9-)
if [ "$PSHA" != "$PLUGSHA" ]; then
  if [ "$DRY" = 1 ] && [ "$ALLOWOLD" = 1 ]; then echo "+ plugin check: SKIP (exl3r4d sha256 ${PSHA:0:16} != ${PLUGSHA:0:16}; X3DS_DRY_ALLOW_OLD=1, dry run only)"
  else xlog "VIRHE $IMAGE's exl3r4d.so sha256 ${PSHA:-none} is not the ds41-x3d 579d0bc plugin's ${PLUGSHA:0:16}"; exit 1; fi
fi
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)): all $(echo "$IMGLITS" | grep -c .) int9 / KVR / profiler literals present$([ -z "$MISS9" ] && echo " + all $(echo "$IMGLITS9B" | grep -c .) int9b literals (the registry tail now ending in ds41_arcoded, sched_row_cap, exl3_x3d) + all $(echo "$IMGLITS10" | grep -c .) int10 literals (ds41_arcoded, GET arc counters, draft_depth0 3 and its start line, libr4d's ARC env / stats / coded launch name, r4d_selftest's ar_arcoded, int10-PROVENANCE) + all $(echo "$IMGLITSX" | grep -c .) int10x literals (exl3_x3d in bin/radiance, the exl3r4d's rad_fn4_* hooks, int10x-PROVENANCE)") + exl3r4d.so sha256 ${PSHA:0:16} (= X3DS_PLUGSHA) + bin/rad-sidecar + int7 / int8 / int9 PROVENANCE -- OK"
HMISS=""
for f in "$LIB29" "$LIBPR" $Y $CTR $GS $LD $SPB $NCMB $MBP $TESTS $DQ $D/fastcheck.py $NGS/ngshare.sh $NGS/ds41-ngshare-tp4.yaml $NGS/ngram-keeper.sh $NGS/ngram-shm-cleanup.sh $NGS/README.md; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $CTR $GS $LD $SPB $NCMB $MBP $TESTS $DQ $D/fastcheck.py 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
CTT_=$(timeout 60 python3 -I -B $CTR selftest 2>&1 | tail -1); case "$CTT_" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$CTR(${CTT_:0:300})" ;; esac
ST=$(timeout 60 python3 -I $LD selftest --gs $GS --mb $MBP 2>&1 | tail -1); case "$ST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$LD(${ST:0:300})" ;; esac
grep -q '"essay"' $TESTS || HMISS="$HMISS stepprobe-prompts:$TESTS"
for f in $NGS/ngshare.sh $NGS/ngram-keeper.sh $NGS/ngram-shm-cleanup.sh; do bash -n $f 2>/dev/null || HMISS="$HMISS syntax:$f"; done
grep -q 'ds41-stack2-tp4.yaml' $NGS/ngshare.sh && grep -q '/dev/shm/ds41ng:/ngram' $NGS/ds41-ngshare-tp4.yaml || HMISS="$HMISS ngshare:not-the-README's-opt-in"
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $NGS/ds41-ngshare-tp4.yaml config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$NGS/ds41-ngshare-tp4.yaml(docker compose config refuses it)"
NGLEFT=$(ls /dev/shm/ds41ng/ngram-*.tab 2>/dev/null | wc -l)
NMT=$(timeout 60 python3 -I -B $NCMB --mb $MBP --selftest 2>&1 | tail -1); case "$NMT" in *OK*) ;; *) HMISS="$HMISS selftest:$NCMB(${NMT:0:300})" ;; esac
if [ -s $NCMB2 ] && [ -s $MBT2 ]; then   # the harness (present: it runs the c8 windows; absent at run time: ncmb.py, logged)
  M2T=$(timeout 60 python3 -I -B $MBT2 --mb $MBP --selftest 2>&1 | tail -1); case "$M2T" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$MBT2(${M2T:0:300})" ;; esac
  N2T=$(timeout 60 python3 -I -B $NCMB2 --mb $MBP --selftest 2>&1 | tail -1); case "$N2T" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$NCMB2(${N2T:0:300})" ;; esac
  CMT=$(timeout 60 python3 -I -B $C8MIX selftest 2>&1 | tail -1); case "$CMT" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$C8MIX(${CMT:0:300})" ;; esac
else M2T="absent"; N2T="absent"; CMT="absent"; fi
FNS="log run mkd restart check down stopfile best_vars desc serve_args xlog elapsed armj NAMEOF pinj stackenv keeplog capture thermline dfree diskok cmpfile idrec identw modes vramstart fn4get fn4set nudge fn4code pcreset arm msnap mdelta pskstat ssnap sdelta gate1 alone cmpx budok ctrinfo arcnow capd drafted readback pinset startinfo blk coldx swnote ablk acold coldid essayx idchk c8mixes xstart readings ngclean summary"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee sha256sum gzip zcat; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
for a in B C; do python3 -c 'import json,sys; d=json.loads(sys.argv[1]); sw = sys.argv[2] == "1"; a = sys.argv[3]
assert d["ds41_psk"] == 4 and d["ds41_psk_cr"] == 1 and d["draft_sample"] == 0 and d["draft_lend"] == 1
if sw: assert d["mhc_fuse"] == int(sys.argv[4]) and d["ix_lds"] == 0 and d["ix_ni"] == 0 and d["draft_depth0"] == int(sys.argv[5]) and d["ds41_arcoded"] == 0 and d["exl3_x3d"] == (0 if a == "B" else 1) and d["stage_full_min"] == int(sys.argv[7]) and d["sched_row_cap"] == int(sys.argv[8])
else: assert d["draft_depth0"] == 1 and not any(k in d for k in ("mhc_fuse", "ix_lds", "ix_ni", "ds41_arcoded", "exl3_x3d", "stage_full_min", "stage_full_pick", "sched_row_cap"))' "$(pinj $a)" $SW $a $MHC $D0 $ARCN $SPN $CAP 2>/dev/null || HMISS="$HMISS json:$a"; done
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
HMD5="i9b/i9bctr.py $(md5sum < $CTR | cut -c1-8) (INFO only), stack4/loadid.py $(md5sum < $LD | cut -c1-8), nc/ncmb.py $(md5sum < $NCMB | cut -c1-8), harness mbt2.py $(md5sum < $MBT2 2>/dev/null | cut -c1-8) / ncmb2.py $(md5sum < $NCMB2 2>/dev/null | cut -c1-8) / c8mix.py $(md5sum < $C8MIX 2>/dev/null | cut -c1-8), deepq.py $(md5sum < $DQ | cut -c1-8), mb.py $(md5sum < $MBP | cut -c1-8), stepprobe.py $(md5sum < $SPB | cut -c1-8), greedyset.py $(md5sum < $GS | cut -c1-8), ds41_tests.py $(md5sum < $TESTS | cut -c1-8), ngramshare: ngshare.sh $(md5sum < $NGS/ngshare.sh | cut -c1-8) / yaml $(md5sum < $NGS/ds41-ngshare-tp4.yaml | cut -c1-8) / keeper $(md5sum < $NGS/ngram-keeper.sh | cut -c1-8) / cleanup $(md5sum < $NGS/ngram-shm-cleanup.sh | cut -c1-8), dsm-x3d.sh $(md5sum < $0 | cut -c1-8) (from ngreuse/dsm-ngreuse.sh $(md5sum < $HD/../ngreuse/dsm-ngreuse.sh 2>/dev/null | cut -c1-8)), lib-29.sh $(md5sum < $LIB29 | cut -c1-8), stack2 yaml $(md5sum < $Y | cut -c1-8)"
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing; i9bctr.py $CTT_ | loadid.py $ST | ncmb.py $NMT | harness: mbt2.py $M2T / ncmb2.py $N2T / c8mix.py $CMT | ngramshare scripts parse, ngshare.sh is the README's opt-in (stack2's yaml -> ds41-ngshare-tp4.yaml, /dev/shm/ds41ng:/ngram), both yamls accepted by docker compose config | /dev/shm/ds41ng tables now: $NGLEFT (the start fills or reuses them; kept at the end unless X3DS_NGCLEAN=1) | all $(echo $FNS | wc -w) functions resolve, tools on PATH, both arms' pins (S5T, draft_sample 0, draft_lend 1$([ $SW = 1 ] && echo ", SP 0, M1 on, IX off, draft_depth0 $D0, sched_row_cap $CAP, ds41_arcoded 0, exl3_x3d 0 / 1" || echo "; X3DS_SW=0: no int9 .. int10x switch")), sidecars $SIDECAR + $SIDE sha256 OK, LEVENV with MHC_FUSE=0, without KVR / ARC / CAP -- OK | md5: $HMD5"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -iE '^radiance' | grep -vxF "${ATTACH_CLAIMED_CN:-.}" | grep -q . && { xlog "VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { xlog "VIRHE $M225 missing or not verified"; exit 1; }
  [ -s $WARM_REF_225 ] || xlog "note -- reference $WARM_REF_225 missing (not compared)"
  xlog "$ITEM: helpers md5: $HMD5"
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDE; do [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  for f in $WARM_REF_225 $RS5/s5-n5-stats0.json $RS5/guard-s5-n5.txt $RM/ngreuse/blocks.jsonl $RM/ngreuse/cold.jsonl $RM/spread/blocks.jsonl $RM/spread/essay-1.txt $RM/spread/essay-2.txt; do [ -e $f ] && echo "+ present (reference): $f" || echo "+ absent (reference, information only): $f"; done
  echo "+ arm B: $(pinj B)"
  echo "+ arm C: $(pinj C)"
  echo "+ ONE start (ngshare) | readings: identity B; c4 warm-up (B); c8 B, C (time window $C8WARM..$C8DUR s); c4 C, B (gens $GENS4); c1 B C C B (gens $GENS1); essay B C C B ($EGEN); cold $DEPTH B, C ($NTOK tok); identity C | free RAM $(free -g | awk 'NR==2{print $7}') GiB, /dev/shm used $(df -h --output=used /dev/shm | tail -1 | tr -d ' ')"
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(information here; the real run refuses to start beside one)"
  echo "+ disk now: $(dfree) MB free on / (guard $DISKMIN MB)"
fi
mkd $O $O/guard $O/blk $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt
[ "$DRY" = 1 ] || xlog "n-gram tables before the start: $NGLEFT in /dev/shm/ds41ng ($([ "$NGLEFT" -gt 0 ] && echo "the start reuses them if the probes match" || echo "the start fills them")); /dev/shm used $(df -h --output=used /dev/shm | tail -1 | tr -d ' ')"
XOK=0
if ! stopfile; then
  . $NGS/ngshare.sh || { xlog "VIRHE ngshare.sh did not load (Y $Y) -- not started"; exit 1; }
  NGB=1
  xlog "ngshare.sh sourced -- Y $(basename $Y), restart() wrapped (keeper prep before, keeper start --wait 120 --expect 2 after)"
  xstart x3d && XOK=1
fi
if [ $XOK = 1 ]; then
  readings
  capture $O/x3d-engine.log.gz
  c8mixes
  xlog "x3d: at the end: $(arcnow) | $([ "$DRY" = 1 ] || docker logs --since 6h $CN 2>&1 | grep -cE '^E ') E lines | $(elapsed) min"
fi
down
if [ $NGB = 1 ]; then
  if [ $NGC = 1 ]; then ngclean
  else xlog "ngshare: the n-gram tables stay in /dev/shm/ds41ng for the next item ($(ls /dev/shm/ds41ng/ngram-*.tab 2>/dev/null | wc -l) table(s), /dev/shm used $(df -h --output=used /dev/shm | tail -1 | tr -d ' ')); ngram-shm-cleanup.sh frees them before a copy-path speed block or produp"; fi
fi
summary
unset $SWITCHES
down
xlog "$ITEM: done ($(elapsed) min) -- $O | $([ $NV = 0 ] && echo "no VIRHE line" || echo "$NV VIRHE line(s): see above")"
