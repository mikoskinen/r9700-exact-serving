#!/bin/bash
# dsm-arc.sh (12.11.2026, queue-ar/dsmain/arc, ds-main-port ARC builder) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzzz-arc.sh. Log tag "RESULT xarc ", out runs/xdsm/arc (XARC_OUT). Made from a COPY of
# i9b/dsm-i9b.sh (64d65518: the start, LEVENV / LITS, the fixed pin and the S5T base, stackenv .. gate1, alone, cmpx, budok,
# gpurun, c8x / c8s on the c8 block-mix harness; tag xi9b -> xarc); the original is not changed. Helpers read only:
# harness/ncmb2.py + mbt2.py + c8mix.py (c8 on the fixed schedule, the co-batch mix, the newcomer step), nc/ncmb.py (its fallback
# only), stack4/loadid.py (greedy-set compare: text + token ids), i9b/i9bctr.py (counter deltas, INFO only), stepprobe.py,
# deepq.py, mb.py, greedyset.py, ds41_tests.py; d0ab/d0blocks.py (+ its s5ctr.py; the D block, d0ab's harness and args) with
# stack4/mbt.py, editprobe.py, sdprobe.py as its arguments.
# LEVER ARC on the ds41-int10 image (ds41-int9b f119c56 + libr4d fn4 ds41_arcoded appended after ix_ni (55; kSwitches 56); 0 and
# RADIANCE_DS41_ARCODED unset = the int9b image; + draft_depth0 3, depth 0 by occupancy only; + the ds41-cap48 merge, fn4
# sched_row_cap 56): the four-rank exact two-shot all-reduce (row ar_twoshot_4rank_exact, deepseek41's
# attention / MoE / embedding all-reduces, declared without overlap) sends libr4d's lossless CODED wire (~12.2 bits a bf16 value,
# until now only at overlap 1) for a launch of >= N rows, at 8 / 16 / 24 blocks by the size (the ARW item's best grid; ARW 12.11.:
# 189 byte comparisons IDENTICAL, -15..-17 % a prefill-sized call, decode sizes slower -> kept plain); below N and at every decode
# height the plain wire as before. GET /admin/fn4 "arc": coded_launches / coded_rows / plain_launches / plain_rows / below /
# declined (rank 0) and coded_all_ranks.
#   B  ds41_arcoded 0 (OFF: today's plain wire)
#   C  ds41_arcoded XARC_N (1024: coded from 1024 rows; the ARW table: -17 % at T 1100, nothing measured between 48 and 1100)
# THE BASE (the coordinator 12.11.): ONE NST 5 START = stack5's (stack4's + RADIANCE_LOG_STEPS=1 + RADIANCE_DS41_ENGDD=0 +
#   RADIANCE_PLACE_HB_RANKS=1,2) + RADIANCE_DS41_MHC_FUSE=0 (lever M1 declared, as i9b's screen start); KVR OFF
#   (RADIANCE_DS41_KV_REPLICATED unset), SP off. BASE PIN = stack5's S5T with draft_sample 0 + SP pinned 0 / 0 + M1 ON (mhc_fuse
#   XARC_MHC = 1, i9b's "M" arm: the i9b screen's -2.62 % candidate) + ix_lds 0 + ix_ni 0 + draft_depth0 XARC_D0 = 3 (int10:
#   depth 0 by occupancy only -- draft_occ's rule holds 0 at >= 6 rows for 32 cycles, below it every cycle drafts with nothing
#   weighed; the coordinator 12.11. after d0ab: draft_depth0 0 gave c4 +9.6 % but dropped the occupancy rule, c8 -19 %) + the
#   CAP48 (sched_row_cap XARC_CAP = 48: a step's decode rows held to 48 by trimming drafts; the coordinator's GO 12.11.: c16
#   +9.5 %, strict identity under load) + the fixed pin (stack4's + ds41_psk_k 8; draft_lend 1) + XARC_BASEX (extra base pins as JSON, default none). Every arm = the base +
#   its ds41_arcoded, POST + epoch_bump, a nudge, every switch READ BACK.
# P0 (ALL FOUR cards, no server, before the start; a failure stops the item): the image's r4d_selftest --case ar_arcoded (the row's
#   own entry under ds41_arcoded 0 / 1024 at T 1023 .. 4160 x 5120, three inputs, every rank == the fp32 fold, the counters, the
#   threshold edge; INFO back-to-back times), then libr4d's own --case ar_twoshot_4rank_exact and ar_twoshot_4rank_exact_lane.
# READINGS (in process, B / C alternating ABBA):
#   (0) D BLOCK under B (ARC off) on the base, d0ab's harness (d0blocks.py read only, one invocation a phase): ONE mb c4 (T 0, dur
#       XARC_DDUR 30 warm XARC_DWARM 10), then ONE mb c8 (the old loop, nc/ncmb.py, as the baseline): c4 agg vs d0ab B
#       (draft_depth0 0) 157.1, zero_cycles ~0; c8 agg vs base1113 S5T 186.7, occ_cycles > 0, acceptance 0, ~8 tok a step; the
#       CAP48 counters ("cap": steps_capped / drafts_trimmed / steps_gt48) a phase -- c8 should trim nothing (8 x 6 = 48 rows,
#       drafting off by occ). Each a NOTE when not, never a VIRHE.
#   (a) COLD prefill TTFT: deepq's fixed prompt at 135000 and 64000 tokens after a prefix-cache reset, XARC_NTOK 512 greedy tokens,
#       XARC_COLD "135000B 135000C 64000C 64000B" (one a depth an arm); the mean non-decode step ms beside; GUARD: each depth's C
#       answer == its B answer, byte for byte (cmpfile), else VIRHE.
#   (b) c8 newcomer steps: mb c8 at T 0 on the c8 block-mix harness (harness/ncmb2.py = nc/ncmb.py's taps around harness/mbt2.py's
#       FIXED, SEEDED schedule: --sched fixed --pattern 1x8 --seed 1 --window gens --gens 2 --edge 0.5 --ignore-eos 1 = 16 newcomer
#       steps a block, ~85 s), XARC_C8ORDER B C C B: per block c8mix.py (the co-batch mix + the newcomer step walls); the C-vs-B
#       delta (agg tok/s, newcomer step p50) read only when the two B mixes MATCH, else INCONCLUSIVE (mix) with the numbers.
#   (c) DECODE unchanged: ONE 1u essay at T 0 a arm (stepprobe essay XARC_EGEN 1024, XARC_EORDER C B): decode tok/s, ms a step.
#   (d) IDENTITY (~2.5 min): greedyset.py's 20 prompts at c1 alone under B then C, text + token ids strict (loadid.py cmp); then
#       the 20 at conc 4 under B vs d0ab's A (runs/xdsm/d0ab/s5-n5-c4-A.jsonl: int9, draft_depth0 1, M1 off -- every lever exact);
#       a difference = VIRHE.
#   GUARDS: switch read back (every POST) + 0 request errors; the ARC COUNTERS per reading: every C reading of (a) / (b) moves
#     coded_launches (else "path did not run": VIRHE), no B reading moves it (else VIRHE); (c) INFO; cached > 0 on a cold run =
#     not cold (left out of the TTFT line, named).
#   OUTPUT: per depth the C-vs-B TTFT delta and the chunk ms; the c8 line (agg, newcomer step p50, the B-vs-B mix check); the
#     decode line; the identity line; the counters; a summary.
# ~25 min estimate (P0 ~3.5, start + checks ~5.5, switch note ~0.5, (0) ~2.5, (a) 2 x ~45 s + 2 x ~20 s + resets = ~2.5, (b) 4 x
#   ~85 s + mix parse = ~6.5, (d) 2 x ~65 s + ~20 s = ~2.6, (c) 2 x ~15 s = ~0.5, end ~0.5). BUDGET XARC_BUDGET_MIN 28 (a section
#   that would end past it is skipped, logged; (d) runs before (c)). NO evict, no cc_sim, no power / host / fan settings; production compose files
#   read only. Exact by design; identity reported, never hidden.
# MOCK: XARC_SW=0 = a server without the int9 / int9b / int10 switches (queue-ar/tools/mockrad): the base and arms post NO
#   mhc_fuse / ix_lds / ix_ni / stage_full_* / ds41_arcoded and keep draft_depth0 1 (the mock knows 0..2), the ARC counters are INFO, P0's failure does not stop the item; the
#   start line carries a "VIRHE XARC_SW=0" marker (a mock run only), as i9b's XI9_SW=0.
# Knobs: XARC_IMAGE, XARC_ITEM, XARC_OUT, XARC_SW (1), XARC_N (1024), XARC_COLD ("135000B 135000C 64000C 64000B"), XARC_NTOK (512),
#   XARC_C8ORDER (BCCB), XARC_C8DUR (30), XARC_C8WARM (10), XARC_C8GENS (2), XARC_EORDER (CB), XARC_EGEN (1024), XARC_IDN (1),
#   XARC_MHC (1), XARC_D0 (3), XARC_CAP (48), XARC_BASEX ('' = none; JSON pins merged into the base), XARC_SPN (0) / XARC_SPK (0), XARC_D (1: the D
#   block), XARC_DDUR (30), XARC_DWARM (10), XARC_C4ID (1: the c4 identity vs d0ab A), XARC_C4REF (d0ab's s5-n5-c4-A.jsonl),
#   XARC_P0 ("ar_arcoded:29:420 ar_twoshot_4rank_exact:17:150 ar_twoshot_4rank_exact_lane:24:150" = case:min ok lines:seconds),
#   XARC_DISKMIN_MB (11000), XARC_BUDGET_MIN (28), XARC_SIDECAR, XARC_SIDESHA; XARC_DRY_ALLOW_OLD=1 (DRYRUN only).
# DRYRUN=1 bash dsm-arc.sh : prints every command; checks the image literals (int9's + int9b's + int10's), every helper (files,
#   parse, selftests), the sidecars, the yaml, the disk; no server, no GPU, writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/../ds41-stack2-tp4.yaml                   # stack2's yaml, read only (env_file ${XS2_ENVFILE})
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh
O=${XARC_OUT:-$R/xdsm/arc}; export STATSDIR=$O/stats
RD=$R/xdsm/draft; RM=$R/xdsm; RS5=$R/xdsm/stack5
export IMAGE=${XARC_IMAGE:?set XARC_IMAGE}
ITEM=${XARC_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzzz-arc}
PSKK=8; BIC=3; NOA=2; OCC=6; OCCK=32; DSA=0; HBR=0; HBRANKS=1,2      # stack5's S5T values, draft_sample 0 (i9b's base)
SW=${XARC_SW:-1}; ARCN=${XARC_N:-1024}; COLD=${XARC_COLD:-135000B 135000C 64000C 64000B}; NTOK=${XARC_NTOK:-512}
C8ORDER=${XARC_C8ORDER:-BCCB}; C8DUR=${XARC_C8DUR:-30}; C8WARM=${XARC_C8WARM:-10}; C8GENS=${XARC_C8GENS:-2}
EORDER=${XARC_EORDER-CB}; EGEN=${XARC_EGEN:-1024}; DOID=${XARC_IDN:-1}
MHC=${XARC_MHC:-1}; D0=${XARC_D0:-3}; CAP=${XARC_CAP:-48}; BASEX=${XARC_BASEX:-}; SPN=${XARC_SPN:-0}; SPK=${XARC_SPK:-0}
DOD=${XARC_D:-1}; DDUR=${XARC_DDUR:-30}; DWARM=${XARC_DWARM:-10}; DOC4=${XARC_C4ID:-1}
P0=${XARC_P0-ar_arcoded:29:420 ar_twoshot_4rank_exact:17:150 ar_twoshot_4rank_exact_lane:24:150}
DISKMIN=${XARC_DISKMIN_MB:-11000}; BUDGET=${XARC_BUDGET_MIN:-28}; ALLOWOLD=${XARC_DRY_ALLOW_OLD:-0}
SIDE=${XARC_SIDECAR:-dspark-dlw-q8_0.rad}; SIDESHA=${XARC_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
MT=192; SEED=1234
case $SW in 0|1) ;; *) echo "XARC_SW is 1 (the item) or 0 (a server without the int9 / int9b / int10 switches: the mock) (got '$SW')"; exit 1 ;; esac
for v in $ARCN $NTOK $C8DUR $C8WARM $C8GENS $EGEN $DOID $MHC $D0 $CAP $DOD $DDUR $DWARM $DOC4 $SPN $SPK $DISKMIN $BUDGET; do case $v in ''|*[!0-9]*) echo "a numeric knob is not a number ('$v')"; exit 1 ;; esac; done
[ $ARCN -ge 1 ] && [ $ARCN -le 1048576 ] || { echo "XARC_N 1..1048576 (ds41_arcoded's on values)"; exit 1; }
for c in $COLD; do case $c in [0-9]*[BC]) case ${c%?} in *[!0-9]*) echo "XARC_COLD takes DEPTH followed by B or C (got '$c')"; exit 1 ;; esac ;; *) echo "XARC_COLD takes DEPTH followed by B or C (got '$c')"; exit 1 ;; esac; done
case $C8ORDER in *[!BC]*) echo "XARC_C8ORDER takes B and C (got '$C8ORDER')"; exit 1 ;; esac
case $C8ORDER in B*B|"") ;; *) echo "XARC_C8ORDER must start and end with B (the B-vs-B mix check; got '$C8ORDER')"; exit 1 ;; esac
case $EORDER in *[!BC]*) echo "XARC_EORDER takes B and C (got '$EORDER')"; exit 1 ;; esac
[ $C8WARM -lt $C8DUR ] || { echo "the c8 warm-up is not shorter than the window"; exit 1; }
[ $MHC -le 1048576 ] || { echo "XARC_MHC 0..1048576 (mhc_fuse's values)"; exit 1; }
[ $SPN -le 1048576 ] && [ $SPK -le 64 ] || { echo "XARC_SPN 0..1048576, XARC_SPK 0..64"; exit 1; }
[ $EGEN -ge 64 ] || { echo "XARC_EGEN >= 64"; exit 1; }
[ $D0 -le 3 ] || { echo "XARC_D0 0..3 (draft_depth0's values; 3 = occupancy only, int10)"; exit 1; }
[ $CAP -le 4096 ] || { echo "XARC_CAP 0..4096 (sched_row_cap's values; 48 = CAP48)"; exit 1; }
[ $DWARM -lt $DDUR ] || { echo "the D block's warm-up is not shorter than its phase"; exit 1; }
for c in $P0; do case $c in *:*[!0-9:]*|*::*|*:*:*:*) echo "XARC_P0 takes case:min_ok:seconds (got '$c')"; exit 1 ;; esac; done
if [ -n "$BASEX" ]; then python3 -c 'import json,sys; d=json.loads(sys.argv[1]); assert isinstance(d, dict) and "ds41_arcoded" not in d' "$BASEX" 2>/dev/null \
  || { echo "XARC_BASEX must be a JSON object of extra base pins without ds41_arcoded (got '$BASEX')"; exit 1; }; fi
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
S4=$HD/../stack4
LD=$S4/loadid.py                                                                                    # stack4's, read only
CTR=$HD/../i9b/i9bctr.py                                                                            # i9b's (INFO only), read only
NCMB=$HD/../nc/ncmb.py                                                                              # nc's, read only (the fallback only)
MBT2=$HD/../harness/mbt2.py; NCMB2=$HD/../harness/ncmb2.py; C8MIX=$HD/../harness/c8mix.py          # the harness (read only): c8 on mbt2's fixed schedule
DQ=$QA/deepq.py; MBP=$D/mb.py; TESTS=$D/ds41_tests.py; SPB=$D/stepprobe.py
MBT=$S4/mbt.py; EP=$S4/editprobe.py; PRB=$S4/sdprobe.py                                            # stack4's, read only (the D block's d0blocks.py args)
D0B=$HD/../d0ab/d0blocks.py                                                                         # d0ab's (+ its s5ctr.py beside it), read only
C4REF=${XARC_C4REF:-$R/xdsm/d0ab/s5-n5-c4-A.jsonl}                                                 # d0ab's greedy set at conc 4 under A (int9, 12.11. 12:23)
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
T00=$(date +%s); NV=0
xlog() { case "$*" in *VIRHE*) NV=$((NV + 1)) ;; esac; log "RESULT xarc $*"; }
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
SPJ=", \"stage_full_min\": $SPN, \"stage_full_pick\": $SPK"; [ $SW = 1 ] || SPJ=""        # XARC_SW=0: the mock has no int9 switch
TJ="{$S5J, \"ds41_psk\": 4, \"ds41_psk_cr\": 1, \"stage_short\": 0, \"stage_pick\": 0, \"place_hb\": 0$SPJ}"   # = stack5's armj T, draft_sample 0, + SP
MJ="{\"mhc_fuse\": $MHC, \"ix_lds\": 0, \"ix_ni\": 0, \"draft_depth0\": $D0, \"sched_row_cap\": $CAP}"; [ $SW = 1 ] || MJ='{}'   # i9b's "M" arm (M1 on, IX-LDSR2 / IX-NI off) + draft_depth0 3 + CAP48 (int10)
armj() {   # armj B|C: the arm's ds41_arcoded (JSON) on top of the base; {} under XARC_SW=0 (the mock)
  [ $SW = 1 ] || { echo '{}'; return 0; }
  case $1 in
    B) echo '{"ds41_arcoded": 0}' ;;
    C) echo "{\"ds41_arcoded\": $ARCN}" ;;
  esac
}
NAMEOF() { case $1 in B) echo "B (plain wire)" ;; C) echo "C (coded wire >= $ARCN rows)" ;; esac; }
pinj() {   # pinj ARM [EXTRAJSON]: the fixed pin + the base (S5T, draft_sample 0, SP, M1 on, IX off, XARC_BASEX) + the arm + extra as one POST body
  local x=${2:-} bx=$BASEX; [ -n "$x" ] || x='{}'; [ -n "$bx" ] || bx='{}'
  python3 -c 'import json,sys
d = {}
for a in sys.argv[1:]: d.update(json.loads(a))
print(json.dumps(d))' "$FIX" "$TJ" "$MJ" "$bx" "$(armj $1)" "$x"
}
DESC="NST 5 stack5 start (stack4's + RADIANCE_LOG_STEPS=1, E1 declared off, HB ranks $HBRANKS) + RADIANCE_DS41_MHC_FUSE=0 (M1 declared) | KVR OFF (RADIANCE_DS41_KV_REPLICATED unset), SP off | base pin = stack5's S5T with draft_sample 0 + SP pinned (stage_full_min $SPN, stage_full_pick $SPK) + M1 on / IX off / draft_depth0 $D0 (int10: depth 0 by occupancy only) / sched_row_cap $CAP (CAP48) $MJ + the fixed pin (ds41_psk_k $PSKK; draft_lend 1)$([ -n "$BASEX" ] && echo " + XARC_BASEX $BASEX") | arms B $(armj B), C $(armj C)"
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xarc $1 warm-up identity vs $2"; return 0; }
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
  [ "$DRY" = 1 ] && { echo "+ GET /stats > $O/$1-stats0.json; RESULT xarc $1 VRAM at start (experts per tier; vs stack4 s4-n5)"; return 0; }
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xarc $tag $3: greedy set c1 alone under $(NAMEOF $2) $(armj $2): 20 of 20 answered (else a request-error line); counters INFO"; return 0; }
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
arcnow() {   # arcnow: GET /admin/fn4's ds41_arcoded + the int9b switches + draft_depth0 and the "arc" counters, one line
  [ "$DRY" = 1 ] && { echo "(dry)"; return 0; }
  fn4get | python3 -c 'import json,sys
d = json.load(sys.stdin); w = d.get("switches") or {}
print("ds41_arcoded %s sched_row_cap %s mhc_fuse %s ix_lds %s ix_ni %s draft_depth0 %s | arc %s | cap %s" % (w.get("ds41_arcoded"), w.get("sched_row_cap"), w.get("mhc_fuse"), w.get("ix_lds"), w.get("ix_ni"), w.get("draft_depth0"), json.dumps(d.get("arc")), json.dumps(d.get("cap"))))' 2>/dev/null | cut -c1-600
}
arcd() {   # arcd A B -> JSON: the "arc" counter deltas between GET /admin/fn4 snapshots A and B (present: whether B has "arc")
  [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }
  python3 -c 'import json,sys
def rd(p):
    try: return json.load(open(p)).get("arc")
    except Exception: return None
a, b = rd(sys.argv[1]) or {}, rd(sys.argv[2])
o = {"present": b is not None}
for k in ("coded_launches", "coded_rows", "plain_launches", "plain_rows", "below", "declined", "coded_all_ranks"):
    x, y = a.get(k, 0), (b or {}).get(k, 0)
    o[k] = (y - x) if isinstance(x, (int, float)) and isinstance(y, (int, float)) else None
print(json.dumps(o))' $O/fn-$1.json $O/fn-$2.json 2>/dev/null || echo '{}'
}
arcg() {   # arcg ARM DELTAJSON WHAT: the ARC counter guard of one reading -> "" (ok) or a VIRHE text; XARC_SW=0: INFO only
  [ "$DRY" = 1 ] && return 0
  python3 -c 'import json,sys
d = json.loads(sys.argv[2] or "{}"); a = sys.argv[1]; sw = sys.argv[4] == "1"
c, ca = d.get("coded_launches"), d.get("coded_all_ranks")
if not sw: print("INFO (XARC_SW=0) arc %s" % json.dumps(d)); sys.exit(0)
if not d.get("present"): print("VIRHE GET /admin/fn4 has no \"arc\" counters (not the int10 image?)"); sys.exit(0)
if a == "C" and not (c or 0) > 0: print("VIRHE path did not run: no coded launch in a C reading (%s)" % sys.argv[3]); sys.exit(0)
if a == "C" and ca != 4 * (c or 0): print("VIRHE coded launches over the ranks %s != 4 x rank 0 %s" % (ca, c)); sys.exit(0)
if a == "B" and ((c or 0) != 0 or (ca or 0) != 0): print("VIRHE %s coded launch(es) in a B (OFF) reading" % c); sys.exit(0)' "$1" "$2" "$3" $SW 2>&1 | head -1
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
    if [ "$DRY" = 1 ]; then echo "+ RESULT xarc P0 $cs on 4 cards: rc 0, 0 FAIL, 0 skip, >= $nmin ok lines (bounded at $tmo s), else a failure line and the item stops (no server start); info lines logged"; continue; fi
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
swnote() {   # swnote TAG: start values; refusals; every arm read back (base: M1 on, IX off, draft_depth0 3, SP 0)
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 at start (gg_gemv 2, aql_acq 2, K1 three 1, epoch 0, draft_capable 1, draft_sample 0; ds41_psk 0, sched_bi_chunk / sched_noa / ds41_psk_l20 / dspark_pfctx / ds41_engdd / ds41_psk_cr / stage_short / stage_pick / place_hb 0$([ $SW = 1 ] && echo "; stage_full_min / stage_full_pick / mhc_fuse / ix_lds / ix_ni / ds41_arcoded / sched_row_cap 0; \"arc\" present, coded 0 (plain may count the gate check's prefill)" || echo "; int9 / int9b / int10 NOT CHECKED (XARC_SW=0)")); refusals stage_short 65537$([ $SW = 1 ] && echo " / ds41_arcoded -1 / ds41_arcoded 1048577 / draft_depth0 4 / sched_row_cap 4097") -> 400; arms B C B each applied and read back ($(pinj B | cut -c1-200)...)"; return 0; }
  local s0 g c1=400 c2=400 c3=400 c5=400 c4 a sets="" bad=""
  s0=$(fn4get)
  c4=$(fn4code '{"stage_short": 65537}')
  [ $SW = 1 ] && { c1=$(fn4code '{"ds41_arcoded": -1}'); c2=$(fn4code '{"ds41_arcoded": 1048577}'); c3=$(fn4code '{"draft_depth0": 4}'); c5=$(fn4code '{"sched_row_cap": 4097}'); }
  for a in B C B; do
    arm $a || bad="$bad $a"; g=$(fn4get)
    sets="$sets $a:$(echo "$g" | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print("arc=%s,cap=%s,mhc=%s,lds=%s,ni=%s,d0=%s,lend=%s,occ=%s,ds=%s,psk=%s/%s,cr=%s,bi=%s,sfm=%s,sfp=%s,ep=%s" % (w.get("ds41_arcoded"), w.get("sched_row_cap"), w.get("mhc_fuse"), w.get("ix_lds"), w.get("ix_ni"), w.get("draft_depth0"), w.get("draft_lend"), w.get("draft_occ"), w.get("draft_sample"), w.get("ds41_psk"), w.get("ds41_psk_k"), w.get("ds41_psk_cr"), w.get("ds41_bi"), w.get("stage_full_min"), w.get("stage_full_pick"), d.get("epoch")))' 2>/dev/null)"
  done
  xlog "$1 switch note: at start $(echo "$s0" | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print("switches %s | arc %s | epoch %s" % (json.dumps({k: w.get(k) for k in ("ds41_arcoded", "sched_row_cap", "mhc_fuse", "ix_lds", "ix_ni", "draft_depth0", "stage_full_min", "stage_full_pick", "draft_sample", "ds41_psk", "ds41_bi", "gg_gemv", "aql_acq")}), json.dumps(d.get("arc")), d.get("epoch")))' 2>/dev/null | cut -c1-1200) | stage_short 65537 -> HTTP $c4$([ $SW = 1 ] && echo ", ds41_arcoded -1 -> HTTP $c1, ds41_arcoded 1048577 -> HTTP $c2, draft_depth0 4 -> HTTP $c3, sched_row_cap 4097 -> HTTP $c5" || echo " | int9 / int9b / int10 NOT CHECKED (XARC_SW=0)") | arms applied (read back):$sets$([ -n "$bad" ] && echo " -- VIRHE not applied:$bad")"
  python3 -c 'import json,sys
s0 = json.loads(sys.argv[1]); sw = sys.argv[2] == "1"; w = s0["switches"]; ac = s0.get("arc")
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) and s0["epoch"] == int(__import__("os").environ.get("ATTACH_EPOCH0") or 0)
ok = ok and w["draft_adaptive"] == 1 and s0["draft_capable"] == 1 and w.get("ds41_psk") == 0 and w.get("draft_sample") == 0
ok = ok and all(w.get(k) == 0 for k in ("sched_bi_chunk", "sched_noa", "ds41_psk_l20", "dspark_pfctx", "ds41_engdd", "ds41_psk_cr", "stage_short", "stage_pick", "place_hb"))
if sw:
    ok = ok and all(w.get(k) == 0 for k in ("stage_full_min", "stage_full_pick", "mhc_fuse", "ix_lds", "ix_ni", "ds41_arcoded", "sched_row_cap"))
    ok = ok and isinstance(ac, dict) and ac.get("coded_launches") == 0 and ac.get("coded_all_ranks") == 0
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
dblock() {   # (0) under B (ARC off) on the base (draft_depth0 3, M1 on, sched_row_cap 48): ONE mb c4 block, then ONE mb c8 block (old
          # loop), d0ab's d0blocks.py read only (the d0ab / baseline harness and args), one invocation a phase so each has its own
          # counter window -> c4 agg + zero_cycles; c8 agg + occ_cycles / acceptance / tok a step + the CAP48 counters
  local ph tag
  for ph in c4 c8; do
    tag=arc-d$ph
    ssnap $tag-0
    run "mkdir -p $O/c8 $O/snap; timeout 1800 python3 -I $D0B --url $URL --mb-dir $D --edit $EP --sdprobe $PRB --seed $SEED --t1-prompts essay --label $tag --arm 'B=$(armj B)' --guard 'B=/sa,hb' --hb $HBR --hb-ranks $HBRANKS --snap-dir $O/snap --settle 5 --order B --phases $ph --dur $DDUR --warm $DWARM --mbtemp 0 --mbt $MBT --c8-ncmb $NCMB --c8-dir $O/c8 --fixed '$(pinj B)' --out $O/$tag-blocks.jsonl > $O/$tag-blocks.txt 2>&1"
    ssnap $tag-1
    [ "$DRY" = 1 ] && continue
    while IFS= read -r l; do case "$l" in "GUARD "*"FAIL"*) xlog "D block $ph VIRHE ${l:0:1500}" ;; *) xlog "D block $ph ${l:0:1700}" ;; esac; done < <(grep -E '^(BLOCK|GUARD|OCC|STATS|GGB \{"error)|Traceback' $O/$tag-blocks.txt)
  done
  [ "$DRY" = 1 ] && { echo "+ RESULT xarc D block (B, the base: draft_depth0 $D0, M1 on, sched_row_cap $CAP): mb c4 then mb c8 (old loop), dur $DDUR warm $DWARM, T 0, a counter window each: c4 agg (d0ab B 157.1), zero_cycles ~0; c8 agg (base1113 S5T 186.7), occ_cycles > 0, acceptance 0, ~8 tok/step; CAP48 c8: nothing trimmed expected (8 x 6 = 48 rows, drafting off by occ) -- NOTEs, never a VIRHE"; return 0; }
  xlog "D block (B: ARC off, draft_depth0 $D0, M1 on, sched_row_cap $CAP): $(python3 -c 'import json,sys
def last(p):
    try: return [json.loads(l) for l in open(p) if l.strip()][-1]
    except Exception: return {}
r4, r8 = last(sys.argv[1]), last(sys.argv[2])
c4 = (r4.get("res") or {}).get("c4") or {}; d4 = ((r4.get("res") or {}).get("draftph") or {}).get("c4") or {}
c8 = (r8.get("res") or {}).get("c8") or {}; d8 = ((r8.get("res") or {}).get("draftph") or {}).get("c8") or {}
occ, acc, tps = d8.get("occ_cycles"), c8.get("acceptance"), c8.get("tok_per_decode_step")
c8off = (occ or 0) > 0 and not acc and tps is not None and abs(tps - 8) <= 0.5
z4 = d4.get("zero_cycles")
print("c4 %s tok/s (d0ab B draft_depth0 0: 157.1 -> %s), acceptance %s, tok/step %s, zero_cycles %s, occ_cycles %s, probe_cycles %s -- %s (block ok=%s) | c8 %s tok/s (base1113 S5T 186.7 -> %s), occ_cycles %s, zero_cycles %s, probe_cycles %s, acceptance %s, tok/step %s -- %s (block ok=%s)" % (
    c4.get("agg_dec_tps"), ("%+.1f %%" % (100 * (c4["agg_dec_tps"] / 157.1 - 1))) if c4.get("agg_dec_tps") else "-", c4.get("acceptance"), c4.get("tok_per_decode_step"), z4, d4.get("occ_cycles"), d4.get("probe_cycles"),
    "c4 drafts (zero_cycles ~0)" if z4 is not None and z4 <= 5 else "NOTE: c4 zero_cycles %s (not ~0)" % z4, r4.get("ok"),
    c8.get("agg_dec_tps"), ("%+.1f %%" % (100 * (c8["agg_dec_tps"] / 186.7 - 1))) if c8.get("agg_dec_tps") else "-", occ, d8.get("zero_cycles"), d8.get("probe_cycles"), acc, tps,
    "c8 holds depth 0 under the occ rule" if c8off else "NOTE: c8 does not look like the occ rule (occ_cycles 0, or drafting on)", r8.get("ok")))' $O/arc-dc4-blocks.jsonl $O/arc-dc8-blocks.jsonl 2>&1 | tail -1 | cut -c1-1000)"
  local c4c c8c
  c4c=$(capd arc-dc4-0 arc-dc4-1); c8c=$(capd arc-dc8-0 arc-dc8-1)
  xlog "D block CAP48 counters: c4 window: $c4c | c8 window: $c8c$(echo "$c8c" | grep -qE 'drafts_trimmed \+[1-9]' && echo " -- NOTE: CAP48 trimmed drafts at c8 (expected none: 8 x 6 = 48 rows, drafting off by occ)")"
}
coldx() {   # coldx K DEPTH ARM: deepq's fixed prompt at DEPTH after a reset, NTOK greedy tokens -> TTFT, cached, chunk ms; the ARC guard
  local k=$1 dep=$2 a=$3 nm=cold-$1-$2$3 t0=$(date +%s) j md ad g
  arm $a || return 1
  pcreset $nm
  msnap $nm-0; ssnap $nm-0
  run "(cd $D && timeout 1200 python3 $DQ $O/$nm.txt $dep $NTOK > $O/$nm.out 2>&1)"
  msnap $nm-1; ssnap $nm-1
  [ "$DRY" = 1 ] && { echo "+ RESULT xarc cold $nm: $(NAMEOF $a) $(armj $a), deepq $dep cold, $NTOK tok: TTFT, cached, chunk ms (status 200 or a request-error line); the ARC guard (C: coded launches > 0 on every rank; B: none) -> $O/cold.jsonl"; return 0; }
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
colds() {   # (a) the cold runs in XARC_COLD order; per depth the C answer vs the B answer, byte for byte (a difference = VIRHE)
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xarc c8 $nm ($(NAMEOF $a) $(armj $a), mb c8 T 0 via $([ -s $NCMB2 ] && echo "harness/ncmb2.py: fixed schedule 1x8 seed 1, gens $C8GENS, edge 0.5, ignore_eos" || echo "nc/ncmb.py: the old loop")): MB agg_dec_tps / decode_share / per-stream p50 / the client mix, errors 0 (else a request-error line); the ARC guard -> $O/c8.jsonl"; return 0; }
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
essayx() {   # (c) ONE 1u essay at T 0 under ARM (stepprobe essay EGEN): decode tok/s, ms a step, tok a step; the ARC counters INFO
  local a=$1 nm=essay-$1 j ad
  arm $a || return 1
  ssnap $nm-0
  run "(cd $D && timeout 900 python3 $SPB essay $EGEN > $O/$nm.txt 2>&1)"
  ssnap $nm-1
  [ "$DRY" = 1 ] && { echo "+ RESULT xarc essay $a ($(NAMEOF $a)): stepprobe essay $EGEN at T 0: decode tok/s, ms / tok a decode step, acceptance (a J line or a request-error line); arc counters INFO -> $O/essay.jsonl"; return 0; }
  j=$(grep -m1 '^J ' $O/$nm.txt | cut -c3-)
  [ -n "$j" ] || { xlog "essay $a: VIRHE request error: no J line ($(tail -3 $O/$nm.txt | tr '\n' ' ' | cut -c1-300))"; return 1; }
  ad=$(arcd $nm-0 $nm-1)
  python3 -c 'import json,sys; d = json.loads(sys.argv[1]); d.update(arm=sys.argv[2], arc=json.loads(sys.argv[3] or "{}")); print(json.dumps(d))' "$j" "$a" "$ad" >> $O/essay.jsonl
  xlog "essay $a ($(NAMEOF $a) $(armj $a), 1u, stepprobe essay $EGEN, T 0): $(echo "$j" | cut -c1-400) | INFO arc $ad"
}
idcheck() {   # (d) exactness: the greedy set alone under B then C (strict: VIRHE on a difference); the greedy set at conc 4 under B vs d0ab's A
  local l
  budok "identity (d)" 3 || return 0
  alone arc B a-B
  alone arc C a-C
  cmpx arc C-vs-B $O/arc-a-B.jsonl $O/arc-a-C.jsonl
  if [ "$DRY" != 1 ]; then
    l=$(grep "^arc-C-vs-B " $O/identity.txt 2>/dev/null | tail -1)
    case "$l" in *IDENTICAL*) case "$l" in *differ*|*DIFFER*) xlog "identity greedy set C vs B: VIRHE $l" ;; esac ;;
      *) xlog "identity greedy set C vs B: VIRHE ${l:-not compared} (the coded wire carries the plain wire's bytes: the text must match)" ;; esac
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
summary() {   # the readings, one line each: cold TTFT a depth, c8, decode, identity, counters; a SUMMARY line
  [ "$DRY" = 1 ] && { echo "+ RESULT xarc lines: COLD a depth (C vs B TTFT %, chunk ms, the guard, identity), C8 (agg tok/s + newcomer step p50, C vs B %, read only when the B-vs-B mixes MATCH), DECODE (essay tok/s C vs B), IDENTITY, COUNTERS, SUMMARY -- numbers, no verdict words beyond the guards"; return 0; }
  python3 - "$O" "$ARCN" > $O/summary.txt 2>&1 <<'PY'
import json, os, statistics, sys
o, n = sys.argv[1:3]
def rows(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except OSError: return []
def mean(v):
    v = [x for x in v if isinstance(x, (int, float))]
    return statistics.mean(v) if v else None
def pc(a, b): return 100.0 * (a / b - 1) if a is not None and b else None
def f(x, k=2, pct=False): return "-" if x is None else ("%+.*f %%" % (k, x) if pct else "%.*f" % (k, x))
out, sm = [], []
cold = [r for r in rows(os.path.join(o, "cold.jsonl")) if r.get("status") == 200]
for dep in sorted({r["depth"] for r in cold}, reverse=True):
    rs = [r for r in cold if r["depth"] == dep]
    fresh = [r for r in rs if not r.get("cached")]
    b = [r for r in fresh if r["arm"] == "B"]; c = [r for r in fresh if r["arm"] == "C"]
    tb, tc = mean([r.get("ttft") for r in b]), mean([r.get("ttft") for r in c])
    mb, mc = mean([r.get("nondecode_ms") for r in b]), mean([r.get("nondecode_ms") for r in c])
    cl = sum((r.get("arc") or {}).get("coded_launches") or 0 for r in c); cr = sum((r.get("arc") or {}).get("coded_rows") or 0 for r in c)
    bl = sum((r.get("arc") or {}).get("coded_launches") or 0 for r in b)
    gd = "; ".join("%s %s" % (r["name"], r["guard"]) for r in rs if r.get("guard")) or "OK"
    est = " (the coordinator's estimate at 136k: -3..-4 %)" if dep >= 100000 else ""
    l = "COLD %d: TTFT C %s s vs B %s s = %s%s | non-decode step C %s ms vs B %s ms = %s | C readings: %d coded launches (rank 0), %d rows; B readings: %d coded | guard: %s%s" % (
        dep, f(tc), f(tb), f(pc(tc, tb), 2, True), est, f(mc, 1), f(mb, 1), f(pc(mc, mb), 2, True), cl, cr, bl, gd,
        "" if len(fresh) == len(rs) else " | %d run(s) not cold (cached > 0) left out" % (len(rs) - len(fresh)))
    out.append(l); sm.append("cold %dk %s" % (dep // 1000, f(pc(tc, tb), 2, True)))
c8 = [r for r in rows(os.path.join(o, "c8.jsonl")) if r.get("errors") == 0 and (r.get("requests") or 0) > 0]
if c8:
    def mixj(nm):
        try: return json.load(open(os.path.join(o, "c8", "%s-mix.json" % nm)))
        except Exception: return None
    def nstep(nm):
        m = mixj(nm); return (m.get("single") or {}).get("wall_p50") if m else None
    br = [r for r in c8 if r["arm"] == "B"]; cr_ = [r for r in c8 if r["arm"] == "C"]
    try: bb = [l.strip()[9:] for l in open(os.path.join(o, "c8", "bb-compare.out")) if l.startswith("C8MIXCMP ")][-1:]
    except OSError: bb = []
    match = bool(bb) and bb[0].rstrip().endswith("MATCH")
    ab, ac = mean([r.get("agg_dec_tps") for r in br]), mean([r.get("agg_dec_tps") for r in cr_])
    wb, wc = mean([nstep(r["name"]) for r in br]), mean([nstep(r["name"]) for r in cr_])
    cl = sum((r.get("arc") or {}).get("coded_launches") or 0 for r in cr_); pl = sum((r.get("arc") or {}).get("plain_launches") or 0 for r in cr_)
    gd = "; ".join("%s %s" % (r["name"], r["guard"]) for r in c8 if r.get("guard")) or "OK"
    blocks = ", ".join("%s %s / %s ms" % (r["name"], r.get("agg_dec_tps"), nstep(r["name"])) for r in c8)
    l = "C8 (fixed schedule, 16 newcomer steps a block; %s): agg C %s vs B %s tok/s = %s | newcomer step wall p50 C %s vs B %s ms = %s | B-vs-B mix check: %s | C blocks: %d coded / %d plain two-shot launches (rank 0) | blocks %s | guard: %s" % (
        "read" if match else "INCONCLUSIVE (mix): the two B blocks' mixes do not MATCH", f(ac, 1), f(ab, 1), f(pc(ac, ab), 2, True), f(wc, 1), f(wb, 1), f(pc(wc, wb), 2, True),
        (bb[0][:200] if bb else "none"), cl, pl, blocks, gd)
    out.append(l); sm.append("c8 agg %s, newcomer step %s%s" % (f(pc(ac, ab), 2, True), f(pc(wc, wb), 2, True), "" if match else " (mix INCONCLUSIVE)"))
es = rows(os.path.join(o, "essay.jsonl"))
if es:
    eb = [r for r in es if r["arm"] == "B"]; ec = [r for r in es if r["arm"] == "C"]
    tb, tc = mean([r.get("decode_tps") for r in eb]), mean([r.get("decode_tps") for r in ec])
    out.append("DECODE 1u essay T 0: C %s vs B %s tok/s = %s | ms a decode step C %s vs B %s | tok a step C %s vs B %s | coded launches C %s, B %s (decode stays plain)" % (
        f(tc), f(tb), f(pc(tc, tb), 2, True), f(mean([r.get("ms_per_decode_step") for r in ec])), f(mean([r.get("ms_per_decode_step") for r in eb])),
        f(mean([r.get("tokens_per_step") for r in ec])), f(mean([r.get("tokens_per_step") for r in eb])),
        sum((r.get("arc") or {}).get("coded_launches") or 0 for r in ec), sum((r.get("arc") or {}).get("coded_launches") or 0 for r in eb)))
    sm.append("essay %s" % f(pc(tc, tb), 2, True))
try: ident = [l.strip() for l in open(os.path.join(o, "identity.txt")) if l.strip()]
except OSError: ident = []
bad = [l for l in ident if not ("IDENTICAL" in l and "differ" not in l and "VIRHE" not in l)]
il = ("IDENTICAL (%s)" % "; ".join(ident))[:900] if ident and not bad else ("NOT IDENTICAL: " + "; ".join(bad))[:900] if ident else "not run"
out.append("IDENTITY (ON vs OFF: cold answers, greedy set c1; c4 greedy set vs d0ab A): %s" % il)
for l in out: print(l)
print("SUMMARY ARC ds41_arcoded %s vs 0: %s || identity: %s || %s" % (n, "; ".join(sm) or "-", "IDENTICAL" if ident and not bad else ("NOT IDENTICAL" if ident else "not run"), " || ".join(out)[:4000]))
PY
  while IFS= read -r l; do case "$l" in SUMMARY*) ;; *) xlog "${l:0:3000}" ;; esac; done < $O/summary.txt
  xlog "summary: $(grep -m1 '^SUMMARY ' $O/summary.txt | cut -c9- | cut -c1-6000)"
}
dstart() {   # P0, then the screen start
  local tag=arc-n5
  diskok $tag || return 1
  if ! p0sel; then
    if [ $SW = 1 ]; then xlog "P0: VIRHE a four-card selftest failed -> the item stops (no server start)"; return 1; fi
    xlog "P0: failed under XARC_SW=0 (a mock run: docker run --device is not emulated) -- the mock goes on"
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
  stopfile || { [ $DOD = 1 ] && budok "D block (c4 + c8 under B)" 3 && dblock; }
  stopfile || colds
  stopfile || c8s
  stopfile || { [ $DOID = 1 ] && idcheck; }
  stopfile || { [ -n "$EORDER" ] && budok "essay (c)" 1 && for a in $(echo $EORDER | grep -o .); do essayx $a; done; }
  capture $O/$tag-engine-full.log.gz
  xlog "$tag: at the end: $(pskstat) | $(arcnow) | $(docker logs --since 6h $CN 2>&1 | grep -cE '^E ') E lines | $(elapsed) min"
  down
  xlog "$tag: done"
}
trap 'down' EXIT
trap 'exit 143' TERM INT
# ---------------------------------------------------------------- main
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19))$([ $SW = 1 ] || echo " | VIRHE XARC_SW=0: NO mhc_fuse / ix_lds / ix_ni / draft_depth0 3 / stage_full_* / ds41_arcoded posted, the ARC counters INFO, P0 not fatal (a mock run only)") | P0 r4d_selftest [$P0] on 4 cards | SCREEN start: $DESC | $([ $DOD = 1 ] && echo "D block c4 + c8 (old loop) under B, dur $DDUR warm $DWARM | ")cold [$COLD] ($NTOK tok) | c8 $C8ORDER (fixed schedule, dur $C8DUR warm $C8WARM gens $C8GENS) | identity greedy set c1 C vs B $([ $DOID = 1 ] && echo on || echo off)$([ $DOC4 = 1 ] && echo " + c4 B vs d0ab A") | essay T 0 [$EORDER] ($EGEN tok) | guards: read back + 0 request errors + the ARC counters + byte identity | budget $BUDGET min | no evict"
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
# int9b's own (VIRHE when missing; SKIP only under DRYRUN=1 + XARC_DRY_ALLOW_OLD=1, a dry run against an image without them)
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
# int10's own (VIRHE when missing; SKIP only under DRYRUN=1 + XARC_DRY_ALLOW_OLD=1): lever ARC + draft_depth0 3
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
    echo "+ image check: SKIP (image lacks int9b / int10 literals; XARC_DRY_ALLOW_OLD=1, dry run only): missing $(echo "$MISS9" | tr ';' '\n' | grep -c .): $(echo "$MISS9" | cut -c1-700)"
  else
    xlog "VIRHE $IMAGE is not the int10 image -- missing: $(echo "$MISS9" | cut -c1-900)"; exit 1
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
FNS="log run mkd restart check down stopfile best_vars desc serve_args xlog elapsed armj NAMEOF pinj stackenv keeplog capture thermline dfree diskok cmpfile idrec identw modes vramstart fn4get fn4set nudge fn4code pcreset arm msnap mdelta pskstat ssnap sdelta gate1 alone cmpx budok ctrinfo arcnow arcd arcg gpurun p0sel drafted readback swnote capd dblock coldx colds c8x c8s essayx idcheck c4rr summary dstart"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee sha256sum gzip zcat; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
for a in B C; do python3 -c 'import json,sys; d=json.loads(sys.argv[1]); sw = sys.argv[2] == "1"; a = sys.argv[3]
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
if [ $DOC4 = 1 ] && [ ! -s $C4REF ]; then xlog "note -- the c4 identity reference $C4REF is missing: the c4 greedy set vs d0ab A is not run"; DOC4=0; fi
HMD5="i9b/i9bctr.py $(md5sum < $CTR | cut -c1-8) (INFO only), d0ab/d0blocks.py $(md5sum < $D0B | cut -c1-8) + s5ctr.py $(md5sum < $HD/../d0ab/s5ctr.py | cut -c1-8), stack4/loadid.py $(md5sum < $LD | cut -c1-8), stack4/mbt.py $(md5sum < $MBT | cut -c1-8), nc/ncmb.py $(md5sum < $NCMB | cut -c1-8), harness mbt2.py $(md5sum < $MBT2 2>/dev/null | cut -c1-8) / ncmb2.py $(md5sum < $NCMB2 2>/dev/null | cut -c1-8) / c8mix.py $(md5sum < $C8MIX 2>/dev/null | cut -c1-8), deepq.py $(md5sum < $DQ | cut -c1-8), mb.py $(md5sum < $MBP | cut -c1-8), stepprobe.py $(md5sum < $SPB | cut -c1-8), greedyset.py $(md5sum < $GS | cut -c1-8), ds41_tests.py $(md5sum < $TESTS | cut -c1-8), dsm-arc.sh $(md5sum < $0 | cut -c1-8) (from i9b/dsm-i9b.sh $(md5sum < $HD/../i9b/dsm-i9b.sh | cut -c1-8)), lib-29.sh $(md5sum < $LIB29 | cut -c1-8), yaml $(md5sum < $Y | cut -c1-8)"
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing; i9bctr.py $CTT_ | loadid.py $ST | ncmb.py $NMT | d0blocks.py --help OK | harness: mbt2.py $M2T / ncmb2.py $N2T / c8mix.py $CMT | all $(echo $FNS | wc -w) functions resolve, tools on PATH, both arms' pins (S5T, draft_sample 0, draft_lend 1$([ $SW = 1 ] && echo ", SP 0, M1 on, IX off, draft_depth0 $D0, sched_row_cap $CAP, ds41_arcoded 0 / $ARCN" || echo "; XARC_SW=0: no int9 / int9b / int10 switch")), sidecars $SIDECAR + $SIDE sha256 OK, yaml accepted (env_file, no lever word), LEVENV with MHC_FUSE=0, without KVR / ARC -- OK | c8 windows: $([ -s $NCMB2 ] && [ -s $MBT2 ] && echo "harness/ncmb2.py + mbt2.py (fixed schedule)" || echo "nc/ncmb.py (harness absent)") | c4 identity reference: $([ $DOC4 = 1 ] && echo "$C4REF present" || echo "not run") | md5: $HMD5"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -iE '^radiance' | grep -vxF "${ATTACH_CLAIMED_CN:-.}" | grep -q . && { xlog "VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { xlog "VIRHE $M225 missing or not verified"; exit 1; }
  [ -s $WARM_REF_225 ] || xlog "note -- reference $WARM_REF_225 missing (not compared)"
  xlog "$ITEM: helpers md5: $HMD5"
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDE; do [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  for f in $WARM_REF_225 $RS5/s5-n5-stats0.json $RS5/guard-s5-n5.txt $C4REF; do [ -e $f ] && echo "+ present (reference): $f" || echo "+ absent (reference, information only): $f"; done
  for a in B C; do echo "+ arm $a ($(NAMEOF $a)): $(armj $a)"; done
  echo "+ base pin + fixed pin: $(pinj B)"
  echo "+ readings: D block $DOD (c4 + c8 old loop under B, dur $DDUR warm $DWARM) | cold [$COLD] $NTOK tok | c8 $C8ORDER (dur $C8DUR warm $C8WARM gens $C8GENS, T 0) | identity $DOID (+ c4 vs d0ab A $DOC4) | essay [$EORDER] $EGEN tok"
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(information here; the real run refuses to start beside one)"
  echo "+ disk now: $(dfree) MB free on / (guard $DISKMIN MB)"
fi
mkd $O $O/guard $O/c8 $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt
stopfile || dstart
summary
unset $SWITCHES
down
xlog "$ITEM: done ($(elapsed) min) -- $O | $([ $NV = 0 ] && echo "no VIRHE line" || echo "$NV VIRHE line(s): see above")"
