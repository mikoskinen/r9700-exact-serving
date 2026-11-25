#!/bin/bash
# dsm-i9b.sh (12.11.2026, queue-ar/dsmain/i9b, ds-main-port int9b builder; RESHAPED 12.11. per the coordinator: ONE SCREEN START + one
# short PROFILED start) -- NOT a queue item; the body of 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzz-i9b.sh. Log tag "RESULT xi9b ", out
# runs/xdsm/i9b (XI9_OUT). Made from a COPY of stack5/dsm-stack5.sh 63649414 (the start, LEVENV / LITS, the fixed pin and the S5T
# arm, stackenv .. gate1, alone, cmpx, ncparse's ncsteps c8 reading; tag xs5 -> xi9b) and of sp/dsm-sp.sh's budok; the profiled start
# after nc/dsm-ncprof-p.sh's START P (its modes() rule-w fix under --profile-ops, docker stop for the closing table). Helpers read
# only: harness/ncmb2.py + mbt2.py + c8mix.py (c8 on the fixed schedule, the co-batch mix, the newcomer step), nc/ncmb.py (its
# fallback only), nc/ds_opprof_nc.py (the per-op split, classes incl. dspark_ctx /
# dspark_query), stack4/loadid.py (greedy-set compare: text + token ids), stack4/sdprobe.py (T 1), stepprobe.py, deepq.py, mb.py,
# greedyset.py, ds41_tests.py; + i9b/i9bctr.py (a COPY of stack5/s5ctr.py + the int9b families; here it only PRINTS the counter
# deltas as INFO). The originals are not changed.
# LEVERS on the ds41-int9b image (ds41-int9 0cb6fd3 + three exact kernel levers behind fn4 switches appended after stage_full_pick:
# mhc_fuse (52), ix_lds (53), ix_ni (54); kSwitches 55; every new switch 0 and RADIANCE_DS41_MHC_FUSE unset = the int9 image):
#   M  mhc_fuse 1 (deepseek41, lever M1): a pcn_a launch of >= 512 rows runs as mhc_post_collapse_rmsnorm_mix (post + collapse + norm
#      + that layer's attention mhc_mix); that layer's mix_a is not issued for those rows. DECLARED by RADIANCE_DS41_MHC_FUSE=0.
#   L  ix_lds 1 (libr4d, lever IX-LDSR2): index_score's prefill WMMA form with the workgroup's keys unpacked once into LDS, two rows
#      a wave (non-keep runs, mxfp4 keys).
#   N  ix_ni 1 (libr4d + deepseek41's counters, lever IX-NI): no -inf written past each row's reach; index_candidates reads below
#      reach; the decode path too.
#   X  all three (the exactness check only).
# THE BASE (per the coordinator): ONE NST 5 START = stack5's (stack4's + RADIANCE_LOG_STEPS=1 + RADIANCE_DS41_ENGDD=0 +
#   RADIANCE_PLACE_HB_RANKS=1,2) + RADIANCE_DS41_MHC_FUSE=0 (M1 declared, off) + RADIANCE_DS41_KV_REPLICATED=1 (lever KVR: passed
#   19/19). BASE PIN = stack5's S5T with draft_sample 0 (not 85) -- "draft_sample 0 + KVR=1 base per the coordinator" -- + lever SP
#   pinned 0 / 0 (SP off: its item, cz-x-kvrz-sp done 08:54, gave no SP value: its counter guard failed, no full-plan step armed /
#   streamed in its readings) + the fixed pin (stack4's + ds41_psk_k 8). Every arm = the base + its lever, POST + epoch_bump, a
#   nudge, every switch READ BACK (GET).
# P0 (CARD 0, no server, before the start): the image's r4d_selftest --case mhc_pcn_mix / ix_lds / ix_ni (the new forms byte for
#   byte vs the stock forms, IX on stress data in the v2 image, + mutation controls); a failure stops the item (XI9_SW=1).
# SCREEN (each lever on its target metric only, ABAB with 2 blocks a side, short windows):
#   (a) M, L, N: cold ~64k TTFT (deepq XI9_SDEPTH 64000, a prefix-cache reset before each, XI9_SNTOK 64 greedy tokens), XI9_SORDER
#       B X B X a lever (X = that lever) -> 12 cold 64k prompts; the mean prefill chunk (non-decode step) ms beside. NOT 136k.
#   (b) N: mb c8 at T 0 on the coordinator's c8 block-mix harness (TESTING.md "12.11.2026 c8 block mix"; harness/ncmb2.py = nc/ncmb.py's
#       taps around harness/mbt2.py's FIXED, SEEDED ARRIVAL SCHEDULE: --sched fixed --pattern 1x8 --seed 1 --window gens --gens 2
#       --edge 0.5 --ignore-eos 1, so a block holds exactly 16 newcomer steps, all singles, ~80-90 s), XI9_C8ORDER B N N B: blocks 1 and 4
#       are the B-vs-B mix check (harness/c8mix.py compare); per block c8mix.py block (the co-batch mix, the newcomer step walls -- it
#       replaces nc/ncsteps.py here). The N-vs-B delta (agg tok/s, newcomer step ms) is read only when the two B mixes MATCH, else
#       "INCONCLUSIVE (mix)" with the numbers (not a fail). These fixed-schedule c8 numbers are a NEW reference: not comparable with
#       old-loop c8 numbers. Falls back to nc/ncmb.py (the old loop) only if the harness is absent at run time, logged.
#   (c) T 1 cost of draft_sample 0: ONE 1u essay T 1 block under the base (stack5's T 1 sdprobe phase: temperature 1 = the server's
#       default, seed 1234, thinking on, effort medium): tok/s, tok/step, acceptance, labelled cross-start vs SD 85 (sdtemp 11.11:
#       essay T 1 81.4 tok/s, acceptance 0.720; stack5 S5T T 1 essay 82.3, "RESULT xs5 ARMS T 1").
#   (d) EXACTNESS (~2 min): greedyset.py's 20 prompts at c1 alone, all three ON (X) vs all OFF (B): text + token ids strict
#       (loadid.py cmp). No loaded identity, no cold identity sets, no chains, no capacity lines, no full guard suites, no GC logits
#       dump (the confirm start's). The screen's cold 64k answers vs the first B of their lever: INFORMATION only.
#   GUARDS: only "switch read back (every POST, GET after) + 0 request errors" (a request error = a VIRHE). Each reading logs its i9b
#     counter deltas (and the S5T families') as INFO, no VIRHE, so the reader sees whether the lever acted.
#   OUTPUT (the coordinator's line rules 12.11.: common sense, not absolute rules): ONE VERDICT line a lever -- the verdict, the
#     numbers (mean delta X vs B %, the B-B and X-X block spreads), one sentence of reasoning: "candidate" = a consistent drop of
#     >= 2 % (every X block better than its B partner), "noise" = within +-1 %, else "borderline -- park" with the number;
#     "path did not run, no reading" when the lever's i9b counters stayed 0 in its X blocks; a single odd block is printed and
#     named and may not veto or crown a lever (the verdict is re-read without it; if that changes it: park). N's second line on c8
#     (agg tok/s and the newcomer step ms, the same rules). The T 1 line, the identity line, a final summary -- all INFO.
# PROFILED START (the coordinator's addition, at the END): a SECOND start, same image, same start env + base pin (all three levers 0)
#   + --profile-ops + RADIANCE_PROFILE_EVERY XI9_PEVERY (8): no runtime-switchable profiler exists in the int9b engine (--profile-ops is
#   start-level; under it no pass is recorded, ctx.cpp tape_on_ needs !profiling_ -- the rule-w check reads "env rule-w, no plain rule
#   line" there, as nc/dsm-ncprof-p.sh). ONE 1u NST 5 essay round (stepprobe essay XI9_PGEN 512 at T 0); the window = [request + XI9_PSKIP
#   2 s (past its prefill step), its end]; the cumulative tables' delta (ds_opprof_nc.py --win, read only) -> RESULT lines: per-op ms
#   of a drafting round (a step): drafter (dspark_ctx + dspark_query) / verify trunk / MoE / attention (+ indexer) / mhc / collectives
#   (+ gather_ar) / dense + norms / launch (lane) waits / host gaps, ranked. PROFILED: not a speed, not comparable with the screen.
# ~30 min estimate (P0 ~3, start + checks ~5.5, switch note ~0.5, (a) 12 x ~20 s = ~4.5, (b) 4 x ~85 s + mix parse = ~6.5, (c) ~0.5, (d)
#   2 x ~65 s = ~2.5 -> screen ~23; the profiled start ~4.5 + the round ~1 + stop / parse ~1 = ~6.5). BUDGET XI9_BUDGET_MIN 25 for the screen
#   (a late section skipped, logged), XI9_TOTAL_MIN 36 for the profiled start (skipped if it would end past it). NO evict, no cc_sim,
#   no power / host / fan settings; production compose files read only. Exact by design; identity reported, never hidden.
# MOCK: XI9_SW=0 = a server without the int9 / int9b switches (queue-ar/tools/mockrad): the arms post NO mhc_fuse / ix_lds / ix_ni and
#   no stage_full_min / stage_full_pick, nothing of them is checked, P0's failure does not stop the item; the start line carries a
#   "VIRHE XI9_SW=0" marker (a mock run only), as CAP48's XC48_CAPSW=0.
# Knobs: XI9_IMAGE, XI9_ITEM, XI9_OUT, XI9_SW (1), XI9_LEVERS (MLN), XI9_SDEPTH (64000), XI9_SNTOK (64), XI9_SORDER (BXBX), XI9_C8ORDER
#   (BNNB), XI9_C8DUR (30), XI9_C8WARM (10), XI9_C8GENS (2), XI9_T1 (1), XI9_IDN (1), XI9_PROF (1), XI9_PEVERY (8), XI9_PGEN (512), XI9_PSKIP (2),
#   XI9_MHC (1: the M / X arms' mhc_fuse), XI9_SPN (0) / XI9_SPK (0), XI9_P0MIN ("mhc_pcn_mix:7:2 ix_lds:6:1 ix_ni:6:2"),
#   XI9_DISKMIN_MB (11000), XI9_BUDGET_MIN (25), XI9_TOTAL_MIN (36), XI9_SIDECAR, XI9_SIDESHA; XI9_DRY_ALLOW_OLD=1 (DRYRUN only).
# DRYRUN=1 bash dsm-i9b.sh : prints every command; checks the image literals (int9's + int9b's + KVR's), every helper (files, parse,
#   selftests), the sidecars, the yaml (env_file, RADIANCE_PROFILE_EVERY), the disk; no server, no GPU, writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/../ds41-stack2-tp4.yaml                   # stack2's yaml, read only (env_file ${XS2_ENVFILE}; RADIANCE_PROFILE_EVERY ${PROFEVERY})
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh
O=${XI9_OUT:-$R/xdsm/i9b}; export STATSDIR=$O/stats
RD=$R/xdsm/draft; RM=$R/xdsm; RS5=$R/xdsm/stack5
export IMAGE=${XI9_IMAGE:?set XI9_IMAGE}
ITEM=${XI9_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzzz-i9b}
PSKK=8; BIC=3; NOA=2; OCC=6; OCCK=32; DSA=0; HBR=0; HBRANKS=1,2      # stack5's S5T values, draft_sample 0 (the coordinator)
SW=${XI9_SW:-1}; LEVERS=${XI9_LEVERS:-MLN}; SDEPTH=${XI9_SDEPTH:-64000}; SNTOK=${XI9_SNTOK:-64}; SORDER=${XI9_SORDER:-BXBX}
C8ORDER=${XI9_C8ORDER:-BNNB}; C8DUR=${XI9_C8DUR:-30}; C8WARM=${XI9_C8WARM:-10}; C8GENS=${XI9_C8GENS:-2}
DOT1=${XI9_T1:-1}; DOID=${XI9_IDN:-1}; DOPROF=${XI9_PROF:-1}; PEVERY=${XI9_PEVERY:-8}; PGEN=${XI9_PGEN:-512}; PSKIP=${XI9_PSKIP:-2}
MHC=${XI9_MHC:-1}; SPN=${XI9_SPN:-0}; SPK=${XI9_SPK:-0}
P0MIN=${XI9_P0MIN:-mhc_pcn_mix:7:2 ix_lds:6:1 ix_ni:6:2}
DISKMIN=${XI9_DISKMIN_MB:-11000}; BUDGET=${XI9_BUDGET_MIN:-25}; TOTAL=${XI9_TOTAL_MIN:-36}; ALLOWOLD=${XI9_DRY_ALLOW_OLD:-0}
SIDE=${XI9_SIDECAR:-dspark-dlw-q8_0.rad}; SIDESHA=${XI9_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
MT=192; SEED=1234
case $SW in 0|1) ;; *) echo "XI9_SW is 1 (the item) or 0 (a server without the int9 / int9b switches: the mock) (got '$SW')"; exit 1 ;; esac
for v in $SDEPTH $SNTOK $C8DUR $C8WARM $C8GENS $DOT1 $DOID $DOPROF $PEVERY $PGEN $PSKIP $MHC $SPN $SPK $DISKMIN $BUDGET $TOTAL; do case $v in ''|*[!0-9]*) echo "a numeric knob is not a number ('$v')"; exit 1 ;; esac; done
case $LEVERS in ''|*[!MLN]*) echo "XI9_LEVERS takes M L N (got '$LEVERS')"; exit 1 ;; esac
case $SORDER in *[!BX]*|"") echo "XI9_SORDER takes B and X (X = the lever) (got '$SORDER')"; exit 1 ;; esac
case $C8ORDER in *[!BN]*) echo "XI9_C8ORDER takes B and N (got '$C8ORDER')"; exit 1 ;; esac
case $C8ORDER in B*B) ;; *) echo "XI9_C8ORDER must start and end with B (the B-vs-B mix check; got '$C8ORDER')"; exit 1 ;; esac
[ $C8WARM -lt $C8DUR ] || { echo "the c8 warm-up is not shorter than the window"; exit 1; }
[ $MHC -ge 1 ] && [ $MHC -le 1048576 ] || { echo "XI9_MHC 1..1048576 (mhc_fuse's on values)"; exit 1; }
[ $SPN -le 1048576 ] && [ $SPK -le 64 ] || { echo "XI9_SPN 0..1048576, XI9_SPK 0..64"; exit 1; }
[ $PEVERY -ge 1 ] && [ $PSKIP -ge 1 ] && [ $PGEN -ge 64 ] || { echo "XI9_PEVERY >= 1, XI9_PSKIP >= 1, XI9_PGEN >= 64"; exit 1; }
for c in $P0MIN; do case $c in *:*[!0-9:]*|*::*) echo "XI9_P0MIN takes case:bytes:mutations (got '$c')"; exit 1 ;; esac; done
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
S4=$HD/../stack4
LD=$S4/loadid.py; PRB=$S4/sdprobe.py                                                               # stack4's, read only
CTR=$HD/i9bctr.py                                                                                   # this item's (INFO only)
NCMB=$HD/../nc/ncmb.py; OPNC=$HD/../nc/ds_opprof_nc.py                                             # nc's, read only (ncmb.py: the fallback only)
MBT2=$HD/../harness/mbt2.py; NCMB2=$HD/../harness/ncmb2.py; C8MIX=$HD/../harness/c8mix.py          # the harness (read only): c8 on mbt2's fixed schedule
DQ=$QA/deepq.py; MBP=$D/mb.py; TESTS=$D/ds41_tests.py; SPB=$D/stepprobe.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
T00=$(date +%s); NV=0
xlog() { case "$*" in *VIRHE*) NV=$((NV + 1)) ;; esac; log "RESULT xi9b $*"; }
elapsed() { echo $(( ($(date +%s) - T00) / 60 )); }
# ---------------------------------------------------------------- the start (= stack5's + RADIANCE_DS41_MHC_FUSE=0 + RADIANCE_DS41_KV_REPLICATED=1)
GPTAB=q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422
SIDECAR=dspark-conf.f32
LEVENV="RADIANCE_DS41_GG_PF=1 RADIANCE_DS41_GG_PF_TABLE=$GPTAB RADIANCE_DS41_F32_PF=25 RADIANCE_DS41_ARENA_DECODE_ROWS=8,48
  RADIANCE_DS41_ROUTER_BF16=1 RADIANCE_DS41_EMB_RAM=1 RADIANCE_PLACE_SHADOW_SLOTS=16 RADIANCE_XPREFETCH=0 RADIANCE_DS41_DSPARK_WALK=1
  RADIANCE_DS41_ARGATHER_MAX=48 RADIANCE_DS41_QNFOLD=1 RADIANCE_DS41_SWFOLD=1 RADIANCE_DRAFT_CTXSKIP=0 RADIANCE_DS41_DSPARK_CONF=/models/$SIDECAR
  RADIANCE_DRAFT_LEND=0 RADIANCE_DRAFT_PROBE_BACKOFF=0 RADIANCE_DRAFT_LEND_BORROW=0 RADIANCE_DRAFT_SAMPLE=0 RADIANCE_WEIGHT_SIDECAR=/models/$SIDE
  RADIANCE_DS41_BI=0 RADIANCE_DRAFT_CTXSKIP_SLOTS=129 RADIANCE_DRAFT_LEND_WEIGHTS=0 RADIANCE_DS41_PSK=0
  RADIANCE_LOG_STEPS=1 RADIANCE_DS41_ENGDD=0 RADIANCE_PLACE_HB_RANKS=$HBRANKS
  RADIANCE_DS41_MHC_FUSE=0 RADIANCE_DS41_KV_REPLICATED=1"
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
# + KVR's two lines (kvr item's start log) and lever M1's declare line (int9b)
LITS="$LITS|lever,KVR,declared,(RADIANCE_DS41_KV_REPLICATED=1)|held,once,a,slot:,5.39,KiB,a,slot|lever,M1,(mhc_fuse),declared"
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": 8, "drafter_mx": 1, "gg_pf": 1, "f32_pf": 25, "dspark_walk": 1, "ds41_argather": 48, "idx_split": 1, "ds41_qnfold": 1, "ds41_swfold": 1, "draft_ctxskip": 1, "dspark_conf": 1, "dspark_conf_rows": 1, "draft_lend": 1, "draft_probe_backoff": 3, "draft_hybrid": 1}'
FIX=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update({"draft_lend_hyst": 0, "arena_d0": 0, "draft_lend_w": 0, "draft_occ_k": int(sys.argv[2]), "ds41_psk_k": int(sys.argv[3])}); print(json.dumps(d))' "$FIX5" "$OCCK" "$PSKK") || { echo "pin JSON"; exit 1; }
S5J="\"ds41_bi\": 2, \"draft_sample\": $DSA, \"draft_occ\": $OCC, \"place_lf\": 0, \"sched_bi_chunk\": $BIC, \"sched_noa\": $NOA, \"ds41_psk_l20\": 1, \"dspark_pfctx\": 1, \"ds41_engdd\": 1"
# ---------------------------------------------------------------- the base pin (S5T, draft_sample 0, + SP pinned) and the arms
SPJ=", \"stage_full_min\": $SPN, \"stage_full_pick\": $SPK"; [ $SW = 1 ] || SPJ=""        # XI9_SW=0: the mock has no int9 switch
TJ="{$S5J, \"ds41_psk\": 4, \"ds41_psk_cr\": 1, \"stage_short\": 0, \"stage_pick\": 0, \"place_hb\": 0$SPJ}"   # = stack5's armj T, draft_sample 0, + SP
armj() {   # armj B|M|L|N|X: the arm's int9b switches (JSON) on top of the base; {} under XI9_SW=0 (the mock)
  [ $SW = 1 ] || { echo '{}'; return 0; }
  case $1 in
    B) echo '{"mhc_fuse": 0, "ix_lds": 0, "ix_ni": 0}' ;;
    M) echo "{\"mhc_fuse\": $MHC, \"ix_lds\": 0, \"ix_ni\": 0}" ;;
    L) echo '{"mhc_fuse": 0, "ix_lds": 1, "ix_ni": 0}' ;;
    N) echo '{"mhc_fuse": 0, "ix_lds": 0, "ix_ni": 1}' ;;
    X) echo "{\"mhc_fuse\": $MHC, \"ix_lds\": 1, \"ix_ni\": 1}" ;;
  esac
}
NAMEOF() { case $1 in B) echo B ;; M) echo M1 ;; L) echo IX-LDS ;; N) echo IX-NI ;; X) echo X ;; esac; }
TARGET() { case $1 in M) echo "cold ${SDEPTH} TTFT (mhc_post_collapse_rmsnorm + mhc_mix in prefill)" ;; L) echo "cold ${SDEPTH} TTFT (index_score's prefill WMMA form)" ;;
           N) echo "cold ${SDEPTH} TTFT (index_score's dead-region writes) + c8 (the decode path)" ;; esac; }
pinj() {   # pinj ARM [EXTRAJSON]: the fixed pin + the base (S5T, draft_sample 0, SP) + the arm + extra as one POST body
  local x=${2:-}; [ -n "$x" ] || x='{}'
  python3 -c 'import json,sys
d = {}
for a in sys.argv[1:]: d.update(json.loads(a))
print(json.dumps(d))' "$FIX" "$TJ" "$(armj $1)" "$x"
}
DESC="NST 5 stack5 start (stack4's + RADIANCE_LOG_STEPS=1, E1 declared off, HB ranks $HBRANKS) + RADIANCE_DS41_MHC_FUSE=0 (M1 declared, off) + RADIANCE_DS41_KV_REPLICATED=1 (KVR) | base pin = stack5's S5T with draft_sample 0 + SP pinned (stage_full_min $SPN, stage_full_pick $SPK: SP off) + the fixed pin (ds41_psk_k $PSKK) | arms B $(armj B), M1 $(armj M), IX-LDS $(armj L), IX-NI $(armj N), X $(armj X)"
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
  RADIANCE_DS41_MHC_FUSE RADIANCE_DS41_IX_LDS RADIANCE_DS41_IX_NI"
# ---------------------------------------------------------------- copied from stack5/dsm-stack5.sh (63649414; tag xs5 -> xi9b): stackenv .. gate1 (modes: + nc/dsm-ncprof-p.sh's rule-w check under --profile-ops)
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xi9b $1 warm-up identity vs $2"; return 0; }
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
  [ "$DRY" = 1 ] && { echo "+ GET /stats > $O/$1-stats0.json; RESULT xi9b $1 VRAM at start (experts per tier; vs stack4 s4-n5)"; return 0; }
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xi9b $tag $3: greedy set c1 alone under $(NAMEOF $2) $(armj $2): 20 of 20 answered (else a request-error line); counters INFO"; return 0; }
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
ctrinfo() {   # ctrinfo A B: every counter delta between the snapshots (i9bctr.py, no need / forbid: INFORMATION, never a VIRHE)
  [ "$DRY" = 1 ] && { echo "(dry: python3 -I -B $CTR guard --st0 st-$1 --st1 st-$2 --fn0 fn-$1 --fn1 fn-$2, printed as INFO)"; return 0; }
  timeout 60 python3 -I -B $CTR guard --st0 $O/st-$1.json --st1 $O/st-$2.json --fn0 $O/fn-$1.json --fn1 $O/fn-$2.json 2>&1 | tail -1 | sed 's/^OK | //' | cut -c1-900
}
i9bnow() {   # i9bnow: GET /admin/fn4's int9b switches and "i9b" counters, one line
  [ "$DRY" = 1 ] && { echo "(dry)"; return 0; }
  fn4get | python3 -c 'import json,sys
d = json.load(sys.stdin); w = d.get("switches") or {}
print("mhc_fuse %s ix_lds %s ix_ni %s | i9b %s" % (w.get("mhc_fuse"), w.get("ix_lds"), w.get("ix_ni"), json.dumps(d.get("i9b"))))' 2>/dev/null | cut -c1-600
}
i9bd() {   # i9bd A B -> JSON: the "i9b" counter deltas between GET /admin/fn4 snapshots A and B (declared: B's)
  [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }
  python3 -c 'import json,sys
def rd(p):
    try: return json.load(open(p)).get("i9b") or {}
    except Exception: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
o = {}
for g, ks in (("mhc", ("pairs", "rows", "small")), ("ix", ("lds_launches", "lds_rows", "ni_launches", "ni_cand_launches")), ("ni", ("runs", "dead_bytes"))):
    for k in ks:
        x, y = (a.get(g) or {}).get(k, 0), (b.get(g) or {}).get(k, 0)
        o["%s_%s" % (g, k)] = (y - x) if isinstance(x, (int, float)) and isinstance(y, (int, float)) else None
o["mhc_declared"] = (b.get("mhc") or {}).get("declared")
print(json.dumps(o))' $O/fn-$1.json $O/fn-$2.json 2>/dev/null || echo '{}'
}
DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ulimit memlock=-1 --ipc=host -e HIP_VISIBLE_DEVICES=0"
gpurun() {   # gpurun NAME SECONDS docker-run-args...: bounded, named, removed even if its kernel hangs (e1/dsm-e1.sh's, copied)
  local name=$1 t=$2; shift 2
  timeout -k 15 $t docker run --rm --init --stop-timeout 10 --name $name "$@"; local rc=$?
  docker rm -f $name > /dev/null 2>&1 || true
  { [ $rc = 124 ] || [ $rc = 137 ]; } && echo "gpurun: $name stopped after $t s (rc $rc)" >&2
  return $rc
}
p0sel() {   # p0sel: ONE card (CARD 0), no server -- the image's r4d_selftest --case mhc_pcn_mix / ix_lds / ix_ni (e1's p0sel pattern)
  local c cs nb nm r rc f bad=0 nok nfail nbyte nmut
  for c in $P0MIN; do
    cs=${c%%:*}; nb=$(echo $c | cut -d: -f2); nm=$(echo $c | cut -d: -f3); f=$O/p0-$cs.txt
    run "gpurun xi9b-p0-$cs-$$ 900 $DEV --entrypoint /opt/radiance/bin/r4d_selftest $IMAGE /opt/radiance/share/radiance/kernels/libr4d.so --case $cs > $f 2>&1"; rc=$?
    if [ "$DRY" = 1 ]; then echo "+ RESULT xi9b P0 $cs on CARD 0: rc 0, 0 FAIL, >= $nb ok 'byte for byte' lines and >= $nm ok 'mutation .. caught' lines, else a failure line and the item stops (no server start); info / timing lines logged"; continue; fi
    nok=$(grep -c '^  ok' $f); nfail=$(grep -c '^  FAIL' $f)
    nbyte=$(grep '^  ok' $f | grep -c 'byte for byte'); nmut=$(grep '^  ok' $f | grep -i 'mutation' | grep -c 'caught')
    r="$nok ok, $nfail FAIL, $nbyte byte-for-byte (need $nb), $nmut mutation-caught (need $nm), rc $rc"
    while IFS= read -r l; do xlog "P0 $cs info: ${l:0:600}"; done < <(grep -viE '^  (ok|FAIL)' $f | grep -iE 'info|time| us|ms\b|GB/s|speed' | head -24)
    if [ $rc = 0 ] && [ "$nfail" = 0 ] && [ "$nbyte" -ge $nb ] && [ "$nmut" -ge $nm ]; then
      xlog "P0 $cs on one card (r4d_selftest --case): OK $r | $(grep -E '^  ok' $f | sed 's/^ *ok *//' | cut -c1-110 | tr '\n' ';' | cut -c1-1500)"
    else
      bad=1; xlog "P0 $cs on one card (r4d_selftest --case): VIRHE $r | $(grep -E '^  FAIL|FATAL|error|Error|fault|abort|not emulated' $f | head -6 | sed 's/^ *//' | cut -c1-200 | tr '\n' ';' | cut -c1-1200)"
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
readback() {   # readback TAG: the start's own words read back -- the env (KVR=1, MHC_FUSE=0), KVR's "5.39 KiB a slot" line, M1's declare line
  [ "$DRY" = 1 ] && { echo "+ readback $1: RADIANCE_DS41_KV_REPLICATED=1 + RADIANCE_DS41_MHC_FUSE=0 in the container env; the start log's 'kv tiers: .. held once a slot: 5.39 KiB a slot' line and 'lever KVR declared' line; the 'lever M1 (mhc_fuse) declared' line"; return 0; }
  local e k d m ok=1
  e=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null | grep -cxE 'RADIANCE_DS41_KV_REPLICATED=1|RADIANCE_DS41_MHC_FUSE=0')
  k=$(docker logs $CN 2>&1 | grep -m1 '5.39 KiB a slot' | sed 's/^radiance: //; s/^[A-Z]\[[0-9]\] //; s/^[A-Z] *//' | cut -c1-400)
  d=$(docker logs $CN 2>&1 | grep -m1 'lever KVR declared' | sed 's/^radiance: //' | cut -c1-200)
  m=$(docker logs $CN 2>&1 | grep -m1 'lever M1 (mhc_fuse) declared' | sed 's/^radiance: //; s/^[A-Z]\[[0-9]\] //; s/^[A-Z] *//' | cut -c1-400)
  [ "$e" = 2 ] && [ -n "$k" ] && [ -n "$d" ] && [ -n "$m" ] || ok=0
  xlog "$1 read back: env KVR=1 + MHC_FUSE=0 $e of 2 | KVR: ${k:-NONE} | ${d:-no 'lever KVR declared' line} | M1: ${m:-NONE}$([ $ok = 1 ] || echo " -- VIRHE the start's words are not read back")"
  [ $ok = 1 ]
}
swnote() {   # swnote TAG: start values (base switches; int9b 0, M1 declared, i9b counters 0 under XI9_SW=1); refusals; every arm read back
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 at start (gg_gemv 2, aql_acq 2, K1 three 1, epoch 0, draft_capable 1, draft_sample 0; ds41_psk 0, sched_bi_chunk / sched_noa / ds41_psk_l20 / dspark_pfctx / ds41_engdd / ds41_psk_cr / stage_short / stage_pick / place_hb 0$([ $SW = 1 ] && echo "; stage_full_min / stage_full_pick 0; mhc_fuse 0, ix_lds 0, ix_ni 0; i9b present, mhc declared 1, every i9b counter 0" || echo "; int9 / int9b NOT CHECKED (XI9_SW=0)")); refusals stage_short 65537$([ $SW = 1 ] && echo " / mhc_fuse 1048577 / ix_lds 2 / ix_ni 2") -> 400; arms $(for a in B M L N X; do echo -n "$a $(armj $a) | "; done)each applied and read back (draft_sample 0), then B"; return 0; }
  local s0 g c1=400 c2=400 c3=400 c4 a sets="" bad=""
  s0=$(fn4get)
  c4=$(fn4code '{"stage_short": 65537}')
  [ $SW = 1 ] && { c1=$(fn4code '{"mhc_fuse": 1048577}'); c2=$(fn4code '{"ix_lds": 2}'); c3=$(fn4code '{"ix_ni": 2}'); }
  for a in B M L N X B; do
    arm $a || bad="$bad $a"; g=$(fn4get)
    sets="$sets $a:$(echo "$g" | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print("mhc=%s,lds=%s,ni=%s,ds=%s,psk=%s/%s,cr=%s,bi=%s,sfm=%s,sfp=%s,ep=%s" % (w.get("mhc_fuse"), w.get("ix_lds"), w.get("ix_ni"), w.get("draft_sample"), w.get("ds41_psk"), w.get("ds41_psk_k"), w.get("ds41_psk_cr"), w.get("ds41_bi"), w.get("stage_full_min"), w.get("stage_full_pick"), d.get("epoch")))' 2>/dev/null)"
  done
  xlog "$1 switch note: at start $(echo "$s0" | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print("switches %s | i9b %s | epoch %s" % (json.dumps({k: w.get(k) for k in ("mhc_fuse", "ix_lds", "ix_ni", "stage_full_min", "stage_full_pick", "draft_sample", "ds41_psk", "ds41_bi", "gg_gemv", "aql_acq")}), json.dumps(d.get("i9b")), d.get("epoch")))' 2>/dev/null | cut -c1-1200) | stage_short 65537 -> HTTP $c4$([ $SW = 1 ] && echo ", mhc_fuse 1048577 -> HTTP $c1, ix_lds 2 -> HTTP $c2, ix_ni 2 -> HTTP $c3" || echo " | int9 / int9b NOT CHECKED (XI9_SW=0)") | arms applied (read back):$sets$([ -n "$bad" ] && echo " -- VIRHE not applied:$bad")"
  python3 -c 'import json,sys
s0 = json.loads(sys.argv[1]); sw = sys.argv[2] == "1"; w = s0["switches"]; i9 = s0.get("i9b")
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) and s0["epoch"] == 0
ok = ok and w["draft_adaptive"] == 1 and s0["draft_capable"] == 1 and w.get("ds41_psk") == 0 and w.get("draft_sample") == 0
ok = ok and all(w.get(k) == 0 for k in ("sched_bi_chunk", "sched_noa", "ds41_psk_l20", "dspark_pfctx", "ds41_engdd", "ds41_psk_cr", "stage_short", "stage_pick", "place_hb"))
if sw:
    ok = ok and all(w.get(k) == 0 for k in ("stage_full_min", "stage_full_pick", "mhc_fuse", "ix_lds", "ix_ni"))
    ok = ok and isinstance(i9, dict) and (i9.get("mhc") or {}).get("declared") == 1
    ok = ok and all(v == 0 for g in ("mhc", "ix", "ni") for k, v in (i9.get(g) or {}).items() if k != "declared")
sys.exit(0 if ok else 1)' "$s0" $SW 2>/dev/null && [ "$c1" = 400 ] && [ "$c2" = 400 ] && [ "$c3" = 400 ] && [ "$c4" = 400 ] && [ -z "$bad" ]
}
coldx() {   # coldx LEVER ARM K: deepq's fixed prompt at SDEPTH after a reset, SNTOK greedy tokens -> TTFT, cached, chunk ms; counters INFO
  local lv=$1 a=$2 k=$3 nm=s$1-$3$2 t0=$(date +%s) j md
  arm $a || return 1
  pcreset $nm
  msnap $nm-0; ssnap $nm-0
  run "(cd $D && timeout 900 python3 $DQ $O/$nm.txt $SDEPTH $SNTOK > $O/$nm.out 2>&1)"
  msnap $nm-1; ssnap $nm-1
  [ "$DRY" = 1 ] && { echo "+ RESULT xi9b screen $lv $nm: $(NAMEOF $a) $(armj $a), deepq $SDEPTH cold, $SNTOK tok: TTFT, cached, chunk ms (status 200 or a request-error line); INFO i9b deltas -> $O/screen.jsonl"; return 0; }
  j=$(grep -m1 '^DEEPQ ' $O/$nm.out | cut -c7-)
  [ -n "$j" ] || { xlog "screen $lv $nm: VIRHE request error: no answer ($(tail -2 $O/$nm.out | tr '\n' ' ' | cut -c1-300))"; return 1; }
  md=$(mdelta $nm-0 $nm-1)
  python3 -c 'import json,sys
d = json.loads(sys.argv[1]); m = json.loads(sys.argv[2] or "{}")
d.update(nondecode_ms=m.get("nondecode_ms"), nondecode_steps=m.get("nondecode_steps"), lever=sys.argv[3], arm=sys.argv[4], slot=int(sys.argv[5]), name=sys.argv[6],
         text=sys.argv[7], i9b=json.loads(sys.argv[8] or "{}"))
print(json.dumps(d))' "$j" "$md" "$lv" "$a" "$k" "$nm" "$O/$nm.txt" "$(i9bd $nm-0 $nm-1)" >> $O/screen.jsonl
  xlog "screen $(NAMEOF $lv) $nm ($(NAMEOF $a) $(armj $a), deepq $SDEPTH cold after a reset, $SNTOK greedy tokens): $(python3 -c 'import json,sys
d, m = json.loads(sys.argv[1]), json.loads(sys.argv[2] or "{}")
print("status %s, prompt %s tok, cached %s, TTFT %s s (%s prefill tok/s), %s tok | non-decode steps %s, %s ms mean%s" % (d.get("status"), d.get("prompt_tokens"), d.get("cached"), d.get("ttft"),
      round((d["prompt_tokens"] - (d.get("cached") or 0)) / d["ttft"], 1) if d.get("ttft") and d.get("prompt_tokens") else "-", d.get("tokens"), m.get("nondecode_steps"), m.get("nondecode_ms"),
      "" if d.get("status") == 200 else " -- VIRHE request error (status %s)" % d.get("status")))' "$j" "$md" 2>&1 | cut -c1-500) | INFO i9b $(i9bd $nm-0 $nm-1) | INFO counters: $(ctrinfo $nm-0 $nm-1 | cut -c1-500) | $(thermline $t0 $(date +%s))"
}
screen() {   # (a) per lever in LEVERS: SORDER with X = the lever; the cold 64k answers vs the first B of the lever (INFORMATION)
  local lv k s a fb f
  for lv in $(echo $LEVERS | grep -o .); do
    budok "screen $(NAMEOF $lv)" $(( ${#SORDER} * (10 + SDEPTH / 6000) / 60 + 1 )) || continue
    k=0; fb=""
    for s in $(echo $SORDER | grep -o .); do
      k=$((k + 1)); stopfile && return 0
      a=B; [ $s = X ] && a=$lv
      coldx $lv $a $k
      [ "$DRY" = 1 ] && continue
      f=$O/s$lv-$k$a.txt
      [ -s $f ] || continue
      [ -n "$fb" ] || { [ $a = B ] && { fb=$f; continue; }; }
      [ -n "$fb" ] && echo "info s$lv-$k$a-vs-B $(cmpfile $fb $f)" >> $O/identity-info.txt && xlog "information: s$lv-$k$a ($(NAMEOF $a)) cold $SDEPTH answer vs the first B of the $(NAMEOF $lv) screen: $(cmpfile $fb $f)"
    done
  done
}
c8x() {   # c8x I ARM: one mb c8 block at temperature 0 on the harness's fixed, seeded schedule (harness/ncmb2.py, TESTING.md "12.11.2026
          # c8 block mix": 16 newcomer steps a block, gens 2); 0 request errors. Without the harness (absent at run time): nc/ncmb.py, logged.
  local i=$1 a=$2 nm=c8-$1$2 t0 base=$O/c8/c8-$1$2 j tool
  arm $a || return 1
  ssnap $nm-0
  t0=$(date +%s)
  if [ -s $NCMB2 ] && [ -s $MBT2 ]; then
    tool=ncmb2
    run "(cd $D && timeout $((C8DUR + C8WARM + 900)) python3 -I -B $NCMB2 --mb $MBP --mode phase --n 8 --temperature 0 --tag xi9b-$nm --dur $C8DUR --warm $C8WARM --reqs $base-reqs.jsonl --stats-prefix $base-st --samples $base-samp.jsonl --sched fixed --pattern 1x8 --seed 1 --window gens --gens $C8GENS --edge 0.5 --ignore-eos 1 > $base.txt 2>&1)"
  else
    tool=ncmb
    run "(cd $D && timeout $((C8DUR + C8WARM + 900)) python3 -I -B $NCMB --mb $MBP --mode phase --n 8 --temperature 0 --tag xi9b-$nm --dur $C8DUR --warm $C8WARM --reqs $base-reqs.jsonl --stats-prefix $base-st --samples $base-samp.jsonl > $base.txt 2>&1)"
  fi
  ssnap $nm-1
  [ "$DRY" = 1 ] && { echo "+ RESULT xi9b c8 $nm ($(NAMEOF $a) $(armj $a), mb c8 T 0 via $([ -s $NCMB2 ] && echo "harness/ncmb2.py: fixed schedule 1x8 seed 1, gens $C8GENS, edge 0.5, ignore_eos (a NEW reference)" || echo "nc/ncmb.py: the old loop")): MB agg_dec_tps / decode_share / per-stream p50 / the client mix, errors 0 (else a request-error line), INFO i9b deltas -> $O/c8.jsonl"; return 0; }
  j=$(grep -m1 '^MB ' $base.txt | cut -c4-)
  [ -n "$j" ] || { xlog "c8 $nm: VIRHE request error: no MB line ($(tail -3 $base.txt | tr '\n' ' ' | cut -c1-300))"; return 1; }
  python3 -c 'import json,sys
d = json.loads(sys.argv[1])
print(json.dumps(dict(name=sys.argv[2], arm=sys.argv[3], tool=sys.argv[5], agg_dec_tps=d.get("agg_dec_tps"), decode_share=d.get("decode_share"), per_stream_dec_p50=d.get("per_stream_dec_p50"),
      ttft_p50=d.get("ttft_p50"), requests=d.get("requests"), errors=d.get("errors"), temperature=d.get("temperature"), sched=d.get("sched"), window=d.get("window"),
      window_timeout=d.get("window_timeout"), mix_client=d.get("mix_client"), i9b=json.loads(sys.argv[4] or "{}"))))' \
    "$j" "$nm" "$a" "$(i9bd $nm-0 $nm-1)" "$tool" >> $O/c8.jsonl
  xlog "c8 $nm ($(NAMEOF $a) $(armj $a), mb c8 T 0 via $tool, $(( $(date +%s) - t0 )) s): $(tail -1 $O/c8.jsonl | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
print("agg %s tok/s, decode_share %s, per-stream p50 %s, TTFT p50 %s, %s requests, %s errors%s | sched %s, window %s%s, client mix %s | INFO i9b %s" % (d.get("agg_dec_tps"), d.get("decode_share"), d.get("per_stream_dec_p50"), d.get("ttft_p50"),
      d.get("requests"), d.get("errors"), "" if d.get("errors") == 0 and (d.get("requests") or 0) > 0 else " -- VIRHE request errors", d.get("sched") or "mb", json.dumps(d.get("window"))[:120],
      (" (window timeout: %s)" % d.get("window_timeout")) if d.get("window_timeout") else "", json.dumps(d.get("mix_client"))[:200], json.dumps(d.get("i9b"))))' 2>&1 | cut -c1-900) | INFO counters: $(ctrinfo $nm-0 $nm-1 | cut -c1-500)"
}
c8s() {   # (b) the c8 blocks in C8ORDER (B N N B), then one engine-log capture; per block c8mix.py block (the co-batch mix and the newcomer
          # step walls: it replaces nc/ncsteps.py here), c8mix.py compare B1 vs B4 (the B-vs-B mix check) and over every block (INFO)
  local i=0 a n=${#C8ORDER} ins="" b1="" bl=""
  if [ -s $NCMB2 ] && [ -s $MBT2 ]; then xlog "c8: harness/mbt2.py used through harness/ncmb2.py (TESTING.md 12.11.2026 c8 block mix: fixed, seeded schedule, pattern 1x8, seed 1, window gens $C8GENS = 16 newcomer steps a block, edge 0.5, ignore_eos; order $C8ORDER, blocks 1 and $n the B-vs-B mix check; a NEW reference, not comparable with old-loop c8 numbers; mbt2.py $(md5sum < $MBT2 | cut -c1-8), ncmb2.py $(md5sum < $NCMB2 | cut -c1-8), c8mix.py $(md5sum < $C8MIX 2>/dev/null | cut -c1-8))"
  else xlog "c8: harness/mbt2.py / ncmb2.py NOT present at run time -- mbt2.py NOT used; nc/ncmb.py runs the blocks (mb's closed loop, a time window $C8WARM..$C8DUR s; old-loop numbers)"; fi
  for a in $(echo $C8ORDER | grep -o .); do i=$((i + 1)); stopfile && return 0; c8x $i $a; done
  capture $O/c8-engine.log.gz
  [ "$DRY" = 1 ] && { echo "+ per block: python3 -I -B $C8MIX block --log $O/c8-engine.log.gz --reqs $O/c8/<b>-reqs.jsonl --tag xi9b-<b> --mbline $O/c8/<b>.txt --label <b> --json $O/c8/<b>-mix.json; then $C8MIX compare --in B1=$O/c8/c8-1B-mix.json --in B$n=$O/c8/c8-${n}B-mix.json (MATCH: the N-vs-B delta is read; else INCONCLUSIVE (mix) with the numbers) + compare over every block (INFO)"; return 0; }
  [ -s $C8MIX ] || { xlog "c8: c8mix.py absent -- no mix reading, the N-vs-B delta is printed as INCONCLUSIVE (mix)"; return 0; }
  i=0
  for a in $(echo $C8ORDER | grep -o .); do
    i=$((i + 1)); b=c8-$i$a
    [ -s $O/c8/$b-reqs.jsonl ] || { xlog "c8 $b: no taps (block not run)"; continue; }
    timeout 300 python3 -I -B $C8MIX block --log $O/c8-engine.log.gz --reqs $O/c8/$b-reqs.jsonl --tag xi9b-$b --mbline $O/c8/$b.txt --label $b --json $O/c8/$b-mix.json > $O/c8/$b-mix.out 2>&1
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
t1x() {   # (c) ONE 1u essay T 1 block under the base (stack5's T 1 sdprobe phase: server-default sampling, seed 1234, thinking on)
  local j
  arm B || return 1
  ssnap t1-0
  run "(cd $D && timeout 1500 python3 -I -B $PRB --mode sampled --seed $SEED --runs 1 --thinking on --effort medium --prompts essay --tag xi9b-t1 > $O/t1.txt 2>&1)"
  ssnap t1-1
  [ "$DRY" = 1 ] && { echo "+ RESULT xi9b T1 (draft_sample 0 base, sdprobe essay T 1 seed $SEED thinking on): tok/s, tok/step, acceptance (status 200 or a request-error line) | cross-start SD 85: sdtemp 11.11 essay T 1 81.4 tok/s acc 0.720, stack5 S5T T 1 essay 82.3"; return 0; }
  j=$(grep -m1 '^J ' $O/t1.txt | cut -c3-)
  [ -n "$j" ] || { xlog "T1: VIRHE request error: no J line ($(tail -3 $O/t1.txt | tr '\n' ' ' | cut -c1-300))"; return 1; }
  echo "$j" > $O/t1.json
  xlog "T1 (the cost of draft_sample 0; base $(armj B), sdprobe essay at the server default T 1, seed $SEED, thinking on, effort medium): $(python3 -c 'import json,sys
d = json.loads(sys.argv[1])
print("status %s, %s tok, decode %s tok/s, %s tok/step, acceptance %s, %s ms/step, reasoning_chars %s%s | cross-start, SD 85 (draft_sample 85), other starts: sdtemp 11.11 essay T 1 81.4 tok/s acc 0.720; stack5 S5T T 1 essay 82.3 (RESULT xs5 ARMS T 1) -> %s vs 81.4 / %s vs 82.3" % (
    d.get("status"), d.get("tokens"), d.get("decode_tps"), d.get("tokens_per_step"), d.get("acceptance"), d.get("ms_per_decode_step"), d.get("reasoning_chars"),
    "" if d.get("status") == 200 else " -- VIRHE request error", ("%+.1f %%" % (100 * (d["decode_tps"] / 81.4 - 1))) if d.get("decode_tps") else "-",
    ("%+.1f %%" % (100 * (d["decode_tps"] / 82.3 - 1))) if d.get("decode_tps") else "-"))' "$j" 2>&1 | cut -c1-700)"
}
idcheck() {   # (d) exactness: the greedy set alone, all three OFF (B) then ON (X), text + token ids strict (loadid.py cmp)
  alone i9 B a-B
  alone i9 X a-X
  cmpx i9 X-vs-B $O/i9-a-B.jsonl $O/i9-a-X.jsonl
}
# ---------------------------------------------------------------- the profiled start (nc/dsm-ncprof-p.sh's START P, one 1u round)
PT0=T0; PT1=T1; PS=0; PDONE=0
pstart() {   # a second start: the same env + base + --profile-ops (RADIANCE_PROFILE_EVERY PEVERY); ONE 1u NST 5 essay round
  local tag=i9-p pid
  diskok $tag || return 1
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on --profile-ops" PROFEVERY=$PEVERY
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV > $XS2_ENVFILE"; else printf '%s\n' $LEVENV > $XS2_ENVFILE; fi
  xlog "$tag: start (PROFILED --profile-ops, RADIANCE_PROFILE_EVERY $PEVERY: not a speed, not comparable with the screen start) IMAGE=$IMAGE MODEL=$MODEL $(desc) | base $(armj B) | env: $(env | grep -E '^(RADIANCE_|R4D_)' | grep -vE '(KEY|TOKEN|SECRET)' | sort | tr '\n' ' ' | cut -c1-1600)"
  restart $tag || { keeplog $tag-fail; down; xlog "$tag: VIRHE not healthy -- no profile"; export PROFEVERY=; return 1; }
  run "mkdir -p $O/guard; (cd $D && timeout 600 python3 ds41_tests.py correct --only-q --effort none --outdir $O/guard > $O/guard/$tag-warm-q.txt 2>&1)"   # = check()'s warm-up line, no 16k / fastcheck
  modes $tag || { xlog "$tag: VIRHE the modes are not the start's -> no profile"; down; export PROFEVERY=; return 1; }
  drafted $tag || { down; export PROFEVERY=; return 1; }
  readback $tag || { keeplog $tag-readback; down; export PROFEVERY=; return 1; }
  arm B || { down; export PROFEVERY=; return 1; }
  if [ "$DRY" = 1 ]; then
    echo "+ (cd $D && timeout 600 python3 $SPB essay $PGEN > $O/p-round.txt) & -- T0 = its start + $PSKIP s (past the prefill step), T1 = its end"
  else
    PS=$(date +%s.%N)
    (cd $D && timeout 600 python3 $SPB essay $PGEN > $O/p-round.txt 2>&1) & pid=$!
    sleep $PSKIP; PT0=$(date +%s.%N); wait $pid; PT1=$(date +%s.%N)
    if awk -v a=$PT0 -v b=$PT1 'BEGIN{exit !(b - a < 1.0)}'; then   # the round ended before the skip: the window from its start (its prefill step inside)
      xlog "$tag round: shorter than the $PSKIP s skip -- the window runs from the request's start (its prefill step inside; the wall then comes from no single phase)"; PT0=$PS; fi
    if grep -q '^J ' $O/p-round.txt; then xlog "$tag round (stepprobe essay $PGEN, T 0, PROFILED: not a speed): $(grep -m1 '^J ' $O/p-round.txt | cut -c3- | cut -c1-400) | window $PT0 .. $PT1"
    else xlog "$tag round: VIRHE request error (no J line: $(tail -2 $O/p-round.txt | tr '\n' ' ' | cut -c1-200))"; fi
  fi
  run "docker stop -t 60 $CN > /dev/null 2>&1"        # the shutdown table closes the last interval (nc/dsm-ncprof-p.sh)
  capture $O/p-engine.log.gz
  down
  export PROFEVERY=
  PDONE=1
}
pparse() {   # the round's per-op split (ds_opprof_nc.py read only) -> RESULT lines: the drafting round's ms by class, ranked
  [ $PDONE = 1 ] || [ "$DRY" = 1 ] || return 0
  run "timeout 900 python3 -I -B $OPNC $O/p-engine.log.gz --win round:$PT0:$PT1 --every $PEVERY --label i9-p --json $O/p-prof.json > $O/p-prof.out 2>&1"
  [ "$DRY" = 1 ] && { echo "+ RESULT xi9b profile: ds_opprof_nc.py's DSOPNC line + per rank the drafting round's ms a step: drafter (dspark_ctx + dspark_query) / verify trunk / MoE (experts + moe_other) / attention (+ indexer) / mhc / collectives (+ gather_ar) / dense + norms + other / lane waits / host gaps, ranked (PROFILED: not a speed)"; return 0; }
  xlog "profile (PROFILED, not a speed): $(grep -m1 '^DSOPNC ' $O/p-prof.out | cut -c1-2500 || tail -2 $O/p-prof.out | tr '\n' ' ' | cut -c1-400)"
  while IFS= read -r l; do xlog "profile ${l:0:1500}"; done < <(python3 -c 'import json,sys
d = json.load(open(sys.argv[1])); w = (d.get("windows") or {}).get("round") or {}
rs = w.get("ranks") or {}
if not rs: print("round: VIRHE no profile window (%s)" % "; ".join(d.get("guard") or ["no table pair around the round"])); sys.exit(0)
G = [("drafter (ctx + query)", ("dspark_ctx", "dspark_query")), ("MoE", ("experts", "moe_other")), ("attention + indexer", ("attention", "indexer")),
     ("mhc", ("mhc",)), ("collectives", ("collective", "gather_ar")), ("dense + norms + other", ("dense_gemm", "embed_engram", "norm_elem", "other"))]
for rk in sorted(rs, key=lambda x: int(x)):
    s = rs[rk]; bc = s.get("by_class_ms") or {}
    parts = [(nm, sum(bc.get(k, 0.0) for k in ks)) for nm, ks in G]
    drafter = parts[0][1]; trunk = (s.get("device_ms") or 0.0) - drafter
    wall = s.get("wall_ms"); hr = s.get("host_remainder_ms")
    tot = wall if wall else (s.get("device_ms", 0.0) + s.get("wait_ms", 0.0))
    items = parts + [("lane (launch) waits", s.get("wait_ms") or 0.0)] + ([("host gaps", hr)] if hr is not None else [])
    items.sort(key=lambda kv: -kv[1])
    print("round rank %s (%s steps %s, rows %s): a drafting round (step) %s ms%s | verify trunk %.2f ms, drafter %.2f ms | ranked: %s | ctx passes a step %.2f" % (
        rk, w.get("steps"), w.get("mix"), w.get("rows"), ("%.2f" % wall) if wall else "-", " (wall, step log)" if wall else " (device + waits: no wall)", trunk, drafter,
        "; ".join("%s %.2f ms (%.0f %%)" % (nm, v, 100.0 * v / tot if tot else 0) for nm, v in items), s.get("ctx_calls_per_step") or 0.0))
if d.get("guard"): print("round: INFO parser notes: %s" % "; ".join(d["guard"]))' $O/p-prof.json 2>&1 | head -8)
}
summary() {   # the coordinator's line rules (12.11.): ONE line a lever -- verdict, numbers (mean delta X vs B %, B-B and X-X spreads), one sentence
  [ "$DRY" = 1 ] && { echo "+ RESULT xi9b VERDICT lines: one a lever on cold $SDEPTH TTFT (candidate = a consistent drop >= 2 %, noise = within +-1 %, else 'borderline -- park' with the number; 'path did not run, no reading' when its i9b counters stayed 0; a single odd block printed and named, never vetoing or crowning); N's second line on c8 (agg tok/s, newcomer step ms, the same rules); the T1 line; the identity line; the final summary -- all INFO"; return 0; }
  python3 - "$O" "$LEVERS" "$SDEPTH" > $O/summary.txt 2>&1 <<'PY'
import json, os, statistics, sys
o, levers, sdepth = sys.argv[1:4]
NAME = {"B": "B", "M": "M1 (mhc_fuse)", "L": "IX-LDSR2 (ix_lds)", "N": "IX-NI (ix_ni)", "X": "X"}
def rows(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except OSError: return []
def mean(v):
    v = [x for x in v if isinstance(x, (int, float))]
    return statistics.mean(v) if v else None
def spread(v):
    v = [x for x in v if isinstance(x, (int, float))]
    return 100.0 * (max(v) - min(v)) / statistics.mean(v) if len(v) >= 2 and statistics.mean(v) else None
def pc(a, b): return 100.0 * (a / b - 1) if a is not None and b else None
def f(x, n=2, pct=False): return "-" if x is None else ("%+.*f %%" % (n, x) if pct else "%.*f" % (n, x))
def fs(x): return "-" if x is None else "%.2f %%" % x
def judge(bs, xs, lower_better):
    """bs / xs: [(name, value)] in run order. -> (verdict, sentence, numbers)"""
    b = [v for _, v in bs]; x = [v for _, v in xs]
    d = pc(mean(x), mean(b)); sb, sx = spread(b), spread(x)
    if d is None: return "no reading", "no complete B / X block pair", d, sb, sx, None
    gain = -d if lower_better else d              # > 0 = better
    pairs = [(-pc(xv, bv) if lower_better else pc(xv, bv)) for (_, bv), (_, xv) in zip(bs, xs)]
    def cls(g, prs):
        if abs(g) <= 1.0: return "noise"
        if g >= 2.0 and prs and all(p > 0 for p in prs): return "candidate"
        if g <= -2.0 and prs and all(p < 0 for p in prs): return "worse -- not a candidate"
        return "borderline -- park (%+.2f %%)" % d
    v = cls(gain, pairs)
    # a single odd block: the one farthest from the median of all blocks, when it sits > 2 % off and > 2x the next one's distance
    allb = bs + xs; med = statistics.median([val for _, val in allb]) if allb else None
    odd = None
    if med and len(allb) >= 4:
        dev = sorted(((abs(val / med - 1) * 100, nm, val) for nm, val in allb), reverse=True)
        if dev[0][0] > 2.0 and dev[0][0] > 2 * dev[1][0]: odd = dev[0]
    note = ""
    if odd:
        bs2 = [p for p in bs if p[0] != odd[1]]; xs2 = [p for p in xs if p[0] != odd[1]]
        d2 = pc(mean([q for _, q in xs2]), mean([q for _, q in bs2]))
        g2 = (-d2 if lower_better else d2) if d2 is not None else None
        v2 = cls(g2, [(-pc(q, mean([w for _, w in bs2])) if lower_better else pc(q, mean([w for _, w in bs2]))) for _, q in xs2]) if g2 is not None else "no reading"
        note = "odd block %s %.3f (%.1f %% off the median %.3f); without it %s -> %s" % (odd[1], odd[2], odd[0], med, f(d2, 2, True), v2)
        if v2.split(" ")[0] != v.split(" ")[0]:
            v = "borderline -- park (%+.2f %%; one block decides: %s)" % (d, odd[1])
    return v, note, d, sb, sx, pairs
def sentence(v, d, sb, sx, pairs, what):
    if v.startswith("candidate"): return "%s drops %.2f %% with every X block better than its B partner (pairs %s), beyond the noise band." % (what, -d if d < 0 else d, "/".join("%+.2f" % p for p in pairs))
    if v.startswith("noise"): return "%s moves %+.2f %%, inside +-1 %% (spreads B-B %s, X-X %s): no effect seen at this size." % (what, d, fs(sb), fs(sx))
    if v.startswith("worse"): return "%s is consistently worse by %+.2f %% (pairs %s)." % (what, d, "/".join("%+.2f" % p for p in pairs))
    if v.startswith("borderline"): return "%s moves %+.2f %%: between the noise band and a clear 2 %% win, or not consistent across the pairs (%s) -- park it." % (what, d, "/".join("%+.2f" % p for p in (pairs or [])))
    return "no complete B / X pair."
out, vs = [], []
sc = rows(os.path.join(o, "screen.jsonl"))
for lv in levers:
    s = [r for r in sc if r.get("lever") == lv and r.get("status") == 200]
    fresh = [r for r in s if r.get("cached") == 0]
    xr = [r for r in s if r["arm"] == lv]
    ctr = {}
    for r in s:
        for k, val in (r.get("i9b") or {}).items():
            if isinstance(val, (int, float)) and k != "mhc_declared": ctr.setdefault(r["arm"], {}).setdefault(k, 0); ctr[r["arm"]][k] += val
    acted = "; ".join("%s %s" % (NAME[a].split(" ")[0], ", ".join("%s +%d" % (k, v_) for k, v_ in sorted(c.items()) if v_) or "nothing moved") for a, c in sorted(ctr.items()))
    has_i9 = any((r.get("i9b") or {}).get("mhc_declared") is not None for r in xr)   # i9bd: declared = None when GET has no i9b
    keys = {"M": ("mhc_pairs",), "L": ("ix_lds_launches",), "N": ("ix_ni_launches", "ni_runs")}[lv]
    ran = sum((ctr.get(lv) or {}).get(k, 0) for k in keys) > 0
    bs = [(r["name"], r["ttft"]) for r in fresh if r["arm"] == "B" and r.get("ttft")]
    xs = [(r["name"], r["ttft"]) for r in fresh if r["arm"] == lv and r.get("ttft")]
    cb = mean([r.get("nondecode_ms") for r in fresh if r["arm"] == "B"]); cx = mean([r.get("nondecode_ms") for r in fresh if r["arm"] == lv])
    if not ran:
        v, note, d, sb, sx, pairs = "path did not run, no reading", "", pc(mean([q for _, q in xs]), mean([q for _, q in bs])), spread([q for _, q in bs]), spread([q for _, q in xs]), None
        why = ("its i9b counters stayed 0 in every X block (%s): the lever's path never ran, so the TTFT difference says nothing about it." % (", ".join(keys))) if has_i9 else \
              "GET /admin/fn4 carried no i9b counters (not the int9b image, or a mock run): nothing shows the path ran."
    else:
        v, note, d, sb, sx, pairs = judge(bs, xs, True)
        why = sentence(v, d, sb, sx, pairs, "cold %s TTFT" % sdepth)
    l = "VERDICT %s: %s | cold %s TTFT X %s s vs B %s s = %s, block spreads B-B %s, X-X %s (blocks %s) | %s%s | INFO prefill chunk ms X %s vs B %s (%s); counters %s%s" % (
        NAME[lv], v, sdepth, f(mean([q for _, q in xs])), f(mean([q for _, q in bs])), f(d, 2, True), fs(sb), fs(sx),
        ", ".join("%s %.3f" % p for p in bs + xs) or "-", why, (" | " + note) if note else "", f(cx, 1), f(cb, 1), f(pc(cx, cb), 2, True), acted or "-",
        "" if len(fresh) == len(s) else " | INFO %d run(s) not cold (cached > 0) left out" % (len(s) - len(fresh)))
    out.append(l); vs.append("%s: %s (%s)" % (NAME[lv].split(" ")[0], v, f(d, 2, True)))
c8 = [r for r in rows(os.path.join(o, "c8.jsonl")) if r.get("errors") == 0 and (r.get("requests") or 0) > 0]
if c8 and "N" in levers:
    def mixj(nm):
        try: return json.load(open(os.path.join(o, "c8", "%s-mix.json" % nm)))
        except Exception: return None
    def nstep(nm):   # the newcomer step wall: c8mix's singles (the fixed schedule's only class), p50 (co-batched shown apart)
        m = mixj(nm)
        return (m.get("single") or {}).get("wall_p50") if m else None
    nr = [r for r in c8 if r["arm"] == "N"]; br = [r for r in c8 if r["arm"] == "B"]
    ran = sum(((r.get("i9b") or {}).get("ix_ni_launches") or 0) + ((r.get("i9b") or {}).get("ni_runs") or 0) for r in nr) > 0
    has_i9 = any((r.get("i9b") or {}).get("mhc_declared") is not None for r in nr)
    tool = "fixed schedule (harness ncmb2 / mbt2, 16 newcomer steps a block: a NEW reference, not comparable with old-loop c8)" if all(r.get("tool") == "ncmb2" for r in c8) else "the OLD loop (harness absent)"
    try: bb = [l.strip()[9:] for l in open(os.path.join(o, "c8", "bb-compare.out")) if l.startswith("C8MIXCMP ")][-1:]
    except OSError: bb = []
    match = bool(bb) and bb[0].rstrip().endswith("MATCH")
    # B-vs-B pairs mirror: B1 / N2 and B4 / N3 (order B N N B)
    bt = [(r["name"], r.get("agg_dec_tps")) for r in br if r.get("agg_dec_tps")]
    xt = [(r["name"], r.get("agg_dec_tps")) for r in reversed(nr) if r.get("agg_dec_tps")][::-1]
    bw = [(r["name"], nstep(r["name"])) for r in br if nstep(r["name"])]
    xw = [(r["name"], nstep(r["name"])) for r in nr if nstep(r["name"])]
    mixs = " ; ".join("%s %s" % (r["name"], (lambda m: "single %s (p50 %s ms) / co-batched %s %s, drafted %s" % ((m.get("single") or {}).get("n"), (m.get("single") or {}).get("wall_p50"),
                      (m.get("cobatched") or {}).get("n"), (m.get("cobatched") or {}).get("sizes"), m.get("drafted_decode")) if m else "no mix")(mixj(r["name"]))) for r in c8)
    t = judge(bt, xt, False); w = judge(bw, xw, True)
    nums = "agg N %s vs B %s tok/s = %s (B-B %s, N-N %s) | newcomer step wall p50 N %s vs B %s ms = %s (B-B %s, N-N %s)" % (
        f(mean([q for _, q in xt]), 1), f(mean([q for _, q in bt]), 1), f(t[2], 2, True), fs(t[3]), fs(t[4]),
        f(mean([q for _, q in xw]), 1), f(mean([q for _, q in bw]), 1), f(w[2], 2, True), fs(w[3]), fs(w[4]))
    if not ran:
        v = "path did not run, no reading"
        why = "its i9b counters stayed 0 in the N blocks." if has_i9 else "GET /admin/fn4 carried no i9b counters (not the int9b image, or a mock run)."
    elif not match:
        v = "INCONCLUSIVE (mix)"
        why = "the two B blocks' co-batch mixes do not MATCH (%s), so the N-vs-B delta may be the mix, not the lever (TESTING.md: a c8 delta under ~3.6 %% can be the mix alone)." % (bb[0][:200] if bb else "no B-vs-B compare")
    else:
        v = "agg tok/s %s, newcomer step %s" % (t[0], w[0])
        why = "%s %s" % (sentence(t[0], t[2], t[3], t[4], t[5], "c8 agg tok/s"), sentence(w[0], w[2], w[3], w[4], w[5], "The newcomer step"))
        if t[1]: why += " | " + t[1]
    out.append("VERDICT %s c8: %s | %s | %s | B-vs-B mix check: %s | c8 on the %s | INFO mixes: %s" % (NAME["N"], v, nums, why, (bb[0][:240] if bb else "none"), tool, mixs[:600]))
    vs.append("N c8: %s (%s)" % (v, f(t[2], 2, True)))
try:
    t1 = json.load(open(os.path.join(o, "t1.json")))
    out.append("T1 (INFO) draft_sample 0 base: essay T 1 %s tok/s, %s tok/step, acceptance %s | cross-start SD 85 (other starts): sdtemp 11.11 81.4 tok/s acc 0.720, stack5 S5T 82.3 -> %s / %s" % (
        t1.get("decode_tps"), t1.get("tokens_per_step"), t1.get("acceptance"), f(pc(t1.get("decode_tps"), 81.4), 1, True), f(pc(t1.get("decode_tps"), 82.3), 1, True)))
except Exception: out.append("T1: not run")
try: ident = [l.strip() for l in open(os.path.join(o, "identity.txt")) if l.strip()]
except OSError: ident = []
bad = [l for l in ident if not ("IDENTICAL" in l and "differ" not in l)]
il = ("IDENTICAL (%s)" % "; ".join(ident))[:600] if ident and not bad else ("NOT IDENTICAL: " + "; ".join(bad))[:900] if ident else "not run"
out.append("IDENTITY greedy set alone, all three ON (X) vs all OFF (B), text + token ids: %s" % il)
for l in out: print(l)
print("SUMMARY %s || identity: %s || %s" % ("; ".join(vs) or "-", "IDENTICAL" if ident and not bad else ("NOT IDENTICAL" if ident else "not run"), " || ".join(out)[:4500]))
PY
  while IFS= read -r l; do case "$l" in SUMMARY*) ;; *) xlog "${l:0:3000}" ;; esac; done < $O/summary.txt
  xlog "summary: $(grep -m1 '^SUMMARY ' $O/summary.txt | cut -c9- | cut -c1-6000)"
}
dstart() {   # P0, then the screen start
  local tag=i9-n5
  diskok $tag || return 1
  if ! p0sel; then
    if [ $SW = 1 ]; then xlog "P0: VIRHE a new kernel form's one-card selftest failed -> the item stops (no server start)"; return 1; fi
    xlog "P0: failed under XI9_SW=0 (a mock run: docker run --device is not emulated) -- the mock goes on"
  fi
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on" PROFEVERY=
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV > $XS2_ENVFILE"; else printf '%s\n' $LEVENV > $XS2_ENVFILE; fi
  xlog "$tag: start (SCREEN; $DESC) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $FIX + base | env: $(env | grep -E '^(RADIANCE_|R4D_)' | grep -vE '(KEY|TOKEN|SECRET)' | sort | tr '\n' ' ' | cut -c1-2400)"
  gate1 $tag 70 || { xlog "$tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  modes $tag || { xlog "$tag: VIRHE the modes are not the start's -> stopped"; down; return 1; }
  drafted $tag || { down; return 1; }
  readback $tag || { keeplog $tag-readback; down; return 1; }
  vramstart $tag
  swnote $tag || { xlog "$tag: VIRHE the switches are not as expected -> stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  stopfile || screen
  stopfile || { case $LEVERS in *N*) budok c8 $(( ${#C8ORDER} * 90 / 60 + 1 )) && c8s ;; esac; }
  stopfile || { [ $DOT1 = 1 ] && budok T1 1 && t1x; }
  stopfile || { [ $DOID = 1 ] && budok identity 3 && idcheck; }
  capture $O/$tag-engine-full.log.gz
  xlog "$tag: at the end: $(pskstat) | $(i9bnow) | $(docker logs --since 6h $CN 2>&1 | grep -cE '^E ') E lines | $(elapsed) min"
  down
  xlog "$tag: done"
}
trap 'down' EXIT
trap 'exit 143' TERM INT
# ---------------------------------------------------------------- main
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19))$([ $SW = 1 ] || echo " | VIRHE XI9_SW=0: NO mhc_fuse / ix_lds / ix_ni and no stage_full_min / stage_full_pick posted, nothing of them checked, P0 not fatal (a mock run only)") | P0 r4d_selftest [$P0MIN] on CARD 0 | SCREEN start: $DESC | levers [$LEVERS] cold $SDEPTH TTFT $SORDER a lever ($SNTOK tok) | N: c8 $C8ORDER (ncmb T 0, dur $C8DUR warm $C8WARM) + the newcomer step | T1 essay (draft_sample 0) $([ $DOT1 = 1 ] && echo on || echo off) | exactness X vs B greedy set $([ $DOID = 1 ] && echo on || echo off) | guards: read back + 0 request errors (counters INFO) | budget $BUDGET min | PROFILED start $([ $DOPROF = 1 ] && echo "on (every $PEVERY, one essay $PGEN round; total budget $TOTAL min)" || echo off) | no evict"
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
# int9b's own (VIRHE when missing; SKIP only under DRYRUN=1 + XI9_DRY_ALLOW_OLD=1, a dry run against an image without them)
IMGLITS9B="B|stage_full_pick, mhc_fuse, ix_lds, ix_ni, epoch_bump)
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
docker image inspect $IMAGE > /dev/null 2>&1 || { xlog "VIRHE image $IMAGE not on this host"; exit 1; }
IMGMISS=$( { echo "$IMGLITS"; echo "$IMGLITS9B" | sed 's/^/9/'; } | timeout 120 docker run --rm -i --network none --entrypoint sh $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels/libr4d.so; R=/opt/radiance/bin/r4d_selftest; n=0
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
    echo "+ image check: SKIP (image lacks int9b literals; XI9_DRY_ALLOW_OLD=1, dry run only): missing $(echo "$MISS9" | tr ';' '\n' | grep -c .) of $(echo "$IMGLITS9B" | grep -c .): $(echo "$MISS9" | cut -c1-700)"
  else
    xlog "VIRHE $IMAGE is not the int9b image -- missing: $(echo "$MISS9" | cut -c1-900)"; exit 1
  fi
fi
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)): all $(echo "$IMGLITS" | grep -c .) int9 / KVR / profiler literals present$([ -z "$MISS9" ] && echo " + all $(echo "$IMGLITS9B" | grep -c .) int9b literals (the registry tail mhc_fuse / ix_lds / ix_ni, GET i9b counters, RADIANCE_DS41_MHC_FUSE + the M1 declare line, the fused op in deepseek41.so + libr4d.so, r4d_selftest's mhc_pcn_mix / ix_lds / ix_ni, int9b-PROVENANCE)") + bin/rad-sidecar + int7 / int8 / int9 PROVENANCE -- OK"
HMISS=""
for f in "$LIB29" "$LIBPR" $Y $CTR $GS $LD $PRB $NCMB $OPNC $DQ $MBP $TESTS $SPB $D/fastcheck.py; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $CTR $GS $LD $PRB $NCMB $OPNC $DQ $MBP $TESTS $SPB $D/fastcheck.py 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
CTT_=$(timeout 60 python3 -I -B $CTR selftest 2>&1 | tail -1); case "$CTT_" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$CTR(${CTT_:0:300})" ;; esac
ST=$(timeout 60 python3 -I $LD selftest --gs $GS --mb $MBP 2>&1 | tail -1); case "$ST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$LD(${ST:0:300})" ;; esac
NMT=$(timeout 60 python3 -I -B $NCMB --mb $MBP --selftest 2>&1 | tail -1); case "$NMT" in *OK*) ;; *) HMISS="$HMISS selftest:$NCMB(${NMT:0:300})" ;; esac
OPT=$(timeout 120 python3 -I -B $OPNC selftest 2>&1 | tail -1); case "$OPT" in *OK*) ;; *) HMISS="$HMISS selftest:$OPNC(${OPT:0:300})" ;; esac
if [ -s $NCMB2 ] && [ -s $MBT2 ]; then   # the harness (present: it runs the c8 windows; absent at run time: ncmb.py, logged)
  M2T=$(timeout 60 python3 -I -B $MBT2 --mb $MBP --selftest 2>&1 | tail -1); case "$M2T" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$MBT2(${M2T:0:300})" ;; esac
  N2T=$(timeout 60 python3 -I -B $NCMB2 --mb $MBP --selftest 2>&1 | tail -1); case "$N2T" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$NCMB2(${N2T:0:300})" ;; esac
  CMT=$(timeout 60 python3 -I -B $C8MIX selftest 2>&1 | tail -1); case "$CMT" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$C8MIX(${CMT:0:300})" ;; esac
else M2T="absent"; N2T="absent"; CMT="absent"; fi
(cd $D && python3 -I -B $PRB --help > /dev/null 2>&1) || HMISS="$HMISS run:sdprobe.py--help"
grep -q '"essay"' $TESTS || HMISS="$HMISS stepprobe-prompts:$TESTS"
FNS="log run mkd restart check down stopfile best_vars desc serve_args xlog elapsed armj NAMEOF TARGET pinj stackenv keeplog capture thermline dfree diskok cmpfile idrec identw modes vramstart fn4get fn4set nudge fn4code pcreset arm msnap mdelta pskstat ssnap sdelta gate1 alone cmpx budok ctrinfo i9bnow i9bd gpurun p0sel drafted readback swnote coldx screen c8x c8s t1x idcheck pstart pparse summary dstart"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee sha256sum gzip zcat; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
for a in B M L N X; do python3 -c 'import json,sys; d=json.loads(sys.argv[1]); sw = sys.argv[2] == "1"; assert d["ds41_psk"] == 4 and d["ds41_psk_cr"] == 1 and d["draft_sample"] == 0 and (all(k in d for k in ("mhc_fuse", "ix_lds", "ix_ni", "stage_full_min", "stage_full_pick")) if sw else not any(k in d for k in ("mhc_fuse", "ix_lds", "ix_ni", "stage_full_min", "stage_full_pick")))' "$(pinj $a)" $SW 2>/dev/null || HMISS="$HMISS json:$a"; done
[ "$(sha256sum $MODELDIR/$SIDECAR 2>/dev/null | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256 2>/dev/null)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR"
[ "$(sha256sum $MODELDIR/$SIDE 2>/dev/null | cut -c1-64)" = "$SIDESHA" ] || HMISS="$HMISS sha256:$MODELDIR/$SIDE(want ${SIDESHA:0:16})"
for w in XS2_ENVFILE 'RADIANCE_PROFILE_EVERY: ${PROFEVERY'; do grep -qF "$w" $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no '$w')"; done
grep -qE 'RADIANCE_DS41_KV_REPLICATED|RADIANCE_DS41_MHC_FUSE|RADIANCE_DS41_IX_' $Y 2>/dev/null && HMISS="$HMISS yaml:$Y(names a lever word: they come from the env file only)"
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config refuses it)"
case "$LEVENV" in *RADIANCE_DS41_KV_REPLICATED=1*) ;; *) HMISS="$HMISS levenv:no-KVR" ;; esac
case "$LEVENV" in *RADIANCE_DS41_MHC_FUSE=0*) ;; *) HMISS="$HMISS levenv:no-M1-declare" ;; esac
HMISS=$(echo $HMISS)
[ -z "$HMISS" ] || { xlog "VIRHE helpers / inputs do not resolve: $HMISS"; exit 1; }
HMD5="i9bctr.py $(md5sum < $CTR | cut -c1-8) (copy of stack5/s5ctr.py $(md5sum < $HD/../stack5/s5ctr.py | cut -c1-8) + int9b families; INFO only), stack4/loadid.py $(md5sum < $LD | cut -c1-8), stack4/sdprobe.py $(md5sum < $PRB | cut -c1-8), nc/ncmb.py $(md5sum < $NCMB | cut -c1-8), nc/ds_opprof_nc.py $(md5sum < $OPNC | cut -c1-8), harness mbt2.py $(md5sum < $MBT2 2>/dev/null | cut -c1-8) / ncmb2.py $(md5sum < $NCMB2 2>/dev/null | cut -c1-8) / c8mix.py $(md5sum < $C8MIX 2>/dev/null | cut -c1-8), deepq.py $(md5sum < $DQ | cut -c1-8), mb.py $(md5sum < $MBP | cut -c1-8), stepprobe.py $(md5sum < $SPB | cut -c1-8), greedyset.py $(md5sum < $GS | cut -c1-8), ds41_tests.py $(md5sum < $TESTS | cut -c1-8), dsm-i9b.sh $(md5sum < $0 | cut -c1-8) (from stack5/dsm-stack5.sh $(md5sum < $HD/../stack5/dsm-stack5.sh | cut -c1-8)), lib-29.sh $(md5sum < $LIB29 | cut -c1-8), yaml $(md5sum < $Y | cut -c1-8)"
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing; i9bctr.py $CTT_ | loadid.py $ST | ncmb.py $NMT | ds_opprof_nc.py $OPT | sdprobe.py --help OK | harness: mbt2.py $M2T / ncmb2.py $N2T / c8mix.py $CMT | all $(echo $FNS | wc -w) functions resolve, tools on PATH, every arm's pin (S5T, draft_sample 0$([ $SW = 1 ] && echo ", SP, int9b" || echo "; XI9_SW=0: no int9 / int9b switch")), sidecars $SIDECAR + $SIDE sha256 OK, yaml accepted (env_file, RADIANCE_PROFILE_EVERY, no lever word), LEVENV with KVR=1 + MHC_FUSE=0 -- OK | c8 windows: $([ -s $NCMB2 ] && [ -s $MBT2 ] && echo "harness/ncmb2.py + mbt2.py (fixed schedule)" || echo "nc/ncmb.py (harness absent)") | md5: $HMD5"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { xlog "VIRHE $M225 missing or not verified"; exit 1; }
  [ -s $WARM_REF_225 ] || xlog "note -- reference $WARM_REF_225 missing (not compared)"
  xlog "$ITEM: helpers md5: $HMD5"
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDE; do [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  for f in $WARM_REF_225 $RS5/s5-n5-stats0.json $RS5/guard-s5-n5.txt; do [ -e $f ] && echo "+ present (reference): $f" || echo "+ absent (reference, information only): $f"; done
  for a in B M L N X; do echo "+ arm $a ($(NAMEOF $a)): $(armj $a)$( case $a in M|L|N) echo " | target: $(TARGET $a)" ;; esac)"; done
  echo "+ base pin + fixed pin: $(pinj B)"
  echo "+ screen: levers [$LEVERS] cold $SDEPTH ($SNTOK tok) $SORDER a lever | c8 $C8ORDER (dur $C8DUR warm $C8WARM gens $C8GENS, T 0) | T1 $DOT1 | identity $DOID | profiled start $DOPROF (every $PEVERY, essay $PGEN, skip $PSKIP s)"
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(information here; the real run refuses to start beside one)"
  echo "+ disk now: $(dfree) MB free on / (guard $DISKMIN MB)"
fi
mkd $O $O/guard $O/c8 $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt
# dsm-i9p.sh (12.11. ~12:15, ds-main-port): a COPY of dsm-i9b.sh (unchanged) that runs ONLY the profiled start (pstart + pparse):
# i9b 11:59-12:12 ran the screen; its profiled start failed in 0 s on the ghost compose container 622b5b7a9e64 (before compose-guard).
[ "$DRY" = 1 ] || xlog "$ITEM: PROFILED START ONLY (screen done in i9b 11:59-12:12; this re-runs its failed pstart)"
if [ $DOPROF = 1 ]; then stopfile || { budok "profiled start" 7 $TOTAL && pstart; }; pparse; fi
unset $SWITCHES
down
xlog "$ITEM: done ($(elapsed) min) -- $O | $([ $NV = 0 ] && echo "no VIRHE line" || echo "$NV VIRHE line(s): see above")"
