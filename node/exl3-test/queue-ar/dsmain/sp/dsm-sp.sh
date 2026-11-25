#!/bin/bash
# dsm-sp.sh (12.11.2026, queue-ar/dsmain/sp, ds-main-port int9 builder) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrz-sp.sh. Log tag "RESULT xsp ", out runs/xdsm/sp (XSP_OUT). Made from COPIES (the originals are
# not changed): stack5/dsm-stack5.sh 63649414 (the start, LEVENV / LITS, the fixed pin and S5T arm, stackenv .. gate1, deepx; tag
# xs5 -> xsp), stack5/s5ctr.py 256dfd9a -> spctr.py (+ lever SP's families spfull / sppick / spstream and the light PSK form
# pskl), sahb/ncone.py a24570f8 -> ncone.py (byte for byte: one-token ALONE prompts at --ctx, /stats between them; sahb's
# screen), pskcr/pcr3.py ac7e1658 -> spchain.py (+ the tchain spec: a CC-like turn chain whose every new turn adds a tool-sized
# message). + spsum.py (the cell readings, the need rule, the screen's pick, the summary). Read only beside it: pskcr/pskld.py,
# deepq.py, mb.py, ds41_tests.py.
# LEVER SP on the ds41-int9 image (ds41-int8 7e03384 + ds41-kvr d5a0c60 + lever SP; fn4 stage_full_min / stage_full_pick appended
# after place_hb; every new switch unset and RADIANCE_DS41_KV_REPLICATED unset = the int8 image: graph and start log identical,
# build/int9 hostcheck). A full-plan step (2113 .. 4160 rows: a new turn's tail, a lone prompt, a long prompt's 4096-row chunk)
# arms the prefill stager, which copies EVERY pooled unit (~12.7 GB a rank); scr3's 2.4-2.6k-row tails took ~+0.2 s over
# streaming. SP: stage_full_min N streams full-plan steps below N rows (in-kernel, as every smaller step); stage_full_pick K copies
# only the units the last K stager-size steps routed. Exact (placement only: the same bytes from the ring or the pool, the same
# kernel).
# ONE NST 5 START = stack5's (runs/ds41-exl3.log "RESULT xs5 s5-n5: start": stack4's + RADIANCE_LOG_STEPS=1 + RADIANCE_DS41_ENGDD=0
# + RADIANCE_PLACE_HB_RANKS=1,2), KVR env OFF (RADIANCE_DS41_KV_REPLICATED unset; modes() checks no "lever KVR declared" line):
# ONLY SP is measured. THE DS BASE PIN = stack5's S5T: the fixed pin (stack4's + ds41_psk_k 8) + sched_bi_chunk 3 (D3) + sched_noa 2
# + ds41_psk 4 (tail-only) k 8 + ds41_psk_cr 1 + ds41_psk_l20 1 + dspark_pfctx 1 + ds41_engdd 1 + draft_sample 85 + draft_occ 6 +
# ds41_bi 2 (stage_short / stage_pick / place_hb 0). ARMS in process on top (POST + epoch_bump, a nudge, READ BACK; every counter
# baseline AFTER the read-back): B = SP off (stage_full_min 0, stage_full_pick 0) | S = SP-S, stream (stage_full_min 4161, above
# every full-plan step) | 1 / 2 = SP-P, pick (stage_full_pick 1 / 2) | X = SP-best, the screen's composite (stage_full_min N*,
# stage_full_pick K*) | P = the faster pick at the largest screen size (the cold 64k's pick arm).
# READINGS (decode-free; every prompt fresh, judged on cached == 0 only):
#   (a) STAGER SCREEN: per size XSP_SIZES (2.3k / 3k / 3.9k: ncone --ctx 2200 / 2900 / 3800 -> ~2.3k / 3.0k / 3.9k-token prompts,
#       all 2113 .. 4160 = ONE full-plan step) the cells XSP_SORDER (B S 1 2 2 1 S B) of XSP_SCN (4) fresh one-token alone prompts
#       -> 8 prompts an arm a size: prefill step ms (/stats step_ns between the prompts), reply s, staged GB (all ranks / rank 0),
#       streamed GB per rank, GET "sp" per rank; the faster SP form per size and the composite pin (spsum.py pick).
#   (b) ALONE TAILS: per size XSP_TSIZES (1.5k / 2.5k / 3.5k / 5k: --ctx 1400 / 2400 / 3400 / 4900) the cells XSP_TORDER (B X X B),
#       XSP_SCN prompts each: B vs SP-best (1.5k: below the full plan, a control -- no SP counter may move; 5k: a 4096 chunk + a
#       short rest).
#   (c) TURN CHAIN: XSP_CORDER (B X X B) x one CC-like 4-turn chain, the SAME text and nonce each (spchain.py tchain): turn 1 cold
#       ~XSP_CDEPTH (16000) tokens after a prefix-cache reset, each later turn = the previous + its answer (XSP_CNTOK 64 greedy) + a
#       follow-up carrying XSP_CTT (2300) tokens of tool text -> new-turn tails over 2112 rows; TTFT, cached and tail a turn.
#   (d) COLD 64k: deepq's fixed prompt at XSP_DDEPTH (64000), cold after a reset, 512 greedy tokens, XSP_DORDER (B S P P S B):
#       TTFT and the mean prefill chunk ms (/metrics non-decode steps), staged GB -- the coordinator's question 3 (lone 4096-row
#       chunks arm the stager too: streaming every chunk (S) or picking (P) vs copying every pooled unit (B)).
#   (e) C8 CELL (coordinator 12.11., from KV4: at c8 under NOA 2 co-batched newcomer steps of 2162-3290 rows ARM the full-plan
#       stager -- staged 20-23 GB + streamed 47 GB, 996-1380 ms a step): mb c8 at temperature 0 (stack4/mbt.py, read only), dur
#       XSP_C8DUR 60 warm XSP_C8WARM 15, XSP_C8ORDER (B X X B: twice an arm), B vs SP-best -> agg tok/s, decode_share (mb's MB
#       line), the co-batched full-plan steps' wall (MIXED / prefill of 2113 .. 4160 rows in the window, spsum.py steps on the step
#       log), staged GB and the window's GET "sp" counters; guard: forbid only (whether a full-plan step happens is the load's).
#   IDENTITY, strict: every SP answer vs B -- the chains turn by turn (each X chain and the second B vs the first B), the cold 64k
#       answers vs the first B, and an alone set: deep:DEPTH:OFF (spchain.py, 64 greedy tokens, a reset before each) at XSP_IDEPTHS
#       (2200 3000 3900 5000) under B, S, P, X.
#   COUNTER GUARD (spctr.py) on every reading: cells -- pskl (latched, tail_only_latched grow; plan_mismatch, tail_only_refused
#       +0) + the SP families the cell's own prompt sizes call for under its switches (spsum.py need: B spfull, S spstream, P
#       spfull + sppick; forbidden where they must not move; 1.5k nothing of SP); chains -- psk, tail, l20, pfctx, engdd, crseen
#       (stack5's chain:T) + the arm's SP families; cold 64k -- psk, tail, l20, pfctx, engdd (stack5's cold:T) + the arm's. PSK on
#       the corrected counters (never layers_skipped).
# ~32 min estimate (start + checks ~7, switch note ~1, warm-up + screen 3 x 8 cells x 4 prompts ~6, tails 4 x 4 x 4 ~4, c8 4 x
#   ~80 s ~6, chains 4 x 4 turns ~2, cold 64k 6 x ~25 s ~3, identity 16 requests ~2, parsing ~1); BUDGET XSP_BUDGET_MIN 40: a section that would start
#   past it is skipped (logged). NO evict, no cc_sim, no power / host / fan settings; production compose files read only.
# Knobs: XSP_IMAGE, XSP_ITEM, XSP_OUT, XSP_SIZES ("2.3k:2200 3k:2900 3.9k:3800"), XSP_SORDER (BS1221SB), XSP_SCN (4), XSP_TSIZES
#   ("1.5k:1400 2.5k:2400 3.5k:3400 5k:4900"), XSP_TORDER (BXXB), XSP_CORDER (BXXB), XSP_CDEPTH (16000), XSP_CTT (2300), XSP_CNTOK
#   (64), XSP_DDEPTH (64000), XSP_DORDER (BSPPSB), XSP_IDEPTHS ("2200 3000 3900 5000"), XSP_SFULL (4161), XSP_DISKMIN_MB (11000),
#   XSP_BUDGET_MIN (40), XSP_C8ORDER (BXXB), XSP_C8DUR (60), XSP_C8WARM (15), XSP_SIDECAR, XSP_SIDESHA.
# DRYRUN=1 bash dsm-sp.sh : prints every command; checks the image literals (int9: SP's registry names, range messages, counters,
#   the unknown-switch tail, KVR's env / declare literals, int9-PROVENANCE), every helper (files, parse, selftests: spctr.py /
#   spsum.py / ncone.py / spchain.py selftest + tselftest / stack4's mbt.py), the sidecars, the yaml, the disk; no server, no GPU, writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/../ds41-stack2-tp4.yaml                   # stack2's yaml, read only (env_file ${XS2_ENVFILE} carries every start word)
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh
O=${XSP_OUT:-$R/xdsm/sp}; export STATSDIR=$O/stats
RD=$R/xdsm/draft; RM=$R/xdsm; RS4=$R/xdsm/stack4; RS5=$R/xdsm/stack5
export IMAGE=${XSP_IMAGE:?set XSP_IMAGE}
ITEM=${XSP_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-x-kvrz-sp}
PSKK=8; BIC=3; NOA=2; OCC=6; OCCK=32; DSA=85; HBR=0; HBRANKS=1,2      # stack5's S5T values (XS5_* defaults)
SIZES=${XSP_SIZES:-2.3k:2200 3k:2900 3.9k:3800}; SORDER=${XSP_SORDER:-BS1221SB}; SCN=${XSP_SCN:-4}
TSIZES=${XSP_TSIZES:-1.5k:1400 2.5k:2400 3.5k:3400 5k:4900}; TORDER=${XSP_TORDER:-BXXB}
CORDER=${XSP_CORDER:-BXXB}; CDEPTH=${XSP_CDEPTH:-16000}; CTT=${XSP_CTT:-2300}; CNTOK=${XSP_CNTOK:-64}
DDEPTH=${XSP_DDEPTH:-64000}; DORDER=${XSP_DORDER:-BSPPSB}; IDEPTHS=${XSP_IDEPTHS:-2200 3000 3900 5000}
SFULL=${XSP_SFULL:-4161}; DISKMIN=${XSP_DISKMIN_MB:-11000}; BUDGET=${XSP_BUDGET_MIN:-40}
C8ORDER=${XSP_C8ORDER:-BXXB}; C8DUR=${XSP_C8DUR:-60}; C8WARM=${XSP_C8WARM:-15}
SIDE=${XSP_SIDECAR:-dspark-dlw-q8_0.rad}; SIDESHA=${XSP_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
NTOK=512; COFF=500; TOFF=540; IOFF=440    # stdlib file offsets (gpu02 has 624 files: every first block distinct, checked)
SPN=$SFULL; SPK=1; KP=1                 # the composite and the pick arm until the screen decides (spsum.py pick)
for v in $SCN $CDEPTH $CTT $CNTOK $DDEPTH $SFULL $DISKMIN $BUDGET $IDEPTHS $C8DUR $C8WARM; do case $v in ''|*[!0-9]*) echo "a numeric knob is not a number ('$v')"; exit 1 ;; esac; done
for z in $SIZES $TSIZES; do case $z in *:*[!0-9]*|*:|:*) echo "a size is LABEL:CTX (got '$z')"; exit 1 ;; esac; done
[ $C8WARM -lt $C8DUR ] || { echo "the c8 warm-up is not shorter than the phase"; exit 1; }
case $C8ORDER in *[!BX]*) echo "XSP_C8ORDER takes B and X only ('$C8ORDER')"; exit 1 ;; esac
for o in "$SORDER" "$TORDER" "$CORDER" "$DORDER"; do case $o in *[!BS12XP]*|"") echo "an order names an arm outside B S 1 2 X P ('$o')"; exit 1 ;; esac; done
[ $SFULL -gt 4160 ] && [ $SFULL -le 1048576 ] || { echo "XSP_SFULL must be above the full plan (4160) and <= 1048576: SP-S streams every full-plan step"; exit 1; }
[ $SCN -ge 2 ] && [ $SCN -le 16 ] || { echo "XSP_SCN 2..16"; exit 1; }
DQ=$QA/deepq.py; MBP=$D/mb.py; TESTS=$D/ds41_tests.py
CTR=$HD/spctr.py; SUM=$HD/spsum.py; NCONE=$HD/ncone.py; CHN=$HD/spchain.py                        # this item's
PLC=$HD/../pskcr/pskld.py                                                                          # pskcr's, read only
MBT=$HD/../stack4/mbt.py                                                                           # stack4's, read only
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
T00=$(date +%s); NV=0
xlog() { case "$*" in *VIRHE*) NV=$((NV + 1)) ;; esac; log "RESULT xsp $*"; }
elapsed() { echo $(( ($(date +%s) - T00) / 60 )); }
# ---------------------------------------------------------------- the start (= stack4's + RADIANCE_LOG_STEPS=1 + the lever declares)
GPTAB=q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422
SIDECAR=dspark-conf.f32
LEVENV="RADIANCE_DS41_GG_PF=1 RADIANCE_DS41_GG_PF_TABLE=$GPTAB RADIANCE_DS41_F32_PF=25 RADIANCE_DS41_ARENA_DECODE_ROWS=8,48
  RADIANCE_DS41_ROUTER_BF16=1 RADIANCE_DS41_EMB_RAM=1 RADIANCE_PLACE_SHADOW_SLOTS=16 RADIANCE_XPREFETCH=0 RADIANCE_DS41_DSPARK_WALK=1
  RADIANCE_DS41_ARGATHER_MAX=48 RADIANCE_DS41_QNFOLD=1 RADIANCE_DS41_SWFOLD=1 RADIANCE_DRAFT_CTXSKIP=0 RADIANCE_DS41_DSPARK_CONF=/models/$SIDECAR
  RADIANCE_DRAFT_LEND=0 RADIANCE_DRAFT_PROBE_BACKOFF=0 RADIANCE_DRAFT_LEND_BORROW=0 RADIANCE_DRAFT_SAMPLE=0 RADIANCE_WEIGHT_SIDECAR=/models/$SIDE
  RADIANCE_DS41_BI=0 RADIANCE_DRAFT_CTXSKIP_SLOTS=129 RADIANCE_DRAFT_LEND_WEIGHTS=0 RADIANCE_DS41_PSK=0
  RADIANCE_LOG_STEPS=1 RADIANCE_DS41_ENGDD=0 RADIANCE_PLACE_HB_RANKS=$HBRANKS"
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
# + this item's levers' declare lines (the lever items' own literals: e1/dsm-e1.sh, pskl20/dsm-pskl20.sh, pskcr/dsm-pskcr2.sh, sahb/dsm-sahb.sh)
LITS="$LITS|lever,E1,declared,(RADIANCE_DS41_ENGDD|now,0):,the,Engram,lookup,/,copy,/,kv,GEMM,(25600,x,6144)|rows,at,2,Engram,layer(s)"
LITS="$LITS|now,0;,RADIANCE_DS41_PSK_L20):,layer,20,split,under,lever,PSK"
LITS="$LITS|now,0;,RADIANCE_DS41_PSK_CR):,a,planned,prefix,hit,with,no,snapshot,at,its,grid,point"
LITS="$LITS|lever,SA,/,HB,(ds41-sahb),set,up|stage_short,can,arm,the,prefill,stager,for,a,step,of,1025,..,2112,tokens|place_hb,can,reserve,on,rank,1"
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": 8, "drafter_mx": 1, "gg_pf": 1, "f32_pf": 25, "dspark_walk": 1, "ds41_argather": 48, "idx_split": 1, "ds41_qnfold": 1, "ds41_swfold": 1, "draft_ctxskip": 1, "dspark_conf": 1, "dspark_conf_rows": 1, "draft_lend": 1, "draft_probe_backoff": 3, "draft_hybrid": 1}'
FIX=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update({"draft_lend_hyst": 0, "arena_d0": 0, "draft_lend_w": 0, "draft_occ_k": int(sys.argv[2]), "ds41_psk_k": int(sys.argv[3])}); print(json.dumps(d))' "$FIX5" "$OCCK" "$PSKK") || { echo "pin JSON"; exit 1; }
NEW0='"sched_bi_chunk": 0, "sched_noa": 0, "ds41_psk_l20": 0, "dspark_pfctx": 0, "ds41_engdd": 0, "ds41_psk_cr": 0, "stage_short": 0, "stage_pick": 0, "place_hb": 0'
S5J="\"ds41_bi\": 2, \"draft_sample\": $DSA, \"draft_occ\": $OCC, \"place_lf\": 0, \"sched_bi_chunk\": $BIC, \"sched_noa\": $NOA, \"ds41_psk_l20\": 1, \"dspark_pfctx\": 1, \"ds41_engdd\": 1"
# ---------------------------------------------------------------- the DS base pin (stack5's S5T) and lever SP's arms
TJ="{$S5J, \"ds41_psk\": 4, \"ds41_psk_cr\": 1, \"stage_short\": 0, \"stage_pick\": 0, \"place_hb\": 0}"   # = stack5's armj T
armj() {   # armj B|S|1|2|X|P: the arm's lever SP switches (JSON) on top of S5T
  case $1 in
    B) echo '{"stage_full_min": 0, "stage_full_pick": 0}' ;;
    S) echo "{\"stage_full_min\": $SFULL, \"stage_full_pick\": 0}" ;;
    1|2) echo "{\"stage_full_min\": 0, \"stage_full_pick\": $1}" ;;
    X) echo "{\"stage_full_min\": $SPN, \"stage_full_pick\": $SPK}" ;;
    P) echo "{\"stage_full_min\": 0, \"stage_full_pick\": $KP}" ;;
  esac
}
swof() { case $1 in B) echo 0,0 ;; S) echo $SFULL,0 ;; 1|2) echo 0,$1 ;; X) echo $SPN,$SPK ;; P) echo 0,$KP ;; esac; }
NAMEOF() { case $1 in B) echo B ;; S) echo SP-S ;; 1) echo SP-P1 ;; 2) echo SP-P2 ;; X) echo SP-best ;; P) echo SP-P ;; esac; }
pinj() {   # pinj ARM [EXTRAJSON]: the fixed pin + S5T + the arm + extra as one POST body
  local x=${2:-}; [ -n "$x" ] || x='{}'
  python3 -c 'import json,sys
d = {}
for a in sys.argv[1:]: d.update(json.loads(a))
print(json.dumps(d))' "$FIX" "$TJ" "$(armj $1)" "$x"
}
spfam() {   # spfam ARM -> lever SP's "need/forbid" for a reading whose full-plan steps are not all known in size (chains, cold 64k):
            # every such reading has 4096-row chunks (turn 1 at 16k, the 64k) and maybe tails of 2113 .. 4160 rows
  local n k
  case $1 in
    B) echo "spfull/sppick,spstream" ;;
    S) echo "spstream/spfull,sppick" ;;
    1|2|P) echo "spfull,sppick/spstream" ;;
    X) n=$SPN; k=$SPK
       if [ $n -gt 4160 ]; then echo "spstream/spfull,sppick"
       elif [ $n = 0 ]; then [ $k -gt 0 ] && echo "spfull,sppick/spstream" || echo "spfull/sppick,spstream"
       elif [ $n -le 4096 ]; then [ $k -gt 0 ] && echo "spfull,sppick/" || echo "spfull/sppick"   # 4096-row chunks arm; tails below N may stream
       else [ $k -gt 0 ] && echo "/" || echo "/sppick"; fi ;;
  esac
}
needs() {   # needs cold|chain ARM -> "need/forbid" for spctr.py: stack5's cold:T / chain:T + the arm's lever SP families
  local f b n x
  f=$(spfam $2)
  case $1 in cold) b="psk,tail,l20,pfctx,engdd/sa,hb" ;; chain) b="psk,tail,l20,pfctx,engdd,crseen/sa,hb" ;; esac
  n="${b%%/*},${f%%/*}"; x="${b#*/},${f#*/}"
  echo "${n%,}/${x%,}"
}
DESC="NST 5 stack5 start (stack4's + RADIANCE_LOG_STEPS=1, E1 declared off, HB ranks $HBRANKS; KVR env OFF) | DS base pin = stack5's S5T $TJ + the fixed pin (ds41_psk_k $PSKK) | arms B $(armj B), SP-S $(armj S), SP-P1 / SP-P2 stage_full_pick 1 / 2, SP-best = the screen's composite, SP-P = the faster pick at the largest size"
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
  RADIANCE_SCHED_BI_GRID RADIANCE_PROFILE_EVERY RADIANCE_DS41_KV_REPLICATED RADIANCE_DEBUG_TIER_VERIFY"   # + KVR (ds41-int9): unset = OFF
# ---------------------------------------------------------------- copied from stack5/dsm-stack5.sh (63649414; tag xs5 -> xsp): stackenv .. gate1
# (vramstart / gate1 show stack5's s5-n5 start beside this one; ctrg: spctr.py with an explicit need / forbid)
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xsp $1 warm-up identity vs $2"; return 0; }
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
  [ "$NW" -gt 0 ] && [ "$NR" = 0 ] || { xlog "$1: VIRHE rule-w is not what the recorded passes play ($NW / $NR)"; ok=0; }
  KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-500)
  case "$KL" in *"ds41_kvfold=1"*"ds41_ofold=1"*"ds41_onefill=1"*) ;; *) xlog "$1: VIRHE the K1 switches do not start at 1: '${KL}'"; ok=0 ;; esac
  AL=$(docker logs $CN 2>&1 | grep -m1 'the MoE gather and its all-reduce as one kernel' | sed 's/^radiance: //' | cut -c1-300)
  case "$AL" in *"up to 48 tokens a step (RADIANCE_DS41_ARGATHER, RADIANCE_DS41_ARGATHER_MAX"*) ;; *) xlog "$1: VIRHE the ARGATHER note is not the 48-row one: '${AL}'"; ok=0 ;; esac
  [ "$(docker logs $CN 2>&1 | grep -cE ' step [0-9]+ (prefill|decode|MIXED) n_seq=')" -gt 0 ] || { xlog "$1: VIRHE RADIANCE_LOG_STEPS=1 but no step line in the log"; ok=0; }
  IFS='|' read -ra lit <<< "$LITS"
  for l in "${lit[@]}"; do [ -n "$l" ] || continue
    if docker logs $CN 2>&1 | grep -qF -- "${l//,/ }"; then found="$found ${l%%,*}"; else miss="$miss | ${l//,/ }"; fi; done
  xlog "$1 modes: ${#lit[@]} literal(s), found [$(echo $found)]$([ -n "$miss" ] && echo " -- VIRHE missing:$miss") | $(docker logs $CN 2>&1 | grep -E "lever PSK|lever E1|lever SA / HB|lever BI declared|lever SD|weight sidecar|lever Z0|activation arena|ring of|draft lend:|draft weight lend|no level for|prefill staging:" | sed 's/^radiance: //; s/^[A-Z] *//; s/^[A-Z]\[[0-9]\] //' | cut -c1-260 | sort -u | tr '\n' '|' | cut -c1-3600)"
  [ -z "$miss" ] || ok=0
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
vramstart() {
  [ "$DRY" = 1 ] && { echo "+ GET /stats > $O/$1-stats0.json; RESULT xsp $1 VRAM at start (experts per tier; vs stack4 s4-n5)"; return 0; }
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
ctrg() {   # ctrg NEED/FORBID A B: the counter guard (spctr.py) between snapshots A and B -> "OK | ..." / "FAIL ... | ..."
  local s=$1
  [ "$DRY" = 1 ] && { echo "OK (dry: $CTR guard --need ${s%%/*} --forbid ${s#*/})"; return 0; }
  timeout 60 python3 -I -B $CTR guard --need "${s%%/*}" --forbid "${s#*/}" --st0 $O/st-$2.json --st1 $O/st-$3.json --fn0 $O/fn-$2.json --fn1 $O/fn-$3.json \
    --hb 0 --hb-ranks $HBRANKS 2>&1 | tail -1 | cut -c1-1200
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
deepx() {   # deepx TAG ARM DEPTH NAME [noreset]: deepq.py's fixed prompt at DEPTH, 512 greedy tokens -> TTFT, cached, acceptance, ms, the cold guard
            # (copied from stack5's deepx; the guard = stack5's cold:T + the arm's lever SP families, needs cold ARM)
  local tag=$1 a=$2 dep=$3 nm=$4 t0=$(date +%s) j md gr
  arm $a || return 1
  [ "${5:-}" = noreset ] || pcreset $tag-$nm
  msnap $tag-$nm-0; ssnap $tag-$nm-0
  run "(cd $D && timeout 1800 python3 $DQ $O/$tag-$nm.txt $dep $NTOK > $O/$tag-$nm.out 2>&1)"
  msnap $tag-$nm-1; ssnap $tag-$nm-1
  gr=$(ctrg "$(needs cold $a)" $tag-$nm-0 $tag-$nm-1)
  [ "$DRY" = 1 ] || echo "cold $tag-$nm $(echo "$gr" | cut -c1-200)" >> $O/guard.txt
  [ "$DRY" = 1 ] && { echo "+ RESULT xsp $tag $nm: arm $a depth $dep: TTFT, cached, acceptance, non-decode step ms, /stats deltas, counter guard ($gr) -> $O/deep.jsonl"; return 0; }
  j=$(grep -m1 '^DEEPQ ' $O/$tag-$nm.out | cut -c7-)
  md=$(python3 -c 'import json,sys
d = json.loads(sys.argv[1] or "{}"); d.update(json.loads(sys.argv[2] or "{}")); print(json.dumps(d))' "$(mdelta $tag-$nm-0 $tag-$nm-1)" "$(sdelta $tag-$nm-0 $tag-$nm-1)" 2>/dev/null || mdelta $tag-$nm-0 $tag-$nm-1)
  [ -n "$j" ] || { xlog "$tag $nm: VIRHE no answer ($(tail -2 $O/$tag-$nm.out | tr '\n' ' ' | cut -c1-300))"; return 1; }
  python3 -c 'import json,sys
d = json.loads(sys.argv[1]); d.update(json.loads(sys.argv[2]) if sys.argv[2].strip() else {})
d.update(name=sys.argv[3], arm=sys.argv[4], depth=int(sys.argv[5]), reset=sys.argv[6] != "noreset", text=sys.argv[7], guard=sys.argv[8].split(" | ")[0])
print(json.dumps(d))' "$j" "$md" "$nm" "$a" "$dep" "${5:-reset}" "$O/$tag-$nm.txt" "$gr" >> $O/deep.jsonl
  case "$gr" in OK*) ;; *) gr="VIRHE counter guard $gr" ;; esac
  xlog "$tag $nm (arm $(NAMEOF $a), deepq depth $dep, $NTOK greedy tokens$([ "${5:-}" = noreset ] && echo ", NO reset")): $(python3 -c 'import json,sys
d, m = json.loads(sys.argv[1]), json.loads(sys.argv[2] or "{}")
print("status %s, prompt %s tok, cached %s, TTFT %s s (%s prefill tok/s), %s tok, %s chars | decode acceptance %s (%s of %s drafted) | non-decode steps %s, %s ms mean | snapshots %s, staged %s GB (rank 0 %s GB, %s layers), all-reduce %s GB" % (
    d.get("status"), d.get("prompt_tokens"), d.get("cached"), d.get("ttft"),
    round((d["prompt_tokens"] - (d.get("cached") or 0)) / d["ttft"], 1) if d.get("ttft") and d.get("prompt_tokens") else "-",
    d.get("tokens"), d.get("chars"), m.get("acc"), m.get("draft_acc"), m.get("draft_tok"), m.get("nondecode_steps"), m.get("nondecode_ms"),
    m.get("ckpts"), m.get("staged_gb"), m.get("staged_gb_r0"), m.get("stage_layers_r0"), m.get("ar_gb")))' "$j" "$md" 2>&1 | cut -c1-600) | counters: $gr | $(thermline $t0 $(date +%s))"
  case "$gr" in OK*) return 0 ;; *) return 2 ;; esac
}
# ---------------------------------------------------------------- this item's own functions (lever SP)
kvroff() {   # kvroff TAG: lever KVR is NOT in this start -- no RADIANCE_DS41_KV_REPLICATED in the container, no "lever KVR" line
  [ "$DRY" = 1 ] && { echo "+ kvroff $1: RADIANCE_DS41_KV_REPLICATED not in the container env; no 'lever KVR' line in the log (KVR env OFF: only SP is measured)"; return 0; }
  local e l
  e=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null | grep -c '^RADIANCE_DS41_KV_REPLICATED=')
  l=$(docker logs $CN 2>&1 | grep -c 'lever KVR')
  [ "$e" = 0 ] && [ "$l" = 0 ] && { xlog "$1: lever KVR OFF (no env, no declare line): only SP is measured"; return 0; }
  xlog "$1: VIRHE lever KVR is in this start (env $e, log lines $l) -- the item measures SP alone"; return 1
}
swnote() {   # swnote TAG: start values (stack5's S5T-relevant switches 0, SP 0, sp capable on 4 ranks); refusals; arms B S 1 2 read back
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 at start (gg_gemv 2, aql_acq 2, K1 three 1, epoch 0, draft_capable 1; sched_bi_chunk / sched_noa / ds41_psk_l20 / dspark_pfctx / ds41_engdd / ds41_psk_cr / stage_short / stage_pick / place_hb 0; stage_full_min 0, stage_full_pick 0; sp capable 1 on 4 ranks, every counter 0); refusals stage_full_min 1048577 / stage_full_pick 65 / stage_short 65537 -> 400; arms $(for a in B S 1 2; do echo -n "$a $(pinj $a) | "; done)each applied and read back, then B"; return 0; }
  local s0 g c1 c2 c3 a sets="" bad=""
  s0=$(fn4get)
  c1=$(fn4code '{"stage_full_min": 1048577}'); c2=$(fn4code '{"stage_full_pick": 65}'); c3=$(fn4code '{"stage_short": 65537}')
  for a in B S 1 2 B; do
    arm $a || bad="$bad $a"; g=$(fn4get)
    sets="$sets $a:$(echo "$g" | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print("psk=%s/%s,cr=%s,l20=%s,pf=%s,ed=%s,bic=%s,noa=%s,ds=%s,occ=%s,bi=%s,sfm=%s,sfp=%s,ep=%s" % (w.get("ds41_psk"), w.get("ds41_psk_k"), w.get("ds41_psk_cr"), w.get("ds41_psk_l20"), w.get("dspark_pfctx"), w.get("ds41_engdd"), w.get("sched_bi_chunk"), w.get("sched_noa"), w.get("draft_sample"), w.get("draft_occ"), w.get("ds41_bi"), w.get("stage_full_min"), w.get("stage_full_pick"), d.get("epoch")))' 2>/dev/null)"
  done
  xlog "$1 switch note: at start $(echo "$s0" | cut -c1-1800) | stage_full_min 1048577 -> HTTP $c1, stage_full_pick 65 -> HTTP $c2, stage_short 65537 -> HTTP $c3 | arms applied (read back):$sets$([ -n "$bad" ] && echo " -- VIRHE not applied:$bad")"
  python3 -c 'import json,sys
s0 = json.loads(sys.argv[1]); w = s0["switches"]; sp = s0.get("sp") or {}
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) and s0["epoch"] == 0
ok = ok and w["draft_adaptive"] == 1 and s0["draft_capable"] == 1 and w.get("ds41_psk") == 0
ok = ok and all(w.get(k) == 0 for k in ("sched_bi_chunk", "sched_noa", "ds41_psk_l20", "dspark_pfctx", "ds41_engdd", "ds41_psk_cr", "stage_short",
                                         "stage_pick", "place_hb", "stage_full_min", "stage_full_pick"))
rk = sp.get("ranks") or []
ok = ok and sp.get("capable") == 1 and len(rk) == 4 and all(r.get("full_pick_kept", 1) == 0 and r.get("stream_steps", 1) == 0 for r in rk)
sys.exit(0 if ok else 1)' "$s0" 2>/dev/null && [ "$c1" = 400 ] && [ "$c2" = 400 ] && [ "$c3" = 400 ] && [ -z "$bad" ]
}
budok() {   # budok NAME EST_MIN: 0 = run the section; 1 = skip it (the time now + its estimate would pass XSP_BUDGET_MIN), logged
  [ "$DRY" = 1 ] && { echo "+ budget: $1 runs if elapsed + ~$2 min <= $BUDGET min, else skipped (logged)"; return 0; }
  [ $(( $(elapsed) + $2 )) -le $BUDGET ] && return 0
  xlog "$1 SKIPPED (budget): $(elapsed) min elapsed + ~$2 min > $BUDGET min"; return 1
}
CELLN=0; SCELLS=""
cellx() {   # cellx KIND SIZE CTX ARM: arm read back, snapshot, XSP_SCN fresh one-token alone prompts (ncone.py), snapshot;
            # spsum.py cell (per prompt prefill ms, staged / streamed GB, freshness) + need (from the cell's own prompt sizes); the guard
  local kind=$1 sz=$2 ctx=$3 a=$4 nm t0=$(date +%s) c pts nf gr
  CELLN=$((CELLN + 1)); nm=$kind-$CELLN-$sz-$a
  arm $a || { echo "$nm FAIL pin not applied" >> $O/guard.txt; return 1; }
  ssnap $nm-0
  run "(cd $D && timeout 900 python3 -I -B $NCONE --mb $MBP --url $URL --n $SCN --ctx $ctx --k0 $((300 + CELLN * 7)) --tag $nm --out $O/$nm-reqs.jsonl --stats $O/$nm-stats.jsonl > $O/$nm.txt 2>&1)"
  ssnap $nm-1
  if [ "$DRY" = 1 ]; then
    echo "+ python3 -I -B $SUM cell --reqs $O/$nm-reqs.jsonl --stats $O/$nm-stats.jsonl --fn0 $O/fn-$nm-0.json --fn1 $O/fn-$nm-1.json --arm $a --size $sz --sw $(swof $a) --json $O/$nm.json; guard = spsum.py need --sw $(swof $a) --pts <the cell's prompt sizes> (e.g. ~${ctx}+99 tok: $(python3 -I -B $SUM need --sw $(swof $a) --pts $((ctx + 99)))) -> spctr.py; RESULT xsp $kind $sz $(NAMEOF $a) -> $O/cells.jsonl"
    return 0
  fi
  c=$(timeout 120 python3 -I -B $SUM cell --reqs $O/$nm-reqs.jsonl --stats $O/$nm-stats.jsonl --fn0 $O/fn-$nm-0.json --fn1 $O/fn-$nm-1.json --arm $a --size $sz --sw $(swof $a) --json $O/$nm.json 2>&1 | grep -m1 '^CELL ' | cut -c6-)
  [ -n "$c" ] || c='{}'
  pts=$(python3 -c 'import json,sys; print(",".join(str(x) for x in json.loads(sys.argv[1]).get("prompts", [])))' "$c" 2>/dev/null)
  nf=$(python3 -I -B $SUM need --sw $(swof $a) --pts "${pts:-0}" 2>&1 | tail -1)
  gr=$(ctrg "$nf" $nm-0 $nm-1)
  echo "$nm $(echo "$gr" | cut -c1-200)" >> $O/guard.txt
  python3 -c 'import json,sys
print(json.dumps(dict(kind=sys.argv[1], size=sys.argv[2], arm=sys.argv[3], name=sys.argv[4], json=sys.argv[5], need=sys.argv[6], guard=sys.argv[7].split(" | ")[0])))' \
    "$kind" "$sz" "$a" "$nm" "$O/$nm.json" "$nf" "$gr" >> $O/cells.jsonl
  case "$gr" in OK*) ;; *) gr="VIRHE counter guard $gr" ;; esac
  xlog "$kind $sz $(NAMEOF $a) $(armj $a) [$nm]: $(python3 -c 'import json,sys
d = json.loads(sys.argv[1])
print("%s prompts (%s errors), %s..%s tok | prefill %s ms median (mean %s; steps %s) | reply %s s | staged %s GB all ranks (rank 0 %s), streamed per rank %s GB%s" % (
    d.get("n"), d.get("errors"), d.get("prompt_min"), d.get("prompt_max"), d.get("pf_ms_med"), d.get("pf_ms_mean"), d.get("pf_steps"),
    d.get("t_resp_med"), d.get("staged_gb_med"), d.get("staged_gb_r0_med"), d.get("stream_gb_rank_med"),
    (" -- VIRHE " + "; ".join(d["guard"])[:300]) if d.get("guard") else ""))' "$c" 2>&1 | cut -c1-700) | need [$nf] counters: $gr | $(thermline $t0 $(date +%s))"
}
screen() {   # (a) the stager screen: a warm-up cell, then per size the cells SORDER; spsum.py pick -> SPN / SPK / KP
  local z sz ctx a cells="" p
  cellx warm 3k 2900 B
  for z in $SIZES; do
    sz=${z%%:*}; ctx=${z#*:}
    for a in $(echo $SORDER | grep -o .); do
      stopfile && return 0
      cellx scr $sz $ctx $a
      cells="$cells --cell $sz:$a:$O/scr-$CELLN-$sz-$a.json"
    done
  done
  if [ "$DRY" = 1 ]; then echo "+ python3 -I -B $SUM pick$cells --sizes $(for z in $SIZES; do echo -n "${z%%:*},"; done | sed 's/,$//') --json $O/screen.json > $O/screen.out -> PICK N* K* KP (SP-best = stage_full_min N* / stage_full_pick K*; SP-P = stage_full_pick KP)"; return 0; fi
  timeout 120 python3 -I -B $SUM pick $cells --sizes $(for z in $SIZES; do echo -n "${z%%:*},"; done | sed 's/,$//') --json $O/screen.json > $O/screen.out 2>&1
  p=$(grep -m1 '^PICK ' $O/screen.out | cut -c6-)
  while IFS= read -r l; do xlog "${l:0:2500}"; done < <(grep '^SCREEN ' $O/screen.out)
  case "$p" in *[0-9]" "*[0-9]" "*[0-9]) read -r SPN SPK KP <<< "$p" ;;
    *) xlog "screen: VIRHE no pick ($(tail -2 $O/screen.out | tr '\n' ' ' | cut -c1-300)) -- SP-best = SP-S ($SFULL, 0), SP-P = pick 1"; SPN=$SFULL; SPK=0; KP=1 ;; esac
  xlog "screen verdict: SP-best = $(armj X), SP-P = $(armj P)"
}
tails() {   # (b) alone tails: per size the cells TORDER (B vs SP-best)
  local z sz ctx a
  for z in $TSIZES; do
    sz=${z%%:*}; ctx=${z#*:}
    for a in $(echo $TORDER | grep -o .); do stopfile && return 0; cellx tail $sz $ctx $a; done
  done
}
c8spec() {   # c8spec ARM -> "need/forbid" for a c8 window: forbid only (a co-batched full-plan step is the load's, not certain)
  case $1 in
    B) echo "/sppick,spstream" ;;
    X) if [ $SPN -gt 4160 ]; then echo "/spfull,sppick"; elif [ $SPN = 0 ]; then [ $SPK -gt 0 ] && echo "/spstream" || echo "/sppick,spstream"
       else [ $SPK -gt 0 ] && echo "/" || echo "/sppick"; fi ;;
  esac
}
c8x() {   # c8x I ARM: mb c8 at temperature 0 (mbt.py), dur C8DUR warm C8WARM, under ARM -> agg tok/s, decode_share, staged GB, sp counters
  local i=$1 a=$2 nm=c8-$1$2 t0 t1 j sd gr
  arm $a || { echo "$nm FAIL pin not applied" >> $O/guard.txt; return 1; }
  ssnap $nm-0
  t0=$(date +%s.%N)
  run "(cd $D && timeout 900 python3 -I -B $MBT --mb $MBP --temperature 0 --tag xsp-$nm --phases c8 --dur $C8DUR --warm $C8WARM > $O/$nm.txt 2>&1)"
  t1=$(date +%s.%N)
  ssnap $nm-1
  gr=$(ctrg "$(c8spec $a)" $nm-0 $nm-1)
  [ "$DRY" = 1 ] && { echo "+ RESULT xsp c8 $nm ($(NAMEOF $a) $(armj $a)): MB line agg_dec_tps / decode_share, staged GB, GET sp deltas, guard [$(c8spec $a)] ($gr) -> $O/c8.jsonl (the window [t0 + $C8WARM s, t1] for the step walls)"; return 0; }
  echo "$nm $(echo "$gr" | cut -c1-200)" >> $O/guard.txt
  j=$(grep -m1 '^MB ' $O/$nm.txt | cut -c4-)
  [ -n "$j" ] || { xlog "$nm: VIRHE no MB line ($(tail -3 $O/$nm.txt | tr '\n' ' ' | cut -c1-300))"; return 1; }
  sd=$(sdelta $nm-0 $nm-1)
  python3 -c 'import json,sys
d = json.loads(sys.argv[1]); s = json.loads(sys.argv[2] or "{}")
def sp(p):
    try: return {r["rank"]: r for r in (json.load(open(p)).get("sp") or {}).get("ranks") or []}
    except Exception: return {}
a, b = sp(sys.argv[5]), sp(sys.argv[6])
dl = lambda k: (b.get(0, {}).get(k, 0) - a.get(0, {}).get(k, 0))
print(json.dumps(dict(name=sys.argv[3], arm=sys.argv[4], t0=float(sys.argv[7]), t1=float(sys.argv[8]), agg_dec_tps=d.get("agg_dec_tps"),
      decode_share=d.get("decode_share"), per_stream_dec_p50=d.get("per_stream_dec_p50"), ttft_p50=d.get("ttft_p50"), requests=d.get("requests"),
      errors=d.get("errors"), staged_gb=s.get("staged_gb"), sp_full=dl("full_steps"), sp_stream=dl("stream_steps"),
      sp_pick=dl("full_pick_kept") + dl("full_pick_dropped"), guard=sys.argv[9].split(" | ")[0])))' \
    "$j" "$sd" "$nm" "$a" "$O/fn-$nm-0.json" "$O/fn-$nm-1.json" "$t0" "$t1" "$gr" >> $O/c8.jsonl
  case "$gr" in OK*) ;; *) gr="VIRHE counter guard $gr" ;; esac
  xlog "c8 $nm ($(NAMEOF $a) $(armj $a), mb c8 T 0, dur $C8DUR warm $C8WARM): $(tail -1 $O/c8.jsonl | cut -c1-500) | counters: $(echo "$gr" | cut -c1-600)"
}
c8s() {   # (e) the c8 cells in C8ORDER, then the step walls of each window from one engine-log capture
  local i=0 a
  for a in $(echo $C8ORDER | grep -o .); do i=$((i + 1)); stopfile && return 0; c8x $i $a; done
  capture $O/c8-engine.log.gz
  [ "$DRY" = 1 ] && { echo "+ per c8 window: python3 -I -B $SUM steps --log $O/c8-engine.log.gz --t0 <t0 + $C8WARM> --t1 <t1> -> the co-batched full-plan steps' wall (n, rows, p50 / mean ms), span-level and decode steps; rewritten into $O/c8.jsonl"; return 0; }
  [ -s $O/c8.jsonl ] || return 0
  timeout 600 python3 -c 'import json,subprocess,sys
o, summ, w = sys.argv[1], sys.argv[2], float(sys.argv[3])
rs = [json.loads(l) for l in open(o + "/c8.jsonl") if l.strip()]
for r in rs:
    p = subprocess.run([sys.executable, "-I", "-B", summ, "steps", "--log", o + "/c8-engine.log.gz", "--t0", str(r["t0"] + w), "--t1", str(r["t1"])],
                       capture_output=True, text=True, timeout=300)
    l = [x for x in p.stdout.splitlines() if x.startswith("STEPS ")]
    r["steps"] = json.loads(l[0][6:]) if l else {"error": (p.stderr or p.stdout)[-200:]}
with open(o + "/c8.jsonl", "w") as f:
    for r in rs: f.write(json.dumps(r) + "\n")
for r in rs: print("C8STEPS %s %s %s" % (r["name"], r["arm"], json.dumps(r["steps"])))' $O $SUM $C8WARM > $O/c8-steps.out 2>&1
  while IFS= read -r l; do xlog "c8 step walls ${l#C8STEPS }"; done < <(grep '^C8STEPS ' $O/c8-steps.out | cut -c1-900)
  grep -q '^C8STEPS ' $O/c8-steps.out || xlog "c8 step walls: VIRHE not parsed ($(tail -2 $O/c8-steps.out | tr '\n' ' ' | cut -c1-300))"
}
CHN1=""
chainx() {   # chainx I ARM: the 4-turn tchain under ARM after a reset -- the same text and nonce every chain (strict identity across arms)
  local i=$1 a=$2 p=ch$1$2 t ans="" j sd gr k
  arm $a || { echo "$p FAIL pin not applied" >> $O/guard.txt; return 1; }
  pcreset $p
  ssnap $p-0
  for t in 1 2 3 4; do
    stopfile && return 0
    run "(cd $D && timeout 1200 python3 -I -B $CHN run --pskld $PLC --tests $TESTS --url $URL --spec 'tchain:$CDEPTH:$COFF:$CTT:$TOFF${ans:+:$ans}@N-$T00' --ntok $CNTOK --out $O/$p-t$t.json --text $O/$p-t$t.txt > $O/$p-t$t.log 2>&1)"
    ssnap $p-$t
    ans="${ans:+$ans,}$O/$p-t$t.txt"
    [ "$DRY" = 1 ] && continue
    j=$(grep -m1 '^LONGID ' $O/$p-t$t.log | cut -c8-)
    [ -n "$j" ] || { xlog "$p turn $t: VIRHE no LONGID line ($(tail -3 $O/$p-t$t.log | tr '\n' ' ' | cut -c1-300))"; return 1; }
    sd=$(sdelta $p-$((t - 1)) $p-$t)
    k=$(timeout 60 python3 -I -B $CTR guard --st0 $O/st-$p-$((t - 1)).json --st1 $O/st-$p-$t.json --fn0 $O/fn-$p-$((t - 1)).json --fn1 $O/fn-$p-$t.json 2>&1 | tail -1 | sed 's/^OK | //' | cut -c1-700)
    python3 -c 'import json,sys
d = json.loads(sys.argv[1]); s = json.loads(sys.argv[2] or "{}")
def sp(p):
    try: return {r["rank"]: r for r in (json.load(open(p)).get("sp") or {}).get("ranks") or []}
    except Exception: return {}
a, b = sp(sys.argv[6]), sp(sys.argv[7])
dl = lambda k: (b.get(0, {}).get(k, 0) - a.get(0, {}).get(k, 0))
d.update(chain=sys.argv[3], arm=sys.argv[4], turn=int(sys.argv[5]), staged_gb=s.get("staged_gb"), sp_full=dl("full_steps"),
         sp_stream=dl("stream_steps"), sp_pick=dl("full_pick_kept") + dl("full_pick_dropped"), text=sys.argv[8], ctr=sys.argv[9])
print(json.dumps(d))' "$j" "$sd" "ch$i" "$a" "$t" "$O/fn-$p-$((t - 1)).json" "$O/fn-$p-$t.json" "$O/$p-t$t.txt" "$k" >> $O/chain.jsonl
    xlog "$p turn $t ($(NAMEOF $a) $(armj $a); turn 1 ~$CDEPTH tok cold, later turns + answer ($CNTOK greedy) + ~$CTT tok of tool text, nonce N-$T00): $(python3 -c 'import json,sys
d, s = json.loads(sys.argv[1]), json.loads(sys.argv[2] or "{}")
print("status %s, prompt %s tok, cached %s, tail %s rows, TTFT %s s, %s tok | staged %s GB (rank 0 %s), non-decode-free" % (d.get("status"), d.get("prompt_tokens"), d.get("cached"),
      (d.get("prompt_tokens") or 0) - (d.get("cached") or 0), d.get("ttft"), d.get("tokens"), s.get("staged_gb"), s.get("staged_gb_r0")))' "$j" "$sd" 2>&1 | sed 's/, non-decode-free//' | cut -c1-400) | counters: $k"
    if [ -z "$CHN1" ] || [ "$CHN1" = "$p" ]; then :; else
      idrec "chain-$p-t$t-vs-$CHN1" "$(cmpfile $O/$CHN1-t$t.txt $O/$p-t$t.txt)" "turn $t of the $(NAMEOF $a) chain vs the first B chain (same prompts)"
    fi
  done
  [ "$DRY" = 1 ] && { echo "+ chain guard $p ($(NAMEOF $a)): [$(needs chain $a)] (need/forbid) over turns 1-4"; return 0; }
  [ -z "$CHN1" ] && [ $a = B ] && CHN1=$p
  gr=$(ctrg "$(needs chain $a)" $p-0 $p-4)
  echo "chain $p $(echo "$gr" | cut -c1-300)" >> $O/guard.txt
  case "$gr" in OK*) xlog "$p chain guard: $gr" ;; *) xlog "$p chain guard: VIRHE $gr" ;; esac
}
chains() { local i=0 a; for a in $(echo $CORDER | grep -o .); do i=$((i + 1)); stopfile && return 0; chainx $i $a; done; }
cold64() {   # (d) the cold 64k: DORDER, each cold after a reset; identity vs the first B
  local k=0 a fa="" f
  for a in $(echo $DORDER | grep -o .); do
    k=$((k + 1)); stopfile && return 0
    deepx sp $a $DDEPTH d$DDEPTH-$k$a
    [ "$DRY" = 1 ] && continue
    f=$O/sp-d$DDEPTH-$k$a.txt
    [ -s $f ] || continue
    [ -n "$fa" ] || { [ $a = B ] && { fa=$f; continue; }; }
    [ -n "$fa" ] && idrec "d$DDEPTH-$k$a-vs-B" "$(cmpfile $fa $f)" "cold ${DDEPTH} deepq answer vs the first B"
  done
}
idset() {   # the alone identity set: deep:DEPTH:OFF (64 greedy tokens), a reset before each, under B S P X; each SP answer vs B; the guard
  local dep i=0 a nm j pt nf gr
  for dep in $IDEPTHS; do
    i=$((i + 1))
    for a in B S P X; do
      stopfile && return 0
      nm=id-$dep-$a
      arm $a || { echo "$nm FAIL pin not applied" >> $O/guard.txt; continue; }
      pcreset $nm
      ssnap $nm-0
      run "(cd $D && timeout 900 python3 -I -B $CHN run --pskld $PLC --tests $TESTS --url $URL --spec 'deep:$dep:$((IOFF + 40 * i))' --ntok 64 --out $O/$nm.json --text $O/$nm.txt > $O/$nm.log 2>&1)"
      ssnap $nm-1
      [ "$DRY" = 1 ] && continue
      j=$(grep -m1 '^LONGID ' $O/$nm.log | cut -c8-)
      [ -n "$j" ] || { xlog "$nm: VIRHE no LONGID line ($(tail -3 $O/$nm.log | tr '\n' ' ' | cut -c1-300))"; continue; }
      pt=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("prompt_tokens") or 0)' "$j" 2>/dev/null)
      nf=$(python3 -I -B $SUM need --sw $(swof $a) --pts "${pt:-0}" 2>&1 | tail -1)
      gr=$(ctrg "$nf" $nm-0 $nm-1)
      echo "$nm $(echo "$gr" | cut -c1-200)" >> $O/guard.txt
      case "$gr" in OK*) ;; *) gr="VIRHE counter guard $gr" ;; esac
      xlog "identity set $nm ($(NAMEOF $a) $(armj $a), deep:$dep, cold after a reset, 64 greedy tokens): $(python3 -c 'import json,sys
d = json.loads(sys.argv[1]); print("status %s, prompt %s tok, cached %s, TTFT %s s" % (d.get("status"), d.get("prompt_tokens"), d.get("cached"), d.get("ttft")))' "$j" 2>&1 | cut -c1-200) | need [$nf] counters: $(echo "$gr" | cut -c1-500)"
      [ $a = B ] || idrec "$nm-vs-B" "$(cmpfile $O/id-$dep-B.txt $O/$nm.txt)" "deep:$dep alone, $(NAMEOF $a) vs B"
    done
  done
}
summary() {
  [ "$DRY" = 1 ] && { echo "+ python3 -I -B $SUM summary --out $O > $O/summary.txt -> RESULT xsp SCREEN / TAILS / CHAIN / COLD 64k / IDENTITY + COUNTER GUARD lines"; return 0; }
  timeout 300 python3 -I -B $SUM summary --out $O > $O/summary.txt 2>&1
  while IFS= read -r l; do xlog "summary: ${l:0:3000}"; done < $O/summary.txt
}
dstart() {   # the one NST 5 start
  local tag=s5-n5 a
  diskok $tag || return 1
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on"
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV > $XS2_ENVFILE"; else printf '%s\n' $LEVENV > $XS2_ENVFILE; fi
  xlog "$tag: start ($DESC) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $FIX + S5T | env: $(env | grep -E '^(RADIANCE_|R4D_)' | grep -vE '(KEY|TOKEN|SECRET)' | sort | tr '\n' ' ' | cut -c1-2400)"
  gate1 $tag 70 || { xlog "$tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  modes $tag || { xlog "$tag: VIRHE the modes are not the start's -> stopped"; down; return 1; }
  kvroff $tag || { keeplog $tag-kvr; down; return 1; }
  if [ "$DRY" != 1 ]; then
    a=1
    docker logs $CN 2>&1 | grep -q "DSpark drafter" || { xlog "$tag: VIRHE no DSpark drafter line"; a=0; }
    docker logs $CN 2>&1 | grep -q "draft depth: adaptive" || { xlog "$tag: VIRHE no adaptive draft depth line"; a=0; }
    [ $a = 1 ] || { keeplog $tag-nodraft; down; return 1; }
  fi
  vramstart $tag
  swnote $tag || { xlog "$tag: VIRHE the switches are not as expected -> stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  stopfile || { budok screen 7 && screen; }
  stopfile || { budok tails 5 && tails; }
  stopfile || { budok c8 7 && c8s; }
  stopfile || { budok chains 3 && chains; }
  stopfile || { budok cold64 4 && cold64; }
  stopfile || { budok idset 2 && idset; }
  capture $O/$tag-engine-full.log.gz
  xlog "$tag: lever counters at the end: $(pskstat) | sp $(fn4get | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps(d.get("sp")))' 2>/dev/null | cut -c1-900) | $(docker logs --since 6h $CN 2>&1 | grep -cE '^E ') E lines | $(elapsed) min"
  down
  xlog "$tag: done"
}
trap 'down' EXIT
trap 'exit 143' TERM INT
# ---------------------------------------------------------------- main
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) | one NST 5 start: $DESC | (a) screen sizes [$SIZES] cells $SORDER x $SCN | (b) tails [$TSIZES] $TORDER x $SCN | (e) c8 $C8ORDER (mbt T 0, dur $C8DUR warm $C8WARM) | (c) chains $CORDER (~$CDEPTH + 3 x (~$CTT + $CNTOK)) | (d) cold $DDEPTH $DORDER | identity strict (chains, 64k, alone set [$IDEPTHS]) | counter guard everywhere | budget $BUDGET min | no evict"
IMGLITS="B|sched_short_bypass, sched_bypass_total, ds41_psk, ds41_psk_k, ds41_psk_l20, dspark_pfctx,
B|ds41_psk_cr, ds41_engdd, sched_noa, stage_short, stage_pick, place_hb, stage_full_min, stage_full_pick, epoch_bump)
B|stage_full_min
B|stage_full_pick
B|stage_full_min takes 0 (off) or 1 .. 1048576
B|stage_full_pick takes 0 (every pooled unit) or 1 .. 64
B|no prefill stager on every rank (a dynamic plan with a staging region)
B|full_pick_dropped
B|stream_rows
B|full_bytes
B|ds41_psk_cr
B|ds41_engdd
B|sched_noa
B|stage_short
B|stage_pick
B|place_hb
B|stage_short takes 0 (off) or 1 .. 65536
B|cr_end_snapshots
B|tail_only_latched
B|short_steps
B|lever SA / HB (ds41-sahb) set up, off: stage_short %s; place_hb %s
B|RADIANCE_PLACE_HB_RANKS
B|RADIANCE_LOG_STEPS
B|n_seq=%lld n_tok=%lld
B|reset_prefix_cache
B|radiance:engine_steps_mixed_total
B|radiance:engine_step_seconds_decode_total
A|RADIANCE_DS41_KV_REPLICATED
A|lever KVR declared (RADIANCE_DS41_KV_REPLICATED=1)
A|RADIANCE_DS41_PSK
A|lever PSK declared (RADIANCE_DS41_PSK, fn4 ds41_psk, now
A|lever PSK-L20 (fn4 ds41_psk_l20, now %d; RADIANCE_DS41_PSK_L20): %s
A|lever PSK-CR (fn4 ds41_psk_cr, now %d; RADIANCE_DS41_PSK_CR): %s
A|rad_psk_plan2
A|rad_dspark_pfctx_stats
A|lever E1 declared (RADIANCE_DS41_ENGDD, fn4 ds41_engdd, now
A|rad_engdd_stats
K|gather_rows_w16_kernel
A|lever BI declared (RADIANCE_DS41_BI, fn4 ds41_bi, now
A|DSpark lever SD: the sampled full-vocabulary walk declared
B|RADIANCE_WEIGHT_SIDECAR
B|lever Z0-A set up"
docker image inspect $IMAGE > /dev/null 2>&1 || { xlog "VIRHE image $IMAGE not on this host"; exit 1; }
IMGMISS=$(echo "$IMGLITS" | timeout 120 docker run --rm -i --network none --entrypoint sh $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels/libr4d.so; S=/opt/radiance/share/radiance/kernels/spechub.so; n=0
while IFS= read -r l; do [ -n "$l" ] || continue; n=$((n + 1)); w=${l%%|*}; s=${l#*|}
  case $w in B) f=$B ;; A) f=$A ;; K) f=$K ;; S) f=$S ;; *) f="$B $A $K" ;; esac
  o=0; for x in $f; do grep -qzF -- "$s" $x && { o=1; break; }; done
  [ $o = 1 ] || echo "MISS $w|$s"; done
test -x /opt/radiance/bin/rad-sidecar || echo "MISS bin/rad-sidecar"; for p in int7 int8 int9; do test -s /opt/radiance/share/radiance/$p-PROVENANCE.txt || echo "MISS $p-PROVENANCE.txt"; done; echo "CHECKED $n"' 2>&1)
case "$IMGMISS" in "CHECKED "*) ;; *) xlog "VIRHE $IMAGE is not the int9 image -- missing: $(echo "$IMGMISS" | grep -v '^CHECKED' | tr '\n' ';' | cut -c1-900)"; exit 1 ;; esac
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)) is the int9 image: all $(echo "$IMGMISS" | sed 's/CHECKED //') literals present (lever SP's registry names after place_hb, range / refusal messages, GET sp counters, the unknown-switch tail; KVR's env + declare line in deepseek41.so; the int8 levers PSK / L20 / CR / PSK-DS / E1 / NOA / SA-HB, the step log, BI / SD / DLW / Z0, the /metrics counters) + bin/rad-sidecar + int7 / int8 / int9 PROVENANCE -- OK"
HMISS=""
for f in "$LIB29" "$LIBPR" $Y $CTR $SUM $NCONE $CHN $PLC $MBT $DQ $MBP $TESTS $D/fastcheck.py; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $CTR $SUM $NCONE $CHN $PLC $MBT $DQ $MBP $TESTS $D/fastcheck.py 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
CTT_=$(timeout 60 python3 -I -B $CTR selftest 2>&1 | tail -1); case "$CTT_" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$CTR(${CTT_:0:300})" ;; esac
SMT=$(timeout 60 python3 -I -B $SUM selftest 2>&1 | tail -1); case "$SMT" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$SUM(${SMT:0:300})" ;; esac
MST=$(timeout 60 python3 -I -B $MBT --mb $MBP --selftest 2>&1 | tail -1); case "$MST" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$MBT(${MST:0:300})" ;; esac
NOT=$(timeout 60 python3 -I -B $NCONE --mb $MBP --selftest 2>&1 | tail -1); case "$NOT" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$NCONE(${NOT:0:300})" ;; esac
CHT=$(cd $D && timeout 120 python3 -I -B $CHN tselftest --pskld $PLC --tests $TESTS 2>&1 | tail -1); case "$CHT" in "SELFTEST OK"*) ;; *) HMISS="$HMISS tselftest:$CHN(${CHT:0:300})" ;; esac
CH3=$(cd $D && timeout 120 python3 -I -B $CHN selftest --pskld $PLC --tests $TESTS 2>&1 | tail -1); case "$CH3" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$CHN(${CH3:0:300})" ;; esac
TXT=$(cd $D && timeout 120 python3 -I -B $CHN texts --pskld $PLC --tests $TESTS --doffs $COFF,$TOFF,$((TOFF + 40)),$((TOFF + 80)),$(i=0; for d in $IDEPTHS; do i=$((i + 1)); echo -n "$((IOFF + 40 * i)),"; done | sed 's/,$//') 2>&1 | tail -1); case "$TXT" in "SELFTEST OK"*) ;; *) HMISS="$HMISS texts:$CHN(${TXT:0:300})" ;; esac
DQP=$(timeout 60 python3 -I -c 'import importlib.util, json, sys
def ld(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
t = ld(sys.argv[2], "t"); pl = ld(sys.argv[1], "pl")
import ast
src = open(sys.argv[3]).read()
task = next((ast.literal_eval(n.value) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "task" for t in n.targets)), None)
ok = "CPT, OFF = 3.95, 1200" in src and "# deepq fixed prompt v1" in src and pl.DEEP_CPT == 3.95 and pl.DEEP_OFF == 1200 and task == pl.DEEP_TASK
d = pl.deep_payload(t, 64000, 512)
print("DQ OK" if ok and d["messages"][0]["content"].startswith("# deepq fixed prompt v1\n") and d["max_tokens"] == 512 else "DQ MISMATCH")' $PLC $TESTS $DQ 2>&1 | tail -1)
case "$DQP" in "DQ OK") ;; *) HMISS="$HMISS deepq-vs-pskld:(${DQP:0:200})" ;; esac
FNS="log run mkd restart check down stopfile best_vars desc serve_args xlog elapsed armj swof NAMEOF pinj spfam needs stackenv keeplog capture thermline dfree diskok cmpfile idrec identw modes vramstart fn4get fn4set nudge fn4code pcreset arm msnap mdelta pskstat ssnap sdelta ctrg gate1 deepx kvroff swnote budok cellx screen tails c8spec c8x c8s chainx chains cold64 idset summary dstart"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee sha256sum gzip zcat; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
for a in B S 1 2 X P; do python3 -c 'import json,sys; d=json.loads(sys.argv[1]); assert d["ds41_psk"] == 4 and d["ds41_psk_cr"] == 1 and "stage_full_min" in d and "stage_full_pick" in d' "$(pinj $a)" 2>/dev/null || HMISS="$HMISS json:$a"; [ -n "$(spfam $a)" ] || HMISS="$HMISS spfam:$a"; done
for a in B S 1 2 X P; do for k in cold chain; do case "$(needs $k $a)" in */*) ;; *) HMISS="$HMISS needs:$k:$a" ;; esac; done; done
[ "$(sha256sum $MODELDIR/$SIDECAR 2>/dev/null | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256 2>/dev/null)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR"
[ "$(sha256sum $MODELDIR/$SIDE 2>/dev/null | cut -c1-64)" = "$SIDESHA" ] || HMISS="$HMISS sha256:$MODELDIR/$SIDE(want ${SIDESHA:0:16})"
grep -q 'XS2_ENVFILE' $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no env_file)"
grep -q 'RADIANCE_DS41_KV_REPLICATED' $Y 2>/dev/null && HMISS="$HMISS yaml:$Y(names RADIANCE_DS41_KV_REPLICATED)"
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config refuses it)"
case "$LEVENV" in *KV_REPLICATED*) HMISS="$HMISS levenv:KVR" ;; esac
HMISS=$(echo $HMISS)
[ -z "$HMISS" ] || { xlog "VIRHE helpers / inputs do not resolve: $HMISS"; exit 1; }
HMD5="spctr.py $(md5sum < $CTR | cut -c1-8) (copy of stack5/s5ctr.py $(md5sum < $HD/../stack5/s5ctr.py | cut -c1-8)), spsum.py $(md5sum < $SUM | cut -c1-8), ncone.py $(md5sum < $NCONE | cut -c1-8) (= sahb/ncone.py $(md5sum < $HD/../sahb/ncone.py | cut -c1-8)), spchain.py $(md5sum < $CHN | cut -c1-8) (copy of pskcr/pcr3.py $(md5sum < $HD/../pskcr/pcr3.py | cut -c1-8)), pskcr/pskld.py $(md5sum < $PLC | cut -c1-8), stack4/mbt.py $(md5sum < $MBT | cut -c1-8), deepq.py $(md5sum < $DQ | cut -c1-8), mb.py $(md5sum < $MBP | cut -c1-8), ds41_tests.py $(md5sum < $TESTS | cut -c1-8), dsm-sp.sh $(md5sum < $0 | cut -c1-8) (from stack5/dsm-stack5.sh $(md5sum < $HD/../stack5/dsm-stack5.sh | cut -c1-8)), lib-29.sh $(md5sum < $LIB29 | cut -c1-8), yaml $(md5sum < $Y | cut -c1-8)"
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing; spctr.py $CTT_ | spsum.py $SMT | ncone.py $NOT | mbt.py $MST | spchain.py $CHT | spchain.py (pcr3's) $CH3 | texts $TXT | deepq.py = pskld.py's deep request ($DQP) | all $(echo $FNS | wc -w) functions resolve, tools on PATH, every arm's pin (S5T + SP) and guard spec, sidecars $SIDECAR + $SIDE sha256 OK, yaml accepted (no KVR word), LEVENV without KVR -- OK | md5: $HMD5"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { xlog "VIRHE $M225 missing or not verified"; exit 1; }
  [ -s $WARM_REF_225 ] || xlog "note -- reference $WARM_REF_225 missing (not compared)"
  xlog "$ITEM: helpers md5: $HMD5"
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDE; do [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  for f in $WARM_REF_225 $RS5/s5-n5-stats0.json $RS5/guard-s5-n5.txt; do   # references shown beside this start's (information only)
    [ -e $f ] && echo "+ present (reference): $f" || echo "+ absent (reference, information only): $f"; done
  for a in B S 1 2 X P; do echo "+ arm $a ($(NAMEOF $a)): $(armj $a) | chain guard [$(needs chain $a)] | cold guard [$(needs cold $a)]"; done
  echo "+ DS base pin (S5T) + fixed pin: $(pinj B)"
  echo "+ orders: screen $SORDER at [$SIZES] | tails $TORDER at [$TSIZES] | c8 $C8ORDER (dur $C8DUR warm $C8WARM, T 0) | chains $CORDER | cold $DDEPTH $DORDER | identity set [$IDEPTHS] x B S P X"
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(information here; the real run refuses to start beside one)"
  echo "+ disk now: $(dfree) MB free on / (guard $DISKMIN MB)"
fi
mkd $O $O/guard $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt
stopfile || dstart
summary
unset $SWITCHES
down
xlog "$ITEM: done ($(elapsed) min) -- $O | $([ $NV = 0 ] && echo "no VIRHE line" || echo "$NV VIRHE line(s): see above")"
