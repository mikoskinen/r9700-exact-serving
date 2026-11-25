#!/bin/bash
# dsm-kv4.sh (12.11.2026, queue-ar/dsmain/kv4, ds-main-port kv4 builder) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r00zzz-cz-w-kv4.sh. Log tag "RESULT xkv4 ", out runs/xdsm/kv4 (XKV_OUT). Made from COPIES of
# stack4/dsm-stack4.sh (21ce8c8e: the start words, LITS / FIX, stackenv, modes, identw, cmpfile / idrec / cmpx, the alone /
# loaded identity, the image / helper / sidecar / yaml checks), stack4/mbt.py (94db30da) and nc/ncsteps.py (0a6e0ed2), with
# nc/ncmb.py (ded624e5: mbt.py's phase run + per-request taps + /stats at the window's edges, which ncsteps.py c8 reads) and
# nc/ncone.py (a24570f8: ncsteps.py / ncmb.py load its compact() from their own directory) copied beside them; the originals
# are not changed. + kv4sum.py (capacity, the expert-heavy HOSTPOOL, > 48-row counts, PSK guard, summary).
# LEVER KV4 (capacity for 10+ users): RADIANCE_DS41_LATENT=nvfp4, a start-level env of the DS engine: the compressed latent
#   caches hold each 512-value entry packed (288 bytes instead of 1024: e2m1 codes, an e4m3 scale a block of 16).
#   EXACT BY CODE (radiance-int7 0391a6e; second reading of LEVERS 'NVFP4 latent'): every row the latent store writes is
#   quant_roundtrip(fp4_e2m1, e4m3, 16) output over the whole 512-value row (rope part included) -- arch/deepseek41/
#   deepseek41.cpp:2025-2028 (rope_rows_roundtrip, else rope_rows + quant_roundtrip) before :2031 entry_store (and :1909, the
#   attention-zero rank's never-issued declare); issued over the same rows before the store at :4232 / :4240 -> :4242 (step)
#   and :4868-4870 (the lambda order l_rope, l_q, l_st). quant_roundtrip's output is e2m1 magnitude x an e4m3 scale >= 2^-9
#   (libr4d/r4d_ds41_qrt.h:110-113, :121-137), exact in bf16; the packer (libr4d/r4d_ds41_sparse.hip:2734-2771) takes the first
#   scale amax / c (c = 6, 4, 3, 2, 1.5, 1, 0.5) that is an e4m3 value and reproduces EVERY value's bf16 bits (sign of zero
#   included, ds_e2m1_code :294-299); the block's own scale is always among them, so the lossy rule fallback is unreachable for
#   these rows; readers unpack code x scale -> the stored bf16 (ds_nv4_bf16 :237-241; attn_sparse_sink latent_format nvfp4,
#   deepseek41.cpp:2123, the fold d copy :2138). NOT packed (unchanged, the same bytes under both modes): the window keys
#   (kv_window, fp8 round trip, bf16 slot; lever KVWIN apart), the index keys (mxfp4 under RADIANCE_DS41_IDXKEY, set in both
#   modes here), the compressor tail (f32 state, qsa_tail_store), the DSpark drafter's window cache (ratio 0, no latent).
#   No stored value changes: identity nvfp4 vs bf16 is strict everywhere below.
#   NVG (RADIANCE_DS41_NVGATHER=XKV_NVG, 48): the decode band's packed reads unpacked once into scratch (LEVERS hszzzzz-nvg:
#   ~94 % of the 1u cost recovered, identity IDENTICAL). Set in EVERY start: under bf16 it sizes no scratch and takes no launch
#   (r4d_ds41_sparse.hip:3459-3469 / :3094), so the starts differ in RADIANCE_DS41_LATENT alone.
# THE SPLIT KNOB (scope addition, coordinator 12.11.): core/mem/vram_budget.cpp:179-196 (the lends split): experts get their
#   floor (what the pinned host pool cannot hold: --host-pool-mib), the cache min(its ceiling = MAXSEQS x MML worst case,
#   the rest); the prefill staging region comes out of the experts above their floor, else the cache (:213-238). So:
#   (1) a smaller --host-pool-mib = more expert VRAM, less cache; (2) a ceiling below the room (nvfp4 at MAXSEQS 8, or MML
#   262144) hands the rest to the experts by itself. --prefix-cache-mib / --checkpoint-slots are pinned HOST memory (0 VRAM).
#   On this card the cache does not grow / shrink (no VMM, engine_bringup.cpp:4069-4073) but lends blocks no request holds to
#   the expert slab ("kv-lent" units).
# STARTS (each a separate start; everything else identical: stack4's NST 5 start words + RADIANCE_LOG_STEPS=1 + NVG; one pin
#   S = stack4's fixed pin + the S switches of XKV_PINS + sched_bi_chunk XKV_BIC (3: option D keeps a loaded prompt's chunks on
#   alone's pieces, so the cross-start loaded compares are strict; its over-budget cut 0ea8cec = ds41-bi db15e15 is in int7 and
#   acts only under 3 / 4; the >64-decode-row case of MAXSEQS 16)), order XKV_STARTS (b8 n8 x8 n16 b16 m8):
#     b8  bf16  MAXSEQS 8,  HOSTPOOL 19456, MML 524288 (today)       n8  nvfp4 MAXSEQS 8 (the planner's own split)
#     x8  bf16  MAXSEQS 8,  HOSTPOOL = 19456 - (n8's experts - b8's) MiB (equal expert VRAM to n8: the expert-heavy bf16)
#     n16 nvfp4 MAXSEQS 16  /  b16 bf16 MAXSEQS 16 (the c16 cell; same split both)   m8 bf16 MAXSEQS 8 MML 262144 (smaller pool)
#   M8 starts (b8 n8): PSK guard (cold 16k deepq) -> 1u essay / code x XKV_1UN -> mb c8 x XKV_C8N -> cold 136k -> greedy set
#     alone (a1) -> greedy set conc 8 (gs8) -> c4 + cold 64k probe (ld, stack4/pskld.py --load)
#   X starts (x8 m8): PSK guard -> 1u x 1 -> mb c8 x XKV_C8N -> cold 136k -> a1
#   M16 starts (n16 b16): PSK guard -> mb c16 x XKV_C16N -> mb c8 x 1 -> a1 -> greedy set conc 16 (gs16) -> lid16 = the > 48-row
#     BI check (coordinator 11.11.): the pin + DET (fixed depth 5, no depth 0 / occupancy / hybrid / confidence cut), each
#     greedy prompt under XKV_LIDCONC (14) mb streams + one fresh ~XKV_LIDPF (1100) newcomer (stack4/loadid.py) = 16 sequences
#     x 6 rows = up to 96 decode rows a step; the step log counts the steps with > 48 decode rows, deepseek41's 'lever BI stats'
#     line too; identity 20 / 20 vs alone required before BI is claimed at MAXSEQS 16.
#   mb windows: ncmb.py --mode phase (mbt.py's phase_c at temperature 0, dur XKV_DUR 60 warm XKV_WARM 15) + ncsteps.py c8 over
#     the step log: agg_dec_tps, per-stream p50, decode_share, the newcomer step wall, streamed / staged GB a non-decode step,
#     vram / kv-lent units a rank at the window's edges.
# READINGS: capacity (start log + /stats: experts / kv split, tokens, 524288 sessions, B a token, units a rank, lent) per start,
#   MAXSEQS 8 / 12 / 16 per mode from the pool numbers (12: no start); the smaller-pool prices (MML 262144, one session, the c8
#   working set); speed per start; pairs n8-b8, x8-b8, n8-x8 (equal expert VRAM), m8-b8, n16-b16; the c16 cell; identity.
# COUNTER GUARD: the latent mode read back (container env + deepseek41's indexer note 'latent nvfp4 (288 bytes' / 'latent bf16
#   (1024 bytes'), the pin read back after every switch (POST + epoch_bump, a nudge, GET compared), PSK design 2 on the corrected
#   /stats psk counters over a cold 16k prompt (latched / planned_steps / cut_passes grow, plan_mismatch +0; never
#   layers_skipped). A failed PSK guard: VIRHE, that start's speed readings are skipped (identity still runs).
# ~83 min (estimate; budget XKV_BUDGET_MIN 120 -- a start that would end past it is skipped and logged): M8 ~13 each (start +
#   check ~5, guard 0.5, 1u x 2 0.7, c8 x 2 3.4, 136k 0.9, a1 / gs8 0.6, ld 1.5, capture 0.3), X ~11 each, M16 ~16 each
#   (c16 x 2 3.8, c8 1.7, a1 / gs16 0.7, lid16 ~3.5); + preflight / parse / report ~3. Disk guard XKV_DISKMIN_MB 11000 a start.
# NO cc_sim, no evict, no power / host / fan settings; production compose files read only (--project-directory). Exact.
# Knobs: XKV_IMAGE, XKV_ITEM, XKV_OUT, XKV_PINS (auto | s4 | s5; auto = s5 for a ds41-int8 image), XKV_SPIN (the S switches as
#   JSON; default by XKV_PINS), XKV_S5ENV (s5 start words, "RADIANCE_DS41_ENGDD=0"), XKV_BIC (3), XKV_NVG (48), XKV_STARTS,
#   XKV_HP (19456), XKV_XHP (auto | MiB), XKV_MMLX (262144), XKV_DUR (60), XKV_WARM (15), XKV_C8N (2), XKV_C16N (2), XKV_1UN (2),
#   XKV_LIDCONC (14), XKV_LIDPF (1100), XKV_DISKMIN_MB (11000), XKV_BUDGET_MIN (120), XKV_SIDECAR, XKV_SIDESHA.
# DRYRUN=1 bash dsm-kv4.sh : prints every command; checks the image literals (lever KV4 / NVG / BI stats / the step log / the
#   pool notes + stack4's), every helper (files, parse, selftests), the sidecars, the yaml, the disk; no server, no GPU, no
#   request to :8090, writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/../ds41-stack2-tp4.yaml                   # stack2's yaml, read only (env_file ${XS2_ENVFILE} carries the start words)
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh
O=${XKV_OUT:-$R/xdsm/kv4}
RS4=$R/xdsm/stack4; RD=$R/xdsm/draft; RM=$R/xdsm
export IMAGE=${XKV_IMAGE:?set XKV_IMAGE}
ITEM=${XKV_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-w-kv4}
PINS=${XKV_PINS:-auto}; [ "$PINS" = auto ] && { case $IMAGE in *ds41-int8*) PINS=s5 ;; *) PINS=s4 ;; esac; }
BIC=${XKV_BIC:-3}; NVG=${XKV_NVG:-48}; STARTS=${XKV_STARTS:-b8 n8 x8 n16 b16 m8}
HP=${XKV_HP:-19456}; XHP=${XKV_XHP:-auto}; MMLX=${XKV_MMLX:-262144}
DUR=${XKV_DUR:-60}; WARMS=${XKV_WARM:-15}; C8N=${XKV_C8N:-2}; C16N=${XKV_C16N:-2}; U1N=${XKV_1UN:-2}
LIDC=${XKV_LIDCONC:-14}; LIDPF=${XKV_LIDPF:-1100}; DISKMIN=${XKV_DISKMIN_MB:-11000}; BUDGET=${XKV_BUDGET_MIN:-120}
SIDE=${XKV_SIDECAR:-dspark-dlw-q8_0.rad}; SIDESHA=${XKV_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
case $PINS in s4) DSPIN='{"ds41_bi": 2, "draft_sample": 85, "draft_occ": 6, "place_lf": 0, "ds41_psk": 2, "ds41_psk_k": 4}'; DPEX="" ;;
  s5) DSPIN='{"ds41_bi": 2, "draft_sample": 85, "draft_occ": 6, "place_lf": 0, "ds41_psk": 2, "ds41_psk_k": 8, "ds41_psk_l20": 1, "dspark_pfctx": 1, "ds41_engdd": 1, "sched_noa": 2, "sched_bi_chunk": 3}'
      DPEX=${XKV_S5ENV-RADIANCE_DS41_ENGDD=0} ;;
  *) echo "XKV_PINS must be auto, s4 or s5 (got '$PINS')"; exit 1 ;; esac
SPIN=${XKV_SPIN:-$DSPIN}; PEXTRA=$DPEX
python3 -c 'import json,sys; d=json.loads(sys.argv[1]); assert isinstance(d, dict) and d' "$SPIN" 2>/dev/null || { echo "XKV_SPIN is not a JSON object ('$SPIN')"; exit 1; }
for v in $BIC $NVG $HP $MMLX $DUR $WARMS $C8N $C16N $U1N $LIDC $LIDPF $DISKMIN $BUDGET; do case $v in ''|*[!0-9]*) echo "a numeric knob is not a number ('$v')"; exit 1 ;; esac; done
case $XHP in auto|[0-9]*) ;; *) echo "XKV_XHP takes auto or MiB (got '$XHP')"; exit 1 ;; esac
case $BIC in 0|1|2|3|4) ;; *) echo "XKV_BIC takes 0 .. 4 (got $BIC)"; exit 1 ;; esac
[ $NVG -le 48 ] || { echo "XKV_NVG is 0 (off) .. 48 (the decode band; more is capped there)"; exit 1; }
[ $WARMS -lt $DUR ] || { echo "the warm-up is not shorter than the phase"; exit 1; }
for s in $STARTS; do case $s in b8|n8|x8|m8|b16|n16) ;; *) echo "XKV_STARTS takes b8 n8 x8 m8 b16 n16 (got '$s')"; exit 1 ;; esac; done
if [ "$XHP" = auto ]; then   # x8's HOSTPOOL is read from b8's and n8's capacity: both must run before it
  seen=""; for s in $STARTS; do [ $s = x8 ] && { case "$seen" in *b8*n8*|*n8*b8*) ;; *) echo "x8 with XKV_XHP=auto needs b8 and n8 before it in XKV_STARTS"; exit 1 ;; esac; }; seen="$seen $s"; done
fi
NCS=$HD/ncsteps.py; NCMB=$HD/ncmb.py; NCONE=$HD/ncone.py; MBT=$HD/mbt.py; SUM=$HD/kv4sum.py
LD=$HD/../stack4/loadid.py; PL=$HD/../stack4/pskld.py                                     # stack4's drivers, read only
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
MBP=$D/mb.py; TESTS=$D/ds41_tests.py; SPB=$D/stepprobe.py; DQ=$QA/deepq.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
MT=192; GDEP=16000; T0=$(date +%s); NV=0
xlog() { case "$*" in *VIRHE*) NV=$((NV + 1)) ;; esac; log "RESULT xkv4 $*"; }
# ---------------------------------------------------------------- copied from dsm-stack4.sh (21ce8c8e; tag xs4 -> xkv4)
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
# + this item's: the index keys packed in every start (stack4's RADIANCE_DS41_IDXKEY=mxfp4), the pool notes the capacity reads
LITS="$LITS|keys,mxfp4,(80,bytes,an,entry;,RADIANCE_DS41_IDXKEY)|kv:,the,pool,backs|kv:,the,cache,can,grow,to|VRAM,budget"
[ $PINS = s5 ] && LITS="$LITS|lever,E1,declared,(RADIANCE_DS41_ENGDD"
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": 8, "drafter_mx": 1, "gg_pf": 1, "f32_pf": 25, "dspark_walk": 1, "ds41_argather": 48, "idx_split": 1, "ds41_qnfold": 1, "ds41_swfold": 1, "draft_ctxskip": 1, "dspark_conf": 1, "dspark_conf_rows": 1, "draft_lend": 1, "draft_probe_backoff": 3, "draft_hybrid": 1}'
FIX=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update({"draft_lend_hyst": 0, "arena_d0": 0, "draft_lend_w": 0, "draft_occ_k": 32}); print(json.dumps(d))' "$FIX5") || { echo "pin JSON"; exit 1; }
# the > 48-row check's deterministic drafting overlay (dsm-pskl20.sh's DET): a fixed depth (5), no depth 0 / occupancy / lookup
# hybrid / confidence cut -- exact as every depth choice is (it moves only what a step verifies)
DET='{"draft_adaptive": 0, "draft_depth0": 0, "draft_occ": 0, "draft_hybrid": 0, "dspark_conf": 0}'
armj() {   # armj S|D: the S switches + sched_bi_chunk (D: + DET)
  python3 -c 'import json,sys
d = json.loads(sys.argv[1]); d["sched_bi_chunk"] = int(sys.argv[2])
if sys.argv[3] == "D": d.update(json.loads(sys.argv[4]))
print(json.dumps(d))' "$SPIN" "$BIC" "$1" "$DET"
}
pinj() {   # pinj ARM [EXTRAJSON ...]: the fixed pin + the arm + the extras as one POST body (a later one wins)
  python3 -c 'import json,sys
d = {}
for a in sys.argv[1:]: d.update(json.loads(a or "{}"))
print(json.dumps(d))' "$FIX" "$(armj $1)" "${@:2}"
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
  RADIANCE_DS41_PSK RADIANCE_PLACE_LF RADIANCE_PLACE_LF_DECAY RADIANCE_LOG_STEPS RADIANCE_DS41_ENGDD RADIANCE_DS41_PSK_L20
  RADIANCE_DS41_DSPARK_PFCTX RADIANCE_PLACE_HB_RANKS RADIANCE_PLACE_HB_DECAY"
stackenv() {   # 105's t3 arm + gg_gemv 2 + rule-w + the K1 folds + turnover 0 + LEVENV (+ PEXTRA under s5), every other switch unset
  local kv
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  export RADIANCE_PLACE_TURNOVER=0
  for kv in $LEVENV $PEXTRA; do export "$kv"; done
}
thermline() { type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"; }
cmpfile() {   # cmpfile A B -> IDENTICAL (n chars) | DIFFERENT at char k (na vs nb chars) | not compared
  [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
idrec() {   # idrec NAME RESULT [WHAT]: identity.txt + one RESULT line (a difference reported, not hidden). NAME: S: strict,
  # B48: the > 48-row BI check (strict, reported apart), I: information (cross-start, labelled)
  [ "$DRY" = 1 ] && { echo "+ identity $1 ${3:+($3)}"; return 0; }
  echo "$1 $2" >> $O/identity.txt
  case $2 in IDENTICAL*) xlog "identity $1${3:+ ($3)}: $2" ;; *) xlog "identity $1${3:+ ($3)}: $2 -- reported, not hidden" ;; esac
}
cmpx() {   # cmpx NAME A B: loadid.py cmp (byte for byte + the first differing char / token) -> identity.txt
  local r
  [ "$DRY" = 1 ] && { echo "+ python3 -I $LD cmp --url $URL $2 $3 --json $O/cmp-${1#*:}.json -> identity $1"; return 0; }
  if [ ! -s "$2" ] || [ ! -s "$3" ]; then xlog "$1: not compared (missing $2 or $3)"; echo "$1 not compared" >> $O/identity.txt; return 0; fi
  r=$(timeout 300 python3 -I $LD cmp --url $URL --json $O/cmp-${1#*:}.json --label "$1" "$2" "$3" 2>&1 | grep -m1 '^CMPLINE ' | cut -c9-)
  idrec "$1" "${r:-VIRHE compare failed}" "$(basename $2) vs $(basename $3)"
}
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xkv4 $1 warm-up identity vs $2"; return 0; }
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
modes() {   # modes TAG LAT: MODEWORDS in the container (RADIANCE_DS41_LATENT only under nvfp4), mover 0 kept free x4, rule-w, K1,
  # ARGATHER 48, the latent / key literals of deepseek41's indexer note, every literal of LITS -> 0, else 1
  [ "$DRY" = 1 ] && { echo "+ modes $1: [$MODEWORDS] in the container's env$([ $2 = bf16 ] && echo ", RADIANCE_DS41_LATENT NOT in it"); 'latent $2 ($([ $2 = nvfp4 ] && echo 288 || echo 1024) bytes an entry; RADIANCE_DS41_LATENT)' in the start log; mover '0 more kept free' on 4 ranks; rule-w in every recorded pass; K1 folds =1; ARGATHER 48-row note; literals [$(echo "$LITS" | tr ',' ' ')]"; return 0; }
  local CE kv bad="" ok=1 NW NR KL AL LL nz lit found="" miss="" want
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in RADIANCE_PLACE_TURNOVER=0 $MODEWORDS; do echo "$CE" | grep -qxF "$kv" || bad="$bad ${kv%%=*}"; done
  [ $2 = bf16 ] && echo "$CE" | grep -q '^RADIANCE_DS41_LATENT=' && bad="$bad RADIANCE_DS41_LATENT(set-under-bf16)"
  [ -z "$bad" ] || { xlog "$1: VIRHE start words not as asked:$bad"; ok=0; }
  want="latent $2 ($([ $2 = nvfp4 ] && echo 288 || echo 1024) bytes an entry; RADIANCE_DS41_LATENT)"
  LL=$(docker logs $CN 2>&1 | grep -m1 -o 'latent [a-z0-9]* ([0-9]* bytes an entry; RADIANCE_DS41_LATENT)')
  [ "$LL" = "$want" ] || { xlog "$1: VIRHE the latent mode read back is '$LL', asked '$want'"; ok=0; }
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
  xlog "$1 modes: latent read back '$LL', ${#lit[@]} literal(s), found [$(echo $found)]$([ -n "$miss" ] && echo " -- VIRHE missing:$miss") | $(docker logs $CN 2>&1 | grep -E "kv: the pool backs|kv: the cache can grow|lever PSK declared|lever BI declared|lever E1|chunk geometry" | sed 's/^radiance: //; s/^[A-Z] *//; s/^[A-Z]\[[0-9]\] //' | cut -c1-330 | sort -u | tr '\n' '|' | cut -c1-2600)"
  [ -z "$miss" ] || ok=0
  echo "$1 modes $([ $ok = 1 ] && echo OK || echo VIRHE) | latent '$LL'" >> $O/guard.txt
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
# ---------------------------------------------------------------- copied from dsm-sahb.sh's copies of stack4 / ncprof (the arm read-back, disk, snapshots)
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN | gzip > $O/engine-$1.log.gz"; return 0; }; docker logs $CN 2>&1 | gzip -1 > $O/engine-$1.log.gz || true; }
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
fn4code() { [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
arm() { fn4set "$(pinj $1 "${2:-}" '{"epoch_bump": 1}')"; nudge; }
ARMOK=1
armset() {   # armset TAG ARM: the pin + the arm + epoch_bump, a nudge, every switch READ BACK (else VIRHE, ARMOK 0)
  local g r
  arm $2
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4: every switch of $(pinj $2) applied, nothing pending -> RESULT xkv4 $1 arm $2 applied (read back)"; return 0; }
  g=$(fn4get)
  r=$(python3 -c 'import json,sys
g, w = json.loads(sys.argv[1]), json.loads(sys.argv[2])
bad = {k: g["switches"].get(k) for k, v in w.items() if g["switches"].get(k) != v}
s = g["switches"]
print(("OK" if not bad and not g["pending"] else "FAIL %s pending %s" % (bad, g["pending"])) + " | bi=%s psk=%s/%s bic=%s occ=%s ds=%s adapt=%s d0=%s epoch=%s" % (
    s.get("ds41_bi"), s.get("ds41_psk"), s.get("ds41_psk_k"), s.get("sched_bi_chunk"), s.get("draft_occ"), s.get("draft_sample"), s.get("draft_adaptive"), s.get("draft_depth0"), g.get("epoch")))' "$g" "$(pinj $2)" 2>&1 | cut -c1-400)
  case "$r" in OK*) xlog "$1 arm $2 applied (read back): $r"; return 0 ;; esac
  xlog "$1 arm $2: VIRHE the pin is not applied ($r) -- no reading under it"; echo "$1 pins VIRHE arm $2 $r" >> $O/guard.txt; ARMOK=0; return 1
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
snap() {   # snap NAME: GET /admin/fn4 and the full /stats -> NAME-fn4.json / NAME-st.json
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 > $O/$1-fn4.json; GET /stats > $O/$1-st.json"; return 0; }
  curl -s -m 10 $URL/admin/fn4 > $O/$1-fn4.json 2>/dev/null; curl -s -m 30 $URL/stats > $O/$1-st.json 2>/dev/null
}
capture() {   # capture FILE: the engine log with the daemon's receipt times, gzipped (never written raw)
  run "docker logs -t $CN 2>&1 | gzip -1 > $1"
  [ "$DRY" = 1 ] || xlog "engine log $(basename $1): $(du -m $1 | cut -f1) MB gz, $(zcat $1 | wc -l) lines, $(zcat $1 | grep -cE ' step [0-9]+ (prefill|decode|MIXED) n_seq=') step lines; $(dfree) MB free"
}
now() { [ "$DRY" = 1 ] && echo T || date +%s.%N; }
wmark() { [ "$DRY" = 1 ] || echo "$2:$3:$4" >> $O/$1-wins.txt; }   # wmark TAG NAME T0 T1: a window for the > 48-row count
# ---------------------------------------------------------------- the readings
SPEEDOK=1
startenv() {   # startenv TAG LAT MS HP MML: stack4's dstart serve line at MAXSEQS / HOSTPOOL / MML + RADIANCE_LOG_STEPS=1 + NVG (+ nvfp4)
  local tag=$1 lat=$2 ms=$3 hp=$4 mml=$5 nv=""
  stackenv; export RADIANCE_LOG_STEPS=1
  [ $NVG -gt 0 ] && { export RADIANCE_DS41_NVGATHER=$NVG; nv="RADIANCE_DS41_NVGATHER=$NVG"; }
  [ $lat = nvfp4 ] && export RADIANCE_DS41_LATENT=nvfp4      # the yaml's list passes it from the shell; unset under bf16
  export MODEL=$M225; HP29=$hp; best_vars
  export MNBT=4160 NST=5 MAXSEQS=$ms MML=$mml XARGS="$XARGS --live --draft-window adaptive --draft-coupling on"
  export XS2_ENVFILE=$O/$tag.env STATSDIR=$O/stats-$tag PROFEVERY=
  MODEWORDS="$LEVENV $PEXTRA RADIANCE_LOG_STEPS=1 $nv $([ $lat = nvfp4 ] && echo RADIANCE_DS41_LATENT=nvfp4)"
  MODEWORDS=$(echo $MODEWORDS)
  envfile $XS2_ENVFILE $LEVENV $PEXTRA RADIANCE_LOG_STEPS=1 $nv
  mkd $STATSDIR
}
swstart() {   # swstart TAG: the switches at start (stack4's start values; every switch the pins name present); ds41_psk 99 refused
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 at start: gg_gemv 2, aql_acq 2, K1 three 1, epoch 0, draft_capable 1, ds41_bi 0, ds41_psk 0, sched_bi_chunk 0, every switch of $(pinj D) present; POST ds41_psk 99 -> 400"; return 0; }
  local s0 c1 r
  s0=$(fn4get); c1=$(fn4code '{"ds41_psk": 99}')
  r=$(python3 -c 'import json,sys
s0 = json.loads(sys.argv[1]); w = s0["switches"]; want = json.loads(sys.argv[3])
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) and s0["epoch"] == 0
ok = ok and s0["draft_capable"] == 1 and w.get("ds41_bi") == 0 and w.get("ds41_psk") == 0 and w.get("sched_bi_chunk") == 0
miss = [k for k in want if k not in w]
ok = ok and not miss and sys.argv[2] == "400"
print(("OK" if ok else "FAIL") + " ds41_bi %s ds41_psk %s/%s sched_bi_chunk %s draft_occ %s draft_sample %s | ds41_psk 99 -> HTTP %s%s" % (
    w.get("ds41_bi"), w.get("ds41_psk"), w.get("ds41_psk_k"), w.get("sched_bi_chunk"), w.get("draft_occ"), w.get("draft_sample"), sys.argv[2],
    (" | not in the registry: %s" % miss) if miss else ""))' "$s0" "$c1" "$(pinj D)" 2>&1 | cut -c1-500)
  case "$r" in OK*) xlog "$1 switches at start: $r"; return 0 ;; esac
  xlog "$1 switches at start: VIRHE $r | $(echo "$s0" | cut -c1-900)"; return 1
}
deepr() {   # deepr TAG NAME DEPTH: one cold deepq prompt at DEPTH (prefix-cache reset first), 512 greedy tokens -> deep.jsonl
  local tag=$1 nm=$2 dep=$3 t00=$(date +%s) j
  pcreset $tag-$nm; snap $tag-$nm-0
  run "(cd $D && timeout 1800 python3 $DQ $O/$tag-$nm.txt $dep 512 > $O/$tag-$nm.out 2>&1)"
  snap $tag-$nm-1
  [ "$DRY" = 1 ] && { echo "+ RESULT xkv4 $tag $nm: cold deepq depth $dep: status, prompt, cached, TTFT -> $O/deep.jsonl"; return 0; }
  j=$(grep -m1 '^DEEPQ ' $O/$tag-$nm.out | cut -c7-)
  [ -n "$j" ] || { xlog "$tag $nm: VIRHE no answer ($(tail -2 $O/$tag-$nm.out | tr '\n' ' ' | cut -c1-300))"; return 1; }
  python3 -c 'import json,sys
d = json.loads(sys.argv[1]); d.update(tag=sys.argv[2], name=sys.argv[3], depth=int(sys.argv[4])); print(json.dumps(d))' "$j" $tag $nm $dep >> $O/deep.jsonl
  xlog "$tag $nm (cold deepq depth $dep, 512 greedy tokens, arm S): $(echo "$j" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("status %s, prompt %s tok, cached %s, TTFT %s s (%s prefill tok/s), %s tok" % (d.get("status"), d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), round((d["prompt_tokens"] - (d.get("cached") or 0)) / d["ttft"], 1) if d.get("ttft") and d.get("prompt_tokens") else "-", d.get("tokens")))' 2>&1 | cut -c1-300) | $(thermline $t00 $(date +%s))"
}
guard16() {   # guard16 TAG: PSK on the corrected /stats psk counters over a cold ~16k prompt (before any speed reading)
  local tag=$1 r
  deepr $tag g16 $GDEP
  [ "$DRY" = 1 ] && { echo "+ PSK guard: python3 -I -B $SUM psk --a $O/$tag-g16-0-st.json --b $O/$tag-g16-1-st.json --deepq $O/$tag-g16.out (latched / planned_steps / cut_passes grow, plan_mismatch +0, cold); a failed guard stops the speed readings of this start"; return 0; }
  r=$(timeout 120 python3 -I -B $SUM psk --a $O/$tag-g16-0-st.json --b $O/$tag-g16-1-st.json --deepq $O/$tag-g16.out 2>&1 | tail -1 | cut -c1-500)
  echo "$tag psk $r" >> $O/guard.txt
  case "$r" in OK*) xlog "$tag PSK guard: $r" ;; *) xlog "$tag PSK guard: VIRHE $r -- no speed reading in this start"; SPEEDOK=0 ;; esac
}
u1() {   # u1 TAG K: the 1u stepprobe essay / code (T 0, 512 tok), arm S
  local tag=$1 k=$2
  run "(cd $D && timeout 900 python3 $SPB essay,code 512 > $O/$tag-1u-$k.txt 2>&1)"
  [ "$DRY" = 1 ] || xlog "$tag 1u-$k: $(grep '^J ' $O/$tag-1u-$k.txt | python3 -c 'import json,sys
print(" / ".join("%s %s tok/s %s ms" % (d.get("prompt"), d.get("decode_tps"), d.get("ms_per_decode_step")) for d in (json.loads(l[2:]) for l in sys.stdin)))' 2>/dev/null)"
}
mbw() {   # mbw TAG KIND N K: mb c<N> at temperature 0 (ncmb.py --mode phase = mbt.py's phase_c + taps, /stats at the edges)
  local tag=$1 w=$1-$2-$4 n=$3 t00=$(date +%s) ta tb
  snap $w-0; ta=$(now)
  run "(cd $D && timeout $(( DUR + 1200 )) python3 -I -B $NCMB --mb $MBP --mode phase --temperature 0 --n $n --tag $w --dur $DUR --warm $WARMS --reqs $O/$w-reqs.jsonl --stats-prefix $O/$w-st --samples $O/$w-samp.jsonl > $O/$w.txt 2>&1)"
  tb=$(now); snap $w-1; wmark $tag $w $ta $tb
  [ "$DRY" = 1 ] || xlog "$tag $w (mb c$n, T 0, dur $DUR warm $WARMS, arm S): $(grep -m1 '^MB ' $O/$w.txt | cut -c4- | cut -c1-500) | $(thermline $t00 $(date +%s))"
}
galone() {   # galone TAG: arm S read back, prefix-cache reset, greedyset.py's 20 prompts at c1 ALONE, temperature 0
  local tag=$1 t00=$(date +%s)
  armset $tag S || return 1
  pcreset $tag-a1
  run "(cd $D && timeout 900 python3 $GS $URL $O/$tag-a1.jsonl --conc 1 --max-tokens $MT > $O/$tag-a1.txt 2>&1)"
  [ "$DRY" = 1 ] || xlog "$tag a1 (greedy set c1 ALONE, T 0, $MT tok, arm S): $(tail -1 $O/$tag-a1.txt | cut -c1-200), $(( $(date +%s) - t00 )) s"
}
gsconc() {   # gsconc TAG NAME N: arm S read back, prefix-cache reset, the greedy set at conc N (the decode rows of N streams)
  local tag=$1 t00=$(date +%s) ta tb
  armset $tag S || return 1
  pcreset $tag-$2; ta=$(now)
  run "(cd $D && timeout 900 python3 $GS $URL $O/$tag-$2.jsonl --conc $3 --max-tokens $MT > $O/$tag-$2.txt 2>&1)"
  tb=$(now); wmark $tag $2 $ta $tb
  [ "$DRY" = 1 ] || xlog "$tag $2 (greedy set at conc $3, T 0, $MT tok, arm S): $(tail -1 $O/$tag-$2.txt | cut -c1-200), $(( $(date +%s) - t00 )) s"
}
ldx() {   # ldx TAG: c4 decode streams + ONE cold ~64k deepq prompt (stack4/pskld.py --load), 512 greedy tokens, arm S (D3)
  local tag=$1 nm=ld t00=$(date +%s) j
  armset $tag S || return 1
  pcreset $tag-$nm
  run "(cd $D && timeout 2700 python3 -I $PL run --url $URL --tests $TESTS --mb $MBP --which deep --depth 64000 --ntok 512 --load --conc 4 --warm 10 --pf-tok 0 --out $O/$tag-$nm.json --text $O/$tag-$nm.txt > $O/$tag-$nm.log 2>&1)"
  [ "$DRY" = 1 ] && return 0
  j=$(grep -m1 '^LONGID ' $O/$tag-$nm.log | cut -c8-)
  [ -n "$j" ] || { xlog "$tag $nm: VIRHE no LONGID line ($(tail -3 $O/$tag-$nm.log | tr '\n' ' ' | cut -c1-400))"; return 1; }
  python3 -c 'import json,sys
d = json.loads(sys.argv[1]); d.update(tag=sys.argv[2], name=sys.argv[3]); print(json.dumps(d))' "$j" $tag $nm >> $O/load.jsonl
  xlog "$tag $nm (c4 + one cold ~64k deepq prompt, T 0, arm S = sched_bi_chunk $BIC): $(python3 -c 'import json,sys
d = json.loads(sys.argv[1])
print("probe status %s, prompt %s tok, cached %s, TTFT %s s | prefill window %s s: c4 agg %s tok/s, per-stream p50 %s, TPOT p50 %s / p90 %s ms | c4 %s requests, %s errors" % (
    d.get("status"), d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("pf_win_s"), d.get("c4_pf_agg_tps"), d.get("c4_pf_stream_tps_p50"),
    d.get("tpot_pf_p50"), d.get("tpot_pf_p90"), d.get("c4_requests"), d.get("c4_errors")))' "$j" 2>&1 | cut -c1-600) | $(thermline $t00 $(date +%s))"
}
lid16() {   # lid16 TAG: the > 48-row BI check -- arm D (S + DET) read back, each greedy prompt under LIDC mb streams + one fresh
  # ~LIDPF newcomer at +0.5 s beside its decode (stack4/loadid.py), the window marked for the step-log count, then arm S back
  local tag=$1 t00=$(date +%s) j ta tb
  armset $tag D || return 1
  pcreset $tag-lid16; ta=$(now)
  run "(cd $D && timeout 2700 python3 -I $LD run --url $URL --gs $GS --mb $MBP --out $O/$tag-lid16.jsonl --conc $LIDC --warm 10 --offset 0.5 --pf-tok $LIDPF --max-tokens $MT > $O/$tag-lid16.txt 2>&1)"
  tb=$(now); wmark $tag lid16 $ta $tb
  armset $tag S
  [ "$DRY" = 1 ] && return 0
  j=$(grep -m1 '^LOADID ' $O/$tag-lid16.txt | cut -c8-)
  [ -n "$j" ] || { xlog "$tag lid16: VIRHE no LOADID line ($(tail -3 $O/$tag-lid16.txt | tr '\n' ' ' | cut -c1-400))"; return 1; }
  xlog "$tag lid16 (each greedy prompt under $LIDC mb streams + one fresh ~$LIDPF newcomer at +0.5 s, T 0, arm D = S + DET: depth 5 fixed): $(python3 -c 'import json,sys
d = json.loads(sys.argv[1])
print("%s of %s answered, probe %s s mean | newcomers %s of %s cold, TTFT %s s mean | mixed steps a probe min %s / median %s; probes without one %s | streams %s requests, %s errors" % (
    d["answered"], d["of"], d["probe_s_mean"], d["pf_cold"], d["of"], d["pf_ttft_mean"], d["mixed_min"], d["mixed_median"],
    d["probes_without_mixed"] or "none", d["c4_requests_done"], d["c4_errors"]))' "$j" 2>&1 | cut -c1-600) | BI stats now: $(docker logs $CN 2>&1 | grep 'lever BI stats' | tail -1 | sed 's/^radiance: deepseek41: //' | cut -c1-220) | $(( $(date +%s) - t00 )) s"
}
idstart() {   # idstart TAG: this start's compares while its server is up (loadid.py cmp tokenizes the first differing token)
  local t=$1 x
  cmpx S:a1-$t-vs-d0 $RD/dsd-d0-gs-c1.jsonl $O/$t-a1.jsonl
  for x in $DONE; do cmpx S:a1-$t-vs-a1-$x $O/$x-a1.jsonl $O/$t-a1.jsonl; done
  case $t in
    b8|n8)
      cmpx S:gs8-$t-vs-a1-$t $O/$t-a1.jsonl $O/$t-gs8.jsonl
      for x in $DONE; do case $x in b8|n8) cmpx S:gs8-$t-vs-gs8-$x $O/$x-gs8.jsonl $O/$t-gs8.jsonl
        idrec S:ld-$t-vs-ld-$x "$(cmpfile $O/$x-ld.txt $O/$t-ld.txt)" "the ~64k probe under c4 load, sched_bi_chunk $BIC" ;; esac; done
      idrec S:ld-$t-vs-stack4-alone64k "$(cmpfile $RS4/s4-n5-d64000-2S.txt $O/$t-ld.txt)" "loaded under D$BIC vs stack4's alone 64k S answer (int7, cross-start alone reference)"
      idrec I:ld-$t-vs-stack4-ldS "$(cmpfile $RS4/s4-n5-ld-2S.txt $O/$t-ld.txt)" "cross-start: stack4's loaded S at sched_bi_chunk 0 (a known class (ii) difference), labelled" ;;
    n16|b16)
      cmpx B48:gs16-$t-vs-a1-$t $O/$t-a1.jsonl $O/$t-gs16.jsonl
      cmpx B48:lid16-$t-vs-a1-$t $O/$t-a1.jsonl $O/$t-lid16.jsonl
      for x in $DONE; do case $x in b16|n16) cmpx B48:gs16-$t-vs-gs16-$x $O/$x-gs16.jsonl $O/$t-gs16.jsonl
        cmpx B48:lid16-$t-vs-lid16-$x $O/$x-lid16.jsonl $O/$t-lid16.jsonl ;; esac; done ;;
  esac
  [ -n "$REF16" ] && [ "$REF16" != $t ] && idrec S:g16-$t-vs-g16-$REF16 "$(cmpfile $O/$REF16-g16.txt $O/$t-g16.txt)" "cold 16k answer vs the first start's"
  if [ -e $O/$t-d136.txt ] || [ "$DRY" = 1 ]; then
    case $t in b8|n8|x8|m8)
      idrec S:d136-$t-vs-dsm225 "$(cmpfile $RM/deep-dsm-225.txt $O/$t-d136.txt)" "136k answer vs dsm-225 NST 0 stock cold"
      [ -n "$REF136" ] && [ "$REF136" != $t ] && idrec S:d136-$t-vs-d136-$REF136 "$(cmpfile $O/$REF136-d136.txt $O/$t-d136.txt)" "136k answer vs the first start's" ;;
    esac
  fi
  return 0
}
kparse() {   # kparse TAG: ncsteps.py c8 over every mb window, the > 48-row counts over every marked window
  local tag=$1 w
  for w in $(cd $O 2>/dev/null && ls $tag-c8-[0-9].txt $tag-c16-[0-9].txt 2>/dev/null | sed 's/\.txt$//') $([ "$DRY" = 1 ] && echo "$tag-cN-k"); do
    pyx $w python3 -I -B $NCS c8 --log $O/$tag-engine.log.gz --reqs $O/$w-reqs.jsonl --tag $w --sa $O/$w-st-w0.json --sb $O/$w-st-w1.json --samples $O/$w-samp.jsonl --mbline $O/$w.txt --label "$w (lever KV4 $tag)" --json $O/$w.json
  done
  pyx $tag-bi48 python3 -I -B $SUM bi48 --log $O/$tag-engine.log.gz $( [ -s $O/$tag-wins.txt ] && sed 's/^/--win /' $O/$tag-wins.txt | tr '\n' ' ') --nst 5 --json $O/$tag-bi48.json
}
scfg() {   # scfg TAG -> "KIND LAT MAXSEQS HOSTPOOL MML"
  case $1 in
    b8) echo "M8 bf16 8 $HP 524288" ;;     n8) echo "M8 nvfp4 8 $HP 524288" ;;
    x8) echo "X bf16 8 $XHP 524288" ;;     m8) echo "X bf16 8 $HP $MMLX" ;;
    b16) echo "M16 bf16 16 $HP 524288" ;;  n16) echo "M16 nvfp4 16 $HP 524288" ;;
  esac
}
est() { case $1 in M8) echo 13 ;; X) echo 11 ;; M16) echo 16 ;; esac; }
DONE=""; REF16=""; REF136=""
kstart() {   # kstart TAG: one start, its readings, its compares, its parse
  local tag=$1 kind lat ms hp mml el e k r
  read -r kind lat ms hp mml <<< "$(scfg $tag)"
  e=$(est $kind); el=$(( ($(date +%s) - T0) / 60 ))
  if [ "$DRY" != 1 ] && [ $((el + e)) -gt $BUDGET ]; then xlog "$tag SKIPPED: $el min elapsed + ~$e min > the budget $BUDGET min"; return 0; fi
  if [ "$hp" = auto ]; then
    if [ "$DRY" = 1 ]; then echo "+ python3 -I -B $SUM hp --b $O/b8-cap.json --n $O/n8-cap.json --hp $HP -> HOSTPOOL for $tag (HP - (n8's experts - b8's) MiB)"; hp=HPX
    else
      r=$(timeout 60 python3 -I -B $SUM hp --b $O/b8-cap.json --n $O/n8-cap.json --hp $HP 2>&1 | grep -m1 '^HP ')
      hp=$(echo "$r" | awk '{print $2}')
      case "$hp" in ''|*[!0-9]*) xlog "$tag: VIRHE no expert-heavy HOSTPOOL ($r) -- not started"; return 1 ;; esac
      xlog "$tag: HOSTPOOL $hp MiB -- ${r#HP $hp }"
    fi
  fi
  diskok $tag || return 1
  SPEEDOK=1
  startenv $tag $lat $ms $hp $mml
  xlog "$tag: start (~$e min; latent $lat, MAXSEQS $ms, HOSTPOOL $hp MiB, MML $mml; stack4's NST 5 start + RADIANCE_LOG_STEPS=1 + NVG $NVG$([ -n "$PEXTRA" ] && echo " + $PEXTRA")) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $FIX + S $(armj S) | env: $(envline)"
  gate1 $tag 70 || { xlog "$tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  modes $tag $lat || { xlog "$tag: VIRHE the modes are not the start's -> stopped"; down; return 1; }
  drafted $tag || { keeplog $tag-nodraft; down; return 1; }
  swstart $tag || { keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  armset $tag S || { keeplog $tag-pin; down; return 1; }
  snap $tag-st0
  pyx $tag-cap python3 -I -B $SUM cap --log $O/start-$tag.log --stats $O/$tag-st0-st.json --tag $tag --lat $lat --ms $ms --mml $mml --hp $hp --json $O/$tag-cap.json
  stopfile || guard16 $tag
  [ -n "$REF16" ] || REF16=$tag
  if [ $SPEEDOK = 1 ]; then
    case $kind in
      M8) for k in $(seq 1 $U1N); do stopfile || u1 $tag $k; done
          for k in $(seq 1 $C8N); do stopfile || mbw $tag c8 8 $k; done
          stopfile || deepr $tag d136 135000 ;;
      X)  stopfile || u1 $tag 1
          for k in $(seq 1 $C8N); do stopfile || mbw $tag c8 8 $k; done
          stopfile || deepr $tag d136 135000 ;;
      M16) for k in $(seq 1 $C16N); do stopfile || mbw $tag c16 16 $k; done
           stopfile || mbw $tag c8 8 1 ;;
    esac
  else
    xlog "$tag: VIRHE the PSK guard failed (above): no speed reading in this start -- identity only"
  fi
  [ -z "$REF136" ] && { [ -s $O/$tag-d136.txt ] || [ "$DRY" = 1 ]; } && case $kind in M8|X) REF136=$tag ;; esac
  stopfile || galone $tag
  case $kind in
    M8) stopfile || gsconc $tag gs8 8; stopfile || ldx $tag ;;
    M16) stopfile || gsconc $tag gs16 16; stopfile || lid16 $tag ;;
  esac
  idstart $tag
  [ "$DRY" = 1 ] || xlog "$tag: $(docker logs $CN 2>&1 | grep -cE '^E ') E lines | PSK $(curl -s -m 10 $URL/stats 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("psk"))' 2>/dev/null | cut -c1-300) | BI $(docker logs $CN 2>&1 | grep 'lever BI stats' | tail -1 | sed 's/^radiance: deepseek41: //' | cut -c1-200)"
  capture $O/$tag-engine.log.gz
  down
  kparse $tag
  DONE="$DONE $tag"
  xlog "$tag: done ($(( ($(date +%s) - T0) / 60 )) min into the item)"
}
report() {
  run "python3 -I -B $SUM report --dir $O --starts '$STARTS' > $O/report.txt 2>&1"
  [ "$DRY" = 1 ] && { echo "+ every RESULT line of $O/report.txt -> the log (capacity, MAXSEQS 8 / 12 / 16, smaller-pool prices, speed, pairs, the c16 cell, identity, guards)"; return 0; }
  while IFS= read -r l; do case "$l" in "RESULT xkv4 "*) log "${l:0:3800}" ;; esac; done < $O/report.txt
  local nv; nv=$(( $(grep -c VIRHE $O/report.txt 2>/dev/null) + NV ))
  [ $ARMOK = 1 ] || nv=$((nv + 1))
  log "RESULT xkv4 guards (item): $([ "$nv" = 0 ] && echo "OK (latent read back in every start, pins read back, PSK counters grew, windows clean, step-log lines)" || echo "VIRHE in $nv place(s) -- see the lines above; a reading with VIRHE is not a reading") | starts done:$DONE of [$STARTS]"
}
trap 'down' EXIT
trap 'exit 143' TERM INT
# ---------------------------------------------------------------- main
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) pins $PINS | lever KV4 (RADIANCE_DS41_LATENT=nvfp4, NVG $NVG every start) | starts [$STARTS] (b8 / n8 bf16 / nvfp4 MAXSEQS 8, x8 bf16 at n8's expert VRAM, n16 / b16 MAXSEQS 16 = the c16 cell + the > 48-row BI check, m8 bf16 MML $MMLX) | pin S $(armj S) | mb T 0 dur $DUR warm $WARMS, c8 x $C8N, c16 x $C16N, 1u x $U1N | ~83 min (budget $BUDGET)"
IMGLITS="B|dspark_conf_rows, gg_pf, f32_pf, dspark_walk, ds41_bi, draft_sample, draft_lend_hyst, arena_d0, draft_lend_w, draft_occ, draft_occ_k, place_lf, sched_bi_chunk, sched_interleave, sched_mixed_tok,
B|ds41_psk
B|reset_prefix_cache
B|/tokenize
B|radiance:engine_steps_mixed_total
B|radiance:engine_step_seconds_decode_total
B|radiance:draft_accepted_total
A|RADIANCE_DS41_PSK
A|lever PSK declared (RADIANCE_DS41_PSK, fn4 ds41_psk, now
B|plan_mismatch
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
B|occ_cycles
B|RADIANCE_DRAFT_LEND_WEIGHTS
B|lever Z0-A set up
B|RADIANCE_DRAFT_CTXSKIP_SLOTS
B|ds41_psk_k
A|latent %s (%lld bytes an entry; RADIANCE_DS41_LATENT)
A|RADIANCE_DS41_LATENT
A|latent_format
K|latent_format
K|RADIANCE_DS41_NVGATHER
A|lever BI stats: %lld segmented steps
A|steps with > 48 decode rows
B|RADIANCE_LOG_STEPS
B|n_seq=%lld n_tok=%lld
B|  [%d] slot=%d q=%d ctx=%d
B|kv: the pool backs %.0f%% of the worst case
B|kv: the cache can grow to %s a card
B|VRAM budget, per card"
[ $PINS = s5 ] && IMGLITS="$IMGLITS
X|sched_noa
X|ds41_psk_l20
X|dspark_pfctx
X|ds41_engdd
A|RADIANCE_DS41_ENGDD"
docker image inspect $IMAGE > /dev/null 2>&1 || { xlog "VIRHE image $IMAGE not on this host"; exit 1; }
IMGMISS=$(echo "$IMGLITS" | timeout 120 docker run --rm -i --network none --entrypoint sh -e PINS=$PINS $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels/libr4d.so; S=/opt/radiance/share/radiance/kernels/spechub.so; n=0
while IFS= read -r l; do [ -n "$l" ] || continue; n=$((n + 1)); w=${l%%|*}; s=${l#*|}
  case $w in B) f=$B ;; A) f=$A ;; K) f=$K ;; S) f=$S ;; *) f="$B $A $K" ;; esac
  o=0; for x in $f; do grep -qzF -- "$s" $x && { o=1; break; }; done
  [ $o = 1 ] || echo "MISS $w|$s"; done
test -x /opt/radiance/bin/rad-sidecar || echo "MISS bin/rad-sidecar"
if [ "$PINS" = s4 ]; then test -s /opt/radiance/share/radiance/int7-PROVENANCE.txt || echo "MISS int7-PROVENANCE.txt"
else ls /opt/radiance/share/radiance/*PROVENANCE.txt > /dev/null 2>&1 || echo "MISS a PROVENANCE file"; fi; echo "CHECKED $n"' 2>&1)
case "$IMGMISS" in "CHECKED "*) ;; *) xlog "VIRHE $IMAGE does not carry what this item reads (pins $PINS) -- missing: $(echo "$IMGMISS" | grep -v '^CHECKED' | tr '\n' ';' | cut -c1-900)"; exit 1 ;; esac
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)), pins $PINS: all $(echo "$IMGMISS" | sed 's/CHECKED //') literals present (stack4's registry / BI / SD / DLW / Z0 / PSK + lever KV4's latent note and latent_format, NVG's knob, the BI stats line, the step log, the pool notes$([ $PINS = s5 ] && echo ", the S5 switches")) + bin/rad-sidecar + PROVENANCE -- OK"
HMISS=""
for f in "$LIB29" "$LIBPR" $Y $GS $LD $PL $DQ $MBP $TESTS $SPB $D/fastcheck.py $NCS $NCMB $NCONE $MBT $SUM; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $GS $LD $PL $DQ $MBP $TESTS $SPB $D/fastcheck.py $NCS $NCMB $NCONE $MBT $SUM 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
ST1=$(timeout 60 python3 -I $LD selftest --gs $GS --mb $MBP 2>&1 | tail -1); case "$ST1" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$LD(${ST1:0:300})" ;; esac
ST2=$(timeout 60 python3 -I $PL selftest --tests $TESTS --mb $MBP 2>&1 | tail -1); case "$ST2" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$PL(${ST2:0:300})" ;; esac
ST3=$(timeout 60 python3 -I -B $MBT --mb $MBP --selftest 2>&1 | tail -1); case "$ST3" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$MBT(${ST3:0:300})" ;; esac
ST4=$(timeout 60 python3 -I -B $NCMB --mb $MBP --selftest 2>&1 | tail -1); case "$ST4" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$NCMB(${ST4:0:300})" ;; esac
ST5=$(timeout 60 python3 -I -B $NCONE --mb $MBP --selftest 2>&1 | tail -1); case "$ST5" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$NCONE(${ST5:0:300})" ;; esac
ST6=$(timeout 120 python3 -I -B $NCS selftest 2>&1 | tail -1); case "$ST6" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$NCS(${ST6:0:300})" ;; esac
ST7=$(timeout 60 python3 -I -B $SUM selftest 2>&1 | tail -1); case "$ST7" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$SUM(${ST7:0:300})" ;; esac
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
FNS="log run mkd restart check down stopfile best_vars desc serve_args xlog armj pinj stackenv thermline cmpfile idrec cmpx identw modes keeplog fn4get fn4set fn4code nudge arm armset gate1 dfree diskok envfile envline drafted pyx pcreset snap capture now wmark startenv swstart deepr guard16 u1 mbw galone gsconc ldx lid16 idstart kparse scfg est kstart report"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee sha256sum gzip zcat df du seq; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
for a in S D; do python3 -c 'import json,sys
d = json.loads(sys.argv[1]); a = sys.argv[2]; s = json.loads(sys.argv[3])
det = json.loads(sys.argv[5]) if a == "D" else {}
ok = d["gg_gemv"] == 2 and d["epoch_bump"] == 1 and d["sched_bi_chunk"] == int(sys.argv[4]) and all(d[k] == v for k, v in s.items() if k != "sched_bi_chunk" and k not in det)
ok = ok and ((d["draft_adaptive"] == 0 and d["draft_depth0"] == 0 and d["draft_occ"] == 0) if a == "D" else (d["draft_adaptive"] == 1 and d["draft_depth0"] == 1))
sys.exit(0 if ok else 1)' "$(pinj $a '{"epoch_bump": 1}')" $a "$SPIN" "$BIC" "$DET" 2>/dev/null || HMISS="$HMISS json:$a"; done
[ "$(sha256sum $MODELDIR/$SIDECAR 2>/dev/null | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256 2>/dev/null)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR"
[ "$(sha256sum $MODELDIR/$SIDE 2>/dev/null | cut -c1-64)" = "$SIDESHA" ] || HMISS="$HMISS sha256:$MODELDIR/$SIDE(want ${SIDESHA:0:16})"
for w in XS2_ENVFILE 'RADIANCE_DS41_LATENT:' '${STATSDIR:?set STATSDIR}:/stats'; do grep -qF "$w" $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no '$w')"; done
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$O/stats ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config refuses it)"
HMISS=$(echo $HMISS)
[ -z "$HMISS" ] || { xlog "VIRHE helpers / inputs do not resolve: $HMISS"; exit 1; }
HMD5="ncsteps.py $(md5sum < $NCS | cut -c1-8) (copy of nc/ncsteps.py $(md5sum < $HD/../nc/ncsteps.py 2>/dev/null | cut -c1-8)), ncmb.py $(md5sum < $NCMB | cut -c1-8), ncone.py $(md5sum < $NCONE | cut -c1-8), mbt.py $(md5sum < $MBT | cut -c1-8) (copy of stack4/mbt.py $(md5sum < $HD/../stack4/mbt.py 2>/dev/null | cut -c1-8)), kv4sum.py $(md5sum < $SUM | cut -c1-8), stack4/loadid.py $(md5sum < $LD | cut -c1-8), stack4/pskld.py $(md5sum < $PL | cut -c1-8), greedyset.py $(md5sum < $GS | cut -c1-8), stepprobe.py $(md5sum < $SPB | cut -c1-8), deepq.py $(md5sum < $DQ | cut -c1-8), mb.py $(md5sum < $MBP | cut -c1-8), lib-29.sh $(md5sum < $LIB29 | cut -c1-8), yaml $(md5sum < $Y | cut -c1-8)"
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing; stack4/loadid.py $ST1 | stack4/pskld.py $ST2 | mbt.py $ST3 | ncmb.py $ST4 | ncone.py $ST5 | ncsteps.py $ST6 | kv4sum.py $ST7 | deepq.py = pskld.py's deep request ($DQP) | all $(echo $FNS | wc -w) functions resolve, tools on PATH, sidecars $SIDECAR + $SIDE sha256 OK, yaml accepted (env_file, RADIANCE_DS41_LATENT, /stats mount) -- OK | md5: $HMD5"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { xlog "VIRHE $M225 missing or not verified"; exit 1; }
  for f in $WARM_REF_225 $RD/dsd-d0-gs-c1.jsonl $RM/deep-dsm-225.txt $RS4/s4-n5-d64000-2S.txt $RS4/s4-n5-ld-2S.txt $RS4/guard-s4-n5.txt; do [ -s $f ] || xlog "note -- reference $f missing (not compared)"; done
  xlog "$ITEM: helpers md5: $HMD5"
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDE $WARM_REF_225 $RD/dsd-d0-gs-c1.jsonl $RM/deep-dsm-225.txt $RS4/s4-n5-d64000-2S.txt $RS4/s4-n5-ld-2S.txt $RS4/guard-s4-n5.txt; do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  for a in S D; do echo "+ arm $a: $(armj $a) | POST body $(pinj $a '{"epoch_bump": 1}')"; done
  for s in $STARTS; do echo "+ start $s: $(scfg $s | awk '{print "kind " $1 ", latent " $2 ", MAXSEQS " $3 ", HOSTPOOL " $4 " MiB, MML " $5}') (~$(est $(scfg $s | cut -d" " -f1)) min)"; done
  echo "+ docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(information here; the real run refuses to start beside one)"
fi
mkd $O $O/guard
[ "$DRY" = 1 ] || { : > $O/identity.txt; : > $O/guard.txt; }
for s in $STARTS; do stopfile && break; kstart $s; done
report
unset $SWITCHES
down
xlog "$ITEM: done in $(( ($(date +%s) - T0) / 60 )) min -- $O ($(dfree) MB free)"
