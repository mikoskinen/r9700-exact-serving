#!/bin/bash
# dsm-cc.sh (12.11.2026, queue-ar/dsmain/cc, ds-main-port cc builder) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzz-cc.sh. Log tag "RESULT xcc ", out runs/xdsm/cc (XCC_OUT). USER-ASKED: the EXISTING Claude Code
# simulation (cc_sim) under a bf16 vs an nvfp4 compressed latent. NO new harness: the load is queue-ar/tools/ccsim/cc_sim3.py
# (2512bb16) run through the kit's own wrapper ccsim-exl3-turns.sh (0e00acd2: its "RESULT cc_sim" line, /metrics deltas); this body
# only starts the servers (kv4's b8 / n8 start functions, COPIED from kv4/dsm-kv4.sh f89a6cc2 -- the start words, LITS / FIX, stackenv,
# modes, armset, gate1, swstart, the image / helper / sidecar / yaml checks -- and stack5's S5T pin from stack5/dsm-stack5.sh 63649414;
# originals unchanged), samples /stats and parses: ccsum.py (sampler + readings + report; loads cc_sim3.py READ ONLY for its
# summarize_turns / pct and its offline session builder) + ncsteps.py / ncone.py (unchanged copies of nc/ncsteps.py 0a6e0ed2 /
# nc/ncone.py a24570f8: the step log; ncsteps.py loads ncone.py)
# + kv4/kv4sum.py READ ONLY (d4acd54b: the start's capacity line, "cap").
# THE LOAD: cc_sim3.py --users XCC_USERS (8 = MAXSEQS, memory rule: size the server to the test load) --turns XCC_TURNS (32) a user,
#   the first XCC_WTURNS (3) excluded from the kept readings, --gen-mode natural (the model stops at its own tool call; forced mode's
#   junk tokens collapse draft acceptance -- LEVERS / the cc_sim header), --temperature 0, --trace-seeded (each turn's generation
#   cap / tool result from a stream of its own (seed, session, turn), so a reply that differs between runs cannot shift later
#   turns' token counts), --cpt-fixed, --seed XCC_SEED (1), --effort xhigh,medium, --duration XCC_CAP (1500 s) as the safety cap
#   (drain 180 s). THE SAME seed, users, turns and context settings in every start: the trace set is cc_sim's own seed-1 sessions
#   (default context settings: start 20-40k, preage on = each user's first session pre-filled to U(start, 0.95 x 170000), cap 170000
#   then a new 20-40k session, 109 tools), the set every earlier DS / FN cc run used (x79 u4, 118 / ccrep u16, best-s8 u8). On THIS
#   model's tokenizer its first sessions are 37.0 / 33.5 / 89.2 / 67.1 / 98.9 / 118.4 / 46.5 / 141.9k tokens (u0..u7, server-counted
#   in runs/ccsim-1106-0531-best-s8-u8.jsonl, the same seed / users; cc_sim3's own builder, offline: 38.2 / 36.0 / 94.9 / 73.1 / 99.1 /
#   123.8 / 46.7 / 145.6k at 3.5 chars a token) = 632k tokens live at the start, growing ~1-2k a user turn (x79: u0 37 -> 60k in 15).
#   No recorded replay: the only recorded reference on gpu02 (runs/ccsim-1110-0115-fn4-ccrep-ref-u16) is a Flash-Next u16 trace.
# STARTS (XCC_ORDER; default "b8 n8" = A / B ONCE, b8 first -- labelled as such, order / drift not cancelled; "n8 b8 b8 n8" = B A A B
#   with FRESH starts): b8 = latent bf16, n8 = RADIANCE_DS41_LATENT=nvfp4; both MAXSEQS 8, HOSTPOOL 19456, MML 524288 (each the
#   planner's own split: KV4 measured b8 kv 7.06 GiB = 2.66 M tokens, n8 3.82 GiB = 4.20 M tokens, n8's experts +3.48 GiB), MNBT 4160,
#   NST 5, kv4's start words (stack4's LEVENV + RADIANCE_DS41_ENGDD=0 + RADIANCE_LOG_STEPS=1 + RADIANCE_DS41_NVGATHER=48 (no scratch,
#   no launch under bf16: the starts differ in RADIANCE_DS41_LATENT alone)), --live --draft-window adaptive --draft-coupling on,
#   RADIANCE_DS41_IDXKEY=mxfp4 in both. PIN = stack4's fixed pin + stack5's S5T: ds41_bi 2, draft_sample 85, draft_occ 6, place_lf 0,
#   ds41_psk 4 (tail-only) k 8, ds41_psk_cr 1, ds41_psk_l20 1, dspark_pfctx 1, ds41_engdd 1, sched_noa 2, sched_bi_chunk 3 (+ stack5's
#   explicit stage_short / stage_pick / place_hb 0). LEVER KVR ON in BOTH starts (coordinator 12.11.: KVR 19 / 19 identical, the host
#   tier 32.7 M tokens vs 8.2 M -- so the long conversations' hit rates reflect the shipped capacity): RADIANCE_DS41_KV_REPLICATED=1
#   through the env file, as kvr/dsm-kvr.sh passes it (a declare-time env, no fn4 switch); XCC_KVR auto = 1 on a ds41-int9 image, 0
#   else. Read back: the container env + the start log's "lever KVR declared (RADIANCE_DS41_KV_REPLICATED=1)" and tier line "held once
#   a slot: X a slot rather than Y" (bf16: X must be 5.39 KiB, kvr's B / V; nvfp4: X logged -- KVR never ran beside the packed latent,
#   so its slot size there is a reading, not a guard). int9's SP stays off: fn4 stage_full_min / stage_full_pick 0 (read at start).
# PER START: disk >= XCC_DISKMIN_MB -> gate1 (restart + warm-up + fastcheck) -> modes (start words in the container, the latent mode
#   READ BACK from the start log 'latent bf16 (1024 bytes' / 'latent nvfp4 (288 bytes', literals) -> drafter up -> switches at start ->
#   warm-up answers vs stock -> the pin POSTed + epoch_bump + nudge and READ BACK -> /stats + capacity (kv4sum cap) -> prefix-cache
#   reset -> the /stats sampler (ccsum.py sample, every XCC_SAMP 10 s) + cc_sim3 through ccsim-exl3-turns.sh -> pins read back again
#   -> E lines, the engine log (docker logs -t, gz) -> down -> ccsum.py run.
# READINGS a run: TTFT p50 / p90 (kept turns), decode tok/s a stream (p50 / p90) and aggregate, completed work a minute (requests,
#   generated and uncached-prompt tokens; kept window and whole run), errors / aborted / preemptions; the context distribution;
#   KV pool fill % over time (kv_tokens_used / kv_tokens_total), expert units in VRAM a rank and kv-lent GiB a rank over time (+ a
#   per-minute timeline), streamed GB a non-decode step (least squares over the sample intervals: link.stream_bytes = a x non-decode
#   + b x decode steps) and a newcomer (a x the step log's prefill-run steps), the steps by class; lever counters (information).
#   Report: one table line a run, the n8 vs b8 pair(s) in %, with B A A B the mode means; cc_sim3 --pair per adjacent pair
#   (information). NO identity claims (natural mode); nvfp4's exactness is KV4's result (identity all IDENTICAL).
# GUARDS (VIRHE): the latent mode read back; the pin read back (before and after the cc run); the trace reaching long contexts (max
#   server-counted prompt >= XCC_MINCTX 50000); a cc run with records and kept answers; /stats samples and step-log lines; disk.
# ~53 min A / B (estimate; ~25.5 a start = start + checks ~5, cc ~19 (32 turns: ~3.5 min of cold preaged prefills (632k tokens) +
#   31 turns x ~30 s), capture / down / parse ~1.5; + preflight / report ~2). The cap bounds a start at ~34.5 min (cc 1500 + 180 s
#   drain) -> <= ~71 min. B A A B (XCC_ORDER="n8 b8 b8 n8"): ~1 h 45 min, <= ~2 h 20 min. Budget XCC_BUDGET_MIN (36 a start + 5): a
#   start that would end past it (XCC_EST_MIN 26) is skipped and logged.
# NO evict, no power / host / fan settings; production compose files read only (--project-directory). No API key is passed or printed.
# Knobs: XCC_IMAGE, XCC_ITEM, XCC_OUT, XCC_ORDER ("b8 n8" | "n8 b8 b8 n8" | any list of b8 / n8), XCC_USERS (= MAXSEQS), XCC_TURNS (32),
#   XCC_WTURNS (3), XCC_SEED (1), XCC_CAP (1500), XCC_EFFORT (xhigh,medium), XCC_CCX (more cc_sim3 args, e.g. --record; never the mode /
#   temperature / seed / turns / users), XCC_SAMP (10), XCC_MINCTX (50000), XCC_MAXSEQS (8), XCC_MML (524288), XCC_HP (19456),
#   XCC_NVG (48), XCC_SPIN (the pin's switches as JSON; default S5T), XCC_S5ENV ("RADIANCE_DS41_ENGDD=0"), XCC_SEXTRA (more start
#   words; default none), XCC_KVR (auto | 0 | 1), XCC_DISKMIN_MB (11000), XCC_EST_MIN (26), XCC_BUDGET_MIN, XCC_SIDECAR,
#   XCC_SIDESHA.
# DRYRUN=1 bash dsm-cc.sh : prints every command; checks the image literals, every helper (files, parse, selftests, cc_sim3 offline:
#   --help and its session builder = the trace check), the wrapper, the sidecars, the yaml, the disk; no server, no GPU, no request
#   to :8090, writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/../ds41-stack2-tp4.yaml                   # stack2's yaml, read only (env_file ${XS2_ENVFILE} carries the start words)
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh
O=${XCC_OUT:-$R/xdsm/cc}
RS4=$R/xdsm/stack4
export IMAGE=${XCC_IMAGE:?set XCC_IMAGE}
ITEM=${XCC_ITEM:-57-p4zyzzzzz-ds-r00zzz-cz-x-kvrzzz-cc}
ORDER=$(echo ${XCC_ORDER:-b8 n8}); NRUN=$(echo $ORDER | wc -w)
MS=${XCC_MAXSEQS:-8}; MMLV=${XCC_MML:-524288}; HP=${XCC_HP:-19456}; NVG=${XCC_NVG:-48}
USERS=${XCC_USERS:-$MS}; TURNS=${XCC_TURNS:-32}; WTURNS=${XCC_WTURNS:-3}; SEED=${XCC_SEED:-1}; CAP=${XCC_CAP:-1500}
EFFORT=${XCC_EFFORT:-xhigh,medium}; CCX=${XCC_CCX:-}; SAMP=${XCC_SAMP:-10}; MINCTX=${XCC_MINCTX:-50000}
DISKMIN=${XCC_DISKMIN_MB:-11000}; EST=${XCC_EST_MIN:-26}; BUDGET=${XCC_BUDGET_MIN:-$((36 * NRUN + 5))}
SIDE=${XCC_SIDECAR:-dspark-dlw-q8_0.rad}; SIDESHA=${XCC_SIDESHA:-f1ae116c8e07f69ab98520f6f19b73e7b7f576e72ac36484694e56fb6acdf681}
SEXTRA=$(echo ${XCC_SEXTRA:-})
KVR=${XCC_KVR:-auto}; [ "$KVR" = auto ] && { case $IMAGE in *ds41-int9*) KVR=1 ;; *) KVR=0 ;; esac; }   # coordinator 12.11.: KVR on in both starts (int9)
case $KVR in 0|1) ;; *) echo "XCC_KVR takes auto, 0 or 1 (got '$KVR')"; exit 1 ;; esac
case " $SEXTRA " in *RADIANCE_DS41_KV_REPLICATED*) echo "XCC_SEXTRA must not carry RADIANCE_DS41_KV_REPLICATED (XCC_KVR is its knob)"; exit 1 ;; esac
[ $KVR = 1 ] && SEXTRA=$(echo $SEXTRA RADIANCE_DS41_KV_REPLICATED=1)
S5T='{"ds41_bi": 2, "draft_sample": 85, "draft_occ": 6, "place_lf": 0, "ds41_psk": 4, "ds41_psk_k": 8, "ds41_psk_cr": 1, "ds41_psk_l20": 1, "dspark_pfctx": 1, "ds41_engdd": 1, "sched_noa": 2, "sched_bi_chunk": 3, "stage_short": 0, "stage_pick": 0, "place_hb": 0}'
SPIN=${XCC_SPIN:-$S5T}; PEXTRA=${XCC_S5ENV-RADIANCE_DS41_ENGDD=0}
python3 -c 'import json,sys; d=json.loads(sys.argv[1]); assert isinstance(d, dict) and d' "$SPIN" 2>/dev/null || { echo "XCC_SPIN is not a JSON object ('$SPIN')"; exit 1; }
for v in $MS $MMLV $HP $NVG $USERS $TURNS $WTURNS $SEED $CAP $SAMP $MINCTX $DISKMIN $EST $BUDGET; do case $v in ''|*[!0-9]*) echo "a numeric knob is not a number ('$v')"; exit 1 ;; esac; done
[ $NRUN -ge 1 ] || { echo "XCC_ORDER is empty"; exit 1; }
for s in $ORDER; do case $s in b8|n8) ;; *) echo "XCC_ORDER takes b8 / n8 (got '$s')"; exit 1 ;; esac; done
[ $WTURNS -lt $TURNS ] || { echo "XCC_WTURNS ($WTURNS) must be below XCC_TURNS ($TURNS)"; exit 1; }
[ $NVG -le 48 ] || { echo "XCC_NVG is 0 (off) .. 48"; exit 1; }
[ $SAMP -ge 2 ] || { echo "XCC_SAMP >= 2 s"; exit 1; }
case "$CCX" in *\'*|*\"*|*\;*|*\`*|*'$'*) echo "XCC_CCX: no quotes / ; / \` / \$"; exit 1 ;; esac
for w in --gen-mode --temperature --seed --turns --warmup-turns --users --trace-seeded --cpt-fixed --duration --effort --urls --replay --api-key; do
  case " $CCX " in *" $w "*|*" $w="*) echo "XCC_CCX must not carry $w (a fixed part of this item's load; its own knob where there is one)"; exit 1 ;; esac; done
case " $SEXTRA " in *RADIANCE_DS41_LATENT*) echo "XCC_SEXTRA must not carry RADIANCE_DS41_LATENT (the starts' own switch)"; exit 1 ;; esac
for kv in $SEXTRA; do case $kv in RADIANCE_*=*|R4D_*=*) ;; *) echo "XCC_SEXTRA takes NAME=VALUE start words (got '$kv')"; exit 1 ;; esac; done
CCARGS=$(echo "--duration $CAP --turns $TURNS --warmup-turns $WTURNS --effort $EFFORT --gen-mode natural --seed $SEED --cpt-fixed --temperature 0 --trace-seeded $CCX")
CCW=$HOME/gpu/ds41/exl3-test/ccsim-exl3-turns.sh   # the kit's cc_sim wrapper, read only
CCS=$QA/tools/ccsim/cc_sim3.py                     # the harness, read only
SUM=$HD/ccsum.py; NCS=$HD/ncsteps.py; NCONE=$HD/ncone.py; KSUM=$HD/../kv4/kv4sum.py   # ncone.py: ncsteps.py loads it (selftest)
CCREF=$R/ccsim-1106-0531-best-s8-u8.jsonl          # seed 1, u8 on this model: the server-counted first prompts (trace information)
MBP=$D/mb.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
T0=$(date +%s); NV=0; SPID=""
case "$ORDER" in "b8 n8") ONOTE="A / B once, b8 first: order and drift not cancelled" ;; "n8 b8 b8 n8") ONOTE="B A A B, fresh starts" ;; *) ONOTE="order $ORDER" ;; esac
xlog() { case "$*" in *VIRHE*) NV=$((NV + 1)) ;; esac; log "RESULT xcc $*"; }
# ---------------------------------------------------------------- copied from dsm-kv4.sh (f89a6cc2; tag xkv4 -> xcc; one pin, no arms)
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
LITS="$LITS|keys,mxfp4,(80,bytes,an,entry;,RADIANCE_DS41_IDXKEY)|kv:,the,pool,backs|kv:,the,cache,can,grow,to|VRAM,budget"
LITS="$LITS|lever,E1,declared,(RADIANCE_DS41_ENGDD"
# + stack5's declare lines of the S5T levers (PSK-L20, PSK-CR)
LITS="$LITS|now,0;,RADIANCE_DS41_PSK_L20):,layer,20,split,under,lever,PSK|now,0;,RADIANCE_DS41_PSK_CR):,a,planned,prefix,hit,with,no,snapshot,at,its,grid,point"
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": 8, "drafter_mx": 1, "gg_pf": 1, "f32_pf": 25, "dspark_walk": 1, "ds41_argather": 48, "idx_split": 1, "ds41_qnfold": 1, "ds41_swfold": 1, "draft_ctxskip": 1, "dspark_conf": 1, "dspark_conf_rows": 1, "draft_lend": 1, "draft_probe_backoff": 3, "draft_hybrid": 1}'
FIX=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update({"draft_lend_hyst": 0, "arena_d0": 0, "draft_lend_w": 0, "draft_occ_k": 32}); print(json.dumps(d))' "$FIX5") || { echo "pin JSON"; exit 1; }
pinj() {   # pinj [EXTRAJSON ...]: the fixed pin + S5T (XCC_SPIN) + the extras as one POST body (a later one wins)
  python3 -c 'import json,sys
d = {}
for a in sys.argv[1:]: d.update(json.loads(a or "{}"))
print(json.dumps(d))' "$FIX" "$SPIN" "$@"
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
  RADIANCE_DS41_DSPARK_PFCTX RADIANCE_PLACE_HB_RANKS RADIANCE_PLACE_HB_DECAY RADIANCE_DS41_PSK_CR RADIANCE_SCHED_BI_CHUNK
  RADIANCE_SCHED_BI_GRID RADIANCE_PROFILE_EVERY RADIANCE_DS41_KV_REPLICATED"
stackenv() {   # 105's t3 arm + gg_gemv 2 + rule-w + the K1 folds + turnover 0 + LEVENV + PEXTRA (+ SEXTRA), every other switch unset
  local kv
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  export RADIANCE_PLACE_TURNOVER=0
  for kv in $LEVENV $PEXTRA $SEXTRA; do export "$kv"; done
}
thermline() { type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"; }
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference (the build's sanity, not a natural-mode claim)
  [ "$DRY" = 1 ] && { echo "+ RESULT xcc $1 warm-up answers vs $2"; return 0; }
  [ -s $2 ] || { xlog "$1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  case $r in IDENTICAL*) xlog "$1 warm-up answers vs stock ($(basename $2)): $r" ;; *) xlog "$1 warm-up answers vs stock: VIRHE $r" ;; esac
}
modes() {   # modes TAG LAT: MODEWORDS in the container (RADIANCE_DS41_LATENT only under nvfp4, RADIANCE_DS41_KV_REPLICATED only when
  # XCC_SEXTRA asks), mover 0 kept free x4, rule-w, K1, ARGATHER 48, the latent literal READ BACK, every literal of LITS -> 0, else 1
  [ "$DRY" = 1 ] && { echo "+ modes $1: [$MODEWORDS] in the container's env$([ $2 = bf16 ] && echo ", RADIANCE_DS41_LATENT NOT in it"), RADIANCE_DS41_KV_REPLICATED $([ $KVR = 1 ] && echo "=1 in it (XCC_KVR 1) + 'lever KVR declared (RADIANCE_DS41_KV_REPLICATED=1)' and the tier line 'held once a slot: X a slot rather than Y' in the start log ($([ $2 = bf16 ] && echo "X = 5.39 KiB" || echo "X logged: a reading under nvfp4"))" || echo "NOT in it"); 'latent $2 ($([ $2 = nvfp4 ] && echo 288 || echo 1024) bytes an entry; RADIANCE_DS41_LATENT)' read back from the start log; mover '0 more kept free' on 4 ranks; rule-w in every recorded pass; K1 folds =1; ARGATHER 48-row note; literals [$(echo "$LITS" | tr ',' ' ')]"; return 0; }
  local CE kv bad="" ok=1 NW NR KL AL LL nz lit found="" miss="" want
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in RADIANCE_PLACE_TURNOVER=0 $MODEWORDS; do echo "$CE" | grep -qxF "$kv" || bad="$bad ${kv%%=*}"; done
  [ $2 = bf16 ] && echo "$CE" | grep -q '^RADIANCE_DS41_LATENT=' && bad="$bad RADIANCE_DS41_LATENT(set-under-bf16)"
  case " $SEXTRA " in *RADIANCE_DS41_KV_REPLICATED=*) ;; *) echo "$CE" | grep -q '^RADIANCE_DS41_KV_REPLICATED=' && bad="$bad RADIANCE_DS41_KV_REPLICATED(set)" ;; esac
  [ -z "$bad" ] || { xlog "$1: VIRHE start words not as asked:$bad"; ok=0; }
  if [ $KVR = 1 ]; then   # lever KVR read back: the declare line + the tier's "held once a slot" line (bf16: 5.39 KiB, kvr's B / V)
    KD=$(docker logs $CN 2>&1 | grep -c 'lever KVR declared (RADIANCE_DS41_KV_REPLICATED=1)')
    KS=$(docker logs $CN 2>&1 | grep -m1 -o 'held once a slot: [0-9.]* [KMG]*i*B a slot rather than [0-9.]* [KMG]*i*B, so --prefix-cache-mib [0-9]* holds [0-9]* tokens rather than [0-9]*')
    KX=$(echo "$KS" | sed -E 's/^held once a slot: ([0-9.]+ [KMG]*i*B) a slot.*/\1/')
    if [ "$KD" -gt 0 ] && [ -n "$KS" ] && { [ $2 = nvfp4 ] || [ "$KX" = "5.39 KiB" ]; }; then xlog "$1 KVR read back: env RADIANCE_DS41_KV_REPLICATED=1, declared on $KD line(s), tier '$KS'$([ $2 = nvfp4 ] && echo " (nvfp4: the slot size is a reading; bf16's 5.39 KiB)")"
    else xlog "$1: VIRHE lever KVR not read back (declare lines $KD, tier line '$KS'; want $([ $2 = bf16 ] && echo "5.39 KiB a slot" || echo "a 'held once a slot' line"))"; ok=0; fi
  fi
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
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN | gzip > $O/engine-$1.log.gz"; return 0; }; docker logs $CN 2>&1 | gzip -1 > $O/engine-$1.log.gz || true; }
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
fn4code() { [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
ARMOK=1
pincheck() {   # pincheck TAG WHEN: GET /admin/fn4, every switch of the pin applied, nothing pending (no POST) -> 0, else VIRHE
  local g r
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4: every switch of $(pinj) applied, nothing pending -> RESULT xcc $1 pin $2 (read back)"; return 0; }
  g=$(fn4get)
  r=$(python3 -c 'import json,sys
g, w = json.loads(sys.argv[1]), json.loads(sys.argv[2])
bad = {k: g["switches"].get(k) for k, v in w.items() if g["switches"].get(k) != v}
s = g["switches"]
print(("OK" if not bad and not g["pending"] else "FAIL %s pending %s" % (bad, g["pending"])) + " | bi=%s psk=%s/%s cr=%s l20=%s pfctx=%s engdd=%s noa=%s bic=%s occ=%s ds=%s adapt=%s d0=%s sp=%s/%s epoch=%s" % (
    s.get("ds41_bi"), s.get("ds41_psk"), s.get("ds41_psk_k"), s.get("ds41_psk_cr"), s.get("ds41_psk_l20"), s.get("dspark_pfctx"), s.get("ds41_engdd"),
    s.get("sched_noa"), s.get("sched_bi_chunk"), s.get("draft_occ"), s.get("draft_sample"), s.get("draft_adaptive"), s.get("draft_depth0"),
    s.get("stage_full_min", "-"), s.get("stage_full_pick", "-"), g.get("epoch")))' "$g" "$(pinj)" 2>&1 | cut -c1-500)
  case "$r" in OK*) xlog "$1 pin $2 (read back): $r"; return 0 ;; esac
  xlog "$1 pin $2: VIRHE the pin is not applied ($r) -- no reading under it"; echo "$1 pins VIRHE $2 $r" >> $O/guard.txt; ARMOK=0; return 1
}
armset() {   # armset TAG: the pin + epoch_bump POSTed, a nudge, every switch READ BACK (else VIRHE, ARMOK 0)
  fn4set "$(pinj '{"epoch_bump": 1}')"; nudge
  pincheck $1 "applied"
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
startenv() {   # startenv TAG LAT: kv4's b8 / n8 serve line at MAXSEQS MS / HOSTPOOL HP / MML MMLV + RADIANCE_LOG_STEPS=1 + NVG (+ nvfp4)
  local tag=$1 lat=$2 nv=""
  stackenv; export RADIANCE_LOG_STEPS=1
  [ $NVG -gt 0 ] && { export RADIANCE_DS41_NVGATHER=$NVG; nv="RADIANCE_DS41_NVGATHER=$NVG"; }
  [ $lat = nvfp4 ] && export RADIANCE_DS41_LATENT=nvfp4      # the yaml's list passes it from the shell; unset under bf16
  export MODEL=$M225; HP29=$HP; best_vars
  export MNBT=4160 NST=5 MAXSEQS=$MS MML=$MMLV XARGS="$XARGS --live --draft-window adaptive --draft-coupling on"
  export XS2_ENVFILE=$O/$tag.env STATSDIR=$O/stats-$tag PROFEVERY=
  MODEWORDS="$LEVENV $PEXTRA $SEXTRA RADIANCE_LOG_STEPS=1 $nv $([ $lat = nvfp4 ] && echo RADIANCE_DS41_LATENT=nvfp4)"
  MODEWORDS=$(echo $MODEWORDS)
  envfile $XS2_ENVFILE $LEVENV $PEXTRA $SEXTRA RADIANCE_LOG_STEPS=1 $nv
  mkd $STATSDIR
}
swstart() {   # swstart TAG: the switches at start (stack4's start values; every switch of the pin present; int9's SP 0); ds41_psk 99 refused
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 at start: gg_gemv 2, aql_acq 2, K1 three 1, epoch 0, draft_capable 1, ds41_bi 0, ds41_psk 0, ds41_psk_cr 0, sched_bi_chunk 0, stage_full_min / stage_full_pick 0 where the image has them (int9), every switch of $(pinj) present; POST ds41_psk 99 -> 400"; return 0; }
  local s0 c1 r
  s0=$(fn4get); c1=$(fn4code '{"ds41_psk": 99}')
  r=$(python3 -c 'import json,sys
s0 = json.loads(sys.argv[1]); w = s0["switches"]; want = json.loads(sys.argv[3])
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) and s0["epoch"] == 0
ok = ok and s0["draft_capable"] == 1 and w.get("ds41_bi") == 0 and w.get("ds41_psk") == 0 and w.get("sched_bi_chunk") == 0 and w.get("ds41_psk_cr") == 0
sp = [k for k in ("stage_full_min", "stage_full_pick") if k in w]
ok = ok and all(w[k] == 0 for k in sp)
miss = [k for k in want if k not in w]
ok = ok and not miss and sys.argv[2] == "400"
print(("OK" if ok else "FAIL") + " ds41_bi %s ds41_psk %s/%s ds41_psk_cr %s sched_bi_chunk %s draft_occ %s draft_sample %s | SP %s | ds41_psk 99 -> HTTP %s%s" % (
    w.get("ds41_bi"), w.get("ds41_psk"), w.get("ds41_psk_k"), w.get("ds41_psk_cr"), w.get("sched_bi_chunk"), w.get("draft_occ"), w.get("draft_sample"),
    (" ".join("%s %s" % (k, w[k]) for k in sp)) or "not in this image", sys.argv[2], (" | not in the registry: %s" % miss) if miss else ""))' "$s0" "$c1" "$(pinj)" 2>&1 | cut -c1-500)
  case "$r" in OK*) xlog "$1 switches at start: $r"; return 0 ;; esac
  xlog "$1 switches at start: VIRHE $r | $(echo "$s0" | cut -c1-900)"; return 1
}
# ---------------------------------------------------------------- the cc run
spstop() { [ -n "$SPID" ] && { kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; SPID=""; }; return 0; }
ccrun() {   # ccrun TAG: prefix-cache reset, the /stats sampler, cc_sim3 through ccsim-exl3-turns.sh, the sampler stopped, the run's files copied
  local t=$1 out t00=$(date +%s)
  pcreset $t-cc
  snap $t-cc0
  if [ "$DRY" = 1 ]; then
    echo "+ python3 -I -B $SUM sample --url $URL --every $SAMP --out $O/$t-samp.jsonl --max-s $((CAP + 900)) > $O/$t-samp.out 2>&1 &   (SIGTERM after the cc run)"
  else
    python3 -I -B $SUM sample --url $URL --every $SAMP --out $O/$t-samp.jsonl --max-s $((CAP + 900)) > $O/$t-samp.out 2>&1 &
    SPID=$!
  fi
  run "(CCSIM=$CCS CCARGS='$CCARGS' timeout $((CAP + 900)) bash $CCW $USERS xcc-$t > $O/$t-cc.out 2>&1)"
  spstop
  snap $t-cc1
  [ "$DRY" = 1 ] && { echo "+ the wrapper's 'CCSIM DONE <runs/ccsim-MMDD-HHMM-xcc-$t-u$USERS>': its .jsonl / .txt / .mdelta.txt copied to $O/$t-cc.*"; return 0; }
  out=$(grep -m1 '^CCSIM DONE ' $O/$t-cc.out | cut -c12-)
  if [ -z "$out" ] || [ ! -s "$out.jsonl" ]; then xlog "$t cc: VIRHE no cc_sim output ($(tail -3 $O/$t-cc.out | tr '\n' ' ' | cut -c1-400))"; return 1; fi
  for x in jsonl txt mdelta.txt; do [ -e $out.$x ] && cp -p $out.$x $O/$t-cc.$x; done
  xlog "$t cc_sim ($(grep -m1 '^cc_sim exit' $O/$t-cc.txt), $(( $(date +%s) - t00 )) s, files $out.*): $(grep -m1 '^RESULT cc_sim' $O/$t-cc.out | cut -c1-1500) | sampler: $(tail -1 $O/$t-samp.out | cut -c1-200) | $(thermline $t00 $(date +%s))"
}
ccparse() {   # ccparse TAG LAT: ccsum.py run -> its RESULT lines into the log
  local t=$1 lat=$2 rc l
  [ "$DRY" = 1 ] && { echo "+ python3 -I -B $SUM run --cc $O/$t-cc.jsonl --samples $O/$t-samp.jsonl --log $O/$t-engine.log.gz --tag $t --lat $lat --ccsim $CCS --minctx $MINCTX --nst 5 --every $SAMP --json $O/$t-cc.json > $O/$t-ccsum.out -> its RESULT xcc lines (run, contexts, pool / experts, timeline, streamed, levers, guard)"; return 0; }
  timeout 900 python3 -I -B $SUM run --cc $O/$t-cc.jsonl --samples $O/$t-samp.jsonl --log $O/$t-engine.log.gz --tag $t --lat $lat --ccsim $CCS \
    --minctx $MINCTX --nst 5 --every $SAMP --json $O/$t-cc.json > $O/$t-ccsum.out 2>&1; rc=$?
  # 12.11. fix (ds-main-port): a reading line that says "0 with a guard VIRHE" is NOT a VIRHE (the 10:04 cc run counted it as one)
  while IFS= read -r l; do case "$l" in "RESULT xcc "*) lv=$(printf "%s" "$l" | sed -E "s/(^|[^0-9])0 with a guard VIRHE/\\1/g"); case "$lv" in *VIRHE*) NV=$((NV + 1)) ;; esac; log "${l:0:3800}" ;; esac; done < $O/$t-ccsum.out
  grep -q '^RESULT xcc ' $O/$t-ccsum.out || xlog "$t parse: VIRHE ccsum.py run printed no reading (rc $rc: $(tail -2 $O/$t-ccsum.out | tr '\n' ' ' | cut -c1-400))"
}
DONE=""
kstart() {   # kstart TAG (b8-K / n8-K): one start, its cc run, its parse
  local tag=$1 arm=${1%-*} lat el
  case $arm in b8) lat=bf16 ;; n8) lat=nvfp4 ;; esac
  el=$(( ($(date +%s) - T0) / 60 ))
  if [ "$DRY" != 1 ] && [ $((el + EST)) -gt $BUDGET ]; then xlog "$tag SKIPPED: $el min elapsed + ~$EST min > the budget $BUDGET min"; return 0; fi
  diskok $tag || return 1
  startenv $tag $lat
  xlog "$tag: start (~$EST min; latent $lat, MAXSEQS $MS, HOSTPOOL $HP MiB, MML $MMLV; kv4's NST 5 start + RADIANCE_LOG_STEPS=1 + NVG $NVG$([ -n "$PEXTRA" ] && echo " + $PEXTRA")$([ -n "$SEXTRA" ] && echo " + $SEXTRA")) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $(pinj) | cc_sim3 u$USERS [$CCARGS] | env: $(envline)"
  gate1 $tag 70 || { xlog "$tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  modes $tag $lat || { xlog "$tag: VIRHE the modes are not the start's -> stopped"; down; return 1; }
  drafted $tag || { keeplog $tag-nodraft; down; return 1; }
  swstart $tag || { keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  armset $tag || { keeplog $tag-pin; down; return 1; }
  snap $tag-st0
  pyx $tag-cap python3 -I -B $KSUM cap --log $O/start-$tag.log --stats $O/$tag-st0-st.json --tag $tag --lat $lat --ms $MS --mml $MMLV --hp $HP --json $O/$tag-cap.json
  if stopfile; then xlog "$tag: sweep.stop -- no cc run"; down; return 0; fi
  ccrun $tag
  pincheck $tag "after the cc run"
  [ "$DRY" = 1 ] || xlog "$tag: $(docker logs $CN 2>&1 | grep -cE '^E ') E lines$([ "$(docker logs $CN 2>&1 | grep -cE '^E ')" = 0 ] || echo ": $(docker logs $CN 2>&1 | grep -E '^E ' | head -3 | tr '\n' ' ' | cut -c1-500)") | BI $(docker logs $CN 2>&1 | grep 'lever BI stats' | tail -1 | sed 's/^radiance: deepseek41: //' | cut -c1-200)"
  capture $O/$tag-engine.log.gz
  down
  ccparse $tag $lat
  DONE="$DONE $tag"
  xlog "$tag: done ($(( ($(date +%s) - T0) / 60 )) min into the item)"
}
report() {
  local l x y b n
  run "python3 -I -B $SUM report --dir $O --runs '$RUNS' --order-note '$ONOTE' > $O/report.txt 2>&1"
  if [ "$DRY" = 1 ]; then echo "+ every RESULT xcc line of $O/report.txt -> the log (one table line a run, the n8 vs b8 pair(s) in %, B A A B mode means); cc_sim3 --pair over each adjacent b8 / n8 pair (information)"
  else
    while IFS= read -r l; do case "$l" in "RESULT xcc "*) log "${l:0:3800}" ;; esac; done < $O/report.txt
  fi
  set -- $DONE
  while [ $# -ge 2 ]; do
    x=$1; y=$2; shift
    [ "${x%-*}" = "${y%-*}" ] && continue
    b=$x; n=$y; [ "${x%-*}" = n8 ] && { b=$y; n=$x; }
    if [ "$DRY" = 1 ]; then echo "+ python3 -I -B $CCS --pair $O/$b-cc.jsonl $O/$n-cc.jsonl"; continue; fi
    [ -s $O/$b-cc.jsonl ] && [ -s $O/$n-cc.jsonl ] || continue
    xlog "ccpair $n vs $b (cc_sim3 --pair, B = $n over A = $b; information -- natural mode pairs only the requests whose prompt / gen lengths match, no identity claim): $(timeout 300 python3 -I -B $CCS --pair $O/$b-cc.jsonl $O/$n-cc.jsonl 2>&1 | tr '\n' ';' | cut -c1-1500)"
  done
  [ "$DRY" = 1 ] && set -- && return 0
  local nv; nv=$(( $(grep -c VIRHE $O/report.txt 2>/dev/null) + NV ))
  [ $ARMOK = 1 ] || nv=$((nv + 1))
  log "RESULT xcc guards (item): $([ "$nv" = 0 ] && echo "OK (latent read back in every start$([ $KVR = 1 ] && echo ", KVR read back (env + declare + tier slot)"), pins read back before and after every cc run, the trace reached >= $MINCTX tokens, samples and step-log lines)" || echo "VIRHE in $nv place(s) -- see the lines above; a reading with VIRHE is not a reading") | runs done:$DONE of [$RUNS] ($ONOTE)"
}
cleanup() { spstop; down; }
trap 'cleanup' EXIT
trap 'exit 143' TERM INT
# ---------------------------------------------------------------- main
RUNS=$(seen=""; for s in $ORDER; do k=$(( $(echo $seen | tr ' ' '\n' | grep -cx $s) + 1 )); seen="$seen $s"; echo -n "$s-$k "; done)
RUNS=$(echo $RUNS)
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) | USER-ASKED cc_sim bf16 vs nvfp4 latent | runs [$RUNS] ($ONOTE) | MAXSEQS $MS MML $MMLV HOSTPOOL $HP NVG $NVG | KVR $([ $KVR = 1 ] && echo "ON (RADIANCE_DS41_KV_REPLICATED=1, both starts)" || echo off) | pin S5T $(echo "$SPIN" | tr -d '{}"') | cc_sim3 u$USERS [$CCARGS] via $(basename $CCW) | /stats every $SAMP s | ~$((25 * NRUN + 3)) min (budget $BUDGET)"
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
B|RADIANCE_LOG_STEPS
B|n_seq=%lld n_tok=%lld
B|  [%d] slot=%d q=%d ctx=%d
B|kv: the pool backs %.0f%% of the worst case
B|kv: the cache can grow to %s a card
B|VRAM budget, per card
X|sched_noa
X|ds41_psk_l20
X|dspark_pfctx
X|ds41_engdd
A|RADIANCE_DS41_ENGDD
B|ds41_psk_cr
B|stage_short
B|stage_pick
B|place_hb
A|RADIANCE_DS41_PSK_L20
A|RADIANCE_DS41_PSK_CR
A|lever PSK-L20 (fn4 ds41_psk_l20, now %d; RADIANCE_DS41_PSK_L20): %s
A|lever PSK-CR (fn4 ds41_psk_cr, now %d; RADIANCE_DS41_PSK_CR): %s
A|tail-only
B|tail_only_latched
B|cr_hits
B|kv_tokens_used
B|kv_tokens_cached
B|kv_tokens_total
B|stream_bytes
B|trunk_vram
B|drop_recall
B|steps_mixed"
[ $KVR = 1 ] && IMGLITS="$IMGLITS
A|RADIANCE_DS41_KV_REPLICATED
A|lever KVR declared (RADIANCE_DS41_KV_REPLICATED=1): %s hold the same bytes on"
docker image inspect $IMAGE > /dev/null 2>&1 || { xlog "VIRHE image $IMAGE not on this host"; exit 1; }
IMGMISS=$(echo "$IMGLITS" | timeout 120 docker run --rm -i --network none --entrypoint sh $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels/libr4d.so; S=/opt/radiance/share/radiance/kernels/spechub.so; n=0
while IFS= read -r l; do [ -n "$l" ] || continue; n=$((n + 1)); w=${l%%|*}; s=${l#*|}
  case $w in B) f=$B ;; A) f=$A ;; K) f=$K ;; S) f=$S ;; *) f="$B $A $K" ;; esac
  o=0; for x in $f; do grep -qzF -- "$s" $x && { o=1; break; }; done
  [ $o = 1 ] || echo "MISS $w|$s"; done
test -x /opt/radiance/bin/rad-sidecar || echo "MISS bin/rad-sidecar"
test -s /opt/radiance/share/radiance/int8-PROVENANCE.txt || echo "MISS int8-PROVENANCE.txt (an int8-or-later image)"
for p in /opt/radiance/share/radiance/int9-PROVENANCE.txt; do [ -s $p ] && echo "INT9 $(grep -c "RADIANCE_DS41_KV_REPLICATED unset" $p)"; done
echo "CHECKED $n"' 2>&1)
case "$IMGMISS" in *MISS*|"") xlog "VIRHE $IMAGE does not carry what this item reads -- missing: $(echo "$IMGMISS" | grep -E '^MISS' | tr '\n' ';' | cut -c1-900)"; exit 1 ;; esac
echo "$IMGMISS" | grep -q '^CHECKED ' || { xlog "VIRHE the image check did not run ($(echo "$IMGMISS" | tail -2 | tr '\n' ' ' | cut -c1-300))"; exit 1; }
IMGKIND=$(echo "$IMGMISS" | grep -q '^INT9 ' && echo "int9 (int8 + KVR + SP; KVR $([ $KVR = 1 ] && echo ON || echo "unset"), SP 0)" || echo "int8")
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)) = $IMGKIND: all $(echo "$IMGMISS" | grep '^CHECKED' | sed 's/CHECKED //') literals present (stack4's registry / BI / SD / DLW / Z0 / PSK, KV4's latent note + latent_format + NVG, the S5 / S5T switches and declare lines (PSK-L20 / PSK-CR / tail-only / E1), the step log, the pool notes, the /stats fields ccsum.py reads$([ $KVR = 1 ] && echo ", lever KVR's env + declare line")) + bin/rad-sidecar + int8-PROVENANCE -- OK"
HMISS=""
for f in "$LIB29" "$LIBPR" $Y $CCW $CCS $SUM $NCS $NCONE $KSUM $MBP $D/ds41_tests.py $D/fastcheck.py $D/mdelta.py "$HD/../nc/ncsteps.py"; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
PYBAD=$(python3 -I -c 'import ast, sys
for f in sys.argv[1:]:
    try: ast.parse(open(f).read(), f)
    except Exception as e: print("parse:%s(%s)" % (f, type(e).__name__))' $CCS $SUM $NCS $KSUM $D/ds41_tests.py $D/fastcheck.py $D/mdelta.py 2>&1 | tr '\n' ' ')
HMISS="$HMISS $PYBAD"
bash -n $CCW 2>/dev/null || HMISS="$HMISS bash-n:$CCW"
grep -q 'CCSIM=${CCSIM:-' $CCW && grep -q 'summarize_turns' $CCW && grep -q '^echo "CCSIM DONE \$OUT"' $CCW || HMISS="$HMISS wrapper:$CCW(no CCSIM knob / turns summary / DONE line)"
for x in ncsteps ncone; do [ "$(md5sum < $HD/$x.py | cut -c1-32)" = "$(md5sum < $HD/../nc/$x.py | cut -c1-32)" ] || HMISS="$HMISS copy:$HD/$x.py(differs from nc/$x.py)"; done
ST1=$(timeout 60 python3 -I -B $SUM selftest 2>&1 | tail -1); case "$ST1" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$SUM(${ST1:0:300})" ;; esac
ST2=$(timeout 60 python3 -I -B $KSUM selftest 2>&1 | tail -1); case "$ST2" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$KSUM(${ST2:0:300})" ;; esac
ST3=$(timeout 60 python3 -I -B $NCS selftest 2>&1 | tail -1); case "$ST3" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$NCS(${ST3:0:300})" ;; esac
CCH=$(timeout 60 python3 -I -B $CCS --help 2>&1); for w in --trace-seeded --cpt-fixed --turns --warmup-turns --gen-mode --pair; do case "$CCH" in *"$w"*) ;; *) HMISS="$HMISS ccsim:$CCS(no $w)" ;; esac; done
TRC=$(timeout 120 python3 -I -B $SUM trace --ccsim $CCS --seed $SEED --users $USERS --minctx $MINCTX $([ $SEED = 1 ] && [ $USERS = 8 ] && [ -s $CCREF ] && echo "--ref $CCREF") 2>&1 | tail -1)
case "$TRC" in "TRACE OK"*) ;; *) HMISS="$HMISS trace:(${TRC:0:300})" ;; esac
FNS="log run mkd restart check down stopfile best_vars desc serve_args xlog pinj stackenv thermline identw modes keeplog fn4get fn4set fn4code nudge pincheck armset gate1 dfree diskok envfile envline drafted pyx pcreset snap capture startenv swstart spstop ccrun ccparse kstart report cleanup"
for fn in $FNS; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for b in docker curl python3 awk sed grep timeout tee sha256sum gzip zcat df du md5sum; do command -v $b > /dev/null || HMISS="$HMISS tool:$b"; done
python3 -c 'import json,sys
d = json.loads(sys.argv[1]); s = json.loads(sys.argv[2])
ok = d["gg_gemv"] == 2 and d["epoch_bump"] == 1 and d["draft_adaptive"] == 1 and d["draft_depth0"] == 1 and all(d[k] == v for k, v in s.items())
sys.exit(0 if ok else 1)' "$(pinj '{"epoch_bump": 1}')" "$SPIN" 2>/dev/null || HMISS="$HMISS json:pin"
[ "$(sha256sum $MODELDIR/$SIDECAR 2>/dev/null | cut -c1-64)" = "$(cut -c1-64 $MODELDIR/$SIDECAR.sha256 2>/dev/null)" ] || HMISS="$HMISS sidecar:$MODELDIR/$SIDECAR"
[ "$(sha256sum $MODELDIR/$SIDE 2>/dev/null | cut -c1-64)" = "$SIDESHA" ] || HMISS="$HMISS sha256:$MODELDIR/$SIDE(want ${SIDESHA:0:16})"
for w in XS2_ENVFILE 'RADIANCE_DS41_LATENT:' '${STATSDIR:?set STATSDIR}:/stats'; do grep -qF "$w" $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no '$w')"; done
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$O/stats ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config refuses it)"
HMISS=$(echo $HMISS)
[ -z "$HMISS" ] || { xlog "VIRHE helpers / inputs do not resolve: $HMISS"; exit 1; }
HMD5="ccsum.py $(md5sum < $SUM | cut -c1-8), ncsteps.py $(md5sum < $NCS | cut -c1-8) + ncone.py $(md5sum < $NCONE | cut -c1-8) (= nc/'s $(md5sum < $HD/../nc/ncsteps.py | cut -c1-8) / $(md5sum < $HD/../nc/ncone.py | cut -c1-8)), kv4/kv4sum.py $(md5sum < $KSUM | cut -c1-8), cc_sim3.py $(md5sum < $CCS | cut -c1-8), ccsim-exl3-turns.sh $(md5sum < $CCW | cut -c1-8), mdelta.py $(md5sum < $D/mdelta.py | cut -c1-8), lib-29.sh $(md5sum < $LIB29 | cut -c1-8), lib-pr266.sh $(md5sum < $LIBPR | cut -c1-8), yaml $(md5sum < $Y | cut -c1-8)"
[ "$DRY" = 1 ] && echo "+ helpers: every file present and parsing; ccsum.py $ST1 | kv4sum.py $ST2 | ncsteps.py $ST3 | cc_sim3.py --help offline (trace-seeded / cpt-fixed / turns / warmup-turns / gen-mode / pair) | wrapper $(basename $CCW): bash -n, CCSIM knob, turns summary, DONE line | all $(echo $FNS | wc -w) functions resolve, tools on PATH, pin JSON, sidecars $SIDECAR + $SIDE sha256 OK, yaml accepted (env_file, RADIANCE_DS41_LATENT, /stats mount) -- OK | md5: $HMD5"
[ "$DRY" = 1 ] && echo "+ $TRC"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { xlog "VIRHE $M225 missing or not verified"; exit 1; }
  [ -s $WARM_REF_225 ] || xlog "note -- reference $WARM_REF_225 missing (warm-up answers not compared)"
  xlog "$ITEM: image $IMGKIND | trace: $TRC | helpers md5: $HMD5"
else
  for f in $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDE $WARM_REF_225 $RS4/guard-s4-n5.txt $CCREF; do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  echo "+ pin POST body: $(pinj '{"epoch_bump": 1}')"
  echo "+ cc_sim3 load (every run): CCSIM=$CCS CCARGS='$CCARGS' bash $CCW $USERS xcc-<run> (= cc_sim3.py --urls $URL --users $USERS $CCARGS --tag ds41exl3-xcc-<run>-u$USERS --out runs/ccsim-MMDD-HHMM-xcc-<run>-u$USERS.jsonl)"
  for s in $RUNS; do echo "+ run $s: latent $([ ${s%-*} = n8 ] && echo nvfp4 || echo bf16), MAXSEQS $MS, HOSTPOOL $HP MiB, MML $MMLV (~$EST min)"; done
  echo "+ order: $ONOTE | budget $BUDGET min | docker ps: $(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')(information here; the real run refuses to start beside one)"
fi
mkd $O $O/guard
[ "$DRY" = 1 ] || { : > $O/guard.txt; }
for s in $RUNS; do stopfile && break; kstart $s; done
report
unset $SWITCHES
down
xlog "$ITEM: done in $(( ($(date +%s) - T0) / 60 )) min -- $O ($(dfree) MB free)"
