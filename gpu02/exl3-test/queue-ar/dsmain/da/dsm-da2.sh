#!/bin/bash
# dsm-da2.sh PHASE (11.11.2026, queue-ar/dsmain/da, ds-main-port DA agent) -- = dsm-da.sh (md5 ec2f62f4, which items ieg / ieh run;
# left as it is) with the measure phase extended for item 57-p4zyzzzzz-ds-iei-da-c-measure (coordinator, 11.11. 04:20 UTC):
#  - the NST 5 start env is now stack2's AS IT RAN (hxzzy-stack2: HYB=1, XS2_LEND=1, XS2_VRAM=all, drafter_mx 1): + the VRAM
#    levers' n5 words (arena 48 rows, router bf16, embedding in RAM, shadow 16, expert prefetch off) and lever DL declared
#    (RADIANCE_DRAFT_LEND=0 ...); the stack pin + draft_hybrid 1, draft_lend 1, draft_probe_backoff 3 (dsm-da.sh had hybrid 0,
#    no lend, no VRAM levers); the fixed-window anatomy pin unchanged (hybrid 0, depth 0 off, conf off) + draft_lend 0;
#  - per arm, after the temperature-0 anatomy and stepprobe (kept as they are: the comparison rule): the SERVER-DEFAULT
#    SAMPLING 1-user probe (daprobe.py sampled: no temperature in the request = the engine's temperature 1, a fixed per-request
#    seed 1234, essay / code / edit, 2 runs) and mb c4 / c8 (mb.py: coding + reasoning tasks at the server-default sampling,
#    --dur 75 --warm 10, as the stack items) -- cross-start, since the overlay is start-time;
#  - a THIRD start B2 (the overlay again, after A): temperature-0 stepprobe + the sampled probe + mb c4 / c8 (no anatomy, no
#    identity), so the cross-start speed reads B A B (+ ~13 min); identity (temperature 0) on B as before.
#  - (11.11. 05:30 UTC, coordinator) ARM X between A and B2: an EXTERNAL fine-tune of the same drafter (HF
#    Avifenesh/DeepSeek-V4.1-Flash-DSpark-drafter-ft, MIT, mtp-trained.safetensors sha256 9e10e984 = the HF LFS oid; 72 tensors,
#    68 mapped by trainer/ext2ovl.py onto 86 of our entries -- wo_a split into g0..g7, w1|w3 fused into gate_up, bf16 widened
#    exactly to our f32 norms / router; the 3 gate.bias_vl (vision routing) have no counterpart -- every name / shape / dtype
#    checked, relative distance to ours 0.0-2.6 %) as RADIANCE_WEIGHT_OVERLAY=/stats/ovl/ext.ovl (sha256 ca8700e2) + its own
#    confidence head as the sidecar /stats/ovl/ext-conf.f32 (c0aafc62; 4.9 % from the release's): the same anatomy, stepprobe,
#    sampled probe, mb c4 / c8 and identity as B. Starts B A X B2 (~75 min). No ext.ovl -> X skipped with a line.
# ---- dsm-da.sh's header:
# dsm-da.sh PHASE (11.11.2026, queue-ar/dsmain/da, ds-main-port DA agent) -- NOT a queue item; the body of the three LEVER DA PILOT
# items (57-p4zyzzzzzzzz-da-{a-dump,b-train,c-measure}.sh). Tag "RESULT xdap ", out runs/xdsm/da (DAP_OUT), /stats = its stats/.
# Lever DA = self-distil the DSpark drafter to the EXL3 2.25 target (ds41-da image: core RADIANCE_KLD_DUMP + RADIANCE_WEIGHT_OVERLAY,
# deepseek41 RADIANCE_DS41_DA_TAPS; every one unset = the int3 image's behaviour). The drafter only proposes; verify keeps the
# output exact -- proven by identity on the adapted arm (28k copy + 20-prompt greedy set c1 vs bz d0).
#
#  dump     (~26 min) ONE NST 5 serve start (the stack2 n5 env, -v): the VALIDATION set at c1 under the fixed window 5 + walk 1
#           (dagen.py val: 6 prompts fi / en / code x 256 tokens, the engine's own proposals from -v) -> the SELF-GENERATED set
#           under the stack2 pin (dagen.py gen: Finnish / English prose + code prompts, greedy, 448 tokens, c8, DAP_GENMIN min) ->
#           down; the PUBLIC set (dagen.py public, CPU: Wikipedia fi / en extracts + Python stdlib code, ~1600-token slices) ->
#           ONE KL-RECORD start (--kld-record /stats/kl --kld-top 32, NST 0, RADIANCE_DS41_DA_TAPS=1 + RADIANCE_KLD_DUMP=
#           dspark_main_normed): per position the target's top 32 log p + the rest's mass and main_x (bf16, 10 KiB a token).
#  train    (~32 min) ROCm PyTorch (rocm/dev-ubuntu-24.04:latest + ~/gpu/ds41/da/torchenv, torch 2.10 TheRock gfx120X) on the 4
#           cards, one process a card: selftest (CPU) -> VALIDATION of the bf16 port against the engine's proposals, 2 numerics
#           variants (GPU 0/1) -> TRAIN 4 variants (GPU 0..3; lr 2e-5 / 5e-5 / 1e-4 / 5e-5 uniform rows), dense-only (routed
#           MXFP4 experts frozen; attention, shared expert, router, hc, norms), the KL to the target's top 32 + rest, DAP_TRAINMIN
#           min each, eval before / after on held-out documents (simulated fixed-5 acceptance) -> the best overlay /stats/ovl/best.ovl.
#  measure  (~30 min) TWO NST 5 starts, stack2 n5 env: B (RADIANCE_WEIGHT_OVERLAY=/stats/ovl/best.ovl) first, then A (the
#           container's drafter): fixed window 5 (+ walk 1, conf off) anatomy (daeval.py: 20 prose + 6 code daanat prompts + 10
#           Finnish, 384 tokens, logprobs 20, the -v proposals) -> the stack2 pin (adaptive) stepprobe essay,code 512 x2 + a
#           Finnish essay x2 -> B only: identity (28k copy + greedy set c1 vs bz d0) -> compare B vs A; the KL dump deleted.
# DRYRUN=1 bash dsm-da.sh PHASE : prints every command; checks the images (the da image's literals, the ROCm image), every helper
#   (files present and parsing, functions defined, python selftests on CPU), the inputs (present or produced by an earlier phase),
#   the yaml (docker compose config); no server, no GPU, nothing written. A missing helper fails the dry run (exit 1).
# DAP_IMAGE, DAP_OUT, DAP_GENMIN (8), DAP_TRAINMIN (20), DAP_ITEM. NO cc_sim, no 2.9, no BetterBench; no power / host / fan
# settings; production compose files read only (common.yaml via extends); no cloud.
set -u
PHASE=${1:?PHASE: dump | train | measure}
case $PHASE in dump|train|measure) ;; *) echo "PHASE must be dump, train or measure"; exit 1 ;; esac
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
Y=$HD/ds41-da-tp4.yaml
O=${DAP_OUT:-$R/xdsm/da}; export STATSDIR=$O/stats
RD=$R/xdsm/draft                                    # bz's d0 files (identity reference: copy + greedy set)
export IMAGE=${DAP_IMAGE:-radiance-local:ds41-da-bd2caa0c-5c13e555-l}
ITEM=${DAP_ITEM:-57-p4zyzzzzzzzz-da-$PHASE}
GENMIN=${DAP_GENMIN:-8}; TRAINMIN=${DAP_TRAINMIN:-20}
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
DAA=$HD/../daanat.py; GEN=$HD/dagen.py; EV=$HD/daeval.py; TR=$HD/trainer; PRB=$HD/daprobe.py; SEED=1234
PUB=$HD/public
DAH=$HOME/gpu/ds41/da; TENV=$DAH/torchenv
TIMG=rocm/dev-ubuntu-24.04:latest
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
SIDECAR=dspark-conf.f32
GPTAB=q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422
FINESSAY="Kirjoita pitkä ja yksityiskohtainen essee (noin 900 sanaa) suomalaisen kansanopetuksen historiasta ja sen vaikutuksesta yhteiskuntaan. Käytä useita kappaleita."
# the stack2 n5 pin (dsm-stack2.sh FIX: hx-stack + int3 levers + drafter_mx) and the anatomy pin (fixed window 5, conf off)
FIX='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": 8, "drafter_mx": 1, "gg_pf": 1, "f32_pf": 25, "dspark_walk": 1, "ds41_argather": 48, "idx_split": 1, "ds41_qnfold": 1, "ds41_swfold": 1, "draft_ctxskip": 1, "dspark_conf": 1, "dspark_conf_rows": 1, "draft_hybrid": 1, "draft_lend": 1, "draft_probe_backoff": 3}'
ANAT='{"draft_adaptive": 0, "draft_depth0": 0, "draft_hybrid": 0, "dspark_conf": 0, "dspark_walk": 1, "draft_lend": 0}'
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
  RADIANCE_WEIGHT_OVERLAY RADIANCE_DS41_DA_TAPS RADIANCE_KLD_DUMP"
# ---------------------------------------------------------------- helpers
xlog() { log "RESULT xdap $*"; }
# stackenv n5|kl [OVERLAY] [CONF]: the stack2 n5 start env (CONF: the confidence-head sidecar's container path, default the model dir's) (dsm-stack2.sh stackenv n5 + levenv n5 + turnover 0) exported, the start words
# also written to $ENVF (the yaml's env_file). kl: the numerics-relevant part only (no arena levels -- placement, exact -- no
# drafter words) + the DA tap dump. OVERLAY (container path) adds RADIANCE_WEIGHT_OVERLAY.
stackenv() {
  local kv w
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  export RADIANCE_PLACE_TURNOVER=0
  w="RADIANCE_DS41_GG_PF=1 RADIANCE_DS41_GG_PF_TABLE=$GPTAB RADIANCE_DS41_F32_PF=25"
  if [ $1 = n5 ]; then
    export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0
    w="$w RADIANCE_DS41_DSPARK_WALK=1 RADIANCE_DS41_ARGATHER_MAX=48 RADIANCE_DS41_QNFOLD=1 RADIANCE_DS41_SWFOLD=1 RADIANCE_DRAFT_CTXSKIP=0 RADIANCE_DS41_DSPARK_CONF=${3:-/models/$SIDECAR}"
    # stack2 as it ran: the VRAM levers' n5 words (XS2_VRAM=all) and lever DL declared off at start (XS2_LEND=1; on in the pin)
    w="$w RADIANCE_DS41_ARENA_DECODE_ROWS=48 RADIANCE_DS41_ROUTER_BF16=1 RADIANCE_DS41_EMB_RAM=1 RADIANCE_PLACE_SHADOW_SLOTS=16 RADIANCE_XPREFETCH=0"
    w="$w RADIANCE_DRAFT_LEND=0 RADIANCE_DRAFT_PROBE_BACKOFF=0 RADIANCE_DRAFT_LEND_BORROW=0"
  else
    w="$w RADIANCE_DS41_DA_TAPS=1 RADIANCE_KLD_DUMP=dspark_main_normed"
  fi
  [ -n "${2:-}" ] && w="$w RADIANCE_WEIGHT_OVERLAY=$2"
  for kv in $w; do export "$kv"; done
  LEVENV=$w
}
writeenv() { [ "$DRY" = 1 ] && { echo "+ printf '%s\\n' $LEVENV > $1"; return 0; }; printf '%s\n' $LEVENV > $1; }
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
FOLLOW_PID=""
follow() { [ "$DRY" = 1 ] && { echo "+ docker logs -f $CN > $1 &"; return; }
  docker logs -f $CN > $1 2>&1 & FOLLOW_PID=$!; sleep 2; }
unfollow() { [ -n "$FOLLOW_PID" ] && { kill $FOLLOW_PID 2>/dev/null; wait $FOLLOW_PID 2>/dev/null; FOLLOW_PID=""; }; return 0; }
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
pinset() {   # pinset TAG JSON...: the merged switches + epoch_bump in process, applied at a step (epoch moved, nothing pending)
  local tag=$1 body s0 s1; shift
  body=$(python3 -c 'import json,sys
d={}
for a in sys.argv[1:]: d.update(json.loads(a))
d["epoch_bump"]=1; print(json.dumps(d))' "$@")
  [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $body; a 4-token request; GET -> applied, nothing pending"; return 0; }
  s0=$(fn4get)
  curl -s -m 10 -X POST -H 'Content-Type: application/json' -d "$body" $URL/admin/fn4 > /dev/null
  nudge; s1=$(fn4get)
  python3 -c 'import json,sys
s0, s1, w = (json.loads(x) for x in sys.argv[1:4])
w.pop("epoch_bump", None)
ok = all(s1["switches"].get(k) == v for k, v in w.items()) and not s1["pending"] and s1["epoch"] > s0["epoch"]
bad = [k for k, v in w.items() if s1["switches"].get(k) != v]
print("applied" if ok else "NOT applied: %s" % bad); sys.exit(0 if ok else 1)' "$s0" "$s1" "$body" > $O/pin-$tag.txt 2>&1
  local rc=$?
  xlog "$tag pin: $(cat $O/pin-$tag.txt) -- $(echo "$s1" | cut -c1-600)"
  return $rc
}
modes() {   # modes TAG [overlay]: env in the container, W1 declared over 4 ranks, -v on, the overlay applied (when asked)
  [ "$DRY" = 1 ] && { echo "+ modes $1: env words in the container, the W1 declared line, -v lines${2:+, 'weight overlay: all N entries applied'}"; return 0; }
  local CE ok=1 kv WL OV
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in RADIANCE_PLACE_TURNOVER=0 RADIANCE_DS41_DSPARK_WALK=1 RADIANCE_DS41_GG_GEMV=2 ${2:+RADIANCE_WEIGHT_OVERLAY=$2}; do
    echo "$CE" | grep -qxF "$kv" || { xlog "$1: VIRHE $kv not in the container"; ok=0; }; done
  WL=$(docker logs $CN 2>&1 | grep -m1 'DSpark lever W1.*fn4 dspark_walk=' | sed 's/^radiance: //' | cut -c1-300)
  OV=$(docker logs $CN 2>&1 | grep -m2 'weight overlay' | tr '\n' '|' | cut -c1-500)
  xlog "$1 modes: W1: ${WL:-none} | debug lines $(docker logs $CN 2>&1 | grep -cE '^D ') | overlay: ${OV:-none}"
  case "$WL" in *"over 4 rank(s)"*"dspark_walk=1 "*) ;; *) xlog "$1: VIRHE no W1 declared line over 4 ranks"; ok=0 ;; esac
  if [ -n "${2:-}" ]; then case "$OV" in *"all "*" entries applied"*) ;; *) xlog "$1: VIRHE the overlay was not applied"; ok=0 ;; esac
  else case "$OV" in "") ;; *) xlog "$1: VIRHE an overlay line without RADIANCE_WEIGHT_OVERLAY"; ok=0 ;; esac; fi
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
cmpfile() {
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
gcmp() { [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }; [ -s "$1" ] && [ -s "$2" ] || { echo "not compared (missing $1 or $2)"; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
pcreset() { [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache"; return 0; }
  xlog "$1 prefix cache reset: HTTP $(curl -s -m 60 -o /dev/null -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)"; }
ident() {   # ident TAG: the 28k copy and the c1 greedy set vs bz d0
  local tag=$1 r
  pcreset $tag
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset 0 --tag=-$tag --outdir $O > $O/$tag-copy.txt 2>&1)"
  run "(cd $D && timeout 900 python3 $GS $URL $O/$tag-gs-c1.jsonl --conc 1 --max-tokens 192 > $O/$tag-gs-c1.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  r=$(cmpfile $O/copy-$tag-run1.txt $RD/copy-dsd-d0-i1-run1.txt); echo "$tag copy $r" >> $O/identity.txt
  xlog "$tag identity: 28k copy vs bz d0 (NST 0, stock): $r$(case $r in IDENTICAL*) ;; *) echo " -- reported, not hidden" ;; esac)"
  r=$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$tag-gs-c1.jsonl); echo "$tag gs $r" >> $O/identity.txt
  xlog "$tag identity: greedy set c1 vs bz d0: $r$(case $r in IDENTICAL*) ;; *) echo " -- reported, not hidden" ;; esac)"
}
serve() {   # serve TAG [OVERLAY] [CONF]: the stack2 n5 start (-v), restart + (measure: check) + modes; 0 healthy
  local tag=$1 ov=${2:-} cf=${3:-}
  stackenv n5 "$ov" "$cf"; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive -v"
  export XS2_ENVFILE=$O/$tag.env; writeenv $XS2_ENVFILE
  xlog "$tag: start IMAGE=$IMAGE $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ' | cut -c1-1500)"
  restart $tag || { keeplog $tag-fail; xlog "$tag: VIRHE not healthy: $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; return 1; }
  modes $tag $ov || { down; return 1; }
  return 0
}
stepp() {   # stepp TAG: stepprobe essay,code 512 (c1, the stack2 pin) x2 + the Finnish essay x2
  run "(cd $D && timeout 900 python3 stepprobe.py essay,code 512 > $O/$1-probe1.txt 2>&1; timeout 900 python3 stepprobe.py essay,code 512 > $O/$1-probe2.txt 2>&1; timeout 600 python3 stepprobe.py '$FINESSAY' 512 > $O/$1-probefi.txt 2>&1; timeout 600 python3 stepprobe.py '$FINESSAY' 512 >> $O/$1-probefi.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  xlog "$1 stepprobe (stack2 pin, adaptive, c1, 512 tok): $(cat $O/$1-probe1.txt $O/$1-probe2.txt $O/$1-probefi.txt 2>/dev/null | grep '^J ' | python3 -c 'import json,sys
for l in sys.stdin:
    d=json.loads(l[2:]); print("%s %.2f tok/s %.2f tok/step acc %.3f;" % (d["prompt"][:12], d["decode_tps"], d["tokens_per_step"], d["acceptance"]), end=" ")' 2>&1 | cut -c1-900)"
}
sampled() {   # sampled TAG: the 1-user probe at the server-default sampling (no temperature sent; seed SEED), essay / code / edit x2
  run "(cd $D && timeout 1500 python3 -I -B $PRB --mode sampled --seed $SEED --runs 2 --tag $1 > $O/$1-samp.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  xlog "$1 sampled 1-user (server-default sampling, seed $SEED, 2 runs; mean, t95): $(grep '^PROBE ' $O/$1-samp.txt | cut -c7- | cut -c1-900) $(grep -m1 -E 'Traceback|Error' $O/$1-samp.txt | cut -c1-200)"
}
mbs() {   # mbs TAG: mb c4 and c8 (coding + reasoning at the server-default sampling, --dur 75 --warm 10), as the stack items run them
  local ph
  for ph in c4 c8; do
    run "(cd $D && timeout 1200 python3 mb.py --tag da-$1-$ph --phases $ph --dur 75 --warm 10 > $O/$1-mb-$ph.txt 2>&1)"
    [ "$DRY" = 1 ] || xlog "$1 mb $ph: $(grep '^MB ' $O/$1-mb-$ph.txt | cut -c4- | python3 -c 'import json,sys
try:
    d=json.loads(sys.stdin.read())
    print("aggregate %s tok/s, per-stream p50 %s, %s tok/step, acc %s, %s ms/step, TTFT p50 %s" % tuple(d.get(k) for k in ("agg_dec_tps", "per_stream_dec_p50", "tok_per_decode_step", "acceptance", "ms_decode_step", "ttft_p50")))
except Exception as e: print("no MB line (%s)" % type(e).__name__)' 2>&1 | cut -c1-400)"
  done
}
tdocker() {   # tdocker NAME GPU ARGS...: the trainer image, one card (GPU: 0..3) or none (-), the model ro, /w = runs/xdsm/da/stats
  local n=$1 g=$2; shift 2
  local dev="" w="-v $STATSDIR:/w"; [ "$g" != - ] && dev="--device /dev/kfd --device /dev/dri --group-add $(stat -c %g /dev/kfd) --group-add video -e HIP_VISIBLE_DEVICES=$g"
  [ "$g" = - ] && w=""                              # the CPU selftest mounts no output directory
  echo "docker run --rm --name da-$n --network none -u $(id -u):$(id -g) -e HOME=/tmp -e PYTHONPATH=/env:/t -e OMP_NUM_THREADS=8 $dev --ipc host --shm-size 8g -v $TENV:/env:ro -v $TR:/t:ro -v $MODELDIR:/m:ro $w --entrypoint python3 $TIMG /t/datrain.py $*"
}
trap 'unfollow; down; docker rm -f da-v0 da-v1 da-v2 da-v3 da-val0 da-val1 da-val2 da-val3 > /dev/null 2>&1' EXIT
trap 'exit 143' TERM INT

# ---------------------------------------------------------------- checks (every phase; DRYRUN checks everything)
HMISS=""
for fn in log run mkd restart check down evict stopfile best_vars desc serve_args; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS lib:$fn"; done
for fn in xlog stackenv writeenv keeplog follow unfollow fn4get nudge pinset modes cmpfile gcmp pcreset ident serve stepp sampled mbs tdocker; do
  [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
for f in $Y $DAA $GEN $EV $PRB $D/mb.py $TR/datrain.py $TR/dspark.py $TR/radread.py $GS $D/ds41_tests.py $D/stepprobe.py $D/fastcheck.py \
         $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDECAR $PUB/wiki-fi.jsonl $PUB/wiki-en.jsonl; do
  [ -e $f ] || HMISS="$HMISS file:$f"; done
for f in $DAA $GEN $EV $PRB $D/mb.py $TR/datrain.py $TR/dspark.py $TR/radread.py; do python3 -I -B -c 'import ast,sys; ast.parse(open(sys.argv[1]).read())' $f 2>/dev/null || HMISS="$HMISS parse:$f"; done
python3 -I -B $GEN selftest > /dev/null 2>&1 || HMISS="$HMISS selftest:dagen.py"
python3 -I -B $EV selftest > /dev/null 2>&1 || HMISS="$HMISS selftest:daeval.py"
python3 -I -B $DAA selftest > /dev/null 2>&1 || HMISS="$HMISS selftest:daanat.py"
for f in $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl; do [ -s $f ] || HMISS="$HMISS ref:$f"; done
docker image inspect $IMAGE > /dev/null 2>&1 || HMISS="$HMISS image:$IMAGE"
IMGOK=$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -qzF RADIANCE_KLD_DUMP /opt/radiance/bin/radiance && grep -qzF RADIANCE_WEIGHT_OVERLAY /opt/radiance/bin/radiance && grep -qzF RADOVL1 /opt/radiance/bin/radiance && grep -qzF RADIANCE_DS41_DA_TAPS /opt/radiance/share/radiance/architectures/deepseek41.so && grep -qzF RADIANCE_DS41_DSPARK_WALK /opt/radiance/share/radiance/architectures/deepseek41.so && grep -qzF dspark_walk_step /opt/radiance/share/radiance/kernels/libr4d.so && grep -qzF reset_prefix_cache /opt/radiance/bin/radiance && grep -qzF "proposes%s" /opt/radiance/bin/radiance && echo yes' 2>/dev/null)
[ "$IMGOK" = yes ] || HMISS="$HMISS image-literals:$IMAGE"
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=${STATSDIR} ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config)"
if [ $PHASE = train ]; then
  docker image inspect $TIMG > /dev/null 2>&1 || HMISS="$HMISS image:$TIMG"
  [ -e $TENV/torch/__init__.py ] || HMISS="$HMISS torchenv:$TENV"
  if [ -z "$HMISS" ]; then
    STT=$(timeout 600 $(tdocker selftest - selftest) 2>&1)
    echo "$STT" | grep -q '"ok": true' || HMISS="$HMISS selftest:datrain.py($(echo "$STT" | tail -2 | tr '\n' ' ' | cut -c1-200))"
    [ "$DRY" = 1 ] && echo "+ datrain.py selftest (CPU, the trainer image + torchenv): $(echo "$STT" | grep '^DAT ' | cut -c1-300)"
  fi
fi
INPUTS=""
case $PHASE in
  train)   for f in $STATSDIR/kl/kld.json $STATSDIR/kl/dump-dspark_main_normed.json $STATSDIR/corpus/val.json; do [ -s $f ] || INPUTS="$INPUTS $f"; done ;;
  measure) [ -s $STATSDIR/ovl/best.ovl ] || INPUTS="$INPUTS $STATSDIR/ovl/best.ovl" ;;
esac
if [ "$DRY" = 1 ]; then
  echo "+ phase $PHASE, item $ITEM, image $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)), out $O, GENMIN $GENMIN, TRAINMIN $TRAINMIN"
  echo "+ D=$D R=$R C=$C MODELDIR=$MODELDIR URL=$URL CN=$CN STATSDIR=$STATSDIR TENV=$TENV TIMG=$TIMG"
  echo "+ disk: $(df -h / | awk 'NR==2{print $4}') free on /"
  echo "+ inputs of this phase: ${INPUTS:-all present}$([ -n "$INPUTS" ] && echo " (produced by the earlier phase; checked again when it runs)")"
  [ -z "$HMISS" ] && echo "+ helpers: every file present and parsing, every function defined, selftests pass, image literals (KLD_DUMP / WEIGHT_OVERLAY / DA_TAPS / W1 / -v proposals), yaml accepted -- OK" \
                  || { echo "+ MISSING:$HMISS"; exit 1; }
else
  [ -z "$HMISS" ] || { xlog "$ITEM: VIRHE helpers missing:$HMISS"; exit 1; }
  [ -z "$INPUTS" ] || { xlog "$ITEM: VIRHE inputs missing (the earlier phase did not produce them):$INPUTS -> nothing run"; exit 1; }
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "$ITEM: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  free=$(df -B1G --output=avail / | tail -1)
  need=$([ $PHASE = dump ] && echo 8 || echo 3)
  [ "$free" -ge $need ] || { xlog "$ITEM: VIRHE only $free GB free on / (phase $PHASE needs >= $need)"; exit 1; }
fi
mkd $O $O/guard $STATSDIR $STATSDIR/corpus $STATSDIR/ovl

# ---------------------------------------------------------------- dump
if [ $PHASE = dump ]; then
  xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) | lever DA pilot (a) the dump: validation set (fixed 5, -v) + self-generated set ($GENMIN min, c8, stack2 pin) + public set -> KL record + main_x | disk $(df -h / | awk 'NR==2{print $4}') free"
  evict
  tag=dap-gen
  if serve $tag; then
    if pinset $tag-val "$FIX" "$ANAT"; then
      follow $O/$tag-engine.log
      run "timeout 900 python3 -I -B $GEN val $URL $O/$tag-engine.log $STATSDIR/corpus > $O/dagen-val.txt 2>&1"
      unfollow
      [ "$DRY" = 1 ] || xlog "$tag validation set: $(grep '^DAGEN ' $O/dagen-val.txt | cut -c7- | tr '\n' ' ' | cut -c1-1500)"
    else
      xlog "$tag: VIRHE the fixed-window pin did not apply -> no validation set (the port's validation will be skipped)"
    fi
    if pinset $tag-gen "$FIX"; then
      run "timeout $(( (GENMIN + 6) * 60 )) python3 -I -B $GEN gen $URL $STATSDIR/corpus --minutes $GENMIN --conc 8 --pubdir $PUB > $O/dagen-gen.txt 2>&1"
      [ "$DRY" = 1 ] || xlog "$tag self-generated set: $(grep '^DAGEN ' $O/dagen-gen.txt | tail -1 | cut -c7- | cut -c1-500)"
    fi
    run "docker logs $CN 2>&1 | gzip -1 > $O/$tag-engine-full.log.gz"
    down
  fi
  run "python3 -I -B $GEN public $PUB $STATSDIR/corpus/corpus-pub.jsonl --fi 90000 --en 50000 --code 50000 > $O/dagen-pub.txt 2>&1"
  [ "$DRY" = 1 ] || xlog "public set: $(grep '^DAGEN ' $O/dagen-pub.txt | cut -c7- | cut -c1-300)"
  run "cat $STATSDIR/corpus/corpus-val.jsonl $STATSDIR/corpus/corpus-gen.jsonl $STATSDIR/corpus/corpus-pub.jsonl 2>/dev/null > $STATSDIR/corpus/corpus.jsonl"
  [ "$DRY" = 1 ] || xlog "corpus: $(python3 -c 'import json,sys,collections
c=collections.Counter(); t=collections.Counter()
for l in open(sys.argv[1]):
    d=json.loads(l); k=d["source"].split(":")[0]; c[k]+=1; t[k]+=len(d["ids"]) if "ids" in d else len(d["prompt"])//4
print("%d documents: %s (tokens; public ones estimated at 4 chars a token)" % (sum(c.values()), ", ".join("%s %d / %d" % (k, c[k], t[k]) for k in sorted(c))))' $STATSDIR/corpus/corpus.jsonl 2>&1 | cut -c1-600)"
  # the KL record + the taps (NST 0: the KL mode drafts nothing; the drafter's taps and main_x by RADIANCE_DS41_DA_TAPS=1)
  tag=dap-kl
  run "rm -rf $STATSDIR/kl"
  stackenv kl; export MODEL=$M225
  export XS2_ENVFILE=$O/$tag.env; writeenv $XS2_ENVFILE
  KLARGS="--tp 4 --tp-wire exact --placement expert_tiered --host-pool-mib 19456 --ngram-placement ram --max-num-batched-tokens 4160 --checkpoint-interval 2048 --kv-cache-dtype bf16 --max-num-seqs 8 --max-model-len 524288 --prefill-span 2 --kld-record /stats/kl --kld-corpus /stats/corpus/corpus.jsonl --kld-top 32"
  xlog "$tag: KL record start IMAGE=$IMAGE args: $KLARGS env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ' | cut -c1-1200)"
  if [ "$DRY" = 1 ]; then
    echo "+ docker rm -f $CN; ARGS='$KLARGS' docker compose --project-directory $C -f $Y up -d; wait for the container to exit (<= 50 min); docker logs > $O/$tag.log"
  else
    docker rm -f $CN > /dev/null 2>&1
    t0=$(date +%s)
    ARGS="$KLARGS" docker compose --project-directory $C -f $Y up -d > $O/up-$tag.txt 2>&1
    for i in $(seq 1 600); do
      [ "$(docker inspect -f '{{.State.Status}}' $CN 2>/dev/null)" = running ] || break
      stopfile && { xlog "$tag: sweep.stop -> stopped"; break; }
      sleep 5
    done
    rc=$(docker inspect -f '{{.State.ExitCode}}' $CN 2>/dev/null)
    docker logs $CN > $O/$tag.log 2>&1
    down
    xlog "$tag: KL record exit ${rc:-?} after $(( $(date +%s) - t0 )) s: $(grep -E 'kld: (reference recorded|RADIANCE_KLD_DUMP|[0-9]+ documents in)|lever DA tap dump|^E ' $O/$tag.log | sed 's/^radiance: //' | head -8 | tr '\n' '|' | cut -c1-1500)"
    if [ "${rc:-1}" = 0 ] && [ -s $STATSDIR/kl/kld.json ] && [ -s $STATSDIR/kl/dump-dspark_main_normed.json ]; then
      xlog "$tag: dump OK: $(python3 -c 'import json,sys,os; m=json.load(open(sys.argv[1])); d=json.load(open(sys.argv[2])); print("%d rows, %d documents, main_x %.2f GB, top.bin %.2f GB" % (m["rows"], len(m["docs"]), os.path.getsize(sys.argv[3])/1e9, os.path.getsize(sys.argv[4])/1e9))' $STATSDIR/kl/kld.json $STATSDIR/kl/dump-dspark_main_normed.json $STATSDIR/kl/dump-dspark_main_normed.bin $STATSDIR/kl/top.bin 2>&1) | disk $(df -h / | awk 'NR==2{print $4}') free"
    else
      xlog "$tag: VIRHE the KL record / dump did not complete (exit ${rc:-?}) -> phases b / c will find no input"
    fi
  fi
  unset $SWITCHES
  xlog "$ITEM: done -- $O"
fi

# ---------------------------------------------------------------- train
if [ $PHASE = train ]; then
  xlog "$ITEM: start | lever DA pilot (b) train: the bf16 port validated vs the engine's proposals, 4 variants (one card each, $TRAINMIN min), the best overlay | trainer image $TIMG ($(docker image inspect -f '{{.Id}}' $TIMG 2>/dev/null | cut -c8-19)) + $TENV | disk $(df -h / | awk 'NR==2{print $4}') free"
  run "docker rm -f $CN > /dev/null 2>&1"
  # 1. validation of the port: 4 numerics variants (fp8 activations into the routed experts on / off x into the shared expert
  #    off / on), the head in f32, one card each
  run "$(tdocker val0 0 validate --device cuda:0 --data /w/kl --val /w/corpus/val.json --out /w/ovl/val0.json --head32 --variants 1) > $O/val0.txt 2>&1 &"
  run "$(tdocker val1 1 validate --device cuda:0 --data /w/kl --val /w/corpus/val.json --out /w/ovl/val1.json --head32 --variants 1 --no-expert-aq) > $O/val1.txt 2>&1 &"
  run "$(tdocker val2 2 validate --device cuda:0 --data /w/kl --val /w/corpus/val.json --out /w/ovl/val2.json --head32 --variants 1 --shared-aq) > $O/val2.txt 2>&1 &"
  run "$(tdocker val3 3 validate --device cuda:0 --data /w/kl --val /w/corpus/val.json --out /w/ovl/val3.json --head32 --variants 1 --no-expert-aq --shared-aq) > $O/val3.txt 2>&1 &"
  run "wait"
  [ "$DRY" = 1 ] || xlog "validation of the port vs the engine's own proposals (validation set, the engine's W1 walk at fixed window 5): $(cat $O/val[0-3].txt | grep -E '\"event\": \"validate\"' | cut -c5- | tr '\n' ' ' | cut -c1-1400) $(cat $O/val[0-3].txt | grep -m2 -E 'Traceback|Error' | cut -c1-200)"
  AQ=""
  if [ "$DRY" != 1 ]; then
    AQ=$(python3 -c 'import json,sys
r=[]
for f in sys.argv[1:]:
    try: r.append(json.load(open(f)))
    except Exception: pass
if not r: print("?"); sys.exit()
b=max(r, key=lambda x: (x["whole_block"], x["agree_by_pos"][0]))
print(("" if b["expert_aq"] else "--no-expert-aq") + (" --shared-aq" if b["shared_aq"] else ""))' $STATSDIR/ovl/val0.json $STATSDIR/ovl/val1.json $STATSDIR/ovl/val2.json $STATSDIR/ovl/val3.json)
    [ "$AQ" = "?" ] && { xlog "train: note -- no validation result (no validation set or the port failed); training with the routed-expert fp8 activations on"; AQ=""; }
    P1=$(python3 -c 'import json,sys
r=[]
for f in sys.argv[1:]:
    try: r.append(json.load(open(f)))
    except Exception: pass
print("%.4f" % max(x["agree_by_pos"][0] for x in r) if r and max(x["blocks"] for x in r) > 0 else "none")' $STATSDIR/ovl/val0.json $STATSDIR/ovl/val1.json $STATSDIR/ovl/val2.json $STATSDIR/ovl/val3.json)
    if [ "$P1" != none ] && awk -v p=$P1 'BEGIN{exit !(p < 0.6)}'; then
      xlog "train: VIRHE the port proposes the engine's first token at only $P1 of the validation blocks (< 0.6): the port does not reproduce the engine's drafter -> no training (a model trained under other numerics would not be the engine's)"
      xlog "$ITEM: done -- $O"; exit 1
    fi
    xlog "train: the port's first-token agreement with the engine $P1 (gate >= 0.6; none = no validation set) -> training with the numerics flags [${AQ:-default: routed-expert fp8 activations on, shared off}]"
  fi
  # 2. four variants, one card each
  V0="--lr 2e-5"; V1="--lr 5e-5"; V2="--lr 1e-4"; V3="--lr 5e-5 --pos-decay 1.0"
  for k in 0 1 2 3; do
    eval va=\$V$k
    run "$(tdocker v$k $k train --device cuda:0 --data /w/kl --out /w/ovl/v$k.ovl $va $AQ --max-minutes $TRAINMIN --max-anchors 384 --log-every 20) > $O/train-v$k.txt 2>&1 &"
  done
  run "wait"
  if [ "$DRY" != 1 ]; then
    for k in 0 1 2 3; do
      xlog "train v$k ($(eval echo \$V$k) $AQ): $(grep -E '"event": "(train_start|overlay_written|time_limit)"' $O/train-v$k.txt | cut -c5- | tr '\n' ' ' | cut -c1-700) | last step $(grep '"event": "step"' $O/train-v$k.txt | tail -1 | cut -c5- | cut -c1-300) $(grep -m2 -E 'Traceback|Error|error' $O/train-v$k.txt | cut -c1-200)"
      xlog "train v$k eval before: $(grep '"event": "eval_before"' $O/train-v$k.txt | cut -c5- | cut -c1-900)"
      xlog "train v$k eval after:  $(grep '"event": "eval_after"' $O/train-v$k.txt | cut -c5- | cut -c1-900)"
    done
    BEST=$(python3 -c 'import json,sys,os
best=None
for k in range(4):
    f=os.path.join(sys.argv[1], "train-v%d.txt" % k)
    ev={}
    for l in open(f) if os.path.exists(f) else []:
        if l.startswith("DAT "):
            d=json.loads(l[4:]); ev[d["event"]]=d
    if "eval_after" not in ev or "eval_before" not in ev or "overlay_written" not in ev: continue
    ks=[x for x in ev["eval_after"] if x.startswith(("gen","val")) and isinstance(ev["eval_after"][x], dict)]
    if not ks: continue
    g=sum(ev["eval_after"][x]["tokens_a_step"]/ev["eval_before"][x]["tokens_a_step"] for x in ks)/len(ks)
    if best is None or g>best[1]: best=(k,g)
print("%d %.4f" % best if best else "none")' $O)
    xlog "train: best variant by held-out simulated tokens a step (fixed 5; mean ratio after / before over the generated kinds): $BEST"
    if [ "$BEST" != none ] && awk -v g=${BEST##* } 'BEGIN{exit !(g <= 1.0)}'; then
      xlog "train: no variant raised the held-out simulated tokens a step (best ratio ${BEST##* }) -> no best.ovl: phase c is skipped (it finds no input); the overlays stay for inspection"
    elif [ "$BEST" != none ]; then
      k=${BEST%% *}
      cp $STATSDIR/ovl/v$k.ovl $STATSDIR/ovl/best.ovl
      for j in 0 1 2 3; do [ $j = $k ] || rm -f $STATSDIR/ovl/v$j.ovl; done
      xlog "train: best.ovl = v$k ($(stat -c %s $STATSDIR/ovl/best.ovl) bytes, sha256 $(sha256sum $STATSDIR/ovl/best.ovl | cut -c1-16)); the other overlays deleted | disk $(df -h / | awk 'NR==2{print $4}') free"
    else
      xlog "train: VIRHE no variant finished with an eval -> no overlay, phase c will find no input"
    fi
  fi
  xlog "$ITEM: done -- $O"
fi

# ---------------------------------------------------------------- measure
if [ $PHASE = measure ]; then
  OVL=/stats/ovl/best.ovl
  xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) | lever DA pilot (c) measure: B (overlay $OVL, sha256 $(sha256sum $STATSDIR/ovl/best.ovl 2>/dev/null | cut -c1-16)) then A (the container's drafter), stack2 n5 env | anatomy fixed 5 (prose 20 / code 6 / fi 10 x 384) + stack2 pin stepprobe; identity on B"
  evict
  : > $O/identity.txt 2>/dev/null || true
  for arm in B A X B2; do
    tag=dap-$arm
    ov=""; cf=""; [ $arm != A ] && ov=$OVL
    if [ $arm = X ]; then
      ov=/stats/ovl/ext.ovl; cf=/stats/ovl/ext-conf.f32
      [ -s $STATSDIR/ovl/ext.ovl ] && [ -s $STATSDIR/ovl/ext-conf.f32 ] || { xlog "$tag: no $STATSDIR/ovl/ext.ovl / ext-conf.f32 -> arm X skipped"; continue; }
      [ "$DRY" = 1 ] || xlog "$tag: arm X = the external fine-tune: ext.ovl sha256 $(sha256sum $STATSDIR/ovl/ext.ovl | cut -c1-16), sidecar $(sha256sum $STATSDIR/ovl/ext-conf.f32 | cut -c1-16)"
    fi
    if ! serve $tag "$ov" "$cf"; then
      xlog "$tag: VIRHE no healthy start -> arm skipped"; [ $arm = B ] && { xlog "B failed: A and B2 are not run (no comparison)"; break; }; continue
    fi
    ms=$(check $tag 1); [ "$DRY" = 1 ] || xlog "$tag guard: 1-user $ms ms/step (informational; stack2 A 34-35)"
    if [ $arm != B2 ]; then
      if pinset $tag-anat "$FIX" "$ANAT"; then
        follow $O/$tag-engine.log
        run "timeout 2400 python3 -I -B $EV collect --url $URL --log $O/$tag-engine.log --out $O/$tag --prose 20 --code 6 --fi 10 --max-tokens 384 --top 20 > $O/$tag-collect.txt 2>&1"
        unfollow
        run "python3 -I -B $EV analyse $O/$tag-rows.jsonl > $O/$tag-analyse.txt 2>&1"
        [ "$DRY" = 1 ] || while IFS= read -r l; do xlog "$tag ${l:7:1500}"; done < <(grep '^DALINE ' $O/$tag-analyse.txt)
      else
        xlog "$tag: VIRHE the anatomy pin did not apply"
      fi
    fi
    if pinset $tag-stack "$FIX"; then
      stepp $tag
      sampled $tag
      mbs $tag
    fi
    [ $arm = B ] || [ $arm = X ] && ident $tag-i
    run "docker logs $CN 2>&1 | gzip -1 > $O/$tag-engine-full.log.gz"
    down
  done
  run "python3 -I -B $EV compare $O/dap-A-analyse.txt $O/dap-B-analyse.txt > $O/compare.txt 2>&1"
  [ "$DRY" = 1 ] || while IFS= read -r l; do xlog "B vs A ${l:6:1200}"; done < <(grep '^DACMP ' $O/compare.txt)
  run "python3 -I -B $EV compare $O/dap-A-analyse.txt $O/dap-X-analyse.txt > $O/compare-x.txt 2>&1"
  [ "$DRY" = 1 ] || while IFS= read -r l; do xlog "X (external) vs A ${l:6:1200}"; done < <(grep '^DACMP ' $O/compare-x.txt)
  [ "$DRY" = 1 ] || xlog "summary: $(grep '^DACMP ' $O/compare.txt | cut -c7- | cut -c1-260 | tr '\n' '|') || identity: $(tr '\n' ',' < $O/identity.txt 2>/dev/null)"
  [ "$DRY" = 1 ] || xlog "summary speed (cross-start B A X B2; B = mean of B and B2): $(python3 -c 'import json,sys,os,statistics
O=sys.argv[1]
def probe(tag, f):
    d={}
    try:
        for l in open(os.path.join(O, "%s-%s" % (tag, f))):
            if l.startswith("J "):
                r=json.loads(l[2:]); d.setdefault(r["prompt"], []).append(r)
    except OSError: pass
    return d
def mbv(tag, ph):
    try:
        for l in open(os.path.join(O, "%s-mb-%s.txt" % (tag, ph))):
            if l.startswith("MB "): return json.loads(l[3:]).get("agg_dec_tps")
    except OSError: return None
out=[]
for f, lab in (("samp.txt", "sampled"), ("probe1.txt", "greedy")):
    for p in ("essay", "code", "edit"):
        v={a: [r["decode_tps"] for r in probe("dap-"+a, f).get(p, []) if r.get("decode_tps")] for a in ("A", "B", "B2", "X")}
        A=v["A"]; B=v["B"]+v["B2"]; X=v["X"]
        if A and B:
            out.append("%s %s %.1f (B %s / B2 %s) vs A %.1f tok/s (%+.1f %%)%s" % (lab, p, statistics.mean(B), "/".join("%.1f" % x for x in v["B"]), "/".join("%.1f" % x for x in v["B2"]), statistics.mean(A), 100*(statistics.mean(B)/statistics.mean(A)-1),
                       "; X %.1f (%+.1f %% vs A)" % (statistics.mean(X), 100*(statistics.mean(X)/statistics.mean(A)-1)) if X else ""))
for ph in ("c4", "c8"):
    a, b, b2, x = mbv("dap-A", ph), mbv("dap-B", ph), mbv("dap-B2", ph), mbv("dap-X", ph)
    bb=[y for y in (b, b2) if y]
    if a and bb: out.append("mb %s %.1f (B %s / B2 %s) vs A %.1f tok/s (%+.1f %%)%s" % (ph, statistics.mean(bb), b, b2, a, 100*(statistics.mean(bb)/a-1), "; X %.1f (%+.1f %% vs A)" % (x, 100*(x/a-1)) if x else ""))
print(" | ".join(out) or "none")' $O 2>&1 | cut -c1-1500)"
  # the dump (3.2 GB) stays for item 57-p4zyzzzzz-ds-iej-sz-embscreen (the port's drafter-embedding screen), which deletes it
  [ "$DRY" = 1 ] || xlog "cleanup: the KL dump kept for iej-sz-embscreen (it deletes it); $(du -sh $STATSDIR 2>/dev/null | cut -f1) in $STATSDIR | disk $(df -h / | awk 'NR==2{print $4}') free"
  xlog "$ITEM: done -- $O"
fi
unset $SWITCHES
