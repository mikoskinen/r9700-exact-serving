#!/bin/bash
# ds41-arena-level-check.sh IMAGE MODEL.rad [ENV=VALUE ...] -- deepseek41's arena sizing declares with the REAL model's
# parameters, no device (lever Z0-A, ds41-z0).
#
# One host-backend TP4 start (RADIANCE_HOST_DEVICES=4: four fake host devices, nothing placed) of IMAGE's radiance on
# MODEL with --debug-graph: Engine::declare runs whole, the real declare and then one sizing declare
# (RadBuildCtx::shape_probe) a rung of RADIANCE_DS41_ARENA_DECODE_ROWS and of the ladder. The check passes when every
# decode rung asked for (default 8,48) is accepted ("activation arena: N tokens use") and no level is refused ("no
# level for N-token steps ... the sizing declare failed"); it exits 1 otherwise, printing the arena lines and the
# declare's refusals. NST (default 5) is --num-speculative-tokens: at NST 5 the DSpark head's query rows Q are
# max_seqs x 6 capped by max_tok (40 at 8 sequences), above the 8-row rung -- the case iej-z0ab / iej-z0ac-w stopped on
# (11.11.2026: the head's all_gather numel Q x R x 2 and the drafter block's router, GlmRouter's rows <= max_tok).
# The env is hxzzy-stack2's NST 5 production-test set (the switches that change the declared graph), then ENV=VALUE.
# Host kernels: dspark_walk_step, moe_gather_all_reduce and the fused mhc/attn ops do not resolve on the host backend
# (in the real declare as in a level's); those lines are ignored. CPU only, ~2-4 min, 24 GiB.
#   NST       --num-speculative-tokens (default 5)
#   ROWS      RADIANCE_DS41_ARENA_DECODE_ROWS (default 8,48)
#   SEQS      --max-num-seqs (default 8)
#   CONF      the DSpark confidence table, mounted beside MODEL (default MODEL's dir/dspark-conf.f32 when present)
#   OUT       the stderr log (default ./ds41-arena-level-check.err)
set -u
IMG=${1:?IMAGE}; M=${2:?MODEL.rad}; shift 2
[ -f "$M" ] || { echo "ds41-arena-level-check: $M not found"; exit 2; }
NST=${NST:-5}; ROWS=${ROWS:-8,48}; SEQS=${SEQS:-8}; OUT=${OUT:-$PWD/ds41-arena-level-check.err}
MD=$(cd "$(dirname "$M")" && pwd); MB=$(basename "$M")
CONF=${CONF:-$MD/dspark-conf.f32}
GPTAB='q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422'
E=(-e RADIANCE_DS41_IDXKEY=mxfp4 -e RADIANCE_HEAT_PREFILL_DECAY=0.7 -e RADIANCE_HEAT_DECODE_PIN=0.8 -e RADIANCE_HEAT_MOVES=16
   -e RADIANCE_DS41_LEVELS=1 -e RADIANCE_ARENA_LEVEL_MAX=1024 -e RADIANCE_STAGE_SPAN=0 -e RADIANCE_DS41_ARGATHER=on
   -e RADIANCE_DS41_ROUTER_FUSE=0 -e R4D_DS41_MIX_FUSE=0 -e R4D_DS41_QLDS=0 -e RADIANCE_DS41_AR_PCN=0 -e RADIANCE_HOST_DEVICES=4
   -e RADIANCE_DS41_GG_GEMV=2 -e RADIANCE_AQL_ACQ=rule-w -e RADIANCE_DS41_KVFOLD=1 -e RADIANCE_DS41_OFOLD=1 -e RADIANCE_DS41_ONEFILL=1
   -e RADIANCE_PLACE_TURNOVER=0 -e RADIANCE_DS41_GG_PF=1 -e RADIANCE_DS41_GG_PF_TABLE=$GPTAB -e RADIANCE_DS41_F32_PF=25
   -e RADIANCE_DS41_DSPARK_WALK=1 -e RADIANCE_DS41_ARGATHER_MAX=48 -e RADIANCE_DS41_QNFOLD=1 -e RADIANCE_DS41_SWFOLD=1
   -e RADIANCE_DRAFT_CTXSKIP=0 -e RADIANCE_DS41_ROUTER_BF16=1 -e RADIANCE_DS41_EMB_RAM=1
   -e RADIANCE_PLACE_SHADOW_SLOTS=16 -e RADIANCE_XPREFETCH=0 -e RADIANCE_DS41_ARENA_DECODE_ROWS=$ROWS)
[ -f "$CONF" ] && [ "$(dirname "$CONF")" = "$MD" ] && E+=(-e RADIANCE_DS41_DSPARK_CONF=/m/$(basename "$CONF"))
for e in "$@"; do E+=(-e "$e"); done
NAME=ds41-arena-level-check-$$
timeout -k 15 900 nice -n 19 docker run --rm --init --name $NAME --network none --memory 24g --memory-swap 24g --cpus 4 \
  "${E[@]}" -v "$MD":/m:ro --entrypoint /opt/radiance/bin/radiance "$IMG" \
  --model /m/$MB --tp 4 --tp-wire exact --placement expert_tiered --host-pool-mib 19456 --max-model-len 524288 \
  --max-num-seqs $SEQS --max-num-batched-tokens 4160 --checkpoint-interval 2048 --num-speculative-tokens $NST \
  --draft-window adaptive --kv-cache-dtype bf16 --ngram-placement ram --prefill-span 2 --prefill-interleave 8 \
  --host 127.0.0.1 --port 18099 --debug-graph > /dev/null 2> "$OUT"
rc=$?
docker rm -f $NAME > /dev/null 2>&1
bad=0
grep -q "activation arena: " "$OUT" || { echo "ds41-arena-level-check: no activation-arena line (radiance exit $rc)"; bad=1; }
for r in ${ROWS//,/ }; do
    grep -qE "activation arena: (.*, )?$r tokens use " "$OUT" || { echo "ds41-arena-level-check: the $r-token rung is not accepted"; bad=1; }
done
grep -q "no level for" "$OUT" && bad=1
grep -E "activation arena|sizing declare|refused \(|lever Z0" "$OUT" | grep -v "control vectors" | sed 's/^radiance: //' |
    cut -c1-260 | sort | uniq -c | sort -rn | head -14
echo "ds41-arena-level-check: $([ $bad = 0 ] && echo PASS || echo FAIL) (NST $NST, rungs $ROWS, radiance exit $rc, log $OUT)"
exit $bad
