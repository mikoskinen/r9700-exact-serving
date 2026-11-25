#!/bin/bash
# dsm-sd.sh (11.11.2026, queue-ar/dsmain/sd, ds-main-port SD agent) -- NOT a queue item; the body of the lever SD item. Tag
# "RESULT xsd ", out runs/xdsm/sd (SD_OUT). LEVER SD, SAMPLED DRAFTING (the user's GO 11.11.: distribution-exact, NOT
# byte-identical to non-speculative sampling at temperature > 0, behind a switch, default off): at temperature > 0 the DSpark
# W1 walk SAMPLES its drafts, x ~ Q (the walk's eight best keys of the whole vocabulary, softmax at the request's temperature),
# and the verify keeps x with probability min(1, P(x) / Q(x)), else draws from norm(max(0, P - Q)); temperature 0 drafts and
# verifies exactly as today. ds41-sd 37ac893+ (on ds41-int3 1116248): libr4d dspark_walk_sample_step, deepseek41's coupling
# planes under --draft-coupling on, the engine's fn4 draft_sample (0 today / 1 sampled at the request's temperature / 2..200 at
# temperature x v/100), the sampler's existing mtp_couple_verify (spechub).
#
# ONE NST 5 START (stack2's n5 env as it ran -- walk 1, the small n5 set, the VRAM levers, DL lend 1 + backoff 3, hybrid 1,
# drafter_mx 1: dsm-sam.sh's stackenv / FIX5, copied -- + --draft-coupling on + RADIANCE_DRAFT_SAMPLE=0), all in process:
#   modes        the start words, W1 declared over 4 ranks, "DSpark lever SD ... declared", "lever SD: fn4 draft_sample 0".
#   ABBA         A = draft_sample 0 (today's agreement rule), B = draft_sample 1. A block: the pin (FIX5 + the arm +
#                epoch_bump, applied at a step; FIX5 = dsm-sam.sh's + draft_probe 16, the default, so the chi-square's 4096
#                does not outlive its phase) -> 1 user essay / code / edit at the SERVER DEFAULT sampling (no temperature
#                sent = the engine's 1; per-request seed 1234) WITH THINKING ON as real traffic (sdprobe.py: enable_thinking
#                true + reasoning_effort medium; reasoning chars logged) -> mb c4 / c8 (mb.py --phases c4,c8 --dur 75 --warm 10,
#                as ggblocksdz2 runs it: server-default sampling, reasoning_effort medium) -> fn4 "sample" counters a block.
#   ORDER        chi-square FIRST (a gate: the speed blocks run only when p >= 1e-3 and S bound coupled rounds), then
#                ABBA + the C screen, then identity (always).
#   C screen     one block at draft_sample 70 (Q at 0.7 x the temperature; information, n 1).
#   chi-square   the law on the card: a fixed raw prompt (sdchi.py), 2000 seeded requests at the server default, the first two
#                tokens' text as the category (the second is the verify's row 0, the first a drafter proposes), under
#                S = draft_sample 1 with the fixed window (draft_adaptive 0, depth0 0, hybrid 0, dspark_conf 0: every cycle
#                drafts 5) and under N = draft_sample 0 + draft_depth0 2 + draft_probe 4096 (non-speculative), disjoint seeds;
#                chi-square of homogeneity, the p-value reported; the drafted / bound counters prove the coupling ran.
#   identity     temperature 0 under B (draft_sample 1) and under A: the 28k copy + the 20-answer greedy set at c1 vs bz d0,
#                and B's files vs A's.
#   summary      per arm mean +- t95 over blocks; B vs A % (pooled and by ABBA pairs) for 1-user tok/s, tokens a step,
#                acceptance; c4 / c8 aggregate tok/s, per-stream p50, tokens a step, acceptance, ms a step.
# ~33 min (evict 1, start ~5, 4 blocks x ~3.6, C ~3.6, chi-square ~5, identity ~4). NO cc_sim, no 2.9; no power / host / fan
# settings; production compose files read only (common.yaml via extends).
# DRYRUN=1 bash dsm-sd.sh : prints every command; checks the image (its SD literals), every helper (files present and parsing,
#   functions defined), the yaml (docker compose config); no server, no GPU, nothing written. A missing helper fails it.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
Y=$HD/ds41-sd-tp4.yaml
QA=$HOME/gpu/ds41/exl3-test/queue-ar; TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=${SD_OUT:-$R/xdsm/sd}; export STATSDIR=$O/stats
RD=$R/xdsm/draft
export IMAGE=${SD_IMAGE:?SD_IMAGE (the ds41-sd image) is set by the queue item}
ITEM=${SD_ITEM:-57-p4zyzzzzz-ds-sd}
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
PRB=$HD/sdprobe.py; CHI=$HD/sdchi.py; SEED=1234
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
SIDECAR=dspark-conf.f32
GPTAB=q6_K:8192x1280=426,q6_K:5120x2048=426,q6_K:5120x768=426,q6_K:5120x512=426,q6_K:25600x6144=426,q8_0:5120x768=422,q8_0:5120x512=422
FIX5='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1, "draft_adaptive": 1, "draft_depth0": 1, "draft_hybrid_match": 8, "drafter_mx": 1, "gg_pf": 1, "f32_pf": 25, "dspark_walk": 1, "ds41_argather": 48, "idx_split": 1, "ds41_qnfold": 1, "ds41_swfold": 1, "draft_ctxskip": 1, "dspark_conf": 1, "dspark_conf_rows": 1, "draft_hybrid": 1, "draft_lend": 1, "draft_probe_backoff": 3, "draft_probe": 16}'
ARM_A='{"draft_sample": 0}'; ARM_B='{"draft_sample": 1}'; ARM_C='{"draft_sample": 70}'
CHI_S='{"draft_sample": 1, "draft_adaptive": 0, "draft_depth0": 0, "draft_hybrid": 0, "dspark_conf": 0}'
CHI_N='{"draft_sample": 0, "draft_depth0": 2, "draft_probe": 4096}'
CHIN=${SD_CHIN:-2000}; ORDER=${SD_ORDER:-ABBA}
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
  RADIANCE_WEIGHT_OVERLAY RADIANCE_DS41_DA_TAPS RADIANCE_KLD_DUMP RADIANCE_DRAFT_SAMPLE"
xlog() { log "RESULT xsd $*"; }
# stackenv: dsm-sam.sh's n5 stackenv (the stack2 env as it ran: XS2_VRAM=all, XS2_LEND=1, walk 1, the small n5 set, turnover 0)
# + RADIANCE_DRAFT_SAMPLE=0 (lever SD registered, starts at 0); the start-time words also go to $XS2_ENVFILE (the yaml's env_file)
stackenv() {
  local kv w
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  export RADIANCE_PLACE_TURNOVER=0
  w="RADIANCE_DS41_GG_PF=1 RADIANCE_DS41_GG_PF_TABLE=$GPTAB RADIANCE_DS41_F32_PF=25"
  w="$w RADIANCE_DS41_ROUTER_BF16=1 RADIANCE_DS41_EMB_RAM=1 RADIANCE_PLACE_SHADOW_SLOTS=16 RADIANCE_XPREFETCH=0"
  w="$w RADIANCE_DS41_ARENA_DECODE_ROWS=48 RADIANCE_DS41_DSPARK_WALK=1 RADIANCE_DS41_ARGATHER_MAX=48 RADIANCE_DS41_QNFOLD=1 RADIANCE_DS41_SWFOLD=1"
  w="$w RADIANCE_DRAFT_CTXSKIP=0 RADIANCE_DS41_DSPARK_CONF=/models/$SIDECAR RADIANCE_DRAFT_LEND=0 RADIANCE_DRAFT_PROBE_BACKOFF=0 RADIANCE_DRAFT_LEND_BORROW=0"
  w="$w RADIANCE_DRAFT_SAMPLE=0"
  for kv in $w; do export "$kv"; done
  LEVENV=$w
}
writeenv() { [ "$DRY" = 1 ] && { echo "+ printf '%s\\n' $LEVENV > $1"; return 0; }; printf '%s\n' $LEVENV > $1; }
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
pinset() {   # pinset TAG JSON...: the merged switches + epoch_bump, applied at a step (epoch moved, nothing pending)
  local tag=$1 body s0 s1; shift
  body=$(python3 -c 'import json,sys
d={}
for a in sys.argv[1:]: d.update(json.loads(a))
d["epoch_bump"]=1; print(json.dumps(d))' "$@")
  [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $body; a 4-token request; GET -> applied, nothing pending"; return 0; }
  s0=$(fn4get)
  curl -s -m 10 -X POST -H 'Content-Type: application/json' -d "$body" $URL/admin/fn4 > $O/pin-$tag.post 2>&1
  nudge; s1=$(fn4get)
  python3 -c 'import json,sys
s0, s1, w = (json.loads(x) for x in sys.argv[1:4])
w.pop("epoch_bump", None)
ok = all(s1["switches"].get(k) == v for k, v in w.items()) and not s1["pending"] and s1["epoch"] > s0["epoch"]
bad = [k for k, v in w.items() if s1["switches"].get(k) != v]
print("applied" if ok else "NOT applied: %s" % bad); sys.exit(0 if ok else 1)' "$s0" "$s1" "$body" > $O/pin-$tag.txt 2>&1
  local rc=$?
  xlog "$tag pin: $(cat $O/pin-$tag.txt) $([ $rc = 0 ] || echo "(POST: $(head -c 300 $O/pin-$tag.post))") -- switches $(echo "$s1" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps(d.get("switches")))' 2>/dev/null | cut -c1-700)"
  return $rc
}
smp() {   # smp FILE: GET /admin/fn4 "sample" + "accept" to FILE (the SD counters)
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 sample/accept > $1"; return 0; }
  fn4get | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps({"sample": d.get("sample"), "accept": d.get("accept"), "draft": d.get("draft")}))' > $1 2>/dev/null
}
smpd() {   # smpd F0 F1: the delta of the SD counters between two smp files, one line
  [ "$DRY" = 1 ] && { echo "(dry)"; return 0; }
  python3 -c 'import json,sys
a, b = (json.load(open(f)) for f in sys.argv[1:3])
o = []
for g in ("sample", "accept"):
    x, y = a.get(g) or {}, b.get(g) or {}
    o.append(g + " " + " ".join("%s %+d" % (k, (y.get(k) or 0) - (x.get(k) or 0)) for k in y if k != "capable"))
print("; ".join(o))' $1 $2 2>&1 | cut -c1-500
}
modes() {   # modes TAG: the start words in the container, the W1 line, the SD declared line, the engine's draft_sample line
  [ "$DRY" = 1 ] && { echo "+ modes $1: start words, W1 declared over 4 ranks, DSpark lever SD declared, lever SD: fn4 draft_sample 0, MTP/DFlash coupled LIVE"; return 0; }
  local CE ok=1 kv WL SL EL CL
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in RADIANCE_PLACE_TURNOVER=0 RADIANCE_DS41_GG_GEMV=2 RADIANCE_DS41_ROUTER_BF16=1 RADIANCE_DS41_DSPARK_WALK=1 RADIANCE_DRAFT_LEND=0 RADIANCE_DRAFT_SAMPLE=0; do
    echo "$CE" | grep -qxF "$kv" || { xlog "$1: VIRHE $kv not in the container"; ok=0; }; done
  WL=$(docker logs $CN 2>&1 | grep -m1 'DSpark lever W1.*fn4 dspark_walk=' | sed 's/^radiance: //' | cut -c1-200)
  case "$WL" in *"over 4 rank(s)"*"dspark_walk=1 "*) ;; *) xlog "$1: VIRHE no W1 declared line over 4 ranks"; ok=0 ;; esac
  SL=$(docker logs $CN 2>&1 | grep -m1 'DSpark lever SD: the sampled full-vocabulary walk declared' | sed 's/^radiance: //' | cut -c1-260)
  [ -n "$SL" ] || { xlog "$1: VIRHE no 'DSpark lever SD ... declared' line"; ok=0; }
  EL=$(docker logs $CN 2>&1 | grep -m1 'lever SD: fn4 draft_sample' | sed 's/^radiance: //; s/^[A-Z] *//' | cut -c1-260)
  case "$EL" in *"draft_sample 0 at start"*) ;; *) xlog "$1: VIRHE no 'lever SD: fn4 draft_sample 0 at start' line"; ok=0 ;; esac
  CL=$(docker logs $CN 2>&1 | grep -m1 'coupled LIVE' | sed 's/^radiance: //; s/^[A-Z] *//' | cut -c1-200)
  xlog "$1 modes: $WL | $SL | $EL | $CL | lend: $(docker logs $CN 2>&1 | grep -m1 -i 'draft lend' | sed 's/^radiance: //' | cut -c1-160)"
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
ident() {   # ident TAG: temperature 0 under the pinned arm -- the 28k copy and the c1 greedy set vs bz d0
  local tag=$1 r
  pcreset $tag
  smp $O/$tag-smp0.json
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset 0 --tag=-$tag --outdir $O > $O/$tag-copy.txt 2>&1)"
  run "(cd $D && timeout 900 python3 $GS $URL $O/$tag-gs-c1.jsonl --conc 1 --max-tokens 192 > $O/$tag-gs-c1.txt 2>&1)"
  smp $O/$tag-smp1.json
  [ "$DRY" = 1 ] && return 0
  r=$(cmpfile $O/copy-$tag-run1.txt $RD/copy-dsd-d0-i1-run1.txt); echo "$tag copy $r" >> $O/identity.txt
  xlog "$tag identity (temperature 0): 28k copy vs bz d0 (NST 0, stock): $r$(case $r in IDENTICAL*) ;; *) echo " -- reported, not hidden" ;; esac)"
  r=$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$tag-gs-c1.jsonl); echo "$tag gs $r" >> $O/identity.txt
  xlog "$tag identity (temperature 0): greedy set c1 vs bz d0: $r$(case $r in IDENTICAL*) ;; *) echo " -- reported, not hidden" ;; esac) | SD counters: $(smpd $O/$tag-smp0.json $O/$tag-smp1.json)"
}
block() {   # block TAG ARM K: the sampled 1-user probe with thinking (seed SEED) + mb c4 / c8; the SD counters around it
  local t0=$(date +%s)
  smp $O/$1-b$3$2-smp0.json
  run "(cd $D && timeout 1500 python3 -I -B $PRB --mode sampled --seed $SEED --runs 1 --thinking on --effort medium --tag $1-$2$3 > $O/$1-b$3$2-samp.txt 2>&1)"
  smp $O/$1-b$3$2-smp1.json
  run "(cd $D && timeout 1500 python3 mb.py --tag sd-$1-$2$3 --phases c4,c8 --dur 75 --warm 10 > $O/$1-b$3$2-mb.txt 2>&1)"
  smp $O/$1-b$3$2-smp2.json
  [ "$DRY" = 1 ] && return 0
  xlog "$1 block $3 arm $2 1u (sampled, seed $SEED, thinking on): $(grep '^J ' $O/$1-b$3$2-samp.txt | python3 -c 'import json,sys
for l in sys.stdin:
    d=json.loads(l[2:]); print("%s %s tok/s %s tok/step acc %s (%s tok, think %s ch);" % (d["prompt"], d["decode_tps"], d["tokens_per_step"], d["acceptance"], d["tokens"], d.get("reasoning_chars")), end=" ")' 2>&1 | cut -c1-700) $(grep -m1 -hE 'Traceback|Error' $O/$1-b$3$2-samp.txt | cut -c1-200) | SD $(smpd $O/$1-b$3$2-smp0.json $O/$1-b$3$2-smp1.json)"
  xlog "$1 block $3 arm $2 mb: $(grep '^MB ' $O/$1-b$3$2-mb.txt | python3 -c 'import json,sys
for l in sys.stdin:
    d=json.loads(l[3:]); print("%s %s tok/s p50 %s tok/step %s acc %s %s ms;" % (d.get("phase"), d.get("agg_dec_tps"), d.get("per_stream_dec_p50"), d.get("tok_per_decode_step"), d.get("acceptance"), d.get("ms_decode_step")), end=" ")' 2>&1 | cut -c1-600) | SD $(smpd $O/$1-b$3$2-smp1.json $O/$1-b$3$2-smp2.json) | $(thermline $t0 $(date +%s))"
}
thermline() { type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"; }
chi() {   # chi: the law on the card -- S (draft_sample 1, fixed window 5) vs N (non-speculative), 2 x CHIN seeded requests;
          # CHIOK=1 when p >= 1e-3 and S bound coupled rounds (the speed blocks run only then)
  local cs
  CHIOK=0
  pinset sd-chiS "$FIX5" "$CHI_S" || { xlog "chi-square: VIRHE the S pin did not apply -> skipped"; return 1; }
  pcreset sd-chiS
  smp $O/chiS-smp0.json
  run "(cd $D && timeout 1800 python3 -I -B $CHI collect --out $O/chiS.jsonl --n $CHIN --seed0 1 --conc 4 --max-tokens 2 > $O/chiS.txt 2>&1)"
  smp $O/chiS-smp1.json
  pinset sd-chiN "$FIX5" "$CHI_N" || { xlog "chi-square: VIRHE the N pin did not apply -> skipped"; return 1; }
  smp $O/chiN-smp0.json
  run "(cd $D && timeout 1800 python3 -I -B $CHI collect --out $O/chiN.jsonl --n $CHIN --seed0 100001 --conc 4 --max-tokens 2 > $O/chiN.txt 2>&1)"
  smp $O/chiN-smp1.json
  run "python3 -I -B $CHI compare $O/chiS.jsonl $O/chiN.jsonl > $O/chi.txt 2>&1"
  [ "$DRY" = 1 ] && { CHIOK=1; return 0; }
  cs=$(grep -m1 '^CHI ' $O/chi.txt | cut -c5-)
  python3 -c 'import json,sys
c = json.loads(sys.argv[1]); a, b = (json.load(open(f)) for f in sys.argv[2:4])
bound = ((b.get("sample") or {}).get("rounds_bound") or 0) - ((a.get("sample") or {}).get("rounds_bound") or 0)
sys.exit(0 if c.get("p") is not None and c["p"] >= 1e-3 and bound > 0 else 1)' "$cs" $O/chiS-smp0.json $O/chiS-smp1.json 2>/dev/null && CHIOK=1
  xlog "chi-square gate: $([ $CHIOK = 1 ] && echo "PASSED (p >= 1e-3 and S bound coupled rounds) -> the speed blocks run" || echo "VIRHE FAILED (p < 1e-3, or no coupled round bound under S) -> the speed blocks are skipped, identity still runs")"
  xlog "chi-square (on the card; fixed prompt, server default temperature 1, the first two tokens' text, S = draft_sample 1 fixed window 5 seeds 1..$CHIN vs N = non-speculative (draft_sample 0, depth 0 forced, probe 4096) seeds 100001..): $(echo "$cs" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("chi2 %s df %s p-value %s (n %s / %s, %s cells, pooled %s / %s); top %s" % (d.get("chi2"), d.get("df"), d.get("p"), d.get("n_a"), d.get("n_b"), d.get("cells"), d.get("pooled_a"), d.get("pooled_b"), d.get("top")[:6]))' 2>&1 | cut -c1-700) | collect S $(grep -m1 '^COLLECT' $O/chiS.txt | cut -c9- | cut -c1-160) N $(grep -m1 '^COLLECT' $O/chiN.txt | cut -c9- | cut -c1-160) | S counters $(smpd $O/chiS-smp0.json $O/chiS-smp1.json) | N counters $(smpd $O/chiN-smp0.json $O/chiN-smp1.json)"
}
start() {   # start TAG: restart + check (warm-up + fastcheck) + modes; 0 healthy
  local tag=$1 ms
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on"
  export XS2_ENVFILE=$O/$tag.env; writeenv $XS2_ENVFILE
  xlog "$tag: start IMAGE=$IMAGE $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ' | cut -c1-1600)"
  restart $tag || { keeplog $tag-fail; xlog "$tag: VIRHE not healthy: $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; return 1; }
  ms=$(check $tag 1); [ "$DRY" = 1 ] || xlog "$tag guard: 1-user $ms ms/step (informational)"
  modes $tag || { down; return 1; }
}
trap 'down' EXIT
trap 'exit 143' TERM INT

# ---------------------------------------------------------------- checks
HMISS=""
for fn in log run mkd restart check down evict stopfile best_vars desc serve_args; do [ "$(type -t $fn)" = function ] || HMISS="$HMISS lib:$fn"; done
for fn in xlog stackenv writeenv keeplog fn4get nudge pinset smp smpd modes cmpfile gcmp pcreset ident block thermline chi start; do
  [ "$(type -t $fn)" = function ] || HMISS="$HMISS function:$fn"; done
case $ORDER in *[!AB]*|"") HMISS="$HMISS order:$ORDER" ;; esac
for f in $Y $PRB $CHI $GS $D/ds41_tests.py $D/fastcheck.py $D/mb.py $MODELDIR/$M225 $MODELDIR/$M225.sha256ok $MODELDIR/$SIDECAR \
         $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl; do [ -e $f ] || HMISS="$HMISS file:$f"; done
for f in $PRB $CHI $GS $D/ds41_tests.py $D/mb.py; do python3 -I -B -c 'import ast,sys; ast.parse(open(sys.argv[1]).read())' $f 2>/dev/null || HMISS="$HMISS parse:$f"; done
python3 -I -B $PRB --help > /dev/null 2>&1 || HMISS="$HMISS run:sdprobe.py--help"
python3 -I -B $CHI collect --help > /dev/null 2>&1 || HMISS="$HMISS run:sdchi.py--help"
for j in "$FIX5" "$ARM_A" "$ARM_B" "$ARM_C" "$CHI_S" "$CHI_N"; do python3 -c 'import json,sys; json.loads(sys.argv[1])' "$j" 2>/dev/null || HMISS="$HMISS json:$j"; done
docker image inspect $IMAGE > /dev/null 2>&1 || HMISS="$HMISS image:$IMAGE"
IMGOK=$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -qzF draft_sample /opt/radiance/bin/radiance && grep -qzF RADIANCE_DRAFT_SAMPLE /opt/radiance/bin/radiance && grep -qzF draft_lend /opt/radiance/bin/radiance && grep -qzF dspark_walk_sample_step /opt/radiance/share/radiance/kernels/libr4d.so && grep -qzF dspark_walk_sample_step /opt/radiance/share/radiance/architectures/deepseek41.so && grep -qzF mtp_couple_qtemp /opt/radiance/share/radiance/architectures/deepseek41.so && grep -qzF RADIANCE_DS41_DSPARK_WALK /opt/radiance/share/radiance/architectures/deepseek41.so && grep -qzF RADIANCE_DS41_ARENA_DECODE_ROWS /opt/radiance/share/radiance/architectures/deepseek41.so && grep -qzF mtp_couple_verify /opt/radiance/share/radiance/kernels/spechub.so && grep -qzF reset_prefix_cache /opt/radiance/bin/radiance && echo yes' 2>/dev/null)
[ "$IMGOK" = yes ] || HMISS="$HMISS image-literals:$IMAGE(draft_sample / dspark_walk_sample_step / mtp_couple_qtemp / spechub mtp_couple_verify / W1 / VRAM levers)"
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || HMISS="$HMISS yaml:$Y(docker compose config)"
if [ "$DRY" = 1 ]; then
  echo "+ item $ITEM, image $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)), out $O, seed $SEED; NST 5 pin $FIX5; arms A $ARM_A, B $ARM_B ($ORDER), C $ARM_C (screen); chi-square S $CHI_S vs N $CHI_N, $CHIN each"
  [ -z "$HMISS" ] && echo "+ helpers: every file present and parsing, every function defined, image literals (draft_sample / dspark_walk_sample_step / mtp_couple_qtemp / spechub / W1 / VRAM levers), yaml accepted -- OK" \
                  || { echo "+ MISSING:$HMISS"; exit 1; }
else
  [ -z "$HMISS" ] || { xlog "$ITEM: VIRHE helpers missing:$HMISS"; exit 1; }
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { xlog "$ITEM: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
fi
mkd $O $O/guard $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt

# ---------------------------------------------------------------- run
xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) | lever SD (sampled drafting, the user's GO: distribution-exact, not byte-identical at T > 0) | NST 5 stack2 env + --draft-coupling on, ABBA $ORDER A=draft_sample 0 B=draft_sample 1 (+ C=70 screen) | 1u essay / code / edit at the server default (T 1, seed $SEED) thinking on + mb c4 / c8 | chi-square $CHIN x 2 | identity T 0 under B and A"
evict
if start sd-n5; then
  CHIOK=0
  chi   # the law first: the speed blocks only on a passing chi-square
  if [ $CHIOK = 1 ]; then
    k=0
    for arm in $(echo $ORDER | grep -o .); do
      eval aj=\$ARM_$arm
      if pinset sd-n5-$k$arm "$FIX5" "$aj"; then block sd-n5 $arm $k; else xlog "sd-n5 block $k $arm: VIRHE the pin did not apply -> block skipped"; fi
      k=$((k + 1))
    done
    if pinset sd-n5-${k}C "$FIX5" "$ARM_C"; then block sd-n5 C $k; else xlog "sd-n5 block $k C: VIRHE the pin did not apply -> screen skipped"; fi
  fi
  pinset sd-n5-identB "$FIX5" "$ARM_B" && ident sd-n5-iB
  pinset sd-n5-identA "$FIX5" "$ARM_A" && ident sd-n5-iA
  if [ "$DRY" != 1 ]; then
    xlog "identity (temperature 0) B vs A: copy $(cmpfile $O/copy-sd-n5-iB-run1.txt $O/copy-sd-n5-iA-run1.txt) | greedy set $(gcmp $O/sd-n5-iA-gs-c1.jsonl $O/sd-n5-iB-gs-c1.jsonl)"
    xlog "engine lines: $(docker logs $CN 2>&1 | grep -E 'coupled draft:|coupled verify completed|lever SD|fn4: switches draft_sample' | sed 's/^radiance: //; s/^[A-Z] *//' | cut -c1-220 | sort | uniq -c | sort -rn | head -8 | tr '\n' '|' | cut -c1-1500)"
  fi
  run "docker logs $CN 2>&1 | gzip -1 > $O/sd-n5-engine-full.log.gz"
  down
fi
[ "$DRY" = 1 ] || xlog "summary (mean +- t95 over blocks; 1u = server default T 1 + seed $SEED + thinking on; B vs A pooled, and the ABBA pairs' mean ratio): $(python3 -c 'import glob,json,math,os,re,statistics,sys
O=sys.argv[1]; T95={1:12.706,2:4.303,3:3.182}
def ci(v):
    v=[x for x in v if x is not None]
    if not v: return None
    m=statistics.mean(v); h=T95.get(len(v)-1,1.96)*statistics.stdev(v)/math.sqrt(len(v)) if len(v)>1 else 0.0
    return m,h
rows={}
for f in glob.glob(os.path.join(O, "sd-n5-b*-samp.txt")):
    m=re.search(r"-b(\d+)([ABC])-samp", f); k, arm = int(m.group(1)), m.group(2)
    for l in open(f):
        if l.startswith("J "):
            d=json.loads(l[2:]); rows.setdefault((arm,k),{})["1u "+d["prompt"]]=d
for f in glob.glob(os.path.join(O, "sd-n5-b*-mb.txt")):
    m=re.search(r"-b(\d+)([ABC])-mb", f); k, arm = int(m.group(1)), m.group(2)
    for l in open(f):
        if l.startswith("MB "):
            d=json.loads(l[3:]); rows.setdefault((arm,k),{})[d.get("phase")]=d
mets=[("1u essay","decode_tps","tok/s"),("1u essay","tokens_per_step","tok/step"),("1u essay","acceptance","acc"),
      ("1u code","decode_tps","tok/s"),("1u code","tokens_per_step","tok/step"),("1u code","acceptance","acc"),
      ("1u edit","decode_tps","tok/s"),("1u edit","tokens_per_step","tok/step"),("1u edit","acceptance","acc"),
      ("c4","agg_dec_tps","tok/s"),("c4","per_stream_dec_p50","p50"),("c4","tok_per_decode_step","tok/step"),("c4","acceptance","acc"),("c4","ms_decode_step","ms/step"),
      ("c8","agg_dec_tps","tok/s"),("c8","per_stream_dec_p50","p50"),("c8","tok_per_decode_step","tok/step"),("c8","acceptance","acc"),("c8","ms_decode_step","ms/step")]
ks=sorted(rows, key=lambda t: t[1])
out=[]
for ph,key,lab in mets:
    val=lambda arm: [rows[(a,k)].get(ph,{}).get(key) for (a,k) in ks if a==arm]
    a,b,c=ci(val("A")),ci(val("B")),ci(val("C"))
    if not a: continue
    pr=[]
    seq=[(a_,k) for (a_,k) in ks if a_ in "AB"]
    for i in range(0,len(seq)-1,2):
        (x,kx),(y,ky)=seq[i],seq[i+1]
        va,vb=(rows[(x,kx)].get(ph,{}).get(key),rows[(y,ky)].get(ph,{}).get(key)) if x=="A" else (rows[(y,ky)].get(ph,{}).get(key),rows[(x,kx)].get(ph,{}).get(key))
        if va and vb: pr.append(vb/va)
    s="%s %s: A %.3f+-%.3f" % (ph,lab,a[0],a[1])
    if b: s+=", B %.3f+-%.3f (%+.1f %%%s)" % (b[0],b[1],100*(b[0]/a[0]-1) if a[0] else 0, "; pairs %s" % "/".join("%+.1f" % (100*(r-1)) for r in pr) if pr else "")
    if c: s+=", C %.3f (%+.1f %%)" % (c[0],100*(c[0]/a[0]-1) if a[0] else 0)
    out.append(s)
print(" | ".join(out) or "none")' $O 2>&1 | cut -c1-3600) || identity: $(tr '\n' ',' < $O/identity.txt 2>/dev/null) || chi-square: $(grep -m1 '^CHI ' $O/chi.txt 2>/dev/null | cut -c5- | python3 -c 'import json,sys; d=json.load(sys.stdin); print("chi2 %s df %s p %s" % (d["chi2"], d["df"], d["p"]))' 2>/dev/null)"
unset $SWITCHES
xlog "$ITEM: done -- $O"
