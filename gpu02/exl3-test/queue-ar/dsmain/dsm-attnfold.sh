#!/bin/bash
# dsm-attnfold.sh (10.11.2026, queue-ar/dsmain, ds41-k1 agent, lever K1) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-n-attnfold.sh. Lever K1's exact launch folds on DeepSeek-V4.1's attention path, A/B'd IN ONE PROCESS on one
# start of the ds41-k1 image (XK1_IMAGE) by three runtime switches (POST /admin/fn4; 0 = today's ops, 1 = the fold):
#   ds41_kvfold   (fold c) the window key's rmsnorm_rope_roundtrip + kv_store -> rmsnorm_rope_roundtrip_kv_store: the norm,
#                 the rope, the fp8 round trip and the paged-cache store in one launch (every step, decode and prefill)
#   ds41_ofold    (fold d) attn_sparse_sink + rope_rows_roundtrip (the output's inverse rope + fp8 round trip) ->
#                 attn_sparse_sink_rope_roundtrip: up to 48 rows the decode kernel with the rope + round trip in its epilogue
#                 (one launch); taller passes keep the two launches
#   ds41_onefill  (fold f) the ungated shared expert's gate plane of ones: one fill a step instead of one a MoE layer
# RADIANCE_DS41_KVFOLD / _OFOLD / _ONEFILL = 1 declare the folds beside today's ops and start with them on (the serve log's
# "lever K1 folds" note counts the layers).
#   Why it can pay: a decode layer is ~23-25 dependent launches, each ~3.8 us of wait before it; each fold removes one a layer
#   (43 layers): c ~0.45 ms, d ~0.55 ms, f ~0.2 ms a 1-user step profiled (LEVERS K1). Unlike ROUTER_FUSE / MIX_FUSE (77,
#   two medium kernels merged, lost to the fused kernel's own cost), each puts a small elementwise op into a neighbour's
#   epilogue, so the big kernel's cost should barely move -- the blocks measure that.
#   EXACTNESS (by construction; proven again by the selftest phase on the card before the server starts): each fold forms
#   the very values of the ops it replaces, in the same order with the same roundings (r4d_ds41_qrt.h: one definition of
#   the rope / round-trip rules for both forms; no FMA contraction), and writes the same bytes -- c: y and the cache slot
#   get the same bf16; d: the block amax is a max of NaN-free non-negative floats (order-free); f: the gathers read the same
#   ones. The bytes are the same at every switch value.
# DeepSeek-V4.1 Flash EXL3 2.25, TP4, MML 524288, NST 0, the t3 env and serve line of dsm-serve.sh (105's t3 arm + lib-29
# best_vars + MNBT 4160 + --live), RADIANCE_DS41_ARGATHER=on (today's 4-row bound, pinned 4), F4 off (SHEXP_LANE unset) +
# RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1, every other switch unset.
# ORDER (decode before prefill, TESTING 9.11.):
#   SELFTEST (one card, no server): the image's r4d_selftest, cases rmsnorm_rope_roundtrip_kv_store_fp8_bf16 (M 1..48, 64,
#     130, 512) and ds41_attn_sparse_sink_rope_rt_d512_bf16 (M 1..48, 64, 130; ratio 0 / 4; bf16 / nvfp4 latents), each
#     byte for byte against the rows it folds, + their back-to-back kernel times (info). A FAIL stops the item (VIRHE).
#   -> gate -> modes (t3 notes + the K1 note: every fold declared at > 0 layers) -> switch note (GET at start: the three 1,
#   ds41_argather 4, gg_gemv 0, aql_acq 0, epoch 0; ds41_kvfold 2 refused; all three 0 then 1 applied at a step) -> warm-up
#   answers vs stock -> slow-state probe
#   -> BLOCKS (ggblocks_k1.py, XK1_ORDER default A B B A A B, mb XK1_DUR 75 s; stepprobe essay/code 512 = c1, mb c4, mb c8;
#     gg_gemv / aql_acq 0, ds41_argather 4 pinned; XK1_SETS default "all kv o one"):
#       all  the three switches together, 0 vs 1 (the headline: the 1-user step, est. -1.2 ms profiled)
#       kv   ds41_kvfold 0 / 1 (ofold, onefill 0)      o    ds41_ofold 0 / 1 (kvfold, onefill 0)
#       one  ds41_onefill 0 / 1 (kvfold, ofold 0)
#     each: B vs A % per load point (c1 tok/s + ms a step, c4 / c8 aggregate tok/s, per-stream p50, ms a step), pooled,
#     pairs, t95.
#   -> C4 FLOOR (coordinator's diagnosis, ~6 min): at K1 0 the c4 greedy set twice the normal way and twice with every prompt
#     prefilled first (c4diag_k1.py), then prefilled-first once at K1 1 (info): run 1 vs run 2 per mode and a label (mixed
#     prefill + decode steps named the cause only if the prefilled-first pair is identical, the normal pair is not, and the
#     server kept the prompts' prefix in both prefilled runs; else a hypothesis)
#   -> IDENTITY (prefill-heavy, last), under B (all three 1) then A (all 0): the 28k copy after a ~31k cold prompt, the
#     greedy set at c1, a fresh 16k prefill (TTFT, info) and the 136k deep answer (135k prompt + 512 greedy tokens); each vs
#     today's files (runs/xdsm/copy-dsm-225-i1-run1.txt, runs/xdsm/draft/dsd-d0-gs-c1.jsonl, runs/xdsm/deep-dsm-225.txt) and
#     B vs A, byte for byte. Folds c and f run in prefill too (c at every height, f once a step), so the prefill pair is in.
# References (no re-runs): bz's dsd-d0 (main image, NST 0): probe 59.38 / 62.04 tok/s (16.84 / 16.12 ms), c4 122.7 (p50
#   31.6), c8 150.1 (p50 19.7). The A arm of every block set is today's path in this process.
# The image (XK1_IMAGE, build/k1/assemble.sh): ds41-k1 = ds41-d3 c03c57d + ds41-gg 8b311fc + lever K1, layered on
#   radiance-local:ds41-d3-c03c57d9-5c13e555-l (bin/radiance, bin/r4d_selftest, libr4d.so, deepseek41.so).
# NO cc_sim (user 10.11.), no 2.9. No power / host / fan settings; production compose files read only (--project-directory).
# DRYRUN=1 bash dsm-attnfold.sh : prints every command, runs and writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-k1-tp4.yaml
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=$R/xdsm/attnfold; export STATSDIR=$O/stats
RD=$R/xdsm/draft                                # bz's dsd-d0 (main image, NST 0) files
RM=$R/xdsm                                      # item a's dsm-225 (main image, NST 0) files
export IMAGE=${XK1_IMAGE:-radiance-local:ds41-k1-1d1f782b-5c13e555-l}
ITEM=${XK1_ITEM:-57-p4zyzzzzz-ds-n-attnfold}; ORDER=${XK1_ORDER:-ABBAAB}; DUR=${XK1_DUR:-75}; SETS=${XK1_SETS:-all kv o one}
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
GGB=$HD/ggblocks_k1.py
C4D=$HD/c4diag_k1.py
DQ=$QA/deepq.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
K1SW="ds41_kvfold ds41_ofold ds41_onefill"
PIN='"ds41_argather": 4, "gg_gemv": 0, "aql_acq": 0'
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DS41_ARGATHER_MAX RADIANCE_DS41_SHEXP_LANE RADIANCE_DEBUG_MISS_LOG
  RADIANCE_DS41_KVFOLD RADIANCE_DS41_OFOLD RADIANCE_DS41_ONEFILL"
k1env() {   # 105's t3 arm + the three K1 folds declared (and on at start), every other switch unset
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0 \
         RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
}
modelvars225() {   # -> MODEL, HP29, LIM, WARMREF, C8REF (105's c8 ms a step); lib-29 best_vars (NST 0) + MNBT 4160 + --live
  export MODEL=$M225; HP29=19456; LIM=23; WARMREF=$WARM_REF_225; C8REF=43.1
  best_vars; export MNBT=4160 XARGS="$XARGS --live"
}
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
gate() {   # gate TAG LIMIT: lib-29's gate + keeplog before the retry / after a second slow check
  local tag=$1 lim=$2 ms ms1 st=fast
  restart $tag || { keeplog $tag-fail; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && return 0
  if awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}'; then
    keeplog $tag-try1
    ms1=$ms; restart $tag || { keeplog $tag-fail; return 1; }; ms=$(check $tag 2); st="fast-after-retry (first $ms1)"
    awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}' && { st="SLOW-after-retry (first $ms1)"; keeplog $tag-try2; }
  fi
  echo "$st $ms" > $O/guard-$tag.txt
  log "RESULT guard $tag: $st, 1-user ${ms} ms/step (limit $lim; bz dsd-d0 main image 15.59)"
  case $st in SLOW*) return 2 ;; esac
}
gpurun() {   # gpurun NAME SECONDS docker-run-args...: bounded, named, removed even if its kernel hangs (86a's)
  local name=$1 t=$2; shift 2
  timeout -k 15 $t docker run --rm --init --stop-timeout 10 --name $name "$@"; local rc=$?
  docker rm -f $name > /dev/null 2>&1 || true
  { [ $rc = 124 ] || [ $rc = 137 ]; } && echo "gpurun: $name stopped after $t s (rc $rc)" >&2
  return $rc
}
FOLLOW_PID=""
follow() { [ "$DRY" = 1 ] && { echo "+ docker logs -f $CN > $1/engine.log &"; return; }
  docker logs -f $CN > $1/engine.log 2>&1 & FOLLOW_PID=$!; sleep 2; }
unfollow() { [ -n "$FOLLOW_PID" ] && { kill $FOLLOW_PID 2>/dev/null; wait $FOLLOW_PID 2>/dev/null; FOLLOW_PID=""; }; return 0; }
trap 'unfollow; down' EXIT
trap 'exit 143' TERM INT
cmpfile() {   # cmpfile A B -> IDENTICAL (n chars) | DIFFERENT at char k (na vs nb chars) | not compared
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xk1 $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xk1 $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  case $r in IDENTICAL*) log "RESULT xk1 $1 warm-up answers vs stock ($(basename $2)): $r" ;;
    *) log "RESULT xk1 $1 warm-up answers vs stock ($(basename $2)): VIRHE $r -- NOT EXACT" ;; esac
}
modes() {   # modes TAG: the serve log's notes must be the t3 line's + the K1 note (every fold declared somewhere) -> 0, else 1
  [ "$DRY" = 1 ] && { echo "+ modes $1: indexer note (mxfp4, latent bf16, compressor tail bound, router selection+sort separate, AR_PCN separate at 0 layers), ARGATHER note up to 4 (no ARGATHER_MAX), no shared-expert-lane note, the lever K1 note (kvfold / ofold at > 0 layers, ones declared at > 0 MoE layers, all three switches 1), staging line, no acquire-rule line"; return 0; }
  local MLF SL AL XL KL ok=1
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  AL=$(docker logs $CN 2>&1 | grep -m1 'the MoE gather and its all-reduce as one kernel' | sed 's/^radiance: //' | cut -c1-300)
  XL=$(docker logs $CN 2>&1 | grep -m1 'the shared expert on the second lane' | sed 's/^radiance: //' | cut -c1-300)
  KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-500)
  log "RESULT xk1 $1 modes: $(echo "$MLF" | cut -c1-900) | staging: ${SL:-none} | argather: ${AL:-none} | K1: ${KL:-none}"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate" "post-collapse-norm separate at 0 layers" "next collapse-norm separate at 0 layers"; do
    case "$MLF" in *"$w"*) ;; *) log "RESULT xk1 $1: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  case "$AL" in *"up to 4 tokens a step (RADIANCE_DS41_ARGATHER)"*) ;; *) log "RESULT xk1 $1: VIRHE the ARGATHER note is not today's 4-row one: '${AL}'"; ok=0 ;; esac
  [ -z "$XL" ] || { log "RESULT xk1 $1: VIRHE a shared-expert-lane note (RADIANCE_DS41_SHEXP_LANE should be unset): '$XL'"; ok=0; }
  case "$KL" in
    *"in one launch at 0 of"*|*"rope+round trip at 0 ("*|*"not declared at"*|*"at 0 MoE layers"*|"")
      log "RESULT xk1 $1: VIRHE the lever K1 note does not show all three folds declared: '${KL}'"; ok=0 ;;
  esac
  case "$KL" in *"ds41_kvfold=1"*"ds41_ofold=1"*"ds41_onefill=1"*) ;; *) log "RESULT xk1 $1: VIRHE the K1 switches do not start at 1: '${KL}'"; ok=0 ;; esac
  [ -n "$SL" ] || { log "RESULT xk1 $1: VIRHE no prefill staging line (levels refused?)"; ok=0; }
  docker logs $CN 2>&1 | grep -q 'the acquire rule plays' && { log "RESULT xk1 $1: VIRHE an acquire-rule line (RADIANCE_AQL_ACQ should be unset)"; ok=0; }
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
fn4code() {   # fn4code JSON: POST, print the HTTP status
  [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null
}
allk1() { echo "{\"ds41_kvfold\": $1, \"ds41_ofold\": $1, \"ds41_onefill\": $1, $PIN, \"epoch_bump\": 1}"; }
swnote() {   # swnote TAG: the switches at start, ds41_kvfold 2 refused, all three 0 then 1 applied at a step
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (expect ds41_kvfold / ds41_ofold / ds41_onefill 1, ds41_argather 4, gg_gemv 0, aql_acq 0, epoch 0); POST ds41_kvfold 2 -> 400; POST all three 0 / 1, each + a 4-token request -> applied"; return 0; }
  local s0 s1 s2 c2
  s0=$(fn4get)
  c2=$(fn4code '{"ds41_kvfold": 2}')
  fn4set "$(allk1 0)"; nudge; s1=$(fn4get)
  fn4set "$(allk1 1)"; nudge; s2=$(fn4get)
  log "RESULT xk1 $1 switch note: at start $s0 | ds41_kvfold 2 -> HTTP $c2 | after all 0: $s1 | after all 1: $s2 | core log: $(docker logs $CN 2>&1 | grep -E 'fn4: switches' | tail -2 | sed 's/^[A-Z] *//' | tr '\n' '|' | cut -c1-400)"
  python3 -c 'import json,sys
s0, s1, s2 = (json.loads(x) for x in sys.argv[1:4])
w = lambda s, k: s["switches"].get(k)
k1 = ("ds41_kvfold", "ds41_ofold", "ds41_onefill")
ok = all(w(s0, k) == 1 for k in k1) and w(s0, "ds41_argather") == 4 and w(s0, "gg_gemv") == 0 and w(s0, "aql_acq") == 0 \
     and s0["epoch"] == 0 and all(w(s1, k) == 0 for k in k1) and all(w(s2, k) == 1 for k in k1) and not s1["pending"] \
     and not s2["pending"] and s2["epoch"] == 2
sys.exit(0 if ok else 1)' "$s0" "$s1" "$s2" 2>/dev/null && [ "$c2" = 400 ]
}
mbms() { grep '^MB ' $1 2>/dev/null | head -1 | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()[3:]).get("ms_decode_step", ""))' 2>/dev/null; }
stprobe() {   # stprobe TAG N: mb c8 45 s -> "fast|between|slow MS (ref REF)"
  [ "$DRY" = 1 ] && { echo "+ (cd $D && timeout 400 python3 mb.py --tag xk1-$1-st$2 --phases c8 --dur ${XK1_STATE_DUR:-45} --warm 10 > $O/$1-stprobe$2.txt)" >&2; echo "fast (dry)"; return 0; }
  run "(cd $D && timeout 400 python3 mb.py --tag xk1-$1-st$2 --phases c8 --dur ${XK1_STATE_DUR:-45} --warm 10 > $O/$1-stprobe$2.txt 2>&1)"
  local ms; ms=$(mbms $O/$1-stprobe$2.txt)
  [ -n "$ms" ] || { echo "unknown (no mb line)"; return 0; }
  awk -v m=$ms -v r=$C8REF 'BEGIN{s=(m <= 1.07*r) ? "fast" : (m >= 1.14*r) ? "slow" : "between"; printf "%s %s ms (105 tuned3l c8 %s ms, x%.3f)\n", s, m, r, m/r}'
}
therm() { type thermwin > /dev/null 2>&1 && echo " | $(thermwin $1 $(date +%s) 2>/dev/null | cut -c1-200)"; }
copyt() {   # copyt TAG: the 28k copy after a ~31k-token prompt (105's prompt, offset 0 -- item a's dsm-225 i1, bz d0's i1)
  local t0; t0=$(date +%s)
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset 0 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xk1 $1 copy: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300)$(therm $t0)"
}
gsetc1() { run "(cd $D && timeout 900 python3 $GS $URL $O/$1-gs-c1.jsonl --conc 1 --max-tokens 192 > $O/$1-gs-c1.txt 2>&1)"; }
gsetc4() { run "(cd $D && timeout 900 python3 $GS $URL $O/$1-gs-c4.jsonl --conc 4 --max-tokens 192 > $O/$1-gs-c4.txt 2>&1)"; }
gsetpf() {   # gsetpf TAG: the c4 set with every prompt prefilled first (c4diag_k1.py) + its cache line
  run "(cd $D && timeout 1200 python3 -I $C4D $URL $O/$1-gs-pf4.jsonl --gs $GS --conc 4 --max-tokens 192 > $O/$1-gs-pf4.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xk1 c4floor $1 prefilled-first run: $(grep -m1 '^C4DIAG ' $O/$1-gs-pf4.txt | cut -c8- | cut -c1-300)"
}
pfd() {   # pfd TAG DEPTH: one fresh prefill of DEPTH tokens (nonce first, cache_prompt off; span4's pfd)
  local t0; t0=$(date +%s)
  run "(cd $D && timeout 1800 python3 ds41_tests.py prefill --depths $2 --cpt 3.95 --calibrate 0 --reps 1 > $O/$1-pf$2.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xk1 $1 fresh prefill $2: $(grep '^J ' $O/$1-pf$2.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); print("%s tok, cached %s, TTFT %s s, %s tok/s, %s" % (d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("prefill_tps"), d.get("status")))' 2>&1 | cut -c1-200)$(therm $t0)"
}
deepq() {   # deepq TAG: the fixed ~135k prompt and a 512-token greedy answer (105's deepq.py)
  local t0; t0=$(date +%s)
  run "(cd $D && timeout 1800 python3 $DQ $O/deep-$1.txt 135000 512 > $O/deep-$1.out 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xk1 $1 deep probe: $(grep -m1 '^DEEPQ ' $O/deep-$1.out | cut -c7- | cut -c1-200)$(therm $t0)"
}
gcmp() { [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }; [ -s "$1" ] && [ -s "$2" ] || { echo "not compared (missing $1 or $2)"; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
idline() {   # idline LABEL RESULT: IDENTICAL -> plain, else VIRHE (the folds are exact by construction)
  case "$2" in IDENTICAL*) log "RESULT xk1 identity $1: $2" ;; "not compared"*) log "RESULT xk1 identity $1: $2" ;;
    *) log "RESULT xk1 identity $1: VIRHE $2 -- the K1 folds are exact by construction, so this is a defect (or a reference from another line)" ;; esac
}
blocks() {   # blocks TAG SWITCHES A B FIXED: the A/B blocks over SWITCHES (comma list; FIXED pinned in every block) + summary
  local tag=$1 sw=$2 a=$3 b=$4 fx=$5
  run "timeout 7200 python3 -I $GGB --url $URL --mb-dir $D --label $tag --switch $sw --a $a --b $b --fixed '$fx' --order $ORDER --dur $DUR --warm 10 --out $O/$tag-blocks.jsonl > $O/$tag-blocks.txt 2>&1"
  [ "$DRY" = 1 ] && return 0
  while IFS= read -r l; do log "RESULT xk1 $tag ${l:0:400}"; done < <(grep '^BLOCK ' $O/$tag-blocks.txt)
  log "RESULT xk1 $tag summary (B = $sw $b vs A = $a, $fx pinned, in process, order $ORDER, mb $DUR s): $(grep '^GGB ' $O/$tag-blocks.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=2: "-" if v is None else format(v, ".%df" % n)
out = []
for k, v in d["metrics"].items():
    p = v.get("pooled")
    out.append("%s %s -> %s (%s %%; pairs %s; t95 +- %s %%)" % (k, f(v.get("A")), f(v.get("B")), "-" if p is None else "%+.2f" % (100 * (p - 1)),
               " ".join("%+.2f" % (100 * (x - 1)) for x in v.get("pairs", [])), "-" if v.get("ci95") is None else "%.2f" % (100 * v["ci95"])))
print("; ".join(out) + " | %d blocks, %d flagged, %s s" % (d["blocks"], d["blocks_bad"], d["wall_s"]))' 2>&1 | cut -c1-2400)"
}
selftest() {   # the image's r4d_selftest on card 0, before any server: each fold against the rows it folds, byte for byte
  local DEV="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ulimit memlock=-1 --ipc=host -e HIP_VISIBLE_DEVICES=0"
  local c rc nf nok bad=0 r
  if [ "$DRY" != 1 ]; then
    for w in $(seq 1 30); do [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] && break; sleep 2; done
    [ -z "$(ls -A /sys/class/kfd/kfd/proc 2>/dev/null)" ] || { log "RESULT xk1 selftest: VIRHE a KFD client is running (after 60 s wait)"; return 1; }
  fi
  for c in rmsnorm_rope_roundtrip_kv_store_fp8_bf16 ds41_attn_sparse_sink_rope_rt_d512_bf16; do
    run "gpurun xk1st-$$-${c:0:12} 900 $DEV --entrypoint /opt/radiance/bin/r4d_selftest $IMAGE /opt/radiance/share/radiance/kernels/libr4d.so --case $c > $O/selftest-$c.txt 2>&1"
    rc=$?
    [ "$DRY" = 1 ] && continue
    nf=$(grep -c 'FAIL' $O/selftest-$c.txt); nok=$(grep -cE '^ *ok ' $O/selftest-$c.txt)
    { [ $rc = 0 ] && [ $nf = 0 ] && [ $nok -gt 0 ]; } || bad=$((bad + 1))
    r="$c: rc $rc, $nok ok, $nf FAIL$({ [ $rc = 124 ] || [ $rc = 137 ]; } && echo ' -- STOPPED by the limit (hang)')"
    log "RESULT xk1 selftest $r | $(grep -m1 -E '^ *(ok|FAIL) ' $O/selftest-$c.txt | sed 's/^ *//' | cut -c1-600)"
    while IFS= read -r l; do log "RESULT xk1 selftest info $l"; done < <(grep -E '^ *info ' $O/selftest-$c.txt | sed 's/^ *info *//' | cut -c1-200)
  done
  [ "$DRY" = 1 ] && return 0
  [ $bad = 0 ] || { log "RESULT xk1 selftest: VIRHE $bad case(s) not clean -- an exactness defect, the engine phases do not run"; return 1; }
  log "RESULT xk1 selftest: CLEAN (both folds byte-identical to the rows they fold)"
}
# ---------------------------------------------------------------- main
log "RESULT xk1 $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) A all K1 switches 0 / B 1 (ds41_argather 4), sets [$SETS], order $ORDER mb $DUR s yaml $Y | $(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -m1 "^radiance" /opt/radiance/share/radiance/k1-PROVENANCE.txt' 2>/dev/null | cut -c1-200) | refs bz dsd-d0 (main image, NST 0): probe 59.38 / 62.04 tok/s, c4 122.7 (p50 31.6), c8 150.1 (p50 19.7)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xk1: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xk1: VIRHE image $IMAGE not on gpu02"; exit 1; }
  [ -e $Y ] && grep -q '^ *RADIANCE_DS41_KVFOLD:' $Y && grep -q '^ *RADIANCE_DS41_OFOLD:' $Y && grep -q '^ *RADIANCE_DS41_ONEFILL:' $Y || { log "RESULT xk1: VIRHE $Y missing or does not pass RADIANCE_DS41_KVFOLD / _OFOLD / _ONEFILL"; exit 1; }
  for f in $GS $GGB $DQ $D/stepprobe.py $D/mb.py $D/ds41_tests.py; do [ -e $f ] || { log "RESULT xk1: VIRHE $f missing"; exit 1; }; done
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xk1: VIRHE $M225 missing or not verified"; exit 1; }
  [ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -q RADIANCE_DS41_KVFOLD /opt/radiance/share/radiance/architectures/deepseek41.so && grep -q ds41_onefill /opt/radiance/bin/radiance && grep -q rmsnorm_rope_roundtrip_kv_store /opt/radiance/share/radiance/kernels/libr4d.so && grep -q ds41_attn_sparse_sink_rope_rt_d512_bf16 /opt/radiance/bin/r4d_selftest && echo yes' 2>/dev/null)" = yes ] \
    || { log "RESULT xk1: VIRHE $IMAGE is not the K1 image (no RADIANCE_DS41_KVFOLD in deepseek41.so, ds41_onefill in radiance, the fused rows in libr4d or their selftest cases)"; exit 1; }
  for f in $RM/copy-dsm-225-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $RM/deep-dsm-225.txt; do [ -s $f ] || log "RESULT xk1: note -- reference $f missing (that identity line reads 'not compared')"; done
fi
mkd $O $O/guard $STATSDIR
down
selftest || exit 1
evict
TAG=k1; P=$O/k1
k1env; modelvars225
log "RESULT xk1 $TAG: start IMAGE=$IMAGE MODEL=$MODEL NST=$NST $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
mkd $P
gate $TAG $LIM; rc=$?
[ $rc = 1 ] && { log "RESULT xk1 $TAG: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$TAG.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; exit 1; }
[ $rc = 2 ] && { log "RESULT xk1 $TAG: VIRHE slow state after one restart -> stopped"; exit 1; }
modes $TAG || { log "RESULT xk1 $TAG: VIRHE the modes are not as asked -> stopped"; exit 1; }
swnote $TAG || { log "RESULT xk1 $TAG: VIRHE the switches are not as expected (start all three 1 / argather 4 / epoch 0, 2 refused, 0 / 1 applied) -> stopped"; keeplog $TAG-switch; exit 1; }
identw $TAG $WARMREF
st=$(stprobe $TAG 1)
if [ "${st%% *}" = slow ]; then
  log "RESULT xk1 $TAG state probe 1: $st -> one restart"
  gate $TAG $LIM || { log "RESULT xk1 $TAG: VIRHE no healthy / fast start after the slow-state restart"; exit 1; }
  modes $TAG || exit 1
  swnote $TAG || { log "RESULT xk1 $TAG: VIRHE the switches are not as expected after the restart -> stopped"; exit 1; }
  st=$(stprobe $TAG 2)
fi
log "RESULT xk1 $TAG state: $st (slow-state probe mb c8 ${XK1_STATE_DUR:-45} s)$(case "${st%% *}" in fast) ;; *) echo " -- NOT FAST: flag to the coordinator" ;; esac)"
follow $P
for s in $SETS; do
  stopfile && break
  case $s in
    all) blocks $TAG-all ds41_kvfold,ds41_ofold,ds41_onefill 0 1 "{$PIN}" ;;
    kv)  blocks $TAG-kv ds41_kvfold 0 1 "{\"ds41_ofold\": 0, \"ds41_onefill\": 0, $PIN}" ;;
    o)   blocks $TAG-o ds41_ofold 0 1 "{\"ds41_kvfold\": 0, \"ds41_onefill\": 0, $PIN}" ;;
    one) blocks $TAG-one ds41_onefill 0 1 "{\"ds41_kvfold\": 0, \"ds41_ofold\": 0, $PIN}" ;;
    *) log "RESULT xk1: note -- unknown block set '$s' skipped" ;;
  esac
done
# ---- C4 FLOOR (coordinator 10.11.; diagnosis, no extra start): today the c4 greedy set (greedyset.py --conc 4) differs from
# itself run to run on 6 of 20 answers. Hypothesis (unverified): mixed prefill + decode steps (a decode row's dense / EXL3 /
# attention forms change when it shares a step with prompt rows). At the K1 switches 0 (today's path): the c4 set twice the
# normal way, then twice with every prompt prefilled first (c4diag_k1.py: each prompt alone with max_tokens 1, then the same
# 20 requests four in flight; the server's cached counts show whether the prefix was kept, i.e. whether the continuation
# steps were decode rows only); then the prefilled-first set once under the folds (B), as information. Label: mixed steps
# are the cause only if the prefilled-first pair is IDENTICAL, the normal pair is not, and the prefix was kept in both
# prefilled runs; otherwise it stays a hypothesis. XK1_C4FLOOR=0 skips the phase.
c4floor() {
  local fa fb na nb pk lab
  fn4set "$(allk1 0)"; nudge
  gsetc4 $TAG-c4n1; gsetc4 $TAG-c4n2
  gsetpf $TAG-c4p1; gsetpf $TAG-c4p2
  fn4set "$(allk1 1)"; nudge
  gsetpf $TAG-c4pB
  fn4set "$(allk1 0)"; nudge
  [ "$DRY" = 1 ] && { echo "+ c4floor: normal run 1 vs 2, prefilled-first run 1 vs 2, prefilled B vs A run 1 -> label"; return 0; }
  na=$(gcmp $O/$TAG-c4n1-gs-c4.jsonl $O/$TAG-c4n2-gs-c4.jsonl)
  nb=$(gcmp $O/$TAG-c4p1-gs-pf4.jsonl $O/$TAG-c4p2-gs-pf4.jsonl)
  fa=$(gcmp $O/$TAG-c4n1-gs-c4.jsonl $O/$TAG-c4p1-gs-pf4.jsonl)
  fb=$(gcmp $O/$TAG-c4p1-gs-pf4.jsonl $O/$TAG-c4pB-gs-pf4.jsonl)
  pk=$(for t in c4p1 c4p2; do grep -m1 '^C4DIAG ' $O/$TAG-$t-gs-pf4.txt | cut -c8- | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()).get("prefix_kept"))' 2>/dev/null || echo None; done | tr '\n' ' ')
  log "RESULT xk1 c4floor normal c4 set, run 1 vs run 2 (K1 0 = today's path): $na"
  log "RESULT xk1 c4floor prefilled-first c4 set, run 1 vs run 2 (K1 0): $nb | prefix kept in run 1 / run 2: $pk"
  log "RESULT xk1 c4floor normal run 1 vs prefilled-first run 1 (K1 0, info): $fa"
  log "RESULT xk1 c4floor prefilled-first: folds on (B) vs today (A) run 1 (info): $fb"
  case "$nb|$na|$pk" in
    IDENTICAL*\|DIFFERENT*\|"True True ") lab="CAUSE: mixed prefill + decode steps (the prefilled-first pair is identical, the normal pair is not, the prefix was kept in both)" ;;
    IDENTICAL*\|DIFFERENT*\|*) lab="HYPOTHESIS (not shown): the prefilled-first pair is identical and the normal pair is not, but the prefix was not kept in both prefilled runs ($pk), so their continuation steps were not decode-only" ;;
    IDENTICAL*\|IDENTICAL*\|*) lab="HYPOTHESIS (not shown): both pairs are identical in this run -- the normal c4 floor did not show here" ;;
    *) lab="HYPOTHESIS (not shown): the prefilled-first pair differs too ($nb), so mixed steps are not the whole cause" ;;
  esac
  log "RESULT xk1 c4floor label: $lab"
}
[ "${XK1_C4FLOOR:-1}" = 1 ] && ! stopfile && c4floor
# ---- identity (the prefill-heavy part last): B (all three folds) on the cold prompts, then A (today's ops)
if ! stopfile; then
  for nm in B A; do
    [ $nm = B ] && fn4set "$(allk1 1)" || fn4set "$(allk1 0)"
    nudge
    copyt $TAG-$nm; gsetc1 $TAG-$nm; pfd $TAG-$nm 16384; deepq $TAG-$nm
    [ "$DRY" = 1 ] || log "RESULT xk1 $TAG identity $nm ran under: $(fn4get | cut -c1-300)"
  done
  if [ "$DRY" = 1 ]; then echo "+ identity: copy / greedy set c1 / deep 136k answer, B and A each vs today's files (dsm-225 copy + deep, dsd-d0 greedy set) and B vs A; fresh 16k prefill TTFT B vs A (info)"
  else
    idline "28k copy B (all folds, cold prompt) vs today's (item a dsm-225 i1)" "$(cmpfile $O/copy-$TAG-B-run1.txt $RM/copy-dsm-225-i1-run1.txt)"
    idline "28k copy A (today's ops) vs today's (dsm-225 i1)" "$(cmpfile $O/copy-$TAG-A-run1.txt $RM/copy-dsm-225-i1-run1.txt)"
    idline "28k copy B vs A (in process)" "$(cmpfile $O/copy-$TAG-B-run1.txt $O/copy-$TAG-A-run1.txt)"
    idline "greedy set c1 B vs bz dsd-d0" "$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$TAG-B-gs-c1.jsonl)"
    idline "greedy set c1 A vs bz dsd-d0" "$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$TAG-A-gs-c1.jsonl)"
    idline "greedy set c1 B vs A (in process)" "$(gcmp $O/$TAG-A-gs-c1.jsonl $O/$TAG-B-gs-c1.jsonl)"
    idline "136k deep answer B vs today's (item a dsm-225)" "$(cmpfile $O/deep-$TAG-B.txt $RM/deep-dsm-225.txt)"
    idline "136k deep answer A vs today's (dsm-225)" "$(cmpfile $O/deep-$TAG-A.txt $RM/deep-dsm-225.txt)"
    idline "136k deep answer B vs A (in process)" "$(cmpfile $O/deep-$TAG-B.txt $O/deep-$TAG-A.txt)"
    log "RESULT xk1 fresh 16k prefill B vs A (info, one each): $(python3 -c 'import json,sys
def rd(f):
    try: return next(json.loads(l[2:]) for l in open(f) if l.startswith("J "))
    except (OSError, StopIteration, ValueError): return {}
b, a = rd(sys.argv[1]), rd(sys.argv[2])
f = lambda d, k: d.get(k)
r = (f(b, "ttft") / f(a, "ttft")) if f(a, "ttft") and f(b, "ttft") else None
print("TTFT B %s s / A %s s (%s), prefill B %s / A %s tok/s" % (f(b, "ttft"), f(a, "ttft"), "-" if r is None else "%+.2f %%" % (100 * (r - 1)), f(b, "prefill_tps"), f(a, "prefill_tps")))' $O/$TAG-B-pf16384.txt $O/$TAG-A-pf16384.txt 2>&1 | cut -c1-300)"
  fi
fi
unfollow
run "docker logs $CN > $O/$TAG-engine-full.log 2>&1"
[ "$DRY" = 1 ] || log "RESULT xk1 $TAG engine log: $(grep -cE '^E ' $O/$TAG-engine-full.log) E lines, $(grep -c 'fn4: switches' $O/$TAG-engine-full.log) switch applications"
down
unset $SWITCHES
log "RESULT xk1 $ITEM: done -- $O"
