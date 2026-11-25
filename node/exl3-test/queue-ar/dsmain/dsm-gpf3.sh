#!/bin/bash
# dsm-gpf3.sh (10.11.2026, queue-ar/dsmain, lever GPF agent, branch ds41-gpf) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-<x>-gpfab3.sh = dsm-gpf2.sh (gpfab2's body) with
#  - gpfnote's start check fixed: libr4d parses RADIANCE_DS41_GG_PF_TABLE lazily (pf_table(), a function-local static in
#    r4d_ds41_ggml_pf.hip) and prints its "lever GPF form table ... N entries" line on the first gg_pf rule call -- a prefill row
#    with gg_pf on -- so at start (gg_pf 0) there is no line: gpfab / gpfab2 stopped on that (false VIRHE). Now the start check
#    reads the engine's env listing (RADIANCE_DS41_GG_PF_TABLE / _F32_PF named), records the table line if present, and
#    tblcheck runs after the first B blocks that prefilled: N > 0 entries and no skipped entry, else VIRHE (logged, the item goes on);
#  - an arm C (f32_pf alone, XGQ_ARMC, default on when both levers switch): one A C C A round with the same c8 + fresh prefills
#    after the A / B rounds, so gg's and f32's shares separate (~9 min);
#  - out runs/xdsm/gpfab3 (XGQ_OUT), tag prefix gq3.
# dsm-gpf2.sh's header, unchanged: Helpers copied from dsm-small.sh (stackenv, gate, modes, identw,
# copyt / gsetc1 / ident, deep, pcreset; tag xgq); the A/B blocks by queue-ar/tools/abblocks2.py (read only); yaml
# ds41-gpf-tp4.yaml (= ds41-small-tp4.yaml with env_file ${XGQ_ENVFILE}).
#
# WHAT: levers GPF and GPF-b in process -- gemm_nt_ggml's prefill rows (above the decode band) in the forms of
#   r4d_ds41_ggml_pf.hip (the stock WMMA kernel's per-output arithmetic in other tile geometries; fn4 switch gg_pf) and
#   gemm_nt_f32's prefill rows (router, compressor kv | gate) in r4d_ds41_f32pf.hip's tiled form (the stock 32 lane-strided
#   fmaf chains + xor butterfly; fn4 switch f32_pf): arm A gg_pf 0 + f32_pf 0 (today), arm B gg_pf 1 (the per-shape table
#   RADIANCE_DS41_GG_PF_TABLE, declared at start; a shape it does not name stays stock) + f32_pf <the bench's form> --
#   each only where the bench found a win (a lever without one stays 0 in both arms).
#   GPF-b's form: gpsum.txt's "GPF32FORM ab: <code>" (the least summed us over router N 96 / 384 and compressor N 1024, both
#   weight storages, M 4096 / 2048, >= 2 % under the stock, slower nowhere by > 0.5 %), or XGQ_F32.
#   THE TABLE comes from the one-card bench item xgp: gpsum.txt's "GPFORM table-4096+2048" (per (type, shape) the byte-identical
#   form with the least us at M 4096 + 2048 that beats the stock's sum by >= 1 % and loses at neither M by > 0.5 %), else
#   gpperf.txt's GPTAB (best at M 4096), or XGQ_TABLE. '-' / missing = no form won -> nothing runs.
# ONE START (NST 0, the stack: 105's t3 env + gg_gemv 2 + aql_acq 2 (rule-w) + K1 c / d / f declared and on, turnover 0) with the
#   table declared and gg_pf 0 at start; then
#   (1) abblocks2.py --phase c8 --pf-depths 16384,65536,131072, XGQ_ROUNDS (2) rounds of A B B A: a block = POST the arm + epoch_bump,
#       an mb c8 window (decode: gg_pf must not change it), then FRESH prefills (nonce first, so nothing of the prefix cache is
#       reused) of ~16k / ~64k / ~128k tokens -> per round (B1 + B2) / (A1 + A2) of prefill tok/s per depth, mean +- t95;
#   (2) abblocks2.py --phase c4, one ABBA round (decode c4: no change expected);
#   (3) abblocks2.py --phase pf, one ABBA round (mb pf: 4 decode streams + a fresh ~32k prefill beside them);
#   (4) the 136k COLD prefill (deepq.py, a prefix-cache reset first, 512 greedy tokens) under A and under B: TTFT, and the answer
#       vs dsm-225 (NST 0 stock, cold) -- identity;
#   (5) identity per arm: the 28k copy and the 20-prompt greedy set c1 vs bz d0 (NST 0 stock).
#   Warm-up answers vs the stock reference (identw). Every form is exact by construction (r4d_selftest --gpcheck in xgp, tests/r4d_emu
#   ds41pf), so any difference is a defect and is reported as VIRHE.
# NO cc_sim (user 10.11.), no 2.9. No power / host / fan settings; production compose files read only (--project-directory).
# XGQ_IMAGE, XGQ_ITEM, XGQ_TABLE, XGQ_ROUNDS (2), XGQ_DUR (40 s mb windows), XGQ_PFD (16384,65536,131072), XGQ_BENCH (the xgp dir).
# DRYRUN=1 bash dsm-gpf.sh : prints every command; checks the image (docker image inspect + a no-device, no-network grep of its files),
#   the table source and the yaml (docker compose config --quiet); no server, no GPU, nothing written.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-gpf-tp4.yaml
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=${XGQ_OUT:-$R/xdsm/gpfab3}; export STATSDIR=$O/stats
RD=$R/xdsm/draft                                # bz's d0 files (identity reference: copy + greedy set)
RM=$R/xdsm                                      # item a's dsm-225: deep-dsm-225.txt(.json) (136k cold reference)
export IMAGE=${XGQ_IMAGE:-radiance-local:ds41-gpf-SHA8-5c13e555-l}
F32CODE=${XGQ_F32:-}
ITEM=${XGQ_ITEM:-57-p4zyzzzzz-ds-gpfab}
ROUNDS=${XGQ_ROUNDS:-2}; DUR=${XGQ_DUR:-40}; PFD=${XGQ_PFD:-16384,65536,131072}; BENCH=${XGQ_BENCH:-$R/xdsm/gpbench}
BENCHF=${XGQ_BENCHF:-$R/xdsm/gpbench-f32}   # the follow-up bench on this image: GPF-b's form + both byte checks again
BASEENV="RADIANCE_PLACE_TURNOVER=0"
ABB=$TL/abblocks2.py
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py

DQ=$QA/deepq.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
FIX0='{"gg_gemv": 2, "aql_acq": 2, "ds41_kvfold": 1, "ds41_ofold": 1, "ds41_onefill": 1}'
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
  RADIANCE_DS41_GG_PF RADIANCE_DS41_GG_PF_TABLE RADIANCE_DS41_F32_PF"
stackenv() {   # stackenv ENVWORDS: hx-stack's stack env (105's t3 arm + gg_gemv 2 + rule-w + K1 c/d/f declared and on) + ENVWORDS
  local kv
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  for kv in $1; do export "$kv"; done
}
armset() {   # armset NST ARMJSON [EXTRAJSON]: NST 0's fixed pin + the arm's switches (+ extra) as one POST body
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update(json.loads(sys.argv[2])); d.update(json.loads(sys.argv[3])); print(json.dumps(d))' \
    "$FIX0" "$2" "${3:-{\}}"
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
  log "RESULT xgq guard $tag: $st, 1-user ${ms} ms/step (limit $lim; NST 5 DSpark starts depth0b / hybrid 35.2-35.3, NST 0 bz dsd-d0 15.59)"
  case $st in SLOW*) return 2 ;; esac
}
FOLLOW_PID=""
follow() { [ "$DRY" = 1 ] && { echo "+ docker logs -f $CN > $1/engine.log &"; return; }
  docker logs -f $CN > $1/engine.log 2>&1 & FOLLOW_PID=$!; sleep 2; }
unfollow() { [ -n "$FOLLOW_PID" ] && { kill $FOLLOW_PID 2>/dev/null; wait $FOLLOW_PID 2>/dev/null; FOLLOW_PID=""; }; return 0; }
trap 'unfollow; down' EXIT
trap 'exit 143' TERM INT
thermline() { type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"; }
cmpfile() {   # cmpfile A B -> IDENTICAL (n chars) | DIFFERENT at char k (na vs nb chars) | not compared
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xgq $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xgq $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  echo "warm-up-$1 $r" >> $O/identity.txt
  case $r in IDENTICAL*) log "RESULT xgq $1 warm-up answers vs stock ($(basename $2)): $r" ;;
    *) log "RESULT xgq $1 warm-up answers vs stock ($(basename $2)): VIRHE $r -- NOT EXACT vs the tuned3l line (the start's arm: every switch at its start value)" ;; esac
}
modes() {   # modes TAG "ENVWORDS" LEVERS...: the stack's notes (t3 indexer, rule-w, K1 c/d/f on, no shexp lane) + each declared
  # lever's env in the container and its log literal -> 0, else 1
  local tag=$1 ew=$2 MLF SL AL XL KL NW NR ok=1 kv nm lo CE bad=""; shift 2
  [ "$DRY" = 1 ] && { echo "+ modes $tag: indexer note (mxfp4, latent bf16, compressor tail bound, router selection+sort separate, AR_PCN separate at 0 layers), staging line, no shared-expert-lane note, rule-w in every recorded pass, the lever K1 note (c / d / f declared, switches at 1); env [${ew:-none}] in the container; log literals of [$*]: $(for nm in "$@"; do echo -n "$nm '$(lf $nm 8 | tr , ' ')' "; done); mover kept-free 0 on 4 ranks (turnover 0)"; return 0; }
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  for kv in $ew; do echo "$CE" | grep -qxF "$kv" || bad="$bad env:${kv%%=*}"; done
  for nm in "$@"; do lo=$(lf $nm 8); [ "$lo" = - ] && continue
    docker logs $CN 2>&1 | grep -qF -- "${lo//,/ }" || bad="$bad log:$nm('${lo//,/ }')"; done
  for kv in $ew; do
    docker logs $CN 2>&1 | grep -F -- "${kv%%=*}" | grep -qE 'is not used|is not a count|refused|cannot' && bad="$bad refused:${kv%%=*}($(docker logs $CN 2>&1 | grep -F -- "${kv%%=*}" | grep -m1 -E 'is not used|is not a count|refused|cannot' | cut -c1-160))"
  done
  [ -z "$bad" ] || { log "RESULT xgq $tag: VIRHE a lever did not reach the engine:$bad"; ok=0; }
  case " $ew " in *" RADIANCE_PLACE_TURNOVER=0 "*)
    nz=$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots \([0-9]+ shadow, 0 more kept free')
    [ "$nz" = 4 ] || { log "RESULT xgq $tag: VIRHE turnover 0 asked but $nz of 4 ranks keep 0 slots free"; ok=0; } ;;
  esac
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  AL=$(docker logs $CN 2>&1 | grep -m1 'the MoE gather and its all-reduce as one kernel' | sed 's/^radiance: //' | cut -c1-300)
  XL=$(docker logs $CN 2>&1 | grep -m1 'the shared expert on the second lane' | sed 's/^radiance: //' | cut -c1-300)
  KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-900)
  log "RESULT xgq $tag modes: $(echo "$MLF" | cut -c1-700) | staging: ${SL:-none} | argather: ${AL:-none} | K1: ${KL:-none} | env [${ew:-none}]"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate" "post-collapse-norm separate at 0 layers" "next collapse-norm separate at 0 layers"; do
    case "$MLF" in *"$w"*) ;; *) log "RESULT xgq $tag: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  [ -n "$AL" ] || { log "RESULT xgq $tag: VIRHE no ARGATHER note"; ok=0; }
  [ -z "$XL" ] || { log "RESULT xgq $tag: VIRHE a shared-expert-lane note (RADIANCE_DS41_SHEXP_LANE should be unset): '$XL'"; ok=0; }
  [ -n "$SL" ] || { log "RESULT xgq $tag: VIRHE no prefill staging line (levels refused?)"; ok=0; }
  case "$KL" in
    *"in one launch at 0 of"*|*"rope+round trip at 0 ("*|*"not declared at"*|*"at 0 MoE layers"*|"")
      log "RESULT xgq $tag: VIRHE the lever K1 note does not show folds c / d / f declared: '${KL}'"; ok=0 ;;
  esac
  case "$KL" in *"ds41_kvfold=1"*"ds41_ofold=1"*"ds41_onefill=1"*) ;; *) log "RESULT xgq $tag: VIRHE the K1 c/d/f switches do not start at 1: '${KL}'"; ok=0 ;; esac
  NW=$(docker logs $CN 2>&1 | grep -c 'the acquire rule plays .*RADIANCE_AQL_ACQ=rule-w)'); NR=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_AQL_ACQ=rule)')
  [ "$NW" -gt 0 ] && [ "$NR" = 0 ] || { log "RESULT xgq $tag: VIRHE rule-w is not what the recorded passes play ($NW rule-w, $NR rule)"; ok=0; }
  [ $ok = 1 ] || keeplog $tag-nomode
  [ $ok = 1 ]
}
vramunits() {   # vramunits FILE: the routed experts' VRAM / pinned units of a /stats snapshot
  python3 -c 'import json,sys
try: d = json.load(open(sys.argv[1]))
except Exception as e: print("n/a (%s)" % e); sys.exit(0)
t = {x.get("name"): x.get("units") for x in d.get("experts", {}).get("tiers", [])}
print("vram %s, pinned %s" % (t.get("vram"), t.get("pinned")))' "$1" 2>&1
}
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
fn4try() {   # fn4try JSON -> the HTTP status of the POST (200 applied / 400 refused); dry: 200
  [ "$DRY" = 1 ] && { echo "+ POST $URL/admin/fn4 $1 (status)" >&2; echo 200; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null
}
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
pcreset() {   # pcreset TAG: POST /reset_prefix_cache (loopback operator) so the next prompt is prefilled cold
  [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o $O/$1-reset.json -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  log "RESULT xgq $1 prefix cache reset: HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"
}
copyt() {   # copyt TAG OFFSET: the copy test after a ~31k-token cold prompt (105's prompt: same offset -> same text)
  local t0=$(date +%s)
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset $2 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xgq $1 copy after a cold prompt: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300) | $(thermline $t0 $(date +%s))"
}
gsetc1() { run "(cd $D && timeout 900 python3 $GS $URL $O/$1-gs-c1.jsonl --conc 1 --max-tokens 192 > $O/$1-gs-c1.txt 2>&1)"; }
gcmp() { [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }; [ -s "$1" ] && [ -s "$2" ] || { echo "not compared (missing $1 or $2)"; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
idline() {   # idline KEY LABEL RESULT: identity.txt + one RESULT line (a difference is reported, not hidden)
  echo "$1 $3" >> $O/identity.txt
  case "$3" in IDENTICAL*|"not compared"*) log "RESULT xgq identity $2: $3" ;;
    *) log "RESULT xgq identity $2: VIRHE $3 -- every lever here is exact by construction, so this is a defect" ;; esac
}
deep() {   # deep TAG NST ARM ARMJSON: the arm's set, a reset, the 136k cold prefill (deepq.py, 512 greedy tokens) -> TTFT + the answer vs dsm-225
  local tag=$1-d$3 t0=$(date +%s) j
  fn4set "$(armset $2 "$4" '{"epoch_bump": 1}')"; nudge
  pcreset $tag
  run "(cd $D && timeout 1800 python3 $DQ $O/deep-$tag.txt 135000 512 > $O/deep-$tag.out 2>&1)"
  [ "$DRY" = 1 ] && { echo "+ RESULT xgq $tag 136k cold prefill: TTFT (cached must be 0) vs $RM/deep-dsm-225.txt.json; answer vs $RM/deep-dsm-225.txt"; return 0; }
  j=$(grep -m1 '^DEEPQ ' $O/deep-$tag.out | cut -c7-)
  [ -n "$j" ] || { log "RESULT xgq $tag 136k cold prefill: VIRHE no answer ($(tail -2 $O/deep-$tag.out | tr '\n' ' ' | cut -c1-300))"; return 1; }
  echo "$j" > $O/deep-$tag.json
  log "RESULT xgq $tag 136k cold prefill (deepq.py fixed prompt, 512 greedy tokens; arm $3): $(python3 -c 'import json,sys
d = json.loads(sys.argv[1])
try: r = json.load(open(sys.argv[2]))
except Exception: r = {}
t, rt = d.get("ttft"), r.get("ttft")
print("status %s, prompt %s tok, cached %s%s, TTFT %s s (dsm-225 cold %s s%s), %s prefill tok/s, %s tok" % (
    d.get("status"), d.get("prompt_tokens"), d.get("cached"), "" if d.get("cached") == 0 else " -- VIRHE not cold",
    t, rt, ", %+.1f %%" % ((t / rt - 1) * 100) if t and rt else "", round(d["prompt_tokens"] / t, 1) if t and d.get("prompt_tokens") else "-", d.get("tokens")))' "$j" $RM/deep-dsm-225.txt.json 2>&1 | cut -c1-400) | $(thermline $t0 $(date +%s))"
  idline deep-$tag "$tag 136k deep answer (cold) vs dsm-225 (NST 0, stock, cold)" "$(cmpfile $O/deep-$tag.txt $RM/deep-dsm-225.txt)"
}
ident() {   # ident TAG NST ARM ARMJSON [WHAT]: the arm's set, a reset, the 28k copy and the c1 greedy set vs bz d0
  local tag=$1-i$3 st
  fn4set "$(armset $2 "$4" '{"epoch_bump": 1}')"; nudge
  pcreset $tag
  copyt $tag 0; gsetc1 $tag
  [ "$DRY" = 1 ] && { echo "+ identity $tag (arm $3${5:+, $5}): 28k copy vs bz d0, greedy set c1 vs bz d0"; return 0; }
  st=$(fn4get)
  log "RESULT xgq $tag identity ran under: $(echo "$st" | cut -c1-1100)"
  idline copy-$tag "$tag 28k copy vs bz d0 (NST 0, stock)${5:+ [$5]}" "$(cmpfile $O/copy-$tag-run1.txt $RD/copy-dsd-d0-i1-run1.txt)"
  idline gs-$tag "$tag greedy set c1 vs bz d0${5:+ [$5]}" "$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$tag-gs-c1.jsonl)"
}
# ---------------------------------------------------------------- lever GPF's A / B (one NST 0 start)
abb() {   # abb TAG PHASE ROUNDS [PFDEPTHS [ARM2JSON NAME2]]: abblocks2.py rounds of A X X A (A = ARMA, X = ARMB or ARM2JSON, each
  # over FIX0) -> RESULT lines
  local tag=$1 ph=$2 rn=$3 pfd=${4:-} a b pa="" a2=${5:-$ARMB} n2=${6:-B}
  a=$(armset n0 "$ARMA"); b=$(armset n0 "$a2")
  [ -n "$pfd" ] && pa="--pf-depths $pfd"
  run "timeout 7200 python3 -I $ABB --url $URL --mb-dir $D --label $tag --phase $ph --dur $DUR --warm 6 --rounds $rn --a '$a' --b '$b' $pa --names A,$n2 --out $O/$tag-blocks.jsonl > $O/$tag-abb.txt 2>&1"
  [ "$DRY" = 1 ] && return 0
  local j; j=$(grep -m1 '^ABB ' $O/$tag-abb.txt | cut -c5-)
  [ -n "$j" ] || { log "RESULT xgq $tag: VIRHE no ABB line ($(tail -3 $O/$tag-abb.txt | tr '\n' ' ' | cut -c1-400))"; return 1; }
  echo "$j" > $O/$tag-abb.json
  log "RESULT xgq $tag (phase $ph, $rn round(s) A $n2 $n2 A, $n2 $a2 / A $ARMA; ratios $n2/A): $(python3 -c 'import json,sys
d = json.loads(sys.argv[1]); ph = sys.argv[2]
o = ["blocks %s (bad %s)" % (d.get("blocks"), d.get("blocks_bad"))]
if d.get("tps_ratio") is not None:
    o.append("%s decode tok/s B/A %.4f +- %.4f (A %.1f, B %.1f), ms a decode step B/A %s" % (ph, d["tps_ratio"], d.get("tps_ci95") or 0,
             d.get("A_tps_mean") or 0, d.get("B_tps_mean") or 0, "%.4f" % d["ms_ratio"] if d.get("ms_ratio") else "-"))
for k, v in sorted((d.get("pf") or {}).items(), key=lambda kv: int(kv[0])):
    o.append("fresh prefill %s: B/A %.4f +- %.4f (A %.1f, B %.1f tok/s), rounds %s" % (k, v.get("ratio") or 0, v.get("ci95") or 0, v.get("A_mean") or 0, v.get("B_mean") or 0, " ".join("%.4f" % x for x in v.get("rounds", []))))
print("; ".join(o))' "$j" $ph 2>&1 | cut -c1-1500)"
  # the blocks' absolute values (prefill tok/s per depth, mb pf prefill tok/s, decode tok/s) a block, for the record
  log "RESULT xgq $tag blocks: $(python3 -c 'import json,sys
out = []
for l in open(sys.argv[1]):
    try: r = json.loads(l)
    except ValueError: continue
    m = r.get("mb") or {}; pf = r.get("pf") or {}
    s = "%s%s %s" % (r.get("arm"), r.get("round", ""), "ok" if r.get("ok", True) else "BAD")
    if m.get("agg_dec_tps") is not None: s += " dec %.1f" % m["agg_dec_tps"]
    if m.get("prefill_tps") is not None: s += " mbpf %.1f (TTFT %s)" % (m["prefill_tps"], m.get("prefill_ttft"))
    for k, v in sorted(pf.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0):
        if isinstance(v, dict) and v.get("tps"): s += " pf%s %.1f" % (k, v["tps"])
    out.append(s)
# mb pf (phase pf has no decode counter): the prefill tok/s beside decoders, (B1 + B2) / (A1 + A2) a round
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
for rd in sorted({r.get("round") for r in rows}):
    a = [(r.get("mb") or {}).get("prefill_tps") for r in rows if r.get("round") == rd and r.get("arm") == "A" and r.get("ok", True)]
    b = [(r.get("mb") or {}).get("prefill_tps") for r in rows if r.get("round") == rd and r.get("arm") == "B" and r.get("ok", True)]
    if len(a) == 2 and len(b) == 2 and all(a) and all(b): out.append("round %s mb pf prefill B/A %.4f" % (rd, sum(b) / sum(a)))
print("; ".join(out))' $O/$tag-blocks.jsonl 2>&1 | cut -c1-2500)"
}
gpfnote() {   # gpfnote TAG: the start log's GPF table line (when a table is declared: entries > 0, no skipped entry), GET's gg_pf 0
  # and f32_pf 0 at start (served, not -1), B applies its values, back to A
  local tag=$1 tl sk s0 s1 bad=""
  [ "$DRY" = 1 ] && { echo "+ gpfnote $tag: the engine's env listing names RADIANCE_DS41_GG_PF_TABLE (when declared) and RADIANCE_DS41_F32_PF, no 'skipped' entry (the table line comes at the first gg_pf rule call: tblcheck after the first B prefills); GET gg_pf 0 + f32_pf 0 at start (served, not -1), gg_gemv 2 / aql_acq 2 / K1 c d f 1; POST B $ARMB + nudge -> applied; back to A $ARMA"; return 0; }
  tl=$(docker logs $CN 2>&1 | grep -m1 'lever GPF form table from RADIANCE_DS41_GG_PF_TABLE' | sed 's/^radiance: //')
  sk=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_DS41_GG_PF_TABLE entry')
  # the table is parsed lazily (first gg_pf rule call): no line expected yet; the engine's env listing must name the variables
  if [ -n "$TABLE" ]; then docker logs $CN 2>&1 | grep -qE '^I +RADIANCE_DS41_GG_PF_TABLE$' || bad="$bad env-listing(RADIANCE_DS41_GG_PF_TABLE)"; fi
  docker logs $CN 2>&1 | grep -qE '^I +RADIANCE_DS41_F32_PF$' || bad="$bad env-listing(RADIANCE_DS41_F32_PF)"
  [ "$sk" = 0 ] || bad="$bad skipped-entries($sk)"
  s0=$(fn4get)
  python3 -c 'import json,sys
w = json.loads(sys.argv[1])["switches"]
sys.exit(0 if w.get("gg_pf") == 0 and w.get("f32_pf") == 0 and w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) else 1)' "$s0" 2>/dev/null || bad="$bad at-start"
  fn4set "$(armset n0 "$ARMB" '{"epoch_bump": 1}')"; nudge; s1=$(fn4get)
  python3 -c 'import json,sys
s, b = json.loads(sys.argv[1]), json.loads(sys.argv[2])
sys.exit(0 if all(s["switches"].get(k) == v for k, v in b.items()) and not s["pending"] and s["epoch"] >= 1 else 1)' "$s1" "$ARMB" 2>/dev/null || bad="$bad arm-B"
  fn4set "$(armset n0 "$ARMA" '{"epoch_bump": 1}')"; nudge
  log "RESULT xgq $tag GPF note: ${tl:-no table line yet (parsed at the first gg_pf rule call)} | at start $(echo "$s0" | cut -c1-900) | B applied: $(echo "$s1" | python3 -c 'import json,sys; w = json.load(sys.stdin)["switches"]; print("gg_pf %s f32_pf %s" % (w.get("gg_pf"), w.get("f32_pf")))' 2>&1)$([ -n "$bad" ] && echo " -- VIRHE not as expected:$bad")"
  [ -z "$bad" ] || keeplog $tag-gpfnote
  [ -z "$bad" ]
}
tblcheck() {   # tblcheck TAG: after B blocks that prefilled -- the table line (N > 0 entries) and no skipped entry; VIRHE logged, no stop
  local tag=$1 tl sk
  [ -n "$TABLE" ] || return 0
  [ "$DRY" = 1 ] && { echo "+ tblcheck $tag: engine log 'lever GPF form table from RADIANCE_DS41_GG_PF_TABLE: N entries' with N > 0 (as many as the table's entries), no 'RADIANCE_DS41_GG_PF_TABLE entry ... skipped' line"; return 0; }
  tl=$(docker logs $CN 2>&1 | grep -m1 'lever GPF form table from RADIANCE_DS41_GG_PF_TABLE' | sed 's/^radiance: //')
  sk=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_DS41_GG_PF_TABLE entry')
  local n; n=$(echo "$TABLE" | tr ',' '\n' | grep -c .)
  case "$tl" in
    *": $n entries"*) [ "$sk" = 0 ] && { log "RESULT xgq $tag table check: '$tl' ($n declared, 0 skipped) -- the B blocks ran the table"; return 0; } ;;
  esac
  log "RESULT xgq $tag table check: VIRHE '${tl:-no table line after B prefills}' ($n declared, $sk skipped) -- B's gg part may not have run as declared (results logged anyway)"
  return 1
}
# ---------------------------------------------------------------- the table, the image, the plan
if [ -n "${XGQ_TABLE:-}" ]; then TABLE=$XGQ_TABLE; TSRC="XGQ_TABLE"
elif grep -q '^GPFORM table-4096+2048: ' $BENCH/gpsum.txt 2>/dev/null; then
  TABLE=$(grep -m1 '^GPFORM table-4096+2048: ' $BENCH/gpsum.txt | cut -c25-); TSRC="$BENCH/gpsum.txt (best at M 4096 + 2048, no loss at either)"
elif [ -s $BENCH/gpperf.txt ]; then TABLE=$(grep -m1 '^GPTAB ' $BENCH/gpperf.txt | cut -c7-); TSRC="$BENCH/gpperf.txt GPTAB"
else TABLE=""; TSRC="none ($BENCH/gpperf.txt missing)"; fi
case "$TABLE" in
  -) TABLE="" ;;
  *[!A-Za-z0-9_:x=,]*) log "RESULT xgq $ITEM: VIRHE the table '$TABLE' ($TSRC) has characters a GPTAB line does not"; exit 1 ;;
esac
GPCHK="$(grep -m1 '^GPVERDICT' $BENCH/gpcheck.txt 2>/dev/null) | this image: $(grep -m1 '^GPVERDICT' $BENCHF/gpcheck.txt 2>/dev/null)"
case "$GPCHK" in *"PASS (byte-identical"*"PASS (byte-identical"*) ;; *) [ -n "${XGQ_TABLE:-}" ] || [ -z "$TABLE" ] || { log "RESULT xgq $ITEM: VIRHE a gg byte check did not pass ('${GPCHK}') -- gg_pf left out of the A/B (forms not proven identical on this image)"; TABLE=""; } ;; esac
F32SRC="XGQ_F32"
if [ -z "$F32CODE" ]; then F32CODE=$(grep -m1 '^GPF32FORM ab: ' $BENCHF/gpsum.txt 2>/dev/null | awk '{print $3}'); F32SRC="$BENCHF/gpsum.txt GPF32FORM ab"; fi
case "$F32CODE" in 2[0-9]) ;; *) F32CODE="" ;; esac
F32CHK=$(grep -m1 '^GPF32VERDICT' $BENCHF/gpcheck.txt 2>/dev/null)
case "$F32CHK" in *"PASS (byte-identical"*) ;; *) [ -n "${XGQ_F32:-}" ] || [ -z "$F32CODE" ] || { log "RESULT xgq $ITEM: VIRHE the bench's f32 byte check did not pass ('${F32CHK:-no verdict}') -- f32_pf left out of the A/B"; F32CODE=""; } ;; esac
if [ -z "$TABLE" ] && [ -z "$F32CODE" ]; then
  log "RESULT xgq $ITEM: nothing to switch -- gg table (${TSRC}) none, f32 form (${F32SRC}) none: no form beat the stock kernels in xgp (or the bench did not run); the A/B does not run, no GPU time"
  log "RESULT xgq $ITEM: done -- nothing run"; exit 0
fi
ARMA='{"gg_pf": 0, "f32_pf": 0}'
ARMB="{\"gg_pf\": $([ -n "$TABLE" ] && echo 1 || echo 0), \"f32_pf\": ${F32CODE:-0}}"
ARMC=""; [ -n "$TABLE" ] && [ -n "$F32CODE" ] && [ "${XGQ_ARMC:-1}" = 1 ] && ARMC="{\"gg_pf\": 0, \"f32_pf\": $F32CODE}"
IID=$(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null)
[ -n "$IID" ] || { log "RESULT xgq: VIRHE image $IMAGE not on this host"; exit 1; }
HAVE=$(timeout 180 docker run --rm --network none --entrypoint sh $IMAGE -c 'B=/opt/radiance/bin/radiance; A=/opt/radiance/share/radiance/architectures/deepseek41.so; K=/opt/radiance/share/radiance/kernels
s=yes; for l in draft_depth0 reset_prefix_cache gg_pf f32_pf; do grep -qzF "$l" $B || s=no; done
grep -qzF RADIANCE_DS41_KVFOLD $A && grep -qzF rmsnorm_rope_roundtrip_kv_store $K/libr4d.so && grep -qzF r4d_gg_mr_pin $K/libr4d.so && grep -qzF RADIANCE_DS41_GG_PF_TABLE $K/libr4d.so && grep -qzF RADIANCE_DS41_F32_PF $K/libr4d.so || s=no
echo "STACK $s"' 2>/dev/null)
echo "$HAVE" | grep -qx "STACK yes" || { log "RESULT xgq: VIRHE $IMAGE is not a GPF / GPF-b stack image (gg_pf / f32_pf in the registry, RADIANCE_DS41_GG_PF_TABLE / _F32_PF in libr4d, K1, lever D2, reset_prefix_cache)"; exit 1; }
EW="$BASEENV RADIANCE_DS41_GG_PF=0 RADIANCE_DS41_F32_PF=0$([ -n "$TABLE" ] && echo " RADIANCE_DS41_GG_PF_TABLE=$TABLE")"
MIN=$(( 6 + ROUNDS * 4 * 2 + 4 + 5 + 4 + 8 + 9 ))
log "RESULT xgq $ITEM: start IMAGE=$IMAGE (${IID:7:12}) | gg table ($TSRC): ${TABLE:-none} | gg check: ${GPCHK:-n/a} | f32 form ($F32SRC): ${F32CODE:-none} | f32 check: ${F32CHK:-n/a} | NST 0 stack, A $ARMA / B $ARMB / C ${ARMC:-none} | c8 + fresh prefill $PFD x $ROUNDS ABBA rounds ($DUR s mb), c4 ABBA, mb pf ABBA, 136k cold A / B, identity A / B | ~$MIN min | refs: in-process A; identity bz d0 (copy, greedy c1), dsm-225 (136k deep, cold TTFT 51.56 s) | target 136k <= 35 s"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xgq: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  for v in RADIANCE_DS41_GG_GEMV RADIANCE_AQL_ACQ RADIANCE_DS41_KVFOLD RADIANCE_DS41_OFOLD RADIANCE_DS41_ONEFILL; do grep -q "^ *$v:" $Y || { log "RESULT xgq: VIRHE $Y does not pass $v"; exit 1; }; done
  grep -q 'XGQ_ENVFILE' $Y || { log "RESULT xgq: VIRHE $Y has no env_file entry"; exit 1; }
  for f in $GS $ABB $DQ $D/mb.py $D/ds41_tests.py; do [ -e $f ] || { log "RESULT xgq: VIRHE $f missing"; exit 1; }; done
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xgq: VIRHE $M225 missing or not verified"; exit 1; }
  for f in $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $RM/deep-dsm-225.txt $RM/deep-dsm-225.txt.json $WARM_REF_225; do [ -s $f ] || log "RESULT xgq: note -- reference $f missing (not compared)"; done
else
  for f in $Y $GS $ABB $DQ $D/mb.py $D/ds41_tests.py $MODELDIR/$M225 $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $RM/deep-dsm-225.txt $RM/deep-dsm-225.txt.json $WARM_REF_225; do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  echo "+ image $IMAGE (${IID:7:12}): GPF stack image OK"
fi
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XGQ_ENVFILE=/dev/null docker compose --project-directory $C -f $Y config --quiet > /dev/null 2>&1 \
  || { log "RESULT xgq: VIRHE docker compose config refuses $Y"; exit 1; }
[ "$DRY" = 1 ] && echo "+ yaml $Y: docker compose config OK (env_file \${XGQ_ENVFILE})"
mkd $O $O/guard $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt
evict
tag=gq3-n0; P=$O/$tag
stackenv "$EW"; export MODEL=$M225; HP29=19456; best_vars
export MNBT=4160 NST=0 XARGS="$XARGS --live"
export XGQ_ENVFILE=$O/$tag.env
log "RESULT xgq $tag: start (NST 0; A $ARMA / B $ARMB) IMAGE=$IMAGE MODEL=$MODEL $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ' | cut -c1-2500)| env file $XGQ_ENVFILE"
mkd $P
if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $EW > $XGQ_ENVFILE"; else printf '%s\n' $EW > $XGQ_ENVFILE; fi
gate $tag 30; rc=$?
[ $rc = 1 ] && { log "RESULT xgq $tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; exit 1; }
[ $rc = 2 ] && { log "RESULT xgq $tag: VIRHE over the gate limit after one restart -> stopped"; down; exit 1; }
modes $tag "$EW" || { log "RESULT xgq $tag: VIRHE the modes are not as expected -> stopped"; down; exit 1; }
gpfnote $tag || { log "RESULT xgq $tag: VIRHE the GPF table / switch is not as expected -> stopped"; down; exit 1; }
identw $tag $WARM_REF_225
follow $P
stopfile || abb $tag-pf c8 $ROUNDS $PFD
stopfile || tblcheck $tag
[ -n "$ARMC" ] && ! stopfile && abb $tag-pfC c8 1 $PFD "$ARMC" C
stopfile || abb $tag-c4 c4 1
stopfile || abb $tag-mbpf pf 1
for a in A B; do stopfile || deep $tag n0 $a "$([ $a = A ] && echo "$ARMA" || echo "$ARMB")"; done
for a in A B; do stopfile || ident $tag n0 $a "$([ $a = A ] && echo "$ARMA" || echo "$ARMB")"; done
unfollow
run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
down
if [ "$DRY" != 1 ]; then
  log "RESULT xgq summary: $(python3 -c 'import json,sys,os
O = sys.argv[1]; out = []
def rd(f):
    try: return json.load(open(f))
    except Exception: return None
p = rd(O + "/gq3-n0-pf-abb.json") or {}
for k, v in sorted((p.get("pf") or {}).items(), key=lambda kv: int(kv[0])): out.append("fresh %s B/A %.4f +- %.4f" % (k, v.get("ratio") or 0, v.get("ci95") or 0))
c = rd(O + "/gq3-n0-pfC-abb.json") or {}
for k, v in sorted((c.get("pf") or {}).items(), key=lambda kv: int(kv[0])): out.append("fresh %s C/A %.4f (f32_pf alone, 1 round)" % (k, v.get("ratio") or 0))
for t, ph in (("pf", "c8"), ("c4", "c4"), ("mbpf", "mb pf decode")):
    d = rd(O + "/gq3-n0-%s-abb.json" % t) or {}
    if d.get("tps_ratio") is not None: out.append("%s B/A %.4f +- %.4f" % (ph, d["tps_ratio"], d.get("tps_ci95") or 0))
for a in "AB":
    d = rd(O + "/deep-gq3-n0-d%s.json" % a) or {}
    if d.get("ttft"): out.append("136k cold %s TTFT %.2f s" % (a, d["ttft"]))
try: idl = [l.strip() for l in open(O + "/identity.txt")]
except OSError: idl = []
bad = [l for l in idl if "IDENTICAL" not in l and "not compared" not in l]
out.append("identity %d checks, %s" % (len(idl), "all IDENTICAL" if not bad else "VIRHE: " + " | ".join(bad)[:400]))
print("; ".join(out))' $O 2>&1 | cut -c1-3000)"
fi
unset $SWITCHES
log "RESULT xgq $ITEM: done -- $O"
