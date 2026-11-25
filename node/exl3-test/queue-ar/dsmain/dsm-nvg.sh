#!/bin/bash
# dsm-nvg.sh (10.11.2026, queue-ar/dsmain, ds-main-port nvlat agent) -- NOT a queue item; the body of the lever NVG A/B item
# (57-p4zyzzzzz-ds-<pos>-nvg). Log tag "RESULT xng ", out runs/xdsm/nvg. Helpers nvlatprof.py / nvlatmet.py (beside it),
# yaml ds41-nvg-tp4.yaml (= ds41-nvlat-tp4.yaml + RADIANCE_DS41_NVGATHER).
# LEVER NVG (branch ds41-nvlat, libr4d only): RADIANCE_DS41_NVGATHER=<rows> -- a decode-band pass of at most <rows> rows over
#   the packed (nvfp4) latent unpacks each row's selections to bf16 once (one launch into the attention's scratch), then runs
#   the plain bf16 decode kernel over them, instead of the packed kernel that unpacks every key in every head workgroup in
#   both its score and PV loops. Exact by construction (the stored bf16 bits, the bf16 latent's kernel); tests/r4d_emu
#   test_ds41attn / test_ds41fold hold it to the bf16 cache's bytes. Unset = today's launches and arena.
# WHY: item hszzz-nvlat (the NVFP4 latent's cost) and latb3 (9.11.): the packed decode attention 66.5 vs 15.7 us a call at 1 row.
# DESIGN: image radiance-local:ds41-nvlat-<sha8>-5c13e555-l (XNG_IMAGE; = the integrated image ds41-int c05de277 + lever NVG's
#   libr4d), NST 0, the stack env (as dsm-nvlat.sh: gg_gemv 2, aql_acq 2, K1 c/d/f, turnover 0), BOTH arms
#   RADIANCE_DS41_LATENT=nvfp4: A = NVGATHER unset (today's packed kernel), B = RADIANCE_DS41_NVGATHER=XNG_ROWS (16). Starts
#   XNG_STARTS (default "A:plain B:plain B:prof"):
#     plain: clean c1 short (stepprobe essay,code 512, twice), c1 at ~31k (the 28k copy's decode, cold + prefix hit), mb c8
#       XNG_C8DUR (45) s, the copy's cold TTFT and the warm-up's 16k prefill; IDENTITY: the copies and the 20-prompt greedy set
#       at c1 vs bz d0 (NST 0, stock, bf16 latent) -- must read IDENTICAL;
#     prof: --profile-ops windows short / long / c8 -> B's ops against hszzzz-nvlatp's B (packed kernel, the int image;
#       XNG_REFPROF / XNG_REFWIN, no re-run). The rule-w guard is skipped in a prof start (no recorded pass, ctx.cpp:181).
#   SUMMARY (nvlatprof.py summary): A (packed kernel) / B (NVG) ms a step and %, against the nvlat item's bf16 arm (context).
# ~27 min. NO cc_sim, no 2.9. No power / host / fan settings; production compose files read only. Nothing goes into a
# production line from here.
# XNG_IMAGE, XNG_ITEM, XNG_STARTS, XNG_ROWS, XNG_C8DUR, XNG_REFPROF, XNG_REFWIN.
# DRYRUN=1 bash dsm-nvg.sh : prints every command; checks the image and the yaml; no server, no GPU, nothing written.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-nvg-tp4.yaml
O=$R/xdsm/nvg; export STATSDIR=$O/stats
RN=$R/xdsm/nvlat                                # item hszzz-nvlat (the bf16 arm: context; its B prof: the ops reference)
RD=$R/xdsm/draft                                # bz's d0 files (identity reference: copy + greedy set; NST 0 stock, bf16 latent)
export IMAGE=${XNG_IMAGE:?XNG_IMAGE (radiance-local:ds41-nvlat-<sha8>-5c13e555-l)}
ITEM=${XNG_ITEM:-57-p4zyzzzzz-ds-nvg}
NROWS=${XNG_ROWS:-16}
RNP=$R/xdsm/nvlat-prof                         # the follow-up hszzzz-nvlatp's profiled pair (its B = the packed kernel)
REFPROF=${XNG_REFPROF:-$RNP/p3B-engine.log.gz}; REFWIN=${XNG_REFWIN:-$RNP/p3B-win.txt}
case $NROWS in [1-9]|[1-3][0-9]|4[0-8]) ;; *) echo "XNG_ROWS must be 1..48 (got '$NROWS')"; exit 1 ;; esac
STARTS=${XNG_STARTS:-A:plain B:plain B:prof}
C8DUR=${XNG_C8DUR:-45}
for s in $STARTS; do case $s in [AB]:plain|[AB]:stamp|[AB]:prof) ;; *) echo "XNG_STARTS takes A|B:plain|stamp|prof words (got '$s')"; exit 1 ;; esac; done
case $C8DUR in [1-9][0-9]|[1-9][0-9][0-9]) ;; *) echo "XNG_C8DUR must be 10..999 s (got '$C8DUR')"; exit 1 ;; esac
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
NP=$HD/nvlatprof.py; NM=$HD/nvlatmet.py
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
  RADIANCE_DS41_ARENA_DECODE_ROWS RADIANCE_DS41_ROUTER_BF16 RADIANCE_DS41_EMB_RAM RADIANCE_PLACE_SHADOW_SLOTS RADIANCE_XPREFETCH
  RADIANCE_DRAFT_LEND RADIANCE_DRAFT_PROBE_BACKOFF RADIANCE_DRAFT_LEND_BORROW RADIANCE_DEBUG_VRAM RADIANCE_DS41_NVFORM
  RADIANCE_DS41_NVGATHER"
nvenv() {   # nvenv A|B: dsm-stack.sh's stackenv (105's t3 arm + gg_gemv 2 + rule-w + K1 c/d/f at start) + turnover 0 + the nvfp4 latent; B adds lever NVG
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  export RADIANCE_PLACE_TURNOVER=0
  export RADIANCE_DS41_LATENT=nvfp4
  [ "$1" = B ] && export RADIANCE_DS41_NVGATHER=$NROWS
  return 0
}
servevars() {   # servevars plain|stamp|prof: dsm-stack.sh's NST 0 serve line (+ --profile-ops / the stamp env)
  export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=0 XARGS="$XARGS --live" AQLSTAMP= PROFEVERY=
  case $1 in stamp) export AQLSTAMP=1 ;; prof) export PROFEVERY=16 XARGS="$XARGS --profile-ops" ;; esac
}
now() { python3 -c 'import time; print(time.clock_gettime_ns(time.CLOCK_MONOTONIC))'; }   # the stamps' clock
loglines() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN | wc -l"; return 0; }; docker logs $CN 2>&1 | wc -l; }
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
trap 'down' EXIT
trap 'exit 143' TERM INT
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
pcreset() {   # pcreset TAG: POST /reset_prefix_cache so the next prompt is prefilled cold
  [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o $O/$1-reset.json -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  [ "$c" = 200 ] || log "RESULT xng $1 prefix cache reset: HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"
}
cmpfile() {   # cmpfile A B -> IDENTICAL (n chars) | DIFFERENT at char k (na vs nb chars) | not compared
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
gcmp() { [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }; [ -s "$1" ] && [ -s "$2" ] || { echo "not compared (missing $1 or $2)"; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
idline() {   # idline TAG WHAT RESULT: identity.txt + a RESULT line
  [ "$DRY" = 1 ] && { echo "+ RESULT xng $1 identity ($2): $3"; return 0; }
  echo "$1 $2: $3" >> $O/identity.txt
  case $3 in IDENTICAL*) log "RESULT xng $1 identity: $2: $3" ;; *) log "RESULT xng $1 identity: $2: $3 -- reported, not hidden (expected IDENTICAL: the nvfp4 latent is lossless)" ;; esac
}
modes() {   # modes TAG ARM: the indexer note names the arm's latent; the K1 note (all three declared and on); rule-w; turnover 0; no drafter
  [ "$DRY" = 1 ] && { echo "+ modes $1: container env RADIANCE_DS41_NVGATHER $([ $2 = B ] && echo "= $NROWS" || echo unset); indexer note with 'latent nvfp4' and 'keys mxfp4', the K1 note (ds41_kvfold=1 ds41_ofold=1 ds41_onefill=1, every fold declared), rule-w in every recorded pass ($([ "${3:-}" = prof ] && echo "SKIPPED in a prof start: no recorded pass" || echo checked)), 4 ranks '0 more kept free', no DSpark line"; return 0; }
  local MLF KL NW NR NZ CE ok=1 lat=nvfp4
  CE=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null)
  if [ $2 = B ]; then echo "$CE" | grep -qx "RADIANCE_DS41_NVGATHER=$NROWS" || { log "RESULT xng $1: VIRHE RADIANCE_DS41_NVGATHER=$NROWS is not in the container's env"; ok=0; }
  else echo "$CE" | grep -q "^RADIANCE_DS41_NVGATHER=." && { log "RESULT xng $1: VIRHE RADIANCE_DS41_NVGATHER is set in arm A"; ok=0; }; fi
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-500)
  NW=$(docker logs $CN 2>&1 | grep -c 'the acquire rule plays .*RADIANCE_AQL_ACQ=rule-w)'); NR=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_AQL_ACQ=rule)')
  NZ=$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots \([0-9]+ shadow, 0 more kept free')
  log "RESULT xng $1 modes: $(echo "$MLF" | cut -c1-700) | K1: ${KL:-none} | rule-w passes $NW, rule $NR | turnover 0 on $NZ of 4 ranks"
  case "$MLF" in *"latent $lat"*) ;; *) log "RESULT xng $1: VIRHE the indexer note does not say 'latent $lat' (arm $2)"; ok=0 ;; esac
  case "$MLF" in *"keys mxfp4"*) ;; *) log "RESULT xng $1: VIRHE the indexer note lacks 'keys mxfp4'"; ok=0 ;; esac
  case "$KL" in *"in one launch at 0 of"*|*"rope+round trip at 0 ("*|*"not declared at"*|*"at 0 MoE layers"*|"") log "RESULT xng $1: VIRHE the K1 note does not show all three folds declared"; ok=0 ;; esac
  case "$KL" in *"ds41_kvfold=1"*"ds41_ofold=1"*"ds41_onefill=1"*) ;; *) log "RESULT xng $1: VIRHE the K1 switches do not start at 1"; ok=0 ;; esac
  if [ "${3:-}" = prof ]; then
    log "RESULT xng $1: rule-w check skipped (--profile-ops: pass recording off, core/runtime/ctx.cpp:181; RADIANCE_AQL_ACQ=rule-w set in the env as in every arm; $NW rule-w / $NR rule pass lines)"
  else
    [ "$NW" -gt 0 ] && [ "$NR" = 0 ] || { log "RESULT xng $1: VIRHE rule-w is not what the recorded passes play"; ok=0; }
  fi
  [ "$NZ" = 4 ] || { log "RESULT xng $1: VIRHE turnover 0 asked but $NZ of 4 ranks keep 0 slots free"; ok=0; }
  docker logs $CN 2>&1 | grep -qiE "DSpark drafter|'dspark' proposes" && { log "RESULT xng $1: VIRHE a DSpark drafter line at NST 0"; ok=0; }
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
swnote() {   # swnote TAG: GET at start (gg_gemv 2, aql_acq 2, the K1 three 1, no drafter) and FIX0 + epoch_bump applied at a step
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (expect gg_gemv 2, aql_acq 2, ds41_kvfold / ds41_ofold / ds41_onefill 1, draft_capable 0); POST $FIX0 + epoch_bump + a 4-token request -> applied"; return 0; }
  local s0 s1
  s0=$(fn4get)
  fn4set "$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d["epoch_bump"]=1; print(json.dumps(d))' "$FIX0")"; nudge; s1=$(fn4get)
  log "RESULT xng $1 switch note: at start $(echo "$s0" | cut -c1-400) | after the pin: $(echo "$s1" | cut -c1-400)"
  python3 -c 'import json,sys
s0, s1 = json.loads(sys.argv[1]), json.loads(sys.argv[2]); n = json.loads(sys.argv[3]); w = s0["switches"]
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) and s0["draft_capable"] == 0
ok = ok and all(s1["switches"].get(k) == v for k, v in n.items()) and not s1["pending"] and s1["epoch"] >= 1
sys.exit(0 if ok else 1)' "$s0" "$s1" "$FIX0" 2>/dev/null
}
identw() {   # identw TAG: the warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xng $1 warm-up answers vs $WARM_REF_225"; return 0; }
  [ -s $WARM_REF_225 ] || { log "RESULT xng $1 warm-up answers: no reference $WARM_REF_225"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $WARM_REF_225 2>&1 | cut -c1-300)
  idline $1 "warm-up answers vs stock ($(basename $WARM_REF_225))" "$r"
}
spline() {   # spline FILE: stepprobe's J lines -> "ms tps n"
  python3 -c 'import json,statistics,sys
r = [json.loads(l[2:]) for l in open(sys.argv[1]) if l.startswith("J ")]
print("%.3f ms a step, %.2f tok/s, %d prompts (%s)" % (statistics.mean(x["ms_per_decode_step"] for x in r), statistics.mean(x["decode_tps"] for x in r), len(r),
      ", ".join("%s %.2f ms" % (x["prompt"], x["ms_per_decode_step"]) for x in r)) if r else "none")' "$1" 2>/dev/null || echo none
}
c1short() {   # c1short TAG FILE PROMPTS GEN
  run "(cd $D && timeout 900 python3 stepprobe.py $3 $4 > $2 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xng $1 c1 short (stepprobe $3 $4): $(spline $2)"
}
copyrun() {   # copyrun TAG FILE CTAG: the 28k copy (bz d0's arguments) between two /metrics snapshots -> NVM + the copy's J line
  run "(cd $D && timeout 1500 python3 -I $NM $URL -- python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset 0 --tag=-$3 --outdir $O > $2 2>&1)"
  [ "$DRY" = 1 ] && return 0
  log "RESULT xng $1 copy $3 (~31k prompt): $(python3 -c 'import json,sys
L = open(sys.argv[1]).read().splitlines()
n = next((json.loads(l[4:]) for l in L if l.startswith("NVM ")), {}); j = next((json.loads(l[2:]) for l in L if l.startswith("J ") and "\"copy\"" in l), {})
u = j.get("usage") or {}
print("prompt %s tok, cached %s, TTFT %s s, exact %s, %s chars out | decode %s steps, %s ms a step, %s tokens" % (u.get("prompt_tokens"), j.get("cached"),
      j.get("ttft"), j.get("exact"), j.get("out_chars"), n.get("decode_steps"), n.get("ms_per_decode_step"), n.get("decode_tokens")))' $2 2>&1 | cut -c1-400)"
  idline $1 "28k copy $3 vs bz d0 (NST 0, stock, bf16)" "$(cmpfile $O/copy-$3-run1.txt $RD/copy-dsd-d0-i1-run1.txt)"
}
# ---------------------------------------------------------------- one start
onestart() {   # onestart ARM KIND
  local arm=$1 kind=$2 tag rc ms P f l0 l1 t0 t1
  tag=nv-$(echo $arm | tr AB ab)$kind; P=p1$arm; case $kind in stamp) P=p2$arm ;; prof) P=p3$arm ;; esac
  nvenv $arm; servevars $kind
  case $kind in stamp) export STATSDIR=$O/stats-stamp-$arm; mkd $STATSDIR; run "find $STATSDIR -maxdepth 1 -name 'stamps.csv*' -delete" ;;
                *) export STATSDIR=$O/stats ;; esac
  log "RESULT xng $tag: start arm $arm (nvfp4 latent, $([ $arm = B ] && echo "RADIANCE_DS41_NVGATHER=$NROWS: lever NVG" || echo "NVGATHER unset: the packed kernel")), $kind$([ $kind = stamp ] && echo " (RADIANCE_AQL_STAMP=1: ~8 % inflated)")$([ $kind = prof ] && echo " (--profile-ops, PROFILE_EVERY 16: PROFILED, not a speed)") IMAGE=$IMAGE $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
  if [ $kind = plain ]; then
    gate $tag 30; rc=$?
    [ $rc = 1 ] && { keeplog $tag-fail; log "RESULT xng $tag: VIRHE not healthy"; down; return 1; }
    [ $rc = 2 ] && log "RESULT xng $tag: note -- slow after one restart (read its numbers with that in mind)"
  else
    restart $tag || { keeplog $tag-fail; log "RESULT xng $tag: VIRHE not healthy"; down; return 1; }
    ms=$(check $tag 1)
    [ "$DRY" = 1 ] || log "RESULT xng $tag warm-up + 1-user fastcheck ($kind, not gated): $ms ms/step"
  fi
  modes $tag $arm $kind || { log "RESULT xng $tag: VIRHE not the asked configuration -> start stopped"; down; return 1; }
  swnote $tag || { log "RESULT xng $tag: VIRHE the switches are not as expected -> start stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag
  case $kind in
    plain)
      stopfile || c1short $tag $O/$P-c1s-r1.txt essay,code 512
      if ! stopfile; then pcreset $tag-cold; copyrun $tag $O/$P-c1L-cold.txt $P-cold; fi
      stopfile || copyrun $tag $O/$P-c1L-hit.txt $P-hit
      if ! stopfile; then
        run "(cd $D && timeout 1200 python3 mb.py --tag nv-$P --phases c8 --dur $C8DUR --warm 10 > $O/$P-c8.txt 2>&1)"
        [ "$DRY" = 1 ] || log "RESULT xng $tag mb c8 $C8DUR s: $(grep -m1 '^MB ' $O/$P-c8.txt | cut -c4- | cut -c1-500)"
      fi
      stopfile || c1short $tag $O/$P-c1s-r2.txt essay,code 512
      if ! stopfile; then
        pcreset $tag-gs
        run "(cd $D && timeout 900 python3 $GS $URL $O/$P-gs-c1.jsonl --conc 1 --max-tokens 192 > $O/$P-gs-c1.txt 2>&1)"
        idline $tag "greedy set c1 vs bz d0 (NST 0, stock, bf16)" "$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$P-gs-c1.jsonl)"
      fi ;;
    stamp)
      stopfile || run "(cd $D && timeout 600 python3 stepprobe.py essay 128 > /dev/null 2>&1)"   # settle (untimed)
      if ! stopfile; then
        if [ "$DRY" = 1 ]; then t0=T0; else t0=$(now); fi
        c1short $tag $O/$P-c1s.txt essay 256
        if [ "$DRY" = 1 ]; then t1=T1; else t1=$(now); fi
        run "echo 'short $t0 $t1' >> $O/$P-win.txt"
      fi
      if ! stopfile; then pcreset $tag-cold; copyrun $tag $O/$P-c1L-cold.txt $P-cold; fi
      if ! stopfile; then
        if [ "$DRY" = 1 ]; then t0=T0; else t0=$(now); fi
        copyrun $tag $O/$P-c1L-hit.txt $P-hit
        if [ "$DRY" = 1 ]; then t1=T1; else t1=$(now); fi
        run "echo 'long $t0 $t1' >> $O/$P-win.txt"
      fi ;;
    prof)
      stopfile || run "(cd $D && timeout 600 python3 stepprobe.py essay 64 > /dev/null 2>&1)"   # settle
      if ! stopfile; then
        l0=$(loglines); c1short $tag $O/$P-c1s.txt essay,code 256; run "sleep 2"; l1=$(loglines)
        run "echo 'short $l0 $l1' >> $O/$P-win.txt"
      fi
      if ! stopfile; then pcreset $tag-cold; copyrun $tag $O/$P-c1L-cold.txt $P-cold; fi
      if ! stopfile; then
        l0=$(loglines); copyrun $tag $O/$P-c1L-hit.txt $P-hit; run "sleep 2"; l1=$(loglines)
        run "echo 'long $l0 $l1' >> $O/$P-win.txt"
      fi
      if ! stopfile; then
        l0=$(loglines)
        run "(cd $D && timeout 900 python3 mb.py --tag nv-$P --phases c8 --dur 30 --warm 5 > $O/$P-c8.txt 2>&1)"
        run "sleep 2"; l1=$(loglines)
        run "echo 'c8 $l0 $l1' >> $O/$P-win.txt"
        [ "$DRY" = 1 ] || log "RESULT xng $tag mb c8 30 s (PROFILED): $(grep -m1 '^MB ' $O/$P-c8.txt | cut -c4- | cut -c1-300)"
      fi
      run "docker logs $CN > $O/$P-engine.log 2>&1" ;;
  esac
  keeplog $tag
  down   # the stamp writer flushes at exit
  export STATSDIR=$O/stats AQLSTAMP= PROFEVERY=
  log "RESULT xng $tag: done"
}
# ---------------------------------------------------------------- analysis
analyse_ops() {   # B's profiled windows against the nvlat item's B (the packed kernel; no re-run)
  local w wa wb f REF=$O/ref-p3B-engine.log
  if [ "$DRY" = 1 ]; then
    echo "+ zcat $REFPROF > $REF; nvlatprof.py ops --a $REF:<$REFWIN> --b $O/p3B-engine.log:L0:L1 (short, long, c8) -> $O/op-{short,long,c8}.json"
    return 0
  fi
  [ -s $REFPROF ] && [ -s $REFWIN ] || { log "RESULT xng ops: no reference profile ($REFPROF, $REFWIN) -- B's ops alone"; }
  case $REFPROF in *.gz) zcat $REFPROF > $REF 2>/dev/null ;; *) cp $REFPROF $REF 2>/dev/null ;; esac
  for w in short long c8; do
    wa=$(awk -v w=$w '$1 == w {print $2 ":" $3}' $REFWIN 2>/dev/null | tail -1); wb=$(awk -v w=$w '$1 == w {print $2 ":" $3}' $O/p3B-win.txt 2>/dev/null | tail -1)
    if [ -n "$wa" ] && [ -n "$wb" ] && [ -s $REF ] && [ -s $O/p3B-engine.log ]; then
      timeout 1800 python3 -I $NP ops --a $REF:$wa --b $O/p3B-engine.log:$wb --label op-$w --json $O/op-$w.json > $O/op-$w.txt 2>&1
      while IFS= read -r l; do log "RESULT xng ops $w (A = nvlat item's packed-kernel prof, B = NVG) ${l:0:1500}"; done < <(grep '^NVLP op-' $O/op-$w.txt | sed "s/^NVLP op-$w //")
    else log "RESULT xng ops $w: no windows or no engine log (ref '$wa', B '$wb')"; fi
  done
  for f in $REF $O/p3B-engine.log; do [ -e $f ] && gzip -f $f; done
  return 0
}
summary() {
  [ "$DRY" = 1 ] && { echo "+ nvlatprof.py summary $O -> RESULT xng summary: clean c1 short / c1 ~31k / c8 / prefill A vs B, stamped critical path vs profiled device sum per window, kernels only in B, identity"; return 0; }
  timeout 600 python3 -I $NP summary $O --json $O/summary.json > $O/summary.txt 2>&1
  local s; s=$(grep '^NVLP summary ' $O/summary.txt | sed 's/^NVLP summary //' | sed 's/A = bf16 latent, B = nvfp4/A = the packed kernel, B = lever NVG/' | tr '\n' ' ' | sed 's/  */ /g')
  log "RESULT xng summary (A = packed kernel, B = lever NVG $NROWS rows; both nvfp4): ${s:0:5000}"
  [ -d $RN ] && log "RESULT xng context (item hszzz-nvlat, the int image, bf16 latent arm A / packed B): $(timeout 600 python3 -I $NP summary $RN 2>/dev/null | grep -m1 '^NVLP summary CLEAN' | sed 's/^NVLP summary //' | cut -c1-1500)"
}
# ---------------------------------------------------------------- main
log "RESULT xng $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) starts [$STARTS], c8 $C8DUR s, yaml $(basename $Y) | lever NVG rows $NROWS (B), A = the packed kernel, both nvfp4 | LEVERS NVFP4 latent: 1 user +10-12 % (15.81 -> 17.71 ms), c8 -4..-6 %; latb3 packed decode attention 66.5 vs 15.7 us a call"
docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xng: VIRHE image $IMAGE not on this host"; exit 1; }
[ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -qzF RADIANCE_DS41_LATENT /opt/radiance/share/radiance/architectures/deepseek41.so && grep -qzF latent_format /opt/radiance/share/radiance/kernels/libr4d.so && grep -qzF RADIANCE_DS41_KVFOLD /opt/radiance/share/radiance/architectures/deepseek41.so && grep -qzF rmsnorm_rope_roundtrip_kv_store /opt/radiance/share/radiance/kernels/libr4d.so && grep -qzF reset_prefix_cache /opt/radiance/bin/radiance && grep -qzF RADIANCE_AQL_STAMP /opt/radiance/bin/radiance && grep -qzF RADIANCE_DS41_NVGATHER /opt/radiance/share/radiance/kernels/libr4d.so && echo yes' 2>/dev/null)" = yes ] \
  || { log "RESULT xng: VIRHE $IMAGE lacks the nvfp4 latent / K1 / reset_prefix_cache / AQL stamp / lever NVG strings"; exit 1; }
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)) carries RADIANCE_DS41_LATENT, latent_format, K1, reset_prefix_cache, RADIANCE_AQL_STAMP, RADIANCE_DS41_NVGATHER -- OK"
for v in RADIANCE_DS41_NVGATHER RADIANCE_DS41_LATENT RADIANCE_AQL_STAMP RADIANCE_PROFILE_EVERY RADIANCE_DS41_GG_GEMV RADIANCE_AQL_ACQ RADIANCE_DS41_KVFOLD RADIANCE_DS41_OFOLD RADIANCE_DS41_ONEFILL RADIANCE_PLACE_TURNOVER; do
  grep -q "^ *$v:" $Y || { log "RESULT xng: VIRHE $Y does not pass $v"; exit 1; }
done
if [ "$DRY" = 1 ]; then
  ( nvenv B; servevars prof; export ARGS="$(serve_args)" STATSDIR=$O/stats; docker compose --project-directory $C -f $Y config --quiet && echo "+ compose config ($(basename $Y), arm B, prof): OK" ) || echo "+ compose config: FAILED"
  for f in $REFPROF $REFWIN $Y $GS $NP $NM $HD/ds_opprof.py $D/stepprobe.py $D/mb.py $D/ds41_tests.py $MODELDIR/$M225 $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $WARM_REF_225; do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  python3 -I $NP summary /nonexistent-nvlat > /dev/null 2>&1 && echo "+ nvlatprof.py runs" || echo "+ nvlatprof.py: FAILED"
  python3 -I $NM > /dev/null 2>&1; [ $? = 2 ] && echo "+ nvlatmet.py runs" || echo "+ nvlatmet.py: FAILED"
else
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xng: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  for f in $GS $NP $NM $HD/ds_opprof.py $D/stepprobe.py $D/mb.py $D/ds41_tests.py; do [ -e $f ] || { log "RESULT xng: VIRHE $f missing"; exit 1; }; done
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xng: VIRHE $M225 missing or not verified"; exit 1; }
  for f in $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $WARM_REF_225; do [ -s $f ] || log "RESULT xng: note -- reference $f missing (not compared)"; done
fi
mkd $O $O/guard $O/stats
[ "$DRY" = 1 ] || { : > $O/identity.txt; rm -f $O/p3A-win.txt $O/p3B-win.txt; }
evict
for s in $STARTS; do
  stopfile && break
  onestart ${s%%:*} ${s##*:}
done
analyse_ops
summary
unset $SWITCHES
export AQLSTAMP= PROFEVERY=
down
log "RESULT xng $ITEM: done -- $O"
