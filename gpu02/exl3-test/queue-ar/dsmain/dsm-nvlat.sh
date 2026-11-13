#!/bin/bash
# dsm-nvlat.sh (10.11.2026, queue-ar/dsmain, ds-main-port nvlat agent) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-hszzz-nvlat.sh. Log tag "RESULT xnv ", out runs/xdsm/nvlat.
# WHERE THE NVFP4 LATENT'S 1-USER DECODE COST GOES (LEVERS "NVFP4 latent" / "NVFP4 latent decode plateau": exact,
#   RADIANCE_DS41_LATENT=nvfp4 = 288 instead of 1024 bytes a compressed latent entry, the 1M-context capacity lever;
#   cost 1 user +10-12 % (15.81 -> 17.71 ms), c8 -4..-6 %, prefill -15.1 %; kernel v3 = v2; latb3 bench 9.11.: the
#   decode attention at 1 row 15.7 -> 66.5 us a call (x4.2) at 135k, 110 us at 16k). The latent is read at declare
#   (deepseek41.cpp Model::lnv), so A and B are separate starts.
# DESIGN: the integrated image radiance-local:ds41-int-c05de277-5c13e555-l (XNV_IMAGE), NST 0, the stack env (105's t3 arm +
#   gg_gemv 2 + aql_acq 2 (rule-w) + the K1 folds c / d / f, every one declared and on at start, + turnover 0 =
#   RADIANCE_PLACE_TURNOVER=0); A = RADIANCE_DS41_LATENT unset (bf16), B = RADIANCE_DS41_LATENT=nvfp4; nothing else differs.
#   Six starts, XNV_STARTS (default "A:plain B:plain B:stamp A:stamp A:prof B:prof", ABBAAB):
#     plain (gated 30 ms): CLEAN readings -- stepprobe essay,code 512 (c1 short) twice (start and end of the start), the
#       28k copy (ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --offset 0: a ~31k-token cold
#       prompt; its TTFT = a cold prefill reading, its 900-token decode between two /metrics snapshots = c1 at ~31k,
#       nvlatmet.py) then the same request again (prefix hit: a second ~31k decode reading), mb c8 XNV_C8DUR (45) s;
#       IDENTITY: both copies and the 20-prompt greedy set at c1 vs bz d0 (runs/xdsm/draft, NST 0 stock, bf16) byte for byte;
#     stamp (RADIANCE_AQL_STAMP=1, ~8 % inflated, not gated): windows SHORT (stepprobe essay 256 after a 128 settle) and
#       LONG (the copy's prefix-hit request after its cold one) -> the trunk pass's critical path (chain run + gaps), the
#       kernels' device / exposed / on-chain us a pass (nvlatprof.py stamps, k1prof.py's chain walk); the copies vs bz d0;
#     prof (--profile-ops, RADIANCE_PROFILE_EVERY 16; PROFILED, not a speed): windows SHORT (stepprobe essay,code 256),
#       LONG (the copy's hit request), C8 (mb c8 30 s) -> per op and class device ms a step (nvlatprof.py ops, ds_opprof.py's
#       parser); the copies vs bz d0.
#   SUMMARY (nvlatprof.py summary): per clean reading ms a step A / B and %; per window the stamped critical-path difference
#   against the profiled device-sum difference and the clean one; per kernel family B vs A, the kernels only in B; identity.
# ~50 min (six starts of ~4.5 min, plain readings ~4 min each, stamp ~1.5, prof ~4, analysis ~3). EXACT expected (the
# nvfp4 latent is lossless): every identity line should read IDENTICAL; a DIFFERENT is reported, not hidden.
# NO cc_sim, no 2.9. No power / host / fan settings; production compose files read only (--project-directory). Nothing goes
# into a production line from here.
# XNV_IMAGE, XNV_ITEM, XNV_STARTS, XNV_C8DUR.
# DRYRUN=1 bash dsm-nvlat.sh : prints every command; checks the image (docker image inspect + a no-device, no-network grep of
#   its files) and the yaml (docker compose config --quiet); no server, no GPU, nothing written.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-nvlat-tp4.yaml
O=$R/xdsm/nvlat; export STATSDIR=$O/stats
RD=$R/xdsm/draft                                # bz's d0 files (identity reference: copy + greedy set; NST 0 stock, bf16 latent)
export IMAGE=${XNV_IMAGE:-radiance-local:ds41-int-c05de277-5c13e555-l}
ITEM=${XNV_ITEM:-57-p4zyzzzzz-ds-hszzz-nvlat}
STARTS=${XNV_STARTS:-A:plain B:plain B:stamp A:stamp A:prof B:prof}
C8DUR=${XNV_C8DUR:-45}
for s in $STARTS; do case $s in [AB]:plain|[AB]:stamp|[AB]:prof) ;; *) echo "XNV_STARTS takes A|B:plain|stamp|prof words (got '$s')"; exit 1 ;; esac; done
case $C8DUR in [1-9][0-9]|[1-9][0-9][0-9]) ;; *) echo "XNV_C8DUR must be 10..999 s (got '$C8DUR')"; exit 1 ;; esac
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
  RADIANCE_DRAFT_LEND RADIANCE_DRAFT_PROBE_BACKOFF RADIANCE_DRAFT_LEND_BORROW RADIANCE_DEBUG_VRAM RADIANCE_DS41_NVFORM"
nvenv() {   # nvenv A|B: dsm-stack.sh's stackenv (105's t3 arm + gg_gemv 2 + rule-w + K1 c/d/f at start) + turnover 0; B adds the nvfp4 latent
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2 RADIANCE_AQL_ACQ=rule-w RADIANCE_DS41_KVFOLD=1 RADIANCE_DS41_OFOLD=1 RADIANCE_DS41_ONEFILL=1
  export RADIANCE_PLACE_TURNOVER=0
  [ "$1" = B ] && export RADIANCE_DS41_LATENT=nvfp4
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
  [ "$c" = 200 ] || log "RESULT xnv $1 prefix cache reset: HTTP $c $(head -c 200 $O/$1-reset.json 2>/dev/null)"
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xnv $1 identity ($2): $3"; return 0; }
  echo "$1 $2: $3" >> $O/identity.txt
  case $3 in IDENTICAL*) log "RESULT xnv $1 identity: $2: $3" ;; *) log "RESULT xnv $1 identity: $2: $3 -- reported, not hidden (expected IDENTICAL: the nvfp4 latent is lossless)" ;; esac
}
modes() {   # modes TAG ARM: the indexer note names the arm's latent; the K1 note (all three declared and on); rule-w; turnover 0; no drafter
  [ "$DRY" = 1 ] && { echo "+ modes $1: indexer note with 'latent $([ $2 = B ] && echo nvfp4 || echo bf16)' and 'keys mxfp4', the K1 note (ds41_kvfold=1 ds41_ofold=1 ds41_onefill=1, every fold declared), rule-w in every recorded pass, 4 ranks '0 more kept free', no DSpark line"; return 0; }
  local MLF KL NW NR NZ ok=1 lat=bf16
  [ $2 = B ] && lat=nvfp4
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  KL=$(docker logs $CN 2>&1 | grep -m1 'lever K1 folds' | sed 's/^radiance: //' | cut -c1-500)
  NW=$(docker logs $CN 2>&1 | grep -c 'the acquire rule plays .*RADIANCE_AQL_ACQ=rule-w)'); NR=$(docker logs $CN 2>&1 | grep -c 'RADIANCE_AQL_ACQ=rule)')
  NZ=$(docker logs $CN 2>&1 | grep -cE '^I\[[0-3]\] mover: [0-9]+ slab slots \([0-9]+ shadow, 0 more kept free')
  log "RESULT xnv $1 modes: $(echo "$MLF" | cut -c1-700) | K1: ${KL:-none} | rule-w passes $NW, rule $NR | turnover 0 on $NZ of 4 ranks"
  case "$MLF" in *"latent $lat"*) ;; *) log "RESULT xnv $1: VIRHE the indexer note does not say 'latent $lat' (arm $2)"; ok=0 ;; esac
  case "$MLF" in *"keys mxfp4"*) ;; *) log "RESULT xnv $1: VIRHE the indexer note lacks 'keys mxfp4'"; ok=0 ;; esac
  case "$KL" in *"in one launch at 0 of"*|*"rope+round trip at 0 ("*|*"not declared at"*|*"at 0 MoE layers"*|"") log "RESULT xnv $1: VIRHE the K1 note does not show all three folds declared"; ok=0 ;; esac
  case "$KL" in *"ds41_kvfold=1"*"ds41_ofold=1"*"ds41_onefill=1"*) ;; *) log "RESULT xnv $1: VIRHE the K1 switches do not start at 1"; ok=0 ;; esac
  [ "$NW" -gt 0 ] && [ "$NR" = 0 ] || { log "RESULT xnv $1: VIRHE rule-w is not what the recorded passes play"; ok=0; }
  [ "$NZ" = 4 ] || { log "RESULT xnv $1: VIRHE turnover 0 asked but $NZ of 4 ranks keep 0 slots free"; ok=0; }
  docker logs $CN 2>&1 | grep -qiE "DSpark drafter|'dspark' proposes" && { log "RESULT xnv $1: VIRHE a DSpark drafter line at NST 0"; ok=0; }
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
swnote() {   # swnote TAG: GET at start (gg_gemv 2, aql_acq 2, the K1 three 1, no drafter) and FIX0 + epoch_bump applied at a step
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (expect gg_gemv 2, aql_acq 2, ds41_kvfold / ds41_ofold / ds41_onefill 1, draft_capable 0); POST $FIX0 + epoch_bump + a 4-token request -> applied"; return 0; }
  local s0 s1
  s0=$(fn4get)
  fn4set "$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d["epoch_bump"]=1; print(json.dumps(d))' "$FIX0")"; nudge; s1=$(fn4get)
  log "RESULT xnv $1 switch note: at start $(echo "$s0" | cut -c1-400) | after the pin: $(echo "$s1" | cut -c1-400)"
  python3 -c 'import json,sys
s0, s1 = json.loads(sys.argv[1]), json.loads(sys.argv[2]); n = json.loads(sys.argv[3]); w = s0["switches"]
ok = w["gg_gemv"] == 2 and w["aql_acq"] == 2 and all(w[k] == 1 for k in ("ds41_kvfold", "ds41_ofold", "ds41_onefill")) and s0["draft_capable"] == 0
ok = ok and all(s1["switches"].get(k) == v for k, v in n.items()) and not s1["pending"] and s1["epoch"] >= 1
sys.exit(0 if ok else 1)' "$s0" "$s1" "$FIX0" 2>/dev/null
}
identw() {   # identw TAG: the warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xnv $1 warm-up answers vs $WARM_REF_225"; return 0; }
  [ -s $WARM_REF_225 ] || { log "RESULT xnv $1 warm-up answers: no reference $WARM_REF_225"; return 0; }
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
  [ "$DRY" = 1 ] || log "RESULT xnv $1 c1 short (stepprobe $3 $4): $(spline $2)"
}
copyrun() {   # copyrun TAG FILE CTAG: the 28k copy (bz d0's arguments) between two /metrics snapshots -> NVM + the copy's J line
  run "(cd $D && timeout 1500 python3 -I $NM $URL -- python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset 0 --tag=-$3 --outdir $O > $2 2>&1)"
  [ "$DRY" = 1 ] && return 0
  log "RESULT xnv $1 copy $3 (~31k prompt): $(python3 -c 'import json,sys
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
  log "RESULT xnv $tag: start arm $arm ($([ $arm = B ] && echo "RADIANCE_DS41_LATENT=nvfp4" || echo "latent bf16")), $kind$([ $kind = stamp ] && echo " (RADIANCE_AQL_STAMP=1: ~8 % inflated)")$([ $kind = prof ] && echo " (--profile-ops, PROFILE_EVERY 16: PROFILED, not a speed)") IMAGE=$IMAGE $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
  if [ $kind = plain ]; then
    gate $tag 30; rc=$?
    [ $rc = 1 ] && { keeplog $tag-fail; log "RESULT xnv $tag: VIRHE not healthy"; down; return 1; }
    [ $rc = 2 ] && log "RESULT xnv $tag: note -- slow after one restart (read its numbers with that in mind)"
  else
    restart $tag || { keeplog $tag-fail; log "RESULT xnv $tag: VIRHE not healthy"; down; return 1; }
    ms=$(check $tag 1)
    [ "$DRY" = 1 ] || log "RESULT xnv $tag warm-up + 1-user fastcheck ($kind, not gated): $ms ms/step"
  fi
  modes $tag $arm || { log "RESULT xnv $tag: VIRHE not the asked configuration -> start stopped"; down; return 1; }
  swnote $tag || { log "RESULT xnv $tag: VIRHE the switches are not as expected -> start stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag
  case $kind in
    plain)
      stopfile || c1short $tag $O/$P-c1s-r1.txt essay,code 512
      if ! stopfile; then pcreset $tag-cold; copyrun $tag $O/$P-c1L-cold.txt $P-cold; fi
      stopfile || copyrun $tag $O/$P-c1L-hit.txt $P-hit
      if ! stopfile; then
        run "(cd $D && timeout 1200 python3 mb.py --tag nv-$P --phases c8 --dur $C8DUR --warm 10 > $O/$P-c8.txt 2>&1)"
        [ "$DRY" = 1 ] || log "RESULT xnv $tag mb c8 $C8DUR s: $(grep -m1 '^MB ' $O/$P-c8.txt | cut -c4- | cut -c1-500)"
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
        [ "$DRY" = 1 ] || log "RESULT xnv $tag mb c8 30 s (PROFILED): $(grep -m1 '^MB ' $O/$P-c8.txt | cut -c4- | cut -c1-300)"
      fi
      run "docker logs $CN > $O/$P-engine.log 2>&1" ;;
  esac
  keeplog $tag
  down   # the stamp writer flushes at exit
  export STATSDIR=$O/stats AQLSTAMP= PROFEVERY=
  log "RESULT xnv $tag: done"
}
# ---------------------------------------------------------------- analysis
analyse_stamps() {   # the two stamp starts' windows (CPU only; started in the background once both stamp starts are done)
  local w wa wb f
  if [ "$DRY" = 1 ]; then
    echo "+ (background, once both stamp starts are done) nvlatprof.py stamps --a $O/stats-stamp-A/stamps.csv:T0:T1 --b $O/stats-stamp-B/stamps.csv:T0:T1 (short, long) -> $O/st-{short,long}.json"
    return 0
  fi
  for w in short long; do
    wa=$(awk -v w=$w '$1 == w {print $2 ":" $3}' $O/p2A-win.txt 2>/dev/null | tail -1); wb=$(awk -v w=$w '$1 == w {print $2 ":" $3}' $O/p2B-win.txt 2>/dev/null | tail -1)
    if [ -n "$wa" ] && [ -n "$wb" ] && [ -s $O/stats-stamp-A/stamps.csv ] && [ -s $O/stats-stamp-B/stamps.csv ]; then
      timeout 2400 nice -n 10 python3 -I $NP stamps --a $O/stats-stamp-A/stamps.csv:$wa --b $O/stats-stamp-B/stamps.csv:$wb --label st-$w --json $O/st-$w.json > $O/st-$w.txt 2>&1
      while IFS= read -r l; do log "RESULT xnv stamps $w ${l:0:1500}"; done < <(grep '^NVLP st-' $O/st-$w.txt | sed "s/^NVLP st-$w //")
    else log "RESULT xnv stamps $w: no windows or no stamp file (A '$wa', B '$wb')"; fi
  done
  for f in $O/stats-stamp-A/stamps.csv $O/stats-stamp-B/stamps.csv; do [ -e $f ] && gzip -f $f; done
  return 0
}
analyse_ops() {   # the two profiled starts' windows
  local w wa wb f
  if [ "$DRY" = 1 ]; then
    echo "+ nvlatprof.py ops --a $O/p3A-engine.log:L0:L1 --b $O/p3B-engine.log:L0:L1 (short, long, c8) -> $O/op-{short,long,c8}.json"
    return 0
  fi
  for w in short long c8; do
    wa=$(awk -v w=$w '$1 == w {print $2 ":" $3}' $O/p3A-win.txt 2>/dev/null | tail -1); wb=$(awk -v w=$w '$1 == w {print $2 ":" $3}' $O/p3B-win.txt 2>/dev/null | tail -1)
    if [ -n "$wa" ] && [ -n "$wb" ] && [ -s $O/p3A-engine.log ] && [ -s $O/p3B-engine.log ]; then
      timeout 1800 python3 -I $NP ops --a $O/p3A-engine.log:$wa --b $O/p3B-engine.log:$wb --label op-$w --json $O/op-$w.json > $O/op-$w.txt 2>&1
      while IFS= read -r l; do log "RESULT xnv ops $w ${l:0:1500}"; done < <(grep '^NVLP op-' $O/op-$w.txt | sed "s/^NVLP op-$w //")
    else log "RESULT xnv ops $w: no windows or no engine log (A '$wa', B '$wb')"; fi
  done
  for f in $O/p3A-engine.log $O/p3B-engine.log; do [ -e $f ] && gzip -f $f; done
  return 0
}
summary() {
  [ "$DRY" = 1 ] && { echo "+ nvlatprof.py summary $O -> RESULT xnv summary: clean c1 short / c1 ~31k / c8 / prefill A vs B, stamped critical path vs profiled device sum per window, kernels only in B, identity"; return 0; }
  timeout 600 python3 -I $NP summary $O --json $O/summary.json > $O/summary.txt 2>&1
  local s; s=$(grep '^NVLP summary ' $O/summary.txt | sed 's/^NVLP summary //' | tr '\n' ' ' | sed 's/  */ /g')
  log "RESULT xnv summary: ${s:0:5000}"
}
# ---------------------------------------------------------------- main
log "RESULT xnv $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) starts [$STARTS], c8 $C8DUR s, yaml $(basename $Y) | LEVERS NVFP4 latent: 1 user +10-12 % (15.81 -> 17.71 ms), c8 -4..-6 %, prefill -15.1 %; latb3 decode attention at 1 row 15.7 -> 66.5 us a call (135k)"
docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xnv: VIRHE image $IMAGE not on this host"; exit 1; }
[ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -qzF RADIANCE_DS41_LATENT /opt/radiance/share/radiance/architectures/deepseek41.so && grep -qzF latent_format /opt/radiance/share/radiance/kernels/libr4d.so && grep -qzF RADIANCE_DS41_KVFOLD /opt/radiance/share/radiance/architectures/deepseek41.so && grep -qzF rmsnorm_rope_roundtrip_kv_store /opt/radiance/share/radiance/kernels/libr4d.so && grep -qzF reset_prefix_cache /opt/radiance/bin/radiance && grep -qzF RADIANCE_AQL_STAMP /opt/radiance/bin/radiance && echo yes' 2>/dev/null)" = yes ] \
  || { log "RESULT xnv: VIRHE $IMAGE lacks the nvfp4 latent / K1 / reset_prefix_cache / AQL stamp strings"; exit 1; }
[ "$DRY" = 1 ] && echo "+ image check: $IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE | cut -c8-19)) carries RADIANCE_DS41_LATENT, latent_format, K1, reset_prefix_cache, RADIANCE_AQL_STAMP -- OK"
for v in RADIANCE_DS41_LATENT RADIANCE_AQL_STAMP RADIANCE_PROFILE_EVERY RADIANCE_DS41_GG_GEMV RADIANCE_AQL_ACQ RADIANCE_DS41_KVFOLD RADIANCE_DS41_OFOLD RADIANCE_DS41_ONEFILL RADIANCE_PLACE_TURNOVER; do
  grep -q "^ *$v:" $Y || { log "RESULT xnv: VIRHE $Y does not pass $v"; exit 1; }
done
if [ "$DRY" = 1 ]; then
  ( nvenv B; servevars prof; export ARGS="$(serve_args)" STATSDIR=$O/stats; docker compose --project-directory $C -f $Y config --quiet && echo "+ compose config ($(basename $Y), arm B, prof): OK" ) || echo "+ compose config: FAILED"
  for f in $Y $GS $NP $NM $HD/ds_opprof.py $D/stepprobe.py $D/mb.py $D/ds41_tests.py $MODELDIR/$M225 $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $WARM_REF_225; do
    [ -e $f ] && echo "+ present: $f" || echo "+ MISSING: $f"; done
  python3 -I $NP summary /nonexistent-nvlat > /dev/null 2>&1 && echo "+ nvlatprof.py runs" || echo "+ nvlatprof.py: FAILED"
  python3 -I $NM > /dev/null 2>&1; [ $? = 2 ] && echo "+ nvlatmet.py runs" || echo "+ nvlatmet.py: FAILED"
else
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xnv: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  for f in $GS $NP $NM $HD/ds_opprof.py $D/stepprobe.py $D/mb.py $D/ds41_tests.py; do [ -e $f ] || { log "RESULT xnv: VIRHE $f missing"; exit 1; }; done
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xnv: VIRHE $M225 missing or not verified"; exit 1; }
  for f in $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $WARM_REF_225; do [ -s $f ] || log "RESULT xnv: note -- reference $f missing (not compared)"; done
fi
mkd $O $O/guard $O/stats
[ "$DRY" = 1 ] || { : > $O/identity.txt; rm -f $O/p2A-win.txt $O/p2B-win.txt $O/p3A-win.txt $O/p3B-win.txt; }
evict
STPID=""; NSTAMP=$(echo "$STARTS" | tr ' ' '\n' | grep -c ':stamp$'); DSTAMP=0
for s in $STARTS; do
  stopfile && break
  onestart ${s%%:*} ${s##*:}
  case $s in *:stamp) DSTAMP=$((DSTAMP + 1)) ;; esac
  # both stamp starts done -> their analysis in the background (CPU only, nice 10) while the next starts run
  if [ -z "$STPID" ] && [ $NSTAMP -gt 0 ] && [ $DSTAMP = $NSTAMP ]; then
    if [ "$DRY" = 1 ]; then analyse_stamps; STPID=dry; else analyse_stamps > $O/analyse-stamps.out 2>&1 & STPID=$!; fi
  fi
done
[ -z "$STPID" ] && analyse_stamps
[ -n "$STPID" ] && [ "$STPID" != dry ] && wait $STPID
analyse_ops
summary
unset $SWITCHES
export AQLSTAMP= PROFEVERY=
down
log "RESULT xnv $ITEM: done -- $O"
