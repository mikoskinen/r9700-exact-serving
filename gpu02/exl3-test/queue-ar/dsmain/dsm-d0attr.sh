#!/bin/bash
# dsm-d0attr.sh (10.11.2026, queue-ar/dsmain, ds-main-port) -- NOT a queue item; the body of 57-p4zyzzzzz-ds-hczz-d0attr.sh.
# WHY A DEPTH-0 STEP OF AN NST 5 (DSpark, adaptive, draft_depth0 1) SERVER COSTS MORE THAN A PLAIN NST 0 STEP. hcy-depth0b
# (1b8848a2, block B = draft_depth0 1, gg_gemv 2): c8 53.2 ms a decode step, 9.76 tok a step, 146.9 tok/s (zero 1019 / probe
# 140 cycles) vs NST 0 gg_gemv 2 38.9 ms, 170.0 tok/s (gz-gemvacq gg2-gemv2); c4 118.9 vs 131.7; c1 essay 19.45 ms 57.8 tok/s
# (zero 404 / probe 52) vs ~16 ms 62.5. Candidates (hypotheses): (1) the block drafter's context pass still runs on every
# depth-0 step (sched/draft.h); (2) VRAM: the drafter holds ~217 routed-expert units a rank (start: vram 2731 / pinned 1205 vs
# NST 0 2948 / 892), so the trunk's expert hit rate falls; (3) the probes (2 drafting cycles every 16 at 0; a c8 drafting step
# ~111 ms); (4) verify-shaped step forms / buffers in NST 5 mode, or host work around the drafter's passes.
# THREE STARTS of the ds41-gg 1b8848a2 image (XDA_IMAGE), EXL3 2.25, TP4, MML 524288, depth0b's t3 env + serve line, DSpark
# --num-speculative-tokens 5 --draft-window adaptive, gg_gemv 2 at start; draft_adaptive 1, draft_depth0 1, draft_hybrid 0:
#   S1 CLEAN (gated 70 ms): the switch note (draft_probe 1 refused, 4096 / 16 applied at a step), then IN-PROCESS blocks over
#      draft_probe (d0attrblocks.py, XDA_ORDER default A B B A; A = 16 = today, B = 4096 = probes almost never, so a pure
#      depth-0 plain step; A first: the P16 block measures every row class's plain step, which a P4096 block cannot (its
#      plain probes from the drafting depth come at most once in 4096 cycles), and without a class's plain step depth 0
#      never fires there). Before every block draft_depth0 0 is applied at a step and then 1 with the block's draft_probe and
#      epoch_bump: depth 0 off ends any probe and returns D to the drafting depth, on again spends its hold, so the first
#      measured cycle of every block decides at once from the same state (sched/draft.cpp set_depth0 / plan_zero). Phases:
#      the 1-user essay probe between /stats snapshots (acceptance by draft position, as depth0b), mb c4 and mb c8 (75 s,
#      warm 10) with /stats + the draft counters snapshotted at the mb window's ends: decode steps at depth 0 / in probes,
#      routed units in VRAM, rank-0 hit %, missed units and host-stream MB a step. Identity under P4096: the 28k copy and
#      the 20-prompt greedy set at c1 vs bz's d0 (NST 0, stock), byte for byte.
#   S2 STAMPED (RADIANCE_AQL_STAMP=1: every played pass on card 0 timed per dispatch; not gated, not a speed): for gg_gemv 0
#      (the stock dense forms = the NST 0 references') and then 2 (the line): a P16 settle at c8 (30 s; depth 0 needs each
#      class's plain step and drafting yield measured at that form), then a P4096 c8 window (40 s) -> d0attrprof.py stamps: pass kinds, the step (trunk start
#      to trunk start), the trunk pass, the drafter's context pass, card 0's idle time a step, kernel families a pass; the
#      gg_gemv 0 window against dsm-225-gap's c8 stamps (NST 0, main e736c019, stock forms, the same t3 env and serve line,
#      RADIANCE_AQL_STAMP=1; runs/xdsm/stats-gap-225/stamps.csv.gz, window dsm-225-gap-win.txt) by the same reading.
#   S3 PROFILED (--profile-ops, RADIANCE_PROFILE_EVERY 16; synchronised steps, not a speed): the same settles + windows
#      -> d0attrprof.py ops: per rank the trunk's classes apart from the drafter's ops (DSpark's context pass at depth 0),
#      rank 0's drafter ops by name with calls a step; the gg_gemv 0 window against dsm-225-prof-c8 (NST 0, main e736c019,
#      stock forms, PROFILE_EVERY 16, c8 30 s; runs/xdsm/dsm-225-prof-c8-op.json) by class and by op.
#   SUMMARY: c8 / c4 / c1 ms a decode step, P4096 vs NST 0 (gz-gemvacq gg2-gemv2 B blocks, gg_gemv 2, no re-run) and P16 vs
#   P4096 (value, reference, %), depth-0 shares, expert counters vs NST 0 (bz dsd-d0 /stats per phase); the c8 attribution:
#   probe share, context-pass ms, trunk pass (expert kernels vs the rest) and card idle vs NST 0, the remainder.
# NO NST 0 re-run: dsm-225-gap (stamps) and dsm-225-prof (op profile) are NST 0 c8 readings with the same plugin ops as this
# item's gg_gemv 0 windows (k1prof's are c1 only, the ds41-k1 image, gg_gemv 0: not used). ~37 min: S1 ~21, S2 ~8, S3 ~8
# (XDA_STARTS="S1 S3" ~29 min).
# NO cc_sim, no 2.9. No power / host / fan settings; production compose files read only (--project-directory).
# DRYRUN=1 bash dsm-d0attr.sh : prints every command and runs the read-only preflight (image, files, references); writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-gg-tp4.yaml
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=$R/xdsm/d0attr; export STATSDIR=$O/stats
RX=$R/xdsm
RD=$RX/draft                                    # bz's d0 files (NST 0, stock): copy, greedy set, /stats per phase
RG=$RX/gg2                                      # gz-gemvacq's blocks (NST 0, gg_gemv 0 / 2)
REFOP=$RX/dsm-225-prof-c8-op.json               # NST 0 op profile, c8 (main e736c019, stock forms)
REFST=$RX/stats-gap-225/stamps.csv.gz           # NST 0 stamps (main e736c019, stock forms) ...
REFWIN=$RX/dsm-225-gap-win.txt                  # ... and its c4 / c8 windows (CLOCK_MONOTONIC ns)
REFPH=$RD/dsd-d0/stats-phases.txt               # NST 0 expert counters per phase
export IMAGE=${XDA_IMAGE:-radiance-local:ds41-gg-1b8848a2-5c13e555-l}
ITEM=${XDA_ITEM:-57-p4zyzzzzz-ds-hczz-d0attr}; ORDER=${XDA_ORDER:-ABBA}; DUR=${XDA_DUR:-75}; PA=${XDA_PA:-16}; PB=${XDA_PB:-4096}
SDUR=${XDA_SDUR:-30}; WDUR=${XDA_WDUR:-40}; STARTS=${XDA_STARTS:-S1 S2 S3}
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
DAB=$HD/d0attrblocks.py; DAP=$HD/d0attrprof.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
FIXED='{"gg_gemv": 2, "draft_adaptive": 1, "draft_hybrid": 0}'
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DRAFT_DEPTH0 RADIANCE_DRAFT_PROBE RADIANCE_DRAFT_HYBRID RADIANCE_DRAFT_HYBRID_MATCH"
t3env() {   # 105's t3 arm, every other switch unset; gg_gemv 2 from the start (= dsm-depth0b.sh's)
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2
}
d5env() {   # depth0b's start: the t3 env, best_vars, MNBT 4160, DSpark NST 5 --live --draft-window adaptive
  t3env; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive" AQLSTAMP= PROFEVERY= PROFSYNC=
}
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
gate() {   # gate TAG LIMIT: lib-29's gate + keeplog before the retry / after a second slow check (= dsm-depth0b.sh's)
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
  log "RESULT guard $tag: $st, 1-user ${ms} ms/step (limit $lim; depth0b dz-dspark: fast 35.32)"
  case $st in SLOW*) return 2 ;; esac
}
FOLLOW_PID=""; PARSE_PID=""
follow() { [ "$DRY" = 1 ] && { echo "+ docker logs -f $CN > $1/engine.log &"; return; }
  mkdir -p $1; docker logs -f $CN > $1/engine.log 2>&1 & FOLLOW_PID=$!; sleep 2; }
unfollow() { [ -n "$FOLLOW_PID" ] && { kill $FOLLOW_PID 2>/dev/null; wait $FOLLOW_PID 2>/dev/null; FOLLOW_PID=""; }; return 0; }
trap 'unfollow; down' EXIT
trap 'exit 143' TERM INT
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
fn4code() {   # fn4code JSON: POST, print the HTTP status
  [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null
}
thermline() { type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"; }
cmpfile() {   # cmpfile A B -> IDENTICAL (n chars) | DIFFERENT at char k (na vs nb chars) | not compared
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference (= dsm-depth0b.sh's)
  [ "$DRY" = 1 ] && { echo "+ RESULT xda $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xda $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  case $r in IDENTICAL*) log "RESULT xda $1 warm-up answers vs stock ($(basename $2)): $r" ;;
    *) log "RESULT xda $1 warm-up answers vs stock ($(basename $2)): VIRHE $r -- NOT EXACT vs the tuned3l line" ;; esac
}
modes() {   # modes TAG: the serve log's notes must be the t3 line's + the DSpark / adaptive lines -> 0, else 1 (= depth0b's + its draft checks)
  [ "$DRY" = 1 ] && { echo "+ modes $1: indexer note (mxfp4, latent bf16, compressor tail bound, router selection+sort separate, AR_PCN separate at 0 layers), ARGATHER note, staging line, no acquire-rule line; DSpark drafter + adaptive draft depth lines"; return 0; }
  local MLF SL ok=1
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  log "RESULT xda $1 modes: $(echo "$MLF" | cut -c1-700) | staging: ${SL:-none} | $(docker logs $CN 2>&1 | grep -m2 -E 'draft depth|DSpark drafter' | sed 's/^[A-Z] *//' | tr '\n' '|' | cut -c1-500)"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate" "post-collapse-norm separate at 0 layers" "next collapse-norm separate at 0 layers"; do
    case "$MLF" in *"$w"*) ;; *) log "RESULT xda $1: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  docker logs $CN 2>&1 | grep -q 'the MoE gather and its all-reduce as one kernel' || { log "RESULT xda $1: VIRHE no RADIANCE_DS41_ARGATHER note"; ok=0; }
  [ -n "$SL" ] || { log "RESULT xda $1: VIRHE no prefill staging line (levels refused?)"; ok=0; }
  docker logs $CN 2>&1 | grep -q 'the acquire rule plays' && { log "RESULT xda $1: VIRHE an acquire-rule line (RADIANCE_AQL_ACQ should be unset)"; ok=0; }
  docker logs $CN 2>&1 | grep -q "DSpark drafter" || { log "RESULT xda $1: VIRHE no DSpark drafter line"; ok=0; }
  docker logs $CN 2>&1 | grep -q "draft depth: adaptive" || { log "RESULT xda $1: VIRHE no adaptive draft depth line"; ok=0; }
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
vramunits() {   # vramunits FILE: the routed experts' VRAM / pinned units of a /stats snapshot
  python3 -c 'import json,sys
try: d = json.load(open(sys.argv[1]))
except Exception as e: print("n/a (%s)" % e); sys.exit(0)
t = {x.get("name"): x.get("units") for x in d.get("experts", {}).get("tiers", [])}
print("vram %s, pinned %s" % (t.get("vram"), t.get("pinned")))' "$1" 2>&1
}
swnote() {   # swnote TAG: at start gg_gemv 2, adaptive, draft_depth0 0, draft_probe 16, epoch 0, the draft counters reported;
             # draft_probe 1 refused (2 .. 4096); 4096 and back to 16 applied at a step
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (expect gg_gemv 2, draft_adaptive 1, draft_depth0 0, draft_probe 16, draft_capable 1, epoch 0, a draft counter block); POST draft_probe 1 -> 400; POST draft_probe 4096 / 16, each + a 4-token request -> applied"; return 0; }
  local s0 s1 s2 c1
  s0=$(fn4get); c1=$(fn4code '{"draft_probe": 1}')
  fn4set '{"draft_probe": 4096}'; nudge; s1=$(fn4get)
  fn4set '{"draft_probe": 16}'; nudge; s2=$(fn4get)
  log "RESULT xda $1 switch note: at start $(echo "$s0" | cut -c1-500) | draft_probe 1 -> HTTP $c1 | after 4096: $(echo "$s1" | cut -c1-200) | after 16: $(echo "$s2" | cut -c1-200)"
  python3 -c 'import json,sys
s0, s1, s2 = (json.loads(x) for x in sys.argv[1:4])
w = s0["switches"]
ok = w["gg_gemv"] == 2 and w["draft_adaptive"] == 1 and w["draft_depth0"] == 0 and w["draft_probe"] == 16 and w["draft_hybrid"] == 0 \
     and s0["draft_capable"] == 1 and s0["epoch"] == 0 and "zero_cycles" in s0.get("draft", {}) \
     and s1["switches"]["draft_probe"] == 4096 and not s1["pending"] and s2["switches"]["draft_probe"] == 16 and not s2["pending"] and s2["epoch"] == 2
sys.exit(0 if ok else 1)' "$s0" "$s1" "$s2" 2>/dev/null && [ "$c1" = 400 ]
}
copyt() {   # copyt TAG OFFSET: the copy test after a ~31k-token cold prompt (= dsm-depth0b.sh's)
  local t0=$(date +%s)
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset $2 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xda $1 copy after a cold prompt: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300) | $(thermline $t0 $(date +%s))"
}
gsetc1() { run "(cd $D && timeout 900 python3 $GS $URL $O/$1-gs-c1.jsonl --conc 1 --max-tokens 192 > $O/$1-gs-c1.txt 2>&1)"; }
gcmp() { [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }; [ -s "$1" ] && [ -s "$2" ] || { echo "not compared (missing $1 or $2)"; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
winget() {   # winget FILE LABEL KEY1 KEY2: two fields of a window row (d0attrblocks.py window --out), "a:b"
  python3 -c 'import json,sys
for l in open(sys.argv[1]):
    try: d = json.loads(l)
    except ValueError: continue
    if d.get("label") == sys.argv[2]: print("%s:%s" % (d.get(sys.argv[3]), d.get(sys.argv[4])))' "$1" "$2" "$3" "$4" 2>/dev/null | tail -1
}
winlog() {   # winlog TAG FILE LABEL: one window's line
  [ "$DRY" = 1 ] && return 0
  log "RESULT xda $1 window $3: $(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    try: d = json.loads(l)
    except ValueError: continue
    if d.get("label") != sys.argv[2]: continue
    m, w = d.get("mb") or {}, d.get("win") or {}; z = w.get("draft") or {}
    print("ok %s, switches %s | mb %s tok/s, %s ms a decode step, %s tok a step | in the window: %s decode steps, depth 0 %s probe %s moves %s (share %s / %s), %s ms a decode step (/stats), vram %s units, hit %s %%, missed %s / stream %s MB a step, passes a step %s" % (
        d.get("ok"), {k: d.get("applied", {}).get(k) for k in ("gg_gemv", "draft_depth0", "draft_probe", "draft_adaptive", "draft_hybrid")},
        m.get("agg_dec_tps"), m.get("ms_decode_step"), m.get("tok_per_decode_step"), w.get("decode_steps"), z.get("zero_cycles"),
        z.get("probe_cycles"), z.get("zero_moves"), w.get("zero_share"), w.get("probe_share"), w.get("ms_decode_step"),
        w.get("vram_units_end"), w.get("hit_pct_rank0"), w.get("missed_step_rank0"), w.get("stream_MB_step"), w.get("passes_step")))' $2 $3 2>&1 | tail -1 | cut -c1-1000)"
}
# ---------------------------------------------------------------- S1: the clean start (blocks + identity)
s1() {
  local tag=da-clean P=$O/da-clean rc
  d5env; export STATSDIR=$O/stats
  log "RESULT xda $tag: start IMAGE=$IMAGE MODEL=$MODEL $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
  mkd $P
  gate $tag 70; rc=$?
  [ $rc = 1 ] && { log "RESULT xda $tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  [ $rc = 2 ] && { log "RESULT xda $tag: VIRHE over the gate limit after one restart -> start skipped"; down; return 1; }
  modes $tag || { log "RESULT xda $tag: VIRHE the modes are not the t3 / DSpark adaptive line's -> start stopped"; down; return 1; }
  if [ "$DRY" != 1 ]; then
    curl -s -m 15 $URL/stats > $O/$tag-stats0.json 2>/dev/null
    log "RESULT xda $tag experts at start: $(vramunits $O/$tag-stats0.json) | bz d0 (NST 0): $(vramunits $RD/dsd-d0/stats-start.json) | depth0b dz-dspark: $(vramunits $RX/depth0b/dz-dspark-stats0.json)"
  fi
  swnote $tag || { log "RESULT xda $tag: VIRHE the draft_probe switch is not as expected -> start stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  follow $P
  run "timeout 5400 python3 -I $DAB blocks --url $URL --mb-dir $D --label $tag --a $PA --b $PB --order $ORDER --dur $DUR --warm 10 --gen 512 --fixed '$FIXED' --stop $R/sweep.stop --out $O/$tag-blocks.jsonl > $O/$tag-blocks.txt 2>&1"
  if [ "$DRY" != 1 ]; then
    while IFS= read -r l; do log "RESULT xda $tag ${l:0:1500}"; done < <(grep '^BLOCK ' $O/$tag-blocks.txt)
    log "RESULT xda $tag blocks (P$PB vs P$PA pooled; pairs; t95): $(grep '^D0AB ' $O/$tag-blocks.txt | cut -c6- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v: "-" if v is None else format(v, ".3g")
out = []
for k, v in d["metrics"].items():
    p = v.get("pooled")
    out.append("%s %s -> %s (%s; pairs %s; t95 +- %s %%)" % (k, f(v.get("A")), f(v.get("B")), "-" if p is None else "%+.2f %%" % (100 * (p - 1)),
               " ".join("%+.1f" % (100 * (x - 1)) for x in v.get("pairs", [])), "-" if v.get("ci95") is None else "%.2f" % (100 * v["ci95"])))
print("; ".join(out) + " | %d blocks, %d flagged, %s s" % (d["blocks"], d["blocks_bad"], d["wall_s"]))' 2>&1 | cut -c1-3600)"
  fi
  if ! stopfile; then
    fn4set '{"draft_depth0": 0}'; nudge
    fn4set "{\"draft_depth0\": 1, \"draft_probe\": $PB, \"gg_gemv\": 2, \"draft_adaptive\": 1, \"draft_hybrid\": 0, \"epoch_bump\": 1}"
    copyt $tag-i1 0; gsetc1 $tag
    [ "$DRY" = 1 ] || log "RESULT xda $tag identity ran under: $(fn4get | cut -c1-500)"
    if [ "$DRY" = 1 ]; then echo "+ identity: 28k copy vs bz d0, greedy set c1 vs bz d0"
    else
      local r
      r=$(cmpfile $O/copy-$tag-i1-run1.txt $RD/copy-dsd-d0-i1-run1.txt)
      case $r in IDENTICAL*) log "RESULT xda $tag identity (P$PB, draft_depth0 1, gg_gemv 2): 28k copy vs bz d0 (NST 0, stock): $r" ;;
        *) log "RESULT xda $tag identity (P$PB, draft_depth0 1, gg_gemv 2): DIFFERENT 28k copy vs bz d0 (NST 0, stock): $r -- reported, not hidden" ;; esac
      r=$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$tag-gs-c1.jsonl)
      case $r in IDENTICAL*) log "RESULT xda $tag identity (P$PB, draft_depth0 1, gg_gemv 2): greedy set c1 vs bz d0: $r" ;;
        *) log "RESULT xda $tag identity (P$PB, draft_depth0 1, gg_gemv 2): DIFFERENT greedy set c1 vs bz d0: $r -- reported, not hidden" ;; esac
    fi
  fi
  unfollow
  run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
  down
}
# ---------------------------------------------------------------- S2 / S3: instrumented starts, P16 settle + P4096 c8 windows
instr() {   # instr KIND (stamp|prof): restart (not gated), warm-up, settle, windows g0 / g2
  local kind=$1 tag=da-$1 W=$O/$1-win.jsonl mk="" ms
  d5env
  case $kind in
    stamp) export AQLSTAMP=1 STATSDIR=$O/stats-stamp ;;
    prof)  export PROFEVERY=16 XARGS="$XARGS --profile-ops" STATSDIR=$O/stats-prof; mk="wc -l < $O/prof/engine.log" ;;
  esac
  mkd $STATSDIR
  run "find $STATSDIR -maxdepth 1 -name 'stamps.csv*' -delete; rm -f $W"
  log "RESULT xda $tag: start ($kind: $([ $kind = stamp ] && echo 'RADIANCE_AQL_STAMP=1, every played pass on card 0' || echo '--profile-ops RADIANCE_PROFILE_EVERY=16, synchronised steps'); NOT a speed) IMAGE=$IMAGE $(desc)"
  restart $tag || { keeplog $tag-fail; log "RESULT xda $tag: VIRHE not healthy -- no $kind reading"; down; export AQLSTAMP= PROFEVERY= STATSDIR=$O/stats; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] || log "RESULT xda $tag warm-up + 1-user fastcheck ($kind, not gated): $ms ms/step"
  modes $tag || { log "RESULT xda $tag: VIRHE modes -> no $kind reading"; down; export AQLSTAMP= PROFEVERY= STATSDIR=$O/stats; return 1; }
  [ $kind = prof ] && follow $O/prof
  # per gg_gemv (0 = the NST 0 references' forms, then 2 = the line): a P16 settle at c8 under that gg_gemv (depth 0 re-armed),
  # so each row class's plain step and drafting yield are measured at that form and depth 0 can fire; then the P4096 window
  # (no probes: D stays where the settle's measurements put it, which the window's counters show)
  local gw
  for gw in 0 2; do
    stopfile && break
    run "timeout 900 python3 -I $DAB window --url $URL --mb-dir $D --label $tag-s$gw --phase c8 --dur $SDUR --warm 10 --reset --set '{\"draft_depth0\": 1, \"draft_probe\": $PA, \"gg_gemv\": $gw, \"draft_adaptive\": 1, \"draft_hybrid\": 0, \"epoch_bump\": 1}' --out $W > /dev/null 2>&1"
    winlog $tag $W $tag-s$gw
    stopfile && break
    run "timeout 900 python3 -I $DAB window --url $URL --mb-dir $D --label $tag-g$gw --phase c8 --dur $WDUR --warm 10 --set '{\"draft_probe\": $PB}' ${mk:+--mark-cmd '$mk'} --out $W > /dev/null 2>&1"
    winlog $tag $W $tag-g$gw
  done
  run "sleep 2"
  if [ $kind = prof ]; then unfollow; else keeplog $tag; fi   # the profiled start's log is the followed one
  down   # the stamp writer appends as it goes (k1prof's P2 the same)
  local g0 g2 rf
  if [ $kind = stamp ]; then
    if [ "$DRY" = 1 ]; then echo "+ (background) nice python3 -I $DAP stamps $STATSDIR/stamps.csv --win g0:T0:T1 --win g2:T0:T1 --ref nst0:$REFST:<dsm-225-gap c8 window> --json $O/stamps.json; gzip stamps.csv"
    else
      g0=$(winget $W $tag-g0 t0 t1); g2=$(winget $W $tag-g2 t0 t1); rf=$(awk '$1 == "c8" {print $2 ":" $3}' $REFWIN 2>/dev/null)
      if [ -s $STATSDIR/stamps.csv ] && [ -n "$g0" ] && [ -n "$g2" ] && [ -n "$rf" ]; then
        ( nice -n 10 timeout 2400 python3 -I $DAP stamps $STATSDIR/stamps.csv --win g0:$g0 --win g2:$g2 --ref nst0:$REFST:$rf \
            --json $O/stamps.json --label s2 > $O/stamps.txt 2>&1; nice -n 10 gzip -f $STATSDIR/stamps.csv ) &
        PARSE_PID=$!
        log "RESULT xda $tag: stamps parsed in the background (pid $PARSE_PID): windows g0 $g0, g2 $g2, ref nst0 $rf"
      else log "RESULT xda $tag: VIRHE no stamps / windows ($(ls -la $STATSDIR 2>&1 | tail -2 | tr '\n' ' '), g0 '$g0' g2 '$g2' ref '$rf')"; fi
    fi
  else
    if [ "$DRY" = 1 ]; then echo "+ python3 -I $DAP ops $O/prof/engine.log --win g0:L0:L1 --win g2:L0:L1 --ref-json $REFOP --json $O/ops.json; gzip the log"
    else
      g0=$(winget $W $tag-g0 mark0 mark1); g2=$(winget $W $tag-g2 mark0 mark1)
      if [ -n "$g0" ] && [ -n "$g2" ]; then
        timeout 1200 python3 -I $DAP ops $O/prof/engine.log --win g0:$g0 --win g2:$g2 --ref-json $REFOP --json $O/ops.json --label s3 > $O/ops.txt 2>&1
        while IFS= read -r l; do log "RESULT xda $tag ${l:0:1600}"; done < <(grep '^D0AP s3 ops' $O/ops.txt | sed 's/^D0AP s3 ops //')
      else log "RESULT xda $tag: VIRHE no profile windows (g0 '$g0' g2 '$g2')"; fi
      run "gzip -f $O/prof/engine.log"
    fi
  fi
  export AQLSTAMP= PROFEVERY= STATSDIR=$O/stats
}
summary() {
  if [ -n "$PARSE_PID" ]; then wait $PARSE_PID 2>/dev/null; PARSE_PID=""; fi
  if [ "$DRY" != 1 ] && [ -s $O/stamps.txt ]; then
    while IFS= read -r l; do log "RESULT xda da-stamp ${l:0:1600}"; done < <(grep '^D0AP s2 stamps' $O/stamps.txt | sed 's/^D0AP s2 stamps //')
  fi
  [ "$DRY" = 1 ] && { echo "+ python3 -I $DAP summary --blocks $O/da-clean-blocks.jsonl --nst0 $RG/gg2-gemv2-blocks.jsonl --nst0-stats $REFPH --stamps $O/stamps.json --ops $O/ops.json --json $O/summary.json"; echo "+ RESULT xda summary: c8 / c4 / c1 ms a decode step P$PB vs NST 0 and P$PA vs P$PB, depth-0 shares, experts; c8 attribution"; return 0; }
  timeout 600 python3 -I $DAP summary --blocks $O/da-clean-blocks.jsonl --nst0 $RG/gg2-gemv2-blocks.jsonl --nst0-stats $REFPH \
    --stamps $O/stamps.json --ops $O/ops.json --json $O/summary.json --label sum > $O/summary.txt 2>&1
  log "RESULT xda summary: $(grep '^D0AP sum summary' $O/summary.txt | sed 's/^D0AP sum summary //' | tr '\n' '#' | sed 's/#$//; s/#/ || /g' | cut -c1-5000)"
}
# ---------------------------------------------------------------- main
log "RESULT xda $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) starts [$STARTS] draft_probe A $PA / B $PB order $ORDER mb $DUR s, windows settle $SDUR s + $WDUR s | refs gz-gemvacq gg2-gemv2 (NST 0 gg_gemv 2 blocks), bz d0 (copy, greedy set, /stats per phase), dsm-225-gap c8 stamps + dsm-225-prof c8 op profile (NST 0, stock forms)"
PRE=1
docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xda: $([ "$DRY" = 1 ] && echo 'note (dry)' || echo VIRHE) a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; [ "$DRY" = 1 ] || exit 1; }
docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xda: VIRHE image $IMAGE not on gpu02"; PRE=0; }
[ -e $Y ] && grep -q '^ *RADIANCE_DS41_GG_GEMV:' $Y && grep -q 'RADIANCE_AQL_STAMP: ' $Y && grep -q 'RADIANCE_PROFILE_EVERY: ' $Y \
  || { log "RESULT xda: VIRHE $Y missing or does not pass RADIANCE_DS41_GG_GEMV / RADIANCE_AQL_STAMP / RADIANCE_PROFILE_EVERY"; PRE=0; }
for f in $GS $DAB $DAP $HD/ds_opprof_draft.py $D/stepprobe.py $D/mb.py $D/ds41_tests.py $D/fastcheck.py; do [ -e $f ] || { log "RESULT xda: VIRHE $f missing"; PRE=0; }; done
for f in $DAB $DAP; do python3 -c 'import ast,sys; ast.parse(open(sys.argv[1]).read())' $f 2>/dev/null || { log "RESULT xda: VIRHE $f does not parse"; PRE=0; }; done
[ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xda: VIRHE $M225 missing or not verified"; PRE=0; }
if docker image inspect $IMAGE > /dev/null 2>&1; then   # read-only: a no-network, no-device container greps the binary
  [ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -q "draft depth 0: a step drafts nothing" /opt/radiance/bin/radiance && grep -q "draft_probe takes 2 .. 4096" /opt/radiance/bin/radiance && grep -q "RADIANCE_AQL_STAMP" /opt/radiance/bin/radiance && grep -q r4d_gg_mr_pin /opt/radiance/share/radiance/kernels/libr4d.so && echo yes' 2>/dev/null)" = yes ] \
    || { log "RESULT xda: VIRHE $IMAGE lacks depth 0 / draft_probe / the stamps / lever D2"; PRE=0; }
fi
for f in $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl $RD/dsd-d0/stats-start.json $REFPH $RG/gg2-gemv2-blocks.jsonl $REFOP $REFST $REFWIN $WARM_REF_225; do
  [ -s $f ] || log "RESULT xda: note -- reference $f missing (not compared)"; done
[ $PRE = 1 ] || { log "RESULT xda $ITEM: VIRHE preflight failed -- nothing run"; exit 1; }
[ "$DRY" = 1 ] && echo "+ preflight OK (image $(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19), files, references)"
mkd $O $O/guard $O/stats
evict
for st in $STARTS; do
  stopfile && break
  case $st in S1) s1 ;; S2) instr stamp ;; S3) instr prof ;; *) log "RESULT xda: note -- unknown start $st" ;; esac
done
summary
unset $SWITCHES
export AQLSTAMP= PROFEVERY=
down
log "RESULT xda $ITEM: done -- $O"
