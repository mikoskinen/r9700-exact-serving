#!/bin/bash
# dsm-epbal.sh (10.11.2026, queue-ar/dsmain, ds41-slot agent, levers D3c / F1 = D3s stage 2) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-q-epbal.sh. Expert-parallel BALANCING on the slot-order MoE sum (lever D3s, RADIANCE_DS41_SLOTSUM=48, whose
# bytes do not depend on the placement -- item p's acceptance): placements measured IN ONE PROCESS (ds41_place 0 / 1 / 2) and
# against today's line (ds41_slotsum 0).
#   F1  epplace.py replicate --hot H: a layer's H hottest experts (decode counts) on every rank; each step the kernel serves a
#       replicated expert from its least-loaded copy (moe_scatter_placed: single-copy loads first, then the replicated ones in
#       ascending id to the least-loaded copy by cost units, the same choice on every rank). Model (balrows.py): slowest-rank MoE
#       + wait at 1 / 6 / 8 / 24 rows 3.51 / 9.26 / 11.01 / 20.98 ms -> 3.01 / 7.52 / 9.06 / 18.10 at hot-32, no misses.
#   D3c epplace.py solve: per layer, the expected rank load (P(live in a step of R rows) x (B, + M missed)) balanced by moving
#       experts off the heaviest rank within an extras budget (a moved expert's decode copy lives on another rank; its home copy
#       stays for prefill), the dispatch costs from each copy's expected miss rate. Rounds: round 1 from the decode counts
#       (item p's start 1 --expert-stats, else 6.11. cc-x117), measured (blocks + the miss log's per-(step, layer) rank loads);
#       round 2 re-solved at a wider decode band with a replicated head, measured against round 1's best.
#   VRAM honesty: every copy off home is an extra expert unit on its rank (planner / heat engine: VRAM or the pinned host pool);
#   the host pool grows by the extras' slots, and the VRAM the replicas take is the hit-rate trade the blocks measure (each
#   block's missed experts a step from RADIANCE_DEBUG_MISS_LOG).
# DeepSeek-V4.1 Flash EXL3 2.25, TP4, MML 524288, NST 0, the t3 env and serve line (as dsm-slotsum.sh), ARGATHER on.
# ORDER: per start: gate -> modes (the D3s note: tables, extras) -> switch note -> slow-state probe -> BLOCKS (ggblocks3.py,
#   A B B A A B, mb 75 s: stepprobe c1, mb c4, mb c8 each block; per-stream p50 beside the aggregate) -> per block the measured
#   skew (epplace.py measure over the block's steps of the miss log: live experts a layer, max - mean over ranks, missed a step)
#   START a (round 1; tables 0 contiguous, 1 F1 hot-XEB_HOT, 2 D3c round 1): b1 place 0 vs 1, b2 place 0 vs 2 (slot-order sum
#     in every block) -> BEST1 = the table with the larger pooled c8 aggregate gain (else c4; 0 if neither gains)
#   START b (round 2; tables 0, 1 = BEST1's file, 2 = D3c round 2): b3 place 1 vs 2 -> BEST = the better -> b4 THE LINE: today
#     (ds41_slotsum 0, place 0) vs the best (slotsum 1, place BEST) -> PREFILL (after decode) today vs best: fresh 16k A B B A,
#     136k A B, mb pf A B (prefill runs today's path in both; the extras' VRAM is what can move it).
# References (no re-runs): bz's dsd-d0 (main image, NST 0): probe 59.38 / 62.04 tok/s, c4 122.7 (p50 31.6), c8 150.1 (p50 19.7);
#   item p's cost blocks (today vs the slot-order sum, contiguous) for the lever's own cost.
# NO cc_sim, no BetterBench (user 10.11.). No power / host / fan settings; production compose files read only.
# DRYRUN=1 bash dsm-epbal.sh : prints every command, runs and writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-slot-tp4.yaml
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=$R/xdsm/epbal; export STATSDIR=$O/stats
export IMAGE=${XEB_IMAGE:-radiance-local:ds41-slot-4cf3501f-5c13e555-l}
ITEM=${XEB_ITEM:-57-p4zyzzzzz-ds-q-epbal}; ORDER=${XEB_ORDER:-ABBAAB}; DUR=${XEB_DUR:-75}
SLROWS=48; HOT=${XEB_HOT:-8}; ROWS1=${XEB_ROWS:-8}; BUDGET1=${XEB_BUDGET:-12}; MUS=${XEB_M:-250}
ROWS2=${XEB_ROWS2:-16}; BUDGET2=${XEB_BUDGET2:-24}; HOT2=${XEB_HOT2:-4}
GGB=$HD/ggblocks3.py; EPP=$HD/epplace.py
STATS=$R/xdsm/slotsum/stats/xs-cost.json; [ -s $STATS ] || STATS=$HD/cc-x117.json
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DS41_ARGATHER_MAX RADIANCE_DS41_SHEXP_LANE RADIANCE_DEBUG_MISS_LOG RADIANCE_DS41_SLOTSUM
  RADIANCE_DS41_SLOTSUM_ON RADIANCE_DS41_EP_PLACE RADIANCE_DS41_EP_PLACE2 RADIANCE_DS41_EP_PLACE_TAB"
t3env() {   # t3env FILE1 FILE2 MISSLOG: 105's t3 arm + ARGATHER on + the slot path (on at start, table 0) over the two files
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0 \
         RADIANCE_DS41_SLOTSUM=$SLROWS RADIANCE_DS41_SLOTSUM_ON=1 RADIANCE_DS41_EP_PLACE_TAB=0 \
         RADIANCE_DS41_EP_PLACE=$1 RADIANCE_DS41_EP_PLACE2=$2 RADIANCE_DEBUG_MISS_LOG=$3
}
modelvars225() {   # -> MODEL, HP29 ($1), LIM, C8REF; lib-29 best_vars (NST 0) + MNBT 4160 + --live
  export MODEL=$M225; HP29=$1; LIM=23; C8REF=43.1
  best_vars; export MNBT=4160 XARGS="$XARGS --live"
}
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
gate() {
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
FOLLOW_PID=""
follow() { [ "$DRY" = 1 ] && { echo "+ docker logs -f $CN > $1/engine.log &"; return; }
  docker logs -f $CN > $1/engine.log 2>&1 & FOLLOW_PID=$!; sleep 2; }
unfollow() { [ -n "$FOLLOW_PID" ] && { kill $FOLLOW_PID 2>/dev/null; wait $FOLLOW_PID 2>/dev/null; FOLLOW_PID=""; }; return 0; }
trap 'unfollow; down' EXIT
trap 'exit 143' TERM INT
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
fn4code() {
  [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null
}
FIX='"gg_gemv": 0, "aql_acq": 0, "ds41_shexp_lane": 0, "ds41_argather": 4'
arm() { fn4set "{\"ds41_slotsum\": $1, \"ds41_place\": $2, $FIX, \"epoch_bump\": 1}"; nudge; }
modes() {   # modes TAG: the D3s note with three tables, every layer resolved
  [ "$DRY" = 1 ] && { echo "+ modes $1: the D3s note (up to $SLROWS rows, 40 of 40 trunk layers, tables 0 / 1 / 2), no unresolved layer, staging line"; return 0; }
  local SN SL ok=1
  SN=$(docker logs $CN 2>&1 | grep -m1 'the slot-order MoE sum (lever D3s)' | sed 's/^radiance: deepseek41: //' | cut -c1-900)
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-200)
  log "RESULT xeb $1 modes: D3s: ${SN:-none} | staging: ${SL:-none} | $(docker logs $CN 2>&1 | grep -m1 -E 'heat engine: [0-9]+ movable units' | cut -c1-120)"
  case "$SN" in *"up to $SLROWS rows: 40 of 40 trunk layers"*"tables: 0 contiguous, 1 "*", 2 "*) ;; *) log "RESULT xeb $1: VIRHE the D3s note is not as asked: '$SN'"; ok=0 ;; esac
  [ "$(docker logs $CN 2>&1 | grep -c 'did not resolve; the layer keeps')" = 0 ] || { log "RESULT xeb $1: VIRHE a layer's slot path did not resolve"; ok=0; }
  [ -n "$SL" ] || { log "RESULT xeb $1: VIRHE no prefill staging line"; ok=0; }
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
swnote() {   # swnote TAG: start slotsum 1 / place 0; ds41_place 3 refused; 2 then 0 applied
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (ds41_slotsum 1, ds41_place 0, epoch 0); ds41_place 3 -> 400; place 2, then 0 -> applied"; return 0; }
  local s0 s1 s2 c3
  s0=$(fn4get); c3=$(fn4code '{"ds41_place": 3}')
  fn4set '{"ds41_place": 2}'; nudge; s1=$(fn4get)
  fn4set '{"ds41_place": 0}'; nudge; s2=$(fn4get)
  log "RESULT xeb $1 switch note: at start $s0 | ds41_place 3 -> HTTP $c3 | after 2: $s1 | after 0: $s2"
  python3 -c 'import json,sys
s0, s1, s2 = (json.loads(x) for x in sys.argv[1:4]); w = lambda s, k: s["switches"].get(k)
sys.exit(0 if w(s0, "ds41_slotsum") == 1 and w(s0, "ds41_place") == 0 and w(s1, "ds41_place") == 2 and w(s2, "ds41_place") == 0 and not s2["pending"] else 1)' "$s0" "$s1" "$s2" 2>/dev/null && [ "$c3" = 400 ]
}
mbms() { grep '^MB ' $1 2>/dev/null | head -1 | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()[3:]).get("ms_decode_step", ""))' 2>/dev/null; }
stprobe() {
  [ "$DRY" = 1 ] && { echo "+ mb c8 45 s state probe -> $O/$1-stprobe$2.txt" >&2; echo "fast (dry)"; return 0; }
  run "(cd $D && timeout 400 python3 mb.py --tag xeb-$1-st$2 --phases c8 --dur 45 --warm 10 > $O/$1-stprobe$2.txt 2>&1)"
  local ms; ms=$(mbms $O/$1-stprobe$2.txt)
  [ -n "$ms" ] || { echo "unknown (no mb line)"; return 0; }
  awk -v m=$ms -v r=$C8REF 'BEGIN{s=(m <= 1.07*r) ? "fast" : (m >= 1.14*r) ? "slow" : "between"; printf "%s %s ms (105 tuned3l c8 %s ms, x%.3f)\n", s, m, r, m/r}'
}
pfd() {
  local t0=$(date +%s)
  run "(cd $D && timeout 1800 python3 ds41_tests.py prefill --depths $2 --cpt 3.95 --calibrate 0 --reps 1 > $O/$1-pf$2.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xeb $1 fresh prefill $2: $(grep '^J ' $O/$1-pf$2.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); print("%s tok, cached %s, TTFT %s s, %s tok/s, %s" % (d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("prefill_tps"), d.get("status")))' 2>&1 | cut -c1-200)$(type thermwin > /dev/null 2>&1 && echo " | $(thermwin $t0 $(date +%s) 2>/dev/null | cut -c1-200)")"
}
mbpf() {
  local t0=$(date +%s)
  run "(cd $D && timeout 1500 python3 mb.py --tag xeb-$1 --phases pf > $O/$1-pf.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xeb $1 mb pf: $(grep '^MB ' $O/$1-pf.txt 2>/dev/null | cut -c4- | cut -c1-500)$(type thermwin > /dev/null 2>&1 && echo " | $(thermwin $t0 $(date +%s) 2>/dev/null | cut -c1-200)")"
}
blocks() {   # blocks TAG ARM_A ARM_B MISSLOG: in-process blocks + each block's measured skew from the miss log
  local tag=$1 aa=$2 bb=$3 ml=$4
  run "timeout 7200 python3 -I $GGB --url $URL --mb-dir $D --label $tag --arm-a '$aa' --arm-b '$bb' --fixed '{$FIX}' --miss-log $ml --order $ORDER --dur $DUR --warm 10 --out $O/$tag-blocks.jsonl > $O/$tag-blocks.txt 2>&1"
  [ "$DRY" = 1 ] && { echo "+ per block: epplace.py measure --miss $ml --from-step S0 --to-step S1"; return 0; }
  while IFS= read -r l; do log "RESULT xeb $tag ${l:0:400}"; done < <(grep '^BLOCK ' $O/$tag-blocks.txt)
  log "RESULT xeb $tag summary (B = $bb vs A = $aa, $FIX pinned, in process, order $ORDER, mb $DUR s): $(grep '^GGB ' $O/$tag-blocks.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=2: "-" if v is None else format(v, ".%df" % n)
out = []
for k, v in d["metrics"].items():
    p = v.get("pooled")
    out.append("%s %s -> %s (%s %%; pairs %s; t95 +- %s %%)" % (k, f(v.get("A")), f(v.get("B")), "-" if p is None else "%+.2f" % (100 * (p - 1)),
               " ".join("%+.2f" % (100 * (x - 1)) for x in v.get("pairs", [])), "-" if v.get("ci95") is None else "%.2f" % (100 * v["ci95"])))
print("; ".join(out) + " | %d blocks, %d flagged, %s s" % (d["blocks"], d["blocks_bad"], d["wall_s"]))' 2>&1 | cut -c1-2400)"
  python3 - $O/$tag-blocks.jsonl $ml $EPP > $O/$tag-skew.txt 2>&1 <<'PY'
import json, subprocess, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
for r in rows:
    s0, s1 = (r.get("steps") or [None, None])
    if s0 is None or s1 is None or s1 <= s0:
        print("SKEW block %d %s no steps" % (r["block"], r["arm"])); continue
    p = subprocess.run([sys.executable, "-I", sys.argv[3], "measure", "--miss", sys.argv[2], "--from-step", str(s0 + 1),
                        "--to-step", str(s1)], capture_output=True, text=True)
    m = next((l for l in p.stdout.splitlines() if l.startswith("EPMEASURE ")), "EPMEASURE {}")
    print("SKEW block %d %s %s %s" % (r["block"], r["arm"], json.dumps(r["value"]), m[10:]))
PY
  while IFS= read -r l; do log "RESULT xeb $tag measured ${l:0:400}"; done < <(grep '^SKEW ' $O/$tag-skew.txt)
}
pooled() {   # pooled TAG METRIC -> the pooled B/A of a blocks run ("" when missing)
  grep '^GGB ' $O/$1-blocks.txt 2>/dev/null | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read()); v = d["metrics"].get(sys.argv[1], {}).get("pooled")
print("" if v is None else "%.5f" % v)' "$2" 2>/dev/null
}
better() {   # better TAG: B (1) when its pooled c8 aggregate (else c4) gain is > 0, else A (0)
  local g; g=$(pooled $1 "c8 aggregate tok/s"); [ -n "$g" ] || g=$(pooled $1 "c4 aggregate tok/s")
  awk -v g="${g:-0}" 'BEGIN{print (g > 1.0) ? 1 : 0}'
}
startup() {   # startup TAG FILE1 FILE2: host pool for the extras, start, modes, switches, state
  local tag=$1 f1=$2 f2=$3 xm hp st rc
  run "python3 -I $EPP extras --place $f1 --place $f2 > $O/$tag-extras.txt 2>&1"
  if [ "$DRY" = 1 ]; then xm=900; else xm=$(grep -m1 '^EPEXTRAS ' $O/$tag-extras.txt | cut -c10- | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["extras_max"])' 2>/dev/null || echo 0); fi
  hp=$(( 19456 + (xm * 87 + 9) / 10 + 1024 ))
  t3env /stats/place/$(basename $f1) /stats/place/$(basename $f2) /stats/miss-$tag.log; modelvars225 $hp
  log "RESULT xeb $tag: start IMAGE=$IMAGE MODEL=$MODEL NST=$NST $(desc) extras a rank max $xm -> host pool $hp MiB env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
  mkd $O/$tag
  gate $tag $LIM; rc=$?
  [ $rc = 1 ] && { log "RESULT xeb $tag: VIRHE not healthy; $(grep -E '^E |deepseek41: ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; return 1; }
  [ $rc = 2 ] && { log "RESULT xeb $tag: VIRHE slow state after one restart"; return 1; }
  modes $tag || return 1
  swnote $tag || { keeplog $tag-switch; log "RESULT xeb $tag: VIRHE the switches are not as expected"; return 1; }
  st=$(stprobe $tag 1)
  log "RESULT xeb $tag state: $st$(case "${st%% *}" in fast) ;; *) echo " -- NOT FAST: flag to the coordinator" ;; esac)"
  follow $O/$tag
}
# ---------------------------------------------------------------- main
log "RESULT xeb $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) stats $STATS, F1 hot $HOT, D3c rows $ROWS1 budget $BUDGET1 M $MUS us (round 2 rows $ROWS2 budget $BUDGET2 hot $HOT2), order $ORDER mb $DUR s"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xeb: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xeb: VIRHE image $IMAGE not on gpu02"; exit 1; }
  for f in $GGB $EPP $Y $D/stepprobe.py $D/mb.py $STATS; do [ -e $f ] || { log "RESULT xeb: VIRHE $f missing"; exit 1; }; done
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xeb: VIRHE $M225 missing or not verified"; exit 1; }
fi
mkd $O $O/guard $STATSDIR $STATSDIR/place
PL=$STATSDIR/place
# ---- round 1 tables and the model's numbers for them
run "python3 -I $EPP replicate --stats $STATS --hot $HOT -o $PL/f1-hot$HOT.place > $O/gen1.txt 2>&1"
run "python3 -I $EPP solve --stats $STATS --rows $ROWS1 --budget $BUDGET1 --M $MUS -o $PL/d3c-r1.place >> $O/gen1.txt 2>&1"
run "python3 -I $EPP eval --stats $STATS --rows 1,6,8,24 --M $MUS --place $PL/f1-hot$HOT.place --place $PL/d3c-r1.place >> $O/gen1.txt 2>&1"
[ "$DRY" = 1 ] || while IFS= read -r l; do log "RESULT xeb round 1 tables (model): ${l:0:300}"; done < <(grep '^EP' $O/gen1.txt)
evict
# ================================================================ START a: round 1
BEST1=0
if ! stopfile && startup a $PL/f1-hot$HOT.place $PL/d3c-r1.place; then
  blocks a-f1 '{"ds41_slotsum": 1, "ds41_place": 0}' '{"ds41_slotsum": 1, "ds41_place": 1}' $STATSDIR/miss-a.log
  stopfile || blocks a-d3c '{"ds41_slotsum": 1, "ds41_place": 0}' '{"ds41_slotsum": 1, "ds41_place": 2}' $STATSDIR/miss-a.log
  if [ "$DRY" != 1 ]; then
    g1=$(pooled a-f1 "c8 aggregate tok/s"); g2=$(pooled a-d3c "c8 aggregate tok/s")
    BEST1=$(awk -v a="${g1:-0}" -v b="${g2:-0}" 'BEGIN{print (a <= 1 && b <= 1) ? 0 : (a >= b) ? 1 : 2}')
    log "RESULT xeb round 1: c8 pooled B/A F1 ${g1:-?}, D3c ${g2:-?} -> best table $BEST1 (0 = contiguous: neither gained)"
  fi
  unfollow; run "docker logs $CN > $O/a-engine-full.log 2>&1"; down
else
  log "RESULT xeb: start a did not come up as asked -- round 2 runs from the round 1 tables as they are"
fi
# ---- round 2 tables: round 1's best (or F1) and a re-solve at a wider band with a replicated head
B1F=$PL/f1-hot$HOT.place; [ "$BEST1" = 2 ] && B1F=$PL/d3c-r1.place
run "cp $B1F $PL/best-r1.place"
run "python3 -I $EPP solve --stats $STATS --rows $ROWS2 --budget $BUDGET2 --hot $HOT2 --M $MUS -o $PL/d3c-r2.place > $O/gen2.txt 2>&1"
run "python3 -I $EPP eval --stats $STATS --rows 1,6,8,24 --M $MUS --place $PL/best-r1.place --place $PL/d3c-r2.place >> $O/gen2.txt 2>&1"
[ "$DRY" = 1 ] || while IFS= read -r l; do log "RESULT xeb round 2 tables (model): ${l:0:300}"; done < <(grep '^EP' $O/gen2.txt)
# ================================================================ START b: round 2, the line, prefill
if ! stopfile && startup b $PL/best-r1.place $PL/d3c-r2.place; then
  blocks b-r2 '{"ds41_slotsum": 1, "ds41_place": 1}' '{"ds41_slotsum": 1, "ds41_place": 2}' $STATSDIR/miss-b.log
  BEST=1; [ "$DRY" = 1 ] || BEST=$(( 1 + $(better b-r2) ))
  log "RESULT xeb round 2: best table $BEST ($([ $BEST = 2 ] && echo 'D3c round 2' || echo "round 1's best, table $BEST1"))"
  stopfile || blocks b-line '{"ds41_slotsum": 0, "ds41_place": 0}' "{\"ds41_slotsum\": 1, \"ds41_place\": $BEST}" $STATSDIR/miss-b.log
  if ! stopfile; then
    NPF=0; for v in A:0:0 B:1:$BEST B:1:$BEST A:0:0; do NPF=$((NPF + 1)); x=${v#*:}; arm ${x%%:*} ${x#*:}; pfd b-${v%%:*}$NPF 16000; done
    for v in A:0:0 B:1:$BEST; do x=${v#*:}; arm ${x%%:*} ${x#*:}; pfd b-${v%%:*} 136000; done
    for v in A:0:0 B:1:$BEST; do x=${v#*:}; arm ${x%%:*} ${x#*:}; mbpf b-${v%%:*}; done
  fi
  unfollow; run "docker logs $CN > $O/b-engine-full.log 2>&1"
  [ "$DRY" = 1 ] || log "RESULT xeb b engine log: $(grep -cE '^E ' $O/b-engine-full.log) E lines, $(grep -c 'fn4: switches' $O/b-engine-full.log) switch applications"
  down
fi
unset $SWITCHES
down
log "RESULT xeb $ITEM: done -- $O (placement files $PL)"
