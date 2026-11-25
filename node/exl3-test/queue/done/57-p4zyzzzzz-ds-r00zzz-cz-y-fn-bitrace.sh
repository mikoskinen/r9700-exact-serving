#!/bin/bash
# RE-SCOPED 12.11. (user: too slow, quick screens; coordinator: bitrace to the minimum that names the next diverging op, ~30 min):
# ONE start (exact wire, BI-dual declared), ONE phase sg = exl3_bi 0 + exl3_dual 1 + hc_bi 2 + bi_seg 1 (every BI piece built so far),
# runs a1 (alone) / c4 / pf (each prompt under mb c4 + one fresh ~16k prefill, was 32k) / a2 (control), first divergence per
# probe only. Dropped: the def start (the wire's own question is r09's), the wx / du / hb phases, the 179 cross-check. ~30 min.
# 180-fn-bitrace.sh (queue item, 11.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~80 min) -- fn4 BI:
# THE FLASH-NEXT OP-BY-OP SURVEY (coordinator 11.11.: exl3_bi alone does not make Flash-Next batch-invariant -- 179: c4 floor
# 10 / 20, c4 vs alone 13 / 20 under bi 1 -- run the op-by-op byte survey on the Flash-Next line, a row alone vs inside M,
# decode and mixed). One start of the EXL3 line (179's: x3stk + X3CODES + exl3_gd 1) with fn4 BI's engine tracer on
# (RADIANCE_FN4_BITRACE=/stats/bt: after every traced op a 64-bit fingerprint of each row; ~19 points a layer -- the connection
# reads + codes, the attention's qg / k / v / q / att or the delta net's in / ab / q / k / v / o, the block output, the
# mixer write's stream, the router / ids / ew / shared arm, the MoE output, the feed-forward write's stream -- then final and
# logits; rows of the greedy prompts' requests below position 256; passes not recorded), exl3_bi 1 throughout. Runs in order,
# a prefix-cache reset before each: a1 = the 20-prompt greedy set c1 ALONE; c4 = the same at conc 4; pf = each prompt under mb
# c4 streams + one fresh ~32k prefill at +0.5 s (fn-loadid.py = the DS agent's loadid.py ee7d9663 verbatim); a2 = alone again
# (control). Then fn-bitrace.py: per probe and per loaded run (c4, pf) the FIRST op whose row differs from a1, with both
# steps' shapes; text compares beside it (the tracer still reproduces 179's count?) and a1 vs 179's untraced bi-1 alone answers.
# TWO STARTS (X180_ARMS; arms of one start switched in process, coordinator 11.11.: fewer starts): def = the line's serve flags (--tp-wire wht6 --tp-wire-min-kb 128: at four ranks a message of
# 128 KiB or more, i.e. 26+ rows, goes out on the lossy tiered-int8 wire, below it exact -- a decode row alone is exact, the
# same row in a c8 / mixed / prefill step lossy: a suspected BI breaker by construction); wx = the same + --tp-wire exact
# (every message exact; the greedy prompts and decode alone are under the floor, so its a1 should equal def's) and du = wx with
# exl3_bi 0 + fn4 BI-dual (exl3_dual 1: the EXL3 downs take the step's row bands -- today's decode band bytes for the decode
# rows in a mixed step's pass): what flips once the wire and the EXL3 band switch are out, and hb = du + hc_bi 2 (hc_read up to
# 64 rows through today's per-row kernels, the reads cut at the step's segments: 185 invariant to 64, 186 the alone answers
# IDENTICAL and decode ms/step within +0.3 %), and sg = hb + bi_seg 1 (every trunk int8 linear cut at the step's segments:
# a decode row's linears at its own height, the narrow M-invariant tiles; 185: the tiled form above 64 differs in rare elements) -- wx, du, hb in ONE start
# (RADIANCE_FN4_X3DUAL=1 declares the bands; exl3_dual 0 issues none), switched in process with an epoch bump.
# Image radiance-local:c16-x3bs (fn4-x3 94ffbdac = the tracer + BI-dual + hc_bi + bi_seg; exl3r4d 630ecea at DZ 2; the smoke first), compose
# fn-tp4zt.yaml (= fn-tp4zq.yaml + the tracer's three variables + RADIANCE_FN4_X3DUAL). ~125 min. Measurement only. Power untouched. NO cc_sim.
# DRYRUN=1 bash 180-fn-bitrace.sh : prints every command; checks the helpers (files, both python selftests), the yaml (compose
# config), the image id; no server, no GPU, writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41
YN=fn-tp4zt.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/fn-bitrace.py" ] && [ -e "$f/fn-loadid.py" ] && { TL=$f; break; }; done
IMG=${X180_IMAGE:-radiance-local:c16-x3bs}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
GT=fn4-bitr; TAG=$GT; BT=$O/stats180/bt-def     # GT: the guard files; TAG / BT: the arm's (armrun)
ARMS=${X180_ARMS:-bi}   # 12.11. re-scope (user: quick screens): ONE start, ONE phase -- the full BI set as it stands
export FNIMAGE=$IMG FNMODELDIR=${X180_X3_DIR:-$HOME/models/flashnext-exl3} FNMODEL=${X180_X3_MODEL:-qwen3.8-flash-next-exl3.rad}
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
EXTRA0="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0"
export EXTRA="$EXTRA0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats180
export FN4NR=1 FN4GI=1 FN4FD="" FN4NW=1 FN4SW="" FN4WV=1 FN4RV=11 FN4AL="" FN4UM=5 FN4SS=1 FN4CH="" FN4XM="" FN4SPHC="" FN4X3C=1 FN4QSA="" FN4P2X="" FN4MP="" FN4X3GD=1 FN4X3DZ="" PROFEVERY=""
export FN4BT=/stats/bt-def FN4BTP=256 FN4BTL=128 FN4X3DUAL=""
MT=192; PFT=${X180_PFT:-16384}; OFF=0.5; CONC=4; WARM=10
REF179=$O/fn4-bigr-gs-b1-c1.jsonl                # 179's untraced alone answers at exl3_bi 1
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
compose() { docker compose --project-directory $C -f $Y "$@"; }
down() { run "compose down > /dev/null 2>&1; docker rm -f $CN > /dev/null 2>&1"; }
caps() { local d h out=""; for d in /sys/class/drm/card*/device; do [ -e $d/unique_id ] || continue; h=$(ls -d $d/hwmon/hwmon* 2>/dev/null | head -1); [ -n "$h" ] || continue
  out="$out $(basename $(readlink -f $d)):cap=$(( $(cat $h/power1_cap 2>/dev/null || echo 0) / 1000000 ))W,hot=$(( $(cat $h/temp2_input 2>/dev/null || echo 0) / 1000 ))C"; done; echo "$out"; }
tguard() { local d h j m n=0 hot
  while [ ! -e $O/$GT.tstop ]; do hot=""
    for d in /sys/class/drm/card*/device; do [ -e $d/unique_id ] || continue; h=$(ls -d $d/hwmon/hwmon* 2>/dev/null | head -1); [ -n "$h" ] || continue
      j=$(( $(cat $h/temp2_input 2>/dev/null || echo 0) / 1000 )); m=$(( $(cat $h/temp3_input 2>/dev/null || echo 0) / 1000 ))
      { [ $j -ge 100 ] || [ $m -ge 96 ]; } && hot="$hot $(basename $(readlink -f $d)):j$j/m$m"; done
    if [ -n "$hot" ]; then n=$((n + 1)); else n=0; fi
    [ $n -ge 10 ] && { : > $O/$GT.abort; log "RESULT fn4 180: VIRHE temperature guard ($hot) -> runs stopped"; return 0; }
    sleep 1; done; }
restart() {   # restart TAG -> 0 healthy and the tracer on
  local tag=$1 ok=0 t0 i S=$O/start-$1.log
  [ "$DRY" = 1 ] && { echo "+ FNIMAGE=$FNIMAGE FN4BT=$FN4BT FN4BTP=$FN4BTP FN4BTL=$FN4BTL FN4X3GD=$FN4X3GD FN4X3C=$FN4X3C EXTRA='$EXTRA' compose -f $Y up -d; health <= 1200 s; docker logs > $S; tracer ON + passes off lines"; return 0; }
  down; t0=$(date +%s); compose up -d > $O/up-$tag.txt 2>&1
  for i in $(seq 1 240); do
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' $URL/health)" = 200 ] && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Status}}' $CN 2>/dev/null)" != running ] && break
    sleep 5
  done
  docker logs $CN > $S 2>&1
  [ $ok = 1 ] || { log "RESULT fn4 180 restart-$tag: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E |radiance: qwen4exp' $S | head -3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
  grep -q "fn4 BI's tracer ON" $S && grep -q 'recorded passes: off, the program traces rows' $S || {
    log "RESULT fn4 180 restart-$tag: VIRHE the tracer is not on (no 'tracer ON' / 'recorded passes: off' line; env: $(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null | grep BITRACE | tr '\n' ' '))"; return 1; }
  log "RESULT fn4 180 restart-$tag: healthy in $(( $(date +%s) - t0 )) s; $(grep -m1 "tracer ON" $S | cut -c1-260); fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300)"
}
setsw() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
swcheck() {   # swcheck WHAT SWITCHES_JSON: every switch named there has that value now
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4: $2"; return 0; }
  local g; g=$(curl -s -m 10 $URL/admin/fn4)
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); w=json.loads(sys.argv[2]); sys.exit(0 if all(d["switches"].get(k) == v for k, v in w.items()) else 1)' "$g" "$2" 2>/dev/null \
    || { log "RESULT fn4 180 $1: VIRHE the switches are not $2 ($(echo "$g" | cut -c1-400))"; return 1; }
}
pcreset() { [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o /dev/null -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  [ "$c" = 200 ] || log "RESULT fn4 180 $1 prefix cache reset: HTTP $c"; }
GS=; for f in "$H/tools/greedyset.py" "$H/../tools/greedyset.py" "$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py"; do [ -z "$GS" ] && [ -e "$f" ] && GS=$f; done
LD=$TL/fn-loadid.py; BTP=$TL/fn-bitrace.py; MBP=$D/mb.py
alive() {   # alive WHAT: the server still answers, else logs + down + exit
  [ "$DRY" = 1 ] && return 0
  local code; code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' $URL/health)
  [ "$code" = 200 ] && return 0
  docker logs $CN > $O/$TAG-dead-engine.log 2>&1
  log "RESULT fn4 180: VIRHE the server is not healthy after $1 (/health $code; last E lines: $(grep -E '^E |trap|abort|HSA_STATUS' $O/$TAG-dead-engine.log | tail -3 | tr '\n' ' ' | cut -c1-400)) -> logs saved, down"
  down; exit 1
}
gset() {   # gset CONC NAME
  local t0=$(date +%s)
  [ -e $O/$GT.abort ] && return 0
  pcreset $2
  run "(cd $D && timeout 1500 python3 $GS $URL $O/$TAG-$2.jsonl --conc $1 --max-tokens $MT > $O/$TAG-$2.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT fn4 180 $2 (greedy set c$1, temperature 0, $MT tok, traced): $(tail -1 $O/$TAG-$2.txt | cut -c1-200), $(( $(date +%s) - t0 )) s"
}
pfrun() {   # pfrun NAME: each prompt under mb c4 + one fresh ~32k prefill
  local t0=$(date +%s) j
  [ -e $O/$GT.abort ] && return 0
  pcreset $1
  run "(cd $D && timeout 3600 python3 -I $LD run --url $URL --gs $GS --mb $MBP --out $O/$TAG-$1.jsonl --conc $CONC --warm $WARM --offset $OFF --pf-tok $PFT --max-tokens $MT > $O/$TAG-$1.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  j=$(grep -m1 '^LOADID ' $O/$TAG-$1.txt | cut -c8-)
  [ -n "$j" ] || { log "RESULT fn4 180 $1: VIRHE no LOADID line ($(tail -3 $O/$TAG-$1.txt | tr '\n' ' ' | cut -c1-400))"; return 1; }
  log "RESULT fn4 180 $1 (each prompt under mb c4 + one fresh ~${PFT}-token prefill at +${OFF} s, traced): $(python3 -c 'import json,sys
d = json.loads(sys.argv[1])
print("%s of %s answered, probe %s s mean | prefills %s tok mean, %s of %s cold, TTFT %s s mean | in flight %s..%s %% of a probe | mixed steps a probe min %s, median %s; probes without one %s | c4 %s done, %s errors" % (
    d["answered"], d["of"], d["probe_s_mean"], d["pf_prompt_mean"], d["pf_cold"], d["of"], d["pf_ttft_mean"],
    round(100 * (d["ov_frac_min"] or 0)), round(100 * (d["ov_frac_median"] or 0)), d["mixed_min"], d["mixed_median"],
    d["probes_without_mixed"] or "none", d["c4_requests_done"], d["c4_errors"]))' "$j" 2>&1 | cut -c1-900), $(( $(date +%s) - t0 )) s"
}
lcmp() {   # lcmp LABEL A B: byte for byte + the first differing char / token
  local r
  [ "$DRY" = 1 ] && { echo "+ python3 -I $LD cmp --url $URL $2 $3 --json $O/$TAG-cmp-$1.json"; return 0; }
  [ -s "$2" ] && [ -s "$3" ] || { log "RESULT fn4 180 text $1: not compared (missing $2 or $3)"; return 0; }
  r=$(timeout 300 python3 -I $LD cmp --url $URL --json $O/$TAG-cmp-$1.json --label "$1" "$2" "$3" 2>&1 | grep -m1 '^CMPLINE ' | cut -c9-)
  log "RESULT fn4 180 text $1: ${r:-VIRHE compare failed}"
}
btcheck() {   # btcheck WHAT: every rank wrote records
  [ "$DRY" = 1 ] && { echo "+ ranks 0..3: $BT/bt-r<k>-*.bin non-empty, .names present"; return 0; }
  local k miss=""
  for k in 0 1 2 3; do
    ls $BT/bt-r$k-*.bin > /dev/null 2>&1 && [ -n "$(find $BT -name "bt-r$k-*.bin" -size +0 2>/dev/null)" ] && ls $BT/bt-r$k-*.names > /dev/null 2>&1 || miss="$miss r$k"
  done
  [ -z "$miss" ] && return 0
  docker logs $CN > $O/$TAG-engine.log 2>&1
  log "RESULT fn4 180 $1: VIRHE the tracer wrote no records for$miss ($(grep -E "tracer" $O/$TAG-engine.log | tail -2 | tr '\n' ' ' | cut -c1-300)) -> down"
  down; exit 1
}
log "RESULT fn4 180-fn-bitrace: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) the EXL3 line, the BI set (sg), fn4 BI tracer (PMAX $FN4BTP, PLEN $FN4BTL): arms $ARMS, each a1 / c4 / pf / a2 | power: $( [ "$DRY" = 1 ] && echo dry || caps)"

if [ "$DRY" != 1 ]; then
  [ "$(cat $O/fn4-smoke-ok 2>/dev/null)" = "$(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)" ] || { log "RESULT fn4 180: VIRHE the smoke (187) has not passed on this image -- not started"; exit 1; }
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 180: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 180: VIRHE image $IMG missing"; exit 1; }
  [ -e "$Y" ] && [ -n "$TL" ] && [ -n "$GS" ] && [ -e "$MBP" ] || { log "RESULT fn4 180: VIRHE $YN, tools/fn-bitrace.py / fn-loadid.py, greedyset.py or mb.py missing"; exit 1; }
  fr=$(df -BG --output=avail $R | tail -1 | tr -dc 0-9)
  [ "${fr:-0}" -ge 4 ] || { log "RESULT fn4 180: VIRHE only ${fr} GB free under $R (the trace needs ~1-2 GB)"; exit 1; }
  for a in $ARMS; do mkdir -p $O/stats180/bt-$a && chmod 777 $O/stats180/bt-$a && find $O/stats180/bt-$a -maxdepth 1 -name 'bt-r*' -delete; done
  find $O -maxdepth 1 -name "$GT.abort" -delete; find $O -maxdepth 1 -name "$GT.tstop" -delete
  trap 'touch $O/$GT.tstop 2>/dev/null; down' EXIT
  trap 'exit 143' TERM INT
  tguard & TGPID=$!
else
  echo "+ guards (no radiance container, image, files, >= 4 GB free); arms '$ARMS': mkdir $O/stats180/bt-<arm> (777), old bt-r* removed; tguard &"; TGPID=""
  echo "+ $(python3 -I $TL/fn-bitrace.py selftest 2>&1 | tail -1)"
  echo "+ $(python3 -I $LD selftest --gs $GS --mb $MBP 2>&1 | tail -1)"
  compose config -q > /dev/null 2>&1 && echo "+ compose config $YN: OK" || echo "+ compose config $YN: VIRHE"
  echo "+ image $IMG: $(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)"
  echo "+ tools: TL=$TL GS=$GS MBP=$MBP REF179=$( [ -s $REF179 ] && echo present || echo MISSING) free $(df -BG --output=avail $R | tail -1)"
fi
armrun() {   # armrun START DUAL "SERVE WORDS" PHASE=SWITCHES_JSON...: one start with the tracer; per phase (switched in
  # process, an epoch bump) a1 / c4 / pf / a2 and the text compares; then the analysis (--group 4 over the phases)
  local st=$1 du=$2 add=$3; shift 3
  local phases="" ph sw nph=$#
  BT=$O/stats180/bt-$st; export FN4BT=/stats/bt-$st EXTRA="$EXTRA0${add:+ $add}" FN4X3DUAL=$du; TAG=$GT-$st
  [ -e $O/$GT.abort ] && return 0
  log "RESULT fn4 180 start $st: serve words EXTRA='$EXTRA', RADIANCE_FN4_X3DUAL='$du', phases $*, trace $BT"
  restart $TAG || return 1
  local first=1
  for p in "$@"; do
    ph=${p%%=*}; sw=${p#*=}; phases="$phases${phases:+,}$ph"; TAG=$GT-$ph
    [ -e $O/$GT.abort ] && break
    setsw "$(echo "$sw" | sed 's/}$/, "epoch_bump": 1}/')"; nudge; swcheck "$ph start" "$sw" || { down; return 1; }
    if [ $first = 1 ] && [ "$DRY" != 1 ]; then
      mkdir -p $O/guard
      (cd $D && timeout 900 python3 ds41_tests.py correct --only-q --effort none --outdir $O/guard > $O/guard/$TAG-warm-q.txt 2>&1)
    fi
    alive "$ph warm-up"
    gset 1 a1; alive "$ph a1"; [ $first = 1 ] && btcheck "$ph a1"
    gset 4 c4; alive "$ph c4"
    pfrun pf; alive "$ph pf"
    gset 1 a2; alive "$ph a2"
    swcheck "$ph after a2" "$sw"
    lcmp $ph-c4-vs-a1 $O/$TAG-a1.jsonl $O/$TAG-c4.jsonl
    lcmp $ph-pf-vs-a1 $O/$TAG-a1.jsonl $O/$TAG-pf.jsonl
    lcmp $ph-a2-vs-a1 $O/$TAG-a1.jsonl $O/$TAG-a2.jsonl
    [ -n "${X180_REF179:-}" ] && lcmp $ph-a1-vs-179-untraced $REF179 $O/$TAG-a1.jsonl
    first=0
  done
  TAG=$GT-$st
  run "docker logs $CN > $O/$TAG-engine.log 2>&1"
  down
  local grp=""; [ $nph -gt 1 ] && grp="--group 4 --groups $phases"
  [ "$DRY" = 1 ] && { echo "+ python3 -I $BTP analyze $BT --labels c4,pf $grp --json $O/$TAG-bitrace.json > $O/$TAG-bitrace.txt; log the summaries"; return 0; }
  log "RESULT fn4 180 $st tracer files: $(ls $BT/bt-r*.bin 2>/dev/null | wc -l) files, $(du -sm $BT | cut -f1) MB; engine: $(grep -cE 'fn4: switches .* applied' $O/$TAG-engine.log) applied changes, $(grep -c "tracer cannot open" $O/$TAG-engine.log) open failures"
  timeout 2400 python3 -I $BTP analyze $BT --labels c4,pf $grp --json $O/$TAG-bitrace.json > $O/$TAG-bitrace.txt 2>&1
  grep -q '^BITRACE summary' $O/$TAG-bitrace.txt || log "RESULT fn4 180 $st bitrace: VIRHE no summary ($(tail -3 $O/$TAG-bitrace.txt | tr '\n' ' ' | cut -c1-400))"
  while IFS= read -r l; do log "RESULT fn4 180 $st $(echo "$l" | cut -c1-1200)"; done < <(grep -E '^BITRACE summary' $O/$TAG-bitrace.txt)
  log "RESULT fn4 180 $st per-probe first divergences: $O/$TAG-bitrace.txt ($(grep -c 'FIRST DIVERGENCE' $O/$TAG-bitrace.txt) lines)"
}
for a in $ARMS; do
  case $a in
    bi)   armrun bi 1 "--tp-wire exact" 'sg={"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 2, "bi_seg": 1}' ;;
    def)  armrun def "" "" 'def={"exl3_bi": 1}' ;;
    wxdu) armrun wxdu 1 "--tp-wire exact" 'wx={"exl3_bi": 1, "exl3_dual": 0, "hc_bi": 0, "bi_seg": 0}' 'du={"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 0, "bi_seg": 0}' 'hb={"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 2, "bi_seg": 0}' 'sg={"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 2, "bi_seg": 1}' ;;
    *) log "RESULT fn4 180: VIRHE no start '$a'" ;;
  esac
done
run "touch $O/$GT.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
log "RESULT fn4 180-fn-bitrace: done"
