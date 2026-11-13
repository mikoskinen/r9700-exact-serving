#!/bin/bash
# 220-fn-g16p.sh (queue item, 13.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~35 min) -- two in-process decode
# levers on the STEP LOG (TESTING.md 13.11.), radiance-local:c16-x3sp17 (engagement counter) on fn-tp4zu.yaml:
#  (1) G16p (hc_readv bit 8: G16 with two tiles a block, from 32 rows; x19 byte-identical): xeon x26 COLD (--cold 256, one card)
#      M32 G16 49.32 -> G16p 46.12 us (-6.5 %), M64 69.32 -> 64.76 (-6.6 %) [G16q cold 52.60 / 77.88: +7 / +12 %, matching 217's
#      engine loss]; today's non-BI decode read cold M32 49.28 / M64 63.72. S7 (hc_readv 75) vs S7P (331).
#  (2) att_bi 2 (the sparse decode sequences in ONE attn_paged_gate_quant launch, split_rows = 1 + n_spec) vs att_bi 1 (a sequence a
#      launch + a separate gate_quant_fp8): 200 read -0.9 % on mb (inside mb's noise); xeon x25 (TP2 c8 profile): att_bi 1 = attn_paged
#      16.0 us + gate_quant_fp8 6.3 us a sequence vs the fused batch op 44 us a layer-pass.
# START 1 (unprofiled, step log): greedy-20 a1 vs c16 under S7P + att_bi 2 (both), warm-up, then mbt2 --gens 2 A B B A A B B A:
#   g16p c16 / c8 (S7 vs S7P), attbi2 c16 / c8 / c4 (S7 vs S7 + att_bi 2); tools/fnsteps.py (full-row p50 + forms a block).
# START 2 (--profile-ops every 4 steps): c8 windows S7 / S7 + att_bi 2 / S7 + att_bi 2 / S7 -> opwin (split_rows key = att_bi 2's
#   launch engaged) + opdiff.
# DRYRUN=1 bash 220-fn-g16p.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4zu.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/heatsoak.sh" ] && { TL=$f; break; }; done
IMG=${X220_IMAGE:-radiance-local:c16-x3sp17}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-gp
export FNIMAGE=$IMG FNMODELDIR=${X220_X3_DIR:-$HOME/models/flashnext-exl3} FNMODEL=${X220_X3_MODEL:-qwen3.8-flash-next-exl3.rad}
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
EXTRA0="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0"
export EXTRA="$EXTRA0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats220
export FN4BT="" FN4BTP="" FN4BTL="" FN4X3DUAL=1 FN4NR=1 FN4GI=1 FN4FD="" FN4NW=1 FN4SW="" FN4WV=1 FN4RV=11 FN4AL="" FN4UM=5 FN4SS=1 FN4CH="" FN4XM=1 FN4SPHC=1 FN4X3C=1 FN4QSA="" FN4P2X="" FN4MP="" FN4X3GD=1 FN4X3DZ=1 PROFEVERY="" FN4MTPF=1 LOGSTEPS=1 FN4FC=1
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
compose() { docker compose --project-directory $C -f $Y "$@"; }
down() { run "compose down > /dev/null 2>&1; docker rm -f $CN > /dev/null 2>&1"; }
[ -n "$TL" ] && . $TL/heatsoak.sh
caps() { local d h out=""; for d in /sys/class/drm/card*/device; do [ -e $d/unique_id ] || continue; h=$(ls -d $d/hwmon/hwmon* 2>/dev/null | head -1); [ -n "$h" ] || continue
  out="$out $(basename $(readlink -f $d)):cap=$(( $(cat $h/power1_cap 2>/dev/null || echo 0) / 1000000 ))W,hot=$(( $(cat $h/temp2_input 2>/dev/null || echo 0) / 1000 ))C"; done; echo "$out"; }
tguard() { local d h j m n=0 hot
  while [ ! -e $O/$TAG.tstop ]; do hot=""
    for d in /sys/class/drm/card*/device; do [ -e $d/unique_id ] || continue; h=$(ls -d $d/hwmon/hwmon* 2>/dev/null | head -1); [ -n "$h" ] || continue
      j=$(( $(cat $h/temp2_input 2>/dev/null || echo 0) / 1000 )); m=$(( $(cat $h/temp3_input 2>/dev/null || echo 0) / 1000 ))
      { [ $j -ge 100 ] || [ $m -ge 96 ]; } && hot="$hot $(basename $(readlink -f $d)):j$j/m$m"; done
    if [ -n "$hot" ]; then n=$((n + 1)); else n=0; fi
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 220: VIRHE temperature guard ($hot) -> blocks stopped"; return 0; }
    sleep 1; done; }
restart() {   # restart TAG -> 0 healthy
  local tag=$1 ok=0 t0 i S=$O/start-$1.log
  [ "$DRY" = 1 ] && { echo "+ FNIMAGE=$FNIMAGE FN4NR=$FN4NR FN4GI=$FN4GI FN4NW=$FN4NW FN4WV=$FN4WV FN4RV=$FN4RV FN4UM=$FN4UM FN4CH=$FN4CH EXTRA='$EXTRA' compose -f $Y up -d; health <= 1200 s; docker logs > $S"; return 0; }
  down; t0=$(date +%s); compose up -d > $O/up-$tag.txt 2>&1
  for i in $(seq 1 240); do
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' $URL/health)" = 200 ] && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Status}}' $CN 2>/dev/null)" != running ] && break
    sleep 5
  done
  docker logs $CN > $S 2>&1
  [ $ok = 1 ] || { log "RESULT fn4 220 restart-$tag: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
  log "RESULT fn4 220 restart-$tag: healthy in $(( $(date +%s) - t0 )) s; $(grep -m1 -E '^I kv: ' $S | cut -c3-160); fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300)"
}
probe8() {   # probe8 TAG -> "ms tok/s" of a 45 s mb c8
  [ "$DRY" = 1 ] && { echo "+ (cd $D && timeout 400 python3 mb.py --tag $1 --phases c8 --dur 45 --warm 10 > $O/$1-c8.txt)" >&2; echo "25.6 744"; return 0; }
  (cd $D && timeout 400 python3 mb.py --tag $1 --phases c8 --dur 45 --warm 10 > $O/$1-c8.txt 2>&1)
  grep '^MB ' $O/$1-c8.txt 2>/dev/null | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[3:]); print(d.get("ms_decode_step"), d.get("agg_dec_tps"))' 2>/dev/null || echo "0 0"
}
setsw() {   # setsw JSON: queue switches (they land at the next step release)
  run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"
}
# 12.11. 16:3x: re-run of the T byte check only; 198 called nudge/swcheck without defining them (T skipped in 0 s)
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
swcheck() {   # swcheck WHAT SWITCHES_JSON: every switch named there has that value now
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4: $2"; return 0; }
  local g; g=$(curl -s -m 10 $URL/admin/fn4)
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); w=json.loads(sys.argv[2]); sys.exit(0 if all(d["switches"].get(k) == v for k, v in w.items()) else 1)' "$g" "$2" 2>/dev/null \
    || { log "RESULT fn4 220 $1: VIRHE the switches are not $2 ($(echo "$g" | cut -c1-400))"; return 1; }
}
alive() {   # alive STEP: /health 200, else log it (the T arm then yields no answers)
  [ "$DRY" = 1 ] && { echo "+ curl -m 5 $URL/health == 200"; return 0; }
  local code; code=$(curl -s -m 5 -o /dev/null -w %{http_code} $URL/health)
  [ "$code" = 200 ] && return 0
  log "RESULT fn4 220 $1: VIRHE the server is not healthy (/health $code; $(docker inspect -f {{.State.Status}} exit {{.State.ExitCode}} $CN 2>/dev/null || echo gone))"; return 1
}
GS=; for f in "$H/tools/greedyset.py" "$H/../tools/greedyset.py" "$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py"; do [ -z "$GS" ] && [ -e "$f" ] && GS=$f; done
LD=$TL/fn-loadid.py; MBT=$TL/fnharness/mbt2.py; GL=$TL/greedylong.py; BTP=$TL/fn-bitrace.py
S0='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 4, "hc_sp": 0, "mtp_flat": 0, "ar_xpush": 0}'
S0X='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 4, "hc_sp": 0, "mtp_flat": 0, "ar_xpush": 1}'
S2='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 4, "hc_sp": 2, "mtp_flat": 0, "ar_xpush": 0}'
S2X='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 4, "hc_sp": 2, "mtp_flat": 0, "ar_xpush": 1}'
pcr() { run "curl -s -m 60 -o /dev/null -X POST -d '' $URL/reset_prefix_cache"; }
sw() { setsw "$(echo "$1" | sed 's/}$/, "epoch_bump": 1}/')"; nudge; swcheck "$2" "$1"; }
glr() { pcr; run "(cd $D && GL_N=4 timeout 1800 python3 $GL $URL $O/$TAG-$1.jsonl --conc 1 --max-tokens 64 > $O/$TAG-$1.txt 2>&1)"; }
gsr() { pcr; run "(cd $D && timeout 900 python3 $GS $URL $O/$TAG-$1.jsonl --conc 1 --max-tokens 192 > $O/$TAG-$1.txt 2>&1)"; }
lc() { [ "$DRY" = 1 ] && { echo "+ cmp $1 $2"; return 0; }; log "RESULT fn4 220 text $3: $(timeout 300 python3 -I $LD cmp --url $URL --label "$3" $O/$TAG-$1.jsonl $O/$TAG-$2.jsonl 2>&1 | grep -m1 '^CMPLINE ' | cut -c9- | cut -c1-400)"; }
abr() {   # abr NAME A B (extra abblocks2 args...)
  local nm=$1 A=$2 B=$3; shift 3
  run "timeout 2400 python3 -I $TL/abblocks2.py --url $URL --mb-dir $D --label $TAG-$nm --rounds 1 --a '$A' --b '$B' --names a,b --out $O/$TAG-blocks.jsonl $* > $O/$TAG-$nm.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 220 $nm (A B B A): $(grep '^ABB ' $O/$TAG-$nm.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=4: "-" if v is None else format(v, ".%df" % n)
s = "decode tok/s B/A %s %s ms/step B/A %s (A %s | B %s)" % (f(d["tps_ratio"]), (d.get("sep") or {}).get("tps"), f(d.get("ms_ratio")), " ".join("%.1f" % x for x in d.get("A_blocks", [])), " ".join("%.1f" % x for x in d.get("B_blocks", []))) if d.get("rounds") else ""
for k, v in (d.get("pf") or {}).items():
    s += " | prefill %s B/A %s %s (A %s | B %s)" % (k, f(v["ratio"]), v.get("sep"), " ".join("%.0f" % x for x in v.get("A_blocks", [])), " ".join("%.0f" % x for x in v.get("B_blocks", [])))
print(s + " | %d blocks, %d flagged, %s s" % (d["blocks"], d["blocks_bad"], d["wall_s"]))' 2>&1 | cut -c1-700)"
  [ "$DRY" = 1 ] || log "RESULT fn4 220 $nm blocks (tok/s, ms/step, acceptance): $(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("label") == sys.argv[2]: print("%s %s tok/s %s ms/step acc %s;" % (r["arm"], r.get("tps"), r.get("ms"), (r.get("mb") or {}).get("acceptance")), end=" ")' $O/$TAG-blocks.jsonl $TAG-$nm 2>&1 | cut -c1-300)"
}
log "RESULT fn4 220-fn-g16p: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) exact wire + SPHC + MTPFLAT declared + step log + form counter; G16p and att_bi 2 on the step log (ABBAABBA), then a profiled start | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 220: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 220: VIRHE image $IMG missing"; exit 1; }
  [ -e "$Y" ] && [ -n "$TL" ] && [ -e $TL/abblocks2.py ] && [ -e $MBT ] && [ -e $LD ] && [ -e $GL ] && [ -e $BTP ] && [ -n "$GS" ] && [ -e $D/mb.py ] && [ -e $TL/fnsteps.py ] && [ -e $TL/opwin.py ] && [ -e $TL/opdiff.py ] || { log "RESULT fn4 220: VIRHE tools missing"; exit 1; }
  grep -q 'RADIANCE_FN4_FORMCTR' $Y && grep -q 'RADIANCE_LOG_STEPS' $Y && grep -q 'RADIANCE_FN4_MTPFLAT' $Y || { log "RESULT fn4 220: VIRHE $YN does not pass RADIANCE_FN4_MTPFLAT / _FORMCTR / RADIANCE_LOG_STEPS"; exit 1; }
  mkdir -p $O $STATSDIR/bt-sp && chmod 777 $STATSDIR/bt-sp && find $STATSDIR/bt-sp -maxdepth 1 -name 'bt-r*' -delete
  find $O -maxdepth 1 -name "$TAG.abort" -delete; find $O -maxdepth 1 -name "$TAG.tstop" -delete
  find $O -maxdepth 1 \( -name "$TAG-*.jsonl" -o -name "$TAG-*.txt" \) -delete
  trap 'touch $O/$TAG.tstop 2>/dev/null; down' EXIT
  trap 'exit 143' TERM INT
  tguard & TGPID=$!
else
  echo "+ guards; tguard &"; TGPID=""
fi
export EXTRA="$EXTRA0 --tp-wire exact"
FREEG=$(df --output=avail -BG / 2>/dev/null | tail -1 | tr -dc 0-9)
[ "$DRY" != 1 ] && [ "${FREEG:-0}" -lt 12 ] && { log "RESULT fn4 220: VIRHE only ${FREEG:-?} GB free on / -- not started"; exit 1; }
S7='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 4, "hc_readv": 75}'
S7P='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 4, "hc_readv": 331}'
S7A='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 2, "arw_rows": 16, "sched_bi_chunk": 4, "hc_readv": 75}'
S7PA='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 2, "arw_rows": 16, "sched_bi_chunk": 4, "hc_readv": 331}'
gs() { pcr; run "(cd $D && timeout 1500 python3 $GS $URL $O/$TAG-$2.jsonl --conc $1 --max-tokens 192 > $O/$TAG-$2.txt 2>&1)"; }
lc1() { [ "$DRY" = 1 ] && { echo "+ cmp $1"; return 0; }; log "RESULT fn4 220 text $1: $(timeout 300 python3 -I $LD cmp --url $URL --label "$1" $O/$TAG-a1.jsonl $O/$TAG-$1.jsonl 2>&1 | grep -m1 '^CMPLINE ' | cut -c9- | cut -c1-500)"; }
# ---------------- START 1: step log
restart $TAG-S || exit 1
nudge
sw "$S7PA" "S7PA" || { down; exit 1; }
gs 1 a1; alive "a1"
gs 16 c16; alive "c16"
lc1 c16
run "(cd $D && timeout 300 python3 mb.py --tag $TAG-warm --phases c16 --dur 15 --warm 5 > $O/$TAG-warm.txt 2>&1)"; alive "warm-up"
abr g16p-c16 "$S7" "$S7P" --mbt2 $MBT --gens 2 --phase c16 --warm 8 --rounds 2; alive "g16p c16"
abr g16p-c8 "$S7" "$S7P" --mbt2 $MBT --gens 2 --phase c8 --warm 8 --rounds 2; alive "g16p c8"
abr attbi2-c16 "$S7" "$S7A" --mbt2 $MBT --gens 2 --phase c16 --warm 8 --rounds 2; alive "attbi2 c16"
abr attbi2-c8 "$S7" "$S7A" --mbt2 $MBT --gens 2 --phase c8 --warm 8 --rounds 2; alive "attbi2 c8"
abr attbi2-c4 "$S7" "$S7A" --mbt2 $MBT --gens 2 --phase c4 --warm 8 --rounds 2; alive "attbi2 c4"
sw "$S7" "S7 end"
run "docker logs -t $CN > $O/$TAG-S-engine.log 2>&1"
if [ "$DRY" = 1 ]; then echo "+ fnsteps.py $O/$TAG-S-engine.log --blocks $O/$TAG-blocks.jsonl --label $TAG-g16p-c16:16:64 --label $TAG-g16p-c8:8:32 --label $TAG-attbi2-c16:16:64 --label $TAG-attbi2-c8:8:32 --label $TAG-attbi2-c4:4:16"
else
  timeout 300 python3 -I $TL/fnsteps.py $O/$TAG-S-engine.log --blocks $O/$TAG-blocks.jsonl --label $TAG-g16p-c16:16:64 --label $TAG-g16p-c8:8:32 \
    --label $TAG-attbi2-c16:16:64 --label $TAG-attbi2-c8:8:32 --label $TAG-attbi2-c4:4:16 --json $O/$TAG-steps.json > $O/$TAG-steps.txt 2>&1
  while IFS= read -r l; do log "RESULT fn4 220 ${l:0:900}"; done < <(grep -E '^FNSTEPS |^FNSTEPS VIRHE|^Traceback' $O/$TAG-steps.txt)
  log "RESULT fn4 220 forms a block (dev 0, first A / first B of each label): $(grep '^FNFORM ' $O/$TAG-steps.txt | grep -v total | awk '$5 == "dev" && $6 == "0"' | awk '!s[$2" "$3]++' | cut -c8- | tr '\n' '|' | cut -c1-900)"
  log "RESULT fn4 220 engine S: $(grep -cE 'fn4: switches .* applied' $O/$TAG-S-engine.log) applied changes, $(grep -cE 'fn4: .* refused' $O/$TAG-S-engine.log) refused; E lines $(grep -cE ' E ' $O/$TAG-S-engine.log); step lines $(grep -cE ' I step [0-9]+ ' $O/$TAG-S-engine.log)"
fi
down
run "gzip -f $O/$TAG-S-engine.log $O/start-$TAG-S.log 2>/dev/null; true"
# ---------------- START 2: profiled (att_bi 2 engagement by the split_rows key + op attribution)
FREEG=$(df --output=avail -BG / 2>/dev/null | tail -1 | tr -dc 0-9)
if [ "$DRY" != 1 ] && [ "${FREEG:-0}" -lt 12 ]; then log "RESULT fn4 220: VIRHE only ${FREEG:-?} GB free on / -- profiled start skipped"
else
  export PROFEVERY=4 EXTRA="$EXTRA0 --tp-wire exact --profile-ops"
  restart $TAG-P || { log "RESULT fn4 220-fn-g16p: done (profiled start failed)"; exit 1; }
  nudge
  WE=""
  win() {   # win LABEL SWITCHES: switch (+ epoch bump), the epoch, one seeded mbt2 c8 generation
    sw "$2" "$1" || return 1
    local e; if [ "$DRY" = 1 ]; then e=0; else e=$(curl -s -m 5 $URL/admin/fn4 | python3 -c 'import json,sys; print(json.load(sys.stdin).get("epoch"))' 2>/dev/null); fi
    run "(cd $D && timeout 900 python3 $MBT --mb mb.py --temperature 0 --tag $TAG-$1 --phases c8 --pattern 1x8 --seed 1 --gens 1 --warm 8 > $O/$TAG-$1-mb.txt 2>&1)"
    WE="$WE $1:${e:-x}"
  }
  sw "$S7" "S7 P"; run "(cd $D && timeout 600 python3 $MBT --mb mb.py --temperature 0 --tag $TAG-pwarm --phases c8 --pattern 1x8 --seed 1 --gens 1 --warm 8 > $O/$TAG-pwarm-mb.txt 2>&1)"
  win A1 "$S7"; win B1 "$S7A"; win B2 "$S7A"; win A2 "$S7"
  sw "$S7" "end P"
  run "docker logs $CN > $O/$TAG-P-engine.log 2>&1"
  down
  if [ "$DRY" = 1 ]; then echo "+ windows by epochs [$WE] -> opwin --ops . -> opdiff A (S7) vs B (att_bi 2)"
  else
    WINS=$(python3 - $O/$TAG-P-engine.log $WE <<'PY'
import re, sys
f = sys.argv[1]; we = [x.split(":") for x in sys.argv[2:]]
ln = {}
for n, l in enumerate(open(f, errors="replace")):
    m = re.search(r"fn4: switches .* applied at step \d+ \(epoch (\d+)\)", l)
    if m: ln[int(m.group(1))] = n
eps = sorted(ln); out = []
for lab, e in we:
    if not e.isdigit() or int(e) not in ln: continue
    e = int(e); nxt = [x for x in eps if x > e]
    if nxt: out += ["--win", "%s:%d:%d" % (lab, ln[e], ln[nxt[0]])]
print(" ".join(out))
PY
)
    log "RESULT fn4 220 profiled windows as log lines: $WINS"
    timeout 1500 python3 -I $TL/opwin.py $O/$TAG-P-engine.log $WINS --ops 'attn|gate_quant|hc_read' --top 8 --json $O/$TAG-opwin-attn.json > $O/$TAG-opwin-attn.txt 2>&1
    grep '^OPWIN ' $O/$TAG-opwin-attn.txt | grep ' rank 0:' | while IFS= read -r l; do log "RESULT fn4 220 ${l:0:900}"; done
    timeout 1500 python3 -I $TL/opwin.py $O/$TAG-P-engine.log $WINS --ops . --top 8 --json $O/$TAG-opwin.json > $O/$TAG-opwin.txt 2>&1
    timeout 300 python3 -I $TL/opdiff.py $O/$TAG-opwin.json --arm A=A1,A2 --arm B=B1,B2 --base A --top 10 > $O/$TAG-opdiff.txt 2>&1
    grep -h '^OPDIFF ' $O/$TAG-opdiff.txt | grep ' rank 0 ' | while IFS= read -r l; do log "RESULT fn4 220 ${l:0:1600}"; done
    log "RESULT fn4 220 engine P: $(grep -cE 'fn4: switches .* applied' $O/$TAG-P-engine.log) applied, $(grep -cE 'fn4: .* refused' $O/$TAG-P-engine.log) refused; E lines $(grep -cE '^E ' $O/$TAG-P-engine.log)"
  fi
  run "gzip -f $O/$TAG-P-engine.log $O/start-$TAG-P.log 2>/dev/null; true"
fi
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
log "RESULT fn4 220-fn-g16p: done"
