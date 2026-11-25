#!/bin/bash
# 216-fn-sdma3.sh (queue item, 12.11.2026, SDMA-AR agent, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~30 min) --
# SDMA-AR IN THE ENGINE: switch ar_sdma (a prefill piece's collectives at overlap 1 -- the lane all-reduce and SP's raw
# reduce-scatter / all-gather, from 1048576 elements -- on each card's SDMA engine 1; the same fold, the same bytes; decode
# keeps the CU forms) on radiance-local:c16-x3sdma2 (fn4-x3-sp3-sdma fd9db762 = fn4-x3-sp3 69917ad7 + SDMA-AR), today's best
# exact prefill set: S7 + hc_sp 2 + ar_xpush 1 (+ hc_readv 75), SPHC + MTPFLAT declared, MNBT 4096, --prefill-span 2;
# RADIANCE_FN4_AR_SDMA=0 declares (contexts built, switch at 0); fn-tp4zs.yaml passes it.
#   T (traced, PLEN 4096 / PMAX 6144): 4 long prompts alone under the base (a1), base + ar_sdma 1 (sd), base (a2): the first
#     differing point (want none) + the texts
#   S (untraced): one fresh 64k warm-up prefill, then fresh prefill 64k / 128k A B B A: ar_sdma 0 vs 1; then c16 decode
#     (mbt2) ar_sdma 0 vs 1 (gated by size: expect 0)
# Disk: not started under 12 GB free on /; the bitrace bins deleted once summarised; logs gzipped.
# Gate: when 215-fn-sdma2's engine-path selftest (runs/fn4/sdma2-eng.txt) exists, it must show 0 bad bytes, else not started.
# DRYRUN=1 bash 216-fn-sdma3.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4zs.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/heatsoak.sh" ] && { TL=$f; break; }; done
IMG=${XSD3_IMAGE:-radiance-local:c16-x3sdma2}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-sd3
export FNIMAGE=$IMG FNMODELDIR=${XSD3_X3_DIR:-$HOME/models/flashnext-exl3} FNMODEL=${XSD3_X3_MODEL:-qwen3.8-flash-next-exl3.rad}
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
EXTRA0="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0"
export EXTRA="$EXTRA0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats-sd3
export FN4BT="" FN4BTP="" FN4BTL="" FN4X3DUAL=1 FN4NR=1 FN4GI=1 FN4FD="" FN4NW=1 FN4SW="" FN4WV=1 FN4RV=11 FN4AL="" FN4UM=5 FN4SS=1 FN4CH="" FN4XM=1 FN4SPHC=1 FN4X3C=1 FN4QSA="" FN4P2X="" FN4MP="" FN4X3GD=1 FN4X3DZ=1 PROFEVERY="" FN4MTPF=1
export FN4SDMA=0 SDMAENG=1
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
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 sdma3: VIRHE temperature guard ($hot) -> blocks stopped"; return 0; }
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
  [ $ok = 1 ] || { log "RESULT fn4 sdma3 restart-$tag: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
  log "RESULT fn4 sdma3 restart-$tag: healthy in $(( $(date +%s) - t0 )) s; $(grep -m1 -E '^I kv: ' $S | cut -c3-160); fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300)"
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
    || { log "RESULT fn4 sdma3 $1: VIRHE the switches are not $2 ($(echo "$g" | cut -c1-400))"; return 1; }
}
alive() {   # alive STEP: /health 200, else log it (the T arm then yields no answers)
  [ "$DRY" = 1 ] && { echo "+ curl -m 5 $URL/health == 200"; return 0; }
  local code; code=$(curl -s -m 5 -o /dev/null -w %{http_code} $URL/health)
  [ "$code" = 200 ] && return 0
  log "RESULT fn4 sdma3 $1: VIRHE the server is not healthy (/health $code; $(docker inspect -f {{.State.Status}} exit {{.State.ExitCode}} $CN 2>/dev/null || echo gone))"; return 1
}
GS=; for f in "$H/tools/greedyset.py" "$H/../tools/greedyset.py" "$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py"; do [ -z "$GS" ] && [ -e "$f" ] && GS=$f; done
LD=$TL/fn-loadid.py; MBT=$TL/fnharness/mbt2.py; GL=$TL/greedylong.py; BTP=$TL/fn-bitrace.py
# today's best exact prefill set (the Qwen agent 12.11.: S7 + hc_sp 2 + ar_xpush 1; hc_readv 75 = G16, decode only)
SB='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 4, "hc_readv": 75, "hc_sp": 2, "mtp_flat": 0, "ar_xpush": 1, "ar_sdma": 0}'
SD='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 1, "arw_rows": 16, "sched_bi_chunk": 4, "hc_readv": 75, "hc_sp": 2, "mtp_flat": 0, "ar_xpush": 1, "ar_sdma": 1}'
pcr() { run "curl -s -m 60 -o /dev/null -X POST -d '' $URL/reset_prefix_cache"; }
sw() { setsw "$(echo "$1" | sed 's/}$/, "epoch_bump": 1}/')"; nudge; swcheck "$2" "$1"; }
glr() { pcr; run "(cd $D && GL_N=4 timeout 1800 python3 $GL $URL $O/$TAG-$1.jsonl --conc 1 --max-tokens 64 > $O/$TAG-$1.txt 2>&1)"; }
gsr() { pcr; run "(cd $D && timeout 900 python3 $GS $URL $O/$TAG-$1.jsonl --conc 1 --max-tokens 192 > $O/$TAG-$1.txt 2>&1)"; }
lc() { [ "$DRY" = 1 ] && { echo "+ cmp $1 $2"; return 0; }; log "RESULT fn4 sdma3 text $3: $(timeout 300 python3 -I $LD cmp --url $URL --label "$3" $O/$TAG-$1.jsonl $O/$TAG-$2.jsonl 2>&1 | grep -m1 '^CMPLINE ' | cut -c9- | cut -c1-400)"; }
abr() {   # abr NAME A B (extra abblocks2 args...)
  local nm=$1 A=$2 B=$3; shift 3
  run "timeout 2400 python3 -I $TL/abblocks2.py --url $URL --mb-dir $D --label $TAG-$nm --rounds 1 --a '$A' --b '$B' --names a,b --out $O/$TAG-blocks.jsonl $* > $O/$TAG-$nm.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 sdma3 $nm (A B B A): $(grep '^ABB ' $O/$TAG-$nm.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=4: "-" if v is None else format(v, ".%df" % n)
s = "decode tok/s B/A %s %s ms/step B/A %s (A %s | B %s)" % (f(d["tps_ratio"]), (d.get("sep") or {}).get("tps"), f(d.get("ms_ratio")), " ".join("%.1f" % x for x in d.get("A_blocks", [])), " ".join("%.1f" % x for x in d.get("B_blocks", []))) if d.get("rounds") else ""
for k, v in (d.get("pf") or {}).items():
    s += " | prefill %s B/A %s %s (A %s | B %s)" % (k, f(v["ratio"]), v.get("sep"), " ".join("%.0f" % x for x in v.get("A_blocks", [])), " ".join("%.0f" % x for x in v.get("B_blocks", [])))
print(s + " | %d blocks, %d flagged, %s s" % (d["blocks"], d["blocks_bad"], d["wall_s"]))' 2>&1 | cut -c1-700)"
  [ "$DRY" = 1 ] || log "RESULT fn4 sdma3 $nm blocks (tok/s, ms/step, acceptance): $(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("label") == sys.argv[2]: print("%s %s tok/s %s ms/step acc %s;" % (r["arm"], r.get("tps"), r.get("ms"), (r.get("mb") or {}).get("acceptance")), end=" ")' $O/$TAG-blocks.jsonl $TAG-$nm 2>&1 | cut -c1-300)"
}
log "RESULT fn4 sdma3-fn-sdma: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) exact wire + SPHC + MTPFLAT + AR_SDMA=0 declared; T (traced byte check) then S (speed) | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 sdma3: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 sdma3: VIRHE image $IMG missing"; exit 1; }
  [ -e "$Y" ] && [ -n "$TL" ] && [ -e $TL/abblocks2.py ] && [ -e $MBT ] && [ -e $LD ] && [ -e $GL ] && [ -e $BTP ] && [ -n "$GS" ] && [ -e $D/mb.py ] || { log "RESULT fn4 sdma3: VIRHE tools missing"; exit 1; }
  grep -q 'RADIANCE_FN4_MTPFLAT' $Y && grep -q 'RADIANCE_FN4_AR_SDMA' $Y || { log "RESULT fn4 sdma3: VIRHE $YN does not pass RADIANCE_FN4_MTPFLAT / _AR_SDMA"; exit 1; }
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
[ "$DRY" != 1 ] && [ "${FREEG:-0}" -lt 12 ] && { log "RESULT fn4 sdma3: VIRHE only ${FREEG:-?} GB free on / -- not started"; exit 1; }
# gate: 215's engine-path selftest, when it ran, must show every SDMA byte equal (else this item would only measure a bug)
E2=$O/sdma2-eng.txt
if [ "$DRY" != 1 ] && [ -e $E2 ]; then
  BADE=$(python3 -c 'import json,sys
b = 0; n = 0
for l in open(sys.argv[1], errors="replace"):
    if l.startswith("SDMAENG {"):
        d = json.loads(l[8:]); n += 1
        b += d["bad_vs_cu"] + d["bad_vs_fold"] + d["bad_xstream"] + (1 if d["failed"] or d["ctx"] is not True else 0)
print(b if n else -1)' $E2 2>/dev/null || echo -2)
  [ "$BADE" = 0 ] || { log "RESULT fn4 sdma3: VIRHE 215's engine-path selftest ($E2) shows $BADE bad / failed (-1: no SDMAENG line) -- not started"; exit 1; }
  log "RESULT fn4 sdma3: 215's engine-path selftest clean ($E2)"
fi
# ---- T: the traced byte check
export FN4BT=/stats/bt-sp FN4BTP=6144 FN4BTL=4096
restart $TAG-T || exit 1
nudge
[ "$DRY" = 1 ] || log "RESULT fn4 sdma3 T engine notes: $(grep -E 'fn4 SP-hc|tracer ON|fn4 SDMA-AR' $O/start-$TAG-T.log | sed 's/^[A-Z] //' | cut -c1-200 | tr '\n' ' ' | cut -c1-600)"
sw "$SB" "SB" && glr a1; alive "T a1"
sw "$SD" "SD" && glr sd; alive "T sd"
sw "$SB" "SB" && glr a2; alive "T a2"
lc a1 sd "ar_sdma 1 long, vs ar_sdma 0"; lc a1 a2 "long control a2 vs a1"
run "docker logs $CN > $O/$TAG-T-engine.log 2>&1"
down
if [ "$DRY" != 1 ]; then
  timeout 2400 python3 -I $BTP analyze $STATSDIR/bt-sp --labels sd --json $O/$TAG-bitrace.json > $O/$TAG-bitrace.txt 2>&1
  grep -E '^BITRACE summary' $O/$TAG-bitrace.txt | while IFS= read -r l; do log "RESULT fn4 sdma3 ar_sdma byte check $(echo "$l" | cut -c1-700)"; done
  grep -q '^BITRACE summary' $O/$TAG-bitrace.txt && find $STATSDIR/bt-sp -maxdepth 1 -name 'bt-r*.bin' -delete
  grep -q '^BITRACE summary' $O/$TAG-bitrace.txt || log "RESULT fn4 sdma3 ar_sdma byte check: VIRHE no summary ($(tail -2 $O/$TAG-bitrace.txt | tr '\n' ' ' | cut -c1-300))"
fi
# ---- S: speed, untraced
export FN4BT="" FN4BTP="" FN4BTL=""
restart $TAG-S || exit 1
nudge
[ "$DRY" = 1 ] || log "RESULT fn4 sdma3 S engine notes: $(grep -E 'fn4 SDMA-AR|pipelined' $O/start-$TAG-S.log | sed 's/^[A-Z] //' | cut -c1-200 | tr '\n' ' ' | cut -c1-500)"
sw "$SB" "SB"
# warm-up: one fresh 64k prefill before any block (206: the first prefill after a start runs cold)
pcr; run "(cd $D && timeout 900 python3 ds41_tests.py prefill --depths 65536 --cpt 3.6 --calibrate 0 --reps 1 > $O/$TAG-warm-pf.txt 2>&1)"; alive "S warm-up"
abr pf "$SB" "$SD" --no-mb --pf-depths 65536,131072; alive "S prefill ar_sdma 0 vs 1"
abr dec "$SB" "$SD" --mbt2 $MBT --gens 1 --phase c16 --warm 8; alive "S c16 ar_sdma 0 vs 1"
run "docker logs $CN > $O/$TAG-S-engine.log 2>&1"
[ "$DRY" = 1 ] || log "RESULT fn4 sdma3 engine: $(grep -cE 'fn4: switches .* applied' $O/$TAG-S-engine.log) applied changes, $(grep -cE 'fn4: .* refused' $O/$TAG-S-engine.log) refused"
[ "$DRY" = 1 ] || log "RESULT fn4 sdma3 engine: SDMA passes left untaped (said once a rank): $(grep -c 'fn4 SDMA all-reduce' $O/$TAG-S-engine.log); context lines: $(grep -m2 'fn4 SDMA-AR' $O/$TAG-S-engine.log | cut -c1-160 | tr '\n' ' ')"
down
run "gzip -f $O/$TAG-T-engine.log $O/$TAG-S-engine.log $O/start-$TAG-T.log $O/start-$TAG-S.log 2>/dev/null; true"
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
log "RESULT fn4 sdma3-fn-sdma: done"
