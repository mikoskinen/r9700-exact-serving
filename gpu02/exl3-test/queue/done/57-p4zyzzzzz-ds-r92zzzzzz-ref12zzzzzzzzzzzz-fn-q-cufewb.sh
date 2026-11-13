#!/bin/bash
# 229b-fn-cufew.sh (13.11.2026; 229 RERUN: its pre-check summed "bad" over every perfar4 row, but the SP rows are unchecked in the harness
# (checked false, bad = all elements, the same at the default budget: 204 / 228) and ti8 is lossy -- now only checked exact rows count; at 1 block
# every checked exact row was bad 0). 229-fn-cufew.sh (queue item, 13.11.2026, queue-ar; ~25 min) -- 227's trend the other way (coordinator: lane blocks 8 / 16 / 24 cost prefill
# -1.4 / -3.5 / -5.4 %, threads 512 ~0): FEWER lane blocks than the tunable's 4 -- ar_laneblocks 2 / 1 and 2 x 512 threads (fn4-x3-sp3: the
# switch takes 1 / 2 now), radiance-local:c16-x3sp23. First a selftest pre-check (--perfar4 2048x2560 at RADIANCE_FN4_AR_LANE_BLOCKS 1 / 2: the
# lane rows' fold on every rank, bad 0), then one engine start: warm-up, prefill A B B A 64k / 128k base vs lb2 / lb1 / lb2+lt512, and a long
# greedy identity (4 prompts) under lb1 against base. Below: 227's header.
# 227-fn-cubudget.sh (queue item, 13.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~40 min) -- THE COMM LANE'S CU BUDGET
# in the pipelined prefill (coordinator 13.11. after 209: the RS / AG lane is the critical path, lane waits RS 2803 / AG 347 ms a 54k window,
# compute 2x slower beside it -- gemm_nt 66 -> 132 us). Exact (the shards and each element's rank fold do not depend on blocks / threads).
# radiance-local:c16-x3sp21, S9 + hc_sp 2 + ar_xpush 1 (the prefill config), switches ar_laneblocks (0 = the tunable's 4, else 8 / 16 / 24) and
# ar_lanethreads (0 = tunable, 512 / 1024): START 1 (unprofiled, the log followed): a warm-up 64k prefill, then abblocks2 --no-mb A B B A
# prefill 64k / 128k (fresh, nonce first) for base vs blocks 8 / blocks 16 / blocks 24 / threads 512 -- wall-clock TTFT and tok/s; START 2
# (--profile-ops every step, the log followed, windows by epoch lines): one fresh 64k window an arm (base, 8, 16, 24, 512) -> spprof.py lane
# waits (collective / gated residual) + opwin us a call of reduce_scatter_rows / all_gather_rows / gemm_nt / hc_read kc_rows.
# DRYRUN=1 bash 227-fn-cubudget.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4zu.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/heatsoak.sh" ] && { TL=$f; break; }; done
IMG=${X229_IMAGE:-radiance-local:c16-x3sp23}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-cfb
export FNIMAGE=$IMG FNMODELDIR=${X229_X3_DIR:-$HOME/models/flashnext-exl3} FNMODEL=${X229_X3_MODEL:-qwen3.8-flash-next-exl3.rad}
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
EXTRA0="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0"
export EXTRA="$EXTRA0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats229b
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
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 229b: VIRHE temperature guard ($hot) -> blocks stopped"; return 0; }
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
  [ $ok = 1 ] || { log "RESULT fn4 229b restart-$tag: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
  log "RESULT fn4 229b restart-$tag: healthy in $(( $(date +%s) - t0 )) s; $(grep -m1 -E '^I kv: ' $S | cut -c3-160); fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300)"
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
    || { log "RESULT fn4 229b $1: VIRHE the switches are not $2 ($(echo "$g" | cut -c1-400))"; return 1; }
}
alive() {   # alive STEP: /health 200, else log it (the T arm then yields no answers)
  [ "$DRY" = 1 ] && { echo "+ curl -m 5 $URL/health == 200"; return 0; }
  local code; code=$(curl -s -m 5 -o /dev/null -w %{http_code} $URL/health)
  [ "$code" = 200 ] && return 0
  log "RESULT fn4 229b $1: VIRHE the server is not healthy (/health $code; $(docker inspect -f {{.State.Status}} exit {{.State.ExitCode}} $CN 2>/dev/null || echo gone))"; return 1
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
lc() { [ "$DRY" = 1 ] && { echo "+ cmp $1 $2"; return 0; }; log "RESULT fn4 229b text $3: $(timeout 300 python3 -I $LD cmp --url $URL --label "$3" $O/$TAG-$1.jsonl $O/$TAG-$2.jsonl 2>&1 | grep -m1 '^CMPLINE ' | cut -c9- | cut -c1-400)"; }
abr() {   # abr NAME A B (extra abblocks2 args...)
  local nm=$1 A=$2 B=$3; shift 3
  run "timeout 2400 python3 -I $TL/abblocks2.py --url $URL --mb-dir $D --label $TAG-$nm --rounds 1 --a '$A' --b '$B' --names a,b --out $O/$TAG-blocks.jsonl $* > $O/$TAG-$nm.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 229b $nm (A B B A): $(grep '^ABB ' $O/$TAG-$nm.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=4: "-" if v is None else format(v, ".%df" % n)
s = "decode tok/s B/A %s %s ms/step B/A %s (A %s | B %s)" % (f(d["tps_ratio"]), (d.get("sep") or {}).get("tps"), f(d.get("ms_ratio")), " ".join("%.1f" % x for x in d.get("A_blocks", [])), " ".join("%.1f" % x for x in d.get("B_blocks", []))) if d.get("rounds") else ""
for k, v in (d.get("pf") or {}).items():
    s += " | prefill %s B/A %s %s (A %s | B %s)" % (k, f(v["ratio"]), v.get("sep"), " ".join("%.0f" % x for x in v.get("A_blocks", [])), " ".join("%.0f" % x for x in v.get("B_blocks", [])))
print(s + " | %d blocks, %d flagged, %s s" % (d["blocks"], d["blocks_bad"], d["wall_s"]))' 2>&1 | cut -c1-700)"
  [ "$DRY" = 1 ] || log "RESULT fn4 229b $nm blocks (tok/s, ms/step, acceptance): $(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("label") == sys.argv[2]: print("%s %s tok/s %s ms/step acc %s;" % (r["arm"], r.get("tps"), r.get("ms"), (r.get("mb") or {}).get("acceptance")), end=" ")' $O/$TAG-blocks.jsonl $TAG-$nm 2>&1 | cut -c1-300)"
}
log "RESULT fn4 229b-fn-cufewb: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) exact wire + SPHC declared; prefill lane FEWER blocks (2 / 1 / 2 x 512) A B B A 64k / 128k + identity, selftest pre-check | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 229b: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 229b: VIRHE image $IMG missing"; exit 1; }
  [ -e "$Y" ] && [ -n "$TL" ] && [ -e $TL/abblocks2.py ] && [ -e $MBT ] && [ -e $LD ] && [ -e $GL ] && [ -e $BTP ] && [ -n "$GS" ] && [ -e $D/mb.py ] && [ -e $TL/spprof.py ] && [ -e $TL/opwin.py ] && [ -e $D/ds41_tests.py ] || { log "RESULT fn4 229b: VIRHE tools missing"; exit 1; }
  grep -q 'RADIANCE_FN4_FORMCTR' $Y && grep -q 'RADIANCE_LOG_STEPS' $Y && grep -q 'RADIANCE_FN4_MTPFLAT' $Y || { log "RESULT fn4 229b: VIRHE $YN does not pass RADIANCE_FN4_MTPFLAT / _FORMCTR / RADIANCE_LOG_STEPS"; exit 1; }
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
[ "$DRY" != 1 ] && [ "${FREEG:-0}" -lt 12 ] && { log "RESULT fn4 229b: VIRHE only ${FREEG:-?} GB free on / -- not started"; exit 1; }
B0='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 3, "arw_rows": 16, "sched_bi_chunk": 4, "hc_readv": 331, "hc_sp": 2, "ar_xpush": 1, "ar_laneblocks": 0, "ar_lanethreads": 0}'
mk() { python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d[sys.argv[2]]=int(sys.argv[3]); print(json.dumps(d))' "$B0" "$1" "$2"; }
pfab() {   # pfab NAME A B: prefill A B B A at 64k / 128k
  run "timeout 2400 python3 -I $TL/abblocks2.py --url $URL --mb-dir $D --label $TAG-$1 --rounds 1 --a '$2' --b '$3' --pf-depths 65536,131072 --no-mb --names a,b --out $O/$TAG-blocks.jsonl > $O/$TAG-$1.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 229b $1 prefill (B/A of prefill tok/s, A B B A): $(grep '^ABB ' $O/$TAG-$1.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
for k, v in (d.get("pf") or {}).items():
    print("%s B/A %.4f %s (A %s | B %s tok/s);" % (k, v["ratio"] or 0, v.get("sep"), " ".join("%.0f" % x for x in v.get("A_blocks", [])), " ".join("%.0f" % x for x in v.get("B_blocks", []))), end=" ")
print("%d blocks, %d flagged" % (d["blocks"], d["blocks_bad"]))' 2>&1 | cut -c1-600)"
}
# ---------------- selftest pre-check: the lane rows at 1 / 2 blocks fold on every rank
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
for lb in 1 2; do
  f=$O/229b-ar-b$lb.txt
  if [ "$DRY" = 1 ]; then echo "+ docker run ... RADIANCE_FN4_AR_LANE_BLOCKS=$lb r4d_selftest --perfar4 2048x2560 100 > $f"; continue; fi
  timeout -k 15 600 docker run --rm --init --name fn4-229b-$$ --network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
    -e ROCR_VISIBLE_DEVICES=$C3,$C83,$C86,$C03 -e RADIANCE_FN4_AR_LANE_BLOCKS=$lb --entrypoint /opt/radiance/bin/r4d_selftest $IMG \
    /opt/radiance/share/radiance/kernels/libr4d.so --perfar4 2048x2560 100 > $f 2>&1
  docker rm -f fn4-229b-$$ > /dev/null 2>&1
  log "RESULT fn4 229b selftest lane blocks $lb: $(python3 -c 'import json,sys
rows = [json.loads(l[4:]) for l in open(sys.argv[1]) if l.startswith("AR4 ")]
lane = [r for r in rows if "lane" in r["op"]]
ex = [r for r in rows if r.get("checked") and "ti8" not in r["op"] and "lossy" not in r["op"]]
print("%d rows, %d checked exact, bad %d; %s" % (len(rows), len(ex), sum(r.get("bad", 0) for r in ex), "; ".join("%s %.1f us" % (r["op"], r["us"]) for r in lane)))' $f 2>&1 | cut -c1-700)"
  python3 -c 'import json,sys; rows=[json.loads(l[4:]) for l in open(sys.argv[1]) if l.startswith("AR4 ")]; sys.exit(1 if sum(r.get("bad", 0) for r in rows if r.get("checked") and "ti8" not in r["op"] and "lossy" not in r["op"]) else 0)' $f \
    || { log "RESULT fn4 229b: VIRHE lane blocks $lb folds differ in the selftest -- engine part skipped"; exit 1; }
done
# ---------------- START 1: unprofiled A / B
restart $TAG-S || exit 1
if [ "$DRY" = 1 ]; then CAPPID=""; else (docker logs -f -t $CN 2>&1 | gzip -1 > $O/$TAG-S-engine.log.gz) & CAPPID=$!; fi
nudge
sw "$B0" "B0" || { down; exit 1; }
pcr; run "(cd $D && timeout 900 python3 ds41_tests.py prefill --depths 65536 --cpt 3.6 --calibrate 0 --reps 1 > $O/$TAG-warm-pf.txt 2>&1)"; alive "warm-up"
pfab lb2 "$B0" "$(mk ar_laneblocks 2)"; alive "lb2"
pfab lb1 "$B0" "$(mk ar_laneblocks 1)"; alive "lb1"
LB2T='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 5, "bi_seg": 2, "att_bi": 3, "arw_rows": 16, "sched_bi_chunk": 4, "hc_readv": 331, "hc_sp": 2, "ar_xpush": 1, "ar_laneblocks": 2, "ar_lanethreads": 512}'
pfab lb2t512 "$B0" "$LB2T"; alive "lb2t512"
# identity: the long greedy set (4 prompts, 64 tokens) under lb1 against base
glr() { pcr; run "(cd $D && GL_N=4 timeout 1800 python3 $GL $URL $O/$TAG-$1.jsonl --conc 1 --max-tokens 64 > $O/$TAG-$1.txt 2>&1)"; }
sw "$B0" "id base" && glr Lbase; alive "id base"
sw "$(mk ar_laneblocks 1)" "id lb1" && glr Llb1; alive "id lb1"
sw "$(mk ar_laneblocks 2)" "id lb2" && glr Llb2; alive "id lb2"
[ "$DRY" = 1 ] || log "RESULT fn4 229b text long lb1 vs base: $(timeout 300 python3 -I $LD cmp --url $URL --label lb1 $O/$TAG-Lbase.jsonl $O/$TAG-Llb1.jsonl 2>&1 | grep -m1 '^CMPLINE ' | cut -c9- | cut -c1-300)"
[ "$DRY" = 1 ] || log "RESULT fn4 229b text long lb2 vs base: $(timeout 300 python3 -I $LD cmp --url $URL --label lb2 $O/$TAG-Lbase.jsonl $O/$TAG-Llb2.jsonl 2>&1 | grep -m1 '^CMPLINE ' | cut -c9- | cut -c1-300)"
down
[ -n "$CAPPID" ] && { for i in $(seq 1 60); do kill -0 $CAPPID 2>/dev/null || break; sleep 2; done; kill $CAPPID 2>/dev/null; }
[ "$DRY" = 1 ] || log "RESULT fn4 229b engine S: $(zcat $O/$TAG-S-engine.log.gz | grep -cE 'fn4: switches .* applied') applied, $(zcat $O/$TAG-S-engine.log.gz | grep -cE 'fn4: .* refused') refused; E lines $(zcat $O/$TAG-S-engine.log.gz | grep -cE ' E ')"
run "gzip -f $O/start-$TAG-S.log 2>/dev/null; true"
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
log "RESULT fn4 229b-fn-cufewb: done"
