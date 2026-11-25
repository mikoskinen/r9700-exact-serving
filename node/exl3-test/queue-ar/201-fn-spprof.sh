#!/bin/bash
# 201-fn-spprof.sh (queue item, 12.11.2026, queue-ar; gpu02, all 4 cards, every radiance container DOWN; ~10 min) -- SP-hc's
# SEPARATING PROFILE (198 S: SP-hc +2.8 / +3.6 / +3.2 % prefill at 16k / 64k / 128k vs the audit's -10..-25 %; the user asks
# whether the implementation is at fault). One start of 198 S's config (image radiance-local:c16-x3sp2, exact wire,
# RADIANCE_FN4_SPHC=1 + MTPFLAT declared, BI set S0 on) + --profile-ops with RADIANCE_PROFILE_EVERY 1 (a cumulative per-op
# table a rank a step). Four windows, each a fresh 64k prefill (the prefix cache reset, ds41_tests prefill, max_tokens 8):
# hc_sp 0, 1, 0, 1. tools/spprof.py subtracts each rank's tables around a window: the collectives an op (AR vs RS + AG), the
# piece hc_read vs SP's kc-pinned quarter twin (us a call, ms a 2048-row piece), hc_write, the hc_codes redo, device / lane-wait
# ms by class, the top waiting ops (the AG wait shows on the op after it), and the twin's calls an instance (engagement).
# Profiled steps synchronise: per-op ms and waits are the reading; the speeds are 198 S's.
# DRYRUN=1 bash 201-fn-spprof.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn4; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
YN=fn-tp4zt.yaml; for Y in "$H/$YN" "$H/../$YN" "$HOME/gpu/ds41/exl3-test/queue-ar/$YN"; do [ -e "$Y" ] && break; done
CN=radiance-fntest3; URL=http://127.0.0.1:8090; export DS41_URL=$URL
TL=; for f in "$H/tools" "$H/../tools" "$HOME/gpu/ds41/exl3-test/queue-ar/tools"; do [ -e "$f/heatsoak.sh" ] && { TL=$f; break; }; done
IMG=${X201_IMAGE:-radiance-local:c16-x3sp2}
C3=GPU-dbba5c851cfef7b3; C83=GPU-4d61f52cafe37588; C86=GPU-5943fea82aeead65; C03=GPU-27d4920317825fce
TAG=fn4-spprof
export FNIMAGE=$IMG FNMODELDIR=${X201_X3_DIR:-$HOME/models/flashnext-exl3} FNMODEL=${X201_X3_MODEL:-qwen3.8-flash-next-exl3.rad}
export TP=4 CARDS=$C3,$C83,$C86,$C03 PLACE=all_vram MML=262144 ROPE="" MAXSEQS=16 MNBT=4096 NST=3 WIREKB=128 PMIN="" PMINB=""
EXTRA0="--prefill-span 2 --prefix-resume nearest --prefill-interleave 0"
export EXTRA="$EXTRA0" AQLSTAMP="" AQLCARDS="" STATSDIR=$O/stats201
export FN4BT="" FN4BTP="" FN4BTL="" FN4X3DUAL=1 FN4NR=1 FN4GI=1 FN4FD="" FN4NW=1 FN4SW="" FN4WV=1 FN4RV=11 FN4AL="" FN4UM=5 FN4SS=1 FN4CH="" FN4XM=1 FN4SPHC=1 FN4X3C=1 FN4QSA="" FN4P2X="" FN4MP="" FN4X3GD=1 FN4X3DZ=1 PROFEVERY=1 FN4MTPF=1
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
    [ $n -ge 10 ] && { : > $O/$TAG.abort; log "RESULT fn4 201: VIRHE temperature guard ($hot) -> blocks stopped"; return 0; }
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
  [ $ok = 1 ] || { log "RESULT fn4 201 restart-$tag: VIRHE not healthy after $(( $(date +%s) - t0 )) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-400)"; return 1; }
  log "RESULT fn4 201 restart-$tag: healthy in $(( $(date +%s) - t0 )) s; $(grep -m1 -E '^I kv: ' $S | cut -c3-160); fn4 state $(curl -s -m 5 $URL/admin/fn4 | cut -c1-300)"
}
probe8() {   # probe8 TAG -> "ms tok/s" of a 45 s mb c8
  [ "$DRY" = 1 ] && { echo "+ (cd $D && timeout 400 python3 mb.py --tag $1 --phases c8 --dur 45 --warm 10 > $O/$1-c8.txt)" >&2; echo "25.6 744"; return 0; }
  (cd $D && timeout 400 python3 mb.py --tag $1 --phases c8 --dur 45 --warm 10 > $O/$1-c8.txt 2>&1)
  grep '^MB ' $O/$1-c8.txt 2>/dev/null | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[3:]); print(d.get("ms_decode_step"), d.get("agg_dec_tps"))' 2>/dev/null || echo "0 0"
}
setsw() {   # setsw JSON: queue switches (they land at the next step release)
  run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"
}
GS=; for f in "$H/tools/greedyset.py" "$H/../tools/greedyset.py" "$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py"; do [ -z "$GS" ] && [ -e "$f" ] && GS=$f; done
alive() {   # alive WHAT: the server still answers, else logs + down + exit
  [ "$DRY" = 1 ] && return 0
  local code; code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' $URL/health)
  [ "$code" = 200 ] && return 0
  docker logs $CN > $O/$TAG-dead-engine.log 2>&1
  log "RESULT fn4 201: VIRHE the server is not healthy after $1 (/health $code; last E lines: $(grep -E '^E |trap|abort|HSA_STATUS' $O/$TAG-dead-engine.log | tail -3 | tr '\n' ' ' | cut -c1-400)) -> logs saved, down"
  down; exit 1
}
LD=$TL/fn-loadid.py
PFD=${X201_PFD:-16384,65536}
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
swcheck() {   # swcheck WHAT SWITCHES_JSON: every switch named there has that value now
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4: $2"; return 0; }
  local g; g=$(curl -s -m 10 $URL/admin/fn4)
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); w=json.loads(sys.argv[2]); sys.exit(0 if all(d["switches"].get(k) == v for k, v in w.items()) else 1)' "$g" "$2" 2>/dev/null \
    || { log "RESULT fn4 201 $1: VIRHE the switches are not $2 ($(echo "$g" | cut -c1-400))"; return 1; }
}
pcreset() { [ "$DRY" = 1 ] && { echo "+ POST $URL/reset_prefix_cache ($1)"; return 0; }
  local c; c=$(curl -s -m 60 -o /dev/null -w '%{http_code}' -X POST -d '' $URL/reset_prefix_cache 2>/dev/null)
  [ "$c" = 200 ] || log "RESULT fn4 201 $1 prefix cache reset: HTTP $c"; }
c16blk() { [ -e $O/$TAG.abort ] && return 0; run "(cd $D && timeout 400 python3 mb.py --tag $TAG-$1 --phases c16 --dur 20 --warm 6 > $O/$TAG-$1-c16.txt 2>&1)"; }
pfblk() { [ -e $O/$TAG.abort ] && return 0; run "(cd $D && timeout 600 python3 ds41_tests.py prefill --depths $PFD --cpt 3.6 --calibrate 0 --reps 1 > $O/$TAG-$1-pf.txt 2>&1)"; }
gset() { run "(cd $D && timeout 900 python3 $GS $URL $O/$TAG-gs-$1.jsonl --conc 1 --max-tokens 192 > $O/$TAG-gs-$1.txt 2>&1)"; }
copyt() {   # copyt NAME: a FRESH 33k copy (the prefix cache reset first), greedy; output $O/copy-$TAG-NAME-run1.txt
  pcreset copy-$1
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset 0 --tag=-$TAG-$1 --outdir $O > $O/$TAG-$1-copy.txt 2>&1)"
}
cmpf() {   # cmpf FILE_A FILE_B LABEL
  if [ "$DRY" = 1 ]; then echo "+ cmp $1 $2"
  elif [ -s $1 ] && [ -s $2 ]; then
    if cmp -s $1 $2; then log "RESULT fn4 201 identity $3: IDENTICAL ($(wc -c < $2) bytes)"
    else log "RESULT fn4 201 identity $3: DIFFERENT from byte $(cmp $1 $2 2>&1 | grep -oE 'byte [0-9]+' | head -1)"; fi
  else log "RESULT fn4 201 identity $3: not compared (missing $1 or $2)"; fi
}
mpair() {   # mpair N: moe_pair N, landed (an epoch bump, a nudge, checked)
  setsw "{\"moe_pair\": $1, \"epoch_bump\": 1}"; nudge; swcheck "moe_pair $1" "{\"moe_pair\": $1}"
}
abpf() {   # abpf NAME A_JSON B_JSON: prefill-only blocks A B B A (one round = 2 blocks a side)
  [ -e $O/$TAG.abort ] && return 0
  run "timeout 1800 python3 -I $TL/abblocks2.py --url $URL --mb-dir $D --label $TAG-$1 --rounds 1 --a '$2' --b '$3' --pf-depths $PFD --no-mb --names off,$1 --out $O/$TAG-blocks.jsonl > $O/$TAG-$1.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT fn4 201 $1 prefill (B/A of prefill tok/s, A B B A, 2 blocks a side): $(grep '^ABB ' $O/$TAG-$1.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
for k, v in (d.get("pf") or {}).items():
    print("%s B/A %.4f %s (A %s | B %s tok/s);" % (k, v["ratio"] or 0, v.get("sep"), " ".join("%.0f" % x for x in v.get("A_blocks", [])), " ".join("%.0f" % x for x in v.get("B_blocks", []))), end=" ")
print("%d blocks, %d flagged, %s s" % (d["blocks"], d["blocks_bad"], d["wall_s"]))' 2>&1 | cut -c1-700)"
}
GS=; for f in "$H/tools/greedyset.py" "$H/../tools/greedyset.py" "$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py"; do [ -z "$GS" ] && [ -e "$f" ] && GS=$f; done
SPP=$TL/spprof.py
S0='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 3, "bi_seg": 2, "att_bi": 1, "sched_bi_chunk": 4, "hc_sp": 0, "mtp_flat": 0}'
S1='{"exl3_bi": 0, "exl3_dual": 1, "hc_bi": 3, "bi_seg": 2, "att_bi": 1, "sched_bi_chunk": 4, "hc_sp": 1, "mtp_flat": 0}'
pcr() { run "curl -s -m 60 -o /dev/null -X POST -d '' $URL/reset_prefix_cache"; }
sw() { setsw "$(echo "$1" | sed 's/}$/, "epoch_bump": 1}/')"; nudge; swcheck "$2" "$1"; }
lines() { [ "$DRY" = 1 ] && { echo 0; return 0; }; docker logs $CN 2>&1 | wc -l; }
WINS=""
win() {   # win LABEL SWITCHES: switch, mark, a fresh 64k prefill, mark
  sw "$2" "$1" || return 1
  local lo hi pt tt
  lo=$(lines); pcr
  run "(cd $D && timeout 900 python3 ds41_tests.py prefill --depths 65536 --cpt 3.6 --calibrate 0 --reps 1 > $O/$TAG-$1-pf.txt 2>&1)"
  nudge; hi=$(lines)
  [ "$DRY" = 1 ] && { echo "+ window $1 lines lo..hi"; return 0; }
  pt=$(grep -o '"prompt_tokens": *[0-9]*' $O/$TAG-$1-pf.txt | head -1 | grep -o '[0-9]*$'); tt=$(grep -o '"ttft": *[0-9.]*' $O/$TAG-$1-pf.txt | head -1 | grep -o '[0-9.]*$')
  WINS="$WINS --win $1:$lo:$hi:${pt:-0}:${tt:-}"
  log "RESULT fn4 201 window $1 ($2): engine-log lines $lo..$hi, prompt ${pt:-?} tokens, profiled TTFT ${tt:-?} s"
}
log "RESULT fn4 201-fn-spprof: start IMAGE=$IMG ($(docker image inspect -f '{{.Id}}' $IMG 2>/dev/null | cut -c8-19)) 198 S config + --profile-ops (every step); 64k fresh prefill windows hc_sp 0 / 1 / 0 / 1 | power: $( [ "$DRY" = 1 ] && echo dry || caps)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn4 201: VIRHE a radiance container is running -- the cards are not free"; exit 1; }
  docker image inspect $IMG > /dev/null 2>&1 || { log "RESULT fn4 201: VIRHE image $IMG missing"; exit 1; }
  [ -e "$Y" ] && [ -n "$TL" ] && [ -e $SPP ] && [ -e $TL/opprof.py ] && [ -e $D/ds41_tests.py ] || { log "RESULT fn4 201: VIRHE tools missing"; exit 1; }
  grep -q RADIANCE_PROFILE_EVERY $Y || { log "RESULT fn4 201: VIRHE $YN does not pass RADIANCE_PROFILE_EVERY"; exit 1; }
  mkdir -p $O $STATSDIR
  find $O -maxdepth 1 -name "$TAG.abort" -delete; find $O -maxdepth 1 -name "$TAG.tstop" -delete
  find $O -maxdepth 1 \( -name "$TAG-*.json" -o -name "$TAG-*.txt" \) -delete
  trap 'touch $O/$TAG.tstop 2>/dev/null; down' EXIT
  trap 'exit 143' TERM INT
  tguard & TGPID=$!
else
  echo "+ guards; tguard &"; TGPID=""
fi
export EXTRA="$EXTRA0 --tp-wire exact --profile-ops"
restart $TAG || exit 1
nudge
[ "$DRY" = 1 ] || log "RESULT fn4 201 engine notes: $(docker logs $CN 2>&1 | grep -E 'fn4 SP-hc|pipelined' | sed 's/^[A-Z] //' | cut -c1-200 | tr '\n' ' ' | cut -c1-500)"
win A0 "$S0"; alive "window A0"
win B1 "$S1"; alive "window B1"
win A2 "$S0"; alive "window A2"
win B3 "$S1"; alive "window B3"
run "docker logs $CN > $O/$TAG-engine.log 2>&1"
down
if [ "$DRY" != 1 ]; then
  timeout 900 python3 -I $SPP $O/$TAG-engine.log $WINS --json $O/$TAG.json > $O/$TAG.txt 2>&1
  python3 - $O/$TAG.json <<'PY' | while IFS= read -r l; do log "RESULT fn4 201 $l"; done
import json, sys
R = json.load(open(sys.argv[1]))
for lab, w in R.items():
    rk = w["ranks"]
    if not rk: print("%s: no tables" % lab); continue
    r0 = rk.get("0") or rk[sorted(rk)[0]]
    pc = max(1, w["pieces"])
    co = "; ".join("%s %d calls %.1f ms/piece %.0f us wait %.1f ms" % (k, v["calls"], v["ms_per_piece"], v["us_call"], v["wait_ms"]) for k, v in r0["coll"].items())
    rp, rs, hw, cd = r0["hc_read_piece"], r0["hc_read_sp"], r0["hc_write"], r0["hc_codes"]
    devmax = max(v["dev_ms"] for v in rk.values())
    print("%s rank0: TTFT %s s, %d pieces, steps %s | dev %.0f ms (max rank %.0f) wait %.0f ms | %s | hc_read piece %d calls / %d inst %.1f ms/piece %.0f us | SP quarter %d calls / %d inst (calls an inst %s) %.1f ms/piece %.0f us | hc_write %.1f ms/piece %.0f us | hc_codes %d calls %.1f ms/piece | wait by class %s | top waits %s" % (
        lab, w["ttft_s"], w["pieces"], w["steps"].get("0"), r0["dev_ms"], devmax, r0["wait_ms"], co, rp["calls"], rp["inst"], rp["ms_per_piece"], rp["us_call"],
        rs["calls"], rs["inst"], rs["calls_per_inst_min_max"], rs["ms_per_piece"], rs["us_call"], hw["ms_per_piece"], hw["us_call"], cd["calls"], cd["ms_per_piece"],
        json.dumps(r0["cls_wait_ms"]), "; ".join("%s %.1f ms (%d calls)" % (t["op"], t["wait_ms"], t["calls"]) for t in r0["top_wait"][:3])))
PY
fi
run "touch $O/$TAG.tstop"; [ -n "$TGPID" ] && wait $TGPID 2>/dev/null
log "RESULT fn4 201-fn-spprof: done ($O/$TAG.txt, $O/$TAG.json)"
