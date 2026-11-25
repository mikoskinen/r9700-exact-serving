#!/bin/bash
# after-sweep.sh (6.11.2026) -- the 3.11. DeepSeek profiling / 1M / capacity steps for the EXL3 2.25 bpw model, plus the coordinator's
# additions (the PR author's TP4 serve configuration, profiled with tier counters; Engram on NVMe). Run by chain-full.sh after
# sweep-exl3.sh sweep-exl3c.cfg -- production down, all four cards free -- ONLY when runs/after-sweep.reviewed exists. chain-full.sh
# runs window.sh PHASES=down,produp after this script; this script leaves the EXL3 server down.
# Outputs runs/after-sweep/<step>/; one RESULT line per measurement in runs/ds41-exl3.log with the 3.11. names (live0-run, prof0-run,
# profiling decode step, profpf-run, 1M probe, cc_sim fable-1m, 1M probe extra); compare-after.py prints EXL3 vs 3.11. UD-IQ1_S.
#
#   DRYRUN=1           print the commands; runs nothing and writes nothing (no log line, no directory); everything to stdout
#   STEPS=a,b,...      order, default live0,prof0,prcfg,onem,nvme,cap,profpf (onem includes fable-1m and, if time is left, the
#                      ~700k extra; cap includes u12, then the RAM 140k reference and u16 if time is left)
#   BUDGET_MIN=120     wall clock from start; a step whose estimate no longer fits is skipped with a RESULT line
#   touch runs/sweep.stop  -> stop before the next step (chain-full.sh then restores production)
#   AS_GUARD_MS=24     1-user guard of the unprofiled instances (2.25 bpw fast 19.36 ms on 6.11.; MML 1M was +3 % on 3.11.;
#                      the slow state is ~2x). exl3-gate.sh restarts once; SLOW-after-retry -> VIRHE, measurements skipped
#   PROF_GUARD_X=1.85  profiled instances (prof0, profpf): rank-0 DEVICE op ms per step during a fastcheck must be <= X times
#                      live0's guarded unprofiled ms/step (3.11.: prof0 op 23.9-24.6 ms = 1.5x the unprofiled 16.2; the
#                      instance with ~2x memory-bound GEMMs, profpf, 36.0 = 2.2x). One restart, then VIRHE + skip
#   NVME_GUARD_MS=36   Engram on NVMe may cost decode time by itself: SLOW-after-retry there is logged and measured anyway
#   PRCFG_NST=0        the PR author's serve line has no --num-speculative-tokens (= auto, DSpark); 0 keeps the profile
#                      comparable with prof0 (NST0). PRCFG_NST=auto for the literal line
#   DEEP=240000        onem.py target of prof0's deep phase (~243k-token prompt + hit turn, under MML 262144)
set -u
T=$HOME/gpu/ds41/exl3-test
. $T/env.sh                                   # T R L C D Q URL + the window defaults (read only)
AS=$R/after-sweep; DRY=${DRYRUN:-0}; CN=radiance-ds41-exl3
F0="--project-directory $C -f $T/ds41-exl3-tp4.yaml"
PF="$F0 -f $T/ds41-exl3-profenv.yaml"         # + RADIANCE_PROFILE_EVERY=${PROFEVERY}
STEPS=${STEPS:-noxp,live0,prof0,prcfg,onem,nvme,profpf}   # 6.11.: noxp first (engine fix A/B); cap (cc_sim u12/u16) dropped, user: "vain 4 cc ei 8"
BUDGET_MIN=${BUDGET_MIN:-200}; AS_GUARD_MS=${AS_GUARD_MS:-24}; PROF_GUARD_X=${PROF_GUARD_X:-1.85}; NVME_GUARD_MS=${NVME_GUARD_MS:-36}
AS_FAST_REF_MS=${AS_FAST_REF_MS:-19.4}; PRCFG_NST=${PRCFG_NST:-0}; DEEP=${DEEP:-240000}
BEST2X="--prefill-span 2 --prefill-interleave 8"
FABLE_ARGS="--duration 900 --warmup 420 --effort medium --ctx-cap 750000 --start-min 100000 --start-max 190000 --no-preage"
T0=$(date +%s); FOLLOW_PID=""; RAN=""; SKIPPED=""

# ---------------------------------------------------------------- plumbing
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
q() { printf ' %q' "$@"; }
run() { if [ "$DRY" = 1 ]; then echo "+$(q "$@")"; else "$@"; fi; }
run_to() { local o=$1; shift; if [ "$DRY" = 1 ]; then echo "+$(q "$@") > $o 2>&1"; else "$@" > "$o" 2>&1; fi; }
in_d() { local tmo=$1 o=$2; shift 2   # harness call in ~/gpu/ds41 (as 3.11.)
  if [ "$DRY" = 1 ]; then echo "+ (cd $D && timeout $tmo$(q "$@") > $o 2>&1)"; else (cd $D && timeout $tmo "$@" > "$o" 2>&1); fi; }
newdir() { if [ "$DRY" = 1 ]; then echo "+ mkdir -p $1"; else mkdir -p "$1"; fi; }
left() { echo $(( BUDGET_MIN * 60 - ($(date +%s) - T0) )); }
stopfile() { [ -e $R/sweep.stop ]; }
fits() { # NAME EST_MIN [RESERVE_MIN] -> 0 run, 1 skip (budget), 2 stop (sweep.stop)
  local n=$1 e=$2 rs=${3:-0}
  if [ "$DRY" = 1 ]; then echo "# step $n: estimate $e min (reserve kept for later steps: $rs min)"; return 0; fi
  stopfile && { log "RESULT after-sweep: sweep.stop found before $n -> stop"; return 2; }
  if [ $(( (e + rs) * 60 )) -gt $(left) ]; then
    log "RESULT after-sweep $n: skipped (estimate $e min + $rs min kept for later steps > $(( $(left) / 60 )) min left of $BUDGET_MIN)"
    SKIPPED="$SKIPPED $n"; return 1; fi
  return 0; }
follow() { # DIR: stream the container log from its start into DIR/engine.log (marks count its lines; survives log rotation)
  if [ "$DRY" = 1 ]; then echo "+ docker logs -f $CN > $1/engine.log 2>&1 &"; return; fi
  docker logs -f $CN > $1/engine.log 2>&1 & FOLLOW_PID=$!; sleep 2; }
unfollow() { if [ -n "$FOLLOW_PID" ]; then kill $FOLLOW_PID 2>/dev/null; wait $FOLLOW_PID 2>/dev/null; FOLLOW_PID=""; fi; }
trap unfollow EXIT
mark() { # PHASE DIR: "phase logline epoch" (3.11. marks.txt) + engine step counters (marks-metrics.txt)
  if [ "$DRY" = 1 ]; then echo "+ mark $1 -> $2/marks.txt (+ /metrics step counters -> $2/marks-metrics.txt)"; return; fi
  sleep 3
  echo "$1 $(wc -l < $2/engine.log) $(date +%s)" >> $2/marks.txt
  echo "$1 $(curl -s -m 5 $URL/metrics | awk '$1 ~ /^radiance:engine_steps_total/ {a=$NF} $1 ~ /^radiance:engine_steps_decode_total/ {b=$NF}
      $1 ~ /^radiance:engine_step_seconds_total/ {c=$NF} $1 ~ /^radiance:engine_step_seconds_decode_total/ {d=$NF} END {print a+0, b+0, c+0, d+0}')" >> $2/marks-metrics.txt; }
gmark() { if [ "$DRY" = 1 ]; then echo "+ mark $1 -> $2/gmarks.txt"; return; fi; sleep 3; echo "$1 $(wc -l < $2/engine.log) $(date +%s)" >> $2/gmarks.txt; }
resetmarks() { [ "$DRY" = 1 ] || : > $1/marks.txt; [ "$DRY" = 1 ] || : > $1/marks-metrics.txt; }
# one-line extracts for RESULT lines ('-' when missing)
probefmt() { python3 $D/sweepfmt.py probe "$1" 2>/dev/null | cut -c1-200 || echo -; }
mbkey() { grep -o "\"$2\": [0-9.]*" "$1" 2>/dev/null | head -1 || echo -; }
mbline() { grep '^MB ' "$1" 2>/dev/null | cut -c4- || echo -; }
omline() { grep '^OM ' "$1" 2>/dev/null | cut -c4- | tr '\n' ' ' || echo -; }
fcms() { grep '^FC ' "$1" 2>/dev/null | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()[3:])["ms_per_step"])' 2>/dev/null || echo 999; }
pssum() { python3 - "$1" <<'PY' 2>/dev/null || echo -
import json, sys
o = []
for l in open(sys.argv[1]):
    if not l.startswith("PS "): continue
    r = json.loads(l[3:]); c = r["cats"]
    o.append("%s: op %.2f ms/step (device %.2f, host-site %.2f, lane waits %.2f; %d steps%s) = %s; MoE %s" % (r["phase"], r["op_ms_step"],
             r["dev_ms_step"], r["host_ms_step"], r["wait_ms_step"], r["steps"], ", profiled wall %s" % r["wall_ms_step"] if "wall_ms_step" in r else "",
             ", ".join("%s %.2f" % (k, v) for k, v in list(c.items())[:9]), "/".join("%s %.2f" % (k, v) for k, v in r["moe"].items())))
print(" || ".join(o) or "-")
PY
}
lvsum() { python3 - "$1" <<'PY' 2>/dev/null || echo -
import json, sys
o = []
for l in open(sys.argv[1]):
    if not l.startswith("LV "): continue
    r = json.loads(l[3:]); m = r["per_step_MB"]; t = r["tiers_end"]
    o.append("%s: %d steps, MB/step ar %s h2d %s d2h %s stream %s ssd %s; expert units vram/pinned/host/ssd/mapped %s; resident %s %% hit %s %%; promo %d%s" % (
        r["phase"], r["steps"], m["ar"], m["h2d"], m["d2h"], m["stream"], m["ssd"], "/".join("%d" % t.get(k, 0) for k in ("vram", "pinned", "host", "ssd", "mapped")),
        r["resident_end"], r["hit_mean"], r["promo"],
        "".join("; %s steps %d: stream %s ssd %s MB/step" % (k, g["steps"], g["stream"], g["ssd"]) for k, g in r.get("split", {}).items())))
print(" || ".join(o) or "-")
PY
}
fastref() { local g; g=$(cat $R/guard-as-live0.txt 2>/dev/null); case "$g" in fast*) echo "${g##* }";; *) echo $AS_FAST_REF_MS;; esac; }
evict() { # STEP: drop the clean page cache of the DeepSeek files (cold mapped tier / Engram), log MemFree
  run docker compose $F0 down
  run_to $AS/evict-$1.txt python3 $T/evict-exl3.py
  log "RESULT after-sweep $1 evict: $(tail -1 $AS/evict-$1.txt 2>/dev/null)"; }

gate() { # TAG VAR=val ... : exl3-gate.sh as-TAG (guard AS_GUARD_MS; a GUARD_MS= in the list wins). 0 = instance up and fast
  local tag=$1; shift
  run_to $AS/gate-$tag.txt env GUARD_MS=$AS_GUARD_MS "$@" $T/exl3-gate.sh as-$tag || {
    log "RESULT after-sweep $tag: VIRHE no healthy server ($(tail -1 $AS/gate-$tag.txt 2>/dev/null | cut -c1-300))"; return 1; }
  [ "$DRY" = 1 ] && return 0
  case "$(cat $R/guard-as-$tag.txt 2>/dev/null)" in
    SLOW*|"") [ "${GATE_SOFT:-0}" = 1 ] && return 0
      log "RESULT after-sweep $tag: VIRHE slow-state instance after one restart (guard '$(cat $R/guard-as-$tag.txt 2>/dev/null)') -> measurements skipped"
      return 1;;
  esac; return 0; }

prof_restart() { # TAG VAR=val ... : exl3-restart.sh with the profenv overlay -> RESULT restart-TAG line (same fields + PROFEVERY, n-gram)
  local tag=$1; shift
  if [ "$DRY" = 1 ]; then echo "+ (export$(q "$@"); . env.sh; docker compose $PF down; docker compose $PF up -d; /health <= 20 min; docker logs > $AS/start-$tag.log; RESULT restart-$tag)"; return 0; fi
  ( export "$@"; . $T/env.sh
    docker compose $PF down 2>&1 | tail -1
    t0=$(date +%s); docker compose $PF up -d 2>&1 | tail -1; ok=0
    for i in $(seq 1 240); do
      [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' $URL/health)" = 200 ] && { ok=1; break; }
      [ "$(docker inspect -f '{{.State.Status}}' $CN 2>/dev/null)" != running ] && break
      sleep 5
    done
    t1=$(date +%s); S=$AS/start-$tag.log; docker logs $CN > $S 2>&1
    if [ $ok != 1 ]; then echo "$(date '+%F %T') RESULT restart-$tag: VIRHE not healthy after $((t1-t0)) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-600)" | tee -a $L; exit 1; fi
    kv=$(grep -m1 -E '^I kv: the pool' $S | cut -c3-200); ex=$(grep -m1 -E '^ +experts +[0-9.]+ GiB' $S | awk '{print $2, $3}')
    kvc=$(grep -m1 -E '^I +kv cache +[0-9.]+ GiB a card' $S | sed -E 's/^I +kv cache +//' | cut -c1-80)
    ng=$(grep -m1 -E 'placement: n-gram table' $S | sed -E 's/^I *(\[[0-9]+\] )?//' | cut -c1-120)
    pin=$(grep -m1 -o 'pinned by this process [0-9.]* GiB' $S); mem=$(free -g | awk 'NR==2{print "used " $3 " / avail " $7 " GiB"}')
    vram=$(for d in /sys/class/drm/card*/device; do [ -e $d/mem_info_vram_used ] && printf "%s " $(( $(cat $d/mem_info_vram_used)/1048576 )); done)
    echo "$(date '+%F %T') RESULT restart-$tag: healthy in $((t1-t0)) s (IMAGE=$IMAGE MODEL=$MODEL HOSTPOOL=$HOSTPOOL MAXSEQS=$MAXSEQS NST=$NST CKINT=$CKINT MNBT=$MNBT KVD=$KVD WIRE=$WIRE MML=$MML PROFEVERY=${PROFEVERY:-32} EXTRA2='$EXTRA2' EXTRA='$EXTRA'); $kv; VRAM experts $ex, KV $kvc; $ng; $pin; RAM $mem; VRAM used MiB [$vram]" | tee -a $L
  ); }

prof_gate() { # TAG FCN EVERY SOFT VAR=val ... : profiled restart, follower, warm-up (as exl3-gate.sh), fastcheck FCN between gmarks,
              # rank-0 device op ms/step <= PROF_GUARD_X x fastref. SOFT=1: no retry, measured anyway. 0 = go on
  local tag=$1 fcn=$2 every=$3 soft=$4; shift 4; local O=$AS/$tag n op op1="" wall lim ref st
  ref=$(fastref); lim=$(awk -v r=$ref -v x=$PROF_GUARD_X 'BEGIN{printf "%.1f", r*x}')
  for n in 1 2; do
    prof_restart $tag PROFEVERY=$every "$@" || { log "RESULT after-sweep $tag: VIRHE no healthy profiled server"; return 1; }
    newdir $O; [ "$DRY" = 1 ] || : > $O/gmarks.txt; follow $O
    in_d 600 $O/warm-q$n.txt python3 ds41_tests.py correct --only-q --effort none --outdir $O
    in_d 600 $O/warm-pf$n.txt python3 ds41_tests.py prefill --depths 16384 --cpt 3.95 --calibrate 0
    gmark g0 $O; in_d 300 $O/guard-fc$n.txt python3 fastcheck.py $fcn; gmark g1 $O
    if [ "$DRY" = 1 ]; then echo "+ python3 $T/profsum.py opstep $O $O/gmarks.txt g0 g1 $every 0   # device op ms/step <= $lim (= $PROF_GUARD_X x $ref)"; return 0; fi
    op=$(python3 $T/profsum.py opstep $O $O/gmarks.txt g0 g1 $every 0 2>/dev/null | sed -n 's/^OPSTEP //p' | python3 -c 'import json,sys; print(json.load(sys.stdin)["dev_ms_step"])' 2>/dev/null || echo 999)
    wall=$(fcms $O/guard-fc$n.txt)
    if awk -v m=$op -v g=$lim 'BEGIN{exit !(m <= g)}'; then st=fast; [ -n "$op1" ] && st="fast-after-retry (first $op1)"
    elif [ "$soft" = 1 ]; then st="ABOVE-LIMIT (soft: measured anyway, state uncertain)"
    elif [ $n = 1 ]; then op1=$op; unfollow; mv $O/engine.log $O/engine-try1.log; continue
    else log "RESULT guard $tag (profiled): SLOW-after-retry (device op $op1 then $op ms/step > $lim = $PROF_GUARD_X x unprofiled $ref)"
         log "RESULT after-sweep $tag: VIRHE slow-state profiled instance after one restart -> measurements skipped"; unfollow; return 1; fi
    echo "$st $op" > $O/guard.txt
    log "RESULT guard $tag (profiled, every $every): $st, rank-0 device op $op ms/step during fastcheck $fcn, profiled wall $wall ms/step (limit $lim = $PROF_GUARD_X x unprofiled $ref)"
    return 0
  done; }

ccsim() { # TAG USERS [CCARGS] -> ccsim-exl3.sh USERS as-TAG (it writes its own RESULT cc_sim line); outputs moved to runs/after-sweep/
  local tag=as-$1 u=$2 o=$AS/ccsim-$1-u$2.out f
  if [ $# -ge 3 ]; then run_to $o env "CCARGS=$3" $T/ccsim-exl3.sh $u $tag; else run_to $o $T/ccsim-exl3.sh $u $tag; fi
  [ "$DRY" = 1 ] && return 0
  f=$(sed -n 's/^CCSIM DONE //p' $o); [ -n "$f" ] && mv $f.* $AS/ 2>/dev/null
  grep -q '^RESULT cc_sim' $o || log "RESULT cc_sim $tag u$u: VIRHE $(tail -2 $o | tr '\n' ' ' | cut -c1-300)"; }
keeplog() { run_to $1/engine.log docker logs $CN; }

# ---------------------------------------------------------------- steps
st_live0() {  # = 3.11. live0 (chain6/7): NST0 CKINT 8192, no span/IL, --live, guarded; 1u, c4, pf (+ c8 new) with marks
  local O=$AS/live0
  gate live0 NST=0 CKINT=8192 MAXSEQS=8 MML=262144 "EXTRA2=--live" || return 0
  newdir $O; resetmarks $O; follow $O; mark start $O
  in_d 900 $O/1u.txt python3 stepprobe.py essay,code 512; mark 1u $O
  in_d 1200 $O/c4.txt python3 mb.py --tag live0 --phases c4 --dur 75 --warm 15; mark c4 $O
  in_d 1200 $O/c8.txt python3 mb.py --tag live0 --phases c8 --dur 75 --warm 15; mark c8 $O
  in_d 1200 $O/pf.txt python3 mb.py --tag live0 --phases pf; mark pf $O
  unfollow
  run_to $O/livestat.txt python3 $D/livestat.py $O
  run_to $O/live.txt python3 $T/profsum.py live $O
  log "RESULT live0-run (NST0 --live, guard $(cat $R/guard-as-live0.txt 2>/dev/null)): 1u $(probefmt $O/1u.txt); c4 $(mbkey $O/c4.txt ms_decode_step); c8 $(mbkey $O/c8.txt ms_decode_step); pf $(mbkey $O/pf.txt ms_step_during) -- runs/after-sweep/live0/ (marks.txt) | 3.11. UD-IQ1_S live0: essay 62.6 / code 68.7 tok/s, c4 26.1, pf 1011.1 ms"
  log "RESULT live0 traffic (--live counters, all cards): $(lvsum $O/live.txt) | 3.11. UD-IQ1_S live0 1u: stream 38.45 MB/step, ar 0.9 MB/step, resident 75 %, hit 95 %"
}
st_prof0() {  # = 3.11. prof0 (chain5): NST0 CKINT 8192 --profile-ops (dump every 32); 1u, c4, pf (+ c8, deep ~243k new)
  local O=$AS/prof0
  prof_gate prof0 384 32 0 NST=0 CKINT=8192 MAXSEQS=8 MML=262144 "EXTRA2=--profile-ops" || return 0
  resetmarks $O; mark start $O
  in_d 900 $O/1u.txt python3 stepprobe.py essay,code 512; mark 1u $O
  in_d 1200 $O/c4.txt python3 mb.py --tag prof0 --phases c4 --dur 75 --warm 15; mark c4 $O
  in_d 1200 $O/c8.txt python3 mb.py --tag prof0 --phases c8 --dur 75 --warm 15; mark c8 $O
  in_d 1200 $O/pf.txt python3 mb.py --tag prof0 --phases pf; mark pf $O
  in_d 1500 $O/deep.txt python3 onem.py $DEEP; mark deep $O
  unfollow
  run_to $O/profparse.txt python3 $D/profparse.py $O 0 --cats
  run_to $O/profsum.txt python3 $T/profsum.py phases $O 32 0
  log "RESULT prof0-run (NST0 --profile-ops, guard $(cat $O/guard.txt 2>/dev/null)): 1u $(probefmt $O/1u.txt); c4 $(mbkey $O/c4.txt ms_decode_step); c8 $(mbkey $O/c8.txt ms_decode_step); pf $(mbkey $O/pf.txt ms_step_during); deep $(omline $O/deep.txt | cut -c1-400) -- runs/after-sweep/prof0/ (marks.txt) | 3.11. prof0: 1u 32.4 ms/step, c4 41.2, pf 1137.5"
  log "RESULT profiling decode step EXL3 (prof0, --profile-ops rank 0, profiled op ms/step by category; 3.11. UD-IQ1_S prof0 1u: all-reduce 5.03, attn proj 4.55, routed MoE 3.93, attn core 2.82, mHC 2.05, shexp 1.82, elementwise 1.53, Engram 0.44 = 23.9): $(pssum $O/profsum.txt)"
}
st_prcfg() {  # the PR author's TP4 serve line (deepseek41-wip.md): host pool 4096, Engram disk, ckint 8192, no RAM KV tier; cold page
              # cache; --profile-ops (every 32) + --live (tier units, hit %, ssd bytes). Soft guard (its tier cost is the measurement)
  local O=$AS/prcfg
  evict prcfg
  prof_gate prcfg 384 32 1 HOSTPOOL=4096 NST=$PRCFG_NST CKINT=8192 MNBT=8192 KVD=bf16 WIRE=exact MAXSEQS=8 MML=262144 "EXTRA=" \
    "EXTRA2=--ngram-placement disk --weights-disk-tier --profile-ops --live" || return 0
  resetmarks $O; mark start $O
  in_d 300 $O/1u.txt python3 stepprobe.py essay,code 512; mark 1u $O
  in_d 300 $O/c4.txt python3 mb.py --tag prcfg --phases c4 --dur 75 --warm 15; mark c4 $O
  # 6.11. 07.50: mb pf dropped (experts on the O_DIRECT disk tier: ~56 GiB per 8192 chunk, would run to its timeout)
  in_d 300 $O/pf16k.txt python3 ds41_tests.py prefill --depths 16384 --cpt 3.95 --calibrate 0; mark pf16k $O
  unfollow
  run_to $O/profsum.txt python3 $T/profsum.py phases $O 32 0
  run_to $O/live.txt python3 $T/profsum.py live $O
  log "RESULT prcfg-run (PR author's TP4 line: host pool 4096, --ngram-placement disk, ckint 8192, MNBT 8192, no RAM KV tier, NST $PRCFG_NST, MAXSEQS 8, MML 262144; cold page cache; --profile-ops + --live; guard $(cat $O/guard.txt 2>/dev/null)): 1u $(probefmt $O/1u.txt); c4 $(mbkey $O/c4.txt ms_decode_step); pf $(mbkey $O/pf.txt ms_step_during); 16k prefill $(grep -o '"prefill_tps": [0-9.]*' $O/pf16k.txt 2>/dev/null | head -1) (profiled) | PR author UD-IQ1_S 62 GiB host: 77.4/77.2 tok/s, 16k 3746 tok/s (unprofiled)"
  log "RESULT profiling prcfg ops (rank 0, profiled op ms/step; host-site = mapped/host-tier work timed on the host clock): $(pssum $O/profsum.txt)"
  log "RESULT profiling prcfg tiers (--live, all cards): $(lvsum $O/live.txt)"
}
st_onem() {   # = 3.11. final2 1M step: MML 1048576 MAXSEQS 4 NST0 CKINT 2048 span 2 IL8, gated; onem.py; fable-1m cc_sim; ~700k extra
  local O=$AS/onem tmo
  gate onem MML=1048576 MAXSEQS=4 NST=0 CKINT=2048 "EXTRA2=$BEST2X" || return 0
  newdir $O; tmo=$(( $(left) < 2700 ? $(left) : 2700 )); [ $tmo -lt 600 ] && tmo=600
  in_d $tmo $O/onem.txt python3 onem.py 262144,524288,745472
  log "RESULT 1M probe (MML 1048576, MAXSEQS 4, NST0 CKINT 2048 span 2 IL8) [guard $(cat $R/guard-as-onem.txt 2>/dev/null)]: $(omline $O/onem.txt) | 3.11. UD-IQ1_S: 266k 1805, 501k 1287, 644k 1081 tok/s, 22.0 / 24.4 / 25.1 ms/step; hit turn 644k 8.01 s"
  fits fable-1m 18; case $? in 0) ccsim fable-1m 4 "$FABLE_ARGS"; RAN="$RAN fable-1m";; 2) keeplog $O; return 2;; esac
  fits onem-extra 15 31; case $? in
    0) in_d 1800 $O/onem-extra.txt python3 onem.py 843000; RAN="$RAN onem-extra"
       log "RESULT 1M probe extra (target 843000 = ~700k tokens, same instance) [guard $(cat $R/guard-as-onem.txt 2>/dev/null)]: $(omline $O/onem-extra.txt) | 3.11. UD-IQ1_S: 700k 1033 tok/s, 25.8 ms/step; hit turn 5.87 s";;
    2) keeplog $O; return 2;; esac
  keeplog $O
}
st_nvme() {   # window best config with Engram read from the container (--ngram-placement disk; mmap, MADV_RANDOM, host gather +
              # row prefetch), page cache evicted first; probe / mb c4 / mb turns 1:1 with the window's RAM runs + ~140k + hit turn
  local O=$AS/nvme ng
  evict nvme
  GATE_SOFT=1 gate nvme "GUARD_MS=$NVME_GUARD_MS" NST=0 CKINT=2048 MAXSEQS=8 MML=262144 "EXTRA2=$BEST2X --ngram-placement disk" || return 0
  newdir $O
  if [ "$DRY" != 1 ]; then
    ng=$(grep -m1 -E 'placement: n-gram table' $R/start-as-nvme.log 2>/dev/null | cut -c1-140)
    log "RESULT nvme placement: ${ng:-?}; $(grep -m1 -o 'pinned by this process [0-9.]* GiB' $R/start-as-nvme.log 2>/dev/null); guard $(cat $R/guard-as-nvme.txt 2>/dev/null) (RAM same config 6.11.: fast 19.36)"
    echo "$ng" | grep -q ' disk' || { log "RESULT after-sweep nvme: VIRHE n-gram table not on disk ('$ng') -> skipped"; return 0; }
  fi
  in_d 900 $O/probe.txt python3 stepprobe.py essay,code 512
  log "RESULT nvme probe (Engram on NVMe, 1 user, greedy, 512 tok): $(probefmt $O/probe.txt) | RAM, same config + window: $(probefmt $R/probe.txt)"
  in_d 1800 $O/mb-c4.txt python3 mb.py --tag nvme --phases c4
  log "RESULT nvme mb c4: $(mbline $O/mb-c4.txt) | RAM window: $(mbkey $R/mb-c4.txt ms_decode_step) $(mbkey $R/mb-c4.txt agg_dec_tps)"
  in_d 1800 $O/mb-turns.txt python3 mb.py --tag nvme --phases turns
  log "RESULT nvme mb turns (56k cold + 5 hit turns): $(mbline $O/mb-turns.txt | cut -c1-260) | RAM window: $(mbkey $R/mb-turns.txt cold_ttft) $(mbkey $R/mb-turns.txt ttft_hit_mean)"
  in_d 1500 $O/onem140.txt python3 onem.py 140000
  log "RESULT nvme 140k (onem.py 140000: cold ~142k prefill + 256 tok + hit turn, Engram on NVMe): $(omline $O/onem140.txt) | RAM reference: cap step 'cap ref140' (MML 524288 MAXSEQS 16) if run"
  keeplog $O
}
st_cap() {    # capacity: MAXSEQS 16 at MML 524288, window best config otherwise, gated; cc_sim u12 (3.11. confirm args), then the
              # RAM 140k reference for nvme and u16 if time is left
  local O=$AS/cap
  gate cap MML=524288 MAXSEQS=16 NST=0 CKINT=2048 "EXTRA2=$BEST2X" || return 0
  newdir $O
  ccsim cap-s16 12
  fits cap-ref140 4; case $? in
    0) in_d 1500 $O/onem140.txt python3 onem.py 140000; RAN="$RAN cap-ref140"
       log "RESULT cap ref140 (RAM Engram, onem.py 140000 on the MAXSEQS 16 / MML 524288 instance, after u12): $(omline $O/onem140.txt)";;
    2) keeplog $O; return 2;; esac
  fits cap-u16 14; case $? in 0) ccsim cap-s16 16; RAN="$RAN cap-u16";; 2) keeplog $O; return 2;; esac
  keeplog $O
}
st_profpf() { # = 3.11. profpf (chain7): RADIANCE_PROFILE_EVERY=1; fastcheck 96, 31k prefill alone, 20 s idle, mb pf --warm 0
  local O=$AS/profpf
  prof_gate profpf 96 1 0 NST=0 CKINT=8192 MAXSEQS=8 MML=262144 "EXTRA2=--profile-ops" || return 0
  resetmarks $O; mark start $O
  in_d 300 $O/fc.txt python3 fastcheck.py 96; mark fc $O
  in_d 900 $O/alone.txt python3 ds41_tests.py prefill --depths 31416 --cpt 3.95 --calibrate 0; mark alone $O
  run sleep 20; mark idle $O
  in_d 1200 $O/pf.txt python3 mb.py --tag profpf --phases pf --warm 0; mark pf $O
  unfollow
  run_to $O/profsteps.txt python3 $D/profsteps.py $O 0
  run_to $O/profsum.txt python3 $T/profsum.py phases $O 1 0
  log "RESULT profpf-run (NST0 --profile-ops every step, guard $(cat $O/guard.txt 2>/dev/null)): profiled 1u $(grep -o '"ms_per_step": [0-9.]*' $O/fc.txt 2>/dev/null); alone $(grep -o '"prefill_tps": [0-9.]*' $O/alone.txt 2>/dev/null | head -1); pf $(grep -o '"prefill_tps": [0-9.]*' $O/pf.txt 2>/dev/null) -- runs/after-sweep/profpf/ | 3.11.: 61.25 ms, 2614.5, 2585.8"
  log "RESULT profiling profpf per-step classes (profsteps.py; 3.11.: prefill-chunk ops not in the dumps): $(grep '^== ' $O/profsteps.txt 2>/dev/null | tr '\n' ' ' | cut -c1-900)"
}

. $T/st_noxp.inc   # 6.11. 08.00 (reviewed): engine without the idle expert-prefetch queue

# ---------------------------------------------------------------- main
declare -A EST=([noxp]=13 [live0]=10 [prof0]=15 [prcfg]=11 [onem]=28 [nvme]=13 [cap]=18 [profpf]=9)
if [ "$DRY" = 1 ]; then
  tot=0; for s in ${STEPS//,/ }; do tot=$((tot + ${EST[$s]:-0})); done
  echo "# DRYRUN: nothing runs, nothing is written. STEPS=$STEPS; estimates (min): $(for s in ${STEPS//,/ }; do printf '%s %s, ' $s ${EST[$s]:-?}; done)total $tot (+ fable-1m 18 inside onem; optional when time is left: onem-extra 15, cap-ref140 4, cap-u16 14); budget $BUDGET_MIN"
else
  mkdir -p $AS
fi
log "RESULT after-sweep: start STEPS=$STEPS BUDGET_MIN=$BUDGET_MIN IMAGE=$IMAGE MODEL=$MODEL AS_GUARD_MS=$AS_GUARD_MS PROF_GUARD_X=$PROF_GUARD_X"
for s in ${STEPS//,/ }; do
  fits $s ${EST[$s]:-15}; rc=$?; [ $rc = 2 ] && break; [ $rc = 1 ] && continue
  ts=$(date +%s); st_$s; rc=$?; unfollow; RAN="$RAN $s"
  log "RESULT after-sweep $s: step done in $(( ($(date +%s) - ts) / 60 )) min ($(( ($(date +%s) - T0) / 60 )) of $BUDGET_MIN min used)"
  [ $rc = 2 ] && break
done
unfollow
run docker compose $F0 down
log "RESULT after-sweep: done in $(( ($(date +%s) - T0) / 60 )) min; ran:${RAN:- none}; skipped:${SKIPPED:- none}; EXL3 server down; compare: python3 $T/compare-after.py"
