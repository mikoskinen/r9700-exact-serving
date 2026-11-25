#!/bin/bash
# dsm-span4.sh (10.11., copy of dsm-pipe.sh acf871e5 with arm S = --prefill-span 4 + MNBT 8256, run alone and compared
# with f-pipe's arm A files in the same runs/xpipe dir: no baseline re-run). Original header: dsm-pipe.sh (10.11.2026, queue-ar/dsmain, ds41-pipe agent) -- NOT a queue item; the body of 57-p4zyzzzzz-ds-f-pipe.sh.
# DeepSeek-V4.1 Flash EXL3 2.25 bpw, gpu02, all 4 cards, TP4, MML 524288, t3 env: the pipelined prefill
# (RADIANCE_DS41_PIPE=1, deepseek41 of ds41-pipe fd215624) against the serial step on ONE image, two starts:
#   A  RADIANCE_DS41_PIPE unset  = ds41-main 967d4ea's serial step, byte for byte (nothing new declared or issued)
#   B  RADIANCE_DS41_PIPE=1      = a lone prefill step wider than the span's shared budget (2112 rows on this line) in two
#                                  pieces: output projection, attention all-reduce, post+collapse+norm, MoE + its
#                                  all-reduce a piece at a time, every collective on the second lane beside the other
#                                  piece's work (deepseek41.cpp, "A prefill step wider than the shared budget")
# IMAGE (XPIPE_IMAGE): radiance-local:ds41-pipe-fd215624-5c13e555-l = radiance-local:ds41-main-967d4eae-5c13e555-l + that
#   deepseek41.so (build/pipe/assemble.sh on the workstation; /opt/radiance/share/radiance/pipe-PROVENANCE.txt).
# EXACT CLAIM: B's every output is A's, byte for byte (the pieces compute the serial step's rows by the same kernels and
#   forms; the exact all-reduce folds each element in rank order at any message size). Checked here: warm-up answers vs
#   stock (both arms), the i1 copy after a ~31k cold prompt and the 136k deep answer, B vs A byte for byte (+ each vs
#   105's tuned3l files, information). A difference is a VIRHE line (NOT EXACT), not hidden; the numbers are still read.
# PER ARM (A then B), decode before prefill (TESTING 9.11.): gate (restart, warm-up incl. a 16k prefill, 1-user fastcheck
#   384; one retry over 23 ms) -> modes (the t3 notes; the pipelining note in B only, none in A) -> warm-up answers vs
#   stock -> mb c4 -> mb c8 (decode must not move: B's decode steps are the serial ones) -> THEN fresh prefill 16k, 64k,
#   136k (ds41_tests.py prefill, one depth a call, a thermwin line each) -> mb pf -> i1 copy (copyt) -> deep answer
#   (deepq) -> per-phase /stats (statsum) -> B only: the first pipelined step's note must be in the log.
# THEN one B-vs-A line (% differences: guard, c4 / c8 tok/s + ms, prefill 16k / 64k / 136k tok/s, pf, copy / deep TTFT,
#   stats' prefill ms a step) and the identity summary. ~45 min (each arm ~4 min start + ~2 warm-up + 7.5 decode +
#   ~5 prefill-side).
# GOAL / BASIS: 136k cold prefill 2642 tok/s (target >= 4000); rank 0's 4160-row step 1206 ms of which collectives 305 ms
#   (25 %). Flash-Next's same cut gave +8.2 % prefill, exact. Expected here +5..12 % prefill (the attention all-reduce of
#   piece 0 and the MoE all-reduce of piece 1 stay partly exposed; see the source).
# NO cc_sim (user 10.11.). No power / host / fan settings; production compose files read only (--project-directory).
# DRYRUN=1 bash dsm-pipe.sh : prints every command, runs and writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-pipe-tp4.yaml                        # = ds41-dsm-tp4.yaml + RADIANCE_DS41_PIPE
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=$R/xpipe; export STATSDIR=$O/stats
RX=$R/x79                                       # 105's tuned3l reference files (information)
export IMAGE=${XPIPE_IMAGE:-radiance-local:ds41-pipe-fd215624-5c13e555-l}
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
DQ=$QA/deepq.py
DEPTHS=${XPIPE_DEPTHS:-16384 65536 136000}
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_PIPE"
# ---------------------------------------------------------------- helpers (dsm-serve.sh's, 10.11., copied; log tag xpipe)
t3env() {   # 105's t3 arm, every other switch unset (RADIANCE_DS41_PIPE included)
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
}
modelvars() {   # modelvars 225 -> MODEL, HP29, LIM, WARMREF, C8REF (105's c8 ms a step)
  case $1 in
    225) export MODEL=$M225; HP29=19456; LIM=23; WARMREF=$WARM_REF_225; C8REF=43.1 ;;
    *)   return 1 ;;
  esac
  best_vars; export MNBT=4160 XARGS="$XARGS --live"
}
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
gate() {   # gate TAG LIMIT: lib-29's gate + keeplog before the retry / after a second slow check
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
  log "RESULT guard $tag: $st, 1-user ${ms} ms/step (limit $lim; 105 tuned3l 225: 15.81)"
  case $st in SLOW*) return 2 ;; esac
}
FOLLOW_PID=""
follow() { [ "$DRY" = 1 ] && { echo "+ docker logs -f $CN > $1/engine.log &"; return; }
  docker logs -f $CN > $1/engine.log 2>&1 & FOLLOW_PID=$!; sleep 2; }
unfollow() { [ -n "$FOLLOW_PID" ] && { kill $FOLLOW_PID 2>/dev/null; wait $FOLLOW_PID 2>/dev/null; FOLLOW_PID=""; }; return 0; }
trap 'unfollow; down' EXIT
trap 'exit 143' TERM INT
mark() { [ "$DRY" = 1 ] && { echo "+ mark $1 -> $2/marks.txt"; return; }; echo "$1 $(wc -l < $2/engine.log) $(date +%s)" >> $2/marks.txt; }
snap() { run "curl -s -m 15 $URL/stats > $1/stats-$2.json 2>/dev/null; echo $2 >> $1/stats-order.txt"; }
ph() { mark $2 $1; snap $1 $2; }
thermline() {   # thermline T0 T1: the window's full-power rows (heatsoak.sh thermwin) or the cards now
  type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"
}
cmpfile() {   # cmpfile A B -> IDENTICAL (n chars) | DIFFERENT at char k (na vs nb chars) | missing
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xpipe $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xpipe $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  case $r in IDENTICAL*) log "RESULT xpipe $1 warm-up answers vs stock ($(basename $2)): $r (105 tuned3l: IDENTICAL)" ;;
    *) log "RESULT xpipe $1 warm-up answers vs stock ($(basename $2)): VIRHE $r -- NOT EXACT vs the tuned3l line" ;; esac
}
modes() {   # modes TAG ARM: the t3 line's notes, and the pipelining note in B only -> 0, else 1 (arm stopped)
  [ "$DRY" = 1 ] && { echo "+ modes $1: indexer note (mxfp4, latent bf16, compressor tail bound, router selection+sort separate, AR_PCN separate at 0 layers), ARGATHER note, staging line, no acquire-rule line, interleave; pipelining note: arm $2 (A none, B 'pipelined prefill on')"; return 0; }
  local MLF SL AL IL PL ok=1
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  AL=$(docker logs $CN 2>&1 | grep -m3 -E 'activation arena' | cut -c1-200 | tr '\n' '|')
  IL=$(grep -iE 'interleave' $O/start-$1.log 2>/dev/null | sed 's/^[A-Z] *//' | tr -s ' ' | head -3 | tr '\n' '|' | cut -c1-300)
  PL=$(docker logs $CN 2>&1 | grep -E 'pipelined prefill|are pipelined' | cut -c1-400 | tr '\n' '|')
  log "RESULT xpipe $1 modes: $(echo "$MLF" | cut -c1-1100) | arena: $AL | staging: ${SL:-none} | interleave: ${IL:-no line} | pipelining: ${PL:-none}"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate" "post-collapse-norm separate at 0 layers" "next collapse-norm separate at 0 layers"; do
    case "$MLF" in *"$w"*) ;; *) log "RESULT xpipe $1: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  docker logs $CN 2>&1 | grep -q 'the MoE gather and its all-reduce as one kernel' || { log "RESULT xpipe $1: VIRHE no RADIANCE_DS41_ARGATHER note"; ok=0; }
  [ -n "$SL" ] || { log "RESULT xpipe $1: VIRHE no prefill staging line (levels refused?)"; ok=0; }
  docker logs $CN 2>&1 | grep -q 'the acquire rule plays' && { log "RESULT xpipe $1: VIRHE an acquire-rule line (RADIANCE_AQL_ACQ should be unset)"; ok=0; }
  case $2 in
    A) [ -z "$PL" ] || { log "RESULT xpipe $1: VIRHE a pipelining line in arm A (RADIANCE_DS41_PIPE unset): $PL"; ok=0; } ;;
    SP) case "$PL" in *"pipelined prefill on"*) ;; *) log "RESULT xpipe $1: VIRHE no 'pipelined prefill on' note in arm SP: ${PL:-none}"; ok=0 ;; esac ;;
    S) [ -z "$PL" ] || { log "RESULT xpipe $1: VIRHE a pipelining line in arm S (RADIANCE_DS41_PIPE unset): $PL"; ok=0; }
       grep -q -- '--prefill-span *4' $O/start-$1.log 2>/dev/null || docker logs $CN 2>&1 | grep -q -E 'prefill-span +4' || log "RESULT xpipe $1: note -- the start log does not echo --prefill-span 4" ;;
    B) case "$PL" in *"pipelined prefill on"*) ;; *) log "RESULT xpipe $1: VIRHE no 'pipelined prefill on' note in arm B: ${PL:-none}"; ok=0 ;; esac ;;
  esac
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
copyt() {   # copyt TAG OFFSET: the copy test after a ~31k-token cold prompt (105's prompt: same offset -> same text)
  local t0=$(date +%s)
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset $2 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xpipe $1 copy after a cold prompt: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300) | $(thermline $t0 $(date +%s))"
}
deepq() {   # deepq TAG: the fixed ~135k prompt and a 512-token greedy answer (105's deepq.py)
  local t0=$(date +%s)
  run "(cd $D && timeout 1800 python3 $DQ $O/deep-$1.txt 135000 512 > $O/deep-$1.out 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xpipe $1 deep probe: $(grep -m1 '^DEEPQ ' $O/deep-$1.out | cut -c7- | cut -c1-200) | $(thermline $t0 $(date +%s))"
}
mbpf() {   # mbpf TAG PHASE: lib-pr266's mbp + a thermal line
  local t0=$(date +%s)
  mbp $1 $2
  [ "$DRY" = 1 ] || log "RESULT xpipe $1 mb $2 window $(thermline $t0 $(date +%s))"
}
pfd() {   # pfd TAG DEPTH: one fresh prefill of DEPTH tokens (nonce first, cache_prompt off; chars at 3.95 a token, no
          # calibration request) and its thermal line
  local t0=$(date +%s)
  run "(cd $D && timeout 1800 python3 ds41_tests.py prefill --depths $2 --cpt 3.95 --calibrate 0 --reps 1 > $O/$1-pf$2.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xpipe $1 fresh prefill $2: $(grep '^J ' $O/$1-pf$2.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); print("%s tok, cached %s, TTFT %s s, %s tok/s, %s" % (d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("prefill_tps"), d.get("status")))' 2>&1 | cut -c1-200) | $(thermline $t0 $(date +%s))"
}
statsum() {   # statsum DIR TAG: per-phase deltas of the /stats snapshots
  [ "$DRY" = 1 ] && { echo "+ RESULT xpipe $2 stats per phase from $1/stats-*.json"; return 0; }
  python3 - $1 > $1/stats-phases.txt 2>&1 <<'PY'
import json, os, sys
d = sys.argv[1]
order = [l.strip() for l in open(os.path.join(d, "stats-order.txt")) if l.strip()]
snaps = []
for p in order:
    try: snaps.append((p, json.load(open(os.path.join(d, f"stats-{p}.json")))))
    except Exception: pass
G = 1e9
for (pa, a), (pb, b) in zip(snaps, snaps[1:]):
    ea, eb = a.get("experts", {}), b.get("experts", {}); la, lb = a.get("link", {}), b.get("link", {})
    ds = b["steps"] - a["steps"]
    if ds <= 0: print(f"PH {pb}: no steps"); continue
    dpm = (b["steps_prefill"] + b["steps_mixed"]) - (a["steps_prefill"] + a["steps_mixed"]); dd = b["steps_decode"] - a["steps_decode"]
    dr = eb.get("routed", 0) - ea.get("routed", 0)
    o = {"phase": pb, "steps": ds, "prefill_mixed_steps": dpm, "decode_steps": dd,
         "ms_step": round((b["step_ns"] - a["step_ns"]) / ds / 1e6, 2),
         "ms_decode_step": round((b["step_ns_decode"] - a["step_ns_decode"]) / max(1, dd) / 1e6, 2) if dd else None,
         "ms_prefill_mixed_step": round(((b["step_ns"] - b["step_ns_decode"]) - (a["step_ns"] - a["step_ns_decode"])) / max(1, dpm) / 1e6, 1) if dpm else None,
         "hit_pct_rank0": round(100 * (eb.get("routed_res", 0) - ea.get("routed_res", 0)) / dr, 2) if dr else None,
         "stream_MB_step": round((lb.get("stream_bytes", 0) - la.get("stream_bytes", 0)) / ds / 1e6, 1),
         "h2d_MB_step": round((lb.get("h2d_bytes", 0) - la.get("h2d_bytes", 0)) / ds / 1e6, 1),
         "stage_layers": eb.get("stage_layers", 0) - ea.get("stage_layers", 0),
         "staged_GB": round((eb.get("staged_bytes", 0) - ea.get("staged_bytes", 0)) / G, 2),
         "stage_overflow": eb.get("stage_overflow", 0) - ea.get("stage_overflow", 0),
         "vram_units_end": next((t.get("units") for t in eb.get("tiers", []) if t.get("name") == "vram"), None)}
    print("PH " + json.dumps(o))
PY
  log "RESULT xpipe $2 stats per phase: $(grep '^PH ' $1/stats-phases.txt | cut -c4- | python3 -c 'import json,sys
out = []
for l in sys.stdin:
    try: d = json.loads(l)
    except Exception: continue
    out.append("%s: %s st (%s pf/mx), %s ms/st, dec %s ms, pf/mx %s ms, hit %s %%, stream %s MB/st, staged %s layers %s GB (%s overflow), vram %s" % (
        d["phase"], d["steps"], d["prefill_mixed_steps"], d["ms_step"], d["ms_decode_step"], d["ms_prefill_mixed_step"], d["hit_pct_rank0"],
        d["stream_MB_step"], d["stage_layers"], d["staged_GB"], d["stage_overflow"], d["vram_units_end"]))
print("; ".join(out))' 2>&1 | cut -c1-2000)"
}
# ---------------------------------------------------------------- the arms
arm() {   # arm A|B|S (S = span 4: a lone prefill step of 8192 rows, MNBT 8256; the shared budget beside decode rows stays 2112)
  local a=$1 tag=pipe-$1 P=$O/pipe-$1 rc d
  t3env; modelvars 225 || return 1
  [ "$a" = B ] && export RADIANCE_DS41_PIPE=1
  case $a in S|SP) export MNBT=8256 XARGS="${XARGS/--prefill-span 2/--prefill-span 4}" ;; esac
  [ "$a" = SP ] && export RADIANCE_DS41_PIPE=1   # 10.11.: lever 7 lost 11-13 % at span 2 (pieces of 2048 -> the MoE pass reads every expert twice a step?); at span 4 the pieces are 4096 = today's MoE pass
  log "RESULT xpipe $tag: start IMAGE=$IMAGE MODEL=$MODEL HOSTPOOL=$HOSTPOOL MNBT=$MNBT env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
  if [ "$DRY" != 1 ]; then
    [ -f $MODELDIR/$MODEL ] && [ -s $MODELDIR/$MODEL.sha256ok ] || { log "RESULT xpipe $tag: VIRHE $MODELDIR/$MODEL missing or not verified"; return 1; }
  fi
  mkd $P; run "rm -f $P/marks.txt $P/stats-order.txt"
  gate $tag $LIM; rc=$?
  [ $rc = 1 ] && { log "RESULT xpipe $tag: VIRHE not healthy ($(desc)); start log $O/start-$tag.log: $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  [ $rc = 2 ] && { log "RESULT xpipe $tag: VIRHE slow state after one restart -> arm skipped"; down; return 1; }
  modes $tag $a || { log "RESULT xpipe $tag: VIRHE the modes are not the expected ones -> arm stopped"; down; return 1; }
  identw $tag $WARMREF
  follow $P; ph $P start
  for p in c4 c8; do stopfile && break; mbp $tag $p; ph $P $p; done
  [ "$DRY" = 1 ] || log "RESULT xpipe $tag state: c8 $(mbms $O/$tag-c8.txt) ms a decode step (105 tuned3l c8 $C8REF ms; > 1.14x = the slow state, flagged to the coordinator)"
  for d in $DEPTHS; do stopfile && break; pfd $tag $d; ph $P pf$d; done
  stopfile || { mbpf $tag pf; ph $P pf; }
  stopfile || { copyt $tag-i1 0; ph $P i1; }
  stopfile || { deepq $tag; ph $P deep; }
  if [ "$DRY" = 1 ]; then echo "+ info: cmp $O/copy-$tag-i1-run1.txt $RX/copy-x79-225pa-t3-i1-run1.txt; cmp $O/deep-$tag.txt $RX/deep-x79-225pa-t3.txt"
  else
    log "RESULT xpipe $tag identity vs tuned3l (105 x79-225pa-t3, information): i1 copy $(cmpfile $O/copy-$tag-i1-run1.txt $RX/copy-x79-225pa-t3-i1-run1.txt); deep $(cmpfile $O/deep-$tag.txt $RX/deep-x79-225pa-t3.txt)"
  fi
  if { [ "$a" = B ] || [ "$a" = SP ]; } && [ "$DRY" != 1 ]; then   # the path ran: rank 0's note at the first pipelined step
    local SN; SN=$(docker logs $CN 2>&1 | grep -m1 'are pipelined:' | sed 's/^radiance: //' | cut -c1-300)
    [ -n "$SN" ] && log "RESULT xpipe $tag pipelined steps ran: $SN" || log "RESULT xpipe $tag: VIRHE no pipelined step ran (no 'are pipelined:' note) -- B measured the serial path"
  fi
  unfollow
  run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
  statsum $P $tag
  run "python3 $T/profsum.py live $P > $P/live.txt 2>&1"
  down
  log "RESULT xpipe $tag: done"
}
mbms() { grep '^MB ' $1 2>/dev/null | head -1 | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()[3:]).get("ms_decode_step", ""))' 2>/dev/null; }
summary() {   # summary REF NEW LABEL: NEW vs REF, % differences, and the identity summary
  local RT=${1:-pipe-A} NT=${2:-pipe-S} LB=${3:-"S (span 4, MNBT 8256) vs A (f-pipe's serial arm, span 2, MNBT 4160)"}
  [ "$DRY" = 1 ] && { echo "+ RESULT xpipe B vs A: guard, c4 / c8 tok/s + ms, prefill $DEPTHS tok/s + TTFT, pf, copy / deep TTFT, prefill ms a step; identity B vs A: warm-up, i1 copy, deep"; return 0; }
  log "RESULT xpipe $LB, same image $IMAGE: $(RT=$RT NT=$NT python3 - $O "$DEPTHS" <<'PY' 2>&1 | cut -c1-2400
import json, sys
O, depths = sys.argv[1], sys.argv[2].split()
def mb(tag, p):
    try:
        for l in open(f"{O}/{tag}-{p}.txt"):
            if l.startswith("MB "): return json.loads(l[3:])
    except OSError: pass
    return {}
def gd(tag):
    try: return float(open(f"{O}/guard-{tag}.txt").read().split()[-1])
    except Exception: return None
def pf(tag, d):
    try:
        for l in open(f"{O}/{tag}-pf{d}.txt"):
            if l.startswith("J "): return json.loads(l[2:])
    except OSError: pass
    return {}
def cp(tag):
    try:
        for l in open(f"{O}/{tag}-i1-copy.txt"):
            if l.startswith("J "): return json.loads(l[2:]).get("ttft")
    except OSError: pass
    return None
def dq(tag):
    try:
        for l in open(f"{O}/deep-{tag}.out"):
            if l.startswith("DEEPQ "): return json.loads(l[6:]).get("ttft")
    except OSError: pass
    return None
def pfms(tag):   # the stats' prefill/mixed ms a step over the three fresh prefills
    try:
        tot = n = 0.0
        for l in open(f"{O}/{tag}/stats-phases.txt"):
            if not l.startswith("PH "): continue
            d = json.loads(l[3:])
            if str(d.get("phase", "")).startswith("pf") and d["phase"] != "pf" and d.get("ms_prefill_mixed_step"):
                tot += d["ms_prefill_mixed_step"] * d["prefill_mixed_steps"]; n += d["prefill_mixed_steps"]
        return round(tot / n, 1) if n else None
    except (OSError, ValueError): return None
import os
A, B = os.environ.get("RT", "pipe-A"), os.environ.get("NT", "pipe-S")
rows = [("guard ms", gd(A), gd(B)),
        ("c4 tok/s", mb(A, "c4").get("agg_dec_tps"), mb(B, "c4").get("agg_dec_tps")), ("c4 ms", mb(A, "c4").get("ms_decode_step"), mb(B, "c4").get("ms_decode_step")),
        ("c8 tok/s", mb(A, "c8").get("agg_dec_tps"), mb(B, "c8").get("agg_dec_tps")), ("c8 ms", mb(A, "c8").get("ms_decode_step"), mb(B, "c8").get("ms_decode_step"))]
for d in depths:
    rows.append((f"prefill {d} tok/s", pf(A, d).get("prefill_tps"), pf(B, d).get("prefill_tps")))
rows += [("pf tok/s", mb(A, "pf").get("prefill_tps"), mb(B, "pf").get("prefill_tps")), ("pf decode kept", mb(A, "pf").get("decode_kept"), mb(B, "pf").get("decode_kept")),
         ("copy 31k TTFT s", cp(A), cp(B)), ("deep 136k TTFT s", dq(A), dq(B)), ("stats prefill ms a step (fresh prefills)", pfms(A), pfms(B))]
out = []
for n, x, y in rows:
    if x is None or y is None: out.append(f"{n} {x} -> {y}"); continue
    x, y = float(x), float(y)
    out.append(f"{n} {x:g} -> {y:g} ({(y / x - 1) * 100:+.1f} %)" if x else f"{n} {x} -> {y}")
print("; ".join(out))
PY
)"
  local w c d
  w=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): (d.get("content"), d.get("finish")) for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return None
a, b = rd(sys.argv[1]), rd(sys.argv[2])
print("not compared" if a is None or b is None else ("IDENTICAL (%d answers)" % len(a) if a == b else "DIFFERENT"))' $O/guard/$RT-warm-q.txt $O/guard/$NT-warm-q.txt 2>&1)
  c=$(cmpfile $O/copy-$NT-i1-run1.txt $O/copy-$RT-i1-run1.txt)
  d=$(cmpfile $O/deep-$NT.txt $O/deep-$RT.txt)
  case "$w|$c|$d" in IDENTICAL*\|IDENTICAL*\|IDENTICAL*) log "RESULT xpipe identity $NT vs $RT: EXACT -- warm-up $w; i1 copy $c; deep 136k answer $d" ;;
    *) log "RESULT xpipe identity $NT vs $RT: VIRHE NOT EXACT -- warm-up $w; i1 copy $c; deep 136k answer $d" ;; esac
}
# ---------------------------------------------------------------- main
log "RESULT xpipe 57-p4zyzzzzz-ds-f2-span4: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) arm S (--prefill-span 4, MNBT 8256; vs f-pipe's arm A files), yaml $Y, depths [$DEPTHS] | goal 136k cold 2642 -> >= 4000 tok/s; Flash-Next's same cut +8.2 %"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xpipe: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xpipe: VIRHE image $IMAGE not on gpu02"; exit 1; }
  for v in $SWITCHES; do grep -q "^ *$v:" $Y || { log "RESULT xpipe: VIRHE $Y does not pass $v"; exit 1; }; done
  for v in RADIANCE_DS41_PIPE RADIANCE_DS41_ARGATHER RADIANCE_DS41_AR_PCN RADIANCE_STAGE_SPAN RADIANCE_DS41_LEVELS; do
    [ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c "cat /opt/radiance/bin/radiance /opt/radiance/share/radiance/architectures/deepseek41.so | grep -c $v" 2>/dev/null)" -gt 0 ] 2>/dev/null \
      || { log "RESULT xpipe: VIRHE $IMAGE's engine / plugin does not know $v"; exit 1; }
  done
  for f in $DQ $T/profsum.py; do [ -e $f ] || { log "RESULT xpipe: VIRHE $f missing"; exit 1; }; done
  for f in $RX/copy-x79-225pa-t3-i1-run1.txt $RX/deep-x79-225pa-t3.txt; do [ -s $f ] || log "RESULT xpipe: note -- reference $f missing (information only)"; done
else
  echo "+ image $IMAGE on this host: $(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null || echo MISSING)"
  echo "+ yaml $Y passes RADIANCE_DS41_PIPE: $(grep -c '^ *RADIANCE_DS41_PIPE:' $Y 2>/dev/null)"
fi
mkd $O $O/guard $STATSDIR
evict
done_arms=""
for a in ${XSPAN_ARMS:-S SP}; do stopfile && break; arm $a && done_arms="$done_arms $a"; done
summary pipe-A pipe-S "S (span 4, MNBT 8256) vs A (f-pipe's serial arm, span 2, MNBT 4160)"
case " $done_arms " in *" SP "*) summary pipe-S pipe-SP "SP (span 4 + RADIANCE_DS41_PIPE=1, pieces of 4096) vs S (span 4 serial)"; summary pipe-B pipe-SP "SP (span 4 pipelined) vs f-pipe's B (span 2 pipelined)" ;; esac
unset $SWITCHES
down
log "RESULT xpipe 57-p4zyzzzzz-ds-f2-span4: done -- arms [$done_arms ]"
