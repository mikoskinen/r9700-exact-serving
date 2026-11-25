#!/bin/bash
# dsm-idxbal.sh (10.11.2026, queue-ar/dsmain, ds41-idxb agent, lever I1) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-kb-idxbal.sh. DeepSeek-V4.1 Flash EXL3 2.25 bpw, gpu02, all 4 cards, TP4, MML 524288, NST 0, the t3 env and
# the dsm serve line (lib-29 best_vars + MNBT 4160 + --live). IMAGE (XIDX_IMAGE): radiance-local:ds41-idxb-<sha8>-5c13e555-l =
# the gg -l image (d5bed91f) + ds41-idxb's bin/radiance (fn4 registry: idx_split) and deepseek41.so (the balanced deal);
# /opt/radiance/share/radiance/idxb-PROVENANCE.txt.
# THE LEVER (idx_split, POST /admin/fn4, in process): the indexer's row split deals a step's n = ceil(T / 256) score runs
#   ceil(n / 4) a rank. Where 4 does not divide n the deal leaves ranks short: a MIXED step (a 2048-row prefill chunk beside
#   decode rows, planned at 2112 here) is 9 runs of ~230 rows dealt 3 / 3 / 3 / 0, so three ranks score 3/4 of the step's
#   rows each while rank 3 waits at the gather (the max rank 33 % over the mean); the last partial chunk of a prompt too.
#   idx_split 1 gives every rank an equal block of rows in equal runs (12 runs of ~171, 3 a rank). A LONE prefill step of
#   4096 (or 2048) rows is 16 (8) runs, 4 (2) a rank, under both values: nothing moves there.
# THE FINDING IT ANSWERS: the 2:2:1:1 index_score "skew" of the dsm-225-prof 128k window (138 / 145 / 68 / 68 ms a step)
#   is opsplit.py's per-instance matching (a row_part instance runs on one rank only; rank 0's window lacks the ratio-1
#   parts 8-15, so ranks 2-3 count only their ratio-2 instances). Per rank summed, that window is 4135 / 4148 / 4195 /
#   4260 ms (max/min +3.0 %: the causal reach, later rows see more positions; contiguous blocks keep it). So EXPECTED:
#   fresh lone prefill +-0 (only the last partial step moves); mb pf (4 decode streams + a 32k prefill = mixed steps)
#   ~+0.3..1 % prefill tok/s (index_score is ~9 % of a step and the max rank's rows drop 684 -> 513 at 2052 rows).
# EXACT CLAIM: idx_split 1's every output is idx_split 0's, byte for byte: a row's scores (index_score: a wave a row, the
#   form by the pass height, kept on the same side of 48 rows), candidates and selection (exact selections) are its own
#   whichever run holds it, and the gather lays every row where the plain deal does (ds41_index_split.h; the r4d_emu
#   ds41idx test compares every score bit). Checked here byte for byte; a difference is a VIRHE line (NOT EXACT).
# START 1 (unprofiled, in process; decode before prefill, TESTING 9.11.): gate -> modes (the t3 notes + the deal note) ->
#   switch note (GET at start idx_split 0 / epoch 0; idx_split 2 refused; 1 then 0 applied at a step) -> warm-up answers vs
#   stock -> DECODE CHECK mb c4 30 s at 0 and at 1 (decode steps are never split: must not move) -> FRESH PREFILL 16k / 64k /
#   136k in blocks A B B A (A = idx_split 0, B = 1; each prompt fresh, a thermal line each) -> MB PF in A B B A (the mixed
#   steps, where the lever acts) -> IDENTITY B cold: the 28k copy (31k cold prompt) and the 136k deep answer at idx_split 1,
#   each vs today's files (runs/xdsm copy-dsm-225-i1-run1.txt, deep-dsm-225.txt: the plain deal, main image); then the same
#   two at idx_split 0 (their prompts now cached: only the tail is A's; information).
# START 2 (--profile-ops, RADIANCE_PROFILE_EVERY 16; PROFILED, not a speed): IDENTITY A cold first (copy + deep at idx_split
#   0, a fresh process) vs B (start 1) and vs today's files -> W0 mb pf at idx_split 0 -> W1 mb pf at 1 -> W2 one fresh 128k
#   prompt at 1; idxbal.py per window: index_score per rank summed (acceptance: W2 and W1 within ~5 % across ranks), the
#   row_parts each rank ran; ds_opprof.py classes for W2.
# ~40 min. NO cc_sim, no 2.9 (user 10.11.). No power / host / fan settings; production compose files read only.
# DRYRUN=1 bash dsm-idxbal.sh : prints every command, runs and writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-idxb-tp4.yaml                        # = ds41-gg-tp4.yaml + RADIANCE_DS41_IDX_BALANCE (unset here)
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=$R/xdsm/idxb; export STATSDIR=$O/stats
RT=$R/xdsm                                      # today's files: copy-dsm-225-i1-run1.txt, deep-dsm-225.txt
export IMAGE=${XIDX_IMAGE:-radiance-local:ds41-idxb-d30fd7d9-5c13e555-l}
ITEM=57-p4zyzzzzz-ds-kb-idxbal
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
DQ=$QA/deepq.py                                 # 105's deepq.py: one fixed ~135k prompt, 512 greedy tokens
IB=$HD/idxbal.py; OPP=$HD/ds_opprof.py
DEPTHS=${XIDX_DEPTHS:-16384 65536 136000}
ORDER=${XIDX_ORDER:-ABBA}
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DS41_IDX_BALANCE"
# ---------------------------------------------------------------- helpers (dsm-pipe.sh's / dsm-gemvacq.sh's, 10.11., copied; log tag xidx)
t3env() {   # 105's t3 arm, every other switch unset (RADIANCE_DS41_IDX_BALANCE unset: idx_split 0 at start)
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
}
modelvars225() {   # -> MODEL, HP29, LIM, WARMREF; lib-29 best_vars (NST 0) + MNBT 4160 + --live
  export MODEL=$M225; HP29=19456; LIM=23; WARMREF=$WARM_REF_225
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
  log "RESULT guard $tag: $st, 1-user ${ms} ms/step (limit $lim; bz dsd-d0 main image 15.59)"
  case $st in SLOW*) return 2 ;; esac
}
FOLLOW_PID=""
follow() { [ "$DRY" = 1 ] && { echo "+ docker logs -f $CN > $1/engine.log &"; return; }
  docker logs -f $CN > $1/engine.log 2>&1 & FOLLOW_PID=$!; sleep 2; }
unfollow() { [ -n "$FOLLOW_PID" ] && { kill $FOLLOW_PID 2>/dev/null; wait $FOLLOW_PID 2>/dev/null; FOLLOW_PID=""; }; return 0; }
trap 'unfollow; down' EXIT
trap 'exit 143' TERM INT
thermline() {   # thermline T0 T1: the window's full-power rows (heatsoak.sh thermwin) or the cards now
  type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"
}
cmpfile() {   # cmpfile A B -> IDENTICAL (n chars) | DIFFERENT at char k (na vs nb chars) | not compared
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
idline() {   # idline LABEL RESULT: IDENTICAL -> plain, not compared -> plain, else VIRHE
  case "$2" in IDENTICAL*|"not compared"*) log "RESULT xidx identity $1: $2" ;;
    *) log "RESULT xidx identity $1: VIRHE $2 -- NOT EXACT (the balanced deal is exact by construction: a defect, or a reference from another line)" ;; esac
}
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xidx $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xidx $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  case $r in IDENTICAL*) log "RESULT xidx $1 warm-up answers vs stock ($(basename $2)): $r" ;;
    *) log "RESULT xidx $1 warm-up answers vs stock ($(basename $2)): VIRHE $r -- NOT EXACT at idx_split 0" ;; esac
}
modes() {   # modes TAG: the t3 notes + the deal note (idx_split 0 at start, the split on) -> 0, else 1 (stopped)
  [ "$DRY" = 1 ] && { echo "+ modes $1: indexer note (mxfp4, latent bf16, compressor tail bound, router selection+sort separate, AR_PCN separate at 0 layers, row split on), ARGATHER note, staging line, no acquire-rule line; deal note 'idx_split 0 ... 1 = an equal block of rows a rank'"; return 0; }
  local MLF DL SL ok=1
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer: candidate' | sed 's/^radiance: //')
  DL=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer deal:' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  log "RESULT xidx $1 modes: $(echo "$MLF" | cut -c1-900) | deal: ${DL:-none} | staging: ${SL:-none}"
  for w in "row split on" "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate" "post-collapse-norm separate at 0 layers" "next collapse-norm separate at 0 layers"; do
    case "$MLF" in *"$w"*) ;; *) log "RESULT xidx $1: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  case "$DL" in *"idx_split 0"*"an equal block of rows a rank"*) ;; *) log "RESULT xidx $1: VIRHE no deal note 'idx_split 0 ... an equal block of rows a rank' (not the idxb plugin, or the split off): ${DL:-none}"; ok=0 ;; esac
  docker logs $CN 2>&1 | grep -q 'the MoE gather and its all-reduce as one kernel' || { log "RESULT xidx $1: VIRHE no RADIANCE_DS41_ARGATHER note"; ok=0; }
  [ -n "$SL" ] || { log "RESULT xidx $1: VIRHE no prefill staging line (levels refused?)"; ok=0; }
  docker logs $CN 2>&1 | grep -q 'the acquire rule plays' && { log "RESULT xidx $1: VIRHE an acquire-rule line (RADIANCE_AQL_ACQ should be unset)"; ok=0; }
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
fn4code() {   # fn4code JSON: POST, print the HTTP status
  [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null
}
setsw() {   # setsw V: idx_split V + a new epoch, applied at the next step (a 4-token request), then checked -> 0 / 1
  fn4set "{\"idx_split\": $1, \"epoch_bump\": 1}"; nudge
  [ "$DRY" = 1 ] && return 0
  fn4get | python3 -c 'import json,sys; s=json.loads(sys.stdin.read()); sys.exit(0 if s["switches"].get("idx_split") == int(sys.argv[1]) and not s["pending"] else 1)' $1 2>/dev/null \
    || { log "RESULT xidx: VIRHE idx_split $1 not applied: $(fn4get | cut -c1-300)"; return 1; }
}
swnote() {   # swnote TAG: idx_split 0 / epoch 0 at start, 2 refused, 1 then 0 applied at a step (deepseek41's hook found)
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (expect idx_split 0, epoch 0); POST idx_split 2 -> 400; POST idx_split 1 / 0, each + a 4-token request -> applied, plugins >= 2"; return 0; }
  local s0 s1 s2 c2
  s0=$(fn4get)
  c2=$(fn4code '{"idx_split": 2}')
  fn4set '{"idx_split": 1}'; nudge; s1=$(fn4get)
  fn4set '{"idx_split": 0}'; nudge; s2=$(fn4get)
  log "RESULT xidx $1 switch note: at start $s0 | idx_split 2 -> HTTP $c2 | after idx_split 1: $s1 | after 0: $s2 | core log: $(docker logs $CN 2>&1 | grep -E 'fn4: switches' | tail -2 | sed 's/^[A-Z] *//' | tr '\n' '|' | cut -c1-400)"
  python3 -c 'import json,sys
s0, s1, s2 = (json.loads(x) for x in sys.argv[1:4])
ok = s0["switches"].get("idx_split") == 0 and s0["epoch"] == 0 and s1["switches"].get("idx_split") == 1 and s1["plugins"] >= 2 \
     and s2["switches"].get("idx_split") == 0 and not s1["pending"] and not s2["pending"] and s2["epoch"] == 2
sys.exit(0 if ok else 1)' "$s0" "$s1" "$s2" 2>/dev/null && [ "$c2" = 400 ]
}
mbline() { grep '^MB ' $1 2>/dev/null | head -1 | cut -c4-; }
mbc4() {   # mbc4 TAG V: idx_split V, mb c4 30 s
  setsw $2 || return 1
  run "(cd $D && timeout 600 python3 mb.py --tag xidx-$1 --phases c4 --dur 30 --warm 10 > $O/$1.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xidx $1 (idx_split $2) mb c4 30 s: $(mbline $O/$1.txt | cut -c1-260)"
}
pfd() {   # pfd TAG DEPTH: one fresh prefill of DEPTH tokens (nonce first; chars at 3.95 a token, no calibration request) + thermal line
  local t0=$(date +%s)
  run "(cd $D && timeout 1800 python3 ds41_tests.py prefill --depths $2 --cpt 3.95 --calibrate 0 --reps 1 > $O/$1-pf$2.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xidx $1 fresh prefill $2: $(grep '^J ' $O/$1-pf$2.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); print("%s tok, cached %s, TTFT %s s, %s tok/s, %s" % (d.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("prefill_tps"), d.get("status")))' 2>&1 | cut -c1-200) | $(thermline $t0 $(date +%s))"
}
mbpf() {   # mbpf TAG: mb pf (4 decode streams + a fresh ~32k prefill: mixed steps) + a thermal line
  local t0=$(date +%s)
  run "(cd $D && timeout 1200 python3 mb.py --tag xidx-$1 --phases pf --warm 10 > $O/$1.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xidx $1 mb pf: $(mbline $O/$1.txt | cut -c1-300) | $(thermline $t0 $(date +%s))"
}
copyt() {   # copyt TAG: the 28k copy after a ~31k-token prompt (105's prompt, offset 0: the bytes of today's copy-dsm-225-i1)
  local t0=$(date +%s)
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset 0 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xidx $1 copy: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300) | $(thermline $t0 $(date +%s))"
}
deepq() {   # deepq TAG: the fixed ~135k prompt and a 512-token greedy answer (deepq.py's bytes)
  local t0=$(date +%s)
  run "(cd $D && timeout 1800 python3 $DQ $O/deep-$1.txt 135000 512 > $O/deep-$1.out 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xidx $1 deep probe: $(grep -m1 '^DEEPQ ' $O/deep-$1.out | cut -c7- | cut -c1-200) | $(thermline $t0 $(date +%s))"
}
balnote() {   # balnote: rank 0's note at the first balanced step (the path ran)
  [ "$DRY" = 1 ] && return 0
  local n; n=$(docker logs $CN 2>&1 | grep -m1 'idx_split 1 took its first step' | sed 's/^radiance: //' | cut -c1-260)
  [ -n "$n" ] && log "RESULT xidx the balanced deal ran: $n" || log "RESULT xidx: VIRHE no 'idx_split 1 took its first step' note -- B measured the plain deal"
}
abphase() {   # abphase NAME FUNC...: blocks in ORDER (A = idx_split 0, B = 1), FUNC TAG run in each block
  local nm=$1 i=0 arm v; shift
  for arm in $(echo $ORDER | grep -o .); do
    stopfile && break
    [ $arm = A ] && v=0 || v=1
    setsw $v || return 1
    "$@" idx-$nm-b$i$arm
    i=$((i + 1))
  done
}
pfblock() { local d; for d in $DEPTHS; do stopfile && break; pfd $1 $d; done; }
abpf() {   # B vs A per depth: pooled (mean B / mean A) and each pair's B/A (prefill tok/s and TTFT)
  [ "$DRY" = 1 ] && { echo "+ RESULT xidx fresh prefill B vs A per depth (pooled, pairs)"; return 0; }
  log "RESULT xidx fresh prefill idx_split 1 (B) vs 0 (A), order $ORDER, in process: $(python3 - $O "$DEPTHS" "$ORDER" <<'PY' 2>&1 | cut -c1-1500
import json, sys, statistics as st
O, depths, order = sys.argv[1], sys.argv[2].split(), sys.argv[3]
def j(tag, d):
    try:
        for l in open(f"{O}/{tag}-pf{d}.txt"):
            if l.startswith("J "): return json.loads(l[2:])
    except OSError: pass
    return {}
out = []
for d in depths:
    v = {"A": [], "B": []}; tt = {"A": [], "B": []}; seq = []
    for i, arm in enumerate(order):
        r = j(f"idx-pf-b{i}{arm}", d)
        if r.get("prefill_tps"): v[arm].append(r["prefill_tps"]); tt[arm].append(r["ttft"]); seq.append((arm, r["prefill_tps"]))
    pairs = [(y / x if a == "A" else x / y) for (a, x), (b, y) in zip(seq[0::2], seq[1::2]) if a != b]
    if v["A"] and v["B"]:
        out.append("%s: A %.1f B %.1f tok/s (%+.2f %%; pairs %s), TTFT A %.2f B %.2f s" % (d, st.mean(v["A"]), st.mean(v["B"]),
                   100 * (st.mean(v["B"]) / st.mean(v["A"]) - 1), " ".join("%+.2f" % (100 * (p - 1)) for p in pairs), st.mean(tt["A"]), st.mean(tt["B"])))
    else: out.append(f"{d}: incomplete")
print("; ".join(out))
PY
)"
}
abmbpf() {   # B vs A over the mb pf blocks: prefill tok/s, TTFT, decode kept, ms a step during
  [ "$DRY" = 1 ] && { echo "+ RESULT xidx mb pf B vs A (pooled, pairs)"; return 0; }
  log "RESULT xidx mb pf idx_split 1 (B) vs 0 (A), order $ORDER, in process (the mixed steps): $(python3 - $O "$ORDER" <<'PY' 2>&1 | cut -c1-1200
import json, sys, statistics as st
O, order = sys.argv[1], sys.argv[2]
rows = []
for i, arm in enumerate(order):
    try:
        for l in open(f"{O}/idx-mbpf-b{i}{arm}.txt"):
            if l.startswith("MB "): rows.append((arm, json.loads(l[3:]))); break
    except OSError: pass
out = []
for k in ("prefill_tps", "prefill_ttft", "decode_kept", "ms_step_during"):
    a = [r[k] for arm, r in rows if arm == "A" and r.get(k) is not None]; b = [r[k] for arm, r in rows if arm == "B" and r.get(k) is not None]
    seq = [(arm, r.get(k)) for arm, r in rows]
    pairs = [(y / x if p == "A" else x / y) for (p, x), (q, y) in zip(seq[0::2], seq[1::2]) if p != q and x and y]
    if a and b: out.append("%s A %.3g B %.3g (%+.2f %%; pairs %s)" % (k, st.mean(a), st.mean(b), 100 * (st.mean(b) / st.mean(a) - 1), " ".join("%+.2f" % (100 * (p - 1)) for p in pairs)))
print("; ".join(out) if out else "incomplete")
PY
)"
}
loglines() { [ "$DRY" = 1 ] && echo L || docker logs $CN 2>&1 | wc -l; }
# ---------------------------------------------------------------- start 1: in process, unprofiled
start1() {
  local tag=idx1 P=$O/idx1 rc
  t3env; modelvars225
  log "RESULT xidx $tag: start IMAGE=$IMAGE MODEL=$MODEL $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
  mkd $P; run "rm -f $P/marks.txt"
  gate $tag $LIM; rc=$?
  [ $rc = 1 ] && { log "RESULT xidx $tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  [ $rc = 2 ] && { log "RESULT xidx $tag: VIRHE slow state after one restart -> stopped"; down; return 1; }
  modes $tag || { log "RESULT xidx $tag: VIRHE the modes are not the expected ones -> stopped"; down; return 1; }
  swnote $tag || { log "RESULT xidx $tag: VIRHE the switch is not as expected (idx_split 0 / epoch 0 at start, 2 refused, 1 / 0 applied, deepseek41's hook found) -> stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARMREF
  follow $P
  # decode check: decode steps are never split, so neither value may move them
  stopfile || mbc4 idx-c4-A 0
  stopfile || mbc4 idx-c4-B 1
  [ "$DRY" = 1 ] || log "RESULT xidx decode check mb c4 30 s idx_split 1 vs 0: $(python3 -c 'import json,sys
def mb(f):
    for l in open(f):
        if l.startswith("MB "): return json.loads(l[3:])
    return {}
try: a, b = mb(sys.argv[1]), mb(sys.argv[2])
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
print("; ".join("%s %s -> %s (%+.2f %%)" % (k, a.get(k), b.get(k), 100 * (b[k] / a[k] - 1)) if a.get(k) and b.get(k) else "%s %s -> %s" % (k, a.get(k), b.get(k))
                for k in ("agg_dec_tps", "per_stream_dec_p50", "ms_decode_step")))' $O/idx-c4-A.txt $O/idx-c4-B.txt 2>&1 | cut -c1-400) (one 30 s window each: the start-to-start-free in-process spread; decode is not split, so expected 0 within noise)"
  stopfile || { abphase pf pfblock; abpf; }
  stopfile || { abphase mbpf mbpf; abmbpf; }
  balnote
  # identity: B cold (idx_split 1 on prompts this process has not seen) vs today's files; then A over the cached prefixes
  if ! stopfile; then
    setsw 1 && { copyt idx-B; deepq idx-B; }
    setsw 0 && { copyt idx-Acached; deepq idx-Acached; }
    if [ "$DRY" = 1 ]; then echo "+ identity: copy / deep idx-B vs $RT/copy-dsm-225-i1-run1.txt / $RT/deep-dsm-225.txt; idx-Acached vs idx-B"
    else
      idline "28k copy idx_split 1 (cold) vs today's copy-dsm-225-i1 (plain deal, main image)" "$(cmpfile $O/copy-idx-B-run1.txt $RT/copy-dsm-225-i1-run1.txt)"
      idline "136k deep answer idx_split 1 (cold) vs today's deep-dsm-225" "$(cmpfile $O/deep-idx-B.txt $RT/deep-dsm-225.txt)"
      log "RESULT xidx identity (information) idx_split 0 over idx_split 1's cached prefixes (only the uncached tail is A's): copy $(cmpfile $O/copy-idx-Acached-run1.txt $O/copy-idx-B-run1.txt); deep $(cmpfile $O/deep-idx-Acached.txt $O/deep-idx-B.txt)"
    fi
  fi
  unfollow
  run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
  down
  log "RESULT xidx $tag: done"
}
# ---------------------------------------------------------------- start 2: --profile-ops; A cold identity, then the windows
start2() {
  local tag=idx2prof E=$O/idx2prof-engine.log l0 l1 l2 l3 l4 t2 t3
  t3env; modelvars225
  export PROFEVERY=${XIDX_PROFEVERY:-16} XARGS="$XARGS --profile-ops" STATSDIR=$O/stats-prof
  mkd $STATSDIR
  log "RESULT xidx $tag: start IMAGE=$IMAGE MODEL=$MODEL --profile-ops RADIANCE_PROFILE_EVERY=$PROFEVERY (PROFILED: not a speed) $(desc)"
  restart $tag || { keeplog $tag-fail; down; log "RESULT xidx $tag: VIRHE not healthy -- no A-cold identity, no profile"; export PROFEVERY= STATSDIR=$O/stats; return 1; }
  local ms; ms=$(check $tag 1)
  [ "$DRY" = 1 ] || log "RESULT xidx $tag warm-up + 1-user fastcheck (PROFILED, not gated): $ms ms/step; deal: $(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer deal:' | sed 's/^radiance: //' | cut -c1-200)"
  # A cold (a fresh process: idx_split 0 has not seen these prompts); profiling adds syncs, not other arithmetic
  setsw 0 && { copyt idx-A; deepq idx-A; }
  if [ "$DRY" = 1 ]; then echo "+ identity: copy / deep idx-B vs idx-A (both cold); idx-A vs today's files"
  else
    idline "28k copy idx_split 1 (start 1, cold) vs idx_split 0 (start 2, cold)" "$(cmpfile $O/copy-idx-B-run1.txt $O/copy-idx-A-run1.txt)"
    idline "136k deep answer idx_split 1 (start 1, cold) vs idx_split 0 (start 2, cold)" "$(cmpfile $O/deep-idx-B.txt $O/deep-idx-A.txt)"
    idline "28k copy idx_split 0 (cold) vs today's copy-dsm-225-i1" "$(cmpfile $O/copy-idx-A-run1.txt $RT/copy-dsm-225-i1-run1.txt)"
    idline "136k deep answer idx_split 0 (cold) vs today's deep-dsm-225" "$(cmpfile $O/deep-idx-A.txt $RT/deep-dsm-225.txt)"
  fi
  # the windows: W0 mb pf at 0, W1 mb pf at 1, W2 a fresh 128k prompt at 1 (edges = the dumps before them, +-15 steps)
  setsw 0; l0=$(loglines)
  run "(cd $D && timeout 1200 python3 mb.py --tag xidx-$tag-w0 --phases pf --warm 10 > $O/$tag-w0.txt 2>&1)"
  run "sleep 3"; l1=$(loglines)
  setsw 1; l2=$(loglines)
  run "(cd $D && timeout 1200 python3 mb.py --tag xidx-$tag-w1 --phases pf --warm 10 > $O/$tag-w1.txt 2>&1)"
  run "sleep 3"; l3=$(loglines)
  run "curl -s -m 15 $URL/stats > $O/$tag-pf128-sta.json 2>/dev/null"; l3=$(loglines); t2=$(date +%s)
  run "(cd $D && timeout 1800 python3 ds41_tests.py prefill --depths 131072 --cpt 3.95 --calibrate 0 --reps 1 > $O/$tag-pf128.txt 2>&1)"
  run "sleep 3; curl -s -m 15 $URL/stats > $O/$tag-pf128-stb.json 2>/dev/null"; t3=$(date +%s)
  run "docker stop -t 60 $CN > /dev/null 2>&1; docker logs $CN > $E 2>&1"
  down
  if [ "$DRY" = 1 ]; then echo "+ idxbal.py $E W0 [l0, l1] / W1 [l2, l3] / W2 [l3, end] (index_score per rank, limit 5 %); ds_opprof.py W2"
  else
    log "RESULT xidx $tag W0 mb pf at idx_split 0 (PROFILED): $(mbline $O/$tag-w0.txt | cut -c1-200)"
    log "RESULT xidx $tag W1 mb pf at idx_split 1 (PROFILED): $(mbline $O/$tag-w1.txt | cut -c1-200)"
    timeout 600 python3 -I $IB $E --from $l0 --to $l1 --label W0-mbpf-idx0 --json $O/$tag-w0-idx.json > $O/$tag-w0-idx.txt 2>&1
    timeout 600 python3 -I $IB $E --from $l2 --to $l3 --label W1-mbpf-idx1 --json $O/$tag-w1-idx.json > $O/$tag-w1-idx.txt 2>&1
    timeout 600 python3 -I $IB $E --from $l3 --label W2-pf128-idx1 --json $O/$tag-w2-idx.json > $O/$tag-w2-idx.txt 2>&1
    for w in w0 w1 w2; do while IFS= read -r l; do log "RESULT xidx $tag ${l:0:1500}"; done < $O/$tag-$w-idx.txt; done
    timeout 900 python3 -I $OPP $E --from $l3 --label $tag-pf128 --json $O/$tag-pf128-op.json --stats $O/$tag-pf128-sta.json $O/$tag-pf128-stb.json > $O/$tag-pf128-op.txt 2>&1
    log "RESULT xidx $tag W2 fresh 128k prefill at idx_split 1 (PROFILED, not a speed): $(cut -c1-1500 $O/$tag-pf128-op.txt | tr '\n' ' ') | prompt (profiled): $(grep '^J ' $O/$tag-pf128.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); print("%s tok TTFT %s s %s tok/s" % (d.get("prompt_tokens"), d.get("ttft"), d.get("prefill_tps")))' 2>&1 | cut -c1-160) | $(thermline $t2 $t3)"
    log "RESULT xidx $tag acceptance (index_score per rank within ~5 % with the balance on): W1 mixed $(grep -o -- '-> [A-Z]*' $O/$tag-w1-idx.txt | head -1), W2 lone 128k $(grep -o -- '-> [A-Z]*' $O/$tag-w2-idx.txt | head -1); W0 (balance off, mixed) $(grep -o 'spread max/min [-+0-9.]* %' $O/$tag-w0-idx.txt | head -1)"
    run "gzip -f $E"
  fi
  export PROFEVERY= STATSDIR=$O/stats
  log "RESULT xidx $tag: done"
}
# ---------------------------------------------------------------- main
log "RESULT xidx $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) order $ORDER depths [$DEPTHS] yaml $Y | lever I1 idx_split (exact); refs dsm-225: pf 1603.1 tok/s (TTFT 19.6 s), c4 122.3 tok/s 28.2 ms; dsm-225-prof 128k window index_score per rank 4135 / 4148 / 4195 / 4260 ms"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xidx: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xidx: VIRHE image $IMAGE not on gpu02"; exit 1; }
  for v in $SWITCHES; do grep -q "^ *$v:" $Y || { log "RESULT xidx: VIRHE $Y does not pass $v"; exit 1; }; done
  grep -q RADIANCE_PROFILE_EVERY $Y || { log "RESULT xidx: VIRHE $Y lacks RADIANCE_PROFILE_EVERY"; exit 1; }
  [ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -c "idx_split" /opt/radiance/bin/radiance /opt/radiance/share/radiance/architectures/deepseek41.so | grep -c ":[1-9]"' 2>/dev/null)" = 2 ] \
    || { log "RESULT xidx: VIRHE $IMAGE's engine or deepseek41 plugin does not know idx_split"; exit 1; }
  for f in $DQ $IB $OPP $D/mb.py $D/ds41_tests.py; do [ -e $f ] || { log "RESULT xidx: VIRHE $f missing"; exit 1; }; done
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xidx: VIRHE $M225 missing or not verified"; exit 1; }
  for f in $RT/copy-dsm-225-i1-run1.txt $RT/deep-dsm-225.txt; do [ -s $f ] || log "RESULT xidx: note -- reference $f missing (that identity not compared)"; done
else
  echo "+ image $IMAGE on this host: $(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null || echo MISSING)"
  echo "+ yaml $Y passes RADIANCE_DS41_IDX_BALANCE: $(grep -c '^ *RADIANCE_DS41_IDX_BALANCE:' $Y 2>/dev/null); helpers: $(for f in $DQ $IB $OPP; do [ -e $f ] && echo -n "$(basename $f) " || echo -n "MISSING:$f "; done)"
  echo "+ references: $(ls $RT/copy-dsm-225-i1-run1.txt $RT/deep-dsm-225.txt 2>&1 | tr '\n' ' ')"
fi
mkd $O $O/guard $STATSDIR
evict
start1 && { stopfile || start2; }
unset $SWITCHES
down
log "RESULT xidx $ITEM: done -- $O"
