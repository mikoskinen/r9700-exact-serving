#!/bin/bash
# dsm-acc.sh (10.11.2026, queue-ar/dsmain, ds41-acc agent) -- NOT a queue item; the body of 57-p4zyzzzzz-ds-r-confhead.sh (XAC_MODE=conf)
# and 57-p4zyzzzzz-ds-s-tree.sh (XAC_MODE=tree). Two acceptance levers of DeepSeek-V4.1's DSpark drafter on the ds41-acc image
# (ds41-gg 8b311fc + the confidence head and the shadow tree; XAC_IMAGE). ONE START each, EXL3 2.25, TP4, MML 524288, the t3 env
# and serve line of dsm-draftmr.sh's S1 (105's t3 arm + lib-29 best_vars + MNBT 4160 + --live), DSpark NST 5 --draft-window
# adaptive, gg_gemv 2 at start (RADIANCE_DS41_GG_GEMV=2, held by every block), the confidence head's sidecar mounted with the
# .rad (RADIANCE_DS41_DSPARK_CONF=/models/dspark-conf.f32: the release's mtp.2.confidence_head.proj, 5376 f32, sha256 a97178fa...).
#   conf  C1, the confidence gate: blocks over dspark_conf A = 0 (off: today's passes, nothing waits), B = 1 (the reference rule:
#         each sequence verifies the longest prefix with sigmoid(logit) >= 0.5, in place of the adaptive window's stop),
#         C = 150 (the same at 0.5 capped by the window), order ABCCBA (each arm's mean centred on one time); then one OBSERVE
#         block (dspark_conf 2: the head runs, its calibration is counted, nothing cut). Identity: the 28k copy and the greedy
#         set under B, the greedy set under C, vs bz's d0.
#   tree  T1, the shadow tree at position 1 (RADIANCE_DS41_DSPARK_ALT=3: the walk's 2nd, 3rd and 4th best first drafts, each
#         walked on): blocks over dspark_alt A = 0 / B = 1, ABBA, dspark_conf 0 held. A real tree verify is not built (the
#         trunk's compressor, indexer and Engram are not tree-aware); B blocks count, verify by verify, how many more tokens a
#         tree of 1-3 branches x depth 1-5 beside the chain would have emitted (exact for greedy: the accepted path is the one
#         the target's own tokens take), and A vs B measures the branches' own cost. Identity under B vs bz's d0.
# PER START: gate (limit 70 ms) -> modes (the t3 notes, the DSpark lines and the extras line) -> the switch note (start values,
#   refusals) -> warm-up answers vs stock -> BLOCKS (accblocks.py, epoch_bump each block): essay / code (stepprobe 512, each
#   alone between /stats snapshots: tokens a step, acceptance, ms a step, drafts verified a step, essay acceptance BY POSITION),
#   editprobe (1500 tok), mb c4 and mb c8 (XAC_DUR 75 s, warm 10), each with its draft_conf delta (the gate's cut histogram,
#   calibration; the shadow tree's tree tokens a step) -> summary lines (each arm vs A pooled + pairs, t95) -> identity.
# NO cc_sim, no 2.9, no BetterBench. No power / host / fan settings; production compose files read only (--project-directory).
# DRYRUN=1 bash dsm-acc.sh : prints every command, runs and writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-acc-tp4.yaml
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
MODE=${XAC_MODE:-conf}
case $MODE in conf|tree) ;; *) echo "XAC_MODE=$MODE: conf or tree"; exit 1 ;; esac
O=$R/xdsm/acc-$MODE; export STATSDIR=$O/stats
RD=$R/xdsm/draft                                # bz's d0 / d5a files
export IMAGE=${XAC_IMAGE:-radiance-local:ds41-acc-1f2f03b5-5c13e555-l}
DUR=${XAC_DUR:-75}
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
EP=$HD/editprobe.py; ACB=$HD/accblocks.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
SIDECAR=dspark-conf.f32; SIDECAR_SHA=a97178fa6374686d95f833e38ad8aaf51f989cdbb1fe2312ce7b60aefc1e5b6d
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
if [ $MODE = conf ]; then
  ITEM=57-p4zyzzzzz-ds-r-confhead; SW=dspark_conf; ARMS=${XAC_ARMS:-A=0,B=1,C=150}; ORDER=${XAC_ORDER:-ABCCBA}
  FIXED='{"gg_gemv": 2}'; NALT=
else
  ITEM=57-p4zyzzzzz-ds-s-tree; SW=dspark_alt; ARMS=${XAC_ARMS:-A=0,B=1}; ORDER=${XAC_ORDER:-ABBA}
  FIXED='{"gg_gemv": 2, "dspark_conf": 0}'; NALT=3
fi
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DS41_DSPARK_CONF RADIANCE_DS41_DSPARK_ALT RADIANCE_FN4_DSPARK_CONF RADIANCE_FN4_DSPARK_ALT"
t3env() {   # 105's t3 arm, every other switch unset
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
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
  log "RESULT guard $tag: $st, 1-user ${ms} ms/step (limit $lim; bz d5a 32.98)"
  case $st in SLOW*) return 2 ;; esac
}
FOLLOW_PID=""
follow() { [ "$DRY" = 1 ] && { echo "+ docker logs -f $CN > $1/engine.log &"; return; }
  docker logs -f $CN > $1/engine.log 2>&1 & FOLLOW_PID=$!; sleep 2; }
unfollow() { [ -n "$FOLLOW_PID" ] && { kill $FOLLOW_PID 2>/dev/null; wait $FOLLOW_PID 2>/dev/null; FOLLOW_PID=""; }; return 0; }
trap 'unfollow; down' EXIT
trap 'exit 143' TERM INT
thermline() { type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"; }
cmpfile() {   # cmpfile A B -> IDENTICAL (n chars) | DIFFERENT at char k (na vs nb chars) | missing
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xac $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xac $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  case $r in IDENTICAL*) log "RESULT xac $1 warm-up answers vs stock ($(basename $2)): $r" ;;
    *) log "RESULT xac $1 warm-up answers vs stock ($(basename $2)): VIRHE $r -- NOT EXACT vs the tuned3l line" ;; esac
}
modes() {   # modes TAG: the serve log's notes must be the t3 line's -> 0, else 1 (start stopped)
  [ "$DRY" = 1 ] && { echo "+ modes $1: indexer note, ARGATHER note, staging line, no acquire-rule line"; return 0; }
  local MLF SL ok=1
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  log "RESULT xac $1 modes: $(echo "$MLF" | cut -c1-900) | staging: ${SL:-none}"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate"; do
    case "$MLF" in *"$w"*) ;; *) log "RESULT xac $1: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  docker logs $CN 2>&1 | grep -q 'the MoE gather and its all-reduce as one kernel' || { log "RESULT xac $1: VIRHE no RADIANCE_DS41_ARGATHER note"; ok=0; }
  [ -n "$SL" ] || { log "RESULT xac $1: VIRHE no prefill staging line (levels refused?)"; ok=0; }
  docker logs $CN 2>&1 | grep -q 'the acquire rule plays' && { log "RESULT xac $1: VIRHE an acquire-rule line (RADIANCE_AQL_ACQ should be unset)"; ok=0; }
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
copyt() {   # copyt TAG OFFSET: the copy test after a ~31k-token cold prompt (105's prompt: same offset -> same text)
  local t0=$(date +%s)
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset $2 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xac $1 copy after a cold prompt: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300) | $(thermline $t0 $(date +%s))"
}
gsetc1() { run "(cd $D && timeout 900 python3 $GS $URL $O/$1-gs-c1.jsonl --conc 1 --max-tokens 192 > $O/$1-gs-c1.txt 2>&1)"; }
gcmp() { [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }; [ -s "$1" ] && [ -s "$2" ] || { echo "not compared (missing $1 or $2)"; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
draftlines() {   # draftlines TAG: every drafter / speculation line of the serve log, logged
  [ "$DRY" = 1 ] && { echo "+ RESULT xac $1 drafter lines"; return 0; }
  log "RESULT xac $1 drafter / speculation lines: $(docker logs $CN 2>&1 | grep -iE 'dspark|drafter|speculat|draft-window|draft window|confidence|shadow' | sed 's/^[A-Z][^ ]* //' | sort -u | head -10 | tr '\n' '|' | cut -c1-1800)"
}
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
fn4code() {   # fn4code JSON: POST, print the HTTP status
  [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null
}
swnote() {   # swnote TAG: the switches at start (gg_gemv 2, dspark_conf 0, dspark_alt 0), the refusals, the lever on + back to 0
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (gg_gemv 2, dspark_conf 0, dspark_alt 0); dspark_conf 3 -> 400; dspark_alt 1 -> 400 (conf) / 200 (tree); $SW 1 + a 4-token request, back to 0"; return 0; }
  local s0 s1 s2 c3 ca want
  s0=$(fn4get); c3=$(fn4code '{"dspark_conf": 3}'); ca=$(fn4code '{"dspark_alt": 1}')
  [ $MODE = tree ] && fn4set '{"dspark_alt": 0}'
  fn4set "{\"$SW\": 1}"; nudge; s1=$(fn4get)
  fn4set "{\"$SW\": 0}"; nudge; s2=$(fn4get)
  log "RESULT xac $1 switch note: at start $s0 | dspark_conf 3 -> HTTP $c3 | dspark_alt 1 -> HTTP $ca | after $SW 1: $s1 | after 0: $s2"
  want=$([ $MODE = conf ] && echo 400 || echo 200)
  [ "$c3" = 400 ] && [ "$ca" = "$want" ] || return 1
  python3 -c 'import json,sys
s0, s1, s2 = (json.loads(x) for x in sys.argv[1:4]); sw = sys.argv[4]
w = s0["switches"]
ok = w.get("gg_gemv") == 2 and w.get("dspark_conf") == 0 and w.get("dspark_alt") == 0 and s1["switches"].get(sw) == 1 \
     and s2["switches"].get(sw) == 0 and not s1["pending"] and not s2["pending"]
sys.exit(0 if ok else 1)' "$s0" "$s1" "$s2" $SW 2>/dev/null
}
blocks() {   # blocks TAG ARMS ORDER LABEL: accblocks.py and the per-block lines
  run "timeout 7200 python3 -I $ACB --url $URL --mb-dir $D --edit $EP --label $4 --switch $SW --arms '$2' --order $3 --dur $DUR --warm 10 --fixed '$FIXED' --out $O/$4-blocks.jsonl > $O/$4-blocks.txt 2>&1"
  [ "$DRY" = 1 ] && return 0
  while IFS= read -r l; do log "RESULT xac $1 ${l:0:1500}"; done < <(grep '^BLOCK ' $O/$4-blocks.txt)
}
summary() {   # summary TAG LABEL: each arm vs A pooled + pairs + t95; the gate's cuts and calibration; the shadow tree
  [ "$DRY" = 1 ] && { echo "+ RESULT xac $1 summary ($2): each arm vs A pooled + pairs; gate cuts / calibration; shadow tree tokens a step"; return 0; }
  log "RESULT xac $1 summary ($2; $SW arms $ARMS, in process, order $ORDER, mb $DUR s, fixed $FIXED): $(grep '^ACC ' $O/$2-blocks.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=2: "-" if v is None else format(v, ".%df" % n)
out = []
for k, v in d["metrics"].items():
    m = v["mean"]
    s = "%s %s" % (k, " / ".join("%s %s" % (a, f(m.get(a))) for a in sorted(m)))
    for a in sorted(m):
        if a == "A" or a not in v: continue
        e = v[a]
        s += " (%s/A %s %%; pairs %s; t95 +- %s %%)" % (a, "-" if e.get("pooled") is None else "%+.2f" % (100 * (e["pooled"] - 1)),
             " ".join("%+.2f" % (100 * (x - 1)) for x in e.get("pairs", [])) or "-", "-" if e.get("ci95") is None else "%.2f" % (100 * e["ci95"]))
    out.append(s)
print("; ".join(out) + " | %d blocks, %d flagged, %s s" % (d["blocks"], d["blocks_bad"], d["wall_s"]))' 2>&1 | cut -c1-5000)"
  log "RESULT xac $1 extras ($2; each arm summed over its blocks): $(grep '^ACC ' $O/$2-blocks.txt | cut -c5- | python3 -c 'import json,math,sys
d = json.loads(sys.stdin.read())
out = []
for arm, phs in sorted(d.get("sums", {}).items()):
    for ph, a in phs.items():
        dc = a.get("draft_conf") or {}
        c, sh = dc.get("conf") or {}, dc.get("shadow") or {}
        s = "%s %s: %s decode steps, %s tokens" % (arm, ph, a.get("decode_steps"), a.get("decode_tokens"))
        if c.get("cuts"):
            s += "; gate %s cuts, hist %s, drafts %s" % (c["cuts"], c.get("cut_hist"), c.get("cut_drafts"))
        cal = c.get("calib") or []
        if cal:
            s += "; calibration (pos: decile reached/accepted) " + " ".join("p%d[%s]" % (j + 1, ",".join("%d:%d/%d" % (q, x[0], x[1]) for q, x in enumerate(row) if x[0])) for j, row in enumerate(cal))
        if sh.get("verifies"):
            s += "; shadow %s verifies, p1 miss %s, hits %s, chain %s tok/step, tree tok/step %s" % (sh["verifies"], sh.get("p1_miss"), sh.get("alt_hit"),
                 a.get("chain_tok_step"), " | ".join("%d br: %s" % (i + 1, " ".join("%.3f" % x for x in row)) for i, row in enumerate(a.get("tree_tok_step") or [])))
        out.append(s)
print("; ".join(out) or "no extras counted")' 2>&1 | cut -c1-6000)"
}
# ---------------------------------------------------------------- the start
astart() {
  local tag=acc-$MODE P=$O/acc-$MODE rc ok r
  t3env; export MODEL=$M225; HP29=19456; best_vars
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive" RADIANCE_DS41_GG_GEMV=2
  export RADIANCE_DS41_DSPARK_CONF=/models/$SIDECAR
  [ -n "$NALT" ] && export RADIANCE_DS41_DSPARK_ALT=$NALT
  log "RESULT xac $tag: start IMAGE=$IMAGE MODEL=$MODEL $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
  mkd $P
  gate $tag 70; rc=$?
  [ $rc = 1 ] && { log "RESULT xac $tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  [ $rc = 2 ] && { log "RESULT xac $tag: VIRHE over the gate limit after one restart -> start skipped"; down; return 1; }
  modes $tag || { log "RESULT xac $tag: VIRHE the modes are not the t3 line's -> start stopped"; down; return 1; }
  draftlines $tag
  if [ "$DRY" = 1 ]; then echo "+ the DSpark drafter line, the extras line (confidence head from /models/$SIDECAR${NALT:+, shadow branches $NALT}), the gg_gemv 2 line"
  else
    ok=1
    docker logs $CN 2>&1 | grep -q "DSpark drafter" || { log "RESULT xac $tag: VIRHE no DSpark drafter line"; ok=0; }
    docker logs $CN 2>&1 | grep -q "DSpark extras (ds41-acc): confidence head from /models/$SIDECAR" || { log "RESULT xac $tag: VIRHE no confidence-head line"; ok=0; }
    [ -n "$NALT" ] && { docker logs $CN 2>&1 | grep -q "shadow branches $NALT" || { log "RESULT xac $tag: VIRHE no shadow-branches line"; ok=0; }; }
    [ $ok = 1 ] || { keeplog $tag-nodraft; down; return 1; }
  fi
  swnote $tag || { log "RESULT xac $tag: VIRHE the switches are not as expected -> start stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  follow $P
  stopfile || blocks $tag "$ARMS" $ORDER $MODE
  if [ $MODE = conf ] && ! stopfile; then
    blocks $tag "A=2" A obs     # one OBSERVE block: the head's calibration over today's chain (extra, not paired)
  fi
  if ! stopfile; then
    fn4set "{\"$SW\": 1, \"gg_gemv\": 2, \"epoch_bump\": 1}"
    copyt $tag-i1 0; gsetc1 $tag
    [ "$DRY" = 1 ] || log "RESULT xac $tag identity ran under: $(fn4get | cut -c1-200)"
    if [ $MODE = conf ]; then fn4set '{"dspark_conf": 150, "epoch_bump": 1}'; gsetc1 $tag-c150; fi
    if [ "$DRY" = 1 ]; then echo "+ identity: 28k copy vs bz d0, greedy set c1 vs bz d0 ($SW 1; conf: and dspark_conf 150)"
    else
      r=$(cmpfile $O/copy-$tag-i1-run1.txt $RD/copy-dsd-d0-i1-run1.txt)
      case $r in IDENTICAL*) log "RESULT xac $tag identity ($SW 1): 28k copy vs bz d0 (NST 0, stock): $r" ;;
        *) log "RESULT xac $tag identity ($SW 1): DIFFERENT 28k copy vs bz d0 (NST 0, stock): $r -- reported, not hidden (the lever changes how many drafts are verified / what is measured, never a token)" ;; esac
      r=$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$tag-gs-c1.jsonl)
      case $r in IDENTICAL*) log "RESULT xac $tag identity ($SW 1): greedy set c1 vs bz d0: $r" ;;
        *) log "RESULT xac $tag identity ($SW 1): DIFFERENT greedy set c1 vs bz d0: $r -- reported, not hidden" ;; esac
      if [ $MODE = conf ]; then
        r=$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$tag-c150-gs-c1.jsonl)
        case $r in IDENTICAL*) log "RESULT xac $tag identity (dspark_conf 150): greedy set c1 vs bz d0: $r" ;;
          *) log "RESULT xac $tag identity (dspark_conf 150): DIFFERENT greedy set c1 vs bz d0: $r -- reported, not hidden" ;; esac
      fi
    fi
  fi
  unfollow
  run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
  down
  summary $tag $MODE
  [ $MODE = conf ] && summary $tag obs
  log "RESULT xac $tag: done"
}
# ---------------------------------------------------------------- main
log "RESULT xac $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) mode $MODE switch $SW arms $ARMS order $ORDER mb $DUR s | refs bz d0 (identity), bz d5a (draft numbers)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xac: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xac: VIRHE image $IMAGE not on gpu02"; exit 1; }
  [ -e $Y ] && grep -q '^ *RADIANCE_DS41_DSPARK_CONF:' $Y || { log "RESULT xac: VIRHE $Y missing or does not pass RADIANCE_DS41_DSPARK_CONF"; exit 1; }
  for f in $GS $EP $ACB $D/stepprobe.py $D/mb.py $D/ds41_tests.py; do [ -e $f ] || { log "RESULT xac: VIRHE $f missing"; exit 1; }; done
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xac: VIRHE $M225 missing or not verified"; exit 1; }
  [ "$(sha256sum $MODELDIR/$SIDECAR 2>/dev/null | cut -c1-64)" = $SIDECAR_SHA ] || { log "RESULT xac: VIRHE $MODELDIR/$SIDECAR missing or not sha256 $SIDECAR_SHA"; exit 1; }
  [ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -q "RADIANCE_DS41_DSPARK_CONF" /opt/radiance/share/radiance/architectures/deepseek41.so && grep -q dspark_conf_bf16 /opt/radiance/share/radiance/kernels/libr4d.so && grep -q "dspark_alt" /opt/radiance/bin/radiance && echo yes' 2>/dev/null)" = yes ] \
    || { log "RESULT xac: VIRHE $IMAGE lacks the DSpark extras"; exit 1; }
  for f in $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl; do [ -s $f ] || log "RESULT xac: note -- reference $f missing (not compared)"; done
fi
mkd $O $O/guard $STATSDIR
evict
stopfile || astart
unset $SWITCHES
down
log "RESULT xac $ITEM: done -- $O"
