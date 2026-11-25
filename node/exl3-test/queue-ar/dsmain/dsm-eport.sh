#!/bin/bash
# dsm-eport.sh (10.11.2026, queue-ar/dsmain, ds41 eport agent) -- NOT a queue item; the body of 57-p4zyzzzzz-ds-o-eport.sh.
# fn4's EXL3 expert-kernel levers on the DeepSeek-V4.1 line, IN ONE PROCESS a start, on the ds41-eport image (XEP_IMAGE =
# ds41-gg 8b311fcc + the eport layer: bin/radiance + deepseek41.so of ds41-eport, exl3r4d / exl3ref / exl3_quant of
# radiance-exl3 exl3-ds-eport; see the image's eport-PROVENANCE):
#   DZ  exl3_dz 0 / 1        (exl3r4d, fn4's EXL3R4D_DZ direct dead-row zeroing at run time: the spare workgroups of the
#                             decode-band and prefill-band grids store the same zero rows without the integer-modulo scan)
#   E9  exl3_mapshare 0 / 1  (deepseek41, fn4-x3 e22f6f0b: gate_up writes the routing map, down reads it instead of its
#                             own map kernel -- exl3r4d's PREFILL band only, passes of 17+ tokens)
#   B = both 1 (fn4 bundles them), A = both 0 (= today's code paths); gg_gemv 2 (lever D2) pinned on in every block of both
#   arms and at start (RADIANCE_DS41_GG_GEMV=2): today's best decode form. E10 (the fused gate_up_down) is not here: the
#   fused launch is K = 4 only and takes no SwiGLU limit; DS 2.25 bpw's experts are K 2 / 2.5 under limit 10.
# EXL3 2.25, TP4, MML 524288, the t3 env and serve line of dsm-draftmr.sh (105's t3 arm + lib-29 best_vars + MNBT 4160 +
# --live). TWO STARTS (XEP_STARTS, default "nst0 dspark"):
#   nst0    NST 0: c1 / c4 / c8 are 1 / 4 / 8-token passes, all in the decode band: B vs A reads DZ (E9 has no pass to act on)
#   dspark  --num-speculative-tokens 5 --draft-window adaptive (the DSpark head): verify passes of up to 6 / 24 / 48
#           tokens, so c4 / c8 verify in the prefill band: B vs A reads DZ + E9 together
# PER START (decode before prefill-heavy): gate (limit 23 ms nst0, 70 ms dspark) -> modes (the t3 notes; dspark the
#   DSpark lines) -> the switch note (GET at start: gg_gemv 2, exl3_dz 0, exl3_mapshare 0, epoch 0; exl3_dz 2 and
#   exl3_mapshare 2 refused; both 1, then both 0, applied at a step) -> warm-up answers vs stock -> BLOCKS
#   (ggblocks_eport.py, XEP_ORDER default A B B A; POST {exl3_dz, exl3_mapshare, gg_gemv 2, epoch_bump}; each block
#   stepprobe essay,code 512 (c1), [dspark: an essay-only stepprobe between two /stats snapshots, acceptance by draft
#   position], mb c4 and mb c8 (XEP_DUR 75 s, warm 10)) -> the start's summary line (B vs A pooled per metric, each
#   pair's B/A, t95) -> IDENTITY (prefill-heavy, last): B on the cold copy prompt (the 28k copy after a ~31k-token prompt:
#   E9 and DZ in the prefill band) + the 20-prompt greedy set at c1, then A the same; B vs A byte for byte, and each vs
#   bz's dsd-d0 (NST 0, main image; gg_gemv 2 and both levers are exact by construction).
# FIRST, P0 (~5 min, one card, no server; XEP_P0=0 skips): the image's perfmoe-eport at DS-V4.1's EP4 rank geometry --
#   the routed-expert chain's y A vs B byte for byte at M 4 / 8 / 16 (decode band) and 24 / 48 (prefill band, + map; a
#   difference stops the item), then DZ's dispatch us A B B A (down M 4 / 16, gate_up M 16) and the M 48 prefill chain.
# Wall time ~65 min: P0 ~5, nst0 ~27, dspark ~30 (XEP_STARTS=nst0: ~33).
# References (no re-runs, read beside the log): gz-gemvacq gg2-gemv2 (NST 0, gg_gemv 0 / 2 blocks), draftmr dm-dspark (NST
#   5 adaptive, gg_gemv 0 / 2). NO cc_sim (user 10.11.). No power / host / fan settings; production
#   compose files read only (--project-directory). Nothing goes into a production line without the user's go.
# DRYRUN=1 bash dsm-eport.sh : prints every command, runs and writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-gg-tp4.yaml
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=$R/xdsm/eport; export STATSDIR=$O/stats
RD=$R/xdsm/draft                                # bz's d0 files (NST 0, main image)
export IMAGE=${XEP_IMAGE:-radiance-local:ds41-eport-3fc3d3c0-04285c7c-l}
IMGID=${XEP_IMGID-22140b07b31f}
ITEM=${XEP_ITEM:-57-p4zyzzzzz-ds-o-eport}; ORDER=${XEP_ORDER:-ABBA}; DUR=${XEP_DUR:-75}
STARTS=${XEP_STARTS-"nst0 dspark"}
SW=exl3_dz,exl3_mapshare
GS=$HOME/gpu/ds41/exl3-test/queue/tools/greedyset.py
GGB=$HD/ggblocks_eport.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DS41_X3MAP"
t3env() {   # 105's t3 arm, every other switch unset; gg_gemv 2 (lever D2) at start
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
  export RADIANCE_DS41_GG_GEMV=2
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
cmpfile() {   # cmpfile A B -> IDENTICAL (n chars) | DIFFERENT at char k (na vs nb chars) | not compared
  python3 -c 'import sys
try: a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read()
except OSError as e: print("not compared (%s)" % e); sys.exit(0)
f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b)))
print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' "$1" "$2" 2>&1
}
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xep $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xep $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  case $r in IDENTICAL*) log "RESULT xep $1 warm-up answers vs stock ($(basename $2)): $r" ;;
    *) log "RESULT xep $1 warm-up answers vs stock ($(basename $2)): VIRHE $r -- NOT EXACT with both levers at 0" ;; esac
}
modes() {   # modes TAG: the serve log's notes must be the t3 line's -> 0, else 1 (start stopped)
  [ "$DRY" = 1 ] && { echo "+ modes $1: indexer note (mxfp4, latent bf16, compressor tail bound, router selection+sort separate, AR_PCN separate at 0 layers), ARGATHER note, staging line, no acquire-rule line, EXL3 expert lines"; return 0; }
  local MLF SL XL ok=1
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')
  SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)
  XL=$(docker logs $CN 2>&1 | grep -oE 'EXL3 experts gate_up [a-z0-9]+ / down [a-z0-9]+ \([a-z0-9]+\)' | sort | uniq -c | tr -s ' ' | tr '\n' '|')
  log "RESULT xep $1 modes: $(echo "$MLF" | cut -c1-900) | staging: ${SL:-none} | EXL3 layers: ${XL:-no line}"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "router selection+sort separate" "post-collapse-norm separate at 0 layers" "next collapse-norm separate at 0 layers"; do
    case "$MLF" in *"$w"*) ;; *) log "RESULT xep $1: VIRHE the indexer note lacks '$w'"; ok=0 ;; esac
  done
  docker logs $CN 2>&1 | grep -q 'the MoE gather and its all-reduce as one kernel' || { log "RESULT xep $1: VIRHE no RADIANCE_DS41_ARGATHER note"; ok=0; }
  [ -n "$SL" ] || { log "RESULT xep $1: VIRHE no prefill staging line (levels refused?)"; ok=0; }
  docker logs $CN 2>&1 | grep -q 'the acquire rule plays' && { log "RESULT xep $1: VIRHE an acquire-rule line (RADIANCE_AQL_ACQ should be unset)"; ok=0; }
  [ $ok = 1 ] || keeplog $1-nomode
  [ $ok = 1 ]
}
draftlines() {   # draftlines TAG: every drafter / speculation line of the serve log, logged
  [ "$DRY" = 1 ] && { echo "+ RESULT xep $1 drafter lines"; return 0; }
  log "RESULT xep $1 drafter / speculation lines: $(docker logs $CN 2>&1 | grep -iE 'dspark|drafter|speculat|draft-window|draft window|verify' | sed 's/^[A-Z][^ ]* //' | sort -u | head -8 | tr '\n' '|' | cut -c1-1500)"
}
fn4get() { [ "$DRY" = 1 ] && { echo '{"dry": 1}'; return 0; }; curl -s -m 10 $URL/admin/fn4 2>/dev/null; }
fn4set() { run "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '$1' $URL/admin/fn4 > /dev/null"; }
nudge() { run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"n_predict\": 4, \"temperature\": 0}' $URL/completion > /dev/null"; }
fn4code() {   # fn4code JSON: POST, print the HTTP status
  [ "$DRY" = 1 ] && { echo "+ POST /admin/fn4 $1" >&2; echo 400; return 0; }
  curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d "$1" $URL/admin/fn4 2>/dev/null
}
swnote() {   # swnote TAG: at start gg_gemv 2 / exl3_dz 0 / exl3_mapshare 0 / epoch 0; value 2 refused; both 1, then both 0
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4 (expect gg_gemv 2, exl3_dz 0, exl3_mapshare 0, epoch 0, plugins >= 3); POST exl3_dz 2 / exl3_mapshare 2 -> 400; POST both 1 / both 0, each + a 4-token request -> applied"; return 0; }
  local s0 s1 s2 cz cm
  s0=$(fn4get); cz=$(fn4code '{"exl3_dz": 2}'); cm=$(fn4code '{"exl3_mapshare": 2}')
  fn4set '{"exl3_dz": 1, "exl3_mapshare": 1}'; nudge; s1=$(fn4get)
  fn4set '{"exl3_dz": 0, "exl3_mapshare": 0}'; nudge; s2=$(fn4get)
  log "RESULT xep $1 switch note: at start $s0 | exl3_dz 2 -> HTTP $cz, exl3_mapshare 2 -> HTTP $cm | after both 1: $s1 | after both 0: $s2 | core log: $(docker logs $CN 2>&1 | grep -E 'fn4: switches|fn4: .* refused' | tail -3 | sed 's/^[A-Z] *//' | tr '\n' '|' | cut -c1-500)"
  python3 -c 'import json,sys
s0, s1, s2 = (json.loads(x) for x in sys.argv[1:4])
w = lambda s, k: s["switches"].get(k)
ok = w(s0, "gg_gemv") == 2 and w(s0, "exl3_dz") == 0 and w(s0, "exl3_mapshare") == 0 and s0["epoch"] == 0 \
     and w(s1, "exl3_dz") == 1 and w(s1, "exl3_mapshare") == 1 and w(s1, "gg_gemv") == 2 and s1["plugins"] >= 3 \
     and w(s2, "exl3_dz") == 0 and w(s2, "exl3_mapshare") == 0 and not s1["pending"] and not s2["pending"] and s2["epoch"] == 2
sys.exit(0 if ok else 1)' "$s0" "$s1" "$s2" 2>/dev/null && [ "$cz" = 400 ] && [ "$cm" = 400 ]
}
blocks() {   # blocks TAG NOPOS: the A/B blocks (A = exl3_dz 0 + exl3_mapshare 0, B = both 1; gg_gemv 2 pinned)
  run "timeout 7200 python3 -I $GGB --url $URL --mb-dir $D --edit /dev/null --no-edit $2 --label $1 --switch $SW --a 0 --b 1 --fixed '{\"gg_gemv\": 2}' --order $ORDER --dur $DUR --warm 10 --out $O/$1-blocks.jsonl > $O/$1-blocks.txt 2>&1"
  [ "$DRY" = 1 ] && return 0
  while IFS= read -r l; do log "RESULT xep $1 ${l:0:900}"; done < <(grep '^BLOCK ' $O/$1-blocks.txt)
}
summary() {   # summary TAG ARGS: B vs A pooled, each pair, t95
  [ "$DRY" = 1 ] && { echo "+ RESULT xep $1 summary: B (exl3_dz 1 + exl3_mapshare 1) vs A (both 0) pooled + pairs + t95"; return 0; }
  log "RESULT xep $1 summary ($2; B = exl3_dz 1 + exl3_mapshare 1 vs A = both 0, gg_gemv 2 pinned, in process, order $ORDER, mb $DUR s): $(grep '^GGB ' $O/$1-blocks.txt | cut -c5- | python3 -c 'import json,sys
d = json.loads(sys.stdin.read())
f = lambda v, n=2: "-" if v is None else format(v, ".%df" % n)
out = []
for k, v in d["metrics"].items():
    if v.get("A") is None and v.get("B") is None: continue
    p = v.get("pooled")
    out.append("%s %s -> %s (%s %%; pairs %s; t95 +- %s %%)" % (k, f(v.get("A")), f(v.get("B")), "-" if p is None else "%+.2f" % (100 * (p - 1)),
               " ".join("%+.2f" % (100 * (x - 1)) for x in v.get("pairs", [])), "-" if v.get("ci95") is None else "%.2f" % (100 * v["ci95"])))
print("; ".join(out) + " | %d blocks, %d flagged, %s s" % (d["blocks"], d["blocks_bad"], d["wall_s"]))' 2>&1 | cut -c1-3000)"
}
copyt() {   # copyt TAG: the 28k copy after a ~31k-token prompt (105's prompt, offset 0)
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset 0 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xep $1 copy: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300)"
}
gsetc1() { run "(cd $D && timeout 900 python3 $GS $URL $O/$1-gs-c1.jsonl --conc 1 --max-tokens 192 > $O/$1-gs-c1.txt 2>&1)"; }
gcmp() { [ "$DRY" = 1 ] && { echo "IDENTICAL (dry)"; return 0; }; [ -s "$1" ] && [ -s "$2" ] || { echo "not compared (missing $1 or $2)"; return 0; }; python3 $GS --cmp $1 $2 2>&1 | cut -c1-300; }
idline() {   # idline LABEL RESULT STRICT: IDENTICAL -> plain; else VIRHE (STRICT 1: an exact lever changed bytes) or reported
  case "$2" in IDENTICAL*|"not compared"*) log "RESULT xep identity $1: $2" ;;
    *) if [ "${3:-1}" = 1 ]; then log "RESULT xep identity $1: VIRHE $2 -- both levers are exact by construction, so this is a defect"
       else log "RESULT xep identity $1: DIFFERENT $2 -- reported, not hidden (a drafting start vs an NST 0 reference)"; fi ;;
  esac
}
estart() {   # estart nst0|dspark: one start, its blocks, identity B vs A, its summary line
  local s=$1 tag=ep-$1 P=$O/ep-$1 rc lim nopos strict
  t3env; export MODEL=$M225; HP29=19456; best_vars
  case $s in
    nst0)   export MNBT=4160 NST=0 XARGS="$XARGS --live"; lim=23; nopos=--no-pos; strict=1 ;;
    dspark) export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive"; lim=70; nopos=""; strict=0 ;;
    *) log "RESULT xep: VIRHE unknown start $s"; return 1 ;;
  esac
  log "RESULT xep $tag: start IMAGE=$IMAGE MODEL=$MODEL $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
  mkd $P
  gate $tag $lim; rc=$?
  [ $rc = 1 ] && { log "RESULT xep $tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  [ $rc = 2 ] && { log "RESULT xep $tag: VIRHE over the gate limit after one restart -> start skipped"; down; return 1; }
  modes $tag || { log "RESULT xep $tag: VIRHE the modes are not the t3 line's -> start stopped"; down; return 1; }
  if [ $s = dspark ]; then
    draftlines $tag
    [ "$DRY" = 1 ] || docker logs $CN 2>&1 | grep -q "DSpark drafter" || { log "RESULT xep $tag: VIRHE no DSpark drafter line"; keeplog $tag-nodraft; down; return 1; }
  fi
  swnote $tag || { log "RESULT xep $tag: VIRHE the switches are not as expected (start gg_gemv 2 / exl3_dz 0 / exl3_mapshare 0 / epoch 0, value 2 refused, both 1 / both 0 applied at a step, >= 3 hook objects) -> start stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  follow $P
  stopfile || blocks $tag "$nopos"
  if ! stopfile; then   # identity, prefill-heavy, last: B on the cold copy prompt, then A (prompt cached)
    fn4set '{"exl3_dz": 1, "exl3_mapshare": 1, "gg_gemv": 2, "epoch_bump": 1}'
    copyt $tag-B; gsetc1 $tag-B
    [ "$DRY" = 1 ] || log "RESULT xep $tag identity B ran under: $(fn4get | cut -c1-260)"
    fn4set '{"exl3_dz": 0, "exl3_mapshare": 0, "gg_gemv": 2, "epoch_bump": 1}'
    copyt $tag-A; gsetc1 $tag-A
    [ "$DRY" = 1 ] || log "RESULT xep $tag identity A ran under: $(fn4get | cut -c1-260)"
    if [ "$DRY" = 1 ]; then echo "+ identity: 28k copy B (cold) vs A, greedy set c1 B vs A; A and B vs bz d0"
    else
      idline "$tag 28k copy B (both 1, cold prompt) vs A (both 0, prompt cached)" "$(cmpfile $O/copy-$tag-B-run1.txt $O/copy-$tag-A-run1.txt)" 1
      idline "$tag greedy set c1 B vs A" "$(gcmp $O/$tag-A-gs-c1.jsonl $O/$tag-B-gs-c1.jsonl)" 1
      idline "$tag 28k copy B vs bz d0 (NST 0, main image)" "$(cmpfile $O/copy-$tag-B-run1.txt $RD/copy-dsd-d0-i1-run1.txt)" $strict
      idline "$tag greedy set c1 A vs bz d0" "$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$tag-A-gs-c1.jsonl)" $strict
      idline "$tag greedy set c1 B vs bz d0" "$(gcmp $RD/dsd-d0-gs-c1.jsonl $O/$tag-B-gs-c1.jsonl)" $strict
    fi
  fi
  unfollow
  run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
  [ "$DRY" = 1 ] || log "RESULT xep $tag engine: $(grep -cE 'fn4: switches .* applied' $O/$tag-engine-full.log) applied switch changes, $(grep -cE 'fn4: .* refused' $O/$tag-engine-full.log) refused"
  down
  summary $tag "$s: $(desc | grep -oE "NST=[^ ]*") $(echo $XARGS | grep -oE -- '--draft-window [a-z]+' | tr '\n' ' ')"
  log "RESULT xep $tag: done"
}
# ---------------------------------------------------------------- P0: one card, no server (~5 min, XEP_P0=0 skips)
# The image's perfmoe-eport (exl3-main37 6f02e2d's perfmoe, the same build as its exl3r4d.so) at DeepSeek-V4.1's EP4 rank
# geometry (96 experts a rank of 384, top 6, hidden 5120, n_ff 2304, K 2 = 73 % of the 2.25's experts; --ep-rank -1 the
# rank with the most live experts). IDENTITY first: the routed-expert chain's y (layer 0, --split-seed, every slot incl.
# the dead rows the spares zero) A (exl3_dz 0, no map) vs B (exl3_dz 1, + --x3map in the prefill band), byte for byte, at
# M 4 / 8 / 16 (decode band: DZ) and M 24 / 48 (prefill band, the dspark start's c4 / c8 verify widths: DZ + E9). A
# difference stops the item (both levers are exact by construction: a difference is a defect). Then TIMING A B B A:
# down M 4 / 16 and gate_up M 16 (decode band, exl3_dz 0 / 1) and the prefill-band chain at M 48 (A vs B).
PM=/opt/radiance/bin/perfmoe-eport; X3=/opt/radiance/share/radiance/kernels/exl3r4d.so
GEO_DS="--tp 1 --ep 4 --ep-rank -1 --h-rot 1 --bits 2 --experts 96 --topk 6 --hidden 5120 --width 2304 --pooled 0 --ceiling 640 --warm-ms 500"
P0TIM="--iters 20 --reps 5"
P0IDC="--chain --layers 1 --pools 1 --split-seed --rand-data --iters 2 --reps 1"
pm() {   # pm NAME perfmoe-args...
  local nm=$1; shift
  local cn=xep-p0-$$-$nm
  if [ "$DRY" = 1 ]; then echo "+ docker run (card HIP 0) --entrypoint $PM $IMAGE --exl3 $X3 $* > $O/p0/$nm.txt"; return 0; fi
  timeout -k 15 300 docker run --rm --init --stop-timeout 10 --name $cn --network none --device /dev/kfd --device /dev/dri \
    --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0 --user $(id -u):$(id -g) --group-add $(getent group render | cut -d: -f3) \
    --group-add $(getent group video | cut -d: -f3) -e HOME=/tmp -w /tmp -v $O/p0:/o --entrypoint $PM $IMAGE --exl3 $X3 "$@" > $O/p0/$nm.txt 2>&1
  local rc=$?; docker rm -f $cn > /dev/null 2>&1; return $rc
}
usl() { local v; v=$(grep -m1 "^RESULT exl3r4d" $O/p0/$1.txt 2>/dev/null | sed -nE 's/.*M=[0-9]+ +([0-9.]+) us.*/\1/p'); echo ${v:-NA}; }
avg2() { python3 -c "import sys
a=[float(x) for x in sys.argv[1:] if x!='NA']
print('%.2f' % (sum(a)/len(a)) if len(a)==len(sys.argv[1:]) else 'NA')" "$@"; }
pct() { python3 -c "import sys
a,b=sys.argv[1],sys.argv[2]
print('NA' if 'NA' in (a,b) else '%+.1f %%' % (100*(float(b)/float(a)-1)))" "$1" "$2"; }
p0() {   # -> 0, or 1 when a y differs (the item stops)
  local M band xb bad=0 line a b
  run "mkdir -p $O/p0; chmod 777 $O/p0"
  for M in 4 8 16 24 48; do
    if [ $M -le 16 ]; then band=_m16; xb=""; else band=_pf; xb=--x3map; fi
    pm id$M-a $GEO_DS $P0IDC --chain-band $band --M $M --fn4 exl3_dz=0 --dump-y /o/id$M-a.y
    pm id$M-b $GEO_DS $P0IDC --chain-band $band --M $M --fn4 exl3_dz=1 $xb --dump-y /o/id$M-b.y
    [ "$DRY" = 1 ] && continue
    if [ -s $O/p0/id$M-a.y ] && [ -s $O/p0/id$M-b.y ]; then
      if cmp -s $O/p0/id$M-a.y $O/p0/id$M-b.y; then log "RESULT xep P0 identity M $M ($band${xb:+ + map}): chain y A vs B IDENTICAL ($(stat -c %s $O/p0/id$M-a.y) bytes)"
      else bad=1; log "RESULT xep P0 identity M $M ($band${xb:+ + map}): VIRHE chain y A vs B DIFFERENT ($(cmp -l $O/p0/id$M-a.y $O/p0/id$M-b.y | wc -l) bytes differ)"; fi
    else log "RESULT xep P0 identity M $M ($band): NA (no dump: A $(tail -1 $O/p0/id$M-a.txt | cut -c1-120) | B $(tail -1 $O/p0/id$M-b.txt | cut -c1-120))"; fi
  done
  [ $bad = 1 ] && return 1
  for t in "down 4" "down 16" "gate_up 16"; do
    set -- $t
    pm $1$2-a1 $GEO_DS $P0TIM --op $1 --M $2 --fn4 exl3_dz=0
    pm $1$2-b1 $GEO_DS $P0TIM --op $1 --M $2 --fn4 exl3_dz=1
    pm $1$2-b2 $GEO_DS $P0TIM --op $1 --M $2 --fn4 exl3_dz=1
    pm $1$2-a2 $GEO_DS $P0TIM --op $1 --M $2 --fn4 exl3_dz=0
    [ "$DRY" = 1 ] && continue
    a=$(avg2 $(usl $1$2-a1) $(usl $1$2-a2)); b=$(avg2 $(usl $1$2-b1) $(usl $1$2-b2))
    log "RESULT xep P0 DZ $1 M $2 (decode band, one card): exl3_dz 0 $a us ($(usl $1$2-a1) / $(usl $1$2-a2)) -> 1 $b us ($(usl $1$2-b1) / $(usl $1$2-b2)) $(pct $a $b)"
  done
  pm ch48-a1 $GEO_DS $P0TIM --chain --chain-band _pf --M 48 --fn4 exl3_dz=0
  pm ch48-b1 $GEO_DS $P0TIM --chain --chain-band _pf --M 48 --fn4 exl3_dz=1 --x3map
  pm ch48-b2 $GEO_DS $P0TIM --chain --chain-band _pf --M 48 --fn4 exl3_dz=1 --x3map
  pm ch48-a2 $GEO_DS $P0TIM --chain --chain-band _pf --M 48 --fn4 exl3_dz=0
  if [ "$DRY" != 1 ]; then
    a=$(avg2 $(usl ch48-a1) $(usl ch48-a2)); b=$(avg2 $(usl ch48-b1) $(usl ch48-b2))
    log "RESULT xep P0 DZ + E9 chain M 48 (prefill band, one card): A $a us ($(usl ch48-a1) / $(usl ch48-a2)) -> B $b us ($(usl ch48-b1) / $(usl ch48-b2)) $(pct $a $b)"
  fi
  return 0
}
# ---------------------------------------------------------------- main
log "RESULT xep $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) starts [$STARTS] order $ORDER mb $DUR s | B = exl3_dz 1 + exl3_mapshare 1, A = both 0, gg_gemv 2 in both | eport: $(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -m3 -E "^(radiance|exl3) " /opt/radiance/share/radiance/eport-PROVENANCE.txt' 2>/dev/null | tr '\n' '|' | cut -c1-400)"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xep: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xep: VIRHE image $IMAGE not on gpu02"; exit 1; }
  [ -z "$IMGID" ] || [ "$(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)" = "$IMGID" ] || { log "RESULT xep: VIRHE image $IMAGE is not $IMGID"; exit 1; }
  [ -e $Y ] && grep -q '^ *RADIANCE_DS41_GG_GEMV:' $Y || { log "RESULT xep: VIRHE $Y missing or does not pass RADIANCE_DS41_GG_GEMV"; exit 1; }
  docker ps --format '{{.Names}}' | grep -qiE '^(radiance|ds41|fn)' && [ "${XEP_P0:-1}" = 1 ] && log "RESULT xep: note -- containers running at P0 ($(docker ps --format '{{.Names}}' | tr '\n' ' ')); P0 is one card, HIP device 0"
  for f in $GS $GGB $D/stepprobe.py $D/mb.py $D/ds41_tests.py; do [ -e $f ] || { log "RESULT xep: VIRHE $f missing"; exit 1; }; done
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xep: VIRHE $M225 missing or not verified"; exit 1; }
  [ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -q exl3_mapshare /opt/radiance/bin/radiance && grep -q exl3_dz /opt/radiance/share/radiance/kernels/exl3r4d.so && grep -q moe_exl3_map /opt/radiance/share/radiance/architectures/deepseek41.so && grep -q r4d_gg_mr_pin /opt/radiance/share/radiance/kernels/libr4d.so && echo yes' 2>/dev/null)" = yes ] \
    || { log "RESULT xep: VIRHE $IMAGE is not the eport image (no exl3_mapshare in radiance, exl3_dz in exl3r4d, moe_exl3_map in deepseek41 or D2 in libr4d)"; exit 1; }
  for f in $RD/copy-dsd-d0-i1-run1.txt $RD/dsd-d0-gs-c1.jsonl; do [ -s $f ] || log "RESULT xep: note -- reference $f missing (not compared)"; done
fi
mkd $O $O/guard $STATSDIR
if [ "${XEP_P0:-1}" = 1 ]; then
  p0 || { log "RESULT xep $ITEM: VIRHE P0's chain y differs between A and B -- no server phases"; exit 1; }
fi
evict
done_starts=""
for s in $STARTS; do stopfile && break; estart $s && done_starts="$done_starts $s"; done
unset $SWITCHES
down
log "RESULT xep $ITEM: done -- starts [$done_starts ] $O"
