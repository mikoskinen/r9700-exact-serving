#!/bin/bash
# dsm-resume2.sh (10.11.2026, queue-ar/dsmain, ds41 resume agent) -- NOT a queue item; the body of 57-p4zyzzzzz-ds-hcz-resume2.sh.
# QUESTION (finding R1): why did hcz's resumes of the chain written by d1 (the first long prefill after start + warm-up) differ
#   from cold (token-0 logprob diff 1.5e-4 / 6.2e-4), while every chain written after the /reset_prefix_cache resumed exactly?
# THREE CANDIDATES (all unverified):
#   H1 first sighting: d1's prefill passes are the first sighting of their pass keys (core/runtime/ctx.cpp run_pass, issued live,
#      not recorded); something in that path orders the snapshot copy / KV writes / copy-out differently from the compute.
#      Code reading found no such ordering: the snapshot is a blit on the step's own stream after the pass
#      (core/engine.cpp save_checkpoints -> core/mem/kv.cpp checkpoint_copy), the copy-out waits on an event of the compute
#      stream (core/runtime/tierexec.cpp enqueue_copy). Predicts: the FIRST long prefill of a start resumes wrong, whatever
#      came before it; the second is exact.
#   H2 pool pressure: d1 alone dropped 216 VRAM entries mid-prefill (warm-up chains filled the pool); a page or slot reused
#      before its copy landed. Predicts: wrong only where the cold run dropped entries (dropped > 5).
#   H3 shared head entry (code reading, core/mem/prefix.cpp insert): every chat prompt starts with the same 2-token entry
#      (BOS, <User>). With snapshots (has_linear_) an existing entry keeps the bytes it was FIRST published with; the cache
#      never takes a later publisher's copy. hcz's first publisher of entry 0 was the warm-up's 27-token question (a one-chunk
#      prefill of 27 rows); d1 recomputed entry 0 inside its 2048-row first chunk and wrote its 33 snapshots from ITS bytes, but
#      the cache kept the 27-row bytes. A hit restores the 27-row bytes beside d1's snapshot: not the state d1 used. After the
#      reset, entry 0 was first published by s2's own 2048-row chunk, which every later prompt (all > 2048 tokens) computes
#      identically -> exact. Predicts: wrong exactly where a SHORT request published the head before the cold run, first long
#      prefill or not; and a hit of such a deep chain is byte-identical to hcz's d2 (md5 bfbea1e2: same 27-row bytes, same d1).
# THE ITEM: two starts of hcz's image + serve line (XHR_IMAGE, gg_gemv 0 = today's arithmetic, t3 env, MNBT 4160, --live),
#   sequential requests on an idle server, logprobs on (resume_ho.py, unchanged), the /sessions tier counters per request:
#   S1 as hcz (warm-up + fastcheck = the 27/17/43-token questions, a 16k prefill, a 50-token fastcheck), and
#      RADIANCE_DEBUG_TIER_VERIFY=1 (host-side hashes of rank 0's host slots at landing and before each restore; an E line
#      "came back different" if the tier moved wrong bytes; no device work, no arithmetic change):
#        a1 nonce-A 136k cold = the FIRST long prefill of the start; a2 its hit (tail 133120)   H1 / H2 / H3: all differ
#        b1 deep cold (2nd long prefill; deepq's prompt, hcz d1);    b2 its hit                H1: same  H2: same if b1
#                                                                                              dropped <= 5  H3: DIFFERS and
#                                                                                              b2 == hcz d2 byte for byte
#   S2 no warm-up, no fastcheck (TIER_VERIFY off):
#        f1 deep cold = the first request of the start;   f2 its hit                           H1: differs  H2 / H3: same
#        /reset_prefix_cache; the warm-up's three short questions alone (ds41_tests correct --only-q: 27/17/43 tokens);
#        f3 deep cold (2nd long prefill);                 f4 its hit                           H1 / H2: same  H3: DIFFERS
#                                                                                              and f4 == hcz d2
#   S3 only when XHR_FIX_IMAGE is set (the H3 fix image): as S1 without TIER_VERIFY:
#        g1 nonce-G 136k cold (first long prefill after the warm-up); g2 its hit             fix: same
#        h1 deep cold (== hcz d1 fc895faf: cold unchanged by the fix); h2 its hit             fix: same
#   Then every pair byte-compared (resume_ho.py cmp: text, first differing token, token-0 logprob diff) and one summary.
# ~16 min (S1 ~7, S2 ~6.5, slack), ~23 min with S3. Writes runs/xdsm/resume2/. Log tag "RESULT xhr ". NO cc_sim. No power / host /
# fan settings; production compose files read only (--project-directory). Nothing goes into a production line from here.
# DRYRUN=1 bash dsm-resume2.sh : prints every command, runs and writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
Y=$HD/ds41-xhr-tp4.yaml                         # = ds41-gg-tp4.yaml + RADIANCE_DEBUG_TIER_VERIFY passed through when set
O=$R/xdsm/resume2; export STATSDIR=$O/stats
RH=$R/xdsm/resume                               # hcz: d1.txt (cold, md5 fc895faf), d2.txt (the wrong hit, md5 bfbea1e2)
RM=$R/xdsm                                      # item a's dsm-225 cold: deep-dsm-225.txt
IMG0=${XHR_IMAGE:-radiance-local:ds41-gg-8b311fcc-5c13e555-l}
FIXIMG=${XHR_FIX_IMAGE:-}
export IMAGE=$IMG0
ITEM=${XHR_ITEM:-57-p4zyzzzzz-ds-hcz-resume2}
DRV=$HD/resume_ho.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
NONCE=${XHR_NONCE:-xhr$(date +%s)}
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DS41_ARGATHER_MAX RADIANCE_DS41_SHEXP_LANE RADIANCE_DEBUG_MISS_LOG
  RADIANCE_DS41_KVFOLD RADIANCE_DS41_OFOLD RADIANCE_DS41_ONEFILL RADIANCE_DS41_PIPE RADIANCE_DEBUG_TIER_VERIFY"
# ---------------------------------------------------------------- helpers (dsm-resume.sh's, copied; log tag xhr)
t3env() {   # 105's t3 arm, every other switch unset (RADIANCE_DS41_GG_GEMV unset = gg_gemv 0, the stock rule)
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
trap 'down' EXIT
trap 'exit 143' TERM INT
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xhr $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xhr $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $3 $2 2>&1 | cut -c1-300)
  log "RESULT xhr $1 short answers vs stock ($(basename $2)): $r"
}
modes() {   # modes TAG [fix]: the t3 indexer note and the exact-resume / snapshot lines -> 0, else 1 (stopped: the premise is gone)
  [ "$DRY" = 1 ] && { echo "+ modes $1: indexer note (mxfp4, latent bf16, compressor tail bound, keep from 32768, split from 4096) + 'prefix resume: exact' + 'checkpoints every 2048' + keep-all + 'a checkpoint slot carries its window' + /admin/fn4 gg_gemv 0 ${2:+ + the fix line 'takes a later publisher'}"; return 0; }
  local MLF PL SW ok=0
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //' | cut -c1-700)
  PL=$(grep -E 'prefix resume:|^I prefix cache: |carries its window|^I chunk geometry|^I checkpoint slots|later publisher' $O/start-$1.log 2>/dev/null | sed 's/^[A-Z] *//' | cut -c1-260 | tr '\n' '|')
  SW=$(curl -s -m 10 $URL/admin/fn4 2>/dev/null | cut -c1-400)
  log "RESULT xhr $1 modes: $MLF | prefix: $PL | switches: $SW | TIER_VERIFY in the container: $(docker exec $CN printenv RADIANCE_DEBUG_TIER_VERIFY 2>/dev/null || echo unset)"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "candidate restriction on from 32768" "row split on from 4096"; do
    echo "$MLF" | grep -q "$w" || { log "RESULT xhr $1: VIRHE the indexer note lacks '$w'"; ok=1; }; done
  for w in "prefix resume: exact" "checkpoints every 2048 tokens" "retention 'keep-all'" "a checkpoint slot carries its window"; do
    echo "$PL" | grep -q "$w" || { log "RESULT xhr $1: VIRHE the start log lacks '$w'"; ok=1; }; done
  if [ -n "${2:-}" ]; then echo "$PL" | grep -q "later publisher" || { log "RESULT xhr $1: VIRHE the fix image's start line ('takes a later publisher's bytes') is missing"; ok=1; }; fi
  echo "$SW" | grep -q '"gg_gemv": 0' || { log "RESULT xhr $1: VIRHE gg_gemv is not 0 at /admin/fn4 ($SW)"; ok=1; }
  return $ok
}
# expected cached tokens of a hit: the exact grid's tail point (scheduler.cpp tail_checkpoint: grid_floor(n - margin), margin 1,
# or 16 where only a window is restored; grid 2048 + 4096 k) -- both printed when they differ
exptail() { python3 -c 'import sys
n=int(sys.argv[1]); f=lambda p: 0 if p < 2048 else 2048 + (p - 2048) // 4096 * 4096
a, b = f(n - 1), f(n - 16); print(a if a == b else "%d|%d" % (a, b))' "$1" 2>/dev/null || echo "?"; }
rho() {   # rho TAG KIND EXPECT [driver args]: one request -> its RESULT line (EXPECT: 0 cold, a number, or tail)
  local tag=$1 kind=$2 ex=$3; shift 3
  run "(cd $HD && timeout 1800 python3 -I $DRV run $O/$tag.txt $kind $* > $O/$tag.out 2>&1)"
  [ "$DRY" = 1 ] && return 0
  local j; j=$(grep -m1 '^RHO ' $O/$tag.out | cut -c5-)
  [ -n "$j" ] || { log "RESULT xhr $tag ($kind): VIRHE no answer ($(tail -2 $O/$tag.out | tr '\n' ' ' | cut -c1-300))"; return 1; }
  [ "$ex" = tail ] && ex=$(exptail $(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("prompt_tokens") or 0)' "$j"))
  log "RESULT xhr $tag ($kind): $(python3 -c 'import json,sys
d=json.loads(sys.argv[1]); ex=sys.argv[2]; c=d.get("cached")
ok = str(c) in ex.split("|")
b, a = d.get("tiers_before") or {}, d.get("tiers_after") or {}
dl = lambda k: (a.get(k) - b.get(k)) if isinstance(a.get(k), (int, float)) and isinstance(b.get(k), (int, float)) else "?"
print("status %s, prompt %s tok, cached %s (expected %s: %s), TTFT %s s, %s tok, %s chars md5 %s, logprobs %s (%s tokens) | tier: promoted +%s, restores +%s (+%s ms), to_host +%s, dropped +%s, snapshots written +%s%s" % (
  d.get("status"), d.get("prompt_tokens"), c, ex, "as expected" if ok else "VIRHE not as expected -- the pair below is not the planned comparison",
  d.get("ttft"), d.get("tokens"), d.get("chars"), d.get("md5"), d.get("lp"), d.get("lp_tokens"),
  dl("promoted"), dl("restores"), round(dl("restore_ms"), 1) if dl("restore_ms") != "?" else "?", dl("to_host"), dl("dropped"), dl("ckpt_written"),
  "" if not d.get("err") else " | err " + str(d.get("err"))[:200]))' "$j" "$ex" 2>&1 | cut -c1-700)"
}
pair() {   # pair LABEL A B: resume_ho.py cmp -> RESULT line; appends "LABEL identical|different" to $O/pairs.txt
  [ "$DRY" = 1 ] && { echo "+ RESULT xhr pair $1: resume_ho.py cmp $2 $3"; return 0; }
  local c; c=$(python3 -I $DRV cmp $2 $3 2>&1 | grep -m1 '^CMP ' | cut -c5-)
  log "RESULT xhr pair $1: $(python3 -c 'import json,sys
d=json.loads(sys.argv[1]); s=d.get("text")
if "first_token_diff" in d:
    s += " | first differing token %s, logprob |diff| token 0 %s, first logprob diff at token %s, max before %s, equal-logprob tokens before %s, top-2 margin there %s" % (
        d.get("first_token_diff"), d.get("lp_diff_tok0"), d.get("first_lp_diff_at"), d.get("lp_diff_max_before"), d.get("lp_equal_tokens_before"), d.get("margin_at_diff"))
elif d.get("lp"): s += " | " + d["lp"]
print(s)' "$c" 2>&1 | cut -c1-600)"
  case "$c" in *'"identical": true'*) echo "$1 identical" >> $O/pairs.txt ;; *'"identical": false'*) echo "$1 different" >> $O/pairs.txt ;; *) echo "$1 n/a" >> $O/pairs.txt ;; esac
}
pcreset() {   # pcreset TAG: POST /reset_prefix_cache (loopback operator) -> RESULT line
  run "(cd $HD && timeout 120 python3 -I $DRV reset > $O/$1.out 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xhr $1: $(grep -m1 '^RHO ' $O/$1.out | cut -c5- | cut -c1-300 || tail -1 $O/$1.out)"
}
shortq() {   # shortq TAG: the warm-up's three short questions alone (27/17/43 tokens), as check() sends them first
  run "(cd $D && timeout 300 python3 ds41_tests.py correct --only-q --effort none --outdir $O/poison > $O/$1-q.txt 2>&1)"
  [ "$DRY" = 1 ] || identw $1 $WARMREF $O/$1-q.txt
}
tierlog() {   # tierlog TAG: E lines of the round-trip check (RADIANCE_DEBUG_TIER_VERIFY) and of the engine
  [ "$DRY" = 1 ] && { echo "+ RESULT xhr $1 engine log: E lines, 'came back different', 'differ between rank 0', republish lines"; return 0; }
  local f=$O/engine-$1.log
  log "RESULT xhr $1 engine log: $(grep -cE '^E ' $f 2>/dev/null) E lines; tier round trip 'came back different': $(grep -c 'came back different' $f 2>/dev/null); replicated mismatch: $(grep -c 'differ between rank 0' $f 2>/dev/null); fix republish lines: $(grep -c 'later publisher' $f 2>/dev/null); resets: $(grep -c 'prefix cache: reset' $f 2>/dev/null); $(grep -E '^E ' $f 2>/dev/null | head -2 | tr '\n' ' ' | cut -c1-300)"
}
start() {   # start TAG IMAGE WARM(1|0) [fix]: restart (+ warm-up + fastcheck when WARM) + modes; 1 = stop the item
  local tag=$1 w=$3; export IMAGE=$2
  t3env; modelvars225
  [ "$tag" = s1 ] && export RADIANCE_DEBUG_TIER_VERIFY=1
  log "RESULT xhr $tag: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) warm-up $w MODEL=$MODEL NST=$NST $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
  restart $tag || { keeplog $tag-fail; log "RESULT xhr $tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; return 1; }
  if [ "$w" = 1 ]; then
    local ms; ms=$(check $tag 1)
    [ "$DRY" = 1 ] || log "RESULT guard xhr $tag: 1-user ${ms} ms/step (information only: exactness does not depend on the slow state)"
  fi
  modes $tag ${4:-} || { keeplog $tag-modes; log "RESULT xhr $tag: VIRHE the modes are not as asked -> stopped"; return 1; }
  [ "$w" = 1 ] && identw $tag $WARMREF $O/guard/$tag-warm-q.txt
  return 0
}
# ---------------------------------------------------------------- main
log "RESULT xhr $ITEM: start IMAGE=$IMG0 ($(docker image inspect -f '{{.Id}}' $IMG0 2>/dev/null | cut -c8-19)) fix image ${FIXIMG:-none}; S1 warm-up + nonce 136k first + deep, S2 no warm-up: deep first, reset, 3 short questions, deep${FIXIMG:+, S3 fix image: nonce + deep}; cold vs resume byte for byte, nonce $NONCE"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xhr: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  for im in $IMG0 $FIXIMG; do
    docker image inspect $im > /dev/null 2>&1 || { log "RESULT xhr: VIRHE image $im not on gpu02"; exit 1; }
    [ "$(timeout 120 docker run --rm --network none --entrypoint sh $im -c 'grep -q "prefix resume: exact" /opt/radiance/bin/radiance && grep -q reset_prefix_cache /opt/radiance/bin/radiance && echo yes' 2>/dev/null)" = yes ] \
      || { log "RESULT xhr: VIRHE $im's engine has no exact prefix resume or no /reset_prefix_cache"; exit 1; }
  done
  [ -e $Y ] && [ -e $DRV ] && [ -e $HOME/gpu/ds41/ds41_tests.py ] || { log "RESULT xhr: VIRHE $Y, $DRV or ds41_tests.py missing"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xhr: VIRHE $M225 missing or not verified"; exit 1; }
  for f in $RH/d1.txt $RH/d2.txt $RM/deep-dsm-225.txt; do [ -s $f ] || log "RESULT xhr: note -- reference $f missing (that pair reads 'not compared')"; done
fi
mkd $O $O/guard $O/poison $STATSDIR
run "rm -f $O/pairs.txt"
down
evict
# -------- S1: as hcz (warm-up + fastcheck), TIER_VERIFY on; the first long prefill is a fresh nonce chain, then deep
if start s1 $IMG0 1; then
  rho a1 nonce 0 --nonce ${NONCE}a
  rho a2 nonce tail --nonce ${NONCE}a
  stopfile && exit 0
  rho b1 deep 0
  rho b2 deep tail
  keeplog s1; tierlog s1
fi
down
stopfile && exit 0
# -------- S2: no warm-up; deep first; reset; the three short questions; deep cold again and its hit
if start s2 $IMG0 0; then
  rho f1 deep 0
  rho f2 deep tail
  pcreset r1
  shortq p1
  rho f3 deep 0
  rho f4 deep tail
  keeplog s2; tierlog s2
fi
down
# -------- S3 (fix image only): as S1 without TIER_VERIFY; first long prefill a nonce chain, then deep
if [ -n "$FIXIMG" ] && ! stopfile; then
  if start s3 $FIXIMG 1 fix; then
    rho g1 nonce 0 --nonce ${NONCE}g
    rho g2 nonce tail --nonce ${NONCE}g
    rho h1 deep 0
    rho h2 deep tail
    keeplog s3; tierlog s3
  fi
  down
fi
# -------- the comparisons (labels: the first word is the arm; "hit vs cold" pairs carry the question)
pair "a2 hit vs a1 cold (S1 nonce, first long prefill after warm-up)" $O/a2.txt $O/a1.txt
pair "b2 hit vs b1 cold (S1 deep, second long prefill) [H1/H2 same, H3 differs]" $O/b2.txt $O/b1.txt
pair "f2 hit vs f1 cold (S2 deep, first request of the start) [H1 differs, H2/H3 same]" $O/f2.txt $O/f1.txt
pair "f4 hit vs f3 cold (S2 deep after reset + 3 short questions) [H1/H2 same, H3 differs]" $O/f4.txt $O/f3.txt
pair "ref b2 vs hcz d2 (H3: byte-identical)" $O/b2.txt $RH/d2.txt
pair "ref f4 vs hcz d2 (H3: byte-identical)" $O/f4.txt $RH/d2.txt
pair "ctl b1 cold vs hcz d1 cold" $O/b1.txt $RH/d1.txt
pair "ctl f1 cold vs hcz d1 cold (no warm-up)" $O/f1.txt $RH/d1.txt
pair "ctl f3 cold vs f1 cold (in one start)" $O/f3.txt $O/f1.txt
pair "ctl f1 cold vs dsm-225 cold" $O/f1.txt $RM/deep-dsm-225.txt
if [ -n "$FIXIMG" ]; then
  pair "g2 hit vs g1 cold (S3 fix, nonce, first long prefill after warm-up) [fix: same]" $O/g2.txt $O/g1.txt
  pair "h2 hit vs h1 cold (S3 fix, deep) [fix: same]" $O/h2.txt $O/h1.txt
  pair "ctl h1 cold (fix) vs hcz d1 cold (stock image) [fix: cold unchanged]" $O/h1.txt $RH/d1.txt
fi
if [ "$DRY" = 1 ]; then echo "+ RESULT xhr groups (deep family by md5) + summary"
else
  log "RESULT xhr groups, deep prompt answers by md5 8: $(python3 -I $DRV groups $RM/deep-dsm-225.txt $RH/d1.txt $RH/d2.txt $O/b1.txt $O/b2.txt $O/f1.txt $O/f2.txt $O/f3.txt $O/f4.txt $O/h1.txt $O/h2.txt 2>&1 | grep -m1 '^GROUPS ' | cut -c8- | cut -c1-500)"
  log "RESULT xhr summary: $(python3 -c 'import json,sys
p = dict(l.rstrip().rsplit(" ", 1) for l in open(sys.argv[1]) if l.strip())
g = lambda a: next((v for k, v in p.items() if k.split(" ")[0] == a), "n/a")
def dropped(tag):
    try:
        d = json.load(open(sys.argv[2] + "/" + tag + ".txt.json")); b, a = d.get("tiers_before") or {}, d.get("tiers_after") or {}
        return a.get("dropped") - b.get("dropped")
    except Exception: return "?"
A, B, F1, F2 = g("a2"), g("b2"), g("f2"), g("f4")
refs = [x for k, x in p.items() if k.startswith("ref ")]
same = lambda v: v == "identical"
diff = lambda v: v == "different"
if diff(B) and diff(F2) and same(F1): v = "H3 (shared head entry from a short request): resumes differ exactly where short requests published the head first" + (", byte-identical to hcz d2" if refs and all(same(x) for x in refs) else ", NOT byte-identical to hcz d2 (" + "/".join(refs) + ")")
elif diff(F1) and same(B) and same(F2): v = "H1 (first long prefill of a start): only the first long prefill resumes wrong"
elif same(B) and same(F1) and same(F2): v = "H2 or another first-chain-only cause (b, f2, f4 exact)" + (": a2 differs" if diff(A) else ": a2 exact too -- R1 not reproduced")
else: v = "MIXED / UNDECIDED"
fix = ""
if g("g2") != "n/a":
    fix = " | FIX: g2/g1 %s, h2/h1 %s, h1 vs stock d1 %s" % (g("g2"), g("h2"), next((v for k, v in p.items() if k.startswith("ctl h1")), "n/a"))
print("%s | a2/a1 %s (a1 dropped %s), b2/b1 %s (b1 dropped %s), f2/f1 %s (f1 dropped %s), f4/f3 %s (f3 dropped %s) | ctl: %s%s" % (
    v, A, dropped("a1"), B, dropped("b1"), F1, dropped("f1"), F2, dropped("f3"),
    ", ".join(k[4:].split(" (")[0] + " " + x for k, x in p.items() if k.startswith("ctl ")), fix))' $O/pairs.txt $O 2>&1 | cut -c1-900)"
fi
unset $SWITCHES
log "RESULT xhr $ITEM: done -- $O"
