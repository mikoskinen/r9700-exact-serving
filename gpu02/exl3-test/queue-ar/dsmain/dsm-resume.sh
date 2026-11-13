#!/bin/bash
# dsm-resume.sh (10.11.2026, queue-ar/dsmain, ds41 resume agent) -- NOT a queue item; the body of 57-p4zyzzzzz-ds-ho-resume.sh.
# QUESTION (lever R1, correctness): does a prefix-cache resume answer exactly as the cold prefill of the same prompt does?
# ONE READING SO FAR (10.11. 57-p4zyzzzzz-ds-gzzz-attnfold, image ds41-k1-1d1f782b): the 136k deep prompt resumed at 133120 of
#   136216 tokens (arm A) answered 1863 chars, DIFFERENT at char 170 from the cold answer (1877 chars = runs/xdsm/deep-dsm-225.txt);
#   the 28k copy resumed at 26624 was IDENTICAL (a copy reproduces its source through small numeric differences: weak probe).
# THE ENGINE'S CLAIM (startup line, core/sched/scheduler.cpp plan_chunk / tail_checkpoint / admit_one): "prefix resume: exact --
#   a hit resumes only where an idle server's cold prefill of the same prompt starts a chunk (every 4096 tokens from 2048) and a
#   prompt's tail snapshot is taken where its last chunk starts, so the hit recomputes the cold run's own chunks and answers
#   bit-identically". The snapshot carries the compressor tail rings, the Engram token window and the 128-row sliding window;
#   the compressed latent / index keys come back as prefix-cache blocks (from VRAM, or from the RAM tier).
# ONE START: image ds41-gg-8b311fcc (XHO_IMAGE; gg_gemv 0 = today's arithmetic, checked at /admin/fn4), the t3 env and
#   dsm-serve.sh's serve line (lib-29 best_vars: expert_tiered, ngram ram, ckint 2048, span 2, IL 8, MAXSEQS 8, MML 524288,
#   NST 0, RAM KV tier 84 GiB, keep-all, 512 slots) + MNBT 4160 + --live; start + warm-up + one 1-user fastcheck (information:
#   exactness does not depend on the slow state, so no retry restart). Every request greedy, thinking off, ignore_eos, 512 tokens
#   (resume_ho.py, streamed as deepq.py; logprobs top 2, a refusal -> plain), cached tokens and the /sessions tier counters
#   (promoted, restores) logged per request, sequential on an idle server (the claim's condition):
#   d1  deep cold (deepq.py's exact prompt, 136216 tokens)                 -> vs runs/xdsm/deep-dsm-225.txt (this image's cold)
#   d2  deep again: full hit, resumes at the tail snapshot 133120          -> vs d1                       [THE QUESTION]
#   d2b deep again: a second hit                                           -> vs d2 (resume repeatable?) and vs d1
#   s1  swap (deep, the filler's last 15500 chars ~4k tokens other text): partial hit on d1's chain at the INTERVAL
#       checkpoint 129024, two chunks recomputed                           -> vs s2 below
#   reset (POST /reset_prefix_cache); s2 swap cold                         -> s1 vs s2
#   d3  deep: partial hit on s2's chain at 129024                          -> vs d1
#   n1  "# nonce" + deep (a fresh chain) cold; n2 its hit                  -> n2 vs n1
#   m1  nonce 36k cold; m2 its hit (an r1 indexer layer is past the 32768-position keep threshold, the r2 layers are not)
#   m3  nonce 74k cold; m4 its hit (every indexer layer past it)
#   c   the 28k copy (ds41_tests.py copy --runs 2: run 1 cold, run 2 the hit at 26624) -> run2 vs run1, run1 vs dsm-225's i1
#   n3  n1's prompt after all that: a late hit (promoted / restores say whether its chain came back from the RAM tier) -> vs n1
#   reset; d4 deep cold again                                              -> vs d1 (cold vs cold in one start: the control)
#   then GROUPS over the deep family (dsm-225 d1 d2 d2b d3 d4) and one summary line.
# ~20 min (start ~6 min, the runs ~8 min, slack). Writes runs/xdsm/resume/. NO cc_sim. No power / host / fan settings;
# production compose files read only (--project-directory). Nothing goes into a production line from here.
# DRYRUN=1 bash dsm-resume.sh : prints every command, runs and writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
Y=$HD/ds41-gg-tp4.yaml
O=$R/xdsm/resume; export STATSDIR=$O/stats
RM=$R/xdsm                                      # item a's dsm-225 (main image, NST 0): deep-dsm-225.txt, copy-dsm-225-i1-run1.txt
export IMAGE=${XHO_IMAGE:-radiance-local:ds41-gg-8b311fcc-5c13e555-l}
ITEM=${XHO_ITEM:-57-p4zyzzzzz-ds-ho-resume}
DRV=$HD/resume_ho.py
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
SWAPC=${XHO_SWAPC:-15500}
NONCE=${XHO_NONCE:-ho$(date +%s)}
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_GG_GEMV RADIANCE_DS41_ARGATHER_MAX RADIANCE_DS41_SHEXP_LANE RADIANCE_DEBUG_MISS_LOG
  RADIANCE_DS41_KVFOLD RADIANCE_DS41_OFOLD RADIANCE_DS41_ONEFILL RADIANCE_DS41_PIPE"
# ---------------------------------------------------------------- helpers (dsm-serve.sh / dsm-pipe.sh's, copied; log tag xho)
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xho $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xho $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  log "RESULT xho $1 warm-up answers vs stock ($(basename $2)): $r"
}
modes() {   # modes TAG: the t3 indexer note and the exact-resume / snapshot lines -> 0, else 1 (stopped: the premise is gone)
  [ "$DRY" = 1 ] && { echo "+ modes $1: indexer note (mxfp4, latent bf16, compressor tail bound, keep from 32768, split from 4096) + 'prefix resume: exact' + 'checkpoints every 2048' + keep-all + 'a checkpoint slot carries its window' + /admin/fn4 gg_gemv 0"; return 0; }
  local MLF PL SW ok=0
  MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //' | cut -c1-700)
  PL=$(grep -E 'prefix resume:|^I prefix cache: |carries its window|^I chunk geometry|^I checkpoint slots' $O/start-$1.log 2>/dev/null | sed 's/^[A-Z] *//' | cut -c1-260 | tr '\n' '|')
  SW=$(curl -s -m 10 $URL/admin/fn4 2>/dev/null | cut -c1-400)
  log "RESULT xho $1 modes: $MLF | prefix: $PL | switches: $SW"
  for w in "keys mxfp4" "latent bf16" "compressor tail bound" "candidate restriction on from 32768" "row split on from 4096"; do
    echo "$MLF" | grep -q "$w" || { log "RESULT xho $1: VIRHE the indexer note lacks '$w'"; ok=1; }; done
  for w in "prefix resume: exact" "checkpoints every 2048 tokens" "retention 'keep-all'" "a checkpoint slot carries its window"; do
    echo "$PL" | grep -q "$w" || { log "RESULT xho $1: VIRHE the start log lacks '$w'"; ok=1; }; done
  echo "$SW" | grep -q '"gg_gemv": 0' || { log "RESULT xho $1: VIRHE gg_gemv is not 0 at /admin/fn4 ($SW)"; ok=1; }
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
  [ -n "$j" ] || { log "RESULT xho $tag ($kind): VIRHE no answer ($(tail -2 $O/$tag.out | tr '\n' ' ' | cut -c1-300))"; return 1; }
  [ "$ex" = tail ] && ex=$(exptail $(python3 -c 'import json,sys; print(json.loads(sys.argv[1]).get("prompt_tokens") or 0)' "$j"))
  log "RESULT xho $tag ($kind): $(python3 -c 'import json,sys
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
  [ "$DRY" = 1 ] && { echo "+ RESULT xho pair $1: resume_ho.py cmp $2 $3"; return 0; }
  local c; c=$(python3 -I $DRV cmp $2 $3 2>&1 | grep -m1 '^CMP ' | cut -c5-)
  log "RESULT xho pair $1: $(python3 -c 'import json,sys
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
  [ "$DRY" = 1 ] || log "RESULT xho $1: $(grep -m1 '^RHO ' $O/$1.out | cut -c5- | cut -c1-300 || tail -1 $O/$1.out)"
}
# ---------------------------------------------------------------- main
log "RESULT xho $ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) one start, deep / swap ($SWAPC chars) / nonce ($NONCE) 136k / 36k / 74k / 28k copy, cold vs prefix-cache resume byte for byte; reference $RM/deep-dsm-225.txt"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xho: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xho: VIRHE image $IMAGE not on gpu02"; exit 1; }
  [ -e $Y ] && [ -e $DRV ] && [ -e $HOME/gpu/ds41/ds41_tests.py ] || { log "RESULT xho: VIRHE $Y, $DRV or ds41_tests.py missing"; exit 1; }
  [ -f $MODELDIR/$M225 ] && [ -s $MODELDIR/$M225.sha256ok ] || { log "RESULT xho: VIRHE $M225 missing or not verified"; exit 1; }
  [ "$(timeout 120 docker run --rm --network none --entrypoint sh $IMAGE -c 'grep -q "prefix resume: exact" /opt/radiance/bin/radiance && grep -q reset_prefix_cache /opt/radiance/bin/radiance && echo yes' 2>/dev/null)" = yes ] \
    || { log "RESULT xho: VIRHE $IMAGE's engine has no exact prefix resume or no /reset_prefix_cache"; exit 1; }
  for f in $RM/deep-dsm-225.txt $RM/copy-dsm-225-i1-run1.txt; do [ -s $f ] || log "RESULT xho: note -- reference $f missing (that pair reads 'not compared')"; done
fi
mkd $O $O/guard $STATSDIR
run "rm -f $O/pairs.txt"
down
evict
TAG=ho
t3env; modelvars225
log "RESULT xho $TAG: start IMAGE=$IMAGE MODEL=$MODEL NST=$NST $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
restart $TAG || { keeplog $TAG-fail; log "RESULT xho $TAG: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$TAG.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; exit 1; }
ms=$(check $TAG 1)
[ "$DRY" = 1 ] || log "RESULT guard xho $TAG: 1-user ${ms} ms/step (information only: exactness does not depend on the slow state; fast is <= $LIM, bz dsd-d0 15.59)"
modes $TAG || { keeplog $TAG-modes; log "RESULT xho $TAG: VIRHE the modes are not as asked -> stopped"; exit 1; }
identw $TAG $WARMREF
# -------- the deep prompt: cold, full hit twice, partial hit at an interval checkpoint, its cold, the partial hit back
rho d1 deep 0
rho d2 deep tail
rho d2b deep tail
stopfile && exit 0
rho s1 swap 129024 --swap-chars $SWAPC
pcreset r1
rho s2 swap 0 --swap-chars $SWAPC
rho d3 deep 129024
stopfile && exit 0
# -------- a fresh chain at 136k, then 36k and 74k (the indexer's keep threshold 32768 positions), then the 28k copy
rho n1 nonce 0 --nonce ${NONCE}a
rho n2 nonce tail --nonce ${NONCE}a
rho m1 nonce 0 --nonce ${NONCE}b --depth 36000
rho m2 nonce tail --nonce ${NONCE}b --depth 36000
rho m3 nonce 0 --nonce ${NONCE}c --depth 74000
rho m4 nonce tail --nonce ${NONCE}c --depth 74000
stopfile && exit 0
run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 2 --offset 0 --tag=-ho --outdir $O > $O/copy-ho.txt 2>&1)"
[ "$DRY" = 1 ] || log "RESULT xho copy (28k, run 1 cold, run 2 the hit): $(grep '^J ' $O/copy-ho.txt | grep '"test": "copy"' | python3 -c 'import json,sys
for l in sys.stdin:
    d=json.loads(l[2:]); u=d.get("usage") or {}
    print("run %s: prompt %s tok, cached %s, TTFT %.2f s, exact %s, %s chars;" % (d.get("run"), u.get("prompt_tokens"), d.get("cached"), d.get("ttft") or 0, d.get("exact"), d.get("out_chars")), end=" ")' 2>&1 | cut -c1-400) (expected run 2 cached $(exptail 28262) at 28262 tokens)"
# -------- a late hit of n1's chain (the RAM tier's way back, when its VRAM was given up), then the cold control
rho n3 nonce tail --nonce ${NONCE}a
pcreset r2
rho d4 deep 0
keeplog $TAG
down
# -------- the comparisons
pair "d1 cold vs dsm-225 cold (this image vs item a)" $O/d1.txt $RM/deep-dsm-225.txt
pair "d2 full hit 133120 vs d1 cold [the question]" $O/d2.txt $O/d1.txt
pair "d2b second hit vs d2 hit (resume repeatable)" $O/d2b.txt $O/d2.txt
pair "d2b second hit vs d1 cold" $O/d2b.txt $O/d1.txt
pair "s1 partial hit 129024 vs s2 cold (swap prompt)" $O/s1.txt $O/s2.txt
pair "d3 partial hit 129024 vs d1 cold" $O/d3.txt $O/d1.txt
pair "n2 hit vs n1 cold (nonce 136k)" $O/n2.txt $O/n1.txt
pair "m2 hit vs m1 cold (nonce 36k)" $O/m2.txt $O/m1.txt
pair "m4 hit vs m3 cold (nonce 74k)" $O/m4.txt $O/m3.txt
pair "copy run 2 hit vs run 1 cold (28k)" $O/copy-ho-run2.txt $O/copy-ho-run1.txt
pair "copy run 1 cold vs dsm-225 i1" $O/copy-ho-run1.txt $RM/copy-dsm-225-i1-run1.txt
pair "n3 late hit vs n1 cold" $O/n3.txt $O/n1.txt
pair "d4 cold vs d1 cold (in one start, the control)" $O/d4.txt $O/d1.txt
if [ "$DRY" = 1 ]; then echo "+ RESULT xho groups (deep family by md5) + summary"
else
  log "RESULT xho groups, deep prompt answers by md5 8: $(python3 -I $DRV groups $RM/deep-dsm-225.txt $O/d1.txt $O/d2.txt $O/d2b.txt $O/d3.txt $O/d4.txt 2>&1 | grep -m1 '^GROUPS ' | cut -c8- | cut -c1-400)"
  log "RESULT xho engine log: $(grep -cE '^E ' $O/engine-$TAG.log 2>/dev/null) E lines; resets: $(grep -c 'prefix cache: reset' $O/engine-$TAG.log 2>/dev/null); $(grep -E '^E ' $O/engine-$TAG.log 2>/dev/null | head -2 | tr '\n' ' ' | cut -c1-300)"
  log "RESULT xho summary: $(python3 -c 'import sys
p = dict(l.rstrip().rsplit(" ", 1) for l in open(sys.argv[1]) if l.strip())
hits = [k for k in p if (" hit" in k.split(" vs ")[0]) and " cold" in k]
diff = [k.split(" (")[0].split(" [")[0] for k in hits if p[k] == "different"]
same = [k for k in hits if p[k] == "identical"]
ctl = next((p[k] for k in p if k.startswith("d4 ")), "n/a"); img = next((p[k] for k in p if k.startswith("d1 ")), "n/a")
rep = next((p[k] for k in p if k.startswith("d2b second hit vs d2")), "n/a")
v = ("RESUME DIFFERS FROM COLD (a second reading of the 10.11. attnfold observation)" if diff and ctl == "identical" else
     "resume matches cold in every pair (the 10.11. observation NOT reproduced)" if hits and not diff and len(same) == len(hits) and ctl == "identical" else
     "UNDECIDED: the cold control d4 vs d1 is " + ctl)
print("%s | hit vs cold: %d of %d IDENTICAL%s | cold vs cold d4/d1 %s | d1 vs dsm-225 %s | hit vs hit d2b/d2 %s" % (
    v, len(same), len(hits), (" -- DIFFERENT: " + ", ".join(diff)) if diff else "", ctl, img, rep))' $O/pairs.txt 2>&1 | cut -c1-700)"
fi
unset $SWITCHES
log "RESULT xho $ITEM: done -- $O"
