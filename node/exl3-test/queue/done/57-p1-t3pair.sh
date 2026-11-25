#!/bin/bash
# 57-p1: tuned3 candidate vs off, paired fixed-work cc_sim (--trace-seeded, X79_SFX=p -> x79-29p-*, x79-225p-*), deep answer vs off
export X79_SFX=${X79_SFX:-p} X79_DEEP=${X79_DEEP:-1} X79_CCSIM=${X79_CCSIM:-$HOME/gpu/ds41/exl3-test/tools/ccsim/cc_sim.py} X79_CCWRAP=${X79_CCWRAP:-$HOME/gpu/ds41/exl3-test/ccsim-exl3-turns.sh}
export X79_CCARGS=${X79_CCARGS:-"--duration 2700 --turns 15 --warmup-turns 3 --effort xhigh,medium --gen-mode forced --seed 1 --cpt-fixed --temperature 0 --trace-seeded"}
# 79-shp-levels.sh (queue item, 7.11.2026, queue-ar; gpu02, all 4 cards, TP4, MML 524288) -- the arena levels + prefill stager
# on DeepSeek-V4.1 EXL3, off (today) vs on, 2.9 bpw first, then 2.25 bpw.
# WHY. deepseek41 never declared at a smaller step size (RadArchProbe::shape_probe_ok), so the core made no arena levels:
# the 2.58 GiB activation arena is never lent to the expert slab at decode, and the prefill stager (core/place/stager.h:
# the next layers' host-pinned experts copied onto the card by DMA ahead of the layers that read them) is never armed. 61b
# /stats: stage_layers 0; a 2.9 prefill/mixed step streams 57.9 GB (all ranks) through zero-copy GEMMs, 46 % of cc_sim busy
# time; 78 (pf-zc): the prefill band's host reads are fully exposed (exposure 1.0-1.7). Estimate (tools/pfstage.py): a
# prefill chunk -15 .. -44 % at 2.9, -21 .. -41 % at 2.25 (DMA 12-24 GB/s), plus ~1.5-2.2 GiB more slab at decode (2.9
# model -0.5 .. -1.0 ms/step at 1 user).
# IMAGE (one image, both arms): radiance-local:ds41-exl3-117-<ds41-shp>-fa0e9580-tuned2-h<ds41-ar>-shp (build/ar/assemble-shp.sh)
#   = the 61b image (7af8fee3-fa0e9580-tuned2-h7e342763) with deepseek41.so of ds41-shp (RADIANCE_DS41_LEVELS=1: the sizing
#   declare; unset = refused as before) and bin/radiance of ds41-ar (RADIANCE_ARENA_LEVEL_MAX; unset = every rung).
# ARMS (ARMS="29:off 29:on 225:off 225:on"; model:arm), every arm on the tuned2 61b line -- idx (keep,split), mx4 keys,
# dual heat 0.7 + decode pin 0.8, gu25c kernels, MNBT 4160 + moves 16, ckint 2048, span 2, IL 8, MAXSEQS 8, RAM KV tier,
# --live -- with the BOUNDED compressor tail (RADIANCE_DS41_CTAIL unset; 61b ran full; levels need bound):
#   off  RADIANCE_DS41_LEVELS unset                                   (= today's engine with the bounded tail)
#   on   RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024         (levels 256 / 1024 / 2112: decode lends the arena;
#                                                                      steps above 1024 rows are staged)
#   onl  on + RADIANCE_STAGE_MIXED=0 (7.11.: a pass with decode rows reads the pool in place; lone chunks staged)
#   onh  onl + RADIANCE_ARENA_HOLD_MS=10000 (X79_HOLD_MS; the arena loan floor held 10 s instead of 2: fewer recalls)
#   onc  on + RADIANCE_STAGE_MIXED=2 (a mixed pass staged only when its chunk leaves more of the prompt: pf-like long
#        prompts beside decoders yes, one-chunk prompts no; lone chunks always) -- the -he6270a8a image
#   ons  on + RADIANCE_STAGE_SPAN=0 (the shared level keeps no ring room: it lends it; mixed steps unstaged, full-plan
#        steps staged) -- the -hd4a11cab image; onsh  ons + RADIANCE_ARENA_HOLD_MS=10000
#   t3   ons + RADIANCE_DS41_ARGATHER=on on the tuned3 image (X79_T3_IMAGE, default the t3ag tag below); ROUTER_FUSE /
#        MIX_FUSE / QLDS / AR_PCN set 0 by name (that image carries none of their code)
#   onl needs the -h94ececa9 image or later, onh the -hffb02130 image or later (X79_IMAGE; default below: hd4a11cab, all switches)
#   29: the 2.9 bpw .rad, HP29 31488, guard limit 30 ms; 225: the 2.25 bpw .rad, host pool 19456, limit 23 ms.
# PER ARM: gate; modes (indexer note; the serve log's "activation arena" / "prefill staging" lines -- an on arm without
# "prefill staging" is stopped, VIRHE); warm-up answers vs the stock reference (60 x29-best524 / x117-best524: IDENTICAL
# expected, every piece exact); i1 = copy test after a ~31k-token COLD prompt (120k chars of context + 2400 to copy, greedy,
# 900 tokens: the decode right after staged prefill chunks), compared byte for byte off vs on; probe; mb c4 / c8 / pf /
# turns; cc_sim u4 (forced); i2 = the same copy test cold at another offset after cc_sim (the lent slab churned), off vs on;
# a /stats snapshot at every phase end (stage_layers / staged_units / staged_bytes / stage_overflow / stage_copies, routed
# hit, stream and h2d bytes, steps, step ns) -> per-phase table; --live lines -> live.txt (profsum.py live).
# END: on vs off per model (guard, probe, c4, c8, pf, turns cold TTFT, cc_sim) with % differences.
# References (no baseline re-runs): 61b x29-t2-idx-mx4-heat-gu25c-vram (CTAIL full, levels off): guard 22.16, essay/code
# 23.2/21.6 ms, c4 72.8 tok/s 43.8 ms, c8 86.6, pf 980.1 tok/s TTFT 32.05 s, turns cold 36.13 s, cc_sim u4 74.9 tok/s TTFT p50
# 3.82 s, decode 31.9 ms/step; 2.25 xar-p80m16n4160: guard 17.15, c4 117.8 tok/s 28.9 ms, pf 1505.3 tok/s, cc_sim 80.4 tok/s.
# Time ~30 min an arm (61b: 27 min + two copy tests ~2 min): 2 h for the four; ARMS="29:off 29:on" = 1 h.
# Bounded: lib-pr266's restart (compose up, /health <= 900 s), every client under timeout, curl -m, docker logs -f killed on
# every exit (trap), down at the end.
# DRYRUN=1 bash 79-shp-levels.sh : prints every command, runs and writes nothing.
H79="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$H79/lib-29.sh" "$H79/../lib-29.sh" "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
for YS in "$H79/ds41-shp-tp4.yaml" "$H79/../ds41-shp-tp4.yaml" "$HOME/gpu/ds41/exl3-test/queue-ar/ds41-shp-tp4.yaml"; do [ -e "$YS" ] && break; done
[ -e "$YS" ] || { echo "ds41-shp-tp4.yaml not found"; exit 1; }
Y=$YS
O=$R/x79; export STATSDIR=$O/stats
export IMAGE=${X79_IMAGE:-radiance-local:ds41-exl3-117-d12aba46-fa0e9580-tuned2-hd4a11cab-shp}
ARMS=${ARMS:-"29:off 29:t3 225:off 225:t3"}   # 57-p1 (8.11. coordinator): tuned3 vs off, fixed-work paired cc_sim with --trace-seeded + deep exactness vs off
M29=deepseek-v4.1-flash-1017-exl3-2.9bpw.rad; M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_29=$R/x29/guard/x29-best524-warm-q.txt; WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
# keeplog NAME: the server's whole log so far (docker logs works on an exited container until it is removed), kept before the
# next restart or down removes the container -- a failed or slow start keeps its engine log (7.11.)
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
# lib-29's gate with keeplog before the retry and after a second slow check
gate() {
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
  log "RESULT guard $tag: $st, 1-user ${ms} ms/step (limit $lim)"
  case $st in SLOW*) return 2 ;; esac
}
FOLLOW_PID=""
follow() { [ "$DRY" = 1 ] && { echo "+ docker logs -f $CN > $1/engine.log &"; return; }
  docker logs -f $CN > $1/engine.log 2>&1 & FOLLOW_PID=$!; sleep 2; }
unfollow() { [ -n "$FOLLOW_PID" ] && { kill $FOLLOW_PID 2>/dev/null; wait $FOLLOW_PID 2>/dev/null; FOLLOW_PID=""; }; return 0; }
trap unfollow EXIT
mark() { [ "$DRY" = 1 ] && { echo "+ mark $1 -> $2/marks.txt"; return; }; echo "$1 $(wc -l < $2/engine.log) $(date +%s)" >> $2/marks.txt; }
snap() {   # snap DIR PHASE: /stats now (cumulative counters)
  run "curl -s -m 15 $URL/stats > $1/stats-$2.json 2>/dev/null; echo $2 >> $1/stats-order.txt"
}
ph() { mark $2 $1; snap $1 $2; }   # ph DIR PHASE: a phase ends
copyt() {   # copyt TAG OFFSET: the copy test after a ~31k-token cold prompt (deterministic prompt: same in every arm)
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset $2 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT x79 $1 copy after a cold prompt: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s ratio %s, %s chars out" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact"), d.get("ratio"), d.get("out_chars")))' 2>&1 | cut -c1-300)"
}
cmpcopy() {   # cmpcopy A B: two copy outputs byte for byte
  [ "$DRY" = 1 ] && { echo "+ cmp $O/copy-$1-run1.txt $O/copy-$2-run1.txt"; return 0; }
  [ -s $O/copy-$1-run1.txt ] && [ -s $O/copy-$2-run1.txt ] || { log "RESULT x79 copy $1 vs $2: not compared (missing output)"; return 0; }
  local r
  r=$(python3 -c 'import sys; a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read(); f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b))); print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' $O/copy-$1-run1.txt $O/copy-$2-run1.txt 2>&1)
  case $r in IDENTICAL*) log "RESULT x79 copy $1 vs $2: $r" ;; *) log "RESULT x79 copy $1 vs $2: VIRHE $r -- not exact" ;; esac
}
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT x79 $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT x79 $1 warm-up identity: no reference $2"; return 0; }
  log "RESULT x79 $1 warm-up answers vs stock ($(basename $2)): $(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("VIRHE DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)"
}
statsum() {   # statsum DIR TAG: per-phase deltas of the /stats snapshots
  [ "$DRY" = 1 ] && { echo "+ RESULT x79 $2 stats per phase from $1/stats-*.json"; return 0; }
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
         "staged_units": eb.get("staged_units", 0) - ea.get("staged_units", 0),
         "staged_GB": round((eb.get("staged_bytes", 0) - ea.get("staged_bytes", 0)) / G, 2),
         "stage_overflow": eb.get("stage_overflow", 0) - ea.get("stage_overflow", 0),
         "stage_copies": eb.get("stage_copies", 0) - ea.get("stage_copies", 0),
         "promotions": eb.get("promotions", 0) - ea.get("promotions", 0),
         "vram_units_end": next((t.get("units") for t in eb.get("tiers", []) if t.get("name") == "vram"), None)}
    print("PH " + json.dumps(o))
PY
  log "RESULT x79 $2 stats per phase: $(grep '^PH ' $1/stats-phases.txt | cut -c4- | python3 -c 'import json,sys
out = []
for l in sys.stdin:
    try: d = json.loads(l)
    except Exception: continue
    out.append("%s: %s st (%s pf/mx), %s ms/st, dec %s ms, pf/mx %s ms, hit %s %%, stream %s MB/st, staged %s layers %s GB (%s overflow), vram %s" % (
        d["phase"], d["steps"], d["prefill_mixed_steps"], d["ms_step"], d["ms_decode_step"], d["ms_prefill_mixed_step"], d["hit_pct_rank0"],
        d["stream_MB_step"], d["stage_layers"], d["staged_GB"], d["stage_overflow"], d["vram_units_end"]))
print("; ".join(out))' 2>&1 | cut -c1-1800)"
}
# deepq TAG (X79_DEEP=1): the fixed ~135k prompt (no nonce) and a 512-token greedy free-text answer (deepq.py) -> $O/deep-TAG.txt
DQ=$H79/deepq.py; [ -e $DQ ] || DQ=$HOME/gpu/ds41/exl3-test/queue-ar/deepq.py
deepq() {
  run "(cd $D && timeout 1800 python3 $DQ $O/deep-$1.txt ${X79_DEEP_TOK:-135000} 512 > $O/deep-$1.out 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT x79 $1 deep probe: $(grep -m1 '^DEEPQ ' $O/deep-$1.out | cut -c7- | cut -c1-200 || tail -2 $O/deep-$1.out | tr '\n' ' ')"
}
# cmpdeep MODEL ARM: the deep answer against the reference arm's (X79_REF, default off) of this or an earlier run, else the
# first arm of this run that wrote one
cmpdeep() {
  [ "${X79_DEEP:-0}" = 1 ] || return 0
  local a=x79-$1-$2 r=x79-$1-${X79_REF:-off}
  [ "$DRY" = 1 ] && { echo "+ cmp $O/deep-$a.txt $O/deep-$r.txt (or the first deep answer of the run)"; return 0; }
  [ -s $O/deep-$a.txt ] || { log "RESULT x79 deep $a: no answer"; return 0; }
  [ -s $O/deep-$r.txt ] || r=${DEEP_FIRST[$1]:-}
  [ -n "$r" ] && [ "$r" != "$a" ] && [ -s $O/deep-$r.txt ] || { DEEP_FIRST[$1]=${DEEP_FIRST[$1]:-$a}; log "RESULT x79 deep $a: the run's first answer (reference for later arms)"; return 0; }
  local res; res=$(python3 -c 'import sys; a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read(); f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b))); print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' $O/deep-$a.txt $O/deep-$r.txt 2>&1)
  case $res in IDENTICAL*) log "RESULT x79 deep $a vs $r: $res" ;; *) log "RESULT x79 deep $a vs $r: VIRHE $res -- not exact at depth" ;; esac
}
declare -A DEEP_FIRST
arm() {   # arm MODEL ARM
  local mo=$1 a=$2 tag P lim warmref rc ML AL SL
  local img0=$IMAGE
  case $a in t3*) export IMAGE=${X79_T3_IMAGE:-radiance-local:ds41-exl3-117-390664b3-fa0e9580-tuned3-hd4a11cab} ;; esac
  tag=x79-$mo${X79_SFX:-}-$a; P=$O/$tag   # X79_SFX: a set of arms of its own (e.g. -t for fixed-work cc_sim), not overwriting earlier ones
  unset RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT \
        RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN \
        RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN \
        RADIANCE_DS41_ARGATHER RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  case $a in
    off) ;;
    on)  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=${X79_LEVEL_MAX:-1024} ;;
    onl) export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=${X79_LEVEL_MAX:-1024} RADIANCE_STAGE_MIXED=0 ;;   # lone chunks only
    onc) export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=${X79_LEVEL_MAX:-1024} RADIANCE_STAGE_MIXED=2 ;;   # mixed staged only while the prompt continues
    ons) export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=${X79_LEVEL_MAX:-1024} RADIANCE_STAGE_SPAN=0 ;;   # no ring room at the shared level: lent, mixed unstaged
    onsh) export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=${X79_LEVEL_MAX:-1024} RADIANCE_STAGE_SPAN=0 \
                 RADIANCE_ARENA_HOLD_MS=${X79_HOLD_MS:-10000} ;;   # ons + the loan floor held 10 s (no recall a prompt)
    t3*) export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=${X79_LEVEL_MAX:-1024} RADIANCE_STAGE_SPAN=0 \
                RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
         # the tuned3 candidate (X79_T3_IMAGE): ons + the fused MoE gather/all-reduce; the untested pieces off by name. Variants:
         case $a in
           t3)    ;;
           t3l)   export RADIANCE_DS41_LATENT=nvfp4 ;;                          # 68: the packed compressed latent
           t3l16) export RADIANCE_DS41_LATENT=nvfp4 ;;                          # 68's capacity probe (MAXSEQS 16, else 12)
           t3A)   ;;                                                            # 69-16k: A (keep from 32768, split from 4096)
           t3B)   export RADIANCE_DS41_INDEX=keep ;;                            # B (no split)
           t3C)   export RADIANCE_DS41_SPLIT_MIN_LIVE=65536 ;;                  # C (split only from 64k live)
           *)     log "RESULT x79: VIRHE unknown arm '$a'"; export IMAGE=$img0; return 1 ;;
         esac ;;
    onh) export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=${X79_LEVEL_MAX:-1024} RADIANCE_STAGE_MIXED=0 \
                RADIANCE_ARENA_HOLD_MS=${X79_HOLD_MS:-10000} ;;   # + the loan floor held 10 s (fewer recalls; -hffb02130 image)
    *)   log "RESULT x79: VIRHE unknown arm '$a'"; return 1 ;;
  esac
  case $mo in
    29)  export MODEL=$M29; HP29=31488; lim=$GATE29; warmref=$WARM_REF_29 ;;
    225) export MODEL=$M225; HP29=19456; lim=23; warmref=$WARM_REF_225 ;;
    *)   log "RESULT x79: VIRHE unknown model '$mo'"; return 1 ;;
  esac
  best_vars; export MNBT=4160 XARGS="$XARGS --live"
  [ $a = t3l16 ] && export MAXSEQS=${X79_CAP_SEQS:-16}
  log "RESULT x79 $tag: start IMAGE=$IMAGE MODEL=$MODEL HOSTPOOL=$HOSTPOOL MNBT=$MNBT env: $(env | grep -E '^RADIANCE_' | sort | tr '\n' ' ')"
  if [ "$DRY" != 1 ]; then
    [ -f $MODELDIR/$MODEL ] && [ -s $MODELDIR/$MODEL.sha256ok ] || { log "RESULT x79 $tag: VIRHE $MODELDIR/$MODEL missing or not verified"; return 1; }
  fi
  mkd $P
  run "rm -f $P/marks.txt $P/stats-order.txt"
  gate $tag $lim; rc=$?
  if [ $rc = 1 ] && [ $a = t3l16 ] && [ "$MAXSEQS" -gt 12 ]; then   # 68: 16 does not start -> 12
    log "RESULT x79 $tag: MAXSEQS $MAXSEQS did not start ($(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -2 | tr '\n' ' ' | cut -c1-300)) -> MAXSEQS 12"
    export MAXSEQS=12; gate $tag $lim; rc=$?
  fi
  [ $rc = 1 ] && { log "RESULT x79 $tag: VIRHE not healthy ($(desc)); start log $O/start-$tag.log: $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  [ $rc = 2 ] && { log "RESULT x79 $tag: VIRHE slow state after one restart -> arm skipped; engine logs $O/engine-$tag-try1.log / -try2.log: $(grep -hE '^(E|F) |fault|abort' $O/engine-$tag-try1.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  if [ "$DRY" != 1 ]; then
    MLF=$(docker logs $CN 2>&1 | grep -m1 'deepseek41 indexer' | sed 's/^radiance: //')   # whole note: the tail mode is at its end
    ML=$(echo "$MLF" | cut -c1-900)
    AL=$(docker logs $CN 2>&1 | grep -m3 -E 'activation arena' | cut -c1-400 | tr '\n' '|')
    SL=$(docker logs $CN 2>&1 | grep -m1 -E 'prefill staging: a .* ring' | cut -c1-300)   # the stager's own line, not the budget row
    log "RESULT x79 $tag modes: $ML | arena: $AL | staging: ${SL:-none} | $(grep -m1 -E '^  activation arena' $O/start-$tag.log | tr -s ' ' | cut -c1-120)"
    case "$MLF" in *"compressor tail bound"*) ;; *) log "RESULT x79 $tag: VIRHE the indexer note does not say 'compressor tail bound' -> arm stopped"; down; return 1 ;; esac
    if [[ $a == t3* ]] && ! docker logs $CN 2>&1 | grep -q 'the MoE gather and its all-reduce as one kernel'; then
      keeplog $tag-nomode; log "RESULT x79 $tag: VIRHE no RADIANCE_DS41_ARGATHER note in the serve log (image $IMAGE) -> arm stopped"; down; export IMAGE=$img0; return 1; fi
    want=""
    case $a in t3l|t3l16) want="latent nvfp4" ;; t3B) want="row split off" ;; t3C) want="split on from 65536" ;; esac
    if [ -n "$want" ]; then case "$MLF" in *"$want"*) ;; *)
      keeplog $tag-nomode; log "RESULT x79 $tag: VIRHE the indexer note lacks '$want' -> arm stopped"; down; export IMAGE=$img0; return 1 ;; esac; fi
    if [ $a = onc ] && ! docker logs $CN 2>&1 | grep -q 'staged only while its prompt continues'; then
      keeplog $tag-nomode; log "RESULT x79 $tag: VIRHE the engine does not report RADIANCE_STAGE_MIXED=2 (image too old?) -> arm stopped"; down; return 1; fi
    if [ $a != off ] && [ -z "$SL" ]; then keeplog $tag-nostage; log "RESULT x79 $tag: VIRHE no prefill staging line (levels refused?) -> arm stopped"; down; return 1; fi
    if [ $a = off ] && [ -n "$SL" ]; then log "RESULT x79 $tag: VIRHE the off arm staged -> arm stopped"; down; return 1; fi
  fi
  identw $tag $warmref
  follow $P; ph $P start
  case $a in
  t3l16)   # 68's capacity probe: the cache's size at this MAXSEQS, then decode at 8 and at MAXSEQS streams
    [ "$DRY" = 1 ] || log "RESULT x79 $tag capacity MAXSEQS $MAXSEQS: $(grep -m1 -E '^I kv: the pool' $O/start-$tag.log | cut -c3-220) | $(grep -m1 -E '^ +kv cache +[0-9.]+ GiB' $O/start-$tag.log | tr -s ' ' | cut -c1-200) | $(grep -m1 -oE 'KV [0-9.]+ GiB a card: [0-9]+ tokens, [0-9.]+ sessions at [0-9]+' $L 2>/dev/null | tail -1) | $(grep -m2 -iE 'kv.*(blocks|pages)' $O/start-$tag.log | tr -s ' ' | cut -c1-160 | tr '\n' '|')"
    probe $tag; ph $P probe
    for p in c8 c$MAXSEQS; do stopfile && break; mbp $tag $p; ph $P $p; done ;;
  t3A|t3B|t3C)   # 69-16k: fresh cold prompts only
    run "(cd $D && timeout 900 python3 ds41_tests.py prefill --depths 16384,32768,65536 --cpt 3.95 --reps 1 > $O/$tag-cold.txt 2>&1)"
    [ "$DRY" = 1 ] || log "RESULT x79 $tag fresh cold prefill: $(python3 -c 'import json,sys; print("; ".join("%s tok %s tok/s TTFT %s s" % (d.get("prompt_tokens"), d.get("prefill_tps"), d.get("ttft")) for d in (json.loads(l[2:]) for l in open(sys.argv[1]) if l.startswith("J "))))' $O/$tag-cold.txt 2>&1 | cut -c1-300)"
    ph $P cold ;;
  *)
    copyt $tag-i1 0; ph $P i1
    [ "${X79_DEEP:-0}" = 1 ] && { deepq $tag; ph $P deep; }
    probe $tag; ph $P probe
    for p in c4 c8 pf turns; do stopfile && break; mbp $tag $p; ph $P $p; done
    # X79_CCARGS / X79_CCSIM / X79_CCWRAP: cc_sim's args, script and wrapper (a fixed-work run: CCARGS with --turns N
    # --warmup-turns K --cpt-fixed, CCSIM a cc_sim.py that has them, CCWRAP queue-ar/tools/ccsim/ccsim-exl3.sh); unset = as before
    stopfile || run "CCARGS='${X79_CCARGS:-}' CCSIM='${X79_CCSIM:-}' timeout 3300 ${X79_CCWRAP:-$T/ccsim-exl3.sh} 4 $tag > $O/$tag-ccsim-u4.out 2>&1"
    ph $P ccsim
    copyt $tag-i2 7000; ph $P i2 ;;
  esac
  unfollow
  run "docker logs $CN > $O/$tag-engine-full.log 2>&1"
  statsum $P $tag
  run "python3 $T/profsum.py live $P > $P/live.txt 2>&1"
  down
  export IMAGE=$img0
  log "RESULT x79 $tag: done"
}
ccpair() {   # ccpair MODEL ARM: the two arms' fixed-work cc_sim runs request by request (cc_sim.py --pair), when both are --turns runs
  [ "$DRY" = 1 ] && { echo "+ cc_sim.py --pair <x79-$1-${X79_REF:-off} jsonl> <x79-$1-$2 jsonl>"; return 0; }
  local A B
  A=$(ls -t $R/ccsim-*-x79-$1-${X79_REF:-off}-u4.jsonl 2>/dev/null | head -1); B=$(ls -t $R/ccsim-*-x79-$1-$2-u4.jsonl 2>/dev/null | head -1)
  [ -n "$A" ] && [ -n "$B" ] || return 0
  head -1 $A | grep -q '"turns": [1-9]' && head -1 $B | grep -q '"turns": [1-9]' || { log "RESULT x79 $1 $2 vs off cc_sim: not paired (not both --turns runs)"; return 0; }
  log "RESULT x79 $1 $2 vs off cc_sim paired: $(timeout 120 python3 ${X79_CCSIM:-$HOME/gpu/bench/cc_sim.py} --pair $A $B 2>&1 | tr '\n' ';' | cut -c1-900)"
}
cmparm() {   # cmparm MODEL [ARM]: ARM (default on) vs off from the arms' outputs
  local b=${2:-on}
  [ "$DRY" = 1 ] && { echo "+ RESULT x79 $1 $b vs ${X79_REF:-off}"; return 0; }
  log "RESULT x79 $1 $b vs ${X79_REF:-off}: $(python3 - $O x79-$1-${X79_REF:-off} x79-$1-$b <<'PY' 2>&1 | cut -c1-1500
import json, re, sys, os
O, a, b = sys.argv[1:4]
def mb(tag, p):
    try:
        for l in open(f"{O}/{tag}-{p}.txt"):
            if l.startswith("MB "): return json.loads(l[3:])
    except OSError: pass
    return {}
def cc(tag):
    try: t = open(f"{O}/{tag}-ccsim-u4.out").read()
    except OSError: return {}
    m = re.search(r"ttft p50/p95/max ([\d.]+)/([\d.]+)/[\d.]+ s, gen/agent p50 ([\d.]+) tok/s, total gen ([\d.]+) tok/s.*?([\d.]+) ms/decode step; all steps \d+, ([\d.]+) ms/step", t)
    return dict(zip(("ttft50", "ttft95", "agent", "total", "dec_ms", "all_ms"), map(float, m.groups()))) if m else {}
def gd(tag):
    try: return float(open(f"{O}/guard-{tag}.txt").read().split()[-1])
    except Exception: return None
rows = [("guard ms", gd(a), gd(b)),
        ("c4 tok/s", mb(a, "c4").get("agg_dec_tps"), mb(b, "c4").get("agg_dec_tps")), ("c4 ms", mb(a, "c4").get("ms_decode_step"), mb(b, "c4").get("ms_decode_step")),
        ("c8 tok/s", mb(a, "c8").get("agg_dec_tps"), mb(b, "c8").get("agg_dec_tps")),
        ("pf tok/s", mb(a, "pf").get("prefill_tps"), mb(b, "pf").get("prefill_tps")), ("pf TTFT s", mb(a, "pf").get("prefill_ttft"), mb(b, "pf").get("prefill_ttft")),
        ("turns cold s", mb(a, "turns").get("cold_ttft"), mb(b, "turns").get("cold_ttft")), ("turns hit s", mb(a, "turns").get("ttft_hit_mean"), mb(b, "turns").get("ttft_hit_mean"))]
ca, cb = cc(a), cc(b)
rows += [("cc total tok/s", ca.get("total"), cb.get("total")), ("cc agent tok/s", ca.get("agent"), cb.get("agent")),
         ("cc TTFT p50", ca.get("ttft50"), cb.get("ttft50")), ("cc decode ms", ca.get("dec_ms"), cb.get("dec_ms")), ("cc all ms/step", ca.get("all_ms"), cb.get("all_ms"))]
out = []
for n, x, y in rows:
    if x is None or y is None: out.append(f"{n} {x} -> {y}"); continue
    out.append(f"{n} {x} -> {y} ({(y / x - 1) * 100:+.1f} %)" if x else f"{n} {x} -> {y}")
print("; ".join(out))
PY
)"
}
log "RESULT x79 79-shp-levels: start ARMS [$ARMS] IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) yaml $Y"
if [ "$DRY" != 1 ]; then
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT x79: VIRHE image $IMAGE not on gpu02"; exit 1; }
  for v in RADIANCE_DS41_LEVELS RADIANCE_ARENA_LEVEL_MAX RADIANCE_DS41_CTAIL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES; do
    grep -q "^ *$v:" $Y || { log "RESULT x79: VIRHE $Y does not pass $v"; exit 1; }
  done
  # the arms' switches must reach the container and exist in the image's engine
  for x in $ARMS; do case ${x##*:} in onl|onc) need=RADIANCE_STAGE_MIXED ;; t3|t3A) need="RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER" ;; t3l|t3l16) need="RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER RADIANCE_DS41_LATENT" ;;
      t3B) need="RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER RADIANCE_DS41_INDEX" ;; t3C) need="RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER RADIANCE_DS41_SPLIT_MIN_LIVE" ;; ons) need=RADIANCE_STAGE_SPAN ;; onsh) need="RADIANCE_STAGE_SPAN RADIANCE_ARENA_HOLD_MS" ;; onh) need="RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS" ;; *) need= ;; esac
    for v in $need; do
      grep -q "^ *$v:" $Y || { log "RESULT x79: VIRHE $Y does not pass $v (arm $x)"; exit 1; }
      img=$IMAGE; [[ ${x##*:} == t3* ]] && img=${X79_T3_IMAGE:-radiance-local:ds41-exl3-117-390664b3-fa0e9580-tuned3-hd4a11cab}
      [ "$(timeout 120 docker run --rm --network none --entrypoint sh $img -c "cat /opt/radiance/bin/radiance /opt/radiance/share/radiance/architectures/deepseek41.so | grep -c $v" 2>/dev/null)" -gt 0 ] 2>/dev/null \
        || { log "RESULT x79: VIRHE $img's engine / plugin does not know $v (arm $x)"; exit 1; }
    done
  done
  [ -n "$(docker ps --format '{{.Names}}' | grep -E '^radiance-prod')" ] && { log "RESULT x79: VIRHE production is running"; exit 1; }
fi
mkd $O $STATSDIR
evict
done_arms=""
for x in $ARMS; do
  stopfile && break
  arm ${x%%:*} ${x##*:} && done_arms="$done_arms $x"
  mo=${x%%:*}; b=${x##*:}; ms=$mo${X79_SFX:-}; rf=${X79_REF:-off}
  [[ " $done_arms " == *" $x "* ]] && cmpdeep $ms $b
  case $b in t3l16|t3A|t3B|t3C) continue ;; esac   # their own RESULT lines (capacity, cold prefill)
  if [ $b != $rf ] && [[ " $done_arms " == *" $x "* ]] && { [ "$DRY" = 1 ] || [[ " $done_arms " == *" $mo:$rf "* ]] || [ -s $O/guard-x79-$ms-$rf.txt ]; }; then
    cmpcopy x79-$ms-$rf-i1 x79-$ms-$b-i1; cmpcopy x79-$ms-$rf-i2 x79-$ms-$b-i2; cmparm $ms $b; ccpair $ms $b   # the reference: this run or an earlier one
  fi
done
# 69-16k: A / B / C side by side per model (fresh cold prefill tok/s at 16k / 32k / 64k, B and C against A)
for mo in 29 225; do
  ms=$mo${X79_SFX:-}
  [ "$DRY" = 1 ] && { case " $ARMS " in *" $mo:t3A "*) echo "+ RESULT x79 $ms 16k A/B/C table" ;; esac; continue; }
  [ -s $O/x79-$ms-t3A-cold.txt ] || continue
  log "RESULT x79 $ms 16k A/B/C (fresh cold prefill tok/s, B/C vs A): $(python3 - $O $ms <<'PY' 2>&1 | cut -c1-900
import json, sys
O, ms = sys.argv[1:3]
def rd(arm):
    try: return {d.get("target"): d.get("prefill_tps") for d in (json.loads(l[2:]) for l in open(f"{O}/x79-{ms}-{arm}-cold.txt") if l.startswith("J "))}
    except OSError: return {}
A = rd("t3A"); out = []
for t in sorted(A):
    row = [f"{t // 1024}k A {A[t]}"]
    for arm, n in (("t3B", "B keep"), ("t3C", "C split>=64k")):
        v = rd(arm).get(t)
        if v and A[t]: row.append(f"{n} {v} ({(v / A[t] - 1) * 100:+.1f} %)")
    out.append(", ".join(row))
print("; ".join(out) or "no data")
PY
)"
done
unset RADIANCE_DS41_IDXKEY RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_DS41_LEVELS RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER RADIANCE_DS41_ROUTER_FUSE \
      R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN
down
log "RESULT x79 79-shp-levels: done --$done_arms | refs 61b (CTAIL full) guard 22.16, c4 72.8 tok/s, pf 980.1 tok/s, cc_sim 74.9 tok/s; 2.25 p80m16n4160 guard 17.15, pf 1505.3 tok/s, cc_sim 80.4 tok/s"
