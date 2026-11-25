#!/bin/bash
# 84-fn-tuned.sh (queue item, 7.11.2026, queue-ar; gpu02, cards 83:00 + 86:00, every radiance container DOWN) -- does the
# tuned engine help Qwen3.8-Flash-Next (arch qwen4exp_fp8)? The production serve line (flashnext-prod-dp-a.yaml, read 7.11.)
# on the tuned engine, arms against each other; NO 1.15.1 baseline run -- the production references are cited below.
# FNSET picks the model and its default arms:
#   A  ~/models/flashnext-iq4r/qwen3.8-next-flash-fp8-iq4r-moe.rad (production's iq4r)   ARMS "stock heat nomix"
#   B  ~/models/flashnext-exl3/qwen3.8-flash-next-exl3.rad (xeon's EXL3 4.05 bpw)       ARMS "stock heat"
#      B first: its .shaok, rad-info (the container's expert encoding must be exl3) and rad-info --plugins on the image.
# IMAGE (FN_IMAGE): A = the tuned3 image (radiance-local:ds41-exl3-117-390664b3-fa0e9580-tuned3-hd4a11cab, 28e67c5efc4c;
#   its qwen4exp_fp8.so is the 1.17 base's, which reads iq4r); B = its -fnx2 variant (1ff524ff4fc3, 8.11. evening: -fnx's
#   declare refused the container twice -- EXL3 experts cannot be sliced, and 4 layers keep 2-4 experts bf16 -- so the
#   plugins of ds41-fnx 996e4e4 serve EXL3 experts expert-parallel (whole per rank; the plugin note prices the routing
#   imbalance at ~1 ms a step at one sequence) with the bf16 ones beside them; a 2-fake-device host start declares and stops
#   where the iq4r container's does). -fnx was: the -fnq image
#   (qwen4exp_fp8.so rebuilt from the same tree, ds41-t3ag, with the EXL3 expert path the base's lacks) + qwen4exp_x3.so
#   (ds41-fnx 5319308: qwen4exp_fp8's source under the (qwen4exp, exl3) key). 7.11.'s FN-B on -fnq stopped at startup: "no
#   architecture plugin claims 'qwen4exp' at quantisation 'exl3'" -- the EXL3 container's descriptor is "exl3", -fnq's
#   plugin claims "" only. B's plugin check now requires "qwen4exp @ exl3" in rad-info --plugins before any start.
#   bin/radiance in both = ds41-ar d4a11ca (heat dual / pin / moves, levels ladder cap, STAGE_MIXED / ARENA_HOLD / STAGE_SPAN).
# Arms (engine env; everything else = the production line, fn-tp2.yaml):
#   stock  no tuned switch (the image's defaults: qwen4exp already declares arena levels, so the prefill stager and the arena
#          loan are on as in production)
#   heat   RADIANCE_HEAT_PREFILL_DECAY 0.7 + RADIANCE_HEAT_DECODE_PIN 0.8 + RADIANCE_HEAT_MOVES 16 (the DS41 tuned heat)
#   nomix  RADIANCE_STAGE_MIXED=0 (the one generic staging switch that applies here: a prefill chunk sharing its step with
#          decoders reads the pool in place; 79 found staged mixed steps 8-12 % slower on DeepSeek-V4.1). Not applicable to
#          this line: RADIANCE_DS41_LEVELS (deepseek41's plugin only; qwen4exp sets shape_probe_ok itself),
#          RADIANCE_STAGE_SPAN (needs --prefill-span; production has none), RADIANCE_ARENA_LEVEL_MAX (the ladder at MNBT 2048
#          is 256 / 1024, no rung just under the chunk), RADIANCE_ARENA_HOLD_MS (a fallback, not tested here).
#   heatnm heat + nomix (ARMS="... heatnm" adds it)
# Per arm: restart (fn-tp2.yaml, health <= 20 min, start log), gate = warm-up (ds41_tests correct --only-q + one 16k
# prefill) + prodfc.py (1-user greedy code decode, 384 tokens) with the 21 ms/step limit, one restart on a slow check; the
# serve log's arena / staging / heat lines; warm-up answers vs the run's first arm (identity: every switch is placement
# only) and, for information, vs a 1.15.1 wht6 answer set (q/guard/dp2w6-A-warm-q1.txt; another engine, so not a gate);
# i1 = copy test after a ~33k cold prompt; probe (essay, code, 512 tok); mb c4, c8, c16, pf (a 31k prompt beside 4
# decoders), turns; i2 = the copy test cold at another offset; engine log; down. Then each arm against the first.
# Production references (no rerun; ~/gpu/ds41/runs/qwen-tp4-1103.log = QL):
#   1-user gate, exact production compose: prod A 14.8 ms/step 241.4 tok/s (4.11., runs/prod-500k-1104.log:3), 14.99-15.22
#     ms 234.8-238.4 tok/s (5.-6.11., QL:635-665); prod B 15.0 ms
#   c4 / c8 / pf / turns: stock396 (5.11., prod B's cards c3+03, exact wire, MML 262144, no yarn, HP 16384, MAXSEQS 16, MTP 3,
#     MNBT 2048; QL:629-632): c4 381.7 tok/s 25.6 ms, c8 523.6 tok/s 34.8 ms, pf 4923.7 tok/s TTFT 6.52 s kept 0.164, turns
#     cold 57223 tok 9.24 s hit 0.71 s -- other cards (A's 83+86 prefill ~28 % slower), other wire, other MML
#   c16: no TP2 mb.py c16 exists; nearest = decbench on live prod B (5.11.; pr396/runs/1105-0422/bench-table.txt) c16 916.1
#     tok/s 44.42 ms/step (cached 25k prompt, temperature 1 -- another method)
# Time: restart ~5 min (113.6 GiB + a 47.7 GiB private n-gram copy), gate ~1.5, i1 0.5, probe 0.5, c4 / c8 / c16 3.5 each,
# pf 1.5, turns 1.5, i2 0.5 -> ~20 min an arm (+ ~10 min BetterBench with FN_BB=1): FNSET A (3 arms) ~90 min, B (2 arms +
# checks) ~65 min.
# Bounded: compose up/down of this item's own project, health wait capped, every client under timeout, curl -m, the engine
# log kept on a failed / slow start. Never touches radiance-prod-* or /dev/shm/ngram-*.
# DRYRUN=1 FNSET=A bash 84-fn-tuned.sh : prints every command, runs and writes nothing.
set -u
DRY=${DRYRUN:-0}
H=$(cd "$(dirname "$0")" && pwd)
FNSET=${FNSET:-A}
C=$HOME/radiance-1.15.1/compose; R=$HOME/gpu/ds41/exl3-test/runs; O=$R/fn; L=$R/ds41-exl3.log; D=$HOME/gpu/ds41; Q=$HOME/gpu/ds41/runs/q
for Y in "$H/fn-tp2.yaml" "$H/../fn-tp2.yaml" "$HOME/gpu/ds41/exl3-test/queue-ar/fn-tp2.yaml"; do [ -e "$Y" ] && break; done
CN=radiance-fntest; URL=http://127.0.0.1:8090; export DS41_URL=$URL
T3=radiance-local:ds41-exl3-117-390664b3-fa0e9580-tuned3-hd4a11cab
case $FNSET in B) export FNIMAGE=${FN_IMAGE:-$T3-fnx2} ;; *) export FNIMAGE=${FN_IMAGE:-$T3} ;; esac   # A: the base qwen4exp plugin; B: the rebuilt one (EXL3 path) + the exl3-key plugin
LIM=${FN_GATE_MS:-21}
case $FNSET in
  A) export FNMODELDIR=$HOME/models/flashnext-iq4r FNMODEL=qwen3.8-next-flash-fp8-iq4r-moe.rad; ARMS=${ARMS:-"stock heat nomix"} ;;
  B) export FNMODELDIR=$HOME/models/flashnext-exl3 FNMODEL=qwen3.8-flash-next-exl3.rad; ARMS=${ARMS:-"stock heat"} ;;
  *) echo "FNSET is A or B"; exit 1 ;;
esac
WREF=$Q/guard/dp2w6-A-warm-q1.txt
# BetterBench (8.11., FN_BB=1, the default): after each arm's phases, BetterBench 0.6.1 (GGZ14/BetterBench f72bb9011a83, corpus
# 1.0) through tools/bb_run.py (~/gpu/ext/betterbench + numpy from ~/gpu/ext/bb-deps, python3 -I, --no-update-check):
# --quick (5 passes a category after 1 warm-up, temperature 0.7), its prefill sweep (2k-64k, 4 runs) and concurrency
# 1/2/4/8/16 (32 requests a level), thinking off ("reasoning_effort": "none" on every request, as the Radiance docs ran it;
# checked: no answer may carry reasoning text). --out $O/bb-<tag>/results.json, notes image/tp/cards/model/arm. ~10 min an arm.
FN_BB=${FN_BB:-1}
BBROOT=${BB_ROOT:-$HOME/gpu/ext/betterbench}; BBDEPS=${BB_DEPS:-$HOME/gpu/ext/bb-deps}
BBRUN=; for f in "$H/tools/bb_run.py" "$H/../tools/bb_run.py" "$HOME/gpu/ds41/exl3-test/queue-ar/tools/bb_run.py"; do [ -e "$f" ] && { BBRUN=$f; break; }; done
log() { if [ "$DRY" = 1 ]; then echo "[dry, not logged] $(date '+%F %T') $*"; else echo "$(date '+%F %T') $*" | tee -a $L; fi; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
compose() { docker compose --project-directory $C -f $Y "$@"; }
down() { run "compose down > /dev/null 2>&1; docker rm -f $CN > /dev/null 2>&1"; }
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
restart() {   # restart TAG
  local tag=$1 ok=0 t0 t1 i S=$O/start-$1.log
  [ "$DRY" = 1 ] && { echo "+ FNIMAGE=$FNIMAGE FNMODEL=$FNMODEL compose -f $Y up -d; health <= 1200 s; docker logs > $S"; return 0; }
  down; t0=$(date +%s)
  compose up -d > $O/up-$tag.txt 2>&1
  for i in $(seq 1 240); do
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' $URL/health)" = 200 ] && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Status}}' $CN 2>/dev/null)" != running ] && break
    sleep 5
  done
  t1=$(date +%s); docker logs $CN > $S 2>&1
  [ $ok = 1 ] || { log "RESULT fn restart-$tag: VIRHE not healthy after $((t1-t0)) s: $(grep -E '^E ' $S | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  log "RESULT fn restart-$tag: healthy in $((t1-t0)) s (IMAGE=$FNIMAGE MODEL=$FNMODEL MAXSEQS=${MAXSEQS:-16} HP=${HP:-16384}); $(grep -m1 -E '^I kv: ' $S | cut -c3-200); VRAM experts $(grep -m1 -E '^ +experts +[0-9.]+ GiB' $S | awk '{print $2, $3, $4, $5}'), KV $(grep -m1 -E '^ +kv cache +[0-9.]+ GiB' $S | awk '{print $3, $4}'); $(grep -m1 -oE 'pinned by this process [0-9.]* GiB' $S); E lines $(grep -cE '^E ' $S)"
}
check() {   # check TAG N -> ms/step of prodfc.py (999 when it fails)
  local tag=$1 n=$2
  [ "$DRY" = 1 ] && { echo "+ warm-up (correct --only-q, prefill 16384) + prodfc.py $URL 384 -> $O/guard/$tag-fc$n.txt" >&2; echo 0; return; }
  mkdir -p $O/guard
  (cd $D && timeout 600 python3 ds41_tests.py correct --only-q --effort none --outdir $O/guard > $O/guard/$tag-warm-q.txt 2>&1)
  (cd $D && timeout 600 python3 ds41_tests.py prefill --depths 16384 --cpt 3.6 --calibrate 0 > $O/guard/$tag-warm-pf.txt 2>&1)
  timeout 300 python3 $Q/prodfc.py $URL 384 > $O/guard/$tag-fc$n.txt 2>&1
  grep '^FC ' $O/guard/$tag-fc$n.txt | python3 -c 'import json,sys; print(json.loads(sys.stdin.read()[3:])["ms_per_step"])' 2>/dev/null || echo 999
}
gate() {   # gate TAG: 0 fast / 1 not healthy / 2 slow twice
  local tag=$1 ms ms1 st=fast
  restart $tag || return 1
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && return 0
  if awk -v m=$ms -v g=$LIM 'BEGIN{exit !(m > g)}'; then
    keeplog $tag-try1; ms1=$ms
    restart $tag || return 1; ms=$(check $tag 2); st="fast-after-retry (first $ms1)"
    awk -v m=$ms -v g=$LIM 'BEGIN{exit !(m > g)}' && { st="SLOW-after-retry (first $ms1)"; keeplog $tag-try2; }
  fi
  echo "$st $ms" > $O/guard-$tag.txt
  log "RESULT fn guard $tag: $st, 1-user ${ms} ms/step (limit $LIM; production A 14.8-15.22 ms, B 15.0)"
  case $st in SLOW*) return 2 ;; esac
}
probe() { run "(cd $D && timeout 900 python3 stepprobe.py essay,code 512 > $O/$1-probe.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT fn $1 probe (1 user, greedy, thinking off, 512 tok): $(python3 $D/sweepfmt.py probe $O/$1-probe.txt 2>&1 | cut -c1-300)"; }
mbp() { run "(cd $D && timeout 1200 python3 mb.py --tag fn-$1 --phases $2 > $O/$1-$2.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT fn $1 mb $2: $(grep '^MB ' $O/$1-$2.txt 2>/dev/null | cut -c4- | cut -c1-600 || tail -1 $O/$1-$2.txt)"; }
copyt() { run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 120000 --chars 2400 --max-tokens 900 --runs 1 --offset $2 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT fn $1 copy after a cold prompt: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); u=d.get("usage") or {}; print("prompt %s tok, cached %s, TTFT %s s, exact %s" % (u.get("prompt_tokens"), d.get("cached"), d.get("ttft"), d.get("exact")))' 2>&1 | cut -c1-200)"; }
warmcmp() {   # warmcmp FILE_A FILE_B LABEL
  [ "$DRY" = 1 ] && { echo "+ warm-up answers $1 vs $2"; return 0; }
  [ -s $1 ] && [ -s $2 ] || { log "RESULT fn warm-up $3: not compared (missing $1 or $2)"; return 0; }
  log "RESULT fn warm-up $3: $(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if t in a and ((a[t].get("content") or "") != (b[t].get("content") or "") or a[t].get("finish") != b[t].get("finish"))]
n = len([t for t in b if t in a])
print(("IDENTICAL: %d answers" % n) if n and not bad else ("DIFFERENT: " + ", ".join(bad) if n else "no common answers"))' $1 $2 2>&1 | cut -c1-300)"
}
cmparm() {   # cmparm A B: arm B against arm A
  [ "$DRY" = 1 ] && { echo "+ RESULT fn $FNSET $2 vs $1"; return 0; }
  log "RESULT fn $FNSET $2 vs $1: $(python3 - $O fn$FNSET-$1 fn$FNSET-$2 <<'PY' 2>&1 | cut -c1-1500
import json, re, sys
O, a, b = sys.argv[1:4]
def mb(tag, p):
    try:
        for l in open(f"{O}/{tag}-{p}.txt"):
            if l.startswith("MB "): return json.loads(l[3:])
    except OSError: pass
    return {}
def gd(tag):
    try: return float(open(f"{O}/guard-{tag}.txt").read().split()[-1])
    except Exception: return None
def pr(tag):
    try: t = open(f"{O}/{tag}-probe.txt").read()
    except OSError: return {}
    out = {}
    for name in ("essay", "code"):
        m = re.search(name + r".*?([\d.]+) ms/step", t)
        if m: out[name] = float(m.group(1))
    return out
pa, pb = pr(a), pr(b)
rows = [("guard ms", gd(a), gd(b)), ("essay ms", pa.get("essay"), pb.get("essay")), ("code ms", pa.get("code"), pb.get("code"))]
for c in ("c4", "c8", "c16"):
    rows += [(f"{c} tok/s", mb(a, c).get("agg_dec_tps"), mb(b, c).get("agg_dec_tps")), (f"{c} ms", mb(a, c).get("ms_decode_step"), mb(b, c).get("ms_decode_step"))]
rows += [("pf tok/s", mb(a, "pf").get("prefill_tps"), mb(b, "pf").get("prefill_tps")), ("pf TTFT", mb(a, "pf").get("prefill_ttft"), mb(b, "pf").get("prefill_ttft")),
         ("pf kept", mb(a, "pf").get("decode_kept"), mb(b, "pf").get("decode_kept")),
         ("turns cold s", mb(a, "turns").get("cold_ttft"), mb(b, "turns").get("cold_ttft")), ("turns hit s", mb(a, "turns").get("ttft_hit_mean"), mb(b, "turns").get("ttft_hit_mean"))]
out = []
for n, x, y in rows:
    out.append(f"{n} {x} -> {y} ({(y / x - 1) * 100:+.1f} %)" if x and y else f"{n} {x} -> {y}")
print("; ".join(out))
PY
)"
}
bbph() {   # bbph TAG NOTE...: BetterBench --quick + prefill sweep + concurrency 1/2/4/8/16, thinking off
  local tag=$1 nt="" n; shift
  for n in "$@"; do nt="$nt --note $n"; done
  run "(BETTERBENCH_HOME=$O/bbhome timeout 1800 python3 -I $BBRUN --bb-root $BBROOT --deps $BBDEPS -- run --endpoint $URL/v1 --model qwen3.8-flash-next --quick --config $O/bb-config.json --name $tag --out $O/bb-$tag/results.json$nt > $O/$tag-bb.txt 2>&1)"
  [ "$DRY" = 1 ] && return 0
  log "RESULT fn $tag BetterBench: $(python3 -I $BBRUN --summary $O/bb-$tag/results.json 2>&1 | cut -c1-1400)"
  python3 -I $BBRUN --check $O/bb-$tag/results.json > /dev/null 2>&1 || log "RESULT fn $tag BetterBench: VIRHE single-stream answers carried reasoning text -- thinking was not off ($O/bb-$tag/results.json)"
}
arm() {   # arm NAME
  local a=$1 tag=fn$FNSET-$1 rc
  unset RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES \
        RADIANCE_PLACE_TURNOVER RADIANCE_STAGE_MIXED RADIANCE_STAGE_SPAN RADIANCE_ARENA_LEVEL_MAX RADIANCE_ARENA_HOLD_MS
  case $a in
    stock)  ;;
    heat)   export RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16 ;;
    nomix)  export RADIANCE_STAGE_MIXED=0 ;;
    heatnm) export RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16 RADIANCE_STAGE_MIXED=0 ;;
    *)      log "RESULT fn: VIRHE unknown arm '$a'"; return 1 ;;
  esac
  export MAXSEQS=16   # c16 needs 16 (= production)
  log "RESULT fn $tag: start IMAGE=$FNIMAGE MODEL=$FNMODELDIR/$FNMODEL env: $(env | grep -E '^RADIANCE_(HEAT|PLACE|STAGE|ARENA)' | sort | tr '\n' ' ')"
  gate $tag; rc=$?
  [ $rc = 1 ] && { keeplog $tag-fail; log "RESULT fn $tag: VIRHE not healthy -- $O/start-$tag.log, $O/engine-$tag-fail.log"; return 1; }
  [ $rc = 2 ] && { log "RESULT fn $tag: VIRHE slow state after one restart -> arm skipped (engine logs $O/engine-$tag-try1.log / -try2.log)"; down; return 1; }
  [ "$DRY" = 1 ] || log "RESULT fn $tag modes: $(docker logs $CN 2>&1 | grep -m4 -E 'activation arena|prefill staging: a|heat engine: (dual|the decode)|prefill staging: a pass' | cut -c1-260 | tr '\n' '|')"
  copyt $tag-i1 0
  probe $tag
  for p in c4 c8 c16 pf turns; do [ -e $R/sweep.stop ] && break; mbp $tag $p; done
  copyt $tag-i2 7000
  [ "$FN_BB" = 1 ] && [ ! -e $R/sweep.stop ] && bbph $tag image=$FNIMAGE engine=1.17-tuned3 tp=2 cards=83+86 placement=expert_tiered hp=${HP:-16384} mml=${MML:-524288}-yarn2 mtp=3 wire=wht6 model=$FNMODEL arm=$a
  run "docker logs $CN > $O/$tag-engine.log 2>&1"
  down
  log "RESULT fn $tag: done"
}
log "RESULT fn 84-fn-tuned: start FNSET $FNSET ARMS [$ARMS] IMAGE=$FNIMAGE ($(docker image inspect -f '{{.Id}}' $FNIMAGE 2>/dev/null | cut -c8-19)) model $FNMODELDIR/$FNMODEL compose $Y"
if [ "$DRY" != 1 ]; then
  [ -e "$Y" ] || { log "RESULT fn: VIRHE fn-tp2.yaml not found"; exit 1; }
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT fn: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  docker image inspect $FNIMAGE > /dev/null 2>&1 || { log "RESULT fn: VIRHE image $FNIMAGE missing"; exit 1; }
  [ -f $FNMODELDIR/$FNMODEL ] || { log "RESULT fn: VIRHE $FNMODELDIR/$FNMODEL missing"; exit 1; }
  ls $FNMODELDIR/$FNMODEL.sha*ok > /dev/null 2>&1 || { log "RESULT fn: VIRHE $FNMODELDIR/$FNMODEL has no .shaok / .sha256ok (copy not verified)"; exit 1; }
  [ -x $Q/prodfc.py ] || [ -e $Q/prodfc.py ] || { log "RESULT fn: VIRHE $Q/prodfc.py missing"; exit 1; }
  if [ "$FN_BB" = 1 ]; then
    [ -e "$BBRUN" ] && [ -d "$BBROOT/betterbench" ] && [ -d "$BBDEPS/numpy" ] || { log "RESULT fn: VIRHE BetterBench not ready (bb_run.py '$BBRUN', $BBROOT/betterbench, $BBDEPS/numpy); FN_BB=0 runs without it"; exit 1; }
    mkdir -p $O; printf '%s\n' '{"concurrency_levels": [1, 2, 4, 8, 16], "concurrency_requests": 32, "prefill_runs": 4, "prefill_warmup": 1}' > $O/bb-config.json
    log "RESULT fn: BetterBench $(git -C $BBROOT rev-parse --short=12 HEAD 2>/dev/null || cat $BBROOT/COMMIT 2>/dev/null) ($(python3 -I -c 'import sys; sys.path.insert(0, sys.argv[1]); import betterbench as b; print(b.__version__, "corpus", b.CORPUS_VERSION)' $BBROOT 2>&1)), config $(cat $O/bb-config.json), thinking off, --quick, temperature 0.7"
  fi
  mkdir -p $O $O/guard
else
  echo "+ mkdir -p $O $O/guard"
fi
if [ $FNSET = B ]; then   # does the image take the EXL3 container at all
  run "timeout -k 15 300 docker run --rm --init --name fn-radinfo-$$ --network none -v $FNMODELDIR:/m:ro --entrypoint /opt/radiance/bin/rad-info $FNIMAGE /m/$FNMODEL > $O/radinfo-B.txt 2>&1; docker rm -f fn-radinfo-$$ > /dev/null 2>&1"
  run "timeout -k 15 120 docker run --rm --init --name fn-plugins-$$ --network none --entrypoint /opt/radiance/bin/rad-info $FNIMAGE --plugins > $O/plugins-B.txt 2>&1; docker rm -f fn-plugins-$$ > /dev/null 2>&1"
  if [ "$DRY" != 1 ]; then
    log "RESULT fn B rad-info: $(grep -iE 'arch|encoding|exl3|expert' $O/radinfo-B.txt | head -6 | tr -s ' ' | tr '\n' '|' | cut -c1-600) | plugins: $(grep -ciE 'refus|error|disagree' $O/plugins-B.txt) refusals"
    grep -qi exl3 $O/radinfo-B.txt || { log "RESULT fn B: VIRHE rad-info shows no exl3 encoding in $FNMODEL -> stopped (see $O/radinfo-B.txt)"; exit 1; }
    grep -qE 'qwen4exp @ exl3' $O/plugins-B.txt || { log "RESULT fn B: VIRHE no plugin of $FNIMAGE claims 'qwen4exp @ exl3' (rad-info --plugins: $(grep -i qwen4exp $O/plugins-B.txt | tr -s ' ' | tr '\n' '|' | cut -c1-200)) -> stopped"; exit 1; }
    log "RESULT fn B plugins: $(grep -E 'qwen4exp @' $O/plugins-B.txt | tr -s ' ' | tr '\n' '|' | cut -c1-200)"
  fi
fi
done_arms=""; first=""
for a in $ARMS; do
  [ -e $R/sweep.stop ] && break
  arm $a && { done_arms="$done_arms $a"; [ -z "$first" ] && first=$a; }
  if [ -n "$first" ] && [ "$a" != "$first" ] && [[ " $done_arms " == *" $a "* ]]; then
    warmcmp $O/guard/fn$FNSET-$first-warm-q.txt $O/guard/fn$FNSET-$a-warm-q.txt "$FNSET $a vs $first (placement only: must be IDENTICAL)"
    cmparm $first $a
  fi
  [ "$a" = "$first" ] && warmcmp $WREF $O/guard/fn$FNSET-$a-warm-q.txt "$FNSET $a vs 1.15.1 wht6 (q/guard/dp2w6-A-warm-q1.txt; another engine: information only)"
done
down
log "RESULT fn 84-fn-tuned: done FNSET $FNSET --$done_arms | refs: prod A gate 14.8-15.22 ms; stock396 (prod B cards, exact, MML 262144) c4 381.7 tok/s 25.6 ms, c8 523.6 34.8 ms, pf 4923.7 tok/s TTFT 6.52 s kept 0.164, turns cold 9.24 s hit 0.71 s; c16 none at TP2 (decbench prod B c16 916.1 tok/s 44.42 ms, another method)"
