#!/bin/bash
# dsm-pipeprof.sh (10.11.2026, queue-ar/dsmain, ds41-pipe agent) -- NOT a queue item; the body of 57-p4zyzzzzz-ds-ka-pipeprof.sh.
# WHY THE PIPELINED PREFILL LOSES: a per-op prefill profile of RADIANCE_DS41_PIPE=1 (B) against the serial step (A), ONE
# image (radiance-local:ds41-pipe-fd215624-5c13e555-l, XPP_IMAGE), two starts, both with --profile-ops. PROFILED: not a
# speed (event pairs around every launch, one sync a step); ratios B / A per op role are the reading.
#   Measured before (f-pipe / fz-span4, unprofiled, exact): fresh prefill B / A 16k -10.6 %, 64k -13.0 %, 136k -12.9 %;
#   span 4 SP / S (pieces of 4096 = today's MoE pass) -4.8 / -3.6 / -3.7 %, so ~4 % is the overlap itself.
# THE PROFILER DOES NOT SERIALISE THE LANES (core/runtime/issue.cpp + ctx.cpp profile_collect, core/device/aql.cpp
#   a_event_record): each launch is bracketed by two card-local events recorded on ITS OWN lane's queue (ProfSlot.lane), an
#   event record is one barrier packet on that queue only, and the one host sync a step comes after the runtime has
#   joined lane 1 into lane 0. So lane 1's all-reduces still run beside lane 0's kernels, and a lane-0 kernel's event pair
#   measures it with the AR beside it. ('wait us' = the op's own lane idle before it: for lane 0 the joins on lane 1 and the
#   stager's copy waits before a layer's first expert op.) No AQL stamps needed.
# PER ARM (A then B): t3 env + --profile-ops, RADIANCE_PROFILE_EVERY 8 (XPP_EVERY) -> restart -> warm-up + 1-user fastcheck
#   (lib's check; PROFILED, not gated) -> warm-up answers vs stock -> the pipelining note (B only) -> one fresh ~64k prefill
#   answered with ONE token (pf1.py: no decode step follows it; 17 steps, so the window's first dump falls inside it) ->
#   mark + /stats -> two more fresh ~64k prefills, one token each (~34 lone prefill steps) -> /stats + mark -> docker stop
#   (the shutdown table closes the window) -> ds_opprof.py on the window (per rank by class, lane waits, wall).
#   The window = each rank's last profile dump at or before the first mark to its last at or before the second, so it holds
#   only lone prefill steps (up to 7 from the end of the first prompt, none of the last ones); the /stats pair covers the
#   two measured prompts exactly. Steps of a ~64k prompt: one of 2048 (a prompt's first interval, never spanned), fifteen
#   of 4096 (B: 2048 + 2048 pieces), a short tail -- the same mix in both arms.
# THEN ds_pipeprof.py B vs A, ms a prefill step a rank, op instance by op instance grouped by role: the ops B runs BESIDE a
#   lane-1 all-reduce (o_a / o_b, mix_a, router selection, sort, EXL3 gate_up / down, shared expert, MoE gather, the next
#   post + collapse + norm) vs CONTROL ops nothing runs beside (attention core, indexer, q / kv / compressor GEMMs, Engram,
#   mix_f, the ffn post + collapse + norm, the router's GEMM: B / A should be 1.00); the all-reduce kernel time serial (A) vs
#   lane form (B) with each instance's floor (fastest rank) and skew; the gathers B issues alone on lane 1 (router x2,
#   indexer); the extra whole-step moe_scatter; lane-wait growth by the role that waited; /stats: wall ms a step, staged GB
#   and copies a step, h2d GB a step (the prefill stager's expert copies share the inbound link with the all-reduce, and a
#   collective holds the link for its whole run -- rad_link_hold -- so a long lane-1 AR pauses the copies).
# ~18 min (each arm ~4 min start, ~3 min warm-up, ~2 min of prompts, ~1 min stop + parse). No mb / decode phases, no cc_sim.
# No power / host / fan settings; production compose files read only (--project-directory). Not dsm-pipe.sh / dsm-span4.sh.
# DRYRUN=1 bash dsm-pipeprof.sh : prints every command, runs and writes nothing.
set -u
HD="$(cd "$(dirname "$0")" && pwd)"
for LIB29 in "$HOME/gpu/ds41/exl3-test/queue-29/lib-29.sh" "$HD/../../queue-29/lib-29.sh"; do [ -e "$LIB29" ] && break; done
[ -e "$LIB29" ] && . "$LIB29" || { echo "lib-29.sh not found"; exit 1; }
QA=$HOME/gpu/ds41/exl3-test/queue-ar
Y=$HD/ds41-pipe-tp4.yaml                        # = ds41-dsm-tp4.yaml + RADIANCE_DS41_PIPE (read only here)
TL=$QA/tools
[ -e $TL/heatsoak.sh ] && . $TL/heatsoak.sh   # thermnow / thermwin (read-only)
O=$R/xpipeprof; export STATSDIR=$O/stats
export IMAGE=${XPP_IMAGE:-radiance-local:ds41-pipe-fd215624-5c13e555-l}
M225=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
WARM_REF_225=$R/x117/guard/x117-best524-warm-q.txt
OPP=$HD/ds_opprof.py; PPP=$HD/ds_pipeprof.py; PF1=$HD/pf1.py
EVERY=${XPP_EVERY:-8}; DEPTH=${XPP_DEPTH:-65536}
SWITCHES="RADIANCE_DS41_INDEX RADIANCE_DS41_KEEP_MIN_LIVE RADIANCE_DS41_SPLIT_MIN_LIVE RADIANCE_DS41_IDXKEY RADIANCE_DS41_LATENT
  RADIANCE_DS41_CTAIL RADIANCE_EXL3_SPLITK RADIANCE_HEAT_PREFILL RADIANCE_HEAT_PREFILL_DECAY RADIANCE_HEAT_PREFILL_GAIN
  RADIANCE_HEAT_DECODE_PIN RADIANCE_HEAT_MOVES RADIANCE_HEAT_CREDIT_CAP RADIANCE_PLACE_TURNOVER RADIANCE_DS41_LEVELS
  RADIANCE_ARENA_LEVEL_MAX RADIANCE_STAGE_MIXED RADIANCE_ARENA_HOLD_MS RADIANCE_STAGE_SPAN RADIANCE_DS41_ARGATHER
  RADIANCE_DS41_ROUTER_FUSE R4D_DS41_MIX_FUSE R4D_DS41_QLDS RADIANCE_DS41_AR_PCN RADIANCE_AQL_ACQ RADIANCE_DS41_DSPARK_CAND
  RADIANCE_DS41_PIPE"
# ---------------------------------------------------------------- helpers (dsm-pipe.sh's, 10.11., copied; log tag xpipeprof)
t3env() {   # 105's t3 arm, every other switch unset (RADIANCE_DS41_PIPE included)
  unset $SWITCHES
  export RADIANCE_DS41_IDXKEY=mxfp4 RADIANCE_HEAT_PREFILL_DECAY=0.7 RADIANCE_HEAT_DECODE_PIN=0.8 RADIANCE_HEAT_MOVES=16
  export RADIANCE_DS41_LEVELS=1 RADIANCE_ARENA_LEVEL_MAX=1024 RADIANCE_STAGE_SPAN=0 \
         RADIANCE_DS41_ARGATHER=on RADIANCE_DS41_ROUTER_FUSE=0 R4D_DS41_MIX_FUSE=0 R4D_DS41_QLDS=0 RADIANCE_DS41_AR_PCN=0
}
modelvars() {   # modelvars 225 -> MODEL, HP29, WARMREF; the serve line (lib-29 best_vars + MNBT 4160 + --live)
  case $1 in
    225) export MODEL=$M225; HP29=19456; WARMREF=$WARM_REF_225 ;;
    *)   return 1 ;;
  esac
  best_vars; export MNBT=4160 XARGS="$XARGS --live"
}
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
trap 'down' EXIT
trap 'exit 143' TERM INT
nl() { [ "$DRY" = 1 ] && echo L || docker logs $CN 2>&1 | wc -l; }
thermline() {   # thermline T0 T1: the window's full-power rows (heatsoak.sh thermwin) or the cards now
  type thermwin > /dev/null 2>&1 && echo "thermal $(thermwin $1 $2) | now $(thermnow)" || echo "thermal n/a"
}
identw() {   # identw TAG REFFILE: warm-up answers vs the stock reference
  [ "$DRY" = 1 ] && { echo "+ RESULT xpipeprof $1 warm-up identity vs $2"; return 0; }
  [ -s $2 ] || { log "RESULT xpipeprof $1 warm-up identity: no reference $2"; return 0; }
  local r; r=$(python3 -c 'import json,sys
def rd(f):
    try: return {d.get("test"): d for d in (json.loads(l[2:]) for l in open(f) if l.startswith("J "))}
    except OSError: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
bad = [t for t in b if (a.get(t, {}).get("content") or "") != (b[t].get("content") or "") or a.get(t, {}).get("finish") != b[t].get("finish")]
print(("IDENTICAL: %d answers" % len(b)) if b and not bad else ("DIFFERENT: " + ", ".join(bad) if b else "no reference answers"))' $O/guard/$1-warm-q.txt $2 2>&1 | cut -c1-300)
  case $r in IDENTICAL*) log "RESULT xpipeprof $1 warm-up answers vs stock ($(basename $2)): $r (profiled run)" ;;
    *) log "RESULT xpipeprof $1 warm-up answers vs stock ($(basename $2)): VIRHE $r -- NOT EXACT" ;; esac
}
pf1() {   # pf1 TAG N OFFSET: N fresh ~DEPTH prompts, one token each
  local t0=$(date +%s)
  run "(cd $D && timeout 1800 python3 $PF1 $1 $DEPTH $2 $3 > $O/$1-pf1.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xpipeprof $1 fresh ~$DEPTH prefills x$2, one token each (PROFILED): $(grep '^J ' $O/$1-pf1.txt | python3 -c 'import json,sys
print("; ".join("%s tok cached %s %s s %s tok/s %s" % (d.get("prompt_tokens"), d.get("cached"), d.get("s"), d.get("prefill_tps"), d.get("finish")) for d in (json.loads(l[2:]) for l in sys.stdin)))' 2>&1 | cut -c1-400) | $(thermline $t0 $(date +%s))"
}
# ---------------------------------------------------------------- the arms
arm() {   # arm A|B
  local a=$1 tag=pp-$1 E=$O/pp-$1-engine.log l0 l1 ms PL
  t3env; modelvars 225 || return 1
  [ "$a" = B ] && export RADIANCE_DS41_PIPE=1
  export PROFEVERY=$EVERY XARGS="$XARGS --profile-ops" STATSDIR=$O/stats-$a
  mkd $STATSDIR
  log "RESULT xpipeprof $tag: start IMAGE=$IMAGE MODEL=$MODEL --profile-ops RADIANCE_PROFILE_EVERY=$PROFEVERY (PROFILED: not a speed) $(desc) env: $(env | grep -E '^(RADIANCE_|R4D_)' | sort | tr '\n' ' ')"
  if [ "$DRY" != 1 ]; then
    [ -f $MODELDIR/$MODEL ] && [ -s $MODELDIR/$MODEL.sha256ok ] || { log "RESULT xpipeprof $tag: VIRHE $MODELDIR/$MODEL missing or not verified"; return 1; }
  fi
  restart $tag || { keeplog $tag-fail; down; log "RESULT xpipeprof $tag: VIRHE not healthy -- no profile"; return 1; }
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] || log "RESULT xpipeprof $tag warm-up + 1-user fastcheck (PROFILED, not gated): $ms ms/step"
  identw $tag $WARMREF
  if [ "$DRY" != 1 ]; then
    PL=$(docker logs $CN 2>&1 | grep -E 'pipelined prefill' | cut -c1-200 | tr '\n' '|')
    case $a in
      A) [ -z "$PL" ] || log "RESULT xpipeprof $tag: VIRHE a pipelining line in arm A: $PL" ;;
      B) case "$PL" in *"pipelined prefill on"*) ;; *) log "RESULT xpipeprof $tag: VIRHE no 'pipelined prefill on' note in arm B: ${PL:-none}" ;; esac ;;
    esac
  fi
  pf1 $tag-w 1 300                                   # the window's first dump falls inside this prompt
  l0=$(nl); run "curl -s -m 15 $URL/stats > $O/$tag-sta.json 2>/dev/null"
  pf1 $tag 2 700                                     # the measured prompts
  run "sleep 2; curl -s -m 15 $URL/stats > $O/$tag-stb.json 2>/dev/null"; l1=$(nl)
  run "echo '$a $l0 $l1' >> $O/windows.txt"
  # stop first (the shutdown table closes the last window), read the whole log, then remove
  run "docker stop -t 60 $CN > /dev/null 2>&1; docker logs $CN > $E 2>&1"
  if [ "$a" = B ] && [ "$DRY" != 1 ]; then
    local SN; SN=$(grep -m1 'are pipelined:' $E | sed 's/^radiance: //' | cut -c1-300)
    [ -n "$SN" ] && log "RESULT xpipeprof $tag pipelined steps ran: $SN" || log "RESULT xpipeprof $tag: VIRHE no pipelined step ran -- B profiled the serial path"
  fi
  down
  if [ "$DRY" = 1 ]; then echo "+ ds_opprof.py $E --from L0 --to L1 --stats $O/$tag-sta.json $O/$tag-stb.json --json $O/$tag-op.json"
  else
    timeout 900 python3 -I $OPP $E --from $l0 --to $l1 --label $tag-pf64x2 --json $O/$tag-op.json --stats $O/$tag-sta.json $O/$tag-stb.json > $O/$tag-op.txt 2>&1
    log "RESULT xpipeprof $tag two fresh ~$DEPTH prefills (PROFILED, not a speed): $(cut -c1-3000 $O/$tag-op.txt | tr '\n' ' ')"
  fi
  export PROFEVERY= STATSDIR=$O/stats
  log "RESULT xpipeprof $tag: done"
}
# ---------------------------------------------------------------- main
log "RESULT xpipeprof 57-p4zyzzzzz-ds-ka-pipeprof: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19)) arms A (RADIANCE_DS41_PIPE unset) then B (=1), --profile-ops every $EVERY, ~$DEPTH-token prompts, yaml $Y | unprofiled refs (f-pipe): fresh prefill B / A 64k 2371.5 / 2727.1 tok/s (-13.0 %); fz-span4 SP / S 64k -3.6 %"
if [ "$DRY" != 1 ]; then
  docker ps --format '{{.Names}}' | grep -qiE '^radiance' && { log "RESULT xpipeprof: VIRHE a radiance container is running ($(docker ps --format '{{.Names}}' | grep -iE '^radiance' | tr '\n' ' ')) -- the cards are not free"; exit 1; }
  docker image inspect $IMAGE > /dev/null 2>&1 || { log "RESULT xpipeprof: VIRHE image $IMAGE not on gpu02"; exit 1; }
  for v in $SWITCHES; do grep -q "^ *$v:" $Y || { log "RESULT xpipeprof: VIRHE $Y does not pass $v"; exit 1; }; done
  grep -q RADIANCE_PROFILE_EVERY $Y || { log "RESULT xpipeprof: VIRHE $Y lacks RADIANCE_PROFILE_EVERY"; exit 1; }
  for f in $OPP $PPP $PF1 $D/ds41_tests.py; do [ -e $f ] || { log "RESULT xpipeprof: VIRHE $f missing"; exit 1; }; done
else
  echo "+ image $IMAGE on this host: $(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null || echo MISSING)"
  echo "+ yaml $Y passes RADIANCE_DS41_PIPE: $(grep -c '^ *RADIANCE_DS41_PIPE:' $Y 2>/dev/null), RADIANCE_PROFILE_EVERY: $(grep -c 'RADIANCE_PROFILE_EVERY' $Y 2>/dev/null)"
  echo "+ tools: $(ls $OPP $PPP $PF1 $D/ds41_tests.py 2>&1 | tr '\n' ' ')"
fi
mkd $O $O/guard $STATSDIR
run "rm -f $O/windows.txt"
evict
done_arms=""
for a in A B; do stopfile && break; arm $a && done_arms="$done_arms $a"; done
if [ "$DRY" = 1 ]; then echo "+ ds_pipeprof.py --a $O/pp-A-engine.log L0 L1 --b $O/pp-B-engine.log L0 L1 --sa / --sb the /stats pairs --table $O/pp-BvsA.txt --json $O/pp-BvsA.json"
elif [ "$done_arms" = " A B" ]; then
  read -r _ la0 la1 <<< "$(grep '^A ' $O/windows.txt | tail -1)"
  read -r _ lb0 lb1 <<< "$(grep '^B ' $O/windows.txt | tail -1)"
  timeout 900 python3 -I $PPP --a $O/pp-A-engine.log $la0 $la1 --b $O/pp-B-engine.log $lb0 $lb1 \
    --sa $O/pp-A-sta.json $O/pp-A-stb.json --sb $O/pp-B-sta.json $O/pp-B-stb.json --label pp-BvsA \
    --table $O/pp-BvsA.txt --json $O/pp-BvsA.json > $O/pp-BvsA.out 2>&1
  log "RESULT xpipeprof B vs A, the two measured prompts (PROFILED): $(python3 -c 'import json,sys
def tps(f):
    try: v = [json.loads(l[2:]).get("prefill_tps") for l in open(f) if l.startswith("J ")]
    except OSError: return None
    v = [x for x in v if x]
    return sum(v) / len(v) if v else None
a, b = tps(sys.argv[1]), tps(sys.argv[2])
print("mean prefill tok/s A %s -> B %s (%s)" % (a and round(a, 1), b and round(b, 1), "%+.1f %%" % ((b / a - 1) * 100) if a and b else "n/a"))' $O/pp-A-pf1.txt $O/pp-B-pf1.txt 2>&1 | cut -c1-200); unprofiled f-pipe 64k: -13.0 %"
  log "RESULT xpipeprof B vs A, per op role (PROFILED, not a speed; table $O/pp-BvsA.txt): $(cut -c1-4000 $O/pp-BvsA.out | tr '\n' ' ')"
else
  log "RESULT xpipeprof: VIRHE no B-vs-A table -- arms done [$done_arms ]"
fi
run "gzip -f $O/pp-A-engine.log $O/pp-B-engine.log 2>/dev/null || true"
unset $SWITCHES
down
log "RESULT xpipeprof 57-p4zyzzzzz-ds-ka-pipeprof: done -- arms [$done_arms ]"
