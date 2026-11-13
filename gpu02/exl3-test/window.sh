#!/bin/bash
# window.sh -- the EXL3 DeepSeek GPU window on gpu02, end to end, with production stopped before and restored after.
#   PHASES=... (comma list, default below)  DRYRUN=1 (print the commands, run nothing)  IMAGE=... MODEL=... (or edit env.sh)
#   LITELLM_SWITCHED=1 is REQUIRED for prodstop: LiteLLM must already route the Claude tiers away from gpu02 (workstation step).
#   touch ~/gpu/ds41/exl3-test/runs/STOP  -> skip the remaining test phases and go straight to down + produp.
# Every step appends a RESULT line to runs/ds41-exl3.log (same style as ~/gpu/ds41/runs/ds41-1103.log). Any failure of a test
# phase -> down + produp (production is never left down), unless NORESTORE=1.
#
# Phases (default order; optional ones in brackets):
#   pre       image, model + its .sha256ok marker, disk, free -g, temperatures, no other test server, production healthy
#   prodstop  needs LITELLM_SWITCHED=1; waits until :8000/:8001 are idle 30 s (prodidle.py), then B and A down
#   evict     evict-exl3.py (clean page cache of the DeepSeek files) + MemFree >= MEMFREE_MIN_GB (150) else abort -> produp
#   best      tempmon + 95 C watchdog, EXL3 server with the 3.11. confirm config, gated (<= GUARD_MS 21 ms, one restart), ident save
#   probe     1 user essay/code, 512 tokens greedy (stepprobe.py)
#   correct   facts/code, reasoning efforts, tool calls stream/non-stream + tool turn, literal copy 15k chars x2, copy after 25k context x3
#   mb        mb.py c4, c8, pf (31k prefill beside 4 decoders), turns (56k session + 5 hit turns)
#   [needle]  needles at ~32k / ~128k tokens (qneedle.py, the 3.11. Qwen quality haystacks; 248k left out: > 262144 DS tokens)
#   ccsim     Claude Code simulation 4 then 8 users, 3.11. confirm args (660 s, warm-up 300, effort xhigh,medium)
#   ident     fresh restart (same config, gated) + ident cmp vs best + 1-user probe again (slow-state check)
#   prefill   restart with the 3.11. MORNING flags (NST 5, ckint 8192, MAXSEQS 4, no RAM tier; unguarded as then): 16k/59k/140k x2
#   [onem]    restart MML 1048576 MAXSEQS 4 confirm config (gated): onem.py 262144,524288,745472 + hit turn (~40 min)
#   down      EXL3 server down, tempmon/watchdog stopped
#   produp    evict + MemFree check, ~/gpu/ds41/runs/q/produp.sh (A then B, prodfc gate <= 21 ms, one restart), /health 200 x2
#   compare   compare.py -> runs/compare.md (EXL3 vs 3.11. UD-IQ1_S, % differences)
set -u
cd "$(dirname "$0")"; . ./env.sh
PHASES=${PHASES:-pre,prodstop,evict,best,probe,correct,mb,ccsim,ident,prefill,down,produp,compare}
DRY=${DRYRUN:-0}
log() { echo "$(date '+%F %T') $*" | tee -a $L; }
run() { if [ "$DRY" = 1 ]; then echo "+ $*"; else eval "$@"; fi; }
TEMPCSV=$R/temps-exl3-$(date +%m%d).csv
restored=0
restore() {  # down + produp, once
  [ "${NORESTORE:-0}" = 1 ] && { log "RESULT window: NORESTORE=1 -> production left as it is"; return; }
  [ $restored = 1 ] && return; restored=1
  ph_down; ph_produp
}
fail() { log "RESULT window: VIRHE $* -> server down and production restore"; restore; ph_compare; exit 1; }
stopped() { [ -e $R/STOP ] && { log "RESULT window: STOP file found before $1 -> restore"; restore; ph_compare; exit 0; }; return 0; }

ph_pre() {
  local ok=1 m
  docker image inspect "$IMAGE" > /dev/null 2>&1 || { log "RESULT pre: VIRHE image $IMAGE not on gpu02 (docker load first)"; ok=0; }
  [ -f "$MODELDIR/$MODEL" ] || { log "RESULT pre: VIRHE model $MODELDIR/$MODEL missing"; ok=0; }
  [ -f "$MODELDIR/$MODEL.sha256ok" ] || { log "RESULT pre: VIRHE $MODEL.sha256ok missing (copy not verified)"; ok=0; }
  docker ps --format '{{.Names}}' | grep -E '^radiance-(ds41|q)' && { log "RESULT pre: VIRHE another test server is running"; ok=0; }
  m=$(for d in /sys/class/drm/card*/device/hwmon/hwmon*; do [ -e $d/temp2_input ] && echo $(( $(cat $d/temp2_input)/1000 )); done | sort -n | tail -1)
  log "RESULT pre: image $IMAGE ($(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null | cut -c8-19)), model $MODEL $(stat -c %s "$MODELDIR/$MODEL" 2>/dev/null) B sha $(cut -c1-16 "$MODELDIR/$MODEL.sha256ok" 2>/dev/null); disk free $(df -h / | awk 'NR==2{print $4}'); RAM $(free -g | awk 'NR==2{print "used " $3 " / avail " $7 " GiB"}'); hottest junction ${m:-?} C; prod health $(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8000/health)/$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8001/health)"
  [ $ok = 1 ] || { log "RESULT window: pre-checks failed, nothing stopped"; [ "$DRY" = 1 ] || exit 1; }
}
ph_prodstop() {
  [ "${LITELLM_SWITCHED:-0}" = 1 ] || { log "RESULT prodstop: VIRHE LITELLM_SWITCHED=1 not given (LiteLLM must route the Claude tiers away from gpu02 first); nothing stopped"; exit 1; }
  if [ "$DRY" != 1 ]; then python3 $T/prodidle.py 30 1500 | tee -a $R/prodidle.txt | grep -q '^IDLE' || { log "RESULT prodstop: VIRHE production not idle in 25 min ($(tail -1 $R/prodidle.txt)); nothing stopped"; exit 1; }; fi
  run "(cd $C && docker compose -f flashnext-prod-dp-b.yaml down 2>&1 | tail -1 && docker compose -f flashnext-prod-dp-a.yaml down 2>&1 | tail -1)"
  log "RESULT prodstop: production down ($(tail -1 $R/prodidle.txt 2>/dev/null | cut -c1-80)); running radiance containers: $(docker ps --format '{{.Names}}' | grep -E '^radiance' | tr '\n' ' ')-"
}
memcheck() {  # $1 = phase name
  local f; f=$(awk '/^MemFree:/{printf "%d", $2/1048576}' /proc/meminfo)
  log "RESULT $1: MemFree $f GiB (minimum $MEMFREE_MIN_GB)"
  [ "$DRY" = 1 ] || [ "$f" -ge "$MEMFREE_MIN_GB" ]
}
ph_evict() { run "python3 $T/evict-exl3.py >> $R/evict.txt 2>&1"; log "RESULT evict: $(tail -1 $R/evict.txt 2>/dev/null)"; memcheck evict || fail "MemFree below $MEMFREE_MIN_GB GiB after evict"; }
ph_best() {
  if ! pgrep -f "[t]empmon.sh $TEMPCSV" > /dev/null; then run "nohup $HOME/gpu/bench/tempmon.sh $TEMPCSV > /dev/null 2>&1 &"; fi
  rm -f $R/watchdog.stop; run "nohup $T/watchdog-exl3.sh $TEMPCSV > /dev/null 2>&1 &"
  run "$T/exl3-gate.sh exl3-best" || fail "EXL3 server not healthy (exl3-best)"
  run "python3 $T/ident.py save best >> $R/ident.txt 2>&1"; log "RESULT ident save best: $(tail -1 $R/ident.txt 2>/dev/null)"
}
ph_probe() { run "(cd $D && timeout 900 python3 stepprobe.py essay,code 512 > $R/probe.txt 2>&1)"; log "RESULT probe (1 user, greedy, thinking off, 512 tok) [guard $(cat $R/guard-exl3-best.txt 2>/dev/null)]: $(python3 $D/sweepfmt.py probe $R/probe.txt 2>&1) | 3.11. UD-IQ1_S: essay 62.6 / code 68.7 tok/s, 15.9 / 14.5 ms/step"; }
ph_correct() {
  run "(cd $D && timeout 1800 python3 ds41_tests.py correct --outdir $R > $R/correct.txt 2>&1)"
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --chars 15000 --runs 2 --outdir $R > $R/copy.txt 2>&1)"
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --chars 15000 --prefix-chars 25000 --runs 3 --tag ctx --outdir $R > $R/copy-ctx.txt 2>&1)"
  log "RESULT correct: $(python3 - $R/correct.txt $R/copy.txt $R/copy-ctx.txt <<'PY' 2>&1
import json, sys
J = lambda f: [json.loads(l[2:]) for l in open(f) if l.startswith("J ")]
c = J(sys.argv[1]); t = [x for x in c if x["test"] == "tool"]
q = [x for x in c if x["test"] in ("factual", "factual2", "code")]
e = [x for x in c if x["test"].startswith("effort-")]
cp = [x for x in J(sys.argv[2]) + J(sys.argv[3]) if x["test"] == "copy"]
print("tool calls %d/%d parse; effort %s; answers: %s ;; copies exact %d/%d (%s)" % (sum(x["parse_ok"] for x in t), len(t),
      ", ".join("%s %s %s" % (x["test"][7:], x["status"], (x.get("content") or "")[:12].replace("\n", " ")) for x in e),
      " | ".join((x.get("content") or "")[:80].replace("\n", " ") for x in q), sum(x["exact"] for x in cp), len(cp),
      ", ".join("%s r%s cached %s" % (x.get("tag") or "-", x["run"], x.get("cached")) for x in cp)))
PY
)"
}
ph_mb() {
  for p in c4 c8 pf turns; do stopped mb-$p
    run "(cd $D && timeout 1800 python3 mb.py --tag exl3 --phases $p > $R/mb-$p.txt 2>&1)"
    log "RESULT mb $p [guard $(cat $R/guard-exl3-best.txt 2>/dev/null)]: $(grep '^MB ' $R/mb-$p.txt 2>/dev/null | cut -c4- || tail -1 $R/mb-$p.txt)"
  done
}
ph_needle() {
  run "python3 -c \"import json; d=json.load(open('$Q/qual/needle.json')); d['hays']=d['hays'][:2]; json.dump(d, open('$R/needle.json','w'))\""
  run "timeout 3600 python3 $Q/qual/qneedle.py $URL exl3 $R > $R/needle.txt 2>&1"
  log "RESULT needle (~32k/~128k, 6 codes each): $(grep -c '\"correct\": true' $R/needle.txt 2>/dev/null) correct of $(grep -c '\"correct\"' $R/needle.txt 2>/dev/null) answers; $(grep '\"cold\"' $R/needle.txt 2>/dev/null | python3 -c 'import json,sys; print("; ".join("cold %s tok %s tok/s" % (d.get("prompt_tokens"), d.get("prefill_tps")) for d in (json.loads(l[3:]) for l in sys.stdin)))' 2>/dev/null)"
}
ph_ccsim() { for u in 4 8; do stopped ccsim-u$u; run "$T/ccsim-exl3.sh $u best-s8 > $R/ccsim-u$u.out 2>&1"; done; }
ph_ident() {
  run "$T/exl3-gate.sh exl3-ident" || fail "EXL3 server not healthy (exl3-ident)"
  run "python3 $T/ident.py save ident >> $R/ident.txt 2>&1"; run "python3 $T/ident.py cmp best ident >> $R/ident.txt 2>&1"
  run "(cd $D && timeout 900 python3 stepprobe.py essay,code 512 > $R/probe2.txt 2>&1)"
  log "RESULT ident (temp 0 across a restart): $(tail -1 $R/ident.txt 2>/dev/null); probe again [guard $(cat $R/guard-exl3-ident.txt 2>/dev/null)]: $(python3 $D/sweepfmt.py probe $R/probe2.txt 2>&1)"
}
ph_prefill() {
  run "env MAXSEQS=4 NST=5 CKINT=8192 EXTRA= EXTRA2= GUARD_MS=0 $T/exl3-gate.sh exl3-morning" || fail "EXL3 server not healthy (exl3-morning)"
  run "(cd $D && timeout 1800 python3 ds41_tests.py prefill --reps 2 > $R/prefill.txt 2>&1)"
  log "RESULT prefill (3.11. morning flags: DSpark 5, ckint 8192, MAXSEQS 4, fresh, no cache hits): $(python3 -c 'import json,sys; print("; ".join("%s tok %s tok/s TTFT %s s" % (d["prompt_tokens"], d["prefill_tps"], d["ttft"]) for d in (json.loads(l[2:]) for l in open(sys.argv[1]) if l.startswith("J "))))' $R/prefill.txt 2>&1) | 3.11. UD-IQ1_S: 16k 2844, ~59k 2641, ~140k 2316 tok/s"
}
ph_onem() {
  run "env MML=1048576 MAXSEQS=4 $T/exl3-gate.sh exl3-onem" || fail "EXL3 server not healthy (exl3-onem)"
  run "(cd $D && timeout 7200 python3 onem.py 262144,524288,745472 > $R/onem.txt 2>&1)"
  log "RESULT 1M probe (MML 1048576, MAXSEQS 4, confirm config): $(grep '^OM ' $R/onem.txt 2>/dev/null | cut -c4- | tr '\n' ' ') | 3.11. UD-IQ1_S: 266k 1805, 501k 1287, 644k 1081 tok/s; hit turn 644k 8.01 s"
}
ph_down() {
  run "docker compose --project-directory $C -f $T/ds41-exl3-tp4.yaml down 2>&1 | tail -1"
  touch $R/watchdog.stop; run "pkill -f '[t]empmon.sh $TEMPCSV'"
  run "docker logs radiance-ds41-exl3 > $R/engine-last.log 2>&1 || true"
  log "RESULT down: EXL3 server down; radiance containers: $(docker ps --format '{{.Names}}' | grep -E '^radiance' | tr '\n' ' ')-"
}
ph_produp() {
  run "python3 $T/evict-exl3.py >> $R/evict.txt 2>&1"; log "RESULT produp-evict: $(tail -1 $R/evict.txt 2>/dev/null)"
  memcheck produp-memfree || log "RESULT produp: WARNING MemFree below $MEMFREE_MIN_GB GiB, starting production anyway"
  run "bash $Q/produp.sh > $R/produp.out 2>&1"
  local a b; a=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8000/health); b=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8001/health)
  log "RESULT produp: $(grep 'RESULT produp' $R/produp.out 2>/dev/null | cut -c21- | tr '\n' ' ' | cut -c1-700); health :8000 $a :8001 $b -> NEXT: switch LiteLLM back to gpu02 on the workstation"
}
ph_compare() { run "python3 $T/compare.py $R > $R/compare.md 2>&1"; [ "$DRY" = 1 ] || cat $R/compare.md; }

log "RESULT window: start PHASES=$PHASES IMAGE=$IMAGE MODEL=$MODEL HOSTPOOL=$HOSTPOOL DRYRUN=$DRY"
for p in ${PHASES//,/ }; do
  case $p in
    pre|prodstop|evict|down|produp|compare) ph_$p ;;
    best|probe|correct|mb|needle|ccsim|ident|prefill|onem) stopped $p; ph_$p ;;
    *) log "RESULT window: unknown phase $p (skipped)" ;;
  esac
done
log "RESULT window: done"
