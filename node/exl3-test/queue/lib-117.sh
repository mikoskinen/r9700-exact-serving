# lib-117.sh (6.11.2026) -- sourced by the 1.17 EXL3 queue items 45/46/47. NOT a queue item. = lib-pr266.sh (sourced first:
# restart / check / gate / probe / mbp / evict / down, DRYRUN) with the EXL3 1.17 image and model, its own outputs (runs/x117)
# and ds41-117-tp4.yaml (the profiling environment). Needs lib-pr266.sh, ds41-117-tp4.yaml, stampgaps.py and profchunks.py in queue/.
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib-pr266.sh" || { echo "lib-pr266.sh not found"; exit 1; }
Y=$QD/ds41-117-tp4.yaml
O=$R/x117
export IMAGE=${X3IMAGE:-radiance-local:ds41-exl3-117-abf7f9a0-f2f3c3f7}
X3P=${X3P:-radiance-local:ds41-exl3-117-abf7f9a0-f2f3c3f7-profsync-ec7fba36}
export MODEL=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad MODELDIR=$HOME/models/ds41-exl3
export STATSDIR=$O/stats
export PROFEVERY= PROFSYNC= AQLSTAMP=
# 6.11. EXL3 (acc9ae58 + EXL3, same .rad), ~/gpu/ds41/exl3-test/runs/ds41-exl3.log:
REF_X3_BEST="6.11. EXL3 fin-best: 19.36 ms; essay 52.9 / code 51.6 tok/s (18.9 / 19.4 ms); c4 99.0 tok/s 34.5 ms; c8 123.8 54.5 ms; pf 1284.4 tok/s TTFT 24.46 s; turns cold 29.91 hit 2.27; cc_sim u4 TTFT p50/p95 3.88/6.45 s, 21.5 tok/s/agent, 71.9 total, 35.7 ms/decode step"
REF_X3_MORNING="6.11. EXL3 morning flags (NST5 ckint 8192 MAXSEQS 4): rep1 13.7k 1607, 59.7k 1989, 142k 1949; rep2 16.5k 2343, 59k 2221, 137k 1978 tok/s | 3.11. UD-IQ1_S 16k 2844, 59k 2641, 140k 2316"
REF_X3_NST5="6.11. EXL3 NST5 ckint 8192 MAXSEQS 8 RAM tier (sweep base): essay 33.3 tok/s 45.3 ms/round 1.52 tok/round acc 0.103; code 67.2 tok/s 53.2 ms 3.58 acc 0.508; c4 67.9 tok/s 128.1 ms; c8 74.4 230.0 ms"
# The 6.11. fin-best serve line: TP4 exact, expert_tiered, host pool 19456, ngram ram, MNBT 8192, ckint 2048, span 2, IL 8,
# MAXSEQS 8, MML 262144, NST 0, RAM KV tier 84 GiB.
best_vars() { export HOSTPOOL=19456 NGRAM=ram CKINT=2048 MNBT=8192 NST=0 MAXSEQS=8 MML=524288 WIRE=exact \
                     XARGS="$RAMTIER --prefill-span 2 --prefill-interleave 8"; }
# code ms per round from a stepprobe output (999 when missing)
codems() { python3 -c 'import json,sys; p={d.get("prompt","")[:4]: d for d in (json.loads(l[2:]) for l in open(sys.argv[1]) if l.startswith("J "))}; print(p.get("code", {}).get("ms_per_decode_step") or 999)' "$1" 2>/dev/null || echo 999; }
slower() { awk -v a="$1" -v b="$2" -v x="$3" 'BEGIN{exit !(a > b * x)}'; }   # A > B * X ?
follow() { [ "$DRY" = 1 ] && { echo "+ docker logs -f $CN > $1/engine.log &"; return; }
  docker logs -f $CN > $1/engine.log 2>&1 & FOLLOW_PID=$!; sleep 2; }
unfollow() { [ -n "${FOLLOW_PID:-}" ] && { kill $FOLLOW_PID 2>/dev/null; wait $FOLLOW_PID 2>/dev/null; FOLLOW_PID=""; }; return 0; }
mark() { [ "$DRY" = 1 ] && { echo "+ mark $1 -> $2/marks.txt"; return; }; echo "$1 $(wc -l < $2/engine.log) $(date +%s)" >> $2/marks.txt; }
pfline() { python3 -c 'import json,sys; print("; ".join("%s tok %s tok/s TTFT %s s" % (d.get("prompt_tokens"), d.get("prefill_tps"), d.get("ttft")) for d in (json.loads(l[2:]) for l in open(sys.argv[1]) if l.startswith("J "))))' "$1" 2>&1 | cut -c1-400; }
# gate TAG LIMIT_MS: lib-pr266.sh's gate with the EXL3 references in its RESULT line (0 fast / 1 not healthy / 2 slow twice)
gate() {
  local tag=$1 lim=$2 ms ms1 st=fast
  restart $tag || return 1
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && return 0
  if awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}'; then
    ms1=$ms; restart $tag || return 1; ms=$(check $tag 2); st="fast-after-retry (first $ms1)"
    awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}' && st="SLOW-after-retry (first $ms1)"
  fi
  echo "$st $ms" > $O/guard-$tag.txt
  log "RESULT guard x117-$tag: $st, 1-user ${ms} ms/step (limit $lim; EXL3 6.11. NST0 fast 19.2-19.56, slow first tries 33.22 / 34.0; NST5 rounds 45.3-53.2)"
  case $st in SLOW*) return 2 ;; esac
}
