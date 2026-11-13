# lib-29.sh (6.11.2026) -- sourced by the queue-29 items. NOT a queue item. = gpu02 queue/lib-117.sh for the LibertAI EXL3 2.9 bpw
# .rad (LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw: K=3 layers 0-17 + 23-39, K=2 layers 18-22; converted 6.11. on the
# workstation with the 2.25 bpw's converter and baseline, ~/gpu/ds41/exl3/scripts/chain-29b.sh). Same image and serve line as the
# 6.11. x117-best524 row, so every RESULT has a same-image, same-flag 2.25 bpw reference.
#
# Sources lib-pr266.sh (restart / check / probe / mbp / evict / down, DRYRUN): gpu02's queue/ copy when it is there (it may carry
# later fixes), else the snapshot beside this file (6.11. 16:23, sha256 76cd75ec..., restart = docker rm -f + compose up).
# The compose file: queue/ds41-117-tp4.yaml, else the snapshot beside this file (sha256 1c912ea0...).
#
# HOST POOL. The task line says 19456 (the 2.25 bpw's). With the 2.9 bpw .rad that cannot start: a card's routed experts are
# 182.59 GiB / 4 = 45.65 GiB (2.25: 33.98), and expert_tiered keeps on the card what the pool cannot hold (+ ~0.65 GiB slack):
# 45.65 - 19.0 + 0.65 = 27.3 GiB of experts, beside ~8.1 GiB of non-expert weights/workspace (2.25 x117-best524: 32421 MiB used
# - 15.63 experts - 7.96 KV) = 35.4 GiB > the card's 31.9 GiB, before the one-sequence KV floor (524288 tok = 1.64 GiB).
# Default HP29=31488 MiB keeps the 2.25's split (VRAM experts 45.65 - 30.75 + 0.65 = 15.55 GiB vs 15.63, KV ~7.96 GiB a card
# = ~4.9 sessions at 524288). Pinned host memory: 2.25 x117-best524 274.88 GiB + 4 x (31488 - 19456) MiB = ~321.9 GiB of 499.
# The smallest pool that still leaves one 524288 session is ~24700 MiB (HP29=24576 is below it by the same arithmetic).
K29="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -e "$HOME/gpu/ds41/exl3-test/queue/lib-pr266.sh" ]; then LIBPR="$HOME/gpu/ds41/exl3-test/queue/lib-pr266.sh"; else LIBPR="$K29/lib-pr266.sh"; fi
. "$LIBPR" || { echo "lib-pr266.sh not found"; exit 1; }
if [ -e "$QD/ds41-117-tp4.yaml" ]; then Y=$QD/ds41-117-tp4.yaml; else Y=$K29/ds41-117-tp4.yaml; fi
O=$R/x29
export IMAGE=${X3IMAGE:-radiance-local:ds41-exl3-117-abf7f9a0-f2f3c3f7}
export MODEL=deepseek-v4.1-flash-1017-exl3-2.9bpw.rad MODELDIR=$HOME/models/ds41-exl3
export STATSDIR=$O/stats
export PROFEVERY= PROFSYNC= AQLSTAMP=
HP29=${HP29:-31488}
GATE29=${GATE29:-30}
# References, ~/gpu/ds41/exl3-test/runs/ds41-exl3.log (2.25 bpw .rad, sha256 ebf06cb5...):
REF_X117_524="6.11. x117-best524 (2.25 bpw, same image + flags, hp 19456): healthy 232 s, VRAM experts 15.63 GiB, KV 7.96 GiB a card 4.9 sessions, pinned 274.88 GiB; guard 19.71 ms; essay 47.7 / code 50.6 tok/s (20.9 / 19.8 ms); c4 95.0 tok/s 35.8 ms TTFT 1.92; c8 123.4 56.1 ms TTFT 2.44; pf 1249.9 tok/s TTFT 25.13 s; turns cold 30.61 hit 2.31; cc_sim u4 not completed (item killed 13:54)"
REF_X3_MORNING_BEST="6.11. morning fin-best (2.25 bpw, acc9ae58 EXL3 image 9e8ccb64-3684c0b1, MML 262144): guard 19.36 ms; essay 52.9 / code 51.6 tok/s; c4 99.0 tok/s 34.5 ms; c8 123.8 54.5 ms; pf 1284.4 tok/s TTFT 24.46 s; turns cold 29.91 hit 2.27; cc_sim u4 TTFT p50/p95 3.88/6.45 s, 21.5 tok/s/agent, 71.9 total, 35.7 ms/decode step"
# The serve line: 6.11. x117-best524's (TP4 exact, expert_tiered, ngram ram, MNBT 8192, ckint 2048, span 2, IL 8, MAXSEQS 8,
# MML 524288, NST 0 = MTP off, RAM KV tier 84 GiB), host pool HP29.
best_vars() { export HOSTPOOL=$HP29 NGRAM=ram CKINT=2048 MNBT=8192 NST=0 MAXSEQS=8 MML=524288 WIRE=exact \
                     XARGS="$RAMTIER --prefill-span 2 --prefill-interleave 8"; }
# gate TAG LIMIT_MS: restart + warm-up + fastcheck; over the limit one restart and check again (0 fast / 1 not healthy / 2 slow
# twice). LIMIT 30 ms (GATE29), not the 2.25's 21: the 2.9's experts read 34 % more bytes and a card holds 34 % of them in VRAM
# (2.25: 46 %), so its fast state is expected above 21 ms; the slow state (6.11.: 33.22 / 34.0 vs 19.2-19.71 fast, x1.7) of a
# 22-27 ms fast state would be 37-46 ms.
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
  log "RESULT guard $tag: $st, 1-user ${ms} ms/step (limit $lim; 2.25 bpw x117-best524 19.71, 6.11. fast 19.2-19.71, slow 33.22 / 34.0)"
  case $st in SLOW*) return 2 ;; esac
}
# warm-up answers (ds41_tests.py correct --only-q, written by check) beside the 2.25's, one line: a broken .rad shows here first
warmq() { # TAG
  [ "$DRY" = 1 ] && { echo "+ RESULT x29 warm-up answers from $O/guard/$1-warm-q.txt vs $R/x117/guard/x117-best524-warm-q.txt"; return; }
  log "RESULT x29 $1 warm-up answers: $(python3 -c 'import json,sys
def rd(f):
    try: return [json.loads(l[2:]) for l in open(f) if l.startswith("J ")]
    except OSError: return []
a, b = rd(sys.argv[1]), {d.get("test"): d for d in rd(sys.argv[2])}
print(" | ".join("%s %s %s: %r (2.25: %r)" % (d.get("test"), d.get("status"), d.get("finish"), (d.get("content") or "")[:70],
      (b.get(d.get("test"), {}).get("content") or "")[:40]) for d in a) or "NONE")' $O/guard/$1-warm-q.txt $R/x117/guard/x117-best524-warm-q.txt 2>&1 | cut -c1-1200)"
}
