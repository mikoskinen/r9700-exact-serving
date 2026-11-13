# lib-ar.sh (6.11.2026) -- sourced by the queue-ar items (70-). NOT a queue item. = lib-fix.sh (sourced first; it sources
# lib-117.sh and lib-pr266.sh: restart / check / gate / probe / mbp / evict / down / ident / deep / follow / mark, DRYRUN)
# with
#   - X3HEAT: the -idx image (radiance-local:ds41-exl3-117-b4624774-f2f3c3f7-idx: its kernels, deepseek41.so, EXL3 plugin)
#     with bin/radiance from branch ds41-ar (= ds41-idx + the heat engine's prefill share, RADIANCE_HEAT_PREFILL, and
#     RADIANCE_HEAT_CREDIT_CAP). Both unset = the -idx engine's placement policy, bit for bit;
#   - ds41-ar-tp4.yaml: ds41-fix-tp4.yaml plus RADIANCE_HEAT_PREFILL / RADIANCE_HEAT_CREDIT_CAP, passed only when exported;
#   - outputs in runs/xar, RESULT lines in runs/ds41-exl3.log as before.
# Needs in ~/gpu/ds41/exl3-test/queue/: lib-fix.sh, lib-117.sh, lib-pr266.sh (as on gpu02 6.11.), ds41-ar-tp4.yaml, this file.
# Uses ~/gpu/ds41/exl3-test/profsum.py (live mode) and runs/xfix/copy-xfix-idx-run1.txt (the -idx image's copy output).
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib-fix.sh" 2>/dev/null || . $HOME/gpu/ds41/exl3-test/queue/lib-fix.sh || { echo "lib-fix.sh not found"; exit 1; }
Y=$QD/ds41-ar-tp4.yaml; [ -e $Y ] || Y=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ds41-ar-tp4.yaml
O=$R/xar
export STATSDIR=$O/stats
X3HEAT=${X3HEAT:-radiance-local:ds41-exl3-117-058faddc-f2f3c3f7-idx-heat}   # b801fc2ab3d4
unset RADIANCE_HEAT_PREFILL RADIANCE_HEAT_CREDIT_CAP
IDREF=$R/xfix/copy-xfix-idx-run1.txt
# References (no baseline re-runs; gpu02 runs/ds41-exl3.log):
REF_50="50 xfix-idx (-idx image, MML 524288 MAXSEQS 4 NST0 ckint 8192, heat as shipped): guard 19.75 ms; deep 134698 tok TTFT 54.03 s 2493.2 tok/s 28.8 ms/step, 265878 tok TTFT 110.56 s 2404.9 tok/s 30.3 ms/step; probe after ident + cold 16k/135k/322k + deep: essay 22.2 / code 22.4 ms/step"
REF_BEST="50 xfix-idx-best524 (fin-best): probe essay 21.0 / code 19.8 ms/step; c4 95.5 tok/s 35.6 ms"
REF_55="55 x117-ccstats (base 117 image, fin-best MML 524288) cc_sim u4: total gen 79.2 tok/s, gen/agent p50 22.7 tok/s, TTFT p50/p95 3.94/7.0 s, prompt p50 75322, cache 95.0 %, uncached p50 3642"
# the gate with this kit's tag and the heat knobs in its RESULT line (lib-117's limits and slow-state retry)
# keeplog NAME: the server's whole log so far (docker logs works on an exited container until it is removed), so a failed
# or slow start keeps its engine log before the next restart or down removes the container (7.11.: 76's on arm left none)
keeplog() { [ "$DRY" = 1 ] && { echo "+ docker logs $CN > $O/engine-$1.log"; return 0; }; docker logs $CN > $O/engine-$1.log 2>&1 || true; }
gate() {
  local tag=$1 lim=$2 ms ms1 st=fast
  restart $tag || return 1
  ms=$(check $tag 1)
  [ "$DRY" = 1 ] && return 0
  if awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}'; then
    keeplog $tag-try1
    ms1=$ms; restart $tag || return 1; ms=$(check $tag 2); st="fast-after-retry (first $ms1)"
    awk -v m=$ms -v g=$lim 'BEGIN{exit !(m > g)}' && { st="SLOW-after-retry (first $ms1)"; keeplog $tag-try2; }
  fi
  echo "$st $ms" > $O/guard-$tag.txt
  log "RESULT guard xar-$tag: $st, 1-user ${ms} ms/step (limit $lim; 50 -idx fast 19.68-19.75) [IMAGE=$IMAGE RADIANCE_HEAT_PREFILL=${RADIANCE_HEAT_PREFILL-unset} RADIANCE_HEAT_CREDIT_CAP=${RADIANCE_HEAT_CREDIT_CAP-unset}]"
  case $st in SLOW*) return 2 ;; esac
}
# heatmode TAG: the engine's own heat lines (rank 0), so the RESULT says what the server ran with
heatmode() {
  [ "$DRY" = 1 ] && { echo "+ docker logs | grep heat engine > $O/$1-heat.txt"; return; }
  docker logs $CN 2>&1 | grep -E "heat engine" | grep -E "^I\[0\]|^I heat|prefill share" | head -3 > $O/$1-heat.txt
  log "RESULT xar $1 heat: $(tr '\n' '|' < $O/$1-heat.txt | cut -c1-400)"
}
# xprobe TAG WHAT: probe with its RESULT line under xar
xprobe() {
  run "(cd $D && timeout 900 python3 stepprobe.py essay,code 512 > $O/$1-probe.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xar $1 probe $2 (1 user, greedy, thinking off, 512 tok) [guard $(guardtxt ${1%-*})]: $(python3 $D/sweepfmt.py probe $O/$1-probe.txt 2>&1 | cut -c1-300)"
}
# identref TAG: the copy output (ident TAG) byte for byte against the -idx image's (50, same flags, heat as shipped)
identref() {
  [ "$DRY" = 1 ] && { echo "+ cmp $O/copy-$1-run1.txt $IDREF"; return 0; }
  local r
  r=$(python3 -c 'import sys; a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read(); f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b))); print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' $O/copy-$1-run1.txt $IDREF 2>&1)
  log "RESULT xar ident $1 vs 50 xfix-idx: $r; copy: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); print("exact", d.get("exact"), "ratio", d.get("ratio"), "prompt", (d.get("usage") or {}).get("prompt_tokens"), "TTFT", d.get("ttft"))' 2>&1)"
  case $r in IDENTICAL*) return 0 ;; *) return 1 ;; esac
}
# xdeep TAG DEPTHS: onem.py as lib-fix's deep, RESULT under xar with 50's numbers beside
xdeep() {
  run "(cd $D && timeout 2400 python3 onem.py $2 > $O/$1-deep.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xar $1 deep decode [guard $(guardtxt $1)]: $(grep '^OM ' $O/$1-deep.txt | grep '"depth"' | cut -c4- | python3 -c 'import json,sys; print("; ".join("%s tok TTFT %s s prefill %s tok/s decode %s tok/s %s ms/step" % (d.get("prompt_tokens"), d.get("ttft"), d.get("prefill_tps"), d.get("decode_tps"), d.get("ms_per_decode_step")) for d in map(json.loads, sys.stdin)))' 2>&1 | cut -c1-400) | 50: 135k 28.8, 266k 30.3 ms/step"
}
# livesum DIR TAG: the --live lines between DIR/marks.txt's marks (profsum.py live): expert tiers, resident %, hit %, promotions,
# link MB a step (stream = experts read across the link, the misses) split into prefill and decode steps
livesum() {
  run "python3 $T/profsum.py live $1 > $1/live.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT xar $2 live: $(grep '^LV ' $1/live.txt | cut -c4- | python3 -c 'import json,sys
out = []
for l in sys.stdin:
    d = json.loads(l); s = d.get("split", {}).get("decode", {})
    out.append("%s: %d steps, decode stream %s MB/step (%d steps), vram %s, hit(mean) %s %%, promo %d demo %d" % (d["phase"], d["steps"], s.get("stream"), s.get("steps", 0), d["tiers_end"].get("vram"), d["hit_mean"], d["promo"], d["demo"]))
print("; ".join(out))' 2>&1 | cut -c1-1200)"
}
cc() { stopfile || run "timeout 3300 $T/ccsim-exl3.sh 4 $1 > $O/$1-ccsim-u4.out 2>&1"; }
# fcrun TAG WHAT: one unrelated short request decoding 384 tokens (fastcheck.py, the gate's own check): ms/step and the decode
# steps it gives the heat engine to recover
fcrun() {
  run "(cd $D && timeout 300 python3 fastcheck.py 384 > $O/$1-fc.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xar $1 fastcheck 384 $2: $(grep '^FC ' $O/$1-fc.txt | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[3:]); print("%s ms/step" % d.get("ms_per_step"))' 2>&1 | cut -c1-120)"
}
# resetpc TAG: POST /reset_prefix_cache (loopback, no key: the operator's reset) -- drops every cached prefix, so the KV blocks
# finished sessions held go back to the pool and the mover can lend them to the experts again
resetpc() {
  run "curl -s -m 30 -X POST $URL/reset_prefix_cache > $O/$1-reset.txt 2>&1"
  [ "$DRY" = 1 ] || log "RESULT xar $1 reset_prefix_cache: $(head -c 200 $O/$1-reset.txt)"
}
