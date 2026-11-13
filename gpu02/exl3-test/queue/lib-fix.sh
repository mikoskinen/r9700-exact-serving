# lib-fix.sh (6.11.2026) -- sourced by the queue-fix items 50/51/52. NOT a queue item. = lib-117.sh (sourced first; it
# sources lib-pr266.sh: restart / check / gate / probe / mbp / evict / down, DRYRUN) with
#   - the fix images (X3IDX, X3SK, X3BOTH; override by env),
#   - ds41-fix-tp4.yaml: lib-117's compose plus the two A/B switches, passed through only when exported:
#       RADIANCE_DS41_INDEX   keep,split (default when unset) | keep | split | off   (the indexer fixes, both exact)
#       RADIANCE_EXL3_SPLITK  1 = the EXL3 EP split-K rules E1/E2 on; unset = off (the default)
#   - outputs in runs/xfix and RESULT lines in runs/ds41-exl3.log as before,
#   - today's 1.17 EXL3 numbers as the references (no baseline re-runs: the base image is not run again; the only
#     "off" arms are the identity check's and the deep-decode A/B, which have no earlier number).
# Needs in ~/gpu/ds41/exl3-test/queue/: lib-117.sh, lib-pr266.sh (as on gpu02 6.11.), ds41-fix-tp4.yaml, this file.
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib-117.sh" 2>/dev/null || . $HOME/gpu/ds41/exl3-test/queue/lib-117.sh || { echo "lib-117.sh not found"; exit 1; }
Y=$QD/ds41-fix-tp4.yaml; [ -e $Y ] || Y=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ds41-fix-tp4.yaml
O=$R/xfix
export STATSDIR=$O/stats
X3IDX=${X3IDX:-radiance-local:ds41-exl3-117-b4624774-f2f3c3f7-idx}
X3SK=${X3SK:-radiance-local:ds41-exl3-117-1371447b-9de01671-splitk}
X3BOTH=${X3BOTH:-radiance-local:ds41-exl3-117-3ee8358a-9de01671-idx-splitk}
unset RADIANCE_DS41_INDEX RADIANCE_EXL3_SPLITK
# Today's references (gpu02 runs/ds41-exl3.log, radiance-local:ds41-exl3-117-abf7f9a0-f2f3c3f7, same .rad):
REF_PF="6.11. x117 47 (MML 524288 MAXSEQS 4 NST0 ckint 8192 MNBT 8192, fresh): 16432 tok 2402.2 tok/s TTFT 6.841 s; 134894 tok 2002.2 / 67.374 s; 322444 tok 1522.3 / 211.807 s"
REF_BEST524="6.11. x117-best524 (fin-best, MML 524288): guard 19.71 ms; essay 47.7 tok/s 20.9 ms, code 50.6 tok/s 19.8 ms; c4 95.0 tok/s 35.8 ms; c8 123.4 tok/s 56.1 ms; pf 1249.9 tok/s TTFT 25.13 s kept 0.227; turns cold 30.61 s (56k) hit 2.31 s"
REF_PROF="6.11. prefill profile (x117 47, card 0, ms a 8192-token chunk): index_score 14.3 + 11.687 per 1k ctx, index_topk 37.1 + 1.467, index_candidates 0.165; TOTAL 3274 + 14.094"
# the morning-47 flags: the cold prefill set and the identity check
pfflags() { export HOSTPOOL=19456 NGRAM=ram CKINT=8192 MNBT=8192 NST=0 MAXSEQS=4 MML=524288 WIRE=exact XARGS=; }
# pct NEW OLD -> signed percent (NEW vs OLD)
pct() { awk -v a="$1" -v b="$2" 'BEGIN{ if (b > 0 && a != "") printf "%+.1f %%", 100 * (a - b) / b; else printf "n/a" }'; }
pfv() { python3 -c 'import json,sys; t=int(sys.argv[2]); v=[d for d in (json.loads(l[2:]) for l in open(sys.argv[1]) if l.startswith("J ")) if d.get("target")==t]; print(v[0].get("prefill_tps") if v else "")' "$1" "$2" 2>/dev/null; }
# cold TAG: 47's prompts (targets 16384, then 140000 + 256000 = 16432 / 134894 / 322444 tokens today), fresh, unprofiled
cold() {
  run "(cd $D && timeout 600 python3 ds41_tests.py prefill --depths 16384 --cpt 3.95 --reps 1 > $O/$1-pf.txt 2>&1)"
  run "(cd $D && timeout 2400 python3 ds41_tests.py prefill --depths 140000,256000 --cpt 3.95 --reps 1 >> $O/$1-pf.txt 2>&1)"
  [ "$DRY" = 1 ] && return
  log "RESULT xfix $1 cold prefill (MML 524288 MAXSEQS 4 NST0 ckint 8192, fresh) [guard $(guardtxt $1)]: $(pfline $O/$1-pf.txt) | vs 6.11.: 16k $(pct "$(pfv $O/$1-pf.txt 16384)" 2402.2), 135k $(pct "$(pfv $O/$1-pf.txt 140000)" 2002.2), 322k $(pct "$(pfv $O/$1-pf.txt 256000)" 1522.3) | ref $REF_PF"
}
# ident TAG: the exactness check -- ds41_tests.py copy over a ~145k-token context (560k chars of the stdlib filler, then
# a 2.4k-char copy task; greedy, thinking off, one run), its output saved as $O/copy-TAG-run1.txt
ident() {
  run "(cd $D && timeout 1200 python3 ds41_tests.py copy --prefix-chars 560000 --chars 2400 --max-tokens 900 --runs 1 --tag=-$1 --outdir $O > $O/$1-copy.txt 2>&1)"
}
# identcmp A B: the two outputs byte for byte (the indexer fixes are exact: same selections -> same logits -> same tokens)
identcmp() {
  [ "$DRY" = 1 ] && { echo "+ cmp $O/copy-$1-run1.txt $O/copy-$2-run1.txt"; return 0; }
  local r
  r=$(python3 -c 'import sys; a=open(sys.argv[1]).read(); b=open(sys.argv[2]).read(); f=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None if len(a)==len(b) else min(len(a),len(b))); print("IDENTICAL (%d chars)" % len(a) if a==b else "DIFFERENT at char %s (%d vs %d chars)" % (f, len(a), len(b)))' $O/copy-$1-run1.txt $O/copy-$2-run1.txt 2>&1)
  log "RESULT xfix ident $1 vs $2: $r; copy fidelity: $(grep '^J ' $O/$1-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); print("exact", d.get("exact"), "ratio", d.get("ratio"), "prompt", (d.get("usage") or {}).get("prompt_tokens"), "TTFT", d.get("ttft"))' 2>&1) / $(grep '^J ' $O/$2-copy.txt | head -1 | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()[2:]); print("exact", d.get("exact"), "ratio", d.get("ratio"), "TTFT", d.get("ttft"))' 2>&1)"
  case $r in IDENTICAL*) return 0 ;; *) return 1 ;; esac
}
# deep TAG DEPTHS: onem.py (one fresh request a depth, 256 greedy tokens: TTFT, prefill tok/s, decode ms/step from /metrics)
deep() {
  run "(cd $D && timeout 2400 python3 onem.py $2 > $O/$1-deep.txt 2>&1)"
  [ "$DRY" = 1 ] || log "RESULT xfix $1 deep decode [guard $(guardtxt $1)]: $(grep '^OM ' $O/$1-deep.txt | grep '"depth"' | cut -c4- | python3 -c 'import json,sys; print("; ".join("%s tok TTFT %s s prefill %s tok/s decode %s tok/s %s ms/step" % (d.get("prompt_tokens"), d.get("ttft"), d.get("prefill_tps"), d.get("decode_tps"), d.get("ms_per_decode_step")) for d in map(json.loads, sys.stdin)))' 2>&1 | cut -c1-500)"
}
# the gate with this kit's tag in its RESULT line (lib-117's limits and slow-state retry)
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
  log "RESULT guard xfix-$tag: $st, 1-user ${ms} ms/step (limit $lim; 6.11. 1.17 EXL3 fast 19.54-19.71, slow first tries 33.22 / 34.0) [IMAGE=$IMAGE RADIANCE_DS41_INDEX=${RADIANCE_DS41_INDEX-unset} RADIANCE_EXL3_SPLITK=${RADIANCE_EXL3_SPLITK-unset}]"
  case $st in SLOW*) return 2 ;; esac
}
imgcheck() { [ "$DRY" = 1 ] || docker image inspect $1 > /dev/null 2>&1 || { log "RESULT xfix: VIRHE image $1 not on gpu02"; exit 1; }; }
