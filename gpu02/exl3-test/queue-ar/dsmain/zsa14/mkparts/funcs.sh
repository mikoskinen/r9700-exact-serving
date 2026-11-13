sahbline() {   # sahbline A B -> one line: lever SA's GET /admin/fn4 "sahb" deltas between snapshots fn-A / fn-B (armed steps = rank 0's
               # short_steps; staged GB, pick kept / dropped, expected GB and the arena loan recalled summed over the ranks) + the switches in B
  [ "$DRY" = 1 ] && { echo "sahb deltas $1 -> $2 (dry)"; return 0; }
  python3 -c 'import json,sys
def rd(p):
    try: return json.load(open(p))
    except Exception: return None
a, b = rd(sys.argv[1]), rd(sys.argv[2])
if not a or not b or not (b.get("sahb") or {}).get("ranks"): print("sahb n/a (no GET /admin/fn4 \"sahb\" snapshot)"); sys.exit(0)
ra = {r.get("rank"): r for r in (a.get("sahb") or {}).get("ranks") or []}
t = dict(steps=0, big=0, gb=0.0, kept=0, dropped=0, exp=0.0, rec=0.0)
for r in b["sahb"]["ranks"]:
    x = ra.get(r.get("rank")) or {}
    d = lambda k: (r.get(k) or 0) - (x.get(k) or 0)
    if r.get("rank") == 0: t["steps"] = d("short_steps"); t["big"] = d("big_steps")
    t["gb"] += d("short_bytes") / 1e9; t["kept"] += d("pick_kept"); t["dropped"] += d("pick_dropped"); t["exp"] += d("expect_bytes") / 1e9; t["rec"] += d("sa_recall_bytes") / 2 ** 30
w = b.get("switches") or {}
print("SA armed %d step(s) (rank 0; stager-size steps %d), staged %.2f GB all ranks (%s GB an armed step), pick kept %d / dropped %d, expected in place %.2f GB, arena loan recalled %.2f GiB | stage_short %s stage_pick %s, stage_capable %s" % (
    t["steps"], t["big"], t["gb"], ("%.2f" % (t["gb"] / t["steps"])) if t["steps"] else "-", t["kept"], t["dropped"], t["exp"], t["rec"], w.get("stage_short"), w.get("stage_pick"), b["sahb"].get("stage_capable")))' $O/fn-$1.json $O/fn-$2.json 2>&1 | head -1 | cut -c1-500
}
sahbn() {   # sahbn A B -> rank 0's short_steps delta between fn-A / fn-B (a number, or na)
  [ "$DRY" = 1 ] && { echo 1; return 0; }
  python3 -c 'import json,sys
def r0(p):
    try: return {r.get("rank"): r for r in (json.load(open(p)).get("sahb") or {}).get("ranks") or []}.get(0)
    except Exception: return None
a, b = r0(sys.argv[1]), r0(sys.argv[2])
print(int((b.get("short_steps") or 0) - (a.get("short_steps") or 0)) if a is not None and b is not None else "na")' $O/fn-$1.json $O/fn-$2.json 2>/dev/null || echo na
}
sa14id() {   # 1 THE IDENTITY BLOCK (strict, T 0 greedy; module header): every check runs (each failure named), then sa14reads stops on IDFAIL
  local a g u0 u1 n
  idok n5   # the warm-up answers vs stock (identw at the start; no line = not compared = the stop)
  for a in $IDARMS; do
    stopfile && return 0
    alone xsa14 $a a1-$a
    xlog "a1-$a lever SA engagement (INFO: the greedy set's prompts are short, c1 alone): $(sahbline xsa14-a1-$a-0 xsa14-a1-$a-1)"
    if [ -s "$REF12A1" ] || [ "$DRY" = 1 ]; then cmpx xsa14 a1-$a-vs-ref12-a1 $REF12A1 $O/xsa14-a1-$a.jsonl; idok a1-$a-vs-ref12-a1
    else xlog "identity a1-$a: VIRHE $REF12A1 missing -- not compared"; IDFAIL="$IDFAIL a1-$a-reference-missing"; fi
  done
  if [ -s "$YZA1" ] || [ "$DRY" = 1 ]; then cmpx xsa14 a1-B-vs-yzint14c-a1-Z $YZA1 $O/xsa14-a1-B.jsonl; idok a1-B-vs-yzint14c-a1-Z
  else xlog "identity a1-B vs yzint14c a1-Z: NOTE $YZA1 missing -- not compared (ref12's a1 is the reference)"; fi
  stopfile && return 0
  u0=$(utcnow); ssnap xsa14-gs$GSC-$GSA-0; gsx $GSA; ssnap xsa14-gs$GSC-$GSA-1; u1=$(utcnow)
  [ "$DRY" = 1 ] || echo "$u0 $u1" > $O/gs$GSC-$GSA.win
  g=$(gt48 xsa14-gs$GSC-$GSA)
  xlog "gs$GSC-$GSA cap (MAXSEQS ${MAXSEQS:-?}, all $GSC in flight, sched_row_cap $CAP): $(capd xsa14-gs$GSC-$GSA-0 xsa14-gs$GSC-$GSA-1) | steps_gt48 delta ${g:-na}$([ "${g:-na}" = 0 ] || { [ $SW = 1 ] && echo " -- VIRHE > 48-row steps under the cap (or no cap counters)" || echo " -- INFO (SW=0 mock: no cap counters)"; }) | lever SA: $(sahbline xsa14-gs$GSC-$GSA-0 xsa14-gs$GSC-$GSA-1)"
  cmpx xsa14 gs$GSC-$GSA-vs-a1-$GSA $O/xsa14-a1-$GSA.jsonl $O/xsa14-gs$GSC-$GSA.jsonl; idok gs$GSC-$GSA-vs-a1-$GSA
  stopfile && return 0
  stagx $LDA $LDA || true
  n=$(sahbn xsa14-ld-$LDA-0 xsa14-ld-$LDA-1)
  [ "$DRY" = 1 ] && { echo "+ RESULT xsa14 staggered $LDA engagement: GET /admin/fn4 \"sahb\" short_steps delta over the ld run > 0 (else VIRHE: the armed steps' exactness is not shown; the speed blocks still run)"; return 0; }
  case "$n" in ''|na|0) xlog "staggered $LDA lever SA engagement: VIRHE SA armed ${n:-na} step(s) during ld-$LDA -- the exactness of ARMED steps is not shown by this item (the speed blocks still run) | $(sahbline xsa14-ld-$LDA-0 xsa14-ld-$LDA-1)" ;;
    *) xlog "staggered $LDA lever SA engagement: $(sahbline xsa14-ld-$LDA-0 xsa14-ld-$LDA-1) -> the probes' decode rows rode $n armed step(s)" ;; esac
  return 0
}
bestsa() {   # bestsa -> the SA arm (of the c8 order) with the higher mean c8 work-based agg tok/s in $O/mb.jsonl; none read: LDA
  [ "$DRY" = 1 ] && { echo $LDA; return 0; }
  python3 -c 'import json,sys
R = []
try: R = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
except Exception: pass
m = {}
for r in R:
    if r.get("n") == 8 and r.get("sched") == "G" and r.get("arm") in sys.argv[3:] and isinstance(r.get("agg_dec_tps"), (int, float)): m.setdefault(r["arm"], []).append(r["agg_dec_tps"])
print(max(m, key=lambda a: sum(m[a]) / len(m[a])) if m else sys.argv[2])' $O/mb.jsonl $LDA $SAARMS 2>/dev/null || echo $LDA
}
sa14reads() {   # the readings in their order (module header): 1 identity (THE STOP on any failure), 2 c8 work-based, 3 c16 B + the better SA arm
  local i a X
  if budok "identity (warm-up, a1 [$IDARMS] vs ref12 a1, gs$GSC-$GSA, staggered ld-$LDA)" 6; then sa14id || true
  else IDFAIL="$IDFAIL budget"; xlog "identity SKIPPED by the budget: VIRHE"; fi
  stopfile && return 0
  if [ -n "$IDFAIL" ]; then
    STOPPED=1
    xlog "IDENTITY GATE: VIRHE strict identity failed [$(echo $IDFAIL)] -> the item STOPS before every speed block (exact = byte-identical: NEEDS USER) | arms B $(armj B) S1 $(armj S1) S2 $(armj S2) on the int14 pin | the failing comparisons are in $O/identity.txt and $O/xsa14-*-cmp-*.json"
    return 0
  fi
  xlog "IDENTITY GATE: every strict check IDENTICAL ($([ "$DRY" = 1 ] && echo "dry: no" || grep -c IDENTICAL $O/identity.txt 2>/dev/null) lines) -> the speed blocks run"
  # 2 c8 WORK-BASED (TESTING.md 12.11. WINDOW QUANTUM): --window gens, in-process GORD (B S1 S2 S2 S1 B)
  i=0; for a in $GORD; do i=$((i + 1)); stopfile && return 0
    budok "c8 work-based block $i $a (gens $C8GN)" 2 || continue
    mbx 8 $i $a G || true
    xlog "c8-$i-G-$a lever SA: $(sahbline c8-$i-G-$a-0 c8-$i-G-$a-1)"
  done
  # 3 c16: one gens block for B and the better SA arm by the c8 agg
  [ $C16N -gt 0 ] || return 0
  X=$(bestsa)
  xlog "c16: B and $X (the SA arm with the higher c8 work-based agg; default $LDA), one gens block each (--gens $C16GN = $((16 * C16GN)) newcomer steps)"
  i=0; for a in B $X; do i=$((i + 1)); stopfile && return 0
    budok "c16 gens block $i $a" 2 || continue
    mbx 16 $i $a G || true
    xlog "c16-$i-G-$a lever SA: $(sahbline c16-$i-G-$a-0 c16-$i-G-$a-1)"
  done
  return 0
}
ncmset() {   # ncmset LOG LABEL MIN REQS...: ncmatch.py over the blocks (NAME = REQS minus -reqs.jsonl, ARM = its last field) -> ncm-LABEL.json / .txt
  local el=$1 lb=$2 mn=$3 f nm bl="" l; shift 3
  for f in "$@"; do [ -s "$f" ] || continue; nm=$(basename $f -reqs.jsonl); bl="$bl --block $nm:${nm##*-}:$f"; done
  [ -n "$bl" ] || return 0
  timeout 300 python3 -I -B $NCM $el --set $lb $bl --min $mn --json $O/ncm-$lb.json > $O/ncm-$lb.txt 2>&1
  while IFS= read -r l; do xlog "${l:0:1600}"; done < <(grep -h '^NCMATCH' $O/ncm-$lb.txt 2>/dev/null)
  grep -q '^Traceback' $O/ncm-$lb.txt 2>/dev/null && xlog "summary: VIRHE ncmatch.py $lb: $(grep -A4 '^Traceback' $O/ncm-$lb.txt | tail -2 | tr '\n' ' ' | cut -c1-300)"
  return 0
}
stepsx() {   # stepsx PLAIN REQS: the coordinator's steps.py (zsa14/steps.py, md5 STPMD5) over one gens block: every non-decode step and every step
             # with a prefill segment in [w0 - 2 s, w1 + 2 s] with its wall and in-window flag -> c8/NAME-steps.txt; logs its in-window non-decode walls (INFO)
  local nm w t0 t1 w0 w1
  nm=$(basename $2 -reqs.jsonl); w=$(mwin $2)
  [ -n "$w" ] || return 0
  read -r t0 t1 w0 w1 < <(python3 -c 'import sys
from datetime import datetime, timezone
a, b = float(sys.argv[1]), float(sys.argv[2])
iso = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
print(iso(a - 2.0), iso(b + 2.0), iso(a), iso(b))' ${w%%:*} ${w#*:})
  timeout 120 python3 -I -B $STP $1 $t0 $t1 $w0 $w1 > $O/c8/$nm-steps.txt 2>&1
  xlog "$nm steps.py (raw, INFO): $(awk '$NF == "True" && $3 != "decode" { n++; if ($(NF-1) != "None") s += $(NF-1) } END { printf "%d in-window non-decode step(s), walls %.1f ms", n, s }' $O/c8/$nm-steps.txt 2>/dev/null)$(grep -q '^Traceback' $O/c8/$nm-steps.txt 2>/dev/null && echo " -- VIRHE steps.py: $(tail -1 $O/c8/$nm-steps.txt | cut -c1-200)")"
}
sa14summary() {   # ncmatch.py's matched-prompt newcomer sums (c8 gens), steps.py's raw listings, then sa14sum.py (every SA14 line)
  local EL=$O/xsa14-n5-engine-full.log.gz f l
  if [ "$DRY" = 1 ]; then
    echo "+ python3 -I -B $NCM $EL --set c8g --block c8-<i>-G-<arm>:<arm>:$O/c8/c8-<i>-G-<arm>-reqs.jsonl ... --min 4 --json $O/ncm-c8g.json (the matched-prompt newcomer sums, the first block's arm B the base)"
    echo "+ zcat $EL | grep step lines > $O/steplines.tmp; per gens block python3 -I -B $STP $O/steplines.tmp <w0-2s> <w1+2s> <w0> <w1> > $O/c8/<block>-steps.txt; rm the tmp"
    echo "+ python3 -I -B $SAS $O --log $EL --sw $SW --post $POSTN --base B --c8gens $C8GN --c16gens $C16GN --json $O/sa14sum.json -> RESULT xsa14 SA14 lines (BLOCK / ARM / MODEL / C16 / HEADLINE)"
    echo "+ tar -czf $O/raw-snaps.tar.gz (st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json) and remove them; gzip start log, ld text"
    return 0
  fi
  if [ $STOPPED = 1 ]; then xlog "summary: STOPPED at the identity gate -- no speed block ran, no SA14 reading"
  elif [ -s $EL ]; then
    ncmset $EL c8g 4 $(ls $O/c8/c8-*-G-*-reqs.jsonl 2>/dev/null)   # ls order = the run order (c8-1 .. c8-6): the first arm (B) is the base
    zcat $EL 2>/dev/null | grep -E ' step [0-9]+ (prefill|decode|MIXED) n_seq=' > $O/steplines.tmp
    for f in $(ls $O/c8/c8-*-G-*-reqs.jsonl $O/c8/c16-*-G-*-reqs.jsonl 2>/dev/null); do stepsx $O/steplines.tmp $f; done
    rm -f $O/steplines.tmp
  else xlog "summary: NOTE no engine log $EL (the start did not come up): no step-log readings"; fi
  if [ $STOPPED = 0 ]; then
    timeout 300 python3 -I -B $SAS $O --log $EL --sw $SW --post $POSTN --base B --c8gens $C8GN --c16gens $C16GN --json $O/sa14sum.json > $O/sa14sum.txt 2>&1
    while IFS= read -r l; do xlog "${l:0:2600}"; done < <(grep -h '^SA14' $O/sa14sum.txt 2>/dev/null)
    grep -q '^SA14' $O/sa14sum.txt 2>/dev/null || xlog "summary: VIRHE sa14sum.py printed no SA14 line ($(tail -3 $O/sa14sum.txt 2>/dev/null | tr '\n' ' ' | cut -c1-300))"
  fi
  grep -lq '^Traceback' $O/sa14sum.txt $O/ncm-*.txt 2>/dev/null && xlog "summary: VIRHE a Traceback in the analysis ($(grep -h -A4 '^Traceback' $O/sa14sum.txt $O/ncm-*.txt 2>/dev/null | tail -2 | tr '\n' ' ' | cut -c1-300))"
  ( cd $O && tar -czf raw-snaps.tar.gz $(ls st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json 2>/dev/null) 2>/dev/null \
      && rm -f st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json ) || xlog "summary: NOTE the raw snapshots were not archived (left as they are)"
  gzip -f $O/start-*.log $O/xsa14-ld-*.txt $O/c8/*-steps.txt 2>/dev/null || true
  xlog "summary: raw snapshots -> raw-snaps.tar.gz ($(du -m $O/raw-snaps.tar.gz 2>/dev/null | cut -f1) MB); $O $(du -sm $O 2>/dev/null | cut -f1) MB; $(dfree) MB free on /"
  return 0
}
