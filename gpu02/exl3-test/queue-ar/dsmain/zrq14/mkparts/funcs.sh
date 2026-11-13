zqid() {   # zqid TAG ARMS...: THE IDENTITY BLOCK of one start (strict, T 0 greedy; module header): the warm-up answers vs stock (identw at the
           # start; idok on TAG's line), then per arm the greedy set c1 ALONE (a1-<arm><ZQSFX>) vs ref12's a1 (int12 bytes) -> IDFAIL (THE STOP)
  local tag=$1 a; shift
  idok ${tag#xzq14-}
  for a in "$@"; do
    stopfile && return 0
    alone xzq14 $a a1-$a$ZQSFX
    if [ -s "$REF12A1" ] || [ "$DRY" = 1 ]; then cmpx xzq14 a1-$a$ZQSFX-vs-ref12-a1 $REF12A1 $O/xzq14-a1-$a$ZQSFX.jsonl; idok a1-$a$ZQSFX-vs-ref12-a1
    else xlog "identity a1-$a$ZQSFX: VIRHE $REF12A1 missing -- not compared"; IDFAIL="$IDFAIL a1-$a$ZQSFX-reference-missing"; fi
  done
  return 0
}
zqbridge() {   # zqbridge N: ONE c<N> block under Z at ref14's window (BRWIN s) right after the set: the window bridge (in-process Z 35 s vs Z 20 s)
  local n=$1 w0
  [ $BRIDGE = 1 ] || return 0
  stopfile && return 0
  budok "c$n window bridge (Z, $BRWIN s window = ref14's)" 1 $((BUD1 - 3)) || return 0
  if [ $n = 1 ]; then w0=$C1WIN; C1WIN=$BRWIN; mbx 1 m16-x$BRWIN Z S || true; C1WIN=$w0
  else w0=$C4WIN; C4WIN=$BRWIN; mbx 4 m16-x$BRWIN Z S || true; C4WIN=$w0; fi
  return 0
}
zqreads16() {   # START 1's readings in their order (module header): 1 identity (THE STOP on any failure), 2 c1 B Z Z B + bridge, 3 c4 warm-up
                # + B Z Z B + bridge, 4 1u essay B (discard + U1N), Z (discard + U1N), B again x U1N
  local i a k
  if budok "identity (warm-up, a1-B + a1-Z vs ref12 a1)" 2 $BUD1; then zqid xzq14-m16 B Z || true
  else IDFAIL="$IDFAIL budget"; xlog "identity SKIPPED by the budget: VIRHE"; fi
  stopfile && return 0
  if [ -n "$IDFAIL" ]; then
    STOPPED=1
    xlog "IDENTITY GATE: VIRHE strict identity failed [$(echo $IDFAIL)] -> the item STOPS before every speed block and start 2 (exact = byte-identical: NEEDS USER) | B $(armj B), Z $(armj Z) | the failing comparisons are in $O/identity.txt and $O/xzq14-*-cmp-*.json"
    return 0
  fi
  xlog "IDENTITY GATE: every strict check IDENTICAL ($([ "$DRY" = 1 ] && echo "dry: no" || grep -c IDENTICAL $O/identity.txt 2>/dev/null) lines) -> the speed blocks run"
  # 2 c1 in ref12's window (C1WARM + C1WIN), the fixed prompt set reset per block
  i=0; for a in $(echo $C1ORD | grep -o .); do i=$((i + 1)); stopfile && return 0; budok "c1 block $i $a" 1 $BUD1 && { mbx 1 m16-$i $a S || true; }; done
  zqbridge 1
  # 3 c4: one discarded warm-up block (B), then the arms
  stopfile && return 0
  budok "c4 warm-up (discarded, B)" 1 $BUD1 && { mbx 4 m16-w B S wB || true; }
  i=0; for a in $(echo $C4ORD | grep -o .); do i=$((i + 1)); stopfile && return 0; budok "c4 block $i $a" 1 $BUD1 && { mbx 4 m16-$i $a S || true; }; done
  zqbridge 4
  # 4 1u essay: B (discard + U1N), Z (discard + U1N), B again (U1N)
  for a in B Z; do
    stopfile && return 0
    budok "1u essay under $a (1 discarded + $U1N)" 1 $BUD1 || continue
    SETN=${a}1; SETK=1
    blockx U1W $a essay || true
    for k in $(seq 1 $U1N); do stopfile && return 0; blockx U1 $a essay || true; done
  done
  stopfile && return 0
  if budok "1u essay under B again (x$U1N)" 1 $BUD1; then
    SETN=B2; SETK=2
    for k in $(seq 1 $U1N); do stopfile && return 0; blockx U1 B essay || true; done
  fi
  return 0
}
zqgate() {   # START 2's rule (coordinator 13.11.): it runs only if start 1's B shows the drop vs ref12 -- c1 B mean < G1 OR c4 B mean < G4
             # (start 1's c1-m16-<i>-S-B / c4-m16-<i>-S-B blocks, ref12's windows); ZQ14_START2 0 / 1 forces it; the budget must hold.
             # Logged either way (rule, values, decision) + gate2.json -> 0 = run start 2
  local v1 v4 dec
  if [ "$DRY" = 1 ]; then echo "+ START 2 GATE: c1 B mean (c1-m16-<i>-S-B) < $G1 OR c4 B mean (c4-m16-<i>-S-B) < $G4 -> start 2 (MAXSEQS $MS2, B only); ZQ14_START2 $S2 (auto = the rule); and elapsed + ~$S2EST min <= $BUDGET; logged either way -> $O/gate2.json"; return 0; fi
  read -r v1 v4 < <(python3 -c 'import json,re,statistics,sys
v = {1: [], 4: []}
try:
    for l in open(sys.argv[1]):
        try: d = json.loads(l)
        except ValueError: continue
        m = re.match(r"^c([14])-m16-\d+-S-B$", d.get("name") or "")
        if m and isinstance(d.get("agg_dec_tps"), (int, float)): v[int(m.group(1))].append(d["agg_dec_tps"])
except OSError: pass
print(" ".join("%.2f" % statistics.mean(v[n]) if v[n] else "na" for n in (1, 4)))' $O/mb.jsonl 2>/dev/null || echo "na na")
  dec=$(python3 -c 'import sys
v1, v4, g1, g4, force = sys.argv[1:6]
f = lambda x: None if x == "na" else float(x)
a, b = f(v1), f(v4)
why = [w for w, c in (("c1 B %s < %s" % (v1, g1), a is not None and a < float(g1)), ("c4 B %s < %s" % (v4, g4), b is not None and b < float(g4))) if c]
if force == "1": print("RUN (forced: ZQ14_START2=1; the rule says %s)" % (", ".join(why) or "no drop"))
elif force == "0": print("NOT RUN (forced: ZQ14_START2=0; the rule says %s)" % (", ".join(why) or "no drop"))
elif a is None and b is None: print("NOT RUN (no c1 / c4 B value from start 1)")
elif why: print("RUN (" + ", ".join(why) + ")")
else: print("NOT RUN (no drop: c1 B %s >= %s and c4 B %s >= %s)" % (v1, g1, v4, g4))' ${v1:-na} ${v4:-na} $G1 $G4 $S2 2>&1 | head -1)
  case "$dec" in RUN*) [ $(( $(elapsed) + S2EST )) -le $BUDGET ] || dec="NOT RUN (budget: $(elapsed) min elapsed + ~$S2EST min > $BUDGET min; the rule said ${dec#RUN })" ;; esac
  python3 -c 'import json,sys
f = lambda x: None if x in ("na", "") else float(x)
json.dump(dict(rule="c1 B mean < %s OR c4 B mean < %s" % (sys.argv[3], sys.argv[4]), c1=f(sys.argv[1]), c4=f(sys.argv[2]), decision=sys.argv[5], force=sys.argv[6]), open(sys.argv[7], "w"))' ${v1:-na} ${v4:-na} $G1 $G4 "$dec" $S2 $O/gate2.json 2>/dev/null
  xlog "START 2 GATE (MAXSEQS $MS2, B only): rule 'c1 B mean < $G1 OR c4 B mean < $G4' (start 1's B blocks, ref12's windows) | values c1 B ${v1:-na}, c4 B ${v4:-na} (ref12 124.90 / 173.20) | $dec"
  case "$dec" in RUN*) return 0 ;; esac
  return 1
}
zqreads8() {   # START 2's readings (MAXSEQS MS2, B only): identity a1-B8 vs ref12 a1 (strict; a difference -> no speed block here), c1 x C1N2,
               # c4 one discarded warm-up + C4N2, 1u essay discard + U1N -- the same blocks / windows as start 1's B
  local k
  IDFAIL=""
  if budok "start 2 identity (warm-up, a1-B8 vs ref12 a1)" 1; then zqid xzq14-m8 B || true
  else IDFAIL="$IDFAIL budget"; xlog "start 2 identity SKIPPED by the budget: VIRHE"; fi
  stopfile && return 0
  if [ -n "$IDFAIL" ]; then
    STOPPED2=1
    xlog "START 2 IDENTITY GATE: VIRHE strict identity failed [$(echo $IDFAIL)] -> start 2's speed blocks do not run (NEEDS USER) | $O/identity.txt"
    return 0
  fi
  xlog "START 2 IDENTITY GATE: a1-B8 IDENTICAL to ref12's a1 -> start 2's blocks run"
  for k in $(seq 1 $C1N2); do stopfile && return 0; budok "c1 @$MS2 block $k (B)" 1 && { mbx 1 m8-$k B S || true; }; done
  stopfile && return 0
  budok "c4 @$MS2 warm-up (discarded, B)" 1 && { mbx 4 m8-w B S wB || true; }
  for k in $(seq 1 $C4N2); do stopfile && return 0; budok "c4 @$MS2 block $k (B)" 1 && { mbx 4 m8-$k B S || true; }; done
  stopfile && return 0
  if budok "1u essay @$MS2 under B (1 discarded + $U1N)" 1; then
    SETN=B8; SETK=3
    blockx U1W B essay || true
    for k in $(seq 1 $U1N); do stopfile && return 0; blockx U1 B essay || true; done
  fi
  return 0
}
zqrows() {   # zqrows START LOG: p12win.py rows over every block / read window of START (zq14sum.py wins) -> rows-START.json / .txt
  local st=$1 el=$2 w
  if [ "$DRY" = 1 ]; then echo "+ python3 -I -B $R12S rows $el \$(python3 -I -B $YDS wins $O $st) --json $O/rows-$st.json > $O/rows-$st.txt   (every mb block's two marks + every 1u read's utc0 / utc1 of start $st)"; return 0; fi
  [ -s $el ] || { xlog "summary: NOTE no engine log $el: no step-log readings for $st"; return 0; }
  mapfile -t w < <(python3 -I -B $YDS wins $O $st 2>/dev/null)
  [ ${#w[@]} -gt 0 ] || { xlog "summary: NOTE no block / read window for $st"; return 0; }
  timeout 300 python3 -I -B $R12S rows $el "${w[@]}" --json $O/rows-$st.json > $O/rows-$st.txt 2>&1
  grep -q '^Traceback' $O/rows-$st.txt && xlog "summary: VIRHE p12win.py rows $st: $(grep -A4 '^Traceback' $O/rows-$st.txt | tail -2 | tr '\n' ' ' | cut -c1-300)"
  xlog "summary: step log $st ($(basename $el)): $(grep -c '^P12ROWS' $O/rows-$st.txt) windows read ($(grep -c 'VIRHE' $O/rows-$st.txt) without a decode step) -> rows-$st.json"
  return 0
}
zq14summary() {   # per start p12win.py rows over its engine log, then zq14sum.py run (every ZQ14 line), then the raw snapshots archived
  local st l
  if [ "$DRY" = 1 ]; then
    for st in m16 m8; do zqrows $st $O/xzq14-$st-engine-full.log.gz; done
    echo "+ python3 -I -B $YDS run $O --sw $SW > $O/zq14sum.txt -> RESULT xzq14 ZQ14 lines (every block / read: tok/s, tok a step, ms a step, acceptance, step-log row buckets, depth-0 share, q histogram, draft controller deltas; Z vs B c1 / c4 / 1u; window bridge; B @16 vs @8; refs INFO; gate; identity; VERDICT; TABLE)"
    echo "+ tar -czf $O/raw-snaps.tar.gz (st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json) and remove them; gzip start logs"
    return 0
  fi
  for st in m16 m8; do [ -s $O/xzq14-$st-engine-full.log.gz ] && zqrows $st $O/xzq14-$st-engine-full.log.gz; done
  timeout 180 python3 -I -B $YDS run $O --sw $SW > $O/zq14sum.txt 2>&1
  while IFS= read -r l; do xlog "${l:0:2600}"; done < <(grep -h '^ZQ14' $O/zq14sum.txt 2>/dev/null)
  grep -q '^ZQ14 VERDICT' $O/zq14sum.txt 2>/dev/null || xlog "summary: VIRHE zq14sum.py printed no VERDICT ($(tail -3 $O/zq14sum.txt 2>/dev/null | tr '\n' ' ' | cut -c1-300))"
  grep -lq '^Traceback' $O/rows-*.txt $O/zq14sum.txt 2>/dev/null && xlog "summary: VIRHE a Traceback in the analysis ($(grep -h -A4 '^Traceback' $O/rows-*.txt $O/zq14sum.txt 2>/dev/null | tail -2 | tr '\n' ' ' | cut -c1-300))"
  ( cd $O && tar -czf raw-snaps.tar.gz $(ls st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json 2>/dev/null) 2>/dev/null \
      && rm -f st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json ) || xlog "summary: NOTE the raw snapshots were not archived (left as they are)"
  gzip -f $O/start-*.log 2>/dev/null || true
  xlog "summary: raw snapshots -> raw-snaps.tar.gz ($(du -m $O/raw-snaps.tar.gz 2>/dev/null | cut -f1) MB); $O $(du -sm $O 2>/dev/null | cut -f1) MB; $(dfree) MB free on /"
  return 0
}
