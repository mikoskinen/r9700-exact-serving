FUCHKN=0; FUCHKLOG=""   # the fusion switches read back: checks done / the last section's arms
fusewant() {   # fusewant ARM -> "RF MF RC": the ds41_rfuse / ds41_mix_fin / route_copy_ov values the arm runs (B = int15: 0 0 0; O / F = THE int16 PIN: 2 0 1)
  case $1 in B) echo "0 0 0" ;; *) echo "$FRF $FMF $FRC" ;; esac
}
fusechk() {   # fusechk ARM: the fusion switches READ BACK right after arm() applied ARM (arm() already compared every switch of the arm's pin) --
              # GET /admin/fn4 switches.ds41_rfuse / ds41_mix_fin / route_copy_ov / exl3_x3d = the arm's (fusewant) + XB -> 0 (logged per section);
              # else VIRHE -> 1 (the block is not read); ZC16_SW=0 (mock: not posted): INFO, 0
  local want got
  want=$(fusewant $1)
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4: switches ds41_rfuse / ds41_mix_fin / route_copy_ov == $want under arm $1 + exl3_x3d $XB (asserted before the block; else VIRHE: the block is not read)"; return 0; }
  got=$(fn4get | python3 -c 'import json,sys; w=json.load(sys.stdin).get("switches") or {}; print(w.get("ds41_rfuse"), w.get("ds41_mix_fin"), w.get("route_copy_ov"), w.get("exl3_x3d"))' 2>/dev/null)
  FUCHKN=$((FUCHKN + 1)); FUCHKLOG="$FUCHKLOG $1:$(echo ${got:-none} | tr ' ' '/')"
  echo "$(date '+%F %T') arm $1 ds41_rfuse/ds41_mix_fin/route_copy_ov/exl3_x3d ${got:-none} (want $want $XB)" >> $O/fuse-readback.txt
  [ "$got" = "$want $XB" ] && return 0
  [ $SW = 0 ] && return 0
  xlog "arm $1: VIRHE GET /admin/fn4 ds41_rfuse / ds41_mix_fin / route_copy_ov / exl3_x3d ${got:-none} after the arm was applied (want $want $XB) -- this block is not read"; ARMOK=0; return 1
}
fusesay() {   # fusesay SECTION: one line -- the switch values read back before the section's blocks (then the list restarts)
  [ "$DRY" = 1 ] && return 0
  xlog "$1: ds41_rfuse/ds41_mix_fin/route_copy_ov/exl3_x3d read back before each block (GET /admin/fn4, arm:values):$FUCHKLOG$([ $SW = 0 ] && echo " (INFO: ZC16_SW=0 mock, not posted)") | $FUCHKN check(s) so far, $O/fuse-readback.txt"
  FUCHKLOG=""
}
pin16() {   # pin16 TAG: THE int16 PIN READ BACK beyond rankrb (which compared EVERY switch of pinj O on GET /admin/fn4, nothing pending, and the 4 ranks'
            # SWL values on GET /stats): GET /admin/fn4 ds41_rfuse / ds41_mix_fin / route_copy_ov / exl3_x3d = 2 / 0 / 1 / XB, "rco".in_force 1,
            # "fuse".rf_declared and rf_layers > 0; then ENGAGEMENT under the pin: snapshot, arm O again (epoch_bump: every pass re-records) + one
            # 64-token greedy request, snapshot -> "fuse" rf_form2_passes / rgs_launches and "rco" recordings / freed MOVED, rf_form1_passes /
            # mixfin_launches did NOT -> 0; else VIRHE -> 1 (nothing is read); ZC16_SW=0 (the mock serves no "fuse" / "rco"): INFO, 0
  local r
  ssnap $1-eng-0
  arm O || { xlog "$1 int16 pin: VIRHE arm O (THE int16 PIN) not applied on the engagement probe -- nothing read"; return 1; }
  run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Write one sentence about the sea.\", \"n_predict\": 64, \"temperature\": 0}' $URL/completion > /dev/null"
  ssnap $1-eng-1
  if [ "$DRY" = 1 ]; then echo "+ RESULT xzc16 $1 THE int16 PIN engagement: GET /admin/fn4 ds41_rfuse $FRF / ds41_mix_fin $FMF / route_copy_ov $FRC / exl3_x3d $XB, rco.in_force 1, fuse.rf_declared / rf_layers > 0; over the probe (arm O + 64 tok): rf_form2_passes / rgs_launches / rco recordings / rco freed > 0, rf_form1_passes / mixfin_launches + 0 (else VIRHE: nothing read)"; return 0; fi
  r=$(python3 -c 'import json,sys
def rd(p):
    try: return json.load(open(p))
    except Exception: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2]); want = tuple(int(x) for x in sys.argv[3:7])
w = b.get("switches") or {}; fa, fb = a.get("fuse") or {}, b.get("fuse"); ra, rb = a.get("rco") or {}, b.get("rco")
bad = []
sw = (w.get("ds41_rfuse"), w.get("ds41_mix_fin"), w.get("route_copy_ov"), w.get("exl3_x3d"))
if sw != want: bad.append("switches %s (want %s)" % (list(sw), list(want)))
if not isinstance(fb, dict): bad.append("no \"fuse\" in GET /admin/fn4")
if not isinstance(rb, dict): bad.append("no \"rco\" in GET /admin/fn4")
fb, rb = fb or {}, rb or {}
d = lambda x, y, k: (y.get(k) or 0) - (x.get(k) or 0)
if rb.get("in_force") != 1: bad.append("rco in_force %s (want 1)" % rb.get("in_force"))
if not (fb.get("rf_declared") or 0) > 0 or not (fb.get("rf_layers") or 0) > 0: bad.append("fuse rf_declared %s rf_layers %s (want > 0)" % (fb.get("rf_declared"), fb.get("rf_layers")))
mv = {k: d(fa, fb, k) for k in ("rf_form1_passes", "rf_form2_passes", "rgs_launches", "mixfin_launches")}
mv.update({"rco_" + k: d(ra, rb, k) for k in ("recordings", "copies", "freed")})
for k in ("rf_form2_passes", "rgs_launches", "rco_recordings", "rco_freed"):
    if not mv[k] > 0: bad.append("%s +%s (want > 0: the int16 form did not run)" % (k, mv[k]))
for k in ("rf_form1_passes", "mixfin_launches"):
    if mv[k]: bad.append("%s +%s (want 0)" % (k, mv[k]))
txt = "GET /admin/fn4 ds41_rfuse/ds41_mix_fin/route_copy_ov/exl3_x3d %s, rco in_force %s, fuse rf_declared %s rf_layers %s | over the probe (arm O, epoch_bump, + 64 tok): %s" % (
    "/".join(str(x) for x in sw), rb.get("in_force"), fb.get("rf_declared"), fb.get("rf_layers"), ", ".join("%s +%s" % kv for kv in mv.items()))
print(("OK " if not bad else "") + txt + ("" if not bad else " -- " + "; ".join(bad)))' $O/fn-$1-eng-0.json $O/fn-$1-eng-1.json $FRF $FMF $FRC $XB 2>&1 | head -1 | cut -c1-1500)
  case "$r" in OK*) xlog "$1 THE int16 PIN read back + engagement: ${r#OK }"; return 0 ;; esac
  [ $SW = 0 ] && { xlog "$1 THE int16 PIN engagement: INFO (ZC16_SW=0 mock: no fusion switches / counters served) $r"; return 0; }
  xlog "$1 THE int16 PIN read back + engagement: VIRHE $r -- nothing is read"; return 1
}
c16id() {   # 1 THE IDENTITY BLOCK (strict, T 0 greedy; module header) under F = THE int16 PIN: the warm-up answers vs stock (identw at the start);
            # a1-F vs ref12's a1 AND vs zzzfuse's a1-B (int15); gs16-F vs a1-F; STRICT staggered ld-F vs a1-F -- every check runs (each failure
            # named), then c16reads stops on IDFAIL
  local g u0 u1
  idok n5   # the warm-up answers vs stock (no line = not compared = the stop)
  stopfile && return 0
  alone xzc16 F a1-F
  if [ -s "$REF12A1" ] || [ "$DRY" = 1 ]; then cmpx xzc16 a1-F-vs-ref12-a1 $REF12A1 $O/xzc16-a1-F.jsonl; idok a1-F-vs-ref12-a1
  else xlog "identity a1-F vs ref12 a1: VIRHE $REF12A1 missing -- not compared"; IDFAIL="$IDFAIL a1-F-vs-ref12-a1(reference-missing)"; fi
  if [ -s "$ZFA1" ] || [ "$DRY" = 1 ]; then cmpx xzc16 a1-F-vs-zzzfuse-a1-B $ZFA1 $O/xzc16-a1-F.jsonl; idok a1-F-vs-zzzfuse-a1-B
  else xlog "identity a1-F vs zzzfuse a1-B: VIRHE $ZFA1 missing -- not compared"; IDFAIL="$IDFAIL a1-F-vs-zzzfuse-a1-B(reference-missing)"; fi
  stopfile && return 0
  u0=$(utcnow); ssnap xzc16-gs$GSC-F-0; gsx F; ssnap xzc16-gs$GSC-F-1; u1=$(utcnow)
  [ "$DRY" = 1 ] || echo "$u0 $u1" > $O/gs$GSC-F.win
  g=$(gt48 xzc16-gs$GSC-F)
  xlog "gs$GSC-F cap (MAXSEQS ${MAXSEQS:-?}, all $GSC in flight, sched_row_cap $CAP): $(capd xzc16-gs$GSC-F-0 xzc16-gs$GSC-F-1) | steps_gt48 delta ${g:-na}$([ "${g:-na}" = 0 ] || { [ $SW = 1 ] && echo " -- VIRHE > 48-row steps under the cap (or no cap counters)" || echo " -- INFO (SW=0 mock: no cap counters)"; }) | fuse counters over gs16: $(fusectr xzc16-gs$GSC-F-0 xzc16-gs$GSC-F-1)"
  cmpx xzc16 gs$GSC-F-vs-a1-F $O/xzc16-a1-F.jsonl $O/xzc16-gs$GSC-F.jsonl; idok gs$GSC-F-vs-a1-F
  stopfile && return 0
  stagx F F || true
  fusesay "identity"
  return 0
}
fusectr() {   # fusectr A B -> one line: GET /admin/fn4 "fuse" / "rco" counter deltas between the snapshots fn-A.json and fn-B.json (engagement: a recorded pass counts at recording)
  [ "$DRY" = 1 ] && { echo "(dry: fuse / rco deltas)"; return 0; }
  python3 -c 'import json,sys
def g(p, o):
    try: return (json.load(open(p)).get(o) or {})
    except (OSError, ValueError): return {}
a, b = g(sys.argv[1], "fuse"), g(sys.argv[2], "fuse")
ra, rb = g(sys.argv[1], "rco"), g(sys.argv[2], "rco")
k = ("rf_form1_passes", "rf_form2_passes", "rgs_launches", "mixfin_launches", "mixfin_rows")
print("declared %s, layers %s | " % (b.get("rf_declared"), b.get("rf_layers")) + ", ".join("%s +%s" % (x, (b.get(x) or 0) - (a.get(x) or 0)) for x in k)
      + " | rco in_force %s, " % rb.get("in_force") + ", ".join("%s +%s" % (x, (rb.get(x) or 0) - (ra.get(x) or 0)) for x in ("recordings", "copies", "freed")))' $O/fn-$1.json $O/fn-$2.json 2>/dev/null || echo "fuse counters n/a"
}
coldid() {   # coldid: the cold CDEP answer (F) byte for byte vs ref14's (R14C, md5 R14CMD5 = ref12's) and zzzfuse's (ZFC, only if that run made one)
             # -> identity.txt; a difference = VIRHE (reported; nothing follows it, so no stop)
  local f=$O/cold-$CDEP-1-F.txt r m
  if [ "$DRY" = 1 ]; then echo "+ cmpfile $R14C $f (ref14's $CDEP answer, md5 $R14CMD5 $([ -s $R14C ] && echo present || echo MISSING)) + cmpfile $ZFC $f ($([ -s $ZFC ] && echo present || echo "absent: zzzfuse ran no $CDEP -> not compared")) -> identity.txt (DIFFERENT = VIRHE)"; return 0; fi
  [ -s $f ] || { xlog "cold $CDEP answer: VIRHE no answer text $f -- not compared"; return 0; }
  m=$(md5sum < $R14C 2>/dev/null | cut -c1-8)
  [ "$m" = "$R14CMD5" ] || xlog "cold $CDEP: NOTE ref14's answer $R14C is md5 ${m:-?}, not $R14CMD5"
  r=$(cmpfile $R14C $f); idrec xzc16-cold$CDEP-F-vs-ref14 "$r" "$(basename $R14C) vs $(basename $f), deepq $CDEP, $NTOK greedy tokens"
  case "$r" in IDENTICAL*) ;; *) xlog "cold $CDEP F vs ref14: $([ $SW = 1 ] && echo VIRHE || echo "INFO (ZC16_SW=0 mock: the mock's text is not a real answer)") the answer differs ($r; F md5 $(md5sum < $f | cut -c1-8), ref14 ${m:-?})" ;; esac
  if [ -s $ZFC ]; then r=$(cmpfile $ZFC $f); idrec xzc16-cold$CDEP-F-vs-zzzfuse "$r" "$(basename $ZFC) vs $(basename $f)"
    case "$r" in IDENTICAL*) ;; *) xlog "cold $CDEP F vs zzzfuse: $([ $SW = 1 ] && echo VIRHE || echo "INFO (ZC16_SW=0 mock)") the answer differs ($r)" ;; esac
  else xlog "cold $CDEP F vs zzzfuse: not compared (zzzfuse ran no $CDEP read: $ZFC absent; ref14's is the reference)"; fi
}
c16reads() {   # the readings in their order (module header): 1 identity under F (THE STOP on any failure), 2 1u ABBA, 3 c4 gens (one discarded +
               # B F F B), 4 the optional cold 136k (F; the first section the budget drops)
  local i a k
  if budok "identity (warm-up, a1-F vs ref12 a1 + zzzfuse a1-B, gs$GSC-F, staggered ld-F)" 4; then c16id || true
  else IDFAIL="$IDFAIL budget"; xlog "identity SKIPPED by the budget: VIRHE"; fi
  stopfile && return 0
  if [ -n "$IDFAIL" ]; then
    STOPPED=1
    xlog "IDENTITY GATE: VIRHE strict identity failed [$(echo $IDFAIL)] -> the item STOPS before every speed block (exact = byte-identical: NEEDS USER) | arm F $(armj F) on THE int16 PIN | the failing comparisons are in $O/identity.txt and $O/xzc16-*-cmp-*.json"
    return 0
  fi
  xlog "IDENTITY GATE: every strict check IDENTICAL ($([ "$DRY" = 1 ] && echo "dry: no" || grep -c IDENTICAL $O/identity.txt 2>/dev/null) lines) -> the speed blocks run | B $(armj B) | F $(armj F)"
  # 2 1u ABBA (ESSAY RESIDENCY WARM-UP + REF12 lesson: the same prompt back to back): each block = one discarded warm-up read + U1N reads
  if [ $ONEU = 1 ]; then
    SETK=0
    for a in $U1AB; do SETK=$((SETK + 1)); SETN=U1-$SETK; stopfile && return 0
      budok "1u block $SETK $a (1 discarded + $U1N essay reads back to back)" 1 || continue
      blockx U1W $a essay || true
      for k in $(seq 1 $U1N); do stopfile && return 0; blockx U1 $a essay || true; done
    done
    SETN=""; SETK=0
    fusesay "1u ABBA [$U1AB]"
  else xlog "1u ABBA: OFF (ZC16_1U=0)"; fi
  # 3 c4 WORK-BASED (WINDOW QUANTUM): one DISCARDED warm-up block, then C4AB (B F F B), gens C4GN each
  if [ $C4ON = 1 ]; then
    xlog "c4: harness/ncmb3.py --window gens --gens $C4GN ($((4 * C4GN)) newcomer steps a block), the fixed prompt set reset per block, T 0 | $([ $C4WARMUP = 1 ] && echo "one DISCARDED warm-up block (arm $C4WA), then " || echo "no warm-up block (ZC16_C4WARMUP=0), ")[$C4AB]"
    if [ $C4WARMUP = 1 ] && budok "c4 warm-up (one DISCARDED gens block, arm $C4WA)" 2; then mbx 4 w1 $C4WA G w$C4WA || true; fi
    i=0; for a in $C4AB; do i=$((i + 1)); stopfile && return 0
      budok "c4 gens block $i $a (gens $C4GN)" 2 || continue
      mbx 4 $i $a G || true
    done
    fusesay "c4 [$C4AB]"
  else xlog "c4: OFF (ZC16_C4=0)"; fi
  # 4 cold CDEP under F (optional; the first section dropped)
  if [ $COLD = 1 ]; then
    stopfile && return 0
    if budok "cold $CDEP (deepq after a prefix-cache reset, F; the first section the budget drops)" $COLDEST; then coldx $CDEP F 1 || true; coldid; fi
  else xlog "cold $CDEP: OFF (ZC16_COLD=0)"; fi
  return 0
}
c16summary() {   # zc16sum.py (every INT16 line: identity, the pin read-back, 1u ABBA, c4 gens, 136k), then the raw snapshots into one tar.gz and the logs gzipped
  local EL=$O/xzc16-n5-engine-full.log.gz l
  if [ "$DRY" = 1 ]; then
    echo "+ python3 -I -B $SAS $O --log $EL --sw $SW --tag xzc16-n5 --post $POSTN --want 'B=0/0/0,F=$FRF/$FMF/$FRC' --xb $XB --swlg $SWLG --rfdecl $RFDECL --u1n $U1N --c4gens $C4GN --essay-md5 $ESSMD5 --cold-depth $CDEP --cold-ref-md5 $R14CMD5 --stopped <0|1> --json $O/zc16sum.json > $O/zc16sum.txt -> RESULT xzc16 INT16 lines (PIN / IDENTITY / 1u BLOCK / 1u ARM / 1u F vs B / 1u VERDICT / c4 BLOCK / c4 F vs B / c4 INFO / 136k / SUMMARY)"
    echo "+ tar -czf $O/raw-snaps.tar.gz (st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json) and remove them; gzip the start log, the ld text, the 1u / mb / cold texts"
    return 0
  fi
  [ $STOPPED = 1 ] && xlog "summary: STOPPED at the identity gate -- no speed block ran (the INT16 lines give the identity and the pin read-back)"
  [ -s $EL ] || xlog "summary: NOTE no engine log $EL (the start did not come up): no step-log readings"
  timeout 300 python3 -I -B $SAS $O --log $EL --sw $SW --tag xzc16-n5 --post $POSTN --want "B=0/0/0,F=$FRF/$FMF/$FRC" --xb $XB --swlg $SWLG --rfdecl $RFDECL --u1n $U1N --c4gens $C4GN --essay-md5 $ESSMD5 --cold-depth $CDEP --cold-ref-md5 $R14CMD5 --stopped $STOPPED --json $O/zc16sum.json > $O/zc16sum.txt 2>&1
  while IFS= read -r l; do xlog "${l:0:2600}"; done < <(grep -h '^INT16' $O/zc16sum.txt 2>/dev/null)
  grep -q '^INT16' $O/zc16sum.txt 2>/dev/null || xlog "summary: VIRHE zc16sum.py printed no INT16 line ($(tail -3 $O/zc16sum.txt 2>/dev/null | tr '\n' ' ' | cut -c1-300))"
  grep -q '^Traceback' $O/zc16sum.txt 2>/dev/null && xlog "summary: VIRHE a Traceback in zc16sum.py ($(grep -A4 '^Traceback' $O/zc16sum.txt | tail -2 | tr '\n' ' ' | cut -c1-300))"
  ( cd $O && tar -czf raw-snaps.tar.gz $(ls st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json 2>/dev/null) 2>/dev/null \
      && rm -f st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json ) || xlog "summary: NOTE the raw snapshots were not archived (left as they are)"
  gzip -f $O/start-*.log $O/xzc16-ld-*.txt $O/c8/*.txt $O/b*-U1*.txt $O/cold-*.out 2>/dev/null || true
  xlog "summary: raw snapshots -> raw-snaps.tar.gz ($(du -m $O/raw-snaps.tar.gz 2>/dev/null | cut -f1) MB); logs gzipped; $O $(du -sm $O 2>/dev/null | cut -f1) MB; $(dfree) MB free on /"
  return 0
}
