X3DN=0; X3DLOG=""   # lever K25U's switch read back: checks done / the last section's arms
x3dchk() {   # x3dchk ARM: lever K25U's switch READ BACK right after arm() applied ARM -- GET /admin/fn4 switches.exl3_x3d = XZ under Z, XB
             # under every other arm (B, O) -> 0 (logged per section); else VIRHE -> 1 (the block is not read); K25S_SW=0 (mock: exl3_x3d
             # is not posted): INFO, 0
  local want got
  case $1 in Z) want=$XZ ;; *) want=$XB ;; esac
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4: switches.exl3_x3d == $want under arm $1 (asserted before the block; else VIRHE: the block is not read)"; return 0; }
  got=$(fn4get | python3 -c 'import json,sys; print((json.load(sys.stdin).get("switches") or {}).get("exl3_x3d"))' 2>/dev/null)
  X3DN=$((X3DN + 1)); X3DLOG="$X3DLOG $1:${got:-none}"
  echo "$(date '+%F %T') arm $1 exl3_x3d ${got:-none} (want $want)" >> $O/x3d-readback.txt
  [ "$got" = "$want" ] && return 0
  [ $SW = 0 ] && return 0
  xlog "arm $1: VIRHE GET /admin/fn4 exl3_x3d ${got:-none} after the arm was applied (want $want) -- this block is not read"; ARMOK=0; return 1
}
x3dsay() {   # x3dsay SECTION: one line -- the exl3_x3d values read back before the section's blocks (then the list restarts)
  [ "$DRY" = 1 ] && return 0
  xlog "$1: exl3_x3d read back before each block (GET /admin/fn4, arm:value):$X3DLOG$([ $SW = 0 ] && echo " (INFO: K25S_SW=0 mock, exl3_x3d not posted)") | $X3DN check(s) so far, $O/x3d-readback.txt"
  X3DLOG=""
}
k25id() {   # 1 THE IDENTITY BLOCK (strict, T 0 greedy; module header): every check runs (each failure named), then k25reads stops on IDFAIL
  local a g u0 u1
  idok n5   # the warm-up answers vs stock (identw at the start; no line = not compared = the stop)
  for a in $IDARMS; do
    stopfile && return 0
    alone xk25s $a a1-$a
    if [ -s "$REF12A1" ] || [ "$DRY" = 1 ]; then cmpx xk25s a1-$a-vs-ref12-a1 $REF12A1 $O/xk25s-a1-$a.jsonl; idok a1-$a-vs-ref12-a1
    else xlog "identity a1-$a: VIRHE $REF12A1 missing -- not compared"; IDFAIL="$IDFAIL a1-$a-reference-missing"; fi
  done
  if [ -s "$YZA1" ] || [ "$DRY" = 1 ]; then cmpx xk25s a1-B-vs-yzint14c-a1-Z $YZA1 $O/xk25s-a1-B.jsonl; idok a1-B-vs-yzint14c-a1-Z
  else xlog "identity a1-B vs yzint14c a1-Z: NOTE $YZA1 missing -- not compared (ref12's a1 is the reference)"; fi
  stopfile && return 0
  u0=$(utcnow); ssnap xk25s-gs$GSC-$GSA-0; gsx $GSA; ssnap xk25s-gs$GSC-$GSA-1; u1=$(utcnow)
  [ "$DRY" = 1 ] || echo "$u0 $u1" > $O/gs$GSC-$GSA.win
  g=$(gt48 xk25s-gs$GSC-$GSA)
  xlog "gs$GSC-$GSA cap (MAXSEQS ${MAXSEQS:-?}, all $GSC in flight, sched_row_cap $CAP): $(capd xk25s-gs$GSC-$GSA-0 xk25s-gs$GSC-$GSA-1) | steps_gt48 delta ${g:-na}$([ "${g:-na}" = 0 ] || { [ $SW = 1 ] && echo " -- VIRHE > 48-row steps under the cap (or no cap counters)" || echo " -- INFO (SW=0 mock: no cap counters)"; })"
  cmpx xk25s gs$GSC-$GSA-vs-a1-$GSA $O/xk25s-a1-$GSA.jsonl $O/xk25s-gs$GSC-$GSA.jsonl; idok gs$GSC-$GSA-vs-a1-$GSA
  stopfile && return 0
  stagx $LDA $LDA || true
  x3dsay "identity"
  return 0
}
k25reads() {   # the readings in their order (module header): 1 identity (THE STOP on any failure), 2 c8 work-based, 3 c16 B / Z, 4 1u essay B then Z
  local i a k
  if budok "identity (warm-up, a1 [$IDARMS] vs ref12 a1, gs$GSC-$GSA, staggered ld-$LDA)" 5; then k25id || true
  else IDFAIL="$IDFAIL budget"; xlog "identity SKIPPED by the budget: VIRHE"; fi
  stopfile && return 0
  if [ -n "$IDFAIL" ]; then
    STOPPED=1
    xlog "IDENTITY GATE: VIRHE strict identity failed [$(echo $IDFAIL)] -> the item STOPS before every speed block (exact = byte-identical: NEEDS USER) | arms B $(armj B) Z $(armj Z) on the int14 pin | the failing comparisons are in $O/identity.txt and $O/xk25s-*-cmp-*.json"
    return 0
  fi
  xlog "IDENTITY GATE: every strict check IDENTICAL ($([ "$DRY" = 1 ] && echo "dry: no" || grep -c IDENTICAL $O/identity.txt 2>/dev/null) lines) -> the speed blocks run"
  # 2 c8 WORK-BASED (TESTING.md 12.11. WINDOW QUANTUM): --window gens, in-process GORD (B Z Z B)
  i=0; for a in $GORD; do i=$((i + 1)); stopfile && return 0
    budok "c8 work-based block $i $a (gens $C8GN)" 2 || continue
    mbx 8 $i $a G || true
  done
  x3dsay "c8"
  # 3 c16: one gens block an arm (K16A, B Z)
  if [ $C16N -gt 0 ]; then
    xlog "c16: [$K16A] one gens block each (--gens $C16GN = $((16 * C16GN)) newcomer steps)"
    i=0; for a in $K16A; do i=$((i + 1)); stopfile && return 0
      budok "c16 gens block $i $a" 2 || continue
      mbx 16 $i $a G || true
    done
    x3dsay "c16"
  fi
  # 4 1u essay (ESSAY RESIDENCY WARM-UP + REF12 lesson: the same prompt back to back): B = one discarded + U1N, then Z = one discarded + U1N
  [ $ONEU = 1 ] || return 0
  SETN=U1
  for a in $U1A; do stopfile && return 0
    budok "1u essay $a (1 discarded + $U1N back to back)" 1 || continue
    blockx U1W $a essay || true
    for k in $(seq 1 $U1N); do stopfile && return 0; blockx U1 $a essay || true; done
  done
  x3dsay "1u"
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
stepsx() {   # stepsx PLAIN REQS: the coordinator's steps.py (zzk25s/steps.py = zsa14's, md5 STPMD5) over one gens block: every non-decode step and every
             # step with a prefill segment in [w0 - 2 s, w1 + 2 s] with its wall and in-window flag -> c8/NAME-steps.txt; logs its in-window non-decode walls (INFO)
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
k25summary() {   # ncmatch.py's matched-prompt newcomer sums (c8 gens), steps.py's raw listings, then k25sum.py (every K25S line)
  local EL=$O/xk25s-n5-engine-full.log.gz f l
  if [ "$DRY" = 1 ]; then
    echo "+ python3 -I -B $NCM $EL --set c8g --block c8-<i>-G-<arm>:<arm>:$O/c8/c8-<i>-G-<arm>-reqs.jsonl ... --min 4 --json $O/ncm-c8g.json (the matched-prompt newcomer sums, the first block's arm B the base)"
    echo "+ zcat $EL | grep step lines > $O/steplines.tmp; per gens block python3 -I -B $STP $O/steplines.tmp <w0-2s> <w1+2s> <w0> <w1> > $O/c8/<block>-steps.txt; rm the tmp"
    echo "+ python3 -I -B $SAS $O --log $EL --sw $SW --post $POSTN --base B --zarm Z --xb $XB --xz $XZ --c8gens $C8GN --c16gens $C16GN --essay-md5 $ESSMD5 --json $O/k25sum.json -> RESULT xk25s K25S lines (BLOCK / ARM / MODEL / C16 / 1u / REFS / HEADLINE)"
    echo "+ tar -czf $O/raw-snaps.tar.gz (st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json) and remove them; gzip start log, ld text"
    return 0
  fi
  if [ $STOPPED = 1 ]; then xlog "summary: STOPPED at the identity gate -- no speed block ran, no K25S reading"
  elif [ -s $EL ]; then
    ncmset $EL c8g 4 $(ls $O/c8/c8-*-G-*-reqs.jsonl 2>/dev/null)   # ls order = the run order (c8-1 .. c8-4): the first arm (B) is the base
    zcat $EL 2>/dev/null | grep -E ' step [0-9]+ (prefill|decode|MIXED) n_seq=' > $O/steplines.tmp
    for f in $(ls $O/c8/c8-*-G-*-reqs.jsonl $O/c8/c16-*-G-*-reqs.jsonl 2>/dev/null); do stepsx $O/steplines.tmp $f; done
    rm -f $O/steplines.tmp
  else xlog "summary: NOTE no engine log $EL (the start did not come up): no step-log readings"; fi
  if [ $STOPPED = 0 ]; then
    timeout 300 python3 -I -B $SAS $O --log $EL --sw $SW --post $POSTN --base B --zarm Z --xb $XB --xz $XZ --c8gens $C8GN --c16gens $C16GN --essay-md5 $ESSMD5 --json $O/k25sum.json > $O/k25sum.txt 2>&1
    while IFS= read -r l; do xlog "${l:0:2600}"; done < <(grep -h '^K25S' $O/k25sum.txt 2>/dev/null)
    grep -q '^K25S' $O/k25sum.txt 2>/dev/null || xlog "summary: VIRHE k25sum.py printed no K25S line ($(tail -3 $O/k25sum.txt 2>/dev/null | tr '\n' ' ' | cut -c1-300))"
  fi
  grep -lq '^Traceback' $O/k25sum.txt $O/ncm-*.txt 2>/dev/null && xlog "summary: VIRHE a Traceback in the analysis ($(grep -h -A4 '^Traceback' $O/k25sum.txt $O/ncm-*.txt 2>/dev/null | tail -2 | tr '\n' ' ' | cut -c1-300))"
  ( cd $O && tar -czf raw-snaps.tar.gz $(ls st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json 2>/dev/null) 2>/dev/null \
      && rm -f st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json ) || xlog "summary: NOTE the raw snapshots were not archived (left as they are)"
  gzip -f $O/start-*.log $O/xk25s-ld-*.txt $O/c8/*-steps.txt 2>/dev/null || true
  xlog "summary: raw snapshots -> raw-snaps.tar.gz ($(du -m $O/raw-snaps.tar.gz 2>/dev/null | cut -f1) MB); $O $(du -sm $O 2>/dev/null | cut -f1) MB; $(dfree) MB free on /"
  return 0
}
