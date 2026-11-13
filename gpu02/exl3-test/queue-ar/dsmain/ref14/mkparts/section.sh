# ---------------------------------------------------------------- ref14's own functions (the tables, the int14 pin read back, identity + the stop, readings, summary)
M16TAB=$STATSDIR/m16-forms.txt   # A4's table (ref14/m16-forms.txt = yzint14c's = yd2m's P0 bench; the server reads it at /stats/m16-forms.txt)
tabs14() {   # tabs14: int14's two tables, NOT re-benched -- A4's (ref14/m16-forms.txt, md5 A4MD5, A4N lines) copied to M16TAB; A11's ranged
             # entries (ref14/a11-entries.txt = yzint14c's P0 runs/xdsm/yzint14c/x1/p0-tab.entries, md5 A11MD5, A11NE entries) -> A11ENT.
             # A mismatch: VIRHE, not started (the reference reads int14 exactly) -> 1
  local m4 n4 m11 n11
  m4=$(md5sum < $A4TAB 2>/dev/null | cut -c1-8); n4=$(grep -c . $A4TAB 2>/dev/null)
  [ "$m4" = "$A4MD5" ] && [ "$n4" = "$A4N" ] || { xlog "tables: VIRHE A4's table $A4TAB is md5 ${m4:-?} / ${n4:-?} lines, not $A4MD5 / $A4N -- not started"; return 1; }
  m11=$(md5sum < $A11TAB 2>/dev/null | cut -c1-8); A11ENT=$(tr -d ' \n' < $A11TAB 2>/dev/null); n11=$(echo "$A11ENT" | tr ',' '\n' | grep -c .)
  [ "$m11" = "$A11MD5" ] && [ "$n11" = "$A11NE" ] || { xlog "tables: VIRHE A11's entries $A11TAB are md5 ${m11:-?} / ${n11:-?} entries, not $A11MD5 / $A11NE -- not started"; return 1; }
  case "$A11ENT" in *[!A-Za-z0-9_:x@=,-]*) xlog "tables: VIRHE A11's entries hold a character outside the entry syntax -- not started"; return 1 ;; esac
  run "cp $A4TAB $M16TAB"
  xlog "tables (NOT re-benched; yzint14c's): A4 $(basename $A4TAB) md5 $m4, $n4 lines -> /stats/m16-forms.txt | A11 $(basename $A11TAB) md5 $m11, $n11 ranged entries: $A11ENT"
}
xpline() {   # xpline TAG: the start log's "RADIANCE_FN4_AR_XPUSH_MIN=XPMIN: ar_xpush's XPUSH form from XPMIN elements" (libr4d's load line:
             # the bound in force, read back at start); missing / another bound: VIRHE -> 1 (not read: the pin is int14's) (R14_SW=0: INFO)
  [ "$DRY" = 1 ] && { echo "+ RESULT xref14 $1: XPUSH gate at start (read back): the engine log names 'RADIANCE_FN4_AR_XPUSH_MIN=$XPMIN: ar_xpush's XPUSH form from $XPMIN elements' (else VIRHE: not read)"; return 0; }
  local l; l=$(docker logs $CN 2>&1 | grep -m1 "RADIANCE_FN4_AR_XPUSH_MIN=" | sed 's/^radiance: //' | cut -c1-260)
  case "$l" in *"RADIANCE_FN4_AR_XPUSH_MIN=$XPMIN: ar_xpush's XPUSH form from $XPMIN elements"*) xlog "$1 XPUSH gate at start (read back): $l"; return 0 ;; esac
  if [ $SW = 0 ]; then xlog "$1 XPUSH gate at start: INFO (R14_SW=0 mock) '${l:-no line}'"; return 0; fi
  xlog "$1 XPUSH gate at start: VIRHE '${l:-no load line}' is not the bound $XPMIN -- int14's pin cannot be read: not read"; return 1
}
a4line() {   # a4line TAG: the start log's "lever A4 (gg_gemv_m16): N forms from /stats/m16-forms.txt" with N = A4N
  [ "$DRY" = 1 ] && { echo "+ RESULT xref14 $1: the engine log names 'lever A4 (gg_gemv_m16): $A4N forms from /stats/m16-forms.txt' (else VIRHE: gg_gemv_m16 1 would run D2's forms)"; return 0; }
  local l n
  l=$(docker logs $CN 2>&1 | grep -m1 'lever A4 (gg_gemv_m16):' | sed 's/^radiance: //' | cut -c1-300); n=$(echo "$l" | sed -n 's/.*: \([0-9]*\) forms from.*/\1/p')
  if [ -n "$n" ] && [ "$n" = "$A4N" ]; then xlog "$1 A4 table: $l (the table's $A4N lines)"
  elif [ $SW = 0 ]; then xlog "$1 A4 table: INFO (R14_SW=0 mock) '${l:-no line}'"
  else xlog "$1 A4 table: VIRHE '${l:-no line}' vs the table's $A4N lines"; return 1; fi
}
rankrb() {   # rankrb TAG: THE int14 PIN READ BACK right after swnote posted it -- GET /admin/fn4: every switch of the pin (pinj O) applied, nothing
             # pending, the int14 values named; GET /stats experts.ranks: FOUR ranks, each place_swl 1, heat_min_gain_x100 SWLG, heat_hyst_x100
             # 125, place_lf 0 (SWL lives in each rank's mover) -> 0; else VIRHE -> 1 (not read) (R14_SW=0: INFO, 0)
  [ "$DRY" = 1 ] && { echo "+ RESULT xref14 $1: THE int14 PIN read back -- GET /admin/fn4 ds41_bi 6, gg_pf 2, ar_xpush 1, gg_gemv_m16 1, place_swap_lent 1, heat_min_gain_x100 $SWLG, heat_hyst_x100 125 (+ every other switch of the pin), nothing pending; GET /stats on each of the 4 ranks place_swl 1, heat_min_gain_x100 $SWLG, heat_hyst_x100 125, place_lf 0 (else VIRHE: not read)"; return 0; }
  local r
  ssnap $1-pin
  r=$(python3 -c 'import json,sys
def rd(p):
    try: return json.load(open(p))
    except Exception: return None
f, s, w, g = rd(sys.argv[1]), rd(sys.argv[2]), json.loads(sys.argv[3]), int(sys.argv[4])
bad = []
sw = (f or {}).get("switches") or {}
if not f: bad.append("no GET /admin/fn4")
else:
    bad += ["%s %s (want %s)" % (k, sw.get(k), v) for k, v in w.items() if sw.get(k) != v]
    if f.get("pending"): bad.append("pending %s" % f.get("pending"))
rk = ((s or {}).get("experts") or {}).get("ranks") or []
if len(rk) != 4: bad.append("%d ranks in /stats (want 4)" % len(rk))
for r in rk:
    for k, v in (("place_swl", 1), ("heat_min_gain_x100", g), ("heat_hyst_x100", 125), ("place_lf", 0)):
        if r.get(k) != v: bad.append("rank %s %s %s (want %s)" % (r.get("rank"), k, r.get(k), v))
K = ("ds41_bi", "gg_pf", "ar_xpush", "gg_gemv_m16", "place_swap_lent", "heat_min_gain_x100", "heat_hyst_x100", "sched_row_cap", "exl3_x3d", "mhc_fuse", "draft_depth0", "ds41_gq", "ds41_e2", "ds41_mix_wv")
txt = "fn4 %s (epoch %s, applied_step %s; all %d switches of the pin compared) | /stats ranks: %s" % (" ".join("%s %s" % (k, sw.get(k)) for k in K), (f or {}).get("epoch"), (f or {}).get("applied_step"), len(w),
      "; ".join("r%s swl %s mg %s hy %s lf %s" % (r.get("rank"), r.get("place_swl"), r.get("heat_min_gain_x100"), r.get("heat_hyst_x100"), r.get("place_lf")) for r in rk) or "none")
print(("OK " if not bad else "") + txt + ("" if not bad else " -- " + ", ".join(bad[:12])))' $O/fn-$1-pin.json $O/st-$1-pin.json "$(pinj O)" $SWLG 2>&1 | head -1 | cut -c1-1800)
  case "$r" in OK*) xlog "$1 THE int14 PIN read back (fn4 + every rank): ${r#OK }"; return 0 ;; esac
  [ $SW = 0 ] && { xlog "$1 THE int14 PIN read back: INFO (R14_SW=0 mock: the int14 switches are not posted) $r"; return 0; }
  xlog "$1 THE int14 PIN read back: VIRHE $r"; return 1
}
stagx() {   # stagx ARM REF: STAGGERED arrivals under ARM (zepb's epbstag, stack4/loadid.py run, read only): LDCONC decode streams + the 20
            # greedy probes one at a time, each with a FRESH LDPF-token prefill joining 0.5 s in (its chunks ride the probe's decode: MIXED
            # steps) -> ld-ARM vs a1-REF (this start's c1 alone), STRICT: a difference is a VIRHE line with the first differing char / token
            # and goes into IDFAIL (THE STOP)
  local a=$1 r=$2 t0=$(date +%s)
  arm $a || { IDFAIL="$IDFAIL ld-$a(no-pin)"; return 1; }
  pcreset xref14-ld-$a; ssnap xref14-ld-$a-0
  run "(cd $D && timeout 1800 python3 -I $LD run --url $URL --gs $GS --mb $MBP --out $O/xref14-ld-$a.jsonl --conc $LDCONC --warm 10 --offset 0.5 --pf-tok $LDPF --max-tokens $MT > $O/xref14-ld-$a.txt 2>&1)"
  ssnap xref14-ld-$a-1
  cmpx xref14 ld-$a-vs-a1-$r $O/xref14-a1-$r.jsonl $O/xref14-ld-$a.jsonl
  [ "$DRY" = 1 ] && { echo "+ RESULT xref14 staggered $a: LOADID (probes, MIXED steps during the probes, prefills cold) + ld-$a vs a1-$r strict (a difference = VIRHE -> THE STOP before the speed blocks)"; return 0; }
  xlog "staggered $a ($(NAMEOF $a) $(armj $a); loadid.py run, conc $LDCONC streams, fresh $LDPF-token prefill 0.5 s into each probe, T 0, $MT tok, $(( $(date +%s) - t0 )) s; compared with a1-$r): $(grep -m1 '^LOADID ' $O/xref14-ld-$a.txt | cut -c8-700) | probes in MIXED steps: $(python3 -c 'import json,sys
n = m = 0
for l in open(sys.argv[1]):
    try: d = json.loads(l)
    except ValueError: continue
    if "mixed" in d: n += 1; m += 1 if (d.get("mixed") or 0) > 0 else 0
print("%d of %d probes had MIXED steps in their life" % (m, n))' $O/xref14-ld-$a.jsonl 2>&1 | cut -c1-200) | cap $(capd xref14-ld-$a-0 xref14-ld-$a-1) | $(docker logs --since 10m $CN 2>&1 | grep 'lever A9 stats\|lever BI stats' | tail -2 | sed 's/^radiance: //; s/^[A-Z]\[[0-9]\] //' | cut -c1-200 | tr '\n' '|')"
  idok ld-$a-vs-a1-$r
  return 0
}
coldid() {   # coldid DEPTH: this start's cold answer (O) vs yzint14c's Z and ref12's at the same deepq prompt, where they exist, byte for byte;
             # a difference is a VIRHE line (after the speed blocks: reported, NEEDS USER); R14_SW=0: INFO (real-run references)
  local k=$1 r f lab n=0 p
  [ "$DRY" = 1 ] && { echo "+ identity cold-$k: cmpfile $O/cold-$k-1-O.txt vs $YZ/cold-$k-2-Z.txt (yzint14c Z) and $R/xdsm/ref12/cold-$k.txt (ref12), where present"; return 0; }
  for p in "yzint14c-Z $YZ/cold-$k-2-Z.txt" "ref12 $R/xdsm/ref12/cold-$k.txt"; do
    lab=${p%% *}; f=${p#* }
    [ -s "$f" ] || continue
    n=$((n + 1))
    if [ $SW = 0 ]; then xlog "identity cold-$k vs $lab: INFO (R14_SW=0 mock: a real-run reference, not compared)"; continue; fi
    r=$(cmpfile $f $O/cold-$k-1-O.txt); idrec "xref14-cold-$k-vs-$lab" "$r" "the $NTOK-token answers after a $k-token cold prefill vs $lab's ($f), byte for byte"
    case "$r" in IDENTICAL*) ;; *) xlog "identity cold-$k vs $lab: VIRHE the answers differ (strict, T 0 greedy, the same deepq prompt) -- NEEDS USER (after the speed blocks: reported)" ;; esac
  done
  [ $n -gt 0 ] || xlog "identity cold-$k: INFO no yzint14c / ref12 answer at this depth -- not compared"
}
r14id() {   # 1 THE IDENTITY BLOCK (strict, T 0 greedy; module header): every check runs (each failure named), then r14reads stops on IDFAIL
  local k r g u0 u1
  idok n5   # the warm-up answers vs stock (identw at the start; no line = not compared = the stop)
  alone xref14 O a1-O
  if { [ -s "$YZA1" ] && [ -s "$REF12A1" ]; } || [ "$DRY" = 1 ]; then
    cmpx xref14 a1-O-vs-yzint14c-a1-Z $YZA1 $O/xref14-a1-O.jsonl; idok a1-O-vs-yzint14c-a1-Z
    cmpx xref14 a1-O-vs-ref12-a1 $REF12A1 $O/xref14-a1-O.jsonl; idok a1-O-vs-ref12-a1
  else xlog "identity a1-O: VIRHE $YZA1 or $REF12A1 missing -- not compared"; IDFAIL="$IDFAIL a1-reference-missing"; fi
  stopfile && return 0
  u0=$(utcnow); ssnap xref14-gs$GSC-O-0; gsx O; ssnap xref14-gs$GSC-O-1; u1=$(utcnow)
  [ "$DRY" = 1 ] || echo "$u0 $u1" > $O/gs$GSC-O.win
  g=$(gt48 xref14-gs$GSC-O)
  xlog "gs$GSC-O cap (MAXSEQS ${MAXSEQS:-?}, all $GSC in flight, sched_row_cap $CAP): $(capd xref14-gs$GSC-O-0 xref14-gs$GSC-O-1) | steps_gt48 delta ${g:-na}$([ "${g:-na}" = 0 ] || { [ $SW = 1 ] && echo " -- VIRHE > 48-row steps under the cap (or no cap counters)" || echo " -- INFO (SW=0 mock: no cap counters)"; })"
  cmpx xref14 gs$GSC-O-vs-a1-O $O/xref14-a1-O.jsonl $O/xref14-gs$GSC-O.jsonl; idok gs$GSC-O-vs-a1-O
  for k in $A11ID; do   # lever A11 IN RANGE: one prefill launch of ~k + 60 rows (deepq's task) alone, cold, vs yzint14c's Z answer (the same prompt)
    stopfile && return 0
    coldx $k O 1 || { IDFAIL="$IDFAIL cold-$k(no-answer)"; continue; }
    [ "$DRY" = 1 ] && { echo "+ identity cold-$k vs yzint14c-Z: cmpfile $YZ/cold-$k-2-Z.txt $O/cold-$k-1-O.txt (strict -> the stop)"; continue; }
    if [ $SW = 0 ]; then xlog "identity cold-$k vs yzint14c-Z: INFO (R14_SW=0 mock: a real-run reference, not compared)"; continue; fi
    r=$(cmpfile $YZ/cold-$k-2-Z.txt $O/cold-$k-1-O.txt); idrec "xref14-cold-$k-vs-yzint14c-Z" "$r" "the $NTOK-token answers after a $k-token cold prefill (one in-range launch: A11's ranged entries) vs yzint14c's Z, byte for byte"
    idok cold-$k-vs-yzint14c-Z
  done
  stopfile && return 0
  stagx O O || true
  return 0
}
r14reads() {   # the readings in their order (module header): 1 identity (THE STOP on any failure), 2 1u, 3 c8 work-based B Z Z B + mb, 4 cold, 5 c4 / c1
  local i k a
  if budok "identity (warm-up, a1 vs yzint14c a1-Z + ref12 a1, gs$GSC, A11 in range [$A11ID], staggered)" 5; then r14id || true
  else IDFAIL="$IDFAIL budget"; xlog "identity SKIPPED by the budget: VIRHE"; fi
  stopfile && return 0
  if [ -n "$IDFAIL" ]; then
    STOPPED=1
    xlog "IDENTITY GATE: VIRHE strict identity failed [$(echo $IDFAIL)] -> the item STOPS before every speed block (exact = byte-identical: NEEDS USER) | O = the int14 pin $(armj O) | the failing comparisons are in $O/identity.txt and $O/xref14-*-cmp-*.json"
    return 0
  fi
  if [ "$DRY" != 1 ]; then
    local xl; xl=$(docker logs $CN 2>&1 | grep -m1 '\[r4d\] ar_xpush 1:' | cut -c1-240)
    case "$xl" in *"from $XPMIN elements; the first at "*) xlog "XPUSH in the int14 pin (read back, the first XPUSH launch of the run): $xl" ;;
      '') xlog "XPUSH in the int14 pin: NOTE no XPUSH launch logged yet (no message of >= $XPMIN elements so far)" ;;
      *) xlog "XPUSH in the int14 pin: $([ $SW = 1 ] && echo VIRHE || echo "INFO (R14_SW=0)") the first XPUSH launch names another bound: $xl" ;; esac
  fi
  xlog "IDENTITY GATE: every strict check IDENTICAL ($([ "$DRY" = 1 ] && echo "dry: no" || grep -c IDENTICAL $O/identity.txt 2>/dev/null) lines) -> the speed blocks run"
  # 2 1u: the warm essay set (prof1's P12CONFIRM set: one discarded + U1N back to back), then the switched essay (right after code) x SWN
  if budok "1u essay (1 discarded + $U1N back to back) + switched essay x$SWN" 2; then
    SETN=U1; SETK=1
    blockx U1W O essay || true
    for k in $(seq 1 $U1N); do stopfile && return 0; blockx U1 O essay || true; done
    SETN=SW; SETK=2
    for k in $(seq 1 $SWN); do stopfile && return 0; blockx XC O code || true; stopfile && return 0; blockx XE O essay || true; done
  fi
  # 3 c8 WORK-BASED (coordinator 12.11. method fix): --window gens, in-process C8GORD (B Z Z B); then the 35 s blocks (INFO), c16, CAP48
  i=0; for a in $(echo $C8GORD | grep -o .); do i=$((i + 1)); stopfile && return 0; budok "c8 work-based block $i $a (gens $C8GN)" 2 && { mbx 8 $i $a G || true; }; done
  for k in $(seq 1 $C8N); do stopfile && return 0; budok "c8 35 s block $k (INFO)" 1 && { mbx 8 $k O S || true; }; done
  for k in $(seq 1 $C16N); do stopfile && return 0; budok "c16 fixed harness block $k" 2 && { mbx 16 $k O S || true; }; done
  stopfile && return 0; budok "c16 CAP48 method" 1 && { c16x 1 O || true; }
  # 4 cold
  for k in $CDEPTHS; do
    stopfile && return 0
    budok "cold $k" $(( k / 100000 + 1 )) || continue
    coldx $k O 1 || continue
    coldid $k
  done
  # 5 c4 (one discarded warm-up, TESTING.md 12.11.) and c1 -- last: the budget drops c1, then c4's second block, first
  if [ $C4N -gt 0 ]; then
    [ $C4WARMUP = 1 ] && { stopfile && return 0; c4warm "once a start, before the c4 blocks" || true; }
    for k in $(seq 1 $C4N); do stopfile && return 0; budok "c4 block $k" 1 && { mbx 4 $k O S || true; }; done
  fi
  for k in $(seq 1 $C1N); do stopfile && return 0; budok "c1 block $k" 1 && { mbx 1 $k O S || true; }; done
  return 0
}
splitblk() {   # splitblk LOG NAME: p12win.py split + rows of one mb block (its two marks) -> split-NAME.txt / rows-NAME.json
  local w; w=$(mwin $O/c8/$2-reqs.jsonl)
  [ -n "$w" ] || { xlog "summary: $2 -- no window (no marks in $O/c8/$2-reqs.jsonl)"; return 1; }
  timeout 300 python3 -I -B $R12S split $1 --win $2:$w --mbline $O/c8/$2.txt --sw0 $O/c8/$2-st-w0.json --sw1 $O/c8/$2-st-w1.json --target 200 > $O/split-$2.txt 2>&1
  timeout 300 python3 -I -B $R12S rows $1 --win $2:$w --json $O/rows-$2.json > $O/rows-$2.txt 2>&1
}
ncmset() {   # ncmset LOG LABEL MIN REQS...: ncmatch.py over the blocks (NAME = REQS minus -reqs.jsonl, ARM = its last field) -> ncm-LABEL.json / .txt;
            # MIN = the fewest matched prompts that is not a VIRHE (4 for the c8 / c16 newcomer readings; 1 for c4 / c1, INFO, few newcomers a window)
  local el=$1 lb=$2 mn=$3 f nm bl="" l; shift 3
  for f in "$@"; do [ -s "$f" ] || continue; nm=$(basename $f -reqs.jsonl); bl="$bl --block $nm:${nm##*-}:$f"; done
  [ -n "$bl" ] || return 0
  timeout 300 python3 -I -B $NCM $el --set $lb $bl --min $mn --json $O/ncm-$lb.json > $O/ncm-$lb.txt 2>&1
  while IFS= read -r l; do xlog "${l:0:1600}"; done < <(grep -h '^NCMATCH' $O/ncm-$lb.txt 2>/dev/null)
  grep -q '^Traceback' $O/ncm-$lb.txt 2>/dev/null && xlog "summary: VIRHE ncmatch.py $lb: $(grep -A4 '^Traceback' $O/ncm-$lb.txt | tail -2 | tr '\n' ' ' | cut -c1-300)"
  return 0
}
r14summary() {   # per mb block p12win.py split (decode) + rows, ncmatch.py's matched-prompt newcomer sums per set, the gs16 step log, then r14sum.py
  local EL=$O/xref14-n5-engine-full.log.gz nm u0 u1 l f
  if [ "$DRY" = 1 ]; then
    echo "+ per mb block <c1|c4|c8|c16>-<i>-<S|G>-<arm> + c16cap-1-O: python3 -I -B $R12S split $EL --win <name>:<marks> --mbline $O/c8/<name>.txt --sw0/--sw1 ... > $O/split-<name>.txt (its (a) decode ms a step; its (b) newcomer mean NOT used); rows --json $O/rows-<name>.json"
    echo "+ python3 -I -B $NCM $EL --set c8g --block c8-<i>-G-<B|Z>:<arm>:$O/c8/c8-<i>-G-<arm>-reqs.jsonl ... --min 4 --json $O/ncm-c8g.json (the matched-prompt newcomer sums, B Z Z B); likewise c8t (35 s) and c16 (--min 4), c4 and c1 (--min 1, INFO)"
    echo "+ python3 -I -B $R12S rows $EL --win gs$GSC-O:<u0>:<u1>: > 48-row decode steps = 0 under the cap, else VIRHE; python3 -I -B $R12S confirm $O/blocks.jsonl --lever U1 -> P12CONFIRM (INFO)"
    echo "+ python3 -I -B $YDS $O --sw $SW --gens $C8GN --essay-md5 $ESSMD5 --code-md5 $CODEMD5 -> RESULT xref14 REF14 lines (every reading: value, spread, target, int12 refs prof1 / ref12 / res12 / yzint14c B + yzint14c Z as INFO)"
    echo "+ tar -czf $O/raw-snaps.tar.gz (st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json) and remove them; gzip start log"
    return 0
  fi
  if [ -s $EL ]; then
    for f in $(ls $O/c8/c1-*-reqs.jsonl $O/c8/c4-*-reqs.jsonl $O/c8/c8-*-reqs.jsonl $O/c8/c16-*-reqs.jsonl $O/c8/c16cap-*-reqs.jsonl 2>/dev/null); do
      nm=$(basename $f -reqs.jsonl)
      splitblk $EL $nm || continue
      while IFS= read -r l; do xlog "$nm ${l:0:1600}"; done < <(grep -hE '^P12SPLIT' $O/split-$nm.txt 2>/dev/null)
    done
    ncmset $EL c8g 4 $(ls $O/c8/c8-*-G-*-reqs.jsonl 2>/dev/null)   # ls order = the run order (c8-1 .. c8-4): the first arm (B) is the base
    ncmset $EL c8t 4 $(ls $O/c8/c8-*-S-O-reqs.jsonl 2>/dev/null)
    ncmset $EL c16 4 $(ls $O/c8/c16-*-S-O-reqs.jsonl 2>/dev/null)
    ncmset $EL c4 1 $(ls $O/c8/c4-[0-9]*-S-O-reqs.jsonl 2>/dev/null)
    ncmset $EL c1 1 $(ls $O/c8/c1-*-S-O-reqs.jsonl 2>/dev/null)
    if [ -s $O/gs$GSC-O.win ]; then
      read -r u0 u1 < $O/gs$GSC-O.win
      timeout 300 python3 -I -B $R12S rows $EL --win gs$GSC-O:$u0:$u1 --json $O/gsrows-O.json > $O/gsrows-O.txt 2>&1
      python3 -c 'import json,sys; d = json.load(open(sys.argv[1]))["windows"]; w = list(d.values())[0]; sys.exit(0 if w.get("gt48", 0) == 0 else 1)' $O/gsrows-O.json 2>/dev/null \
        || { [ $SW = 1 ] && xlog "gs$GSC-O step log: VIRHE > 48-row decode steps under the cap (or no step-log reading)" || xlog "gs$GSC-O step log: INFO > 48-row steps (SW=0 mock: no row cap emulated)"; }
    fi
  else xlog "summary: NOTE no engine log $EL (the start did not come up): no splits"; fi
  [ -s $O/blocks.jsonl ] && timeout 120 python3 -I -B $R12S confirm $O/blocks.jsonl --lever U1 --expect-tps 90 --expect-mb 27 > $O/confirm.txt 2>&1
  while IFS= read -r l; do xlog "${l:0:1500} (INFO; the REF14 1u line is the reading)"; done < <(grep -h '^P12CONFIRM' $O/confirm.txt 2>/dev/null)
  timeout 120 python3 -I -B $YDS $O --sw $SW --gens $C8GN --essay-md5 $ESSMD5 --code-md5 $CODEMD5 > $O/r14sum.txt 2>&1
  while IFS= read -r l; do xlog "${l:0:2400}"; done < <(grep -h '^REF14' $O/r14sum.txt 2>/dev/null)
  grep -q '^REF14' $O/r14sum.txt 2>/dev/null || xlog "summary: VIRHE r14sum.py printed no REF14 line ($(tail -3 $O/r14sum.txt 2>/dev/null | tr '\n' ' ' | cut -c1-300))"
  grep -lq '^Traceback' $O/split-*.txt $O/rows-*.txt $O/gsrows-*.txt $O/r14sum.txt $O/confirm.txt 2>/dev/null && xlog "summary: VIRHE a Traceback in the analysis ($(grep -h -A4 '^Traceback' $O/split-*.txt $O/rows-*.txt $O/gsrows-*.txt $O/r14sum.txt $O/confirm.txt 2>/dev/null | tail -2 | tr '\n' ' ' | cut -c1-300))"
  ( cd $O && tar -czf raw-snaps.tar.gz $(ls st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json 2>/dev/null) 2>/dev/null \
      && rm -f st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json ) || xlog "summary: NOTE the raw snapshots were not archived (left as they are)"
  gzip -f $O/start-*.log $O/xref14-ld-*.txt 2>/dev/null || true
  xlog "summary: raw snapshots -> raw-snaps.tar.gz ($(du -m $O/raw-snaps.tar.gz 2>/dev/null | cut -f1) MB); $O $(du -sm $O 2>/dev/null | cut -f1) MB; $(dfree) MB free on /"
  return 0
}
ngclean() {   # the end (or an early exit): ngram-shm-cleanup.sh (keeper stop + the tables and .lock files removed: FN items follow), logged (yswl's)
  local r b
  NGCLEAN=1
  run "docker rm -f $CN > /dev/null 2>&1"
  if [ "$DRY" = 1 ]; then echo "+ $NGS/ngram-shm-cleanup.sh   (keeper stop + $NGDIR/ngram-*.tab + .lock removed; refuses beside a radiance container; logs to runs/queue-changes.log) -- now: $(DRYRUN=1 $NGS/ngram-shm-cleanup.sh 2>&1 | tail -3 | tr '\n' ' ' | cut -c1-300)"; return 0; fi
  b=$(df -B1 --output=used /dev/shm 2>/dev/null | tail -1 | awk '{printf "%.1f GiB", $1 / 1073741824}')
  r=$("$NGS/ngram-shm-cleanup.sh" 2>&1 | tail -3 | tr '\n' ' ' | cut -c1-600)
  case "$r" in *REFUSED*) xlog "ngshare cleanup: VIRHE $r -- the tables stay in $NGDIR (run ngram-shm-cleanup.sh before the FN items)" ;;
    *) xlog "ngshare cleanup: $r | /dev/shm used $b -> $(df -B1 --output=used /dev/shm 2>/dev/null | tail -1 | awk '{printf "%.1f GiB", $1 / 1073741824}') | left in $NGDIR: $(ls $NGDIR 2>/dev/null | tr '\n' ' ' | cut -c1-200)" ;; esac
}
dstart() {   # the tables (no P0: yzint14c's, not re-benched), then the ONE fresh start (ngshare), int14 pinned right after it and read back
  local tag=xref14-n5
  diskok $tag || return 1
  tabs14 || return 1
  LEVENV="$LEVENV RADIANCE_DS41_GG_M16_FORMS=/stats/m16-forms.txt RADIANCE_FN4_AR_XPUSH_MIN=$XPMIN"   # A4's table + XPUSH's bound (coordinator 13.11.: 1048576, FN 5f0352a8)
  LEVENV=$(echo "$LEVENV" | sed "s#RADIANCE_DS41_GG_PF_TABLE=$GPTAB #RADIANCE_DS41_GG_PF_TABLE=$GPTAB,$A11ENT #")
  case "$LEVENV" in *"RADIANCE_DS41_GG_PF_TABLE=$GPTAB,$A11ENT "*) ;; *) xlog "$tag: VIRHE the ranged entries did not go into LEVENV -> NOT STARTED"; return 1 ;; esac
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  [ -n "${R14_MAXSEQS:-}" ] && export MAXSEQS=$R14_MAXSEQS
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on" PROFEVERY=
  export AQLSTAMP=$([ $STAMPN -gt 0 ] && echo $STAMPN)
  export NGRAM=ram=/ngram   # ngshare.sh's restart sets it too; here it shows in the start line
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV > $XS2_ENVFILE; rm -f $STATSDIR/stamps.csv*"; else printf '%s\n' $LEVENV > $XS2_ENVFILE; rm -f $STATSDIR/stamps.csv*; fi
  xlog "$tag: start (REFERENCE; ngshare: Y $(basename $Y), NGRAM=ram=/ngram, keeper; n-gram tables now $(ls $NGDIR/ngram-*.tab 2>/dev/null | wc -l) (0: this start fills them, ~308 s); $DESC) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $(pinj O | cut -c1-1100) | AQLSTAMP=${AQLSTAMP:-unset} | env: $(env | grep -E '^(RADIANCE_|R4D_)' | grep -vE '(KEY|TOKEN|SECRET)' | sort | tr '\n' ' ' | cut -c1-2800)"
  gate1 $tag 70 || { xlog "$tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  modes $tag || { xlog "$tag: VIRHE the modes are not the start's -> stopped"; down; return 1; }
  drafted $tag || { down; return 1; }
  readback $tag || { keeplog $tag-readback; down; return 1; }
  a11line $tag || { keeplog $tag-table; down; return 1; }
  a4line $tag || { keeplog $tag-a4table; down; return 1; }
  xpline $tag || { keeplog $tag-xpush; down; return 1; }
  vramstart $tag
  swnote $tag || { xlog "$tag: VIRHE the switches are not as expected -> stopped"; keeplog $tag-switch; down; return 1; }   # posts THE int14 PIN (arm O) and reads it back
  rankrb $tag || { keeplog $tag-pin; down; return 1; }
  identw $tag $WARM_REF_225
  [ "$DRY" = 1 ] || { : > $O/blocks.jsonl; : > $O/mb.jsonl; : > $O/c16cap.jsonl; }
  xlog "THE int14 PIN O (read back, fn4 + every rank): $(pinj O | cut -c1-1300) | B (c8 work-based set only) = $(armj B) | readings: 1 identity warm-up, a1 vs yzint14c a1-Z + ref12 a1, gs$GSC, A11 in range [$A11ID], staggered (ANY failure: STOP) | 2 1u essay 1 + $U1N, switched x$SWN | 3 c8 work-based (gens $C8GN) [$C8GORD], c8 35 s x$C8N (INFO), c16 fixed x$C16N, c16 CAP48 | 4 cold [$CDEPTHS] | 5 c4 (warm-up $C4WARMUP) x$C4N, c1 x$C1N"
  stopfile || r14reads
  capture $O/$tag-engine-full.log.gz
  xlog "$tag: at the end: $(i11now) | $(fn4get | python3 -c 'import json,sys; w=json.load(sys.stdin)["switches"]; print(" ".join("%s %s" % (k, w.get(k)) for k in ("ds41_bi", "gg_pf", "ar_xpush", "gg_gemv_m16", "place_swap_lent", "heat_min_gain_x100", "heat_hyst_x100")))' 2>/dev/null) | $(docker logs --since 6h $CN 2>&1 | grep -E 'lever A9 stats|lever BI stats' | tail -2 | sed 's/^radiance: //; s/^[A-Z]\[[0-9]\] //' | cut -c1-200 | tr '\n' '|') | $(docker logs --since 6h $CN 2>&1 | grep -cE '^E ') E lines | $(elapsed) min"
  down   # the stamp writer flushes at exit
  [ "$DRY" = 1 ] || sleep 3
  xlog "$tag: done$([ $STOPPED = 1 ] && echo " (STOPPED at the identity gate: no speed block ran)")"
}
trap 'down; [ $NGB = 1 ] && [ $NGCLEAN = 0 ] && ngclean' EXIT   # an early exit frees /dev/shm too (FN items follow)
trap 'exit 143' TERM INT
