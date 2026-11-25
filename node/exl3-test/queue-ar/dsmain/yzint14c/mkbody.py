#!/usr/bin/env python3
"""mkbody.py SRC YSWL PARTS DST -- yzint14c/dsm-yzint14c.sh from ync/dsm-ync.sh (SRC, 19b9148d): every edit an exact, asserted replacement
(tag ync -> yzi, knobs YNC_ -> YZI_; the A9 / A11 screen's arms / readings / summary out, int14c's in). YSWL = yswl/dsm-yswl.sh (its blockx:
the resprobe read with the residency / SWL rank checks); PARTS = yzint14c/mkparts (header.sh: the module header, section.sh: int14c's own
functions + dstart + the traps)."""
import os, re, sys

src, yswl, parts, dst = sys.argv[1:5]
s = open(src).read()
y = open(yswl).read()
n_edits = 0


def rep(old, new, count=1):
    global s, n_edits
    c = s.count(old)
    assert c == count, "%d x (want %d): %r" % (c, count, old[:200])
    s = s.replace(old, new)
    n_edits += 1


def line(prefix, new):
    """replace the one line starting with prefix (whole line) by new (no newline)"""
    global s, n_edits
    L = s.split("\n")
    hit = [i for i, l in enumerate(L) if l.startswith(prefix)]
    assert len(hit) == 1, "%d lines start with %r" % (len(hit), prefix[:120])
    L[hit[0]] = new
    s = "\n".join(L)
    n_edits += 1


def cut(start_marker, end_marker, new):
    """replace [start_marker, end_marker) with new (both markers must occur once)"""
    global s, n_edits
    assert s.count(start_marker) == 1, start_marker[:120]
    assert s.count(end_marker) == 1, end_marker[:120]
    i, j = s.index(start_marker), s.index(end_marker)
    assert i < j
    s = s[:i] + new + s[j:]
    n_edits += 1


# ---------------------------------------------------------------- the tag: ync -> yzi, YNC_ -> YZI_ (path strings kept / re-aimed first)
for a, b in (("ync/nctab.py", "@@NCT@@"), ("ync/ncsum.py", "yzint14c/yzsum.py"), ("ync/mkbody.py", "yzint14c/mkbody.py"),
             ("dsm-ync.sh", "dsm-yzint14c.sh"), ("$R/xdsm/ync/x1", "$R/xdsm/yzint14c/x1")):
    assert s.count(a) >= 1, a
    s = s.replace(a, b)
s = re.sub(r"(?<![A-Za-z])ync(?![A-Za-z])", "yzi", s)
s = s.replace("YNC_", "YZI_").replace("@@NCT@@", "ync/nctab.py")
n_edits += 1
# ---------------------------------------------------------------- the header (everything above `set -u`; after the tag rename: it names ync's files)
cut("#!/bin/bash\n", "set -u\n", open(os.path.join(parts, "header.sh")).read())

# ---------------------------------------------------------------- knobs
rep("ITEM=${YZI_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yync}", "ITEM=${YZI_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yzint14c}")
rep("YDS=$HD/ncsum.py; NCT=$HD/nctab.py;", "YDS=$HD/yzsum.py; NCT=$HD/../ync/nctab.py;")
rep("C48REF=${YZI_C48REF:-223.2}", "C48REF=${YZI_C48REF:-300.7}")
rep("CDEPTHS=${YZI_CDEPTHS-64000 136000}", "CDEPTHS=${YZI_CDEPTHS-136000}")
rep("BUDGET=${YZI_BUDGET_MIN:-42}", "BUDGET=${YZI_BUDGET_MIN:-45}")
rep('PFOK=0; A10=0; PDEPTHS=""   # (yd2m\'s lever A10: not in this item)\n',
    'PFOK=0; A10=0; PDEPTHS=""   # (yd2m\'s lever A10: not in this item)\n'
    'XPON=1; A4ON=1; IDFAIL=""; STOPPED=0   # int14c: a lever whose P0 fails is dropped (XPON / A4ON / A11ON 0); IDFAIL = the failed strict identities (THE STOP)\n'
    'SWLON=${YZI_SWL:-1}; SWLG=${YZI_SWLG:-75}; SWLREADS=${YZI_SWLREADS:-1}   # SWL in Z + its reads (coordinator 12.11. after yswl: L1G075 = place_swap_lent 1 + heat_min_gain_x100 75)\n'
    'C8ORD1=${YZI_C8ORDER1-BAGB}; C8ORD2=${YZI_C8ORDER2-BZZB}; U1N=${YZI_U1N:-3}; ALLEST=${YZI_ALLEST:-9}   # step 2 c8, step 3 c8, 1u reads an arm, step 3 minutes (kept when the SWL reads are weighed)\n'
    'A4TAB=$HD/m16-forms.txt; A4MD5=${YZI_A4MD5:-d55ad354}; A4N=${YZI_A4N:-95}; REF12A1=${YZI_REF12A1:-$R/xdsm/ref12/xref12-a1.jsonl}   # A4\'s table = yd2m\'s P0 bench (not re-benched); ref12\'s a1\n'
    'NGS=$HOME/gpu/ds41/exl3-test/queue-ar/tools/ngramshare; NGB=0; NGCLEAN=0; NGDIR=${NGSHARE_DIR:-/dev/shm/ds41ng}   # n-gram reuse (TESTING.md BODY RULE)\n'
    'SARM=BASE   # yswl\'s blockx names the start arm (one start here)\n'
    'XPMIN=${YZI_XPMIN:-1048576}; XPROWS=${YZI_XPROWS:-48,256,1024}   # XPUSH\'s bound (coordinator 13.11., FN 5f0352a8; RADIANCE_FN4_AR_XPUSH_MIN at start) and the P0 gate check\'s rows x 5120\n'
    'case $XPMIN in \'\'|*[!0-9]*) echo "YZI_XPMIN is an element count"; exit 1 ;; esac; case $XPROWS in *[!0-9,]*|\'\') echo "YZI_XPROWS rows, comma separated"; exit 1 ;; esac\n'
    '[ -n "$(echo $XPROWS | tr \',\' \'\\n\' | awk -v m=$XPMIN \'$1 * 5120 >= m\' | head -1)" ] && [ -n "$(echo $XPROWS | tr \',\' \'\\n\' | awk -v m=$XPMIN \'$1 * 5120 < m\' | head -1)" ] || { echo "YZI_XPROWS needs a size below and one above YZI_XPMIN (x 5120)"; exit 1; }\n'
    'case $SWLON$SWLREADS in [01][01]) ;; *) echo "YZI_SWL / YZI_SWLREADS take 0 or 1"; exit 1 ;; esac\n'
    'for v in $SWLG $U1N $ALLEST $A4N; do case $v in \'\'|*[!0-9]*) echo "a knob is not a number (\'$v\')"; exit 1 ;; esac; done\n'
    '[ $SWLG -le 10000 ] || { echo "YZI_SWLG 0..10000 (heat_min_gain_x100)"; exit 1; }\n'
    'case $C8ORD1 in *[!BAG]*|\'\') echo "YZI_C8ORDER1 takes B, A (A9) and G (A11) (got \'$C8ORD1\')"; exit 1 ;; esac\n'
    'case $C8ORD2 in *[!BZ]*|\'\') echo "YZI_C8ORDER2 takes B and Z (ALL) (got \'$C8ORD2\')"; exit 1 ;; esac\n')
rep("DQ=$QA/deepq.py; MBP=$D/mb.py; TESTS=$D/ds41_tests.py; SPB=$D/stepprobe.py",
    "DQ=$QA/deepq.py; MBP=$D/mb.py; TESTS=$D/ds41_tests.py; SPB=$HD/../yswl/resprobe.py; SPB0=$D/stepprobe.py   # resprobe = stepprobe + the answer text (yswl's, read in place)")
# ---------------------------------------------------------------- the base pin (+ the new switches at int12's values) and the arms
rep('\\"sched_cb_hold_k\\": $CBHK, \\"exl3_x3d\\": 1}"\n[ $SW = 1 ] || MJ=',
    '\\"sched_cb_hold_k\\": $CBHK, \\"exl3_x3d\\": 1, \\"ar_xpush\\": 0, \\"gg_gemv_m16\\": 0, \\"place_swap_lent\\": 0, \\"heat_min_gain_x100\\": 150, \\"heat_hyst_x100\\": 125}"\n[ $SW = 1 ] || MJ=')
rep("""    B) [ $SW = 1 ] && echo '{"ds41_bi": 2, "gg_pf": 1}' || echo '{}' ;;   # the nc arms (before C*: dspark_conf's): B = ref12's two values
    A) [ $SW = 1 ] && echo '{"ds41_bi": 6, "gg_pf": 1}' || echo '{}' ;;   # lever A9: one MoE all-reduce a segmented step (4 = 2 + the cheap issue)
    G) [ $SW = 1 ] && echo '{"ds41_bi": 2, "gg_pf": 2}' || echo '{}' ;;   # lever A11: the P0 table's ranged entries
    N) [ $SW = 1 ] && echo '{"ds41_bi": 6, "gg_pf": 2}' || echo '{}' ;;   # A9 + A11
    Q) [ $SW = 1 ] && echo '{"ds41_bi": 4, "gg_pf": 1}' || echo '{}' ;;   # attribution only (A's staggered check failed): the cheap issue without the one all-reduce
""", """    B) [ $SW = 1 ] && echo '{"ds41_bi": 2, "gg_pf": 1}' || echo '{}' ;;   # int14c's arms (before C*: dspark_conf's): B = ref12 / int12 (the base pin's new switches at int12's values)
    A) [ $SW = 1 ] && echo '{"ds41_bi": 6}' || echo '{}' ;;   # lever A9: one MoE all-reduce a segmented step (6 = 4 + the one all-reduce)
    G) [ $SW = 1 ] && echo '{"gg_pf": 2}' || echo '{}' ;;   # lever A11: the P0 table's ranged entries
    C) [ $SW = 1 ] && echo '{"gg_gemv_m16": 1}' || echo '{}' ;;   # lever A4: the A4 table's forms (/stats/m16-forms.txt)
    X) [ $SW = 1 ] && echo '{"ar_xpush": 1}' || echo '{}' ;;   # XPUSH (no read of its own here: its screen ran; in Z)
    W) [ $SW = 1 ] && echo "{\\"place_swap_lent\\": 1, \\"heat_min_gain_x100\\": $SWLG}" || echo '{}' ;;   # SWL's winner (yswl's L1G075)
    Z) zarm ;;   # ALL: every lever whose P0 passed
    Q) [ $SW = 1 ] && echo '{"ds41_bi": 4, "gg_pf": 1}' || echo '{}' ;;   # (kept from ync: ds41_bi 4, the cheap issue without the one all-reduce)
""")
rep("""armj() {   # armj ARM: the arm's own switch (JSON) on top of the base; under YZI_SW=0 only the int8 switches (F4, C1T)
""", """zarm() {   # Z = ALL: ar_xpush 1 (XPON), gg_gemv_m16 1 (A4ON), ds41_bi 6, gg_pf 2 (A11ON), place_swap_lent 1 + heat_min_gain_x100 SWLG (SWLON);
           # YZI_SW=0 (the mock): ds41_shexp_lane 1 only (the mock's --fault differ hook for the stop path)
  [ $SW = 1 ] || { echo '{"ds41_shexp_lane": 1}'; return 0; }
  python3 -c 'import json,sys
xp, a4, a11, swl, g = (int(x) for x in sys.argv[1:6])
d = {"ds41_bi": 6, "ar_xpush": 1 if xp else 0, "gg_gemv_m16": 1 if a4 else 0, "gg_pf": 2 if a11 else 1}
d.update({"place_swap_lent": 1, "heat_min_gain_x100": g} if swl else {"place_swap_lent": 0, "heat_min_gain_x100": 150})
print(json.dumps(d))' $XPON $A4ON $A11ON $SWLON $SWLG
}
swlwant() { case $1 in W) echo "1 $SWLG" ;; Z) [ $SWLON = 1 ] && echo "1 $SWLG" || echo "0 150" ;; *) echo "0 150" ;; esac; }   # ARM -> place_swap_lent heat_min_gain_x100 (yswl's blockx checks the ranks)
armj() {   # armj ARM: the arm's own switch (JSON) on top of the base; under YZI_SW=0 only the int8 switches (F4, C1T)
""")
rep("""  B) echo "B (ds41_bi 2 + gg_pf 1: int12 / ref12)" ;; A) echo "A (lever A9: ds41_bi 6)" ;; G) echo "G (lever A11: gg_pf 2, the P0 table's ranged entries)" ;;
  N) echo "N (A9 + A11: ds41_bi 6, gg_pf 2)" ;; Q) echo "Q (ds41_bi 4: A9 without its one all-reduce, attribution only)" ;; C*)""",
    """  B) echo "B (int12 / ref12: ds41_bi 2, gg_pf 1, ar_xpush 0, gg_gemv_m16 0, place_swap_lent 0, heat_min_gain_x100 150)" ;; A) echo "A (lever A9: ds41_bi 6)" ;; G) echo "G (lever A11: gg_pf 2, the P0 table's ranged entries)" ;;
  C) echo "C (lever A4: gg_gemv_m16 1, yd2m's table)" ;; X) echo "X (XPUSH: ar_xpush 1)" ;; W) echo "W (SWL: place_swap_lent 1 + heat_min_gain_x100 $SWLG)" ;;
  Z) echo "Z (ALL: $(zarm | tr -d '{}\\"'))" ;; Q) echo "Q (ds41_bi 4: A9 without its one all-reduce, attribution only)" ;; C*)""")
line('DESC="A9 / A11 NEWCOMER SCREEN',
     'DESC="DS BASE-CANDIDATE int14c (int12 + XPUSH ar_xpush + A4 gg_gemv_m16 + A9 ds41_bi 6 + A11 gg_pf 2 + SWL place_swap_lent / heat_min_gain_x100; '
     'arms B / A / G / C / W / Z = ALL in-process) NST 5 stack5 start (stack4\'s + RADIANCE_LOG_STEPS=1, E1 declared off, HB ranks $HBRANKS) + '
     'RADIANCE_DS41_MHC_FUSE=0 (M1 declared) + RADIANCE_DS41_GQ=3 + RADIANCE_DS41_E2=1 (1U-GQ / 1U-E2 declared) + RADIANCE_DS41_KV_REPLICATED=1 (KVR) '
     '$([ $STAMPN -gt 0 ] && echo "+ RADIANCE_AQL_STAMP=$STAMPN (card 0, every ${STAMPN}th pass)" || echo "(no stamping)") | SP / ARC / CBH off, ngshare, no ATTACH | '
     'reference pin = stack5\'s S5T with draft_sample 0 + SP 0 / 0 + $MJ + the fixed pin (ds41_psk_k $PSKK; draft_lend 1; dspark_conf 1, dspark_conf_rows 1)"')
rep("RADIANCE_DS41_GG_M16 RADIANCE_DS41_GG_M16_FORMS RADIANCE_DS41_GG_PF_TABLE\"",
    "RADIANCE_DS41_GG_M16 RADIANCE_DS41_GG_M16_FORMS RADIANCE_DS41_GG_PF_TABLE RADIANCE_PLACE_SWAP_LENT RADIANCE_HEAT_MIN_GAIN RADIANCE_HEAT_HYSTERESIS RADIANCE_FN4_AR_XPUSH_MIN\"")
# ---------------------------------------------------------------- readback / swnote: the new switches
rep("""grep -c '^RADIANCE_DS41_ARCODED=\\|^RADIANCE_SCHED_CB_HOLD_\\|^RADIANCE_FN4_AR_XPUSH=\\|^RADIANCE_DS41_GG_M16=')""",
    """grep -c '^RADIANCE_DS41_ARCODED=\\|^RADIANCE_SCHED_CB_HOLD_\\|^RADIANCE_FN4_AR_XPUSH=\\|^RADIANCE_DS41_GG_M16=\\|^RADIANCE_PLACE_SWAP_LENT=\\|^RADIANCE_HEAT_MIN_GAIN=\\|^RADIANCE_HEAT_HYSTERESIS=')""")
rep("ARCODED / SCHED_CB_HOLD_* / FN4_AR_XPUSH / GG_M16 $ac (want 0: CBH and A4 come by fn4)",
    "ARCODED / SCHED_CB_HOLD_* / FN4_AR_XPUSH / GG_M16 / PLACE_SWAP_LENT / HEAT_MIN_GAIN / HEAT_HYSTERESIS $ac (want 0: CBH, XPUSH, A4 and SWL come by fn4)")
rep("local s0 g c1=400 c2=400 c3=400 c5=400 c6=400 c7=400 c8=400 c9=400 c10=400 c11=400 c4 a",
    "local s0 g c1=400 c2=400 c3=400 c5=400 c6=400 c7=400 c8=400 c9=400 c10=400 c11=400 c12=400 c13=400 c14=400 c15=400 c4 a")
rep("""c10=$(fn4code '{"ds41_bi": 7}'); c11=$(fn4code '{"gg_pf": 3}'); fi""",
    """c10=$(fn4code '{"ds41_bi": 7}'); c11=$(fn4code '{"gg_pf": 3}'); c12=$(fn4code '{"ar_xpush": 2}'); c13=$(fn4code '{"gg_gemv_m16": 2}'); c14=$(fn4code '{"place_swap_lent": 2}'); c15=$(fn4code '{"heat_min_gain_x100": 10001}'); fi""")
rep('("ds41_bi", "gg_pf", "exl3_x3d",', '("ds41_bi", "gg_pf", "ar_xpush", "gg_gemv_m16", "place_swap_lent", "heat_min_gain_x100", "heat_hyst_x100", "exl3_x3d",')
rep("ds41_bi 7 -> HTTP $c10, gg_pf 3 -> HTTP $c11\"",
    "ds41_bi 7 -> HTTP $c10, gg_pf 3 -> HTTP $c11, ar_xpush 2 -> HTTP $c12, gg_gemv_m16 2 -> HTTP $c13, place_swap_lent 2 -> HTTP $c14, heat_min_gain_x100 10001 -> HTTP $c15\"")
rep('w.get("ds41_bi") == 0 and w.get("gg_pf") == 1',
    'w.get("ds41_bi") == 0 and w.get("gg_pf") == 1\n    ok = ok and w.get("ar_xpush") == 0 and w.get("gg_gemv_m16") == 0 and w.get("place_swap_lent") == 0 and w.get("heat_min_gain_x100") == 150 and w.get("heat_hyst_x100") == 125')
rep('[ "$c11" = 400 ] && [ -z "$bad" ]', '[ "$c11" = 400 ] && [ "$c12" = 400 ] && [ "$c13" = 400 ] && [ "$c14" = 400 ] && [ "$c15" = 400 ] && [ -z "$bad" ]')
rep("/ ds41_bi 7 / gg_pf 3\") -> 400; sched_cb_hold_ms 0 / k 2, exl3_x3d 0, ds41_bi 0, gg_pf 1",
    "/ ds41_bi 7 / gg_pf 3 / ar_xpush 2 / gg_gemv_m16 2 / place_swap_lent 2 / heat_min_gain_x100 10001\") -> 400; sched_cb_hold_ms 0 / k 2, exl3_x3d 0, ds41_bi 0, gg_pf 1, ar_xpush 0, gg_gemv_m16 0, place_swap_lent 0, heat_min_gain_x100 150, heat_hyst_x100 125")
# ---------------------------------------------------------------- blockx: yswl's (resprobe + the residency / SWL rank checks), tag xswl -> yzi
i, j = y.index("NB=0; SETN=\"\"; SETK=0; SETS=0; C8S=\"\"\nblockx() {"), y.index("coldx() {   # coldx DEPTH: deepq's")
yb = y[i:j].replace("RESULT xswl ", "RESULT yzi ").replace("XSW_SW", "YZI_SW")
assert "xswl" not in yb and "XSW_" not in yb, "yswl blockx keeps a tag"
cut("NB=0\nblockx() {", "coldx() {   # coldx DEPTH [ARM I]", yb)
# ---------------------------------------------------------------- c16x: an arm and an index, a row into c16cap.jsonl
cut("c16x() {   # c16x: ONE c16 block", "p1reads() {", r'''c16x() {   # c16x I ARM: ONE c16 block, EXACTLY the CAP48 screen's phase mb16 (cap48/dsm-cap48.sh: arm, prefix-cache reset, snapshots, its runner
          # cap48/ncmb.py --mode phase --temperature 0 --n 16 --dur 30 --warm 10) under ARM -> agg, decode_share, the cap counters -> $O/c16cap.jsonl
  local i=${1:-1} a=${2:-O} w=c16cap-${1:-1}-${2:-O} t00=$(date +%s) j
  arm $a || return 1
  pcreset $w; ssnap $w-0
  run "(cd $D && timeout $(( C16DUR + 1200 )) python3 -I -B $C48N --mb $MBP --mode phase --temperature 0 --n 16 --tag yzi-$w --dur $C16DUR --warm $C16WARM --reqs $O/c8/$w-reqs.jsonl --stats-prefix $O/c8/$w-st --samples $O/c8/$w-samp.jsonl > $O/c8/$w.txt 2>&1)"
  ssnap $w-1
  [ "$DRY" = 1 ] && { echo "+ RESULT yzi c16 $w (CAP48 method under $(NAMEOF $a) $(armj $a): cap48/ncmb.py phase c16, T 0, dur $C16DUR warm $C16WARM, MAXSEQS ${MAXSEQS:-?}, sched_row_cap $CAP): agg tok/s, decode_share, ms / tok a decode step, cap steps_gt48 (GET cap) -> $O/c16cap.jsonl; int12 prof1 $C48REF INFO"; return 0; }
  j=$(grep -m1 '^MB ' $O/c8/$w.txt | cut -c4-)
  [ -n "$j" ] || { xlog "c16 $w: VIRHE request error: no MB line ($(tail -3 $O/c8/$w.txt | tr '\n' ' ' | cut -c1-300))"; return 1; }
  python3 -c 'import json,sys
d = json.loads(sys.argv[1]); d.update(name=sys.argv[2], arm=sys.argv[3], gt48=sys.argv[4])
print(json.dumps(d))' "$j" $w $a "$(gt48 $w)" >> $O/c16cap.jsonl
  xlog "c16 $w (CAP48 method: cap48/ncmb.py phase c16, T 0, dur $C16DUR warm $C16WARM, MAXSEQS ${MAXSEQS:-?}, $(NAMEOF $a) $(armj $a), sched_row_cap $CAP, $(( $(date +%s) - t00 )) s): $(echo "$j" | python3 -c 'import json,sys
d = json.loads(sys.stdin.read()); ref = float(sys.argv[1])
a = d.get("agg_dec_tps")
print("agg %s tok/s (INFO vs int12 prof1 %.1f: %s), decode_share %s, %s tok / %s ms a decode step, acceptance %s, steps %s, %s errors%s" % (a, ref, ("%+.1f %%" % (100.0 * (a - ref) / ref)) if a else "-",
      d.get("decode_share"), d.get("tok_per_decode_step"), d.get("ms_decode_step"), d.get("acceptance"), d.get("steps"), d.get("errors"), "" if not d.get("errors") else " -- VIRHE request errors"))' $C48REF 2>&1 | cut -c1-500) | cap $(capd $w-0 $w-1) | steps_gt48 $(gt48 $w) | $(thermline $t00 $(date +%s))"
}
''')
# ---------------------------------------------------------------- p0nc's words: A11 out = arm G and Z's gg_pf 2
rep("(arms G / N)", "(arm G, Z's gg_pf 2)", 2)
rep("A11ON=0 (arms G / N dropped;", "A11ON=0 (arm G and Z's gg_pf 2 dropped;")
rep("-> arms G / N not run (the bench is A11's reading)", "-> arm G not run, Z keeps gg_pf 1 (the bench is A11's reading)")
# ---------------------------------------------------------------- int14c's own section (identity + the stop, readings, summary, dstart, traps)
cut("idok() {   # idok KEY ...: each identity.txt line yzi-KEY", "# ---------------------------------------------------------------- main\n",
    open(os.path.join(parts, "section.sh")).read())
# ---------------------------------------------------------------- main
line('xlog "$ITEM: start IMAGE=$IMAGE',
     'xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f \'{{.Id}}\' $IMAGE 2>/dev/null | cut -c8-19))$([ $SW = 1 ] || echo " | MOCK-RUN YZI_SW=0 (NOT A READING): '
     'NO int9b .. int14c switch posted (Z posts ds41_shexp_lane 1: the mock\'s differ hook), the counters INFO (a mock run only)") | DS BASE-CANDIDATE int14c SCREEN, ONE '
     'FRESH START on ngshare (no ATTACH), ref12\'s pins: $DESC | MAXSEQS ${YZI_MAXSEQS:-the base value} | P0: XPUSH four cards by UUID (ar_twoshot / ar_oneshot '
     'xpush 0 / 1), A4 --mrcheck16 with yd2m\'s table ($A4MD5, $A4N lines) and A11 --gpcheck M $A11CMS + --perfgp M $A11MS -> nctab.py (one card $P0UUID); '
     'a P0 failure drops that lever only | Z = ALL (SWL $SWLON: heat_min_gain_x100 $SWLG) | 1 identity a1-Z / a1-B vs ref12, gs$GSC-Z, A11 in range [$A11ID], '
     'staggered Z + A (ANY failure: STOP before the speed blocks) | 2 c8 [$C8ORD1] ${C8WIN} s, A4 c16 fixed + CAP48 B / C, SWL reads B / W | 3 c8 [$C8ORD2], '
     'c16 Z, 1u essay B / Z (1 + $U1N), cold [$CDEPTHS] B / Z | decode guard +$DECTH % | budget $BUDGET min | no evict"')
cut('IMGLITS13="A|lever A9 stats', 'PLUGMD5=', '''IMGLITS13="A|lever A9 stats
A|segmented steps with one MoE all-reduce
K|lever A11 (GPF-1k)
K|type:NxK@LO-HI=form
K|r4d_gg_pf_rule
R|--gpcheck
R|--perfgp
R|GPVERDICT
B|exl3_x3d, ar_xpush, gg_gemv_m16, place_swap_lent, heat_min_gain_x100, heat_hyst_x100, epoch_bump)
B|ar_xpush
B|gg_gemv_m16
B|place_swap_lent
B|heat_min_gain_x100
B|heat_hyst_x100
B|RADIANCE_PLACE_SWAP_LENT
B|RADIANCE_HEAT_MIN_GAIN
K|ar_xpush
K|RADIANCE_FN4_AR_XPUSH
K|RADIANCE_FN4_AR_XPUSH_MIN
K|ar_xpush's XPUSH form from %lld elements
K|gg_gemv_m16
K|RADIANCE_DS41_GG_M16_FORMS
K|lever A4 (gg_gemv_m16): %d forms from %s
R|ar_xpush_perf
R|--mrcheck16
R|A4VERDICT
P|int14c-PROVENANCE.txt"
''')
rep("is not the nc (ds41-nc) image -- missing:", "is not the int14c image -- missing:")
rep("nc literals (deepseek41.so's A9 stats line, libr4d.so's A11 line / ranged-entry syntax / r4d_gg_pf_rule, r4d_selftest --gpcheck / --perfgp / GPVERDICT, nc-PROVENANCE)",
    "int14c literals (deepseek41.so's A9 stats line, libr4d.so's A11 line / ranged-entry syntax / r4d_gg_pf_rule, ar_xpush / RADIANCE_FN4_AR_XPUSH, gg_gemv_m16 / the A4 table loader, the registry tail ending in heat_hyst_x100, SWL's env words, r4d_selftest --gpcheck / --perfgp / --mrcheck16 / ar_xpush_perf, int14c-PROVENANCE)")
rep("$YDS $NCT $DQ; do [ -s \"$f\" ]", "$YDS $NCT $DQ $SPB0 $A4TAB $NGS/ngshare.sh $NGS/ds41-ngshare-tp4.yaml $NGS/ngram-keeper.sh $NGS/ngram-shm-cleanup.sh $NGS/README.md; do [ -s \"$f\" ]")
rep('XPT=$(timeout 60 python3 -I -B $YDS selftest',
    'RPT=$(timeout 60 python3 -I -B $SPB selftest 2>&1 | tail -1); case "$RPT" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$SPB(${RPT:0:300})" ;; esac\n'
    '[ "$(md5sum < $A4TAB 2>/dev/null | cut -c1-8)" = "$A4MD5" ] && [ "$(grep -c . $A4TAB 2>/dev/null)" = "$A4N" ] || HMISS="$HMISS a4table:$A4TAB(want md5 $A4MD5, $A4N lines)"\n'
    'XPT=$(timeout 60 python3 -I -B $YDS selftest')
line('FNS="stagx p0nc a11line ncarms ncid splitblk pickbest ncreads ncsummary ',
     'FNS="stagx p0nc a11line p0xp p0a4 a4line xpline p0json yzid levreads allreads yzreads splitblk yzsummary ngclean zarm swlwant log run mkd restart check down stopfile '
     'best_vars desc serve_args xlog elapsed armj NAMEOF pinj stackenv keeplog capture thermline dfree diskok cmpfile idrec identw modes vramstart fn4get fn4set nudge '
     'fn4code pcreset arm msnap mdelta pskstat ssnap sdelta gate1 alone cmpx budok c16x p1reads mwin idx16 ctrinfo i11now i11d i11g monons utcnow gpurun drafted '
     'readback swnote capd mbx cbhd cbhg mbdelta c4warm c8mixall c8run mbsec blockx coldx refreads gsx summary dstart idok gt48"')
rep("for a in O M0 B A G N; do python3 -c 'import json,sys;", "for a in O M0 B A G C X W Z; do python3 -c 'import json,sys;")
rep("f4 = int(sys.argv[9])\n", "f4 = int(sys.argv[9])\nxp, a4, a11, swl, g = (int(x) for x in sys.argv[10:15])\n")
rep('''    assert d["ds41_bi"] == (6 if a in ("A", "N") else 2) and d["gg_pf"] == (2 if a in ("G", "N") else 1) and "gg_gemv_m16" not in d and "exl3_x3d_pf" not in d''',
    '''    z = a == "Z"
    assert d["ds41_bi"] == (6 if a in ("A", "Z") else 2) and d["gg_pf"] == (2 if a == "G" or (z and a11) else 1) and "exl3_x3d_pf" not in d
    assert d["ar_xpush"] == (1 if a == "X" or (z and xp) else 0) and d["gg_gemv_m16"] == (1 if a == "C" or (z and a4) else 0)
    assert d["place_swap_lent"] == (1 if a == "W" or (z and swl) else 0) and d["heat_min_gain_x100"] == (g if a == "W" or (z and swl) else 150) and d["heat_hyst_x100"] == 125''')
rep('''"sched_cb_hold_ms", "exl3_x3d"))' "$(pinj $a)" $SW $a $MHC $D0 $CAP $MXV $CBHK $F4V 2>/dev/null''',
    '''"sched_cb_hold_ms", "exl3_x3d", "place_swap_lent", "heat_min_gain_x100", "heat_hyst_x100"))' "$(pinj $a)" $SW $a $MHC $D0 $CAP $MXV $CBHK $F4V $XPON $A4ON $A11ON $SWLON $SWLG 2>/dev/null''')
rep("|RADIANCE_FN4_AR_XPUSH|RADIANCE_DS41_GG_M16|X3D_PF' $Y 2>/dev/null && HMISS=",
    "|RADIANCE_FN4_AR_XPUSH|RADIANCE_DS41_GG_M16|X3D_PF|RADIANCE_PLACE_SWAP_LENT|RADIANCE_HEAT_MIN_GAIN|RADIANCE_HEAT_HYST' $Y $NGS/ds41-ngshare-tp4.yaml 2>/dev/null && HMISS=")
rep('case "$LEVENV" in *RADIANCE_DS41_MHC_FUSE=0*',
    '''for f in $NGS/ngshare.sh $NGS/ngram-keeper.sh $NGS/ngram-shm-cleanup.sh; do bash -n $f 2>/dev/null || HMISS="$HMISS syntax:$f"; done
grep -q 'ds41-stack2-tp4.yaml' $NGS/ngshare.sh && grep -q '/dev/shm/ds41ng:/ngram' $NGS/ds41-ngshare-tp4.yaml || HMISS="$HMISS ngshare:not-the-README's-opt-in"
IMAGE=$IMAGE MODEL=$M225 MODELDIR=$MODELDIR STATSDIR=$STATSDIR ARGS=x XS2_ENVFILE=/dev/null docker compose --project-directory $C -f $NGS/ds41-ngshare-tp4.yaml config --quiet > /dev/null 2>&1 \\
  || HMISS="$HMISS yaml:$NGS/ds41-ngshare-tp4.yaml(docker compose config refuses it)"
case "$LEVENV" in *RADIANCE_DS41_MHC_FUSE=0*''')
rep("*RADIANCE_FN4_AR_XPUSH*|*RADIANCE_DS41_GG_M16*|*@*) HMISS=", "*RADIANCE_FN4_AR_XPUSH*|*RADIANCE_DS41_GG_M16*|*RADIANCE_PLACE_SWAP_LENT*|*RADIANCE_HEAT_MIN_GAIN*|*RADIANCE_HEAT_HYST*|*@*) HMISS=")
rep("levenv:ARC-CAP-CBH-XPUSH-A4-or-ranged-at-start (A11's entries are added by dstart after P0)",
    "levenv:ARC-CAP-CBH-XPUSH-A4-SWL-or-ranged-at-start (A11's entries and A4's table are added by dstart after P0)")
rep('HMD5="yzint14c/yzsum.py $(md5sum < $YDS | cut -c1-8),',
    'HMD5="yzint14c/yzsum.py $(md5sum < $YDS | cut -c1-8), yzint14c/m16-forms.txt $(md5sum < $A4TAB | cut -c1-8) (yd2m\'s), yswl/resprobe.py $(md5sum < $SPB | cut -c1-8), '
    'ngramshare: ngshare.sh $(md5sum < $NGS/ngshare.sh | cut -c1-8) / yaml $(md5sum < $NGS/ds41-ngshare-tp4.yaml | cut -c1-8) / keeper $(md5sum < $NGS/ngram-keeper.sh | cut -c1-8) / cleanup $(md5sum < $NGS/ngram-shm-cleanup.sh | cut -c1-8),')
rep("stepprobe.py $(md5sum < $SPB | cut -c1-8)", "stepprobe.py $(md5sum < $SPB0 | cut -c1-8)")
rep("(a copy of yd2m/dsm-yd2m.sh via yzint14c/mkbody.py)", "(a copy of ync/dsm-ync.sh via yzint14c/mkbody.py)")
rep("ncsum.py $XPT | nctab.py $NTT", "yzsum.py $XPT | nctab.py $NTT | resprobe.py $RPT | A4 table md5 $A4MD5 ($A4N lines) | ngramshare scripts parse, its yaml accepted")
rep('  for a in O B A G N; do echo "+ arm $a ($(NAMEOF $a)): $(armj $a)"; done', '  for a in O B A G C X W Z; do echo "+ arm $a ($(NAMEOF $a)): $(armj $a)"; done')
line('  echo "+ readings: identity a1 / gs$GSC',
     '  echo "+ readings: 1 identity a1-Z / a1-B vs ref12 a1 ($([ -s $REF12A1 ] && echo present || echo MISSING)), gs$GSC-Z (MAXSEQS ${YZI_MAXSEQS:-base}), A11 in range [$A11ID] B / Z, '
     'staggered Z vs a1-Z + A vs a1-B (conc $LDCONC, $LDPF-token prefills) -- ANY failure STOPS before the speed blocks | 2 mb c8 [$C8ORD1] ${C8WIN} s, A4 c16 fixed + CAP48 B / C, '
     'SWL reads B / W ($([ $SWLON = 1 ] && [ $SWLREADS = 1 ] && echo on || echo off)) | 3 c8 [$C8ORD2], c16 Z, 1u essay B / Z (1 + $U1N), cold [$CDEPTHS] B / Z | decode guard +$DECTH % | '
     'P0 card $P0UUID ($(cat /sys/class/drm/card*/device/unique_id 2>/dev/null | grep -c "^${P0UUID#GPU-}$") sysfs match), four cards $(cat /sys/class/drm/card*/device/unique_id 2>/dev/null | wc -l) unique_id | '
     'n-gram tables in $NGDIR now: $(ls $NGDIR/ngram-*.tab 2>/dev/null | wc -l) | free RAM $(free -g | awk \'NR==2{print $7}\') GiB"')
cut("mkd $O $O/guard $O/c8 $O/snap $STATSDIR\n", "\nncsummary\n", '''mkd $O $O/guard $O/c8 $O/snap $STATSDIR
[ "$DRY" = 1 ] || : > $O/identity.txt
if ! stopfile; then
  . $NGS/ngshare.sh || { xlog "VIRHE ngshare.sh did not load (Y $Y) -- not started"; exit 1; }
  NGB=1
  xlog "ngshare.sh sourced -- Y $(basename $Y), restart() wrapped (keeper prep before, keeper start --wait 120 --expect 2 after); n-gram tables before the start: $(ls $NGDIR/ngram-*.tab 2>/dev/null | wc -l) ($([ "$(ls $NGDIR/ngram-*.tab 2>/dev/null | wc -l)" -gt 0 ] && echo "reused if the probes match" || echo "this start fills them, ~308 s"))"
  dstart || true
fi
yzsummary
unset $SWITCHES
down
[ $NGB = 1 ] && ngclean
xlog "$ITEM: done ($(elapsed) min) -- $O | $([ $STOPPED = 1 ] && echo "STOPPED at the identity gate (no speed block ran): NEEDS USER | ")$([ $NV = 0 ] && echo "no VIRHE line" || echo "$NV VIRHE line(s): see above")"''')
rep("\nncsummary\n", "\n")
assert "ncsummary\n" not in s.split("# ---------------------------------------------------------------- main")[1], "ncsummary still called in main"
open(dst, "w").write(s)
os.chmod(dst, 0o755)
print("%s: %d edits, %d lines" % (dst, n_edits, s.count("\n")))
