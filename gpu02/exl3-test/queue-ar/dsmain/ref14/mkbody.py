#!/usr/bin/env python3
"""mkbody.py SRC PARTS DST -- ref14/dsm-ref14.sh from yzint14c/dsm-yzint14c.sh (SRC, 2a58559d as staged and run 12.11.): every edit an
exact, asserted replacement (tag yzi -> xref14, knobs YZI_ -> R14_; the P0s / lever arms / screen readings / summary out, the int14
pin as the base (arm O), the tables from the body dir, the reference read set + the c8 work-based B Z Z B set + ncmatch.py in).
PARTS = ref14/mkparts (header.sh: the module header; section.sh: ref14's own functions + dstart + the traps)."""
import os, re, sys

src, parts, dst = sys.argv[1:4]
s = open(src).read()
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


# ---------------------------------------------------------------- the tag: yzi -> xref14, YZI_ -> R14_ (path strings re-aimed first)
for a, b in (("yzint14c/yzsum.py", "ref14/r14sum.py"), ("yzint14c/mkbody.py", "ref14/mkbody.py"), ("dsm-yzint14c.sh", "dsm-ref14.sh"),
             ("$R/xdsm/yzint14c/x1", "$R/xdsm/ref14"), ("yzint14c/m16-forms.txt", "ref14/m16-forms.txt")):
    assert s.count(a) >= 1, a
    s = s.replace(a, b)
s = re.sub(r"(?<![A-Za-z])yzi(?![A-Za-z])", "xref14", s)
s = s.replace("YZI_", "R14_")
n_edits += 1
# ---------------------------------------------------------------- the header (everything above `set -u`)
cut("#!/bin/bash\n", "set -u\n", open(os.path.join(parts, "header.sh")).read())
# ---------------------------------------------------------------- knobs
rep("ITEM=${R14_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-yzint14c}", "ITEM=${R14_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zref14}")
rep("YDS=$HD/yzsum.py; NCT=$HD/../ync/nctab.py;", "YDS=$HD/r14sum.py; NCM=$HD/ncmatch.py; NCT=$HD/../ync/nctab.py;")
rep("CDEPTHS=${R14_CDEPTHS-136000}", "CDEPTHS=${R14_CDEPTHS-64000 136000}")
rep("BUDGET=${R14_BUDGET_MIN:-45}", "BUDGET=${R14_BUDGET_MIN:-35}")
rep("${R14_C4WIN:-20}", "${R14_C4WIN:-35}", 2)
rep("${R14_C1WIN:-20}", "${R14_C1WIN:-35}", 2)
rep("SARM=BASE   # yswl's blockx names the start arm (one start here)\n",
    "SARM=BASE   # yswl's blockx names the start arm (one start here)\n"
    "YZ=${R14_YZ:-$R/xdsm/yzint14c/x1}; YZA1=${R14_YZA1:-$YZ/yzi-a1-Z.jsonl}   # yzint14c's run (ALL arm Z = the provisional int14 zero point): its a1-Z and cold answers\n"
    "A11TAB=$HD/a11-entries.txt; A11MD5=${R14_A11MD5:-8e4fc84e}; A11NE=${R14_A11NE:-8}   # A11's ranged entries = yzint14c's P0 (runs/xdsm/yzint14c/x1/p0-tab.entries), copied, NOT re-benched\n"
    "C8GN=${R14_C8GN:-2}; C8GORD=${R14_C8GORDER-BZZB}; C8N=${R14_C8N:-2}; C16N=${R14_C16N:-2}; C4N=${R14_C4N:-2}; C1N=${R14_C1N:-1}; C4WARMUP=${R14_C4WARMUP:-1}; SWN=${R14_SWN:-2}   # the reading counts\n"
    "ESSMD5=${R14_ESSAY_MD5:-47a39dac581a}; CODEMD5=${R14_CODE_MD5:-c5e7c82e3e11}   # yzint14c's resprobe answers (text_md5; every arm, every read)\n"
    "for v in $C8GN $C8N $C16N $C4N $C1N $SWN; do case $v in ''|*[!0-9]*) echo \"a ref14 count knob is not a number ('$v')\"; exit 1 ;; esac; done\n"
    "case $C4WARMUP in 0|1) ;; *) echo \"R14_C4WARMUP 0 or 1\"; exit 1 ;; esac; [ $C8GN -ge 1 ] || { echo \"R14_C8GN >= 1\"; exit 1; }\n"
    "case $C8GORD in *[!BZ]*) echo \"R14_C8GORDER takes B (every int14 lever at 0 = int12 bytes) and Z (the int14 pin) (got '$C8GORD')\"; exit 1 ;; esac\n")
# ---------------------------------------------------------------- THE int14 PIN as the base (arm O); B = every int14 lever at 0; Z = O
rep('\\"exl3_x3d\\": 1, \\"ar_xpush\\": 0, \\"gg_gemv_m16\\": 0, \\"place_swap_lent\\": 0, \\"heat_min_gain_x100\\": 150, \\"heat_hyst_x100\\": 125}"\n[ $SW = 1 ] || MJ=',
    '\\"exl3_x3d\\": 1, \\"ds41_bi\\": 6, \\"gg_pf\\": 2, \\"ar_xpush\\": 1, \\"gg_gemv_m16\\": 1, \\"place_swap_lent\\": 1, \\"heat_min_gain_x100\\": $SWLG, \\"heat_hyst_x100\\": 125}"   # ref14: THE int14 PIN (yzint14c\'s Z)\n[ $SW = 1 ] || MJ=')
rep("""           # R14_SW=0 (the mock): ds41_shexp_lane 1 only (the mock's --fault differ hook for the stop path)
  [ $SW = 1 ] || { echo '{"ds41_shexp_lane": 1}'; return 0; }""",
    """           # R14_SW=0 (the mock): {} (ref14: Z = O, the int14 pin)
  [ $SW = 1 ] || { echo '{}'; return 0; }""")
rep('swlwant() { case $1 in W) echo "1 $SWLG" ;;', 'swlwant() { case $1 in O|W) echo "1 $SWLG" ;;')
line("    B) [ $SW = 1 ] && echo '{\"ds41_bi\": 2, \"gg_pf\": 1}'",
     "    B) [ $SW = 1 ] && echo '{\"ds41_bi\": 2, \"gg_pf\": 1, \"ar_xpush\": 0, \"gg_gemv_m16\": 0, \"place_swap_lent\": 0, \"heat_min_gain_x100\": 150, \"heat_hyst_x100\": 125}' || echo '{}' ;;   "
     "# ref14: B = every int14 lever switch at 0 = int12 bytes (the c8 work-based set's base); the base pin O is int14")
rep('O) echo "O (THE REFERENCE PIN)" ;;', 'O) echo "O (THE int14 PIN)" ;;')
line('DESC="DS BASE-CANDIDATE int14c',
     'DESC="THE int14 REFERENCE READ (ref14: image int14c, THE int14 PIN = ref12\'s pins + ar_xpush 1, gg_gemv_m16 1, ds41_bi 6, gg_pf 2, place_swap_lent 1, '
     'heat_min_gain_x100 $SWLG; A4 table + A11 ranged entries from yzint14c, not re-benched; O = int14, B = every int14 lever at 0 for the c8 work-based set only) '
     'NST 5 stack5 start (stack4\'s + RADIANCE_LOG_STEPS=1, E1 declared off, HB ranks $HBRANKS) + RADIANCE_DS41_MHC_FUSE=0 (M1 declared) + RADIANCE_DS41_GQ=3 + '
     'RADIANCE_DS41_E2=1 (1U-GQ / 1U-E2 declared) + RADIANCE_DS41_KV_REPLICATED=1 (KVR) $([ $STAMPN -gt 0 ] && echo "+ RADIANCE_AQL_STAMP=$STAMPN (card 0, every ${STAMPN}th pass)" '
     '|| echo "(no stamping)") | SP / ARC / CBH off, ngshare, no ATTACH | the pin = stack5\'s S5T with draft_sample 0 + SP 0 / 0 + $MJ + the fixed pin (ds41_psk_k $PSKK; '
     'draft_lend 1; dspark_conf 1, dspark_conf_rows 1)"')
# ---------------------------------------------------------------- mbx: sched G = the work-based gens window (coordinator 12.11. method fix)
rep('local wo="--window time"; [ $n = 8 ] && [ $C8GENS -gt 0 ] && wo="--window gens --gens $C8GENS --edge 0.5"',
    'local wo="--window time"; [ $sc = G ] && wo="--window gens --gens $C8GN --edge 0.5"   # ref14: sched G = the work-based window (gens x groups newcomer steps, decode-only edges)')
# ---------------------------------------------------------------- the pin check: O / Z = int14, B = int12's values
cut("for a in O M0 B A G C X W Z; do python3 -c 'import json,sys; d=json.loads(sys.argv[1]); sw = sys.argv[2] == \"1\"; a = sys.argv[3]\n",
    '[ "$(sha256sum $MODELDIR/$SIDECAR',
    """for a in O Z B; do python3 -c 'import json,sys; d=json.loads(sys.argv[1]); sw = sys.argv[2] == "1"; a = sys.argv[3]
assert d["ds41_psk"] == 4 and d["ds41_psk_cr"] == 1 and d["draft_sample"] == 0 and d["draft_lend"] == 1 and d["dspark_conf"] == 1 and d["dspark_conf_rows"] == 1
if sw:
    assert d["mhc_fuse"] == int(sys.argv[4]) and d["ix_lds"] == 0 and d["ix_ni"] == 0 and d["draft_depth0"] == int(sys.argv[5]) and d["sched_row_cap"] == int(sys.argv[6]) and d["stage_full_min"] == 0
    assert d["ds41_mix_wv"] == int(sys.argv[7]) and d["ds41_gq"] == 3 and d["ds41_e2"] == 1 and d["ds41_arcoded"] == 0 and d["exl3_x3d"] == 1
    assert d["sched_cb_hold_ms"] == 0 and d["sched_cb_hold_k"] == int(sys.argv[8]) and d["ds41_shexp_lane"] == int(sys.argv[9]) and "exl3_x3d_pf" not in d
    i14 = a in ("O", "Z"); g = int(sys.argv[10])
    assert d["ds41_bi"] == (6 if i14 else 2) and d["gg_pf"] == (2 if i14 else 1) and d["ar_xpush"] == (1 if i14 else 0) and d["gg_gemv_m16"] == (1 if i14 else 0)
    assert d["place_swap_lent"] == (1 if i14 else 0) and d["heat_min_gain_x100"] == (g if i14 else 150) and d["heat_hyst_x100"] == 125
else: assert d["draft_depth0"] == 1 and not any(k in d for k in ("gg_gemv_m16", "exl3_x3d_pf", "ar_xpush", "mhc_fuse", "ix_lds", "ix_ni", "ds41_arcoded", "stage_full_min", "sched_row_cap", "ds41_mix_wv", "ds41_gq", "ds41_e2", "sched_cb_hold_ms", "exl3_x3d", "place_swap_lent", "heat_min_gain_x100", "heat_hyst_x100"))' "$(pinj $a)" $SW $a $MHC $D0 $CAP $MXV $CBHK $F4V $SWLG 2>/dev/null || HMISS="$HMISS json:$a"; done
""")
# ---------------------------------------------------------------- helpers: the tables, ncmatch.py, the cross-run references
rep("$YDS $NCT $DQ $SPB0 $A4TAB $NGS/ngshare.sh", "$YDS $NCM $NCT $DQ $SPB0 $A4TAB $A11TAB $YZA1 $REF12A1 $NGS/ngshare.sh")
rep("$U1S $R12S $MBT2 $NCMB2 $C8MIX $YDS $NCT 2>&1", "$U1S $R12S $MBT2 $NCMB2 $C8MIX $YDS $NCM $NCT 2>&1")
rep('XPT=$(timeout 60 python3 -I -B $YDS selftest',
    '[ "$(md5sum < $A11TAB 2>/dev/null | cut -c1-8)" = "$A11MD5" ] && [ "$(tr -d \' \\n\' < $A11TAB 2>/dev/null | tr \',\' \'\\n\' | grep -c .)" = "$A11NE" ] || HMISS="$HMISS a11table:$A11TAB(want md5 $A11MD5, $A11NE entries)"\n'
    'NMT2=$(timeout 60 python3 -I -B $NCM selftest 2>&1 | tail -1); case "$NMT2" in "SELFTEST OK"*) ;; *) HMISS="$HMISS selftest:$NCM(${NMT2:0:300})" ;; esac\n'
    'XPT=$(timeout 60 python3 -I -B $YDS selftest')
rep("(yd2m's), yswl/resprobe.py", "(yzint14c's = yd2m's), ref14/a11-entries.txt $(md5sum < $A11TAB | cut -c1-8) (yzint14c's P0), ref14/ncmatch.py $(md5sum < $NCM | cut -c1-8), yswl/resprobe.py")
rep("(a copy of ync/dsm-ync.sh via ref14/mkbody.py)", "(a copy of yzint14c/dsm-yzint14c.sh 2a58559d via ref14/mkbody.py)")
rep("| yzsum.py $XPT |", "| r14sum.py $XPT | ncmatch.py $NMT2 |")
rep("| A4 table md5 $A4MD5 ($A4N lines) |", "| A4 table md5 $A4MD5 ($A4N lines), A11 entries md5 $A11MD5 ($A11NE) |")
line('FNS="stagx p0nc a11line p0xp',
     'FNS="stagx coldid tabs14 rankrb a11line a4line xpline r14id r14reads splitblk ncmset r14summary ngclean zarm swlwant log run mkd restart check down stopfile '
     'best_vars desc serve_args xlog elapsed armj NAMEOF pinj stackenv keeplog capture thermline dfree diskok cmpfile idrec identw modes vramstart fn4get fn4set nudge '
     'fn4code pcreset arm msnap mdelta pskstat ssnap sdelta gate1 alone cmpx budok c16x mwin ctrinfo i11now i11d i11g monons utcnow gpurun drafted readback swnote '
     'capd mbx cbhd cbhg c4warm dstart idok gt48"')
rep('  for a in O B A G C X W Z; do echo "+ arm $a ($(NAMEOF $a)): $(armj $a)"; done', '  for a in O B Z; do echo "+ arm $a ($(NAMEOF $a)): $(armj $a)"; done')
line('  echo "+ readings: 1 identity a1-Z',
     '  echo "+ readings: 1 identity warm-up vs stock, a1-O vs yzint14c a1-Z ($([ -s $YZA1 ] && echo present || echo MISSING)) + ref12 a1 ($([ -s $REF12A1 ] && echo present || echo MISSING)), '
     'gs$GSC-O (MAXSEQS ${R14_MAXSEQS:-base}), A11 in range [$A11ID] vs yzint14c Z ($(for k in $A11ID; do [ -s $YZ/cold-$k-2-Z.txt ] && echo -n "$k present " || echo -n "$k MISSING "; done)), '
     'staggered O vs a1-O (conc $LDCONC, $LDPF-token prefills) -- ANY failure STOPS before the speed blocks | 2 1u essay 1 + $U1N, switched x$SWN | 3 c8 work-based gens $C8GN [$C8GORD], '
     'c8 35 s x$C8N (INFO), c16 fixed x$C16N, c16 CAP48 | 4 cold [$CDEPTHS] (refs: yzint14c Z 136k $([ -s $YZ/cold-136000-2-Z.txt ] && echo present || echo MISSING), ref12 64k '
     '$([ -s $R/xdsm/ref12/cold-64000.txt ] && echo present || echo MISSING)) | 5 c4 (warm-up $C4WARMUP) x$C4N, c1 x$C1N | windows c8 / c4 / c1 $C8WIN / $C4WIN / $C1WIN s | '
     'budget $BUDGET min | four cards $(cat /sys/class/drm/card*/device/unique_id 2>/dev/null | wc -l) unique_id | n-gram tables in $NGDIR now: $(ls $NGDIR/ngram-*.tab 2>/dev/null | wc -l) | '
     'free RAM $(free -g | awk \'NR==2{print $7}\') GiB"')
# ---------------------------------------------------------------- the section: yzint14c's own functions -> ref14's
cut("# ---------------------------------------------------------------- int14c's own functions (P0 XPUSH / A4, identity + the stop, readings, summary)\n",
    "# ---------------------------------------------------------------- main\n", open(os.path.join(parts, "section.sh")).read())
# ---------------------------------------------------------------- main
line('xlog "$ITEM: start IMAGE=$IMAGE',
     'xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f \'{{.Id}}\' $IMAGE 2>/dev/null | cut -c8-19))$([ $SW = 1 ] || echo " | MOCK-RUN R14_SW=0 (NOT A READING): NO int9b .. int14c '
     'switch posted, cross-run references INFO") | THE int14 REFERENCE READ, ONE FRESH START on ngshare (no ATTACH), int14 pinned right after it (fn4) and read back on every rank: '
     '$DESC | MAXSEQS ${R14_MAXSEQS:-the base value} | tables: A4 $A4MD5 ($A4N lines), A11 $A11MD5 ($A11NE entries), NOT re-benched | 1 identity warm-up, a1-O vs yzint14c a1-Z + '
     'ref12 a1, gs$GSC-O, A11 in range [$A11ID], staggered (ANY failure: STOP before the speed blocks) | 2 1u essay 1 + $U1N, switched x$SWN | 3 c8 work-based gens $C8GN '
     '[$C8GORD] (B = every int14 lever at 0), c8 35 s x$C8N (INFO), c16 fixed x$C16N, c16 CAP48 | 4 cold [$CDEPTHS] | 5 c4 (warm-up $C4WARMUP) x$C4N, c1 x$C1N | budget $BUDGET '
     'min | no evict"')
rep("\nyzsummary\nunset $SWITCHES\n", "\nr14summary\nunset $SWITCHES\n")
for w in ("yzsummary", "yzreads", "p0xp", "p0a4", "p0json", "yzid ", "levreads", "allreads"):
    assert w not in s.split("# ---------------------------------------------------------------- main")[1], "main still names %s" % w
open(dst, "w").write(s)
os.chmod(dst, 0o755)
print("%s: %d edits, %d lines" % (dst, n_edits, s.count("\n")))
