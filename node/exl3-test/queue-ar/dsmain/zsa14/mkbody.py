#!/usr/bin/env python3
"""mkbody.py SRC PARTS DST [SRCMD5] -- zsa14/dsm-zsa14.sh from ref14/dsm-ref14.sh (SRC, md5 SRCMD5 as staged 12.11.): every edit an exact,
asserted replacement (tag xref14 -> xsa14, knobs R14_ -> SA14_; the arms B / S1 / S2 of lever SA on the int14 pin; ref14's identity /
readings / summary functions out, zsa14's in). PARTS = zsa14/mkparts (header.sh: the module header; funcs.sh: zsa14's identity, readings
and summary functions, in place of ref14's coldid .. r14summary; dstart.sh: zsa14's start, in place of ref14's)."""
import hashlib, os, re, sys

src, parts, dst = sys.argv[1:4]
want = sys.argv[4] if len(sys.argv) > 4 else None
raw = open(src, "rb").read()
md5 = hashlib.md5(raw).hexdigest()[:8]
assert want is None or md5 == want, "SRC md5 %s, want %s" % (md5, want)
s = raw.decode()
n_edits = 0
P = lambda n: open(os.path.join(parts, n)).read()


def rep(old, new, count=1):
    global s, n_edits
    c = s.count(old)
    assert c == count, "%d x (want %d): %r" % (c, count, old[:200])
    s = s.replace(old, new)
    n_edits += 1


def line(prefix, new):
    global s, n_edits
    L = s.split("\n")
    hit = [i for i, l in enumerate(L) if l.startswith(prefix)]
    assert len(hit) == 1, "%d lines start with %r" % (len(hit), prefix[:120])
    L[hit[0]] = new
    s = "\n".join(L)
    n_edits += 1


def cut(start_marker, end_marker, new):
    global s, n_edits
    assert s.count(start_marker) == 1, start_marker[:120]
    assert s.count(end_marker) == 1, end_marker[:120]
    i, j = s.index(start_marker), s.index(end_marker)
    assert i < j
    s = s[:i] + new + s[j:]
    n_edits += 1


# ---------------------------------------------------------------- before the tag change: the strings that name ref14's own files / provenance
rep("dsm-ref14.sh $(md5sum < $0 | cut -c1-8) (a copy of yzint14c/dsm-yzint14c.sh 2a58559d via ref14/mkbody.py)",
    "dsm-zsa14.sh $(md5sum < $0 | cut -c1-8) (a copy of ref14/dsm-ref14.sh %s via zsa14/mkbody.py)" % md5)
rep('HMD5="ref14/r14sum.py $(md5sum < $YDS | cut -c1-8), ref14/m16-forms.txt $(md5sum < $A4TAB | cut -c1-8) (yzint14c\'s = yd2m\'s), '
    'ref14/a11-entries.txt $(md5sum < $A11TAB | cut -c1-8) (yzint14c\'s P0), ref14/ncmatch.py $(md5sum < $NCM | cut -c1-8), ',
    'HMD5="zsa14/sa14sum.py $(md5sum < $YDS | cut -c1-8), zsa14/steps.py $(md5sum < $STP | cut -c1-8) (the coordinator\'s), zsa14/m16-forms.txt '
    '$(md5sum < $A4TAB | cut -c1-8) (= ref14\'s = yzint14c\'s), zsa14/a11-entries.txt $(md5sum < $A11TAB | cut -c1-8) (= ref14\'s = yzint14c\'s P0), '
    'zsa14/ncmatch.py $(md5sum < $NCM | cut -c1-8) (ref14\'s, copied), ')
# ---------------------------------------------------------------- the header (everything above `set -u`)
cut("#!/bin/bash\n", "set -u\n", P("header.sh").replace("REFMD5 below", "md5 %s" % md5))
# ---------------------------------------------------------------- the tag: xref14 -> xsa14, R14_ -> SA14_, the out dir
for a, b in (("dsm-ref14.sh", "dsm-zsa14.sh"), ("$R/xdsm/ref14", "$R/xdsm/zsa14")):
    assert s.count(a) >= 1, a
    s = s.replace(a, b)
s = re.sub(r"(?<![A-Za-z])xref14(?![A-Za-z0-9])", "xsa14", s)
s = s.replace("R14_", "SA14_")
assert "xref14" not in s and "R14_" not in s.replace("SA14_", "")
n_edits += 1
# ---------------------------------------------------------------- knobs
rep("ITEM=${SA14_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zref14}", "ITEM=${SA14_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zsa14}")
rep("YDS=$HD/r14sum.py; NCM=$HD/ncmatch.py;", "YDS=$HD/sa14sum.py; SAS=$HD/sa14sum.py; STP=$HD/steps.py; NCM=$HD/ncmatch.py;")
rep("BUDGET=${SA14_BUDGET_MIN:-35}", "BUDGET=${SA14_BUDGET_MIN:-28}")
rep("ESSMD5=${SA14_ESSAY_MD5:-47a39dac581a}; CODEMD5=${SA14_CODE_MD5:-c5e7c82e3e11}   # yzint14c's resprobe answers (text_md5; every arm, every read)\n",
    "ESSMD5=${SA14_ESSAY_MD5:-47a39dac581a}; CODEMD5=${SA14_CODE_MD5:-c5e7c82e3e11}   # yzint14c's resprobe answers (text_md5; every arm, every read)\n"
    "# zsa14: lever SA's arms and readings (module header)\n"
    "SSN=${SA14_SSN:-1025}; PK1=${SA14_PICK1:-1}; PK2=${SA14_PICK2:-2}   # S1 = stage_short SSN + stage_pick PK1, S2 = stage_short SSN + stage_pick PK2 (B: both 0)\n"
    "GORD=${SA14_GORDER-B S1 S2 S2 S1 B}; C16GN=${SA14_C16GN:-1}; POSTN=${SA14_POST:-10}; IDARMS=${SA14_IDARMS-B S1 S2}; GSA=${SA14_GSARM:-S2}; LDA=${SA14_LDARM:-S2}; SAARMS=\"S1 S2\"\n"
    "STPMD5=${SA14_STPMD5:-569fd59b}   # the coordinator's steps.py (zsa14/steps.py, copied from its scratchpad)\n"
    "C16N=${SA14_C16N:-1}   # zsa14: 1 = the c16 gens pair (B + the better SA arm), 0 = none\n"
    "for v in $SSN $PK1 $PK2 $C16GN $POSTN $C16N; do case $v in ''|*[!0-9]*) echo \"a zsa14 knob is not a number ('$v')\"; exit 1 ;; esac; done\n"
    "[ $SSN -ge 1 ] && [ $SSN -le 65536 ] || { echo \"SA14_SSN 1 .. 65536 (stage_short)\"; exit 1; }; [ $PK1 -le 64 ] && [ $PK2 -le 64 ] || { echo \"SA14_PICK1 / SA14_PICK2 0 .. 64 (stage_pick)\"; exit 1; }\n"
    "[ $C16GN -ge 1 ] || { echo \"SA14_C16GN >= 1\"; exit 1; }; case $C16N in 0|1) ;; *) echo \"SA14_C16N 0 or 1\"; exit 1 ;; esac\n"
    "[ -n \"$GORD\" ] && [ -n \"$IDARMS\" ] || { echo \"SA14_GORDER / SA14_IDARMS must name arms\"; exit 1; }\n"
    "for a in $GORD $IDARMS; do case $a in B|S1|S2) ;; *) echo \"SA14_GORDER / SA14_IDARMS take B S1 S2 (got '$a')\"; exit 1 ;; esac; done\n"
    "for a in $GSA $LDA; do case $a in S1|S2) ;; *) echo \"SA14_GSARM / SA14_LDARM take S1 or S2 (got '$a')\"; exit 1 ;; esac; done\n"
    "case \" $IDARMS \" in *\" $GSA \"*) ;; *) echo \"SA14_GSARM $GSA needs its a1 (SA14_IDARMS)\"; exit 1 ;; esac; case \" $IDARMS \" in *\" $LDA \"*\" B \"*|*\" B \"*\" $LDA \"*) ;; *) echo \"SA14_IDARMS needs B and SA14_LDARM $LDA\"; exit 1 ;; esac\n")
# the c16 gens block: --gens C16GN (c8: C8GN)
rep('local wo="--window time"; [ $sc = G ] && wo="--window gens --gens $C8GN --edge 0.5"',
    'local wo="--window time"; [ $sc = G ] && wo="--window gens --gens $([ $n = 16 ] && echo $C16GN || echo $C8GN) --edge 0.5"')
# ---------------------------------------------------------------- the arms: B = the int14 pin (SA off), S1 / S2 = lever SA on
line("    B) [ $SW = 1 ] && echo '{\"ds41_bi\": 2, \"gg_pf\": 1, \"ar_xpush\": 0,",
     "    B) echo '{}' ;;   # zsa14: B = THE int14 PIN with lever SA off (the base pin's stage_short 0, stage_pick 0) = ref14's O\n"
     "    S1) echo \"{\\\"stage_short\\\": $SSN, \\\"stage_pick\\\": $PK1}\" ;;   # lever SA: the prefill stager armed for steps of >= SSN tokens at the span's shared level, pick K1\n"
     "    S2) echo \"{\\\"stage_short\\\": $SSN, \\\"stage_pick\\\": $PK2}\" ;;   # lever SA, pick K2 (pre-move: the fastest pick of the usahb screen)")
rep('B) echo "B (int12 / ref12: ds41_bi 2, gg_pf 1, ar_xpush 0, gg_gemv_m16 0, place_swap_lent 0, heat_min_gain_x100 150)" ;;',
    'B) echo "B (THE int14 PIN, lever SA off: stage_short 0, stage_pick 0)" ;; S1) echo "S1 (lever SA: stage_short $SSN, stage_pick $PK1)" ;; S2) echo "S2 (lever SA: stage_short $SSN, stage_pick $PK2)" ;;')
rep('swlwant() { case $1 in O|W) echo "1 $SWLG" ;;', 'swlwant() { case $1 in O|W|B|S1|S2) echo "1 $SWLG" ;;')
line('DESC="THE int14 REFERENCE READ',
     'DESC="LEVER SA ON int14 (zsa14: image int14c, THE int14 PIN = ref14\'s O; arms B = the pin (SA off), S1 = stage_short $SSN + stage_pick $PK1, '
     'S2 = stage_short $SSN + stage_pick $PK2; RADIANCE_STAGE_MIXED unset) NST 5 stack5 start (stack4\'s + RADIANCE_LOG_STEPS=1, E1 declared off, HB ranks $HBRANKS) '
     '+ RADIANCE_DS41_MHC_FUSE=0 (M1 declared) + RADIANCE_DS41_GQ=3 + RADIANCE_DS41_E2=1 + RADIANCE_DS41_KV_REPLICATED=1 (KVR) + A4 table + A11 ranged entries '
     '(ref14\'s copies) + RADIANCE_FN4_AR_XPUSH_MIN=$XPMIN $([ $STAMPN -gt 0 ] && echo "+ RADIANCE_AQL_STAMP=$STAMPN" || echo "(no stamping)") | SP / ARC / CBH off, '
     'ngshare, no ATTACH | the pin = stack5\'s S5T with draft_sample 0 + SP 0 / 0 + $MJ + the fixed pin (ds41_psk_k $PSKK; draft_lend 1; dspark_conf 1, dspark_conf_rows 1)"')
# ---------------------------------------------------------------- the pin check: every arm int14; B stage_short / stage_pick 0, S1 / S2 lever SA's values
cut("for a in O Z B; do python3 -c 'import json,sys; d=json.loads(sys.argv[1]); sw = sys.argv[2] == \"1\"; a = sys.argv[3]\n",
    '[ "$(sha256sum $MODELDIR/$SIDECAR',
    """for a in O B S1 S2; do python3 -c 'import json,sys; d=json.loads(sys.argv[1]); sw = sys.argv[2] == "1"; a = sys.argv[3]
assert d["ds41_psk"] == 4 and d["ds41_psk_cr"] == 1 and d["draft_sample"] == 0 and d["draft_lend"] == 1 and d["dspark_conf"] == 1 and d["dspark_conf_rows"] == 1
ss, sp = (0, 0) if a in ("O", "B") else (int(sys.argv[11]), int(sys.argv[12] if a == "S1" else sys.argv[13]))
assert d["stage_short"] == ss and d["stage_pick"] == sp and d["place_hb"] == 0
if sw:
    assert d["mhc_fuse"] == int(sys.argv[4]) and d["ix_lds"] == 0 and d["ix_ni"] == 0 and d["draft_depth0"] == int(sys.argv[5]) and d["sched_row_cap"] == int(sys.argv[6]) and d["stage_full_min"] == 0 and d["stage_full_pick"] == 0
    assert d["ds41_mix_wv"] == int(sys.argv[7]) and d["ds41_gq"] == 3 and d["ds41_e2"] == 1 and d["ds41_arcoded"] == 0 and d["exl3_x3d"] == 1
    assert d["sched_cb_hold_ms"] == 0 and d["sched_cb_hold_k"] == int(sys.argv[8]) and d["ds41_shexp_lane"] == int(sys.argv[9]) and "exl3_x3d_pf" not in d
    g = int(sys.argv[10])
    assert d["ds41_bi"] == 6 and d["gg_pf"] == 2 and d["ar_xpush"] == 1 and d["gg_gemv_m16"] == 1 and d["place_swap_lent"] == 1 and d["heat_min_gain_x100"] == g and d["heat_hyst_x100"] == 125
else: assert d["draft_depth0"] == 1 and not any(k in d for k in ("gg_gemv_m16", "exl3_x3d_pf", "ar_xpush", "mhc_fuse", "ix_lds", "ix_ni", "ds41_arcoded", "stage_full_min", "sched_row_cap", "ds41_mix_wv", "ds41_gq", "ds41_e2", "sched_cb_hold_ms", "exl3_x3d", "place_swap_lent", "heat_min_gain_x100", "heat_hyst_x100"))' "$(pinj $a)" $SW $a $MHC $D0 $CAP $MXV $CBHK $F4V $SWLG $SSN $PK1 $PK2 2>/dev/null || HMISS="$HMISS json:$a"; done
""")
# ---------------------------------------------------------------- helpers: sa14sum.py (selftest), steps.py (md5 + parse)
rep("$YDS $NCM $NCT $DQ $SPB0 $A4TAB $A11TAB $YZA1 $REF12A1 $NGS/ngshare.sh", "$YDS $NCM $NCT $DQ $SPB0 $A4TAB $A11TAB $REF12A1 $STP $NGS/ngshare.sh")
rep("$U1S $R12S $MBT2 $NCMB2 $C8MIX $YDS $NCM $NCT 2>&1", "$U1S $R12S $MBT2 $NCMB2 $C8MIX $YDS $NCM $NCT $STP 2>&1")
rep('XPT=$(timeout 60 python3 -I -B $YDS selftest',
    '[ "$(md5sum < $STP 2>/dev/null | cut -c1-8)" = "$STPMD5" ] || HMISS="$HMISS steps.py:$STP(want md5 $STPMD5)"\n'
    'XPT=$(timeout 60 python3 -I -B $YDS selftest')
rep("| r14sum.py $XPT | ncmatch.py $NMT2 |", "| sa14sum.py $XPT | ncmatch.py $NMT2 | steps.py md5 $STPMD5 |")
line('FNS="stagx coldid tabs14',
     'FNS="stagx tabs14 rankrb a11line a4line xpline sahbline sahbn sa14id bestsa sa14reads ncmset stepsx sa14summary ngclean swlwant log run mkd restart check '
     'down stopfile best_vars desc serve_args xlog elapsed armj NAMEOF pinj stackenv keeplog capture thermline dfree diskok cmpfile idrec identw modes vramstart '
     'fn4get fn4set nudge fn4code pcreset arm msnap mdelta ssnap gate1 alone cmpx budok mwin ctrinfo i11d monons utcnow drafted readback swnote capd mbx cbhd cbhg '
     'gsx dstart idok gt48"')
rep('  for a in O B Z; do echo "+ arm $a ($(NAMEOF $a)): $(armj $a)"; done', '  for a in O B S1 S2; do echo "+ arm $a ($(NAMEOF $a)): $(armj $a)"; done')
line('  echo "+ readings: 1 identity warm-up vs stock, a1-O',
     '  echo "+ readings: 1 identity warm-up vs stock, a1 [$IDARMS] vs ref12 a1 ($([ -s $REF12A1 ] && echo present || echo MISSING)) + a1-B vs yzint14c a1-Z '
     '($([ -s $YZA1 ] && echo present || echo "absent: not compared")), gs$GSC-$GSA vs a1-$GSA (MAXSEQS ${SA14_MAXSEQS:-base}), staggered ld-$LDA vs a1-$LDA (conc $LDCONC, '
     '$LDPF-token prefills; SA armed steps counted) -- ANY difference STOPS before the speed blocks | 2 c8 work-based gens $C8GN [$GORD] | 3 c16 gens $C16GN: B + the '
     'better SA arm ($([ $C16N -gt 0 ] && echo on || echo off)) | decode p50 all + $POSTN after each newcomer | budget $BUDGET min | four cards $(cat '
     '/sys/class/drm/card*/device/unique_id 2>/dev/null | wc -l) unique_id | n-gram tables in $NGDIR now: $(ls $NGDIR/ngram-*.tab 2>/dev/null | wc -l) | free RAM '
     '$(free -g | awk \'NR==2{print $7}\') GiB"')
# ---------------------------------------------------------------- the section: ref14's identity / readings / summary / start -> zsa14's
rep("# ---------------------------------------------------------------- ref14's own functions (the tables, the int14 pin read back, identity + the stop, readings, summary)\n",
    "# ---------------------------------------------------------------- zsa14's own functions (ref14's tables, int14 pin read back and staggered check;\n"
    "# lever SA's identity + the stop, the gens readings, the summary, the start)\n")
cut("coldid() {", "ngclean() {", P("funcs.sh"))
cut("dstart() {", "trap 'down;", P("dstart.sh"))
# ---------------------------------------------------------------- main
line('xlog "$ITEM: start IMAGE=$IMAGE',
     'xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f \'{{.Id}}\' $IMAGE 2>/dev/null | cut -c8-19))$([ $SW = 1 ] || echo " | MOCK-RUN SA14_SW=0 (NOT A READING): NO int9b .. int14c '
     'switch posted, cross-run references INFO") | LEVER SA ON int14 (zero-build screen), ONE FRESH START on ngshare (no ATTACH), int14 pinned right after it (fn4) and read back on '
     'every rank: $DESC | MAXSEQS ${SA14_MAXSEQS:-the base value} | tables: A4 $A4MD5 ($A4N lines), A11 $A11MD5 ($A11NE entries), NOT re-benched | 1 identity warm-up, a1 '
     '[$IDARMS] vs ref12 a1, gs$GSC-$GSA, staggered ld-$LDA (ANY difference: STOP before the speed blocks) | 2 c8 work-based gens $C8GN [$GORD] | 3 c16 gens $C16GN B + '
     'the better SA arm | budget $BUDGET min | no evict"')
rep("\nr14summary\nunset $SWITCHES\n", "\nsa14summary\nunset $SWITCHES\n")
main = s.split("# ---------------------------------------------------------------- main")[1]
for w in ("r14summary", "r14reads", "r14id", "REF14", "coldid"):
    assert w not in main, "main still names %s" % w
for w in ("r14reads", "r14id ", "r14summary", "coldid"):
    assert w not in s, "the body still names %s" % w
open(dst, "w").write(s)
os.chmod(dst, 0o755)
print("%s: %d edits, %d lines (from %s md5 %s)" % (dst, n_edits, s.count("\n"), src, md5))
