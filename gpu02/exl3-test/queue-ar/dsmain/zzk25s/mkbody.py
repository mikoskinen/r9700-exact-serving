#!/usr/bin/env python3
"""mkbody.py SRC PARTS DST [SRCMD5] -- zzk25s/dsm-zzk25s.sh from zsa14/dsm-zsa14.sh (SRC, md5 SRCMD5 fe5c6c3b as staged + run 12.11.):
every edit an exact, asserted replacement (tag xsa14 -> xk25s, knobs SA14_ -> K25S_; lever SA's arms B / S1 / S2 -> lever K25U's B
(exl3_x3d 1) / Z (exl3_x3d 2) on the int14 pin; the plugin md5 = the K25U plugin's; swnote's exl3_x3d refusal probe moved 2 -> 5
(TESTING.md 13.11. K25U SWITCH-VALUE NOTE); arm() also asserts the exl3_x3d read back (x3dchk); zsa14's identity / readings / summary
functions out, zzk25s's in (+ the 1u essay sets)). PARTS = zzk25s/mkparts (header.sh: the module header; funcs.sh: zzk25s's functions
in place of zsa14's sahbline .. sa14summary; dstart.sh: zzk25s's start, in place of zsa14's)."""
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


# ---------------------------------------------------------------- before the tag change: the strings that name zsa14's own files / provenance
rep("dsm-zsa14.sh $(md5sum < $0 | cut -c1-8) (a copy of ref14/dsm-zsa14.sh 74981d12 via zsa14/mkbody.py)",
    "dsm-zzk25s.sh $(md5sum < $0 | cut -c1-8) (a copy of zsa14/dsm-zsa14.sh %s via zzk25s/mkbody.py)" % md5)
rep('HMD5="zsa14/sa14sum.py $(md5sum < $YDS | cut -c1-8), zsa14/steps.py $(md5sum < $STP | cut -c1-8) (the coordinator\'s), zsa14/m16-forms.txt '
    '$(md5sum < $A4TAB | cut -c1-8) (= ref14\'s = yzint14c\'s), zsa14/a11-entries.txt $(md5sum < $A11TAB | cut -c1-8) (= ref14\'s = yzint14c\'s P0), '
    'zsa14/ncmatch.py $(md5sum < $NCM | cut -c1-8) (ref14\'s, copied), ',
    'HMD5="zzk25s/k25sum.py $(md5sum < $YDS | cut -c1-8) (zsa14/sa14sum.py\'s copy + K25U), zzk25s/steps.py $(md5sum < $STP | cut -c1-8) (= zsa14\'s, the '
    'coordinator\'s), zzk25s/m16-forms.txt $(md5sum < $A4TAB | cut -c1-8) (= zsa14\'s = ref14\'s), zzk25s/a11-entries.txt $(md5sum < $A11TAB | cut -c1-8) '
    '(= zsa14\'s = ref14\'s), zzk25s/ncmatch.py $(md5sum < $NCM | cut -c1-8) (= zsa14\'s = ref14\'s), ')
# ---------------------------------------------------------------- the header (everything above `set -u`)
cut("#!/bin/bash\n", "set -u\n", P("header.sh").replace("SRCMD5 below", "md5 %s" % md5))
# ---------------------------------------------------------------- the tag: xsa14 -> xk25s, SA14_ -> K25S_, the body name, the out dir
for a, b in (("$R/xdsm/zsa14", "$R/xdsm/zzk25s"),):
    assert s.count(a) >= 1, a
    s = s.replace(a, b)
s = re.sub(r"(?<![A-Za-z])xsa14(?![A-Za-z0-9])", "xk25s", s)
s = s.replace("SA14_", "K25S_")
assert "xsa14" not in s and "SA14_" not in s
n_edits += 1
# ---------------------------------------------------------------- knobs
rep("ITEM=${K25S_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zsa14}", "ITEM=${K25S_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzk25s}")
rep("YDS=$HD/sa14sum.py; SAS=$HD/sa14sum.py;", "YDS=$HD/k25sum.py; SAS=$HD/k25sum.py;")
cut("# zsa14: lever SA's arms and readings (module header)\n", "for v in $C8GN $C8N $C16N $C4N $C1N $SWN; do",
    "# zzk25s: lever K25U's arms and readings (module header)\n"
    "XB=${K25S_XB:-1}; XZ=${K25S_XZ:-2}; XREF=${K25S_XREFUSE:-5}   # B = exl3_x3d XB (the int14 kernels exactly), Z = exl3_x3d XZ (X3D + K2.5 gate_up G3 + down U4); XREF: swnote's refusal probe (400)\n"
    "GORD=${K25S_GORDER-B Z Z B}; C16GN=${K25S_C16GN:-1}; POSTN=${K25S_POST:-10}; IDARMS=${K25S_IDARMS-B Z}; GSA=${K25S_GSARM:-Z}; LDA=${K25S_LDARM:-Z}\n"
    "K16A=${K25S_C16ARMS-B Z}; U1A=${K25S_1UARMS-B Z}; ONEU=${K25S_1U:-1}   # c16: a gens block an arm; 1u: per arm one discarded + U1N essay reads (B first)\n"
    "STPMD5=${K25S_STPMD5:-569fd59b}   # the coordinator's steps.py (zzk25s/steps.py = zsa14's)\n"
    "C16N=${K25S_C16N:-1}   # zzk25s: 1 = the c16 gens blocks (K16A), 0 = none\n"
    "for v in $XB $XZ $XREF $C16GN $POSTN $C16N $ONEU; do case $v in ''|*[!0-9]*) echo \"a zzk25s knob is not a number ('$v')\"; exit 1 ;; esac; done\n"
    "case $XB in 0|1) ;; *) echo \"K25S_XB 0 or 1 (the int14 kernels: 1 = X3D, the int14 pin's)\"; exit 1 ;; esac; case $XZ in 2|3|4) ;; *) echo \"K25S_XZ 2 / 3 / 4 (K25U's values)\"; exit 1 ;; esac\n"
    "[ $XREF -ge 5 ] || { echo \"K25S_XREFUSE >= 5 (0 .. 4 are VALID on the K25U plugin: a probe there SETS the switch)\"; exit 1; }\n"
    "[ $C16GN -ge 1 ] || { echo \"K25S_C16GN >= 1\"; exit 1; }; case $C16N$ONEU in [01][01]) ;; *) echo \"K25S_C16N / K25S_1U 0 or 1\"; exit 1 ;; esac\n"
    "[ -n \"$GORD\" ] && [ -n \"$IDARMS\" ] || { echo \"K25S_GORDER / K25S_IDARMS must name arms\"; exit 1; }\n"
    "for a in $GORD $IDARMS $K16A $U1A $GSA $LDA; do case $a in B|Z) ;; *) echo \"K25S_GORDER / IDARMS / C16ARMS / 1UARMS / GSARM / LDARM take B and Z (got '$a')\"; exit 1 ;; esac; done\n"
    "case \" $IDARMS \" in *\" $GSA \"*) ;; *) echo \"K25S_GSARM $GSA needs its a1 (K25S_IDARMS)\"; exit 1 ;; esac; case \" $IDARMS \" in *\" $LDA \"*) ;; *) echo \"K25S_LDARM $LDA needs its a1 (K25S_IDARMS)\"; exit 1 ;; esac\n"
    "case \" $IDARMS \" in *\" B \"*) ;; *) echo \"K25S_IDARMS needs B (vs ref12's a1)\"; exit 1 ;; esac; case \" $IDARMS \" in *\" Z \"*) ;; *) echo \"K25S_IDARMS needs Z (vs ref12's a1)\"; exit 1 ;; esac\n")
# ---------------------------------------------------------------- the arms: B = the int14 pin (exl3_x3d XB), Z = the int14 pin with exl3_x3d XZ
line("    B) echo '{}' ;;   # zsa14: B = THE int14 PIN with lever SA off",
     "    B) [ $SW = 1 ] && echo \"{\\\"exl3_x3d\\\": $XB}\" || echo '{}' ;;   # zzk25s: B = THE int14 PIN, exl3_x3d XB 1 (= the pin's own value: the int14 kernels exactly); the mock has no exl3_x3d")
line("    S1) echo \"{\\\"stage_short\\\": $SSN, \\\"stage_pick\\\": $PK1}\" ;;",
     "    # (zsa14's S1 / S2 lever-SA arms removed)")
line("    S2) echo \"{\\\"stage_short\\\": $SSN, \\\"stage_pick\\\": $PK2}\" ;;", "")
rep("    Z) zarm ;;   # ALL: every lever whose P0 passed\n",
    "    Z) [ $SW = 1 ] && echo \"{\\\"exl3_x3d\\\": $XZ}\" || echo '{}' ;;   # zzk25s: lever K25U, exl3_x3d XZ 2 = X3D + K2.5 gate_up G3 + down U4 (the K25U plugin)\n")
rep('B) echo "B (THE int14 PIN, lever SA off: stage_short 0, stage_pick 0)" ;; S1) echo "S1 (lever SA: stage_short $SSN, stage_pick $PK1)" ;; S2) echo "S2 (lever SA: stage_short $SSN, stage_pick $PK2)" ;;',
    'B) echo "B (THE int14 PIN, exl3_x3d $XB: the int14 kernels)" ;;')
rep('  Z) echo "Z (ALL: $(zarm | tr -d \'{}\\"\'))" ;;', '  Z) echo "Z (lever K25U: the int14 pin + exl3_x3d $XZ, X3D + K2.5 gate_up G3 + down U4)" ;;')
line('DESC="LEVER SA ON int14',
     'DESC="LEVER K25U ON int14 (zzk25s: image ds41-k25u = int14c + the K25U exl3r4d.so, THE int14 PIN = ref14\'s O; arms B = the pin (exl3_x3d $XB), '
     'Z = the pin + exl3_x3d $XZ (X3D + K2.5 gate_up G3 + down U4)) NST 5 stack5 start (stack4\'s + RADIANCE_LOG_STEPS=1, E1 declared off, HB ranks $HBRANKS) '
     '+ RADIANCE_DS41_MHC_FUSE=0 (M1 declared) + RADIANCE_DS41_GQ=3 + RADIANCE_DS41_E2=1 + RADIANCE_DS41_KV_REPLICATED=1 (KVR) + A4 table + A11 ranged entries '
     '(zsa14\'s = ref14\'s copies) + RADIANCE_FN4_AR_XPUSH_MIN=$XPMIN $([ $STAMPN -gt 0 ] && echo "+ RADIANCE_AQL_STAMP=$STAMPN" || echo "(no stamping)") | SP / ARC / CBH / SA off, '
     'ngshare, no ATTACH | the pin = stack5\'s S5T with draft_sample 0 + SP 0 / 0 + $MJ + the fixed pin (ds41_psk_k $PSKK; draft_lend 1; dspark_conf 1, dspark_conf_rows 1)"')
# ---------------------------------------------------------------- swnote: the exl3_x3d refusal probe 2 -> XREF (5): 2 is VALID on the K25U plugin
rep("sched_cb_hold_k 1 / exl3_x3d 2 / ds41_bi 7", "sched_cb_hold_k 1 / exl3_x3d $XREF / ds41_bi 7")
rep("c9=$(fn4code '{\"exl3_x3d\": 2}')", "c9=$(fn4code \"{\\\"exl3_x3d\\\": $XREF}\")")
rep("sched_cb_hold_k 1 -> HTTP $c8, exl3_x3d 2 -> HTTP $c9,", "sched_cb_hold_k 1 -> HTTP $c8, exl3_x3d $XREF -> HTTP $c9 (the K25U plugin takes 0 .. 4),")
assert "\"exl3_x3d\": 2" not in s and "exl3_x3d 2 ->" not in s and "exl3_x3d 2 /" not in s
# ---------------------------------------------------------------- arm(): + lever K25U's switch read back (x3dchk) before every block
rep('  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4: every switch of arm $1 applied, nothing pending (polled <= 30 s, a nudge a try)"; return 0; }',
    '  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4: every switch of arm $1 applied, nothing pending (polled <= 30 s, a nudge a try)"; x3dchk $1; return 0; }')
rep('    case "$r" in OK*) return 0 ;; esac', '    case "$r" in OK*) x3dchk $1; return $? ;; esac   # zzk25s: + exl3_x3d asserted (B XB, Z XZ)')
# ---------------------------------------------------------------- the plugin: the K25U exl3r4d.so (ds41-k25u 38b88ec)
rep("PLUGMD5=${K25S_PLUGMD5:-9823cdde}", "PLUGMD5=${K25S_PLUGMD5:-7b2f66d6}   # zzk25s: the K25U plugin (ds41-k25u 38b88ec; zzk25 checked it = the image's)")
rep('echo "+ exl3r4d.so md5 ${PLUGGOT:-?} (want $PLUGMD5 = int12\'s X3D-F plugin ds41-x3d 579d0bc): SKIP (K25S_DRY_ALLOW_OLD=1)"',
    'echo "+ exl3r4d.so md5 ${PLUGGOT:-?} (want $PLUGMD5 = the K25U plugin ds41-k25u 38b88ec): SKIP (K25S_DRY_ALLOW_OLD=1)"')
rep('xlog "VIRHE $IMAGE\'s kernels/exl3r4d.so md5 ${PLUGGOT:-?}, not int12\'s plugin $PLUGMD5 (ds41-x3d 579d0bc)"',
    'xlog "VIRHE $IMAGE\'s kernels/exl3r4d.so md5 ${PLUGGOT:-?}, not the K25U plugin $PLUGMD5 (ds41-k25u 38b88ec): exl3_x3d 2 would be refused (or be another kernel)"')
rep('echo "+ exl3r4d.so md5 $PLUGGOT = int12\'s X3D-F plugin (ds41-x3d 579d0bc) -- OK"', 'echo "+ exl3r4d.so md5 $PLUGGOT = the K25U plugin (ds41-k25u 38b88ec: exl3_x3d 0 .. 4) -- OK"')
# ---------------------------------------------------------------- the pin check: every arm int14, stage_short / stage_pick 0; exl3_x3d XB (O, B) / XZ (Z)
cut("for a in O B S1 S2; do python3 -c 'import json,sys; d=json.loads(sys.argv[1]); sw = sys.argv[2] == \"1\"; a = sys.argv[3]\n",
    '[ "$(sha256sum $MODELDIR/$SIDECAR',
    """for a in O B Z; do python3 -c 'import json,sys; d=json.loads(sys.argv[1]); sw = sys.argv[2] == "1"; a = sys.argv[3]
assert d["ds41_psk"] == 4 and d["ds41_psk_cr"] == 1 and d["draft_sample"] == 0 and d["draft_lend"] == 1 and d["dspark_conf"] == 1 and d["dspark_conf_rows"] == 1
assert d["stage_short"] == 0 and d["stage_pick"] == 0 and d["place_hb"] == 0
if sw:
    assert d["mhc_fuse"] == int(sys.argv[4]) and d["ix_lds"] == 0 and d["ix_ni"] == 0 and d["draft_depth0"] == int(sys.argv[5]) and d["sched_row_cap"] == int(sys.argv[6]) and d["stage_full_min"] == 0 and d["stage_full_pick"] == 0
    assert d["ds41_mix_wv"] == int(sys.argv[7]) and d["ds41_gq"] == 3 and d["ds41_e2"] == 1 and d["ds41_arcoded"] == 0 and d["exl3_x3d"] == int(sys.argv[12] if a == "Z" else sys.argv[11])
    assert d["sched_cb_hold_ms"] == 0 and d["sched_cb_hold_k"] == int(sys.argv[8]) and d["ds41_shexp_lane"] == int(sys.argv[9]) and "exl3_x3d_pf" not in d
    g = int(sys.argv[10])
    assert d["ds41_bi"] == 6 and d["gg_pf"] == 2 and d["ar_xpush"] == 1 and d["gg_gemv_m16"] == 1 and d["place_swap_lent"] == 1 and d["heat_min_gain_x100"] == g and d["heat_hyst_x100"] == 125
else: assert d["draft_depth0"] == 1 and not any(k in d for k in ("gg_gemv_m16", "exl3_x3d_pf", "ar_xpush", "mhc_fuse", "ix_lds", "ix_ni", "ds41_arcoded", "stage_full_min", "sched_row_cap", "ds41_mix_wv", "ds41_gq", "ds41_e2", "sched_cb_hold_ms", "exl3_x3d", "place_swap_lent", "heat_min_gain_x100", "heat_hyst_x100"))' "$(pinj $a)" $SW $a $MHC $D0 $CAP $MXV $CBHK $F4V $SWLG $XB $XZ 2>/dev/null || HMISS="$HMISS json:$a"; done
""")
# ---------------------------------------------------------------- helpers: k25sum.py (selftest), the function list, the dry text
rep("| sa14sum.py $XPT | ncmatch.py $NMT2 | steps.py md5 $STPMD5 |", "| k25sum.py $XPT | ncmatch.py $NMT2 | steps.py md5 $STPMD5 |")
line('FNS="stagx tabs14 rankrb a11line a4line xpline sahbline sahbn sa14id bestsa sa14reads ncmset stepsx sa14summary ngclean',
     'FNS="stagx tabs14 rankrb a11line a4line xpline x3dchk x3dsay k25id k25reads ncmset stepsx k25summary ngclean blockx i11g swlwant log run mkd restart check '
     'down stopfile best_vars desc serve_args xlog elapsed armj NAMEOF pinj stackenv keeplog capture thermline dfree diskok cmpfile idrec identw modes vramstart '
     'fn4get fn4set nudge fn4code pcreset arm msnap mdelta ssnap gate1 alone cmpx budok mwin ctrinfo i11d monons utcnow drafted readback swnote capd mbx cbhd cbhg '
     'gsx dstart idok gt48"')
rep('  for a in O B S1 S2; do echo "+ arm $a ($(NAMEOF $a)): $(armj $a)"; done', '  for a in O B Z; do echo "+ arm $a ($(NAMEOF $a)): $(armj $a)"; done')
line('  echo "+ readings: 1 identity warm-up vs stock, a1 [$IDARMS] vs ref12 a1',
     '  echo "+ readings: 1 identity warm-up vs stock, a1 [$IDARMS] vs ref12 a1 ($([ -s $REF12A1 ] && echo present || echo MISSING)) + a1-B vs yzint14c a1-Z '
     '($([ -s $YZA1 ] && echo present || echo "absent: not compared")), gs$GSC-$GSA vs a1-$GSA (MAXSEQS ${K25S_MAXSEQS:-base}), staggered ld-$LDA vs a1-$LDA (conc $LDCONC, '
     '$LDPF-token prefills) -- ANY difference STOPS before the speed blocks | 2 c8 work-based gens $C8GN [$GORD] | 3 c16 gens $C16GN [$K16A] ($([ $C16N -gt 0 ] && echo on || echo off)) '
     '| 4 1u essay [$U1A] 1 discarded + $U1N each ($([ $ONEU = 1 ] && echo on || echo off); answers vs $ESSMD5) | exl3_x3d asserted before every block (B $XB, Z $XZ); refusal probe exl3_x3d $XREF '
     '| decode p50 all + full-row bucket + $POSTN after each newcomer | budget $BUDGET min | four cards $(cat /sys/class/drm/card*/device/unique_id 2>/dev/null | wc -l) unique_id '
     '| n-gram tables in $NGDIR now: $(ls $NGDIR/ngram-*.tab 2>/dev/null | wc -l) | free RAM $(free -g | awk \'NR==2{print $7}\') GiB"')
# ---------------------------------------------------------------- the section: zsa14's identity / readings / summary / start -> zzk25s's
rep("# ---------------------------------------------------------------- zsa14's own functions (ref14's tables, int14 pin read back and staggered check;\n"
    "# lever SA's identity + the stop, the gens readings, the summary, the start)\n",
    "# ---------------------------------------------------------------- zzk25s's own functions (zsa14's: ref14's tables, int14 pin read back and staggered\n"
    "# check; zzk25s's: lever K25U's switch read back, identity + the stop, the gens / 1u readings, the summary, the start)\n")
cut("sahbline() {", "ngclean() {", P("funcs.sh"))
cut("dstart() {", "trap 'down;", P("dstart.sh"))
# ---------------------------------------------------------------- main
line('xlog "$ITEM: start IMAGE=$IMAGE',
     'xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f \'{{.Id}}\' $IMAGE 2>/dev/null | cut -c8-19))$([ $SW = 1 ] || echo " | MOCK-RUN K25S_SW=0 (NOT A READING): NO int9b .. int14c '
     'switch posted, exl3_x3d not posted, cross-run references INFO") | LEVER K25U ON int14 (zero-build screen), ONE FRESH START on ngshare (no ATTACH), int14 pinned right after it (fn4) '
     'and read back on every rank: $DESC | MAXSEQS ${K25S_MAXSEQS:-the base value} | tables: A4 $A4MD5 ($A4N lines), A11 $A11MD5 ($A11NE entries), NOT re-benched | exl3r4d.so '
     'md5 wanted ${K25S_PLUGMD5:-7b2f66d6} (checked next) | 1 identity warm-up, a1 [$IDARMS] vs ref12 a1, gs$GSC-$GSA, staggered ld-$LDA (ANY difference: STOP before the speed blocks) | 2 c8 work-based '
     'gens $C8GN [$GORD] | 3 c16 gens $C16GN [$K16A] | 4 1u essay [$U1A] 1 discarded + $U1N each | exl3_x3d asserted before every block, refusal probe exl3_x3d $XREF | budget $BUDGET min | no evict"')
rep("\nsa14summary\nunset $SWITCHES\n", "\nk25summary\nunset $SWITCHES\n")
main = s.split("# ---------------------------------------------------------------- main")[1]
for w in ("sa14summary", "sa14reads", "sa14id", "sahbline", "sahbn", "bestsa"):   # (the image literal "lever SA / HB (ds41-sahb)" stays: int9's)
    assert w not in main, "main still names %s" % w
for w in ("sa14reads", "sa14id ", "sa14summary", "sahbline", "sahbn", "bestsa", "$SSN", "$PK1", "$PK2", "SAARMS", "SA14"):
    assert w not in s, "the body still names %s" % w
left = [l for l in s.split("\n") if "zsa14" in l and "zsa14's" not in l and "zsa14/" not in l and "zsa14 B" not in l]
assert not left, "zsa14 left in: %r" % [l[:120] for l in left]
open(dst, "w").write(s)
os.chmod(dst, 0o755)
print("%s: %d edits, %d lines (from %s md5 %s)" % (dst, n_edits, s.count("\n"), src, md5))
