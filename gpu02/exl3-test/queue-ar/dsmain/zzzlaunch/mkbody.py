#!/usr/bin/env python3
"""mkbody.py SRC PARTS DST [SRCMD5] -- zzzlaunch/dsm-zzzlaunch.sh from zzk25s/dsm-zzk25s.sh (SRC, md5 SRCMD5 abbf540f as staged + run 13.11.):
every edit an exact, asserted replacement (tag xk25s -> xlch, knobs K25S_ -> LCH_, functions k25id / k25reads / k25summary -> lchid /
lchreads / lchsummary; lever K25U's arms B (exl3_x3d 1) / Z (exl3_x3d 2) on the int14 pin -> lever RCO's B (route_copy_ov 0) / Z
(route_copy_ov 1) on THE int15 PIN (exl3_x3d 2 in the base pin); x3dchk / x3dsay read route_copy_ov + exl3_x3d + GET "rco"; swnote also
probes route_copy_ov 2 -> 400 and asserts route_copy_ov 0 + "rco" at start; RADIANCE_AQL_STAMP=LCH_STAMP (16) at the start; clock pairs
around the c8 gens blocks; the AQL read (aqlread: aqlcp.py monowin + lchaql.py) after lchsum.py; the image literals + route_copy_ov).
PARTS = zzzlaunch/mkparts (header.sh: the module header; funcs.sh: x3dchk / x3dsay in place of zzk25s's + clockpair / clk / aqlread)."""
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


# ---------------------------------------------------------------- before the tag change: the strings naming zzk25s's own files / provenance
line('HMD5="zzk25s/k25sum.py $(md5sum < $YDS | cut -c1-8)',
     'HMD5="zzzlaunch/lchsum.py $(md5sum < $YDS | cut -c1-8) (zzk25s/k25sum.py\'s copy + lever RCO\'s switch + the int15 pin guard), '
     'zzzlaunch/lchaql.py $(md5sum < $LAQ | cut -c1-8) (the one-in-N AQL site reader), zzzlaunch/aqlcp.py $(md5sum < $AQL | cut -c1-8) '
     '(zzzaql\'s, copied: monowin / synth), zzzlaunch/steps.py $(md5sum < $STP | cut -c1-8) (= zzk25s\'s, the coordinator\'s), '
     'zzzlaunch/m16-forms.txt $(md5sum < $A4TAB | cut -c1-8) (= zzk25s\'s = ref14\'s), zzzlaunch/a11-entries.txt $(md5sum < $A11TAB | cut -c1-8) '
     '(= zzk25s\'s = ref14\'s), zzzlaunch/ncmatch.py $(md5sum < $NCM | cut -c1-8) (= zzk25s\'s), dsm-zzzlaunch.sh $(md5sum < $0 | cut -c1-8) '
     '(a copy of zzk25s/dsm-zzk25s.sh %s via zzzlaunch/mkbody.py), yswl/resprobe.py $(md5sum < $SPB | cut -c1-8), ngramshare: ngshare.sh '
     '$(md5sum < $NGS/ngshare.sh | cut -c1-8) / yaml $(md5sum < $NGS/ds41-ngshare-tp4.yaml | cut -c1-8) / keeper $(md5sum < $NGS/ngram-keeper.sh | cut -c1-8) '
     '/ cleanup $(md5sum < $NGS/ngram-shm-cleanup.sh | cut -c1-8), harness mbt3.py $(md5sum < $MBT2 | cut -c1-8) / ncmb3.py $(md5sum < $NCMB2 | cut -c1-8), '
     'mb.py $(md5sum < $MBP | cut -c1-8), greedyset.py $(md5sum < $GS | cut -c1-8), stack4/loadid.py $(md5sum < $LD | cut -c1-8), lib-29.sh '
     '$(md5sum < $LIB29 | cut -c1-8), yaml $(md5sum < $Y | cut -c1-8)"' % md5)
# ---------------------------------------------------------------- the header (everything above `set -u`)
cut("#!/bin/bash\n", "set -u\n", P("header.sh").replace("SRCMD5 below", "md5 %s" % md5))
# ---------------------------------------------------------------- the tag, knobs, function names, out dir, the summary files
for a, b in (("$R/xdsm/zzk25s", "$R/xdsm/zzzlaunch"),):
    assert s.count(a) >= 1, a
    s = s.replace(a, b)
s = re.sub(r"(?<![A-Za-z])xk25s(?![A-Za-z0-9])", "xlch", s)
s = s.replace("K25S_", "LCH_")
s = re.sub(r"\bk25id\b", "lchid", s)
s = re.sub(r"\bk25reads\b", "lchreads", s)
s = re.sub(r"\bk25summary\b", "lchsummary", s)
s = s.replace("$O/k25sum.json", "$O/lchsum.json").replace("$O/k25sum.txt", "$O/lchsum.txt")
assert "xk25s" not in s and "K25S_" not in s and "k25id" not in s and "k25summary" not in s
n_edits += 1
# ---------------------------------------------------------------- knobs
rep("ITEM=${LCH_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzk25s}", "ITEM=${LCH_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzzlaunch}")
rep("YDS=$HD/k25sum.py; SAS=$HD/k25sum.py;", "YDS=$HD/lchsum.py; SAS=$HD/lchsum.py; LAQ=$HD/lchaql.py; AQL=$HD/aqlcp.py;")
rep("MXV=${LCH_MX:-12}; EGEN=${LCH_EGEN:-512}; ARCV=0; DOID=0; STAMPN=0",
    "MXV=${LCH_MX:-12}; EGEN=${LCH_EGEN:-512}; ARCV=0; DOID=0; STAMPN=${LCH_STAMP:-16}   # zzzlaunch: RADIANCE_AQL_STAMP (one played pass in N on card 0)")
rep("DISKMIN=${LCH_DISKMIN_MB:-11000}; BUDGET=${LCH_BUDGET_MIN:-28};", "DISKMIN=${LCH_DISKMIN_MB:-11000}; BUDGET=${LCH_BUDGET_MIN:-30};")
cut("# zzk25s: lever K25U's arms and readings (module header)\n", "for v in $C8GN $C8N $C16N $C4N $C1N $SWN; do",
    "# zzzlaunch: lever RCO's arms and readings (module header)\n"
    "XPIN=${LCH_XPIN:-2}; RB=0; RZ=1; XREF=${LCH_XREFUSE:-5}   # THE int15 PIN's exl3_x3d (every arm); B = route_copy_ov RB 0 (int15), Z = route_copy_ov RZ 1 (lever RCO = ALL); XREF: swnote's exl3_x3d refusal probe (400)\n"
    "GORD=${LCH_GORDER-B Z Z B}; C16GN=${LCH_C16GN:-1}; POSTN=${LCH_POST:-10}; IDARMS=${LCH_IDARMS-B Z}; GSA=${LCH_GSARM:-Z}; LDA=${LCH_LDARM:-Z}\n"
    "K16A=${LCH_C16ARMS-B Z}; U1A=${LCH_1UARMS-B Z}; ONEU=${LCH_1U:-1}   # c16: a gens block an arm; 1u: per arm one discarded + U1N essay reads (B first)\n"
    "STPMD5=${LCH_STPMD5:-569fd59b}   # the coordinator's steps.py (zzzlaunch/steps.py = zzk25s's = zsa14's)\n"
    "C16N=${LCH_C16N:-1}   # zzzlaunch: 1 = the c16 gens blocks (K16A), 0 = none\n"
    "for v in $XPIN $XREF $C16GN $POSTN $C16N $ONEU $STAMPN; do case $v in ''|*[!0-9]*) echo \"a zzzlaunch knob is not a number ('$v')\"; exit 1 ;; esac; done\n"
    "[ $XPIN = 2 ] || { echo \"LCH_XPIN 2 (THE int15 PIN: X3D + K2.5 gate_up G3 + down U4)\"; exit 1; }\n"
    "[ $XREF -ge 5 ] || { echo \"LCH_XREFUSE >= 5 (0 .. 4 are VALID on the K25U plugin: a probe there SETS the switch)\"; exit 1; }\n"
    "[ $STAMPN = 0 ] || [ $STAMPN -ge 4 ] || { echo \"LCH_STAMP 0 (no stamping) or >= 4 (one pass in N: lchaql.py reads inside each stamped pass; N 1..3 would cost the speed blocks several %)\"; exit 1; }\n"
    "[ $C16GN -ge 1 ] || { echo \"LCH_C16GN >= 1\"; exit 1; }; case $C16N$ONEU in [01][01]) ;; *) echo \"LCH_C16N / LCH_1U 0 or 1\"; exit 1 ;; esac\n"
    "[ -n \"$GORD\" ] && [ -n \"$IDARMS\" ] || { echo \"LCH_GORDER / LCH_IDARMS must name arms\"; exit 1; }\n"
    "for a in $GORD $IDARMS $K16A $U1A $GSA $LDA; do case $a in B|Z) ;; *) echo \"LCH_GORDER / IDARMS / C16ARMS / 1UARMS / GSARM / LDARM take B and Z (got '$a')\"; exit 1 ;; esac; done\n"
    "case \" $IDARMS \" in *\" $GSA \"*) ;; *) echo \"LCH_GSARM $GSA needs its a1 (LCH_IDARMS)\"; exit 1 ;; esac; case \" $IDARMS \" in *\" $LDA \"*) ;; *) echo \"LCH_LDARM $LDA needs its a1 (LCH_IDARMS)\"; exit 1 ;; esac\n"
    "case \" $IDARMS \" in *\" B \"*) ;; *) echo \"LCH_IDARMS needs B (vs ref12's a1)\"; exit 1 ;; esac; case \" $IDARMS \" in *\" Z \"*) ;; *) echo \"LCH_IDARMS needs Z (vs ref12's a1)\"; exit 1 ;; esac\n"
    "[ $GSA = Z ] && [ $LDA = Z ] || echo \"NOTE: LCH_GSARM / LCH_LDARM not Z: the strict gs16 / staggered checks are not under ALL-on\"\n")
# ---------------------------------------------------------------- THE int15 PIN: the base pin carries exl3_x3d XPIN 2 + route_copy_ov 0
rep('\\"sched_cb_hold_ms\\": 0, \\"sched_cb_hold_k\\": $CBHK, \\"exl3_x3d\\": 1, \\"ds41_bi\\": 6,',
    '\\"sched_cb_hold_ms\\": 0, \\"sched_cb_hold_k\\": $CBHK, \\"exl3_x3d\\": $XPIN, \\"route_copy_ov\\": 0, \\"ds41_bi\\": 6,')
# ---------------------------------------------------------------- the arms: B = the int15 pin (route_copy_ov 0), Z = + route_copy_ov 1
line("    B) [ $SW = 1 ] && echo \"{\\\"exl3_x3d\\\": $XB}\" || echo '{}' ;;",
     "    B) [ $SW = 1 ] && echo \"{\\\"route_copy_ov\\\": $RB}\" || echo '{}' ;;   # zzzlaunch: B = THE int15 PIN, route_copy_ov RB 0 (int15 exactly); the mock has no route_copy_ov")
line("    Z) [ $SW = 1 ] && echo \"{\\\"exl3_x3d\\\": $XZ}\" || echo '{}' ;;",
     "    Z) [ $SW = 1 ] && echo \"{\\\"route_copy_ov\\\": $RZ}\" || echo '{}' ;;   # zzzlaunch: lever RCO = ALL, route_copy_ov RZ 1 (a routing copy recorded with its ranges)")
rep('B) echo "B (THE int14 PIN, exl3_x3d $XB: the int14 kernels)" ;;', 'B) echo "B (THE int15 PIN, route_copy_ov $RB: int15 exactly)" ;;')
rep('Z) echo "Z (lever K25U: the int14 pin + exl3_x3d $XZ, X3D + K2.5 gate_up G3 + down U4)" ;;',
    'Z) echo "Z (lever RCO = ALL: the int15 pin + route_copy_ov $RZ, the dispatch after a routing copy may play unordered)" ;;')
rep('NAMEOF() { case $1 in O) echo "O (THE int14 PIN)" ;;', 'NAMEOF() { case $1 in O) echo "O (THE int15 PIN: the int14 pin + exl3_x3d $XPIN, route_copy_ov 0)" ;;')
line('DESC="LEVER K25U ON int14',
     'DESC="LEVER RCO ON int15 (zzzlaunch: image ds41-launch = the int15 image (ds41-k25u) + bin/radiance from ds41-launch 0f65666 (core fn4 route_copy_ov 68), '
     'THE int15 PIN = zzk25s\'s Z (the int14 pin + exl3_x3d $XPIN); arms B = the pin (route_copy_ov $RB), Z = the pin + route_copy_ov $RZ (lever RCO = ALL)) '
     'NST 5 stack5 start (stack4\'s + RADIANCE_LOG_STEPS=1, E1 declared off, HB ranks $HBRANKS) + RADIANCE_DS41_MHC_FUSE=0 (M1 declared) + RADIANCE_DS41_GQ=3 '
     '+ RADIANCE_DS41_E2=1 + RADIANCE_DS41_KV_REPLICATED=1 (KVR) + A4 table + A11 ranged entries (zzk25s\'s = ref14\'s copies) + RADIANCE_FN4_AR_XPUSH_MIN=$XPMIN '
     '$([ $STAMPN -gt 0 ] && echo "+ RADIANCE_AQL_STAMP=$STAMPN (one played pass in $STAMPN on card 0, both arms)" || echo "(no stamping)") | SP / ARC / CBH / SA off, '
     'ngshare, no ATTACH | the pin = stack5\'s S5T with draft_sample 0 + SP 0 / 0 + $MJ + the fixed pin (ds41_psk_k $PSKK; draft_lend 1; dspark_conf 1, dspark_conf_rows 1)"')
# ---------------------------------------------------------------- swnote: + route_copy_ov 2 -> 400 and route_copy_ov 0 / "rco" at start
rep("c13=400 c14=400 c15=400 c4 a sets=\"\" bad=\"\"", "c13=400 c14=400 c15=400 c16=400 c4 a sets=\"\" bad=\"\"")
rep("c15=$(fn4code '{\"heat_min_gain_x100\": 10001}'); fi",
    "c15=$(fn4code '{\"heat_min_gain_x100\": 10001}'); c16=$(fn4code '{\"route_copy_ov\": 2}'); fi   # zzzlaunch: + lever RCO's refusal")
rep("place_swap_lent 2 / heat_min_gain_x100 10001\") -> 400;", "place_swap_lent 2 / heat_min_gain_x100 10001 / route_copy_ov 2\") -> 400;")
rep('heat_min_gain_x100 150, heat_hyst_x100 125 and "cbh" at start', 'heat_min_gain_x100 150, heat_hyst_x100 125, route_copy_ov 0, "cbh" and "rco" at start')
rep("place_swap_lent 2 -> HTTP $c14, heat_min_gain_x100 10001 -> HTTP $c15", "place_swap_lent 2 -> HTTP $c14, heat_min_gain_x100 10001 -> HTTP $c15, route_copy_ov 2 -> HTTP $c16")
rep('''    ok = ok and w.get("ar_xpush") == 0 and w.get("gg_gemv_m16") == 0 and w.get("place_swap_lent") == 0 and w.get("heat_min_gain_x100") == 150 and w.get("heat_hyst_x100") == 125''',
    '''    ok = ok and w.get("ar_xpush") == 0 and w.get("gg_gemv_m16") == 0 and w.get("place_swap_lent") == 0 and w.get("heat_min_gain_x100") == 150 and w.get("heat_hyst_x100") == 125
    ok = ok and w.get("route_copy_ov") == 0 and isinstance(s0.get("rco"), dict) and s0["rco"].get("recordings") == 0   # zzzlaunch: lever RCO off at start''')
rep('[ "$c14" = 400 ] && [ "$c15" = 400 ] && [ -z "$bad" ]', '[ "$c14" = 400 ] && [ "$c15" = 400 ] && [ "$c16" = 400 ] && [ -z "$bad" ]')
rep('"heat_min_gain_x100", "heat_hyst_x100", "exl3_x3d", "sched_cb_hold_ms", "sched_cb_hold_k", "ds41_mix_wv",',
    '"heat_min_gain_x100", "heat_hyst_x100", "route_copy_ov", "exl3_x3d", "sched_cb_hold_ms", "sched_cb_hold_k", "ds41_mix_wv",')
# ---------------------------------------------------------------- the pin check: exl3_x3d XPIN in every arm; route_copy_ov 0 (O, B) / 1 (Z)
rep('and d["ds41_arcoded"] == 0 and d["exl3_x3d"] == int(sys.argv[12] if a == "Z" else sys.argv[11])',
    'and d["ds41_arcoded"] == 0 and d["exl3_x3d"] == int(sys.argv[11]) and d["route_copy_ov"] == (int(sys.argv[12]) if a == "Z" else 0)')
rep('"sched_cb_hold_ms", "exl3_x3d", "place_swap_lent", "heat_min_gain_x100", "heat_hyst_x100"))\' "$(pinj $a)" $SW $a $MHC $D0 $CAP $MXV $CBHK $F4V $SWLG $XB $XZ',
    '"sched_cb_hold_ms", "exl3_x3d", "route_copy_ov", "place_swap_lent", "heat_min_gain_x100", "heat_hyst_x100"))\' "$(pinj $a)" $SW $a $MHC $D0 $CAP $MXV $CBHK $F4V $SWLG $XPIN $RZ')
# ---------------------------------------------------------------- x3dchk / x3dsay -> lever RCO's (+ clockpair, clk, aqlread)
cut('X3DN=0; X3DLOG=""   # lever K25U\'s switch read back', "lchid() {", P("funcs.sh"))
# ---------------------------------------------------------------- the c8 gens blocks: clock pairs before and after (the AQL windows)
rep('''    budok "c8 work-based block $i $a (gens $C8GN)" 2 || continue
    mbx 8 $i $a G || true
  done''', '''    budok "c8 work-based block $i $a (gens $C8GN)" 2 || continue
    clk c8-$i-G-$a; mbx 8 $i $a G || true; clk c8-$i-G-$a   # zzzlaunch: clock pairs for the AQL window (aqlread)
  done''')
# ---------------------------------------------------------------- the summary: lchsum.py's switches, then the AQL read
rep("--base B --zarm Z --xb $XB --xz $XZ --c8gens $C8GN --c16gens $C16GN --essay-md5 $ESSMD5 --json $O/lchsum.json -> RESULT xlch K25S lines (BLOCK / ARM / MODEL / C16 / 1u / REFS / HEADLINE)\"",
    "--base B --zarm Z --xb $RB --xz $RZ --xpin $XPIN --c8gens $C8GN --c16gens $C16GN --essay-md5 $ESSMD5 --json $O/lchsum.json -> RESULT xlch LCH lines (BLOCK / ARM / MODEL / C16 / 1u / REFS / HEADLINE)\"; aqlread")
rep("--base B --zarm Z --xb $XB --xz $XZ --c8gens $C8GN --c16gens $C16GN --essay-md5 $ESSMD5 --json $O/lchsum.json > $O/lchsum.txt",
    "--base B --zarm Z --xb $RB --xz $RZ --xpin $XPIN --c8gens $C8GN --c16gens $C16GN --essay-md5 $ESSMD5 --json $O/lchsum.json > $O/lchsum.txt")
rep("""    while IFS= read -r l; do xlog "${l:0:2600}"; done < <(grep -h '^K25S' $O/lchsum.txt 2>/dev/null)
    grep -q '^K25S' $O/lchsum.txt 2>/dev/null || xlog "summary: VIRHE k25sum.py printed no K25S line ($(tail -3 $O/lchsum.txt 2>/dev/null | tr '\\n' ' ' | cut -c1-300))"
  fi""", """    while IFS= read -r l; do xlog "${l:0:2600}"; done < <(grep -h '^LCH ' $O/lchsum.txt 2>/dev/null)
    grep -q '^LCH ' $O/lchsum.txt 2>/dev/null || xlog "summary: VIRHE lchsum.py printed no LCH line ($(tail -3 $O/lchsum.txt 2>/dev/null | tr '\\n' ' ' | cut -c1-300))"
    aqlread   # zzzlaunch: THE AQL READ (lchaql.py over the c8 gens blocks' stamped passes)
  fi""")
rep("lchsummary() {   # ncmatch.py's matched-prompt newcomer sums (c8 gens), steps.py's raw listings, then k25sum.py (every K25S line)",
    "lchsummary() {   # ncmatch.py's matched-prompt newcomer sums (c8 gens), steps.py's raw listings, then lchsum.py (every LCH line), then aqlread (LCHAQL lines)")
# ---------------------------------------------------------------- dstart: the pin's exl3_x3d is XPIN; route_copy_ov 0 under the pin
rep('''  if [ "$DRY" = 1 ]; then echo "+ GET /admin/fn4: exl3_x3d $XB under the pin (B = the int14 kernels; exl3_x3d $XREF refused by swnote) and the start log's kernel selection line (else VIRHE: not read)"''',
    '''  if [ "$DRY" = 1 ]; then echo "+ GET /admin/fn4: exl3_x3d $XPIN and route_copy_ov 0 under the pin (B = int15; exl3_x3d $XREF and route_copy_ov 2 refused by swnote) and the start log's kernel selection line (else VIRHE: not read)"''')
rep('''    sc=$(fn4get | python3 -c 'import json,sys; print((json.load(sys.stdin).get("switches") or {}).get("exl3_x3d"))' 2>/dev/null)
    if [ "$sc" = "$XB" ]; then xlog "$tag lever K25U: GET /admin/fn4 exl3_x3d $sc under the pin (B)''',
    '''    sc=$(fn4get | python3 -c 'import json,sys; w = json.load(sys.stdin).get("switches") or {}; print("%s/%s" % (w.get("exl3_x3d"), w.get("route_copy_ov")))' 2>/dev/null)
    if [ "$sc" = "$XPIN/0" ]; then xlog "$tag THE int15 PIN: GET /admin/fn4 exl3_x3d/route_copy_ov $sc under the pin (B)''')
rep('''    elif [ $SW = 0 ]; then xlog "$tag lever K25U: INFO (LCH_SW=0 mock) GET /admin/fn4 exl3_x3d ${sc:-none} -- not posted under the mock"
    else xlog "$tag lever K25U: VIRHE GET /admin/fn4 exl3_x3d '${sc:-none}' under the pin (want $XB) -- K25U cannot be read here: stopped"''',
    '''    elif [ $SW = 0 ]; then xlog "$tag THE int15 PIN: INFO (LCH_SW=0 mock) GET /admin/fn4 exl3_x3d/route_copy_ov ${sc:-none} -- not posted under the mock"
    else xlog "$tag THE int15 PIN: VIRHE GET /admin/fn4 exl3_x3d/route_copy_ov '${sc:-none}' under the pin (want $XPIN/0) -- lever RCO cannot be read here: stopped"''')
rep("lever K25U screen on int14 + the K25U exl3r4d.so;", "lever RCO screen on int15 (+ ds41-launch's bin/radiance), RADIANCE_AQL_STAMP=${STAMPN} one pass in N;")
# ---------------------------------------------------------------- the image literals: the unknown-switch message now ends with route_copy_ov; + RCO's own
rep("B|exl3_x3d, ar_xpush, gg_gemv_m16, place_swap_lent, heat_min_gain_x100, heat_hyst_x100, epoch_bump)",
    "B|exl3_x3d, ar_xpush, gg_gemv_m16, place_swap_lent, heat_min_gain_x100, heat_hyst_x100, route_copy_ov, epoch_bump)\n"
    "B|route_copy_ov\nB|RADIANCE_AQL_ROUTE_OV\nB|route copy overlap (fn4 route_copy_ov 1)\nP|launch-PROVENANCE.txt")
# ---------------------------------------------------------------- helpers: lchsum.py / lchaql.py / aqlcp.py selftests, the function list, the dry text
rep("XPT=$(timeout 60 python3 -I -B $YDS selftest 2>&1 | tail -1); case \"$XPT\" in \"SELFTEST OK\"*) ;; *) HMISS=\"$HMISS selftest:$YDS(${XPT:0:300})\" ;; esac",
    "XPT=$(timeout 60 python3 -I -B $YDS selftest 2>&1 | tail -1); case \"$XPT\" in \"SELFTEST OK\"*) ;; *) HMISS=\"$HMISS selftest:$YDS(${XPT:0:300})\" ;; esac\n"
    "LQT=$(timeout 60 python3 -I -B $LAQ selftest 2>&1 | tail -1); case \"$LQT\" in \"lchaql selftest: OK\"*) ;; *) HMISS=\"$HMISS selftest:$LAQ(${LQT:0:300})\" ;; esac\n"
    "AQT=$(timeout 120 python3 -I -B $AQL selftest 2>&1 | tail -1); case \"$AQT\" in *OK*) ;; *) HMISS=\"$HMISS selftest:$AQL(${AQT:0:300})\" ;; esac")
rep("| k25sum.py $XPT | ncmatch.py $NMT2 | steps.py md5 $STPMD5 |", "| lchsum.py $XPT | lchaql.py $LQT | aqlcp.py $AQT | ncmatch.py $NMT2 | steps.py md5 $STPMD5 |")
rep('FNS="stagx tabs14 rankrb a11line a4line xpline x3dchk x3dsay lchid lchreads ncmset stepsx lchsummary ngclean ',
    'FNS="stagx tabs14 rankrb a11line a4line xpline x3dchk x3dsay clockpair clk aqlread lchid lchreads ncmset stepsx lchsummary ngclean ')
rep('''  for a in O B Z; do echo "+ arm $a ($(NAMEOF $a)): $(armj $a)"; done''',
    '''  for a in O B Z; do echo "+ arm $a ($(NAMEOF $a)): $(armj $a)"; done
  echo "+ zzzlaunch: RADIANCE_AQL_STAMP=$STAMPN at the start (one played pass in $STAMPN on card 0 timed: both arms; the stamps -> $STATSDIR/stamps.csv); clock pairs around each c8 gens block -> $O/clock-c8-<i>-G-<arm>.txt; the AQL read after lchsum.py (aqlcp.py monowin + lchaql.py read / compare)"''')
rep("exl3_x3d asserted before every block (B $XB, Z $XZ); refusal probe exl3_x3d $XREF",
    "route_copy_ov asserted before every block (B $RB, Z $RZ) + exl3_x3d $XPIN (the int15 pin) + GET \\\"rco\\\" logged; refusal probes exl3_x3d $XREF / route_copy_ov 2 | AQL stamps one pass in $STAMPN, read by lchaql.py per arm")
# ---------------------------------------------------------------- rankrb: the int15 pin's words (the pin itself is pinj O: route_copy_ov 0 + exl3_x3d 2 compared)
rep('"sched_row_cap", "exl3_x3d", "mhc_fuse", "draft_depth0", "ds41_gq", "ds41_e2", "ds41_mix_wv")',
    '"sched_row_cap", "exl3_x3d", "route_copy_ov", "mhc_fuse", "draft_depth0", "ds41_gq", "ds41_e2", "ds41_mix_wv")')
rep('THE int14 PIN read back', 'THE int15 PIN read back', 4)
# ---------------------------------------------------------------- main's start line
line('xlog "$ITEM: start IMAGE=$IMAGE',
     'xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f \'{{.Id}}\' $IMAGE 2>/dev/null | cut -c8-19))$([ $SW = 1 ] || echo " | MOCK-RUN LCH_SW=0 (NOT A READING): NO int9b .. int15 '
     'switch posted, route_copy_ov / exl3_x3d not posted, SYNTHETIC stamps, cross-run references INFO") | LEVER RCO ON int15 (core fn4 route_copy_ov, ds41-launch 0f65666), ONE FRESH START on '
     'ngshare (no ATTACH), THE int15 PIN right after it (fn4) and read back on every rank: $DESC | MAXSEQS ${LCH_MAXSEQS:-the base value} | tables: A4 $A4MD5 ($A4N lines), '
     'A11 $A11MD5 ($A11NE entries), NOT re-benched | exl3r4d.so md5 wanted ${LCH_PLUGMD5:-7b2f66d6} (checked next) | 1 identity warm-up, a1 [$IDARMS] vs ref12 a1, gs$GSC-$GSA, '
     'staggered ld-$LDA under ALL-on (ANY difference: STOP before the speed blocks) | 2 c8 work-based gens $C8GN [$GORD] (clock pairs: the AQL windows) | 3 c16 gens $C16GN [$K16A] '
     '| 4 1u essay [$U1A] 1 discarded + $U1N each | route_copy_ov + exl3_x3d asserted before every block, refusal probes exl3_x3d $XREF / route_copy_ov 2 | AQL: RADIANCE_AQL_STAMP=$STAMPN, '
     'lchaql.py per arm | budget $BUDGET min | no evict"')
# ---------------------------------------------------------------- checks on the result
main = s.split("# ---------------------------------------------------------------- main")[1] if "# ---------------------------------------------------------------- main" in s else s
for w in ("$XB", "$XZ", "K25S'", "'^K25S'"):
    assert w not in s, "the body still names %s" % w
for w in ("lchid", "lchreads", "lchsummary", "aqlread", "clk c8-$i-G-$a", "route_copy_ov", "$LAQ", "STAMPN=${LCH_STAMP:-16}"):
    assert w in s, "the body lacks %s" % w
open(dst, "w").write(s)
os.chmod(dst, 0o755)
print("%s: %d edits, %d lines (from %s md5 %s)" % (dst, n_edits, s.count("\n"), src, md5))
