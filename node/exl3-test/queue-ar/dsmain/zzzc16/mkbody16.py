#!/usr/bin/env python3
"""mkbody16.py SRC PARTS DST [SRCMD5] -- zzzc16/dsm-zzzc16.sh from zzzfuse/dsm-zzzfuse.sh (SRC, md5 SRCMD5 2a0f352d as staged + run 13.11.):
every edit an exact, asserted replacement (tag xzfuse -> xzc16, knobs ZFU_ -> ZC16_; the fusion screen's arms B / F / Y / R1 / R2 / M / R ->
THE int16 confirmation's B (int15) / F (int16 = the pin O); THE int16 PIN = the int15 pin + ds41_rfuse 2 + ds41_mix_fin 0 + route_copy_ov 1
posted as O and read back; the start + RADIANCE_DS41_RFUSE=2; no P0; swlwant aligned with int15 / int16 (every arm place_swap_lent 1 +
heat_min_gain_x100 75 on the ranks: zzzfuse's 1u F VIRHEs were the stale int12 "0 150"); zzzfuse's identity / readings / summary / start out,
zzzc16's in). PARTS = zzzc16/mkparts16 (header.sh: the module header; knobs.sh: the arms / readings knobs; funcs.sh: zzzc16's functions in
place of zzzfuse's fusewant .. fusesummary; dstart.sh: fusestart + dstart)."""
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


# ---------------------------------------------------------------- before the tag change: the strings that name zzzfuse's own files / provenance
rep("dsm-zzzfuse.sh $(md5sum < $0 | cut -c1-8) (a copy of zzk25s/dsm-zzk25s.sh abbf540f via zzzfuse/mkbody.py)",
    "dsm-zzzc16.sh $(md5sum < $0 | cut -c1-8) (a copy of zzzfuse/dsm-zzzfuse.sh %s via zzzc16/mkbody16.py)" % md5)
rep('HMD5="zzzfuse/fusesum.py $(md5sum < $YDS | cut -c1-8) (zzk25s/k25sum.py\'s copy + the fusion switches / counters / singles), zzzfuse/steps.py '
    '$(md5sum < $STP | cut -c1-8) (= zzk25s\'s, the coordinator\'s), zzzfuse/m16-forms.txt $(md5sum < $A4TAB | cut -c1-8) (= zzk25s\'s = ref14\'s), '
    'zzzfuse/a11-entries.txt $(md5sum < $A11TAB | cut -c1-8) (= zzk25s\'s = ref14\'s), zzzfuse/ncmatch.py $(md5sum < $NCM | cut -c1-8) (= zzk25s\'s = ref14\'s), ',
    'HMD5="zzzc16/zc16sum.py $(md5sum < $YDS | cut -c1-8) (new: the int16 confirmation summary, zzzfuse/fusesum.py\'s step-log readers), zzzc16/steps.py '
    '$(md5sum < $STP | cut -c1-8) (= zzzfuse\'s, the coordinator\'s), zzzc16/m16-forms.txt $(md5sum < $A4TAB | cut -c1-8) (= zzzfuse\'s = ref14\'s), '
    'zzzc16/a11-entries.txt $(md5sum < $A11TAB | cut -c1-8) (= zzzfuse\'s = ref14\'s), zzzc16/ncmatch.py $(md5sum < $NCM | cut -c1-8) (= zzzfuse\'s; selftest only), '
    'references: zzzfuse a1-B $(md5sum < $ZFA1 | cut -c1-8), ref14 cold $(md5sum < $R14C 2>/dev/null | cut -c1-8), ')
# ---------------------------------------------------------------- the header (everything above `set -u`)
cut("#!/bin/bash\n", "set -u\n", P("header.sh").replace("SRCMD5 below", "md5 %s" % md5))
# ---------------------------------------------------------------- the tag: xzfuse -> xzc16, ZFU_ -> ZC16_, the out dir
for a, b in (("$R/xdsm/zzzfuse}", "$R/xdsm/zzzc16}"),):
    assert s.count(a) == 1, a
    s = s.replace(a, b)
s = re.sub(r"(?<![A-Za-z])xzfuse(?![A-Za-z0-9])", "xzc16", s)
s = s.replace("ZFU_", "ZC16_")
assert "xzfuse" not in s and "ZFU_" not in s
n_edits += 1
# ---------------------------------------------------------------- knobs
rep("ITEM=${ZC16_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzzb-fuse}", "ITEM=${ZC16_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzzc-int16}")
rep("YDS=$HD/fusesum.py; SAS=$HD/fusesum.py;", "YDS=$HD/zc16sum.py; SAS=$HD/zc16sum.py;")
cut("# zzzfuse: the fusion levers' arms and readings (module header)\n", "for v in $C8GN $C8N $C16N $C4N $C1N $SWN; do", P("knobs.sh"))
# ---------------------------------------------------------------- THE int16 PIN: the int15 pin + ds41_rfuse 2 + ds41_mix_fin 0 + route_copy_ov 1 (arm O)
rep(r'''\"heat_min_gain_x100\": $SWLG, \"heat_hyst_x100\": 125}"   # ref14: THE int14 PIN (yzint14c's Z)''',
    r'''\"heat_min_gain_x100\": $SWLG, \"heat_hyst_x100\": 125, \"ds41_rfuse\": $FRF, \"ds41_mix_fin\": $FMF, \"route_copy_ov\": $FRC}"   # zzzc16: THE int16 PIN = the int15 pin (ref14's int14 pin + exl3_x3d XB) + ds41_rfuse 2 + ds41_mix_fin 0 + route_copy_ov 1''')
# ---------------------------------------------------------------- the 1u per-rank check: int15 / int16 values in every arm (ds-main-port 13.11.)
line("swlwant() { case $1 in O|W|B|S1|S2) echo \"1 $SWLG\" ;; Z) [ $SWLON = 1 ] && echo \"1 $SWLG\" || echo \"0 150\" ;; *) echo \"0 150\" ;; esac; }",
     "swlwant() { echo \"1 $SWLG\"; }   # ARM -> place_swap_lent heat_min_gain_x100 the ranks must read (blockx): EVERY arm here (B = int15, F / O = int16) runs SWL's winner, "
     "place_swap_lent 1 + heat_min_gain_x100 SWLG 75 (zzzfuse's 1u F VIRHEs were the stale int12 expectation '0 150'; ds-main-port 13.11.)")
# ---------------------------------------------------------------- the arms: B (int15) / F (int16)
cut("    B) [ $SW = 1 ] && echo '{\"ds41_rfuse\": 0, \"ds41_mix_fin\": 0, \"route_copy_ov\": 0}' || echo '{}' ;;", "    # (zsa14's S1 / S2 lever-SA arms removed)",
    "    B) [ $SW = 1 ] && echo '{\"ds41_rfuse\": 0, \"ds41_mix_fin\": 0, \"route_copy_ov\": 0}' || echo '{}' ;;   # zzzc16: B = int15 (THE int16 PIN with every fusion off: the int15 kernels and issue)\n"
    "    F) [ $SW = 1 ] && echo \"{\\\"ds41_rfuse\\\": $FRF, \\\"ds41_mix_fin\\\": $FMF, \\\"route_copy_ov\\\": $FRC}\" || echo '{}' ;;   # zzzc16: F = int16 (= the pin O: ds41_rfuse 2 + ds41_mix_fin 0 + route_copy_ov 1)\n"
    "    # (zzzfuse's Y / R1 / R2 / M / R arms removed)\n")
rep('O) echo "O (THE int14 PIN)" ;;', 'O) echo "O (THE int16 PIN)" ;;')
rep('B) echo "B (THE int15 PIN, every fusion off: the int15 kernels)" ;; F) echo "F (ALL: ds41_rfuse $FRF + ds41_mix_fin $FMF + route_copy_ov $FRC)" ;; Y) echo "Y (identity: every exact form on, ds41_rfuse $YRF + ds41_mix_fin $YMF + route_copy_ov 1)" ;; '
    'R1) echo "R1 (FUSE-R form 1: ds41_rfuse 1)" ;; R2) echo "R2 (FUSE-R form 2: ds41_rfuse 2)" ;; M) echo "M (FUSE-M: ds41_mix_fin 1)" ;; R) echo "R (RCO: route_copy_ov 1)" ;;',
    'B) echo "B (int15: THE int16 PIN with ds41_rfuse 0 + ds41_mix_fin 0 + route_copy_ov 0)" ;; F) echo "F (int16 = THE int16 PIN: ds41_rfuse $FRF + ds41_mix_fin $FMF + route_copy_ov $FRC)" ;;')
line('DESC="LEVERS FUSE-R / FUSE-M ON int15',
     'DESC="THE int16 CONFIRMATION (zzzc16: image ds41-fuse = int15 + ds41-fuse\'s radiance / r4d_selftest / libr4d.so / deepseek41.so; THE int16 PIN O = the int15 pin (the int14 pin + exl3_x3d $XB) '
     '+ ds41_rfuse $FRF + ds41_mix_fin $FMF + route_copy_ov $FRC; arms B = int15 (the three fusion switches 0), F = int16 (= O)) NST 5 stack5 start (stack4\'s + RADIANCE_LOG_STEPS=1, E1 declared off, HB ranks $HBRANKS) '
     '+ RADIANCE_DS41_MHC_FUSE=0 (M1 declared) + RADIANCE_DS41_GQ=3 + RADIANCE_DS41_E2=1 + RADIANCE_DS41_KV_REPLICATED=1 (KVR) + A4 table + A11 ranged entries '
     '(zzzfuse\'s = ref14\'s copies) + RADIANCE_FN4_AR_XPUSH_MIN=$XPMIN + RADIANCE_DS41_RFUSE=$RFDECL (FUSE-R declared and started at form $RFDECL; RCO and FUSE-M by fn4 only) $([ $STAMPN -gt 0 ] && echo "+ RADIANCE_AQL_STAMP=$STAMPN" || echo "(no stamping)") | SP / ARC / CBH / SA off, '
     'ngshare, no ATTACH | the pin = stack5\'s S5T with draft_sample 0 + SP 0 / 0 + $MJ + the fixed pin (ds41_psk_k $PSKK; draft_lend 1; dspark_conf 1, dspark_conf_rows 1)"')
rep('    case "$r" in OK*) fusechk $1; return $? ;; esac   # zzzfuse: + ds41_rfuse / ds41_mix_fin / exl3_x3d asserted',
    '    case "$r" in OK*) fusechk $1; return $? ;; esac   # zzzc16: + ds41_rfuse / ds41_mix_fin / route_copy_ov / exl3_x3d asserted')
# ---------------------------------------------------------------- mbx: c4's gens count (C4GN)
rep('wo="--window gens --gens $([ $n = 16 ] && echo $C16GN || echo $C8GN) --edge 0.5"',
    'wo="--window gens --gens $(case $n in 16) echo $C16GN ;; 4) echo $C4GN ;; *) echo $C8GN ;; esac) --edge 0.5"')
# ---------------------------------------------------------------- rankrb: THE int16 PIN (the fusion switches named first)
rep("THE int14 PIN read back -- GET /admin/fn4 ds41_bi 6,", "THE int16 PIN read back -- GET /admin/fn4 ds41_rfuse $FRF, ds41_mix_fin $FMF, route_copy_ov $FRC, exl3_x3d $XB, ds41_bi 6,")
rep("THE int14 PIN read back", "THE int16 PIN read back", 3)
rep('K = ("ds41_bi", "gg_pf", "ar_xpush", "gg_gemv_m16", "place_swap_lent", "heat_min_gain_x100", "heat_hyst_x100", "sched_row_cap", "exl3_x3d",',
    'K = ("ds41_rfuse", "ds41_mix_fin", "route_copy_ov", "ds41_bi", "gg_pf", "ar_xpush", "gg_gemv_m16", "place_swap_lent", "heat_min_gain_x100", "heat_hyst_x100", "sched_row_cap", "exl3_x3d",')
# ---------------------------------------------------------------- the pin check: O / F = THE int16 PIN, B = int15 (the fusion switches 0)
cut("for a in O B F Y R1 R2 M R; do python3 -c 'import json,sys; d=json.loads(sys.argv[1]); sw = sys.argv[2] == \"1\"; a = sys.argv[3]\n",
    '[ "$(sha256sum $MODELDIR/$SIDECAR',
    r"""for a in O B F; do python3 -c 'import json,sys; d=json.loads(sys.argv[1]); sw = sys.argv[2] == "1"; a = sys.argv[3]
assert d["ds41_psk"] == 4 and d["ds41_psk_cr"] == 1 and d["draft_sample"] == 0 and d["draft_lend"] == 1 and d["dspark_conf"] == 1 and d["dspark_conf_rows"] == 1
assert d["stage_short"] == 0 and d["stage_pick"] == 0 and d["place_hb"] == 0
if sw:
    assert d["mhc_fuse"] == int(sys.argv[4]) and d["ix_lds"] == 0 and d["ix_ni"] == 0 and d["draft_depth0"] == int(sys.argv[5]) and d["sched_row_cap"] == int(sys.argv[6]) and d["stage_full_min"] == 0 and d["stage_full_pick"] == 0
    assert d["ds41_mix_wv"] == int(sys.argv[7]) and d["ds41_gq"] == 3 and d["ds41_e2"] == 1 and d["ds41_arcoded"] == 0 and d["exl3_x3d"] == int(sys.argv[11])
    assert d["sched_cb_hold_ms"] == 0 and d["sched_cb_hold_k"] == int(sys.argv[8]) and d["ds41_shexp_lane"] == int(sys.argv[9]) and "exl3_x3d_pf" not in d
    g = int(sys.argv[10])
    assert d["ds41_bi"] == 6 and d["gg_pf"] == 2 and d["ar_xpush"] == 1 and d["gg_gemv_m16"] == 1 and d["place_swap_lent"] == 1 and d["heat_min_gain_x100"] == g and d["heat_hyst_x100"] == 125
    i16 = tuple(int(x) for x in sys.argv[12:15])
    assert i16 == (2, 0, 1) and (d["ds41_rfuse"], d["ds41_mix_fin"], d["route_copy_ov"]) == {"O": i16, "F": i16, "B": (0, 0, 0)}[a]
else: assert d["draft_depth0"] == 1 and not any(k in d for k in ("gg_gemv_m16", "exl3_x3d_pf", "ar_xpush", "mhc_fuse", "ix_lds", "ix_ni", "ds41_arcoded", "stage_full_min", "sched_row_cap", "ds41_mix_wv", "ds41_gq", "ds41_e2", "sched_cb_hold_ms", "exl3_x3d", "place_swap_lent", "heat_min_gain_x100", "heat_hyst_x100", "ds41_rfuse", "ds41_mix_fin", "route_copy_ov"))' "$(pinj $a)" $SW $a $MHC $D0 $CAP $MXV $CBHK $F4V $SWLG $XB $FRF $FMF $FRC 2>/dev/null || HMISS="$HMISS json:$a"; done
[ -s "$ZFA1" ] || HMISS="$HMISS file:$ZFA1(zzzfuse's a1-B: the identity reference)"
[ $COLD = 0 ] || [ "$(md5sum < $R14C 2>/dev/null | cut -c1-8)" = "$R14CMD5" ] || HMISS="$HMISS file:$R14C(ref14's $CDEP answer, want md5 $R14CMD5)"
""")
# ---------------------------------------------------------------- helpers: zc16sum.py (selftest), the function list, the dry text
rep("| fusesum.py $XPT | ncmatch.py $NMT2 | steps.py md5 $STPMD5 |", "| zc16sum.py $XPT | ncmatch.py $NMT2 | steps.py md5 $STPMD5 |")
line('FNS="stagx tabs14 rankrb a11line a4line xpline fusewant fusechk fusesay p0fuse fuseid fusectr fusereads ncmset stepsx fusesummary fusestart ngclean',
     'FNS="stagx tabs14 rankrb a11line a4line xpline fusewant fusechk fusesay pin16 c16id fusectr coldid c16reads c16summary fusestart ngclean blockx i11g swlwant log run mkd restart check '
     'down stopfile best_vars desc serve_args xlog elapsed armj NAMEOF pinj stackenv keeplog capture thermline dfree diskok cmpfile idrec identw modes vramstart '
     'fn4get fn4set nudge fn4code pcreset arm msnap mdelta ssnap gate1 alone cmpx budok mwin ctrinfo i11d monons utcnow drafted readback swnote capd mbx cbhd cbhg '
     'gsx gpurun dstart idok gt48 coldx"')
rep('  for a in O B F Y R1 R2 M R; do echo "+ arm $a ($(NAMEOF $a)): $(armj $a)"; done', '  for a in O B F; do echo "+ arm $a ($(NAMEOF $a)): $(armj $a)"; done')
line('  echo "+ readings: 0 P0 one card $P0UUID',
     '  echo "+ readings: 1 identity under F (THE int16 PIN): warm-up vs stock, a1-F vs ref12 a1 ($([ -s $REF12A1 ] && echo present || echo MISSING)) + vs zzzfuse a1-B ($([ -s $ZFA1 ] && echo present || echo MISSING)), '
     'gs$GSC-F vs a1-F (MAXSEQS ${ZC16_MAXSEQS:-base}), staggered ld-F vs a1-F (conc $LDCONC, $LDPF-token prefills) -- ANY difference STOPS before the speed blocks | 2 1u ABBA [$U1AB] 1 discarded + $U1N essay '
     'reads a block ($([ $ONEU = 1 ] && echo on || echo off); answers vs $ESSMD5) | 3 c4 gens $C4GN $([ $C4WARMUP = 1 ] && echo "1 discarded ($C4WA) + ")[$C4AB] ($([ $C4ON = 1 ] && echo on || echo off)) '
     '| 4 cold $CDEP F ($([ $COLD = 1 ] && echo on || echo off); vs ref14 $(basename $R14C) $([ -s $R14C ] && echo "present, md5 $(md5sum < $R14C | cut -c1-8)" || echo MISSING); zzzfuse\'s $([ -s $ZFC ] && echo present || echo "absent: not compared")) '
     '| ds41_rfuse / ds41_mix_fin / route_copy_ov / exl3_x3d asserted before every block; refusal probes exl3_x3d $XREF, ds41_rfuse 3, ds41_mix_fin 2, route_copy_ov 2 | budget $BUDGET min '
     '| four cards $(cat /sys/class/drm/card*/device/unique_id 2>/dev/null | wc -l) unique_id | n-gram tables in $NGDIR now: $(ls $NGDIR/ngram-*.tab 2>/dev/null | wc -l) | free RAM $(free -g | awk \'NR==2{print $7}\') GiB '
     '| /tmp inodes $(df --output=ipcent /tmp 2>/dev/null | tail -1 | tr -d \' \'), / inodes $(df --output=ipcent / 2>/dev/null | tail -1 | tr -d \' \')"')
# ---------------------------------------------------------------- the section: zzzfuse's identity / readings / summary / start -> zzzc16's
rep("# ---------------------------------------------------------------- zzzfuse's own functions (zsa14's: ref14's tables, the pin read back and staggered\n"
    "# check; zzzfuse's: the fusion switches read back, the P0, identity + the stop, the gens / singles / 1u readings, the summary, the start)\n",
    "# ---------------------------------------------------------------- zzzc16's own functions (zsa14's: ref14's tables, the pin read back and staggered\n"
    "# check; zzzc16's: the fusion switches read back, the int16 pin's engagement, identity + the stop, 1u ABBA, c4 gens, cold, the summary, the start)\n")
cut('FUCHKN=0; FUCHKLOG=""', "ngclean() {", P("funcs.sh"))
cut("fusestart() {", "trap 'down;", P("dstart.sh"))
# ---------------------------------------------------------------- main
line('xlog "$ITEM: start IMAGE=$IMAGE',
     'xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f \'{{.Id}}\' $IMAGE 2>/dev/null | cut -c8-19))$([ $SW = 1 ] || echo " | MOCK-RUN ZC16_SW=0 (NOT A READING): NO int9b .. int16 '
     'switch posted, the fusion switches not posted, cross-run references INFO") | THE int16 CONFIRMATION (int15 + ds41_rfuse 2 + route_copy_ov 1; ds41_mix_fin 0), ONE FRESH START on ngshare (no ATTACH) '
     'with RADIANCE_DS41_RFUSE=$RFDECL, THE int16 PIN posted right after it (fn4) and read back (fn4 every switch, every rank, the declare lines, rco / fuse engagement): $DESC | MAXSEQS ${ZC16_MAXSEQS:-the base value} '
     '| tables: A4 $A4MD5 ($A4N lines), A11 $A11MD5 ($A11NE entries), NOT re-benched | exl3r4d.so md5 wanted ${ZC16_PLUGMD5:-7b2f66d6} (checked next) | 1 identity under F: warm-up, a1-F vs ref12 a1 '
     '+ zzzfuse a1-B, gs$GSC-F, staggered ld-F (ANY difference: STOP before the speed blocks) | 2 1u ABBA [$U1AB] 1 discarded + $U1N each | 3 c4 gens $C4GN [$C4AB] after $C4WARMUP discarded '
     '| 4 cold $CDEP F (optional, the first dropped) | ds41_rfuse / ds41_mix_fin / route_copy_ov / exl3_x3d asserted before every block | budget $BUDGET min | no evict"')
rep("\nfusesummary\nunset $SWITCHES\n", "\nc16summary\nunset $SWITCHES\n")
# ---------------------------------------------------------------- checks: nothing of the fusion screen's arms / P0 / summary is left
main = s.split("# ---------------------------------------------------------------- main")[1]
for w in ("fusesummary", "fusereads", "fuseid", "p0fuse"):
    assert w not in s, "the body still names %s" % w
for w in ("$GORD", "$SINGLES", "$K16A", "$U1A", "$YRF", "$YMF", "$R1ON", "$R2ON", "$MON", "$RCON", "$P0HALT", "$P0SEC", "$P0MARGIN", "$P0STOP", "$P0WHY", "$XZ"):
    assert not re.search(re.escape(w) + r"(?![A-Za-z0-9_])", s), "the body still names %s" % w
left = [l for l in s.split("\n") if "zzzfuse" in l]
for l in left:
    assert any(k in l for k in ("REFZF", "zzzfuse a1-B", "zzzfuse's", "zzzfuse/", "zzzfuse:", "zzzfuse ran", "vs zzzfuse", "zzzfuse /", "zzzfuse,", "after zzzfuse", "a1-F-vs-zzzfuse", "-vs-zzzfuse",
                                "zzzfuse)", "zzzfuse's run")), "an unexpected zzzfuse reference: %s" % l[:200]
assert "c16summary\nunset" in main and "c16reads" in s and "pin16 $tag" in s
open(dst, "w").write(s)
os.chmod(dst, 0o755)
print("%s: %d edits, %d lines (from %s md5 %s)" % (dst, n_edits, s.count("\n"), src, md5))
