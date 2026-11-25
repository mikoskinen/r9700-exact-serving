#!/usr/bin/env python3
"""mkbody.py SRC PARTS DST [SRCMD5] -- zzzfuse/dsm-zzzfuse.sh from zzk25s/dsm-zzk25s.sh (SRC, md5 SRCMD5 abbf540f as staged + run 12./13.11.):
every edit an exact, asserted replacement (tag xk25s -> xzfuse, knobs K25S_ -> ZFU_; lever K25U's arms B / Z -> the fusion arms B / F / Y /
R1 / R2 / M on THE int15 PIN (the int14 pin + exl3_x3d 2); arm() asserts the fusion switches + exl3_x3d (fusechk) instead of exl3_x3d alone;
the image literals + the fuse image's; the start + RADIANCE_DS41_RFUSE=0 (FUSE-R declared, off) and the P0 on one card before it;
zzk25s's identity / readings / summary functions out, zzzfuse's in). PARTS = zzzfuse/mkparts (header.sh: the module header; funcs.sh:
zzzfuse's functions in place of zzk25s's x3dchk .. k25summary; dstart.sh: fusestart + zzzfuse's start, in place of zzk25s's)."""
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


# ---------------------------------------------------------------- before the tag change: the strings that name zzk25s's own files / provenance
rep("dsm-zzk25s.sh $(md5sum < $0 | cut -c1-8) (a copy of zsa14/dsm-zsa14.sh fe5c6c3b via zzk25s/mkbody.py)",
    "dsm-zzzfuse.sh $(md5sum < $0 | cut -c1-8) (a copy of zzk25s/dsm-zzk25s.sh %s via zzzfuse/mkbody.py)" % md5)
rep('HMD5="zzk25s/k25sum.py $(md5sum < $YDS | cut -c1-8) (zsa14/sa14sum.py\'s copy + K25U), zzk25s/steps.py $(md5sum < $STP | cut -c1-8) (= zsa14\'s, the '
    'coordinator\'s), zzk25s/m16-forms.txt $(md5sum < $A4TAB | cut -c1-8) (= zsa14\'s = ref14\'s), zzk25s/a11-entries.txt $(md5sum < $A11TAB | cut -c1-8) '
    '(= zsa14\'s = ref14\'s), zzk25s/ncmatch.py $(md5sum < $NCM | cut -c1-8) (= zsa14\'s = ref14\'s), ',
    'HMD5="zzzfuse/fusesum.py $(md5sum < $YDS | cut -c1-8) (zzk25s/k25sum.py\'s copy + the fusion switches / counters / singles), zzzfuse/steps.py '
    '$(md5sum < $STP | cut -c1-8) (= zzk25s\'s, the coordinator\'s), zzzfuse/m16-forms.txt $(md5sum < $A4TAB | cut -c1-8) (= zzk25s\'s = ref14\'s), '
    'zzzfuse/a11-entries.txt $(md5sum < $A11TAB | cut -c1-8) (= zzk25s\'s = ref14\'s), zzzfuse/ncmatch.py $(md5sum < $NCM | cut -c1-8) (= zzk25s\'s = ref14\'s), ')
# ---------------------------------------------------------------- the header (everything above `set -u`)
cut("#!/bin/bash\n", "set -u\n", P("header.sh").replace("SRCMD5 below", "md5 %s" % md5))
# ---------------------------------------------------------------- the tag: xk25s -> xzfuse, K25S_ -> ZFU_, the out dir
for a, b in (("$R/xdsm/zzk25s", "$R/xdsm/zzzfuse"),):
    assert s.count(a) >= 1, a
    s = s.replace(a, b)
s = re.sub(r"(?<![A-Za-z])xk25s(?![A-Za-z0-9])", "xzfuse", s)
s = s.replace("K25S_", "ZFU_")
assert "xk25s" not in s and "K25S_" not in s
n_edits += 1
# ---------------------------------------------------------------- knobs
rep("ITEM=${ZFU_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzk25s}", "ITEM=${ZFU_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzzb-fuse}")
rep("YDS=$HD/k25sum.py; SAS=$HD/k25sum.py;", "YDS=$HD/fusesum.py; SAS=$HD/fusesum.py;")
line("YZ=${ZFU_YZ:-$R/xdsm/yzint14c/x1}; YZA1=${ZFU_YZA1:-$YZ/yzi-a1-Z.jsonl}",
     "YZ=${ZFU_YZ:-$R/xdsm/zzk25s}; YZA1=${ZFU_YZA1:-$YZ/xk25s-a1-Z.jsonl}   # zzk25s's run (its arm Z = int15): its a1-Z (a1-B here must equal it)")
cut("# zzk25s: lever K25U's arms and readings (module header)\n", "for v in $C8GN $C8N $C16N $C4N $C1N $SWN; do",
    "# zzzfuse: the fusion levers' arms and readings (module header)\n"
    "XB=${ZFU_XB:-2}; XREF=${ZFU_XREFUSE:-5}   # THE int15 PIN's exl3_x3d (2: X3D + K2.5 gate_up G3 + down U4) in every arm; XREF: swnote's refusal probe (400)\n"
    "GORD=${ZFU_GORDER-B F F B}; SINGLES=${ZFU_SINGLES-R2 R1 R M}; C16GN=${ZFU_C16GN:-1}; POSTN=${ZFU_POST:-10}; IDARMS=${ZFU_IDARMS-B Y}; GSA=${ZFU_GSARM:-Y}; LDA=${ZFU_LDARM:-Y}\n"
    "K16A=${ZFU_C16ARMS-B F}; U1A=${ZFU_1UARMS-B F}; ONEU=${ZFU_1U:-1}   # c16: a gens block an arm; 1u: per arm one discarded + U1N essay reads (B first)\n"
    "STPMD5=${ZFU_STPMD5:-569fd59b}   # the coordinator's steps.py (zzzfuse/steps.py = zzk25s's)\n"
    "C16N=${ZFU_C16N:-1}; P0SEC=${ZFU_P0SEC:-300}   # 1 = the c16 gens blocks (K16A), 0 = none; P0: each selftest case bounded at P0SEC s\n"
    "for v in $XB $XREF $C16GN $POSTN $C16N $ONEU $P0SEC ${ZFU_ALLMIX:-0} ${ZFU_ALLRCO:-1}; do case $v in ''|*[!0-9]*) echo \"a zzzfuse knob is not a number ('$v')\"; exit 1 ;; esac; done\n"
    "[ $XB = 2 ] || { echo \"ZFU_XB 2 (THE int15 PIN's exl3_x3d)\"; exit 1; }; case ${ZFU_ALLMIX:-0}${ZFU_ALLRCO:-1} in [01][01]) ;; *) echo \"ZFU_ALLMIX / ZFU_ALLRCO 0 or 1\"; exit 1 ;; esac\n"
    "[ $XREF -ge 5 ] || { echo \"ZFU_XREFUSE >= 5 (0 .. 4 are VALID on the K25U plugin: a probe there SETS the switch)\"; exit 1; }\n"
    "[ $C16GN -ge 1 ] || { echo \"ZFU_C16GN >= 1\"; exit 1; }; case $C16N$ONEU in [01][01]) ;; *) echo \"ZFU_C16N / ZFU_1U 0 or 1\"; exit 1 ;; esac\n"
    "[ -n \"$GORD\" ] && [ -n \"$IDARMS\" ] || { echo \"ZFU_GORDER / ZFU_IDARMS must name arms\"; exit 1; }\n"
    "for a in $GORD $K16A $U1A; do case $a in B|F) ;; *) echo \"ZFU_GORDER / C16ARMS / 1UARMS take B and F (got '$a')\"; exit 1 ;; esac; done\n"
    "for a in $SINGLES; do case $a in R1|R2|M|R) ;; *) echo \"ZFU_SINGLES take R1 R2 M R (got '$a')\"; exit 1 ;; esac; done\n"
    "for a in $IDARMS $GSA $LDA; do case $a in B|Y) ;; *) echo \"ZFU_IDARMS / GSARM / LDARM take B and Y (got '$a')\"; exit 1 ;; esac; done\n"
    "case \" $IDARMS \" in *\" $GSA \"*) ;; *) echo \"ZFU_GSARM $GSA needs its a1 (ZFU_IDARMS)\"; exit 1 ;; esac; case \" $IDARMS \" in *\" $LDA \"*) ;; *) echo \"ZFU_LDARM $LDA needs its a1 (ZFU_IDARMS)\"; exit 1 ;; esac\n"
    "case \" $IDARMS \" in *\" B \"*) ;; *) echo \"ZFU_IDARMS needs B (vs ref12's a1)\"; exit 1 ;; esac; case \" $IDARMS \" in *\" Y \"*) ;; *) echo \"ZFU_IDARMS needs Y (every fusion on, vs ref12's a1)\"; exit 1 ;; esac\n")
# ---------------------------------------------------------------- THE int15 PIN: the int14 pin with exl3_x3d 2 (XB)
rep('\\"sched_cb_hold_k\\": $CBHK, \\"exl3_x3d\\": 1, \\"ds41_bi\\": 6,', '\\"sched_cb_hold_k\\": $CBHK, \\"exl3_x3d\\": $XB, \\"ds41_bi\\": 6,')
# ---------------------------------------------------------------- the arms: B / F / Y / R1 / R2 / M (the fusion switches on top of the int15 pin)
line("    B) [ $SW = 1 ] && echo \"{\\\"exl3_x3d\\\": $XB}\" || echo '{}' ;;",
     "    B) [ $SW = 1 ] && echo '{\"ds41_rfuse\": 0, \"ds41_mix_fin\": 0, \"route_copy_ov\": 0}' || echo '{}' ;;   # zzzfuse: B = THE int15 PIN, every fusion off (the int15 kernels and issue)\n"
     "    F) [ $SW = 1 ] && echo \"{\\\"ds41_rfuse\\\": $FRF, \\\"ds41_mix_fin\\\": $FMF, \\\"route_copy_ov\\\": $FRC}\" || echo '{}' ;;   # ALL: ds41_rfuse 2 + route_copy_ov 1 (+ ds41_mix_fin ZFU_ALLMIX), less what P0 drops\n"
     "    Y) [ $SW = 1 ] && echo \"{\\\"ds41_rfuse\\\": $YRF, \\\"ds41_mix_fin\\\": $YMF, \\\"route_copy_ov\\\": 1}\" || echo '{}' ;;   # identity: every byte-checked form on + RCO\n"
     "    R1) [ $SW = 1 ] && echo '{\"ds41_rfuse\": 1, \"ds41_mix_fin\": 0, \"route_copy_ov\": 0}' || echo '{}' ;;   # FUSE-R form 1: the selection + the sort in one launch\n"
     "    R2) [ $SW = 1 ] && echo '{\"ds41_rfuse\": 2, \"ds41_mix_fin\": 0, \"route_copy_ov\": 0}' || echo '{}' ;;   # FUSE-R form 2: the projection + selection + sort in one launch\n"
     "    M) [ $SW = 1 ] && echo '{\"ds41_rfuse\": 0, \"ds41_mix_fin\": 1, \"route_copy_ov\": 0}' || echo '{}' ;;   # FUSE-M: mhc_mix's FIN form at <= 48 rows\n"
     "    R) [ $SW = 1 ] && echo '{\"ds41_rfuse\": 0, \"ds41_mix_fin\": 0, \"route_copy_ov\": 1}' || echo '{}' ;;   # RCO (ds41-launch 0f65666): the routing copy recorded with its ranges")
line("    Z) [ $SW = 1 ] && echo \"{\\\"exl3_x3d\\\": $XZ}\" || echo '{}' ;;", "    # (zzk25s's Z arm removed)")
rep('B) echo "B (THE int14 PIN, exl3_x3d $XB: the int14 kernels)" ;;',
    'B) echo "B (THE int15 PIN, every fusion off: the int15 kernels)" ;; F) echo "F (ALL: ds41_rfuse $FRF + ds41_mix_fin $FMF + route_copy_ov $FRC)" ;; Y) echo "Y (identity: every exact form on, ds41_rfuse $YRF + ds41_mix_fin $YMF + route_copy_ov 1)" ;; '
    'R1) echo "R1 (FUSE-R form 1: ds41_rfuse 1)" ;; R2) echo "R2 (FUSE-R form 2: ds41_rfuse 2)" ;; M) echo "M (FUSE-M: ds41_mix_fin 1)" ;; R) echo "R (RCO: route_copy_ov 1)" ;;')
rep('  Z) echo "Z (lever K25U: the int14 pin + exl3_x3d $XZ, X3D + K2.5 gate_up G3 + down U4)" ;;', '')
line('DESC="LEVER K25U ON int14',
     'DESC="LEVERS FUSE-R / FUSE-M ON int15 (zzzfuse: image ds41-fuse = int15 + ds41-fuse\'s radiance / r4d_selftest / libr4d.so / deepseek41.so, THE int15 PIN = the int14 pin + exl3_x3d $XB; '
     'arms B = the pin (every fusion 0), F = ALL, Y = every exact form (identity), R1 / R2 = ds41_rfuse 1 / 2, M = ds41_mix_fin 1, R = route_copy_ov 1) NST 5 stack5 start (stack4\'s + RADIANCE_LOG_STEPS=1, E1 declared off, HB ranks $HBRANKS) '
     '+ RADIANCE_DS41_MHC_FUSE=0 (M1 declared) + RADIANCE_DS41_GQ=3 + RADIANCE_DS41_E2=1 + RADIANCE_DS41_KV_REPLICATED=1 (KVR) + A4 table + A11 ranged entries '
     '(zzk25s\'s = ref14\'s copies) + RADIANCE_FN4_AR_XPUSH_MIN=$XPMIN + RADIANCE_DS41_RFUSE=0 (FUSE-R declared, off; RCO and FUSE-M by fn4 only) $([ $STAMPN -gt 0 ] && echo "+ RADIANCE_AQL_STAMP=$STAMPN" || echo "(no stamping)") | SP / ARC / CBH / SA off, '
     'ngshare, no ATTACH | the pin = stack5\'s S5T with draft_sample 0 + SP 0 / 0 + $MJ + the fixed pin (ds41_psk_k $PSKK; draft_lend 1; dspark_conf 1, dspark_conf_rows 1)"')
# the switches the start unsets: + the fusion levers' words
rep("RADIANCE_HEAT_MIN_GAIN RADIANCE_HEAT_HYSTERESIS RADIANCE_FN4_AR_XPUSH_MIN\"", "RADIANCE_HEAT_MIN_GAIN RADIANCE_HEAT_HYSTERESIS RADIANCE_FN4_AR_XPUSH_MIN RADIANCE_DS41_RFUSE RADIANCE_DS41_MIX_FIN RADIANCE_AQL_ROUTE_OV\"")
# ---------------------------------------------------------------- arm(): the fusion switches + exl3_x3d asserted (fusechk)
rep('  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4: every switch of arm $1 applied, nothing pending (polled <= 30 s, a nudge a try)"; x3dchk $1; return 0; }',
    '  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4: every switch of arm $1 applied, nothing pending (polled <= 30 s, a nudge a try)"; fusechk $1; return 0; }')
rep('    case "$r" in OK*) x3dchk $1; return $? ;; esac   # zzk25s: + exl3_x3d asserted (B XB, Z XZ)',
    '    case "$r" in OK*) fusechk $1; return $? ;; esac   # zzzfuse: + ds41_rfuse / ds41_mix_fin / exl3_x3d asserted')
# ---------------------------------------------------------------- the image: + the fuse literals (the registry tail now ends in ds41_rfuse, ds41_mix_fin)
rep("B|exl3_x3d, ar_xpush, gg_gemv_m16, place_swap_lent, heat_min_gain_x100, heat_hyst_x100, epoch_bump)",
    "B|exl3_x3d, ar_xpush, gg_gemv_m16, place_swap_lent, heat_min_gain_x100, heat_hyst_x100, ds41_rfuse, ds41_mix_fin, route_copy_ov, epoch_bump)")
rep('P|int14c-PROVENANCE.txt"', 'P|int14c-PROVENANCE.txt\n'
    'B|ds41_rfuse\nB|ds41_mix_fin\nB|rf_form2_passes\nB|mixfin_launches\nB|route_copy_ov\nB|RADIANCE_AQL_ROUTE_OV\nB|route copy overlap (fn4 route_copy_ov 1)\n'
    'A|RADIANCE_DS41_RFUSE\nA|lever FUSE-R (ds41_rfuse, RADIANCE_DS41_RFUSE, now %d) %s\nA|rad_rfuse_stats\nA|router_gemm_f32_scored_topk_scatter\n'
    'K|router_gemm_f32_scored_topk_scatter\nK|RADIANCE_DS41_MIX_FIN\nK|ds41_mix_fin\nK|rad_fuse_stats\n'
    'R|FUSEGATE\nR|FUSEPERF\nR|rfuse\nR|mixfin\nP|k25u-PROVENANCE.txt\nP|fuse-PROVENANCE.txt"')
rep('xlog "VIRHE $IMAGE is not the int14c image -- missing:', 'xlog "VIRHE $IMAGE is not the ds41-fuse image (int15 + ds41-fuse) -- missing:')
# ---------------------------------------------------------------- the pin check: every arm the int15 pin (exl3_x3d XB), the fusion switches the arm's
cut("for a in O B Z; do python3 -c 'import json,sys; d=json.loads(sys.argv[1]); sw = sys.argv[2] == \"1\"; a = sys.argv[3]\n",
    '[ "$(sha256sum $MODELDIR/$SIDECAR',
    """for a in O B F Y R1 R2 M R; do python3 -c 'import json,sys; d=json.loads(sys.argv[1]); sw = sys.argv[2] == "1"; a = sys.argv[3]
assert d["ds41_psk"] == 4 and d["ds41_psk_cr"] == 1 and d["draft_sample"] == 0 and d["draft_lend"] == 1 and d["dspark_conf"] == 1 and d["dspark_conf_rows"] == 1
assert d["stage_short"] == 0 and d["stage_pick"] == 0 and d["place_hb"] == 0
if sw:
    assert d["mhc_fuse"] == int(sys.argv[4]) and d["ix_lds"] == 0 and d["ix_ni"] == 0 and d["draft_depth0"] == int(sys.argv[5]) and d["sched_row_cap"] == int(sys.argv[6]) and d["stage_full_min"] == 0 and d["stage_full_pick"] == 0
    assert d["ds41_mix_wv"] == int(sys.argv[7]) and d["ds41_gq"] == 3 and d["ds41_e2"] == 1 and d["ds41_arcoded"] == 0 and d["exl3_x3d"] == int(sys.argv[11])
    assert d["sched_cb_hold_ms"] == 0 and d["sched_cb_hold_k"] == int(sys.argv[8]) and d["ds41_shexp_lane"] == int(sys.argv[9]) and "exl3_x3d_pf" not in d
    g = int(sys.argv[10])
    assert d["ds41_bi"] == 6 and d["gg_pf"] == 2 and d["ar_xpush"] == 1 and d["gg_gemv_m16"] == 1 and d["place_swap_lent"] == 1 and d["heat_min_gain_x100"] == g and d["heat_hyst_x100"] == 125
    want = {"O": None, "B": (0, 0, 0), "F": (2, int(sys.argv[12]), int(sys.argv[13])), "Y": (2, 1, 1), "R1": (1, 0, 0), "R2": (2, 0, 0), "M": (0, 1, 0), "R": (0, 0, 1)}[a]
    assert (want is None and not any(k in d for k in ("ds41_rfuse", "ds41_mix_fin", "route_copy_ov"))) or (want is not None and (d["ds41_rfuse"], d["ds41_mix_fin"], d["route_copy_ov"]) == want)
else: assert d["draft_depth0"] == 1 and not any(k in d for k in ("gg_gemv_m16", "exl3_x3d_pf", "ar_xpush", "mhc_fuse", "ix_lds", "ix_ni", "ds41_arcoded", "stage_full_min", "sched_row_cap", "ds41_mix_wv", "ds41_gq", "ds41_e2", "sched_cb_hold_ms", "exl3_x3d", "place_swap_lent", "heat_min_gain_x100", "heat_hyst_x100", "ds41_rfuse", "ds41_mix_fin", "route_copy_ov"))' "$(pinj $a)" $SW $a $MHC $D0 $CAP $MXV $CBHK $F4V $SWLG $XB $FMF $FRC 2>/dev/null || HMISS="$HMISS json:$a"; done
""")
# ---------------------------------------------------------------- helpers: fusesum.py (selftest), the function list, the dry text
rep("| k25sum.py $XPT | ncmatch.py $NMT2 | steps.py md5 $STPMD5 |", "| fusesum.py $XPT | ncmatch.py $NMT2 | steps.py md5 $STPMD5 |")
line('FNS="stagx tabs14 rankrb a11line a4line xpline x3dchk x3dsay k25id k25reads ncmset stepsx k25summary ngclean',
     'FNS="stagx tabs14 rankrb a11line a4line xpline fusewant fusechk fusesay p0fuse fuseid fusectr fusereads ncmset stepsx fusesummary fusestart ngclean blockx i11g swlwant log run mkd restart check '
     'down stopfile best_vars desc serve_args xlog elapsed armj NAMEOF pinj stackenv keeplog capture thermline dfree diskok cmpfile idrec identw modes vramstart '
     'fn4get fn4set nudge fn4code pcreset arm msnap mdelta ssnap gate1 alone cmpx budok mwin ctrinfo i11d monons utcnow drafted readback swnote capd mbx cbhd cbhg '
     'gsx gpurun dstart idok gt48"')
rep('  for a in O B Z; do echo "+ arm $a ($(NAMEOF $a)): $(armj $a)"; done', '  for a in O B F Y R1 R2 M R; do echo "+ arm $a ($(NAMEOF $a)): $(armj $a)"; done')
line('  echo "+ readings: 1 identity warm-up vs stock, a1 [$IDARMS] vs ref12 a1',
     '  echo "+ readings: 0 P0 one card $P0UUID (r4d_selftest --case rfuse / mixfin: bytes + FUSEPERF / FUSEGATE) | 1 identity warm-up vs stock, a1 [$IDARMS] vs ref12 a1 ($([ -s $REF12A1 ] && echo present || echo MISSING)) + a1-B vs zzk25s a1-Z '
     '($([ -s $YZA1 ] && echo present || echo "absent: not compared")), gs$GSC-$GSA vs a1-$GSA (MAXSEQS ${ZFU_MAXSEQS:-base}), staggered ld-$LDA vs a1-$LDA (conc $LDCONC, '
     '$LDPF-token prefills) -- ANY difference STOPS before the speed blocks | 2 c8 work-based gens $C8GN [$GORD] | 3 singles [$SINGLES] one c8 block each | 4 c16 gens $C16GN [$K16A] ($([ $C16N -gt 0 ] && echo on || echo off)) '
     '| 5 1u essay [$U1A] 1 discarded + $U1N each ($([ $ONEU = 1 ] && echo on || echo off); answers vs $ESSMD5) | ds41_rfuse / ds41_mix_fin / route_copy_ov / exl3_x3d asserted before every block; refusal probes exl3_x3d $XREF, ds41_rfuse 3, ds41_mix_fin 2, route_copy_ov 2 '
     '| decode p50 all + full-row bucket + $POSTN after each newcomer | budget $BUDGET min | four cards $(cat /sys/class/drm/card*/device/unique_id 2>/dev/null | wc -l) unique_id '
     '| n-gram tables in $NGDIR now: $(ls $NGDIR/ngram-*.tab 2>/dev/null | wc -l) | free RAM $(free -g | awk \'NR==2{print $7}\') GiB"')
# ---------------------------------------------------------------- the section: zzk25s's identity / readings / summary / start -> zzzfuse's
rep("# ---------------------------------------------------------------- zzk25s's own functions (zsa14's: ref14's tables, int14 pin read back and staggered\n"
    "# check; zzk25s's: lever K25U's switch read back, identity + the stop, the gens / 1u readings, the summary, the start)\n",
    "# ---------------------------------------------------------------- zzzfuse's own functions (zsa14's: ref14's tables, the pin read back and staggered\n"
    "# check; zzzfuse's: the fusion switches read back, the P0, identity + the stop, the gens / singles / 1u readings, the summary, the start)\n")
cut('X3DN=0; X3DLOG=""', "ngclean() {", P("funcs.sh"))
cut("dstart() {", "trap 'down;", P("dstart.sh"))
# ---------------------------------------------------------------- main
line('xlog "$ITEM: start IMAGE=$IMAGE',
     'xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f \'{{.Id}}\' $IMAGE 2>/dev/null | cut -c8-19))$([ $SW = 1 ] || echo " | MOCK-RUN ZFU_SW=0 (NOT A READING): NO int9b .. int15 '
     'switch posted, the fusion switches not posted, cross-run references INFO") | LEVERS FUSE-R / FUSE-M / RCO ON int15 (the dependent-launch fusions + the routing copy overlap), ONE FRESH START on ngshare (no ATTACH), int15 pinned right '
     'after it (fn4) and read back on every rank: $DESC | MAXSEQS ${ZFU_MAXSEQS:-the base value} | tables: A4 $A4MD5 ($A4N lines), A11 $A11MD5 ($A11NE entries), NOT re-benched | exl3r4d.so '
     'md5 wanted ${ZFU_PLUGMD5:-7b2f66d6} (checked next) | 0 P0 one card (bytes + per-call times, the gate) | 1 identity warm-up, a1 [$IDARMS] vs ref12 a1, gs$GSC-$GSA, staggered ld-$LDA (ANY difference: STOP '
     'before the speed blocks) | 2 c8 work-based gens $C8GN [$GORD] | 3 singles [$SINGLES] | 4 c16 gens $C16GN [$K16A] | 5 1u essay [$U1A] 1 discarded + $U1N each | the fusion / RCO switches + exl3_x3d asserted '
     'before every block | budget $BUDGET min | no evict"')
rep("\nk25summary\nunset $SWITCHES\n", "\nfusesummary\nunset $SWITCHES\n")
main = s.split("# ---------------------------------------------------------------- main")[1]
for w in ("k25summary", "k25reads", "k25id", "x3dchk", "x3dsay"):
    assert w not in main, "main still names %s" % w
for w in ("k25reads", "k25id ", "k25summary", "x3dchk", "x3dsay", "$XZ"):
    assert w not in s, "the body still names %s" % w
assert "k25sum" not in s.replace("zzk25s/k25sum.py", ""), "the body still names k25sum outside its provenance"
open(dst, "w").write(s)
os.chmod(dst, 0o755)
print("%s: %d edits, %d lines (from %s md5 %s)" % (dst, n_edits, s.count("\n"), src, md5))
