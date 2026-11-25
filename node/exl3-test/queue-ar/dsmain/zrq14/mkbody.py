#!/usr/bin/env python3
"""mkbody.py SRC PARTS DST [SRCMD5] -- zrq14/dsm-zrq14.sh from ref14/dsm-ref14.sh (SRC, md5 SRCMD5 = 74981d12 as staged and run 12.11.):
every edit an exact, asserted replacement (tag xref14 -> xzq14, knobs R14_ -> ZQ14_; c1 / c4 windows back to ref12's 20 s; ref14's
identity / readings / summary functions out, zrq14's in; ref14's one start -> two starts, the second conditional). PARTS = zrq14/mkparts
(header.sh: the module header; funcs.sh: zrq14's identity, readings, gate and summary functions, in place of ref14's coldid .. r14summary;
dstart.sh: zrq14's two starts, in place of ref14's dstart)."""
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
    "dsm-zrq14.sh $(md5sum < $0 | cut -c1-8) (a copy of ref14/dsm-ref14.sh %s via zrq14/mkbody.py)" % md5)
rep('HMD5="ref14/r14sum.py $(md5sum < $YDS | cut -c1-8), ref14/m16-forms.txt $(md5sum < $A4TAB | cut -c1-8) (yzint14c\'s = yd2m\'s), '
    'ref14/a11-entries.txt $(md5sum < $A11TAB | cut -c1-8) (yzint14c\'s P0), ref14/ncmatch.py $(md5sum < $NCM | cut -c1-8), ',
    'HMD5="zrq14/zq14sum.py $(md5sum < $YDS | cut -c1-8), prof12/p12win.py $(md5sum < $R12S | cut -c1-8) (rows, read in place), zrq14/m16-forms.txt '
    '$(md5sum < $A4TAB | cut -c1-8) (= ref14\'s = yzint14c\'s), zrq14/a11-entries.txt $(md5sum < $A11TAB | cut -c1-8) (= ref14\'s = yzint14c\'s P0), '
    'ref14/ncmatch.py $(md5sum < $NCM | cut -c1-8) (selftest only), ')
# ---------------------------------------------------------------- the header (everything above `set -u`)
cut("#!/bin/bash\n", "set -u\n", P("header.sh").replace("REFMD5 below", "md5 %s" % md5))
# ---------------------------------------------------------------- the tag: xref14 -> xzq14, R14_ -> ZQ14_, the out dir
for a, b in (("dsm-ref14.sh", "dsm-zrq14.sh"), ("$R/xdsm/ref14", "$R/xdsm/zrq14")):
    assert s.count(a) >= 1, a
    s = s.replace(a, b)
s = re.sub(r"(?<![A-Za-z])xref14(?![A-Za-z0-9])", "xzq14", s)
s = re.sub(r"(?<![A-Za-z0-9_])R14_", "ZQ14_", s)
assert "xref14" not in s and not re.search(r"(?<![A-Za-z0-9_])R14_", s)
n_edits += 1
# ---------------------------------------------------------------- knobs
rep("ITEM=${ZQ14_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zref14}", "ITEM=${ZQ14_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zrq14}")
rep("YDS=$HD/r14sum.py; NCM=$HD/ncmatch.py;", "YDS=$HD/zq14sum.py; NCM=$HD/../ref14/ncmatch.py;")
rep("BUDGET=${ZQ14_BUDGET_MIN:-35}", "BUDGET=${ZQ14_BUDGET_MIN:-31}")
rep("${ZQ14_C4WIN:-35}", "${ZQ14_C4WIN:-20}", 2)   # zrq14: ref12's c4 / c1 windows (X12_C4WIN / X12_C1WIN 20; ref14's mkbody had set 35)
rep("${ZQ14_C1WIN:-35}", "${ZQ14_C1WIN:-20}", 2)
rep("ESSMD5=${ZQ14_ESSAY_MD5:-47a39dac581a}; CODEMD5=${ZQ14_CODE_MD5:-c5e7c82e3e11}   # yzint14c's resprobe answers (text_md5; every arm, every read)\n",
    "ESSMD5=${ZQ14_ESSAY_MD5:-47a39dac581a}; CODEMD5=${ZQ14_CODE_MD5:-c5e7c82e3e11}   # yzint14c's resprobe answers (text_md5; every arm, every read)\n"
    "# zrq14: the two starts, the arm orders, the window bridge, the gate (module header)\n"
    "MS1=${ZQ14_MAXSEQS:-16}; MS2=${ZQ14_MAXSEQS2:-8}; C1ORD=${ZQ14_C1ARMS-BZZB}; C4ORD=${ZQ14_C4ARMS-BZZB}; BRIDGE=${ZQ14_BRIDGE:-1}; BRWIN=${ZQ14_BRIDGEWIN:-35}\n"
    "BUD1=${ZQ14_BUDGET1:-20}; G1=${ZQ14_GATE_C1:-121.0}; G4=${ZQ14_GATE_C4:-170.0}; S2=${ZQ14_START2:-auto}; S2EST=${ZQ14_S2EST:-9}; C1N2=${ZQ14_C1N2:-2}; C4N2=${ZQ14_C4N2:-2}\n"
    "ZQSFX=\"\"; STOPPED2=0   # ZQSFX: the a1 name suffix of start 2 (a1-B8)\n"
    "for v in $MS1 $MS2 $BRIDGE $BRWIN $BUD1 $S2EST $C1N2 $C4N2; do case $v in ''|*[!0-9]*) echo \"a zrq14 knob is not a number ('$v')\"; exit 1 ;; esac; done\n"
    "for v in $G1 $G4; do case $v in ''|*[!0-9.]*) echo \"ZQ14_GATE_C1 / ZQ14_GATE_C4 are tok/s values (got '$v')\"; exit 1 ;; esac; done\n"
    "case $S2 in auto|0|1) ;; *) echo \"ZQ14_START2 auto / 0 / 1 (got '$S2')\"; exit 1 ;; esac; case $BRIDGE in 0|1) ;; *) echo \"ZQ14_BRIDGE 0 or 1\"; exit 1 ;; esac\n"
    "for o in \"$C1ORD\" \"$C4ORD\"; do case $o in ''|*[!BZ]*) echo \"ZQ14_C1ARMS / ZQ14_C4ARMS take B and Z (got '$o')\"; exit 1 ;; esac; done\n"
    "[ $MS1 -ge 1 ] && [ $MS2 -ge 1 ] || { echo \"ZQ14_MAXSEQS / ZQ14_MAXSEQS2 >= 1\"; exit 1; }\n")
# ---------------------------------------------------------------- the arms: B = every int14 lever at 0 (ref14's B), Z = the int14 pin (= O): unchanged
rep('B) echo "B (int12 / ref12: ds41_bi 2, gg_pf 1, ar_xpush 0, gg_gemv_m16 0, place_swap_lent 0, heat_min_gain_x100 150)" ;;',
    'B) echo "B (every int14 lever at 0 = int12 bytes: ds41_bi 2, gg_pf 1, ar_xpush 0, gg_gemv_m16 0, place_swap_lent 0, heat_min_gain_x100 150)" ;;')
line('DESC="THE int14 REFERENCE READ',
     'DESC="ref14 vs ref12 (zrq14: image int14c, THE int14 PIN = ref14\'s O; arms in process B = every int14 lever at 0 = int12 bytes, Z = the int14 pin; '
     'start 1 MAXSEQS $MS1, start 2 MAXSEQS $MS2 B only if the gate says so; c1 / c4 in ref12\'s windows ${ZQ14_C1WIN:-20} / ${ZQ14_C4WIN:-20} s + a $BRWIN s bridge) NST 5 stack5 start '
     '(stack4\'s + RADIANCE_LOG_STEPS=1, E1 declared off, HB ranks $HBRANKS) + RADIANCE_DS41_MHC_FUSE=0 (M1 declared) + RADIANCE_DS41_GQ=3 + RADIANCE_DS41_E2=1 + '
     'RADIANCE_DS41_KV_REPLICATED=1 (KVR) + A4 table + A11 ranged entries (ref14\'s copies) + RADIANCE_FN4_AR_XPUSH_MIN=$XPMIN $([ $STAMPN -gt 0 ] && echo "+ '
     'RADIANCE_AQL_STAMP=$STAMPN" || echo "(no stamping)") | SP / ARC / CBH off, ngshare, no ATTACH | the pin = stack5\'s S5T with draft_sample 0 + SP 0 / 0 + $MJ + '
     'the fixed pin (ds41_psk_k $PSKK; draft_lend 1; dspark_conf 1, dspark_conf_rows 1)"')
# ---------------------------------------------------------------- helpers: zq14sum.py (selftest), the tables (copies in zrq14/), YZA1 no longer read
rep("$YDS $NCM $NCT $DQ $SPB0 $A4TAB $A11TAB $YZA1 $REF12A1 $NGS/ngshare.sh", "$YDS $NCM $NCT $DQ $SPB0 $A4TAB $A11TAB $REF12A1 $NGS/ngshare.sh")
rep("| r14sum.py $XPT | ncmatch.py $NMT2 |", "| zq14sum.py $XPT | ncmatch.py $NMT2 (selftest only) |")
line('FNS="stagx coldid tabs14',
     'FNS="tabs14 rankrb a11line a4line xpline zqid zqbridge zqreads16 zqgate zqreads8 zqrows zq14summary env14 zstart zstop dstart ngclean swlwant zarm log run '
     'mkd restart check down stopfile best_vars desc serve_args xlog elapsed armj NAMEOF pinj stackenv keeplog capture thermline dfree diskok cmpfile idrec identw '
     'modes vramstart fn4get fn4set nudge fn4code pcreset arm msnap mdelta ssnap gate1 alone cmpx budok mwin ctrinfo i11now i11d i11g monons utcnow drafted readback '
     'swnote capd mbx cbhd cbhg blockx idok gt48"')
line('  echo "+ readings: 1 identity warm-up vs stock, a1-O',
     '  echo "+ readings: START 1 MAXSEQS $MS1: 1 identity warm-up vs stock, a1-B + a1-Z vs ref12 a1 ($([ -s $REF12A1 ] && echo present || echo MISSING)) -- ANY '
     'difference STOPS | 2 c1 [$C1ORD] (warm $C1WARM + $C1WIN s) + bridge Z $BRWIN s ($([ $BRIDGE = 1 ] && echo on || echo off)) | 3 c4 warm-up (B) + [$C4ORD] (warm '
     '$C4WARM + $C4WIN s) + bridge | 4 1u essay B (1 discarded + $U1N), Z (1 + $U1N), B again x$U1N || GATE c1 B < $G1 OR c4 B < $G4 (ZQ14_START2 $S2) -> START 2 '
     'MAXSEQS $MS2, B only: a1-B8 vs ref12, c1 x$C1N2, c4 warm-up + $C4N2, 1u 1 + $U1N | budget start 1 $BUD1 min, all $BUDGET min | four cards $(cat '
     '/sys/class/drm/card*/device/unique_id 2>/dev/null | wc -l) unique_id | n-gram tables in $NGDIR now: $(ls $NGDIR/ngram-*.tab 2>/dev/null | wc -l) | free RAM '
     '$(free -g | awk \'NR==2{print $7}\') GiB"')
# ---------------------------------------------------------------- the section: ref14's identity / readings / summary / start -> zrq14's
rep("# ---------------------------------------------------------------- ref14's own functions (the tables, the int14 pin read back, identity + the stop, readings, summary)\n",
    "# ---------------------------------------------------------------- zrq14's own functions (ref14's tables and int14 pin read back; the identity +\n"
    "# the stop, start 1's readings, the gate, start 2's readings, the summary, the two starts)\n")
cut("coldid() {", "ngclean() {", P("funcs.sh"))
cut("dstart() {", "trap 'down;", P("dstart.sh"))
# ---------------------------------------------------------------- main
line('xlog "$ITEM: start IMAGE=$IMAGE',
     'xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f \'{{.Id}}\' $IMAGE 2>/dev/null | cut -c8-19))$([ $SW = 1 ] || echo " | MOCK-RUN ZQ14_SW=0 (NOT A READING): NO int9b .. int14c '
     'switch posted, cross-run references INFO") | ref14 vs ref12: int14 or MAXSEQS 16 or the window? START 1 (fresh, ngshare, no ATTACH) MAXSEQS $MS1, int14 pinned and read back '
     'on every rank, arms B (int12 bytes) / Z (int14) in process: identity a1-B / a1-Z vs ref12 (ANY difference: STOP), c1 [$C1ORD] + bridge, c4 warm-up + [$C4ORD] + '
     'bridge (ref12\'s $C1WIN / $C4WIN s windows, bridge $BRWIN s), 1u essay B / Z / B | GATE c1 B < $G1 OR c4 B < $G4 -> START 2 MAXSEQS $MS2 B only | $DESC | '
     'budget $BUD1 / $BUDGET min | no evict"')
rep("\nr14summary\nunset $SWITCHES\n", "\nzq14summary\nunset $SWITCHES\n")
rep('$([ $STOPPED = 1 ] && echo "STOPPED at the identity gate (no speed block ran): NEEDS USER | ")',
    '$([ $STOPPED = 1 ] && echo "STOPPED at the identity gate (no speed block ran): NEEDS USER | ")$([ $STOPPED2 = 1 ] && echo "START 2 STOPPED at its identity gate: NEEDS USER | ")')
main = s.split("# ---------------------------------------------------------------- main")[1]
for w in ("r14summary", "r14reads", "r14id", "REF14", "coldid"):
    assert w not in main, "main still names %s" % w
for w in ("r14reads", "r14id ", "r14summary", "coldid", "ncmset", "splitblk"):
    assert w not in s, "the body still names %s" % w
open(dst, "w").write(s)
os.chmod(dst, 0o755)
print("%s: %d edits, %d lines (from %s md5 %s)" % (dst, n_edits, s.count("\n"), src, md5))
