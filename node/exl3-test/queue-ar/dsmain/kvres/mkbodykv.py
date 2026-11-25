#!/usr/bin/env python3
"""mkbodykv.py SRC PARTS DST [SRCMD5] -- kvres/dsm-kvres.sh from zzzc16/dsm-zzzc16.sh (SRC, md5 SRCMD5 00cfc8b8 as staged + run 13.11.):
every edit an exact, asserted replacement (tag xzc16 -> xkvr, knobs ZC16_ -> KVR_; zzzc16's ONE start (dstart) -> kvprep once + kvstart ARM HP a
fresh start per arm with --host-pool-mib HP (HP29), the placement read-back (kvplace), the readings (kvreads: identity, c8 / c16 gens, 1u);
zc16sum.py -> kvres/kvsum.py; steps.py / ncmatch.py / the A4 table / the A11 entries read from ../zzzc16 (md5-checked as there); zzzc16's
functions kept). PARTS = kvres/mkparts (header.sh: the module header; knobs.sh: the arms / readings knobs; funcs.sh: ncmset / stepsx (zzzfuse's) +
kvprep / kvplace / kvreads / kvstart / kvsummary in place of zzzc16's dstart)."""
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


# ---------------------------------------------------------------- 1 before the tag change: the strings naming zzzc16's own files / provenance
rep("dsm-zzzc16.sh $(md5sum < $0 | cut -c1-8) (a copy of zzzfuse/dsm-zzzfuse.sh 2a0f352d via zzzc16/mkbody16.py)",
    "dsm-kvres.sh $(md5sum < $0 | cut -c1-8) (a copy of zzzc16/dsm-zzzc16.sh %s via kvres/mkbodykv.py)" % md5)
rep('HMD5="zzzc16/zc16sum.py $(md5sum < $YDS | cut -c1-8) (new: the int16 confirmation summary, zzzfuse/fusesum.py\'s step-log readers), zzzc16/steps.py '
    '$(md5sum < $STP | cut -c1-8) (= zzzfuse\'s, the coordinator\'s), ',
    'HMD5="kvres/kvsum.py $(md5sum < $YDS | cut -c1-8) (new: the placement read-back + the KVRES summary; zc16sum.py\'s step-log readers), kvres/mkbodykv.py '
    '$(md5sum < $HD/mkbodykv.py | cut -c1-8), references: zzzc16 start log $(md5sum < $REFSTART | cut -c1-8) / a1-F $(md5sum < $ZCA1 | cut -c1-8), zzzfuse raw-snaps '
    '$(md5sum < $REFZF/raw-snaps.tar.gz | cut -c1-8) / mb.jsonl $(md5sum < $REFZF/mb.jsonl | cut -c1-8), zzzc16/steps.py $(md5sum < $STP | cut -c1-8) (= zzzfuse\'s, the coordinator\'s), ')
rep("zc16sum.py $XPT", "kvsum.py $XPT")
# ---------------------------------------------------------------- 2 the header (everything above `set -u`)
cut("#!/bin/bash\n", "set -u\n", P("header.sh").replace("SRCMD5 below", "md5 %s" % md5).replace("set -u\n", ""))
# ---------------------------------------------------------------- 3 the out dir, the item, the helpers' paths; then the tag xzc16 -> xkvr, ZC16_ -> KVR_
rep("O=${ZC16_OUT:-$R/xdsm/zzzc16}; export STATSDIR=$O/stats", "O=${ZC16_OUT:-$R/xdsm/kvres}; export STATSDIR=$O/stats")
rep("ITEM=${ZC16_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzzc-int16}", "ITEM=${ZC16_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzzzzzzzz-zsoak16zz-kvres}")
rep("YDS=$HD/zc16sum.py; SAS=$HD/zc16sum.py; STP=$HD/steps.py; NCM=$HD/ncmatch.py;",
    "YDS=$HD/kvsum.py; SAS=$HD/kvsum.py; STP=$HD/../zzzc16/steps.py; NCM=$HD/../zzzc16/ncmatch.py;")
rep("A4TAB=$HD/m16-forms.txt;", "A4TAB=$HD/../zzzc16/m16-forms.txt;")
rep("A11TAB=$HD/a11-entries.txt;", "A11TAB=$HD/../zzzc16/a11-entries.txt;")
s = re.sub(r"(?<![A-Za-z])xzc16(?![A-Za-z0-9])", "xkvr", s)
s = s.replace("ZC16_", "KVR_")
assert "xzc16" not in s and "ZC16_" not in s
n_edits += 1
# ---------------------------------------------------------------- 4 the kvres knobs, after zzzc16's (its reading knobs kept: mbx / blockx read them)
rep("IDARMS=F; GSA=F; LDA=F   # identity runs under F (THE int16 PIN) only\n",
    "IDARMS=F; GSA=F; LDA=F   # identity runs under F (THE int16 PIN) only\n" + P("knobs.sh"))
# ---------------------------------------------------------------- 5 zzzc16's dstart -> the kvres functions (ncmset / stepsx / kvprep / kvplace / kvreads / kvstart / kvsummary)
cut("dstart() {   # the tables (zzzfuse's = ref14's, not re-benched), then the ONE fresh start (ngshare)",
    "trap 'down; [ $NGB = 1 ] && [ $NGCLEAN = 0 ] && ngclean' EXIT", P("funcs.sh"))
# ---------------------------------------------------------------- 6 main: the start line, the checks, the starts loop, the summary
line('xlog "$ITEM: start IMAGE=$IMAGE ',
     'xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f \'{{.Id}}\' $IMAGE 2>/dev/null | cut -c8-19))$([ $SW = 1 ] || echo " | MOCK-RUN KVR_SW=0 (NOT A READING): '
     'NO int9b .. int16 switch posted, the fusion switches not posted, the mock serves a real 19456 start log, cross-run references INFO") | THE KV-RESERVATION vs '
     'EXPERT-RESIDENCY TRADE-OFF on int16 (measurement only): one FRESH START per arm [$KVARMS] (ARM:--host-pool-mib; K0 19456 = zzzc16 / zzzfuse, not re-run) = zzzc16\'s '
     'int16 start (ngshare, no ATTACH, RADIANCE_DS41_RFUSE=$RFDECL, MAXSEQS ${KVR_MAXSEQS:-the base value}) with only --host-pool-mib moved; THE int16 PIN posted right after '
     'each start and read back (fn4 every switch, every rank, the declare lines, rco / fuse engagement) | per start: 0 placement read-back (kvsum.py place: the knob in force, '
     'owned / KV- / arena-lendable slots, KV tokens + sessions; a mismatch -> no reading) | 1 identity: warm-up, a1-F vs ref12 a1 + zzzc16 a1-F (ANY difference: that start\'s speed '
     'reads STOP) | 2 c8 gens $C8GN x $C8B | 3 c16 gens $C16GN x $C16B | 4 1u essay 1 discarded + $U1N | tables: A4 $A4MD5 ($A4N lines), A11 $A11MD5 ($A11NE entries), NOT '
     're-benched | exl3r4d.so md5 wanted ${KVR_PLUGMD5:-7b2f66d6} (checked next) | budget $BUDGET min, a start ~$STARTEST min | no evict"')
rep('[ -s "$ZFA1" ] || HMISS="$HMISS file:$ZFA1(zzzfuse\'s a1-B: the identity reference)"\n',
    '[ -s "$ZFA1" ] || HMISS="$HMISS file:$ZFA1(zzzfuse\'s a1-B: the identity reference)"\n'
    '[ -s "$ZCA1" ] || HMISS="$HMISS file:$ZCA1(zzzc16\'s a1-F: the K0 identity reference)"\n'
    '[ -s "$REFSTART" ] || HMISS="$HMISS file:$REFSTART(zzzc16\'s start log: the K0 placement)"\n'
    '[ -s "$REFZF/raw-snaps.tar.gz" ] && [ -s "$REFZF/mb.jsonl" ] || HMISS="$HMISS file:$REFZF/raw-snaps.tar.gz+mb.jsonl(zzzfuse\'s F blocks: the K0 window residency)"\n'
    '[ -s "$REFZC/blocks.jsonl" ] || HMISS="$HMISS file:$REFZC/blocks.jsonl(zzzc16\'s F 1u reads: the K0 1u)"\n'
    'KVPT=$(timeout 120 python3 -I -B $SAS place $REFSTART --arm K0 --hp $REFHP --ref $REFSTART 2>&1 | grep -m1 "^KVPLACE"); case "$KVPT" in *" -- OK"*) ;; *) HMISS="$HMISS place:$SAS(K0 on $REFSTART: ${KVPT:0:200})" ;; esac\n')
rep("FNS=\"stagx tabs14 rankrb", "FNS=\"ncmset stepsx kvprep kvplace kvreads kvstart kvsummary stagx tabs14 rankrb")
rep(" gsx gpurun dstart idok gt48 coldx\"", " gsx gpurun idok gt48 coldx\"")
line('  echo "+ readings: 1 identity under F (THE int16 PIN): warm-up vs stock',
     '  echo "+ arms: $(for e in $KVARMS; do echo -n "${e%%:*} --host-pool-mib ${e#*:}; "; done)K0 = 19456 (zzzc16\'s start, not re-run; KV 6.95 GiB a card) | K0 placement: ${KVPT:0:600}"\n'
     '  echo "+ readings a start: 0 placement read-back (kvsum.py place; VIRHE -> no reading) | 1 identity: warm-up vs stock, a1-F vs ref12 a1 ($([ -s $REF12A1 ] && echo present || echo MISSING)) '
     '+ vs zzzc16 a1-F ($([ -s $ZCA1 ] && echo present || echo MISSING)) -- ANY difference STOPS that start\'s speed reads | 2 c8 gens $C8GN x $C8B | 3 c16 gens $C16GN x $C16B | '
     '4 1u essay 1 discarded + $U1N ($([ $U1ON = 1 ] && echo on || echo off); answers vs $ESSMD5) | ds41_rfuse / ds41_mix_fin / route_copy_ov / exl3_x3d asserted before every block; '
     'refusal probes exl3_x3d $XREF, ds41_rfuse 3, ds41_mix_fin 2, route_copy_ov 2 | budget $BUDGET min, a start ~$STARTEST min | four cards $(cat /sys/class/drm/card*/device/unique_id 2>/dev/null | wc -l) unique_id '
     '| n-gram tables in $NGDIR now: $(ls $NGDIR/ngram-*.tab 2>/dev/null | wc -l) | free RAM $(free -g | awk \'NR==2{print $7}\') GiB | /tmp inodes $(df --output=ipcent /tmp 2>/dev/null | tail -1 | tr -d \' \'), / inodes $(df --output=ipcent / 2>/dev/null | tail -1 | tr -d \' \')"')
rep("  dstart || true\n",
    "  if kvprep; then\n"
    "    for e in $KVARMS; do\n"
    "      stopfile && break\n"
    "      budok \"start ${e%%:*} (--host-pool-mib ${e#*:})\" $STARTEST || continue\n"
    "      kvstart ${e%%:*} ${e#*:} || true\n"
    "    done\n"
    "  fi\n"
    "  xlog \"starts: read [$(echo $KVDONE)] | stopped / no reading [$(echo $KVSTOPS)] | $(elapsed) min\"\n")
rep("\nc16summary\n", "\nkvsummary\n")
rep('xlog "$ITEM: done ($(elapsed) min) -- $O | $([ $STOPPED = 1 ] && echo "STOPPED at the identity gate (no speed block ran): NEEDS USER | ")',
    'xlog "$ITEM: done ($(elapsed) min) -- $O | $([ -n "$(echo $KVSTOPS)" ] && echo "arms with NO reading or a STOP: $(echo $KVSTOPS) (see the VIRHE lines) | ")')
open(dst, "w").write(s)
os.chmod(dst, 0o755)
print("mkbodykv: %s (md5 %s) -> %s: %d edits, md5 %s" % (src, md5, dst, n_edits, hashlib.md5(s.encode()).hexdigest()[:8]))
