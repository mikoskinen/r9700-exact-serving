#!/usr/bin/env python3
"""mkbody.py: zepb/dsm-zepb.sh from a COPY of prof12/dsm-prof1.sh (+ prof2's profiled summary); every edit asserted."""
import re, sys

src = open(sys.argv[1]).read()          # prof12/dsm-prof1.sh
p2 = open(sys.argv[2]).read()           # prof12/dsm-prof2.sh
out = sys.argv[3]
s = src


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, "expected %d of %r, found %d" % (count, old[:90], n)
    s = s.replace(old, new)


def func(text, name):
    """the text of bash function `name` (from its 'name() {' line to the first line that is exactly '}')"""
    m = re.search(r"(?m)^%s\(\) \{.*?^\}\n" % re.escape(name), text, re.S)
    assert m, name
    return m.group(0)


# ---- 1. the header
h0 = s.index("# dsm-prof1.sh")
h1 = s.index("set -u\n")
HEADER = r'''# dsm-zepb.sh (12.11.2026, queue-ar/dsmain/zepb, the EPB builder for ds-main-port) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zepb.sh. Log tag "RESULT xepb ", out runs/xdsm/epb (XEP_OUT; the profiled start in
# runs/xdsm/epb/prof). A COPY of prof12/dsm-prof1.sh (91d75e4d's body d3efd03a; tag xp1 -> xepb, knobs XP1_ -> XEP_; the original
# not changed) + prof12/dsm-prof2.sh's profiled start / summary: the SAME start, pins, guards and image checks as ref12 / prof1.
# LEVER EPB (user 12.11. via the coordinator: "tasapainotuksen saa mitata" = MEASURE ONLY; nothing goes into a base): the
#   rank-independent SLOT-ORDER MoE sum + least-loaded hot-expert replication (ds41-slot 4cf3501 D3s, ported onto int12 bf8f557 as
#   branch ds41-epb; image radiance-local:ds41-epb-*). RADIANCE_DS41_SLOTSUM=16 declares, beside today's MoE path, the slot path for
#   every MoE pass of <= 16 rows (the EXL3 decode band; with ds41_bi 2 every decode-band pass, also a mixed step's decode pieces:
#   the EPB pass rule reads the pass's rows only, so the arm is batch-invariant). Tables: 0 contiguous, 1 XEP_PLACE1 (hot8.place:
#   epplace.py replicate --hot 8 from cc-x117.json, the 8 hottest decode experts a layer on every rank), 2 XEP_PLACE2 (hot4.place,
#   a subset: its extras are hot8's). RADIANCE_DS41_SLOTSUM_ON=0: the start runs today's path; arms by fn4 (in process):
#     O  = ds41_slotsum 0 (today's path; the extras declared, idle) -- NOT a plain int12 start: priced vs prof1 (c8 184.0)
#     E  = ds41_slotsum 1, ds41_place 1 (EPB: slot-order sum + hot-8 replication, least-loaded copy a step)
#     S  = ds41_slotsum 1, ds41_place 0 (the sum alone, contiguous; INFO)   E4 = ds41_slotsum 1, ds41_place 2 (hot-4; INFO)
# NOT EXACT vs today ONCE (the slot-order sum); within the E arm it must be deterministic and batch-invariant.
# ORDER: P0 (4 cards, no server: r4d_selftest moe_gather_all_reduce_slot_4rank + moe_scatter_placed_i32; a failure stops the item)
#   -> START A (unprofiled) -> modes / readback / switch note (ds41_slotsum 2 and ds41_place 3 are REFUSED by design: HTTP 400 is
#   the expected answer, as in ds41-slot's p-slotsum) / the EPB note (40 of 40 layers, tables, extras a rank) / VRAM at start vs
#   prof1 -> P0b DETERMINISM + BI (E: a1, a2 alone; gs16 all in flight; a1 vs a2 and gs16 vs a1 strict; O's a1 vs ref12's a1 strict
#   = the declaration keeps today's bytes; a difference stops the item) -> READINGS (1) c8 ABBA O E E O + S + E4 (harness/ncmb3.py
#   fixed prompts, 35 s, P12SPLIT each) (2) c16 CAP48 method E, O + the fixed-harness newcomer block c16 E (ncmb3 1x16, 35 s,
#   P12SPLIT) (3) 1u essay E: one discarded read + 3 back to back, then O the same (4) STAGGERED arrivals under E (stack4/loadid.py:
#   4 decode streams, the 20 greedy probes one at a time, each with a fresh XEP_LDPF-token prefill joining 0.5 s in, so its decode
#   rows ride MIXED steps) vs E's a1 alone, strict (5) vs-int12 texts INFO: E a1 vs ref12 a1 (differing answers, first-divergence
#   char / token each) -> capture + summary -> START B (PROFILED --profile-ops, every XEP_PEVERY 8; disk >= XEP_PDISK_MB 12000):
#   ONE c8 window under E -> P12PROF / P12TOP: moe_gather_all_reduce_slot us a call, per-rank MoE ms vs prof2 (154.6 us / 6.11 ms).
# References (no re-runs; int12 prof1 17:11-17:18 / prof2 17:25-17:34): c8 184.0 tok/s, 36.18 ms a step at 8.47 rows, newcomer
#   538.2 ms x 13; c16 CAP48 method 300.7 (55.5 ms a step); 1u essay back to back 91.01; prof2 c8 gather_ar 6.11 ms = 154.6 us a call.
# BUDGET XEP_BUDGET_MIN 55 (a section past it skipped, logged). Logs gzipped; no raw dumps kept. NO evict, no cc_sim, no power / host /
#   fan settings; production compose files read only. MOCK: XEP_SW=0 (queue-ar/tools/mockrad; no EPB switch posted).
# DRYRUN=1 bash dsm-zepb.sh : prints every command; checks the image literals (int9 .. int12 + EPB), every helper, the placement files.
'''
s = s[:h0] + HEADER.lstrip("\n").split("\n", 0)[0] + s[h1:]
assert s.startswith("#!/bin/bash\n# dsm-zepb.sh")

# ---- 2. the tag / knob renames (the prof1 reference paths are added below, after this)
s = s.replace("xp1", "xepb").replace("XP1_", "XEP_")
rep("O=${XEP_OUT:-$R/xdsm/prof12/p1}; export STATSDIR=$O/stats", "O=${XEP_OUT:-$R/xdsm/epb}; export STATSDIR=$O/stats")
rep("ITEM=${XEP_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzz-prof1}", "ITEM=${XEP_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zepb}")
rep("R12S=$HD/p12win.py;", "R12S=$HD/../prof12/p12win.py;")
rep("BUDGET=${XEP_BUDGET_MIN:-35}", "BUDGET=${XEP_BUDGET_MIN:-55}")

# ---- 3. the EPB knobs (before LEVENV)
rep("MT=192; SEED=1234\n", r'''MT=192; SEED=1234
# lever EPB (this item): the declaration, the placement files (staged beside the body, copied under /stats/place at the start)
SLROWS=${XEP_SLOTSUM:-16}; PLDIR=$HD/place; PL1=${XEP_PLACE1:-hot8.place}; PL2=${XEP_PLACE2:-hot4.place}
C8ORDER=${XEP_C8ORDER-O E E O S E4}; U1ARMS=${XEP_U1ARMS-E O}; C16ARMS=${XEP_C16ARMS-E O}; C16NC=${XEP_C16NC:-1}; PPLAN=${XEP_PPLAN-E}
LDPF=${XEP_LDPF:-8192}; LDCONC=${XEP_LDCONC:-4}; DOPROF=${XEP_PROF:-1}; PDISK=${XEP_PDISK_MB:-12000}; PEVERY=${XEP_PEVERY:-8}
P0CASES=${XEP_P0CASES-moe_gather_all_reduce_slot_4rank moe_scatter_placed_i32}; P0TMO=${XEP_P0TMO:-600}
OPNC=$HD/../nc/ds_opprof_nc.py
P1DIR=$R/xdsm/prof12/p1; P1A1=$P1DIR/xp1-a1.jsonl; R12A1=$R/xdsm/ref12/xref12-a1.jsonl; P1STATS=$P1DIR/xp1-n5-stats0.json
P1C8=184.0; P1C16=300.7; P1U1=91.01; P2GARUS=154.6; P2GARMS=6.11
STOPALL=0
''')
rep('''  RADIANCE_DS41_MHC_FUSE=0 RADIANCE_DS41_GQ=3 RADIANCE_DS41_E2=1 RADIANCE_DS41_KV_REPLICATED=1"''',
    '''  RADIANCE_DS41_MHC_FUSE=0 RADIANCE_DS41_GQ=3 RADIANCE_DS41_E2=1 RADIANCE_DS41_KV_REPLICATED=1
  RADIANCE_DS41_SLOTSUM=$SLROWS RADIANCE_DS41_SLOTSUM_ON=0 RADIANCE_DS41_EP_PLACE=/stats/place/$PL1 RADIANCE_DS41_EP_PLACE2=/stats/place/$PL2
  RADIANCE_DS41_EP_PLACE_TAB=0"''')
rep('''LITS="$LITS|lever,M1,(mhc_fuse),declared"''', '''LITS="$LITS|lever,M1,(mhc_fuse),declared"
LITS="$LITS|the,slot-order,MoE,sum,(lever,D3s,/,EPB),in,the,decode,band|40,of,40,trunk,layers,and"   # lever EPB's start note''')
rep('''  RADIANCE_DS41_MIX_WV RADIANCE_DS41_GQ RADIANCE_DS41_E2 RADIANCE_AQL_STAMP RADIANCE_AQL_STAMP_OUT"''',
    '''  RADIANCE_DS41_MIX_WV RADIANCE_DS41_GQ RADIANCE_DS41_E2 RADIANCE_AQL_STAMP RADIANCE_AQL_STAMP_OUT
  RADIANCE_DS41_SLOTSUM RADIANCE_DS41_SLOTSUM_ON RADIANCE_DS41_EP_PLACE RADIANCE_DS41_EP_PLACE2 RADIANCE_DS41_EP_PLACE_TAB"''')

# ---- 4. the pin and the arms
rep('''\\"sched_cb_hold_k\\": $CBHK, \\"exl3_x3d\\": 1}"''', '''\\"sched_cb_hold_k\\": $CBHK, \\"exl3_x3d\\": 1, \\"ds41_slotsum\\": 0, \\"ds41_place\\": 0}"''')
rep('''    O) echo '{}' ;;\n''', '''    O) echo '{}' ;;
    E) [ $SW = 1 ] && echo '{"ds41_slotsum": 1, "ds41_place": 1}' || echo '{}' ;;    # lever EPB: slot-order sum + table 1
    E4) [ $SW = 1 ] && echo '{"ds41_slotsum": 1, "ds41_place": 2}' || echo '{}' ;;   # + table 2 (INFO)
    S) [ $SW = 1 ] && echo '{"ds41_slotsum": 1, "ds41_place": 0}' || echo '{}' ;;    # the sum alone, contiguous (INFO)
''')
rep('''NAMEOF() { case $1 in O) echo "O (THE REFERENCE PIN)" ;;''',
    '''NAMEOF() { case $1 in O) echo "O (today's MoE path, ds41_slotsum 0; the extras declared)" ;; E) echo "E (EPB: ds41_slotsum 1, ds41_place 1 = $PL1)" ;;
  E4) echo "E4 (ds41_slotsum 1, ds41_place 2 = $PL2)" ;; S) echo "S (ds41_slotsum 1, ds41_place 0: the sum alone)" ;;''')

# ---- 5. swnote: the EPB refusals (by design) and start values
rep('''  if [ $SW = 1 ]; then c9=$(fn4code '{"exl3_x3d": 2}'); fi''',
    '''  if [ $SW = 1 ]; then c9=$(fn4code '{"exl3_x3d": 2}'); fi
  local c10=400 c11=400   # lever EPB: past-the-end values, REFUSED BY DESIGN (HTTP 400 expected; ds41-slot's p-slotsum checked the same)
  if [ $SW = 1 ]; then c10=$(fn4code '{"ds41_slotsum": 2}'); c11=$(fn4code '{"ds41_place": 3}'); fi
  xlog "$1 EPB switch refusals (expected, by design): ds41_slotsum 2 -> HTTP $c10, ds41_place 3 -> HTTP $c11 (want 400 / 400: 0 / 1 and tables 0..2 only)"''')
rep('''    ok = ok and w.get("sched_cb_hold_ms") == 0 and w.get("sched_cb_hold_k") == 2 and isinstance(s0.get("cbh"), dict) and w.get("exl3_x3d") == 0''',
    '''    ok = ok and w.get("sched_cb_hold_ms") == 0 and w.get("sched_cb_hold_k") == 2 and isinstance(s0.get("cbh"), dict) and w.get("exl3_x3d") == 0
    ok = ok and w.get("ds41_slotsum") == 0 and w.get("ds41_place") == 0   # lever EPB: RADIANCE_DS41_SLOTSUM_ON=0 / _EP_PLACE_TAB=0''')
rep('''[ "$c8" = 400 ] && [ "$c9" = 400 ] && [ -z "$bad" ]''', '''[ "$c8" = 400 ] && [ "$c9" = 400 ] && [ "$c10" = 400 ] && [ "$c11" = 400 ] && [ -z "$bad" ]''')

# ---- 6. blockx: any *W lever is a discarded warm-up read
rep('''$([ $lv = U1W ] && echo ", a DISCARDED warm-up read")''', '''$(case $lv in *W) echo ", a DISCARDED warm-up read" ;; esac)''')

# ---- 7. c16x: the arm as an argument (name c16-cap-ARM)
c16 = func(s, "c16x")
c16n = c16.replace("  local w=c16-cap t00=$(date +%s) j\n  arm O || return 1\n", "  local a=${1:-O} w=c16-cap-${1:-O} t00=$(date +%s) j\n  arm $a || return 1\n")
c16n = c16n.replace("the reference pin with sched_row_cap $CAP", "arm $(NAMEOF $a), sched_row_cap $CAP")
c16n = c16n.replace("print(\"agg %s tok/s (vs CAP48 %.1f: %s)", "print(\"agg %s tok/s (vs prof1 int12 c16 %.1f: %s)")
c16n = c16n.replace("' $C48REF 2>&1", "' $P1C16 2>&1")
assert c16n != c16 and c16n.count("arm $a") == 1
s = s.replace(c16, c16n)

# ---- 8. gsx: the arm and the name as arguments
gs = func(s, "gsx")
gsn = r'''gsx() {   # gsx ARM NAME: greedyset.py's 20 prompts at conc GSC (all in flight at MAXSEQS 16; CAP48), temperature 0 -> $O/xepb-NAME.jsonl
  local a=${1:-O} nm=${2:-gs$GSC} t0=$(date +%s) n
  arm $a || return 1; pcreset xepb-$nm
  run "(cd $D && timeout 900 python3 $GS $URL $O/xepb-$nm.jsonl --conc $GSC --max-tokens $MT > $O/xepb-$nm.txt 2>&1)"
  [ "$DRY" = 1 ] && { echo "+ RESULT xepb $nm: greedy set at conc $GSC under $(NAMEOF $a) (MAXSEQS ${XEP_MAXSEQS:-the base value}): 20 of 20 answered"; return 0; }
  n=$(python3 -c 'import json,sys; print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l).get("text") is not None))' $O/xepb-$nm.jsonl 2>/dev/null || echo 0)
  xlog "$nm (greedy set at conc $GSC, temperature 0, $MT tok, $(NAMEOF $a), MAXSEQS ${MAXSEQS:-?}): $n of 20 answered$([ "$n" = 20 ] || echo " -- VIRHE request errors"), $(tail -1 $O/xepb-$nm.txt | cut -c1-160), $(( $(date +%s) - t0 )) s"
}
'''
s = s.replace(gs, gsn)

# ---- 9. the image literals: + EPB
rep('''IMGLITS12="B|sched_cb_hold_ms, sched_cb_hold_k, exl3_x3d, epoch_bump)''',
    '''IMGLITS12="B|sched_cb_hold_ms, sched_cb_hold_k, exl3_x3d, ds41_slotsum, ds41_place, epoch_bump)
B|ds41_slotsum
A|RADIANCE_DS41_SLOTSUM
A|RADIANCE_DS41_EP_PLACE2
A|MoE passes of up to
K|moe_gather_all_reduce_slot
K|moe_scatter_placed
R|moe_gather_all_reduce_slot_4rank
R|moe_scatter_placed_i32
P|epb-PROVENANCE.txt''')

# ---- 10. the DRYRUN / helper checks
rep('''for f in "$LIB29" "$LIBPR" $Y $CTR $GS $LD $NCMB $DQ $MBP $TESTS $SPB $D/fastcheck.py $D0B $HD/../d0ab/s5ctr.py $MBT $EP $PRB $U1S $R12S $C48N $C4R $MBT2 $NCMB2 $C8MIX; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done''',
    '''for f in "$LIB29" "$LIBPR" $Y $CTR $GS $LD $NCMB $DQ $MBP $TESTS $SPB $D/fastcheck.py $D0B $HD/../d0ab/s5ctr.py $MBT $EP $PRB $U1S $R12S $C48N $C4R $MBT2 $NCMB2 $C8MIX $OPNC $PLDIR/$PL1 $PLDIR/$PL2; do [ -s "$f" ] || HMISS="$HMISS file:$f"; done
OPT=$(timeout 120 python3 -I -B $OPNC selftest 2>&1 | tail -1); case "$OPT" in *OK*) ;; *) HMISS="$HMISS selftest:$OPNC(${OPT:0:300})" ;; esac
grep -qF 'RADIANCE_PROFILE_EVERY: ${PROFEVERY' $Y 2>/dev/null || HMISS="$HMISS yaml:$Y(no RADIANCE_PROFILE_EVERY from PROFEVERY)"
PLBAD=$(python3 -I -c 'import sys
for f in sys.argv[1:]:
    L = [l.split("#")[0].split() for l in open(f)]; L = [x for x in L if x]
    ok = len([x for x in L if x[0] == "L"]) == 40 and all(x[0] in ("L", "C") and len(x) == 3 for x in L) and all(len(x[2]) == 384 for x in L if x[0] == "L")
    if not ok: print("place:%s" % f)' $PLDIR/$PL1 $PLDIR/$PL2 2>&1 | tr '\\n' ' ')
HMISS="$HMISS $PLBAD"''')
fns_old = re.search(r'FNS="log run mkd restart check down stopfile best_vars desc serve_args xlog elapsed armj NAMEOF pinj [^"]*"', s).group(0)
s = s.replace(fns_old, fns_old[:-1] + ' p0epb epbnote p0id epbreads epbstag dstartp summaryp epbprof"')
rep('''for a in O M0; do python3 -c 'import json,sys; d=json.loads(sys.argv[1]); sw = sys.argv[2] == "1"; a = sys.argv[3]''',
    '''for a in O M0 E E4 S; do python3 -c 'import json,sys; d=json.loads(sys.argv[1]); sw = sys.argv[2] == "1"; a = sys.argv[3]
if sw: assert (d["ds41_slotsum"], d["ds41_place"]) == {"O": (0, 0), "M0": (0, 0), "E": (1, 1), "E4": (1, 2), "S": (1, 0)}[a]''')
rep('''  for a in O M0; do echo "+ arm $a ($(NAMEOF $a)): $(armj $a)"; done''', '''  for a in O E S E4; do echo "+ arm $a ($(NAMEOF $a)): $(armj $a)"; done''')
rep('''HMD5="prof12/p12win.py $(md5sum < $R12S | cut -c1-8),''', '''HMD5="zepb/place/$PL1 $(md5sum < $PLDIR/$PL1 | cut -c1-8), zepb/place/$PL2 $(md5sum < $PLDIR/$PL2 | cut -c1-8), nc/ds_opprof_nc.py $(md5sum < $OPNC | cut -c1-8), prof12/p12win.py $(md5sum < $R12S | cut -c1-8),''')
m = re.search(r'(?m)^  echo "\+ readings: 1u essay .*$', s)
s = s[:m.start()] + '''  echo "+ readings (START A): P0 4-card selftest [$P0CASES] -> P0b E a1 / a2 / gs$GSC + O a1 (strict) -> c8 blocks [$C8ORDER] ${C8WIN} s (ncmb3 fixed, P12SPLIT each) -> c16 CAP48 method [$C16ARMS] (vs prof1 $P1C16) + c16 fixed-harness newcomer block E ($C16NC) -> 1u essay $EGEN [$U1ARMS] (W + $NREADS each, vs prof1 $P1U1) -> staggered loadid E (conc $LDCONC, pf $LDPF) vs E a1 -> E a1 vs ref12 a1 INFO ($([ -s $R12A1 ] && echo present || echo MISSING)) | START B (profiled, every $PEVERY, disk >= $PDISK MB): c8 [$PPLAN] -> P12PROF / P12TOP vs prof2 $P2GARUS us / $P2GARMS ms"''' + s[m.end():]

# ---- 11. dstart: P0 first, the placement files, the EPB note, the P0b gate, the readings
ds = func(s, "dstart")
dsn = ds.replace('''  diskok $tag || return 1
  p0sel
''', '''  diskok $tag || return 1
  p0epb || { STOPALL=1; return 1; }
  mkd $STATSDIR/place
  run "cp $PLDIR/$PL1 $PLDIR/$PL2 $STATSDIR/place/"
''')
dsn = dsn.replace('''  swnote $tag || { xlog "$tag: VIRHE the switches are not as expected -> stopped"; keeplog $tag-switch; down; return 1; }
''', '''  swnote $tag || { xlog "$tag: VIRHE the switches are not as expected -> stopped"; keeplog $tag-switch; down; STOPALL=1; return 1; }
  epbnote $tag || { xlog "$tag: VIRHE lever EPB is not declared as asked -> stopped"; keeplog $tag-epb; down; STOPALL=1; return 1; }
''')
dsn = dsn.replace('''  stopfile || p1reads
''', '''  stopfile || epbreads
''')
dsn = dsn.replace('''PROF1 readings: 1u confirm, c8 + split, c16 (CAP48), identity gs$GSC vs a1''', '''EPB readings: P0b determinism / BI, c8 [$C8ORDER], c16 [$C16ARMS] + newcomer E, 1u [$U1ARMS], staggered E, vs-int12 INFO''')
dsn = dsn.replace("start (SCREEN; $DESC)", "start (START A, unprofiled; lever EPB declared: RADIANCE_DS41_SLOTSUM=$SLROWS, tables $PL1 / $PL2, ON=0; $DESC)")
assert dsn.count("epbreads") == 1 and dsn.count("p0epb") == 1 and dsn.count("epbnote") == 1 and "PROF1 readings" not in dsn
s = s.replace(ds, dsn)

# ---- 12. the new functions (before main)
NEWF = open(sys.argv[4]).read()
rep("trap 'down' EXIT\n", NEWF + "trap 'down' EXIT\n")

# ---- 13. main: START A, its summary, START B (profiled), its summary
rep('''stopfile || dstart
summary
''', '''stopfile || dstart
summary
if [ $STOPALL = 1 ]; then xlog "$ITEM: STOPPED after START A (P0 / P0b / the EPB declaration failed): no profiled start"
elif [ $DOPROF = 1 ] && ! stopfile; then
  O=$O/prof; export STATSDIR=$O/stats; mkd $O $O/guard $O/c8 $O/snap $STATSDIR
  [ "$DRY" = 1 ] || : > $O/identity.txt
  dstartp; summaryp; epbprof
fi
''')
rep('''[ "$DRY" = 1 ] || : > $O/identity.txt\nstopfile || dstart''', '''[ "$DRY" = 1 ] || : > $O/identity.txt\nODIR=$O\nstopfile || dstart''')
rep('''xlog "$ITEM: done ($(elapsed) min) -- $O |''', '''O=$ODIR; epbtidy
xlog "$ITEM: done ($(elapsed) min) -- $O |''')

# ---- 14. summary: every c8 block, both c16 methods, the 1u reads by arm, the gs / loadid windows
sm = func(s, "summary")
SUMN = open(sys.argv[5]).read()
s = s.replace(sm, SUMN)
# prof2's profiled summary, renamed, its windows = this start's c8 blocks
p2s = func(p2, "summary")
p2s = p2s.replace("summary() {", "summaryp() {", 1).replace("xp2", "xepb").replace("XP2_", "XEP_")
p2s = p2s.replace("for w in c16:c16-cap c8:c8-1-S-O c4:c4-1-S-O; do", "for w in $PWINS; do")
p2s = p2s.replace("for w in c8:c8-1-S-O c16:c16-cap; do", "for w in $PWINS; do")
p2s = p2s.replace("local EL=$O/xepb-n5-engine-full.log.gz", "local EL=$O/xepb-p5-engine-full.log.gz PWINS=\"\" a k=1")
p2s = p2s.replace("  if [ \"$DRY\" = 1 ]; then", "  for a in $PPLAN; do PWINS=\"$PWINS c8$a:c8-$k-S-$a\"; k=$((k + 1)); done\n  if [ \"$DRY\" = 1 ]; then", 1)
p2s = p2s.replace("$([ ${w%%:*} = c8 ] && echo \"--ref ref12_c8:8.28:36.1:0.803:184.5\")", "--ref prof1_c8:8.32:36.18:0.801:184.0")
p2s = p2s.replace("for w in warm c8 c4 c16; do", "for w in $(for a in $PPLAN; do echo c8$a; done); do")
p2s = p2s.replace("(not a speed; MAXSEQS ${MAXSEQS:-?} vs ref12's 8)", "(not a speed; EPB profiled start, MAXSEQS ${MAXSEQS:-?})")
p2s = p2s.replace("  timeout 120 python3 -I -B $R12S fit", "  [ ${#FW[@]} -ge 4 ] && timeout 120 python3 -I -B $R12S fit")
assert "$PWINS" in p2s and "summaryp" in p2s and "-ge 4 ] && timeout 120" in p2s
rep("trap 'down' EXIT\n", p2s + "trap 'down' EXIT\n")

open(out, "w").write(s)
print("wrote", out, len(s), "bytes,", s.count("\n"), "lines")
