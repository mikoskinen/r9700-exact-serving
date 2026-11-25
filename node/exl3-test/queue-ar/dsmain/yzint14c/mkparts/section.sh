idok() {   # idok KEY ...: each identity.txt line yzi-KEY must be IDENTICAL (strict, T 0 greedy), else VIRHE + the key into IDFAIL (THE STOP)
  local k l
  [ "$DRY" = 1 ] && return 0
  for k in "$@"; do
    l=$(grep "^yzi-$k " $O/identity.txt 2>/dev/null | tail -1)
    case "$l" in *IDENTICAL*) case "$l" in *differ*|*DIFFER*) xlog "identity $k: VIRHE $l"; IDFAIL="$IDFAIL $k" ;; esac ;;
      *) xlog "identity $k: VIRHE ${l:-not compared} (strict, T 0 greedy)"; IDFAIL="$IDFAIL $k" ;; esac
  done
}
gt48() {   # gt48 NAME: the GET cap steps_gt48 delta between snapshots NAME-0 / NAME-1 -> a number or na
  [ "$DRY" = 1 ] && { echo 0; return 0; }
  python3 -c 'import json,sys
def rd(p):
    try: return json.load(open(p)).get("cap") or {}
    except Exception: return {}
a, b = rd(sys.argv[1]), rd(sys.argv[2])
print(int(b.get("steps_gt48", 0) - a.get("steps_gt48", 0)) if b else "na")' $O/fn-$1-0.json $O/fn-$1-1.json 2>/dev/null || echo na
}
# ---------------------------------------------------------------- int14c's own functions (P0 XPUSH / A4, identity + the stop, readings, summary)
p0json() {   # p0json KEY VALUE: one P0 outcome into $O/p0.json (yzsum.py reads it)
  [ "$DRY" = 1 ] && return 0
  python3 -c 'import json,sys
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: d = {}
d[sys.argv[2]] = sys.argv[3]
json.dump(d, open(p, "w"))' $O/p0.json "$1" "$2" 2>/dev/null || true
}
p0xp() {   # p0xp: XPUSH's P0 on the FOUR cards (BY CHIP UUID: ROCR_VISIBLE_DEVICES = every R9700's GPU-<unique_id>), no server --
           # r4d_selftest --case ar_twoshot_4rank_exact / ar_oneshot_4rank_exact under RADIANCE_FN4_AR_XPUSH 0 and 1: rc 0, 0 FAIL,
           # >= P0MIN bit-for-bit lines, the XPUSH line exactly in the =1 runs (naming the bound XPMIN), the SAME ok lines at 0 and 1;
           # then THE GATE (coordinator 13.11.: XPUSH from 1048576 elements, FN 5f0352a8): --case ar_xpush_perf at XPROWS (48,256,1024)
           # x 5120 bf16, each checked bit for bit at ar_xpush 0 AND 1 -- 48 rows (245760, below the bound: lockstep at 1), 256 / 1024
           # rows (above: XPUSH); the XPUSH line must say "from XPMIN elements; the first at 1310720" (256 x 5120: 48 rows stayed
           # lockstep). Every run gets RADIANCE_FN4_AR_XPUSH_MIN=XPMIN (the server's start word). A failure: XPON=0 (VIRHE).
  local cs x f rc nok nfail nbyte nxp r bad=0 devs u d4
  devs=$(for u in /sys/class/drm/card*/device/unique_id; do d4=$(dirname $u); [ "$(cat $d4/device 2>/dev/null)" = 0x7551 ] && echo "GPU-$(cat $u)"; done 2>/dev/null | sort -u | tr '\n' ',' | sed 's/,$//')
  if [ "$DRY" != 1 ] && [ $SW = 1 ] && [ "$(echo "$devs" | tr ',' '\n' | grep -c .)" != 4 ]; then
    XPON=0; p0json xpush "VIRHE not four R9700 cards by unique_id"; xlog "P0 XPUSH: VIRHE not four R9700 cards by unique_id ('$devs') -> XPUSH DROPPED (ar_xpush stays 0 in Z)"; return 0; fi
  local DEV4="--network none --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --ulimit memlock=-1 --ipc=host -e ROCR_VISIBLE_DEVICES=$devs"
  for cs in ar_twoshot_4rank_exact ar_oneshot_4rank_exact; do
    for x in 0 1; do
      f=$O/p0-$cs-xp$x.txt
      run "gpurun yzi-p0-$cs-$x-$$ 300 $DEV4 -e RADIANCE_FN4_AR_XPUSH=$x -e RADIANCE_FN4_AR_XPUSH_MIN=$XPMIN --entrypoint /opt/radiance/bin/r4d_selftest $IMAGE /opt/radiance/share/radiance/kernels/libr4d.so --case $cs > $f 2> $f.err"; rc=$?
      if [ "$DRY" = 1 ]; then echo "+ RESULT yzi P0 XPUSH $cs RADIANCE_FN4_AR_XPUSH=$x on the four cards (${devs:-<the four GPU-unique_id>}): rc 0, 0 FAIL, >= $P0MIN 'bit for bit' lines, the '[r4d] ar_xpush 1:' line $([ $x = 1 ] && echo present || echo absent) (bounded at 300 s), else VIRHE and XPUSH DROPPED (the item goes on)"; continue; fi
      nok=$(grep -c '^  ok' $f); nfail=$(grep -c '^  FAIL' $f); nbyte=$(grep '^  ok' $f | grep -c 'bit for bit'); nxp=$(grep '\[r4d\] ar_xpush 1:' $f.err | grep -c "from $XPMIN elements")
      r="$nok ok, $nfail FAIL, $nbyte bit-for-bit lines (need $P0MIN), XPUSH line naming the bound $XPMIN x$nxp (want $([ $x = 1 ] && echo ">= 1" || echo 0)), rc $rc"
      if [ $rc = 0 ] && [ "$nfail" = 0 ] && [ "$nbyte" -ge $P0MIN ] && { { [ $x = 1 ] && [ "$nxp" -ge 1 ]; } || { [ $x = 0 ] && [ "$nxp" = 0 ]; }; }; then
        xlog "P0 XPUSH $cs RADIANCE_FN4_AR_XPUSH=$x on the four cards (r4d_selftest --case): OK $r$([ $x = 1 ] && echo " | $(grep -m1 '\[r4d\] ar_xpush 1:' $f.err | cut -c1-200)")"
      else
        bad=1; xlog "P0 XPUSH $cs RADIANCE_FN4_AR_XPUSH=$x on the four cards (r4d_selftest --case): $([ $SW = 1 ] && echo VIRHE || echo "failed under YZI_SW=0 (a mock: docker run --device is not emulated)") $r | $(cat $f $f.err 2>/dev/null | grep -E '^  FAIL|^  skip|FATAL|error|Error|fault|abort|stopped after' | head -6 | sed 's/^ *//' | cut -c1-200 | tr '\n' ';' | cut -c1-1200)"
      fi
    done
    [ "$DRY" = 1 ] && continue
    if diff -q <(grep '^  ok' $O/p0-$cs-xp0.txt) <(grep '^  ok' $O/p0-$cs-xp1.txt) > /dev/null 2>&1 && [ "$(grep -c '^  ok' $O/p0-$cs-xp1.txt)" -gt 0 ]; then
      xlog "P0 XPUSH $cs: xpush 0 and 1 give the SAME $(grep -c '^  ok' $O/p0-$cs-xp1.txt) ok lines (each size == the fp32 ascending-rank fold on every rank, bit for bit, at both)"
    else bad=1; xlog "P0 XPUSH $cs: $([ $SW = 1 ] && echo VIRHE || echo "INFO (YZI_SW=0 mock)") the ok lines at xpush 0 and 1 differ ($(diff <(grep '^  ok' $O/p0-$cs-xp0.txt) <(grep '^  ok' $O/p0-$cs-xp1.txt) 2>&1 | grep '^[<>]' | head -3 | tr '\n' ';' | cut -c1-400))"; fi
  done
  # THE GATE: ar_xpush_perf's per-size byte checks at XPROWS, ar_xpush 0 and 1 (+ its AR4XP timing lines, INFO)
  f=$O/p0-ar_xpush_gate.txt
  run "gpurun yzi-p0-xpgate-$$ 300 $DEV4 -e RADIANCE_FN4_AR_XPUSH_MIN=$XPMIN -e R4D_SELFTEST_XPUSH_ROWS=$XPROWS --entrypoint /opt/radiance/bin/r4d_selftest $IMAGE /opt/radiance/share/radiance/kernels/libr4d.so --case ar_xpush_perf > $f 2> $f.err"; rc=$?
  if [ "$DRY" = 1 ]; then echo "+ RESULT yzi P0 XPUSH GATE (r4d_selftest --case ar_xpush_perf, rows $XPROWS x 5120 bf16, the bound $XPMIN): rc 0, 0 FAIL, every size bit for bit at ar_xpush 0 and 1 ($(( 2 * $(echo $XPROWS | tr ',' '\n' | grep -c .) )) ok lines), the load line 'RADIANCE_FN4_AR_XPUSH_MIN=$XPMIN', the XPUSH line 'from $XPMIN elements; the first at $(( $(echo $XPROWS | tr ',' '\n' | awk -v m=$XPMIN '$1 * 5120 >= m' | head -1) * 5120 ))' (the sizes below the bound stayed lockstep at 1); AR4XP INFO; else VIRHE and XPUSH DROPPED"; return 0; fi
  local want1 nsz ngate nld xl
  nsz=$(echo $XPROWS | tr ',' '\n' | grep -c .); want1=$(( $(echo $XPROWS | tr ',' '\n' | awk -v m=$XPMIN '$1 * 5120 >= m' | head -1) * 5120 ))
  nok=$(grep '^  ok  *ar_xpush_perf .* bit for bit' $f | grep -c .); nfail=$(grep -c '^  FAIL' $f)
  xl=$(grep -m1 '\[r4d\] ar_xpush 1:' $f.err | cut -c1-220); ngate=$(echo "$xl" | grep -c "from $XPMIN elements; the first at $want1\$")
  nld=$(grep -c "RADIANCE_FN4_AR_XPUSH_MIN=$XPMIN: ar_xpush's XPUSH form from $XPMIN elements" $f.err)
  while IFS= read -r l; do xlog "P0 XPUSH gate (INFO) ${l#  info  }"; done < <(grep -E '^  info  AR4XP' $f | head -6)
  if [ $rc = 0 ] && [ "$nfail" = 0 ] && [ "$nok" = $((2 * nsz)) ] && [ "$ngate" = 1 ] && [ "$nld" -ge 1 ]; then
    xlog "P0 XPUSH GATE (r4d_selftest --case ar_xpush_perf, rows $XPROWS x 5120, the four cards): OK $nok of $((2 * nsz)) size x switch byte checks (== the fp32 fold on every rank, ar_xpush 0 and 1), the bound $XPMIN read at load ($nld load line(s)), the first XPUSH launch at $want1 elements: the rows below the bound stayed lockstep at ar_xpush 1 | $xl"
  else
    bad=1; xlog "P0 XPUSH GATE: $([ $SW = 1 ] && echo VIRHE || echo "failed under YZI_SW=0 (a mock: docker run --device is not emulated)") rc $rc, $nok of $((2 * nsz)) byte checks ok, $nfail FAIL, load line x$nld, the XPUSH line '${xl:-none}' (want 'from $XPMIN elements; the first at $want1') | $(cat $f $f.err 2>/dev/null | grep -E '^  FAIL|^  skip|FATAL|error|fault|stopped after' | head -4 | cut -c1-200 | tr '\n' ';')"
  fi
  gzip -f $O/p0-ar_*-xp*.txt $O/p0-ar_xpush_gate.txt 2>/dev/null || true
  if [ $bad = 0 ]; then p0json xpush "PASS (twoshot / oneshot exact at 0 and 1; gate $XPMIN: rows $XPROWS bit for bit at 0 and 1, the first XPUSH launch at $want1)"; return 0; fi
  [ $SW = 0 ] && { p0json xpush "mock (not emulated)"; xlog "P0 XPUSH failed under YZI_SW=0 (a mock run: docker run --device is not emulated) -- the mock goes on"; return 0; }
  XPON=0; p0json xpush VIRHE; xlog "P0 XPUSH: VIRHE -> XPUSH DROPPED (ar_xpush stays 0 in Z); the item goes on with the other levers"; return 0
}
M16TAB=$STATSDIR/m16-forms.txt   # A4's table (yd2m's P0 bench, copied from the body dir; the server reads it at /stats/m16-forms.txt)
p0a4() {   # p0a4: A4's P0 on ONE card (P0UUID), no server -- the table (yzint14c/m16-forms.txt, md5 A4MD5, A4N lines) copied to M16TAB, then
           # r4d_selftest --mrcheck16 with it mounted read only (RADIANCE_DS41_GG_M16_FORMS): every A4 form + the switch vs the stock band,
           # plain / pair / folds a e, M 1..16 -> A4VERDICT PASS, the table loaded A4N of A4N. A failure: A4ON=0 (VIRHE; no table at start).
  local c=$O/p0-a4check.txt rc nchk nfail ntab v m
  m=$(md5sum < $A4TAB 2>/dev/null | cut -c1-8)
  if [ "$m" != "$A4MD5" ] || [ "$(grep -c . $A4TAB 2>/dev/null)" != "$A4N" ]; then
    A4ON=0; p0json a4 "VIRHE table md5 $m"; xlog "P0 A4: VIRHE the table $A4TAB is md5 ${m:-?} / $(grep -c . $A4TAB 2>/dev/null) lines, not $A4MD5 / $A4N -> A4 DROPPED"; return 0; fi
  run "cp $A4TAB $M16TAB"
  run "gpurun yzi-p0-a4check-$$ $P0CHECKS $DEV1 -v $M16TAB:/m16-forms.txt:ro -e RADIANCE_DS41_GG_M16_FORMS=/m16-forms.txt --entrypoint /opt/radiance/bin/r4d_selftest $IMAGE /opt/radiance/share/radiance/kernels/libr4d.so --mrcheck16 > $c 2>&1"; rc=$?
  if [ "$DRY" = 1 ]; then echo "+ RESULT yzi P0 A4 byte check (one card $P0UUID, bounded $P0CHECKS s, the table md5 $A4MD5 mounted read only): rc 0, 0 FAIL, 'lever A4 (gg_gemv_m16): $A4N forms from /m16-forms.txt', every MRCHECK line 'identical in every form', A4VERDICT ... PASS (byte-identical) -- else VIRHE and A4 DROPPED (the item goes on)"; return 0; fi
  nchk=$(grep -c 'identical in every form' $c); nfail=$(grep -c '^  FAIL\|DIFFERS' $c); v=$(grep -m1 '^A4VERDICT' $c)
  ntab=$(grep -m1 'lever A4 (gg_gemv_m16):' $c | sed -n 's/.*: \([0-9]*\) forms from.*/\1/p')
  xlog "P0 A4 byte check (r4d_selftest --mrcheck16, one card $P0UUID, M 1..16, yd2m's table md5 $m): rc $rc, $nchk MRCHECK lines identical in every form, $nfail FAIL / DIFFERS, table loaded ${ntab:-?} of $A4N | ${v:-no A4VERDICT line} | $(grep '^  ok' $c | grep -c 'A4 gemm_nt_ggml_pair') pair lines ok"
  gzip -f $c 2>/dev/null || true
  if [ $SW = 0 ]; then p0json a4 "mock (not emulated)"; xlog "P0 A4: YZI_SW=0 (a mock: docker run --device is not emulated) -- the mock goes on"; return 0; fi
  case "$v" in *"PASS (byte-identical)"*) ;; *) A4ON=0; p0json a4 VIRHE; xlog "P0 A4: VIRHE A4VERDICT not PASS -> A4 DROPPED (gg_gemv_m16 stays 0 in Z; no table at start)"; return 0 ;; esac
  [ "$rc" = 0 ] && [ "$nfail" = 0 ] && [ "${ntab:-x}" = "$A4N" ] || { A4ON=0; p0json a4 VIRHE; xlog "P0 A4: VIRHE rc $rc / FAIL $nfail / table ${ntab:-?} vs $A4N -> A4 DROPPED"; return 0; }
  p0json a4 PASS
}
xpline() {   # xpline TAG: the start log's "RADIANCE_FN4_AR_XPUSH_MIN=XPMIN: ar_xpush's XPUSH form from XPMIN elements" (libr4d's load line:
             # the bound in force, read back at start); missing / another bound: VIRHE and XPUSH DROPPED from Z (YZI_SW=0: INFO)
  [ "$DRY" = 1 ] && { echo "+ RESULT yzi $1: XPUSH gate at start (read back): the engine log names 'RADIANCE_FN4_AR_XPUSH_MIN=$XPMIN: ar_xpush's XPUSH form from $XPMIN elements' (else VIRHE and XPUSH DROPPED from Z)"; return 0; }
  local l; l=$(docker logs $CN 2>&1 | grep -m1 "RADIANCE_FN4_AR_XPUSH_MIN=" | sed 's/^radiance: //' | cut -c1-260)
  case "$l" in *"RADIANCE_FN4_AR_XPUSH_MIN=$XPMIN: ar_xpush's XPUSH form from $XPMIN elements"*) xlog "$1 XPUSH gate at start (read back): $l"; return 0 ;; esac
  if [ $SW = 0 ]; then xlog "$1 XPUSH gate at start: INFO (YZI_SW=0 mock) '${l:-no line}'"; return 0; fi
  [ $XPON = 1 ] && { XPON=0; p0json xpush "VIRHE the bound not read back at start"; }
  xlog "$1 XPUSH gate at start: VIRHE '${l:-no load line}' is not the bound $XPMIN -> XPUSH DROPPED from Z (ar_xpush stays 0)"; return 0
}
a4line() {   # a4line TAG: the start log's "lever A4 (gg_gemv_m16): N forms from /stats/m16-forms.txt" with N = A4N (A4 off: no such line)
  [ "$DRY" = 1 ] && { echo "+ RESULT yzi $1: the engine log names 'lever A4 (gg_gemv_m16): $A4N forms from /stats/m16-forms.txt' (else VIRHE: C / Z would run D2's forms); A4 off: no such line"; return 0; }
  local l n
  l=$(docker logs $CN 2>&1 | grep -m1 'lever A4 (gg_gemv_m16):' | sed 's/^radiance: //' | cut -c1-300); n=$(echo "$l" | sed -n 's/.*: \([0-9]*\) forms from.*/\1/p')
  if [ $A4ON = 0 ]; then
    [ -z "$l" ] && { xlog "$1 A4 table: A4 off -- no table at start"; return 0; }
    xlog "$1 A4 table: VIRHE A4 is off but the log says '$l'"; return 1
  fi
  if [ -n "$n" ] && [ "$n" = "$A4N" ]; then xlog "$1 A4 table: $l (the table's $A4N lines)"
  elif [ $SW = 0 ]; then xlog "$1 A4 table: INFO (YZI_SW=0 mock) '${l:-no line}'"
  else xlog "$1 A4 table: VIRHE '${l:-no line}' vs the table's $A4N lines"; return 1; fi
}
stagx() {   # stagx ARM REF: STAGGERED arrivals under ARM (zepb's epbstag, stack4/loadid.py run, read only): LDCONC decode streams + the 20
            # greedy probes one at a time, each with a FRESH LDPF-token prefill joining 0.5 s in (its chunks ride the probe's decode: MIXED
            # steps) -> ld-ARM vs a1-REF (this start's c1 alone), STRICT: a difference is a VIRHE line with the first differing char / token
            # and goes into IDFAIL (THE STOP)
  local a=$1 r=$2 t0=$(date +%s)
  arm $a || { IDFAIL="$IDFAIL ld-$a(no-pin)"; return 1; }
  pcreset yzi-ld-$a; ssnap yzi-ld-$a-0
  run "(cd $D && timeout 1800 python3 -I $LD run --url $URL --gs $GS --mb $MBP --out $O/yzi-ld-$a.jsonl --conc $LDCONC --warm 10 --offset 0.5 --pf-tok $LDPF --max-tokens $MT > $O/yzi-ld-$a.txt 2>&1)"
  ssnap yzi-ld-$a-1
  cmpx yzi ld-$a-vs-a1-$r $O/yzi-a1-$r.jsonl $O/yzi-ld-$a.jsonl
  [ "$DRY" = 1 ] && { echo "+ RESULT yzi staggered $a: LOADID (probes, MIXED steps during the probes, prefills cold) + ld-$a vs a1-$r strict (a difference = VIRHE -> THE STOP before the speed blocks)"; return 0; }
  xlog "staggered $a ($(NAMEOF $a) $(armj $a); loadid.py run, conc $LDCONC streams, fresh $LDPF-token prefill 0.5 s into each probe, T 0, $MT tok, $(( $(date +%s) - t0 )) s; compared with a1-$r): $(grep -m1 '^LOADID ' $O/yzi-ld-$a.txt | cut -c8-700) | probes in MIXED steps: $(python3 -c 'import json,sys
n = m = 0
for l in open(sys.argv[1]):
    try: d = json.loads(l)
    except ValueError: continue
    if "mixed" in d: n += 1; m += 1 if (d.get("mixed") or 0) > 0 else 0
print("%d of %d probes had MIXED steps in their life" % (m, n))' $O/yzi-ld-$a.jsonl 2>&1 | cut -c1-200) | cap $(capd yzi-ld-$a-0 yzi-ld-$a-1) | $(docker logs --since 10m $CN 2>&1 | grep 'lever A9 stats\|lever BI stats' | tail -2 | sed 's/^radiance: //; s/^[A-Z]\[[0-9]\] //' | cut -c1-200 | tr '\n' '|')"
  idok ld-$a-vs-a1-$r
  return 0
}
yzid() {   # 1 THE IDENTITY BLOCK (strict, T 0 greedy; module header): every check runs (each failure named), then yzreads stops on IDFAIL
  local k r g u0 u1
  alone yzi Z a1-Z
  if [ -s "$REF12A1" ] || [ "$DRY" = 1 ]; then cmpx yzi a1-Z-vs-ref12-a1 $REF12A1 $O/yzi-a1-Z.jsonl; idok a1-Z-vs-ref12-a1
  else xlog "identity a1-Z vs ref12 a1: VIRHE $REF12A1 missing -- not compared"; IDFAIL="$IDFAIL ref12-a1-missing"; fi
  alone yzi B a1-B
  if [ -s "$REF12A1" ] || [ "$DRY" = 1 ]; then cmpx yzi a1-B-vs-ref12-a1 $REF12A1 $O/yzi-a1-B.jsonl; idok a1-B-vs-ref12-a1; fi
  cmpx yzi a1-Z-vs-a1-B $O/yzi-a1-B.jsonl $O/yzi-a1-Z.jsonl; idok a1-Z-vs-a1-B
  stopfile && return 0
  u0=$(utcnow); ssnap yzi-gs$GSC-Z-0; gsx Z; ssnap yzi-gs$GSC-Z-1; u1=$(utcnow)
  [ "$DRY" = 1 ] || echo "$u0 $u1" > $O/gs$GSC-Z.win
  g=$(gt48 yzi-gs$GSC-Z)
  xlog "gs$GSC-Z cap (MAXSEQS ${MAXSEQS:-?}, all $GSC in flight, sched_row_cap $CAP): $(capd yzi-gs$GSC-Z-0 yzi-gs$GSC-Z-1) | steps_gt48 delta ${g:-na}$([ "${g:-na}" = 0 ] || { [ $SW = 1 ] && echo " -- VIRHE > 48-row steps under the cap (or no cap counters)" || echo " -- INFO (SW=0 mock: no cap counters)"; })"
  cmpx yzi gs$GSC-Z-vs-a1-Z $O/yzi-a1-Z.jsonl $O/yzi-gs$GSC-Z.jsonl; idok gs$GSC-Z-vs-a1-Z
  if [ $A11ON = 1 ]; then
    for k in $A11ID; do   # lever A11 IN RANGE: one prefill launch of ~k + 60 rows (deepq's task) alone, cold, B then Z
      stopfile && return 0
      coldx $k B 1 || { IDFAIL="$IDFAIL cold-$k(no-answer)"; continue; }
      coldx $k Z 2 || { IDFAIL="$IDFAIL cold-$k(no-answer)"; continue; }
      [ "$DRY" = 1 ] && continue
      r=$(cmpfile $O/cold-$k-1-B.txt $O/cold-$k-2-Z.txt); idrec "yzi-cold-$k-Z-vs-B" "$r" "the $NTOK-token answers after a $k-token cold prefill (one in-range launch: A11's ranged entries), byte for byte"
      idok cold-$k-Z-vs-B
    done
  fi
  stopfile && return 0
  stagx Z Z || true
  stopfile && return 0
  stagx A B || true   # A9 alone (ds41_bi 6, everything else B) vs a1-B (= a1-Z = ref12's a1 when the checks above passed)
  return 0
}
levreads() {   # 2 THE PER-LEVER READS (in-process): c8 C8ORD1 (A9 / A11's newcomer ms), A4's c16 (fixed harness + CAP48 method), SWL's read set
  local i=0 a k
  for a in $(echo $C8ORD1 | grep -o .); do
    i=$((i + 1)); stopfile && return 0
    case $a in G) [ $A11ON = 1 ] || { xlog "c8 block $i G SKIPPED (A11 off: ${GATEWHY:-its P0 failed})"; continue; } ;; esac
    budok "c8 block $i $a (lever reads)" 2 && { mbx 8 $i $a S || true; }
  done
  if [ $A4ON = 1 ]; then
    for a in B C; do stopfile && return 0; budok "c16 fixed harness $a (A4)" 2 && { mbx 16 $([ $a = B ] && echo 1 || echo 2) $a S || true; }; done
    for a in B C; do stopfile && return 0; budok "c16 CAP48 method $a (A4)" 2 && { c16x $([ $a = B ] && echo 1 || echo 2) $a || true; }; done
  else xlog "A4 off (its P0 failed): no A4 c16 reads"; fi
  stopfile && return 0
  if [ $SWLON = 1 ] && [ $SWLREADS = 1 ]; then
    if budok "SWL lever reads (B then W: code, essay x2 switched, warm essay; the all-on reads ~$ALLEST min kept after)" $((2 + ALLEST)); then
      for a in B W; do
        SETN=$a; SETK=$((SETK + 1))
        stopfile && return 0
        for k in 1 2; do blockx XC $a code || true; stopfile && return 0; blockx XE $a essay || true; stopfile && return 0; done
        blockx XW $a essay || true
      done
    else xlog "SWL lever reads DROPPED by the budget (coordinator: they go first; yswl measured them)"; fi
  fi
  return 0
}
allreads() {   # 3 THE ALL-ON READS: c8 C8ORD2, c16 fixed harness Z (+ B when step 2 ran none), 1u essay confirm B / Z, cold CDEPTHS B / Z
  local i=10 a k r
  for a in $(echo $C8ORD2 | grep -o .); do
    i=$((i + 1)); stopfile && return 0
    budok "c8 block $i $a (all-on)" 2 && { mbx 8 $i $a S || true; }
  done
  stopfile && return 0
  ls $O/c8/c16-*-S-B-reqs.jsonl > /dev/null 2>&1 || [ "$DRY" = 1 ] || { budok "c16 fixed harness B (no B block in step 2)" 2 && { mbx 16 3 B S || true; }; }
  budok "c16 fixed harness Z" 2 && { mbx 16 4 Z S || true; }
  for a in B Z; do
    stopfile && return 0
    SETN=U1$a; SETK=$((SETK + 1))
    budok "1u essay confirm $a (1 discarded + $U1N back to back)" 1 || continue
    blockx U1W $a essay || true
    for k in $(seq 1 $U1N); do stopfile && return 0; blockx U1 $a essay || true; done
  done
  for k in $CDEPTHS; do
    stopfile && return 0
    budok "cold $k B + Z" $(( 2 * (k / 100000 + 1) )) || continue
    coldx $k B 1 || continue
    coldx $k Z 2 || continue
    [ "$DRY" = 1 ] && continue
    r=$(cmpfile $O/cold-$k-1-B.txt $O/cold-$k-2-Z.txt); idrec "yzi-cold-$k-Z-vs-B" "$r" "the $NTOK-token answers, byte for byte"
    case "$r" in IDENTICAL*) ;; *) xlog "identity cold-$k Z vs B: VIRHE the answers differ (strict, T 0 greedy, the same prompt) -- NEEDS USER (after the speed blocks: reported)" ;; esac
  done
  return 0
}
yzreads() {   # the readings in their order (module header): 1 identity (THE STOP on any failure), 2 lever reads, 3 all-on reads
  if budok "identity (a1 Z / B, gs$GSC Z, A11 in range, staggered Z + A)" 8; then yzid || true
  else IDFAIL="$IDFAIL budget"; xlog "identity SKIPPED by the budget: VIRHE"; fi
  stopfile && return 0
  if [ -n "$IDFAIL" ]; then
    STOPPED=1
    xlog "IDENTITY GATE: VIRHE strict identity failed [$(echo $IDFAIL)] -> the item STOPS before every speed block (exact = byte-identical: NEEDS USER) | Z = $(armj Z) | the failing comparisons are in $O/identity.txt and $O/yzi-*-cmp-*.json"
    return 0
  fi
  if [ "$DRY" != 1 ] && [ $XPON = 1 ]; then
    local xl; xl=$(docker logs $CN 2>&1 | grep -m1 '\[r4d\] ar_xpush 1:' | cut -c1-240)
    case "$xl" in *"from $XPMIN elements; the first at "*) xlog "XPUSH in Z (read back, the first XPUSH launch of the run): $xl" ;;
      '') xlog "XPUSH in Z: NOTE no XPUSH launch logged yet (no message of >= $XPMIN elements under Z so far)" ;;
      *) xlog "XPUSH in Z: $([ $SW = 1 ] && echo VIRHE || echo "INFO (YZI_SW=0)") the first XPUSH launch names another bound: $xl" ;; esac
  fi
  xlog "IDENTITY GATE: every strict check IDENTICAL ($([ "$DRY" = 1 ] && echo "dry: no" || grep -c IDENTICAL $O/identity.txt 2>/dev/null) lines) -> the speed blocks run"
  levreads || true
  stopfile && return 0
  allreads || true
  return 0
}
splitblk() {   # splitblk LOG NAME: p12win.py split + rows of one mb block (its two marks) -> split-NAME.txt / rows-NAME.json
  local w; w=$(mwin $O/c8/$2-reqs.jsonl)
  [ -n "$w" ] || { xlog "summary: $2 -- no window (no marks in $O/c8/$2-reqs.jsonl)"; return 1; }
  timeout 300 python3 -I -B $R12S split $1 --win $2:$w --mbline $O/c8/$2.txt --sw0 $O/c8/$2-st-w0.json --sw1 $O/c8/$2-st-w1.json --target 200 > $O/split-$2.txt 2>&1
  timeout 300 python3 -I -B $R12S rows $1 --win $2:$w --json $O/rows-$2.json > $O/rows-$2.txt 2>&1
}
yzsummary() {   # per c8 / c16 block p12win.py split + rows, the gs16 step log (> 48 rows), then yzsum.py (the DECISION table)
  local EL=$O/yzi-n5-engine-full.log.gz nm u0 u1 l f
  if [ "$DRY" = 1 ]; then
    echo "+ per mb block <c8|c16>-<i>-S-<arm>: python3 -I -B $R12S split $EL --win <name>:<marks> --mbline $O/c8/<name>.txt --sw0/--sw1 $O/c8/<name>-st-w0/1.json --target 200 > $O/split-<name>.txt; rows --json $O/rows-<name>.json"
    echo "+ python3 -I -B $R12S rows $EL --win gs$GSC-Z:<u0>:<u1>: > 48-row decode steps = 0 under the cap, else VIRHE"
    echo "+ python3 -I -B $YDS $O --dec-th $DECTH --sw $SW --swlg $SWLG -> RESULT yzi YZSUM lines (the DECISION table: A9 / A11 c8 newcomer + decode vs B, A4 c16 fixed + CAP48 vs B, SWL switched / warm essay vs B, ALL c8 / c16 / 1u / cold 136k vs B, the identity / staggered outcomes, the P0s, the int12 references as INFO)"
    echo "+ tar -czf $O/raw-snaps.tar.gz (st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json) and remove them"
    return 0
  fi
  if [ -s $EL ]; then
    for f in $(ls $O/c8/c8-*-reqs.jsonl $O/c8/c16-*-reqs.jsonl 2>/dev/null); do
      nm=$(basename $f -reqs.jsonl)
      splitblk $EL $nm || continue
      while IFS= read -r l; do xlog "$nm ${l:0:1600}"; done < <(grep -hE '^P12SPLIT' $O/split-$nm.txt 2>/dev/null)
    done
    if [ -s $O/gs$GSC-Z.win ]; then
      read -r u0 u1 < $O/gs$GSC-Z.win
      timeout 300 python3 -I -B $R12S rows $EL --win gs$GSC-Z:$u0:$u1 --json $O/gsrows-Z.json > $O/gsrows-Z.txt 2>&1
      python3 -c 'import json,sys; d = json.load(open(sys.argv[1]))["windows"]; w = list(d.values())[0]; sys.exit(0 if w.get("gt48", 0) == 0 else 1)' $O/gsrows-Z.json 2>/dev/null \
        || { [ $SW = 1 ] && xlog "gs$GSC-Z step log: VIRHE > 48-row decode steps under the cap (or no step-log reading)" || xlog "gs$GSC-Z step log: INFO > 48-row steps (SW=0 mock: no row cap emulated)"; }
    fi
  else xlog "summary: NOTE no engine log $EL (the start did not come up): no splits"; fi
  timeout 120 python3 -I -B $YDS $O --dec-th $DECTH --sw $SW --swlg $SWLG > $O/yzsum.txt 2>&1
  while IFS= read -r l; do xlog "${l:0:2000}"; done < <(grep -h '^YZSUM' $O/yzsum.txt 2>/dev/null)
  grep -q '^YZSUM' $O/yzsum.txt 2>/dev/null || xlog "summary: VIRHE yzsum.py printed no YZSUM line ($(tail -3 $O/yzsum.txt 2>/dev/null | tr '\n' ' ' | cut -c1-300))"
  grep -lq '^Traceback' $O/split-*.txt $O/rows-*.txt $O/gsrows-*.txt $O/yzsum.txt 2>/dev/null && xlog "summary: VIRHE a Traceback in the analysis ($(grep -h -A4 '^Traceback' $O/split-*.txt $O/rows-*.txt $O/gsrows-*.txt $O/yzsum.txt 2>/dev/null | tail -2 | tr '\n' ' ' | cut -c1-300))"
  ( cd $O && tar -czf raw-snaps.tar.gz $(ls st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json 2>/dev/null) 2>/dev/null \
      && rm -f st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json ) || xlog "summary: NOTE the raw snapshots were not archived (left as they are)"
  gzip -f $O/start-*.log 2>/dev/null || true
  xlog "summary: raw snapshots -> raw-snaps.tar.gz ($(du -m $O/raw-snaps.tar.gz 2>/dev/null | cut -f1) MB); $O $(du -sm $O 2>/dev/null | cut -f1) MB; $(dfree) MB free on /"
  return 0
}
ngclean() {   # the end (or an early exit): ngram-shm-cleanup.sh (keeper stop + the tables and .lock files removed: FN items follow), logged (yswl's)
  local r b
  NGCLEAN=1
  run "docker rm -f $CN > /dev/null 2>&1"
  if [ "$DRY" = 1 ]; then echo "+ $NGS/ngram-shm-cleanup.sh   (keeper stop + $NGDIR/ngram-*.tab + .lock removed; refuses beside a radiance container; logs to runs/queue-changes.log) -- now: $(DRYRUN=1 $NGS/ngram-shm-cleanup.sh 2>&1 | tail -3 | tr '\n' ' ' | cut -c1-300)"; return 0; fi
  b=$(df -B1 --output=used /dev/shm 2>/dev/null | tail -1 | awk '{printf "%.1f GiB", $1 / 1073741824}')
  r=$("$NGS/ngram-shm-cleanup.sh" 2>&1 | tail -3 | tr '\n' ' ' | cut -c1-600)
  case "$r" in *REFUSED*) xlog "ngshare cleanup: VIRHE $r -- the tables stay in $NGDIR (run ngram-shm-cleanup.sh before the FN items)" ;;
    *) xlog "ngshare cleanup: $r | /dev/shm used $b -> $(df -B1 --output=used /dev/shm 2>/dev/null | tail -1 | awk '{printf "%.1f GiB", $1 / 1073741824}') | left in $NGDIR: $(ls $NGDIR 2>/dev/null | tr '\n' ' ' | cut -c1-200)" ;; esac
}
dstart() {   # P0 (XPUSH four cards, A4 + A11 one card), then the ONE screen start (ngshare)
  local tag=yzi-n5
  diskok $tag || return 1
  p0xp
  p0a4
  p0nc
  [ $A4ON = 1 ] && LEVENV="$LEVENV RADIANCE_DS41_GG_M16_FORMS=/stats/m16-forms.txt"
  LEVENV="$LEVENV RADIANCE_FN4_AR_XPUSH_MIN=$XPMIN"   # XPUSH's bound (coordinator 13.11.: 1048576, FN 5f0352a8), declared so the start log names it
  if [ $A11ON = 1 ]; then LEVENV=$(echo "$LEVENV" | sed "s#RADIANCE_DS41_GG_PF_TABLE=$GPTAB #RADIANCE_DS41_GG_PF_TABLE=$GPTAB,$A11ENT #")
    case "$LEVENV" in *"RADIANCE_DS41_GG_PF_TABLE=$GPTAB,$A11ENT "*) ;; *) xlog "$tag: VIRHE the ranged entries did not go into LEVENV -> NOT STARTED"; return 1 ;; esac
  else xlog "A11 off (${GATEWHY:-its P0 failed}): no G block, Z without gg_pf 2"; fi
  p0json a11 "$([ $A11ON = 1 ] && echo "PASS, $(echo "$A11ENT" | tr ',' '\n' | grep -c .) ranged entries: $A11ENT" || echo "off: ${GATEWHY:-its P0 failed}")"
  p0json levers "XPUSH $XPON A4 $A4ON A11 $A11ON A9 1 SWL $SWLON (heat_min_gain_x100 $SWLG)"
  xlog "P0 done ($(elapsed) min): XPUSH $([ $XPON = 1 ] && echo on || echo DROPPED), A4 $([ $A4ON = 1 ] && echo on || echo DROPPED), A11 $([ $A11ON = 1 ] && echo "on ($A11ENT)" || echo "off (${GATEWHY:-P0})"), A9 on (no one-card check), SWL $([ $SWLON = 1 ] && echo "on (heat_min_gain_x100 $SWLG)" || echo off) -> Z = $(armj Z)"
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  [ -n "${YZI_MAXSEQS:-}" ] && export MAXSEQS=$YZI_MAXSEQS
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on" PROFEVERY=
  export AQLSTAMP=$([ $STAMPN -gt 0 ] && echo $STAMPN)
  export NGRAM=ram=/ngram   # ngshare.sh's restart sets it too; here it shows in the start line
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV > $XS2_ENVFILE; rm -f $STATSDIR/stamps.csv*"; else printf '%s\n' $LEVENV > $XS2_ENVFILE; rm -f $STATSDIR/stamps.csv*; fi
  xlog "$tag: start (SCREEN; ngshare: Y $(basename $Y), NGRAM=ram=/ngram, keeper; n-gram tables now $(ls $NGDIR/ngram-*.tab 2>/dev/null | wc -l) (0: this start fills them, ~308 s); $DESC) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $(pinj O | cut -c1-900) | AQLSTAMP=${AQLSTAMP:-unset} | env: $(env | grep -E '^(RADIANCE_|R4D_)' | grep -vE '(KEY|TOKEN|SECRET)' | sort | tr '\n' ' ' | cut -c1-2800)"
  gate1 $tag 70 || { xlog "$tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  modes $tag || { xlog "$tag: VIRHE the modes are not the start's -> stopped"; down; return 1; }
  drafted $tag || { down; return 1; }
  readback $tag || { keeplog $tag-readback; down; return 1; }
  a11line $tag || { keeplog $tag-table; down; return 1; }
  a4line $tag || { keeplog $tag-a4table; down; return 1; }
  xpline $tag
  vramstart $tag
  swnote $tag || { xlog "$tag: VIRHE the switches are not as expected -> stopped"; keeplog $tag-switch; down; return 1; }
  identw $tag $WARM_REF_225
  [ "$DRY" = 1 ] || { : > $O/blocks.jsonl; : > $O/mb.jsonl; : > $O/c16cap.jsonl; }
  xlog "THE BASE PIN B (ref12's, read back): $(pinj B | cut -c1-1200) | A = $(armj A) | G = $(armj G)$([ $A11ON = 1 ] || echo " (A11 OFF)") | C = $(armj C)$([ $A4ON = 1 ] || echo " (A4 OFF)") | W = $(armj W) | Z (ALL) = $(armj Z) | readings: 1 identity a1-Z / a1-B vs ref12, gs$GSC-Z, A11 in range [$A11ID], staggered Z + A (ANY failure: STOP) | 2 c8 [$C8ORD1], A4 c16 fixed + CAP48 B / C, SWL reads B / W | 3 c8 [$C8ORD2], c16 Z, 1u essay B / Z (1 + $U1N), cold [$CDEPTHS] B / Z"
  stopfile || yzreads
  capture $O/$tag-engine-full.log.gz
  xlog "$tag: at the end: $(i11now) | $(fn4get | python3 -c 'import json,sys; w=json.load(sys.stdin)["switches"]; print(" ".join("%s %s" % (k, w.get(k)) for k in ("ds41_bi", "gg_pf", "ar_xpush", "gg_gemv_m16", "place_swap_lent", "heat_min_gain_x100", "heat_hyst_x100")))' 2>/dev/null) | $(docker logs --since 6h $CN 2>&1 | grep -E 'lever A9 stats|lever BI stats' | tail -2 | sed 's/^radiance: //; s/^[A-Z]\[[0-9]\] //' | cut -c1-200 | tr '\n' '|') | $(docker logs --since 6h $CN 2>&1 | grep -cE '^E ') E lines | $(elapsed) min"
  down   # the stamp writer flushes at exit
  [ "$DRY" = 1 ] || sleep 3
  xlog "$tag: done$([ $STOPPED = 1 ] && echo " (STOPPED at the identity gate: no speed block ran)")"
}
trap 'down; [ $NGB = 1 ] && [ $NGCLEAN = 0 ] && ngclean' EXIT   # an early exit frees /dev/shm too (FN items follow)
trap 'exit 143' TERM INT
