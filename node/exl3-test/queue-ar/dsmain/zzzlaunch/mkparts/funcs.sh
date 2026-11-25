X3DN=0; X3DLOG=""   # lever RCO's switch read back (zzzlaunch; zzk25s's names kept): checks done / the last section's arms
x3dchk() {   # x3dchk ARM: lever RCO's switch READ BACK right after arm() applied ARM -- GET /admin/fn4 switches.route_copy_ov = RZ under Z, RB
             # under every other arm (B, O), AND exl3_x3d = XPIN (the int15 pin) -> 0 (logged per section, with GET "rco": recordings /
             # copies / freed, the engagement counter); else VIRHE -> 1 (the block is not read); LCH_SW=0 (mock: neither is posted): INFO, 0
  local want got gx rc
  case $1 in Z) want=$RZ ;; *) want=$RB ;; esac
  [ "$DRY" = 1 ] && { echo "+ GET /admin/fn4: switches.route_copy_ov == $want and exl3_x3d == $XPIN under arm $1, \"rco\" logged (asserted before the block; else VIRHE: the block is not read)"; return 0; }
  read -r got gx rc < <(fn4get | python3 -c 'import json,sys
d = json.load(sys.stdin); w = d.get("switches") or {}; r = d.get("rco") or {}
print(w.get("route_copy_ov"), w.get("exl3_x3d"), "%s/%s/%s" % (r.get("recordings"), r.get("copies"), r.get("freed")))' 2>/dev/null)
  X3DN=$((X3DN + 1)); X3DLOG="$X3DLOG $1:${got:-none}/x3d${gx:-none}/rco${rc:-none}"
  echo "$(date '+%F %T') arm $1 route_copy_ov ${got:-none} (want $want) exl3_x3d ${gx:-none} (want $XPIN) rco recordings/copies/freed ${rc:-none}" >> $O/rco-readback.txt
  [ "$got" = "$want" ] && [ "$gx" = "$XPIN" ] && return 0
  [ $SW = 0 ] && return 0
  xlog "arm $1: VIRHE GET /admin/fn4 route_copy_ov ${got:-none} / exl3_x3d ${gx:-none} after the arm was applied (want $want / $XPIN) -- this block is not read"; ARMOK=0; return 1
}
x3dsay() {   # x3dsay SECTION: one line -- route_copy_ov / exl3_x3d / "rco" (recordings/copies/freed, cumulative) read back before the section's blocks
  [ "$DRY" = 1 ] && return 0
  xlog "$1: route_copy_ov / exl3_x3d / rco (recordings/copies/freed, cumulative: they grow only while route_copy_ov 1 records passes) read back before each block (GET /admin/fn4, arm:value):$X3DLOG$([ $SW = 0 ] && echo " (INFO: LCH_SW=0 mock, neither switch posted)") | $X3DN check(s) so far, $O/rco-readback.txt"
  X3DLOG=""
}
clockpair() {   # clockpair: "UTC_S MONO_NS" from ONE process (the harness marks' clock and the stamps' clock, read together; zzzaql's)
  [ "$DRY" = 1 ] && { echo "0 0"; return 0; }
  python3 -c 'import time; print("%.6f %d" % (time.time(), time.clock_gettime_ns(time.CLOCK_MONOTONIC)))'
}
clk() {   # clk NAME: one clock pair onto $O/clock-NAME.txt (before and after a stamped block)
  if [ "$DRY" = 1 ]; then echo "+ clockpair >> $O/clock-$1.txt"; else clockpair >> $O/clock-$1.txt; fi
}
aqlread() {   # THE AQL READ (module header): the c8 gens blocks' windows on the monotonic clock (aqlcp.py monowin), lchaql.py read per arm, then compare
  local ST=$STATSDIR/stamps.csv f nm a w wins="" swins="" l
  if [ "$DRY" = 1 ]; then
    echo "+ per c8 gens block: python3 -I -B $AQL monowin $O/c8/c8-<i>-G-<arm>-reqs.jsonl $O/clock-c8-<i>-G-<arm>.txt -> M0:M1 (the block's window on the monotonic clock)"
    echo "+ $([ $SW = 0 ] && echo "MOCK: python3 -I -B $AQL synth --out $ST --win ... (SYNTHETIC stamps over the windows, logged MOCK); ")nice -n 10 python3 -I -B $LAQ read $ST --win c8-<i>-G-<arm>:<arm>:M0:M1 ... --json $O/aql.json > $O/aql.txt; python3 -I -B $LAQ compare --json $O/aql.json --base B --arm Z (lines LCHAQL ...); gzip the stamps"
    return 0
  fi
  [ $STAMPN -gt 0 ] || { xlog "AQL: NOTE LCH_STAMP 0 -- no stamps, no site table"; return 0; }
  for f in $(ls $O/c8/c8-*-G-*-reqs.jsonl 2>/dev/null); do
    nm=$(basename $f -reqs.jsonl); a=${nm##*-}
    if w=$(timeout 60 python3 -I -B $AQL monowin $f $O/clock-$nm.txt 2>> $O/aql-err.txt) && [ -n "$w" ]; then wins="$wins --win $nm:$a:$w"; swins="$swins --win $nm:$w"
    else xlog "AQL: VIRHE no monotonic window for $nm ($(tail -1 $O/aql-err.txt 2>/dev/null | cut -c1-200))"; fi
  done
  [ -n "$wins" ] || { xlog "AQL: VIRHE no c8 gens window mapped to the monotonic clock -- no site table"; return 0; }
  if [ $SW = 0 ] && [ ! -s $ST ] && [ ! -s $ST.gz ]; then
    timeout 300 python3 -I -B $AQL synth --out $ST $swins > /dev/null 2>> $O/aql-err.txt \
      && xlog "MOCK (LCH_SW=0): the mock server writes no stamps -- aqlcp.py synth wrote SYNTHETIC stamps over the windows (NOT a reading)"
  fi
  [ -s $ST ] || { [ -s $ST.gz ] && ST=$ST.gz; }
  [ -s $ST ] || { xlog "AQL: VIRHE no stamps file $STATSDIR/stamps.csv (RADIANCE_AQL_STAMP=$STAMPN) -- no site table"; return 0; }
  xlog "AQL: stamps $(du -m $ST | cut -f1) MB ($(basename $ST)); engine: $(zcat $O/xlch-n5-engine-full.log.gz 2>/dev/null | grep -m1 'RADIANCE_AQL_STAMP times' | sed 's/^[^ ]* //' | cut -c1-160); windows$(echo $wins | sed 's/--win / /g')"
  nice -n 10 timeout 900 python3 -I -B $LAQ read $ST $wins --json $O/aql.json > $O/aql.txt 2>> $O/aql-err.txt
  timeout 120 python3 -I -B $LAQ compare --json $O/aql.json --base B --arm Z >> $O/aql.txt 2>> $O/aql-err.txt
  while IFS= read -r l; do xlog "${l:0:3200}"; done < <(grep -h '^LCHAQL' $O/aql.txt 2>/dev/null)
  grep -q '^LCHAQL COMPARE lever RCO' $O/aql.txt 2>/dev/null || xlog "AQL: VIRHE lchaql.py printed no COMPARE line ($(tail -2 $O/aql-err.txt 2>/dev/null | tr '\n' ' ' | cut -c1-300))"
  grep -q '^Traceback' $O/aql-err.txt 2>/dev/null && xlog "AQL: VIRHE a Traceback in the AQL read ($(grep -A4 '^Traceback' $O/aql-err.txt | tail -2 | tr '\n' ' ' | cut -c1-300))"
  case $ST in *.gz) ;; *) gzip -1 -f $ST && xlog "AQL: stamps kept gzipped for a re-parse ($(du -m $ST.gz | cut -f1) MB)" ;; esac
  return 0
}
