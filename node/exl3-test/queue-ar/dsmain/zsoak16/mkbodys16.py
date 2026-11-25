#!/usr/bin/env python3
"""mkbodys16.py (13.11.2026, queue-ar/dsmain/zsoak16, a builder for ds-main-port) -- writes dsm-zsoak16.sh (the body of
57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzzzzzzzz-zsoak16.sh) from zzzc16/dsm-zzzc16.sh (md5 00cfc8b8, the int16 confirmation body:
THE source of truth for the start, the pin, the tables, the read-back and the engagement probe). Every edit is an asserted
replacement (the anchor must occur exactly as often as stated); zzzc16's body is read only. The soak's own functions are inserted
before dstart(); the summary / pick / engagement rule live in zsoak16/soaksum.py.
  python3 mkbodys16.py SRC DST       (SRC = zzzc16/dsm-zzzc16.sh, DST = zsoak16/dsm-zsoak16.sh)
"""
import hashlib, sys

SRC, DST = sys.argv[1], sys.argv[2]
s = open(SRC).read()
m = hashlib.md5(s.encode()).hexdigest()
assert m.startswith("00cfc8b8"), "source md5 %s, not zzzc16's 00cfc8b8" % m


def rep(old, new, n=1):
    global s
    c = s.count(old)
    assert c == n, "anchor %r occurs %d times (want %d)" % (old[:90], c, n)
    s = s.replace(old, new)



# ---------------------------------------------------------------------------------------------- 2 the tag, the knobs, the paths
n_x, n_k = s.count("xzc16"), s.count("ZC16_")
assert n_x > 50 and n_k > 100, (n_x, n_k)
s = s.replace("xzc16", "xzs16").replace("ZC16_", "ZS16_")
rep("O=${ZS16_OUT:-$R/xdsm/zzzc16}; export STATSDIR=$O/stats", "O=${ZS16_OUT:-$R/xdsm/zsoak16}; export STATSDIR=$O/stats")
rep("ITEM=${ZS16_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzzc-int16}", "ITEM=${ZS16_ITEM:-57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzzzzzzzz-zsoak16}")
rep("YDS=$HD/zc16sum.py; SAS=$HD/zc16sum.py; STP=$HD/steps.py; NCM=$HD/ncmatch.py;",
    "YDS=$HD/soaksum.py; SAS=$HD/soaksum.py; STP=$HD/../zzzc16/steps.py; NCM=$HD/../zzzc16/ncmatch.py;")
rep("A4TAB=$HD/m16-forms.txt;", "A4TAB=$HD/../zzzc16/m16-forms.txt;")
rep("A11TAB=$HD/a11-entries.txt;", "A11TAB=$HD/../zzzc16/a11-entries.txt;")

# ---------------------------------------------------------------------------------------------- 3 the pin auto-choice + the soak knobs
PICK = r'''RFDECL=${ZS16_RFDECL:-2}; FMF=0   # RADIANCE_DS41_RFUSE=2 declared at start in BOTH choices (zzzc16's start; its arm B = int15 ran on it); ds41_mix_fin 0
# zsoak16: THE PIN AUTO-CHOICE (module header) -- soaksum.py pick over the RESULT log (ZS16_PICKLOG: tests only); a picker failure = "missing" -> int16
PICKLOG=${ZS16_PICKLOG:-$L}
PICK=$(timeout 60 python3 -I -B $HD/soaksum.py pick --log "$PICKLOG" 2>&1 | grep -m1 '^PICK ' | cut -c6-)
case "$PICK" in
  "int15 "*) PINNAME=int15; FRF=0; FRC=0; PICKWHY="because ${PICK#int15 FAIL }" ;;
  "int16 "*) PINNAME=int16; FRF=2; FRC=1; PICKWHY="${PICK#int16 }" ;;
  *) PINNAME=int16; FRF=2; FRC=1; PICKWHY="MISSING: the picker gave no answer ('${PICK:-nothing}' from $HD/soaksum.py on $PICKLOG) -- treated as the xzc16 result missing" ;;
esac
SOAKN=${ZS16_CYCLES:-8}; SOAKMIN=${ZS16_CYCLES_MIN:-6}; CYCS=${ZS16_CYCLE_S:-280}; SWARM0=${ZS16_WARM:-60}; WMIN=${ZS16_WARM_MIN:-10}; WMAX=${ZS16_WARM_MAX:-240}
SCONC=${ZS16_LOADCONC:-12}; SPF=${ZS16_PFTOK:-8192}; SOFF=${ZS16_OFFSET:-0.5}; CLOSES=${ZS16_CLOSE_S:-150}
TELS=${ZS16_TELS:-5}; TJMAX=${ZS16_TJMAX:-100}; TMMAX=${ZS16_TMMAX:-96}; TNHOT=${ZS16_TNHOT:-2}; MOCKDIFF=${ZS16_MOCKDIFF:-0}
ZC16DIR=${ZS16_ZC16DIR:-$R/xdsm/zzzc16}; TEMPLOG=${ZS16_TEMPLOG_DIR:-$HOME/gpu/temps}
for v in $SOAKN $SOAKMIN $CYCS $SWARM0 $WMIN $WMAX $SCONC $SPF $CLOSES $TELS $TJMAX $TMMAX $TNHOT $MOCKDIFF; do case $v in ''|*[!0-9]*) echo "a zsoak16 knob is not a number ('$v')"; exit 1 ;; esac; done
case $SOFF in ''|*[!0-9.]*) echo "ZS16_OFFSET is seconds (got '$SOFF')"; exit 1 ;; esac
[ $SOAKN -ge 1 ] && [ $SOAKN -le 20 ] || { echo "ZS16_CYCLES 1..20"; exit 1; }
[ $SCONC -ge 1 ] && [ $SCONC -le 14 ] || { echo "ZS16_LOADCONC 1..14 (MAXSEQS 16: the load + 1 probe + 1 prefill)"; exit 1; }
[ $WMIN -ge 1 ] && [ $WMIN -le $WMAX ] && [ $TELS -ge 1 ] && [ $TNHOT -ge 1 ] || { echo "ZS16_WARM_MIN 1..ZS16_WARM_MAX, ZS16_TELS / ZS16_TNHOT >= 1"; exit 1; }
[ "$MOCKDIFF" = 0 ] || [ "${ZS16_SW:-1}" = 0 ] || { echo "ZS16_MOCKDIFF is a mock-only knob (needs ZS16_SW=0)"; exit 1; }
if [ "${ZS16_SW:-1}" = 0 ]; then SYSD=${ZS16_SYSFS:-FAKE}; else SYSD=${ZS16_SYSFS:-/sys/class/drm}; fi   # the mock never reads the real cards
case "$PINNAME$FRF$FMF$FRC" in int16201|int15000) ;; *) echo "the pin choice is inconsistent ($PINNAME $FRF $FMF $FRC)"; exit 1 ;; esac'''
rep("RFDECL=${ZS16_RFDECL:-2}; FRF=2; FMF=0; FRC=1   # RADIANCE_DS41_RFUSE at start (int16: FUSE-R declared AND started at form 2); THE int16 PIN's ds41_rfuse / ds41_mix_fin / route_copy_ov",
    PICK + "\n#", 1)
# the pin's name everywhere the int16 confirmation named it (labels only; the values come from FRF / FMF / FRC)
c = s.count("THE int16 PIN")
assert c >= 15, c
s = s.replace("THE int16 PIN", "THE $PINNAME PIN")
rep('DESC="THE int16 CONFIRMATION (zzzc16: ', 'DESC="THE SOAK ON THE $PINNAME PIN (zsoak16; the start, the pin post, the tables, the read-back = zzzc16\'s; ')
rep('assert i16 == (2, 0, 1) and (d["ds41_rfuse"]', 'assert i16 in ((2, 0, 1), (0, 0, 0)) and (d["ds41_rfuse"]')

# ---------------------------------------------------------------------------------------------- 4 pin16 = the engagement probe, its rule in soaksum.py
j0 = s.index("pin16() {   # pin16 TAG: THE $PINNAME PIN READ BACK beyond rankrb")
j1 = s.index("c16id() {   # 1 THE IDENTITY BLOCK")
PIN16 = r'''pin16() {   # pin16 TAG [WHEN]: THE PIN's ENGAGEMENT (zsoak16; the rule = soaksum.py fuse_check + its form1_expected() hook, module header):
            # snapshot, arm O again (epoch_bump: every pass re-records; arm() reads back every switch) + one 64-token greedy request,
            # snapshot -> soaksum.py engage --recorded 1 -> 0; else VIRHE -> 1 (at the start: nothing is read); ZS16_SW=0 (the mock serves
            # no "fuse" / "rco"): INFO, 0. The verdict line goes to $O/engage.txt for the summary.
  local r
  ssnap $1-eng-0
  arm O || { xlog "$1 engagement: VIRHE arm O (THE $PINNAME PIN) not applied on the engagement probe${2:+ ($2)}"; [ "$DRY" = 1 ] || echo "$1 VIRHE arm O not applied" >> $O/engage.txt; return 1; }
  run "curl -s -m 120 -X POST -H 'Content-Type: application/json' -d '{\"prompt\": \"Write one sentence about the sea.\", \"n_predict\": 64, \"temperature\": 0}' $URL/completion > /dev/null"
  ssnap $1-eng-1
  if [ "$DRY" = 1 ]; then echo "+ RESULT xzs16 $1 THE $PINNAME PIN engagement${2:+ ($2)}: python3 -I -B $YDS engage --fn0 fn-$1-eng-0.json --fn1 fn-$1-eng-1.json --pin $PINNAME --xb $XB --recorded 1 --prefills 1 (int16: ds41_rfuse/ds41_mix_fin/route_copy_ov/exl3_x3d $FRF/$FMF/$FRC/$XB, rco in force, form 2 + rgs + rco recordings / freed move, mixfin 0, form 1 per form1_expected: INFO up to 80 a prefilled request, else VIRHE; int15: no fused form) -> else VIRHE"; return 0; fi
  r=$(timeout 120 python3 -I -B $YDS engage --fn0 $O/fn-$1-eng-0.json --fn1 $O/fn-$1-eng-1.json --pin $PINNAME --xb $XB --recorded 1 --prefills 1 2>&1 | grep -m1 '^ENGAGE ' | cut -c8-1600)
  echo "$1 ${r:-VIRHE soaksum.py engage gave no line}" >> $O/engage.txt
  case "$r" in OK*) xlog "$1 THE $PINNAME PIN read back + engagement${2:+ ($2)} (rule: soaksum.py fuse_check / form1_expected): ${r#OK }"; return 0 ;; esac
  [ $SW = 0 ] && { xlog "$1 THE $PINNAME PIN engagement${2:+ ($2)}: INFO (ZS16_SW=0 mock: no fusion switches / counters served) ${r#VIRHE }"; return 0; }
  xlog "$1 THE $PINNAME PIN read back + engagement${2:+ ($2)}: VIRHE ${r#VIRHE }"; return 1
}
'''
s = s[:j0] + PIN16 + s[j1:]

# ---------------------------------------------------------------------------------------------- 5 the soak's functions (before dstart)
SOAK = r'''# ---------------------------------------------------------------- zsoak16's own functions (mkbodys16.py): the engine-log capture from the start,
# the telemetry sampler + temperature guard, the read-only pin hold check, the alone reads, the strict compares, one cycle, the readings, the summary
elsec() { echo $(( $(date +%s) - T00 )); }
CAPPID=""; TELPID=""; PINDRIFT=""; SCMP=""; ZREF=""
capstart() {   # capstart TAG: the engine log FROM THE START, followed and gzipped while it runs (TESTING.md 13.11. lesson 4: json-file keeps 5 x 100 MB)
  [ "$DRY" = 1 ] && { echo "+ (docker logs -f -t $CN 2>&1 | gzip -1 > $O/$1-engine-full.log.gz) &   # ends with the container"; return 0; }
  (docker logs -f -t $CN 2>&1 | gzip -1 > $O/$1-engine-full.log.gz) & CAPPID=$!
}
capwait() {   # capwait: after down the follow ends with the container; waited <= 120 s, then stopped; its size + step lines logged
  [ -n "$CAPPID" ] || return 0
  local i f; for i in $(seq 1 60); do kill -0 $CAPPID 2>/dev/null || break; sleep 2; done
  kill $CAPPID 2>/dev/null; wait $CAPPID 2>/dev/null; CAPPID=""
  f=$(ls $O/*-engine-full.log.gz 2>/dev/null | head -1)
  [ -n "$f" ] && xlog "engine log $(basename $f) (followed from the start): $(du -m $f | cut -f1) MB gz, $(zcat $f 2>/dev/null | grep -cE ' step [0-9]+ (prefill|decode|MIXED) n_seq=') step lines; $(dfree) MB free"
}
mkfakesys() {   # mkfakesys DIR: the mock's sysfs tree (ZS16_SW=0 never reads the real cards): 4 cards, unique_id / pp_dpm_sclk / hwmon
  local k pci dv
  for k in 1 2 3 4; do pci=$(echo 03 48 83 c3 | cut -d' ' -f$k); dv=$1/devices/0000:$pci:00.0
    mkdir -p $dv/hwmon/hwmon$k $1/card$k; ln -sfn $dv $1/card$k/device
    echo "00000000mock000$k" > $dv/unique_id; printf '0: 600Mhz\n1: 2831Mhz *\n' > $dv/pp_dpm_sclk
    echo $((200 + k))000000 > $dv/hwmon/hwmon$k/power1_average; echo $((60 + k))000 > $dv/hwmon/hwmon$k/temp2_input; echo $((54 + k))000 > $dv/hwmon/hwmon$k/temp3_input
    echo 50000 > $dv/hwmon/hwmon$k/temp1_input; echo 2831000000 > $dv/hwmon/hwmon$k/freq1_input; echo $((2500 + 10 * k)) > $dv/hwmon/hwmon$k/fan1_input
  done
}
TELHDR="epoch,uuid,pci,power_w,junction_c,mem_c,edge_c,sclk_mhz,sclk_level,fan_rpm"
telemon() {   # background until $O/soak.tstop: every TELS s one row a card (chip UUID = GPU-<unique_id>) -> $O/telemetry.csv, READ-ONLY sysfs reads
              # (nothing is set); the temperature guard: junction >= TJMAX or memory >= TMMAX on any card for TNHOT samples in a row -> $O/soak.hot
  local d h u p j m e s f lv a b c t pci hot n=0
  while [ ! -e $O/soak.tstop ]; do
    hot=""; t=$(date +%s.%N | cut -c1-14)
    for d in $SYSD/card*/device; do
      [ -e $d/unique_id ] || continue
      h=$(ls -d $d/hwmon/hwmon* 2>/dev/null | head -1); [ -n "$h" ] || continue
      read -r u < $d/unique_id
      p=$(cat $h/power1_average 2>/dev/null || cat $h/power1_input 2>/dev/null || echo 0)
      j=$(cat $h/temp2_input 2>/dev/null || echo 0); m=$(cat $h/temp3_input 2>/dev/null || echo 0); e=$(cat $h/temp1_input 2>/dev/null || echo 0)
      s=$(cat $h/freq1_input 2>/dev/null || echo 0); f=$(cat $h/fan1_input 2>/dev/null || echo "")
      lv=""; if [ -r $d/pp_dpm_sclk ]; then while read -r a b c; do [ "$c" = "*" ] && lv=${a%:}; done < $d/pp_dpm_sclk; fi
      pci=$(basename $(readlink -f $d))
      echo "$t,GPU-$u,$pci,$((p / 1000000)),$((j / 1000)),$((m / 1000)),$((e / 1000)),$((s / 1000000)),${lv:-na},$f"
      { [ $((j / 1000)) -ge $TJMAX ] || [ $((m / 1000)) -ge $TMMAX ]; } && hot="$hot GPU-$u:j$((j / 1000))/m$((m / 1000))"
    done >> $O/telemetry.csv
    if [ -n "$hot" ]; then n=$((n + 1)); else n=0; fi
    if [ $n -ge $TNHOT ] && [ ! -e $O/soak.hot ]; then echo "$hot" > $O/soak.hot; xlog "temperature guard: VIRHE$hot for $n samples in a row (junction >= $TJMAX C or memory >= $TMMAX C) -> no new cycle"; fi
    sleep $TELS
  done
}
telstart() {   # the sampler on (once); the mock's fake tree first
  [ "$DRY" = 1 ] && { echo "+ telemon & -- every $TELS s each card under $SYSD (chip UUID, read-only sysfs: power1_average, temp1/2/3_input, freq1_input, pp_dpm_sclk, fan1_input) -> $O/telemetry.csv; guard junction >= $TJMAX C / memory >= $TMMAX C x $TNHOT -> no new cycle | cards now: $( [ "$SYSD" = FAKE ] && echo "the mock's fake tree" || for d in $SYSD/card*/device; do [ -e $d/unique_id ] && echo -n "GPU-$(cat $d/unique_id)@$(basename $(readlink -f $d)) "; done)"; return 0; }
  [ -z "$TELPID" ] || return 0
  if [ "$SYSD" = FAKE ]; then SYSD=$O/fakesys; mkfakesys $SYSD; fi
  rm -f $O/soak.tstop $O/soak.hot; [ -s $O/telemetry.csv ] || echo "$TELHDR" > $O/telemetry.csv
  telemon & TELPID=$!
  xlog "telemetry: every $TELS s, $(ls -d $SYSD/card*/device 2>/dev/null | while read -r d; do [ -e $d/unique_id ] && echo "GPU-$(cat $d/unique_id)@$(basename $(readlink -f $d))"; done | tr '\n' ' ')(chip UUID @ PCI; read-only sysfs$([ $SW = 0 ] && echo ", the mock's FAKE tree")) -> $O/telemetry.csv | guard junction >= $TJMAX C or memory >= $TMMAX C for $TNHOT samples -> no new cycle"
}
telstop() { [ "$DRY" = 1 ] && return 0; touch $O/soak.tstop 2>/dev/null; [ -n "$TELPID" ] && { kill $TELPID 2>/dev/null; wait $TELPID 2>/dev/null; }; TELPID=""; return 0; }
telnow() {   # telnow T0 T1 -> one line: each card over the window (soaksum.py tel)
  [ "$DRY" = 1 ] && { echo "(dry: soaksum.py tel over the cycle)"; return 0; }
  timeout 60 python3 -I -B $YDS tel --csv $O/telemetry.csv --t0 $1 --t1 $2 2>&1 | grep -m1 '^TEL ' | cut -c5-1400
}
pinhold() {   # pinhold TAG -> "held (..)" | "VIRHE .." | "INFO (mock) ..": READ ONLY (no POST, no epoch bump) -- GET /admin/fn4 every switch of THE pin
              # (pinj O) at its value, nothing pending; GET /stats the 4 ranks place_swl 1, heat_min_gain_x100 SWLG, heat_hyst_x100 125, place_lf 0
  [ "$DRY" = 1 ] && { echo "held (dry: GET /admin/fn4 every switch of pinj O + nothing pending; GET /stats 4 ranks place_swl 1 / heat_min_gain_x100 $SWLG / heat_hyst_x100 125 / place_lf 0; read only)"; return 0; }
  local r
  ssnap $1-pin
  r=$(python3 -c 'import json,sys
def rd(p):
    try: return json.load(open(p))
    except Exception: return None
f, s, w, g = rd(sys.argv[1]), rd(sys.argv[2]), json.loads(sys.argv[3]), int(sys.argv[4])
bad = []
sw = (f or {}).get("switches") or {}
if not f: bad.append("no GET /admin/fn4")
else:
    bad += ["%s %s (want %s)" % (k, sw.get(k), v) for k, v in w.items() if sw.get(k) != v]
    if f.get("pending"): bad.append("pending %s" % f.get("pending"))
rk = ((s or {}).get("experts") or {}).get("ranks") or []
if len(rk) != 4: bad.append("%d ranks in /stats (want 4)" % len(rk))
for r in rk:
    for k, v in (("place_swl", 1), ("heat_min_gain_x100", g), ("heat_hyst_x100", 125), ("place_lf", 0)):
        if r.get(k) != v: bad.append("rank %s %s %s (want %s)" % (r.get("rank"), k, r.get(k), v))
print(("OK all %d switches of the pin, nothing pending, epoch %s, 4 ranks SWL" % (len(w), (f or {}).get("epoch"))) if not bad else ", ".join(bad[:10]))' $O/fn-$1-pin.json $O/st-$1-pin.json "$(pinj O)" $SWLG 2>&1 | head -1 | cut -c1-700)
  case "$r" in OK*) echo "held (${r#OK })" ;; *) if [ $SW = 0 ]; then echo "INFO (ZS16_SW=0 mock: the int14 switches are not posted) $r"; else echo "VIRHE $r"; fi ;; esac
}
alone0() {   # alone0 NAME: THE pin held (read only, NOT re-posted), prefix-cache reset, greedyset.py's 20 prompts at c1 ALONE, T 0, MT tok -> $O/xzs16-NAME.jsonl
  local t0=$(date +%s) n h
  h=$(pinhold xzs16-$1-pre)
  pcreset xzs16-$1
  ssnap xzs16-$1-0
  run "(cd $D && timeout 900 python3 $GS $URL $O/xzs16-$1.jsonl --conc 1 --max-tokens $MT > $O/xzs16-$1.txt 2>&1)"
  ssnap xzs16-$1-1
  [ "$DRY" = 1 ] && { echo "+ RESULT xzs16 alone $1: greedy-20 c1 ALONE, T 0, $MT tok, THE $PINNAME PIN (pin $h): 20 of 20 answered (else VIRHE)"; return 0; }
  case "$h" in VIRHE*) PINDRIFT="$PINDRIFT $1" ;; esac
  n=$(python3 -c 'import json,sys; print(sum(1 for l in open(sys.argv[1]) if l.strip() and json.loads(l).get("text") is not None))' $O/xzs16-$1.jsonl 2>/dev/null || echo 0)
  xlog "alone $1 (greedy-20 c1 ALONE, T 0, $MT tok, THE $PINNAME PIN, not re-posted): $n of 20 answered$([ "$n" = 20 ] || echo " -- VIRHE request errors"), $(tail -1 $O/xzs16-$1.txt | cut -c1-160), $(( $(date +%s) - t0 )) s | pin $([ "${h%% *}" = VIRHE ] && echo "VIRHE ${h#VIRHE }" || echo "$h")"
}
scmp() {   # scmp NAME A B KIND: loadid.py cmp (byte for byte + the first differing char / token) -> $O/xzs16-cmp-NAME.json + identity.txt + one line;
           # KIND same (this start: a difference = VIRHE) | xrun (another run's file: VIRHE; INFO under ZS16_SW=0). NEVER a stop.
  local r
  SCMP=""
  [ "$DRY" = 1 ] && { echo "+ python3 -I $LD cmp --url $URL --json $O/xzs16-cmp-$1.json $2 $3   ($4; $([ -s "$2" ] && echo "A present" || echo "A MISSING now"))"; SCMP="IDENTICAL (dry)"; return 0; }
  local lab; lab=$([ $4 = xrun ] && [ $SW = 0 ] && echo "INFO (ZS16_SW=0 mock: a real-run reference)" || echo VIRHE)
  if [ ! -s "$2" ] || [ ! -s "$3" ]; then SCMP="not compared"; echo "xzs16-$1 not compared" >> $O/identity.txt; xlog "identity $1: $lab not compared (missing:$([ -s "$2" ] || echo " $2")$([ -s "$3" ] || echo " $3"))"; return 0; fi
  r=$(timeout 300 python3 -I $LD cmp --url $URL --json $O/xzs16-cmp-$1.json --label "xzs16 $1" "$2" "$3" 2>&1 | grep -m1 '^CMPLINE ' | cut -c9-)
  SCMP=${r:-compare failed}
  echo "xzs16-$1 $SCMP" >> $O/identity.txt
  case "$r" in IDENTICAL*) xlog "identity $1 ($(basename $2) vs $(basename $3)): $r" ;;
    *) xlog "identity $1 ($(basename $2) vs $(basename $3)): $lab $SCMP -- recorded, the soak continues" ;; esac
}
mockfault() {   # mockfault set|clear: the mock's "differ" fault (ZS16_SW=0 + ZS16_MOCKDIFF only; never reached by the real item)
  [ $SW = 0 ] || return 0
  case $1 in set) run "curl -s -m 10 -o /dev/null -X POST -H 'Content-Type: application/json' -d '{\"set\": {\"differ\": \"ds41_psk\"}}' $URL/mock/fault" ;;
    clear) run "curl -s -m 10 -o /dev/null -X POST -H 'Content-Type: application/json' -d '{\"clear\": [\"differ\"]}' $URL/mock/fault" ;; esac
  xlog "MOCK: the mock's differ fault $1 (ZS16_MOCKDIFF=$MOCKDIFF: this cycle's answers must differ, be recorded, and not stop the soak)"
}
soakcyc() {   # soakcyc I WARM: ONE cycle -- pinhold (read only), prefix-cache reset, loadid.py run (c SCONC load, WARM s, then the 20 probes each with a
              # fresh SPF-token prefill SOFF s in), strict compare vs a1, the record (soak-cycles.jsonl), one line
  local i=$1 w=$2 nm=c$1 t0=$(date +%s) u0 u1 rc=0 h hok=1 ld en tl
  h=$(pinhold xzs16-$nm-pre)
  case "$h" in held*|INFO*) ;; *) hok=0; PINDRIFT="$PINDRIFT $nm" ;; esac
  pcreset xzs16-$nm
  ssnap xzs16-$nm-0
  [ "$MOCKDIFF" = "$i" ] && mockfault set
  u0=$(utcnow)
  run "(cd $D && timeout 1800 python3 -I $LD run --url $URL --gs $GS --mb $MBP --out $O/xzs16-$nm.jsonl --conc $SCONC --warm $w --offset $SOFF --pf-tok $SPF --max-tokens $MT > $O/xzs16-$nm.txt 2>&1)" || rc=$?
  u1=$(utcnow)
  [ "$MOCKDIFF" = "$i" ] && mockfault clear
  ssnap xzs16-$nm-1
  scmp $nm-vs-a1 $O/xzs16-a1.jsonl $O/xzs16-$nm.jsonl same
  [ "$DRY" = 1 ] || cp -f $O/xzs16-cmp-$nm-vs-a1.json $O/xzs16-cmp-$nm.json 2>/dev/null
  if [ "$DRY" = 1 ]; then echo "+ RESULT xzs16 SOAK cycle $i: text vs a1 (strict; a difference = VIRHE, recorded, the soak continues) | pin held (read only) | load (LOADID) | engagement (soaksum.py engage --recorded 0 --prefills <the cycle's prefilled requests>: form 2 must run, form 1 <= 80 a prefilled request, else VIRHE) | telemetry a card | -> $O/soak-cycles.jsonl"; return 0; fi
  ld=$(grep -m1 '^LOADID ' $O/xzs16-$nm.txt | cut -c8- | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); w=d.get("window") or {}
print("%s/%s answered, c%s-load %s requests / %s errors, prefills cold %s of %s (%s tok), pf TTFT mean %s s, probe in flight median %s %%, mixed steps in probes min %s, probe phase ~%.0f s, %s decode tok (ms/decode step %s, acceptance %s)" % (d.get("answered"), d.get("of"), d.get("conc"), d.get("c4_requests_done"), d.get("c4_errors"), d.get("pf_cold"), d.get("of"), d.get("pf_prompt_mean"), d.get("pf_ttft_mean"), round(100 * (d.get("ov_frac_median") or 0)), d.get("mixed_min"), (d.get("window_s") or 0) - float(sys.argv[1]), w.get("dec_tok"), w.get("ms_decode_step"), w.get("acceptance")))' $w 2>&1 | cut -c1-500)
  npf=$(grep -m1 '^LOADID ' $O/xzs16-$nm.txt | cut -c8- | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print((d.get("c4_requests_done") or 0) + (d.get("answered") or 0) + (d.get("pf_cold") or 0))' 2>/dev/null || echo 0)
  en=$(timeout 60 python3 -I -B $YDS engage --fn0 $O/fn-xzs16-$nm-0.json --fn1 $O/fn-xzs16-$nm-1.json --pin $PINNAME --xb $XB --recorded 0 --prefills ${npf:-0} 2>&1 | grep -m1 '^ENGAGE ' | cut -c8- | cut -c1-900)
  eok=1; case "$en" in OK*) ;; *) if [ $SW = 0 ]; then en="INFO (ZS16_SW=0 mock: no fuse / rco counters) ${en#VIRHE }"; else eok=0; fi ;; esac
  python3 -c 'import json,sys; print(json.dumps(dict(cycle=int(sys.argv[1]), u0=float(sys.argv[2]), u1=float(sys.argv[3]), warm=int(sys.argv[4]), rc=int(sys.argv[5]), pin_ok=int(sys.argv[6]), pin_txt=sys.argv[7], cmp=sys.argv[8], eng_ok=int(sys.argv[9]), eng=sys.argv[10], prefills=int(sys.argv[11]))))' \
    $i $u0 $u1 $w $rc $hok "$h" "$SCMP" $eok "$en" ${npf:-0} >> $O/soak-cycles.jsonl
  tl=$(telnow ${u0%.*} ${u1%.*})
  xlog "SOAK cycle $i of $SOAKN (THE $PINNAME PIN): text vs a1 $(case "$SCMP" in IDENTICAL*) echo "$SCMP" ;; *) echo "VIRHE $SCMP" ;; esac) | pin $( [ $hok = 1 ] && echo "$h" || echo "VIRHE ${h#VIRHE }") | load: ${ld:-no LOADID line (rc $rc)} | engagement (soaksum.py fuse_check: form 2 must run, form 1 <= 80 a prefilled request): $([ $eok = 1 ] && echo "$en" || echo "VIRHE ${en#VIRHE }") | cards: $tl | warm $w s, cycle $(( $(date +%s) - t0 )) s, elapsed $(( $(elsec) / 60 )) min"
}
soakreads() {   # the readings (module header): telemetry on; a1 + its cross-run compares; the cycles; a2 + the control compare; the end engagement
  local i w=$SWARM0 t0 dur est=$CYCS nw lim=$(( BUDGET * 60 - CLOSES )) ran=0
  telstart
  alone0 a1
  scmp a1-vs-ref12-a1 $REF12A1 $O/xzs16-a1.jsonl xrun
  if [ $PINNAME = int16 ]; then ZREF=$ZC16DIR/xzc16-a1-F.jsonl
  elif [ -s $ZC16DIR/xzc16-a1-B.jsonl ]; then ZREF=$ZC16DIR/xzc16-a1-B.jsonl
  else ZREF=$ZFA1; fi
  scmp a1-vs-zzzc16 $ZREF $O/xzs16-a1.jsonl xrun
  [ "$DRY" = 1 ] && echo "+ cycles: up to $SOAKN (>= $SOAKMIN wanted), ~$CYCS s each (warm $SWARM0 s first, then adapted $WMIN..$WMAX s), load c$SCONC + ${SPF}-token prefill $SOFF s into each probe; a cycle starts only if elapsed + the last cycle + $CLOSES s <= $BUDGET min; stop file / temperature guard stop new cycles"
  for i in $(seq 1 $SOAKN); do
    stopfile && { xlog "cycles: the queue stop file -> no cycle $i"; break; }
    [ -e $O/soak.hot ] && { xlog "cycles: VIRHE the temperature guard fired ($(cat $O/soak.hot)) -> no cycle $i"; break; }
    if [ "$DRY" != 1 ] && [ $(( $(elsec) + est )) -gt $lim ]; then xlog "cycle $i SKIPPED (budget): $(( $(elsec) / 60 )) min elapsed + ~$(( est / 60 )) min for a cycle + $(( CLOSES / 60 )) min close > $BUDGET min"; break; fi
    t0=$(date +%s)
    soakcyc $i $w
    ran=$i
    dur=$(( $(date +%s) - t0 )); est=$dur
    nw=$(( CYCS - (dur - w) )); [ $nw -lt $WMIN ] && nw=$WMIN; [ $nw -gt $WMAX ] && nw=$WMAX; w=$nw
  done
  alone0 a2
  scmp a2-vs-a1 $O/xzs16-a1.jsonl $O/xzs16-a2.jsonl same
  pin16 xzs16-end "after the soak" || true
  telstop
  xlog "soak: $ran of $SOAKN cycles ran$([ $ran -lt $SOAKMIN ] && echo " (NOTE: fewer than $SOAKMIN)"), pin $([ -z "$PINDRIFT" ] && echo "held at every check" || echo "VIRHE NOT held at:$PINDRIFT"), $(( $(elsec) / 60 )) min elapsed"
}
soaksummary() {   # soaksum.py (every SOAK16 line), then the raw snapshots into one tar.gz (removed) and the logs gzipped
  local EL=$O/xzs16-n5-engine-full.log.gz l
  if [ "$DRY" = 1 ]; then
    echo "+ python3 -I -B $YDS summary $O --cycles $O/soak-cycles.jsonl --tel $O/telemetry.csv --log $EL --pin $PINNAME --pickwhy '..' --sw $SW --min-cycles $SOAKMIN --templog-dir $TEMPLOG --engage $O/engage.txt --json $O/soaksum.json > $O/soaksum.txt -> RESULT xzs16 SOAK16 lines (PIN / ALONE / CYCLE i / CONTROL / TOTALS / TELEMETRY a card / ENGAGEMENT / VERDICT)"
    echo "+ tar -czf $O/raw-snaps.tar.gz (st-*.json fn-*.json m-*.txt) and remove them; gzip start-*.log xzs16-c*.txt xzs16-a*.txt telemetry.csv; rm -rf fakesys"
    return 0
  fi
  [ -s $EL ] || xlog "summary: NOTE no engine log $EL (the start did not come up): no step-log readings"
  timeout 300 python3 -I -B $YDS summary $O --cycles $O/soak-cycles.jsonl --tel $O/telemetry.csv --log $EL --pin $PINNAME --pickwhy "$PICKWHY" --sw $SW --min-cycles $SOAKMIN --templog-dir $TEMPLOG --engage $O/engage.txt --json $O/soaksum.json > $O/soaksum.txt 2>&1
  while IFS= read -r l; do xlog "${l:0:3000}"; done < <(grep -h '^SOAK16' $O/soaksum.txt 2>/dev/null)
  grep -q '^SOAK16' $O/soaksum.txt 2>/dev/null || xlog "summary: VIRHE soaksum.py printed no SOAK16 line ($(tail -3 $O/soaksum.txt 2>/dev/null | tr '\n' ' ' | cut -c1-300))"
  grep -q '^Traceback' $O/soaksum.txt 2>/dev/null && xlog "summary: VIRHE a Traceback in soaksum.py ($(grep -A4 '^Traceback' $O/soaksum.txt | tail -2 | tr '\n' ' ' | cut -c1-300))"
  ( cd $O && tar -czf raw-snaps.tar.gz $(ls st-*.json fn-*.json m-*.txt 2>/dev/null) 2>/dev/null && rm -f st-*.json fn-*.json m-*.txt ) || xlog "summary: NOTE the raw snapshots were not archived (left as they are)"
  gzip -f $O/start-*.log $O/xzs16-c*.txt $O/xzs16-a*.txt $O/telemetry.csv 2>/dev/null || true
  rm -rf $O/fakesys
  xlog "summary: raw snapshots -> raw-snaps.tar.gz ($(du -m $O/raw-snaps.tar.gz 2>/dev/null | cut -f1) MB); logs gzipped; $O $(du -sm $O 2>/dev/null | cut -f1) MB; $(dfree) MB free on /"
  return 0
}
'''
k = s.index("dstart() {   # the tables")
s = s[:k] + SOAK + s[k:]

# ---------------------------------------------------------------------------------------------- 6 dstart: capture from the start, the soak, no end dump
rep("  xlog \"$tag: start (THE int16 CONFIRMATION; ngshare:", "  xlog \"$tag: start (THE SOAK ON THE $PINNAME PIN; ngshare:")
rep("""  gate1 $tag 70 || { xlog "$tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\\n' ' ' | cut -c1-500)"; down; return 1; }
""", """  gate1 $tag 70 || { xlog "$tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\\n' ' ' | cut -c1-500)"; down; return 1; }
  capstart $tag   # zsoak16: the engine log followed from here (it holds the whole start log too)
""")
a = s.index('  xlog "THE $PINNAME PIN O (read back, fn4 + every rank + engagement): $(pinj O | cut -c1-1400) | arms B (int15)')
b = s.index("\n", a)
s = s[:a] + '''  xlog "THE $PINNAME PIN O (read back, fn4 + every rank + engagement): $(pinj O | cut -c1-1400) | readings: telemetry on; a1 alone (pin read only) + strict cross-run vs ref12 a1 / zzzc16's; up to $SOAKN cycles of c$SCONC load + 20 probes each with a fresh ${SPF}-token prefill $SOFF s in (strict vs a1, never a stop); a2 alone + the end engagement | budget $BUDGET min"''' + s[b:]
rep("  [ \"$DRY\" = 1 ] || { : > $O/blocks.jsonl; : > $O/mb.jsonl; : > $O/c16cap.jsonl; }",
    "  [ \"$DRY\" = 1 ] || { : > $O/blocks.jsonl; : > $O/mb.jsonl; : > $O/c16cap.jsonl; : > $O/soak-cycles.jsonl; }")
rep("  stopfile || c16reads\n  capture $O/$tag-engine-full.log.gz\n",
    "  stopfile || soakreads\n  :   # zsoak16: no end dump -- capstart's follow capture holds the whole engine log (closed after down by capwait)\n")
rep("  down   # the stamp writer flushes at exit\n  [ \"$DRY\" = 1 ] || sleep 3\n",
    "  down   # the stamp writer flushes at exit\n  [ \"$DRY\" = 1 ] || sleep 3\n  capwait\n")
rep("trap 'down; [ $NGB = 1 ] && [ $NGCLEAN = 0 ] && ngclean' EXIT", "trap 'telstop; down; capwait; [ $NGB = 1 ] && [ $NGCLEAN = 0 ] && ngclean' EXIT")

# ---------------------------------------------------------------------------------------------- 7 main: the start line, the pin line, checks, summary
a = s.index('xlog "$ITEM: start IMAGE=$IMAGE (')
b = s.index("\n", a)
s = s[:a] + '''xlog "$ITEM: start IMAGE=$IMAGE ($(docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null | cut -c8-19))$([ $SW = 1 ] || echo " | MOCK-RUN ZS16_SW=0 (NOT A READING): NO int9b .. int16 switch posted, the pin checks INFO, cross-run references INFO, fake sysfs") | THE DS SOAK (exact under load over a long window): ONE FRESH START on ngshare (no ATTACH) = zzzc16's (RADIANCE_DS41_RFUSE=$RFDECL declared), THE $PINNAME PIN posted right after it and read back (fn4 every switch, every rank, the declare lines, the engagement probe) | MAXSEQS ${ZS16_MAXSEQS:-the base value} | tables: zzzc16's A4 $A4MD5 ($A4N lines), A11 $A11MD5 ($A11NE entries), NOT re-benched | a1 alone, up to $SOAKN cycles (>= $SOAKMIN wanted) of mb c$SCONC + the 20 greedy probes with a fresh ${SPF}-token prefill $SOFF s in, ~$CYCS s a cycle, strict vs a1 (recorded, never a stop), a2 control, telemetry every $TELS s a card | budget $BUDGET min | no evict"
xlog "SOAK PIN = $PINNAME ($PICKWHY) | posted: ds41_rfuse $FRF, ds41_mix_fin $FMF, route_copy_ov $FRC (+ the int15 pin: exl3_x3d $XB ...) | auto-choice over $PICKLOG (soaksum.py pick: the last xzc16 run; any identity line not IDENTICAL or its STOP line -> int15; missing -> int16)"''' + s[b:]
rep("FNS=\"stagx tabs14 rankrb", "FNS=\"elsec capstart capwait mkfakesys telemon telstart telstop telnow pinhold alone0 scmp mockfault soakcyc soakreads soaksummary stagx tabs14 rankrb")
rep('HMD5="zzzc16/zc16sum.py $(md5sum < $YDS | cut -c1-8) (new: the int16 confirmation summary, zzzfuse/fusesum.py\'s step-log readers), zzzc16/steps.py',
    'HMD5="zsoak16/soaksum.py $(md5sum < $YDS | cut -c1-8) (new: the soak summary, the pin auto-choice, the engagement rule), zzzc16/steps.py')
rep("dsm-zzzc16.sh $(md5sum < $0 | cut -c1-8) (a copy of zzzfuse/dsm-zzzfuse.sh 2a0f352d via zzzc16/mkbody16.py)",
    "dsm-zsoak16.sh $(md5sum < $0 | cut -c1-8) (a copy of zzzc16/dsm-zzzc16.sh 00cfc8b8 via zsoak16/mkbodys16.py $(md5sum < $HD/mkbodys16.py 2>/dev/null | cut -c1-8))")
a = s.index('  echo "+ readings: 1 identity under F (THE $PINNAME PIN): warm-up vs stock,')
b = s.index("\n", a)
s = s[:a] + '''  echo "+ SOAK PIN = $PINNAME ($PICKWHY) -> ds41_rfuse $FRF, ds41_mix_fin $FMF, route_copy_ov $FRC | readings: a1 alone vs ref12 a1 ($([ -s $REF12A1 ] && echo present || echo MISSING)) + vs zzzc16's a1-F ($([ -s $ZC16DIR/xzc16-a1-F.jsonl ] && echo present || echo MISSING)) / a1-B (zzzc16's $([ -s $ZC16DIR/xzc16-a1-B.jsonl ] && echo present || echo absent), zzzfuse's $([ -s $ZFA1 ] && echo present || echo MISSING)) | up to $SOAKN cycles of mb c$SCONC + 20 probes + fresh ${SPF}-token prefills, strict vs a1 | a2 control | engagement start + end | budget $BUDGET min | four cards $(cat /sys/class/drm/card*/device/unique_id 2>/dev/null | wc -l) unique_id | n-gram tables in $NGDIR now: $(ls $NGDIR/ngram-*.tab 2>/dev/null | wc -l) | free RAM $(free -g | awk 'NR==2{print $7}') GiB | /tmp inodes $(df --output=ipcent /tmp 2>/dev/null | tail -1 | tr -d ' '), / inodes $(df --output=ipcent / 2>/dev/null | tail -1 | tr -d ' ')"''' + s[b:]
rep("\nc16summary\n", "\nsoaksummary\n")
rep('xlog "$ITEM: done ($(elapsed) min) -- $O | $([ $STOPPED = 1 ] && echo "STOPPED at the identity gate (no speed block ran): NEEDS USER | ")',
    'xlog "$ITEM: done ($(elapsed) min) -- $O | $(grep -h "^SOAK16 VERDICT" $O/soaksum.txt 2>/dev/null | cut -c8-200) | ')

# ---------------------------------------------------------------------------------------------- 8 the SOAK PIN line after the output dir exists
a = s.index('\nxlog "SOAK PIN = $PINNAME ($PICKWHY) | posted:')
b = s.index("\n", a + 1)
pinline = s[a + 1:b]
s = s[:a] + s[b:]
rep("[ \"$DRY\" = 1 ] || : > $O/identity.txt\nif ! stopfile; then\n",
    "[ \"$DRY\" = 1 ] || : > $O/identity.txt\n[ \"$DRY\" = 1 ] || : > $O/engage.txt\n" + pinline + "\nif ! stopfile; then\n")

# ---------------------------------------------------------------------------------------------- 1 the header (the module comment), written last
i0 = s.index("# dsm-zzzc16.sh (13.11.2026")
i1 = s.index("\nset -u\n")
HEADER = r'''# dsm-zsoak16.sh (13.11.2026, queue-ar/dsmain/zsoak16, a builder for ds-main-port) -- NOT a queue item; the body of
# 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzzzzzzzz-zsoak16.sh. Log tag "RESULT xzs16 ", out runs/xdsm/zsoak16 (ZS16_OUT). A COPY of
# zzzc16/dsm-zzzc16.sh (md5 00cfc8b8, the int16 confirmation as staged + run 13.11.; tag xzc16 -> xzs16, knobs ZC16_ -> ZS16_; generated
# by zsoak16/mkbodys16.py, every edit an asserted replacement; zzzc16's body not changed; unused functions kept): the SAME start words,
# guards, image literal checks, n-gram reuse (tools/ngramshare/ngshare.sh), A4's table + A11's ranged entries (zzzc16's files, read in
# place, md5-checked, NOT re-benched), the same pin post + read-back (swnote, rankrb: every switch on GET /admin/fn4 + the 4 ranks on
# GET /stats: place_swl 1, heat_min_gain_x100 75, heat_hyst_x100 125, place_lf 0), the exl3_x3d line, the fusion start state
# (fusestart: RADIANCE_DS41_RFUSE=2 declared, refusals) and the engagement probe (pin16; its rule now in soaksum.py, see below).
# THE DS int16 SOAK (ds-main-port 13.11.; the Qwen line's S9 soak = FN item 231 queue/57-...-ref12zzzzzzzzzzzz-fn-s-s9soak.sh is the
#   model: cycles, load, probes, telemetry, alone control): EXACT UNDER LOAD over a LONG window for the production candidate.
# PIN AUTO-CHOICE at item start (soaksum.py pick over runs/ds41-exl3.log, the LAST "RESULT xzc16 <item>: start" run): any "RESULT
#   xzc16 identity" line that is not IDENTICAL, or its STOP line -> THE int15 PIN (ds41_rfuse 0, route_copy_ov 0, ds41_mix_fin 0, the
#   same image and start); every identity line IDENTICAL -> THE int16 PIN (ds41_rfuse 2, route_copy_ov 1, ds41_mix_fin 0); the
#   xzc16 result missing (no start line / no identity line) -> int16, logged. One line "SOAK PIN = int16" / "SOAK PIN = int15 (because ..)".
#   The start is the same in both cases (zzzc16's: RADIANCE_DS41_RFUSE=2 declared; its arm B = int15 ran on it); only the posted pin
#   differs. ZS16_PICKLOG (tests only) reads another log.
# ENGAGEMENT RULE (ds-main-port 13.11.): soaksum.py fuse_check + its form1_expected() hook. int16: mixfin 0, rco in force, over the
#   epoch-bumped probe also rco recordings / freed > 0; int15: no fused form moves. FORM 1 (zzzc16's c4 F blocks counted +18 / +19 beside
#   form 2 +1920..2360) = ds-main-port's rule, verified in code (rad_block_moe_fp8.h:2288-2290, bounds :2000 / :2019; deepseek41 kBiMoe 16): form 2 serves MoE
#   passes <= 48 rows, form 1 those of 49 .. 170, so form 1 comes only from a prefill segment; a form-1 delta is INFO when the span
#   prefilled >= 1 request and form 1 <= 80 x the prefilled requests, VIRHE in a decode-only span or above the bound; form 2 must run
#   in every int16 span (reference: zzzc16/zc16sum.py 7f241a37). Run at the start (the gate: a VIRHE = nothing is read), on every
#   cycle (a VIRHE makes the cycle not clean) and once more after the soak (logged, part of the verdict).
# NO P0 (ds-main-port 13.11. ~03:20: the rfuse M 49 .. 170 selftest is a separate item after the soak).
# READINGS, in order (one fresh start, MAXSEQS 16, ngshare, no ATTACH, compose guard in the item):
#   0 the start + read-back (zzzc16's; a mismatch anywhere = VIRHE and NOTHING is read); the engine log followed from the start
#     (docker logs -f -t | gzip, TESTING.md 13.11. lesson 4); the telemetry sampler on (every ZS16_TELS 5 s, each card by chip
#     UUID = GPU-<sysfs unique_id>: power W, junction / memory / edge C, sclk MHz, the sclk level, fan rpm -- READ-ONLY sysfs, nothing
#     set; a temperature guard: junction >= 100 C or memory >= 96 C on a card for 2 samples in a row -> no new cycle, VIRHE).
#   1 ALONE REFERENCE a1: greedyset.py's 20 prompts at c1 alone, T 0, 192 tok, after a prefix-cache reset; THE pin checked READ ONLY
#     (pinhold: every switch + nothing pending + the 4 ranks; never re-posted during the soak). Strict cross-run compares (logged, never
#     a stop): a1 vs ref12's a1, a1 vs zzzc16's a1-F (int16) / a1-B (int15: zzzc16's if it exists, else zzzfuse's a1-B, the one zzzc16
#     itself compared with).
#   2 CYCLES (ZS16_CYCLES 8, >= ZS16_CYCLES_MIN 6 wanted, ~ZS16_CYCLE_S 280 s each): pinhold (read only) + prefix-cache reset, then
#     stack4/loadid.py run --conc 12 (mb.py's closed-loop c12 load: fixed seeded texts behind a nonce, ctx 1000, gen 512, continuous for
#     the cycle) --warm W --offset 0.5 --pf-tok 8192: after W s of load the 20 greedy probes go one at a time, each joined 0.5 s in by a
#     fresh ~8k-token prefill (~c14-16 in flight, staggered; the probes' decode rows ride MIXED steps). W adapts so a cycle lasts ~280 s
#     (first cycle ZS16_WARM 60 s). Each cycle's answers vs a1 STRICTLY, byte for byte (loadid.py cmp: the first differing char /
#     token): a difference = VIRHE, RECORDED, and the soak CONTINUES (the rate). A cycle starts only if the elapsed time + the last
#     cycle + ~2.5 min close stays inside ZS16_BUDGET_MIN; the queue stop file and the temperature guard also stop new cycles.
#     Per cycle: the ENGAGEMENT RULE over the cycle's span (form 2 + rgs must move; form 1 <= 80 x the cycle's prefilled requests =
#     LOADID's load requests + 20 probes + 20 prefills, else VIRHE; rco recordings INFO: no epoch bump in the soak); INFO: agg decode
#     tok/s over the probe phase, the step-log decode p50, telemetry min / max a card.
#   3 CLOSE: a2 = the greedy-20 at c1 alone again, strict vs a1 (the control); then the end engagement probe (arm O + epoch bump).
# SUMMARY (zsoak16/soaksum.py, lines "SOAK16 ..."): PIN, ALONE, one CYCLE line each (n of 20 identical, first diff), CONTROL, TOTALS
#   (cycles x 20 probes, differences), TELEMETRY min / max a card (+ the ~/gpu/temps templog cross-check), ENGAGEMENT, VERDICT "EXACT
#   UNDER LOAD over N cycles / X min: YES / NO". Logs gzipped, the raw snapshots archived into raw-snaps.tar.gz and removed.
# ~48 min (start ~5.5 with the n-gram fill, read-back + a1 ~1, 8 cycles ~37.5, close + summary ~2); <= 50 min GPU time. NO evict, NO
#   build, no P0, no cc_sim, no power / host / fan settings; production compose files read only; nothing profiled, nothing stamped.
# MOCK: ZS16_SW=0 = a server without the int9b .. int16 switches (queue-ar/tools/mockrad; zsoak16/mockrun-zsoak16.sh): the pin checks
#   are INFO there, the telemetry reads a fake sysfs tree under the output dir (never the real cards); ZS16_MOCKDIFF=K (ZS16_SW=0 only)
#   sets the mock's "differ" fault for cycle K (with --no-replay) so a differing cycle must be recorded and must not stop the soak.
#   DRYRUN=1 bash dsm-zsoak16.sh : prints every command; checks the image literals, the K25U plugin md5, every helper (+ selftests),
#   the tables' and steps.py's md5s, the ngramshare files, the sidecars, the yamls, the references, the disk; no server, no GPU.'''
s = s[:i0] + HEADER + s[i1:]

# the inserted text names zzzc16's files and log tag on purpose; nothing else may still say xzc16 / ZC16_
left = [l for l in s.splitlines() if "xzc16" in l and not any(x in l for x in ("xzc16-a1-F.jsonl", "xzc16-a1-B.jsonl", "RESULT xzc16", "\"RESULT xzc16", "xzc16 identity", "xzc16 run", "xzc16 result", "tag xzc16 ->", "xzc16 <item>", "the last xzc16"))]
assert not left, left[:5]
assert "ZC16_" not in s.replace("tag xzc16 -> xzs16, knobs ZC16_ -> ZS16_", ""), [l for l in s.splitlines() if "ZC16_" in l][:3]
open(DST, "w").write(s)
print("mkbodys16: %s -> %s (%d lines, md5 %s; source md5 %s; xzc16 %d / ZC16_ %d renamed)" % (SRC, DST, s.count("\n"), hashlib.md5(s.encode()).hexdigest()[:8], m[:8], n_x, n_k))
