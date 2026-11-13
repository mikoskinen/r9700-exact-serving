# ---------------------------------------------------------------- zepb2 (this item's own functions; the zepb ones above are kept, some unused)
rescheck() {   # rescheck TAG STRICT: the RESIDENT UNITS at start vs prof1's int12 start (same pins, MAXSEQS 16): per rank GET /stats
               # experts.ranks[] vram / trunk_vram / flex_bytes and the start log's "mover: N slab slots"; STRICT 1 (start 1, no extras
               # declared): a rank's vram units or flex bytes lower by > 1 %, or its slab slots off by > 1 %, = VIRHE; STRICT 0: printed only
  [ "$DRY" = 1 ] && { echo "+ rescheck $1: GET /stats experts.ranks[] vram / trunk_vram / flex_bytes ($O/$1-stats0.json) + 'mover: N slab slots' a rank vs prof1 ($P1STATS, $P1DIR/xp1-n5-engine-full.log.gz); $([ $2 = 1 ] && echo "VIRHE if lower by > 1 %" || echo "printed only")"; return 0; }
  local r s0 s1
  s1=$(docker logs $CN 2>&1 | grep -oE '^I\[[0-3]\] mover: [0-9]+ slab slots' | sed -E 's/^I\[([0-3])\] mover: ([0-9]+) .*/\1:\2/' | sort -u | tr '\n' ' ')
  s0=$(zcat $P1DIR/xp1-n5-engine-full.log.gz 2>/dev/null | grep -oE 'I\[[0-3]\] mover: [0-9]+ slab slots' | sed -E 's/^I\[([0-3])\] mover: ([0-9]+) .*/\1:\2/' | sort -u | tr '\n' ' ')
  r=$(python3 -c 'import json,sys
def rk(p):
    try: return {x["rank"]: x for x in json.load(open(p))["experts"]["ranks"]}
    except Exception: return {}
def sl(s):
    d = {}
    for t in s.split():
        a, b = t.split(":"); d[int(a)] = int(b)
    return d
a, b, sa, sb = rk(sys.argv[1]), rk(sys.argv[2]), sl(sys.argv[3]), sl(sys.argv[4]); strict = sys.argv[5] == "1"
pc = lambda x, y: ("%+.2f %%" % (100.0 * (x - y) / y)) if x is not None and y else "-"
bad, parts = [], []
for r in sorted(set(a) | set(b)):
    x, y = a.get(r, {}), b.get(r, {})
    v, v0, f, f0, s, s0 = x.get("vram"), y.get("vram"), x.get("flex_bytes"), y.get("flex_bytes"), sa.get(r), sb.get(r)
    parts.append("rank %d: vram %s vs %s (%s), trunk_vram %s vs %s, flex %.2f vs %.2f GB (%s), slab %s vs %s (%s)" % (r, v, v0, pc(v, v0), x.get("trunk_vram"), y.get("trunk_vram"),
        (f or 0) / 1e9, (f0 or 0) / 1e9, pc(f, f0), s, s0, pc(s, s0)))
    if v is None or v0 is None or v < 0.99 * v0: bad.append("rank %d vram" % r)
    if f is not None and f0 and f < 0.99 * f0: bad.append("rank %d flex" % r)
    if s is not None and s0 and abs(s - s0) > 0.01 * s0: bad.append("rank %d slab" % r)
print(" | ".join(parts) + ((" -- %s %s" % ("VIRHE" if strict else "NOTE", ", ".join(bad))) if bad else " -- equal to prof1 within 1 %"))' $O/$1-stats0.json $P1STATS "$s1" "$s0" $2 2>&1 | cut -c1-1500)
  case "$r" in *VIRHE*) [ $SW = 1 ] || r="${r//VIRHE/INFO (XE2_SW=0 mock)}" ;; esac
  xlog "$1 RESIDENT UNITS at start vs prof1's int12 start (same pins, MAXSEQS 16): $r"
}
p0idx() {   # p0idx ARM: strict at this start -- ARM a1 = a2 (A2=1), gs16 = a1, gs8 = a1 (G8=1); O a1 = ref12's a1; ARM a1 != O a1 (the slot
            # path ran); XREF (a file: start 1's a1S) = ARM a1 (placement invariance, strict) -> 0 / 1 (1 stops this start's speed readings)
  local a=$1 k l bad=0 u0 u1 keys
  alone xep2 $a a1$a
  keys="gs${GSC}$a-vs-a1$a"
  [ ${A2:-1} = 1 ] && { alone xep2 $a a2$a; cmpx xep2 a2$a-vs-a1$a $O/xep2-a1$a.jsonl $O/xep2-a2$a.jsonl; keys="$keys a2$a-vs-a1$a"; }
  u0=$(utcnow); ssnap xep2-gs${GSC}$a-0; gsx $a gs${GSC}$a; ssnap xep2-gs${GSC}$a-1; u1=$(utcnow)
  [ "$DRY" = 1 ] || echo "$u0 $u1" > $O/gs${GSC}E.win
  cmpx xep2 gs${GSC}$a-vs-a1$a $O/xep2-a1$a.jsonl $O/xep2-gs${GSC}$a.jsonl
  if [ ${G8:-1} = 1 ]; then local gsc0=$GSC; GSC=8; gsx $a gs8$a; GSC=$gsc0; cmpx xep2 gs8$a-vs-a1$a $O/xep2-a1$a.jsonl $O/xep2-gs8$a.jsonl; keys="$keys gs8$a-vs-a1$a"; fi
  alone xep2 O a1O
  cmpx xep2 a1O-vs-ref12-a1 $R12A1 $O/xep2-a1O.jsonl; keys="$keys a1O-vs-ref12-a1"
  cmpx xep2 a1$a-vs-a1O-INFO $O/xep2-a1O.jsonl $O/xep2-a1$a.jsonl
  if [ -n "${XREF:-}" ]; then
    if [ -s "$XREF" ] || [ "$DRY" = 1 ]; then cmpx xep2 a1$a-vs-start1-a1S $XREF $O/xep2-a1$a.jsonl; keys="$keys a1$a-vs-start1-a1S"
    else xlog "identity a1$a vs start 1's a1S: $XREF missing -- not compared"; fi
  fi
  [ "$DRY" = 1 ] && { echo "+ RESULT xep2 P0b $a: [$keys] IDENTICAL (strict) and a1$a vs a1O DIFFERENT (the slot path ran), else VIRHE (this start's speed readings stop)"; return 0; }
  for k in $keys; do
    l=$(grep "^xep2-$k " $O/identity.txt 2>/dev/null | tail -1)
    case "$l" in *IDENTICAL*) ;; *) xlog "P0b $k: VIRHE ${l:-not compared} (strict)"; bad=1 ;; esac
  done
  l=$(grep "^xep2-a1$a-vs-a1O-INFO " $O/identity.txt 2>/dev/null | tail -1)
  case "$l" in *IDENTICAL*) [ $SW = 1 ] && { xlog "P0b a1$a vs a1O: VIRHE IDENTICAL -- the slot path did not run under $a"; bad=1; } ;; esac
  xlog "P0b gs${GSC}$a cap (all $GSC in flight, sched_row_cap $CAP): $(capd xep2-gs${GSC}$a-0 xep2-gs${GSC}$a-1)"
  [ $bad = 0 ] && { xlog "P0b $a: strict [$keys] IDENTICAL; $a differs from O (the slot path ran)"; return 0; }
  [ $SW = 1 ] && return 1
  xlog "P0b $a: INFO under XE2_SW=0 (a mock run) -- the mock goes on"; return 0
}
stagx() {   # stagx ARM: STAGGERED arrivals under ARM (stack4/loadid.py run, read only) vs ARM's a1 alone, strict (zepb's epbstag with the arm)
  local a=$1 t0=$(date +%s) l
  arm $a || return 1
  pcreset xep2-ld$a; ssnap xep2-ld$a-0
  run "(cd $D && timeout 1800 python3 -I $LD run --url $URL --gs $GS --mb $MBP --out $O/xep2-ld$a.jsonl --conc $LDCONC --warm 10 --offset 0.5 --pf-tok $LDPF --max-tokens $MT > $O/xep2-ld$a.txt 2>&1)"
  ssnap xep2-ld$a-1
  cmpx xep2 ld$a-vs-a1$a $O/xep2-a1$a.jsonl $O/xep2-ld$a.jsonl
  [ "$DRY" = 1 ] && { echo "+ RESULT xep2 staggered $a: LOADID (probes, MIXED steps during the probes, prefills cold) + ld$a vs a1$a strict (a difference = VIRHE)"; return 0; }
  xlog "staggered $a (loadid.py run, conc $LDCONC streams, fresh $LDPF-token prefill 0.5 s into each probe, T 0, $MT tok, $(( $(date +%s) - t0 )) s): $(grep -m1 '^LOADID ' $O/xep2-ld$a.txt | cut -c8-700) | probes in MIXED steps: $(python3 -c 'import json,sys
n = m = 0
for l in open(sys.argv[1]):
    try: d = json.loads(l)
    except ValueError: continue
    if "mixed" in d: n += 1; m += 1 if (d.get("mixed") or 0) > 0 else 0
print("%d of %d probes had MIXED steps in their life" % (m, n))' $O/xep2-ld$a.jsonl 2>&1 | cut -c1-200) | cap $(capd xep2-ld$a-0 xep2-ld$a-1)"
  l=$(grep "^xep2-ld$a-vs-a1$a " $O/identity.txt 2>/dev/null | tail -1)
  case "$l" in *IDENTICAL*) ;; *) [ $SW = 1 ] && xlog "staggered $a: VIRHE ${l:-not compared} (strict: a decode row riding MIXED steps must keep its alone bytes)" ;; esac
}
reads1() {   # START 1 (no extras): S's identity gate, c8 O S S O, c16 newcomer block O / S, 1u O / S, staggered S
  local a i k
  if budok "P0b S (a1, a2, gs$GSC, gs8; O a1)" 4; then
    A2=1 G8=1 XREF= p0idx S || { xlog "P0b S: VIRHE -> START 1's speed readings STOP (no reading on a non-invariant arm)"; STOPALL=1; return 1; }
  else STOPALL=1; return 1; fi
  i=1
  budok "streamdiag O, S (the zepb incident's two requests)" 2 && { sdiag O || true; sdiag S || true; }
  for a in $C8ORDER1; do stopfile && return 0; budok "c8 block $i ($a)" 2 && { mbg 8 $i $a || true; }; i=$((i + 1)); done
  i=1
  for a in $C16ARMS1; do stopfile && return 0; budok "c16 newcomer block $i ($a)" 2 && { mbg 16 $i $a || true; }; i=$((i + 1)); done
  for a in $U1ARMS1; do
    stopfile && return 0
    budok "1u essay $a (one discarded + $NREADS back to back)" 2 || continue
    blockx ${a}U1W $a essay || true
    for k in $(seq 1 $NREADS); do blockx ${a}U1 $a essay || true; done
  done
  stopfile && return 0
  budok "staggered arrivals (S)" 4 && { stagx S || true; }
  return 0
}
reads2() {   # START 2 (hot-4 extras): E4's identity gate (+ placement invariance vs start 1's S), c8 O E4 E4 O, 1u E4 / O
  local a i k
  if budok "P0b E4 (a1, gs$GSC; O a1; vs start 1's a1S)" 3; then
    A2=0 G8=0 XREF=$ODIR/s1/xep2-a1S.jsonl p0idx E4 || { xlog "P0b E4: VIRHE -> START 2's speed readings STOP"; return 1; }
  else return 1; fi
  budok "streamdiag E4" 1 && { sdiag E4 || true; }
  i=1
  for a in $C8ORDER2; do stopfile && return 0; budok "c8 block $i ($a)" 2 && { mbg 8 $i $a || true; }; i=$((i + 1)); done
  for a in $U1ARMS2; do
    stopfile && return 0
    budok "1u essay $a (one discarded + $NREADS back to back)" 2 || continue
    blockx ${a}U1W $a essay || true
    for k in $(seq 1 $NREADS); do blockx ${a}U1 $a essay || true; done
  done
  return 0
}
mbg() {   # mbg N I ARM: mbx N I ARM S + the BLOCK GUARD (the zepb c8-5-S-S incident 12.11.: two first-generation requests streamed no
          # text, the fixed schedule waited --max-wait for group 5's first token and the time window collapsed to 0 s): requests answered
          # > 0 and a window >= 90 % of its length, else VIRHE and the arm's remaining blocks SKIPPED (ARMDEAD); every request that
          # streamed no text (ttft null) is listed with the window_timeout either way
  local n=$1 i=$2 a=$3 nm=c$1-$2-S-$3 r
  case " $ARMDEAD " in *" $a "*) xlog "mb c$n $nm: SKIPPED -- arm $a failed its block guard earlier at this start"; return 1 ;; esac
  mbx $n $i $a S || true
  [ "$DRY" = 1 ] && { echo "+ RESULT xep2 block guard $nm: requests > 0 and window >= 90 % of $([ $n = 8 ] || [ $n = 16 ] && echo $C8WIN) s (else VIRHE and arm $a's remaining blocks SKIPPED); requests with no streamed text listed"; return 0; }
  r=$(python3 -c 'import json,sys
d = {}
for l in open(sys.argv[1]):
    if l.startswith("MB "): d = json.loads(l[3:]); break
nt = []
try:
    for l in open(sys.argv[2]):
        x = json.loads(l)
        if x.get("ev") == "done" and not x.get("err") and x.get("ttft") is None: nt.append("w%s k%s j%s max %s n %s" % (x.get("w"), x.get("k"), x.get("j"), x.get("max_tokens"), x.get("n")))
except OSError: pass
win, req, need = d.get("window_s") or 0.0, d.get("requests") or 0, 0.9 * float(sys.argv[3])
ok = req > 0 and win >= need
print("%s | requests %s, window %s s (need >= %.1f), window_timeout %s, requests with no streamed text: %s" % ("OK" if ok else "FAIL", req, win, need,
      d.get("window_timeout"), "; ".join(nt) or "none"))' $O/c8/$nm.txt $O/c8/$nm-reqs.jsonl $C8WIN 2>&1 | cut -c1-900)
  case "$r" in
    OK*) case "$r" in *"no streamed text: none"*) ;; *) xlog "block guard $nm: NOTE ${r#OK | }" ;; esac ;;
    *) if [ $SW = 1 ]; then xlog "block guard $nm: VIRHE ${r#FAIL | } -> arm $a's remaining blocks at this start SKIPPED"; ARMDEAD="$ARMDEAD $a"
       else xlog "block guard $nm: INFO (XE2_SW=0 mock) $r"; fi ;;
  esac
}
sdiag() {   # sdiag ARM: zepb2/streamdiag.py under ARM -- the two c8 first-generation requests that streamed no text in zepb's S block,
            # streamed and non-streamed (what the server sent; the reply text) -> DIAG / STREAMDIAG lines (INFORMATION, never a VIRHE)
  local a=$1
  arm $a || return 1
  run "(cd $D && timeout 600 python3 -I -B $SDG --mb $MBP --mbt3 $MBT2 --temperature 0 --pairs 0:192,4:128 --label $a --out $O/sdiag.jsonl > $O/sdiag-$a.txt 2>&1)"
  [ "$DRY" = 1 ] && { echo "+ RESULT xep2 streamdiag $a: DIAG w0 max 192 / w4 max 128 (stream events, events with text, first s; non-stream reasoning / content heads) + STREAMDIAG {empty_stream} (INFO)"; return 0; }
  while IFS= read -r l; do xlog "streamdiag $a (INFO; the zepb c8-5-S-S incident's two requests): ${l:0:900}"; done < <(grep -hE '^(DIAG|STREAMDIAG) ' $O/sdiag-$a.txt)
  grep -q '^STREAMDIAG ' $O/sdiag-$a.txt || xlog "streamdiag $a: NOTE no STREAMDIAG line ($(tail -2 $O/sdiag-$a.txt | tr '\n' ' ' | cut -c1-300))"
}
