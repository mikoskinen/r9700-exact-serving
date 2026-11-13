ncmset() {   # ncmset LOG LABEL MIN REQS...: ncmatch.py over the blocks (NAME = REQS minus -reqs.jsonl, ARM = its last field) -> ncm-LABEL.json / .txt (zzzfuse's)
  local el=$1 lb=$2 mn=$3 f nm bl="" l; shift 3
  for f in "$@"; do [ -s "$f" ] || continue; nm=$(basename $f -reqs.jsonl); bl="$bl --block $nm:${nm##*-}:$f"; done
  [ -n "$bl" ] || return 0
  timeout 300 python3 -I -B $NCM $el --set $lb $bl --min $mn --json $O/ncm-$lb.json > $O/ncm-$lb.txt 2>&1
  while IFS= read -r l; do xlog "${l:0:1600}"; done < <(grep -h '^NCMATCH' $O/ncm-$lb.txt 2>/dev/null)
  grep -q '^Traceback' $O/ncm-$lb.txt 2>/dev/null && xlog "summary: VIRHE ncmatch.py $lb: $(grep -A4 '^Traceback' $O/ncm-$lb.txt | tail -2 | tr '\n' ' ' | cut -c1-300)"
  return 0
}
stepsx() {   # stepsx PLAIN REQS: the coordinator's steps.py (zzzc16/steps.py = zzzfuse's, md5 STPMD5) over one gens block: every non-decode step and every
             # step with a prefill segment in [w0 - 2 s, w1 + 2 s] with its wall and in-window flag -> c8/NAME-steps.txt; logs its in-window non-decode walls (INFO; zzzfuse's)
  local nm w t0 t1 w0 w1
  nm=$(basename $2 -reqs.jsonl); w=$(mwin $2)
  [ -n "$w" ] || return 0
  read -r t0 t1 w0 w1 < <(python3 -c 'import sys
from datetime import datetime, timezone
a, b = float(sys.argv[1]), float(sys.argv[2])
iso = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
print(iso(a - 2.0), iso(b + 2.0), iso(a), iso(b))' ${w%%:*} ${w#*:})
  timeout 120 python3 -I -B $STP $1 $t0 $t1 $w0 $w1 > $O/c8/$nm-steps.txt 2>&1
  xlog "$nm steps.py (raw, INFO): $(awk '$NF == "True" && $3 != "decode" { n++; if ($(NF-1) != "None") s += $(NF-1) } END { printf "%d in-window non-decode step(s), walls %.1f ms", n, s }' $O/c8/$nm-steps.txt 2>/dev/null)$(grep -q '^Traceback' $O/c8/$nm-steps.txt 2>/dev/null && echo " -- VIRHE steps.py: $(tail -1 $O/c8/$nm-steps.txt | cut -c1-200)")"
}
kvprep() {   # kvprep: ONCE before the first start -- int14's tables (zzzc16's = ref14's, NOT re-benched) and zzzc16's start words (LEVENV + A4's
             # table, XPUSH's bound, FUSE-R declared at RFDECL, A11's ranged entries in the GPF table) -> 0, else VIRHE -> 1 (no start)
  tabs14 || return 1
  LEVENV="$LEVENV RADIANCE_DS41_GG_M16_FORMS=/stats/m16-forms.txt RADIANCE_FN4_AR_XPUSH_MIN=$XPMIN RADIANCE_DS41_RFUSE=$RFDECL"   # = zzzc16's dstart
  LEVENV=$(echo "$LEVENV" | sed "s#RADIANCE_DS41_GG_PF_TABLE=$GPTAB #RADIANCE_DS41_GG_PF_TABLE=$GPTAB,$A11ENT #")
  case "$LEVENV" in *"RADIANCE_DS41_GG_PF_TABLE=$GPTAB,$A11ENT "*) ;; *) xlog "kvprep: VIRHE the ranged entries did not go into LEVENV -> NOT STARTED"; return 1 ;; esac
  return 0
}
kvplace() {   # kvplace TAG ARM HP: THE PLACEMENT READ-BACK -- kvsum.py place over the start log (restart's start-TAG.log, written right after /health)
              # + /stats right after the pin -> one KVPLACE line; the knob not in force (--host-pool-mib != HP) or the split not moved = VIRHE -> 1
              # (no reading for this start); KVR_SW=0 (the mock serves a real 19456 start log): INFO -> 0
  local r
  ssnap $1-place
  if [ "$DRY" = 1 ]; then
    echo "+ python3 -I -B $SAS place $O/start-$1.log --arm $2 --hp $3 --stats $O/st-$1-place.json --ref $REFSTART --ref-hp $REFHP --sw $SW --json $O/place-$2.json -> RESULT xkvr KVPLACE $2: --host-pool-mib $3 in force, VRAM split (experts / kv cache / staging), owned / KV- / arena- / drafter-lendable slots a rank, KV tokens + sessions, deltas vs zzzc16's start (VIRHE: knob not in force / split not moved -> no reading)"
    return 0
  fi
  [ -s $O/start-$1.log ] || docker logs $CN > $O/start-$1.log 2>&1
  timeout 180 python3 -I -B $SAS place $O/start-$1.log --arm $2 --hp $3 --stats $O/st-$1-place.json --ref $REFSTART --ref-hp $REFHP --sw $SW --json $O/place-$2.json > $O/place-$2.txt 2>&1
  r=$(grep -m1 '^KVPLACE' $O/place-$2.txt 2>/dev/null)
  if [ -z "$r" ]; then
    xlog "$1 KVPLACE $2: VIRHE kvsum.py place printed no KVPLACE line ($(grep -A4 '^Traceback' $O/place-$2.txt 2>/dev/null | tail -2 | tr '\n' ' ' | cut -c1-300))"
    [ $SW = 0 ] && return 0; return 1
  fi
  xlog "${r:0:2800}"
  case "$r" in *VIRHE*) [ $SW = 0 ] && return 0; return 1 ;; esac
  return 0
}
kvreads() {   # kvreads TAG ARM: the readings of one start in their order (module header): 1 identity (warm-up vs stock, a1-F vs ref12's a1 + vs zzzc16's
              # a1-F: ANY difference -> VIRHE, this start's speed reads STOP), 2 c8 gens x C8B, 3 c16 gens x C16B, 4 1u essay (one discarded + U1N)
  local t=$1 a=$2 k nf
  nf=$(echo $IDFAIL | wc -w)
  idok $a   # identw's line "xkvr-ARM warm-up ..." (the warm-up answers vs stock)
  if budok "$a identity (a1-F c1 alone + 2 strict compares)" 1; then
    alone $t F a1-F || IDFAIL="$IDFAIL $a-a1-F(not-answered)"
    if [ -s "$REF12A1" ] || [ "$DRY" = 1 ]; then cmpx $t a1-F-vs-ref12-a1 $REF12A1 $O/$t-a1-F.jsonl; idok $a-a1-F-vs-ref12-a1
    else xlog "$a identity a1-F vs ref12 a1: VIRHE $REF12A1 missing -- not compared"; IDFAIL="$IDFAIL $a-a1-F-vs-ref12-a1(reference-missing)"; fi
    if [ -s "$ZCA1" ] || [ "$DRY" = 1 ]; then cmpx $t a1-F-vs-zzzc16-a1-F $ZCA1 $O/$t-a1-F.jsonl; idok $a-a1-F-vs-zzzc16-a1-F
    else xlog "$a identity a1-F vs zzzc16 a1-F: VIRHE $ZCA1 missing -- not compared"; IDFAIL="$IDFAIL $a-a1-F-vs-zzzc16-a1-F(reference-missing)"; fi
  else IDFAIL="$IDFAIL $a-identity(budget)"; fi
  if [ $(echo $IDFAIL | wc -w) -gt $nf ]; then
    KVSTOPS="$KVSTOPS $a"
    xlog "$a IDENTITY GATE: VIRHE strict identity failed [$(echo $IDFAIL | tr ' ' '\n' | tail -n +$((nf + 1)) | tr '\n' ' ')] -> THIS START's speed reads STOP (placement only must be byte-identical: NEEDS USER); the next start still runs | $O/identity.txt, $O/$t-cmp-*.json"
    return 0
  fi
  xlog "$a IDENTITY GATE: warm-up vs stock + a1-F vs ref12 a1 + a1-F vs zzzc16 a1-F IDENTICAL$([ "$DRY" = 1 ] && echo " (dry)") -> the speed reads run under F $(armj F)"
  for k in $(seq 1 $C8B); do stopfile && return 0
    budok "$a c8 gens block $k (gens $C8GN)" 2 || continue
    mbx 8 ${a}x$k F G || true
  done
  fusesay "$a c8 gens [$C8B block(s)]"
  for k in $(seq 1 $C16B); do stopfile && return 0
    budok "$a c16 gens block $k (gens $C16GN)" 2 || continue
    mbx 16 ${a}x$k F G || true
  done
  [ $C16B -gt 0 ] && fusesay "$a c16 gens [$C16B block(s)]"
  if [ $U1ON = 1 ]; then
    stopfile && return 0
    if budok "$a 1u essay (one discarded + $U1N back to back)" 1; then
      SETN=$a-U1; SETK=$STARTN; SARM=$a
      blockx U1W F essay || true
      for k in $(seq 1 $U1N); do stopfile && break; blockx U1 F essay || true; done
      SETN=""; SETK=0
      fusesay "$a 1u"
    fi
  else xlog "$a 1u: OFF (KVR_1U=0)"; fi
  return 0
}
kvstart() {   # kvstart ARM HP: ONE fresh start = zzzc16's int16 start (its dstart: ngshare, no ATTACH, MAXSEQS 16, RADIANCE_DS41_RFUSE=RFDECL, every
              # start guard, THE int16 PIN posted as O and read back) with --host-pool-mib HP; then the placement read-back (kvplace), the
              # readings (kvreads), the engine log (docker logs -t) captured, down
  local a=$1 hp=$2 tag=xkvr-$1 sc
  STARTN=$((STARTN + 1))
  diskok $tag || return 1
  stackenv; export MODEL=$M225; HP29=$hp; best_vars   # best_vars: HOSTPOOL=$HP29 -> --host-pool-mib HP (lib-pr266 serve_args)
  [ -n "${KVR_MAXSEQS:-}" ] && export MAXSEQS=$KVR_MAXSEQS
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on" PROFEVERY=
  export AQLSTAMP=$([ $STAMPN -gt 0 ] && echo $STAMPN)
  export NGRAM=ram=/ngram   # ngshare.sh's restart sets it too; here it shows in the start line
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV > $XS2_ENVFILE; rm -f $STATSDIR/stamps.csv*"; else printf '%s\n' $LEVENV > $XS2_ENVFILE; rm -f $STATSDIR/stamps.csv*; fi
  xlog "$tag: start $STARTN of $(echo $KVARMS | wc -w) (arm $a: --host-pool-mib $hp, K0 $REFHP; the int16 start of zzzc16 otherwise; ngshare: Y $(basename $Y), NGRAM=ram=/ngram, keeper; n-gram tables now $(ls $NGDIR/ngram-*.tab 2>/dev/null | wc -l) (0: this start fills them, ~308 s)) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $(pinj O | cut -c1-1200) | exl3r4d.so md5 ${PLUGGOT:-?} | AQLSTAMP=${AQLSTAMP:-unset} | env: $(env | grep -E '^(RADIANCE_|R4D_)' | grep -vE '(KEY|TOKEN|SECRET)' | sort | tr '\n' ' ' | cut -c1-1500)"
  gate1 $tag 70 || { xlog "$tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  modes $tag || { xlog "$tag: VIRHE the modes are not the start's -> stopped"; down; return 1; }
  drafted $tag || { down; return 1; }
  readback $tag || { keeplog $tag-readback; down; return 1; }
  a11line $tag || { keeplog $tag-table; down; return 1; }
  a4line $tag || { keeplog $tag-a4table; down; return 1; }
  xpline $tag || { keeplog $tag-xpush; down; return 1; }
  vramstart $tag
  fusestart $tag || { keeplog $tag-fuse; down; return 1; }   # before swnote: the start values (ds41_rfuse RFDECL) and the refusals of the fusion switches
  swnote $tag || { xlog "$tag: VIRHE the switches are not as expected -> stopped"; keeplog $tag-switch; down; return 1; }   # posts THE int16 PIN (arm O) and reads it back
  rankrb $tag || { keeplog $tag-pin; down; return 1; }   # every switch of THE int16 PIN on GET /admin/fn4 + the 4 ranks on GET /stats
  if [ "$DRY" = 1 ]; then echo "+ GET /admin/fn4: exl3_x3d $XB under the pin (THE int15 PIN's X3D + K2.5 gate_up G3 + down U4; exl3_x3d $XREF refused by swnote)"
  else
    sc=$(fn4get | python3 -c 'import json,sys; print((json.load(sys.stdin).get("switches") or {}).get("exl3_x3d"))' 2>/dev/null)
    if [ "$sc" = "$XB" ]; then xlog "$tag int16: GET /admin/fn4 exl3_x3d $sc under the pin | exl3r4d.so md5 ${PLUGGOT:-?} (want $PLUGMD5: ds41-k25u 38b88ec) | exl3_x3d $XREF refused by swnote"
    elif [ $SW = 0 ]; then xlog "$tag int16: INFO (KVR_SW=0 mock) GET /admin/fn4 exl3_x3d ${sc:-none} -- not posted under the mock"
    else xlog "$tag int16: VIRHE GET /admin/fn4 exl3_x3d '${sc:-none}' under the pin (want $XB) -- not the int15 base: stopped"; keeplog $tag-x3d; down; return 1; fi
  fi
  pin16 $tag || { keeplog $tag-pin16; down; return 1; }   # the fusion switches + rco / fuse engagement under the pin
  identw $tag $WARM_REF_225
  if kvplace $tag $a $hp; then
    xlog "$tag THE int16 PIN O (read back, fn4 + every rank + engagement) on --host-pool-mib $hp: $(pinj O | cut -c1-900) | readings: 1 identity (warm-up, a1-F vs ref12 a1 + zzzc16 a1-F; ANY failure: this start's speed reads STOP) | 2 c8 gens $C8GN x $C8B | 3 c16 gens $C16GN x $C16B | 4 1u essay 1 discarded + $U1N ($([ $U1ON = 1 ] && echo on || echo off)) | budget $BUDGET min"
    stopfile || kvreads $tag $a
    KVDONE="$KVDONE $a"
  else
    KVSTOPS="$KVSTOPS $a"
    xlog "$tag: VIRHE the placement read-back failed (KVPLACE above) -> NO reading for arm $a (the start is not the arm it should be)"
  fi
  ssnap $tag-end
  capture $O/$tag-engine-full.log.gz
  xlog "$tag: at the end: $(fn4get | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print(" ".join("%s %s" % (k, w.get(k)) for k in ("ds41_rfuse", "ds41_mix_fin", "route_copy_ov", "exl3_x3d", "ds41_bi", "gg_pf", "ar_xpush", "gg_gemv_m16", "place_swap_lent", "heat_min_gain_x100", "draft_lend")) + " | fuse " + json.dumps(d.get("fuse")) + " | rco " + json.dumps(d.get("rco")))' 2>/dev/null) | $(grep -c ' E ' $O/start-$tag.log 2>/dev/null) E lines in the start log | $(elapsed) min"
  down   # the stamp writer flushes at exit; the n-gram tables stay for the next start (ngshare keeper)
  [ "$DRY" = 1 ] || sleep 3
  xlog "$tag: done"
  return 0
}
kvsummary() {   # per start: ncmatch.py's matched-prompt newcomer sums (c8) + steps.py's raw listings (c8 / c16) from its engine log (INFO, zzzfuse's
                # method); then kvsum.py (every KVRES line); the raw snapshots into one tar.gz, the logs gzipped
  local e a f l EL
  if [ "$DRY" = 1 ]; then
    echo "+ per arm: python3 -I -B $NCM $O/xkvr-<ARM>-engine-full.log.gz --set c8g-<ARM> --block c8-<ARM>x<k>-G-F:F:$O/c8/c8-<ARM>x<k>-G-F-reqs.jsonl ... --min 4 --json $O/ncm-c8g-<ARM>.json; zcat <the log> | step lines > $O/steplines.tmp; per gens block python3 -I -B $STP $O/steplines.tmp <w0-2s> <w1+2s> <w0> <w1> > $O/c8/<block>-steps.txt; rm the tmp"
    echo "+ python3 -I -B $SAS $O --arms '$KVARMSC' --sw $SW --ref-zf $REFZF --ref-zc $REFZC --c8gens $C8GN --c16gens $C16GN --u1n $U1N --essay-md5 $ESSMD5 --json $O/kvsum.json > $O/kvsum.txt -> RESULT xkvr KVRES lines (REFERENCES / PLACE / IDENTITY / c8 BLOCK / c8 / c16 BLOCK / c16 / 1u / <ARM> / TABLE / SLOPE / SUMMARY)"
    echo "+ tar -czf $O/raw-snaps.tar.gz (st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json) and remove them; gzip the start logs, the 1u / mb texts, the steps listings"
    return 0
  fi
  for e in $KVARMS; do a=${e%%:*}; EL=$O/xkvr-$a-engine-full.log.gz
    [ -s $EL ] || { xlog "summary $a: NOTE no engine log $EL (the start did not come up): no step-log readings"; continue; }
    ncmset $EL c8g-$a 4 $(ls $O/c8/c8-${a}x*-G-F-reqs.jsonl 2>/dev/null)
    zcat $EL 2>/dev/null | grep -E ' step [0-9]+ (prefill|decode|MIXED) n_seq=' > $O/steplines.tmp
    for f in $(ls $O/c8/c8-${a}x*-G-F-reqs.jsonl $O/c8/c16-${a}x*-G-F-reqs.jsonl 2>/dev/null); do stepsx $O/steplines.tmp $f; done
    rm -f $O/steplines.tmp
  done
  timeout 600 python3 -I -B $SAS $O --arms "$KVARMSC" --sw $SW --ref-zf $REFZF --ref-zc $REFZC --c8gens $C8GN --c16gens $C16GN --u1n $U1N --essay-md5 $ESSMD5 --json $O/kvsum.json > $O/kvsum.txt 2>&1
  while IFS= read -r l; do xlog "${l:0:2800}"; done < <(grep -h '^KVRES' $O/kvsum.txt 2>/dev/null)
  grep -q '^KVRES' $O/kvsum.txt 2>/dev/null || xlog "summary: VIRHE kvsum.py printed no KVRES line ($(tail -3 $O/kvsum.txt 2>/dev/null | tr '\n' ' ' | cut -c1-300))"
  grep -lq '^Traceback' $O/kvsum.txt $O/ncm-*.txt 2>/dev/null && xlog "summary: VIRHE a Traceback in the analysis ($(grep -h -A4 '^Traceback' $O/kvsum.txt $O/ncm-*.txt 2>/dev/null | tail -2 | tr '\n' ' ' | cut -c1-300))"
  ( cd $O && tar -czf raw-snaps.tar.gz $(ls st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json 2>/dev/null) 2>/dev/null \
      && rm -f st-*.json fn-*.json m-*.txt c8/*-samp.jsonl c8/*-st-w*.json ) || xlog "summary: NOTE the raw snapshots were not archived (left as they are)"
  gzip -f $O/start-*.log $O/xkvr-*-ld-*.txt $O/c8/*.txt $O/b*-U1*.txt 2>/dev/null || true
  xlog "summary: raw snapshots -> raw-snaps.tar.gz ($(du -m $O/raw-snaps.tar.gz 2>/dev/null | cut -f1) MB); logs gzipped; $O $(du -sm $O 2>/dev/null | cut -f1) MB; $(dfree) MB free on /"
  return 0
}
