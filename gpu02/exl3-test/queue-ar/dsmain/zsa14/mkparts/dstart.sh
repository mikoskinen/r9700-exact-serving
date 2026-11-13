dstart() {   # the tables (ref14's = yzint14c's, not re-benched), then the ONE fresh start (ngshare), int14 pinned right after it and read back
  local tag=xsa14-n5 sc
  diskok $tag || return 1
  tabs14 || return 1
  LEVENV="$LEVENV RADIANCE_DS41_GG_M16_FORMS=/stats/m16-forms.txt RADIANCE_FN4_AR_XPUSH_MIN=$XPMIN"   # A4's table + XPUSH's bound (int14)
  LEVENV=$(echo "$LEVENV" | sed "s#RADIANCE_DS41_GG_PF_TABLE=$GPTAB #RADIANCE_DS41_GG_PF_TABLE=$GPTAB,$A11ENT #")
  case "$LEVENV" in *"RADIANCE_DS41_GG_PF_TABLE=$GPTAB,$A11ENT "*) ;; *) xlog "$tag: VIRHE the ranged entries did not go into LEVENV -> NOT STARTED"; return 1 ;; esac
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  [ -n "${SA14_MAXSEQS:-}" ] && export MAXSEQS=$SA14_MAXSEQS
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on" PROFEVERY=
  export AQLSTAMP=$([ $STAMPN -gt 0 ] && echo $STAMPN)
  export NGRAM=ram=/ngram   # ngshare.sh's restart sets it too; here it shows in the start line
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV > $XS2_ENVFILE; rm -f $STATSDIR/stamps.csv*"; else printf '%s\n' $LEVENV > $XS2_ENVFILE; rm -f $STATSDIR/stamps.csv*; fi
  xlog "$tag: start (lever SA screen on int14; ngshare: Y $(basename $Y), NGRAM=ram=/ngram, keeper; n-gram tables now $(ls $NGDIR/ngram-*.tab 2>/dev/null | wc -l) (0: this start fills them, ~308 s); $DESC) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $(pinj O | cut -c1-1100) | RADIANCE_STAGE_MIXED ${RADIANCE_STAGE_MIXED-unset} (unset = 1: mixed passes may arm) | AQLSTAMP=${AQLSTAMP:-unset} | env: $(env | grep -E '^(RADIANCE_|R4D_)' | grep -vE '(KEY|TOKEN|SECRET)' | sort | tr '\n' ' ' | cut -c1-2800)"
  gate1 $tag 70 || { xlog "$tag: VIRHE not healthy ($(desc)); $(grep -E '^E ' $O/start-$tag.log 2>/dev/null | head -3 | tr '\n' ' ' | cut -c1-500)"; down; return 1; }
  modes $tag || { xlog "$tag: VIRHE the modes are not the start's -> stopped"; down; return 1; }
  drafted $tag || { down; return 1; }
  readback $tag || { keeplog $tag-readback; down; return 1; }
  a11line $tag || { keeplog $tag-table; down; return 1; }
  a4line $tag || { keeplog $tag-a4table; down; return 1; }
  xpline $tag || { keeplog $tag-xpush; down; return 1; }
  vramstart $tag
  swnote $tag || { xlog "$tag: VIRHE the switches are not as expected -> stopped"; keeplog $tag-switch; down; return 1; }   # posts THE int14 PIN (arm O) and reads it back
  rankrb $tag || { keeplog $tag-pin; down; return 1; }
  if [ "$DRY" = 1 ]; then echo "+ GET /admin/fn4 \"sahb\": stage_capable 1 and 4 ranks (lever SA can arm on every rank), stage_short 0 / stage_pick 0 under the pin (else VIRHE: not read)"
  else
    sc=$(fn4get | python3 -c 'import json,sys; d = json.load(sys.stdin); s = d.get("sahb") or {}; w = d.get("switches") or {}
print("%s %d %s %s" % (s.get("stage_capable"), len(s.get("ranks") or []), w.get("stage_short"), w.get("stage_pick")))' 2>/dev/null)
    case "$sc" in "1 4 0 0") xlog "$tag lever SA: GET /admin/fn4 sahb.stage_capable 1, 4 ranks, stage_short 0 / stage_pick 0 under the pin (B) | the start's line: $(docker logs $CN 2>&1 | grep -m1 'lever SA / HB (ds41-sahb) set up' | sed 's/^radiance: //; s/^[A-Z]\[[0-9]\] //; s/^[A-Z] *//' | cut -c1-400)" ;;
      *) xlog "$tag lever SA: VIRHE sahb stage_capable / ranks / stage_short / stage_pick '${sc:-none}' (want '1 4 0 0') -- SA cannot be read here: stopped"; keeplog $tag-sahb; down; return 1 ;; esac
  fi
  identw $tag $WARM_REF_225
  [ "$DRY" = 1 ] || { : > $O/blocks.jsonl; : > $O/mb.jsonl; : > $O/c16cap.jsonl; }
  xlog "THE int14 PIN O (read back, fn4 + every rank): $(pinj O | cut -c1-1300) | arms B $(armj B) (= O), S1 $(armj S1), S2 $(armj S2) | readings: 1 identity warm-up, a1 [$IDARMS] vs ref12 a1 (+ a1-B vs yzint14c a1-Z), gs$GSC-$GSA, staggered ld-$LDA (ANY failure: STOP) | 2 c8 work-based (gens $C8GN) [$GORD] | 3 c16 gens $C16GN: B + the better SA arm ($([ $C16N -gt 0 ] && echo on || echo off))"
  stopfile || sa14reads
  capture $O/$tag-engine-full.log.gz
  xlog "$tag: at the end: $(fn4get | python3 -c 'import json,sys; w=json.load(sys.stdin)["switches"]; print(" ".join("%s %s" % (k, w.get(k)) for k in ("stage_short", "stage_pick", "place_hb", "ds41_bi", "gg_pf", "ar_xpush", "gg_gemv_m16", "place_swap_lent", "heat_min_gain_x100", "heat_hyst_x100")))' 2>/dev/null) | $(docker logs --since 6h $CN 2>&1 | grep -cE '^E ') E lines | $(elapsed) min"
  down   # the stamp writer flushes at exit
  [ "$DRY" = 1 ] || sleep 3
  xlog "$tag: done$([ $STOPPED = 1 ] && echo " (STOPPED at the identity gate: no speed block ran)")"
}
