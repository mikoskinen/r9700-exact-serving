dstart() {   # the tables (zsa14's = ref14's, not re-benched), then the ONE fresh start (ngshare), int14 pinned right after it and read back
  local tag=xk25s-n5 sc
  diskok $tag || return 1
  tabs14 || return 1
  LEVENV="$LEVENV RADIANCE_DS41_GG_M16_FORMS=/stats/m16-forms.txt RADIANCE_FN4_AR_XPUSH_MIN=$XPMIN"   # A4's table + XPUSH's bound (int14)
  LEVENV=$(echo "$LEVENV" | sed "s#RADIANCE_DS41_GG_PF_TABLE=$GPTAB #RADIANCE_DS41_GG_PF_TABLE=$GPTAB,$A11ENT #")
  case "$LEVENV" in *"RADIANCE_DS41_GG_PF_TABLE=$GPTAB,$A11ENT "*) ;; *) xlog "$tag: VIRHE the ranged entries did not go into LEVENV -> NOT STARTED"; return 1 ;; esac
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  [ -n "${K25S_MAXSEQS:-}" ] && export MAXSEQS=$K25S_MAXSEQS
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on" PROFEVERY=
  export AQLSTAMP=$([ $STAMPN -gt 0 ] && echo $STAMPN)
  export NGRAM=ram=/ngram   # ngshare.sh's restart sets it too; here it shows in the start line
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV > $XS2_ENVFILE; rm -f $STATSDIR/stamps.csv*"; else printf '%s\n' $LEVENV > $XS2_ENVFILE; rm -f $STATSDIR/stamps.csv*; fi
  xlog "$tag: start (lever K25U screen on int14 + the K25U exl3r4d.so; ngshare: Y $(basename $Y), NGRAM=ram=/ngram, keeper; n-gram tables now $(ls $NGDIR/ngram-*.tab 2>/dev/null | wc -l) (0: this start fills them, ~308 s); $DESC) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $(pinj O | cut -c1-1100) | exl3r4d.so md5 ${PLUGGOT:-?} | AQLSTAMP=${AQLSTAMP:-unset} | env: $(env | grep -E '^(RADIANCE_|R4D_)' | grep -vE '(KEY|TOKEN|SECRET)' | sort | tr '\n' ' ' | cut -c1-2800)"
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
  if [ "$DRY" = 1 ]; then echo "+ GET /admin/fn4: exl3_x3d $XB under the pin (B = the int14 kernels; exl3_x3d $XREF refused by swnote) and the start log's kernel selection line (else VIRHE: not read)"
  else
    sc=$(fn4get | python3 -c 'import json,sys; print((json.load(sys.stdin).get("switches") or {}).get("exl3_x3d"))' 2>/dev/null)
    if [ "$sc" = "$XB" ]; then xlog "$tag lever K25U: GET /admin/fn4 exl3_x3d $sc under the pin (B) | exl3r4d.so md5 ${PLUGGOT:-?} (want $PLUGMD5: ds41-k25u 38b88ec) | exl3_x3d $XREF refused by swnote | the start's kernel selection: $(docker logs $CN 2>&1 | grep -m1 'kernel selection' | sed 's/^radiance: //; s/^[A-Z]\[[0-9]\] //; s/^[A-Z] *//' | cut -c1-300)"
    elif [ $SW = 0 ]; then xlog "$tag lever K25U: INFO (K25S_SW=0 mock) GET /admin/fn4 exl3_x3d ${sc:-none} -- not posted under the mock"
    else xlog "$tag lever K25U: VIRHE GET /admin/fn4 exl3_x3d '${sc:-none}' under the pin (want $XB) -- K25U cannot be read here: stopped"; keeplog $tag-x3d; down; return 1; fi
  fi
  identw $tag $WARM_REF_225
  [ "$DRY" = 1 ] || { : > $O/blocks.jsonl; : > $O/mb.jsonl; : > $O/c16cap.jsonl; }
  xlog "THE int14 PIN O (read back, fn4 + every rank): $(pinj O | cut -c1-1300) | arms B $(armj B) (= O), Z $(armj Z) | readings: 1 identity warm-up, a1 [$IDARMS] vs ref12 a1 (+ a1-B vs yzint14c a1-Z), gs$GSC-$GSA, staggered ld-$LDA (ANY failure: STOP) | 2 c8 work-based (gens $C8GN) [$GORD] | 3 c16 gens $C16GN [$K16A] ($([ $C16N -gt 0 ] && echo on || echo off)) | 4 1u essay [$U1A] 1 discarded + $U1N each ($([ $ONEU = 1 ] && echo on || echo off))"
  stopfile || k25reads
  capture $O/$tag-engine-full.log.gz
  xlog "$tag: at the end: $(fn4get | python3 -c 'import json,sys; w=json.load(sys.stdin)["switches"]; print(" ".join("%s %s" % (k, w.get(k)) for k in ("exl3_x3d", "place_hb", "ds41_bi", "gg_pf", "ar_xpush", "gg_gemv_m16", "place_swap_lent", "heat_min_gain_x100", "heat_hyst_x100")))' 2>/dev/null) | $(docker logs --since 6h $CN 2>&1 | grep -cE '^E ') E lines | $(elapsed) min"
  down   # the stamp writer flushes at exit
  [ "$DRY" = 1 ] || sleep 3
  xlog "$tag: done$([ $STOPPED = 1 ] && echo " (STOPPED at the identity gate: no speed block ran)")"
}
