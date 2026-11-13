ENV14=0
env14() {   # the int14 start words ONCE for both starts: the tables (ref14's = yzint14c's, not re-benched), A4's table + XPUSH's bound + A11's
            # ranged entries into LEVENV (ref14's dstart lines, run once)
  [ $ENV14 = 1 ] && return 0
  tabs14 || return 1
  LEVENV="$LEVENV RADIANCE_DS41_GG_M16_FORMS=/stats/m16-forms.txt RADIANCE_FN4_AR_XPUSH_MIN=$XPMIN"   # A4's table + XPUSH's bound (int14)
  LEVENV=$(echo "$LEVENV" | sed "s#RADIANCE_DS41_GG_PF_TABLE=$GPTAB #RADIANCE_DS41_GG_PF_TABLE=$GPTAB,$A11ENT #")
  case "$LEVENV" in *"RADIANCE_DS41_GG_PF_TABLE=$GPTAB,$A11ENT "*) ;; *) xlog "int14 start words: VIRHE the ranged entries did not go into LEVENV -> NOT STARTED"; return 1 ;; esac
  ENV14=1
}
zstart() {   # zstart TAG MAXSEQS: ONE fresh start (ngshare), int14 pinned right after it and read back (ref14's dstart, MAXSEQS a parameter)
             # -> 0 = up and pinned (the caller reads, captures and stops it), 1 = not (stopped, logged)
  local tag=$1 ms=$2
  diskok $tag || return 1
  env14 || return 1
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  export MAXSEQS=$ms
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on" PROFEVERY=
  export AQLSTAMP=$([ $STAMPN -gt 0 ] && echo $STAMPN)
  export NGRAM=ram=/ngram   # ngshare.sh's restart sets it too; here it shows in the start line
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV > $XS2_ENVFILE; rm -f $STATSDIR/stamps.csv*"; else printf '%s\n' $LEVENV > $XS2_ENVFILE; rm -f $STATSDIR/stamps.csv*; fi
  xlog "$tag: start (MAXSEQS $ms; ngshare: Y $(basename $Y), NGRAM=ram=/ngram, keeper; n-gram tables now $(ls $NGDIR/ngram-*.tab 2>/dev/null | wc -l) (0: this start fills them, ~308 s; 2: reused); $DESC) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $(pinj O | cut -c1-1100) | AQLSTAMP=${AQLSTAMP:-unset} | env: $(env | grep -E '^(RADIANCE_|R4D_)' | grep -vE '(KEY|TOKEN|SECRET)' | sort | tr '\n' ' ' | cut -c1-2800)"
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
  identw $tag $WARM_REF_225
  [ "$DRY" = 1 ] || { local mq; mq=$(docker inspect -f '{{join .Args " "}}' $CN 2>/dev/null | grep -oE -- '--max-num-seqs[= ][0-9]+' | head -1); }
  [ "$DRY" = 1 ] || xlog "$tag MAXSEQS read back: the container's serve line '${mq:-?}' (want --max-num-seqs $ms$([ "${mq##* }" = "$ms" ] || echo "; INFO not read back from docker inspect -- the start line's desc says MAXSEQS=$MAXSEQS")) | GET /admin/fn4 draft switches $(fn4get | python3 -c 'import json,sys; w = json.load(sys.stdin)["switches"]; print(" ".join("%s %s" % (k, w.get(k)) for k in ("draft_depth0", "draft_occ", "draft_occ_k", "draft_adaptive", "draft_lend", "draft_probe", "sched_row_cap", "dspark_conf", "dspark_conf_rows")))' 2>/dev/null)"
  xlog "THE int14 PIN O (read back, fn4 + every rank): $(pinj O | cut -c1-1300) | arms B $(armj B) (int12 bytes), Z $(armj Z) (= O)"
  return 0
}
zstop() {   # zstop TAG: the start's engine log captured (docker logs -t, gzipped), the end state, the stop
  local tag=$1
  capture $O/$tag-engine-full.log.gz
  xlog "$tag: at the end: $(i11now) | $(fn4get | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print(" ".join("%s %s" % (k, w.get(k)) for k in ("ds41_bi", "gg_pf", "ar_xpush", "gg_gemv_m16", "place_swap_lent", "heat_min_gain_x100", "draft_depth0", "draft_occ"))); print("| draft %s | lend %s" % (json.dumps(d.get("draft")), json.dumps({k: (d.get("lend") or {}).get(k) for k in ("lent_cycles", "state", "backoff_k", "lends", "reclaims")})))' 2>/dev/null | tr '\n' ' ') | $(docker logs --since 6h $CN 2>&1 | grep -cE '^E ') E lines | $(elapsed) min"
  down   # the stamp writer flushes at exit
  [ "$DRY" = 1 ] || sleep 3
  xlog "$tag: done"
}
dstart() {   # zrq14: START 1 (MAXSEQS MS1: identity, c1 / c4 B Z Z B + the window bridges, 1u B / Z / B) -> THE GATE -> START 2 (MAXSEQS MS2, B only)
  [ "$DRY" = 1 ] || { : > $O/blocks.jsonl; : > $O/mb.jsonl; : > $O/c16cap.jsonl; }
  SARM=m16; ZQSFX=""
  zstart xzq14-m16 $MS1 || { xlog "start 1 (MAXSEQS $MS1): VIRHE not up -> no reading, no start 2"; return 1; }
  stopfile || zqreads16
  zstop xzq14-m16
  stopfile && return 0
  if [ $STOPPED = 1 ]; then xlog "START 2: NOT RUN (start 1 stopped at the identity gate)"; [ "$DRY" = 1 ] || echo '{"decision": "NOT RUN (start 1 stopped at the identity gate)"}' > $O/gate2.json; return 0; fi
  zqgate || return 0
  SARM=m8; ZQSFX=8
  zstart xzq14-m8 $MS2 || { xlog "start 2 (MAXSEQS $MS2): VIRHE not up -> no start-2 reading"; return 1; }
  stopfile || zqreads8
  zstop xzq14-m8
  return 0
}
