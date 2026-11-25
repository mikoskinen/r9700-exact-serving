fusestart() {   # fusestart TAG: the int16 start state of the fusion levers -- the FUSE-R declare line "now RFDECL" with form 2 beside every trunk router
                # (> 0 layers), GET /admin/fn4 ds41_rfuse RFDECL / ds41_mix_fin 0 / route_copy_ov 0 (RCO and FUSE-M come by fn4 only), "fuse".rf_declared
                # and "rco" present, RADIANCE_DS41_MIX_FIN / RADIANCE_AQL_ROUTE_OV NOT in the container env, the refusals ds41_rfuse 3 / ds41_mix_fin 2 /
                # route_copy_ov 2 -> 400 -> 0; anything else: VIRHE -> 1 (nothing read; int16 needs form 2)
  local ln g r3 m2 c2 nf1 nf2 ce ev
  if [ "$DRY" = 1 ]; then
    echo "+ fusestart $1: the start log's 'lever FUSE-R (ds41_rfuse, RADIANCE_DS41_RFUSE, now $RFDECL) declared: beside the router of N MoE layers -- router_scored_topk_scatter (form 1 ..) at N1, router_gemm_f32_scored_topk_scatter (form 2 ..) at N2' (now $RFDECL, N2 > 0, else VIRHE); GET /admin/fn4 ds41_rfuse $RFDECL, ds41_mix_fin 0, route_copy_ov 0, \"fuse\".rf_declared > 0, \"rco\" present; RADIANCE_DS41_MIX_FIN / RADIANCE_AQL_ROUTE_OV not in the env; POST ds41_rfuse 3 / ds41_mix_fin 2 / route_copy_ov 2 -> 400"
    return 0
  fi
  ln=$(docker logs $CN 2>&1 | grep -m1 'lever FUSE-R (ds41_rfuse' | sed 's/^radiance: //; s/^[A-Z]\[[0-9]\] //; s/^[A-Z] *//' | cut -c1-400)
  g=$(fn4get | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d.get("switches") or {}; f=d.get("fuse") or {}; print(w.get("ds41_rfuse"), w.get("ds41_mix_fin"), w.get("route_copy_ov"), f.get("rf_declared"), f.get("rf_layers"), "rco" if isinstance(d.get("rco"), dict) else "no-rco")' 2>/dev/null)
  ce=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CN 2>/dev/null); ev=$(echo "$ce" | grep -c '^RADIANCE_DS41_MIX_FIN=\|^RADIANCE_AQL_ROUTE_OV=')
  r3=400; m2=400; c2=400   # the refusal probes: not under ZC16_SW=0 (the mock knows none of these switches)
  [ $SW = 1 ] && { r3=$(fn4code '{"ds41_rfuse": 3}'); m2=$(fn4code '{"ds41_mix_fin": 2}'); c2=$(fn4code '{"route_copy_ov": 2}'); }
  read -r nf1 nf2 < <(echo "$ln" | python3 -c 'import re,sys
m = re.search(r"form 1, [^)]*\) at (\d+), .*form 2, [^)]*\) at (\d+)", sys.stdin.read())
print(m.group(1), m.group(2)) if m else print(-1, -1)' 2>/dev/null || echo "-1 -1")
  xlog "$1 levers FUSE-R / FUSE-M / RCO at start (RADIANCE_DS41_RFUSE=$RFDECL declared): '${ln:-NO FUSE-R line}' | GET ds41_rfuse / ds41_mix_fin / route_copy_ov / fuse.rf_declared / rf_layers / rco ${g:-none} | env RADIANCE_DS41_MIX_FIN / RADIANCE_AQL_ROUTE_OV $ev (want 0) | refusals ds41_rfuse 3 -> HTTP $r3, ds41_mix_fin 2 -> HTTP $m2, route_copy_ov 2 -> HTTP $c2"
  if [ $SW = 0 ]; then xlog "$1 levers FUSE-R / FUSE-M / RCO: INFO (ZC16_SW=0 mock: not served) -- the mock's arms all run the int8 pin"; return 0; fi
  case "$ln" in *"now $RFDECL)"*) ;; *) xlog "$1: VIRHE the FUSE-R declare line does not say 'now $RFDECL' ('${ln:-none}'): RADIANCE_DS41_RFUSE=$RFDECL did not take"; return 1 ;; esac
  [ "${nf2:--1}" -gt 0 ] 2>/dev/null || { xlog "$1: VIRHE FUSE-R form 2 not declared ($nf2 layers): int16 cannot run"; return 1; }
  case "$g" in "$RFDECL 0 0 "*" rco") ;; *) xlog "$1: VIRHE GET /admin/fn4 ds41_rfuse / ds41_mix_fin / route_copy_ov '${g:-none}' at start (want $RFDECL 0 0, \"rco\" present)"; return 1 ;; esac
  case "$g" in "$RFDECL 0 0 0 "*|"$RFDECL 0 0 None "*) xlog "$1: VIRHE GET /admin/fn4 fuse.rf_declared '${g:-none}' (want > 0)"; return 1 ;; esac
  [ "$ev" = 0 ] || { xlog "$1: VIRHE RADIANCE_DS41_MIX_FIN / RADIANCE_AQL_ROUTE_OV in the container env ($ev): int16 sets them by fn4 only"; return 1; }
  [ "$r3" = 400 ] && [ "$m2" = 400 ] && [ "$c2" = 400 ] || { xlog "$1: VIRHE a fusion refusal probe was not refused (ds41_rfuse 3 -> $r3, ds41_mix_fin 2 -> $m2, route_copy_ov 2 -> $c2)"; return 1; }
  return 0
}
dstart() {   # the tables (zzzfuse's = ref14's, not re-benched), then the ONE fresh start (ngshare) with RADIANCE_DS41_RFUSE=RFDECL, THE int16 PIN right
             # after it (swnote posts O), read back (rankrb: every switch + the 4 ranks; the exl3_x3d line; pin16: the fusion switches + engagement)
  local tag=xzc16-n5 sc
  diskok $tag || return 1
  tabs14 || return 1
  LEVENV="$LEVENV RADIANCE_DS41_GG_M16_FORMS=/stats/m16-forms.txt RADIANCE_FN4_AR_XPUSH_MIN=$XPMIN RADIANCE_DS41_RFUSE=$RFDECL"   # A4's table + XPUSH's bound (int14) + FUSE-R declared at form RFDECL (int16)
  LEVENV=$(echo "$LEVENV" | sed "s#RADIANCE_DS41_GG_PF_TABLE=$GPTAB #RADIANCE_DS41_GG_PF_TABLE=$GPTAB,$A11ENT #")
  case "$LEVENV" in *"RADIANCE_DS41_GG_PF_TABLE=$GPTAB,$A11ENT "*) ;; *) xlog "$tag: VIRHE the ranged entries did not go into LEVENV -> NOT STARTED"; return 1 ;; esac
  stackenv; export MODEL=$M225; HP29=19456; best_vars
  [ -n "${ZC16_MAXSEQS:-}" ] && export MAXSEQS=$ZC16_MAXSEQS
  export MNBT=4160 NST=5 XARGS="$XARGS --live --draft-window adaptive --draft-coupling on" PROFEVERY=
  export AQLSTAMP=$([ $STAMPN -gt 0 ] && echo $STAMPN)
  export NGRAM=ram=/ngram   # ngshare.sh's restart sets it too; here it shows in the start line
  export XS2_ENVFILE=$O/$tag.env
  if [ "$DRY" = 1 ]; then echo "+ printf '%s\\n' $LEVENV > $XS2_ENVFILE; rm -f $STATSDIR/stamps.csv*"; else printf '%s\n' $LEVENV > $XS2_ENVFILE; rm -f $STATSDIR/stamps.csv*; fi
  xlog "$tag: start (THE int16 CONFIRMATION; ngshare: Y $(basename $Y), NGRAM=ram=/ngram, keeper; n-gram tables now $(ls $NGDIR/ngram-*.tab 2>/dev/null | wc -l) (0: this start fills them, ~308 s); $DESC) IMAGE=$IMAGE MODEL=$MODEL $(desc) | pin $(pinj O | cut -c1-1200) | exl3r4d.so md5 ${PLUGGOT:-?} | AQLSTAMP=${AQLSTAMP:-unset} | env: $(env | grep -E '^(RADIANCE_|R4D_)' | grep -vE '(KEY|TOKEN|SECRET)' | sort | tr '\n' ' ' | cut -c1-2800)"
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
    elif [ $SW = 0 ]; then xlog "$tag int16: INFO (ZC16_SW=0 mock) GET /admin/fn4 exl3_x3d ${sc:-none} -- not posted under the mock"
    else xlog "$tag int16: VIRHE GET /admin/fn4 exl3_x3d '${sc:-none}' under the pin (want $XB) -- not the int15 base: stopped"; keeplog $tag-x3d; down; return 1; fi
  fi
  pin16 $tag || { keeplog $tag-pin16; down; return 1; }   # the fusion switches + rco / fuse engagement under the pin
  identw $tag $WARM_REF_225
  [ "$DRY" = 1 ] || { : > $O/blocks.jsonl; : > $O/mb.jsonl; : > $O/c16cap.jsonl; }
  xlog "THE int16 PIN O (read back, fn4 + every rank + engagement): $(pinj O | cut -c1-1400) | arms B (int15) $(armj B), F (int16 = O) $(armj F) | readings: 1 identity under F: warm-up, a1-F vs ref12 a1 + zzzfuse a1-B, gs$GSC-F, staggered ld-F (ANY failure: STOP) | 2 1u ABBA [$U1AB] 1 discarded + $U1N each ($([ $ONEU = 1 ] && echo on || echo off)) | 3 c4 gens $C4GN $([ $C4WARMUP = 1 ] && echo "1 discarded ($C4WA) + ")[$C4AB] ($([ $C4ON = 1 ] && echo on || echo off)) | 4 cold $CDEP F ($([ $COLD = 1 ] && echo "on, budget allowing" || echo off))"
  stopfile || c16reads
  capture $O/$tag-engine-full.log.gz
  xlog "$tag: at the end: $(fn4get | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["switches"]; print(" ".join("%s %s" % (k, w.get(k)) for k in ("ds41_rfuse", "ds41_mix_fin", "route_copy_ov", "exl3_x3d", "ds41_bi", "gg_pf", "ar_xpush", "gg_gemv_m16", "place_swap_lent", "heat_min_gain_x100")) + " | fuse " + json.dumps(d.get("fuse")) + " | rco " + json.dumps(d.get("rco")))' 2>/dev/null) | $(docker logs --since 6h $CN 2>&1 | grep -cE '^E ') E lines | $(elapsed) min"
  down   # the stamp writer flushes at exit
  [ "$DRY" = 1 ] || sleep 3
  xlog "$tag: done$([ $STOPPED = 1 ] && echo " (STOPPED at the identity gate: no speed block ran)")"
}
