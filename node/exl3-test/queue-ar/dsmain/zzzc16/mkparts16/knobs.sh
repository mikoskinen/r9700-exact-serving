# zzzc16: THE int16 confirmation's arms and readings (module header)
XB=${ZC16_XB:-2}; XREF=${ZC16_XREFUSE:-5}   # THE int15 PIN's exl3_x3d (2: X3D + K2.5 gate_up G3 + down U4) in every arm; XREF: swnote's refusal probe (400)
RFDECL=${ZC16_RFDECL:-2}; FRF=2; FMF=0; FRC=1   # RADIANCE_DS41_RFUSE at start (int16: FUSE-R declared AND started at form 2); THE int16 PIN's ds41_rfuse / ds41_mix_fin / route_copy_ov
U1AB=${ZC16_U1ABBA-B F F B}; C4AB=${ZC16_C4ABBA-B F F B}; C4GN=${ZC16_C4GN:-3}; C4WA=${ZC16_C4WARMARM:-F}; C16GN=${ZC16_C16GN:-1}   # 1u ABBA blocks; c4 gens blocks (+ C4WARMUP's discarded one)
ONEU=${ZC16_1U:-1}; C4ON=${ZC16_C4:-1}; COLD=${ZC16_COLD:-1}; CDEP=${ZC16_CDEPTH:-136000}; COLDEST=${ZC16_COLDEST:-2}; POSTN=${ZC16_POST:-10}   # sections on / off; the cold depth and its budget estimate
IDARMS=F; GSA=F; LDA=F   # identity runs under F (THE int16 PIN) only
STPMD5=${ZC16_STPMD5:-569fd59b}   # the coordinator's steps.py (zzzc16/steps.py = zzzfuse's = zzk25s's)
REFZF=${ZC16_REFZF:-$R/xdsm/zzzfuse}; ZFA1=${ZC16_ZFA1:-$REFZF/xzfuse-a1-B.jsonl}   # zzzfuse's run: its a1-B (int15, IDENTICAL to ref12's a1 there) -- a1-F here must equal it
R14C=${ZC16_R14COLD:-$R/xdsm/ref14/cold-136000-1-O.txt}; R14CMD5=${ZC16_R14CMD5:-3623688c}; ZFC=${ZC16_ZFCOLD:-$REFZF/cold-136000-1-F.txt}   # the 136k answers byte-compared: ref14's (md5 3623688c = ref12's); zzzfuse's only if it exists (it ran none)
for v in $XB $XREF $RFDECL $C4GN $C16GN $ONEU $C4ON $COLD $CDEP $COLDEST $POSTN; do case $v in ''|*[!0-9]*) echo "a zzzc16 knob is not a number ('$v')"; exit 1 ;; esac; done
[ $XB = 2 ] || { echo "ZC16_XB 2 (THE int15 PIN's exl3_x3d)"; exit 1; }; [ $RFDECL = 2 ] || { echo "ZC16_RFDECL 2 (int16 declares FUSE-R and starts at form 2)"; exit 1; }
[ $XREF -ge 5 ] || { echo "ZC16_XREFUSE >= 5 (0 .. 4 are VALID on the K25U plugin: a probe there SETS the switch)"; exit 1; }
[ $C4GN -ge 1 ] || { echo "ZC16_C4GN >= 1"; exit 1; }; case $ONEU$C4ON$COLD in [01][01][01]) ;; *) echo "ZC16_1U / ZC16_C4 / ZC16_COLD 0 or 1"; exit 1 ;; esac
[ -n "$U1AB" ] && [ -n "$C4AB" ] || { echo "ZC16_U1ABBA / ZC16_C4ABBA must name blocks"; exit 1; }
for a in $U1AB $C4AB $C4WA; do case $a in B|F) ;; *) echo "ZC16_U1ABBA / C4ABBA / C4WARMARM take B (int15) and F (int16) (got '$a')"; exit 1 ;; esac; done
case " $U1AB " in *" B "*) ;; *) echo "ZC16_U1ABBA needs B (the reading is F vs B)"; exit 1 ;; esac; case " $U1AB " in *" F "*) ;; *) echo "ZC16_U1ABBA needs F"; exit 1 ;; esac
