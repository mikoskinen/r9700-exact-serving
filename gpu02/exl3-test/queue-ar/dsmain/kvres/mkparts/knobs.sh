# kvres: the arms (one fresh start each) and the readings (module header)
KVARMS=${KVR_ARMS-K35:16800 K60:14900}; KVARMSC=$(echo $KVARMS | tr ' ' ',')   # ARM:--host-pool-mib; K0 = 19456 = zzzc16's start (not re-run)
C8B=${KVR_C8B:-2}; C16B=${KVR_C16B:-1}; U1ON=${KVR_1U:-1}; STARTEST=${KVR_STARTEST:-11}; STARTN=0; KVSTOPS=""; KVDONE=""   # c8 / c16 gens blocks a start, 1u on, minutes a start
REFHP=19456; REFZC=${KVR_REFZC:-$R/xdsm/zzzc16}; REFZF=${KVR_REFZF:-$R/xdsm/zzzfuse}   # the K0 references: zzzc16 (int16 start: placement, a1-F, 1u F), zzzfuse (c8 / c16 F gens blocks)
REFSTART=${KVR_REFSTART:-$REFZC/start-xzc16-n5.log.gz}; ZCA1=${KVR_ZCA1:-$REFZC/xzc16-a1-F.jsonl}
for v in $C8B $C16B $U1ON $STARTEST; do case $v in ''|*[!0-9]*) echo "a kvres knob is not a number ('$v')"; exit 1 ;; esac; done
[ -n "$KVARMS" ] || { echo "KVR_ARMS names no arm"; exit 1; }
for e in $KVARMS; do
  case $e in [A-Za-z]*:[0-9]*) ;; *) echo "KVR_ARMS entries are NAME:HOSTPOOL_MIB (got '$e')"; exit 1 ;; esac
  case ${e%%:*} in *[!A-Za-z0-9]*) echo "KVR_ARMS: an arm name is letters / digits (got '${e%%:*}')"; exit 1 ;; esac
  case ${e#*:} in *[!0-9]*) echo "KVR_ARMS: the host pool is MiB (got '${e#*:}')"; exit 1 ;; esac
  [ ${e#*:} -ge 12288 ] && [ ${e#*:} -le 24576 ] || { echo "KVR_ARMS: --host-pool-mib 12288 .. 24576 (below ~13000 the KV pool nears one 524288-token sequence + the staging ring) (got ${e#*:})"; exit 1; }
done
[ "$(echo $KVARMS | tr ' ' '\n' | cut -d: -f1 | sort | uniq -d)" = "" ] || { echo "KVR_ARMS: an arm name twice"; exit 1; }
[ $C8B -ge 1 ] && [ $C16B -le 4 ] && [ $C8B -le 4 ] || { echo "KVR_C8B 1..4, KVR_C16B 0..4"; exit 1; }
case $U1ON in 0|1) ;; *) echo "KVR_1U 0 or 1"; exit 1 ;; esac
