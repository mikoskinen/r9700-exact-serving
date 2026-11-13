#!/bin/bash
# exl3-restart.sh TAG -- (re)start the EXL3 DeepSeek test server with the env.sh variables (override any of them in the environment),
# wait for health (<= 20 min), log load time, KV / expert split, EXL3 plugin lines, pinned RAM, RAM and VRAM as one RESULT line.
# = ~/gpu/ds41/ds41-restart2.sh (3.11.) for ds41-exl3-tp4.yaml.
set -u
TAG=$1; . "$(dirname "$0")/env.sh"
F="--project-directory $C -f $T/ds41-exl3-tp4.yaml"
docker compose $F down 2>&1 | tail -1
t0=$(date +%s)
docker compose $F up -d 2>&1 | tail -1
ok=0
for i in $(seq 1 240); do
  [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' $URL/health)" = 200 ] && { ok=1; break; }
  [ "$(docker inspect -f '{{.State.Status}}' radiance-ds41-exl3 2>/dev/null)" != running ] && break
  sleep 5
done
t1=$(date +%s)
docker logs radiance-ds41-exl3 > $R/start-$TAG.log 2>&1
if [ $ok != 1 ]; then
  echo "$(date '+%F %T') RESULT restart-$TAG: VIRHE not healthy after $((t1-t0)) s: $(grep -E '^E ' $R/start-$TAG.log | head -3 | tr '\n' ' ' | cut -c1-600)" | tee -a $L; exit 1
fi
kv=$(grep -m1 -E '^I kv: the pool' $R/start-$TAG.log | cut -c3-200)
ex=$(grep -m1 -E '^ +experts +[0-9.]+ GiB' $R/start-$TAG.log | awk '{print $2, $3}')
kvc=$(grep -m1 -E '^I +kv cache +[0-9.]+ GiB a card' $R/start-$TAG.log | sed -E 's/^I +kv cache +//' | cut -c1-80)
pin=$(grep -m1 -o 'pinned by this process [0-9.]* GiB' $R/start-$TAG.log)
x3=$(grep -i -m3 -E 'exl3' $R/start-$TAG.log | cut -c1-160 | tr '\n' ' ')
mem=$(free -g | awk 'NR==2{print "used " $3 " / avail " $7 " GiB"}')
vram=$(for d in /sys/class/drm/card*/device; do [ -e $d/mem_info_vram_used ] && printf "%s " $(( $(cat $d/mem_info_vram_used)/1048576 )); done)
echo "$(date '+%F %T') RESULT restart-$TAG: healthy in $((t1-t0)) s (IMAGE=$IMAGE MODEL=$MODEL HOSTPOOL=$HOSTPOOL MAXSEQS=$MAXSEQS NST=$NST CKINT=$CKINT MNBT=$MNBT KVD=$KVD WIRE=$WIRE MML=$MML EXTRA2='$EXTRA2' EXTRA='$EXTRA'); $kv; VRAM experts $ex, KV $kvc; $pin; RAM $mem; VRAM used MiB [$vram]; exl3 lines: ${x3:-none}" | tee -a $L
