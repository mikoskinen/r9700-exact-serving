#!/bin/bash
# fanboot.sh (4.11.2026; user before moving gpu02: "saatko tuulettimet iskuun ennen sammutusta?") -- run from the user crontab
# (@reboot). Waits for r9700-od.service (od-apply.sh writes the acoustic 3500/80/25 settings at boot, which replace a manual curve),
# then re-applies the flat curves of ~/gpu/fan-state.txt (incl. fanguard's raises) with fanflat.sh and starts fanguard.sh.
# No sudo: the render group writes gpu_od. A ~/gpu/fanguard.stop file keeps fanguard off. Log: ~/gpu/fanboot.log.
# Remove: crontab -e (the @reboot line); backup ~/gpu/crontab.bak-20261104-fanboot.
ST=$HOME/gpu/fan-state.txt
exec >> $HOME/gpu/fanboot.log 2>&1
echo "$(date '+%F %T') fanboot start"
for i in $(seq 60); do   # at most 10 min
  s=$(systemctl show -p ActiveState --value r9700-od.service)
  [ "$s" = active ] || [ "$s" = failed ] && break
  sleep 10
done
echo "$(date '+%F %T') r9700-od.service: $s"
sleep 5
args=$(awk 'NF >= 2 {printf "%s=%s ", $1, $2; t = $3} END {if (t) printf "TOP=%s", t}' $ST)
[ -n "$args" ] && $HOME/gpu/fanflat.sh $args
if [ -e $HOME/gpu/fanguard.stop ]; then
  echo "$(date '+%F %T') fanguard.stop exists: fanguard not started"
elif pgrep -f "[f]anguard.sh" > /dev/null; then
  echo "$(date '+%F %T') fanguard already running"
else
  setsid nohup $HOME/gpu/fanguard.sh > /dev/null 2>&1 < /dev/null &
  echo "$(date '+%F %T') fanguard started (pid $!)"
fi
