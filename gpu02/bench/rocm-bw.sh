#!/bin/bash
# rocm-bw.sh -- inside rocm/dev-ubuntu-24.04 (/w = ~/gpu/bench): rocm-bandwidth-test per card + -p (P2P),
# own p2p-bw (per card, P2P, all cards at once), host memory bandwidth (membw). 31.10., gpu02.
cd /w
echo "== $(date '+%F %T') rocm-bandwidth-test"
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rocm-bandwidth-test) >/tmp/apt.log 2>&1 || tail -3 /tmp/apt.log
RBT=$(command -v rocm-bandwidth-test || command -v rocm_bandwidth_test || ls /opt/rocm*/bin/rocm*bandwidth* 2>/dev/null | head -1)
if [ -n "$RBT" ]; then
  echo "binary: $RBT"; $RBT -h 2>&1 | head -30
  echo "-- default (all agents)"; timeout 600 $RBT 2>&1 | tail -60
  echo "-- -p"; timeout 600 $RBT -p 2>&1 | tail -40
else echo "rocm-bandwidth-test not available"; fi
echo "== $(date '+%F %T') p2p-bw"
hipcc -O2 -o p2p-bw p2p-bw.cpp 2>&1 | grep -iE "error" | head -5
./p2p-bw 256 20
echo "== $(date '+%F %T') host memory bandwidth"
gcc -O3 -fopenmp -march=native -o membw membw.c && for t in 8 16 32; do OMP_NUM_THREADS=$t OMP_PROC_BIND=spread ./membw 2 10; done
