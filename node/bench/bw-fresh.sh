#!/bin/bash
# bw-fresh.sh -- inside rocm/dev-ubuntu-24.04 (/w = ~/gpu/bench): all bandwidth checks in one go (TR bw/bw4, p2p-bw,
# dma-check pinned/register x SDMA 1/0, rocm-bandwidth-test -a / -A, host STREAM). Re-run after slot/BIOS changes.
cd /w
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
echo "== TR bw";  runs/rocm-bw/bw
echo "== TR bw4"; runs/rocm-bw/bw4
echo "== p2p-bw"; ./p2p-bw 256 20
for sdma in 1 0; do for mode in pinned register; do
  echo "== dma-check $mode HSA_ENABLE_SDMA=$sdma"; HSA_ENABLE_SDMA=$sdma ./dma-check $mode 1 10
done; done
echo "== rocm-bandwidth-test -a"; timeout 900 rbt/rocm-bandwidth-test -a 2>&1 | grep -vE '^\s*$'
echo "== rocm-bandwidth-test -A"; timeout 900 rbt/rocm-bandwidth-test -A 2>&1 | grep -vE '^\s*$' | tail -12
echo "== host STREAM"
for t in 8 16 32; do OMP_NUM_THREADS=$t OMP_PROC_BIND=spread ./membw 2 10; done
echo "BW DONE $(date +%T)"
