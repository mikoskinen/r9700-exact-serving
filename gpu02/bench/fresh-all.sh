#!/bin/bash
# fresh-all.sh (2.11.) -- bw-fresh.sh in the rocm/dev container, then queue-tp2.sh (no pgrep wait, runs strictly in sequence).
# Usage: setsid nohup ~/gpu/bench/fresh-all.sh <tag> > ~/gpu/bench/runs/fresh-all-<tag>.log 2>&1 < /dev/null &
cd ~/gpu/bench; TAG=${1:?tag}
docker run --rm --name bw-fresh --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 --ipc=host \
  --security-opt seccomp=unconfined --ulimit memlock=-1 -v $HOME/gpu/bench:/w rocm/dev-ubuntu-24.04:latest bash /w/bw-fresh.sh \
  > runs/rocm-bw/gpu02-$TAG.txt 2>&1
echo "BW finished $(date +%T) -> runs/rocm-bw/gpu02-$TAG.txt"
./queue-tp2.sh
