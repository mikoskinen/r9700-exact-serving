# DeepSeek V4.1 Flash and Qwen Flash-Next (EXL3) on 4x AMD Radeon AI PRO R9700

This repository is written for AI coding agents more than for people. Hand it to your agent when you
set up or tune a similar system.

This repository documents how two large mixture-of-experts models run locally in EXL3 format on four
AMD Radeon AI PRO R9700 cards as the backend for Claude Code. A prompt gets byte-identical output
whether it runs alone or beside other requests. The notes are written for people who run large MoE
models on AMD GPUs or EPYC hosts or tune a multi-GPU PCIe box, and for anyone who needs reproducible
output from a coding agent.

- **Models** (only these two were optimised)
  - DeepSeek V4.1 Flash, EXL3 2.25 bits per weight
  - Qwen Flash-Next, EXL3 4.05 bits per weight
- **Hardware and software**
  - 4x AMD Radeon AI PRO R9700 (RDNA4, 32 GB each) on PCIe 4.0 x16, one card per CPU root complex
  - AMD EPYC 7302P (Rome) on an ASRock Rack ROMED8-2T board
  - 512 GB of RAM (8 x 64 GB)
  - A C++/HIP fork of [radiance](https://codeberg.org/StillDeadcode/radiance) with the
    [radiance-exl3](https://github.com/Dyluhn/radiance-exl3) plugin, tensor parallel over the four cards
  - A LiteLLM proxy between Claude Code and the engine

## Results

Every number in this section is at stock power: 300 W a card with no voltage offset, set with
`sudo run/power.sh 300` after the [host setup](#host-setup).

### Qwen Flash-Next, one machine

| Measure | Value |
|---|---|
| 1 request | 295 tok/s |
| 4 requests | 656 tok/s |
| 8 requests | 919 tok/s |
| 16 requests | 1,222 tok/s |
| 24 requests | 1,360 tok/s |
| 32 requests | 1,467 tok/s |
| Prefill 16k (15,536 tokens) | 10,530 tok/s (1.48 s) |
| Prefill 64k (53,800 tokens) | 10,633 tok/s (5.06 s) |
| Prefill 128k (119,540 tokens) | 10,615 tok/s (11.26 s) |

Run settings, image `ghcr.io/mikoskinen/r9700-exact-serving:qwen-7d70dccc`:

```bash
sudo run/power.sh 300
docker run -d --name qwen --init --network host \
  --device /dev/kfd --device /dev/dri \
  --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) \
  --security-opt seccomp=unconfined --ulimit memlock=-1 --user $(id -u):$(id -g) \
  -v ~/models/flashnext-exl3:/models:ro --env-file run/qwen/qwen.env \
  -e RADIANCE_IMAGE_DEFAULTS=/opt/radiance/share/radiance/bench/defaults.env \
  -e RADIANCE_IMAGE_SWITCHES=/opt/radiance/share/radiance/bench/switches.json \
  ghcr.io/mikoskinen/r9700-exact-serving:qwen-7d70dccc \
  --model /models/qwen3.8-flash-next-exl3.rad --served-model-name qwen3.8-flash-next \
  --host 127.0.0.1 --port 8090 --tp 4 --tp-wire exact --tp-wire-min-kb 128 \
  --max-num-seqs 32 --max-model-len 262144 --max-num-batched-tokens 8192 \
  --prefill-span 4 --prefill-interleave 0 --prefix-resume exact \
  --placement all_vram --gpu-headroom-mib 108 --kv-cache-dtype fp8 --ngram-placement ram \
  --prefix-cache-mib 61440 --prefix-cache-ttl 28800 --checkpoint-slots 512 --checkpoint-policy keep-all \
  --num-speculative-tokens 4 --reasoning-effort medium --reasoning-effort-map high=xhigh \
  --thinking-token-budget 16000 --default-max-tokens 32768
```

[run/qwen/bench.env](run/qwen/bench.env) and [run/qwen/bench-switches.json](run/qwen/bench-switches.json)
list what the two `RADIANCE_IMAGE_*` variables select.

AgentX: [agentx-harness](https://github.com/SemiAnalysisAI/agentx-harness) 1.0.14, Claude Code trace set
`semianalysis_cc_traces_weka_062126_256k`, 32 concurrent sessions, thinking on, one warm-up request a
session, then a 20-minute window.

| Measure | Value |
|---|---|
| Output speed of a request, p50 | 22.72 tok/s |
| Output, all requests | 457.4 tok/s |
| Requests in the window | 905 |
| Time to first token p50 / p90 / p99 | 0.65 / 3.40 / 10.55 s |
| Time between output tokens p50 / p90 / p99 | 44.0 / 75.9 / 125.8 ms |
| Prompt tokens read from the prefix cache | 94.36 % |
| Prompt / output tokens a request, mean | 82,949 / 622 |
| Failed requests | 0 |

Run settings, image `ghcr.io/mikoskinen/r9700-exact-serving:qwen-7d70dccc`, server:

```bash
sudo run/power.sh 300
docker run -d --name qwen --init --network host \
  --device /dev/kfd --device /dev/dri \
  --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) \
  --security-opt seccomp=unconfined --ulimit memlock=-1 --user $(id -u):$(id -g) \
  -v ~/models/flashnext-exl3:/models:ro \
  -e RADIANCE_DRAFT_FORCE_AL=1:1.65,2:2.06,3:2.32,4:2.47 -e RADIANCE_SPEC_DECODE_METRICS=1 \
  ghcr.io/mikoskinen/r9700-exact-serving:qwen-7d70dccc \
  --model /models/qwen3.8-flash-next-exl3.rad --served-model-name qwen3.8-flash-next \
  --host 0.0.0.0 --port 8090 --tp 4 --tp-wire exact --tp-wire-min-kb 128 \
  --max-num-seqs 32 --max-model-len 262144 --max-num-batched-tokens 4096 \
  --prefill-span 2 --prefill-interleave 0 --prefix-resume exact \
  --placement all_vram --gpu-headroom-mib 108 --kv-cache-dtype fp8 --ngram-placement ram \
  --prefix-cache-mib 61440 --prefix-cache-ttl 28800 --checkpoint-slots 1024 --checkpoint-policy keep-all \
  --num-speculative-tokens 4 --reasoning-effort medium --reasoning-effort-map high=xhigh \
  --thinking-token-budget 16000 --default-max-tokens 32768 \
  --keep-alive-timeout 1800 --vision-tower-placement ram --kv-resident 32768
```

Client:

```bash
AIPERF_HTTP_KEEPALIVE_TIMEOUT=300 agentx profile --url http://SERVER:8090 --endpoint /v1/chat/completions \
  --model qwen3.8-flash-next --tokenizer TOKENIZER_DIR --concurrency 32 \
  --public-dataset semianalysis_cc_traces_weka_062126_256k --benchmark-duration 1200 \
  --warmup-requests-per-lane 1 --extra-inputs '{"chat_template_kwargs":{"enable_thinking":true}}'
```

### DeepSeek V4.1 Flash, two machines

| Measure | Value |
|---|---|
| 1 request, prose / code | 114 / 233 tok/s |
| 4 requests | 482 tok/s |
| 8 requests | 619 tok/s |
| 16 requests | 871 tok/s |
| 24 requests | 935 tok/s |
| 32 requests | 998 tok/s |
| Prefill 16k (15,777 tokens) | 13,258 tok/s (1.19 s) |
| Prefill 64k (65,650 tokens) | 16,964 tok/s (3.87 s) |
| Prefill 128k (131,137 tokens) | 17,532 tok/s (7.48 s) |

Run settings, image `ghcr.io/mikoskinen/r9700-exact-serving:deepseek-tp8-6b0bda62`, on each machine
after the [host setup](#host-setup) and the [model weights](#model-weights), with the sidecars from
[run/deepseek-tp8/mk-twin-sidecar.sh](run/deepseek-tp8/mk-twin-sidecar.sh) and then
[run/deepseek-tp8/mk-w8s-sidecar.sh](run/deepseek-tp8/mk-w8s-sidecar.sh) in `MODEL_DIR`, and an
address each on the RDMA link between the machines (`LEADER_IP`, `FOLLOWER_IP`). The follower first:

```bash
sudo run/power.sh 300
mkdir -p /dev/shm/ds41ng
docker run -d --name deepseek-fol --init --network host \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --user $(id -u):$(id -g) \
  -v "$MODEL_DIR":/models:ro -v /dev/shm/ds41ng:/ngram -v "$PWD/run/deepseek":/cfg:ro \
  --env-file run/deepseek-tp8/claude-code.env --env-file run/deepseek-tp8/claude-code-follower.env \
  -e ROCR_VISIBLE_DEVICES=0,1,2,3 \
  ghcr.io/mikoskinen/r9700-exact-serving:deepseek-tp8-6b0bda62 \
  --model /models/deepseek-v4.1-flash-1017-exl3-2.25bpw.rad --mm-max-patches 0 \
  --served-model-name deepseek-v4.1-flash --tp 4 --tp-wire exact \
  --placement expert_tiered --host-pool-mib 0 --ngram-placement ram=/ngram \
  --max-num-seqs 32 --max-model-len 1048576 --max-num-batched-tokens 4160 \
  --kv-cache-dtype bf16 --checkpoint-interval 2048 --num-speculative-tokens 5 \
  --prefix-cache-mib 86016 --prefix-cache-ttl 28800 --checkpoint-slots 512 --checkpoint-policy keep-all \
  --prefill-span 2 --prefill-interleave 8 --live --draft-window adaptive --draft-coupling on \
  --gpu-headroom-mib 256 --pp-lanes 4 --host 127.0.0.1 --port 8091 --keep-alive-timeout 1800 \
  --pp-nodes 2 --pp-cut 10,32 --pp-peer "$LEADER_IP":29517 --node-link verbs --pp-rank 1
```

Then the leader:

```bash
sudo run/power.sh 300
mkdir -p /dev/shm/ds41ng
docker run -d --name deepseek --init --network host \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --user $(id -u):$(id -g) \
  -v "$MODEL_DIR":/models:ro -v /dev/shm/ds41ng:/ngram -v "$PWD/run/deepseek":/cfg:ro \
  --env-file run/deepseek-tp8/claude-code.env --env-file run/deepseek-tp8/claude-code-leader.env \
  -e ROCR_VISIBLE_DEVICES=0,1,2,3 \
  ghcr.io/mikoskinen/r9700-exact-serving:deepseek-tp8-6b0bda62 \
  --model /models/deepseek-v4.1-flash-1017-exl3-2.25bpw.rad --mm-max-patches 0 \
  --served-model-name deepseek-v4.1-flash --tp 4 --tp-wire exact \
  --placement expert_tiered --host-pool-mib 0 --ngram-placement ram=/ngram \
  --max-num-seqs 32 --max-model-len 1048576 --max-num-batched-tokens 4160 \
  --kv-cache-dtype bf16 --checkpoint-interval 2048 --num-speculative-tokens 5 \
  --prefix-cache-mib 86016 --prefix-cache-ttl 28800 --checkpoint-slots 512 --checkpoint-policy keep-all \
  --prefill-span 2 --prefill-interleave 8 --live --draft-window adaptive --draft-coupling on \
  --gpu-headroom-mib 256 --pp-lanes 4 --host 127.0.0.1 --port 8090 --keep-alive-timeout 1800 \
  --pp-nodes 2 --pp-cut 10,32 --pp-peer "$FOLLOWER_IP":29517 --node-link verbs --pp-rank 0 \
  --draft-short-from 12
run/switches.sh run/deepseek-tp8/claude-code-switches.json
curl -s -X POST http://127.0.0.1:8090/reset_prefix_cache
```

AgentX: [agentx-harness](https://github.com/SemiAnalysisAI/agentx-harness) 1.0.14, Claude Code trace set
`semianalysis_cc_traces_weka_062126` (contexts up to 990k tokens), 32 concurrent sessions, thinking on,
one warm-up request a session, then a 20-minute window.

| Measure | Value |
|---|---|
| Output speed of a request, p50 | 24.59 tok/s |
| Output, all requests | 490.2 tok/s |
| Requests in the window | 833 |
| Time to first token p50 / p90 / p99 | 0.83 / 3.46 / 18.16 s |
| Time between output tokens p50 / p90 / p99 | 40.7 / 80.7 / 127.6 ms |
| Prompt tokens read from the prefix cache | 94.23 % |
| Prompt / output tokens a request, mean | 100,750 / 724 |
| Failed requests | 0 |

Run settings, image `ghcr.io/mikoskinen/r9700-exact-serving:deepseek-tp8-6b0bda62`, the same machines and
files. The follower runs the follower command above with
`--env-file run/deepseek-tp8/claude-code-agentx-follower.env` after the other two `--env-file` options. The
leader:

```bash
sudo run/power.sh 300
mkdir -p /dev/shm/ds41ng
docker run -d --name deepseek --init --network host \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --user $(id -u):$(id -g) \
  -v "$MODEL_DIR":/models:ro -v /dev/shm/ds41ng:/ngram -v "$PWD/run/deepseek":/cfg:ro \
  --env-file run/deepseek-tp8/claude-code.env --env-file run/deepseek-tp8/claude-code-leader.env \
  --env-file run/deepseek-tp8/claude-code-agentx-leader.env -e ROCR_VISIBLE_DEVICES=0,1,2,3 \
  ghcr.io/mikoskinen/r9700-exact-serving:deepseek-tp8-6b0bda62 \
  --model /models/deepseek-v4.1-flash-1017-exl3-2.25bpw.rad --mm-max-patches 0 \
  --served-model-name deepseek-v4.1-flash --tp 4 --tp-wire exact \
  --placement expert_tiered --host-pool-mib 0 --ngram-placement ram=/ngram \
  --max-num-seqs 32 --max-model-len 1048576 --max-num-batched-tokens 4160 \
  --kv-cache-dtype bf16 --checkpoint-interval 2048 --num-speculative-tokens 5 \
  --prefix-cache-mib 86016 --prefix-cache-ttl 28800 --checkpoint-slots 512 --checkpoint-policy keep-all \
  --prefill-span 2 --prefill-interleave 8 --live --draft-window adaptive --draft-coupling on \
  --gpu-headroom-mib 256 --pp-lanes 4 --host 127.0.0.1 --port 8090 --keep-alive-timeout 1800 \
  --pp-nodes 2 --pp-cut 10,32 --pp-peer "$FOLLOWER_IP":29517 --node-link verbs --pp-rank 0 \
  --draft-short-from 12
run/switches.sh run/deepseek-tp8/claude-code-switches.json
curl -s -X POST http://127.0.0.1:8090/reset_prefix_cache
```

Client:

```bash
AIPERF_HTTP_KEEPALIVE_TIMEOUT=300 agentx profile --url http://127.0.0.1:8090 --endpoint /v1/chat/completions \
  --model deepseek-v4.1-flash --tokenizer TOKENIZER_DIR --concurrency 32 \
  --public-dataset semianalysis_cc_traces_weka_062126 --benchmark-duration 1200 \
  --warmup-requests-per-lane 1 --extra-inputs '{"chat_template_kwargs":{"thinking":true,"reasoning_effort":"high"}}'
```

## One machine at both power settings

All values are tok/s, summed over all concurrent requests. Stock is the card default of 300 W with no
voltage offset, and the everyday setting caps each card at 210 W with a -50 mV voltage offset.
[Run it](#run-it) has the commands for both settings.

| Measure | DeepSeek, stock 300 W | DeepSeek, 210 W | Qwen, stock 300 W | Qwen, 210 W |
|---|---|---|---|---|
| 1 request | 107 prose / 187 code | 103 prose / 177 code | 295 | 283 |
| 4 requests | 247 | 227 | 654 | 611 |
| 8 requests | 305 | 282 | 900 | 859 |
| 16 requests | 346 | 319 | 1,163 | 1,108 |
| 24 requests | 375 | 342 | 1,289 | 1,232 |
| 32 requests | 415 | 378 | 1,386 | 1,326 |
| Prefill 16k | 4,480 | 4,640 | 10,500 | 10,220 |
| Prefill 64k | 5,980 | 5,800 | 10,470 | 10,200 |
| Prefill 128k | 6,330 | 5,810 | 10,310 | 10,040 |

## Run it

Both models listen on port 8090 and use all four cards, so you run one at a time.
[Model weights](#model-weights) has the download commands for the files the commands below mount.

### Host setup

The results ran on Ubuntu 26.04 with the kernel's own amdgpu driver and Docker. ROCm is inside the
images, so the host doesn't need it. The host needs:

- `amdgpu.ppfeaturemask=0xffffffff ttm.pages_limit=104857600` on the kernel command line, which allows
  the voltage offset and lets the driver pin up to 400 GiB of host memory
- 512 GB of RAM, because Qwen pins about 166 GiB and DeepSeek about 276 GiB of host memory
- the CPU settings in `run/host/`, which turn off deep idle states, set the performance governor and
  turn off NUMA balancing. Without them, decode on this host was 15 to 18 % slower.

```bash
git clone https://github.com/mikoskinen/r9700-exact-serving.git
cd r9700-exact-serving
sudo install -m 755 run/host/r9700-host-tuning /usr/local/sbin/
sudo install -m 644 run/host/r9700-host-tuning.service /etc/systemd/system/
sudo install -m 644 run/host/90-r9700-numa.conf /etc/sysctl.d/
sudo systemctl daemon-reload && sudo systemctl enable --now r9700-host-tuning.service && sudo sysctl --system
sudo run/power.sh 300    # or 210 for the 210 W columns
```

[run/power.sh](run/power.sh) sets the power cap, the voltage offset and a flat fan curve on every card
until the next reboot. The cards cooled to 45 °C or less before each DeepSeek section and before each Qwen column.

### Qwen

```bash
docker run -d --name qwen --init --network host \
  --device /dev/kfd --device /dev/dri \
  --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) \
  --security-opt seccomp=unconfined --ulimit memlock=-1 --user $(id -u):$(id -g) \
  -v ~/models/flashnext-exl3:/models:ro --env-file run/qwen/qwen.env \
  -e RADIANCE_FN4_LANE_EAGER=1 -e RADIANCE_SCHED_PUB_PROMPT=1 -e RADIANCE_FN4_X3GD_HI=32 \
  -e RADIANCE_FN4_CHAIN=1 -e RADIANCE_FN4_SP_PLE=1 -e RADIANCE_FN4_GNQ8=1 -e RADIANCE_FN4_RESUME_COLD=1 \
  -e RADIANCE_SCHED_TAIL_GRID=2048 -e RADIANCE_FN4_DRAFT_UNI=1 -e RADIANCE_FN4_P2X=1 \
  -e RADIANCE_FN4_DRZ_KEEP=prot -e RADIANCE_FN4_MSP=1 -e RADIANCE_FN4_TIP=1 \
  -e RADIANCE_FN4_QSA_SPLIT=1 -e RADIANCE_FN4_QSA_SPLIT_MIN=8192 \
  ghcr.io/mikoskinen/r9700-exact-serving:qwen-ff94ef27 \
  --model /models/qwen3.8-flash-next-exl3.rad --served-model-name qwen3.8-flash-next \
  --host 127.0.0.1 --port 8090 --tp 4 --tp-wire exact --tp-wire-min-kb 128 \
  --max-num-seqs 32 --max-model-len 262144 --max-num-batched-tokens 8192 \
  --prefill-span 4 --prefill-interleave 0 --prefix-resume exact \
  --placement all_vram --gpu-headroom-mib 108 --kv-cache-dtype fp8 --ngram-placement ram \
  --prefix-cache-mib 61440 --prefix-cache-ttl 28800 --checkpoint-slots 512 --checkpoint-policy keep-all \
  --num-speculative-tokens 4 --reasoning-effort medium --reasoning-effort-map high=xhigh \
  --thinking-token-budget 16000 --default-max-tokens 32768
run/switches.sh run/qwen/switches-ff94ef27.json
```

The server is ready in about two minutes. [run/switches.sh](run/switches.sh) waits for it, then sets
the runtime switches of the batch-invariant kernel set and checks that each one took effect. The
result runs also set the rank order with `ROCR_VISIBLE_DEVICES`, putting the cards at PCI
bus 03, c4, 83 and 48 on ranks 0 to 3.

### Qwen for Claude Code

This command is for 24 to 32 concurrent Claude Code sessions with up to 262,144 tokens of context
each. Image `qwen-7d70dccc` carries its start environment and runtime switches as defaults, so the
command needs no environment file and no switch call.
[run/qwen/claude-code.env](run/qwen/claude-code.env) and [run/qwen/switches.json](run/qwen/switches.json)
list what the image sets.

```bash
docker run -d --name qwen --init --network host \
  --device /dev/kfd --device /dev/dri \
  --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) \
  --security-opt seccomp=unconfined --ulimit memlock=-1 --user $(id -u):$(id -g) \
  -v ~/models/flashnext-exl3:/models:ro \
  ghcr.io/mikoskinen/r9700-exact-serving:qwen-7d70dccc \
  --model /models/qwen3.8-flash-next-exl3.rad --served-model-name qwen3.8-flash-next \
  --host 127.0.0.1 --port 8090 --tp 4 --tp-wire exact --tp-wire-min-kb 128 \
  --max-num-seqs 32 --max-model-len 262144 --max-num-batched-tokens 4096 \
  --prefill-span 2 --prefill-interleave 0 --prefix-resume exact \
  --placement all_vram --gpu-headroom-mib 108 --kv-cache-dtype fp8 --ngram-placement ram \
  --prefix-cache-mib 61440 --prefix-cache-ttl 28800 --checkpoint-slots 1024 --checkpoint-policy keep-all \
  --num-speculative-tokens 4 --reasoning-effort medium --reasoning-effort-map high=xhigh \
  --thinking-token-budget 16000 --default-max-tokens 32768 \
  --keep-alive-timeout 1800 --vision-tower-placement ram --kv-resident 32768
```

The server is ready in about two minutes. The KV cache is kept in host memory and streamed to the
cards. It holds 7,077,888 tokens, which is 27 full-length sessions, and the engine uses about 346 GiB
of host memory. [Results](#results) has its AgentX numbers at 300 W.

At temperature 0 this command gives each prompt the same answer alone and beside 31 other requests
while new prompts of 14,672 tokens arrive: 20 short prompts, four prompts of 945 to 1,001 tokens, and
four prompts of 9,153 to 73,602 tokens, both cold and as prefix-cache hits.

### DeepSeek

```bash
mkdir -p /dev/shm/ds41ng
docker run -d --name deepseek --init --network host \
  --device /dev/kfd --device /dev/dri \
  --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) \
  --security-opt seccomp=unconfined --ulimit memlock=-1 --user $(id -u):$(id -g) \
  -v ~/models/ds41-exl3:/models:ro -v /dev/shm/ds41ng:/ngram -v "$PWD/run/deepseek":/cfg:ro \
  --env-file run/deepseek/deepseek.env \
  ghcr.io/mikoskinen/r9700-exact-serving:deepseek-d3874235 \
  --model /models/deepseek-v4.1-flash-1017-exl3-2.25bpw.rad --vision-tower-placement ram \
  --served-model-name deepseek-v4.1-flash --host 127.0.0.1 --port 8090 --tp 4 --tp-wire exact \
  --placement expert_tiered --host-pool-mib 19456 --ngram-placement ram=/ngram \
  --max-num-seqs 32 --max-model-len 1048576 --max-num-batched-tokens 4160 \
  --kv-cache-dtype bf16 --checkpoint-interval 2048 --num-speculative-tokens 5 \
  --prefix-cache-mib 86016 --prefix-cache-ttl 28800 --checkpoint-slots 512 --checkpoint-policy keep-all \
  --prefill-span 2 --prefill-interleave 8 --live --draft-window adaptive --draft-coupling on \
  --gpu-headroom-mib 256
run/switches.sh run/deepseek/switches.json
```

The server is ready in about five minutes, most of which goes to writing the 103 GiB n-gram table to
`/dev/shm/ds41ng`. A request can use up to 1,048,576 tokens of context, and the KV cache has room
for 4,380,530 tokens, which is 4.2 full-length sessions at once.

### Check the answer

Both servers sample by default, so send `"temperature": 0` for exact output. DeepSeek answers this
request with "The capital of Australia is Canberra, and it officially became the capital in 1913 when
the city was formally named and construction began." in every result run:

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4.1-flash", "temperature": 0, "max_tokens": 600, "reasoning_effort": "none",
  "messages": [{"role": "user", "content": "What is the capital of Australia, and in which year did it officially become the capital? Answer in two sentences."}]}'
```

For Qwen, use `"model": "qwen3.8-flash-next"`, and the [Qwen](#qwen) command answers "The capital of
Australia is Canberra. It officially became the capital in 1913 when the city was formally named and
established."

## DeepSeek on two machines

Two of these machines, each with four R9700 cards, serve DeepSeek as one server over a 200 Gbit/s
RDMA link (a ConnectX-6 card in each machine). The model is split by layers: the leader runs layers
0 to 9 and 32 to 39 and the draft model and serves the API, and the follower runs layers 10 to 31.
Each machine runs its layers tensor parallel over its own four cards. Prompt rows use an 8-bit twin
of the dense weights of layers 0 to 39, and decode rows use the model file's own weights.

In the tables, cN is N concurrent requests, and its value is decode tok/s summed over the N
requests. 1 request is the decode tok/s of a single request on a prose prompt and on a code prompt.
Prefill is one cold prompt of 16k, 64k or 128k tokens, in prompt tokens divided by the time to first
token, with that time in brackets.

**210 W a card with a -50 mV voltage offset.** Two machines run image `deepseek-tp8-9061bc79` with
the commands below. The one-machine column is the DeepSeek 210 W column of
[One machine at both power settings](#one-machine-at-both-power-settings), which runs the
[DeepSeek](#deepseek) command.

| Measure | Two machines, tok/s | One machine, tok/s | Two / one |
|---|---|---|---|
| 1 request | 111.6 prose / 236.6 code | 103 prose / 177 code | 1.08x / 1.34x |
| c4 | 443.6 | 227 | 1.95x |
| c8 | 554.2 | 282 | 1.97x |
| c16 | 750.8 | 319 | 2.35x |
| c24 | 852.8 | 342 | 2.49x |
| c32 | 914.2 | 378 | 2.42x |
| Prefill 16k | 12,326 (1.28 s) | 4,640 (3.40 s) | 2.66x |
| Prefill 64k | 15,594 (4.21 s) | 5,800 (11.32 s) | 2.69x |
| Prefill 128k | 16,130 (8.13 s) | 5,810 (22.58 s) | 2.78x |

**[BetterBench](https://github.com/GGZ14/BetterBench) 0.6.1 at stock 300 W** on image
`deepseek-tp8-0735a8e2`, with its default settings (temperature 0.7, 20 runs a task category, 48
requests a concurrency level) and the server's default thinking. Decode is BetterBench's weighted
median over its task categories, concurrency N is tok/s summed over N concurrent requests, and prefill
is the median of cold prompts of each length. The reports are in
[results/betterbench/](results/betterbench/).

| BetterBench | Two machines | One machine | Two / one |
|---|---|---|---|
| Decode, 1 request | 161.0 tok/s | 153.3 tok/s | 1.05x |
| Time to first token p50, 1 request | 76 ms | 173 ms | 0.44x |
| Concurrency 1 / 2 / 4 / 8 | 149.8 / 262.8 / 394.6 / 509.8 tok/s | 139.3 / 197.5 / 252.5 / 302.1 tok/s | 1.08x / 1.33x / 1.56x / 1.69x |
| Prefill 1.5k / 5.9k / 11.8k tokens | 4,861 / 8,179 / 12,355 tok/s | 3,714 / 6,143 / 7,224 tok/s | 1.31x / 1.33x / 1.71x |
| Prefill 23.6k / 47.1k tokens | 14,810 / 15,752 tok/s | 7,222 / 8,043 tok/s | 2.05x / 1.96x |

Each machine needs the [host setup](#host-setup) with `sudo run/power.sh 210` (or `300`), the files
from [Model weights](#model-weights) in `MODEL_DIR`, and the twin sidecar `p30t-i8-twin.rad`, which
[run/deepseek-tp8/mk-twin-sidecar.sh](run/deepseek-tp8/mk-twin-sidecar.sh) writes into `MODEL_DIR`
from the model file. Run it once on each machine with no server running. The sidecar is 7.5 GB, the
script needs 12 GB free in `MODEL_DIR`, and it takes 5 to 7 minutes on 16 CPU cores:

```bash
MODEL_DIR=~/models/ds41-exl3 run/deepseek-tp8/mk-twin-sidecar.sh
```

The two machines also need an address each on the RDMA link: `LEADER_IP` on the leader and
`FOLLOWER_IP` on the follower. Start the follower first:

```bash
mkdir -p /dev/shm/ds41ng
docker run -d --name deepseek-fol --init --network host \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --user $(id -u):$(id -g) \
  -v "$MODEL_DIR":/models:ro -v /dev/shm/ds41ng:/ngram -v "$PWD/run/deepseek":/cfg:ro \
  --env-file run/deepseek-tp8/deepseek.env -e ROCR_VISIBLE_DEVICES=0,1,2,3 \
  --env-file run/deepseek-tp8/follower.env \
  ghcr.io/mikoskinen/r9700-exact-serving:deepseek-tp8-9061bc79 \
  --model /models/deepseek-v4.1-flash-1017-exl3-2.25bpw.rad --mm-max-patches 0 \
  --served-model-name deepseek-v4.1-flash --tp 4 --tp-wire exact \
  --placement expert_tiered --host-pool-mib 0 --ngram-placement ram=/ngram \
  --max-num-seqs 32 --max-model-len 1048576 --max-num-batched-tokens 4160 \
  --kv-cache-dtype bf16 --checkpoint-interval 2048 --num-speculative-tokens 5 \
  --prefix-cache-mib 86016 --prefix-cache-ttl 28800 --checkpoint-slots 512 --checkpoint-policy keep-all \
  --prefill-span 2 --prefill-interleave 8 --live --draft-window adaptive --draft-coupling on \
  --gpu-headroom-mib 256 --pp-lanes 4 --host 127.0.0.1 --port 8091 \
  --pp-nodes 2 --pp-cut 10,32 --pp-peer "$LEADER_IP":29517 --node-link verbs --pp-rank 1
```

Then the leader:

```bash
mkdir -p /dev/shm/ds41ng
docker run -d --name deepseek --init --network host \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --user $(id -u):$(id -g) \
  -v "$MODEL_DIR":/models:ro -v /dev/shm/ds41ng:/ngram -v "$PWD/run/deepseek":/cfg:ro \
  --env-file run/deepseek-tp8/deepseek.env -e ROCR_VISIBLE_DEVICES=0,1,2,3 \
  --env-file run/deepseek-tp8/leader.env \
  ghcr.io/mikoskinen/r9700-exact-serving:deepseek-tp8-9061bc79 \
  --model /models/deepseek-v4.1-flash-1017-exl3-2.25bpw.rad --mm-max-patches 0 \
  --served-model-name deepseek-v4.1-flash --tp 4 --tp-wire exact \
  --placement expert_tiered --host-pool-mib 0 --ngram-placement ram=/ngram \
  --max-num-seqs 32 --max-model-len 1048576 --max-num-batched-tokens 4160 \
  --kv-cache-dtype bf16 --checkpoint-interval 2048 --num-speculative-tokens 5 \
  --prefix-cache-mib 86016 --prefix-cache-ttl 28800 --checkpoint-slots 512 --checkpoint-policy keep-all \
  --prefill-span 2 --prefill-interleave 8 --live --draft-window adaptive --draft-coupling on \
  --gpu-headroom-mib 256 --pp-lanes 4 --host 127.0.0.1 --port 8090 \
  --pp-nodes 2 --pp-cut 10,32 --pp-peer "$FOLLOWER_IP":29517 --node-link verbs --pp-rank 0 \
  --draft-short-from 12
run/switches.sh run/deepseek-tp8/switches.json
```

The pair is ready about three minutes after the leader starts, and the API is on the leader's port
8090. The switches go to the leader, which passes them on to the follower. The KV cache holds
4,445,880 tokens, which is 4.2 full-length sessions.

One machine runs image `deepseek-tp8-0735a8e2` with the twin sidecar and is ready in four to six
minutes:

```bash
mkdir -p /dev/shm/ds41ng
docker run -d --name deepseek --init --network host \
  --device /dev/kfd --device /dev/dri \
  --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --user $(id -u):$(id -g) \
  -v "$MODEL_DIR":/models:ro -v /dev/shm/ds41ng:/ngram -v "$PWD/run/deepseek":/cfg:ro \
  --env-file run/deepseek-tp8/deepseek.env -e ROCR_VISIBLE_DEVICES=0,1,2,3 \
  --env-file run/deepseek-tp8/one-machine.env \
  ghcr.io/mikoskinen/r9700-exact-serving:deepseek-tp8-0735a8e2 \
  --model /models/deepseek-v4.1-flash-1017-exl3-2.25bpw.rad --mm-max-patches 0 \
  --served-model-name deepseek-v4.1-flash --tp 4 --tp-wire exact \
  --placement expert_tiered --host-pool-mib 19456 --ngram-placement ram=/ngram \
  --max-num-seqs 32 --max-model-len 1048576 --max-num-batched-tokens 4160 \
  --kv-cache-dtype bf16 --checkpoint-interval 2048 --num-speculative-tokens 5 \
  --prefix-cache-mib 86016 --prefix-cache-ttl 28800 --checkpoint-slots 512 --checkpoint-policy keep-all \
  --prefill-span 2 --prefill-interleave 8 --live --draft-window adaptive --draft-coupling on \
  --gpu-headroom-mib 1024 --host 127.0.0.1 --port 8090 --draft-short-from 12
run/switches.sh run/deepseek-tp8/one-machine-switches.json
```

One machine needs `--gpu-headroom-mib 1024` with the twin sidecar: at 256 MiB the driver keeps
moving memory and every decode step takes about 150 ms. The KV cache holds 2,247,140 tokens, which
is 2.1 full-length sessions. Drafting is off above 16 concurrent requests. The one-machine c4 to c16
cells ran with the switches in `one-machine-switches.json`. The one-machine cells for 1 request, c24,
c32 and prefill and the one-machine BetterBench run had `draft_occ` 13, `draft_dcap_n` 8,
`sched_tail_grid` 512, `draft_cost_age` 0 and `draft_head_cut` 0 in place of the file's values, and
started without `--draft-short-from 12`, `RADIANCE_DS41_DSPARK_SHORT=2` and `RADIANCE_DRAFT_HEAD_CUT=1`.

The two-machine commands answer the request in [Check the answer](#check-the-answer) with "The capital
of Australia is Canberra, and it officially became the capital in 1913 when the city was founded and
named." The one-machine command gives the answer shown there. The two-machine answers are
deterministic: a request gets the same answer alone, beside other requests and after it is preempted
and recomputed. They are the two-machine configuration's own answers, not the one-machine builds'
([What exact means](#what-exact-means)).

## What exact means

Exact means the output is byte-identical to each model's reference engine, and a prompt gets the same
answer alone or next to other requests. This matters for a coding agent, whose answer to the same
context shouldn't change because other sessions are running on the same cards. DeepSeek's reference
is the build of 15.11.2026, and in a 33-minute run under load all 160 of its answers stay identical.
Qwen's answers to 20 short prompts and to 12 long prompts of up to 5,934 tokens stay identical beside
up to 31 other requests.

DeepSeek on two machines is its own reference, and its answers differ from the one-machine builds'.
Its routed-expert, router and compressor kernels add their sums in another order than those builds,
and its prompt rows use the 8-bit twin that the [DeepSeek](#deepseek) command doesn't use. Its
answers to 20 test prompts are the same in five separate starts and beside 32 concurrent requests
(c32). With the KV cache cut to 626,480 tokens, 32 requests of 17,603 prompt tokens give the same 32
answers with 9 preemptions by recompute as without any, in two runs.

## Host tuning

Measurements are in [docs/hardware-tuning.md](docs/hardware-tuning.md).

- The machine had 4 x 32 GB and 4 x 64 GB DIMMs, which split the eight memory channels into two sets.
  With 8 x 64 GB all channels are in one interleave set, and CPU read bandwidth went from 35.9 to
  104 GB/s.
- Two cards in the same IO-die quadrant share a root port of about 36 GB/s. Three cards were moved onto
  risers so that each quadrant has one card, and four-card host-to-GPU copies went from 61.3 to
  105.3 GB/s.
- The CPU sometimes stayed in a deep idle state, and the server then ran about 16 % slower. Idle states
  with an exit latency of 100 µs or more are now disabled at boot, and the governor is set to
  `performance`.

## Engine changes

Each change leaves the output exact. [registry/](registry/) lists every tested change, including the
rejected ones.

- Qwen's kernels compute a request's rows the same way whatever else is in the step, which is what
  makes an answer independent of load. They are also 5.9 % faster per step at 16 concurrent requests
  than the kernels they replaced.
- The four-card all-reduce sends each card's data to all peers in one interleaved pass instead of one
  link at a time, and a 42 MB all-reduce takes 2,508 µs instead of 3,436.
- In Qwen prefill each card runs the hyper-connection stages on a quarter of the rows, which raised
  64k prefill from 8,293 to 8,754 tok/s.
- DeepSeek's layers 21 to 39 write almost no cache that later tokens read, so a cold prefill runs them
  only for the end of the prompt. The time to first token for a 136k-token prompt went from 50.5 to
  33.3 s.
- DeepSeek's spare VRAM holds expert weights instead of being kept free for swapping experts in and
  out, which raised throughput at 8 concurrent requests from 143.9 to 169.5 tok/s.

## Model weights

Qwen runs from a ready file, [Dyluhn/qwen3.8-flash-next-exl3](https://huggingface.co/Dyluhn/qwen3.8-flash-next-exl3)
(120.5 GB):

```bash
hf download Dyluhn/qwen3.8-flash-next-exl3 qwen3.8-flash-next-exl3.rad --local-dir ~/models/flashnext-exl3
cd ~/models/flashnext-exl3
echo "187ae8186cc1f2ba8bce96565592af7fc6f6ec91c9ece6b4d352eaf0432ee26f  qwen3.8-flash-next-exl3.rad" | sha256sum -c
```

DeepSeek runs from [mikoskinen/DeepSeek-V4.1-Flash-1017-exl3-2.25bpw-rad](https://huggingface.co/mikoskinen/DeepSeek-V4.1-Flash-1017-exl3-2.25bpw-rad)
(273.2 GB in six parts) and two small files that the engine loads with it. The download needs about
322 GB of free disk while the parts are joined:

```bash
hf download mikoskinen/DeepSeek-V4.1-Flash-1017-exl3-2.25bpw-rad --local-dir ~/models/ds41-exl3
cd ~/models/ds41-exl3
for p in deepseek-v4.1-flash-1017-exl3-2.25bpw.rad.part-0[0-5]; do
  cat "$p" >> deepseek-v4.1-flash-1017-exl3-2.25bpw.rad && rm "$p"
done
sha256sum -c SHA256SUMS
```

The DeepSeek file has the routed experts of the `2.25bpw` branch of
[turboderp/DeepSeek-V4.1-Flash-1017-exl3](https://huggingface.co/turboderp/DeepSeek-V4.1-Flash-1017-exl3),
and every other weight, the vision tower included, is converted from the official release
[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).
To build the file yourself, follow
[Build it yourself](https://huggingface.co/mikoskinen/DeepSeek-V4.1-Flash-1017-exl3-2.25bpw-rad#build-it-yourself)
on the model card, which uses the scripts and inputs in [run/deepseek/build/](run/deepseek/build/).

## Where things are

The `main` branch has:

- [docs/DETAILS.md](docs/DETAILS.md): the long version of this page, with a source for every number and
  the LiteLLM setup for Claude Code.
- [docs/hardware-tuning.md](docs/hardware-tuning.md): the host, memory and PCIe tuning.
- [docs/LEVERS.md](docs/LEVERS.md): the working log of every change and its numbers.
- [docs/TESTING.md](docs/TESTING.md): the test method rules and the reasons for them.
- [registry/](registry/): one row per tested change, rejected ones included.
- [node/](node/): the test scripts, the host tuning scripts and the run summaries.
- [run/](run/): the power script, the host tuning files, the environment files and the runtime switches
  that [Results](#results), [Run it](#run-it) and [DeepSeek on two machines](#deepseek-on-two-machines) use.
- [results/betterbench/](results/betterbench/): the BetterBench reports of
  [DeepSeek on two machines](#deepseek-on-two-machines).

The engine code is on the `deepseek/`, `qwen/` and `exl3/` branches. The DeepSeek image was built from
`deepseek/current` with `exl3/current`, the Qwen images from `qwen/qfn-on-ds` with `exl3/qfn-x3gd`
(engine commit ff94ef27 for the image in [Qwen](#qwen)). Images `qwen-7d70dccc` and
`deepseek-tp8-6b0bda62` in [Results](#results) are built from engine commits 7d70dccc and 6b0bda62,
which aren't on a branch of this repository. The two-machine image `deepseek-tp8-9061bc79` was
built from `deepseek/ds41-pp-dev-pf` with `exl3/ds41-swk` compiled with `-DEXL3R4D_X6_SGB=1 -DEXL3R4D_X5_SUM=3
-DEXL3R4D_SWK_DMOVE=2 -DEXL3R4D_X6_PRE=2 -DEXL3R4D_X15_INSUM=1 -DEXL3R4D_X17_PIN=1
-DEXL3R4D_X17_FORM=1` (HIP) and `-DEXL3R4D_DK_MAX=64` (C), and image `deepseek-tp8-0735a8e2` from
`deepseek/ds41-pp-dev` with `exl3/ds41-x7-route` compiled with `-DEXL3R4D_X6_SGB=1
-DEXL3R4D_X5_SUM=3` (HIP) and `-DEXL3R4D_DK_MAX=64` (C). Every other branch is a single experiment or
an earlier build, kept for the record.

## License

- **CC BY 4.0** ([LICENSE-docs.md](LICENSE-docs.md)) covers the documents and data: `README.md`,
  `docs/`, `registry/` (except `registry/build.py`), `results/`, and the run summaries and logs under
  `node/exl3-test/runs/` and `node/bench/runs/`.
- **Apache 2.0** ([LICENSE](LICENSE)) covers the scripts and tools: `registry/build.py`, everything under
  `node/exl3-test/` outside `runs/`, `node/host/` and `node/bench/` outside `runs/`, and `run/`. The engine
  branches are forks of Apache 2.0 projects and keep their own LICENSE and NOTICE files.
