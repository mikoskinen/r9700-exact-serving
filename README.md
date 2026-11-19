# DeepSeek V4.1 Flash and Qwen Flash-Next (EXL3) on 4x AMD Radeon AI PRO R9700

This repository is written for AI coding agents more than for people. Hand it to your agent when you
set up or tune a similar system.

Update 15.11.2026: DeepSeek splits every routed expert across the four cards and keeps 4.2 full
1M-token sessions in its KV cache, and Qwen gives identical answers under load up to 32 concurrent
requests, long prompts included.

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

All values are tok/s, summed over all concurrent requests. Stock is the card default of 300 W with no
voltage offset, and the everyday setting caps each card at 210 W with a -50 mV voltage offset.
[Run it](#run-it) has the commands for both settings.

| Measure | DeepSeek, stock 300 W | DeepSeek, 210 W | Qwen, stock 300 W | Qwen, 210 W |
|---|---|---|---|---|
| 1 request | 107 prose / 187 code | 103 prose / 177 code | 269 | 265 |
| 4 requests | 247 | 227 | 635 | 607 |
| 8 requests | 305 | 282 | 849 | 810 |
| 16 requests | 346 | 319 | 1,123 | 1,075 |
| 24 requests | 375 | 342 | 1,249 | 1,183 |
| 32 requests | 415 | 378 | 1,345 | 1,284 |
| Prefill 16k | 4,480 | 4,640 | 9,420 | 9,090 |
| Prefill 64k | 5,980 | 5,800 | 9,540 | 9,220 |
| Prefill 128k | 6,330 | 5,810 | 9,400 | 9,090 |

## Run it

Both models listen on port 8090 and use all four cards, so you run one at a time.
[Model weights](#model-weights) has the download commands for the files the commands below mount.

### Host setup

The results ran on Ubuntu 26.04 with the kernel's own amdgpu driver and Docker. ROCm is inside the
images, so the host doesn't need it. The host needs:

- `amdgpu.ppfeaturemask=0xffffffff ttm.pages_limit=104857600` on the kernel command line, which allows
  the voltage offset and lets the driver pin up to 400 GiB of host memory
- 512 GB of RAM, because Qwen pins about 166 GiB and DeepSeek about 276 GiB of host memory
- the CPU settings in `gpu02/host/`, which turn off deep idle states, set the performance governor and
  turn off NUMA balancing. Without them, decode on this host was 15 to 18 % slower.

```bash
git clone https://github.com/mikoskinen/r9700-exact-serving.git
cd r9700-exact-serving
sudo install -m 755 gpu02/host/gpu02-host-tuning /usr/local/sbin/
sudo install -m 644 gpu02/host/gpu02-host-tuning.service /etc/systemd/system/
sudo install -m 644 gpu02/host/90-gpu02-numa.conf /etc/sysctl.d/
sudo systemctl daemon-reload && sudo systemctl enable --now gpu02-host-tuning.service && sudo sysctl --system
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
  ghcr.io/mikoskinen/r9700-exact-serving:qwen-4f74a8fa \
  --model /models/qwen3.8-flash-next-exl3.rad --served-model-name qwen3.8-flash-next \
  --host 127.0.0.1 --port 8090 --tp 4 --tp-wire exact --tp-wire-min-kb 128 \
  --max-num-seqs 32 --max-model-len 262144 --max-num-batched-tokens 8192 \
  --prefill-span 4 --prefill-interleave 0 --prefix-resume nearest \
  --placement all_vram --gpu-headroom-mib 108 --kv-cache-dtype fp8 --ngram-placement ram \
  --prefix-cache-mib 61440 --prefix-cache-ttl 28800 --checkpoint-slots 512 --checkpoint-policy keep-all \
  --num-speculative-tokens 3 --reasoning-effort medium --reasoning-effort-map high=xhigh \
  --thinking-token-budget 16000 --default-max-tokens 32768
run/switches.sh run/qwen/switches.json
```

The server is ready in about two minutes. [run/switches.sh](run/switches.sh) waits for it, then sets
the runtime switches of the batch-invariant kernel set and checks that each one took effect. The
result runs on gpu02 also set the rank order with `ROCR_VISIBLE_DEVICES`, putting the cards at PCI
bus 03, c3, 84 and 48 on ranks 0 to 3.

### Qwen for Claude Code

Pick the command by the number of concurrent Claude Code sessions and the context length you want. At
24 to 32 concurrent requests and at 524,288 tokens the KV cache is kept in host memory
(`--kv-resident`), which takes about 40 GiB of RAM a card.

**16 concurrent requests, 262,144-token context** (KV cache of 1,746,316 tokens a card):

```bash
docker run -d --name qwen --init --network host \
  --device /dev/kfd --device /dev/dri \
  --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) \
  --security-opt seccomp=unconfined --ulimit memlock=-1 --user $(id -u):$(id -g) \
  -v ~/models/flashnext-exl3:/models:ro --env-file run/qwen/claude-code.env \
  -e RADIANCE_FN4_RESUME_COLD=1 -e RADIANCE_FN4_PREEMPT_KEEP=1 -e RADIANCE_FN4_SP_PLE=1 \
  ghcr.io/mikoskinen/r9700-exact-serving:qwen-30903db6 \
  --model /models/qwen3.8-flash-next-exl3.rad --served-model-name qwen3.8-flash-next \
  --host 127.0.0.1 --port 8090 --tp 4 --tp-wire exact --tp-wire-min-kb 128 \
  --max-num-seqs 16 --max-model-len 262144 --max-num-batched-tokens 8192 \
  --prefill-span 4 --prefill-interleave 0 --prefix-resume nearest \
  --placement all_vram --gpu-headroom-mib 108 --kv-cache-dtype fp8 --ngram-placement ram \
  --prefix-cache-mib 61440 --prefix-cache-ttl 28800 --checkpoint-slots 512 --checkpoint-policy keep-all \
  --num-speculative-tokens 3 --reasoning-effort medium --reasoning-effort-map high=xhigh \
  --thinking-token-budget 16000 --default-max-tokens 32768 \
  --vision-tower-placement ram
run/switches.sh run/qwen/switches.json
```

**24 to 32 concurrent requests, 262,144-token context:**

```bash
docker run -d --name qwen --init --network host \
  --device /dev/kfd --device /dev/dri \
  --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) \
  --security-opt seccomp=unconfined --ulimit memlock=-1 --user $(id -u):$(id -g) \
  -v ~/models/flashnext-exl3:/models:ro --env-file run/qwen/claude-code.env \
  -e RADIANCE_FN4_RESUME_COLD=1 -e RADIANCE_FN4_PREEMPT_KEEP=1 -e RADIANCE_FN4_SP_PLE=1 \
  -e RADIANCE_FN4_KVS_BI=3 -e RADIANCE_FN4_KVS_APART=1 -e RADIANCE_FN4_KVS_VIC=1 \
  -e RADIANCE_FN4_KVS_CTX=6291456 \
  -e RADIANCE_FN4_QSA_SPLIT=1 -e RADIANCE_FN4_QSA_SPLIT_MIN=8192 \
  ghcr.io/mikoskinen/r9700-exact-serving:qwen-30903db6 \
  --model /models/qwen3.8-flash-next-exl3.rad --served-model-name qwen3.8-flash-next \
  --host 127.0.0.1 --port 8090 --tp 4 --tp-wire exact --tp-wire-min-kb 128 \
  --max-num-seqs 32 --max-model-len 262144 --max-num-batched-tokens 8192 \
  --prefill-span 4 --prefill-interleave 0 --prefix-resume nearest \
  --placement all_vram --gpu-headroom-mib 108 --kv-cache-dtype fp8 --ngram-placement ram \
  --prefix-cache-mib 61440 --prefix-cache-ttl 28800 --checkpoint-slots 512 --checkpoint-policy keep-all \
  --num-speculative-tokens 3 --reasoning-effort medium --reasoning-effort-map high=xhigh \
  --thinking-token-budget 16000 --default-max-tokens 32768 \
  --vision-tower-placement ram --kv-resident 32768
run/switches.sh run/qwen/switches.json
```

**16 concurrent requests, 524,288-token context:**

```bash
docker run -d --name qwen --init --network host \
  --device /dev/kfd --device /dev/dri \
  --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) \
  --security-opt seccomp=unconfined --ulimit memlock=-1 --user $(id -u):$(id -g) \
  -v ~/models/flashnext-exl3:/models:ro --env-file run/qwen/claude-code.env \
  -e RADIANCE_FN4_RESUME_COLD=1 -e RADIANCE_FN4_PREEMPT_KEEP=1 -e RADIANCE_FN4_SP_PLE=1 \
  -e RADIANCE_FN4_KVS_BI=3 -e RADIANCE_FN4_KVS_APART=1 -e RADIANCE_FN4_KVS_VIC=1 \
  -e RADIANCE_FN4_KVS_CTX=7077888 \
  ghcr.io/mikoskinen/r9700-exact-serving:qwen-30903db6 \
  --model /models/qwen3.8-flash-next-exl3.rad --served-model-name qwen3.8-flash-next \
  --host 127.0.0.1 --port 8090 --tp 4 --tp-wire exact --tp-wire-min-kb 128 \
  --max-num-seqs 16 --max-model-len 524288 --max-num-batched-tokens 8192 \
  --prefill-span 4 --prefill-interleave 0 --prefix-resume nearest \
  --placement all_vram --gpu-headroom-mib 108 --kv-cache-dtype fp8 --ngram-placement ram \
  --prefix-cache-mib 61440 --prefix-cache-ttl 28800 --checkpoint-slots 512 --checkpoint-policy keep-all \
  --num-speculative-tokens 3 --reasoning-effort medium --reasoning-effort-map high=xhigh \
  --thinking-token-budget 16000 --default-max-tokens 32768 \
  --vision-tower-placement ram --kv-resident 262144 --rope-scaling yarn:2
run/switches.sh run/qwen/switches.json
```

**24 to 32 concurrent requests, 524,288-token context:**

```bash
docker run -d --name qwen --init --network host \
  --device /dev/kfd --device /dev/dri \
  --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) \
  --security-opt seccomp=unconfined --ulimit memlock=-1 --user $(id -u):$(id -g) \
  -v ~/models/flashnext-exl3:/models:ro --env-file run/qwen/claude-code.env \
  -e RADIANCE_FN4_RESUME_COLD=1 -e RADIANCE_FN4_PREEMPT_KEEP=1 -e RADIANCE_FN4_SP_PLE=1 \
  -e RADIANCE_FN4_KVS_BI=3 -e RADIANCE_FN4_KVS_APART=1 -e RADIANCE_FN4_KVS_VIC=1 \
  -e RADIANCE_FN4_KVS_CTX=7077888 \
  ghcr.io/mikoskinen/r9700-exact-serving:qwen-30903db6 \
  --model /models/qwen3.8-flash-next-exl3.rad --served-model-name qwen3.8-flash-next \
  --host 127.0.0.1 --port 8090 --tp 4 --tp-wire exact --tp-wire-min-kb 128 \
  --max-num-seqs 32 --max-model-len 524288 --max-num-batched-tokens 8192 \
  --prefill-span 4 --prefill-interleave 0 --prefix-resume nearest \
  --placement all_vram --gpu-headroom-mib 108 --kv-cache-dtype fp8 --ngram-placement ram \
  --prefix-cache-mib 61440 --prefix-cache-ttl 28800 --checkpoint-slots 512 --checkpoint-policy keep-all \
  --num-speculative-tokens 3 --reasoning-effort medium --reasoning-effort-map high=xhigh \
  --thinking-token-budget 16000 --default-max-tokens 32768 \
  --vision-tower-placement ram --kv-resident 32768 --rope-scaling yarn:2
run/switches.sh run/qwen/switches.json
```

Measured with a Claude Code simulation: every session starts at once and runs 15 turns at
temperature 0. The times are for the turns after a session's first.

| Command | KV tokens a card | Sessions | Preemptions | Follow-up TTFT p50 / p90 | Decode a stream |
|---|---|---|---|---|---|
| 16 requests, 262,144 tokens | 1,746,316 | 16 | 0 | 1.07 / 2.60 s | 69.2 tok/s |
| 24 to 32 requests, 262,144 tokens | 6,291,456, streamed | 24 | 0 | 1.70 / 5.11 s | 53.0 tok/s |
| 24 to 32 requests, 262,144 tokens | 6,291,456, streamed | 32 | 0 | 1.99 / 4.61 s | 41.9 tok/s |
| 16 requests, 524,288 tokens | 7,077,888, streamed | 16 | 0 | 2.91 / 5.15 s | 72.6 tok/s |
| 24 to 32 requests, 524,288 tokens | 7,077,888, streamed | 24 | 0 | 3.07 / 6.86 s | 55.7 tok/s |
| 24 to 32 requests, 524,288 tokens | 7,077,888, streamed | 32 | 0 | 3.99 / 26.75 s | 29.9 tok/s |

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

For Qwen, use `"model": "qwen3.8-flash-next"`, and the answer is "The capital of Australia is
Canberra. It officially became the capital in 1913 when the city was formally named and established."
With `--rope-scaling yarn:2` at 524,288 tokens, Qwen answers "The capital of Australia is Canberra. It
officially became the capital in 1913 when the city was named and construction began."

## DeepSeek on two machines

Two of these machines, each with four R9700 cards, serve DeepSeek as one server over a 200 Gbit/s
RDMA link (a ConnectX-6 card in each machine). The model is split by layers: the leader runs layers
0 to 9 and 32 to 39 and the draft model and serves the API, and the follower runs layers 10 to 31.
Each machine runs its layers tensor parallel over its own four cards.

Both machines ran at 210 W a card with a -50 mV voltage offset. Values are tok/s summed over all
concurrent requests, and prefill is prompt tokens divided by the time to first token, which is in
brackets. The one-machine values are the DeepSeek 210 W column of [Results](#results).

| Measure | Two machines | One machine | Two / one |
|---|---|---|---|
| 1 request | 104 prose / 209 code | 103 prose / 177 code | 1.01x / 1.18x |
| 4 requests | 379 | 227 | 1.67x |
| 8 requests | 484 | 282 | 1.71x |
| 16 requests | 620 | 319 | 1.94x |
| 24 requests | 695 | 342 | 2.03x |
| 32 requests | 749 | 378 | 1.98x |
| Prefill 16k | 9,985 (1.58 s) | 4,640 (3.40 s) | 2.15x |
| Prefill 64k | 12,529 (5.24 s) | 5,800 (11.32 s) | 2.16x |
| Prefill 128k | 12,920 (10.15 s) | 5,810 (22.58 s) | 2.22x |

Both machines need the [host setup](#host-setup) with `sudo run/power.sh 210`, the files from
[Model weights](#model-weights) in `MODEL_DIR`, and an address on the RDMA link: `LEADER_IP` on the
leader and `FOLLOWER_IP` on the follower. Start the follower first:

```bash
mkdir -p /dev/shm/ds41ng
docker run -d --name deepseek-fol --init --network host \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --user $(id -u):$(id -g) \
  -v "$MODEL_DIR":/models:ro -v /dev/shm/ds41ng:/ngram -v "$PWD/run/deepseek":/cfg:ro \
  --env-file run/deepseek/deepseek.env -e ROCR_VISIBLE_DEVICES=0,1,2,3 \
  --env-file run/deepseek-tp8/follower.env \
  ghcr.io/mikoskinen/r9700-exact-serving:deepseek-tp8-5201ad92 \
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
  --env-file run/deepseek/deepseek.env -e ROCR_VISIBLE_DEVICES=0,1,2,3 \
  --env-file run/deepseek-tp8/leader.env \
  ghcr.io/mikoskinen/r9700-exact-serving:deepseek-tp8-5201ad92 \
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

The pair is ready in about five minutes, and the API is on the leader's port 8090. The switches go
to the leader, which passes them on to the follower.

## What exact means

Exact means the output is byte-identical to each model's reference engine, and a prompt gets the same
answer alone or next to other requests. This matters for a coding agent, whose answer to the same
context shouldn't change because other sessions are running on the same cards. DeepSeek's reference
is the build of 15.11.2026, and in a 33-minute run under load all 160 of its answers stay identical.
Qwen's answers to 20 short prompts and to 12 long prompts of up to 5,934 tokens stay identical beside
up to 31 other requests.

## Host tuning

Measurements are in [docs/hardware-tuning-gpu02.md](docs/hardware-tuning-gpu02.md).

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
- [docs/hardware-tuning-gpu02.md](docs/hardware-tuning-gpu02.md): the host, memory and PCIe tuning.
- [docs/LEVERS.md](docs/LEVERS.md): the working log of every change and its numbers.
- [docs/TESTING.md](docs/TESTING.md): the test method rules and the reasons for them.
- [registry/](registry/): one row per tested change, rejected ones included.
- [gpu02/](gpu02/): the test scripts, the host tuning scripts and the run summaries.
- [run/](run/): the power script, the environment files and the runtime switches that
  [Run it](#run-it) and [DeepSeek on two machines](#deepseek-on-two-machines) use.

The engine code is on the `deepseek/`, `qwen/` and `exl3/` branches. The DeepSeek image was built from
`deepseek/current` with `exl3/current`, and both Qwen images from `qwen/current` with `exl3/fn4-dual`
compiled with `-DEXL3R4D_DZ=2`. The two-machine image was built from `deepseek/ds41-pp-dev` with
`exl3/ds41-x7-route` compiled with `-DEXL3R4D_X6_SGB=1 -DEXL3R4D_X5_SUM=3` (HIP) and
`-DEXL3R4D_DK_MAX=64` (C). Every other branch is a single experiment or an earlier build, kept for
the record.

## License

- **CC BY 4.0** ([LICENSE-docs.md](LICENSE-docs.md)) covers the documents and data: `README.md`,
  `docs/`, `registry/` (except `registry/build.py`), and the run summaries and logs under
  `gpu02/exl3-test/runs/` and `gpu02/bench/runs/`.
- **Apache 2.0** ([LICENSE](LICENSE)) covers the scripts and tools: `registry/build.py`, everything under
  `gpu02/exl3-test/` outside `runs/`, `gpu02/host/` and `gpu02/bench/` outside `runs/`, and `run/`. The engine
  branches are forks of Apache 2.0 projects and keep their own LICENSE and NOTICE files.
