# DeepSeek V4.1 Flash and Qwen Flash-Next (EXL3) on 4x AMD Radeon AI PRO R9700

This repository is written for AI coding agents more than for people. Hand it to your agent when you
set up or tune a similar system.

The code and the settings are optimised for Claude Code style use: long conversations, coding agents
and repeated tool use.

This repository documents how two large mixture-of-experts models run locally in EXL3 format on four
AMD Radeon AI PRO R9700 cards as the backend for Claude Code. A prompt gets byte-identical output
whether it runs alone or beside other requests. The notes are written for people who run large MoE
models on AMD GPUs or EPYC hosts or tune a multi-GPU PCIe box, and for anyone who needs reproducible
output from a coding agent.

- **Models** (only these two were optimised)
  - DeepSeek V4.1 Flash, EXL3 2.25 bits per weight
  - Qwen Flash-Next, EXL3 4.05 bits per weight
- **Hardware and software**
  - One node: 4x AMD Radeon AI PRO R9700 (RDNA4, 32 GB each) on PCIe 4.0 x16, one card per CPU root
    complex, an AMD EPYC 7302P (Rome) on an ASRock Rack ROMED8-2T board, 512 GB of RAM (8 x 64 GB)
  - Qwen Flash-Next runs on one node (4x R9700)
  - DeepSeek V4.1 Flash runs on one node (4x R9700) or on two such nodes (8x R9700) joined by an RDMA link, a
    ConnectX-6 card in each node
  - A C++/HIP fork of [radiance](https://codeberg.org/StillDeadcode/radiance) with the
    [radiance-exl3](https://github.com/Dyluhn/radiance-exl3) plugin, tensor parallel over the four cards
  - A LiteLLM proxy between Claude Code and the engine

## Results

Decode and prefill:

| Measure | Qwen Flash-Next, 4x R9700 | DeepSeek V4.1 Flash, 8x R9700 (2 nodes) |
|---|---|---|
| 1 request | 295 tok/s | 114 prose / 233 code tok/s |
| 4 requests | 656 tok/s | 482 tok/s |
| 8 requests | 919 tok/s | 619 tok/s |
| 16 requests | 1,222 tok/s | 871 tok/s |
| 24 requests | 1,360 tok/s | 935 tok/s |
| 32 requests | 1,467 tok/s | 998 tok/s |
| Prefill 16k | 10,530 tok/s | 13,258 tok/s |
| Prefill 64k | 10,633 tok/s | 16,964 tok/s |
| Prefill 128k | 10,615 tok/s | 17,532 tok/s |

AgentX, 32 sessions:

| Measure | Qwen Flash-Next, 4x R9700 | DeepSeek V4.1 Flash, 8x R9700 (2 nodes) |
|---|---|---|
| Requests in 20 minutes | 905 | 833 |
| Output, all requests | 457.4 tok/s | 490.2 tok/s |
| Output speed of a request, p50 | 22.72 tok/s | 24.59 tok/s |
| Time to first token p50 / p90 | 0.65 / 3.40 s | 0.83 / 3.46 s |
| Time between output tokens p50 | 44.0 ms | 40.7 ms |
| Prompt tokens read from the prefix cache | 94.36 % | 94.23 % |
| Prompt / output tokens a request, mean | 82,949 / 622 | 100,750 / 724 |

## Models

- Qwen Flash-Next, EXL3 4.05 bpw: [Dyluhn/qwen3.8-flash-next-exl3](https://huggingface.co/Dyluhn/qwen3.8-flash-next-exl3)
- DeepSeek V4.1 Flash, EXL3 2.25 bpw: [mikoskinen/DeepSeek-V4.1-Flash-1017-exl3-2.25bpw-rad](https://huggingface.co/mikoskinen/DeepSeek-V4.1-Flash-1017-exl3-2.25bpw-rad)

## Run

### Host setup

Ubuntu 26.04 with the kernel's amdgpu driver and Docker, 512 GB of RAM a node, and
`ttm.pages_limit=104857600` on the kernel command line.

```bash
git clone https://github.com/mikoskinen/r9700-exact-serving.git
cd r9700-exact-serving
```

### Qwen Flash-Next, 4x R9700

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

Ready in about two minutes. It uses about 346 GiB of host memory, and the KV cache holds 7,077,888 tokens.

### DeepSeek V4.1 Flash, 8x R9700 (2 nodes)

Once on each node:

```bash
export MODEL_DIR=~/models/ds41-exl3 LEADER_IP=LEADER_RDMA_ADDRESS FOLLOWER_IP=FOLLOWER_RDMA_ADDRESS
run/deepseek-tp8/mk-twin-sidecar.sh
run/deepseek-tp8/mk-w8s-sidecar.sh
```

Follower:

```bash
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

Leader:

```bash
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

The KV cache holds 20,502,774 tokens.

### DeepSeek V4.1 Flash, 4x R9700

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

Ready in about five minutes. The KV cache holds 4,380,530 tokens.

### Check the answer

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4.1-flash", "temperature": 0, "max_tokens": 600, "reasoning_effort": "none",
  "messages": [{"role": "user", "content": "What is the capital of Australia, and in which year did it officially become the capital? Answer in two sentences."}]}'
```

For Qwen Flash-Next, use `"model": "qwen3.8-flash-next"`. Both servers sample by default, so send `"temperature": 0` for exact output.

## Exact output

At temperature 0 a prompt gets the same answer every time, alone or next to other requests.

## More

- [docs/DETAILS.md](docs/DETAILS.md): the long version of this page, with a source for every number, the
  LiteLLM setup for Claude Code, the engine branches and the BetterBench results
- [docs/DETAILS.md#host-setup-and-tuning](docs/DETAILS.md#host-setup-and-tuning): the kernel parameters,
  CPU settings and power script the results ran with
- [docs/hardware-tuning.md](docs/hardware-tuning.md): the host, memory and PCIe tuning
- [docs/LEVERS.md](docs/LEVERS.md) and [registry/](registry/): every tested change and its numbers
- [docs/TESTING.md](docs/TESTING.md): the test method
- [run/](run/): the power script, the host tuning files, the environment files and the runtime switches

## License

- **CC BY 4.0** ([LICENSE-docs.md](LICENSE-docs.md)) covers the documents and data: `README.md`,
  `docs/`, `registry/` (except `registry/build.py`), `results/`, and the run summaries and logs under
  `node/exl3-test/runs/` and `node/bench/runs/`.
- **Apache 2.0** ([LICENSE](LICENSE)) covers the scripts and tools: `registry/build.py`, everything under
  `node/exl3-test/` outside `runs/`, `node/host/` and `node/bench/` outside `runs/`, and `run/`. The engine
  branches are forks of Apache 2.0 projects and keep their own LICENSE and NOTICE files.
