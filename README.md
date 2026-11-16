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
| Prefill 16k | 4,480 | 4,290 | 9,420 | 9,090 |
| Prefill 64k | 5,980 | 5,480 | 9,540 | 9,220 |
| Prefill 128k | 6,330 | 5,640 | 9,400 | 9,090 |

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

Pick the command by how many Claude Code requests run at once and how much context they need. At 24
to 32 concurrent requests the KV cache is kept in host memory (`--kv-resident`), which takes about
40 GiB of RAM a card.

**16 concurrent requests, 262,144-token context** (KV cache of 1,746,316 tokens a card):

```bash
docker run -d --name qwen --init --network host \
  --device /dev/kfd --device /dev/dri \
  --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) \
  --security-opt seccomp=unconfined --ulimit memlock=-1 --user $(id -u):$(id -g) \
  -v ~/models/flashnext-exl3:/models:ro --env-file run/qwen/claude-code.env \
  -e RADIANCE_FN4_RESUME_COLD=1 -e RADIANCE_FN4_PREEMPT_KEEP=1 \
  ghcr.io/mikoskinen/r9700-exact-serving:qwen-db846ee1 \
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
  -e RADIANCE_FN4_RESUME_COLD=1 -e RADIANCE_FN4_PREEMPT_KEEP=1 \
  -e RADIANCE_FN4_KVS_BI=3 -e RADIANCE_FN4_KVS_APART=1 -e RADIANCE_FN4_KVS_VIC=1 \
  -e RADIANCE_FN4_KVS_CTX=6291456 \
  ghcr.io/mikoskinen/r9700-exact-serving:qwen-db846ee1 \
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

**24 to 32 concurrent requests, 524,288-token context:**

```bash
docker run -d --name qwen --init --network host \
  --device /dev/kfd --device /dev/dri \
  --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) \
  --security-opt seccomp=unconfined --ulimit memlock=-1 --user $(id -u):$(id -g) \
  -v ~/models/flashnext-exl3:/models:ro --env-file run/qwen/claude-code.env \
  -e RADIANCE_FN4_RESUME_COLD=1 -e RADIANCE_FN4_PREEMPT_KEEP=1 \
  -e RADIANCE_FN4_KVS_BI=3 -e RADIANCE_FN4_KVS_APART=1 -e RADIANCE_FN4_KVS_VIC=1 \
  -e RADIANCE_FN4_KVS_CTX=7077888 \
  ghcr.io/mikoskinen/r9700-exact-serving:qwen-db846ee1 \
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

| Command | KV tokens a card | Sessions | Requests done | Preemptions | Follow-up TTFT p50 / p90 | Decode a stream |
|---|---|---|---|---|---|---|
| 24 to 32 requests, 262,144 tokens | 6,291,456, streamed | 24 | 360 of 360 | 0 | 1.70 / 5.11 s | 53.0 tok/s |
| 24 to 32 requests, 262,144 tokens | 6,291,456, streamed | 32 | 480 of 480 | 0 | 2.07 / 8.92 s | 35.1 tok/s |
| 24 to 32 requests, 524,288 tokens | 7,077,888, streamed | 24 | 345 of 360 | 0 | 3.07 / 6.86 s | 55.7 tok/s |
| 24 to 32 requests, 524,288 tokens | 7,077,888, streamed | 32 | 322 of 480 | 0 | 3.99 / 26.75 s | 29.9 tok/s |

The simulation stops sending new turns after 3,000 s, which ends the 524,288-token runs early. No
request fails in any run.

### DeepSeek

```bash
mkdir -p /dev/shm/ds41ng
docker run -d --name deepseek --init --network host \
  --device /dev/kfd --device /dev/dri \
  --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) \
  --security-opt seccomp=unconfined --ulimit memlock=-1 --user $(id -u):$(id -g) \
  -v ~/models/ds41-exl3:/models:ro -v /dev/shm/ds41ng:/ngram -v "$PWD/run/deepseek":/cfg:ro \
  --env-file run/deepseek/deepseek.env \
  ghcr.io/mikoskinen/r9700-exact-serving:deepseek-5f60dee1 \
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
for 4,380,530 tokens, which is 4.2 full-length sessions at once. The server also accepts pictures as
`image_url` parts with a data URL, and `--vision-tower-placement ram` keeps the vision tower in host
memory.

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

## What exact means

Exact means the output is byte-identical to each model's reference engine, and a prompt gets the same
answer alone or next to other requests. This matters for a coding agent, whose answer to the same
context shouldn't change because other sessions are running on the same cards. DeepSeek's reference
is the build of 15.11.2026, and in a 33-minute run under load all 160 of its answers stay identical.
Qwen's answers to 20 short prompts and to 12 long prompts of up to 5,934 tokens stay identical beside
up to 31 other requests.

## Hardware changes

Raw values are in [docs/hardware-tuning-gpu02.md](docs/hardware-tuning-gpu02.md).

- **Same-size DIMMs:** replacing mixed 4 x 32 GB and 4 x 64 GB DIMMs with 8 x 64 GB put all eight
  memory channels in one interleave set, which raised CPU read bandwidth from 35.9 to 104 GB/s and
  two-card host-to-GPU copies from 45.1 to 55.4 GB/s.
- **One card per IO-die quadrant:** moving three cards onto risers so that every quadrant has its own
  card raised four-card host-to-GPU copies from 61.3 to 105.3 GB/s, because two cards in one quadrant
  share a root port of about 36 GB/s.
- **CPU idle states and governor:** disabling every idle state with an exit latency of 100 µs or more
  and setting the `performance` governor at boot removed a random ~16 % slowdown (Qwen at 8 users went
  from ~630 to ~750 tok/s).
- **The CPU swap that didn't help:** an EPYC 7302P in place of the 7282 on the mixed DIMMs moved
  two-card copies from 44.2 to 45.1 GB/s and made a two-card Qwen server 5.7 % slower at 16 users,
  because the 2-channel memory regions were the limit.

## Software changes

Each of these changes keeps the output exact, and [registry/](registry/) lists every tested
change, the rejected ones included:

- **Batch-invariant kernels:** Qwen's batch-invariant kernel set makes each answer independent of the
  other requests in a step and also cuts the 16-user step time from 37.95 to 35.70 ms (-5.9 %) against
  the kernels it replaced.
- **Interleaved all-reduce push:** the exact four-card all-reduce now pushes each card's data to every
  peer in one interleaved pass instead of one link at a time, which cut a 42 MB all-reduce from 3,436
  to 2,508 µs (-27 %).
- **Prefill skip above the last cache source:** DeepSeek's layers 21 to 39 write almost no cache that
  later tokens read, so a cold prefill runs them only for the end of the prompt.
  The time to first token for a 136k-token prompt fell from 50.5 to 33.3 s (-34 %).
- **Expert residency:** giving DeepSeek's spare VRAM to resident expert weights instead of holding it
  back for swapping experts in and out raised 8-user throughput from 143.9 to 169.5 tok/s (+17.8 %).
- **Sequence-parallel prefill stages:** in Qwen prefill each card runs the hyper-connection stages on a
  quarter of the rows between a reduce-scatter and an all-gather, which raised 64k prefill from 8,293
  to 8,754 tok/s (+5.7 %).

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
  [Run it](#run-it) uses.

The engine code is on the `deepseek/`, `qwen/` and `exl3/` branches. The DeepSeek image was built from
`deepseek/current` with `exl3/current`, and both Qwen images from `qwen/current` with `exl3/fn4-dual`
compiled with `-DEXL3R4D_DZ=2`. Every other branch is a single experiment or an earlier build, kept
for the record.

## License

- **CC BY 4.0** ([LICENSE-docs.md](LICENSE-docs.md)) covers the documents and data: `README.md`,
  `docs/`, `registry/` (except `registry/build.py`), and the run summaries and logs under
  `gpu02/exl3-test/runs/` and `gpu02/bench/runs/`.
- **Apache 2.0** ([LICENSE](LICENSE)) covers the scripts and tools: `registry/build.py`, everything under
  `gpu02/exl3-test/` outside `runs/`, `gpu02/host/` and `gpu02/bench/` outside `runs/`, and `run/`. The engine
  branches are forks of Apache 2.0 projects and keep their own LICENSE and NOTICE files.
