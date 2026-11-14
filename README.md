# DeepSeek V4.1 Flash and Qwen Flash-Next (EXL3) on 4x AMD Radeon AI PRO R9700

Update 14.11.2026: a new matrix kernel runs DeepSeek's dense layers on the RDNA4 matrix cores, and MTP
drafting stays on with up to 12 concurrent requests, which raises the DeepSeek results considerably.

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

All values are tok/s, summed over all concurrent users. Stock is the card default of 300 W with no
voltage offset, and the everyday setting caps each card at 210 W with a -50 mV voltage offset.
[Run it](#run-it) has the commands for both settings.

| Measure | DeepSeek, stock 300 W | DeepSeek, 210 W | Qwen, stock 300 W | Qwen, 210 W |
|---|---|---|---|---|
| 1 user | 105 prose / 182 code | 101 prose / 173 code | 269 | 262 |
| 4 users | 201 | 192 | 618 | 583 |
| 8 users | 246 | 234 | 833 | 779 |
| 16 users | 294 | 277 | 1,093 | 1,027 |
| Prefill 16k | 4,040 | 3,910 | 9,540 | 9,100 |
| Prefill 64k | 6,120 | 5,600 | 9,810 | 9,340 |
| Prefill 128k | 6,390 | 5,800 | 9,700 | 9,240 |

## What exact means

Exact means the output is byte-identical to each model's reference engine, and a prompt gets the same
answer alone or next to other requests. This matters for a coding agent, whose answer to the same
context shouldn't change because other sessions are running on the same cards. In a 38-minute run
under load, all 160 of DeepSeek's answers stay identical.

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
(272.3 GB in six parts) and two small files that the engine loads with it. The download needs about
320 GB of free disk while the parts are joined:

```bash
hf download mikoskinen/DeepSeek-V4.1-Flash-1017-exl3-2.25bpw-rad --local-dir ~/models/ds41-exl3
cd ~/models/ds41-exl3
for p in deepseek-v4.1-flash-1017-exl3-2.25bpw.rad.part-0[0-5]; do
  cat "$p" >> deepseek-v4.1-flash-1017-exl3-2.25bpw.rad && rm "$p"
done
sha256sum -c SHA256SUMS
```

The DeepSeek file was converted from the routed experts of the `2.25bpw` branch of
[turboderp/DeepSeek-V4.1-Flash-1017-exl3](https://huggingface.co/turboderp/DeepSeek-V4.1-Flash-1017-exl3)
and every other weight of [bakon3/DeepSeek-V4.1-Flash-1017-uncensored-rad](https://huggingface.co/bakon3/DeepSeek-V4.1-Flash-1017-uncensored-rad),
an uncensored build of the same model.

## Run it

Both models listen on port 8090 and use all four cards, so you run one at a time.

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
until the next reboot. Before each measurement, the cards cooled to 45 °C or less.

### Qwen

```bash
docker run -d --name qwen --init --network host \
  --device /dev/kfd --device /dev/dri \
  --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) \
  --security-opt seccomp=unconfined --ulimit memlock=-1 --user $(id -u):$(id -g) \
  -v ~/models/flashnext-exl3:/models:ro --env-file run/qwen/qwen.env \
  ghcr.io/mikoskinen/r9700-exact-serving:qwen-b9282bf7 \
  --model /models/qwen3.8-flash-next-exl3.rad --served-model-name qwen3.8-flash-next \
  --host 127.0.0.1 --port 8090 --tp 4 --tp-wire exact --tp-wire-min-kb 128 \
  --max-num-seqs 16 --max-model-len 262144 --max-num-batched-tokens 8192 \
  --prefill-span 4 --prefill-interleave 0 --prefix-resume nearest \
  --placement all_vram --gpu-headroom-mib 96 --kv-cache-dtype fp8 --ngram-placement ram \
  --prefix-cache-mib 61440 --prefix-cache-ttl 28800 --checkpoint-slots 512 --checkpoint-policy keep-all \
  --num-speculative-tokens 3 --reasoning-effort medium --reasoning-effort-map high=xhigh \
  --thinking-token-budget 16000 --default-max-tokens 32768
run/switches.sh run/qwen/switches.json
```

The server is ready in about two minutes. [run/switches.sh](run/switches.sh) waits for it, then sets
the runtime switches of the batch-invariant kernel set and checks that each one took effect. The
result runs on gpu02 also set the rank order with `ROCR_VISIBLE_DEVICES`, putting the cards at PCI
bus 03, c3, 83 and 48 on ranks 0 to 3.

### DeepSeek

```bash
mkdir -p /dev/shm/ds41ng
docker run -d --name deepseek --init --network host \
  --device /dev/kfd --device /dev/dri \
  --group-add $(getent group render | cut -d: -f3) --group-add $(getent group video | cut -d: -f3) \
  --security-opt seccomp=unconfined --ulimit memlock=-1 --user $(id -u):$(id -g) \
  -v ~/models/ds41-exl3:/models:ro -v /dev/shm/ds41ng:/ngram -v "$PWD/run/deepseek":/cfg:ro \
  --env-file run/deepseek/deepseek.env \
  ghcr.io/mikoskinen/r9700-exact-serving:deepseek-ac0bf129 \
  --model /models/deepseek-v4.1-flash-1017-exl3-2.25bpw.rad --served-model-name deepseek-v4.1-flash \
  --host 127.0.0.1 --port 8090 --tp 4 --tp-wire exact \
  --placement expert_tiered --host-pool-mib 19456 --ngram-placement ram=/ngram \
  --max-num-seqs 32 --max-model-len 524288 --max-num-batched-tokens 4160 \
  --kv-cache-dtype bf16 --checkpoint-interval 2048 --num-speculative-tokens 5 \
  --prefix-cache-mib 86016 --prefix-cache-ttl 28800 --checkpoint-slots 512 --checkpoint-policy keep-all \
  --prefill-span 2 --prefill-interleave 8 --live --draft-window adaptive --draft-coupling on \
  --gpu-headroom-mib 256
run/switches.sh run/deepseek/switches.json
```

The server is ready in about five minutes, most of which goes to writing the 103 GiB n-gram table to
`/dev/shm/ds41ng`.

### Check the answer

Both servers sample by default, so send `"temperature": 0` for exact output. DeepSeek answers this
request with "The capital of Australia is Canberra, and it officially became the capital in 1913 when
the city was founded and named." in every result run:

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4.1-flash", "temperature": 0, "max_tokens": 600, "reasoning_effort": "none",
  "messages": [{"role": "user", "content": "What is the capital of Australia, and in which year did it officially become the capital? Answer in two sentences."}]}'
```

For Qwen, use `"model": "qwen3.8-flash-next"`.

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
`deepseek/current` with `exl3/current`, and the Qwen image from `qwen/current` with `exl3/fn4-dual`
compiled with `-DEXL3R4D_DZ=2`. Every other branch is a single experiment or an earlier build, kept
for the record.

## License

- **CC BY 4.0** ([LICENSE-docs.md](LICENSE-docs.md)) covers the documents and data: `README.md`,
  `docs/`, `registry/` (except `registry/build.py`), and the run summaries and logs under
  `gpu02/exl3-test/runs/` and `gpu02/bench/runs/`.
- **Apache 2.0** ([LICENSE](LICENSE)) covers the scripts and tools: `registry/build.py`, everything under
  `gpu02/exl3-test/` outside `runs/`, `gpu02/host/` and `gpu02/bench/` outside `runs/`, and `run/`. The engine
  branches are forks of Apache 2.0 projects and keep their own LICENSE and NOTICE files.
