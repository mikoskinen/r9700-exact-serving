# Exact local LLM serving on 4x AMD Radeon AI PRO R9700

This repository documents how two large mixture-of-experts models run locally with exact output on four
AMD Radeon AI PRO R9700 cards (RDNA4, 32 GB each). The models are DeepSeek V4.1 Flash (EXL3, 2.25 bits
per weight) and Qwen Flash-Next (EXL3, 4.05 bits per weight), and only these two were optimised. The
host has an AMD EPYC 7302P (Rome) on an ASRock Rack ROMED8-2T board with 512 GB of RAM. The engine is a
C++/HIP fork of [radiance](https://codeberg.org/StillDeadcode/radiance) with the
[radiance-exl3](https://github.com/Dyluhn/radiance-exl3) plugin, and it runs tensor parallel over the
four cards. Claude Code uses it as its local backend through a LiteLLM proxy.

The notes are written for people who run large MoE models on AMD consumer or workstation GPUs or on EPYC
hosts, and for anyone tuning a multi-GPU PCIe box without NVLink. The exactness method matters if you
need a coding agent to give the same answer to the same context every time.

## Results

Decode rows give the aggregate tok/s of all concurrent users. Stock is the card default of 300 W, and
the everyday setting caps each card at 210 W with a -50 mV voltage offset.

| Measure | DeepSeek, stock 300 W | DeepSeek, 210 W | Qwen, stock 300 W | Qwen, 210 W |
|---|---|---|---|---|
| 1 user, essay prompt | 96.5 | 90.8 | _pending_ | ~9,200 tok/s |
| 1 user, code prompt | _pending_ | _pending_ | _pending_ | ~9,200 tok/s |
| 4 users | 176.1 | 171.6 | _pending_ | ~9,200 tok/s |
| 8 users | 208.1 | 192.0 | _pending_ | ~790 |
| 16 users | 259.1 | 237.0 | _pending_ | ~1,030 |
| 16k cold prefill | _pending_ | _pending_ | _pending_ | ~9,200 tok/s |
| 64k cold prefill | _pending_ | _pending_ | _pending_ | ~9,330 tok/s |
| 128k (Qwen) or 136k (DeepSeek) cold prefill | 6,130 tok/s (TTFT 22.4 s) | 5,570 tok/s (TTFT 24.6 s) | _pending_ | ~9,200 tok/s |

Decoding is greedy at temperature 0. Each prefill is a single cold prompt after a prefix-cache reset,
and the cards were cooled before each 300 W run. Qwen's output is exact up to 16 concurrent sequences,
which is where the table stops. The sources and the measurement details are in
[docs/DETAILS.md](docs/DETAILS.md).

## What exact means

Exact means byte-identical greedy output compared with each model's reference engine. A prompt has to
get the same answer after a speed change and when other requests share the cards, which is why the
kernels are batch-invariant and an answer that changes under load counts as a bug. Before a change is
adopted, it passes identity sets of 20 short and 10 long prompts (plus a cold 136k-token answer for
DeepSeek) and a run beside 16 concurrent sequences, and the test confirms from the switch values and
kernel call counters that the change ran. Soak tests on 13.11.2026 ran 8 cycles of 20 probes under 12
concurrent streams with long prompts arriving (36 minutes for DeepSeek), and both models gave 160 of
160 identical answers.

## Hardware changes

These host changes are measured with DMA tests and short serving runs, and
[docs/hardware-tuning-gpu02.md](docs/hardware-tuning-gpu02.md) has every raw value:

- **Same-size DIMMs:** replacing mixed 4 x 32 GB and 4 x 64 GB DIMMs with 8 x 64 GB put all eight
  memory channels in one interleave set, which raised CPU read bandwidth from 35.9 to 104 GB/s and
  two-card host-to-GPU copies from 45.1 to 55.4 GB/s.
- **One card per IO-die quadrant:** moving three cards onto risers so that every quadrant has its own
  card raised four-card host-to-GPU copies from 61.3 to 105.3 GB/s, because two cards in one quadrant
  share a root port of about 36 GB/s.
- **CPU idle states and governor:** disabling every idle state with an exit latency of 100 µs or more
  and setting the `performance` governor at boot ended the random slow engine starts, and Qwen at 8
  users went from ~630 to ~750 tok/s.
- **The CPU swap that didn't help:** an EPYC 7302P in place of the 7282 on the mixed DIMMs moved
  two-card copies from 44.2 to 45.1 GB/s and made a two-card Qwen server 5.7 % slower at 16 users,
  because the 2-channel memory regions were the limit.

## Software changes

Each of these changes passed the exactness checks above, and [registry/](registry/) lists every tested
change, the rejected ones included:

- **Batch-invariant kernels:** Qwen's batch-invariant kernel set makes each answer independent of the
  other requests in a step and also cuts the 16-user step time from 37.95 to 35.70 ms (-5.9 %) against
  the kernels it replaced.
- **Interleaved all-reduce push:** the exact four-card all-reduce now pushes each card's data to every
  peer in one interleaved pass instead of one link at a time, which cut a 42 MB all-reduce from 3,436
  to 2,508 µs (-27 %).
- **Prefill skip above the last cache source:** DeepSeek's layers 21 to 39 write almost no cache that
  later tokens read, so a cold prefill runs them only for the end of the prompt and at an anchor every
  fourth chunk. The time to first token for a 136k-token prompt fell from 50.5 to 33.3 s (-34 %).
- **Expert residency:** giving DeepSeek's spare VRAM to resident expert weights instead of an expert
  turnover reserve raised 8-user throughput from 143.9 to 169.5 tok/s (+17.8 %).
- **Sequence-parallel prefill stages:** in Qwen prefill each card runs the hyper-connection stages on a
  quarter of the rows between a reduce-scatter and an all-gather, which raised 64k prefill from 8,293
  to 8,754 tok/s (+5.7 %).

## Where things are

The `main` branch has:

- [docs/DETAILS.md](docs/DETAILS.md): the long version of this page, with a source for every number and
  the LiteLLM setup for Claude Code.
- [docs/hardware-tuning-gpu02.md](docs/hardware-tuning-gpu02.md): the host, memory and PCIe tuning.
- [docs/LEVERS.md](docs/LEVERS.md): the working log of every change and its numbers.
- [docs/TESTING.md](docs/TESTING.md): the test method rules and the reasons for them.
- [registry/](registry/): one row per tested change, rejected ones included.
- [gpu02/](gpu02/): the test scripts, the host tuning scripts and the run summaries.

The engine code is on the `deepseek/*`, `qwen/*` and `exl3/*` branches. The current DeepSeek build is
`deepseek/ds41-fuse` with `exl3/ds41-k25u`, and the current Qwen build is `qwen/fn4-x3-stack`.

## License

- **CC BY 4.0** ([LICENSE-docs.md](LICENSE-docs.md)) covers the documents and data: `README.md`,
  `docs/`, `registry/` (except `registry/build.py`), and the run summaries and logs under
  `gpu02/exl3-test/runs/` and `gpu02/bench/runs/`.
- **Apache 2.0** ([LICENSE](LICENSE)) covers the scripts and tools: `registry/build.py`, everything under
  `gpu02/exl3-test/` outside `runs/`, and `gpu02/host/` and `gpu02/bench/` outside `runs/`. The engine
  branches are forks of Apache 2.0 projects and keep their own LICENSE and NOTICE files.
