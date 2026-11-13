# DeepSeek V4.1 Flash and Qwen Flash-Next (EXL3) on 4x AMD Radeon AI PRO R9700

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

All values are tok/s, summed over all concurrent users. Stock is the card default of 300 W, and
the everyday setting caps each card at 210 W with a -50 mV voltage offset.

| Measure | DeepSeek, stock 300 W | DeepSeek, 210 W | Qwen, stock 300 W | Qwen, 210 W |
|---|---|---|---|---|
| 1 user | 96 prose / 165 code | 91 prose / 152 code | 269 | – |
| 4 users | 176 | 172 | 619 | – |
| 8 users | 208 | 192 | 831 | ~790 |
| 16 users | 259 | 237 | 1,094 | ~1,030 |
| Prefill 16k | 3,740 | 3,700 | 9,540 | – |
| Prefill 64k | 6,020 | 5,530 | 9,810 | ~9,330 |
| Prefill 128k | 6,140 | 5,610 | 9,710 | ~9,200 |

A dash means the value wasn't measured at that setting.

## What exact means

Exact means the output is byte-identical to each model's reference engine, and a prompt gets the same
answer alone or next to other requests. This matters for a coding agent, whose answer to the same
context shouldn't change because other sessions are running on the same cards. In a 36-minute run
under load, all 160 of DeepSeek's answers stayed identical.

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

## Where things are

The `main` branch has:

- [docs/DETAILS.md](docs/DETAILS.md): the long version of this page, with a source for every number and
  the LiteLLM setup for Claude Code.
- [docs/hardware-tuning-gpu02.md](docs/hardware-tuning-gpu02.md): the host, memory and PCIe tuning.
- [docs/LEVERS.md](docs/LEVERS.md): the working log of every change and its numbers.
- [docs/TESTING.md](docs/TESTING.md): the test method rules and the reasons for them.
- [registry/](registry/): one row per tested change, rejected ones included.
- [gpu02/](gpu02/): the test scripts, the host tuning scripts and the run summaries.

The engine code is on the `deepseek/`, `qwen/` and `exl3/` branches, and the current builds are
`deepseek/current` (with `exl3/current`) and `qwen/current`. Every other branch is a single experiment
or an earlier build, kept for the record.

## License

- **CC BY 4.0** ([LICENSE-docs.md](LICENSE-docs.md)) covers the documents and data: `README.md`,
  `docs/`, `registry/` (except `registry/build.py`), and the run summaries and logs under
  `gpu02/exl3-test/runs/` and `gpu02/bench/runs/`.
- **Apache 2.0** ([LICENSE](LICENSE)) covers the scripts and tools: `registry/build.py`, everything under
  `gpu02/exl3-test/` outside `runs/`, and `gpu02/host/` and `gpu02/bench/` outside `runs/`. The engine
  branches are forks of Apache 2.0 projects and keep their own LICENSE and NOTICE files.
