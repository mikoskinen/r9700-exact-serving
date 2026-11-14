# Details: results, methods and sources

This repository is the investigation history of serving two large mixture-of-experts models locally
with **exact** output: DeepSeek V4.1 Flash (EXL3, 2.25 bits per weight) and Qwen Flash-Next (EXL3,
4.05 bits per weight). The models run on four AMD Radeon AI PRO R9700 cards (RDNA4, gfx1201, 32 GB
each). The host, `gpu02`, has an AMD EPYC 7302P (Rome) on an ASRock Rack ROMED8-2T board with 512 GB
of RAM in 8 x 64 GB DIMMs [H0]. The engine is a fork of [radiance](https://codeberg.org/StillDeadcode/radiance)
with the [radiance-exl3](https://github.com/Dyluhn/radiance-exl3) extension. The setup is the local
backend for Claude Code through a LiteLLM proxy.

This repository has the documents, the test-queue scripts, the per-run summaries, and a registry of
every change that was tried, whether it was kept or rejected. The engine code is on this repository's
`deepseek/`, `qwen/` and `exl3/` branches (see [Repository map](#repository-map)). This page is the
long version of the top-level README, with a source for every number.

## Scope: two models only

Only two model lines were optimised:

- **DeepSeek V4.1 Flash**, EXL3 2.25 bpw, tensor parallel over 4 cards (TP4).
- **Qwen Flash-Next**, EXL3 4.05 bpw, TP4. Earlier rows also cover Radiance's own iq4r quantisation
  of the same model.

Other models and quantisations weren't tuned. The numbers, levers and kernel forms here apply to
these two lines on this hardware.

## Results

Every card runs at a 210 W cap with a -50 mV voltage offset unless a column says stock 300 W, which is
the card default of 300 W with no voltage offset. The four cards run tensor parallel, and
decoding is greedy (temperature 0). In "mb cN", a microbenchmark runs N concurrent streams and the
aggregate tok/s counts their generated tokens. Per-user tok/s is the aggregate divided by N, computed
here rather than measured.

### DeepSeek V4.1 Flash (EXL3 2.25 bpw), 14.11.2026

| load point | stock 300 W [S1] | 210 W [S2] | target | how it was measured |
|---|---|---|---|---|
| 1 user, prose | 104.91 tok/s | 101.30 tok/s | >= 90 | 512-token greedy answer to an essay prompt, mean of 6 readings in two blocks of 3, each block after one discarded reading |
| 1 user, code | 181.98 tok/s | 173.29 tok/s | >= 90 | the same with a code prompt, mean of 3 readings after one discarded reading |
| mb c4 | 201.15 tok/s | 191.60 tok/s | >= 160 | fixed prompts, gens window of 3, mean of 2 blocks after one discarded block |
| mb c8 | 246.35 tok/s | 234.15 tok/s | >= 200 | fixed prompts, gens window of 2, mean of 2 blocks |
| mb c16 | 293.85 tok/s | 276.90 tok/s | (none) | fixed prompts, gens window of 1, mean of 2 blocks |
| prefill 16k (15,777 tokens) | 4,035 tok/s, TTFT 3.91 s | 3,915 tok/s, TTFT 4.03 s | (none) | one request after a prefix-cache reset, prompt tokens divided by the TTFT |
| prefill 64k (65,650 tokens) | 6,124 tok/s, TTFT 10.72 s | 5,597 tok/s, TTFT 11.73 s | (none) | the same |
| prefill 128k (131,137 tokens) | 6,388 tok/s, TTFT 20.53 s | 5,797 tok/s, TTFT 22.62 s | (none) | the same |
| 136k-token cold prefill (137,135 tokens) | 6,291 tok/s, TTFT 21.80 s | 5,714 tok/s, TTFT 24.00 s | >= 4000 tok/s, TTFT <= 35 s | the same |

Every target in the table is met at both settings [S16]. The README rounds decode to whole tok/s and
prefill to the nearest 10 tok/s.

Each column comes from one server start of the image 4c779f21d4ea
(`ghcr.io/mikoskinen/r9700-exact-serving:deepseek-ac0bf129`) with the README's `docker run` command
and `run/switches.sh`. The cards cool to 45 °C or less before the start and before each speed
section. The stock start sets every card to the values of `run/power.sh 300` and checks that every
card reports them. Three seconds later the fan guard in `gpu02/host/fanguard.sh` lowers the flat fan
value from 90 to 88 %, where it stays until the end of the run [S1][S17]. In both starts the 73
runtime switches read back as expected before each section and at the end. The 20 identity probes
give the same answers alone, all at once, one at a time beside 16 decode streams with a fresh
7,772-token prefill 0.5 s into each probe, and alone again at the end. At both settings these answers
equal the stored alone answers of this build (`ref-alone.jsonl`) [S1][S2].

### Qwen Flash-Next (EXL3 4.05 bpw), batch-invariant set "S9"

| load point | result | per user | target | vs target | how it was measured |
|---|---|---|---|---|---|
| mb c16 decode | 1030.5 tok/s (blocks 1039.2 / 1035.3 / 1032.6 / 1014.7) | ~64.4 | >= 1000 | +3.1 % | S9 vs S8, in-process blocks, one start (item 223) [S3] |
| mb c8 / c4 decode | S9 / S8 = 1.0322 / 1.0163 | | | | same item [S3] |
| prefill, fresh 64k / 128k prompts | 9164.9 tok/s (mean of 8 readings, range 9093-9289) | | >= 8500 | +7.8 % | starts A B A B (item 230) [S4] |
| prefill 64k with the 233 prefill stack | ~9330 tok/s (base ~9230) | | | ~+1.0 % | in-process A/B (item 233), not yet adopted [S5] |

10,000 tok/s prefill is the next goal and is still open. Collectives take about a third of the
prefill device time on rank 0 [S6].

S9 is exact only up to 16 sequences (64 decode rows a step). At 24 and 32 sequences, 17 of 20 answers
differ from the answers given alone. The higher speeds there (~1080 and ~1185 tok/s) aren't exact and
aren't counted as results [S5].

## Quality doesn't degrade

"Exact" here means **byte-identical greedy output** compared with the line's reference engine. The
answer to a prompt must not change because of a speed change, and it must not change because other
requests are running at the same time. An answer that changes under load counts as a bug, not noise
[S7].

The DeepSeek matrix kernel of 14.11.2026 is the one change that moves the reference. It sums the dense
layers in a different order than the earlier builds, and 13 of the 20 probe answers differ from theirs
(gpu02/exl3-test/runs/xdsm/soak-ds-1114/identity.txt:4). This build is DeepSeek's reference from here
on, and the checks below compare against its own answers.

Each candidate change goes through the following checks:

1. **Identity sets:** a 20-prompt short greedy set and a 10-prompt long set, plus a cold 136k-token
   deep answer for DeepSeek, each compared byte for byte with the reference [S3][S8].
2. **Under load (batch invariance):** the same prompts are run beside other decoders, with staggered
   prefill arrivals and 16 concurrent sequences. The answers must still match the answers given
   alone [S7].
3. **Soaks:** long windows of loaded cycles, compared strictly each cycle:
   - DeepSeek, 14.11.2026: 8 cycles x 20 probes beside 8 to 32 decode streams, with a fresh
     7,772-token prefill 0.5 s into every probe. 160 of 160 identical, 0 errors, over 37.9 minutes
     of load windows. The soak start runs the same image, and its switches read back the same 73
     values, at `--gpu-headroom-mib 245` [S18].
   - Qwen S9: 8 cycles x 20 probes under c12 load plus a staggered ~16k-token prefill, with about 62 %
     of probes in flight. 160 of 160 identical, and the control run afterwards was also identical.
     Every card stayed at its 210 W cap [S5][S9].
4. **Engagement:** before an A/B result is read, the run must prove that the lever actually ran: the
   switch values are read back and the per-form call counters are checked [S10].

A speed lever is adopted only after these checks pass. Batch invariance has a cost of its own. For
Qwen, the full BI set costs +2.0 % step time at c4. It is 0.5 % faster than BI off at c8 and 5.9 %
faster at c16, because the BI kernel forms are also faster forms [S3].

Lossy options are kept off the exact lines. One example is the `wht6` low-precision tensor-parallel
wire, which the production servers use [S11].

## Using it from Claude Code

Claude Code speaks the Anthropic Messages API. To run it on local models, point it at a LiteLLM proxy
that offers an Anthropic-compatible endpoint: set `ANTHROPIC_BASE_URL` to the proxy and
`ANTHROPIC_API_KEY` to a proxy key, and Claude Code sends its requests there [S12]. The proxy maps each
model alias Claude Code sends, for the opus, sonnet and haiku tiers, to a local Radiance server through
its OpenAI-compatible API. Each alias can carry its own reasoning effort [S12] and its own input
limit (`max_input_tokens` in the example below), which lets a long-context tier and a fast tier share
the same hardware. Radiance's prefix cache reuses the repeated part of a session's prompt, and its
host-memory tier sets how many sessions stay cached: replicating DeepSeek's KV groups raised that
tier's capacity from 8.2M to 32.7M tokens (DS-L172) [S15].

Exact serving matters for this use: a coding agent's answer to the same context shouldn't change
because other sessions are running on the same cards. This repository treats such a change as a bug
[S7].

A minimal example of the proxy's model list, written for this README with placeholder hosts and
example limits:

```yaml
model_list:
  - model_name: claude-opus-4-5            # the alias Claude Code sends for the opus tier
    litellm_params:
      model: hosted_vllm/deepseek-v4.1-flash
      api_base: http://engine-a.example:8090/v1
      api_key: os.environ/RADIANCE_API_KEY
      reasoning_effort: high
    model_info:
      max_input_tokens: 500000             # long-context tier
  - model_name: claude-sonnet-4-5
    litellm_params:
      model: hosted_vllm/qwen3.8-flash-next
      api_base: http://engine-b.example:8090/v1
      api_key: os.environ/RADIANCE_API_KEY
      reasoning_effort: medium
    model_info:
      max_input_tokens: 262144
  - model_name: claude-haiku-4-5
    litellm_params:
      model: hosted_vllm/qwen3.8-flash-next
      api_base: http://engine-b.example:8090/v1
      api_key: os.environ/RADIANCE_API_KEY
      reasoning_effort: low
    model_info:
      max_input_tokens: 200000

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

Claude Code's own setting for which alias each tier uses (the opus, sonnet and haiku defaults) can pin
these names [S12].

Measured through this chain (Claude Code -> LiteLLM -> local engine), the TBLite agentic test set (60
tasks, the sonnet alias at medium effort) scored 65.4 % mean reward for Qwen Flash-Next on the earlier
2-card xeon server, against 52.0 %, 43.8 % and 42.2 % for three other local models (Ornith-1.5-35B,
Sage-A1.5 and Sabi-Noema) [S13]. The
production servers are two 2-card tensor-parallel servers (DP2) serving Qwen Flash-Next with the lossy
`wht6` wire [S11][S14]. The exact 4-card lines in this repository are the optimisation work, and the
DeepSeek line targets a 1M-token context [S16].

## Hardware tips (EPYC Rome + 4 GPUs)

The largest gains this month came from the host. The full write-up, with the raw measurement files, is
[hardware-tuning-gpu02.md](hardware-tuning-gpu02.md) [H1]. The main points, with before and
after values:

| change | before | after |
|---|---|---|
| **Same-size DIMMs in one interleave set:** mixed 4 x 32 GB + 4 x 64 GB DIMMs made Rome split memory into four 2-channel regions, and every buffer used one channel pair. 8 x 64 GB put all eight channels in one set. | 2-card concurrent H2D / D2H 45.1 / 42.9 GB/s, STREAM triad 27.8 GB/s, CPU read 35.9 GB/s, Radiance TP2 c16 402.65 / 411.35 tok/s | 55.4 / 55.3 GB/s (the PCIe limit), 86.1 GB/s, 103.2-104.0 GB/s, 438.2 (+8.8 %) / 444.25 (+8.0 %) tok/s |
| **Check the address map** after any DIMM change with `df-map.sh` (DF / UMC registers). Count how many UMCs share an address range, because the printed interleave code isn't a channel count. | mixed set: four ranges, each on one channel pair | 8 x 64 GB: UMC 0-7 all share 0-513.25 GiB |
| **CPU swap 7282 -> 7302P** with the same mixed DIMMs. It didn't help, because the channel pairs were the limit. | 2-card H2D 44.2 GB/s, STREAM triad 27.6 GB/s, TP2 1.15.1 c16 429.2 / 443.4 tok/s | 45.1 GB/s, 27.8 GB/s, 403.2 / 419.5 tok/s (-5.7 %) |
| **One card per IO-die quadrant:** two cards in one quadrant share a root port of about 36 GB/s. Risers moved three cards so that each quadrant has one. | 4 cards at once: H2D 61.3 GB/s, D2H 75.6 GB/s. Same-quadrant pair: 35.3 / 37.9 GB/s against 55.8 / 55.3 for other pairs. | H2D 105.3 GB/s, D2H 110.6 GB/s |
| **The shared-quadrant penalty** showed in serving before the move: a TP2 server on the shared pair against one on separate quadrants, in an 8-agent replay. | shared pair 193 tok/s, decode -21 %, prefill -42 % | separate pair 253 tok/s |
| **Host C-states and governor:** cpuidle C2 (400 us exit latency) with schedutil made fresh starts slow at random. Disable every C-state with an exit latency of 100 us or more and set `performance` at boot (systemd unit), with `kernel.numa_balancing=0`. The BIOS C-state switch alone didn't remove C2. | Flash-Next TP4 mb c8 ~630 tok/s in a slow start | ~750 tok/s, TP2 1-user step 17.7 -> 17.1 ms |
| **Power cap and undervolt:** 210 W per card (default 300, max 330) with -50 mV VDDGFX. Under a cap an undervolt raises clocks: -90 mV gave ~+200 MHz on the xeon. 330 W gave only +5-7 % at 98 C junction and wasn't pursued. | | |
| **Fans:** GDDR6 memory, not the junction, is the hot part on an open bench. With an 1800 rpm limit one card's memory reached 96 C. The acoustic control jumped by more than 150 rpm 83 times an hour on one card, and a flat curve (fixed duty to 85 C hotspot, linear to 50 % at 100 C) removed that. A guard script adds +2 % at memory >= 92 C or hotspot >= 95 C. | | |
| **PCIe settings:** 8-bit extended tags, relaxed ordering, MPS 256 / MRRS 512. 10-bit tags are a dead end, because the Rome root ports support neither completer nor requester. PCIe DPM pinning was never isolated. SDMA vs blit copies don't change the H2D limit. | | |
| **Seat risers firmly** (a loose riser made a card fall off the bus on wake from idle) and **check link widths after a BIOS reset** (a cleared CMOS left a slot at x4 on the Threadripper machine, and an x8 pair gave 54-57 % of an x16 pair's prefill). | | |

## Software tips

### Exact levers that paid off

The values come from [registry/](../registry/), where "exact" means the identity checks passed and the
batch-invariant (BI) rows also passed the under-load checks.

| line | change | measured | registry |
|---|---|---|---|
| DeepSeek | **Prefill skip above the last cache source with anchors** (PSK design 2, K4) | 136k cold TTFT 50.51 -> 33.25 s (-34.2 %) | DS-L234-b |
| DeepSeek | **Exact cone replay on a prefix hit + tail-only anchors** | 150k cold TTFT 36.85 -> 32.14 s (-12.8 %) | DS-L287-a |
| DeepSeek | **Expert residency over speculative turnover** (VRAM sweep: resident expert units instead of turnover 150) | mb c8 143.9 -> 169.5 tok/s (+17.8 %) | DS-L320 |
| DeepSeek | **Exact full-vocabulary DSpark Markov walk** for drafting (W1) | 1-user essay 57.67 -> 64.54 tok/s (+11.9 %) | DS-L314-a |
| DeepSeek | **Lend the drafter's slab to experts while drafting is off** (DL, part of load-aware drafting, which drafts mainly at low load) | mb c8 150.9 -> 166.1 tok/s (+10.1 %) | DS-L229-a |
| DeepSeek | **Multi-row decode-band GEMV kernel** (D2) | mb c8 154.83 -> 170.03 tok/s (+9.8 %) | DS-L196 |
| DeepSeek | **Decode-row cap of 48** (makes 16 concurrent sequences batch-invariant and faster) | mb c16 203.9 -> 223.2 tok/s (+9.5 %), BI | DS-L238 |
| DeepSeek | **XPUSH**: interleaved phase-1 push in the exact four-rank two-shot all-reduce | 42 MB all-reduce 3436 -> 2508 us (-27 %), 136k cold TTFT 24.57 -> 24.15 s (-1.7 %) | DS-L219, DS-L261 |
| DeepSeek | **Fused router** GEMM + top-k + scatter (`ds41_rfuse 2`, part of int16) | c8 step p50 33.67 -> 33.62 ms (-0.15 %), int16 1-user 90.24 -> 90.77 tok/s | DS-L252-a |
| Qwen | **Batch-invariant kernel set S9** | c16 full-row step p50 37.952 -> 35.704 ms vs BI off (-5.9 %) | QW-L374-g |
| Qwen | **BI hc_read in one launch** (hc_bi 5) | c16 ms/step x0.8751 (-12.5 %) | QW-L373-e |
| Qwen | **Sequence-parallel hc with a pipelined issue order** (hc_sp 2) | prefill 64k 8293 / 8257 -> 8754 / 8744 tok/s (+5.7 %), 128k +5.8 % | DS-L208-b (Qwen row) |
| Qwen | **Dual-band EXL3 MoE pass** for mixed steps | one card, a layer: 1699.3 -> 1555.9 us (-8.4 %) | QW-L372-a |
| Qwen | **Pipelined prefill** (MNBT 4096 + `--prefill-span 2`) | prefill 16k 7533.3 -> 8154.2 tok/s (+8.2 %) | QW-L25 |
| Qwen | **EXL3 line moved to the main-line base** | mb c16 913.9 -> 975.5 tok/s (+6.7 %) | QW-L128 |
| Qwen | **Spare workgroups zero dead rows directly** (DZ) | one-card down at M16 57.84 -> 49.69 us (-14.1 %) | QW-L138 |

### What didn't help (selected)

| change | result | registry |
|---|---|---|
| DSpark block drafting at 5 tokens on the first DeepSeek main line | mb c4 99.0 -> 67.9-75.7 tok/s (rounds cost 2.1x a step) | DS-L338, DS-L187 |
| fp8 block dot in the DeepSeek indexer | 56 % faster kernel but **not identical** | DS-L291-c |
| Plain acquire rule (skipping an acquire between write-after-write launches) | stale elements in 300 trials: **not identical** | DS-L336 |
| Static expert pin (`--expert-pin-top 0.3`) | mb c8 123.4 -> 119.8 tok/s (-2.9 %) | DS-L335 |
| Shared expert on a second lane at decode (F4) | 1-user -2.4 % | DS-L275 |
| Router + mhc_mix kernel fusions (first try) | mb c4 -1.0..-1.4 % | DS-L168 |
| More CUs for the prefill communication lane (8 / 16 / 24 blocks) | prefill 64k x0.986 / x0.965 / x0.947 | QW-L382-a |
| Coded (compressed) reduce-scatter / all-gather in prefill | 2048 x 2560 all-reduce 670 -> 1280 us | QW-L382-b |
| S9 above 16 sequences | 17 of 20 answers differ: not batch-invariant | QW-L387-a |
| CPU 7282 -> 7302P, 330 W power cap, 10-bit PCIe tags | see the hardware table | HW rows |

Every row with its reason is in [registry/](../registry/). All rejected rows are kept.

### Measurement lessons

These rules came out of measurement mistakes ([TESTING.md](TESTING.md)).

- **Judge small decode levers on the step-log full-row p50, not on tok/s.** Microbenchmark blocks
  spread 2-6 %, while the step log spreads 0.1-0.3 %. One lever measured 0.974 on mb tok/s and +2.0 % on
  the step log [M1].
- **Choose memory-bound kernel forms on a cold microbenchmark** (`--cold 256` MiB). A warm cache said
  -10..-24 %, while the engine gave +2.0 % [M2].
- **Prove a lever engaged before reading its A/B**: read back the switch and check the per-form call
  counters [M3].
- **Run arms in A B B A order** so linear drift cancels. Run decode phases before prefill: prefill at a
  210 W cap drifts -2..-3 % while the cards warm up [M4].
- **AQL timestamps overstate launch gaps about 5x here.** A gap stamped at 307 us was about 60 us in
  reality [M5].
- **Follow docker logs from container start.** The json-file driver rotates at 5 x 100 MB, and one item
  lost 15 of its 17 measurement windows to rotation [M6].
- **Mock sandboxes can exhaust /tmp inodes.** About 51k files x 21 directories filled all 1,048,576
  inodes, and 4 items failed [M7].
- **`pgrep -f` wait loops match their own command line.** Five stale loops ran for ~7.5 h [M8].
- **Forced-generation mode in the Claude Code replay breaks draft acceptance** (0.02-0.10 vs
  0.4-0.75). Speculative levers need natural mode at temperature 0 [M9].

## Tested changes and results

The [change registry](../registry/README.md) has one row per tested change: what changed, why, whether it
is exact, how it was measured, the before / after numbers and the status. Each row links back to its
line in [LEVERS.md](LEVERS.md) and, where one exists, to its run summary. 392 rows were
converted from LEVERS.md and the host / method notes. 116 rows are marked `needs-check` because a value
was missing or ambiguous in the source.

| line | ADOPTED | OPEN | NEEDS-USER | PARKED | CLOSED-small | REJECTED | total |
|---|---|---|---|---|---|---|---|
| DeepSeek | 60 | 34 | 1 | 31 | 12 | 44 | 182 |
| Qwen | 35 | 26 | 5 | 8 | 9 | 35 | 118 |
| host-hardware | 8 | 9 | 0 | 1 | 0 | 14 | 32 |
| method-tooling | 51 | 0 | 0 | 6 | 0 | 3 | 60 |

Per-line tables, sorted by status and then by gain:

- [registry/deepseek.md](../registry/deepseek.md)
- [registry/qwen.md](../registry/qwen.md)
- [registry/host-hardware.md](../registry/host-hardware.md)
- [registry/method.md](../registry/method.md)

To add rows, edit `registry/registry.csv` and run `python3 registry/build.py`.

## Repository map

### Engine branches (full history including rejected branches)

The engine code lives in this repository as branches next to `main`. Both model lines are forks of the same
upstream radiance history, so they share their base commits. The EXL3 plugin is a separate upstream project with
its own history.

| branch prefix | upstream | contents |
|---|---|---|
| `deepseek/` | radiance (Codeberg StillDeadcode/radiance, Apache 2.0) | 120 branches, where `ds41-*` are DeepSeek V4.1 Flash levers (one branch per lever) and `ds41-int*` are integration stacks. The README's DeepSeek image is built from engine commit ac0bf129, which `deepseek/current` contains. |
| `qwen/` | radiance main (451b136) | 17 branches, where `fn4-*` are Qwen Flash-Next TP4 levers. S9 with the prefill stack is `qwen/fn4-x3-stack`. |
| `exl3/` | radiance-exl3 (GitHub Dyluhn/radiance-exl3, Apache 2.0) | 24 branches of EXL3 kernel work for both lines. The README's DeepSeek image uses EXL3 commit df1105a6, which `exl3/current` contains. |

Upstream release tags carry the same prefixes, for example `radiance/v1.15.1` and `exl3/v1.3.0`. The registry
names branches without the prefix. A branch name alone doesn't tell whether its lever was adopted.
The registry row (column `branch_commit_image`) and its status do.

### This repository

| path | what |
|---|---|
| [LEVERS.md](LEVERS.md) | The raw lever log. Every lever's numbers, reference and status, as written during the work. |
| [TESTING.md](TESTING.md) | Test method rules and the reasons for them. |
| [R9700-CAPABILITIES-20261112.md](R9700-CAPABILITIES-20261112.md) | What the R9700 / RDNA4 and this host can and can't do. |
| []() (other) | Design notes and audits for both lines, xeon benchmark notes, and the Claude Code quality-test setup (`tbench/`). |
| [gpu02/exl3-test/queue*/](../gpu02/exl3-test/) | Test queue items. Each `NN-*.sh` is one measured item, `DRYRUN=1` prints its commands. `queue/done/` has the items that ran. |
| [gpu02/exl3-test/tools/](../gpu02/exl3-test/tools/), top-level `*.sh` / `*.py` | Harness scripts: identity compare, microbenchmarks, profile summaries, watchdog. |
| [gpu02/exl3-test/runs/](../gpu02/exl3-test/runs/) | Per-run summaries such as `*sum*.txt`, `identity.txt` and `results.json`, without the raw logs and dumps. `runs/queue-changes.log` records every queue change. |
| [gpu02/host/](../gpu02/host/) | The host tuning service (C-states, governor), the NUMA sysctl, the fan and overdrive scripts, and the host facts read on 13.11.2026. |
| [gpu02/bench/](../gpu02/bench/) | Host and PCIe bandwidth tools and their raw results (`runs/rocm-bw/`, `runs/logs/`). |
| `gpu02/exl3-test/runs/ds41-exl3.log.gz` | The full RESULT log of the gpu02 test queue (5.11.-13.11.2026), compressed. |
| [registry/](../registry/) | The change registry. |

### Reading a run summary

A summary line looks like this (from `runs/xdsm/zzzfuse/fusesum.txt`):

```
ZFUSE BLOCK c8-1-G-B (B, c8 gens): agg 191.7 tok/s, window 42.6 s, 16 requests, 0 errors, ...
  decode steps 1010: p50 33.67 / mean 34.17 ms | full-row bucket (n_seq 8, n_tok 8): 1010 steps p50 33.67 ms | ...
```

- `B` / `F` (or `A` / `B`) are the arms.
- `c8` is the concurrency, `agg` the aggregate tok/s, and `full-row bucket p50` the step time used
  for decisions.
- `identity.txt` next to it records the byte-identity compares (IDENTICAL / DIFFER).
- The verdict line at the end of each summary states the decision and the reference it was measured
  against.

## Sources

Line numbers refer to the copies in this repository (`docs/`), as of 13.11.2026.

- [S1] gpu02/exl3-test/runs/xdsm/hero-ds-300w-stock/ (the stock 300 W start): u1.jsonl:2-4 and :6-8 (1 user, prose), :10-12 (1 user, code), mb.jsonl:3-4 (c4), :6-7 (c8), :8-9 (c16), cold.jsonl:4, :3, :1 and :2 (prefill 16k, 64k, 128k and 136k: prompt_tokens and ttft), run.log:3 (the power setting read back on every card) and its COOL lines, fn/ (the switch read-backs), identity.txt (the six identity compares), docker-inspect.json and ../hero-ds-300w-stock.out (the start command and image), hero-sum.txt (every value with the lines it comes from, written by gpu02/exl3-test/queue-ar/hero/hero-sum.py, except its engine-log line, which comes from the server log that isn't in this repository), targets docs/LEVERS.md:150.
- [S2] gpu02/exl3-test/runs/xdsm/hero-ds-210w/ (the 210 W start), with the same files and line numbers as [S1], except that run.log:1 has the caps and offsets read back before the start.
- [S3] docs/LEVERS.md:374 (item 223 S9 vs S8, BI cost, item 225 identity sets), target docs/LEVERS.md:14.
- [S4] docs/LEVERS.md:384 (item 230), base S9 + hc_sp 2 + xpush per docs/LEVERS.md:382 (item 227).
- [S5] docs/LEVERS.md:387 (S9 scope 232, soak 231, prefill stack 233), docs/TESTING.md:133, :135.
- [S6] docs/LEVERS.md:381 (PF-RANK, item 209).
- [S7] docs/LEVERS.md:9 (exact = byte-identical), :367 (batch invariance is part of exactness).
- [S8] docs/LEVERS.md:250, gpu02/exl3-test/runs/xdsm/zsoak16/soaksum.txt, docs/TESTING.md:131.
- [S9] gpu02/exl3-test/runs/fn4-extract/231-s9soak-result-lines.txt (item 231 RESULT lines, 13.11. 03:06-03:22).
- [S10] docs/TESTING.md:121.
- [S11] docs/LEVERS.md:62 (production's `--tp-wire wht6`), docs/R9700-CAPABILITIES-20261112.md:105 (wht6 is lossy).
- [S12] docs/tbench/README.md:91-95 (`ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, the opus / sonnet / haiku defaults), docs/tbench/TBLITE.md:5, :108-111 (the sonnet alias at medium effort, the haiku alias at low effort).
- [S13] docs/tbench/TBLITE.md:107 (engines on the xeon), docs/tbench/runs/night1021-flash-prod-v60-n16/summary.md (65.4 %), docs/tbench/runs/tblite-ornith-20261024-1921/summary-tbsummary.md (52.0 %), tblite-sage-20261024-1814 (43.8 %), tblite-sabi-20261024-2037 (42.2 %).
- [S14] docs/LEVERS.md:341-349 (production DP2 = 2 x TP2, Flash-Next line).
- [S15] registry DS-L172, docs/LEVERS.md:172.
- [S16] docs/LEVERS.md:150 (DeepSeek targets, context 1M).
- [S17] gpu02/exl3-test/runs/log-extract/od-changes-lines-245-256.txt (lines 245-256 of the gpu02 power and fan change log: the stock setting, the fan guard's step from 90 to 88 % on every card and the restore after the run), gpu02/host/fanguard.sh:5-6 (the 2 % step down).
- [S18] gpu02/exl3-test/runs/xdsm/soak-ds-1114/: identity.txt:7-14 (the eight soak cycles, 20 of 20 each), the eight `*.win` files (start and end of each load window, 37.9 minutes in total), the cycle summaries beside them (streams, prefill offset and tokens, errors), the switch read-back and the start environment. gpu02/exl3-test/runs/log-extract/soak-ds-1114-queue-log-line-21755.txt (the image and its ID) and soak-ds-1114-start-lines.txt (the `--gpu-headroom-mib 245` lines, with their line numbers in the server log).
- [H0] gpu02/host/hostfacts-20261113.txt (CPU, board, RAM, cards), gpu02/bench/runs/rocm-bw/gpu02-8x64-1102-1632.txt (the 8 x 64 GB set).
- [H1] docs/hardware-tuning-gpu02.md, which cites every number in the hardware table to a raw file in gpu02/bench/runs/, a script in gpu02/host/, LEVERS.md or R9700-CAPABILITIES-20261112.md. A few numbers (CPU read, the address map, the TP2 c16 for the CPU swap, the fan tests, the 8-agent pair replay, the Threadripper link-width lesson) come only from the original Finnish results logs, which aren't published, and the doc says so per number.
- [M1] docs/TESTING.md:119. [M2] docs/TESTING.md:120. [M3] docs/TESTING.md:121. [M4] docs/TESTING.md:59, :119.
- [M5] docs/LEVERS.md:248, :225. [M6] docs/TESTING.md:125. [M7] docs/TESTING.md:126. [M8] docs/TESTING.md:127.
- [M9] docs/TESTING.md:43, docs/LEVERS.md:30.

## License and third-party content

This repository has two licenses:

- **CC BY 4.0** ([LICENSE-docs.md](../LICENSE-docs.md)) covers the documents and data: `README.md`,
  `docs/`, `registry/` (except `registry/build.py`), and the run summaries and logs under
  `gpu02/exl3-test/runs/` and `gpu02/bench/runs/`.
- **Apache 2.0** ([LICENSE](../LICENSE)) covers the scripts and tools: `registry/build.py`, everything under
  `gpu02/exl3-test/` outside `runs/`, and `gpu02/host/` and `gpu02/bench/` outside `runs/`.

The engine repositories are forks of Apache 2.0 projects and keep their own LICENSE and NOTICE files.
