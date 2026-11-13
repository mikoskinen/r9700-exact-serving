# Change registry: what was tested and how it worked

This directory is a browsable registry of every tested change. Each row is one tested change: a kernel
form, a scheduler switch, a host setting, or a measurement rule. Rejected, closed and parked rows are
kept on purpose.

| file | what it is |
|---|---|
| [registry.csv](registry.csv) | The file every other file here is generated from. New rows go here. |
| [registry.json](registry.json) | The same rows as JSON, generated. |
| [deepseek.md](deepseek.md) | DeepSeek V4.1 Flash (EXL3 2.25 bpw) rows, generated. |
| [qwen.md](qwen.md) | Qwen Flash-Next (EXL3 4.05 bpw, plus the earlier iq4r rows), generated. |
| [host-hardware.md](host-hardware.md) | EPYC Rome host and R9700 / PCIe rows, generated. |
| [method.md](method.md) | Measurement-method and tooling rules, generated. |
| [counts.md](counts.md) | Status counts per line, generated. |
| [build.py](build.py) | Regenerates every generated file from the CSV. |

To regenerate, run `python3 registry/build.py` from the repository root. The script uses only the
Python standard library.

## Where the rows come from

The raw source is [../docs/LEVERS.md](../docs/LEVERS.md), the working lever log kept during the project.
It is dense: one LEVERS row can run to thousands of characters. Each registry row points back to it
through `source_levers_line`, the line number in that file. Rows that come from other files (host and
hardware notes, TESTING.md rules) give a `file:line` pointer in that column instead.
`source_run_summary` points to a run summary under `../gpu02/exl3-test/runs/` when one was copied into
this repository.

The values were taken from the sources as written, with semicolons changed to commas. Where a value was missing or ambiguous, it was left
blank and the row was marked `needs-check` instead of being guessed. The rows were converted with
agent help and spot-checked, so the LEVERS.md line remains the reference for any number.

## Columns

| column | meaning |
|---|---|
| id | `DS-`, `QW-`, `HW-` or `ME-` followed by `L<LEVERS line>`, or a short slug for rows from other files. |
| line | `DeepSeek`, `Qwen`, `host-hardware` or `method-tooling`. |
| name, what_changed, why | The change in plain English and the profile reading or cause behind it. |
| exact | See [Exactness](#exactness) below. |
| how_measured | The measurement method. See [Methods](#methods) below. |
| metric, base_value, new_value, pct_change | The primary target metric, before, after, and the signed % change of the metric. |
| gain_pct | The same change with the sign set so that positive = better. Lower-is-better metrics (ms, s, TTFT) are flipped. |
| status | See [Status](#status) below. |
| levers_status | The original status words from LEVERS.md. |
| date | The last measurement date (2026). |
| branch_commit_image | Engine branch, commit or container image, where known. Branches are in the engine repositories. |
| run_items | Queue item names or numbers, with the scripts under `../gpu02/exl3-test/queue*/`. |
| needs_check | `needs-check` when a value is missing or ambiguous. |

## Status

| status | meaning |
|---|---|
| ADOPTED | In the line's integration stack, the main line or production. |
| OPEN | A candidate, or still queued, being built or being measured. |
| NEEDS-USER | Waiting for an owner decision, usually about cost, production or a lossy option. |
| PARKED | Not pursued now. It could matter later, for example on other hardware or another load shape. |
| CLOSED-small | The gain was measured and real, but within noise or too small to justify the code. |
| REJECTED | Slower, failed the identity check, or broke something. The reason is in the notes. |

## Exactness

"Exact" means byte-identical greedy (temperature 0) output against the line's reference engine, in two
settings:

- **Alone:** the identity sets of 20 short and 10 long prompts give byte-identical answers.
- **Under load (batch invariance):** the same prompts still give byte-identical answers while other
  requests are decoding and prefilling beside them. This covers staggered-prefill load and the
  multi-cycle soaks.

An answer that changes under load counts as a bug, not noise.

| value | meaning |
|---|---|
| byte-identical | Passed the identity gate against the reference. |
| batch-invariant | Also passed an under-load identity check (staggered prefill / BI soak). |
| lossy | Changes the output by design, for example a lower-precision wire or quantised KV. Kept off the exact lines. |
| not-identical | Failed the identity check. |
| n/a | Host and method rows. |

## Methods

| value | meaning |
|---|---|
| step-log ABBA | Engine step-log full-row p50 step time. Arms run in A B B A order to cancel drift. Used for small decode levers. |
| mb ABBA / mb A/B | Fixed-concurrency microbenchmark (c1 / c4 / c8 / c16 streams), aggregate tok/s. |
| prefill ABBA | Cold or warm long-prompt prefill, tok/s or TTFT. |
| in-process A/B | Both arms switched at run time in the same server process, so start-to-start variance cancels out. |
| microbench | Kernel-level timing, cold-cache for memory-bound kernels. |
| soak | Repeated identity cycles under mixed load (for example 8 cycles x 20 prompts). |
| identity | The identity gate only. |
| profile | Per-op / per-rank / link profile reading. |
| cc_sim | Claude Code transcript replay, kept for history. Later decisions didn't rely on it (see TESTING.md). |

## Coverage and backlog

Every table row of LEVERS.md (lines 12-387 of the 13.11.2026 copy in `../docs/`) has been converted, and
there is no backlog. Line numbers refer to that copy. LEVERS.md was still being edited while the
registry was built, so a newer LEVERS.md can have shifted line numbers.

Lines were skipped only when they aren't tested changes: section headers, target and base
descriptions, profile-only paragraphs, and table headers. The skipped lines are listed below.

- 1-11, 12-21, 41, 52-71, 90-102, 111-119, 139-154, 155-160 (distance table, summarised in DS-L175),
  161, 163-165, 174, 176, 181-186, 189-193, 198, 236 (riser move, covered by HW-L3), 240 (card lost
  from the bus, covered by HW-L3), 247, 265, 284, 323-331, 340, and 360-362 / 366-369 (headers).
- Host and method rows whose only source was a working note that isn't published here were left out.
- Several LEVERS rows describe the same lever as another row. Those were merged into one registry row,
  and the merged row lists every source line in `source_levers_line`.

`needs-check` marks 116 of the 392 rows. Most are rows where the status was inferred from a later row, the
exactness wasn't stated, or the date was missing. Treat those values as unconfirmed.
