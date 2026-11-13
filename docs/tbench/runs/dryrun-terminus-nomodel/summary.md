## dryrun-terminus-nomodel
agent `terminus`, model alias `claude-sonnet-4-5`, proxy mode `flash`, n-concurrent 1, harbor 0.24.0

| metric | value |
|---|---|
| trials / tasks | 1 / 1 |
| score (mean reward) | **0.0%** (95% bootstrap CI 0.0-0.0) |
| strict pass (reward = 1) | 0/1 = 0.0% (Wilson 95% CI 0.0-79.3) |
| wall time | 0.05 h (2026-10-21 08:44Z .. 08:47Z) |
| throughput | 21.0 trials/hour |
| agent time per trial | median 2.1 min, p90 2.1 min, sum 0.0 h |
| exceptions | InternalServerError: 1 |
| not verified (counted as 0) | 1 |
| proxy requests | 27 (27 failed); tokens in 0.00 M (cached 0.00 M), out 0.00 M |
| proxy non-local deployments | none |

| difficulty (author label) | n | score |
|---|---|---|
| easy | 1 | 0% |

| proxy: requested -> deployment | status | n |
|---|---|---|
| cc-sonnet -> qwen3.8-flash-next | fail | 27 |

### per task

| task | diff | reward | agent min | timeout min | in k | out k | exception |
|---|---|---|---|---|---|---|---|
| log-summary | easy | 0.00 | 2.1 | 15 | 0 | 0.0 | InternalServerError |
