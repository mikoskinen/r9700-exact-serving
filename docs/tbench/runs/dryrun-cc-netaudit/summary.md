## dryrun-cc-netaudit
agent `claude-code`, model alias `claude-sonnet-4-5`, proxy mode `flash`, n-concurrent 1, harbor 0.24.0, Claude Code 2.1.303

| metric | value |
|---|---|
| trials / tasks | 1 / 1 |
| score (mean reward) | **0.0%** (95% bootstrap CI 0.0-0.0) |
| strict pass (reward = 1) | 0/1 = 0.0% (Wilson 95% CI 0.0-79.3) |
| wall time | 0.03 h (2026-10-21 08:52Z .. 08:54Z) |
| throughput | 33.1 trials/hour |
| agent time per trial | median 1.5 min, p90 1.5 min, sum 0.0 h |
| exceptions | AgentTimeoutError: 1 |
| proxy requests | 32 (32 failed); tokens in 0.00 M (cached 0.00 M), out 0.00 M |
| proxy non-local deployments | none |

| difficulty (author label) | n | score |
|---|---|---|
| easy | 1 | 0% |

| proxy: requested -> deployment | status | n |
|---|---|---|
| cc-sonnet -> qwen3.8-flash-next | fail | 32 |

### per task

| task | diff | reward | agent min | timeout min | in k | out k | exception |
|---|---|---|---|---|---|---|---|
| broken-python | easy | 0.00 | 1.5 | 15 | 0 | 0.0 | AgentTimeoutError |
