## dryrun-cc-nomodel
agent `claude-code`, model alias `claude-sonnet-4-5`, proxy mode `flash`, n-concurrent 2, harbor 0.24.0, Claude Code 2.1.303

| metric | value |
|---|---|
| trials / tasks | 2 / 2 |
| score (mean reward) | **0.0%** (95% bootstrap CI 0.0-0.0) |
| strict pass (reward = 1) | 0/2 = 0.0% (Wilson 95% CI 0.0-65.8) |
| wall time | 0.07 h (2026-10-21 08:38Z .. 08:42Z) |
| throughput | 29.6 trials/hour |
| agent time per trial | median 3.0 min, p90 3.0 min, sum 0.1 h |
| exceptions | AgentTimeoutError: 2 |
| proxy requests | 84 (84 failed); tokens in 0.00 M (cached 0.00 M), out 0.00 M |
| proxy non-local deployments | **84** {('cc-sonnet', 'qwen3.8-flash-next', None): 84} |

| difficulty (author label) | n | score |
|---|---|---|
| easy | 2 | 0% |

| proxy: requested -> deployment | status | n |
|---|---|---|
| cc-sonnet -> qwen3.8-flash-next | fail | 84 |

### per task

| task | diff | reward | agent min | timeout min | in k | out k | exception |
|---|---|---|---|---|---|---|---|
| jq-data-processing | easy | 0.00 | 3.0 | 15 | 0 | 0.0 | AgentTimeoutError |
| log-summary | easy | 0.00 | 3.0 | 15 | 0 | 0.0 | AgentTimeoutError |
