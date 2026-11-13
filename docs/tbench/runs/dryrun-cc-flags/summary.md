## dryrun-cc-flags
agent `claude-code`, model alias `claude-opus-5-5`, proxy mode `flash`, n-concurrent 1, harbor 0.24.0, Claude Code 2.1.303

| metric | value |
|---|---|
| trials / tasks | 1 / 1 |
| score (mean reward) | **0.0%** (95% bootstrap CI 0.0-0.0) |
| strict pass (reward = 1) | 0/1 = 0.0% (Wilson 95% CI 0.0-79.3) |
| wall time | 0.02 h (2026-10-21 08:50Z .. 08:51Z) |
| throughput | 42.8 trials/hour |
| agent time per trial | median 1.0 min, p90 1.0 min, sum 0.0 h |
| exceptions | AgentTimeoutError: 1 |
| proxy requests | 28 (28 failed); tokens in 0.00 M (cached 0.00 M), out 0.00 M |
| proxy non-local deployments | none |

| difficulty (author label) | n | score |
|---|---|---|
| hard | 1 | 0% |

| proxy: requested -> deployment | status | n |
|---|---|---|
| claude-opus-5-5 -> qwen3.8-flash-next | fail | 28 |

### per task

| task | diff | reward | agent min | timeout min | in k | out k | exception |
|---|---|---|---|---|---|---|---|
| application-debug | hard | 0.00 | 1.0 | 10 | 0 | 0.0 | AgentTimeoutError |
