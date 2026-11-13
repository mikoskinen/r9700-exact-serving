## mellum-pilot10-cc
agent `claude-code`, model alias `claude-sonnet-4-5` -> engine model `hosted_vllm/mellum2.1-12b`, proxy mode `sabi`, n-concurrent 8, harbor 0.24.0, Claude Code 2.1.303

| metric | value |
|---|---|
| trials / tasks | 10 / 10 |
| score (mean reward) | **30.0%** (95% bootstrap CI 0.0-60.0) |
| strict pass (reward = 1) | 3/10 = 30.0% (Wilson 95% CI 10.8-60.3) |
| wall time | 0.26 h (2026-10-21 14:13Z .. 14:28Z) |
| throughput | 38.3 trials/hour |
| agent time per trial | median 4.7 min, p90 10.0 min, sum 1.0 h |
| agent tokens (harbor) | in 5.85 M (cached 5.45 M), out 0.11 M; per trial in 585 k / out 10.9 k |
| exceptions | AgentTimeoutError: 2 |
| proxy requests | 148 (0 failed); tokens in 6.09 M (cached 5.61 M), out 0.13 M |
| proxy non-local deployments | none |

| difficulty (author label) | n | score |
|---|---|---|
| medium | 4 | 0% |
| hard | 3 | 33% |
| easy | 3 | 67% |

| proxy: requested -> deployment | status | n |
|---|---|---|
| cc-sonnet-b -> mellum2.1-12b | ok | 119 |
| cc-sonnet-a -> mellum2.1-12b | ok | 29 |

### per task

| task | diff | reward | agent min | timeout min | in k | out k | exception |
|---|---|---|---|---|---|---|---|
| application-debug | hard | 0.00 | 10.0 | 10 | 202 | 0.6 | AgentTimeoutError |
| broken-python | easy | 1.00 | 4.7 | 15 | 1400 | 5.9 |  |
| cpp-daemon-sighup-segfault | medium | 0.00 | 15.0 | 15 | 1386 | 37.9 | AgentTimeoutError |
| csv-json-jsonl-merger | medium | 0.00 | 1.9 | 15 | 85 | 2.5 |  |
| db-migration-local-storage | hard | 1.00 | 3.6 | 10 | 174 | 12.1 |  |
| fix_async_worker_queue | hard | 0.00 | 4.2 | 10 | 928 | 14.6 |  |
| jq-data-processing | easy | 0.00 | 1.7 | 15 | 28 | 4.6 |  |
| log-summary | easy | 1.00 | 1.0 | 15 | 109 | 0.8 |  |
| malicious-package-forensics | medium | 0.00 | 8.2 | 15 | 406 | 15.4 |  |
| python-api-rate-limit | medium | 0.00 | 7.8 | 15 | 1126 | 14.5 |  |
