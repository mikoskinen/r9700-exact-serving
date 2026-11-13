## tblite-sage-20261024-1814
agent `claude-code`, model alias `claude-sonnet-4-5` -> engine model `hosted_vllm/sage-a1.5-cpp`, proxy mode `sabi`, n-concurrent 24, harbor 0.24.0, Claude Code 2.1.303

| metric | value |
|---|---|
| trials / tasks | 60 / 60 |
| score (mean reward) | **43.8%** (95% bootstrap CI 32.0-56.3) |
| strict pass (reward = 1) | 23/60 = 38.3% (Wilson 95% CI 27.1-51.0) |
| wall time | 0.87 h (2026-10-24 15:15Z .. 16:07Z) |
| throughput | 69.3 trials/hour |
| agent time per trial | median 12.0 min, p90 20.0 min, sum 12.3 h |
| agent tokens (harbor) | in 61.65 M (cached 57.23 M), out 1.74 M; per trial in 1027 k / out 29.0 k |
| exceptions | AgentTimeoutError: 37, UnknownApiError: 2 |
| proxy requests | 1026 (4 failed); tokens in 64.39 M (cached 58.77 M), out 1.94 M |
| proxy non-local deployments | **4** {(None, 'claude-sonnet-4-5', None): 4} |

| difficulty (author label) | n | score |
|---|---|---|
| medium | 27 | 35% |
| easy | 18 | 50% |
| hard | 15 | 52% |

| proxy: requested -> deployment | status | n |
|---|---|---|
| cc-sonnet-b -> sage-a1.5-cpp | ok | 513 |
| cc-sonnet-a -> sage-a1.5-cpp | ok | 509 |
| None -> claude-sonnet-4-5 | fail | 4 |

### per task

| task | diff | reward | agent min | timeout min | in k | out k | exception |
|---|---|---|---|---|---|---|---|
| acl-permissions-inheritance | medium | 1.00 | 3.9 | 15 | 106 | 1.4 |  |
| anomaly-detection-ranking | hard | 1.00 | 13.4 | 60 | 945 | 19.4 |  |
| api-endpoint-permission-canonicalizer | medium | 0.00 | 15.0 | 15 | 27 | 16.1 | AgentTimeoutError |
| application-debug | hard | 0.00 | 10.0 | 10 | 671 | 9.6 | AgentTimeoutError |
| auth_token_race_condition | hard | 1.00 | 3.8 | 10 | 168 | 1.4 |  |
| bandit-delayed-feedback | medium | 0.00 | 15.0 | 15 | 672 | 21.8 | AgentTimeoutError |
| bash-log-processor-fix | hard | 0.75 | 30.5 | 60 | 2860 | 63.7 |  |
| broken-python | easy | 1.00 | 6.9 | 15 | 344 | 3.9 |  |
| build-merkle-tree-cli-sha512 | easy | 1.00 | 15.0 | 15 | 855 | 39.1 | AgentTimeoutError |
| build-system-task-ordering | medium | 0.00 | 15.0 | 15 | 0 | 0.0 | AgentTimeoutError |
| california-housing-api | easy | 0.00 | 10.0 | 10 | 536 | 8.0 | AgentTimeoutError |
| chained-forensic-extraction_20260101_011957 | hard | 0.00 | 30.0 | 30 | 954 | 68.9 | AgentTimeoutError |
| container-registry-optimization | hard | 0.00 | 5.0 | 5 | 84 | 8.2 | AgentTimeoutError |
| convolutional-layers | easy | 0.00 | 12.0 | 12 | 163 | 16.8 | AgentTimeoutError |
| cosign-keyless-signing | easy | 1.00 | 15.0 | 15 | 798 | 27.6 | AgentTimeoutError |
| cpp-daemon-sighup-segfault | medium | 1.00 | 15.0 | 15 | 966 | 18.4 | AgentTimeoutError |
| cryptographic-protocol-verifier | easy | 0.00 | 15.0 | 15 | 121 | 19.6 | AgentTimeoutError |
| csv-json-jsonl-merger | medium | 1.00 | 4.3 | 15 | 147 | 2.1 |  |
| db-migration-local-storage | hard | 1.00 | 10.0 | 10 | 579 | 15.9 | AgentTimeoutError |
| distributed-test-execution-scheduler | medium | 0.00 | 20.0 | 20 | 293 | 31.2 | AgentTimeoutError |
| ekf-localization | easy | 0.00 | 10.0 | 10 | 27 | 0.2 | AgentTimeoutError |
| fix-js-network-controller | medium | 0.00 | 10.0 | 10 | 54 | 14.1 | AgentTimeoutError |
| fix_async_worker_queue | hard | 1.00 | 10.0 | 10 | 444 | 7.4 | AgentTimeoutError |
| hydra-debug-slurm-mode | easy | 0.00 | 6.0 | 6 | 138 | 1.3 | AgentTimeoutError |
| industrial-kiln-controller | medium | 0.00 | 10.0 | 10 | 182 | 43.1 | AgentTimeoutError |
| iot-device-registration-server | medium | 1.00 | 16.1 | 60 | 2615 | 44.8 |  |
| jq-data-processing | easy | 1.00 | 1.7 | 15 | 812 | 8.7 |  |
| jsonl-aggregator | easy | 1.00 | 0.6 | 15 | 193 | 2.3 |  |
| log-summary | easy | 1.00 | 0.2 | 15 | 133 | 0.8 |  |
| malicious-package-forensics | medium | 1.00 | 6.7 | 15 | 472 | 28.6 |  |
| maven-slf4j-conflict | medium | 0.00 | 10.0 | 10 | 1737 | 23.9 | AgentTimeoutError |
| mech-system | hard | 1.00 | 15.0 | 15 | 1207 | 47.0 | AgentTimeoutError |
| mlflow-register | easy | 1.00 | 2.3 | 15 | 644 | 6.1 |  |
| monorepo-changelog-cli | medium | 0.00 | 10.0 | 10 | 53 | 16.2 | AgentTimeoutError |
| mtls-cert-rotation | medium | 1.00 | 4.4 | 10 | 474 | 10.8 |  |
| multi-labeller | hard | 0.00 | 15.0 | 15 | 27 | 16.1 | AgentTimeoutError |
| neural-architecture-search-final | medium | 0.00 | 13.3 | 13 | 164 | 32.6 | AgentTimeoutError |
| neutron-submission | medium | 0.00 | 15.0 | 15 | 195 | 36.8 | AgentTimeoutError |
| pandas-etl | easy | 1.00 | 1.9 | 15 | 205 | 4.2 |  |
| pgn-chess-repair-puzzles | medium | 0.00 | 40.2 | 60 | 3542 | 200.6 | UnknownApiError |
| python-api-rate-limit | medium | 0.00 | 15.0 | 15 | 262 | 29.5 | AgentTimeoutError |
| raft-log-repair-concurrent-access | easy | 0.00 | 15.0 | 15 | 73 | 33.7 | AgentTimeoutError |
| reproducibility-and-envsetup | hard | 1.00 | 6.0 | 6 | 563 | 12.6 | AgentTimeoutError |
| reverse-engineer-stack-vm | hard | 0.00 | 13.3 | 13 | 1273 | 28.7 | AgentTimeoutError |
| rsa-jwt-token-api-redis-blacklist | medium | 0.00 | 15.0 | 15 | 2012 | 32.6 | AgentTimeoutError |
| sales-data-csv-analysis | medium | 0.00 | 9.8 | 60 | 739 | 26.0 |  |
| scan-linux-persistence-artifacts | medium | 0.00 | 15.0 | 15 | 328 | 26.5 | AgentTimeoutError |
| schedule-vacation | easy | 1.00 | 2.7 | 15 | 271 | 4.6 |  |
| security-breach-incident-response | medium | 0.83 | 8.0 | 60 | 783 | 16.2 |  |
| security-incident-log-analysis | medium | 0.88 | 7.4 | 60 | 581 | 16.0 |  |
| service-deployment-wave-planner | hard | 0.00 | 28.5 | 30 | 12476 | 161.0 | UnknownApiError |
| sign-vector-game | medium | 0.00 | 15.0 | 15 | 532 | 31.4 | AgentTimeoutError |
| submission_a63937a5_20251224_152124 | hard | 0.00 | 30.0 | 30 | 2528 | 122.2 | AgentTimeoutError |
| supply-chain-fulfillment | easy | 0.00 | 10.0 | 10 | 201 | 15.3 | AgentTimeoutError |
| systemd-log-monitoring | easy | 0.00 | 15.0 | 15 | 1531 | 29.1 | AgentTimeoutError |
| task-xxe-exploit | medium | 1.00 | 3.7 | 5 | 509 | 6.7 |  |
| todos-api | medium | 0.00 | 8.3 | 8 | 97 | 16.6 | AgentTimeoutError |
| token-auth-websocket | hard | 1.00 | 26.4 | 30 | 8479 | 89.4 |  |
| tsl-test-case-generation | medium | 0.82 | 16.1 | 60 | 1571 | 74.1 |  |
| vimscript-vim-quine | easy | 0.00 | 15.0 | 15 | 2230 | 31.1 | AgentTimeoutError |
