## night1021-flash-prod-v60-n16
agent `claude-code`, model alias `claude-sonnet-4-5` -> engine model `hosted_vllm/qwen3.8-flash-next`, proxy mode `flash`, n-concurrent 16, harbor 0.24.0, Claude Code 2.1.303

| metric | value |
|---|---|
| trials / tasks | 60 / 60 |
| score (mean reward) | **65.4%** (95% bootstrap CI 53.6-76.7) |
| strict pass (reward = 1) | 36/60 = 60.0% (Wilson 95% CI 47.4-71.4) |
| wall time | 1.37 h (2026-10-21 20:10Z .. 21:33Z) |
| throughput | 43.7 trials/hour |
| agent time per trial | median 11.2 min, p90 19.2 min, sum 12.3 h |
| agent tokens (harbor) | in 41.81 M (cached 39.76 M), out 0.89 M; per trial in 697 k / out 14.8 k |
| exceptions | AgentTimeoutError: 31 |
| proxy requests | 899 (0 failed); tokens in 42.81 M (cached 40.01 M), out 0.94 M |
| proxy non-local deployments | none |

| difficulty (author label) | n | score |
|---|---|---|
| medium | 27 | 57% |
| easy | 18 | 67% |
| hard | 15 | 78% |

| proxy: requested -> deployment | status | n |
|---|---|---|
| cc-sonnet -> qwen3.8-flash-next | ok | 899 |

### per task

| task | diff | reward | agent min | timeout min | in k | out k | exception |
|---|---|---|---|---|---|---|---|
| acl-permissions-inheritance | medium | 1.00 | 5.7 | 15 | 219 | 2.2 |  |
| anomaly-detection-ranking | hard | 1.00 | 20.1 | 60 | 1033 | 18.3 |  |
| api-endpoint-permission-canonicalizer | medium | 0.00 | 15.0 | 15 | 562 | 17.1 | AgentTimeoutError |
| application-debug | hard | 0.00 | 7.8 | 10 | 344 | 4.8 |  |
| auth_token_race_condition | hard | 1.00 | 3.8 | 10 | 112 | 2.0 |  |
| bandit-delayed-feedback | medium | 0.00 | 15.0 | 15 | 375 | 13.7 | AgentTimeoutError |
| bash-log-processor-fix | hard | 0.75 | 21.5 | 60 | 1007 | 22.2 |  |
| broken-python | easy | 1.00 | 8.2 | 15 | 342 | 4.1 |  |
| build-merkle-tree-cli-sha512 | easy | 1.00 | 15.0 | 15 | 669 | 16.6 | AgentTimeoutError |
| build-system-task-ordering | medium | 0.00 | 15.0 | 15 | 443 | 20.0 | AgentTimeoutError |
| california-housing-api | easy | 0.00 | 10.0 | 10 | 342 | 6.2 | AgentTimeoutError |
| chained-forensic-extraction_20260101_011957 | hard | 1.00 | 8.0 | 30 | 334 | 5.2 |  |
| container-registry-optimization | hard | 0.00 | 5.0 | 5 | 181 | 4.1 | AgentTimeoutError |
| convolutional-layers | easy | 1.00 | 12.0 | 12 | 572 | 8.0 | AgentTimeoutError |
| cosign-keyless-signing | easy | 1.00 | 15.0 | 15 | 569 | 17.1 | AgentTimeoutError |
| cpp-daemon-sighup-segfault | medium | 0.00 | 15.0 | 15 | 566 | 12.0 | AgentTimeoutError |
| cryptographic-protocol-verifier | easy | 0.00 | 15.0 | 15 | 125 | 21.0 | AgentTimeoutError |
| csv-json-jsonl-merger | medium | 1.00 | 7.0 | 15 | 350 | 5.0 |  |
| db-migration-local-storage | hard | 1.00 | 9.0 | 10 | 426 | 7.6 |  |
| distributed-test-execution-scheduler | medium | 0.00 | 20.0 | 20 | 853 | 26.6 | AgentTimeoutError |
| ekf-localization | easy | 1.00 | 3.8 | 10 | 169 | 2.7 |  |
| fix-js-network-controller | medium | 1.00 | 8.0 | 10 | 167 | 10.1 |  |
| fix_async_worker_queue | hard | 1.00 | 6.3 | 10 | 372 | 2.7 |  |
| hydra-debug-slurm-mode | easy | 0.00 | 6.0 | 6 | 197 | 2.3 | AgentTimeoutError |
| industrial-kiln-controller | medium | 0.00 | 10.0 | 10 | 150 | 9.6 | AgentTimeoutError |
| iot-device-registration-server | medium | 1.00 | 9.3 | 60 | 416 | 7.8 |  |
| jq-data-processing | easy | 1.00 | 6.5 | 15 | 330 | 3.9 |  |
| jsonl-aggregator | easy | 1.00 | 7.5 | 15 | 331 | 5.2 |  |
| log-summary | easy | 1.00 | 1.2 | 15 | 78 | 0.4 |  |
| malicious-package-forensics | medium | 1.00 | 11.2 | 15 | 572 | 14.7 |  |
| maven-slf4j-conflict | medium | 0.00 | 10.0 | 10 | 521 | 6.7 | AgentTimeoutError |
| mech-system | hard | 1.00 | 15.0 | 15 | 480 | 22.6 | AgentTimeoutError |
| mlflow-register | easy | 1.00 | 15.0 | 15 | 607 | 6.0 | AgentTimeoutError |
| monorepo-changelog-cli | medium | 1.00 | 9.2 | 10 | 318 | 8.9 |  |
| mtls-cert-rotation | medium | 1.00 | 4.5 | 10 | 206 | 3.1 |  |
| multi-labeller | hard | 1.00 | 15.0 | 15 | 178 | 23.8 | AgentTimeoutError |
| neural-architecture-search-final | medium | 0.00 | 13.3 | 13 | 90 | 11.7 | AgentTimeoutError |
| neutron-submission | medium | 1.00 | 14.2 | 15 | 409 | 17.2 |  |
| pandas-etl | easy | 1.00 | 6.6 | 15 | 285 | 2.5 |  |
| pgn-chess-repair-puzzles | medium | 0.00 | 60.0 | 60 | 8967 | 99.1 | AgentTimeoutError |
| python-api-rate-limit | medium | 1.00 | 15.0 | 15 | 508 | 13.0 | AgentTimeoutError |
| raft-log-repair-concurrent-access | easy | 1.00 | 15.0 | 15 | 395 | 18.5 | AgentTimeoutError |
| reproducibility-and-envsetup | hard | 1.00 | 6.0 | 6 | 241 | 4.2 | AgentTimeoutError |
| reverse-engineer-stack-vm | hard | 0.00 | 13.3 | 13 | 342 | 8.2 | AgentTimeoutError |
| rsa-jwt-token-api-redis-blacklist | medium | 0.00 | 15.0 | 15 | 522 | 16.4 | AgentTimeoutError |
| sales-data-csv-analysis | medium | 1.00 | 10.5 | 60 | 306 | 10.7 |  |
| scan-linux-persistence-artifacts | medium | 1.00 | 15.0 | 15 | 392 | 17.7 | AgentTimeoutError |
| schedule-vacation | easy | 1.00 | 8.7 | 15 | 310 | 7.4 |  |
| security-breach-incident-response | medium | 0.77 | 20.1 | 60 | 1284 | 24.5 |  |
| security-incident-log-analysis | medium | 0.88 | 6.9 | 60 | 376 | 4.3 |  |
| service-deployment-wave-planner | hard | 1.00 | 30.0 | 30 | 4844 | 85.2 | AgentTimeoutError |
| sign-vector-game | medium | 1.00 | 7.8 | 15 | 277 | 5.4 |  |
| submission_a63937a5_20251224_152124 | hard | 1.00 | 11.3 | 30 | 639 | 9.8 |  |
| supply-chain-fulfillment | easy | 0.00 | 10.0 | 10 | 212 | 13.4 | AgentTimeoutError |
| systemd-log-monitoring | easy | 0.00 | 15.0 | 15 | 727 | 26.1 | AgentTimeoutError |
| task-xxe-exploit | medium | 0.00 | 5.0 | 5 | 201 | 3.3 | AgentTimeoutError |
| todos-api | medium | 1.00 | 8.3 | 8 | 152 | 14.6 | AgentTimeoutError |
| token-auth-websocket | hard | 1.00 | 19.2 | 30 | 2559 | 49.9 |  |
| tsl-test-case-generation | medium | 0.83 | 14.1 | 60 | 2109 | 29.1 |  |
| vimscript-vim-quine | easy | 0.00 | 15.0 | 15 | 1147 | 32.6 | AgentTimeoutError |
