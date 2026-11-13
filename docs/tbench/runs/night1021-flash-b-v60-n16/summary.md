## night1021-flash-b-v60-n16
agent `claude-code`, model alias `claude-sonnet-4-5` -> engine model `hosted_vllm/qwen3.8-flash-next`, proxy mode `flash`, n-concurrent 16, harbor 0.24.0, Claude Code 2.1.303

| metric | value |
|---|---|
| trials / tasks | 60 / 60 |
| score (mean reward) | **65.0%** (95% bootstrap CI 52.9-76.0) |
| strict pass (reward = 1) | 35/60 = 58.3% (Wilson 95% CI 45.7-69.9) |
| wall time | 1.34 h (2026-10-21 21:37Z .. 22:57Z) |
| throughput | 44.9 trials/hour |
| agent time per trial | median 10.1 min, p90 20.0 min, sum 12.5 h |
| agent tokens (harbor) | in 37.87 M (cached 35.89 M), out 0.85 M; per trial in 631 k / out 14.2 k |
| exceptions | AgentTimeoutError: 29 |
| proxy requests | 873 (0 failed); tokens in 38.99 M (cached 36.66 M), out 0.90 M |
| proxy non-local deployments | none |

| difficulty (author label) | n | score |
|---|---|---|
| medium | 27 | 64% |
| easy | 18 | 67% |
| hard | 15 | 65% |

| proxy: requested -> deployment | status | n |
|---|---|---|
| cc-sonnet -> qwen3.8-flash-next | ok | 873 |

### per task

| task | diff | reward | agent min | timeout min | in k | out k | exception |
|---|---|---|---|---|---|---|---|
| acl-permissions-inheritance | medium | 1.00 | 5.2 | 15 | 195 | 3.0 |  |
| anomaly-detection-ranking | hard | 1.00 | 10.1 | 60 | 455 | 8.3 |  |
| api-endpoint-permission-canonicalizer | medium | 1.00 | 10.7 | 15 | 315 | 13.0 |  |
| application-debug | hard | 1.00 | 5.1 | 10 | 204 | 2.5 |  |
| auth_token_race_condition | hard | 1.00 | 3.7 | 10 | 139 | 1.2 |  |
| bandit-delayed-feedback | medium | 0.00 | 15.0 | 15 | 612 | 11.1 | AgentTimeoutError |
| bash-log-processor-fix | hard | 0.75 | 41.5 | 60 | 1787 | 32.9 |  |
| broken-python | easy | 1.00 | 6.3 | 15 | 280 | 3.1 |  |
| build-merkle-tree-cli-sha512 | easy | 1.00 | 15.0 | 15 | 649 | 16.3 | AgentTimeoutError |
| build-system-task-ordering | medium | 1.00 | 14.7 | 15 | 495 | 17.1 |  |
| california-housing-api | easy | 0.00 | 10.0 | 10 | 409 | 6.2 | AgentTimeoutError |
| chained-forensic-extraction_20260101_011957 | hard | 1.00 | 9.3 | 30 | 385 | 7.7 |  |
| container-registry-optimization | hard | 0.00 | 5.0 | 5 | 216 | 5.0 | AgentTimeoutError |
| convolutional-layers | easy | 1.00 | 12.0 | 12 | 548 | 12.4 | AgentTimeoutError |
| cosign-keyless-signing | easy | 1.00 | 12.0 | 15 | 465 | 13.5 |  |
| cpp-daemon-sighup-segfault | medium | 0.00 | 15.0 | 15 | 600 | 11.0 | AgentTimeoutError |
| cryptographic-protocol-verifier | easy | 0.00 | 15.0 | 15 | 309 | 11.2 | AgentTimeoutError |
| csv-json-jsonl-merger | medium | 1.00 | 5.4 | 15 | 257 | 3.6 |  |
| db-migration-local-storage | hard | 1.00 | 10.0 | 10 | 435 | 8.1 | AgentTimeoutError |
| distributed-test-execution-scheduler | medium | 1.00 | 20.0 | 20 | 736 | 21.7 | AgentTimeoutError |
| ekf-localization | easy | 1.00 | 7.9 | 10 | 317 | 8.0 |  |
| fix-js-network-controller | medium | 1.00 | 10.0 | 10 | 265 | 9.7 | AgentTimeoutError |
| fix_async_worker_queue | hard | 1.00 | 9.6 | 10 | 513 | 4.6 |  |
| hydra-debug-slurm-mode | easy | 0.00 | 6.0 | 6 | 110 | 4.4 | AgentTimeoutError |
| industrial-kiln-controller | medium | 0.00 | 10.0 | 10 | 85 | 8.6 | AgentTimeoutError |
| iot-device-registration-server | medium | 1.00 | 7.0 | 60 | 335 | 5.1 |  |
| jq-data-processing | easy | 1.00 | 3.8 | 15 | 175 | 2.5 |  |
| jsonl-aggregator | easy | 1.00 | 5.9 | 15 | 293 | 4.4 |  |
| log-summary | easy | 1.00 | 3.0 | 15 | 159 | 0.7 |  |
| malicious-package-forensics | medium | 1.00 | 11.3 | 15 | 604 | 16.0 |  |
| maven-slf4j-conflict | medium | 0.00 | 10.0 | 10 | 401 | 4.7 | AgentTimeoutError |
| mech-system | hard | 0.00 | 15.0 | 15 | 454 | 17.3 | AgentTimeoutError |
| mlflow-register | easy | 1.00 | 7.0 | 15 | 288 | 3.8 |  |
| monorepo-changelog-cli | medium | 1.00 | 10.0 | 10 | 402 | 9.9 | AgentTimeoutError |
| mtls-cert-rotation | medium | 1.00 | 3.9 | 10 | 172 | 2.7 |  |
| multi-labeller | hard | 0.00 | 15.0 | 15 | 283 | 21.0 | AgentTimeoutError |
| neural-architecture-search-final | medium | 0.00 | 13.3 | 13 | 207 | 19.1 | AgentTimeoutError |
| neutron-submission | medium | 0.00 | 15.0 | 15 | 437 | 20.4 | AgentTimeoutError |
| pandas-etl | easy | 1.00 | 5.4 | 15 | 254 | 2.7 |  |
| pgn-chess-repair-puzzles | medium | 0.76 | 60.0 | 60 | 5856 | 82.5 | AgentTimeoutError |
| python-api-rate-limit | medium | 1.00 | 12.3 | 15 | 523 | 9.8 |  |
| raft-log-repair-concurrent-access | easy | 1.00 | 15.0 | 15 | 496 | 19.1 | AgentTimeoutError |
| reproducibility-and-envsetup | hard | 0.00 | 6.0 | 6 | 207 | 4.1 | AgentTimeoutError |
| reverse-engineer-stack-vm | hard | 1.00 | 6.7 | 13 | 262 | 3.3 |  |
| rsa-jwt-token-api-redis-blacklist | medium | 1.00 | 15.0 | 15 | 526 | 16.5 | AgentTimeoutError |
| sales-data-csv-analysis | medium | 0.00 | 12.0 | 60 | 489 | 10.4 |  |
| scan-linux-persistence-artifacts | medium | 1.00 | 15.0 | 15 | 308 | 21.0 | AgentTimeoutError |
| schedule-vacation | easy | 1.00 | 6.5 | 15 | 326 | 4.7 |  |
| security-breach-incident-response | medium | 0.83 | 25.5 | 60 | 1857 | 39.3 |  |
| security-incident-log-analysis | medium | 0.84 | 13.3 | 60 | 775 | 10.1 |  |
| service-deployment-wave-planner | hard | 0.00 | 30.0 | 30 | 1509 | 57.1 | AgentTimeoutError |
| sign-vector-game | medium | 1.00 | 6.2 | 15 | 235 | 5.0 |  |
| submission_a63937a5_20251224_152124 | hard | 1.00 | 24.5 | 30 | 2203 | 36.6 |  |
| supply-chain-fulfillment | easy | 0.00 | 10.0 | 10 | 256 | 10.0 | AgentTimeoutError |
| systemd-log-monitoring | easy | 0.00 | 15.0 | 15 | 846 | 19.0 | AgentTimeoutError |
| task-xxe-exploit | medium | 0.00 | 5.0 | 5 | 143 | 2.6 | AgentTimeoutError |
| todos-api | medium | 0.00 | 8.3 | 8 | 142 | 15.1 | AgentTimeoutError |
| token-auth-websocket | hard | 1.00 | 21.3 | 30 | 3120 | 44.2 |  |
| tsl-test-case-generation | medium | 0.84 | 18.8 | 60 | 2224 | 43.9 |  |
| vimscript-vim-quine | easy | 0.00 | 15.0 | 15 | 326 | 21.5 | AgentTimeoutError |
