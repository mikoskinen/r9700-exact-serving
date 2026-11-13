## tblite-sage-20261024-1814
OpenThoughts-TBLite @ f075e463 (Harbor `openthoughts-tblite@2.0`), subset `easy,medium` (60 tasks, statuses oracle,oracle-partial,public-solved), attempts 1
agent `claude-code`, model alias `claude-sonnet-4-5` -> engine `sage-a1.5-cpp`, proxy mode `sabi`, n-concurrent 24, agent timeout x1.0, harbor 0.24.0, Claude Code 2.1.303

| metric | value |
|---|---|
| trials / tasks | 60 / 60 |
| **score** (mean reward) | **43.8%** (95% bootstrap CI 32.0%-56.3%) |
| strict pass (reward = 1) | 23/60 = 38.3% (Wilson 95% CI 27.1%-51.0%) |
| agent timeouts | 37/60 (of which 7 still reward 1: the verifier grades what is on disk) |
| other exceptions | UnknownApiError: 2 |
| wall time | 0.87 h (2026-10-24 15:15Z .. 16:07Z), 69.3 trials/h |
| agent minutes per trial | median 11.0, mean 12.3, p90 20.0, max 40.2; sum 12.3 h |
| non-agent minutes per trial (build+setup+verify) | median 0.53, max 4.4 |
| tokens per trial (harbor) | in mean 1027 k (median 509 k, cached 93%), out mean 29.0 k (median 18.4 k); total in 61.65 M, out 1.74 M |
| proxy requests | 1026 (4 failed) |
| proxy non-local deployments | **4** {(None, 'claude-sonnet-4-5', None): 4} |

| difficulty (author label) | trials | score | strict pass (Wilson 95% CI) | timeouts | agent min median |
|---|---|---|---|---|---|
| easy | 18 | 50.0% | 9/18 (29.0%-71.0%) | 11 | 10.0 |
| medium | 27 | 35.3% | 7/27 (13.2%-44.7%) | 16 | 13.3 |
| hard | 15 | 51.6% | 7/15 (24.8%-69.9%) | 10 | 13.4 |

| validation status | trials | score | strict |
|---|---|---|---|
| oracle | 56 | 41.1% | 23/56 |
| oracle-partial | 4 | 82.1% | 0/4 |

| proxy: requested -> deployment | status | n |
|---|---|---|
| cc-sonnet-b -> sage-a1.5-cpp | ok | 513 |
| cc-sonnet-a -> sage-a1.5-cpp | ok | 509 |
| None -> claude-sonnet-4-5 | fail | 4 |

### per task

| task | diff | status | reward | agent min | timeout min | in k | out k | exception |
|---|---|---|---|---|---|---|---|---|
| broken-python | easy | oracle | 1.00 | 6.9 | 15 | 344 | 3.9 |  |
| build-merkle-tree-cli-sha512 | easy | oracle | 1.00 | 15.0 | 15 | 855 | 39.1 | AgentTimeoutError |
| california-housing-api | easy | oracle | 0.00 | 10.0 | 10 | 536 | 8.0 | AgentTimeoutError |
| convolutional-layers | easy | oracle | 0.00 | 12.0 | 12 | 163 | 16.8 | AgentTimeoutError |
| cosign-keyless-signing | easy | oracle | 1.00 | 15.0 | 15 | 798 | 27.6 | AgentTimeoutError |
| cryptographic-protocol-verifier | easy | oracle | 0.00 | 15.0 | 15 | 121 | 19.6 | AgentTimeoutError |
| ekf-localization | easy | oracle | 0.00 | 10.0 | 10 | 27 | 0.2 | AgentTimeoutError |
| hydra-debug-slurm-mode | easy | oracle | 0.00 | 6.0 | 6 | 138 | 1.3 | AgentTimeoutError |
| jq-data-processing | easy | oracle | 1.00 | 1.7 | 15 | 812 | 8.7 |  |
| jsonl-aggregator | easy | oracle | 1.00 | 0.6 | 15 | 193 | 2.3 |  |
| log-summary | easy | oracle | 1.00 | 0.2 | 15 | 133 | 0.8 |  |
| mlflow-register | easy | oracle | 1.00 | 2.3 | 15 | 644 | 6.1 |  |
| pandas-etl | easy | oracle | 1.00 | 1.9 | 15 | 205 | 4.2 |  |
| raft-log-repair-concurrent-access | easy | oracle | 0.00 | 15.0 | 15 | 73 | 33.7 | AgentTimeoutError |
| schedule-vacation | easy | oracle | 1.00 | 2.7 | 15 | 271 | 4.6 |  |
| supply-chain-fulfillment | easy | oracle | 0.00 | 10.0 | 10 | 201 | 15.3 | AgentTimeoutError |
| systemd-log-monitoring | easy | oracle | 0.00 | 15.0 | 15 | 1531 | 29.1 | AgentTimeoutError |
| vimscript-vim-quine | easy | oracle | 0.00 | 15.0 | 15 | 2230 | 31.1 | AgentTimeoutError |
| acl-permissions-inheritance | medium | oracle | 1.00 | 3.9 | 15 | 106 | 1.4 |  |
| api-endpoint-permission-canonicalizer | medium | oracle | 0.00 | 15.0 | 15 | 27 | 16.1 | AgentTimeoutError |
| bandit-delayed-feedback | medium | oracle | 0.00 | 15.0 | 15 | 672 | 21.8 | AgentTimeoutError |
| build-system-task-ordering | medium | oracle | 0.00 | 15.0 | 15 | 0 | 0.0 | AgentTimeoutError |
| cpp-daemon-sighup-segfault | medium | oracle | 1.00 | 15.0 | 15 | 966 | 18.4 | AgentTimeoutError |
| csv-json-jsonl-merger | medium | oracle | 1.00 | 4.3 | 15 | 147 | 2.1 |  |
| distributed-test-execution-scheduler | medium | oracle | 0.00 | 20.0 | 20 | 293 | 31.2 | AgentTimeoutError |
| fix-js-network-controller | medium | oracle | 0.00 | 10.0 | 10 | 54 | 14.1 | AgentTimeoutError |
| industrial-kiln-controller | medium | oracle | 0.00 | 10.0 | 10 | 182 | 43.1 | AgentTimeoutError |
| iot-device-registration-server | medium | oracle | 1.00 | 16.1 | 60 | 2615 | 44.8 |  |
| malicious-package-forensics | medium | oracle | 1.00 | 6.7 | 15 | 472 | 28.6 |  |
| maven-slf4j-conflict | medium | oracle | 0.00 | 10.0 | 10 | 1737 | 23.9 | AgentTimeoutError |
| monorepo-changelog-cli | medium | oracle | 0.00 | 10.0 | 10 | 53 | 16.2 | AgentTimeoutError |
| mtls-cert-rotation | medium | oracle | 1.00 | 4.4 | 10 | 474 | 10.8 |  |
| neural-architecture-search-final | medium | oracle | 0.00 | 13.3 | 13 | 164 | 32.6 | AgentTimeoutError |
| neutron-submission | medium | oracle | 0.00 | 15.0 | 15 | 195 | 36.8 | AgentTimeoutError |
| pgn-chess-repair-puzzles | medium | oracle | 0.00 | 40.2 | 60 | 3542 | 200.6 | UnknownApiError |
| python-api-rate-limit | medium | oracle | 0.00 | 15.0 | 15 | 262 | 29.5 | AgentTimeoutError |
| rsa-jwt-token-api-redis-blacklist | medium | oracle | 0.00 | 15.0 | 15 | 2012 | 32.6 | AgentTimeoutError |
| sales-data-csv-analysis | medium | oracle | 0.00 | 9.8 | 60 | 739 | 26.0 |  |
| scan-linux-persistence-artifacts | medium | oracle | 0.00 | 15.0 | 15 | 328 | 26.5 | AgentTimeoutError |
| security-breach-incident-response | medium | oracle-partial | 0.83 | 8.0 | 60 | 783 | 16.2 |  |
| security-incident-log-analysis | medium | oracle-partial | 0.88 | 7.4 | 60 | 581 | 16.0 |  |
| sign-vector-game | medium | oracle | 0.00 | 15.0 | 15 | 532 | 31.4 | AgentTimeoutError |
| task-xxe-exploit | medium | oracle | 1.00 | 3.7 | 5 | 509 | 6.7 |  |
| todos-api | medium | oracle | 0.00 | 8.3 | 8 | 97 | 16.6 | AgentTimeoutError |
| tsl-test-case-generation | medium | oracle-partial | 0.82 | 16.1 | 60 | 1571 | 74.1 |  |
| anomaly-detection-ranking | hard | oracle | 1.00 | 13.4 | 60 | 945 | 19.4 |  |
| application-debug | hard | oracle | 0.00 | 10.0 | 10 | 671 | 9.6 | AgentTimeoutError |
| auth_token_race_condition | hard | oracle | 1.00 | 3.8 | 10 | 168 | 1.4 |  |
| bash-log-processor-fix | hard | oracle-partial | 0.75 | 30.5 | 60 | 2860 | 63.7 |  |
| chained-forensic-extraction_20260101_011957 | hard | oracle | 0.00 | 30.0 | 30 | 954 | 68.9 | AgentTimeoutError |
| container-registry-optimization | hard | oracle | 0.00 | 5.0 | 5 | 84 | 8.2 | AgentTimeoutError |
| db-migration-local-storage | hard | oracle | 1.00 | 10.0 | 10 | 579 | 15.9 | AgentTimeoutError |
| fix_async_worker_queue | hard | oracle | 1.00 | 10.0 | 10 | 444 | 7.4 | AgentTimeoutError |
| mech-system | hard | oracle | 1.00 | 15.0 | 15 | 1207 | 47.0 | AgentTimeoutError |
| multi-labeller | hard | oracle | 0.00 | 15.0 | 15 | 27 | 16.1 | AgentTimeoutError |
| reproducibility-and-envsetup | hard | oracle | 1.00 | 6.0 | 6 | 563 | 12.6 | AgentTimeoutError |
| reverse-engineer-stack-vm | hard | oracle | 0.00 | 13.3 | 13 | 1273 | 28.7 | AgentTimeoutError |
| service-deployment-wave-planner | hard | oracle | 0.00 | 28.5 | 30 | 12476 | 161.0 | UnknownApiError |
| submission_a63937a5_20251224_152124 | hard | oracle | 0.00 | 30.0 | 30 | 2528 | 122.2 | AgentTimeoutError |
| token-auth-websocket | hard | oracle | 1.00 | 26.4 | 30 | 8479 | 89.4 |  |
