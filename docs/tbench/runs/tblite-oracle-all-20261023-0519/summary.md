## tblite-oracle-all-20261023-0519
OpenThoughts-TBLite @ f075e463 (Harbor `openthoughts-tblite@2.0`), subset `all` (100 tasks, statuses oracle,oracle-partial,public-solved,broken,unverified), attempts 1
agent `oracle`, model alias `None` -> engine `(oracle)`, proxy mode `-`, n-concurrent 8, agent timeout x1.0, harbor 0.24.0

| metric | value |
|---|---|
| trials / tasks | 100 / 100 |
| **score** (mean reward) | **59.2%** (95% bootstrap CI 49.3%-68.9%) |
| strict pass (reward = 1) | 56/100 = 56.0% (Wilson 95% CI 46.2%-65.3%) |
| oracle-verified tasks only | 98.6% on 60 trials; strict 56/60 (Wilson 84.1%-97.4%) |
| agent timeouts | 0/100 (of which 0 still reward 1: the verifier grades what is on disk) |
| other exceptions | none |
| wall time | 0.14 h (2026-10-23 02:19Z .. 02:27Z), 699.8 trials/h |
| agent minutes per trial | median 0.0, mean 0.1, p90 0.2, max 2.1; sum 0.2 h |
| non-agent minutes per trial (build+setup+verify) | median 0.42, max 4.2 |

| difficulty (author label) | trials | score | strict pass (Wilson 95% CI) | timeouts | agent min median |
|---|---|---|---|---|---|
| easy | 24 | 75.0% | 18/24 (55.1%-88.0%) | 0 | 0.0 |
| medium | 43 | 61.8% | 24/43 (41.1%-69.6%) | 0 | 0.0 |
| hard | 30 | 48.7% | 14/30 (30.2%-63.9%) | 0 | 0.0 |
| expert | 3 | 0.0% | 0/3 (0.0%-56.2%) | 0 | 0.0 |

| validation status | trials | score | strict |
|---|---|---|---|
| oracle | 56 | 100.0% | 56/56 |
| oracle-partial | 4 | 79.0% | 0/4 |
| public-solved | 21 | 0.0% | 0/21 |
| unverified | 5 | 0.0% | 0/5 |
| broken | 14 | 0.0% | 0/14 |

### per task

| task | diff | status | reward | agent min | timeout min | in k | out k | exception |
|---|---|---|---|---|---|---|---|---|
| amuse-install | easy | broken | 0.00 | 1.2 | 6 | 0 | 0.0 |  |
| broken-python | easy | oracle | 1.00 | 0.1 | 15 | 0 | 0.0 |  |
| build-merkle-tree-cli-sha512 | easy | oracle | 1.00 | 0.0 | 15 | 0 | 0.0 |  |
| california-housing-api | easy | oracle | 1.00 | 0.7 | 10 | 0 | 0.0 |  |
| convolutional-layers | easy | oracle | 1.00 | 0.0 | 12 | 0 | 0.0 |  |
| cosign-keyless-signing | easy | oracle | 1.00 | 0.0 | 15 | 0 | 0.0 |  |
| cryptographic-protocol-verifier | easy | oracle | 1.00 | 0.0 | 15 | 0 | 0.0 |  |
| ekf-localization | easy | oracle | 1.00 | 0.0 | 10 | 0 | 0.0 |  |
| hydra-debug-slurm-mode | easy | oracle | 1.00 | 0.1 | 6 | 0 | 0.0 |  |
| image-tile-identification | easy | public-solved | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| jq-data-processing | easy | oracle | 1.00 | 0.0 | 15 | 0 | 0.0 |  |
| jsonl-aggregator | easy | oracle | 1.00 | 0.1 | 15 | 0 | 0.0 |  |
| legal-summary-extraction | easy | unverified | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| log-summary | easy | oracle | 1.00 | 0.0 | 15 | 0 | 0.0 |  |
| mlflow-register | easy | oracle | 1.00 | 0.2 | 15 | 0 | 0.0 |  |
| pandas-etl | easy | oracle | 1.00 | 0.2 | 15 | 0 | 0.0 |  |
| playing-card-recognition | easy | public-solved | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| protein-sequence | easy | broken | 0.00 | 0.0 | 15 | 0 | 0.0 |  |
| raft-log-repair-concurrent-access | easy | oracle | 1.00 | 0.0 | 15 | 0 | 0.0 |  |
| schedule-vacation | easy | oracle | 1.00 | 0.0 | 15 | 0 | 0.0 |  |
| smiles-data-lab | easy | broken | 0.00 | 0.0 | 15 | 0 | 0.0 |  |
| supply-chain-fulfillment | easy | oracle | 1.00 | 0.0 | 10 | 0 | 0.0 |  |
| systemd-log-monitoring | easy | oracle | 1.00 | 0.0 | 15 | 0 | 0.0 |  |
| vimscript-vim-quine | easy | oracle | 1.00 | 0.0 | 15 | 0 | 0.0 |  |
| acl-permissions-inheritance | medium | oracle | 1.00 | 0.0 | 15 | 0 | 0.0 |  |
| api-endpoint-permission-canonicalizer | medium | oracle | 1.00 | 0.0 | 15 | 0 | 0.0 |  |
| bandit-delayed-feedback | medium | oracle | 1.00 | 0.2 | 15 | 0 | 0.0 |  |
| battery-charging-optimization | medium | public-solved | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| bracket-sequence-restoration | medium | public-solved | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| build-system-task-ordering | medium | oracle | 1.00 | 0.0 | 15 | 0 | 0.0 |  |
| competitive-programming-solver | medium | public-solved | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| cpp-daemon-sighup-segfault | medium | oracle | 1.00 | 0.5 | 15 | 0 | 0.0 |  |
| csv-json-jsonl-merger | medium | oracle | 1.00 | 0.0 | 15 | 0 | 0.0 |  |
| distributed-test-execution-scheduler | medium | oracle | 1.00 | 0.0 | 20 | 0 | 0.0 |  |
| fix-js-network-controller | medium | oracle | 1.00 | 0.0 | 10 | 0 | 0.0 |  |
| floor-plan-geometry | medium | public-solved | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| game-of-stones | medium | public-solved | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| git-repo-forensics | medium | broken | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| grpc-plant-position-server | medium | unverified | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| html-index-analysis | medium | public-solved | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| industrial-kiln-controller | medium | oracle | 1.00 | 0.0 | 10 | 0 | 0.0 |  |
| iot-device-registration-server | medium | oracle | 1.00 | 1.0 | 60 | 0 | 0.0 |  |
| iris-dataset-classification | medium | public-solved | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| malicious-package-forensics | medium | oracle | 1.00 | 0.0 | 15 | 0 | 0.0 |  |
| maven-slf4j-conflict | medium | oracle | 1.00 | 0.9 | 10 | 0 | 0.0 |  |
| monorepo-changelog-cli | medium | oracle | 1.00 | 0.0 | 10 | 0 | 0.0 |  |
| mtls-cert-rotation | medium | oracle | 1.00 | 0.0 | 10 | 0 | 0.0 |  |
| multi-server-configuration | medium | public-solved | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| neural-architecture-search-final | medium | oracle | 1.00 | 0.0 | 13 | 0 | 0.0 |  |
| neutron-submission | medium | oracle | 1.00 | 0.0 | 15 | 0 | 0.0 |  |
| okhttp-trailers-crash | medium | broken | 0.00 | 0.0 | 20 | 0 | 0.0 |  |
| parking-lot-pathfinding | medium | public-solved | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| permutation-construction-100k | medium | public-solved | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| pgn-chess-repair-puzzles | medium | oracle | 1.00 | 0.0 | 60 | 0 | 0.0 |  |
| prediction-model-evaluation | medium | public-solved | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| python-api-rate-limit | medium | oracle | 1.00 | 0.0 | 15 | 0 | 0.0 |  |
| rsa-jwt-token-api-redis-blacklist | medium | oracle | 1.00 | 0.0 | 15 | 0 | 0.0 |  |
| sales-data-csv-analysis | medium | oracle | 1.00 | 0.0 | 60 | 0 | 0.0 |  |
| scan-linux-persistence-artifacts | medium | oracle | 1.00 | 0.0 | 15 | 0 | 0.0 |  |
| security-breach-incident-response | medium | oracle-partial | 0.75 | 0.0 | 60 | 0 | 0.0 |  |
| security-incident-log-analysis | medium | oracle-partial | 0.92 | 0.0 | 60 | 0 | 0.0 |  |
| sign-vector-game | medium | oracle | 1.00 | 1.5 | 15 | 0 | 0.0 |  |
| sql-injection-forensics | medium | broken | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| sympy-bug-fix | medium | unverified | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| task-xxe-exploit | medium | oracle | 1.00 | 0.0 | 5 | 0 | 0.0 |  |
| todos-api | medium | oracle | 1.00 | 0.0 | 8 | 0 | 0.0 |  |
| tsl-test-case-generation | medium | oracle-partial | 0.88 | 0.0 | 60 | 0 | 0.0 |  |
| anomaly-detection-ranking | hard | oracle | 1.00 | 0.0 | 60 | 0 | 0.0 |  |
| application-debug | hard | oracle | 1.00 | 0.0 | 10 | 0 | 0.0 |  |
| auth_token_race_condition | hard | oracle | 1.00 | 0.0 | 10 | 0 | 0.0 |  |
| bash-log-processor-fix | hard | oracle-partial | 0.61 | 0.0 | 60 | 0 | 0.0 |  |
| bloom-filter-cache-penetration-prevention | hard | broken | 0.00 | 0.0 | 15 | 0 | 0.0 |  |
| breast-cancer-mlflow | hard | broken | 0.00 | 2.1 | 15 | 0 | 0.0 |  |
| chained-forensic-extraction_20260101_011957 | hard | oracle | 1.00 | 0.0 | 30 | 0 | 0.0 |  |
| container-registry-optimization | hard | oracle | 1.00 | 0.0 | 5 | 0 | 0.0 |  |
| db-migration-local-storage | hard | oracle | 1.00 | 0.0 | 10 | 0 | 0.0 |  |
| etl_checkpoint_resume_bug | hard | broken | 0.00 | 0.0 | 10 | 0 | 0.0 |  |
| fix_async_worker_queue | hard | oracle | 1.00 | 0.0 | 10 | 0 | 0.0 |  |
| grid-pathfinding | hard | public-solved | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| mech-system | hard | oracle | 1.00 | 0.0 | 15 | 0 | 0.0 |  |
| multi-labeller | hard | oracle | 1.00 | 0.0 | 15 | 0 | 0.0 |  |
| network-log-normalization | hard | broken | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| pandas-numpy-data-analysis | hard | public-solved | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| pdf-table-parsing | hard | broken | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| publisher-market-analysis | hard | unverified | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| publisher-market-analysis-v2 | hard | public-solved | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| react-typescript-debugg | hard | broken | 0.00 | 0.0 | 15 | 0 | 0.0 |  |
| reproducibility-and-envsetup | hard | oracle | 1.00 | 0.2 | 6 | 0 | 0.0 |  |
| reverse-engineer-stack-vm | hard | oracle | 1.00 | 0.0 | 13 | 0 | 0.0 |  |
| sakila-sqlite-queries | hard | public-solved | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| server-log-analysis | hard | public-solved | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| service-deployment-wave-planner | hard | oracle | 1.00 | 0.0 | 30 | 0 | 0.0 |  |
| simple-database-query-tool | hard | public-solved | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| submission_a63937a5_20251224_152124 | hard | oracle | 1.00 | 0.3 | 30 | 0 | 0.0 |  |
| symlink-chain-traversal | hard | broken | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| token-auth-websocket | hard | oracle | 1.00 | 0.5 | 30 | 0 | 0.0 |  |
| word-derangement-mapping | hard | public-solved | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| basic-message-queue | expert | public-solved | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| book-portfolio-analysis | expert | unverified | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
| corrupted-filesystem-recovery | expert | broken | 0.00 | 0.0 | 60 | 0 | 0.0 |  |
