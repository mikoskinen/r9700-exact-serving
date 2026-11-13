# tools/ngramshare -- reuse the DS Engram n-gram tables across sequential CORRECTNESS-ONLY starts

12.11.2026, ds-main-port ngshare builder. No GPU use in any of these files; no docker run / stop.

A DS-V4.1 start with `--ngram-placement ram` copies the Engram tables into pinned memory every time:
91.56 GiB in ~74 s + 11.44 GiB in ~9 s (+ a 16 B scale), ~83 s of a ~245 s start.
With `--ngram-placement ram=/ngram` the pinned memory is a shared tmpfs file per table
(`/ngram/ngram-<id>.tab`, Radiance core/format/table_share.*, unchanged since the 1.15.0 merge acc9ae5):
the first start fills it ("filled"), a later start maps it after a probe check ("reused (probes match)").
Without help a graceful stop (SIGTERM: `docker stop`, `compose down`) deletes the file, because the last
lock holder deletes it (table_share.cpp:218-249). The keeper is a second holder, so the engine is never last.
(`docker rm -f` = SIGKILL skips ~Engine, so that path leaves the file even without the keeper.)

**Correctness-only.** gpu02's tmpfs has 4 KiB pages (`shmem_enabled = never`); the copy path gets THP.
BI traces, identity / greedy-set checks, smoke: yes. Speed / mb / prefill / gate numbers: no, keep
`ds41-stack2-tp4.yaml` + `NGRAM=ram`.

## Files

| file | what |
|---|---|
| `ds41-ngshare-tp4.yaml` | = `queue-ar/dsmain/ds41-stack2-tp4.yaml` (md5 be8fb6d6) + one volume `/dev/shm/ds41ng:/ngram` |
| `ngram-keeper.sh prep / start [--wait S] [--expect N] / stop / status` | holds LOCK_SH on each READY `/dev/shm/ds41ng/ngram-*.tab` (one `flock` process per file, pid files in `runs/ngshare/`, log `runs/ngshare/keeper.log`) |
| `ngram-shm-cleanup.sh` (`DRYRUN=1` to look) | stops the keeper, removes the tables + `.lock` files; refuses while any radiance container runs or any other process holds a file; logs to `runs/queue-changes.log` |
| `ngshare.sh` | one-line opt-in for a body (below) |
| `ngshare_hdr.py` | reads a table file's header like Radiance's classify() (lock-free, read-only) |
| `test_ngshare.py` | the no-GPU harness: imitates the lock protocol on dummy files, drives the real keeper + cleanup |

## How an item opts in

Prefer the helper. In a NEW or copied body (do not edit existing bodies), after the lib is sourced and after
the `Y=$HD/ds41-stack2-tp4.yaml` line:

```bash
. $HOME/gpu/ds41/exl3-test/queue-ar/tools/ngramshare/ngshare.sh || exit 1
```

That switches `Y` to `ds41-ngshare-tp4.yaml` and wraps lib-pr266's `restart()` (so `gate` / `gate1` too):
`NGRAM=ram=/ngram` (after whatever `best_vars` set), `ngram-keeper.sh prep` before the start, and after a
healthy start `ngram-keeper.sh start --wait 120 --expect 2` plus a `RESULT ngshare <tag>:` line with the
engine's placement lines (filled / reused and seconds). `DRYRUN=1` prints the extra steps.

By hand, the same thing is:

```bash
NGS=$HOME/gpu/ds41/exl3-test/queue-ar/tools/ngramshare
Y=$NGS/ds41-ngshare-tp4.yaml                 # instead of ds41-stack2-tp4.yaml
$NGS/ngram-keeper.sh prep                    # BEFORE the first start: docker would make /dev/shm/ds41ng root's,
                                             # and the engine (uid 1000) then cannot create its file
# ... the body's best_vars / stackenv ...
export NGRAM=ram=/ngram                      # AFTER best_vars (it sets NGRAM=ram)
restart TAG                                  # 1st start: "filled"; later starts: "reused (probes match)"
$NGS/ngram-keeper.sh start --wait 120 --expect 2   # after each healthy start (idempotent)
```

## Memory and the end of a block

While the files exist they take ~103 GiB of RAM in `/dev/shm` (not in MemAvailable) whether the keeper
runs or not. A speed start on the copy path then has ~103 GiB less room, and production DP2 (A + B
~381 of 385 GiB) does not fit beside it. So when the correctness block ends:

```bash
~/gpu/ds41/exl3-test/queue-ar/tools/ngramshare/ngram-shm-cleanup.sh   # with no radiance container running
```

`ngram-keeper.sh stop` alone only releases the holds: the files stay, and the next engine that stops
gracefully as the last holder deletes them.

A changed .rad (path, size or mtime) gets a new `<id>`, so a new file; the keeper keeps holding the old one
until cleanup (see `ngram-keeper.sh status`).

## Test

```bash
python3 -I ~/gpu/ds41/exl3-test/queue-ar/tools/ngramshare/test_ngshare.py   # ~15 s, temp dir under /dev/shm, removed
```
