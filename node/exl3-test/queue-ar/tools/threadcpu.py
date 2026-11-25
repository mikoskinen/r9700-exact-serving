#!/usr/bin/env python3
"""threadcpu.py PID SECONDS [INTERVAL_S=0.2] (9.11.2026, queue-ar, item 112) -- where a process's named threads run: every
INTERVAL it reads /proc/PID/task/*/comm and the last-run CPU (field 39 of /proc/PID/task/TID/stat) -- host /proc only,
nothing of the cards -- for SECONDS, then prints per thread name (rad-rank<i>, rad-step; the rest summed as "other") the
CPU histogram, the share per CCD (Zen 2 EPYC 7302P: CPU c is on CCD (c % 16) // 4) and how often it changed CPU between
samples. One line a name: "THREADCPU <name> tid <tid> samples N | ccd 0..3 shares | top cpus | moves M"."""
import os
import sys
import time
from collections import Counter, defaultdict


def tasks(pid):
    out = {}
    for t in os.listdir(f"/proc/{pid}/task"):
        try:
            name = open(f"/proc/{pid}/task/{t}/comm").read().strip()
            st = open(f"/proc/{pid}/task/{t}/stat").read()
            f = st[st.rindex(")") + 2:].split()
            out[int(t)] = (name, int(f[36]))   # field 39 = index 38 overall = 36 after pid and comm
        except (OSError, ValueError, IndexError):
            pass
    return out


def resolve(pid, want="radiance", depth=0):
    """The container's main pid is its init (compose init: true); walk the children to the engine's process."""
    try:
        if open(f"/proc/{pid}/comm").read().strip() == want:
            return pid
        kids = open(f"/proc/{pid}/task/{pid}/children").read().split()
    except OSError:
        return None
    for k in kids:
        r = resolve(int(k), want, depth + 1) if depth < 4 else None
        if r:
            return r
    return None


def main():
    pid0, secs = int(sys.argv[1]), float(sys.argv[2])
    pid = resolve(pid0)
    if pid is None:
        print(f"THREADCPU none: no 'radiance' process under pid {pid0}"); return
    print(f"THREADCPU process radiance pid {pid} (under {pid0})")
    ivl = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    hist = defaultdict(Counter)
    names = {}
    last = {}
    moves = Counter()
    n = 0
    t_end = time.monotonic() + secs
    while time.monotonic() < t_end:
        try:
            cur = tasks(pid)
        except OSError:
            break
        for tid, (name, cpu) in cur.items():
            key = name if name.startswith("rad-") else "other"
            names.setdefault(key, set()).add(tid)
            hist[key][cpu] += 1
            if tid in last and last[tid] != cpu and key != "other":
                moves[key] += 1
            last[tid] = cpu
        n += 1
        time.sleep(ivl)
    for key in sorted(hist, key=lambda k: (k == "other", k)):
        h = hist[key]
        tot = sum(h.values())
        ccd = Counter()
        for c, k in h.items():
            ccd[(c % 16) // 4] += k
        top = " ".join(f"{c}:{100 * k / tot:.0f}%" for c, k in h.most_common(4))
        print(f"THREADCPU {key} tids {','.join(str(t) for t in sorted(names[key]))[:60]} samples {n} | ccd " +
              " ".join(f"{c}:{100 * ccd[c] / tot:.0f}%" for c in range(4)) + f" | top cpus {top} | moves {moves[key]}")


if __name__ == "__main__":
    main()
