#!/usr/bin/env python3
"""keepawake.py [--interval-ms N] [--node NODE] [--cards UUID,UUID,...|all] [--log FILE] [--report-s S] [--seconds S]

Read-only keep-awake poller for the R9700s (9.11.2026, queue-ar). gpu02's TP4 runs fall into a sporadic "slow state" (the
routed-expert kernels ~2-3x slower at >= 16 rows, -15..-20 % decode) unless the host keeps talking to the cards' SMU;
item 101 showed reading each card's gpu_metrics every 5 ms keeps every start fast (B 2/2, D 1/1 fast; A / C / E / P / H
0/8). This is that read and nothing else: open the node once per card, seek + read in a loop, sleep the interval.

  --interval-ms N  the poll interval per card (default 5; 103 sizes the cheapest one that stays fast)
  --node NODE      gpu_metrics (default, the measured cure) | power1_average | temp2_input | gpu_busy_percent |
                   mem_busy_percent | pp_dpm_mclk | pp_dpm_fclk  (the others are candidates for a cheaper trigger, 103b)
                   | unique_id (the amdgpu sysfs node: the driver, no SMU) | current_link_speed (a PCIe config-space
                   read: link traffic, no SMU) | proc_loadavg (/proc/loadavg once a round: no GPU
                   access at all, the CPU wake alone; 103c)
  --spin           no reads: one thread busy-spins a whole core (the host kept out of idle states with no driver or
                   GPU access; 103c); --node / --interval-ms are ignored
  --cards          ROCR UUIDs (comma-separated) or "all" (every amdgpu card with a unique_id; default)
  --log FILE       one line every --report-s (default 60) seconds: reads, reads/s, mean / p99 read latency (us), the
                   process's own CPU seconds and % of one core since the last line; default stderr
  --seconds S      stop after S seconds (default: run until SIGTERM / SIGINT)

Writes nothing to sysfs and opens nothing writable. Standard library only. Exit 0 on a signal, 1 if no card is found."""
import argparse, glob, os, resource, signal, sys, time

NODES = {
    "gpu_metrics": "gpu_metrics",
    "unique_id": "unique_id",
    "current_link_speed": "current_link_speed",
    "pp_dpm_mclk": "pp_dpm_mclk",
    "pp_dpm_fclk": "pp_dpm_fclk",
    "gpu_busy_percent": "gpu_busy_percent",
    "mem_busy_percent": "mem_busy_percent",
}
HWMON = {"power1_average", "temp2_input"}


SYS = os.environ.get("KEEPAWAKE_SYSFS", "/sys")   # a test tree's root; the real one otherwise


def card_dirs(spec):
    out = []
    for d in sorted(glob.glob(os.path.join(SYS, "class/drm/card*/device"))):
        try:
            uid = open(os.path.join(d, "unique_id")).read().strip().lower()
        except OSError:
            continue
        if spec == "all" or uid in spec:
            out.append((uid, d))
    return out


def node_path(d, node):
    if node in HWMON:
        h = sorted(glob.glob(os.path.join(d, "hwmon", "hwmon*")))
        return os.path.join(h[0], node) if h else None
    return os.path.join(d, NODES[node])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval-ms", type=float, default=5.0)
    ap.add_argument("--node", default="gpu_metrics", choices=sorted(set(NODES) | HWMON | {"proc_loadavg"}))
    ap.add_argument("--spin", action="store_true")
    ap.add_argument("--cards", default="all")
    ap.add_argument("--log", default="")
    ap.add_argument("--report-s", type=float, default=60.0)
    ap.add_argument("--seconds", type=float, default=0.0)
    a = ap.parse_args()
    spec = "all" if a.cards == "all" else {u.lower().replace("gpu-", "") for u in a.cards.split(",")}
    log = open(a.log, "a", buffering=1) if a.log else sys.stderr
    fhs = []
    if a.spin:
        pass
    elif a.node == "proc_loadavg":
        fhs.append(("host", os.open("/proc/loadavg", os.O_RDONLY)))
    else:
        for uid, d in card_dirs(spec):
            p = node_path(d, a.node)
            if not p:
                continue
            try:
                fhs.append((uid, os.open(p, os.O_RDONLY)))
            except OSError as e:
                print(f"keepawake: {p}: {e}", file=log)
    if not fhs and not a.spin:
        print("keepawake: no card found", file=log)
        return 1
    stop = [False]
    def on_sig(*_):
        stop[0] = True
    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)
    ivl = a.interval_ms / 1000.0
    t_end = time.monotonic() + a.seconds if a.seconds > 0 else None
    print(f"keepawake: start {'spin (no reads)' if a.spin else 'node=' + a.node} interval={0 if a.spin else a.interval_ms} ms "
          f"targets={','.join(u for u, _ in fhs) or '-'} pid={os.getpid()}", file=log)
    n = 0; lat = []; t_rep = time.monotonic(); ru0 = resource.getrusage(resource.RUSAGE_SELF)
    while not stop[0]:
        if a.spin:   # a busy core and nothing else: the clock is read only every ~65k iterations
            for _ in range(65536):
                pass
            n += 1
            now = time.monotonic()
            if now - t_rep >= a.report_s:
                ru = resource.getrusage(resource.RUSAGE_SELF)
                cpu = (ru.ru_utime - ru0.ru_utime) + (ru.ru_stime - ru0.ru_stime)
                print(f"keepawake: {time.strftime('%F %T')} spin rounds {n} ({n / (now - t_rep):.0f}/s), CPU {cpu:.2f} s "
                      f"({100 * cpu / (now - t_rep):.2f} % of a core)", file=log)
                n = 0; t_rep = now; ru0 = ru
            if t_end and now >= t_end:
                break
            continue
        for _, fd in fhs:
            t0 = time.perf_counter_ns()
            try:
                os.pread(fd, 4096, 0)
            except OSError:
                pass
            lat.append((time.perf_counter_ns() - t0) / 1e3)
            n += 1
        now = time.monotonic()
        if now - t_rep >= a.report_s:
            ru = resource.getrusage(resource.RUSAGE_SELF)
            cpu = (ru.ru_utime - ru0.ru_utime) + (ru.ru_stime - ru0.ru_stime)
            lat.sort()
            print(f"keepawake: {time.strftime('%F %T')} reads {n} ({n / (now - t_rep):.0f}/s), read latency mean "
                  f"{sum(lat) / len(lat):.1f} p99 {lat[int(0.99 * (len(lat) - 1))]:.1f} us, CPU {cpu:.2f} s "
                  f"({100 * cpu / (now - t_rep):.2f} % of a core: user {ru.ru_utime - ru0.ru_utime:.2f} s, sys "
                  f"{ru.ru_stime - ru0.ru_stime:.2f} s)", file=log)
            n = 0; lat = []; t_rep = now; ru0 = ru
        if t_end and now >= t_end:
            break
        time.sleep(ivl)
    for _, fd in fhs:
        os.close(fd)
    print(f"keepawake: stop {time.strftime('%F %T')}", file=log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
