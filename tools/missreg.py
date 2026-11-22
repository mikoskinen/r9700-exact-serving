#!/usr/bin/env python3
"""missreg.py -- routed-expert kernel time against host-tier misses, from a RADIANCE_DEBUG_MISS_LOG file and the
engine log of the same run under --profile-ops.

    python3 tools/missreg.py MISS.log ENGINE.log [--op REGEX] [--every N] [--draft exclude|include]
                             [--keep-prefill] [--keep-first] [--split] [--per-layer] [-v]

The miss log (core/place/misslog.h, format radiance-miss-log/1) has one line per (step, rank, trunk or draft-head
layers): the live routed experts of each layer and the tier each was read from (vram, host = the pinned pool read
zero-copy, other = pageable / file / mapped). Its "# profile K step N" lines say which step --profile-ops printed its
K-th table set behind (one table per rank, counters cumulative; RADIANCE_PROFILE_EVERY, default 32); without them
dump K is taken at step K * --every and a last extra dump at the log's last step.

Per rank and dump interval (N[K-1], N[K]]: the ops matching --op (default the moe_exl3 family) give ms a step, the
miss log the mean live experts a step resident in VRAM (R) and missed, i.e. off the card (M = host + other). Then a
least-squares fit over the intervals:

    ms/step = a + b * M + c * R        (and without the intercept: b0 * M + c0 * R)

b is ms per missed expert, c ms per resident live expert. --split fits host and other apart (ms/step = a + bh * H +
bo * O + c * R). --per-layer also pools (interval, layer) points: each op instance is put on a routed layer from its
weight's name (layers.N / blk.N; mtp.N after the trunk) or else by op order, and the fit runs on that layer's ms a step
against its own M and R, many more points than intervals.

The draft head's layer: the profile times every draft pass, the miss log keeps only each step's last draft pass (its
bank is overwritten pass by pass), so by default (--draft exclude) the head's ops and lines are left out wherever op
instances can be put on layers. Intervals holding prefill or mixed steps are left out (their GEMMs are prefill-shaped)
unless --keep-prefill, and so is the span up to the first table, which also holds bring-up's passes, unless
--keep-first.
--profile-ops synchronises around every op, so these are not unprofiled times; the regression is about their slope."""
import argparse
import re
import sys
from collections import defaultdict

ROW = re.compile(r"\s(\d+)\s+([\d.]+)\s+([\d.]+)\s+(dev|host)\s+(\S+)\s+([\d.]+)\s+(\S+)\s+(.*?)\s+#(\d+)\s+(\S+)\s*$")
HEAD = re.compile(r"per-op timing, rank (\d+)")
TOTAL = re.compile(r"^\s*total\s+[\d.]+")
MARK = re.compile(r"^# profile (\d+) step (-?\d+)( final)?")
LAYER = re.compile(r"(?:^|[._/])(?:layers?|blk|blocks?|h)[._](\d+)(?=[._/]|$)")
MTP = re.compile(r"(?:^|[._/])mtp[._](?:layers?[._])?(\d+)(?=[._/]|$)")


def read_miss(path):
    """-> header dict, marks [(k, step, final)], lines [(step, kind, rank, st, v, h, o, {layer: (v, h, o)})]."""
    hdr, marks, lines = {}, [], []
    for ln in open(path, errors="replace"):
        if ln.startswith("#"):
            m = MARK.match(ln)
            if m:
                marks.append((int(m.group(1)), int(m.group(2)), bool(m.group(3))))
            elif ln.startswith("# model "):
                for key in ("tp", "routed_layers", "trunk_layers", "experts"):
                    m = re.search(r"\b%s (\d+)" % key, ln)
                    if m:
                        hdr[key] = m.group(1)
            continue
        f = ln.split()
        if len(f) != 16:
            continue
        step, kind, rank = int(f[0]), f[1], int(f[2])
        st = int(f[6])
        v, h, o = int(f[9]), int(f[10]), int(f[11])
        first, _, packed = f[15].partition(":")
        per = {}
        for i, e in enumerate(packed.split(",")):
            if e == "-":
                continue
            p = [int(x) for x in e.split(".")] + [0, 0]
            per[int(first) + i] = (p[0], p[1], p[2])
        lines.append((step, kind, rank, st, v, h, o, per))
    return hdr, marks, lines


def read_dumps(path):
    """-> {rank: [ {(op, idx, wname): (ms, calls)} cumulative, one per dump ]}"""
    out, cur, rk = defaultdict(list), None, None
    for ln in open(path, errors="replace"):
        m = HEAD.search(ln)
        if m:
            rk, cur = int(m.group(1)), {}
            continue
        if cur is None:
            continue
        m = ROW.search(ln)
        if m:
            cur[(m.group(7), int(m.group(9)), m.group(10))] = (float(m.group(2)), int(m.group(1)))
        elif TOTAL.match(ln):
            out[rk].append(cur)
            cur = None
    return out


def solve(rows, ys):
    """Least squares by the normal equations; -> (coefficients, r2) or (None, None) when singular."""
    n = len(rows[0])
    A = [[sum(r[i] * r[j] for r in rows) for j in range(n)] for i in range(n)]
    b = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(n)]
    for c in range(n):   # Gauss-Jordan with partial pivoting
        p = max(range(c, n), key=lambda i: abs(A[i][c]))
        if abs(A[p][c]) < 1e-12:
            return None, None
        A[c], A[p], b[c], b[p] = A[p], A[c], b[p], b[c]
        for i in range(n):
            if i != c:
                f = A[i][c] / A[c][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[c])]
                b[i] -= f * b[c]
    x = [b[i] / A[i][i] for i in range(n)]
    my = sum(ys) / len(ys)
    ss = sum((y - my) ** 2 for y in ys)
    res = sum((y - sum(c * v for c, v in zip(x, r))) ** 2 for r, y in zip(rows, ys))
    return x, (1.0 - res / ss) if ss > 0 else float("nan")


def fit_report(label, pts, split):
    """pts: [(ms, M, H, O, R)]. Prints the fits with and without the intercept."""
    if len(pts) < 2:
        print(f"  {label}: {len(pts)} point(s), nothing to fit")
        return
    ys = [p[0] for p in pts]
    if split:
        reg, names = [(p[2], p[3], p[4]) for p in pts], ["ms/host", "ms/other", "ms/resident"]
    else:
        reg, names = [(p[1], p[4]) for p in pts], ["ms/miss", "ms/resident"]
    need = len(names) + 1
    for icpt in (True, False):
        rows = [((1.0,) if icpt else ()) + r for r in reg]
        x, r2 = solve(rows, ys) if len(pts) >= need - (0 if icpt else 1) else (None, None)
        if x is None:
            print(f"  {label} {'a + ' if icpt else ''}fit: {len(pts)} points, singular or too few")
            continue
        a = f"a {x[0]:8.4f} ms  " if icpt else " " * 15
        co = x[1:] if icpt else x
        print(f"  {label} {'with' if icpt else 'no  '} intercept: {a}" +
              "  ".join(f"{n} {c:9.5f}" for n, c in zip(names, co)) + f"   R2 {r2:6.3f}  n {len(pts)}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("miss", nargs="?")
    ap.add_argument("log", nargs="?")
    ap.add_argument("--op", default=r"^moe_exl3", help="regex for the routed-expert ops (default ^moe_exl3)")
    ap.add_argument("--every", type=int, default=32, help="RADIANCE_PROFILE_EVERY when the log has no markers")
    ap.add_argument("--draft", choices=("exclude", "include"), default="exclude")
    ap.add_argument("--keep-prefill", action="store_true")
    ap.add_argument("--keep-first", action="store_true")
    ap.add_argument("--split", action="store_true", help="fit host and other misses apart")
    ap.add_argument("--per-layer", action="store_true", help="also fit (interval, layer) points")
    ap.add_argument("--selftest", action="store_true", help="fit synthetic files with known slopes and check")
    ap.add_argument("-v", action="store_true", help="print every interval")
    return ap.parse_args(argv)


def run(a):
    """Prints the report; -> ({rank: per-step points}, {rank: per-layer points})."""
    hdr, marks, lines = read_miss(a.miss)
    dumps = read_dumps(a.log)
    if not lines:
        sys.exit(f"{a.miss}: no data lines")
    if not dumps:
        sys.exit(f"{a.log}: no --profile-ops tables")
    step_pts, layer_pts = {}, {}
    trunk = int(hdr.get("trunk_layers", 1 << 30))
    op_re = re.compile(a.op)
    last = max(l[0] for l in lines)

    by_rank = defaultdict(list)
    for l in lines:
        by_rank[l[2]].append(l)

    print(f"miss log {a.miss}: {len(lines)} lines, steps {min(l[0] for l in lines)}..{last}, "
          f"ranks {sorted(by_rank)}, trunk layers {hdr.get('trunk_layers', '?')}; "
          f"{len(marks)} profile marker(s); engine log: " +
          ", ".join(f"rank {r} {len(d)} table(s)" for r, d in sorted(dumps.items())))
    pooled_step, pooled_layer = [], []
    for rk in sorted(dumps):
        ds = dumps[rk]
        if marks:
            if len(marks) != len(ds):
                print(f"  rank {rk}: {len(ds)} tables against {len(marks)} markers; pairing the first "
                      f"{min(len(ds), len(marks))}")
            steps = [m[1] for m in marks[:len(ds)]]
            ds = ds[:len(steps)]
        else:
            steps = [(k + 1) * a.every for k in range(len(ds))]
            if steps and steps[-1] > last:
                steps[-1] = last
        # the op instances on routed layers: by name, else by op order over the layers the log saw
        keys = sorted({k for d in ds for k in d if op_re.search(k[0])}, key=lambda k: k[1])
        layer_of = {}
        for k in keys:
            m = MTP.search(k[2])
            if m:
                layer_of[k] = trunk + int(m.group(1))
                continue
            m = LAYER.search(k[2])
            if m:
                layer_of[k] = int(m.group(1))
        if len(layer_of) < len(keys):
            seen = sorted({L for l in by_rank.get(rk, []) for L in l[7]})
            per_op = defaultdict(list)
            for k in keys:
                per_op[k[0]].append(k)
            layer_of = {}
            for op, ks in per_op.items():
                if len(ks) == len(seen):
                    layer_of.update({k: L for k, L in zip(sorted(ks, key=lambda k: k[1]), seen)})
            how = "op order" if layer_of else "none"
        else:
            how = "weight names"
        use = [k for k in keys if a.draft == "include" or k not in layer_of or layer_of[k] < trunk]
        if not layer_of and a.draft == "exclude":
            print(f"  rank {rk}: no op instance could be put on a layer, so the draft head's ops stay in while its "
                  f"lines are left out; --draft include keeps both")
        print(f"\n== rank {rk}: {len(keys)} op instance(s) match {a.op!r} "
              f"({len({k[0] for k in keys})} op name(s)); layers by {how}"
              + ("" if a.draft == "include" else f"; draft head left out ({len(keys) - len(use)} instance(s))"))
        if a.v:
            print(f"   {'K':>3s} {'steps':>9s} {'n':>4s} {'ms/step':>8s} {'live':>7s} {'vram':>7s} {'host':>7s} "
                  f"{'other':>6s} {'zc':>6s}  kinds")
        pts = []
        for K in range(len(ds)):
            s0, s1 = steps[K - 1] if K else -1, steps[K]
            prev = ds[K - 1] if K else {}
            sel = [l for l in by_rank.get(rk, []) if s0 < l[0] <= s1]
            kinds = defaultdict(int)
            for l in sel:
                kinds[l[1]] += 1
            tsteps = {l[0] for l in sel if l[1] != "draft"}
            n = len(tsteps)
            skip = None
            if n == 0:
                skip = "no lines"
            elif K == 0 and not a.keep_first:
                skip = "first table (holds bring-up's passes)"
            elif not a.keep_prefill and (kinds.get("prefill") or kinds.get("mixed")):
                skip = "prefill/mixed"
            dms = sum(ds[K].get(k, (0, 0))[0] - prev.get(k, (0, 0))[0] for k in use)
            agg = defaultdict(lambda: [0, 0, 0])
            zc = 0
            for l in sel:
                if l[1] == "draft" and a.draft == "exclude":
                    continue
                for L, (v, h, o) in l[7].items():
                    g = agg[L]
                    g[0] += v
                    g[1] += h
                    g[2] += o
                if l[3] == 0:
                    zc += l[5]
            V = sum(g[0] for g in agg.values())
            H = sum(g[1] for g in agg.values())
            O = sum(g[2] for g in agg.values())
            if a.v:
                kd = " ".join(f"{k}:{c}" for k, c in sorted(kinds.items()))
                print(f"   {K:3d} {s0:4d}-{s1:<4d} {n:4d} {dms / max(n, 1):8.3f} {(V + H + O) / max(n, 1):7.1f} "
                      f"{V / max(n, 1):7.1f} {H / max(n, 1):7.2f} {O / max(n, 1):6.2f} {zc / max(n, 1):6.2f}  {kd}"
                      + (f"  [skipped: {skip}]" if skip else ""))
            if skip:
                continue
            pts.append((dms / n, (H + O) / n, H / n, O / n, V / n))
            if a.per_layer and layer_of:
                for L, g in agg.items():
                    lk = [k for k in use if layer_of.get(k) == L]
                    if not lk:
                        continue
                    lms = sum(ds[K].get(k, (0, 0))[0] - prev.get(k, (0, 0))[0] for k in lk)
                    pooled_layer.append((rk, (lms / n, (g[1] + g[2]) / n, g[1] / n, g[2] / n, g[0] / n)))
        if pts:
            m = [sum(p[i] for p in pts) / len(pts) for i in range(5)]
            print(f"  {len(pts)} interval(s): mean {m[0]:.3f} ms a step, {m[1]:.2f} missed (host {m[2]:.2f}, other "
                  f"{m[3]:.2f}) and {m[4]:.1f} resident live experts a step")
        fit_report(f"rank {rk} per step", pts, a.split)
        pooled_step += pts
        step_pts[rk] = pts
        if a.per_layer:
            layer_pts[rk] = [p for r, p in pooled_layer if r == rk]
            fit_report(f"rank {rk} per layer", layer_pts[rk], a.split)
    if len(dumps) > 1:
        print("\n== all ranks pooled")
        fit_report("per step", pooled_step, a.split)
        if a.per_layer:
            fit_report("per layer", [p for _, p in pooled_layer], a.split)
    return step_pts, layer_pts


def selftest():
    """Two ranks, four trunk layers and a head, a dump every 8 steps: each layer's routed-expert op takes
    0.010 + 0.002 * resident + 0.050 * host + 0.120 * other ms a step. The fits must give those slopes back."""
    import os
    import random
    import tempfile
    rng = random.Random(3)
    d = tempfile.mkdtemp(prefix="missreg-")
    miss, log = os.path.join(d, "miss.log"), os.path.join(d, "engine.log")
    T, every, n_dump = 4, 8, 12
    cost = lambda v, h, o: 0.010 + 0.002 * v + 0.050 * h + 0.120 * o
    cum = {r: defaultdict(float) for r in (0, 1)}
    calls = {r: defaultdict(int) for r in (0, 1)}
    ml = ["# radiance-miss-log/1", f"# model toy arch t tp 2 routed_layers {T + 1} trunk_layers {T} experts 8"]
    el = []
    for step in range(n_dump * every + 1):
        for r in (0, 1):
            phase = (step // every) % 3          # intervals differ in how much misses
            cells = []
            for L in range(T + 1):
                h = rng.randint(0, 1 + 2 * phase)
                o = rng.randint(0, phase)
                v = rng.randint(4, 14)
                cells.append((v, h, o))
                for op in ("moe_exl3_gate_up", "moe_exl3_down"):
                    name = f"layers.{L}.ffn.experts.0.w" if L < T else "mtp.0.ffn.experts.0.w"
                    cum[r][(op, 100 + 10 * L + (op == "moe_exl3_down"), name)] += cost(v, h, o) / 2
                    calls[r][(op, 100 + 10 * L + (op == "moe_exl3_down"), name)] += 1
            tr, hd = cells[:T], cells[T]
            pk = lambda cs: ",".join(f"{v}.{h}.{o}" for v, h, o in cs)
            sv = lambda cs, i: sum(c[i] for c in cs)
            ml.append(f"{step} decode {r} 4 4 1 0 0 {sv(tr, 0) + sv(tr, 1) + sv(tr, 2)} {sv(tr, 0)} {sv(tr, 1)} "
                      f"{sv(tr, 2)} {sv(tr, 1)} 0 0 0:{pk(tr)}")
            ml.append(f"{step} draft {r} 4 4 1 0 0 {sum(hd)} {hd[0]} {hd[1]} {hd[2]} {hd[1]} 0 0 {T}:{pk([hd])}")
        if step > 0 and step % every == 0:
            ml.append(f"# profile {step // every} step {step}")
            for r in (0, 1):
                el.append(f"\nper-op timing, rank {r}. Not comparable with an unprofiled run:\n")
                el.append("  %10s %12s %10s %5s %10s %10s  %s" % ("calls", "total ms", "us/call", "site", "join us",
                                                                   "wait us", "op / geometry"))
                for (op, idx, name), ms in sorted(cum[r].items(), key=lambda kv: -kv[1]):
                    c = calls[r][(op, idx, name)]
                    el.append("  %10d %12.3f %10.2f %5s %s %10.2f  %s  %s  #%d %s" % (
                        c, ms, ms * 1000.0 / c, "dev", "         -", 1.0, op, "n=4096 k=2048 E=96", idx, name))
                el.append("  %10s %12.3f   (waited %.3f ms besides)\n" % ("total", sum(cum[r].values()), 0.0))
    open(miss, "w").write("\n".join(ml) + "\n")
    open(log, "w").write("\n".join(el) + "\n")
    a = parse_args([miss, log, "--split", "--per-layer"])
    step_pts, layer_pts = run(a)
    ok = True
    for r in (0, 1):
        x, _ = solve([(1.0, p[2], p[3], p[4]) for p in step_pts[r]], [p[0] for p in step_pts[r]])
        want = (0.010 * T, 0.050, 0.120, 0.002)
        good = x is not None and all(abs(g - w) < 2e-3 + 0.02 * abs(w) for g, w in zip(x, want))
        xl, _ = solve([(1.0, p[2], p[3], p[4]) for p in layer_pts[r]], [p[0] for p in layer_pts[r]])
        wl = (0.010, 0.050, 0.120, 0.002)
        good_l = xl is not None and all(abs(g - w) < 2e-3 + 0.02 * abs(w) for g, w in zip(xl, wl))
        print(f"selftest rank {r}: per step {x} want {want} {'ok' if good else 'BAD'}; per layer {xl} want {wl} "
              f"{'ok' if good_l else 'BAD'}")
        ok = ok and good and good_l
    print("selftest", "OK" if ok else f"FAILED (files kept in {d})")
    if ok:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    return 0 if ok else 1


def main():
    a = parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.miss or not a.log:
        sys.exit("usage: missreg.py MISS.log ENGINE.log [options]; --help")
    run(a)


if __name__ == "__main__":
    main()
