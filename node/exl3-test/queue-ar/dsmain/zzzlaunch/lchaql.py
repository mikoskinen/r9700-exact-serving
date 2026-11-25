#!/usr/bin/env python3
"""lchaql.py (13.11.2026, queue-ar/dsmain/zzzlaunch, a builder for ds-main-port; item 57-p4zyzzzzz-ds-r92zzzzzz-ref12zzzzzz-zzzlaunch's
AQL reader) -- the INTRA-PASS launch-gap table of the c8 decode pass on card 0 under ONE-IN-N stamping (RADIANCE_AQL_STAMP=N, N > 1),
per arm (lever RCO, fn4 route_copy_ov: B = 0, Z = 1). Standard library only. Read only (writes only --json).

Why not zzzaql/aqlcp.py's step read: aqlcp prices a decode STEP = one stamped trunk pass's start to the NEXT one's, which needs every
pass stamped (N = 1, +4 % a step: the speed blocks of this item could not be read then). Under N > 1 the stamped passes are N steps
apart, so this reader stays INSIDE each stamped pass: [its first dispatch's start, its last dispatch's end]. Lever RCO's sites are all
inside the pass (a routing copy and the dispatch after it); the step boundary ("between": the sampler chain, the step end, the
advance, then the next pass) is not priced here (zzzaql's N = 1 read has it: 185 us a step, int14).

  lchaql.py read STAMPS.csv[.gz] --win NAME:ARM:M0:M1 [...] [--json OUT] [--top 40]
      RADIANCE_AQL_STAMP's CSV (pass,seq,queue,kernel,ordered,chained,start_ns,end_ns,play,records; card 0, CLOCK_MONOTONIC ns);
      windows on the monotonic clock (zzzaql/aqlcp.py monowin REQS CLOCK gives M0:M1 for a harness block). Per window: the TRUNK kind
      (the most common `records` among passes of >= 300 dispatches starting in the window), its complete passes (>= 0.95 x the
      kind's median dispatch count); per pass: span, busy (union of run times), idle = span - busy, every idle interval typed by the
      dispatch b that ends it and p = the one issued before b in the tape (seq) -- dep (same play, same queue, seq + 1), rec (a record
      between / another queue), play (b starts another play of the pass: host work between the played stretches) -- and priced to the
      site (type, prev kernel -> next kernel). Pooled per ARM: us a pass by site, the sites' counts a pass and median gap, the
      ordered share of every next kernel at a site (engagement: under route_copy_ov 1 the dispatch after a routing copy plays
      unordered), dispatches, unordered share. Lines "LCHAQL ...".
  lchaql.py compare --json OUT.json [--base B] [--arm Z] [--top 16]
      the arms side by side: span / busy / idle a pass, the gap classes, and the site table (the union of each arm's top sites +
      lever RCO's four sites) with the difference.
  lchaql.py selftest
"""
import collections, csv, gzip, io, json, re, statistics, sys, tempfile, os

RCO_SITES = (("play", "scatter_small_kernel", "aql_copy16"), ("play", "aql_copy16", "exl3r4d_prep_row_kernel"),
             ("play", "r4d_ar_gather_twoshot_kernel", "aql_copy16"), ("dep", "aql_copy16", "post_collapse_norm_kernel"))


def short(k):
    """the kernel's base name from its mangled symbol (zzzaql/aqlcp.py's rule)"""
    m = re.match(r"_ZN?(?:12_GLOBAL__N_1)?(?:\d+r4d_moesort12_GLOBAL__N_1)?(\d+)", k)
    if m:
        n = int(m.group(1))
        s = k[m.end():m.end() + n]
        if s:
            return s
    return k[:40]


def opener(path):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), newline="")
    return open(path, newline="")


def wins_of(a):
    out = []
    for i, x in enumerate(a):
        if x == "--win":
            n, arm, lo, hi = a[i + 1].split(":")
            out.append((n, arm, int(float(lo)), int(float(hi))))
    return out


def load(path, W):
    """passes starting in a window: name -> {pass: [rows]}; row = (s, e, seq, queue, kernel, ordered, play, records)"""
    per = {w[0]: collections.defaultdict(list) for w in W}
    lo_all = min(w[2] for w in W)
    hi_all = max(w[3] for w in W) + 2_000_000_000
    names, kid = [], {}
    total = untimed = 0
    with opener(path) as f:
        r = csv.reader(f)
        hdr = next(r, None)
        if not hdr or hdr[:2] != ["pass", "seq"]:
            raise SystemExit("lchaql: %s is not a RADIANCE_AQL_STAMP CSV (header %s)" % (path, hdr))
        for row in r:
            total += 1
            if len(row) < 10:
                continue
            if not row[6] or not row[7]:
                untimed += 1
                continue
            s = int(row[6])
            if s < lo_all or s > hi_all:
                continue
            k = row[3]
            i = kid.get(k)
            if i is None:
                i = kid[k] = len(names)
                names.append(short(k))
            t = (s, int(row[7]), int(row[1]), int(row[2]), i, row[4] == "1", int(row[8]), int(row[9]))
            p = int(row[0])
            for wn, _, lo, hi in W:
                if lo <= s <= hi + 2_000_000_000:
                    per[wn][p].append(t)
    return per, names, {"rows": total, "untimed": untimed}


def price_pass(rows, names):
    """one pass: (span, busy, gaps [(type, ns, prev, next, next_ordered)])"""
    bys = sorted(rows, key=lambda x: x[2])
    prev = {}
    for k, x in enumerate(bys):
        prev[x[2]] = bys[k - 1] if k else None
    D = sorted(rows)
    S, E = D[0][0], max(x[1] for x in D)
    run_end = S
    gaps = []
    busy = 0
    cur0 = S
    for x in D:
        if x[0] > run_end:
            busy += run_end - cur0
            cur0 = x[0]
            p = prev[x[2]]
            if p is None:
                t = "first"
            elif x[6] != p[6]:
                t = "play"
            elif x[3] != p[3] or x[2] - p[2] >= 2:
                t = "rec"
            else:
                t = "dep"
            gaps.append((t, x[0] - run_end, names[p[4]] if p else "-", names[x[4]], x[5]))
        run_end = max(run_end, x[1])
    busy += run_end - cur0
    return E - S, busy, gaps


def read_cmd(a):
    path = a[1]
    W = wins_of(a)
    if not W:
        raise SystemExit("lchaql read: no --win NAME:ARM:M0:M1")
    jout = a[a.index("--json") + 1] if "--json" in a else None
    top = int(a[a.index("--top") + 1]) if "--top" in a else 40
    per, names, cnt = load(path, W)
    arms = collections.OrderedDict()
    out = {"counts": cnt, "windows": {}, "arms": {}}
    for wn, arm, lo, hi in W:
        P = {p: rows for p, rows in per[wn].items() if lo <= min(x[0] for x in rows) <= hi}
        big = collections.Counter(rows[0][7] for rows in P.values() if len(rows) >= 300)
        if not big:
            out["windows"][wn] = {"arm": arm, "error": "no stamped pass of >= 300 dispatches in the window"}
            print("LCHAQL window %s (%s): VIRHE no stamped trunk pass in the window" % (wn, arm))
            continue
        TR = big.most_common(1)[0][0]
        T = [rows for rows in P.values() if rows[0][7] == TR]
        med = statistics.median(len(r) for r in T)
        full = [r for r in T if len(r) >= 0.95 * med]
        out["windows"][wn] = {"arm": arm, "records": TR, "passes": len(full), "partial": len(T) - len(full),
                              "dispatches": med, "other_kinds": {str(k): v for k, v in big.items() if k != TR}}
        a_ = arms.setdefault(arm, {"passes": 0, "span": [], "busy": [], "idle": [], "cls": collections.Counter(), "ac_n": collections.Counter(),
                                   "ac_ord": collections.Counter(), "ac_s2s": collections.defaultdict(list),
                                   "site_ns": collections.Counter(), "site_n": collections.Counter(), "site_g": collections.defaultdict(list),
                                   "site_ord": collections.Counter(), "disp": 0, "ordered": 0, "records": set(), "windows": []})
        a_["records"].add(TR)
        a_["windows"].append(wn)
        for rows in full:
            span, busy, gaps = price_pass(rows, names)
            a_["passes"] += 1
            a_["span"].append(span)
            a_["busy"].append(busy)
            a_["idle"].append(span - busy)
            a_["disp"] += len(rows)
            bys = {x[2]: x for x in rows}
            for x in rows:   # the dispatch right after a routing copy (aql_copy16) in the tape: ordered? its start after the copy's start
                p = bys.get(x[2] - 1)
                if p is not None and names[p[4]] == "aql_copy16":
                    k = names[x[4]]
                    a_["ac_n"][k] += 1
                    a_["ac_ord"][k] += int(x[5])
                    if len(a_["ac_s2s"][k]) < 20000:
                        a_["ac_s2s"][k].append(x[0] - p[0])
            a_["ordered"] += sum(1 for x in rows if x[5])
            for t, g, pk, nk, nord in gaps:
                a_["cls"][t] += g
                k = (t, pk, nk)
                a_["site_ns"][k] += g
                a_["site_n"][k] += 1
                a_["site_ord"][k] += int(nord)
                if len(a_["site_g"][k]) < 20000:
                    a_["site_g"][k].append(g)
        print("LCHAQL window %s (%s): trunk kind %s records, %d complete stamped passes (%d partial), %s dispatches a pass%s" % (
            wn, arm, TR, len(full), len(T) - len(full), med,
            (" | other big kinds %s" % dict(big)) if len(big) > 1 else ""))
    for arm, a_ in arms.items():
        n = max(1, a_["passes"])
        if not a_["passes"]:
            out["arms"][arm] = {"passes": 0}
            continue
        sites = []
        keys = sorted(a_["site_ns"], key=lambda k: -a_["site_ns"][k])
        want = set(keys[:top]) | set(k for k in RCO_SITES if k in a_["site_ns"])
        for k in keys:
            if k not in want:
                continue
            sites.append({"type": k[0], "prev": k[1], "next": k[2], "us_pass": round(a_["site_ns"][k] / 1e3 / n, 2),
                          "n_pass": round(a_["site_n"][k] / n, 2), "median_us": round(statistics.median(a_["site_g"][k]) / 1e3, 2),
                          "next_ordered": round(a_["site_ord"][k] / a_["site_n"][k], 3)})
        rco = {}
        for k in RCO_SITES:
            rco["%s %s -> %s" % k] = {"us_pass": round(a_["site_ns"].get(k, 0) / 1e3 / n, 2), "n_pass": round(a_["site_n"].get(k, 0) / n, 2),
                                      "median_us": round(statistics.median(a_["site_g"][k]) / 1e3, 2) if a_["site_g"].get(k) else None,
                                      "next_ordered": round(a_["site_ord"][k] / a_["site_n"][k], 3) if a_["site_n"].get(k) else None}
        ac = {k: {"n_pass": round(a_["ac_n"][k] / n, 2), "ordered": round(a_["ac_ord"][k] / a_["ac_n"][k], 3),
                  "s2s_median_us": round(statistics.median(a_["ac_s2s"][k]) / 1e3, 2)} for k in a_["ac_n"]}
        m = lambda v: round(statistics.mean(v) / 1e6, 4)
        md = lambda v: round(statistics.median(v) / 1e6, 4)
        out["arms"][arm] = {"passes": a_["passes"], "windows": a_["windows"], "records": sorted(a_["records"]),
                            "span_ms": m(a_["span"]), "span_p50_ms": md(a_["span"]), "busy_ms": m(a_["busy"]), "idle_ms": m(a_["idle"]),
                            "idle_p50_ms": md(a_["idle"]), "cls_us_pass": {k: round(v / 1e3 / n, 1) for k, v in a_["cls"].items()},
                            "dispatches_pass": round(a_["disp"] / n, 1), "ordered_share": round(a_["ordered"] / max(1, a_["disp"]), 4),
                            "sites": sites, "rco_sites": rco, "after_copy": ac}
        A = out["arms"][arm]
        print("LCHAQL arm %s: %d stamped trunk passes (windows %s, kind %s) | inside the pass: span %.3f ms (p50 %.3f), busy %.3f, idle %.3f "
              "(p50 %.3f) | idle by class us a pass %s | %.1f dispatches a pass, ordered %.1f %%" % (
                  arm, a_["passes"], ",".join(a_["windows"]), "/".join(str(x) for x in sorted(a_["records"])), A["span_ms"], A["span_p50_ms"],
                  A["busy_ms"], A["idle_ms"], A["idle_p50_ms"], json.dumps(A["cls_us_pass"]), A["dispatches_pass"], 100 * A["ordered_share"]))
        print("LCHAQL arm %s after a routing copy (aql_copy16; the next dispatch in the tape: count a pass, ordered share, its start after the copy's "
              "start, median us): %s" % (arm, " | ".join("%s %s x, ordered %s, +%s us" % (k, v["n_pass"], v["ordered"], v["s2s_median_us"])
                                                         for k, v in sorted(ac.items(), key=lambda kv: -kv[1]["n_pass"])) or "none"))
        print("LCHAQL arm %s lever RCO's sites (us a pass, count a pass, median gap us, the next kernel's ordered share): %s" % (
            arm, " | ".join("%s %s (%s x, median %s, ordered %s)" % (k, v["us_pass"], v["n_pass"], v["median_us"], v["next_ordered"])
                            for k, v in rco.items())))
    if jout:
        json.dump(out, open(jout, "w"), indent=1)
    return 0


def compare_cmd(a):
    j = json.load(open(a[a.index("--json") + 1]))
    base = a[a.index("--base") + 1] if "--base" in a else "B"
    arm = a[a.index("--arm") + 1] if "--arm" in a else "Z"
    top = int(a[a.index("--top") + 1]) if "--top" in a else 16
    A, Z = j["arms"].get(base) or {}, j["arms"].get(arm) or {}
    if not A.get("passes") or not Z.get("passes"):
        print("LCHAQL COMPARE %s vs %s: VIRHE an arm has no stamped trunk pass (%s %s / %s %s)" % (arm, base, base, A.get("passes"), arm, Z.get("passes")))
        return 1
    if A.get("records") != Z.get("records"):
        print("LCHAQL COMPARE NOTE: the arms' trunk kinds differ (%s vs %s): another pass shape -- the table is INFO" % (A.get("records"), Z.get("records")))
    d = lambda x, y: "%+.3f ms (%+.2f %%)" % (y - x, 100 * (y - x) / x) if x else "na"
    print("LCHAQL COMPARE %s vs %s (inside the stamped c8 trunk pass, card 0; %d / %d passes): span %.3f -> %.3f ms = %s | busy %.3f -> %.3f | idle %.3f -> "
          "%.3f ms = %s | idle us a pass by class %s -> %s | dispatches a pass %s / %s, ordered %.1f %% -> %.1f %%" % (
              arm, base, A["passes"], Z["passes"], A["span_ms"], Z["span_ms"], d(A["span_ms"], Z["span_ms"]), A["busy_ms"], Z["busy_ms"],
              A["idle_ms"], Z["idle_ms"], d(A["idle_ms"], Z["idle_ms"]), json.dumps(A["cls_us_pass"]), json.dumps(Z["cls_us_pass"]),
              A["dispatches_pass"], Z["dispatches_pass"], 100 * A["ordered_share"], 100 * Z["ordered_share"]))
    sa = {(s["type"], s["prev"], s["next"]): s for s in A["sites"]}
    sz = {(s["type"], s["prev"], s["next"]): s for s in Z["sites"]}
    keys = sorted(set(list(sa)[:top]) | set(list(sz)[:top]) | set(k for k in RCO_SITES if k in sa or k in sz),
                  key=lambda k: -(sa.get(k) or sz.get(k))["us_pass"])
    rows = []
    for k in keys:
        x, y = sa.get(k), sz.get(k)
        rows.append("%s %s -> %s %s -> %s us (%s; n %s / %s, median %s / %s, ordered %s / %s)%s" % (
            k[0], k[1], k[2], x["us_pass"] if x else "<top", y["us_pass"] if y else "<top",
            "%+.1f" % (y["us_pass"] - x["us_pass"]) if x and y else "na", x["n_pass"] if x else "-", y["n_pass"] if y else "-",
            x["median_us"] if x else "-", y["median_us"] if y else "-", x["next_ordered"] if x else "-", y["next_ordered"] if y else "-",
            " [RCO]" if k in RCO_SITES else ""))
    print("LCHAQL COMPARE sites (us a pass %s -> %s): %s" % (base, arm, " | ".join(rows)))
    ra, rz = A["rco_sites"], Z["rco_sites"]
    tot_a = sum(v["us_pass"] for v in ra.values())
    tot_z = sum(v["us_pass"] for v in rz.values())
    aa, az = A.get("after_copy") or {}, Z.get("after_copy") or {}
    ek = [k for k in ("exl3r4d_prep_row_kernel", "post_collapse_norm_kernel") if (az.get(k) or {}).get("n_pass")]
    eng = bool(ek) and all(az[k]["ordered"] < 0.5 for k in ek)
    print("LCHAQL COMPARE after a routing copy (the next dispatch; %s -> %s): %s" % (base, arm, " | ".join(
        "%s: %s -> %s a pass, ordered %s -> %s, start after the copy's start %s -> %s us" % (
            k, (aa.get(k) or {}).get("n_pass"), (az.get(k) or {}).get("n_pass"), (aa.get(k) or {}).get("ordered"), (az.get(k) or {}).get("ordered"),
            (aa.get(k) or {}).get("s2s_median_us"), (az.get(k) or {}).get("s2s_median_us"))
        for k in sorted(set(aa) | set(az), key=lambda k: -((aa.get(k) or az.get(k))["n_pass"]))) or "none"))
    print("LCHAQL COMPARE lever RCO: its four idle sites %.1f -> %.1f us a pass (%+.1f; a dispatch that starts beside the copy leaves no idle gap there) | "
          "ordered share of the dispatch after a routing copy: prep_row %s -> %s, post_collapse_norm %s -> %s | %s" % (
              tot_a, tot_z, tot_z - tot_a, (aa.get("exl3r4d_prep_row_kernel") or {}).get("ordered"), (az.get("exl3r4d_prep_row_kernel") or {}).get("ordered"),
              (aa.get("post_collapse_norm_kernel") or {}).get("ordered"), (az.get("post_collapse_norm_kernel") or {}).get("ordered"),
              "ENGAGED (the dispatches after a routing copy play unordered under %s)" % arm if eng else
              ("NOT ENGAGED by the stamps (the dispatches after a routing copy still play ordered under %s; or the sites moved)" % arm if ek else
               "NOT READ (no dispatch after a routing copy in %s's stamped passes)" % arm)))
    return 0


def selftest():
    """a synthetic two-arm stamps file: B orders the dispatch after each routing copy (gap 4 us after the copy's end), Z starts it 1 us
    after the copy's start; one-in-4 stamping (passes 4 steps apart); the reader must see the per-pass idle drop and the engagement"""
    d = tempfile.mkdtemp(prefix="lchaql-")
    p = os.path.join(d, "s.csv")
    K = {"sc": "_ZN12_GLOBAL__N_120scatter_small_kernelEv", "cp": "_ZN12_GLOBAL__N_110aql_copy16EPKDv4_jPS0_m",
         "pr": "_ZN12_GLOBAL__N_123exl3r4d_prep_row_kernelEv", "g": "_ZN12_GLOBAL__N_112gg_mr_kernelEv",
         "ag": "_Z28r4d_ar_gather_twoshot_kernelv", "pc": "_ZN12_GLOBAL__N_125post_collapse_norm_kernelEv"}
    seq_k = ["g", "sc", "cp", "pr", "g", "ag", "cp", "pc"] * 40  # 320 dispatches a pass
    with open(p, "w") as f:
        f.write("pass,seq,queue,kernel,ordered,chained,start_ns,end_ns,play,records\n")
        t = 1_000_000_000
        for arm, p0 in (("B", 0), ("Z", 1000)):
            for pi in range(40):
                pn = p0 + pi * 4 + 4
                play = 0
                cur = t
                for i, k in enumerate(seq_k):
                    dur = {"g": 20000, "sc": 2000, "cp": 1200, "pr": 4800, "ag": 30000, "pc": 6000}[k]
                    prevk = seq_k[i - 1] if i else None
                    if k == "cp":
                        play += 1
                        gap = 3900
                    elif prevk == "cp":
                        play += 1 if k == "pr" else 0
                        gap = 3900 if arm == "B" else -200       # Z: starts 1.0 us after the copy's start (1.2 us copy)
                    else:
                        gap = 3900
                    s = cur + gap
                    ordered = 0 if (arm == "Z" and prevk == "cp") else 1
                    f.write("%d,%d,1033,%s,%d,0,%d,%d,%d,%d\n" % (pn, i, K[k], ordered, s, s + dur, play, 326))
                    cur = max(cur, s + dur) if not (arm == "Z" and prevk == "cp") else s + dur
                t = cur + 4 * 30_000_000
            if arm == "B":
                tb = t
    j = os.path.join(d, "o.json")
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        read_cmd(["read", p, "--win", "c8-1-G-B:B:%d:%d" % (1_000_000_000, tb - 1), "--win", "c8-2-G-Z:Z:%d:%d" % (tb, t), "--json", j])
        compare_cmd(["compare", "--json", j])
    txt = buf.getvalue()
    o = json.load(open(j))
    why = []
    B, Z = o["arms"].get("B", {}), o["arms"].get("Z", {})
    if B.get("passes") != 40 or Z.get("passes") != 40: why.append("passes %s / %s" % (B.get("passes"), Z.get("passes")))
    if not (Z.get("idle_ms", 1) < B.get("idle_ms", 0)): why.append("idle %s / %s" % (B.get("idle_ms"), Z.get("idle_ms")))
    if B["rco_sites"]["play aql_copy16 -> exl3r4d_prep_row_kernel"]["next_ordered"] != 1.0: why.append("B prep_row ordered")
    if (B.get("after_copy") or {}).get("exl3r4d_prep_row_kernel", {}).get("ordered") != 1.0 or (Z.get("after_copy") or {}).get("exl3r4d_prep_row_kernel", {}).get("ordered") != 0.0:
        why.append("after_copy %s / %s" % (B.get("after_copy"), Z.get("after_copy")))
    if "ENGAGED (the dispatches after a routing copy play unordered" not in txt: why.append("engagement line: %r" % [l for l in txt.split("\n") if "lever RCO" in l])
    if "[RCO]" not in txt: why.append("site table")
    print("lchaql selftest: %s" % ("OK" if not why else "FAIL %s" % why))
    return 0 if not why else 1


def main(a):
    if len(a) >= 1 and a[0] == "selftest":
        return selftest()
    if len(a) >= 3 and a[0] == "read":
        return read_cmd(a)
    if len(a) >= 2 and a[0] == "compare":
        return compare_cmd(a)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
