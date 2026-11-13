#!/usr/bin/env python3
"""nctab.py (12.11.2026, queue-ar/dsmain/ync, the ds41-nc builder for ds-main-port) -- lever A11's (GPF-1k) table from the P0 bench.
Standard library only.

  nctab.py PERF.txt --gptab GPTAB --range LO-HI [--min-pct 1.0] [--out TAB.txt]
      PERF.txt = r4d_selftest --perfgp M1,M2,.. (one card): GPPERF lines, every identical form's us a call at each M (ablations and
      forms whose bytes differ left out). Per (type, N, K): the SUM over the listed Ms of each form's us (stock = form 0, gg_wm_kernel;
      322 = the stock geometry in gg_pf_kernel) against the SUM of what int12's rule runs there (GPTAB's entry for the shape, else the
      stock kernel). The fastest form becomes a ranged entry "type:NxK@LO-HI=form" when it beats int12's pick by >= --min-pct %
      (form 0 = the stock kernel in that range: where int12's 426 / 422 entry is slower than the stock at these heights).
      Prints NCTAB lines (one a shape: int12's pick, the best, us at each M, the gain) and "NCTABLE <entries or ->"; --out writes the
      comma-joined entries (empty file: no entry). Exit 0; 3 when a GPPERF line says OUTPUT DIFFERS (a form's bytes differ: VIRHE).
  nctab.py selftest"""
import re, sys, tempfile, os

FORM = re.compile(r" (stock|m\d+n\d+w\d+f?(?:-abl\d+)?) (-?[\d.]+) \([\d.]+ TF\)( \(ablation: differs\)| OUTPUT DIFFERS)?;")
HEAD = re.compile(r"^GPPERF (\S+) (\S+) (\d+)x(\d+) M=(\d+) ")


def code_of(name):
    if name == "stock": return 0
    m = re.match(r"^m(\d+)n(\d+)w(\d+)(f?)$", name)
    if not m: return None
    mt, nt, wn, fi = int(m.group(1)), int(m.group(2)), int(m.group(3)), 1 if m.group(4) else 0
    return 100 * mt + 10 * nt + wn // 4 + 4 * fi


def gptab(s):
    t = {}
    for e in (s or "").split(","):
        m = re.match(r"^(\w+):(\d+)x(\d+)=(\d+)$", e.strip())
        if m: t[(m.group(1), int(m.group(2)), int(m.group(3)))] = int(m.group(4))
    return t


def parse(path):
    """-> {(type, N, K): {"name": shape, "M": {M: {code: us}}}}, differs"""
    sh, differs = {}, []
    for l in open(path):
        h = HEAD.match(l)
        if not h: continue
        ty, nm, N, K, M = h.group(1), h.group(2), int(h.group(3)), int(h.group(4)), int(h.group(5))
        d = sh.setdefault((ty, N, K), {"name": nm, "M": {}})
        row = d["M"].setdefault(M, {})
        for f in FORM.finditer(l):
            name, us, tag = f.group(1), float(f.group(2)), f.group(3) or ""
            if "-abl" in name: continue
            if "OUTPUT DIFFERS" in tag: differs.append("%s %s M=%d %s" % (ty, nm, M, name)); continue
            c = code_of(name)
            if c is not None and us > 0: row[c] = us
    return sh, differs


def build(path, gpt, lo, hi, minpct):
    sh, differs = parse(path)
    t = gptab(gpt)
    lines, ent = [], []
    for (ty, N, K), d in sorted(sh.items()):
        Ms = sorted(d["M"])
        if not Ms: continue
        common = set.intersection(*(set(d["M"][m]) for m in Ms))
        rule = t.get((ty, N, K), 0)
        if rule not in common:
            lines.append("NCTAB %s %s %dx%d: int12's pick %d not timed at every M -- no entry" % (ty, d["name"], N, K, rule)); continue
        tot = {c: sum(d["M"][m][c] for m in Ms) for c in common}
        best = min(sorted(tot), key=lambda c: tot[c])
        gain = 100.0 * (1.0 - tot[best] / tot[rule]) if tot[rule] > 0 else 0.0
        take = best != rule and gain >= minpct
        per = " ".join("M%d %s %.1f -> %.1f" % (m, rule, d["M"][m][rule], d["M"][m][best]) for m in Ms)
        lines.append("NCTAB %s %s %dx%d: int12's pick %s %.1f us summed over M %s, best %s %.1f us (%+.2f %%)%s | %s" % (
            ty, d["name"], N, K, rule or "stock", tot[rule], ",".join(map(str, Ms)), best or "stock", tot[best], -gain,
            " -> ENTRY" if take else (" (int12's pick is the best)" if best == rule else " (under %.1f %%: no entry)" % minpct), per))
        if take: ent.append("%s:%dx%d@%d-%d=%d" % (ty, N, K, lo, hi, best))
    return lines, ent, differs


def main(argv):
    if not argv or argv[0] in ("-h", "--help"): print(__doc__); return 2
    if argv[0] == "selftest": return selftest()
    path = argv[0]
    kv = dict(zip(argv[1::2], argv[2::2]))
    lo, hi = (int(x) for x in kv.get("--range", "1000-1536").split("-"))
    lines, ent, differs = build(path, kv.get("--gptab", ""), lo, hi, float(kv.get("--min-pct", 1.0)))
    for l in lines: print(l)
    for x in differs: print("NCTAB VIRHE OUTPUT DIFFERS: %s" % x)
    print("NCTABLE %s" % (",".join(ent) if ent else "-"))
    if "--out" in kv: open(kv["--out"], "w").write(",".join(ent))
    return 3 if differs else 0


def selftest():
    d = tempfile.mkdtemp(prefix="nctab-")
    p = os.path.join(d, "perf.txt")
    L = []
    for M, a, b in ((1040, 300.0, 280.0), (1104, 320.0, 300.0)):
        L.append("GPPERF q6_K attn_output_b 5120x2048 M=%d (8.20 MiB): stock %.1f (63.2 TF); m3n2w8 330.0 (55.5 TF); m4n2w8f %.1f (60.1 TF); "
                 "m3n2w8-abl1 200.0 (70.0 TF) (ablation: differs);" % (M, b, a))
        L.append("GPPERF q6_K attn_q_b 8192x1280 M=%d (8.20 MiB): stock 400.0 (62.1 TF); m2n2w8 300.0 (58.2 TF); m4n2w8f 350.0 (62.7 TF);" % M)
        L.append("GPPERF q5_K attn_kv 512x5120 M=%d (2.05 MiB): stock 100.0 (60.5 TF); m4n2w8 99.5 (67.2 TF);" % M)
        L.append("GPPERF q8_0 shexp_down 5120x768 M=%d (3.98 MiB): stock 120.0 (66.2 TF); m3n2w8 110.0 (67.2 TF); m4n2w8 118.0 (62.1 TF); m3n2w4 100.0 (55.6 TF);" % M)
    open(p, "w").write("lever GPF timing\n" + "\n".join(L) + "\n")
    g = "q6_K:8192x1280=426,q6_K:5120x2048=426,q8_0:5120x768=422"
    lines, ent, differs = build(p, g, 1000, 1536, 1.0)
    want = ["q6_K:5120x2048@1000-1536=0", "q6_K:8192x1280@1000-1536=222", "q8_0:5120x768@1000-1536=321"]
    why = []
    if sorted(ent) != sorted(want): why.append("entries %s" % ent)
    if differs: why.append("differs")
    if not any("attn_kv" in l and "under 1.0 %" in l for l in lines): why.append("kv under min")
    open(p, "a").write("GPPERF q8_0 shexp_down_w512 5120x512 M=1040 (2.66 MiB): stock 186.6 (57.6 TF); m3n2w8 170.4 (63.0 TF) OUTPUT DIFFERS;\n")
    _, _, differs = build(p, g, 1000, 1536, 1.0)
    if not differs: why.append("differs not caught")
    if code_of("m4n2w8f") != 426 or code_of("m3n2w4") != 321 or code_of("m2n4w8f") != 246 or code_of("stock") != 0: why.append("codes")
    if why:
        print("\n".join(lines)); print("SELFTEST FAIL: " + ", ".join(why)); return 1
    print("SELFTEST OK: int12's 426 slower than stock -> =0; 426 slower than 222 -> =222; 422 -> 321; a < 1 % gain no entry; OUTPUT DIFFERS caught; form codes")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
