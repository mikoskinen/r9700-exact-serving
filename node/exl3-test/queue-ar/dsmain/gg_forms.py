#!/usr/bin/env python3
"""gg_forms.py -- libr4d/r4d_ds41_gg_forms.h, gemm_nt_ggml's measured decode-band form table, from r4d_selftest --perfgg.

    python3 tools/gg_forms.py [--min-gain PCT] [--out FILE] BENCH.txt...

Reads every GGPERF line (r4d_selftest --perfgg: one (type, shape, M) with the stock rule's time and every pinned
form's, "OUTPUT DIFFERS" after a form whose bytes were not the rule's) and every GGSUM line (queue-ar kbsum.py's
digest of them: the rule's time and the best form's) in the files given; a queue log's "RESULT ... GGSUM ..." lines
work too. One key is (ggml type, N, K, M). The table gets a form for a key only when:

  - its bytes matched the rule's in every run that measured the key (a GGPERF line names a differing form; a GGSUM
    line says "bytes match", or names the forms that differed);
  - it is not the stock rule's own launch (rows a wave and the grid the rule picks; a cap above the row groups is no
    cap), so an entry always changes something;
  - it beats the rule by at least --min-gain percent (default 3) in the mean of the runs' form/rule time ratios, and
    is not slower than the rule in any run. The best such form by that mean is kept.

Every other (type, N, K, M) keeps the stock rule. The bench's own M values are exact keys: an entry covers one M
(m_lo == m_hi), as only that M was measured. The bench lines carry no N x K beyond the shape, so the K of a key is
the bench's shape, the same numbers the engine passes (gemm_nt_ggml's N and K at this rank's extents)."""
import argparse
import re
import sys

TYPES = {"q8_0": "GG_Q8_0", "q5_K": "GG_Q5_K", "q6_K": "GG_Q6_K", "iq4_nl": "GG_IQ4_NL", "iq2_xxs": "GG_IQ2_XXS",
         "iq1_s": "GG_IQ1_S"}
HEAD = re.compile(r"GGPERF (\S+) (\S+) (\d+)x(\d+) M=(\d+)(?: \([\d.]+ MiB\))?([^:]*):(.*)$")
FORM = re.compile(r" (rule|r(\d)g(\d+)) ([\d.]+) us(?: (\d+) GB/s)?( OUTPUT DIFFERS)?;")
SUM = re.compile(r"GGSUM (\S+) (\S+) (\d+)x(\d+) M=(\d+)(?: \([\d.]+ MiB\))?[^|]*\| rule ([\d.]+) us \| best (rule|r(\d)g(\d+)) "
                 r"([\d.]+) us[^|]*\|(.*)$")


def stock(N, M, ty):
    """The stock rule (gg_gemm's decode band): rows a wave, grid."""
    rpw = 2 if (M >= 3 and ty != "f32" and (N + 15) // 16 >= 128) else 1
    nrg = (N + 8 * rpw - 1) // (8 * rpw)
    return rpw, min(nrg, 256)


def launch(N, rpw, cap):
    nrg = (N + 8 * rpw - 1) // (8 * rpw)
    return rpw, min(nrg, cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--min-gain", type=float, default=3.0)
    ap.add_argument("--out", default="-")
    a = ap.parse_args()
    runs = {}      # key -> list of {form: (us, differs)}
    names = {}     # key -> shape name
    srcs = []
    for path in a.files:
        n_perf = n_sum = 0
        for line in open(path, errors="replace"):
            i = line.find("GGPERF ")
            if i >= 0:
                h = HEAD.match(line[i:].rstrip("\n"))
                if not h:
                    continue
                ty, shape, N, K, M = h.group(1), h.group(2), int(h.group(3)), int(h.group(4)), int(h.group(5))
                forms = {}
                for f in FORM.finditer(h.group(7)):
                    nm = "rule" if f.group(1) == "rule" else (int(f.group(2)), int(f.group(3)))
                    forms[nm] = (float(f.group(4)), bool(f.group(6)))
                if "rule" not in forms:
                    continue
                key = (ty, N, K, M)
                runs.setdefault(key, []).append(forms)
                names[key] = shape
                n_perf += 1
                continue
            i = line.find("GGSUM ")
            if i >= 0:
                s = SUM.match(line[i:].rstrip("\n"))
                if not s:
                    continue
                ty, shape, N, K, M = s.group(1), s.group(2), int(s.group(3)), int(s.group(4)), int(s.group(5))
                rule = float(s.group(6))
                tail = s.group(11)
                forms = {"rule": (rule, False)}
                if s.group(7) != "rule":
                    nm = (int(s.group(8)), int(s.group(9)))
                    differs = "OUTPUT DIFFERS" in tail and (s.group(7) in tail.split("OUTPUT DIFFERS in", 1)[-1])
                    forms[nm] = (float(s.group(10)), differs)
                key = (ty, N, K, M)
                # a GGSUM line digests GGPERF lines; when both are given for a key, the GGPERF ones are used
                runs.setdefault(("sum",) + key, []).append(forms)
                names[key] = shape
                n_sum += 1
        srcs.append("%s (%d GGPERF, %d GGSUM)" % (path, n_perf, n_sum))
    keys = sorted({k for k in runs if k[0] != "sum"} | {k[1:] for k in runs if k[0] == "sum"},
                  key=lambda k: (list(TYPES).index(k[0]) if k[0] in TYPES else 99, k[1], k[2], k[3]))
    rows = []
    for key in keys:
        ty, N, K, M = key
        if ty not in TYPES:
            continue
        rs = runs.get(key) or runs.get(("sum",) + key)
        st = stock(N, M, ty)
        cands = {}
        for forms in rs:
            for nm in forms:
                if nm != "rule":
                    cands[nm] = True
        best = None
        for nm in cands:
            if launch(N, nm[0], nm[1]) == st:
                continue                        # the rule's own launch
            ratios, ok = [], True
            for forms in rs:
                if nm not in forms:
                    continue
                us, differs = forms[nm]
                if differs:
                    ok = False
                    break
                ratios.append(us / forms["rule"][0])
            if not ok or not ratios or max(ratios) > 1.0:
                continue
            mean = sum(ratios) / len(ratios)
            if (1.0 - mean) * 100.0 < a.min_gain:
                continue
            if best is None or mean < best[1]:
                best = (nm, mean, len(ratios))
        if best:
            (rpw, cap), mean, n = best
            rule_us = sum(f["rule"][0] for f in rs) / len(rs)
            rows.append((TYPES[ty], N, K, M, M, rpw, cap,
                         "%s %s M=%d: rule %.1f us, r%dg%d %+.1f %% (%d run%s)" % (
                             ty, names[key], M, rule_us, rpw, cap, (mean - 1) * 100, n, "" if n == 1 else "s")))
    out = sys.stdout if a.out == "-" else open(a.out, "w")
    out.write("/* r4d_ds41_gg_forms.h -- GENERATED by tools/gg_forms.py; edit the bench data, not this file.\n"
              " *\n"
              " * gemm_nt_ggml's decode band (r4d_ds41_ggml.hip, gg_gemm, up to 48 rows) under gg_gemv = 1: a measured form\n"
              " * (rows a wave, grid cap) a (ggml type, N, K, M band); every other key keeps the stock rule. A form here was\n"
              " * byte-identical to the rule at its exact (type, shape, M) in every bench run, is not the rule's own launch,\n"
              " * and beat it by >= %.1f %% (mean form / rule time) without losing in any run.\n"
              " *\n"
              " * Sources:\n" % a.min_gain)
    for s in srcs:
        out.write(" *   %s\n" % s)
    out.write(" * %d entr%s. */\n" % (len(rows), "y" if len(rows) == 1 else "ies"))
    out.write("#define R4D_GG_FORMS_N %d\n" % len(rows))
    out.write("static const GgForm kGgForms[] = {\n")
    out.write("    /* type      N       K   M lo  M hi  rows a wave  grid cap */\n")
    for r in rows:
        out.write("    { %-8s %6d, %6d, %4d, %4d, %d, %5d },   /* %s */\n" % (r[0] + ",", r[1], r[2], r[3], r[4], r[5], r[6], r[7]))
    if not rows:
        out.write("    { -1, 0, 0, 0, 0, 0, 0 },   /* none: every key keeps the stock rule */\n")
    out.write("};\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
