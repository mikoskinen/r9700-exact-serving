#!/usr/bin/env python3
"""scan.py (12.11.2026, queue-ar/tools/mockrad) -- the report of one mockrun.sh run: everything a script bug leaves behind.
  scan.py --sb SANDBOX --name N --target T --rc RC [--timeout 0|1] [--seconds S] [--knobs ..] [--faults ..] [--swaps ..] [--fresh REL ...]
Reads the sandbox's body.out, its RESULT log (runs/ds41-exl3.log of the sandbox), every text file of the output dirs (--fresh) and the
mock's own logs (guard.log, docker-calls.log, mockrad-events.log, mockrad.out), then prints:
  TRACEBACKS   every Python traceback: file, exception, the innermost frame (the bug's line)
  BASH         unbound variable / syntax error / command not found / integer expression / bad substitution / ... in body.out
  VIRHE        every RESULT line with VIRHE (grouped; a line that compares with a REAL-run reference is labelled "mock: real ref")
  MISSING      "No such file", [Errno 2], MISSING, "missing", "not compared" lines
  MOCK         refused connects / writes / commands, docker calls not emulated, unknown endpoints, fn4 POSTs refused (the body's
               own range probes -- ds41_psk 99, draft_occ 65, ... -- are labelled), the leak check
Exit 0 = clean (no traceback, no bash error, no VIRHE), 1 = findings."""
import argparse, collections, gzip, json, os, re, sys

BASHRE = re.compile(r"unbound variable|syntax error|command not found|integer expression expected|bad substitution|ambiguous redirect|"
                    r"unary operator expected|too many arguments|binary operator expected|invalid arithmetic|division by 0|"
                    r"value too great for base|not a valid identifier|bad array subscript")
MISSRE = re.compile(r"No such file or directory|\[Errno 2\]|\bMISSING\b|\bmissing\b|not compared|FileNotFoundError|cannot access")
REALREF = re.compile(r"vs stock|vs dsm-225|vs bz d0|vs the stock|stack4 s4-n5|cross-start|x117|int7 image|dsd-d0|warm-up answers")
PROBES = re.compile(r"ds41_psk takes 0 \.\. 4 \(got 99\)|draft_occ takes 0 \.\. 64 \(got 65\)|sched_noa takes 0 \.\. 2 \(got 3\)|"
                    r"stage_short takes 0 \.\. 65536 \(got 65537\)|place_hb takes 0 \.\. 4096 \(got 4097\)")


def rd(p, limit=50_000_000):
    try:
        if p.endswith(".gz"):
            return ""
        if os.path.getsize(p) > limit: return ""
        return open(p, errors="replace").read()
    except OSError:
        return ""


def tracebacks(text, src):
    out, L, i = [], text.splitlines(), 0
    while i < len(L):
        if "Traceback (most recent call last):" in L[i]:
            frames, j = [], i + 1
            while j < len(L) and (L[j].startswith(" ") or L[j].startswith("\t") or not L[j].strip()):
                if L[j].strip().startswith("File "): frames.append(L[j].strip() + " :: " + (L[j + 1].strip() if j + 1 < len(L) else ""))
                j += 1
            exc = L[j].strip() if j < len(L) else "(no exception line: the log keeps only the Traceback line)"
            if "RESULT " in L[i] and not frames: exc = "(logged line only) " + L[i].split("RESULT ", 1)[1][:300]
            out.append((src, exc[:400], frames[-1][:400] if frames else ""))
            i = j
        else:
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("sb", "name", "target", "rc", "knobs", "faults", "swaps"): ap.add_argument("--" + k, default="")
    ap.add_argument("--timeout", type=int, default=0)
    ap.add_argument("--seconds", type=int, default=0)
    ap.add_argument("--fresh", action="append", default=[])
    a = ap.parse_args()
    sb, st, home = a.sb, os.path.join(a.sb, "state"), os.path.join(a.sb, "home")
    log = os.path.join(home, "gpu/ds41/exl3-test/runs/ds41-exl3.log")
    body = rd(os.path.join(sb, "body.out"))
    L = rd(log)
    res = [l for l in L.splitlines() if " RESULT " in l]
    files = [(os.path.join(sb, "body.out"), body), (log, L)]
    for f in a.fresh:
        d = os.path.join(home, f)
        for dp, _, fns in os.walk(d):
            for fn in fns:
                p = os.path.join(dp, fn)
                if not p.endswith(".gz") and os.path.isfile(p): files.append((p, rd(p, 20_000_000)))
    tb = []
    for p, t in files:
        if "Traceback" in t: tb += tracebacks(t, os.path.relpath(p, sb))
    seen, tbu = {}, []
    for x in tb:
        k = (x[1], x[2])
        if k in seen: seen[k] += 1; continue
        seen[k] = 0; tbu.append(x)
    silent = [l.split("RESULT ", 1)[1][:200] for l in res if "Traceback" in l and "VIRHE" not in l]
    bash = [l.strip() for l in body.splitlines() if BASHRE.search(l)]
    virhe = [l for l in res if "VIRHE" in l and not re.search(r"no VIRHE line|\d+ VIRHE line\(s\): see above", l)]
    vsum = [l for l in res if re.search(r"\d+ VIRHE line\(s\): see above", l)]
    miss = []
    for p, t in ((os.path.join(sb, "body.out"), body), (log, L)):
        for l in t.splitlines():
            if MISSRE.search(l) and not l.lstrip().startswith("File "): miss.append(l.strip()[:400])
    g = rd(os.path.join(st, "guard.log")).splitlines()
    dk = [l for l in rd(os.path.join(st, "docker-calls.log")).splitlines() if "NOT-EMULATED" in l]
    ev = rd(os.path.join(st, "mockrad-events.log")).splitlines()
    unk = collections.Counter(l.split("unknown endpoint ", 1)[1] for l in ev if "unknown endpoint" in l)
    refused = [l.split("fn4 POST refused: ", 1)[1] for l in ev if "fn4 POST refused" in l]
    probes = [x for x in refused if PROBES.search(x) and len(x) < 60]
    refused = [x for x in refused if x not in probes]
    mo = rd(os.path.join(st, "mockrad.out"))
    vf = rd(os.path.join(sb, "verify.txt")).strip()
    nreq = sum(1 for l in ev if l[20:].startswith("req "))

    P = print
    P("MOCKRUN REPORT %s" % a.name)
    P("target %s | body rc %s%s | %d s | knobs: %s | faults: %s | swaps: %s" % (a.target, a.rc, " (TIMEOUT: stopped)" if a.timeout else "",
      a.seconds, a.knobs or "-", a.faults or "-", a.swaps or "-"))
    P("output dirs (fresh): %s | %d RESULT lines, %d mock requests | last RESULT: %s" % (", ".join(a.fresh) or "-", len(res), nreq,
      (res[-1].split("RESULT ", 1)[1][:300] if res else "none")))
    P("")
    P("TRACEBACKS (%d distinct)" % len(tbu))
    for src, exc, fr in tbu:
        n = seen.get((exc, fr), 0)
        P("  %s: %s%s%s" % (src, exc, " (+%d more place(s))" % n if n else "", ("\n      at " + fr) if fr else ""))
    if silent:
        P("  NOTE: %d RESULT line(s) carry a Traceback WITHOUT a VIRHE -- the body does not count a crashed helper as VIRHE: %s" % (
          len(silent), " | ".join(silent[:3])))
    P("BASH ERRORS (%d)" % len(bash))
    for l in list(dict.fromkeys(bash))[:30]: P("  " + l[:400])
    P("VIRHE LINES (%d)" % len(virhe))
    grp = collections.OrderedDict()
    for l in virhe:
        msg = l.split("RESULT ", 1)[1] if "RESULT " in l else l
        key = re.sub(r"\d+(\.\d+)?", "#", msg)[:160]
        grp.setdefault(key, []).append(msg)
    for k, v in grp.items():
        lab = "  [mock: real ref]" if REALREF.search(v[0]) else ""
        P("  %dx%s %s" % (len(v), lab, v[0][:700]))
    P("MISSING / NOT FOUND (%d)" % len(miss))
    for l in list(dict.fromkeys(miss))[:30]: P("  " + l)
    P("MOCK SIDE")
    gs = [x for x in g if "GPU sysfs" in x]; go = [x for x in g if "GPU sysfs" not in x]
    P("  GPU sysfs telemetry reads refused: %d (fastcheck / thermal / keepawake readers; expected, the real cards are not polled)" % len(gs))
    P("  other guard refusals (%d)%s" % (len(go), "".join("\n    " + x[:300] for x in list(dict.fromkeys(x.split(": ", 1)[-1] for x in go))[:15])))
    P("  docker calls not emulated (%d)%s" % (len(dk), "".join("\n    " + x[:300] for x in dk[:10])))
    P("  unknown endpoints: %s" % (", ".join("%s x%d" % kv for kv in unk.items()) or "none"))
    P("  fn4 POSTs refused: %d by the body's own range probes, %d other%s" % (len(probes), len(refused), "".join("\n    " + x[:400] for x in refused[:10])))
    if "Traceback" in mo: P("  mockrad.py itself raised: %s" % [l for l in mo.splitlines() if l.strip()][-1][:300])
    P("  leak check (real files behind the farm's symlinks; the real queue and loggers write some of them too): %s" % (vf.replace("\n", "\n    ") or "not run"))
    bad = bool(tbu or bash or virhe or silent)
    P("")
    P("VERDICT: %s" % ("FINDINGS -- see above (a VIRHE labelled 'mock: real ref' compares with a real-run reference and is expected under the mock)" if bad else "CLEAN: no traceback, no bash error, no VIRHE"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
