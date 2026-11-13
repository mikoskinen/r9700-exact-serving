#!/usr/bin/env python3
"""servediff.py A.yaml B.yaml -- the serve lines (`command: >` blocks) of two compose files, flag by flag (read-only).

${VAR:-default} reads as its default, ${VAR:?msg} / ${VAR} as <VAR>. Prints one line: identical, or the flags only in A,
only in B, and those whose values differ."""
import re
import sys


def serve(path):
    lines = open(path).read().splitlines()
    out, on, ind = [], False, 0
    for ln in lines:
        if not on:
            m = re.match(r"^(\s*)command:\s*[>|]-?\s*$", ln)
            if m:
                on, ind = True, len(m.group(1))
            continue
        if ln.strip() and (len(ln) - len(ln.lstrip())) <= ind:
            break
        out.append(ln.split(" #", 1)[0])
    s = " ".join(out)
    s = re.sub(r"\$\{(\w+):-([^}]*)\}", r"\2", s)
    s = re.sub(r"\$\{(\w+)(:\?[^}]*)?\}", r"<\1>", s)
    flags, cur = {}, None
    for t in s.split():
        if t.startswith("--"):
            cur = t
            flags.setdefault(cur, [])
        elif cur:
            flags[cur].append(t)
    return {k: " ".join(v) for k, v in flags.items()}


a, b = serve(sys.argv[1]), serve(sys.argv[2])
if not a or not b:
    print("not compared: no command block in %s" % (sys.argv[1] if not a else sys.argv[2]))
    sys.exit(0)
only_a = ["%s %s" % (k, a[k]) for k in a if k not in b]
only_b = ["%s %s" % (k, b[k]) for k in b if k not in a]
diff = ["%s: %s -> %s" % (k, a[k], b[k]) for k in a if k in b and a[k] != b[k]]
if not (only_a or only_b or diff):
    print("IDENTICAL serve lines (%d flags)" % len(a))
else:
    print("%d flags; only in A: [%s]; only in B: [%s]; differ: [%s]" % (
        len(a), "; ".join(only_a), "; ".join(only_b), "; ".join(diff)))
