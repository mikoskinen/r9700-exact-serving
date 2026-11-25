#!/usr/bin/env python3
"""damark.py (12.11.2026, queue-ar/dsmain/spread, ds-main-port) -- the DA harness's live self-check, read only: the FIXED
marker derivation of queue-ar/dsmain/da/daheld.py (its run(): /tokenize of the reference with messages + reasoning_effort none,
minus the reference text's own ids) against a live server, then how many documents of a corpus (default
runs/xdsm/da/stats/corpus/corpus-gen.jsonl, 192 docs) get a usable prefix with that marker -- daheld.run's own loop and
condition (the last occurrence of the marker; skipped when pl <= 0 or pl >= len(ids) - 6).

Nothing is copied from daheld.py: it is IMPORTED (importlib, no bytecode written), its http() is used, and the two code pieces
that live inline in daheld.run() are taken from its source (inspect.getsource) by their anchor lines and executed as they
stand -- so this checks exactly the lines daheld.run will run. No /completion call; the corpus is only read.

  damark.py check --daheld PATH --url URL --corpus FILE     -> "DAMARK {json}" (marker, docs, usable, skipped, first skips)
  damark.py selftest --daheld PATH                          -> offline: daheld.py loads, both anchors found, the prefix loop
                                                               on synthetic docs (a stub http), "SELFTEST OK ..."
"""
import importlib.util
import inspect
import json
import sys
import textwrap

sys.dont_write_bytecode = True

MARK_FIRST = 'ref = "Hello there"'
MARK_LAST = 'marker = [t for t in marker if t not in body] or tail[-2:]'
LOOP_FIRST = 'ids, pl = d["ids"], -1'
LOOP_LAST = 'if pl <= 0 or pl >= len(ids) - 6:'


def load(path):
    spec = importlib.util.spec_from_file_location("daheld", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def piece(src, first, last):
    """the lines of src from the one holding `first` through the one holding `last`, dedented"""
    lines = src.split("\n")
    i = next((k for k, l in enumerate(lines) if first in l), None)
    j = next((k for k in range(i or 0, len(lines)) if last in lines[k]), None) if i is not None else None
    if i is None or j is None:
        raise RuntimeError("anchor not found in daheld.run: %r .. %r" % (first, last))
    return textwrap.dedent("\n".join(lines[i:j + 1]))


def marker_of(dh, url):
    src = inspect.getsource(dh.run)
    code = piece(src, MARK_FIRST, MARK_LAST)
    ns = {"http": dh.http, "url": url.rstrip("/"), "json": json}
    exec(code, ns)
    return ns["marker"], code


def usable(dh, docs, marker):
    """daheld.run's per-document prefix search, its own lines; the skip condition -> False"""
    src = inspect.getsource(dh.run)
    loop = piece(src, LOOP_FIRST, LOOP_LAST)
    # the piece ends in the skip test's `if` header: give it a body that records the verdict
    body = loop + "\n    skip = True\nelse:\n    skip = False\n"
    out = []
    for d in docs:
        ns = {"d": d, "marker": marker}
        exec(body, ns)
        out.append((not ns["skip"], ns.get("pl")))
    return out


def check(a):
    dh = load(a.daheld)
    marker, code = marker_of(dh, a.url)
    docs = [json.loads(l) for l in open(a.corpus) if l.strip()]
    res = usable(dh, docs, marker)
    n = sum(1 for ok, _ in res if ok)
    skips = [d.get("source") for d, (ok, _) in zip(docs, res) if not ok][:5]
    pls = [pl for ok, pl in res if ok]
    print("DAMARK " + json.dumps({"marker": marker, "docs": len(docs), "usable": n, "skipped": len(docs) - n,
                                  "first_skips": skips, "prompt_len_min": min(pls) if pls else None,
                                  "prompt_len_max": max(pls) if pls else None, "derivation_lines": len(code.split("\n"))}), flush=True)
    return 0


def selftest(a):
    dh = load(a.daheld)
    src = inspect.getsource(dh.run)
    c1 = piece(src, MARK_FIRST, MARK_LAST)
    c2 = piece(src, LOOP_FIRST, LOOP_LAST)

    def stub(url, body=None, timeout=900):   # a template: <bos> <User> text <Assistant></think>
        ids = [7, 8, 9] if body.get("content") == "Hello there" else None
        if body.get("messages"):
            return {"tokens": [0, 100, 7, 8, 9, 101, 102]}
        return {"tokens": ids}
    ns = {"http": stub, "url": "http://x", "json": json}
    exec(c1, ns)
    ok1 = ns["marker"] == [101, 102]
    docs = [{"ids": [0, 100, 5, 6, 101, 102] + list(range(200, 220)), "source": "a"},   # usable: pl 6
            {"ids": [0, 100, 5, 6] + list(range(200, 220)), "source": "b"},             # no marker
            {"ids": [0, 100, 5, 101, 102, 1, 2, 3], "source": "c"}]                     # marker too near the end
    r = usable(dh, docs, [101, 102])
    ok2 = [x for x, _ in r] == [True, False, False] and r[0][1] == 6
    print(("SELFTEST OK" if ok1 and ok2 else "SELFTEST FAIL") + ": daheld.py loads, marker lines %d (stub -> %s), prefix loop lines %d (synthetic docs -> %s)" % (
        len(c1.split("\n")), ns["marker"], len(c2.split("\n")), [x for x, _ in r]))
    return 0 if ok1 and ok2 else 1


def main():
    import argparse
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd")
    c = sp.add_parser("check")
    c.add_argument("--daheld", required=True); c.add_argument("--url", required=True); c.add_argument("--corpus", required=True)
    s = sp.add_parser("selftest")
    s.add_argument("--daheld", required=True)
    a = ap.parse_args()
    if a.cmd == "check":
        return check(a)
    if a.cmd == "selftest":
        return selftest(a)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
