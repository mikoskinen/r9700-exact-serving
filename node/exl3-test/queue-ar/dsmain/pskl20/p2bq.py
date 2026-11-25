#!/usr/bin/env python3
"""p2bq.py (11.11.2026, item 57-p4zyzzzzz-ds-r00zz-psk2b, ds-main-port PSK design-2 builder) -- one deepq-shaped request on
another TEXT: pskld.py (r00z's copy, loaded read only via importlib) with its DEEP_OFF set to --doff, the file offset into
ds41_tests.SRC the body text starts at (deepq.py's 1200 is past SRC's end and wraps to file 0). Each arm's prompts then
share no prefix with another arm's, so a prefix hit can only use the snapshots that arm's own runs wrote (/reset_prefix_cache
drops the cached blocks, not the checkpoint records). Depths of one offset share their prefix (deepq's construction).
  p2bq.py run --pskld P --url URL --tests T --depth D --ntok N --doff O --out J --text TXT
      = pskld.py run --which deep --depth D --ntok N (no --load) on offset O: prints "LONGID {json}" (status, prompt_tokens,
        cached, ttft, tokens, finish, chars, seconds).
  p2bq.py selftest --pskld P --tests T
      offline: two offsets give different first blocks; one offset's 90000-depth body is a prefix of its 135000-depth body.
Standard library only."""
import argparse, importlib.util, sys


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def body(pl, t, depth, off):
    pl.DEEP_OFF = off
    return pl.deep_payload(t, depth, 512)["messages"][0]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "selftest"))
    ap.add_argument("--pskld", required=True)
    ap.add_argument("--tests", required=True)
    ap.add_argument("--url", default="")
    ap.add_argument("--depth", type=int, default=135000)
    ap.add_argument("--ntok", type=int, default=512)
    ap.add_argument("--doff", type=int, required=False, default=1200)
    ap.add_argument("--out")
    ap.add_argument("--text")
    a = ap.parse_args()
    pl = load(a.pskld, "pskld_ro")
    if a.mode == "selftest":
        t = load(a.tests, "ds41_tests_ro")
        b1, b2 = body(pl, t, 2000, 200), body(pl, t, 2000, 400)
        h = "# deepq fixed prompt v1\n"
        task = pl.DEEP_TASK
        x9, x13 = body(pl, t, 90000, 400), body(pl, t, 135000, 400)
        ok = b1.startswith(h) and b2.startswith(h) and b1[:200] != b2[:200]
        ok = ok and x9.endswith(task) and x13.endswith(task) and x13.startswith(x9[: -len(task)])
        print("SELFTEST OK (offsets differ from the first block; depths of one offset share their body)" if ok else "SELFTEST FAIL")
        return 0 if ok else 1
    pl.DEEP_OFF = a.doff
    ns = argparse.Namespace(url=a.url, tests=a.tests, mb=None, which="deep", out=a.out, text=a.text, load=False, conc=4,
                            warm=10, offset=0.5, pf_tok=0, ctx=1000, gen=512, depth=a.depth, ntok=a.ntok)
    return pl.run(ns)


if __name__ == "__main__":
    sys.exit(main())
