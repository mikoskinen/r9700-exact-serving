#!/usr/bin/env python3
"""mbt.py (11.11.2026, queue-ar/dsmain/stack3, ds-main-port stack3 builder) -- mb.py's phases at a FIXED TEMPERATURE.

mb.py (~/gpu/ds41/mb.py) sends no temperature (the server-default sampling, temperature 1). The stack3 item measures its
in-process A/B at temperature 0 as well (lever SD's draft_sample does nothing to a greedy request, so a T 0 block reads lever
BI's cost alone). This wrapper loads mb.py READ ONLY from --mb (importlib; nothing of it changes, no bytecode written) and
replaces its module-level stream() with one that adds {"temperature": T} to every request (mb.py's closed_loop / phase_pf /
phase_turns call stream() through the module's globals), then runs the phases as mb.py's __main__ does. Output: mb.py's
"MB {json}" line a phase with "temperature": T added.

  mbt.py --mb ~/gpu/ds41/mb.py --temperature 0 --tag T [--phases c4,c8,pf] [--dur 200] [--warm 20] [--ctx 1000] [--gen 512]
  mbt.py --mb ~/gpu/ds41/mb.py --selftest      offline: mb.py loads, has what this wrapper calls, stream() takes `extra`.
Standard library only.
"""
import argparse, importlib.util, inspect, json, sys, urllib.request

sys.dont_write_bytecode = True


def load(path):
    spec = importlib.util.spec_from_file_location("mb", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", required=True)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--tag", default="")
    ap.add_argument("--phases", default="c4,c8,pf,turns")
    ap.add_argument("--dur", type=float, default=200)
    ap.add_argument("--warm", type=float, default=20)
    ap.add_argument("--ctx", type=int, default=1000)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    mb = load(a.mb)
    need = ("stream", "phase_c", "phase_pf", "phase_turns", "closed_loop", "metrics", "URL")
    miss = [n for n in need if not hasattr(mb, n)]
    if a.selftest:
        ok = not miss and "extra" in inspect.signature(mb.stream).parameters
        print("SELFTEST " + ("OK: mb.py %s loads, %s present, stream() takes extra" % (a.mb, ", ".join(need)) if ok
                             else "FAILED: missing %s%s" % (miss, "" if "extra" in inspect.signature(getattr(mb, "stream", lambda: 0)).parameters else ", stream() has no extra")))
        return 0 if ok else 1
    if miss or a.temperature is None:
        print("MB " + json.dumps({"error": "mb.py lacks %s or no --temperature" % miss}), flush=True)
        return 1
    orig = mb.stream
    t = a.temperature

    def stream(msgs, max_tokens, effort="medium", stamps=None, extra=None):
        e = dict(extra or {})
        e["temperature"] = t
        return orig(msgs, max_tokens, effort=effort, stamps=stamps, extra=e)

    mb.stream = stream
    with urllib.request.urlopen(mb.URL + "/v1/models", timeout=60) as r:
        mb.MODEL = json.load(r)["data"][0]["id"]
    for ph in a.phases.split(","):
        if ph.startswith("c"):
            res = mb.phase_c(int(ph[1:]), a)
        elif ph == "pf":
            res = mb.phase_pf(a)
        else:
            res = mb.phase_turns(a)
        print("MB " + json.dumps(dict(tag=a.tag, temperature=t, **res)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
