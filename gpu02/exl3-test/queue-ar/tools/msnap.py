#!/usr/bin/env python3
"""msnap.py -- the engine's own speculation counters around a phase (read-only HTTP GETs of /metrics and /stats).

  msnap.py snap URL OUT.json      save radiance:* counters (summed over labels) and /stats "passes" into OUT.json
  msnap.py delta A.json B.json    one line: B - A per decode step

Derived (named as such in the line): sequence rows = decode tokens - accepted drafts (every verified sequence row emits
1 + its accepted drafts); verify rows = sequence rows + verified drafts. "drafts" are the drafts the steps VERIFIED
(radiance:draft_tokens_total) -- under RADIANCE_DRAFT_PMIN the head proposes more; its own engine-log line counts those.
"""
import json
import sys
import time
import urllib.request

KEYS = ("radiance:engine_steps_total", "radiance:engine_steps_decode_total", "radiance:engine_steps_mixed_total",
        "radiance:engine_steps_prefill_total", "radiance:engine_steps_ahead_total", "radiance:decode_tokens_total",
        "radiance:draft_tokens_total", "radiance:draft_accepted_total", "radiance:decode_rows_unspeculated_total",
        "radiance:engine_step_seconds_decode_total", "radiance:prefill_tokens_total")


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def snap(base, out):
    d = {"t": time.time(), "m": {}, "passes": {}}
    try:
        for line in get(base.rstrip("/") + "/metrics").splitlines():
            if not line or line[0] == "#":
                continue
            name = line.split("{", 1)[0].split(" ", 1)[0]
            if name not in KEYS:
                continue
            try:
                d["m"][name] = d["m"].get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    except Exception as e:  # noqa: BLE001 -- a missing endpoint is reported, not fatal
        d["err"] = "metrics: %s" % e
    try:
        st = json.loads(get(base.rstrip("/") + "/stats"))
        p = st.get("passes") if isinstance(st, dict) else None
        if p is None and isinstance(st, dict):
            for v in st.values():
                if isinstance(v, dict) and "passes" in v:
                    p = v["passes"]
                    break
        d["passes"] = p or {}
    except Exception as e:  # noqa: BLE001
        d["err"] = (d.get("err", "") + " stats: %s" % e).strip()
    with open(out, "w") as f:
        json.dump(d, f)


def delta(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    dm = {k: b["m"].get(k, 0.0) - a["m"].get(k, 0.0) for k in KEYS}
    dec = dm["radiance:engine_steps_decode_total"]
    mix = dm["radiance:engine_steps_mixed_total"]
    steps = dec + mix
    drafts, acc = dm["radiance:draft_tokens_total"], dm["radiance:draft_accepted_total"]
    toks = dm["radiance:decode_tokens_total"]
    seqrows = toks - acc
    out = ["%.0f s" % (b["t"] - a["t"]), "decode steps %.0f (+%.0f mixed, %.0f ahead)" % (dec, mix, dm["radiance:engine_steps_ahead_total"])]
    if steps > 0:
        out.append("verified drafts %.0f = %.2f a step, accepted %.0f = %.2f a step, acceptance %s" % (
            drafts, drafts / steps, acc, acc / steps, ("%.3f" % (acc / drafts)) if drafts else "n/a"))
        out.append("derived: sequence rows %.2f a step, verify rows %.2f a step, %.3f drafts a sequence row, "
                   "unspeculated rows %.0f" % (seqrows / steps, (seqrows + drafts) / steps,
                                               drafts / seqrows if seqrows else 0.0,
                                               dm["radiance:decode_rows_unspeculated_total"]))
        if dec > 0:
            out.append("decode step %.2f ms" % (1000.0 * dm["radiance:engine_step_seconds_decode_total"] / dec))
    pa, pb = a.get("passes") or {}, b.get("passes") or {}
    if pb:
        dp = {k: (pb.get(k) or 0) - (pa.get(k) or 0) for k in ("played", "recorded", "audited", "issued_unkeyed",
                                                                "issued_refused", "issued_first")}
        tot = sum(dp.values())
        out.append("passes played %d / recorded %d / issued live %d (%s played)" % (
            dp["played"], dp["recorded"], dp["issued_unkeyed"] + dp["issued_refused"] + dp["issued_first"] + dp["audited"],
            ("%.1f %%" % (100.0 * dp["played"] / tot)) if tot else "n/a"))
    for x in (a, b):
        if x.get("err"):
            out.append("note: " + x["err"][:160])
    print(" | ".join(out))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "snap":
        snap(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[1] == "delta":
        delta(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
