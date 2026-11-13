#!/usr/bin/env python3
"""daeval.py (11.11.2026, queue-ar/dsmain/da, lever DA pilot, tag xdap) -- the pilot's acceptance anatomy, adapted vs current
drafter, at the fixed window 5 (the walk on): daanat.py's collection and replay (imported from ../daanat.py, the xda2 screen's,
unchanged) over THREE prompt sets -- "prose" = daanat's 20 English prose prompts (the 1-user probe's essay first), "code" =
daanat's 6 code prompts (the probe's code first), "fi" = 10 Finnish prose prompts written for this pilot (none in the training
corpus). Never a user transcript.
  collect URL LOG OUT --prose N --code N --fi N --max-tokens T   rows -> OUT-rows.jsonl (daanat's row form, kind prose/code/fi)
  analyse ROWS   per kind: daanat.score_row / summarise -> "DALINE ..." lines and "DASUM {json}"
  compare A_SUM B_SUM   B (adapted) vs A (current) per kind: tokens a step (fixed window 5, replay + chain), p1..p5
  selftest   daanat's selftest + the Finnish set's size"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import daanat as DA   # noqa: E402

FI = [
    "Kirjoita pitkä ja yksityiskohtainen essee (noin 700 sanaa) kirjapainotaidon historiasta ja sen vaikutuksesta Euroopan tieteeseen ja uskontoon.",
    "Selitä suurelle yleisölle, miten ilmastonmuutos vaikuttaa Itämeren ekosysteemiin. Käytä useita kappaleita ja esimerkkejä.",
    "Kirjoita noin 500 sanan novelli majakanvartijasta, joka löytää pullopostin.",
    "Kirjoita jäsennelty essee vuoden 2008 finanssikriisin syistä ja seurauksista Suomelle.",
    "Kirjoita matkaoppaan osio kolmen päivän vierailusta Helsingissä käytännön vinkein jokaiselle päivälle.",
    "Kirjoita kaupunginvaltuustolle perusteltu kirje joukkoliikenteen kehittämisen puolesta konkreettisin esimerkein.",
    "Selitä lukion oppikirjan tyyliin, miten fotosynteesi toimii ja miksi se on elämän kannalta keskeinen.",
    "Kirjoita yksityiskohtainen arvostelu keskihintaisesta sähköpyörästä: rakenne, ajotuntuma, akku ja hinta-laatusuhde.",
    "Tiivistä Seitsemän veljeksen juoni ja keskeiset teemat noin 600 sanan esseeksi.",
    "Kirjoita blogikirjoitus, jossa annat käytännön neuvoja aikuiselle, joka aloittaa kitaransoiton.",
]


def collect(a):
    url = a.url.rstrip("/")
    model = DA.model_id(url)
    jobs = [("prose", i, p) for i, p in enumerate(DA.PROSE[:a.prose])] + [("code", i, p) for i, p in enumerate(DA.CODE[:a.code])] + \
           [("fi", i, p) for i, p in enumerate(FI[:a.fi])]
    with open(a.out + "-rows.jsonl", "w") as fo:
        for kind, i, p in jobs:
            o0 = DA.logsize(a.log)
            t0 = time.time()
            d = DA.chat(url, model, p, a.max_tokens, a.top)
            dt = time.time() - t0
            time.sleep(a.settle)
            o1 = DA.logsize(a.log)
            with open(a.log, "rb") as f:
                f.seek(o0)
                chunk = f.read(max(0, o1 - o0)).decode(errors="replace")
            props = {}
            for m in DA.PAT.finditer(chunk):
                props.setdefault(m.group(1), []).append([int(x) for x in m.group(2).split()])
            seq, blocks = max(props.items(), key=lambda kv: len(kv[1])) if props else (None, [])
            c = d["choices"][0]
            lp = (c.get("logprobs") or {}).get("content") or []
            text = (c.get("message") or {}).get("content") or ""
            row = {"kind": kind, "i": i, "seq": seq, "seqs_in_window": len(props), "finish": c.get("finish_reason"),
                   "s": round(dt, 3), "ids": [e.get("id") for e in lp], "lp": [e.get("logprob") for e in lp],
                   "top": [[[t.get("id"), t.get("logprob")] for t in (e.get("top_logprobs") or [])] for e in lp],
                   "blocks": blocks, "timings": d.get("timings"), "usage": d.get("usage"), "chars": len(text), "text": text}
            fo.write(json.dumps(row) + "\n")
            fo.flush()
            print("DACOLLECT " + json.dumps({"kind": kind, "i": i, "tokens": len(lp), "blocks": len(blocks), "seqs": len(props),
                                             "s": round(dt, 2), "finish": c.get("finish_reason")}), flush=True)
    return 0


def analyse(a):
    rows = [json.loads(l) for l in open(a.rows) if l.strip()]
    res, lines = {}, []
    for kind, now in (("prose", DA.NOW["prose"]), ("code", DA.NOW["code"]), ("fi", None)):
        S = DA.new_acc()
        for r in rows:
            if r.get("kind") == kind:
                DA.score_row(r, S)
        if not S["rows"] and not S["rows_bad"]:
            continue
        o, L = DA.summarise(kind, S, now)
        res[kind] = o
        lines += L[:1] + L[1:2]
    for l in lines:
        print("DALINE " + l)
    print("DASUM " + json.dumps(res))
    return 0


def compare(a):
    def rd(p):
        for l in open(p):
            if l.startswith("DASUM "):
                return json.loads(l[6:])
        return {}
    A, B = rd(a.a), rd(a.b)
    out = []
    for k in ("prose", "fi", "code"):
        x, y = A.get(k) or {}, B.get(k) or {}
        bx, by = x.get("bounds"), y.get("bounds")
        if not bx or not by:
            out.append("%s: not compared (A %s, B %s)" % (k, bool(bx), bool(by)))
            continue
        f3 = lambda v: " ".join("%.3f" % t for t in v)
        out.append("%s: tokens a step (fixed 5, replay) A %.3f -> B %.3f (%+.1f %%), chain %.3f -> %.3f (%+.1f %%); p1..p5 A %s | "
                   "B %s; first rejections A-class (target p_top1 >= 0.8) %d -> %d; acc at p_top1 >= 0.9 %d/%d -> %d/%d" % (
                       k, bx["replay"], by["replay"], 100 * (by["replay"] / bx["replay"] - 1), bx["now"], by["now"],
                       100 * (by["now"] / bx["now"] - 1), f3(x["cond_now"]), f3(y["cond_now"]), sum(x["A"]), sum(y["A"]),
                       x["acc_at_ptop1_ge_0.9"][0], x["acc_at_ptop1_ge_0.9"][1], y["acc_at_ptop1_ge_0.9"][0],
                       y["acc_at_ptop1_ge_0.9"][1]))
    for l in out:
        print("DACMP " + l)
    return 0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd")
    c = sp.add_parser("collect")
    c.add_argument("--url", required=True); c.add_argument("--log", required=True); c.add_argument("--out", required=True)
    c.add_argument("--prose", type=int, default=20); c.add_argument("--code", type=int, default=6); c.add_argument("--fi", type=int, default=10)
    c.add_argument("--max-tokens", type=int, default=384); c.add_argument("--top", type=int, default=20)
    c.add_argument("--settle", type=float, default=1.5)
    n = sp.add_parser("analyse"); n.add_argument("rows")
    m = sp.add_parser("compare"); m.add_argument("a"); m.add_argument("b")
    sp.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        rc = DA.selftest()
        print("DAEVAL selftest fi prompts %d, daanat prose %d code %d" % (len(FI), len(DA.PROSE), len(DA.CODE)))
        return rc
    return {"collect": collect, "analyse": analyse, "compare": compare}.get(a.cmd, lambda a: 2)(a)


if __name__ == "__main__":
    sys.exit(main())
