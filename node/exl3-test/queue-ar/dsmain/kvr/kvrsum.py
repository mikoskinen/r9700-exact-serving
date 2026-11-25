#!/usr/bin/env python3
"""kvrsum.py (12.11.2026, queue-ar dsmain/kvr, ds-main-port kvr builder) -- the parser of 57-p4zyzzzzz-ds-r00zzz-cz-x-kvr
(body dsm-kvr.sh). Lever KVR: RADIANCE_DS41_KV_REPLICATED=1 declares deepseek41's prefix-cached groups (kv_index_r*,
kv_compressed_r*) rad_kv_group_replicated, so a host-tier slot holds them once instead of once a rank. Standard library only.

  kvrsum.py cap --log START_LOG --sessions SESS.json --env ENV.txt --tag T --mode A|B|V [--ref CAP_A.json] --json OUT
      the start log's tier lines (the transfer entry, the tier's capacity and slot, the 'held once a slot' line, deepseek41's
      'lever KVR declared' line), /sessions' occupancy (host_cap_bytes), the container env read back -> OUT and one
      "CAP OK|FAIL ..." line. A: no replicated window, no KVR line, the env word absent. B / V: the slot = the transfer entry /
      the world (quartered at TP4: every cached group replicated), tokens x the world against the 'rather than' figure (and
      A's when --ref), the four groups named in both lines, the env word read back (V: + RADIANCE_DEBUG_TIER_VERIFY=1).
  kvrsum.py hit --cold C.json --hit H.json --s0 SESS0.json --s1 SESS1.json --s2 SESS2.json --tag T --json OUT
      resume_ho.py's cold and hit records (their /sessions tier counters before / after) + /sessions around them (s0 before
      the cold, s1 when the cold's VRAM copies were dropped, s2 after the hit) -> host bytes an entry (occupancy / to_host),
      the hit's promoted entries, restores and restore ms -> "HIT OK|FAIL ...". FAIL: the hit not 200, nothing cached, no
      entry promoted from the host tier or no restore in the hit's window.
  kvrsum.py verify --log ENGINE_LOG(.gz) --tag T
      RADIANCE_DEBUG_TIER_VERIFY's lines: 'differ between rank' (a replicated group not alike), 'came back different' (a
      round trip that moved wrong bytes), 'no pinned host memory ... not compared' (the check off) -> "VERIFY OK|FAIL ...".
  kvrsum.py report --dir O --starts "A B V"
      RESULT lines: capacity per start, the tier hit per start, speed per start (1u essay / code, mb c8 at T 0; B vs A,
      cross-start, labelled), identity (identity.txt), guards (guard.txt).
  kvrsum.py selftest
"""
import gzip, json, math, os, re, statistics, sys

UNIT = {"B": 1, "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3, "TiB": 1024 ** 4}
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
GROUPS = {"kv_index_r1", "kv_index_r2", "kv_compressed_r1", "kv_compressed_r2"}


def arg(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a and a.index(name) + 1 < len(a) else default


def size(s):
    m = re.match(r"\s*([0-9.]+)\s*(B|KiB|MiB|GiB|TiB)\s*$", s or "")
    return float(m.group(1)) * UNIT[m.group(2)] if m else None


def lines(path):
    try:
        op = gzip.open if path.endswith(".gz") else open
        with op(path, "rt", errors="replace") as f:
            return f.read().splitlines()
    except OSError:
        return []


def rdj(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError, TypeError):
        return {}


RE_XFER = re.compile(r"kv tier transfer: (\d+) paged group\(s\), ([0-9.]+ \w+) an entry(?: in (\d+) blocks)?"
                     r"(?: \(([0-9.]+ \w+) of it replicated on every rank and held once a slot\))?")
RE_TIER = re.compile(r"kv tiers: a finished conversation's KV is copied to host memory at once \(([0-9.]+ \w+), (\d+) entries, "
                     r"(\d+) tokens of context\).*One entry is ([0-9.]+ \w+)")
RE_ONCE = re.compile(r"kv tiers: (.+?) \(([0-9.]+ \w+) of each entry\) is the same on every rank and held once a slot: "
                     r"([0-9.]+ \w+) a slot rather than ([0-9.]+ \w+), so --prefix-cache-mib (\d+) holds (\d+) tokens rather than (\d+)")
RE_KVR = re.compile(r"lever KVR declared \(RADIANCE_DS41_KV_REPLICATED=1\): (.+?) hold the same bytes on all (\d+) ranks")
RE_KVRNO = re.compile(r"lever KVR not declared \(RADIANCE_DS41_KV_REPLICATED=1\): (.+)$")


def parse_cap(ls):
    o = {}
    for l in ls:
        if "xfer" not in o and (m := RE_XFER.search(l)):
            o["xfer"] = {"groups": int(m.group(1)), "entry": m.group(2), "entry_b": size(m.group(2)),
                         "blocks": int(m.group(3)) if m.group(3) else None, "rep": m.group(4), "rep_b": size(m.group(4))}
        if "tier" not in o and (m := RE_TIER.search(l)):
            o["tier"] = {"cap": m.group(1), "cap_b": size(m.group(1)), "entries": int(m.group(2)), "tokens": int(m.group(3)),
                         "slot": m.group(4), "slot_b": size(m.group(4))}
        if "once" not in o and (m := RE_ONCE.search(l)):
            o["once"] = {"names": [x.strip() for x in m.group(1).split(",")], "rep": m.group(2), "slot": m.group(3),
                         "slot_b": size(m.group(3)), "each": m.group(4), "each_b": size(m.group(4)), "mib": int(m.group(5)),
                         "tokens": int(m.group(6)), "tokens_each": int(m.group(7))}
        if "kvr" not in o and (m := RE_KVR.search(l)):
            o["kvr"] = {"names": [x.strip() for x in m.group(1).split(",")], "world": int(m.group(2))}
        if "kvr_no" not in o and (m := RE_KVRNO.search(l)):
            o["kvr_no"] = m.group(1)[:200]
        if "verify_off" not in o and "RADIANCE_DEBUG_TIER_VERIFY: no pinned host memory" in l:
            o["verify_off"] = l[:200]
    return o


def judge_cap(o, mode, env, ref=None):
    """-> (why list, summary string). mode A: off; B: on; V: on + RADIANCE_DEBUG_TIER_VERIFY=1."""
    why = []
    x, t, once, kvr = o.get("xfer"), o.get("tier"), o.get("once"), o.get("kvr")
    on = mode in ("B", "V")
    have = lambda w: any(e.strip() == w for e in env)
    if on and not have("RADIANCE_DS41_KV_REPLICATED=1"): why.append("RADIANCE_DS41_KV_REPLICATED=1 not in the container")
    if not on and any(e.startswith("RADIANCE_DS41_KV_REPLICATED=") for e in env): why.append("RADIANCE_DS41_KV_REPLICATED set in an A start")
    if mode == "V" and not have("RADIANCE_DEBUG_TIER_VERIFY=1"): why.append("RADIANCE_DEBUG_TIER_VERIFY=1 not in the container")
    if mode != "V" and any(e.startswith("RADIANCE_DEBUG_TIER_VERIFY=") for e in env): why.append("RADIANCE_DEBUG_TIER_VERIFY set outside V")
    if not x: why.append("no 'kv tier transfer' line")
    if not t: why.append("no 'kv tiers' capacity line")
    factor = None
    if x and t and x.get("entry_b") and t.get("slot_b"):
        factor = round(t["slot_b"] / x["entry_b"], 2)   # the slot in rank entries (4 = every rank's own window, 1 = held once)
    if not on:
        if x and x.get("rep"): why.append("a replicated window in an A start (%s)" % x["rep"])
        if once: why.append("a 'held once a slot' line in an A start")
        if kvr: why.append("a 'lever KVR declared' line in an A start")
        if factor is not None and abs(factor - 4.0) > 0.02: why.append("the slot is %s rank entries, want 4 (one a rank)" % factor)
    else:
        if not kvr: why.append("no 'lever KVR declared' line" + (" (%s)" % o["kvr_no"] if o.get("kvr_no") else ""))
        elif set(kvr["names"]) != GROUPS or kvr["world"] != 4: why.append("KVR declared %s on %s ranks" % (kvr["names"], kvr["world"]))
        if not once: why.append("no 'held once a slot' line")
        elif set(once["names"]) != GROUPS: why.append("held once: %s" % once["names"])
        if x and not x.get("rep"): why.append("the transfer line names no replicated window")
        if x and x.get("rep_b") and x.get("entry_b") and abs(x["rep_b"] - x["entry_b"]) > 0.006 * x["entry_b"]:
            why.append("replicated %s of a %s entry (want all of it)" % (x["rep"], x["entry"]))
        if factor is not None and abs(factor - 1.0) > 0.02: why.append("the slot is %s rank entries, want 1 (quartered)" % factor)
        if once and t and once["tokens"] != t["tokens"]: why.append("held-once tokens %s vs the tier line's %s" % (once["tokens"], t["tokens"]))
        if once and once.get("tokens_each"):
            r = t["tokens"] / once["tokens_each"] if t else None
            if r is None or abs(r - 4.0) > 0.01: why.append("tokens %s vs %s without the flag: x%s, want x4" % (t and t["tokens"], once["tokens_each"], r and round(r, 4)))
        rt = ((ref or {}).get("tier") or {}).get("tokens")
        if rt and t and abs(t["tokens"] / rt - 4.0) > 0.01: why.append("tokens %s vs A's %s: x%.4f, want x4" % (t["tokens"], rt, t["tokens"] / rt))
    if mode == "V" and o.get("verify_off"): why.append("the verify scratch was refused: " + o["verify_off"])
    s = "transfer entry %s%s | tier %s, %s entries, %s tokens, slot %s (%s rank entries)%s%s" % (
        x and x["entry"], (" (%s replicated)" % x["rep"]) if x and x.get("rep") else "", t and t["cap"], t and t["entries"],
        t and t["tokens"], t and t["slot"], factor,
        (" | held once: %s, %s a slot rather than %s, %s tokens rather than %s" % (", ".join(once["names"]), once["slot"], once["each"],
                                                                                  once["tokens"], once["tokens_each"])) if once else "",
        (" | KVR declared on %s ranks: %s" % (kvr["world"], ", ".join(kvr["names"]))) if kvr else "")
    return why, s


def cmd_cap():
    tag, mode = arg("--tag"), arg("--mode")
    o = parse_cap(lines(arg("--log")))
    env = [l.strip() for l in lines(arg("--env")) if l.strip()]
    ses = rdj(arg("--sessions"))
    occ = ses.get("occupancy") or {}
    o["host_cap_bytes"] = occ.get("host_cap_bytes")
    ref = rdj(arg("--ref")) if arg("--ref") else None
    why, s = judge_cap(o, mode, env, ref)
    o.update(tag=tag, mode=mode, why=why, ok=not why, summary=s, env_kvr=[e for e in env if "KV_REPLICATED" in e or "TIER_VERIFY" in e])
    if arg("--json"): json.dump(o, open(arg("--json"), "w"))
    print("CAP %s %s (mode %s) | %s | /sessions host_cap_bytes %s | env %s" % ("OK" if not why else "FAIL " + "; ".join(why), tag, mode, s,
          o["host_cap_bytes"], " ".join(o["env_kvr"]) or "(no KVR / verify word)"))
    return 0


def moved(s):
    m, st, oc = (s or {}).get("moved") or {}, (s or {}).get("stall") or {}, (s or {}).get("occupancy") or {}
    return {"to_host": m.get("to_host"), "dropped": m.get("dropped"), "promoted": m.get("promoted"), "host_drops": m.get("host_drops"),
            "failures": m.get("failures"), "restores": st.get("restores"), "restore_ms": st.get("restore_ms"),
            "host_bytes": oc.get("host_bytes"), "host_cap_bytes": oc.get("host_cap_bytes")}


def d(a, b, k):
    x, y = (a or {}).get(k), (b or {}).get(k)
    return (y - x) if isinstance(x, (int, float)) and isinstance(y, (int, float)) else None


def judge_hit(cold, hit, s0, s1, s2):
    why = []
    c0, c1, h0, h1 = cold.get("tiers_before") or {}, cold.get("tiers_after") or {}, hit.get("tiers_before") or {}, hit.get("tiers_after") or {}
    m0, m1, m2 = moved(s0), moved(s1), moved(s2)
    if cold.get("status") != 200: why.append("cold status %s" % cold.get("status"))
    if hit.get("status") != 200: why.append("hit status %s" % hit.get("status"))
    if (cold.get("cached") or 0) != 0: why.append("the cold run found %s tokens cached" % cold.get("cached"))
    if not hit.get("cached"): why.append("the hit found nothing cached")
    pr, rs = d(h0, h1, "promoted"), d(h0, h1, "restores")
    if not pr or pr < 1: why.append("no entry promoted from the host tier in the hit's window (%s)" % pr)
    if not rs or rs < 1: why.append("no restore in the hit's window (%s)" % rs)
    th, dr = d(m0, m1, "to_host"), d(m0, m1, "dropped")
    hb = d(m0, m1, "host_bytes")
    per = round(hb / th, 1) if hb is not None and th else None
    ent = (hit.get("cached") or 0) / 2.0
    o = {"cold_prompt": cold.get("prompt_tokens"), "cold_ttft": cold.get("ttft"), "cold_to_host": th, "cold_dropped": dr,
         "host_bytes_added": hb, "host_bytes_per_entry": per, "hit_cached": hit.get("cached"), "hit_ttft": hit.get("ttft"),
         "hit_promoted": pr, "hit_restores": rs, "hit_restore_ms": d(h0, h1, "restore_ms"), "hit_dropped": d(h0, h1, "dropped"),
         "promoted_share": round(pr / ent, 3) if pr and ent else None, "hit_md5": hit.get("md5"), "cold_md5": cold.get("md5"),
         "failures": d(m0, m2, "failures"), "host_drops": d(m0, m2, "host_drops")}
    if o["failures"]: why.append("tier failures +%s" % o["failures"])
    return why, o


def cmd_hit():
    cold, hit = rdj(arg("--cold")), rdj(arg("--hit"))
    why, o = judge_hit(cold, hit, rdj(arg("--s0")), rdj(arg("--s1")), rdj(arg("--s2")))
    o.update(tag=arg("--tag"), why=why, ok=not why)
    if arg("--json"): json.dump(o, open(arg("--json"), "w"))
    print("HIT %s %s | cold: prompt %s tok, TTFT %s s, %s entries copied to host, %s dropped from VRAM before the hit, host bytes +%s "
          "(%s B an entry) | hit: cached %s, TTFT %s s, promoted %s entries (%s of the cached entries) in %s restore(s), %s ms | "
          "md5 cold %s hit %s | tier failures %s, host drops %s" % (
              "OK" if not why else "FAIL " + "; ".join(why), o["tag"], o["cold_prompt"], o["cold_ttft"], o["cold_to_host"], o["cold_dropped"],
              o["host_bytes_added"], o["host_bytes_per_entry"], o["hit_cached"], o["hit_ttft"], o["hit_promoted"], o["promoted_share"],
              o["hit_restores"], o["hit_restore_ms"], o["cold_md5"], o["hit_md5"], o["failures"], o["host_drops"]))
    return 0


def judge_verify(ls):
    diff = sum(1 for l in ls if "differ between rank" in l)
    back = sum(1 for l in ls if "came back different" in l)
    off = sum(1 for l in ls if "RADIANCE_DEBUG_TIER_VERIFY: no pinned host memory" in l)
    fail = sum(1 for l in ls if "kv tiers:" in l and "failed with" in l)
    why = []
    if diff: why.append("%d 'differ between rank' line(s)" % diff)
    if back: why.append("%d 'came back different' line(s)" % back)
    if off: why.append("the verify scratch refused on %d rank(s)" % off)
    if fail: why.append("%d tier transfer failure line(s)" % fail)
    return why, {"differ": diff, "came_back": back, "off": off, "fail": fail}


def cmd_verify():
    ls = lines(arg("--log"))
    why, o = judge_verify(ls)
    print("VERIFY %s %s | %d log lines: replicated mismatch %d, round trip mismatch %d, verify off %d, tier failures %d" % (
        "OK" if not why else "FAIL " + "; ".join(why), arg("--tag"), len(ls), o["differ"], o["came_back"], o["off"], o["fail"]))
    return 0


def ci(v):
    v = [x for x in v if isinstance(x, (int, float))]
    if not v: return None, None, 0
    m = statistics.mean(v)
    return m, (T95.get(len(v) - 1, 1.96) * statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else None), len(v)


def speed(path):
    rows = []
    try:
        rows = [json.loads(l) for l in open(path) if l.strip()]
    except (OSError, ValueError):
        pass
    ok = [r for r in rows if r.get("ok")]
    g = lambda r, *k: (lambda x: x if isinstance(x, (int, float)) else None)(
        __import__("functools").reduce(lambda a, b: a.get(b) if isinstance(a, dict) else None, k, r.get("res") or {}))
    return {"blocks": len(rows), "ok": len(ok),
            "essay": ci([g(r, "c1", "essay", "decode_tps") for r in ok]), "code": ci([g(r, "c1", "code", "decode_tps") for r in ok]),
            "c8": ci([g(r, "c8", "agg_dec_tps") for r in ok]), "c8p50": ci([g(r, "c8", "per_stream_dec_p50") for r in ok])}


def f1(x, n=1): return "-" if x is None else format(x, ".%df" % n)


def cmd_report():
    o, starts = arg("--dir"), (arg("--starts") or "").split()
    out = []
    caps = {s: rdj(os.path.join(o, "cap-%s.json" % s)) for s in starts}
    for s in starts:
        c = caps[s]
        if c: out.append("capacity %s (%s): %s | %s" % (s, {"A": "KVR off", "B": "KVR on", "V": "KVR on + TIER_VERIFY"}.get(c.get("mode"), c.get("mode")),
                                                   c.get("summary"), "OK" if c.get("ok") else "VIRHE " + "; ".join(c.get("why") or [])))
    ta = (caps.get("A") or {}).get("tier") or {}
    for s in starts:
        tb = (caps.get(s) or {}).get("tier") or {}
        if s != "A" and ta.get("tokens") and tb.get("tokens"):
            out.append("capacity %s vs A: host-tier tokens %d vs %d (x%.4f), slot %s vs %s, entries %d vs %d" % (
                s, tb["tokens"], ta["tokens"], tb["tokens"] / ta["tokens"], tb.get("slot"), ta.get("slot"), tb.get("entries"), ta.get("entries")))
    for s in starts:
        h = rdj(os.path.join(o, "hit-%s.json" % s))
        if h: out.append("tier hit %s: %s | cold %s tok, %s entries to host at %s B an entry | hit cached %s, promoted %s (share %s), restores %s, %s ms, TTFT %s s (cold %s s) | md5 cold %s hit %s" % (
            s, "OK" if h.get("ok") else "VIRHE " + "; ".join(h.get("why") or []), h.get("cold_prompt"), h.get("cold_to_host"), h.get("host_bytes_per_entry"),
            h.get("hit_cached"), h.get("hit_promoted"), h.get("promoted_share"), h.get("hit_restores"), f1(h.get("hit_restore_ms")), h.get("hit_ttft"), h.get("cold_ttft"),
            h.get("cold_md5"), h.get("hit_md5")))
    sp = {s: speed(os.path.join(o, "%s-sp-blocks.jsonl" % s)) for s in starts}
    for s in starts:
        v = sp[s]
        if v["blocks"]:
            out.append("speed %s (%d of %d blocks ok, T 0, stack4 S pins): 1u essay %s +- %s, code %s +- %s tok/s | mb c8 %s +- %s tok/s (per-stream p50 %s)" % (
                s, v["ok"], v["blocks"], f1(v["essay"][0], 2), f1(v["essay"][1], 2), f1(v["code"][0], 2), f1(v["code"][1], 2),
                f1(v["c8"][0]), f1(v["c8"][1]), f1(v["c8p50"][0], 2)))
    a = sp.get("A")
    for s in starts:
        if s == "A" or not a or not a["ok"] or not sp[s]["ok"]: continue
        pc = lambda k: ("%+.1f %%" % (100 * (sp[s][k][0] / a[k][0] - 1))) if sp[s][k][0] and a[k][0] else "-"
        out.append("speed %s vs A (CROSS-START: two starts, not in-process; A's spread above is the noise floor): 1u essay %s, code %s, mb c8 %s" % (
            s, pc("essay"), pc("code"), pc("c8")))
    try:
        ident = [l.strip() for l in open(os.path.join(o, "identity.txt")) if l.strip()]
    except OSError:
        ident = []
    strict = [l for l in ident if l.startswith("S:")]
    info = [l for l in ident if l.startswith("I:")]
    bad = [l for l in strict if not ("IDENTICAL" in l.split(" ", 1)[1][:40] and "DIFFER" not in l.upper().split(" ", 1)[1][:40])]
    out.append("identity: %s%s" % (("IDENTICAL at every strict compare (%d)" % len(strict)) if strict and not bad else
                                  ("VIRHE NOT IDENTICAL: " + "; ".join(bad))[:1500] if strict else "none",
                                  (" | information: " + "; ".join(l[:160] for l in info)) if info else ""))
    try:
        gl = [l.strip() for l in open(os.path.join(o, "guard.txt")) if l.strip()]
    except OSError:
        gl = []
    gb = [l for l in gl if "VIRHE" in l or " FAIL" in l]
    out.append("guards: %s" % ("OK (%d checks: env read back, tier lines, hit from the host tier%s)" % (len(gl), ", TIER_VERIFY clean" if "V" in starts else "")
                               if gl and not gb else ("VIRHE " + " | ".join(gb))[:1500] if gb else "none recorded"))
    for l in out: print("RESULT xkvr " + l)
    return 0


def cmd_selftest():
    ls = ["I kv tier transfer: 4 paged group(s), 5.39 KiB an entry in 6 blocks, batches of 3039 through gather_rows/scatter_rows at u8",
          "I kv tiers: a finished conversation's KV is copied to host memory at once (84.00 GiB, 4084887 entries, 8169774 tokens of context) "
          "and its VRAM is given up when the expert loan or a request wants it; a full host tier drops its least recently used entries, "
          "the end of a conversation before its start. One entry is 21.56 KiB"]
    on = ["I kv tier transfer: 4 paged group(s), 5.39 KiB an entry in 6 blocks (5.39 KiB of it replicated on every rank and held once a "
          "slot), batches of 3039 through gather_rows/scatter_rows at u8",
          "I kv tiers: kv_index_r2, kv_compressed_r2, kv_index_r1, kv_compressed_r1 (5.39 KiB of each entry) is the same on every rank and "
          "held once a slot: 5.39 KiB a slot rather than 21.56 KiB, so --prefix-cache-mib 86016 holds 32679098 tokens rather than 8169774",
          "I kv tiers: a finished conversation's KV is copied to host memory at once (84.00 GiB, 16339549 entries, 32679098 tokens of "
          "context) and its VRAM is given up when the expert loan or a request wants it; a full host tier drops its least recently used "
          "entries, the end of a conversation before its start. One entry is 5.39 KiB",
          "radiance: deepseek41: lever KVR declared (RADIANCE_DS41_KV_REPLICATED=1): kv_index_r1, kv_index_r2, kv_compressed_r1, "
          "kv_compressed_r2 hold the same bytes on all 4 ranks (rad_kv_group_replicated), so a host-tier slot holds them once instead of "
          "4 times; start-time only"]
    a = parse_cap(ls)
    w, _ = judge_cap(a, "A", ["RADIANCE_DS41_PSK=0"])
    assert not w and a["tier"]["tokens"] == 8169774 and a["xfer"]["blocks"] == 6, (w, a)
    b = parse_cap(on)
    w, s = judge_cap(b, "B", ["RADIANCE_DS41_KV_REPLICATED=1"], a)
    assert not w and b["once"]["tokens"] == 32679098 and set(b["kvr"]["names"]) == GROUPS, (w, b)
    w, _ = judge_cap(b, "V", ["RADIANCE_DS41_KV_REPLICATED=1"], a)
    assert w and "RADIANCE_DEBUG_TIER_VERIFY=1 not in the container" in w[0], w
    w, _ = judge_cap(b, "A", ["RADIANCE_DS41_KV_REPLICATED=1"])
    assert len(w) >= 3, w                        # an A start carrying the flag fails several ways
    w, _ = judge_cap(a, "B", ["RADIANCE_DS41_KV_REPLICATED=1"])
    assert any("no 'lever KVR declared'" in x for x in w) and any("rank entries, want 1" in x for x in w), w
    half = [l.replace("(5.39 KiB of it replicated", "(2.70 KiB of it replicated") for l in on]
    w, _ = judge_cap(parse_cap(half), "B", ["RADIANCE_DS41_KV_REPLICATED=1"], a)
    assert any("want all of it" in x for x in w), w
    ses = lambda th, dr, pr, rs, hb: {"moved": {"to_host": th, "dropped": dr, "promoted": pr, "failures": 0, "host_drops": 0},
                                      "stall": {"restores": rs, "restore_ms": 1.0 * rs}, "occupancy": {"host_bytes": hb, "host_cap_bytes": 9}}
    tb = lambda pr, rs: {"promoted": pr, "restores": rs, "restore_ms": 0.0, "dropped": 0, "to_host": 0}
    cold = {"status": 200, "cached": 0, "prompt_tokens": 33500, "tiers_before": tb(0, 0), "tiers_after": tb(0, 0), "md5": "x"}
    hit = {"status": 200, "cached": 32768, "tiers_before": tb(0, 0), "tiers_after": tb(16384, 1), "md5": "x"}
    w, o = judge_hit(cold, hit, ses(0, 0, 0, 0, 0), ses(16800, 16700, 0, 0, 16800 * 5520), ses(16800, 16700, 16384, 1, 16800 * 5520))
    assert not w and o["host_bytes_per_entry"] == 5520.0 and o["promoted_share"] == 1.0, (w, o)
    w, _ = judge_hit(cold, dict(hit, tiers_after=tb(0, 0)), ses(0, 0, 0, 0, 0), ses(1, 0, 0, 0, 1), ses(1, 0, 0, 0, 1))
    assert len(w) == 2 and "no entry promoted" in w[0], w
    w, o = judge_verify(["E kv tiers: the replicated groups of x differ between rank 0 (1) and rank 2 (2). ...", "I fine"])
    assert w and o["differ"] == 1, w
    w, _ = judge_verify(["I kv tiers: kv_index_r2 ... held once a slot", "I fine"])
    assert not w
    print("SELFTEST OK (cap A / B / V, the half-replicated and missing-flag cases, hit OK / not from the tier, verify)")
    return 0


if __name__ == "__main__":
    cmds = {"cap": cmd_cap, "hit": cmd_hit, "verify": cmd_verify, "report": cmd_report, "selftest": cmd_selftest}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    sys.exit(cmds[sys.argv[1]]())
