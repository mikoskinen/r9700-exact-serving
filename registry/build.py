#!/usr/bin/env python3
"""Regenerate the registry views from registry.csv.

    python3 registry/build.py            # from the history repo root, or from registry/

Reads   registry/registry.csv   (the source of truth; add new rows there)
Writes  registry/registry.json  (same rows, as a list of objects)
        registry/deepseek.md, registry/qwen.md, registry/host-hardware.md, registry/method.md
        registry/counts.md      (status counts per line, used by the top README)

Rows are sorted by status (ADOPTED, OPEN, NEEDS-USER, PARKED, CLOSED-small, REJECTED), then by gain_pct
(largest improvement first; rows without a gain last), then by id.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "registry.csv")

COLUMNS = [
    "id", "line", "name", "what_changed", "why", "exact", "how_measured", "metric", "base_value",
    "new_value", "pct_change", "gain_pct", "status", "levers_status", "date", "branch_commit_image",
    "run_items", "source_levers_line", "source_run_summary", "needs_check", "notes",
]
STATUSES = ["ADOPTED", "OPEN", "NEEDS-USER", "PARKED", "CLOSED-small", "REJECTED"]
LINES = [
    ("DeepSeek", "deepseek.md", "DeepSeek V4.1 Flash (EXL3 2.25 bpw, TP4)"),
    ("Qwen", "qwen.md", "Qwen Flash-Next (EXL3 4.05 bpw and iq4r, TP4)"),
    ("host-hardware", "host-hardware.md", "Host and hardware (EPYC Rome + 4x R9700)"),
    ("method-tooling", "method.md", "Measurement method and tooling"),
]


def gain(row):
    try:
        return float(str(row.get("gain_pct", "")).replace("%", "").replace("+", "").strip())
    except ValueError:
        return None


def sort_key(row):
    st = row.get("status", "")
    g = gain(row)
    return (STATUSES.index(st) if st in STATUSES else len(STATUSES), 0 if g is not None else 1, -(g or 0.0), row.get("id", ""))


def cell(s, limit=None):
    s = (s or "").replace("|", "\\|").replace("\n", " ").strip()
    if limit and len(s) > limit:
        s = s[: limit - 1].rstrip() + "…"
    return s


def result(row):
    parts = []
    if row.get("metric"):
        parts.append(row["metric"] + ":")
    if row.get("base_value") or row.get("new_value"):
        parts.append("%s → %s" % (row.get("base_value") or "?", row.get("new_value") or "?"))
    if row.get("pct_change"):
        pc = row["pct_change"].strip()
        parts.append("(%s%s)" % (pc, "" if pc.endswith("%") else " %"))
    return " ".join(parts)


def main():
    with open(CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    bad = [r.get("id") for r in rows if r.get("status") not in STATUSES]
    if bad:
        print("warning: rows with an unknown status:", ", ".join(bad[:20]), file=sys.stderr)
    ids = [r["id"] for r in rows]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        print("warning: duplicate ids:", ", ".join(dup), file=sys.stderr)

    with open(os.path.join(HERE, "registry.json"), "w", encoding="utf-8") as f:
        json.dump([{c: r.get(c, "") for c in COLUMNS} for r in rows], f, indent=1, ensure_ascii=False)
        f.write("\n")

    counts_lines = ["| line | " + " | ".join(STATUSES) + " | total |", "|---" * (len(STATUSES) + 2) + "|"]
    for line, fname, title in LINES:
        sel = sorted((r for r in rows if r.get("line") == line), key=sort_key)
        c = {s: sum(1 for r in sel if r.get("status") == s) for s in STATUSES}
        counts_lines.append("| %s | %s | %d |" % (line, " | ".join(str(c[s]) for s in STATUSES), len(sel)))
        out = [
            "# %s" % title,
            "",
            "`registry/build.py` generates this file from `registry/registry.csv`, so edit the CSV instead.",
            "Sorted by status, then by gain (positive = better). Status meanings and the exactness definition: "
            "[README.md](README.md). `needs-check` = a value was missing or ambiguous in the source and was left blank.",
            "",
            "Counts: " + ", ".join("%s %d" % (s, c[s]) for s in STATUSES) + ", total %d." % len(sel),
            "",
        ]
        for st in STATUSES:
            group = [r for r in sel if r.get("status") == st]
            if not group:
                continue
            out += ["## %s (%d)" % (st, len(group)), "",
                    "| id | change | exact | measured by | result | gain % | date | source | check |",
                    "|---|---|---|---|---|---|---|---|---|"]
            for r in group:
                src = "LEVERS %s" % r["source_levers_line"] if r.get("source_levers_line", "").strip()[:1].isdigit() else cell(r.get("source_levers_line"))
                if r.get("source_run_summary"):
                    src += " · [run](../%s)" % r["source_run_summary"]
                out.append("| %s | **%s** %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    cell(r["id"]), cell(r.get("name")), cell(r.get("what_changed"), 220), cell(r.get("exact")),
                    cell(r.get("how_measured")), cell(result(r), 160), cell(r.get("gain_pct")), cell(r.get("date")),
                    src, cell(r.get("needs_check"))))
            out.append("")
        with open(os.path.join(HERE, fname), "w", encoding="utf-8") as f:
            f.write("\n".join(out))
    other = [r for r in rows if r.get("line") not in [l[0] for l in LINES]]
    if other:
        print("warning: rows with an unknown line:", ", ".join(r["id"] for r in other[:20]), file=sys.stderr)
    with open(os.path.join(HERE, "counts.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(counts_lines) + "\n")
    print("rows: %d -> registry.json, %s, counts.md" % (len(rows), ", ".join(l[1] for l in LINES)))


if __name__ == "__main__":
    main()
