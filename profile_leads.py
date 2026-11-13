#!/usr/bin/env python3
"""
profile_leads.py - profile a leads CSV so you can design your qualification rubric.

Reads one or more lead files (name, company, company_size, industry, source,
last_interaction_date), prints a summary to the console and, optionally, writes
a markdown file you can paste into your README.

Uses only the Python standard library (no pip install needed).

Examples
--------
    python profile_leads.py leads.csv
    python profile_leads.py leads.csv --out DATA_PROFILE.md
    python profile_leads.py leads.csv leads_testing.csv leads_full_scale_100.csv --ref 2024-01-31
"""

import argparse
import csv
import statistics
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

EXPECTED_COLUMNS = [
    "name", "company", "company_size", "industry", "source", "last_interaction_date",
]

# Values that mean "missing". The data README says blanks are missing, but the
# column example shows "NA", so we treat both as missing.
MISSING_VALUES = {"", "na", "n/a", "none", "null", "nan", "unknown", "-"}

# Company size bands (label, min inclusive, max inclusive). None = no upper limit.
SIZE_BANDS = [
    ("1 (solo)", 1, 1),
    ("2-10", 2, 10),
    ("11-50", 11, 50),
    ("51-200", 51, 200),
    ("201-500", 201, 500),
    ("501-5,000", 501, 5000),
    ("5,001-20,000", 5001, 20000),
    ("20,000+", 20001, None),
]

# Recency bands in days since last interaction (label, min inclusive, max inclusive).
RECENCY_BANDS = [
    ("0-30 days", 0, 30),
    ("31-90 days", 31, 90),
    ("91-180 days", 91, 180),
    ("181-365 days", 181, 365),
    ("Over 365 days", 366, None),
]


# --------------------------------------------------------------------------
# Loading and cleaning
# --------------------------------------------------------------------------
def is_missing(value):
    """True if a cell is blank or one of the placeholder strings."""
    return value is None or value.strip().lower() in MISSING_VALUES


def load_rows(path):
    """Read a CSV into a list of dicts with stripped string values."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = [c.strip() for c in (reader.fieldnames or [])]
        rows = []
        for raw in reader:
            rows.append({(k or "").strip(): (v or "").strip() for k, v in raw.items()})
    return columns, rows


def parse_size(value):
    """Return (int size, None) or (None, problem). Problem is None if just missing."""
    if is_missing(value):
        return None, None
    try:
        size = int(float(value.replace(",", "")))
    except ValueError:
        return None, f"non-numeric size '{value}'"
    if size <= 0:
        return None, f"non-positive size '{value}'"
    return size, None


def parse_date(value):
    """Return (date, None) or (None, problem). Problem is None if just missing."""
    if is_missing(value):
        return None, None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date(), None
    except ValueError:
        return None, f"invalid date '{value}'"


def band_for(number, bands):
    """Return the label of the band containing number."""
    for label, low, high in bands:
        if number >= low and (high is None or number <= high):
            return label
    return "Out of range"


# --------------------------------------------------------------------------
# Profiling
# --------------------------------------------------------------------------
def profile(path, ref_override=None):
    """Build a profile dict for one CSV file."""
    columns, rows = load_rows(path)
    n = len(rows)

    result = {
        "file": Path(path).name,
        "rows": n,
        "columns": columns,
        "missing_columns": [c for c in EXPECTED_COLUMNS if c not in columns],
        "extra_columns": [c for c in columns if c not in EXPECTED_COLUMNS],
        "issues": [],  # (row number, problem)
    }

    # Missing values per column
    missing = {}
    for col in EXPECTED_COLUMNS:
        missing[col] = sum(1 for r in rows if is_missing(r.get(col, "")))
    result["missing"] = missing

    # Company size
    sizes = []
    for i, r in enumerate(rows, start=2):  # row 1 is the header
        size, problem = parse_size(r.get("company_size", ""))
        if problem:
            result["issues"].append((i, problem))
        if size is not None:
            sizes.append(size)
    result["size_stats"] = (
        {
            "min": min(sizes),
            "median": statistics.median(sizes),
            "max": max(sizes),
        }
        if sizes else None
    )
    size_bands = Counter(band_for(s, SIZE_BANDS) for s in sizes)
    result["size_bands"] = [(label, size_bands.get(label, 0)) for label, _, _ in SIZE_BANDS]

    # Dates and recency
    dates = []
    for i, r in enumerate(rows, start=2):
        d, problem = parse_date(r.get("last_interaction_date", ""))
        if problem:
            result["issues"].append((i, problem))
        if d is not None:
            dates.append(d)

    if dates:
        result["date_min"] = min(dates)
        result["date_max"] = max(dates)
        # Fixed reference date for recency: default is the latest date in the
        # file, so results are reproducible (never "today").
        ref = ref_override or max(dates)
    else:
        result["date_min"] = result["date_max"] = None
        ref = ref_override
    result["ref_date"] = ref

    recency_counts = Counter()
    future = 0
    if ref:
        for d in dates:
            days = (ref - d).days
            if days < 0:
                future += 1
                continue
            recency_counts[band_for(days, RECENCY_BANDS)] += 1
    result["recency_bands"] = [
        (label, recency_counts.get(label, 0)) for label, _, _ in RECENCY_BANDS
    ]
    result["future_dates"] = future

    # Categorical columns (case-insensitive grouping, original spelling kept)
    for col in ("industry", "source"):
        counter = Counter()
        spelling = {}
        for r in rows:
            v = r.get(col, "")
            if is_missing(v):
                continue
            key = v.lower()
            counter[key] += 1
            spelling.setdefault(key, v)
        result[col] = [(spelling[k], c) for k, c in counter.most_common()]

    # Source x size band (useful for spotting patterns)
    cross = Counter()
    for r in rows:
        src = r.get("source", "")
        size, _ = parse_size(r.get("company_size", ""))
        if is_missing(src) or size is None:
            continue
        cross[(src, band_for(size, SIZE_BANDS))] += 1
    result["source_by_size"] = cross

    # Duplicates (same name + company, ignoring case)
    seen = Counter()
    for r in rows:
        name = r.get("name", "").lower()
        company = r.get("company", "").lower()
        if name and company:
            seen[(name, company)] += 1
    result["duplicates"] = [k for k, c in seen.items() if c > 1]

    # Rows missing at least one key field (matches "send to review" logic)
    key_fields = ("company_size", "industry", "source", "last_interaction_date")
    result["rows_missing_key"] = sum(
        1 for r in rows if any(is_missing(r.get(f, "")) for f in key_fields)
    )
    result["rows_missing_name"] = missing.get("name", 0)
    return result


# --------------------------------------------------------------------------
# Markdown output
# --------------------------------------------------------------------------
def pct(count, total):
    return f"{(count / total * 100):.0f}%" if total else "0%"


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def to_markdown(p):
    n = p["rows"]
    out = [f"## Data profile: `{p['file']}`", ""]
    out.append(f"- **Rows:** {n}")
    if p["missing_columns"]:
        out.append(f"- **Missing expected columns:** {', '.join(p['missing_columns'])}")
    if p["extra_columns"]:
        out.append(f"- **Extra columns:** {', '.join(p['extra_columns'])}")
    if p["date_min"]:
        out.append(f"- **Date range:** {p['date_min']} to {p['date_max']}")
        out.append(
            f"- **Reference date for recency:** {p['ref_date']} "
            "(fixed, so results are reproducible)"
        )
    out.append(
        f"- **Rows missing at least one key field:** {p['rows_missing_key']} "
        f"({pct(p['rows_missing_key'], n)})"
    )
    out.append("")

    out += ["### Missing values", ""]
    out.append(table(
        ["Column", "Missing", "Share"],
        [(c, p["missing"][c], pct(p["missing"][c], n)) for c in EXPECTED_COLUMNS],
    ))
    out.append("")

    out += ["### Company size", ""]
    if p["size_stats"]:
        s = p["size_stats"]
        out.append(f"Min {s['min']:,}, median {s['median']:,.0f}, max {s['max']:,}.")
        out.append("")
    out.append(table(
        ["Size band", "Leads", "Share"],
        [(label, c, pct(c, n)) for label, c in p["size_bands"]],
    ))
    out.append("")

    out += ["### Source", ""]
    out.append(table(["Source", "Leads", "Share"],
                     [(s, c, pct(c, n)) for s, c in p["source"]]))
    out.append("")

    out += ["### Industry", ""]
    out.append(table(["Industry", "Leads", "Share"],
                     [(s, c, pct(c, n)) for s, c in p["industry"]]))
    out.append("")

    out += ["### Recency (days before the reference date)", ""]
    out.append(table(["Recency band", "Leads", "Share"],
                     [(label, c, pct(c, n)) for label, c in p["recency_bands"]]))
    if p["future_dates"]:
        out.append("")
        out.append(f"Note: {p['future_dates']} date(s) are after the reference date.")
    out.append("")

    out += ["### Data quality notes", ""]
    notes = []
    for row_num, problem in p["issues"]:
        notes.append(f"Row {row_num}: {problem}")
    if p["duplicates"]:
        notes.append(f"{len(p['duplicates'])} duplicate name and company pair(s)")
    if p["rows_missing_name"]:
        notes.append(f"{p['rows_missing_name']} row(s) have no name; messages need a fallback")
    out += [f"- {x}" for x in notes] if notes else ["- No problems found."]
    out.append("")
    return "\n".join(out)


def print_console(p):
    """Compact console summary."""
    n = p["rows"]
    print(f"\n=== {p['file']} ({n} rows) ===")
    if p["missing_columns"]:
        print("MISSING COLUMNS:", ", ".join(p["missing_columns"]))
    print("Missing values:", {c: v for c, v in p["missing"].items() if v} or "none")
    if p["size_stats"]:
        print("Size min/median/max:", p["size_stats"])
    print("Size bands:", [(l, c) for l, c in p["size_bands"] if c])
    print("Sources:", p["source"])
    print("Industries:", len(p["industry"]), "distinct; top 5:", p["industry"][:5])
    print("Dates:", p["date_min"], "to", p["date_max"], "| ref:", p["ref_date"])
    print("Recency bands:", [(l, c) for l, c in p["recency_bands"] if c])
    print("Issues:", p["issues"] or "none")
    print("Duplicates:", p["duplicates"] or "none")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Profile one or more leads CSV files.")
    parser.add_argument("files", nargs="+", help="path(s) to leads CSV file(s)")
    parser.add_argument("--ref", help="fixed reference date YYYY-MM-DD for recency "
                                      "(default: latest date in each file)")
    parser.add_argument("--out", help="write a markdown profile to this file")
    args = parser.parse_args()

    ref = None
    if args.ref:
        ref, problem = parse_date(args.ref)
        if problem:
            sys.exit(f"--ref must be YYYY-MM-DD ({problem})")

    markdown_parts = ["# Data Profile", ""]
    for path in args.files:
        if not Path(path).is_file():
            print(f"Skipping {path}: file not found", file=sys.stderr)
            continue
        try:
            p = profile(path, ref)
        except Exception as exc:  # keep going if one file is broken
            print(f"Skipping {path}: could not read ({exc})", file=sys.stderr)
            continue
        print_console(p)
        markdown_parts.append(to_markdown(p))

    if args.out:
        Path(args.out).write_text("\n".join(markdown_parts), encoding="utf-8")
        print(f"\nMarkdown profile written to {args.out}")


if __name__ == "__main__":
    main()
