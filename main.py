#!/usr/bin/env python3
"""
main.py - Lead Intelligence System (stage 1: rule-based scoring).

What this stage does
--------------------
1. Loads the rubric from config.yaml.
2. Loads and validates a leads CSV (blank, NA and similar values count as missing).
3. Scores every lead 0-10 on four factors (source, size, recency, industry),
   combines them with the weights from the config and applies the thresholds.
4. Prints a table sorted by score, a score distribution and a threshold sweep,
   so you can choose sensible thresholds from the training leads.

Not built yet (later stages): batched LLM calls for unlisted industries and
reasoning, outreach messages, and the JSON report.

Usage
-----
    python main.py --input leads.csv
    python main.py --input leads.csv --csv-out scores.csv
"""

import argparse
import csv
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import yaml  # pip install pyyaml


# --------------------------------------------------------------------------
# Config and data loading
# --------------------------------------------------------------------------
def load_config(path):
    """Read config.yaml and check the weights add up to 1."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    total = sum(cfg["weights"].values())
    if abs(total - 1.0) > 1e-6:
        sys.exit(f"config.yaml: weights add up to {total}, they must add up to 1.0")
    # Pre-compute helpers used for every lead.
    cfg["_missing"] = {str(v).strip().lower() for v in cfg.get("missing_values", [""])}
    cfg["_ref_date"] = datetime.strptime(str(cfg["reference_date"]), "%Y-%m-%d").date()
    cfg["_source_scores"] = {
        k.lower(): v for k, v in cfg["source_scores"].items() if k != "missing_or_unknown"
    }
    # industry name -> best score across the groups it belongs to
    industry_map = {}
    for group in cfg["industry_groups"].values():
        for member in group["members"]:
            key = member.lower()
            industry_map[key] = max(industry_map.get(key, 0), group["score"])
    cfg["_industry_map"] = industry_map
    return cfg


def is_missing(value, cfg):
    """True if a cell is blank or a placeholder such as NA."""
    return value is None or str(value).strip().lower() in cfg["_missing"]


def load_leads(path):
    """Read the CSV into a list of dicts with stripped string values."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [
            {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            for row in reader
        ]


def parse_size(value, cfg):
    """Return (size, problem). Size is None when missing or invalid."""
    if is_missing(value, cfg):
        return None, "missing size"
    try:
        size = int(float(value.replace(",", "")))
    except ValueError:
        return None, f"invalid size '{value}'"
    if size <= 0:
        return None, f"invalid size '{value}'"
    return size, None


def parse_date(value, cfg):
    """Return (date, problem). Date is None when missing or invalid."""
    if is_missing(value, cfg):
        return None, "missing date"
    try:
        return datetime.strptime(value, "%Y-%m-%d").date(), None
    except ValueError:
        return None, f"invalid date '{value}'"


# --------------------------------------------------------------------------
# Factor scoring (each returns points out of 10, plus a flag or None)
# --------------------------------------------------------------------------
def band_score(number, bands):
    """Look up the score of the band that contains number."""
    for band in bands:
        low, high = band["min"], band["max"]
        if number >= low and (high is None or number <= high):
            return band["score"]
    return None


def score_source(source, cfg):
    if is_missing(source, cfg):
        return cfg["source_scores"]["missing_or_unknown"], "missing source"
    points = cfg["_source_scores"].get(source.lower())
    if points is None:
        return cfg["source_scores"]["missing_or_unknown"], f"unknown source '{source}'"
    return points, None


def score_size(size_text, cfg):
    size, problem = parse_size(size_text, cfg)
    if size is None:
        return cfg["size_scores"]["missing"], problem
    return band_score(size, cfg["size_scores"]["bands"]), None


def score_recency(date_text, cfg):
    """Returns (points, flag, days). days is None when unknown."""
    d, problem = parse_date(date_text, cfg)
    if d is None:
        return cfg["recency_scores"]["missing"], problem, None
    days = (cfg["_ref_date"] - d).days
    if days < 0:  # date after the reference date
        fut = cfg["recency_scores"]["future_date"]
        flag = "date after reference date" if fut.get("flag_for_review") else None
        days = fut["days_used"]
        return band_score(days, cfg["recency_scores"]["bands"]), flag, days
    return band_score(days, cfg["recency_scores"]["bands"]), None, days


def score_industry(industry, cfg):
    """Dictionary lookup. Unlisted industries get a neutral score for now;
    in a later stage the LLM will map them to the nearest group."""
    if is_missing(industry, cfg):
        return cfg["unlisted_industry"]["no_match"]["score"], "missing industry"
    points = cfg["_industry_map"].get(industry.lower())
    if points is None:
        neutral = cfg["unlisted_industry"]["no_match"]["score"]
        return neutral, f"unlisted industry '{industry}' (LLM mapping pending)"
    return points, None


# --------------------------------------------------------------------------
# Combine and decide
# --------------------------------------------------------------------------
def decide(final, flags, cfg):
    """Apply thresholds. Any flag (missing data, unlisted industry) means review."""
    dec = cfg["decision"]
    if flags and dec.get("missing_key_field_to_review", True):
        return "review"
    if final >= dec["qualified_min"]:
        return "qualified"
    if final >= dec["review_min"]:
        return "review"
    return "rejected"


def score_lead(lead, cfg):
    """Score one lead and return a result dict."""
    w = cfg["weights"]
    src, f1 = score_source(lead.get("source", ""), cfg)
    siz, f2 = score_size(lead.get("company_size", ""), cfg)
    rec, f3, days = score_recency(lead.get("last_interaction_date", ""), cfg)
    ind, f4 = score_industry(lead.get("industry", ""), cfg)
    flags = [f for f in (f1, f2, f3, f4) if f]

    final = round(
        src * w["source"] + siz * w["size"] + rec * w["recency"] + ind * w["industry"], 1
    )
    return {
        "name": lead.get("name", "") or "(no name)",
        "company": lead.get("company", ""),
        "size": lead.get("company_size", ""),
        "industry": lead.get("industry", ""),
        "source": lead.get("source", ""),
        "days": days,
        "src": src, "siz": siz, "rec": rec, "ind": ind,
        "final": final,
        "decision": decide(final, flags, cfg),
        "flags": "; ".join(flags),
    }


# --------------------------------------------------------------------------
# Printing helpers
# --------------------------------------------------------------------------
def short(text, width):
    text = str(text)
    return text if len(text) <= width else text[: width - 1] + "~"


def print_table(results):
    header = (f"{'#':>3} {'Company':<22} {'Size':>6} {'Industry':<15} {'Source':<20} "
              f"{'Days':>4} {'Src':>3} {'Siz':>3} {'Rec':>3} {'Ind':>3} "
              f"{'Score':>5} {'Decision':<9} Flags")
    print(header)
    print("-" * len(header))
    for i, r in enumerate(results, start=1):
        days = "" if r["days"] is None else r["days"]
        print(f"{i:>3} {short(r['company'], 22):<22} {short(r['size'], 6):>6} "
              f"{short(r['industry'], 15):<15} {short(r['source'], 20):<20} "
              f"{days:>4} {r['src']:>3} {r['siz']:>3} {r['rec']:>3} {r['ind']:>3} "
              f"{r['final']:>5.1f} {r['decision']:<9} {r['flags']}")


def print_distribution(results):
    """Text histogram of final scores by whole-number bucket."""
    print("\nScore distribution (whole-number buckets)")
    buckets = Counter(int(r["final"]) for r in results)
    for b in range(10, -1, -1):
        count = buckets.get(b, 0)
        if count:
            print(f"  {b:>2}-{b + 0.9:<4.1f} | {'#' * count} {count}")


def print_summary(results, cfg):
    total = len(results)
    counts = Counter(r["decision"] for r in results)
    dec = cfg["decision"]
    print(f"\nDecisions at current thresholds "
          f"(qualified >= {dec['qualified_min']}, review >= {dec['review_min']})")
    for label in ("qualified", "review", "rejected"):
        n = counts.get(label, 0)
        print(f"  {label:<10} {n:>3}  ({n / total * 100:.0f}%)")
    flagged = sum(1 for r in results if r["flags"])
    print(f"  leads with data flags: {flagged}")


def print_sweep(results):
    """Show how many leads qualify at different cut-offs."""
    total = len(results)
    print("\nThreshold sweep: leads at or above each cut-off (unflagged leads only)")
    clean = [r for r in results if not r["flags"]]
    for cut in (6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0):
        n = sum(1 for r in clean if r["final"] >= cut)
        print(f"  >= {cut:<4} {n:>3} leads ({n / total * 100:.0f}% of all)")
    print("  The company pursues about 5% of leads today (60 of 1,200).")


def write_csv(results, path):
    fields = ["name", "company", "size", "industry", "source", "days",
              "src", "siz", "rec", "ind", "final", "decision", "flags"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nScores written to {path}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Lead Intelligence System (stage 1).")
    parser.add_argument("--input", required=True, help="leads CSV file")
    parser.add_argument("--config", default="config.yaml", help="rubric config file")
    parser.add_argument("--csv-out", help="also write the scored leads to this CSV")
    args = parser.parse_args()

    for p in (args.input, args.config):
        if not Path(p).is_file():
            sys.exit(f"File not found: {p}")

    cfg = load_config(args.config)
    leads = load_leads(args.input)
    if not leads:
        sys.exit("No leads found in the input file.")

    print(f"Input: {args.input} | leads: {len(leads)} | "
          f"reference date: {cfg['_ref_date']}")

    results = []
    for i, lead in enumerate(leads, start=2):  # row 1 is the header
        try:
            results.append(score_lead(lead, cfg))
        except Exception as exc:  # never let one bad row stop the run
            print(f"Row {i} skipped: {exc}", file=sys.stderr)

    # Best score first; ties broken by the most recent interaction.
    results.sort(key=lambda r: (-r["final"], r["days"] if r["days"] is not None else 9999))

    print()
    print_table(results)
    print_distribution(results)
    print_summary(results, cfg)
    print_sweep(results)
    if args.csv_out:
        write_csv(results, args.csv_out)


if __name__ == "__main__":
    main()
