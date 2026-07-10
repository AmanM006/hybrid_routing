#!/usr/bin/env python3
"""
token_budget_check.py -- Pre-submit token budget CI gate.

Parses TASK_LOG lines emitted by main.py and produces:
  - Per-category hit rate by tier (deterministic / local / cheap / mid / reasoning / code)
  - Average approx_tokens per task per category
  - Validation pass rate

Exit code 0 if avg_tokens < TOKEN_BUDGET_THRESHOLD, else 1.

Usage:
    python token_budget_check.py run_stdout.txt [--threshold 25]
"""
import sys
import re
import argparse
from collections import defaultdict

DEFAULT_THRESHOLD = 30

TIER_ORDER = ["deterministic", "local", "cheap-remote", "mid-remote",
              "reasoning-fallback", "direct-remote", "mid-fallback",
              "cheap-fallback", "code-fallback", "unknown"]

def parse_task_logs(logfile):
    records = []
    pattern = re.compile(
        r"TASK_LOG:\s+task_id=(\S+)\s*\|\s*category=(\S+)\s*\|\s*tier=(\S+)\s*\|"
        r"\s*model=(\S+)\s*\|\s*approx_tokens=(\d+)\s*\|\s*validation=(\S+)\s*\|"
        r"\s*latency=([\d.]+)s"
    )
    with open(logfile, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                records.append({
                    "task_id": m.group(1),
                    "category": m.group(2),
                    "tier": m.group(3),
                    "model": m.group(4),
                    "approx_tokens": int(m.group(5)),
                    "validation": m.group(6),
                    "latency": float(m.group(7)),
                })
    return records


def analyze(records, threshold):
    if not records:
        print("ERROR: No TASK_LOG lines found in log file.")
        sys.exit(1)

    total_tasks = len(records)
    total_tokens = sum(r["approx_tokens"] for r in records)
    avg_tokens = total_tokens / total_tasks
    pass_count = sum(1 for r in records if r["validation"] == "PASS")
    pass_rate = pass_count / total_tasks * 100

    print("=" * 65)
    print("  TOKEN BUDGET REPORT")
    print("=" * 65)
    print(f"  Total tasks    : {total_tasks}")
    print(f"  Total tokens   : {total_tokens}")
    print(f"  Avg tokens/task: {avg_tokens:.1f}  (threshold: {threshold})")
    print(f"  Validation PASS: {pass_count}/{total_tasks} ({pass_rate:.1f}%)")
    print()

    by_cat = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)

    print(f"  {'Category':<35} {'Tasks':>5} {'AvgTok':>7} {'Pass%':>6} {'TopTier'}")
    print("  " + "-" * 63)
    for cat in sorted(by_cat):
        recs = by_cat[cat]
        n = len(recs)
        avg_tok = sum(r["approx_tokens"] for r in recs) / n
        p_rate = sum(1 for r in recs if r["validation"] == "PASS") / n * 100
        tier_counts = defaultdict(int)
        for r in recs:
            tier_counts[r["tier"]] += 1
        top_tier = max(tier_counts, key=tier_counts.get)
        print(f"  {cat:<35} {n:>5} {avg_tok:>7.1f} {p_rate:>5.1f}% {top_tier}")

    print()
    tier_counts = defaultdict(int)
    tier_tokens = defaultdict(int)
    for r in records:
        tier_counts[r["tier"]] += 1
        tier_tokens[r["tier"]] += r["approx_tokens"]

    print(f"  {'Tier':<25} {'Tasks':>5} {'%':>6} {'AvgTok':>8}")
    print("  " + "-" * 45)
    for tier in TIER_ORDER:
        if tier in tier_counts:
            n = tier_counts[tier]
            avg_t = tier_tokens[tier] / n
            pct = n / total_tasks * 100
            print(f"  {tier:<25} {n:>5} {pct:>5.1f}% {avg_t:>8.1f}")
    for tier in tier_counts:
        if tier not in TIER_ORDER:
            n = tier_counts[tier]
            avg_t = tier_tokens[tier] / n
            pct = n / total_tasks * 100
            print(f"  {tier:<25} {n:>5} {pct:>5.1f}% {avg_t:>8.1f}")

    print()
    print("=" * 65)
    if avg_tokens > threshold:
        print(f"  BUDGET EXCEEDED: avg {avg_tokens:.1f} > threshold {threshold}")
        print("=" * 65)
        return False
    else:
        print(f"  BUDGET OK: avg {avg_tokens:.1f} <= threshold {threshold}")
        print("=" * 65)
        return True


def main():
    parser = argparse.ArgumentParser(description="Token budget CI gate")
    parser.add_argument("logfile", help="Path to log file containing TASK_LOG lines")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    records = parse_task_logs(args.logfile)
    print(f"Parsed {len(records)} TASK_LOG records from {args.logfile}\n")
    ok = analyze(records, args.threshold)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
