#!/usr/bin/env python3
"""
token_budget_check.py -- Pre-submit token budget CI gate.

Parses TASK_LOG lines emitted by main.py and produces:
  - Per-category hit rate by tier (deterministic / local / cheap / mid / reasoning / code)
  - Average approx_tokens and real fw_tokens per task per category
  - Validation pass rate

Exit code 0 if avg budget tokens < TOKEN_BUDGET_THRESHOLD, else 1.
Budget gate prefers fw_tokens (real Fireworks usage) when present.

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
    # Prefer new format with fw_tokens; fall back to legacy without it.
    pattern_new = re.compile(
        r"TASK_LOG:\s+task_id=(\S+)\s*\|\s*category=(\S+)\s*\|\s*tier=(\S+)\s*\|"
        r"\s*model=(\S+)\s*\|\s*approx_tokens=(\d+)\s*\|\s*fw_tokens=(\d+)\s*\|"
        r"\s*validation=(\S+)\s*\|\s*latency=([\d.]+)s"
    )
    pattern_legacy = re.compile(
        r"TASK_LOG:\s+task_id=(\S+)\s*\|\s*category=(\S+)\s*\|\s*tier=(\S+)\s*\|"
        r"\s*model=(\S+)\s*\|\s*approx_tokens=(\d+)\s*\|\s*validation=(\S+)\s*\|"
        r"\s*latency=([\d.]+)s"
    )
    with open(logfile, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern_new.search(line)
            if m:
                records.append({
                    "task_id": m.group(1),
                    "category": m.group(2),
                    "tier": m.group(3),
                    "model": m.group(4),
                    "approx_tokens": int(m.group(5)),
                    "fw_tokens": int(m.group(6)),
                    "validation": m.group(7),
                    "latency": float(m.group(8)),
                })
                continue
            m = pattern_legacy.search(line)
            if m:
                records.append({
                    "task_id": m.group(1),
                    "category": m.group(2),
                    "tier": m.group(3),
                    "model": m.group(4),
                    "approx_tokens": int(m.group(5)),
                    "fw_tokens": None,
                    "validation": m.group(6),
                    "latency": float(m.group(7)),
                })
    return records


def _budget_tokens(r):
    """Prefer real Fireworks tokens when available."""
    if r.get("fw_tokens") is not None:
        return r["fw_tokens"]
    return r["approx_tokens"]


def analyze(records, threshold):
    if not records:
        print("ERROR: No TASK_LOG lines found in log file.")
        sys.exit(1)

    total_tasks = len(records)
    has_fw = any(r.get("fw_tokens") is not None for r in records)
    total_approx = sum(r["approx_tokens"] for r in records)
    total_fw = sum(r["fw_tokens"] or 0 for r in records if r.get("fw_tokens") is not None)
    budget_vals = [_budget_tokens(r) for r in records]
    total_budget = sum(budget_vals)
    avg_budget = total_budget / total_tasks
    pass_count = sum(1 for r in records if r["validation"] == "PASS")
    pass_rate = pass_count / total_tasks * 100

    print("=" * 72)
    print("  TOKEN BUDGET REPORT")
    print("=" * 72)
    print(f"  Total tasks       : {total_tasks}")
    print(f"  Total approx_tok  : {total_approx}")
    if has_fw:
        print(f"  Total fw_tokens   : {total_fw}  (real Fireworks prompt+completion)")
        print(f"  Avg fw_tokens/task: {total_fw / total_tasks:.1f}")
    print(f"  Budget metric     : {'fw_tokens' if has_fw else 'approx_tokens'}")
    print(f"  Avg budget/task   : {avg_budget:.1f}  (threshold: {threshold})")
    print(f"  Validation PASS   : {pass_count}/{total_tasks} ({pass_rate:.1f}%)")
    print()

    by_cat = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)

    header = f"  {'Category':<35} {'Tasks':>5} {'Approx':>7} {'FW':>7} {'Pass%':>6} {'TopTier'}"
    print(header)
    print("  " + "-" * 70)
    for cat in sorted(by_cat):
        recs = by_cat[cat]
        n = len(recs)
        avg_approx = sum(r["approx_tokens"] for r in recs) / n
        fw_recs = [r for r in recs if r.get("fw_tokens") is not None]
        avg_fw = (sum(r["fw_tokens"] for r in fw_recs) / len(fw_recs)) if fw_recs else 0.0
        p_rate = sum(1 for r in recs if r["validation"] == "PASS") / n * 100
        tier_counts = defaultdict(int)
        for r in recs:
            tier_counts[r["tier"]] += 1
        top_tier = max(tier_counts, key=tier_counts.get)
        fw_col = f"{avg_fw:>7.1f}" if fw_recs else f"{'—':>7}"
        print(f"  {cat:<35} {n:>5} {avg_approx:>7.1f} {fw_col} {p_rate:>5.1f}% {top_tier}")

    print()
    tier_counts = defaultdict(int)
    tier_approx = defaultdict(int)
    tier_fw = defaultdict(int)
    for r in records:
        tier_counts[r["tier"]] += 1
        tier_approx[r["tier"]] += r["approx_tokens"]
        if r.get("fw_tokens") is not None:
            tier_fw[r["tier"]] += r["fw_tokens"]

    print(f"  {'Tier':<25} {'Tasks':>5} {'%':>6} {'Approx':>8} {'FW':>8}")
    print("  " + "-" * 55)
    for tier in TIER_ORDER:
        if tier in tier_counts:
            n = tier_counts[tier]
            avg_a = tier_approx[tier] / n
            pct = n / total_tasks * 100
            avg_f = tier_fw[tier] / n if tier in tier_fw else 0.0
            fw_col = f"{avg_f:>8.1f}" if tier in tier_fw else f"{'—':>8}"
            print(f"  {tier:<25} {n:>5} {pct:>5.1f}% {avg_a:>8.1f} {fw_col}")
    for tier in tier_counts:
        if tier not in TIER_ORDER:
            n = tier_counts[tier]
            avg_a = tier_approx[tier] / n
            pct = n / total_tasks * 100
            avg_f = tier_fw[tier] / n if tier in tier_fw else 0.0
            fw_col = f"{avg_f:>8.1f}" if tier in tier_fw else f"{'—':>8}"
            print(f"  {tier:<25} {n:>5} {pct:>5.1f}% {avg_a:>8.1f} {fw_col}")

    print()
    print("=" * 72)
    if avg_budget > threshold:
        print(f"  BUDGET EXCEEDED: avg {avg_budget:.1f} > threshold {threshold}")
        print("=" * 72)
        return False
    else:
        print(f"  BUDGET OK: avg {avg_budget:.1f} <= threshold {threshold}")
        print("=" * 72)
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
