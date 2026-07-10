#!/usr/bin/env python3
"""Phase 2: run 32-task diagnostic and report actual Fireworks token usage."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BATCH = ROOT / "input" / "final_diagnostic_batch.json"
LOG_PATH = ROOT / "output" / "token_audit_log.txt"
REPORT_PATH = ROOT / "output" / "token_audit_report.json"

TOP3_BENCHMARKS = [
    {"label": "Top-3 #1", "accuracy_pct": 94.7, "total_tokens": 1377, "tasks": 19},
    {"label": "Top-3 #2", "accuracy_pct": 92.1, "total_tokens": 1797, "tasks": 19},
    {"label": "Top-3 #3", "accuracy_pct": 89.5, "total_tokens": 3180, "tasks": 19},
]


def parse_token_usage(log_text: str) -> list[dict]:
    rows = []
    pat = re.compile(
        r"TOKEN_USAGE:\s+task_id=(\S+)\s*\|\s*category=(\S+)\s*\|\s*model=(\S+)\s*\|"
        r"\s*prompt_tokens=(\d+)\s*\|\s*completion_tokens=(\d+)\s*\|\s*total_tokens=(\d+)"
    )
    for line in log_text.splitlines():
        m = pat.search(line)
        if m:
            rows.append({
                "task_id": m.group(1),
                "category": m.group(2),
                "model": m.group(3),
                "prompt_tokens": int(m.group(4)),
                "completion_tokens": int(m.group(5)),
                "total_tokens": int(m.group(6)),
            })
    return rows


def parse_audit_summary(log_text: str) -> dict | None:
    for line in log_text.splitlines():
        if line.startswith("TOKEN_AUDIT_SUMMARY:"):
            return json.loads(line.split(":", 1)[1].strip())
    return None


def build_report(usage_rows: list[dict], summary: dict | None, n_tasks: int) -> dict:
    exclude = {"__healthcheck__"}
    rows = [r for r in usage_rows if r["task_id"] not in exclude]
    if summary is None:
        by_cat = {}
        by_task = {}
        total = prompt = completion = 0
        for r in rows:
            total += r["total_tokens"]
            prompt += r["prompt_tokens"]
            completion += r["completion_tokens"]
            by_task[r["task_id"]] = by_task.get(r["task_id"], 0) + r["total_tokens"]
            c = r["category"]
            if c not in by_cat:
                by_cat[c] = {"calls": 0, "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
            by_cat[c]["calls"] += 1
            by_cat[c]["total_tokens"] += r["total_tokens"]
            by_cat[c]["prompt_tokens"] += r["prompt_tokens"]
            by_cat[c]["completion_tokens"] += r["completion_tokens"]
        summary = {
            "fireworks_calls": len(rows),
            "tasks_with_fireworks_calls": len(by_task),
            "total_prompt_tokens": prompt,
            "total_completion_tokens": completion,
            "total_tokens": total,
            "avg_tokens_per_call": round(total / len(rows), 1) if rows else 0,
            "avg_tokens_per_task": round(total / len(by_task), 1) if by_task else 0,
            "by_category": by_cat,
            "by_task": by_task,
        }

    per_cat_avg = {}
    for cat, data in summary.get("by_category", {}).items():
        per_cat_avg[cat] = {
            **data,
            "avg_tokens_per_call": round(data["total_tokens"] / data["calls"], 1) if data["calls"] else 0,
        }

    total_tokens = summary["total_tokens"]
    avg_per_batch_task = round(total_tokens / n_tasks, 1) if n_tasks else 0
    avg_per_fireworks_task = summary["avg_tokens_per_task"]

    benchmark_compare = []
    for b in TOP3_BENCHMARKS:
        bench_avg = round(b["total_tokens"] / b["tasks"], 1)
        benchmark_compare.append({
            **b,
            "avg_tokens_per_task": bench_avg,
            "our_total_vs_theirs": round(total_tokens - b["total_tokens"]),
            "our_avg_vs_theirs": round(avg_per_batch_task - bench_avg, 1),
        })

    return {
        "batch_tasks": n_tasks,
        "summary": summary,
        "per_category": per_cat_avg,
        "avg_tokens_per_batch_task": avg_per_batch_task,
        "avg_tokens_per_fireworks_task": avg_per_fireworks_task,
        "benchmark_compare": benchmark_compare,
        "usage_rows": rows,
    }


def print_report(report: dict) -> None:
    s = report["summary"]
    print("\n" + "=" * 70)
    print("PHASE 2 — REAL FIREWORKS TOKEN AUDIT (32-task diagnostic batch)")
    print("=" * 70)
    print(f"  Batch tasks                         : {report['batch_tasks']}")
    print(f"  Fireworks API calls (excl. health)  : {s['fireworks_calls']}")
    print(f"  Tasks that used Fireworks           : {s['tasks_with_fireworks_calls']}")
    print(f"  Total prompt tokens                 : {s['total_prompt_tokens']}")
    print(f"  Total completion tokens             : {s['total_completion_tokens']}")
    print(f"  TOTAL FIREWORKS TOKENS              : {s['total_tokens']}")
    print(f"  Avg tokens / batch task (all 32)    : {report['avg_tokens_per_batch_task']}")
    print(f"  Avg tokens / Fireworks task         : {report['avg_tokens_per_fireworks_task']}")
    print()
    print(f"  {'Category':<32} {'Calls':>5} {'TotalTok':>9} {'Avg/Call':>9} {'Prompt':>8} {'Compl':>8}")
    print("  " + "-" * 68)
    for cat in sorted(report["per_category"]):
        d = report["per_category"][cat]
        print(
            f"  {cat:<32} {d['calls']:>5} {d['total_tokens']:>9} "
            f"{d['avg_tokens_per_call']:>9.1f} {d['prompt_tokens']:>8} {d['completion_tokens']:>8}"
        )
    print()
    print("  Top-3 benchmark comparison (leaderboard uses ~19 tasks; we ran 32):")
    print(f"  {'Benchmark':<12} {'Their total':>11} {'Their avg/t':>10} {'Our avg/32':>11} {'Delta avg':>10}")
    print("  " + "-" * 58)
    for b in report["benchmark_compare"]:
        print(
            f"  {b['label']:<12} {b['total_tokens']:>11} {b['avg_tokens_per_task']:>10.1f} "
            f"{report['avg_tokens_per_batch_task']:>11.1f} {b['our_avg_vs_theirs']:>+10.1f}"
        )
    print("=" * 70)


def main():
    if not os.environ.get("FIREWORKS_API_KEY"):
        print("ERROR: Set FIREWORKS_API_KEY", file=sys.stderr)
        sys.exit(1)

    tasks = json.loads(BATCH.read_text(encoding="utf-8"))
    in_path = ROOT / "input" / "tasks.json"
    in_path.write_text(
        json.dumps([{"task_id": t["task_id"], "prompt": t["prompt"]} for t in tasks], indent=2),
        encoding="utf-8",
    )
    os.environ.setdefault("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    os.environ.setdefault(
        "ALLOWED_MODELS",
        "minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4",
    )

    proc = subprocess.run(
        [sys.executable, "main.py"],
        cwd=str(ROOT),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log = proc.stdout + proc.stderr
    LOG_PATH.write_text(log, encoding="utf-8")

    usage_rows = parse_token_usage(log)
    summary = parse_audit_summary(log)
    report = build_report(usage_rows, summary, len(tasks))
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_report(report)
    print(f"\nWrote {LOG_PATH}")
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
