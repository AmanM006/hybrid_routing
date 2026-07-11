#!/usr/bin/env python3
"""Phase 5: Run fresh sentiment stress batch with real Fireworks calls."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_final_diagnostic import parse_task_logs  # noqa: E402
from validators import validate_sentiment  # noqa: E402

BATCH = ROOT / "input" / "sentiment_fresh_batch.json"
REGRESSION = ROOT / "input" / "final_diagnostic_batch.json"
OUT = ROOT / "output" / "sentiment_fresh_report.json"

REASON_RE = re.compile(r"\b(because|since|due to|although|but)\b", re.I)


def grade_sentiment(task: dict, answer: str) -> tuple[str, list[str]]:
    """Manual-style grading: label correctness + justification keyword."""
    issues: list[str] = []
    gold = task["gold"].lower()
    al = answer.lower().strip()
    a = answer.strip()

    if not a:
        return "WRONG", ["empty answer"]

    if not validate_sentiment(task["prompt"], a):
        issues.append("failed validator")

    if len(a.split()) < 4:
        issues.append("too short (<4 words)")

    if not REASON_RE.search(al):
        issues.append("no because/but/since")

    if gold == "mixed":
        has_mixed = "mixed" in al or ("positive" in al and "negative" in al)
        if not has_mixed:
            issues.append(f"expected mixed, got {a[:80]}")
    elif gold == "positive":
        if "positive" not in al and not any(w in al for w in ("love", "great", "best", "excellent", "happy")):
            issues.append(f"expected positive, got {a[:80]}")
    elif gold == "negative":
        if "negative" not in al and not any(w in al for w in ("worst", "garbage", "hate", "awful", "bad", "poor")):
            issues.append(f"expected negative, got {a[:80]}")
    elif gold == "neutral":
        if "neutral" not in al and not any(w in al for w in ("neither", "fine", "nothing special", "factual", "no strong")):
            issues.append(f"expected neutral, got {a[:80]}")

    return ("WRONG", issues) if issues else ("CORRECT", [])


def run_batch(tasks: list[dict], label: str) -> tuple[list[dict], str]:
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
    log_path = ROOT / "output" / f"sentiment_{label}_log.txt"
    log_path.write_text(log, encoding="utf-8")

    results_path = ROOT / "output" / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    ans_map = {r["task_id"]: r.get("answer", "") for r in results}
    meta = parse_task_logs(log)

    rows = []
    for t in tasks:
        tid = t["task_id"]
        ans = ans_map.get(tid, "")
        m = meta.get(tid, {})
        verdict, issues = grade_sentiment(t, ans)
        has_reason = bool(REASON_RE.search(ans.lower()))
        rows.append({
            "task_id": tid,
            "gold": t["gold"],
            "answer": ans,
            "verdict": verdict,
            "has_reason_keyword": has_reason,
            "validator_pass": validate_sentiment(t["prompt"], ans) if ans else False,
            "tier": m.get("tier", "?"),
            "model": m.get("model", "?"),
            "issues": issues,
        })
    return rows, log


def summarize(rows: list[dict], title: str) -> dict:
    correct = sum(1 for r in rows if r["verdict"] == "CORRECT")
    total = len(rows)
    return {
        "title": title,
        "total": total,
        "correct": correct,
        "wrong": total - correct,
        "pct": round(100 * correct / total, 1) if total else 0,
        "rows": rows,
    }


def main():
    if not os.environ.get("FIREWORKS_API_KEY"):
        print("ERROR: Set FIREWORKS_API_KEY", file=sys.stderr)
        sys.exit(1)

    fresh = json.loads(BATCH.read_text(encoding="utf-8"))
    print(f"\n=== Phase 5: Sentiment fresh batch ({len(fresh)} cases) ===")
    fresh_rows, _ = run_batch(fresh, "fresh")
    fresh_report = summarize(fresh_rows, "sentiment_fresh_batch")

    regression_tasks = [
        t for t in json.loads(REGRESSION.read_text(encoding="utf-8"))
        if re.match(r"^fd-s\d+$", t["task_id"])
    ]
    print(f"\n=== Sentiment regression ({len(regression_tasks)} fd-s* cases) ===")
    reg_rows, _ = run_batch(regression_tasks, "regression")
    reg_report = summarize(reg_rows, "sentiment_regression")

    report = {"fresh": fresh_report, "regression": reg_report}
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for block in (fresh_report, reg_report):
        print(f"\n{block['title']}: {block['correct']}/{block['total']} ({block['pct']}%)")
        for r in block["rows"]:
            flag = "OK" if r["verdict"] == "CORRECT" else "FAIL"
            reason = "yes" if r["has_reason_keyword"] else "NO"
            print(f"  [{flag}] {r['task_id']} gold={r['gold']} reason={reason} tier={r['tier']}")
            print(f"       {r['answer'][:120]}")
            if r["issues"]:
                print(f"       issues: {r['issues']}")

    print(f"\nWrote {OUT}")
    all_ok = fresh_report["wrong"] == 0 and reg_report["wrong"] == 0
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
