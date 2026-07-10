#!/usr/bin/env python3
"""Run fresh NER-only batch (Phase 1 verification)."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_final_diagnostic import grade_honest, parse_task_logs  # noqa: E402

BATCH = ROOT / "input" / "ner_fresh_batch.json"
OUT = ROOT / "output" / "ner_fresh_report.json"


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
    (ROOT / "output" / "ner_fresh_log.txt").write_text(log, encoding="utf-8")

    results_path = ROOT / "output" / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    ans_map = {r["task_id"]: r.get("answer", "") for r in results}
    meta = parse_task_logs(log)

    rows = []
    correct = wrong = 0
    det = remote = 0
    for t in tasks:
        tid = t["task_id"]
        ans = ans_map.get(tid, "")
        m = meta.get(tid, {})
        verdict, issues = grade_honest(t, ans, m.get("tier"), m.get("model"))
        if verdict == "CORRECT":
            correct += 1
        else:
            wrong += 1
        tier = m.get("tier", "?")
        if tier == "deterministic":
            det += 1
        else:
            remote += 1
        rows.append({"task_id": tid, "verdict": verdict, "tier": tier, "issues": issues})

    report = {
        "total": len(tasks),
        "correct": correct,
        "wrong": wrong,
        "pct": round(100 * correct / len(tasks), 1),
        "deterministic_hits": det,
        "remote_hits": remote,
        "rows": rows,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"NER fresh batch: {correct}/{len(tasks)} ({report['pct']}%) | det={det} remote={remote}")
    for r in rows:
        flag = "OK" if r["verdict"] == "CORRECT" else "FAIL"
        print(f"  [{flag}] {r['task_id']} tier={r['tier']} {r['issues']}")
    return 0 if wrong == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
