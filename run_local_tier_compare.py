#!/usr/bin/env python3
"""Compare local-tier pass rate: 1.5B vs 3B on easy-category tasks."""
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_final_diagnostic import grade_honest, parse_task_logs  # noqa: E402

BATCH = [
    {"task_id": "lt-f01", "prompt": "Random trivia — what's the chemical symbol for gold?", "category": "factual_knowledge", "gold": "Au"},
    {"task_id": "lt-f02", "prompt": "Which planet is known as the Red Planet?", "category": "factual_knowledge", "gold": "Mars"},
    {"task_id": "lt-f03", "prompt": "In networking, what does DNS stand for?", "category": "factual_knowledge", "gold": "Domain Name System"},
    {"task_id": "lt-f04", "prompt": "Who wrote the novel '1984'?", "category": "factual_knowledge", "gold": "George Orwell"},
    {"task_id": "lt-s01", "prompt": "Give me a one-sentence summary (max 15 words) of: The city council approved a $2M bike lane expansion after months of public hearings and a narrow 5-4 vote.", "category": "summarization", "gold": "≤15 words, mentions bike lane approval"},
    {"task_id": "lt-s02", "prompt": "TL;DR in exactly 2 sentences: Researchers found that short naps under 30 minutes improved memory recall in adults, but longer naps caused grogginess that lasted an hour.", "category": "summarization", "gold": "2 sentences about naps/memory"},
    {"task_id": "lt-s03", "prompt": "Summarize for a busy manager (under 20 words): Our Q3 revenue rose 8% year-over-year driven by enterprise subscriptions, while consumer churn ticked up slightly.", "category": "summarization", "gold": "≤20 words, revenue + subscriptions"},
    {"task_id": "lt-s04", "prompt": "Condense this email thread gist in one line: Customer reported login failures; engineering traced it to an expired OAuth certificate; fix deployed at 14:00 UTC.", "category": "summarization", "gold": "login/OAuth cert fix deployed"},
]

MODELS = {
    "1.5b": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "3b": "qwen2.5-3b-instruct-q4_k_m.gguf",
}


def run_batch(model_file: str) -> dict:
    env = os.environ.copy()
    env["LOCAL_MODEL_FILE"] = model_file
    env["MAX_LOCAL_CONCURRENCY"] = "2"
    env.setdefault("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    env.setdefault(
        "ALLOWED_MODELS",
        "minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4",
    )
    in_path = ROOT / "input" / "tasks.json"
    in_path.write_text(
        json.dumps([{"task_id": t["task_id"], "prompt": t["prompt"]} for t in BATCH], indent=2),
        encoding="utf-8",
    )
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "main.py"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.time() - t0
    log = proc.stdout + proc.stderr
    results = json.loads((ROOT / "output" / "results.json").read_text(encoding="utf-8"))
    ans_map = {r["task_id"]: r.get("answer", "") for r in results}
    meta = parse_task_logs(log)

    local_pass = remote_pass = fail = 0
    latencies = []
    rows = []
    for t in BATCH:
        tid = t["task_id"]
        ans = ans_map.get(tid, "")
        m = meta.get(tid, {})
        tier = m.get("tier", "?")
        verdict, issues = grade_honest(t, ans, tier, m.get("model"))
        if verdict == "CORRECT":
            if tier == "local":
                local_pass += 1
            else:
                remote_pass += 1
        else:
            fail += 1
        rows.append({"task_id": tid, "tier": tier, "verdict": verdict, "issues": issues})
        for line in log.splitlines():
            if f"task_id={tid}" in line and "latency=" in line:
                try:
                    lat = float(line.split("latency=")[1].split("s")[0])
                    latencies.append(lat)
                except ValueError:
                    pass

    return {
        "model": model_file,
        "total": len(BATCH),
        "local_pass": local_pass,
        "remote_pass": remote_pass,
        "fail": fail,
        "accuracy_pct": round(100 * (local_pass + remote_pass) / len(BATCH), 1),
        "local_hit_rate": round(100 * local_pass / len(BATCH), 1),
        "elapsed_s": round(elapsed, 1),
        "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "max_latency_s": round(max(latencies), 2) if latencies else None,
        "rows": rows,
        "log_tail": log[-2000:],
    }


def main():
    if not os.environ.get("FIREWORKS_API_KEY"):
        print("ERROR: Set FIREWORKS_API_KEY", file=sys.stderr)
        sys.exit(1)

    report = {}
    for key, filename in MODELS.items():
        path = ROOT / "models" / filename
        if not path.exists():
            print(f"Skipping {key}: missing {path}")
            continue
        print(f"\n=== Running local-tier batch with {key} ({filename}) ===")
        report[key] = run_batch(filename)
        r = report[key]
        print(
            f"{key}: accuracy={r['accuracy_pct']}% local_hits={r['local_pass']}/{r['total']} "
            f"remote_fallback={r['remote_pass']} fail={r['fail']} "
            f"elapsed={r['elapsed_s']}s avg_lat={r['avg_latency_s']}s max_lat={r['max_latency_s']}s"
        )

    out = ROOT / "output" / "local_tier_compare.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
