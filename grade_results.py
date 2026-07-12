"""Grade a results.json against the official 10-task criteria.

Usage: python grade_results.py <results.json>
"""
import json
import sys

from official_local_bench import grade_official

results_path = sys.argv[1]
with open("official_validation/input/tasks.json", encoding="utf-8") as f:
    tasks = {t["task_id"]: t["prompt"] for t in json.load(f)}
with open(results_path, encoding="utf-8") as f:
    results = json.load(f)

passed = 0
for r in results:
    tid = r["task_id"]
    ok, reasons = grade_official(tid, tasks[tid], r["answer"])
    passed += ok
    status = "PASS" if ok else "FAIL"
    print(f"{tid}: {status}" + ("" if ok else f"  -> {reasons}"))
print(f"\n{passed}/{len(results)} passed official criteria")
