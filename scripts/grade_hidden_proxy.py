#!/usr/bin/env python3
"""Grade hidden_proxy_19 results with keyword/rubric checks (stricter than validators).

Usage: python scripts/grade_hidden_proxy.py output/results.json
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Each requirement is either a string (must appear) or a tuple (any-of alternatives).
RUBRICS = {
    "hp-f01": ["red", "green", "blue", "rgb", "additive"],
    "hp-f02": ["machine learning", "deep learning", "feature"],
    "hp-f03": ["ram", "rom", "memory"],
    "hp-m01": ["272"],
    "hp-m02": ["1.875", "4.5"],
    "hp-s01": ["mixed", "because"],
    "hp-s02": [("mixed", "positive", "negative"), "because"],
    "hp-sum01": ["remote", "flexib", "isol"],
    "hp-sum02": ["compost", "odor", "bin"],
    "hp-ner01": ["Sundar", "Google", "Zurich", "March"],
    "hp-ner02": ["Serena", "Wimbledon", "London", "2012"],
    "hp-l01": ["juice"],
    "hp-l02": ["1/2"],
    "hp-c01": ["def total", "return"],
    "hp-c02": ["def bsearch", "while"],
    "hp-d01": [("s +=", "s = s +", "sum(nums)")],
    "hp-d02": [("if not arr", "len(arr) == 0", "if len(arr) == 0", "if arr == []", "not arr")],
    "hp-f04": ["domain name system"],
    "hp-s03": ["mixed", "because"],
}


def _req_met(req, low: str) -> bool:
    if isinstance(req, tuple):
        return any(alt.lower() in low for alt in req)
    return req.lower() in low


def grade(task_id: str, answer: str) -> tuple[bool, list[str]]:
    keys = RUBRICS.get(task_id, [])
    if not keys:
        return bool(answer.strip()), ["no rubric"]
    low = answer.lower()
    missing = [str(k) for k in keys if not _req_met(k, low)]
    return len(missing) == 0, missing


def main() -> None:
    results_path = Path(sys.argv[1])
    tasks_path = ROOT / "input/hidden_proxy_19.json"
    tasks = {t["task_id"]: t for t in json.loads(tasks_path.read_text(encoding="utf-8"))}
    results = json.loads(results_path.read_text(encoding="utf-8"))
    passed = 0
    for r in results:
        tid = r["task_id"]
        ok, reasons = grade(tid, r.get("answer", ""))
        passed += ok
        status = "PASS" if ok else "FAIL"
        extra = "" if ok else f"  missing={reasons}"
        print(f"{tid}: {status}{extra}")
    need = 17
    print(f"\n{passed}/{len(results)} passed (need >={need} to submit)")
    sys.exit(0 if passed >= need else 1)


if __name__ == "__main__":
    main()
