#!/usr/bin/env python3
"""Run final 32-task diagnostic batch against v22 pipeline with live Fireworks."""
import asyncio
import json
import os
import re
import sys
import subprocess
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
BATCH = ROOT / "input" / "final_diagnostic_batch.json"
OUT_DIR = ROOT / "output"
LOG_PATH = OUT_DIR / "final_diagnostic_log.txt"
REPORT_PATH = OUT_DIR / "final_diagnostic_report.json"


def grade_honest(task, answer, tier, model):
    """Brutal semantic grading — borderline = WRONG."""
    tid = task["task_id"]
    cat = task["category"]
    gold = task.get("gold", "")
    a = (answer or "").strip()
    al = a.lower()
    issues = []

    if not a or "system interrupted" in al or "temporarily unavailable" in al:
        return "WRONG", ["empty or fallback garbage"]

    if cat == "math_reasoning":
        nums = re.findall(r"-?\d+(?:\.\d+)?", a.replace(",", ""))
        # prefer Answer: line
        m = re.search(r"answer:\s*(-?\d+(?:\.\d+)?)", al)
        got = m.group(1) if m else (nums[-1] if nums else None)
        gold_num = re.findall(r"-?\d+(?:\.\d+)?", gold)
        if got and gold_num and float(got) == float(gold_num[0]):
            return "CORRECT", []
        if got == gold or a == gold:
            return "CORRECT", []
        return "WRONG", [f"expected {gold}, got {a[:100]}"]

    if cat == "named_entity_recognition":
        try:
            data = json.loads(a)
            texts = [e.get("text", "") for e in data.get("entities", [])]
        except json.JSONDecodeError:
            return "WRONG", ["not valid JSON — factual answer instead of entities"]
        tl = " ".join(texts).lower()
        missing = []
        for part in re.split(r",\s*", gold):
            if part.lower() not in tl and not any(part.lower() in t.lower() for t in texts):
                missing.append(part)
        if missing:
            return "WRONG", [f"missing entities: {missing}", f"got texts: {texts}"]
        return "CORRECT", []

    if cat == "sentiment_classification":
        if len(a.split()) < 4:
            issues.append("too short")
        if not re.search(r"\b(because|since|due to|although|but)\b", al):
            issues.append("no justification keyword")
        if "mixed" in gold.lower() and "mixed" not in al and not ("positive" in al and "negative" in al):
            if "negative" in gold.lower() and "negative" not in al:
                issues.append("missed negative/mixed")
        if "positive" in gold.lower() and "positive" not in al:
            issues.append("missed positive")
        if "neutral" in gold.lower() and "neutral" not in al:
            issues.append("missed neutral")
        return ("WRONG", issues) if issues else ("CORRECT", [])

    if cat == "factual_knowledge":
        g = gold.lower()
        if g in al or (g == "au" and " au" in f" {al}"):
            return "CORRECT", []
        if g == "domain name system" and "domain name" in al:
            return "CORRECT", []
        if g == "george orwell" and "orwell" in al:
            return "CORRECT", []
        return "WRONG", [f"expected {gold}, got {a[:80]}"]

    if cat == "summarization":
        wc = len(a.split())
        if "max 15" in task["prompt"].lower() or "15 words" in task["prompt"].lower():
            if wc > 20:
                return "WRONG", [f"too long: {wc} words"]
        if "under 20" in task["prompt"].lower() and wc > 25:
            return "WRONG", [f"too long: {wc} words"]
        if "2 sentences" in task["prompt"].lower():
            sents = [s for s in re.split(r"[.!?]+", a) if s.strip()]
            if len(sents) < 2:
                return "WRONG", ["needs 2 sentences"]
        # content check keywords from gold
        keywords = re.findall(r"[a-z]{4,}", gold.lower())
        hit = sum(1 for k in keywords if k in al)
        if hit < max(1, len(keywords) // 3):
            return "WRONG", [f"summary misses key content, hits={hit}"]
        return "CORRECT", []

    if cat == "logical_reasoning":
        if tid == "fd-l01":
            return ("CORRECT", []) if "juice" in al else ("WRONG", [f"expected juice, got {a[:60]}"])
        if tid == "fd-l02":
            if re.search(r"\bno\b", al) and "yes" not in al.split()[-3:]:
                return "CORRECT", []
            return "WRONG", ["syllogism fallacy — should be no"]
        if tid == "fd-l03":
            if "1/2" in a or "0.5" in a or "50%" in al:
                return "CORRECT", []
            return "WRONG", [f"expected 1/2 (independent trials), got {a[:60]}"]
        if tid == "fd-l04":
            if any(w in al for w in ["heat", "warm", "on for", "minutes", "toggle"]):
                return "CORRECT", []
            return "WRONG", ["classic 3-switch puzzle needs heat/on strategy"]

    if cat == "code_generation":
        if "```" not in a and "def " not in a and "function" not in al:
            return "WRONG", ["no code block"]
        if tid == "fd-c01" and "def" in al and ("1.8" in a or "* 9" in a or "*9" in a):
            return "CORRECT", []
        if tid == "fd-c02" and ("function" in al or "const" in al) and "unique" in al:
            return "CORRECT", []
        return "REVIEW", ["needs manual code inspection"]

    if cat == "code_debugging":
        if "```" not in a:
            return "WRONG", ["no code fence"]
        if tid == "fd-d01" and ("+=" in a or "s + n" in a.replace(" ", "") or "s=s+" in a.replace(" ", "")):
            return "CORRECT", []
        if tid == "fd-d02" and ("len(a)" in a or "not a" in al or "-1" in a):
            return "CORRECT", []
        return "REVIEW", ["needs manual code inspection"]

    return "REVIEW", ["unhandled grader"]


def parse_task_logs(text):
    logs = {}
    pat = re.compile(
        r"TASK_LOG:\s+task_id=(\S+)\s*\|\s*category=(\S+)\s*\|\s*tier=(\S+)\s*\|"
        r"\s*model=(\S+)\s*\|\s*approx_tokens=(\d+)"
        r"(?:\s*\|\s*fw_tokens=(\d+))?"
        r"\s*\|\s*validation=(\S+)"
    )
    for line in text.splitlines():
        m = pat.search(line)
        if m:
            logs[m.group(1)] = {
                "category": m.group(2),
                "tier": m.group(3),
                "model": m.group(4),
                "fw_tokens": int(m.group(6)) if m.group(6) else None,
                "validation": m.group(7),
            }
    return logs


def main():
    if not os.environ.get("FIREWORKS_API_KEY"):
        print("ERROR: Set FIREWORKS_API_KEY", file=sys.stderr)
        sys.exit(1)

    tasks = json.loads(BATCH.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(exist_ok=True)
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

    # Current workspace by default. Set FINAL_DIAGNOSTIC_USE_V22=1 only when
    # explicitly comparing against the historical baseline.
    cwd = ROOT
    v22 = ROOT.parent / "hybrid_routing_agent_v22"
    if os.environ.get("FINAL_DIAGNOSTIC_USE_V22") == "1" and v22.exists():
        cwd = v22
        in_path_v22 = v22 / "input" / "tasks.json"
        in_path_v22.parent.mkdir(exist_ok=True)
        in_path_v22.write_text(in_path.read_text(encoding="utf-8"), encoding="utf-8")
        out_v22 = v22 / "output"
        out_v22.mkdir(exist_ok=True)
        print(f"Running v22 baseline from {v22}")

    proc = subprocess.run(
        [sys.executable, "main.py"],
        cwd=str(cwd),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_text = proc.stdout + "\n" + proc.stderr
    LOG_PATH.write_text(log_text, encoding="utf-8")

    results_path = cwd / "output" / "results.json"
    if not results_path.exists():
        print("ERROR: no results.json", proc.returncode, file=sys.stderr)
        sys.exit(1)

    results = {r["task_id"]: r.get("answer", "") for r in json.loads(results_path.read_text(encoding="utf-8"))}
    tiers = parse_task_logs(log_text)

    rows = []
    buckets = defaultdict(lambda: {"correct": 0, "wrong": 0, "review": 0})

    print("\n" + "=" * 100)
    print(f"{'ID':8s} {'CATEGORY':28s} {'TIER':18s} {'MODEL':22s} {'VERDICT':8s}")
    print("=" * 100)

    for t in tasks:
        tid = t["task_id"]
        ans = results.get(tid, "")
        meta = tiers.get(tid, {"tier": "?", "model": "?", "category": t["category"]})
        verdict, issues = grade_honest(t, ans, meta.get("tier"), meta.get("model"))
        b = buckets[t["category"]]
        if verdict == "CORRECT":
            b["correct"] += 1
        elif verdict == "WRONG":
            b["wrong"] += 1
        else:
            b["review"] += 1

        flag = "OK" if verdict == "CORRECT" else ("??" if verdict == "REVIEW" else "FAIL")
        print(f"{tid:8s} {t['category']:28s} {meta.get('tier','?'):18s} {meta.get('model','?'):22s} {flag:8s}")
        print(f"  ANSWER: {ans[:200]!r}")
        if issues:
            print(f"  ISSUES: {issues}")
        rows.append({
            "task_id": tid,
            "category": t["category"],
            "tier": meta.get("tier"),
            "model": meta.get("model"),
            "validation_log": meta.get("validation"),
            "answer": ans,
            "verdict": verdict,
            "issues": issues,
            "gold": t.get("gold"),
        })

    print("\n" + "=" * 60)
    print("PER-CATEGORY REAL ACCURACY (borderline counted WRONG)")
    print("=" * 60)
    summary = {}
    for cat in sorted(buckets):
        b = buckets[cat]
        total = b["correct"] + b["wrong"] + b["review"]
        pct = 100 * b["correct"] / total if total else 0
        summary[cat] = {"correct": b["correct"], "wrong": b["wrong"], "review": b["review"], "pct": round(pct, 1)}
        print(f"  {cat:32s}  {b['correct']}/{total} correct  ({pct:.0f}%)  review={b['review']}")

    REPORT_PATH.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
