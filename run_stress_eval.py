#!/usr/bin/env python3
"""
Stress eval: 63 solver prompts + adversarial + manual grading report.
Run: python run_stress_eval.py [--live]
  --live  runs full main.py pipeline (requires FIREWORKS_API_KEY)
"""
import argparse
import asyncio
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

from classifier import classify_prompt
from deterministic_solvers import (
    solve_math_deterministically,
    solve_ner_deterministically,
    solve_logic_deterministically,
)
from validators import validate_category_output, validate_ner

# --- 63 solver-suite cases (from test_solvers.py + test_solvers_edge.py) ---
SOLVER_CASES = [
    # test_solvers.py math (13)
    ("s-m01", "math", "A car travels at 90 km per hour. How many kilometers does it cover in 2.5 hours?", "225", True),
    ("s-m02", "math", "What is the value of 8 factorial?", "40320", True),
    ("s-m03", "math", "A rectangle has a width of 7 cm and a length of 11 cm. What is its area in square centimeters?", "77", True),
    ("s-m04", "math", "A store sells apples for $0.75 each. How much do 16 apples cost in total?", "12", True),
    ("s-m05", "math", "A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?", "144", True),
    ("s-m06", "math", "What is 15% of 200?", "30", True),
    ("s-m07", "math", "What is 25% of 80?", "20", True),
    ("s-m08", "math", "What is 18 + 24?", "42", True),
    ("s-m09", "math", "A triangle has a base of 6 and a height of 4. What is its area?", "12", True),
    ("s-m10", "math", "If a worker earns $120 per day, how much will they earn in 5 days?", "600", True),
    ("s-m11", "math", "Explain the quadratic formula.", None, False),
    ("s-m12", "math", "Solve the differential equation dy/dx = 2x.", None, False),
    ("s-m13", "math", "What is the sum of all prime numbers below 100?", None, False),
    # test_solvers.py NER (5)
    ("s-n01", "ner", "Extract all named entities and their types from: Elon Musk founded SpaceX in Hawthorne, California in 2002.", ["Elon Musk", "SpaceX", "Hawthorne", "California", "2002"], True),
    ("s-n02", "ner", "Extract all named entities and their types from: The Nobel Peace Prize was awarded to Malala Yousafzai in Oslo, Norway in 2014.", ["Malala Yousafzai", "Oslo", "Norway", "2014"], True),
    ("s-n03", "ner", "Extract all named entities and their types from: Apple Inc. released the first iPhone in January 2007 under CEO Steve Jobs.", ["Apple Inc.", "iPhone", "Steve Jobs", "January 2007"], True),
    ("s-n04", "ner", "Who is the current president of France?", None, False),
    ("s-n05", "ner", "Summarize this text about entities.", None, False),
    # edge math (15)
    ("e-m01", "math", "A price of 200 increased by 15%. What is the new price?", "230", True),
    ("e-m02", "math", "500 increased by 10%. What is the result?", "550", True),
    ("e-m03", "math", "120 decreased by 25%. What is the result?", "90", True),
    ("e-m04", "math", "80 reduced by 50%. What is the value?", "40", True),
    ("e-m05", "math", "What is the average of 10, 20, and 30?", "20", True),
    ("e-m06", "math", "Find the mean of 4, 8, 12, and 16.", "10", True),
    ("e-m07", "math", "Solve for x: 2x + 3 = 11", "4", True),
    ("e-m08", "math", "If 3x - 6 = 9, what is x?", "5", True),
    ("e-m09", "math", "5x = 25. What is x?", "5", True),
    ("e-m10", "math", "What is 0 factorial?", "1", True),
    ("e-m11", "math", "What is 0% of 500?", "0", True),
    ("e-m12", "math", "What is the average of 0 and 100?", "50", True),
    ("e-m13", "math", "Solve x^2 + 3x - 10 = 0", None, False),
    ("e-m14", "math", "Solve dy/dx = 2x + 1 with y(0) = 0", None, False),
    ("e-m15", "math", "What is the sum of all prime numbers below 50?", None, False),
    # edge NER (15)
    ("e-n01", "ner", "Extract all named entities from: Satya Nadella is the CEO of Microsoft in Redmond, Washington.", ["Satya Nadella", "Microsoft", "Redmond"], True),
    ("e-n02", "ner", "Extract all named entities and their types from: Jeff Bezos founded Amazon in Seattle in 1994.", ["Jeff Bezos", "Amazon", "Seattle", "1994"], True),
    ("e-n03", "ner", "Extract all named entities from: NASA launched the Mars Rover mission in July 2020.", ["NASA"], True),
    ("e-n04", "ner", "Extract named entities from: Marie Curie won the Nobel Prize in Physics in Paris in 1903.", ["Marie Curie", "Paris"], True),
    ("e-n05", "ner", "Extract all named entities from: Sam Altman announced GPT at OpenAI in San Francisco.", ["Sam Altman", "OpenAI"], True),
    ("e-n06", "ner", "Find all named entities in: Elon Musk co-founded Tesla Inc. in 2003 in San Carlos, California.", ["Elon Musk", "Tesla"], True),
    ("e-n07", "ner", "Who is the president of France?", None, False),
    ("e-n08", "ner", "Classify the sentiment of: Apple is a great company.", None, False),
    ("e-n09", "ner", "Summarize this text: Barack Obama was born in Hawaii.", None, False),
    ("e-n10", "ner", "Write a Python function to extract named entities.", None, False),
    ("e-n11", "ner", "Extract all named entities from: Paris is beautiful.", None, True),  # may solve or fallthrough
    ("e-n12", "ner", "Extract named entities from: Will Smith slapped Chris Rock at the Oscars in Hollywood in 2022.", ["Hollywood"], True),
    ("e-n13", "ner", "Extract named entities: Meta Corp reported earnings for Q3 in Menlo Park, California in October 2023.", ["Meta", "Menlo Park", "California"], True),
    ("e-n14", "ner", "Extract named entities: The WHO held its annual summit in Geneva in June 2024.", ["WHO", "Geneva"], True),
    ("e-n15", "ner", "Extract all named entities: The BBC broadcast the London Marathon in April 2023.", ["BBC", "London"], True),
    # edge logic (15)
    ("e-l01", "logic", "Sam, Jo, and Lee each own one of: cat, dog, bird. Sam does not own the bird. Jo owns the dog. Who owns the cat?", "Sam", True),
    ("e-l02", "logic", "Alice, Bob, and Carol each like one of: red, blue, green. Alice likes red. Bob does not like blue. Who likes green?", "Bob", True),
    ("e-l03", "logic", "Mike, Sara, and Tom each play one of: tennis, soccer, basketball. Mike plays tennis. Sara does not play soccer. Who plays basketball?", "Sara", True),
    ("e-l04", "logic", "Sam, Jo, and Lee each own one of: cat, dog, bird. Sam does not own the bird. Jo owns the dog. What does Sam own?", "Cat", True),
    ("e-l05", "logic", "Alice, Bob, Carol, and Dan each drink one of: tea, coffee, juice, water. Alice drinks tea. Bob drinks coffee. Carol does not drink juice. Who drinks juice?", "Dan", True),
    ("e-l06", "logic", "Every mammal is warm-blooded. A dolphin is a mammal. Is a dolphin warm-blooded? Answer yes or no.", None, False),
    ("e-l07", "logic", "Anna is faster than Ben. Ben is faster than Carlos. Is Anna faster than Carlos? Answer yes or no.", None, False),
    ("e-l08", "logic", "If the power goes out, the lights turn off. The lights are on. Did the power go out? Answer yes or no.", None, False),
    ("e-l09", "logic", "A bag contains 5 white marbles and 3 black marbles. What is the minimum number of marbles you must draw to guarantee at least one black marble?", None, False),
    ("e-l10", "logic", "Sam, Jo, and Lee each own one of: cat, dog, bird. Sam does not own the cat. Who owns the dog?", None, False),
    ("e-l11", "logic", "Sam, Jo, and Lee each own one of: cat, dog, bird. Sam owns the cat. Sam does not own the cat. Jo owns the dog. Who owns the bird?", None, False),
    ("e-l12", "logic", "A bag has 3 red and 2 blue balls. What is the probability of drawing red? Who gets the prize?", None, False),
    ("e-l13", "logic", "Sam, Jo, and Lee each own one of: cat, dog, bird. Sam does not own the bird. Jo owns the dog.", None, False),
    ("e-l14", "logic", "Every mammal is warm-blooded. A dolphin is a mammal. Is a dolphin warm-blooded? Answer yes or no.", None, False),
    ("e-l15", "logic", "If the power goes out, the lights turn off. The lights are on. Did the power go out? Answer yes or no.", None, False),
]

# 12 NEW adversarial stress tasks
NEW_ADVERSARIAL = [
    ("st-m01", "math_reasoning", "A store marks down an $80 jacket by 25%. What is the sale price?", "60", "markdown % — not 'decreased by' pattern"),
    ("st-m02", "math_reasoning", "What is the average of 5, 10, and 15? Index number 99.", "10", "extra index digit must not pollute average"),
    ("st-m03", "math_reasoning", "Tom has 3 apples, buys 12 more, then eats 7. How many apples does he have?", None, "multi-step — must fall through, not guess"),
    ("st-m04", "math_reasoning", "What is 50% off of $120?", "60", "'off' phrasing vs 'of'"),
    ("st-m05", "math_reasoning", "twelve minus four equals what?", None, "word numbers — no digit pattern"),
    ("st-s01", "sentiment_classification", "Classify sentiment (positive/negative/neutral): Stunning lobby and friendly staff, but mildew smell and broken shower.", "negative/mixed", "mixed review — needs label + because"),
    ("st-s02", "sentiment_classification", "Classify sentiment: The package arrived on Tuesday.", "neutral", "neutral needs justification"),
    ("st-n01", "named_entity_recognition", "Extract entities: Paris Hilton attended Paris Fashion Week in Paris, France.", "Paris Hilton=PERSON, Paris=LOCATION disambiguation", "ambiguous Paris"),
    ("st-n02", "named_entity_recognition", "Extract entities: Amazon announced profits while the Amazon river flooded Brazil.", "Amazon ORG vs river", "entity category ambiguity"),
    ("st-n03", "named_entity_recognition", "Extract entities: John Doe works at TechCorp in Tokyo.", "John Doe, TechCorp, Tokyo", "harness-style NER — TechCorp often missed"),
    ("st-c01", "code_debugging", "Bug: binary search returns wrong index when target is last element:\ndef bs(a, t):\n    lo, hi = 0, len(a)-1\n    while lo < hi:\n        mid = (lo+hi)//2\n        if a[mid] < t: lo = mid+1\n        else: hi = mid\n    return lo\nFix it.", "use lo<=hi or check a[lo]==t", "classic off-by-one"),
    ("st-c02", "code_debugging", "Bug: append to list uses same object every call:\ndef add_item(item, bucket=[]):\n    bucket.append(item)\n    return bucket\nFix mutable default.", "def add_item(item, bucket=None): bucket=bucket or []", "mutable default arg"),
    ("st-g01", "code_generation", "Write Python: return nth Fibonacci number iteratively (n>=0).", "valid iterative fib", "not trivial palindrome"),
]

CATEGORY_MAP = {"math": "math_reasoning", "ner": "named_entity_recognition", "logic": "logical_reasoning"}


def run_deterministic(task_id, kind, prompt, expected, must_solve):
    if kind == "math":
        got = solve_math_deterministically(prompt)
    elif kind == "ner":
        got = solve_ner_deterministically(prompt)
    else:
        got = solve_logic_deterministically(prompt)

    if must_solve:
        if got is None:
            return got, "MISS", "acceptable fallthrough"
        if kind == "ner":
            if not validate_ner(got):
                return got, "WRONG", "invalid NER schema"
            data = json.loads(got)
            found = [e["text"] for e in data["entities"]]
            if isinstance(expected, list):
                missing = [r for r in expected if not any(r.lower() in f.lower() for f in found)]
                if missing:
                    return got, "WRONG", f"missing entities: {missing}"
            return got, "CORRECT", "all required entities present"
        if str(got) == str(expected):
            return got, "CORRECT", "exact match"
        return got, "WRONG", f"expected {expected!r}"
    else:
        if got is None:
            return got, "CORRECT", "safe fallthrough"
        return got, "WRONG", "should not have solved"


def has_sentiment_justification(answer: str) -> bool:
    a = answer.strip()
    if len(a.split()) < 4:
        return False
    return bool(re.search(r"\b(because|since|due to|as the|given that|although|but)\b", a, re.I))


async def classify_all(tasks):
    out = {}
    for t in tasks:
        out[t["task_id"]] = await classify_prompt(t["prompt"])
    return out


def build_agent_tasks():
    tasks = []
    for tid, cat, prompt, exp, note in NEW_ADVERSARIAL:
        tasks.append({"task_id": tid, "prompt": prompt, "_expected": exp, "_note": note, "_category": cat})
    # existing adversarial
    with open("input/adversarial_tasks.json", encoding="utf-8") as f:
        for t in json.load(f):
            t2 = dict(t)
            t2["_expected"] = t.get("expected")
            t2["_category"] = t.get("category_hint")
            tasks.append(t2)
    return tasks


def grade_agent_answer(task_id, category, prompt, answer, expected, note):
    """Manual-style grading against semantic correctness."""
    a = (answer or "").strip()
    al = a.lower()
    issues = []

    if category == "math_reasoning" or task_id.startswith(("s-m", "e-m", "adv-m", "st-m")):
        if expected is None:
            # shouldn't be a confident wrong number from deterministic
            nums = re.findall(r"-?\d+(?:\.\d+)?", a)
            if nums and "fallthrough" not in str(note):
                return "UNCERTAIN", ["LLM answered number on should-fallthrough"]
            return "PASS" if not nums else "REVIEW", issues
        # extract final numeric answer
        m = re.search(r"answer:\s*(-?\d+(?:\.\d+)?)", al)
        num = m.group(1) if m else re.findall(r"-?\d+(?:\.\d+)?", a)
        num = num[-1] if isinstance(num, list) and num else num
        if num is not None and str(num) == str(expected):
            return "CORRECT", issues
        if str(a) == str(expected):
            return "CORRECT", issues
        return "WRONG", [f"expected {expected}, got {a[:120]}"]

    if category == "sentiment_classification" or "sentiment" in task_id:
        if not has_sentiment_justification(a):
            issues.append("NO_JUSTIFICATION (no because/since clause)")
        if "mixed" in str(expected).lower() and "mixed" not in al and not ("positive" in al and "negative" in al):
            if "negative" not in al and "positive" not in al:
                issues.append(f"expected mixed/negative, got label from: {a[:80]}")
        if "neutral" in str(expected).lower() and "neutral" not in al:
            issues.append("expected neutral")
        if "positive" in prompt.lower() and "excellent" in prompt.lower() and "positive" not in al:
            issues.append("missed obvious positive")
        if issues:
            return "WRONG", issues
        return "CORRECT", issues

    if category == "named_entity_recognition" or task_id.startswith(("s-n", "e-n", "adv-n", "st-n")):
        try:
            data = json.loads(a)
            texts = [e.get("text", "") for e in data.get("entities", [])]
        except json.JSONDecodeError:
            return "WRONG", ["invalid JSON"]
        if isinstance(expected, str):
            for part in re.split(r",\s*", expected):
                if part and not any(part.lower() in t.lower() for t in texts):
                    issues.append(f"missing {part}")
        if "techcorp" in prompt.lower() and not any("techcorp" in t.lower() for t in texts):
            issues.append("missing TechCorp")
        if issues:
            return "WRONG", issues
        return "CORRECT", issues

    if category in ("code_debugging", "code_generation") or task_id.startswith(("adv-d", "adv-c", "st-c", "st-g")):
        if "return none" in al or "task-specific fallback" in al:
            return "WRONG", ["emergency/fallback stub code"]
        if "def " not in a:
            return "WRONG", ["no function in output"]
        if category == "code_debugging":
            if "bucket=[]" in prompt and "bucket=[]" in a.replace(" ", ""):
                return "WRONG", ["mutable default not fixed"]
            if "bs(" in prompt.lower() or "binary" in prompt.lower():
                if "lo < hi" in a.replace(" ", "") and "lo <= hi" not in a.replace(" ", ""):
                    return "WRONG", ["binary search loop still lo<hi"]
        return "REVIEW", ["needs manual code inspection"]

    if category == "logical_reasoning" or task_id.startswith(("e-l", "adv-l")):
        if expected and str(expected).lower() in al:
            return "CORRECT", issues
        if expected and str(expected).lower() == "no" and re.search(r"\bno\b", al):
            return "CORRECT", issues
        if expected and str(expected).lower() == "yes" and re.search(r"\byes\b", al):
            return "CORRECT", issues
        return "WRONG", [f"expected {expected}, got {a[:100]}"]

    if category == "factual_knowledge":
        if "temporarily unavailable" in al:
            return "WRONG", ["emergency factual fallback"]
        return "REVIEW", []

    if category == "summarization":
        if a.lower() == prompt.lower() or (len(prompt) > 20 and a in prompt):
            return "WRONG", ["echoed prompt"]
        return "REVIEW", []

    return "REVIEW", []


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Run full agent via main.py (needs API key)")
    args = parser.parse_args()

    print("=" * 72)
    print("PART 1: 63 DETERMINISTIC SOLVER CASES")
    print("=" * 72)
    det_results = []
    wrong_det = []
    for row in SOLVER_CASES:
        tid, kind, prompt, expected, must_solve = row
        got, verdict, detail = run_deterministic(tid, kind, prompt, expected, must_solve)
        det_results.append({
            "task_id": tid, "kind": kind, "prompt": prompt,
            "expected": expected, "got": got, "verdict": verdict, "detail": detail,
        })
        mark = "✓" if verdict in ("CORRECT", "MISS") else "✗"
        print(f"{mark} {tid:8s} [{verdict:7s}] got={str(got)[:60]!r} | {detail}")
        if verdict == "WRONG":
            wrong_det.append(tid)

    print(f"\nDeterministic: {len(SOLVER_CASES) - len(wrong_det)}/{len(SOLVER_CASES)} safe, {len(wrong_det)} WRONG")
    if wrong_det:
        print("WRONG deterministic:", wrong_det)

    print("\n" + "=" * 72)
    print("PART 2: 12 NEW ADVERSARIAL + 18 EXISTING — CLASSIFIER ROUTING")
    print("=" * 72)
    agent_tasks = build_agent_tasks()
    # only new 12 first
    new_tasks = [t for t in agent_tasks if t["task_id"].startswith("st-")]
    routes = await classify_all(new_tasks)
    for t in new_tasks:
        exp_cat = t["_category"]
        got_cat = routes[t["task_id"]]
        ok = "✓" if got_cat == exp_cat or (exp_cat == "ambiguous" and got_cat) else "✗"
        print(f"{ok} {t['task_id']}: expected={exp_cat} routed={got_cat}")

    results_json = None
    if args.live and os.environ.get("FIREWORKS_API_KEY"):
        print("\n" + "=" * 72)
        print("PART 3: LIVE AGENT RUN")
        print("=" * 72)
        # combine all agent-runnable tasks
        combined = []
        for tid, kind, prompt, expected, must_solve in SOLVER_CASES:
            combined.append({
                "task_id": tid, "prompt": prompt,
                "_expected": expected, "_category": CATEGORY_MAP[kind], "_note": kind,
            })
        combined.extend(agent_tasks)
        os.makedirs("input", exist_ok=True)
        in_path = "input/tasks.json"
        with open(in_path, "w", encoding="utf-8") as f:
            json.dump([{"task_id": t["task_id"], "prompt": t["prompt"]} for t in combined], f, indent=2)
        os.environ.setdefault("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
        os.environ.setdefault("ALLOWED_MODELS", "minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4")
        import subprocess
        subprocess.run([sys.executable, "main.py"], check=True, env=os.environ.copy())
        with open("output/results.json", encoding="utf-8") as f:
            results_json = json.load(f)
    else:
        print("\n[LIVE RUN SKIPPED — set FIREWORKS_API_KEY and pass --live]")
        if os.path.exists("output/results.json"):
            with open("output/results.json", encoding="utf-8") as f:
                results_json = json.load(f)
            print("[Using existing output/results.json from prior run]")

    print("\n" + "=" * 72)
    print("PART 4: MANUAL GRADING REPORT")
    print("=" * 72)

    grades = []
    if results_json:
        ans_map = {r["task_id"]: r.get("answer", "") for r in results_json}
        task_meta = {t["task_id"]: t for t in agent_tasks}
        for tid, kind, prompt, expected, must_solve in SOLVER_CASES:
            task_meta[tid] = {
                "task_id": tid, "prompt": prompt, "_expected": expected,
                "_category": CATEGORY_MAP[kind], "_note": kind,
            }
        for tid, answer in sorted(ans_map.items(), key=lambda x: x[0]):
            meta = task_meta.get(tid, {})
            cat = meta.get("_category") or routes.get(tid, "unknown")
            verdict, issues = grade_agent_answer(
                tid, cat, meta.get("prompt", ""), answer,
                meta.get("_expected"), meta.get("_note", ""),
            )
            grades.append({"task_id": tid, "category": cat, "verdict": verdict, "issues": issues, "answer": answer})
            flag = "✗" if verdict == "WRONG" else ("?" if verdict in ("REVIEW", "UNCERTAIN") else "✓")
            print(f"{flag} {tid:12s} [{cat:28s}] {verdict:9s} | {answer[:90]!r}")
            for iss in issues:
                print(f"     → {iss}")

    # category confidence
    from collections import defaultdict
    buckets = defaultdict(lambda: {"correct": 0, "wrong": 0, "review": 0})
    for g in grades:
        b = buckets[g["category"]]
        if g["verdict"] == "CORRECT":
            b["correct"] += 1
        elif g["verdict"] == "WRONG":
            b["wrong"] += 1
        else:
            b["review"] += 1

    print("\n" + "=" * 72)
    print("PER-CATEGORY CONFIDENCE (from graded agent output)")
    print("=" * 72)
    for cat, b in sorted(buckets.items()):
        total = b["correct"] + b["wrong"] + b["review"]
        conf = b["correct"] / total if total else 0
        print(f"  {cat:32s}  correct={b['correct']} wrong={b['wrong']} review={b['review']}  → {conf:.0%} confident")

    out = {
        "deterministic_63": det_results,
        "agent_grades": grades,
        "category_confidence": dict(buckets),
    }
    os.makedirs("output", exist_ok=True)
    with open("output/stress_eval_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    if results_json:
        with open("output/stress_results_raw.json", "w", encoding="utf-8") as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"\nWrote output/stress_eval_report.json")
    if results_json:
        print(f"Wrote output/stress_results_raw.json ({len(results_json)} tasks)")


if __name__ == "__main__":
    asyncio.run(main())
