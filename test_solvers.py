"""
Validate deterministic solvers against known prompts and expected answers.
"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
from deterministic_solvers import solve_math_deterministically, solve_ner_deterministically
from validators import validate_ner

MATH_CASES = [
    # (prompt, expected_answer_or_None, expect_deterministic)
    ("A car travels at 90 km per hour. How many kilometers does it cover in 2.5 hours?", "225", True),
    ("What is the value of 8 factorial?", "40320", True),
    ("A rectangle has a width of 7 cm and a length of 11 cm. What is its area in square centimeters?", "77", True),
    ("A store sells apples for $0.75 each. How much do 16 apples cost in total?", "12", True),
    # Practice-02 style
    ("A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?", "144", True),
    # Extra edge cases
    ("What is 15% of 200?", "30", True),
    ("What is 25% of 80?", "20", True),
    ("What is 18 + 24?", "42", True),
    ("A triangle has a base of 6 and a height of 4. What is its area?", "12", True),
    ("If a worker earns $120 per day, how much will they earn in 5 days?", "600", True),
    # Ambiguous - should NOT try to solve
    ("Explain the quadratic formula.", None, False),
    ("Solve the differential equation dy/dx = 2x.", None, False),
    ("What is the sum of all prime numbers below 100?", None, False),
]

NER_CASES = [
    (
        "Extract all named entities and their types from: Elon Musk founded SpaceX in Hawthorne, California in 2002.",
        ["Elon Musk", "SpaceX", "Hawthorne", "California", "2002"]
    ),
    (
        "Extract all named entities and their types from: The Nobel Peace Prize was awarded to Malala Yousafzai in Oslo, Norway in 2014.",
        ["Malala Yousafzai", "Oslo", "Norway", "2014"]
    ),
    (
        "Extract all named entities and their types from: Apple Inc. released the first iPhone in January 2007 under CEO Steve Jobs.",
        ["Apple Inc.", "iPhone", "Steve Jobs", "January 2007"]
    ),
]

SHOULD_FALLTHROUGH = [
    "Who is the current president of France?",  # not a NER request
    "Summarize this text about entities.",        # not an extraction request
]

print("=" * 60)
print("MATH SOLVER TESTS")
print("=" * 60)
math_pass = 0
math_fail = 0
for prompt, expected, should_solve in MATH_CASES:
    result = solve_math_deterministically(prompt)
    if should_solve:
        if result == expected:
            print(f"  PASS | Expected={expected!r:8s} | Got={result!r:8s} | {prompt[:70]}")
            math_pass += 1
        else:
            print(f"  FAIL | Expected={expected!r:8s} | Got={result!r:8s} | {prompt[:70]}")
            math_fail += 1
    else:
        if result is None:
            print(f"  PASS (fallthrough) | {prompt[:70]}")
            math_pass += 1
        else:
            print(f"  FAIL (should NOT solve) | Got={result!r} | {prompt[:70]}")
            math_fail += 1

print(f"\nMath: {math_pass} PASS, {math_fail} FAIL")

print()
print("=" * 60)
print("NER SOLVER TESTS")
print("=" * 60)
ner_pass = 0
ner_fail = 0
for prompt, expected_entities in NER_CASES:
    result = solve_ner_deterministically(prompt)
    if result is None:
        print(f"  FAIL (returned None) | {prompt[:70]}")
        ner_fail += 1
        continue
    # Validate schema
    schema_ok = validate_ner(result)
    # Check expected entities are present
    try:
        data = json.loads(result)
        found = [e["text"] for e in data["entities"]]
        missing = [e for e in expected_entities if not any(e.lower() in f.lower() for f in found)]
        if schema_ok and not missing:
            print(f"  PASS | found={found}")
            ner_pass += 1
        else:
            print(f"  FAIL | missing={missing} | schema_ok={schema_ok} | found={found}")
            ner_fail += 1
    except Exception as ex:
        print(f"  FAIL (parse error: {ex}) | result={result!r}")
        ner_fail += 1

print()
print("NER FALLTHROUGH TESTS")
for prompt in SHOULD_FALLTHROUGH:
    result = solve_ner_deterministically(prompt)
    if result is None:
        print(f"  PASS (fallthrough) | {prompt}")
        ner_pass += 1
    else:
        print(f"  FAIL (should NOT extract) | Got={result!r}")
        ner_fail += 1

print(f"\nNER: {ner_pass} PASS, {ner_fail} FAIL")
print()
print(f"TOTAL: {math_pass + ner_pass} PASS, {math_fail + ner_fail} FAIL")
