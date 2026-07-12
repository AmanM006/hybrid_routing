#!/usr/bin/env python3
"""v40 regression + readiness audit — run after token cuts."""
import json
import sys
from pathlib import Path

from deterministic_solvers import (
    solve_logic_deterministically,
    solve_sentiment_deterministically,
    solve_math_deterministically,
    solve_ner_deterministically,
)
from validators import validate_category_output
from client import SYSTEM_PROMPTS, get_user_prompt_suffix, get_max_tokens

ROOT = Path(__file__).resolve().parent
FAILURES = []


def check(label, ok, detail=""):
    if not ok:
        FAILURES.append(f"{label}: {detail}")
        print(f"  FAIL  | {label} | {detail}")
    else:
        print(f"  PASS  | {label}")


print("=" * 70)
print("v40 REGRESSION AUDIT")
print("=" * 70)

# --- v39 regressions must be gone ---
ls = (
    "Three switches control one bulb in another room. You may inspect the bulb only once. "
    "What is the minimum number of switch toggles needed to identify which switch controls the bulb?"
)
check("light-switch falls through (v39 revert)", solve_logic_deterministically(ls) is None)

pos_only = (
    "Sentiment check on this Slack message: "
    "'Shipped on time, works perfectly, already recommending it to the team.'"
)
check("positive-only falls through (v39 revert)", solve_sentiment_deterministically(pos_only) is None)

# --- v38 wins must hold ---
check("constraint Sam/Jo/Lee", solve_logic_deterministically(
    "Sam, Jo, and Lee each own one of: cat, dog, bird. Sam does not own the bird. Jo owns the dog. Who owns the cat?"
) == "Sam")

mixed = (
    "What's the overall sentiment here? "
    "'Food was incredible and service was warm, but we waited 45 minutes and the table was sticky.'"
)
mixed_ans = solve_sentiment_deterministically(mixed)
check("mixed sentiment with quote fix", mixed_ans is not None)
check(
    "mixed validates",
    mixed_ans and validate_category_output("sentiment_classification", mixed, mixed_ans),
)

official = ROOT / "official_validation" / "input" / "tasks.json"
if official.exists():
    tasks = json.loads(official.read_text(encoding="utf-8"))
    for t in tasks:
        tid, p = t["task_id"], t["prompt"]
        det = None
        if "sentiment" in p.lower():
            det = solve_sentiment_deterministically(p)
        elif any(k in p.lower() for k in ("who owns", "probability", "fraction")):
            det = solve_logic_deterministically(p)
        elif re_match := ("percent" in p.lower() or "recipe" in p.lower() or "$" in p):
            det = solve_math_deterministically(p)
        if det:
            cat = "sentiment_classification" if "sentiment" in p.lower() else (
                "logical_reasoning" if "probability" in p.lower() or "who" in p.lower() else "math_reasoning"
            )
            check(f"official {tid} deterministic", validate_category_output(cat, p, det), det[:60])

print()
print("--- Prompt token budget (chars as proxy) ---")
sample_prompt = "Classify the sentiment of this review as Positive, Negative, or Neutral: 'Great product but slow shipping.'"
sys_len = len(SYSTEM_PROMPTS["sentiment_classification"])
suf_len = len(get_user_prompt_suffix("sentiment_classification", sample_prompt))
print(f"  sentiment system={sys_len} suffix={suf_len} max_tokens={get_max_tokens('sentiment_classification', sample_prompt)}")
print(f"  logic max_tokens={get_max_tokens('logical_reasoning', ls)}")
print(f"  factual explain max_tokens={get_max_tokens('factual_knowledge', 'Explain the difference between X and Y.')}")

print()
if FAILURES:
    print(f"AUDIT FAILED ({len(FAILURES)} issues)")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("AUDIT PASSED")
sys.exit(0)
