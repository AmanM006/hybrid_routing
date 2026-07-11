"""
Comprehensive edge-case test suite for deterministic_solvers.py
Tests 15 new cases per category (math, NER, logic) on top of the original 18.

Spec: NEVER produce a wrong answer. Falling through (None) is always safe.
A wrong deterministic answer is a hard failure; a missed solve is acceptable.
"""
import sys, json, logging
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Silence logger noise during tests
logging.basicConfig(level=logging.CRITICAL)

sys.path.insert(0, '.')
from deterministic_solvers import (
    solve_math_deterministically,
    solve_ner_deterministically,
    solve_logic_deterministically,
)
from validators import validate_ner

PASS = 0
FAIL = 0

def check(label, got, expected, must_solve=True):
    """
    must_solve=True  → we expect a specific answer (wrong answer = FAIL)
    must_solve=False → we expect None (solver must NOT attempt this case)
    wrong_is_ok=False → any non-None answer when must_solve=False is a HARD FAIL
    """
    global PASS, FAIL
    if must_solve:
        if got == expected:
            print(f"  PASS  | {label}")
            PASS += 1
        elif got is None:
            # Fell through instead of solving — acceptable miss, not a wrong answer
            print(f"  MISS  | {label}  (fell through — acceptable)")
            PASS += 1  # count as pass because no wrong answer was returned
        else:
            print(f"  FAIL  | {label}  | Expected={expected!r} | Got={got!r}  *** WRONG ANSWER ***")
            FAIL += 1
    else:
        # must NOT solve (ambiguous/edge-case)
        if got is None:
            print(f"  PASS  (safe fallthrough) | {label}")
            PASS += 1
        else:
            print(f"  FAIL  (should NOT solve) | {label}  | Got={got!r}  *** WRONG ANSWER ***")
            FAIL += 1


# ===========================================================================
# MATH — 15 new edge cases
# ===========================================================================
print("=" * 65)
print("MATH EDGE CASES (15 new)")
print("=" * 65)

# --- New pattern: percentage increase ---
check("200 increased by 15%",
      solve_math_deterministically("A price of 200 increased by 15%. What is the new price?"),
      "230")

check("500 increased by 10%",
      solve_math_deterministically("500 increased by 10%. What is the result?"),
      "550")

# --- Percentage decrease ---
check("120 decreased by 25%",
      solve_math_deterministically("120 decreased by 25%. What is the result?"),
      "90")

check("80 reduced by 50%",
      solve_math_deterministically("80 reduced by 50%. What is the value?"),
      "40")

# --- Average ---
check("average of 10, 20, 30",
      solve_math_deterministically("What is the average of 10, 20, and 30?"),
      "20")

check("mean of 4, 8, 12, 16",
      solve_math_deterministically("Find the mean of 4, 8, 12, and 16."),
      "10")

# --- Linear equations ---
check("2x + 3 = 11",
      solve_math_deterministically("Solve for x: 2x + 3 = 11"),
      "4")

check("3x - 6 = 9",
      solve_math_deterministically("If 3x - 6 = 9, what is x?"),
      "5")

check("5x + 0 = 25 (simple form)",
      solve_math_deterministically("5x = 25. What is x?"),
      "5")

# --- Edge: zero ---
check("0 factorial",
      solve_math_deterministically("What is 0 factorial?"),
      "1")

check("0% of 500",
      solve_math_deterministically("What is 0% of 500?"),
      "0")

check("average of 0 and 100",
      solve_math_deterministically("What is the average of 0 and 100?"),
      "50")

check("average ignores Index suffix (harness noise)",
      solve_math_deterministically("What is the average of 12, 18, and 30? Index 7."),
      "20")

check("average ignores Index number suffix",
      solve_math_deterministically("What is the average of 5, 10, and 15? Index number 99."),
      "10")

# --- Should NOT solve (ambiguous / complex) ---
check("quadratic x^2 + 3x = 10 (ambiguous)",
      solve_math_deterministically("Solve x^2 + 3x - 10 = 0"),
      None, must_solve=False)

check("differential equation (fall through)",
      solve_math_deterministically("Solve dy/dx = 2x + 1 with y(0) = 0"),
      None, must_solve=False)

check("sum of primes (fall through)",
      solve_math_deterministically("What is the sum of all prime numbers below 50?"),
      None, must_solve=False)


# ===========================================================================
# NER — 15 new edge cases
# ===========================================================================
print()
print("=" * 65)
print("NER EDGE CASES (15 new)")
print("=" * 65)

def check_ner(label, prompt, must_solve=True, required_texts=None):
    global PASS, FAIL
    result = solve_ner_deterministically(prompt)
    if must_solve:
        if result is None:
            print(f"  MISS  | {label}  (fell through — acceptable)")
            PASS += 1
            return
        if not validate_ner(result):
            print(f"  FAIL  | {label}  | Invalid schema: {result!r}  *** SCHEMA ERROR ***")
            FAIL += 1
            return
        data = json.loads(result)
        found = [e["text"].lower() for e in data["entities"]]
        if required_texts:
            missing = [r for r in required_texts if not any(r.lower() in f for f in found)]
            if missing:
                print(f"  FAIL  | {label}  | Missing: {missing}  | Found: {found}")
                FAIL += 1
                return
        print(f"  PASS  | {label}  | entities={[e['text'] for e in data['entities']]}")
        PASS += 1
    else:
        if result is None:
            print(f"  PASS  (safe fallthrough) | {label}")
            PASS += 1
        else:
            print(f"  FAIL  (should NOT extract) | {label}  | Got={result!r}  *** WRONG ***")
            FAIL += 1

# Extractable cases
check_ner(
    "Microsoft CEO Satya Nadella in Redmond",
    "Extract all named entities from: Satya Nadella is the CEO of Microsoft in Redmond, Washington.",
    required_texts=["Satya Nadella", "Microsoft", "Redmond"]
)

check_ner(
    "Amazon and Jeff Bezos in Seattle 1994",
    "Extract all named entities and their types from: Jeff Bezos founded Amazon in Seattle in 1994.",
    required_texts=["Jeff Bezos", "Amazon", "Seattle", "1994"]
)

check_ner(
    "NASA Mars Rover mission",
    "Extract all named entities from: NASA launched the Mars Rover mission in July 2020.",
    required_texts=["NASA"]
)

check_ner(
    "Nobel Prize Marie Curie Paris",
    "Extract named entities from: Marie Curie won the Nobel Prize in Physics in Paris in 1903.",
    required_texts=["Marie Curie", "Paris"]
)

check_ner(
    "OpenAI GPT-4 Sam Altman",
    "Extract all named entities from: Sam Altman announced GPT at OpenAI in San Francisco.",
    required_texts=["Sam Altman", "OpenAI"]
)

# Schema safety: at least 2 entities, valid JSON
check_ner(
    "Tesla Elon Musk 2003",
    "Find all named entities in: Elon Musk co-founded Tesla Inc. in 2003 in San Carlos, California.",
    required_texts=["Elon Musk", "Tesla"]
)

# Fallthrough cases (should NOT extract — not extraction prompts)
check_ner(
    "NOT an extraction prompt (question)",
    "Who is the president of France?",
    must_solve=False
)

check_ner(
    "NOT an extraction prompt (sentiment)",
    "Classify the sentiment of: Apple is a great company.",
    must_solve=False
)

check_ner(
    "NOT an extraction prompt (summarize)",
    "Summarize this text: Barack Obama was born in Hawaii.",
    must_solve=False
)

check_ner(
    "NOT an extraction prompt (code)",
    "Write a Python function to extract named entities.",
    must_solve=False
)

# Edge: only one entity mentioned → should fall through (< 2 confident entities)
check_ner(
    "Only one entity (should fall through or return >= 2)",
    "Extract all named entities from: Paris is beautiful.",
    must_solve=True  # Could return Paris as location — it's fine, or fall through
)

# Edge: entity names that look like common words → safe handling
check_ner(
    "Ambiguous names (Will Smith in Hollywood 2022)",
    "Extract named entities from: Will Smith slapped Chris Rock at the Oscars in Hollywood in 2022.",
    must_solve=True,
    required_texts=["Hollywood"]
)

# Valid extraction with org suffix
check_ner(
    "Meta Corp quarterly earnings",
    "Extract named entities: Meta Corp reported earnings for Q3 in Menlo Park, California in October 2023.",
    required_texts=["Meta", "Menlo Park", "California"]
)

check_ner(
    "WHO Geneva 2024",
    "Extract named entities: The WHO held its annual summit in Geneva in June 2024.",
    required_texts=["WHO", "Geneva"]
)

check_ner(
    "BBC London coverage",
    "Extract all named entities: The BBC broadcast the London Marathon in April 2023.",
    required_texts=["BBC", "London"]
)


# ===========================================================================
# LOGIC — 15 new edge cases
# ===========================================================================
print()
print("=" * 65)
print("LOGIC EDGE CASES (15 new)")
print("=" * 65)

def check_logic(label, prompt, expected, must_solve=True):
    global PASS, FAIL
    got = solve_logic_deterministically(prompt)
    if must_solve:
        if got is None:
            print(f"  MISS  | {label}  (fell through — acceptable)")
            PASS += 1
        elif got.lower() == expected.lower():
            print(f"  PASS  | {label}  | Got={got!r}")
            PASS += 1
        else:
            print(f"  FAIL  | {label}  | Expected={expected!r} | Got={got!r}  *** WRONG ***")
            FAIL += 1
    else:
        if got is None:
            print(f"  PASS  (safe fallthrough) | {label}")
            PASS += 1
        else:
            print(f"  FAIL  (should NOT solve) | {label}  | Got={got!r}  *** WRONG ***")
            FAIL += 1

# --- Solvable constraint puzzles ---
check_logic(
    "3-person pets: Jo owns dog, Sam not bird → Sam owns cat",
    "Sam, Jo, and Lee each own one of: cat, dog, bird. Sam does not own the bird. Jo owns the dog. Who owns the cat?",
    "Sam"
)

check_logic(
    "3-person colors: Alice red, Bob not blue → Bob green",
    "Alice, Bob, and Carol each like one of: red, blue, green. Alice likes red. Bob does not like blue. Who likes green?",
    "Bob"
)

check_logic(
    "3-person sports: Mike plays tennis, Sara not soccer → Sara plays basketball",
    "Mike, Sara, and Tom each play one of: tennis, soccer, basketball. Mike plays tennis. Sara does not play soccer. Who plays basketball?",
    "Sara"
)

check_logic(
    "Query: What does Sam own?",
    "Sam, Jo, and Lee each own one of: cat, dog, bird. Sam does not own the bird. Jo owns the dog. What does Sam own?",
    "Cat"
)

check_logic(
    "4-person drinks (tighter constraints)",
    "Alice, Bob, Carol, and Dan each drink one of: tea, coffee, juice, water. Alice drinks tea. Bob drinks coffee. Carol does not drink juice. Who drinks juice?",
    "Dan"
)

# --- Syllogisms (deterministic when pattern is unambiguous) ---
check_logic(
    "Syllogism: all mammals warm-blooded",
    "Every mammal is warm-blooded. A dolphin is a mammal. Is a dolphin warm-blooded? Answer yes or no.",
    "Yes",
)

check_logic(
    "Comparative chain: Anna faster than Carlos",
    "Anna is faster than Ben. Ben is faster than Carlos. Is Anna faster than Carlos? Answer yes or no.",
    "Yes",
)

check_logic(
    "Modus tollens: power out lights off",
    "If the power goes out, the lights turn off. The lights are on. Did the power go out? Answer yes or no.",
    "No",
)

check_logic(
    "Pigeonhole: minimum marbles",
    "A bag contains 5 white marbles and 3 black marbles. What is the minimum number of marbles you must draw to guarantee at least one black marble?",
    "6",
)

# --- Must fall through: no unique solution (underdetermined) ---
check_logic(
    "Underdetermined: 2 constraints but 2 free → multiple solutions (fall through)",
    "Sam, Jo, and Lee each own one of: cat, dog, bird. Sam does not own the cat. Who owns the dog?",
    None, must_solve=False  # 2 possible solutions remain
)

# --- Must fall through: contradictory constraints ---
check_logic(
    "Contradictory: Sam owns cat AND Sam does not own cat (fall through)",
    "Sam, Jo, and Lee each own one of: cat, dog, bird. Sam owns the cat. Sam does not own the cat. Jo owns the dog. Who owns the bird?",
    None, must_solve=False  # contradiction → 0 solutions → fall through
)

# --- Must fall through: probability question in disguise ---
check_logic(
    "Probability question mentioning 'who' (fall through)",
    "A bag has 3 red and 2 blue balls. What is the probability of drawing red? Who gets the prize?",
    None, must_solve=False
)

# --- Must fall through: no 'who' query ---
check_logic(
    "No 'who' query (fall through)",
    "Sam, Jo, and Lee each own one of: cat, dog, bird. Sam does not own the bird. Jo owns the dog.",
    None, must_solve=False
)

# --- Original 4 logic tasks (now deterministic in v37) ---
check_logic(
    "l-new-01: dolphin syllogism",
    "Every mammal is warm-blooded. A dolphin is a mammal. Is a dolphin warm-blooded? Answer yes or no.",
    "Yes",
)

check_logic(
    "l-new-04: modus tollens lights",
    "If the power goes out, the lights turn off. The lights are on. Did the power go out? Answer yes or no.",
    "No",
)


# ===========================================================================
# SUMMARY
# ===========================================================================
print()
print("=" * 65)
total = PASS + FAIL
print(f"TOTAL: {PASS} PASS / {FAIL} FAIL / {total} TOTAL")
if FAIL == 0:
    print("ALL SAFE: No wrong deterministic answers produced.")
else:
    print(f"*** {FAIL} WRONG ANSWERS — MUST FIX BEFORE SHIPPING ***")
