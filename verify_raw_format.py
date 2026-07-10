"""
Raw format verification script.
Checks the EXACT string the solver returns and whether our own validator accepts it.
This is the honest self-audit — do the formats match?
"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
from deterministic_solvers import solve_math_deterministically, solve_ner_deterministically, solve_logic_deterministically
from validators import validate_category_output

# ---- MATH ---------------------------------------------------------------
math_cases = [
    ("A car travels at 90 km per hour. How many kilometers does it cover in 2.5 hours?", "225"),
    ("What is the value of 8 factorial?", "40320"),
    ("A store sells apples for $0.75 each. How much do 16 apples cost in total?", "12"),
    ("A rectangle has a width of 7 cm and a length of 11 cm. What is its area in square centimeters?", "77"),
    ("200 increased by 15%. What is the new price?", "230"),
    ("Solve for x: 2x + 3 = 11", "4"),
    ("What is the average of 10, 20, and 30?", "20"),
]

print("=" * 70)
print("MATH — RAW ANSWERS + VALIDATOR ACCEPTANCE")
print("=" * 70)
for prompt, expected in math_cases:
    raw = solve_math_deterministically(prompt)
    valid = validate_category_output("math_reasoning", prompt, raw) if raw else False
    correct = (raw == expected)
    flag = "✓" if (raw is not None and correct and valid) else ("MISS" if raw is None else "✗ MISMATCH")
    print(f"  {flag}")
    print(f"    Prompt:   {prompt[:75]}")
    print(f"    raw=      {raw!r:15s}  expected={expected!r}")
    print(f"    validator={valid}")
    print()

# ---- NER ---------------------------------------------------------------
ner_cases = [
    (
        "Extract all named entities and their types from: Elon Musk founded SpaceX in Hawthorne, California in 2002.",
        ["Elon Musk", "SpaceX", "Hawthorne", "2002"]
    ),
    (
        "Extract all named entities and their types from: The Nobel Peace Prize was awarded to Malala Yousafzai in Oslo, Norway in 2014.",
        ["Malala Yousafzai", "Oslo", "2014"]
    ),
    (
        "Extract all named entities and their types from: Apple Inc. released the first iPhone in January 2007 under CEO Steve Jobs.",
        ["Apple Inc.", "Steve Jobs", "January 2007", "iPhone"]
    ),
]

print("=" * 70)
print("NER — RAW ANSWERS + VALIDATOR ACCEPTANCE")
print("=" * 70)
for prompt, required in ner_cases:
    raw = solve_ner_deterministically(prompt)
    valid = validate_category_output("named_entity_recognition", prompt, raw) if raw else False
    if raw:
        data = json.loads(raw)
        found_texts = [e["text"] for e in data["entities"]]
        missing = [r for r in required if not any(r.lower() in f.lower() for f in found_texts)]
    else:
        found_texts = []
        missing = required

    flag = "✓" if (raw and valid and not missing) else ("MISS" if not raw else "✗")
    print(f"  {flag}")
    print(f"    Prompt: {prompt[:75]}")
    raw_repr = (raw[:120] + '...') if raw and len(raw) > 120 else (raw or 'None')
    print(f"    raw repr: {raw_repr}")

    print(f"    entities found: {found_texts}")
    print(f"    missing required: {missing}")
    print(f"    validator={valid}")
    print()

# ---- LOGIC ---------------------------------------------------------------
logic_cases = [
    (
        "Sam, Jo, and Lee each own one of: cat, dog, bird. Sam does not own the bird. Jo owns the dog. Who owns the cat?",
        "Sam"
    ),
    (
        "Alice, Bob, and Carol each like one of: red, blue, green. Alice likes red. Bob does not like blue. Who likes green?",
        "Bob"
    ),
    (
        "Mike, Sara, and Tom each play one of: tennis, soccer, basketball. Mike plays tennis. Sara does not play soccer. Who plays basketball?",
        "Sara"
    ),
]

print("=" * 70)
print("LOGIC — RAW ANSWERS + VALIDATOR ACCEPTANCE")
print("=" * 70)
for prompt, expected in logic_cases:
    raw = solve_logic_deterministically(prompt)
    valid = validate_category_output("logical_reasoning", prompt, raw) if raw else False
    correct = (raw is not None and raw.lower() == expected.lower())
    flag = "✓" if correct else ("MISS" if raw is None else "✗ WRONG")
    print(f"  {flag}")
    print(f"    Prompt:   {prompt[:75]}")
    print(f"    raw=      {raw!r:15s}  expected={expected!r}")
    print(f"    validator={valid}")
    print()

# ---- HONEST SELF-AUDIT NOTES -------------------------------------------
print("=" * 70)
print("SELF-AUDIT NOTES")
print("=" * 70)
print("""
Math format: plain number string ("225", "40320", "12", "4").
  - No "Answer: X" prefix. No units. Just the bare number.
  - LLM judge for math typically accepts bare numbers or numbers in sentences.
  - Risk: if judge expects units ("225 km") this fails. But our validator
    (validate_math) only checks the number is present, not units — same
    tolerance the judge likely applies.

NER format: {"entities": [{"text": "...", "type": "..."}]}
  - Valid JSON, flat list of entities with text+type keys.
  - This exactly matches the schema defined in validators.py validate_ner().
  - Risk: judge may expect PERSON/ORG/LOCATION/DATE/GPE vs our PERSON/ORG/LOCATION/DATE.
    "LOCATION" vs "GPE" (geopolitical entity) is a common spaCy vs plain-English split.

Logic format: bare name string ("Sam", "Bob", "Sara").
  - Single capitalized name, no punctuation, no sentence.
  - Our validator (validate_logical_reasoning) checks non-empty + not echoing prompt.
  - Risk: judge may expect "Sam owns the cat." full sentence. Bare name is minimal.
    But looking at our l-new tasks (syllogisms), the expected answers are "yes"/"no" 
    or a number — short bare answers. So bare name is consistent style.

Self-grading risk: YES, solver and tests were written by same author.
  - Math: arithmetic verified by Python itself (2+2=4 is not a correlated blind spot).
  - NER: schema structure verified by validate_ner() which is independent code.
  - Logic: brute-force enumeration is verified by exhaustion — if 1 solution remains,
    it IS the unique solution. This is mathematically self-certifying.
  - The only real correlated-blind-spot risk is FORMAT (not correctness):
    if both solver and test expect "225" but judge expects "225 km", we'd both
    fail together. This is the main format risk to check manually.
""")
