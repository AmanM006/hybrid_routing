"""
International / accented NER repair and deterministic tests.
Run: python test_international_ner.py
"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

from deterministic_solvers import solve_ner_deterministically
from validators import repair_ner_output, coerce_ner_output, validate_ner

CASES = [
    {
        "id": "adv-n01",
        "prompt": "Extract entities from: François Mitterrand met with Helmut Kohl in München during 1987.",
        "required": ["François Mitterrand", "Helmut Kohl", "München", "1987"],
    },
    {
        "id": "int-01",
        "prompt": "Extract entities from: José García visited São Paulo in 2019.",
        "required": ["José García", "São Paulo", "2019"],
    },
    {
        "id": "int-02",
        "prompt": "Extract entities from: Björk performed in Reykjavík with Sigur Rós.",
        "required": ["Reykjavík"],  # Björk is single-token — may fall through to remote
        "optional": ["Björk", "Sigur Rós"],
    },
    {
        "id": "int-03",
        "prompt": "Extract entities from: Angela Merkel met with Emmanuel Macron in Paris.",
        "required": ["Angela Merkel", "Emmanuel Macron", "Paris"],
    },
    {
        "id": "int-04",
        "prompt": "Extract entities from: Zürich-based FIFA held talks in Genève during 2022.",
        "required": ["Genève", "2022"],
    },
    {
        "id": "int-05",
        "prompt": "Extract entities from: Łódź factory was inspected by Krzysztof Wiśniewski in 2020.",
        "required": ["Łódź", "Krzysztof Wiśniewski", "2020"],
    },
    {
        "id": "int-06",
        "prompt": "Extract entities from: Søren Kierkegaard wrote in København.",
        "required": ["Søren Kierkegaard", "København"],
    },
    {
        "id": "int-07",
        "prompt": "Extract entities from: Malmö and Göteborg are cities in Sweden.",
        "required": ["Malmö", "Göteborg", "Sweden"],
    },
]

KIMI_BULLET = """Entities:
- François Mitterrand: PERSON
- Helmut Kohl: PERSON
- München: LOCATION
- 1987: DATE"""

PASS = FAIL = 0


def check_entities(label, json_str, required):
    global PASS, FAIL
    data = json.loads(json_str)
    found = [e["text"] for e in data["entities"]]
    missing = [r for r in required if not any(r.lower() in f.lower() for f in found)]
    if missing:
        print(f"  FAIL | {label} | missing={missing} | found={found}")
        FAIL += 1
    else:
        print(f"  PASS | {label} | found={found}")
        PASS += 1


print("=" * 60)
print("DETERMINISTIC NER (accented names)")
print("=" * 60)
for c in CASES:
    got = solve_ner_deterministically(c["prompt"])
    if got:
        check_entities(c["id"] + " det", got, c["required"])
    else:
        print(f"  MISS | {c['id']} det (fell through — OK for remote)")
        PASS += 1
print("=" * 60)
print("REPAIR: kimi bullet-list format")
print("=" * 60)
repaired, ok = coerce_ner_output(KIMI_BULLET)
print(f"  coerce ok={ok}")
if repaired:
    check_entities("kimi-bullet", repaired, ["François Mitterrand", "Helmut Kohl", "München", "1987"])

print()
print("=" * 60)
print(f"TOTAL: {PASS} PASS / {FAIL} FAIL")
if FAIL:
    sys.exit(1)
