"""
deterministic_solvers.py — Zero-token, zero-risk deterministic layer.

Design contract:
  - Every public function returns either a string (confident answer) or None (fall through).
  - NEVER return a wrong answer. When in doubt, return None.
  - Returning None is always safe; the cascade handles it.
"""

import re
import json
import math
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MATH SOLVER
# ---------------------------------------------------------------------------

def _fmt(value: float) -> str:
    """Format a numeric result: strip trailing .0 for whole numbers."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    # Up to 4 significant decimal places, strip trailing zeros
    result = f"{value:.4f}".rstrip("0").rstrip(".")
    return result


def _solve_math_deterministically(prompt: str):
    """
    Try to solve a math problem purely with code.
    Returns a string answer if confident, otherwise None to fall through.
    """
    p = prompt.strip()
    pl = p.lower()

    # --- 1. Factorial ----------------------------------------------------------
    # "What is the value of 8 factorial?" / "calculate 8!" / "8 factorial"
    m = re.search(r"\b(\d+)\s*(?:factorial|!)\b", pl)
    if m:
        n = int(m.group(1))
        if 0 <= n <= 20:
            logger.info(f"[DETERM-MATH] factorial({n}) = {math.factorial(n)}")
            return str(math.factorial(n))
        return None  # too large to be safe

    # --- 2. Percentage of a number --------------------------------------------
    # "What is 15% of 200?" / "15 percent of 60"
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", pl)
    if m:
        pct, total = float(m.group(1)), float(m.group(2))
        result = pct / 100.0 * total
        logger.info(f"[DETERM-MATH] {pct}% of {total} = {result}")
        return _fmt(result)

    m = re.search(r"(\d+(?:\.\d+)?)\s*percent\s+of\s+(\d+(?:\.\d+)?)", pl)
    if m:
        pct, total = float(m.group(1)), float(m.group(2))
        result = pct / 100.0 * total
        logger.info(f"[DETERM-MATH] {pct}% of {total} = {result}")
        return _fmt(result)

    # --- 3. Rectangle area/perimeter ------------------------------------------
    # "A rectangle has a width of 7 cm and a length of 11 cm. What is its area?"
    rect_nums = re.findall(r"\b(\d+(?:\.\d+)?)\s*(?:cm|m|km|ft|in|mm)?\b", pl)
    is_rect = bool(re.search(r"\brectangle\b", pl))
    if is_rect and len(rect_nums) >= 2:
        a, b = float(rect_nums[0]), float(rect_nums[1])
        if re.search(r"\barea\b", pl):
            result = a * b
            logger.info(f"[DETERM-MATH] rect area {a}*{b} = {result}")
            return _fmt(result)
        if re.search(r"\bperimeter\b", pl):
            result = 2 * (a + b)
            logger.info(f"[DETERM-MATH] rect perimeter 2*({a}+{b}) = {result}")
            return _fmt(result)

    # --- 4. Triangle area ------------------------------------------------------
    # "A triangle has a base of 6 and a height of 4. What is its area?"
    is_tri = bool(re.search(r"\btriangle\b", pl))
    if is_tri and re.search(r"\barea\b", pl):
        m_base = re.search(r"\bbase\s+(?:of\s+)?(\d+(?:\.\d+)?)", pl)
        m_height = re.search(r"\bheight\s+(?:of\s+)?(\d+(?:\.\d+)?)", pl)
        if m_base and m_height:
            base = float(m_base.group(1))
            height = float(m_height.group(1))
            result = 0.5 * base * height
            logger.info(f"[DETERM-MATH] triangle area 0.5*{base}*{height} = {result}")
            return _fmt(result)

    # --- 5. Speed × Time = Distance -------------------------------------------
    # "A car travels at 90 km per hour. How many kilometers does it cover in 2.5 hours?"
    m_speed = re.search(r"(\d+(?:\.\d+)?)\s*(?:km|miles?|mi)\s*(?:per\s*hour|/\s*h(?:our)?|ph)\b", pl)
    m_time = re.search(r"(\d+(?:\.\d+)?)\s*hours?\b", pl)
    if m_speed and m_time and re.search(r"\b(how many|how far|how much|distance|cover)\b", pl):
        speed = float(m_speed.group(1))
        time = float(m_time.group(1))
        result = speed * time
        logger.info(f"[DETERM-MATH] distance {speed}*{time} = {result}")
        return _fmt(result)

    # --- 6. Unit price × Quantity = Total cost --------------------------------
    # "A store sells apples for $0.75 each. How much do 16 apples cost?"
    # "Each apple costs $1.20. What is the total for 8 apples?"
    m_price = re.search(r"\$\s*(\d+(?:\.\d+)?)\s+each", pl)
    m_qty = re.search(r"(\d+(?:\.\d+)?)\s+(?:\w+\s+)?(?:apples?|items?|units?|tickets?|books?|pens?|oranges?|bananas?)\b", pl)
    if m_price and m_qty and re.search(r"\b(total|cost|much)\b", pl):
        price = float(m_price.group(1))
        qty = float(m_qty.group(1))
        result = price * qty
        logger.info(f"[DETERM-MATH] cost {price}*{qty} = {result}")
        return _fmt(result)

    # --- 7. Percentage subtraction word problem --------------------------------
    # "A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many remain?"
    m_total = re.search(r"has\s+(\d+)\s+items?\b", pl)
    m_pct_sell = re.search(r"sells?\s+(\d+(?:\.\d+)?)\s*%", pl)
    m_extra_sell = re.search(r"and\s+(\d+)\s+more", pl)
    if m_total and m_pct_sell and m_extra_sell and re.search(r"\b(remain|remaining|left|how many)\b", pl):
        total = float(m_total.group(1))
        pct = float(m_pct_sell.group(1))
        extra = float(m_extra_sell.group(1))
        result = total - (total * pct / 100.0) - extra
        if result >= 0:
            logger.info(f"[DETERM-MATH] inventory remain {total}-(pct%+extra) = {result}")
            return _fmt(result)

    # --- 8. Simple arithmetic: "What is X + Y?" / "X - Y" / "X * Y" / "X / Y"
    # Matches things like "What is 18 + 24?" or "Calculate 100 - 37"
    m = re.search(
        r"(?:what\s+is|calculate|compute|find|evaluate|solve)[\s:]*"
        r"(\d+(?:\.\d+)?)\s*([\+\-\*\/×÷])\s*(\d+(?:\.\d+)?)\s*\?",
        pl
    )
    if m:
        a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
        try:
            op_map = {"+": a + b, "-": a - b, "*": a * b, "×": a * b, "/": a / b, "÷": a / b}
            result = op_map[op]
            logger.info(f"[DETERM-MATH] {a} {op} {b} = {result}")
            return _fmt(result)
        except ZeroDivisionError:
            return None

    # --- 9. Days × rate problems ----------------------------------------------
    # "If a worker earns $120 per day. How much will they earn in 5 days?"
    m_rate = re.search(r"\$\s*(\d+(?:\.\d+)?)\s*per\s+day", pl)
    m_days = re.search(r"(\d+)\s+days?\b", pl)
    if m_rate and m_days and re.search(r"\b(earn|make|receive|total)\b", pl):
        rate = float(m_rate.group(1))
        days = float(m_days.group(1))
        result = rate * days
        logger.info(f"[DETERM-MATH] earnings {rate}*{days} = {result}")
        return _fmt(result)

    # --- 10. Percentage increase / decrease ------------------------------------
    # "A price of $200 increased by 15%. What is the new price?"
    # "120 decreased by 25%. What is the result?"
    m_base_inc = re.search(r"(\d+(?:\.\d+)?)\s+(?:is\s+)?increased\s+by\s+(\d+(?:\.\d+)?)\s*%", pl)
    if m_base_inc:
        base = float(m_base_inc.group(1))
        pct = float(m_base_inc.group(2))
        result = base * (1 + pct / 100.0)
        logger.info(f"[DETERM-MATH] {base} increased by {pct}% = {result}")
        return _fmt(result)

    m_base_dec = re.search(r"(\d+(?:\.\d+)?)\s+(?:is\s+)?decreased\s+by\s+(\d+(?:\.\d+)?)\s*%", pl)
    if not m_base_dec:
        m_base_dec = re.search(r"(\d+(?:\.\d+)?)\s+(?:is\s+)?reduced\s+by\s+(\d+(?:\.\d+)?)\s*%", pl)
    if m_base_dec:
        base = float(m_base_dec.group(1))
        pct = float(m_base_dec.group(2))
        result = base * (1 - pct / 100.0)
        logger.info(f"[DETERM-MATH] {base} decreased by {pct}% = {result}")
        return _fmt(result)

    # "X% increase over Y" / "X% more than Y"
    m_pct_more = re.search(r"(\d+(?:\.\d+)?)\s*%\s+(?:increase|more)\s+(?:over|than)\s+(\d+(?:\.\d+)?)", pl)
    if m_pct_more:
        pct = float(m_pct_more.group(1))
        base = float(m_pct_more.group(2))
        result = base * (1 + pct / 100.0)
        logger.info(f"[DETERM-MATH] {base} + {pct}% = {result}")
        return _fmt(result)

    # --- 11. Average of a list of numbers -------------------------------------
    # "What is the average of 10, 20, and 30?"
    # "Find the mean of 5, 15, 25, 35."
    if re.search(r"\b(average|mean)\b", pl):
        nums = re.findall(r"(\d+(?:\.\d+)?)", p)
        if len(nums) >= 2:
            values = [float(n) for n in nums]
            result = sum(values) / len(values)
            logger.info(f"[DETERM-MATH] average({values}) = {result}")
            return _fmt(result)

    # --- 12. Simple linear equation: "solve for x" ----------------------------
    # "Solve for x: 2x + 3 = 11" → x = 4
    # "If 3x - 6 = 9, what is x?"
    # Pattern: ax + b = c  or  ax - b = c  (single variable, integer coefficients)
    # GUARD: skip if differential or exponent notation detected
    _has_exp_or_diff = bool(re.search(r"(\^|\*\*|d[yx]/d[yx]|dy|dx|d/d)", pl))
    if not _has_exp_or_diff:
        m_eq = re.search(
            r"(\d*)\s*x\s*([\+\-])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)",
            pl
        )
        if m_eq:
            a_str = m_eq.group(1).strip()
            a = float(a_str) if a_str else 1.0
            if a == 0:
                return None
            op = m_eq.group(2)
            b = float(m_eq.group(3))
            c = float(m_eq.group(4))
            # ax + b = c  →  x = (c - b) / a
            # ax - b = c  →  x = (c + b) / a
            if op == "+":
                x = (c - b) / a
            else:
                x = (c + b) / a
            logger.info(f"[DETERM-MATH] linear eq: {a}x {op} {b} = {c} -> x = {x}")
            return _fmt(x)

        # Also handle: "Nx = C" (no +/- term)
        m_simple = re.search(r"(\d+)\s*x\s*=\s*(\d+(?:\.\d+)?)", pl)
        if m_simple:
            a = float(m_simple.group(1))
            if a == 0:
                return None
            c = float(m_simple.group(2))
            x = c / a
            logger.info(f"[DETERM-MATH] simple linear: {a}x = {c} -> x = {x}")
            return _fmt(x)

    return None  # Could not solve deterministically — fall through to LLM



# ---------------------------------------------------------------------------
# NER SOLVER
# ---------------------------------------------------------------------------

# Month names for date matching
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|"
    "october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec"
)

# Common known organizations not ending in a suffix
_KNOWN_ORGS = {
    "spacex", "tesla", "google", "amazon", "microsoft", "apple", "meta",
    "openai", "nasa", "nato", "un", "who", "imf", "cnn", "bbc",
    "facebook", "twitter", "netflix", "uber", "lyft", "airbnb",
}

# Well-known product names (not orgs)
_KNOWN_PRODUCTS = {"iphone", "android", "windows", "macos", "linux", "ios",
                   "chatgpt", "gpt", "gemini", "pixel", "galaxy", "kindle"}

_ORG_SUFFIXES = r"(?:Inc\.?|Corp\.?|Ltd\.?|LLC|Co\.?|Group|Foundation|Institute|University|College|School|Hospital|Bank|Trust|Fund|Labs?|Technologies|Tech|Systems|Services|Partners|Associates|International|Global)"

# Prepositions that introduce locations
_LOC_PREPS = r"(?:in|at|from|near|to|of)\s+"

# Month names to exclude from person/location detection
_MONTH_RE = re.compile(
    r"^(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)$",
    re.IGNORECASE
)

# Country / major-city blocklist that are unambiguously locations
_KNOWN_LOCATIONS = {
    "usa", "uk", "us", "china", "india", "france", "germany", "russia",
    "japan", "canada", "australia", "brazil", "mexico", "italy", "spain",
    "norway", "sweden", "denmark", "finland", "switzerland", "austria",
    "california", "texas", "florida", "new york", "london", "paris",
    "berlin", "oslo", "tokyo", "beijing", "sydney", "toronto", "dubai",
    "hawthorne", "palo alto", "cupertino", "seattle", "chicago", "boston",
}


def _extract_dates(text: str):
    """Extract dates: month+year, year alone, or full dates."""
    dates = []
    # "January 2007", "Jan 2007"
    for m in re.finditer(rf"\b({_MONTHS})\s+(\d{{4}})\b", text, re.IGNORECASE):
        dates.append(m.group(0).strip())
    # Standalone 4-digit years in a date context
    for m in re.finditer(r"\bin\s+(\d{4})\b", text, re.IGNORECASE):
        year = m.group(1)
        if year not in [d for d in dates]:
            dates.append(year)
    return list(dict.fromkeys(dates))  # dedupe, preserve order


def _extract_persons(text: str):
    """Extract PERSON entities: two+ adjacent Title-Case words."""
    # Avoid extracting organisation names again
    persons = []
    # Pattern: 2 or 3 adjacent title-cased words not followed by org suffix
    for m in re.finditer(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b",
        text
    ):
        candidate = m.group(1).strip()
        # Skip known orgs or products
        if candidate.lower().rstrip(".") in _KNOWN_ORGS:
            continue
        if candidate.lower().rstrip(".") in _KNOWN_PRODUCTS:
            continue
        # Skip if any word is a month name
        if any(_MONTH_RE.match(w) for w in candidate.split()):
            continue
        # Skip if ends in org suffix
        if re.search(_ORG_SUFFIXES + r"$", candidate):
            continue
        # Skip common non-person title-case phrases
        skip_words = {"the", "a", "an", "this", "that", "it", "internet", "things"}
        first_word = candidate.split()[0].lower()
        if first_word in skip_words:
            continue
        persons.append(candidate)
    return list(dict.fromkeys(persons))


def _extract_orgs(text: str):
    """Extract ORG entities: known orgs + Title-Case words before suffix."""
    orgs = []
    # Explicit suffix pattern: "Apple Inc.", "SpaceX Corp"
    for m in re.finditer(
        rf"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*)\s+{_ORG_SUFFIXES}",
        text
    ):
        orgs.append(m.group(0).strip())
    # Known standalone org names (case-insensitive lookup but preserve original case)
    for word in re.findall(r"\b[A-Za-z][A-Za-z0-9]+\b", text):
        if word.lower() in _KNOWN_ORGS and word not in orgs:
            orgs.append(word)
    return list(dict.fromkeys(orgs))


def _extract_locations(text: str):
    """Extract LOCATION entities: prep + capitalized word, or known locations."""
    locations = []
    # Preposition-triggered locations: "in Oslo", "in California"
    for m in re.finditer(
        rf"{_LOC_PREPS}([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
        text
    ):
        candidate = m.group(1).strip()
        if candidate.lower() not in _KNOWN_ORGS:
            # Don't add if it's a month name
            if not _MONTH_RE.match(candidate):
                locations.append(candidate)
    # Known location names
    text_lower = text.lower()
    for loc in _KNOWN_LOCATIONS:
        # Find properly-cased version in text
        for m in re.finditer(rf"\b{re.escape(loc)}\b", text_lower):
            original = text[m.start():m.end()]
            if original not in locations:
                locations.append(original)
    return list(dict.fromkeys(locations))


def _solve_ner_deterministically(prompt: str):
    """
    Attempt to extract named entities using pure regex rules.

    Returns a JSON string matching the {entities:[{text,type}]} schema if
    we can confidently extract at least 2 entities. Returns None otherwise
    so the cascade falls through to the LLM.

    Critical safety rule: NEVER return an empty entity list.
    If unsure, return None.
    """
    # Only attempt on prompts explicitly requesting entity extraction
    if not re.search(
        r"\b(extract|identify|find|list)\b.{0,40}\b(entities|names|people|places|organizations|dates)\b",
        prompt, re.IGNORECASE
    ):
        return None

    # Extract the source sentence (after "from:" or the full prompt)
    source_match = re.search(r"from\s*:\s*(.+)$", prompt, re.IGNORECASE | re.DOTALL)
    source_text = source_match.group(1).strip() if source_match else prompt

    dates = _extract_dates(source_text)
    orgs = _extract_orgs(source_text)
    locations = _extract_locations(source_text)
    persons = _extract_persons(source_text)

    # Extract known products
    products = []
    for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9]+)\b", source_text):
        word = m.group(1)
        if word.lower() in _KNOWN_PRODUCTS:
            products.append(word)

    # Remove person candidates already captured as orgs
    org_lower = {o.lower() for o in orgs}
    persons = [p for p in persons if p.lower() not in org_lower]

    # Remove location candidates already captured as orgs
    locations = [l for l in locations if l.lower() not in org_lower]

    # Deduplicate locations and persons against each other
    person_lower = {p.lower() for p in persons}
    locations = [l for l in locations if l.lower() not in person_lower]

    # Deduplicate products from orgs
    product_lower = {p.lower() for p in products}
    orgs = [o for o in orgs if o.lower() not in product_lower]

    entities = []
    for p in persons:
        entities.append({"text": p, "type": "PERSON"})
    for o in orgs:
        entities.append({"text": o, "type": "ORG"})
    for l in locations:
        entities.append({"text": l, "type": "LOCATION"})
    for d in dates:
        entities.append({"text": d, "type": "DATE"})
    for pr in products:
        entities.append({"text": pr, "type": "PRODUCT"})

    if len(entities) < 2:
        logger.info(f"[DETERM-NER] Too few entities ({len(entities)}) — falling through to LLM.")
        return None

    result = json.dumps({"entities": entities})
    logger.info(f"[DETERM-NER] Extracted {len(entities)} entities deterministically.")
    return result


# ---------------------------------------------------------------------------
# LOGIC CONSTRAINT SOLVER
# ---------------------------------------------------------------------------
# Only handles finite-domain assignment puzzles: N entities each assigned one
# of N values, with positive ("X owns Y") and negative ("X does not own Y")
# constraints.  Falls through on ANYTHING else (syllogisms, comparatives,
# probability, pigeonhole, modus ponens/tollens).

from itertools import permutations as _permutations


# Verb stems with optional 's' for third-person singular ("own", "owns", etc.)
_OWNS_VERBS = r"(?:owns?|has|haves?|likes?|prefers?|eats?|drinks?|drives?|wears?|plays?|uses?|gets?|takes?|sits?|stands?|live(?:s| in))"


# Explicit fallthrough triggers — these are NOT constraint puzzles
_FALLTHROUGH_PATTERNS = [
    r"\b(warm[- ]blooded|mammal|faster|older|taller|heavier|smarter|bigger|smaller)\b",
    r"\b(if the|power goes|lights (are|turn)|modus)\b",
    r"\b(probability|chance|likely|marble|ball|bag contains)\b",
    r"\b(syllogism|therefore|implies|entails)\b",
    r"\b(is a|is an|is the)\b.{0,20}\b(is|are)\b",  # "A is a B. B is C. Is A C?"
]


def _parse_constraint_puzzle(pl: str):
    """
    Try to parse a finite-domain assignment puzzle.
    Returns (entities_list, values_list, positive_constraints, negative_constraints)
    or None if parsing fails.
    """
    # --- Reject non-constraint prompts immediately ---
    for pat in _FALLTHROUGH_PATTERNS:
        if re.search(pat, pl, re.IGNORECASE):
            return None

    # --- Require a "who" or "what does X" query ---
    if not re.search(
        r"\bwho\b.{0,50}\b" + _OWNS_VERBS + r"|"
        r"what\s+does\s+[a-z]+\s+" + _OWNS_VERBS,
        pl, re.IGNORECASE
    ):
        return None

    # --- Extract people: comma/and-separated Title-Case names before the verb ---
    # Pattern: "Sam, Jo, and Lee each own one of: cat, dog, bird."
    entity_match = re.search(
        r"([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*(?:,?\s+and\s+[A-Z][a-z]+)?)\s+"
        r"(?:each\s+)?(?:" + _OWNS_VERBS + r")",
        pl
    )
    if not entity_match:
        return None
    entity_str = entity_match.group(1)
    entities = re.findall(r"[A-Z][a-z]+", entity_str)
    if len(entities) < 2 or len(entities) > 5:
        return None

    entities_l = [e.lower() for e in entities]

    # --- Extract values: multiple patterns ---
    # Priority 1: "one of: X, Y, Z" or "one of the following: X, Y, Z"
    # Priority 2: "different pets: X, Y, Z" (category noun MUST be followed by colon)
    val_match = re.search(
        r"(?:"
        r"one of[:\s]+"
        r"|(?:different\s+)?(?:pet|color|sport|subject|house|car|drink|flower|job|fruit|language|country)s?\s*:\s*"
        r")"
        r"([a-z]+(?:,\s*[a-z]+)*(?:(?:,\s*)?(?:and|or)\s+[a-z]+)?)",
        pl.lower()
    )
    if not val_match:
        return None

    val_str = val_match.group(1)
    values = [v.strip().rstrip(".") for v in re.split(r",\s*|\s+(?:and|or)\s+", val_str) if v.strip()]
    values = [v for v in values if len(v) > 1 and v not in ("the", "a", "an", "of")]

    if len(values) != len(entities):
        return None  # domain size mismatch — unsafe

    values_l = values  # already lowercase from .lower()

    # --- Parse constraints ---
    pos_constraints = []
    neg_constraints = []

    sentences = re.split(r"[.!?]\s*", pl.lower())
    for sent in sentences:
        s = sent.strip()

        # Positive: "Jo owns the dog" / "Sam likes cats"
        m_pos = re.search(
            r"\b([a-z]+)\b\s+(?:" + _OWNS_VERBS + r")\s+(?:the\s+|a\s+)?([a-z]+)\b",
            s
        )
        if m_pos:
            ent, val = m_pos.group(1), m_pos.group(2)
            if ent in entities_l and val in values_l:
                pos_constraints.append((ent, val))

        # Negative: "Sam does not own the bird" / "Lee doesn't like cats"
        m_neg = re.search(
            r"\b([a-z]+)\b\s+(?:does\s+not|doesn'?t|cannot|can'?t|is\s+not)\s+"
            r"(?:" + _OWNS_VERBS + r")\s+(?:the\s+|a\s+)?([a-z]+)\b",
            s
        )
        if m_neg:
            ent, val = m_neg.group(1), m_neg.group(2)
            if ent in entities_l and val in values_l:
                neg_constraints.append((ent, val))

    return entities_l, values_l, pos_constraints, neg_constraints


def _apply_constraints(assignment, pos_constraints, neg_constraints):
    """
    assignment: dict {entity_l: value_l}
    Returns True if all constraints satisfied.
    """
    for ent, val in pos_constraints:
        if assignment.get(ent) != val:
            return False
    for ent, val in neg_constraints:
        if assignment.get(ent) == val:
            return False
    return True


def _solve_logic_deterministically(prompt: str):
    """
    Attempt to solve a finite-domain assignment constraint puzzle.

    Safety rules:
    - Only handles "who owns/likes/has X" puzzles with N entities and N values.
    - Brute-forces all permutations and returns only if EXACTLY ONE solution exists.
    - NEVER attempts syllogisms, comparative chains, probability, pigeonhole.
    - Returns None on any ambiguity, parse failure, or multiple/zero solutions.
    """
    pl = prompt.lower()

    parsed = _parse_constraint_puzzle(prompt)  # pass original for Title-Case parsing
    if parsed is None:
        return None

    entities_l, values_l, pos_constraints, neg_constraints = parsed

    # If we have no constraints at all, it's ambiguous — fall through
    if not pos_constraints and not neg_constraints:
        logger.info("[DETERM-LOGIC] No constraints found — falling through.")
        return None

    # Brute force all permutations
    valid_solutions = []
    for perm in _permutations(values_l):
        assignment = dict(zip(entities_l, perm))
        if _apply_constraints(assignment, pos_constraints, neg_constraints):
            valid_solutions.append(assignment)

    if len(valid_solutions) != 1:
        # 0 = contradiction in puzzle, >1 = underdetermined — both fall through
        logger.info(
            f"[DETERM-LOGIC] {len(valid_solutions)} solutions found — "
            "falling through (need exactly 1)."
        )
        return None

    solution = valid_solutions[0]
    logger.info(f"[DETERM-LOGIC] Unique solution: {solution}")

    # Find what the query asks for
    # "Who owns the cat?" → find entity whose value == "cat"
    query_match = re.search(
        r"who\s+(?:" + _OWNS_VERBS + r")\s+(?:the\s+|a\s+)?([a-z]+)\??",
        pl
    )
    if query_match:
        queried_val = query_match.group(1).rstrip("?").strip()
        for ent, val in solution.items():
            if val == queried_val:
                return ent.capitalize()
        return None  # queried value not in solution — shouldn't happen

    # "What does Sam own?" → find value assigned to "sam"
    query_ent_match = re.search(
        r"what\s+(?:does\s+)?([a-z]+)\s+(?:" + _OWNS_VERBS + r")",
        pl
    )
    if query_ent_match:
        queried_ent = query_ent_match.group(1)
        if queried_ent in solution:
            return solution[queried_ent].capitalize()

    return None  # can't determine what is being asked


def solve_math_deterministically(prompt: str):
    """Public wrapper — never raises; returns None on any error."""
    try:
        return _solve_math_deterministically(prompt)
    except Exception:
        logger.exception("[DETERM-MATH] Unexpected error — falling through")
        return None


def solve_ner_deterministically(prompt: str):
    """Public wrapper — never raises; returns None on any error."""
    try:
        return _solve_ner_deterministically(prompt)
    except Exception:
        logger.exception("[DETERM-NER] Unexpected error — falling through")
        return None


def solve_logic_deterministically(prompt: str):
    """Public wrapper — never raises; returns None on any error."""
    try:
        return _solve_logic_deterministically(prompt)
    except Exception:
        logger.exception("[DETERM-LOGIC] Unexpected error — falling through")
        return None

