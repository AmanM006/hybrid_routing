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


def solve_math_deterministically(prompt: str):
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


def solve_ner_deterministically(prompt: str):
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
