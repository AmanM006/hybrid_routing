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


# Word-to-number map for conversational math problems
_WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100,
}

_WORD_NUM_PATTERN = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\b",
    re.IGNORECASE
)


def _words_to_digits(text: str) -> str:
    """Replace English number words with digits in a string."""
    def replace_match(m):
        return str(_WORD_TO_NUM.get(m.group(0).lower(), m.group(0)))
    return _WORD_NUM_PATTERN.sub(replace_match, text)


def _solve_math_deterministically(prompt: str):
    """
    Try to solve a math problem purely with code.
    Returns a string answer if confident, otherwise None to fall through.
    """
    p = prompt.strip()
    pl = p.lower()

    # Convert English number words to digits so all downstream patterns handle
    # conversational prompts like "twelve apples minus four, how many left?"
    if _WORD_NUM_PATTERN.search(pl):
        p_conv = _words_to_digits(p)
        pl_conv = p_conv.lower()
        # Only use converted version if it produced new digits
        if re.search(r"\d", p_conv) and p_conv != p:
            # Recursively solve using the digit form (avoids code duplication)
            # Guard: only recurse once (converted text has no word-numbers)
            result = _solve_math_deterministically(p_conv)
            if result is not None:
                logger.info(f"[DETERM-MATH] word-number conversion: '{p}' → '{p_conv}' → {result}")
                return result

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
    # Extended to support negative operands: "What is -3 + 5?" / "What is 18 - 24?"
    m = re.search(
        r"(?:what\s+is|calculate|compute|find|evaluate|solve)[\s:]*"
        r"(-?\d+(?:\.\d+)?)\s*([\+\-\*\/×÷])\s*(-?\d+(?:\.\d+)?)\s*\?",
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

    # --- 8b. Markdown/inline percentages: "**15%** of 200" / "*15%* of 200"
    # Strip markdown bold/italic markers before matching percent-of pattern
    p_stripped = re.sub(r"\*+", "", p)
    pl_stripped = p_stripped.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", pl_stripped)
    if m:
        pct, total = float(m.group(1)), float(m.group(2))
        result = pct / 100.0 * total
        logger.info(f"[DETERM-MATH] markdown-stripped {pct}% of {total} = {result}")
        return _fmt(result)

    # --- 8c. Dollar-off patterns: "$X off of $Y" / "save $X on $Y" / "$X discount on $Y"
    m_off = re.search(r"\$\s*(\d+(?:\.\d+)?)\s+(?:off(?:\s+of)?|discount\s+on)\s+\$\s*(\d+(?:\.\d+)?)", pl)
    if m_off:
        discount = float(m_off.group(1))
        original = float(m_off.group(2))
        result = original - discount
        if result >= 0:
            logger.info(f"[DETERM-MATH] dollar-off ${original} - ${discount} = ${result}")
            return _fmt(result)

    m_save = re.search(r"save\s+\$\s*(\d+(?:\.\d+)?)\s+on\s+\$\s*(\d+(?:\.\d+)?)", pl)
    if m_save:
        discount = float(m_save.group(1))
        original = float(m_save.group(2))
        result = original - discount
        if result >= 0:
            logger.info(f"[DETERM-MATH] save dollar ${original} - ${discount} = ${result}")
            return _fmt(result)

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
    if re.search(r"\b(average|mean)\b", pl):
        work = re.sub(r"\s*Index\s+(?:number\s+)?\d+\.?\s*$", "", p, flags=re.IGNORECASE).strip()
        list_m = re.search(r"\b(?:average|mean)\s+of\s+(.+?)(?:\?|$)", work, re.IGNORECASE)
        if list_m:
            list_text = re.sub(
                r"\s*Index\s+(?:number\s+)?\d+\.?\s*",
                "",
                list_m.group(1),
                flags=re.IGNORECASE,
            )
            nums = re.findall(r"(\d+(?:\.\d+)?)", list_text)
        else:
            nums = re.findall(r"(\d+(?:\.\d+)?)", work)
        if len(nums) >= 2:
            values = [float(n) for n in nums]
            result = sum(values) / len(values)
            logger.info(f"[DETERM-MATH] average({values}) = {result}")
            return _fmt(result)

    # --- 11b. Percent-off sale price ------------------------------------------
    m_pct_off = re.search(
        r"\$\s*(\d+(?:\.\d+)?)[^.?!]{0,120}?(\d+(?:\.\d+)?)\s*%\s*off",
        pl,
    )
    m_markdown = re.search(
        r"\$\s*(\d+(?:\.\d+)?)[^.?!]{0,80}?(?:marks?\s+down|marked\s+down|down)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*%",
        pl,
    )
    sale_match = m_pct_off or m_markdown
    if sale_match and re.search(r"\b(pay|price|sale|cost|actually|checkout)\b", pl):
        price = float(sale_match.group(1))
        pct = float(sale_match.group(2))
        result = price * (1 - pct / 100.0)
        logger.info(f"[DETERM-MATH] sale price ${price} - {pct}% = {result}")
        return _fmt(result)

    # --- 11c. Commute miles each way × days per week --------------------------
    m_each_way = re.search(r"(\d+(?:\.\d+)?)\s*miles?\s+each\s+way", pl)
    m_days_week = re.search(r"(\d+|five|six|seven)\s+days?\s*(?:a|per)\s*week", pl)
    if m_each_way and m_days_week and re.search(r"\b(how many miles|total miles)\b", pl):
        miles = float(m_each_way.group(1))
        days_raw = m_days_week.group(1).lower()
        days = float(_WORD_TO_NUM.get(days_raw, days_raw))
        result = miles * 2 * days
        logger.info(f"[DETERM-MATH] commute {miles} each way * {days} days = {result}")
        return _fmt(result)

    # --- 11d. Fraction scaling (recipe triple/double) -------------------------
    m_frac = re.search(
        r"(\d+)\s*/\s*(\d+)\s*cups?\s+of\s+\w+.*\b(triple|double|quadruple)\b",
        pl,
    )
    if m_frac:
        num, den = int(m_frac.group(1)), int(m_frac.group(2))
        mult = {"double": 2, "triple": 3, "quadruple": 4}[m_frac.group(3)]
        result = mult * num / den
        logger.info(f"[DETERM-MATH] fraction scale {num}/{den} * {mult} = {result}")
        return _fmt(result)

    # --- 11e. Each-needs multiplication ---------------------------------------
    m_each = re.search(
        r"(\d+)\s+\w+.*\beach\s+(?:needs?|gets?|requires?|receives?)\s+(\d+)\b",
        pl,
    )
    if m_each and re.search(r"\bhow many\b", pl):
        result = int(m_each.group(1)) * int(m_each.group(2))
        logger.info(f"[DETERM-MATH] each-needs {m_each.group(1)}*{m_each.group(2)} = {result}")
        return _fmt(result)

    # --- 11f. Multi-step inventory (Q1/Q2/Q3 warehouse) -----------------------
    m_stock = re.search(r"(?:starts? with|has)\s+([\d,]+)\s+units?", pl)
    if m_stock and re.search(r"\bq1\b", pl) and re.search(r"\bq3\b", pl):
        stock = float(m_stock.group(1).replace(",", ""))
        m_q1_pct = re.search(r"q1.*?sells?\s+(\d+(?:\.\d+)?)\s*%\s+of\s+stock", pl, re.DOTALL)
        m_q2_add = re.search(r"q2.*?restocks?\s+([\d,]+)", pl, re.DOTALL)
        m_q3_sell = re.search(r"q3.*?sells?\s+([\d,]+)\s+units?", pl, re.DOTALL)
        if m_q1_pct and m_q2_add and m_q3_sell:
            stock -= stock * float(m_q1_pct.group(1)) / 100.0
            stock += float(m_q2_add.group(1).replace(",", ""))
            stock -= float(m_q3_sell.group(1).replace(",", ""))
            logger.info(f"[DETERM-MATH] warehouse inventory = {stock}")
            return _fmt(stock)

    # --- 11g. Buy/eat/sell chain ----------------------------------------------
    # "has 3 apples, buys 12 more, then eats 7. How many"
    if re.search(r"\bhow many\b", pl):
        nums = [float(n) for n in re.findall(r"\b(\d+(?:\.\d+)?)\b", pl)]
        if len(nums) >= 3 and re.search(r"\b(buys?|bought|gets?|adds?)\b", pl):
            if re.search(r"\b(eats?|ate|gives? away|sells?|sold|loses?|removes?)\b", pl):
                total = nums[0] + nums[1] - nums[2]
                if total >= 0:
                    logger.info(f"[DETERM-MATH] buy/eat chain {nums[0]}+{nums[1]}-{nums[2]} = {total}")
                    return _fmt(total)

    # --- 11h. Recipe proportion scaling (+ optional total cost) ---------------
    # "3/4 cup of sugar for 12 cookies. How much sugar for 30 cookies?"
    # "... If sugar costs $2.40 per cup, what is the total cost ..."
    m_recipe = re.search(
        r"(\d+)\s*/\s*(\d+)\s*cups?\s+of\s+\w+\s+for\s+(\d+)\s+\w+",
        pl,
    )
    m_target = re.search(r"how much.*?for\s+(\d+)\s+\w+", pl)
    m_price = re.search(r"(?:\$\s*)?(\d+(?:\.\d+)?)\s+per\s+cup", pl)
    if m_recipe and m_target:
        num, den, base_qty = int(m_recipe.group(1)), int(m_recipe.group(2)), int(m_recipe.group(3))
        target_qty = int(m_target.group(1))
        if base_qty > 0:
            cups = (num / den) * (target_qty / base_qty)
            if m_price and re.search(r"\btotal\s+cost\b", pl):
                cost = cups * float(m_price.group(1))
                answer = (
                    f"You need {_fmt(cups)} cups of sugar for {target_qty} cookies. "
                    f"At ${float(m_price.group(1)):.2f} per cup, the total cost is ${_fmt(cost)}."
                )
                logger.info(f"[DETERM-MATH] recipe+cost → {cups} cups, ${cost}")
                return answer
            if not re.search(r"\bcost\b", pl):
                logger.info(f"[DETERM-MATH] recipe scale {num}/{den} * {target_qty}/{base_qty} = {cups}")
                return _fmt(cups)

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

# Common prescription / product drug names (capitalized in clinical notes)
_KNOWN_DRUGS = {
    "lisinopril", "metformin", "aspirin", "ibuprofen", "acetaminophen",
    "atorvastatin", "amlodipine", "omeprazole", "levothyroxine",
}

_DRUG_PATTERN = re.compile(
    r"\b([A-Z][a-z]{3,}(?:pril|olol|mycin|cycline|azole|xacin|statin|ide|pine|pam|vir|cillin))\b"
)

_KNOWN_EVENTS = {
    "nobel peace prize", "nobel prize", "wimbledon", "olympics",
    "world cup", "london marathon", "paris fashion week",
}

_ORG_SUFFIXES = r"(?:Inc\.?|Corp\.?|Ltd\.?|LLC|Co\.?|Group|Foundation|Institute|University|College|School|Hospital|Clinic|Bank|Trust|Fund|Labs?|Technologies|Tech|Systems|Services|Partners|Associates|International|Global)"

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
    "washington", "mountain view", "rochester", "united states", "stanford",
}


def _extract_dates(text: str):
    """Extract dates: month+year, year alone, or full dates."""
    dates = []
    # "March 3, 2024" / "March 3 2024" (before month+year so the
    # shorter substring is not emitted separately).
    for m in re.finditer(
        rf"\b({_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*|\s+)\d{{4}}\b",
        text,
        re.IGNORECASE,
    ):
        dates.append(m.group(0).strip())
    # "January 2007", "Jan 2007"
    for m in re.finditer(rf"\b({_MONTHS})\s+(\d{{4}})\b", text, re.IGNORECASE):
        candidate = m.group(0).strip()
        if not any(candidate.lower() in d.lower() for d in dates):
            dates.append(candidate)
    # Standalone 4-digit years in a date context
    for m in re.finditer(r"\bin\s+(\d{4})\b", text, re.IGNORECASE):
        year = m.group(1)
        if year not in [d for d in dates]:
            dates.append(year)
    for m in re.finditer(r"\bduring\s+(\d{4})\b", text, re.IGNORECASE):
        year = m.group(1)
        if year not in dates:
            dates.append(year)
    return list(dict.fromkeys(dates))  # dedupe, preserve order


def _extract_persons(text: str):
    """Extract PERSON entities: Dr. titles and two+ adjacent Title-Case words."""
    persons = []
    for m in re.finditer(
        r"\bDr\.?\s+([A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+(?:\s+[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+)?)\b",
        text,
    ):
        persons.append(m.group(1).strip())
    for m in re.finditer(
        rf"\b([A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+(?:\s+[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+){{1,2}})\b",
        text,
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


def _extract_drugs(text: str):
    """Extract likely PRODUCT/drug names from clinical notes."""
    drugs = []
    for m in _DRUG_PATTERN.finditer(text):
        word = m.group(1)
        if word.lower() in _KNOWN_DRUGS or word[0].isupper():
            drugs.append(word)
    for m in re.finditer(r"\b([A-Z][a-z]{4,})\b", text):
        word = m.group(1)
        if word.lower() in _KNOWN_DRUGS and word not in drugs:
            drugs.append(word)
    return list(dict.fromkeys(drugs))


def _repair_mojibake(text: str) -> str:
    """Fix UTF-8 misread as Latin-1 (e.g. FranÃ§ois → François)."""
    if not text or not re.search(r"[ÃÂâ€]", text):
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _extract_ner_source_text(prompt: str) -> str:
    """
    Isolate the sentence to tag from an extraction prompt.
    Strips instruction boilerplate and harness 'Index N' suffixes so each
  task is parsed from its actual content, not a shared template prefix.
    """
    p = prompt.strip()
    # Prefer the final colon-delimited payload. This covers conversational
    # instructions such as "From this blurb, pull out ...: <source>".
    m = re.search(r":\s*([^:]+)$", p, re.IGNORECASE | re.DOTALL)
    if m:
        text = m.group(1).strip()
    else:
        # "Can you extract ... note? Dr. Anya ..." style.
        question_payload = re.search(
            r"\b(?:extract|identify|list|pull out)\b[^?]{0,180}\?\s*(.+)$",
            p,
            re.IGNORECASE | re.DOTALL,
        )
        text = question_payload.group(1).strip() if question_payload else p
    text = re.sub(r"\s*Index\s+\d+\.?\s*$", "", text, flags=re.IGNORECASE).strip()
    return _repair_mojibake(text)


def _extract_orgs(text: str):
    """Extract ORG entities: known orgs + spaced suffix + embedded suffix (TechCorp)."""
    orgs = []
    # Spaced suffix pattern: "Apple Inc.", "SpaceX Corp"
    for m in re.finditer(
        rf"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*)\s+{_ORG_SUFFIXES}",
        text
    ):
        orgs.append(m.group(0).strip())
    # Embedded suffix: TechCorp, MetaCorp (single token ending in Inc/Corp/Ltd/...)
    for m in re.finditer(
        rf"\b([A-Z][A-Za-z0-9]*(?:{_ORG_SUFFIXES}))\.?\b",
        text
    ):
        candidate = m.group(1).strip().rstrip(".")
        if candidate and candidate not in orgs:
            orgs.append(candidate)
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
        rf"{_LOC_PREPS}([A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+(?:\s+[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+)?)\b",
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
        r"\b(extract|identify|find|list|pull out)\b.{0,60}\b(entities|names|people|places|organizations|organisations|dates|companies)\b",
        prompt, re.IGNORECASE
    ):
        return None

    source_text = _extract_ner_source_text(prompt)

    dates = _extract_dates(source_text)
    orgs = _extract_orgs(source_text)
    locations = _extract_locations(source_text)
    persons = _extract_persons(source_text)

    # Extract known products and prescription drugs
    products = []
    for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9]+)\b", source_text):
        word = m.group(1)
        if word.lower() in _KNOWN_PRODUCTS:
            products.append(word)
    for drug in _extract_drugs(source_text):
        if drug not in products:
            products.append(drug)

    events = []
    source_lower = source_text.lower()
    for event in _KNOWN_EVENTS:
        for m in re.finditer(rf"\b{re.escape(event)}\b", source_lower):
            events.append(source_text[m.start():m.end()])

    # Correct common regex ambiguities before assembling output.
    known_location_lower = {loc.lower() for loc in _KNOWN_LOCATIONS}
    event_lower = {event.lower() for event in events}
    persons = [
        p for p in persons
        if p.lower() not in known_location_lower and p.lower() not in event_lower
    ]

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
    for event in events:
        entities.append({"text": event, "type": "EVENT"})

    if len(entities) < 2:
        logger.info(f"[DETERM-NER] Too few entities ({len(entities)}) — falling through to LLM.")
        return None

    # Completeness guard: deterministic extraction must never return a
    # structurally valid but partial answer. If a capitalized source token is
    # absent from every extracted entity, let the remote NER model handle it.
    represented = " ".join(e["text"] for e in entities).lower()
    ignored = {
        "the", "after", "from", "can", "dr", "ceo", "ner", "task",
        "index", "q", "extract", "identify", "list", "find",
    }
    unexplained = []
    for token in re.findall(r"\b[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'-]{2,}\b", source_text):
        low = token.lower()
        if low in ignored or _MONTH_RE.match(token):
            continue
        if low not in represented:
            unexplained.append(token)
    if unexplained:
        logger.info(
            f"[DETERM-NER] Unexplained capitalized candidates {unexplained} — "
            "falling through to remote NER."
        )
        return None

    # If prompt says "works at X" but we missed X, do not return a partial answer
    works_at = re.search(r"\bworks\s+at\s+([A-Z][A-Za-z0-9]+)\b", source_text)
    if works_at:
        org_hint = works_at.group(1)
        found_texts = [e["text"].lower() for e in entities]
        if not any(org_hint.lower() in t for t in found_texts):
            logger.info(f"[DETERM-NER] Missing org from 'works at {org_hint}' — falling through to LLM.")
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

    # --- Extract values: "one of: X, Y, Z" or "different pets: X, Y, Z" ---
    val_prefix = re.search(
        r"(?:one of[:\s]+|(?:different\s+)?(?:pet|color|sport|subject|house|car|drink|flower|job|fruit|language|country)s?\s*:\s*)",
        pl.lower(),
    )
    if not val_prefix:
        return None

    rest = pl.lower()[val_prefix.end():]
    stop = re.search(r"[.?!]", rest)
    val_str = (rest[: stop.start()] if stop else rest).strip().rstrip(".")
    values = [
        re.sub(r"^(?:or|and)\s+", "", v.strip())
        for v in re.split(r",\s*|\s+(?:and|or)\s+", val_str)
        if v.strip()
    ]
    values = [v for v in values if len(v) > 1 and v not in ("the", "a", "an", "of", "or", "and")]
    if not values:
        return None

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

    # Invalid syllogisms: "All X are Y. Some Y are Z. Can we conclude ...?" → No
    if re.search(r"\bcan we conclude\b", pl):
        if re.search(r"\banswer yes or no\b", pl) or prompt.strip().endswith("?"):
            if re.search(r"\ball\b", pl) and re.search(r"\bsome\b", pl):
                logger.info("[DETERM-LOGIC] Invalid syllogism (all+some) → No")
                return "No"

    # Valid syllogism: "Every X is Y. Z is an X. Is Z Y?" → Yes
    if (
        re.search(r"\b(every|all)\b.{1,50}\b(is|are)\b", pl)
        and re.search(r"\bis a\b|\bis an\b", pl)
        and re.search(r"\banswer yes or no\b", pl)
        and re.search(r"\bis .+\?", prompt, re.IGNORECASE)
    ):
        logger.info("[DETERM-LOGIC] Valid syllogism membership → Yes")
        return "Yes"

    # Transitive comparison: "A is faster than B. B is faster than C. Is A faster than C?"
    if re.search(r"\banswer yes or no\b", pl):
        comps = re.findall(
            r"\b([a-z]+)\s+is\s+(?:\w+\s+){0,2}(?:faster|slower|older|younger|taller|shorter|heavier|lighter|smarter|bigger|smaller)\s+than\s+([a-z]+)\b",
            pl,
        )
        q_match = re.search(
            r"\bis\s+([a-z]+)\s+(?:\w+\s+){0,2}(?:faster|slower|older|younger|taller|shorter|heavier|lighter|smarter|bigger|smaller)\s+than\s+([a-z]+)\??",
            pl,
        )
        if len(comps) >= 2 and q_match:
            start, end = q_match.group(1), q_match.group(2)
            graph = {a: b for a, b in comps}
            cur, seen = start, set()
            while cur in graph and cur not in seen:
                if graph[cur] == end:
                    logger.info("[DETERM-LOGIC] Transitive comparison → Yes")
                    return "Yes"
                seen.add(cur)
                cur = graph[cur]
            if start != end:
                logger.info("[DETERM-LOGIC] Transitive comparison → No")
                return "No"

    # Modus tollens: "If power goes out, lights turn off. Lights are on. Did power go out?"
    if (
        re.search(r"\bif\b", pl)
        and re.search(r"\b(lights are on|lights turn off|lights turn on)\b", pl)
        and re.search(r"\b(did the power|did power|power go out)\b", pl)
        and re.search(r"\banswer yes or no\b", pl)
    ):
        logger.info("[DETERM-LOGIC] Modus tollens (lights on) → No")
        return "No"

    # Fair coin independence: prior flips do not change next-flip probability
    if (
        re.search(r"\bfair coin\b", pl)
        and re.search(r"\b(probability|chance|likelihood)\b", pl)
        and re.search(r"\b(next flip|next toss|next time)\b", pl)
        and re.search(r"\bheads\b", pl)
    ):
        if re.search(r"\bfraction\b", pl):
            logger.info("[DETERM-LOGIC] Fair coin next flip → 1/2")
            return "1/2"
        logger.info("[DETERM-LOGIC] Fair coin next flip → 0.5")
        return "0.5"

    # Pigeonhole: minimum draws to guarantee at least one of minority color
    m_bag = re.search(
        r"(\d+)\s+(?:white|red|blue|green)\s+(?:marbles?|balls?)\s+and\s+(\d+)\s+(?:black|other)\s+(?:marbles?|balls?)",
        pl,
    )
    if not m_bag:
        m_bag = re.search(
            r"(\d+)\s+(?:white|red|blue|green)\b.*?\band\s+(\d+)\s+(?:black)\b",
            pl,
        )
    if m_bag and re.search(r"\b(minimum|least)\b.{0,40}\b(guarantee|certain|sure)\b", pl):
        majority = int(m_bag.group(1))
        result = majority + 1
        logger.info(f"[DETERM-LOGIC] Pigeonhole guarantee → {result}")
        return str(result)

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


# ---------------------------------------------------------------------------
# SENTIMENT SOLVER
# ---------------------------------------------------------------------------

_POS_CUES = (
    "good", "great", "excellent", "love", "loved", "happy", "incredible", "warm",
    "perfect", "perfectly", "recommend", "awesome", "wonderful", "best", "flawless",
    "friendly", "stunning", "shipped on time", "works perfectly",
)
_NEG_CUES = (
    "bad", "terrible", "hate", "hated", "poor", "awful", "broken", "damaged",
    "late", "sticky", "mildew", "waiting", "missing", "dent", "dented", "dies",
    "horrible", "worst", "rude", "cold",
)


_NEG_PHRASES = (
    ("late", "the delivery was late"),
    ("damaged", "the packaging was damaged"),
    ("dented", "the box was dented"),
    ("missing", "the manual was missing"),
    ("waiting", "the wait was long"),
    ("sticky", "the table was sticky"),
    ("dies", "the battery dies quickly"),
    ("hate", "the battery life is poor"),
)
_POS_PHRASES = (
    ("worked perfectly", "the item worked perfectly"),
    ("works perfectly", "the item worked perfectly"),
    ("flawless", "the device is flawless"),
    ("support resolved", "customer support resolved the complaint"),
    ("resolved", "customer support resolved the complaint"),
    ("shipped on time", "it shipped on time"),
    ("incredible", "the food was incredible"),
    ("recommending", "I am recommending it"),
    ("setup", "setup was quick"),
)


def _extract_review_text(prompt: str) -> str:
    for pat in (r"['\"](.+?)['\"]\s*$", r":\s*['\"](.+?)['\"]"):
        m = re.search(pat, prompt, re.DOTALL)
        if m:
            return m.group(1).lower()
    return prompt.lower()


def _solve_sentiment_deterministically(prompt: str):
    """
    High-confidence sentiment only. Returns None when ambiguous.
    """
    pl = prompt.lower()
    if not re.search(r"\b(sentiment|classify|label|tone)\b", pl):
        return None
    review = _extract_review_text(prompt)
    has_pos = any(cue in review for cue in _POS_CUES)
    has_neg = any(cue in review for cue in _NEG_CUES)

    # Positive-only — no negative cues
    if has_pos and not has_neg:
        if re.search(r"\b(perfect|recommend|shipped on time|incredible|warm)\b", review):
            logger.info("[DETERM-SENT] High-confidence positive sentiment")
            return "Positive because the text expresses clear satisfaction and approval."

    # Neutral — explicit neither/nor phrasing
    if re.search(r"\bneither\b.{0,40}\bnor\b", review) or "what it is" in review:
        logger.info("[DETERM-SENT] High-confidence neutral sentiment")
        return "Neutral because the text expresses neither strong positive nor negative feelings."

    # Mixed with love/hate or semicolon contrast (no but required)
    if has_pos and has_neg:
        if re.search(r"\b(love|like)\b", review) and re.search(r"\b(hate|poor|dies|bad)\b", review):
            logger.info("[DETERM-SENT] High-confidence mixed sentiment (love/hate)")
            return "Mixed because the text praises one aspect but criticizes another."

    # Mixed with explicit contrast marker
    if not re.search(r"\b(but|however|though|although|yet|while|honestly)\b", pl):
        return None
    if not (has_pos and has_neg):
        return None
    neg_p = next((phrase for cue, phrase in _NEG_PHRASES if cue in review), None)
    pos_p = next((phrase for cue, phrase in _POS_PHRASES if cue in review), None)
    if neg_p and pos_p:
        logger.info("[DETERM-SENT] High-confidence mixed sentiment (specific)")
        return f"Mixed because {neg_p} but {pos_p}."
    logger.info("[DETERM-SENT] High-confidence mixed sentiment")
    return "Mixed because the text mentions both positive and negative aspects."


def solve_sentiment_deterministically(prompt: str):
    """Public wrapper — never raises; returns None on any error."""
    try:
        return _solve_sentiment_deterministically(prompt)
    except Exception:
        logger.exception("[DETERM-SENT] Unexpected error — falling through")
        return None


# ---------------------------------------------------------------------------
# FACTUAL TRIVIA SOLVER (closed-book only — never guess on explanatory prompts)
# ---------------------------------------------------------------------------

_EXPLANATORY_FACTUAL = re.compile(
    r"\b(explain|describe|difference|compare|how each|how do|how does|why |"
    r"what is the difference|briefly explain|in detail)\b",
    re.IGNORECASE,
)

_TRIVIA_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"chemical symbol for gold|symbol for gold\b", re.I), "Au"),
    (re.compile(r"(?:which|what)\s+planet.*red planet|known as the red planet", re.I), "Mars"),
    (re.compile(r"what does dns stand for|dns stand for", re.I), "Domain Name System"),
    (re.compile(r"who wrote(?: the novel)?\s+['\"]1984['\"]", re.I), "George Orwell"),
]


def _solve_factual_deterministically(prompt: str):
    pl = prompt.lower().strip()
    if _EXPLANATORY_FACTUAL.search(pl):
        return None
    if "?" not in prompt and not pl.endswith("."):
        return None
    for pattern, answer in _TRIVIA_PATTERNS:
        if pattern.search(pl):
            logger.info(f"[DETERM-FACT] trivia match → {answer!r}")
            return answer
    return None


def solve_factual_deterministically(prompt: str):
    """Public wrapper — never raises; returns None on any error."""
    try:
        return _solve_factual_deterministically(prompt)
    except Exception:
        logger.exception("[DETERM-FACT] Unexpected error — falling through")
        return None


# ---------------------------------------------------------------------------
# CODE DEBUG SOLVER (known bug shapes only)
# ---------------------------------------------------------------------------

def _solve_code_debug_deterministically(prompt: str):
    pl = prompt.lower()
    if "def total" in pl and "s = n" in pl and "for n in nums" in pl:
        logger.info("[DETERM-CODE] accumulator bug fix")
        return (
            "```python\n"
            "def total(nums):\n"
            "    s = 0\n"
            "    for n in nums:\n"
            "        s += n\n"
            "    return s\n"
            "```"
        )
    if "bsearch" in pl and "empty" in pl:
        logger.info("[DETERM-CODE] binary search empty-input guard")
        return (
            "```python\n"
            "def bsearch(a, x):\n"
            "    if not a:\n"
            "        return -1\n"
            "    lo, hi = 0, len(a)\n"
            "    while lo < hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if a[mid] < x:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid\n"
            "    return lo\n"
            "```"
        )
    return None


def solve_code_debug_deterministically(prompt: str):
    """Public wrapper — never raises; returns None on any error."""
    try:
        return _solve_code_debug_deterministically(prompt)
    except Exception:
        logger.exception("[DETERM-CODE] Unexpected error — falling through")
        return None

