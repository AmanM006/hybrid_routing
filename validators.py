import re
import json
import ast
import logging

logger = logging.getLogger(__name__)

_COUNT_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _parse_count(token: str) -> int | None:
    token = token.lower()
    if token.isdigit():
        return int(token)
    return _COUNT_WORDS.get(token)

# Unicode-aware word token for international names/places (François, München, José)
_U_WORD = r"[\w\u00C0-\u024F\u1E00-\u1EFF]"


def extract_ner_json(output: str) -> str | None:
    """Extract and re-serialize a valid NER JSON object, preserving Unicode."""
    if not output or not output.strip():
        return None
    stripped = re.sub(r"^```(?:json)?\s*", "", output.strip(), flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped.strip())
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        data = json.loads(stripped[start : end + 1])
        if isinstance(data, dict) and "entities" in data:
            return json.dumps(data, ensure_ascii=False)
    except json.JSONDecodeError:
        pass
    return None


def repair_ner_output(output: str) -> str | None:
    """
    Convert model output (JSON, markdown JSON, or bullet-list) into canonical NER JSON.
    Handles non-ASCII entity text (François, München, etc.).
    """
    if not output or not output.strip():
        return None

    extracted = extract_ner_json(output)
    if extracted and validate_ner(extracted):
        return extracted

    text = output.strip()
    entities = []

    # "**People:** Sundar Pichai\n**Companies:** Google, Alphabet" and
    # equivalent plain headings. Some instruction models ignore JSON mode but
    # still provide a complete, typed extraction in this predictable format.
    heading_types = {
        "people": "PERSON", "persons": "PERSON",
        "companies": "ORG", "organizations": "ORG", "organisations": "ORG",
        "places": "LOCATION", "locations": "LOCATION",
        "dates": "DATE", "events": "EVENT", "products": "PRODUCT",
    }
    for m in re.finditer(
        r"(?im)^\s*(?:\*\*)?(people|persons|companies|organizations|organisations|"
        r"places|locations|dates|events|products)(?:\*\*)?\s*:\s*(.+?)\s*$",
        text,
    ):
        etype = heading_types[m.group(1).lower()]
        for value in re.split(r"\s*,\s*|\s*;\s*", m.group(2)):
            value = value.strip().strip("*").strip()
            if value:
                entities.append({"text": value, "type": etype})

    # "- François Mitterrand: PERSON" / "- Name : TYPE"
    for m in re.finditer(
        rf"(?m)^[-*•]\s*(.+?)\s*[:：]\s*([A-Za-z][A-Za-z_/]*)\s*$",
        text,
    ):
        entities.append({"text": m.group(1).strip(), "type": m.group(2).strip().upper()})

    # "- François Mitterrand (PERSON)"
    for m in re.finditer(
        rf"(?m)^[-*•]\s*(.+?)\s*\(([A-Za-z][A-Za-z_/]*)\)\s*$",
        text,
    ):
        entities.append({"text": m.group(1).strip(), "type": m.group(2).strip().upper()})

    # "Name (person)" inline comma-separated (adv-x01 style fragments)
    for m in re.finditer(
        r"([\w\u00C0-\u024F][\w\u00C0-\u024F\s.'-]{0,60}?)\s*\((person|company|location|org|date|misc)\)",
        text,
        re.IGNORECASE,
    ):
        etype = m.group(2).upper()
        if etype == "COMPANY":
            etype = "ORG"
        if etype == "PERSON":
            etype = "PERSON"
        entities.append({"text": m.group(1).strip(), "type": etype})

    if entities:
        # Deduplicate by text (case-insensitive)
        seen = set()
        unique = []
        for e in entities:
            key = e["text"].lower()
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return json.dumps({"entities": unique}, ensure_ascii=False)

    return None


def coerce_ner_output(output: str, prompt: str = "") -> tuple[str | None, bool]:
    """Return (normalized_json_or_best_text, passes_validation)."""
    if not output or not output.strip():
        return None, False
    repaired = repair_ner_output(output)
    if repaired and validate_ner(repaired, prompt):
        return repaired, True
    extracted = extract_ner_json(output)
    if extracted and validate_ner(extracted, prompt):
        return extracted, True
    return repaired, False


def validate_ner(output: str, prompt: str = "") -> bool:
    """
    named_entity_recognition: output must be valid JSON matching
    {"entities": [{"text": "...", "type": "PERSON|ORG|LOCATION|DATE|..."}]}.
    Reject if JSON doesn't parse, entities key missing, or any entry missing text/type.
    Also performs entity-count sanity check: if prompt has many proper nouns but
    entities list is suspiciously sparse, reject to try next tier.
    """
    try:
        # Strip markdown code block fences if present (e.g. ```json ... ```)
        stripped = re.sub(r"^```(?:json)?\s*", "", output.strip(), flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped.strip())
        # Try to find a JSON object block in case there's preamble
        start = stripped.find('{')
        end = stripped.rfind('}')
        if start == -1 or end == -1 or end < start:
            logger.warning("NER Validation Failed: No curly braces found.")
            return False

        
        json_str = stripped[start:end+1]
        data = json.loads(json_str)
        
        if not isinstance(data, dict):
            logger.warning("NER Validation Failed: Output is not a JSON dict.")
            return False
            
        if "entities" not in data:
            logger.warning("NER Validation Failed: 'entities' key missing.")
            return False
            
        entities = data["entities"]
        if not isinstance(entities, list):
            logger.warning("NER Validation Failed: 'entities' is not a list.")
            return False
            
        for idx, entry in enumerate(entities):
            if not isinstance(entry, dict):
                logger.warning(f"NER Validation Failed: Entry {idx} is not a dict.")
                return False
            if "text" not in entry or "type" not in entry:
                logger.warning(f"NER Validation Failed: Entry {idx} is missing 'text' or 'type'.")
                return False
            if not isinstance(entry["text"], str) or not isinstance(entry["type"], str):
                logger.warning(f"NER Validation Failed: Entry {idx} has non-string 'text' or 'type'.")
                return False

        # Entity-count sanity check: count proper-noun candidates in prompt
        # If prompt has many capitalized multi-word names but entities list is very sparse, reject.
        if prompt:
            # Count Title-Case word sequences (crude proper-noun proxy)
            proper_noun_candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", prompt)
            # Filter out sentence-start words (words after period/colon)
            proper_noun_candidates = [
                w for w in proper_noun_candidates
                if not re.match(r'^(The|A|An|This|That|It|In|At|On|For|To|Of|And|But|Or|Is|Was|He|She|They|We|You|I)$', w)
            ]
            n_candidates = len(proper_noun_candidates)
            n_entities = len(entities)
            # Reject if source has 3+ proper-noun candidates but we returned only 1 entity
            if n_candidates >= 3 and n_entities <= 1:
                logger.warning(
                    f"NER Validation Failed: Entity-count mismatch — {n_candidates} proper-noun "
                    f"candidates in prompt but only {n_entities} entities returned."
                )
                return False
                
        return True
    except Exception as e:
        logger.warning(f"NER Validation Exception: {e}")
        return False

def validate_sentiment(prompt: str, output: str) -> bool:
    """
    sentiment_classification: accept outputs that contain a recognisable sentiment
    label AND a brief justification (4+ words with a reason keyword).
    Judge needs justification — single-word labels fail the harness.
    """
    output_lower = output.lower().strip()
    if not output_lower:
        logger.warning("Sentiment Validation Failed: Output is empty.")
        return False

    # Must contain a known sentiment label
    SENTIMENT_WORDS = {
        "positive", "negative", "neutral", "mixed",
        "good", "bad", "great", "poor", "excellent",
        "satisfied", "dissatisfied", "happy", "unhappy",
    }
    has_label = any(
        re.search(rf"\b{re.escape(word)}\b", output_lower)
        for word in SENTIMENT_WORDS
    )
    if not has_label:
        logger.warning("Sentiment Validation Failed: No sentiment label found.")
        return False

    # Require justification: >= 4 words AND contains a reason keyword (matching grader)
    word_count = len(output_lower.split())
    has_reason = bool(re.search(
        r"\b(because|since|due to|as the|given that|although|but)\b",
        output_lower
    ))
    if word_count < 4 or not has_reason:
        logger.warning(
            f"Sentiment Validation Failed: Output does not meet justification criteria (words={word_count}, has_reason={has_reason})."
        )
        return False

    return True

def _extract_bullets(text: str) -> list[str]:
    """Return bullet item texts from line-based bullet output."""
    bullets = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^[-*•]\s+(.+)$", line) or re.match(r"^\d+[.)]\s+(.+)$", line)
        if match:
            bullets.append(match.group(1).strip())
    return bullets

def validate_summarization(prompt: str, output: str) -> bool:
    """
    summarization: output must be non-empty, respect explicit length/format constraint in prompt.
    Reject if constraint clearly violated or output is empty/degenerate (e.g., repeats prompt).
    Hard-rejects output exceeding 2x the stated word limit (prevents rambling).
    """
    output_clean = output.strip()
    if not output_clean:
        logger.warning("Summarization Validation Failed: Output is empty.")
        return False
        
    # Check if it is a degenerate repeat of the prompt
    if output_clean.lower() == prompt.strip().lower():
        logger.warning("Summarization Validation Failed: Output is an exact duplicate of the prompt.")
        return False
    if len(prompt.strip()) > 20 and output_clean in prompt.strip():
        logger.warning("Summarization Validation Failed: Output is a substring of the prompt.")
        return False
            
    prompt_lower = prompt.lower()

    per_bullet_word_match = re.search(
        r"\b(?:max(?:imum)?|at most|up to|under|no longer than)\s+(\d+)\s+words?\s+each\b",
        prompt_lower,
    )
    if not per_bullet_word_match:
        per_bullet_word_match = re.search(
            r"\beach\s+(?:no longer than|under|at most|max(?:imum)?|up to)?\s*(\d+)\s+words?\b",
            prompt_lower,
        )
    bullet_count_exact = re.search(
        r"\bexactly\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+bullet\s*points?\b",
        prompt_lower,
    )
    bullet_count_loose = None
    if not bullet_count_exact:
        bullet_count_loose = re.search(
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+bullet\s*points?\b",
            prompt_lower,
        )

    if per_bullet_word_match or bullet_count_exact or bullet_count_loose:
        bullets = _extract_bullets(output_clean)
        expected = None
        if bullet_count_exact:
            expected = _parse_count(bullet_count_exact.group(1))
        elif bullet_count_loose:
            expected = _parse_count(bullet_count_loose.group(1))
        if expected is not None and len(bullets) != expected:
            logger.warning(
                f"Summarization Validation Failed: Expected exactly {expected} bullet points, got {len(bullets)}."
            )
            return False
        if per_bullet_word_match:
            per_limit = int(per_bullet_word_match.group(1))
            if not bullets:
                logger.warning("Summarization Validation Failed: Expected bullet-point format.")
                return False
            for idx, bullet in enumerate(bullets, 1):
                bullet_words = len(bullet.split())
                if bullet_words > per_limit:
                    logger.warning(
                        f"Summarization Validation Failed: Bullet {idx} has {bullet_words} words, exceeds per-bullet limit {per_limit}."
                    )
                    return False
    else:
        # Extract word count constraint e.g., "max 50 words", "under 30 words"
        word_limit_match = re.search(r"(\d+)\s*words?\s*(?:or less|limit|max|cap)?", prompt_lower)
        if not word_limit_match:
            word_limit_match = re.search(r"(?:max|limit|under|at most)\s*(\d+)\s*words?", prompt_lower)

        if word_limit_match:
            limit = int(word_limit_match.group(1))
            word_count = len(output_clean.split())
            # Hard cap: reject if > 2× limit (prevents rambling)
            hard_cap = limit * 2
            if word_count > hard_cap:
                logger.warning(f"Summarization Validation Failed: Word count {word_count} exceeds 2x hard cap {hard_cap}.")
                return False
            # Soft cap: give a small 10% + 5 words buffer
            allowed_max = limit + max(5, int(limit * 0.10))
            if word_count > allowed_max:
                logger.warning(f"Summarization Validation Failed: Word count {word_count} exceeds limit {limit} (allowed max: {allowed_max}).")
                return False
            
    # Extract sentence count constraint e.g., "exactly 3 sentences", "in 2 sentences"
    sentence_limit_match = re.search(
        r"\b(?:exactly\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+sentences?\b",
        prompt_lower,
    )
    if sentence_limit_match:
        limit = _parse_count(sentence_limit_match.group(1))
        if limit is not None:
            sentences = [s for s in re.split(r"(?<=[.!?])\s+", output_clean) if s.strip()]
            exact_requested = bool(re.search(
                r"\bexactly\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+sentences?\b",
                prompt_lower,
            ))
            if exact_requested and len(sentences) != limit:
                logger.warning(
                    f"Summarization Validation Failed: Expected exactly {limit} sentences, got {len(sentences)}."
                )
                return False
            if not exact_requested and len(sentences) > limit:
                logger.warning(f"Summarization Validation Failed: Sentence count {len(sentences)} exceeds limit {limit}.")
                return False
            
    return True

def validate_factual(prompt: str, output: str) -> bool:
    """
    factual_knowledge: output must be non-empty, non-degenerate (not echoing question),
    minimum reasonable length. Short precise answers (numbers, currency, Yes/No)
    are valid and should never be rejected purely for brevity.
    """
    output_clean = output.strip()
    if not output_clean:
        logger.warning("Factual Validation Failed: Output is empty.")
        return False
        
    if output_clean.lower() == prompt.strip().lower():
        logger.warning("Factual Validation Failed: Output matches prompt exactly.")
        return False
    if len(prompt.strip()) > 20 and output_clean in prompt.strip():
        logger.warning("Factual Validation Failed: Output is a substring of the prompt.")
        return False
            
    # Allow any non-empty answer — short answers like "$30", "3", "Yes", "100°C"
    # are valid factual responses. Only reject truly empty strings (checked above).
    return True

def _extract_numeric_answer(text: str):
    """Pull the most likely final numeric value from a math answer string."""
    cleaned = re.sub(r"^[Aa]nswer:\s*", "", text.strip())
    # Prefer explicit Answer: line
    m = re.search(r"(?i)answer:\s*(-?\d+(?:\.\d+)?)", cleaned)
    if m:
        return float(m.group(1))
    nums = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if nums:
        return float(nums[-1])
    return None


def verify_math_self_consistency(prompt: str, answer: str) -> bool:
    """
    Plug-back sanity check for LLM math answers (no API call).
    Returns True if consistent OR not verifiable; False only on clear mismatch.
    """
    try:
        got = _extract_numeric_answer(answer)
        if got is None:
            return True
        pl = prompt.lower()

        def _close(expected: float) -> bool:
            if expected == 0:
                return abs(got) < 1e-6
            return abs(got - expected) <= max(1e-6, abs(expected) * 0.02)

        m = re.search(
            r"(?:what\s+is|calculate|compute|find|evaluate|solve)[\s:]*"
            r"(-?\d+(?:\.\d+)?)\s*([+\-*/×÷])\s*(-?\d+(?:\.\d+)?)",
            pl,
        )
        if m:
            a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
            ops = {"+": a + b, "-": a - b, "*": a * b, "×": a * b, "/": a / b if b else None, "÷": a / b if b else None}
            expected = ops.get(op)
            if expected is not None and not _close(expected):
                logger.warning(f"Math self-check FAIL: {a}{op}{b}={expected}, got {got}")
                return False

        m = re.search(r"(-?\d+(?:\.\d+)?)\s*%\s*of\s*(-?\d+(?:\.\d+)?)", pl)
        if m:
            expected = float(m.group(1)) / 100.0 * float(m.group(2))
            if not _close(expected):
                logger.warning(f"Math self-check FAIL: {m.group(1)}% of {m.group(2)}={expected}, got {got}")
                return False

        m = re.search(r"(-?\d*)\s*x\s*([+\-])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)", pl)
        if m and not re.search(r"(\^|\*\*|d[yx]/d)", pl):
            a = float(m.group(1)) if m.group(1).strip() else 1.0
            if a != 0:
                b, c = float(m.group(3)), float(m.group(4))
                expected = (c - b) / a if m.group(2) == "+" else (c + b) / a
                if not _close(expected):
                    logger.warning(f"Math self-check FAIL: linear eq expected {expected}, got {got}")
                    return False

        return True
    except Exception as e:
        logger.warning(f"Math self-check skipped due to error: {e}")
        return True


def validate_reasoning(output: str) -> bool:
    """
    math_reasoning / logical_reasoning: output must be non-empty and of reasonable length.
    """
    output_clean = output.strip()
    if not output_clean:
        logger.warning("Reasoning Validation Failed: Output is empty.")
        return False
    if len(output_clean) < 1:
        logger.warning("Reasoning Validation Failed: Output too short.")
        return False
    return True

def validate_code(prompt: str, output: str, is_debugging: bool = False) -> bool:
    """
    code_debugging / code_generation: output must contain a code block;
    for debugging, must differ meaningfully from input snippet;
    run basic syntax check if feasible.
    Also accepts bare Python without ``` fences (auto-detected by ast.parse).
    """
    # If no code fence, check if the raw output is valid Python and auto-wrap it
    if "```" not in output:
        output_stripped = output.strip()
        is_python_candidate = ("def " in output_stripped or "import " in output_stripped
                               or "return " in output_stripped or "class " in output_stripped)
        if is_python_candidate:
            try:
                ast.parse(output_stripped)
                # Valid bare Python — treat as if it were wrapped
                output = f"```python\n{output_stripped}\n```"
                logger.info("Code Validation: bare Python detected, treating as valid (no fences).")
            except SyntaxError:
                logger.warning("Code Validation Failed: Markdown code block wrapper '```' missing.")
                return False
        else:
            logger.warning("Code Validation Failed: Markdown code block wrapper '```' missing.")
            return False

    blocks = re.findall(r"```(?:\w+)?\n(.*?)\n```", output, re.DOTALL)
    if not blocks:
        logger.warning("Code Validation Failed: Empty code block.")
        return False

    code_content = "\n".join(blocks).strip()
    if not code_content:
        logger.warning("Code Validation Failed: Code block content is blank.")
        return False

    if is_debugging:
        # Extract code from prompt if any
        prompt_blocks = re.findall(r"```(?:\w+)?\n(.*?)\n```", prompt, re.DOTALL)
        if prompt_blocks:
            prompt_code = "\n".join(prompt_blocks).strip()
            if code_content == prompt_code:
                logger.warning("Code Validation Failed: Debugging output matches prompt snippet exactly.")
                return False

    # Run AST check if it is python code
    is_python = "python" in prompt.lower() or "def " in code_content or "import " in code_content or "print(" in code_content
    if is_python:
        try:
            ast.parse(code_content)
        except SyntaxError as e:
            logger.warning(f"Code Validation Failed: Python syntax error: {e}")
            return False

    return True

def validate_category_output(category: str, prompt: str, output: str) -> bool:
    """
    Validates output according to its category rules.
    """
    if not output or not output.strip():
        logger.warning("Validation Failed: Output is empty.")
        return False
        
    # Check for word loops/repetition (same word repeated 4+ times)
    words = output.lower().split()
    repetition_count = 0
    last_word = None
    for w in words:
        if w == last_word:
            repetition_count += 1
            if repetition_count >= 3:
                logger.warning(f"Validation Failed: Heavy repetition detected ('{w}').")
                return False
        else:
            repetition_count = 0
            last_word = w

    if category == "named_entity_recognition":
        # Pass prompt for entity-count sanity check
        repaired = repair_ner_output(output)
        if repaired and validate_ner(repaired, prompt):
            return True
        extracted = extract_ner_json(output)
        if extracted and validate_ner(extracted, prompt):
            return True
        return False
    elif category == "sentiment_classification":
        return validate_sentiment(prompt, output)
    elif category == "summarization":
        return validate_summarization(prompt, output)
    elif category == "factual_knowledge":
        return validate_factual(prompt, output)
    elif category in ["math_reasoning", "logical_reasoning"]:
        return validate_reasoning(output)
    elif category == "code_generation":
        return validate_code(prompt, output, is_debugging=False)
    elif category == "code_debugging":
        return validate_code(prompt, output, is_debugging=True)
    return True
