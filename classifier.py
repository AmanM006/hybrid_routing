import re
import logging

logger = logging.getLogger(__name__)

CATEGORIES = [
    "factual_knowledge",
    "math_reasoning",
    "sentiment_classification",
    "summarization",
    "named_entity_recognition",
    "code_debugging",
    "logical_reasoning",
    "code_generation"
]

# Quick regexes and keyword matchers
CODE_KEYWORDS = re.compile(
    r"\b(python|javascript|java|c\+\+|c#|html|css|sql|bash|shell|rust|golang|code|programming|script|function|class|method|develop|implement)\b", 
    re.IGNORECASE
)
DEBUG_KEYWORDS = re.compile(
    r"\b(debug|fix|bug|error|broken|fail|incorrect|syntax|traceback|exception|crash|wrong|compile|output matches)\b", 
    re.IGNORECASE
)
MATH_KEYWORDS = re.compile(
    r"\b(solve|equation|calculate|math|integral|derivative|algebra|geometry|trigonometry|matrix|probability|statistics|sum|product|fraction|percentage|ratio|arithmetic|plus|minus|multiplied|divided|equals|factorial|combination|permutation|exponent|square root|cube root|power of)\b", 
    re.IGNORECASE
)
LOGIC_KEYWORDS = re.compile(
    r"\b(puzzle|riddle|logic|deduce|conclude|premise|valid|sequence|pattern|grid|sudoku|knights|knaves|statement|truth|satisfy|rules|constraints|contradiction|if and only if|minimum number|guarantee|at least|no more than|possible|impossible|always|never|some|all|none)\b", 
    re.IGNORECASE
)
# Comparative reasoning patterns (X is older/taller/faster/bigger than Y → logical deduction)
COMPARATIVE_LOGIC = re.compile(
    r"\b(older|younger|taller|shorter|faster|slower|heavier|lighter|smarter|richer|bigger|smaller|more|less)\s+than\b",
    re.IGNORECASE
)
SENTIMENT_KEYWORDS = re.compile(
    r"\b(sentiment|classify sentiment|positive|negative|neutral|tone|feeling|angry|happy|sad|review sentiment|opinion)\b", 
    re.IGNORECASE
)
SUMMARIZE_KEYWORDS = re.compile(
    r"\b(summarize|summarise|summary|tl;dr|tldr|shorten|bullet points|gist|condense|main points|brief overview)\b", 
    re.IGNORECASE
)
NER_KEYWORDS = re.compile(
    r"\b(extract entities|ner|named entities|extract names|extract places|extract organizations|extract dates|identify people|locations|dates|entities)\b", 
    re.IGNORECASE
)
FACTUAL_KEYWORDS = re.compile(
    r"\b(what|who|where|when|why|how|which|define|explain|tell|list|describe|capital|president|ceo|born|date|year|history|fact|factual)\b",
    re.IGNORECASE
)

# Spelled-out small numbers for conversational word problems (e.g. "twelve apples...four left")
_NUMBER_WORDS = (
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred"
)

async def classify_prompt(prompt: str, local_llm_callable=None) -> str:
    """
    Classifies a prompt into one of the 8 categories using regex rules first, 
    with safety biases toward hard categories, and local LLM as a tie-breaker.
    """
    prompt_lower = prompt.lower()
    
    # High-priority logic puzzle check
    if any(x in prompt_lower for x in ["who owns", "different pet", "does not own", "knights and knaves", "logic puzzle"]) or "if all" in prompt_lower or ("is a" in prompt_lower and "always" in prompt_lower):
        return "logical_reasoning"
        
    # Comparative deductive reasoning (X older/taller than Y, therefore...)
    comparative_matches = COMPARATIVE_LOGIC.findall(prompt_lower)
    if len(comparative_matches) >= 2:  # Two comparisons → transitive deduction
        return "logical_reasoning"
        
    # Pigeonhole / minimum guarantee problems
    if re.search(r"minimum.*(?:guarantee|certain|sure)|guarantee.*minimum|must draw|to guarantee|minimum number", prompt_lower):
        return "logical_reasoning"

    # High-priority math — before propositional "if...then" (word problems often say "if someone bought...")
    has_math_pattern = (
        re.search(r"\b\d+\b.*?(?:percent|%|remain|remains|remaining|total|each|sells|items)\b", prompt_lower) or
        re.search(r"(?:percent|%|remain|remains|remaining|total|each|sells|items)\b.*?\b\d+\b", prompt_lower) or
        re.search(r"\d+%", prompt_lower) or
        re.search(r"\b\d+\b.*?\bper\b", prompt_lower) or
        re.search(r"\b(how long|how many|how much|how far|how fast)\b.*?\b\d+\b", prompt_lower) or
        (re.search(r"\b(average|mean)\b", prompt_lower) and re.search(r"\d", prompt)) or
        re.search(r"\bwhat is\s+[-\d].*?[+\-*/×÷]", prompt_lower) or
        re.search(r"\bwhat is\s+.*?\d+\s*[+\-*/×÷]\s*[-\d]", prompt_lower) or
        (
            re.search(r"\bhow many\b.{0,120}\b(?:left|remain|remaining)\b", prompt_lower)
            and (
                re.search(r"\d", prompt)
                or re.search(rf"\b(?:{_NUMBER_WORDS})\b", prompt_lower)
                or re.search(r"\b(?:gave away|gives away|subtract|minus|sold|bought)\b", prompt_lower)
            )
        ) or
        re.search(r"\b(squared|cubed|equals|equation|solve for|expression|variable|coefficient)\b", prompt_lower) or
        (re.search(r"\b(perimeter|area|volume|radius|diameter|circumference|hypotenuse)\b", prompt_lower) and re.search(r"\d", prompt)) or
        (re.search(r"\b(length|width|height|base)\b", prompt_lower) and re.search(r"\d+\s*(cm|m|km|ft|in|mm)", prompt_lower))
    )
    if has_math_pattern:
        return "math_reasoning"
        
    # Conditional/propositional logic: "if X then Y", "did it necessarily"
    # Also catches syllogism patterns: "Every X is Y. A Z is an X."
    # Also catches "lights are on/off" type consequent checking.
    # BUT exclude "neither/nor" sentiment phrasing — check sentiment cues first.
    has_sentiment_cue = bool(re.search(r"\b(sentiment|positive|negative|neutral|liked|disliked|enjoyed|hated|feeling|tone|review|opinion)\b", prompt_lower))
    # Don't apply conditional-logic check to code prompts (they contain 'if' inside function bodies)
    has_code_signal = ("def " in prompt or "```" in prompt or
                       bool(re.search(r"\b(function|return|python|implement|write a)\b", prompt_lower)))
    if not has_sentiment_cue and not has_code_signal:
        # Remove \b after period/comma — period is non-word char so \b never matches after it
        if re.search(r"\bif\b.{1,80}(?:\.|,|\bthen\b)|\bnecessarily\b|\bif and only if\b", prompt_lower):
            return "logical_reasoning"

    # Syllogism: "Every X is Y" or "All X are Y" followed by deductive question
    if re.search(r"\b(every|all)\b.{1,40}\b(is|are)\b", prompt_lower) and re.search(r"\b(is it|does it|will it|can it|answer yes|answer no)\b", prompt_lower):
        return "logical_reasoning"
    # "all ... are" (original pattern retained)
    if re.search(r"\ball .{1,40} are\b", prompt_lower):
        return "logical_reasoning"
    
    # Check for code blocks
    has_code_block = "```" in prompt
    
    # Compute keyword matching scores
    code_score = len(CODE_KEYWORDS.findall(prompt_lower))
    debug_score = len(DEBUG_KEYWORDS.findall(prompt_lower))
    math_score = len(MATH_KEYWORDS.findall(prompt_lower))
    
    # Exclude false-positive math match on commercial "product" in reviews if no digits exist
    if "product" in prompt_lower and not re.search(r"\d", prompt):
        math_score = max(0, math_score - len(re.findall(r"\bproduct\b", prompt_lower)))
        
    logic_score = len(LOGIC_KEYWORDS.findall(prompt_lower))
    sentiment_score = len(SENTIMENT_KEYWORDS.findall(prompt_lower))
    summarize_score = len(SUMMARIZE_KEYWORDS.findall(prompt_lower))
    ner_score = len(NER_KEYWORDS.findall(prompt_lower))
    
    # Safety bias: Check hard categories first
    # 1. Code Debugging: code present + debugging words (incl. bare def + Bug/Fix prompts)
    if (has_code_block or code_score > 0 or "def " in prompt) and debug_score > 0:
        return "code_debugging"
        
    # 2. Code Generation: contains code keywords or is asking to write a program/function/class
    if "write a" in prompt_lower or "write python" in prompt_lower or "implement a" in prompt_lower or "create a function" in prompt_lower:
        if code_score > 0 or has_code_block:
            return "code_generation"
            
    # 3. Math Reasoning: clear math keywords or mathematical expressions
    math_symbol_match = re.search(r"[\d\+\-\*\/=\^]+", prompt)
    if (math_score >= 2 or
        (math_score >= 1 and math_symbol_match and len(math_symbol_match.group(0)) > 2) or
        (math_score >= 1 and re.search(r"\d", prompt))):
        return "math_reasoning"
        
    # 4. Logical Reasoning: logic constraints or puzzle context
    # Don't let 'statement' alone beat sentiment cues
    has_sentiment_cue2 = bool(re.search(r"\b(sentiment|positive|negative|neutral|liked|disliked|enjoyed|hated|feeling|tone|review|opinion|classify)\b", prompt_lower))
    if has_sentiment_cue2:
        logic_score = max(0, logic_score - len(re.findall(r"\bstatement\b", prompt_lower)))
    if logic_score >= 2 or (logic_score >= 1 and ("truth" in prompt_lower or "puzzle" in prompt_lower or "deduce" in prompt_lower)):
        return "logical_reasoning"
        
    # Let's check easy categories
    # 5. Sentiment
    if sentiment_score >= 1 or "sentiment" in prompt_lower:
        return "sentiment_classification"
        
    # 6. Summarization
    if summarize_score >= 1 or "summarize" in prompt_lower or "summarise" in prompt_lower:
        return "summarization"
        
    # 7. NER
    if ner_score >= 1 or "extract entities" in prompt_lower or "extract names" in prompt_lower:
        return "named_entity_recognition"

    # 8. Factual Knowledge
    factual_score = len(FACTUAL_KEYWORDS.findall(prompt_lower))
    if factual_score >= 1 or "factual" in prompt_lower:
        return "factual_knowledge"

    # Catch remaining code generation
    if code_score >= 2 or (code_score >= 1 and "write" in prompt_lower):
        return "code_generation"

    # If we have any trace of hard categories, fail towards safety!
    if code_score > 0:
        return "code_generation"
    if math_score > 0:
        return "math_reasoning"
    if logic_score > 0:
        return "logical_reasoning"

    # If we have a local model callable, use it as a tie-breaker
    if local_llm_callable:
        try:
            tie_break_prompt = (
                "Classify the following text into exactly one of these categories: "
                "factual_knowledge, math_reasoning, sentiment_classification, summarization, "
                "named_entity_recognition, code_debugging, logical_reasoning, code_generation. "
                "Only return the category name, nothing else.\n\n"
                f"Text: {prompt}\n\nCategory:"
            )
            raw_resp = await local_llm_callable(tie_break_prompt, max_tokens=15)
            response = raw_resp.strip().lower()
            for cat in CATEGORIES:
                if cat in response:
                    logger.info(f"Local LLM tie-breaker classified prompt as: {cat}")
                    return cat
        except Exception as e:
            logger.warning(f"Local LLM tie-breaker failed: {e}")

    # Factual Knowledge is the default fallback for general queries
    return "factual_knowledge"
