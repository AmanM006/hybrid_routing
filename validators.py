import re
import json
import ast
import logging

logger = logging.getLogger(__name__)

def validate_ner(output: str) -> bool:
    """
    named_entity_recognition: output must be valid JSON matching
    {"entities": [{"text": "...", "type": "PERSON|ORG|LOCATION|DATE|..."}]}.
    Reject if JSON doesn't parse, entities key missing, or any entry missing text/type.
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
                
        return True
    except Exception as e:
        logger.warning(f"NER Validation Exception: {e}")
        return False

def validate_sentiment(prompt: str, output: str) -> bool:
    """
    sentiment_classification: output label must be one of the expected set
    plus non-empty justification if requested. Reject if label missing/unrecognized.
    """
    prompt_lower = prompt.lower()
    output_lower = output.lower()
    
    # 1. Determine the expected labels
    expected_labels = set()
    # Try to extract options from parenthesis like (positive/negative/neutral) or [positive, negative]
    options_match = re.findall(r"\(([^)]+)\)", prompt_lower)
    for opt in options_match:
        if "/" in opt or "|" in opt:
            parts = [p.strip() for p in re.split(r'[/|]', opt)]
            if all(p.replace(" ", "").isalpha() for p in parts):
                expected_labels.update(parts)
                
    if not expected_labels:
        # Default options
        expected_labels = {"positive", "negative", "neutral"}
        
    # 2. Check if a label is present as a distinct word in the output
    label_found = None
    for label in expected_labels:
        if re.search(rf"\b{re.escape(label)}\b", output_lower):
            label_found = label
            break
            
    if not label_found:
        logger.warning(f"Sentiment Validation Failed: No expected label {expected_labels} found in output.")
        return False
        
    # 3. Always require a non-trivial justification (at least 4 words total including the label)
    words = output.strip().split()
    if len(words) < 4:
        logger.warning("Sentiment Validation Failed: Justification not provided alongside the label.")
        return False
            
    return True

def validate_summarization(prompt: str, output: str) -> bool:
    """
    summarization: output must be non-empty, respect explicit length/format constraint in prompt.
    Reject if constraint clearly violated or output is empty/degenerate (e.g., repeats prompt).
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
    
    # Extract word count constraint e.g., "max 50 words", "under 30 words"
    word_limit_match = re.search(r"(\d+)\s*words?\s*(?:or less|limit|max|cap)?", prompt_lower)
    if not word_limit_match:
        word_limit_match = re.search(r"(?:max|limit|under|at most)\s*(\d+)\s*words?", prompt_lower)
        
    if word_limit_match:
        limit = int(word_limit_match.group(1))
        word_count = len(output_clean.split())
        # Give a small 10% + 5 words buffer
        allowed_max = limit + max(5, int(limit * 0.10))
        if word_count > allowed_max:
            logger.warning(f"Summarization Validation Failed: Word count {word_count} exceeds limit {limit} (allowed max: {allowed_max}).")
            return False
            
    # Extract sentence count constraint e.g., "exactly 3 sentences", "in 2 sentences"
    sentence_limit_match = re.search(r"(\d+)\s*sentences?", prompt_lower)
    if sentence_limit_match:
        limit = int(sentence_limit_match.group(1))
        # Simple sentence splitter
        sentences = [s for s in re.split(r"[.!?]\s+", output_clean) if s.strip()]
        # Allow small buffer for sentence counts (e.g. limit + 1)
        if len(sentences) > limit + 1:
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
    """
    if "```" not in output:
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
        return validate_ner(output)
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
