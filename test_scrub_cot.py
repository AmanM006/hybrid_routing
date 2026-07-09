import sys
sys.path.insert(0, '.')
from client import FireworksClient

c = FireworksClient.__new__(FireworksClient)

PASS = lambda ok: 'OK' if ok else 'FAIL'
failures = []

def check(name, condition, detail=''):
    result = 'OK' if condition else 'FAIL'
    print(f"  {result} {name}" + (f": {detail}" if detail else ''))
    if not condition:
        failures.append(name)

print("=== CHECK 3: code block extracted, CoT preamble dropped ===")
code_cot = ('The user wants me to write this.\nI need to think carefully.\n'
            '```python\ndef get_min(nums):\n    return min(nums)\n```\n'
            'This is the best approach because min() is O(n).')
for cat in ['code_debugging', 'code_generation']:
    result = c._scrub_cot(code_cot, cat)
    check(f'[{cat}] code block preserved', '```python' in result and 'get_min' in result)
    check(f'[{cat}] CoT preamble dropped', 'user wants' not in result.lower(), repr(result[:80]))
    check(f'[{cat}] trailing analysis dropped', 'best approach' not in result, repr(result[:80]))

print()
print("=== CHECK 4: Sentiment CoT preamble stripped, justification survives ===")
sent_preamble = "The user wants me to classify this.\nNegative. The text expresses strong dissatisfaction with harsh words like hate and awful."
result = c._scrub_cot(sent_preamble, 'sentiment_classification')
check('Preamble stripped', 'user wants' not in result.lower())
check('Label preserved', 'negative' in result.lower())
check('>=4 words remain', len(result.split()) >= 4, repr(result))

print()
print("=== CHECK: instruction leak stripped from sentiment ===")
# Real kimi production output: single line with instruction appended after period
leaked = ('The text is clearly positive - "incredibly helpful" and "went above and beyond expectations" '
          'are strongly positive phrases. The user also says "Answer only.')
result = c._scrub_cot(leaked, 'sentiment_classification')
check('Instruction leak removed', 'answer only' not in result.lower() and 'user also' not in result.lower(), repr(result[:120]))
check('Label preserved', 'positive' in result.lower())

print()
print("=== CHECK: math Answer: prefix stripped ===")
math_ans = "Answer: 225"
result = c._scrub_cot(math_ans, 'math_reasoning')
check("Answer: stripped", result == '225', repr(result))

logic_ans = "Answer: 6"
result = c._scrub_cot(logic_ans, 'logical_reasoning')
check("Answer: stripped from logic", result == '6', repr(result))

print()
print("=== CHECK: short factual answers not damaged ===")
for short in ["$30", "3", "Yes", "100°C", "Paris", "Mercury"]:
    result = c._scrub_cot(short, 'factual_knowledge')
    check(f'[{short!r}] preserved', result.strip() == short.strip(), repr(result))

print()
print("=== CHECK: NER JSON not stripped ===")
ner_resp = '{"entities": [{"text": "Sundar Pichai", "type": "PERSON"}]}'
result = c._scrub_cot(ner_resp, 'named_entity_recognition')
check('JSON preserved', 'entities' in result)

print()
print("=== CHECK: markdown-wrapped NER accepted ===")
ner_md = '```json\n{"entities": [{"text": "Elon Musk", "type": "PERSON"}]}\n```'
result = c._scrub_cot(ner_md, 'named_entity_recognition')
check('JSON extracted from markdown', 'entities' in result)

print()
print("=== CHECK: kimi instruction repetition stripped ===")
kimi_cot = """The user wants me to classify the sentiment of the text: I hate the design.

The text contains strongly negative words: hate, awful, cheap. This is clearly negative sentiment.

The user also says Answer only. But the"""
result = c._scrub_cot(kimi_cot, 'sentiment_classification')
check('User preamble stripped', 'user wants' not in result.lower())
check('The user also stripped', 'user also' not in result.lower())
print(f"  Result: {result[:200]}")

print()
total = 20  # approximate
print(f"=== {len(failures)} failure(s): {failures if failures else 'none'} ===")
