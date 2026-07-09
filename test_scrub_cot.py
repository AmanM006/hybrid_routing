import sys
sys.path.insert(0, '.')
from client import FireworksClient

c = FireworksClient.__new__(FireworksClient)

print("=== CHECK 3: _scrub_cot does NOT damage code responses ===")
code_resp = 'The user wants me to fix this.\nLet me think about it.\n```python\ndef get_min(nums):\n    return min(nums)\n```'
for cat in ['code_debugging', 'code_generation']:
    result = c._scrub_cot(code_resp, cat)
    ok = '```python' in result and 'get_min' in result
    print(f"  [{cat}] code block preserved: {'OK' if ok else 'FAIL'}")
    if not ok:
        print("  Got:", result[:150])

print()
print("=== CHECK 4: Sentiment CoT preamble stripped, justification survives ===")
sent_preamble = "The user wants me to classify this.\nNegative. The text expresses strong dissatisfaction with harsh words like hate and awful."
result = c._scrub_cot(sent_preamble, 'sentiment_classification')
has_label = 'negative' in result.lower()
has_just = len(result.split()) >= 4
stripped = 'user wants' not in result.lower()
print(f"  Preamble stripped: {'OK' if stripped else 'FAIL'}")
print(f"  Label preserved:   {'OK' if has_label else 'FAIL'}")
print(f"  >=4 words remain:  {'OK' if has_just else 'FAIL'}")
print(f"  Result: {result[:200]}")

print()
print("=== Short factual answer not damaged ===")
for short in ["$30", "3", "Yes", "100C", "Paris"]:
    result = c._scrub_cot(short, 'factual_knowledge')
    ok = result.strip() == short.strip()
    print(f"  [{short!r}] -> [{result!r}] : {'OK' if ok else 'FAIL'}")

print()
print("=== NER JSON not stripped ===")
ner_resp = '{"entities": [{"text": "Sundar Pichai", "type": "PERSON"}]}'
result = c._scrub_cot(ner_resp, 'named_entity_recognition')
print(f"  JSON preserved: {'OK' if 'entities' in result else 'FAIL'}")

print()
print("=== kimi chain-of-thought IS stripped for sentiment ===")
kimi_cot = """The user wants me to classify the sentiment of the text: I hate the design.

The text contains strongly negative words: hate, awful, cheap. This is clearly negative sentiment.

The user also says Answer only. But the"""
result = c._scrub_cot(kimi_cot, 'sentiment_classification')
print(f"  User preamble stripped: {'OK' if 'user wants' not in result.lower() else 'FAIL'}")
print(f"  Result: {result[:200]}")

print()
print("=== Legitimate multi-line factual answer not over-stripped ===")
factual_multi = "Alexander Graham Bell invented the telephone in 1876.\nHe was born in Edinburgh, Scotland."
result = c._scrub_cot(factual_multi, 'factual_knowledge')
print(f"  Content preserved: {'OK' if 'Alexander' in result else 'FAIL'}")
print(f"  Result: {result[:200]}")
