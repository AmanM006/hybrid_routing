import asyncio
import json
import os
import sys

sys.path.insert(0, ".")
from client import FireworksClient
from validators import validate_ner, validate_category_output
from main import classify_model_roles

PROMPT = json.load(open("input/adversarial_tasks.json", encoding="utf-8"))[4]["prompt"]


async def main():
    print("prompt:", repr(PROMPT))
    roles = classify_model_roles(
        ["minimax-m3", "kimi-k2p7-code", "gemma-4-31b-it", "gemma-4-26b-a4b-it", "gemma-4-31b-it-nvfp4"]
    )
    print("roles:", roles)
    c = FireworksClient(os.environ["FIREWORKS_API_KEY"], os.environ["FIREWORKS_BASE_URL"])
    for role in ["reasoning", "mid", "cheap", "code"]:
        model = roles.get(role)
        if not model:
            continue
        try:
            ans = await c.call_api(
                model=model, category="named_entity_recognition", prompt=PROMPT, timeout=14.0
            )
            scrubbed = c._scrub_cot(ans, "named_entity_recognition")
            print(f"\n--- {role} ({model}) ---")
            print("raw len:", len(ans), "valid:", validate_ner(ans))
            print("scrub len:", len(scrubbed), "valid:", validate_ner(scrubbed))
            print("raw:", repr(ans[:400]))
            print("scrub:", repr(scrubbed[:400]))
        except Exception as e:
            print(f"\n--- {role} ({model}) ERROR ---", e)


if __name__ == "__main__":
    asyncio.run(main())
