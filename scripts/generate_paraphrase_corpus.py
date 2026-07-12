#!/usr/bin/env python3
"""Generate paraphrase training corpus for v51 LoRA fine-tune.

Usage:
  python scripts/generate_paraphrase_corpus.py --out training/paraphrase_corpus.jsonl
  FIREWORKS_API_KEY=... python scripts/generate_paraphrase_corpus.py --remote-gold

Seeds from official-10 + diagnostic-28 templates; emits JSONL rows:
  {"category": "...", "prompt": "...", "answer": "...", "source": "..."}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from classifier import classify_prompt

# Category templates — paraphrase variants for fine-tune (no retired T01 IDs in prompts)
SEEDS: list[dict] = [
    {
        "category": "summarization",
        "variants": [
            "Summarize the following in exactly two sentences, covering both benefits and challenges:\n\n{text}",
            "Write a two-sentence summary that mentions both advantages and difficulties:\n\n{text}",
            "Condense into 2 sentences including pros and cons:\n\n{text}",
        ],
        "texts": [
            "Remote work saves commute time and offers flexibility, but many employees report isolation and blurred work-life boundaries. Companies benefit from lower office costs yet struggle to maintain team culture.",
            "Solar panels reduce electricity bills and carbon emissions, though upfront installation costs remain high. Maintenance is low but output varies with weather and season.",
        ],
    },
    {
        "category": "sentiment_classification",
        "variants": [
            "Classify sentiment and explain briefly: {text}",
            "What is the sentiment of this review? Give label and reason: {text}",
        ],
        "texts": [
            "The hotel room was spotless but the front desk kept us waiting forty minutes at check-in.",
            "Battery life is amazing although the screen feels cheap and flexes under pressure.",
        ],
        "gold": [
            "Mixed because the room was spotless but check-in took forty minutes.",
            "Mixed because battery life is amazing but the screen feels cheap.",
        ],
    },
    {
        "category": "factual_knowledge",
        "variants": [
            "{q}",
            "In plain prose, answer: {q}",
            "Briefly explain: {q}",
        ],
        "questions": [
            "Name the three primary additive colors used in digital displays and why screens use them instead of paint primaries.",
            "How does machine learning differ from deep learning in terms of feature engineering?",
            "Compare RAM and ROM: purpose and typical use in a computer.",
        ],
    },
    {
        "category": "math_reasoning",
        "variants": ["{q}"],
        "questions": [
            "A warehouse receives 120 boxes on Monday, 85 on Tuesday, and 140 on Wednesday. If 73 boxes ship out Thursday, how many remain?",
            "A recipe needs 1.25 cups sugar for 20 cookies. How many cups for 30 cookies if sugar costs $2.40 per cup?",
        ],
        "gold": ["1672", "You need 1.875 cups of sugar for 30 cookies. At $2.40 per cup, the total cost is $4.50."],
    },
    {
        "category": "named_entity_recognition",
        "variants": [
            "Extract named entities as JSON with PERSON, ORGANIZATION, LOCATION, DATE types:\n{text}",
        ],
        "texts": [
            "On March 15 2023, Sundar Pichai announced Google would open an AI lab in Zurich with ETH Zurich on LLM safety.",
        ],
    },
]


def _expand_seeds(rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    for seed in SEEDS:
        cat = seed["category"]
        if cat == "summarization":
            for text in seed["texts"]:
                for tmpl in seed["variants"]:
                    rows.append({
                        "category": cat,
                        "prompt": tmpl.format(text=text),
                        "answer": "",
                        "source": "template",
                        "needs_gold": True,
                    })
        elif cat == "sentiment_classification":
            for text, gold in zip(seed["texts"], seed["gold"]):
                for tmpl in seed["variants"]:
                    rows.append({
                        "category": cat,
                        "prompt": tmpl.format(text=text),
                        "answer": gold,
                        "source": "template",
                        "needs_gold": False,
                    })
        elif cat == "math_reasoning":
            for q, gold in zip(seed["questions"], seed["gold"]):
                for tmpl in seed["variants"]:
                    rows.append({
                        "category": cat,
                        "prompt": tmpl.format(q=q),
                        "answer": gold,
                        "source": "template",
                        "needs_gold": False,
                    })
        elif cat == "factual_knowledge":
            for q in seed["questions"]:
                for tmpl in seed["variants"]:
                    rows.append({
                        "category": cat,
                        "prompt": tmpl.format(q=q),
                        "answer": "",
                        "source": "template",
                        "needs_gold": True,
                    })
        elif cat == "named_entity_recognition":
            for text in seed["texts"]:
                for tmpl in seed["variants"]:
                    rows.append({
                        "category": cat,
                        "prompt": tmpl.format(text=text),
                        "answer": "",
                        "source": "template",
                        "needs_gold": True,
                    })
    rng.shuffle(rows)
    return rows


async def _fill_gold_remote(rows: list[dict]) -> None:
    from client import FireworksClient

    api_key = os.environ.get("FIREWORKS_API_KEY")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    model = os.environ.get("GOLD_MODEL", "accounts/fireworks/models/minimax-m3")
    if not api_key:
        raise SystemExit("FIREWORKS_API_KEY required for --remote-gold")

    client = FireworksClient(api_key=api_key, base_url=base_url, max_remote_calls=None)
    for row in rows:
        if not row.get("needs_gold") or row.get("answer"):
            continue
        cat = row["category"]
        row["answer"] = await client.call_api(
            model=model,
            category=cat,
            prompt=row["prompt"],
            timeout=20.0,
            count_toward_budget=False,
        )
        row.pop("needs_gold", None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="training/paraphrase_corpus.jsonl")
    parser.add_argument("--remote-gold", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = _expand_seeds(rng)

    # Load official + diagnostic seeds
    for path in [
        ROOT / "official_validation/input/tasks.json",
        ROOT / "input/final_diagnostic_batch.json",
    ]:
        if path.exists():
            tasks = json.loads(path.read_text(encoding="utf-8"))
            for t in tasks:
                cat = asyncio.run(classify_prompt(t["prompt"]))
                rows.append({
                    "category": cat,
                    "prompt": t["prompt"],
                    "answer": "",
                    "source": path.name,
                    "needs_gold": True,
                })

    if args.remote_gold:
        asyncio.run(_fill_gold_remote(rows))

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({
                "category": row["category"],
                "prompt": row["prompt"],
                "answer": row.get("answer", ""),
                "source": row.get("source", ""),
            }, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
