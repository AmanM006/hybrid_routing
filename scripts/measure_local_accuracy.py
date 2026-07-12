#!/usr/bin/env python3
"""Measure per-category local vs deterministic accuracy on paraphrase holdout.

Usage:
  python scripts/measure_local_accuracy.py training/paraphrase_corpus.jsonl
  # Prints categories >=95% safe for LOCAL_ALLOWED_CATEGORIES
"""
from __future__ import annotations

import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from official_local_bench import grade_official
from deterministic_solvers import (
    solve_math_deterministically,
    solve_sentiment_deterministically,
    solve_ner_deterministically,
)


def _deterministic_answer(category: str, prompt: str) -> str | None:
    if category == "math_reasoning":
        return solve_math_deterministically(prompt)
    if category == "sentiment_classification":
        return solve_sentiment_deterministically(prompt)
    if category == "named_entity_recognition":
        return solve_ner_deterministically(prompt)
    return None


def _grade_row(category: str, prompt: str, answer: str, task_id: str) -> bool:
    if not answer or not answer.strip():
        return False
    if task_id.startswith("T0"):
        ok, _ = grade_official(task_id, prompt, answer)
        return ok
    # Generic: non-empty + basic structural checks
    if category == "math_reasoning":
        return any(c.isdigit() for c in answer)
    if category == "sentiment_classification":
        low = answer.lower()
        return any(l in low for l in ("positive", "negative", "neutral", "mixed")) and "because" in low
    if category == "summarization":
        return len(answer.split()) >= 10
    if category == "factual_knowledge":
        return len(answer.split()) >= 20
    if category == "named_entity_recognition":
        return "entities" in answer
    return len(answer.strip()) > 5


async def _local_answer(category: str, prompt: str) -> str:
    import aiohttp
    from client import SYSTEM_PROMPTS, get_max_tokens, get_user_prompt_suffix

    system = SYSTEM_PROMPTS.get(category, "Answer the query.")
    user = prompt + get_user_prompt_suffix(category, prompt)
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": get_max_tokens(category, prompt),
    }
    url = "http://127.0.0.1:8085/v1/chat/completions"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=45.0) as resp:
            data = await resp.json()
            return data["choices"][0]["message"]["content"].strip()


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "training/paraphrase_corpus.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # Hold out last 20% per category
    by_cat: dict[str, list] = defaultdict(list)
    for i, row in enumerate(rows):
        by_cat[row["category"]].append((i, row))

    det_stats: dict[str, list[bool]] = defaultdict(list)
    local_stats: dict[str, list[bool]] = defaultdict(list)

    for cat, items in by_cat.items():
        split = max(1, len(items) // 5)
        holdout = items[-split:]
        for idx, row in holdout:
            prompt = row["prompt"]
            gold = row.get("answer", "")
            tid = f"T{idx:02d}" if "T0" not in row.get("source", "") else row.get("task_id", f"T{idx:02d}")

            det = _deterministic_answer(cat, prompt)
            if det is not None:
                ok = _grade_row(cat, prompt, det, tid) if not gold else (gold.strip() in det or det.strip() in gold)
                det_stats[cat].append(ok)

            try:
                loc = asyncio.run(_local_answer(cat, prompt))
                if gold:
                    ok = gold.strip()[:40] in loc or loc.strip()[:40] in gold
                else:
                    ok = _grade_row(cat, prompt, loc, tid)
                local_stats[cat].append(ok)
            except Exception:
                local_stats[cat].append(False)

    print("\n=== Per-category accuracy (holdout) ===")
    recommended: list[str] = []
    for cat in sorted(set(list(det_stats) + list(local_stats))):
        d = det_stats.get(cat, [])
        l = local_stats.get(cat, [])
        d_acc = sum(d) / len(d) if d else 0.0
        l_acc = sum(l) / len(l) if l else 0.0
        print(f"{cat:30s}  det={d_acc:.0%} ({len(d)})  local={l_acc:.0%} ({len(l)})")
        if l_acc >= 0.95 and len(l) >= 3:
            recommended.append(cat)

    print("\nLOCAL_ALLOWED_CATEGORIES (local >=95%):")
    print(",".join(recommended) if recommended else "summarization  # default until fine-tune")


if __name__ == "__main__":
    main()
