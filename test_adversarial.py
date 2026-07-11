"""
Adversarial + cascade-path tests. Run: python test_adversarial.py
"""
import asyncio
import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, ".")
os.environ.setdefault("FIREWORKS_API_KEY", "fake")
os.environ.setdefault("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")

from classifier import classify_prompt
from deterministic_solvers import (
    solve_math_deterministically,
    solve_ner_deterministically,
    solve_logic_deterministically,
)
from validators import verify_math_self_consistency
from main import classify_model_roles, execute_task_pipeline
import main


ADV_PATH = os.path.join("input", "adversarial_tasks.json")


class TestAdversarial(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        with open(ADV_PATH, encoding="utf-8") as f:
            cls.tasks = json.load(f)

    async def test_classifications(self):
        for task in self.tasks:
            cat = await classify_prompt(task["prompt"])
            print(f"  {task['task_id']}: classified={cat} (hint={task['category_hint']})")
            self.assertIsNotNone(cat)

    def test_deterministic_catches(self):
        catches = []
        for task in self.tasks:
            p = task["prompt"]
            ans = (
                solve_math_deterministically(p)
                or solve_ner_deterministically(p)
                or solve_logic_deterministically(p)
            )
            if ans is not None:
                catches.append((task["task_id"], ans[:80]))
        print(f"\nDeterministic zero-token catches: {len(catches)}/{len(self.tasks)}")
        for tid, ans in catches:
            print(f"  {tid}: {ans!r}")
        # At least some adversarial math should be caught
        self.assertGreaterEqual(len(catches), 2)

    def test_math_self_consistency(self):
        self.assertTrue(verify_math_self_consistency("What is 2+2?", "4"))
        self.assertFalse(verify_math_self_consistency("What is 2+2?", "5"))
        self.assertTrue(verify_math_self_consistency("What is 15% of 200?", "30"))
        self.assertFalse(verify_math_self_consistency("What is 15% of 200?", "40"))


class TestGemmaSkipPropagation(unittest.IsolatedAsyncioTestCase):
    """When Gemma models are dead, easy categories must never call them."""

    async def test_no_gemma_after_prune(self):
        allowed = [
            "minimax-m3", "kimi-k2p7-code",
            "gemma-4-26b-a4b-it", "gemma-4-31b-it-nvfp4",
        ]
        live = ["minimax-m3", "kimi-k2p7-code"]
        roles = classify_model_roles(live)
        self.assertEqual(roles["cheap"], "minimax-m3")
        self.assertEqual(roles["mid"], "minimax-m3")

        called_models = []
        mock_client = AsyncMock()
        mock_client.total_fireworks_tokens = MagicMock(return_value=0)

        async def track_call(model, category, prompt, timeout=12.0):
            called_models.append(model)
            if category == "named_entity_recognition":
                return '{"entities": [{"text": "Test", "type": "ORG"}, {"text": "Place", "type": "LOCATION"}]}'
            if category == "sentiment_classification":
                return "positive. The tone is upbeat."
            if category == "summarization":
                return "Brief summary of the topic."
            return "Paris"

        mock_client.call_api.side_effect = track_call

        prompts = [
            ("f1", "What is the capital of Japan?", "factual_knowledge"),
            ("s1", "Classify sentiment: I love this!", "sentiment_classification"),
            ("sum1", "Summarize: AI is transforming industries.", "summarization"),
            ("n1", "Extract entities: John works at Acme in Boston.", "named_entity_recognition"),
        ]

        local_sem = asyncio.Semaphore(1)
        remote_sem = asyncio.Semaphore(1)

        with patch("main.local_disabled", True), patch(
            "main.classify_prompt",
            side_effect=lambda p, **kw: next(c for _, pr, c in prompts if pr == p),
        ):
            for tid, prompt, _ in prompts:
                await execute_task_pipeline(
                    tid, prompt, roles, mock_client, local_sem, remote_sem
                )

        gemma_calls = [m for m in called_models if "gemma" in m.lower()]
        print(f"\nModels called for easy categories: {called_models}")
        print(f"Gemma calls: {gemma_calls}")
        self.assertEqual(gemma_calls, [], "Dead Gemma models must not be called")


if __name__ == "__main__":
    unittest.main(verbosity=2)
