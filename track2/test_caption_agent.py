#!/usr/bin/env python3
"""Unit tests for Track 2 video captioning agent."""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from client import CaptionClient, qualify_model, scrub_caption
from main import fallback_caption, write_output_results
from prompts import (
    GROUNDING_INSTRUCTION,
    STYLES,
    STYLE_SYSTEM_PROMPTS,
    formal_polish_prompt,
    style_user_prompt,
)


class TestPrompts(unittest.TestCase):
    def test_all_styles_have_system_prompts(self):
        for style in STYLES:
            self.assertIn(style, STYLE_SYSTEM_PROMPTS)
            self.assertTrue(len(STYLE_SYSTEM_PROMPTS[style]) > 20)

    def test_style_user_prompt_includes_grounding(self):
        prompt = style_user_prompt("A cat sits in a garden.", "sarcastic")
        self.assertIn(GROUNDING_INSTRUCTION, prompt)
        self.assertIn("A cat sits in a garden.", prompt)

    def test_formal_polish_prompt(self):
        prompt = formal_polish_prompt("Traffic moves along a boulevard.")
        self.assertIn("Traffic moves along a boulevard.", prompt)


class TestClientHelpers(unittest.TestCase):
    def test_qualify_model(self):
        self.assertEqual(
            qualify_model("minimax-m3"),
            "accounts/fireworks/models/minimax-m3",
        )
        self.assertEqual(
            qualify_model("accounts/fireworks/models/gemma-4-31b-it"),
            "accounts/fireworks/models/gemma-4-31b-it",
        )

    def test_scrub_caption(self):
        self.assertEqual(scrub_caption("Caption: A dog runs."), "A dog runs.")
        self.assertEqual(scrub_caption('"Hello world."'), "Hello world.")


class TestOutputWrite(unittest.TestCase):
    def test_atomic_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "results.json")
            results = {
                "v1": {
                    "task_id": "v1",
                    "captions": {s: f"caption_{s}" for s in STYLES},
                }
            }
            write_output_results(results, out_path)
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(len(data), 1)
            self.assertEqual(set(data[0]["captions"].keys()), set(STYLES))


class TestFallbackCaptions(unittest.TestCase):
    def test_fallback_never_empty(self):
        for style in STYLES:
            caption = fallback_caption("A kitten plays in foliage.", style)
            self.assertTrue(len(caption) > 5)


class TestMultimodalClient(unittest.IsolatedAsyncioTestCase):
    async def test_understand_video_url_message_shape(self):
        client = CaptionClient(api_key="test", base_url="https://example.com/v1")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Scene brief.", reasoning_content=None, model_extra=None))]

        with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            result = await client.understand_video_url(
                "minimax-m3",
                "https://example.com/clip.mp4",
                "system",
                "describe",
            )

        self.assertEqual(result, "Scene brief.")
        kwargs = mock_create.call_args.kwargs
        messages = kwargs["messages"]
        user_content = messages[1]["content"]
        self.assertEqual(user_content[0]["type"], "video_url")
        self.assertEqual(user_content[0]["video_url"]["url"], "https://example.com/clip.mp4")
        self.assertEqual(user_content[0]["video_url"]["fps"], 1.0)

    async def test_understand_keyframes_message_shape(self):
        client = CaptionClient(api_key="test", base_url="https://example.com/v1")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Keyframe brief.", reasoning_content=None, model_extra=None))]

        frames = ["data:image/jpeg;base64,abc", "data:image/jpeg;base64,def"]
        with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            result = await client.understand_keyframes("minimax-m3", frames, "system", "describe")

        self.assertEqual(result, "Keyframe brief.")
        user_content = mock_create.call_args.kwargs["messages"][1]["content"]
        self.assertEqual(len(user_content), 3)
        self.assertEqual(user_content[0]["type"], "image_url")
        self.assertEqual(user_content[-1]["type"], "text")


class TestInputSchema(unittest.TestCase):
    def test_sample_tasks_json(self):
        tasks_path = ROOT / "input" / "tasks.json"
        with open(tasks_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        self.assertIsInstance(tasks, list)
        for task in tasks:
            self.assertIn("task_id", task)
            self.assertIn("video_url", task)
            self.assertIn("styles", task)
            self.assertTrue(task["video_url"].startswith("https://"))


if __name__ == "__main__":
    unittest.main()
