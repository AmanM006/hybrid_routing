"""
Resilience tests — partial failure, empty input, edge prompts.
Run: python test_resilience.py
"""
import asyncio
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("FIREWORKS_API_KEY", "fake_key")
os.environ.setdefault("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
os.environ.setdefault("ALLOWED_MODELS", "minimax-m3,kimi-k2p7-code")

import main


class TestResilience(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.input_path = os.path.join(self.tmp, "tasks.json")
        self.output_path = os.path.join(self.tmp, "results.json")

    def _write_tasks(self, tasks):
        with open(self.input_path, "w", encoding="utf-8") as f:
            json.dump(tasks, f)

    def _read_results(self):
        with open(self.output_path, encoding="utf-8") as f:
            return json.load(f)

    async def _run_batch(self, tasks, fail_task_id=None):
        self._write_tasks(tasks)
        roles = {
            "code": "kimi-k2p7-code",
            "reasoning": "minimax-m3",
            "cheap": "minimax-m3",
            "mid": "minimax-m3",
        }
        mock_client = AsyncMock()
        mock_client.call_api.return_value = "Paris"

        results_map = {
            t["task_id"]: {"task_id": t["task_id"], "answer": "System interrupted"}
            for t in tasks
        }
        local_sem = asyncio.Semaphore(3)
        remote_sem = asyncio.Semaphore(4)

        original = main.execute_task_pipeline

        async def pipeline_with_inject(task_id, prompt, *args, **kwargs):
            if fail_task_id and task_id == fail_task_id:
                raise RuntimeError("Injected test failure")
            return await original(task_id, prompt, *args, **kwargs)

        with patch("main.local_disabled", True), patch(
            "main.execute_task_pipeline", side_effect=pipeline_with_inject
        ):
            for task in tasks:
                await main.process_single_task(
                    task, roles, mock_client, results_map, self.output_path, local_sem, remote_sem
                )
        return results_map

    async def test_partial_failure_all_tasks_complete(self):
        tasks = [{"task_id": f"t{i}", "prompt": f"What is {i}+{i}?"} for i in range(1, 20)]
        results = await self._run_batch(tasks, fail_task_id="t10")
        self.assertEqual(len(results), 19)
        outfile = self._read_results()
        self.assertEqual(len(outfile), 19)
        ids = {r["task_id"] for r in outfile}
        self.assertEqual(ids, {t["task_id"] for t in tasks})

    async def test_failing_task_gets_fallback_answer(self):
        tasks = [{"task_id": f"t{i}", "prompt": "Capital of France?"} for i in range(1, 6)]
        results = await self._run_batch(tasks, fail_task_id="t3")
        self.assertIn("t3", results)
        self.assertTrue(results["t3"]["answer"])
        self.assertNotEqual(results["t3"]["answer"], "System interrupted")

    async def test_empty_prompt_handled(self):
        tasks = [{"task_id": "empty1", "prompt": "   "}]
        results = await self._run_batch(tasks)
        self.assertEqual(results["empty1"]["answer"], "No content provided.")

    async def test_very_long_prompt_handled(self):
        long_prompt = "Summarize: " + ("word " * 8000)
        tasks = [{"task_id": "long1", "prompt": long_prompt}]
        results = await self._run_batch(tasks)
        self.assertTrue(results["long1"]["answer"])

    async def test_empty_tasks_json(self):
        self._write_tasks([])
        results_map = {}
        main.write_output_results(results_map, self.output_path)
        data = self._read_results()
        self.assertEqual(data, [])


class TestDockerCopyAudit(unittest.TestCase):
    def test_docker_copy_check_passes(self):
        import check_docker_copy
        self.assertEqual(check_docker_copy.main(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
