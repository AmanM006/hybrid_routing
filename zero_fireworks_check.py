#!/usr/bin/env python3
"""Verify ZERO_FIREWORKS mode never calls Fireworks API."""
import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("FIREWORKS_API_KEY", "fake")
os.environ.setdefault("FIREWORKS_BASE_URL", "https://example.invalid/v1")
os.environ.setdefault("ALLOWED_MODELS", "minimax-m3,gemma-test")
os.environ["ZERO_FIREWORKS"] = "true"
os.environ["SKIP_FW_HEALTHCHECK"] = "true"

import main
from main import execute_task_pipeline, classify_model_roles
from client import FireworksClient

FAILURES = []


def load_tasks(*paths):
    tasks = []
    for path in paths:
        p = ROOT / path
        if p.exists():
            tasks.extend(json.loads(p.read_text(encoding="utf-8")))
    return tasks


async def run_checks():
    tasks = load_tasks(
        "official_validation/input/tasks.json",
        "input/final_diagnostic_batch.json",
    )
    if not tasks:
        FAILURES.append("No task files found for zero-fireworks audit")
        return

    client = FireworksClient("fake", "https://example.invalid/v1")
    roles = classify_model_roles(["minimax-m3", "gemma-test"])
    local_sem = asyncio.Semaphore(1)
    remote_sem = asyncio.Semaphore(1)

    api_calls = {"count": 0}

    async def blocked_call_api(*args, **kwargs):
        api_calls["count"] += 1
        raise AssertionError("call_api invoked under ZERO_FIREWORKS")

    with patch.object(FireworksClient, "call_api", side_effect=blocked_call_api):
        with patch.object(main, "local_disabled", True):
            for task in tasks:
                tid = task.get("task_id", task.get("id", "?"))
                prompt = task["prompt"]
                try:
                    await execute_task_pipeline(tid, prompt, roles, client, local_sem, remote_sem)
                except AssertionError as e:
                    FAILURES.append(f"{tid}: {e}")

    if api_calls["count"] != 0:
        FAILURES.append(f"call_api count={api_calls['count']} (expected 0)")

    total_fw = client.total_fireworks_tokens()
    if total_fw != 0:
        FAILURES.append(f"client total_fireworks_tokens={total_fw} (expected 0)")


def main_fn():
    print("=" * 70)
    print("ZERO FIREWORKS CHECK")
    print("=" * 70)
    asyncio.run(run_checks())
    if FAILURES:
        print(f"FAILED ({len(FAILURES)} issues):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print(f"PASSED — {len(load_tasks('official_validation/input/tasks.json', 'input/final_diagnostic_batch.json'))} tasks, 0 call_api")
    sys.exit(0)


if __name__ == "__main__":
    main_fn()
