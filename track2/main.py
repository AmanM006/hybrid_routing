#!/usr/bin/env python3
"""Track 2: Video Captioning Agent — reads /input/tasks.json, writes /output/results.json."""

import asyncio
import json
import logging
import os
import sys
import time

from dotenv import load_dotenv

from client import CaptionClient
from prompts import (
    STYLES,
    STYLE_SYSTEM_PROMPTS,
    VISION_SYSTEM,
    VISION_USER,
    formal_polish_prompt,
    style_user_prompt,
)
from video import (
    ensure_local_video,
    extract_keyframes,
    preprocess_video_base64,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

DEFAULT_VISION_MODEL = "minimax-m3"
DEFAULT_STYLE_MODEL = "minimax-m3"
DEFAULT_FALLBACK_STYLE_MODEL = "minimax-m3"


def write_output_results(results_map: dict, output_path: str) -> None:
    """Write results atomically as a JSON array."""
    try:
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        temp_path = output_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(list(results_map.values()), f, indent=2)
        os.replace(temp_path, output_path)
    except Exception as exc:
        logger.error("Failed to write results: %s", exc)


def fallback_caption(scene_brief: str, style: str) -> str:
    """Best-effort caption when API calls fail."""
    brief = scene_brief.strip() or "A scene unfolds in the video clip."
    if style == "formal":
        return brief
    if style == "sarcastic":
        return f"Apparently, {brief[0].lower()}{brief[1:]}" if len(brief) > 1 else brief
    if style == "humorous_tech":
        return f"{brief} (Still loading… please do not refresh the browser.)"
    return f"{brief} — classic human behavior, honestly."


async def understand_video(
    client: CaptionClient,
    vision_model: str,
    video_url: str,
) -> str:
    """Hybrid vision pipeline: URL → base64 video → keyframe images."""
    errors: list[str] = []

    try:
        logger.info("Vision pass 1: video_url for %s", video_url[:80])
        return await client.understand_video_url(
            vision_model, video_url, VISION_SYSTEM, VISION_USER
        )
    except Exception as exc:
        errors.append(f"video_url: {exc}")
        logger.warning("Video URL pass failed: %s", exc)

    local_path = None
    try:
        local_path = await ensure_local_video(video_url)
        try:
            logger.info("Vision pass 2: preprocessed base64 video")
            video_b64 = preprocess_video_base64(local_path)
            return await client.understand_video_base64(
                vision_model, video_b64, VISION_SYSTEM, VISION_USER
            )
        except Exception as exc:
            errors.append(f"base64_video: {exc}")
            logger.warning("Base64 video pass failed: %s", exc)

        try:
            logger.info("Vision pass 3: keyframe images")
            frames = extract_keyframes(local_path)
            if not frames:
                raise RuntimeError("No keyframes extracted")
            return await client.understand_keyframes(
                vision_model, frames, VISION_SYSTEM, VISION_USER
            )
        except Exception as exc:
            errors.append(f"keyframes: {exc}")
            logger.warning("Keyframe pass failed: %s", exc)
    finally:
        if local_path and os.path.exists(local_path):
            os.unlink(local_path)

    raise RuntimeError("; ".join(errors) or "All vision passes failed")


async def generate_formal_caption(
    client: CaptionClient,
    style_model: str,
    fallback_style_model: str,
    scene_brief: str,
) -> str:
    try:
        return await client.generate_style_caption(
            style_model,
            STYLE_SYSTEM_PROMPTS["formal"],
            formal_polish_prompt(scene_brief),
        )
    except Exception as exc:
        logger.warning("Formal polish with %s failed: %s", style_model, exc)
        try:
            return await client.generate_style_caption(
                fallback_style_model,
                STYLE_SYSTEM_PROMPTS["formal"],
                formal_polish_prompt(scene_brief),
            )
        except Exception:
            return fallback_caption(scene_brief, "formal")


async def generate_style_caption(
    client: CaptionClient,
    style_model: str,
    fallback_style_model: str,
    scene_brief: str,
    style: str,
) -> str:
    if style == "formal":
        return await generate_formal_caption(
            client, style_model, fallback_style_model, scene_brief
        )

    user_prompt = style_user_prompt(scene_brief, style)
    system_prompt = STYLE_SYSTEM_PROMPTS[style]
    try:
        return await client.generate_style_caption(
            style_model, system_prompt, user_prompt
        )
    except Exception as exc:
        logger.warning("Style %s with %s failed: %s", style, style_model, exc)
        try:
            return await client.generate_style_caption(
                fallback_style_model, system_prompt, user_prompt
            )
        except Exception:
            return fallback_caption(scene_brief, style)


async def process_single_task(
    task: dict,
    client: CaptionClient,
    vision_model: str,
    style_model: str,
    fallback_style_model: str,
    sem: asyncio.Semaphore,
    results_map: dict,
    output_path: str,
) -> None:
    task_id = task["task_id"]
    video_url = task["video_url"]
    requested_styles = task.get("styles", list(STYLES))

    async with sem:
        start = time.time()
        captions: dict[str, str] = {}
        scene_brief = ""

        try:
            scene_brief = await understand_video(client, vision_model, video_url)
            logger.info("Task %s scene brief (%d chars)", task_id, len(scene_brief))
        except Exception as exc:
            logger.error("Task %s vision failed: %s", task_id, exc)
            scene_brief = "Unable to fully analyze the video; describing visible activity in the clip."

        style_tasks = []
        for style in requested_styles:
            if style not in STYLES:
                logger.warning("Task %s: unknown style %s, skipping", task_id, style)
                continue
            style_tasks.append(
                generate_style_caption(
                    client, style_model, fallback_style_model, scene_brief, style
                )
            )

        style_results = await asyncio.gather(*style_tasks, return_exceptions=True)
        style_names = [s for s in requested_styles if s in STYLES]

        for style, result in zip(style_names, style_results):
            if isinstance(result, Exception):
                logger.error("Task %s style %s failed: %s", task_id, style, result)
                captions[style] = fallback_caption(scene_brief, style)
            else:
                captions[style] = result or fallback_caption(scene_brief, style)

        for style in STYLES:
            if style not in captions:
                captions[style] = fallback_caption(scene_brief, style)

        elapsed = time.time() - start
        logger.info("Task %s completed in %.1fs", task_id, elapsed)

        results_map[task_id] = {"task_id": task_id, "captions": captions}
        write_output_results(results_map, output_path)


async def async_main() -> None:
    api_key = os.environ.get("FIREWORKS_API_KEY")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    vision_model = os.environ.get("CAPTION_VISION_MODEL", DEFAULT_VISION_MODEL)
    style_model = os.environ.get("CAPTION_STYLE_MODEL", DEFAULT_STYLE_MODEL)
    fallback_style_model = os.environ.get(
        "CAPTION_STYLE_FALLBACK_MODEL", DEFAULT_FALLBACK_STYLE_MODEL
    )
    max_concurrency = int(os.environ.get("MAX_CONCURRENCY", "4"))

    if not api_key:
        logger.error("Missing required environment variable: FIREWORKS_API_KEY")
        sys.exit(1)

    print("=== TRACK 2 CONFIG ===")
    print(f"Vision model: {vision_model}")
    print(f"Style model: {style_model}")
    print(f"Style fallback: {fallback_style_model}")
    print(f"Max concurrency: {max_concurrency}")
    print("======================")

    input_path = "/input/tasks.json"
    output_path = "/output/results.json"
    if not os.path.exists("/input") or not os.path.exists("/output"):
        input_path = "./input/tasks.json"
        output_path = "./output/results.json"
        os.makedirs("./input", exist_ok=True)
        os.makedirs("./output", exist_ok=True)

    if not os.path.exists(input_path):
        logger.error("Input file not found at %s", input_path)
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    if not isinstance(tasks, list):
        logger.error("Input file must be a JSON array of tasks.")
        sys.exit(1)

    logger.info("Loaded %d tasks", len(tasks))

    client = CaptionClient(api_key=api_key, base_url=base_url)
    sem = asyncio.Semaphore(max_concurrency)

    results_map = {
        task["task_id"]: {
            "task_id": task["task_id"],
            "captions": {style: "Processing..." for style in STYLES},
        }
        for task in tasks
    }
    write_output_results(results_map, output_path)

    try:
        await asyncio.gather(
            *[
                process_single_task(
                    task,
                    client,
                    vision_model,
                    style_model,
                    fallback_style_model,
                    sem,
                    results_map,
                    output_path,
                )
                for task in tasks
            ]
        )
    finally:
        write_output_results(results_map, output_path)

    logger.info("All tasks complete. Results written to %s", output_path)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
