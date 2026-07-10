"""Fireworks API client for multimodal video and text caption generation."""

import asyncio
import logging
import random
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


def qualify_model(model: str) -> str:
    if "/" not in model:
        return f"accounts/fireworks/models/{model}"
    return model


def scrub_caption(text: str) -> str:
    """Strip preamble and return the caption text."""
    if not text:
        return text

    cleaned = text.strip()
    skip_prefixes = (
        "here is", "here's", "the caption", "sure,", "certainly,",
        "of course,", "let me", "i'll", "i will",
    )
    lines = []
    for line in cleaned.splitlines():
        stripped_line = line.strip()
        lower = stripped_line.lower()
        if lower.startswith("caption:"):
            stripped_line = stripped_line.split(":", 1)[1].strip()
            lower = stripped_line.lower()
        if lower.startswith(skip_prefixes):
            continue
        lines.append(stripped_line)
    result = " ".join(line.strip() for line in lines if line.strip()).strip()

    # Drop surrounding quotes
    if len(result) >= 2 and result[0] == result[-1] and result[0] in "\"'":
        result = result[1:-1].strip()

    return result if result else text.strip()


class CaptionClient:
    def __init__(self, api_key: str, base_url: str):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 200,
        temperature: float = 0.7,
        timeout: float = 120.0,
    ) -> str:
        model = qualify_model(model)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        if "minimax-m3" in model:
            kwargs["extra_body"] = {"reasoning_effort": "none"}

        attempts = 5
        for attempt in range(attempts):
            try:
                logger.info("API call to %s, attempt %d", model, attempt + 1)
                try:
                    response = await self.client.chat.completions.create(**kwargs)
                except Exception as exc:
                    if "extra_body" in kwargs and any(
                        err in str(exc).lower()
                        for err in ("reasoning_effort", "invalid", "unexpected", "400")
                    ):
                        del kwargs["extra_body"]
                        response = await self.client.chat.completions.create(**kwargs)
                    else:
                        raise

                msg = response.choices[0].message
                content = (msg.content or "").strip()
                reasoning = getattr(msg, "reasoning_content", None)
                if not reasoning and hasattr(msg, "model_extra") and msg.model_extra:
                    reasoning = msg.model_extra.get("reasoning_content")

                final_text = content or (reasoning or "").strip()
                if final_text:
                    return scrub_caption(final_text)
                raise ValueError("Empty response from model")
            except Exception as exc:
                logger.warning("API attempt %d failed: %s", attempt + 1, exc)
                err_str = str(exc)
                if "404" in err_str or "NOT_FOUND" in err_str or "not found" in err_str.lower():
                    raise
                if attempt == attempts - 1:
                    raise
                is_rate_limit = "429" in err_str or "RATE_LIMIT" in err_str
                wait = (3.0 * (attempt + 1) + random.uniform(0.5, 1.5)) if is_rate_limit else (0.5 * (attempt + 1))
                await asyncio.sleep(wait)

        raise RuntimeError("API call failed after retries")

    async def understand_video_url(
        self,
        model: str,
        video_url: str,
        system_prompt: str,
        user_prompt: str,
        *,
        fps: float = 1.0,
        timeout: float = 180.0,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": video_url,
                            "fps": fps,
                            "detail": "default",
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        return await self.chat_completion(
            model,
            messages,
            max_tokens=400,
            temperature=0.2,
            timeout=timeout,
        )

    async def understand_video_base64(
        self,
        model: str,
        video_b64: str,
        system_prompt: str,
        user_prompt: str,
        *,
        fps: float = 1.0,
        timeout: float = 180.0,
    ) -> str:
        data_url = f"data:video/mp4;base64,{video_b64}"
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": data_url,
                            "fps": fps,
                            "detail": "default",
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        return await self.chat_completion(
            model,
            messages,
            max_tokens=400,
            temperature=0.2,
            timeout=timeout,
        )

    async def understand_keyframes(
        self,
        model: str,
        frame_data_urls: list[str],
        system_prompt: str,
        user_prompt: str,
        *,
        timeout: float = 180.0,
    ) -> str:
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": url, "detail": "default"}}
            for url in frame_data_urls
        ]
        content.append({"type": "text", "text": user_prompt})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        return await self.chat_completion(
            model,
            messages,
            max_tokens=400,
            temperature=0.2,
            timeout=timeout,
        )

    async def generate_style_caption(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        *,
        timeout: float = 60.0,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        temperature = 0.85 if "humor" in system_prompt.lower() or "sarcastic" in system_prompt.lower() else 0.3
        return await self.chat_completion(
            model,
            messages,
            max_tokens=150,
            temperature=temperature,
            timeout=timeout,
        )
