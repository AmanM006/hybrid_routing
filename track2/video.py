"""Video download and ffmpeg preprocessing helpers."""

import asyncio
import base64
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)

MAX_VIDEO_SECONDS = 120
KEYFRAME_COUNT = 12


def get_video_duration(source_path: str, max_seconds: int = MAX_VIDEO_SECONDS) -> float:
    """Return clip duration capped at max_seconds."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            source_path,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return float(max_seconds)
    try:
        return min(float(result.stdout.strip()), float(max_seconds))
    except ValueError:
        return float(max_seconds)


async def download_video(url: str, dest_path: str, timeout: float = 120.0) -> None:
    """Download a video clip to a local file."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(1024 * 256):
                    f.write(chunk)


def _run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"ffmpeg failed: {stderr[:500]}")


def preprocess_video_base64(source_path: str, max_seconds: int = MAX_VIDEO_SECONDS) -> str:
    """Downscale to 360p, 1 FPS, strip audio, return base64 mp4."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        out_path = tmp.name
    try:
        _run_ffmpeg([
            "ffmpeg", "-y", "-i", source_path,
            "-t", str(max_seconds),
            "-vf", "fps=1,scale=-1:360",
            "-c:v", "libx264", "-preset", "fast",
            "-an",
            out_path,
        ])
        with open(out_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


def extract_keyframes(source_path: str, count: int = KEYFRAME_COUNT, max_seconds: int = MAX_VIDEO_SECONDS) -> list[str]:
    """Extract evenly spaced JPEG frames as base64 data URLs."""
    duration = max(get_video_duration(source_path, max_seconds), 1.0)
    fps_val = count / duration
    with tempfile.TemporaryDirectory() as tmpdir:
        pattern = str(Path(tmpdir) / "frame_%03d.jpg")
        _run_ffmpeg([
            "ffmpeg", "-y", "-i", source_path,
            "-t", str(max_seconds),
            "-vf", f"fps={fps_val:.6f},scale=-1:480",
            "-frames:v", str(count),
            pattern,
        ])
        frames = sorted(Path(tmpdir).glob("frame_*.jpg"))
        data_urls = []
        for frame_path in frames:
            with open(frame_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            data_urls.append(f"data:image/jpeg;base64,{b64}")
        return data_urls


async def ensure_local_video(video_url: str) -> str:
    """Download video to a temp file and return its path."""
    suffix = ".mp4"
    if "." in video_url.rsplit("/", 1)[-1]:
        ext = video_url.rsplit(".", 1)[-1].split("?")[0].lower()
        if ext in ("mp4", "mov", "webm", "mkv"):
            suffix = f".{ext}"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    await download_video(video_url, tmp.name)
    return tmp.name
