# Track 2 — Video Captioning Agent

An AI agent that watches video clips and generates captions in four styles: `formal`, `sarcastic`, `humorous_tech`, and `humorous_non_tech`.

## Architecture

Two-stage hybrid pipeline:

1. **Vision (MiniMax M3)**: Understand the clip via `video_url`, with ffmpeg/base64/keyframe fallbacks
2. **Style (Gemma 4 31B)**: Rewrite the scene brief into each requested tone

## Environment Variables

| Variable | Required | Default |
| :--- | :--- | :--- |
| `FIREWORKS_API_KEY` | Yes | — |
| `FIREWORKS_BASE_URL` | No | `https://api.fireworks.ai/inference/v1` |
| `CAPTION_VISION_MODEL` | No | `minimax-m3` |
| `CAPTION_STYLE_MODEL` | No | `minimax-m3` |
| `CAPTION_STYLE_FALLBACK_MODEL` | No | `minimax-m3` |
| `CAPTION_STYLE_FALLBACK_MODEL` | No | `minimax-m3` |
| `MAX_CONCURRENCY` | No | `4` |

## Local Run

```bash
cd track2
pip install -r requirements.txt
export FIREWORKS_API_KEY=your_key
export FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
python main.py
```

## Docker

```bash
cd track2
docker build --platform linux/amd64 -t video-caption-agent .
docker run \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  -e FIREWORKS_API_KEY=$FIREWORKS_API_KEY \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  video-caption-agent
```

**Submission image:**
```
ghcr.io/amanm006/video_captioning-v1:v1
```

Publish tag: `git tag track2-v1 && git push origin track2-v1`

Package must be **Public** on GitHub Packages.

## Tests

```bash
cd track2
python check_docker_copy.py
python test_caption_agent.py
```
