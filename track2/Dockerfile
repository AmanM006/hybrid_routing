FROM mwader/static-ffmpeg:7.1.1 AS ffmpeg

FROM python:3.11-slim

WORKDIR /app

COPY --from=ffmpeg /ffmpeg /usr/local/bin/ffmpeg
COPY --from=ffmpeg /ffprobe /usr/local/bin/ffprobe

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py client.py video.py prompts.py ./

ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/bin:${PATH}"

ENTRYPOINT ["python", "main.py"]
