# Stage 1: Build virtual env and download model
FROM --platform=linux/amd64 python:3.11-slim AS builder

WORKDIR /build

# Install download and extraction tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY requirements.txt .

# Install dependencies in virtual env
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download precompiled llama-server Ubuntu binary
RUN curl -L -o /tmp/llama.zip https://github.com/ggml-org/llama.cpp/releases/download/b3620/llama-b3620-bin-ubuntu-x64.zip && \
    unzip /tmp/llama.zip -d /tmp/llama_extracted && \
    cp /tmp/llama_extracted/build/bin/llama-server /build/llama-server && \
    chmod +x /build/llama-server

# Download local model file (Qwen2.5-1.5B q4 — proven on 4GB RAM grading env)
RUN mkdir -p models && \
    curl -L -o models/qwen2.5-1.5b-instruct-q4_k_m.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf

# Stage 2: Final runtime image
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# Copy the prepared virtual environment, model file, and server binary from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/models /app/models
COPY --from=builder /build/llama-server /usr/local/bin/llama-server

# Copy source python modules
COPY main.py classifier.py validators.py client.py deterministic_solvers.py ./

# Set paths and python environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Set the standard entrypoint
ENTRYPOINT ["python", "main.py"]
