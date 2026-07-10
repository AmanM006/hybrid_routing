import os
import urllib.request

MODELS = {
    "1.5b": (
        "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    ),
    "3b": (
        "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        "qwen2.5-3b-instruct-q4_k_m.gguf",
    ),
}

DEFAULT_KEY = os.environ.get("LOCAL_MODEL_KEY", "3b")
MODEL_URL, MODEL_FILENAME = MODELS[DEFAULT_KEY]
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)


def download(url: str = MODEL_URL, path: str = MODEL_PATH):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    if os.path.exists(path):
        print(f"Model already exists at {path}")
        return path

    print(f"Downloading model from {url} to {path}...")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(path, "wb") as out_file:
        file_size = int(response.info().get("Content-Length", 0))
        print(f"Size: {file_size / (1024*1024):.2f} MB")
        bytes_downloaded = 0
        block_size = 8192 * 8
        last_reported = -5
        while True:
            buffer = response.read(block_size)
            if not buffer:
                break
            bytes_downloaded += len(buffer)
            out_file.write(buffer)
            percent = int(bytes_downloaded * 100 / file_size) if file_size else 0
            if percent - last_reported >= 5:
                print(f"Downloaded: {percent}% ({bytes_downloaded / (1024*1024):.2f} MB)")
                last_reported = percent
    print("Download complete!")
    return path


if __name__ == "__main__":
    import sys
    key = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_KEY
    if key not in MODELS:
        raise SystemExit(f"Unknown model key {key!r}; choose from {list(MODELS)}")
    url, filename = MODELS[key]
    download(url, os.path.join(MODEL_DIR, filename))
