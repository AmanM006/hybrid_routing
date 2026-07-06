import os
import urllib.request

MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "qwen2.5-1.5b-instruct-q4_k_m.gguf")

def download():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}")
        return
        
    print(f"Downloading model from {MODEL_URL} to {MODEL_PATH}...")
    
    try:
        req = urllib.request.Request(
            MODEL_URL, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(MODEL_PATH, 'wb') as out_file:
            meta = response.info()
            file_size = int(meta.get("Content-Length", 0))
            print(f"Size: {file_size / (1024*1024):.2f} MB")
            
            bytes_downloaded = 0
            block_size = 8192 * 8  # 64KB chunks for faster transfer
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
    except Exception as e:
        print(f"Error downloading model: {e}")
        # Try to delete incomplete file
        if os.path.exists(MODEL_PATH):
            try:
                os.remove(MODEL_PATH)
            except:
                pass
        raise

if __name__ == "__main__":
    download()
