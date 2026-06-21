"""
download_models.py
Downloads the 3 HuggingFace models used by the KB-Aware Input Guardrail
into a local ./models folder, so the pipeline can load them offline.

Run from your project root:
    python download_models.py
"""

import os
import time
from huggingface_hub import snapshot_download

# Folder where models will be stored: <project_root>/models
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODELS = {
    "protectai/deberta-v3-small-prompt-injection-v2": "protectai__deberta-v3-small-prompt-injection-v2",
    "oxyapi/albert-moderation-001": "oxyapi__albert-moderation-001",
    "cross-encoder/nli-MiniLM2-L6-H768": "cross-encoder__nli-MiniLM2-L6-H768",
}

# Only pull files we actually need - skips duplicate weight formats
# (.bin, .h5, .onnx, .msgpack) that bloat the download
ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.safetensors",
    "*.model",
    "tokenizer*",
    "vocab*",
    "merges.txt",
]

MAX_RETRIES = 3


def download_with_retry(repo_id, local_dir):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                allow_patterns=ALLOW_PATTERNS,
            )
            return True
        except Exception as e:
            print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                print("  Retrying in 5 seconds...")
                time.sleep(5)
    return False


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"Models will be saved under: {BASE_DIR}\n")

    for repo_id, folder_name in MODELS.items():
        local_dir = os.path.join(BASE_DIR, folder_name)
        print(f"Downloading {repo_id} -> {local_dir}")
        success = download_with_retry(repo_id, local_dir)
        if success:
            print(f"  Done: {repo_id}\n")
        else:
            print(f"  FAILED after {MAX_RETRIES} attempts: {repo_id}\n")

    print("All downloads attempted. Check above for any FAILED entries.")


if __name__ == "__main__":
    main()