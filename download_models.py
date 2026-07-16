from transformers import pipeline
from sentence_transformers import SentenceTransformer

MODELS = {
    "prompt_injection": "protectai/deberta-v3-small-prompt-injection-v2",
    "moderation": "oxyapi/albert-moderation-001",
    "sentence_transformer": "sentence-transformers/all-MiniLM-L6-v2",
}

print("Downloading prompt-injection model...")
pipeline("text-classification", model=MODELS["prompt_injection"], tokenizer=MODELS["prompt_injection"])
print("Downloading moderation model...")
pipeline("text-classification", model=MODELS["moderation"], tokenizer=MODELS["moderation"], top_k=None)
print("Downloading SentenceTransformer model...")
SentenceTransformer(MODELS["sentence_transformer"])
print("Done.")
