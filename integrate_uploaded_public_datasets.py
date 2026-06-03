"""
integrate_uploaded_public_datasets.py
=====================================

Integrates uploaded public/Kaggle datasets into one classifier training CSV.

Expected local raw files:
- data/public_raw/synthetic_dataset.csv
- data/public_raw/Prompt_INJECTION_And_Benign_DATASET.jsonl

CEAS_08.csv is already integrated in the packaged merged CSV. If you want to rebuild
from raw CEAS, put CEAS_08.csv inside data/public_raw/ and run this script.

Outputs:
- data/guardrail_training_uploaded_merged.csv
- data/guardrail_training_template.csv
- data/uploaded_dataset_label_report.csv
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import re

RAW = Path("data/public_raw")
OUT = Path("data")


def clean_text(x):
    return re.sub(r"\s+", " ", "" if x is None else str(x)).strip()


def map_synthetic_category(category, label):
    category = str(category).strip().lower()
    label = str(label).strip().lower()
    if label == "safe" or category == "safe": return "safe"
    return {
        "cybercrime": "cyber_abuse",
        "disinformation": "misinformation",
        "harassment": "hate_abuse",
        "hate": "hate_abuse",
        "sexual": "sexual_content",
        "drugs": "dangerous_content",
        "fraud": "fraud_scam",
        "copyright": "copyright_violation",
    }.get(category, "dangerous_content" if label == "harmful" else "safe")


def load_synthetic():
    path = RAW / "synthetic_dataset.csv"
    if not path.exists(): return pd.DataFrame(columns=["text", "label", "source"])
    df = pd.read_csv(path)
    text_col = "prompt_clean" if "prompt_clean" in df.columns else "prompt"
    return pd.DataFrame({
        "text": df[text_col].map(clean_text),
        "label": [map_synthetic_category(c, l) for c, l in zip(df.get("category", ""), df.get("label", ""))],
        "source": "uploaded:synthetic_dataset.csv",
    })


def map_prompt_json(row):
    label = str(row.get("label", "")).lower()
    attack = str(row.get("attack_type", "")).lower()
    if label in {"benign", "safe"} or attack == "none": return "safe"
    if "jailbreak" in attack or "role" in attack: return "jailbreak"
    if "code" in attack: return "script_or_command_injection"
    if "obfuscation" in attack or "leak" in attack: return "prompt_injection"
    return "prompt_injection"


def load_prompt_jsonl():
    path = RAW / "Prompt_INJECTION_And_Benign_DATASET.jsonl"
    if not path.exists(): return pd.DataFrame(columns=["text", "label", "source"])
    df = pd.read_json(path, lines=True)
    return pd.DataFrame({
        "text": df["prompt"].map(clean_text),
        "label": df.apply(map_prompt_json, axis=1),
        "source": "uploaded:Prompt_INJECTION_And_Benign_DATASET.jsonl",
    })


def load_ceas_if_present():
    path = RAW / "CEAS_08.csv"
    if not path.exists(): return pd.DataFrame(columns=["text", "label", "source"])
    df = pd.read_csv(path, low_memory=False)
    text = df.get("subject", "").fillna("").astype(str) + " " + df.get("body", "").fillna("").astype(str)
    return pd.DataFrame({
        "text": text.map(clean_text),
        "label": df["label"].map(lambda x: "fraud_scam" if int(x) == 1 else "safe"),
        "source": "uploaded:CEAS_08.csv",
    })


def main():
    frames = []
    for loader in [load_synthetic, load_prompt_jsonl, load_ceas_if_present]:
        d = loader()
        if not d.empty:
            print(f"Loaded {loader.__name__}: {len(d)} rows")
            frames.append(d)
    if not frames:
        raise RuntimeError("No uploaded datasets found in data/public_raw")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].astype(str).str.len() >= 2]
    df = df.drop_duplicates(subset=["text", "label"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "guardrail_training_uploaded_merged.csv", index=False)
    df.to_csv(OUT / "guardrail_training_template.csv", index=False)
    report = df.groupby("label").size().reset_index(name="count").sort_values("label")
    report.to_csv(OUT / "uploaded_dataset_label_report.csv", index=False)
    print(f"Saved {len(df)} rows.")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
