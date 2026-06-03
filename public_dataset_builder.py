"""
public_dataset_builder.py
=========================

Builds a stronger classifier training dataset from:

1. Existing seed dataset:
   data/guardrail_training_template.csv

2. Public HuggingFace datasets listed in:
   dataset_sources_config.json

3. Your own downloaded Kaggle/CSV datasets placed in:
   data/public_raw/*.csv

Output:
   data/guardrail_training_merged_public.csv

Important:
- This script does NOT claim to include every public row in the world.
- It downloads/merges datasets only when internet access is available.
- Always check dataset licenses before final submission/production use.
- Public datasets have inconsistent schemas; mappings may need small edits.

Run:
   pip install -r requirements_public_datasets.txt
   python public_dataset_builder.py

Then train:
   python train_sklearn_classifier.py --data data/guardrail_training_merged_public.csv --output models/sklearn_safety_classifier.joblib
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


RANDOM_SEED = 42
random.seed(RANDOM_SEED)


LABEL_SET = {
    "safe",
    "prompt_injection",
    "jailbreak",
    "cyber_abuse",
    "self_harm",
    "violence_threat",
    "dangerous_content",
    "hate_abuse",
    "extremism",
    "child_safety",
    "privacy_pii",
    "format_issue",
    "script_or_command_injection",
    "out_of_scope",
}


def obfuscate_text(text: str, probability: float = 0.18) -> str:
    """Generate lightweight adversarial variants for classifier robustness."""
    maps = {
        "a": ["4", "@", "á"],
        "e": ["3", "ë"],
        "i": ["1", "!", "|"],
        "o": ["0", "о"],  # Cyrillic o included intentionally
        "s": ["5", "$"],
        "t": ["7"],
        "b": ["8"],
        "l": ["1", "|"],
    }
    out = []
    for ch in text:
        low = ch.lower()
        if low in maps and random.random() < probability:
            repl = random.choice(maps[low])
            out.append(repl if ch.islower() else repl.upper())
        else:
            out.append(ch)

    s = "".join(out)
    # occasionally separate risky keywords with spaces
    if random.random() < 0.12:
        s = re.sub(
            r"\b(make|bomb|hack|suicide|kill|password|explosive|attack|poison|jailbreak|ignore)\b",
            lambda m: " ".join(list(m.group(0))),
            s,
            flags=re.I,
        )
    return s


def clean_text(text) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(label: str, target_label: Optional[str] = None) -> Optional[str]:
    """
    Maps public dataset labels into this project's label taxonomy.
    This is intentionally conservative.
    """
    if label is None and target_label:
        return target_label

    lab = str(label).strip().lower()

    if lab in LABEL_SET:
        return lab

    safe_values = {"0", "false", "safe", "benign", "harmless", "normal", "non-toxic", "nontoxic", "clean"}
    unsafe_values = {"1", "true", "unsafe", "harmful", "toxic", "attack", "jailbreak", "injection", "malicious"}

    if lab in safe_values:
        return "safe"

    if "prompt" in lab and "inject" in lab:
        return "prompt_injection"
    if "jailbreak" in lab:
        return "jailbreak"
    if "toxic" in lab or "hate" in lab or "insult" in lab or "abuse" in lab:
        return "hate_abuse"
    if "cyber" in lab or "hack" in lab or "malware" in lab or "phish" in lab:
        return "cyber_abuse"
    if "self" in lab or "suicide" in lab:
        return "self_harm"
    if "violence" in lab or "threat" in lab:
        return "violence_threat"
    if "child" in lab or "minor" in lab or "csam" in lab:
        return "child_safety"
    if "terror" in lab or "extrem" in lab:
        return "extremism"
    if "pii" in lab or "privacy" in lab:
        return "privacy_pii"

    if lab in unsafe_values and target_label:
        return target_label

    if target_label:
        return target_label

    return None


def choose_first_existing(columns: Iterable[str], candidates: List[str]) -> Optional[str]:
    col_set = set(columns)
    for c in candidates:
        if c in col_set:
            return c
    return None


def load_seed(seed_path: Path) -> pd.DataFrame:
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed dataset not found: {seed_path}")
    df = pd.read_csv(seed_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Seed dataset must have text,label columns.")
    df = df[["text", "label"]].copy()
    df["source"] = "seed_synthetic_curated"
    return df


def load_local_csvs(folder: Path) -> pd.DataFrame:
    rows = []
    if not folder.exists():
        return pd.DataFrame(columns=["text", "label", "source"])

    for path in folder.glob("*.csv"):
        try:
            df = pd.read_csv(path)
            if not {"text", "label"}.issubset(df.columns):
                print(f"[SKIP] {path.name}: requires text,label columns.")
                continue
            tmp = df[["text", "label"]].copy()
            tmp["source"] = f"local_csv:{path.name}"
            rows.append(tmp)
            print(f"[OK] Loaded local CSV {path.name}: {len(tmp)} rows")
        except Exception as exc:
            print(f"[WARN] Could not load {path}: {exc}")

    if not rows:
        return pd.DataFrame(columns=["text", "label", "source"])
    return pd.concat(rows, ignore_index=True)


def load_hf_source(src: Dict) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except Exception as exc:
        print("[WARN] HuggingFace datasets library not installed. Run pip install -r requirements_public_datasets.txt")
        return pd.DataFrame(columns=["text", "label", "source"])

    name = src["name"]
    split = src.get("split", "train")
    max_rows = int(src.get("max_rows", 20000))
    target_label = src.get("target_label")
    text_columns = src.get("text_columns", ["text", "prompt", "input", "instruction"])
    label_columns = src.get("label_columns", ["label", "labels", "category"])

    try:
        ds = load_dataset(name, split=split)
    except Exception as exc:
        print(f"[WARN] Failed to load {name}: {exc}")
        return pd.DataFrame(columns=["text", "label", "source"])

    df = ds.to_pandas()
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=RANDOM_SEED)

    text_col = choose_first_existing(df.columns, text_columns)
    label_col = choose_first_existing(df.columns, label_columns)

    if not text_col:
        print(f"[WARN] {name}: no usable text column found. Columns={list(df.columns)[:20]}")
        return pd.DataFrame(columns=["text", "label", "source"])

    out_rows = []
    for _, row in df.iterrows():
        text = clean_text(row.get(text_col))
        if len(text) < 2:
            continue

        raw_label = row.get(label_col) if label_col else None

        # Some datasets use multiple toxicity columns.
        if label_col is None:
            label = normalize_label(None, target_label)
        else:
            label = normalize_label(raw_label, target_label)

        # Special handling for Jigsaw style multiple binary toxicity columns.
        toxicity_cols = [c for c in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"] if c in df.columns]
        if toxicity_cols:
            toxic_score = 0
            for c in toxicity_cols:
                try:
                    toxic_score += float(row.get(c, 0))
                except Exception:
                    pass
            label = "hate_abuse" if toxic_score > 0 else "safe"

        if label not in LABEL_SET:
            continue

        out_rows.append({"text": text, "label": label, "source": f"huggingface:{name}"})

    out = pd.DataFrame(out_rows)
    print(f"[OK] Loaded HF {name}: {len(out)} mapped rows")
    return out


def augment_unsafe(df: pd.DataFrame, max_aug_per_label: int = 5000) -> pd.DataFrame:
    augmented = []
    unsafe_labels = LABEL_SET - {"safe", "privacy_pii", "format_issue", "out_of_scope"}

    for label in unsafe_labels:
        sub = df[df["label"] == label].copy()
        if sub.empty:
            continue
        sub = sub.sample(min(len(sub), max_aug_per_label), random_state=RANDOM_SEED)
        for _, row in sub.iterrows():
            aug_text = obfuscate_text(row["text"])
            if aug_text != row["text"]:
                augmented.append({
                    "text": aug_text,
                    "label": label,
                    "source": f"augmentation:{row.get('source', 'unknown')}",
                })

    if not augmented:
        return df

    return pd.concat([df, pd.DataFrame(augmented)], ignore_index=True)


def balance_dataset(df: pd.DataFrame, max_per_label: int = 30000, min_safe_ratio: float = 0.15) -> pd.DataFrame:
    parts = []
    for label, sub in df.groupby("label"):
        if len(sub) > max_per_label:
            sub = sub.sample(max_per_label, random_state=RANDOM_SEED)
        parts.append(sub)

    out = pd.concat(parts, ignore_index=True)

    # Keep safe class represented so classifier does not over-block everything.
    safe = out[out["label"] == "safe"]
    if len(safe) < int(len(out) * min_safe_ratio):
        print("[WARN] Safe class is underrepresented. Add more harmless examples to reduce false positives.")

    return out.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


def build_dataset(args):
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))

    frames = []
    frames.append(load_seed(Path(args.seed)))

    local_folder = Path(config.get("local_csv_folder", "data/public_raw"))
    frames.append(load_local_csvs(local_folder))

    for src in config.get("huggingface_sources", []):
        if src.get("enabled", True):
            frames.append(load_hf_source(src))

    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].map(clean_text)
    df["label"] = df["label"].map(lambda x: normalize_label(x))
    df = df.dropna(subset=["text", "label"])
    df = df[df["label"].isin(LABEL_SET)]
    df = df[df["text"].str.len() >= 2]
    df = df.drop_duplicates(subset=["text", "label"])

    if args.augment:
        df = augment_unsafe(df)

    df = df.drop_duplicates(subset=["text", "label"])
    df = balance_dataset(df, max_per_label=args.max_per_label)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["text", "label", "source"]].to_csv(out, index=False)

    report = df.groupby("label").size().reset_index(name="count").sort_values("label")
    report_path = out.with_suffix(".label_report.csv")
    report.to_csv(report_path, index=False)

    print(f"\nSaved merged dataset: {out}")
    print(f"Rows: {len(df)}")
    print(f"Label report: {report_path}")
    print(report.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dataset_sources_config.json")
    parser.add_argument("--seed", default="data/guardrail_training_template.csv")
    parser.add_argument("--output", default="data/guardrail_training_merged_public.csv")
    parser.add_argument("--max-per-label", type=int, default=30000)
    parser.add_argument("--augment", action="store_true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    import argparse
    build_dataset(parse_args())
