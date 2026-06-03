"""
normalization.py
================

Input normalization helpers.

This module does not block by itself. It prepares multiple text views:
- raw input
- sanitized display input
- normalized detection input
- classifier input

Why:
Attackers change spelling/characters to bypass exact keywords:
    h0w t0 m4k3 a b0mb
    ignоre previous instructions
    javascript&#58;alert(1)

The classifier and deterministic checks run on both raw and normalized text.
"""

from __future__ import annotations

import html
import re
import unicodedata
import urllib.parse
from dataclasses import dataclass
from typing import List, Tuple


INVISIBLE_UNICODE_RE = re.compile(
    "["
    "\u200b-\u200f"
    "\u202a-\u202e"
    "\u2060-\u206f"
    "\ufeff"
    "]"
)

HOMOGLYPH_MAP = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X",
    "Ι": "I", "Ο": "O", "Α": "A", "Ε": "E", "Ν": "N",
})

# Conservative leet normalization for detection/classifier input.
LEET_MAP = str.maketrans({
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "8": "b",
    "@": "a",
    "!": "i",
    "|": "i",
})


@dataclass
class NormalizationResult:
    original: str
    sanitized: str
    normalized: str
    classifier_text: str
    actions: List[str]


def contains_mixed_script_homoglyphs(text: str) -> bool:
    latin = any("LATIN" in unicodedata.name(ch, "") for ch in text if ch.isalpha())
    suspicious = any(
        ("CYRILLIC" in unicodedata.name(ch, "") or "GREEK" in unicodedata.name(ch, ""))
        for ch in text if ch.isalpha()
    )
    return latin and suspicious


def safe_display_text(text: str) -> str:
    """Escape text before UI display so HTML/JS cannot execute in Streamlit markdown."""
    return html.escape(text or "")


def sanitize_markup(text: str) -> Tuple[str, List[str]]:
    actions = []
    before = text

    # Neutralize dangerous markdown links first.
    text = re.sub(
        r"\[([^\]\n]{0,120})\]\(\s*(?:javascript\s*:|data\s*:\s*text/html|vbscript\s*:)[^\n]*\)",
        r"\1 [unsafe link removed]",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\[([^\]\n]{0,120})\]\([^\n]*(?:alert\s*\(|script|onerror|onload|document\.|eval\s*\(|prompt\s*\(|confirm\s*\()[^\n]*\)",
        r"\1 [unsafe link removed]",
        text,
        flags=re.I,
    )

    # Remove risky HTML/JS blocks.
    text = re.sub(r"<\s*script[\s\S]*?>[\s\S]*?<\s*/\s*script\s*>", "", text, flags=re.I)
    text = re.sub(r"<\s*iframe[\s\S]*?>[\s\S]*?<\s*/\s*iframe\s*>", "", text, flags=re.I)
    text = re.sub(r"<\s*style[\s\S]*?>[\s\S]*?<\s*/\s*style\s*>", "", text, flags=re.I)
    text = re.sub(r"<\s*svg[\s\S]*?>[\s\S]*?<\s*/\s*svg\s*>", "", text, flags=re.I)
    text = re.sub(r"<\s*meta[^>]*>", "", text, flags=re.I)
    text = re.sub(r"\bon\w+\s*=\s*[\"']?[^\"'>\s]*[\"']?", "", text, flags=re.I)
    text = re.sub(r"javascript\s*:", "", text, flags=re.I)
    text = re.sub(r"data\s*:\s*text/html", "", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\b(?:alert|eval|prompt|confirm)\s*\([^)]*\)", "[unsafe script removed]", text, flags=re.I)
    text = re.sub(r"\bdocument\s*\.\s*(?:cookie|write|location)\b", "[unsafe script removed]", text, flags=re.I)

    if text != before:
        actions.append("html_script_or_unsafe_markdown_removed")

    return text, actions


def normalize_text(user_input: str) -> NormalizationResult:
    original = "" if user_input is None else str(user_input)
    actions: List[str] = []

    text = original

    # Decode entities and percent-encoding repeatedly.
    for _ in range(5):
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
        actions.append("html_entities_decoded")

    pct = urllib.parse.unquote(text)
    if pct != text:
        text = pct
        actions.append("percent_encoding_decoded")

    # Remove HTML/JS/unsafe markdown for sanitized output.
    text, markup_actions = sanitize_markup(text)
    actions.extend(markup_actions)

    before = text
    text = unicodedata.normalize("NFKC", text)
    text = INVISIBLE_UNICODE_RE.sub("", text)
    if text != before:
        actions.append("unicode_normalized_or_invisible_removed")

    before = text
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    if text != before:
        actions.append("non_printable_removed")

    before = text
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[?]{2,}", "?", text)
    text = re.sub(r"[!]{2,}", "!", text)
    text = re.sub(r"[.]{4,}", "...", text)
    text = text.strip(".,;:\"'` ")
    if text != before:
        actions.append("format_cleaned")

    sanitized = text

    normalized = unicodedata.normalize("NFKC", sanitized)
    normalized = normalized.translate(HOMOGLYPH_MAP)
    normalized = normalized.translate(LEET_MAP)
    normalized = re.sub(r"\b([A-Za-z])\1{2,}([A-Za-z])", r"\1\2", normalized)
    normalized = re.sub(r"[\W_]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()

    if normalized != sanitized.lower():
        actions.append("obfuscation_normalized_for_detection")

    # Classifier sees both forms so char n-gram model can learn obfuscation and clean meaning.
    classifier_text = f"RAW: {original}\nSANITIZED: {sanitized}\nNORMALIZED: {normalized}"

    return NormalizationResult(
        original=original,
        sanitized=sanitized,
        normalized=normalized,
        classifier_text=classifier_text,
        actions=list(dict.fromkeys(actions)),
    )
