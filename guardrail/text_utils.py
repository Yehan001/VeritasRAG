import base64
import html
import re
import unicodedata
from typing import List, Tuple

ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
SPACE_RE = re.compile(r"\s+")

LEET_TABLE = str.maketrans({
    "0": "o", "1": "i", "2": "z", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b", "9": "g",
    "@": "a", "$": "s", "!": "i", "|": "i", "+": "t", "€": "e", "£": "l",
})

HTML_DANGEROUS_REPLACEMENTS = [
    (re.compile(r"<\s*script\b[^>]*>.*?<\s*/\s*script\s*>", re.I | re.S), "[unsafe markup removed]"),
    (re.compile(r"<\s*iframe\b[^>]*>.*?<\s*/\s*iframe\s*>", re.I | re.S), "[unsafe markup removed]"),
    (re.compile(r"javascript\s*:", re.I), "[unsafe link removed]:"),
    (re.compile(r"on\w+\s*=", re.I), "[unsafe event removed]="),
]


def normalize_text(text: str) -> str:
    """Normalize text for safety checks. This is not used for display."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = text.translate(LEET_TABLE)
    text = text.lower()
    # reduce simple repeated characters: kiiill -> kiill, boooomb -> boomb
    text = re.sub(r"([a-z])\1{2,}", r"\1\1", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def sanitize_for_display(text: str) -> str:
    """Remove dangerous markup from the version shown/sent downstream."""
    if text is None:
        return ""
    cleaned = html.unescape(text)
    for pattern, replacement in HTML_DANGEROUS_REPLACEMENTS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned.strip()


def find_base64_payloads(text: str, min_len: int = 16) -> List[Tuple[str, str]]:
    """Return suspicious base64 tokens and their decoded text."""
    if not text:
        return []
    candidates = re.findall(r"(?<![A-Za-z0-9+/=])(?:[A-Za-z0-9+/]{%d,}={0,2})(?![A-Za-z0-9+/=])" % min_len, text)
    decoded = []
    for token in candidates:
        try:
            # pad and decode
            padded = token + "=" * (-len(token) % 4)
            raw = base64.b64decode(padded, validate=True)
            value = raw.decode("utf-8", errors="ignore")
            printable_ratio = sum(ch.isprintable() for ch in value) / max(1, len(value))
            if len(value.strip()) >= 8 and printable_ratio > 0.85:
                decoded.append((token, value.strip()))
        except Exception:
            continue
    return decoded


def simple_tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", normalize_text(text))
