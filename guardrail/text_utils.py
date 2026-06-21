import base64
import html
import re
import unicodedata
from typing import List, Tuple

LEET_MAP = str.maketrans({
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
    "!": "i",
})

ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
SCRIPT_RE = re.compile(r"<\s*(script|iframe|object|embed|svg|img)\b[^>]*>|on\w+\s*=|javascript\s*:", re.I)
MARKDOWN_JS_RE = re.compile(r"\[([^\]]+)\]\(\s*javascript:[^)]+\)", re.I)
HTML_TAG_RE = re.compile(r"<[^>]+>")
BASE64_RE = re.compile(r"\b(?:[A-Za-z0-9+/]{20,}={0,2})\b")
# Whole words made of one letter repeated (e.g. "aa", "iii") -> single letter.
# Scoped to entire-word matches so real words with double letters (kill, bomb,
# bypass, passwords, ...) are never touched. This closes the evasion where a
# short word like the article "a" is doubled to dodge a literal "(a )?" group
# in a harmful-pattern regex, e.g. "how to create aa bomb".
SINGLE_CHAR_WORD_RE = re.compile(r"\b([a-zA-Z])\1+\b")


def normalize_text(text: str) -> str:
    """Normalize user input for safer detection."""
    if text is None:
        return ""
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = text.translate(LEET_MAP)
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)  # killllll -> killl-ish but reduces spam
    text = SINGLE_CHAR_WORD_RE.sub(r"\1", text)  # "aa" -> "a", "iii" -> "i" (doubled-article evasion)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sanitize_for_display(text: str) -> Tuple[str, List[str]]:
    """Remove risky HTML/JS from text displayed or forwarded downstream."""
    events: List[str] = []
    out = text or ""
    if MARKDOWN_JS_RE.search(out):
        events.append("markdown_javascript_link_removed")
        out = MARKDOWN_JS_RE.sub(r"\1 [unsafe link removed]", out)
    if SCRIPT_RE.search(out):
        events.append("unsafe_html_or_script_detected")
        out = SCRIPT_RE.sub(" [unsafe markup removed] ", out)
    if HTML_TAG_RE.search(out):
        events.append("html_tags_removed")
        out = HTML_TAG_RE.sub(" ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out, events


def try_decode_base64_payloads(text: str) -> List[str]:
    """Return readable decoded base64 payloads found in text."""
    decoded: List[str] = []
    for match in BASE64_RE.findall(text or ""):
        try:
            padded = match + "=" * (-len(match) % 4)
            raw = base64.b64decode(padded, validate=True)
            s = raw.decode("utf-8", errors="ignore")
            s_norm = normalize_text(s)
            if len(s_norm) >= 8 and re.search(r"[A-Za-z]", s_norm):
                decoded.append(s_norm)
        except Exception:
            continue
    return decoded


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]*", (text or "").lower())