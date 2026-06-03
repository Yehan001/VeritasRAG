"""
deterministic_rules.py
======================

Manual deterministic rules.

Important design decision:
This file is NOT meant to manually define every harmful sentence in the world.

Rules are used only where manual detection is appropriate:
- technical attack signatures
- prompt-injection templates
- jailbreak templates
- URLs
- encoded payloads
- structure/spam checks
- obvious high-risk lexical cues

Flexible harmful meaning is handled by the classifier layer.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import List

from normalization import NormalizationResult, contains_mixed_script_homoglyphs


@dataclass
class RuleEvent:
    name: str
    action: str  # BLOCK | WARN | PASS
    risk: str
    message: str


URL_RE = re.compile(r"\b(?:https?://|www\.)\S+|\b\S+\.(?:com|net|org|io|lk|edu|gov|co|me|app|dev|xyz)\b", re.I)
SHORT_URL_RE = re.compile(r"\b(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|is\.gd|rebrand\.ly|ow\.ly|buff\.ly|cutt\.ly|shorturl\.at|tiny\.cc)\b", re.I)

BASE64_RE = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/])")
HEX_RE = re.compile(r"\b(?:0x)?[A-Fa-f0-9]{80,}\b")

PROMPT_INJECTION_RE = re.compile(
    r"(ignore|disregard|forget|override)\s+(all\s+)?(previous|prior|above|earlier)?\s*"
    r"(instructions?|prompts?|context|rules?)|"
    r"(reveal|print|show)\s+(the\s+)?(system|developer|hidden)\s+(prompt|instructions?|message)|"
    r"system\s*prompt|developer\s+(message|instruction|prompt)|"
    r"use\s+outside\s+knowledge\s+instead|answer\s+from\s+your\s+(training|memory|knowledge)",
    re.I | re.S,
)

JAILBREAK_RE = re.compile(
    r"\b(DAN|jailbreak|developer\s+mode|unrestricted\s+mode|god\s+mode|do\s+anything\s+now)\b|"
    r"you\s+are\s+now\s+(free|unrestricted|unfiltered)|"
    r"no\s+(ethical|moral|content)\s+(constraints?|restrictions?|limits?)",
    re.I | re.S,
)

SCRIPT_COMMAND_RE = re.compile(
    r"<\s*script|</\s*script\s*>|javascript\s*:|vbscript\s*:|data\s*:\s*text/html|"
    r"on(load|click|error|mouseover|focus|submit)\s*=|<\s*iframe|<\s*img|<\s*svg|"
    r"document\.(cookie|write|location)|eval\s*\(|alert\s*\(|prompt\s*\(|confirm\s*\(|"
    r"\brm\s+-rf\b|\bcat\s+/etc/passwd\b|\bcurl\s+https?://|\bwget\s+https?://|"
    r"\bpowershell\b|\bcmd\.exe\b|\bchmod\s+777\b|\bsudo\s+rm\b|\bmkfs\b|"
    r"\bSELECT\s+.+FROM\s+.+WHERE\b|\bUNION\s+SELECT\b|"
    r"\b(DROP|DELETE|INSERT|UPDATE)\s+(TABLE|FROM|INTO)\b",
    re.I | re.S,
)

PATH_TEMPLATE_RE = re.compile(
    r"(\.\./|\.\.\\){1,}|/etc/passwd|c:\\windows\\|file\s*:\s*//|"
    r"\{\{.*?\}\}|\$\{.*?\}|<%.*?%>|\{%.*?%\}",
    re.I | re.S,
)


EXTREMISM_RE = re.compile(
    r"\b(?:isis|islamic\s+state|isil|al[-\s]?qaeda|al[-\s]?qaida|taliban|terrorist\s+(?:group|organization|organisation)|extremist\s+(?:group|organization|organisation))\b|"
    r"\b(?:act|pretend|roleplay)\s+(?:like|as)\s+(?:a|an)?\s*(?:isis|islamic\s+state|isil|al[-\s]?qaeda|al[-\s]?qaida|taliban|terrorist|extremist)\s*(?:member|fighter|supporter)?\b|"
    r"\b(?:join|support|fund|recruit|promote|praise|pledge\s+allegiance\s+to)\b.{0,80}\b(?:isis|islamic\s+state|isil|al[-\s]?qaeda|al[-\s]?qaida|taliban|terrorist|extremist)\b|"
    r"\b(?:terrorist|extremist)\b.{0,80}\b(?:manifesto|propaganda|recruitment|training|attack|member)\b",
    re.I | re.S,
)

CHILD_SAFETY_RE = re.compile(
    r"\b(?:csam|child\s+exploitation|sexual\s+(?:content|images?|material)\s+(?:of|involving)\s+(?:a\s+)?(?:child|minor)|minor\s+sexual\s+content|groom\s+(?:a\s+)?minor|exploit\s+(?:a\s+)?child)\b",
    re.I | re.S,
)

# Not an exhaustive harmful list. This is a high-confidence lexical safety net.
# The classifier is responsible for broader semantic coverage.
HIGH_RISK_OBJECT_RE = re.compile(
    r"\b(bomb|explosive|detonator|grenade|weapon|poison|toxin|malware|ransomware|keylogger|phishing|credential|suicide|self\s*harm)\b",
    re.I,
)

INSTRUCTIONAL_RE = re.compile(
    r"\b(how|ways?|methods?|steps?|guide|instructions?|tutorial|make|build|create|synthesize|produce|bypass|hack|crack|steal|kill|hurt|attack)\b",
    re.I,
)


def _decode_base64(blob: str) -> str:
    try:
        padded = blob + "=" * ((4 - len(blob) % 4) % 4)
        return base64.b64decode(padded, validate=False).decode("utf-8", errors="ignore")
    except Exception:
        return ""


def run_deterministic_rules(norm: NormalizationResult, block_urls: bool = True) -> List[RuleEvent]:
    events: List[RuleEvent] = []
    raw = norm.original
    sanitized = norm.sanitized
    normalized = norm.normalized
    combined = f"{raw}\n{sanitized}\n{normalized}"

    if not raw or not raw.strip():
        events.append(RuleEvent("empty_input", "BLOCK", "LOW", "Input is empty."))
        return events

    if "\x00" in raw:
        events.append(RuleEvent("null_byte_injection", "BLOCK", "HIGH", "Null byte detected."))

    if contains_mixed_script_homoglyphs(raw):
        events.append(RuleEvent("homoglyph_attack", "BLOCK", "HIGH", "Mixed-script lookalike characters detected."))

    if norm.actions:
        for action in norm.actions:
            risk = "MEDIUM" if "html" in action or "script" in action else "LOW"
            events.append(RuleEvent(action, "WARN", risk, f"Normalization/sanitization applied: {action}"))

    # Structure checks
    words = re.findall(r"\b[A-Za-z]{2,}\b", sanitized)
    if len(sanitized) < 3:
        events.append(RuleEvent("too_short", "BLOCK", "LOW", "Input is too short."))
    if len(sanitized) > 1000:
        events.append(RuleEvent("too_long", "BLOCK", "MEDIUM", "Input is too long."))
    if not re.search(r"[A-Za-z]", sanitized):
        events.append(RuleEvent("no_real_words", "BLOCK", "LOW", "Input contains no meaningful words."))

    compact = re.sub(r"\s+", "", sanitized.lower())
    longest_run = max((len(m.group(0)) for m in re.finditer(r"(.)\1+", compact)), default=1)
    if longest_run >= 7:
        events.append(RuleEvent("repeated_chars", "BLOCK", "LOW", "Repeated character spam detected."))

    if len(words) >= 4:
        lowered = [w.lower() for w in words]
        max_count = max(lowered.count(w) for w in set(lowered))
        if max_count >= 4 and max_count / len(lowered) >= 0.60:
            events.append(RuleEvent("repeated_phrase", "BLOCK", "LOW", "Repeated word/phrase spam detected."))

    # Deterministic attacks
    if PROMPT_INJECTION_RE.search(combined):
        events.append(RuleEvent("prompt_injection", "BLOCK", "HIGH", "Prompt injection or instruction override detected."))

    if JAILBREAK_RE.search(combined):
        events.append(RuleEvent("jailbreak", "BLOCK", "HIGH", "Jailbreak attempt detected."))

    if SCRIPT_COMMAND_RE.search(combined):
        events.append(RuleEvent("script_command_sql_injection", "BLOCK", "HIGH", "Script, command, or SQL injection pattern detected."))

    if PATH_TEMPLATE_RE.search(combined):
        events.append(RuleEvent("path_or_template_injection", "BLOCK", "HIGH", "Path traversal or template injection detected."))

    if EXTREMISM_RE.search(combined):
        events.append(RuleEvent("extremism", "BLOCK", "HIGH", "Extremist or terrorist support/roleplay/instruction content detected."))

    if CHILD_SAFETY_RE.search(combined):
        events.append(RuleEvent("child_safety", "BLOCK", "HIGH", "Child-safety violation content detected."))

    # URL handling
    no_email_raw = re.sub(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "[EMAIL]", raw)
    if SHORT_URL_RE.search(no_email_raw):
        events.append(RuleEvent("short_url", "BLOCK", "HIGH", "Shortened URL detected."))
    elif URL_RE.search(no_email_raw) and block_urls:
        events.append(RuleEvent("url_present", "BLOCK", "MEDIUM", "External URLs are blocked by this input layer."))

    # Encoded payloads
    if HEX_RE.search(raw):
        events.append(RuleEvent("encoded_payload_hex", "BLOCK", "HIGH", "Long hexadecimal payload detected."))

    for blob in BASE64_RE.findall(raw):
        decoded = _decode_base64(blob)
        if decoded:
            decoded_norm = decoded.lower()
            if PROMPT_INJECTION_RE.search(decoded_norm) or SCRIPT_COMMAND_RE.search(decoded_norm) or HIGH_RISK_OBJECT_RE.search(decoded_norm):
                events.append(RuleEvent("encoded_payload_base64", "BLOCK", "HIGH", "Base64 payload decodes to unsafe content."))
            else:
                events.append(RuleEvent("encoded_payload_base64", "BLOCK", "MEDIUM", "Base64-like encoded payload detected."))

    # High-confidence harmful cue + instruction pattern.
    # This is not meant to list every sentence. It catches obvious short obfuscations,
    # then the classifier handles semantic variations.
    if HIGH_RISK_OBJECT_RE.search(combined) and INSTRUCTIONAL_RE.search(combined):
        events.append(RuleEvent("high_risk_instructional_cue", "BLOCK", "HIGH", "High-risk object combined with instructional intent detected."))

    return events
