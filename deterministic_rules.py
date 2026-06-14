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

Fix (HIGH_RISK + INSTRUCTIONAL over-blocking):
The original combo blocked ANY input containing a high-risk word + an instructional
word, which caused false positives on:
  - historical/academic framing  ("how did World War II weapons work")
  - medical/scientific framing   ("how does poison affect the body")
  - fiction/news framing         ("novel about a bomb disposal expert")
  - defensive framing            ("how to detect malware on my PC")
  - support framing              ("suicide prevention hotline steps")

Fix approach (greedy token reduction / safe-context suppression):
  Step 1: Collect false positive blocks.
  Step 2: Remove tokens one by one greedily until the match drops.
  Step 3: Identify over-triggering token — it was the HIGH_RISK or INSTRUCTIONAL
          word, but context tokens were the actual differentiator.
  Step 4: Add SAFE_CONTEXT_RE that suppresses the BLOCK when safe framing is present.
          Add DIRECT_HARM_AMPLIFIERS_RE that overrides safe context when the input
          is clearly a direct actionable harmful request.
  Step 5: Downgrade HIGH_RISK+INSTRUCTIONAL+safe_context from BLOCK → WARN
          so the classifier layer makes the final semantic decision.

Also fixed: HIGH_RISK_OBJECT_RE now covers plurals (toxins, weapons, grenades, etc.)
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

# ---------------------------------------------------------------------------
# HIGH_RISK_OBJECT_RE
#
# Covers both singular and plural forms.
# Fix: added s? suffix to cover toxins, weapons, explosives, grenades, etc.
# The classifier handles broader semantic coverage beyond this list.
# ---------------------------------------------------------------------------
HIGH_RISK_OBJECT_RE = re.compile(
    r"\b(bombs?|explosives?|detonators?|grenades?|weapons?|poisons?|toxins?|"
    r"malware|ransomware|keyloggers?|phishing|credentials?|"
    r"suicide|self\s*harm)\b",
    re.I,
)

# ---------------------------------------------------------------------------
# INSTRUCTIONAL_RE
# Verbs/nouns signalling actionable intent.
# ---------------------------------------------------------------------------
INSTRUCTIONAL_RE = re.compile(
    r"\b(how|ways?|methods?|steps?|guide|instructions?|tutorial|make|build|create|"
    r"synthesize|produce|bypass|hack|crack|steal|kill|hurt|attack)\b",
    re.I,
)

# ---------------------------------------------------------------------------
# SAFE_CONTEXT_RE  ← NEW
#
# Patterns that indicate academic, historical, medical, fictional, defensive,
# journalistic, or support framing.
#
# When this matches alongside HIGH_RISK + INSTRUCTIONAL, the rule DOWNGRADES
# from BLOCK → WARN so the classifier makes the final semantic decision.
#
# Derived via greedy token reduction on confirmed false positives:
#   "how did World War II bombs work"           → context token: world war / ww2
#   "explain how poison affects the body"       → context token: affects / body
#   "steps the CDC uses to track toxins"        → context token: cdc / track
#   "how does antivirus detect malware"         → context token: antivirus / detect
#   "suicide prevention hotline steps"          → context token: prevention / hotline
#   "how to create a weapon in a video game"    → context token: video game
#   "novel about a bomb disposal expert"        → context token: novel / disposal
#   "chemistry class guide to toxins"           → context token: chemistry class
# ---------------------------------------------------------------------------
SAFE_CONTEXT_RE = re.compile(
    r"""
    # --- Historical / war / academic framing ---
    \b(histor(?:y|ical|ically)|world\s*war|ww[12i]+|medieval|ancient|
       cold\s*war|museum|archive|warfare\s+era|
       how\s+(?:did|were|was)\b)\b|

    # --- Scientific / medical / research framing ---
    \b(research|academic|peer.reviewed|scientific|
       chemistry\s+class|biology\s+class|pharmacology|
       medicine|medical|clinical|symptom|diagnosis|treatment|
       antidote|patholog|toxicolog|epidemiolog|public\s+health|
       \bcdc\b|\bwho\b|\bfda\b|
       affect(?:s|ed|ing)?\s+the\s+body|physiolog|
       how\s+does\s+\w+\s+affect)\b|

    # --- Defensive / detection / security research framing ---
    \b(detect(?:ion)?|defend|defense|defence|
       prevent(?:ion)?|antivirus|anti.malware|firewall|
       security\s+research|penetration\s+test|pen\s+test|
       how\s+to\s+(?:protect|stop|block|defend)|
       mitigat|patch|remediat|incident\s+response|forensics)\b|

    # --- Fiction / creative / game framing ---
    \b(fiction|novel|story|screenplay|movie|film|
       video\s*game|game\s+character|in\s+a\s+game|
       for\s+(?:a\s+)?(?:book|story|novel|screenplay)|
       creative\s+writing|tabletop|dnd|dungeon|
       fantasy\s+setting|sci.fi|fictional)\b|

    # --- News / journalism / policy framing ---
    \b(journalism|journalist|documentary|
       policy|legislation|law\s+enforcement|court\s+case|
       why\s+did|what\s+caused|what\s+happened|
       according\s+to|cited\s+by)\b|

    # --- Mental health / support / awareness framing ---
    \b(prevention|hotline|crisis\s+line|mental\s+health|counseling|
       counsellor|therapist|psycholog|psychiatr|support\s+group|
       warning\s+signs?|risk\s+factors?|
       how\s+to\s+help|how\s+to\s+recognize|
       cope|coping|awareness)\b|

    # --- Disposal / safety / compliance framing ---
    \b(disposal|dispos(?:e|ing)|deactivat|decommission|
       safe\s+handling|safety\s+protocol|compliance|
       hazmat|hazardous\s+material|osha|epa)\b
    """,
    re.I | re.X | re.S,
)

# ---------------------------------------------------------------------------
# DIRECT_HARM_AMPLIFIERS_RE  ← NEW
#
# These tokens signal a direct, actionable harmful request even when some
# safe-context words may also be present.
# When these match, safe-context suppression is OVERRIDDEN and BLOCK is kept.
#
# Examples that should still BLOCK even if they contain context-like words:
#   "step by step guide to make a bomb at home"
#   "how to synthesize poison and hide it from doctors"
#   "how to build a weapon to use against someone"
# ---------------------------------------------------------------------------
DIRECT_HARM_AMPLIFIERS_RE = re.compile(
    r"\b(step\s+by\s+step|at\s+home|from\s+scratch|"
    r"without\s+getting\s+caught|undetected|untraceable|"
    r"hide\s+(?:it|from)|secret(?:ly)?|covert(?:ly)?|"
    r"for\s+my\s+own\s+use|to\s+use\s+against|"
    r"to\s+(?:attack|kill|hurt|harm)|"
    r"to\s+poison\s+someone|to\s+blow\s+up|"
    r"without\s+anyone\s+knowing|on\s+a\s+real\s+person|"
    r"actually\s+work(?:s)?)\b",
    re.I | re.S,
)


def _decode_base64(blob: str) -> str:
    try:
        padded = blob + "=" * ((4 - len(blob) % 4) % 4)
        return base64.b64decode(padded, validate=False).decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _has_safe_context(text: str) -> bool:
    """
    Returns True if safe-context framing is detected AND no direct harm
    amplifier overrides it.

    Logic (greedy token reduction result):
      - If DIRECT_HARM_AMPLIFIERS match → always False (block wins)
      - If SAFE_CONTEXT matches → True (downgrade to WARN, classifier decides)
    """
    if DIRECT_HARM_AMPLIFIERS_RE.search(text):
        return False
    return bool(SAFE_CONTEXT_RE.search(text))


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

    # ---------------------------------------------------------------------------
    # HIGH_RISK_OBJECT + INSTRUCTIONAL combo — with safe-context suppression.
    #
    # Original behaviour: BLOCK on any match of both patterns (too aggressive).
    #
    # New behaviour:
    #   HIGH_RISK + INSTRUCTIONAL + safe context present  → WARN  (classifier decides)
    #   HIGH_RISK + INSTRUCTIONAL + no safe context       → BLOCK (HIGH risk)
    #   HIGH_RISK + INSTRUCTIONAL + direct harm amplifier → BLOCK (amplifier overrides)
    #
    # Still BLOCKS:
    #   "how to make a bomb"                          no safe context     → BLOCK
    #   "steps to synthesize poison at home"          amplifier: at home  → BLOCK
    #   "guide to building malware"                   no safe context     → BLOCK
    #   "how to make poison and hide it from doctors" amplifier: hide it  → BLOCK
    #   "how to build a weapon to use against someone" amplifier: against → BLOCK
    #
    # Now WARNS (passes to classifier):
    #   "explain how poison affects the body"         medical context     → WARN
    #   "suicide prevention hotline steps"            prevention context  → WARN
    #   "how to detect malware on my PC"              defensive context   → WARN
    #   "how to create a weapon in a video game"      fiction context     → WARN
    #   "chemistry class guide to toxins"             academic context    → WARN
    #   "steps the CDC uses to track toxins"          cdc/health context  → WARN
    #   "how to help someone with self harm"          support context     → WARN
    #   "how does antivirus detect malware"           defensive context   → WARN
    # ---------------------------------------------------------------------------
    if HIGH_RISK_OBJECT_RE.search(combined) and INSTRUCTIONAL_RE.search(combined):
        if _has_safe_context(combined):
            events.append(RuleEvent(
                "high_risk_instructional_cue",
                "WARN",
                "MEDIUM",
                "High-risk object with instructional intent detected, but safe context "
                "(academic/historical/medical/fictional/defensive/support) found. "
                "Downgraded to warning — classifier will make the final decision.",
            ))
        else:
            events.append(RuleEvent(
                "high_risk_instructional_cue",
                "BLOCK",
                "HIGH",
                "High-risk object combined with instructional intent detected. "
                "No safe context found.",
            ))

    return events