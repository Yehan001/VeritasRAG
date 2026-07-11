import re
from typing import Dict, List, Pattern, Tuple

from .schema import SafetyResult
from .text_utils import normalize_text, sanitize_for_display, try_decode_base64_payloads

# ---------------------------------------------------------------------------
# Fuzzy matching: tolerate letter-doubling/typo evasion INSIDE words
# (e.g. "boomb", "kiill", "bbypass") that the whole-word fix in
# text_utils.SINGLE_CHAR_WORD_RE cannot catch, since that only collapses
# standalone repeated-letter tokens like "aa", not repeats inside longer words.
# ---------------------------------------------------------------------------

_REGEX_METACHARS = set("\\()|?*+.[]{}^$")


def fuzzify(pattern: str, max_repeat: int = 3) -> str:
    """
    Rewrite a literal-word regex pattern so every plain letter tolerates being
    repeated up to `max_repeat` times.

        'kill' -> 'k{1,3}i{1,3}l{1,3}l{1,3}'

    Regex syntax passes through untouched: escape sequences (\\s, \\b, \\1, ...)
    and metacharacters ( ) | ? * + . [ ] { } ^ $ are preserved, so existing
    alternations like (kill|murder|stab) and groups like (a )? keep working
    exactly as written -- only the literal letters inside become fuzzy.

    max_repeat=3 is deliberately small and bounded (not '+' or '*'): every
    input already passes through text_utils.normalize_text first, which
    collapses any run of 4+ identical characters down to at most 2 before
    these patterns ever run. A bounded {1,3} is enough headroom for that
    upstream guarantee, keeps matching fast, and avoids any backtracking
    blow-up risk that unbounded quantifiers could introduce on adversarial input.
    """
    out = []
    i, n = 0, len(pattern)
    while i < n:
        ch = pattern[i]
        if ch == "\\" and i + 1 < n:
            out.append(pattern[i:i + 2])
            i += 2
            continue
        if ch in _REGEX_METACHARS:
            out.append(ch)
            i += 1
            continue
        if ch.isalpha():
            out.append(f"{ch}{{1,{max_repeat}}}")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _compile_fuzzy_list(patterns: List[str]) -> List[Tuple[str, Pattern]]:
    """Pair each raw pattern with its compiled fuzzy version, so reason
    messages shown to users/audit logs can still display the original
    readable pattern instead of the expanded fuzzy regex."""
    return [(raw, re.compile(fuzzify(raw), re.I)) for raw in patterns]


def _compile_fuzzy_dict(pattern_dict: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, Pattern]]]:
    return {label: _compile_fuzzy_list(patterns) for label, patterns in pattern_dict.items()}


# ---------------------------------------------------------------------------
# Pattern definitions -- unchanged from before. Only how they're matched
# (fuzzy-compiled below) has changed, so this list stays the single source
# of truth and is easy to keep extending.
# ---------------------------------------------------------------------------

PROMPT_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"follow (my )?instructions (from now on|from now onwards|from here on)",
    r"reveal (the )?(system|developer) prompt",
    r"show (me )?(the )?(system|developer) prompt",
    r"show (me )?(the )?(system|developer) instructions",
    r"give me (the )?(system|developer) prompt",
    r"tell me (the )?(system|developer) prompt",
    r"act as dan\b",
    r"developer mode",
    r"jailbreak",
    r"bypass (your|the) (rules|safety|restrictions)",
    r"you are now unrestricted",
    r"override (your|the) (rules|safety|restrictions)",
    r"forget (the )?(previous|prior|above) instructions",
]

TECH_ATTACK_PATTERNS = [
    r"<\s*script\b",
    r"javascript\s*:",
    r"onerror\s*=",
    r"\.\./\.\.",
    r"/etc/passwd",
    r"\brm\s+-rf\b",
    r"\bcat\s+/etc/passwd\b",
    r"\{\{.*\}\}",
    r"\bselect\b.+\bfrom\b.+\busers\b",
    r"\bdrop\s+table\b",
]

HARMFUL_PATTERNS = {
    "violence": [
        r"how to\s+(kill|murder|stab|poison|hurt|attack)\s+(someone|a person|people|him|her|them)",
        r"i want to\s+(kill|murder|hurt|attack)\s+(someone|a person|people|him|her|them|you)",
        r"ways to\s+(kill|murder|stab|poison)\s+(someone|a person|people)",
    ],
    "self_harm": [
        r"\bkill myself\b", r"\bend my life\b", r"\bsuicide\b", r"\bhurt myself\b",
        r"how to die", r"i want to die",
    ],
    "cyber_abuse": [
        r"steal (passwords?|credentials?|logins?)",
        r"hack (a )?(wifi|website|account|server|network)",
        r"bypass (a )?(login|password|authentication)",
        r"phishing page", r"keylogger", r"credential theft",
    ],
    "dangerous_content": [
        r"make (a )?(bomb|explosive|detonator)",
        r"how to (make|build|create) (a )?(bomb|explosive|weapon)",
        r"poison recipe", r"chemical weapon",
    ],
    "extremism": [
        r"join (isis|al qaeda|terrorist)", r"act like (a )?(isis|terrorist)",
        r"terrorist propaganda", r"support (isis|terrorism)",
    ],
    "child_safety": [
        r"\bcsam\b", r"child sexual", r"sexual content involving (a )?(minor|child)",
        r"groom (a )?(minor|child)",
    ],
    "fraud_scam": [
        r"phishing email", r"fake login", r"credit card fraud", r"scam someone",
        r"steal bank", r"fake bank", r"social engineering script",
    ],
    "copyright_violation": [
        r"give me the full copyrighted", r"pirated copy", r"crack (this )?software",
        r"bypass license", r"remove drm",
    ],
    "hate_abuse": [
        r"subhuman", r"deserve to die", r"exterminate (all )?",
    ],
}

# NOT fuzzified: this is a structural spam/flood check (a backreference
# counting repeated characters), not a natural-language word list, so
# letter-by-letter fuzzification doesn't apply to it.
FORMAT_PATTERNS = [
    r"(.)\1{12,}",
]

# Compiled once at import time, not on every check() call.
_FUZZY_PROMPT_INJECTION = _compile_fuzzy_list(PROMPT_INJECTION_PATTERNS)
_FUZZY_TECH_ATTACK = _compile_fuzzy_list(TECH_ATTACK_PATTERNS)
_FUZZY_HARMFUL = _compile_fuzzy_dict(HARMFUL_PATTERNS)
_COMPILED_FORMAT = [re.compile(p, re.I) for p in FORMAT_PATTERNS]


def _has_prompt_injection_intent(normalized: str) -> bool:
    words = set(re.findall(r"[a-z0-9]+", normalized.lower()))
    if not words:
        return False

    if {"ignore", "instructions"}.issubset(words):
        return True
    if {"follow", "instructions"}.issubset(words):
        return True
    if {"forget", "instructions"}.issubset(words):
        return True
    if {"reveal", "system"}.issubset(words) and "prompt" in words:
        return True
    if {"show", "system"}.issubset(words) and "prompt" in words:
        return True
    if {"tell", "system"}.issubset(words) and "prompt" in words:
        return True
    if {"give", "system"}.issubset(words) and "prompt" in words:
        return True
    if {"bypass", "rules"}.issubset(words) or {"override", "rules"}.issubset(words):
        return True
    if {"bypass", "safety"}.issubset(words) or {"override", "safety"}.issubset(words):
        return True
    return False


def _has_violent_intent(normalized: str) -> bool:
    words = set(re.findall(r"[a-z0-9]+", normalized.lower()))
    if not words:
        return False

    violent_terms = {"kill", "murder", "stab", "poison", "hurt", "attack", "harm", "injure", "shoot", "destroy", "assault", "torture"}
    target_terms = {"someone", "person", "people", "them", "him", "her", "you", "victim", "another", "anyone"}

    if violent_terms & words and target_terms & words:
        return True
    if {"how", "do", "i"}.issubset(words) and violent_terms & words:
        return True
    return False


def fallback_safety_check(text: str) -> SafetyResult:
    normalized = normalize_text(text).lower()

    sanitized, sanitize_events = sanitize_for_display(text)
    if sanitize_events:
        return SafetyResult(
            label="script_or_command_injection",
            decision="block",
            risk_level="high",
            confidence=0.99,
            backend="fallback_rules",
            reasons=sanitize_events,
        )

    decoded_payloads = try_decode_base64_payloads(text)
    for payload in decoded_payloads:
        payload_result = fallback_safety_check(payload)
        if payload_result.decision == "block":
            payload_result.backend = "fallback_rules_base64"
            payload_result.reasons.append("base64_payload_decoded_as_unsafe")
            return payload_result

    if _has_prompt_injection_intent(normalized):
        return SafetyResult("prompt_injection", "block", "high", 0.97, "fallback_rules", ["matched prompt-injection intent heuristics"])

    if _has_violent_intent(normalized):
        return SafetyResult("violence", "block", "high", 0.96, "fallback_rules", ["matched violent-intent heuristics"])

    for raw, pattern in _FUZZY_TECH_ATTACK:
        if pattern.search(normalized):
            return SafetyResult("script_or_command_injection", "block", "high", 0.98, "fallback_rules", [f"matched technical pattern: {raw}"])

    for raw, pattern in _FUZZY_PROMPT_INJECTION:
        if pattern.search(normalized):
            return SafetyResult("prompt_injection", "block", "high", 0.98, "fallback_rules", [f"matched prompt injection pattern: {raw}"])

    for label, patterns in _FUZZY_HARMFUL.items():
        for raw, pattern in patterns:
            if pattern.search(normalized):
                return SafetyResult(label, "block", "high", 0.96, "fallback_rules", [f"matched harmful pattern: {raw}"])

    for pattern in _COMPILED_FORMAT:
        if pattern.search(normalized):
            return SafetyResult("format_issue", "warn", "low", 0.75, "fallback_rules", ["format/spam-like input"])

    return SafetyResult("safe", "allow", "low", 0.50, "fallback_rules", ["no fallback rule triggered"])