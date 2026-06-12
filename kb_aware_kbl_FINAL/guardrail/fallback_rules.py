import re
from typing import List

from .schema import SafetyResult
from .text_utils import normalize_text, sanitize_for_display, try_decode_base64_payloads

PROMPT_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"reveal (the )?(system|developer) prompt",
    r"show (me )?(the )?(system|developer) instructions",
    r"act as dan\b",
    r"developer mode",
    r"jailbreak",
    r"bypass (your|the) (rules|safety|restrictions)",
    r"you are now unrestricted",
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

FORMAT_PATTERNS = [
    r"(.)\1{12,}",
]


def fallback_safety_check(text: str) -> SafetyResult:
    normalized = normalize_text(text).lower()
    reasons: List[str] = []

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

    for pattern in TECH_ATTACK_PATTERNS:
        if re.search(pattern, normalized, re.I):
            return SafetyResult("script_or_command_injection", "block", "high", 0.98, "fallback_rules", [f"matched technical pattern: {pattern}"])

    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, normalized, re.I):
            return SafetyResult("prompt_injection", "block", "high", 0.98, "fallback_rules", [f"matched prompt injection pattern: {pattern}"])

    for label, patterns in HARMFUL_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, normalized, re.I):
                return SafetyResult(label, "block", "high", 0.96, "fallback_rules", [f"matched harmful pattern: {pattern}"])

    for pattern in FORMAT_PATTERNS:
        if re.search(pattern, normalized, re.I):
            return SafetyResult("format_issue", "warn", "low", 0.75, "fallback_rules", ["format/spam-like input"])

    return SafetyResult("safe", "allow", "low", 0.50, "fallback_rules", ["no fallback rule triggered"])
