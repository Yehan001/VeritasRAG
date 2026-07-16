import re
from typing import List, Tuple

from .schema import StageResult
from .text_utils import find_base64_payloads, normalize_text
from .taxonomy import LABEL_RISK


def _block(label: str, reason: str, score: float = 1.0) -> StageResult:
    return StageResult("deterministic_rules", "BLOCK", label, LABEL_RISK.get(label, "high"), reason, "rules", score)


def _warn(label: str, reason: str, score: float = 1.0) -> StageResult:
    return StageResult("deterministic_rules", "WARN", label, LABEL_RISK.get(label, "medium"), reason, "rules", score)


class RuleScanner:
    """High-confidence deterministic checks. This runs before all ML models."""

    def __init__(self):
        self.block_patterns: List[Tuple[str, re.Pattern, str]] = [
            ("script_or_command_injection", re.compile(r"<\s*/?\s*(script|iframe|object|embed|svg|img)\b|javascript\s*:|on\w+\s*=", re.I | re.S), "unsafe HTML/JavaScript markup"),
            ("script_or_command_injection", re.compile(r"\.\./|\.\.\\|/etc/passwd|/etc/shadow|cmd\.exe|powershell\s+-|rm\s+-rf|curl\s+http|wget\s+http|nc\s+-e", re.I), "path traversal or command execution pattern"),
            ("script_or_command_injection", re.compile(r"\b(union\s+select|drop\s+table|select\s+.+\s+from\s+users|or\s+1\s*=\s*1)\b", re.I | re.S), "SQL injection-like pattern"),
            ("script_or_command_injection", re.compile(r"\{\{.*?\}\}|<%.*?%>|\$\{.*?\}", re.I | re.S), "template injection-like pattern"),
            ("prompt_injection", re.compile(r"\b(ignore|disregard|forget|override)\b.{0,45}\b(previous|above|system|developer|instructions|rules)\b", re.I), "instruction override attempt"),
            ("prompt_injection", re.compile(r"\b(reveal|show|print|leak|dump)\b.{0,60}\b(system prompt|developer message|hidden prompt|internal instructions|chain of thought)\b", re.I), "system/developer prompt extraction attempt"),
            ("jailbreak", re.compile(r"\b(DAN|developer mode|jailbreak|unrestricted mode|no restrictions|bypass safety|bypass rules)\b", re.I), "jailbreak or unrestricted-role attempt"),
            ("cyber_abuse", re.compile(r"\b(hack|crack|break into|bypass)\b.{0,40}\b(wifi|wi-fi|website|login|account|server|password|authentication|admin panel)\b", re.I), "actionable unauthorized access request"),
            ("cyber_abuse", re.compile(r"\b(steal|dump|exfiltrate|harvest|phish|phishing|get)\b.{0,50}\b(passwords?|credentials?|cookies?|tokens?|session ids?|api keys?)\b", re.I), "credential theft or phishing request"),
            ("cyber_abuse", re.compile(r"\b(keylogger|ransomware|malware|trojan|reverse shell|botnet|payload to bypass login|sql injection payloads?)\b", re.I), "malware/exploit payload request"),
            ("violence", re.compile(r"\b(how\s+to\s+)?(kill|murder|stab|shoot|poison|strangle|assassinate)\b.{0,50}\b(someone|person|people|him|her|them|my enemy|teacher|boss)\b", re.I), "violent wrongdoing request"),
            ("violence", re.compile(r"\b(i\s+will|i'm\s+going\s+to|i\s+want\s+to)\b.{0,30}\b(kill|hurt|stab|shoot|attack)\b", re.I), "violent threat or intent"),
            ("self_harm", re.compile(r"\b(how\s+to\s+)?(suicide|kill myself|end my life|self harm|cut myself|overdose)\b", re.I), "self-harm request"),
            ("dangerous_content", re.compile(r"\b(make|m\$ke|mske|m4ke|m5ke|build|create|instructions?|recipe)\b.{0,50}\b(bomb|b0mb|boomb|explosive|detonator|poison|ricin|meth|napalm)\b", re.I), "dangerous weapons or harmful substance request"),
            ("extremism", re.compile(r"\b(isis|al[- ]?qaeda|terrorist|terrorism)\b.{0,50}\b(join|support|propaganda|recruit|manifesto|attack)\b", re.I), "extremism or terrorist content request"),
            ("child_safety", re.compile(r"\b(CSAM|child sexual|minor sexual|groom a minor|underage sexual)\b", re.I), "child safety violation"),
            ("fraud_scam", re.compile(r"\b(fake login|phishing page|scam email|spoof bank|carding|credit card dump|otp bypass)\b", re.I), "fraud or scam request"),
            ("hate_abuse", re.compile(r"\b(kill all|exterminate|gas)\b.{0,40}\b(immigrants|muslims|christians|jews|black people|women|tamils|sinhalese)\b", re.I), "hateful violent abuse"),
        ]
        self.warn_patterns: List[Tuple[str, re.Pattern, str]] = [
            ("format_issue", re.compile(r"(.)\1{20,}|([!?]){10,}"), "repeated characters or punctuation spam"),
            ("copyright_violation", re.compile(r"\b(copy|reproduce|give me)\b.{0,40}\b(full book|entire article|paid course|pirated|crack license)\b", re.I), "possible copyright misuse"),
            ("misinformation", re.compile(r"\b(fake news|spread false|make a convincing lie|disinformation campaign)\b", re.I), "possible misinformation request"),
        ]

    def scan(self, text: str) -> StageResult:
        raw = text or ""
        norm = normalize_text(raw)

        # Check raw and normalized versions before sanitization.
        for label, pattern, reason in self.block_patterns:
            if pattern.search(raw) or pattern.search(norm):
                return _block(label, f"Matched harmful pattern: {reason}")

        # Decode possible hidden prompt payloads and re-check.
        for token, decoded in find_base64_payloads(raw):
            decoded_result = self.scan(decoded)
            if decoded_result.action == "BLOCK":
                decoded_result.reason = f"Base64 payload decoded to unsafe text. {decoded_result.reason}"
                decoded_result.details["base64_token"] = token
                decoded_result.details["decoded_preview"] = decoded[:160]
                return decoded_result

        for label, pattern, reason in self.warn_patterns:
            if pattern.search(raw) or pattern.search(norm):
                return _warn(label, f"Matched warning pattern: {reason}")

        return StageResult("deterministic_rules", "ALLOW", "safe", "low", "No deterministic rule blocked the input.", "rules", 0.0)
