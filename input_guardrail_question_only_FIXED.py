"""
input_guardrail.py
==================

Standalone QUESTION-ONLY input filtering and guardrail system for a
Document QA / RAG pipeline.

Purpose
-------
The uploaded document is treated as the trusted knowledge source for retrieval.
Only the user's question is filtered before it reaches retrieval and the LLM.

Filtering layers covered
-----------------------
1. Structural / format:
   empty, too short, too long, no real words, repeated character spam,
   non-printable/binary characters, unicode abuse, excessive punctuation,
   excessive caps, repeated phrases.

2. Security / injection:
   prompt injection, jailbreak, script/HTML/JS/SQL injection,
   template injection, path traversal, null-byte injection,
   homoglyph/lookalike attacks.

3. Content safety:
   harmful intent, hate speech, profanity, self-harm, child safety,
   extremist/terrorist content.

4. Privacy:
   email, phone numbers, Sri Lankan NIC, credit cards, passport numbers,
   bank account-like patterns, address/location-like patterns.

5. Language & linguistics:
   non-English detection, spelling-quality warning, question-structure warning,
   gibberish/random letters, leetspeak/obfuscation.

6. Semantic / intent:
   out-of-scope questions, system-prompt requests, social engineering,
   roleplay/pretend attempts, unrelated opinions/predictions.

Design notes
------------
- The guardrail blocks high-risk input.
- Some quality issues only warn unless strict_mode=True.
- Sanitization never rewrites meaning and never auto-corrects spelling.
- The output keeps the original and sanitized question separately.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    QUESTION_MIN_CHARS = 5
    QUESTION_MAX_CHARS = 800
    QUESTION_MIN_WORDS = 2
    QUESTION_MAX_WORDS = 120

    GARBAGE_CHAR_THRESHOLD = 0.30
    REPEATED_CHAR_THRESHOLD = 0.60
    CAPS_THRESHOLD = 0.72
    SPELL_ERROR_THRESHOLD = 0.45
    GIBBERISH_THRESHOLD = 0.58

    MAX_WARNINGS_BEFORE_MEDIUM_RISK = 3


# ─────────────────────────────────────────────────────────────────────────────
# PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

class P:
    WORD = re.compile(r"\b[a-zA-Z]{2,}\b")
    TOKEN = re.compile(r"[A-Za-z0-9_@.+:/\\-]+")
    EXCESS_PUNCT = re.compile(r"[!?.,;:]{4,}")
    REPEAT_PHRASE = re.compile(
        r"\b(\w{3,}(?:\s+\w+){0,3})\b(?:.*\b\1\b){2,}",
        re.IGNORECASE,
    )

    # Unicode control / invisible / direction override characters.
    UNICODE_ABUSE = re.compile(
        "["
        "\u200b-\u200f"   # zero-width + direction marks
        "\u202a-\u202e"   # bidi overrides
        "\u2060-\u206f"   # word joiners/invisible controls
        "\ufeff"          # BOM
        "]"
    )

    NULL_BYTE = re.compile(r"\x00")

    PROMPT_INJECTION = [re.compile(p, re.IGNORECASE) for p in [
        r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?)",
        r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
        r"forget\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?|context)",
        r"do\s+not\s+use\s+(the\s+)?(provided\s+)?(document|context)",
        r"use\s+outside\s+knowledge\s+instead",
        r"answer\s+from\s+your\s+(training|memory|knowledge)",
        r"system\s*:\s*you\s+are\s+now",
        r"system\s*prompt",
        r"\[INST\]|\[/INST\]|\[SYS\]|\[/SYS\]",
        r"<\|im_start\|>|<\|im_end\|>",
        r"###\s*(system|instruction|prompt)",
        r"your\s+(new\s+)?(instructions?|rules?|prompt)\s+(are|is)\s*:",
        r"override\s+(your\s+)?(safety|filter|restriction|guideline|rule|instruction)",
        r"new\s+instructions?\s*:",
        r"reveal\s+(the\s+)?(hidden|system|developer)\s+(prompt|instructions?)",
        r"print\s+(the\s+)?(system|developer)\s+(prompt|instructions?)",
    ]]

    JAILBREAK = [re.compile(p, re.IGNORECASE) for p in [
        r"\bDAN\b",
        r"jailbreak",
        r"developer\s+mode",
        r"unrestricted\s+mode",
        r"god\s+mode",
        r"you\s+are\s+now\s+(free|unrestricted|unfiltered)",
        r"no\s+(ethical|moral|content)\s+(constraints?|restrictions?|limits?)",
        r"bypass\s+(your\s+)?(filter|safety|guideline|restriction)",
        r"do\s+not\s+(follow|apply|use)\s+(your\s+)?(guideline|restriction|rule|training)",
        r"evil\s+(mode|version|twin)",
        r"without\s+(any\s+)?restriction",
        r"do\s+anything\s+now",
        r"act\s+as\s+(a\s+)?(unrestricted|unfiltered|evil|different)\s+(ai|assistant|model|bot)?",
    ]]

    SCRIPT_INJECTION = [re.compile(p, re.IGNORECASE) for p in [
        r"<script[\s>]",
        r"</script\s*>",
        r"javascript\s*:",
        r"on(load|click|error|mouseover|focus)\s*=",
        r"<\s*iframe",
        r"<\s*img\s+src\s*=",
        r"document\.(cookie|write|location)",
        r"eval\s*\(",
        r"SELECT\s+.+FROM\s+.+WHERE",
        r"(DROP|DELETE|INSERT|UPDATE)\s+(TABLE|FROM|INTO)",
        r"UNION\s+SELECT",
        r";\s*(DROP|DELETE|INSERT|SELECT)",
    ]]

    TEMPLATE_INJECTION = [re.compile(p, re.IGNORECASE) for p in [
        r"\{\{.*?\}\}",
        r"\$\{.*?\}",
        r"<%.*?%>",
        r"\{%.*?%\}",
        r"\{\#.*?\#\}",
    ]]

    PATH_TRAVERSAL = [re.compile(p, re.IGNORECASE) for p in [
        r"(\.\./|\.\.\\){1,}",
        r"/etc/passwd",
        r"c:\\windows\\",
        r"file\s*:\s*//",
    ]]

    HARMFUL = [re.compile(p, re.IGNORECASE) for p in [
        r"\bhow\s+to\s+(make|build|create|synthesize|produce)\s+(bomb|weapon|poison|drug|explosive|grenade|napalm)",
        r"\bhow\s+to\s+(hack|crack|bypass|exploit|phish)\s+(a\s+)?(system|account|password|network|server)",
        r"\bhow\s+to\s+(kill|murder|harm|hurt|attack|assault)\s+(a\s+)?(person|people|human|someone|child)",
        r"(make|build|create)\s+(malware|ransomware|virus|trojan|spyware|keylogger)",
        r"(buy|sell|obtain)\s+(drugs?|cocaine|heroin|meth|fentanyl|weapons?|firearms?)",
        r"(evade|bypass)\s+(police|law\s+enforcement|security|authentication)",
    ]]

    SELF_HARM = [re.compile(p, re.IGNORECASE) for p in [
        r"(suicide|self[-\s]?harm|kill\s+myself)\s+(method|way|how|step|guide|tutorial)",
        r"how\s+to\s+(hurt|cut|poison|hang)\s+(myself|yourself)",
    ]]

    CHILD_SAFETY = [re.compile(p, re.IGNORECASE) for p in [
        r"child\s+(pornography|abuse|exploitation|grooming)",
        r"\bcsam\b",
        r"sexual\s+(content|images?)\s+of\s+(a\s+)?(minor|child|teen)",
    ]]

    EXTREMISM = [re.compile(p, re.IGNORECASE) for p in [
        r"(join|support|fund|recruit\s+for)\s+(isis|isil|al[-\s]?qaeda|terrorist)",
        r"(make|share|write)\s+(terrorist|extremist)\s+(propaganda|manifesto)",
        r"how\s+to\s+(carry\s+out|plan)\s+(a\s+)?terrorist\s+attack",
    ]]

    HATE_SPEECH = [re.compile(p, re.IGNORECASE) for p in [
        r"\b(nigger|nigga|faggot|kike|spic|chink|gook|raghead|sandnigger)\b",
        r"(all|those|these)\s+(jews?|muslims?|christians?|blacks?|whites?|asians?)\s+(are|should|must|deserve)\s+(die|be\s+killed|exterminated|removed)",
        r"(white|black|asian|jewish|muslim)\s+(race|people|community)\s+(is|are)\s+(inferior|superior|subhuman)",
    ]]

    PROFANITY = {
        "fuck", "shit", "asshole", "bastard", "bitch", "cunt", "damn",
        "dick", "piss", "prick", "pussy", "slut", "twat", "whore",
        "wanker", "cock", "arse", "bollocks", "motherfucker", "fucker",
    }

    EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
    PHONE_LK = re.compile(r"\b(?:0\d{9}|\+94\d{9}|\+94[\s\-]?\d{2}[\s\-]?\d{3}[\s\-]?\d{4})\b")
    PHONE_INTL = re.compile(r"\b(?:\+\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}\b")
    CREDIT_CARD = re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b")
    NIC_LK = re.compile(r"\b\d{9}[vVxX]\b|\b\d{12}\b")
    PASSPORT = re.compile(r"\b[A-Z]{1,2}\d{6,9}\b")
    BANK_ACCOUNT_LIKE = re.compile(r"\b(?:account|acct|bank)\s*(?:no|number|#)?\s*[:\-]?\s*\d{8,18}\b", re.IGNORECASE)
    ADDRESS_LIKE = re.compile(
        r"\b(?:no\.?|number)\s*\d+[A-Za-z]?(?:[/\-]\d+)?\s*,?\s*"
        r"(?:[A-Za-z]+\s+){1,5}(?:road|rd|street|st|lane|mawatha|avenue|ave)\b",
        re.IGNORECASE,
    )

    ENGLISH = re.compile(
        r"\b(the|is|are|was|were|what|how|why|when|where|who|which|does|do|did|"
        r"can|will|would|should|could|have|has|had|be|been|a|an|of|in|on|"
        r"at|to|for|with|by|from|that|this|it|its|and|or|but|if|not|"
        r"explain|describe|tell|list|define|summarize|find|show|give|provide|state)\b",
        re.IGNORECASE,
    )

    QUESTION_WORDS = re.compile(
        r"\b(what|how|why|when|where|who|which|whose|whom|is|are|was|were|"
        r"does|do|did|can|will|would|should|could|explain|describe|tell|"
        r"list|define|summarize|compare|find|show|give|provide|state)\b",
        re.IGNORECASE,
    )

    OUT_OF_SCOPE = [re.compile(p, re.IGNORECASE) for p in [
        r"\b(who\s+are\s+you|what\s+model\s+are\s+you|are\s+you\s+chatgpt)\b",
        r"\bwhat\s+can\s+you\s+do\b",
        r"\bwrite\s+(me\s+)?(a\s+)?(poem|story|essay|song)\b",
        r"\btranslate\s+this\b",
        r"\bsolve\s+my\s+homework\b",
    ]]

    SOCIAL_ENGINEERING = [re.compile(p, re.IGNORECASE) for p in [
        r"pretend\s+(this\s+is|to\s+be)\s+(authorized|allowed|legal)",
        r"for\s+(research|education|testing)\s+purposes\s+only",
        r"my\s+(teacher|boss|admin|supervisor)\s+said\s+it\s+is\s+okay",
        r"do\s+not\s+tell\s+(anyone|the\s+user|the\s+admin)",
    ]]

    ROLEPLAY = [re.compile(p, re.IGNORECASE) for p in [
        r"\broleplay\b",
        r"pretend\s+(you\s+are|to\s+be)",
        r"act\s+as\s+(a|an)\s+",
        r"simulate\s+(being|a|an)",
    ]]

    OPINION_PREDICTION_UNRELATED = [re.compile(p, re.IGNORECASE) for p in [
        r"\bwhat\s+is\s+your\s+opinion\b",
        r"\bwho\s+will\s+win\b",
        r"\bpredict\s+(the\s+)?future\b",
        r"\bshould\s+i\s+(buy|invest|marry|quit)\b",
    ]]


# Common Latin-lookalike Cyrillic/Greek characters used for bypass attempts.
HOMOGLYPH_MAP = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X",
    "Ι": "I", "Ο": "O", "Α": "A", "Ε": "E", "Ν": "N",
})

LEET_MAP = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i",
})


# ─────────────────────────────────────────────────────────────────────────────
# RESULT TYPES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    passed: bool
    blocked: bool = False
    filtered: bool = False
    warning: bool = False
    message: str = ""


@dataclass
class GuardrailResult:
    passed: bool
    sanitized_question: str = ""
    original_question: str = ""
    rejection_reason: str = ""
    risk_level: str = "LOW"
    blocked_checks: List[str] = field(default_factory=list)
    warned_checks: List[str] = field(default_factory=list)
    filtered_checks: List[str] = field(default_factory=list)
    passed_checks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    filter_log: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        sep = "=" * 64
        dash = "-" * 64
        lines = [
            "",
            sep,
            f"  QUESTION GUARDRAIL : {'✓ PASSED' if self.passed else '✗ BLOCKED'}  |  Risk: {self.risk_level}",
        ]
        if not self.passed:
            lines.append(f"  Reason    : {self.rejection_reason}")
        if self.filter_log:
            lines += [dash, "  FILTERS APPLIED:"]
            lines.extend(f"    → {f}" for f in self.filter_log)
        if self.warnings:
            lines += [dash, "  WARNINGS:"]
            lines.extend(f"    ⚠  {w}" for w in self.warnings)
        lines += [
            dash,
            f"  Blocked  : {', '.join(self.blocked_checks) or 'none'}",
            f"  Filtered : {', '.join(self.filtered_checks) or 'none'}",
            f"  Warned   : {', '.join(self.warned_checks) or 'none'}",
            f"  Passed   : {len(self.passed_checks)} checks",
        ]
        if self.passed:
            lines += [
                dash,
                f"  Original  : {self.original_question}",
                f"  Sanitized : {self.sanitized_question}",
            ]
        lines += [sep, ""]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _match(text: str, patterns: Iterable[re.Pattern]) -> Optional[str]:
    for pattern in patterns:
        found = pattern.search(text)
        if found:
            return found.group(0)
    return None


def _normalized_for_attack_checks(text: str) -> str:
    # Used only for detection, not for final answer text.
    folded = unicodedata.normalize("NFKC", text).translate(HOMOGLYPH_MAP)
    return folded.translate(LEET_MAP)


def _spell_rate(text: str) -> float:
    try:
        from spellchecker import SpellChecker
        words = P.WORD.findall(text.lower())
        if not words:
            return 0.0
        return len(SpellChecker().unknown(words)) / len(words)
    except Exception:
        # Optional dependency. If unavailable, do not block.
        return 0.0


def _lang(text: str) -> Tuple[str, float]:
    try:
        from langdetect import detect_langs
        top = detect_langs(text[:600])[0]
        return top.lang, round(top.prob, 2)
    except Exception:
        hits = len(P.ENGLISH.findall(text.lower()))
        words = max(len(P.WORD.findall(text)), 1)
        conf = min(hits / words, 1.0)
        return ("en" if conf > 0.08 else "unknown"), round(conf, 2)


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = {ch: text.count(ch) for ch in set(text)}
    return -sum((count / len(text)) * math.log2(count / len(text)) for count in counts.values())


def _looks_gibberish(text: str) -> bool:
    words = P.WORD.findall(text.lower())
    if len(words) < 2:
        return False

    suspicious = 0
    for word in words:
        vowels = sum(1 for c in word if c in "aeiou")
        vowel_ratio = vowels / max(len(word), 1)
        has_long_consonant_run = bool(re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", word))
        has_long_randomish = len(word) >= 10 and vowel_ratio < 0.18
        if has_long_consonant_run or has_long_randomish:
            suspicious += 1

    ratio = suspicious / len(words)
    entropy = _shannon_entropy("".join(words))
    return ratio >= Config.GIBBERISH_THRESHOLD or (len("".join(words)) > 20 and entropy > 4.2 and ratio > 0.35)


def _contains_mixed_script_homoglyphs(text: str) -> bool:
    latin = any("LATIN" in unicodedata.name(ch, "") for ch in text if ch.isalpha())
    suspicious_script = any(
        ("CYRILLIC" in unicodedata.name(ch, "") or "GREEK" in unicodedata.name(ch, ""))
        for ch in text if ch.isalpha()
    )
    return latin and suspicious_script


# ─────────────────────────────────────────────────────────────────────────────
# SANITIZER
# ─────────────────────────────────────────────────────────────────────────────

def sanitize(text: str) -> Tuple[str, List[str]]:
    """
    Cleans the question text and returns (sanitized_text, change_log).
    Never changes word meaning and never auto-corrects spelling.
    """
    log: List[str] = []
    text = "" if text is None else str(text)

    normalized = unicodedata.normalize("NFKC", text)
    if normalized != text:
        log.append("Normalized Unicode characters")
    text = normalized

    without_controls = P.UNICODE_ABUSE.sub("", text)
    if without_controls != text:
        log.append("Removed invisible Unicode control characters")
    text = without_controls

    cleaned = re.sub(r"[^\x20-\x7E\t\n]", "", text)
    if cleaned != text:
        log.append("Removed non-printable / non-ASCII characters")
    text = cleaned

    collapsed = re.sub(r"\s+", " ", text).strip()
    if collapsed != text:
        log.append("Collapsed extra whitespace")
    text = collapsed

    fixed = re.sub(r"[?]{2,}", "?", text)
    fixed = re.sub(r"[!]{2,}", "!", fixed)
    fixed = re.sub(r"[.]{4,}", "...", fixed)
    if fixed != text:
        log.append("Fixed repeated punctuation")
    text = fixed

    no_noise = re.sub(r"[%#@$^&*_=+~`|\\]{3,}", "", text)
    if no_noise != text:
        log.append("Removed repeated noisy symbols")
    text = no_noise

    stripped = text.strip(".,;:\"'`")
    if stripped != text:
        log.append("Stripped leading/trailing punctuation")
    text = stripped.strip()

    return text, log


# ─────────────────────────────────────────────────────────────────────────────
# QUESTION CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def _q_empty(original: str) -> CheckResult:
    if not original or not original.strip():
        return CheckResult("empty", False, True, message="Question is empty. Please type a question.")
    return CheckResult("empty", True)


def _q_nonprintable(original: str) -> CheckResult:
    if P.NULL_BYTE.search(original):
        return CheckResult("null_byte_injection", False, True, message="Question contains a null byte injection pattern.")
    bad = sum(1 for c in original if not c.isprintable() and c not in "\t\n")
    ratio = bad / max(len(original), 1)
    if ratio > Config.GARBAGE_CHAR_THRESHOLD:
        return CheckResult(
            "non_printable",
            False,
            True,
            message=f"Question contains {ratio:.0%} non-printable characters. Input may be corrupted.",
        )
    return CheckResult("non_printable", True)


def _q_unicode_abuse(original: str) -> CheckResult:
    if P.UNICODE_ABUSE.search(original):
        return CheckResult(
            "unicode_abuse",
            True,
            warning=True,
            message="Question contained invisible Unicode/control characters. They were removed during sanitization.",
        )
    return CheckResult("unicode", True)


def _q_homoglyph(original: str) -> CheckResult:
    if _contains_mixed_script_homoglyphs(original):
        return CheckResult(
            "homoglyph_attack",
            False,
            True,
            message="Question mixes Latin letters with lookalike Unicode characters, which may be a bypass attempt.",
        )
    return CheckResult("homoglyph", True)




def _q_leetspeak_obfuscation(original: str) -> CheckResult:
    normalized = _normalized_for_attack_checks(original)
    # If normalization changes many characters and creates suspicious safety words,
    # treat it as an obfuscation attempt rather than a spelling issue.
    changed = sum(1 for a, b in zip(original, normalized) if a != b)
    changed_ratio = changed / max(len(original), 1)
    suspicious_terms = re.search(
        r"\b(bomb|weapon|hack|malware|virus|kill|suicide|drugs?|terrorist|bypass)\b",
        normalized,
        re.IGNORECASE,
    )
    if changed_ratio > 0.08 and suspicious_terms:
        return CheckResult(
            "leetspeak_obfuscation",
            False,
            True,
            message="Question appears to obfuscate unsafe content using leetspeak/lookalike characters.",
        )
    if changed_ratio > 0.12:
        return CheckResult(
            "leetspeak_obfuscation",
            True,
            warning=True,
            message="Question contains leetspeak or obfuscated characters. This may reduce retrieval quality.",
        )
    return CheckResult("leetspeak", True)


def _q_length(text: str) -> CheckResult:
    chars = len(text.strip())
    words = len(P.WORD.findall(text))
    if chars < Config.QUESTION_MIN_CHARS:
        return CheckResult("too_short", False, True, message=f"Question is too short ({chars} chars). Minimum is {Config.QUESTION_MIN_CHARS}.")
    if words < Config.QUESTION_MIN_WORDS:
        return CheckResult("too_short", False, True, message=f"Question needs at least {Config.QUESTION_MIN_WORDS} words.")
    if chars > Config.QUESTION_MAX_CHARS:
        return CheckResult("too_long", False, True, message=f"Question is too long ({chars} chars, max {Config.QUESTION_MAX_CHARS}).")
    if words > Config.QUESTION_MAX_WORDS:
        return CheckResult("too_long", False, True, message=f"Question has too many words ({words}, max {Config.QUESTION_MAX_WORDS}).")
    return CheckResult("length", True)


def _q_repeated_chars(text: str) -> CheckResult:
    """Detect obvious character spam using consecutive runs, not normal letter frequency."""
    compact = re.sub(r"\s+", "", text.lower())
    if not compact:
        return CheckResult("repeated_chars", True)

    # Blocks examples like aaaaaaaa, ???????, ssssssoooooolllllar.
    longest_run = max((len(m.group(0)) for m in re.finditer(r"(.)\1+", compact)), default=1)
    run_ratio = longest_run / max(len(compact), 1)

    # Also catch tiny alphabets repeated many times: ababababab, xyzxyzxyzxyz.
    repeated_unit = bool(re.fullmatch(r"(.{1,4})\1{3,}", compact))

    if longest_run >= 7 or run_ratio >= 0.55 or repeated_unit:
        return CheckResult(
            "repeated_chars",
            False,
            True,
            message="Input appears to be character/repetition spam. Please type a normal question.",
        )
    return CheckResult("repeated_chars", True)


def _q_real_words(text: str) -> CheckResult:
    if not P.WORD.findall(text):
        return CheckResult("no_real_words", False, True, message="Input contains no real words. Please type a meaningful question.")
    return CheckResult("no_real_words", True)


def _q_gibberish(text: str) -> CheckResult:
    """Block strong random text, but only warn for weak/possibly misspelled questions."""
    words = P.WORD.findall(text.lower())
    joined = "".join(words)

    if _looks_gibberish(text):
        return CheckResult(
            "gibberish",
            False,
            True,
            message="Question appears to be gibberish or random character sequences.",
        )

    if len(words) >= 2:
        odd_words = 0
        for word in words:
            vowel_ratio = sum(1 for c in word if c in "aeiou") / max(len(word), 1)
            no_vowel_long = len(word) >= 5 and not re.search(r"[aeiou]", word)
            too_many_rare = len(re.findall(r"[qxzj]", word)) >= 3
            if no_vowel_long or vowel_ratio < 0.12 or too_many_rare:
                odd_words += 1
        if odd_words / len(words) >= 0.50 and len(joined) >= 12:
            return CheckResult(
                "gibberish",
                True,
                warning=True,
                message="Question may contain gibberish or heavy spelling errors. Retrieval quality may be poor.",
            )

    return CheckResult("gibberish", True)


def _q_caps(text: str) -> CheckResult:
    letters = [c for c in text if c.isalpha()]
    if len(letters) > 10:
        ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if ratio > Config.CAPS_THRESHOLD:
            return CheckResult("excessive_caps", True, warning=True, message=f"Question is {ratio:.0%} uppercase. ALL CAPS may affect answer quality.")
    return CheckResult("caps", True)


def _q_punct(original: str) -> CheckResult:
    # Check the original text, because the sanitizer may already reduce ?????/!!!!!.
    if P.EXCESS_PUNCT.search(original):
        return CheckResult(
            "excessive_punctuation",
            True,
            warning=True,
            message="Question contained excessive punctuation. The sanitized version reduced it.",
        )
    return CheckResult("punctuation", True)


def _q_repetition(text: str) -> CheckResult:
    """Detect repeated words and repeated short phrases more reliably than one regex."""
    words = [w.lower() for w in P.WORD.findall(text)]
    if len(words) < 4:
        return CheckResult("repetition", True)

    # Same word repeated 3+ times: solar solar solar.
    for i in range(len(words) - 2):
        if words[i] == words[i + 1] == words[i + 2]:
            return CheckResult("repeated_phrase", True, warning=True, message="Question contains repeated words/phrases. This may affect retrieval quality.")

    # Same 2-4 word phrase repeated.
    for n in range(2, 5):
        grams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        if any(grams.count(g) >= 3 for g in set(grams)):
            return CheckResult("repeated_phrase", True, warning=True, message="Question contains repeated words/phrases. This may affect retrieval quality.")

    return CheckResult("repetition", True)


def _q_security(text: str, raw: str) -> List[CheckResult]:
    attack_text = _normalized_for_attack_checks(text)
    raw_attack_text = _normalized_for_attack_checks(raw)

    checks: List[CheckResult] = []

    hit = _match(attack_text, P.SCRIPT_INJECTION)
    checks.append(CheckResult(
        "script_injection", not bool(hit), bool(hit),
        message=f"Question contains script/code injection: \"{hit[:60]}\"." if hit else "",
    ))

    hit = _match(attack_text, P.TEMPLATE_INJECTION)
    checks.append(CheckResult(
        "template_injection", not bool(hit), bool(hit),
        message=f"Question contains template injection syntax: \"{hit[:60]}\"." if hit else "",
    ))

    hit = _match(raw_attack_text, P.PATH_TRAVERSAL)
    checks.append(CheckResult(
        "path_traversal", not bool(hit), bool(hit),
        message=f"Question contains path traversal or local file access syntax: \"{hit[:60]}\"." if hit else "",
    ))

    hit = _match(attack_text, P.PROMPT_INJECTION)
    checks.append(CheckResult(
        "prompt_injection", not bool(hit), bool(hit),
        message=f"Question contains a prompt injection attempt: \"{hit[:60]}\"." if hit else "",
    ))

    hit = _match(attack_text, P.JAILBREAK)
    checks.append(CheckResult(
        "jailbreak", not bool(hit), bool(hit),
        message=f"Question contains a jailbreak attempt: \"{hit[:60]}\"." if hit else "",
    ))

    return checks


def _q_safety(text: str) -> List[CheckResult]:
    checks: List[CheckResult] = []

    for name, patterns, message in [
        ("harmful_intent", P.HARMFUL, "Question requests harmful information and cannot be processed."),
        ("self_harm", P.SELF_HARM, "Question appears to request self-harm instructions and cannot be processed."),
        ("child_safety", P.CHILD_SAFETY, "Question contains child-safety violating content and cannot be processed."),
        ("extremism", P.EXTREMISM, "Question contains extremist or terrorist instruction/support content and cannot be processed."),
        ("hate_speech", P.HATE_SPEECH, "Question contains hate speech and cannot be processed."),
    ]:
        hit = _match(text, patterns)
        checks.append(CheckResult(name, not bool(hit), bool(hit), message=message if hit else ""))

    words = {w.lower() for w in P.WORD.findall(text)}
    hits = words & P.PROFANITY
    checks.append(CheckResult(
        "profanity",
        not bool(hits),
        bool(hits),
        message="Question contains inappropriate language. Please rephrase." if hits else "",
    ))

    return checks


def _q_privacy(text: str) -> CheckResult:
    found = []
    if P.EMAIL.search(text): found.append("email address")
    if P.PHONE_LK.search(text) or P.PHONE_INTL.search(text): found.append("phone number")
    if P.NIC_LK.search(text): found.append("NIC number")
    if P.CREDIT_CARD.search(text): found.append("credit card number")
    if P.PASSPORT.search(text): found.append("passport number")
    if P.BANK_ACCOUNT_LIKE.search(text): found.append("bank account number")
    if P.ADDRESS_LIKE.search(text): found.append("address/location detail")

    if found:
        return CheckResult(
            "pii_detected",
            True,
            warning=True,
            message=f"Question contains personal information ({', '.join(found)}). Avoid including unnecessary personal data.",
        )
    return CheckResult("privacy", True)


def _q_language(text: str) -> CheckResult:
    lang, conf = _lang(text)
    if lang != "en" and conf > 0.72:
        return CheckResult("non_english", False, True, message=f"Question appears to be non-English (detected: {lang.upper()}, confidence: {conf:.0%}).")
    return CheckResult("language", True)


def _q_spelling(text: str) -> CheckResult:
    rate = _spell_rate(text)
    if rate > Config.SPELL_ERROR_THRESHOLD:
        return CheckResult(
            "spelling_quality",
            True,
            warning=True,
            message=f"{rate:.0%} of words may be misspelled. No auto-correction was applied, but retrieval quality may be reduced.",
        )
    return CheckResult("spelling", True)


def _q_grammar(text: str) -> CheckResult:
    """Warn when it does not look like a question; do not block keyword-style RAG queries."""
    has_q_word = bool(P.QUESTION_WORDS.search(text))
    has_q_mark = text.strip().endswith("?")
    words = P.WORD.findall(text)

    if not has_q_word and not has_q_mark:
        if len(words) <= 3:
            return CheckResult(
                "question_structure",
                True,
                warning=True,
                message="Input looks like a keyword search, not a full question. It can still work, but a clearer question improves retrieval.",
            )
        return CheckResult(
            "question_structure",
            True,
            warning=True,
            message="Input does not clearly look like a question. The system works best with a question word or '?'.",
        )
    return CheckResult("question_structure", True)


def _q_semantic_intent(text: str) -> List[CheckResult]:
    checks: List[CheckResult] = []

    hit = _match(text, P.OUT_OF_SCOPE)
    checks.append(CheckResult(
        "out_of_scope",
        not bool(hit),
        bool(hit),
        message="Question appears to be out of scope for a document QA system. Ask something answerable from the uploaded document." if hit else "",
    ))

    hit = _match(text, P.SOCIAL_ENGINEERING)
    checks.append(CheckResult(
        "social_engineering",
        not bool(hit),
        bool(hit),
        message=f"Question contains social-engineering style language: \"{hit[:60]}\"." if hit else "",
    ))

    hit = _match(text, P.ROLEPLAY)
    checks.append(CheckResult(
        "roleplay_or_pretend",
        not bool(hit),
        bool(hit),
        message="Roleplay/pretend instructions are not allowed in document QA questions." if hit else "",
    ))

    hit = _match(text, P.OPINION_PREDICTION_UNRELATED)
    checks.append(CheckResult(
        "unrelated_opinion_prediction",
        not bool(hit),
        bool(hit),
        message="Question asks for an opinion/prediction unrelated to the document." if hit else "",
    ))

    return checks


# ─────────────────────────────────────────────────────────────────────────────
# RISK SCORER
# ─────────────────────────────────────────────────────────────────────────────

_HIGH_RISK = {
    "script_injection", "template_injection", "path_traversal", "null_byte_injection",
    "prompt_injection", "jailbreak", "harmful_intent", "self_harm", "child_safety",
    "extremism", "hate_speech", "homoglyph_attack", "leetspeak_obfuscation", "social_engineering",
}
_MEDIUM_RISK = {
    "profanity", "pii_detected", "non_english", "too_long", "out_of_scope",
    "roleplay_or_pretend", "unrelated_opinion_prediction", "gibberish",
}


def _risk(blocked: List[str], warned: List[str]) -> str:
    blocked_set = set(blocked)
    warned_set = set(warned)
    if blocked_set & _HIGH_RISK:
        return "HIGH"
    if blocked_set & _MEDIUM_RISK:
        return "MEDIUM"
    if blocked:
        return "LOW"
    if warned_set & _MEDIUM_RISK or len(warned) >= Config.MAX_WARNINGS_BEFORE_MEDIUM_RISK:
        return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GUARDRAIL CLASS
# ─────────────────────────────────────────────────────────────────────────────

class InputGuardrail:
    """
    Question-only input guardrail for Document QA / RAG.

    Parameters
    ----------
    strict_mode:
        If True, warnings become blocks.
    allow_non_english:
        If True, non-English input is allowed instead of blocked.
    check_spelling:
        If True, runs optional spelling-quality warning.
    """

    def __init__(
        self,
        strict_mode: bool = False,
        allow_non_english: bool = False,
        check_spelling: bool = True,
    ):
        self.strict = strict_mode
        self.non_eng_ok = allow_non_english
        self.spell = check_spelling

    def _build(
        self,
        checks: List[CheckResult],
        sanitized: str,
        original: str,
        filter_log: List[str],
    ) -> GuardrailResult:
        # Convert warnings to blocks in strict mode.
        if self.strict:
            for c in checks:
                if c.warning and c.passed:
                    c.passed = False
                    c.blocked = True

        blocked = [c.name for c in checks if not c.passed]
        warned = [c.name for c in checks if c.warning and c.passed]
        filtered = [c.name for c in checks if c.filtered]
        passed = [c.name for c in checks if c.passed and not c.warning]
        block_messages = [c.message for c in checks if not c.passed and c.message]
        warning_messages = [c.message for c in checks if c.warning and c.passed and c.message]

        return GuardrailResult(
            passed=len(blocked) == 0,
            sanitized_question=sanitized if len(blocked) == 0 else "",
            original_question=original,
            rejection_reason=block_messages[0] if block_messages else "",
            risk_level=_risk(blocked, warned),
            blocked_checks=blocked,
            warned_checks=warned,
            filtered_checks=filtered,
            passed_checks=passed,
            warnings=warning_messages,
            filter_log=filter_log,
        )

    def check_question(self, question: str) -> GuardrailResult:
        original = "" if question is None else str(question)
        sanitized, filter_log = sanitize(original)

        checks: List[CheckResult] = [
            _q_empty(original),
            _q_nonprintable(original),
            _q_unicode_abuse(original),
            _q_homoglyph(original),
            _q_leetspeak_obfuscation(original),
            _q_length(sanitized),
            _q_repeated_chars(sanitized),
            _q_real_words(sanitized),
            _q_gibberish(sanitized),
            _q_caps(sanitized),
            _q_punct(original),
            _q_repetition(sanitized),
        ]

        checks.extend(_q_security(sanitized, original))
        checks.extend(_q_safety(_normalized_for_attack_checks(original)))
        checks.append(_q_privacy(sanitized))

        if self.non_eng_ok:
            checks.append(CheckResult("language", True))
        else:
            checks.append(_q_language(sanitized))

        checks.append(_q_spelling(sanitized) if self.spell else CheckResult("spelling", True))
        checks.append(_q_grammar(sanitized))
        checks.extend(_q_semantic_intent(sanitized))

        return self._build(checks, sanitized, original, filter_log)

    # Backward-compatible alias for old integrations that called guardrail(question)
    def __call__(self, question: str) -> GuardrailResult:
        return self.check_question(question)


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE
# ─────────────────────────────────────────────────────────────────────────────

def check_question(question: str, **kwargs) -> GuardrailResult:
    return InputGuardrail(**kwargs).check_question(question)
