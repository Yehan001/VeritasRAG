from typing import List, Tuple

from .schema import ContextProfile, SafetyResult, SupportResult
from .text_utils import simple_tokenize

ALLOWED_ROLES = {
    "public_user",
    "student",
    "employee",
    "security_trainee",
    "security_admin",
    "internal_red_team",
}

# These labels attack the guardrail/application itself, so they are blocked even
# in KB-authoritative mode.
ALWAYS_BLOCK_LABELS = {
    "prompt_injection",
    "jailbreak",
    "script_or_command_injection",
}

# Phrases showing the KB is refusing or warning against providing the requested
# harmful details. This prevents a sentence like "this document does not provide
# attack payloads" from being treated as permission to answer attack-payload questions.
DENIAL_OR_SAFETY_DISCLAIMER_PATTERNS = [
    "does not provide",
    "do not provide",
    "not provide",
    "doesn't provide",
    "do not include",
    "not include",
    "should not",
    "must not",
    "not allowed",
    "unauthorized",
    "without permission",
    "for defensive education only",
    "defensive education only",
    "instead of clicking",
    "report suspicious",
    "do not attempt",
]

# Common short/stop words excluded from lexical overlap so matches like "the",
# "how", "what" don't count as evidence the KB actually discusses the question.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "for", "in",
    "on", "at", "and", "or", "how", "what", "why", "who", "with", "this",
    "that", "it", "do", "does", "did", "can", "could", "would", "should",
    "will", "you", "your", "i", "me", "my", "someone", "people", "person",
    "using", "used", "use", "described", "document", "documents",
}

# Generic infrastructure/domain nouns that show up throughout almost any
# cybersecurity (or similarly broad technical) KB regardless of what specific
# topic is being asked about. Overlapping on words like "system" or "network"
# alone is not meaningful evidence the KB actually discusses the question --
# only overlap on more specific, topic-identifying words counts.
_GENERIC_DOMAIN_NOISE = {
    "system", "systems", "network", "networks", "security", "data",
    "firewall", "firewalls", "access", "information", "technology",
    "organization", "organizations", "employee", "employees",
}

# Minimum number of shared, non-stopword, non-generic tokens (length >= 4)
# required between the question and the KB's top-matched chunks before a
# grounding score is trusted for the KB-authoritative override below.
MIN_LEXICAL_OVERLAP_TOKENS = 3


def _has_context_disclaimer(support: SupportResult) -> bool:
    text = " ".join(support.top_chunks).lower()
    return any(pattern in text for pattern in DENIAL_OR_SAFETY_DISCLAIMER_PATTERNS)


def _lexical_overlap_ok(question: str, support: SupportResult) -> bool:
    """True only if the question and the matched context chunks actually share
    real words. This guards against embedding models returning a moderate
    cosine similarity for two semantically unrelated sentences (a known
    property of sentence embeddings), which would otherwise let an unrelated
    harmful question slip through a context-based override just because
    it scored above the grounding threshold by embedding-space noise.
    """
    if not support.top_chunks:
        return False
    q_tokens = {
        t for t in simple_tokenize(question)
        if len(t) >= 4 and t not in _STOPWORDS and t not in _GENERIC_DOMAIN_NOISE
    }
    if not q_tokens:
        return False
    context_tokens = {
        t
        for chunk in support.top_chunks
        for t in simple_tokenize(chunk)
        if len(t) >= 4 and t not in _STOPWORDS and t not in _GENERIC_DOMAIN_NOISE
    }
    shared = q_tokens & context_tokens
    return len(shared) >= MIN_LEXICAL_OVERLAP_TOKENS


def apply_policy(
    safety: SafetyResult,
    support: SupportResult,
    context_profile: ContextProfile,
    strict_mode: bool,
    user_role: str,
    context_override_enabled: bool = False,
    question_text: str = "",
) -> Tuple[str, bool, str, str, List[str]]:
    """Return decision, passed, risk, reason, events.

    Main policy modes:
    - Normal safety: unsafe/actionable requests are blocked.
    - Context-supported mode: unsafe/actionable requests can pass WITH WARNING
      only when they are strongly supported by the provided context (both by
      embedding similarity AND by actually sharing real words with the
      matched context text) and the content does not contain a refusal/disclaimer
      phrase for that content.

    Prompt injection and technical attacks are always blocked because they target
    the application/guardrail rather than asking about the provided context.
    """
    events: List[str] = []
    if user_role not in ALLOWED_ROLES:
        user_role = "public_user"
        events.append("unknown_role_defaulted_to_public_user")

    # 1. Unsafe categories.
    if safety.decision == "block":
        if safety.label in ALWAYS_BLOCK_LABELS:
            return (
                "BLOCKED",
                False,
                safety.risk_level,
                f"System/security attack blocked: {safety.label}. " + "; ".join(safety.reasons[:3]),
                events + ["always_block_category"],
            )

        return (
            "BLOCKED",
            False,
            safety.risk_level,
            f"Unsafe input detected: {safety.label}. " + "; ".join(safety.reasons[:3]),
            events,
        )

    # 2. Warning categories, such as PII.
    if safety.decision == "warn":
        return (
            "PASSED_WITH_WARNING",
            True,
            safety.risk_level,
            f"Input can continue, but warning category detected: {safety.label}. " + "; ".join(safety.reasons[:2]),
            events,
        )

    # 3. Safe questions pass without any KB relevance requirement.
    return (
        "PASSED",
        True,
        "low",
        "Question is safe.",
        events,
    )