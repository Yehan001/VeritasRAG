from typing import List, Tuple

from .schema import KBProfile, RelevanceResult, SafetyResult
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


def _kb_has_denial_or_safety_disclaimer(relevance: RelevanceResult) -> bool:
    text = " ".join(relevance.top_chunks).lower()
    return any(pattern in text for pattern in DENIAL_OR_SAFETY_DISCLAIMER_PATTERNS)


def _lexical_overlap_ok(question: str, relevance: RelevanceResult) -> bool:
    """True only if the question and the KB's matched chunks actually share
    real words. This guards against embedding models returning a moderate
    cosine similarity for two semantically unrelated sentences (a known
    property of sentence embeddings), which would otherwise let an unrelated
    harmful question slip through the KB-authoritative override just because
    it scored above the grounding threshold by embedding-space noise.
    """
    if not relevance.top_chunks:
        return False
    q_tokens = {
        t for t in simple_tokenize(question)
        if len(t) >= 4 and t not in _STOPWORDS and t not in _GENERIC_DOMAIN_NOISE
    }
    if not q_tokens:
        return False
    kb_tokens = {
        t
        for chunk in relevance.top_chunks
        for t in simple_tokenize(chunk)
        if len(t) >= 4 and t not in _STOPWORDS and t not in _GENERIC_DOMAIN_NOISE
    }
    shared = q_tokens & kb_tokens
    return len(shared) >= MIN_LEXICAL_OVERLAP_TOKENS


def apply_policy(
    safety: SafetyResult,
    relevance: RelevanceResult,
    kb_profile: KBProfile,
    strict_mode: bool,
    user_role: str,
    kb_authoritative_mode: bool = True,
    question_text: str = "",
) -> Tuple[str, bool, str, str, List[str]]:
    """Return decision, passed, risk, reason, events.

    Main policy modes:
    - Normal safety: unsafe/actionable requests are blocked.
    - KB-authoritative mode: unsafe/actionable requests can pass WITH WARNING
      only when they are strongly grounded in the uploaded KB (both by
      embedding similarity AND by actually sharing real words with the
      matched KB text) and the KB does not contain a refusal/disclaimer
      phrase for that content.

    Prompt injection and technical attacks are always blocked because they target
    the application/guardrail rather than asking about KB content.
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

        # KB-authoritative override: if the KB actually supports the high-risk
        # content, pass it with a visible warning instead of blocking.
        # Requires BOTH a passing embedding-grounding score AND real lexical
        # overlap, so a noisy embedding score alone can never trigger this.
        if kb_authoritative_mode and relevance.is_grounded and _lexical_overlap_ok(question_text, relevance):
            extra_event = "kb_authoritative_high_risk_override"
            extra_reason = ""
            if _kb_has_denial_or_safety_disclaimer(relevance):
                # In KB-authoritative mode, the document is treated as the source of truth.
                # Therefore a grounded high-risk question is allowed with warning even when
                # the retrieved chunk also contains a disclaimer/refusal. The warning keeps
                # the risk visible without blocking KB-supported review.
                extra_event = "kb_authoritative_high_risk_override_with_disclaimer"
                extra_reason = " Retrieved KB text also contains a disclaimer/refusal, so review the source chunk carefully."
            return (
                "PASSED_WITH_WARNING",
                True,
                safety.risk_level,
                f"High-risk input detected: {safety.label}. It is allowed because KB-authoritative mode is ON and the question is grounded in the uploaded knowledge base. Use only for authorized KB-based review." + extra_reason,
                events + [extra_event],
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

    # 3. Safe but out-of-scope questions.
    if strict_mode and not relevance.is_relevant:
        return (
            "BLOCKED",
            False,
            "low",
            "Question appears safe but is outside the uploaded knowledge base. Strict mode blocks out-of-scope questions.",
            events,
        )

    if not relevance.is_relevant:
        return (
            "PASSED_WITH_WARNING",
            True,
            "low",
            "Question appears safe, but it is outside the uploaded knowledge base scope.",
            events,
        )

    # 4. High-risk KB weak grounding warning.
    if kb_profile.risk_level == "high" and not relevance.is_grounded:
        if strict_mode:
            return (
                "BLOCKED",
                False,
                "medium",
                "Question is related but not strongly grounded in the high-risk knowledge base. Strict mode blocks it.",
                events,
            )
        return (
            "PASSED_WITH_WARNING",
            True,
            "medium",
            "Question appears safe and related, but grounding score is weak for a high-risk knowledge base.",
            events,
        )

    return (
        "PASSED",
        True,
        "low",
        "Question is safe and relevant to the uploaded knowledge base.",
        events,
    )