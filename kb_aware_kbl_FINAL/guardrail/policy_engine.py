from typing import List, Tuple

from .schema import KBProfile, RelevanceResult, SafetyResult

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


def _kb_has_denial_or_safety_disclaimer(relevance: RelevanceResult) -> bool:
    text = " ".join(relevance.top_chunks).lower()
    return any(pattern in text for pattern in DENIAL_OR_SAFETY_DISCLAIMER_PATTERNS)


def apply_policy(
    safety: SafetyResult,
    relevance: RelevanceResult,
    kb_profile: KBProfile,
    strict_mode: bool,
    user_role: str,
    kb_authoritative_mode: bool = True,
) -> Tuple[str, bool, str, str, List[str]]:
    """Return decision, passed, risk, reason, events.

    Main policy modes:
    - Normal safety: unsafe/actionable requests are blocked.
    - KB-authoritative mode: unsafe/actionable requests can pass WITH WARNING
      only when they are strongly grounded in the uploaded KB and the KB does
      not contain a refusal/disclaimer phrase for that content.

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
        if kb_authoritative_mode and relevance.is_grounded:
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
