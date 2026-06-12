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

# Expand this later if your organization has real role permissions.
DUAL_USE_ALLOWED_ROLES = {"security_admin", "internal_red_team", "security_trainee"}


def apply_policy(
    safety: SafetyResult,
    relevance: RelevanceResult,
    kb_profile: KBProfile,
    strict_mode: bool,
    user_role: str,
) -> Tuple[str, bool, str, str, List[str]]:
    """Return decision, passed, risk, reason, events."""
    events: List[str] = []
    if user_role not in ALLOWED_ROLES:
        user_role = "public_user"
        events.append("unknown_role_defaulted_to_public_user")

    if safety.decision == "block":
        return (
            "BLOCKED",
            False,
            safety.risk_level,
            f"Unsafe input detected: {safety.label}. " + "; ".join(safety.reasons[:3]),
            events,
        )

    if safety.decision == "warn":
        # Some warnings may still pass if not harmful.
        return (
            "PASSED_WITH_WARNING",
            True,
            safety.risk_level,
            f"Input appears safe enough to continue, but warning category detected: {safety.label}.",
            events,
        )

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

    # For high-risk documents, groundedness is recorded. Safe questions can pass, but the UI shows score.
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
