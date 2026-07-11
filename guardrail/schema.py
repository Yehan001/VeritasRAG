from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ContextProfile:
    domain: str
    confidence: float
    risk_level: str
    matched_terms: List[str] = field(default_factory=list)


@dataclass
class SupportResult:
    is_relevant: bool
    is_grounded: bool
    score: float
    top_chunks: List[str] = field(default_factory=list)
    method: str = "sentence_transformers"


KBProfile = ContextProfile
RelevanceResult = SupportResult


@dataclass
class SafetyResult:
    label: str
    decision: str
    risk_level: str
    confidence: float
    backend: str
    reasons: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PIIResult:
    # masked_text always holds original text unchanged — detection only, no masking
    masked_text: str
    found: bool
    entities: List[str] = field(default_factory=list)


@dataclass
class APIResponse:
    """Clean response model returned to calling apps via the API.
    Excludes internal debug fields, raw input text, and source content details.
    """
    decision: str
    passed: bool
    risk_level: str
    reason: str
    safety_label: str
    safety_backend: str
    safety_confidence: float
    relevance_score: float
    relevance_is_relevant: bool
    relevance_is_grounded: bool
    kb_domain: str
    kb_risk_level: str
    pii_found: bool
    pii_entities: List[str] = field(default_factory=list)
    user_role: str = "public_user"
    tenant_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FilterResult:
    decision: str
    passed: bool
    risk_level: str
    reason: str
    original_input: str
    sanitized_input: str
    context_profile: ContextProfile
    support: SupportResult
    safety: SafetyResult
    pii: PIIResult
    user_role: str
    events: List[Dict[str, Any]] = field(default_factory=list)
    tenant_id: Optional[str] = None

    @property
    def kb_profile(self) -> ContextProfile:
        return self.context_profile

    @property
    def relevance(self) -> SupportResult:
        return self.support

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_api_response(self) -> APIResponse:
        """Convert full internal result to clean API response for calling apps."""
        return APIResponse(
            decision=self.decision,
            passed=self.passed,
            risk_level=self.risk_level,
            reason=self.reason,
            safety_label=self.safety.label,
            safety_backend=self.safety.backend,
            safety_confidence=self.safety.confidence,
            relevance_score=self.support.score,
            relevance_is_relevant=self.support.is_relevant,
            relevance_is_grounded=self.support.is_grounded,
            kb_domain=self.context_profile.domain,
            kb_risk_level=self.context_profile.risk_level,
            pii_found=self.pii.found,
            pii_entities=self.pii.entities,
            user_role=self.user_role,
            tenant_id=self.tenant_id,
        )