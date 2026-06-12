from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class KBProfile:
    domain: str
    confidence: float
    risk_level: str
    matched_terms: List[str] = field(default_factory=list)


@dataclass
class RelevanceResult:
    is_relevant: bool
    is_grounded: bool
    score: float
    top_chunks: List[str] = field(default_factory=list)
    method: str = "tfidf"


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
    masked_text: str
    found: bool
    entities: List[str] = field(default_factory=list)


@dataclass
class FilterResult:
    decision: str
    passed: bool
    risk_level: str
    reason: str
    original_input: str
    sanitized_input: str
    pii_masked_input: str
    kb_profile: KBProfile
    relevance: RelevanceResult
    safety: SafetyResult
    user_role: str
    events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
