from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class StageResult:
    stage: str
    action: str  # ALLOW, WARN, BLOCK, SKIPPED, UNAVAILABLE
    label: str = "safe"
    risk: str = "low"
    reason: str = ""
    backend: str = ""
    score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PIIResult:
    has_pii: bool
    masked_text: str
    entities: List[Dict[str, Any]] = field(default_factory=list)
    backend: str = "regex"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GuardrailResult:
    decision: str  # PASSED, PASSED_WITH_WARNING, BLOCKED
    passed: bool
    risk: str
    safety_label: str
    reason: str
    original_input: str
    sanitized_input: str
    triggered_stage: str
    safety_backend: str
    checked_stages: List[StageResult]
    skipped_stages: List[str]
    pii: PIIResult
    audit_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["checked_stages"] = [s.to_dict() for s in self.checked_stages]
        data["pii"] = self.pii.to_dict()
        return data
