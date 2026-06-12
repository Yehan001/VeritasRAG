from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .audit_logger import append_audit_log
from .hf_safety import HFModelConfig, HFSafetyEngine
from .kb_profiler import profile_kb
from .kb_relevance import KBRelevanceChecker
from .pii_masker import mask_pii
from .policy_engine import apply_policy
from .schema import FilterResult, KBProfile, RelevanceResult, SafetyResult
from .text_utils import normalize_text, sanitize_for_display


@dataclass
class GuardrailSettings:
    strict_mode: bool = False
    enable_hf_models: bool = True
    use_presidio: bool = True
    kb_backend: str = "tfidf"  # tfidf or sentence_transformers
    relevance_threshold: float = 0.08
    grounding_threshold: float = 0.12
    user_role: str = "public_user"
    prompt_injection_model: str = "protectai/deberta-v3-base-prompt-injection-v2"
    moderation_model: str = "oxyapi/albert-moderation-001"
    zero_shot_model: str = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"
    enable_audit_log: bool = True
    audit_log_path: str = "guardrail_audit_log.jsonl"


class KBAwareInputFilter:
    def __init__(self, kb_text: str, settings: Optional[GuardrailSettings] = None) -> None:
        self.settings = settings or GuardrailSettings()
        self.kb_text = kb_text or ""
        self.kb_profile = profile_kb(self.kb_text)
        self.relevance_checker = KBRelevanceChecker(
            backend=self.settings.kb_backend,
            relevance_threshold=self.settings.relevance_threshold,
            grounding_threshold=self.settings.grounding_threshold,
        )
        self.relevance_checker.fit(self.kb_text)
        hf_config = HFModelConfig(
            prompt_injection_model=self.settings.prompt_injection_model,
            moderation_model=self.settings.moderation_model,
            zero_shot_model=self.settings.zero_shot_model,
        )
        self.safety_engine = HFSafetyEngine(config=hf_config, enable_hf=self.settings.enable_hf_models)

    def check(self, user_input: str, user_role: Optional[str] = None) -> FilterResult:
        events: List[Dict[str, Any]] = []
        original = user_input or ""

        normalized = normalize_text(original)
        events.append({"stage": "normalization", "output_preview": normalized[:180]})

        sanitized, sanitize_events = sanitize_for_display(original)
        if sanitize_events:
            events.append({"stage": "sanitization", "events": sanitize_events})

        pii = mask_pii(sanitized, use_presidio=self.settings.use_presidio)
        if pii.found:
            events.append({"stage": "pii_masking", "entities": pii.entities})

        # Safety should see normalized/sanitized text and masked text can be used downstream.
        safety = self.safety_engine.check(sanitized)
        if pii.found and safety.decision == "allow":
            safety = SafetyResult(
                label="privacy_pii",
                decision="warn",
                risk_level="medium",
                confidence=0.90,
                backend="pii_masking",
                reasons=[f"PII detected and masked: {', '.join(pii.entities)}"],
                raw={},
            )
        events.append({"stage": "safety", "label": safety.label, "decision": safety.decision, "backend": safety.backend})

        relevance = self.relevance_checker.check(pii.masked_text)
        events.append({"stage": "kb_relevance", "score": relevance.score, "method": relevance.method})

        role = user_role or self.settings.user_role
        decision, passed, risk, reason, policy_events = apply_policy(
            safety=safety,
            relevance=relevance,
            kb_profile=self.kb_profile,
            strict_mode=self.settings.strict_mode,
            user_role=role,
        )
        if policy_events:
            events.append({"stage": "policy", "events": policy_events})

        result = FilterResult(
            decision=decision,
            passed=passed,
            risk_level=risk,
            reason=reason,
            original_input=original,
            sanitized_input=sanitized,
            pii_masked_input=pii.masked_text,
            kb_profile=self.kb_profile,
            relevance=relevance,
            safety=safety,
            user_role=role,
            events=events,
        )
        if self.settings.enable_audit_log:
            append_audit_log(result.to_dict(), self.settings.audit_log_path)
        return result
