from dataclasses import dataclass
from typing import List, Optional

from .audit_logger import AuditLogger
from .hf_models import ModerationModel, PromptInjectionModel
from .pii_masker import PIIMasker
from .rules import RuleScanner
from .schema import GuardrailResult, PIIResult, StageResult
from .semantic_matcher import SentenceTransformerRiskMatcher
from .taxonomy import LABEL_RISK, MODERATION_MODEL, PROMPT_INJECTION_MODEL, SENTENCE_TRANSFORMER_MODEL
from .text_utils import sanitize_for_display


@dataclass
class GuardrailSettings:
    enable_hf_models: bool = True
    enable_sentence_transformer: bool = True
    use_presidio: bool = True
    enable_audit_log: bool = True
    prompt_injection_threshold: float = 0.75
    moderation_threshold: float = 0.65
    semantic_threshold: float = 0.62
    prompt_injection_model: str = PROMPT_INJECTION_MODEL
    moderation_model: str = MODERATION_MODEL
    sentence_transformer_model: str = SENTENCE_TRANSFORMER_MODEL


class InputGuardrail:
    """Input-filtering-only guardrail with short-circuit execution.

    Execution order:
    1. Deterministic rules
    2. Prompt-injection model
    3. Moderation model
    4. SentenceTransformer semantic risk matcher

    If any stage blocks, later stages are skipped.
    """

    STAGE_ORDER = [
        "deterministic_rules",
        "prompt_injection_model",
        "moderation_model",
        "sentence_transformer_risk_matcher",
    ]

    def __init__(self, settings: Optional[GuardrailSettings] = None):
        self.settings = settings or GuardrailSettings()
        self.rules = RuleScanner()
        self.pii_masker = PIIMasker(use_presidio=self.settings.use_presidio)
        self.prompt_model = PromptInjectionModel(
            model_name=self.settings.prompt_injection_model,
            threshold=self.settings.prompt_injection_threshold,
            enabled=self.settings.enable_hf_models,
        )
        self.moderation_model = ModerationModel(
            model_name=self.settings.moderation_model,
            threshold=self.settings.moderation_threshold,
            enabled=self.settings.enable_hf_models,
        )
        self.semantic_matcher = SentenceTransformerRiskMatcher(
            model_name=self.settings.sentence_transformer_model,
            threshold=self.settings.semantic_threshold,
            enabled=self.settings.enable_sentence_transformer,
        )
        self.audit = AuditLogger(enabled=self.settings.enable_audit_log)

    def _skipped_after(self, stage_name: str) -> List[str]:
        if stage_name not in self.STAGE_ORDER:
            return []
        i = self.STAGE_ORDER.index(stage_name)
        return self.STAGE_ORDER[i + 1:]

    def _finalize(self, decision: str, label: str, reason: str, original: str, pii: PIIResult, checked: List[StageResult], triggered: StageResult, skipped: List[str]) -> GuardrailResult:
        risk = LABEL_RISK.get(label, triggered.risk or "low")
        sanitized = sanitize_for_display(pii.masked_text)
        result = GuardrailResult(
            decision=decision,
            passed=(decision != "BLOCKED"),
            risk=risk,
            safety_label=label,
            reason=reason,
            original_input=original,
            sanitized_input=sanitized,
            triggered_stage=triggered.stage,
            safety_backend=triggered.backend,
            checked_stages=checked,
            skipped_stages=skipped,
            pii=pii,
        )
        audit_id = self.audit.write(result)
        result.audit_id = audit_id
        return result

    def check(self, user_input: str) -> GuardrailResult:
        original = user_input or ""
        checked: List[StageResult] = []

        # PII masking is not a blocking stage; it prepares safe downstream text.
        pii = self.pii_masker.mask(original)

        # 1. Deterministic rules on raw input before sanitization.
        r = self.rules.scan(original)
        checked.append(r)
        if r.action == "BLOCK":
            return self._finalize("BLOCKED", r.label, f"Unsafe input detected by deterministic rules. {r.reason}", original, pii, checked, r, self._skipped_after(r.stage))

        warning_stage: Optional[StageResult] = r if r.action == "WARN" else None

        # 2. Prompt injection model.
        r = self.prompt_model.scan(original)
        checked.append(r)
        if r.action == "BLOCK":
            return self._finalize("BLOCKED", r.label, f"Unsafe input detected by prompt-injection model. {r.reason}", original, pii, checked, r, self._skipped_after(r.stage))
        if r.action == "WARN" and warning_stage is None:
            warning_stage = r

        # 3. Moderation model.
        r = self.moderation_model.scan(original)
        checked.append(r)
        if r.action == "BLOCK":
            return self._finalize("BLOCKED", r.label, f"Unsafe input detected by moderation model. {r.reason}", original, pii, checked, r, self._skipped_after(r.stage))
        if r.action == "WARN" and warning_stage is None:
            warning_stage = r

        # 4. SentenceTransformer semantic matcher.
        r = self.semantic_matcher.scan(original)
        checked.append(r)
        if r.action == "BLOCK":
            return self._finalize("BLOCKED", r.label, f"Unsafe input detected by SentenceTransformer semantic matcher. {r.reason}", original, pii, checked, r, self._skipped_after(r.stage))
        if r.action == "WARN" and warning_stage is None:
            warning_stage = r

        # Non-blocking warning priority: PII first, then other warnings.
        if pii.has_pii:
            trigger = StageResult("pii_masking", "WARN", "privacy_pii", "medium", "PII detected and masked.", pii.backend, 1.0, {"entities": pii.entities})
            checked.append(trigger)
            return self._finalize("PASSED_WITH_WARNING", "privacy_pii", "Input is safe, but private information was detected and masked.", original, pii, checked, trigger, [])

        if warning_stage is not None:
            return self._finalize("PASSED_WITH_WARNING", warning_stage.label, warning_stage.reason, original, pii, checked, warning_stage, [])

        trigger = StageResult("final_decision", "ALLOW", "safe", "low", "No stage blocked or warned.", "policy", 0.0)
        checked.append(trigger)
        return self._finalize("PASSED", "safe", "Input passed all enabled guardrail stages.", original, pii, checked, trigger, [])
