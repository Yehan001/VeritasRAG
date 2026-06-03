"""
input_guardrail.py
==================

Hybrid domain-independent input filtering system.

This is the main orchestrator.

Design:
1. Normalize/sanitize input.
2. Mask PII.
3. Run deterministic rules for exact technical/security patterns.
4. Run classifier for semantic harmful meaning.
5. Produce explainable decision report.

This does NOT generate answers.
This does NOT depend on a document or domain.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from deterministic_rules import RuleEvent, run_deterministic_rules
from normalization import normalize_text, safe_display_text
from pii_masking import mask_pii
from semantic_classifier import UNSAFE_LABELS, WARN_LABELS


@dataclass
class FilterEvent:
    name: str
    layer: str
    action: str
    risk: str
    message: str


@dataclass
class GuardrailResult:
    passed: bool
    decision: str
    risk_level: str
    original_input: str
    sanitized_input: str
    classifier_label: str = ""
    classifier_confidence: float = 0.0
    blocked_checks: List[str] = field(default_factory=list)
    warning_checks: List[str] = field(default_factory=list)
    sanitized_checks: List[str] = field(default_factory=list)
    masked_checks: List[str] = field(default_factory=list)
    events: List[FilterEvent] = field(default_factory=list)
    reason: str = ""
    action_taken: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "decision": self.decision,
            "risk_level": self.risk_level,
            "original_input": self.original_input,
            "sanitized_input": self.sanitized_input,
            "classifier_label": self.classifier_label,
            "classifier_confidence": self.classifier_confidence,
            "blocked_checks": self.blocked_checks,
            "warning_checks": self.warning_checks,
            "sanitized_checks": self.sanitized_checks,
            "masked_checks": self.masked_checks,
            "reason": self.reason,
            "action_taken": self.action_taken,
            "events": [e.__dict__ for e in self.events],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class GuardrailConfig:
    block_urls = True
    strict_mode = False

    # Classifier is the main layer for semantic harmful meaning.
    use_classifier = True
    classifier_block_threshold = 0.45
    classifier_warn_threshold = 0.25

    # If classifier is uncertain but predicts an unsafe label, block these critical labels.
    critical_labels = {
        "self_harm",
        "violence_threat",
        "dangerous_content",
        "cyber_abuse",
        "child_safety",
        "extremism",
    }


class InputGuardrail:
    def __init__(
        self,
        classifier: Optional[Callable[[str], Dict[str, Any]]] = None,
        config: Optional[GuardrailConfig] = None,
    ):
        self.classifier = classifier
        self.config = config or GuardrailConfig()

    def _add_rule_events(self, events: List[FilterEvent], rule_events: List[RuleEvent]):
        for e in rule_events:
            events.append(FilterEvent(
                name=e.name,
                layer="Layer 1: deterministic_rules",
                action=e.action,
                risk=e.risk,
                message=e.message,
            ))

    def _finalize(
        self,
        original: str,
        sanitized: str,
        events: List[FilterEvent],
        classifier_label: str = "",
        classifier_confidence: float = 0.0,
    ) -> GuardrailResult:
        if self.config.strict_mode:
            for e in events:
                if e.action == "WARN":
                    e.action = "BLOCK"
                    e.risk = "MEDIUM"
                    e.message = "Strict mode: " + e.message

        blocked = [e.name for e in events if e.action == "BLOCK"]
        warnings = [e.name for e in events if e.action == "WARN"]
        sanitized_checks = [e.name for e in events if e.action == "SANITIZE"]
        masked_checks = [e.name for e in events if e.action == "MASK"]

        high = any(e.risk == "HIGH" for e in events)
        medium = any(e.risk == "MEDIUM" for e in events)

        if blocked:
            reason = next((e.message for e in events if e.action == "BLOCK"), "Input blocked.")
            return GuardrailResult(
                passed=False,
                decision="BLOCKED",
                risk_level="HIGH" if high else ("MEDIUM" if medium else "LOW"),
                original_input=original,
                sanitized_input="",
                classifier_label=classifier_label,
                classifier_confidence=classifier_confidence,
                blocked_checks=blocked,
                warning_checks=warnings,
                sanitized_checks=sanitized_checks,
                masked_checks=masked_checks,
                events=events,
                reason=reason,
                action_taken="Input stopped. It should not be sent to the downstream system.",
            )

        if warnings or sanitized_checks or masked_checks:
            return GuardrailResult(
                passed=True,
                decision="PASSED_WITH_WARNING",
                risk_level="MEDIUM" if medium else "LOW",
                original_input=original,
                sanitized_input=sanitized,
                classifier_label=classifier_label,
                classifier_confidence=classifier_confidence,
                blocked_checks=[],
                warning_checks=warnings,
                sanitized_checks=sanitized_checks,
                masked_checks=masked_checks,
                events=events,
                reason="Input allowed after warning, sanitization, or masking.",
                action_taken="Safe sanitized input may continue to the next application layer.",
            )

        return GuardrailResult(
            passed=True,
            decision="PASSED",
            risk_level="LOW",
            original_input=original,
            sanitized_input=sanitized,
            classifier_label=classifier_label,
            classifier_confidence=classifier_confidence,
            events=events,
            reason="Input allowed.",
            action_taken="Safe input may continue to the next application layer.",
        )

    def check(self, user_input: str) -> GuardrailResult:
        events: List[FilterEvent] = []

        # 1. Normalize and sanitize
        norm = normalize_text(user_input)

        for action in norm.actions:
            events.append(FilterEvent(
                name=action,
                layer="Layer 3: normalization",
                action="SANITIZE" if "removed" in action or "cleaned" in action else "WARN",
                risk="MEDIUM" if "html" in action or "script" in action else "LOW",
                message=f"Normalization/sanitization action: {action}",
            ))

        # 2. PII masking after sanitization
        pii_masked_text, pii_labels = mask_pii(norm.sanitized)
        if pii_labels:
            norm.sanitized = pii_masked_text
            for label in pii_labels:
                events.append(FilterEvent(
                    name=f"privacy_{label}",
                    layer="Layer 3: pii_masking",
                    action="MASK",
                    risk="MEDIUM",
                    message=f"PII masked: {label}",
                ))

        # Rebuild classifier text after PII masking
        classifier_text = f"RAW: {norm.original}\nSANITIZED: {norm.sanitized}\nNORMALIZED: {norm.normalized}"

        # 3. Deterministic rules
        rule_events = run_deterministic_rules(norm, block_urls=self.config.block_urls)
        self._add_rule_events(events, rule_events)

        classifier_label = ""
        classifier_confidence = 0.0

        # 4. Classifier semantic layer
        if self.config.use_classifier and self.classifier is not None:
            try:
                cls = self.classifier(classifier_text)
                label = str(cls.get("label", "unknown"))
                confidence = float(cls.get("confidence", 0.0))
                safe = bool(cls.get("safe", label not in UNSAFE_LABELS))
                reason = str(cls.get("reason", f"Classifier predicted {label}."))
                model_name = str(cls.get("model_name", "classifier"))

                classifier_label = label
                classifier_confidence = confidence

                if (not safe and confidence >= self.config.classifier_block_threshold) or (
                    label in self.config.critical_labels and confidence >= self.config.classifier_warn_threshold
                ):
                    events.append(FilterEvent(
                        name=f"classifier_{label}",
                        layer=f"Layer 2: semantic_classifier/{model_name}",
                        action="BLOCK",
                        risk="HIGH",
                        message=reason,
                    ))
                elif not safe or label in WARN_LABELS:
                    events.append(FilterEvent(
                        name=f"classifier_{label}",
                        layer=f"Layer 2: semantic_classifier/{model_name}",
                        action="WARN",
                        risk="MEDIUM",
                        message=reason,
                    ))
                else:
                    events.append(FilterEvent(
                        name="classifier_safe",
                        layer=f"Layer 2: semantic_classifier/{model_name}",
                        action="PASS",
                        risk="LOW",
                        message=reason,
                    ))
            except Exception as exc:
                events.append(FilterEvent(
                    name="classifier_error",
                    layer="Layer 2: semantic_classifier",
                    action="WARN",
                    risk="LOW",
                    message=f"Classifier skipped due to error: {exc}",
                ))

        return self._finalize(
            original=norm.original,
            sanitized=norm.sanitized,
            events=events,
            classifier_label=classifier_label,
            classifier_confidence=classifier_confidence,
        )


# Session-level multi-turn escalation
_STAGE_1 = re.compile(r"\b(chemical|chemicals|precursors?|industrial\s+facilit(?:y|ies)|fertilizer|oxidizer|solvent|reactive\s+materials?|access|credentials?|password|network)\b", re.I)
_STAGE_2 = re.compile(r"\b(unstable|reactive|combine|combination|mixture|energetic|detonation|explosive|blast|ignite|bypass|steal|extract|crack)\b", re.I)


def check_escalation_with_history(question: str, history: List[Dict[str, Any]], window_seconds: int = 900):
    now = time.time()
    history = [h for h in history if now - h.get("time", 0) <= window_seconds]

    stage = 0
    if _STAGE_1.search(question or ""):
        stage = max(stage, 1)
    if _STAGE_2.search(question or ""):
        stage = max(stage, 2)

    if stage:
        history.append({"question_preview": (question or "")[:120], "stage": stage, "time": now})
        history = history[-10:]

    stages = [h.get("stage", 0) for h in history]
    if len(stages) >= 2 and max(stages) >= 2 and min(stages) >= 1:
        return True, "Multi-turn harmful intent escalation detected across recent inputs.", history

    return False, "", history
