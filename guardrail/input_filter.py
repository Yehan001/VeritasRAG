import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .audit_logger import append_audit_log
from .hf_safety import HFModelConfig, HFSafetyEngine
from .kb_profiler import profile_kb
from .kb_relevance import KBRelevanceChecker
from .pii_masker import detect_pii
from .policy_engine import apply_policy
from .schema import FilterResult, KBProfile, PIIResult, RelevanceResult, SafetyResult
from .text_utils import normalize_text, sanitize_for_display


@dataclass
class GuardrailSettings:
    strict_mode: bool = False
    kb_authoritative_mode: bool = True
    enable_hf_models: bool = True
    use_presidio: bool = True
    relevance_threshold: float = 0.25
    grounding_threshold: float = 0.35
    user_role: str = "public_user"
    prompt_injection_model: str = "protectai/deberta-v3-small-prompt-injection-v2"
    moderation_model: str = "oxyapi/albert-moderation-001"
    zero_shot_model: str = "cross-encoder/nli-MiniLM2-L6-H768"
    enable_audit_log: bool = True
    audit_log_path: str = "guardrail_audit_log.jsonl"


class KBAwareInputFilter:
    def __init__(
        self,
        kb_text: str,
        settings: Optional[GuardrailSettings] = None,
        safety_engine: Optional[HFSafetyEngine] = None,
        kb_profile: Optional[KBProfile] = None,
        relevance_checker: Optional[KBRelevanceChecker] = None,
    ) -> None:
        start_init = time.perf_counter()

        self.settings = settings or GuardrailSettings()
        self.kb_text = kb_text or ""

        if kb_profile is not None and relevance_checker is not None:
            # API path: reuse the profile/index already fitted and cached by
            # kb_registry.py instead of recomputing them on every request.
            self.kb_profile = kb_profile
            self.relevance_checker = relevance_checker
        else:
            # Streamlit / CLI path: unchanged behaviour, fit fresh each time.
            start_profile = time.perf_counter()
            self.kb_profile = profile_kb(self.kb_text)
            profile_latency = time.perf_counter() - start_profile
            print(f"[KBAwareInputFilter.__init__] profile_kb latency={profile_latency:.4f} seconds")

            self.relevance_checker = KBRelevanceChecker(
                relevance_threshold=self.settings.relevance_threshold,
                grounding_threshold=self.settings.grounding_threshold,
            )
            self.relevance_checker.fit(self.kb_text)

        if safety_engine is not None:
            # API path: reuse the one process-wide engine (already warmed up
            # by api.py's startup event) instead of loading the 3 HF models
            # again for every tenant KB.
            self.safety_engine = safety_engine
        else:
            # Streamlit / CLI path: unchanged behaviour, own engine per filter.
            hf_config = HFModelConfig(
                prompt_injection_model=self.settings.prompt_injection_model,
                moderation_model=self.settings.moderation_model,
                zero_shot_model=self.settings.zero_shot_model,
            )
            self.safety_engine = HFSafetyEngine(config=hf_config, enable_hf=self.settings.enable_hf_models)
            self.safety_engine.warmup()

        init_latency = time.perf_counter() - start_init
        print(f"[KBAwareInputFilter.__init__] total_setup_latency={init_latency:.4f} seconds")

    def check(
        self,
        user_input: str,
        user_role: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> FilterResult:
        start_total = time.perf_counter()

        events: List[Dict[str, Any]] = []
        original = user_input or ""

        start = time.perf_counter()
        normalized = normalize_text(original)
        latency = time.perf_counter() - start
        print(f"[KBAwareInputFilter.check] normalize_text latency={latency:.4f} seconds")
        events.append({"stage": "normalization", "output_preview": normalized[:180]})

        start = time.perf_counter()
        sanitized, sanitize_events = sanitize_for_display(original)
        latency = time.perf_counter() - start
        print(f"[KBAwareInputFilter.check] sanitize_for_display latency={latency:.4f} seconds")
        if sanitize_events:
            events.append({"stage": "sanitization", "events": sanitize_events})

        # PII detection only — detects and flags PII, original text passed through unchanged
        pii = detect_pii(sanitized, use_presidio=self.settings.use_presidio)
        if pii.found:
            events.append({"stage": "pii_detection", "entities": pii.entities})

        # Safety checks raw original input before sanitization
        safety = self.safety_engine.check(original)
        if pii.found and safety.decision == "allow":
            safety = SafetyResult(
                label="privacy_pii",
                decision="warn",
                risk_level="medium",
                confidence=0.90,
                backend="pii_detection",
                reasons=[f"PII detected: {', '.join(pii.entities)}"],
                raw={},
            )
        events.append({"stage": "safety", "label": safety.label, "decision": safety.decision, "backend": safety.backend})

        # KB relevance runs on sanitized text
        relevance = self.relevance_checker.check(sanitized)
        events.append({"stage": "kb_relevance", "score": relevance.score, "method": relevance.method})

        role = user_role or self.settings.user_role

        start = time.perf_counter()
        decision, passed, risk, reason, policy_events = apply_policy(
            safety=safety,
            relevance=relevance,
            kb_profile=self.kb_profile,
            strict_mode=self.settings.strict_mode,
            user_role=role,
            kb_authoritative_mode=self.settings.kb_authoritative_mode,
            question_text=sanitized,
        )
        latency = time.perf_counter() - start
        print(f"[KBAwareInputFilter.check] apply_policy latency={latency:.4f} seconds")
        if policy_events:
            events.append({"stage": "policy", "events": policy_events})

        result = FilterResult(
            decision=decision,
            passed=passed,
            risk_level=risk,
            reason=reason,
            original_input=original,
            sanitized_input=sanitized,
            kb_profile=self.kb_profile,
            relevance=relevance,
            safety=safety,
            pii=pii,
            user_role=role,
            events=events,
            tenant_id=tenant_id,
        )

        if self.settings.enable_audit_log:
            start = time.perf_counter()
            append_audit_log(result.to_dict(), self.settings.audit_log_path)
            latency = time.perf_counter() - start
            print(f"[KBAwareInputFilter.check] append_audit_log latency={latency:.4f} seconds")

        total_latency = time.perf_counter() - start_total
        print(f"[KBAwareInputFilter.check] TOTAL_FULL_PIPELINE_LATENCY={total_latency:.4f} seconds")

        return result