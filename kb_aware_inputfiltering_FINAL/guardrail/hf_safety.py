from dataclasses import dataclass
from typing import Dict, List, Optional

from .fallback_rules import fallback_safety_check
from .schema import SafetyResult
from .text_utils import normalize_text, try_decode_base64_payloads


@dataclass
class HFModelConfig:
    prompt_injection_model: str = "protectai/deberta-v3-base-prompt-injection-v2"
    moderation_model: str = "oxyapi/albert-moderation-001"
    zero_shot_model: str = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"
    prompt_threshold: float = 0.70
    moderation_threshold: float = 0.55
    zero_shot_threshold: float = 0.62
    device: int = -1


from .risk_taxonomy import BLOCK_LABELS, WARN_LABELS, ZERO_SHOT_LABELS, ZERO_SHOT_MAP

UNSAFE_LABEL_KEYWORDS = [
    "unsafe", "toxic", "hate", "harassment", "sexual", "self", "harm", "violence",
    "threat", "dangerous", "illegal", "minor", "child", "abuse", "graphic",
]

SAFE_LABEL_KEYWORDS = ["safe", "benign", "legitimate", "normal", "clean"]


class HFSafetyEngine:
    """Free local HF safety ensemble with fallback rules.

    The system can run without transformers installed. If a model fails to load, fallback rules still run.
    """

    def __init__(self, config: Optional[HFModelConfig] = None, enable_hf: bool = True) -> None:
        self.config = config or HFModelConfig()
        self.enable_hf = enable_hf
        self._prompt_pipe = None
        self._moderation_pipe = None
        self._zero_shot_pipe = None
        self.load_errors: List[str] = []

    def _load_pipeline(self, task: str, model: str, **kwargs):
        try:
            from transformers import pipeline
            return pipeline(task, model=model, device=self.config.device, **kwargs)
        except Exception as exc:
            self.load_errors.append(f"{model}: {exc}")
            return None

    @property
    def prompt_pipe(self):
        if not self.enable_hf:
            return None
        if self._prompt_pipe is None:
            self._prompt_pipe = self._load_pipeline("text-classification", self.config.prompt_injection_model, truncation=True)
        return self._prompt_pipe

    @property
    def moderation_pipe(self):
        if not self.enable_hf:
            return None
        if self._moderation_pipe is None:
            self._moderation_pipe = self._load_pipeline("text-classification", self.config.moderation_model, truncation=True, top_k=None)
        return self._moderation_pipe

    @property
    def zero_shot_pipe(self):
        if not self.enable_hf:
            return None
        if self._zero_shot_pipe is None:
            self._zero_shot_pipe = self._load_pipeline("zero-shot-classification", self.config.zero_shot_model)
        return self._zero_shot_pipe

    def check(self, text: str) -> SafetyResult:
        # Always run fallback first for exact technical/prompt/safety attacks.
        fallback = fallback_safety_check(text)
        if fallback.decision == "block":
            return fallback

        # Check decoded base64 with the whole engine.
        for payload in try_decode_base64_payloads(text):
            decoded_result = fallback_safety_check(payload)
            if decoded_result.decision == "block":
                decoded_result.backend = "base64_decoded_fallback"
                decoded_result.reasons.append("unsafe decoded base64 payload")
                return decoded_result

        if not self.enable_hf:
            return fallback

        candidates: List[SafetyResult] = [fallback]

        prompt_result = self._check_prompt_injection(text)
        if prompt_result:
            candidates.append(prompt_result)

        moderation_result = self._check_moderation(text)
        if moderation_result:
            candidates.append(moderation_result)

        zero_result = self._check_zero_shot(text)
        if zero_result:
            candidates.append(zero_result)

        # Priority: block > warn > allow, then highest confidence.
        blocks = [r for r in candidates if r.decision == "block"]
        if blocks:
            return sorted(blocks, key=lambda r: r.confidence, reverse=True)[0]
        warns = [r for r in candidates if r.decision == "warn"]
        if warns:
            return sorted(warns, key=lambda r: r.confidence, reverse=True)[0]
        return sorted(candidates, key=lambda r: r.confidence, reverse=True)[0]

    def _check_prompt_injection(self, text: str) -> Optional[SafetyResult]:
        pipe = self.prompt_pipe
        if pipe is None:
            return None
        try:
            out = pipe(text[:2000])
            item = out[0] if isinstance(out, list) else out
            label = str(item.get("label", "")).lower()
            score = float(item.get("score", 0.0))
            unsafe = any(k in label for k in ["inject", "attack", "jailbreak", "malicious", "unsafe"])
            # Many binary HF models expose LABEL_1 as positive class.
            if label in {"label_1", "1"}:
                unsafe = True
            if unsafe and score >= self.config.prompt_threshold:
                return SafetyResult("prompt_injection", "block", "high", score, "hf_prompt_injection", [f"HF prompt model label={label}"], {"output": out})
            if any(k in label for k in SAFE_LABEL_KEYWORDS):
                return SafetyResult("safe", "allow", "low", score, "hf_prompt_injection", [f"HF prompt model label={label}"], {"output": out})
        except Exception as exc:
            self.load_errors.append(f"prompt check failed: {exc}")
        return None

    def _check_moderation(self, text: str) -> Optional[SafetyResult]:
        pipe = self.moderation_pipe
        if pipe is None:
            return None
        try:
            out = pipe(text[:2000])
            flat = out[0] if out and isinstance(out[0], list) else out
            if isinstance(flat, dict):
                flat = [flat]
            best_unsafe = None
            best_safe = None
            for item in flat:
                label = str(item.get("label", "")).lower()
                score = float(item.get("score", 0.0))
                if any(k in label for k in UNSAFE_LABEL_KEYWORDS):
                    if best_unsafe is None or score > best_unsafe[1]:
                        best_unsafe = (label, score)
                if any(k in label for k in SAFE_LABEL_KEYWORDS):
                    if best_safe is None or score > best_safe[1]:
                        best_safe = (label, score)
            if best_unsafe and best_unsafe[1] >= self.config.moderation_threshold:
                internal = self._map_moderation_label(best_unsafe[0])
                return SafetyResult(internal, "block", "high", best_unsafe[1], "hf_moderation", [f"HF moderation label={best_unsafe[0]}"], {"output": out})
            if best_safe:
                return SafetyResult("safe", "allow", "low", best_safe[1], "hf_moderation", [f"HF moderation label={best_safe[0]}"], {"output": out})
        except Exception as exc:
            self.load_errors.append(f"moderation check failed: {exc}")
        return None

    def _check_zero_shot(self, text: str) -> Optional[SafetyResult]:
        pipe = self.zero_shot_pipe
        if pipe is None:
            return None
        try:
            out = pipe(text[:2000], candidate_labels=ZERO_SHOT_LABELS, multi_label=False)
            labels = out.get("labels", [])
            scores = out.get("scores", [])
            if not labels:
                return None
            best_label = labels[0]
            score = float(scores[0])
            internal = ZERO_SHOT_MAP.get(best_label, "safe")
            if score < self.config.zero_shot_threshold:
                return None
            if internal in BLOCK_LABELS:
                return SafetyResult(internal, "block", "high", score, "hf_zero_shot", [f"zero-shot best={best_label}"], {"output": out})
            if internal in WARN_LABELS:
                return SafetyResult(internal, "warn", "medium", score, "hf_zero_shot", [f"zero-shot best={best_label}"], {"output": out})
            return SafetyResult("safe", "allow", "low", score, "hf_zero_shot", [f"zero-shot best={best_label}"], {"output": out})
        except Exception as exc:
            self.load_errors.append(f"zero-shot check failed: {exc}")
        return None

    @staticmethod
    def _map_moderation_label(label: str) -> str:
        label = normalize_text(label).lower()
        if "self" in label or "suicide" in label:
            return "self_harm"
        if "hate" in label or "harassment" in label or "abuse" in label:
            return "hate_abuse"
        if "sexual" in label or "minor" in label or "child" in label:
            return "sexual_content"
        if "violence" in label or "threat" in label or "graphic" in label:
            return "violence"
        if "danger" in label or "illegal" in label:
            return "dangerous_content"
        return "unsafe_content"
