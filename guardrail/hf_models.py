from typing import Any, Dict, List, Optional

from .schema import StageResult
from .taxonomy import LABEL_RISK, MODERATION_MODEL, PROMPT_INJECTION_MODEL


def _flatten_pipeline_output(output: Any) -> List[Dict[str, Any]]:
    if output is None:
        return []
    if isinstance(output, dict):
        return [output]
    if isinstance(output, list):
        if output and isinstance(output[0], list):
            flat = []
            for x in output:
                flat.extend(_flatten_pipeline_output(x))
            return flat
        return [x for x in output if isinstance(x, dict)]
    return []


class PromptInjectionModel:
    def __init__(self, model_name: str = PROMPT_INJECTION_MODEL, threshold: float = 0.75, enabled: bool = True):
        self.model_name = model_name
        self.threshold = threshold
        self.enabled = enabled
        self._pipe = None
        self._load_error = None

    def _load(self):
        if not self.enabled or self._pipe is not None or self._load_error is not None:
            return
        try:
            from transformers import pipeline
            self._pipe = pipeline("text-classification", model=self.model_name, tokenizer=self.model_name, truncation=True)
        except Exception as e:
            self._load_error = str(e)

    def scan(self, text: str) -> StageResult:
        if not self.enabled:
            return StageResult("prompt_injection_model", "SKIPPED", "safe", "low", "Hugging Face prompt-injection model disabled.", "disabled", 0.0)
        self._load()
        if self._pipe is None:
            return StageResult("prompt_injection_model", "UNAVAILABLE", "safe", "low", f"Prompt-injection model unavailable: {self._load_error}", "hf_prompt_injection_unavailable", 0.0)
        try:
            output = _flatten_pipeline_output(self._pipe(text[:4096]))
            if not output:
                return StageResult("prompt_injection_model", "ALLOW", "safe", "low", "Prompt-injection model returned no labels.", "hf_prompt_injection", 0.0)
            best = max(output, key=lambda x: float(x.get("score", 0.0)))
            label = str(best.get("label", "")).lower()
            score = float(best.get("score", 0.0))
            unsafe_by_name = any(k in label for k in ["injection", "jailbreak", "attack", "malicious", "unsafe"])
            unsafe_by_label_id = label in {"label_1", "1"}
            if score >= self.threshold and (unsafe_by_name or unsafe_by_label_id):
                return StageResult("prompt_injection_model", "BLOCK", "prompt_injection", "high", f"Prompt-injection model flagged input. label={best.get('label')}", "hf_prompt_injection", score, {"raw_output": output})
            return StageResult("prompt_injection_model", "ALLOW", "safe", "low", f"Prompt-injection model did not flag input. label={best.get('label')}", "hf_prompt_injection", score, {"raw_output": output})
        except Exception as e:
            return StageResult("prompt_injection_model", "UNAVAILABLE", "safe", "low", f"Prompt-injection model error: {e}", "hf_prompt_injection_error", 0.0)


class ModerationModel:
    def __init__(self, model_name: str = MODERATION_MODEL, threshold: float = 0.65, enabled: bool = True):
        self.model_name = model_name
        self.threshold = threshold
        self.enabled = enabled
        self._pipe = None
        self._load_error = None

    def _load(self):
        if not self.enabled or self._pipe is not None or self._load_error is not None:
            return
        try:
            from transformers import pipeline
            self._pipe = pipeline("text-classification", model=self.model_name, tokenizer=self.model_name, top_k=None, truncation=True)
        except Exception as e:
            self._load_error = str(e)

    def _map_label(self, label: str) -> str:
        l = label.lower().replace("-", "_").replace("/", "_")
        if l in {"safe", "ok", "normal", "benign", "label_0"}:
            return "safe"
        if "self" in l and "harm" in l or l == "sh":
            return "self_harm"
        if "viol" in l or l in {"v", "v1", "v2"}:
            return "violence"
        if "hate" in l or "harass" in l or l in {"h", "h1", "h2", "hr"}:
            return "hate_abuse"
        if "sexual" in l or l in {"s", "s1", "s2", "s3"}:
            return "sexual_content"
        if "danger" in l or "weapon" in l or "illegal" in l:
            return "dangerous_content"
        if "cyber" in l or "phishing" in l or "fraud" in l:
            return "cyber_abuse"
        return l

    def scan(self, text: str) -> StageResult:
        if not self.enabled:
            return StageResult("moderation_model", "SKIPPED", "safe", "low", "Hugging Face moderation model disabled.", "disabled", 0.0)
        self._load()
        if self._pipe is None:
            return StageResult("moderation_model", "UNAVAILABLE", "safe", "low", f"Moderation model unavailable: {self._load_error}", "hf_moderation_unavailable", 0.0)
        try:
            outputs = _flatten_pipeline_output(self._pipe(text[:4096]))
            unsafe_candidates = []
            for item in outputs:
                mapped = self._map_label(str(item.get("label", "")))
                score = float(item.get("score", 0.0))
                if mapped != "safe":
                    unsafe_candidates.append((mapped, score, item))
            if unsafe_candidates:
                mapped, score, item = max(unsafe_candidates, key=lambda x: x[1])
                if score >= self.threshold:
                    return StageResult("moderation_model", "BLOCK", mapped, LABEL_RISK.get(mapped, "high"), f"Moderation model flagged input. label={item.get('label')}", "hf_moderation", score, {"raw_output": outputs})
            best = max(outputs, key=lambda x: float(x.get("score", 0.0))) if outputs else {}
            return StageResult("moderation_model", "ALLOW", "safe", "low", f"Moderation model did not flag input. top_label={best.get('label')}", "hf_moderation", float(best.get("score", 0.0) or 0.0), {"raw_output": outputs})
        except Exception as e:
            return StageResult("moderation_model", "UNAVAILABLE", "safe", "low", f"Moderation model error: {e}", "hf_moderation_error", 0.0)
