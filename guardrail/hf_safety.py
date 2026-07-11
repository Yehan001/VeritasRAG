import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Block ALL outbound Hugging Face / datasets network calls at the library level.
# These must be set before any transformers / huggingface_hub import happens.
# ---------------------------------------------------------------------------
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from .fallback_rules import fallback_safety_check
from .schema import SafetyResult
from .text_utils import normalize_text, try_decode_base64_payloads

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_PACKAGE_DIR)
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")


def _local(folder_name: str) -> str:
    """Return the absolute path to a downloaded model folder."""
    return os.path.join(_MODELS_DIR, folder_name)


@dataclass
class HFModelConfig:
    prompt_injection_model: str = "meta-llama/Llama-Prompt-Guard-2-86M"
    moderation_model: str = _local("oxyapi__albert-moderation-001")
    zero_shot_model: str = _local("cross-encoder__nli-MiniLM2-L6-H768")
    prompt_threshold: float = 0.70
    moderation_threshold: float = 0.55
    zero_shot_threshold: float = 0.62
    device: int = -1  # -1 = CPU; set to 0 for first GPU


from .risk_taxonomy import BLOCK_LABELS, WARN_LABELS, ZERO_SHOT_LABELS, ZERO_SHOT_MAP

UNSAFE_LABEL_KEYWORDS = [
    "unsafe", "toxic", "hate", "harassment", "sexual", "self", "harm", "violence",
    "threat", "dangerous", "illegal", "minor", "child", "abuse", "graphic",
]

SAFE_LABEL_KEYWORDS = ["safe", "benign", "legitimate", "normal", "clean"]


class HFSafetyEngine:
    """Free local HF safety ensemble with fallback rules.

    The system can run without transformers installed. If a model fails to load,
    fallback rules still run.  All model loading uses local_files_only=True so
    no network call is ever attempted.
    """

    def __init__(self, config: Optional[HFModelConfig] = None, enable_hf: bool = True) -> None:
        self.config = config or HFModelConfig()
        self.enable_hf = enable_hf
        self._prompt_pipe = None
        self._moderation_pipe = None
        self._zero_shot_pipe = None
        self.load_errors: List[str] = []

    def _load_pipeline(self, task: str, model: str, **kwargs):
        start = time.perf_counter()
        try:
            from transformers import pipeline
            loaded = pipeline(
                task,
                model=model,
                device=self.config.device,
                local_files_only=True,
                **kwargs,
            )
            latency = time.perf_counter() - start
            print(f"[_load_pipeline] task={task} model={model} status=loaded latency={latency:.4f} seconds")
            return loaded
        except Exception as exc:
            latency = time.perf_counter() - start
            print(f"[_load_pipeline] task={task} model={model} status=failed latency={latency:.4f} seconds error={exc}")
            self.load_errors.append(f"{model}: {exc}")
            return None

    @property
    def prompt_pipe(self):
        if not self.enable_hf:
            return None
        if self._prompt_pipe is None:
            self._prompt_pipe = self._load_pipeline(
                "text-classification",
                self.config.prompt_injection_model,
                truncation=True,
            )
        return self._prompt_pipe

    @property
    def moderation_pipe(self):
        if not self.enable_hf:
            return None
        if self._moderation_pipe is None:
            self._moderation_pipe = self._load_pipeline(
                "text-classification",
                self.config.moderation_model,
                truncation=True,
                top_k=None,
            )
        return self._moderation_pipe

    @property
    def zero_shot_pipe(self):
        if not self.enable_hf:
            return None
        if self._zero_shot_pipe is None:
            self._zero_shot_pipe = self._load_pipeline(
                "zero-shot-classification",
                self.config.zero_shot_model,
            )
        return self._zero_shot_pipe

    # ------------------------------------------------------------------
    # Main check entry point
    # ------------------------------------------------------------------
    def check(self, text: str) -> SafetyResult:
        start = time.perf_counter()

        # Always run fallback first for exact technical/prompt/safety attacks.
        fallback = fallback_safety_check(text)
        if fallback.decision == "block":
            latency = time.perf_counter() - start
            print(f"[HFSafetyEngine.check] backend=fallback_rules latency={latency:.4f} seconds")
            return fallback

        # Check decoded base64 with the whole engine.
        for payload in try_decode_base64_payloads(text):
            decoded_result = fallback_safety_check(payload)
            if decoded_result.decision == "block":
                decoded_result.backend = "base64_decoded_fallback"
                decoded_result.reasons.append("unsafe decoded base64 payload")
                latency = time.perf_counter() - start
                print(f"[HFSafetyEngine.check] backend=base64_decoded_fallback latency={latency:.4f} seconds")
                return decoded_result

        if not self.enable_hf:
            latency = time.perf_counter() - start
            print(f"[HFSafetyEngine.check] backend=fallback_rules_only latency={latency:.4f} seconds")
            return fallback

        candidates: List[SafetyResult] = [fallback]

        prompt_result = self._check_prompt_injection(text)
        if prompt_result:
            candidates.append(prompt_result)

        moderation_result = self._check_moderation(text)
        if moderation_result:
            candidates.append(moderation_result)

        # Zero-shot is the slowest model (~0.4-0.5s) and adds little once a
        # confident block has already been found by the cheaper models above.
        already_confident_block = any(
            r.decision == "block" and r.confidence >= 0.85 for r in candidates
        )
        if already_confident_block:
            latency = time.perf_counter() - start
            print(f"[HFSafetyEngine.check] backend=hf_ensemble_short_circuit latency={latency:.4f} seconds")
            blocks = [r for r in candidates if r.decision == "block"]
            return sorted(blocks, key=lambda r: r.confidence, reverse=True)[0]

        zero_result = self._check_zero_shot(text)
        if zero_result:
            candidates.append(zero_result)

        # Priority: block > warn > allow, then highest confidence.
        blocks = [r for r in candidates if r.decision == "block"]
        if blocks:
            result = sorted(blocks, key=lambda r: r.confidence, reverse=True)[0]
        else:
            warns = [r for r in candidates if r.decision == "warn"]
            if warns:
                result = sorted(warns, key=lambda r: r.confidence, reverse=True)[0]
            else:
                result = sorted(candidates, key=lambda r: r.confidence, reverse=True)[0]

        latency = time.perf_counter() - start
        print(f"[HFSafetyEngine.check] backend=hf_ensemble latency={latency:.4f} seconds")
        return result

    # ------------------------------------------------------------------
    # Individual model checks
    # ------------------------------------------------------------------
    def _check_prompt_injection(self, text: str) -> Optional[SafetyResult]:
        pipe = self.prompt_pipe
        if pipe is None:
            return None
        try:
            start = time.perf_counter()
            out = pipe(text[:2000])
            latency = time.perf_counter() - start
            print(f"[_check_prompt_injection] model={self.config.prompt_injection_model} latency={latency:.4f} seconds")

            item = out[0] if isinstance(out, list) else out
            label = str(item.get("label", "")).lower()
            score = float(item.get("score", 0.0))
            unsafe = any(k in label for k in ["inject", "attack", "jailbreak", "malicious", "unsafe"])
            if label in {"label_1", "1"}:
                unsafe = True
            if unsafe and score >= self.config.prompt_threshold:
                return SafetyResult(
                    "prompt_injection", "block", "high", score,
                    "hf_prompt_injection", [f"HF prompt model label={label}"], {"output": out},
                )
            if any(k in label for k in SAFE_LABEL_KEYWORDS):
                return SafetyResult(
                    "safe", "allow", "low", score,
                    "hf_prompt_injection", [f"HF prompt model label={label}"], {"output": out},
                )
        except Exception as exc:
            self.load_errors.append(f"prompt check failed: {exc}")
        return None

    def _check_moderation(self, text: str) -> Optional[SafetyResult]:
        pipe = self.moderation_pipe
        if pipe is None:
            return None
        try:
            start = time.perf_counter()
            out = pipe(text[:2000])
            latency = time.perf_counter() - start
            print(f"[_check_moderation] model={self.config.moderation_model} latency={latency:.4f} seconds")

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
                return SafetyResult(
                    internal, "block", "high", best_unsafe[1],
                    "hf_moderation", [f"HF moderation label={best_unsafe[0]}"], {"output": out},
                )
            if best_safe:
                return SafetyResult(
                    "safe", "allow", "low", best_safe[1],
                    "hf_moderation", [f"HF moderation label={best_safe[0]}"], {"output": out},
                )
        except Exception as exc:
            self.load_errors.append(f"moderation check failed: {exc}")
        return None

    def _check_zero_shot(self, text: str) -> Optional[SafetyResult]:
        pipe = self.zero_shot_pipe
        if pipe is None:
            return None
        try:
            start = time.perf_counter()
            out = pipe(text[:2000], candidate_labels=ZERO_SHOT_LABELS, multi_label=False)
            latency = time.perf_counter() - start
            print(f"[_check_zero_shot] model={self.config.zero_shot_model} latency={latency:.4f} seconds")

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
                return SafetyResult(
                    internal, "block", "high", score,
                    "hf_zero_shot", [f"zero-shot best={best_label}"], {"output": out},
                )
            if internal in WARN_LABELS:
                return SafetyResult(
                    internal, "warn", "medium", score,
                    "hf_zero_shot", [f"zero-shot best={best_label}"], {"output": out},
                )
            return SafetyResult(
                "safe", "allow", "low", score,
                "hf_zero_shot", [f"zero-shot best={best_label}"], {"output": out},
            )
        except Exception as exc:
            self.load_errors.append(f"zero-shot check failed: {exc}")
        return None
    
    def warmup(self) -> None:
        if not self.enable_hf:
            return
        start = time.perf_counter()
        from transformers import pipeline  # noqa: F401  (import once, single-threaded, to avoid race)
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(lambda: self.prompt_pipe),
                executor.submit(lambda: self.moderation_pipe),
                executor.submit(lambda: self.zero_shot_pipe),
            ]
            for f in futures:
                f.result()

        latency = time.perf_counter() - start
        print(f"[HFSafetyEngine.warmup] parallel_load_all_3_models latency={latency:.4f} seconds")

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


# ---------------------------------------------------------------------------
# Shared, process-wide engine for API usage.
#
# In the Streamlit app, one KBAwareInputFilter (and its own HFSafetyEngine)
# is created per session, which is fine for a single-user demo. In the API,
# many tenants share one running process, so the 3 HF models must be loaded
# and warmed up exactly ONCE for the whole process, not once per tenant KB.
# api.py imports this instance, warms it up in its startup event, and injects
# it into every KBAwareInputFilter it creates (see input_filter.py).
# ---------------------------------------------------------------------------
_SHARED_SAFETY_ENGINE = HFSafetyEngine(enable_hf=True)