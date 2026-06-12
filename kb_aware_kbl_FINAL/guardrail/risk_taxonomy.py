"""Shared safety taxonomy and model registry.

This mirrors the common architecture used by modern guardrail systems:
1. A safety taxonomy for input labels.
2. Separate prompt-attack detection.
3. Separate content-safety detection.
4. A policy/decision layer which decides pass, warn, or block.

The labels are intentionally stable internal labels, even if the external Hugging Face
models use different label names.
"""

BLOCK_LABELS = {
    "prompt_injection",
    "jailbreak",
    "script_or_command_injection",
    "cyber_abuse",
    "dangerous_content",
    "self_harm",
    "violence",
    "hate_abuse",
    "extremism",
    "child_safety",
    "fraud_scam",
    "sexual_content",
}

WARN_LABELS = {
    "privacy_pii",
    "misinformation",
    "copyright_violation",
    "format_issue",
}

HF_MODEL_PRESETS = {
    "prompt_injection": {
        "protectai_deberta_v2": "protectai/deberta-v3-base-prompt-injection-v2",
        "llama_prompt_guard_2_86m": "meta-llama/Llama-Prompt-Guard-2-86M"
    },
    "moderation": {
        "albert_moderation": "oxyapi/albert-moderation-001",
        "koala_text_moderation": "KoalaAI/Text-Moderation",
        "toxic_bert": "unitary/toxic-bert"
    },
    "zero_shot": {
        "deberta_xsmall_zeroshot": "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"
    },
    "semantic_relevance": {
        "minilm": "sentence-transformers/all-MiniLM-L6-v2"
    }
}

ZERO_SHOT_LABELS = [
    "safe educational question",
    "fraud scam or phishing",
    "misinformation or fake news",
    "sexual content",
    "copyright violation request",
    "format spam or low quality input",
    "hate abuse or harassment",
    "script command or code injection",
    "cyber abuse or credential theft",
    "dangerous weapons or explosives request",
    "self harm request",
    "violent wrongdoing request",
    "extremism or terrorist content",
    "child safety violation",
]

ZERO_SHOT_MAP = {
    "safe educational question": "safe",
    "fraud scam or phishing": "fraud_scam",
    "misinformation or fake news": "misinformation",
    "sexual content": "sexual_content",
    "copyright violation request": "copyright_violation",
    "format spam or low quality input": "format_issue",
    "hate abuse or harassment": "hate_abuse",
    "script command or code injection": "script_or_command_injection",
    "cyber abuse or credential theft": "cyber_abuse",
    "dangerous weapons or explosives request": "dangerous_content",
    "self harm request": "self_harm",
    "violent wrongdoing request": "violence",
    "extremism or terrorist content": "extremism",
    "child safety violation": "child_safety",
}
