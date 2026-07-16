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

LABEL_RISK = {
    "safe": "low",
    "privacy_pii": "medium",
    "format_issue": "low",
    "copyright_violation": "medium",
    "misinformation": "medium",
    "prompt_injection": "high",
    "jailbreak": "high",
    "script_or_command_injection": "high",
    "cyber_abuse": "high",
    "dangerous_content": "high",
    "self_harm": "high",
    "violence": "high",
    "hate_abuse": "high",
    "extremism": "high",
    "child_safety": "high",
    "fraud_scam": "high",
    "sexual_content": "high",
}

PROMPT_INJECTION_MODEL = "protectai/deberta-v3-small-prompt-injection-v2"
MODERATION_MODEL = "oxyapi/albert-moderation-001"
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
