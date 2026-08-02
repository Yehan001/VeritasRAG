from openai import OpenAI
import os

from guardrail.veritasrag_input_guardrail import InputGuardrail, GuardrailSettings

# Use environment variables for credentials and endpoint (do NOT commit keys to source).
client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1",
)

# Guardrail instance used to check and sanitize user input before sending to the model
# Lightweight settings by default: disable heavy HF models and Presidio unless available.
_guardrail = InputGuardrail(
    GuardrailSettings(
        enable_hf_models=False,
        enable_semantic_matcher=True,
        use_presidio=False,
        enable_audit_log=False,
    )
)


def get_ai_response(messages):
    # Find the last user message and run it through the input guardrail
    user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        try:
            if messages[i].get("role") == "user":
                user_idx = i
                break
        except Exception:
            continue

    if user_idx is not None:
        user_content = messages[user_idx].get("content", "")
        result = _guardrail.check(user_content)
        if result.decision == "BLOCKED":
            return f"Request blocked: {result.reason}"
        # Replace the user's content with the sanitized/masked version before calling the model
        messages[user_idx]["content"] = result.sanitized_input

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content
