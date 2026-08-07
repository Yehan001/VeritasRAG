import os
from dotenv import load_dotenv
from openai import OpenAI

from guardrail.veritasrag_input_guardrail import InputGuardrail, GuardrailSettings
from rag_faithfulness_checker import check_faithfulness

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

_guardrail = InputGuardrail(
    GuardrailSettings(
        enable_hf_models=False,
        enable_semantic_matcher=True,
        use_presidio=False,
        enable_audit_log=False,
    )
)

_KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base", "sample_kb.txt")


def _load_knowledge_base() -> str:
    try:
        with open(_KB_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def get_ai_response(messages):
    user_idx = next((i for i in range(len(messages) - 1, -1, -1)
                      if messages[i].get("role") == "user"), None)
    if user_idx is None:
        return "No user message found."

    # 1. Input filtering (already working)
    input_result = _guardrail.check(messages[user_idx].get("content", ""))
    if input_result.decision == "BLOCKED":
        return f"Request blocked: {input_result.reason}"
    messages[user_idx]["content"] = input_result.sanitized_input

    source_text = _load_knowledge_base()

    # 2. Inject KB content so the model actually answers from it,
    #    instead of guessing from its own general knowledge.
    if source_text.strip():
        context_message = {
            "role": "system",
            "content": (
                "Answer the user's question using ONLY the information in the "
                "context below. If the answer is not in the context, say clearly "
                "that you don't have that information — do not guess or use "
                "outside knowledge.\n\n"
                f"Context:\n{source_text}"
            ),
        }
        messages_for_model = [context_message] + messages
    else:
        messages_for_model = messages

    # 3. Get the model's response
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=messages_for_model,
    )
    raw_answer = response.choices[0].message.content

    # 4. Output faithfulness check
    if not source_text.strip():
        return raw_answer  # no KB yet — skip check, don't block the demo

    try:
        faith_result = check_faithfulness(
            answer=raw_answer,
            source_text=source_text,
            api_key=os.getenv("GROQ_API_KEY"),
            question=messages[user_idx]["content"],
        )
    except Exception as e:
        print(f"Faithfulness check error: {e}")
        return raw_answer  # fail-open for now — decide as a team if this should fail-closed instead

    if faith_result.verdict == "OUT OF CONTEXT":
        return "I don't have reliable information on that in my current knowledge base."

    return faith_result.final_answer