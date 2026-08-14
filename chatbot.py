import os
import re
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

_CHITCHAT_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|good morning|good afternoon|good evening|"
    r"thanks|thank you|bye|goodbye|how are you|what's up)\s*[!.?]*\s*$",
    re.I,
)


def _is_chitchat(text: str) -> bool:
    return bool(_CHITCHAT_PATTERNS.match(text.strip()))


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
    is_chitchat = _is_chitchat(messages[user_idx]["content"])

    # 2. Inject KB content ONLY for real questions — chit-chat gets no
    #    restrictive system prompt, so the model can respond naturally.
    if source_text.strip() and not is_chitchat:
        context_message = {
            "role": "system",
            "content": (
                "Answer the user's question using the information in the context "
                "below. You may combine or reasonably infer from multiple facts "
                "stated in the context — you don't need an exact literal match. "
                "Only say you don't have the information if the context genuinely "
                "doesn't address the topic at all. Do not use outside knowledge "
                "beyond what's in the context.\n\n"
                f"Context:\n{source_text}"
            ),
        }
        messages_for_model = [context_message] + messages
    else:
        messages_for_model = messages

    # 3. Get the model's response
    print("Sending to LLM:", messages_for_model)
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=messages_for_model,
    )
    raw_answer = response.choices[0].message.content

    # 4. Output faithfulness check — skip for chit-chat/greetings and when there's no KB yet
    if not source_text.strip() or is_chitchat:
        return raw_answer

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