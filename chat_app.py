import streamlit as st
from openai import OpenAI

from guardrail.engine import InputGuardrail

st.set_page_config(page_title="AI Chatbot", layout="wide")
st.title("OpenRouter Chatbot")
st.caption("Chat with the OpenRouter-backed model through a Streamlit interface.")

client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1",
)

# Guardrail instance used to check and sanitize user input before sending to the model
_guardrail = InputGuardrail()

if "messages" not in st.session_state:
    st.session_state.messages = []


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

    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as exc:
        return f"Error: {exc}"


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        reply = get_ai_response(st.session_state.messages)

    with st.chat_message("assistant"):
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
