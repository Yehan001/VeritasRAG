import json

import streamlit as st

from guardrail import GuardrailSettings, InputGuardrail

st.set_page_config(page_title="Input Filtering Guardrail", layout="wide")
st.title("Input Filtering Guardrail")
st.caption("Safety-only input filtering: rules + 3 models + short-circuit blocking")

with st.sidebar:
    st.header("Settings")
    enable_hf = st.toggle("Enable Hugging Face models", value=True)
    enable_semantic = st.toggle("Enable SentenceTransformer matcher", value=True)
    use_presidio = st.toggle("Use Presidio PII if installed", value=True)
    enable_audit = st.toggle("Enable audit log", value=True)

    with st.expander("Advanced thresholds"):
        prompt_threshold = st.slider("Prompt-injection threshold", 0.30, 0.99, 0.75, 0.01)
        moderation_threshold = st.slider("Moderation threshold", 0.30, 0.99, 0.65, 0.01)
        semantic_threshold = st.slider("SentenceTransformer threshold", 0.30, 0.95, 0.62, 0.01)

    st.info("Short-circuit is always ON: if one stage blocks, later stages are skipped.")


@st.cache_resource(show_spinner=False)
def build_guardrail(enable_hf, enable_semantic, use_presidio, enable_audit, prompt_threshold, moderation_threshold, semantic_threshold):
    settings = GuardrailSettings(
        enable_hf_models=enable_hf,
        enable_sentence_transformer=enable_semantic,
        use_presidio=use_presidio,
        enable_audit_log=enable_audit,
        prompt_injection_threshold=prompt_threshold,
        moderation_threshold=moderation_threshold,
        semantic_threshold=semantic_threshold,
    )
    return InputGuardrail(settings)


guardrail = build_guardrail(enable_hf, enable_semantic, use_presidio, enable_audit, prompt_threshold, moderation_threshold, semantic_threshold)

st.subheader("1. Enter user input")
question = st.text_area("User question", height=150, placeholder="Type a user question here...")

if st.button("Run input filter", type="primary"):
    result = guardrail.check(question)
    if result.decision == "BLOCKED":
        st.error(f"{result.decision} — {result.risk.upper()}")
    elif result.decision == "PASSED_WITH_WARNING":
        st.warning(f"{result.decision} — {result.risk.upper()}")
    else:
        st.success(f"{result.decision} — {result.risk.upper()}")

    st.write(f"**Reason:** {result.reason}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Safety label", result.safety_label)
    c2.metric("Triggered stage", result.triggered_stage)
    c3.metric("Backend", result.safety_backend)
    c4.metric("Skipped stages", len(result.skipped_stages))

    st.subheader("Sanitized / masked input")
    st.code(result.sanitized_input or "", language="text")

    st.subheader("Checked stages")
    rows = []
    for s in result.checked_stages:
        rows.append({
            "stage": s.stage,
            "action": s.action,
            "label": s.label,
            "risk": s.risk,
            "backend": s.backend,
            "score": round(float(s.score), 4),
            "reason": s.reason,
        })
    st.dataframe(rows, use_container_width=True)

    if result.skipped_stages:
        st.subheader("Skipped stages due to short-circuit")
        st.write(result.skipped_stages)

    if result.pii.has_pii:
        st.subheader("PII entities")
        st.json(result.pii.to_dict())

    with st.expander("Full JSON output"):
        st.json(result.to_dict())
