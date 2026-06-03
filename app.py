"""
app.py
======

Streamlit UI for the hybrid domain-independent input filtering system.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from input_guardrail import GuardrailConfig, InputGuardrail, check_escalation_with_history
from normalization import safe_display_text


st.set_page_config(page_title="Hybrid Input Filtering", layout="wide")

st.title("Hybrid Input Filtering / Guardrail Layer")
st.caption("Domain-independent input preprocessing. No answer generation. Rules + classifier + sanitization.")

if "blocked_attempts" not in st.session_state:
    st.session_state.blocked_attempts = 0
if "blocked_until" not in st.session_state:
    st.session_state.blocked_until = 0.0
if "intent_history" not in st.session_state:
    st.session_state.intent_history = []
if "audit_events" not in st.session_state:
    st.session_state.audit_events = []


def audit(event_type: str, question: str, result: dict):
    row = {
        "time_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "question_preview": (question or "")[:200],
        "details": result,
    }
    st.session_state.audit_events.append(row)
    try:
        with open("input_filter_audit_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def maybe_train_sklearn():
    model_path = Path("models/sklearn_safety_classifier.joblib")
    if model_path.exists():
        return True, "Existing sklearn classifier loaded."
    return False, (
        "Local sklearn classifier is not trained yet. Run: "
        "python train_sklearn_classifier.py --data data/guardrail_training_template.csv "
        "--output models/sklearn_safety_classifier.joblib"
    )


def load_classifier(mode: str):
    if mode == "None":
        return None

    if mode == "Local sklearn classifier (train first)":
        ok, msg = maybe_train_sklearn()
        if ok:
            st.sidebar.success(msg)
            from semantic_classifier import SklearnSafetyClassifier
            return SklearnSafetyClassifier()
        st.sidebar.warning(msg)
        return None

    if mode == "Groq semantic classifier":
        from semantic_classifier import GroqSemanticSafetyClassifier
        return GroqSemanticSafetyClassifier()

    if mode == "Local HuggingFace classifier":
        from semantic_classifier import HFIntentClassifier
        return HFIntentClassifier("models/hf_intent_classifier")

    return None


with st.sidebar:
    st.header("Settings")

    strict_mode = st.toggle("Strict mode: warnings become blocks", value=False)
    block_urls = st.toggle("Block URLs", value=True)
    enable_classifier = st.toggle("Enable classifier layer", value=True)
    enable_escalation = st.toggle("Multi-turn escalation tracking", value=True)
    enable_throttle = st.toggle("Abuse throttle / temporary lock", value=True)

    st.divider()
    st.subheader("Classifier Layer")
    classifier_mode = st.selectbox(
        "Classifier",
        ["Local sklearn classifier (train first)", "Groq semantic classifier", "Local HuggingFace classifier", "None"],
        index=0,
    )
    block_threshold = st.slider("Classifier block threshold", 0.20, 0.95, 0.45, 0.05)
    warn_threshold = st.slider("Critical-label warning threshold", 0.10, 0.90, 0.25, 0.05)

    st.caption("Use Local sklearn classifier (train first) for free local semantic filtering.")

    if st.button("Retrain sklearn classifier"):
        try:
            from train_sklearn_classifier import train
            train("data/guardrail_training_template.csv", "models/sklearn_safety_classifier.joblib")
            st.success("Sklearn classifier retrained.")
        except Exception as exc:
            st.error(f"Training failed: {exc}")

    st.divider()
    if st.button("Reset session"):
        st.session_state.blocked_attempts = 0
        st.session_state.blocked_until = 0.0
        st.session_state.intent_history = []
        st.session_state.audit_events = []
        st.success("Session reset.")


def build_guardrail():
    config = GuardrailConfig()
    config.strict_mode = strict_mode
    config.block_urls = block_urls
    config.use_classifier = enable_classifier and classifier_mode != "None"
    config.classifier_block_threshold = block_threshold
    config.classifier_warn_threshold = warn_threshold

    classifier = None
    if config.use_classifier:
        try:
            classifier = load_classifier(classifier_mode)
        except Exception as exc:
            st.warning(f"Classifier unavailable, using deterministic rules only. Details: {exc}")
            classifier = None
            config.use_classifier = False

    return InputGuardrail(classifier=classifier, config=config)


def is_locked():
    now = time.time()
    if st.session_state.blocked_until > now:
        return True, int(st.session_state.blocked_until - now)
    return False, 0


st.subheader("Single Input Test")
user_input = st.text_area(
    "Enter any user input",
    height=130,
    placeholder="Type any input to check whether it is passed, warned, masked, sanitized, or blocked.",
)

col1, col2 = st.columns(2)
run = col1.button("Run Input Filter", type="primary", use_container_width=True)
clear = col2.button("Clear", use_container_width=True)

if clear:
    st.rerun()

if run:
    locked, left = is_locked()
    if locked:
        st.error(f"Session temporarily locked due to repeated blocked attempts. Try again in {left} seconds.")
        st.stop()

    if enable_escalation:
        escalated, reason, new_history = check_escalation_with_history(user_input, st.session_state.intent_history)
        st.session_state.intent_history = new_history
        if escalated:
            result_dict = {
                "passed": False,
                "decision": "BLOCKED",
                "risk_level": "HIGH",
                "original_input": user_input,
                "sanitized_input": "",
                "blocked_checks": ["multi_turn_escalation"],
                "reason": reason,
                "action_taken": "Input stopped due to session-level harmful escalation.",
            }
            st.error("BLOCKED")
            st.json(result_dict)
            audit("blocked_multi_turn_escalation", user_input, result_dict)
            st.stop()

    guardrail = build_guardrail()
    result = guardrail.check(user_input)

    if enable_throttle:
        if not result.passed:
            st.session_state.blocked_attempts += 1
            if st.session_state.blocked_attempts >= 5:
                st.session_state.blocked_until = time.time() + 120
        else:
            st.session_state.blocked_attempts = 0

    audit("blocked" if not result.passed else result.decision.lower(), user_input, result.to_dict())

    if result.decision == "BLOCKED":
        st.error("BLOCKED")
    elif result.decision == "PASSED_WITH_WARNING":
        st.warning("PASSED WITH WARNING")
    else:
        st.success("PASSED")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Decision", result.decision)
    m2.metric("Risk Level", result.risk_level)
    m3.metric("Passed", str(result.passed))
    m4.metric("Classifier", result.classifier_label or "N/A")

    st.subheader("Before / After")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Original Input**")
        st.code(result.original_input or "", language="text")
    with c2:
        st.markdown("**Sanitized Input**")
        st.code(result.sanitized_input or "(blocked / empty)", language="text")

    st.subheader("Reason and Action")
    st.write("**Reason:**", result.reason)
    st.write("**Action Taken:**", result.action_taken)

    st.subheader("Triggered Filter Types")
    t1, t2, t3, t4 = st.columns(4)
    t1.write("**Blocked**")
    t1.write(result.blocked_checks or "None")
    t2.write("**Warnings**")
    t2.write(result.warning_checks or "None")
    t3.write("**Sanitized**")
    t3.write(result.sanitized_checks or "None")
    t4.write("**Masked**")
    t4.write(result.masked_checks or "None")

    st.subheader("Detailed Event Log")
    if result.events:
        st.dataframe(pd.DataFrame([e.__dict__ for e in result.events]), use_container_width=True)
    else:
        st.info("No events.")

    st.subheader("Safe UI Display Preview")
    st.code(safe_display_text(result.sanitized_input or result.original_input), language="html")

    st.subheader("JSON Output")
    st.code(result.to_json(), language="json")


st.divider()
st.subheader("Batch CSV Test")
uploaded = st.file_uploader("Upload CSV with a column named text", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    if "text" not in df.columns:
        st.error("CSV must contain a column named text.")
    else:
        if st.button("Run Batch Test"):
            guardrail = build_guardrail()
            rows = []
            for txt in df["text"].astype(str).tolist():
                res = guardrail.check(txt)
                rows.append({
                    "text": txt,
                    "decision": res.decision,
                    "risk_level": res.risk_level,
                    "passed": res.passed,
                    "classifier_label": res.classifier_label,
                    "classifier_confidence": res.classifier_confidence,
                    "sanitized_input": res.sanitized_input,
                    "blocked_checks": " | ".join(res.blocked_checks),
                    "warning_checks": " | ".join(res.warning_checks),
                    "sanitized_checks": " | ".join(res.sanitized_checks),
                    "masked_checks": " | ".join(res.masked_checks),
                    "reason": res.reason,
                })
            out = pd.DataFrame(rows)
            st.dataframe(out, use_container_width=True)
            st.download_button(
                "Download Batch Results CSV",
                out.to_csv(index=False),
                file_name="input_filter_batch_results.csv",
                mime="text/csv",
            )

st.divider()
st.subheader("Session Audit Events")
if st.session_state.audit_events:
    audit_df = pd.DataFrame(st.session_state.audit_events)
    st.dataframe(audit_df, use_container_width=True)
    st.download_button(
        "Download Session Audit CSV",
        audit_df.to_csv(index=False),
        file_name="session_audit.csv",
        mime="text/csv",
    )
else:
    st.info("No audit events yet.")
