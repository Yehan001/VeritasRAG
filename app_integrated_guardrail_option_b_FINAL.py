"""
app_integrated_guardrail_option_b_FINAL.py
==========================================

Integrated Streamlit app:
1) Upload document
2) Ask question
3) Run QUESTION-ONLY input guardrail
4) If the question passes, send the sanitized question into the Final Option B
   RAGAS-style faithfulness filtering pipeline
5) Display guardrail output + retrieved context + raw answer + statement verdicts
   + final cleaned answer + CSV export

Required local files in the same folder:
- input_guardrail_question_only_FIXED.py
- final_option_b_faithfulness_filter_FINAL.py

Run:
    streamlit run app_integrated_guardrail_option_b_FINAL.py
"""

from __future__ import annotations

import html
import io

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# MODIFIED - import the question-only guardrail and the PII masking helper function from the fixed guardrail module.
# Reason:
# InputGuardrail protects the retrieval/LLM pipeline, but the Streamlit UI
# and CSV export can still leak the raw original question if we display/export
# it directly. mask_pii is used here to hide PII before showing/exporting.
from input_guardrail_question_only_FIXED import InputGuardrail, mask_pii

# Option B pipeline:
# FAISS retrieval + answer generation + statement-level faithfulness checking.
from final_option_b_faithfulness_filter_FINAL import OptimizedRAGFaithfulnessChecker

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Integrated Guardrail + Option B Faithfulness Filter",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.2rem;
}
.hero h1 { font-family: 'JetBrains Mono', monospace; color:#e6edf3; font-size:1.7rem; margin:0 0 0.4rem 0; }
.hero p { color:#8b949e; margin:0; }
.tag {
    display:inline-block; background:#1f6feb22; border:1px solid #1f6feb66; color:#58a6ff;
    padding:3px 10px; border-radius:20px; font-size:0.78rem; font-family:'JetBrains Mono', monospace;
    margin-right:6px; margin-top:10px;
}
.section-title {
    font-family:'JetBrains Mono', monospace; color:#58a6ff; font-size:0.82rem; font-weight:600;
    text-transform:uppercase; letter-spacing:0.1em; margin:1rem 0 0.6rem 0;
    padding-bottom:0.4rem; border-bottom:1px solid #30363d;
}
.verdict-pass, .verdict-block, .verdict-warn {
    border-radius:8px; padding:0.9rem 1.2rem; font-family:'JetBrains Mono', monospace;
    font-size:1rem; font-weight:600; margin-bottom:0.7rem;
}
.verdict-pass { background:#0d2b1a; border:1px solid #238636; border-left:4px solid #2ea043; color:#3fb950; }
.verdict-block { background:#2d1117; border:1px solid #f85149; border-left:4px solid #da3633; color:#f85149; }
.verdict-warn { background:#2b1f0a; border:1px solid #d29922; border-left:4px solid #e3b341; color:#e3b341; }
.pill-pass, .pill-block, .pill-warn {
    display:inline-block; padding:3px 11px; border-radius:20px; font-size:0.76rem;
    font-family:'JetBrains Mono', monospace; margin:3px;
}
.pill-pass { background:#0d2b1a; border:1px solid #2ea04366; color:#3fb950; }
.pill-block { background:#2d1117; border:1px solid #f8514966; color:#f85149; }
.pill-warn { background:#2b1f0a; border:1px solid #e3b34166; color:#e3b341; }
.q-box, .answer-box {
    background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:0.8rem 1rem;
    color:#e6edf3; margin:0.35rem 0 0.8rem 0;
}
.q-box { font-family:'JetBrains Mono', monospace; font-size:0.88rem; white-space:pre-wrap; }
.answer-box { border-left:4px solid #1f6feb; line-height:1.65; white-space:pre-wrap; }
.q-label { font-size:0.75rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; font-weight:600; }
.warn-box { background:#2b1f0a; border:1px solid #e3b34166; border-radius:8px; padding:0.55rem 0.8rem; color:#e3b341; margin:3px 0; }
.reason-box { background:#2d1117; border:1px solid #f8514966; border-radius:8px; padding:0.7rem 0.9rem; color:#f85149; margin-top:0.5rem; }
.filter-item { background:#1f2937; border-left:3px solid #58a6ff; padding:4px 10px; border-radius:0 4px 4px 0; color:#79c0ff; font-family:'JetBrains Mono', monospace; font-size:0.8rem; margin:3px 0; }
.html-attack-box { background:#2d1117; border:1px solid #f8514966; border-left:4px solid #f85149; border-radius:8px; padding:0.7rem 1rem; color:#f85149; margin:4px 0; font-family:'JetBrains Mono', monospace; font-size:0.82rem; }
.html-attack-title { color:#f85149; font-family:'JetBrains Mono', monospace; font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.4rem; }
.html-clean-box { background:#0d2b1a; border:1px solid #23863666; border-left:4px solid #2ea043; border-radius:8px; padding:0.7rem 1rem; color:#3fb950; margin:4px 0; font-family:'JetBrains Mono', monospace; font-size:0.82rem; }
.pii-box { background:#2b1f0a; border:1px solid #e3b34166; border-left:4px solid #e3b341; border-radius:8px; padding:0.7rem 1rem; color:#e3b341; margin:4px 0; font-family:'JetBrains Mono', monospace; font-size:0.82rem; }
.pii-title { color:#e3b341; font-family:'JetBrains Mono', monospace; font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.4rem; }
.pii-clean-box { background:#0d2b1a; border:1px solid #23863666; border-left:4px solid #2ea043; border-radius:8px; padding:0.7rem 1rem; color:#3fb950; margin:4px 0; font-family:'JetBrains Mono', monospace; font-size:0.82rem; }
.pii-item { background:#2b1a00; border-left:3px solid #e3b341; padding:4px 10px; border-radius:0 4px 4px 0; color:#e3b341; font-family:'JetBrains Mono', monospace; font-size:0.8rem; margin:3px 0; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>⚡ Integrated Guardrail + Option B Faithfulness Filter</h1>
  <p>Question-only input filtering first. Then FAISS retrieval, Groq answer generation, statement-level verification, and final cleaned answer.</p>
  <span class="tag">Question Guardrail</span>
  <span class="tag">FAISS Retrieval</span>
  <span class="tag">70B Answering</span>
  <span class="tag">8B Verification</span>
  <span class="tag">RAGAS-style Faithfulness</span>
</div>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def safe(text: str) -> str:
    return html.escape(text or "")

# MODIFIED -
# Helper functions for UI/export privacy.
# These prevent raw PII from appearing on screen or inside the CSV export.
def mask_question_for_display(text: str) -> str:
    """
    Mask PII before displaying or exporting the original user question.

    Important:
    This is only for UI/export privacy.
    The actual pipeline still uses q_result.sanitized_question after the
    guardrail has processed the question.

    Example:
        "What is the policy for john@gmail.com?"
        -> "What is the policy for [EMAIL]?"
    """
    masked_text, _ = mask_pii(text or "")
    return masked_text

# MODIFIED - helper to check if PII masking was applied based on the filter log, for UI display purposes.
def pii_masking_applied(filter_log: list[str]) -> bool:
    """
    Returns True if PII masking happened inside the guardrail sanitizer.
    """
    return any("pii masked" in item.lower() for item in filter_log)

# MODIFIED - helper to extract and format PII masking log entries for UI display, showing which types of PII were masked.
def pii_masking_log(filter_log: list[str]) -> str:
    """
    Return only privacy-safe PII log entries.

    The filter log contains PII type names, not actual PII values.
    Example:
        "PII masked: email address, phone number"
    """
    return " | ".join(
        item for item in filter_log
        if "pii masked" in item.lower()
    )

def extract_txt(file) -> str:
    return file.read().decode("utf-8", errors="ignore")


def extract_pdf(file) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file.read()))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_docx(file) -> str:
    import docx
    document = docx.Document(io.BytesIO(file.read()))
    return "\n".join(p.text for p in document.paragraphs if p.text.strip())


def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return extract_txt(uploaded_file)
    if name.endswith(".pdf"):
        return extract_pdf(uploaded_file)
    if name.endswith(".docx"):
        return extract_docx(uploaded_file)
    return ""

# MODIFIED - 
# Updated this function so the raw original question is never displayed.
# Instead, the original question is masked again before rendering.
def render_guardrail_result(q_result, original_question: str) -> None:
    """
    Render question guardrail result.

    PII PRIVACY FIX:
    The original question may contain sensitive data.
    Therefore, the UI shows a masked version of the original question instead
    of the raw user input.
    """
    st.markdown('<div class="section-title">1) Question Guardrail Result</div>', unsafe_allow_html=True)

    # The raw original question may contain emails, phone numbers, NICs, cards, addresses, IPs, etc. Do not display it directly.
    # This does not affect pipeline logic. The real pipeline uses
    # q_result.sanitized_question after the guardrail has processed it.
    display_original_question = mask_question_for_display(original_question)

    if q_result.passed:
        if q_result.warned_checks:
            st.markdown(
                f'<div class="verdict-warn">⚠ Question Passed with Warnings &nbsp; [{safe(q_result.risk_level)} RISK]</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="verdict-pass">✓ Question Passed</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="verdict-block">✗ Question Blocked &nbsp; [{safe(q_result.risk_level)} RISK]</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="reason-box">🚫 {safe(q_result.rejection_reason)}</div>', unsafe_allow_html=True)

    pills_html = ""
    for check in q_result.passed_checks:
        pills_html += f'<span class="pill-pass">✓ {safe(check)}</span>'
    for check in q_result.warned_checks:
        pills_html += f'<span class="pill-warn">⚠ {safe(check)}</span>'
    for check in q_result.blocked_checks:
        pills_html += f'<span class="pill-block">✗ {safe(check)}</span>'
    st.markdown(pills_html, unsafe_allow_html=True)

    if q_result.warnings:
        st.markdown("**Warnings:**")
        for warning in q_result.warnings:
            st.markdown(f'<div class="warn-box">⚠️ {safe(warning)}</div>', unsafe_allow_html=True)

    # MODIFIED - 
    # PII PRIVACY FIX:
    # Do not render raw original_question here.
    # Show masked original question instead.
    st.markdown("**Question — before and after sanitization:**")

    st.markdown('<div class="q-label">Original question masked for privacy</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="q-box">{safe(display_original_question)}</div>', 
        unsafe_allow_html=True,
    )

    st.markdown('<div class="q-label">Sanitized question sent to Option B</div>', unsafe_allow_html=True)
    if q_result.passed:
        st.markdown(
            f'<div class="q-box" style="border-color:#1f6feb">{safe(q_result.sanitized_question)}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="q-box" style="border-color:#f85149">Blocked — nothing sent to retrieval/LLM</div>',
            unsafe_allow_html=True,
        )

    # ── HTML Attack Warning Block ─────────────────────────────────────────────
    html_attacks_detected = any(
        "html" in item.lower() or "script" in item.lower()
        for item in q_result.filter_log
    )

    if html_attacks_detected:
        st.markdown(
            '<div class="html-attack-title">⚠️ HTML / Script Attack Detected & Removed</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="html-attack-box">🔴 HTML/script attack content was found in your question and removed before processing.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="html-clean-box">✓ HTML attack content stripped — clean question passed forward</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="html-clean-box">✓ No HTML / script attack content detected</div>',
            unsafe_allow_html=True,
        )

    # MODIFIED - added a new block to check for PII masking in the filter log 
    # and display which types of PII were masked, 
    # without showing any raw PII values in the UI.

    # ── PII Masking Warning Block ────────────────────────────────────────────
    pii_log_items = [item for item in q_result.filter_log if "pii masked" in item.lower()]

    if pii_log_items:
        st.markdown(
            '<div class="pii-title">🟡 Personal Information (PII) Detected & Masked</div>',
            unsafe_allow_html=True,
        )

        # Map each masked PII type to a friendly label and token.
        # These labels must match the labels returned by mask_pii().
        pii_labels = {
            "email address"           : ("📧", "Email address", "[EMAIL]"),
            "phone number (lk)"       : ("📞", "Phone number (LK)", "[PHONE]"),
            "phone number"            : ("📞", "Phone number", "[PHONE]"),
            "nic number"              : ("🪪", "NIC number", "[NIC]"),
            "credit card number"      : ("💳", "Credit card number", "[CARD]"),
            "passport number"         : ("🛂", "Passport number", "[PASSPORT]"),
            "bank account number"     : ("🏦", "Bank account number", "[BANK_ACCOUNT]"),
            "address/location detail" : ("📍", "Address/location", "[ADDRESS]"),
            "ip address"              : ("🌐", "IP address", "[IP_ADDRESS]"),
            "mac address"             : ("🖥️", "MAC address", "[MAC_ADDRESS]"),
        }

        for log_item in pii_log_items:
            # log entry format:
            # "PII masked: email address, phone number"
            after_colon = log_item.split(":", 1)[-1].strip().lower()
            detected_types = [t.strip() for t in after_colon.split(",")]

            for pii_type in detected_types:
                if pii_type in pii_labels:
                    icon, label, token = pii_labels[pii_type]
                    st.markdown(
                        f'<div class="pii-item">{icon} {label} detected → replaced with <b>{token}</b></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="pii-item">⚠️ {safe(pii_type)} detected → masked</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown(
            '<div class="pii-clean-box">✓ PII masked in question — safe version sent to retrieval and LLM</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="pii-clean-box">✓ No personal information (PII) detected</div>',
            unsafe_allow_html=True,
        )

    if q_result.filter_log:
        st.markdown("**Filters applied:**")
        for item in q_result.filter_log:
            # This should contain only filter names/type labels.
            # Do not put raw PII values in filter_log.
            st.markdown(f'<div class="filter-item">→ {safe(item)}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Guardrail")
    strict_mode = st.toggle("Strict mode (warnings become blocks)", value=False)
    allow_non_english = st.toggle("Allow non-English questions", value=False)
    check_spelling = st.toggle("Spelling quality warning", value=True)

    st.markdown("---")
    st.subheader("Option B Models")
    answer_model = st.selectbox(
        "Answer generation model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "llama3-8b-8192",
        ],
        index=0,
    )

    verifier_model = st.selectbox(
        "Statement generation + verification model",
        [
            "llama-3.1-8b-instant",
            "llama3-8b-8192",
            "llama-3.3-70b-versatile",
            "llama3-70b-8192",
        ],
        index=0,
    )

    rebuild_model = st.selectbox(
        "Final rebuild model",
        [
            "llama-3.1-8b-instant",
            "llama3-8b-8192",
            "llama-3.3-70b-versatile",
            "llama3-70b-8192",
        ],
        index=0,
    )

    st.markdown("---")
    st.subheader("Retrieval")
    chunk_size = st.slider("Chunk size", 300, 1200, 450, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 40, 10)
    top_k = st.slider("Top-k retrieved chunks", 1, 10, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────────────────────

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-title">Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload document", type=["txt", "pdf", "docx"])

    source_text = ""
    if uploaded_file:
        source_text = extract_text(uploaded_file)
        st.caption(f"Extracted {len(source_text):,} characters from **{uploaded_file.name}**")
        with st.expander("Document preview"):
            st.text(source_text[:1500] + ("..." if len(source_text) > 1500 else ""))

with right:
    st.markdown('<div class="section-title">Question</div>', unsafe_allow_html=True)
    question = st.text_area(
        "Ask a question from the document",
        height=110,
        placeholder="Example: What is solar energy?",
        label_visibility="collapsed",
    )
    run = st.button("Run Integrated Pipeline", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

if run:
    if not uploaded_file:
        st.error("Please upload a document.")
        st.stop()

    if not source_text.strip():
        st.error("Could not extract readable text from this document.")
        st.stop()

    if not question.strip():
        st.error("Please enter a question.")
        st.stop()

    # MODIFIED 
    # Create a masked version of the original question for UI and CSV only.
    # - This is NOT sent to the LLM.
    # - This is only used to avoid leaking raw PII on screen or in downloads.
    # - The actual pipeline uses q_result.sanitized_question after guardrail.

    display_question = mask_question_for_display(question)

    # ======================= GUARDRAIL STAGE =======================
    # The guardrail checks the raw user question, masks PII, removes unsafe
    # input patterns, and decides whether the question can continue.
    # ===============================================================

    guardrail = InputGuardrail(
        strict_mode=strict_mode,
        allow_non_english=allow_non_english,
        check_spelling=check_spelling,
    )

    with st.spinner("Running question guardrail..."):
        q_result = guardrail.check_question(question)

    render_guardrail_result(q_result, question)

    # ======================= BLOCK UNSAFE QUESTIONS =======================
    # If the guardrail blocks the question, stop here.
    # Nothing is sent to FAISS retrieval or to the LLM.
    # ====================================================================

    if not q_result.passed:
        st.error("⛔ Pipeline stopped. The blocked question was not sent to FAISS retrieval or the LLM.")
        st.stop()

    # MODIFIED 
    # Security/privacy critical:
    # Only the sanitized question from the guardrail is sent to Option B.
    # Do NOT use:
    #     question
    #
    # Use only:
    #     q_result.sanitized_question
    #
    # This prevents raw emails, phone numbers, NICs, cards, addresses,
    # IP addresses, and MAC addresses from reaching FAISS/retrieval/LLM.

    sanitized_question = q_result.sanitized_question

    st.markdown('<div class="section-title">2) Option B Faithfulness Pipeline Output</div>', unsafe_allow_html=True)

    try:
        # ======================= OPTION B PIPELINE =======================
        # This stage performs:
        # 1. FAISS retrieval using the sanitized question
        # 2. LLM answer generation
        # 3. Statement-level faithfulness verification
        # 4. Final answer rebuilding if unsupported statements are found
        # ================================================================
        with st.spinner("Running FAISS retrieval + answer generation + statement verification..."):
            checker = OptimizedRAGFaithfulnessChecker(
                api_key=None,
                answer_model=answer_model,
                verifier_model=verifier_model,
                rebuild_model=rebuild_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=top_k,
            )
            # ======================= THIS IS CHANGED / CONFIRMED =======================
            # The checker receives only sanitized_question.
            # This is the correct privacy-safe flow.
            # ========================================================================

            result = checker.check(
                question=sanitized_question,
                source_text=source_text,
                use_cache=True,
            )

        st.success("Integrated pipeline completed.")

        # ======================= METRICS DISPLAY =======================
        # Show summary values from the faithfulness checker.
        # ===============================================================

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Faithfulness", result.faithfulness_score)
        c2.metric("Verdict", result.verdict)
        c3.metric("Grounded", len(result.grounded_statements))
        c4.metric("Hallucinated", len(result.hallucinated_statements))
        c5.metric("LLM Calls", result.estimated_llm_calls)
        c6.metric("Chunks", result.chunks_indexed)

    # MODIFIED 
    # Do not show the raw original question in the caption.
    # Use display_question where PII is already masked.
    
        st.caption(
            f"Original user question masked for privacy: {display_question} | "
            f"Guardrail sanitized question: {sanitized_question} | "
            f"Option B normalized question: {result.normalized_question} | "
            f"Answer model: {result.answer_model} | "
            f"Verifier model: {result.verifier_model} | "
            f"Rebuild model: {result.rebuild_model}"
        )

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Final Answer",
            "Raw Answer",
            "Statement Verdicts",
            "Retrieved Context",
            "Export",
        ])

        with tab1:
            st.subheader("Final Clean Answer")
            st.markdown(f'<div class="answer-box">{safe(result.final_answer)}</div>', unsafe_allow_html=True)
            if result.was_cleaned:
                st.warning("Unsupported statements were removed and the final answer was rebuilt.")
            else:
                st.success("No unsupported statements were detected. Rebuild was skipped to save cost.")

        with tab2:
            st.subheader("Raw LLM Answer")
            st.markdown(f'<div class="answer-box">{safe(result.raw_answer)}</div>', unsafe_allow_html=True)

        with tab3:
            col_g, col_h = st.columns(2)
            with col_g:
                st.markdown("### verdict = 1 — Grounded")
                if result.grounded_statements:
                    for statement in result.grounded_statements:
                        st.success(statement)
                else:
                    st.info("No grounded statements found.")
            with col_h:
                st.markdown("### verdict = 0 — Hallucinated / Unsupported")
                if result.hallucinated_statements:
                    for statement in result.hallucinated_statements:
                        st.error(statement)
                else:
                    st.info("No hallucinated statements found.")

        with tab4:
            st.subheader("Retrieved Context Used for Verification")
            st.text_area("Context", result.retrieved_context, height=360, label_visibility="collapsed")

        with tab5:
            export_df = pd.DataFrame([{
                # MODIFIED - 
                # for privacy, export the masked version of the original question 
                # instead of the raw user input.
                "original_user_question_masked": display_question,
                "guardrail_sanitized_question": sanitized_question,
                "pii_masking_applied": pii_masking_applied(q_result.filter_log),
                "pii_masking_log": pii_masking_log(q_result.filter_log),

                "guardrail_passed": q_result.passed,
                "guardrail_risk_level": q_result.risk_level,
                "guardrail_blocked_checks": " | ".join(q_result.blocked_checks),
                "guardrail_warned_checks": " | ".join(q_result.warned_checks),
                "guardrail_warnings": " | ".join(q_result.warnings),
                
                # Option B receives sanitized_question, so this should not contain raw PII.
                # contain raw PII if the guardrail worked correctly.
                "option_b_original_question": result.original_question,
                "option_b_normalized_question": result.normalized_question,
                
                "faithfulness_score": result.faithfulness_score,
                "verdict": result.verdict,
                "was_cleaned": result.was_cleaned,
                "llm_calls": result.estimated_llm_calls,
                "chunks_indexed": result.chunks_indexed,
                "answer_model": result.answer_model,
                "verifier_model": result.verifier_model,
                "rebuild_model": result.rebuild_model,
                "grounded_statements": " | ".join(result.grounded_statements),
                "hallucinated_statements": " | ".join(result.hallucinated_statements),
                "raw_answer": result.raw_answer,
                "final_answer": result.final_answer,
                "retrieved_context": result.retrieved_context,
            }])

            st.dataframe(export_df, use_container_width=True)
            st.download_button(
                "Download Integrated Results CSV",
                data=export_df.to_csv(index=False),
                file_name="integrated_guardrail_option_b_results.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Check that GROQ_API_KEY is available in your .env file and all required packages are installed.")