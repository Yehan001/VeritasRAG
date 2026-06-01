"""
app_integrated_guardrail_option_b_FINAL_PATCHED.py
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
- input_guardrail_question_only_FINAL_PATCHED.py
- final_option_b_faithfulness_filter_FINAL_PATCHED.py

Run:
    streamlit run app_integrated_guardrail_option_b_FINAL_PATCHED.py
"""

from __future__ import annotations

import html
import io
import re
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from input_guardrail_question_only_FINAL_ANYDOC import InputGuardrail
from final_option_b_faithfulness_filter_FINAL_ANYDOC import OptimizedRAGFaithfulnessChecker

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MAX Guardrail + Evidence-Aware Faithfulness Filter",
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
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>⚡ MAX Guardrail + Evidence-Aware Faithfulness Filter</h1>
  <p>Maximum practical question-only input filtering first. Then FAISS retrieval, Groq answer generation, statement-level verification, and final cleaned answer.</p>
  <span class="tag">Question Guardrail</span>
  <span class="tag">FAISS Retrieval</span>
  <span class="tag">70B Answering</span>
  <span class="tag">8B Verification</span>
  <span class="tag">Evidence-Aware Verification</span>
</div>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def safe(text: str) -> str:
    return html.escape(text or "")


# ─────────────────────────────────────────────────────────────────────────────
# Output-side PII masking patterns
# ─────────────────────────────────────────────────────────────────────────────
_OUTPUT_PII_PATTERNS = [
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b(?:0\d{9}|\+94\d{9}|\+?\d[\d\s\-]{8,}\d)\b"), "[PHONE_OR_NUMBER]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b\d{9}[vVxX]\b|\b\d{12}\b"), "[NIC_OR_ID]"),
    (re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"), "[CARD]"),
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), "[IBAN]"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP_ADDRESS]"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Output-side sanitization
# ─────────────────────────────────────────────────────────────────────────────
# Even after faithfulness filtering, text can originate from an uploaded document.
# Therefore, final answers and raw answers are sanitized before UI display/export.

_DANGEROUS_OUTPUT_PATTERNS = [
    re.compile(r"<\s*script[\s\S]*?>[\s\S]*?<\s*/\s*script\s*>", re.IGNORECASE),
    re.compile(r"<\s*iframe[\s\S]*?>[\s\S]*?<\s*/\s*iframe\s*>", re.IGNORECASE),
    re.compile(r"<\s*style[\s\S]*?>[\s\S]*?<\s*/\s*style\s*>", re.IGNORECASE),
    re.compile(r"<\s*svg[\s\S]*?>[\s\S]*?<\s*/\s*svg\s*>", re.IGNORECASE),
    re.compile(r"<\s*meta[^>]*>", re.IGNORECASE),
    re.compile(r"<\s*form[\s\S]*?>[\s\S]*?<\s*/\s*form\s*>", re.IGNORECASE),
    re.compile(r"\bon\w+\s*=\s*[\"']?[^\"'>\s]*[\"']?", re.IGNORECASE),
    re.compile(r"javascript\s*:", re.IGNORECASE),
    re.compile(r"data\s*:\s*text/html", re.IGNORECASE),
    re.compile(r"expression\s*\(", re.IGNORECASE),
]


def sanitize_output_text(text: str) -> tuple[str, list[str]]:
    """Remove dangerous output-side HTML/JS/markdown before rendering/exporting."""
    cleaned = "" if text is None else str(text)
    changes = []

    # Decode HTML entities repeatedly so encoded attacks become visible.
    for _ in range(5):
        decoded = html.unescape(cleaned)
        if decoded == cleaned:
            break
        cleaned = decoded
        changes.append("Decoded HTML entities in output")

    before = cleaned
    # Remove dangerous markdown links BEFORE removing javascript:, otherwise
    # [Click](javascript:alert(1)) can become [Click](alert(1)).
    # These patterns consume the whole link target up to the end of the markdown link/line
    # so nested payloads such as alert(1) do not leave "[Click](alert(1))" behind.
    cleaned = re.sub(
        r"\[([^\]\n]{0,120})\]\(\s*(?:javascript\s*:|data\s*:\s*text/html|vbscript\s*:)[^\n]*\)",
        r"\1 [unsafe link removed]",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Remove any remaining markdown link whose target still contains script-like payloads.
    cleaned = re.sub(
        r"\[([^\]\n]{0,120})\]\([^\n]*(?:alert\s*\(|script|onerror|onload|document\.|eval\s*\(|prompt\s*\(|confirm\s*\()[^\n]*\)",
        r"\1 [unsafe link removed]",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Fallback: if script residue remains inside markdown link parentheses, remove the target.
    cleaned = re.sub(
        r"\[([^\]\n]{0,120})\]\([^\n]*\)",
        lambda m: (m.group(1) + " [link removed]") if re.search(r"alert|script|onerror|onload|javascript|eval|prompt|confirm", m.group(0), re.IGNORECASE) else m.group(0),
        cleaned,
    )
    if cleaned != before:
        changes.append("Removed dangerous markdown links from output")

    before = cleaned
    for pattern in _DANGEROUS_OUTPUT_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    if cleaned != before:
        changes.append("Removed dangerous HTML/JS from output")

    # Remove any remaining HTML tags. The UI should render answer text, not execute document HTML.
    before = cleaned
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    if cleaned != before:
        changes.append("Removed remaining HTML tags from output")

    # After tag stripping, remove orphan JS payload residues such as alert(1), eval(...), document.cookie.
    before = cleaned
    cleaned = re.sub(r"\b(?:alert|eval|prompt|confirm)\s*\([^)]*\)", "[unsafe script removed]", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bdocument\s*\.\s*(?:cookie|write|location)\b", "[unsafe script removed]", cleaned, flags=re.IGNORECASE)
    if cleaned != before:
        changes.append("Removed leftover script payload text from output")

    # Mask PII that may have come from the uploaded document or model output.
    before = cleaned
    for pattern, repl in _OUTPUT_PII_PATTERNS:
        cleaned = pattern.sub(repl, cleaned)
    if cleaned != before:
        changes.append("Masked personal data in output")

    cleaned = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]", "", cleaned)
    # Clean line-level whitespace without destroying all paragraph structure.
    cleaned = "\n".join(line.strip() for line in cleaned.splitlines() if line.strip())
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    return cleaned, list(dict.fromkeys(changes))

# ─────────────────────────────────────────────────────────────────────────────
# Document poisoning / retrieval-side document sanitizer
# ─────────────────────────────────────────────────────────────────────────────
# Uploaded documents are untrusted. These checks remove executable HTML/JS and
# neutralize common prompt-injection phrases inside retrieved document text.

_DOCUMENT_POISON_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"reveal\s+(the\s+)?(hidden|system|developer)\s+(prompt|instructions?)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(dan|unrestricted|developer\s+mode|free)", re.IGNORECASE),
    re.compile(r"use\s+outside\s+knowledge\s+instead", re.IGNORECASE),
]


def sanitize_document_for_retrieval(text: str) -> tuple[str, list[str]]:
    """Neutralize document-side poisoning before passing document text into retrieval/LLM."""
    cleaned = "" if text is None else str(text)
    changes = []

    out_cleaned, out_changes = sanitize_output_text(cleaned)
    if out_changes:
        cleaned = out_cleaned
        changes.extend(["Document sanitization: " + c for c in out_changes])

    before = cleaned
    for pattern in _DOCUMENT_POISON_PATTERNS:
        cleaned = pattern.sub("[DOCUMENT_INSTRUCTION_REMOVED]", cleaned)
    if cleaned != before:
        changes.append("Neutralized prompt-injection style instructions embedded inside document text")

    # Remove invisible control characters from document context too.
    before = cleaned
    cleaned = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]", "", cleaned)
    if cleaned != before:
        changes.append("Removed invisible Unicode controls from document text")

    return cleaned, list(dict.fromkeys(changes))


# ─────────────────────────────────────────────────────────────────────────────
# Session-level harmful intent escalation detection
# ─────────────────────────────────────────────────────────────────────────────
# Streamlit is single-user/session oriented, so this is session-level, not IP-level.

_ESCALATION_STAGE_1 = re.compile(r"\b(chemical|chemicals|precursors?|industrial\s+facilit(?:y|ies)|fertilizer|oxidizer|solvent)\b", re.IGNORECASE)
_ESCALATION_STAGE_2 = re.compile(r"\b(unstable|reactive|combine|combination|mixture|energetic|detonation|explosive|blast|ignite)\b", re.IGNORECASE)


def check_session_intent_escalation(question: str) -> tuple[bool, str]:
    q = question or ""
    history = st.session_state.setdefault("intent_risk_history", [])
    stage = 0
    if _ESCALATION_STAGE_1.search(q):
        stage = max(stage, 1)
    if _ESCALATION_STAGE_2.search(q):
        stage = max(stage, 2)
    if stage:
        history.append({"question": q[:120], "stage": stage, "time": time.time()})
        st.session_state["intent_risk_history"] = history[-8:]

    recent_stages = [h.get("stage", 0) for h in st.session_state.get("intent_risk_history", []) if time.time() - h.get("time", 0) < 900]
    if len(recent_stages) >= 2 and max(recent_stages) >= 2 and min(recent_stages) >= 1:
        return True, "Multi-turn harmful intent escalation detected across recent questions."
    return False, ""


def abuse_locked() -> tuple[bool, int]:
    """Simple Streamlit session-level lockout for repeated blocked questions."""
    lock_until = st.session_state.get("blocked_until", 0)
    now = time.time()
    if lock_until > now:
        return True, int(lock_until - now)
    return False, 0


def record_blocked_attempt() -> None:
    st.session_state["blocked_attempts"] = st.session_state.get("blocked_attempts", 0) + 1
    if st.session_state["blocked_attempts"] >= 5:
        st.session_state["blocked_until"] = time.time() + 120


def record_successful_attempt() -> None:
    st.session_state["blocked_attempts"] = 0


def write_audit_log(event_type: str, question: str, details: dict) -> None:
    """Local JSONL audit log for blocked/warned attempts. Do not store raw PII."""
    import json, datetime
    safe_question = re.sub(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", "[EMAIL]", question or "")
    safe_question = re.sub(r"\b(?:0\d{9}|\+94\d{9}|\+?\d[\d\s\-]{8,}\d)\b", "[PHONE_OR_NUMBER]", safe_question)
    record = {
        "time_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "question_preview": safe_question[:300],
        "details": details,
    }
    try:
        with open("guardrail_audit_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass
    st.session_state["blocked_until"] = 0


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


def render_guardrail_result(q_result, original_question: str) -> None:
    st.markdown('<div class="section-title">1) Question Guardrail Result</div>', unsafe_allow_html=True)

    risk_cls = q_result.risk_level.lower()
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

    st.markdown("**Question — before and after sanitization:**")
    st.markdown('<div class="q-label">Original</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="q-box">{safe(original_question)}</div>', unsafe_allow_html=True)

    st.markdown('<div class="q-label">Sanitized question sent to Option B</div>', unsafe_allow_html=True)
    if q_result.passed:
        st.markdown(f'<div class="q-box" style="border-color:#1f6feb">{safe(q_result.sanitized_question)}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="q-box" style="border-color:#f85149">Blocked — nothing sent to retrieval/LLM</div>', unsafe_allow_html=True)

    if q_result.filter_log:
        st.markdown("**Filters applied:**")
        for item in q_result.filter_log:
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
    semantic_safety = st.toggle("Optional LLM semantic safety classifier", value=False)

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
    retrieval_threshold = st.slider("Retrieval distance threshold", 0.50, 2.50, 1.35, 0.05)
    st.caption("Lower threshold = stricter document-boundary enforcement. Calibrate with your dataset.")


# ─────────────────────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────────────────────

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-title">Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload document", type=["txt", "pdf", "docx"])

    source_text = ""
    if uploaded_file:
        raw_source_text = extract_text(uploaded_file)
        source_text, doc_sanitize_log = sanitize_document_for_retrieval(raw_source_text)
        st.caption(f"Extracted {len(raw_source_text):,} characters from **{uploaded_file.name}**; sanitized retrieval text has {len(source_text):,} characters")
        if doc_sanitize_log:
            with st.expander("Document safety preprocessing applied"):
                for item in doc_sanitize_log:
                    st.warning(item)
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
    locked, seconds_left = abuse_locked()
    if locked:
        st.error(f"Too many blocked attempts. Please wait {seconds_left} seconds before trying again.")
        st.stop()

    if not uploaded_file:
        st.error("Please upload a document.")
        st.stop()

    if not source_text.strip():
        st.error("Could not extract readable text from this document.")
        st.stop()

    if not question.strip():
        st.error("Please enter a question.")
        st.stop()

    escalated, escalation_reason = check_session_intent_escalation(question)
    if escalated:
        record_blocked_attempt()
        write_audit_log(
            "blocked_session_escalation",
            question,
            {"reason": escalation_reason},
        )
        st.error(f"⛔ {escalation_reason}")
        st.stop()

    guardrail = InputGuardrail(
        strict_mode=strict_mode,
        allow_non_english=allow_non_english,
        check_spelling=check_spelling,
        semantic_safety=semantic_safety,
    )

    with st.spinner("Running question guardrail..."):
        q_result = guardrail.check_question(question)

    render_guardrail_result(q_result, question)

    if not q_result.passed:
        record_blocked_attempt()
        write_audit_log("blocked_question", question, {"blocked_checks": q_result.blocked_checks, "risk": q_result.risk_level, "reason": q_result.rejection_reason})
        st.error("⛔ Pipeline stopped. The blocked question was not sent to FAISS retrieval or the LLM.")
        remaining = max(0, 5 - st.session_state.get("blocked_attempts", 0))
        if remaining:
            st.warning(f"Repeated blocked attempts will temporarily lock this session. Attempts before lock: {remaining}")
        else:
            st.warning("Session temporarily locked for 2 minutes due to repeated blocked attempts.")
        st.stop()

    record_successful_attempt()
    if q_result.warned_checks:
        write_audit_log("warned_question", question, {"warned_checks": q_result.warned_checks, "risk": q_result.risk_level})

    sanitized_question = q_result.sanitized_question

    st.markdown('<div class="section-title">2) Option B Faithfulness Pipeline Output</div>', unsafe_allow_html=True)

    try:
        with st.spinner("Running FAISS retrieval + answer generation + statement verification..."):
            checker = OptimizedRAGFaithfulnessChecker(
                api_key=None,
                answer_model=answer_model,
                verifier_model=verifier_model,
                rebuild_model=rebuild_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=top_k,
                retrieval_distance_threshold=retrieval_threshold,
                enable_query_translation=True,
                enable_retrieval_spell_fix=True,
            )

            result = checker.check(
                question=sanitized_question,
                source_text=source_text,
                use_cache=True,
            )

        safe_final_answer, final_output_changes = sanitize_output_text(result.final_answer)
        safe_raw_answer, raw_output_changes = sanitize_output_text(result.raw_answer)
        output_changes = list(dict.fromkeys(final_output_changes + raw_output_changes))

        st.success("Integrated pipeline completed.")
        if output_changes:
            st.warning("Output-side sanitizer modified the displayed/exported answer for UI safety: " + "; ".join(output_changes))

        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("Faithfulness", result.faithfulness_score)
        c2.metric("Verdict", result.verdict)
        c3.metric("Grounded", len(result.grounded_statements))
        c4.metric("Hallucinated", len(result.hallucinated_statements))
        c5.metric("LLM Calls", result.estimated_llm_calls)
        c6.metric("Chunks", result.chunks_indexed)

        if hasattr(result, "second_pass_used") or hasattr(result, "retrieval_best_distance"):
            st.caption(
                f"Retrieval confidence: {getattr(result, 'retrieval_confidence', 0):.3f} | "
                f"Best distance: {getattr(result, 'retrieval_best_distance', None)} | "
                f"Context sufficient: {getattr(result, 'context_sufficient', True)} | "
                f"Second-pass verification: {getattr(result, 'second_pass_used', False)}"
            )
        c7.metric("Retrieval Conf.", getattr(result, "retrieval_confidence", 0.0))

        if not getattr(result, "context_sufficient", True):
            st.error("Document-boundary enforcement stopped generation: " + getattr(result, "refusal_reason", "retrieved context was insufficient."))
            write_audit_log("context_insufficient", sanitized_question, {"retrieval_confidence": getattr(result, "retrieval_confidence", 0.0), "reason": getattr(result, "refusal_reason", "")})

        st.caption(
            f"Original user question: {question} | "
            f"Guardrail sanitized question: {sanitized_question} | "
            f"Option B normalized/retrieval question: {result.normalized_question} | "
            f"Retrieval corrections: {', '.join(getattr(result, 'retrieval_corrections', [])) or 'none'} | "
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
            st.markdown(f'<div class="answer-box">{safe(safe_final_answer)}</div>', unsafe_allow_html=True)
            if result.was_cleaned:
                st.warning("Unsupported statements were removed and the final answer was rebuilt.")
            else:
                st.success("No unsupported statements were detected. Rebuild was skipped to save cost.")

        with tab2:
            st.subheader("Raw LLM Answer")
            st.markdown(f'<div class="answer-box">{safe(safe_raw_answer)}</div>', unsafe_allow_html=True)

        with tab3:
            if getattr(result, "statement_verdicts", None):
                st.subheader("Evidence-aware verifier details")
                st.dataframe(pd.DataFrame(result.statement_verdicts), use_container_width=True)
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
                "original_user_question": question,
                "guardrail_sanitized_question": sanitized_question,
                "guardrail_passed": q_result.passed,
                "guardrail_risk_level": q_result.risk_level,
                "guardrail_blocked_checks": " | ".join(q_result.blocked_checks),
                "guardrail_warned_checks": " | ".join(q_result.warned_checks),
                "guardrail_warnings": " | ".join(q_result.warnings),
                "option_b_original_question": result.original_question,
                "option_b_normalized_question": result.normalized_question,
                "retrieval_corrections": " | ".join(getattr(result, "retrieval_corrections", [])),
                "faithfulness_score": result.faithfulness_score,
                "verdict": result.verdict,
                "was_cleaned": result.was_cleaned,
                "llm_calls": result.estimated_llm_calls,
                "chunks_indexed": result.chunks_indexed,
                "retrieval_confidence": getattr(result, "retrieval_confidence", 0.0),
                "context_sufficient": getattr(result, "context_sufficient", True),
                "refusal_reason": getattr(result, "refusal_reason", ""),
                "answer_model": result.answer_model,
                "verifier_model": result.verifier_model,
                "rebuild_model": result.rebuild_model,
                "grounded_statements": " | ".join(result.grounded_statements),
                "hallucinated_statements": " | ".join(result.hallucinated_statements),
                "raw_answer": safe_raw_answer,
                "final_answer": safe_final_answer,
                "output_sanitizer_changes": " | ".join(output_changes),
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
