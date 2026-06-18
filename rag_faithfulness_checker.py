"""
rag_faithfulness_checker.py
============================
RAG Faithfulness Filter — Clean Callable API
=============================================

This module exposes a single callable function:

    check_faithfulness(answer, source_text, api_key, **kwargs)

Given an answer and the original source document text, it returns
a structured result indicating whether the answer is grounded in
the source, partially grounded, or entirely out of context.

Verification is performed using:
  1. LLM self-report detection  — instant phrase check
  2. Cosine similarity          — deterministic, no LLM needed for clear cases
  3. LLM tiebreaker             — only for borderline cases (score 0.40–0.59)

Designed to be plugged directly into any chatbot pipeline.

Author  : RAG Faithfulness Filter Team
Version : 2.0.0
"""

import os
import re
import json
import time
import logging
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# SECURITY AUDIT LOGGER
# ─────────────────────────────────────────────────────────────────────────────

SECURITY_LOG = logging.getLogger("rag_security_audit")
if not SECURITY_LOG.handlers:
    _handler = logging.FileHandler("security_incidents.log")
    _handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    SECURITY_LOG.addHandler(_handler)
    SECURITY_LOG.setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_MODEL         = "llama-3.3-70b-versatile"
DEFAULT_CHUNK_SIZE    = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K         = 5
SIMILARITY_THRESHOLD  = 0.60   # cosine score >= this → grounded immediately
BORDERLINE_LOW        = 0.40   # cosine score <  this → out of context immediately
GROUNDING_THRESHOLD   = 0.35   # min cosine similarity for question↔document


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT-AWARE SECURITY CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Keywords that indicate the source document is a legitimate cybersecurity
# resource (e.g. a penetration-testing guide, CVE report, OWASP document).
# At least SECURITY_DOC_MIN_HITS must be present in the document for it to
# be treated as a valid security resource.
SECURITY_DOC_KEYWORDS = [
    "security",
    "vulnerability",
    "penetration testing",
    "cybersecurity",
    "threat model",
    "attack surface",
    "defense mechanism",
    "compliance",
    "audit",
    "secure coding",
    "owasp",
    "cwe",
    "cvss",
]
SECURITY_DOC_MIN_HITS = 3   # how many of the above must appear in the document

# Patterns that flag a question or answer as hacking-related.
# These are checked against the question AND the answer.
HACKING_PATTERNS = [
    r"sql\s+injection",
    r"cross.?site\s+scripting",
    r"xss\s+attack",
    r"buffer\s+overflow",
    r"privilege\s+escalation",
    r"brute\s+force",
    r"denial\s+of\s+service",
    r"\bddos\b",
    r"\bransomware\b",
    r"\bkeylogger\b",
    r"\bbackdoor\b",
    r"\bexploit\b",
    r"zero.?day",
    r"\bmalware\b",
    r"\bphishing\b",
    r"\bpayload\b",
    r"\bshellcode\b",
    r"reverse\s+shell",
    r"remote\s+code\s+execution",
    r"\brce\b",
    r"code\s+injection",
    r"path\s+traversal",
    r"directory\s+traversal",
]


# ═══════════════════════════════════════════════════════════════════════════════
# PHRASES THE LLM USES TO SELF-REPORT MISSING INFORMATION
# ═══════════════════════════════════════════════════════════════════════════════

OUT_OF_CONTEXT_PHRASES = [
    "not in the provided context",
    "outside the provided context",
    "not found in the context",
    "not mentioned in the context",
    "not covered in the context",
    "not available in the context",
    "context does not contain",
    "context does not mention",
    "context does not provide",
    "context does not include",
    "no information in the context",
    "cannot be found in the context",
    "is not provided in the context",
    "does not appear in the context",
]


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT DATACLASS — what check_faithfulness() returns
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FaithfulnessResult:
    """
    Returned by check_faithfulness().

    Fields
    ------
    verdict : str
        "FULLY GROUNDED"     — every claim in the answer is in the source.
        "PARTIALLY GROUNDED" — some claims are grounded, some are not.
        "OUT OF CONTEXT"     — no claims could be verified against the source.

    is_grounded : bool
        True only when verdict == "FULLY GROUNDED".

    grounded_ratio : float
        Fraction of claims that passed verification. 0.0 – 1.0.

    grounded_claims : list[str]
        Claims (atomic facts) verified as present in the source text.

    hallucinated_claims : list[str]
        Claims that could not be verified — removed from the final answer.

    final_answer : str
        Cleaned answer containing only grounded claims.
        Identical to the input answer when verdict == "FULLY GROUNDED".

    raw_answer : str
        The original answer passed in (unchanged).

    retrieved_context : str
        The document chunks retrieved and used for verification.
    """
    verdict:             str
    is_grounded:         bool
    grounded_ratio:      float
    grounded_claims:     list[str]
    hallucinated_claims: list[str]
    final_answer:        str
    raw_answer:          str
    retrieved_context:   str

    def __str__(self):
        lines = [
            "",
            "=" * 60,
            f"  VERDICT          : {self.verdict}",
            f"  Grounded         : {self.is_grounded}",
            f"  Grounded ratio   : {self.grounded_ratio:.0%}",
            "-" * 60,
            "  GROUNDED CLAIMS:",
        ]
        for i, c in enumerate(self.grounded_claims, 1):
            lines.append(f"    {i}. {c}")
        if self.hallucinated_claims:
            lines.append("  OUT-OF-CONTEXT CLAIMS (removed):")
            for i, c in enumerate(self.hallucinated_claims, 1):
                lines.append(f"    {i}. {c}")
        lines.append(f"\n  FINAL ANSWER:\n    {self.final_answer.strip()}")
        lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

_DECOMPOSE_PROMPT = """You are a text analysis tool.

Split the answer below into atomic claims. Each claim = one single fact.

Answer:
{answer}

RULES:
- Each claim must be one complete sentence expressing exactly one fact.
- Copy wording from the answer directly — do not rephrase or summarise.
- Do NOT add any information not present in the answer.
- Do NOT include meta-sentences like "Here is the answer" or "Based on the context".
- DO include sentences that say information is not in the context — keep them as-is.

Output: a JSON array of strings only. No explanation, no markdown, no extra text.
Example: ["Solar panels convert sunlight into electricity.", "This information is not in the provided context."]"""


_VERIFY_PROMPT = """You are a fact-checker. Decide if the CLAIM is supported by the CONTEXT.

Context:
{context}

Claim:
{claim}

RULES:
- Reply YES if the claim is directly supported, even with slightly different wording.
- Reply NO if the claim contains any fact NOT in the context.
- Reply NO if the claim contradicts the context.
- Do NOT use outside knowledge.

Your answer (one word only — YES or NO):"""


_REBUILD_PROMPT = """You are a helpful assistant. Write a clear, fluent answer using ONLY the facts below.

Question / original request:
{question}

Verified grounded facts:
{facts}

RULES:
- Use ONLY the listed facts. Do not add new information.
- Write in natural prose.
- If the list is empty, say: "I cannot provide a reliable answer based on the provided context."

Answer:"""


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT-AWARE SECURITY HELPERS  (new — used only inside check_faithfulness)
# ═══════════════════════════════════════════════════════════════════════════════

def _has_hacking_content(text: str) -> bool:
    """
    Return True if *text* matches any hacking-related pattern.
    Checked against both the user question and the LLM answer.
    """
    lower = text.lower()
    return any(re.search(p, lower) for p in HACKING_PATTERNS)


def _is_valid_security_document(source_text: str) -> bool:
    """
    Return True when the uploaded document is a legitimate cybersecurity
    resource — identified by the presence of at least SECURITY_DOC_MIN_HITS
    security-domain keywords in its text.

    A document that passes this test is considered a valid context for
    discussing hacking-related topics (e.g. OWASP guide, CVE report,
    penetration-testing manual).
    """
    lower = source_text.lower()
    hits  = sum(1 for kw in SECURITY_DOC_KEYWORDS if kw in lower)
    return hits >= SECURITY_DOC_MIN_HITS


def _is_question_grounded_in_document(
    question:    str,
    source_text: str,
    embeddings:  HuggingFaceEmbeddings,
) -> bool:
    """
    Return True when the user's question is semantically related to the
    source document (cosine similarity > GROUNDING_THRESHOLD).

    This prevents an attacker from uploading an unrelated document and then
    asking hacking questions that have nothing to do with it.

    Falls back to keyword-overlap if the embedding call fails.
    """
    try:
        q_vec   = embeddings.embed_query(question)
        doc_vec = embeddings.embed_query(source_text[:3000])   # first 3 k chars
        return _cosine(q_vec, doc_vec) > GROUNDING_THRESHOLD
    except Exception:
        # Keyword-overlap fallback
        q_words  = set(question.lower().split())
        d_words  = set(source_text.lower().split())
        overlap  = len(q_words & d_words) / max(len(q_words), 1)
        return overlap > 0.30


def _run_context_aware_security_check(
    question:    str,
    answer:      str,
    source_text: str,
    embeddings:  HuggingFaceEmbeddings,
    user_id:     str,
) -> None:
    """
    Context-aware security gate — called once before faithfulness checking.

    Decision logic
    ──────────────
    1.  If neither the question nor the answer contains hacking-related
        keywords  →  nothing to do, return immediately.

    2.  If hacking keywords ARE present, allow the call to proceed ONLY when
        BOTH of the following are true:
          a. The source document is a valid cybersecurity resource
             (_is_valid_security_document returns True).
          b. The user's question is semantically grounded in that document
             (_is_question_grounded_in_document returns True).

    3.  If either condition fails, log a security incident and raise
        ValueError to block the call.

    Parameters
    ──────────
    question    : effective question (or answer used as fallback)
    answer      : LLM-generated answer being checked
    source_text : full document text
    embeddings  : already-initialised HuggingFaceEmbeddings instance
    user_id     : identifier for audit logging
    """

    # Step 1 — quick exit if no hacking content detected
    if not (_has_hacking_content(question) or _has_hacking_content(answer)):
        return

    # Step 2 — hacking content found; evaluate legitimacy
    is_security_doc    = _is_valid_security_document(source_text)
    is_grounded_in_doc = _is_question_grounded_in_document(
        question, source_text, embeddings
    )

    if is_security_doc and is_grounded_in_doc:
        # Legitimate security discussion — log for audit and allow
        SECURITY_LOG.info(
            "SECURITY_DISCUSSION_ALLOWED | user=%s | "
            "is_security_doc=True | is_grounded=True | "
            "question_preview=%r",
            user_id, question[:120],
        )
        return

    # Step 3 — block and log
    SECURITY_LOG.warning(
        "MALICIOUS_HACKING_QUERY_BLOCKED | user=%s | "
        "is_security_doc=%s | is_grounded=%s | "
        "question_preview=%r",
        user_id,
        is_security_doc,
        is_grounded_in_doc,
        question[:120],
    )
    raise ValueError(
        "Security policy violation: hacking-related content detected outside "
        "a legitimate cybersecurity document context.\n"
        "This request has been blocked and logged.\n\n"
        "To discuss security topics, upload a valid cybersecurity resource "
        "(e.g. an OWASP guide, CVE report, or penetration-testing manual) "
        "and ensure your question is grounded in that document."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS  (unchanged from v1)
# ═══════════════════════════════════════════════════════════════════════════════

def _cosine(a: list, b: list) -> float:
    """Cosine similarity between two embedding vectors."""
    a, b  = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _is_self_reported_ooc(claim: str) -> bool:
    """True if the LLM flagged this claim as out of context."""
    lower = claim.lower()
    return any(p in lower for p in OUT_OF_CONTEXT_PHRASES)


def _chat(client: Groq, model: str, prompt: str) -> str:
    """Send a prompt to Groq and return the text response."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def _decompose(client: Groq, model: str, answer: str) -> list[str]:
    """Break an answer into atomic claims using the LLM."""
    raw = _chat(client, model, _DECOMPOSE_PROMPT.format(answer=answer))
    raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`").strip()
    try:
        claims = json.loads(raw)
        if isinstance(claims, list):
            return [str(c).strip() for c in claims if str(c).strip()]
    except json.JSONDecodeError:
        pass
    # Fallback: split on newlines or numbered list patterns
    parts = re.split(r"\n+|\d+\.\s+", raw)
    return [p.strip().strip('"').strip("'") for p in parts if p.strip()]


def _verify_claim(
    client:      Groq,
    model:       str,
    embeddings:  HuggingFaceEmbeddings,
    context:     str,
    context_vec: list,
    claim:       str,
) -> bool:
    """
    Two-stage verification:
      Stage 1 — Cosine similarity (deterministic, fast, no LLM call).
                >= 0.60 → grounded immediately.
                <  0.40 → out of context immediately.
      Stage 2 — LLM YES/NO tiebreaker only for borderline (0.40 – 0.59).
    """
    claim_vec = embeddings.embed_query(claim)
    score     = _cosine(claim_vec, context_vec)

    if score >= SIMILARITY_THRESHOLD:
        return True    # clear pass — no LLM needed

    if score < BORDERLINE_LOW:
        return False   # clear fail — no LLM needed

    # Borderline: send to LLM for final YES/NO decision
    reply = _chat(client, model, _VERIFY_PROMPT.format(
        context=context, claim=claim
    )).upper().strip()
    return reply.startswith("YES")


def _rebuild(client: Groq, model: str, question: str, grounded: list[str]) -> str:
    """Rebuild a fluent answer from grounded claims only."""
    if not grounded:
        return "I cannot provide a reliable answer based on the provided context."
    facts = "\n".join(f"- {c}" for c in grounded)
    return _chat(client, model, _REBUILD_PROMPT.format(
        question=question, facts=facts
    ))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PUBLIC FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def check_faithfulness(
    answer:        str,
    source_text:   str,
    api_key:       str = "",
    question:      str = "",
    user_id:       str = "anonymous",   # NEW — used for security audit logging
    model:         str = DEFAULT_MODEL,
    chunk_size:    int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    top_k:         int = DEFAULT_TOP_K,
) -> FaithfulnessResult:
    """
    Check whether an answer is faithful to a source document.

    Parameters
    ----------
    answer : str
        The answer text to verify (e.g. what a chatbot returned).

    source_text : str
        The full source document text to verify the answer against.

    api_key : str, optional
        Groq API key. If not passed, reads from GROQ_API_KEY env var.

    question : str, optional
        The original question that produced the answer.
        Used for context retrieval and rebuilding a cleaned answer.
        If omitted, the answer itself is used as a fallback.

    user_id : str, optional
        User identifier used exclusively for security audit logging.
        Default: "anonymous".

    model : str, optional
        Groq model name. Default: "llama-3.3-70b-versatile".

    chunk_size : int, optional
        Character size of each document chunk. Default: 500.

    chunk_overlap : int, optional
        Overlap between consecutive chunks. Default: 50.

    top_k : int, optional
        Number of document chunks to retrieve for context. Default: 5.

    Returns
    -------
    FaithfulnessResult
        A dataclass containing:
          - verdict           : FULLY GROUNDED / PARTIALLY GROUNDED / OUT OF CONTEXT
          - is_grounded       : True / False
          - grounded_ratio    : 0.0 – 1.0
          - grounded_claims   : list of verified facts
          - hallucinated_claims : list of removed facts
          - final_answer      : cleaned answer with only grounded claims
          - raw_answer        : original input answer unchanged
          - retrieved_context : document chunks used for verification

    Raises
    ------
    ValueError
        If api_key is missing, source_text is empty, answer is empty,
        or a security policy violation is detected (hacking-related content
        outside a valid cybersecurity document context).

    Security behaviour
    ------------------
    This function includes a context-aware security check that runs after
    the embeddings model is initialised but before faithfulness verification.

    The check works in three steps:

      1. Scan the question and answer for hacking-related keywords/patterns.
         If none are found, the check passes immediately with no overhead.

      2. If hacking content is detected, the source document is examined:
           - _is_valid_security_document()  — requires 3+ security keywords
             (e.g. "owasp", "vulnerability", "penetration testing") to be
             present in the document.
           - _is_question_grounded_in_document()  — requires the question's
             embedding to be semantically close (cosine > 0.35) to the
             document, preventing unrelated documents being used as cover.

      3. If BOTH conditions pass  →  legitimate security discussion, allowed.
         If EITHER condition fails →  incident logged to security_incidents.log
                                      and ValueError raised to block the call.

    Examples
    --------
    Basic usage (unchanged):

        from rag_faithfulness_checker import check_faithfulness

        result = check_faithfulness(
            answer      = "Solar energy is generated by PV panels.",
            source_text = open("my_document.txt").read(),
            api_key     = "gsk_...",
            question    = "What is solar energy?",
        )

        print(result.verdict)          # "FULLY GROUNDED"
        print(result.is_grounded)      # True
        print(result.grounded_ratio)   # 1.0
        print(result.final_answer)     # cleaned answer

    Chatbot integration (unchanged):

        result = check_faithfulness(
            answer      = chatbot_response,
            source_text = document_text,
            question    = user_question,
        )

        if result.is_grounded:
            send_to_user(result.final_answer)
        else:
            send_to_user(
                result.final_answer + "\\n\\n"
                f"Note: {len(result.hallucinated_claims)} claim(s) "
                "could not be verified against the source document."
            )

    Security-sensitive usage:

        # This will be BLOCKED — hacking keywords present, document is not a
        # recognised cybersecurity resource.
        result = check_faithfulness(
            answer      = "SQL injection exploits parameterized query flaws.",
            source_text = open("annual_report.txt").read(),   # not a security doc
            question    = "How does SQL injection work?",
            user_id     = "user_42",
        )
        # → raises ValueError("Security policy violation ...")

        # This will be ALLOWED — document is an OWASP guide, question is
        # semantically grounded in it.
        result = check_faithfulness(
            answer      = "SQL injection exploits unsanitised user input.",
            source_text = open("owasp_top10.txt").read(),    # valid security doc
            question    = "How does SQL injection work?",
            user_id     = "researcher_01",
        )
        # → proceeds normally, returns FaithfulnessResult
    """

    # ── Validate inputs ───────────────────────────────────────────────────────
    key = api_key or os.getenv("GROQ_API_KEY", "")
    if not key:
        raise ValueError(
            "Groq API key is required.\n"
            "Pass api_key='gsk_...' or set GROQ_API_KEY in your .env file."
        )
    if not source_text.strip():
        raise ValueError("source_text must not be empty.")
    if not answer.strip():
        raise ValueError("answer must not be empty.")

    # Use answer as fallback if no question provided
    effective_question = question.strip() if question.strip() else answer.strip()

    # ── Initialise models ─────────────────────────────────────────────────────
    groq_client = Groq(api_key=key)
    embeddings  = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ── Context-aware security check (NEW) ───────────────────────────────────
    # Placed here so embeddings are ready for semantic similarity.
    # Raises ValueError and logs if a hacking query is detected outside
    # a valid cybersecurity document context.
    _run_context_aware_security_check(
        question    = effective_question,
        answer      = answer,
        source_text = source_text,
        embeddings  = embeddings,
        user_id     = user_id,
    )

    # ── Build FAISS index from source document ────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(source_text)
    if not chunks:
        raise ValueError("Could not extract any text chunks from source_text.")

    vectorstore = FAISS.from_texts(chunks, embeddings)

    # ── Retrieve most relevant chunks ─────────────────────────────────────────
    docs    = vectorstore.similarity_search(effective_question, k=top_k)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Pre-compute context embedding — reused for every claim check
    context_vec = embeddings.embed_query(context)

    # ── Decompose answer into atomic claims ───────────────────────────────────
    claims = _decompose(groq_client, model, answer)

    # ── Verify each claim ─────────────────────────────────────────────────────
    grounded:       list[str] = []
    out_of_context: list[str] = []

    for claim in claims:
        if _is_self_reported_ooc(claim):
            # LLM already admitted this is not in the context
            out_of_context.append(claim)
        elif _verify_claim(groq_client, model, embeddings, context, context_vec, claim):
            grounded.append(claim)
        else:
            out_of_context.append(claim)

    # ── Compute verdict ───────────────────────────────────────────────────────
    total = len(claims)
    ratio = len(grounded) / total if total > 0 else 0.0

    if ratio == 1.0:
        verdict     = "FULLY GROUNDED"
        is_grounded = True
    elif ratio == 0.0:
        verdict     = "OUT OF CONTEXT"
        is_grounded = False
    else:
        verdict     = "PARTIALLY GROUNDED"
        is_grounded = False

    # ── Build final answer ────────────────────────────────────────────────────
    if verdict == "FULLY GROUNDED":
        final_answer = answer        # nothing was removed — return as-is
    else:
        final_answer = _rebuild(groq_client, model, effective_question, grounded)

    # ── Return structured result ──────────────────────────────────────────────
    return FaithfulnessResult(
        verdict=verdict,
        is_grounded=is_grounded,
        grounded_ratio=ratio,
        grounded_claims=grounded,
        hallucinated_claims=out_of_context,
        final_answer=final_answer,
        raw_answer=answer,
        retrieved_context=context,
    )