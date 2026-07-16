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
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_MODEL         = "llama-3.3-70b-versatile"
DEFAULT_CHUNK_SIZE    = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K         = 5
SIMILARITY_THRESHOLD  = 0.60   # cosine score >= this → grounded immediately
BORDERLINE_LOW        = 0.40   # cosine score <  this → out of context immediately


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
# INTERNAL HELPERS
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
        If api_key is missing, source_text is empty, or answer is empty.

    Examples
    --------
    Basic usage:

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

    Chatbot integration:

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