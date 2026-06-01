"""
final_option_b_faithfulness_filter_FINAL_PATCHED.py
===========================================

Audited robust RAGAS-style faithfulness filtering backend for Document QA.

Improvements over the ENTERPRISE version:
- Multi-query retrieval for better evidence recall.
- Retrieval confidence + LLM context sufficiency gate before answer generation.
- Evidence-constrained answer generation.
- Atomic statement generation.
- Evidence-aware statement verification with reason, evidence quote, and confidence.
- Conservative verification: unsupported / partial / uncertain / contradictory = hallucinated.
- Optional second-pass verification using a stronger model for borderline cases.
- Evidence-quote validation: verifier must provide evidence that appears in retrieved context.
- Final answer is rebuilt only from grounded statements.
- If no grounded statements remain, the system refuses instead of guessing.

Important:
This is still not 100% hallucination-proof. It substantially reduces risk but remains
limited by retrieval quality, document extraction quality, and LLM verifier reliability.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


@dataclass
class StatementVerdict:
    statement: str
    verdict: int
    reason: str = ""
    evidence_quote: str = ""
    confidence: float = 0.0
    label: str = "unsupported"


@dataclass
class FaithfulnessResult:
    original_question: str
    normalized_question: str
    retrieved_context: str
    raw_answer: str
    final_answer: str
    grounded_statements: List[str]
    hallucinated_statements: List[str]
    faithfulness_score: float
    verdict: str
    was_cleaned: bool
    chunks_indexed: int
    estimated_llm_calls: int
    answer_model: str
    verifier_model: str
    rebuild_model: str
    retrieval_confidence: float = 0.0
    context_sufficient: bool = True
    context_sufficiency_reason: str = ""
    refusal_reason: str = ""
    statement_verdicts: List[Dict[str, Any]] = field(default_factory=list)
    second_pass_used: bool = False
    retrieval_best_distance: Optional[float] = None

    def __str__(self) -> str:
        sep = "=" * 70
        dash = "-" * 70
        lines = [
            "",
            sep,
            f"  VERDICT                : {self.verdict}",
            f"  Faithfulness score     : {self.faithfulness_score:.2f}",
            f"  Context sufficient     : {self.context_sufficient}",
            f"  Retrieval confidence   : {self.retrieval_confidence:.3f}",
            f"  Grounded statements    : {len(self.grounded_statements)}",
            f"  Hallucinated/unsupported: {len(self.hallucinated_statements)}",
            f"  Answer rebuilt         : {'Yes' if self.was_cleaned else 'No'}",
            f"  Second pass used       : {self.second_pass_used}",
            f"  LLM calls made         : {self.estimated_llm_calls}",
            f"  Chunks indexed         : {self.chunks_indexed}",
            dash,
            f"  Original question      : {self.original_question}",
            f"  Normalized question    : {self.normalized_question}",
            dash,
            f"  Refusal reason         : {self.refusal_reason or 'none'}",
            dash,
            "  FINAL ANSWER:",
            f"    {self.final_answer.strip()}",
            sep,
            "",
        ]
        return "\n".join(lines)


class RobustRAGFaithfulnessChecker:
    """
    Robust document-QA faithfulness checker.

    The design is intentionally conservative:
    - If evidence is weak, refuse.
    - If verifier is unsure, mark unsupported.
    - If evidence quote is not found in retrieved context, mark unsupported.
    - If the final grounded set is empty, refuse.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        answer_model: str = "llama-3.3-70b-versatile",
        verifier_model: str = "llama-3.1-8b-instant",
        rebuild_model: str = "llama-3.1-8b-instant",
        second_pass_model: Optional[str] = "llama-3.3-70b-versatile",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 450,
        chunk_overlap: int = 60,
        top_k: int = 5,
        retrieval_distance_threshold: float = 1.35,
        min_retrieval_confidence: float = 0.05,
        require_llm_context_sufficiency: bool = True,
        use_second_pass: bool = True,
        verifier_confidence_threshold: float = 0.70,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY missing. Add it to .env or pass api_key directly.")

        self.client = Groq(api_key=self.api_key)
        self.answer_model = answer_model
        self.verifier_model = verifier_model
        self.rebuild_model = rebuild_model
        self.second_pass_model = second_pass_model or answer_model
        self.top_k = top_k
        self.retrieval_distance_threshold = retrieval_distance_threshold
        self.min_retrieval_confidence = min_retrieval_confidence
        self.require_llm_context_sufficiency = require_llm_context_sufficiency
        self.use_second_pass = use_second_pass
        self.verifier_confidence_threshold = verifier_confidence_threshold

        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vectorstore = None
        self.chunks_indexed = 0
        self.llm_call_count = 0
        self.cache: Dict[str, FaithfulnessResult] = {}

    # ----------------------------- extraction -----------------------------

    @staticmethod
    def extract_text_from_txt_bytes(file_bytes: bytes) -> str:
        return file_bytes.decode("utf-8", errors="ignore")

    @staticmethod
    def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    @staticmethod
    def extract_text_from_docx_bytes(file_bytes: bytes) -> str:
        import docx
        document = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in document.paragraphs if p.text.strip())

    @staticmethod
    def extract_text_from_file_path(file_path: str) -> str:
        with open(file_path, "rb") as f:
            data = f.read()
        lower = file_path.lower()
        if lower.endswith(".txt"):
            return RobustRAGFaithfulnessChecker.extract_text_from_txt_bytes(data)
        if lower.endswith(".pdf"):
            return RobustRAGFaithfulnessChecker.extract_text_from_pdf_bytes(data)
        if lower.endswith(".docx"):
            return RobustRAGFaithfulnessChecker.extract_text_from_docx_bytes(data)
        raise ValueError("Unsupported file type. Use .txt, .pdf, or .docx.")

    # ----------------------------- helpers -----------------------------

    @staticmethod
    def normalize_question(question: str) -> str:
        question = (question or "").strip()
        question = re.sub(r"\s+", " ", question)
        question = re.sub(r"[?!.]{2,}", "?", question)
        question = re.sub(r"[%#@$^&*_=+]{2,}", "", question)
        return question

    @staticmethod
    def _make_cache_key(question: str, source_text: str) -> str:
        source_hash = hashlib.sha256(source_text.encode("utf-8", errors="ignore")).hexdigest()[:16]
        return f"{source_hash}:{question.lower().strip()}"

    @staticmethod
    def _extract_json(text: str) -> Any:
        text = (text or "").strip()
        text = re.sub(r"```json|```", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _normalize_for_quote(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").lower()).strip()

    def _chat(self, prompt: str, model: str, max_tokens: int = 900) -> str:
        self.llm_call_count += 1
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    # ----------------------------- retrieval -----------------------------

    def load_document(self, source_text: str) -> int:
        chunks = self.splitter.split_text(source_text)
        if not chunks:
            raise ValueError("No valid text found in document.")
        self.vectorstore = FAISS.from_texts(chunks, self.embeddings)
        self.chunks_indexed = len(chunks)
        return self.chunks_indexed

    def _generate_retrieval_queries(self, question: str) -> List[str]:
        """Use low-cost LLM query expansion, then fall back to original only."""
        prompt = f"""
Create up to 3 short retrieval search queries for finding evidence in a document.
Rules:
- Preserve the meaning of the user's question.
- Do not add new facts.
- Include the original question as one query if useful.
- Return ONLY a JSON list of strings.

Question: {question}
"""
        try:
            raw = self._chat(prompt, model=self.verifier_model, max_tokens=180)
            parsed = self._extract_json(raw)
            if isinstance(parsed, list):
                queries = []
                for q in parsed:
                    q = str(q).strip()
                    if q and q.lower() not in {x.lower() for x in queries}:
                        queries.append(q)
                if question not in queries:
                    queries.insert(0, question)
                return queries[:4]
        except Exception:
            pass
        return [question]

    def retrieve_context_with_scores(self, question: str) -> Tuple[str, float, Optional[float], List[Tuple[str, float]]]:
        if self.vectorstore is None:
            raise RuntimeError("Document not loaded. Call load_document() first.")

        queries = self._generate_retrieval_queries(question)
        scored: List[Tuple[str, float]] = []
        seen = set()

        for q in queries:
            try:
                pairs = self.vectorstore.similarity_search_with_score(q, k=self.top_k)
            except Exception:
                docs = self.vectorstore.similarity_search(q, k=self.top_k)
                pairs = [(d, self.retrieval_distance_threshold) for d in docs]
            for doc, score in pairs:
                text = doc.page_content.strip()
                key = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
                if text and key not in seen:
                    seen.add(key)
                    scored.append((text, float(score)))

        scored.sort(key=lambda x: x[1])
        scored = scored[: max(self.top_k, 5)]

        if not scored:
            return "", 0.0, None, []

        best_distance = min(score for _, score in scored)
        confidence = max(0.0, min(1.0, 1.0 - (best_distance / max(self.retrieval_distance_threshold * 2, 1e-6))))
        context_parts = []
        for i, (text, score) in enumerate(scored, 1):
            context_parts.append(f"[CHUNK {i} | distance={score:.4f}]\n{text}")
        return "\n\n".join(context_parts), round(confidence, 3), round(best_distance, 4), scored

    def llm_context_sufficiency_check(self, question: str, context: str) -> Dict[str, Any]:
        if not context.strip():
            return {"sufficient": False, "reason": "No retrieved context."}
        prompt = f"""
You are checking whether the retrieved document context is sufficient to answer the user's question.
Use ONLY the context. Do not answer the question.

Return ONLY valid JSON:
{{"sufficient": true/false, "reason": "short reason"}}

Mark sufficient=false if:
- the context is unrelated,
- the context only partially answers the question,
- the answer would require outside knowledge,
- the context is too vague.

Question:
{question}

Retrieved context:
{context}
"""
        raw = self._chat(prompt, model=self.verifier_model, max_tokens=250)
        parsed = self._extract_json(raw)
        if isinstance(parsed, dict) and "sufficient" in parsed:
            return {
                "sufficient": bool(parsed.get("sufficient")),
                "reason": str(parsed.get("reason", ""))[:500],
            }
        return {"sufficient": True, "reason": "Sufficiency parser failed; continuing with conservative downstream verification."}

    # ----------------------------- LLM stages -----------------------------

    def generate_answer(self, question: str, context: str) -> str:
        prompt = f"""
You are a strict document-based QA assistant.

Use ONLY the retrieved context below. Do not use outside knowledge. Do not guess.
If the context does not contain the answer, say exactly:
"This information is not in the provided context."

When giving factual statements, stay close to the wording of the context.
Do not introduce numbers, dates, causes, examples, names, or conclusions unless directly supported.

Retrieved context:
{context}

Question:
{question}

Answer:
"""
        return self._chat(prompt=prompt, model=self.answer_model, max_tokens=700)

    def generate_statements(self, question: str, answer: str) -> List[str]:
        if not answer or "not in the provided context" in answer.lower():
            return []
        prompt = f"""
Break the answer into atomic factual statements.

Rules:
- Each statement must contain exactly one factual claim.
- Do not add information.
- Do not include opinions or filler.
- If the answer contains no factual claims, return [].
- Return ONLY a JSON list of strings.

Question:
{question}

Answer:
{answer}
"""
        raw = self._chat(prompt=prompt, model=self.verifier_model, max_tokens=500)
        parsed = self._extract_json(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]

    def verify_statements_once(self, context: str, statements: List[str], model: str) -> List[StatementVerdict]:
        if not statements:
            return []
        prompt = f"""
You are a strict faithfulness verifier.
Judge each statement using ONLY the retrieved context.

For each statement:
- verdict = 1 ONLY if the statement is fully and directly supported by the context.
- verdict = 0 if unsupported, partially supported, contradicted, too broad, vague, inferred without evidence, or uncertain.
- confidence must be 0.0 to 1.0.
- evidence_quote must be a short exact quote from the context that supports the statement.
- If verdict=0, evidence_quote may be empty.
- label must be one of: supported, unsupported, contradicted, partial, uncertain.

Return ONLY valid JSON list in this exact schema:
[
  {{"statement":"...", "verdict":1, "label":"supported", "confidence":0.95, "evidence_quote":"...", "reason":"..."}}
]

Retrieved context:
{context}

Statements:
{json.dumps(statements, indent=2)}
"""
        raw = self._chat(prompt=prompt, model=model, max_tokens=1200)
        parsed = self._extract_json(raw)
        verdicts: List[StatementVerdict] = []
        if isinstance(parsed, list):
            for i, item in enumerate(parsed):
                if not isinstance(item, dict):
                    continue
                default_statement = statements[min(i, len(statements) - 1)]
                statement = str(item.get("statement", default_statement)).strip()
                try:
                    verdict = 1 if int(item.get("verdict", 0)) == 1 else 0
                except Exception:
                    verdict = 0
                try:
                    confidence = float(item.get("confidence", 0.0))
                except Exception:
                    confidence = 0.0
                verdicts.append(StatementVerdict(
                    statement=statement,
                    verdict=verdict,
                    label=str(item.get("label", "supported" if verdict == 1 else "unsupported")),
                    confidence=max(0.0, min(1.0, confidence)),
                    evidence_quote=str(item.get("evidence_quote", "")).strip(),
                    reason=str(item.get("reason", "")).strip(),
                ))
        if verdicts:
            return verdicts
        return [StatementVerdict(statement=s, verdict=0, label="uncertain", confidence=0.0, reason="Verifier JSON parsing failed.") for s in statements]

    def _evidence_quote_is_valid(self, quote: str, context: str) -> bool:
        quote_norm = self._normalize_for_quote(quote)
        context_norm = self._normalize_for_quote(context)
        if not quote_norm:
            return False
        if quote_norm in context_norm:
            return True
        # Allow a small quote fragment match for long quotes.
        words = quote_norm.split()
        if len(words) >= 6:
            for window in range(min(10, len(words)), 5, -1):
                for i in range(0, len(words) - window + 1):
                    frag = " ".join(words[i:i+window])
                    if frag in context_norm:
                        return True
        return False

    def verify_statements(self, context: str, statements: List[str]) -> Tuple[List[StatementVerdict], bool]:
        first = self.verify_statements_once(context, statements, self.verifier_model)
        second_pass_used = False

        final: List[StatementVerdict] = []
        borderline: List[str] = []
        borderline_indexes: List[int] = []

        for idx, v in enumerate(first):
            # Evidence gate: supported statements must include evidence appearing in context.
            if v.verdict == 1:
                if v.confidence < self.verifier_confidence_threshold or not self._evidence_quote_is_valid(v.evidence_quote, context):
                    borderline.append(v.statement)
                    borderline_indexes.append(idx)
                else:
                    final.append(v)
            else:
                final.append(v)

        if self.use_second_pass and borderline:
            second_pass_used = True
            second = self.verify_statements_once(context, borderline, self.second_pass_model)
            second_map = {v.statement.strip().lower(): v for v in second}
            rebuilt: List[StatementVerdict] = []
            b_iter = iter(borderline)
            for v in first:
                needs_second = v.statement in borderline
                if needs_second:
                    key = v.statement.strip().lower()
                    sv = second_map.get(key)
                    if sv is None:
                        sv = StatementVerdict(statement=v.statement, verdict=0, label="uncertain", confidence=0.0, reason="Second-pass verification unavailable.")
                    if sv.verdict == 1 and sv.confidence >= self.verifier_confidence_threshold and self._evidence_quote_is_valid(sv.evidence_quote, context):
                        rebuilt.append(sv)
                    else:
                        sv.verdict = 0
                        sv.label = sv.label if sv.label != "supported" else "uncertain"
                        if not sv.reason:
                            sv.reason = "Statement failed second-pass/evidence validation."
                        rebuilt.append(sv)
                else:
                    rebuilt.append(v)
            final = rebuilt
        else:
            # Convert any remaining borderline supported claims to unsupported.
            rebuilt = []
            for v in first:
                if v.verdict == 1 and (v.confidence < self.verifier_confidence_threshold or not self._evidence_quote_is_valid(v.evidence_quote, context)):
                    v.verdict = 0
                    v.label = "uncertain"
                    v.reason = v.reason or "Supported verdict lacked enough confidence or valid evidence quote."
                rebuilt.append(v)
            final = rebuilt

        return final, second_pass_used

    def rebuild_answer(self, question: str, grounded_statements: List[str]) -> str:
        if not grounded_statements:
            return "This information is not in the provided context."
        prompt = f"""
Rewrite the verified statements into a concise final answer.

Rules:
- Use ONLY the verified statements.
- Do not add new facts.
- Do not use outside knowledge.
- Keep the answer natural and clear.
- If the verified statements do not answer the question, say exactly:
  "This information is not in the provided context."

Question:
{question}

Verified statements:
{json.dumps(grounded_statements, indent=2)}

Final answer:
"""
        return self._chat(prompt=prompt, model=self.rebuild_model, max_tokens=500)

    # ----------------------------- main -----------------------------

    def _refusal_result(
        self,
        original_question: str,
        normalized_question: str,
        context: str,
        refusal: str,
        reason: str,
        retrieval_confidence: float,
        best_distance: Optional[float],
    ) -> FaithfulnessResult:
        return FaithfulnessResult(
            original_question=original_question,
            normalized_question=normalized_question,
            retrieved_context=context,
            raw_answer=refusal,
            final_answer=refusal,
            grounded_statements=[],
            hallucinated_statements=[reason],
            faithfulness_score=0.0,
            verdict="FAIL",
            was_cleaned=False,
            chunks_indexed=self.chunks_indexed,
            estimated_llm_calls=self.llm_call_count,
            answer_model=self.answer_model,
            verifier_model=self.verifier_model,
            rebuild_model=self.rebuild_model,
            retrieval_confidence=retrieval_confidence,
            context_sufficient=False,
            context_sufficiency_reason=reason,
            refusal_reason=reason,
            retrieval_best_distance=best_distance,
        )

    def check(self, question: str, source_text: str, use_cache: bool = True) -> FaithfulnessResult:
        if not question or not question.strip():
            raise ValueError("question must not be empty.")
        if not source_text or not source_text.strip():
            raise ValueError("source_text must not be empty.")

        original_question = question
        normalized_question = self.normalize_question(question)
        cache_key = self._make_cache_key(normalized_question, source_text)
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]

        self.llm_call_count = 0
        self.load_document(source_text)

        context, retrieval_confidence, best_distance, _scored = self.retrieve_context_with_scores(normalized_question)

        if not context.strip() or retrieval_confidence < self.min_retrieval_confidence:
            result = self._refusal_result(
                original_question,
                normalized_question,
                context,
                "This information is not sufficiently supported by the retrieved document context.",
                "Retrieved evidence was too weak or unrelated.",
                retrieval_confidence,
                best_distance,
            )
            if use_cache:
                self.cache[cache_key] = result
            return result

        suff = {"sufficient": True, "reason": "LLM sufficiency check disabled."}
        if self.require_llm_context_sufficiency:
            suff = self.llm_context_sufficiency_check(normalized_question, context)
            if not suff.get("sufficient", True):
                result = self._refusal_result(
                    original_question,
                    normalized_question,
                    context,
                    "This information is not in the provided context.",
                    str(suff.get("reason", "Retrieved context insufficient.")),
                    retrieval_confidence,
                    best_distance,
                )
                if use_cache:
                    self.cache[cache_key] = result
                return result

        raw_answer = self.generate_answer(normalized_question, context)

        # If model correctly refused, do not count it as hallucination.
        if "not in the provided context" in raw_answer.lower():
            result = FaithfulnessResult(
                original_question=original_question,
                normalized_question=normalized_question,
                retrieved_context=context,
                raw_answer=raw_answer,
                final_answer="This information is not in the provided context.",
                grounded_statements=[],
                hallucinated_statements=[],
                faithfulness_score=1.0,
                verdict="REFUSED",
                was_cleaned=False,
                chunks_indexed=self.chunks_indexed,
                estimated_llm_calls=self.llm_call_count,
                answer_model=self.answer_model,
                verifier_model=self.verifier_model,
                rebuild_model=self.rebuild_model,
                retrieval_confidence=retrieval_confidence,
                context_sufficient=True,
                context_sufficiency_reason=str(suff.get("reason", "")),
                retrieval_best_distance=best_distance,
            )
            if use_cache:
                self.cache[cache_key] = result
            return result

        statements = self.generate_statements(normalized_question, raw_answer)
        verdict_objs, second_pass_used = self.verify_statements(context, statements)

        grounded: List[str] = []
        hallucinated: List[str] = []
        for v in verdict_objs:
            if v.verdict == 1:
                grounded.append(v.statement)
            else:
                hallucinated.append(v.statement)

        total = len(grounded) + len(hallucinated)
        score = len(grounded) / total if total else 0.0

        if not grounded:
            final_answer = "This information is not in the provided context."
            was_cleaned = True
            verdict = "FAIL"
        else:
            if hallucinated:
                final_answer = self.rebuild_answer(normalized_question, grounded)
                was_cleaned = True
            else:
                final_answer = raw_answer
                was_cleaned = False
            if score >= 0.90:
                verdict = "PASS"
            elif score >= 0.60:
                verdict = "PARTIAL"
            else:
                verdict = "FAIL"

        result = FaithfulnessResult(
            original_question=original_question,
            normalized_question=normalized_question,
            retrieved_context=context,
            raw_answer=raw_answer,
            final_answer=final_answer,
            grounded_statements=grounded,
            hallucinated_statements=hallucinated,
            faithfulness_score=round(score, 2),
            verdict=verdict,
            was_cleaned=was_cleaned,
            chunks_indexed=self.chunks_indexed,
            estimated_llm_calls=self.llm_call_count,
            answer_model=self.answer_model,
            verifier_model=self.verifier_model,
            rebuild_model=self.rebuild_model,
            retrieval_confidence=retrieval_confidence,
            context_sufficient=True,
            context_sufficiency_reason=str(suff.get("reason", "")),
            statement_verdicts=[v.__dict__ for v in verdict_objs],
            second_pass_used=second_pass_used,
            retrieval_best_distance=best_distance,
        )
        if use_cache:
            self.cache[cache_key] = result
        return result


# Backward-compatible name for existing app code.
OptimizedRAGFaithfulnessChecker = RobustRAGFaithfulnessChecker


def create_checker(
    api_key: Optional[str] = None,
    answer_model: str = "llama-3.3-70b-versatile",
    verifier_model: str = "llama-3.1-8b-instant",
    rebuild_model: str = "llama-3.1-8b-instant",
    second_pass_model: Optional[str] = "llama-3.3-70b-versatile",
    chunk_size: int = 450,
    chunk_overlap: int = 60,
    top_k: int = 5,
    retrieval_distance_threshold: float = 1.35,
    min_retrieval_confidence: float = 0.05,
    require_llm_context_sufficiency: bool = True,
    use_second_pass: bool = True,
) -> RobustRAGFaithfulnessChecker:
    return RobustRAGFaithfulnessChecker(
        api_key=api_key,
        answer_model=answer_model,
        verifier_model=verifier_model,
        rebuild_model=rebuild_model,
        second_pass_model=second_pass_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        retrieval_distance_threshold=retrieval_distance_threshold,
        min_retrieval_confidence=min_retrieval_confidence,
        require_llm_context_sufficiency=require_llm_context_sufficiency,
        use_second_pass=use_second_pass,
    )


def check_faithfulness(
    question: str,
    source_text: str,
    api_key: Optional[str] = None,
    use_cache: bool = False,
    **kwargs,
) -> FaithfulnessResult:
    checker = create_checker(api_key=api_key, **kwargs)
    return checker.check(question=question, source_text=source_text, use_cache=use_cache)



# ─────────────────────────────────────────────────────────────────────────────
# MAX EXTENSIONS — standalone additions
# ─────────────────────────────────────────────────────────────────────────────
# This section intentionally does NOT import older AUDITED/ROBUST files.
# Everything needed is contained in this single module.

import re as _max_re
from typing import Optional as _MaxOptional

_DOMAIN_TERMS = {
    "rag", "ragas", "faiss", "groq", "llama", "llm", "streamlit", "pyspark", "spark",
    "graphframes", "langchain", "huggingface", "transformer", "transformers", "api", "apis",
    "nic", "pdf", "docx", "csv", "sql", "python", "javascript", "html", "css", "json",
    "pv", "photovoltaic", "ceylon", "slt", "veritasrag", "veritas", "kwh", "dbscan",
    "kmeans", "k-means", "ceb", "lirneasia", "airflow", "docker", "postgres", "postgresql",
}

_COMMON_RAG_SPELL_FIXES = {
    "solr": "solar",
    "enrgy": "energy",
    "engy": "energy",
    "whts": "what is",
    "wht": "what",
    "whhat": "what",
    "whats": "what is",
    "wats": "what is",
    "energi": "energy",
    "enery": "energy",
    "photovoltic": "photovoltaic",
    "photovoltiac": "photovoltaic",
    "renwable": "renewable",
    "renweable": "renewable",
    "eletricity": "electricity",
    "electricty": "electricity",
    "advantges": "advantages",
    "benifits": "benefits",
    "turbins": "turbines",
}


def _safe_spell_correct_for_retrieval(text: str) -> str:
    """
    Retrieval-only spelling correction.

    IMPORTANT:
    - This correction is used ONLY for retrieval/verification.
    - The UI still shows the user's original question.
    - Domain terms/acronyms are protected.
    - Known RAG/document-QA misspellings are always applied, even if the character
      difference is large, because phrases like "whats solr engy" must retrieve
      "what is solar energy".
    """
    original = "" if text is None else str(text)

    # Normalize very common shorthand first.
    working = _max_re.sub(r"\bwhats\b", "what is", original, flags=_max_re.IGNORECASE)
    working = _max_re.sub(r"\bwhts\b", "what is", working, flags=_max_re.IGNORECASE)
    working = _max_re.sub(r"\bwht\b", "what", working, flags=_max_re.IGNORECASE)
    working = _max_re.sub(r"\bwhhat\b", "what", working, flags=_max_re.IGNORECASE)

    dictionary_changed = working != original

    def replace_token(match):
        nonlocal dictionary_changed
        token = match.group(0)
        lower = token.lower()
        if lower in _COMMON_RAG_SPELL_FIXES:
            fixed = _COMMON_RAG_SPELL_FIXES[lower]
            dictionary_changed = True
            return fixed.capitalize() if token[:1].isupper() else fixed
        return token

    working = _max_re.sub(r"\b[A-Za-z]{3,}\b", replace_token, working)

    # Optional conservative spellchecker for unknown minor typos.
    try:
        from spellchecker import SpellChecker
        sp = SpellChecker()
        tokens = _max_re.findall(r"\b[A-Za-z][A-Za-z\-]{2,}\b", working)
        replacements = {}
        for tok in tokens:
            low = tok.lower().strip("-")
            if low in _DOMAIN_TERMS:
                continue
            if tok.isupper() or any(ch.isdigit() for ch in tok):
                continue
            if _max_re.search(r"[A-Z][a-z]+[A-Z]", tok):
                continue
            if low not in sp:
                cand = sp.correction(low)
                if cand and cand != low and abs(len(cand) - len(low)) <= 2:
                    replacements[tok] = cand.capitalize() if tok[:1].isupper() else cand
        for old_tok, new_tok in replacements.items():
            working = _max_re.sub(rf"\b{_max_re.escape(old_tok)}\b", new_tok, working)
    except Exception:
        pass

    # If only the generic spellchecker changed a lot, revert those speculative changes.
    # But keep trusted dictionary fixes like solr->solar and engy/enrgy->energy.
    if not dictionary_changed and len(original) >= 5:
        changed_chars = sum(1 for a, b in zip(original, working) if a != b) + abs(len(original) - len(working))
        if changed_chars / max(len(original), 1) > 0.25:
            return original

    working = _max_re.sub(r"\s+", " ", working).strip()
    return working or original




# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT-ADAPTIVE RETRIEVAL SPELL CORRECTION
# ─────────────────────────────────────────────────────────────────────────────
# This is the important general fix: it does NOT depend on knowing the topic.
# For every uploaded document, it builds a vocabulary from that document and
# corrects misspelled query words against the document's own terms before FAISS.

_DOC_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", "where", "who", "whom",
    "whose", "what", "which", "why", "how", "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "can", "could", "should", "would", "will", "may", "might", "must", "shall",
    "to", "of", "in", "on", "at", "by", "for", "from", "with", "without", "about", "into", "over", "under",
    "this", "that", "these", "those", "it", "its", "as", "not", "no", "yes", "explain", "describe", "tell",
    "list", "define", "summarize", "show", "give", "provide", "document", "context", "according", "based"
}

_PROTECTED_QUERY_TERMS = _DOMAIN_TERMS | _DOC_STOPWORDS


def _tokenize_for_vocab(text: str) -> list[str]:
    return [t.lower() for t in _max_re.findall(r"\b[A-Za-z][A-Za-z\-]{2,}\b", text or "")]


def _build_document_vocabulary(source_text: str, max_terms: int = 6000) -> set[str]:
    """Build topic vocabulary dynamically from any uploaded document."""
    from collections import Counter
    tokens = _tokenize_for_vocab(source_text)
    counts = Counter(t for t in tokens if len(t) >= 4 and t not in _DOC_STOPWORDS)
    # Keep frequent and document-specific terms. Singletons are kept too for small docs.
    terms = [t for t, _ in counts.most_common(max_terms)]
    return set(terms)


def _best_doc_term(token: str, doc_vocab: set[str]) -> tuple[str | None, float]:
    """Find closest term from uploaded document vocabulary using RapidFuzz if available."""
    if not doc_vocab:
        return None, 0.0
    token_l = token.lower()
    if token_l in doc_vocab or token_l in _PROTECTED_QUERY_TERMS:
        return None, 0.0

    # Restrict candidates by first letter / nearby length to reduce bad corrections.
    candidates = [
        t for t in doc_vocab
        if abs(len(t) - len(token_l)) <= max(2, int(len(token_l) * 0.45))
        and (t[0] == token_l[0] or len(token_l) <= 4)
    ]
    if not candidates:
        candidates = [t for t in doc_vocab if abs(len(t) - len(token_l)) <= 2]
    if not candidates:
        return None, 0.0

    try:
        from rapidfuzz import process, fuzz
        match = process.extractOne(token_l, candidates, scorer=fuzz.WRatio)
        if not match:
            return None, 0.0
        return str(match[0]), float(match[1])
    except Exception:
        import difflib
        best = max(candidates, key=lambda c: difflib.SequenceMatcher(None, token_l, c).ratio())
        score = difflib.SequenceMatcher(None, token_l, best).ratio() * 100.0
        return best, score


def _looks_like_misspelled_query_token(token: str) -> bool:
    """Avoid rewriting ordinary valid words unless they are likely misspelled."""
    low = token.lower()
    if low in _PROTECTED_QUERY_TERMS:
        return False
    if token.isupper() or any(ch.isdigit() for ch in token):
        return False
    if len(low) < 3:
        return False
    try:
        from spellchecker import SpellChecker
        sp = SpellChecker()
        return low not in sp
    except Exception:
        # Fallback: low vowel ratio often signals typo, but keep this permissive.
        vowels = sum(1 for c in low if c in "aeiou")
        return vowels == 0 or vowels / max(len(low), 1) < 0.25


def _document_adaptive_correct_query(query: str, source_text: str) -> tuple[str, list[str]]:
    """
    Correct query terms using vocabulary extracted from the uploaded document.

    Example for any topic:
      document has: photosynthesis, chlorophyll
      user asks: "what is fotosyntesis and clorofil"
      retrieval query becomes closer to: "what is photosynthesis and chlorophyll"
    """
    doc_vocab = _build_document_vocabulary(source_text)
    changes: list[str] = []

    def repl(match):
        tok = match.group(0)
        low = tok.lower()
        if low in _COMMON_RAG_SPELL_FIXES:
            fixed = _COMMON_RAG_SPELL_FIXES[low]
            if fixed.lower() != low:
                changes.append(f"{tok}→{fixed}")
            return fixed.capitalize() if tok[:1].isupper() else fixed
        if not _looks_like_misspelled_query_token(tok):
            return tok
        best, score = _best_doc_term(tok, doc_vocab)
        if not best:
            return tok
        # Conservative thresholds: enough to fix realistic typos, not rewrite normal words.
        threshold = 72.0 if len(low) <= 5 else 78.0
        if score >= threshold:
            fixed = best.capitalize() if tok[:1].isupper() else best
            if fixed.lower() != low:
                changes.append(f"{tok}→{fixed}")
            return fixed
        return tok

    corrected = _max_re.sub(r"\b[A-Za-z][A-Za-z\-]{2,}\b", repl, query or "")
    corrected = _max_re.sub(r"\s+", " ", corrected).strip()
    return corrected or query, list(dict.fromkeys(changes))


# ─────────────────────────────────────────────────────────────────────────────
# FINAL ANY-DOCUMENT SPELLING PATCH
# ─────────────────────────────────────────────────────────────────────────────
# The uploaded document is unknown-topic. So we correct query tokens against
# vocabulary extracted from that exact document, not a fixed topic dictionary
# and not a generic spellchecker that can change "enegy" -> "enemy".

_QUERY_FUNCTION_FIXES = {
    "whts": "what is", "whats": "what is", "wht": "what", "wat": "what",
    "whhat": "what", "hwat": "what", "whta": "what", "hw": "how",
    "abt": "about", "defintion": "definition", "defn": "definition",
}

_STOP_QUERY_WORDS = {
    "what", "is", "are", "was", "were", "the", "a", "an", "of", "for", "to", "in", "on", "by", "with",
    "how", "why", "when", "where", "who", "which", "does", "do", "did", "can", "could", "would", "should",
    "explain", "describe", "summarize", "list", "give", "show", "tell", "me", "document", "say", "contain"
}


def _token_similarity(a: str, b: str) -> float:
    try:
        from rapidfuzz import fuzz
        return float(max(
            fuzz.ratio(a, b),
            fuzz.partial_ratio(a, b),
            fuzz.token_sort_ratio(a, b),
        ))
    except Exception:
        import difflib
        return difflib.SequenceMatcher(None, a, b).ratio() * 100.0


def _best_doc_term_anydoc(token: str, doc_vocab: set[str]) -> tuple[str, float]:
    low = token.lower()
    if not doc_vocab or low in doc_vocab:
        return low, 100.0

    # Compare mainly with terms close in length to avoid nonsense corrections.
    candidates = [t for t in doc_vocab if abs(len(t) - len(low)) <= max(2, len(low)//3)]
    if not candidates:
        candidates = list(doc_vocab)

    best = ""
    best_score = 0.0
    for term in candidates:
        score = _token_similarity(low, term)
        if score > best_score:
            best, best_score = term, score
    return best, best_score


def _document_adaptive_correct_query(query: str, source_text: str) -> tuple[str, list[str]]:  # noqa: F811
    """
    Any-document retrieval correction. It builds vocabulary from the uploaded
    document and corrects misspelled query words toward those document terms.
    It never uses generic SpellChecker for content words.
    """
    doc_vocab = _build_document_vocabulary(source_text)
    changes: list[str] = []

    def repl(match):
        tok = match.group(0)
        low = tok.lower()

        # Safe fixes for query function words.
        if low in _QUERY_FUNCTION_FIXES:
            fixed = _QUERY_FUNCTION_FIXES[low]
            if fixed != low:
                changes.append(f"{tok}→{fixed}")
            return fixed.capitalize() if tok[:1].isupper() else fixed

        # Never rewrite known function words/protected acronyms.
        if low in _STOP_QUERY_WORDS or low in _PROTECTED_QUERY_TERMS or tok.isupper() or any(ch.isdigit() for ch in tok):
            return tok

        # If exact document term exists, keep it.
        if low in doc_vocab:
            return tok

        # Correct only against uploaded document vocabulary.
        best, score = _best_doc_term_anydoc(low, doc_vocab)
        if not best or best == low:
            return tok

        # Conservative but practical thresholds for realistic typos:
        # enegy->energy, solr->solar, fotosyntesis->photosynthesis.
        if len(low) <= 4:
            threshold = 82.0
        elif len(low) <= 6:
            threshold = 72.0
        else:
            threshold = 68.0

        if score >= threshold:
            fixed = best.capitalize() if tok[:1].isupper() else best
            changes.append(f"{tok}→{fixed} ({score:.0f}%)")
            return fixed
        return tok

    corrected = _max_re.sub(r"\b[A-Za-z][A-Za-z\-]{2,}\b", repl, query or "")
    corrected = _max_re.sub(r"\s+", " ", corrected).strip()
    return corrected or query, list(dict.fromkeys(changes))

_GENERIC_DOCUMENT_SUMMARY_RE = _max_re.compile(r"\b(what\s+does\s+(?:the\s+)?document\s+(?:say|contain)|summari[sz]e\s+(?:the\s+)?document|what\s+is\s+in\s+(?:the\s+)?document|give\s+(?:me\s+)?(?:a\s+)?summary)\b", _max_re.IGNORECASE)

def _is_generic_document_summary_question(text: str) -> bool:
    return bool(_GENERIC_DOCUMENT_SUMMARY_RE.search(text or ""))

class OptimizedRAGFaithfulnessChecker(RobustRAGFaithfulnessChecker):
    """
    MAX checker: audited evidence-aware checker + retrieval-only spelling correction
    + optional Groq-based translation for non-English questions.
    """

    def __init__(self, *args, enable_query_translation: bool = True, enable_retrieval_spell_fix: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_query_translation = enable_query_translation
        self.enable_retrieval_spell_fix = enable_retrieval_spell_fix

    def llm_context_sufficiency_check(self, question: str, context: str) -> Dict[str, Any]:  # type: ignore[override]
        # Generic document-summary questions are answerable from any non-empty retrieved context.
        # The previous LLM sufficiency gate sometimes labeled them as unrelated.
        if _is_generic_document_summary_question(question) and context.strip():
            return {"sufficient": True, "reason": "Generic document-summary question with retrieved context."}
        return super().llm_context_sufficiency_check(question, context)

    def _translate_question_to_english_if_needed(self, text: str) -> str:
        if not self.enable_query_translation:
            return text
        try:
            from langdetect import detect_langs
            langs = detect_langs(text[:500])
            if not langs:
                return text
            top = langs[0]
            if top.lang == "en" or top.prob < 0.72:
                return text
        except Exception:
            return text

        try:
            prompt = f"""
Translate the following user question into clear English for document retrieval.
Rules:
- Return ONLY the translated English question.
- Do not answer the question.
- Do not add new facts.
- Preserve technical terms and names.

Question:
{text}
"""
            translated = self._chat(prompt=prompt, model=self.verifier_model, max_tokens=120).strip()
            translated = _max_re.sub(r"^['\"]|['\"]$", "", translated).strip()
            if translated and len(translated) >= 5:
                return translated
        except Exception:
            return text
        return text

    @staticmethod
    def normalize_question(question: str) -> str:  # type: ignore[override]
        # Use the robust/audited base normalization first, then apply retrieval-only corrections.
        normalized = RobustRAGFaithfulnessChecker.normalize_question(question)
        return normalized

    def _prepare_query_for_retrieval(self, question: str, source_text: str = "") -> str:
        """
        Prepare the internal retrieval query.

        CRITICAL FIX:
        Document-adaptive correction must run BEFORE any generic spellchecker.
        A generic spellchecker can wrongly change document terms, e.g.
        "enegy" -> "enemy". The document vocabulary should win, because
        the uploaded document is the retrieval target.

        Therefore:
        1. Normalize punctuation/spacing.
        2. Optional translation.
        3. Correct misspelled query tokens against the uploaded document vocabulary.
        4. Use tiny common function-word fixes only for leftovers.
        5. Do NOT use generic SpellChecker to rewrite content words.
        """
        query = RobustRAGFaithfulnessChecker.normalize_question(question)
        query = self._translate_question_to_english_if_needed(query)

        self.last_retrieval_corrections = []
        if self.enable_retrieval_spell_fix:
            corrected, changes = _document_adaptive_correct_query(query, source_text)
            query = corrected
            self.last_retrieval_corrections = changes

        return query

    def check(self, question: str, source_text: str, use_cache: bool = True) -> FaithfulnessResult:  # type: ignore[override]
        """
        MAX final pipeline. Uses a retrieval/verification query that may be translated
        or spelling-corrected while preserving the original user question in the result.

        Important UX fix:
        - Misspelled questions such as "what is solr engy" are corrected only for retrieval.
        - The displayed original user question remains unchanged.
        - Generic document-summary questions are allowed to use the retrieved document text.
        """
        prepared_question = self._prepare_query_for_retrieval(question, source_text)

        # For generic document-summary prompts, reduce false refusal by not enforcing
        # overly strict retrieval confidence before the LLM sufficiency gate.
        old_min_conf = self.min_retrieval_confidence
        if _is_generic_document_summary_question(prepared_question):
            self.min_retrieval_confidence = 0.0
        try:
            result = super().check(question=prepared_question, source_text=source_text, use_cache=use_cache)
        finally:
            self.min_retrieval_confidence = old_min_conf

        result.original_question = question
        result.normalized_question = prepared_question
        result.retrieval_corrections = getattr(self, "last_retrieval_corrections", [])
        return result


def create_checker(
    api_key: _MaxOptional[str] = None,
    answer_model: str = "llama-3.3-70b-versatile",
    verifier_model: str = "llama-3.1-8b-instant",
    rebuild_model: str = "llama-3.1-8b-instant",
    second_pass_model: _MaxOptional[str] = "llama-3.3-70b-versatile",
    chunk_size: int = 450,
    chunk_overlap: int = 60,
    top_k: int = 5,
    retrieval_distance_threshold: float = 1.35,
    min_retrieval_confidence: float = 0.05,
    require_llm_context_sufficiency: bool = True,
    use_second_pass: bool = True,
    enable_query_translation: bool = True,
    enable_retrieval_spell_fix: bool = True,
) -> OptimizedRAGFaithfulnessChecker:
    return OptimizedRAGFaithfulnessChecker(
        api_key=api_key,
        answer_model=answer_model,
        verifier_model=verifier_model,
        rebuild_model=rebuild_model,
        second_pass_model=second_pass_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        retrieval_distance_threshold=retrieval_distance_threshold,
        min_retrieval_confidence=min_retrieval_confidence,
        require_llm_context_sufficiency=require_llm_context_sufficiency,
        use_second_pass=use_second_pass,
        enable_query_translation=enable_query_translation,
        enable_retrieval_spell_fix=enable_retrieval_spell_fix,
    )


def check_faithfulness(
    question: str,
    source_text: str,
    api_key: _MaxOptional[str] = None,
    use_cache: bool = False,
    **kwargs,
) -> FaithfulnessResult:
    checker = create_checker(api_key=api_key, **kwargs)
    return checker.check(question=question, source_text=source_text, use_cache=use_cache)


if __name__ == "__main__":
    sample_document = """
    Solar energy is generated by capturing sunlight using photovoltaic panels.
    These panels convert sunlight directly into electricity. Wind energy is
    generated using wind turbines, which convert the kinetic energy of moving
    air into electricity.
    """
    checker = create_checker()
    print(checker.check("WHhat is solr enrgy?", sample_document))
