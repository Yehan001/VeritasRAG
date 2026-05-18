"""
final_option_b_faithfulness_filter_FINAL.py
===========================================

FINAL reusable backend module for the Cost-Optimized Option B
RAGAS-Style Faithfulness Filtering System.

Final decisions:
- No cosine similarity for verification.
- No TextBlob / automatic spell correction.
- Safe query normalization only: spaces, repeated punctuation, noisy repeated symbols.
- Original user wording is preserved and tracked.
- Normalized question is also tracked for transparency.
- Uses 70B mainly for answer generation.
- Uses cheaper 8B model for statement generation, verification, and rebuild.
- Batch-verifies statements where possible.
- Skips rebuild when no hallucinated statements are detected.
- Provides create_checker() for reusable integration with working in-memory cache.
- Provides check_faithfulness() for simple one-off use.

Important:
This is a RAGAS-style implementation. It does not call private RAGAS internals.
"""

import os
import re
import io
import json
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from groq import Groq

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()


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

    def __str__(self) -> str:
        sep  = "=" * 66
        dash = "-" * 66
        ragas = "RAGAS PASS" if self.faithfulness_score >= 0.85 else "RAGAS FAIL"
        lines = [
            "",
            sep,
            f"  VERDICT              : {self.verdict}",
            f"  Faithfulness score   : {self.faithfulness_score:.2f}  ({ragas})",
            f"  Grounded statements  : {len(self.grounded_statements)}",
            f"  Hallucinated         : {len(self.hallucinated_statements)}",
            f"  Answer rebuilt       : {'Yes' if self.was_cleaned else 'No — all statements grounded'}",
            f"  LLM calls made       : {self.estimated_llm_calls}",
            f"  Chunks indexed       : {self.chunks_indexed}",
            dash,
            f"  Original question    : {self.original_question}",
            f"  Normalized question  : {self.normalized_question}",
            dash,
            f"  Models               : answer={self.answer_model}",
            f"                         verifier={self.verifier_model}",
            f"                         rebuild={self.rebuild_model}",
            dash,
            "  GROUNDED STATEMENTS:",
        ]
        if self.grounded_statements:
            for i, s in enumerate(self.grounded_statements, 1):
                lines.append(f"    {i}. {s}")
        else:
            lines.append("    (none)")

        lines.append("  HALLUCINATED STATEMENTS:")
        if self.hallucinated_statements:
            for i, s in enumerate(self.hallucinated_statements, 1):
                lines.append(f"    {i}. {s}")
        else:
            lines.append("    (none)")

        lines += [
            dash,
            "  FINAL ANSWER:",
            f"    {self.final_answer.strip()}",
            sep,
            "",
        ]
        return "\n".join(lines)


class OptimizedRAGFaithfulnessChecker:
    """
    FINAL reusable Option B faithfulness checker.

    Pipeline:
        user question + document
        -> safe question normalization
        -> FAISS retrieval
        -> strict answer generation
        -> atomic statement generation
        -> statement-level verification
        -> unsupported statement removal
        -> optional answer rebuild
        -> FaithfulnessResult

    Verification:
        verdict = 1 -> supported by retrieved context
        verdict = 0 -> unsupported / hallucinated / contradictory / missing-info statement
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        answer_model: str = "llama-3.3-70b-versatile",
        verifier_model: str = "llama-3.1-8b-instant",
        rebuild_model: str = "llama-3.1-8b-instant",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 450,
        chunk_overlap: int = 40,
        top_k: int = 3,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY missing. Add it to .env or pass api_key directly."
            )

        self.client = Groq(api_key=self.api_key)

        self.answer_model = answer_model
        self.verifier_model = verifier_model
        self.rebuild_model = rebuild_model
        self.top_k = top_k

        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.vectorstore = None
        self.chunks_indexed = 0
        self.llm_call_count = 0
        self.cache: Dict[str, FaithfulnessResult] = {}

    # ------------------------------------------------------------------
    # Optional text extraction helpers
    # ------------------------------------------------------------------

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
        path = file_path.lower()

        with open(file_path, "rb") as f:
            data = f.read()

        if path.endswith(".txt"):
            return OptimizedRAGFaithfulnessChecker.extract_text_from_txt_bytes(data)

        if path.endswith(".pdf"):
            return OptimizedRAGFaithfulnessChecker.extract_text_from_pdf_bytes(data)

        if path.endswith(".docx"):
            return OptimizedRAGFaithfulnessChecker.extract_text_from_docx_bytes(data)

        raise ValueError("Unsupported file type. Use .txt, .pdf, or .docx.")

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_question(question: str) -> str:
        """
        Safe query normalization only.

        Handles:
        - extra spaces
        - repeated punctuation
        - repeated noisy symbols

        Automatic spelling correction is intentionally NOT used because it can
        silently corrupt technical terms, names, acronyms, product names,
        medical/legal vocabulary, and document-specific terminology.

        Example:
            "what   is solar energy????%%%" -> "what is solar energy?"
        """
        question = question.strip()

        # Fix repeated spaces
        question = re.sub(r"\s+", " ", question)

        # Fix repeated punctuation
        question = re.sub(r"[?!.]{2,}", "?", question)

        # Remove repeated noisy symbols without rewriting words
        question = re.sub(r"[%#@$^&*_=+]{2,}", "", question)

        return question

    @staticmethod
    def _make_cache_key(question: str, source_text: str) -> str:
        source_hash = hashlib.sha256(
            source_text.encode("utf-8", errors="ignore")
        ).hexdigest()[:16]
        return f"{source_hash}:{question.lower().strip()}"

    @staticmethod
    def _extract_json(text: str):
        text = text.strip()
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

    def _chat(self, prompt: str, model: str, max_tokens: int = 900) -> str:
        self.llm_call_count += 1

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def load_document(self, source_text: str) -> int:
        chunks = self.splitter.split_text(source_text)

        if not chunks:
            raise ValueError("No valid text found in document.")

        self.vectorstore = FAISS.from_texts(chunks, self.embeddings)
        self.chunks_indexed = len(chunks)
        return self.chunks_indexed

    def retrieve_context(self, question: str) -> str:
        if self.vectorstore is None:
            raise RuntimeError("Document not loaded. Call load_document() first.")

        docs = self.vectorstore.similarity_search(question, k=self.top_k)
        return "\n\n".join(doc.page_content for doc in docs)

    # ------------------------------------------------------------------
    # LLM stages
    # ------------------------------------------------------------------

    def generate_answer(self, question: str, context: str) -> str:
        prompt = f"""
You are a strict document-based assistant.

Use ONLY the provided context to answer the question.

Rules:
1. Do not use outside knowledge.
2. Do not guess.
3. If the answer is not in the context, say exactly:
   "This information is not in the provided context."
4. If the question has multiple parts, answer each part separately.
5. Keep the answer clear and concise.

Context:
{context}

Question:
{question}

Answer:
"""
        return self._chat(
            prompt=prompt,
            model=self.answer_model,
            max_tokens=700,
        )

    def generate_statements(self, question: str, answer: str) -> List[str]:
        prompt = f"""
Given a question and answer, create atomic statements from the answer.

An atomic statement means:
- one clear fact only
- complete sentence
- no extra explanation
- no added information

Question:
{question}

Answer:
{answer}

Return ONLY a JSON list of strings.
"""
        raw = self._chat(
            prompt=prompt,
            model=self.verifier_model,
            max_tokens=500,
        )
        parsed = self._extract_json(raw)

        if isinstance(parsed, list):
            statements = [str(x).strip() for x in parsed if str(x).strip()]
            if statements:
                return statements

        # Fallback if JSON parsing fails
        fallback = re.split(r"(?<=[.!?])\s+", answer.strip())
        return [s.strip() for s in fallback if s.strip()]

    def verify_statements(
        self,
        context: str,
        statements: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Batch verification:
        Tries to verify all statements in one LLM call.
        Falls back to one-by-one verification only if JSON parsing fails.
        """
        if not statements:
            return []

        prompt = f"""
Your task is to judge the faithfulness of statements based only on the given context.

For each statement:
- verdict = 1 if the statement is directly supported by the context.
- verdict = 0 if the statement is not supported by the context.
- verdict = 0 if the statement contradicts the context.
- verdict = 0 if the statement says information is not in the context.
- Do not use outside knowledge.
- If unsure, use verdict = 0.

Context:
{context}

Statements:
{json.dumps(statements, indent=2)}

Return ONLY valid JSON in this exact format.

Rules for JSON:
- Copy each original statement exactly into the "statement" field.
- Use verdict = 1 if supported.
- Use verdict = 0 if unsupported.

Example:
[
  {{"statement": "Solar energy is generated using photovoltaic panels.", "verdict": 1}},
  {{"statement": "Solar panels are 99% efficient.", "verdict": 0}}
]
"""
        raw = self._chat(
            prompt=prompt,
            model=self.verifier_model,
            max_tokens=900,
        )
        parsed = self._extract_json(raw)

        if isinstance(parsed, list):
            cleaned = []

            for idx, item in enumerate(parsed):
                if isinstance(item, dict) and "verdict" in item:
                    default_statement = statements[min(idx, len(statements) - 1)]
                    statement = str(item.get("statement", default_statement)).strip()

                    try:
                        verdict = 1 if int(item["verdict"]) == 1 else 0
                    except Exception:
                        verdict = 0

                    cleaned.append({
                        "statement": statement,
                        "verdict": verdict,
                    })

            if cleaned:
                return cleaned

        # Fallback: verify one by one
        results = []

        for statement in statements:
            single_prompt = f"""
Check whether this statement is supported by the context.

Context:
{context}

Statement:
{statement}

Return ONLY valid JSON.

Rules for JSON:
- Copy the given statement exactly into the "statement" field.
- Use verdict = 1 if supported.
- Use verdict = 0 if unsupported.

Example format:
{{"statement": "{statement}", "verdict": 1}}
"""
            raw_single = self._chat(
                prompt=single_prompt,
                model=self.verifier_model,
                max_tokens=200,
            )
            parsed_single = self._extract_json(raw_single)

            if isinstance(parsed_single, dict) and "verdict" in parsed_single:
                try:
                    verdict = 1 if int(parsed_single["verdict"]) == 1 else 0
                except Exception:
                    verdict = 0
            else:
                verdict = 0

            results.append({
                "statement": statement,
                "verdict": verdict,
            })

        return results

    def rebuild_answer(
        self,
        question: str,
        grounded_statements: List[str],
    ) -> str:
        if not grounded_statements:
            return "I cannot provide a reliable answer based on the provided context."

        prompt = f"""
Rewrite the following verified statements into a fluent final answer.

Rules:
- Use ONLY the verified statements.
- Do not add new facts.
- Do not use outside knowledge.
- Keep the answer concise.

Question:
{question}

Verified statements:
{json.dumps(grounded_statements, indent=2)}

Final answer:
"""
        return self._chat(
            prompt=prompt,
            model=self.rebuild_model,
            max_tokens=500,
        )

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def check(
        self,
        question: str,
        source_text: str,
        use_cache: bool = True,
    ) -> FaithfulnessResult:
        """
        Run the complete final Option B pipeline.
        """
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
        context = self.retrieve_context(normalized_question)

        raw_answer = self.generate_answer(
            question=normalized_question,
            context=context,
        )

        statements = self.generate_statements(
            question=normalized_question,
            answer=raw_answer,
        )

        verdicts = self.verify_statements(
            context=context,
            statements=statements,
        )

        grounded = []
        hallucinated = []

        for item in verdicts:
            statement = str(item.get("statement", "")).strip()

            if not statement:
                continue

            if item.get("verdict") == 1:
                grounded.append(statement)
            else:
                hallucinated.append(statement)

        total = len(grounded) + len(hallucinated)
        score = len(grounded) / total if total else 0.0

        if score >= 0.85:
            verdict = "PASS"
        elif score >= 0.50:
            verdict = "PARTIAL"
        else:
            verdict = "FAIL"

        # Cost optimization: skip rebuild if nothing was removed.
        if hallucinated:
            final_answer = self.rebuild_answer(
                question=normalized_question,
                grounded_statements=grounded,
            )
            was_cleaned = True
        else:
            final_answer = raw_answer
            was_cleaned = False

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
        )

        if use_cache:
            self.cache[cache_key] = result

        return result


def create_checker(
    api_key: Optional[str] = None,
    answer_model: str = "llama-3.3-70b-versatile",
    verifier_model: str = "llama-3.1-8b-instant",
    rebuild_model: str = "llama-3.1-8b-instant",
    chunk_size: int = 450,
    chunk_overlap: int = 40,
    top_k: int = 3,
) -> OptimizedRAGFaithfulnessChecker:
    """
    Recommended factory for chatbot/API integration.

    Use this when you want caching to work across multiple questions:

        checker = create_checker()
        result1 = checker.check(question="Question 1", source_text=document)
        result2 = checker.check(question="Question 2", source_text=document)

    The same checker instance keeps the in-memory cache.
    """
    return OptimizedRAGFaithfulnessChecker(
        api_key=api_key,
        answer_model=answer_model,
        verifier_model=verifier_model,
        rebuild_model=rebuild_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
    )


def check_faithfulness(
    question: str,
    source_text: str,
    api_key: Optional[str] = None,
    answer_model: str = "llama-3.3-70b-versatile",
    verifier_model: str = "llama-3.1-8b-instant",
    rebuild_model: str = "llama-3.1-8b-instant",
    chunk_size: int = 450,
    chunk_overlap: int = 40,
    top_k: int = 3,
    use_cache: bool = False,
) -> FaithfulnessResult:
    """
    Simple one-call helper.

    Best for quick testing or one-off use.

    Note:
        This creates a new checker instance for each call.
        Therefore, in-memory caching does not persist across repeated calls
        to this wrapper function.

    For chatbot/API integration where caching matters, use create_checker()
    and reuse the same checker object.
    """
    checker = create_checker(
        api_key=api_key,
        answer_model=answer_model,
        verifier_model=verifier_model,
        rebuild_model=rebuild_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
    )

    return checker.check(
        question=question,
        source_text=source_text,
        use_cache=use_cache,
    )


if __name__ == "__main__":
    sample_document = """
    Solar energy is generated by capturing sunlight using photovoltaic panels.
    These panels convert sunlight directly into electricity. Wind energy is
    generated using wind turbines, which convert the kinetic energy of moving
    air into electricity.
    """

    # Recommended pattern because cache persists inside this checker object.
    checker = create_checker()

    result = checker.check(
        question="What is solar energy and how is wind energy generated?",
        source_text=sample_document,
    )

    print(result)
