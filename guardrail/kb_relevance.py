import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Block ALL outbound Hugging Face / datasets network calls at the library level.
# These must be set before any sentence_transformers / huggingface_hub import.
# ---------------------------------------------------------------------------
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from .schema import RelevanceResult
from .text_utils import normalize_text

# ---------------------------------------------------------------------------
# Resolve the local models/ directory.
# Expected layout:
#
#   <project_root>/
#       guardrail/
#           kb_relevance.py       ← this file
#       models/
#           sentence-transformers__all-MiniLM-L6-v2/
#
# __file__ → .../guardrail/kb_relevance.py
# parent   → .../guardrail/
# parent^2 → .../  (project root)
# ---------------------------------------------------------------------------
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_PACKAGE_DIR)
_ST_LOCAL_PATH = os.path.join(
    _PROJECT_ROOT, "models", "sentence-transformers__all-MiniLM-L6-v2"
)


def chunk_text(text: str, max_words: int = 120, overlap: int = 25) -> List[str]:
    words = re.findall(r"\S+", text or "")
    if not words:
        return []
    chunks = []
    step = max(1, max_words - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + max_words]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


@dataclass
class KBRelevanceChecker:
    backend: str = "tfidf"          # "tfidf" or "sentence_transformers"
    relevance_threshold: float = 0.08
    grounding_threshold: float = 0.12
    # Points to the locally downloaded model folder — never contacts HF hub.
    st_model_name: str = _ST_LOCAL_PATH
    chunks: List[str] = field(default_factory=list)
    method_used: str = "tfidf"
    _vectorizer: Optional[object] = None
    _matrix: Optional[object] = None
    _st_model: Optional[object] = None
    _st_embeddings: Optional[object] = None

    def fit(self, kb_text: str) -> None:
        start_total = time.perf_counter()
        self.chunks = chunk_text(kb_text)
        if not self.chunks:
            self.chunks = [""]

        if self.backend == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer

                start_load = time.perf_counter()
                # local_files_only=True → raises immediately if folder missing,
                # no network attempt is made.
                self._st_model = SentenceTransformer(
                    self.st_model_name,
                    local_files_only=True,   # ← OFFLINE: never attempt download
                )
                load_latency = time.perf_counter() - start_load
                print(f"[KBRelevanceChecker.fit] model_load={self.st_model_name} latency={load_latency:.4f} seconds")

                start_encode = time.perf_counter()
                self._st_embeddings = self._st_model.encode(
                    self.chunks, normalize_embeddings=True
                )
                encode_latency = time.perf_counter() - start_encode
                print(f"[KBRelevanceChecker.fit] chunk_encode chunks={len(self.chunks)} latency={encode_latency:.4f} seconds")

                self.method_used = "sentence_transformers"
                total_latency = time.perf_counter() - start_total
                print(f"[KBRelevanceChecker.fit] backend={self.method_used} total_latency={total_latency:.4f} seconds")
                return
            except Exception as exc:
                # Model folder missing or corrupt — fall through to TF-IDF.
                print(f"[KBRelevanceChecker.fit] sentence_transformers failed ({exc}), falling back to tfidf")
                self.method_used = "tfidf_fallback"

        try:
            start_fit = time.perf_counter()
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
            self._matrix = self._vectorizer.fit_transform(
                [normalize_text(c) for c in self.chunks]
            )
            self.method_used = "tfidf"
            fit_latency = time.perf_counter() - start_fit
            print(f"[KBRelevanceChecker.fit] tfidf_fit_transform latency={fit_latency:.4f} seconds")
        except Exception as exc:
            print(f"[KBRelevanceChecker.fit] tfidf failed ({exc}), using keyword_overlap")
            self.method_used = "keyword_overlap"

        total_latency = time.perf_counter() - start_total
        print(f"[KBRelevanceChecker.fit] backend={self.method_used} total_latency={total_latency:.4f} seconds")

    def check(self, question: str, top_k: int = 3) -> RelevanceResult:
        start_total = time.perf_counter()
        q = normalize_text(question)
        if not q or not self.chunks:
            latency = time.perf_counter() - start_total
            print(f"[KBRelevanceChecker.check] backend={self.method_used} empty_input latency={latency:.4f} seconds")
            return RelevanceResult(False, False, 0.0, [], self.method_used)

        if self.method_used == "sentence_transformers" and self._st_model is not None:
            start_score = time.perf_counter()
            q_emb = self._st_model.encode([q], normalize_embeddings=True)[0]
            scores = np.asarray(self._st_embeddings @ q_emb)
            result = self._make_result(scores, top_k)
            latency = time.perf_counter() - start_score
            total_latency = time.perf_counter() - start_total
            print(f"[KBRelevanceChecker.check] backend=sentence_transformers score_latency={latency:.4f} total_latency={total_latency:.4f} seconds")
            return result

        if self.method_used.startswith("tfidf") and self._vectorizer is not None and self._matrix is not None:
            start_score = time.perf_counter()
            q_vec = self._vectorizer.transform([q])
            scores = (self._matrix @ q_vec.T).toarray().ravel()
            result = self._make_result(scores, top_k)
            latency = time.perf_counter() - start_score
            total_latency = time.perf_counter() - start_total
            print(f"[KBRelevanceChecker.check] backend={self.method_used} score_latency={latency:.4f} total_latency={total_latency:.4f} seconds")
            return result

        # Last-resort lexical overlap fallback
        start_score = time.perf_counter()
        q_terms = set(re.findall(r"[a-zA-Z]{3,}", q.lower()))
        scores = []
        for chunk in self.chunks:
            c_terms = set(re.findall(r"[a-zA-Z]{3,}", chunk.lower()))
            overlap = len(q_terms & c_terms) / max(1, len(q_terms))
            scores.append(overlap)
        result = self._make_result(np.asarray(scores), top_k)
        latency = time.perf_counter() - start_score
        total_latency = time.perf_counter() - start_total
        print(f"[KBRelevanceChecker.check] backend=keyword_overlap score_latency={latency:.4f} total_latency={total_latency:.4f} seconds")
        return result

    def _make_result(self, scores: np.ndarray, top_k: int) -> RelevanceResult:
        if scores.size == 0:
            return RelevanceResult(False, False, 0.0, [], self.method_used)
        idx = np.argsort(scores)[::-1][:top_k]
        best = float(scores[idx[0]]) if idx.size else 0.0
        top_chunks = [self.chunks[i] for i in idx if scores[i] > 0]
        return RelevanceResult(
            is_relevant=best >= self.relevance_threshold,
            is_grounded=best >= self.grounding_threshold,
            score=round(best, 4),
            top_chunks=top_chunks,
            method=self.method_used,
        )