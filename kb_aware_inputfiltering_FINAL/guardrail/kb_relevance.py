import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .schema import RelevanceResult
from .text_utils import normalize_text


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
    backend: str = "tfidf"  # tfidf or sentence_transformers
    relevance_threshold: float = 0.08
    grounding_threshold: float = 0.12
    st_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunks: List[str] = field(default_factory=list)
    method_used: str = "tfidf"
    _vectorizer: Optional[object] = None
    _matrix: Optional[object] = None
    _st_model: Optional[object] = None
    _st_embeddings: Optional[object] = None

    def fit(self, kb_text: str) -> None:
        self.chunks = chunk_text(kb_text)
        if not self.chunks:
            self.chunks = [""]

        if self.backend == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer
                self._st_model = SentenceTransformer(self.st_model_name)
                self._st_embeddings = self._st_model.encode(self.chunks, normalize_embeddings=True)
                self.method_used = "sentence_transformers"
                return
            except Exception:
                self.method_used = "tfidf_fallback"

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
            self._matrix = self._vectorizer.fit_transform([normalize_text(c) for c in self.chunks])
            self.method_used = "tfidf"
        except Exception:
            self.method_used = "keyword_overlap"

    def check(self, question: str, top_k: int = 3) -> RelevanceResult:
        q = normalize_text(question)
        if not q or not self.chunks:
            return RelevanceResult(False, False, 0.0, [], self.method_used)

        if self.method_used == "sentence_transformers" and self._st_model is not None:
            q_emb = self._st_model.encode([q], normalize_embeddings=True)[0]
            scores = np.asarray(self._st_embeddings @ q_emb)
            return self._make_result(scores, top_k)

        if self.method_used.startswith("tfidf") and self._vectorizer is not None and self._matrix is not None:
            q_vec = self._vectorizer.transform([q])
            scores = (self._matrix @ q_vec.T).toarray().ravel()
            return self._make_result(scores, top_k)

        # Last-resort lexical overlap fallback
        q_terms = set(re.findall(r"[a-zA-Z]{3,}", q.lower()))
        scores = []
        for chunk in self.chunks:
            c_terms = set(re.findall(r"[a-zA-Z]{3,}", chunk.lower()))
            overlap = len(q_terms & c_terms) / max(1, len(q_terms))
            scores.append(overlap)
        return self._make_result(np.asarray(scores), top_k)

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
