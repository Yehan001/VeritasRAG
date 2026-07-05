import re
import time
from functools import lru_cache
from typing import List, Tuple

from .schema import PIIResult


@lru_cache(maxsize=1)
def _get_presidio_analyzer():
    """Build Presidio's AnalyzerEngine ONCE per process.
    AnalyzerEngine() loads a spaCy NLP model from disk internally — that is
    the multi-second cost previously paid on every single call.
    """
    from presidio_analyzer import AnalyzerEngine
    return AnalyzerEngine()


REGEX_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3}[\s-]?\d{4}(?!\d)")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("SRI_LANKA_NIC", re.compile(r"\b(?:\d{9}[vVxX]|\d{12})\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
]


def detect_pii(text: str, use_presidio: bool = True) -> PIIResult:
    """Detect PII in text and return what was found.
    Does NOT mask or modify the text — original text is always passed through.
    """
    original = text or ""
    start = time.perf_counter()

    if use_presidio:
        try:
            analyzer = _get_presidio_analyzer()
            results = analyzer.analyze(text=original, language="en")
            if results:
                entities = sorted({r.entity_type for r in results})
                latency = time.perf_counter() - start
                print(f"[detect_pii] backend=presidio found={entities} latency={latency:.4f} seconds")
                # masked_text returns original unchanged — detection only
                return PIIResult(masked_text=original, found=True, entities=entities)
        except Exception:
            pass

    # Regex fallback — detect only, no substitution
    found_entities = []
    for entity, pattern in REGEX_PATTERNS:
        if pattern.search(original):
            found_entities.append(entity)

    latency = time.perf_counter() - start
    print(f"[detect_pii] backend=regex_fallback found={found_entities} latency={latency:.4f} seconds")
    return PIIResult(masked_text=original, found=bool(found_entities), entities=found_entities)