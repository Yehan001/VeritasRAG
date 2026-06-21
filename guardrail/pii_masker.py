import re
import time
from functools import lru_cache
from typing import List, Tuple

from .schema import PIIResult


@lru_cache(maxsize=1)
def _get_presidio_engines():
    """Build Presidio's AnalyzerEngine/AnonymizerEngine ONCE per process.
    AnalyzerEngine() loads a spaCy NLP model from disk internally — that is
    the multi-second cost previously paid on every single mask_pii() call.
    """
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    return AnalyzerEngine(), AnonymizerEngine()


REGEX_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3}[\s-]?\d{4}(?!\d)")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("SRI_LANKA_NIC", re.compile(r"\b(?:\d{9}[vVxX]|\d{12})\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
]


def mask_pii(text: str, use_presidio: bool = True) -> PIIResult:
    original = text or ""
    start = time.perf_counter()

    if use_presidio:
        try:
            analyzer, anonymizer = _get_presidio_engines()
            results = analyzer.analyze(text=original, language="en")
            if results:
                anonymized = anonymizer.anonymize(text=original, analyzer_results=results)
                entities = sorted({r.entity_type for r in results})
                latency = time.perf_counter() - start
                print(f"[mask_pii] backend=presidio latency={latency:.4f} seconds")
                return PIIResult(masked_text=anonymized.text, found=True, entities=entities)
        except Exception:
            pass

    masked = original
    found_entities = []
    for entity, pattern in REGEX_PATTERNS:
        if pattern.search(masked):
            found_entities.append(entity)
            masked = pattern.sub(f"[{entity}]", masked)

    latency = time.perf_counter() - start
    print(f"[mask_pii] backend=regex_fallback latency={latency:.4f} seconds")
    return PIIResult(masked_text=masked, found=bool(found_entities), entities=found_entities)