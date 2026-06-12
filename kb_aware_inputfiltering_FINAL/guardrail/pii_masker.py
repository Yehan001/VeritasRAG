import re
from typing import List, Tuple

from .schema import PIIResult

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
    if use_presidio:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            analyzer = AnalyzerEngine()
            anonymizer = AnonymizerEngine()
            results = analyzer.analyze(text=original, language="en")
            if results:
                anonymized = anonymizer.anonymize(text=original, analyzer_results=results)
                entities = sorted({r.entity_type for r in results})
                return PIIResult(masked_text=anonymized.text, found=True, entities=entities)
        except Exception:
            pass

    masked = original
    found_entities = []
    for entity, pattern in REGEX_PATTERNS:
        if pattern.search(masked):
            found_entities.append(entity)
            masked = pattern.sub(f"[{entity}]", masked)
    return PIIResult(masked_text=masked, found=bool(found_entities), entities=found_entities)
