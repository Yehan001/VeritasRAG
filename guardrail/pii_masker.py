import re
from dataclasses import dataclass
from typing import List, Dict, Optional

from .schema import PIIResult

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?94|0)?7\d[\s\-]?\d{3}[\s\-]?\d{4}(?!\d)")
CREDIT_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
SL_NIC_RE = re.compile(r"\b(?:\d{9}[VvXx]|\d{12})\b")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", re.I)

REGEX_PATTERNS = [
    ("EMAIL", EMAIL_RE, "[EMAIL]"),
    ("PHONE_NUMBER", PHONE_RE, "[PHONE]"),
    ("CREDIT_CARD", CREDIT_CARD_RE, "[CARD_NUMBER]"),
    ("IP_ADDRESS", IP_RE, "[IP_ADDRESS]"),
    ("SRI_LANKAN_NIC", SL_NIC_RE, "[NIC]"),
    ("IBAN", IBAN_RE, "[IBAN]"),
]


class PIIMasker:
    def __init__(self, use_presidio: bool = True, presidio_score_threshold: float = 0.85):
        self.use_presidio = use_presidio
        self.presidio_score_threshold = presidio_score_threshold
        self._analyzer = None
        self._anonymizer = None
        if use_presidio:
            try:
                from presidio_analyzer import AnalyzerEngine
                from presidio_anonymizer import AnonymizerEngine
                self._analyzer = AnalyzerEngine()
                self._anonymizer = AnonymizerEngine()
            except Exception:
                self._analyzer = None
                self._anonymizer = None

    def mask(self, text: str) -> PIIResult:
        if not text:
            return PIIResult(False, text or "", [], "none")

        if self._analyzer and self._anonymizer:
            try:
                results = self._analyzer.analyze(text=text, language="en")
                filtered = [r for r in results if getattr(r, "score", 0.0) >= self.presidio_score_threshold]
                if filtered:
                    anonymized = self._anonymizer.anonymize(text=text, analyzer_results=filtered)
                    entities = [
                        {"entity_type": r.entity_type, "start": r.start, "end": r.end, "score": round(float(r.score), 4)}
                        for r in filtered
                    ]
                    return PIIResult(True, anonymized.text, entities, "presidio")
            except Exception:
                pass

        masked = text
        entities: List[Dict[str, object]] = []
        for entity_type, pattern, replacement in REGEX_PATTERNS:
            matches = list(pattern.finditer(masked))
            if matches:
                entities.extend({"entity_type": entity_type, "match": m.group(0)} for m in matches)
                masked = pattern.sub(replacement, masked)
        return PIIResult(bool(entities), masked, entities, "regex")
