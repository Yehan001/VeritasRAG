"""
pii_masking.py
==============

Broad free PII masking layer.

This is not perfect every-country identity detection. It combines:
- regex patterns
- phonenumbers library if installed
- Luhn validation for credit cards
- optional Microsoft Presidio if installed

The goal is practical global coverage, not a false claim of 100% worldwide PII detection.
"""

from __future__ import annotations

import re
from typing import List, Tuple

try:
    from country_pii_patterns import apply_country_pii_patterns
except Exception:
    apply_country_pii_patterns = None


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:0\d{9}|\+94\d{9}|\+?\d[\d\s\-]{8,}\d)\b")
NIC_LK_RE = re.compile(r"\b\d{9}[vVxX]\b|\b\d{12}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
AADHAAR_RE = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")
UK_NINO_RE = re.compile(r"\b(?!BG|GB|KN|NK|NT|TN|ZZ)[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b", re.I)
CANADA_SIN_RE = re.compile(r"\b\d{3}[-\s]?\d{3}[-\s]?\d{3}\b")
SINGAPORE_NRIC_RE = re.compile(r"\b[STFGM]\d{7}[A-Z]\b", re.I)
HKID_RE = re.compile(r"\b[A-Z]{1,2}\d{6}\(?[0-9A]\)?\b", re.I)
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
IPV6_RE = re.compile(r"\b(?:[A-Fa-f0-9]{1,4}:){2,7}[A-Fa-f0-9]{1,4}\b")
MAC_RE = re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b")
PASSPORT_CONTEXT_RE = re.compile(r"\b(?:passport|travel\s*document)\s*(?:no|number|#)?\s*[:#-]?\s*[A-Z0-9]{6,12}\b", re.I)
PASSPORT_LIKE_RE = re.compile(r"\b[A-Z]{1,2}\d{6,9}\b")
BANK_RE = re.compile(r"\b(?:account|acct|bank)\s*(?:no|number|#)?\s*[:\-]?\s*\d{8,18}\b", re.I)
TAX_ID_CONTEXT_RE = re.compile(r"\b(?:tax\s*id|tin|vat|gst|nif|nie|pan|nric|fin|sin|ssn|national\s*id|identity\s*number)\s*[:#-]?\s*[A-Z0-9 -]{6,25}\b", re.I)

PHONE_REGIONS = [
    "LK", "IN", "US", "CA", "GB", "AU", "NZ", "SG", "MY", "AE",
    "SA", "ZA", "BR", "MX", "DE", "FR", "IT", "ES", "NL", "SE",
    "NO", "DK", "FI", "JP", "KR", "CN", "TH", "PH", "ID",
]


def _luhn_valid(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if len(digits) < 13 or len(digits) > 19:
        return False
    total = 0
    for i, d in enumerate(digits[::-1]):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _mask_cards(text: str) -> Tuple[str, List[str]]:
    labels = []
    candidate_re = re.compile(r"\b(?:\d[ -]?){13,19}\b")

    def repl(match):
        value = match.group(0)
        digits = re.sub(r"\D", "", value)
        if _luhn_valid(digits):
            labels.append("credit_card_luhn")
            return "[CARD]"
        return value

    return candidate_re.sub(repl, text), labels


def _mask_phones(text: str) -> Tuple[str, List[str]]:
    labels = []
    try:
        import phonenumbers
    except Exception:
        return text, labels

    spans = []
    for match in phonenumbers.PhoneNumberMatcher(text, None):
        if phonenumbers.is_valid_number(match.number):
            spans.append((match.start, match.end))

    for region in PHONE_REGIONS:
        try:
            for match in phonenumbers.PhoneNumberMatcher(text, region):
                raw = text[match.start:match.end]
                digit_count = len(re.sub(r"\D", "", raw))
                if digit_count >= 7 and phonenumbers.is_valid_number(match.number):
                    spans.append((match.start, match.end))
        except Exception:
            pass

    spans = sorted(set(spans))
    merged = []
    for s, e in spans:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    out = text
    for s, e in reversed(merged):
        out = out[:s] + "[PHONE]" + out[e:]
        labels.append("phone_number_global")

    return out, list(set(labels))


def _optional_presidio(text: str) -> Tuple[str, List[str]]:
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
    except Exception:
        return text, []

    try:
        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        results = analyzer.analyze(text=text, language="en")
        if not results:
            return text, []
        anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text, ["presidio_entities"]
    except Exception:
        return text, []


def mask_pii(text: str) -> Tuple[str, List[str]]:
    labels = []
    text = "" if text is None else str(text)

    text, presidio_labels = _optional_presidio(text)
    labels.extend(presidio_labels)

    if EMAIL_RE.search(text):
        text = EMAIL_RE.sub("[EMAIL]", text)
        labels.append("email")

    text, card_labels = _mask_cards(text)
    labels.extend(card_labels)

    text, phone_labels = _mask_phones(text)
    labels.extend(phone_labels)

    if PHONE_RE.search(text):
        text = PHONE_RE.sub("[PHONE]", text)
        labels.append("phone_regex")

    replacements = [
        (NIC_LK_RE, "[NIC_OR_ID]", "sri_lanka_nic_or_id"),
        (SSN_RE, "[SSN]", "us_ssn"),
        (AADHAAR_RE, "[AADHAAR]", "india_aadhaar_like"),
        (UK_NINO_RE, "[UK_NINO]", "uk_nino"),
        (CANADA_SIN_RE, "[CANADA_SIN]", "canada_sin_like"),
        (SINGAPORE_NRIC_RE, "[SG_NRIC_FIN]", "singapore_nric_fin"),
        (HKID_RE, "[HKID]", "hong_kong_id"),
        (IBAN_RE, "[IBAN]", "iban"),
        (IPV4_RE, "[IP_ADDRESS]", "ipv4_address"),
        (IPV6_RE, "[IPV6_ADDRESS]", "ipv6_address"),
        (MAC_RE, "[MAC_ADDRESS]", "mac_address"),
        (PASSPORT_CONTEXT_RE, "[PASSPORT]", "passport_context"),
        (PASSPORT_LIKE_RE, "[PASSPORT]", "passport_like"),
        (BANK_RE, "[BANK_ACCOUNT]", "bank_account_context"),
        (TAX_ID_CONTEXT_RE, "[TAX_ID]", "tax_or_national_id_context"),
    ]

    for pattern, token, label in replacements:
        if pattern.search(text):
            text = pattern.sub(token, text)
            labels.append(label)

    if apply_country_pii_patterns is not None:
        text, country_labels = apply_country_pii_patterns(text)
        labels.extend(country_labels)

    return text, list(dict.fromkeys(labels))
