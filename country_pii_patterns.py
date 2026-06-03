"""
country_pii_patterns.py
=======================

Additional country/region-oriented PII recognizers.

This file improves global masking coverage. It still does not guarantee
perfect every-country PII detection because formats and laws differ widely.

For production/global deployment, combine:
- Microsoft Presidio
- phonenumbers
- country-specific recognizers
- local compliance review
"""

from __future__ import annotations

import re
from typing import List, Tuple


COUNTRY_PII_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    # Sri Lanka
    (re.compile(r"\b\d{9}[vVxX]\b|\b\d{12}\b"), "[LK_NIC]", "sri_lanka_nic"),

    # United States
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[US_SSN]", "us_ssn"),

    # India
    (re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"), "[IN_AADHAAR]", "india_aadhaar_like"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.I), "[IN_PAN]", "india_pan"),

    # UK
    (re.compile(r"\b(?!BG|GB|KN|NK|NT|TN|ZZ)[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b", re.I), "[UK_NINO]", "uk_nino"),
    (re.compile(r"\b\d{3}\s?\d{3}\s?\d{4}\b"), "[UK_NHS_LIKE]", "uk_nhs_like"),

    # Canada
    (re.compile(r"\b\d{3}[-\s]?\d{3}[-\s]?\d{3}\b"), "[CA_SIN_LIKE]", "canada_sin_like"),

    # Singapore
    (re.compile(r"\b[STFGM]\d{7}[A-Z]\b", re.I), "[SG_NRIC_FIN]", "singapore_nric_fin"),

    # Hong Kong
    (re.compile(r"\b[A-Z]{1,2}\d{6}\(?[0-9A]\)?\b", re.I), "[HKID]", "hong_kong_id"),

    # Brazil
    (re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"), "[BR_CPF_LIKE]", "brazil_cpf_like"),
    (re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b"), "[BR_CNPJ_LIKE]", "brazil_cnpj_like"),

    # Mexico
    (re.compile(r"\b[A-Z]{4}\d{6}[HM][A-Z]{5}[A-Z0-9]\d\b", re.I), "[MX_CURP]", "mexico_curp"),
    (re.compile(r"\b[A-ZÑ&]{3,4}\d{6}[A-Z0-9]{3}\b", re.I), "[MX_RFC_LIKE]", "mexico_rfc_like"),

    # South Africa
    (re.compile(r"\b\d{6}\s?\d{4}\s?\d{3}\b"), "[ZA_ID_LIKE]", "south_africa_id_like"),

    # Australia
    (re.compile(r"\b\d{3}\s?\d{3}\s?\d{3}\b|\b\d{2}\s?\d{3}\s?\d{3}\b"), "[AU_TFN_LIKE]", "australia_tfn_like"),

    # EU / international
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), "[IBAN]", "iban"),
    (re.compile(r"\b[A-Z]{2}[A-Z0-9]{6,12}\b"), "[PASSPORT_LIKE]", "passport_like"),
]


def apply_country_pii_patterns(text: str):
    labels = []
    output = text
    for pattern, token, label in COUNTRY_PII_PATTERNS:
        if pattern.search(output):
            output = pattern.sub(token, output)
            labels.append(label)
    return output, list(dict.fromkeys(labels))
