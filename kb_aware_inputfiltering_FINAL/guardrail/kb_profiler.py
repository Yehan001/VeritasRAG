from collections import Counter
from typing import Dict, List

from .schema import KBProfile
from .text_utils import normalize_text

# Future-proof note:
# This is a configurable domain profile. Add domains/terms here without changing the rest of the pipeline.
DOMAIN_TERMS: Dict[str, List[str]] = {
    "cybersecurity": [
        "cybersecurity", "ethical hacking", "penetration testing", "sql injection", "xss",
        "firewall", "malware", "phishing", "authentication", "vulnerability", "exploit",
        "owasp", "encryption", "network security", "incident response", "threat model",
    ],
    "medical": [
        "patient", "symptom", "diagnosis", "medicine", "treatment", "clinical", "doctor",
        "hospital", "disease", "health", "dosage", "prescription", "therapy",
    ],
    "finance": [
        "loan", "interest rate", "bank", "credit", "debit", "investment", "insurance",
        "tax", "invoice", "payment", "mortgage", "financial", "budget",
    ],
    "legal": [
        "contract", "clause", "liability", "law", "legal", "court", "agreement",
        "policy", "compliance", "regulation", "statute", "privacy policy",
    ],
    "energy": [
        "solar", "renewable", "electricity", "battery", "inverter", "photovoltaic",
        "energy", "wind power", "grid", "kilowatt", "power generation",
    ],
    "education": [
        "lesson", "student", "curriculum", "assignment", "learning", "teacher", "exam",
        "course", "lecture", "study", "training", "module",
    ],
}

HIGH_RISK_DOMAINS = {"cybersecurity", "medical", "finance", "legal"}


def profile_kb(text: str) -> KBProfile:
    normalized = normalize_text(text).lower()
    scores = Counter()
    matched: Dict[str, List[str]] = {}

    for domain, terms in DOMAIN_TERMS.items():
        domain_matches = []
        for term in terms:
            if term in normalized:
                domain_matches.append(term)
                scores[domain] += 2 if " " in term else 1
        matched[domain] = domain_matches

    if not scores:
        return KBProfile(domain="general", confidence=0.40, risk_level="low", matched_terms=[])

    domain, score = scores.most_common(1)[0]
    total = sum(scores.values()) or 1
    confidence = min(0.95, max(0.50, score / total))
    risk = "high" if domain in HIGH_RISK_DOMAINS else "low"
    return KBProfile(domain=domain, confidence=round(confidence, 3), risk_level=risk, matched_terms=matched[domain][:20])
