import re

from streamlit import text
from config import LEET_MAP, BLOCKED_TERMS, SUSPICIOUS_PATTERNS


class LeetspeakDetector:

    def __init__(self):
        self.leet_map = LEET_MAP
        self.blocked_terms = BLOCKED_TERMS
        self.suspicious_patterns = SUSPICIOUS_PATTERNS

    def normalize_text(self, text):

        normalized = text.lower()

        # Replace leetspeak characters
        for leet_char, normal_char in self.leet_map.items():
            normalized = normalized.replace(leet_char, normal_char)

        # Replace suspicious separators with spaces
        normalized = re.sub(r'[-_.]+', ' ', normalized)

        # Remove special characters but preserve spaces
        normalized = re.sub(r'[^a-zA-Z0-9\s]', '', normalized)

        # Normalize multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        return normalized

    def calculate_risk_score(self, text):

        leet_count = sum(
            1 for char in text if char in self.leet_map
        )

        if len(text) == 0:
            return 0

        return round(leet_count / len(text), 2)

    def detect_blocked_terms(self, normalized_text):

        detected_terms = []

        for term in self.blocked_terms:
            if term in normalized_text:
                detected_terms.append(term)

        return detected_terms
    
    def detect_patterns(self, text):

        matches = []

        for pattern in self.suspicious_patterns:
            if re.search(pattern, text):
                matches.append(pattern)

        return matches

    def analyze(self, text):

        normalized = self.normalize_text(text)

        detected_terms = self.detect_blocked_terms(normalized)
        suspicious_patterns = self.detect_patterns(text)
        risk_score = self.calculate_risk_score(text)

        return {
            "original_text": text,
            "normalized_text": normalized,
            "risk_score": risk_score,
            "detected_terms": detected_terms,
            "suspicious_patterns": suspicious_patterns,
            "is_suspicious": len(detected_terms) > 0 or len(suspicious_patterns) > 0
        }