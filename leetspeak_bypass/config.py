LEET_MAP = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
    "!": "i",
    "+": "t",
    "8": "b"
}

BLOCKED_TERMS = [
    "hate",
    "hack",
    "bypass",
    "fraud",
    "scam",
    "password",
    "otp",
    "admin",
    "sql injection",
    "ignore previous instructions"
]

SUSPICIOUS_PATTERNS = [
    r"ignore.*instruction",
    r"bypass.*security",
    r"hack.*system",
]