# safety.py - simple safety checks and crisis detection hooks
import re

CRISIS_KEYWORDS = ["kill myself", "suicide", "end my life", "want to die", "harm myself", "suicidal"]

def check_crisis(text):
    if not text:
        return False
    t = text.lower()
    for k in CRISIS_KEYWORDS:
        if k in t:
            return True
    return False

def simple_toxicity_filter(text):
    if not text:
        return True
    bad_patterns = [r"\bfuck you\b", r"\bdie\b", r"\bi hate you\b"]
    t = text.lower()
    for p in bad_patterns:
        if re.search(p, t):
            return False
    return True
