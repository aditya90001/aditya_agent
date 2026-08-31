import re


def preprocess_query(query: str) -> dict:
    """Return a dict with normalized, lexical and extracted signals."""
    q = query.strip()
    lower = q.lower()

    # extract standard numbers like IS 12345 or IS-12345 or IS12345
    std_re = re.compile(r"\bIS[- ]?(\d{2,6})\b", re.IGNORECASE)
    stds = std_re.findall(q)

    # extract clauses like Clause 5.2 or clause 5
    clause_re = re.compile(r"\b[Cc]lause\s+([0-9]+(?:\.[0-9]+)*)")
    clauses = clause_re.findall(q)

    # certification intent keywords
    cert_keywords = [
        "certification",
        "mandatory",
        "license",
        "licence",
        "scheme",
        "approval",
    ]

    intent = None
    for kw in cert_keywords:
        if kw in lower:
            intent = "certification"
            break

    return {
        "original": q,
        "lexical": lower,
        "standards": stds,
        "clauses": clauses,
        "intent": intent,
    }
