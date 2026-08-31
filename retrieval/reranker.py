import os
from typing import List, Tuple

RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "false").lower() in ("1", "true", "yes")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

_reranker = None

def _init_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker

    if not RERANKER_ENABLED:
        return None

    try:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANKER_MODEL)
        return _reranker
    except Exception:
        return None


def rerank(query: str, docs: List[Tuple[str, dict]], top_k: int = 5) -> List[Tuple[float, dict]]:
    """
    docs: list of tuples (text, metadata)
    Returns list of (score, metadata+text) sorted desc
    """

    model = _init_reranker()

    pairs = [(query, text) for text, _ in docs]

    if not model:
        # fallback: naive stable scoring
        scored = [(1.0 - (i / max(1, len(docs))), {"text": docs[i][0], **docs[i][1]}) for i in range(min(top_k, len(docs)))]
        return scored

    scores = model.predict(pairs)

    scored = []
    for s, (text, meta) in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True):
        scored.append((float(s), {"text": text, **meta}))

    return scored[:top_k]
