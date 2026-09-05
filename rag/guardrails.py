from typing import Any


DEFAULT_MIN_RELEVANCE_SCORE = 0.15


def validate_retrieval(
    results: list[dict[str, Any]],
    min_score: float = DEFAULT_MIN_RELEVANCE_SCORE
) -> dict:
    """
    Validate retrieved documents before sending them
    to the LLM.
    """

    if not results:
        return {
            "allowed": False,
            "reason": "No relevant documents were retrieved.",
            "results": []
        }

    valid_results = [
        result
        for result in results
        if result.get("score", 0) >= min_score
    ]

    if not valid_results:
        return {
            "allowed": False,
            "reason": "Retrieved documents were not sufficiently relevant.",
            "results": []
        }

    return {
        "allowed": True,
        "reason": "Relevant documents found.",
        "results": valid_results
    }


def build_context(
    results: list[dict[str, Any]]
) -> str:
    """
    Convert retrieved documents into a clean context
    for the LLM.
    """

    if not results:
        return ""

    context_parts = []

    for index, result in enumerate(results, start=1):

        document = result["document"]

        metadata = document.metadata

        source = metadata.get(
            "source",
            "Unknown source"
        )

        page = metadata.get(
            "page",
            "Unknown page"
        )

        context_parts.append(
            f"""
--- SOURCE {index} ---
Source: {source}
Page: {page}

Content:
{document.page_content}
"""
        )

    return "\n".join(context_parts)


def check_answer_grounding(
    answer: str,
    context: str
) -> bool:
    """
    Basic grounding check.

    This is intentionally conservative.

    A more advanced semantic grounding check can be
    added later.
    """

    if not answer.strip():
        return False

    if not context.strip():
        return False

    return True


def no_information_response() -> str:
    """
    Response used when the RAG system cannot verify
    an answer from the college knowledge base.
    """

    return (
        "I couldn't verify this information from the "
        "available college documents. Please provide a "
        "relevant document or check the official college "
        "source."
    )