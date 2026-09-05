def format_citations(results: list[dict]) -> str:
    """
    Create citation text from retrieved documents.
    """

    if not results:
        return ""

    citations = []

    seen = set()

    for index, result in enumerate(results, start=1):

        document = result["document"]
        metadata = document.metadata

        source = metadata.get(
            "source",
            "Unknown source"
        )

        page = metadata.get(
            "page"
        )

        section = metadata.get(
            "section"
        )

        key = (
            source,
            page,
            section
        )

        if key in seen:
            continue

        seen.add(key)

        citation = f"[{index}] {source}"

        if page is not None:
            citation += f", page {page}"

        if section:
            citation += f", section {section}"

        citations.append(citation)

    return "\n".join(citations)