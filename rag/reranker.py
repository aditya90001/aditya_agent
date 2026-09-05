from flashrank import Ranker, RerankRequest


# FlashRank model
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"

# Load the model once when this module is imported
ranker = Ranker(model_name=RERANK_MODEL)


def rerank_documents(
    query: str,
    documents: list,
    top_k: int = 5
):
    """
    Rerank retrieved documents using FlashRank.

    Parameters
    ----------
    query : str
        Original user query.

    documents : list
        Documents returned by hybrid retrieval.

    top_k : int
        Number of final documents to keep.

    Returns
    -------
    list
        Reranked documents with scores.
    """

    if not documents:
        return []

    passages = []

    for index, item in enumerate(documents):

        document = item["document"]

        passages.append(
            {
                "id": str(index),
                "text": document.page_content,
                "meta": document.metadata
            }
        )

    rerank_request = RerankRequest(
        query=query,
        passages=passages
    )

    results = ranker.rerank(rerank_request)

    final_results = []

    for result in results[:top_k]:

        original_index = int(result["id"])

        original_document = documents[original_index]["document"]

        final_results.append(
            {
                "document": original_document,
                "score": float(result["score"])
            }
        )

    return final_results