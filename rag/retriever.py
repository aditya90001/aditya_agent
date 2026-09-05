from collections import defaultdict

from rank_bm25 import BM25Okapi

from .vector_store import similarity_search


class HybridRetriever:

    def __init__(
        self,
        alpha: float = 0.7,
        dense_k: int = 15,
        bm25_k: int = 15,
    ):
        """
        alpha controls the importance of dense retrieval.

        alpha = 0.7
        means:
            70% dense
            30% BM25
        """

        self.alpha = alpha
        self.dense_k = dense_k
        self.bm25_k = bm25_k

        self.documents = []
        self.bm25 = None

    def build_bm25(self, documents):
        """
        Build BM25 index from documents.
        """

        self.documents = documents

        if not documents:
            self.bm25 = None
            return

        tokenized_documents = [
            document.page_content.lower().split()
            for document in documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def dense_search(
        self,
        query: str,
        filter: dict | None = None
    ):
        return similarity_search(
            query=query,
            k=self.dense_k,
            filter=filter
        )

    def bm25_search(self, query: str):
        if not self.bm25:
            return []

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for index in ranked_indices[:self.bm25_k]:

            results.append(
                (
                    self.documents[index],
                    float(scores[index])
                )
            )

        return results

    def retrieve(
        self,
        query: str,
        filter: dict | None = None,
        top_k: int = 15
    ):
        """
        Perform dense + BM25 retrieval and combine results.
        """

        dense_documents = self.dense_search(
            query=query,
            filter=filter
        )

        bm25_results = self.bm25_search(query)

        scores = defaultdict(float)
        documents = {}

        # Dense ranking
        for rank, document in enumerate(dense_documents):

            key = self._document_key(document)

            dense_score = 1.0 / (rank + 1)

            scores[key] += (
                self.alpha * dense_score
            )

            documents[key] = document

        # BM25 ranking
        for rank, (document, bm25_score) in enumerate(
            bm25_results
        ):

            key = self._document_key(document)

            bm25_rank_score = 1.0 / (rank + 1)

            scores[key] += (
                (1 - self.alpha) * bm25_rank_score
            )

            documents[key] = document

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        results = []

        for key, score in ranked[:top_k]:

            document = documents[key]

            results.append(
                {
                    "document": document,
                    "score": float(score)
                }
            )

        return results

    @staticmethod
    def _document_key(document):

        metadata = document.metadata

        source = metadata.get(
            "source",
            "unknown"
        )

        page = metadata.get(
            "page",
            ""
        )

        chunk_id = metadata.get(
            "chunk_id",
            ""
        )

        return f"{source}:{page}:{chunk_id}"