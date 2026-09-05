from typing import Any

from langchain_core.documents import Document

from .chunker import split_documents
from .citations import format_citations
from .generator import generate_answer
from .guardrails import (
    build_context,
    no_information_response,
    validate_retrieval,
)
from .reranker import rerank_documents
from .retriever import HybridRetriever
from .vector_store import add_documents


# Retrieval configuration
HYBRID_ALPHA = 0.7

DENSE_TOP_K = 15
BM25_TOP_K = 15

HYBRID_TOP_K = 15
FINAL_TOP_K = 5


class RAGPipeline:

    def __init__(
        self,
        alpha: float = HYBRID_ALPHA
    ):

        self.retriever = HybridRetriever(
            alpha=alpha,
            dense_k=DENSE_TOP_K,
            bm25_k=BM25_TOP_K,
        )

    # --------------------------------------------------
    # INGESTION
    # --------------------------------------------------

    def ingest_documents(
        self,
        documents: list[Document]
    ) -> list[Document]:

        if not documents:
            return []

        chunks = split_documents(documents)

        if not chunks:
            return []

        add_documents(chunks)

        # Build/update BM25 index
        self.retriever.build_bm25(chunks)

        return chunks

    # --------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------

    def retrieve(
        self,
        query: str,
        metadata_filter: dict | None = None
    ) -> list[dict[str, Any]]:

        results = self.retriever.retrieve(
            query=query,
            filter=metadata_filter,
            top_k=HYBRID_TOP_K,
        )

        return results

    # --------------------------------------------------
    # RERANKING
    # --------------------------------------------------

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:

        return rerank_documents(
            query=query,
            documents=results,
            top_k=FINAL_TOP_K,
        )

    # --------------------------------------------------
    # COMPLETE QUERY
    # --------------------------------------------------

    def query(
        self,
        question: str,
        metadata_filter: dict | None = None,
        model_name: str | None = None
    ) -> dict:

        # ----------------------------------------------
        # 1. Retrieve
        # ----------------------------------------------

        retrieved = self.retrieve(
            query=question,
            metadata_filter=metadata_filter,
        )

        if not retrieved:

            return {
                "answer": no_information_response(),
                "sources": [],
                "results": [],
            }

        # ----------------------------------------------
        # 2. Rerank
        # ----------------------------------------------

        reranked = self.rerank(
            query=question,
            results=retrieved,
        )

        # ----------------------------------------------
        # 3. Guardrails
        # ----------------------------------------------

        validation = validate_retrieval(
            reranked
        )

        if not validation["allowed"]:

            return {
                "answer": no_information_response(),
                "sources": [],
                "results": reranked,
            }

        valid_results = validation["results"]

        # ----------------------------------------------
        # 4. Build context
        # ----------------------------------------------

        context = build_context(
            valid_results
        )

        # ----------------------------------------------
        # 5. Generate answer
        # ----------------------------------------------

        answer = generate_answer(
            question=question,
            context=context,
            model_name=model_name,
        )

        # ----------------------------------------------
        # 6. Citations
        # ----------------------------------------------

        citations = format_citations(
            valid_results
        )

        if citations:

            answer = (
                f"{answer}\n\n"
                f"Sources:\n"
                f"{citations}"
            )

        # ----------------------------------------------
        # 7. Return structured result
        # ----------------------------------------------

        return {
            "answer": answer,
            "sources": citations,
            "results": valid_results,
        }


# ------------------------------------------------------
# Global pipeline instance
# ------------------------------------------------------

rag_pipeline = RAGPipeline()


def query_college_knowledge(
    question: str,
    model_name: str | None = None
) -> dict:

    return rag_pipeline.query(
        question=question,
        metadata_filter={
            "knowledge_type": "college"
        },
        model_name=model_name,
    )


def query_uploaded_documents(
    question: str,
    thread_id: str,
    model_name: str | None = None
) -> dict:

    return rag_pipeline.query(
        question=question,
        metadata_filter={
            "$and": [
                {
                    "knowledge_type": "user"
                },
                {
                    "thread_id": thread_id
                }
            ]
        },
        model_name=model_name,
    )