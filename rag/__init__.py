from .pipeline import (
    rag_pipeline,
    query_college_knowledge,
    query_uploaded_documents,
)

from .document_loader import load_document


def add_document_to_rag(
    file_path: str,
    thread_id: str,
) -> dict:

    documents = load_document(
        file_path=file_path,
        thread_id=thread_id,
    )

    if not documents:
        return {
            "success": False,
            "filename": file_path,
            "chunks": 0,
            "message": "No readable content found.",
        }

    chunks = rag_pipeline.ingest_documents(
        documents
    )

    return {
        "success": True,
        "filename": file_path,
        "documents": len(documents),
        "chunks": len(chunks),
    }


__all__ = [
    "rag_pipeline",
    "query_college_knowledge",
    "query_uploaded_documents",
    "add_document_to_rag",
]