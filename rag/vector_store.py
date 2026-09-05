from pathlib import Path

import chromadb
from langchain_chroma import Chroma

from .embeddings import get_embedding_model


CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "college_knowledge"


def get_vector_store() -> Chroma:
    """
    Return the persistent ChromaDB vector store.
    """

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    return vector_store


def add_documents(documents):
    """
    Add LangChain Document objects to ChromaDB.
    """

    if not documents:
        return []

    vector_store = get_vector_store()

    ids = vector_store.add_documents(documents)

    return ids


def similarity_search(
    query: str,
    k: int = 15,
    filter: dict | None = None
):
    """
    Perform dense vector similarity search.
    """

    vector_store = get_vector_store()

    return vector_store.similarity_search(
        query=query,
        k=k,
        filter=filter
    )


def similarity_search_with_score(
    query: str,
    k: int = 15,
    filter: dict | None = None
):
    """
    Perform dense retrieval and return similarity scores.
    """

    vector_store = get_vector_store()

    return vector_store.similarity_search_with_score(
        query=query,
        k=k,
        filter=filter
    )


def get_collection():
    """
    Return the underlying Chroma collection.
    Useful for diagnostics and collection statistics.
    """

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME
    )


def collection_count() -> int:
    """
    Return the number of vectors currently stored.
    """

    collection = get_collection()

    return collection.count()