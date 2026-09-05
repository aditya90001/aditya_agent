from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL = "BAAI/bge-m3"


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Create and cache the BGE-M3 embedding model.

    The model is loaded only once for the application.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )