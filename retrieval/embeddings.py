import os
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)


def get_embedding_model():
    """Return a HuggingFaceEmbeddings instance based on ENV config."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
