import json
from pathlib import Path

from langchain_core.documents import Document

from rag.chunker import split_documents
from rag.vector_store import add_documents, collection_count


KNOWLEDGE_BASE_DIR = Path("knowledge_base")


def load_knowledge_base():
    documents = []

    json_files = list(KNOWLEDGE_BASE_DIR.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            "No JSON files found in knowledge_base/"
        )

    for json_file in json_files:

        print(f"\nLoading: {json_file}")

        with open(
            json_file,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        for item in data:

            text = item.get("text", "").strip()

            if not text:
                continue

            metadata = item.get("metadata", {}).copy()

            metadata["knowledge_type"] = "college"
            metadata["source_file"] = json_file.name
            metadata["document_id"] = item.get("id")

            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata
                )
            )

    return documents


def main():

    print("=" * 60)
    print("COLLEGE KNOWLEDGE BASE INGESTION")
    print("=" * 60)

    # 1. Load JSON
    documents = load_knowledge_base()

    print(f"\nDocuments loaded: {len(documents)}")

    # 2. Chunk documents
    chunks = split_documents(documents)

    print(f"Chunks created: {len(chunks)}")

    # 3. Add to ChromaDB
    print("\nAdding documents to ChromaDB...")

    ids = add_documents(chunks)

    print(f"Vectors added: {len(ids)}")

    # 4. Verify collection
    count = collection_count()

    print(f"\nChromaDB collection count: {count}")

    print("\n" + "=" * 60)
    print("INGESTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()