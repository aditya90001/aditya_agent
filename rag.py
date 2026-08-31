
from pathlib import Path
from typing import List
import os
import json
import certifi

from dotenv import load_dotenv

# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

# ============================================================
# SSL CONFIGURATION
# ============================================================

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# ============================================================
# LANGCHAIN IMPORTS
# ============================================================

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pypdf import PdfReader
import docx2txt

# ============================================================
# RETRIEVAL IMPORTS
# ============================================================

from retrieval.embeddings import get_embedding_model
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import rerank
from retrieval.query_processor import preprocess_query

# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================

HYBRID_ALPHA = float(
    os.getenv("HYBRID_ALPHA", "0.7")
)

VECTOR_TOP_K = int(
    os.getenv("VECTOR_TOP_K", "15")
)

FINAL_TOP_K = int(
    os.getenv("FINAL_TOP_K", "5")
)

RETRIEVAL_SCORE_THRESHOLD = float(
    os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.35")
)

# ============================================================
# DIRECTORIES
# ============================================================

Path("uploads").mkdir(exist_ok=True)
Path("chroma_db").mkdir(exist_ok=True)
Path("bis_knowledge").mkdir(exist_ok=True)

# ============================================================
# EMBEDDING MODEL
# ============================================================

embeddings = get_embedding_model()

# ============================================================
# HYBRID RETRIEVER
# ============================================================

hybrid_retriever = HybridRetriever(
    alpha=HYBRID_ALPHA
)

# ============================================================
# CHROMA VECTOR DATABASE
# ============================================================

vectorstore = Chroma(
    collection_name="bis_knowledge",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)

# ============================================================
# TEXT SPLITTER
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150
)


# ============================================================
# FILE TEXT EXTRACTION
# ============================================================

def read_file_text(file_path: str) -> str:
    """
    Read text from supported file formats.

    Supported:
    PDF
    DOCX
    JSON
    TXT
    MD
    PY
    CSV
    """

    path = Path(file_path)
    suffix = path.suffix.lower()

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if suffix == ".pdf":

        reader = PdfReader(file_path)

        text_parts = []

        for page in reader.pages:

            page_text = page.extract_text() or ""

            if page_text.strip():
                text_parts.append(page_text)

        return "\n".join(text_parts)

    # --------------------------------------------------------
    # DOCX
    # --------------------------------------------------------

    if suffix == ".docx":

        return docx2txt.process(file_path)

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    if suffix == ".json":

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        return json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # TEXT FILES
    # --------------------------------------------------------

    if suffix in [
        ".txt",
        ".md",
        ".py",
        ".csv"
    ]:

        return path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    raise ValueError(
        "Unsupported file type. "
        "Supported formats: PDF, DOCX, JSON, "
        "TXT, MD, PY, CSV."
    )


# ============================================================
# PDF PAGE READER
# ============================================================

def read_pdf_pages(
    file_path: str
) -> List[Document]:

    reader = PdfReader(file_path)

    documents = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text() or ""

        if not text.strip():
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "page": page_number,
                    "source": Path(file_path).name
                }
            )
        )

    return documents


# ============================================================
# USER UPLOADED DOCUMENT RAG
# ============================================================

def add_document_to_rag(
    file_path: str,
    thread_id: str
):

    path = Path(file_path)
    suffix = path.suffix.lower()

    docs: List[Document] = []

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if suffix == ".pdf":

        page_documents = read_pdf_pages(
            file_path
        )

        for page_doc in page_documents:

            chunks = splitter.split_text(
                page_doc.page_content
            )

            for chunk in chunks:

                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "knowledge_type": "user",
                            "thread_id": thread_id,
                            "source": path.name,
                            "page": page_doc.metadata["page"]
                        }
                    )
                )

    # --------------------------------------------------------
    # OTHER FILES
    # --------------------------------------------------------

    else:

        text = read_file_text(
            file_path
        )

        if not text.strip():

            raise ValueError(
                "No text could be extracted "
                "from this file."
            )

        chunks = splitter.split_text(
            text
        )

        for chunk in chunks:

            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "knowledge_type": "user",
                        "thread_id": thread_id,
                        "source": path.name,
                        "page": None
                    }
                )
            )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not docs:

        raise ValueError(
            "No usable content was extracted "
            "from the document."
        )

    # --------------------------------------------------------
    # CHROMA
    # --------------------------------------------------------

    vectorstore.add_documents(
        docs
    )

    # --------------------------------------------------------
    # HYBRID INDEX
    # --------------------------------------------------------

    try:

        hybrid_retriever.index_documents(
            docs
        )

    except Exception as e:

        print(
            f"Hybrid indexing warning: {e}"
        )

    return {
        "filename": path.name,
        "chunks": len(docs)
    }


# ============================================================
# USER DOCUMENT RETRIEVAL
# ============================================================

def retrieve_from_rag(
    query: str,
    thread_id: str,
    k: int = 4
) -> str:

    processed_query = preprocess_query(
        query
    )

    candidates = hybrid_retriever.hybrid_search(

        lambda q, k, filter:
        vectorstore.similarity_search(
            q,
            k=k,
            filter=filter
        ),

        processed_query,

        knowledge_type="user",

        thread_id=thread_id,

        vector_k=max(
            k,
            VECTOR_TOP_K
        ),

        top_k=max(
            k,
            VECTOR_TOP_K
        ),

        alpha=HYBRID_ALPHA
    )

    if not candidates:

        return (
            "No relevant uploaded "
            "document content found."
        )

    # --------------------------------------------------------
    # RERANK
    # --------------------------------------------------------

    docs_for_rerank = [
        (
            doc.page_content,
            doc.metadata or {}
        )
        for score, doc in candidates
    ]

    reranked = rerank(
        processed_query,
        docs_for_rerank,
        top_k=k
    )

    if not reranked:

        return (
            "No relevant uploaded "
            "document content found."
        )

    # --------------------------------------------------------
    # NORMALIZE SCORES
    # --------------------------------------------------------

    scores = [
        score
        for score, _ in reranked
    ]

    max_s = max(scores) if scores else 0.0
    min_s = min(scores) if scores else 0.0

    def normalize(score):

        if max_s == min_s:

            return (
                1.0
                if max_s > 0
                else 0.0
            )

        return (
            (score - min_s)
            / (max_s - min_s)
        )

    # --------------------------------------------------------
    # FORMAT RESULTS
    # --------------------------------------------------------

    formatted = []
    seen = set()

    for rank, (score, payload) in enumerate(
        reranked,
        start=1
    ):

        text = payload.get(
            "text",
            ""
        )

        meta = {
            key: value
            for key, value in payload.items()
            if key != "text"
        }

        key = (
            meta.get("source"),
            meta.get("page"),
            text.strip()[:300]
        )

        if key in seen:
            continue

        seen.add(key)

        relevance = normalize(
            score
        )

        if (
            rank == 1
            and relevance < RETRIEVAL_SCORE_THRESHOLD
        ):

            return (
                "No sufficiently relevant "
                "information was found in "
                "the uploaded documents."
            )

        formatted.append(
            {
                "source": meta.get(
                    "source",
                    "uploaded document"
                ),
                "page": meta.get(
                    "page",
                    "N/A"
                ),
                "relevance": round(
                    float(relevance),
                    3
                ),
                "content": text
            }
        )

        if len(formatted) >= k:
            break

    # --------------------------------------------------------
    # BUILD OUTPUT
    # --------------------------------------------------------

    if not formatted:

        return (
            "No relevant uploaded "
            "document content found."
        )

    out_parts = []

    for i, item in enumerate(
        formatted,
        start=1
    ):

        out_parts.append(
            f"""
[USER DOCUMENT SOURCE {i}]

Document: {item['source']}
Page: {item['page']}
Relevance: {item['relevance']}

Content:
{item['content']}
"""
        )

    return "\n\n".join(
        out_parts
    )


# ============================================================
# JSON → BIS DOCUMENT
# ============================================================

def json_record_to_document(
    record: dict,
    file_name: str
) -> Document:

    lab_id = record.get(
        "lab_id",
        "Not available"
    )

    lab_name = record.get(
        "lab_name",
        "Not available"
    )

    location = record.get(
        "location",
        "Not available"
    )

    region = record.get(
        "region",
        "Not available"
    )

    accreditation = record.get(
        "accreditation",
        "Not available"
    )

    # --------------------------------------------------------
    # CONTACT
    # --------------------------------------------------------

    contact = record.get(
        "contact",
        {}
    )

    if not isinstance(
        contact,
        dict
    ):
        contact = {}

    email = contact.get(
        "email",
        "Not available"
    )

    phone = contact.get(
        "phone",
        "Not available"
    )

    nodal_officer = contact.get(
        "nodal_officer",
        "Not available"
    )

    # --------------------------------------------------------
    # TESTING CAPABILITIES
    # --------------------------------------------------------

    capabilities = record.get(
        "testing_capabilities",
        []
    )

    capability_parts = []

    if isinstance(
        capabilities,
        list
    ):

        for capability in capabilities:

            if not isinstance(
                capability,
                dict
            ):
                continue

            standard = capability.get(
                "standard",
                "Not available"
            )

            equipment = capability.get(
                "equipment",
                "Not available"
            )

            turnaround = capability.get(
                "turnaround_days",
                "Not available"
            )

            capability_parts.append(
                f"""
Standard: {standard}
Equipment: {equipment}
Turnaround Time: {turnaround} days
"""
            )

    capabilities_text = (
        "\n".join(capability_parts)
        if capability_parts
        else "No testing capabilities listed."
    )

    # --------------------------------------------------------
    # SEARCHABLE DOCUMENT
    # --------------------------------------------------------

    lab_text = f"""
BIS LABORATORY INFORMATION

Lab ID:
{lab_id}

Lab Name:
{lab_name}

Location:
{location}

Region:
{region}

Accreditation:
{accreditation}

TESTING CAPABILITIES:

{capabilities_text}

CONTACT INFORMATION:

Email:
{email}

Phone:
{phone}

Nodal Officer:
{nodal_officer}
"""

    return Document(
        page_content=lab_text.strip(),
        metadata={
            "knowledge_type": "bis",
            "source": file_name,
            "document_type": "BIS_LAB",
            "lab_id": str(lab_id),
            "lab_name": str(lab_name),
            "location": str(location),
            "region": str(region),
            "accreditation": str(accreditation)
        }
    )


# ============================================================
# ADD BIS KNOWLEDGE
# ============================================================

def add_bis_knowledge():

    knowledge_path = Path(
        "bis_knowledge"
    )

    if not knowledge_path.exists():

        raise ValueError(
            "bis_knowledge folder does not exist."
        )

    all_docs: List[Document] = []

    processed_files = 0
    skipped_files = 0

    # --------------------------------------------------------
    # FIND ALL FILES RECURSIVELY
    # --------------------------------------------------------

    files = list(
        knowledge_path.rglob("*")
    )

    for file_path in files:

        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        # ====================================================
        # JSON
        # ====================================================

        if suffix == ".json":

            print(
                f"Processing JSON: {file_path}"
            )

            try:

                with open(
                    file_path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    data = json.load(f)

            except Exception as e:

                print(
                    f"Error reading JSON "
                    f"{file_path.name}: {e}"
                )

                skipped_files += 1
                continue

            # ------------------------------------------------
            # JSON CAN BE LIST OR SINGLE OBJECT
            # ------------------------------------------------

            if isinstance(
                data,
                list
            ):

                records = data

            elif isinstance(
                data,
                dict
            ):

                records = [data]

            else:

                print(
                    f"Unsupported JSON structure: "
                    f"{file_path.name}"
                )

                skipped_files += 1
                continue

            # ------------------------------------------------
            # CREATE DOCUMENT FOR EACH RECORD
            # ------------------------------------------------

            for record in records:

                if not isinstance(
                    record,
                    dict
                ):
                    continue

                document = (
                    json_record_to_document(
                        record,
                        file_path.name
                    )
                )

                # ------------------------------------------------
                # SPLIT LARGE JSON RECORD
                # ------------------------------------------------

                chunks = splitter.split_text(
                    document.page_content
                )

                for chunk in chunks:

                    metadata = dict(
                        document.metadata
                    )

                    all_docs.append(
                        Document(
                            page_content=chunk,
                            metadata=metadata
                        )
                    )

            processed_files += 1

        # ====================================================
        # PDF
        # ====================================================

        elif suffix == ".pdf":

            print(
                f"Processing PDF: {file_path}"
            )

            try:

                page_documents = (
                    read_pdf_pages(
                        str(file_path)
                    )
                )

                for page_doc in page_documents:

                    chunks = splitter.split_text(
                        page_doc.page_content
                    )

                    for chunk in chunks:

                        all_docs.append(
                            Document(
                                page_content=chunk,
                                metadata={
                                    "knowledge_type": "bis",
                                    "source": file_path.name,
                                    "page": page_doc.metadata.get(
                                        "page"
                                    ),
                                    "document_type": "BIS"
                                }
                            )
                        )

                processed_files += 1

            except Exception as e:

                print(
                    f"Error reading PDF "
                    f"{file_path.name}: {e}"
                )

                skipped_files += 1

        # ====================================================
        # TXT / MD
        # ====================================================

        elif suffix in [
            ".txt",
            ".md"
        ]:

            print(
                f"Processing text file: "
                f"{file_path}"
            )

            try:

                text = read_file_text(
                    str(file_path)
                )

                if not text.strip():

                    skipped_files += 1
                    continue

                chunks = splitter.split_text(
                    text
                )

                for chunk in chunks:

                    all_docs.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "knowledge_type": "bis",
                                "source": file_path.name,
                                "page": None,
                                "document_type": "BIS"
                            }
                        )
                    )

                processed_files += 1

            except Exception as e:

                print(
                    f"Error reading "
                    f"{file_path.name}: {e}"
                )

                skipped_files += 1

        # ====================================================
        # UNSUPPORTED FILE
        # ====================================================

        else:

            continue

    # ========================================================
    # NO DOCUMENTS
    # ========================================================

    if not all_docs:

        return {
            "message": (
                "No BIS knowledge documents "
                "were found."
            ),
            "files_processed": processed_files,
            "files_skipped": skipped_files,
            "chunks": 0
        }

    # ========================================================
    # ADD TO CHROMA
    # ========================================================

    print(
        f"\nAdding {len(all_docs)} "
        f"chunks to Chroma..."
    )

    vectorstore.add_documents(
        all_docs
    )

    print(
        "Chroma ingestion completed."
    )

    # ========================================================
    # HYBRID INDEX
    # ========================================================

    try:

        hybrid_retriever.index_documents(
            all_docs
        )

        print(
            "Hybrid indexing completed."
        )

    except Exception as e:

        print(
            f"Hybrid indexing warning: {e}"
        )

    # ========================================================
    # RESULT
    # ========================================================

    return {
        "message": (
            "BIS knowledge added successfully."
        ),
        "files_processed": processed_files,
        "files_skipped": skipped_files,
        "chunks": len(all_docs)
    }


# ============================================================
# BIS KNOWLEDGE RETRIEVAL
# ============================================================

def retrieve_bis_knowledge(
    query: str,
    k: int = 5
) -> str:

    """
    BIS retrieval pipeline:

    Query
      ↓
    Preprocessing
      ↓
    Hybrid Retrieval
      ↓
    Reranking
      ↓
    Deduplication
      ↓
    Relevance Filtering
      ↓
    Final Context
    """

    # --------------------------------------------------------
    # QUERY PREPROCESSING
    # --------------------------------------------------------

    processed_query = preprocess_query(
        query
    )

    # --------------------------------------------------------
    # HYBRID SEARCH
    # --------------------------------------------------------

    candidates = (
        hybrid_retriever.hybrid_search(

            lambda q, k, filter:
            vectorstore.similarity_search(
                q,
                k=k,
                filter=filter
            ),

            processed_query,

            knowledge_type="bis",

            vector_k=max(
                k,
                VECTOR_TOP_K
            ),

            top_k=max(
                k,
                VECTOR_TOP_K
            ),

            alpha=HYBRID_ALPHA
        )
    )

    if not candidates:

        return (
            "No relevant BIS knowledge "
            "was found."
        )

    # --------------------------------------------------------
    # PREPARE RERANKING INPUT
    # --------------------------------------------------------

    docs_for_rerank = [
        (
            doc.page_content,
            doc.metadata or {}
        )
        for score, doc in candidates
    ]

    # --------------------------------------------------------
    # RERANK
    # --------------------------------------------------------

    reranked = rerank(
        processed_query,
        docs_for_rerank,
        top_k=k
    )

    if not reranked:

        return (
            "No relevant BIS knowledge "
            "was found."
        )

    # --------------------------------------------------------
    # SCORE NORMALIZATION
    # --------------------------------------------------------

    scores = [
        score
        for score, _ in reranked
    ]

    max_s = max(
        scores
    ) if scores else 0.0

    min_s = min(
        scores
    ) if scores else 0.0

    def normalize(
        score
    ):

        if max_s == min_s:

            return (
                1.0
                if max_s > 0
                else 0.0
            )

        return (
            (score - min_s)
            / (max_s - min_s)
        )

    # --------------------------------------------------------
    # FORMAT RESULTS
    # --------------------------------------------------------

    formatted = []
    seen = set()

    for rank, (
        score,
        payload
    ) in enumerate(
        reranked,
        start=1
    ):

        text = payload.get(
            "text",
            ""
        )

        meta = {
            key: value
            for key, value
            in payload.items()
            if key != "text"
        }

        # ----------------------------------------------------
        # DEDUPLICATION
        # ----------------------------------------------------

        key = (
            meta.get("source"),
            meta.get("page"),
            meta.get("lab_id"),
            text.strip()[:300]
        )

        if key in seen:
            continue

        seen.add(key)

        # ----------------------------------------------------
        # RELEVANCE
        # ----------------------------------------------------

        relevance = normalize(
            score
        )

        # Only reject if top result itself is weak
        if (
            rank == 1
            and relevance
            < RETRIEVAL_SCORE_THRESHOLD
        ):

            return (
                "No sufficiently relevant "
                "information was found in "
                "the BIS knowledge base."
            )

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        source = meta.get(
            "source",
            "BIS knowledge base"
        )

        page = meta.get(
            "page",
            "N/A"
        )

        standard_number = meta.get(
            "standard_number",
            "Not available"
        )

        title = meta.get(
            "title",
            "Not available"
        )

        clause = meta.get(
            "clause",
            "Not available"
        )

        lab_id = meta.get(
            "lab_id",
            "Not available"
        )

        lab_name = meta.get(
            "lab_name",
            "Not available"
        )

        location = meta.get(
            "location",
            "Not available"
        )

        region = meta.get(
            "region",
            "Not available"
        )

        formatted.append(
            {
                "source": source,
                "page": page,
                "standard_number": standard_number,
                "title": title,
                "clause": clause,
                "lab_id": lab_id,
                "lab_name": lab_name,
                "location": location,
                "region": region,
                "relevance": round(
                    float(relevance),
                    3
                ),
                "content": text
            }
        )

        if len(formatted) >= k:
            break

    # --------------------------------------------------------
    # NO RESULTS
    # --------------------------------------------------------

    if not formatted:

        return (
            "No relevant BIS knowledge "
            "was found."
        )

    # --------------------------------------------------------
    # BUILD FINAL CONTEXT
    # --------------------------------------------------------

    out_parts = []

    for i, item in enumerate(
        formatted,
        start=1
    ):

        out_parts.append(
            f"""
[BIS SOURCE {i}]

Source:
{item['source']}

Lab ID:
{item['lab_id']}

Lab Name:
{item['lab_name']}

Location:
{item['location']}

Region:
{item['region']}

Page:
{item['page']}

Standard:
{item['standard_number']}

Title:
{item['title']}

Clause:
{item['clause']}

Relevance:
{item['relevance']}

Content:
{item['content']}
"""
        )

    return "\n\n".join(
        out_parts
    )

