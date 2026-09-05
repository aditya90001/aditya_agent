from pathlib import Path
import csv

from langchain_core.documents import Document

from pypdf import PdfReader
import docx2txt


def load_document(
    file_path: str,
    thread_id: str,
) -> list[Document]:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(
            path,
            thread_id
        )

    elif suffix == ".docx":
        return load_docx(
            path,
            thread_id
        )

    elif suffix in [".txt", ".md"]:
        return load_text(
            path,
            thread_id
        )

    elif suffix == ".csv":
        return load_csv(
            path,
            thread_id
        )

    else:
        raise ValueError(
            f"Unsupported file type: {suffix}"
        )


# ============================================================
# PDF
# ============================================================

def load_pdf(
    path: Path,
    thread_id: str,
) -> list[Document]:

    reader = PdfReader(
        str(path)
    )

    documents = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if not text or not text.strip():
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "knowledge_type": "user",
                    "thread_id": thread_id,
                    "source": path.name,
                    "page": page_number,
                    "file_type": "pdf",
                },
            )
        )

    return documents


# ============================================================
# DOCX
# ============================================================

def load_docx(
    path: Path,
    thread_id: str,
) -> list[Document]:

    text = docx2txt.process(
        str(path)
    )

    if not text or not text.strip():
        return []

    return [
        Document(
            page_content=text,
            metadata={
                "knowledge_type": "user",
                "thread_id": thread_id,
                "source": path.name,
                "page": 1,
                "file_type": "docx",
            },
        )
    ]


# ============================================================
# TXT / MD
# ============================================================

def load_text(
    path: Path,
    thread_id: str,
) -> list[Document]:

    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    if not text.strip():
        return []

    return [
        Document(
            page_content=text,
            metadata={
                "knowledge_type": "user",
                "thread_id": thread_id,
                "source": path.name,
                "page": 1,
                "file_type": path.suffix.lower().replace(
                    ".",
                    ""
                ),
            },
        )
    ]


# ============================================================
# CSV
# ============================================================

def load_csv(
    path: Path,
    thread_id: str,
) -> list[Document]:

    documents = []

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore",
        newline="",
    ) as file:

        reader = csv.reader(file)

        rows = list(reader)

    if not rows:
        return []

    header = rows[0]

    for row_number, row in enumerate(
        rows[1:],
        start=2
    ):

        values = []

        for column, value in zip(
            header,
            row
        ):
            values.append(
                f"{column}: {value}"
            )

        text = "\n".join(values)

        if not text.strip():
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "knowledge_type": "user",
                    "thread_id": thread_id,
                    "source": path.name,
                    "page": row_number,
                    "file_type": "csv",
                },
            )
        )

    return documents