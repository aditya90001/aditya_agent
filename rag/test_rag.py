from rag.pipeline import rag_pipeline

from langchain_core.documents import Document


documents = [
    Document(
        page_content="""
        The minimum attendance requirement for students
        is 75 percent. Students with attendance below the
        required percentage may be subject to university
        rules regarding examination eligibility.
        """,
        metadata={
            "knowledge_type": "college",
            "source": "academic_rules.pdf",
            "page": 12,
            "section": "Attendance Rules",
        },
    ),

    Document(
        page_content="""
        The Training and Placement Cell coordinates
        placement activities, company visits, aptitude
        tests, technical interviews and placement drives.
        """,
        metadata={
            "knowledge_type": "college",
            "source": "placement_handbook.pdf",
            "page": 5,
            "section": "Placement Cell",
        },
    ),
]


chunks = rag_pipeline.ingest_documents(documents)

print(f"Indexed {len(chunks)} chunks.")


result = rag_pipeline.query(
    question="What is the minimum attendance requirement?"
)

print("\nANSWER:")
print(result["answer"])