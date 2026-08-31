import os
from pathlib import Path

from rag import retrieve_bis_knowledge, add_document_to_rag, retrieve_from_rag


def test_bis_retrieval_smoke():
    # Basic smoke: should return a string (even if no results)
    res = retrieve_bis_knowledge("hallmarking", k=3)
    assert isinstance(res, str)


def test_thread_isolation(tmp_path):
    # Create a simple text file and ingest under thread A
    p = tmp_path / "sample.txt"
    p.write_text("This is a secret doc about unique-token-12345.")

    thread_a = "thread-A"
    thread_b = "thread-B"

    add_document_to_rag(str(p), thread_id=thread_a)

    # Search from thread B should not return the content of thread A
    res_b = retrieve_from_rag("unique-token-12345", thread_id=thread_b, k=3)
    assert "unique-token-12345" not in res_b.lower()
