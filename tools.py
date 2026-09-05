from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from rag.pipeline import (
    query_college_knowledge,
    query_uploaded_documents,
)


# ============================================================
# COLLEGE KNOWLEDGE RAG
# ============================================================

@tool
def search_college_knowledge(
    question: str,
) -> dict[str, Any]:
    """
    Search the college knowledge base.

    Use this tool for questions about:
    - college subjects
    - syllabus
    - notes
    - PYQs
    - academic material
    - college-specific information

    The tool returns an answer, sources and retrieved results.
    """

    result = query_college_knowledge(
        question=question
    )

    return result


# ============================================================
# UPLOADED DOCUMENT RAG
# ============================================================

@tool
def search_uploaded_documents(
    question: str,
    config: RunnableConfig,
) -> dict[str, Any]:
    """
    Search documents uploaded by the user in the current
    conversation.

    The current LangGraph thread_id is automatically obtained
    from the RunnableConfig.

    Do NOT ask the user for a thread_id.
    """

    configurable = config.get(
        "configurable",
        {}
    )

    thread_id = configurable.get(
        "thread_id"
    )

    if not thread_id:
        return {
            "answer": (
                "I cannot search the uploaded documents because "
                "the current conversation thread could not be identified."
            ),
            "sources": [],
            "results": [],
        }

    result = query_uploaded_documents(
        question=question,
        thread_id=thread_id,
    )

    return result


# ============================================================
# MEMORY
# ============================================================

@tool
def remember_this(
    information: str,
) -> str:
    """
    Remember important information provided by the user.

    Use this when the user explicitly asks the assistant
    to remember something.

    Memory persistence will be connected to the application's
    SQLite memory system separately.
    """

    return (
        "The information has been identified for memory storage: "
        f"{information}"
    )


@tool
def recall_memory(
    query: str,
) -> str:
    """
    Recall previously remembered information.

    Use this when the user asks about something that should
    have been remembered from an earlier conversation.

    Persistent memory lookup will be connected to the
    SQLite memory system separately.
    """

    return (
        "Memory lookup requested for: "
        f"{query}"
    )


# ============================================================
# WEB SEARCH
# ============================================================

@tool
def web_search(
    query: str,
) -> str:
    """
    Search the web for current or external information.

    Use this when the required information is not available
    in the college knowledge base or uploaded documents.

    Actual web-search integration will be connected separately.
    """

    return (
        "Web search requested for: "
        f"{query}"
    )


# ============================================================
# ALL TOOLS
# ============================================================

tools = [
    search_college_knowledge,
    search_uploaded_documents,
    remember_this,
    recall_memory,
    web_search,
]


# Backward compatibility
TOOLS = tools