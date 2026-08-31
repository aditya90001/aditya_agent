import math

from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from database import save_memory, search_memory

from rag import (
    retrieve_from_rag,
    retrieve_bis_knowledge
)

load_dotenv()


# --------------------------------------------------
# CURRENT THREAD
# --------------------------------------------------

CURRENT_THREAD_ID = "default"


def set_current_thread_id(thread_id: str):

    global CURRENT_THREAD_ID

    CURRENT_THREAD_ID = thread_id


# --------------------------------------------------
# WEB SEARCH
# --------------------------------------------------

web_search = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)


# ==================================================
# CALCULATOR
# ==================================================

@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """

    try:

        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(
            expression,
            {"__builtins__": {}},
            allowed
        )

        return str(result)

    except Exception as e:

        return f"Calculation error: {str(e)}"


# ==================================================
# USER UPLOADED DOCUMENT SEARCH
# ==================================================

@tool
def search_uploaded_documents(query: str) -> str:
    """
    Search uploaded user documents.

    Use this when the user asks about their uploaded
    PDFs, DOCX, TXT, notes, files, or documents.
    """

    return retrieve_from_rag(
        query=query,
        thread_id=CURRENT_THREAD_ID
    )


# ==================================================
# GDG INTERNAL KNOWLEDGE SEARCH
# ==================================================

@tool
def search_bis_knowledge(query: str) -> str:
    """
    Search the authorized BIS knowledge base.

    Use this for:
    - Indian Standards
    - Product standards
    - BIS certification
    - BIS licensing
    - Testing requirements
    - BIS schemes
    - Hallmarking
    - Laboratories
    - Consumer information
    """

    return retrieve_bis_knowledge(query=query)

# ==================================================
# LONG TERM MEMORY
# ==================================================

@tool
def remember_this(memory: str) -> str:
    """
    Save an important user preference or fact
    into long-term memory.
    """

    return save_memory(
        thread_id=CURRENT_THREAD_ID,
        memory=memory
    )


@tool
def recall_memory(query: str) -> str:
    """
    Recall saved long-term memories about
    the user or this conversation.
    """

    return search_memory(
        thread_id=CURRENT_THREAD_ID,
        query=query
    )


# ==================================================
# ALL TOOLS
# ==================================================

tools = [

    calculator,

    search_uploaded_documents,

    search_bis_knowledge,       # NEW

    remember_this,

    recall_memory,

    web_search

]