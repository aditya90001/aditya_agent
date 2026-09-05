import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from .prompts import build_rag_prompt

load_dotenv()


DEFAULT_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)


def get_llm(model_name: str | None = None):
    """
    Create the Groq LLM.
    """

    selected_model = model_name or DEFAULT_MODEL

    return ChatGroq(
        model=selected_model,
        temperature=0.2,
        streaming=True,
    )


def generate_answer(
    question: str,
    context: str,
    model_name: str | None = None
) -> str:
    """
    Generate an answer using retrieved RAG context.
    """

    llm = get_llm(model_name)

    prompt = build_rag_prompt(
        question=question,
        context=context
    )

    response = llm.invoke(prompt)

    return response.content