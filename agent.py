import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
import certifi

# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

# SSL configuration
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# ============================================================
# LANGCHAIN / LANGGRAPH
# ============================================================

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from tools import tools


# ============================================================
# DATA DIRECTORY
# ============================================================

Path("data").mkdir(exist_ok=True)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

DEFAULT_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)

ALLOWED_MODELS = {
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a helpful Agentic AI assistant for college students.

Your job is to answer questions accurately and use the
available tools whenever they are required.

AVAILABLE TOOLS:

1. search_college_knowledge
   Use this for:
   - college notes
   - subjects
   - syllabus
   - previous year questions
   - academic material
   - college-related knowledge
   - information stored in the college knowledge base

2. search_uploaded_documents
   Use this when the user asks a question about a document
   uploaded in the current conversation.

3. remember_this
   Use this when the user explicitly asks you to remember
   something for future conversations.

4. recall_memory
   Use this when the user asks about something that was
   previously remembered.

5. web_search
   Use this when current or external information is required.

GENERAL RULES:

- Do not hallucinate information.
- Do not invent sources or citations.
- For college-related questions, prefer the college knowledge
  base.
- For questions about uploaded files, use the uploaded-document
  search tool.
- If the required information is unavailable, clearly say so.
- If a tool provides sources, use those sources when answering.
- Keep answers clear and useful.
- Do not use web search when the required information is already
  available in the college knowledge base or uploaded documents.
- Use tools when the question requires information from external
  knowledge.
"""


# ============================================================
# MODEL NORMALIZATION
# ============================================================

def normalize_model_name(
    model_name: str | None
) -> str:
    """
    Validate the model selected by the frontend.

    If the model is missing or invalid, use DEFAULT_MODEL.
    """

    if not model_name:
        return DEFAULT_MODEL

    model_name = model_name.strip()

    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL

    return model_name


# ============================================================
# CREATE LLM
# ============================================================

def create_llm(
    model_name: str | None = None
):
    """
    Create the Groq LLM.
    """

    selected_model = normalize_model_name(
        model_name
    )

    llm = ChatGroq(
        model=selected_model,
        temperature=0.3,
        streaming=True,
    )

    return llm


# ============================================================
# SQLITE CHECKPOINTER
# ============================================================

def create_checkpointer():
    """
    Create the SQLite checkpointer used by LangGraph.

    This stores conversation state based on thread_id.
    """

    db_path = Path(
        "data/langgraph_checkpoints.sqlite"
    )

    conn = sqlite3.connect(
        str(db_path),
        check_same_thread=False,
    )

    checkpointer = SqliteSaver(conn)

    return checkpointer


# ============================================================
# BUILD AGENT
# ============================================================

def build_agent(
    model_name: str | None = None
):
    """
    Build and compile a LangGraph agent.
    """

    selected_model = normalize_model_name(
        model_name
    )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    llm = create_llm(
        selected_model
    )

    # --------------------------------------------------------
    # Bind tools
    # --------------------------------------------------------

    llm_with_tools = llm.bind_tools(
        tools
    )

    # --------------------------------------------------------
    # Chatbot node
    # --------------------------------------------------------

    def chatbot_node(
        state: MessagesState
    ):
        """
        Main LLM node.

        Only the latest 10 messages are sent to the LLM
        to prevent the request payload from becoming too large.

        LangGraph/SQLite still keeps the complete conversation
        history in the checkpoint.
        """

        recent_messages = state["messages"][-10:]

        messages = [
            SystemMessage(
                content=SYSTEM_PROMPT
            )
        ] + recent_messages

        response = llm_with_tools.invoke(
            messages
        )

        return {
            "messages": [response]
        }

    # --------------------------------------------------------
    # Tool node
    # --------------------------------------------------------

    tool_node = ToolNode(
        tools
    )

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    workflow = StateGraph(
        MessagesState
    )

    workflow.add_node(
        "chatbot",
        chatbot_node
    )

    workflow.add_node(
        "tools",
        tool_node
    )

    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------

    workflow.add_edge(
        START,
        "chatbot"
    )

    workflow.add_conditional_edges(
        "chatbot",
        tools_condition
    )

    workflow.add_edge(
        "tools",
        "chatbot"
    )

    # --------------------------------------------------------
    # SQLite checkpoint
    # --------------------------------------------------------

    checkpointer = create_checkpointer()

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    agent = workflow.compile(
        checkpointer=checkpointer
    )

    return agent


# ============================================================
# AGENT CACHE
# ============================================================

_AGENT_CACHE = {}


def get_agent(
    model_name: str | None = None
):
    """
    Return a cached agent for the selected model.

    The agent is created only once for each model.
    """

    selected_model = normalize_model_name(
        model_name
    )

    if selected_model not in _AGENT_CACHE:

        _AGENT_CACHE[selected_model] = (
            build_agent(
                selected_model
            )
        )

    return _AGENT_CACHE[selected_model]


# ============================================================
# RUN AGENT
# ============================================================

def run_agent(
    question: str,
    thread_id: str,
    model_name: str | None = None,
):
    """
    Run the agent for a specific conversation thread.

    thread_id is important because SQLite checkpointing
    uses it to maintain conversation history.
    """

    agent = get_agent(
        model_name
    )

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        },
        config=config,
    )

    return response