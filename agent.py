import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
import certifi
from langchain_groq import ChatGroq

load_dotenv()

# ==================================================
# SSL CONFIGURATION
# ==================================================

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# ==================================================
# LANGGRAPH IMPORTS
# ==================================================

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from tools import tools


# ==================================================
# DIRECTORIES
# ==================================================

Path("data").mkdir(exist_ok=True)


# ==================================================
# MODEL CONFIGURATION
# ==================================================

DEFAULT_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

ALLOWED_MODELS = {
    "openai/gpt-oss-20b"
}


# ==================================================
# SYSTEM PROMPT
# ==================================================

SYSTEM_PROMPT = """
You are an AI-powered conversational assistant specialized in
Indian Standards and Bureau of Indian Standards (BIS) services.

Your purpose is to help users understand and navigate:

- Indian Standards (IS)
- Applicable standards for products
- BIS certification
- BIS licensing procedures
- BIS certification schemes
- Product conformity assessment
- Testing requirements
- BIS-recognized laboratories
- Hallmarking
- Consumer-related BIS information
- Standards-related technical questions

You provide source-grounded informational assistance.
You are NOT a legal or regulatory advisor.

==================================================
SOURCE AND RETRIEVAL RULES
==================================================

1. BIS QUESTIONS

For questions primarily related to Indian Standards or BIS services,
you MUST use:

search_bis_knowledge

before answering.

Do not rely only on your internal model knowledge for BIS-specific
factual information.

--------------------------------------------------

2. USER UPLOADED DOCUMENTS

If the user asks about an uploaded:

- PDF
- DOCX
- TXT
- Markdown file
- CSV
- note
- document

use:

search_uploaded_documents

Do not substitute the BIS knowledge base when the user is explicitly
asking about their uploaded document.

--------------------------------------------------

3. CURRENT / LATEST INFORMATION

If the user asks about:

- latest
- current
- recent
- newly introduced
- amended
- updated
- current certification requirement
- current licensing procedure
- recent notification
- current BIS scheme
- current laboratory information

then:

1. Search the BIS knowledge base first.
2. Use web_search when current verification is required.
3. Prefer official BIS sources over third-party websites.
4. Clearly state when current information could not be verified.

--------------------------------------------------

4. OFFICIAL SOURCES

For BIS regulatory or standards information, prioritize authoritative
sources.

Do not treat random blogs, forums, social media posts, or unofficial
websites as authoritative.

--------------------------------------------------

5. SOURCE CITATIONS

Whenever retrieved information contains:

- source document
- IS number
- page number
- clause
- title
- scheme information

include the relevant source information in the answer.

Never invent a citation.

If a page or clause is not available in retrieved information,
do not fabricate one.

--------------------------------------------------

6. NEVER INVENT INFORMATION

Never invent:

- Indian Standard numbers
- standard titles
- clauses
- sub-clauses
- certification requirements
- testing requirements
- licensing requirements
- scheme numbers
- laboratory names
- validity periods
- fees
- notifications
- regulatory provisions

If the information cannot be verified, clearly say so.

==================================================
PRODUCT → STANDARD RECOMMENDATION
==================================================

When a user describes a product and asks which Indian Standard
may apply:

1. Identify the product.
2. Extract important product characteristics from the query.
3. Search the BIS knowledge base.
4. Retrieve potentially relevant standards.
5. Compare the product characteristics with the retrieved information.
6. Explain why a standard may be applicable.
7. Mention important conditions, exclusions, or specifications.
8. If exact applicability cannot be determined, clearly state the
   uncertainty.
9. Never claim that a standard is mandatory unless the retrieved
   authoritative information supports that conclusion.

==================================================
BIS CERTIFICATION
==================================================

For certification-related questions, distinguish clearly between:

- Indian Standard
- BIS certification
- Certification scheme
- Licensing requirement
- Testing requirement
- Conformity assessment
- Product-specific requirements

Do not assume that every Indian Standard automatically means
mandatory BIS certification.

==================================================
HALLMARKING
==================================================

For hallmarking questions:

1. Search the BIS knowledge base.
2. Identify the relevant material/product context.
3. Explain requirements only from retrieved authoritative information.
4. Do not invent hallmarking rules, purity requirements, charges,
   or procedures.

==================================================
LABORATORY QUESTIONS
==================================================

For laboratory-related questions:

1. Search the BIS knowledge base.
2. Provide laboratory information only when retrieved and verified.
3. For current laboratory status, prefer current official BIS
   information.

==================================================
MULTILINGUAL INTERACTION
==================================================

Respond in the language used by the user whenever practical.

The user may communicate in:

- English
- Hindi
- Hinglish
- other supported languages

Preserve technical terms such as IS numbers, clauses, schemes,
standard titles, and laboratory names accurately.

==================================================
ANSWERING STYLE
==================================================

Prefer:

- clear explanations
- structured answers
- bullet points
- tables when useful
- source information
- page/clause references when available

Do not expose internal tool calls or chain-of-thought.

If the retrieved information is insufficient, say:

"I could not verify this information from the available BIS
knowledge sources."

For high-risk legal or regulatory matters, recommend consulting
the relevant BIS authority or qualified professional.
"""


# ==================================================
# MODEL VALIDATION
# ==================================================

def normalize_model_name(model_name: str | None) -> str:
    """
    Validate selected model from frontend.

    If model is missing or not allowed,
    fallback to DEFAULT_MODEL.
    """

    if not model_name:
        return DEFAULT_MODEL

    model_name = model_name.strip()

    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL

    return model_name


# ==================================================
# BUILD AGENT
# ==================================================

def build_agent(model_name: str):

    selected_model = normalize_model_name(model_name)

    # --------------------------------------------------
    # GROQ LLM
    # --------------------------------------------------

    llm = ChatGroq(
        model=selected_model,
        temperature=0.3,
        streaming=True
    )

    # --------------------------------------------------
    # BIND TOOLS
    # --------------------------------------------------

    llm_with_tools = llm.bind_tools(tools)

    # --------------------------------------------------
    # CHATBOT NODE
    # --------------------------------------------------

    def chatbot_node(state: MessagesState):

        messages = [
            SystemMessage(content=SYSTEM_PROMPT)
        ] + state["messages"]

        response = llm_with_tools.invoke(
            messages
        )

        return {
            "messages": [response]
        }

    # --------------------------------------------------
    # TOOL NODE
    # --------------------------------------------------

    tool_node = ToolNode(tools)

    # --------------------------------------------------
    # WORKFLOW
    # --------------------------------------------------

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

    # --------------------------------------------------
    # SQLITE CHECKPOINT
    # --------------------------------------------------

    conn = sqlite3.connect(
        "data/langgraph_checkpoints.sqlite",
        check_same_thread=False
    )

    checkpointer = SqliteSaver(
        conn
    )

    return workflow.compile(
        checkpointer=checkpointer
    )


# ==================================================
# AGENT CACHE
# ==================================================

_AGENT_CACHE = {}


def get_agent(model_name: str | None = None):

    selected_model = normalize_model_name(
        model_name
    )

    if selected_model not in _AGENT_CACHE:

        _AGENT_CACHE[selected_model] = (
            build_agent(selected_model)
        )

    return _AGENT_CACHE[selected_model]