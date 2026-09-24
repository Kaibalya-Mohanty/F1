import os

from dotenv import load_dotenv
from groq import Groq

from .retriever import (
    load_documents,
    build_index,
    search,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY was not found in .env"
    )


# ============================================================
# GROQ
# ============================================================

client = Groq(
    api_key=GROQ_API_KEY
)

MODEL = "openai/gpt-oss-120b"


# ============================================================
# RAG INITIALIZATION
# ============================================================

print("Initializing F1 RAG system...")

documents = load_documents()

index = build_index(
    documents
)

print("F1 RAG system ready.")


# ============================================================
# RAG AGENT
# ============================================================

def answer_question(
    query,
    top_k=3,
):
    """
    Answer an F1 knowledge question using retrieved
    knowledge-base chunks.
    """

    retrieved_chunks = search(
        query,
        documents,
        index,
        top_k=top_k,
    )

    if not retrieved_chunks:

        return {
            "answer": (
                "I could not find relevant information "
                "in the F1 knowledge base."
            ),
            "sources": [],
            "retrieved_chunks": [],
        }


    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context_sections = []

    for number, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):

        section = (
            f"[Source {number}: "
            f"{chunk['source']}]\n"
            f"{chunk['text']}"
        )

        context_sections.append(
            section
        )

    context = "\n\n".join(
        context_sections
    )


    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    system_prompt = """
You are the Race Knowledge Agent inside an F1 Strategy Copilot.

Your job is to answer Formula 1 knowledge and strategy questions using
ONLY the supplied retrieved context.

Rules:

1. Use the retrieved knowledge as your factual basis.
2. Do not invent F1 facts that are not supported by the context.
3. If the context does not contain enough information, clearly say so.
4. Explain the answer clearly and concisely.
5. When appropriate, explain the strategic reasoning involved.
6. Mention which supplied source or sources support the answer.
7. Do not pretend that general model knowledge came from the retrieved
   documents.

The purpose of this agent is grounded F1 knowledge retrieval.
"""


    user_prompt = f"""
USER QUESTION:

{query}


RETRIEVED F1 KNOWLEDGE:

{context}


Answer the user's question using the retrieved knowledge above.
"""


    # --------------------------------------------------------
    # GROQ CALL
    # --------------------------------------------------------

    response = client.chat.completions.create(
        model=MODEL,

        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],

        temperature=0.2,
    )


    answer = (
        response
        .choices[0]
        .message
        .content
    )


    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    sources = []

    for chunk in retrieved_chunks:

        source = chunk["source"]

        if source not in sources:
            sources.append(source)


    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
    }   