import os

from dotenv import load_dotenv
from groq import Groq

from retriever import load_documents, build_index, search
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")


MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# Initialize clients
# --------------------------------------------------

client = Groq(api_key=GROQ_API_KEY)

embedding_model = SentenceTransformer(EMBEDDING_MODEL)


# --------------------------------------------------
# Build knowledge base
# --------------------------------------------------

documents = load_documents()

index = build_index(
    documents,
    embedding_model
)


# --------------------------------------------------
# RAG Agent
# --------------------------------------------------

def answer_question(query):

    # Retrieve relevant knowledge
    results = search(
        query,
        embedding_model,
        index,
        documents,
        top_k=3
    )

    # Combine retrieved chunks
    context_parts = []

    for result in results:

        context_parts.append(
            f"Source: {result['source']}\n"
            f"Category: {result['category']}\n"
            f"Content:\n{result['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)


    # --------------------------------------------------
    # Prompt the LLM
    # --------------------------------------------------

    system_prompt = """
You are the F1 Strategy Copilot RAG Agent.

Your job is to answer Formula 1 questions using ONLY
the retrieved knowledge provided by the system.

Rules:

1. Use the retrieved context as your primary source.
2. Do not invent facts that are not supported by the context.
3. If the context does not contain enough information,
   clearly say that the available knowledge base does
   not contain enough information.
4. Explain the answer clearly.
5. Keep the answer concise but useful.
6. Mention the source files used.
"""


    user_prompt = f"""
User Question:
{query}

Retrieved F1 Knowledge:
{context}

Using the retrieved knowledge, answer the user's question.
"""


    # --------------------------------------------------
    # Generate answer
    # --------------------------------------------------

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content


    # --------------------------------------------------
    # Return answer + sources
    # --------------------------------------------------

    sources = []

    for result in results:

        if result["source"] not in sources:
            sources.append(result["source"])


    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": results
    }


# --------------------------------------------------
# Test RAG Agent
# --------------------------------------------------

if __name__ == "__main__":

    question = "Why is an undercut useful in Formula 1?"

    print("\nQuestion:")
    print(question)

    result = answer_question(question)

    print("\n==============================")
    print("RAG ANSWER")
    print("==============================")

    print(result["answer"])

    print("\n==============================")
    print("SOURCES")
    print("==============================")

    for source in result["sources"]:
        print("-", source)