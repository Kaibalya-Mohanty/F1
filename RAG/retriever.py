from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


# --------------------------------------------------
# Load knowledge files
# --------------------------------------------------

def load_documents():
    documents = []

    for file_path in KNOWLEDGE_DIR.rglob("*.txt"):

        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        category = file_path.parent.name
        source = file_path.name

        start = 0
        chunk_index = 0

        while start < len(text):

            end = start + CHUNK_SIZE
            chunk = text[start:end].strip()

            if chunk:
                documents.append({
                    "text": chunk,
                    "source": source,
                    "category": category,
                    "chunk_index": chunk_index
                })

            chunk_index += 1

            if end >= len(text):
                break

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return documents


# --------------------------------------------------
# Build FAISS index
# --------------------------------------------------

def build_index(documents, model):

    texts = [doc["text"] for doc in documents]

    embeddings = model.encode_document(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# --------------------------------------------------
# Search knowledge base
# --------------------------------------------------

def search(query, model, index, documents, top_k=3):

    query_embedding = model.encode_query(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    ).reshape(1, -1)

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_id in zip(scores[0], indices[0]):

        if index_id == -1:
            continue

        document = documents[index_id]

        results.append({
            "score": float(score),
            "source": document["source"],
            "category": document["category"],
            "chunk_index": document["chunk_index"],
            "text": document["text"]
        })

    return results


# --------------------------------------------------
# Test the retriever
# --------------------------------------------------

if __name__ == "__main__":

    print("Loading knowledge base...")

    documents = load_documents()

    print(f"Loaded {len(documents)} knowledge chunks.")

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Building FAISS index...")

    index = build_index(documents, model)

    print("FAISS index ready.")

    query = "Why is an undercut useful in Formula 1?"

    print(f"\nQuery: {query}")

    results = search(
        query,
        model,
        index,
        documents,
        top_k=3
    )

    print("\nTop results:\n")

    for i, result in enumerate(results, start=1):

        print(f"--- Result {i} ---")
        print(f"Source: {result['source']}")
        print(f"Category: {result['category']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text:\n{result['text']}\n")