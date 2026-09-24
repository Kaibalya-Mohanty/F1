from pathlib import Path

import faiss
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


# ============================================================
# PATHS / CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"

MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
MAX_LENGTH = 256


# ============================================================
# ONNX EMBEDDING MODEL
# ============================================================

class ONNXEmbedder:
    """
    Lightweight MiniLM embedding model using ONNX Runtime.

    This implementation intentionally does NOT use:
        - torch
        - sentence_transformers
        - transformers

    This avoids the PyTorch DLL that is blocked by
    Windows Device Guard on this machine.
    """

    def __init__(self):

        print("Loading ONNX embedding model...")

        # Download tokenizer.json directly from Hugging Face
        self.tokenizer_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="tokenizer.json",
        )

        # Download the official ONNX model
        self.model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="onnx/model.onnx",
        )

        # Load tokenizer without transformers
        self.tokenizer = Tokenizer.from_file(
            self.tokenizer_path
        )

        # Enable truncation
        self.tokenizer.enable_truncation(
            max_length=MAX_LENGTH
        )

        # Enable dynamic padding
        self.tokenizer.enable_padding()

        # Load ONNX model
        self.session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
        )

        # Find expected model inputs
        self.input_names = {
            item.name
            for item in self.session.get_inputs()
        }

        print("ONNX embedding model ready.")
        print("Model inputs:", self.input_names)


    # --------------------------------------------------------
    # TOKENIZATION
    # --------------------------------------------------------

    def _tokenize(self, texts):

        encodings = self.tokenizer.encode_batch(texts)

        input_ids = np.array(
            [encoding.ids for encoding in encodings],
            dtype=np.int64,
        )

        attention_mask = np.array(
            [
                encoding.attention_mask
                for encoding in encodings
            ],
            dtype=np.int64,
        )

        token_type_ids = np.zeros_like(
            input_ids,
            dtype=np.int64,
        )

        inputs = {}

        if "input_ids" in self.input_names:
            inputs["input_ids"] = input_ids

        if "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask

        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = token_type_ids

        return inputs, attention_mask


    # --------------------------------------------------------
    # MEAN POOLING
    # --------------------------------------------------------

    @staticmethod
    def _mean_pooling(
        token_embeddings,
        attention_mask,
    ):
        """
        Mean pooling while ignoring padding tokens.
        """

        mask = attention_mask[..., None].astype(
            np.float32
        )

        summed_embeddings = np.sum(
            token_embeddings * mask,
            axis=1,
        )

        summed_mask = np.clip(
            np.sum(mask, axis=1),
            a_min=1e-9,
            a_max=None,
        )

        return summed_embeddings / summed_mask


    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    @staticmethod
    def _normalize(embeddings):

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        norms = np.clip(
            norms,
            a_min=1e-12,
            a_max=None,
        )

        return embeddings / norms


    # --------------------------------------------------------
    # ENCODE
    # --------------------------------------------------------

    def encode(self, texts, batch_size=32):

        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        for start in range(
            0,
            len(texts),
            batch_size,
        ):

            batch = texts[
                start:start + batch_size
            ]

            inputs, attention_mask = self._tokenize(
                batch
            )

            outputs = self.session.run(
                None,
                inputs,
            )

            # MiniLM ONNX model's first output contains
            # token-level embeddings.
            token_embeddings = outputs[0]

            sentence_embeddings = self._mean_pooling(
                token_embeddings,
                attention_mask,
            )

            sentence_embeddings = self._normalize(
                sentence_embeddings
            )

            all_embeddings.append(
                sentence_embeddings.astype(
                    np.float32
                )
            )

        return np.vstack(all_embeddings)


# ============================================================
# GLOBAL EMBEDDING MODEL
# ============================================================

_embedder = None


def get_embedder():

    global _embedder

    if _embedder is None:
        _embedder = ONNXEmbedder()

    return _embedder


# ============================================================
# DOCUMENT CHUNKING
# ============================================================

def chunk_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):
    """
    Split a document into overlapping character chunks.
    """

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start += chunk_size - overlap

    return chunks


# ============================================================
# DOCUMENT LOADING
# ============================================================

def load_documents():
    """
    Recursively load all .txt files from knowledge/.
    """

    documents = []

    if not KNOWLEDGE_DIR.exists():
        print(
            f"Knowledge directory not found: "
            f"{KNOWLEDGE_DIR}"
        )
        return documents

    for file_path in KNOWLEDGE_DIR.rglob("*.txt"):

        try:

            text = file_path.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError:

            text = file_path.read_text(
                encoding="latin-1"
            )

        chunks = chunk_text(text)

        relative_path = file_path.relative_to(
            KNOWLEDGE_DIR
        )

        category = (
            relative_path.parts[0]
            if len(relative_path.parts) > 1
            else "general"
        )

        for index, chunk in enumerate(chunks):

            documents.append(
                {
                    "text": chunk,
                    "source": str(relative_path),
                    "category": category,
                    "chunk_index": index,
                }
            )

    print(
        f"Loaded {len(documents)} knowledge chunks."
    )

    return documents


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_index(documents):
    """
    Convert document chunks into embeddings and create
    a FAISS cosine-similarity index.
    """

    if not documents:
        return None

    texts = [
        document["text"]
        for document in documents
    ]

    embedder = get_embedder()

    print(
        f"Generating embeddings for "
        f"{len(texts)} chunks..."
    )

    embeddings = embedder.encode(
        texts,
        batch_size=32,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    dimension = embeddings.shape[1]

    # Embeddings are already normalized, therefore
    # inner product behaves as cosine similarity.
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    print(
        f"FAISS index ready. "
        f"Dimension: {dimension}"
    )

    return index


# ============================================================
# SEARCH
# ============================================================

def search(
    query,
    documents,
    index,
    top_k=3,
):
    """
    Search the FAISS index for the most relevant chunks.
    """

    if not documents or index is None:
        return []

    embedder = get_embedder()

    query_embedding = embedder.encode(
        [query]
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0],
    ):

        if index_position < 0:
            continue

        document = documents[
            int(index_position)
        ]

        results.append(
            {
                "text": document["text"],
                "source": document["source"],
                "category": document["category"],
                "chunk_index": document[
                    "chunk_index"
                ],
                "score": float(score),
            }
        )

    return results


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print("\nLoading documents...")

    documents = load_documents()

    print("\nBuilding index...")

    index = build_index(documents)

    print("\nRunning test search...")

    results = search(
        "What is an undercut in Formula 1?",
        documents,
        index,
        top_k=3,
    )

    print("\nTOP RESULTS:")

    for result in results:

        print("\n" + "=" * 70)

        print(
            "Source:",
            result["source"],
        )

        print(
            "Score:",
            round(result["score"], 4),
        )

        print(
            "Text:",
            result["text"][:500],
        )