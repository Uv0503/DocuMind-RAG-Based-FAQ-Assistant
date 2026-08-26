# Retrieval layer: owns the active knowledge document, chunk metadata, embeddings, and search index.
# The index is process-local memory; uploaded text on disk is what allows it to be rebuilt at startup.
import os
import re
from collections import Counter
from math import sqrt
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    # FAISS is used when available. On Windows, the package may not install, so the code falls back to NumPy search.
    import faiss
except ImportError:
    faiss = None


# Keep all backend file paths relative to this folder so uvicorn can be started from faq-bot/.
BASE_DIR = Path(__file__).resolve().parent
# CONTEXT_FILE = BASE_DIR / "context.txt"
UPLOAD_DIR = BASE_DIR / "uploaded_docs"


class RAGIndex:
    # Small embedding model keeps the project lightweight enough for a beginner-friendly RAG demo.
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = None
        self.index = None
        self.embeddings = None
        self.chunks: List[Dict] = []
        self.is_built = False

    def _load_model(self):
        # Load lazily so importing the API does not immediately download/load the embedding model.
        if self.model is None:
            if os.getenv("USE_EMBEDDINGS", "true").lower() != "true":
                raise RuntimeError("Embeddings are disabled; using lexical retrieval")
            from sentence_transformers import SentenceTransformer

            allow_download = os.getenv("ALLOW_MODEL_DOWNLOAD", "true").lower() == "true"
            self.model = SentenceTransformer(self.model_name, local_files_only=not allow_download)
        return self.model

    def split_text(self, text: str) -> List[str]:
        # Normalize whitespace before chunking so copied documents do not create messy chunks.
        cleaned = " ".join(text.split())
        if not cleaned:
            return []

        # Use overlapping character chunks to keep related sentences together across chunk boundaries.
        chunks = []
        start = 0
        while start < len(cleaned):
            end = start + self.chunk_size
            chunk = cleaned[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(cleaned):
                break
            start = max(0, end - self.chunk_overlap)
        return chunks

    def load_documents(self) -> List[Dict[str, str]]:
        documents = []
        UPLOAD_DIR.mkdir(exist_ok=True)

        # If files were uploaded, use only the newest upload so old documents do not affect answers.
        uploaded_paths = sorted(UPLOAD_DIR.glob("*.txt"), key=lambda path: path.stat().st_mtime, reverse=True)

        if uploaded_paths:
            latest_upload = uploaded_paths[0]
            documents.append({
                "source": latest_upload.name,
                "text": latest_upload.read_text(encoding="utf-8"),
            })
            return documents

        # Fall back to the bundled context file when the user has not uploaded a document yet.
        # if CONTEXT_FILE.exists():
        #     documents.append({
        #         "source": CONTEXT_FILE.name,
        #         "text": CONTEXT_FILE.read_text(encoding="utf-8"),
        #     })

        return documents

    def build_from_documents(self, documents: List[Dict[str, str]]) -> int:
        """Build the index from explicit documents without touching uploaded files.

        The evaluation framework uses this public seam to exercise the same chunking,
        embedding, and vector-search code as the API. Each document must contain a
        ``source`` filename and its decoded ``text``.
        """
        self.chunks = []

        chunk_id = 1
        for document in documents:
            for text in self.split_text(document["text"]):
                self.chunks.append({
                    "chunk_id": chunk_id,
                    "source": document["source"],
                    "text": text,
                })
                chunk_id += 1

        if not self.chunks:
            self.index = None
            self.embeddings = None
            self.is_built = True
            return 0

        # Normalized embeddings let inner product behave like cosine similarity.
        try:
            model = self._load_model()
            embeddings = model.encode(
                [chunk["text"] for chunk in self.chunks],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")
        except Exception as exc:
            # Keep the API available when the model is not cached and Hugging Face is offline.
            print(f"Embedding model unavailable; using lexical retrieval: {exc}")
            self.model = None
            self.index = None
            self.embeddings = None
            self.is_built = True
            return len(self.chunks)

        self.embeddings = embeddings
        if faiss is not None:
            # FAISS handles fast vector search in deployments where faiss-cpu is available.
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)
        else:
            # NumPy fallback keeps local Windows development working without FAISS.
            self.index = None
        self.is_built = True
        return len(self.chunks)

    def rebuild(self) -> int:
        # Rebuild the production index from the latest uploaded document.
        return self.build_from_documents(self.load_documents())

    @staticmethod
    def _term_counts(text: str) -> Counter:
        return Counter(re.findall(r"[a-z0-9]+", text.lower()))

    def _lexical_retrieve(self, question: str, top_k: int) -> List[Dict]:
        # This dependency-free fallback compares normalized word-frequency vectors.
        # It keeps exact-word retrieval working when the semantic embedding model is unavailable.
        query = self._term_counts(question)
        query_norm = sqrt(sum(value * value for value in query.values()))
        if not query_norm:
            return []

        scored = []
        for chunk in self.chunks:
            terms = self._term_counts(chunk["text"])
            term_norm = sqrt(sum(value * value for value in terms.values()))
            if not term_norm:
                continue
            score = sum(query[word] * terms.get(word, 0) for word in query) / (query_norm * term_norm)
            if score > 0:
                scored.append((score, chunk))

        return [
            {**chunk, "score": round(float(score), 4)}
            for score, chunk in sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
        ]

    def retrieve(self, question: str, top_k: int = 3, min_score: float = 0.10) -> List[Dict]:
        # Build the index on first use if startup did not already do it.
        if not self.is_built:
            self.rebuild()

        if not self.chunks:
            return []

        if self.embeddings is None:
            return self._lexical_retrieve(question, top_k)

        model = self._load_model()
        query_embedding = model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        # Search with FAISS when possible; otherwise compute cosine-like scores directly with NumPy.
        if self.index is not None:
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
            score_list = list(zip(scores[0], indices[0]))
        else:
            scores = np.matmul(self.embeddings, query_embedding[0])
            indices = np.argsort(scores)[::-1][:top_k]
            score_list = [(scores[index], index) for index in indices]

        results = []
        for score, index in score_list:
            # Filter weak matches so unrelated questions do not get answers from random chunks.
            if index < 0 or float(score) < min_score:
                continue
            chunk = self.chunks[int(index)]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "text": chunk["text"],
                "score": round(float(score), 4),
            })
        return results

    def chunk_count(self) -> int:
        # Useful for health/debug endpoints and for deciding when to return the fallback answer.
        if not self.chunks:
            self.rebuild()
        return len(self.chunks)

    def active_sources(self) -> List[str]:
        # Shows which file is currently powering answers.
        if not self.chunks:
            self.rebuild()
        return sorted({chunk["source"] for chunk in self.chunks})


def save_uploaded_text(filename: str, content: bytes) -> int:
    # Keep upload support intentionally simple: only UTF-8 text files are accepted.
    safe_name = os.path.basename(filename)
    if not safe_name.lower().endswith(".txt"):
        raise ValueError("Only .txt files are supported")

    text = content.decode("utf-8").strip()
    if not text:
        raise ValueError("Uploaded file is empty")

    UPLOAD_DIR.mkdir(exist_ok=True)

    # Replace previous uploaded knowledge so answers come from the latest uploaded file only.
    for path in UPLOAD_DIR.glob("*.txt"):
        path.unlink()

    target = UPLOAD_DIR / safe_name
    target.write_text(text, encoding="utf-8")
    return len(RAGIndex().split_text(text))


# Shared singleton used by FastAPI routes.
# All requests handled by this Python process therefore see the same active index.
rag_index = RAGIndex()
