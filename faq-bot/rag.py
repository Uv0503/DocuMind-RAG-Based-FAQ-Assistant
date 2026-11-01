# Retrieval layer: owns the active knowledge document, chunk metadata, embeddings, and search index.
# The index is process-local memory; uploaded text on disk is what allows it to be rebuilt at startup.
import os
import re
import hashlib
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
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
CACHE_DIR = BASE_DIR / ".rag_cache"
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@lru_cache(maxsize=2)
def _embedding_model(name: str, allow_download: bool):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, local_files_only=not allow_download)


@lru_cache(maxsize=3)
def _cross_encoder(name: str, allow_download: bool, max_length: int):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(name, local_files_only=not allow_download, max_length=max_length)


def bound_sources(sources: List[Dict], token_budget: int) -> List[Dict]:
    """Fit retrieved passages in a context budget while retaining source fields."""
    bounded = []
    for source in sources:
        if token_budget <= 0:
            break
        matches = list(TOKEN_RE.finditer(source["text"]))
        if not matches:
            continue
        if len(matches) > token_budget:
            source = {**source, "text": source["text"][:matches[token_budget - 1].end()]}
            token_budget = 0
        else:
            token_budget -= len(matches)
        bounded.append(source)
    return bounded


def evidence_sufficient(question: str, sources: List[Dict]) -> bool:
    """Reject context that omits an explicit alphanumeric record ID in the question."""
    if not sources:
        return False
    identifiers = {match.casefold() for match in re.findall(
        r"\b[A-Za-z]{2,}[A-Za-z0-9]*\d[A-Za-z0-9]*\b", question)}
    if not identifiers:
        return True
    context_tokens = set(re.findall(r"\w+", " ".join(source["text"] for source in sources).casefold()))
    return identifiers <= context_tokens


class RAGIndex:
    # Small embedding model keeps the project lightweight enough for a beginner-friendly RAG demo.
    def __init__(
        self,
        model_name: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        chunk_strategy: str = None,
        candidate_k: int = None,
        rrf_k: int = None,
        sparse_weight: float = None,
        rerank_top_n: int = None,
        reranker_max_length: int = None,
        reranker_name: str = None,
        cache_dir: Path = None,
    ):
        chunk_size = int(chunk_size or os.getenv("RAG_CHUNK_TOKENS", "512"))
        chunk_overlap = int(chunk_overlap if chunk_overlap is not None else os.getenv("RAG_CHUNK_OVERLAP", "51"))
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_strategy = chunk_strategy or os.getenv("RAG_CHUNK_STRATEGY", "fixed")
        self.candidate_k = int(candidate_k or os.getenv("RAG_CANDIDATE_K", "20"))
        self.rrf_k = int(rrf_k or os.getenv("RAG_RRF_K", "60"))
        self.sparse_weight = float(sparse_weight if sparse_weight is not None else os.getenv("RAG_SPARSE_WEIGHT", "1"))
        self.rerank_top_n = int(rerank_top_n if rerank_top_n is not None else os.getenv("RAG_RERANK_TOP_N", "8"))
        self.reranker_max_length = int(reranker_max_length if reranker_max_length is not None else os.getenv("RERANKER_MAX_LENGTH", "256"))
        self.reranker_name = reranker_name or os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.cache_dir = cache_dir or CACHE_DIR
        if (self.chunk_strategy not in {"fixed", "structure", "legacy_char"} or self.candidate_k <= 0
                or self.rrf_k <= 0 or self.sparse_weight <= 0 or self.rerank_top_n <= 0
                or self.reranker_max_length <= 0):
            raise ValueError("invalid retrieval configuration")
        self.model = None
        self.reranker = None
        self.index = None
        self.embeddings = None
        self.query_embeddings = {}
        self.chunks: List[Dict] = []
        self.postings = defaultdict(list)
        self.doc_lengths = []
        self.avg_length = 0.0
        self.is_built = False

    def _load_model(self):
        # Load lazily so importing the API does not immediately download/load the embedding model.
        if self.model is None:
            if os.getenv("USE_EMBEDDINGS", "true").lower() != "true":
                raise RuntimeError("Embeddings are disabled; using lexical retrieval")
            allow_download = os.getenv("ALLOW_MODEL_DOWNLOAD", "true").lower() == "true"
            self.model = _embedding_model(self.model_name, allow_download)
        return self.model

    def split_text(self, text: str) -> List[str]:
        return [part for _, part in self._split_with_headings(text)]

    def _split_with_headings(self, text: str):
        if self.chunk_strategy == "legacy_char":
            cleaned = " ".join(text.split())
            result = []
            for start in range(0, len(cleaned), self.chunk_size - self.chunk_overlap):
                result.append(("", cleaned[start:start + self.chunk_size]))
                if start + self.chunk_size >= len(cleaned):
                    break
            return result
        sections, heading, body = [], "", []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#") or (len(line) < 100 and line.endswith(":")):
                if body:
                    sections.append((heading, "\n".join(body)))
                heading, body = line.lstrip("# ").rstrip(":"), []
            else:
                body.append(line)
        if body:
            sections.append((heading, "\n".join(body)))
        if self.chunk_strategy == "fixed":
            tokens, headings = [], []
            for section_heading, section in sections:
                section_tokens = TOKEN_RE.findall(section)
                tokens.extend(section_tokens)
                headings.extend([section_heading] * len(section_tokens))
            capacity = max(1, self.chunk_size - max((len(TOKEN_RE.findall(h)) for h, _ in sections), default=0))
            step = max(1, capacity - self.chunk_overlap)
            result = []
            for start in range(0, len(tokens), step):
                window = tokens[start:start + capacity]
                section_heading = headings[start]
                passage = " ".join(window)
                result.append((section_heading, f"{section_heading}\n{passage}" if section_heading else passage))
                if start + capacity >= len(tokens):
                    break
            return result
        result = []
        for heading, section in sections:
            tokens = TOKEN_RE.findall(section)
            capacity = max(1, self.chunk_size - len(TOKEN_RE.findall(heading)))
            step = max(1, capacity - self.chunk_overlap)
            for start in range(0, len(tokens), step):
                window = tokens[start:start + capacity]
                if not window:
                    break
                passage = " ".join(window)
                result.append((heading, f"{heading}\n{passage}" if heading else passage))
                if start + capacity >= len(tokens):
                    break
        return result

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
        self.query_embeddings.clear()
        seen = set()
        for document in documents:
            for heading, text in self._split_with_headings(document["text"]):
                fingerprint = hashlib.sha256(" ".join(re.findall(r"\w+", text.casefold())).encode()).hexdigest()
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                self.chunks.append({
                    "chunk_id": len(self.chunks) + 1,
                    "source": document["source"],
                    "heading": heading,
                    "text": text,
                })

        self.postings = defaultdict(list)
        self.doc_lengths = []
        for index, chunk in enumerate(self.chunks):
            counts = self._term_counts(chunk["text"])
            self.doc_lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self.postings[term].append((index, frequency))
        self.avg_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0

        if not self.chunks:
            self.index = None
            self.embeddings = None
            self.is_built = True
            return 0

        # Cache normalized embeddings by content and model; uploaded text remains authoritative.
        try:
            key = hashlib.sha256(json.dumps([self.model_name, self.chunk_size, self.chunk_overlap,
                self.chunk_strategy, [(c["source"], c["text"]) for c in self.chunks]], ensure_ascii=False).encode()).hexdigest()
            cache_path = self.cache_dir / f"{key}.npy"
            if cache_path.exists():
                embeddings = np.load(cache_path, allow_pickle=False)
                if embeddings.shape[0] != len(self.chunks):
                    raise ValueError("stale embedding cache")
            else:
                model = self._load_model()
                embeddings = model.encode([chunk["text"] for chunk in self.chunks],
                    batch_size=64, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                temporary_path = cache_path.with_suffix(".tmp.npy")
                np.save(temporary_path, embeddings)
                temporary_path.replace(cache_path)
            embeddings = np.asarray(embeddings, dtype="float32")
            embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
        except Exception as exc:
            # Keep the API available when the model is not cached and Hugging Face is offline.
            print(f"Embedding model unavailable; using BM25 retrieval: {exc}")
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
        return [{**self.chunks[i], "score": round(score, 5)} for i, score in self._sparse(question, top_k)]

    def _sparse(self, question: str, top_k: int):
        scores = defaultdict(float)
        count = len(self.chunks)
        for term in set(self._term_counts(question)):
            posting = self.postings.get(term, [])
            if not posting:
                continue
            idf = math.log(1 + (count - len(posting) + .5) / (len(posting) + .5))
            for index, frequency in posting:
                denominator = frequency + 1.2 * (.25 + .75 * self.doc_lengths[index] / max(1, self.avg_length))
                scores[index] += idf * frequency * 2.2 / denominator
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]

    def _dense(self, question: str, top_k: int):
        if self.embeddings is None:
            return []
        vector = self.query_embeddings.get(question)
        if vector is None:
            vector = self._load_model().encode([question], convert_to_numpy=True,
                normalize_embeddings=True).astype("float32")
            vector /= np.maximum(np.linalg.norm(vector, axis=1, keepdims=True), 1e-12)
            if len(self.query_embeddings) >= 256:
                self.query_embeddings.clear()
            self.query_embeddings[question] = vector
        if self.index is not None:
            scores, indices = self.index.search(vector, min(top_k, len(self.chunks)))
            return [(int(i), float(s)) for s, i in zip(scores[0], indices[0]) if i >= 0]
        scores = np.matmul(self.embeddings, vector[0])
        indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in indices]

    def _load_reranker(self):
        if self.reranker is None:
            allow_download = os.getenv("ALLOW_MODEL_DOWNLOAD", "true").lower() == "true"
            self.reranker = _cross_encoder(self.reranker_name, allow_download, self.reranker_max_length)
        return self.reranker

    def retrieve(self, question: str, top_k: int = 3, min_score: float = 0.10,
                 mode: str = None, rerank: bool = None) -> List[Dict]:
        if not self.is_built:
            self.rebuild()
        if not self.chunks or not question.strip() or top_k <= 0:
            return []
        mode = mode or os.getenv("RAG_RETRIEVAL_MODE", "sparse")
        if mode not in {"baseline", "sparse", "dense", "hybrid"}:
            raise ValueError("invalid retrieval mode")
        rerank = os.getenv("RAG_RERANK", "false").lower() == "true" if rerank is None else rerank
        candidate_k = min(len(self.chunks), max(top_k, self.candidate_k))
        sparse = self._sparse(question, candidate_k) if mode in {"sparse", "hybrid"} else []
        dense = []
        if mode in {"baseline", "dense", "hybrid"}:
            try:
                dense = self._dense(question, candidate_k)
            except Exception as exc:
                print(f"Dense query unavailable; using BM25 retrieval: {exc}")
        if mode in {"baseline", "dense"} and not dense:
            sparse = self._sparse(question, candidate_k)
        sparse_scores, dense_scores = dict(sparse), dict(dense)
        scores = defaultdict(float)
        if mode == "hybrid" and dense:
            for ranking, weight in ((sparse, self.sparse_weight), (dense, 1.0)):
                for rank, (index, _) in enumerate(ranking, 1):
                    scores[index] += weight / (self.rrf_k + rank)
        else:
            for index, score in dense or sparse:
                scores[index] = score
        ordered = sorted(scores, key=lambda index: (-scores[index], index))
        ordered = [index for index in ordered if index in sparse_scores or dense_scores.get(index, -1) >= min_score]
        # An explicit record identifier must occur in the evidence. This prevents
        # shared policy words from making a nonexistent record look answerable.
        identifiers = re.findall(r"\b(?=\w*[A-Za-z])(?=\w*\d)[A-Za-z0-9]{4,}\b", question)
        if identifiers:
            if not all(any(identifier.casefold() in self.chunks[index]["text"].casefold()
                           for index in ordered) for identifier in identifiers):
                return []
            ordered = [index for index in ordered if any(identifier.casefold() in
                       self.chunks[index]["text"].casefold() for identifier in identifiers)]
        ordered = ordered[:candidate_k]
        rerank_scores = {}
        if rerank and mode != "baseline" and ordered:
            try:
                rerank_indices = ordered[:self.rerank_top_n]
                pairs = [(question, self.chunks[index]["text"]) for index in rerank_indices]
                predictions = self._load_reranker().predict(pairs, batch_size=32)
                rerank_scores = {index: float(score) for index, score in zip(rerank_indices, predictions)}
                ordered[:len(rerank_indices)] = sorted(rerank_indices,
                    key=lambda index: (-rerank_scores[index], -scores[index], index))
            except Exception as exc:
                print(f"Reranker unavailable; preserving retrieval order: {exc}")
        return [{**self.chunks[index], "score": round(scores[index], 5),
                 "dense_score": round(dense_scores[index], 5) if index in dense_scores else None,
                 "bm25_score": round(sparse_scores[index], 5) if index in sparse_scores else None,
                 "rerank_score": round(rerank_scores[index], 5) if index in rerank_scores else None}
                for index in ordered[:top_k]]

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
