"""RAG pipeline: chunk -> embed -> retrieve -> generate.

Embeddings run locally via sentence-transformers (no key). Generation runs
locally via Ollama (no key). The whole pipeline is offline once models are
pulled.
"""
from dataclasses import dataclass, field

import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

from corpus import fetch_articles

EMBED_MODEL = "all-MiniLM-L6-v2"   # small, fast, local
GEN_MODEL = "llama3.2:3b"          # pulled via: ollama pull llama3.2:3b (memory-friendly on Apple Silicon)
CHUNK_CHARS = 800                  # approx chunk size
CHUNK_OVERLAP = 100


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping character windows."""
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_CHARS
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return chunks


@dataclass
class RagResult:
    """Everything we log about one question -> answer cycle."""
    question: str
    answer: str
    retrieved_chunks: list[str]
    similarities: list[float]
    top_k: int
    context_chars: int = field(init=False)

    def __post_init__(self):
        self.context_chars = sum(len(c) for c in self.retrieved_chunks)

    @property
    def avg_similarity(self) -> float:
        return float(np.mean(self.similarities)) if self.similarities else 0.0


class RagSystem:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.chunks: list[str] = []
        self.embeddings: np.ndarray | None = None

    def build_index(self, refresh: bool = False) -> None:
        """Fetch corpus, chunk it, and embed all chunks into memory."""
        articles = fetch_articles(refresh=refresh)
        for text in articles.values():
            self.chunks.extend(chunk_text(text))
        print(f"Embedding {len(self.chunks):,} chunks...")
        self.embeddings = self.embedder.encode(
            self.chunks, normalize_embeddings=True, show_progress_bar=True
        )

    def retrieve(self, query: str, top_k: int) -> tuple[list[str], list[float]]:
        """Return top_k chunks and their cosine similarities for `query`."""
        q = self.embedder.encode([query], normalize_embeddings=True)[0]
        sims = self.embeddings @ q          # cosine since both normalized
        idx = np.argsort(sims)[::-1][:top_k]
        return [self.chunks[i] for i in idx], [float(sims[i]) for i in idx]

    def answer(self, question: str, top_k: int = 4) -> RagResult:
        """Retrieve context and generate an answer with the local LLM."""
        chunks, sims = self.retrieve(question, top_k)
        context = "\n\n---\n\n".join(chunks)
        prompt = (
            "Answer the question using ONLY the context below. "
            "If the context does not contain the answer, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        resp = ollama.generate(model=GEN_MODEL, prompt=prompt)
        return RagResult(
            question=question,
            answer=resp["response"].strip(),
            retrieved_chunks=chunks,
            similarities=sims,
            top_k=top_k,
        )
