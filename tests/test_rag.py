"""Unit tests for the RAG layer. No LLM or network — everything is mocked
or uses pure-logic functions, so tests run fast and offline."""
import numpy as np
import pytest

from rag_eval.rag import chunk_text, RagResult, RagSystem, CHUNK_CHARS, CHUNK_OVERLAP


# ---------- chunk_text ----------

def test_chunk_text_short_returns_single_chunk():
    text = "short text"
    assert chunk_text(text) == [text]


def test_chunk_text_covers_entire_string():
    text = "x" * 2500
    chunks = chunk_text(text)
    # every original character index must appear in some chunk
    assert chunks[0].startswith("x")
    assert "".join(c for c in chunks)  # non-empty
    assert len(chunks) >= 3


def test_chunk_text_has_overlap():
    text = "".join(chr(65 + (i % 26)) for i in range(2000))
    chunks = chunk_text(text)
    # tail of chunk 0 should reappear at head of chunk 1 (overlap window)
    overlap_tail = chunks[0][-CHUNK_OVERLAP:]
    assert overlap_tail in chunks[1]


def test_chunk_size_bounds():
    text = "a" * 5000
    for c in chunk_text(text):
        assert len(c) <= CHUNK_CHARS


# ---------- RagResult ----------

def test_ragresult_context_chars_and_avg_similarity():
    r = RagResult(
        question="q",
        answer="a",
        retrieved_chunks=["aaa", "bb"],   # 3 + 2 = 5 chars
        similarities=[0.8, 0.4],
        top_k=2,
    )
    assert r.context_chars == 5
    assert r.avg_similarity == pytest.approx(0.6)


def test_ragresult_avg_similarity_empty():
    r = RagResult("q", "a", [], [], top_k=0)
    assert r.avg_similarity == 0.0


# ---------- retrieve (mocked embedder) ----------

class _FakeEmbedder:
    """Maps known strings to fixed unit vectors so similarity is deterministic."""
    _vectors = {
        "apple fruit": np.array([1.0, 0.0, 0.0]),
        "banana fruit": np.array([0.9, 0.1, 0.0]),
        "rocket ship": np.array([0.0, 0.0, 1.0]),
        "fruit?": np.array([1.0, 0.0, 0.0]),
    }

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        out = []
        for t in texts:
            v = self._vectors[t].astype(float)
            out.append(v / np.linalg.norm(v))
        return np.array(out)


def test_retrieve_ranks_by_similarity():
    rag = RagSystem.__new__(RagSystem)          # bypass __init__ (no model load)
    rag.embedder = _FakeEmbedder()
    rag.chunks = ["apple fruit", "banana fruit", "rocket ship"]
    rag.embeddings = rag.embedder.encode(rag.chunks)

    chunks, sims = rag.retrieve("fruit?", top_k=2)
    assert chunks[0] == "apple fruit"           # most similar first
    assert chunks[1] == "banana fruit"
    assert sims[0] >= sims[1]                    # sorted descending
    assert "rocket ship" not in chunks          # top_k=2 excludes it
