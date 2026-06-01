"""FastAPI service exposing the RAG pipeline and the LLM-as-judge.

Local:   uvicorn app.main:app --reload   (needs Ollama running on localhost)
Docker:  see docker-compose.yml          (API + Ollama as separate containers)

Endpoints:
  GET  /health    — liveness + how many chunks are indexed
  POST /query     — retrieve context and generate an answer
  POST /evaluate  — generate an answer AND score it with the LLM-as-judge
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag_eval.rag import RagSystem
from rag_eval.judge import judge as judge_answer

# Built once at startup and reused across requests.
rag: RagSystem | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Build the embedding index once when the service boots."""
    global rag
    rag = RagSystem()
    rag.build_index()
    yield


app = FastAPI(title="RAG Eval Judge", version="1.0.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(..., examples=["What is a black hole?"])
    top_k: int = Field(4, ge=1, le=20, description="How many chunks to retrieve")


@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": len(rag.chunks) if rag else 0}


@app.post("/query")
def query(req: QueryRequest):
    """Run retrieval + generation and return the answer with its context."""
    result = rag.answer(req.question, top_k=req.top_k)
    return {
        "question": result.question,
        "answer": result.answer,
        "top_k": result.top_k,
        "avg_similarity": result.avg_similarity,
        "retrieved_chunks": result.retrieved_chunks,
    }


@app.post("/evaluate")
def evaluate(req: QueryRequest):
    """Run the full pipeline and score the answer with the LLM-as-judge."""
    result = rag.answer(req.question, top_k=req.top_k)
    scores = judge_answer(result)
    return {
        "question": result.question,
        "answer": result.answer,
        "avg_similarity": result.avg_similarity,
        "scores": scores,
    }
