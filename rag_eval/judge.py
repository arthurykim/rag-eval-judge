"""LLM-as-judge: score a RAG answer on faithfulness & relevance (1-5).

Uses a local Ollama model. We request JSON output and parse it defensively
since small local models don't always honor the schema perfectly.
"""
import json
import re

import ollama

from rag_eval.rag import RagResult

JUDGE_MODEL = "llama3.2:3b"   # could be a different/stronger model than GEN_MODEL

RUBRIC = """You are an impartial evaluator of a question-answering system.

Given a QUESTION, the CONTEXT the system retrieved, and its ANSWER, score it.

Score each dimension from 1 (terrible) to 5 (excellent):
- faithfulness: Is every claim in the ANSWER supported by the CONTEXT?
  Penalize hallucinations / unsupported claims.
- relevance: Does the ANSWER actually address the QUESTION?

Respond with ONLY a JSON object, no prose:
{"faithfulness": <int>, "relevance": <int>, "reason": "<one sentence>"}"""


def _parse_scores(raw: str) -> dict:
    """Pull the first JSON object out of the model's text."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"faithfulness": None, "relevance": None, "reason": "parse_failed"}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"faithfulness": None, "relevance": None, "reason": "parse_failed"}


def judge(result: RagResult) -> dict:
    """Return {'faithfulness': int, 'relevance': int, 'reason': str}."""
    context = "\n\n---\n\n".join(result.retrieved_chunks)
    prompt = (
        f"{RUBRIC}\n\n"
        f"QUESTION: {result.question}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"ANSWER: {result.answer}\n\n"
        "JSON:"
    )
    resp = ollama.generate(model=JUDGE_MODEL, prompt=prompt)
    return _parse_scores(resp["response"])
