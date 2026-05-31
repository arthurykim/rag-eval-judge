# RAG Eval Judge

A self-contained project that builds a small **RAG (retrieval-augmented
generation)** system, scores its answers with a **local LLM-as-judge**, and runs
**regression analysis** to find which retrieval features predict answer quality.

No API keys required — everything runs locally:
- **Data:** Wikipedia public API (`wikipedia` package)
- **Embeddings:** `sentence-transformers` (local)
- **Generation + Judge:** [Ollama](https://ollama.com) running `llama3` (local)

## Pipeline

```
Wikipedia API ──▶ chunk + embed ──▶ retrieve top-k ──▶ LLM answer
                                                            │
                                                            ▼
                          regression + graphs ◀── LLM-as-judge scores
                                                  (faithfulness, relevance)
```

## The research question

> Do retrieval features (similarity score, amount of context, top-k) predict
> the LLM-judge's quality scores?

We run every question at several `top_k` settings to create variance, log
features + judge scores to a CSV, then fit an OLS regression and plot it.

## Setup

```bash
# 1. Install + start Ollama, pull the model
brew install ollama
ollama serve &
ollama pull llama3.2:3b   # memory-friendly on Apple Silicon (8B llama3 OOMs on base M-series)

# 2. Python deps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python corpus.py          # fetch + cache Wikipedia articles
python run_experiment.py  # RAG + judge over all questions -> outputs/results.csv
python analyze.py         # OLS regression + outputs/analysis.png
```

## Files

| File | Role |
|------|------|
| `corpus.py` | Fetch/cache Wikipedia knowledge base |
| `rag.py` | Chunk, embed, retrieve, generate (the RAG system) |
| `questions.py` | Evaluation questions |
| `judge.py` | LLM-as-judge scoring (faithfulness, relevance) |
| `run_experiment.py` | Orchestrates the eval, writes results CSV |
| `analyze.py` | Regression + visualization |

## Extending

- Swap `JUDGE_MODEL` in `judge.py` to a different/stronger model than the
  generator to reduce self-bias.
- Add features (answer length, question type) to `FEATURES` in `analyze.py`.
- Add ground-truth answers to compute correctness vs. the judge's scores.
