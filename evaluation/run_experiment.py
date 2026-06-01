"""Run the full eval: for each question, at several top_k settings, run RAG,
judge the answer, and log features + scores to a CSV.

The varying top_k is what gives the regression something to chew on:
does retrieving more context change answer quality?
"""
import pandas as pd

from rag_eval.judge import judge
from rag_eval.questions import QUESTIONS
from rag_eval.rag import RagSystem

TOP_K_SETTINGS = [1, 2, 4, 8]
OUTPUT_CSV = "outputs/results.csv"


def main():
    rag = RagSystem()
    rag.build_index()

    rows = []
    total = len(QUESTIONS) * len(TOP_K_SETTINGS)
    n = 0
    for question in QUESTIONS:
        for top_k in TOP_K_SETTINGS:
            n += 1
            print(f"[{n}/{total}] top_k={top_k} :: {question[:60]}")
            result = rag.answer(question, top_k=top_k)
            scores = judge(result)
            rows.append({
                "question": question,
                "top_k": top_k,
                "avg_similarity": result.avg_similarity,
                "max_similarity": max(result.similarities),
                "context_chars": result.context_chars,
                "answer_chars": len(result.answer),
                "faithfulness": scores.get("faithfulness"),
                "relevance": scores.get("relevance"),
                "reason": scores.get("reason"),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows -> {OUTPUT_CSV}")
    print(df[["top_k", "avg_similarity", "faithfulness", "relevance"]]
          .describe())


if __name__ == "__main__":
    main()
