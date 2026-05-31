"""Regression analysis + graphs over the judged results.

Core question: do retrieval features (similarity, context size, top_k)
predict the LLM-judge quality scores? We fit an OLS model and plot it.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

RESULTS_CSV = "outputs/results.csv"
TARGET = "faithfulness"   # swap to "relevance" to analyze the other dimension
FEATURES = ["avg_similarity", "max_similarity", "context_chars", "top_k"]


def load() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    return df.dropna(subset=[TARGET] + FEATURES)


def fit_regression(df: pd.DataFrame):
    """OLS: judge score ~ retrieval features. Prints full summary."""
    X = sm.add_constant(df[FEATURES])
    y = df[TARGET]
    model = sm.OLS(y, X).fit()
    print(model.summary())
    return model


def plot(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1) Scatter: retrieval quality vs judge score, with fitted line
    ax = axes[0]
    x, y = df["avg_similarity"], df[TARGET]
    ax.scatter(x, y, alpha=0.6)
    m, b = np.polyfit(x, y, 1)
    xs = sorted(x)
    ax.plot(xs, [m * v + b for v in xs], color="crimson", lw=2)
    ax.set_xlabel("avg retrieval similarity")
    ax.set_ylabel(f"judge {TARGET} (1-5)")
    ax.set_title(f"{TARGET} vs retrieval similarity")

    # 2) Mean judge score by top_k
    ax = axes[1]
    by_k = df.groupby("top_k")[[TARGET, "relevance"]].mean()
    by_k.plot(kind="bar", ax=ax)
    ax.set_xlabel("top_k (chunks retrieved)")
    ax.set_ylabel("mean judge score")
    ax.set_title("Quality vs amount of context retrieved")

    fig.tight_layout()
    out = "outputs/analysis.png"
    fig.savefig(out, dpi=120)
    print(f"\nSaved plot -> {out}")


if __name__ == "__main__":
    df = load()
    fit_regression(df)
    plot(df)
