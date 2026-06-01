"""Knowledge base layer: fetch Wikipedia articles, cache to local JSON.

This is our 'external API' data source. No API key required — the `wikipedia`
package hits Wikipedia's public REST endpoints.
"""
import json
from pathlib import Path

import wikipedia

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"  # project-root /cache
CACHE_DIR.mkdir(exist_ok=True)

# Topics that form our knowledge base. Keep curated + factual so the judge
# has clear ground to evaluate faithfulness against.
TOPICS = [
    "Photosynthesis",
    "Black hole",
    "French Revolution",
    "DNA",
    "Roman Empire",
    "Quantum mechanics",
    "Mount Everest",
    "Great Barrier Reef",
    "Albert Einstein",
    "Industrial Revolution",
]


def fetch_articles(refresh: bool = False) -> dict[str, str]:
    """Return {topic: full_article_text}. Cached locally as JSON."""
    cache_file = CACHE_DIR / "articles.json"
    if cache_file.exists() and not refresh:
        return json.loads(cache_file.read_text())

    articles: dict[str, str] = {}
    for topic in TOPICS:
        try:
            articles[topic] = wikipedia.page(topic, auto_suggest=False).content
            print(f"  fetched: {topic} ({len(articles[topic]):,} chars)")
        except Exception as e:  # disambiguation, network, etc.
            print(f"  SKIP {topic}: {e}")

    cache_file.write_text(json.dumps(articles))
    return articles


if __name__ == "__main__":
    arts = fetch_articles()
    print(f"\nLoaded {len(arts)} articles, "
          f"{sum(len(v) for v in arts.values()):,} total chars")
