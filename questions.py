"""Evaluation questions. Mix of easy/factual and harder/multi-hop so the
judge scores have variance for the regression to explain."""

QUESTIONS = [
    "What pigment makes photosynthesis possible and what does it absorb?",
    "What happens to time near the event horizon of a black hole?",
    "What were the main causes of the French Revolution?",
    "Describe the double helix structure of DNA.",
    "Who was the first Roman emperor and how did the empire begin?",
    "What is the uncertainty principle in quantum mechanics?",
    "How tall is Mount Everest and where is it located?",
    "What threats does the Great Barrier Reef face?",
    "What is Einstein's theory of general relativity about?",
    "How did the Industrial Revolution change manufacturing?",
    # Intentionally harder / partially out-of-corpus to create score spread:
    "Compare the social causes of the French and Industrial Revolutions.",
    "What is the relationship between DNA mutations and evolution?",
    "How does quantum mechanics relate to black hole physics?",
    "What was Einstein's role in the development of quantum theory?",
    "Why is the Great Barrier Reef visible from space?",
]
