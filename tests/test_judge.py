"""Unit tests for the LLM-as-judge layer. The Ollama call is mocked, so we
test our parsing/robustness logic without invoking a model."""
from unittest.mock import patch

import judge
from judge import _parse_scores, judge as run_judge
from rag import RagResult


# ---------- _parse_scores ----------

def test_parse_clean_json():
    raw = '{"faithfulness": 4, "relevance": 5, "reason": "good"}'
    out = _parse_scores(raw)
    assert out["faithfulness"] == 4
    assert out["relevance"] == 5


def test_parse_json_embedded_in_prose():
    raw = 'Sure! Here is my evaluation:\n{"faithfulness": 3, "relevance": 2} \nHope that helps.'
    out = _parse_scores(raw)
    assert out["faithfulness"] == 3
    assert out["relevance"] == 2


def test_parse_garbage_returns_none_scores():
    out = _parse_scores("I cannot produce JSON, sorry.")
    assert out["faithfulness"] is None
    assert out["relevance"] is None
    assert out["reason"] == "parse_failed"


def test_parse_malformed_json_does_not_raise():
    out = _parse_scores('{"faithfulness": 4, "relevance":}')  # invalid
    assert out["faithfulness"] is None
    assert out["reason"] == "parse_failed"


# ---------- judge() with mocked Ollama ----------

def _result():
    return RagResult("What is X?", "X is a thing.", ["context about X"], [0.7], top_k=1)


def test_judge_parses_model_response():
    fake = {"response": '{"faithfulness": 5, "relevance": 4, "reason": "ok"}'}
    with patch.object(judge.ollama, "generate", return_value=fake) as m:
        out = run_judge(_result())
    assert out["faithfulness"] == 5
    assert out["relevance"] == 4
    m.assert_called_once()                      # the model was actually called


def test_judge_handles_model_garbage_gracefully():
    fake = {"response": "no json here"}
    with patch.object(judge.ollama, "generate", return_value=fake):
        out = run_judge(_result())
    assert out["faithfulness"] is None          # degrades, does not crash


def test_judge_prompt_includes_question_and_answer():
    captured = {}

    def _spy(model, prompt):
        captured["prompt"] = prompt
        return {"response": '{"faithfulness":3,"relevance":3,"reason":"x"}'}

    with patch.object(judge.ollama, "generate", side_effect=_spy):
        run_judge(_result())
    assert "What is X?" in captured["prompt"]
    assert "X is a thing." in captured["prompt"]
    assert "context about X" in captured["prompt"]
