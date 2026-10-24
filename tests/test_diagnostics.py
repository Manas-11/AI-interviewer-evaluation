from pydantic import ValidationError
import pytest

from engine.evaluator import evaluate_answer, evaluate_many
from engine.reference import generate_reference_answer
from evaluation.transcript import process_transcript


def test_evaluate_answer_returns_bounded_scores_and_evidence():
    result = evaluate_answer(
        "What is the difference between a list and a tuple in Python?",
        "A list is mutable, whereas a tuple is immutable. Lists use square brackets.",
        "A list is a mutable ordered collection, while a tuple is an immutable ordered collection.",
    )

    assert set(result["scores"]) == {"correctness", "relevance", "completeness", "clarity"}
    assert all(0 <= score <= 10 for score in result["scores"].values())
    assert result["total_score"] == sum(result["scores"].values())
    assert result["maximum_score"] == 40
    assert "mutable" in result["matched_concepts"]


def test_empty_answer_is_safe_and_explicit():
    result = evaluate_answer("Explain caching", "", "Caching stores reusable results.")
    assert result["total_score"] == 0
    assert result["feedback"] == ["No answer was provided."]


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_is_rejected(question):
    with pytest.raises(ValueError, match="Question cannot be empty"):
        evaluate_answer(question, "answer", "reference")


def test_evaluate_many_preserves_order():
    results = evaluate_many([
        ("What is alpha?", "alpha", "alpha"),
        ("What is beta?", "beta", "beta"),
    ])
    assert len(results) == 2
    assert results[0]["matched_concepts"] == ["alpha"]
    assert results[1]["matched_concepts"] == ["beta"]
    assert all(0 <= item["total_score"] <= 40 for item in results)


def test_process_transcript_normalizes_and_counts_turns():
    result = process_transcript(
        {
            "role": " Backend Developer ",
            "instructions": " Be precise ",
            "transcript": [
                {"speaker": "interviewer", "text": " Question? "},
                {"speaker": "candidate", "text": " Answer. "},
            ],
        }
    )
    assert result["role"] == "Backend Developer"
    assert result["turn_count"] == 2
    assert result["candidate_turn_count"] == 1
    assert result["turns"][1]["text"] == "Answer."


def test_process_transcript_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        process_transcript(
            {
                "role": "Engineer",
                "instructions": "Assess",
                "unexpected": True,
                "transcript": [{"speaker": "candidate", "text": "Answer"}],
            }
        )


def test_reference_answer_is_generated_without_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = generate_reference_answer("What is dependency injection?")
    assert result["source"] == "Local fallback"
    assert "dependencies" in result["answer"]
    assert result["is_model_generated"] is False
