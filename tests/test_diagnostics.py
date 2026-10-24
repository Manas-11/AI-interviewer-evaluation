from pydantic import ValidationError
import pytest

from engine.evaluator import evaluate_answer, evaluate_answer_with_model, evaluate_many
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
    assert result["matched_concepts"] or result["missing_concepts"]
    assert all(len(concept.split()) > 1 for concept in result["matched_concepts"])
    assert set(result["score_reasons"]) == set(result["scores"])
    assert all(result["score_reasons"].values())


def test_empty_answer_is_safe_and_explicit():
    result = evaluate_answer("Explain caching", "", "Caching stores reusable results.")
    assert result["total_score"] == 0
    assert result["feedback"] == ["No answer was provided."]
    assert all("No score awarded" in reason for reason in result["score_reasons"].values())


def test_score_reasons_explain_deductions():
    result = evaluate_answer(
        "Explain caching and its trade-offs.",
        "Caching makes things faster.",
        "Caching stores reusable results to reduce latency. A complete answer "
        "should discuss invalidation, stale data, TTL, and consistency trade-offs.",
    )

    assert "Deducted" in result["score_reasons"]["correctness"]
    assert "Deducted" in result["score_reasons"]["completeness"]
    assert "Deducted" in result["score_reasons"]["clarity"]


def test_semantic_evaluator_uses_local_baseline_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    baseline = evaluate_answer(
        "What is caching?",
        "Caching stores data so later reads are faster.",
        "Caching stores reusable results to reduce latency.",
    )
    result = evaluate_answer_with_model(
        "What is caching?",
        "Caching stores data so later reads are faster.",
        "Caching stores reusable results to reduce latency.",
    )
    assert result == baseline
    assert result["evaluation_source"] == "Local meaning-aware evaluation"


def test_local_evaluator_recognizes_meaningful_paraphrase():
    result = evaluate_answer(
        "What is dependency injection?",
        "A class gets what it needs from an external provider instead of creating "
        "those objects internally, which makes testing easier.",
        "Dependency injection supplies a class with its dependencies from outside. "
        "This improves testability.",
    )

    assert len(result["matched_concepts"]) == 2
    assert result["missing_concepts"] == []
    assert result["scores"]["correctness"] >= 8


def test_evidence_uses_concept_statements_and_supports_paraphrases():
    result = evaluate_answer(
        "What is dependency injection?",
        "It supplies a class with dependencies from outside, which reduces coupling.",
        "Dependency injection supplies a class with its dependencies from outside. "
        "This reduces coupling and improves testability.",
    )

    assert any("dependencies from outside" in concept for concept in result["matched_concepts"])
    assert all(len(concept.split()) > 1 for concept in result["matched_concepts"])
    assert any("testability" in concept for concept in result["missing_concepts"])


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_is_rejected(question):
    with pytest.raises(ValueError, match="Question cannot be empty"):
        evaluate_answer(question, "answer", "reference")


def test_evaluate_many_preserves_order():
    results = evaluate_many([
        ("What is alpha?", "Alpha is the first value.", "Alpha is the first value."),
        ("What is beta?", "Beta is the second value.", "Beta is the second value."),
    ])
    assert len(results) == 2
    assert results[0]["matched_concepts"] == ["Alpha is the first value"]
    assert results[1]["matched_concepts"] == ["Beta is the second value"]
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


def test_reference_generation_falls_back_when_model_fails(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fail_generation(question):
        raise RuntimeError("The configured OpenAI model could not generate a reference answer.")

    monkeypatch.setattr("engine.reference._openai_reference", fail_generation)
    result = generate_reference_answer("What is caching?")
    assert result["source"] == "Local fallback"
    assert result["is_model_generated"] is False
    assert "warning" in result
    assert "caching" in result["answer"].lower()
