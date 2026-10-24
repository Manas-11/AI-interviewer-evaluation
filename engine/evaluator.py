"""Deterministic interview-answer evaluation.

The evaluator intentionally has no network or model dependency. It provides a
transparent baseline that can be used in production and replaced behind the
same interface with an LLM-based evaluator later.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[+#][A-Za-z0-9]+|[.-][A-Za-z0-9]+)*")
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "what", "when", "where", "which", "why", "with",
}
_CRITERIA = ("correctness", "relevance", "completeness", "clarity")
_SYNONYMS = {
    "changeable": "mutable",
    "changes": "mutable",
    "change": "mutable",
    "fixed": "immutable",
    "unchangeable": "immutable",
    "external": "outside",
    "externally": "outside",
    "provided": "supply",
    "provides": "supply",
    "supplies": "supply",
    "dependencies": "dependency",
    "classes": "class",
    "functions": "function",
    "faster": "latency",
    "speed": "latency",
    "retries": "retry",
    "gets": "supply",
    "getting": "supply",
    "receives": "supply",
    "receiving": "supply",
    "needs": "dependency",
    "needed": "dependency",
    "objects": "dependency",
    "object": "dependency",
    "outside": "external",
    "provider": "external",
    "providers": "external",
    "replaceable": "replace",
    "replaced": "replace",
    "replaceable": "replace",
    "easier": "improve",
    "improves": "improve",
    "improved": "improve",
    "testing": "testability",
    "tested": "testability",
    "quick": "latency",
    "quickly": "latency",
}
_PHRASE_ALIASES = (
    (r"\bgets?\s+what\s+it\s+needs\b", "supply dependency"),
    (r"\bwhat\s+it\s+needs\b", "dependency"),
    (r"\bfrom\s+an?\s+external\s+(?:provider|source)\b", "outside"),
    (r"\bwithout\s+creating\s+(?:them|those\s+objects?)\s+(?:itself|internally)\b", "outside"),
    (r"\bmake(?:s)?\s+it\s+easier\s+to\s+test\b", "improve testability"),
    (r"\bmake(?:s)?\s+testing\s+easier\b", "improve testability"),
)


def _validate_text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{name} cannot be empty.")
    return cleaned


def _tokens(text: str, *, include_stop_words: bool = False) -> set[str]:
    for pattern, replacement in _PHRASE_ALIASES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    tokens = {match.group(0).lower() for match in _TOKEN_RE.finditer(text)}
    tokens = {_SYNONYMS.get(token, token) for token in tokens}
    if include_stop_words:
        return tokens
    return {token for token in tokens if token not in _STOP_WORDS}


def _concepts(text: str) -> list[str]:
    """Extract meaningful concept statements, rather than individual words."""

    chunks = re.split(r"(?<=[.!?;])\s+|,\s+(?=(?:and|while|whereas|because|which)\b)", text.strip())
    concepts: list[str] = []
    for chunk in chunks:
        cleaned = chunk.strip(" .;,:")
        if len(_tokens(cleaned)) >= 2 and cleaned.lower() not in concepts:
            concepts.append(cleaned)
    return concepts


def _concept_is_supported(concept: str, candidate_tokens: set[str]) -> bool:
    concept_tokens = _tokens(concept)
    if not concept_tokens:
        return False
    exact_overlap = len(concept_tokens & candidate_tokens) / len(concept_tokens)
    if exact_overlap >= 0.6 or (exact_overlap >= 0.5 and len(concept_tokens & candidate_tokens) >= 3):
        return True
    fuzzy_matches = sum(
        any(SequenceMatcher(None, token, candidate).ratio() >= 0.78 for candidate in candidate_tokens)
        for token in concept_tokens
    )
    return fuzzy_matches / len(concept_tokens) >= 0.6


def _score(value: float) -> int:
    return max(0, min(10, round(value)))


def _overlap(candidate: set[str], reference: set[str]) -> float:
    if not reference:
        return 0.0
    return len(candidate & reference) / len(reference)


def _feedback(scores: dict[str, int], answer: str, reference: str) -> list[str]:
    feedback: list[str] = []
    if scores["correctness"] >= 8:
        feedback.append("The answer includes terminology aligned with the reference.")
    else:
        feedback.append("Add the core concepts and technical terms expected for this question.")
    if scores["completeness"] < 7:
        feedback.append("Cover more of the reference concepts, including the reasoning behind them.")
    if scores["relevance"] < 7:
        feedback.append("Answer the question more directly and remove unrelated detail.")
    if scores["clarity"] < 7:
        feedback.append("Use complete sentences and a concise structure: claim, explanation, and example.")
    if len(answer.split()) > max(80, len(reference.split()) * 4):
        feedback.append("Keep the response focused; it is substantially longer than the reference.")
    return feedback


def _score_reasons(
    scores: dict[str, int],
    matched_concepts: list[str],
    missing_concepts: list[str],
    answer: str,
) -> dict[str, str]:
    """Explain deductions in plain language for every scoring criterion."""

    reasons: dict[str, str] = {}
    if scores["correctness"] >= 8:
        reasons["correctness"] = "No major deduction: the response aligns well with the expected concepts."
    elif missing_concepts:
        reasons["correctness"] = (
            f"Deducted because {len(missing_concepts)} expected concept(s) were not supported: "
            f"{'; '.join(missing_concepts[:2])}."
        )
    else:
        reasons["correctness"] = "Deducted because the response contains limited evidence of the expected technical concepts."

    if scores["relevance"] >= 8:
        reasons["relevance"] = "No major deduction: the response stays focused on the question."
    else:
        reasons["relevance"] = (
            "Deducted because the answer does not address enough of the question's key terms "
            "or includes insufficient direct explanation."
        )

    if scores["completeness"] >= 8:
        reasons["completeness"] = "No major deduction: the important reference concepts are covered."
    elif missing_concepts:
        reasons["completeness"] = (
            f"Deducted because {len(missing_concepts)} reference concept(s) are missing. "
            "Review the missing concept statements below."
        )
    else:
        reasons["completeness"] = "Deducted because the answer covers only part of the expected explanation."

    word_count = len(answer.split())
    if scores["clarity"] >= 8:
        reasons["clarity"] = "No major deduction: the response is structured and easy to follow."
    elif word_count < 5:
        reasons["clarity"] = "Deducted because the response is too short to explain the reasoning clearly."
    else:
        reasons["clarity"] = "Deducted because the response would benefit from clearer structure, explanation, or an example."
    return reasons


def evaluate_answer(
    question: str,
    candidate_answer: str,
    reference_answer: str,
) -> dict[str, Any]:
    """Score one answer from 0-10 across four explainable criteria.

    This is a lexical baseline, not a claim of human-level semantic judgment.
    Scores are bounded integers and the result is JSON serializable.
    """

    question = _validate_text(question, "Question")
    candidate_answer = candidate_answer.strip() if isinstance(candidate_answer, str) else candidate_answer
    reference_answer = _validate_text(reference_answer, "Reference answer")
    if not isinstance(candidate_answer, str):
        raise TypeError("Candidate answer must be a string.")
    if not candidate_answer:
        scores = {criterion: 0 for criterion in _CRITERIA}
        return {
            "scores": scores,
            "total_score": 0,
            "maximum_score": 40,
            "percentage": 0.0,
            "feedback": ["No answer was provided."],
            "evaluation_source": "Local meaning-aware evaluation",
            "score_reasons": {
                criterion: "No score awarded because no candidate answer was provided."
                for criterion in _CRITERIA
            },
            "matched_concepts": [],
            "missing_concepts": sorted(_tokens(reference_answer)),
        }

    candidate_tokens = _tokens(candidate_answer)
    reference_tokens = _tokens(reference_answer)
    question_tokens = _tokens(question)
    reference_concepts = _concepts(reference_answer) or [reference_answer]
    matched = [
        concept for concept in reference_concepts
        if _concept_is_supported(concept, candidate_tokens)
    ]
    missing = [
        concept for concept in reference_concepts
        if concept not in matched
    ]

    concept_coverage = len(matched) / len(reference_concepts)
    lexical_coverage = _overlap(candidate_tokens, reference_tokens)
    concept_coverage = (concept_coverage * 0.75) + (lexical_coverage * 0.25)
    question_coverage = _overlap(candidate_tokens, question_tokens)
    has_structure = len(re.findall(r"[.!?]", candidate_answer)) > 0
    word_count = len(candidate_answer.split())

    correctness = _score(concept_coverage * 10)
    relevance = _score((question_coverage * 0.45 + concept_coverage * 0.55) * 10)
    completeness = _score(concept_coverage * 10)
    clarity = _score(
        10 if word_count >= 8 and has_structure else
        7 if word_count >= 5 else
        4 if word_count >= 2 else 2
    )
    scores = {
        "correctness": correctness,
        "relevance": relevance,
        "completeness": completeness,
        "clarity": clarity,
    }
    total = sum(scores.values())
    score_reasons = _score_reasons(scores, matched, missing, candidate_answer)
    return {
        "scores": scores,
        "total_score": total,
        "maximum_score": 40,
        "percentage": round(total / 40 * 100, 1),
        "feedback": _feedback(scores, candidate_answer, reference_answer),
        "evaluation_source": "Local meaning-aware evaluation",
        "score_reasons": score_reasons,
        "matched_concepts": matched,
        "missing_concepts": missing,
    }


def evaluate_many(
    answers: Iterable[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    """Evaluate multiple ``(question, answer, reference)`` tuples."""

    return [evaluate_answer(question, answer, reference) for question, answer, reference in answers]


def _semantic_evaluation_prompt(
    question: str,
    candidate_answer: str,
    reference_answer: str,
) -> str:
    return (
        "You are comparing an interview candidate's meaning with a rubric. "
        "First derive the essential, independently checkable concepts from the "
        "reference answer. Then determine which of those concepts are actually "
        "supported by the candidate answer. Treat accurate paraphrases, "
        "synonyms, reordered explanations, and concrete examples as equivalent "
        "when they express the same idea. Do not award credit for a shared word "
        "without matching meaning. Do not invent evidence that is absent from "
        "the candidate. A concept is matched only when the candidate supports "
        "it; otherwise put the reference concept in missing_concepts. Score "
        "correctness and completeness from this concept comparison, relevance "
        "from alignment with the question, and clarity from the candidate's "
        "organization and understandable wording. Return JSON only with this "
        "shape: "
        '{"scores":{"correctness":0,"relevance":0,"completeness":0,"clarity":0},'
        '"matched_concepts":[],"missing_concepts":[],"score_reasons":{},'
        '"feedback":[]}. Scores must be integers from 0 to 10. Concept lists '
        "must contain complete reference ideas, not isolated words. "
        f"\nQuestion:\n{question}\nCandidate answer:\n{candidate_answer}"
        f"\nGenerated reference answer:\n{reference_answer}"
    )


def evaluate_answer_with_model(
    question: str,
    candidate_answer: str,
    reference_answer: str,
) -> dict[str, Any]:
    """Evaluate meaning with an optional model, falling back to the baseline.

    The model is instructed to award credit for accurate paraphrases and to
    cite only concepts supported by the candidate response. This keeps the
    local evaluator available when no provider is configured.
    """

    baseline = evaluate_answer(question, candidate_answer, reference_answer)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and candidate_answer.strip():
        try:
            from openai import APIError, OpenAI

            response = OpenAI(api_key=api_key, timeout=12.0, max_retries=0).responses.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                input=_semantic_evaluation_prompt(question, candidate_answer, reference_answer),
                temperature=0.1,
                max_output_tokens=700,
            )
            parsed = json.loads(response.output_text)
            scores = parsed["scores"]
            if set(scores) != set(_CRITERIA) or any(
                not isinstance(scores[key], int) or not 0 <= scores[key] <= 10
                for key in _CRITERIA
            ):
                raise ValueError("OpenAI returned invalid scores.")
            total = sum(scores.values())
            return {
                **baseline,
                "scores": scores,
                "total_score": total,
                "percentage": round(total / 40 * 100, 1),
                "matched_concepts": [x for x in parsed.get("matched_concepts", []) if isinstance(x, str)],
                "missing_concepts": [x for x in parsed.get("missing_concepts", []) if isinstance(x, str)],
                "score_reasons": {
                    criterion: str(parsed.get("score_reasons", {}).get(
                        criterion, baseline["score_reasons"][criterion]
                    ))
                    for criterion in _CRITERIA
                },
                "feedback": [x for x in parsed.get("feedback", []) if isinstance(x, str)]
                    or baseline["feedback"],
                "evaluation_source": "OpenAI semantic evaluation",
            }
        except (ImportError, APIError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            baseline["evaluation_warning"] = (
                f"OpenAI semantic evaluation was unavailable ({error}); "
                "the local meaning-aware evaluation was used instead."
            )
            return baseline
    return baseline
